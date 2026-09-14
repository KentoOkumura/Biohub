# exp015_oracle_stage_limits 結果

## 仮説

固定公開モデルの候補を既知中心、既知辺、母と2娘、最終graph選択に分けると、固定検出器のまま回復可能な誤りと候補生成から不足する誤りを区別できる。

## 実行証拠

Kaggle上でGPU inference version 1とCPU diagnostic version 6が完走した。大容量のinference outputはローカルへdownloadせず、diagnosticからKaggle kernel sourceとして直接参照した。

- inference: [exp015 oracle stage limits inference](https://www.kaggle.com/code/kentookumura/exp015-oracle-stage-limits-inference)、version 1、id 134210702、T4、internet無効、status COMPLETE。
- GPU予測処理: 23,994.20秒、約6時間39分54秒。これはNotebook全体ではなく予測区間の実測値である。
- inference生成物: sample、ILP前candidate graph、最終graphが各199件。GTは参照していない。
- train window cache: 199 dataset、19,701 window、4,151,337,848 bytes、612,395,776 feature values。各NPZのarray round-tripは完全一致し、現在の予測はin-memory featureを使用し、cache用のedge logits再計算はない。
- Kaggle working output: manifest作成前4,813,320,625 bytes。20GB上限と18GB soft guardを下回った。
- cache identity SHA-256: 440891c4550adf8540e3c47f9784e68b6ca1b64bbe56dbb93a25eb87c46bc1ee。
- cache summary SHA-256: 040d1f6437e27149e0e34ad4aac8bba3c8cf63dc266d33df497acd969972194c。
- inference manifest file SHA-256: 0cae5963848c2797d04b526f9daa48db00e3bcc3b2db9a0c6e4ed4ee6cee35b2。
- diagnostic: [exp015 oracle stage limits diagnostic](https://www.kaggle.com/code/kentookumura/exp015-oracle-stage-limits-diagnostic)、version 6、id 134271528、CPU、internet無効、status COMPLETE。
- diagnostic: 199/199 sampleが有効、failure 0、実測423.65秒。入力bundle SHA-256は 0d7f66e4ea6ac496dd63bf200a9612c33067908166bc279c52aab09a3d0e9fa1、oracle manifest SHA-256は 997e578b73ee7c8b098dd1049828a779022476c8beb9c4a9367c7f20f28138c5。
- 出力SHAは metrics.json の evidence.artifacts を正とする。Kaggle outputにはper-sample、stage、5条件、brightness x candidate density、degraded 16件、summary、manifestの7ファイルが存在する。
- diagnostic versions 1から4はruntime dependencyのdebug履歴である。version 5が最初の有効結果、version 6が同じ計算にKaggle log用receiptを追加した最終結果である。

## 段階別結果

値は各scope内の既知GTに対するmicro recallである。divisionの分母は全199件151、44b6全71件26、degraded 16件7と小さい。

| 段階 | 全199件 | 44b6全71件 | degraded 16件 |
|---|---:|---:|---:|
| 候補中心に対応 | 99.286% | 99.579% | 99.320% |
| 最終graphに中心が残る | 96.057% | 95.930% | 95.655% |
| edge両端が候補中心に対応 | 98.843% | 99.248% | 98.850% |
| 候補graphに既知edgeが存在 | 94.829% | 94.048% | 93.827% |
| 最終graphに既知edgeが存在 | 91.602% | 90.159% | 89.916% |
| 候補graphに母と2娘が同時に存在 | 40.397% | 30.769% | 57.143% |
| 最終graphに母と2娘が残る | 9.934% | 7.692% | 14.286% |

全199件では、既知中心の候補欠損は952/133,318で0.714%だった。edgeは、両端の候補対応までに1.157 percentage point、候補graphでさらに4.014 points、最終選択でさらに3.227 points失われた。divisionは151件中61件だけが候補graphにあり、最終graphに残ったのは15件だった。候補段階で59.603%を失い、候補にあったdivisionも75.410%が最終選択で失われている。

degraded 16件のcandidate edge recallは44b6全件より0.222 points、final selected edge recallは0.244 points低いだけだった。したがってexp012の悪化16件を、段階別edge coverageの集約差だけで説明する証拠は弱い。一方、7 µm以内に複数候補を持つGT nodeは全199件で16.37%、44b6全件で33.70%、degraded 16件で34.72%だった。44b6では一対一matchingの曖昧さが大きく、sample別・条件別の解釈ではこの注意が必要である。

## 解釈

中心候補の回収率は十分高く、固定公開構成での主要な上限は中心検出そのものよりedge候補生成と最終graph選択にある。特にdivisionは通常edgeより大きく失われており、母と2娘を候補として同時に残す処理と、その候補を最終graphで選ぶ処理を分けて改善する余地がある。

ただし公開modelはtrain 199動画を学習に含むため、これは固定公開model下の条件付き診断であり独立CVではない。train cacheも同じ固定公開modelの特徴であり、独立OOF特徴ではない。未対応予測候補は疎いGEFFのためfalse positiveとして数えていない。条件間には重なりがあり、brightnessまたはcandidate densityの因果効果とは扱わない。

## ユーザー判断

2026-09-14、ユーザーがGitへのcommit・pushを依頼したことを、この実験をdiagnosticとして完了する判断として記録した。metrics.jsonのstatusをcompletedへ更新する。ただしモデル改善やhidden test有効性を確認した実験ではない。
