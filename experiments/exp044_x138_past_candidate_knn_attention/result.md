# exp044_x138_past_candidate_knn_attention 結果

## 仮説

exp043の候補・画像特徴に対し、直前frame候補8点のattentionが既知edgeの接続改善に寄与するかを調べる。primary trackerとattentionを同時学習する。

## 記録の参照先

- 設定、系譜、再現性方針: [`config.yaml`](config.yaml)
- CV/LB、実験status、kernel情報、Kaggle Notebook実行時間、生成物SHA: [`metrics.json`](metrics.json)
- 実行コマンドと途中経過: [`SESSION_NOTES.md`](SESSION_NOTES.md)

設定とSHAをこのファイルへ独立した記録として転記しない。CV・Public LB・Private LBは`metrics.json`を数値の正本とし、比較や乖離の説明に必要な場合は、参照キーを添えて同じ値を本文で使用してよい。

## 実行証拠

- 比較対象: exp044がexp043の入力処理で生成した同じpair capture内の固定x138融合logit。別実験の旧cacheは流用していない。
- [Kaggle train Notebook version 1](https://www.kaggle.com/code/kentookumura/exp044-x138-past-candidate-knn-attention-train) は `COMPLETE`。20動画から1823の隣接frame pairを取得し、反対の胚で評価する2 checkpointを学習した。全graph推論・公式評価・submissionは行っていない。
- `metrics.json` の `evidence.kaggle` にkernel version、T4、internet off、Notebook実行時間の推定値を記録。log末尾の相対時刻による約6534秒で、APIが保証する正確な終了時刻ではない。capture 2195秒、学習3927秒は `evidence.capture` と `evidence.training` を参照。
- `metrics.json` の `evidence.baseline_parity` は6窓すべてで最大絶対差0。`evidence.candidate_retention` は44b6が3275/3275、6bbaが7181/7183で、両胚とも受け入れ閾値99%以上。
- 生成物は [`artifacts/kaggle_v1/`](artifacts/kaggle_v1/) の `pair_metrics.json`、`baseline_parity.json`、`candidate_retention.json`、`train_receipt.json`、fold別progress、2 checkpointとKaggle log。取得したcheckpointのSHA-256はreceiptと一致し、`metrics.json` の `evidence.artifacts.model_shas` に記録。追加診断用に既存pair capture 1980ファイルをローカル取得した（Git管理外）。

| 外側評価の胚 | pair窓 | 既知edge recall: 対照 → attention | active-pair errors: 対照 → attention | 既知分裂親の回収: 対照 → attention |
| --- | ---: | ---: | ---: | ---: |
| 44b6 | 894 | 3162/3302 (95.760%) → 3144/3302 (95.215%) | 383 → 364 | 1/2 → 1/2 |
| 6bba | 929 | 7016/7275 (96.440%) → 6884/7275 (94.625%) | 627 → 658 | 1/11 → 0/11 |

数値の正本は `metrics.json` の `evidence.pair_evaluation`。false edges on active pairsは44b6で243→206、6bbaで368→267だが、部分注釈下では未注釈の真のedgeを誤接続と数える可能性がある。active-pair errorsは既知edgeの欠落も含むため、false edgesだけの減少を総合的な改善と解釈しない。

## 追加CPU診断（2026-09-26）

既存captureのSHAをKaggle receiptと照合し、44b6の894窓と6bbaの929窓、計1823窓で4条件を再計算した。公開対照と再学習primary＋attentionは、両胚とも既存の`pair_metrics.json`と全8項目で一致した。閾値0.05〜0.95（0.01刻み）の既知edge recallは `metrics.json` の `evidence.full_pair_diagnostic`、詳細はローカルの `artifacts/kaggle_v1_diagnostic_inputs/full_pair_diagnostic.json` に記録した。

| 胚 | 公開primary・attentionなし | 公開primary・学習済みattention delta | 再学習primary・attentionなし | 再学習primary・attentionあり |
| --- | ---: | ---: | ---: | ---: |
| 44b6 | 3162 / 383 | 3164 / 381 | 3144 / 363 | 3144 / 364 |
| 6bba | 7016 / 627 | 7015 / 627 | 6886 / 656 | 6884 / 658 |

各セルは「回収した既知edge数 / active-pair errors」。既知edge総数は44b6が3302、6bbaが7275。公開primary＋attentionは、再学習primaryと共同学習したdeltaを公開primaryへ流用した反実仮想であり、独立に再学習したモデルではない。

閾値0.48で公開primaryから再学習primaryへ置き換えると判定が変わったactive pairは44b6で72、6bbaで265。一方、再学習primary上でattentionを追加して変わったのはそれぞれ1、2 pairで、今回の既知edge回収低下は追加attentionの推論時deltaよりprimary再学習時の変化と整合する。ただし共同学習した2つの重みの相互作用まで分離した因果実験ではない。

44b6では再学習primaryの閾値を0.43〜0.45に下げると、公開対照の閾値0.48に対し既知edge回収数とactive-pair errorsを両立できた。6bbaでは調べた91閾値のどれも公開対照の7016回収かつ627 errors以下を同時に満たさない。7016以上を回収する条件で最少でも642 errors（閾値0.40）となる。したがって6bbaの悪化を単一閾値0.48のずれだけでは説明できない。部分注釈のため、未注釈edgeを含む「false edge」は真の誤接続と断定しない。

## 解釈

- exp043の候補・画像特徴と座標補正を使い、公開primary trackerと直前frame近傍8候補へのattentionを同時学習した今回の構成では、既知edge recallが両胚で低下した。44b6のactive-pair errorsは19件減ったが、6bbaでは31件増えた。受け入れ条件である「両胚とも既知edge recallとactive-pair error rateが悪化しない」を満たさない。
- したがって、現時点では全graph推論へ進めないことを推奨する。公式scoreは未計測で、pair単体指標からILP・後処理込みの最終スコアは断定できない。公開画像モデルの学習由来が不透明で、GEFFは部分注釈なので、この外側胚評価は独立したCVでもない。
- この結果が否定するのは、今回の入力・K=8選択・損失・同時学習・2-frame復号を組み合わせた実装の改善仮説である。過去候補を参照するattention全般や、別の候補選択・情報源まで不採用と判断する証拠ではない。

## ユーザー判断

- 判断: 未判断
- 確認日時 / 依頼メッセージ: 2026-09-25に結果を提示して確認する。
- 理由: 事前のgraph進行条件は不通過。今回の実験をここで終了するか、条件未達でも全graph評価を追加で依頼するか、別設計へ進むかはユーザー判断。
- 追加診断後の選択肢: (1) 全graph評価へ進まず、今回の実験の完了・採否を判断する（推奨）、(2) primaryの学習方法や教師の扱いを変える別実験を設計する、(3) pairの進行条件未達を承知で全graph評価を明示的に依頼する。いずれもユーザー判断前に実行・確定しない。

## 次

今回のexp044は全graph推論を保留し、ユーザーに実験の終了・採否判断を求める。別の候補選択または情報源を試す場合は、今回と異なる実験契約を先に決める。submissionは未承認。
