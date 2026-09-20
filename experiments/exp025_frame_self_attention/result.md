# exp025_frame_self_attention 結果

## 仮説

固定画像特徴からの接続学習で、各フレームのcell同士をSelf-Attentionで文脈化すると、現行Cross-Attentionのみより両胚の接続指標が改善するかを調べる。Model AはSelf-Attentionから直接pair scoreを出し、Model Bはその後に時刻間Cross-Attentionを行う。

## 記録の参照先

- 設定、系譜、再現性方針: [`config.yaml`](config.yaml)
- 実験status、胚別の数値、kernel情報、Notebook実行時間、SHA: [`metrics.json`](metrics.json)
- 実行中の時系列: [`SESSION_NOTES.md`](SESSION_NOTES.md)

## 実行証拠

[Model AのKaggle学習](https://www.kaggle.com/code/kentookumura/exp025-frame-self-attention-a-train) v1、[Model BのKaggle学習](https://www.kaggle.com/code/kentookumura/exp025-frame-self-attention-b-train) v4、[恒等初期化したModel BのKaggle学習](https://www.kaggle.com/code/kentookumura/exp025-frame-self-attention-b-identity-train) v1は、それぞれ両fold・各3エポックを完走した。6個のcheckpoint、各model source、manifestのSHAは回収した出力と一致した。cache summary・identity、評価window数、教師監査値は全variantと保存済みexp016で一致した。Model Bのv1–v3は学習開始前にKaggle入力探索またはNotebook同期で停止しており、比較用の重みを生成していない。公式graph評価とKaggle submissionは実行していない。

## 隣接2-frameの評価

同じ外側胚評価windowで、source軸softmax後の確率を0.5で判定したpair指標を比較する。数値の正本は `metrics.json` の `train_stage.pair_level_comparison`。

| 評価胚 | 構成 | 既知edge recall | positive pair precision | false-positive pair数 | 分裂親の回収 |
| --- | --- | ---: | ---: | ---: | ---: |
| 6bba | 現行（exp016保存済み） | 96.94% | 95.84% | 4,355 | 48/108 |
| 6bba | Model A | 94.87% | 93.17% | 7,192 | 45/108 |
| 6bba | Model B | 96.75% | 95.08% | 5,176 | 56/108 |
| 6bba | Model B・恒等初期化 | 96.65% | 95.75% | 4,431 | 50/108 |
| 44b6 | 現行（exp016保存済み） | 94.78% | 93.05% | 1,341 | 6/22 |
| 44b6 | Model A | 91.69% | 90.47% | 1,830 | 5/22 |
| 44b6 | Model B | 95.21% | 92.35% | 1,495 | 4/22 |
| 44b6 | Model B・恒等初期化 | 93.84% | 93.86% | 1,164 | 4/22 |

## 解釈

Model Aは現行より両胚で既知edge recallとpositive pair precisionが低く、false-positive pairが多かった。Model BはAより両胚で既知edge recallとprecisionが高い。一方、保存済み現行との比較では、Model Bのpositive pair precisionが両胚で下がり、false-positive pairが6bbaで821件、44b6で154件増えた。既知edge recallは6bbaでわずかに低く44b6で高い。分裂親の回収は6bbaで増え、44b6で減った。したがって、同じ2層共有Self-Attentionを追加した設定が現行より一貫して良いとは、この処理単位の結果からは言えない。

precisionとfalse-positive pair数は、保存されたactive pair数、edge accuracy、positive edge recall、既知edge数から再構成した。正例のない領域は部分注釈で、active pairにも未知endpointが多い。ここでのfalse-positiveは注釈済み教師との不一致を表し、真の誤接続件数や復号後のgraph品質と同一視しない。公開画像encoderと初期trackerの学習来歴にtrain動画が含まれるため、この外側胚比較は固定公開モデル下の条件付き比較であり独立したCVではない。ILP、graph repair、長い動画での誤り伝播はこの評価では測れず、**公式graph scoreは未計測**である。

## ランダム初期化Model A/Bの結果からの考察（2026-09-20）

恒等初期化を試す前のModel A/Bで強く示唆されたのは、追加Encoderによる初期性能の低下と、その後の再適応である。以下の学習開始前・epoch別数値とparameter数の正本を `metrics.json` の `train_stage.self_attention_diagnostics` に記録した。これは保存済み出力とsourceの分析であり、追加学習による原因の切り分けではない。

### 学習済みtrackerへの挿入時に、初期性能が大きく落ちている

Model Bは公開初期trackerの共通重みを読み込み、追加したSelf Encoderだけをランダム初期化する。Encoder内に残差接続はあるが、Attentionとfeed-forwardの出力をゼロに初期化する処理や、追加分をゼロから増やすgateはない。したがって、追加直後の出力は元の入力と一致せず、既存Cross-AttentionとPair MLPが受け取る特徴も変わる。

| 評価胚 | 公開初期trackerの既知edge recall | Model B挿入直後 | Model Bの3epoch後 | exp016の3epoch後 |
| --- | ---: | ---: | ---: | ---: |
| 6bba | 96.76% | 66.28% | 96.75% | 96.94% |
| 44b6 | 95.45% | 50.36% | 95.21% | 94.78% |

この落差は実測である。「新しいcontextを利用する学習に加え、変わった表現へ既存重みを再適応させる必要があった」はその解釈で、寄与の分離は未実施。Model AはさらにCross-Attentionの経路を外し、そこで処理された特徴を前提に学習済みのPair MLPを引き継ぐため、初期の特徴の不一致がある。単純にSelf-Attentionの表現力が低いと結論しない。

### 3epochで収束した証拠はなく、単純な延長が効く証拠も十分ではない

Model Bの内部検証lossはfold 0で0.000229→0.000164→0.000152と低下し、train lossも低下した。両foldとも最終の3epoch目が選ばれ、同じ3epochでも既存重みの微調整と追加層の新規学習で条件が異なる。Model Bは約58万から約85万parameterへ増え、すべて同じ学習率0.0001で更新している。

一方、fold 1の内部検証lossは0.000126→0.000135→0.000127で単調改善していない。学習不足は有力な仮説だが、長く学習すれば現行を上回るとは言えない。隣接windowは同じ動画を共有し、各foldの学習側は1胚のみである。window数の多さを、独立した多様な学習事例の多さと同一視できない。

### 既存Cross-Attentionにも同一フレーム内の情報が間接的に伝わる

現行の逐次更新では、更新したt側はt+1側の全cellを参照し、そのt側を使ってt+1側を更新する。次のblockではさらに戻った情報を参照できる。このため現行4blockにも同側の別cellから情報が伝わる経路があり、Self-Attentionの追加がまったく未使用だった情報を初めて与えるとは限らない。ただし直接のSelf-Attentionと同じ計算ではなく、追加の有効性は残る。

追加Encoderは画像32次元＋位置32次元の特徴を使う全cellへのAttentionで、同一フレーム内の距離に応じたattention biasや近傍制限はない。時刻間の相対座標は従来どおりPair MLPに渡される。近傍の配置が重要な対応問題に対して、遠方cellの混合が個々のcellの識別を弱める可能性はあるが、Attention分布や特徴の類似度を測っておらず、実際に起きたとは断定しない。

### checkpoint選択と教師に、目的とのずれが残る

選択は `edge_accuracy * candidate_node_recall`。固定候補なので後者はepoch間で一定であり、実質は同じmask上のfalse-positiveとfalse-negativeの合計最小化である。分裂親の回収や公式graph指標は含まれない。Model Bのfold 0では内部検証のfalse-positiveが172→138件へ減る一方、false-negativeは83→94件へ増え、分裂親回収は1/3→0/3に下がった3epoch目が選ばれた。fold 1でも分裂親回収は6/9から3/9へ下がる。この少数事例だけで学習全体を判定できないが、選択指標の改善が分裂の改善を保証しない実例である。負例が多くaccuracyが高いだけで選択が無効とは言わない。

loss対象pairのうち未知endpointを含む割合は6bbaで97.20%、44b6で99.28%。未知対応を負例に含めうる既存教師を、表現力を増やしたモデルも学習する。これは教師の制約が改善を妨げる可能性を示すが、この割合はラベル誤り率ではなく、未知pairのすべてが正例でもない。現行も同じ教師なので、これだけでA/Bとの差を説明したことにはならない。source軸softmaxは娘ごとに親を競合させ、同一親から複数娘への接続は表現できるため、softmaxが分裂を禁止することが原因ではない。

### NFLとの対応と、今回否定できる範囲

添付NFL Notebookの `TemporalTransformerBranch` は `[B*N,T,D]` に変形した選手ごとの時間列を処理し、時間位置埋め込みを使う。2branchを結合した後には選手間の処理もある。exp025で承認したのは、branchを独立に文脈化する考え方をフレーム内cell集合へ適用する変更であり、この追加層は新たなcellの移動履歴や速度を入力に加えていない。既存の固定画像特徴が持つ情報を前提とする。NFLで使われる構成を参考にできても、その性能改善がこの設定へ直接引き継がれる根拠にはならない。共有Encoderも、同じ意味のt/t+1特徴を扱う今回の条件では自然で、重み共有が主因とする証拠はない。

現時点で言えるのは、共有2層Self-Attentionの通常のランダム初期化・固定loss・3epochという設定で、現行に対する一貫したpair改善を確認できなかったこと。Self-Attention全体の無効性や、公式graph scoreの悪化は示していない。単一seed・2胚の結果から小さな差の再現性も判断できない。

### 原因を切り分ける次の確認

最優先はModel Bの追加層を初期状態で恒等写像として働かせ、同じ初期trackerのlogitsを維持できるかを確かめること。例えば追加残差の寄与をゼロから学習するgate、または各残差branchの出力をゼロに初期化する方式を、次の比較候補として検討する。これが構造追加直後の性能低下をなくす直接の確認になる。その条件で内部検証の学習曲線と既知edgeの順位・precision/recall・分裂を追い、必要なら学習量を検討する。最初からepoch、loss、近傍制限をまとめて変更すると原因を分離できない。外側胚の結果をepoch・閾値の選択に使わず、選択方法の変更は新しい契約として扱う。その後、ユーザーの承認を受けて恒等初期化だけを変更した比較を実施した。結果は次節に記録する。

## Model Bの恒等初期化による切り分け（2026-09-20）

学習済みtrackerを引き継ぎ、追加する共有Self-Attentionの残差出力をゼロに初期化した。実データの初期出力は、6bba評価側の2 window・14,450 pair、44b6評価側の2 window・100,800 pairで、公開初期trackerとのlogits最大差がいずれも0.0だった。追加直後に特徴が変わる問題は、この構成では解消できた。2foldとも3エポック完走し、内部検証で選ばれたcheckpointは6bbaが3エポック目、44b6が1エポック目。Notebook実行時間は4,845.19秒、学習stageは3,985.06秒。manifest、model source、checkpoint・stateのSHAは `metrics.json` に記録し、取得した実ファイルと照合した。

学習後は現行比で、6bbaの既知edge recallが96.94%→96.65%、positive pair precisionが95.84%→95.75%、教師上のfalse-positive pairが4,355→4,431、分裂親回収が48/108→50/108。44b6ではrecallが94.78%→93.84%へ下がる一方、precisionが93.05%→93.86%、false-positive pairが1,341→1,164へ改善し、分裂親回収は6/22→4/22だった。ランダム初期化したModel Bに対しては両胚でprecisionとfalse-positive pair数が改善し、recallは両胚で低下した。恒等初期化は初期性能の破壊を防ぎ、予測のprecision/recallの釣り合いも変えたが、同じ3エポック・選択指標で現行を一貫して上回る証拠にはならなかった。原因を初期化だけに帰すこともできない。

4構成は同一の外側window・候補・教師に対する隣接2-frameの条件付き比較である。教師は部分注釈なのでfalse-positive pairは真の誤接続数ではない。分裂親は特に44b6で22件と少ない。外側胚の数値でcheckpointを再選択しておらず、公式graph scoreは未計測。既知edge recallの低下と分裂指標の混在があるため、現時点では全動画graph推論へ自動で進めない。

## 恒等初期化後にも残る要因（2026-09-20）

初期logitsの保持だけでは、学習後に現行を一貫して上回れなかったことを説明できない。保存checkpointの追加Self-Attentionを再確認すると、両fold・両層とも、ゼロ初期化したAttentionとfeed-forwardの出力射影は非ゼロへ更新されていた。構造化された確認値は `metrics.json` の `train_stage.post_identity_diagnostics`。これは重みが更新された証拠であり、有用な文脈を学んだ証拠ではない。実データ上の残差出力の大きさやAttention分布は未計測。

1. **checkpointの選択基準と欲しい改善のずれ。** 固定候補では `edge_accuracy * candidate_node_recall` は実質的に教師上のfalse-positiveとfalse-negativeの合計で選ぶ。44b6を外側評価するfoldでは、内部検証の1エポック目がFP 195・FN 208、2エポック目がFP 240・FN 174。合計403対414のため1エポック目が選ばれ、既知edge recallと分裂親回収が良い2エポック目（3/9→5/9）は選ばれない。これは選択の設計上の釣り合いであり、2エポック目なら外側胚や公式graphが改善したとは未確認。外側胚の結果を用いて再選択しない。
2. **学習済みtracker全体の再学習による変化。** 引き継いだCross-Attention・Pair MLPも、新規Self Encoderと同じ学習率0.0001で更新する。恒等初期化が保証するのは学習開始時の出力であり、その後の既存機能の保持ではない。追加層の改善と既存重みの変化を分離できていない。内部検証が良くても外側胚への汎化が下がる可能性がある。既存層固定や層ごとの学習率の効果は未検証。
3. **学習する胚が1種類で、候補集合の分布も異なる。** 各foldは1胚で学習し、別胚で外側評価する。隣接windowは同じ動画・frameを共有し、多数のwindowを独立した多様な事例数と同一視できない。評価windowで重複frameを含めて平均した1-frame当たり候補数は6bbaで193.49、44b6で335.51。真のcell密度ではないが、全cellへのAttentionが扱う集合規模の違いは実測である。胚固有の配置・候補数に適応した可能性はあるが、両胚とも同一条件の現行対照を持つため、分布差だけで追加層の差を説明したことにはならない。44b6評価foldは1エポック目が内部検証で最良で、単純な学習延長を裏付ける結果ではない。
4. **同一フレームの文脈と局所的な対応に必要な情報の違い。** 入力は固定の画像32次元・位置32次元で、Self-Attentionに同一フレーム内の相対距離biasや近傍制限はない。位置情報は存在するが、近傍の配置を重視する計算は明示していない。遠方cellの混合が識別に役立たない可能性や、既存4blockのCross-Attentionが既に間接的に取り込む情報との重複が残る。長い移動履歴・速度をこの追加層が新規入力として使うこともない。特徴の均一化や文脈の冗長性を示す実測はなく、仮説として扱う。
5. **部分注釈から作った教師の限界。** active pairの未知endpoint比率は6bbaで97.20%、44b6で99.28%。教師対応は距離順の1対1割当、距離上限5µmで、未知候補も既知正例と同じ行または列にあれば負例として扱いうる。これはラベル誤り率を表さず、未知pairがすべて誤ラベルでもない。ただし表現力を増やしても、欲しい文脈や分裂を十分に教えられていない可能性は残る。候補生成を固定しているので、未検出cellの回収はこの追加層だけでは増やせないが、それは今回の既知edge recall低下の直接説明ではない。
6. **固定閾値では予測の順位と確信度を分離できない。** pair評価はsource軸softmax後の確率0.5で行う。恒等初期化Bはランダム初期化Bより両胚でprecisionが上がりrecallが下がった。この集約結果だけでは、正しいpairの順位が悪くなったのか、確率の分布が変わったのか不明。precision-recall曲線、同じprecisionでのrecall、正解親の順位・誤りの種類の確認が必要である。閾値を変えるだけで現行を上回るという証拠はない。

次に原因を切り分けるなら、同じ保存checkpointについて、内部検証でのprecision-recall曲線と候補数・分裂別の誤りを確認し、追加層と既存層の実データでの出力変化を測ることが有用。現時点でどれか1つを主因と断定せず、学習量だけを増やせば改善するともしない。この分析では新しい学習・graph推論・閾値選択・実験の採否変更を行っていない。

## ユーザー判断

- 判断: 未判断
- 確認日時 / 依頼メッセージ: 結果提示後に記録する
- 理由: 公式graph評価とModel Bの採否判断は未実施

## 次

全199動画のgraph推論は自動で開始せず、恒等初期化で初期logitsは維持できたが、学習後のpair指標が混在する結果を踏まえてユーザーの判断を待つ。現行controlの再学習とKaggle submissionは別の明示承認が必要。
