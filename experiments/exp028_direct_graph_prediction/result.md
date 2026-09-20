# exp028_direct_graph_prediction 結果

## 仮説

同じ固定検出候補から、主催者の既知接続で学習したTransformerが娘ごとの母または対応なしを直接選ぶと、保存済みexp016の接続選択より両胚の追跡を改善できるか。

## 実行証拠

- private Kaggle train Notebook [version 1](https://www.kaggle.com/code/kentookumura/exp028-direct-graph-prediction-train)はCOMPLETE。2-fold学習と、各外側胚の隣接2-frame評価を実行した。Notebook実行4995.77秒、GPU peak 3611434496 bytes。
- [metrics.json](metrics.json)に胚別の数値、モデル・gateのSHA、runtimeを記録した。[早期gate](artifacts/kaggle_train_v1/early_gate.json)と[model manifest](artifacts/kaggle_train_v1/model_manifest.json)のSHA、および2つのモデルファイルのSHAを照合した。詳しい時系列は[SESSION_NOTES.md](SESSION_NOTES.md)。
- 保存済みexp016のfold別primary trackerを再学習せず読み込み、同じcache・教師・2-frame単位でILPとgraph repair前の接続を比較した。対照の最終graphについては[exp016の結果](../exp016_frozen_image_encoder/result.md)を参照。

| 評価胚 | 既知edge recall 直接 / 対照 | 観測可能な誤接続率 直接 / 対照 | 既知分裂母の回収 直接 / 対照 | gate |
| --- | ---: | ---: | ---: | --- |
| 6bba | 95.35% / 97.20% | 5.93% / 2.94% | 45/108 / 50/108 | 不成立 |
| 44b6 | 94.37% / 95.30% | 7.21% / 3.61% | 12/22 / 6/22 | 不成立 |

## 解釈

両胚とも構造違反は0で、同じ対象について対照と比較した。6bbaは既知edge recallと分裂回収が低下し、両胚で観測可能な誤接続率が上昇した。44b6の既知分裂母の回収は増えたが、22件だけの部分注釈に基づく値で、誤接続率の悪化を打ち消すものではない。

主催者の注釈は疎く、未知の娘に選ばれた接続の真偽はこの評価では確定できない。観測可能な誤接続率は全graphの誤接続率ではない。公開checkpointの学習来歴を引き継ぐため独立CVとは呼ばない。

事前に定めた両胚のgateを満たさなかったため全graph inferenceを開始していない。公式scoreは未計測。2-frame指標を公式scoreの代用としない。

## ユーザー判断

採否・完了は未判断。現設計の全graph実行は保留を推奨する。実験をこの診断結果で完了・不採用とするかはユーザーが判断する。

## 次

ユーザーの判断を受け、必要なら誤接続と対応なし閾値の原因分析を別の実験候補として設計する。外側胚で閾値を選び直してこの結果を救済しない。
