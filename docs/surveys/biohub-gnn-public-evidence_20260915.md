---
title: BiohubでのGNNの有用性と公開投稿の確認
date: '2026-09-15'
types:
- survey
hypotheses: []
experiments:
- exp015
- exp016
topics:
- tracking
- gnn
- public-notebooks
status: final
summary: GNNは固定検出候補の接続学習に適合するが、コンペでの優位性は未実証。公開Discussion、GNN表記の実演コード、Transformerによる追跡実装を区別して確認した。
---

# BiohubでのGNNの有用性と公開投稿の確認

作成日: 2026-09-15

## 結論

**GNN（Graph Neural Network、グラフニューラルネットワーク）は試す価値がある。ただし、このコンペで現行のNode Transformerを上回る公開実証は今回確認できなかった。** 現在の「公開検出器・画像特徴抽出器を固定してトラッカーを学習する」方針に沿って、周囲の細胞の配置・動きから接続得点を改善する用途が適している。

検討価値が高い比較は次の3つ。いずれも転用仮説であり、実験の実装承認・採否判断ではない。

1. **固定特徴から接続を予測するGNNと現行Transformerの比較。** 検出点、画像特徴、時間窓、教師、損失関数、接続を最終選択する処理を揃え、候補同士の情報を混ぜるモデルを変える。
2. **周囲の細胞の相対配置・移動の追加。** 同一時点の近傍細胞と前後時点の候補を使い、近接する別細胞への取り違えが減るか調べる。情報を伝えるための辺と、最終的に出力する親子の辺は役割を区別する。画像モデルの入力窓は固定し、多時点化は別比較にする。
3. **母と2娘の組の得点予測。** 分裂前後の候補を同時に評価する。ただし、現行候補辺だけでは分裂の上限が低いため、組の候補生成・疎い教師・通常継続との競合を先に定義する必要がある。

## 調査目的

Biohub - Cell Tracking During DevelopmentにおけるGNNの適性と、公開Notebook・Discussionにおける提案、実装、成績の証拠を確認する。

## 対象と証拠範囲

- 対応する上位仮説: なし
- 対象実験: exp015の保存済み診断とexp016の固定画像特徴を使う学習契約。既存上位仮説の支持・棄却を判定する調査ではない。
- 調査日: 2026-09-15。Kaggle CLIで取得した現在公開されているソースを静的に確認し、Notebookを実行してはいない。
- 検索範囲: コンペ指定のvote順Notebook一覧100件、同コンペ内の`GNN`検索1件と`graph`検索100件、new順Discussionの1・2ページ、関連Web検索。Discussion 738778は本文のWeb検索結果とCLIの返信2件を確認した。全Notebook本文・全Discussionを網羅した検索ではない。
- 入力データ・生成物: exp015の`metrics.json`、取得した公開Notebook4件、MAGIK・GNN-DOLの一次論文、Trackastra公式実装。
- 評価方法: 公開コードの入出力と実際の提出処理への接続を確認し、著者のスコア表記をモデル単独の効果と混同しない。
- このレポートからは判断できないこと: GNNの本コンペでの改善値、学習所要時間、全公開・非公開実装の有無、公開Notebookの現在の公式スコア。

## 公開DiscussionとNotebook

### GNNを直接提案したDiscussion

[I think we should use GNN for higher scores](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/738778)は、Zenpai1977による2026-09-01の投稿。取得時点で返信は2件だった。

- hengck23は、モデルより先にデータを考えるべきだと指摘。
- Shrey Gandhiは、正解が4,435個の分離した木からなり、分岐を持つのは最大151個なので、少なくとも96.6%は直線的な系列だと指摘。
- 投稿と確認できた返信には、GNNの学習結果、対照実験、改善スコアはない。

後者は**疎い注釈グラフの構造についての指摘**であり、実際の胚で分裂がほとんどないことを意味しない。また、GNNの入力には誤った対応候補や近隣細胞も含められるため、正解の多くが直線的な軌跡でも接続判定への有用性は残る。これは本調査の解釈である。

### ソースを取得して確認したNotebook

| 公開Notebook | 確認した処理 | GNNの証拠としての扱い |
| --- | --- | --- |
| [🔬 [0.146 PB] BioHub 4D Cell Tracking](https://www.kaggle.com/code/beraterolelk/0-146-pb-biohub-4d-cell-tracking) / beraterolelk | `CellLineageGNN`は両端のnode特徴とedge特徴を個別に変換し、連結して3分類するMLP（多層パーセプトロン）。乱数10組で形状を確認する。実際のCSV生成は閾値処理・連結成分・速度外挿・ハンガリアン法。 | 近隣nodeから特徴を集約するmessage passingも学習ループもなく、GNNと名付けたモデルの出力は提出処理へ渡されない。タイトルの0.146をGNNの成績として引用できない。 |
| [Biohub Cell Tracking: Learned Graph w Gap Recovery](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-learned-graph-w-gap-recovery) / Pilkwang Kim | `TemporalUNet3D`、node間のcross-attention Transformer、ILP（整数線形計画法）、軌跡修復。公開ソースでは`predict_unet_transformer.py`を呼ぶ。 | graphを学習・選択する実装はあるが、ここで検討する局所近傍でmessage passingを行うGNNとの比較結果ではない。 |
| [Biohub🧫CellTrack: DoG & Trackastra Graph Trans](https://www.kaggle.com/code/jirkaborovec/biohub-celltrack-dog-trackastra-graph-trans) / Jirka | DoG（異なる幅のGaussian平滑化の差）で点を検出し、watershedによる領域を作り、学習済みTrackastraへ画像と領域を渡す。取得版は`MASK_MODE="watershed"`、`TRACK_MODE="greedy"`。 | Transformerを用いたグラフの接続予測の公開例。点と保存済み特徴だけを入力する現行実験とは前提が異なる。 |
| [Graph Patches for Cell Tracking: 0.917 LB](https://www.kaggle.com/code/reyhanksatria/graph-patches-for-cell-tracking-0-917-lb) / Reyhan Ksatria | 3D U-Net、ハンガリアン法、距離・類似度を使った再接続や欠落補完。 | グラフ修復の公開例。確認したソースにGNNのmessage passingによる接続学習はない。タイトルのスコアを現在の公式値として再検証していない。 |

Trackastra自身も[公式リポジトリ](https://github.com/weigertlab/trackastra)でTransformerによるcell associationと説明している。領域分割済み画像を入力するため、このNotebookの領域作成処理を省略して点だけを渡す設計にはできない。Notebookの説明にある大域的な割当と、取得版の実設定`greedy`も区別した。

## 分析結果

### 手元の診断が示す改善余地

数値の正本は[exp015 metrics](../../experiments/exp015_oracle_stage_limits/metrics.json)の`oracle_stage_limits.overall`。199動画の既知注釈に対する候補対応・残存率であり、公式Jaccard scoreではない。

| 診断 | 結果 |
| --- | ---: |
| 既知中心が検出候補に対応 | 99.286% |
| 既知接続の両端が検出候補に対応 | 98.843% |
| 既知接続が候補グラフに存在 | 94.829% |
| 既知接続が最終グラフに残存 | 91.602% |
| 既知分裂の母・2娘と両辺が候補に存在 | 61 / 151 |
| 既知分裂が最終グラフに残存 | 15 / 151 |

候補から最終選択へ進む際の接続損失は約3.227 percentage points。接続の得点予測には改善余地がある一方、これがすべて学習モデルの誤りとは限らず、得点尺度・ILP・後処理の影響も含む。GNNによる改善量とは解釈できない。

固定した候補辺を採点し直すだけなら、候補から消えた正解辺を回収できない。分裂も現行候補に限定すると61件の上限がある。検出点自体を固定したまま追加の辺を生成する比較は可能だが、候補集合を変えた効果として分けて測る。

### 現行Transformerとの差

[採用公開Notebookの構造説明](biohub-cell-tracking-0946-notebook-explanation_20260913.md#6-simplenodetransformerによるassociation)では、候補点の画像特徴・位置情報に対して、前後frame間で双方向cross-attentionを4 blocks適用し、各pairの得点を出している。すでに候補同士の文脈を使うモデルである。

したがってGNNの価値は、モデル名よりも、物理的な近傍を明示すること、同一frame内の相対配置を使うこと、複数の対応候補から情報を集約することにある。局所化で計算量を減らせる可能性はあるが、近傍探索と集約にも費用がかかり、遠くへ移動した正解候補を除外する危険もある。速度向上は実測が必要。

### 研究上の根拠

- [MAGIK: Geometric deep learning reveals the spatiotemporal features of microscopic motion](https://www.nature.com/articles/s42256-022-00595-0), Pineda et al., 2023。検出をnode、時空間的に近い候補間をedgeとし、attentionを組み込んだGNNで接続を分類する。Cell Tracking Challengeの細胞領域由来特徴を使った評価と[著者の実装](https://github.com/DeepTrackAI/MAGIK_CellLinkingBenchmark)がある。固定検出点から接続を学ぶ構成の根拠になるが、Biohubの疎い教師と公式評価での成績は示していない。領域の形態特徴を固定画像特徴へ置き換えるのは転用案であり、論文そのままの再現ではない。実装難度は中程度。
- [Differentiable optimization layers enhance GNN-based mitosis detection](https://www.nature.com/articles/s41598-023-41562-y), Zhang et al., 2023。GNNに、次frameへ対応できる細胞数が最大2個という制約を組み込む微分可能な最適化層を加え、4培養条件で分裂検出の改善を報告する。分裂に制約を組み合わせる根拠になる。ただし現行実装にもILPがあり、新たな最適化層が追加利益をもたらすとは限らない。追加層の学習・オフライン依存・計算費用も必要になり、実装難度は高い。

## 解釈

- 支持されたこと: GNNによる細胞の接続予測には研究上の実績があり、固定検出器の下流を学習する現在の方針に適合する。コンペ内にも直接の提案と関連するTransformer実装がある。
- 否定されたこと: GNNを説明した公開Notebookがあることや、タイトルにスコアがあることだけでは、GNNがそのスコアを達成した証拠にならない。
- 未解決: 現行Transformerに対する精度・費用の優位性、近隣情報の追加による効果、分裂教師の不足と候補生成の制約。
- 教師と検証: 未注釈を一律に負例とせず、予測候補と既知注釈の対応・教師maskを明示する。最初のモデル比較は対照と同じ教師・損失で行い、教師maskや損失の改善は別比較とする。両胚で公式指標と成分を評価し、設定選択を学習側内部に限定する。公開画像重みの学習来歴による制限は残るため、トラッカーを胚で分割しても独立した交差検証とは断定しない。
- 計算費用: Kaggleの週30 GPU時間以内に初回入力準備・学習・検証・提出推論を含める。既存cacheを再利用できる範囲から始め、GNNの追加packageを使う場合はオフラインwheelの互換性も確認する。未実測のruntimeは保証しない。

## 関連ファイル

- 現在の方針: [KAGGLE_DIRECTION.md](../../backlog/KAGGLE_DIRECTION.md#今後の学習方針)
- 診断: [exp015 result](../../experiments/exp015_oracle_stage_limits/result.md)、[metrics](../../experiments/exp015_oracle_stage_limits/metrics.json)
- 対照の契約: [exp016 requirements](../../experiments/exp016_frozen_image_encoder/requirements.md)
- 調査コード: [inspect_notebooks.py](../../studies/gnn_public_20260915/inspect_notebooks.py)。CLIで`/tmp/biohub-gnn-review/`へ取得したNotebookを読み、セルを実行せずsource・出力を調べる。
- 検索時のNotebook一覧: [GNN検索](../../studies/gnn_public_20260915/gnn_search.csv)、[vote順100件](../../studies/gnn_public_20260915/notebooks_top100.csv)

取得ソースのSHA-256は次のとおり。Kaggleの公開ソースは更新されるため、後日の同じURLが同じ内容とは限らない。

| Notebook owner / slug | SHA-256 |
| --- | --- |
| `beraterolelk/0-146-pb-biohub-4d-cell-tracking` | `74e44b7438f8aeca3f6c4c6e75e22ca44684b67909716162864de0830295a80c` |
| `pilkwang/biohub-cell-tracking-learned-graph-w-gap-recovery` | `5d616f969bed8800076b9213f39c4f89791fc9855a0c65efc626bf0b740b4c42` |
| `jirkaborovec/biohub-celltrack-dog-trackastra-graph-trans` | `0cb707977433e2932861b67dc28be780077f749beadf93fb35792a2b7b614461` |
| `reyhanksatria/graph-patches-for-cell-tracking-0-917-lb` | `9fc0ebe7eb2fefb4253cd74e7ce4b03dab64470a6328c3cee76a385642f0bb02` |

## 次のアクション

exp016の対照結果を基準に、同じ固定候補・画像特徴・教師・復号でGNNを比較する案を推奨する。分裂特化や多時点化を同時に変更せず、何が効いたか判断できる比較にする。今回は調査と提案までであり、実験化、バックログの更新、Kaggle実行、submissionは行っていない。
