# 発想評価用の入力

情報時点: 2026-09-27 17:41:06 JST / Git 7b648a3。
目的は当時の情報から反証可能な改善案を作ること。学習・提出は行わない。
現在のskillを当時の情報へ適用する評価であり、当時のモデルやskillの再現ではない。
事実として不明な入力・教師・費用は不明のまま扱う。


## 入力資料: docs/01_competition.md

# コンペ概要

このファイルは、コンペ全体の運用メモです。機械可読なコンペ設定の正は `project.yml` とし、ここには公式情報の解釈、制約、未解決事項を残します。公式評価ページからの抜粋と出典は `docs/official/evaluation.md` に残します。

## 基本情報

- コンペ名: Biohub - Cell Tracking During Development
- URL: [Kaggle公式ページ](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development)
- 主催: Biohub SF
- 目的: 3次元の経時蛍光顕微鏡画像から細胞を検出し、時点間の対応と細胞分裂を推定して細胞系譜を再構成する。
- 予測対象: 各データセットの細胞検出を表すnodeと、同一細胞または分裂後の娘細胞への時間方向の接続を表すedge。
- 提出形式: Kaggle Notebookが生成する`submission.csv`。各行はnodeまたはedgeで、列は`id,dataset,row_type,node_id,t,z,y,x,source_id,target_id`。
- 開始: 2026-06-29 00:00 UTC
- 参加・チーム統合期限: 2026-09-22 23:59 UTC
- 最終提出期限: 2026-09-29 23:59 UTC

## 重要ポイント

- Notebook-onlyのCode Competition。提出時にKaggleが公開例とは異なるhidden testへ入力を差し替えてNotebookを再実行する。
- trainとtestは胚（embryo）単位で分離される。サンプル名の最初の区切りまでが胚IDで、同じ胚に属する複数サンプルがあり得る。
- 画像はZarr v3、学習用の疎な正解tracking graphはGEFFで提供される。
- 評価は`adjusted_edge_jaccard + 0.1 * division_jaccard`を最大化する。疎な正解annotationを考慮するため、スコアは1.0を超える場合がある。

## 制約

- インターネット: 無効。
- GPU/CPU: どちらも利用可能。`project.yml`の共通既定値はCPUで、GPUが必要な実験だけ`config.yaml`で上書きする。
- 実行時間: CPU/GPUとも12時間以内。
- 外部データ: 無償で公開され、全参加者が合理的にアクセスできるデータとpre-trained modelを利用できる。
- チーム/マージ規則: 最大5人。チーム統合期限は2026-09-22 23:59 UTC。
- 提出回数: 1日5回まで。最終審査用に最大2件を選択できる。
- winner license: MIT。Competition Dataの利用条件はCC0。

## 提出

- ファイル: `submission.csv`
- ID 列: `id`。0から始まる連続整数のthrowaway index。
- 必須列: `id,dataset,row_type,node_id,t,z,y,x,source_id,target_id`
- node行: `row_type=node`とし、`node_id,t,z,y,x`を設定する。`source_id,target_id`は`-1`。
- edge行: `row_type=edge`とし、接続する`source_id,target_id`を設定する。`node_id,t,z,y,x`は`-1`。
- 必要行数: 固定ではない。hidden test内の全データセットを含め、予測したnodeとedgeを1行ずつ出力する。
- `dataset`: testのフォルダ名から`.zarr`を除いた値と一致させる。
- 公式出典: [Evaluation](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/overview/evaluation)、[Code Requirements](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/overview/code-requirements)



## 入力資料: docs/02_metric.md

# 評価指標

このファイルは、公式メトリックをローカル実装と実験判断に落とし込むための運用メモです。実験テンプレートが読む metric 名は `project.yml` の `defaults.metric` を正とし、公式資料からの抜粋と出典は `docs/official/evaluation.md` に置きます。

## 公式メトリック要約

- 名前: adjusted edge Jaccardとdivision Jaccardを組み合わせたtracking metric
- 最適化方向: maximize
- 式: `score = adjusted_edge_jaccard + 0.1 * division_jaccard`
- 実装の出典: [Kaggle Evaluation](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/overview/evaluation)、[主催者公開のmetric仕様](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md)

`adjusted_edge_jaccard`は各サンプルのedge Jaccardをnode数の過剰予測に対する係数で調整し、`TP + FP + FN`で重み付け平均する。nodeの対応付けは同じ時点で行い、物理距離7.0 µm以内のscaled centroid distanceに対するoptimal bipartite assignmentを使う。voxel scaleはz=1.625、y=x=0.40625 µm/voxel。

`division_jaccard`は、2本以上の出力edgeを持つnodeを予測された分裂として扱い、正解の分裂前後にある局所的な系譜との対応からTP、FP、FNを数え、全サンプルでmicro-averageする。正解annotationは疎であり、未annotationの予測を一律にfalse positiveとはしない。



## 入力資料: docs/04_data.md

# データ

実験テンプレートが読む標準データパス、ID 列、提出ターゲット列は `project.yml` を正とします。このファイルは公式データの内容とリスクの運用メモです。

## ファイル

| ファイル | サンプル数 | 形式 | 用途 | メモ |
| --- | ---: | ---: | --- | --- |
| `train/*.zarr` | 199 | Zarr v3、`uint16` | 学習画像 | 全sampleが`(T,Z,Y,X)=(100,64,256,256)` |
| `train/*.geff` | 199 | GEFF / Zarr v3 | 疎な正解tracking graph | 全`.zarr`と1対1で対応 |
| `test/*.zarr` | 4 | Zarr v3、`uint16` | 公開例の推論入力 | 全sampleが同じshape。Notebook rerun時にhidden testへ差し替えられる |
| `sample_submission.csv` | 4 dataset / 20行 | CSV | 提出形式の例 | 12 node行、8 edge行。ローカルraw dataとしては未取得 |

## スキーマ

- サンプルID: `{embryo_id}_{field_of_view}`形式のディレクトリ名。公式例には区切りが追加された長いfield-of-view表現もある。
- 正解node: `nodes/ids`と`nodes/props/{t,z,y,x}/values`。
- 正解edge: `(source_id,target_id)`の2列を持つ`edges/ids`。
- グループ: ディレクトリ名の最初のsegmentである`embryo_id`。
- 時間: Zarr画像配列の第1軸`T`、GEFF node propertyの`t`。
- 画像座標: `z,y,x`。物理scaleはz=1.625、y=x=0.40625 µm/voxel。
- 画像chunk: `(1,64,256,256)`が標準で、timepoint`t`は`0/c/{t}/0/0/0`。

## 分割

- Train: `.zarr`画像と`.geff`正解graphのpair。正解は疎で、`.geff/zarr.json`の`estimated_number_of_nodes`が真の総cell数の概算を持つ。
- Test: `.zarr`画像のみ。公開されるtestはtrainから複製された例で、提出Notebookのrerun時にhidden testへ差し替えられる。
- サンプル提出: node行とedge行を同じ10列CSVに格納する。全test datasetを含める。
- 公式出典: [Data Description](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/data)、[Evaluation](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/overview/evaluation)

## EDA メモ

- Kaggle Notebookで画像chunkを開かずmetadataだけを監査した。trainは199 sample、胚IDは`44b6`と`6bba`の2個で、sample数はそれぞれ71と128。
- 疎な正解は合計133,318 node、128,883 edge。sample単位ではnode数50～1,950（median 659）、edge数49～1,879（median 639）。`estimated_number_of_nodes`の合計は4,725,117。
- train 199 sampleと公開test 4 sampleの画像shapeはすべて`(100,64,256,256)`、chunk shapeは`(1,64,256,256)`。
- `sample_submission.csv`は10列、20行の形式例であり、本番予測の必要行数を指定するtemplateではない。
- 輝度分布、画像内容、division数は画像・edge array本体を読んでいないため未確認。ローカルの`data/raw/`にはデータ本体を置いていない。
- 実行証拠: [`studies/biohub_repository_setup/input_metadata_audit.json`](../studies/biohub_repository_setup/input_metadata_audit.json)。調査コードは同じディレクトリの`input_metadata_audit.py` / `.ipynb`に置く。

## データリスク

- リーク候補: 同じ胚の別field of viewを異なるfoldへ入れると、公式の胚分離を再現できず過大評価になり得る。
- 重複: 公開testはtrainから複製された例であるため、公開testだけを使った動作確認のスコアをhidden test性能の証拠にしない。
- 欠損値: 正解annotationは意図的に疎であり、未annotation cellを通常の欠損labelとして扱わない。
- 分布シフト: hidden testはtrainと胚単位で分離される。胚ごとのcell density、noise、shape、division頻度の差を確認する。


## 当時の学習・計算方針


- **新規候補はtrackerより後の選択・修復を優先する。** 2026-09-27のユーザー依頼を反映し、公開検出器・画像encoder・座標head・既存trackerを固定して比較する。2026-09-12の固定画像モデル下のtracker学習方針を下流処理へ重点化したもので、進行中の実験は維持する。下流採点器を学ぶ場合は入力・教師・損失・復号を候補ごとに定義する。
- **計算環境はKaggle Notebookのみ、GPUは週45時間以内、課金なし。** Colabや外部GPUを前提にしない。Colab例外はユーザー承認済みのexp032と、2026-09-23にGPU不足で切替指示を受けたexp040の学習・診断に限る。初回の検出・特徴抽出、トラッカー学習、検証、提出用推論を予算に含め、実行前の残量と小規模実測で実行数を決める。保存済み予測の集計・graphの再選択は可能ならCPU Notebookで行う。Notebook経過時間と実際のGPU割当消費は区別し、未実測の所要時間を保証しない。
- 公開Notebook・重み・利用条件・学習来歴の選定は[exp011](../experiments/exp011_public_detector_selection/)を参照する。現行比較はexp043の採用構成を使い、自前検出器の再学習や全層更新を必須対照にしない。
- 検出候補・物理座標・検出得点・必要な画像特徴を保存して再利用する。時間を扱う画像モデルの特徴は入力窓に依存するため、元の全時間窓・前処理・crop・変形・padding・精度・抽出座標・重みの版を対応付ける。初回は候補点特徴を中心に等価性と容量を測り、全voxel特徴の一括保存やframe単独の特徴への置換を前提にしない。画像・窓・候補座標等を変える比較は必要な特徴を再抽出し、費用に含める。
- トラッカーの入力は固定検出器の予測と画像特徴、正解の根拠は主催者のGEFFに記録された中心・接続・分裂とする。予測候補を既知注釈に対応付けて教師を作り、検出予測自体を正解とはみなさない。疎い注釈の未知部分を真の負例と断定せず、既存lossの教師maskとその変更を明示的に比較する。未承認の人手注釈や未取得の密な領域教師をあるものとして設計しない。
- [`exp028_direct_graph_prediction`](../experiments/exp028_direct_graph_prediction/)は主催者の既知接続から娘ごとに母または対応なしを学ぶ。ILP擬似教師と未知の負例化は使わない。詳細は同実験の要件を参照。
- 初回の学習比較では検出候補生成・座標・前処理・復号を対照と揃える。候補回収、中心補正、局所再推論は後続の別比較として明示し、公開検出器の重み更新を含めない。検出不足が判明しても検出器の学習を自動で再開せず、固定候補の上限と追加候補回収の条件を記録する。
- 評価は現行公式指標とその成分を使い、両胚別の改善・悪化、失敗と有効件数を示す。公開重みが評価胚を学習した、または来歴不明の場合は、固定公開モデル下の条件付きの比較と明記し、独立した交差検証（CV）とは呼ばない。トラッカーだけの分割では画像モデル由来の学習内評価は解消しない。既存の胚を分けた自前予測は補助診断として区別し、教師・設定の選択は学習側内部に限定する。
- 検出器の再学習、画像特徴抽出器の更新・事前学習、新規画像モデルの学習を要する原案はP4で保留し、再開条件を各詳細に記す。方針変更の明示承認なしに全層更新へ戻さない。固定特徴で別の比較に組み直す場合も、入力・学習対象・対照の変更を詳細へ明記する。



## 評価の作業範囲

主な5案は上記方針で実行可能なものから選ぶ。方針変更を必要とする案も探索の候補として記載できるが、必要な変更を明記し、承認済みの実験や現在実行可能な案と混同しない。新しい画像モデルを固定特徴で置き換えた場合は別案とする。リンク先へ移動せず、渡した資料の本文だけを使う。
