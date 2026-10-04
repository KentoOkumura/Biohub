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
- 初回のmetadata監査では画像・edge array本体を読んでいない。その後の[公式アノテーション分布調査](surveys/biohub-official-annotation-distribution_20260815.md)で全train GEFFを読み、注釈済みの分裂数、胚別の注釈密度、完全注釈領域を示すmaskがないことを確認した。輝度分布や真の未注釈細胞の分布は、このGEFF調査では確認できない。ローカルの`data/raw/`にはデータ本体を置いていない。
- 実行証拠: [`studies/biohub_repository_setup/input_metadata_audit.json`](../studies/biohub_repository_setup/input_metadata_audit.json)。調査コードは同じディレクトリの`input_metadata_audit.py` / `.ipynb`に置く。

## データリスク

- リーク候補: 同じ胚の別field of viewを異なるfoldへ入れると、公式の胚分離を再現できず過大評価になり得る。
- 重複: 公開testはtrainから複製された例であるため、公開testだけを使った動作確認のスコアをhidden test性能の証拠にしない。
- 欠損値: 正解annotationは意図的に疎であり、未annotation cellを通常の欠損labelとして扱わない。
- 分布シフト: hidden testはtrainと胚単位で分離される。胚ごとのcell density、noise、shape、division頻度の差を確認する。
