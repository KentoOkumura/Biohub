# 検証方針

実験テンプレートが読む fold 数、seed、metric、共通 validation 方針は `project.yml` を正とします。このファイルは設計意図とリークチェックの運用メモです。

## CV 設計

- Fold 作成方法: primary validationは、一方の胚で学習してもう一方で評価する2方向のleave-one-embryo-outとする。同じ`embryo_id`のsampleをtrainとvalidationに分割しない。
- グループキー: サンプルディレクトリ名の最初の`_`より前にある`embryo_id`。
- 層化キー: 未設定。各胚のサンプル数、node/edge数、分裂数を確認してから必要性を判断する。
- ランダムシード: 42（暫定）。
- Fold 数: 2。胚groupが2個なので、group分離を保つ5-foldは行わない。
- メトリック: `adjusted_edge_jaccard + 0.1 * division_jaccard`。
- 主な検証コマンド:
  ```bash
  task validate-exp EXP=expXXX_title
  task prepare-kaggle-notebooks EXP=expXXX_title EXTRA_ARGS="--strict"
  task push-kaggle-train EXP=expXXX_title
  task kaggle-logs KERNEL=<username>/<train-kernel-slug>
  ```

## リークチェックリスト

- [ ] 同一グループのデータが train/valid にまたがっていないか。
- [ ] 推論入力に正解ラベルなど、推論時に利用できない情報を混ぜていないか。
- [ ] test 由来の統計量を train に使っていないか。
- [ ] target encoding が fold 外データを参照していないか。
- [ ] augmentation や前処理が validation に不適切に影響していないか。
- [ ] 学習時と推論時の前処理が一致しているか。
- [ ] CV と LB の数値を`metrics.json`へ記録し、乖離の解釈を`result.md`へ記録しているか。
- [ ] 教師生成・内部予測・校正まで含めた分割を記録し、評価側の正解を教師や設定選択へ混ぜていないか。外側の評価結果を繰り返し見て方式を選んだ結果を、未使用データでの評価と呼んでいないか。

Biohubでは同一test動画の後続画像も推論入力に含まれるため、後続フレームの利用だけをリークと判定しない（[公式baselineの時間入力](temporal_unet3d_explainer.md#1-入力と3d座標)）。学習側のGEFFから教師を作ることと、評価側の正解を学習に使うことを区別する。

## CV/LB 乖離の確認

横断比較は自動生成される`experiment_summary.md`で確認し、数値を手作業で転記しません。乖離の原因、比較条件、未解決事項は対応する実験の`result.md`へ記録します。

## 検証判断

- 公式splitが胚単位で分離されるため、同じ`embryo_id`がtrainとvalidationにまたがる分割は禁止する。
- fold間のcombined scoreだけでなく、adjusted edge Jaccard、division Jaccard、node数の過剰予測率、edge/divisionのTP/FP/FNを別々に保存する。
- 実測は199 sample、2胚で、`44b6`が71 sample、`6bba`が128 sample。2-foldは公式の胚分離を再現できる一方、各foldの学習が1胚だけになり分散が大きい。sample単位の5-foldは学習量を確保できる一方、同じ胚がtrain/validationにまたがるためprimary scoreには使わない。
- secondary validationとして、公開Notebookと同じ固定8 sampleのholdoutを使う。primary scoreとは混ぜず、公開Notebook由来の手法を同条件で比較する補助評価として記録する。
- 実行証拠: [`input_metadata_audit.json`](../studies/biohub_repository_setup/input_metadata_audit.json)。

## 主催者baselineと公開Notebookの設定

- 主催者公開baselineの学習コードは、precomputed split JSONの`train`と`test`を読み、split index 0～4を扱える。splitの内容はJSON次第で、学習script自体は`embryo_id`の分離を検査しない。split fileがない場合はseed 0でsampleをshuffleした90%/10%の単一splitを作る。
- 2026-08-14にvote順上位15件の公開Notebookを保存してコードを確認した。15件すべてで学習ループと`KFold`/`GroupKFold`の呼び出しはなく、11件は学習済みの`weights/unet_transformer/split_0/edge_predictor_best.pth`を読み込む推論Notebookだった。`split_0`というpathだけでは5-fold学習済みであることを意味しない。
- `Clean Approach + Lightweight Local CV`は学習を行わず、`44b6`から4 sample、`6bba`から4 sampleの固定8 sampleを`split_0` checkpointで推論する。8-foldではなく「8 sampleのholdout」であり、foldの反復はない。secondary validationには、このNotebookと同じ`44b6_0113de3b`、`44b6_0b24845f`、`44b6_341df25f`、`44b6_e57ff5c6`、`6bba_05b6850b`、`6bba_05db0fb1`、`6bba_969618f6`、`6bba_fc83837d`を使う。checkpointの学習データから同じ胚の他sampleが除外された証拠はNotebook内にないため、embryo-disjoint primary CVの代わりにはしない。
- 上記Notebookのローカル評価は、主催者の`metrics.md`をもとに参加者が書いた実装で、コード自身が`OFFICIAL_SPEC_EXACT_SOURCE_COPY = False`と記録している。公開Notebook間のA/B比較用の補助validationとしては参照できるが、Kaggle hidden scorerそのものではない。
- 詳細と保存したNotebook: [`Biohub リポジトリ設定・validation調査`](surveys/biohub-repository-setup-validation_20260814.md)、[`docs/notebooks`](notebooks/biohub-cell-tracking-during-development/)。

## 「validator」と「評価」の区別

- Kaggle hidden scorer: 提出後にKaggle側で実行される公式採点。ローカルから直接実行できない。
- 主催者公開のmetric実装: `royerlab/kaggle-cell-tracking-competition`の`tracking_cellmot.metrics`と`scripts/evaluate.py`。正解GEFFがあるtrain sampleでscoreを計算する。
- 参加者Notebookのローカル評価: 上記仕様を参加者が再実装したもの。実装差があり得る。
- 当リポジトリの[`scripts/validate_submission.py`](../scripts/validate_submission.py): 主催者提供物ではない提出形式検証。本コンペではsample submissionとの行数一致を要求せず、列順、連続`id`、node/edge行の`-1`、dataset内のnode ID、edge参照、次時点への接続を確認する。test `.zarr`が存在する場合はdataset網羅と座標範囲も確認する。
