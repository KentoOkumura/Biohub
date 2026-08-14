# 検証方針

実験テンプレートが読む fold 数、seed、metric、共通 validation 方針は `project.yml` を正とします。このファイルは設計意図とリークチェックの運用メモです。

## CV 設計

- Fold 作成方法: 同じ胚のサンプルをtrainとvalidationに分割しない方法をprimary候補とする。Kaggle上の監査で胚は2個だけと分かったため、実行可能なのは一方の胚で学習して他方で評価する2方向のleave-one-embryo-outである。採用はユーザー判断前。
- グループキー: サンプルディレクトリ名の最初の`_`より前にある`embryo_id`。
- 層化キー: 未設定。各胚のサンプル数、node/edge数、分裂数を確認してから必要性を判断する。
- ランダムシード: 42（暫定）。
- Fold 数: 胚groupが2個なのでgroup分離を保つ5-foldは不可能。暫定値の5は削除し、`project.yml`では`TODO`とした。2-foldへ変更するかは未判断。
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
- [ ] 時系列データで未来情報を使っていないか。
- [ ] test 由来の統計量を train に使っていないか。
- [ ] target encoding が fold 外データを参照していないか。
- [ ] augmentation や前処理が validation に不適切に影響していないか。
- [ ] 学習時と推論時の前処理が一致しているか。
- [ ] CV と LB の数値を`metrics.json`へ記録し、乖離の解釈を`result.md`へ記録しているか。

## CV/LB 乖離の確認

横断比較は自動生成される`experiment_summary.md`で確認し、数値を手作業で転記しません。乖離の原因、比較条件、未解決事項は対応する実験の`result.md`へ記録します。

## 検証判断

- 公式splitが胚単位で分離されるため、同じ`embryo_id`がtrainとvalidationにまたがる分割は禁止する。
- fold間のcombined scoreだけでなく、adjusted edge Jaccard、division Jaccard、node数の過剰予測率、edge/divisionのTP/FP/FNを別々に保存する。
- 実測は199 sample、2胚で、`44b6`が71 sample、`6bba`が128 sample。2-foldは公式の胚分離を再現できる一方、各foldの学習が1胚だけになり分散が大きい。sample単位の5-foldは学習量を確保できる一方、同じ胚がtrain/validationにまたがるためprimary scoreには使わない。
- fold数と補助validationの採否は、このtrade-offをユーザーが判断した後に確定する。
- 実行証拠: [`input_metadata_audit.json`](../studies/biohub_repository_setup/input_metadata_audit.json)。

## 主催者baselineと公開Notebookの設定

- 主催者公開baselineの学習コードは、precomputed split JSONの`train`と`test`を読み、split index 0～4を扱える。splitの内容はJSON次第で、学習script自体は`embryo_id`の分離を検査しない。split fileがない場合はseed 0でsampleをshuffleした90%/10%の単一splitを作る。
- 2026-08-14にvote順上位15件の公開Notebookを保存してコードを確認した。15件すべてで学習ループと`KFold`/`GroupKFold`の呼び出しはなく、11件は学習済みの`weights/unet_transformer/split_0/edge_predictor_best.pth`を読み込む推論Notebookだった。`split_0`というpathだけでは5-fold学習済みであることを意味しない。
- `Clean Approach + Lightweight Local CV`は学習を行わず、`44b6`から4 sample、`6bba`から4 sampleの固定8 sampleを`split_0` checkpointで推論する。8-foldではなく「8 sampleのholdout」であり、foldの反復はない。checkpointの学習データから同じ胚の他sampleが除外された証拠はNotebook内にないため、embryo-disjoint primary CVの代わりにはしない。
- 上記Notebookのローカル評価は、主催者の`metrics.md`をもとに参加者が書いた実装で、コード自身が`OFFICIAL_SPEC_EXACT_SOURCE_COPY = False`と記録している。公開Notebook間のA/B比較用の補助validationとしては参照できるが、Kaggle hidden scorerそのものではない。
- 詳細と保存したNotebook: [`Biohub リポジトリ設定・validation調査`](surveys/biohub-repository-setup-validation_20260814.md)、[`docs/notebooks`](notebooks/biohub-cell-tracking-during-development/)。

## 「validator」と「評価」の区別

- Kaggle hidden scorer: 提出後にKaggle側で実行される公式採点。ローカルから直接実行できない。
- 主催者公開のmetric実装: `royerlab/kaggle-cell-tracking-competition`の`tracking_cellmot.metrics`と`scripts/evaluate.py`。正解GEFFがあるtrain sampleでscoreを計算する。
- 参加者Notebookのローカル評価: 上記仕様を参加者が再実装したもの。実装差があり得る。
- 当リポジトリの[`scripts/validate_submission.py`](../scripts/validate_submission.py): 提出CSVの列、行数、ID、欠損などを確認する汎用形式検証。主催者提供物ではなく、現状は本コンペの可変行数graphに未対応。
