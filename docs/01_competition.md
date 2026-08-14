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

## リポジトリ設定の確認結果

- コンペデータはユーザー指定によりローカルへ未ダウンロード。Kaggle上のmetadata監査ではtrain 199 sample、胚ID 2個（`44b6`: 71、`6bba`: 128）、公開test 4 sampleを確認した。
- このリポジトリ独自の汎用CSV検証スクリプト[`scripts/validate_submission.py`](../scripts/validate_submission.py)は、sample submissionと同じ行数・同じ`id`列を要求する。そのため、予測graphに応じて行数が変わる本コンペには未対応である。これは主催者提供のvalidatorではない。
- 主催者が公開しているのは、評価仕様と[`tracking_cellmot.metrics`](https://github.com/royerlab/kaggle-cell-tracking-competition)を含む参照実装である。Kaggle上のhidden scorer、この参照実装、参加者のローカル再実装、当リポジトリの提出形式検証を混同しない。
- 実行証拠: [`studies/biohub_repository_setup/input_metadata_audit.json`](../studies/biohub_repository_setup/input_metadata_audit.json)。実行したprivate Kaggle Notebookは`kentookumura/exp001-input-audit-diagnostic` version 1で、名称に当初の誤分類が残る。
- 詳細調査: [`Biohub リポジトリ設定・validation調査`](surveys/biohub-repository-setup-validation_20260814.md)。

## 未解決の質問

- 胚groupが2個しかないため、胚をtrain/validationにまたがせない5-foldは作れない。2方向のleave-one-embryo-outをprimary validationにするかはユーザー判断前であり、`project.yml`の`primary_validation`と`n_folds`は`TODO`とした。
- 最初の提出候補を作る前に、本コンペ用の可変行数graph検証を実装する必要がある。検証契約と実装範囲は未承認。
