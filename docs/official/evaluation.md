# 評価

公式資料から確認した一次情報と出典を置く場所です。ローカル実装、エッジケース、スコア解釈は `docs/02_metric.md` に記録します。提出形式の運用仕様は `docs/01_competition.md` に集約します。

## 公式メトリック

- 名前: adjusted edge Jaccardとdivision Jaccardを組み合わせたtracking metric
- 良い値の向き: 大きいほど良い
- 式: `score = adjusted_edge_jaccard + 0.1 * division_jaccard`
- 出典: [Kaggle Evaluation](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/overview/evaluation)
- 詳細仕様: [Biohub SF / Royer Lab公開のmetrics.md](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md)

### Edge

- 同じtimepointの予測nodeと正解nodeを、scaled centroid distanceが7.0 µm以内になるoptimal bipartite assignmentで対応付ける。
- 両端が正解edgeの両端へ対応する予測edgeをTPとし、edge Jaccardを`TP / (TP + FP + FN)`で計算する。
- 疎な正解に対応するため未対応node/edgeの扱いに例外があり、総予測node数が真のnode数の概算を超える場合はpenaltyを掛ける。
- 各サンプルのadjusted edge Jaccardを`TP + FP + FN`で重み付け平均する。

### Division

- 2本以上の出力edgeを持つ予測nodeを分裂候補として扱う。
- 正解の分裂前後にある局所的な系譜と予測graphを対応付け、divisionのTP、FP、FNからJaccardを計算する。
- 全サンプルのdivision TP、FP、FNを合計するmicro-averageを使う。
- 疎な正解を考慮するmetricの性質上、combined scoreは1.0を超える場合がある。

## 公式例との照合

- 公式サンプル/例: `sample_submission.csv`と主催者公開metric repository。
- 照合結果: データをダウンロードしていないため未実施。
- 参照した実装/ページ: [Kaggle Evaluation](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/overview/evaluation)、[metrics.md](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md)

## 公開実装の位置付け

- 主催者側のRoyer Labは、baselineとmetricの参照実装を[`royerlab/kaggle-cell-tracking-competition`](https://github.com/royerlab/kaggle-cell-tracking-competition)で公開している。
- `scripts/evaluate.py`と`tracking_cellmot.metrics`は、正解GEFFがあるデータに対するローカル評価用である。Kaggle上でhidden testを採点するscorerそのものではない。
- このリポジトリの`scripts/validate_submission.py`は上記repositoryから提供されたものではなく、このリポジトリ独自の提出形式検証である。本コンペ向けの可変行数tracking graph検証を持つが、scoreは計算しない。

## 提出形式の公式抜粋

- ID 列: `id`。0から始まる連続整数。
- 必須列: `id,dataset,row_type,node_id,t,z,y,x,source_id,target_id`
- node行: `row_type=node`、`node_id,t,z,y,x`を設定し、`source_id,target_id=-1`。
- edge行: `row_type=edge`、`source_id,target_id`を設定し、`node_id,t,z,y,x=-1`。
- 必要行数: 固定ではない。hidden testの全datasetについて予測nodeとedgeを出力する。
- サンプルファイル: `data/raw/sample_submission.csv`
- ローカル状態: ユーザー指定により未ダウンロード。
