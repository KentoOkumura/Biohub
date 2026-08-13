# 評価指標

このファイルは、公式メトリックをローカル実装と実験判断に落とし込むための運用メモです。実験テンプレートが読む metric 名は `project.yml` の `defaults.metric` を正とし、公式資料からの抜粋と出典は `docs/official/evaluation.md` に置きます。

## 公式メトリック要約

- 名前: adjusted edge Jaccardとdivision Jaccardを組み合わせたtracking metric
- 最適化方向: maximize
- 式: `score = adjusted_edge_jaccard + 0.1 * division_jaccard`
- 実装の出典: [Kaggle Evaluation](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/overview/evaluation)、[主催者公開のmetric仕様](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md)

`adjusted_edge_jaccard`は各サンプルのedge Jaccardをnode数の過剰予測に対する係数で調整し、`TP + FP + FN`で重み付け平均する。nodeの対応付けは同じ時点で行い、物理距離7.0 µm以内のscaled centroid distanceに対するoptimal bipartite assignmentを使う。voxel scaleはz=1.625、y=x=0.40625 µm/voxel。

`division_jaccard`は、2本以上の出力edgeを持つnodeを予測された分裂として扱い、正解の分裂前後にある局所的な系譜との対応からTP、FP、FNを数え、全サンプルでmicro-averageする。正解annotationは疎であり、未annotationの予測を一律にfalse positiveとはしない。

## ローカル実装

- 実装先: 未実装。最初の実験契約で主催者公開実装を固定し、対象実験の補助モジュールまたは共通利用する場合は`src/`へ置く。
- 入力: 予測tracking graph、疎な正解tracking graph、正解総node数の概算値、voxel scale。
- 出力: adjusted edge Jaccard、division Jaccard、combined score、および解釈に必要なTP/FP/FN。
- 公式例との照合: データと公式metric実装を未取得のため未実施。

## エッジケース

- 欠損予測: hidden test内の全データセットを提出に含める。node/edgeがない場合の許容形式は実装前に公式metricで確認する。
- 重複 ID: `id`はCSV内で一意な連続整数、`node_id`は少なくとも各dataset内でedge参照を一意に解決できる必要がある。
- 不正な値域: node座標はvoxel単位の整数。edge行の`node_id,t,z,y,x`とnode行の`source_id,target_id`には`-1`を設定する。
- 同点: 公式のtie-break規則は未確認。

## 解釈

- 意味のあるスコア変動: baselineと複数foldの分布を得るまで閾値を設定しない。
- 想定されるpublic/privateの変動: train/testが胚単位で分離され、public/private testの胚構成は公開されないため、少数胚への過適合を主要なリスクとして扱う。
- 既知の不一致リスク: 疎なannotation、node過剰予測penalty、物理距離に基づくnode matching、分裂前後の局所的な対応を簡略化すると公式評価と一致しない。
