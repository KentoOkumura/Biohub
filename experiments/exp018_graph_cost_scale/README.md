# exp018_graph_cost_scale

## 概要

- 仮説要約: exp015の固定ILP前候補graphでは、`edge_prob`と固定event costの相対スケールを変えるだけで、現行の`alpha=1`より正しい接続と分裂を残せる可能性がある。
- 変更点要約: `edge_weight=-alpha * edge_prob`の`alpha`だけを`[0.25, 0.5, 1, 2, 4]`で比較する。44b6で選び6bbaで評価する方向と、その逆方向を分ける。
- リスク: 公開モデルを使うtrain上の条件付き比較で独立CVではない。初段はrepairなしなので、最終pipelineの改善を直接示さない。
- 次: 初段は全199件・failure 0で完了したが、両方向の公式combined score改善gateが不成立だった。ユーザー判断により実験は完了、手法は不採用（`discarded`）とし、固定repair段階へ進まない。

## 正の記録

- 数値、実験status、構造化された実行証拠: [`metrics.json`](metrics.json)
- route、設定、系譜: [`config.yaml`](config.yaml)
- 実装前の要件、実装方法、受け入れ条件: [`requirements.md`](requirements.md)
- 証拠への参照、結果の解釈、ユーザー判断: [`result.md`](result.md)
- 実行中の作業ログ: [`SESSION_NOTES.md`](SESSION_NOTES.md)

## 実行入口

- CPU alpha shard Notebook: `exp018_graph_cost_scale_alpha_025.ipynb`、`alpha_05.ipynb`、`alpha_1.ipynb`、`alpha_2.ipynb`、`alpha_4.ipynb`。
- 集約Notebook: `exp018_graph_cost_scale_aggregate.ipynb`。
- 共通実装: `exp018_graph_cost_scale_diagnostic.py`。
- Kaggle準備: `make prepare-kaggle-notebooks EXP=exp018_graph_cost_scale EXTRA_ARGS="--notebook alpha_025 --run-on-push"`の`alpha_025`を各shard名へ置き換えて実行し、5 shard完了後に`aggregate`を実行する。
- 正式実行はKaggle CPU、internet無効とする。各shardはsample順を固定した逐次実行で、aggregateはSHAとcoverageを満たす全shardだけを受け入れる。

## 初段と後段

初段は固定candidate graphを5つのalphaでILP solveし、repair前の公式metricだけを比較する。後段は初段gateを通った後に同じ`exp018_graph_cost_scale`へ追加し、exp015のmotion、gap、safe-division、DeepCenter repairを変更せず比較する。初段gateに失敗した場合、後段は実装・実行しない。
