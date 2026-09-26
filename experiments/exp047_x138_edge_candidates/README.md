# exp047_x138_edge_candidates

## 概要

exp043と同じ得点・ILP費用・後処理で、ILPへ渡す接続候補を広げる固定比較です。学習側20動画の前段診断と各胚1動画のpilotを通過した場合に、head学習外20動画の公式評価へ進みます。

- 差分: `p > 0.48`の全辺を残し、`p > 0.10`かつ娘ごと上位3親の辺を追加する。
- リスク: 候補増加によるILPの実行時間。公開画像モデルの学習動画を使うため独立CVではない。
- 次: 保存済み得点から接続得点・ILP・後処理の寄与を[exp049の段階別診断](../exp049_x138_edge_selection_diagnostic/)で確認した。

## 正の記録

契約は[requirements.md](requirements.md)、設定は[config.yaml](config.yaml)、時系列は[SESSION_NOTES.md](SESSION_NOTES.md)、数値と証拠は[metrics.json](metrics.json)、解釈は[result.md](result.md)へ記録する。

## 実行入口

[pilot Notebook](exp047_x138_edge_candidates_pilot.ipynb)で学習側20動画の前段診断と各胚1動画の全graphを実行する。進行条件を通過した場合、[推論Notebook](exp047_x138_edge_candidates_inference.ipynb)で評価側20動画の全graphと公式評価を実行する。
