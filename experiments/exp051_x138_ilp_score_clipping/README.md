# exp051_x138_ilp_score_clipping

## 概要

- 目的: 固定した拡張候補で、ILP接続得点の上限処理が費用調整のみを上回るか両胚の全graph公式指標で比較する。
- 差分: 接続確率の候補・後処理での使用を維持し、ILP費用だけに上限を適用する。
- リスク: 同点の増加でILP選択と実行時間が変わる。固定ID順、最適終了と再実行一致を確認する。
- 次: 外側20動画の比較とユーザー判断の根拠は[result.md](result.md)を参照する。後段で選択が消えた問題と提出推論時間の制約は、後続候補を設計する際に個別に確認する。

## 正の記録

- [要件](requirements.md)
- [設定](config.yaml)
- [実行記録](SESSION_NOTES.md)
- [数値と証拠](metrics.json)
- [結果の解釈](result.md)

## 実行入口

- 内部調整: `exp051_x138_ilp_score_clipping_tune0_control.ipynb`、`exp051_x138_ilp_score_clipping_tune0_clipped.ipynb`、`exp051_x138_ilp_score_clipping_tune1_control.ipynb`、`exp051_x138_ilp_score_clipping_tune1_clipped.ipynb`
- 外側評価: `exp051_x138_ilp_score_clipping_evaluate.ipynb`
