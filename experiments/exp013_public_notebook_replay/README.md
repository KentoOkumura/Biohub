# exp013_public_notebook_replay

## 概要

- 仮説要約: 固定した公開Notebook、canonical artifact、T4 2基を使う2回のclean runで、候補座標・graph・`submission.csv`を同じ内容として再生成できる。
- 変更点要約: exp011の静的選定から、公開sourceのfull inferenceへ進める。予測処理は保持し、canonical mount、入力・wheel manifest、replay receipt、2 run比較を追加する。
- リスク: CUDA・SCIP・2 GPU processの決定性は強制していない。公開testで一致してもhidden testや別環境の一致、独立validationは保証しない。
- 結果: private Kaggle Notebook version 1と2はともに成功し、公開test全4動画について13比較項目と241,282行の`submission.csv`が一致した。ユーザー判断により、後続比較の基準として実験statusを`usable`に確定した。
- 提出: code submission ref `56199738`は`COMPLETE`、Public LBは`0.944`。現行の自前提出ベスト`0.693`を`0.251`上回った。
- 次: 固定予測と実測費用を、`exp014_exact_window_cache`以降のtracker比較と診断へ引き継ぐ。

## 正の記録

- 数値、実験status、構造化された実行証拠: [`metrics.json`](metrics.json)
- route、設定、系譜: [`config.yaml`](config.yaml)
- 実装前の要件、実装方法、受け入れ条件: [`requirements.md`](requirements.md)
- 証拠への参照、結果の解釈、ユーザー判断: [`result.md`](result.md)
- 実行中の作業ログ: [`SESSION_NOTES.md`](SESSION_NOTES.md)

## 実行入口

- 推論 notebook: `exp013_public_notebook_replay_inference.ipynb`
- 2 run比較: `compare_replays.py <first-output-dir> <second-output-dir>`
- Kaggle 準備と実行: [`SESSION_NOTES.md`](SESSION_NOTES.md)の予定を埋め、`kaggle-review-exp`と`kaggle-platform`の手順に従う
- notebook 実行: Kaggle kernel run を正とする。ローカル実行は `--allow-local` を付けた smoke debug のみに限定する。

## 表記

用語は`AGENTS.md`の規則を正とし、公式資料、参加者の説明、論文、既存コードで実際に使われている専門用語を優先する。コンペ固有の略語とリポジトリ内の管理用語は`docs/glossary.md`で定義する。
