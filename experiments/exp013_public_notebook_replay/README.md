# exp013_public_notebook_replay

## 概要

- 仮説要約: 固定した公開Notebook、canonical artifact、T4 2基を使う2回のclean runで、候補座標・graph・`submission.csv`を同じ内容として再生成できる。
- 変更点要約: exp011の静的選定から、公開sourceのfull inferenceへ進める。予測処理は保持し、canonical mount、入力・wheel manifest、replay receipt、2 run比較を追加する。
- リスク: CUDA・SCIP・2 GPU processの決定性は強制していない。公開testで一致してもhidden testや別環境の一致、独立validationは保証しない。
- 結果: private Kaggle Notebook version 1と2はともに成功し、公開test全4動画について13比較項目と241,282行の`submission.csv`が一致した。実験statusはユーザー判断待ちの`debug_completed`。
- 提出: code submission ref `56199738`を作成した。現在はKaggleの採点待ちで、Public LBは未取得。
- 次: ref `56199738`の採点完了後にPublic LBと採点所要時間を記録し、第1段階と提出結果の完了判断をユーザーへ依頼する。

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
