# exp007_graph_checkpoint_selection

## 概要

- 仮説要約: 学習内proxyではなく固定公式graph指標でepochを選ぶと、外側の胚holdoutでより良いcheckpointを選べる可能性がある。
- 変更点要約: exp005の全epoch checkpointを保存し、同じ内部選択動画へ現行proxyと公式graph指標を適用する。
- リスク: outer評価胚を選択へ使うleakage、checkpoint保存容量、selectorが選ぶunique model数による推論時間増加。
- 次: 実行中のevaluation version 1について、2 selectorの選択epochと外側公式指標を確認する。

## 現在の状態

- backlog `graph_checkpoint`から実験化し、train/evaluation Notebookと実験固有testを実装済み。Kaggle train version 1は正常終了した。
- `metrics.json.status`: `running`
- 6 checkpointとmanifestを保存済み。内部動画での2 selector比較とunique checkpointだけの外側評価を行うKaggle evaluation version 1は実行中。

## 正の記録

- 数値、実験status、構造化された実行証拠: [`metrics.json`](metrics.json)
- route、設定、系譜: [`config.yaml`](config.yaml)
- 実装前の要件、実装方法、受け入れ条件: [`requirements.md`](requirements.md)
- 証拠への参照、結果の解釈、ユーザー判断: [`result.md`](result.md)
- 実行中の作業ログ: [`SESSION_NOTES.md`](SESSION_NOTES.md)

## 実行入口

- train Notebookは各foldのepoch 0、1、2を保存し、SHAとproxy scoreをmanifestへ記録する。
- evaluation Notebookは全6 checkpointを内部選択動画で公式評価し、同点時は現行の`>=`保存と一致するlatest epochを選ぶ。外側ではselectorが選んだunique checkpointだけを推論して公式指標を計算する。
- train実行証拠とevaluationの予定は[`SESSION_NOTES.md`](SESSION_NOTES.md)を参照する。

## 表記

用語は`AGENTS.md`と`docs/glossary.md`を正とする。
