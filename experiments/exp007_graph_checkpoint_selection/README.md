# exp007_graph_checkpoint_selection

## 概要

- 仮説要約: 学習内proxyではなく固定公式graph指標でepochを選ぶと、外側の胚holdoutでより良いcheckpointを選べる可能性がある。
- 変更点要約: exp005の全epoch checkpointを保存し、同じ内部選択動画へ現行proxyと公式graph指標を適用する。
- リスク: outer評価胚を選択へ使うleakage、checkpoint保存容量、selectorが選ぶunique model数による推論時間増加。
- 次: ユーザーが実装を指示した後、exp005を参照してepoch保存と公式graph selectorを実装する。

## 現在の状態

- backlog `graph_checkpoint`から実験化済み、未実装、未実行。
- `metrics.json.status`: `planned`
- Notebookは雛形のままであり、Kaggleへpushしない。

## 正の記録

- 数値、実験status、構造化された実行証拠: [`metrics.json`](metrics.json)
- route、設定、系譜: [`config.yaml`](config.yaml)
- 実装前の要件、実装方法、受け入れ条件: [`requirements.md`](requirements.md)
- 証拠への参照、結果の解釈、ユーザー判断: [`result.md`](result.md)
- 実行中の作業ログ: [`SESSION_NOTES.md`](SESSION_NOTES.md)

## 実行入口

- train/inference Notebookは未実装の雛形であり、実装承認後に必要な処理とテストを追加する。
- Kaggle prepare、push、実行は今回の実験化に含めない。予定は[`SESSION_NOTES.md`](SESSION_NOTES.md)を参照する。

## 表記

用語は`AGENTS.md`と`docs/glossary.md`を正とする。
