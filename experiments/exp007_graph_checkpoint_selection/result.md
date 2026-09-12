# exp007_graph_checkpoint_selection 結果

## 仮説

固定公式graph指標でcheckpointを選ぶと、現行のedge accuracyとnode recallの積による選択より、外側の胚holdoutで良いepochを選べる。

## 記録の参照先

- 設定、系譜、再現性方針: [`config.yaml`](config.yaml)
- CV/LB、実験status、kernel情報、Kaggle Notebook実行時間、生成物SHA: [`metrics.json`](metrics.json)
- 実行コマンドと途中経過: [`SESSION_NOTES.md`](SESSION_NOTES.md)

## 実行証拠

- 比較対象: 同じ学習runの現行proxy selector。exp005は既存基準の副参照。
- `metrics.json` の参照キー: `status=running`、`evidence.kaggle.kernel_version=1`、`evidence.kaggle.notebook_runtime_seconds=23633.582541708998`、`evidence.artifacts.model_count=6`、`evidence.artifacts.model_manifest_sha`。
- `SESSION_NOTES.md` の実行記録: 実装、静的検査、push、正常終了、実測T4 2基、runtime gate、生成物確認を記録。
- 参照する生成物: Kaggle train version 1の6 checkpoint、bundle/fold manifest、split、smoke/training summary。小規模な証拠だけを`/tmp/kaggle-output/exp007_graph_checkpoint_selection/train-evidence/`へ取得し、モデル本体はKaggle kernel outputを正とする。

## 解釈

trainは正常終了し、2 fold × 3 epochのcheckpointを保存できた。現行proxyは両foldともepoch 2を選んだ。固定公式graph指標による内部選択と外側公式指標を計算するevaluation version 1は実行中であり、selectorの差や両胚での改善可否についてまだ結論を出さない。

## ユーザー判断

- 判断: 未判断
- 確認日時 / 依頼メッセージ: 2026-09-12の「完了しました」はKaggle train runの完了通知として扱う。実験全体の完了・採用・不採用の判断ではない。
- 理由: 仮説を判定する2 selector比較と外側胚評価が未実行のため。

## 次

evaluation version 1の完了後、両胚の公式指標、2 selectorの選択epoch、失敗件数、Notebook実行時間を提示して判断を求める。
