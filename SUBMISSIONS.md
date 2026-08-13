# 提出履歴

この表はsubmission ref単位の横断履歴であり、各提出の最終スナップショットを保持します。採点状態と所要時間の詳細な時系列の正は対応実験の`SESSION_NOTES.md`、CV/LBとNotebook実行時間の正は`metrics.json`です。CV/LBは`record-submission`が`metrics.json`から取得します。メモ欄に横断比較用の最終値を置く場合は、Kaggle submissionの採点状態を`submission_status`、Notebook全体の実行時間を`notebook_runtime_seconds`、提出から採点確定までの所要時間を`scoring_elapsed_minutes`で記録します。Notebook内の部分処理時間は処理名を付けた`*_elapsed_seconds`とし、意味が曖昧な`status`、`runtime`は使いません。

| バージョン | 日付 | 実験 | ファイル | 行数 | 列 | SHA256 | CV | Public LB | Private LB | submission ref | メモ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
