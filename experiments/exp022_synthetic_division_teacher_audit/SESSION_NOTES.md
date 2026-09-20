# exp022_synthetic_division_teacher_audit セッションノート

## 目的

公開合成系譜の完全なnode・edge・division配列から、分裂組と通常継続母の候補組を数える。元のdivision_tripletsの学習設計はこの結果を見て判断する。

## 現在の作業

- 2026-09-19: ユーザーが推奨の公開合成系譜小規模診断を選択。直接承認の診断実験として作成。
- 公開Notebook source、outputのmanifest・metadata、先頭1時系列のSHAとschemaを確認。先頭32時系列を固定した。
- self-contained CPU診断Notebookを実装。先頭1時系列のlocal smokeで47分裂中45組を回収。local smokeは公式実行ではない。
- 2026-09-20: ユーザーがexp022の完了を判断。次は固定公開検出器との接続を小規模診断する。

## コマンドログ

- `make new-exp EXP=exp022_synthetic_division_teacher_audit` — 雛形を作成。
- `uv run kaggle kernels status josefreitasalvesneto/biohub-synthetic-dataset` — 公開sourceがCOMPLETEと確認。
- `uv run kaggle kernels pull josefreitasalvesneto/biohub-synthetic-dataset -p /tmp/biohub-synthetic-source -m` — sourceとmetadataを取得。
- `uv run kaggle kernels output josefreitasalvesneto/biohub-synthetic-dataset -p /tmp/biohub-synthetic-sample --file-pattern '(^|/)(manifest\.json|metadata\.json|seq_0000\.npz)$' --page-size 200` — manifest、metadata、先頭時系列を取得。
- `uv run --extra notebook jupytext --to ipynb experiments/exp022_synthetic_division_teacher_audit/exp022_synthetic_division_teacher_audit_diagnostic.py` — 診断Notebookを生成。

## 次のアクション

固定公開検出器から得る候補・特徴と合成系譜の教師の対応を、別の小規模診断で検証する。

## Kaggle実行前確認

- 2026-09-19 14:27 UTC: Jupytext `--test`、`make validate-exp`、`make check-exp`、`make test-exp`を通過。2テスト成功。
- 生成metadataは`kentookumura/exp022-synthetic-division-teacher-audit`、CPU、TPU無効、internet無効、run_on_push有効、公開合成Notebookをkernel sourceに指定。CPUのためGPU週次quota確認は不要。Kaggle CPUで先頭32時系列の診断を実行する判断。
- 2026-09-19 14:35 UTC: version 1の実行は32/32時系列で成功し、summary・CSVのSHAが一致。正例回収済み母と誤組を同母で比較できる件数が必要と分かり、専用計数を追加。再検証はJupytext、validate-exp、check-exp、test-expとも通過。version 2も同じcanonical kernel id、CPU、TPU無効、internet無効、公開kernel sourceのみ。GPU quota確認不要と判断。
- 2026-09-19 14:35 UTC: `make push-kaggle-notebook EXP=exp022_synthetic_division_teacher_audit NOTEBOOK=diagnostic`で同じkernelのversion 2を作成。`make kaggle-logs`で32/32の完走を確認。
- `make kaggle-output ... OUT=experiments/exp022_synthetic_division_teacher_audit/artifacts/kaggle_v2`で出力を取得。manifestのCSV・summary SHAと取得ファイルを照合し、CSV32行の全集計がsummaryと一致。
- `make record-exp ... STATUS=debug_completed ... --no-summary`で実行状態を記録。ユーザーの完了判断待ちであり、`completed`へ変更していない。

- 2026-09-20: ユーザーがこの診断の完了を判断。関連変更だけをcommit・pushする。
