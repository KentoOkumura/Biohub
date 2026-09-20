# exp023_synthetic_detector_teacher_audit セッションノート

## 目的

固定公開検出器が合成画像で出す中心・特徴に、完全合成系譜の教師を接続できるかを小規模診断する。

## 現在の作業

- 2026-09-20: ユーザーがexp022完了と本診断を承認。直接承認のexp023を作成し、実装前の契約をrequirements.mdへ記録。
- exp011の公開checkpoint2つ、exp015の8視点検出・特徴TTA、exp022の合成sourceを確認。checkpoint SHAとTemporalUNet3D source SHAは選定記録と一致。
- 合成画像は64³、native GT座標はXY方向4倍細かい。公開推論と同じ物理座標へ換算する。
- 実行予定: T4診断Notebook 1本、先頭2時系列・10窓、active variant 1、model config 0、fold学習0、booster 0、control再学習なし。submissionなし。

## コマンドログ

- `make new-exp EXP=exp023_synthetic_detector_teacher_audit` — 雛形作成。
- `uv run kaggle datasets files ...` — 公開support packとsecondaryモデルのsource・重み構造を確認。
- `uv run kaggle datasets download ... -f weights/unet_transformer/split_0/edge_predictor_best.pth` — 2 checkpointを/tmpへ取得、SHAを確認。
- `uv run kaggle datasets download ... -f weights/unet_transformer/split_0/config.json` — 2 model configが一致することを確認。

- 2026-09-20: 正規化出力をfloat32へ固定。Notebook同期、validate-exp、check-exp、test-exp（3件）、Jupytext検査を通過。Kaggle quotaはGPU残38.90/45.00h、9月26日00:00更新。T4で先頭2時系列・10窓のみ実行予定。

- 2026-09-20: `make prepare-kaggle-notebooks ... --notebook diagnostic --run-on-push --no-src`でパッケージ化。metadataはT4 GPU、internet false、公開合成Notebook出力と公開モデルdataset2件。`make push-kaggle-notebook EXP=exp023_synthetic_detector_teacher_audit NOTEBOOK=diagnostic`でKaggle kernel `kentookumura/exp023-synthetic-detector-teacher-audit` version 1をpush。初回status RUNNING。

- 2026-09-20: Kaggle version 1がCOMPLETE。`make kaggle-output KERNEL=kentookumura/exp023-synthetic-detector-teacher-audit OUT=experiments/exp023_synthetic_detector_teacher_audit/artifacts/kaggle_v1`で出力取得。`kaggle kernels pull .../1`は403だったため最新版を指定して取得し、source cell 8/8一致を確認。
- 2026-09-20: 10窓のSHA・shape・特徴有限値・教師ラベル・集計を再検証。正例69、分裂母の誤組179、通常継続組2,482、未対応28。`make record-exp ... STATUS=debug_completed ... --no-summary`でmetricsに記録。実験完了はユーザー未判断。

## 次のアクション

1. 完了記録をcommit・pushする。
2. 後続の学習の教師mask、loss、復号、評価条件は別途設計する。

## 32時系列への拡張

- 2026-09-20: ユーザーは前回提案した「より多くの合成時系列で検出後の教師量を確認し、実データの疎なGTで評価できる範囲を定める」を依頼した。exp023は未完了で同じ診断目的のため、別実験を作らず同Notebookのversion 2として先頭32時系列・160窓に拡張する。固定モデル・候補条件・特徴はversion 1と同じ。実データ側は既存exp020/021の199動画結果を再確認する。
- 予定GPU費用: version 1の2時系列289.47秒から診断処理は約77分と外挿。追加学習variant 0、model config 0、fold 0、booster 0、control再学習なし。submissionなし。Kaggle quotaをpush前に再確認する。
- 2026-09-20: Kaggle quota GPU使用6.19/45.00h、残38.81h、9月26日更新。週30hの内側で約77分の規模拡張を実行可能。version 2 packageは32時系列・160窓、T4、internet false、同じ公開2重みと合成sourceで検証。version 1の証拠をmetrics.evidence.rerunsに退避し、statusをrunningにした。
- 2026-09-20: 1回目pushはmetrics.jsonをstatus=runningに更新したためprepared package staleとしてmetadata検証で停止。直ちに再prepareしてversion 2をpush。Kaggleで32/32時系列・160/160窓を完走した。`make kaggle-output ... OUT=experiments/exp023_synthetic_detector_teacher_audit/artifacts/kaggle_v2`で取得。Kaggle pullしたsource cell 8/8一致。
- 2026-09-20: `uv run python experiments/exp023_synthetic_detector_teacher_audit/audit_outputs.py ...`で160 NPZのSHA、特徴有限値・shape、教師分類、合計、exp022の入力ファイルSHA、先頭10窓のversion 1との計数一致を確認。version 2は正例1,035、同母誤組を持つ正例母445、誤組2,252、通常継続32,087、未対応438。診断処理680.71秒、GPU forward304.39秒、peak659,688,960 bytes。GPU quota表示は6.19hから6.67hへ増加。
- 2026-09-20: exp020/021の実データ疎GTでの評価可能範囲をresultへ明記。`make record-exp ... STATUS=debug_completed ... --no-summary`でversion 2数値とSHAをmetricsに記録。完了はユーザー未判断。

- 2026-09-20: ユーザーはexp023を完了としてcommit・pushするよう明示。完了記録と実験summaryを更新し、exp023関連ファイルのみをcommitする。
- 2026-09-20: `make update-summary`でexp023を反映。生成時に同時作業中で未追跡のexp024/exp025行も入ったため、このcommitからは除外し、exp023行のみ残した。
