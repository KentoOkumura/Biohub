# exp012_group_error_readout 結果

## 仮説

固定した胚 holdout 予測を胚・画像輝度・候補密度・候補の画像境界距離・既知分裂で分けると、全体 score だけでは見えない改善と悪化を特定できる。

## 記録の参照先

- 設定、系譜、再現性方針: [`config.yaml`](config.yaml)
- 実験 status、kernel 情報、Notebook 実行時間、生成物 SHA: [`metrics.json`](metrics.json)
- 実行コマンドと途中経過: [`SESSION_NOTES.md`](SESSION_NOTES.md)

## 実行証拠

- 比較対象: exp005 の固定胚 holdout 予測を基準、exp006 の固定胚 holdout 予測を比較対象とする。exp011 の採用済み公開構成は固定予測の取得後に追加する。
- `metrics.json` の参照キー: `diagnostics.group_error_readout` と `evidence.artifacts`。現時点では実装だけで、正式な実行値は未取得。
- `SESSION_NOTES.md` の実行記録: 2026-09-13 の実装記録と検証予定を参照する。
- 予定する生成物: `artifacts/readout_v1/per_sample_readout.csv`、`group_error_summary.csv`、`paired_route_comparison.csv`、`group_error_summary.json`、`readout_manifest.json`。

## 解釈

未実行。静的 validation と test の合格は、Kaggle 上で candidate cache を含む 2 route・199 動画の条件別集計が完走した証拠ではない。診断結果はモデル精度の改善や hidden test の一般化として扱わない。

## ユーザー判断

- 判断: 未判断。
- 確認日時 / 依頼メッセージ: 未実行のため未確認。
- 理由: 条件別結果と正式な実行証拠をまだ取得していない。

## 次

Kaggle CPU diagnostic を実行し、5 条件の表、対応動画比較、失敗・NaN・有効件数、入力・出力 SHA、kernel version、Notebook 実行時間を取得する。
