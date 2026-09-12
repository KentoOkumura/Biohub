# exp012_group_error_readout 結果

## 仮説

固定した胚 holdout 予測を胚・画像輝度・候補密度・候補の画像境界距離・既知分裂で分けると、全体 score だけでは見えない改善と悪化を特定できる。

## 記録の参照先

- 設定、系譜、再現性方針: [`config.yaml`](config.yaml)
- 実験 status、kernel 情報、Notebook 実行時間、生成物 SHA: [`metrics.json`](metrics.json)
- 実行コマンドと途中経過: [`SESSION_NOTES.md`](SESSION_NOTES.md)

## 実行証拠

- 比較対象: exp005 の固定胚 holdout 予測を基準、exp006 の固定胚 holdout 予測を比較対象とする。exp011 の採用済み公開構成は固定予測の取得後に追加する。
- Kaggle Notebook: private kernel `kentookumura/exp012-group-error-readout-diagnostic` version 1。CPU、internet 無効、実行時間 717.057 秒。
- coverage: 2 route x 199 動画 = 398 行、条件別集計 34 行、対応比較 17 行。failure 0、NaN 0、boundary feature は全件取得済み。
- 生成物: [`per_sample_readout.csv`](artifacts/readout_v1/per_sample_readout.csv)、[`group_error_summary.csv`](artifacts/readout_v1/group_error_summary.csv)、[`paired_route_comparison.csv`](artifacts/readout_v1/paired_route_comparison.csv)、[`group_error_summary.json`](artifacts/readout_v1/group_error_summary.json)、[`readout_manifest.json`](artifacts/readout_v1/readout_manifest.json)。入力と各生成物の SHA-256 は `metrics.json` と manifest に記録した。

## 主な結果

- 全体 score は exp005 の 0.124906 から exp006 の 0.553379 へ改善し、差は +0.428473。199 動画中 183 件で改善、16 件で悪化した。
- 胚別では 44b6 が +0.079779（55 改善、16 悪化）、6bba が +0.500727（128 件すべて改善）。悪化 16 件はすべて 44b6 に含まれる。
- 画像輝度では Q1_low が +0.549952（84 件すべて改善）に対し、Q4_high は +0.121391（43 改善、12 悪化）。
- 候補密度では Q1_low が +0.546934（97 件すべて改善）に対し、Q4_high は +0.129054（50 改善、12 悪化）。
- 画像境界距離の改善幅は Q3 の +0.097161 から Q4_high の +0.475085 まで非単調で、境界距離だけでは悪化群を分離できない。
- known division の有無では両方が改善したが、exp006 の division Jaccard は全体で 0.001189（TP 1、FP 690、FN 150）に留まり、division prediction は未解決である。

## 解釈

exp006 の改善は全動画で一様ではなく、6bba、低輝度、低候補密度で特に大きい。残る悪化は 44b6、高輝度、高候補密度に重なるが、これらの条件は相互に重複するため因果関係とは解釈しない。

この診断は保存済み holdout 予測の再集計であり、予測、モデル、復号を変更していない。結果だけから router、post-process、hidden test での改善を採用しない。

## ユーザー判断

- 判断: `completed`。
- 確認日時 / 依頼メッセージ: 2026-09-13「exp012は閉じてください」。
- 理由: coverage 100%、failure 0、NaN 0で契約した 5 条件と対応動画比較を完走し、結果を今後の実験へ引き継げるため。モデル変更や提出候補の採用判断は別実験とする。

## 次

[`oracle_stage_limits`](../../backlog/oracle_stage_limits.md)へ、悪化 16 件を exp012 の条件別に段階分解する診断を引き継ぐ。exp012 自体の追加実行は予定しない。
