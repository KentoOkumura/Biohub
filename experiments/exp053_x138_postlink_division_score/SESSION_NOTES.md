# exp053_x138_postlink_division_score セッションノート

## 目的

exp043の再接続・欠損補完後に、4時点の接続・座標特徴で第2娘候補を採点して追加し、公開LBで効果を確認する。

## 現在の作業

- 作業内容: ユーザーの完了・採用判断を記録し、exp053に関係する変更だけをcommit・push
- ブロック要因: なし
- 次: 対象変更の差分と検証を確認してmainへcommit・pushする

## コマンドログ

実行したコマンドを時系列で記録します。未実行のコマンドは予定として明記します。

### 予定

共通確認を先に行います。

```bash
task validate-exp EXP=exp053_x138_postlink_division_score
task check-exp EXP=exp053_x138_postlink_division_score
task test-exp EXP=exp053_x138_postlink_division_score
```

次のうち、実験契約に必要なnotebookだけを予定へ残します。学習を伴わないauditやdiagnosticではtrain、提出を目的としない実験ではinferenceを機械的に実行しません。

trainが必要な場合:

```bash
task prepare-kaggle-notebooks EXP=exp053_x138_postlink_division_score EXTRA_ARGS="--notebook train --run-on-push"
task push-kaggle-train EXP=exp053_x138_postlink_division_score
task kaggle-logs KERNEL=<generated-kernel-id>
```

inferenceが必要な場合:

```bash
task prepare-kaggle-notebooks EXP=exp053_x138_postlink_division_score EXTRA_ARGS="--notebook inference --run-on-push"
task push-kaggle-infer EXP=exp053_x138_postlink_division_score
task kaggle-logs KERNEL=<generated-kernel-id>
```

prepare後に生成された`kernel-metadata.json`からkernel idを取得して`KERNEL`へ指定する。id末尾とtitle由来slugが一致し、50文字以内であることを確認する。自動生成では上限や衝突を解消できない場合だけ、`kaggle-platform`の規則に従って意味のある短縮id/titleを明示する。placeholderのまま実行しない。

## 変更点

- exp043の最終再接続後、学習済み4時点採点器で合法な第2娘だけを追加する。その他の推論・後処理を継承する。

設定と再現性方針は`config.yaml`、kernel情報、Kaggle Notebook実行時間、生成物SHA、rerun比較、実験statusは`metrics.json`へ記録する。このファイルには、それらを得たコマンド、時刻、途中経過、失敗と修正を時系列で残す。提出した場合はsubmission ref・提出日時・submission scoring status・監視開始からscore確定までの所要時間を、詳細な時系列の正として記録し、Notebook実行時間と混同しない。`SUBMISSIONS.md`には横断比較に必要な最終スナップショットだけを`record-submission`で記録し、状態遷移、観測時刻、実行コマンドを転記しない。

## 次のアクション

1. exp053固有の変更だけをstageしてcommitする。
2. mainをpushし、commitとpushの結果を報告する。

## 2026-09-29 04:25 UTC: 実装とCPU学習準備

- ユーザーが承認した軽量案をexp053へ移した。4時点と前後遮断対照の2 variant、各1設定、foldの重み学習なし、booster 0。exp043の画像モデル・座標head・trackerの再学習は0。
- GEFFの既知2娘と既知の別親だけを教師にし、外側各胚21～30位、内側各胚31～35位を固定。Kaggle CPU学習Notebookで件数と閾値を確定する。公開test・Public LBから閾値を選ばない。
- push対象はtrain CPU、internet無効。support packのオフラインwheelだけを追加。GPU週枠はこのtrain pushで消費しない。推論GPU残枠と12時間見込みは推論push直前に再確認する。
- make validate-exp、make check-exp、make test-expを実行し、新規4時点特徴と合法性のテスト2件が通った。backlogの移行後のcheck-strategy-docsはHYP-20260910-02、09、11の既存不整合で失敗し、今回移行したHYP-20260910-03は指摘されていない。

## 2026-09-29 04:32 UTC: Kaggle特徴生成とColab CLI

- Kaggle CPU train kernel kentookumura/exp053-x138-postlink-division-score-train v1は、competition inputを一階層だけで探したためFileNotFoundErrorで停止。親exp043/050と同じ二つのmount候補に修正し、v2をpushして実行中。
- ユーザーはDriveなしColab CLIを選択。今回の軽量案では画像特徴cacheは使わず、Kaggle GEFFから作る小型の接続・座標特徴表をColabへ渡す。Colab CPU session exp053-division-scoreを起動し、Python 3.13.15、NumPy 2.1.3、scikit-learn 1.6.1の実行を確認した。
- ユーザーは接続・座標特徴に絞った提出を承認。元の全動画ILPと画像特徴を使う効果はこの実験から判断しない。
- 推論の後処理はユーザー回答によりexp043固定。既存の時間切迫時には分裂追加を停止するフラグも新採点器で維持した。

## 2026-09-29 04:40 UTC: Kaggle CPU学習完了、Colab CLI学習と推論push判定

- Kaggle CPU train v2は518.35秒で完了。学習候補表は合計2345行（fit 1649、inner 378、outer 318）、SHA256は`b985931d768a5554e3918609b94fd612e5cc47b7e57eaaeaf06b6d88473ed345`。胚別外側候補診断は`metrics.json`の`evidence.candidate_diagnostics`を正とする。
- Colab CLI CPU session `exp053-division-score`に90KBの候補表を転送し、SHA一致を確認。4時点と前後遮断の2 variantを学習。innerのみで閾値を固定し、Colab重みSHA256 `b9774daf16861f2ee36f4e82b3b7e3af7efe93de0144c1a5bef80e3d822ef6cf`を回収・推論sourceへ固定。成功マーカー`EXP053_COLAB_TRAIN_COMPLETE`とZIP内のreceiptを確認してsessionを停止した。
- Kaggle推論packageはGPU T4、TPU無効、internet無効、固定asset同梱。`make validate-exp`、`make check-exp`、`make test-exp`、Jupytext roundtrip検査に合格。推論push直前の`uv run kaggle quota --format json`はGPU残6.90h（refresh 2026-10-03 00:00 UTC）。公開test推論はexp043の約9分を基準に残枠内と判断し、pushする。hidden採点実行時間は提出後に監視する。

## 2026-09-29 04:42 UTC: Kaggle推論v1 push

- `make push-kaggle-infer EXP=exp053_x138_postlink_division_score`でmetadata検証後、`kentookumura/exp053-x138-postlink-division-score-inference` v1をpush。GPU T4、internet無効。live SSE logsはbootstrapとoffline wheel installまで進行。

## 2026-09-29 05:02 UTC: 公式Kaggle提出

- Kaggle推論Notebook `kentookumura/exp053-x138-postlink-division-score-inference` v1は約988秒で完了。公開test4動画を動的に検出し、`submission.csv` 225842行を生成。Kaggle側metadataはT4 GPU、TPU無効、internet無効を再確認した。
- `run_stats.csv`で4動画すべて`deadline_degraded=0`、4時点採点による追加分裂は31、2、3、19件、計55件。後処理の最終出力でも各動画の分裂源数が同じだった。新規分裂候補の母・娘の合法性はNotebook内部の最終検査を通過した。
- `make submit-check EXP=exp053_x138_postlink_division_score SUBMISSION=/tmp/kaggle-output/exp053_x138_postlink_division_score/inference_v1/submission.csv`はPASS。sample submissionのschema・ID、重複、欠損、無限値を確認。公開testのCSV SHAとKaggle実行ログのSHAが一致した。
- ユーザーの明示依頼に基づき、`make submit-code COMPETITION=biohub-cell-tracking-during-development KERNEL=kentookumura/exp053-x138-postlink-division-score-inference KERNEL_VERSION=1 OUTPUT_FILE=submission.csv MESSAGE='exp053 postlink four-frame division score'`を実行。提出一覧のdescription・日時・先行refとの差分から、新規refは`56663354`、提出時刻は`2026-09-29T05:02:32.023000` UTC、初期scoring statusはpendingと一意に確認した。
- `kaggle-submit-monitor`のref固定監視を開始。pollingログは`artifacts/submission-monitor.log`（Git管理外）。Public LB確定後にexp043 0.950と比較し、record-expの後にrecord-submissionを行う。実験完了・採否はユーザー判断を待つ。

## 2026-09-29 12:04 UTC: ref 56663354のscoring完了

- Kaggle CLIで対象refの`SubmissionStatus.COMPLETE`、Public LB 0.953、Private LB未公開を確認。exp043 0.950より+0.003、既存exp042 0.953と同点。
- 提出時刻は`2026-09-29T05:02:32.023000` UTC。監視開始は05:04:17 UTC、最後のpending確認は10:40:07 UTC、最初のcomplete確認は12:04:03 UTC。監視開始から最初のcomplete確認まで420分。10:40～12:04に監視観測の空白があるため、実際の採点完了時刻はその範囲内で、420分を厳密なscoring実行時間とは扱わない。
- `make record-exp ... PUBLIC_LB=0.953`を先に実行し、`make record-submission ... SUBMISSION_REF=56663354`で同じrefの提出履歴を記録。候補の胚別Average Precisionは混在し、Public LBはパイプライン全体の比較。実験完了・採否はユーザー判断前に確定させない。

## 2026-09-29: ユーザーが完了・採用を判断

- ユーザーは「完了および採用としてください。commitとpushしてください」と明示した。`make record-exp EXP=exp053_x138_postlink_division_score STATUS=completed`で単一statusを更新。採用範囲と残る限界は`result.md`へ記録した。
- この実験に関係する差分だけを確認してmainへcommit・pushする。ほかの実験や調査の未コミット変更は含めない。
