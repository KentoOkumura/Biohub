# exp013_public_notebook_replay セッションノート

## 目的

採用した公開Notebookをcanonical artifactで2回full inferenceし、中間候補・graph・最終`submission.csv`の再実行一致を検査する。

## 現在の作業

- 作業内容: 再実行一致を確認したprivate Kaggle Notebook version 2をcode submission ref `56199738`として提出し、Kaggleの採点待ち。
- ブロック要因: なし
- 次: ref `56199738`の採点完了後にPublic LBと採点所要時間を記録し、完了・採否をユーザーに判断してもらう。

## コマンドログ

実行したコマンドを時系列で記録します。未実行のコマンドは予定として明記します。

### 予定

共通確認を先に行う。`task`がない環境では同名の`make` targetを使う。

```bash
task validate-exp EXP=exp013_public_notebook_replay
task check-exp EXP=exp013_public_notebook_replay
task test-exp EXP=exp013_public_notebook_replay
```

次のうち、実験契約に必要なnotebookだけを予定へ残します。学習を伴わないauditやdiagnosticではtrain、提出を目的としない実験ではinferenceを機械的に実行しません。

本実験で必要なinference:

```bash
task prepare-kaggle-notebooks EXP=exp013_public_notebook_replay EXTRA_ARGS="--notebook inference --run-on-push"
task push-kaggle-infer EXP=exp013_public_notebook_replay
task kaggle-logs KERNEL=<generated-kernel-id>
```

prepare後に生成された`kernel-metadata.json`からkernel idを取得して`KERNEL`へ指定する。id末尾とtitle由来slugが一致し、50文字以内であることを確認する。自動生成では上限や衝突を解消できない場合だけ、`kaggle-platform`の規則に従って意味のある短縮id/titleを明示する。placeholderのまま実行しない。

## 変更点

- 公開Notebook取得版を`assets/reference_notebook/`へ保存し、content SHAを固定した。
- Jupytext percent sourceを正の編集入口とし、7つの役割別sectionへ分けたinference Notebookを生成した。
- 作者mirror pathをexp011で採用したPilkwangのcanonical dataset 3件へ置き換えた。
- 予測前にT4 2基、3 artifact manifest、全offline wheel、test Zarr構造を検査して`replay_input_manifest.json`へ保存する。
- 予測後にcandidate coordinate、retention guard、graph topology、決定的なrun統計、`submission.csv`をhashして`replay_receipt.json`へ保存する。
- 2 outputの必須fieldとraw `submission.csv`を検査する`compare_replays.py`を追加した。
- active variant 1、model/config 1、fold 0、booster 0。既存controlの再学習は行わない。

### 2026-09-13 実装

- Kaggleから`reyhanksatria/biohub-cell-tracking-0-947-lb`を再取得し、候補で固定したSHAと一致することを確認した。
- `make new-exp EXP=exp013_public_notebook_replay SOURCE=experiments/exp011_public_detector_selection`で実験を作成した。
- 公開source本文をJupytextへ変換し、canonical mountとreplay証拠cellを追加した。
- Jupytext round-trip、Ruff F821、実験固有test 7件を実行し、すべて成功した。
- 予測処理は、replay追加blockを除き、canonical pathを参照sourceへ戻したASTが取得版と一致することをtestで確認した。

### 2026-09-13 10:12 JST 1回目push前確認

- `make validate-exp EXP=exp013_public_notebook_replay`: strict validation成功。
- `make check-exp EXP=exp013_public_notebook_replay`: Ruff check/format成功。exact public source由来のinference `.py`は意味同一性testを正とし、実験ローカルのRuff設定で自動整形対象から除外した。
- `make test-exp EXP=exp013_public_notebook_replay`: 7件成功。
- `make check-strategy-docs`: 成功。
- `make prepare-kaggle-notebooks EXP=exp013_public_notebook_replay EXTRA_ARGS="--notebook inference --run-on-push"`: 成功。
- 生成metadata: private、run-on-push、GPU有効、TPU無効、internet無効、`NvidiaTeslaT4`、canonical dataset 3件、competition source 1件。
- Kaggle quota: GPU 30.00h中18.00h使用、12.00h残り、次回更新は2026-09-19 00:00:00。2回のfull inferenceを開始できる残量と判断した。ただし1回目終了後に実績時間と残量を再確認する。
- kernel id: `kentookumura/exp013-public-notebook-replay-inference`。slugは50文字以内でtitleと整合する。
- 判断: 1回目をpushする。Kaggle submissionは行わない。

### 2026-09-13 1回目push未実施

- `make push-kaggle-infer EXP=exp013_public_notebook_replay`は、外部送信保護により実行前に拒否された。
- KaggleにはNotebook、source、artifactを送信しておらず、kernel versionは作成されていない。
- 再開条件: ユーザーが`kentookumura/exp013-public-notebook-replay-inference`へのprivate Notebook pushと実行を明示的に承認する。
- 実験statusは、static implementationと実行package準備までを表す`scaffold_completed`とする。2 run比較が未実施のため、`completed`、`usable`、deterministic anchorとは記録しない。

### 2026-09-13 12:48 JST 実行承認と再確認

- ユーザーが「実行してください」と明示し、`kentookumura/exp013-public-notebook-replay-inference`へのprivate Notebook pushと実行を承認した。
- CLI認証確認はOAuthとlegacy credentialで成功した。API tokenはないが、このCLI実行には不要である。
- push直前にstrict validation、Ruff check/format、実験固有test 7件を再実行し、すべて成功した。
- 正のNotebookからpackageを再生成した。metadataはprivate、run-on-push、GPU有効、TPU無効、internet無効、`NvidiaTeslaT4`、canonical dataset 3件、competition source 1件である。
- Kaggle quota: GPU 30.00h中20.60h使用、9.40h残り、次回更新は2026-09-19 00:00:00。想定25〜40分の2 runに足りると判断した。
- active variant 1、model/config 1、fold 0、booster 0。学習と既存controlの再学習は行わない。
- 判断: 同じcanonical kernel idで1回目をpushする。Kaggle submissionは行わない。

### 2026-09-13 13:15 JST version 1成功・version 2 push前確認

- version 1を`kentookumura/exp013-public-notebook-replay-inference`へpushした。Kaggle kernel idは`134141368`。
- push後にpullし、private Notebookの存在、docker image SHA `37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461`、`NvidiaTeslaT4`を確認した。
- full inferenceは公開test 4動画・各100 frameで成功した。推論部分は559.07967877388秒。
- outputを`/tmp/kaggle-output/exp013_public_notebook_replay/run1`へ取得した。`submission.csv`は241,282行、SHA-256は`0319ba6d8e864335d3573f6b1a6227c546f17e9247a0c2858fa09b6c2422db3f`。
- `replay_receipt.json`のraw SHA-256は`5054345fa5c3d6515eaffe37b4eb581b626fc4d0af5e276d4cc22024587442d2`、receipt内のcontent SHAは`5a9f8a400685d0d1f0ea010365f1dfa96c9115777cb344bfcfc49b6e374e5611`。
- input manifest content SHAは`e1767a58f7eee45178e6919640f08d3bcf5165f6b4ea3aa9480a846301441b2f`、candidate coordinate content SHAは`303de241f08ae1d3456c84f344e074146844c868d7b0ad66a96d20745602dcc4`、graph topology SHAは`5289d4bcfa24be048ace1a20d9793b4d77e9664bfd53a517c9d808e3b3b774aa`。
- `compare_replays.py`でrun1を自身と比較し、manifestとraw submissionを含む自己整合性検査が成功した。
- version 1後のGPU quotaは30.00h中21.41h使用、8.59h残り。開始前からの差は0.81hで、version 2に足りると判断した。
- 同じ生成packageとcanonical kernel idをversion 2へpushする。Kaggle submissionは行わない。

### 2026-09-13 version 2成功・比較完了

- version 1と同じ生成packageを同じkernel idへpushし、version 2も成功した。live logのSSE接続は途中で切断されたが、再接続後に完了を確認し、outputを取得できた。
- full inferenceは公開test 4動画・各100 frameで成功した。予測部分は546.079577922821秒。
- outputを`/tmp/kaggle-output/exp013_public_notebook_replay/run2`へ取得した。`submission.csv`は241,282行、SHA-256はversion 1と同じ`0319ba6d8e864335d3573f6b1a6227c546f17e9247a0c2858fa09b6c2422db3f`。
- `replay_receipt.json`のraw SHA-256は`527bb9a4008299ad61fd8a7505aea1fe4bf95004394af5fc9cdced5e69601669`、receipt内のcontent SHAは`a94561b4d75ac27af41ebbdadf9b66007cdf23f541d8ee79a0b4eeaef311fe17`。
- version 1と2はinput manifest、wheel manifest、3 checkpoint、source manifest、candidate coordinate、graph topology、retention guard、決定的なrun統計、公開test一覧、submission行数とraw SHAを含む13比較項目がすべて一致した。
- receipt全体のSHA差は、実測の予測時間をreceiptへ含めているためである。比較helperはこの非決定的な計測値を一致条件から除外しており、`mismatches`は空、`byte_identical_to_reference`は`true`だった。
- 比較reportを`experiments/exp013_public_notebook_replay/artifacts/replay_comparison.json`へ保存した。SHA-256は`781da5cb37fd8921bc4f63208190bec0612142dfc91871b912180b4bcf2a3304`。
- version 2後のGPU quotaは30.00h中22.64h使用、7.36h残り、次回更新は2026-09-19 00:00:00。version 1後との差は1.23hだが、quota反映の遅延を含み得るためNotebookの正確な実行時間として扱わない。
- Kaggle CLIからNotebook全体の正確な実行秒数は取得できなかった。pushからoutput取得までの観測区間はversion 1が約25分、version 2が約37分で、予測部分の実測は約9分19秒と約9分06秒だった。
- `metrics.json`のstatusを、実行成功済み・ユーザー判断待ちを表す`debug_completed`へ更新した。Kaggle submissionは行っておらず、CV・Public LB・Private LBは未取得である。

### 2026-09-13 14:20 JST code submission

- ユーザーが「提出までしてください」と明示し、version 2のcode submissionを承認した。
- `make submit-check EXP=exp013_public_notebook_replay SUBMISSION=/tmp/kaggle-output/exp013_public_notebook_replay/run2/submission.csv`はPASS。241,282行、重複ID 0、欠損 0、infinite value 0で、列とID順は`sample_submission.csv`と一致した。
- Notebook packageの補助検査もFAIL 0、WARN 0。metadataはprivate、T4、internet無効、competition source設定済みである。
- sourceは実行時のtest `.zarr`を動的に列挙する。公開test固有のdataset ID、件数、行数、予測値による分岐はなく、保存済み公開test予測をhidden testへ流用しない。
- 提出前一覧のrefは`56182729`と`56153451`だけだった。`make submit-code COMPETITION=biohub-cell-tracking-during-development KERNEL=kentookumura/exp013-public-notebook-replay-inference KERNEL_VERSION=2 OUTPUT_FILE=submission.csv MESSAGE='exp013 public notebook replay v2'`を実行した。
- submission refは`56199738`、提出日時は2026-09-13 05:20:43.280000 UTC、messageは`exp013 public notebook replay v2`。提出直後は`pending`で、Kaggle CLIは当日の残り提出回数を4回と表示した。
- `metrics.json`の`evidence.submission`と`SUBMISSIONS.md`のv003へpendingの最終スナップショットを記録した。採点監視ログは一時生成物としてGitへ保存しない。

設定と再現性方針は`config.yaml`、kernel情報、Kaggle Notebook実行時間、生成物SHA、rerun比較、実験statusは`metrics.json`へ記録する。このファイルには、それらを得たコマンド、時刻、途中経過、失敗と修正を時系列で残す。提出した場合はsubmission ref・提出日時・submission scoring status・監視開始からscore確定までの所要時間を、詳細な時系列の正として記録し、Notebook実行時間と混同しない。`SUBMISSIONS.md`には横断比較に必要な最終スナップショットだけを`record-submission`で記録し、状態遷移、観測時刻、実行コマンドを転記しない。

## 次のアクション

1. submission ref `56199738`を監視し、score確定後にPublic LBと採点所要時間を記録する。
2. 第1段階と提出結果の証拠を提示し、完了・採否をユーザーへ判断してもらう。
3. 完了判断後、この実験に関係する変更だけをcommit・pushする。
4. 採用された場合は固定予測と実測費用を`exact_window_cache`等の後続比較へ渡す。
