# exp042_public_x138_replay セッションノート

## 目的

公開 biohub x138 V1 を追加座標補正checkpointを含めて固定し、公開test全件の2回再実行一致と費用を検証する。

## 2026-09-23 実装

- `public_x138_replay` は「設計可能・未決事項なし」だった。ユーザーの「実装してください」を実験化承認として `exp042` を作成し、候補契約・根拠・判断履歴を `requirements.md` へ移した。
- 公開Notebook SHA256 `6b655e39bbfd2d3d6c762badea69847d3f00f5b548f385cb01b07ee2600fde6d` を `assets/reference_notebook/` に保存。12個のcode cellを元の順序でJupytext sourceに生成し、V1284補正前後を記録するcellだけ推論直前へ挿入した。
- 追加重みは `biohub-v1284-head-s075/v1284_head.pt`。公開metadataのdataset_sourcesに空欄があり、Kaggle CLIで `v1284`、`biohub-v1284-head-s075` を検索してもdatasetなし。作者の公開dataset一覧にも該当なし。`pilkwang/biohub-v1284-head-s075` と `anvithpothula/biohub-v1284-head-s075` のfiles APIは403。正確な取得先・version・SHAは未確認。
- 追加確認: Kaggle公開実行の `GetKernelVersion` 生metadataに4番目のdataset attachmentとして `mountSlug=datasets/anvithpothula/biohub-v1284-head-s075`、`datasetId=12103746`、`sourceId=19822532` がある。後者はDataset Version IDで、version番号と混同しない。[取得したmetadata](../../studies/biohub_public_notebooks_20260923/x138_v1284_attachment.json)を保存した。正確なrefとversion IDを `config.yaml`へ反映した。
- `kaggle datasets files` と `kaggle datasets download` は同refに403を返し、作者の公開Notebook output一覧にも `v1284_head.pt` はない。公開Notebookの入力参照から重みの存在は分かるが、こちらのアカウントではファイル本体とSHAを取得できない。以前のSHA一致は既存3重みだけで、V1284は対象外。
- 3既存artifact、追加重み、T4 2基、動的test一覧、offline wheelsのguardを追加。追加重みの設定が空なら元sourceを実行する前に停止する。
- 座標補正前後のfloat32座標をframe単位でSHA記録し、元sourceの検出座標manifest、graph topology、決定的なrun統計、raw提出SHAとともに `replay_receipt.json` へ集約。`compare_replays.py` はinputとreceiptのSHAを再計算して2実行を照合する。
- active variant 1、config 1、fold 0、booster 0。control再学習なし。
- 実行済み: `make validate-exp EXP=exp042_public_x138_replay`、`make check-exp EXP=exp042_public_x138_replay`、`make test-exp EXP=exp042_public_x138_replay`。いずれも最終再実行で成功した。
- Kaggleへのpush・公開test実行・submissionは行っていない。追加重み未取得の停止条件が成立している。GPU quotaと12時間条件の確認はpush直前に行う。

## 次のアクション

1. 追加checkpointへのアクセス可能な共有先を確保し、ファイル本体のSHA256を計測して `config.yaml`の `data.v1284_head.sha256` を固定する。
2. `build_notebook.py` からJupytext sourceとNotebookを再生成し、同じ3検証を通す。
3. `kaggle-platform` のquota確認後、cleanな2回の公開test推論を同じT4環境で実行し、出力を比較する。
4. `kaggle-submit-check` で提出前検証を行う。実際のsubmissionはユーザーの別途承認後だけ行う。

## 2026-09-26 再実行準備

- 作者のV1284 checkpointをKaggle dataset `anvithpothula/biohub-v1284-head-s075` から取得し、ローカルファイルのSHA256 `625a0d9340f48193f2ec294fc2d81c5bb3c03087eab78ef0ae998a9c4c7da00c`を確認した。`kaggle datasets files`も `v1284_head.pt` を返した。
- `config.yaml`へSHAを固定し、`build_notebook.py` とJupytextで推論Notebookを再生成した。`make validate-exp`、`make check-exp`、`make test-exp` は成功（実験固有テスト5件）。
- 12:49 JSTにKaggle quotaを確認。GPU残り39.80/45.00時間、次の更新は2026-10-03 09:00 JST。公開testの作者実行は約20.5分だったが、今回の2実行の所要時間は未測定。T4 GPUを使用し、各実行後に実測値と残量を再確認する。
- 作者公開ログの4動画ILPは22.1、25.3、4.7、78.0秒。自前headの公開testログは17.5、24.3、7.6、66.8秒。いずれも動画ごとの1200秒上限より短い。hidden testのILP時間は提出refのログが未取得で不明。
- `make prepare-kaggle-notebooks EXP=exp042_public_x138_replay EXTRA_ARGS='--notebook inference --run-on-push'` でmetadataを生成。GPU=true、TPU=false、`NvidiaTeslaT4`、internet=false、4 dataset sourcesを確認した。
- `make push-kaggle-infer EXP=exp042_public_x138_replay` でprivate kernel `kentookumura/exp042-public-x138-replay-inference` V1をpush。Kaggle statusはRUNNING。submissionは行っていない。
- inference V1は `KernelWorkerStatus.COMPLETE`。Kaggle保存ログの最終経過時刻は約1201.3秒。公開4動画のILPは18.6、24.9、4.6、77.5秒。`run_stats.csv`の`repair_fallback`と`deadline_degraded`は全4行で0。
- `make kaggle-output KERNEL=kentookumura/exp042-public-x138-replay-inference/1 OUT=experiments/exp042_public_x138_replay/artifacts/inference_v1`で全出力を取得。receipt上の提出SHA `d52a5da2ae5cb0d1b22499f6ca51a00838a6c32ae9a1986ec756ea72e7909e03`は実ファイルおよび保存済み公開x138 V1のguard reportのSHAと一致した。公開test出力は238260行。
- V2 push前のKaggle GPU quotaは39.06/45.00時間、更新は2026-10-03 09:00 JST。V1の約20分、T4 2基の消費を踏まえてもV2実行分は十分。
- V1完了後、同じ正のNotebookと設定からpackageを再生成し、`make push-kaggle-infer EXP=exp042_public_x138_replay`で同じprivate kernel idへV2をpush。実行中。V1の実行証拠をbootstrapの`metrics.json`へ追記したため生成packageのbootstrapバイト列は更新されたが、予測用Notebook本体の元sourceと設定は変更していない。
- inference V2は `KernelWorkerStatus.COMPLETE`。Kaggle保存ログの最終経過時刻は約1269.8秒。ILPは20.0、23.8、6.6、73.5秒。`repair_fallback`・`deadline_degraded`は全4動画0。
- `make kaggle-output KERNEL=kentookumura/exp042-public-x138-replay-inference/2 OUT=experiments/exp042_public_x138_replay/artifacts/inference_v2`で全出力を取得。`compare_replays.py`で入力・補正前後座標・graph・決定的な統計・raw CSVを含む15項目が全一致。V1・V2のreceipt SHAは`6a3ac053db3139381cab4d1815746e3d6377972635329c3156d8fbc47649a902`、CSV SHAは`d52a5da2ae5cb0d1b22499f6ca51a00838a6c32ae9a1986ec756ea72e7909e03`で、公開元の保存済みCSV SHAと同じ。
- `make submit-check EXP=exp042_public_x138_replay SUBMISSION=experiments/exp042_public_x138_replay/artifacts/inference_v1/submission.csv`はPASS。238260行、重複ID0、欠損0、無限大0。Public LBを測るcompetition submissionは未実施。
- `kaggle kernels pull -m`でV2のDocker image SHA、T4、GPU有効、internet無効を確認。公開元x138 V1および採点済みexp043推論V2のDocker image SHAと一致した。Kaggle上のV2出力一覧に`submission.csv`があり、そのローカル取得ファイルも`make submit-check`でPASS。

## 2026-09-26 competition submission

- ユーザーが作者版 `kentookumura/exp042-public-x138-replay-inference/2` の `submission.csv` 提出とPublic LB測定を明示承認した。
- 提出直前にkernel V2の`COMPLETE`、Kaggle output `submission.csv`、ローカルSHA `d52a5da2ae5cb0d1b22499f6ca51a00838a6c32ae9a1986ec756ea72e7909e03`、`make submit-check` PASSを再確認した。提出前一覧にはref `56569806`はなかった。
- `make submit-code COMPETITION=biohub-cell-tracking-during-development KERNEL=kentookumura/exp042-public-x138-replay-inference KERNEL_VERSION=2 OUTPUT_FILE=submission.csv MESSAGE='exp042 public x138 replay inference v2'`を1回実行。直後の一覧で新規ref `56569806`を一意に確認した。Kaggle上の提出時刻は2026-09-26 05:07:44.530 UTC（14:07:44 JST）、statusはPENDING。
- ref `56569806`を固定し`kaggle-submit-monitor`を開始。pollingログはGit管理外の`artifacts/submission-monitor.log`。Notebook実行時間と採点所要時間は分けて記録する。
- scoringが長時間に及ぶため、Codex heartbeat automation `biohub-exp042-submission-56569806`を15分間隔で作成。refがPENDINGの間は通知せず、COMPLETEまたは失敗で記録と報告を行う。起動確認に使用した対話式monitorは初回PENDING取得後に終了し、以後はautomationで継続する。
- 2026-09-26 11:55 UTC頃のref固定CLI照会では`PENDING`、12:01:33 UTCまでの次の照会でref `56569806`が`SubmissionStatus.COMPLETE`、Public LB `0.953`、Private LB未表示と確認した。正確な採点完了時刻はこの照会間にあり、提出時刻05:07:44.530 UTCから初回の完了確認までは最大6時間53分48秒（`scoring_elapsed_minutes=414`、完了確認時刻を分単位に丸めた上限）。監視開始05:09:08 UTCからは最大6時間52分25秒。定期監視中の最後の`PENDING`確認後、ユーザーの完了連絡を受けて再照会した。この所要時間はNotebook実行時間1269.8秒とは別。
- `make record-exp EXP=exp042_public_x138_replay PUBLIC_LB=0.953`で`metrics.json`を先に更新し、次に`make record-submission EXP=exp042_public_x138_replay SUBMISSION=experiments/exp042_public_x138_replay/artifacts/inference_v2/submission.csv SUBMISSION_REF=56569806`で`SUBMISSIONS.md`のv005に記録した。ローカルCSVは公開test出力であり、採点時のhidden test出力は未取得。
- 確定値の記録と実験検証後、heartbeat automation `biohub-exp042-submission-56569806`を`PAUSED`へ変更し、定期監視を停止した。追加のcompetition submissionは行っていない。
- 2026-09-26、ユーザーがexp042の完了とcommit・pushを指示。`make record-exp EXP=exp042_public_x138_replay STATUS=completed`で実験statusを更新した。作者版の採否判断は未指定。
