# exp052_x138_relink_candidate_scores セッションノート

## 目的

exp043と同じ固定tracker・ILP解のもとで、ILP前に保存した未選択辺の得点を再接続へ渡す効果を調べる。

## 進捗

- 2026-09-27: ユーザーが設計済み候補の実装を依頼。`make new-exp EXP=exp052_x138_relink_candidate_scores SOURCE=experiments/exp047_x138_edge_candidates`で実験を作成。実験の親はexp043とし、exp047はNotebook実行基盤に限って再利用する。
- 予定: 学習variant 0、評価arm 2、fold 0、booster 0。control再学習なし。元の`p > 0.48`のILPと固定trackerを維持する。
- 予定: 内部の各胚11～20位でcacheと対照の再現・割当変更を確認し、進行条件と費用条件を満たす場合だけ各胚21～30位をKaggleで全graph評価する。

- 2026-09-27: 元cache得点照合、再追加node固定の2条件再接続と段階別診断を実装。make validate-exp、make check-exp、make test-exp（3件）、Jupytext整合、make check-strategy-docsが通過。候補詳細をrequirements.mdへ移し、未着手backlogから削除。
- 2026-09-27: 診断Notebookをprepare。GPU=T4、TPU無効、internet無効。Kaggle quotaはGPU残15.71時間、refreshは2026-10-03 00:00:00、確認時の週上限45時間。exp047の20動画実測2.74時間を参考に、内部診断と条件成立時の新規評価の合計を残時間内と判断。control再学習はなく、固定モデルの推論だけを行う。

- 2026-09-27: make push-kaggle-notebookで診断kernel kentookumura/exp052-x138-relink-candidate-scores-diagnostic version 1を起動。KaggleからpullしたmetadataでもT4、TPU無効、internet無効を確認。live logsでは固定重み・依存パッケージの検証を通過し、20動画の推論中。
- 2026-09-27: 診断v1の実行中に監査名を訂正。v1のbelow_cache_threshold_pairs_consideredは保存閾値以下と採点対象外を区別できない件数であり、正しくは得点欠損理由未判別。診断sourceと新規評価sourceでmissing_score_reason_unknown_on_initial_nodesへ変更し、次回package生成時に反映する。診断v1のKaggle実行sourceは不変。
- 2026-09-27: 新規評価sourceに各胚1動画の先行推論費用gateを追加。診断の進行条件通過と実測費用判定後だけ残り18動画へ進む。

## 次のアクション

[結果とユーザー判断](result.md)を参照する。提出パイプラインは採用・完了と判断されており、未実施の学習動画評価を行うかは別途判断する。

## 2026-09-27 CPU試行前の予定（履歴）

1. CPU専用の各胚1動画の所要時間試行をKaggleで実行する。
2. 試行の結果から元cache保存範囲、時間、メモリを確認し、内部診断と新規20動画評価への進行条件を判断する。

## 実行と判断の履歴

- 2026-09-27: ユーザーからGPU不使用の訂正。exp052診断v1はT4 GPUでRUNNINGだったため、このタスクで作ったKaggle kernelを削除。削除後のstatusはアクセス不可、一覧はNot found。GPU残時間は起動前15.71時間、停止後15.35時間で、再確認でも15.35時間。誤って約0.36時間を消費。CPUだけでの再設計が定まるまでKaggle pushは行わない。

- 2026-09-27: ユーザーが「CPUで少数動画を試す」を選択。各胚21位の1動画ずつ、計2動画をCPUのみで測るNotebookを追加。1動画の推論上限1800秒、20動画換算に安全係数1.5を掛けて10時間以上なら2本目へ進まない。GPUを使わないKaggle package metadataをpush前に検証する。

- 2026-09-27: CPU試行packageをstrict prepare。metadataはenable_gpu=false、enable_tpu=false、enable_internet=false、run_on_push=trueでvalidator通過。make validate-exp、make check-exp、make test-exp（3件）、Jupytext、F821検査が通過。Kaggle pushは「Maximum batch CPU session count of 5 reached」で拒否され、試行kernelは未作成・未実行。CLIのexit codeは0でも応答文のerrorを確認した。既存の他実験CPU Notebook 5件がRUNNINGで、所有していない作業を止めずに枠を待つ。

- 2026-09-29: 設計値を再確認。CPU試行の安全係数1.5と新規20動画の推論時間上限36000秒を`config.yaml`へ移し、Notebookが設定を読むようにした。各胚21位の試行では正解ラベルを参照せず、どちらか1本でも実測時間からの予測が上限以上なら拡大を止める。GPU禁止、内部診断と新規評価の進行条件は維持。`make validate-exp`、`make check-exp`、`make test-exp`（3件）、Jupytext通常roundtripと`.py`/`.ipynb`のsource一致を確認。Kaggle上のCPU所要時間と公式scoreは未計測。

- 2026-09-29: 締切前のLB確認に向け、ユーザーが提出用Notebook実装、公開test完走、提出形式チェックを優先する方針を依頼。`submission` Notebookを既存の学習動画評価Notebookとは別に作成し、実行時のtest動画全件、CPUのみ、固定trackerと元ILP graph、保存済み未選択辺得点の再接続を使用。正解GEFFと固定のtest件数には依存しない。`make validate-exp`、`make check-exp`、`make test-exp`（3件）、Jupytext source一致、packageのCPU/internet無効とmetadata事前チェックを確認。Kaggle `kentookumura/exp052-x138-relink-candidate-scores-submission` version 1をpushし、公開testで実行中。実際のcompetition submissionは未実行。

- 2026-09-29: ユーザーが提出用推論で画像モデル・trackerの予測にはGPUが必要と訂正。前項のCPU設定は学習動画のCPU試行条件を提出用推論へ誤適用したもの。提出用NotebookをT4 GPU、internet無効へ修正し、CPU用kernelを上書きしない新規slug `kentookumura/exp052-relink-scores-gpu-submit` version 1をpush。GPU quotaはpush前7.91時間残。`make validate-exp`、`make check-exp`、`make test-exp`（3件）、提出前metadata checkerは通過。GPU公開testの結果とCSV検証は実行中。既存CPU kernelの停止・削除は自動承認審査で拒否されたため実施せず。competition submissionも未実施。

- 2026-09-29: GPU版v1実行中にローカルsourceのreceipt `resource` が`cpu`のままだったことを発見し、`gpu`へ修正。v1の提出CSVや推論処理への影響はなく、v1のreceiptだけが誤表示になる。GPU kernelの実行を中断せず、結果は実行時metadataのT4設定で確認する。CPU kernelのread-only statusは`CANCEL_ACKNOWLEDGED`。

- 2026-09-29: GPU公開test Notebook `kentookumura/exp052-relink-scores-gpu-submit` version 1が`COMPLETE`。live logで`Tesla T4`と`device=cuda`を確認し、4動画すべての予測、対照・変更側後処理、`submission.csv`生成が完走。`kaggle kernels output`の`--file-pattern`でCSVとreceiptだけを`/tmp/kaggle-output/exp052_x138_relink_candidate_scores/submission_gpu_v1/`へ取得。Notebook内のSHAとローカルSHAが一致。`make submit-check EXP=exp052_x138_relink_candidate_scores SUBMISSION=/tmp/kaggle-output/exp052_x138_relink_candidate_scores/submission_gpu_v1/submission.csv`はPASS。kernel version、GPU、実行秒数、CSV SHA、行数、得点参照件数を`metrics.json`に記録し、実験statusは`running`とした。v1のreceiptには前項のCPU誤ラベルが残るが、Kaggle metadataと実行ログはGPUを示す。実際のcompetition submissionとLB評価は未実施。

- 2026-09-29 04:23:51 UTC: ユーザーがcompetition submissionを明示承認。`make submit-code COMPETITION=biohub-cell-tracking-during-development KERNEL=kentookumura/exp052-relink-scores-gpu-submit KERNEL_VERSION=1 OUTPUT_FILE=submission.csv MESSAGE='exp052 relink candidate scores GPU v1'`を実行。提出前後の一覧比較で今回のrefを`56662528`と特定。初回状態は`PENDING`、public/private LBは未確定。出力CSVは提出前にローカル検証済み。

- 2026-09-29 04:25 UTC: `kaggle-submit-monitor`でref `56662528`を固定し、5分間隔の監視を開始。一時ログは`experiments/exp052_x138_relink_candidate_scores/artifacts/submission-monitor.log`、runnerログは`/tmp/submission_exp052_x138_relink_candidate_scores.runner.log`。最初の`nohup ... &`起動では監視プロセスが残らなかったため、`--once`でPENDINGを確認後、`setsid -f`で起動し、process一覧と監視ログで継続稼働を確認。ログはGitに保存しない。

- 2026-09-29 04:41 UTC: 監視ログでref `56662528`が`COMPLETE`に変わったがPublic/Private LBは空欄。Kaggle `GetSubmission`の`error_description`は「Your notebook hit an unhandled error while rerunning your code. Note that the hidden dataset can be larger/smaller/different than the public dataset」。hidden tracebackは公開されない。公開test 4動画は完走したが、公式資料ではhidden testはtrain 199動画と同程度の規模。提出用sourceの`prediction_budget_gate_failed`は1動画目の実測秒数×全hidden動画数×安全係数が36000秒以上なら例外を出す。公開test最速222秒を199動画へ外挿しても約66393秒となるため、このgateが直接の失敗原因である可能性が高い。ただしhidden実行の例外行は取得できていないので断定しない。
- 2026-09-29: hidden対応修正として、提出用sourceだけを変更。予測を最大2基のGPUで動画ごとに並列実行し、対照後処理は候補側へ固定する再追加nodeの決定で止め、候補側の最終graphは元の処理を維持する。予測時間見積もりと11.5時間閾値は警告へ変更し、cache・中間graphは処理後に削除する。公開testでversion 1のCSV SHAと一致することを再提出前の受け入れ条件とする。
- 2026-09-29: ユーザーの訂正により、提出用Notebookの全動画推論時間の外挿、10時間閾値、11.5時間警告を設定・sourceから削除。各動画の実測所要時間は記録し、1800秒の動画単位上限とKaggleの12時間制限のもとで全動画を処理する。上記の「警告へ変更」はversion 2時点の履歴であり、最終版の契約ではない。
- 2026-09-29 05:19 UTC: ユーザーがversion 2公開テストを手動停止。Kaggle statusは`CANCEL_ACKNOWLEDGED`。停止前に2 GPUで最初の2動画の予測・後処理が完了し、`new_scores_considered`は8120と2576。version 2のCSVは未生成。version 3 push前のGPU残時間は6.40h、refreshは2026-10-03 00:00 UTC。T4 GPU、TPU無効、internet無効のmetadataを確認し、時間推計を削除したversion 3で公開testを再実行する。
- 2026-09-29: 同じkernel slugをKaggleからpullして存在を確認した後、strict metadata checkerを通してversion 3をpush。Kaggle側の公開testを実行中。version 2はユーザーが停止したためCSV一致の判定には使用しない。
- 2026-09-29 05:49 UTC: version 3公開testが`COMPLETE`。T4 GPU、internet無効。公開4動画すべての予測とgraph後処理を完了し、`submission.csv` 225836行、SHA `480a7f4f1d7066dae1a6cccccf22114834512c3f4a5b2556e80a47398ffd13be`を生成。version 1とCSVがバイト単位で一致。receiptのNotebook実行時間は1067.33秒、resourceは`gpu`。ローカル取得したCSVに`make submit-check`を実行しPASS、重複ID・欠損・無限大は0。version 3のreceiptは`/tmp/kaggle-output/exp052_x138_relink_candidate_scores/submission_gpu_v3/exp052_submission/submission_receipt.json`。ユーザーから提出を待つよう指示があったため、version 3のcompetition submissionは行わず保留。
- 2026-09-29 12:23:04 UTC: ユーザーがversion 3の提出を明示指示。Kaggle Notebook status `COMPLETE`、Kaggle outputの`submission.csv`、ローカルSHAと提出前履歴を確認し、`make submit-code COMPETITION=biohub-cell-tracking-during-development KERNEL=kentookumura/exp052-relink-scores-gpu-submit KERNEL_VERSION=3 OUTPUT_FILE=submission.csv MESSAGE='exp052 relink candidate scores GPU v3 no time projection'`を実行。新しいsubmission refは`56675101`、初回状態は`PENDING`、Public/Private LBは未確定。CLIには「0 submissions remaining today」と表示され、今回のrefは提出後一覧で確認した。
- 2026-09-29 12:24 UTC: `kaggle-submit-monitor`でref `56675101`を固定し、5分間隔の監視を開始。一時ログは`experiments/exp052_x138_relink_candidate_scores/artifacts/submission-monitor.log`、runnerログは`/tmp/submission_exp052_x138_relink_candidate_scores_v3.runner.log`。初回pollはPENDING。プロセスとログを確認し、`SUBMISSIONS.md`にはref `56675101`のpendingスナップショットを記録した。
- 2026-09-29: このchatに15分間隔のheartbeat automation `biohub-exp052-submission-56675101`を作成。ref `56675101`の結果が確定するまで変化なしの通知を控え、確定時に実験記録とLB比較を更新してユーザーへ知らせ、automationを停止する。
- 2026-09-29 22:15:19 UTC: ref `56675101`の5分間隔監視で最初に`COMPLETE`とPublic LB `0.950`を確認。監視開始から最初の確定確認まで591分。厳密なKaggle側の採点終了時刻は未取得であり、この値はpoll間隔に依存する。
- 2026-09-30 09:51 UTC: ユーザーの指摘を受けてKaggle CLIの提出一覧でref `56675101`を再確認。`SubmissionStatus.COMPLETE`、Public LB `0.950`、Private LB `0.919`。`make record-exp EXP=exp052_x138_relink_candidate_scores PUBLIC_LB=0.950 PRIVATE_LB=0.919`の後に`make record-submission`で同じref `56675101`の履歴行を更新。exp043の保存済みPublic LB `0.950`と同値。実験statusと採否・完了判断は変更しない。
- 2026-09-30: `make validate-exp check-exp test-exp`（3件）と`git diff --check`が通過。採点結果を反映したためheartbeat automation `biohub-exp052-submission-56675101`を`PAUSED`に変更。
- 2026-09-30 10:01 UTC: ユーザーが本実験を採用・完了と明示判断。`metrics.json.status`を`completed`に変更し、`result.md`と`README.md`へ採用・完了と検証できた範囲を反映。Public LBは対照exp043と同値で、Private LBの対照値、学習動画の内部診断・新規20動画評価は未取得のまま。関連変更だけを確認して`main`へcommit・pushする。
