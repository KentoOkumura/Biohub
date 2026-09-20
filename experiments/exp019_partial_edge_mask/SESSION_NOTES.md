# exp019_partial_edge_mask セッションノート

## 目的と作業状況

- 2026-09-19: ユーザーの実装依頼により、設計可能・未決事項なしの候補をexp016から実験化した。対象は正例のある子列へのloss mask一変更。exp016の保存済みtrackerと公式graph指標を対照に使い、control再学習はしない。
- exp016からコピーしたColab・recovery専用経路は新実験では使わない。trainとKaggleの固定cache graph推論を実装した。
- 学習の実行予定: active variant 1、model/config 1、outer fold 2、出力tracker 2個、booster 0、control再学習0。学習時間は64 window/foldのbenchmarkで推定し、12時間gateを適用する。
- 推論の実行予定: 保存済みexp016指標を主対照にし、変更variantのみを新たにgraph replay・公式評価する。公開tracker replayは候補同一性の検査に使う。Kaggle submissionは行わない。
- 現在の状態: 公式graph評価とユーザーの不採用判断を[結果](result.md)に記録済み。実験statusは[metrics.json](metrics.json)を正とする。

## 実装と検証ログ

- 2026-09-19: `make new-exp EXP=exp019_partial_edge_mask SOURCE=experiments/exp016_frozen_image_encoder`を実行。親からの設定・参照を確認した。
- 2026-09-19: `frozen_tracker.py`の損失を正例子列のpairへ限定。現行OR mask、正例子列mask、除去pair、未知親を含むpair、同一GT近傍で一意対応から除外された候補を監査する。target、softmax分母、選択指標は固定した。
- 2026-09-19: trainとinferenceのJupytext sourceを修正し、Notebookを再生成。trainは人工例の勾配をKaggle実行前に検査する。inferenceは新model manifestの入力SHAとfoldモデルSHAを確認し、公開control graphがexp016保存済み公式評価と一致した場合だけ、変更variantと比較する。
- 2026-09-19: `make check-exp`、`make test-exp`、`make validate-exp`を実行。初回は書式、人工例の旧OR mask期待値、要件見出しを修正し、再実行でcheckとvalidateが通過、testは4 passed・1 skipped。skippedはローカル環境にPyTorchがないためで、同じ勾配確認をKaggle train Notebookの起動時ガードへ入れた。
- 2026-09-19: CLIのsandboxはbubblewrap不在で起動できず、ローカルread/writeとcheckを承認済みのunsandboxed実行で行った。

## 実行前の予定

1. Jupytext round-trip、編集後の対象テストと設定検証を再実行する。
2. `kaggle-platform`のpush前手順でGPU残量とActive Sessionsを確認する。
3. 無料GPU枠と12時間gate内ならtrain Notebookをprepare・pushし、fold別model、教師監査、時間とSHAを記録する。
4. train結果を確認して固定cache graph推論と公式評価を行い、exp016との差を両胚別にまとめる。

- 2026-09-19 12:21 UTC: train packageを`make prepare-kaggle-notebooks EXP=exp019_partial_edge_mask EXTRA_ARGS='--notebook train --run-on-push'`で生成。metadataは`kentookumura/exp019-partial-edge-mask-train`、GPU=true、TPU=false、`NvidiaTeslaT4`。CLI credential確認は成功。直前の`uv run kaggle quota --format json`ではGPU残45.00h、refresh 2026-09-26 00:00 UTC。ユーザー方針の週30h以内であり、この学習は12h gate、変更variant 1・fold 2・model 2・booster 0・control再学習0としてpush可能と判断した。

- 2026-09-19 12:21 UTC: `make push-kaggle-train EXP=exp019_partial_edge_mask`は自動承認レビューにより実行前に却下された。理由は、実装依頼はprivate notebookコードのKaggleへの外部送信と実行先の明示承認を含まないとの判断。迂回や再試行は行わない。Kaggleのkernel version、学習model、公式評価は未生成。ユーザーの明示承認を得た場合のみ、packageを再生成し、quotaを改めて確認してpushする。

- 2026-09-19 12:34 UTC: ユーザーが「実行してください」と明示したため、Kaggleでのtrainとinference実行を開始。`make check-exp`、`make test-exp`、`make validate-exp`は通過（testは5 passed・1 skipped）。trainは変更variant 1、model/config 1、outer fold 2、出力tracker 2、booster 0、control再学習なし。train metadataはGPU=true、TPU=false、T4。`uv run kaggle quota --format json`は週GPU残45.00h、refresh 2026-09-26 00:00 UTC。12時間gateを適用してpushする。
- 2026-09-19 12:36 UTC: `make prepare-kaggle-notebooks ... --notebook train --run-on-push`後に`make push-kaggle-train EXP=exp019_partial_edge_mask`が成功し、private kernel `kentookumura/exp019-partial-edge-mask-train` version 1を実行開始。Kaggle pullで存在を確認し、Kaggle側metadataもGPU=true、TPU=false、`machine_shape=NvidiaTeslaT4`。live logsを監視中。
- 2026-09-19 12:55 UTC: trainの人工例勾配guard、cache identity SHA、公開tracker source SHA、GT window filter、T4検出が通過。19701 window中18707が学習対象。64 window/foldのbenchmarkはfold 0が13.38秒、fold 1が13.91秒、保守的な全train予測は29948.88秒（8.32h）で12h gate内。live SSEは一度切断・再接続したが、`kaggle kernels status`は`RUNNING`。以前の学習1–2h見積もりは短すぎたとユーザーへ訂正した。
- 2026-09-19 13:48 UTC: train kernel version 1が`COMPLETE`。Kaggle outputを`artifacts/kaggle_train_v1/`へ取得し、2つのmodel file SHA、manifest SHA、評価胚の対応を照合した。fold 0（6bba評価）はepoch 2、fold 1（44b6評価）はepoch 0を選択。学習本体3643.22秒、Notebook全体4264.85秒。outerの既知edge positive recallは公開初期重みから両foldで約0.0033・0.0038上がった一方、選択指標は両foldで約0.00004下がった。公式graph指標は未計算。Kaggle kernel id/version/URLとtrain-stage証拠を`metrics.json`に記録した。
- 2026-09-19 13:51 UTC: inference packageを`make prepare-kaggle-notebooks ... --notebook inference --run-on-push`で生成。metadataはprivate、GPU=true、TPU=false、`machine_shape=NvidiaTeslaT4`で、exp015 cache kernelと完了済みexp019 train kernelを参照。推論で追加学習0、読み込むfoldモデル2、booster 0。直前quotaはGPU残43.81h、refresh 2026-09-26 00:00 UTC。推論の12h gateと週30h以内の方針を満たすためpush可能と判断。
- 2026-09-19 13:53 UTC: `make push-kaggle-infer EXP=exp019_partial_edge_mask`が成功し、private kernel `kentookumura/exp019-partial-edge-mask-inference` version 1を実行開始。Kaggle pullで存在とGPU=true、TPU=false、`machine_shape=NvidiaTeslaT4`を確認。live logsはT4 2枚を認識、追加学習0、読み込みモデル2、main画像encoder forward 0と報告。Notebook内に表示される公開support packのLB履歴は本実験の結果ではない。
- 2026-09-19 14:06 UTC: inference kernel version 1は`RUNNING`。ユーザーが完了後の結果取得・記録までの自動監視を明示承認したため、15分間隔のthread heartbeat `exp019-inference`を`ACTIVE`で作成した。未変化時は通知せず、完了または要対応時に通知して`PAUSED`へ切り替える。再push・cancel・stop・追加GPU実行・submissionは監視対象外。
- 2026-09-19 19:05 UTC: 15分監視でinference kernel version 1の`COMPLETE`を確認。Kaggle outputの評価JSON、per-sample JSON、実行receipt、事前検証、cache replay summary、fold model manifest、metricsと完了ログを`artifacts/kaggle_inference_v1/`へ取得した。全出力取得はGEFF graphファイルが多数あるため途中停止し、不完全なローカルGEFFディレクトリを削除した。graph本体をローカル全件再検証したわけではないが、Kaggle上のcompact graph内容SHAと評価JSONのSHAをreceiptで確認した。
- 公式評価は199/199動画でskip 0。exp016保存済み対照の再計算値は完全一致。公開tracker候補graphと固定候補座標の一致、評価器preflight、train model manifest参照、追加学習0、T4 2枚、submissionなしを確認。公式評価JSON SHA256は`ce16bbb7afec7b4e765a8c89dc58e04c96fa3829d17ea189dc27fc08801a943a`、per-sample JSON SHA256は`0141aa423321298acbe610ce38bc529609b2dc88d5b1335b20861a6b976a8695`でreceiptと一致。
- exp016保存済みtrackerからの公式score差は全体-0.00307279554（0.91205450131→0.90898170577）、44b6 +0.00035841791（71動画）、6bba -0.00375252474（128動画）。adjusted edge Jaccard差は全体-0.00235070342、division Jaccard差は全体-0.00722092116。分裂TP/FP/FNはexp016の25/93/126からexp019の24/101/127。推論Notebookは17645.01秒、cache replay予測4352.74秒。採否・実験完了はユーザー判断待ちとし、metrics statusは`running`を維持。
- 2026-09-19 19:05 UTC: receiptとfold/cache manifestの自己SHA、評価JSON等のfile SHA、公式score差の算術、199件の一意なsampleと胚別71/128件を再検証した。`make validate-exp`、`make check-exp`、`make test-exp`は通過（5 passed・1 skipped。ローカルPyTorch欠如によるskipはKaggle train起動時の勾配guardで確認済み）。`git diff --check`も通過。結果と未解決の採否判断を`result.md`へ記録したため、heartbeat `exp019-inference`を`PAUSED`に変更し、重複取得を止めた。
- 2026-09-20 00:34 UTC: ユーザーがexp019の不採用と実験完了を明示した。公式scoreはexp016より全体-0.00307279554、44b6 +0.00035841791、6bba -0.00375252474で、両胚改善の受け入れ条件を満たさない。`metrics.json`の実験statusを`discarded`へ変更し、`result.md`に判断を記録した。実験関連の変更だけをcommit・pushする。
