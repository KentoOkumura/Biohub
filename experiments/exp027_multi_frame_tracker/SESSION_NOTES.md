# exp027_multi_frame_tracker セッションノート

## 目的

固定候補と中央ペア教師を維持し、近傍3時点attentionが2時点対照より接続を改善するか検証する。

## 2026-09-20: 実装

- ユーザーが `multi_frame_tracker` の実装を依頼。未決だったモデル構造について「近傍に限った3時点attention」を選択した。
- `make new-exp EXP=exp027_multi_frame_tracker` で雛形を作成し、元候補の仮説ID・根拠・検証範囲・停止条件を `requirements.md` と `config.yaml` へ移行した。
- 中央ペアの既存cache、教師、損失、評価関数を引き継ぎ、直前cacheから前フレームを追加する。重なる時刻の候補ID・座標が一致しない場合は停止する。
- 物理距離15 µm内で最大64近傍のattentionを4層使う。64を超える点があれば省略せず停止する。pair MLPは中央ペア全組を採点する。
- 学習対象は2時点条件と3時点条件それぞれ2fold・3epoch、計4 tracker。booster 0。公開tracker・exp016 controlの再学習なし。
- 中央cacheの二重読込を除き、Jupytext sourceとNotebookを整合させた。
- `make validate-exp EXP=exp027_multi_frame_tracker`: 通過。
- `make check-exp EXP=exp027_multi_frame_tracker`: 通過。
- `make test-exp EXP=exp027_multi_frame_tracker`: cache境界・ID検査の2件通過。PyTorchがローカル環境になく、モデル実行テスト1件はskip。
- Jupytextの変換・testとRuff F821検査: 通過。
- `make check-strategy-docs`: HYP-20260910-11のexp028対応が索引に未反映で失敗。exp027のHYP-20260910-10対応とは別の進行中変更。

## 次のアクション

1. Kaggle train packageの内容・metadata、GPU quotaを確認する。
2. 4 trackerのruntime benchmark gateを含むtrain NotebookをKaggleで実行し、model manifestと両胚別診断を記録する。
3. 前段診断が `requirements.md` の条件を満たす場合だけ、同じexpで固定graph推論を実装・実行する。

## 実行予定のコマンド

```bash
make prepare-kaggle-notebooks EXP=exp027_multi_frame_tracker EXTRA_ARGS="--notebook train --run-on-push"
make push-kaggle-train EXP=exp027_multi_frame_tracker
make kaggle-logs KERNEL=<generated-kernel-id>
```

GPU push前に生成metadataのresourceと `uv run kaggle quota --format json` の残時間を確認し、観測値を本書へ記録する。Kaggle submissionは行わない。

## 2026-09-20 14:25 JST: Kaggle train push前の確認

- `make prepare-kaggle-notebooks EXP=exp027_multi_frame_tracker EXTRA_ARGS='--notebook train --run-on-push'`: strict package生成に成功。
- 生成metadata: `kentookumura/exp027-multi-frame-tracker-train`、GPU `true`、TPU `false`、`NvidiaTeslaT4`、internet `false`、run_on_push `true`。正のtrain Notebook、config、metricsと補助sourceをpackageに含む。
- 14:25 JSTに `uv run kaggle quota --format json` でGPU使用10.62時間、Kaggle残34.38時間、refresh 2026-09-26 00:00 UTCを確認した。ユーザーの週30 GPU時間の制約で残る目安は19.38時間。4 trackerのbenchmark予測が12時間gateを超えた場合はfull train前に停止するため、pushを進める。

## 2026-09-20 14:27 JST: train push未成立

- `make push-kaggle-train EXP=exp027_multi_frame_tracker` はpackage validatorを通過したが、Kaggle CLIが `Kernel push error: Maximum batch GPU session count of 2 reached.` を返した。wrapperの終了コードは0でも、pushとNotebook実行は成立していない。
- 既存のGPU sessionは停止していない。Kaggle kernel version、model、両胚診断は未取得。GPU残時間を消費したとは扱わない。
- session枠が空くまで待つか、停止してよい対象をユーザーに確認する。対象未指定のまま既存sessionを止めない。

## 2026-09-20 14:40 JST: train push再試行前の確認

- ユーザーの実行指示を受け、`make validate-exp`、`make check-exp`、`make test-exp`、train package再生成を実施。検証は通過（テスト2件通過、PyTorch未導入による1件skip）。
- 14:40 JSTの `uv run kaggle quota --format json` はGPU使用10.95時間、Kaggle残34.05時間、refresh 2026-09-26 00:00 UTC。ユーザーの週30 GPU時間制約で残る目安は19.05時間。Notebook内の12時間runtime gateを維持して再pushする。

## 2026-09-20 14:41 JST: train Notebook起動

- `make push-kaggle-train EXP=exp027_multi_frame_tracker` が成功。`kentookumura/exp027-multi-frame-tracker-train` のversion 1が作成され、run_on_pushで実行が開始された。
- Kaggle実行の進捗、runtime gate、2時点対照と3時点条件の両胚別診断を確認する。

## 2026-09-20 14:49 JST: version 1の失敗とversion 2前の確認

- Kaggle train version 1はcache入力探索で停止した。`/kaggle/input/notebooks`以下で`window_cache_summary.json`が見つからなかった。学習・benchmarkは未開始。
- sourceの探索範囲を`/kaggle/input`へ広げ、入力ルートとsummary候補パスをログに出すよう修正した。候補の内容SHA・identity検証は維持する。
- Jupytextを再変換し、`make validate-exp`、`make check-exp`、`make test-exp`を再実行。すべて通過（テスト2件通過、PyTorch未導入による1件skip）。strict train package生成も通過。
- 14:49 JSTのGPU quotaは使用11.10時間、Kaggle残33.90時間、refresh 2026-09-26 00:00 UTC。週30時間制約の残り目安18.90時間。runtime gateは12時間のままversion 2をpushする。

## 2026-09-20 14:50 JST: train Notebook version 2起動

- `make push-kaggle-train EXP=exp027_multi_frame_tracker` でversion 2をpushし、Kaggle実行を開始した。
- 実行ログで入力ルートがフラットな `/kaggle/input/exp015-oracle-stage-limits-inference` であり、cache summaryをそこから発見した。version 1の探索根が誤っていたことを確認した。

## 2026-09-20 15:02 JST: version 2のbenchmark停止

- cache summary SHA、19,701窓のidentity、公開checkpoint SHA、胚分割、T4 GPUは確認できた。有効窓18,707件（GT欠損のため994件除外）。
- 最初のruntime benchmarkで半径15 µm内の近傍数が暫定上限64を超え、モデルのfail-fastが発火した。学習には進んでいない。件数を切り捨てず、各batchの半径内の実測最大件数に合わせてattentionを計算するよう変更した。1024件を安全上限として残し、benchmarkと学習summaryへ実測最大件数を保存する。評価胚の指標を見て設定を調整したものではない。

## 2026-09-20 15:06 JST: version 3のGPU push前確認

- 近傍数を各batchの半径内実測最大に合わせ、上限1024を超える場合だけ停止する。benchmarkと学習summaryへ実測最大近傍数を記録する。確認済みのフラットなcache入力パスを優先して解決する。
- Jupytext変換後、`make validate-exp`、`make check-exp`、`make test-exp`、strict train package生成を再実行して通過（テスト2件通過、PyTorch未導入による1件skip）。
- 15:06 JSTのKaggle GPU使用11.46時間、Kaggle残33.54時間、refresh 2026-09-26 00:00 UTC。週30 GPU時間制約の残り目安18.54時間。12時間runtime gateを保持してpushする。

## 2026-09-20 15:07 JST: train Notebook version 3起動

- `make push-kaggle-train EXP=exp027_multi_frame_tracker` が成功し、version 3のKaggle実行を開始した。benchmarkで安全上限とGPUメモリを確認する。

## 2026-09-20 15:20 JST: version 3 runtime gate停止と証拠回収

- Kaggle train version 3のcache SHA、19,701窓identity、公開checkpoint SHA、18,707有効窓とT4を確認した。
- 各fold・条件で64窓を実測。順に2時点fold 0は23.91秒・最大近傍48・GPU peak 1.30 GB、3時点fold 0は26.86秒・69・1.71 GB、2時点fold 1は27.05秒・52・1.43 GB、3時点fold 1は34.84秒・73・1.89 GB。
- 全学習・validation・外側評価窓への保守的外挿は124,281.88秒（34.52時間）で12時間gateを超えた。Notebookは意図した例外で停止し、Kaggle statusは `KernelWorkerStatus.ERROR`。全件学習とcheckpoint生成は未実施。
- `make kaggle-output KERNEL=kentookumura/exp027-multi-frame-tracker-train OUT=experiments/exp027_multi_frame_tracker/artifacts/kaggle_train_v3` でbenchmark summary、metrics、filter audit、split manifest、ログを回収した。benchmark summaryのSHA-256は `984b8d46f56d4fbf69e29ae6f81874f4357fcd54129d5784a32cd0ea62fe72e7`。
- Kaggleで書いた`metrics.json`を正本へ移し、`make record-exp`で実行状態`failed`とkernel version 3を記録した。これはruntime gateで停止した実行状態であり、実験の採否や完了の判断ではない。CV、公式graph score、Kaggle submissionは未実施。
- 終了後の `uv run kaggle quota --format json` はGPU使用11.66時間、Kaggle残33.34時間。週30時間制約の残り目安は18.34時間。新しい全件学習の方式は未決のため、追加のGPU実行は始めていない。

## 2026-09-20: 予備実験への縮小を承認

- ユーザーが「小規模な予備実験：1epoch・各胚64窓の評価へ縮小する」を選択した。
- 2条件・2fold・計4 trackerの学習は継続し、全学習窓を1epoch、内部validationを全件使用する。外側評価は各胚のeligibleな動画からSHA-256順位で64動画、各動画から1窓を事前選択する。2条件は同じ窓を使い、選択manifestとSHAを保存する。
- version 3の64窓実測速度からの保守的推定は約7.17時間（学習・内部validation・選択した外側評価の合計）。12時間gateと週30 GPU時間制約を維持する。予備結果だけで全graph推論に進まない。
- 旧3epoch・外側全窓のbenchmarkは `metrics.json` の `train_stage.full_scope_benchmark_v3` と `evidence.reruns` に保持した。
- `make validate-exp`、`make check-exp`、`make test-exp`は通過（3件通過、PyTorch未導入による1件skip）。Jupytext round tripとRuff F821検査も通過。

## 2026-09-20 15:33 JST: 予備実験push前確認

- strict package生成、`make validate-exp`、`make check-exp`、`make test-exp`、Jupytext変換とround tripは通過。学習対象は2条件×2fold×1epochの4 tracker、booster 0、公開control再学習なし。外側評価は各胚64窓で、両条件共通。
- 生成metadataは既存の `kentookumura/exp027-multi-frame-tracker-train`、T4 GPU true、TPU false、internet false、run_on_push true。package内configの`stage: pilot`、`epochs: 1`、外側64窓を確認した。`kaggle kernels pull -m`で既存kernelの存在を確認し、同じslugへversionを追加する。
- 15:33 JSTの `uv run kaggle quota --format json` はGPU使用11.79時間、Kaggle残33.21時間、refresh 2026-09-26 00:00 UTC。ユーザーの週30 GPU時間制約で残る目安は18.21時間。予備実験の保守的推定7.17時間と12時間gateの両方を満たすためpushする。

## 2026-09-20 15:34 JST: 予備実験version 4起動

- `make push-kaggle-train EXP=exp027_multi_frame_tracker` が成功し、Kaggle Notebook `kentookumura/exp027-multi-frame-tracker-train` のversion 4を実行開始した。ローカルmetricsの実行状態を`running`へ変更した。
- 初回の `make kaggle-logs` はKaggle APIのSSE endpointがHTTP 500を返した。`make kaggle-status` ではversion 4が`RUNNING`、`kaggle kernels pull -m`でも既存kernelを確認した。同じslugでログ接続を再試行し、起動ログを取得した。
- version 4の64窓×4条件benchmarkは通過。保守的推定24,668.17秒（6.85時間）で12時間gate内。評価窓選択manifest SHA-256は `b46432e42ebb56fadd98d3c358ca28f712e27dbcf399a895939510c504a6016f`。Kaggle側で4 trackerの1epoch学習を開始した。

## 2026-09-20: 予備実験version 4完了と生成物回収

- Kaggle Notebook `kentookumura/exp027-multi-frame-tracker-train` のversion 4が `KernelWorkerStatus.COMPLETE` で終了した。`make kaggle-output EXP=exp027_multi_frame_tracker NOTEBOOK=train KERNEL=kentookumura/exp027-multi-frame-tracker-train OUT=experiments/exp027_multi_frame_tracker/artifacts/kaggle_train_v4` で4つのモデル重み、学習summary、評価窓選択manifest、診断gate、metrics、ログを回収した。
- `training_summary` のstage `pilot`、1epoch、2条件×2foldの計4モデル、各胚64窓、各胚64動画を確認した。4重みのファイルSHA、selection/benchmark/model manifestの参照SHA、診断gateの参照SHAが回収ファイルと一致した。評価窓は両条件で同じ事前選択を使用した。
- 外側評価の既知正例edge recallは、44b6で2時点160/192（83.33%）から3時点162/192（84.38%）、6bbaで489/522（93.68%）から479/522（91.76%）。active pair内の教師負例を正例と予測した件数は44b6で19→18、6bbaで23→21。分裂親は44b6で0、6bbaで1（recallは両条件0）。部分注釈で端点不明のpairが大部分を占めるため、教師負例件数は全接続の真の誤接続数ではない。
- Kaggle Notebook実行時間は4,163.38秒（69.39分）、学習処理時間は3,568.28秒（59.47分）。終了後のKaggle GPU quotaは使用14.27時間・残30.73時間、refresh 2026-09-26 00:00 UTC。週30 GPU時間の自己制限では残15.73時間。quota変化は他sessionの利用も含み得るため、本Notebook単独のGPU消費量とは扱わない。
- `make record-exp STATUS=debug_completed ... --no-summary` でKaggle kernel version 4、実行時間、診断値、生成物SHAを `metrics.json` に記録した。`debug_completed` は予備実行の状態で、実験の完了・採否を表さない。
- 両胚で既知edge recallの改善が揃わず、44b6では分裂親がない。予備実験でもあるため、`requirements.md` の全graph進行条件は成立しない。全graph推論・公式評価・Kaggle submissionは実行せず、実験の完了や追加の全件前段診断はユーザー判断を待つ。

## 2026-09-20: 3時点で改善が揃わなかった理由の考察

- ユーザーの依頼で、実装・内部validation・外側診断・保存済み4重みを監査した。学習や追加Kaggle実行は行っていない。
- 過去用の時間埋め込みは3時点の両foldで更新されており、過去への勾配が完全に切れていた説明は支持されない。局所attentionでの混合、窓内時刻が重複する初期位置特徴、1epochの適応、疎い教師と確率0.5評価を原因候補として整理した。
- 44b6で学習したモデルは内部・外側の両方でrecall低下、6bbaで学習したモデルは両方で上昇した。学習量だけを変えた比較ではないため、更新回数や注釈の差を原因と確定していない。
- 結論と未確認事項の正本は [原因考察](../../docs/surveys/biohub-exp027-three-frame-analysis_20260920.md)、集計コードと生の数値は `studies/exp027_three_frame_review/`。実験の完了・採否は変更していない。

## 2026-09-20: 過去入力の有無を比較する診断を承認

- ユーザーが前ターンの提案を指定し「これに進んでください」と依頼した。train v4の4重みを使い、2fold×3推論条件・各胚64窓を比較する。学習variant 0、再学習model 0、booster 0、control再学習なし。
- 2時点と3時点の元条件の再現確認、3時点の同じ重みでの過去mask、親順位・確率・接続ごとの変化を診断Notebookへ実装する。configとrequirementsへ先に契約を追記した。

## 2026-09-20 19:28 JST: diagnostic push前確認

- 自己完結Notebookは6章で入力検査、既存モデル・教師、接続ごとの集計、3条件推論、再現照合、結果保存を展開する。既存trainの必要な関数だけを抽出し、train_one_epochやoptimizerは含めない。モデル・教師のAST一致をテストした。
- `make validate-exp`、`make check-exp`、`make test-exp`、Jupytext round tripは通過。テスト7件通過、PyTorch未導入のローカルモデルテスト1件skip。
- strict diagnostic packageのresourceはT4 GPU、TPU false、internet false、run_on_push true。学習0回、保存済み4モデル・3推論条件・128窓、1時間gate。
- Kaggle quotaは使用15.62時間、残29.38時間、refresh 2026-09-26 00:00 UTC。週30時間制約の残りは14.38時間で、1時間の診断上限内。新しいdiagnostic slugの既存衝突は確認されなかった。

- `make push-kaggle-notebook EXP=exp027_multi_frame_tracker NOTEBOOK=diagnostic` が成功し、`kentookumura/exp027-multi-frame-tracker-diagnostic` version 1を実行開始した。live SSEへ接続しbootstrap起動を確認、`kernels pull -m`でもkernelの存在を確認した。

## 2026-09-20: diagnostic version 1完了と証拠回収

- live SSEで128窓の推論完了、`reference_reproduced: true`と正常終了を確認し、最終statusも `KernelWorkerStatus.COMPLETE` だった。Notebook実行時間611.62秒（10.19分）。新しい階層のKaggle inputを探索し、元のcache identityと全注釈hashの照合を行った。
- `make kaggle-output EXP=exp027_multi_frame_tracker NOTEBOOK=diagnostic KERNEL=kentookumura/exp027-multi-frame-tracker-diagnostic OUT=experiments/exp027_multi_frame_tracker/artifacts/kaggle_diagnostic_v1` で接続別CSV、128窓の全logits、入力・予測manifest、metrics、ログを取得した。
- `uv run python studies/exp027_three_frame_review/read_context_diagnostic.py --output-root experiments/exp027_multi_frame_tracker/artifacts/kaggle_diagnostic_v1` はPASS。4重みの前後state SHA、全NPZのファイル／内容SHA、714正例×3条件の行数、比較差の分解を確認した。過去のない先頭窓ではfull/maskedの最大logit差3.81e-6で、計算shapeに伴う浮動小数点差の範囲だった。完全bitwise一致とは扱わない。
- 3時点の同じ重みへの過去入力追加で44b6はrecall −1.04ポイント、6bbaは+0.57ポイント。正解親の1位件数は両胚とも純増0。前回2時点／3時点のrecall差も、変化した接続がすべて正解親1位のまま0.5を跨いだ結果と確認できた。詳細と未解決事項はresultと考察を更新した。
- 元のtrain_stageの内容一致を確認して保持し、diagnosticの構造化された数値・SHAだけをmetricsへ統合した。`make record-exp ... STATUS=debug_completed --no-summary` で診断kernel version 1とintegrity検証を記録した。実験の完了・採否は未判断。
- 終了後quotaはGPU使用15.79時間・Kaggle残29.21時間、週30時間制約の残14.21時間。refresh 2026-09-26 00:00 UTC。追加学習、全graph、公式評価、Kaggle submissionは行っていない。

## 2026-09-20: 内部validationの確率・閾値診断を承認

- ユーザーの「次に進んでください」を受け、内部validation全1,849窓で3条件を比較する。保存済み4重み、2fold、学習variant 0、booster 0、control再学習なし。閾値の決定規則を実行前にconfigとrequirementsへ記録した。
- 外側予測はdiagnostic v1を再利用する。内部教師負例数を2時点0.5の件数以下に揃えた閾値を、正例結果を見ずに決めて外側へ固定適用する。

- validation_diagnosticのpush前検証: strict validate-exp、check-exp、test-exp（10件通過・PyTorch未導入による1件skip）、Jupytext round trip、strict package検証が通過。T4、internet false、学習0回、1時間gate。GPU quotaは使用15.79h、Kaggle残29.21h、週30h制約では残14.21h（refresh 2026-09-26 00:00 UTC）。内部入力を全件読み込む時間は窓数で按分してruntimeを推定する。
- validation_diagnostic version 1のpushが成功し、live SSEでbootstrap起動を確認した。出力回収後、内部閾値の再計算と外側の元0.5指標の再現を確認する集計コードを準備した。

## 2026-09-20: validation diagnostic version 1の終了・回収

- Kaggle statusはCOMPLETE、元の内部指標はreference_reproduced=true。実行時間360.15秒。19動画・1,849窓、3推論条件、4保存重み、追加学習0回。
- 全1,870ファイル・82,716,591 bytesを回収した。多数の窓別NPZの取得は、このタスクの直列CLIを停止後、同じKaggle APIを12並列・timeout付きで実行した。全ファイルを取得し、1,849窓のfile/content SHAを検証した。実行Notebookのsource SHAもローカルと一致した。
- 内部だけでの閾値選択を再計算して一致。外側での3時点回収は44b6が163/192、6bbaが484/522。教師負例予測は18件・23件。2時点は160/192・489/522、19件・23件で、6bbaの不足が10件から5件へ縮小したが、両胚の改善は成立しない。
- 内部平均値2箇所のfloat32丸め差5.96e-8、外側softmax再計算の最大差1.74e-6を検出した。正例は保存確率を使用し、教師負例の最小閾値距離2.01e-4と元の0.5件数の再現を確認。閾値や件数・内容SHAを許容誤差で緩めていない。
- 終了後GPU quotaは使用15.93h・Kaggle残29.07h、週30h制約の残14.07h、refresh 2026-09-26 00:00 UTC。quota差は他sessionを含む可能性がある。
- train_stageとcontext_diagnosticを保持し、新しい数値・証拠をvalidation_diagnosticへ追加した。詳細な結論はdocs/surveysの既存考察へ統合。全graph、追加学習、公式評価、submissionは行っていない。実験採否・完了は未判断。

## 2026-09-20: ユーザー判断による実験完了とGit保存

- ユーザーが「実験を完了扱いとしてgit commitとpushしてください」と明示したため、metrics.jsonの実験statusをcompletedへ更新する。完了範囲は承認済みの1epoch・各胚64窓の予備比較、保存重みの過去入力除去、内部validationで選ぶ閾値の診断である。
- 両胚の一貫した改善は未確認で、全graphへの進行条件は未達。公式scoreは未計測であり、モデル採否と上位仮説全体の判断を完了判断に含めない。
- exp027の実装・診断コード、実行証拠の要約、考察、候補移行、速度・位置履歴候補のP2への変更を保存対象とする。重み・Kaggle生成物と他実験の作業変更はcommitへ含めない。
- 保存前検証: make validate-exp、make check-exp、make test-expを実行し、設定・静的検査は成功、テストは10 passed / 1 skipped。skipはローカルにPyTorchがないため。make check-strategy-docs、make validate-surveysとgit diff --checkも成功した。
