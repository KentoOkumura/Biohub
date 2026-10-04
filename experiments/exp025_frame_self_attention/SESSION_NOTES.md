# exp025_frame_self_attention セッションノート

## 目的

固定画像特徴で各フレームのcell集合へSelf-Attentionを追加し、Model A/Bと現行を比較できる実装を作る。

## 現在の作業

- 作業内容: Model A/Bと恒等初期化Model Bの2fold×3epoch学習とpair比較を回収・検証済み。実験の完了・採否はユーザー判断待ち。
- ブロック要因: pair指標が混在し、公式graph scoreは未計測。現行controlの再学習や提出は今回の承認範囲外。
- 次: 結果をユーザーへ提示し、実験の扱いとgraph評価を進めるか判断を待つ。

## コマンドログ

- 2026-09-20: `make new-exp EXP=exp025_frame_self_attention SOURCE=experiments/exp016_frozen_image_encoder` と同等の `uv run --offline python scripts/new_experiment.py ...` で作成。親の実行記録をリセット。
- 2026-09-20: `make validate-exp EXP=exp025_frame_self_attention` がstrictで通過。
- 2026-09-20: `make check-exp EXP=exp025_frame_self_attention` が通過。
- 2026-09-20: `UV_NO_SYNC=1 make test-exp EXP=exp025_frame_self_attention` でPyTorch 2.11 CPUの実装契約テスト9件通過。旧sourceとの差、mask、順序、loss・勾配、保存・復元、train manifestからの推論loadを確認。
- 2026-09-20: `make check-strategy-docs` が通過。候補契約をrequirements/configへ移し、バックログ未着手行を実験リンクへ変更。
- 2026-09-20: `make prepare-kaggle-notebooks` でtrain/inference packageを生成し、両packageのmetadata検証とJupytext変換テストが通過。Kaggle push・実行はしていない。
- 2026-09-20: `make validate-config` は隔離worktreeに `data/raw` とsample submissionがないためroot設定検証で停止。対象実験の `make validate-exp` は通過。全体リンク検査には旧backlog参照とGit管理外artifactへの履歴リンクの欠損が残るが、exp025の新規リンク欠損はない。

- 2026-09-20 02:31 UTC: `uv run kaggle quota --format json` でGPU残38.11時間、使用6.89/45.00時間、refresh 2026-09-26 00:00 UTCを確認。リポジトリの週30時間上限で計算すると残23.11時間。push対象はT4 GPU、TPU無効、Model A/Bの2構成×2fold＝4 tracker学習、booster 0、現行control再学習なし。exp016の学習Notebook実測は約1.28時間だが、今回のSelf-Attention追加後の費用は未測定なので、まずNotebook内benchmark gateを実行する。

- 2026-09-20T02:32:32.954252+00:00: `make push-kaggle-train EXP=exp025_frame_self_attention` が成功し、canonical train kernel v1をpush。Kaggle学習が開始された。model manifest・指標は未取得。

## 2026-09-20 学習開始時の費用境界

- 初期active: Model A/Bの2構成、各1 config、2 outer fold、計4 tracker学習、booster 0。現行control再学習は無効。保存済みexp016を精度参照に使い、実行条件差に注意する。
- 現行controlを追加する場合: 3構成×2foldで計6 tracker学習となり、追加は2 tracker学習。runtime benchmarkで費用を測る前にcontrolのGPU train pushはしない。
- 推論は選択variantごとにfoldモデル2個、追加学習0。A/Bの両方を評価するにはvariant別のKaggle実行が必要。
- runtime、peak GPU memory、公式graph指標、model/graph SHAは未測定。未取得値を0や成功として扱わない。

## 変更点

- 実験固有 `simple_node_transformer.py` に共有2層Transformer Encoderと2 flagを追加。旧Cross block、Pair MLP、座標差、loss interfaceは維持。
- trainにvariant別の初期化、GPU benchmark、fold別checkpointとmanifest、source SHA記録を追加。inferenceは指定variantをmanifestからSHA検証してstrict復元する。
- 親からコピーしたColabと別stageのファイルは今回の契約に不要なため除いた。

## 次のアクション

[結果と残る検証](result.md)を基に、実験の完了・採否と全graph評価へ進むかについてユーザー判断を待つ。学習と隣接2フレーム評価の数値・実行証拠は[metrics.json](metrics.json)を参照する。

## 2026-09-20 学習と結果回収の履歴

- 2026-09-20 02:45 UTC: train kernel v1は4構成のbenchmark完了後、12時間gateで学習前に停止。保存済み `benchmark_summary.json` はA fold0/1=13,536.6/16,616.9秒、B fold0/1=14,641.7/19,703.6秒、合計64,498.7秒（17.92時間）を保守的に予測。A単独8.38時間、B単独9.54時間の予測なので、fold・seed・3 epoch・初期化・損失を固定してA/Bを別Notebookで実行する。v1 outputは `/tmp/kaggle-output/exp025-v1/benchmark_summary.json` に取得。
- 2026-09-20 02:46 UTC: A専用kernel push前に `uv run kaggle quota --format json` でGPU使用7.33/45時間、Kaggle残37.67時間、repo週30時間上限の残22.67時間、refresh 2026-09-26 00:00 UTCを確認。生成metadataはT4 GPU有効、TPU・Internet無効。Model Aの2fold×3 epochだけを実行し、保守的予測8.38時間はNotebook 12時間gateとrepo残余内。Bは別kernel push前に再確認する。
- 2026-09-20 02:47 UTC: A専用kernel `kentookumura/exp025-frame-self-attention-a-train` v1 push成功、status RUNNINGを確認。AはModel Aのみ2fold×3 epoch、model_count=2。
- 2026-09-20 02:49 UTC: B専用kernel push前にGPU使用7.39/45時間、Kaggle残37.61時間、repo週30時間上限の残22.61時間を確認。生成metadataはT4 GPU有効、TPU・Internet無効。A/B保守的予測合計17.92時間は残余内、B単独9.54時間はNotebook 12時間gate内。Aは別kernelで実行中。
- 2026-09-20 02:50 UTC: B専用kernel v1 push試行はKaggle `Maximum batch GPU session count of 2 reached` で拒否。CLI/Makeのexit codeは0だったが、B statusは404で実行未開始を確認。既存セッションを停止せず、A終了後にGPU quotaを再確認してBを再pushする。
- 2026-09-20 02:56 UTC: A専用kernelの64-window/fold GPU benchmark通過。fold0/1は13.7294/14.1992秒、予測14,071.3/16,575.2秒、合計30,646.4秒=8.51時間 < 12時間gate。peak GPU memoryは308,425,216/327,887,360 bytes。3 epoch/foldの学習開始。
- 2026-09-20 03:05 UTC: A fold0 epoch0完了。elapsed 312.16秒、train loss 0.00054256、internal-validation loss 0.00030686、positive edge recall 0.94660、selection score 0.985897。division parentは内部検証で3件、recall 0.0なので単独の成功判定に使わない。
- 2026-09-20 03:05 UTC: B専用kernel再push前にGPU使用7.95/45時間、Kaggle残37.05時間、repo週30時間上限の残22.05時間を確認。A実測benchmark保守予測8.51時間とB v1保守予測9.54時間の合計18.05時間は残余内。B package metadataはT4 GPU有効、TPU・Internet無効。
- 2026-09-20 03:06 UTC: B再pushは、最初metrics更新後のpackage staleで拒否され、再package後もKaggle同時GPUセッション2件上限で拒否。Bは未開始。A完了を待ってからquota/metadataを再確認して再試行する。CLIはセッション上限でもexit code 0を返すため成功扱いしない。
- 2026-09-20 03:15 UTC: A fold0は3 epoch完走。各epoch train elapsed 312.16/311.98/310.66秒、internal-validation selection score 0.98589709/0.98589535/0.98590581、best epoch2。epoch2 positive edge recall 0.93903、division parent recall 0/3。fold1へ進む。
- 2026-09-20 03:36 UTC: heartbeatでA kernel RUNNINGを確認。fold1はepoch0/1完了（各約460秒）。internal-validation selection scoreはいずれも0.97814092、positive edge recall 0.96928/0.97018、division parent recall 3/9。epoch2と最終artifactは未完了。B kernelは404で未開始。
- 2026-09-20 03:54 UTC: A kernel v1 COMPLETEを確認しoutput取得。2fold×3 epoch完走、train stage 2720.82秒、Notebook 3240.62秒。manifest SHA c0e70224a14de199119c0104e51c002f7b6121bc562dc4727f52703998e80e3c、source SHA e4477b188d835c269db2c61c7b62f129a41925cc8ff300ab06ca9a49a5559383、fold0/1 checkpoint SHA 3360201c84349d96b2ae4ca8a4b0d61947478e40098dac7a5f55aaaa445fdb15 / a11d8ec53a75dd1f6b576882b8fdf8b796548f7f15ab0f480b461e4970c5da50を出力と照合。外側pair指標はexp016保存済み現行より両胚でrecall・precision・division recallが低い。構造化数値はmetrics.jsonへ記録。公式graph指標は未計測。
- 2026-09-20 03:59 UTC: B専用push前のfresh quotaはGPU使用9.41/45時間、Kaggle残35.59時間、repo週30時間上限の残20.59時間、refresh 2026-09-26 00:00 UTC。B-onlyは1 variant×2fold=2 tracker学習、各3 epoch、booster 0、現行control再学習なし。前回B benchmark保守予測9.54時間はrepo残余と単Notebook12時間gate内。
- 2026-09-20 04:00 UTC: B専用kernel `kentookumura/exp025-frame-self-attention-b-train` v1 push成功。status RUNNINGを確認。Model Bのみ2fold×3 epoch、model_count=2。
- 2026-09-20 04:15 UTC: B kernel v1はERROR。papermill In[4]で `expected exactly one exp015 cache output root, found []`。約40秒で入力解決に失敗し、benchmark・学習は未開始。remote metadataにはexp015 kernel_sourceがあり、source kernel status COMPLETEを確認。Kaggle入力マウントの一時的欠損として同一設定を再pushする。
- 2026-09-20 04:19 UTC: B再push前のfresh quotaはGPU使用9.58/45時間、Kaggle残35.42時間、repo週30時間上限の残20.42時間。B 2fold×3epochの事前保守予測9.54時間はrepo残余・Notebook12時間gate内。v1はbenchmark前のマウント失敗で追加学習0。
- 2026-09-20 04:20 UTC: 同一設定のB kernel v2 push成功、status RUNNING。v1はマウント欠損による学習前停止として保存し、v2のbenchmarkと入力SHAを待つ。
- 2026-09-20 04:24 UTC: B v2もv1と同じ `expected exactly one exp015 cache output root, found []` でbenchmark前に停止。単純な再pushで解消しないため、train sourceのcache locatorに `/kaggle/input` fallbackとmount一覧の診断を追加。既存のsummary/cache identity SHA検証は維持。`make check-exp` と `make test-exp`（9件）が通過。次回v3で実際のmount配置を確認する。
- 2026-09-20 04:27 UTC: B v3診断push前のquotaはGPU使用9.59/45時間、Kaggle残35.41時間、repo週30時間上限の残20.41時間。B 2fold×3epochの保守予測9.54時間と単Notebook12時間gateは満たす。v1/v2はいずれもbenchmark前に停止。
- 2026-09-20 04:29 UTC: B kernel v3は入力locator修正を含めてpush成功、status RUNNING。cache identity SHA、source SHA、teacherとmodelは変更なし。
- 2026-09-20 04:31 UTC: B v3もERROR。package内の `.py` には修正があったが、実行 `.ipynb` は旧locatorのままだった。prepareはNotebookを再生成しないため、Jupytextでtrain `.py` から `.ipynb` を同期してから再pushする。v3もbenchmark・学習0。
- 2026-09-20 04:35 UTC: B v4 push前にPython/Notebookの全15セル一致とJupytext strict round-trip、`make check-exp` を確認。GPU使用9.60/45時間、Kaggle残35.40時間、repo週30時間上限の残20.40時間。B保守予測9.54時間はgate内。
- 2026-09-20 04:36 UTC: B kernel v4は同期済み`.ipynb`をpush成功、status RUNNING。まだcache SHA・benchmarkは未確認。
- 2026-09-20 04:57 UTC: B v4はcache root `/kaggle/input/exp015-oracle-stage-limits-inference` を発見。summary SHA 040d1f6437e27149e0e34ad4aac8bba3c8cf63dc266d33df497acd969972194c、window counts/初期tracker state SHAはAと一致。benchmark fold0/1は15.3911/15.4246秒、保守予測15,774.28/18,005.62秒、合計33,779.91秒=9.38時間で12時間gate通過。peak GPU memory 316,707,328/341,235,200 bytes。3 epoch/fold学習中。
- 2026-09-20 06:01 UTC: B kernel v4 COMPLETE。2fold×3 epoch完走、train stage 3972.83秒、Notebook 4775.60秒、best epochは両fold 2。manifest SHA 123121e4e882220bfe86af249b6712d6cfe12fa64fed1fa415bdc50d2a21e6ea、source SHA e4477b188d835c269db2c61c7b62f129a41925cc8ff300ab06ca9a49a5559383、fold0/1 checkpoint SHA ad064b9c259f4eaf2482308af7b28faaec41b22b56b32af59ef67de0b26dd2e7 / 63f3dcded84dfb33e3e8b594e9fe27cb9b88c498485e980eca44ffb1892857f8 をoutputと照合。A/B/exp016は外側window数と教師監査値が一致。胚別pair数値はmetrics.jsonへ記録。Model Bは現行比で両胚ともpositive pair precision低下・false-positive pair増加、既知edge recallとdivision parent recallは混在。公式graph score未計測。
- 2026-09-20 06:07 UTC: A/Bの学習結果と胚別pair比較をmetrics/resultへ記録し、`make validate-exp`、`make check-exp`、`make check-strategy-docs` を通過。全graph推論は新AGENTS.mdの早期診断方針により自動開始しない。ユーザー承認の一時heartbeat `exp025-model-a-b` をPAUSEDにした。採否・実験完了は未判断。
- 2026-09-20 06:09 UTC: 学習結果回収後のKaggle account GPU quotaは使用11.49/45.00時間、残33.51時間、repo週30時間上限の残18.51時間、refresh 2026-09-26 00:00 UTC。ほかのセッション使用を含むためexp025の実行時間と同一視しない。

- 2026-09-20 06:30 UTC: ユーザーの改善しない理由の考察依頼に対し、保存済みA/B fold summary、exp016初期・学習後指標、モデル初期化、loss/選択指標、添付NFL branchを再確認。初期性能の低下、epoch別内部検証、parameter数をmetrics.jsonのself_attention_diagnosticsへ記録し、観測と仮説・未検証事項をresult.mdへ追記。初期性能を維持する追加層の比較を最優先の切り分け案とした。学習・graph推論・submission・実験採否の変更なし。

- 2026-09-20: ユーザーの「これで進めてください」により、Model Bの追加Self-Attention残差を恒等初期化する `model_b_identity_init` を同実験の追加variantとして承認。実行予定は1 variant×1 config×2fold×3epoch=2 tracker学習、booster 0。保存済みexp016現行・Model A/Bを比較対象とし、control再学習なし。sourceとconfig・契約・初期logits一致テストを更新中。
- 2026-09-20 06:56 UTC: `model_b_identity_init` 専用Kaggle train packageを生成・検証。metadataは `kentookumura/exp025-frame-self-attention-b-identity-train`、T4 GPU有効、TPU・Internet無効。実行は1 variant×2fold×3epochの2 tracker学習、booster 0、control/A/B再学習なし。最新quotaはGPU使用12.57/45.00時間、アカウント残32.43時間、repo週30時間上限の残17.43時間、refresh 2026-09-26 00:00 UTC。前回Model Bの保守的学習予測9.38時間は残余・単Notebook12時間gate内。学習前の両fold実データlogits完全一致をNotebookで追加検査する。`make validate-exp`、`make check-exp`、`make test-exp` (10件)、Jupytext同期、metadata検証を通過。
- 2026-09-20 06:57 UTC: 専用kernel `kentookumura/exp025-frame-self-attention-b-identity-train` の初回pushはKaggleの `Maximum batch GPU session count of 2 reached` で拒否。Make/CLIはexit 0だったがkernel status 404を確認し、学習未開始。ほかのGPU sessionを停止せず、次回quotaとpackageを再検証してから同じ専用slugを再試行する。
- 2026-09-20 07:15 UTC: identity専用kernel statusは404で未開始。再push前quotaはGPU使用13.20/45.00時間、アカウント残31.80時間、repo週30時間上限の残16.80時間、refresh 2026-09-26 00:00 UTC。前回Bの保守的予測9.38時間と単Notebook12時間gate内。metadataはT4 GPU有効、TPU・Internet無効、package validatorとJupytext同期を再確認。
- 2026-09-20 07:15 UTC: identity専用kernelへの再pushも `Maximum batch GPU session count of 2 reached` で拒否。CLI exit0でもstatus404で未開始を再確認。他セッション停止なし。次回heartbeatで同じslugとfresh quotaを確認して再試行する。
- 2026-09-20 07:32 UTC: identity kernelはstatus404で未開始。再push前GPU quotaは使用14.00/45.00時間、アカウント残31.00時間、repo週30時間上限の残16.00時間、refresh 2026-09-26 00:00 UTC。前回Bの保守的予測9.38時間とNotebook12時間gate内。専用packageのT4/TPU無効・Jupytext同期を再検証。
- 2026-09-20 07:32 UTC: 同じ専用kernelのpushは再び `Maximum batch GPU session count of 2 reached` で拒否。CLI exit0後もstatus404を確認し、学習未開始。既存GPU sessionは停止せず次回確認。
- 2026-09-20 07:55 UTC: 専用kernel status404で未開始。fresh GPU quotaは使用14.27/45.00時間、アカウント残30.73時間、repo週30時間上限の残15.73時間、refresh 2026-09-26 00:00 UTC。前回B保守的予測9.38時間と単Notebook12時間gate内。T4 GPU/TPU無効metadataとpackage、Jupytext同期を検証。
- 2026-09-20 07:57 UTC: identity専用kernel `kentookumura/exp025-frame-self-attention-b-identity-train` v1 push成功、status RUNNINGを確認。Kaggle pull metadataでもNvidiaTeslaT4 GPU有効、TPU・Internet無効。1 variant×2fold×3epochの学習開始。学習前の実データlogits一致、cache SHA、runtime gateと出力は次回確認。source SHA ffd50870b5097183063e43ee1eeec05838b14a9b7aa865ce340861141b447670 をpackageから記録。
- 2026-09-20 08:29 UTC: kernel v1 RUNNING。live SSEでexp015 cache summary SHA一致、cache identity検査を通って学習開始、公開初期tracker state SHA bd30f577962a649bb504d7e0e3466ac773ce4e4365beb54f50f062227e8f5702、64-window/fold benchmark保守予測33,235.33秒（9.23時間）<Notebook12時間gateを確認。fold0の実データ2 window・14,450 pairで公開trackerとの初期logits最大差0.0。fold0 epoch0/1完了、epoch2とfold1待ち。
- 2026-09-20 08:46 UTC: kernel v1 RUNNING。fold0は3epochを完走し内部選択best epoch2。fold1の実データ2 window・100,800 pairでも公開trackerとの初期logits最大差0.0を確認し学習へ進んだ。両foldの初期logits同値性が成立、fold1結果待ち。
- 2026-09-20 09:21 UTC: identity専用kernel v1 COMPLETEを確認しoutputを `/tmp/kaggle-output/exp025-b-identity-v1/` へ回収。2fold×3epoch完走、best epochはfold0が2、fold1が0。実データ上で公開初期trackerとのlogits最大差0.0（fold0 14,450 pair、fold1 100,800 pair）。Notebook実行4,845.19秒、train stage 3,985.06秒。64-window/fold benchmark保守予測33,235.33秒はNotebook12時間gate内。cache summary/identity、annotation、GT window filter、feature schema、初期tracker stateはA/Bと一致した。manifest SHA 442c25536a64419d2460755a13513edd0f4aa76b39df29d954af5a92359faa4c、source SHA ffd50870b5097183063e43ee1eeec05838b14a9b7aa865ce340861141b447670、fold0/1 checkpoint SHA be990153891f15484dba2f210fc91ef4e005e69ecc0fbd8ca15abb7ab425267d / bb6150321ac5ace61d080889411ae3ee2cdc6d0f47e0eca3b4673d9bf0279e3a を実ファイルと照合し、canonical state SHAも一致。
- 2026-09-20 09:26 UTC: 保存済みexp016現行、Model A/B、identity variantを同一外側window・教師で比較。6bbaは現行比でrecall -0.286ポイント、precision -0.082ポイント、教師上false-positive pair +76、分裂親回収50/108（現行48/108）。44b6はrecall -0.939ポイント、precision +0.804ポイント、false-positive pair -177、分裂親回収4/22（現行6/22）。構造化数値はmetrics.json、解釈はresult.mdへ記録。両胚でrecall低下のため全graph推論は自動開始せず、公式graph score未計測。実験完了・採否は未判断。
- 2026-09-20 09:29 UTC: 結果記録後に `make validate-exp`、`make check-exp`、`UV_NO_SYNC=1 make test-exp`（10件）、`git diff --check` が通過。ユーザー承認済みの一時heartbeat `exp025-model-a-b` をPAUSEDにし、重複実行を停止。実験statusは `running` のまま、採否・完了はユーザー判断待ち。
- 2026-09-20 10:10 UTC: ユーザーの「他に精度が上がらなかった理由」に対し、保存checkpointのゼロ初期化射影が更新済みであること、胚別のwindow平均候補数、内部検証のFP/FN・分裂とcheckpoint選択を確認。数値をmetrics.jsonのpost_identity_diagnostics、残る仮説と切り分けの限界をresult.mdへ追記。新規学習・graph推論・実験の採否変更なし。
