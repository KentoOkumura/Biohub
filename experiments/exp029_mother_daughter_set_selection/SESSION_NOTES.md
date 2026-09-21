# exp029_mother_daughter_set_selection セッションノート

## 目的

固定候補を使い、母ごとの娘集合出力をexp028の娘ごとの母選択および保存済みexp016と比較する。

## 現在の作業

- 状態: 第5試行のtrain Notebook（Kaggle kernel version 4）はCOMPLETE。第2試行の保存済みfold 0を復元し、fold 1のみ学習・評価した。両胚で前段条件は不成立。
- 追加診断: 正解2娘が得点段階で1娘・空集合に負けることを確認。絞り込み解除でも回収0件。詳細はresult.md。
- 次: ユーザーの採否・完了判断を待つ。全graph推論とsubmissionは実行しない。

## コマンドログ

- 2026-09-20: `make new-exp EXP=exp029_mother_daughter_set_selection SOURCE=experiments/exp028_direct_graph_prediction`。親のテストはコピーせず、記録はplannedへ初期化した。
- 2026-09-20: `make test-exp EXP=exp029_mother_daughter_set_selection`。4件pass、torch依存の1件はローカル環境にtorchがないためskip。Kaggle実行時にmodel forwardとlossを確認する。
- 2026-09-20: `make check-exp EXP=exp029_mother_daughter_set_selection`。初回は新規コードの行長・import順で失敗し、対象ファイルのみ整形して再確認中。

- 2026-09-20 20:27 JST: `make validate-exp`と`make check-exp`はpass、`make test-exp`は4 pass・PyTorch依存1 skip。train/inferenceのJupytext変換とround-tripはpass。`make prepare-kaggle-notebooks EXP=exp029_mother_daughter_set_selection EXTRA_ARGS="--notebook train --run-on-push"`でtrain packageを作成した。
- 2026-09-20 20:27 JST: train metadataはGPU T4、TPU無効、internet無効。`uv run kaggle quota --format json`でGPU残29.10h、2026-09-26 00:00 refreshを確認した。実験のNotebook 12hゲートと候補監査・復号benchmarkを含めて予算内と判断し、train pushへ進む。

- 2026-09-20 20:28 JST: `make push-kaggle-train EXP=exp029_mother_daughter_set_selection`は実行前の自動承認審査で拒否された。理由はprivate NotebookでもリポジトリコードをKaggleへ送る操作に明示承認が必要というもの。迂回せずユーザーに承認を確認中。Kaggle kernelは作成しておらず、GPU学習も開始していない。
- 2026-09-20: ユーザーが`exp029_mother_daughter_set_selection`のprivate train NotebookをKaggleへpushして実行することを明示承認した。

- 2026-09-20: 明示承認後に`make push-kaggle-train EXP=exp029_mother_daughter_set_selection`を実行し、private kernel `kentookumura/exp029-mother-daughter-set-selection-train` version 1を作成した。Kaggle側metadataをpullし、T4 GPU・TPU無効・internet無効を確認した。statusはRUNNING。live logではbootstrapが完了し、候補監査の出力を待っている。

- 2026-09-20: Kaggle train v1のlive logで候補監査が2,000/18,707 windowに到達。両frameにGEFF注釈があるwindowだけを学習・2-frame診断の対象とするため、全cacheの19,701 windowより少ない。監査結果と学習成否はまだ未確定。

- 2026-09-20: Kaggle train v1は18,707件の候補監査計算後、`candidate_teacher_audit.json`の保存時に`TypeError: Object of type int64 is not JSON serializable`でERROR。GPU学習・2-frame評価は始まっていない。`make_set_supervision`と監査集計の値をPython `int`へ変換し、JSON保存の回帰テストを追加した。`make test-exp`は6 pass・PyTorch依存1 skip。v2で再実行する。

- 2026-09-20: v2 push直前のmetadata再検証はpass。`uv run kaggle quota --format json`でGPU残28.36hを確認した。v1の約0.74h消費後も、Notebook 12hゲートの範囲でv2を実行できると判断した。JSON保存回帰テストとKaggle側でのmodel/decoder早期preflightを追加した。

- 2026-09-20: Kaggle train v2をpush。live logでbootstrap完了、`MODEL_AND_DECODER_PREFLIGHT_PASS`を確認した。Kaggle上のPyTorch forward・backwardとSciPy MILPの小例は通過し、全cache監査を再実行中。

- 2026-09-20: Kaggle train v2で全18,707 windowの監査JSON保存が成功。胚別の既知2娘集合回収と既知娘候補漏れは`metrics.json`の`evidence.preflight_audit`にログ観測値として記録した。監査ファイルSHAはログ値であり、Kaggle output取得後のファイル照合は未実施。
- 2026-09-20: fold 0の学習・MILP復号benchmarkから安全係数込みの2fold見積もりが37,514.85秒（約10.4時間）で、12時間ゲートを通過。復号見積もり22,180.00秒、学習見積もり2,829.90秒。ログの`maximum_window_set_count=274667`は大きい8 windowの集合数合計であり、単一window最大値ではない。実際の最大集合数は監査で記録した44b6の35,273件。学習は進行中で、2-frame指標と公式scoreは未測定。

- 2026-09-20: ユーザーがtrain失敗を報告。Kaggle statusはERROR。make kaggle-logsでfold 1開始前のtraining_runtime_gate_seconds=59272.02を確認した。v2の見積もりはfold 1時点でも2foldを掛け、完了済みfold 0を再計上する欠陥があった。
- 2026-09-20: make kaggle-outputでv2 outputを取得。候補監査SHA e52fefd1718adca0907707acda4b1a65350ea3deaaced03d130de74ec7b72c2e、fold 0 summary SHA 7bb4f30bccf2d54fcc0dba9924a0126c2da496e486ee846d0bd03cb2ff8de764、model SHA 85318b8c64c1b102e4452e885619514a31721cc64188237dea3b12715d2807a0を照合した。ローカル保存先はartifacts/v2_resume/。Kaggle outputのmetrics.jsonはNotebook実行前の古いスナップショットなので実験の正本へ上書きしていない。
- 2026-09-20: fold 0の6bba外側胚では、既知edge recallが92.02%対exp016の97.20%、観測可能な誤接続率が6.78%対2.94%、既知分裂母が0/108対50/108、構造違反0。既知edge・誤接続・分裂の3条件が不成立。これだけで両胚成立の必要条件は満たせず、全graphは自動実行しない。公式scoreは未計測。
- 2026-09-20: 実行時間見積もりを残りfold数だけに修正。v2の候補監査・fold 0 summary・モデルをSHA照合してv3で再利用し、fold 1だけを学習する。make check-expとmake validate-expはpass、make test-expは8 pass・PyTorch依存1 skip。train Jupytext round-tripはpass。GPU quotaはv3準備時に残24.04h。

- 2026-09-20: private train v3 packageは候補監査JSON 1,943 byte、fold 0 summary JSON 32,010 byte、モデル重み 3,567,226 byteを含み、各SHAとmetadataを検証済み。修正後のfold 1残り時間見積もりはv2 benchmarkから約29,636秒（約8.23時間、安全係数込み）で、Kaggle未検証。make push-kaggle-trainは自動承認審査で実行前に拒否された。以前の承認はNotebook本体だけで、保存済み監査・結果・重みのKaggle送信に明示承認がないため。v3 kernelはまだ作成されていない。迂回せずユーザー確認待ち。

- 2026-09-20: ユーザーが候補監査JSON、fold 0結果JSON、学習済み重みを同梱するprivate train v3のKaggle送信・fold 1実行を明示承認。make push-kaggle-trainは成功しversion 3を作成。Kaggle status RUNNING。pullしたmetadataでprivate、T4 GPU、internet無効、exp015/exp016入力を確認した。

- 2026-09-20: v3 live logでMODEL_AND_DECODER_PREFLIGHT_PASS後、resume/candidate_teacher_audit.jsonがKaggle workingに存在せずFileNotFoundError。Kaggle pushは手動追加したpackage内resume/をNotebook実行へ渡さなかった。v3ではfold 1学習も監査再利用も始まっていない。承認済みの同一3生成物をruntime.kaggle.train.bootstrap_dependency_filesへ設定し、Notebook本体のzip bootstrapにSHA付きで同梱するv4を準備。

- 2026-09-20: v4 pushはKaggle API 400。API応答はThe kernel source must be less than 1 megabytes in size.で、同梱後のNotebookは5,532,398 byte。v4 kernelは作成されていない。v2生成物3件のSHAを保ち、private Kaggle DatasetからNotebook入力として読み込むv5へ変更する。Datasetはローカルで準備したが、まだ作成・送信していない。

- 2026-09-20: v5のprivate Dataset packageは監査JSON 1,943 byte、fold 0 summary JSON 32,010 byte、重み 3,567,226 byteのみ。SHAをv2 outputと照合し、metadataのisPrivate=trueと既存同名Datasetなしを確認した。Notebookは170,090 byteでKaggle 1 MB制限内。実生成物を使ったdataset mountのローカル検証、make check-exp、make validate-exp、metadata検証、Jupytext round-tripはpass。make test-expは9 pass・PyTorch依存1 skip。Datasetは未作成、v5 kernelも未作成。

- 2026-09-20: ユーザーがprivate Dataset作成とv5実行を明示承認。Dataset kentookumura/exp029-mother-daughter-set-selection-resume-v2を作成し、Kaggle status ready、metadata isPrivate=true、3ファイルのみを確認した。ダウンロードし直した3ファイルのSHAがv2 outputと一致。Notebook code sourceは約170 KB。v5はfold 1だけを実行し、v2のfold 0を再利用する。

- 2026-09-20: make push-kaggle-trainは成功。v4 pushは拒否されversionが作成されなかったため、第5試行はKaggle kernel version 4。metadata検証後にpushしstatus RUNNING。private Datasetを入力に追加し、Notebook sourceは170,250 byte。

- 2026-09-21 JST: Kaggle kernel version 4のlive logでMODEL_AND_DECODER_PREFLIGHT_PASS、RESUME_DATASET_FILES_VERIFIED、CANDIDATE_TEACHER_AUDIT_REUSED、FOLD_RESUMED fold 0を確認。fold 1の残り実行時間見積もりは28,522.14秒（約7.9時間、安全係数込み）で12時間ゲートを通過し、status RUNNING。2-frame結果は未測定。

- 2026-09-21 02:00 JST: Kaggle kernel version 4はCOMPLETE。make kaggle-outputで出力と保存ログを取得し、make kaggle-logs -fでも完了ログを確認した。resumeの監査・fold 0 summary・fold 0重みと出力側のfold 0重みは第2試行のSHAと一致。fold 1 summary SHA 49c78da22fb12c051302b7b4c26ba9e17ad9198f66451c68dbd587399d968edb、fold 1重み SHA 04da9548b6152882a28d07e95714a6121cd196974e510c4cf4480bd854dea589、early gate SHA 4258e18a7ca89441e201ba9c76ab35094825303d9ea90bc416c7e585920f1014、model manifest SHA 4c7f440e079147f34e68305be08992d1e1a11e24544495144bd5570824c7b51dを確認し、相互の参照とログ値も一致。生成物とログをartifacts/train_v4/へ保存した。Notebook実行時間は5708.21秒。
- 2026-09-21 02:00 JST: fold 1の44b6外側胚は既知edge recall 94.11%対exp016の95.30%、観測可能な誤接続率4.84%対3.61%、既知分裂母0/22対6/22、構造違反0。6bbaとともに既知edge・誤接続・分裂の前段条件が不成立。Kaggle出力のmetrics.jsonはkernel version 3 / READY_TO_PUSHを含む古いスナップショットから始まるためローカル正本へ上書きせず、検証したtrain_stageとfold 1証拠だけを統合した。statusはdebug_completed。全graphと公式scoreは未計測。

- 2026-09-21 08:49 JST: ユーザーの「得点付け・絞り込み・全体最適化のどこで分裂が選ばれなくなるか調べてください」を受け、prepare_division_diagnostic.pyとdiagnose_divisions.pyを追加。既存GEFF exportの全199 sample・4,179 fileのSHAを確認して隣接分裂151組を抽出し、exp015の該当151 NPZとsummaryだけ取得した。cacheは固定identityとsummary SHA、全NPZのschema/content SHAを照合した。APIのversion_label指定は404だったため正規slugの現出力を取得して固定SHAで同一性を確認し、一覧取得中の429には待機と途中再開を追加した。
- 2026-09-21 08:49 JST: 次のローカル診断を実行した。保存済みモデルをCPUで再評価し、元のKaggleコードとローカルsourceのSHA一致も確認した。torch 2.11.0+cpu、4 threads、診断処理224.75秒。GPU再学習・Kaggle pushは行っていない。

```bash
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run --frozen python experiments/exp029_mother_daughter_set_selection/prepare_division_diagnostic.py --geff-archive /tmp/exp029-division-geff/exp032_train_geff.zip --download
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run --frozen --offline --with torch==2.11.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match python experiments/exp029_mother_daughter_set_selection/diagnose_divisions.py
```

- 2026-09-21 08:49 JST: 既知分裂108/22、正解集合候補107/21、通常復号の回収0/0を再現。絞り込み後は93/16件が残るが、候補内の全128件で1娘または空集合の方が高得点。最大37集合を許す復号でも回収0/0。学習側の候補内既知2娘117件も全件回収できない。summary SHA e43394baa60f7acc10cf5d50db8ba2a2dac2abc5c369021d47189e9a3a29158c。詳細はmetrics.jsonのdivision_diagnosticとresult.mdへ記録した。分裂教師の少なさとaccuracyによるcheckpoint選択は根拠のある原因候補だが、再学習による因果の切り分けは未実施。

- 2026-09-21: 追加診断の既知2娘逆伝播テストを含め、CPU版PyTorch環境で対象実験の14 testsがpass。make check-exp、make validate-expもpass。実験の学習設定・保存済み重み・採否statusは変更していない。

## 実行予定と資源

- train variant 1、model config 1、fold 2、booster 0。control再学習なし。exp016の保存済みfold別モデルを再推論する。
- 候補集合とMILP復号の小規模実測で12時間の実行条件を判断する。週30 GPU時間の残量はpush直前に確認する。
- `make validate-exp EXP=exp029_mother_daughter_set_selection`
- `make check-exp EXP=exp029_mother_daughter_set_selection`
- `make test-exp EXP=exp029_mother_daughter_set_selection`
- `make prepare-kaggle-notebooks EXP=exp029_mother_daughter_set_selection EXTRA_ARGS="--notebook train --run-on-push"`
- `make push-kaggle-train EXP=exp029_mother_daughter_set_selection`

## 次のアクション

両胚の同じ2-frame対象の比較は完了。両胚で前段条件が不成立のため、全graphと公式scoreの実行は保留する。実験の採否・完了はユーザー判断を待つ。
