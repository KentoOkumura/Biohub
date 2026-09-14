# exp015_oracle_stage_limits セッションノート

## 目的

固定公開モデルのtrain 199動画予測について、既知中心、既知辺、母と2娘、最終graph選択の回収上限を分ける。exp012で固定した悪化16件を5条件別に段階分解し、全199件および44b6全件と比較する。

## 現在の作業

- 作業内容: Kaggle inference version 1とCPU diagnostic version 6が完走し、cache、段階別結果、入力・出力SHAをmetrics.jsonとresult.mdへ記録済み。
- ブロック要因: なし。
- 次: ユーザー判断によりdiagnosticとして完了した。全体バックログから後続実験を選ぶ。重いinference outputはKaggle側だけに保持する。

## GPUコストガード

- active variant数: 1。
- model/config数: 固定primary、secondary、DeepCenterを組にした1構成。
- fold学習数: 0。
- booster数: 0。
- control再学習: なし。
- train対象: model学習はなし。固定modelによるtrain画像の診断推論と、後続学習用feature cache生成を同時に行う。
- exp014のpublic 4動画実測を単純比例した参考値: 予測wall time約8.25時間、Notebook全体約8.5〜9時間、GPU quota消費も約8.5〜9時間。T4 2基でも台数分を二重加算しない。train 199件の保証値ではない。
- cache容量見積もり: 6,081,397,663 bytes。各GPU shard 6GB、cache合計12GB、/kaggle/working全体18GBで停止し、Kaggle output上限20GBに余裕を残す。
- 実測: GPU予測処理23,994.20秒、cache 4,151,337,848 bytes、Kaggle working outputはmanifest前4,813,320,625 bytes。全ての時間・容量guard内で完走した。
- 実行開始後にquota残量を使い切っても12時間までは継続するというKaggle運用を前提にする。push直前に仕様変更がないか再確認する。
- Kaggle submission: 行わない。

## 作業ログ

- 2026-09-13: oracle_stage_limits の状態が 設計可能・実験化未承認、未決事項なしであることを確認し、ユーザーの実装依頼を実験化承認として扱った。
- 2026-09-13: taskコマンドが利用できないことを確認し、make new-expでexp013をsourceとしてexp015を作成した。lineageの親は同一pipelineとcache等価性の直近証拠であるexp014へ更新した。
- 2026-09-13: 作業ツリーのviewer関連変更は別件として保持し、変更対象から外した。
- 2026-09-13: apply_patchがbubblewrap欠損で利用不能だったため、対象を新規exp015配下に限定した検証付き置換へ切り替えた。
- 2026-09-13: inference sourceをtrain 199件へ切り替え、build_graph直後のILP前候補GEFF、repair後の元ID付きNPZ、ground_truth_accessed=falseのmanifestを追加した。
- 2026-09-13: diagnostic sourceへexp012固定SHA、悪化16件再計算、時刻別一対一対応、edgeとdivisionの累積判定、4 scopeの条件別集計、brightness x candidate density集計、failure・NaN、manifestを実装した。
- 2026-09-13: 初期実装についてinference / diagnostic のJupytext round-trip、strict validate-exp、Ruff、実験固有test 8件を確認した。
- 2026-09-13: inferenceはT4・diagnosticはCPUとしてKaggle packageをstrict生成し、internet無効、competition source、2つのkernel sourceを確認した。Notebookのpush・実行・submissionは行っていない。
- 2026-09-13: diagnostic再実行時も、ユーザー判断済みのusable / completed / deprecated / discarded / leak-risk statusを上書きしない保護を追加した。
- 2026-09-13: backlogの契約移行を確認後、oracle_stage_limitsの未着手行と詳細ファイルを削除し、HYP-20260910-14の対応実験および依存候補をexp015へ更新した。check-strategy-docsは成功した。
- 2026-09-13: metrics statusをscaffold_completedへ更新し、update-summaryでexperiment_summary.mdへ反映した。
- 2026-09-13: ユーザー依頼により、exp015 inferenceの同じdetector forwardからtrain 199動画・19,701 windowのcacheも保存するよう拡張した。各NPZは書き込み直後に完全一致を検査し、現在の予測はin-memory tensorを使うため、cache検証用のedge score再計算は追加していない。
- 2026-09-13: exp014の実測122,239,149 bytes / 396 windowからcacheを6,081,397,663 bytesと見積もり、各GPU shard 6GB、cache合計12GB、全output 18GBのguardを実装した。
- 2026-09-13: cache追加後のinference / diagnostic Jupytext round-trip、strict validate-exp、Ruff、実験固有test 10件を確認した。inference / diagnostic packageをstrict再生成し、inference packageにwindow_cache.py、T4、internet無効が含まれ、diagnosticはCPU、internet無効を維持することを確認した。Kaggleへのpush・実行は行っていない。
- 2026-09-13: ユーザーがKaggle実行を承認した。push前にCLI認証、strict validate-exp、Ruff、実験固有test 10件を再確認した。
- 2026-09-13: inference metadataはGPU有効、TPU無効、NvidiaTeslaT4、internet無効。Kaggle GPU quotaは5.77時間残、30時間中24.23時間使用、refreshは2026-09-19 00:00 UTCだった。推定wall time 8.5〜9時間は残量を上回るが、quotaが残る間に開始したjobはquota枯渇後も12時間上限まで継続するというユーザー確認済みの運用、および12時間・20GBの実装guardに基づきpush可と判断した。CLI 2.2.4ではactive session数を取得できないためpush前gateにせず、同時session上限エラー時だけ停止する。
- 2026-09-13T22:50:06+09:00: canonical kernel kentookumura/exp015-oracle-stage-limits-inference をpushし、version 1を開始した。Kaggle側id_noは134210702。pullしたmetadataでGPU有効、TPU無効、NvidiaTeslaT4、internet無効を再確認した。
- 2026-09-14T09:31:18+09:00: CLI status COMPLETEと保存済みlogでinference version 1の完走を確認した。prediction処理は23,994.20秒、sample/candidate graph/final graphは各199件、cacheは199 dataset・19,701 window・4,151,337,848 bytes、全windowのarray round-trip完全一致、edge logits再計算なし、in-memory featureによる予測継続、総outputはmanifest前4,813,320,625 bytes、peak GPU memoryは688,422,400 bytes。cache summary SHAは040d1f6437e27149e0e34ad4aac8bba3c8cf63dc266d33df497acd969972194c、oracle inference manifest SHAは5a511b8d5c8b25f6cab76563a32a63cde8b1bd74b83d5d38cacd64034396b28e。
- 2026-09-14: full output downloadを開始したが、ユーザー希望により直ちに中断した。中断までに作成されたpartial 170ファイル・4.1MBは対象pathを確認して削除し、ローカルにinference outputを保持していない。以後はKaggle log、Kaggle file metadata、Kaggle kernel sourceを使う。
- 2026-09-14: CPU diagnostic canonical kernel version 1（id_no 134271528）を実行した。Kaggle processはCOMPLETEになったが、summaryはall 199件とdegraded 16件の全件がevaluation_failed、valid sample 0だったため診断は失敗と判断した。
- 2026-09-14: failureがcatchされてもkernelが成功終了していた問題を修正し、summaryへ先頭10件のfailure_reasonを出し、failure_countが1以上ならRuntimeErrorでNotebookを失敗終了するよう変更した。Jupytext round-trip、strict validate-exp、Ruff、実験固有test 10件は成功した。
- 2026-09-14: CPU diagnostic version 2で先頭10件を含む全199件が ModuleNotFoundError: No module named tracksdata で失敗したことをKaggle logから特定した。graph/cache欠損ではなくdiagnostic runtime dependencyの不足だった。
- 2026-09-14: inferenceで利用実績のあるpilkwang/biohub-tracking-support-pack-50ep-v1をdiagnosticのdataset sourceにも追加した。internetは無効のまま、同datasetのoffline wheelsからtracksdata、GEFF、Zarr関連packageを--no-depsで導入し、importを再検証する処理を追加した。Jupytext round-trip、strict validate-exp、Ruff、実験固有test 10件は成功した。
- 2026-09-14: CPU diagnostic version 3はoffline wheel directoryを正しく解決したが、Kaggle CPU imageの既存PolarsがFloat16未対応でtracksdata importに失敗した。
- 2026-09-14: inferenceのdependency recoveryと同様に、Polarsとpolars-runtime-32をoffline wheelから--force-reinstallし、import済みgraph moduleをpurgeした後でtracksdata/GEFF関連packageを導入・再importするよう修正した。Jupytext round-trip、strict validate-exp、Ruff、実験固有test 10件は成功した。
- 2026-09-14: CPU diagnostic version 4はPolars refreshには成功したが、残りのgraph package導入後にPolarsまでpurgeしたことでFloat16対応runtimeを保持できず失敗した。
- 2026-09-14: inference実績コードと同じ順序へ合わせ、Polars refresh直後だけPolars moduleをpurgeして再importし、その後のgraph package導入ではPolarsをpurge・再指定せず、tracksdata等だけをclean importするよう修正した。Jupytext round-trip、strict validate-exp、Ruff、実験固有test 10件は成功した。
- 2026-09-14: CPU diagnostic version 5はoffline package導入と全199件の評価に成功し、failure_count 0、degraded 16件も16件全て有効だった。全199件のcandidate node recall 0.992859、candidate edge recall 0.948286、final selected edge recall 0.916017、candidate division triplet recall 0.403974、final selected division recall 0.099338をKaggle logで確認した。
- 2026-09-14: 重いoutputをローカル取得せず実行証拠を記録できるよう、input bundle SHA、inference manifest SHA、exp012 SHA、各出力表SHA、oracle manifest SHA、elapsed secondsをDIAGNOSTIC_RECEIPTとしてstdoutへ出し、4 scopeの集計もsummaryへ含めるよう追加した。計算方法は変更していない。Jupytext round-trip、strict validate-exp、Ruff、実験固有test 10件は成功した。
- 2026-09-14: CPU diagnostic version 6が423.65秒で完走し、Kaggle status COMPLETE、199/199件有効、failure 0を確認した。input bundle SHAは0d7f66e4ea6ac496dd63bf200a9612c33067908166bc279c52aab09a3d0e9fa1、oracle manifest SHAは997e578b73ee7c8b098dd1049828a779022476c8beb9c4a9367c7f20f28138c5。Kaggle file metadataで必須7出力の存在も確認した。
- 2026-09-14: metrics statusをdebug_completedへ更新した。result.mdへ全199件、44b6全71件、degraded 16件の段階別結果と限界を記録した。ユーザー判断前のためcompletedには変更していない。
- 2026-09-14: kaggle-strategyの手順で全体方針とP0〜P2候補を再点検した。P1はfrozen_image_encoder、P2はpartial_edge_mask、division_triplets、sparse_motion_graph、graph_cost_scaleを維持した。division_tripletsには現行candidate edge外から既存中心候補を組み合わせる範囲を未決事項として追加した。大容量生成物は後続もKaggle kernel sourceから直接参照する。
- 2026-09-14: ユーザーのGit commit・push依頼をdiagnosticとしての完了判断として記録し、metrics statusをcompletedへ更新した。

## 検証コマンドと次の実行

taskがないため同名のmake targetを使う。

- 実行済み: PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache JUPYTER_DATA_DIR=/tmp/jupyter-data uv run --extra notebook jupytext --to ipynb --test experiments/exp015_oracle_stage_limits/exp015_oracle_stage_limits_inference.py experiments/exp015_oracle_stage_limits/exp015_oracle_stage_limits_diagnostic.py
- 実行済み: make validate-exp EXP=exp015_oracle_stage_limits
- 実行済み: make check-exp EXP=exp015_oracle_stage_limits
- 実行済み: make test-exp EXP=exp015_oracle_stage_limits
- 実行済み: make prepare-kaggle-notebooks EXP=exp015_oracle_stage_limits EXTRA_ARGS="--notebook inference --run-on-push"
- 実行済み: make prepare-kaggle-notebooks EXP=exp015_oracle_stage_limits EXTRA_ARGS="--notebook diagnostic --run-on-push"
- 実行済み: make push-kaggle-infer EXP=exp015_oracle_stage_limits。inference version 1はCOMPLETE。
- 実行済み: make push-kaggle-notebook EXP=exp015_oracle_stage_limits NOTEBOOK=diagnostic。debug後のversion 6はCOMPLETE。
- 実行済み: Kaggle logs、status、files metadataからcache、段階別集計、SHA、必須出力を確認した。大容量outputはdownloadしていない。
- 次: 完了記録を検証し、exp015関連の変更だけをcommit・pushする。

## 変更点

- 予測parameter、公開model、候補生成、ILP、graph repairはexp014の固定構成から変更しない。
- 推論対象をcompetition trainへ切り替え、診断に必要なILP前候補graphと最終graphに加え、同じforwardから後続学習用window cacheを保存する。
- GTは別のCPU diagnosticでのみ読み、候補生成や最終graphへ戻さない。
- exp012の条件と悪化16件は固定SHAから再構成し、外側結果に合わせた再選択を行わない。
