# exp040_trackastra_association セッションノート

## 目的

固定候補・固定画像特徴から6時点の対応モデルを学習し、保存済みexp016 primary単体と共通の隣接ペアで比較する。

## 現在の作業

修正版の学習・両胚のペア診断・結果回収は終了した。[結果とユーザー判断](result.md)に従い、全graph推論を保留し、実験の採否・完了判断を待つ。

## 2026-09-23 実装と初回診断の記録

- 2026-09-23: ユーザーが確定設計の実装を依頼。候補詳細を実験契約へ移し、exp040を作成。
- 固定cacheとGEFFの読込、部分教師、公式Trackastra由来モデル、親方向確率、重複窓の予測集約、対照の共通ペア診断を実装。
- 学習Notebookに教師監査、2窓の学習成立確認、64窓/foldの費用推定、10 epochs×2fold、内部checkpoint・閾値選択、両胚の進行条件、生成物SHA記録を実装。Kaggle version 1のpreflightを実行し、費用gateで本学習前に停止した。
- 全graph推論と公式評価は、隣接ペア診断の進行条件を満たした後に判断する。Kaggle submissionは作成しない。

## コマンドログ

- 2026-09-23: `make new-exp EXP=exp040_trackastra_association`。
- 2026-09-23: `make test-exp EXP=exp040_trackastra_association`。合成データの6テストが成功。
- 2026-09-23: `make validate-exp EXP=exp040_trackastra_association`。実験構造はstrictで成功。
- 2026-09-23: `make check-exp EXP=exp040_trackastra_association`。lintとformatが成功。
- 2026-09-23: `uv run --extra notebook jupytext --to ipynb --test experiments/exp040_trackastra_association/exp040_trackastra_association_train.py`。往復変換が成功。
- 2026-09-23: `make prepare-kaggle-notebooks EXP=exp040_trackastra_association EXTRA_ARGS="--notebook train"`。train用Notebookとofflineのinput sourceを生成・確認。`run_on_push=false`であり、push・Kaggle実行は未実施。

- 2026-09-23 10:30 JST: push直前に`uv run kaggle quota --format json`を確認。GPU週次45.00時間中の残り12.78時間、refreshは2026-09-26T00:00:00。train NotebookはT4 GPU、TPUなし、internetなし、上限12時間、`run_on_push=true`。学習するactive variantは1、model/configは1、foldは2、生成modelは2、boosterは0。保存済みexp016 controlを読むだけで再学習なし。Notebook内の費用推定が12時間を超えれば学習前に停止する。

- 2026-09-23 10:33 JST: `make push-kaggle-train EXP=exp040_trackastra_association`でKaggle Notebook `kentookumura/exp040-trackastra-association-train` version 1をpush・実行開始。Kaggle側のpull metadataで`id_no=135452759`、T4 GPU、TPUなし、internetなしを確認。live logでbootstrapと固定モデル設定の出力を確認。入力cache・教師監査・費用gateは進行中。

- 2026-09-23 10:57 JST: Kaggle train version 1は両foldの教師監査、2窓の100 update学習成立確認、各64窓benchmarkを完了。保守的な2fold予測`metrics.json.preflight.projected_seconds`は363591.32秒で12時間gateを超え、Notebookは本学習前に意図したRuntimeErrorで停止（Kaggle status `ERROR`）。学習epoch 0、model 0、外側胚のpair診断なし、公式scoreなし、submissionなし。`make kaggle-output KERNEL=kentookumura/exp040-trackastra-association-train OUT=/tmp/kaggle-output/exp040-trackastra-v1`で`preflight.json`を回収し、`artifacts/preflight_v1.json`へ保存、SHA-256 `aeb393b6605efd16e2888501c23b220caaa140907d2ce0d8d49765d383eedd43`を照合。停止後quotaはGPU週45.00h中12.40h残（開始前12.78h）。

## 2026-09-23時点の次のアクション

ユーザーが選んだ学習窓抽出を毎epoch各fold4,096窓へ固定し、12時間gateを準備した。週30時間GPU方針の確認後に本学習・共通pair診断へ進む。全graphは単体結果の進行条件とユーザー判断を待つ。本学習・共通pair診断・全graphは未実施。

- 2026-09-23 12:54 JST: ユーザーが学習窓の抽出を選択。全窓の教師とtoken数、密度帯別の学習step・validation・推論、および保存済み対照の推論を測るprofiling-only version 2を準備。6時点・幅128・各4層・10 epochs×2foldはこの段階では変更しない。active variant 1、model/config 1、fold 2、本学習model 2、booster 0、対照再学習なし。profile-only実行は診断用modelを作るが、本学習epoch 0のまま停止する。push前のmetadataはprivate、T4 GPU、TPUなし、internetなし、12時間上限、run-on-push有効。Kaggle quotaは週45.00h中残12.40h、refresh 2026-09-26T00:00:00。短時間の計測を行い、測定結果から抽出数と総費用を判定する。
- 2026-09-23 12:55 JST: make push-kaggle-trainで同じcanonical kernelのversion 2を実行。Kaggle pull metadataでid_no=135452759、T4 GPU、TPUなし、internetなしを確認。profiling_only=trueにより全学習へは進まない。
- 2026-09-23 13:01 JST: quota表示の残12.40hとは別に、実行前の週次GPU使用量32.60hが現行の週30h方針を既に超えていたと確認。事前判断の見落としとして記録する。version 2の短時間計測の結果を回収した後、追加のGPU本学習は方針と費用の確認なしに開始しない。
- 2026-09-23 13:25 JST: Kaggle train version 2は両foldの教師・2窓fitを再確認し、各密度帯12窓と最大密度8窓の処理別計測を完了。profiling_only_complete_before_full_trainingの意図したRuntimeErrorで停止し、本学習epoch 0、model 0、共通pair評価なし。make kaggle-outputで計測ファイルを回収し、artifacts/profile_v2へ保存した。runtime_profile.jsonのSHA-256はbff29344da239046b49f22d096b0df49df6989393733354f5fdb1c52f63eaabd。fold 0/1の窓manifest SHAと件数・分裂窓件数を照合した。
- 2026-09-23 13:39 JST: analyze_runtime_profile.pyで全密度帯の実測を学習・内部検証・候補推論・保存済み対照推論・動画読込へ分離して積算。各fold4,096窓/epoch、10 epochs×2foldの予測は35,182.98秒（1.5倍の余裕と1時間の固定予備込み）。最大可能な5,517窓/epochでは43,058秒となり12時間枠に近いため、4,096を採用した。解析JSONのSHA-256はde5789f2315afb7e9fcecf1c8b4303496f228913a0c1cbdfcf86b69107643b62、配布用小規模assetのSHA-256は201f4d6543552eb86535a38f9c344899a7d325728c9d8dd30a5ba751ac7f2226。窓抽出の教師種別・密度帯・epoch間循環と学習manifestを実装し、合成テスト8件、lint、実験構造検査に通した。新しいNotebookはまだpushしていない。
- 2026-09-23 13:39 JST: profile後quotaはGPU週45.00h中33.10h使用、残11.90h、更新2026-09-26T00:00:00。残量上は予測9.77hの学習に足りるが、現行方針の週30hを超過している。追加GPU実行は開始しない。
- 2026-09-23 13:47 JST: 保存済み教師manifestから10 epochs分の窓抽出をローカルで再現し、各foldで毎epoch4,096窓、10 epochs中の教師あり窓全件の被覆を確認。抽出manifest SHA-256はfold 0が7d538e5c50d9e3d4176f070ff987867bdaa707ec84b76c2dab0562bb7d1db4a6、fold 1がaec89874a805c6c0564cbf7f6a6765468031c3a428fabe54b364a3ebfce8ec10。configとNotebookで同じSHAを固定し、不一致なら学習前に停止する。最終のmake check-exp、make test-exp（9件）、make validate-exp、Jupytext round-trip、Kaggle package prepare・metadata validationが成功。pushと本学習はGPU週次方針の判断待ち。

- 2026-09-23: ユーザーがKaggle GPU不足を理由にexp040の実行先をColabへ変更するよう指示。Kaggle版のモデル・教師・窓スケジュールを維持し、Colab CLI T4、Kaggleからのexp015 cache・GEFF直送、保存済み対照の小規模bundle、epoch単位のSHA付き再開state、最終結果回収を実装。Colabの本学習はこの記録時点では未実施。
- 2026-09-23 23:02 JST: exp040 Colab CLI本学習の起動を試みたが、Colabセッション割当前に自動承認審査が拒否。非公開exp015 cache・GEFF等のColab宛転送について個別の承認と転送先の信頼性が確認できない、との理由。GPU学習epoch 0、Colabセッション0、転送0。コード・設定・bundleのローカル検証は継続し、具体的な転送対象を示してユーザー承認を求める。
- 2026-09-24 00:14 JST: ユーザーがexp040入力のColab転送とT4本学習を明示承認。Colab CLI session `exp040-trackastra-1790174893`を起動。exp015出力19,702ファイル・4,151,339,416 bytesをKaggleからColabへ直接取得し、GEFF ZIP 2,387,014 bytesのSHAと199サンプルを照合。Colab Tesla T4でcache 19,701 pairs、対照manifest・source、両fold教師監査が固定値と一致。runtime gateの予測は36,382.59秒/43,200秒、抽出schedule SHAはconfig固定値と一致し、本学習を開始。active variant 1、model/config 1、fold 2、model 2、booster 0、対照再学習なし。
- 2026-09-24 01:06 JST: fold 0のepoch 1/2/3が完了し、内部masked lossは0.875590→0.576665→0.561542、最良checkpointはepoch 3。各epochの約19.7 MBの再開stateをreceiptのbyte数・SHAで検証してローカル回収した。epoch 3回収時にColab CLIの古いproxy tokenがファイル操作へ404を返したが、同一T4割当の現在のruntime_proxy_infoを取得してローカルCLI session stateを更新し、ファイルアクセスと回収を復旧。学習プロセスは再起動していない。
- 2026-09-24 02:39 JST: Colab fold 0の10 epochsが完了。各epoch 4,096窓、全10件のreceiptと再開stateを順次SHA検証してローカル回収し、最終stateを保持。内部masked lossの最小はepoch 3の0.561542、最終epochは0.973135で、内部検証のみで選んだcheckpointはepoch 3。外側胚診断は全fold学習後に行うため、精度・graph進行条件・公式scoreはまだ未判定。実行中にColab CLIのファイル操作用proxyが再度期限切れになったが、同じT4割当へ接続を更新し、学習は再起動していない。
- 2026-09-24 04:38 JST: Colab T4で固定した10 epochs×2foldの学習が完走。fold 0の内部masked loss最小はepoch 3の0.561542、fold 1はepoch 8の0.120992で、それぞれ学習側の内部検証のみでcheckpointを選択。両fold全20件のepoch receiptとSHA検証済み再開stateを順次ローカル回収した。外側胚の全共通ペア診断と最終archive回収は進行中であり、graph進行条件・公式score・実験の採否は未判断。
- 2026-09-24 05:19 JST: fold 0の外側胚6bba共通pair診断が完了。内部検証で固定したTrackastra閾値0.9732687473を適用。既知edge recallは保存済みexp016 primary 0.969408に対してTrackastra 0.653536、正解親1位率0.979235対0.886501、分裂親回収48対21、legacy教師負例予測4,355対6,314、known教師負例予測14対91。事前指定の5条件は全て未達で、両胚進行条件は成立しない。fold 1の外側診断と最終archiveは継続して回収し、全graph推論には進まない。
- 2026-09-24 07:54 JST: Colab T4の本学習・両胚共通pair診断が完了。Notebook本体の実測25,447.14秒。fold 1の外側胚44b6は内部閾値0.9855320454で、既知edge recallがexp016 primary 0.947754に対しTrackastra 0.472743、正解親1位率0.968442対0.900100、分裂親回収6対2。legacy教師負例予測1,341対611、known教師負例予測4対1は非増加だが、進行条件は3項目未達。fold 0も全5項目未達のため、全graph推論へ進まず、公式score・submissionは未実施。
- 2026-09-24 07:54 JST: 最終archive 1,377,076,381 bytes、SHA-256 `e0f7bc1ac9a00c0fa25843255685ccaabc8d727f322c862009ce8a701d69ed98`を受領票と照合し、ColabへACKを送信。未圧縮診断行が4.55 GBあり、既存runnerの一括ZIP member読込ではローカルメモリ3.8 GiBを超えるため、取得済みarchiveを保護してrunnerだけを停止。メモリ一定のストリーム方式で全38ファイルを展開し、archive全体、完了marker、model manifest、2モデル、20 epoch receiptを検証した。Colabセッション`exp040-trackastra-1790174893`は成果物回収後に停止。runnerはストリーム展開へ修正し、回帰テストを追加。実験statusはユーザー判断前の`debug_completed`。
- 2026-09-24 07:59 JST: 最終archiveの展開後、外側胚スコアファイルのSHA-256を独立に再計算し、fold 0 `28a99bcc4fd056dfed283f10e7661ee71d9dd1a0172305f1b9175ae1df4ca09d`、fold 1 `d793d9f092919b3555f51231df745329ffad4ea4a6b8fcbb6056341be190a979`が保存済み診断記録と一致した。`make check-exp EXP=exp040_trackastra_association`、`make validate-exp EXP=exp040_trackastra_association`、`make test-exp EXP=exp040_trackastra_association`（14 passed）が成功。`metrics.json`の実験statusを`debug_completed`へ変更し、Colab証拠・両foldの診断を反映した。採否と実験完了はユーザー未判断。

- 2026-09-24 09:13 JST: ユーザーの低精度懸念を受け、exp040の学習・評価コードと保存済み全ラベル付き予測を監査。Notebookと共通モジュールの62関数・クラスはAST一致、対照の外側胚recall・分母はexp016の保存済み値と一致。全正例を独立抽出した内部閾値での回収件数はfold 0が67,571 / 103,393、fold 1が8,958 / 18,949で、保存済みmetricsと一致。診断閾値0.5でもTrackastraは89,931 / 103,393、16,897 / 18,949で対照を下回り、legacy mask負例予測は33,273対4,355、4,861対1,341。
- 2026-09-24 09:13 JST: 旧masked BCEの確率clampで正例確率が1e-7未満になると主損失の勾配が途切れることを最小例で再現。旧予測の該当既知正例はfold 0が95件、fold 1が4件。log確率・log補確率を使う同値な数値安定計算へ共通モジュールとJupytext学習ソースを修正し、Notebookを再生成。保存済みモデル・予測は旧損失のままで、修正後Colab再学習は未実行。statusはdebug_completed、graph評価・submissionなし。
- 2026-09-24 09:20 JST: Colab epoch再開のidentityへ学習ソースSHAを追加し、変更されたソースで旧stateを読み込まない回帰テストを追加。修正版sourceとconfigの小規模Colab bundleを再構築し、全memberをbyte数・SHAで検証。bundleは旧metricsの学習結果・監査値を除いた初期状態で、4,382,773 bytes、SHA-256 fd0c06333838f6407b48f5e6034c3051eb417be417d1dcacf8b1fc6d2c09cbdf。旧run IDの保存済み成果物は上書きせず、新しいGPU実行は未開始。
- 2026-09-24 09:59 JST: ユーザーの再実行依頼により、修正済みlog確率BCEのColab T4本学習を新run ID `trackastra_logbce_v2`で開始。旧run `trackastra_cap4096_v1`の重み・診断は保持する。config SHA-256 `a55034264b3481e60675bbd9c76e6847db4de8e88c82fd10b7b24549905865f3`、学習ソースSHA-256 `14e09066c890ada8cddfa6b52b2d559e874a193cecf84071cc0a24d5ba258303`、Colab bundle 4,382,775 bytes / SHA-256 `1561ab52bf667fcf2d00481575eec8b22919db38678cc0a815d8b30da24f30ad`。`make check-exp`、`make test-exp`（16 passed）、`make validate-exp`、train Notebook再生成が成功。Kaggleの固定cache 19,702ファイルとGEFF出力1ファイルの署名付きURLを列挙し、Colab session `exp040-trackastra-logbce-v2-20260924`がREADY。学習・評価結果は未取得。
- 2026-09-24 10:30 JST: 新runのColab T4で固定cache 19,702件 / 4,151,339,416 bytesを取得し、GEFF 199サンプルを認識。両foldの教師監査と2窓100 updateの学習成立確認が通過。fold 0の64窓benchmarkは65.26秒、fold 1は73.63秒。修正後の12時間gate予測は36,249.56秒 / 上限43,200秒で通過し、10 epochs×2foldの本学習へ進む。旧runのcheckpointを新runへ持ち込んでいない。
- 2026-09-24 10:47 JST: 修正後runのfold 0 epoch 1が完了。4,096窓、学習masked loss 0.721353、内部masked loss 0.851686。新run ID・config SHA・修正済み学習ソースSHA・抽出schedule SHAを含む受領票を取得し、19,722,891 bytesの再開state SHA-256 `08e557cbf9bf12c7dfd810bb2da330e7f76ccf00c8794f1026bd80d9f6eeea21`をローカルで独立照合。以降のepochを同じセッションで継続。
- 2026-09-24 11:03 JST: fold 0 epoch 2が完了（学習masked loss 0.388132、内部masked loss 0.617170）。Colab CLIのファイル操作用proxy token更新で受領が一時遅れたが、同じ稼働中T4割当へ接続を更新し、学習は再起動せずにepoch 2の受領票・再開stateをSHA検証して回収した。
- 2026-09-24 13:12 JST: 修正版Colab T4学習のfold 0が10 epochs完了。各epoch 4,096窓で、全10件の受領票と再開stateをbyte数・SHAでローカル回収。内部masked loss最小はepoch 3の0.590528、最終epochは1.185018。外側胚の診断はfold 1学習後のため、精度はまだ未判定。Colab CLIのファイル操作用proxyは同じT4割当へ更新しており、学習プロセスは再起動していない。
- 2026-09-24 15:26 JST: 修正版Colab T4でfold 0/1の10 epochs×2foldが完走。各epoch 4,096窓、全20件の受領票・再開stateをbyte数とSHAでローカル回収。内部masked loss最小はfold 0がepoch 3の0.590528、fold 1がepoch 8の0.119732で、外側胚を見ずにcheckpointを選択。両胚の全共通隣接ペア診断と最終archiveの回収は続行中で、graph進行条件と公式scoreはまだ未判定。
- 2026-09-24 16:07 JST: 修正版runのfold 0外側胚6bbaの全共通pair診断が完了。内部検証で固定した閾値0.9798409343で、既知edge回収は63,799 / 103,393（0.617053）。旧run 67,571 / 103,393（0.653536）、保存済みexp016 primary 100,230 / 103,393（0.969408）を下回る。正解親1位率0.873492（旧0.886501、対照0.979235）、分裂親回収17 / 108（旧21、対照48）、legacy mask負例予測6,090（旧6,314、対照4,355）、既知負例予測96（旧91、対照14）。事前指定の5条件は全て未達。fold 1外側胚44b6の診断と最終archive回収は継続し、全graph推論は進めない。
- 2026-09-24 17:30 JST: Colab T4のfold 1内部検証が終了し、44b6外側胚のラベル付きスコアファイルが生成開始したことを読み取り専用のファイル一覧で確認。外側胚の集計値と最終archiveはまだ未取得。6bbaで事前指定の進行条件が未達のため全graph推論は実行しない。
- 2026-09-24 17:44 JST: 修正版runのfold 1外側胚44b6の全共通pair診断が完了。内部検証で固定した閾値0.9855477810で、既知edge回収は8,892 / 18,949（0.469260）。旧run 8,958 / 18,949（0.472743）、保存済みexp016 primary 17,959 / 18,949（0.947754）を下回る。正解親1位率0.900945（旧0.900100、対照0.968442）、分裂親回収2 / 22（旧2、対照6）、legacy mask負例予測596（旧611、対照1,341）、既知負例予測1（旧1、対照4）。接続再現率・親順位・分裂回収の3条件が未達で、両胚進行条件は不成立。全graph推論・公式scoreは未実施。最終archiveの回収とSHA照合は継続中。
- 2026-09-24 18:05 JST: 修正版Colab runの最終archive 1,377,082,588 bytesをローカルに回収し、SHA-256 `25c10aa3fad768b75aa212940d5bd5d304fbda3ae9fcd22b04b641a7354b10e4`がColab受領票と一致。ストリーム展開後に完了マーカー・model manifest・2モデル・20件のepoch受領記録を検証。両胚のラベル付き予測ファイルも独立に再ハッシュし、fold 0 `8d140481c91e6c5716eccda8726d0c00dcbc06619d82ea4dc3d01534b570a0d5`、fold 1 `889d399e343b2fe44f9c1ba6b0aa60c6b0c562384e0b280b9cb56714afcb6ea1`が保存済みmetricsと一致。Notebook本体は27,072.53秒、Colab sessionは成果物ACK後に停止を確認。旧runの証拠を履歴へ保持し、修正版の結果をmetrics.jsonとresult.mdへ反映。statusはdebug_completed、全graph推論・公式score・submissionなし。検証コマンドはこの後実施。
- 2026-09-24 18:09 JST: 修正版の記録後、`make check-exp EXP=exp040_trackastra_association`、`make test-exp EXP=exp040_trackastra_association`（16 passed）、`make validate-exp EXP=exp040_trackastra_association`が成功。現行metricsのtrain_stageとartifactsがColab保存版に一致し、旧runの両胚数値がevidence.rerunsへ保持され、status debug_completed、graph_eligible false、CV/LB nullであることを照合した。ユーザーの実験完了・採否判断待ちのためcommit・pushは行わない。
