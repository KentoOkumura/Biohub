# exp016_frozen_image_encoder 要件と実装方法

この文書を、実装前の契約、実装方法、受け入れ条件の正とする。実行中の進捗は`SESSION_NOTES.md`へ記録し、この文書へ重複させない。

## 実験化の入口・引き継ぎ・承認

- 実験化の入口と承認: backlog候補`frozen_image_encoder`は教師対応とloss maskが未決だった。2026-09-14に選定公開sourceを監査し、ユーザーの「それでは推奨で進めてください」により、公開sourceの教師を初回基準として継承し、mask変更を別比較へ分ける方針と実験化・実装が承認された。
- 移行元backlog: `backlog/frozen_image_encoder.md`。
- 対応する上位仮説: `HYP-20260910-12`。
- 上位仮説のうちこの実験が検証する範囲: 固定公開画像特徴からprimary `SimpleNodeTransformer`だけを再学習できる基準、その費用、同じ外側胚評価での学習前後の接続指標と、候補生成・secondary tracker・ILP・graph repairを固定した公式graph指標を確認する。
- この実験だけで上位仮説を判断できるか: いいえ。
- 上位仮説の判断に残る検証: 学習済みtrackerを固定公開推論へ組み込み、同じ候補生成・secondary contribution・ILP・graph repair・公式評価で比較すること、hidden test全件の費用、`partial_edge_mask`など後続の一変更比較。
- 親実験: 公開構成と初期trackerの正は[`exp011_public_detector_selection`](../exp011_public_detector_selection/)。入力cacheの直接の生成元は[`exp015_oracle_stage_limits`](../exp015_oracle_stage_limits/)で、public testのcache等価性は[`exp014_exact_window_cache`](../exp014_exact_window_cache/)を参照する。
- 根拠 / 一次資料 / 参照実装: exp011の`assets/public_detector_selection.json`、exp014の`metrics.json`、exp015の`requirements.md`・`result.md`・`metrics.json`、Kaggle dataset `pilkwang/biohub-tracking-support-pack-50ep-v1`の`repo/scripts/train_unet_transformer.py`。
- 固定するもの: 公開checkpoint、画像encoder、検出head、画像正規化、2-frame window、8-view D4平均後のprimary feature、検出候補、secondary branch、DeepCenter、decode、公式評価処理。
- 変更するもの: primary `SimpleNodeTransformer`の重みだけを、同じ公開checkpointのtracker初期値から再学習する。検出lossを使わず、optimizerにはprimary tracker parameterだけを渡す。
- 最小の反証可能な検証: exp015の固定cacheと主催者GEFFから公開sourceどおりの教師を作り、いずれかのframeにGT nodeが0個のwindowは公開sourceの`get_window_data`と同じく除外して、2方向のleave-one-embryo-outで各1個のtrackerを学習する。学習前後を同じ外側胚windowで評価し、モデル以外のparameterがoptimizerに入らないこと、cache・公開source・初期checkpointのSHA、除外window監査、教師maskの件数、fold別model SHAを保存する。
- 成功条件: 2foldのtracker学習、内部checkpoint選択、外側胚window評価、model manifest保存が12時間のtrain runtime gateと実行時のGPU quota内で成立し、画像側の更新なしで2個のtrackerを作れること。後半はexp015の検出後cache 199件・19,701 windowを直接読み、main画像encoderと検出headのforwardを0回にする。公式評価器は保存済みcontrol graph 1件でend-to-end preflightを通す。公開trackerで再計算したILP前candidate graphがexp015保存graphと199件すべて完全一致した場合だけ、fold 0を6bba、fold 1を44b6へ適用し、同じILP・graph repair・DeepCenterを通した最終graphを固定公開controlと同じ公式評価器で再採点する。精度仮説の支持は両胚の公式combined scoreが改善した場合に限る。
- 停止条件: 入力cacheのsummary SHA・identity SHA・199 dataset・19,701 window・checkpoint SHA・公開source SHAが一致しない、公開trackerのcache replayによるcandidate node・edge・score・distanceがexp015保存graphと1件でも異なる、外側胚のGEFFが学習またはcheckpoint選択に入る、tracker以外がoptimizerへ入る、12時間gateを超える、OOM、NaN、またはKaggle Notebookの12時間上限内の完走が見込めない場合は停止する。
- 実行しないこと: 画像encoder・検出head・secondary trackerの更新、検出loss、scratch初期値、loss mask変更、7 µm linear sum assignment、別モデル構造、追加seed、control再学習、decode調整、Kaggle submission、Public LB取得。
- 未決事項: なし。学習epochは既存の胚holdout基準と同じ3、learning rateは0.0001とし、先に64 window/foldのbenchmarkを行ってruntime gateを適用する。
- backlog記録から解釈を変更した箇所とユーザー承認: 候補段階で未決だった教師を、公開sourceの5 µm greedy one-to-one対応と既存loss maskへ確定した。2026-09-14のユーザー承認による。

## 判断履歴

- 2026-09-10: 調査I12の候補として追加し、公開検出器固定・下流学習の計算量と精度を未測定として記録した。
- 2026-09-12: ユーザーが公開検出器を固定してtrackerを学習する方針を指定し、exp011の公開構成とprimary tracker初期値を採用した。
- 2026-09-13: exp015でtrain feature cacheも生成する契約を追加し、public同値性を確認済みの経路からcacheを作ることにした。
- 2026-09-14: exp015がtrain 199動画・19,701 window、4,151,337,848 bytesのcacheを生成し、array round-trip完全一致を確認した。
- 2026-09-14: exp011で選定した公開training sourceを取得し、SHA-256 `c4f6317736bb3bb1ec8f3f6e9a6d935a463e3f0f1f685481b2d13218d35dc9ea`、5 µm greedy one-to-one対応、source軸softmax、既存loss maskを確認した。
- 2026-09-14: ユーザーの「まずはトラッカー再学習の基準を作成するために評価基準をそろえるということですか？」に対し、モデル重み以外を固定する基準であることを確認した。
- 2026-09-14: ユーザーの「それでは推奨で進めてください」により、公開sourceの教師を初回基準にし、mask変更を`partial_edge_mask`へ分ける推奨案と実験化・実装が承認された。
- 2026-09-14: version 1の64 window/fold benchmarkから保守的な2fold合計を9.26時間と実測し、当初の6時間gateでfull train前に停止した。ユーザーの「gateは12時間でいいです。それで進めてください」により、モデル・loss・foldを変えずruntime gateのみ12時間へ変更した。
- 2026-09-14: version 2は12時間gateを通過した後、GT nodeが0個のframeを含むwindowで停止した。公開sourceの`get_window_data`が該当windowを`None`として除外することを再確認し、同じ除外を固定cacheのpath選択へ実装することを、不具合修正として実験契約へ明記した。
- 2026-09-14: version 3が2foldのtracker学習を完走した。ユーザーの「後半に進んでください」により、同じ実験内で保存済み2fold modelを読み込み、固定公開graph pipelineと公式評価まで実装・実行することが承認された。Kaggle submissionは承認範囲に含めない。
- 2026-09-15: raw画像からmain画像encoderを再実行したinference version 3は199件のgraph repairまで進んだが、評価器import失敗により公式指標を保存できず、GPU quotaを使い切った。ユーザーの「cacheを利用するように修正してください」により、exp015の検出後cacheからtracker・ILP・repairだけを再生する経路へ変更した。比較対象、fold、model、secondary、ILP、repair、公式評価条件は変更しない。
- 2026-09-18: ユーザーが、この実験の目的を再学習trackerの比較ベンチマークを残すことと確認し、Kaggle環境向け実行ファイルを正規ベンチマークとして保持しつつ、Colabを代替実行経路として分離する方針を承認した。

## 手法契約

実装区分は`docs/glossary.md`に定義したこのリポジトリ内の管理用ラベルを使う。

- 依頼原文: 「`frozen_image_encoder`を実装してください」「それでは推奨で進めてください」。
- 期待する成果: 既存公開trackerと将来のloss変更を同じ条件で比較できる、固定特徴からのprimary tracker再学習基準を作る。
- input: exp015が同じ公開forwardから保存した候補ID、grid座標、物理座標、position feature、candidate mask、primary 32-channel feature。教師には学習側の主催者GEFF node・edgeだけを使う。いずれかのframeにGT nodeが0個のwindowは、公開sourceと同じく学習・内部検証・外側評価から除外する。
- target / objective: 隣接時刻の候補間で既知の親子edgeを予測する。分裂は1つのsourceから複数targetへの正例として同じedge行列内に表す。
- output: 候補間のprimary tracker logit、fold別に選択したprimary `SimpleNodeTransformer` state、学習前後の外側胚window指標、教師監査、model manifest。
- loss: source軸softmax後のbinary cross entropyへfocal係数を掛ける。正例を1本以上持つsource行またはtarget列に触れるpairをmaskへ含める。division追加重みは1.0。これは公開sourceの既存lossを継承する。
- decode: train段階ではgraphへdecodeせず、固定候補上の接続logitと教師指標を評価する。inference段階ではexp015が検出後に保存した全候補ID・座標・mask・primary/secondary featureを読み、main画像encoderと検出headを再実行しない。公開tracker replayのILP前graphをexp015保存graphと完全比較した後、公開full checkpointの`transformer.*`だけを保存済みfold stateへ置換する。fold 0を6bba、fold 1を44b6へ適用し、secondary contribution、bidirectional association、ILP、motion・gap・division repair、DeepCenterを変更せず、保存済みexp015 control graphと同じ公式評価器で比較する。
- context unit: model inputとlossは同じ動画の隣接2-frame window、外側分割は1胚、最終評価とmodel選択の記録はfold単位。
- 実装区分: `staged-faithful`。公開sourceのtracker構造、候補対応、target、lossを保持し、画像側を固定cacheへ置き換えたtrain段階を先に実装した。後半も同じ検出後cacheを入力にし、公開trackerでILP前graphの完全一致を要求してから固定decodeで公式graph scoreを実測する。
- 省略する機構と理由: train段階では画像forward、検出loss、decode、公式graph評価を省略した。後半ではexp015で実行済みのmain画像encoder・検出headを再実行せず、その出力cacheと保存済みpost-retention候補を使う。DeepCenterはrepair候補が変わり得るため従来どおりraw frameから実行する。検出loss、追加学習、Kaggle submission、Public LB取得は行わない。
- proxyで検証できない主張: N/A。proxyではないが、この段階だけでは公式combined score、Public LB、hidden test runtimeを判断できない。
- proxyの場合のユーザー承認: N/A。
- この実験が支持 / 棄却できる主張: 固定公開featureと公開sourceの教師・lossでprimary trackerだけを予算内に学習できるか、固定候補上の外側胚接続指標と固定pipelineの公式graph scoreが公開trackerより改善するか。
- この実験では判断できない主張: 独立CV、Public LB改善、unknown embryo一般化、mask変更の効果、画像encoder更新の効果。

## 実装方法

- アプローチ: train Notebookがexp015 kernel outputのcacheを直接解決し、competition train GEFFを読む。sample prefixから2方向のouter foldを作り、各学習胚内をseed 0で90%/10%へ分ける。各foldで公開trackerを初期化し、internal validationの`edge_accuracy_times_candidate_node_recall`が最大のstateを保存して外側胚windowを評価する。
- inputの実装箇所と変換: `exp016_frozen_image_encoder_train.py`がcache root、公開artifact、competition trainを動的に解決する。`frozen_tracker.py`がNPZ schemaを検査し、`filter_nonempty_gt_window_paths`で公開sourceと同じempty-GT window除外を適用して監査JSONを保存し、primary featureとposition featureを連結する。modelへ渡す座標はgrid座標へdownsample `(1,4,4)`を掛けた元voxel z/y/xとし、教師matchingだけ物理µm座標を使う。
- target / objectiveの構築箇所: `frozen_tracker.py`の`greedy_match_candidates`と`build_legacy_edge_target`。candidate順で最近傍GT距離を求め、距離昇順で未使用GTを割り当て、5 µmを超える候補を未対応とする。
- outputの生成箇所と表現: `/kaggle/working/models/fold_<n>/primary_tracker_best.pth`、`fold_<n>_summary.json`、`teacher_audit.json`、`training_summary.json`、`model_manifest.json`、更新した`metrics.json`。
- lossの実装箇所: `frozen_tracker.py`の`legacy_focal_bce`。公開sourceと同じsource軸softmax、focal gamma 2、正例行OR正例列maskを使う。
- decode / postprocessの実装箇所: `graph_inference.py`がexp015 cacheを2 GPU workerで読み、primaryの順逆方向logit、secondaryのlow-margin consensus、source軸softmaxと0.48 thresholdを再計算する。secondary trackerのforwardは公開・再学習primary間で共有し、1 windowあたりのtracker forwardを5回にする。公開tracker側のILP前graphをexp015保存graphと全199件で完全比較し、再学習tracker側だけ同じILPへ渡す。`exp016_frozen_image_encoder_inference.py`はSHA固定したexp015 sourceのsetupとgraph repairを再利用し、main画像encoder実行部をcache replayへ置換する。exp015と同じ最終graph NPZを保存し、整数丸め・非負clampをrow writerと一致させて一時GEFFへ変換する。公開support datasetの`repo/scripts/evaluate.py`を長時間処理前にSHA・entrypoint・GT directoryまで検査し、保存済みcontrol graph 1件のend-to-end採点も通してから、`evaluate_run`と`biohub_tracking.metrics.summarise`でcontrol・再学習後を各2回採点する。12時間gateはNotebook開始からcache replayとgraph repairを含む全工程へ適用する。
- 実行環境の役割: `exp016_frozen_image_encoder_inference.py`と生成NotebookをKaggle環境の正規ベンチマークとする。全199動画のcache replay、固定graph repair、公式評価までを同じNotebookで完了し、そのreceiptに`execution_environment=kaggle`と`benchmark_role=canonical`を保存する。Colab側の公開control graph SHAと候補座標SHAはKaggle側と完全一致を要求するが、Colabのsmokeまたはtracker・ILP replayだけで公式ベンチマーク完了とは扱わない。
- Colab実装: `exp016_frozen_image_encoder_colab_inference.py`と生成Notebookが1基のColab GPUでcache replayとILPまでを行う。2動画benchmarkで公開trackerのcandidate graph完全一致と12時間内のfull予測を確認した場合だけ199動画へ進む。fullは5動画ごとに`/content`でcandidate graphとILP graphを生成し、1個のarchiveとreceiptへまとめてGoogle Driveへ保存し、完了batchを再利用可能にする。コードと保存済みfold tracker 2件はcredentialを含まない4.4MB未満のbundleへまとめる。Colab Secretの`KAGGLE_API_TOKEN`から入力を取得する経路と、Driveへ手動配置した入力を`/content`へコピーする経路を持つ。CLI full replayでは明示許可を得た期限付きURL manifestでKaggle出力をColabへ直接取得し、終了後のVMへ依存しないようbatch archiveを実行中の標準出力からローカルへ逐次回収する。このstageだけではgraph repairまたは公式指標の完了を主張せず、Kaggle正規ベンチマークを置き換えない。
- context unitを保つ処理箇所: cache pathからsampleとwindow frameを検査し、同じsampleのGEFFだけを教師に使う。split検査で同じembryoが学習と外側評価にまたがらないことをassertする。
- 変更するファイル / component: exp016のconfig、requirements、train/inference Jupytext sourceとNotebook、`frozen_tracker.py`、`graph_inference.py`、実験固有tests、README、SESSION_NOTES、result、metrics。
- 固定事項を保つ確認方法: configとtestで公開source/checkpoint/cache SHA、trainable component、train active variant 1、fold 2、control再学習なし、matching 5 µm、既存mask、submission禁止を固定する。train runtimeではoptimizer parameter IDがmodel parameter IDと完全一致することを検査する。inference runtimeではhybrid checkpointの非`transformer.*` tensorが公開checkpointと完全一致し、cacheのsummary・identity・全window contentを検査する。公開trackerで復元したcandidate node、edge、score、distanceをexp015保存ILP前graphと全199件で完全一致させ、main画像encoder forwardが0回であることをreceiptへ保存する。
- 参照sourceとの一致を確認するテスト: 合成候補でempty-GT window除外、距離昇順greedy one-to-one、division target、source軸softmax、正例行OR正例列mask、zero-positive lossを検査する。公開sourceとmodel sourceのruntime SHAもNotebookで検査する。
- 承認済み差分を確認するテスト: encoder/detection/secondary/decodeがfrozen listにあり、primary trackerだけがtrainable、7 µm assignmentとmask変更が含まれないことを検査する。

## 探索幅とpivot判定

- 変更class: `mechanism`。画像側を固定cacheとして再利用し、学習対象をprimary trackerだけへ限定する。
- 同じ親 / familyで連続した小改善実験数: 0。公開routeで初めての下流学習基準である。
- positiveなoracle headroom / coverage / 誤差非相関性: exp015でcandidate edge recall 94.829%、final selected edge recall 91.602%、candidate division triplet recall 40.397%を観測した。tracker改善余地はあるがend-to-end改善を保証しない。
- 比較したtarget、output、decode、context unitを変える案: `partial_edge_mask`はloss mask、`division_triplets`はoutputとtarget、`sparse_motion_graph`はcontextと候補pairを変える。初回基準へ混ぜない。
- 小改善の継続またはpivotを選ぶ根拠: 公開routeの固定featureと既存tracker初期値が揃い、下流学習基準が未作成なので、まず1構成を反証可能に実行する。
- `kaggle-idea-forge` の実行要否と根拠: 不要。承認済みの中心候補を実装する作業であり、同familyの連続小改善3件目ではない。

## 再現性・リスク

- seed policy: global seed 42、fold offset、fold別DataLoader generator、worker seedを固定する。internal splitはseed 0。
- stochastic 処理の有無: DataLoader shuffle、dropout、CUDA kernelがある。公開sourceのtracker学習にないmixed precisionは追加しない。
- stochastic feature generation / augmentation / seed bagging の有無: featureはexp015の固定cache、augmentationなし、seed baggingなし。
- 並列処理と乱数の関係: DataLoader順はfold別generatorで固定し、workerはPyTorch seedからNumPy seedを設定する。thread内global RNGは使わない。
- CPU/GPU runtime と deterministic flags: Kaggle T4、internet無効、full precision。64 window/foldのbenchmarkへ1.5倍の保守係数を掛け、train予測12時間を超える場合はfull train前に停止する。bitwise deterministicは主張しない。
- train cache / test feature regeneration の SHA 記録方針: exp015のcache summary SHAとidentity SHAを入力証拠とし、runtimeでdataset/window coverageを再検査する。test特徴はこの段階で再生成しない。
- model manifest / prediction / submission SHA 記録方針: 2個のtracker file SHA、canonical state SHA、fold split、初期checkpoint SHA、source SHAをmanifestへ保存する。submissionは生成しない。
- Kaggle package bootstrap 確認方針: prepare後に正のNotebook、config、metrics、`frozen_tracker.py`がbootstrapへ含まれることをvalidatorで確認する。
- リークリスク: 公開画像encoderと初期trackerはtrain 199動画を含む可能性があり、outer foldだけでは独立評価にならない。外側評価胚のGEFFをgradient、checkpoint選択、runtime判断、条件選択に使わない。
- CV/LB 不一致リスク: train段階のedge指標と公式combined scoreは異なる。公式graph scoreは固定train graphで実測し、Public LBは推定しない。
- ランタイム/メモリリスク: 候補pairはwindowごとに二次増加する。trainはbatch padding、pair chunk 32、batch size 2を使う。inferenceは2 GPU workerでsample単位に逐次cacheを読み、公開・再学習primaryと固定secondaryを評価する。12時間のhard gateを使い、OOM時に候補を削って別手法へ読み替えない。raw画像経路version 3のprediction 23,217.36秒に対し、Colab T4のcache replayは199動画・19,701 windowを6,145.75秒で完走した。固定graph repairと公式評価の実測はまだない。
- 再現性リスク: dropoutとCUDAによりmodel SHAは再実行で一致しない可能性がある。入力SHA、seed、環境、数値差を記録する。
- 手法忠実性リスク: cacheのprimary featureは公開0.946経路の8-view D4平均で、公開training sourceの元画像augmentationとは同一でない。これは固定公開inference featureを学習入力にする承認済み差分として記録し、元のend-to-end training再現とは呼ばない。
- 過度な縮小 / proxy化リスク: runtime超過時にfold、epoch、候補、feature channelを黙って縮小しない。停止してユーザーへ結果と選択肢を示す。

## 受け入れ基準

- [x] backlogの上位仮説、検証範囲、残る検証、根拠、差分、固定事項、成功条件、停止条件、実行しないこと、判断履歴を移行した。
- [x] `config.yaml`の`lineage.hypothesis_id`と`lineage.backlog_candidate`が本書と一致する。
- [x] input / target / output / loss / decode / context unitと`staged-faithful`の範囲を記録した。
- [x] active variant 1、model/config 1、outer fold 2、booster 0、control再学習なしを記録した。
- [x] train Jupytext sourceとNotebookが、input確認、教師監査、benchmark gate、2fold学習、評価、model manifest、metrics保存を実装する。
- [x] 合成データの教師・loss・fold・config契約testが通る。
- [x] Jupytext round-trip、validate-exp、check-exp、test-expが通る。
- [x] Kaggle実行前にGPU quotaを確認し、variant/config/fold/model数とcontrol再学習なしをSESSION_NOTESへ記録する。
- [x] Kaggle trainがruntime gateを通って2foldを完走し、model 2個、fold別評価、教師監査、SHAを保存する。
- [x] train結果と費用を提示し、ユーザーが同じexpの公式graph評価用inferenceへ進むことを承認する。
- [x] inference sourceとNotebookがfold別tracker injection、2 GPU cache replay、main画像encoder forward 0回、公開tracker candidate graph全199件完全一致、固定ILP・repair、199件の最終graph、公式評価の再計算一致、metrics保存を実装する。
- [x] Colab sourceとNotebookが1 GPU cache replay、2動画benchmark gate、5動画単位のbatch archiveによるDrive永続化と再開、Kaggle Secret経路と手動Drive入力経路を実装する。
- [x] Colab T4でPython 3.13、PyTorch、必要なgraph packageのimportまでpreflightする。
- [x] Kaggle `inference.py/.ipynb`を全199動画・graph repair・公式評価を含む正規ベンチマークとして保持し、Colabを代替replay環境として設定・test・receiptで区別する。
- [x] Colabで2動画benchmarkを実行し、公開trackerのcandidate graph完全一致とfull所要時間予測を保存する。
- [x] Colabで199動画のtracker・ILP replayを完走し、全件の公開candidate graph完全一致と生成graph SHAを保存する。
- [x] Kaggle inference前にGPU quotaを確認し、評価variant 2、追加学習0、load model 2、booster 0、control再学習なしをSESSION_NOTESへ記録する。
- [ ] Kaggle inferenceが12時間gateとGPU quota内で完走し、候補固定の証拠、fold割当、control・再学習後の公式指標、最終graph SHAを保存する。
- [ ] 実験の完了、採用、不採用は結果提示後のユーザー判断まで確定しない。
