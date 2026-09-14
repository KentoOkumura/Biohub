# exp015_oracle_stage_limits 要件と実装方法

この文書を実装前の契約、実装方法、受け入れ条件の正とする。実行中の進捗は SESSION_NOTES.md へ記録する。

## 実験化の入口・引き継ぎ・承認

- 実験化の入口と承認: backlog候補 oracle_stage_limits は 設計可能・実験化未承認、未決事項は なし。2026-09-13のユーザー依頼「oracle_stage_limitsを実装してください」を実験化承認として扱う。
- 移行元backlog: backlog/oracle_stage_limits.md。
- 対応する上位仮説: HYP-20260910-14。
- 直接検証する範囲: GTを診断にだけ用い、固定公開モデルの既知中心回収、既知辺の候補graph内回収、母と2娘の同時回収、同じ候補構造の最終graph選択を分けて測る。
- この実験だけで上位仮説を判断できるか: いいえ。公開モデルの学習来歴にtrain 199動画が含まれるため、独立CVではなく固定公開モデル下の条件付き診断である。
- 判断に残る検証: 評価器と基準予測の独立性、固定特徴で学習するtrackerの精度、hidden testでの費用。
- 親実験: exp014_exact_window_cache。公開構成の選定はexp011_public_detector_selection、悪化16件と条件列はexp012_group_error_readoutから継承する。
- 固定するもの: exp011で採用した公開Notebook、3 checkpoint、2-frame window、前処理、8-view検出・primary association feature TTA、secondary branch、候補生成、ILP、graph repair、閾値、座標scaleを固定する。
- 変更するもの: competition train 199動画へ同じ固定pipelineを適用し、同じforwardから後続学習用window cache、ILP前の候補graph、最終graphを保存する。別NotebookでGTを時刻別・7 µm以内の一対一対応に使い、段階別の回収率を集計する。
- 最小検証: exp012の固定SHAから exp006 - exp005 の adj_edge_jaccard が負のsampleを再計算し、16件かつ全件44b6であることをassertする。16件と全199件へ同じ4段階判定を適用し、5条件、brightness x candidate density、44b6内の条件別表を作る。
- 成功条件: 199件の候補graph、最終graph、GT、条件が一意に対応し、19,701 windowのcacheが重複・欠損なく完全一致検査を通る。失敗・NaN・有効件数、入力と出力のSHAを含む段階別表を再計算できる。
- 停止条件: 固定checkpoint、sample集合、exp012 SHA、候補graph、最終graph、GTのいずれかが一致しない場合、cacheのround-trip、coverage、容量guardのいずれかが失敗した場合は停止し、欠損を推測で補わない。
- 実行しないこと: model学習、control再学習、閾値・重み・候補生成・復号の選択、外側評価を使うrouterやpost-process、未対応候補の負例扱い、Kaggle submissionを行わない。
- 未決事項: なし。Kaggle実行時間、生成物容量、実測回収率は実行時に確認する値であり、実装前の未決事項ではない。

## exp012から引き継ぐ固定診断

- per_sample_readout.csv のSHA-256は 11dc792242dca26290be6a6734b322b0bfa9238f395b1a56d5d179f84e6b7041。
- paired_route_comparison.csv のSHA-256は 3a6fa2a768290a8cfb3c6c961ede08af6a2a72381b07fe48f50d1c228b91dd50。
- baselineはexp005_embryo_holdout_batch8、comparisonはexp006_embryo_holdout_seed314159。
- 悪化は同一sampleの adj_edge_jaccard(exp006) - adj_edge_jaccard(exp005) が負であることとし、16件であることを固定する。
- 条件列はexp005行の embryo、brightness_bucket、candidate_density_bucket、boundary_distance_bucket、known_divisionを使う。
- 条件は重なりを持つため、brightnessやcandidate densityの独立した因果効果とは呼ばない。

## 手法契約

実装区分は docs/glossary.md に定義したこのリポジトリ内の管理用ラベルである。

- input: 固定公開model、train 199動画、ILP前候補graph、最終graph、主催者GEFF、exp012の固定sample別条件。
- target / objective: 新しい学習目的はない。既知注釈のどこまでが固定候補と最終選択で回収可能かを段階別に測る。
- output: train 199動画・19,701連続windowのNPZ cache、GPU shard別cache manifest、window_cache_summary.json、per_sample_stage_limits.csv、stage_summary.csv、condition_stage_summary.csv、brightness_candidate_density_summary.csv、degraded16_samples.csv、oracle_summary.json、oracle_manifest.json。
- loss: なし。
- decode: 予測側はexp014で等価性を確認した公開pipelineのILPとgraph repairを変更しない。診断側は同じ候補IDに対する累積判定を行い、GTをdecodeへ戻さない。
- context unit: matchingは1動画の同一timepoint、edgeとdivisionは1動画内の注釈graph、独立性は胚単位、条件集計はsample単位。
- 実装区分: staged-faithful。固定公開pipelineの予測とGTを使う診断を別Notebookに分け、Kaggle上でtrain 199動画のinferenceと全件diagnosticを実行した。
- 省略する機構: model学習と提出を省略する。診断仮説に不要であり、固定モデルの上限だけを分離するため。
- proxyで検証できない主張: N/A。proxyではない。段階別回収率、GPU予測時間、保存容量はKaggle実測を result.md と metrics.json に記録した。
- 支持できる主張: 固定公開モデル下で、既知中心・既知辺・既知分裂が候補生成と最終選択のどこで失われるかを再現可能に分けられること。
- 判断できない主張: 未注釈細胞のprecision、未知胚一般化、tracker再学習の改善、Public LB改善、条件の因果効果。

## train window cache契約

- 単位: 各train動画の連続する2 frame。100 frame × 199動画から重複のない19,701 windowを期待する。
- 保存する配列: 両frameのcandidate ID、grid座標、物理座標、検出score、position feature、candidate mask、primary 32-channel feature、secondary 32-channel feature。
- 形式: numpy_npz_uncompressed。各NPZへarray schemaとarray content SHAを埋め込み、書き込み直後に読み戻してdtype、shape、値を完全一致で検査する。
- 現在の推論経路: 書き込み後のcache tensorへ差し替えず、元のGPU tensorをそのまま用いる。cache検証のためにedge logitsを再計算せず、primaryとsecondaryのedge scoringは通常どおり各1回だけ行う。
- 同値性の根拠: exp014でpublic test全396 windowについて配列round-trip、primary/secondary edge logits、exp013最終出力の完全一致を確認済みで、window_cache_summary.jsonのSHA-256は 4c8449d472805b541c00d1b4016c35353837e91115c56117d0153257ca564aa8。exp015ではGPU時間節約のためedge logitsの二重計算を繰り返さない。
- 容量guard: exp014からの単純比例見積もりは6,081,397,663 bytes。各GPU shardを6,000,000,000 bytes、cache合計を12,000,000,000 bytes、/kaggle/working全体を18,000,000,000 bytesで停止し、Kaggle output上限20,000,000,000 bytesに2GB以上の余裕を残す。
- 実行証拠: dataset・window coverage、cache file/schema/content SHA、candidate数、feature値数、画像feature生成・candidate抽出・write・read時間、peak GPU memory、合計容量をGPU shard別manifestとsummaryへ残す。

## 段階判定

1. 既知中心: 各timepointでGT中心と候補中心を物理距離7 µm以内のlinear sum assignmentで一対一対応し、対応したGT node数を数える。
2. 既知辺: 両端のGT nodeが対応し、その候補ID対がILP前候補graphのedgeに存在する場合だけ累積的に回収とする。
3. 母と2娘: GTで2本以上の出edgeを持つ母について、対応した母から異なる2娘への候補edgeが同時に存在する場合だけ回収とする。
4. 最終graph: 第2または第3段階で回収した同じ候補IDとedgeが最終graphにも存在する場合だけ選択済みとする。

複数の対応候補が7 µm以内にあるGT node数と、複数GTが7 µm以内にある候補node数を別列に残す。疎いGEFFに対応しない予測候補はfalse positiveとしない。

## 実装方法

- inference Notebook: exp013の自己完結公開pipelineを基に、inputをtestからtrainへ切り替える。既存forwardで得たcandidate featureをwindow別NPZへ保存して完全一致を検査し、同じin-memory tensorから通常のedge scoringを続ける。動的patchでbuild_graph直後・ILP前のgraphをsample別GEFFへ保存する。既存のILPとrepair後は元candidate IDを保持したnode・edge配列をsample別NPZへ保存する。
- diagnostic Notebook: inference manifest、候補GEFF、最終NPZ、train GEFF、exp012表を解決する。sample集合とSHAを検査し、時刻別matching、edge・divisionの累積判定、条件別micro集計を行う。
- 出力manifest: sample別GT・候補・最終graphのcanonical content SHA、固定exp012 SHA、出力CSV SHA、失敗件数に加え、train cacheの全window coverage、配列同一性、容量、schema/content SHAを記録する。
- Notebook安全性: sourceはPath.cwd()を基準にし、__file__へ依存しない。diagnostic sourceはmain guardを持ち、純粋関数を実験固有testから検査できる。
- 数値設定: detector thresholdは0.965、matching radiusは7 µm、scaleはz/y/x = 1.625/0.40625/0.40625 µmをconfig.yamlに置く。

## 探索幅とpivot判定

- 変更class: postprocess。このリポジトリ内では固定予測へ診断出力を追加する管理上の区分であり、提出予測の後処理変更を意味しない。
- 同じ親または機構familyで連続した小改善実験数: 0。exp015は精度を変更する小改善ではなく、固定候補の上限診断である。
- positiveなoracle headroom、coverage、誤差非相関性: 本実験で初めて公開固定modelについて測る。実行前にpositiveと仮定しない。
- target、output、decode、context unitを変える案: 本実験では選ばない。結果を提示した後、固定特徴で学習するtracker側の候補をユーザーが別途判断する。
- 小改善継続またはpivotの根拠: 候補上限を先に分離し、検出候補不足と接続・分裂・最終選択の不足を混同せずに後続実験を選べるようにする。
- kaggle-idea-forge: 不要。承認済みの診断候補を実装する作業で、新しい候補生成ではない。

## 最小検証と計算量

- active variant数: 1。固定公開pipelineだけを実行する。
- model/config数: 固定済みのprimary、secondary、DeepCenterの1構成。比較のための別configは実行しない。
- fold数: 学習0。train 199動画へ固定modelを適用する診断推論のみ。
- booster数: 0。
- control再学習: なし。
- local検証: 合成graphで一対一対応、曖昧対応、edgeの累積回収、母と2娘、最終選択、条件scope、NPZ契約をtestする。full runの代用にはしない。
- Kaggle資源: inferenceはT4 2基、diagnosticはCPU、internet無効。exp014のpublic 4動画実測を単純比例するとcacheを含む予測処理は約8.25時間、Notebook全体は約8.5〜9時間を見込むが、train 199件の保証値ではない。T4を2基使ってもwall time 1時間のNotebookはGPU quota約1時間として扱う。開始時にquotaが残っていれば、実行中に残量を使い切っても12時間上限までは継続するというKaggleの運用を前提とするが、push直前にquota、active session、仕様変更を再確認する。
- submission: 0件。実行しない。

## 再現性・リスク

- 乱数samplingは追加しない。CUDA kernel、SCIP solver、2 process shardingはdeterministic flagを強制しない。
- exp014で同じ公開pipelineのcache前後とexp013最終出力の一致は確認済み。exp015 version 1でも同じschemaの19,701 NPZを読み戻して完全一致を検査し、edge logitsの二重計算なし、train 199動画のcoverageとSHAをKaggle logで確認した。
- 公開モデルの学習来歴にtrain 199動画が含まれるため、結果を独立CVや一般化性能と呼ばない。
- GTは診断Notebookだけで読み、inference manifestには ground_truth_accessed=false を記録する。
- 一対一対応の曖昧さを件数化し、別matching方式を外側結果に合わせて選ばない。
- candidate graphと最終graphのnode IDが保持されない場合は停止し、座標の再対応で都合よく補わない。
- 公開model下の候補上限が高くても、新しいtrackerのend-to-end改善を意味しない。

## 判断履歴

- 2026-09-13: ユーザーがexp012を診断目的達成として閉じ、悪化16件の5条件別段階分解を本候補へ含めるよう指定した。
- 2026-09-13: exp014でpublic test全396 windowのcache前後配列、edge logits、exp013最終出力の一致を確認し、後続診断へ進める状態になった。
- 2026-09-13: ユーザー依頼「oracle_stage_limitsを実装してください」により実験化を承認された。
- 2026-09-13: ユーザー依頼により、同じGPU runでtrain用window cacheも生成する契約へ拡張した。exp014の完全同値性を参照し、exp015ではedge logitsを二重計算しない。
- backlog記録からの変更: 親を、公開構成選定だけのexp011から、同一pipelineと候補ID・graph等価性を最も近く確認したexp014へ具体化した。公開構成の正は引き続きexp011である。

## 受け入れ基準

- [x] backlogの上位仮説、検証範囲、残る検証、根拠、固定事項、変更事項、成功条件、停止条件、実行しないこと、判断履歴を移行した。
- [x] config.yamlのlineageと本書が一致する。
- [x] input / target / output / loss / decode / context unitと実装区分を記録した。
- [x] inference sourceがtrain 199件、ILP前候補graph、元ID付き最終graph、ground_truth_accessed=falseのmanifestを実装する。
- [x] 同じforwardから19,701 windowのcacheを生成し、各NPZの完全一致、coverage、SHA、容量guardを検査する。cache検証のedge score再計算は行わない。
- [x] diagnostic sourceがexp012 SHA、悪化16件、時刻別一対一対応、4段階、5条件、交差表、failure・NaN、manifestを実装する。
- [x] Jupytext round-trip、validate-exp、check-exp、test-expが通る。
- [x] Kaggle GPUで固定公開pipelineのtrain 199件推論が完走し、候補graphと最終graphが199件、train cacheが19,701 window揃う。
- [x] Kaggle CPUでdiagnosticが完走し、必須表、入力・出力SHA、kernel version、Notebook実行時間が揃う。
- [x] 結果と限界をユーザーへ提示し、2026-09-14にdiagnosticとしての完了判断を受けた。
