# exp035_velocity_features 要件と実装方法

この文書を、実装前の契約、実装方法、受け入れ条件の正とする。実行中の進捗は `SESSION_NOTES.md` へ記録し、この文書へ重複させない。

## 実験化の入口・引き継ぎ・承認

- 実験化の入口と承認: backlog候補 `velocity_features` は履歴生成、特徴表現、分裂後処理、全graphへの進行条件が未決で `検討メモ・設計不可` だった。2026-09-22に推奨設計を提示し、ユーザーがexp016との比較だけを初回目的として同構造controlを省くこと、実装とKaggle実行を承認した。
- 移行元backlog: `backlog/velocity_features.md`。
- 対応する上位仮説: `HYP-20260910-10`。
- 上位仮説のうちこの実験が検証する範囲: 対象sampleを学習に使わないfixed first-pass trackerの予測履歴から直前2点の3次元移動ベクトルを求め、次候補との差をpair logitへ追加すると、固定画像特徴と現在位置だけを使うexp016より両胚の接続指標が改善するかを調べる。
- この実験だけで上位仮説を判断できるか: いいえ。
- 上位仮説の判断に残る検証: 位置列の直接学習、未対応点の複数時点入力、周囲の運動、長い欠測、軌跡断片接続、hidden test。
- 親実験: `exp016_frozen_image_encoder`。入力cacheは `exp015_oracle_stage_limits`、exp027の順位・確率診断、exp025の固定カルマン費用の不採用結果を参照する。
- 固定するもの: exp015の候補・物理座標・画像特徴、exp016の公開checkpoint初期値、教師対応、loss、mask、外側胚分割、3 epoch、内部checkpoint選択、secondary tracker、ILP、graph repair、公式評価。
- 変更するもの: 分割外first-pass予測履歴、移動ベクトル・次候補との差・履歴品質、base logitへ加えるzero-initialized velocity pair branch。
- 最小の反証可能な検証: inner 2-fold cross-fittingで対象sampleを学習に使わない履歴を作り、velocity modelを2方向の外側胚分割で各1個学習する。保存済みexp016 fold modelを同じ外側windowで再評価してbaselineとし、固定0.5で両胚別の既知edge recall、active pair内の教師負例予測、正解親1位率、分裂回収数を比較する。
- 成功条件: exp016に対して両胚の既知edge recallが改善し、active pair内の教師負例予測が各胚で5%を超えて増えず、分裂回収数が減らない。条件を満たした場合だけ同じ実験へ全graph inferenceを追加し、最終成功は両胚の公式combined score改善とする。
- 停止条件: 履歴のsample・frame・candidate ID・物理座標が一致しない、対象sampleが履歴modelの学習またはcheckpoint選択へ入る、非有限値、予測履歴の有効件数が0、12時間Notebook gateまたは週30 GPU時間を超える、pair診断の進行条件を満たさない場合は全graphへ進まない。
- 実行しないこと: exp016 controlの再学習、同構造neutral control、正解軌跡入力、正解分裂によるreset、velocity model自身での再帰的履歴更新、外側評価を見た閾値調整、カルマン費用またはHungarianだけの代用、画像encoder・候補・教師・loss・decodeの同時変更、submission。
- 未決事項: なし。
- backlog記録から解釈を変更した箇所とユーザー承認: 初回の4比較modelを、保存済みexp016とvelocityあり2foldの比較へ変更した。同構造neutral controlは原因分解が必要な場合の後続へ回す。2026-09-22の「それで実装してください。実行まで許可します」による。

## 判断履歴

- 2026-09-20: 速度特徴候補をP2へ上げたが、履歴方式が未決のため設計不可を維持した。
- 2026-09-22: fixed first-pass履歴、直前2点の移動、欠損mask、分裂時reset、逆向き別履歴、pair診断gateを推奨した。
- 2026-09-22: ユーザーはexp016に対する精度改善だけを初回目的とし、neutral controlを省いてvelocityあり2foldだけを学習する設計を選択した。
- 2026-09-22: ユーザーが実装とKaggle実行を承認した。submissionは承認されていない。

## 手法契約

- 依頼原文: 「velocity_featuresを実装してください」「exp016に対して精度が上がったかどうかだけ見たい」「それで実装してください。実行まで許可します」。
- 期待する成果: exp016の固定候補・画像特徴・教師・lossに、自己予測履歴の移動情報だけを追加したtrackerの両胚別pair診断を得る。
- input: exp015 cacheの64次元node特徴、元voxel座標、物理座標、candidate ID。追加入力は直前の予測edgeから得る移動ベクトル、現在から次候補への変位、予測位置との差、速さ、差の大きさ、方向cosine、履歴有無、履歴長、直前edge確率である。vectorは5 µm/frameで割って絶対値4へclipし、履歴長は8でcapする。
- target / objective: 主催者GEFFの既知中心・接続・分裂に対応付けた隣接候補pairの通常接続と母娘接続。
- output: exp016と同じ候補pairのbase logitにvelocity pair branchのdelta logitを加えた接続logit、fold別model、履歴modelと履歴特徴のmanifest、両胚別診断。
- loss: exp016と同じsource軸softmax後のfocal-weighted binary cross entropy、正例行または正例列に触れるpair mask、division weight 1.0。
- decode: train段階はgraphへdecodeしない。first-passでは各targetについてsource軸softmax最大の親を0.5以上で採用し、1親から複数targetが選ばれた場合は娘の履歴をresetする。velocity modelの履歴は固定し、学習中または推論中に自己更新しない。逆方向を使う全graph段階では逆向きfirst-pass履歴を別生成する。
- context unit: 学習・lossは同一sampleの隣接2-frame window。履歴は同一sample内でframe順に生成し、sample境界をまたがない。
- 実装区分: `staged-faithful`。pair診断では承認済みvelocity特徴を実際に学習する。全graph・公式評価は事前gate通過後に同じ実験へ追加する。
- 省略する機構と理由: 初段では全graph decodeと公式評価を進行条件まで省略する。neutral controlはユーザーの比較目的により省略する。
- proxyで検証できない主張: N/A。proxyではないが、pair診断だけでは長い軌跡、最終復号、公式combined scoreを判断できない。
- proxyの場合のユーザー承認: N/A。
- この実験が支持 / 棄却できる主張: 固定first-pass履歴の直前移動をpair logitへ加える今回の表現が、exp016より両胚の固定候補上の接続診断を改善するか。
- この実験では判断できない主張: 時間情報全体、位置列model、カルマン法全体、周囲の運動、独立CV、Public LB、hidden test。

## 実装方法

- アプローチ: 各outer foldの学習胚をsample単位のinner 2-foldへ分け、各target inner foldを除外して公開構造のhistory generatorを3 epoch学習する。outer評価履歴は、評価胚を学習していない保存済みexp016 fold modelから作る。履歴を固定してvelocity modelを公開checkpointから3 epoch学習し、保存済みexp016 fold modelを同じ外側windowで評価する。
- inputの実装箇所と変換: `frozen_tracker.py`がcandidate IDを含むcache、履歴配列、教師を整列する。`velocity_tracker.py`が15次元pair特徴とzero-initialized delta logitを作る。
- target / objectiveの構築箇所: exp016から継承した `greedy_match_candidates`、`build_legacy_edge_target`。
- outputの生成箇所と表現: `models/fold_<n>/velocity_tracker_best.pth`、`history_models/outer_<n>/inner_<k>/history_tracker.pth`、history manifest、fold summary、pair diagnostic、model manifest、metrics。
- lossの実装箇所: `frozen_tracker.py`の既存lossを変更しない。
- decode / postprocessの実装箇所: `frozen_tracker.py`のfixed first-pass history生成。全graph段階はgate通過後に実装する。
- context unitを保つ処理箇所: cache pathのsample、window frame、candidate IDを検査し、履歴辞書をsampleごとに初期化する。
- 変更するファイル / component: config、requirements、train Jupytext source/Notebook、`frozen_tracker.py`、新規 `velocity_tracker.py`、実験固有tests、README、SESSION_NOTES、result、metrics。
- 固定事項を保つ確認方法: config test、公開source/checkpoint/cache SHA、baseline fold割当、teacher/lossの既存test、target sample exclusion assert、base tracker単体のexp016指標再現。
- 参照sourceとの一致を確認するテスト: 公開tracker stateをstrict loadし、velocity branch初期化時のlogitがbase trackerと一致することを確認する。
- 承認済み差分を確認するテスト: synthetic 3-frame系列で移動・残差・履歴長・確率を確認し、欠測と開始時はdelta 0、予測分裂後は娘をreset、candidate ID不一致は停止することを確認する。

## 探索幅とpivot判定

- 変更class: `add-only`。予測対象、loss、decodeを変えず、履歴由来pair特徴とdelta branchだけを追加する。
- 同じ親 / familyで連続した小改善実験数: exp024のepoch変更、exp027の未対応近傍3時点入力とは変更対象が異なる。今回は対応済み予測履歴を明示する承認済みbacklog候補である。
- positiveなoracle headroom / coverage / 誤差非相関性: exp015のcandidate edge recallとexp027の親順位診断から固定候補内の改善余地は残るが、end-to-end改善は未確認。
- 比較したtarget、output、decode、context unitを変える案: position_history_tracker、shared edge graph learning、tracklet joinを別候補として維持する。
- 小改善の継続またはpivotを選ぶ根拠: exp027は未対応近傍を混ぜたが、同じ細胞と予測した履歴を速度として明示していない。今回の1比較でこの差を直接検証する。
- `kaggle-idea-forge` の実行要否と根拠: 不要。ユーザーが既存の反証可能なbacklog候補を選び、未決事項を解消して実験化した。

## 再現性・リスク

- seed policy: global seed 42、outer fold offset、history split seed 17、DataLoader generatorとworker seedを固定する。
- stochastic 処理の有無: DataLoader shuffle、dropout、CUDA kernel。
- stochastic feature generation / augmentation / seed bagging の有無: 履歴は保存modelと固定candidateから決定的に生成し、augmentationとseed baggingは使わない。
- 並列処理と乱数の関係: 学習順はfold別generator、workerはPyTorch seedからNumPy seedを設定する。履歴生成はsampleごとにframe順で行う。
- CPU/GPU runtime と deterministic flags: Kaggle T4、full precision。64 window benchmarkで履歴model 4個とvelocity model 2個を含む全工程を予測し、12時間を超える場合は停止する。
- train cache / test feature regeneration の SHA 記録方針: exp015 cache summary・identity、history feature schema/content、sample/fold/candidate coverageを記録する。
- model manifest / prediction / submission SHA 記録方針: history model 4個、velocity model 2個、history feature content、外側pair予測のSHAを記録する。submissionは生成しない。
- Kaggle package bootstrap 確認方針: prepare後にNotebook、config、metrics、`frozen_tracker.py`、`velocity_tracker.py`の一致をvalidatorで確認する。
- リークリスク: 公開初期重みの学習来歴は条件付き評価である。追加履歴はtarget sampleをgeneratorのgradientとcheckpoint選択から除外し、外側胚GEFFを条件選択へ使わない。
- CV/LB 不一致リスク: pair診断は公式graph metricではない。進行条件を満たすまで公式scoreを主張しない。
- ランタイム/メモリリスク: 新規6 tracker、履歴生成pass、pair特徴tensor。予算超過時にfold、epoch、対象sampleを黙って縮小しない。
- 再現性リスク: GPU学習はbitwise一致を主張せず、入力・特徴・model・予測SHAと数値を記録する。
- 手法忠実性リスク: velocityは物理時間でなくframe indexで正規化するため、秒当たりの物理速度とは呼ばない。
- 過度な縮小 / proxy化リスク: 正解履歴、in-sample履歴、固定運動費用へ置き換えない。

## 受け入れ基準

- [x] backlogの仮説、根拠、差分、固定事項、成功条件、停止条件、実行しないこと、判断履歴を移行した。
- [x] `config.yaml`のlineageとmodel数が本書と一致する。
- [x] history生成、velocity pair branch、診断testを実装する。
- [x] Jupytext round-trip、validate-exp、check-exp、test-expを通す。
- [x] Kaggle push前にGPU quota、velocity model 2、history model 4、control再学習なしを記録する。
- [x] Kaggle trainでruntime gateを確認し、2foldのpair診断とSHAを保存する。
- [x] pair診断gateを判定し、未達のため公式score未計測として停止する。
- [x] gate未達のため全graph inferenceと公式評価を追加しない。
- [ ] 実験の採用・不採用・完了は結果提示後のユーザー判断まで確定しない。
