# exp039_dog_features 要件と実装方法

この文書を、実装前の契約、実装方法、受け入れ条件の正とする。進捗はSESSION_NOTES.md、数値と実験statusはmetrics.jsonを正とする。

## 実験化の入口・引き継ぎ・承認

- 実験化の入口と承認: `backlog/dog_features.md`は`設計可能・実験化未承認`、未決事項`なし`。2026-09-23、ユーザーの「実装に進んでください」を実験化・実装の承認として扱う。
- 移行元backlog: `backlog/dog_features.md`。候補の内容をこの文書へ移した後、候補と未着手行を削除する。
- 対応する上位仮説: `HYP-20260920-03`。
- 上位仮説のうちこの実験が検証する範囲: 固定候補上の元画像DoG 2応答を点特徴へ追加した接続・分裂・全graphへの効果。
- この実験だけで上位仮説を判断できるか: いいえ。検出得点とHOGの効果・相補性が残る。
- 親実験: [`exp016_frozen_image_encoder`](../exp016_frozen_image_encoder/)。候補と固定cacheは[`exp015_oracle_stage_limits`](../exp015_oracle_stage_limits/)。
- 変更するもの: 元解像度DoG抽出、fold別標準化、64→66次元のprimary tracker入力拡張。
- 固定するもの: 親の公開検出器と画像特徴、候補・教師・損失・分割・復号。
- 最小の反証可能な検証: 66次元の特徴ありとゼロ入力対照を各2foldで学習し、両胚の隣接ペア診断を比較する。
- 成功条件: 隣接ペアの進行条件を通過した後、固定全graphの公式combined scoreが両胚で対照より改善し、division成分が悪化しない。
- 停止条件: 入力・候補対応・元画像との座標・SHAの不一致、予算超過、非有限特徴、進行条件未達では次段階へ進まない。
- 実行しないこと: 候補の追加・移動、画像モデルの更新、外側胚での方式選択、進行条件未達時の全graph、Kaggle submission。
- 未決事項: なし。
- backlog記録から解釈を変更した箇所とユーザー承認: 2026-09-23に点特徴の2応答追加をユーザーが選択した。設計本文の定義を変更しない。

## 判断履歴

- 2026-09-20: DoG候補をバックログ化し、入力方式を未決とした。
- 2026-09-23: ユーザーが点特徴への2応答追加を選択し、元解像度の特徴・同構造対照・進行条件を確定した。
- 2026-09-23: ユーザーが実装に進むよう依頼した。初段の元画像特徴生成と学習・隣接ペア診断を開始する。

## 手法契約

- input: exp015の固定候補cache、対応する競技train Zarr元画像、各点の2組のDoG応答。fold内勾配更新用動画だけで応答を標準化する。
- target / objective: 主催者GEFFの既知中心・接続・分裂と、親と同じ5 µm対応。
- output: 隣接候補ペアのprimary接続logitと4個のfold別tracker。候補座標・IDは生成しない。
- loss: 親のsource軸softmax、focal係数2のbinary cross entropy、既存mask。
- decode: 初段はgraphを作らず隣接ペア診断。進行条件を通過した場合のみ同実験で固定secondary寄与、ILP、修復、公式評価を実装・実行する。
- context unit: 同じ動画の隣接2フレームの候補と候補ペア。
- 実装区分: `staged-faithful`（このリポジトリ内の管理用語）。点特徴からの学習を省略せず、全graph処理だけを事前条件の後段へ分ける。
- 省略する機構と理由: 初段の全graph推論。進行条件未達時の費用を避けるため。
- 初段で判断できない主張: 公式combined score、Public LB、hidden test一般化。
- 具体的なファイル: `exp039_dog_features_features.py/.ipynb`で画像・cache対応とDoG生成、`exp039_dog_features_train.py/.ipynb`で4 tracker学習と診断、`dog_features.py`で特徴の純粋関数、`frozen_tracker.py`で教師・loader、`config.yaml`に全固定値。
- 入力証拠: exp015 summary/identity SHA、各元画像の実行時識別情報と特徴content SHA、fold別標準化統計、公開source/checkpoint SHA。
- 受け入れ基準: 2応答の順序と境界・座標の合成テスト、初期logit一致、教師mask・split一致、4 modelと診断・SHA、対象実験のvalidate/check/test、Kaggle実行での費用gateと出力照合。

## 実装方法

- 特徴生成Notebookは固定cacheの候補ID、座標、frameと元Zarr画像を対応させ、1/99.7 percentileでframeごとに正規化した元解像度3D画像から符号付きDoG 2応答を取得する。
- 候補座標では三線形補間を行い、frameごとの特徴とcontent SHAを保存する。GPU学習Notebookはそれを読み、勾配更新用windowの候補だけからfold別平均・標準偏差を算出する。
- `frozen_tracker.py`の既存の教師、mask、loss、隣接ペア処理を保ち、追加2列のみ切り替える。公開trackerの最初の線形層は旧64列を保持し、新2列を0初期化する。
- 2条件×2foldを同じseedと初期stateから学習し、外側胚の隣接ペア診断と進行条件を出力する。
- 2026-09-23の費用gate停止後、ユーザーが12時間上限を維持して同じDoG値を保つ高速化を選択した。互いに独立した元画像frameだけを2 workerで並列処理し、各frame内の正規化、4回のGaussian平滑化、補間、保存、SHA検証は変えない。各胚1 frameを逐次再計算して並列結果とcontent SHAの一致を検査する。

## 探索幅とpivot判定

- 変更class: `add-only`（このリポジトリ内の管理用語）。既存点特徴へ2列だけ追加する。
- 同じ親からの近接した特徴追加実験: exp035の速度、exp036の検出得点、今回のDoG。今回は2026-09-23にユーザーが方式・比較を選択済みで、設計を変えずに検証する。
- positiveなoracle headroomはexp015にあるが、この特徴だけで回収できる保証はない。誤差相補性と公式scoreは実行時に確認する。
- target、output、decode、context unitを変える案の比較や次のpivotは今回の固定実験の判定後に行う。結果を見て本実験の方式を追加探索しない。

## 再現性・リスク

- 特徴生成は決定的なCPU処理。モデル学習はfold別seedとworker seedを固定し、CUDAの非決定性を実行証拠に残す。
- 元画像、固定cache、公開source・checkpoint、DoG特徴、fold別統計、学習済みmodelのhashとKaggle kernel versionを記録する。
- 公開画像encoderと初期trackerの学習来歴を条件付き評価として扱う。部分注釈で確定できない誤接続を真の負例へ変換しない。
- 元解像度3D DoGの費用は両胚のframeを使って事前に計測し、12時間の枠を超える見積もりなら停止する。推論の費用も別途必要になる。

## 受け入れ基準

- [ ] 元画像と固定候補の座標、ID、順序、frame、SHAを一致させる。
- [ ] 2組の符号付きDoG応答と境界処理を合成画像で検証する。
- [ ] 学習用windowだけからfold別統計を算出し、66列ゼロ対照と特徴あり条件の初期出力を一致させる。
- [ ] 2条件×2foldの4 trackerを同じ教師と学習条件で実行する。
- [ ] 両胚の接続・分裂・確定できる誤接続、強度・DoG・境界別診断と進行条件を保存する。
- [ ] `check-exp`と`test-exp`が通り、Kaggle上の実行証拠と生成物SHAが揃う。

## 移行した候補契約

## 観測事実と根拠

- 実測済みの事実: DoG（Difference of Gaussians、ガウス差分）は同じ画像を異なる幅で平滑化した差である。保存済みPilkwang公開Notebookの`CONFIG_OVERRIDE.dog_scales`は`[[1.5, 4.0], [2.2, 5.5]]` µm、`detect_blobs`はframeごとの1 / 99.7 percentile正規化と、小さいぼかしから大きいぼかしを引く応答を使う。同Notebookはxyを4分の1へ間引き、2応答の最大から検出点を生成する。今回は元解像度・固定点・2応答の個別入力へ変更するため、公開検出器や公開スコアの再現とは呼ばない。exp016へDoG特徴を追加した精度の証拠はなく、既存cacheだけからDoG応答を復元できるとは扱わない。
- 根拠ファイル / 一次資料: [保存済み公開Notebook](../../docs/notebooks/biohub-cell-tracking-during-development/pilkwang__biohub-cell-tracking-data-model-eda-baseline/biohub-cell-tracking-data-model-eda-baseline.ipynb)の`detect_blobs`と`CONFIG_OVERRIDE`、[既存のDoG解説](../../docs/surveys/biohub-tracking-techniques-illustrated_20260915.md#dog)、[データ仕様](../../docs/04_data.md)、[exp016の特徴連結・教師・診断](../exp016_frozen_image_encoder/frozen_tracker.py)、[学習初期化](../exp016_frozen_image_encoder/exp016_frozen_image_encoder_train.py)、[exp036の比較設計](../exp036_detection_score_pair_features/requirements.md)。一般的なDoGの参照は[scikit-image blob_dog](https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.blob_dog)、[Lowe, 2004](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf)。今回確認したのはローカルの保存版とコードであり、新たな改善実証はない。
- 利用する保存済み生成物とSHA: [exp015 metrics](../exp015_oracle_stage_limits/metrics.json)と[exp016 metrics](../exp016_frozen_image_encoder/metrics.json)のcache・split・重み・対照を正とする。SHAは内容照合用のhashを指す。親configのsummary_sha256・identity_sha256と取得物を照合し、追加生成物のSHAは実行時に記録する。未生成の特徴のSHAを捏造しない。
- 仮定: Assumption: 局所的な明るい塊の応答を2つの大きさで明示すると、固定画像特徴だけでは曖昧な接続を補える。スケールが細胞の真の半径を表すとは仮定しない。

## この候補が直接検証する仮説と範囲

- 上位仮説のうちこの候補が検証する範囲: 元画像にDoGを適用し、既存検出点で取り出す応答を追加する効果。DoGで新しい検出点を追加する案は含めない。
- この候補の具体的な仮説: 固定検出点の2組のDoG応答を点特徴へ入力すると、同じ候補・教師・復号のゼロ入力対照より両胚の公式combined scoreが改善し、分裂成分が悪化しない。
- 仮説が正しい場合に期待する観測: 強度や大きさの変化で区別できる候補間の誤接続が減り、暗い細胞や分裂前後の回収を損なわない。
- 仮説を棄却する観測: 片胚のみの改善、撮像条件による悪化、または同じ入力構造の対照に対する改善がない。ただし初段だけでは全graphでの精度仮説を判定せず、この特徴表現を先へ進める証拠が不足したと報告する。
- この候補だけで上位仮説を判断できるか: いいえ
- 上位仮説の判断に残る検証: 検出得点・HOG（Histogram of Oriented Gradients、勾配方向ヒストグラム）の個別効果と相補性。DoGによる追加候補回収の効果はこの仮説の比較範囲外とする。

## 入力・予測対象・出力・推論方法

- input: 各固定検出点での符号付き3次元DoG応答2個。順序はぼかし幅`(1.5, 4.0)`、`(2.2, 5.5)` µmとし、学習用動画だけで各成分を標準化する。primary画像特徴32次元、位置特徴32次元、DoG 2次元の順に連結し、source・targetとも66次元の点特徴を使う。最大応答のスケール、前後差、検出得点、HOGは追加しない。
- target / objective: 主催者GEFF（Graph Exchange File Format）の既知中心・接続・分裂へexp016と同じ5 µm以内の距離順greedy one-to-one対応を行い、隣接時刻の通常接続と母娘接続を予測する。
- output: 既存と同じ隣接候補ペアの接続logit。検出点や新しい細胞IDを生成する出力へ変更しない。
- loss: exp016のsource軸softmax、focal係数2のbinary cross entropy、正例source行またはtarget列に触れるpair mask、division重み1を固定する。既存maskは未対応候補を負例として含み得るため、その限界を継承して件数を記録する。未知領域を追加で負例にしたり、今回だけmaskを変更したりしない。
- decode / 推論方法: 点特徴を最初の線形層へ渡し、後続の候補間照合・接続logit生成はexp016と揃える。隣接ペア診断を先に実施し、進行条件を満たした場合に同じsecondary tracker寄与・整数線形計画（ILP）・修復で全graphを比較する。逆方向では既存入力と一緒にsource・targetのDoG配列を交換する。候補の座標は動かさない。
- 処理単位: 同じ動画の隣接2フレームの固定候補と候補ペア。追加情報も同じ候補IDと画像窓へ対応付ける。
- 実装区分: 特定の追跡論文全体の再現ではなく、元解像度のDoG応答を点特徴へ入力して接続を学ぶ比較。特徴抽出・学習・隣接診断を先に行い、全graph評価を進行条件の後へ分ける。このリポジトリ内の管理用語では`staged-faithful`とする。

## 特徴抽出と保存の確定仕様

| 項目 | 初回に固定する処理 |
| --- | --- |
| 画像 | `project.yml.data`のtrain / test配下で、cacheのdataset名とframe番号に対応するZarrの元解像度3次元画像をfloat32で読む。xy間引き、再標本化、時間方向の平滑化は行わない。 |
| 画素間隔 | 元画像のz/y/xの物理間隔をµmへ揃え、Gaussianの各軸sigmaは`指定した幅_um / spacing_zyx_um`とする。現データは`[1.625, 0.40625, 0.40625]`。metadataとcache座標の整合を検査し、単位や対応を確定できない入力では停止する。 |
| 強度正規化 | 各元frame全voxelの1 / 99.7 percentileを線形補間で計算する。`hi <= lo`なら`hi = lo + 1.0`、`I = maximum((raw - lo) / (hi - lo), 0)`。上限clipなし。公開Notebookの規則を元解像度で使い、既存画像encoderの前処理は変更しない。 |
| Gaussian | SciPyの`ndimage.gaussian_filter`、`order=0`、`mode="reflect"`、`truncate=4.0`、float32入出力。幅1.5、4.0、2.2、5.5 µmの4回を順次計算する。実行版を記録する。 |
| 応答 | 小さい幅の平滑化画像から大きい幅の平滑化画像を引き、2応答を別々に残す。負の値も保存する。応答の最大化、負値を0にする処理、sigmaによる追加倍率、細胞半径への変換は行わない。 |
| 抽出座標 | cacheの`coords_*_physical / spacing_zyx_um`を元画像のvoxel中心座標として三線形補間する。現cacheの`coords_*_grid * [1,4,4]`とも照合する。整数丸め、半voxel移動、画像外へのclampは行わず、範囲外なら停止する。 |
| 境界 | Gaussianの反射はdataset画像の外縁にだけ適用する。候補ごとの小patchで独立にぼかさない。補間は画像内の座標だけに適用する。 |
| 計算単位 | 同じdataset・frameの全応答を1回計算して、必要な窓の固定候補へ取り出す。保存するのは候補点の2応答と監査情報とし、全frameの応答画像を一括保持しない。 |
| 候補対応 | 保存キーはdataset、`window_frames`、source / target、candidate ID。配列順、mask、grid / physical座標、参照cacheのcontent SHAを保存して照合する。同じframeの別窓でも、候補IDだけを根拠に配列を使い回さない。 |
| 空・異常入力 | 候補0個は`(0,2)`配列を保存し、padding候補は正規化統計から除外して入力を0にする。定数画像は正規化後0でDoGも0。画像・特徴の欠損、非有限値、候補対応の不一致を0で埋めず停止する。 |

画像のframe内percentileは、推論対象自身にも適用できる決定的な前処理であり、教師や他動画を使うfitではない。一方、DoG応答2成分の平均・標準偏差は、fold内で勾配更新に使う動画の有効候補だけからfloat64で集計する。内部検証・外側胚・testを含めず、各学習窓のsource / targetの出現を1標本ずつ数える。重複したframeの出現もこの規則どおり数え、集計途中で重複除去規則を変えない。

標準化は成分ごとに`(response - mean) / max(population_std, 1e-6)`とし、出力をfloat32にする。両条件で同じ統計を使い、ゼロ対照は標準化後の2成分を0へ置換する。応答の生値0を標準化して対照とする方法は使わない。fold統計、件数、fit対象manifest、強度percentile、spacing、sigma、境界条件、元画像と候補cacheの識別情報・SHAを特徴manifestへ記録する。追加生成物のSHAは実行時に取得する。

## 点特徴の入力と初期化

- 公開primary trackerの最初の線形層`proj`だけを64入力から66入力へ拡張する。既存の64列・biasと後続parameterを公開checkpointからコピーし、DoGの2列を0初期化する。hidden dimensionは128のままなので、追加parameterは256個である。
- DoGありとゼロ対照の両方で同じ66入力の構造を使い、primary tracker全体と追加列を学習する。画像側・secondary trackerは固定する。ペア特徴からlogitへ加える別headは作らない。
- 初期値はexp016と同じ公開tracker checkpointとする。保存済みexp016再学習重みからの追加3 epochにはしない。保存済みexp016モデル・指標は別の比較対象として残す。
- 両条件のstateを同一にし、foldごとに同じseed、DataLoader順、dropoutの乱数開始状態を設定する。初期logitは両条件で一致し、64入力の公開trackerともfloat32の`rtol=1e-5, atol=1e-6`内で一致することを実装時に検査する。追加列以外のcheckpoint tensorは完全一致を要求する。学習後のbitwise一致は要求しない。

## 親実験からの差分

- 変更するもの: 固定候補へのDoG抽出・保存・標準化、点特徴の64→66次元連結、最初の線形層の入力2列追加。primary trackerを同条件で再学習して情報の有無を比較する。
- 固定するもの: 公開検出器・画像特徴抽出器の重みと正規化統計、画像窓・既存特徴、検出候補・座標、接続候補の生成規則、教師・損失・mask、胚split・内部選択、secondary tracker・ILP・修復。初回は他の特徴量を同時追加しない。
- 再利用するコード / config / 生成物: exp015のcacheと候補対応、exp016のconfig・教師作成・学習・隣接ペア診断・公式graph評価。ハイパーパラメータは実験化後のconfig.yamlに置く。
- 新しく作るもの: この実験で3次元DoG抽出、候補IDとの対応、特徴cacheとmanifest、fold別正規化統計、66入力tracker、応答・強度・境界別の診断を作る。生成物は実験の`artifacts/`へ置く。

## 最小の反証可能な検証

- 検証方法: 画像32次元・位置32次元を共通とし、DoG 2次元ありと標準化後の0対照を比較する。初期値・学習量を揃え、保存済みexp016と公開trackerも参照する。4 modelを同じ外側windowと教師で評価する。
- variant / config / fold / booster数: 2条件×外側2fold、計4 tracker、booster 0、追加seed・設定探索・履歴生成用学習0。fold 0は44b6で学習して6bbaを評価、fold 1は逆。exp016の内部動画分割seed 0・10%検証、公開初期値、3 epoch、AdamW、learning rate 0.0001、weight decay 0.01、batch size 2、seed 42とfold offset、augmentationなし、mixed precisionなしを継承する。内部checkpoint選択は`edge_accuracy_times_candidate_node_recall`のままとする。
- control再学習: 同構造のゼロ対照を2foldで学習し、入力拡張と情報の効果を分ける。保存済みexp016の参照値だけを理由なく再学習しない。
- 想定runtime / resource: Kaggle Notebookのみ、週30 GPU時間以内、各Notebookと提出推論は12時間以内。DoG抽出はCPU Notebookを初回計画とし、元画像読出し・percentile・4回の3次元平滑化・候補抽出・保存の時間とメモリを計上する。保存済み資料ではtrain 199動画はいずれも100 frame・64×256×256 voxelであり、元解像度での費用は未測定。
- 実行前の費用確認: 各胚で画像voxel数が最大の動画を選び、同数ならdataset名の辞書順で決める。各10 frameを等間隔に選び、2 workerでの並列読出しから保存までのwall timeを測定する。各胚のbatch wall timeを10で割った実効1 frame時間の大きい方で全対象frame数へ外挿して1.5倍し、setupを加えて12時間以内か確認する。逐次1 frameの再計算と内容SHAが一致することも要求する。学習は両条件・各foldの64 window benchmarkから4 model全体を見積もる。全graphの費用測定は隣接ペア診断通過後に行い、両胚各1動画でsecondary・ILP・修復・公式評価まで含める。GPU割当数を含め残quotaと週30時間に収まらなければ停止し、解像度・scale・fold・epochを自動で減らさない。
- 全graph推論への進行条件: 次節の両胚別の隣接ペア診断をすべて満たし、候補・教師の固定条件と費用確認が通ること。条件未達なら公式score未測定として停止する。単体指標を公式scoreの代用にせず、長い軌跡の整合性と最終復号の効果は全graphの公式評価で確認する。

## 隣接ペア診断と全graphへの進行条件

exp016と同じ、正解注釈（GT）が片方のframeにないwindowの除外、source軸softmaxと`probability > 0.5`の判定を使う。全graph側の0.48 thresholdへ単体診断を変更しない。分母・分子を胚別に集計し、以下を**両胚それぞれ**で要求する。低下を許容する幅は0とし、集約平均で片胚の悪化を相殺しない。

| 診断 | 判定と必要件数 |
| --- | --- |
| 既知接続の回収 | `positive_edge_recall`が同構造対照を厳密に上回る。同一分母なので少なくとも1本多い回収を必要とする。期待する正例数は6bbaが103,393、44b6が18,949。 |
| 親と同じmask内の正解率 | `edge_accuracy`が対照以上。未注釈を含む既存mask上の指標と明記し、真の誤接続率とは呼ばない。 |
| 分裂の回収 | 既知の娘接続をすべて回収した母の割合`division_parent_recall`が対照以上。期待する母数は6bbaが108、44b6が22。候補上で表現できない分裂は別に報告する。 |
| 注釈から誤りを確定できる接続 | targetが隣接元frameに既知の母を1つ持ち、両端の候補が既知nodeへ対応済みの場合に、その母以外のsourceを選んだ本数が対照以下。判定可能な負例pair数を分母として併記し、両胚で分母が正であることを要求する。 |
| 注釈と矛盾する娘の組 | 同じ予測sourceが2 targetを選び、両targetの既知の母が互いに異なる組の数が対照以下。隣接元frameの母が各1つ確定したtargetだけを使い、組を重複計数しない。判定対象source数・target数と、選択結果によらない判定可能なsourceと2 targetの組の総数を併記し、両胚で組の総数が正であることを要求する。これは全分裂のprecisionではない。 |
| 入力と教師の一致 | 評価動画は6bbaが128、44b6が71、有効windowは12,392と6,315。正例数・分裂母数を含めexp016と一致し、両条件のmask・対応・候補数も一致する。分母0を成功として扱わない。 |

追加で、各DoG成分の学習側25 / 50 / 75 percentileで分けた4群、正規化画像の候補位置の強度を同じ規則で分けた4群、画像境界群、動画別の接続回収・判定可能な誤接続・分裂回収・有効件数を保存する。群の境界は勾配更新用動画だけで作り、外側結果に合わせて変更しない。群境界と同値の標本は上側へ入れ、同値の境界が重なって空の群ができても群数を変更しない。境界群はGaussianの最大半径または補間に必要な隣接voxelが画像外へ達する候補を含む群とし、対照と同じ候補で比較する。接続と誤接続はtargetの群、分裂はsourceの群へ割り当てる。分母0の群は欠測として示す。群別の値は診断として全件報告し、結果を見て新しい通過閾値を作らない。

誤りと確定できない未対応端点・未知の母・未記録の分裂は別件数として残し、真の負例へ変換しない。部分注釈のため、上の進行条件は公式指標の改善を保証しない。全体の候補回収と既知分裂の候補内表現率も併記する。

## 成功条件と停止条件

- primary指標: 現行公式combined score。adjusted edge Jaccard、division Jaccard、両胚別の結果と有効件数を併記する。
- 成功条件: 追加情報ありの条件が同構造対照より両胚の公式combined scoreを改善し、分裂成分を悪化させず、全対象で有効な出力を予算内に生成する。exp016に対する差も別報告する。
- 必須guard: 候補ID・時刻・座標・画像窓の一致、学習用動画だけでの特徴統計、内部検証だけでのcheckpoint選択、既存maskと未知教師の扱い、初期出力一致、特徴の欠損・非有限値、両胚・分裂・境界での差を確認する。固定値をconfig.yamlへ移し、Notebook内の暗黙の定数にしない。
- 成功時の次段階: 結果と実行証拠を提示し、採否・完了はユーザー判断とする。他特徴との組合せ、提出は別判断。
- 失敗時の停止範囲: 単体診断・予算条件未達では全graphへ進まない。公式score未測定と記録してユーザーへ示し、外側評価を見た救済探索や検出器更新へ切り替えない。

## 実行しないこと

- 禁止する代替実装、proxy、同一OOF上の救済探索: DoG極大値の候補追加、検出点の移動、検出器の更新、前後画像の差をDoGと呼ぶこと、元解像度から縮小画像への無断変更、外側胚でのスケール・正規化・閾値選択、条件未達後の追加seedや別headへの救済探索。OOFは学習から外した対象への予測を指す。
- 壁打ちで採らなかった案と理由: DoGで検出候補を増やす案はcandidate_unionに関係する別比較。応答画像全体を新しい画像モデルへ入力する方式も初回の範囲外。各端点の応答と絶対差を使う6次元のpair特徴からlogitを補正する案も提示したが、ユーザーが候補同士の照合段階へ情報を渡す点特徴を選択した。最大応答のscaleと前後差は追加する表現を増やすため初回に含めない。xy間引きは計算を減らせるが、選択された元解像度の比較とは別条件になる。

## リスク

- leakage / validation: 方式とスケールは本設計で固定、特徴標準化の統計は勾配更新用動画だけ、checkpoint選択は内部検証だけを使う。公開画像モデル・初期重みの学習来歴による条件付き評価であり、独立した交差検証（CV）とは呼ばない。予測履歴は使わない。
- hidden test: 元画像から同じ処理でDoGを再生成し、fold別統計を固定して使用する。画像の前処理、物理画素間隔、検出座標との対応、crop境界・空フレームを扱い、元画像なしで既存特徴から代用しない。最終提出用modelの学習・選択はこの初回2fold比較の範囲外。
- runtime / memory: 異方的画素間隔、輝度差、平滑化で重なる隣接細胞、スケール間の強度差、境界・paddingによる応答変化。抽出・読出し・学習・全graph推論の費用を分けて記録する。DoG抽出の追加費用を提出推論の見積もりにも含める。
- 再現性: 元画像・cache・重みの版、候補ID・窓、特徴の定義・正規化・単位、split・seed・設定と追加生成物のSHAを記録する。

## 調査・実行時に確認する事項

- 精度の追加効果、既存特徴との冗長性、候補・画像窓の一致、注釈から判定可能な誤接続の母数、特徴量の分布・欠損、fold別標準化統計、追加費用・メモリ・保存容量、実行前のquotaと生成物SHA。測定待ち自体を設計不可の理由にしない。exp036の成否を本候補の必要条件にしない。

## 実装時の受け入れ条件

- 元画像と候補の対応、2応答の順序、負値の保存、異方的spacing、境界反射、空候補を検査できること。一定画像の0応答と、小さい合成volumeでの直接Gaussian計算・候補サンプリングとの一致を確認する。
- 内部検証・外側胚を標準化統計へ含めず、同じraw特徴にfold別統計を適用できること。source / target交換、別windowの候補順違い、paddingを扱えること。
- 66入力trackerの0初期化と公開trackerとの初期出力一致、両条件で初期state・学習量が同じことを確認する。
- 2条件×2foldの4 model、入力・特徴・統計・modelのmanifestとSHA、胚別・群別指標、進行条件の機械判定と停止理由を保存できること。実装後は対象実験の`check-exp`と`test-exp`を実行する。

## 未決事項

- なし

## 判断履歴

- 2026-09-20: ユーザーのバックログ追加依頼により登録。入力表現など結果に影響する方式判断が残るため、状態は`検討メモ・設計不可`。この依頼は実験化・実装・Kaggle実行・submissionの承認ではない。
- 2026-09-20: 4案と既存候補を照合し、検出得点・DoGをP2、速度・HOGをP3とした。既承認実験の進行、他候補の優先度と依存は維持する。
- 2026-09-23: 設計確定依頼を受け、親実験の契約・config・特徴連結・教師診断、保存済み公開DoG実装、exp036のpair入力方式を照合した。元解像度で2組の応答を得て2条件×2胚で比較する共通案と、点特徴への追加 / pair特徴によるlogit補正の選択肢を提示した。
- 2026-09-23: ユーザーが「点特徴へ2応答を追加（推奨）」を選択した。初回scaleは保存済み公開Notebookの2組、強度変換は同Notebookの規則、学習条件と評価母数はexp016を根拠に固定し、追加列0初期化・学習側標準化・隣接ペアの進行条件を具体化した。状態を`設計可能・実験化未承認`へ変更した。依頼は設計の確定であり、実験化・実装・Kaggle実行・submissionは開始しない。
