# exp048_x138_primary_past_feature_attention 要件と実装方法

## 実験化の入口・引き継ぎ・承認

- 2026-09-26、ユーザーは `x138_primary_past_feature_attention` の設計を確定し、primaryと内部attentionの共同学習、親・娘の組別attentionをprimaryのpair MLPに加える方式を選択した。続く「実装に進んでください」は実験ディレクトリとコード・Notebookの実装依頼であり、Kaggleへのpush・学習・pair診断の実行依頼ではない。全graph推論とcompetition submissionも含まない。
- 2026-09-26、ユーザーが改めて「実行してください」と明示したため、実装済みtrain NotebookのKaggle push、2条件の学習、胚別pair診断を実行する。全graph推論とcompetition submissionはこの依頼にも含まれない。
- 移行元は `backlog/x138_primary_past_feature_attention.md`。候補の上位仮説は `HYP-20260910-10`、親の入力と推論は `exp043_x138_self_trained_head`、コードと同一pair対照の直接参照は `exp044_x138_past_candidate_knn_attention`。exp044の採否・完了は未判断。
- 上位仮説のうち、直前1時点の未対応候補の固定画像特徴・座標をprimaryの接続得点計算内で使う効果だけを検証する。他の過去時点数、予測履歴、候補生成、全graphの作用は残る。
- 同条件primaryのみ再学習2 checkpointは、2026-09-26の確定設計に明記され、ユーザーがその後に実装を依頼した対照である。公開primaryの保存済みlogitだけでは共同学習の効果と履歴入力の効果を分離できないため、対照の学習経路を実装する。GPUを使う再学習の実行には別途依頼を要する。

## 手法契約

- input: exp043の固定公開画像encoderが `[t,t+1]` で出す親・娘それぞれの画像特徴32次元と位置特徴32次元、補正済み座標。過去 `t-1` は `[t-1,t]` のsource特徴32+32次元から、親の物理距離で近い8候補を選ぶ。元窓、候補ID、座標と特徴の整列を検査する。
- target / objective: 主催者GEFFの既知接続。既知nodeと予測候補を7 µm以内で一対一対応し、正例行または列の部分注釈maskで学習する。分裂は既知親の全娘回収として診断する。
- output: 公開primaryの `SimpleNodeTransformer` 内で、親・娘の組からquery、過去候補の画像・位置・13次元変位特徴からkey/valueを作る1層4 head幅128のattentionを計算する。ゼロ初期化の追加線形射影をpair MLPの最初の隠れ層へ加え、既存logitを出す。完成後logitへの後付けdeltaにはしない。
- loss: exp044と同じsource軸softmax後のfocal BCE、gamma 2。primary全体と追加層を共同学習する。検出器・画像encoder・座標head・secondaryは固定する。
- decode: forwardとreverseで実時間 `t-1` の同じ履歴を各親・娘の組に対応させ、exp043の双方向harmonic、secondaryのlow-margin consensus、閾値0.48、後続の候補選択・ILP・後処理を保持する。初段はpair診断で止める。
- context unit: 同一動画の直前窓と現在の隣接pair、学習はpair、検証分割は胚。親・娘の各組で最大8過去候補を参照する。
- 実装区分: `staged-faithful`（このリポジトリ内の実装管理用語）。確定した内部attention、正逆採点、学習、同一pair診断を初段で実装する。全graphの公式評価はpair進行条件と費用の実測後に判断する。
- 変更class: `representation`。過去の画像特徴を組別にprimary内部へ導入し、exp044の座標だけのpost-logit補正とは入力と作用箇所を変える。

## 実装方法

- `x138_data.py`: exp044 captureの現在窓と直前窓のsourceを候補ID・補正済み座標・元窓・SHAで結合する。動画途中の直前窓欠損は停止し、動画先頭・真の空集合だけmaskにする。
- `primary_pair_attention.py`: 公開primaryのprojection、4 cross-attention block、pair MLPを保持する。組ごとに物理距離8近傍の過去特徴と13次元幾何特徴を使い、4 headのsoftmaxに値0の過去不使用tokenを含める。pair MLP最初の隠れ層への射影をゼロ初期化し、過去全maskでは追加寄与を厳密に0にする。source32・target32の組へ分割する。
- `x138_tracking.py`: forward、reverseの両方へ同じ実時間の履歴を渡し、exp043のfusionを変更しない。primaryのみ再学習条件では追加attentionを完全に無効にする。
- `attention_train_pipeline.py`: exp044と同じ20動画、外側2fold、内部2動画、各3 epoch、同じ損失・optimizer・seed・pair順で共同学習2個とprimaryのみ2個を独立に学習する。内部active-pair error rate最小、同率なら既知edge回収数、さらに同率なら後のepochでcheckpointを選ぶ。
- `build_notebook.py`と正規train Notebook: exp043推論source SHA固定、公開artifactと座標head SHA検査、2動画pilot、captureまたは検証済みcaptureの再利用、baseline parity、K=8既知親保持率、費用診断、2条件学習、両胚のpair結果とSHA保存をセルごとに示す。
- 実装対象には正逆のpair-level logitまでを含める。全graph用に直前窓のsource特徴を逐次保持する経路は、pair進行条件達成後に同じ実験へ追加して検証する。学習後のモデルで公式scoreを主張するにはKaggle上の全graph・公式評価器が必要。
- model/config/fold: 追加attention 1構造、primaryのみ対照1構造、外側2fold×各3 epoch、合計新規checkpoint4個、booster 0。公開primaryは固定対照として同一pairで再評価する。

## 比較・成功条件・停止条件

- 初期eval時の公開primary forward、reverse、融合後logitとの最大絶対差は1e-4以内。学習後の過去全maskは同じ重みの履歴なし経路と一致する。候補ID・元窓・座標・特徴SHA、K=8既知親保持率99%以上を学習前に確認する。
- 両胚それぞれで共同学習の既知edge回収数が公開primaryとprimaryのみ再学習の両対照より多く、active-pair errorsは両対照以下、既知分裂親の全娘回収数は両対照以上なら全graphを検討する。分裂分母0は判定不能とする。正解親順位・得点差、教師負例予測、密集度、閾値別recall、過去全mask、attention参照先と追加寄与も診断する。
- 入力不一致、非有限値、保持率不足、GPU memory不足、Notebook12時間または週45GPU時間の見込み超過、両胚のpair条件未達なら全graphを自動で実行しない。構造・候補数・epoch・閾値を外側結果で選び直さない。
- 2動画pilotと最大点数pairからcapture・特徴整列・両学習条件・評価を保守係数1.5で見積もり、実行前のKaggle quotaを確認する。Notebook経過時間とGPU割当消費を別に記録する。
- Kaggle version 1では最大pairの1 batch時間を全窓に適用して約31時間と過大見積もりし、学習前に停止した。再実行では最大pairのメモリ確認を保ったまま、固定した5つのサイズ区間の上端pairで両条件の更新・評価時間を測り、全窓の処理数へ積算して1.5倍の保守係数と既定の費用上限で判定する。モデル・教師・分割・epoch・閾値は変更しない。
- 公式score、上位仮説の支持・棄却、実験採否・完了は初段pair結果だけで確定しない。公開画像モデルの学習来歴により外側胚評価を独立CVとは呼ばず、部分注釈下のactive-pair errorsを真の誤接続総数とみなさない。

## 探索幅とpivot判定

- 初回は確定した1構造、直前1時点・近傍8点、外側2fold×各3 epochに限定する。公開primary固定対照、同条件primaryのみ再学習、共同学習、共同学習重みでの過去全maskを同じpair上で評価する。
- exp041の全候補画像特徴attentionとexp044の座標だけのpost-logit deltaは両胚のpair進行条件を満たさなかった。本実験は親・娘の組に依存する画像特徴をprimary内部へ渡す表現変更であり、head数やepochだけの延長ではない。外側結果に応じて構造、閾値、候補数を追加探索しない。
- 同じ機構を繰り返してもend-to-end改善が得られない場合は、接続の予測対象、組の出力、復号、時間文脈の単位を変える案を別途比較する。今回の具体的なユーザー選択を別案へ置換しないため、実装前の新しいアイデア生成は行わない。

## 再現性・リスク

- seed 20260924、動画選択seed 42。動画・fold・capture・特徴schemaと内容・公開primary・座標head・checkpoint・Notebook sourceのSHA、Kaggle kernel version、Notebook実行時間を構造化して保存する。CUDAとILPの完全bitwise再現は主張しない。
- 座標だけのpost-logit delta、source点だけの事前集約、正解軌跡入力、検出器・画像encoder・座標head更新、旧exp016 cacheの無条件流用、外側胚での構造・閾値救済、無断の全graph推論・submissionは実行しない。

## 受け入れ基準

- [ ] 元窓・候補ID・特徴32+32・物理座標・動画端、8近傍、teacher不使用が実験固有テストで確認される。
- [ ] 初期forward/reverse/fusion parity、過去全maskの厳密な追加寄与0、候補順序変更への対応、分割と非分割の出力・勾配一致が確認される。
- [ ] optimizerと凍結範囲、同じsplit・教師・更新回数、checkpoint選択、胚別分母、停止条件、費用gateが実装される。
- [ ] `make validate-exp`、`make check-exp`、`make test-exp`が通る。
- [ ] Kaggle Notebookで学習・pair診断を行い、2条件4 checkpoint、両胚の対照比較、SHAと実行証拠を保存する。

## 未決事項

- なし。captureの現存・Kaggle供給方法・精度・費用・quotaは実験化後の調査・実行時に確認する。

## 判断履歴

- 2026-09-26: ユーザーが「実行してください」と明示し、Kaggle上のtrain Notebook実行を承認した。
- 2026-09-26: ユーザーの指摘により、実装依頼をKaggle実行の承認と扱った解釈を撤回した。pushは2回とも同時GPUセッション上限で拒否され、Notebookの実行は始まっていない。以後は明示依頼までpushしない。
- 2026-09-26: ユーザーは直前候補の画像特徴をprimary内部で使う案をバックログ化した。
- 2026-09-26: ユーザーはprimaryと内部attentionの共同学習、親・娘の組ごとに直前8候補を参照してprimary pair MLP内へ加える初回設計を選択した。
- 2026-09-26: ユーザーが「実装に進んでください」と依頼した。実装範囲には同条件primaryのみ対照の学習コードを含むが、Kaggleへのpush・学習・pair診断の実行は依頼されていない。

## 移行元候補の記録

以下は実験化直前の候補詳細。候補段階の状態・未承認記述は履歴であり、現在の依頼範囲と実験statusは上記および `metrics.json` を正とする。

# x138_primary_past_feature_attention

- 候補名: `x138_primary_past_feature_attention`
- 状態: `設計可能・実験化未承認`
- 対応する上位仮説: `HYP-20260910-10`
- 関連する上位仮説: なし
- 作成日: 2026-09-26
- 最終更新日: 2026-09-26
- 依頼原文: 「primaryにも複座なattentionを組み込んだ方がよくないですか？」「過去候補の情報をprimary内部に入れる」への選択「こっちです」、直前frameの候補について座標だけでなく画像特徴も使う案への選択「後者がいいです」、続く「バックログに追加してください」。
- 期待する成果: 過去候補の画像特徴を接続logitの後付け補正ではなくprimary tracker内で親・娘の組の表現に反映すると、exp043の接続・分裂判断が改善するかを検証する。
- 親実験 / 比較対象: 採用済み[exp043](../exp043_x138_self_trained_head/)の入力・推論構成と公開primary checkpointを基準とする。[exp044](../exp044_x138_past_candidate_knn_attention/)の同じpair capture・公開primary対照を初段の比較に使う。exp044の実験完了・採否は未判断。
- 優先度: P4
- 優先度の理由: exp041の過去画像特徴attention、exp044の直前候補attentionは両胚の単体進行条件を満たしていない。親・娘の組に応じて過去の画像特徴をprimary内部で使う変更には検証価値があるが、同じ失敗を繰り返す費用を避けるためP4で保留する。実験化時は入力の現存・SHAと学習費用を確認する。
- `backlog/KAGGLE_DIRECTION.md` の対応箇所: [検証中の仮説](../../backlog/KAGGLE_DIRECTION.md#検証中の仮説)、[未着手バックログ](../../backlog/KAGGLE_DIRECTION.md#未着手バックログ)

## 観測事実と根拠

- 実測済みの事実: exp044の1823 pair窓では、直前8候補の座標由来deltaとprimaryを同時学習すると既知edge回収が44b6で3162→3144/3302、6bbaで7016→6884/7275となった。追加CPU診断で再学習primaryにattentionを加えた判定変更は各胚1・2 pair、公開primaryから再学習primaryへの置換では72・265 pairだった。再学習による変化との整合はあるが、共同学習の因果分解や本案の効果は未検証。公式scoreは未計測。
- 根拠ファイル / 一次資料: [exp044 result](../exp044_x138_past_candidate_knn_attention/result.md)、[exp044 metrics](../exp044_x138_past_candidate_knn_attention/metrics.json)、[exp044 requirements](../exp044_x138_past_candidate_knn_attention/requirements.md)、[exp041 result](../exp041_past_feature_cross_attention/result.md)、[exp041 requirements](../exp041_past_feature_cross_attention/requirements.md)、[exp043 config](../exp043_x138_self_trained_head/config.yaml)。本案はこれらの結果を受けたユーザーとの設計案であり、改善実証ではない。
- 利用する保存済み生成物とSHA: exp044 `artifacts/kaggle_v1_diagnostic_inputs/`のpair captureとfold checkpointを候補とし、Kaggle receipt・source・checkpoint SHAは[exp044 metrics](../exp044_x138_past_candidate_knn_attention/metrics.json)を正とする。未追跡生成物の現存とSHAは実験化・実行前に再確認する。別窓で抽出した同一frameの特徴を黙って代用しない。
- 仮定: 直前frameの複数候補の見た目と相対位置をprimary内部の親・娘の組表現に反映できれば、座標だけのpost-logit deltaでは変わらなかった競合親順位を変えられる可能性がある。exp044の悪化原因がarchitectureだけだったとは仮定しない。

## この候補が直接検証する仮説と範囲

- 上位仮説のうちこの候補が検証する範囲: 直前1時点の未対応候補集合の固定画像特徴・座標を、exp043 primary trackerの接続得点生成より前に取り込む効果。
- この候補の具体的な仮説: 同じ検出候補・座標補正・固定画像特徴・教師・後処理の下で、親・娘の組に応じて直前候補を参照するattentionをprimaryのpair MLP（組の特徴から接続logitを出す多層パーセプトロン）へ追加すると、公開primary対照および同条件のprimaryのみ再学習対照より両胚の既知接続を改善し、既知edgeに接する行または列の誤判定数（active-pair errors）と既知分裂親の全娘回収を悪化させない。
- 仮説が正しい場合に期待する観測: 公開primaryとの初期logit parityを保ったうえで、過去特徴ありの学習後に正解親順位・既知edge回収が両胚で改善し、過去をmaskした同一重みとの差が入力利用を裏づける。最終graphへ進める場合も公式成分を確認する。
- 仮説を棄却する観測: 確定した初回構成で同一pair対照の進行条件に届かない、分裂や片方の胚が悪化する、入力整列または予算が成立しない。本候補の結果を多時点情報全般の棄却とはしない。
- この候補だけで上位仮説を判断できるか: いいえ
- 上位仮説の判断に残る検証: 他の過去時点数、予測履歴、候補生成、全graphでの作用と公式score。

## 入力・予測対象・出力・推論方法

- input: exp043の現在の隣接2-frame候補・補正座標・公開primary画像特徴32次元と位置特徴32次元に、直前frameの物理距離近傍8候補の同じ64次元特徴・補正座標・候補ID・有効maskを追加する。過去の正解対応や確定予測軌跡は使わない。特徴の元窓と結合規則は「確定した初回設計」を正とする。
- target / objective: 現在の隣接2-frameで主催者GEFFに記録された既知接続・分裂。疎い注釈の未知部分を真の負例とはみなさない。
- output: 親・娘の各組に対する過去候補のattention出力をprimaryのpair MLPの最初の隠れ層へ加え、既存と同じ隣接frame候補間の接続logitを出す。完成後の接続logitへ別branchのdeltaを足す構成にはしない。
- loss: exp044と同じGEFF既知edge教師、正例行または正例列のmask、source軸softmax後のfocal binary cross-entropy（BCE、gamma 2）。公開primaryの全パラメータと内部attentionを共同学習し、画像encoder・座標head・secondaryは固定する。
- decode / 推論方法: exp043の双方向採点、secondary融合、接続候補選択、ILP、後処理を固定する。両方向で同じ実時間の過去候補を参照し、逆方向では親・娘の組に対応する履歴を転置して渡す。直前窓由来の特徴を逐次供給し、動画先頭の過去なしはmaskで扱う。
- 処理単位: 同一動画の直前窓と現在の隣接pair。各親候補・娘候補の組ごとに、その親候補に最も近い直前8候補を参照して接続logitを計算する。
- 実装区分: 既存手法の忠実再現ではなく、exp043 primaryへの明示的な構造変更。exp041の過去3時点・全候補attentionやTrackastraの窓内対応学習とは別の比較とする。

## 親実験からの差分

- 変更するもの: 直前候補の固定画像特徴・座標を参照する親・娘の組別attentionをprimaryのpair MLP内部に加え、公開primaryと追加層を共同学習する。pair MLPの追加入力射影をゼロ初期化して公開primaryの初期logitを再現する。
- 固定するもの: 公開検出器・画像encoder・座標補正head・secondaryの重み、候補生成、現在pairの特徴と教師対応、明示差分以外のexp043推論。学習量を変えた対照は別に記録する。
- 再利用するコード / config / 生成物: exp043のprimary構造と推論、exp044のpair capture・baseline parity・胚別評価・物理距離近傍8点とexp041の入力整列・ゼロ初期化テストを参照する。ただし旧exp016 cacheをexp043入力と同一視しない。
- 新しく作るもの: 実験化後にprimary pair MLP内部の組別attention、直前窓の画像特徴整列、推論時の逐次履歴保持、primaryのみ再学習対照と実験固有テスト。今回作成するのは候補詳細と索引だけ。

## 最小の反証可能な検証

- 検証方法: exp044と同じpair captureから直前窓の画像特徴を候補IDで整列する。公開primary、同条件でprimaryだけを再学習した対照、primaryと内部attentionの共同学習を、同じfold・pair・教師・損失・checkpoint選択・閾値で比較する。学習済みattentionの過去全maskも診断する。初期logit parity、両胚の親順位・既知edge回収・active-pair errors・分裂親の全娘回収を確認する。
- variant / config / fold / booster数: 直前1時点・物理距離近傍8点・1構造、外側2fold、各3 epoch。共同学習2 checkpointとprimaryのみ再学習2 checkpointを作り、boosterは使わない。
- control再学習: exp044の共同学習済みprimaryをprimaryのみ対照に流用しない。公開primary checkpointから同じ乱数seed・動画分割・教師・optimizer・学習量で独立に再学習する。公開primary固定の保存済みlogitも同じpair上で再評価する。
- 全graph進行: 「確定した初回設計」の両胚別数値条件と費用条件が成立しても、結果をユーザーに示してから判断する。pair指標は公式scoreの代わりにしない。

## 成功条件と停止条件

- primary指標: 両胚別の既知edge recall、active-pair errors、既知分裂親の全娘回収。正解親順位・得点差、閾値別recall、教師負例予測数、attentionの参照先と残差も診断する。全graphへ進む場合の主評価は公式combined scoreと接続・分裂成分。
- 成功条件: 両胚それぞれで共同学習モデルの既知edge回収数が公開primaryと同条件primary再学習の両対照を上回り、active-pair errorsが両対照以下、既知分裂親の全娘回収数が両対照以上。0.48以外の閾値や外側評価のepochで救済しない。
- 必須guard: 初期forward・reverse・融合logit parity、過去全mask、候補IDと特徴元窓の一致、近傍8点の既知親保持率、教師mask、非有限値、動画端、胚別の分母と費用。保持率がどちらかの胚で99%未満なら学習前に停止する。
- 成功時の次段階: pair結果と全graphの費用を提示し、全graph・公式評価へ進むかユーザーに確認する。採否・完了・submissionも別判断。
- 失敗時の停止範囲: 両胚の条件未達、入力不整合、近傍保持率不足、予算超過なら全graphへ自動進行せず、公式score未計測として報告する。候補数・epoch・閾値を外側結果で選び直さない。

## 実行しないこと

- 禁止する代替実装、proxy、同一OOF上の救済探索: 座標だけのpost-logit delta、source点だけの事前集約への置換、過去正解軌跡の入力、検出器・画像encoder・座標headの更新、exp016旧cacheとの無条件比較、外側胚での構造・閾値選択、無断の全graph推論・submission。
- 壁打ちで採らなかった案と理由: ユーザーは過去候補の画像特徴をprimary内部で使い、primaryと内部attentionを共同学習する案、親・娘の組ごとに過去8候補を参照してpair MLPへ加える案を選択した。座標だけのpost-logit deltaはexp044と作用箇所が同じになる。source点だけを事前集約する案はexp041の分析で指摘された、組ごとの識別情報が失われる懸念が残る。公開primary凍結で追加層だけ学ぶ案は初回の主実験にしない。[x138_window_features](../../backlog/x138_window_features.md)は前後窓特徴を広く扱い、こちらは直前候補集合からの内部参照に限定する。

## リスク

- leakage / validation: 公開画像モデルと座標headの学習来歴を明示し、外側胚評価を独立CVとは呼ばない。注釈の有無で過去候補を選ばない。
- hidden test: 同じframeでも画像特徴は入力2-frame窓に依存する。学習と推論で直前窓のsource特徴を同じ規則で使う。初回pairまたは直前候補0件はmaskで扱い、動画途中の想定外の直前窓欠損は停止する。
- runtime / memory: 親候補数×娘候補数×近傍8点のattentionをpair単位で分割計算し、前窓特徴を逐次保持する。exp044の座標deltaと異なる追加計算・共同学習・対照再学習を実測する。pair captureのローカル存在だけでKaggle実行時間や入力可用性を保証しない。
- 再現性: 特徴元窓・候補ID・座標単位・モデル/source/capture SHA、初期化、凍結範囲、学習設定、checkpoint選択、maskと閾値を記録する。

## 調査・実行時に確認する事項

- exp044 capture 1980ファイルのうち先頭20件以外の直前窓との候補ID・座標・32次元画像特徴の対応はローカルで事前確認済み。ただし入力の現存・SHA・Kaggle側への供給方法は実験化時に再検証する。追加モデルの精度・費用・公式scoreは未測定。

## 未決事項

- なし

## 判断履歴

- 2026-09-26: ユーザーは過去候補の情報をprimary内部に入れる方針を選び、直前候補の座標だけでなく画像特徴も使う案を選択し、バックログへ追加した。今回の設計確定依頼では、公開primary凍結ではなくprimaryと内部attentionの共同学習、source点だけの事前集約ではなく親・娘の組ごとに過去8候補を参照してpair MLPへ加える案を選択した。実験化・実装・実行は未承認。

## 次セッションへの引き継ぎ確認

- 固定するものを一意に説明できる: はい。exp043の固定検出器・画像encoder・座標head・secondary・明示差分以外の推論。
- 変更するものを一意に説明できる: はい。親・娘の組ごとに直前8候補の画像特徴・座標をprimary pair MLP内部へ渡し、primaryと追加attentionを共同学習する。
- 最小検証と停止条件を一意に説明できる: はい。同一pairの公開primary・同条件primaryのみ再学習・共同学習・過去全maskを比較し、両胚別の既知接続・active-pair errors・分裂の条件を使う。
- 実行しないことを一意に説明できる: はい。post-logit deltaへの置換、画像側の更新、外側結果での救済探索、無断の全graph推論・submission。
- 未決事項が明示されている: なし。実験化を依頼された場合は「確定した初回設計」を契約として引き継ぎ、結果・費用は実行時に確認する。

## 確定した初回設計

2026-09-26の設計確定依頼と学習範囲・挿入位置への回答を反映した、実験化時の契約。実験番号の採番、実装、Kaggle実行、実験の採否は今回の依頼に含めない。

### 入力窓と候補の整列

- 対象は同一動画の隣接pair `[t,t+1]`。対象の親 `t` はこの窓の `src`、娘 `t+1` は `tgt` の公開primary画像特徴32次元と位置特徴32次元を使う。過去 `t-1` は直前窓 `[t-1,t]` の `src` の同じ64次元特徴を使う。同じframeを別窓で抽出した特徴へ置き換えたり平均したりしない。
- 直前窓の候補IDと補正済み物理座標を現在窓の `candidate_ids_prev` と `coords_prev_grid` に対応付け、画像・位置特徴を同じ並びへ並べ直す。候補の重複、座標・元窓・特徴SHAの不一致、動画途中の直前窓欠損は停止条件。動画先頭または実際に直前候補が0件の場合だけ過去なしmaskを使う。
- 各親候補から直前frameの全有効候補を補正済み物理座標の距離で並べ、近い順に8点を選ぶ。同距離は候補ID順で決め、8点未満はmaskで埋める。教師node ID、接続正解、予測軌跡を選択に使わない。exp044と同じ既知親候補の保持率を両胚別に測り、どちらかが99%未満なら学習前に停止する。

### primary内部のattentionと正逆採点

- 公開 `SimpleNodeTransformer` の入力64次元、隠れ幅128、4個の既存cross-attention block、既存のpair MLPとその出力logitを維持する。各親・娘の組のpair MLP入力は、既存どおり両端の128次元表現と相対座標3次元を連結した259次元とする。
- 追加する1層・4 head・幅128のattentionでは、この259次元の組表現をqueryへ射影する。直前8候補ごとに、画像特徴32次元・位置特徴32次元と、exp044と同じ物理座標由来の13次元の変位・距離・方向特徴を連結してkeyとvalueへそれぞれ射影する。変位のscaleは5 µm、cosineのepsilonは1e-8に固定する。mask対象をsoftmaxから除き、値0の過去不使用tokenを1個含める。
- attentionの128次元出力をゼロ初期化した追加の線形射影でpair MLPの最初の128次元隠れ層へ加え、既存のGELUと残りのMLPを通す。追加射影以外の公開primary重みをcheckpointから厳密に読み込む。学習開始前のeval modeでは、過去の有無によらず公開primaryのforward、reverse、融合後logitと最大絶対差1e-4以内で一致させる。過去が全maskなら追加寄与を厳密に0とし、学習後も同じ重みの履歴なしprimary経路と一致させる。
- forwardは実時間の `t→t+1`、reverseは `t+1→t` を同じprimary重みで採点する。reverseでも履歴は実時間の `t-1` とし、親 `t` の候補IDで選んだ8点を対応する親・娘の組へ渡す。逆向き走査の未来候補に置き換えない。追加attentionは両方向のpair MLP内部で働き、logitの転置、双方向harmonic、固定secondaryのlow-margin consensus、source軸softmax、後処理はexp043と同じにする。
- source32件、target32件の組で分割してattentionを計算し、候補8点のsoftmax分母は各組・head内で保つ。GPUメモリが足りない場合は等価なchunk縮小またはgradient checkpointingを使い、候補数・構造・教師を縮小しない。

### 学習と対照

- exp044と同じ20動画・1823の教師付きpair窓を同じ外側胚分割で使う。各foldは片方の胚の8動画を勾配更新、同胚の2動画を内部選択、反対の胚の10動画を外側評価に用いる。疎いGEFF既知edgeを7 µm以内の一対一対応で教師化し、正例行または正例列のみのsource軸softmax focal BCE（gamma 2）を使う。未注釈の接続を確定負例とは呼ばない。
- 主条件は公開primary checkpointからprimary全パラメータと追加attentionを共同学習する。対照は同じcheckpointからprimaryだけを独立に再学習する。画像encoder、検出器、座標補正head、secondaryの重み・正規化統計はどちらも固定する。追加条件を省いたexp044 checkpointを対照に流用しない。
- 両学習条件は同じ動画分割、pair順、seed 20260924、AdamW、学習率1e-4、weight decay 1e-4、batch size 1、FP32、gradient clip norm 1、3 epochとする。各fold・各条件の内部検証でactive-pair error rateが最小のepochを選び、同率なら既知edge回収が多い方、さらに同率なら後のepochを選ぶ。外側胚をcheckpoint、構造、閾値の選択に使わない。新規checkpointは共同学習2個とprimaryのみ2個、公開primaryは保存済み固定対照。
- 学習後の共同学習重みで過去を全maskした推論も行う。この条件と通常入力の差は、同じ再学習primary上での履歴入力の作用を示す診断であり、独立に学習したprimaryのみ対照の代わりにはしない。

### 評価、進行条件、費用

- 公開primary、primaryのみ再学習、共同学習、共同学習の過去全maskを同一pairで評価する。確率はsource軸softmax、正例判定は厳密に0.48超とする。両胚の既知edge回収数と分母、active-pair errorsと分母、既知分裂親の全娘回収数と分母、教師負例予測数、正解親順位・得点差を記録する。閾値別recall、密集度別結果、attentionが選んだ過去候補と追加寄与は解釈用診断に限定する。
- 全graph検討の条件は両胚それぞれで、共同学習の既知edge回収数が公開primaryとprimaryのみ再学習の両方を厳密に上回り、active-pair errorsが両方以下、既知分裂親の全娘回収数が両方以上であること。分裂分母0は非悪化の証拠とせず進行条件を判定不能とする。外側結果を見て閾値・epoch・候補数を選び直さない。
- Kaggle Notebookのみ、課金なし、週45 GPU時間以内、1回12時間以内。exp044のcaptureと学習の実測時間を参考にしつつ、既存captureの入力可用性とSHAを実験化時に確認する。使用できなければ同じexp043推論から再captureする費用を含める。2動画pilotと最大点数のpairでfeature整列・forward/backward・peak GPU memoryを測り、対照を含む2条件×2fold×3 epoch、評価、必要な再captureへ保守係数1.5を掛けて残quotaと12時間内に収まらなければ全量を開始しない。Notebook経過時間とGPU割当消費を別に記録する。
- 単体条件を満たしても、全graphの候補選択・ILP・後処理との相互作用、公式scoreは未測定である。全graphと公式評価はpair結果・費用を提示した後にユーザーが判断する。competition submissionは別途明示承認を要する。

### 実験化時の受け入れ検査

- 現在・直前の元窓、候補ID、特徴32+32次元、物理座標、初回・欠測mask、K=8の同距離順と既知親保持率を検査する。
- 初期forward・reverse・融合logit parity、過去全maskでの厳密な追加寄与0、順序を変えた候補の出力対応、分割計算の出力・勾配一致、正逆での実時間履歴の一致を検査する。
- optimizerのprimaryと追加attention、両学習条件の同じsplit・教師・更新回数、内部checkpoint選択、胚別の分母と停止条件、推論時の逐次特徴保持を検査する。checkpoint、capture、feature schema・内容、Notebook sourceのSHAとKaggle実行証拠を保存する。
