# exp036_detection_score_pair_features 要件と実装方法

この文書を、実装前の契約、実装方法、受け入れ条件の正とする。実行中の進捗は`SESSION_NOTES.md`へ記録し、この文書へ重複させない。

## 実験化の入口・引き継ぎ・承認

- 実験化の入口と承認: backlog候補`detection_score_features`は、入力位置、得点表現、全graphへ進む条件が未決だった。2026-09-22に選択肢を提示し、ユーザーの「案3で進めてください」により、candidate pair特徴から接続logitへの学習可能な残差を使う方針と実験化・実装が承認された。
- 移行元backlog: `backlog/detection_score_features.md`。
- 対応する上位仮説: `HYP-20260920-03`。
- 上位仮説のうちこの実験が検証する範囲: 固定候補に付随する検出得点をcandidate pairごとの追加情報として使い、固定画像特徴・位置だけの同構造対照より隣接接続と分裂回収が改善するかを検証する。
- この実験だけで上位仮説を判断できるか: いいえ。
- 上位仮説の判断に残る検証: DoG応答とHOG特徴の個別効果、それらとの冗長性・相補性、隣接2-frame診断を通過した場合の固定全graph推論と公式評価。
- 親実験: [`exp016_frozen_image_encoder`](../exp016_frozen_image_encoder/)を学習・評価基準とし、入力cacheは[`exp015_oracle_stage_limits`](../exp015_oracle_stage_limits/)を固定利用する。
- 根拠 / 一次資料 / 参照実装: 親の`requirements.md`、`config.yaml`、`frozen_tracker.py`、`graph_inference.py`、exp015の検出得点保存処理、両実験の`metrics.json`。
- 固定するもの: 公開検出器、画像encoderと正規化、検出候補と座標、primary画像特徴、position特徴、教師対応、lossとmask、胚split、epoch、optimizer、secondary tracker、整数線形計画、graph repair、公式評価条件。
- 変更するもの: exp015 cacheの候補検出得点を学習胚だけでlogit標準化し、source、target、小さい方、絶対差の4 pair特徴を小さな学習層へ入力してprimary接続logitへ残差を加える。対照は同じ層へ標準化後の0だけを入力する。
- 最小の反証可能な検証: 2 variantと外側2foldの計4 trackerを同じseedとwindow順で学習し、両foldそれぞれで追加情報ありの`positive_edge_recall`、`edge_accuracy`、`division_parent_recall`が同構造対照以上か確認する。いずれかが低下した場合は全graph推論へ進まない。
- 成功条件: 初段では4 modelとfold別正規化統計、得点帯別診断、入力・model SHAを予算内に生成し、両外側胚で3指標の進行条件を満たすこと。全体の精度仮説は、後続の固定全graph評価で追加情報ありの公式combined scoreが同構造対照より両胚で改善し、division成分が悪化しない場合だけ支持する。
- 停止条件: cache SHA・schema・候補数対応が一致しない、得点が欠損または非有限、正規化へ内部validationまたは外側胚が混入する、同一seedの初期出力がvariant間で一致しない、12時間gate超過、OOM、NaN、または隣接診断の進行条件未達なら停止する。
- 実行しないこと: 低得点候補の削除、検出閾値変更、得点を真の存在確率とみなすこと、固定係数だけのlogit補正、画像側やsecondary trackerの更新、教師・loss・mask・decode変更、他特徴の同時追加、進行条件前の全graph推論、Kaggle submission。
- 未決事項: なし。
- backlog記録から解釈を変更した箇所とユーザー承認: 2026-09-22の「案3で進めてください」により、node入力連結ではなくpair特徴を選択した。各点だけでなく小さい方と絶対差を含め、clipped logitをfold内gradient-update候補だけで標準化し、0初期化した残差headを使う。全graphへの進行条件は両foldで3指標が対照以上と確定した。

## 判断履歴

- 2026-09-20: tracker特徴量案としてバックログ登録。入力表現と進行条件が未決のため`検討メモ・設計不可`とした。
- 2026-09-22: node連結、node埋め込み、pair特徴、learned gate、後段補正を比較した。
- 2026-09-22: ユーザーが「案3で進めてください」と判断し、candidate pair特徴から学習可能なlogit残差を作る実装を承認した。

## 手法契約

- 依頼原文: 「detection_score_featuresを実装してください」「案3で進めてください」。
- 期待する成果: 検出得点と既存画像特徴の重複を許容しつつ、候補ペアの端点信頼度と前後差を接続判定へ直接渡した追加効果を、同構造対照と分離して判断する。
- input: exp015の隣接2-frame cacheにある固定primary 32-channel特徴、position 32-channel特徴、候補座標・mask・ID、`detection_scores_src/tgt`。得点はclipped logitへ変換し、foldのgradient-update候補だけから計算した平均と標準偏差で変換する。
- target / objective: 主催者GEFFの既知中心・接続・分裂へ5 micrometer以内のgreedy one-to-one対応を行い、隣接候補間の通常接続と母娘接続を予測する。
- output: 既存`SimpleNodeTransformer`のprimary接続logitへ4 pair特徴の学習可能な残差を加えた接続logit、variant・fold別checkpoint、正規化統計、外側胚診断。
- loss: exp016と同じsource軸softmax、focal係数2のbinary cross entropy、正例source行またはtarget列に触れるpair mask。検出lossと追加のdivision重みは導入しない。
- decode: 初段ではgraphへdecodeしない。固定候補上の隣接接続を評価し、進行条件を満たした場合だけ同じ実験内でexp016のsecondary寄与、ILP、repair、公式評価へ接続する。
- context unit: 同じ動画の隣接2-frame windowと、そのsource候補・target候補の直積。
- 実装区分: `staged-faithful`。承認されたpair特徴と学習可能な残差は省略せず実装し、固定全graph評価だけを事前条件の後段へ分ける。
- 省略する機構と理由: 初段は画像forwardと全graph decodeを省略し、exp015固定cacheと隣接診断を使う。画像側は比較で固定し、全graphは進行条件未達時の不要な計算を避ける。
- proxyで検証できない主張: N/A。proxyではないが、初段だけでは公式combined score、Public LB、hidden test一般化を判断できない。
- proxyの場合のユーザー承認: N/A。
- この実験が支持 / 棄却できる主張: 指定した検出得点表現とpair残差headが、同じ固定候補・教師・loss・splitで隣接接続と分裂回収を改善するか。
- この実験では判断できない主張: 検出得点の校正精度、検出器更新の効果、他の得点表現全体、DoG・HOGとの組合せ、独立CV、Public LB改善。

## 実装方法

- アプローチ: exp016のtrain Notebookを基に、foldごとのscore normalizerを先にfitし、base trackerと4→16→1のpair headを持つwrapperを作る。head最終層を0初期化し、両variantの初期logitを一致させる。
- inputの実装箇所と変換: `frozen_tracker.py`がcache得点のshape・範囲・有限性を検査し、clipped logit変換とstreaming平均・標準偏差を実装する。`detection_score_tracker.py`がpair特徴と残差headを実装し、`exp036_detection_score_pair_features_train.py`がgradient-update pathだけでfold統計をfitして全splitへ適用する。
- target / objectiveの構築箇所: 親と同じ`greedy_match_candidates`、`build_legacy_edge_target`、`legacy_active_pair_mask`。
- outputの生成箇所と表現: `models/<variant>/fold_<n>/primary_tracker_best.pth`、variant・fold summary、`detection_score_normalizers.json`、`model_manifest.json`、`training_summary.json`、`metrics.json`。
- lossの実装箇所: 親と同じ`legacy_focal_bce`。wrapperが返す補正後logitだけを入力する。
- decode / postprocessの実装箇所: 初段では未実装。進行条件通過後に同じ実験へ追加する。
- context unitを保つ処理箇所: cache path、metadataのsample・連続frame、候補数、得点配列長を検査し、sourceとtargetの直積だけを作る。
- 変更するファイル / component: exp036の`config.yaml`、train Jupytext sourceとNotebook、`frozen_tracker.py`、`detection_score_tracker.py`、実験固有tests、正本記録。親や共通codeは変更しない。
- 固定事項を保つ確認方法: configとtestで親cache・source SHA、64次元base入力、教師、loss、fold、epoch、seed、decode未実装、submission禁止を固定する。公開tracker stateをbaseへstrict loadし、pair headの0初期出力とvariant初期一致を検査する。
- 参照sourceとの一致を確認するテスト: exp016由来のcache schema、教師対応、division target、loss、scoreなしbase forwardを維持する。
- 承認済み差分を確認するテスト: 4 pair特徴の順序、train-only正規化、0固定対照、0初期化head、4 model出力、fold別進行条件を検査する。

## 探索幅とpivot判定

- 変更class: `add-only`。既存target、output、loss、decode、context unitを固定し、接続logitへの学習可能な追加情報だけを加える。
- 同じ親 / familyで連続した小改善実験数: exp016後の承認済みfeature追加として1件目。
- positiveなoracle headroom / coverage / 誤差非相関性: exp015のcandidate edge recall 94.829%、final selected edge recall 91.602%、candidate division triplet recall 40.397%から接続判定余地があるが、検出得点との誤差非相関性は未測定。
- 比較したtarget、output、decode、context unitを変える案: node入力連結、node埋め込み、learned gate、固定後段補正を比較し、ユーザーがpair特徴を選択した。division set出力などはこの実験へ混ぜない。
- 小改善の継続またはpivotを選ぶ根拠: 得点は既存cacheにあり、画像再計算なしで最小比較できる。進行条件未達なら同一外側評価を見た表現探索を行わず停止する。
- `kaggle-idea-forge` の実行要否と根拠: 不要。対象候補の初回実装で、同familyの連続小改善3件目ではない。

## 再現性・リスク

- seed policy: global seed 42とfold offsetを使い、両variantでfoldごとに同じmodel初期化、DataLoader generator、worker seedを再設定する。
- stochastic 処理の有無: DataLoader shuffle、dropout、CUDA kernelがある。bitwise一致は主張しない。
- stochastic feature generation / augmentation / seed bagging の有無: 得点変換はdeterministic、augmentationなし、seed baggingなし。
- 並列処理と乱数の関係: DataLoader workerはPyTorch seedからNumPy seedを設定し、score変換はglobal RNGを使わない。
- CPU/GPU runtime と deterministic flags: Kaggle T4、full precision、12時間gate。64 window/fold benchmarkを行い、親の予測対実測比を2倍のreserve付きで補正する。
- train cache / test feature regeneration のSHA記録方針: exp015 summary・identity SHA、score schema、fold別normalizer内容SHAを記録する。
- model manifest / prediction / submission SHA記録方針: 4 modelのfile・canonical state SHAとmanifest SHAを保存する。初段はprediction、submissionを作らない。
- Kaggle package bootstrap確認方針: 正のsourceからtrain packageを再生成し、config、metrics、helper、Notebook一致をpush前に検査する。
- リークリスク: fold normalizerへinternal validationまたは外側胚のscoreを入れない。外側結果を見た変換や閾値の救済探索を行わない。
- CV/LB不一致リスク: 条件付き外側評価であり独立CVではない。公式scoreとPublic LBは初段で主張しない。
- ランタイム/メモリリスク: 4 tracker再学習とdense pair特徴が追加費用。pair特徴はwindow単位で生成し、cacheへ全pairを保存しない。
- 再現性リスク: GPU学習の非決定性によりmodel SHAは一致しない可能性がある。seed、入力、環境、数値差を記録する。
- 手法忠実性リスク: pair headをTransformer block改善と呼ばず、接続判定headへの追加情報と説明する。
- 過度な縮小 / proxy化リスク: node得点だけへ縮小せず、承認されたsource、target、小さい方、絶対差を実装する。

## 受け入れ基準

- [ ] 手法契約のinput、target、output、loss、decode、context unitがコードと一致する。
- [ ] 4 pair特徴、fold内正規化、0固定対照、0初期化残差headがtestで確認できる。
- [ ] 2 variant×2foldの4 modelと必要なSHAをKaggle実行で保存できる。
- [ ] 得点帯別の既知接続回収、判定可能な誤接続、分裂回収、有効件数が両foldで記録される。
- [ ] 進行条件を機械判定し、未達なら全graph推論を開始しない。
- [ ] `config.yaml`の系譜がこの文書と一致する。
- [ ] `make validate-exp`、`make check-exp`、`make test-exp`が通る。
