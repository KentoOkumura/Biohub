# exp024_tracker_six_epochs 要件と実装方法

- 候補名: `tracker_six_epochs`
- 移行時の候補状態: `設計可能・実験化未承認`
- 対応する上位仮説: `HYP-20260920-01`
- 関連する上位仮説: `HYP-20260910-12`（固定画像特徴cacheの再利用）
- 作成日: 2026-09-20
- 最終更新日: 2026-09-20
- 依頼原文: 「trackerは3エポックですが、これを増やすと精度がよくなる可能性はありますか？」「6エポックに伸ばすアイデアをバックログに追加し最優先としてください。」
- 期待する成果: exp016と同じ固定画像特徴・教師・推論条件でprimary trackerだけを6エポック学習し、3エポック基準から両胚の公式graph指標が改善するか確かめる。
- 親実験 / 比較対象: [exp016の実験契約](../exp016_frozen_image_encoder/requirements.md)、[設定](../exp016_frozen_image_encoder/config.yaml)、[結果](../exp016_frozen_image_encoder/result.md)、[数値](../exp016_frozen_image_encoder/metrics.json)。保存済みの3エポック設定から選んだfold別trackerを主対照とする。
- 優先度: P0
- 優先度の理由: ユーザーが最優先を指定した。対照と固定feature cacheが保存済みで、エポック数だけを変える直接比較が可能。ただし内部検証では両foldとも2エポック目が選ばれており、改善確率が高いと実証されたわけではない。
- `backlog/KAGGLE_DIRECTION.md` の対応箇所: [検証中の仮説](../../backlog/KAGGLE_DIRECTION.md#検証中の仮説)と[未着手バックログ](../../backlog/KAGGLE_DIRECTION.md#未着手バックログ)
- 先行条件 / 依存: exp016の保存済みcache・公開初期tracker・fold別対照と公式評価手順。実行前にKaggle GPU残量を確認する。

## 実験化の入口・引き継ぎ・承認

- 実験化の入口: `backlog/tracker_six_epochs.md`。2026-09-20のユーザー依頼「実装に進んでください」を実験化承認として扱う。
- 実験の状態: `metrics.json`のstatusを正とし、採否・完了は実行証拠を整理してユーザー判断で更新する。
- backlog記録からの重要な解釈変更: なし。exp016の3エポックと同じ公開初期重みから最大6エポックを連続学習する。
- 変更class: このリポジトリ内の管理用語で`parameter`。エポック上限だけを変える。
- 同じ親の小変更: exp016は固定画像特徴から学ぶ基準、exp019はloss mask変更で不採用。今回は学習長だけの比較であり、連続3件目の小変更には該当しない。出力・教師・復号を変えるdivision_triplets等は別候補として保持する。
- 採否・完了: 学習と公式評価の結果を整理してユーザーに判断を求める。submissionは別途明示依頼がある場合だけ実行する。

## 観測事実と根拠

- 実測済みの事実: exp016は公開primary tracker checkpointから3エポック追加学習し、両foldで内部検証の選択指標が最大だったのはepoch index 1（2エポック目）。3エポック目も学習損失は下がったが、内部検証指標は下がった。3エポックでの公式combined scoreは全199動画0.9120545013、44b6で0.9097659161、6bbaで0.9125310093。公開tracker対照との差は全体で微増したが、44b6では悪化した。6エポックの効果は未測定。
- 根拠ファイル / 一次資料: [exp016の学習・公式評価数値](../exp016_frozen_image_encoder/metrics.json)の`train_stage`と`official_graph_evaluation.retrained`、[解釈](../exp016_frozen_image_encoder/result.md)。3エポック、AdamW、学習率、seed、2方向の胚分割は[config](../exp016_frozen_image_encoder/config.yaml)を正とする。
- 利用する保存済み生成物とSHA: exp015のtrain feature cacheは19,701 window、summary SHA256 `040d1f6437e27149e0e34ad4aac8bba3c8cf63dc266d33df497acd969972194c`、identity SHA256 `440891c4550adf8540e3c47f9784e68b6ca1b64bbe56dbb93a25eb87c46bc1ee`。公開初期checkpoint SHA256 `12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771`。exp016 fold別trackerと評価出力のSHA・所在は同実験の`metrics.json`を正とし、実験化時に照合する。
- 仮定: Assumption: 同じ固定特徴・教師・損失でも追加の更新によって接続・分裂の予測が変わり、内部検証で選んだ後半の重みが両胚の公式graph指標を改善し得る。既存の3エポック記録はこの仮定を支持していない。

## この候補が直接検証する仮説と範囲

- 上位仮説のうちこの候補が検証する範囲: 固定画像特徴からprimary trackerを学ぶ既存条件で、学習を3から6エポックへ延長する効果だけを検証する。
- この候補の具体的な仮説: 同じ公開初期値、教師、loss、fold、optimizer、学習率で最大6エポック学習すると、追加の3エポックで選ばれたtrackerがexp016の両胚の公式combined scoreを上回る。
- 仮説が正しい場合に期待する観測: 両foldで4～6エポック目の重みが内部検証で選ばれ、固定候補・固定復号の公式評価で両胚と全体のcombined scoreがexp016を上回る。
- 仮説を棄却する観測: いずれかのfoldで最良重みが1～3エポック目に残る、または4～6エポック目の重みを選んでも公式combined scoreが両胚で改善しない。
- この候補だけで上位仮説を判断できるか: いいえ
- 上位仮説の判断に残る検証: 6エポック以外の長さ、学習率、教師、損失を変えた場合の効果と、公開初期モデルにtrain動画が含まれる条件付き評価から独立した評価への移行。これらを本候補には混ぜない。

## 手法契約

- input: exp015の固定画像特徴cacheにある候補点の画像特徴、位置、検出得点と、学習側GEFFの既知中心・接続。
- target / objective: exp016と同じ、隣接時刻の注釈済み有向edgeと候補点の対応。
- output: exp016と同じprimary `SimpleNodeTransformer`の接続score。foldごとに内部検証で選んだ重みを保存する。
- loss: exp016と同じsource軸softmax後のfocal weighted binary cross entropy、gamma 2、正例行または正例列に触れるpairのmask。検出lossは無効。
- decode / 推論方法: exp016と同じ候補、固定secondary tracker、ILP、graph repair、公式評価器を使う。
- 処理単位: 2時点のwindowで学習し、動画単位でgraphを作り、胚を入れ替える2方向で評価する。
- 実装区分: このリポジトリ内の管理用語では、exp016と同じ固定特徴による段階的な学習を保つ`staged-faithful`。変更はエポック上限だけであり、公開モデル全体を再学習する案ではない。

## 親実験からの差分

- 変更するもの: `training.epochs`を3から6へ変更し、各エポックの内部検証指標と選択された重みを記録する。
- 固定するもの: 初期checkpoint、cache、empty-GT window除外、2方向の胚分割と内部90%/10%分割、seed、教師、loss、AdamW・学習率0.0001・weight decay 0.01・batch size 2、primary trackerのみ学習、候補生成、secondary tracker、decode、評価器。
- 再利用するコード / config / 生成物: exp016のtrain・inferenceコード、config、固定cache、fold別3エポック対照、公式graph評価。比較用に元のexp016ファイルを書き換えない。
- 新しく作るもの: 実験化が承認された場合だけ、新しい実験のconfig、6エポック学習Notebook、fold別model、学習曲線、固定graph推論・公式評価の証拠。

## 最小の反証可能な検証

- 検証方法: 各foldを公開初期重みから同じseed・設定で最大6エポック再学習する。各エポックの重みと内部検証指標を記録し、exp016と同じ指標で1～6エポックから選ぶ。両foldで4～6エポック目が選ばれた場合、保存済みexp016の3エポック対照と同じ候補・復号で199動画の公式graph指標を比較する。外側評価胚の正解をcheckpoint選択に使わない。
- variant / config / fold / booster数: 学習variant 1、config 1、胚holdout 2fold、booster 0、出力model 2件。エポック上限以外の探索は行わない。
- control再学習: なし。保存済みexp016対照を再利用する。再実行した1～3エポック目の指標は対照との再現差を診断する。
- 想定runtime / resource: exp016の3エポック学習Notebook全体は4,620.86秒で、6エポックの所要時間は未実測。Kaggle Notebookのみ、GPU週30時間以内・課金なし、実行前の残量確認と短いbenchmarkで見積もる。trainと推論・公式評価を別計上し、12時間gateを超える見込みならfull実行前に止める。

## 成功条件と停止条件

- primary指標: exp016と同じ公式combined score（adjusted edge Jaccardとdivision Jaccardの加重和）。全体と44b6・6bba別に比較する。
- 成功条件: 両foldで4～6エポック目から内部検証で選ばれた重みを使い、exp016の保存済み3エポック対照に対して44b6と6bbaの両方で公式combined scoreが上がり、対象199動画・候補座標・固定復号の一致を確認できる。
- 必須guard: 全エポックの学習損失と内部検証指標、選択エポック、両胚の公式指標と接続・分裂成分、実行時間、失敗・skip件数を記録する。公開初期モデルにtrain動画が含まれるため独立CVと呼ばない。
- 成功時の次段階: 結果、計算費用、公開trackerとの差、両胚の変化をユーザーへ提示して実験の採否・完了判断を求める。Kaggle submissionは別の明示依頼まで行わない。
- 失敗時の停止範囲: いずれかのfoldで内部検証の1～3エポック目が選ばれた場合は追加の公式graph推論を止め、少なくとも一方のfoldでは延長で選択重みが変わらず、両胚改善条件を満たせないと記録する。4～6エポック目が選ばれても両胚の公式指標が改善しなければ、この6エポック一変更を推奨しない。上位仮説全体の採否はユーザー判断へ残す。

## 実行しないこと

- 禁止する代替実装、proxy、同一OOF上の救済探索: 外側評価胚の正解・公式scoreを使ってエポック、学習率、seed、fold、decodeを選ばない。公開検出器・画像encoder・secondary trackerを更新せず、教師maskや候補数を同時変更しない。単にexp016の保存済み重みからoptimizer状態なしで追加3エポックを始める処理を、公開初期重みからの連続6エポック学習と同一視しない。
- 壁打ちで採らなかった案と理由: 3エポック重みからの追加学習はoptimizer状態と乱数系列が異なり単純なエポック差にならないため採らない。学習率変更や6エポック目固定の選択は別条件を増やすため採らない。exp019の変更済みmaskは対照とlossが異なるため使わない。

## 探索幅とpivot判定

- 変更class: `parameter`。学習上限だけを変える。exp019はloss maskの一変更で不採用となっており、同じ機構での連続3件目の小変更には該当しない。
- exp015では既知edgeのcandidate graph回収94.829%とfinal graph選択91.602%、分裂ではcandidateに母と2娘が揃う61/151件に対してfinalに残る15/151件を観測した。候補と復号の限界が残るため、学習長だけの失敗でtracker改善全体を棄却しない。
- outputとdecodeを変える高upside案としてdivision_tripletsが別候補にある。今回のP0はユーザーの指定であり、その方式や教師変更を混ぜない。
- `kaggle-idea-forge`は今回使わない。承認済みの設計可能な一変更を実装する段階である。

## 再現性・リスク

- leakage / validation: 公開画像encoderと初期trackerはtrain動画を学習に含む可能性がある。胚holdoutはtracker変更の条件付き比較であり、独立CVではない。外側胚を重み選択や失敗後の設定救済に使わない。
- hidden test: exp016と同じ固定推論経路・入力だけを使う。hidden testの精度やPublic LBへの改善は、この条件付き公式train評価からは保証しない。
- runtime / memory: 学習時間は延びる。候補pairの最大サイズと12時間gate、週30時間のGPU予算を小規模実測で確認する。限界時にfoldやエポックを黙って減らさない。
- 再現性: cache・公開重み・source・分割・seed・モデル・graphのSHAを記録し、保存済みexp016の数値と再実行1～3エポック目の差も報告する。

## 調査・実行時に確認する事項

- Kaggle GPU残量、6エポックの実測時間とメモリ、各エポックの学習・内部検証指標、選択エポック、モデルとgraphのSHA、両胚の公式指標。未実測であることだけを理由に設計不可にはしない。

## 未決事項

- なし

## 判断履歴

- 2026-09-20: ユーザーが6エポックへ伸ばす案のバックログ追加と最優先化を依頼した。これはバックログ化の承認であり、実験化・実装・Kaggle実行・submissionの依頼ではない。
- 2026-09-20: 3エポックの内部検証最良が両foldで2エポック目だった事実を記録したうえで、エポック上限だけを変える一変更の比較として設計した。

## 次セッションへの引き継ぎ確認

- 固定するものを一意に説明できる: exp016の設定と保存済みcache・対照を上記で指定した。
- 変更するものを一意に説明できる: 最大エポック数を3から6へ変更する。
- 最小検証と停止条件を一意に説明できる: 内部選択後、後半重みが選ばれた場合だけ固定graphの公式指標を両胚で比較する。
- 実行しないことを一意に説明できる: 外側胚での設定選択、他部品の更新、再開学習による置換、提出を除外した。
- 未決事項が明示されている: なし。所要時間と効果は実行時に測定する。

## 実装方法

- inputの実装箇所: `exp024_tracker_six_epochs_train.py`がexp015のcache summary・identity SHAと全windowのschemaを検査し、`frozen_tracker.py`が2時点の候補点特徴を読み込む。画像encoderは学習・推論とも更新しない。
- target / objectiveの実装箇所: `frozen_tracker.py`のGEFF greedy one-to-one対応と`build_legacy_edge_target`をexp016から引き継ぐ。empty-GT windowの除外も維持する。
- outputの実装箇所: 公開checkpointで初期化したprimary `SimpleNodeTransformer`から接続logitを出し、各foldの内部検証で選んだ重み、選択エポック、model manifestとSHAを保存する。
- lossの実装箇所: `frozen_tracker.py`の既存source軸softmax、正例行または正例列のmaskとfocal weighted binary cross entropy。学習可能parameterがprimary trackerのみであることをruntimeでassertする。
- decode / postprocessの実装箇所: `graph_inference.py`と`exp024_tracker_six_epochs_inference.py`でexp015の固定候補・公開control・secondary tracker・ILP・graph repair・公式評価器を再利用する。推論実装はexp019でKaggle完走した固定cache replayを踏襲し、保存済み公開controlがexp016と一致することを検査する。
- 処理単位: 2時点windowを学習単位、動画をgraph生成単位、胚をholdoutと結果報告の単位とする。
- 参照実装との差: exp016の公開checkpoint・固定特徴・教師・loss・fold・optimizer・復号は維持し、学習エポック上限を3から6へ変更する。公開source全体のend-to-end再学習ではなく、exp016と同じ段階的な学習である。
## 受け入れ基準

- trainの受け入れ: 1 variant、1 config、2 fold、0 booster、対照再学習なし。各foldの6件のtrain・内部検証記録、選択エポック、外側胚の接続診断、model 2件、manifest、SHA、GPU時間を保存する。後半エポックが両foldで選ばれなければ公式graph推論は保留する。
- inferenceの受け入れ: 前段条件成立時にのみ、公開controlの候補graph全199件一致、固定候補座標一致、main画像encoder forward 0、199動画の最終graph、公式指標の再計算一致、exp016保存済み3エポック基準との両胚比較を確認する。
- 検証: `make validate-exp EXP=exp024_tracker_six_epochs`、`make check-exp EXP=exp024_tracker_six_epochs`、`make test-exp EXP=exp024_tracker_six_epochs`、Jupytext round-tripを通す。Kaggle push前にGPU quota、Active Sessions、予算とkernel sourceを確認する。
- 再現性: seed 42とfold offset、exp015 cache SHA、公開checkpoint SHA、学習・推論source SHA、model file/state SHA、Kaggle kernel versionを記録する。再実行のbitwise一致は前提にしない。
- 計算資源: Kaggle Notebookのみ、週30 GPU時間以内、課金なし。exp016の3エポックtrain実測3,379.48秒は費用見積もりの基準であり、6エポックの実測値ではない。Notebookの12時間gateを守る。


## 追加診断: graphを作らない2時点の接続比較

- 2026-09-20のユーザー依頼により、全動画のgraph指標を実行せずにprimary trackerの接続精度を比較する。ユーザーはまず比較用の診断を選択し、既存のcheckpoint選択規則は変えない。
- 比較対象はexp016で選択された2エポック目と同一state SHAのexp024 2エポック目、およびexp024 6エポック目。両foldの外側胚をそれぞれ同じ固定cache、5µmのGEFF対応、同じ2時点windowで評価し、対照を再学習しない。
- source軸softmax後の親候補確率で、既知の親が候補内に1件ある子細胞だけを採点する。親の1位正答率と、固定閾値0.48で採用した正しい接続・誤接続・見逃しから計算する接続Jaccard、既知子細胞あたりの誤接続数、既知分裂親の両娘への接続率を、胚別・checkpoint別に件数とともに示す。注釈のない子細胞を真の負例とみなさない。
- sourceの正解候補が存在しない注釈接続は採点対象から外れる。これはprimary tracker単体の診断であり、双方向・secondary融合、ILP、graph repair、公式評価器の効果、検出候補回収率を測れない。外側胚の結果をcheckpoint選択に使わず、exp024の当初のgraph続行条件も変更しない。
- `exp024_tracker_six_epochs_diagnostic.ipynb`をKaggleで実行し、cache・注釈・モデルのSHA、胚別と動画別の生件数、比率、所要時間を`tracker_pair_diagnostic.json`へ保存する。値を確認して`metrics.json`へ記録し、比較の解釈と公式score未計測を`result.md`へ追記する。分裂件数が少ないときは不確実性を明記する。
