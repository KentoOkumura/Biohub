# frame_self_attention_spatial

- 候補名: `frame_self_attention_spatial`
- 状態: `検討メモ・設計不可`
- 対応する上位仮説: `HYP-20260920-02`
- 関連する上位仮説: HYP-20260910-04
- 作成日: 2026-09-20
- 最終更新日: 2026-09-21
- 依頼原文: 「バックログに追加してください」。直前にPR曲線・誤り分析を先行させ、その結果を受けて近傍制限・距離biasの追加を検討する順序を説明した。
- 期待する成果: 同一フレームのSelf-Attentionが参照するcellまたは参照の重みに位置関係を反映し、固定特徴下で親候補の取り違えを減らせるか調べる。
- 親実験 / 比較対象: [exp025](../experiments/exp025_frame_self_attention/)の`model_b_identity_init`を構造上の主対照とし、[exp016](../experiments/exp016_frozen_image_encoder/)の現行tracker、exp025 Model A/Bも保存済み比較対象として残す。
- 優先度: P4
- 優先度の理由: exp033ではSelf-Attention構成のprecision 0.95時recallが両胚で一貫して現行を上回らず、近距離・高候補数・長距離移動の各条件でも改善とfalse-positive低下が両立しなかった。距離biasまたは近傍制限を選ぶ直接証拠がないため、方式固有の根拠を得るか、高リスクな構造探索として再開するとユーザーが判断するまで保留する。
- `backlog/KAGGLE_DIRECTION.md` の対応箇所: [検証中の仮説](KAGGLE_DIRECTION.md#検証中の仮説)と[未着手バックログ](KAGGLE_DIRECTION.md#未着手バックログ)

## 観測事実と根拠

- 実測済みの事実: exp025は画像32次元＋位置32次元を共有Encoderへ渡すが、Self-Attentionに同一フレーム内の距離bias・近傍制限はない。恒等初期化で初期logitsを保持しても、学習後の既知edge recallは両胚で現行未満。通常Bに対してprecisionは上がりrecallは下がった。評価window内の1-frame当たり平均候補数は6bba約193、44b6約336であり、真のcell密度とは区別する。
- exp033の観測: precision 0.95時recallは6bbaで現行0.9762、A 0.9218、B 0.9683、恒等初期化B 0.9730、44b6で現行0.9102、A 0.7171、B 0.8987、恒等初期化B 0.9137。両胚で一貫した改善はない。子候補の最近傍距離が最小のbucketではBの0.5 recallが現行比で6bba +0.33、44b6 +1.58 percentage pointだったが、false-positive pairも6bba 2,183→2,426、44b6 903→1,007へ増加した。最大移動距離bucketや最大候補数bucketの差も胚間で一貫しなかった。
- 根拠ファイル / 一次資料: [exp025結果](../experiments/exp025_frame_self_attention/result.md)、[metrics](../experiments/exp025_frame_self_attention/metrics.json)の`pair_level_comparison`と`post_identity_diagnostics`、[契約](../experiments/exp025_frame_self_attention/requirements.md)、[config](../experiments/exp025_frame_self_attention/config.yaml)、[model source](../experiments/exp025_frame_self_attention/simple_node_transformer.py)、[当初の設計](../docs/surveys/biohub-node-self-attention-design_20260920.md)。
- 利用する保存済み生成物とSHA: 主対照のidentity train kernel v1はexp025 metricsの該当runを参照し、manifest SHAは`442c25536a64419d2460755a13513edd0f4aa76b39df29d954af5a92359faa4c`。他対照とcache・公開初期trackerのSHAもexp025/exp016のmetrics/configを正とする。exp033のpair shard manifest SHAは`2e34f8fd0b7408bcd0ee1a14cbdc335ebe56327f0b600a02bf64038f5226073d`、診断summary SHAは`c5c93911407cbbe75a4570a02736b7a7a373e42b16a659d3d60ff1fef4fd9ebd`。
- 仮定: Assumption: フレーム内の位置関係をAttentionへ明示すると、集合全体への依存を抑え、有用な近傍情報を使う学習が進む可能性がある。近傍だけで対応が決まる、遠方cellの混合で特徴が均一化した、距離biasが必ず効くとは仮定しない。
- 既存候補との境界: [relative_neighbors](relative_neighbors.md)は時刻間で周囲の相対配置の変化を接続特徴へ使う候補。本候補はexp025のフレーム内Self-Attentionに作用する。[sparse_motion_graph](sparse_motion_graph.md)の時刻間候補制限・多時点化も含めない。

## この候補が直接検証する仮説と範囲

- 上位仮説のうちこの候補が検証する範囲: 同一フレームのSelf-Attentionに空間的な関係を反映することで、制限なしのSelf-Attentionより対応の識別を改善できるか。
- この候補の具体的な仮説: 距離biasまたは近傍制限のどちらかを単独で加えると、同じ初期化・教師・学習量で、主対照より既知edgeの回収と誤接続の釣り合いが改善する。
- 仮説が正しい場合に期待する観測: 先行診断で固定したprecision比較・条件別誤りが両胚で改善し、recall改善と引き換えに教師上false-positiveや分裂を悪化させない傾向がある。公式graphへの移行は別に検証する。
- 仮説を棄却する観測: 空間情報を加えても同条件で改善がなく、片胚だけへの適応や分裂の低下が残る。追加費用に対し改善が再現しなければこの設定を支持しない。
- この候補だけで上位仮説を判断できるか: いいえ
- 上位仮説の判断に残る検証: 選ばなかった空間方式、他のcontext unit、教師・既存層更新の寄与、公式graph・hidden test。個別設定の結果からSelf-Attention全体を支持・棄却しない。

## 入力・予測対象・出力・推論方法

- input: exp015固定特徴・位置特徴・候補座標・cell mask。両時刻のcell集合を同じEncoderへ別々に渡す。
- target / objective: exp025と同じGEFF由来の隣接2-frame対応、分裂を含む既知edge。
- output: 従来と同じcell pairのedge logits。shapeとloss interfaceを維持する。
- loss: exp025のsource軸softmax、既存mask、focal weighted BCEを維持する。損失の変更と同時に比較しない。
- decode / 推論方法: Self-Attention → 現行どおり逐次更新する双方向Cross-Attention → Pair MLP。変更対象はフレーム内Attentionのみ。既存の時刻間相対座標・pair採点・復号を維持する。
- 処理単位: 隣接2-frame window、各フレーム内cell集合。長い時間列・cell順の位置埋め込みへ読み替えない。
- 実装区分: 方式確定後、実際に距離biasまたは参照cell制限をAttentionへ実装する範囲を、[用語集](../docs/glossary.md)のfaithful / staged-faithfulとして契約に記録する。NFL全体の忠実移植とは扱わず、pair特徴追加だけのproxyへ置き換えない。

## 親実験からの差分

- 変更するもの: Self-Attentionの重み計算への距離・相対位置のbias、または参照cell集合の近傍制限。両者のどちらを最初に比較するかは未決。
- 固定するもの: 公開検出器・画像特徴、候補、教師、loss、共有Encoder、Cross-Attentionの逐次更新、Pair MLP、既存座標の扱い、fold、epoch・optimizer・選択基準。公開初期trackerから開始し、追加残差の恒等初期化を維持する。既存層の固定や学習率変更を同時に混ぜない。
- 再利用するコード / config / 生成物: exp025 model・train・inference、maskと初期logits一致テスト、exp015 cache、先行診断の評価定義と保存対照。
- 新しく作るもの: 実験化後のフレーム内位置関係の計算、Attentionへの適用、方式を表すconfig、対応する学習・診断Notebook。現時点では作成しない。

## 最小の反証可能な検証

- 検証方法: exp033のprecision 0.95比較、0.5指標、条件別件数を固定対照として使う。方式固有の根拠を得て下記の未決事項を確定した場合だけ実装し、初期logits同値性・padding除外・空フレーム・座標整合を確認して、同じ胚分割・教師・pair評価で主対照と比較する。
- variant / config / fold / booster数: 初回は方式1つ・config 1つ・2fold×3epochの2 tracker学習を候補とする。2方式の同時学習やパラメータ探索は今回の範囲に含めない。booster 0。方式確定時に総実行数・費用を契約へ明記する。
- control再学習: なしを基本とし、保存済みidentity Bと現行を再評価する。条件差で新しい対照が必要になった場合は、理由・追加費用を示して別途承認を得る。
- 想定runtime / resource: Kaggle Notebookのみ、週30 GPU時間・1 Notebook12時間内。identity Bの実測時間を出発点とするが、新しい距離行列・maskによるAttention実装と最大cell数の時間・メモリを事前benchmarkする。

## 成功条件と停止条件

- primary指標: exp033と同じ同点score一括処理によるprecision 0.95時recallと既知edge recall。教師上のfalse-positive・分裂親回収を両胚別に併記する。
- 成功条件: 同じ評価条件で主対照より両胚の対応付けが改善し、誤接続・分裂の許容条件を満たす。定量的な進行条件は方式確定時の未決事項に含める。
- 必須guard: 候補・特徴・教師・foldの一致、padding cellがAttentionに参加しないこと、初期logits一致、時刻間座標やsource軸softmaxを変えないこと、両胚の有効件数と部分注釈の限界、runtime gate。
- 成功時の次段階: pair診断と費用を示し、ユーザー判断後に公式graph評価を検討する。単体指標を公式scoreと呼ばない。
- 失敗時の停止範囲: 同条件で改善がなければ全graph推論へ自動で進めない。半径・bias強度・epochを外側胚で救済探索しない。上位仮説や兄弟候補を自動で閉じない。

## 実行しないこと

- 禁止する代替実装、proxy、同一OOF上の救済探索: フレーム内Attentionへの変更を単なるPair MLPへの距離特徴追加に置き換えること、時刻間候補の削減、無根拠な固定cell数cutoff、cellをchunkごとに独立Encodeすること、未知pairを確定負例にすること、外側胚で条件選択すること。
- 壁打ちで採らなかった案と理由: 直ちに近傍制限を実装する案は先行診断の後へ移した。PRの変化が確率の出方か順位の悪化か未分離であり、制限で有用な広域情報も失う可能性がある。既存層の固定、loss変更、epoch延長は別の原因を変えるため同時実施しない。

## リスク

- leakage / validation: 公開学習来歴を含む条件付き比較。biasや半径・k・評価閾値を外側胚の正解から選ばない。
- hidden test: cell数・間隔の分布変化、境界・分裂でも処理可能であることが必要。未知の正解やtrain固有IDに依存しない。
- runtime / memory: 距離行列やAttention maskはcell数の二乗で増えうる。単にmaskを作っても高速化するとは限らない。
- 再現性: 距離の単位と座標軸、bias/近傍定義、self参照・空集合の扱い、seed、source/checkpoint/cache SHAを明記する。

## 先行条件 / 依存

- [exp033診断](../experiments/exp033_frame_self_attention_diagnostics/)は完走し、PR曲線・候補数/混雑度/距離/分裂別の誤りと保存対照の再現を取得済み。結果は方式を一意に選ぶ根拠にならなかった。
- exp025は採否・完了未判断であり、その判断を本候補の追加から推測しない。
- 追加学習を確定する前に、`kaggle-strategy`の規則に従い、同機構の連続変更となるかを確認し、該当する場合は`kaggle-idea-forge`でtarget・output・decode・context unitを変える候補も再検討する。今回のバックログ化を次実験確定とは扱わない。

## 調査・実行時に確認する事項

- exp033の実際の誤り分布と生成物SHAは取得済み。未取得なのは、選択した方式のAttention対象数・時間・メモリ、学習後の改善幅・分裂件数。未実測値を未決の方式の代わりにしない。

## 未決事項

- 最初に距離biasと近傍制限のどちらを試すか。前者は距離のみか相対位置ベクトルか、固定/学習可能な関数と初期値、後者は半径/k近傍とself参照・近傍なしの扱いを確定する。
- 空間関係に使う座標・物理単位と正規化。既存のPair MLPへ渡す座標の意味を変更せず、追加計算に必要な変換だけを契約化する。
- 先行診断を踏まえ、共通precisionの比較規則と、recall・教師上のfalse-positive・分裂に対するgraph評価への定量的進行条件を実装前に確定する。

## 判断履歴

- 2026-09-20: ユーザーが近傍制限・距離biasと診断の優先順位を確認。診断を先行、モデル改良の候補として距離biasを上位とする方針を提案した。
- 2026-09-20: ユーザーのバックログ追加依頼により、既存のSelf-Attention上位仮説のP2候補として記録。方式は未確定、実験化・実装・Kaggle実行は未承認。
- 2026-09-21: exp033の診断を反映。precision 0.95時recallはModel A/Bが両胚で現行未満、恒等初期化Bは6bbaで現行未満・44b6で+0.35 percentage pointだった。近距離条件ではModel Bのrecallが両胚で上がったがfalse-positiveも増え、距離biasと近傍制限の選択根拠にはならなかった。候補をP4へ変更し、方式固有の根拠またはユーザーの高リスク探索判断を再開条件とした。

## 次セッションへの引き継ぎ確認

- 固定するものを一意に説明できる: はい。exp025の初期化・入力・教師・loss・逐次Cross-Attention・Pair MLPを引き継ぐ。
- 変更するものを一意に説明できる: フレーム内Attentionの参照または重み。具体的な方式は未決。
- 最小検証と停止条件を一意に説明できる: 同条件のpair比較と停止範囲は明確。方式と定量的進行条件は上記を確定する。
- 実行しないことを一意に説明できる: はい。複数方式・loss・学習量等を同時に変更しない。
- 未決事項が明示されている: はい。
