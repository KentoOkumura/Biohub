# exp040_trackastra_association 要件と実装方法

- 候補名: `trackastra_association`
- 説明: Trackastraの6時点対応学習を、固定検出点・固定画像特徴と疎い接続教師に適用する
- 移行時の候補状態: `設計可能・実験化未承認`
- 対応する上位仮説: `HYP-20260910-10`
- 関連する上位仮説: `HYP-20260910-11`。親候補間の競合と対応なしの扱いに関係する。
- 作成日: 2026-09-22
- 最終更新日: 2026-09-23
- 依頼原文: 「Trackastraは今回のコンペで有用そうですか？」「ここまでの議論を踏まえてバックログ案を考えてください」「3案すべて追加してください」「trackastra_associationの設計を確定させてください」「実装に進んでください」
- 期待する成果: 数時点の検出集合から対応をまとめて学び、保存済みexp016より接続・分裂の取り違えを減らせるか検証する。改善は未実証。
- 親実験 / 比較対象: [exp016 requirements](../exp016_frozen_image_encoder/requirements.md)、[config](../exp016_frozen_image_encoder/config.yaml)、[result](../exp016_frozen_image_encoder/result.md)。固定入力はexp015 cache、比較対象は保存済みexp016の各foldのprimary tracker。
- 優先度: P2
- 優先度の理由: 時間窓・対応教師・正規化を変更する独立候補。座標attentionや履歴案の成功を着手条件にしない。既承認作業を止めず、他候補の順位を維持する。
- `backlog/KAGGLE_DIRECTION.md` の対応箇所: [検証中の仮説](../../backlog/KAGGLE_DIRECTION.md#検証中の仮説)、[未着手バックログ](../../backlog/KAGGLE_DIRECTION.md#未着手バックログ)
- 先行条件 / 依存: exp015の固定候補・画像特徴・時刻、主催者GEFF、exp016の分割と保存済み重み。外部領域分割、軌跡擬似教師、追加注釈を要しない。

この文書は、実装前の契約・方法・受け入れ条件の正とする。元の候補詳細を欠落なく移し、進捗は`SESSION_NOTES.md`へ記録する。

## 実験化の入口・引き継ぎ・承認

- 入口: 設計済み`trackastra_association`。2026-09-23の「実装に進んでください」を実験化承認として受けた。
- 対応する上位仮説: `HYP-20260910-10`。主に6時点の固定検出集合から1・2時点先の対応を学ぶことを検証する。
- 親実験: `exp016_frozen_image_encoder`。cache生成元は`exp015_oracle_stage_limits`。
- 手法契約: 入力は固定点・32次元画像特徴・物理座標・frame、教師は既知祖先と確認できる誤親、出力は窓内対応logitと親方向の確率、損失はmasked BCEと0.01倍の補助BCE、推論は重複窓の確率平均から既存ILPへ、処理単位は6時点窓。
- 実装区分: `staged-faithful`。候補では点入力・疎注釈への変更と、初回の全graph評価前の停止条件を明記している。
- 未決事項: なし。設計確認後の構成変更はユーザーへ確認する。

## 確定した設計の要約

2026-09-23にユーザーが「推奨案で設計を確定」と回答し、以下を初回1構成として確定した。2026-09-23の「実装に進んでください」で実験化と実装が承認された。実行は予算・教師・runtime gateに従う。

| 項目 | 確定した初回仕様 |
| --- | --- |
| 入力 | 未来側も含む連続6時点の全固定検出点と32次元の固定画像特徴。stride 1。動画全履歴は使わない。 |
| 構造 | 公式実装の標準値を基準に、幅128、4 heads、encoder 4層・decoder 4層、dropout 0.1。ランダム初期化。 |
| 教師 | 1・2時点先の既知の祖先―子孫対応。対応先の祖先が確認できる列で、注釈に対応した別の親候補だけを負例にする。 |
| 損失 | 親候補と固定値0の「対応なし」logitを正規化するparental softmax、二値交差エントロピー（BCE）、重み0.01の補助BCE。未知は直接の損失から外す。 |
| 学習 | 胚を分けた2fold、各10 epochs、1構成・モデル2個。学習側内部検証だけでcheckpointと閾値を選ぶ。 |
| 推論 | 重複窓の隣接対応確率を単純平均。逆時間推論とsecondary trackerの融合は初回では使わず、確率を直接既存の整数線形計画（ILP）と後処理へ渡す。 |
| 進行条件 | 両胚で接続recallが改善、正解親1位率・分裂親回収が非減少、教師負例予測の増加が5%以内。全graph実行は結果を示して別途判断する。 |

## 観測事実と根拠

- 実測済みの事実: 本候補の学習・評価は未実施。exp027とexp032の3時点attentionは両胚で保存済みexp016を上回らなかったが、Trackastraの対応教師と損失の比較ではない。
- 根拠ファイル / 一次資料: [設計に必要な実装照合](../../docs/surveys/biohub-trackastra-association-design_20260923.md)、[Trackastra論文](https://arxiv.org/html/2405.15700v2)、[公式実装の固定revision](https://github.com/weigertlab/trackastra/tree/aa57a95160002e0fc70b915ab74178b39c99fd6a)、[exp027結果](../exp027_multi_frame_tracker/result.md)、[exp032結果](../exp032_three_frame_ten_epoch_training/result.md)。
- 利用する保存済み生成物とSHA: exp015 cacheと公開重みのidentityはexp016 config、比較重みのSHAは[exp016 metrics](../exp016_frozen_image_encoder/metrics.json)を正とする。参照sourceは`aa57a95160002e0fc70b915ab74178b39c99fd6a`、BSD-3-Clause。今回cacheを取得・実測してはいない。窓manifest・教師監査・生成物SHAは実験時に記録する。
- 仮定: Assumption: 複数時点の文脈と親候補間の競合を学べば、固定候補内で接続を選ぶ精度が改善する。10 epochsで十分に学べること、費用内に収まることは未実証。固定候補にない細胞は本案だけでは回収できない。

## この候補が直接検証する仮説と範囲

- 上位仮説のうちこの候補が検証する範囲: 短窓のencoder-decoder、1・2時点先の対応教師、親方向の正規化と損失を組み合わせた適用。
- この候補の具体的な仮説: 固定検出・固定画像特徴からのTrackastra対応学習が、exp016より両胚の既知接続回収を改善し、分裂と誤親の指標を保つ。
- 仮説が正しい場合に期待する観測: 共通ペアで正解親順位と接続recallが改善し、分裂親を失わず、密集度・境界別の診断でも改善の範囲を説明できる。
- 仮説を棄却する観測: 事前に固定した1構成で進行条件未達。この10 epochs・部分注釈への適用の結果として記録し、Trackastra一般や時間情報全般の無効とは結論しない。教師・費用が成立しない場合は精度仮説を判定不能とする。
- この候補だけで上位仮説を判断できるか: いいえ
- 上位仮説の判断に残る検証: [past_feature_cross_attention](../exp041_past_feature_cross_attention/)、[multi_time_past_candidate_attention](../../backlog/multi_time_past_candidate_attention.md)、予測履歴、公式graph評価。本比較だけで時間窓・構造・教師mask・初期値の各寄与を分離しない。

## 手法契約

- input: 同一動画の連続6時点の固定検出点、物理座標z/y/x、実際のframe番号、有効mask、32次元`primary_features`。注釈や予測済み軌跡を入力にしない。
- target / objective: 後述の既知祖先対応。隣接と2時点先を学習し、兄弟細胞を同一対応としない。
- output: 窓内の点対のlogitとparental softmax確率。最終graphには時間差1の対応だけを渡す。母1個から娘2個を許す。
- loss: 既知maskに対する正規化確率のBCEと、同じlogitのsigmoidに対する補助BCE。重みと集計規則は後述する。
- decode / 推論方法: 窓ごとの確率を平均して隣接edgeへ戻し、凍結した閾値で候補edgeを選ぶ。親1個・娘最大2個の制約を既存ILPと後処理で適用する。
- 処理単位: modelと教師は6時点の検出集合、共通診断は隣接2時点、公式評価は動画全体。
- 実装区分: 対応モデル・parental softmax・補助BCE・重複窓集約を保ち、入力と部分注釈への適用変更を明示する`staged-faithful`（このリポジトリ内の管理用語）。原論文の完全再現、公式学習済み重みの直接適用とは呼ばない。

### 時間窓と固定特徴

1. 100-frame動画では開始0〜94の95窓を使う。一般の長さでは6時点窓をstride 1で作り、2〜5 frameの短い動画は実frame全体で1窓とする。1 frameは接続出力なし。存在しないframeの複製や時刻の詰め直しをしない。
2. 窓の最後を除く各frameは、そのframeと次frameのexp015 cacheの`src`特徴を使う。最後のframeだけ直前pairの`tgt`特徴を使う。画像特徴の元pairは常に6時点窓内に収まり、新しい画像forwardや窓外画像を使わない。既存の`position_features`は連結せず、位置はTrackastraの位置表現で与える。
3. identityは`sample, actual_frame, candidate_id`。重複cacheのID・順序・grid座標・物理座標が一致することをexp027と同様に検査する。不一致なら停止し、最近傍で重複を統合しない。同じframeでも元pairが違う画像特徴は一致を要求せず、平均して単一frame特徴へ置換しない。
4. 各窓では1検出1 token。同じframeを複数窓で読む際の特徴元pairとsideをmanifestへ記録する。exp016と候補・元cacheは共通だが、全窓で中央pairと同じ特徴を使う比較ではない。
5. frame番号はmetadataから取得し、窓開始からの差を時間座標にする。空frameも時間差に残す。実秒の間隔は未確認なのでframe差を秒と呼ばない。座標はµm、空間cutoffは256 µmに固定。距離mask以外の近傍数制限・候補削減はしない。
6. 窓の後半frameも文脈に含むオフライン推論。全padding窓はmodelを呼ばず空の出力、空frame間のedgeは0件とする。未知の注釈とpaddingを混同しない。

### 疎い教師と損失mask

| 対象 | 教師・扱い |
| --- | --- |
| 候補―注釈対応 | exp016の距離順greedy one-to-one、5 µm。動画・frameごとに固定し、窓ごとに割当を変えない。 |
| 時間差1の正例 | 両端が対応したGEFFの有向edge。 |
| 時間差2の正例 | GEFF上の連続する2本のedgeで保証された祖先―子孫。中間注釈は必要だが中間の検出成功は不要。 |
| 負例 | 子の該当時点の祖先が一意に判明し、その正例祖先の検出も存在する列で、別の注釈IDに対応した親候補。未対応候補は負例にしない。 |
| 未知 | 親不明、対応不明、途切れた注釈経路、正例祖先の検出欠落、窓外の対応、未注釈候補。直接BCEのmaskを0にする。 |
| 分裂 | 既知の母―娘edgeは両娘とも正例。各娘の親方向に正規化し、母の娘方向を合計1にしない。 |
| 対応なし | 今回は明示教師なし。注釈端・見逃し・crop外を出現・消失の正解にしない。 |

- GEFFに時間逆行・複数親・同時刻edgeがあれば停止する。時間差2はframe差が1ずつの2本だけで作る。経路不在を負例の根拠にしない。
- 親候補間の正規化は、子と親frameの組ごとに行う。1時点前と2時点前を同一の分母に混ぜない。分母には距離内の全候補のexp(logit)と固定値1を含める。後者は「対応なし」のlogit 0に相当する。paddingと距離外だけを分母から除き、教師maskで推論候補を絞らない。
- 未対応候補は直接BCEの対象外でも、既知正例との正規化競合を通して勾配を受ける。この間接作用まで「未知に勾配なし」と説明しない。すべて未知の子列では両BCEを無効化し、確証のない対応なし教師を作らない。
- BCEの補助重みは0.01。要素重みは通常正例2、既知の分裂を通る正例11、既知負例1。分裂は完全なGEFFの窓内経路から判定し、娘の検出欠落で通常正例へ戻さない。時間差1・2を同じ重みで扱う。分裂のあるsource行全体を一律重くするのではなく、正例を重くする適用仕様である。
- 有効要素の重み付き損失和をその窓の有効要素数で割る。batch内ではこの窓損失を等重みで平均。公式train scriptの大きい窓への追加重み付けは使わない。focal loss、対照損失、同時刻・逆時間・3時点以上先の損失は追加しない。
- loss計算と正規化はfloat32。logsumexpで固定の「対応なし」を含め、全距離外・空親集合は接続確率0、対応なし確率1とする。未知要素にNaNを作って0を掛ける実装にしない。有効教師0の窓は勾配更新から除き、件数を記録する。

### モデルと学習の固定値

| 項目 | 初回仕様 |
| --- | --- |
| 参照 | `TrackingTransformer`の固定revision。3D座標、入力画像特徴32次元。公式重みは使わない。 |
| 構造 | `d_model=128`, `nhead=4`, encoder/decoder各4層、`dropout=0.1`、`pos_embed_per_dim=32`, `feat_embed_per_dim=1`。 |
| 位置表現 | 公式の位置埋め込みとRoPE（回転位置埋め込み）、`attn_dist_mode=v0`, `attn_positional_bias_n_spatial=16`、空間cutoff 256 µm、時間window 6。時間原点の変更は時間列だけに適用し、空間座標を動かさない。 |
| 対応head | encoder・decoder出力を別々の2層MLPに通した内積。既存pair MLPで代替しない。 |
| 初期化・乱数 | 公式moduleのランダム初期化、seed `42 + fold`。動画・窓の順序はfoldとepochから決まるseedでシャッフルする。公開primary trackerの重みを部分移植しない。 |
| 学習 | 各fold 10 epochs、AdamW、学習率0.0001、weight decay 0.01、schedulerなし、gradient clip norm 1.0。 |
| batch・数値 | microbatch 1、2窓のgradient accumulationで実効batch 2、末尾1窓は実件数で平均。float32、mixed precisionなし。 |
| 入力拡張 | なし。全候補・全6時点を保つ。既存固定特徴に座標だけの回転や時間反転を加えない。 |
| 分割 | exp016と同じleave-one-embryo-out 2fold。学習胚内の動画単位90/10分割、seed 0。window単位のランダム分割はしない。 |
| checkpoint | epoch 1〜10の学習側内部検証における同じmasked loss最小、同値なら早いepoch。外側胚は選択完了後だけ採点。 |
| モデル数 | 1構成×2fold=2、追加seed・boosterなし。control再学習なし。 |

原論文の幅256・各6層とは規模が異なる。公式実装の標準値を使って中核の学習方法を保つ案であり、原論文と同じ学習量・精度の再現とはしない。確定したハイパーパラメータは`config.yaml`に記録した。

### 得点集約と既存graphへの接続

1. 同一の隣接edgeを含む全窓のparental softmax確率を算術平均する。内部edgeは通常5窓、端では実在窓数を分母にする。logitの平均、平均後の再softmax、端を固定5で割る処理は禁止。
2. 初回のTrackastraは時間順方向の出力だけを使う。後の時点を文脈として見ることと、逆時間の接続確率を混ぜることを区別する。secondary tracker・exp016の逆方向logit融合は使わない。exp016のprimary単体も同じ共通ペアで再推論し、学習はしない。
3. 確率を直接受け取るedge選択処理を用意し、候補ID・順序・距離・graph schemaを既存処理へ合わせる。`select_cached_candidate_edges`のsource軸softmaxへlog(probability)を渡すだけの変換は禁止。「対応なし」に残った確率が消えるためである。
4. 閾値の校正はcheckpoint決定後に学習側の内部検証だけで行う。exp016 primary単体の確率0.5超で得るlegacy mask内の教師負例予測数を上限とし、Trackastraの同件数以下となる最小閾値を選ぶ。候補閾値は0、観測した負例得点の全distinct値、1。比較は厳密な`score > threshold`、同値はまとめて除外する。負例分母0なら校正不能として停止する。
5. 閾値はfoldごとに固定して外側胚へ適用し、graphへ進む場合もその値を使う。順位や確率を変換せず、temperature/bias fitting・外側胚での調整はしない。固定0.5と既存graphの0.48は参考診断として併記するだけで、良い方を選ばない。
6. graphへ進む場合は既存ILPのedge=-1.0、appearance=0.0、disappearance=2.0、division=1.2、既存graph repairと公式評価器を固定する。保存済みexp016の通常構成に加え、exp016 primary単体・閾値0.5を同じILPへ渡す対照も評価する。これにより補助trackerを外した差を観測する。Trackastraと通常構成の比較だけを純粋なmodel差とはしない。
7. 隣接候補のIDと座標を変えず、時間差2の対応は学習専用とする。復号前後の入次数1以下・出次数2以下、edge score非有限値、候補消失を検査する。後処理による修復効果はpair診断では測れない。

## 親実験からの差分

- 変更するもの: 6時点入力と特徴元pair、encoder-decoder対応モデル、初期値、教師mask、非隣接教師、損失・正規化、10 epochs、重複窓集約、確率を直接受け取る接続と校正。手法全体の比較として扱う。
- 固定するもの: 公開検出・画像特徴モデルの重みと正規化統計、exp015の検出候補・座標・特徴値、主催者GEFF、5 µm対応、胚分割、ILP費用と後処理、公式評価器。
- 再利用するコード / config / 生成物: exp016のcache・GEFF読込・greedy matching・split、exp027の重複frame identity検査、exp016のgraph schemaとrepair。legacy loss maskは対照診断専用で、新modelの学習へ流用しない。
- 新しく作るもの: 本実験内に窓manifest、部分教師、Trackastra入力adapter、masked loss、集約と確率入力adapter、学習・診断Notebook、受け入れテストを実装する。

## 最小の反証可能な検証

- 検証方法: まず全入力のidentityと教師件数を監査し、時間差1/2・分裂・既知負例・未知の分母をfoldの学習/内部検証別に記録する。学習側64窓に対する費用測定と学習成立確認後、2foldを学習する。外側評価ではexp016の`filter_nonempty_gt_window_paths`と同じく両frameに注釈が存在する隣接ペアを全件比較する。正例0のペアはrecallの分母を増やさず、除外・有効mask件数を共通に監査する。窓境界やGTのない文脈frameを恣意的に除外しない。
- 学習成立確認: 各foldの学習側からSHA256順で通常対応ありの窓と分裂ありの窓を各1つ固定し、1個の診断用modelで最大100 updates。dropoutを無効にした評価lossが初期値より低下し、正例logitが有限で勾配が届くことを確認する。分裂窓がない場合は教師不足として停止。診断重みは破棄し、本学習は同じseedで再初期化する。これは追加の評価variantやOOF選択にしない。
- variant / config / fold / booster数: 本学習1構成・2fold・モデル2個・10 epochs・booster 0。対照は保存重みの再推論だけ。
- control再学習: なし。exp016の保存済みprimary単体を同一隣接ペアで推論する。元のpair特徴を維持し、入力契約の差を明記する。
- 想定runtime / resource（version 1の契約）: Kaggleのみ、GPU週30時間・課金なし、各Notebook12時間以内。学習側の64窓（token数上位32窓とSHA256順32窓）で費用を見積もり、1.5倍の余裕を付けて2foldの総費用を判定する。各Notebookと残GPU割当時間の両方に収まる場合だけ本学習へ進む。実装された測定はforward/backward/optimizerと窓の再読込を一括計時したもので、validation forwardと対照再推論は別途実測せず学習窓の時間で代理した。この差はversion 1の費用推定の限界であり、下記の再設計で測り直す。
- メモリ方針: 全tokenを保持し、2048等で切り捨てない。距離maskがあっても行列の二乗メモリは残る。メモリ不足では同一計算のgradient checkpointingまたは検証済み等価な分割だけを認め、窓・候補・層・幅・epochを無断で減らさない。改善しなければ停止する。
- 受け入れテスト案: identityと元特徴の照合、空frameの実時刻、祖先と兄弟の区別、2-step経路の中間検出欠落、未知BCE maskと正規化経由勾配、1母2娘、親なし集合、padding、全未知窓、数値安定性、親frameごとの確率和1以下、境界窓の平均、確率adapterの再正規化防止、fold漏洩と閾値選択範囲を検査する。

## 成功条件と停止条件

- primary指標: 全共通隣接ペアを集計した胚別の既知接続recall。学習側で校正した閾値を主判定に使う。公式精度は後段の公式combined scoreと成分で判断する。
- 成功条件: 各胚でexp016 primary単体（閾値0.5）より既知接続recallが厳密に改善し、正解親1位率が非減少、両娘とも回収した分裂親数が非減少。共通legacy mask内の教師負例予測数は対照の1.05倍以下。安全な負例として定義した注釈対応済み誤親についても同条件を要求する。対照負例数0なら候補側も0。有効な接続・分裂・負例分母が各胚に必要で、欠ける場合は判定不能。
- 必須guard: 件数・分母・除外理由、固定閾値0.5/0.48、正解親順位、分裂両娘回収と候補娘数、予測親数、学習曲線、token数・空間密度・動画端別の結果。密度は15 µm以内の同frame候補数とし、bucket境界は学習側四分位点で固定する。旧maskの負例指標は未注釈を含み、真の誤接続全数とは呼ばない。
- 成功時の次段階: 単体結果・教師限界・全graph費用を示してユーザーに進行判断を求める。公式評価はKaggle上で全graphを作って実行する。採否・完了・submissionは別の明示判断。
- 失敗時の停止範囲: identity不整合、教師不足、非有限値、費用超過、片胚でも進行条件未達なら全graphへ進まない。非隣接教師がないときに隣接だけへ変更しない。公式score未計測と記録し、設定救済探索を同じ外側予測上で始めない。

## 実行しないこと

- 禁止する代替実装、proxy、同一OOF上の救済探索: 未知を全負例・対応なしにすること、正解軌跡入力、画像encoder更新、検出候補追加、既存pair MLPの窓長変更だけ、外側胚での窓長・閾値・loss選択、候補・時間窓の無断削減、secondaryの無断再融合。
- 壁打ちで採らなかった案と理由: 全履歴はユーザーが不要とした。公式学習済み重みの直接適用は入力特徴が異なる。領域からの擬似教師は[track_teacher](../../backlog/track_teacher.md)の別案。過去画像特徴だけを照会する案は[past_feature_cross_attention](../exp041_past_feature_cross_attention/)、過去座標集約は[multi_time_past_candidate_attention](../../backlog/multi_time_past_candidate_attention.md)で扱う。exp027/032の時点数追加では本案を代替しない。原論文規模の幅256・各6層は初回には採らず、ユーザーが選んだ公式実装標準の幅128・各4層で中核機構を検証する。

## 再現性・リスク

- leakage / validation: 固定公開画像重みの来歴にtrainデータが含まれるため独立CVとは呼ばない。教師構築・校正・checkpoint選択には学習側だけを使う。外側の教師件数は最終診断時に報告し、学習条件を変える根拠にしない。
- hidden test: 空frame、2〜5時点の短い動画、検出密度と特徴元pairを同じ規則で処理する。教師を必要とする入力経路を作らない。test推論費用と公式評価は未測定。
- runtime / memory: 二乗の対応行列とattention。固定特徴でも原論文と同じ収束・速度を保証しない。GPU割当消費とNotebook経過時間を別々に記録する。
- 再現性: source revisionとライセンス、入力/窓/教師/split manifest、seed、checkpoint、校正閾値、modelと予測のSHAを保存する。bitwise一致は保証しない。

## 調査・実行時に確認する事項

- 入力と対照重みの取得・SHA、全窓の候補identity、1・2時点先と分裂の有効教師件数、256 µmでの正例被覆、費用・ピークメモリ、学習成立、精度、hidden test費用。これらの未実測は方式の未決と区別する。
- 距離mask外の正例があれば教師監査に件数を残し、受け入れ前に停止する。密な候補を削減して辻褄を合わせない。

## 12時間gate後の再設計

2026-09-23のKaggle train version 1は教師監査と2窓の学習成立確認を通過したが、保守的な予測363,591秒が12時間gateを超えたため学習前に停止した。64窓の半数は各foldでtoken数が最大の窓であり、その平均は全学習窓の平均ではない。またvalidationと対照推論の実時間を測らず、forward/backward付きの窓時間を代入した。したがって101時間は元構成の実行時間の測定値ではない。ユーザーは同日、学習窓の抽出を含む再設計を了承した。

1. 元の2foldについて全学習窓のtoken数、教師あり窓数、時間差1/2の正例と分裂正例を記録する。外側胚の注釈・スコアは抽出方針や予算決定に使わない。token数の密度帯ごとに窓を抽出して、動画を一度読み込んだ後の教師生成、GPU学習step、学習側validation forward、重複窓の推論forward、保存済み対照の推論を個別に計時する。最大密度の窓は平均時間へ半数混ぜず、メモリと極端な実行時間のstress testとして別集計する。
2. 全窓の密度帯別件数で処理別の測定値を重み付けし、setup、10回の内部validation、両foldの外側胚の全隣接ペア評価、保存済み対照の再推論、生成物保存も計上する。1.5倍の余裕を維持し、Notebookの12時間上限、実際の残GPU割当、週30時間方針のすべてに収まるかを判定する。学習窓を抽出する場合、学習側だけの教師・token情報を使って抽出manifestと全epochの窓IDを学習前に固定し、manifestのSHAを保存する。後から外側結果に合わせて窓数や抽出確率を変えない。
3. 比較の軸を守るため、6時点入力、全token、幅128・encoder/decoder各4層、部分教師、損失重み、10 epochs×2fold、全窓での内部validation、外側胚の全共通隣接ペア、閾値校正と進行条件は維持する。抽出対象はgradient updateに使う窓だけとし、学習に使った窓数と教師種別・密度帯・動画別のcoverageを報告する。全窓で学習した結果とは呼ばず、抽出した窓での10 epochsの結果として解釈する。
   - 学習に有効な教師を持つ窓を、優先順に分裂正例あり、時間差2正例あり、隣接正例あり、既知負例のみへ分類し、token数の学習側四分位帯と組み合わせる。各層の母集団比率に従って最大剰余法で毎epochの抽出数を固定し、どの教師種別も0窓にはしない。窓はseed、fold、動画、開始frameのSHA256順に並べ、各層でepochをまたいで循環させる。1 epoch内では重複させず、同じ層の全窓を回る前に先頭窓を再選択しない。抽出上限は両foldで共通の4,096窓/epochとし、全epochの窓IDを学習前に保存する。
4. Kaggle train version 2の密度帯別計測を用い、毎epoch各fold4,096窓を選ぶ。10 epochs×2foldの合計で81,920窓を学習処理する。全教師あり窓は各foldで少なくとも1回選ばれる。計測帯ごとの平均に標準誤差2倍を加え、処理別に積み上げて1.5倍し、1時間の固定予備を加えた予測は35,182.98秒（約9.77時間）。12時間までの余裕を優先し、最大可能な5,517窓/epoch（予測約11.96時間）は選ばない。Notebook側は固定した費用assetと全窓manifestのSHAを照合し、さらに当日のpreflight経過時間を予測へ加えて12時間gateを判定する。GPU週30時間方針と実際の残割当も実行前に別途確認し、費用が収まらなければ学習を開始しない。単体診断が両胚で進行条件を満たすまでは全graph推論へ進まない。公式scoreと採否・完了は未判断のままとする。

## Colabへの実行経路変更

2026-09-23、ユーザーはKaggle GPUの残量不足を理由に本実験をColabで実行するよう指示した。これはexp040の本学習と共通隣接ペア診断に限る実行環境の変更であり、モデル、部分教師、各fold毎epoch4,096窓の固定スケジュール、10 epochs、保存済みexp016対照、全窓の内部検証と外側胚評価、進行条件は変えない。公式graph評価を主張するときは、別途Kaggleで全graphと公式評価器を実行する。submissionは依頼されていない。

- Colab CLIのT4を既定にする。Driveへの手動配置を要しない。コード・設定・保存済み対照重みと公開モデルsourceだけを小さいZIPで転送する。exp015 cacheと既存の非公開CPU exportで得たGEFFはKaggleの短命HTTPS URLからColabへ直接取得し、ローカルに大容量cacheを保存しない。Kaggle認証情報をColabへ転送しない。
- ZIP各member、GEFF各member、cacheのsummary・identity、対照modelと公開sourceのSHAを学習前に検証する。各epoch終了時にmodel、optimizer、最良checkpoint、乱数状態、historyを保存し、receiptのbyte数・SHAを照合してローカルに回収する。中断時は同一configと窓スケジュールの検証済み最新epochから再開する。
- 固定runtime assetの予測35,182.98秒はKaggle T4上の計測である。Colab T4の実測時間を保証しない。12時間gateを維持し、途中で中断した場合は完了扱いにせず、検証済みepochから再開する。GPU割当・無料枠の制限を追加の学習条件変更の理由にしない。完了時は2モデル、外側胚別の共通ペア指標、model manifest、completion markerのSHAをColab外で検証する。

## 未決事項

- なし

## 判断履歴

- 2026-09-22: 会話で手法と適用可能性を検討。ユーザーの「3案すべて追加してください」に従いP2で登録。方式の未決を残し、実験化・実装・実行は行わなかった。
- 2026-09-23: ユーザーが設計確定を依頼。論文・公式source・exp016/027の接続仕様を照合し、上記の1構成案を作成。
- 2026-09-23: ユーザーが「推奨案で設計を確定」と回答。6時点のオフライン入力、幅128・encoder/decoder各4層、部分教師とparental softmax、10 epochs×2fold、secondary融合なしの直接確率接続を確定した。未決事項をなしとし、状態を`設計可能・実験化未承認`へ更新。この時点では実験化・実装・実行は未承認だった。
- 2026-09-23: ユーザーの「実装に進んでください」を、設計済み候補の実験化と実装承認として受けた。exp040へ移行。
- 2026-09-23: version 1の費用gate停止後、ユーザーがモデルとepoch数を維持して学習窓だけを抽出する方針を選択。version 2のKaggle T4計測から4,096窓/epochの固定抽出と費用gateを設計した。週次GPU使用量は30時間方針を既に超えており、再設計後の本学習はまだ起動していない。

## 引き継ぎ確認

- 固定するものを一意に説明できる: はい。固定入力・候補・教師の出所・胚分割・ILPと後処理。
- 変更するものを一意に説明できる: はい。確定構成の入力・構造・教師・loss・集約と確率接続を上記へ記載。
- 最小検証と停止条件を一意に説明できる: はい。教師監査、学習成立・費用測定、10 epochs×2fold、共通隣接ペアの数値条件。
- 実行しないことを一意に説明できる: はい。全履歴、検出器更新、擬似教師、未知の負例化、submission。
- 未決事項が明示されている: はい。なし。測定待ちは別節へ分離済み。確定済みの方式は本実験契約へ移行済み。

## 実装方法

- 入力cache・特徴の照合: `trackastra_association.py`の`load_video_cache`と`build_context_window`。exp016のcache/GEFF schemaとSHAを検査し、同じframeの候補ID・座標の重複一致を確認する。
- 教師: `trackastra_association.py`の`build_sparse_teacher`。動画・frameごとの5 µm one-to-one候補対応、隣接と2段のGEFF経路、既知誤親・未知maskを構成する。
- モデルと損失: `trackastra_association.py`のTrackastra encoder-decoderと親方向のquiet softmax、masked BCE。source revision `aa57a95160002e0fc70b915ab74178b39c99fd6a`を固定し、必要なコードのBSD-3-Clause表示を保持する。
- 学習: `exp040_trackastra_association_train.py`のJupytextセルと同名Notebook。cache・教師監査、費用測定、2窓の学習成立確認、2fold×10 epochs、内部checkpoint選択、outer胚の共通pair評価、manifest保存を行う。
- 診断と推論: 学習Notebookが共通pairの予測と得点集約・閾値校正を実施する。全graphへの接続とKaggle公式評価は進行条件を満たした場合に同じ実験で実装・実行する。
- 固定事項の検査: optimizerは新しいassociation modelのparameterだけを含み、公開画像encoderと検出器のforward countは0。元cacheと候補IDは変えない。
- テスト: 合成動画で空frame・分裂・2段祖先・未知mask・親方向確率・窓端平均・閾値と入出次数を検査する。`make check-exp EXP=exp040_trackastra_association`、`make test-exp EXP=exp040_trackastra_association`、`make validate-exp EXP=exp040_trackastra_association`を実行する。
- 変更class: `mechanism`。入力・教師・出力と損失が変わるため、小さなparameter変更ではない。
- 再現性: seed `42 + fold`、入力cacheとsource SHA、窓・split・教師manifest、checkpointと共通pair予測のSHA、kernel versionを`metrics.json`と学習生成物に記録する。公開画像重みの学習来歴により独立CVとは呼ばない。
- 完了判定: ユーザーが採否・実験完了を判断するまで`metrics.json`の実行statusだけを記録し、採用・不採用を確定しない。

## 探索幅とpivot判定

- 変更class: `mechanism`。6時点入力、encoder-decoder、1・2時点先の部分教師、親方向の正規化と損失を一組として比較する。
- 比較する構成: 確定済みの1構成だけ。幅256・各6層、secondary融合、teacher maskや閾値の外側胚での調整は別の設計判断を要する。
- pivot条件: 入力・教師・費用が成立しない、または片胚でも事前指定の隣接ペア進行条件を満たさない場合は全graph実行を止め、観測値を提示する。
- 同一系列の小変更を続ける理由: 本案は既存の3時点attentionの層数調整ではなく、出力・教師・正規化まで変える独立の比較である。`kaggle-idea-forge`は今回の確定設計を実装する段階では不要。

## 受け入れ基準

- [x] 元候補の仮説、根拠、固定対象、変更対象、禁止事項、成功・停止条件、ユーザー判断を移し、`lineage`を設定した。
- [x] 固定cacheのidentity、疎い教師、Trackastraモデル、親方向確率、窓間集約、既存対照の共通ペア診断を実装した。
- [x] 学習用Jupytext sourceとNotebookに、教師監査、学習成立確認、費用gate、2fold×10 epochs、内部checkpointと閾値選択、外側胚診断、生成物SHA保存を実装した。
- [x] Notebookのround-trip、実験構造、lint、実験固有テストを通す。
- [x] Kaggle上でGPU quotaを確認し、教師監査・費用gateを実測する。version 1は両foldの教師・学習成立を確認し、保守的費用予測が12時間gateを超えたため本学習前に停止した。
- [ ] 条件を満たす構成が承認された場合だけ2foldを完走し、model 2個と外側胚の診断を保存する。
- [ ] 両胚の進行条件が成立した場合は、同じ確率を固定ILPとgraph repairへ渡すinferenceを実装・検証する。全graph実行の前に結果を示し、進行判断を仰ぐ。
- [ ] 公式score、実験完了、採用・不採用は実測とユーザー判断前に確定しない。Kaggle submissionは別の明示依頼まで行わない。
