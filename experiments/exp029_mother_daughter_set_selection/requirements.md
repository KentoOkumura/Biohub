# exp029_mother_daughter_set_selection 要件と実装方法

この文書を実装前の契約とする。進捗と実行コマンドは`SESSION_NOTES.md`へ記録する。

## 実験化の入口・引き継ぎ・承認

- 候補名: `mother_daughter_set_selection`。移行元は`backlog/mother_daughter_set_selection.md`。
- 実験化の承認: 2026-09-20の「`mother_daughter_set_selection`を実装してください」。同日、母ごとに空・1娘・2娘の集合logitを直接出し、既知edgeと既知の矛盾に整合する集合の確率を周辺化して学習し、母間競合を全体最適化で解く方式をユーザーが選択した。
- 主仮説: `HYP-20260910-11`。関連仮説は`HYP-20260910-03`。本実験は固定候補と既知接続教師を保ち、出力単位を娘から母へ変える効果を検証する。疎い教師全般、通常継続と分裂の識別、他の出力方式、全graphの効果は残るため、これだけで主仮説全体は判断しない。
- 親実験と対照: [`exp028_direct_graph_prediction`](../exp028_direct_graph_prediction/requirements.md)の娘ごとの親選択、および保存済み[`exp016_frozen_image_encoder`](../exp016_frozen_image_encoder/result.md)のprimary tracker。exp028の両胚で観測可能な誤接続が対照より増えたことが動機だが、本方式の改善証拠ではない。
- 根拠: [`exp028 metrics`](../exp028_direct_graph_prediction/metrics.json)、[`exp020`](../exp020_division_triplet_candidates/result.md)と[`exp021`](../exp021_division_teacher_audit/result.md)の分裂候補・教師監査、[公式評価仕様](../../docs/official/evaluation.md)。Zyssらの母と2娘の組を評価する[論文](https://link.springer.com/article/10.1186/s12859-025-06344-5)は参考であり、この実装の再現元ではない。
- 固定するもの: 公開検出器・画像特徴抽出器、exp015 cache identity `440891c4550adf8540e3c47f9784e68b6ca1b64bbe56dbb93a25eb87c46bc1ee`とsummary SHA `040d1f6437e27149e0e34ad4aac8bba3c8cf63dc266d33df497acd969972194c`、候補ID・座標、5µmのGEFF対応、胚入替の2fold、隣接2-frame評価、保存済み対照、全graph時の固定graph repair。secondary trackerと順逆融合は接続選択に使わない。
- 変更するもの: 各母が娘IDの集合を選ぶモデル出力、部分注釈の集合損失、娘の重複を禁止する全体選択。娘の個数だけの分類や単独edge得点の加算にはしない。
- exp028の保存済み対照: 6bbaの既知edge recall 95.35%、観測可能な誤接続率5.93%、既知分裂母45/108。44b6は94.37%、7.21%、12/22。exp016の同単位対照は順に97.20%、2.94%、50/108と95.30%、3.61%、6/22。exp028の全graphと公式scoreは未計測。
- 利用する保存済み生成物の照合: exp028 early gate SHA `82e3ae09c4f6748823d4199605193cd26664dae0a939ec2aa92d731b7f1f0b00`、model manifest SHA `e1309da80ca7d55a0496582c4b89fbc8d7e0e1917a93c185a17e0c2ed1b03244`。必要な娘別予測は保存済み重みから再生成する。
- 採らなかった案: `division_triplets`は分裂組だけを採点し、`division_state_model`は継続・分裂・観測不能の状態を選び、`division_local_ilp`は既存得点による局所再割当である。これらを本実験の代替実装として混ぜない。
- 未決事項: なし。候補詳細の未決事項を上記のユーザー選択と下記の実装仕様で解消した。

## 判断履歴

- 2026-09-20: ユーザーが母ごとの0・1・2娘選択に関心を示し、別の未着手候補として記録するよう依頼。exp028の採否・完了は判断していない。
- 2026-09-20: 実装を依頼し、集合logit、部分観測の周辺化、固定距離・件数上限、漏れた既知娘の集計・学習除外、娘重複を禁じる全体最適化を承認した。

## 手法契約

- input: exp015の隣接2-frame固定候補ID、物理座標、検出得点、primary画像特徴と位置特徴。推論入力にGEFF、GT対応、注釈有無を入れない。
- target / objective: GEFFと5µmで対応した娘の既知入edgeを満たし、既知の別母または候補内に母がいない娘を選ばない母別集合。既知出edgeが0本または1本の母に、真の娘数0または1を割り当てない。
- output: 各母について空集合、娘IDを1つ持つ集合、または異なる娘IDを2つ持つ順序に依存しない集合のlogit。Transformerで両frameを符号化し、空・単娘・2娘の別headを使う。2娘headは娘特徴の和と絶対差を使い、単独edge logitの足し算にしない。
- 候補生成: 物理距離14µm以内の娘を距離とIDで並べ、母あたり最大8件から空・単娘・2娘集合を作る。学習・推論で同じ規則を使う。候補から漏れた既知娘をGTで追加せず、その母を集合損失から除外して件数と既知2娘回収率を記録する。距離・上限は学習側の事前監査で候補回収と費用を確認し、外側胚で変更しない。
- loss: 各母の候補集合softmaxについて、既知正例娘を全て含み、既知の別母または候補内に母なしの娘を含まない集合の確率を足した負対数。許容集合が全候補集合ならその母は教師情報なしとして除外する。既知正例が候補外ならその母のlossを除外し、監査件数に加える。未記録edgeは負例にしない。
- decode: 母ごとに集合を1つ選び、各娘の入次数を最大1とする整数最適化。目的は集合logitと空集合logitの差から、内部検証だけで選んだ娘1人あたりのedge costを引いた合計。母の出次数は最大2、時刻は前向き。母ごとの上位16集合と空集合で最適化し、同点は候補ID順で固定する。最適化失敗または制限時間超過は黙って別方式へ置換せず停止し記録する。
- context unit: 隣接2-frame。windowを時刻順に統合し、候補IDと時刻整合を検査する。全graphではexp028と同じ固定graph repairだけを適用し、前後を別々に評価する。
- 実装区分: 特定論文の再現ではない出力表現の変更。用語集の`faithful` / `proxy`は論文再現の程度を表すため本実験には該当しない。固定候補外の正解、独立CV、hidden testの精度は判断できない。

## 実装方法

- `exp029_mother_daughter_set_selection_train.py`の`read_window`と`make_example`が固定cache・5µm教師を読み、`enumerate_daughter_sets`が注釈を見ずに候補集合を作る。`make_set_supervision`は既知正例と確定した矛盾から許容集合maskを作る。
- 同ファイルの`MotherDaughterSetTransformer`が両frameの候補を符号化し、空・1娘・2娘の別headからlogitを出す。`partial_set_loss`で許容集合の確率を周辺化する。`decode_daughter_sets`は各母1集合・各娘最大1母を整数最適化する。
- train notebookの各foldは学習側の内部splitでcheckpointとedge costを選び、外側胚で同じ2-frameの保存済みexp016対照と比較する。候補集合数と既知2娘回収、MILP処理数、GPUメモリ、実行時間を記録する。
- inference notebookはtrain manifestから保存済みfold別モデルを読み、同じ候補生成とMILP復号を適用する。exp015の固定repair sourceをSHAで検査し、repair前後を公式評価器で比較する。gateが両胚で成立した証拠がなければ開始しない。
- 実験固有テストは順序に依存しない2娘集合、既知1娘と未知第2娘、既知の別母、候補漏れ、競合解決、周辺化損失を確認する。

## 探索幅とpivot判定

- 変更class: `representation`。exp028の娘ごとの母選択から、母ごとの娘ID集合選択へ変える。
- これは同じ親の小さなparameter変更ではなくoutputとdecodeの変更である。exp028では両胚で観測可能な誤接続が対照より増えたため、同じ方式の閾値救済は行わない。
- `kaggle-idea-forge`は不要。ユーザーが別方式の候補と主要な実装契約を明示的に選択した。

## 検証と停止条件

- 先行監査: 既知0・1・2本の母、既知2娘の候補回収、候補集合数、確定負例、娘の競合頻度、最大windowでの学習・復号時間とメモリを確認する。安全な教師または予算内の候補集合を作れなければ学習前に停止する。
- 最小比較: Kaggleで1方式、2fold、各fold1モデル、booster 0。exp016とexp028は保存済み対照を使いGPUで再学習しない。必要なら保存済み重みから同じ2-frame対象の予測だけ再生成する。モデルとedge costは学習側内部で選び、外側胚で選び直さない。
- 前段評価: 胚別の既知edge recall、観測可能な誤接続率、既知分裂母回収、構造違反、接続数を同じcache・教師・評価対象で比較する。両胚で構造違反0、既知edge recallのexp016からの低下が1 percentage point以下、観測可能な誤接続率がexp016以下、既知分裂母の回収がexp016以上の場合だけ全graphへ進む。部分注釈の限界を併記する。
- 成功条件: 上記の前段条件を両胚で満たし、全graphへ進んだ場合は公式combined scoreと接続・分裂成分を両胚で比較して費用も示す。公式scoreは2-frame結果から推定しない。
- 停止条件: 未知を負例化する、既知娘をGTで候補へ追加する、構造制約を破る、計算予算を超える場合は停止。前段条件が不成立なら全graphを自動開始しない。Kaggle submissionは別の明示依頼を要する。
- 禁止する代替: 娘数だけの分類、edge得点の和だけで集合を選ぶ処理、ILPや既存trackerの出力を教師にする処理、外側胚での閾値調整、GTによる候補点追加、secondary tracker・順逆融合の同時導入。

## 再現性・リスク

- seed 42+fold。dropout、DataLoader shuffle、GPU kernelはstochastic。deterministic anchorとは呼ばない。cache・教師対応・fold・候補集合・edge cost・decoder・model・Kaggle kernel versionとSHAを記録する。
- 公開checkpointの学習来歴を引き継ぐため、両胚の評価は独立CVではない。GEFFの疎さから観測可能な誤接続率は全graphの真の誤接続率ではない。
- 娘2人の組は候補数に対して二次に増える。週30 GPU時間、Notebook 12時間以内で、最大windowの候補数とメモリ・時間を計測する。最適化の失敗率も報告する。
- train variant 1、model config 1、fold 2、booster 0。control再学習なし。

## 受け入れ基準

- [ ] 既知0・1・2本、候補漏れ、確定矛盾、未知、娘の競合、同点、空候補の小例で損失と復号を確認する。
- [ ] Kaggleの2fold学習、2-frame胚別診断、生成物SHA、runtimeを記録する。
- [ ] 前段条件成立時だけ全graphで公式評価し、repair前後を区別する。
- [ ] 採否と実験完了はユーザーが判断する。
