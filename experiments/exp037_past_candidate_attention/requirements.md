# exp037_past_candidate_attention 要件と実装方法

この文書を、実装前の契約、実装方法、受け入れ条件の正とする。実行中の進捗は `SESSION_NOTES.md` へ記録し、この文書へ重複させない。

## 実験化の入口・引き継ぎ・承認

- 実験化の入口と承認: backlog候補 `past_candidate_attention` は `設計可能・実験化未承認` かつ未決事項なしだった。2026-09-22のユーザー依頼「past_candidate_attentionを実装してください」を実験化・実装の承認として扱う。Kaggle実行は2026-09-22の追加依頼「実行してください」と外部送信の明示承認により単体診断まで承認された。全graph推論、公式評価、submissionは承認に含めない。
- 移行元backlog: `backlog/past_candidate_attention.md`。
- 対応する上位仮説: `HYP-20260910-10`。
- 上位仮説のうちこの実験が検証する範囲: 全過去候補との3点座標関係を、現在の接続候補ごとに集約して使う効果。
- この実験だけで上位仮説を判断できるか: いいえ。
- 上位仮説の判断に残る検証: 長い位置履歴、見た目や周囲の運動、欠測回収、最終graphと公式score、hidden testへの移行。
- 親実験: `exp016_frozen_image_encoder`。入力cacheは `exp015_oracle_stage_limits` を使う。exp027の3時点cache対応検査とexp035の保存済みexp016対照に対する単体診断を実装上の根拠として再利用するが、比較上の親または追加対照にはしない。
- 観測事実: exp016は隣接2時点の全検出pairへ座標差を入力する。exp035は予測した過去対応から移動量を追加したが、両胚の既知接続recallと分裂親回収数が低下し、全graphへの進行条件を満たさなかった。本実験の効果は未測定である。
- 仮定: Assumption: 接続先ごとに過去候補との変位の整合性を学べば、単一の誤った予測履歴に固定せず、2時点だけでは曖昧な接続を補える。ただし座標だけで真の前身を一意に特定できるとは仮定しない。
- 固定するもの: exp015の検出器・画像特徴・候補、exp016の公開初期checkpoint、2方向の胚分割、教師対応・loss・mask、3 epoch、optimizer・学習率・seed・内部checkpoint選択、閾値、secondary trackerと最終復号。
- 変更するもの: 過去1時点の全候補座標、13次元の3点座標特徴、共有MLP、中央pairごとのattention集約、既存logitへ加えるdelta logit。
- 最小の反証可能な検証: 保存済みexp016 fold modelと新modelを同じ全外側評価window・候補・教師・固定閾値0.5で比較し、両胚を別々に報告する。1構成、外側2fold、各3 epoch、新規model 2個だけを学習する。
- 成功条件: 両胚で保存済みexp016より既知接続recallが厳密に改善し、active pair内の教師負例予測増加が各胚5%以内、分裂親回収数が各胚で非減少となる。最終的な精度改善は全graphで両胚の公式scoreが改善した場合に限る。
- 停止条件: 片方でも進行条件に届かなければ全graphを始めない。candidate ID・frame・座標不整合、非有限値、OOM、12時間Notebook gateまたは週30 GPU時間の超過でも停止する。fold・epoch・候補数を無断縮小しない。
- 実行しないこと: 正解軌跡や前段接続予測の入力、過去候補を1個へ確定、全候補の近傍・半径制限、外側評価を見た閾値・層幅・正規化・epoch探索、教師・loss・検出器・復号の同時変更、control再学習、追加seed、Kaggle submission。
- 未決事項: なし。

## 判断履歴

- 2026-09-22: exp035の結果を受け、ユーザーは予測接続を入力しない過去座標の利用を相談した。
- 2026-09-22: 13次元入力、共有MLPの13→32→32、候補別attention score、過去情報を使わない選択肢、32次元集約、biasなし・ゼロ初期化の追加出力を提示し、ユーザーが同設計のバックログ化を依頼した。
- 2026-09-22: ユーザーの「past_candidate_attentionを実装してください」により、同設計の実験化と実装が承認された。Kaggle実行、全graph、公式評価、submissionは未承認である。
- 2026-09-22: ユーザーの「実行してください」とprivate Kaggle Notebookへのコード送信の確認に対する「実行していいです」により、Kaggle上の学習・単体診断を承認された。全graph、公式評価、submissionは未承認のままとした。

## 手法契約

- 依頼原文: 「past_candidate_attentionを実装してください」。
- 期待する成果: 過去の接続を先に予測・確定せず、3時点の座標関係を入力することで、保存済みexp016より両胚の接続判断が改善するか確認できる実装を作る。
- input: exp016と同じ中央2時点の固定画像特徴・候補座標に、直前1時点の全検出候補の物理座標と有効maskを追加する。過去画像特徴、検出得点、予測接続確率は追加しない。
- target / objective: 主催者GEFFに基づく、中央の隣接2時点間の通常接続・母娘接続。過去候補の対応を確定する教師は追加しない。
- output: exp016と同じ全中央pairの接続logitに、集約した過去情報からのdelta logitを加える。検出中心や候補は変更しない。
- loss: exp016と同じsource軸softmax後のfocal-weighted binary cross entropy、既存の教師mask、division weight 1.0。追加部分とprimary trackerを同時学習し、attention重みへの対応教師・補助lossは加えない。
- decode / 推論方法: 初段は固定閾値0.5の単体診断まで。全graphは進行条件の成立とユーザー承認後に追加し、exp016のsecondary tracker・候補生成・整数線形計画による復号・graph修復を維持する。
- context unit: 同一動画の連続3時点を入力文脈とし、教師・出力・単体評価は中央の隣接2時点。集約は中央pairごとに行う。
- 実装区分: `staged-faithful`（このリポジトリ内の管理用語）。初段でも合意した特徴、全候補、MLP、attentionを省略せず、全graphと公式評価だけを進行条件の後へ分ける。単体診断では公式score改善を主張できない。
- この実験が支持または棄却できる主張: 今回の13次元表現と学習量で、固定候補上の両胚の接続診断がexp016より改善するか。
- この実験では判断できない主張: 過去情報全般、長い履歴、過去画像特徴、最終graph、公式score、Public LB、独立CV、hidden test一般化。

## 合意したネットワーク

時点 $t-2$ の各候補を $P$ 、時点 $t-1$ の検出を $A$ 、時点 $t$ の検出を $B$ とする。過去の移動候補は $u=A-P$ 、今回の移動は $v=B-A$ 、移動の変化は $d=v-u$ とする。 $u$ は確定した速度ではない。

| 入力特徴 | 次元 |
| --- | ---: |
| 過去の移動候補 $u$ | 3 |
| 今回の移動 $v$ | 3 |
| 移動の変化 $d$ | 3 |
| 3ベクトルそれぞれの長さ | 3 |
| $u$ と $v$ のcosine | 1 |
| 合計 | 13 |

1. 物理座標はµm単位を使う。変位と長さは5 µmで割り、cosineは無次元とする。特徴をclipしない。ゼロ移動のcosineは0、分母の数値保護は `1e-8` とする。
2. 全組み合わせに共有するMLPは `Linear(13, 32) → GELU → Linear(32, 32) → GELU` とする。両Linearはbiasあり、dropout・正規化層を設けない。出力を32次元の候補特徴 $z_P$ とする。
3. $z_P$ にbiasありの `Linear(32, 1)` を適用してscore $s_P$ を得る。過去情報を使わない選択肢のscore $s_0$ は全pairで共有する学習可能なスカラー、初期値0とし、同選択肢の特徴は常にゼロとする。
4. 全有効候補と過去情報を使わない選択肢にsoftmaxを適用する。

```math
\alpha_P = \frac{\exp(s_P)}{\exp(s_0)+\sum_{Q\in\mathcal{P}}\exp(s_Q)},\qquad
h_{AB} = \sum_{P\in\mathcal{P}}\alpha_P z_P
```

5. $h_{AB}$ にbiasなしの `Linear(32, 1)` を適用し、既存接続logitへ加える。この最終層だけをゼロ初期化する。追加parameterは1,570個である。
6. 過去候補がない場合は過去情報を使わない選択肢の重みを1、集約値と追加出力を厳密に0にする。学習後もbiasによる非ゼロ出力を生じさせない。

同じ $A$ でも $B$ が違えば候補特徴と重みが変わる。attention重みは過去対応の正解確率ではなく、現在の接続を判断する内部重みである。

## 全候補・時点対応の契約

- 各中央pairに、同じ動画の直前時点の全有効候補を保持する。距離上位、半径、1候補への確定、変位の先行平均は行わない。
- 同じframeが複数windowに現れる場合はsample・frame・candidate ID・grid座標・物理座標の整合を検査する。異なる動画・非隣接frameを混ぜず、paddingは実候補に含めない。
- 分割計算でもsoftmax分母は分割をまたいだ全候補と過去情報を使わない選択肢を含む。分割ごとの独立softmaxや分割平均へ置き換えない。
- 全graphで逆向き採点を追加する場合は、走査方向に対してsourceの1時点前を過去候補とする。正逆で同じ式と重みを使い、逆向きへ正向き変位を流用しない。動画端は追加出力0とする。

## 実装方法

- アプローチ: `past_candidate_data.py` がexp015 cacheの中央windowと直前windowを読み、重複時点を検査して全過去候補をpadding mask付きでbatch化する。`past_candidate_attention.py` が13次元特徴、共有MLP、全候補softmax、delta logitを実装する。`attention_train_pipeline.py` が保存済みexp016対照を同じ外側windowで再評価し、新modelだけを学習する。
- target / loss: exp016から継承した `frozen_tracker.py` のgreedy one-to-one対応、edge target、source軸softmax後のfocal BCEを変更しない。
- 全候補softmax: 数値安定なonline log-sum-expで候補chunkをまたいだ分母と重み付き和を累積する。source・target・過去候補の各軸を分割し、学習時は分割ごとにgradient checkpointingで再計算する。候補集合とsoftmaxは変えず、分割なし実装との出力・勾配一致をtestする。
- 初期化: 公開primary tracker checkpointをstrict loadし、delta最終層だけを0初期化する。初期logitがbase trackerと一致することをtestする。
- runtime gate: 64 windowと候補数積の大きいwindowでforward・backward時間とpeak GPU memoryを測定し、保守係数1.5を掛けた予測が12時間を超えればfull 2-fold学習前に停止する。
- 出力: `models/fold_<n>/past_candidate_attention_tracker_best.pth`、fold summary、runtime benchmark、teacher audit、graph progression gate、training summary、model manifest、metrics。
- 再現性: global seed 42、fold offset、DataLoader generator・worker seedを固定する。cache summary・identity、annotation、feature schema、model、外側pair予測のSHAを記録する。GPUのbitwise一致は主張しない。

## 探索幅とpivot判定

- 変更class: `add-only`（このリポジトリ内の管理用語）。target、output、loss、decodeを変えず、全過去候補の座標集約とdelta branchを追加する。
- 同じ親または機構familyで連続した小改善実験数: exp035は予測履歴、今回は対応未確定の全候補集約であり、入力生成と集約単位が異なる。exp016からの同一parameter調整の3件目ではない。
- positiveなoracle headroomまたはcoverage: exp015の候補edge回収率とexp016の固定候補診断に改善余地はあるが、全graph改善を保証しない。exp035の失敗は予測履歴を1本へ確定する表現の結果であり、本表現の結果ではない。
- target、output、decode、context unitを変える別案: division triplet出力、graph上のedge学習、長い位置列、tracklet接続は別候補とし、この実験へ混ぜない。
- `kaggle-idea-forge` の要否: 不要。ユーザーが設計済みbacklog候補を指定して実装を承認しており、同一parameter調整の3件目ではない。

## 再現性・リスク

- seed policy: global seed 42、fold offset、DataLoader generatorとworker seedを固定する。
- stochastic処理: DataLoader shuffle、tracker dropout、CUDA kernel。augmentationとseed baggingは使わない。
- feature生成: exp015の固定cacheから決定的に読み、中央windowと直前windowのcandidate ID・grid座標・物理座標を検査する。候補順序不変性とchunk分割一致をtestする。
- modelと予測の証拠: cache summary・identity、annotation、feature schema、fold別model、外側pair予測、model manifestのSHAを記録する。GPUのbitwise一致は主張しない。
- leakage: 正解接続は教師以外に使わない。公開初期weightの学習来歴による条件付きvalidationの制約は残る。外側胚で設定を選ばない。
- runtimeとmemory: 計算量は3時点の候補数積に比例する。64 windowと大きい候補積でbenchmarkする。source・target・過去候補の3軸chunkとgradient checkpointingでも12時間gateまたはOOMに達する場合は、候補集合・fold・epochを縮小せず停止する。
- hidden test: 特徴は候補座標だけから作れるが、全graphの逆向き走査、動画端、可変候補数は初段の単体診断では未検証である。
- CVとLB: 単体診断は公式metricではなく、Public LBまたはhidden testの改善を意味しない。

## 受け入れ基準

- [x] backlogの上位仮説ID、検証範囲、残る検証、根拠、差分、固定事項、成功条件、停止条件、実行しないこと、判断履歴を移行した。
- [x] `config.yaml` のlineage、1構成・2fold・3 epoch・model 2個・control再学習なしが本書と一致する。
- [x] 13次元特徴、1,570 parameter、全候補attention、過去情報を使わない選択肢、zero-initialized deltaを実装した。
- [x] 直前windowの全候補読み込みとcandidate ID・座標対応検査を実装した。
- [x] 候補順序不変性、padding、空集合、動画端、分割出力・勾配、初期出力一致を対象とするtestを追加した。
- [x] Jupytext round-trip、`validate-exp`、`check-exp`、`test-exp`を通す。
- [x] Kaggle push前にGPU quota、active variant 1、model 2、control再学習なしを確認する。2026-09-22にユーザー承認を受け、GPU残り16.18時間で実行可能と判断した。
- [x] Kaggle version 1のOOMを記録し、候補集合と学習条件を変えないmemory修正後のversion 2でruntime benchmarkを完了した。
- [x] 198.17時間の保守的予測が12時間gateを超えたため、本学習前に停止した。
- [ ] 実験の採用・不採用・完了は結果提示後のユーザー判断まで確定しない。
