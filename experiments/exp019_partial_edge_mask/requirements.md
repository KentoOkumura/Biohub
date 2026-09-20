# exp019_partial_edge_mask 要件と実装方法

## 実験化の入口・引き継ぎ・承認

- 2026-09-19のユーザー依頼「partial_edge_maskを実装してください」により、設計可能・未決事項なしの `backlog/partial_edge_mask.md` を実験化する。
- 上位仮説: `HYP-20260910-01`。部分注釈の未記録edgeを負例として抑える損失を減らすと、公式graph指標が改善するかを検証する。この一変更だけでは検出教師、過検出、他のmask方式まで判断できない。
- 親と主対照: `exp016_frozen_image_encoder` の保存済みfold別tracker、外側胚window指標、公式graph評価。追加の対照学習は行わない。公開trackerとexp015保存graphは入力および候補同一性の確認に使う。
- 根拠: 元候補 `backlog/partial_edge_mask.md`、exp016の `requirements.md`、`config.yaml`、`metrics.json`、exp015のtrain cache、exp011で選定した公開source。exp016の公開画像encoderは両評価胚を学習した可能性があり、trackerのfold分割を独立CVとは呼ばない。
- 未決事項: なし。実行時のGPU残量、Kaggle生成物の所在、未知親の頻度、時間とメモリ、精度差は実測事項とする。

## 手法契約

- input: exp015が保存した19,701 window、199動画の候補ID・座標・固定画像特徴と、主催者GEFFの部分的なnode・edge注釈。両frameにGT nodeがあるwindowだけをexp016と同様に扱う。
- target / objective: exp016と同じ5 µmの距離順greedy一対一対応から、隣接frameの既知edge行列を作る。分裂は同じ親行から複数の子列への正例とする。
- output: 候補親子pairの接続logit。モデルは公開checkpointから初期化したprimary `SimpleNodeTransformer`のみを各胚方向で学習する。
- loss: source軸softmax、focal係数2のbinary cross entropy、分裂重み1.0は固定する。正例を1本以上持つ**子列**のpairだけをlossへ含める。正例のある親行でも、正例のない子列のpairは含めない。全親候補はsoftmaxの分母に残すため、未知親への間接勾配は残り得る。
- decode: exp016と同じ固定候補、secondary contribution、双方向association、ILP、graph repair、DeepCenter、公式評価器を使う。main画像encoderと検出headは再実行しない。
- context unit: modelとlossは隣接2-frame window。外側評価は胚単位で2方向。最終graph評価は199動画を両胚別に報告する。
- 実装区分: `staged-faithful`（このリポジトリの管理用語）。固定画像特徴のtracker学習段階から始め、保存済みcacheで固定graph推論と公式評価へ進む。検出lossや画像encoder更新は含めない。
- 変更class: `mechanism`。教師の正例行列とモデル構造を保ち、pair loss maskだけを変更する。
- 比較の限界: 正例子列に含まれる未知親は分母と負例項に残る。未知子への直接損失を除く効果だけを分離し、softmax分母変更の効果は検証しない。

## 実装方法

- 固定: exp016のcache SHA、公開sourceとcheckpoint SHA、教師対応、正例行列、fold・internal split、seed、3 epochs、optimizer、checkpoint選択指標、推論候補、復号設定、評価器。
- 変更: `frozen_tracker.py`の学習lossを正例子列maskへ変更し、legacy maskのpair数と除去数、未知親を含むpair数を教師監査へ追加する。評価では新旧lossをともに記録し、接続精度とcheckpoint選択指標はexp016と同じmaskで計算する。
- `exp019_partial_edge_mask_train.py`は固定入力のSHAとcoverage、empty-GT除外、学習対象、64 window/fold benchmarkによる12時間gateを検査して2foldの変更variantだけを学習する。fold別model、教師監査、manifest、metricsを保存する。
- `exp019_partial_edge_mask_inference.py`は変更variantの保存済み2foldモデルを読み、固定cacheからtrackerとgraphを再生する。公開trackerの候補graphがexp015保存graphと全件一致することを要求し、同じ公式評価器で採点する。主対照のexp016保存済み公式graph指標を比較に使う。
- 再現性: global seed 42、fold offset、DataLoader generatorとworker seedを固定する。cache、source、checkpoint、model、graph、kernel versionのSHAを記録する。CUDAのbitwise一致は主張しない。

## 再現性・リスク

- 人工例: 片娘のみ注釈、別親の既知edge、複数候補が同一GT近傍にある場合を含め、正例行列不変、正例子列だけが直接の損失項を持つこと、未知親にはsoftmax分母から間接勾配が残ること、zero-positive windowで勾配が有限であることを確かめる。
- 学習側: mask以外を固定し、2方向の胚入替で1モデルずつ学習する。active variant 1、model/config 1、fold 2、booster 0、control再学習0。64 window/foldの実測から1.5倍係数で12時間gateを判定する。
- primary指標: 公式combined score。接続・分裂成分、両胚別、失敗数、有効件数、train-sideの既知edge指標を併記する。exp016と同じ候補・復号・対象動画で両胚のcombined scoreが改善し、未知娘への直接負例項が減った場合に精度仮説の支持とする。
- 停止: 入力SHAやcoverageの不一致、fold外GEFFの学習・選択への混入、optimizerの対象外更新、非有限loss、OOM、12時間gate超過、公開tracker候補graphの不一致、公式評価器の不一致で停止する。方式、fold、epoch、候補上限を黙って変更しない。
- 実行しないこと: softmax分母変更、正例対応変更、検出器・画像encoder・secondary trackerの再学習、閾値調整、外側胚による方式選択、submission、Public LB取得。
- 採否と完了: 結果と限界を提示し、ユーザー判断まで確定しない。

## 探索幅とpivot判定

- この親経路でのmaskだけの比較は今回が1件目。exp016は固定特徴からtracker学習を成立させる基準であり、mask変更をしていない。現時点では追加の小変更を自動で連続させない。
- 失敗時にsoftmax分母、target、decodeを変更して同一実験を救済しない。候補の別方式を検討する場合は結果と未解決事項を提示してから設計する。

## 受け入れ基準

- [ ] backlogの上位仮説ID、候補名、固定・変更事項、根拠、判断履歴がrequirementsとconfigへ移行している。
- [ ] 人工例で正例子列mask、同一GT近傍の一意対応、未知親の間接勾配、zero-positive lossを確認する。
- [ ] Jupytext sourceとNotebookが一致し、validate-exp、check-exp、test-expを通過する。
- [ ] Kaggle実行前にactive variant 1、2fold、model 2個、booster 0、control再学習なしとquotaを記録する。
- [ ] Kaggleで固定cacheのSHAとcoverage、教師監査、12時間gate、2foldの新tracker学習を確認する。
- [ ] 保存済みexp016の公式graph結果と同じ公開候補・復号・評価器で、新trackerの199動画の公式graph評価を行う。
- [ ] 両胚の成分指標、失敗件数、有効件数、費用、残る未知親の勾配を結果として報告する。
- [ ] 実験の採否と完了はユーザー判断まで確定しない。

## 判断履歴

- 2026-09-10: 未記録の第2娘を負例とする現行maskを確認。子列限定とsoftmax分母変更を別比較にする案を記録した。
- 2026-09-12: 公開検出器固定・下流tracker学習の方針を適用し、正例子列maskの比較を設計可能とした。
- 2026-09-14: exp015のcacheと段階別診断、exp016の再学習tracker・公式graph評価を先行条件にした。
- 2026-09-19: ユーザーが実装を指示。exp016を親にして、loss maskだけを変更する契約で開始した。

## 移行元の観測事実と残る問い

- 主催者GEFFには133,318個の注釈nodeがあり、推定総nodeの2.82%に相当する。2胚で151分裂を記録するが、完全annotation maskはない。これは疎い教師への懸念であり、このmaskの改善値ではない。
- exp015の199動画・19,701 windowの固定cacheで、既知edgeの両端対応は98.843%、candidate graphでの回収は94.829%、final graphでの回収は91.602%だった。cache、公開checkpoint、公開sourceのSHAはconfigと親実験のmetricsを正とする。
- 実候補での一意対応から外れた同一GT近傍候補、未知親が正例子列に残るpair、旧maskから除去されるpairの頻度をteacher auditで測る。どの値も改善の事前証明として扱わない。
- 負例の保証、検出側の過検出、胚ごとの効果、softmax分母の変更は残る検証である。検出損失のmask変更や密な領域教師は本実験へ混ぜない。
- hidden testのGEFFや正解由来の細胞数を推論へ使わず、GTで決めた候補上限をモデル改善と呼ばない。推論入力は画像と利用可能なmetadata、保存済みmodelのみとする。
- GPUは週30時間以内、課金なし。学習と推論の費用を分け、非公開test全件の12時間内完走は未確認として扱う。
