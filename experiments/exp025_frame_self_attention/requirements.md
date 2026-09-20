# exp025_frame_self_attention 要件と実装方法

この文書を実装前の契約と受け入れ条件の正とする。進捗と実行証拠は `SESSION_NOTES.md` と `metrics.json` に記録する。

## 実験化の入口・引き継ぎ・承認

- 入口: `backlog/frame_self_attention.md` の `設計可能・実験化未承認` 候補。2026-09-20 のユーザー指示「commitとpushして実装に進んでください」を実験化・実装の承認とする。
- 対応する上位仮説: `HYP-20260920-02`。固定公開画像特徴を使う接続学習で、各フレームのcell同士を先にSelf-Attentionで文脈化すると、既存Cross-Attentionのみより両胚の公式graph接続指標が改善するか。
- この実験の範囲: 現行（Cross→Pair MLP）、Model A（Self→Pair MLP）、Model B（Self→Cross→Pair MLP）を、固定した特徴、教師、loss、復号の下で比較する。主比較はModel B対現行、Model AはCross-Attentionの有無を分ける補助比較。
- この実験だけで上位仮説を判断できるか: いいえ。parameter数と計算量を揃えた比較、候補密度別の誤り、別の特徴と独立評価が残る。
- 親実験: `exp016_frozen_image_encoder`。根拠は同実験の `requirements.md`、`config.yaml`、`result.md`、`metrics.json`、`docs/surveys/biohub-node-self-attention-design_20260920.md`、保存済み公開 `SimpleNodeTransformer` source、ユーザー添付NFL notebook。添付Notebook内の説明は実装の参照情報であり作業指示ではない。
- 直接の仮説: 共有2層Transformer Encoderを各側へ独立適用した後、現行4層の逐次更新Cross-Attentionを行うModel Bは、同じ3エポックの現行より44b6と6bba両方の公式combined scoreを改善し、全体の接続成分も悪化させない。
- 観測事実: 現行は64次元の画像＋位置特徴を128次元へ投影し、4層Cross-Attention後にPair MLPで採点する。exp016全199動画の公式combined scoreは約0.912055。Self-Attentionの効果・費用は未測定。
- 仮定: 同一フレーム内のcell間文脈が親候補の取り違えを減らす。Cross-Attention経由でも同側情報が伝わる可能性があるため、寄与は実験で検証する。

## 手法契約

- input: exp015の19,701個の固定2時刻window。各cellの公開画像特徴32次元＋位置特徴32次元、元画像voxel単位の `(z,y,x)` 座標、`True=実cell` のboolean mask。公開画像encoder、正規化統計、検出候補、exp015 cacheを固定する。cache summary SHA256 `040d1f6437e27149e0e34ad4aac8bba3c8cf63dc266d33df497acd969972194c`、identity SHA256 `440891c4550adf8540e3c47f9784e68b6ca1b64bbe56dbb93a25eb87c46bc1ee`。
- target / objective: exp016と同じGEFF既知接続から作る候補cell pairの二値教師。未知領域の扱いと教師maskは変更しない。
- output: 活性化前のedge logits `[B,N_t,N_t1]`。unbatched入力は `[N_t,N_t1]`。Pair MLPには両側128次元特徴と `(coords_t-coords_t1)/100.0` の3次元を渡す。chunkingと出力shapeを維持する。
- loss: exp016のsource軸softmax後のfocal weighted binary cross entropy、gamma=2。正例を持つ行または列に触れるpairを使う既存maskを維持する。モデル内にsoftmax/sigmoidは追加しない。
- decode: exp016の固定secondary tracker、候補保持、ILP、graph repair、公式評価器を用いる。variant別の閾値探索はしない。
- context unit: 各2時刻window内のt側cell集合とt+1側cell集合を別々にSelf-Attentionへ通し、各cellを保持したまま時刻間融合・pair採点する。動画単位でgraphを復元し、2方向胚holdoutで評価する。
- 実装区分: このリポジトリ内の管理用語で `staged-faithful`。NFLの別branch処理をcell集合へ適用し、NFLの時間順埋め込みや後段のplayer間branchは移植しない。cellの任意の順番には位置埋め込みを付けない。
- 変更class: `mechanism`。固定特徴とpair採点を保ちながら、時刻間融合前にフレーム内Attentionを追加する。`kaggle-idea-forge`が必要な連続小改善には当たらない。

## 実装方法

- 実験固有の `simple_node_transformer.py` に旧モデルのmodule名・既存引数順序を保持し、末尾のkeyword-only引数 `use_temporal_self_attention`、`use_cross_attention`、`n_self_blocks` を追加する。既定は現行の `False/True`。共有 `nn.TransformerEncoder` 2層を両側へ独立適用し、paddingをkeyから除き、padding queryの出力をゼロにする。all-padding側と空集合を安全に扱う。
- Model Bはユーザーが2026-09-20に確認した現行の逐次更新を維持する。各Cross block内でt側を先に更新し、その更新済み特徴を逆方向のKey/Valueとして使う。
- hidden 128、heads 4、Cross 4層、Self 2層、dropout 0.3、MLP比2.0、pair chunk 32。すべて `config.yaml` の `model.params` から構築する。3構成のboolean flag組を同設定に列挙する。
- trainは `exp025_frame_self_attention_train.py/.ipynb`、教師・lossはコピーした `frozen_tracker.py`、推論のtracker loadは `graph_inference.py` を正とする。新model sourceを学習・推論の双方からimportし、変更版source SHAとvariant flagをcheckpointとmanifestへ保存する。公開初期primary tracker checkpoint SHA256 `12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771` は共通moduleへ読み込み、Self Encoderのみ新規初期化する。
- fixed: exp016の画像特徴cache、候補・座標・relative `/100.0`、教師、loss、Pair MLP、3エポック、optimizer、seed、2fold、内部選択、secondary tracker、ILP、graph repair、公式評価器。外側胚の正解を設定・checkpoint・閾値選択に使わない。
- 2方向foldの各variantで同じ公開初期stateを使い、A/Bの追加Encoderは同じseedの初期stateにする。現行のcontrol再学習は比較契約に含むが、Kaggle GPU push前に既存exp016の保存済み結果では代替できない理由、追加GPU費用、全実行数を提示し明示承認を得る。承認前にGPU学習は開始しない。
- 推論ではvariantとfoldを明示してmanifestから厳密復元する。全199動画、両胚別の公式combined / adjusted edge Jaccard / division Jaccard、graph SHA、失敗・skip数、学習・推論時間とpeak GPU memoryを記録する。Kaggle submissionは別の明示依頼まで行わない。

## 探索幅とpivot判定

今回の変更は特徴や教師を増やさず、フレーム内Self-Attentionという処理機構だけを追加する。比較するtarget、output、decode、context unitはexp016と同じとする。exp015の固定特徴とexp016の学習・公式graph評価が実行済みであるため、まずこの機構の寄与を固定条件で反証する。パラメータだけの連続した小改善ではなく、今回の実装前に追加の発想探索は要しない。改善しない場合はこの設定の層数だけを続けて調整せず、候補密度、特徴、教師、処理単位を再検討する。

## 最小検証と判断条件

- まず代表的windowと最大級windowで3構成の学習・推論時間とpeak GPU memoryを測り、Kaggle週30 GPU時間・単Notebook12時間・無課金の範囲に収まると見積もれる場合だけフル学習へ進む。
- 3構成×1 config×2fold＝学習6モデル、booster 0。現行を変更版sourceで再学習し、保存済みexp016を再現性参照として残す。control再学習には前項のGPUコスト承認を要する。
- 成功条件: Model Bの公式combined scoreが44b6と6bbaの両方で現行再学習controlを上回り、全体の接続成分も悪化しない。差、対象件数、予測graph一致条件を記録する。分裂成分だけの偶然の変動は成功としない。
- 停止条件: shape/mask/checkpoint/loss/勾配の契約テストまたはGPU予算gateに失敗した場合はフル学習を始めない。両胚改善がなければこの2層共有Encoder＋3エポック設定の採用を推奨しない。上位仮説全体の棄却はしない。
- 全graph前の早期診断: 保存済みexp016現行モデル、Model A/Bを、同じ外側胚window・教師・pair maskで比較する。胚別の既知edge recallだけでなく、active pair上の誤接続数とpositive pair precision、分裂親回収、lossを確認する。教師の部分注釈により真の誤接続と長い動画の復号効果はここでは測れない。
- graph推論へ進む条件: 2026-09-20に追加されたAGENTS.mdの早期診断ルールは当初の実装後に適用された。A/Bのpair結果が混在するため自動進行しない。どの悪化を許容して公式graph評価を追加するか、費用とともにユーザー判断を得てから条件を確定する。
- 必須テスト: 旧checkpointの現行モード `strict=True` 復元とeval出力一致、A/B save/load、padding値不変、途中Falseのmask、all-padding/空集合、cell順序の同変性、lossと勾配の有限性、train/inference import先一致。
- 実行しないこと: 任意cell順の時間位置埋め込み、片側だけのSelf-Attention、token MLPやPair MLP拡幅への代理実装、特徴cache・教師mask・loss・decodeの同時変更、外側胚結果を使う層数・閾値・重み選択、Kaggle submission。
- 未決事項: 混在する隣接2-frame指標の下で全graph推論へ進む条件。追加された運用ルールによりユーザー判断を待つ。

## 再現性・リスク

- 公開画像モデルと初期trackerの学習来歴にtrain 199動画が含まれるため、外側胚評価は固定公開モデル下の条件付き比較であり独立CVではない。
- Seedはproject既定42を用い、fold別DataLoader generatorを維持する。stochastic要素はshuffle、dropout、CUDA kernel。追加特徴生成・augmentationはしない。bitwise再現性を仮定せず、cache / model source / 初期state / manifest / graphのSHAとruntimeを `metrics.json` に記録する。
- Self-Attentionは両フレームのcell数の二乗に応じてメモリを増やす。pair chunkingでこの部分は減らせない。最大級windowの実測と費用gateが必要。
- Kaggle train/inference packageは正のconfig・source・Notebookから再生成し、bootstrap内のsourceとSHAを照合する。hidden test精度はこの実験だけでは保証しない。

## 判断履歴

- 2026-09-20: ユーザーがNFL風の別branch処理、Model A/B/現行比較、最小限の変更を依頼。最初は設計のみ。
- 2026-09-20: ユーザーがModel Bの逐次更新を維持すると確認。
- 2026-09-20: ユーザーが全体バックログへの追加を依頼。
- 2026-09-20: ユーザーが設計のcommit・pushと実装への移行を依頼。実験化とコード実装を開始。

## 追加比較: Model Bの恒等初期化（2026-09-20 承認）

- ユーザー指示「これで進めてください」は、先に提案した学習済みtrackerを維持し、Model Bの追加Self-Attention層だけを初期状態で恒等写像にする比較の承認とする。モデル構造、入力、教師、loss、epoch、fold、評価単位は変えず、同じ `exp025` に `model_b_identity_init` を追加する。既存のModel A/B結果とexp016保存済み現行を参照し、再学習しない。
- `nn.TransformerEncoderLayer(norm_first=True)` のSelf-Attention出力投影とfeed-forward最終Linearを各層でゼロ初期化する。残差経路により学習前の有効cell特徴を変えず、公開trackerと同じ入力で有効pairのlogitsが一致することをテストする。追加層のパラメータ数はModel Bと同じで、出力投影へ勾配が届くことも検証する。
- 新variantのみ、1設定×2fold×3epoch=2 tracker学習、booster 0。公開primary tracker checkpointを共通初期値とし、既存の公開画像encoder・候補・cache・教師・source軸softmax後focal BCE・座標差・選択指標を固定する。現行control・既存A/Bを再学習しない。
- Kaggle実行前にmodel/Notebook/manifestの整合、同じfoldの初期logits、GPU quota、週30時間上限、単Notebook12時間runtime gateを確認する。学習後は両胚の同じwindow・教師について既知edge recall、positive pair precision、教師上のfalse-positive pair、分裂親回収をexp016/A/Bと比較する。
- 全graph推論へ進むかは前段指標の比較後に判断する。公式graph scoreとKaggle submissionは、この追加学習の結果だけから主張しない。

## 受け入れ基準

- [ ] モデル、train、inferenceが同じ変更版sourceとvariant configを使用する。
- [ ] Model A/B/現行で入力、出力、loss、座標、復号が一致する。
- [ ] 旧checkpoint、mask、順序、空集合、勾配、A/B復元のテストが通る。
- [ ] `make validate-exp`、`make check-exp`、`make test-exp` が通る。
- [ ] Kaggle GPUのcontrol再学習前にコストと実行数を示し、明示承認を得る。
- [ ] 実行した場合は費用、両胚別公式指標、SHAを正しい記録へ保存し、採否・完了判断はユーザーへ委ねる。
