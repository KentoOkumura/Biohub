# frame_self_attention

- 候補名: `frame_self_attention`
- 状態: `設計可能・実験化未承認`
- 対応する上位仮説: `HYP-20260920-02`
- 関連する上位仮説: `HYP-20260910-12`（固定画像特徴cacheの再利用）
- 作成日: 2026-09-20
- 最終更新日: 2026-09-20
- 依頼原文: 「現在のCell Trackingモデル `simple_node_transformer.py` に、NFL trajectory model の `TemporalTransformerBranch` に近い構成を取り込んでください。まずは設計のみです。」「全体のバックログにも反映されていますか？」「追加してください」
- 期待する成果: 固定した公開画像特徴からの接続学習で、各フレームのcell集合をSelf-Attentionで文脈化すると、現行のCross-Attentionのみのtrackerより、両胚の公式graph接続指標が改善するか検証する。Model A（Self→Pair MLP）とModel B（Self→Cross→Pair MLP）を現行と比較する。
- 親実験 / 比較対象: [exp016の実験契約](../experiments/exp016_frozen_image_encoder/requirements.md)、[設定](../experiments/exp016_frozen_image_encoder/config.yaml)、[結果](../experiments/exp016_frozen_image_encoder/result.md)、[数値](../experiments/exp016_frozen_image_encoder/metrics.json)。現行Cross-Attentionのみを主対照とする。[モデルと比較の設計](../docs/surveys/biohub-node-self-attention-design_20260920.md)を直接の実装根拠とする。
- 優先度: P2
- 優先度の理由: 固定特徴・教師・loss・復号のままtrackerの表現だけを変える反証可能な比較。既存の高優先度候補に続く比較とする。Self-Attentionで計算量が増えるため、GPU週30時間と提出推論12時間の制約に対してKaggleで小規模実測を先に行う。
- `backlog/KAGGLE_DIRECTION.md` の対応箇所: [検証中の仮説](KAGGLE_DIRECTION.md#検証中の仮説)と[未着手バックログ](KAGGLE_DIRECTION.md#未着手バックログ)
- 先行条件 / 依存: exp016の公開初期tracker checkpoint、exp015のtrain feature cache、exp016の2方向胚holdout・固定graph評価経路。3エポック固定の本候補は、他候補の実行結果を必須入力としない。

## 観測事実と根拠

- 実測済みの事実: 現行`SimpleNodeTransformer`は両時刻の64次元特徴を128次元へ投影し、4層の双方向Cross-Attentionの後に、両側特徴と3次元相対位置をPair MLPへ渡す。各フレーム内の明示的なSelf-Attentionはない。exp016は固定特徴からprimary trackerだけを再学習し、公式combined scoreは199動画全体で約0.912055。Self-Attention追加による精度差・学習費用は未測定。
- 根拠ファイル / 一次資料: [静的解析と具体設計](../docs/surveys/biohub-node-self-attention-design_20260920.md)、[保存済み公開source](../experiments/exp001_temporal_unet3d_baseline/official_source/src/tracking_cellmot/models/simple_node_transformer.py)、[exp016の実行数値](../experiments/exp016_frozen_image_encoder/metrics.json)、[現行方針](KAGGLE_DIRECTION.md#今後の学習方針)。NFL Notebookはユーザー添付の`/mnt/c/Users/kento/Downloads/nfl-2026-exp138-infer.ipynb`であり、上記設計書は該当コードを実行せずに照合した。
- 利用する保存済み生成物とSHA: exp015のtrain feature cacheは19,701 window、summary SHA256 `040d1f6437e27149e0e34ad4aac8bba3c8cf63dc266d33df497acd969972194c`、identity SHA256 `440891c4550adf8540e3c47f9784e68b6ca1b64bbe56dbb93a25eb87c46bc1ee`。公開初期primary tracker checkpoint SHA256 `12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771`。保存済みexp016のfold別trackerとgraph評価のSHA・所在は同実験の`metrics.json`を正とし、実験化時に照合する。
- 仮定: Assumption: 同じフレームの複数cellから直接得る文脈が、近傍での親候補の取り違えを減らす。現行Cross-Attentionでも他方の時刻を経由して同側cellの情報が伝わり得るため、独立した寄与は未実証。

## この候補が直接検証する仮説と範囲

- 上位仮説のうちこの候補が検証する範囲: exp016と同じ固定候補・2時刻特徴・教師・loss・graph復号の下で、Pair MLP前に同一フレーム内のSelf-Attentionを足す効果を検証する。主比較はModel B対現行、Model AはCross-Attentionを外したときの補助比較。
- この候補の具体的な仮説: 共有2層Transformer Encoderでt/t+1を別々に文脈化してから既存4層の逐次更新Cross-Attentionを実行したModel Bは、同じ3エポックの現行モデルより両胚の公式combined scoreを改善する。
- 仮説が正しい場合に期待する観測: 同じ候補・復号・3エポック・内部選択の下で、Model Bの公式combined scoreが44b6と6bbaの両方で現行を上回り、全体の接続成分も悪化しない。
- 仮説を棄却する観測: Model Bの両胚改善が得られない、または接続成分の改善がなく分裂成分の偶然の変動だけでcombined scoreが上がる。Model Aの結果と併せてCross-Attentionを保持する価値を解釈する。
- この候補だけで上位仮説を判断できるか: いいえ
- 上位仮説の判断に残る検証: Self-Attentionの層数・parameter数・計算量を揃えた比較、候補密度別の誤り、別の画像特徴や独立した評価での再現性。今回の固定条件で失敗しても、すべてのSelf-Attention設計を棄却しない。

## 入力・予測対象・出力・推論方法

- input: exp015の各2時刻windowにある固定公開検出候補、画像特徴32次元、位置特徴32次元、`(z,y,x)`座標、各側の実cell/padding mask。画像encoderと正規化統計は固定する。
- target / objective: exp016と同じGEFFの既知接続に対応付けた候補cell pairの二値教師。教師の未知領域を新たな負例とは見なさない。
- output: `[B,N_t,N_t1]`の活性化前のedge logits。unbatchedでは`[N_t,N_t1]`。Pair MLP入力は両側hidden特徴と既存の3次元相対位置。
- loss: exp016と同じ、親cell軸softmaxの後のfocal weighted binary cross entropy。正例行または正例列に触れるpairを対象とする既存教師mask、gamma 2を維持する。
- decode / 推論方法: exp016と同じ固定secondary tracker、候補保持、ILP、graph repair、公式評価器を使う。Model A/Bと現行で閾値を別々に調整しない。
- 処理単位: 各2時刻window内でtのcell集合とt+1のcell集合を別々にSelf-Attentionへ通し、その後cell pairを採点。動画単位でgraphを復元し、2方向の胚holdoutで評価。
- 実装区分: このリポジトリ内の管理用語では`staged-faithful`。NFL Notebookの別branch→融合という構成をcell集合へ適応し、既存の入力、pair採点、loss、復号を保つ。NFLの時間軸位置埋め込みや後段の選手間branchをそのまま移植する案ではない。

## 親実験からの差分

- 変更するもの: `SimpleNodeTransformer`へ共有`nn.TransformerEncoder`によるフレーム内Self-Attentionを追加し、`model.params`の`use_temporal_self_attention`と`use_cross_attention`でModel A/B/現行を切り替える。`n_self_blocks=2`、hidden 128、heads 4、dropout 0.3、MLP比率2.0を初回条件とする。Model Bの逆方向Cross-Attentionはユーザー確認済みの現行逐次更新を維持する。
- 固定するもの: 公開検出器・画像encoderと正規化統計、exp015 cache、座標と相対位置の`/100.0`、教師とloss、2方向胚holdoutと内部選択、3エポック、optimizerとseed、Pair MLPとchunking、候補生成、secondary tracker、decodeと公式評価器。
- 再利用するコード / config / 生成物: exp016のtrain・graph inference・model manifest手順とexp015 cache、公開初期tracker checkpoint。実装時は元の`official_source`と取得artifactを変更せず、新しい実験が変更版modelを学習・推論で同じようにimportする。
- 新しく作るもの: 実験化承認後に、新規model source、3構成のconfig、2foldのcheckpointとmanifest、固定graph評価、GPU費用測定、shape・mask・checkpoint互換性テスト。バックログ化の時点では作らない。

## 最小の反証可能な検証

- 検証方法: まずKaggleで代表的なwindowと最大級のwindowを使い、3構成の学習・推論時間とpeak GPU memoryを測る。予算に収まれば同じexp016の学習・内部選択・外側胚評価を3構成×2foldで実行し、199動画の両胚別公式graph指標を比較する。外側胚の正解を設定・重み選択に使わない。
- variant / config / fold / booster数: 現行、Model A、Model Bの3構成、各1 config、2fold、booster 0。A/Bの追加Encoderは同じ初期stateとし、全構成の共通moduleを同じ公開checkpointから初期化する。層数探索は含めない。
- control再学習: あり。現行を同じ変更版source・cache・3エポック条件で再学習し、保存済みexp016結果を再現性の参照値として残す。これによりA/Bとの差に実行条件差を混ぜない。
- 想定runtime / resource: 3構成×2foldの実測所要時間は未取得。Kaggle Notebookのみ、GPU週30時間以内・課金なし、提出推論12時間以内の制約を適用する。小規模benchmarkで全工程を見積もり、収まらなければフル学習前に止めてユーザーへ費用と代替実行範囲を示す。

## 成功条件と停止条件

- primary指標: exp016と同じ公式combined score（adjusted edge Jaccardとdivision Jaccardの加重和）、全体および44b6・6bba別。
- 成功条件: 同じ固定候補・教師・復号の現行再学習対照より、Model Bの公式combined scoreが44b6と6bbaの両方で上がり、全体の接続成分も悪化しない。差・有効件数・予測graphの一致条件を記録する。閾値探索で事後的に改善を作らない。
- 必須guard: 旧checkpointの現行モードへの`strict=True`読み込みとeval出力一致、A/Bの保存・復元、maskと空集合、padding不変性、cell順序の同変性、lossと勾配の有限性、学習・推論import先一致。両胚別の接続・分裂成分、実行時間、peak GPU memory、失敗・skip件数を記録する。
- 成功時の次段階: 指標・計算費用・公開基準との差をユーザーへ提示し、実験の採否・完了判断を求める。submissionは別の明示依頼まで行わない。
- 失敗時の停止範囲: 設計契約のテストまたはGPU予算gateを通らなければフル学習へ進まない。公式評価で両胚改善がなければ、この2層共有Encoder＋3エポック条件の採用を推奨しない。上位仮説全体の結論はユーザー判断へ残す。

## 実行しないこと

- 禁止する代替実装、proxy、同一OOF上の救済探索: cellの任意のindexへNFLの時間位置埋め込みを入れない。Self-Attentionの代わりにtoken単位MLPやpair MLPの拡幅へ置換しない。片側だけ文脈化しない。特徴cache、教師mask、loss、decode、候補数を同時変更しない。外側胚の結果で層数・閾値・重みを選ばない。
- 壁打ちで採らなかった案と理由: 初回からt/t+1別々のEncoder重みを持つ案は、同じ64次元特徴を持つ現行共有投影からの変更点を増やすため採らない。Cross-Attentionの削除を決め打ちする案は、Model AとBを対照に含めるため採らない。更新前のt特徴を逆方向へ渡す同時更新は、現行との比較が変わるため採らない。

## リスク

- leakage / validation: 公開画像モデルと初期trackerの学習来歴から、胚holdoutは固定公開モデル下の条件付き比較であり、独立CVではない。外側胚の正解でcheckpointや設定を選ばない。
- hidden test: 実装を学習・推論双方で同じsource・config・manifestに結び、公開test固有のIDや生成物に依存させない。実際のhidden test精度はこの候補では保証しない。
- runtime / memory: Self-Attentionはcell数の二乗に応じる計算を両フレームで追加する。Pair MLPのchunkingではこの部分のメモリを減らせない。Kaggleで最大級windowを測り、週30時間と12時間gateを確認する。
- 再現性: 現行moduleのstate keyを維持し、新規Encoderの初期化だけを別管理する。checkpointには有効flag、層数、source・初期重み・cacheのSHAを記録し、A/B推論では`strict=True`復元と構成一致を要求する。

## 調査・実行時に確認する事項

- 各構成のKaggle学習・推論時間とpeak GPU memory、モデルの全parameter数と実際にforwardへ使うparameter数、両胚・全体の公式指標、model/graph SHA。未実測だけを理由に設計不可とはしない。

## 未決事項

- なし

## 判断履歴

- 2026-09-20: ユーザーがNFLの別branch構造を参考にした設計と、Selfのみ／Self＋Cross／現行の3構成比較を依頼した。実装前の設計に限定した。
- 2026-09-20: Model Bは更新済みt側を逆方向のKey/Valueに用いる現行の逐次更新を維持する、とユーザーが回答した。
- 2026-09-20: ユーザーが完成した設計を全体バックログへ追加するよう依頼した。これはバックログ化の承認であり、実験化・実装・Kaggle実行・submissionの依頼ではない。

## 次セッションへの引き継ぎ確認

- 固定するものを一意に説明できる: exp016の固定画像特徴、3エポック、教師、loss、2fold、graph復号を上記で指定した。
- 変更するものを一意に説明できる: 共有Self-Attentionを2層追加し、Cross-Attentionの有無でA/B/現行を分ける。
- 最小検証と停止条件を一意に説明できる: Kaggle費用gate後、3構成×2foldの公式指標を両胚で比較する。
- 実行しないことを一意に説明できる: 任意cell順の時間埋め込み、教師・特徴・decodeの同時変更、外側胚での設定選択を除外した。
- 未決事項が明示されている: なし。費用と効果は実行時に測定する。
