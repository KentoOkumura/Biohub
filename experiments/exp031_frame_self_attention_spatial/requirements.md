# exp031_frame_self_attention_spatial 要件と実装方法

この文書を実装契約、実装方法、受け入れ条件の正とする。進捗は `SESSION_NOTES.md`、測定値と実験statusは `metrics.json` に記録する。

## 実験化の入口・引き継ぎ・承認

- 実験化の入口と承認: 2026-09-20のユーザー依頼「frame_self_attention_spatialを実装してください」。設計済みの `frame_self_attention` を、実際のAttention軸がフレーム内の細胞軸であることを示す実験名で実装する。
- 移行元backlog: `backlog/frame_self_attention.md`。同候補は `設計可能・実験化未承認`、未決事項は「なし」だった。
- 対応する上位仮説: `HYP-20260920-02`。
- 上位仮説のうちこの実験が検証する範囲: 固定候補・固定画像特徴・既存教師と復号で、各フレーム内の細胞間Self-AttentionをPair MLPの前に追加する効果。
- この実験だけで上位仮説を判断できるか: いいえ。2層・3 epoch・固定公開特徴という条件に限る。
- 上位仮説の判断に残る検証: parameter数と計算量を揃えた比較、候補密度別の誤り、他の画像特徴と独立評価。
- 親実験: `exp016_frozen_image_encoder`。同実験の保存済み結果を再現性の参照値とし、今回の主対照は同じコード・cache・3 epochで再学習する現行構成。
- 根拠 / 一次資料 / 参照実装: `docs/surveys/biohub-node-self-attention-design_20260920.md`、`experiments/exp001_temporal_unet3d_baseline/official_source/src/tracking_cellmot/models/simple_node_transformer.py`、親実験の `requirements.md` と `config.yaml`。ユーザー添付NFL Notebookから取り入れるのは、入力群を独立に文脈化してから融合する構成であり、選手の時間軸埋め込みは細胞順へ流用しない。
- 保存済み生成物とSHA: exp015 train cacheは19,701 window、summary SHA256 `040d1f6437e27149e0e34ad4aac8bba3c8cf63dc266d33df497acd969972194c`、identity SHA256 `440891c4550adf8540e3c47f9784e68b6ca1b64bbe56dbb93a25eb87c46bc1ee`。公開初期primary tracker checkpoint SHA256は `12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771`。exp016のfold別結果の所在とSHAは同実験 `metrics.json` を正とする。
- 観測と仮定: 現行は64次元入力を128次元へ投影し4層の双方向Cross-Attentionで接続を採点する。exp016の公式combined scoreは199動画全体で約0.912055。Assumption: 同一フレーム内の複数cellを直接参照すると親候補の取り違えが減る。現行Cross-Attentionでも反対側を経由して同側の情報が伝わり得るため独立した寄与は未実証。
- 採らなかった案と理由: t/t+1で別々のEncoder重みを持つ案は変更点を増やす。Crossを削除する決め打ちはModel A/B比較を妨げる。逆方向Crossへ更新前のtを渡す同時更新は現行との比較を変える。いずれも初回条件に含めない。
- 固定するもの: 公開検出器・画像encoderと正規化統計、exp015の19,701 window cache、32次元画像特徴と32次元位置特徴、座標と相対位置の `/100.0`、教師mask、focal weighted BCE、3 epoch、2方向胚holdoutと内部選択、optimizerとseed、Pair MLPとchunking、secondary tracker、ILP、graph repair、公式評価器。
- 変更するもの: 共有2層Transformer Encoderをtとt+1の細胞集合へ別々に適用。現行、Selfのみ、Self後に現行Cross-Attentionを行う3構成を切り替える。
- 先行条件と順序: exp024の結果を順序上先に確認するが、3 epoch固定の設計に必要な入力ではない。公開初期checkpoint、exp015 cache、exp016の評価経路を必須入力とする。[`exp030_frame_self_attention_diagnostics`](../exp030_frame_self_attention_diagnostics/result.md)は旧checkpoint互換性と最大級windowの実行可能性を確認した先行診断であり、接続精度の根拠にはしない。
- 最小の反証可能な検証: Kaggle上で代表的なwindowと最大級windowについて3構成の所要時間とpeak GPU memoryを測定し、予算内なら3構成×2foldを学習する。両胚の隣接2フレーム指標を同一runの現行対照と比較してから、下記の進行条件に従い199動画の公式graph評価を行う。
- 成功条件: Self後Cross構成が同一実行の現行対照より44b6と6bbaの両方で公式combined scoreを改善し、全体の接続成分も悪化しない。
- 停止条件: shape・mask・checkpoint互換性のテストまたはGPU費用gateを通らなければフル学習へ進まない。学習時間見積りと推論時間見積りが12時間を超える場合は条件を無断で縮小しない。学習後の隣接2フレーム評価が下記の進行条件に届かなければ、全graph推論を自動で始めない。
- 実行しないこと: 検出器や画像encoderの再学習、候補・教師・loss・復号の同時変更、cell index由来の時間埋め込み、外側胚での設定選択、閾値探索、Kaggle submission。
- 未決事項: なし。所要時間・精度・メモリは実測事項であり、実行前のGPU残量と公開対照再学習の承認は運用gateとして別に確認する。
- backlog記録から解釈を変更した箇所とユーザー承認: 実験名に `spatial` を付けた。Attention軸・モデル構成・評価条件は設計済み候補と同じ。

## 判断履歴

- 2026-09-20: ユーザーがNFLの別branch構造を参考にした設計と、Selfのみ・Self＋Cross・現行の3構成比較を依頼。設計に限定した。
- 2026-09-20: 逆方向Cross-Attentionは、更新済みt側をKey/Valueに使う現行の逐次更新を維持するとユーザーが確認。
- 2026-09-20: ユーザーが設計をバックログに追加するよう依頼。
- 2026-09-20: ユーザーが `frame_self_attention_spatial` の実装を依頼し、実験化を承認。

## 手法契約

- input: exp015 cacheの隣接2時刻の固定検出候補、画像特徴32次元、位置特徴32次元、voxel単位の `(z,y,x)`、各側の実cell mask。
- target / objective: exp016と同じGEFF既知接続に対応付けた候補cell pairの二値教師。未知注釈を新たに負例化しない。
- output: `[B,N_t,N_t1]`、unbatchedでは `[N_t,N_t1]` の活性化前edge logits。
- loss: 親cell軸softmax後のfocal weighted BCE、gamma 2。既知正例に触れる行または列のpairだけを対象とする既存mask。
- decode: 同一secondary tracker・候補保持・ILP・graph repair・公式評価器。構成ごとの閾値最適化なし。
- context unit: 隣接2時刻window。tとt+1の各フレーム内のcell集合へ同一Encoderを独立に適用し、動画単位でgraphを復元する。
- 実装区分: このリポジトリ内の管理用語では `staged-faithful`。参照Notebookの分岐して文脈化後に融合する構造をcell集合へ適応する。
- 省略する機構と理由: NFLの選手軌跡の時間順位置埋め込みと後段の選手間branchは、cell番号に時間順の意味がなく、今回の接続出力とも対象が異なるため移植しない。
- proxyで検証できない主張 / proxyの場合の承認: N/A。token単位MLPへの置換などのproxyは使わない。
- この実験が支持 / 棄却できる主張: 固定公開特徴・2層Encoder・3 epochで現行Crossのみと比べたフレーム内Self-Attentionの追加効果。
- この実験では判断できない主張: 全Self-Attention構造の一般的な優劣、公開画像モデルから独立したCV、hidden test精度。

## 公式graph評価前の隣接2フレーム比較

- 比較対象: Kaggle学習Notebookが出力する各foldの `trained_outer_evaluation` を使い、`self_only` と `self_cross` を同一runで再学習した `legacy` とそれぞれ比較する。44b6と6bbaを分け、同じexp015 cache、候補、GEFF対応、教師mask、外側胚window、checkpoint選択手順、親軸softmax後の予測閾値0.5を保つ。保存済みexp016の値は履歴上の参考値とし、主対照と混同しない。
- 確認する指標: 既知edgeの `positive_edge_recall` と正例回収数、既存mask内の `edge_accuracy` と予測された教師負例pair数、`division_parent_recall` と既知分裂母の回収数を両胚別に示す。既存の正例数・評価pair数・分裂母数も併記し、同じ分母で比較できることを確認する。教師負例pair数は保存済み集計の正例数、評価pair数、正例recall、accuracyから整数件数として復元し、復元できない場合は追加集計する。
- 自動進行条件: `self_only` または `self_cross` の少なくとも一方が、同じ構成について両胚とも現行対照以上の既知edge recall、現行対照以下の教師負例pair予測数、現行対照以上の既知分裂母回収数を満たすこと。等値を許すのは前段での悪化を検出するためであり、精度改善や実験採用の判定ではない。分母が0の指標は判定不能として理由を記録し、自動進行しない。
- 条件に届かない場合: 両胚の差、教師の部分注釈、分裂の少数例、計算予算を確認し、全graph評価を保留してユーザーへ示す。ユーザーが全graph実行を明示した場合は、取得可能な前段指標と条件不成立の理由を記録したうえで進める。続行・保留の判断は `SESSION_NOTES.md`、数値は `metrics.json` に残す。
- この比較の限界: 既存mask内の教師負例は疎い注釈に依存し、真の誤接続全体を表さない。隣接2フレームの閾値付き予測はsecondary tracker、ILP、graph repair後の接続・分裂と一致するとは限らない。公式scoreを主張するときはKaggleで全graphと公式評価器を実行する。

## 実装方法

- inputの実装箇所と変換: 親実験から引き継ぐ `frozen_tracker.py` がexp015 cacheを検証してwindowを組み立てる。入力投影は `simple_node_transformer.py` の共有 `proj` と `norm_in`。
- target / objectiveの構築箇所: `frozen_tracker.py` の公開5µm greedy対応、正例行・列に触れるpairのloss maskを維持。
- outputの生成箇所: `simple_node_transformer.py`。共有 `self_encoder` をt/t+1へ独立適用し、必要時に既存 `blocks` の逐次Cross-Attention、`norm_out`、相対位置を含む既存Pair MLPへ渡す。
- lossの実装箇所: `frozen_tracker.py` の `batch_legacy_focal_bce` を変更しない。
- decode / postprocessの実装箇所: `graph_inference.py` と inference Notebook。公開secondary trackerとexp016の固定処理を使う。
- context unitを保つ処理箇所: `simple_node_transformer.py` の `_encode_frame` が各フレームを別呼び出しする。2時刻tokenは結合しない。
- 変更するファイル / component: 実験内model source、3構成config、train/inference Notebook source、graph replay、focused tests。
- 固定事項を保つ確認方法: exp015 cacheと公開重みのSHA確認、同じsplit・教師・loss・復号config、保存済み公開candidate graphとの完全一致。
- 参照sourceとの一致を確認するテスト: Self無効のstate dictを `strict=True` 読み込み、eval logitsを旧モデルと一致させる。
- 承認済み差分を確認するテスト: padding不変性、cell順序同変性、別cellへの情報伝播、空集合と有限勾配、A/Bのstrict保存復元、学習・推論source SHA一致。

## 探索幅とpivot判定

- 変更class: このリポジトリ内の管理用語では `mechanism`。cell集合内のAttention演算を追加する。
- 同じ親 / familyで連続した小改善実験数: 今回の変更は単なるparameter変更ではない。
- positiveなoracle headroom / coverage / 誤差非相関性: 親実験・exp015の保存値を参照し、今回の結果で再評価する。
- target、output、decode、context unitを変える案: 本実験では採らず、上位仮説の残る検証として別に判断する。
- 小改善の継続またはpivotを選ぶ根拠: 今回はユーザーが具体的な設計済み機構の実装を指定した。
- `kaggle-idea-forge` の実行要否: 今回は不要。連続した小改善の3件目に該当する変更ではない。

## 再現性・リスク

- seed policy: 親実験と同じ全体seed 42とfold offset、seed付きDataLoader。3構成で共通moduleの初期化seedを揃える。
- stochastic 処理: shuffle、dropout、CUDA演算。画像特徴cache生成は固定。
- stochastic feature generation / augmentation / seed bagging: なし。
- 並列処理と乱数: foldごとにgeneratorとworker seedを固定する。
- CPU/GPU runtime と deterministic flags: Kaggle T4、mixed precisionなし。PyTorch/CUDA環境、所要時間、peak memoryを記録する。
- train cache / test feature regeneration SHA: exp015のsummary・identity SHAと各window契約を確認し、画像encoderの再forwardは行わない。
- model manifest / prediction / submission SHA: 解決済みmodel params、source SHA、公開初期checkpoint SHA、各model state/file SHA、variant graph SHAを記録。submission SHAは対象外。
- Kaggle package bootstrap: train/inferenceへ同じ実験sourceを同梱し、importとmodel source SHAを照合する。
- リークリスク: 公開画像モデルと初期trackerの学習来歴に評価胚が含まれる。比較は固定公開モデル下の条件付き評価。
- CV/LB 不一致リスク: 公式graph評価はtrain 199動画のみ。LBとhidden testは未評価。
- ランタイム/メモリリスク: Self-Attentionは各フレームのcell数の二乗に応じる。大きいwindowで事前測定し週30 GPU時間とNotebook 12時間に収める。
- 再現性リスク: 新Encoderの層ごとの初期値とcheckpoint metadataが必要。欠損keyをSelf層だけに限定する。
- 手法忠実性リスク: Selfのみの構成はCrossを省くがModel Bでは現行の逐次更新を保つ。cell index positional encodingを入れない。
- 過度な縮小 / proxy化リスク: GPU予算超過時に候補数・教師・画像特徴を黙って変更しない。

## 受け入れ基準

- [ ] 3構成×2foldのKaggle学習を実測し、時間・memory・各checkpoint SHAを保存する。
- [ ] 同一runの現行対照と各Self-Attention構成について、両胚の隣接2フレーム指標・件数を記録し、公式graph評価への進行条件を判定する。
- [ ] 全199動画のgraphを同一復号で公式評価し、44b6と6bbaの指標・成分を比較する。
- [ ] モデル互換性・padding・空集合・順序・勾配・復元テストと静的検証を通す。
- [ ] 実験の完了・採否は証拠を提示してユーザーに判断を求める。submissionは別途明示依頼まで実行しない。
