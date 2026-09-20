# frame_self_attention_diagnostics

- 候補名: `frame_self_attention_diagnostics`
- 状態: `検討メモ・設計不可`
- 対応する上位仮説: `HYP-20260920-02`
- 関連する上位仮説: HYP-20260910-14
- 作成日: 2026-09-20
- 最終更新日: 2026-09-20
- 依頼原文: 「バックログに追加してください」。直前に、PR曲線・誤り分析を先に行い、その結果を受けて近傍制限・距離biasの追加を検討する順序を説明した。
- 期待する成果: 保存したtrackerの予測順位と確率0.5での判定の差を区別し、候補数・近傍の混雑度・移動距離・分裂に関連する誤りを示して、次の構造変更の根拠を得る。
- 親実験 / 比較対象: [exp025](../experiments/exp025_frame_self_attention/)のModel A、Model B、model_b_identity_init、および[exp016](../experiments/exp016_frozen_image_encoder/)の保存済み現行tracker。全4構成を同じwindow・候補・教師で比較する。
- 優先度: P1
- 優先度の理由: 恒等初期化Bは通常Bより両胚でprecisionが上がりrecallが下がった。構造をさらに変える前に保存重みの再評価でこの差を調べる。追加学習は不要だが、pair確率の再取得には推論費用がかかる。
- `backlog/KAGGLE_DIRECTION.md` の対応箇所: [検証中の仮説](KAGGLE_DIRECTION.md#検証中の仮説)と[未着手バックログ](KAGGLE_DIRECTION.md#未着手バックログ)

## 観測事実と根拠

- 実測済みの事実: exp025の3variantは各2fold・3epochを完走した。恒等初期化Bは初期logitsの完全一致を実データで確認できたが、学習後は現行に対する既知edge recallが両胚で低下。precision・教師上のfalse-positive pair・分裂指標は混在した。追加Self-Attentionのゼロ初期化した出力射影は学習後に更新されていた。
- 根拠ファイル / 一次資料: [exp025結果](../experiments/exp025_frame_self_attention/result.md)、[metrics](../experiments/exp025_frame_self_attention/metrics.json)の`train_stage.pair_level_comparison`、`post_identity_diagnostics`、[契約](../experiments/exp025_frame_self_attention/requirements.md)、[config](../experiments/exp025_frame_self_attention/config.yaml)、[評価実装](../experiments/exp025_frame_self_attention/frozen_tracker.py)。対照の[exp016契約](../experiments/exp016_frozen_image_encoder/requirements.md)・[config](../experiments/exp016_frozen_image_encoder/config.yaml)も継承する。
- 利用する保存済み生成物とSHA: exp015のtrain window cache、exp016 train v3、exp025 A v1・B v4・identity v1の各fold checkpointを使う。kernel ID、model file/state/source SHAは各実験metricsを正とし、重みの取得時にmanifestと照合する。exp025のmanifest SHAはA=`c0e70224a14de199119c0104e51c002f7b6121bc562dc4727f52703998e80e3c`、B=`123121e4e882220bfe86af249b6712d6cfe12fa64fed1fa415bdc50d2a21e6ea`、identity=`442c25536a64419d2460755a13513edd0f4aa76b39df29d954af5a92359faa4c`。exp016は`6f9548739ae6f58d3c8beb7ccb78464ddc10243841b1fc7e071ed876f757ea56`。cache summary/identity SHAはexp025 configの記録を使う。ローカルの一時取得先だけに依存せずKaggle outputから回収可能にする。
- 仮定: Assumption: 固定閾値でのprecision/recallの差には確率の出方の変化が含まれ、全閾値での順位比較や誤りの条件別集計により、その寄与を調べられる。距離biasの有効性を示す直接証拠はまだない。
- 既存候補との境界: [contrastive_feature_audit](contrastive_feature_audit.md)は対照学習に使う教師資格と固定特徴の識別力を監査する。本候補は学習済み4構成のpair予測を比較し、教師・lossや特徴モデルは変更しない。[video_calibration](video_calibration.md)の動画別画像前処理の変更も含めない。

## この候補が直接検証する仮説と範囲

- 上位仮説のうちこの候補が検証する範囲: Self-Attention追加の効果が、確率0.5の指標だけでは見えない順位や局所的な誤りの差として残るか。
- この候補の具体的な仮説: 4構成のPR曲線と同じprecisionでのrecall、条件別の親候補の取り違えを比較すると、確信度の変化と対応の識別能力の差を区別できる。
- 仮説が正しい場合に期待する観測: 確率0.5ではrecallが低い構成でも共通precisionで回収率が保たれる、または候補数・混雑度・距離・分裂の特定条件で順位の改善・悪化が一貫して現れる。
- 仮説を棄却する観測: 共通precisionでのrecallも低く、条件別の改善傾向も再現しない場合、固定閾値が改善を隠したという説明を支持しない。診断が否定的でも実施の失敗とは扱わない。
- この候補だけで上位仮説を判断できるか: いいえ
- 上位仮説の判断に残る検証: 距離bias等を加える学習比較と公式graph評価。pair指標を公式scoreの代用にしない。

## 入力・予測対象・出力・推論方法

- input: 同じexp015固定画像・位置特徴cache、候補座標、GEFF由来教師、固定済みfold分割、各構成の保存checkpointと対応するmodel source。
- target / objective: 既存の既知親子edgeと分裂親。教師mask・候補対応を変えず、モデルのスコア順位と判定の誤りを比較する。
- output: 胚・構成・評価split別PR曲線、共通precisionでのrecall、既存0.5指標の再現、候補数・混雑度・移動距離・分裂別の誤り、有効件数と未知endpoint件数。必要なpair確率・pair ID・教師・座標の対応を保存し、集約値からPR曲線を捏造しない。
- loss: なし。既存lossは比較条件の参照のみで、勾配更新・再学習をしない。
- decode / 推論方法: 既存trackerのpair logitsから従来どおりsource軸softmaxを計算する。graphは作らず、閾値や重みの運用上の選択も実施しない。
- 処理単位: 隣接2-frame windowで推論し、内部検証と外側胚評価を分けて両胚別に集計する。
- 実装区分: 既存trackerと教師をそのまま再実行する診断。新たな参照モデルを実装しないためfaithful / staged-faithful / proxyの分類対象外。集計を学習や公式評価の代用品にしない。

## 親実験からの差分

- 変更するもの: 確率0.5の集約に加え、pair確率と条件別指標を取得する評価出力。
- 固定するもの: 検出器、画像特徴、候補、座標、padding mask、教師、source軸softmax、checkpoint、fold、既存のmodel architecture。内部検証と外側評価の境界を保つ。
- 再利用するコード / config / 生成物: exp025の`frozen_tracker.py`、source SHAに対応する各model、split・teacher監査。exp016の保存済み対照。
- 新しく作るもの: 実験化後のKaggle診断Notebook、pair予測の保存とPR・条件別集計。今回は作成しない。

## 最小の反証可能な検証

- 検証方法: 最初に0.5閾値の既存指標を同じwindowで再現し、PR曲線と条件別誤りを取得する。集計定義は下記未決事項を解消して実装前に固定する。外側胚のPR曲線は記述的な比較に限り、閾値・層数・checkpointの選択に使わない。
- variant / config / fold / booster数: 保存済み4構成×2fold=8 checkpoint、診断config 1、tracker学習0、booster 0。内部検証と対応する外側胚を別集計する。
- control再学習: なし。保存済みexp016を同条件で再評価する。
- 想定runtime / resource: Kaggle Notebookで小規模再推論の実測後に全件費用を見積もる。保存後の集計はCPU中心。再学習不要でも推論が無料・瞬時とは仮定しない。GPU週30時間、Notebook12時間の上限と実行前quotaを守る。

## 成功条件と停止条件

- primary指標: PR曲線と同じprecisionでの既知edge recall。比較precision・集計規則は未決事項に残す。
- 成功条件: 既存の0.5指標を再現したうえで、4構成の差を両胚・同条件で示し、確信度の変化だけでは説明できない誤りの有無を報告できる。モデルの改善を観測すること自体を診断の完了条件にしない。
- 必須guard: checkpoint/source/cache/teacherのSHA一致、同じwindowと候補、padding除外、部分注釈の限界、有効件数、内部検証と外側評価の分離。false-positiveを真の誤接続数と呼ばない。
- 成功時の次段階: 診断を根拠に[frame_self_attention_spatial](frame_self_attention_spatial.md)の方式と実験化を検討する。局所誤りだけで距離biasが有効と確定せず、追加学習・graph評価の自動開始はしない。
- 失敗時の停止範囲: 既存指標が再現しなければ比較を止めて入力・source・教師を照合する。改善がなければその結果を示し、閾値・教師を変えて同じ外側胚上で救済探索しない。

## 実行しないこと

- 禁止する代替実装、proxy、同一OOF上の救済探索: 集約済みrecall/accuracyからPR曲線を復元する、未知pairを確定負例と呼ぶ、外側胚で閾値・bucket・checkpointを最適化する、全graph推論・submissionを始める。
- 壁打ちで採らなかった案と理由: すぐに距離biasや近傍制限を追加する案は後段とした。現在は順位と確率の変化が未分離で、遠方cellの混合が原因だという直接証拠がない。単純なepoch延長と追加層の同時変更もしない。

## リスク

- leakage / validation: 公開重みの学習来歴にtrain動画が含まれる条件付き比較。内部検証もcheckpoint選択に使用済みであり、新しい独立テストとは呼ばない。外側胚を見て方法を最適化しない。
- hidden test: trainの診断のみ。hidden test入力・未知のGEFF・LBを使わない。
- runtime / memory: 全pair保存は大きいため、source軸softmaxを壊さない単位で推論し、出力を分割して保存する。集計だけのために全画像特徴を再抽出しない。
- 再現性: 構成別sourceとcheckpointを正しく対応させ、pair ID・window・分割・教師mask・確率・集計configを保存する。

## 調査・実行時に確認する事項

- 新たなpair出力のSHAと容量、実推論時間・peak memory、PR曲線とbucket別有効数。未実測であることを設計上の未決事項と混同しない。

## 未決事項

- 「同じprecision」の基準点または共通範囲、および到達不能時・同点scoreの集計規則。外側胚で有利な基準を後選択しない。
- 近傍の混雑度を候補の最近傍距離、固定物理半径内の候補数などのどれで測るか、および候補数・距離のbucket境界と小標本の扱い。指標の意味が変わるため実装前に確定する。

## 判断履歴

- 2026-09-20: 恒等初期化でも一貫したpair改善がなく、ユーザーが近傍制限・距離biasとの優先順位を確認。PR曲線・誤り分析を先行させる順序を提案した。
- 2026-09-20: ユーザーのバックログ追加依頼によりP1で記録。今回の依頼は実験化・実装・Kaggle実行の承認として扱わない。未決事項を推測で埋めず検討メモとする。

## 次セッションへの引き継ぎ確認

- 固定するものを一意に説明できる: はい。4構成の保存重み、入力、教師、splitと既存0.5評価。
- 変更するものを一意に説明できる: 評価出力の追加は明確。precision比較と混雑度・bucketの方式は未決。
- 最小検証と停止条件を一意に説明できる: 既存指標再現と同条件比較、失敗時の停止範囲は明確。新規集計の定義は上記を確定する。
- 実行しないことを一意に説明できる: はい。再学習、外側胚による選択、graph推論、submission。
- 未決事項が明示されている: はい。
