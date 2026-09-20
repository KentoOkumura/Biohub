# exp028_direct_graph_prediction セッションノート

## 目的

固定候補の接続を主催者注釈から直接選ぶTransformerを学習し、既存scoreとILPを使う経路と比較する。

## 現在の作業

- 2026-09-20: ユーザーが娘ごとの親選択と主催者の既知接続だけの損失、2-frame先行評価を選択。全graphではsecondary・順逆融合・ILPを接続選択から外し、固定graph repairだけを適用する。
- 実行予定: active variant 1、model/config 1、fold 2、booster 0、control再学習なし。公開検出器と画像encoderを再学習しない。
- 実装: exp028を作成し、候補の根拠と判断履歴をrequirementsへ移管。Jupytext形式のtrain/inference Notebook、未知を損失から除く教師、親子数制約、早期gateを実装。
- 実行状態: ローカルの小例と静的検証は完了。Kaggle full train、2-frame対照復元、全graph、公式評価は未実行。

## コマンドログ

- 2026-09-20: make new-exp EXP=exp028_direct_graph_prediction
- 2026-09-20: JupytextからtrainとinferenceのNotebookを生成し、変換整合を確認。
- 2026-09-20: 小例テスト13件、check-exp、strict validate-exp、check-strategy-docs、Jupytext往復変換、git diff --checkが成功。PyTorchはローカルの.venvにないためGPU model forwardとGEFF実データのフル処理はKaggle実行で確認する。
- 2026-09-20: train Notebookのprivate T4 packageをローカル生成。cacheからの65次元入力、既知入edgeだけのloss、構造制約、保存済みexp016の同単位対照、2胚gate、固定graph repair、repair前後の公式評価を実装。閾値gridの内部検証はGPU forwardを1回に削減。graph変換はexp016と同じ座標丸め・clampに合わせた。
- 2026-09-20: make push-kaggle-trainはauto-reviewで2回却下。private Notebookの外部送信とGPU実行について、この依頼と設計承認だけでは明示承認が足りないとの判断。迂回せず、Kaggle runは未開始。

## 次のアクション

1. private train NotebookのKaggle push・GPU実行について明示承認を受ける。auto-review却下を迂回しない。
2. 承認後、train Notebookで小規模benchmark後に2-foldを実行する。
3. 同じ2-frame単位の保存済み対照を復元し、gateの根拠を記録する。成立時だけinferenceを有効化する。

## Kaggle train push前のresource確認

- 2026-09-20: 生成metadataはprivate T4 GPU、TPU false、internet false、train variant 1、model/config 1、fold 2、booster 0、control再学習なし。
- 2026-09-20: Kaggle quotaはGPU used 10.97h、remaining 34.03h、refresh 2026-09-26T00:00:00。リポジトリ方針の週30h上限では約19.03hが残る。train Notebookは64 window/foldの実測から12h gateを適用し、この範囲内でのみ2foldへ進む。inferenceは別途残量を再確認する。

## Kaggle実行

- 2026-09-20T15:25:43+09:00: ユーザーの明示承認後、private T4 train Notebook `kentookumura/exp028-direct-graph-prediction-train` version 1をpush。status RUNNING。GPU quota used 11.66h、Kaggle remaining 33.34h（リポジトリ週30h制限では残り約18.34h）。submissionは行っていない。
- 2026-09-20T15:44:58+09:00: liveログでfold 0 epoch 1のloss 2.295568、内部selection accuracy 0.592088、閾値0.30を確認。外側胚の診断とgateは未出力。15分間隔の`monitor-exp028-train`をユーザーの明示承認後に設定。自動監視の範囲はtrain完了・失敗の検知、生成物回収、2-frame gateの記録まで。全graph inferenceとsubmissionは含めない。ローカルのlive SSE接続だけを終了し、Kaggle version 1はRUNNINGのまま。

## train version 1 の結果

- 2026-09-20T16:56:04+09:00: `make kaggle-status`でCOMPLETE、`make kaggle-output KERNEL=kentookumura/exp028-direct-graph-prediction-train OUT=experiments/exp028_direct_graph_prediction/artifacts/kaggle_train_v1`で生成物を回収。model manifest、2 model、2 fold summary、early_gate、cache identityとsummaryのSHAを照合して一致。Notebook実行4995.77秒、GPU peak 3611434496 bytes。
- 6bba評価: 既知edge recallは直接選択95.35%、保存済みexp016対照97.20%。観測可能な誤接続率は5.93%対2.94%。既知分裂母の回収は45/108対50/108。構造違反0。既知edge 103393件、既知の候補外母1983件。早期gate不成立。
- 44b6評価: 既知edge recallは94.37%対95.30%。観測可能な誤接続率は7.21%対3.61%。既知分裂母の回収は12/22対6/22。構造違反0。既知edge 18949件、既知の候補外母379件。誤接続率条件で早期gate不成立。
- 両胚で同じcacheと教師を使う比較、control再学習なし。2胚とも誤接続率が対照より高いため、契約に従い全graph inferenceと公式評価は実行しない。公式score未計測。unknown娘の選択は確定負例とみなさず、誤接続率は観測可能な範囲のみ。採否・完了はユーザー判断待ち。
- 2026-09-20T16:57:06+09:00: train version 1の結果回収と記録が完了したため、承認済みheartbeat `monitor-exp028-train`をPAUSEDへ更新。全graph inferenceはgate不成立で実行しない。
