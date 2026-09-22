# exp034_frame_self_attention_distance_bias

## 概要

- 仮説要約: 恒等初期化Model BのSelf-Attentionへ学習可能な物理距離biasを加えると、固定pipelineの最終graph scoreが改善するか。
- 変更点要約: head別の負の二乗距離biasを1設定だけ、2 embryo holdout × 3 epochで学習する。pair診断による事前gateは使わない。
- 評価: fold 0を6bba、fold 1を44b6へ適用し、固定ILP・graph repair後の最終graphを保存済みexp015 controlと同じ公式評価器で比較する。
- リスク: 密なSelf-Attentionは候補数の二乗で重く、公開画像encoderと初期trackerの学習来歴にtrain動画が含まれるため独立CVではない。
- 学習結果: Kaggle train version 2で2fold×3epochを完走し、2 checkpointとmanifest SHAを検証済み。
- 完了判断: tracker学習結果に明確な改善がなく、週30時間のGPU上限を超える最終graph評価は実行せず、ユーザー判断で実験を完了した。

## 正の記録

- 数値、status、Kaggle実行、生成物SHA: [metrics.json](metrics.json)
- route、設定、系譜: [config.yaml](config.yaml)
- 実験契約、実装方法、受け入れ条件: [requirements.md](requirements.md)
- 結果と解釈: [result.md](result.md)
- 時系列の作業ログ: [SESSION_NOTES.md](SESSION_NOTES.md)

## 実行入口

- 学習: exp034_frame_self_attention_distance_bias_train.ipynb
- 最終graph replayと公式評価: exp034_frame_self_attention_distance_bias_inference.ipynb
- 距離biasモデル: simple_node_transformer.py
- checkpoint復元と固定graph replay: graph_inference.py
- 正のフル実行環境: Kaggle T4。Kaggle competition submissionは作成しない。
