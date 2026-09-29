# exp053_x138_postlink_division_score

## 概要

- 仮説要約: exp043の再接続後のgraphで、4時点の接続情報が第2娘の採点に役立つ。
- 変更点要約: 全動画ILPは増やさず、分裂追加の候補順位と閾値を小さな学習採点器で決める。
- リスク: 疎いGEFF教師と完成graphの候補分布が異なり、既知分裂が候補外の可能性がある。
- 次: 元設計の前後edge同時選択や画像特徴の寄与は別の比較で検証する。

## 正の記録

- 数値、実験status、構造化された実行証拠: [metrics.json](metrics.json)
- route、設定、系譜: [config.yaml](config.yaml)
- 実装前の要件、実装方法、受け入れ条件: [requirements.md](requirements.md)
- 証拠への参照、結果の解釈、ユーザー判断: [result.md](result.md)
- 実行中の作業ログ: [SESSION_NOTES.md](SESSION_NOTES.md)

## 実行入口

- Kaggle CPUの特徴生成と診断: exp053_x138_postlink_division_score_train.ipynb
- Colab CLIの係数学習: exp053_x138_postlink_division_score_colab_train.ipynb
- Kaggle GPUの提出推論: exp053_x138_postlink_division_score_inference.ipynb
