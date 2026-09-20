# exp028_direct_graph_prediction

## 概要

- 仮説要約: 固定検出候補から娘ごとの母候補または対応なしをTransformerが直接選ぶと、既存の接続scoreとILPより接続・分裂を改善できるか。
- 変更点要約: 主催者の既知入edgeだけを教師とし、1娘最大1母、1母最大2娘の制約で選択する。接続選択にILP、secondary tracker、順逆融合は使わない。
- リスク: 注釈の欠落により対応なしの教師が少なく、未知への過剰接続があり得る。固定graph repairは接続を変更し得るため適用前後を分けて評価する。
- 次: [結果](result.md)と早期gateを確認し、この実験の完了・採否をユーザーが判断する。

## 正の記録

- 数値、実験status、構造化された実行証拠: [metrics.json](metrics.json)
- route、設定、系譜: [config.yaml](config.yaml)
- 実装前の要件、実装方法、受け入れ条件: [requirements.md](requirements.md)
- 証拠への参照、結果の解釈、ユーザー判断: [result.md](result.md)
- 実行中の作業ログ: [SESSION_NOTES.md](SESSION_NOTES.md)

## 実行入口

- 学習と2-frame診断: exp028_direct_graph_prediction_train.ipynb
- gate成立後の全graph推論と公式評価: exp028_direct_graph_prediction_inference.ipynb
- 公式のフル実行環境: Kaggle Notebook。推論Notebookはgateの証拠を設定するまで実行を拒否する。
