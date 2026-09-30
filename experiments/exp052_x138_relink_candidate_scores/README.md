# exp052_x138_relink_candidate_scores

## 概要

- 仮説要約: 固定trackerのILP未選択辺の得点をmotion再接続へ渡すと親割当が改善する。
- 変更点要約: 再接続の得点辞書だけを拡張し、同じILP graphと再追加nodeで比較する。
- リスク: cacheの保存範囲と再追加nodeのID来歴を誤ると比較が成立しない。
- 次: ユーザー判断で採用・完了。未実施の学習動画評価と上位仮説の残る候補は別途判断する。

## 正の記録

- 数値、実験status、構造化された実行証拠: [metrics.json](metrics.json)
- route、設定、系譜: [config.yaml](config.yaml)
- 実装前の要件、実装方法、受け入れ条件: [requirements.md](requirements.md)
- 証拠への参照、結果の解釈、ユーザー判断: [result.md](result.md)
- 実行中の作業ログ: [SESSION_NOTES.md](SESSION_NOTES.md)

## 実行入口

- CPU所要時間試行: exp052_x138_relink_candidate_scores_cpu_pilot.ipynb
- 内部診断: exp052_x138_relink_candidate_scores_diagnostic.ipynb
- 条件成立時の新規評価: exp052_x138_relink_candidate_scores_inference.ipynb
- Kaggle準備・実行と進行記録: [SESSION_NOTES.md](SESSION_NOTES.md)
