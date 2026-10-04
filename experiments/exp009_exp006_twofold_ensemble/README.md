# exp009_exp006_twofold_ensemble

## 概要

- 仮説要約: exp006の2foldモデルを検出・edge確率で等重み融合すれば、両胚由来の学習情報を使うhidden-test提出を作れる。
- 変更点要約: 両モデルの検出確率を平均後に共通nodeを抽出し、そのnode上のedge確率も平均してから閾値とILPを各1回適用する。再学習はしない。
- リスク: hidden test規模に対する12時間制約と2モデル同時保持のGPUメモリ。1回の推論ではbitwise再現性を実証しない。
- 次: 推論・提出・採点は終了しており、[結果](result.md)を基に実験の採否判断を待つ。後継の再現性実験も[exp010の途中停止判断](../exp010_exp006_deterministic_replay/result.md)を参照し、追加実行は予定しない。

## 正の記録

- 数値、実験status、構造化された実行証拠: [`metrics.json`](metrics.json)
- route、設定、系譜: [`config.yaml`](config.yaml)
- 実装前の要件、実装方法、受け入れ条件: [`requirements.md`](requirements.md)
- 証拠への参照、結果の解釈、ユーザー判断: [`result.md`](result.md)
- 実行中の作業ログ: [`SESSION_NOTES.md`](SESSION_NOTES.md)

## 実行入口

- 推論 notebook: `exp009_exp006_twofold_ensemble_inference.ipynb`
- 学習は行わず、exp006 train kernelの2 checkpointを読み込む。
- Kaggle実行と提出手順は[`SESSION_NOTES.md`](SESSION_NOTES.md)に記録する。
