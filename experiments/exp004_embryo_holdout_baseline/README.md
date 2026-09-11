# exp004_embryo_holdout_baseline

## 概要

- 仮説要約: exp002のモデル、損失、3 epochs、checkpoint選択、推論条件を固定し、学習と評価の胚を2方向で分ければ、全199動画について学習から除外された胚の基準予測と公式評価を作成できる。
- 変更点要約: 胚`44b6`と`6bba`を入れ替える2foldを作り、内部の重み選択も学習胚内に限定する。動画別に予測graph、検出score、全接続score、閾値・graph入力・最終選択maskを保存する。
- リスク: Kaggle train versions 1-3で固定batch size 16のsmokeがCUDA OOMとなり、fold 1はfold間のCUDA解放後も再現して失敗した。full training、全199動画の推論、公式評価は未実行。
- 次: 固定条件の失敗を本実験に残し、batch sizeなど結果に影響する計算条件を変更した再実験へ進むかユーザー判断を確認する。submissionは本実験の範囲外。

## 正の記録

- 数値、実験status、構造化された実行証拠: [`metrics.json`](metrics.json)
- route、設定、系譜: [`config.yaml`](config.yaml)
- 実装前の要件、実装方法、受け入れ条件: [`requirements.md`](requirements.md)
- 証拠への参照、結果の解釈、ユーザー判断: [`result.md`](result.md)
- 実行中の作業ログ: [`SESSION_NOTES.md`](SESSION_NOTES.md)

## 実行入口

- 学習 notebook: `exp004_embryo_holdout_baseline_train.ipynb`
- 推論と公式評価 notebook: `exp004_embryo_holdout_baseline_inference.ipynb`
- Kaggle準備と実行: [`SESSION_NOTES.md`](SESSION_NOTES.md)の予定と`kaggle-review-exp`、`kaggle-platform`の手順に従う。
- notebook実行: Kaggle kernel runを正とする。ローカル実行は必要な入力と依存が揃う場合のsmoke debugだけに限定する。
