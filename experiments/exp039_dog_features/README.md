# exp039_dog_features

## 概要

- 仮説要約: 固定候補に元解像度画像の2組のDoG応答を加えると、接続・分裂を改善できる。
- 変更点要約: primary trackerの点特徴を64から66次元へ拡張し、同構造のゼロ入力対照と2胚で比較する。
- リスク: 元画像の抽出費用、座標のずれ、部分注釈下の誤接続判定。
- 次: ユーザー判断により不採用で終了。全graph推論と提出は行わない。

## 正の記録

- 数値、実験status、構造化された実行証拠: [metrics.json](metrics.json)
- route、設定、系譜: [config.yaml](config.yaml)
- 実装前の要件、実装方法、受け入れ条件: [requirements.md](requirements.md)
- 証拠への参照、結果の解釈、ユーザー判断: [result.md](result.md)
- 実行中の作業ログ: [SESSION_NOTES.md](SESSION_NOTES.md)

## 実行入口

- 元画像の特徴生成: `exp039_dog_features_features.ipynb`
- 学習と隣接ペア診断: `exp039_dog_features_train.ipynb`
- Kaggle実行の手順と進行条件: [SESSION_NOTES.md](SESSION_NOTES.md)と[requirements.md](requirements.md)
