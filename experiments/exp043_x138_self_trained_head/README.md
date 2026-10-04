# exp043_x138_self_trained_head

## 概要

- 仮説要約: 固定した公開画像特徴から学習する3軸の座標補正headで既知中心との距離を縮められるか。
- 変更点要約: 主催者GEFFで自前headを学習し、x138の推論へ組み込む。作者の追加重みは使用しない。
- リスク: 疎な注釈と公開画像モデルのtrain学習来歴。作者の0.953の再現とは別の結果。
- 次: 座標補正headの後続比較に使う。このx138構成でトラッカーを学習する場合は、同じ入力・候補・後処理の公開トラッカーを評価して対照とする。exp016は指標・評価手順の参考に留め、旧構成の数値を直接比較しない。[比較条件](result.md#今後の実験との比較)を参照する。

## 正の記録

- 数値、実験status、構造化された実行証拠: [metrics.json](metrics.json)
- route、設定、系譜: [config.yaml](config.yaml)
- 実装前の要件、実装方法、受け入れ条件: [requirements.md](requirements.md)
- 証拠への参照、結果の解釈、ユーザー判断: [result.md](result.md)
- 実行中の作業ログ: [SESSION_NOTES.md](SESSION_NOTES.md)

## 実行入口

- 学習Notebook: [exp043_x138_self_trained_head_train.ipynb](exp043_x138_self_trained_head_train.ipynb)
- 推論Notebook: [exp043_x138_self_trained_head_inference.ipynb](exp043_x138_self_trained_head_inference.ipynb)
- Jupytext sourceは同名の`.py`、生成器は`build_notebook.py`と`build_inference_notebook.py`。
- Kaggle実行、competition submission、採点の経過は[SESSION_NOTES.md](SESSION_NOTES.md)を参照する。

## 表記

用語と記録上の区分は[AGENTS.md](../../AGENTS.md)と[docs/glossary.md](../../docs/glossary.md)に従う。
