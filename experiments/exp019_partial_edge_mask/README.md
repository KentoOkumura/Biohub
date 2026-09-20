# exp019_partial_edge_mask

## 概要

- 仮説要約: exp016と同じ固定候補と復号で、正例のある子列だけを接続損失に使うと、未記録の第2娘を抑える学習が減るかを検証する。
- 変更点要約: pair loss maskを正例行または列から、正例子列のみへ変更する。未知親はsoftmax分母に残す。
- リスク: 公開画像encoderの学習来歴によりfold別比較は独立CVでない。未知親への間接勾配と過検出は残る。
- 次: [結果とユーザー判断](result.md)を次候補の教師設計に引き継ぐ。

## 正の記録

- 数値、実験status、構造化された実行証拠: [metrics.json](metrics.json)
- route、設定、系譜: [config.yaml](config.yaml)
- 実装前の要件、実装方法、受け入れ条件: [requirements.md](requirements.md)
- 証拠への参照、結果の解釈、ユーザー判断: [result.md](result.md)
- 実行中の作業ログ: [SESSION_NOTES.md](SESSION_NOTES.md)

## 実行入口

- 学習Notebook: `exp019_partial_edge_mask_train.ipynb`
- 推論Notebook: `exp019_partial_edge_mask_inference.ipynb`
- Kaggle準備と実行: [SESSION_NOTES.md](SESSION_NOTES.md)を参照する。

## 表記

用語はリポジトリの`AGENTS.md`と`docs/glossary.md`に従う。
