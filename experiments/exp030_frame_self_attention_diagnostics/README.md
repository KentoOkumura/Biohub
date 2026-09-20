# exp030_frame_self_attention_diagnostics

## 概要

固定した公開画像特徴を使い、同一フレームのcell集合へのSelf-Attention追加について、旧trackerとの互換性とKaggle GPU費用を診断する。3エポック学習と公式graph評価は後続の`exp031_frame_self_attention_spatial`で扱う。

## 正の記録

- 実装契約: [`requirements.md`](requirements.md)
- 設定と系譜: [`config.yaml`](config.yaml)
- 数値と実行証拠: [`metrics.json`](metrics.json)
- 時系列ログ: [`SESSION_NOTES.md`](SESSION_NOTES.md)
- 結果と判断: [`result.md`](result.md)

## 実行入口

- Kaggle診断Notebook: [`exp030_frame_self_attention_diagnostics_diagnostic.ipynb`](exp030_frame_self_attention_diagnostics_diagnostic.ipynb)
- 編集用Jupytext source: [`exp030_frame_self_attention_diagnostics_diagnostic.py`](exp030_frame_self_attention_diagnostics_diagnostic.py)
- 変更版モデル: [`frame_attention_model.py`](frame_attention_model.py)
