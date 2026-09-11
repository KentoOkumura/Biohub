# exp003_official_metric_audit

## 概要

固定した人工graphと既存予測に、完全な公式評価と公開Notebookの評価関数を適用し、点の対応・接続・分裂・集計の差を確認する実験。

Kaggle CPU実行と結果回収を終えた。実予測では一致し、人工例で差を再現した。解釈・制約・推奨する判断は[result](result.md)、数値と実行状態は[metrics](metrics.json)を参照する。

## 正の記録

- [要件と承認](requirements.md)
- [設定・系譜](config.yaml)
- [実行経緯](SESSION_NOTES.md)

## 実行入口

- [監査Notebook](exp003_official_metric_audit_audit.ipynb)

次は合意済みの胚を分けた基準予測を準備する。
