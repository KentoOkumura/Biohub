# exp033_frame_self_attention_diagnostics

## 概要

保存済みのexp016現行trackerとexp025のModel A、Model B、恒等初期化Bを、同じ固定2-frame windowと教師で比較する診断。確率0.5の再現後、PR曲線、precision 0.95でのrecall、候補数・近傍距離・移動距離・分裂別の誤りを保存する。再学習、graph復元、submissionはしない。

- 変更点: pair確率とID・座標をsample単位で保存し、順位と条件別の誤りを集計する。
- リスク: GEFFの部分注釈、公開画像モデルの学習来歴、全pair保存と再推論の費用。
- 次: [完了判断と診断の限界](result.md)を後続の構造変更の検討へ引き継ぐ。Self-Attention構成の採否は未判断。

## 正の記録

- [契約](requirements.md)
- [設定と系譜](config.yaml)
- [数値と実行証拠](metrics.json)
- [実行中の記録](SESSION_NOTES.md)
- [結果と判断](result.md)

## 実行入口

診断Notebook: [exp033_frame_self_attention_diagnostics_diagnostic.ipynb](exp033_frame_self_attention_diagnostics_diagnostic.ipynb)。
