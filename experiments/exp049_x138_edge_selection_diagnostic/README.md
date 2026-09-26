# exp049_x138_edge_selection_diagnostic

## 概要

- 仮説要約: exp047で増やした正しい接続が最終graphへ届かない段階を、辺IDで特定できる。
- 変更点要約: 固定得点の順位、候補から後処理までの辺遷移、11事象の正解固定ILPを診断した。予測や公式評価は変更していない。
- リスク: GEFFの部分注釈とhead学習外20動画の独立性の限界がある。
- 次: [結果](result.md)を基に、得点とILPのどちらを次の実験で扱うか判断する。

## 正の記録

[要件と実装方法](requirements.md) · [設定と系譜](config.yaml) · [数値と実験status](metrics.json) · [結果の解釈](result.md) · [実行ログ](SESSION_NOTES.md)

## 実行入口

診断Notebook: [exp049_x138_edge_selection_diagnostic_diagnostic.ipynb](exp049_x138_edge_selection_diagnostic_diagnostic.ipynb)。正解固定ILPは診断出力だけに保存し、公式scoreとして使わない。
