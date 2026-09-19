# exp020_division_triplet_candidates

## 概要

- 仮説要約: 固定中心候補から母と2娘の組を作ると、現行candidate edgeに限る61/151件より多くの既知分裂を回収できるか調べる。
- 変更点要約: GTを使わずに隣接2時刻の組候補を列挙し、GTは診断集計だけに使う。
- リスク: 疎いGEFFから第2娘不在を確定できず、学習用の通常継続負例はまだ定義しない。
- 次: 診断で得た教師の不足を踏まえ、学習用mask・loss・decodeの判断を行う。

## 正の記録

- 数値と実行証拠: [metrics.json](metrics.json)
- 設定と系譜: [config.yaml](config.yaml)
- 契約と受け入れ条件: [requirements.md](requirements.md)
- 結果の解釈: [result.md](result.md)
- 実行中の記録: [SESSION_NOTES.md](SESSION_NOTES.md)

## 実行入口

- 診断Notebook: [exp020_division_triplet_candidates_diagnostic.ipynb](exp020_division_triplet_candidates_diagnostic.ipynb)
- 実行はKaggle CPU Notebookを正とする。元の[division_triplets候補](../../backlog/division_triplets.md)は学習設計まで未移行として残す。
