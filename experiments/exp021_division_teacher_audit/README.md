# exp021_division_teacher_audit

## 概要

- 仮説要約: 固定した母・2娘の候補に、疎いGEFFの既知edgeから教師に使える誤組がどれだけあるか調べる。
- 変更点要約: exp020の9/14 µm候補を既知分裂正例、既知分裂母の誤組、別母に注釈接続された娘を含む組、単一edge母の未確定な組に分ける。
- リスク: 単一edge母から真の第2娘不在を確定できない。別母edgeによる誤組も注釈とGT対応の正しさに依存する。
- 次: 全件診断とユーザーの完了判断は記録済み。[教師の不足と判断](result.md)を後続の教師設計へ引き継ぐ。

## 正の記録

- 数値と実行証拠: [metrics.json](metrics.json)
- 設定と系譜: [config.yaml](config.yaml)
- 契約と受け入れ条件: [requirements.md](requirements.md)
- 結果の解釈: [result.md](result.md)
- 実行中の記録: [SESSION_NOTES.md](SESSION_NOTES.md)

## 実行入口

- 診断Notebook: [exp021_division_teacher_audit_diagnostic.ipynb](exp021_division_teacher_audit_diagnostic.ipynb)
- Kaggle CPU Notebookを正とする。元の[division_triplets候補](../../backlog/division_triplets.md)は学習設計まで未移行として残す。
