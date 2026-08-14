# Biohub リポジトリ設定調査

このディレクトリには、リポジトリ初期設定の根拠を得るための一時的な調査コードと生の出力を置く。
確認済みの結論と設定判断は `docs/` を正とし、ここには実験の `metrics.json` や `result.md` を置かない。

- `input_metadata_audit.py` / `.ipynb`: Kaggle 上で画像 chunk を読まず、入力 metadata と sample submission を調べる。
- `input_metadata_audit.json`: 上記 Notebook version 1 の生の出力。
- `inspect_public_notebooks.py`: 保存済み公開 Notebook の学習・fold・評価コードを機械的に抽出する。
- `public_notebook_code_inventory.json`: 公開 Notebook コード抽出の生の出力。

入力監査を実行した private Kaggle Notebook は
`kentookumura/exp001-input-audit-diagnostic` version 1 である。名称には当初の誤分類が残るが、
この実行は実験ではなくリポジトリ設定調査として扱う。
