from pathlib import Path

exp = Path('experiments/exp003_official_metric_audit')
p = exp / 'README.md'
s = p.read_text().replace('\n固定した人工graph', '\n## 概要\n\n固定した人工graph', 1)
s = s.replace('\n- [要件と承認]', '\n## 正の記録\n\n- [要件と承認]', 1)
s = s.replace('- [監査Notebook]', '\n## 実行入口\n\n- [監査Notebook]', 1)
p.write_text(s)
p = exp / 'result.md'
s = p.read_text().replace('## 結論', '## 仮説\n\n同じ予測graphに適用する評価関数だけを変更すると、公開Notebookと固定公式版の採点差を再現できる。モデルと入力予測は変更しない。\n\n## 解釈', 1)
s = s.replace('## ユーザー判断と次の作業', '## ユーザー判断', 1)
s = s.replace('\n既に合意した順序の次は、', '\n## 次の作業\n\n既に合意した順序の次は、', 1)
p.write_text(s)
