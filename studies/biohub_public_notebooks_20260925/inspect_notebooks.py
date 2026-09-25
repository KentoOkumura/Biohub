"""Statically inspect downloaded public notebooks; do not execute notebook code."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / 'docs/notebooks/biohub-cell-tracking-during-development/20260925'
rows = []
for folder in sorted(SOURCE.iterdir()):
    if not folder.is_dir():
        continue
    meta = json.loads((folder / 'kernel-metadata.json').read_text())
    source = folder / meta['code_file']
    nb = json.loads(source.read_text())
    chunks = {'code.py': [], 'markdown.txt': [], 'outputs.txt': []}
    for i, cell in enumerate(nb['cells']):
        suffix = 'code.py' if cell['cell_type'] == 'code' else 'markdown.txt'
        chunks[suffix].append(f'\n# CELL {i}\n' + ''.join(cell.get('source', [])))
        for out in cell.get('outputs', []):
            text = out.get('text', out.get('data', {}).get('text/plain', []))
            chunks['outputs.txt'].append(''.join(text) if isinstance(text, list) else text)
    for suffix, parts in chunks.items():
        (OUT / f'{folder.name}.{suffix}').write_text('\n'.join(parts))
    row = {'ref': meta['id'], 'metadata': meta, 'source': str(source.relative_to(ROOT)), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'cells': len(nb['cells']), 'code_lines': len('\n'.join(chunks['code.py']).splitlines())}
    rows.append(row)
    print(row['ref'], row['cells'], row['code_lines'], row['sha256'])
(OUT / 'notebook_inventory.json').write_text(json.dumps(rows, indent=2) + '\n')
