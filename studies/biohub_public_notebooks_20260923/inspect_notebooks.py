"""Read public notebooks as JSON; never execute their code."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / 'docs/notebooks/biohub-cell-tracking-during-development'
REFS = [
    'anvithpothula/biohub-x138',
    'amanatar/biohub-geometric-fusion',
    'evgendvorkin/biohub-0-947-lb-proxy-score-0-9490',
    'raunakdey07/biohub-harmonic-fusion-v3',
    'flexonafft/biohub-harmonic-fusion',
    'reyhanksatria/biohub-cell-tracking-0-947-lb',
    'flexonafft/biohub-lineage-forge-precision-tracking',
]

inventory = []
for ref in REFS:
    folder = SOURCE / ref.replace('/', '__')
    meta = json.loads((folder / 'kernel-metadata.json').read_text())
    source = folder / meta['code_file']
    nb = json.loads(source.read_text())
    code, markdown, outputs = [], [], []
    for index, cell in enumerate(nb['cells']):
        text = ''.join(cell.get('source', []))
        (code if cell['cell_type'] == 'code' else markdown).append(f'\n# CELL {index}\n{text}\n')
        for output in cell.get('outputs', []):
            outputs.append(''.join(output.get('text', [])))
            plain = output.get('data', {}).get('text/plain', [])
            outputs.append(''.join(plain) if isinstance(plain, list) else plain)
    stem = ref.replace('/', '__')
    for suffix, chunks in [('code.py', code), ('markdown.txt', markdown), ('outputs.txt', outputs)]:
        (OUT / f'{stem}.{suffix}').write_text('\n'.join(chunks))
    inventory.append({
        'ref': ref, 'source': str(source.relative_to(ROOT)),
        'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'metadata': meta, 'notebook_metadata': nb.get('metadata', {}),
        'cells': len(nb['cells']), 'code_lines': len('\n'.join(code).splitlines()),
        'output_chars': sum(map(len, outputs)),
    })
    print(ref, len(nb['cells']), 'cells', len('\n'.join(code).splitlines()), 'code lines')
(OUT / 'notebook_inventory.json').write_text(json.dumps(inventory, indent=2, ensure_ascii=False) + '\n')
