"""Read downloaded public notebooks without executing their cells."""

import hashlib
import json
import re
from pathlib import Path

ROOT = Path('/tmp/biohub-gnn-review')
PATTERN = re.compile(
    r'GNN|GCN|GAT|GraphSAGE|MessagePassing|torch_geometric|'
    r'graph neural|message.passing|SimpleNodeTransformer|Trackastra|'
    r'edge_index|def (train|forward)|class |load_state_dict|torch.load',
    re.I,
)

for path in sorted(ROOT.rglob('*.ipynb')):
    notebook = json.loads(path.read_text())
    source = '\n\n'.join(''.join(c.get('source', [])) for c in notebook['cells'])
    output = '\n'.join(
        ''.join(o.get('text', []))
        for cell in notebook['cells']
        for o in cell.get('outputs', [])
    )
    path.with_suffix('.source.txt').write_text(source)
    path.with_suffix('.outputs.txt').write_text(output)
    hits = [f'{i}: {line[:220]}' for i, line in enumerate(source.splitlines(), 1) if PATTERN.search(line)]
    print(json.dumps({
        'file': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'cells': len(notebook['cells']), 'hits': hits[:70],
        'output_tail': output[-900:],
    }, ensure_ascii=False, indent=2))
