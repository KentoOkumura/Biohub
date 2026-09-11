"""Inspect downloaded Kaggle notebook JSON as data, without executing its cells."""
import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=Path, default=Path('/tmp/biohub-baselines-20260910'))
parser.add_argument('--output', type=Path, default=Path(__file__).with_name('notebook_inventory.json'))
args = parser.parse_args()
rows = []
for path in sorted(args.input.glob('*/*.ipynb')):
    notebook = json.loads(path.read_text())
    metadata = json.loads(path.with_name('kernel-metadata.json').read_text())
    cells = notebook['cells']
    text = '\n\n'.join(''.join(cell.get('source', [])) for cell in cells)
    source = args.input / f'{path.parent.name}-source.txt'
    source.write_text(text)
    settings = re.findall(r'''os\.environ\[['"](BIOHUB_[A-Z0-9_]+)['"]\]\s*=\s*['"]([^'"\n]+)['"]''', text)
    rows.append({
        'ref': metadata['id'],
        'url': 'https://www.kaggle.com/code/' + metadata['id'],
        'title': metadata['title'],
        'id_no': metadata.get('id_no'),
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'cell_count': len(cells),
        'code_bytes': len(text.encode()),
        'dataset_sources': metadata.get('dataset_sources', []),
        'settings_static_assignments': settings,
        'has_component_based_division_proxy': 'components = weakly_connected_components(pred_node_ids, pred_edge_list)' in text,
        'validation_excludes_test_stems': 'candidates = [s for s in train_stems_all if s not in test_stem_set]' in text,
        'selected_snippets': [line[:1500] for line in text.splitlines() if any(term in line for term in ['METHOD =', 'WEIGHTS_RELATIVE =', 'BIOHUB_EDGE_FEATURE_TTA\'] =', 'BIOHUB_SECONDARY_EDGE_FEATURE_TTA_WEIGHT"] =', 'BIOHUB_DEEPCENTER_TTA"] ='])],
        'public_score': None,
        'score_note': 'Kaggle CLI metadata does not expose scores. Do not infer from titles or source comments.',
    })
args.output.write_text(json.dumps({'inspected_at': datetime.now(timezone.utc).isoformat(), 'executed_notebooks': False, 'notebooks': rows}, ensure_ascii=False, indent=2) + '\n')
for row in rows:
    print(row['ref'], row['sha256'][:12], 'component_division_proxy=', row['has_component_based_division_proxy'])
