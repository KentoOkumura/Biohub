import json
import re
import sys
from pathlib import Path

data = json.loads(Path('/home/kento/work/kaggle/Biohub/studies/biohub_source_hidden_eval_20261002/packet/evidence.json').read_text(encoding='utf-8'))
ids = set(sys.argv[1].split(',')) if len(sys.argv) > 1 else set()
pattern = re.compile(sys.argv[2]) if len(sys.argv) > 2 else None
def leafs(x, p=''):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from leafs(v, f'{p}.{k}' if p else k)
    elif isinstance(x, list):
        yield p, x
    else:
        yield p, x
for entry in data['evidence']:
    if ids and entry['id'] not in ids:
        continue
    print(json.dumps({'id': entry['id'], 'experiment': entry['metrics'].get('experiment'), 'matched_fields': {p: v for p, v in leafs(entry['metrics']) if (pattern.search(p) if pattern else p.count('.') < 3)}}, ensure_ascii=False, indent=2))
