import json
from pathlib import Path

packet = Path('/home/kento/work/kaggle/Biohub/studies/biohub_source_hidden_eval_20261002/packet/evidence.json')
output = Path('/home/kento/work/kaggle/Biohub/studies/biohub_source_hidden_eval_20261002/run_a/evidence_summary.json')
data = json.loads(packet.read_text(encoding='utf-8'))
skip_names = {'url', 'kernel_id', 'kernel_source_ids', 'sha256', 'sha', 'source_sha256', 'output_sha256', 'checkpoint_sha256', 'notebook_sha256', 'input_sha256', 'log_path', 'source_path', 'updated_at'}
def walk(value, path=''):
    rows = {}
    if isinstance(value, dict):
        for key, item in value.items():
            if key not in skip_names and 'sha256' not in key:
                rows.update(walk(item, f'{path}.{key}' if path else key))
    elif isinstance(value, list):
        if len(value) <= 8 and all(not isinstance(x, (dict, list)) for x in value):
            rows[path] = value
        elif len(value) <= 8:
            for i, item in enumerate(value):
                rows.update(walk(item, f'{path}[{i}]'))
        else:
            rows[path] = {'list_length': len(value), 'first_two': value[:2], 'last_one': value[-1:]}
    else:
        rows[path] = value
    return rows
summary = {'evidence_cutoff': data['evidence_cutoff'], 'interpretation': data.get('interpretation', []), 'root_keys': list(data), 'entries': []}
for entry in data['evidence']:
    summary['entries'].append({'id': entry['id'], 'source': entry.get('source'), 'fields': walk(entry.get('metrics', entry))})
for key, value in data.items():
    if key not in {'evidence', 'evidence_cutoff', 'interpretation', 'git_commit'}:
        summary[key] = value
output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False, indent=2))
