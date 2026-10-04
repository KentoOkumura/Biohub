import json
import sys
from pathlib import Path

source = Path('/home/kento/work/kaggle/Biohub/studies/biohub_source_hidden_eval_20261002/packet/evidence.json')
data = json.loads(source.read_text(encoding='utf-8'))
def summarize(value):
    if isinstance(value, dict):
        return {k: summarize(v) for k, v in value.items()}
    if isinstance(value, list) and len(value) > 10:
        return {'_list_length': len(value), '_first_6': [summarize(v) for v in value[:6]], '_last_2': [summarize(v) for v in value[-2:]]}
    if isinstance(value, list):
        return [summarize(v) for v in value]
    return value

if len(sys.argv) > 1 and sys.argv[1] == 'overview':
    for item in data['evidence']:
        print(item['id'], item['metrics']['experiment'])
        def shape(value, path='', depth=0):
            if isinstance(value, dict):
                if depth >= 4:
                    print(path, 'keys=', ','.join(value.keys()))
                else:
                    for k, v in value.items():
                        shape(v, path+'.'+k, depth+1)
            elif isinstance(value, list):
                print(path, 'list', len(value))
            else:
                print(path, str(value))
        shape(item['metrics'])
elif len(sys.argv) > 1:
    item = next(e for e in data['evidence'] if e['id'] == sys.argv[1])
    if len(sys.argv) > 2:
        value = item
        for key in sys.argv[2].split('.'):
            value = value[key]
        print(json.dumps(summarize(value), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(summarize(item), ensure_ascii=False, indent=2))
else:
    summary = summarize(data)
    Path('/home/kento/work/kaggle/Biohub/studies/biohub_source_hidden_eval_20261002/run_b/evidence_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
