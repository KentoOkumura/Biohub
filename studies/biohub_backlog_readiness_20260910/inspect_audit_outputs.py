import hashlib
import json
import sys
from pathlib import Path

exp = Path('experiments/exp003_official_metric_audit')
output = exp / 'artifacts' / (sys.argv[1] if len(sys.argv) > 1 else 'kaggle_v1')
artifacts = output / 'artifacts'
manifest = json.loads((artifacts / 'artifact_manifest.json').read_text())
for name, expected in manifest.items():
    observed = hashlib.sha256((artifacts / name).read_bytes()).hexdigest()
    assert observed == expected, (name, observed, expected)
summary = json.loads((artifacts / 'audit_summary.json').read_text())
fixtures = json.loads((artifacts / 'fixture_results.json').read_text())
samples = json.loads((artifacts / 'sample_results.json').read_text())
environment = json.loads((artifacts / 'environment.json').read_text())
failures = json.loads((artifacts / 'failures.json').read_text())
assert len(fixtures) == 9 and len(samples) == 4 and not failures
assert len({r['name'] for r in samples}) == 4
assert summary['diagnostic_only'] and summary['failed_count'] == 0
assert environment['prediction_sha256'] == 'b039062114962347aca22f1268a64196893eaabb62c1046ae0c90acbad29f223'
compact = {'summary': summary, 'environment': environment, 'verified_manifest_file_count': len(manifest)}
for label, rows in [('fixtures', fixtures), ('samples', samples)]:
    compact[label] = [{
        'name': row['name'],
        'official_counts': row['official_counts'],
        'proxy_edge_tp_fp_fn': row['proxy_edge_tp_fp_fn'],
        'proxy_division_tp_fp_fn': row['proxy_division_tp_fp_fn'],
        'official_score': row['official_summary']['score'],
        'proxy_score': row['proxy_score'],
        'difference': row['score_delta_official_minus_proxy'],
        'node_matching_difference_count': row['node_matching_symmetric_difference_count'],
        'warnings': sorted(set(row['warnings'])),
    } for row in rows]
log_paths = sorted(output.glob('*.log'))
if log_paths:
    events = json.loads(log_paths[0].read_text())
    compact['kaggle_log_last_event_seconds'] = max(float(e['time']) for e in events)
    compact['selected_log_events'] = [{'time': e['time'], 'data': e['data'][:180]} for e in events if any(t in e['data'] for t in ['Prepared ', 'Looking in links:', 'SOURCE_VERIFIED', 'FIXTURE {', 'SAMPLE {', 'AUDIT_SUMMARY'])]
(exp / 'artifacts' / (output.name + '_verification.json')).write_text(json.dumps(compact, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(compact, ensure_ascii=False, indent=2))
