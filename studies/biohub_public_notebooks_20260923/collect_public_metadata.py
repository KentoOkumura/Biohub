"""Collect public read-only Kaggle metadata without credentials or signed URLs."""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
inventory = json.loads((OUT / 'notebook_inventory.json').read_text())
session = requests.Session()
rows = []
for item in inventory:
    response = session.post(
        'https://www.kaggle.com/api/i/kernels.KernelsService/GetKernel',
        json={'kernelId': item['metadata']['id_no']}, timeout=30,
    )
    response.raise_for_status()
    detail = response.json()
    row = {key: detail.get(key) for key in (
        'id', 'title', 'currentRunId', 'mostRecentRunId', 'url',
        'bestPublicScore', 'updatedTime', 'hasLinkedSubmission',
    )}
    row['ref'] = item['ref']
    row['checked_at_utc'] = datetime.now(timezone.utc).isoformat()
    if item['ref'].startswith(('anvithpothula/', 'amanatar/')):
        version_response = session.post(
            'https://www.kaggle.com/api/i/kernels.KernelsService/GetKernelVersion',
            json={'kernelSessionId': detail['currentRunId']}, timeout=30,
        )
        version_response.raise_for_status()
        value = version_response.json()['kernelVersionAndRun']
        row['version'] = value['version']
        run = value['run']
        row['run'] = {key: run.get(key) for key in (
            'id', 'kernelId', 'status', 'dateCreated', 'dateEvaluated',
        )}
        # The location of execution result fields can vary between API versions.
        def selected_evidence(obj):
            result = []
            if isinstance(obj, dict):
                chosen = {k: v for k, v in obj.items() if k in (
                    'runTimeSeconds', 'succeeded', 'publicScore', 'bestPublicScore',
                )}
                if chosen:
                    result.append(chosen)
                for child in obj.values():
                    result.extend(selected_evidence(child))
            elif isinstance(obj, list):
                for child in obj:
                    result.extend(selected_evidence(child))
            return result
        row['execution_evidence'] = selected_evidence(value)
    rows.append(row)
    print(item['ref'], row['bestPublicScore'], row.get('version', {}).get('versionNumber'))
(OUT / 'public_scores.json').write_text(json.dumps(rows, indent=2, ensure_ascii=False) + '\n')

receipt = json.loads((OUT / 'x138_output/bidirectional_production_runtime_integrity.json').read_text())
baseline = json.loads((ROOT / 'experiments/exp011_public_detector_selection/assets/public_detector_selection.json').read_text())
expected = dict(zip(('primary', 'secondary', 'deepcenter'),
                    [item['checkpoint_sha256'] for item in baseline['canonical_public_artifacts']]))
assert receipt['checkpoint_sha256'] == expected
stats = list(csv.DictReader((OUT / 'x138_output/run_stats.csv').open()))
summary_path = OUT / 'x138_execution_summary.json'
summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
summary.update({'checkpoint_sha_matches_exp011': True, 'checkpoint_comparison_scope': ['primary', 'secondary', 'deepcenter'], 'visible_test_stats': stats})
(OUT / 'x138_execution_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
print('All three checkpoint SHAs match exp011. Visible test rows:', len(stats))

# Remove the temporary full API response: the retained evidence above excludes signed URLs.
raw_response = OUT / 'x138_version.json'
if raw_response.exists():
    raw_response.unlink()
