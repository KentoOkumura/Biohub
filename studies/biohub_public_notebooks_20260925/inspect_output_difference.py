"""Summarize public execution evidence without evaluating notebook code."""
import csv
import json
import re
from pathlib import Path

OUT = Path(__file__).resolve().parent
OLD = OUT.parent/'biohub_public_notebooks_20260923'
a = list(csv.DictReader((OLD/'x138_output/run_stats.csv').open()))
b = list(csv.DictReader((OUT/'kunal_output/run_stats.csv').open()))
differences = [{k:[x.get(k),y.get(k)] for k in set(x)|set(y) if x.get(k) != y.get(k)} for x,y in zip(a,b)]
print(json.dumps(differences, indent=2))
logs = {'x138':OLD/'x138_run.log','kunal':OUT/'kunal_run.log'}
submissions = {}
for name, path in logs.items():
    text = path.read_text()
    block = re.search(r'"submission":\s*\{(.*?)\n  \}', text, re.S)
    submissions[name] = json.loads('{'+block.group(1)+'}')
comparison_path = OUT/'source_comparison.json'
d = json.loads(comparison_path.read_text())
d['visible_test_run_stats_differences'] = differences
d['log_submission_receipts'] = submissions
d['log_submission_receipts_equal'] = submissions['x138'] == submissions['kunal']
comparison_path.write_text(json.dumps(d, indent=2)+'\n')
print(json.dumps(submissions,indent=2))
