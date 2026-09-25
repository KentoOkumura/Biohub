"""Collect read-only public metadata, retaining no signed URLs or credentials."""
import json
from datetime import datetime, timezone
from pathlib import Path
import requests

OUT = Path(__file__).resolve().parent
s = requests.Session()
rows = []
for item in json.loads((OUT/'notebook_inventory.json').read_text()):
    r = s.post('https://www.kaggle.com/api/i/kernels.KernelsService/GetKernel', json={'kernelId': item['metadata']['id_no']}, timeout=40)
    r.raise_for_status()
    d = r.json()
    row = {k: d.get(k) for k in ('id','title','currentRunId','mostRecentRunId','url','bestPublicScore','updatedTime','hasLinkedSubmission')}
    row.update(ref=item['ref'], checked_at_utc=datetime.now(timezone.utc).isoformat())
    r = s.post('https://www.kaggle.com/api/i/kernels.KernelsService/GetKernelVersion', json={'kernelSessionId': d['currentRunId']}, timeout=40)
    r.raise_for_status()
    full = r.json()
    val = full['kernelVersionAndRun']
    row['version'] = {k: val['version'].get(k) for k in ('id','versionNumber','kernelId')}
    row['run'] = {k: val['run'].get(k) for k in ('id','kernelId','status','dateCreated','dateEvaluated')}
    def scan(obj):
        hits = []
        if isinstance(obj, dict):
            hit = {k:v for k,v in obj.items() if k in ('runTimeSeconds','succeeded','publicScore','bestPublicScore','sourceType','sourceId','datasetId','mountSlug','versionNumber','isPrivate')}
            if hit:
                hits.append(hit)
            for v in obj.values():
                hits.extend(scan(v))
        elif isinstance(obj, list):
            for v in obj:
                hits.extend(scan(v))
        return hits
    row['execution_and_attachment_evidence'] = scan(full)
    rows.append(row)
    print(row['ref'], row['bestPublicScore'], row['version'], flush=True)
(OUT/'public_scores.json').write_text(json.dumps(rows, indent=2) + '\n')
