"""Inspect a downloaded tensor checkpoint using PyTorch weights_only mode."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
p = ROOT/'data/external/biohub-v1284-head-s075/v1284_head.pt'
value = torch.load(p, map_location='cpu', weights_only=True)
def describe(obj):
    if isinstance(obj, torch.Tensor):
        return {'shape':list(obj.shape), 'dtype':str(obj.dtype), 'finite':bool(torch.isfinite(obj).all()), 'min':float(obj.min()), 'max':float(obj.max())}
    if isinstance(obj, dict):
        return {str(k):describe(v) for k,v in obj.items()}
    if isinstance(obj, (str,int,float,bool)) or obj is None:
        return obj
    if isinstance(obj,(list,tuple)):
        return [describe(v) for v in obj]
    return {'type':type(obj).__name__}
row = {'checked_at_utc':datetime.now(timezone.utc).isoformat(), 'dataset_ref':'anvithpothula/biohub-v1284-head-s075', 'file':str(p.relative_to(ROOT)), 'size_bytes':p.stat().st_size, 'sha256':hashlib.sha256(p.read_bytes()).hexdigest(), 'load_mode':'torch.load(map_location=cpu, weights_only=True)', 'parameter_count':sum(t.numel() for t in value['state_dict'].values()), 'contents':describe(value)}
(OUT/'v1284_checkpoint_inspection.json').write_text(json.dumps(row, indent=2)+'\n')
print(json.dumps(row, indent=2))
