"""Read public dataset metadata, excluding server-provided signed download URLs."""
import json
from datetime import datetime, timezone
from pathlib import Path
import requests

out = Path(__file__).resolve().parent
url = 'https://www.kaggle.com/api/v1/datasets/view/anvithpothula/biohub-v1284-head-s075'
r = requests.get(url, timeout=30)
r.raise_for_status()
d = r.json()
keys = ('id','ref','title','currentVersionNumber','lastUpdated','isPrivate','isFeatured','licenseName','totalBytes','versions')
selected = {k:d[k] for k in keys if k in d}
selected['checked_at_utc'] = datetime.now(timezone.utc).isoformat()
selected['endpoint'] = url
(out/'v1284_dataset_version.json').write_text(json.dumps(selected, indent=2)+'\n')
print(json.dumps(selected, indent=2))
print('Available top-level metadata fields:', sorted(d))
