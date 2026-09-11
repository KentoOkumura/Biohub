import argparse
from datetime import datetime
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('message')
args = parser.parse_args()
path = Path('experiments/exp003_official_metric_audit/SESSION_NOTES.md')
with path.open('a') as stream:
    stream.write(f'\n- {datetime.now().astimezone().isoformat()}: {args.message}\n')
print('Recorded audit event.')
