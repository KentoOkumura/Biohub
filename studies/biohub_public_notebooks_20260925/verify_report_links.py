"""Validate local Markdown targets in this investigation report."""
import re
from pathlib import Path

root = Path(__file__).resolve().parents[2]
report = root/'docs/surveys/biohub-public-notebooks-followup_20260925.md'
links = re.findall(r'\]\(([^)]+)\)', report.read_text())
local = [link.split('#',1)[0] for link in links if not link.startswith(('https://','http://','#'))]
missing = [link for link in local if not (report.parent/link).exists()]
if missing:
    raise SystemExit(f'Missing report targets: {missing}')
print(f'All {len(local)} local report links resolve.')
