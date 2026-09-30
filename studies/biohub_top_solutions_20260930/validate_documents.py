"""Validate only the documents produced by this investigation."""
import json
import sys
from pathlib import Path
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from check_markdown_links import LINK_PATTERN, normalized_target
files = [ROOT / 'docs/surveys/biohub-final-top-solutions-comparison_20260930.md', ROOT / 'docs/surveys/README.md']
files += [ROOT / row['archive'] for row in json.loads((OUT / 'archive_inventory.json').read_text())]
failures = []
for path in files:
    text = path.read_text()
    for match in LINK_PATTERN.finditer(text):
        target = normalized_target(match.group(1))
        if target and not (path.parent / target).resolve().exists():
            failures.append(f'{path.relative_to(ROOT)}: {target}')
assert not failures, '\n'.join(failures)
print('Changed-document local links passed:', len(files), 'files')
report = files[0].read_text()
assert 'status: final' in report and 'TODO' not in report
assert report.count('| 3 | yu4u | 0.967') == 1
assert '430位／4,020チーム' in report
print('Report scope, rank and finalized metadata checked')
