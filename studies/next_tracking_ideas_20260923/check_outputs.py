"""Check links and evidence references in this investigation's artifacts."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'docs/surveys/biohub-tracking-next-ideas_20260923.md'
PORTFOLIO = Path(__file__).with_name('idea_portfolio.json')
text = REPORT.read_text()
errors = []
links = 0
for target in re.findall(r'\]\(([^)]+)\)', text):
    if target.startswith(('https://', 'http://', '#')):
        continue
    links += 1
    path = target.split('#', 1)[0]
    if not (REPORT.parent / path).resolve().exists():
        errors.append(f'missing local link: {target}')
for number, line in enumerate(text.splitlines(), 1):
    if line != line.rstrip():
        errors.append(f'trailing whitespace at report line {number}')
portfolio = json.loads(PORTFOLIO.read_text())
for card in portfolio['idea_cards']:
    for evidence_id in card['evidence_ids']:
        if evidence_id not in portfolio['evidence_register']:
            errors.append(f'unknown evidence: {evidence_id}')
for evidence in portfolio['evidence_register'].values():
    for candidate in re.findall(r'(?:experiments|docs)/[^ ,、と]+', evidence):
        candidate = candidate.split('#', 1)[0]
        braces = re.search(r'\{([^}]+)\}', candidate)
        variants = (
            [candidate[:braces.start()] + name + candidate[braces.end():]
             for name in braces[1].split(',')]
            if braces else [candidate]
        )
        # A separate complete pass below handles comma-containing brace expressions.
        if '{' in candidate:
            continue
        for variant in variants:
            if not (ROOT / variant).exists():
                errors.append(f'missing evidence: {variant}')
for evidence in portfolio['evidence_register'].values():
    for base, names in re.findall(r'(experiments/[^/]+/)\{([^}]+)\}', evidence):
        for name in names.split(','):
            if not (ROOT / base / name).exists():
                errors.append(f'missing evidence: {base}{name}')
if errors:
    raise SystemExit('\n'.join(errors))
print(f'PASS: {links} local report links, evidence references, whitespace')
