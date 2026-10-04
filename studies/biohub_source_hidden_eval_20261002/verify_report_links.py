"""Check local links only in the completed evaluation documents."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
from check_markdown_links import LINK_PATTERN, normalized_target  # noqa: E402

paths = [
    ROOT / "docs/surveys/biohub-source-hidden-idea-evaluation_20261002.md",
    BASE / "README.md",
    BASE / "judge/readout.md",
]
errors = []
checked = 0
for path in paths:
    content = path.read_text()
    for match in LINK_PATTERN.finditer(content):
        target = normalized_target(match.group(1))
        if target is not None:
            checked += 1
            if not (path.parent / target).resolve().exists():
                errors.append(f"{path.relative_to(ROOT)}: missing {target}")
    for number, line in enumerate(content.splitlines(), 1):
        if line.rstrip() != line:
            errors.append(f"{path.relative_to(ROOT)}:{number}: trailing whitespace")
result = {"files": [str(path.relative_to(ROOT)) for path in paths],
          "local_links_checked": checked, "errors": errors}
(BASE / "document_verification.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + "\n"
)
print(json.dumps(result, ensure_ascii=False, indent=2))
if errors:
    raise SystemExit(1)
