from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
# Raw external Markdown is stored as text: links are relative to the upstream
# repository, not this investigation directory. Preserve the original bytes/SHA.
renames = {"focus_readme.md": "focus_readme.md.txt", "latest_metrics.md": "latest_metrics.md.txt"}
for old, new in renames.items():
    source, target = HERE / old, HERE / new
    if source.exists():
        assert not target.exists()
        source.rename(target)
manifest_path = HERE / "external_sources.json"
manifest = json.loads(manifest_path.read_text())
for item in manifest["sources"]:
    item["name"] = renames.get(item["name"], item["name"])
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
probe_path = HERE / "inspect_and_probe.py"
text = probe_path.read_text()
for old, new in renames.items():
    text = text.replace('"' + old + '"', '"' + new + '"')
probe_path.write_text(text)
sys.path.insert(0, str(ROOT))
from scripts import check_markdown_links as checker
paths = sorted((ROOT / "backlog").glob("*.md"))
paths += [ROOT / "docs/surveys/biohub-backlog-readiness_20260910.md", ROOT / "docs/surveys/biohub-accuracy-hypotheses_20260910.md", ROOT / "docs/surveys/README.md"]
all_errors = checker.broken_links()
changed_prefixes = tuple(str(p.relative_to(ROOT)) + ":" for p in paths)
changed_errors = [error for error in all_errors if error.startswith(changed_prefixes)]
report = {"checked_document_count": len(paths), "changed_document_link_errors": changed_errors, "unrelated_existing_link_errors": [e for e in all_errors if e not in changed_errors]}
(HERE / "document_validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
assert not changed_errors, changed_errors
print(json.dumps(report, ensure_ascii=False, indent=2))
