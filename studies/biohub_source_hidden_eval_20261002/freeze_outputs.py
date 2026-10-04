"""Hash saved ideas without examining or modifying their content."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--stage", choices=["initial", "final"], required=True)
args = parser.parse_args()
names = ["task_first.json"] if args.stage == "initial" else ["task_first.json", "generation_notes.json", "idea_portfolio.json", "access_log.json"]
record = {"observed_at_utc": datetime.now(timezone.utc).isoformat(), "files": {}}
for run in ("run_a", "run_b"):
    for name in names:
        path = BASE / run / name
        if not path.exists():
            raise SystemExit(f"missing: {path}")
        record["files"][f"{run}/{name}"] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()}
destination = BASE / f"{args.stage}_output_freeze.json"
if destination.exists():
    raise SystemExit(f"already frozen: {destination}")
destination.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(record, ensure_ascii=False, indent=2))
