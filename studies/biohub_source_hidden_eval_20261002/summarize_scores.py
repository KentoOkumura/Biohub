"""Recalculate mechanism counts from the independent judge's per-row ratings."""

import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
payload = json.loads((BASE / "judge/scores.json").read_text())
rubric = json.loads((BASE / "effective_rubric.json").read_text())
policies = {entry["id"]: entry["policy"] for entry in rubric["criteria"]}
rows = payload["criteria"]
if len(rows) != len(policies) or set(row["id"] for row in rows) != set(policies):
    raise SystemExit("judge criteria differ from frozen effective rubric")
summary = {"criterion_count": len(rows), "runs": {}, "union": {}}
for run in ("run_a", "run_b"):
    summary["runs"][run] = {}
    for stage in ("task_first", "all_cards", "top_five"):
        ratings = {row["id"]: row["runs"][run][stage] for row in rows}
        if any(type(value) is not int or value not in {0, 1, 2} for value in ratings.values()):
            raise SystemExit(f"invalid rating: {run}/{stage}")
        summary["runs"][run][stage] = {}
        for scope in ("all", "within_policy", "change_required"):
            scoped = {key: value for key, value in ratings.items() if scope == "all" or policies[key] == scope}
            summary["runs"][run][stage][scope] = {
                "total": len(scoped),
                "precise": [key for key, value in scoped.items() if value == 2],
                "partial": [key for key, value in scoped.items() if value == 1],
                "absent": [key for key, value in scoped.items() if value == 0],
            }
for stage in ("task_first", "all_cards", "top_five"):
    summary["union"][stage] = {row["id"]: max(row["runs"][run][stage] for run in ("run_a", "run_b")) for row in rows}
(BASE / "score_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(summary, ensure_ascii=False, indent=2))
