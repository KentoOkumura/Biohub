"""Verify input provenance, frozen criteria, and generator output structure."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    manifest = json.loads((BASE / "input_manifest.json").read_text())
    errors = []
    for path, expected in manifest["packet_sha256"].items():
        if digest(BASE / path) != expected:
            errors.append(f"packet changed: {path}")
    for source in manifest["sources"]:
        if source["git_commit"] == "current file":
            raw = (ROOT / source["path"]).read_bytes()
        else:
            raw = subprocess.check_output(
                ["git", "show", f"{source['git_commit']}:{source['path']}"], cwd=ROOT
            )
        if hashlib.sha256(raw).hexdigest() != source["source_sha256"]:
            errors.append(f"source changed: {source['path']}")
    protocol_path = BASE / "protocol.json"
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text())
        if digest(BASE / "withheld_rubric.json") != protocol["rubric_sha256"]:
            errors.append("rubric changed after protocol freeze")
        if "effective_rubric_sha256" in protocol:
            if digest(BASE / "effective_rubric.json") != protocol["effective_rubric_sha256"]:
                errors.append("effective rubric changed after source correction freeze")
    for freeze_name in ("initial_output_freeze.json", "final_output_freeze.json"):
        freeze_path = BASE / freeze_name
        if freeze_path.exists():
            frozen = json.loads(freeze_path.read_text())
            for path, record in frozen["files"].items():
                if digest(BASE / path) != record["sha256"]:
                    errors.append(f"output changed after {freeze_name}: {path}")
    module_path = ROOT / ".agents/skills/kaggle-idea-forge/scripts/validate_portfolio.py"
    spec = importlib.util.spec_from_file_location("portfolio_validator", module_path)
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    outputs = {}
    for run in ("run_a", "run_b"):
        portfolio_path = BASE / run / "idea_portfolio.json"
        if not portfolio_path.exists():
            outputs[run] = {"state": "not_saved"}
            continue
        payload = json.loads(portfolio_path.read_text())
        run_errors = validator.validate(payload)
        errors.extend(f"{run}: {error}" for error in run_errors)
        files = {}
        for name in ("task_first.json", "generation_notes.json", "idea_portfolio.json", "access_log.json"):
            path = BASE / run / name
            if not path.exists():
                errors.append(f"{run}: missing {name}")
            else:
                json.loads(path.read_text())
                files[name] = digest(path)
        outputs[run] = {"state": "saved", "card_count": len(payload["idea_cards"]), "top_five": [item["idea_id"] for item in payload["portfolio"]], "files_sha256": files, "schema_errors": run_errors}
    scoring = {"state": "not_saved"}
    scores_path = BASE / "judge/scores.json"
    if scores_path.exists():
        scores = json.loads(scores_path.read_text())
        rubric = json.loads((BASE / "effective_rubric.json").read_text())
        expected = {entry["id"]: entry for entry in rubric["criteria"]}
        rows = scores["criteria"]
        if len(rows) != len(expected) or {row["id"] for row in rows} != set(expected):
            errors.append("score rows differ from frozen rubric")
        if scores["effective_rubric_sha256"] != digest(BASE / "effective_rubric.json"):
            errors.append("scores refer to a different effective rubric")
        checked_quotes = 0
        for row in rows:
            if row["id"] not in expected:
                continue
            criterion = expected[row["id"]]
            if row["policy"] != criterion["policy"] or row["mechanism"] != criterion["mechanism"]:
                errors.append(f"score criterion changed: {row['id']}")
            for run in ("run_a", "run_b"):
                assessment = row["runs"][run]
                initial = json.loads((BASE / run / "task_first.json").read_text())
                final = json.loads((BASE / run / "idea_portfolio.json").read_text())
                selected = {entry["idea_id"] for entry in final["portfolio"]}
                for citation in assessment["evidence"]:
                    permitted = {f"{run}/task_first.json", f"{run}/idea_portfolio.json"}
                    if citation["path"] not in permitted:
                        errors.append(f"citation outside run: {row['id']}/{run}")
                        continue
                    is_initial = citation["path"].endswith("task_first.json")
                    ideas = initial["initial_ideas"] if is_initial else final["idea_cards"]
                    idea = next((entry for entry in ideas if entry["id"] == citation["idea_id"]), None)
                    if idea is None or citation["field"] not in idea:
                        errors.append(f"citation field missing: {row['id']}/{run}")
                        continue
                    value = idea[citation["field"]]
                    source_text = "\n".join(value) if isinstance(value, list) else value
                    if citation["quote"] not in source_text:
                        errors.append(f"citation text differs: {row['id']}/{run}")
                    checked_quotes += 1
                    for stage in citation["stages"]:
                        if (stage == "task_first") != is_initial:
                            errors.append(f"citation stage mismatch: {row['id']}/{run}")
                        if stage == "top_five" and citation["idea_id"] not in selected:
                            errors.append(f"unselected top-five citation: {row['id']}/{run}")
                for stage in ("task_first", "all_cards", "top_five"):
                    rating = assessment[stage]
                    if type(rating) is not int or rating not in {0, 1, 2}:
                        errors.append(f"invalid rating: {row['id']}/{run}/{stage}")
                    if rating and not any(stage in entry["stages"] for entry in assessment["evidence"]):
                        errors.append(f"nonzero rating without evidence: {row['id']}/{run}/{stage}")
                    if rating == 1 and not any(stage in entry["stages"] for entry in assessment["missing_defining_steps"]):
                        errors.append(f"partial rating without missing steps: {row['id']}/{run}/{stage}")
        scoring = {"state": "saved", "criterion_count": len(rows), "quotes_checked": checked_quotes,
                   "scores_sha256": digest(scores_path)}
    result = {"provenance_verified": not errors, "errors": errors, "outputs": outputs, "scoring": scoring}
    (BASE / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
