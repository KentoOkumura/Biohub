"""Read saved evidence only; never import experiment modules or execute notebooks.

Run from the repository root with uv run python <this file>.
Outputs are a dated investigation snapshot, not experiment metrics/status updates.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
USED: set[Path] = set()


def read(path: Path) -> str:
    USED.add(path)
    return path.read_text()


def write_json(name: str, value: object) -> None:
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    inventory = []
    for path in sorted((ROOT / "experiments").glob("exp*/metrics.json")):
        metrics = json.loads(read(path))
        folder = path.parent
        number = int(folder.name[3:6])
        if number >= 16:
            for name in ("result.md", "requirements.md", "config.yaml", "SESSION_NOTES.md"):
                if (folder / name).exists():
                    read(folder / name)
        inventory.append({
            "experiment": folder.name,
            "status": metrics.get("status"),
            "metric": metrics.get("metric"),
            "public_lb": metrics.get("public_lb"),
            "notes": metrics.get("notes"),
            "metrics_path": str(path.relative_to(ROOT)),
        })
    write_json("experiment_inventory.json", inventory)

    pair_rows = []
    bins = []
    for experiment in ("exp036_detection_score_pair_features", "exp039_dog_features"):
        path = ROOT / "experiments" / experiment / "metrics.json"
        data = json.loads(read(path))
        for fold in data["train_stage"]["folds"]:
            value = fold["trained_outer_evaluation"]
            row = {
                "experiment": experiment,
                "variant": fold["variant"],
                "evaluation_embryo": fold["evaluation_embryo"],
                "selected_epoch": fold["best_epoch"],
                "positive_edge_count": value["positive_edge_count"],
                "recovered_positive_edge_count": round(value["positive_edge_recall"] * value["positive_edge_count"]),
                "positive_edge_recall": value["positive_edge_recall"],
                "division_parent_count": value["division_parent_count"],
                "recovered_division_parent_count": round(value["division_parent_recall"] * value["division_parent_count"]),
                "edge_accuracy": value["edge_accuracy"],
                "known_wrong_edge_count": value.get("known_wrong_edge_count"),
                "known_wrong_sibling_pair_count": value.get("known_wrong_sibling_pair_count"),
            }
            score_bins = value.get("detection_score_diagnostics", {}).get("bins", [])
            if score_bins:
                row["legacy_mask_false_positive_pairs"] = sum(x["false_positive_pair_count"] for x in score_bins)
                for group in score_bins:
                    bins.append({"variant": fold["variant"], "evaluation_embryo": fold["evaluation_embryo"], **group})
            pair_rows.append(row)
    write_json("pair_metrics.json", pair_rows)
    with (OUT / "exp036_score_bins.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(bins[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(bins)

    exp043 = ROOT / "experiments/exp043_x138_self_trained_head"
    stats_path = exp043 / "artifacts/inference_v2/run_stats.csv"
    stats = list(csv.DictReader(read(stats_path).splitlines()))
    total_keys = [
        "raw_nodes", "nodes", "raw_edges", "edges", "readmitted_nodes", "gapfill_added_nodes",
        "gap_added_nodes", "gap2_added_nodes", "short_track_nodes_removed",
        "motion_relink_replaced_raw_edges", "motion_relink_fallback_raw", "motion_relink_skipped_large_frame",
    ]
    totals = {key: sum(int(float(row.get(key) or 0)) for row in stats) for key in total_keys}
    receipt = json.loads(read(exp043 / "metrics.json"))
    assert totals["nodes"] + totals["edges"] == receipt["evidence"]["public_test_inference"]["rows"]
    write_json("exp043_public_test_readout.json", {
        "source": str(stats_path.relative_to(ROOT)),
        "datasets": [row["dataset"] for row in stats],
        "totals": totals,
        "limits": [
            "Public test has no GT here: recovery counts do not establish correctness.",
            "readmitted_nodes and gapfill_added_nodes are intermediate counts, not surviving final nodes.",
            "motion_relink_replaced_raw_edges counts input edges to reassignment, not changed edge identities.",
            "Submission rows contain both nodes and edges.",
        ],
        "coordinate_validation": receipt["evidence"]["coordinate_validation"],
        "tracker_training_evaluation": receipt["evidence"]["tracker_training_evaluation"],
    })

    source_path = exp043 / "exp043_x138_self_trained_head_inference.py"
    source = read(source_path)
    tree = ast.parse(source)
    patches = []
    functions = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Tuple) and len(node.elts) == 3:
            try:
                value = ast.literal_eval(node)
            except (ValueError, TypeError):
                continue
            if all(isinstance(x, str) for x in value) and value[0] in (
                "lowdet globals", "lowdet peaks", "lowdet write", "candidate dump", "cache write"
            ):
                patches.append({"line": node.lineno, "label": value[0], "before": value[1], "after": value[2]})
        if isinstance(node, ast.FunctionDef) and node.name in (
            "motion_relink_edges", "load_low_detections", "readmit_discarded_detections",
            "filter_output_graph", "write_test_submission",
        ):
            functions[node.name] = {"line": node.lineno, "end_line": node.end_lineno}
    assert len(patches) == 5
    write_json("exp043_code_locations.json", {
        "source": str(source_path.relative_to(ROOT)), "patches": patches, "functions": functions,
    })

    corrected = json.loads(read(ROOT / "experiments/exp040_trackastra_association/metrics.json"))
    write_json("exp040_corrected_run.json", {
        "status": corrected["status"],
        "notes": corrected["notes"],
        "colab": corrected["evidence"]["colab"],
        "fold_evaluation": corrected["train_stage"]["fold_evaluation"],
        "graph_eligible": corrected["train_stage"]["graph_eligible"],
        "post_run_audit": corrected["post_run_audit"],
    })

    for path in [
        ROOT / "backlog/KAGGLE_DIRECTION.md", ROOT / "SUBMISSIONS.md", ROOT / "experiment_summary.md",
        ROOT / "docs/surveys/biohub-public-0953-differences_20260923.md",
        ROOT / "docs/surveys/biohub-exp041-past-feature-analysis_20260923.md",
        ROOT / "experiments/exp015_oracle_stage_limits/exp015_oracle_stage_limits_inference.py",
        exp043 / "build_inference_notebook.py", Path(__file__).resolve(),
    ]:
        read(path)
    git_state = subprocess.check_output(["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=ROOT, text=True)
    write_json("source_manifest.json", {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "worktree_status": git_state.splitlines(),
        "warning": "Some evidence is uncommitted and other experiment tasks may update it; hashes identify this snapshot.",
        "files": [{"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in sorted(USED)],
    })
    print(json.dumps({"inventory_count": len(inventory), "pair_rows": len(pair_rows), "score_bins": len(bins),
                      "source_files": len(USED), "public_test_totals": totals}, ensure_ascii=False))


if __name__ == "__main__":
    main()
