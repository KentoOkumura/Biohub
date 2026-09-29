"""Audit the downloaded exp051 resume evaluation against its frozen inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARMS = ("expanded_optuna_cost", "expanded_optuna_cost_clipped")
TUNE_PATHS = {
    (0, ARMS[0]): "tune0_control_v1",
    (0, ARMS[1]): "tune0_clipped_v1",
    (1, ARMS[0]): "tune1_control_v1",
    (1, ARMS[1]): "tune1_clipped_v2",
}
OUTPUT_FILES = (
    "evaluation_receipt.json",
    "official_metric.json",
    "stage_diagnostic.json",
    "edge_changes.json",
)


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_sha(path: Path) -> str:
    files = sorted(p for p in path.rglob("*") if p.is_file())
    assert files, f"empty tree: {path}"
    return hashlib.sha256(
        b"".join(
            p.relative_to(path).as_posix().encode()
            + b"\0"
            + hashlib.sha256(p.read_bytes()).digest()
            for p in files
        )
    ).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def finite_tree(value: object, name: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            finite_tree(item, f"{name}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            finite_tree(item, f"{name}[{index}]")
    elif isinstance(value, float):
        assert math.isfinite(value), f"non-finite {name}"


def audit(output: Path, source: Path) -> dict:
    result_root = output / "exp051_clipping"
    source_config = source.parent / "config.yaml"
    source_selection = source.parent.parent / "v2_metadata/exp051_selection.json"
    source_log = source.parent.parent / "evaluate_v2.log"
    raw_source_log = (
        source.parent.parent / "v2_metadata/exp051-x138-ilp-score-clipping-evaluate.log"
    )
    receipt, official, stages, changes = (load(result_root / name) for name in OUTPUT_FILES)
    manifest_path = ROOT / "assets/reused_graph_manifest.json"
    manifest = load(manifest_path)
    assert receipt["status"] == "outer_evaluation_finished"
    assert receipt["reused_graph_count"] == 38
    assert receipt["new_graph_count"] == 2
    assert receipt["reused_graph_manifest_sha256"] == file_sha(manifest_path)
    assert file_sha(output / "assets/reused_graph_manifest.json") == file_sha(manifest_path)
    assert receipt["reused_kernel_id"] == manifest["source_kernel_id"]
    assert receipt["reused_kernel_version"] == manifest["source_kernel_version"] == 2
    assert receipt["config_sha256"] == manifest["source_config_sha256"]
    assert (
        receipt["reused_source_provenance"]["selection_file_sha256"] == manifest["selection_sha256"]
    )
    assert receipt["reused_source_provenance"]["config_file_sha256"] == file_sha(source_config)
    assert receipt["reused_source_provenance"]["selection_file_sha256"] == file_sha(
        source_selection
    )
    assert file_sha(source_log) == manifest["source_log_sha256"]
    assert (
        file_sha(raw_source_log)
        == load(ROOT / "metrics.json")["evidence"]["evaluate_run_v2"]["raw_kaggle_log_sha256"]
    )
    assert receipt["resume_package_config_sha256"] == file_sha(output / "config.yaml")
    for name, receipt_key in (
        ("official_metric.json", "official_metric_sha256"),
        ("stage_diagnostic.json", "stage_diagnostic_sha256"),
        ("edge_changes.json", "edge_changes_sha256"),
    ):
        assert file_sha(result_root / name) == receipt[receipt_key]

    stems = set(receipt["videos"])
    assert len(stems) == 20
    assert {s.split("_", 1)[0] for s in stems} == {"44b6", "6bba"}
    assert sum(s.startswith("44b6_") for s in stems) == 10
    assert sum(s.startswith("6bba_") for s in stems) == 10
    assert set(manifest["reused_stems"]) | set(manifest["missing_stems"]) == stems
    assert manifest["missing_stems"] == ["6bba_fe670320"]
    assert set(stages) == set(changes) == stems
    assert set(official) == set(ARMS)
    assert set(manifest["graphs"]) == {
        f"{arm}/{stem}" for arm in ARMS for stem in manifest["reused_stems"]
    }
    finite_tree(official, "official")
    finite_tree(stages, "stages")
    finite_tree(changes, "changes")
    status_counts = Counter()
    max_solve_seconds = 0.0
    graph_hashes = {}
    stage_hashes = {}
    new_stage_hashes = {}
    for arm in ARMS:
        measured = official[arm]
        assert measured["overall"]["n"] == 20
        assert {r["dataset"] for r in measured["rows"]} == stems
        assert len(measured["rows"]) == 20
        for embryo in ("44b6", "6bba"):
            assert measured["by_embryo"][embryo]["n"] == 10
        for stem in sorted(stems):
            row = receipt["videos"][stem][arm]
            assert set(stages[stem]) == set(ARMS)
            assert "final" in stages[stem][arm]
            assert row["solve"]["solver_status"] in {"OPTIMAL", "TIMELIMIT"}
            assert row["solve"]["seconds"] > 0
            status_counts[row["solve"]["solver_status"]] += 1
            max_solve_seconds = max(max_solve_seconds, row["solve"]["seconds"])
            key = f"{arm}/{stem}"
            graph = result_root / arm / "final_graphs" / f"{stem}.geff"
            graph_hashes[key] = tree_sha(graph)
            assert graph_hashes[key] == row["final_graph_sha256"], key
            fold = 1 if stem.startswith("44b6_") else 0
            tune_path = (
                ROOT
                / "artifacts/kaggle_tuning"
                / TUNE_PATHS[(fold, arm)]
                / "exp051_clipping/tune_receipt.json"
            )
            assert file_sha(tune_path) == receipt["tuning_receipt_sha256"][f"{fold}/{arm}"]
            setting = load(tune_path)["arms"][arm]
            assert row["costs"] == setting["selected_costs"]
            assert row["cap"] == setting["selected_cap"]
            if stem in manifest["reused_stems"]:
                assert row["reused_from_kernel_version"] == 2
                assert graph_hashes[key] == manifest["graphs"][key]["graph_sha256"]
                stage_path = source / "stages" / arm / stem
                stage_hashes[key] = tree_sha(stage_path)
                assert stage_hashes[key] == manifest["graphs"][key]["stages_sha256"]
                assert row["solve"]["solver_status"] == manifest["graphs"][key]["solver_status"]
            else:
                assert row["solve"]["solver_status"] == "OPTIMAL"
                assert row["solve"]["is_proven_optimal"] is True
                assert row["solve"]["seconds"] < 3600
                stage_path = result_root / "stages" / arm / stem
                assert len(list(stage_path.glob("*.npz"))) == 10
                new_stage_hashes[key] = tree_sha(stage_path)
    assert status_counts == {"OPTIMAL": 38, "TIMELIMIT": 2}
    assert receipt["time_limited_incumbents"] == {
        ARMS[0]: [],
        ARMS[1]: ["6bba_3abfe10a", "6bba_ebff6e76"],
    }
    assert 0 < receipt["elapsed_seconds"] < 12 * 3600

    by_embryo = {}
    for embryo in ("44b6", "6bba"):
        subset = [stem for stem in stems if stem.startswith(embryo + "_")]
        by_embryo[embryo] = {
            "scores": {arm: official[arm]["by_embryo"][embryo] for arm in ARMS},
            "edge_changes": {
                key: sum(changes[stem][key] for stem in subset)
                for key in next(iter(changes.values()))
            },
            "final_known_edge_selected": {
                arm: sum(stages[stem][arm]["final"]["known_edges_selected"] for stem in subset)
                for arm in ARMS
            },
            "final_annotated_parent_contradictions": {
                arm: sum(
                    stages[stem][arm]["final"]["annotated_parent_contradictions"] for stem in subset
                )
                for arm in ARMS
            },
        }
    report = {
        "status": "verified",
        "kernel_id": "kentookumura/exp051-x138-ilp-score-clipping-evaluate-resume",
        "kernel_version": 4,
        "graph_count": len(graph_hashes),
        "reused_graph_count": len(stage_hashes),
        "new_graph_count": 2,
        "solver_status_counts": dict(status_counts),
        "max_solve_seconds": max_solve_seconds,
        "resume_elapsed_seconds": receipt["elapsed_seconds"],
        "overall": {arm: official[arm]["overall"] for arm in ARMS},
        "by_embryo": by_embryo,
        "input_sha256": {
            "manifest": file_sha(manifest_path),
            "source_config": file_sha(source_config),
            "source_selection": file_sha(source_selection),
            "resume_config": file_sha(output / "config.yaml"),
            "capture_receipts": receipt["upstream_capture_receipts_sha256"],
        },
        "output_sha256": {name: file_sha(result_root / name) for name in OUTPUT_FILES},
        "graph_hashes": graph_hashes,
        "reused_stage_hashes": stage_hashes,
        "new_stage_hashes": new_stage_hashes,
        "limitations": {
            "reused_source_log_verified_on_mount": receipt["reused_source_log_verified_on_mount"],
            "reused_source_log_sha_verified_locally": True,
            "reused_timelimit_incumbent_audit_detail_persisted": receipt[
                "reused_timelimit_incumbent_audit_detail_persisted"
            ],
            "submission_runtime_verified": False,
            "cpu_gpu_graph_equivalence_verified": False,
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.output, args.source)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    exclude = {"graph_hashes", "reused_stage_hashes", "new_stage_hashes"}
    print(json.dumps({k: v for k, v in report.items() if k not in exclude}, indent=2))


if __name__ == "__main__":
    main()
