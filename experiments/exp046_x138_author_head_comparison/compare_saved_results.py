"""Join the Kaggle author-head run with the frozen exp045 self-head control."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT.parent
CONTROL = EXPERIMENTS / "exp045_x138_coordinate_effect_audit" / "artifacts"
AUTHOR = ROOT / "artifacts" / "author_only" / "exp046_author_only"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def row_score(row: dict) -> float:
    total = sum(row[key] for key in ("division_tp", "division_fp", "division_fn"))
    score = float(row["adj_edge_jaccard"])
    return score + 0.1 * row["division_tp"] / total if total else score


def score_delta(author: dict, control: dict) -> dict:
    fields = ("score", "adj_edge_jaccard", "edge_jaccard", "node_recall")
    return {key: float(author[key]) - float(control[key]) for key in fields}


def main() -> None:
    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    control_receipt_path = (
        CONTROL / "official_v3" / "exp045_official_resume" / "exp045_official_receipt.json"
    )
    require(
        sha256(control_receipt_path) == config["source"]["exp045_control_official_receipt_sha256"],
        "exp045 official control receipt SHA changed",
    )
    control_receipt = read_json(control_receipt_path)
    control_metric_path = (
        CONTROL / "official_v3" / "exp045_official_resume" / "official_metric.json"
    )
    control_metric = read_json(control_metric_path)["refined"]
    control_identity = read_json(CONTROL / "local_fixed_id_v1" / "identity.json")
    control_fixed = read_json(CONTROL / "local_fixed_id_v1" / "fixed_id_diagnostic.json")
    control_embryo = read_json(CONTROL / "local_fixed_id_v1" / "fixed_id_by_embryo.json")
    control_pilot = read_json(CONTROL / "pilot" / "exp045_pilot_receipt.json")
    pilot = read_json(ROOT / "artifacts" / "pilot" / "exp046_audit" / "exp046_pilot_receipt.json")
    author_receipt_path = AUTHOR / "exp046_author_only_receipt.json"
    author_receipt = read_json(author_receipt_path)
    author_metric_path = AUTHOR / "official_metric.json"
    author_metric = read_json(author_metric_path)["author"]
    author_fixed = read_json(AUTHOR / "fixed_id_diagnostic.json")
    author_embryo = read_json(AUTHOR / "fixed_id_by_embryo.json")
    author_manifest = read_json(AUTHOR / "file_manifest.json")
    manifest_by_path = {row["path"]: row for row in author_manifest}
    require(
        len(author_manifest) == author_receipt["output_files"], "author output file count differs"
    )
    require(
        sum(row["bytes"] for row in author_manifest) == author_receipt["output_bytes"],
        "author output byte count differs",
    )
    for name in (
        "official_metric.json",
        "fixed_id_diagnostic.json",
        "fixed_id_by_embryo.json",
        "runtime_gate.json",
    ):
        path = AUTHOR / name
        require(
            path.stat().st_size == manifest_by_path[name]["bytes"]
            and sha256(path) == manifest_by_path[name]["sha256"],
            f"author output SHA differs: {name}",
        )

    selected = control_receipt["selected_videos"]
    require(len(selected) == 20 and len(set(selected)) == 20, "control selection is not 20 videos")
    require(
        selected == pilot["selected_videos"] == author_receipt["selected_videos"],
        "selected videos differ",
    )
    require(
        control_receipt["official_evaluator_sha256"]
        == pilot["public_evaluator_sha256"]
        == author_receipt["official_evaluator_sha256"]
        == config["source"]["public_evaluator_sha256"],
        "evaluator SHA differs",
    )
    require(
        control_receipt["head_sha256"]
        == pilot["head_bundles"]["self"]["sha256"]
        == config["source"]["self_head_sha256"],
        "self head SHA differs",
    )
    require(
        pilot["head_bundles"]["author"]["sha256"]
        == author_receipt["head_bundles"]["author"]["sha256"]
        == config["source"]["author_head_sha256"],
        "author head SHA differs",
    )
    require(
        control_pilot["selected_videos"] == pilot["selected_videos"], "pilot selected videos differ"
    )
    for stem in pilot["pilot_videos"]:
        old = control_pilot["pilot_identity"][stem]
        new = pilot["pilot_identity"][stem]
        require(
            old["candidates"] == new["candidates"]
            and old["raw_detector_sha256"] == new["raw_detector_sha256"]
            and old["initial_graph_nodes"]["refined"] == new["initial_graph_nodes"]["self"]
            and old["max_shift_um"] == new["max_shift_from_original_um"]["self"],
            f"{stem}: pilot control candidate identity differs",
        )
    old_graph_files = {
        row["path"].removeprefix("refined/"): row["sha256"]
        for row in control_pilot["files"]
        if row["path"].startswith("refined/final_graphs/")
    }
    new_graph_files = {
        row["path"].removeprefix("self/"): row["sha256"]
        for row in pilot["files"]
        if row["path"].startswith("self/final_graphs/")
    }
    require(
        len(old_graph_files) == 42 and old_graph_files == new_graph_files,
        "pilot control final graphs differ",
    )
    require(
        control_pilot["pilot_official"]["refined"]["summary"]
        == pilot["pilot_official"]["self"]["summary"],
        "pilot control official summary differs",
    )
    for old, new in zip(
        control_pilot["pilot_official"]["refined"]["rows"],
        pilot["pilot_official"]["self"]["rows"],
        strict=True,
    ):
        for key in set(old) - {"method", "username"}:
            require(old[key] == new[key], f"pilot control official row differs: {key}")
    require(
        not author_receipt["ground_truth_accessed_during_prediction"]
        and not author_receipt["competition_submission_created"],
        "author run provenance differs",
    )
    require(
        author_receipt["official_scores"]["author"]["overall"] == author_metric["overall"],
        "author official summary differs",
    )
    require(
        control_receipt["official_scores"]["refined"]["overall"] == control_metric["overall"],
        "control official summary differs",
    )
    require(
        author_receipt["fixed_id_by_embryo"] == author_embryo,
        "author fixed-ID embryo summary differs",
    )

    control_rows = {row["dataset"]: row for row in control_metric["rows"]}
    author_rows = {row["dataset"]: row for row in author_metric["rows"]}
    require(
        set(control_rows) == set(author_rows) == set(selected),
        "official video rows differ from selection",
    )
    require(
        set(control_identity) == set(author_receipt["identity"]) == set(selected),
        "candidate identity coverage differs",
    )
    require(
        set(control_fixed) == set(author_fixed) == set(selected),
        "fixed-ID diagnostic coverage differs",
    )
    for pilot_row in pilot["pilot_official"]["author"]["rows"]:
        full_row = author_rows[pilot_row["dataset"]]
        for key in set(pilot_row) - {"method", "username"}:
            require(pilot_row[key] == full_row[key], f"pilot author official row differs: {key}")
    pilot_graph_files = {
        row["path"].removeprefix("author/"): row["sha256"]
        for row in pilot["files"]
        if row["path"].startswith("author/final_graphs/")
    }
    full_graph_files = {
        row["path"].removeprefix("author/"): row["sha256"]
        for row in author_manifest
        if row["path"].startswith("author/final_graphs/")
        and any(stem in row["path"] for stem in pilot["pilot_videos"])
    }
    require(
        len(pilot_graph_files) == 42 and pilot_graph_files == full_graph_files,
        "pilot author final graphs differ from full run",
    )

    by_video = {}
    for stem in selected:
        old_identity = control_identity[stem]
        new_identity = author_receipt["identity"][stem]
        require(
            old_identity["raw_detector_sha256"] == new_identity["raw_detector_sha256"]
            and old_identity["candidates"] == new_identity["candidates"]
            and old_identity["raw_frames"] == new_identity["raw_frames"],
            f"{stem}: original detector candidates differ",
        )
        old_fixed, new_fixed = control_fixed[stem], author_fixed[stem]
        for key in ("candidate_count", "matched_gt_nodes", "ambiguous_gt", "ambiguous_candidate"):
            require(old_fixed[key] == new_fixed[key], f"{stem}: fixed GT assignment {key} differs")
        old_row, new_row = control_rows[stem], author_rows[stem]
        old_score, new_score = row_score(old_row), row_score(new_row)
        by_video[stem] = {
            "embryo": stem.split("_", 1)[0],
            "control_score": old_score,
            "author_score": new_score,
            "score_delta": new_score - old_score,
            "control_adj_edge_jaccard": old_row["adj_edge_jaccard"],
            "author_adj_edge_jaccard": new_row["adj_edge_jaccard"],
            "control_divisions": {
                key: old_row[key] for key in ("division_tp", "division_fp", "division_fn")
            },
            "author_divisions": {
                key: new_row[key] for key in ("division_tp", "division_fp", "division_fn")
            },
            "candidates": old_identity["candidates"],
            "raw_detector_sha256": old_identity["raw_detector_sha256"],
            "control_center_distance_um": old_fixed["arms"]["refined"][
                "mean_fixed_match_center_distance_um"
            ],
            "author_center_distance_um": new_fixed["arms"]["author"][
                "mean_fixed_match_center_distance_um"
            ],
        }

    by_embryo = {}
    for embryo in ("44b6", "6bba"):
        old = control_metric["by_embryo"][embryo]
        new = author_metric["by_embryo"][embryo]
        old_fixed, new_fixed = control_embryo[embryo], author_embryo[embryo]
        require(
            old["n"] == new["n"] == old_fixed["videos"] == new_fixed["videos"] == 10,
            f"{embryo}: video coverage differs",
        )
        for key in ("candidate_count", "matched_gt_nodes", "ambiguous_gt", "ambiguous_candidate"):
            require(old_fixed[key] == new_fixed[key], f"{embryo}: {key} differs")
        old_stages = old_fixed["arms"]["refined"]
        new_stages = new_fixed["arms"]["author"]
        require(set(old_stages) == set(new_stages), f"{embryo}: stage names differ")
        stems = [stem for stem in selected if stem.startswith(embryo + "_")]
        matched_nodes = sum(control_fixed[stem]["matched_gt_nodes"] for stem in stems)
        require(
            matched_nodes == old_fixed["matched_gt_nodes"], f"{embryo}: matched GT count differs"
        )
        center_distance = {
            arm: sum(
                control_fixed[stem]["matched_gt_nodes"]
                * report[stem]["arms"][arm_name]["mean_fixed_match_center_distance_um"]
                for stem in stems
            )
            / matched_nodes
            for arm, report, arm_name in (
                ("control", control_fixed, "refined"),
                ("author", author_fixed, "author"),
            )
        }
        edge_counts = {
            arm: {
                key: sum(rows[stem][key] for stem in stems)
                for key in ("edge_tp", "edge_fp", "edge_fn")
            }
            for arm, rows in (("control", control_rows), ("author", author_rows))
        }
        by_embryo[embryo] = {
            "control": old,
            "author": new,
            "fixed_id_weighted_center_distance_um": center_distance,
            "official_edge_counts": edge_counts,
            "delta": score_delta(new, old),
            "videos_author_better": sum(
                by_video[stem]["score_delta"] > 0
                for stem in selected
                if stem.startswith(embryo + "_")
            ),
            "fixed_id_stages": {
                stage: {
                    "control": old_stages[stage],
                    "author": new_stages[stage],
                }
                for stage in old_stages
            },
        }

    result = {
        "experiment": ROOT.name,
        "control_official_receipt_sha256": sha256(control_receipt_path),
        "control_official_metric_sha256": sha256(control_metric_path),
        "author_receipt_sha256": sha256(author_receipt_path),
        "author_official_metric_sha256": sha256(author_metric_path),
        "pilot_receipt_sha256": sha256(
            ROOT / "artifacts" / "pilot" / "exp046_audit" / "exp046_pilot_receipt.json"
        ),
        "selected_videos": selected,
        "overall": {
            "control": control_metric["overall"],
            "author": author_metric["overall"],
            "delta": score_delta(author_metric["overall"], control_metric["overall"]),
        },
        "by_embryo": by_embryo,
        "by_video": by_video,
        "author_notebook_runtime_seconds": author_receipt["notebook_runtime_seconds"],
        "author_output_manifest_file_count": len(author_manifest),
        "retrieved_output_files_sha_verified": 4,
        "pilot_author_final_graph_files_sha_equal": len(pilot_graph_files),
        "pilot_author_official_rows_equal": len(pilot["pilot_official"]["author"]["rows"]),
    }
    output = ROOT / "artifacts" / "comparison.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"Saved {output}")
    print("Overall score delta:", result["overall"]["delta"]["score"])
    for embryo, row in by_embryo.items():
        print(
            embryo,
            "score delta:",
            row["delta"]["score"],
            "author-better videos:",
            row["videos_author_better"],
        )


if __name__ == "__main__":
    main()
