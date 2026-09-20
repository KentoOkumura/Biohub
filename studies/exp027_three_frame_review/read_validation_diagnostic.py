"""Verify internal-only threshold choice, then apply it to saved outer logits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments/exp027_multi_frame_tracker"
sys.path.insert(0, str(EXP))
from context_diagnostic import (  # noqa: E402
    aggregate_diagnostics,
    compare_known_edges,
    diagnose_pair,
    paired_video_bootstrap,
)
from frozen_tracker import array_content_sha256, json_sha256  # noqa: E402
from validation_diagnostic import describe_scores, select_negative_budget_threshold  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_outputs(folder: Path, summary: dict) -> list[dict]:
    assert summary["reference_reproduced"] and summary["model_training_count"] == 0
    for row in summary["weight_evidence"]:
        assert (
            row["before_state_sha256"] == row["after_state_sha256"] == row["canonical_state_sha256"]
        )
    for filename in [
        "input_manifest.json",
        "prediction_manifest.json",
        "known_edge_predictions.csv",
    ]:
        assert (
            sha(folder / filename)
            == summary[filename.replace(".json", "").replace(".csv", "") + "_sha256"]
        )
    manifest = json.loads((folder / "prediction_manifest.json").read_text())
    assert (
        json_sha256(
            [
                {"path": r["path"], "array_content_sha256": r["array_content_sha256"]}
                for r in manifest
            ]
        )
        == summary["prediction_content_sha256"]
    )
    return manifest


def read_arrays(folder: Path, record: dict) -> dict:
    path = folder / record["path"]
    assert sha(path) == record["file_sha256"]
    with np.load(path, allow_pickle=False) as data:
        arrays = dict(data)
    assert array_content_sha256(arrays) == record["array_content_sha256"]
    return arrays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--internal-root", type=Path, required=True)
    parser.add_argument("--outer-root", type=Path, default=EXP / "artifacts/kaggle_diagnostic_v1")
    args = parser.parse_args()
    config = yaml.safe_load((EXP / "config.yaml").read_text())
    cfg = config["validation_diagnostic"]
    diag = config["diagnostic"]
    internal = args.internal_root / "validation_diagnostic"
    summary = json.loads((internal / "diagnostic_summary.json").read_text())
    selected = json.loads((internal / "selected_thresholds.json").read_text())
    assert sha(internal / "selected_thresholds.json") == summary["selected_thresholds_sha256"]
    assert selected == summary["selected_thresholds"]
    assert sha(internal / "split_manifest.json") == cfg["split_manifest_sha256"]
    splits = json.loads((internal / "split_manifest.json").read_text())
    manifest = verify_outputs(internal, summary)
    assert len(manifest) == sum(cfg["expected_window_counts"])
    mean_rounding_differences = []
    # Validate internal-only threshold selection before opening outer predictions.
    for split in splits:
        entry = selected[str(split["fold"])]
        samples = set(split["internal_validation"])
        assert not samples.intersection(split["gradient_update"] + split["outer_evaluation"])
        records = [r for r in manifest if Path(r["path"]).parent.name in samples]
        assert len(records) == cfg["expected_window_counts"][split["fold"]]
        collected = {
            mode: {
                key: []
                for key in ["positive", "negative", "known_child_confidence", "known_child_correct"]
            }
            for mode in entry["modes"]
        }
        for record in records:
            arrays = read_arrays(internal, record)
            for mode in collected:
                for key in collected[mode]:
                    collected[mode][key].append(arrays[f"{key}_{mode}"])
        scores_by_mode = {
            mode: {key: np.concatenate(values) for key, values in score.items()}
            for mode, score in collected.items()
        }
        budget = int(
            np.count_nonzero(
                scores_by_mode[cfg["baseline_mode"]]["negative"] > cfg["baseline_threshold"]
            )
        )
        assert budget == entry["negative_budget"]
        for mode, scores in scores_by_mode.items():
            threshold = select_negative_budget_threshold(scores["negative"], budget)
            assert threshold == entry["modes"][mode]["selected_threshold"]
            actual_scores = describe_scores(
                scores, threshold, cfg["probability_quantiles"], cfg["confidence_bin_edges"]
            )
            saved_scores = entry["modes"][mode]["selected"]
            for key, value in actual_scores.items():
                if value != saved_scores[key]:
                    # Float32 means may differ by one rounding unit across CPUs/NumPy.
                    # Counts, quantiles, bins, threshold and SHA checks stay exact.
                    assert (
                        "mean" in key and abs(value - saved_scores[key]) <= np.finfo(np.float32).eps
                    )
                    mean_rounding_differences.append(
                        {
                            "fold": split["fold"],
                            "mode": mode,
                            "key": key,
                            "local": value,
                            "kaggle": saved_scores[key],
                        }
                    )
            expected = summary["summaries"][split["train_embryo"]][mode]
            assert len(scores["positive"]) == expected["positive_edge_count"]
            assert (
                np.count_nonzero(scores["positive"] > cfg["baseline_threshold"])
                == expected["true_positive_count"]
            )
    outer = args.outer_root / "context_diagnostic"
    assert sha(outer / "diagnostic_summary.json") == cfg["outer_summary_sha256"]
    outer_summary = json.loads((outer / "diagnostic_summary.json").read_text())
    outer_manifest = verify_outputs(outer, outer_summary)
    with (outer / "known_edge_predictions.csv").open() as handle:
        saved_edges = list(csv.DictReader(handle))
    edge_probabilities = {
        (
            r["sample"],
            int(r["source_frame"]),
            r["mode"],
            int(r["source_index"]),
            int(r["target_index"]),
        ): float(r["true_parent_probability"])
        for r in saved_edges
    }
    windows, edges = [], []
    max_probability_difference = 0.0
    minimum_negative_threshold_distance = 1.0
    maximum_roundoff_allowance = 0.0
    for record in outer_manifest:
        arrays = read_arrays(outer, record)
        path = Path(record["path"])
        sample = path.parent.name
        embryo = sample.split("_")[0]
        frame = int(path.stem.split("_")[0])
        fold_id = next(key for key, value in selected.items() if value["outer_embryo"] == embryo)
        target = arrays["target"]
        for mode, values in selected[fold_id]["modes"].items():
            logits = arrays[f"logits_{mode}"].astype(np.float64)
            probabilities = np.exp(logits - logits.max(axis=0))
            probabilities /= probabilities.sum(axis=0)
            # Allow for float32 source-axis reduction on the saved CUDA run.
            roundoff_allowance = (probabilities.shape[0] + 1) * np.finfo(np.float32).eps
            maximum_roundoff_allowance = max(maximum_roundoff_allowance, roundoff_allowance)
            for source, child in zip(*np.nonzero(target > 0.5), strict=True):
                saved = edge_probabilities[(sample, frame, mode, int(source), int(child))]
                assert abs(probabilities[source, child] - saved) <= roundoff_allowance
                max_probability_difference = max(
                    max_probability_difference, abs(probabilities[source, child] - saved)
                )
                probabilities[source, child] = saved
            for condition, threshold in [
                ("fixed", cfg["baseline_threshold"]),
                ("selected", values["selected_threshold"]),
            ]:
                # Keep counts insensitive to reconstruction rounding on teacher negatives.
                active = (target > 0.5).any(axis=1)[:, None] | (target > 0.5).any(axis=0)[None, :]
                near_negative = (
                    (np.abs(probabilities - threshold) <= roundoff_allowance)
                    & active
                    & (target <= 0.5)
                )
                negative_distances = np.abs(probabilities - threshold)[active & (target <= 0.5)]
                if len(negative_distances):
                    minimum_negative_threshold_distance = min(
                        minimum_negative_threshold_distance, float(negative_distances.min())
                    )
                if near_negative.any():
                    raise RuntimeError(
                        f"Reconstructed probability near threshold: {sample} {frame} {mode}"
                    )
                window, rows = diagnose_pair(
                    logits, probabilities, target, threshold=threshold, threshold_grid=[threshold]
                )
                window.update(
                    embryo=embryo,
                    mode=mode,
                    condition=condition,
                    sample=sample,
                    legacy_mask_loss=0.0,
                )
                windows.append(window)
                for row in rows:
                    row.update(
                        embryo=embryo,
                        mode=mode,
                        condition=condition,
                        sample=sample,
                        source_frame=frame,
                        target_frame=frame + 1,
                        source_id=int(arrays["candidate_ids_src"][row["source_index"]]),
                        target_id=int(arrays["candidate_ids_tgt"][row["target_index"]]),
                        predicted_parent_id=int(
                            arrays["candidate_ids_src"][row["predicted_parent_index"]]
                        ),
                    )
                    edges.append(row)
    results, comparisons = {}, {}
    paired_rows = []
    for embryo, reference_modes in outer_summary["summaries"].items():
        results[embryo], comparisons[embryo] = {}, {}
        for mode, reference in reference_modes.items():
            results[embryo][mode] = {}
            for condition in ["fixed", "selected"]:
                ww = [
                    r
                    for r in windows
                    if (r["embryo"], r["mode"], r["condition"]) == (embryo, mode, condition)
                ]
                ee = [
                    r
                    for r in edges
                    if (r["embryo"], r["mode"], r["condition"]) == (embryo, mode, condition)
                ]
                metrics = aggregate_diagnostics(ww, ee)
                metrics.pop("legacy_mask_loss")
                results[embryo][mode][condition] = metrics
                if condition == "fixed":
                    for key in [
                        "true_positive_count",
                        "false_positive_active_pair_count",
                        "top1_correct_count",
                        "division_parent_count",
                        "recovered_division_parent_count",
                    ]:
                        assert metrics[key] == reference[key], (embryo, mode, key)
        for comparison, base_mode, base_condition, changed_mode, changed_condition in [
            (
                "selected_full_minus_fixed_two",
                "two_frame_saved",
                "fixed",
                "three_frame_full",
                "selected",
            ),
            (
                "selected_full_minus_selected_two",
                "two_frame_saved",
                "selected",
                "three_frame_full",
                "selected",
            ),
            (
                "selected_full_minus_selected_masked",
                "three_frame_masked",
                "selected",
                "three_frame_full",
                "selected",
            ),
        ]:
            before = [
                r
                for r in edges
                if (r["embryo"], r["mode"], r["condition"]) == (embryo, base_mode, base_condition)
            ]
            after = [
                r
                for r in edges
                if (r["embryo"], r["mode"], r["condition"])
                == (embryo, changed_mode, changed_condition)
            ]
            delta, pairs = compare_known_edges(before, after)
            delta["video_bootstrap"] = paired_video_bootstrap(
                pairs,
                sorted({r["sample"] for r in windows if r["embryo"] == embryo}),
                repeats=diag["bootstrap_samples"],
                seed=diag["bootstrap_seed"],
                confidence=diag["bootstrap_confidence"],
            )
            comparisons[embryo][comparison] = delta
            paired_rows.extend({**r, "embryo": embryo, "comparison": comparison} for r in pairs)
    report = {
        "integrity": "PASS",
        "mean_rounding_differences": mean_rounding_differences,
        "internal_summary_sha256": sha(internal / "diagnostic_summary.json"),
        "selected_thresholds_sha256": sha(internal / "selected_thresholds.json"),
        "outer_summary_sha256": cfg["outer_summary_sha256"],
        "internal_prediction_content_sha256": summary["prediction_content_sha256"],
        "max_reconstructed_known_probability_difference": max_probability_difference,
        "minimum_negative_threshold_distance": minimum_negative_threshold_distance,
        "maximum_roundoff_allowance": float(maximum_roundoff_allowance),
        "internal_selected_thresholds": selected,
        "outer_results": results,
        "outer_comparisons": comparisons,
        "notebook_runtime_seconds": summary["notebook_runtime_seconds"],
    }
    destination = Path(__file__).with_name("validation_readout.json")
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with destination.with_name("validation_outer_paired_edges.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)
    print(
        json.dumps(
            {"integrity": "PASS", "outer_results": results, "outer_comparisons": comparisons},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
