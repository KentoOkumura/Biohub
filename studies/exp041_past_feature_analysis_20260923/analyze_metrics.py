"""Reproduce exp041 comparisons from saved aggregate metrics without inference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments/exp041_past_feature_cross_attention"
PARENT = ROOT / "experiments/exp016_frozen_image_encoder"
MODES = {
    "exp016": "exp016_saved_control",
    "full": "attention_outer_evaluation",
    "no_past": "no_past_outer_evaluation",
    "reverse_time": "reverse_time_outer_evaluation",
}


def integer_count(value: float) -> int:
    count = round(value)
    if abs(value - count) > 1e-6:
        raise ValueError(f"not an integer count: {value}")
    return count


def epoch_rows(fit: dict) -> list[dict]:
    rows = []
    for epoch in fit["epochs"]:
        val = epoch["internal_validation"]
        tp = integer_count(val["positive_edge_recall"] * val["positive_edge_count"])
        errors = integer_count((1 - val["edge_accuracy"]) * val["active_pair_count"])
        fn = val["positive_edge_count"] - tp
        rows.append(
            {
                "epoch_one_based": epoch["epoch"] + 1,
                "final_selected": epoch["epoch"] == fit["best_epoch"],
                "train_loss": epoch["train"]["legacy_mask_loss"],
                "validation_loss": val["legacy_mask_loss"],
                "true_positive_inferred_from_recall": tp,
                "false_negative_inferred_from_recall": fn,
                "false_positive_inferred_from_accuracy": errors - fn,
                "total_errors_inferred_from_accuracy": errors,
                "positive_edge_recall": val["positive_edge_recall"],
                "division_recovered_inferred_from_recall": integer_count(
                    val["division_parent_recall"] * val["division_parent_count"]
                ),
                "division_parent_count": val["division_parent_count"],
                "positive_pair_fraction": val["positive_edge_count"] / val["active_pair_count"],
                "selection_score": val["selection_score"],
            }
        )
    return rows


def main() -> None:
    current = json.loads((EXP / "metrics.json").read_text())
    parent = json.loads((PARENT / "metrics.json").read_text())
    parent_folds = {f["fold"]: f for f in parent["train_stage"]["folds"]}
    result = {"sources": {}, "folds": []}
    for path in (EXP / "metrics.json", PARENT / "metrics.json"):
        result["sources"][str(path.relative_to(ROOT))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    for fold in current["train_stage"]["folds"]:
        modes = {label: fold[key] for label, key in MODES.items()}
        control, full = modes["exp016"], modes["full"]
        assert all(m["teacher"] == control["teacher"] for m in modes.values())
        assert fold["window_counts"] == parent_folds[fold["fold"]]["window_counts"]
        mode_rows = {}
        for label, data in modes.items():
            assert data["correct_parent_top1_denominator"] == data["positive_edge_count"]
            tp = data["true_positive_edge_count"]
            top1 = data["correct_parent_top1_count"]
            assert top1 >= tp
            mode_rows[label] = {
                key: data[key]
                for key in (
                    "positive_edge_count",
                    "true_positive_edge_count",
                    "positive_edge_recall",
                    "false_positive_active_pair_count",
                    "correct_parent_top1_count",
                    "correct_parent_top1_accuracy",
                    "recovered_division_parent_count",
                    "division_parent_count",
                    "no_past_attention_mass_fraction",
                    "past_context_coverage",
                )
            }
            mode_rows[label]["top1_correct_but_not_above_threshold_count"] = top1 - tp
        density_rows = []
        for bucket, data in control["strata"]["target_density"].items():
            candidate = full["strata"]["target_density"][bucket]
            assert data["positive_edge_count"] == candidate["positive_edge_count"]
            density_rows.append(
                {
                    "neighbors_within_15um": bucket,
                    "positive_edges": data["positive_edge_count"],
                    "control_recall": data["positive_edge_recall"],
                    "full_recall": candidate["positive_edge_recall"],
                    "recall_delta_percentage_points": 100
                    * (candidate["positive_edge_recall"] - data["positive_edge_recall"]),
                    "true_positive_delta": candidate["true_positive_edge_count"]
                    - data["true_positive_edge_count"],
                    "teacher_false_positive_delta": candidate["false_positive_active_pair_count"]
                    - data["false_positive_active_pair_count"],
                }
            )
        delta_tp = full["true_positive_edge_count"] - control["true_positive_edge_count"]
        row = {
            "fold": fold["fold"],
            "train_embryo": fold["train_embryo"],
            "evaluation_embryo": fold["evaluation_embryo"],
            "modes": mode_rows,
            "full_vs_control": {
                "true_positive_delta": delta_tp,
                "recall_delta_percentage_points": 100
                * (full["positive_edge_recall"] - control["positive_edge_recall"]),
                "teacher_false_positive_delta_percent": 100
                * (
                    full["false_positive_active_pair_count"]
                    / control["false_positive_active_pair_count"]
                    - 1
                ),
                "top1_correct_delta": full["correct_parent_top1_count"]
                - control["correct_parent_top1_count"],
                "division_recovered_delta": full["recovered_division_parent_count"]
                - control["recovered_division_parent_count"],
            },
            "target_density": density_rows,
            "exp041_internal_epochs": epoch_rows(fold["attention_fit"]),
            "exp016_internal_epochs": epoch_rows(parent_folds[fold["fold"]]),
        }
        result["folds"].append(row)
    output = Path(__file__).with_name("readout.json")
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    for fold in result["folds"]:
        print(json.dumps(fold, ensure_ascii=False))
    print(f"saved {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
