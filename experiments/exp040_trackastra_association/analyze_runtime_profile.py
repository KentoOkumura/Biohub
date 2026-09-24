"""Estimate the bounded two-fold run from exp040's measured Kaggle profile."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from trackastra_association import training_window_schedule


def upper_mean(values: list[float]) -> float:
    if not values:
        raise ValueError("runtime profile has an empty measurement group")
    mean = statistics.mean(values)
    return mean + 2 * statistics.pstdev(values) / math.sqrt(len(values))


def estimate(profile: dict[str, Any], manifests: dict[int, list[dict]], config: dict) -> dict:
    training = config["model"]["training"]
    folds = config["validation"]["outer_folds"]
    observations = [profile["folds"][str(fold)] for fold in (0, 1)]
    load_upper = max(
        upper_mean([float(row["seconds"]) for row in item["video_loads"]]) for item in observations
    )
    control_per_pair_upper = max(
        upper_mean(
            [float(row["seconds"]) / int(row["pair_count"]) for row in item["control_videos"]]
        )
        for item in observations
    )
    cost_by_fold = {}
    for fold in (0, 1):
        item = profile["folds"][str(fold)]
        cuts = item["quantile_boundaries"][1:-1]
        non_stress = [row for row in item["observations"] if not row["stress"]]
        costs = {}
        for band in range(len(item["band_counts"])):
            rows = [row for row in non_stress if int(row["band"]) == band]
            if int(item["band_counts"][str(band)]) and not rows:
                raise ValueError(f"fold {fold} band {band} lacks a runtime sample")
            if rows:
                costs[band] = {
                    key: upper_mean([float(row[key]) for row in rows])
                    for key in (
                        "teacher_seconds",
                        "train_seconds",
                        "validation_seconds",
                        "inference_seconds",
                    )
                }
        cost_by_fold[fold] = (cuts, costs)

    def window_cost(fold: int, tokens: int, kinds: tuple[str, ...]) -> float:
        cuts, costs = cost_by_fold[fold]
        band = int(np.searchsorted(cuts, tokens, side="right"))
        return sum(costs[band][kind] for kind in kinds)

    candidates = []
    maximum = min(int(profile["folds"][str(fold)]["active_window_count"]) for fold in (0, 1))
    for cap in sorted({*range(128, maximum + 1, 128), maximum}):
        if cap < 128:
            continue
        total = 0.0
        detail = {}
        for fold_cfg in folds:
            fold = int(fold_cfg["fold"])
            schedule = training_window_schedule(
                manifests[fold],
                windows_per_epoch=cap,
                epochs=int(training["epochs"]),
                seed=int(training["seed"]) + fold,
            )
            train_rows = [row for epoch in schedule["epochs"] for row in epoch]
            train_seconds = sum(
                window_cost(
                    fold,
                    int(row["tokens"]),
                    ("teacher_seconds", "train_seconds"),
                )
                for row in train_rows
            )
            internal_tokens = profile["evaluation_tokens"][str(fold)]["internal"]
            outer_tokens = profile["evaluation_tokens"][str(fold)]["outer"]
            validation_seconds = int(training["epochs"]) * sum(
                window_cost(
                    fold,
                    int(tokens),
                    ("teacher_seconds", "validation_seconds"),
                )
                for tokens in internal_tokens
            )
            candidate_seconds = sum(
                window_cost(
                    fold,
                    int(tokens),
                    ("teacher_seconds", "inference_seconds"),
                )
                for tokens in (*internal_tokens, *outer_tokens)
            )
            train_loads = sum(
                len({str(row["sample"]) for row in epoch}) for epoch in schedule["epochs"]
            )
            video_loads = (
                train_loads
                + int(training["epochs"])
                * int(fold_cfg["expected_internal_validation_sample_count"])
                + int(fold_cfg["expected_internal_validation_sample_count"])
                + int(fold_cfg["expected_evaluation_sample_count"])
            )
            loading_seconds = video_loads * load_upper
            control_pairs = 99 * (
                int(fold_cfg["expected_internal_validation_sample_count"])
                + int(fold_cfg["expected_evaluation_sample_count"])
            )
            control_seconds = control_pairs * control_per_pair_upper
            subtotal = (
                train_seconds
                + validation_seconds
                + candidate_seconds
                + loading_seconds
                + control_seconds
            )
            total += subtotal
            detail[str(fold)] = {
                "train_seconds": train_seconds,
                "validation_seconds": validation_seconds,
                "candidate_seconds": candidate_seconds,
                "video_loading_seconds": loading_seconds,
                "control_seconds": control_seconds,
                "distinct_training_windows": len(
                    {(row["sample"], row["start"]) for row in train_rows}
                ),
                "population_windows": int(schedule["population_count"]),
                "training_schedule_stratum_windows_per_epoch": schedule[
                    "stratum_windows_per_epoch"
                ],
            }
        fixed_reserve_seconds = 3600.0
        projected = fixed_reserve_seconds + float(training["runtime_projection_multiplier"]) * total
        candidates.append(
            {
                "windows_per_epoch_per_fold": cap,
                "projected_seconds": projected,
                "projected_hours": projected / 3600,
                "within_12_hour_gate": projected <= float(training["runtime_gate_hours"]) * 3600,
                "folds": detail,
            }
        )
    feasible = [row for row in candidates if row["within_12_hour_gate"]]
    return {
        "method": "density-band upper mean plus 1.5 multiplier and one-hour fixed reserve",
        "fixed_reserve_seconds": 3600.0,
        "video_load_upper_seconds": load_upper,
        "control_per_pair_upper_seconds": control_per_pair_upper,
        "recommended_max_cap": feasible[-1]["windows_per_epoch_per_fold"] if feasible else None,
        "candidates": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text())
    manifests = {}
    for fold in (0, 1):
        record = profile["window_manifests"][str(fold)]
        path = args.profile.parent / record["path"]
        import hashlib

        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"window manifest SHA mismatch: {path}")
        manifests[fold] = json.loads(path.read_text())
    config = yaml.safe_load(args.config.read_text())
    result = estimate(profile, manifests, config)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "recommended_max_cap": result["recommended_max_cap"],
                "candidate_hours": {
                    row["windows_per_epoch_per_fold"]: round(row["projected_hours"], 2)
                    for row in result["candidates"]
                },
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
