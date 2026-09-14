from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize exp017 Kaggle audit outputs.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"count": 0, "min": None, "q25": None, "median": None, "q75": None, "max": None}
    quantiles = np.quantile(array, [0.0, 0.25, 0.5, 0.75, 1.0])
    return {
        key: float(value)
        for key, value in zip(("min", "q25", "median", "q75", "max"), quantiles, strict=True)
    } | {"count": int(len(array))}


def as_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def as_bool(row: dict[str, str], key: str) -> bool:
    return str(row.get(key, "")).lower() == "true"


def pair_key(sample_a: str, sample_b: str) -> tuple[str, str]:
    return tuple(sorted((sample_a, sample_b)))


def main() -> None:
    args = parse_args()
    freeze = read_rows(args.input_dir / "freeze_runs.csv")
    phase = read_rows(args.input_dir / "phase_screen.csv")
    registration = read_rows(args.input_dir / "registration_pairs.csv")
    null = read_rows(args.input_dir / "null_pairs.csv")
    volumes = read_rows(args.input_dir / "volume_confirmation.csv")
    geff = read_rows(args.input_dir / "geff_pair_evidence.csv")

    freeze_by_embryo: dict[str, list[dict[str, str]]] = defaultdict(list)
    schedule_by_sample: dict[str, set[tuple[int, int]]] = defaultdict(set)
    signature_samples: dict[str, set[str]] = defaultdict(set)
    for row in freeze:
        freeze_by_embryo[row["embryo_id"]].append(row)
        schedule_by_sample[row["sample_id"]].add((int(row["start_frame"]), int(row["end_frame"])))
        signature_samples[row["quick_signature"]].add(row["sample_id"])
    freeze_summary = {}
    for embryo, rows in sorted(freeze_by_embryo.items()):
        freeze_summary[embryo] = {
            "run_count": len(rows),
            "sample_count": len({row["sample_id"] for row in rows}),
            "run_length_counts": dict(sorted(Counter(int(row["length"]) for row in rows).items())),
            "longest_run": max(int(row["length"]) for row in rows),
        }
    cross_crop_same_content = [
        {"quick_signature": signature, "samples": sorted(samples)}
        for signature, samples in signature_samples.items()
        if len(samples) > 1
    ]

    phase_lookup = {pair_key(row["sample_a"], row["sample_b"]): row for row in phase}
    schedule_pairs = []
    samples_by_embryo: dict[str, list[str]] = defaultdict(list)
    for sample in schedule_by_sample:
        samples_by_embryo[sample.split("_", maxsplit=1)[0]].append(sample)
    for embryo, sample_ids in sorted(samples_by_embryo.items()):
        for sample_a, sample_b in combinations(sorted(sample_ids), 2):
            left = schedule_by_sample[sample_a]
            right = schedule_by_sample[sample_b]
            intersection = len(left & right)
            union = len(left | right)
            if not intersection:
                continue
            phase_row = phase_lookup.get(pair_key(sample_a, sample_b), {})
            schedule_pairs.append(
                {
                    "sample_a": sample_a,
                    "sample_b": sample_b,
                    "embryo_id": embryo,
                    "shared_repeat_runs": intersection,
                    "schedule_jaccard": intersection / union,
                    "phase_ncc_median": as_float(phase_row, "phase_ncc_median"),
                    "phase_shift_mad_ds": as_float(phase_row, "shift_mad_ds"),
                }
            )
    schedule_pairs.sort(
        key=lambda row: (row["schedule_jaccard"], row["shared_repeat_runs"]),
        reverse=True,
    )

    fixed_threshold = 0.45
    null_threshold = float(
        np.quantile([as_float(row, "thumbnail_ncc_median") for row in null], 0.99)
    )
    effective_threshold = max(fixed_threshold, null_threshold)
    min_margin = 0.03
    max_shift_mad = 2.0
    criterion_counts = {
        "ncc_at_least_effective_threshold": sum(
            as_float(row, "thumbnail_ncc_median") >= effective_threshold for row in registration
        ),
        "same_vs_mismatch_margin_at_least_0_03": sum(
            as_float(row, "same_vs_mismatch_margin") >= min_margin for row in registration
        ),
        "shift_mad_at_most_2": sum(
            as_float(row, "shift_mad_ds") <= max_shift_mad for row in registration
        ),
        "all_2d_criteria": sum(as_bool(row, "passes_2d_evidence") for row in registration),
    }
    near_2d = sorted(
        registration,
        key=lambda row: (
            as_float(row, "thumbnail_ncc_median")
            - effective_threshold
            + min(as_float(row, "same_vs_mismatch_margin") - min_margin, 0.0)
            - max(as_float(row, "shift_mad_ds") - max_shift_mad, 0.0) * 0.02
        ),
        reverse=True,
    )[:10]
    near_2d_rows = [
        {
            key: row[key]
            for key in (
                "sample_a",
                "sample_b",
                "embryo_id",
                "transform",
                "thumbnail_ncc_median",
                "thumbnail_overlap_median",
                "same_vs_mismatch_margin",
                "shift_mad_ds",
                "passes_2d_evidence",
            )
        }
        for row in near_2d
    ]
    repeat_registration = [row for row in registration if as_bool(row, "identical_repeat_schedule")]
    repeat_volumes = [row for row in volumes if as_bool(row, "identical_repeat_schedule")]
    repeat_volume_keys = {pair_key(row["sample_a"], row["sample_b"]) for row in repeat_volumes}
    repeat_geff = [
        row for row in geff if pair_key(row["sample_a"], row["sample_b"]) in repeat_volume_keys
    ]

    geff_nonzero = sorted(
        geff,
        key=lambda row: (
            int(row["spatial_node_matches"]),
            int(row["mapped_edges_present_in_a"]),
        ),
        reverse=True,
    )
    output: dict[str, Any] = {
        "freeze": {
            "by_embryo": freeze_summary,
            "samples_without_any_run_in_6bba": 128
            - freeze_summary.get("6bba", {}).get("sample_count", 0),
            "cross_crop_identical_repeated_frame_content_count": len(cross_crop_same_content),
            "cross_crop_identical_repeated_frame_content": cross_crop_same_content[:20],
            "pair_count_with_shared_repeat_timing": len(schedule_pairs),
            "pair_count_with_identical_nonempty_repeat_schedule": sum(
                float(row["schedule_jaccard"]) == 1.0 for row in schedule_pairs
            ),
            "top_shared_repeat_schedules": schedule_pairs[:20],
        },
        "registration": {
            "same_embryo_pairs_screened": len(phase),
            "d4_pairs_refined": len(registration),
            "null_pairs": len(null),
            "d4_refined_ncc": numeric_summary(
                [as_float(row, "thumbnail_ncc_median") for row in registration]
            ),
            "cross_embryo_null_ncc": numeric_summary(
                [as_float(row, "thumbnail_ncc_median") for row in null]
            ),
            "null_ncc_q99": null_threshold,
            "effective_ncc_threshold": effective_threshold,
            "criterion_counts": criterion_counts,
            "near_2d_candidates": near_2d_rows,
            "identical_repeat_schedule_pairs": {
                "count": len(repeat_registration),
                "thumbnail_ncc": numeric_summary(
                    [as_float(row, "thumbnail_ncc_median") for row in repeat_registration]
                ),
                "same_vs_mismatch_margin": numeric_summary(
                    [as_float(row, "same_vs_mismatch_margin") for row in repeat_registration]
                ),
                "shift_mad_ds": numeric_summary(
                    [as_float(row, "shift_mad_ds") for row in repeat_registration]
                ),
                "passes_2d": sum(as_bool(row, "passes_2d_evidence") for row in repeat_registration),
                "volume_ncc": numeric_summary(
                    [as_float(row, "volume_ncc_median") for row in repeat_volumes]
                ),
                "passes_2d_and_3d": sum(
                    as_bool(row, "passes_registration_evidence") for row in repeat_volumes
                ),
                "pairs_with_spatial_geff_matches": sum(
                    int(row["spatial_node_matches"]) > 0 for row in repeat_geff
                ),
                "maximum_spatial_geff_matches": max(
                    (int(row["spatial_node_matches"]) for row in repeat_geff), default=0
                ),
            },
        },
        "volume": {
            "checked_pairs": len(volumes),
            "ncc": numeric_summary([as_float(row, "volume_ncc_median") for row in volumes]),
            "at_least_0_35": sum(as_float(row, "volume_ncc_median") >= 0.35 for row in volumes),
            "passes_all_registration_evidence": sum(
                as_bool(row, "passes_registration_evidence") for row in volumes
            ),
        },
        "geff": {
            "checked_pairs": len(geff),
            "pairs_with_spatial_matches": sum(int(row["spatial_node_matches"]) > 0 for row in geff),
            "maximum_spatial_matches": max(int(row["spatial_node_matches"]) for row in geff),
            "pairs_with_mapped_edge_agreement": sum(
                int(row["mapped_edges_present_in_a"]) > 0 for row in geff
            ),
            "top_rows": [
                {
                    key: row[key]
                    for key in (
                        "sample_a",
                        "sample_b",
                        "spatial_node_matches",
                        "spatial_match_fraction_of_mapped_b",
                        "mapped_edges_present_in_a",
                        "passes_registration_evidence",
                    )
                }
                for row in geff_nonzero[:10]
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
