"""Verify saved detector-teacher diagnostic outputs and summarize sequence variation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

COUNT_KEYS = (
    "gt_divisions",
    "gt_geometric_divisions",
    "positive_triplets",
    "positive_with_wrong_mother",
    "wrong_division_triplets",
    "continuation_triplets",
    "unmatched_triplets",
)
LABEL_KEYS = {1: "positive_triplets", 2: "wrong_division_triplets", 3: "continuation_triplets"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def verify_window(path: Path, row: dict[str, str]) -> None:
    source_count = int(row["detected_src"])
    target_count = int(row["detected_tgt"])
    known_count = sum(int(row[key]) for key in LABEL_KEYS.values())
    assert int(row["all_triplets"]) == known_count + int(row["unmatched_triplets"])
    with np.load(path, allow_pickle=False) as archive:
        for branch in ("primary", "secondary"):
            for side, count in (("src", source_count), ("tgt", target_count)):
                feature = archive[f"{branch}_features_{side}"]
                assert feature.shape == (count, 32)
                assert np.isfinite(feature).all()
        assert archive["matched_triplet_ids"].shape == (known_count, 3)
        labels = archive["matched_triplet_class"]
        assert labels.shape == (known_count,)
        for value, key in LABEL_KEYS.items():
            assert int(np.count_nonzero(labels == value)) == int(row[key])
        for side, count in (("src", source_count), ("tgt", target_count)):
            assert archive[f"matched_gt_row_{side}"].shape == (count,)
            assert np.isfinite(archive[f"coords_{side}_physical"]).all()
            assert np.isfinite(archive[f"detection_scores_{side}"]).all()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--oracle-csv", type=Path, required=True)
    parser.add_argument("--previous-csv", type=Path)
    args = parser.parse_args()
    root = args.output_dir / "synthetic_detector_teacher_audit"
    summary = json.loads((root / "summary.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    rows = load_csv(root / "per_window.csv")
    oracle = load_csv(args.oracle_csv)
    expected_sequences = summary["selected_sequences"]
    assert len(rows) == expected_sequences * 5 == summary["window_count"]
    assert len(manifest["output"]["window_files"]) == len(rows)
    bundle = json.dumps(
        manifest["output"]["window_files"],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    assert (
        hashlib.sha256(bundle.encode()).hexdigest()
        == manifest["output"]["window_file_bundle_sha256"]
    )
    assert summary["feature_schema_sha256"] == manifest["output"]["feature_schema_sha256"]
    assert sha256(root / "summary.json") == manifest["output"]["summary_sha256"]
    assert sha256(root / "per_window.csv") == manifest["output"]["per_window_sha256"]
    by_sequence: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row, item in zip(rows, manifest["output"]["window_files"], strict=True):
        path = root / item["file"]
        assert sha256(path) == item["sha256"]
        verify_window(path, row)
        by_sequence[row["sequence"]].append(row)
    assert list(by_sequence) == [row["sequence"] for row in oracle[:expected_sequences]]
    assert [item["sha256"] for item in manifest["input"]["sequence_files"]] == [
        row["file_sha256"] for row in oracle[:expected_sequences]
    ]
    assert all(
        [int(row["t"]) for row in seq_rows] == list(range(5)) for seq_rows in by_sequence.values()
    )
    for key, value in summary["totals"].items():
        observed = (
            max(int(row[key]) for row in rows)
            if key == "max_triplets_per_mother"
            else sum(int(row[key]) for row in rows)
        )
        assert observed == value, key
    if args.previous_csv:
        previous = load_csv(args.previous_csv)
        assert len(previous) == 10
        for earlier, current in zip(previous, rows, strict=False):
            for key in earlier:
                if key != "gpu_forward_seconds":
                    assert earlier[key] == current[key], (key, earlier["sequence"], earlier["t"])
    sequence_counts = {
        name: {key: sum(int(row[key]) for row in seq_rows) for key in COUNT_KEYS}
        for name, seq_rows in by_sequence.items()
    }
    distribution = {
        key: {
            "min": min(item[key] for item in sequence_counts.values()),
            "median": statistics.median(item[key] for item in sequence_counts.values()),
            "max": max(item[key] for item in sequence_counts.values()),
        }
        for key in COUNT_KEYS
    }
    first_two = {
        key: sum(item[key] for item in list(sequence_counts.values())[:2]) for key in COUNT_KEYS
    }
    remaining = {key: summary["totals"][key] - first_two[key] for key in COUNT_KEYS}
    report = {
        "checks": (
            f"{len(rows)}-window SHA, feature finite/shape, label count, totals, "
            "oracle input SHA, first-two counts"
        ),
        "manifest_sha256": sha256(root / "manifest.json"),
        "summary_sha256": sha256(root / "summary.json"),
        "per_window_sha256": sha256(root / "per_window.csv"),
        "feature_bundle_sha256": manifest["output"]["window_file_bundle_sha256"],
        "distribution": distribution,
        "first_two": first_two,
        "remaining": remaining,
        "per_sequence": sequence_counts,
    }
    output = args.output_dir / "audit_report.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(
        json.dumps({key: value for key, value in report.items() if key != "per_sequence"}, indent=2)
    )
    print("audit_report_sha256", sha256(output))


if __name__ == "__main__":
    main()
