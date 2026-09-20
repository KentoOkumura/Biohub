"""Verify the paired inference outputs and create compact diagnostic tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments/exp027_multi_frame_tracker"
sys.path.insert(0, str(EXP))
from frozen_tracker import array_content_sha256, json_sha256  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    folder = args.output_root / "context_diagnostic"
    summary = json.loads((folder / "diagnostic_summary.json").read_text())
    assert summary["reference_reproduced"]
    assert summary["model_training_count"] == 0
    for row in summary["weight_evidence"]:
        assert (
            row["before_state_sha256"] == row["after_state_sha256"] == row["canonical_state_sha256"]
        )
    for name, key in [
        ("known_edge_predictions.csv", "known_edge_predictions_sha256"),
        ("prediction_manifest.json", "prediction_manifest_sha256"),
        ("input_manifest.json", "input_manifest_sha256"),
    ]:
        assert sha(folder / name) == summary[key], name
    manifest = json.loads((folder / "prediction_manifest.json").read_text())
    boundary_checks = []
    for row in manifest:
        p = folder / row["path"]
        assert sha(p) == row["file_sha256"], row["path"]
        with np.load(p, allow_pickle=False) as saved:
            arrays = dict(saved)
        assert array_content_sha256(arrays) == row["array_content_sha256"], row["path"]
        if p.name.startswith("000000_"):
            a = arrays["logits_three_frame_full"]
            b = arrays["logits_three_frame_masked"]
            difference = float(np.max(np.abs(a - b)))
            boundary_checks.append({"path": row["path"], "max_abs_logit_difference": difference})
    assert (
        json_sha256(
            [
                {"path": row["path"], "array_content_sha256": row["array_content_sha256"]}
                for row in manifest
            ]
        )
        == summary["prediction_content_sha256"]
    )
    assert len(manifest) == 128
    with (folder / "known_edge_predictions.csv").open() as handle:
        edges = list(csv.DictReader(handle))
    details = {}
    for embryo, modes in summary["summaries"].items():
        details[embryo] = {}
        for mode, values in modes.items():
            rows = [r for r in edges if r["embryo"] == embryo and r["mode"] == mode]
            assert len(rows) == values["positive_edge_count"]
            details[embryo][mode] = {
                key: values[key]
                for key in [
                    "positive_edge_count",
                    "true_positive_count",
                    "positive_edge_recall",
                    "top1_correct_count",
                    "top1_accuracy",
                    "false_positive_active_pair_count",
                    "mean_true_parent_probability",
                    "mean_reciprocal_rank",
                ]
            }
            details[embryo][mode]["top1_below_threshold_count"] = sum(
                r["top1_correct"] == "True" and r["recovered"] == "False" for r in rows
            )
        a = summary["comparisons"][embryo]
        assert np.isclose(
            a["full_minus_two_frame"]["recall_delta"],
            a["full_minus_masked"]["recall_delta"] + a["masked_minus_two_frame"]["recall_delta"],
        )
    report = {
        "integrity": "PASS",
        "no_previous_frame_checks": boundary_checks,
        "window_count": len(manifest),
        "diagnostic_summary_sha256": sha(folder / "diagnostic_summary.json"),
        "metrics": details,
        "comparisons": summary["comparisons"],
        "notebook_runtime_seconds": summary["notebook_runtime_seconds"],
    }
    Path(__file__).with_name("context_readout.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
