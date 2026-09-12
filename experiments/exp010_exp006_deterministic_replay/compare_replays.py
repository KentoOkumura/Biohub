from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

EXPERIMENT = "exp010_exp006_deterministic_replay"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two full exp010 Kaggle replays.")
    parser.add_argument("--run-1-train", type=Path, required=True)
    parser.add_argument("--run-1-inference", type=Path, required=True)
    parser.add_argument("--run-2-train", type=Path, required=True)
    parser.add_argument("--run-2-inference", type=Path, required=True)
    parser.add_argument("--run-1-train-version", type=int, required=True)
    parser.add_argument("--run-1-inference-version", type=int, required=True)
    parser.add_argument("--run-2-train-version", type=int, required=True)
    parser.add_argument("--run-2-inference-version", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise TypeError(f"expected object in {path}")
    return payload


def fold_map(summary: dict[str, Any], key: str) -> dict[str, Any]:
    records = summary.get("folds", [])
    if len(records) != 2:
        raise ValueError(f"expected two fold records, found {len(records)}")
    return {str(int(record["fold"])): record[key] for record in records}


def comparison_values(
    train_summary: dict[str, Any],
    inference_summary: dict[str, Any],
) -> dict[str, Any]:
    if train_summary.get("experiment") != EXPERIMENT:
        raise ValueError("training summary belongs to another experiment")
    if inference_summary.get("experiment") != EXPERIMENT:
        raise ValueError("inference summary belongs to another experiment")
    if int(inference_summary.get("prediction_count", -1)) != 199:
        raise ValueError("inference summary does not cover 199 held-out videos")
    return {
        "config_sha256": train_summary["config_sha256"],
        "source_manifest_sha256": train_summary["source_manifest_sha256"],
        "split_manifest_sha256": train_summary["split_manifest_sha256"],
        "checkpoint_content_sha256_by_fold": fold_map(train_summary, "checkpoint_content_sha256"),
        "model_config_sha256_by_fold": fold_map(train_summary, "model_config_sha256"),
        "inference_input_contract_sha256": inference_summary["input_contract_sha256"],
        "candidate_content_sha256": inference_summary["candidate_content_sha256"],
        "oof_prediction_content_sha256": inference_summary["oof_prediction_content_sha256"],
        "per_sample_metric_content_sha256": inference_summary["per_sample_metric_content_sha256"],
        "official_metric_summary_content_sha256": inference_summary[
            "official_metric_summary_content_sha256"
        ],
        "cv": inference_summary["cv"],
        "prediction_count": inference_summary["prediction_count"],
    }


def compare_runs(
    run_1_train: dict[str, Any],
    run_1_inference: dict[str, Any],
    run_2_train: dict[str, Any],
    run_2_inference: dict[str, Any],
) -> dict[str, Any]:
    run_1 = comparison_values(run_1_train, run_1_inference)
    run_2 = comparison_values(run_2_train, run_2_inference)
    fields = {
        key: {"equal": run_1[key] == run_2[key], "run_1": run_1[key], "run_2": run_2[key]}
        for key in run_1
    }
    return {
        "experiment": EXPERIMENT,
        "passed": all(item["equal"] for item in fields.values()),
        "comparison": fields,
    }


def main() -> None:
    args = parse_args()
    result = compare_runs(
        load_json(args.run_1_train, "artifacts/training_summary.json"),
        load_json(args.run_1_inference, "artifacts/inference_summary.json"),
        load_json(args.run_2_train, "artifacts/training_summary.json"),
        load_json(args.run_2_inference, "artifacts/inference_summary.json"),
    )
    result["runs"] = {
        "run_1": {
            "train_kernel_version": args.run_1_train_version,
            "inference_kernel_version": args.run_1_inference_version,
        },
        "run_2": {
            "train_kernel_version": args.run_2_train_version,
            "inference_kernel_version": args.run_2_inference_version,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
