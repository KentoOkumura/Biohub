from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

RECEIPT_NAME = "replay_receipt.json"
INPUT_MANIFEST_NAME = "replay_input_manifest.json"
SUBMISSION_NAME = "submission.csv"


def json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def resolve_run_dir(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_dir() else path.parent


def validate_run_dir(value: str | Path) -> dict[str, Any]:
    run_dir = resolve_run_dir(value)
    receipt_path = run_dir / RECEIPT_NAME
    input_manifest_path = run_dir / INPUT_MANIFEST_NAME
    submission_path = run_dir / SUBMISSION_NAME
    missing = [
        str(path)
        for path in (receipt_path, input_manifest_path, submission_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError({"missing_replay_files": missing})

    receipt = load_json_object(receipt_path)
    input_manifest = load_json_object(input_manifest_path)
    stored_input_sha = input_manifest.pop("input_manifest_sha256", None)
    actual_input_sha = json_sha256(input_manifest)
    if stored_input_sha != actual_input_sha:
        raise ValueError(
            {
                "input_manifest_sha_mismatch": {
                    "stored": stored_input_sha,
                    "actual": actual_input_sha,
                }
            }
        )
    if receipt.get("input_manifest_sha256") != actual_input_sha:
        raise ValueError("Receipt does not reference the validated input manifest")

    actual_submission_sha = file_sha256(submission_path)
    if receipt.get("submission_sha256") != actual_submission_sha:
        raise ValueError(
            {
                "submission_sha_mismatch": {
                    "stored": receipt.get("submission_sha256"),
                    "actual": actual_submission_sha,
                }
            }
        )

    comparison_fields = receipt.get("comparison_fields")
    if not isinstance(comparison_fields, list) or not comparison_fields:
        raise ValueError("Replay receipt has no comparison_fields contract")
    missing_fields = [field for field in comparison_fields if field not in receipt]
    if missing_fields:
        raise ValueError({"missing_comparison_fields": missing_fields})
    return receipt


def compare_run_dirs(reference: str | Path, rerun: str | Path) -> dict[str, Any]:
    reference_receipt = validate_run_dir(reference)
    rerun_receipt = validate_run_dir(rerun)
    reference_fields = reference_receipt["comparison_fields"]
    rerun_fields = rerun_receipt["comparison_fields"]
    if rerun_fields != reference_fields:
        raise ValueError(
            {
                "comparison_contract_mismatch": {
                    "reference": reference_fields,
                    "rerun": rerun_fields,
                }
            }
        )

    fields: dict[str, dict[str, Any]] = {}
    for field in reference_fields:
        expected = reference_receipt[field]
        actual = rerun_receipt[field]
        fields[field] = {
            "match": actual == expected,
            "reference": expected,
            "rerun": actual,
        }
    mismatches = [field for field, result in fields.items() if not result["match"]]
    return {
        "schema_version": 1,
        "experiment": "exp015_oracle_stage_limits",
        "reference_receipt_sha256": reference_receipt.get("receipt_sha256"),
        "rerun_receipt_sha256": rerun_receipt.get("receipt_sha256"),
        "byte_identical_to_reference": not mismatches,
        "fields": fields,
        "mismatches": mismatches,
        "difference_notes": None
        if not mismatches
        else f"Different fields: {', '.join(mismatches)}",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two exp013 Kaggle output directories.")
    parser.add_argument("reference", help="First Kaggle output directory")
    parser.add_argument("rerun", help="Second Kaggle output directory")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = compare_run_dirs(args.reference, args.rerun)
    rendered = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(args.output)
    else:
        print(rendered, end="")
    return 0 if report["byte_identical_to_reference"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
