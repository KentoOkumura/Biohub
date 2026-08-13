#!/usr/bin/env python3
"""Build a schema-parameterized OOF error readout."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rmse(values: pd.Series) -> float:
    array = values.to_numpy(dtype=float)
    return float(np.sqrt(np.mean(np.square(array)))) if array.size else float("nan")


def metric_row(frame: pd.DataFrame, *, name: str, error_column: str) -> dict[str, Any]:
    error = frame[error_column]
    return {
        "prediction": name,
        "rows": int(len(frame)),
        "rmse": rmse(error),
        "mae": float(error.abs().mean()),
        "mean_error": float(error.mean()),
        "sse": float(np.square(error).sum()),
    }


def grouped_metrics(
    frame: pd.DataFrame,
    *,
    group_column: str,
    prediction_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_value, group in frame.groupby(group_column, dropna=False, observed=False):
        row: dict[str, Any] = {group_column: group_value, "rows": int(len(group))}
        for prediction in prediction_columns:
            error_column = f"__error_{prediction}"
            row[f"{prediction}_rmse"] = rmse(group[error_column])
            row[f"{prediction}_mae"] = float(group[error_column].abs().mean())
        rows.append(row)
    return pd.DataFrame(rows)


def feature_quantile_metrics(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    prediction_columns: list[str],
    bins: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in feature_columns:
        values = pd.to_numeric(frame[feature], errors="coerce")
        valid = frame.loc[values.notna()].copy()
        if valid.empty:
            continue
        try:
            valid["__feature_bin"] = pd.qcut(
                values.loc[values.notna()], q=bins, duplicates="drop"
            )
        except ValueError:
            continue
        for label, group in valid.groupby("__feature_bin", observed=False):
            if group.empty:
                continue
            row: dict[str, Any] = {
                "feature": feature,
                "feature_bin": str(label),
                "rows": int(len(group)),
            }
            for prediction in prediction_columns:
                error_column = f"__error_{prediction}"
                row[f"{prediction}_rmse"] = rmse(group[error_column])
                row[f"{prediction}_mae"] = float(group[error_column].abs().mean())
            rows.append(row)
    return pd.DataFrame(rows)


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_json(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, (np.integer, np.floating)):
        return clean_json(value.item())
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--id-column", required=True)
    parser.add_argument("--target-column", required=True)
    parser.add_argument("--prediction-column", required=True)
    parser.add_argument("--compare-prediction-column")
    parser.add_argument("--group-column")
    parser.add_argument("--bucket-column", action="append", default=[])
    parser.add_argument("--feature-column", action="append", default=[])
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--quantile-bins", type=int, default=5)
    parser.add_argument("--output-prefix", default="oof_readout")
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, columns: list[str], *, source: Path) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def main() -> None:
    args = parse_args()
    if not args.oof.is_file() or args.oof.stat().st_size == 0:
        raise FileNotFoundError(f"OOF file is missing or empty: {args.oof}")

    frame = pd.read_csv(args.oof)
    prediction_columns = [args.prediction_column]
    if args.compare_prediction_column:
        prediction_columns.append(args.compare_prediction_column)
    required = [args.id_column, args.target_column, *prediction_columns]
    if args.group_column:
        required.append(args.group_column)
    required.extend(args.bucket_column)
    require_columns(frame, required, source=args.oof)
    if frame[args.id_column].duplicated().any():
        raise ValueError(f"{args.oof} contains duplicate IDs in {args.id_column}")

    inputs = {"oof": str(args.oof), "oof_sha256": sha256_file(args.oof)}
    if args.feature_cache:
        if not args.feature_cache.is_file() or args.feature_cache.stat().st_size == 0:
            raise FileNotFoundError(f"Feature cache is missing or empty: {args.feature_cache}")
        cache_columns = [args.id_column, *args.feature_column]
        cache = pd.read_csv(args.feature_cache, usecols=cache_columns)
        if cache[args.id_column].duplicated().any():
            raise ValueError(
                f"{args.feature_cache} contains duplicate IDs in {args.id_column}"
            )
        frame = frame.merge(cache, on=args.id_column, how="left", validate="one_to_one")
        inputs.update(
            {
                "feature_cache": str(args.feature_cache),
                "feature_cache_sha256": sha256_file(args.feature_cache),
            }
        )
    else:
        require_columns(frame, args.feature_column, source=args.oof)

    target = pd.to_numeric(frame[args.target_column], errors="raise")
    for prediction in prediction_columns:
        frame[f"__error_{prediction}"] = (
            pd.to_numeric(frame[prediction], errors="raise") - target
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}
    overall = pd.DataFrame(
        [
            metric_row(frame, name=prediction, error_column=f"__error_{prediction}")
            for prediction in prediction_columns
        ]
    )
    overall_path = args.output_dir / f"{args.output_prefix}_overall_metrics.csv"
    overall.to_csv(overall_path, index=False)
    artifacts["overall_metrics"] = overall_path.name

    if args.group_column:
        groups = grouped_metrics(
            frame,
            group_column=args.group_column,
            prediction_columns=prediction_columns,
        )
        group_path = args.output_dir / f"{args.output_prefix}_group_metrics.csv"
        groups.to_csv(group_path, index=False)
        artifacts["group_metrics"] = group_path.name

    for bucket_column in args.bucket_column:
        buckets = grouped_metrics(
            frame,
            group_column=bucket_column,
            prediction_columns=prediction_columns,
        )
        bucket_path = args.output_dir / f"{args.output_prefix}_{bucket_column}_metrics.csv"
        buckets.to_csv(bucket_path, index=False)
        artifacts[f"bucket_{bucket_column}"] = bucket_path.name

    if args.feature_column:
        feature_metrics = feature_quantile_metrics(
            frame,
            feature_columns=args.feature_column,
            prediction_columns=prediction_columns,
            bins=args.quantile_bins,
        )
        feature_path = args.output_dir / f"{args.output_prefix}_feature_quantile_metrics.csv"
        feature_metrics.to_csv(feature_path, index=False)
        artifacts["feature_quantile_metrics"] = feature_path.name

    summary = clean_json(
        {
            "status": "readout_completed",
            "rows": int(len(frame)),
            "groups": int(frame[args.group_column].nunique()) if args.group_column else None,
            "columns": {
                "id": args.id_column,
                "target": args.target_column,
                "predictions": prediction_columns,
                "group": args.group_column,
                "buckets": args.bucket_column,
                "features": args.feature_column,
            },
            "inputs": inputs,
            "metrics": overall.to_dict(orient="records"),
            "artifacts": artifacts,
        }
    )
    summary_path = args.output_dir / f"{args.output_prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
