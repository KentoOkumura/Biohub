"""Compare declared, paired evaluation tables without recomputing official metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any

REQUIRED_CONDITIONS = (
    "evaluation_set",
    "labels",
    "split",
    "evaluator",
    "aggregation_unit",
    "prediction_stage",
    "fixed_inputs",
)
PAIR_COLUMNS = (
    "unit_id",
    "group",
    "metric",
    "direction",
    "baseline",
    "candidate",
    "delta",
    "oriented_delta",
    "change",
)
INTERPRETATION = (
    "Paired unit diagnostics only; unit means are not the official aggregate. "
    "Matching declarations and file hashes do not establish scientific comparability, "
    "independent validation, statistical significance, or experiment acceptance."
)


def nonempty_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def validate_fixed_condition(value: Any, label: str) -> None:
    text = nonempty_text(value, label).strip().casefold()
    marker, separator, reason = text.partition(":")
    placeholders = {"todo", "unknown", "tbd", "none", "null", "未確認", "不明", "未定", "-", "?"}
    if marker.strip() in placeholders:
        raise ValueError(f"{label} is an unverified placeholder")
    if marker.strip() in {"n/a", "na", "not_applicable", "not applicable"} and (
        not separator or not reason.strip() or reason.strip() in placeholders
    ):
        raise ValueError(f"{label} requires a reason for non-applicability")


def load_evaluation(path: Path, provenance: dict[str, Any] | None = None) -> dict[str, Any]:
    provenance = {} if provenance is None else provenance
    provenance["manifest_path"] = str(path.resolve())
    manifest_bytes = path.read_bytes()
    provenance["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes, object_pairs_hook=unique_object)
    # Protect a declared table even when the schema or its hash is invalid.
    if isinstance(manifest, dict) and isinstance(manifest.get("table"), dict):
        declared_path = manifest["table"].get("path")
        if isinstance(declared_path, str) and declared_path.strip():
            provenance["table_path"] = str((path.parent / declared_path).resolve())
    if not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int:
        raise ValueError("manifest must be an object with schema_version=1")
    if manifest["schema_version"] != 1:
        raise ValueError("unsupported schema_version; expected 1")
    table = manifest.get("table")
    if not isinstance(table, dict):
        raise ValueError("table must contain path and sha256")
    relative = Path(nonempty_text(table.get("path"), "table.path"))
    if relative.is_absolute():
        raise ValueError("table.path must be relative to the manifest")
    table_path = (path.parent / relative).resolve()
    provenance["table_path"] = str(table_path)
    nonempty_text(manifest.get("evaluation_kind"), "evaluation_kind")
    unit_column = nonempty_text(manifest.get("unit_column"), "unit_column")
    if "group_column" not in manifest:
        raise ValueError("group_column must be declared; use null when not grouping")
    group_column = manifest["group_column"]
    if group_column is not None:
        nonempty_text(group_column, "group_column")
    metrics = manifest.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError("metrics must be a nonempty column-to-direction object")
    for metric, direction in metrics.items():
        nonempty_text(metric, "metric column")
        if not isinstance(direction, str) or direction not in ("max", "min"):
            raise ValueError(f"metric {metric}: direction must be max or min")
    columns = [unit_column, *([group_column] if group_column is not None else []), *metrics]
    if len(set(columns)) != len(columns):
        raise ValueError("unit, group, and metric columns must be distinct")
    expected = manifest.get("expected_unit_count")
    if type(expected) is not int or expected <= 0:
        raise ValueError("expected_unit_count must be a positive integer")
    conditions = manifest.get("fixed_conditions")
    if not isinstance(conditions, dict):
        raise ValueError("fixed_conditions must be an object")
    for key in REQUIRED_CONDITIONS:
        validate_fixed_condition(conditions.get(key), f"fixed_conditions.{key}")
    for key, value in conditions.items():
        nonempty_text(key, "fixed condition name")
        validate_fixed_condition(value, f"fixed_conditions.{key}")
    expected_sha = nonempty_text(table.get("sha256"), "table.sha256")
    if len(expected_sha) != 64 or any(c not in "0123456789abcdef" for c in expected_sha):
        raise ValueError("table.sha256 must be a lowercase SHA256 hex digest")
    table_bytes = table_path.read_bytes()
    actual_sha = hashlib.sha256(table_bytes).hexdigest()
    provenance["table_sha256"] = actual_sha
    if actual_sha != expected_sha:
        raise ValueError(f"table SHA256 mismatch: expected {expected_sha}, got {actual_sha}")
    reader = csv.DictReader(io.StringIO(table_bytes.decode("utf-8-sig")), strict=True)
    headers = reader.fieldnames
    if not headers or any(not name.strip() for name in headers):
        raise ValueError("CSV must have nonempty column headers")
    if len(set(headers)) != len(headers):
        raise ValueError("CSV has duplicate column headers")
    if missing := sorted(set(columns) - set(headers)):
        raise ValueError(f"CSV missing columns: {missing}")
    rows: dict[str, dict[str, Any]] = {}
    for row in reader:
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"CSV row {reader.line_num} has a different number of fields")
        unit_id = nonempty_text(row[unit_column], f"row {reader.line_num} unit ID")
        if unit_id in rows:
            raise ValueError(f"duplicate unit ID: {unit_id}")
        group = nonempty_text(row[group_column], f"{unit_id} group") if group_column else None
        values = {}
        for metric in metrics:
            try:
                value = float(row[metric])
            except ValueError as exc:
                raise ValueError(f"{unit_id}/{metric}: expected a finite number") from exc
            if not math.isfinite(value):
                raise ValueError(f"{unit_id}/{metric}: expected a finite number")
            values[metric] = value
        rows[unit_id] = {"group": group, "values": values}
    if len(rows) != expected:
        raise ValueError(f"unit count mismatch: expected {expected}, got {len(rows)}")
    provenance["unit_count"] = len(rows)
    return {
        "manifest": manifest,
        "rows": rows,
        "provenance": provenance,
    }


def summarize(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(pairs)
    result: dict[str, Any] = {"unit_count": count}
    for key in ("baseline", "candidate", "delta", "oriented_delta"):
        result[f"unit_mean_{key}"] = math.fsum(row[key] / count for row in pairs)
    for change in ("improved", "worsened", "unchanged"):
        result[change] = sum(row["change"] == change for row in pairs)
    return result


def compare_evaluations(
    baseline: Path, candidate: Path, *, tolerance: float = 0.0
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be a finite nonnegative number")
    report: dict[str, Any] = {
        "schema_version": 1,
        "comparison": "not_comparable",
        "reasons": [],
        "inputs": {},
        "tolerance": tolerance,
        "interpretation": INTERPRETATION,
    }
    loaded = {}
    for name, path in (("baseline", baseline), ("candidate", candidate)):
        report["inputs"][name] = {"manifest_path": str(path.resolve())}
        try:
            loaded[name] = load_evaluation(path, report["inputs"][name])
        except (OSError, ValueError, csv.Error) as exc:
            report["reasons"].append(f"{name}: {exc}")
    if report["reasons"]:
        return report, []
    left, right = loaded["baseline"], loaded["candidate"]
    for key in ("evaluation_kind", "unit_column", "group_column", "metrics", "expected_unit_count"):
        if left["manifest"][key] != right["manifest"][key]:
            report["reasons"].append(f"manifest mismatch: {key}")
    left_conditions = left["manifest"]["fixed_conditions"]
    right_conditions = right["manifest"]["fixed_conditions"]
    for key in sorted(left_conditions.keys() | right_conditions.keys()):
        if left_conditions.get(key) != right_conditions.get(key):
            report["reasons"].append(f"fixed condition mismatch: {key}")
    left_ids, right_ids = set(left["rows"]), set(right["rows"])
    for label, missing in (("candidate", left_ids - right_ids), ("baseline", right_ids - left_ids)):
        if missing:
            report["reasons"].append(f"units missing from {label}: {sorted(missing)}")
    for unit_id in sorted(left_ids & right_ids):
        if left["rows"][unit_id]["group"] != right["rows"][unit_id]["group"]:
            report["reasons"].append(f"group mismatch for unit: {unit_id}")
    if report["reasons"]:
        return report, []
    pairs = []
    for unit_id in sorted(left_ids):
        for metric, direction in left["manifest"]["metrics"].items():
            baseline_value = left["rows"][unit_id]["values"][metric]
            candidate_value = right["rows"][unit_id]["values"][metric]
            delta = candidate_value - baseline_value
            if not math.isfinite(delta):
                report["reasons"].append(f"difference overflow for {unit_id}/{metric}")
                return report, []
            oriented = delta if direction == "max" else -delta
            change = "unchanged"
            if oriented > tolerance:
                change = "improved"
            elif oriented < -tolerance:
                change = "worsened"
            pairs.append(
                dict(
                    zip(
                        PAIR_COLUMNS,
                        (
                            unit_id,
                            left["rows"][unit_id]["group"],
                            metric,
                            direction,
                            baseline_value,
                            candidate_value,
                            delta,
                            oriented,
                            change,
                        ),
                        strict=True,
                    )
                )
            )
    diagnostics = {}
    for metric, direction in left["manifest"]["metrics"].items():
        metric_pairs = [row for row in pairs if row["metric"] == metric]
        groups = {}
        if left["manifest"]["group_column"] is not None:
            for group in sorted({row["group"] for row in metric_pairs}):
                groups[group] = summarize([row for row in metric_pairs if row["group"] == group])
        diagnostics[metric] = {
            "direction": direction,
            "overall": summarize(metric_pairs),
            "by_group": groups,
        }
    report.update(
        comparison="comparable",
        evaluation_kind=left["manifest"]["evaluation_kind"],
        fixed_conditions=left_conditions,
        diagnostics=diagnostics,
    )
    return report, pairs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--tolerance", type=float, default=0.0)
    args = parser.parse_args(argv)
    try:
        report, pairs = compare_evaluations(args.baseline, args.candidate, tolerance=args.tolerance)
        if args.output_dir is not None and any(
            "table_path" not in item for item in report["inputs"].values()
        ):
            report["reasons"].append(
                "output not written: unable to identify all input tables safely; "
                "any previous output files were left unchanged"
            )
            print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
            return 2
        rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.output_dir is not None:
            report_path = args.output_dir / "comparison.json"
            pairs_path = args.output_dir / "paired.csv"
            protected = {args.baseline.resolve(), args.candidate.resolve()}
            protected.update(
                Path(item["table_path"]).resolve()
                for item in report["inputs"].values()
                if "table_path" in item
            )
            if {report_path.resolve(), pairs_path.resolve()} & protected:
                raise ValueError("output paths would overwrite a comparison input")
            args.output_dir.mkdir(parents=True, exist_ok=True)
            # Remove an earlier comparison's pairs even when this run is incompatible.
            pairs_path.unlink(missing_ok=True)
            report_path.write_text(rendered, encoding="utf-8")
            if pairs:
                with pairs_path.open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=PAIR_COLUMNS)
                    writer.writeheader()
                    writer.writerows(pairs)
        print(rendered, end="")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0 if report["comparison"] == "comparable" else 2


if __name__ == "__main__":
    raise SystemExit(main())
