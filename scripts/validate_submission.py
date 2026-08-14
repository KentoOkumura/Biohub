from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from config_utils import ROOT, configured_project_path, get_nested, is_todo, load_project_config
from update_experiment_summary import collect_records, render_auto_block, update_summary

MISSING_VALUE_STRINGS = {"", "na", "n/a", "nan", "none", "null"}
INFINITE_VALUE_STRINGS = {"inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}
BIOHUB_COMPETITION_SLUG = "biohub-cell-tracking-during-development"
BIOHUB_SUBMISSION_COLUMNS = [
    "id",
    "dataset",
    "row_type",
    "node_id",
    "t",
    "z",
    "y",
    "x",
    "source_id",
    "target_id",
]
BIOHUB_NUMERIC_COLUMNS = [
    "id",
    "node_id",
    "t",
    "z",
    "y",
    "x",
    "source_id",
    "target_id",
]
BIOHUB_INTEGER_COLUMNS = ["id", "node_id", "t", "source_id", "target_id"]
BIOHUB_TARGET_COLUMNS = [
    "node_id",
    "t",
    "z",
    "y",
    "x",
    "source_id",
    "target_id",
]


@dataclass
class SubmissionValidationReport:
    passed: bool
    submission: str
    sample: str
    row_count: int | None
    sample_row_count: int | None
    id_column: str | None
    target_columns: list[str]
    duplicate_id_count: int
    missing_value_count: int
    infinite_value_count: int
    submission_sha256: str | None
    target_statistics: dict[str, dict[str, float]]
    errors: list[str]

    def evidence(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "passed": self.passed,
            "checked_at": datetime.now(UTC).isoformat(),
            "row_count": self.row_count,
            "id_column": self.id_column,
            "target_columns": self.target_columns,
            "duplicate_id_count": self.duplicate_id_count,
            "missing_value_count": self.missing_value_count,
            "infinite_value_count": self.infinite_value_count,
            "target_statistics": self.target_statistics,
            "errors": self.errors,
        }
        if len(self.target_columns) == 1:
            statistics = self.target_statistics.get(self.target_columns[0], {})
            value.update(
                {
                    "prediction_min": statistics.get("min"),
                    "prediction_max": statistics.get("max"),
                    "prediction_mean": statistics.get("mean"),
                    "prediction_std": statistics.get("std"),
                }
            )
        return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a submission CSV against sample submission."
    )
    parser.add_argument("--submission", required=True, help="Submission CSV path")
    parser.add_argument("--sample", default=None, help="Override sample submission path")
    parser.add_argument(
        "--test-dir",
        default=None,
        help=(
            "Override the test input directory. For Biohub, existing .zarr metadata "
            "enables dataset coverage and coordinate-bound checks."
        ),
    )
    parser.add_argument(
        "--experiment",
        default="",
        help="Record the validation result in this experiment's metrics.json when set.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format. JSON is intended for other repository scripts.",
    )
    return parser.parse_args()


def resolve_path(path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_header(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        try:
            return next(reader)
        except StopIteration:
            return []


def read_csv_as_text(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def normalized_strings(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.apply(lambda column: column.astype(str).str.strip().str.lower())


def load_biohub_test_shapes(test_dir: Path) -> dict[str, tuple[int, int, int, int] | None]:
    if not test_dir.is_dir():
        raise ValueError(f"test directory not found: {display_path(test_dir)}")
    zarr_paths = sorted(
        path for path in test_dir.iterdir() if path.is_dir() and path.name.endswith(".zarr")
    )
    if not zarr_paths:
        raise ValueError(f"no test .zarr directories found: {display_path(test_dir)}")

    shapes: dict[str, tuple[int, int, int, int] | None] = {}
    for zarr_path in zarr_paths:
        metadata_path = zarr_path / "0" / "zarr.json"
        shape: tuple[int, int, int, int] | None = None
        if metadata_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text())
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid Zarr metadata: {display_path(metadata_path)}: {exc}"
                ) from exc
            raw_shape = metadata.get("shape") if isinstance(metadata, dict) else None
            if (
                isinstance(raw_shape, list)
                and len(raw_shape) == 4
                and all(isinstance(value, int) and value > 0 for value in raw_shape)
            ):
                shape = (raw_shape[0], raw_shape[1], raw_shape[2], raw_shape[3])
        shapes[zarr_path.name.removesuffix(".zarr")] = shape
    return shapes


def validate_biohub_tracking_submission(
    submission_path: Path,
    sample_path: Path,
    *,
    expected_shapes: dict[str, tuple[int, int, int, int] | None] | None = None,
) -> SubmissionValidationReport:
    """Validate Biohub's variable-row node/edge tracking graph submission."""
    errors: list[str] = []
    row_count: int | None = None
    sample_row_count: int | None = None
    duplicate_id_count = 0
    missing_value_count = 0
    infinite_value_count = 0
    submission_sha256: str | None = None
    target_statistics: dict[str, dict[str, float]] = {}

    if not submission_path.is_file():
        errors.append(f"submission not found: {display_path(submission_path)}")
        return SubmissionValidationReport(
            False,
            display_path(submission_path),
            display_path(sample_path),
            row_count,
            sample_row_count,
            "id",
            BIOHUB_TARGET_COLUMNS,
            duplicate_id_count,
            missing_value_count,
            infinite_value_count,
            submission_sha256,
            target_statistics,
            errors,
        )

    submission_sha256 = file_sha256(submission_path)
    submission_header = read_header(submission_path)
    if not submission_header:
        errors.append("submission CSV is empty")
    if len(submission_header) != len(set(submission_header)):
        errors.append("submission contains duplicate column names")
    if submission_header != BIOHUB_SUBMISSION_COLUMNS:
        errors.append(
            "column order/content mismatch: "
            f"expected={BIOHUB_SUBMISSION_COLUMNS}, submission={submission_header}"
        )

    if sample_path.is_file():
        sample_header = read_header(sample_path)
        if sample_header != BIOHUB_SUBMISSION_COLUMNS:
            errors.append(
                "sample submission does not use the expected Biohub columns: "
                f"sample={sample_header}"
            )
        try:
            sample_row_count = len(read_csv_as_text(sample_path))
        except Exception as exc:  # pandas exposes several parser/encoding errors
            errors.append(f"could not read sample submission: {exc}")

    try:
        submission = read_csv_as_text(submission_path)
    except Exception as exc:  # pandas exposes several parser/encoding errors
        errors.append(f"could not read submission: {exc}")
        submission = None
    if submission is None:
        return SubmissionValidationReport(
            False,
            display_path(submission_path),
            display_path(sample_path),
            row_count,
            sample_row_count,
            "id",
            BIOHUB_TARGET_COLUMNS,
            duplicate_id_count,
            missing_value_count,
            infinite_value_count,
            submission_sha256,
            target_statistics,
            errors,
        )

    row_count = len(submission)
    if row_count == 0:
        errors.append("submission contains no node or edge rows")
    if list(submission.columns) != BIOHUB_SUBMISSION_COLUMNS:
        return SubmissionValidationReport(
            False,
            display_path(submission_path),
            display_path(sample_path),
            row_count,
            sample_row_count,
            "id",
            BIOHUB_TARGET_COLUMNS,
            duplicate_id_count,
            missing_value_count,
            infinite_value_count,
            submission_sha256,
            target_statistics,
            errors,
        )

    normalized = normalized_strings(submission)
    missing_mask = normalized.isin(MISSING_VALUE_STRINGS)
    missing_value_count = int(missing_mask.to_numpy().sum())
    if missing_value_count:
        columns = submission.columns[missing_mask.any()].tolist()
        errors.append(f"missing values found: count={missing_value_count}, columns={columns}")

    parsed_numeric: dict[str, pd.Series] = {}
    numeric_columns_valid = True
    for column in BIOHUB_NUMERIC_COLUMNS:
        values = normalized[column]
        numeric = pd.to_numeric(submission[column].str.strip(), errors="coerce")
        infinite_mask = values.isin(INFINITE_VALUE_STRINGS) | numeric.isin(
            [float("inf"), float("-inf")]
        )
        infinite_value_count += int(infinite_mask.sum())
        non_numeric_mask = numeric.isna() & ~values.isin(MISSING_VALUE_STRINGS)
        if int(non_numeric_mask.sum()):
            errors.append(
                f"non-numeric values found in column {column}: "
                f"count={int(non_numeric_mask.sum())}"
            )
        finite_mask = ~numeric.isna() & ~infinite_mask
        non_integer_mask = finite_mask & (numeric % 1 != 0)
        if column in BIOHUB_INTEGER_COLUMNS and int(non_integer_mask.sum()):
            errors.append(
                f"non-integer values found in column {column}: "
                f"count={int(non_integer_mask.sum())}"
            )
        if not finite_mask.all() or (
            column in BIOHUB_INTEGER_COLUMNS and bool(non_integer_mask.any())
        ):
            numeric_columns_valid = False
        parsed_numeric[column] = numeric
        finite_numeric = numeric[finite_mask]
        if column in BIOHUB_TARGET_COLUMNS and not finite_numeric.empty:
            target_statistics[column] = {
                "min": float(finite_numeric.min()),
                "max": float(finite_numeric.max()),
                "mean": float(finite_numeric.mean()),
                "std": float(finite_numeric.std(ddof=0)),
            }
    if infinite_value_count:
        errors.append(f"infinite values found: count={infinite_value_count}")

    dataset_values = submission["dataset"].astype(str)
    datasets = dataset_values.str.strip()
    if bool((dataset_values != datasets).any()):
        errors.append("dataset values must not contain leading or trailing whitespace")
    row_type_values = submission["row_type"].astype(str)
    row_types = row_type_values.str.strip().str.lower()
    if bool((row_type_values != row_types).any()):
        errors.append("row_type values must be exactly 'node' or 'edge'")
    invalid_row_types = sorted(set(row_types) - {"node", "edge"})
    if invalid_row_types:
        errors.append(f"invalid row_type values: {invalid_row_types}")

    actual_datasets = set(datasets)
    if expected_shapes is not None:
        expected_datasets = set(expected_shapes)
        missing_datasets = sorted(expected_datasets - actual_datasets)
        extra_datasets = sorted(actual_datasets - expected_datasets)
        if missing_datasets:
            errors.append(f"missing test datasets: {missing_datasets}")
        if extra_datasets:
            errors.append(f"unexpected datasets: {extra_datasets}")

    if numeric_columns_valid:
        graph = submission.copy()
        graph["dataset"] = datasets
        graph["row_type"] = row_types
        for column, numeric in parsed_numeric.items():
            dtype = "int64" if column in BIOHUB_INTEGER_COLUMNS else "float64"
            graph[column] = numeric.astype(dtype)

        duplicate_id_count = int(graph["id"].duplicated(keep=False).sum())
        if duplicate_id_count:
            errors.append(
                "duplicate IDs found in id: "
                f"rows_in_duplicate_groups={duplicate_id_count}"
            )
        expected_ids = list(range(row_count))
        if graph["id"].tolist() != expected_ids:
            errors.append("id must be consecutive integers from 0 in row order")

        node_rows = graph[graph["row_type"] == "node"].copy()
        edge_rows = graph[graph["row_type"] == "edge"].copy()
        if node_rows.empty:
            errors.append("submission contains no node rows")
        datasets_without_nodes = sorted(actual_datasets - set(node_rows["dataset"]))
        if datasets_without_nodes:
            errors.append(f"datasets without node rows: {datasets_without_nodes}")

        bad_node_sentinels = (node_rows[["source_id", "target_id"]] != -1).any(axis=1)
        if int(bad_node_sentinels.sum()):
            errors.append(
                "node rows must use source_id=target_id=-1: "
                f"count={int(bad_node_sentinels.sum())}"
            )
        bad_node_values = (node_rows[["node_id", "t", "z", "y", "x"]] < 0).any(axis=1)
        if int(bad_node_values.sum()):
            errors.append(
                "node rows must use non-negative node_id,t,z,y,x: "
                f"count={int(bad_node_values.sum())}"
            )
        bad_edge_sentinels = (edge_rows[["node_id", "t", "z", "y", "x"]] != -1).any(
            axis=1
        )
        if int(bad_edge_sentinels.sum()):
            errors.append(
                "edge rows must use node_id=t=z=y=x=-1: "
                f"count={int(bad_edge_sentinels.sum())}"
            )
        bad_edge_values = (edge_rows[["source_id", "target_id"]] < 0).any(axis=1)
        if int(bad_edge_values.sum()):
            errors.append(
                "edge rows must use non-negative source_id,target_id: "
                f"count={int(bad_edge_values.sum())}"
            )

        duplicate_nodes = node_rows.duplicated(["dataset", "node_id"], keep=False)
        if int(duplicate_nodes.sum()):
            errors.append(
                "duplicate node_id values within a dataset: "
                f"rows_in_duplicate_groups={int(duplicate_nodes.sum())}"
            )
        duplicate_edges = edge_rows.duplicated(
            ["dataset", "source_id", "target_id"], keep=False
        )
        if int(duplicate_edges.sum()):
            errors.append(
                "duplicate edges within a dataset: "
                f"rows_in_duplicate_groups={int(duplicate_edges.sum())}"
            )
        self_loops = edge_rows["source_id"] == edge_rows["target_id"]
        if int(self_loops.sum()):
            errors.append(f"self-loop edges found: count={int(self_loops.sum())}")

        if not bool(duplicate_nodes.any()) and not edge_rows.empty:
            node_keys = pd.MultiIndex.from_frame(node_rows[["dataset", "node_id"]])
            source_keys = pd.MultiIndex.from_frame(
                edge_rows[["dataset", "source_id"]].rename(
                    columns={"source_id": "node_id"}
                )
            )
            target_keys = pd.MultiIndex.from_frame(
                edge_rows[["dataset", "target_id"]].rename(
                    columns={"target_id": "node_id"}
                )
            )
            missing_sources = ~source_keys.isin(node_keys)
            missing_targets = ~target_keys.isin(node_keys)
            if int(missing_sources.sum()):
                errors.append(
                    f"edges reference missing source nodes: count={int(missing_sources.sum())}"
                )
            if int(missing_targets.sum()):
                errors.append(
                    f"edges reference missing target nodes: count={int(missing_targets.sum())}"
                )
            valid_edges = ~(missing_sources | missing_targets)
            if bool(valid_edges.any()):
                node_times = pd.Series(node_rows["t"].to_numpy(), index=node_keys)
                source_times = node_times.reindex(source_keys[valid_edges]).to_numpy()
                target_times = node_times.reindex(target_keys[valid_edges]).to_numpy()
                nonconsecutive = target_times != source_times + 1
                if int(nonconsecutive.sum()):
                    errors.append(
                        "edges must connect t to t+1 within a dataset: "
                        f"count={int(nonconsecutive.sum())}"
                    )

        if expected_shapes is not None:
            for dataset, shape in expected_shapes.items():
                if shape is None:
                    continue
                dataset_nodes = node_rows[node_rows["dataset"] == dataset]
                if dataset_nodes.empty:
                    continue
                limits = dict(zip(("t", "z", "y", "x"), shape, strict=True))
                for column, limit in limits.items():
                    out_of_bounds = dataset_nodes[column] >= limit
                    if int(out_of_bounds.sum()):
                        errors.append(
                            f"{dataset}: {column} is outside [0, {limit}): "
                            f"count={int(out_of_bounds.sum())}"
                        )

    return SubmissionValidationReport(
        not errors,
        display_path(submission_path),
        display_path(sample_path),
        row_count,
        sample_row_count,
        "id",
        BIOHUB_TARGET_COLUMNS,
        duplicate_id_count,
        missing_value_count,
        infinite_value_count,
        submission_sha256,
        target_statistics,
        errors,
    )


def validate_submission_files(
    submission_path: Path,
    sample_path: Path,
    *,
    id_column: str | None,
    target_columns: list[str],
    allow_extra_columns: bool,
) -> SubmissionValidationReport:
    errors: list[str] = []
    row_count: int | None = None
    sample_row_count: int | None = None
    duplicate_id_count = 0
    missing_value_count = 0
    infinite_value_count = 0
    submission_sha256: str | None = None
    target_statistics: dict[str, dict[str, float]] = {}

    if not sample_path.is_file():
        errors.append(f"sample submission not found: {display_path(sample_path)}")
    if not submission_path.is_file():
        errors.append(f"submission not found: {display_path(submission_path)}")
    if errors:
        return SubmissionValidationReport(
            False,
            display_path(submission_path),
            display_path(sample_path),
            row_count,
            sample_row_count,
            id_column,
            target_columns,
            duplicate_id_count,
            missing_value_count,
            infinite_value_count,
            submission_sha256,
            target_statistics,
            errors,
        )

    submission_sha256 = file_sha256(submission_path)
    sample_header = read_header(sample_path)
    submission_header = read_header(submission_path)
    if not sample_header:
        errors.append("sample submission CSV is empty")
    if not submission_header:
        errors.append("submission CSV is empty")
    if len(sample_header) != len(set(sample_header)):
        errors.append("sample submission contains duplicate column names")
    if len(submission_header) != len(set(submission_header)):
        errors.append("submission contains duplicate column names")

    try:
        sample = read_csv_as_text(sample_path)
    except Exception as exc:  # pandas exposes parser/encoding errors through several types
        errors.append(f"could not read sample submission: {exc}")
        sample = None
    try:
        submission = read_csv_as_text(submission_path)
    except Exception as exc:  # pandas exposes parser/encoding errors through several types
        errors.append(f"could not read submission: {exc}")
        submission = None

    if sample is not None:
        sample_row_count = len(sample)
    if submission is not None:
        row_count = len(submission)
    if sample is None or submission is None:
        return SubmissionValidationReport(
            False,
            display_path(submission_path),
            display_path(sample_path),
            row_count,
            sample_row_count,
            id_column,
            target_columns,
            duplicate_id_count,
            missing_value_count,
            infinite_value_count,
            submission_sha256,
            target_statistics,
            errors,
        )

    if row_count != sample_row_count:
        errors.append(f"row count mismatch: sample={sample_row_count}, submission={row_count}")

    sample_columns = list(sample.columns)
    submission_columns = list(submission.columns)
    missing_columns = [column for column in sample_columns if column not in submission_columns]
    extra_columns = [column for column in submission_columns if column not in sample_columns]
    if missing_columns:
        errors.append(f"missing columns: {missing_columns}")
    if extra_columns and not allow_extra_columns:
        errors.append(f"extra columns: {extra_columns}")
    if not allow_extra_columns and submission_columns != sample_columns:
        errors.append(
            "column order/content mismatch: "
            f"sample={sample_columns}, submission={submission_columns}"
        )
    elif allow_extra_columns:
        projected_columns = [column for column in submission_columns if column in sample_columns]
        if projected_columns != sample_columns:
            errors.append(
                "configured sample columns are not in sample order: "
                f"sample={sample_columns}, submission={submission_columns}"
            )

    if not id_column:
        errors.append("submission.id_column is not configured")
    elif id_column not in sample.columns or id_column not in submission.columns:
        errors.append(f"id column is missing from sample or submission: {id_column}")
    else:
        duplicate_id_count = int(submission[id_column].duplicated(keep=False).sum())
        if duplicate_id_count:
            errors.append(
                f"duplicate IDs found in {id_column}: rows_in_duplicate_groups={duplicate_id_count}"
            )
        sample_duplicate_count = int(sample[id_column].duplicated(keep=False).sum())
        if sample_duplicate_count:
            errors.append(
                f"sample submission contains duplicate IDs in {id_column}: "
                f"rows_in_duplicate_groups={sample_duplicate_count}"
            )
        if sample[id_column].tolist() != submission[id_column].tolist():
            errors.append(f"id column order/content mismatch: {id_column}")

    normalized = normalized_strings(submission)
    missing_mask = normalized.isin(MISSING_VALUE_STRINGS)
    missing_value_count = int(missing_mask.to_numpy().sum())
    if missing_value_count:
        columns = submission.columns[missing_mask.any()].tolist()
        errors.append(f"missing values found: count={missing_value_count}, columns={columns}")

    if not target_columns:
        errors.append("submission.target_columns is not configured")
    for target in target_columns:
        if target not in submission.columns:
            if target not in missing_columns:
                errors.append(f"configured target column is missing: {target}")
            continue
        values = normalized[target]
        numeric = pd.to_numeric(submission[target].str.strip(), errors="coerce")
        infinite_mask = values.isin(INFINITE_VALUE_STRINGS) | numeric.isin(
            [float("inf"), float("-inf")]
        )
        infinite_value_count += int(infinite_mask.sum())
        non_numeric_mask = numeric.isna() & ~values.isin(MISSING_VALUE_STRINGS)
        non_numeric_count = int(non_numeric_mask.sum())
        if non_numeric_count:
            errors.append(
                f"non-numeric values found in target column {target}: count={non_numeric_count}"
            )
        finite_numeric = numeric[~numeric.isna() & ~infinite_mask]
        if not finite_numeric.empty:
            target_statistics[target] = {
                "min": float(finite_numeric.min()),
                "max": float(finite_numeric.max()),
                "mean": float(finite_numeric.mean()),
                "std": float(finite_numeric.std(ddof=0)),
            }
    if infinite_value_count:
        errors.append(f"infinite target values found: count={infinite_value_count}")

    return SubmissionValidationReport(
        not errors,
        display_path(submission_path),
        display_path(sample_path),
        row_count,
        sample_row_count,
        id_column,
        target_columns,
        duplicate_id_count,
        missing_value_count,
        infinite_value_count,
        submission_sha256,
        target_statistics,
        errors,
    )


def regenerate_experiment_summary() -> None:
    summary_path = ROOT / "experiment_summary.md"
    existing = summary_path.read_text() if summary_path.exists() else ""
    summary_path.write_text(update_summary(existing, render_auto_block(collect_records())))


def record_validation(
    experiment: str,
    report: SubmissionValidationReport,
    *,
    regenerate_summary: bool = True,
) -> None:
    if Path(experiment).name != experiment:
        raise ValueError(f"invalid experiment name: {experiment!r}")
    experiments_dir = configured_project_path("paths.experiments_dir", "experiments", root=ROOT)
    experiment_dir = experiments_dir / experiment
    if not experiment_dir.is_dir():
        raise ValueError(f"experiment does not exist: {experiment_dir}")
    metrics_path = experiment_dir / "metrics.json"
    try:
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid metrics JSON: {metrics_path.relative_to(ROOT)}: {exc}") from exc
    if not isinstance(metrics, dict):
        raise ValueError(f"{metrics_path.relative_to(ROOT)} must contain a JSON object")

    evidence = metrics.setdefault("evidence", {})
    if not isinstance(evidence, dict):
        raise ValueError("metrics.json evidence must be a JSON object")
    validation = evidence.setdefault("submission_validation", {})
    if not isinstance(validation, dict):
        raise ValueError("metrics.json evidence.submission_validation must be a JSON object")
    validation.update(report.evidence())
    artifacts = evidence.setdefault("artifacts", {})
    if not isinstance(artifacts, dict):
        raise ValueError("metrics.json evidence.artifacts must be a JSON object")
    artifacts["submission_sha"] = report.submission_sha256
    metrics["updated_at"] = datetime.now(UTC).isoformat()
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    if regenerate_summary:
        regenerate_experiment_summary()


def build_report(args: argparse.Namespace) -> SubmissionValidationReport:
    config = load_project_config()
    sample_value = getattr(args, "sample", None) or get_nested(
        config, "submission.sample_file"
    )
    if is_todo(sample_value):
        raise ValueError("submission.sample_file is TODO in project.yml")
    id_value = get_nested(config, "submission.id_column")
    id_column = None if is_todo(id_value) else str(id_value)
    target_value = get_nested(config, "submission.target_columns")
    target_columns = (
        [str(value) for value in target_value] if isinstance(target_value, list) else []
    )
    allow_extra_columns = get_nested(config, "submission.allow_extra_columns")
    if not isinstance(allow_extra_columns, bool):
        raise ValueError("submission.allow_extra_columns must be true or false in project.yml")

    competition_slug = get_nested(config, "competition.slug")
    if competition_slug == BIOHUB_COMPETITION_SLUG:
        if id_column != "id":
            raise ValueError("Biohub submission.id_column must be id in project.yml")
        if target_columns != BIOHUB_TARGET_COLUMNS:
            raise ValueError(
                "Biohub submission.target_columns must be "
                f"{BIOHUB_TARGET_COLUMNS} in project.yml"
            )
        if allow_extra_columns:
            raise ValueError(
                "Biohub submission.allow_extra_columns must be false in project.yml"
            )

        expected_shapes = None
        test_dir_override = getattr(args, "test_dir", None)
        if test_dir_override:
            expected_shapes = load_biohub_test_shapes(resolve_path(test_dir_override))
        else:
            configured_test_dir = get_nested(config, "data.test_dir")
            if not is_todo(configured_test_dir):
                test_dir_path = resolve_path(str(configured_test_dir))
                if test_dir_path.is_dir():
                    expected_shapes = load_biohub_test_shapes(test_dir_path)

        return validate_biohub_tracking_submission(
            resolve_path(args.submission),
            resolve_path(str(sample_value)),
            expected_shapes=expected_shapes,
        )

    return validate_submission_files(
        resolve_path(args.submission),
        resolve_path(str(sample_value)),
        id_column=id_column,
        target_columns=target_columns,
        allow_extra_columns=allow_extra_columns,
    )


def print_report(report: SubmissionValidationReport, output_format: str) -> None:
    if output_format == "json":
        print(json.dumps(asdict(report), ensure_ascii=False, sort_keys=True))
        return
    if report.passed:
        print(f"submission validation passed: {report.submission}")
        print(
            f"rows={report.row_count}, duplicate_ids={report.duplicate_id_count}, "
            f"missing={report.missing_value_count}, infinite={report.infinite_value_count}"
        )
    else:
        for error in report.errors:
            print(f"ERROR: {error}")


def main() -> int:
    args = parse_args()
    try:
        report = build_report(args)
    except (OSError, ValueError) as exc:
        report = SubmissionValidationReport(
            False,
            args.submission,
            args.sample or "",
            None,
            None,
            None,
            [],
            0,
            0,
            0,
            None,
            {},
            [str(exc)],
        )

    if args.experiment.strip():
        try:
            record_validation(args.experiment.strip(), report)
        except (OSError, ValueError) as exc:
            report.errors.append(f"could not record validation evidence: {exc}")
            report.passed = False

    print_report(report, args.format)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
