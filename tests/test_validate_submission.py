from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import validate_submission as VALIDATOR  # noqa: E402

BIOHUB_HEADER = ",".join(VALIDATOR.BIOHUB_SUBMISSION_COLUMNS)


def write_csv(path: Path, rows: list[str]) -> None:
    path.write_text("\n".join(rows) + "\n")


def validate(tmp_path: Path, submission_rows: list[str]):
    sample = tmp_path / "sample_submission.csv"
    submission = tmp_path / "submission.csv"
    write_csv(sample, ["id,prediction", "a,0", "b,0", "c,0"])
    write_csv(submission, submission_rows)
    return VALIDATOR.validate_submission_files(
        submission,
        sample,
        id_column="id",
        target_columns=["prediction"],
        allow_extra_columns=False,
    )


def validate_biohub(
    tmp_path: Path,
    submission_rows: list[str],
    *,
    sample_rows: list[str] | None = None,
    expected_shapes: dict[str, tuple[int, int, int, int] | None] | None = None,
):
    sample = tmp_path / "sample_submission.csv"
    submission = tmp_path / "submission.csv"
    write_csv(sample, sample_rows or [BIOHUB_HEADER])
    write_csv(submission, submission_rows)
    return VALIDATOR.validate_biohub_tracking_submission(
        submission,
        sample,
        expected_shapes=expected_shapes,
    )


def test_valid_submission_checks_ids_and_finite_targets(tmp_path: Path) -> None:
    report = validate(tmp_path, ["id,prediction", "a,1.5", "b,2", "c,-3"])

    assert report.passed is True
    assert report.row_count == 3
    assert report.duplicate_id_count == 0
    assert report.missing_value_count == 0
    assert report.infinite_value_count == 0
    assert report.target_statistics["prediction"]["min"] == -3.0
    assert report.submission_sha256 is not None


def test_submission_rejects_id_order_and_duplicates(tmp_path: Path) -> None:
    report = validate(tmp_path, ["id,prediction", "b,1", "b,2", "c,3"])

    assert report.passed is False
    assert report.duplicate_id_count == 2
    assert any("duplicate IDs" in error for error in report.errors)
    assert any("order/content mismatch" in error for error in report.errors)


def test_submission_rejects_missing_infinite_and_non_numeric_targets(
    tmp_path: Path,
) -> None:
    report = validate(
        tmp_path,
        ["id,prediction", "a,", "b,1e309", "c,not-a-number"],
    )

    assert report.passed is False
    assert report.missing_value_count == 1
    assert report.infinite_value_count == 1
    assert any("non-numeric" in error for error in report.errors)
    assert any("infinite" in error for error in report.errors)


def test_submission_rejects_column_order_and_extra_columns(tmp_path: Path) -> None:
    report = validate(
        tmp_path,
        ["prediction,id,debug", "1,a,x", "2,b,x", "3,c,x"],
    )

    assert report.passed is False
    assert any("extra columns" in error for error in report.errors)
    assert any("column order/content mismatch" in error for error in report.errors)


def test_biohub_accepts_variable_rows_and_float_coordinates(tmp_path: Path) -> None:
    report = validate_biohub(
        tmp_path,
        [
            BIOHUB_HEADER,
            "0,sample_a,node,10,0,1.5,2.25,3.75,-1,-1",
            "1,sample_a,node,11,1,2,3,4,-1,-1",
            "2,sample_a,edge,-1,-1,-1,-1,-1,10,11",
        ],
        sample_rows=[
            BIOHUB_HEADER,
            "0,example,node,1,0,0,0,0,-1,-1",
            "1,example,node,2,1,0,0,0,-1,-1",
            "2,example,node,3,2,0,0,0,-1,-1",
            "3,example,edge,-1,-1,-1,-1,-1,1,2",
            "4,example,edge,-1,-1,-1,-1,-1,2,3",
        ],
    )

    assert report.passed is True
    assert report.row_count == 3
    assert report.sample_row_count == 5
    assert report.target_statistics["z"]["min"] == -1.0


def test_biohub_rejects_bad_sentinels_and_edge_references(tmp_path: Path) -> None:
    report = validate_biohub(
        tmp_path,
        [
            BIOHUB_HEADER,
            "0,sample_a,node,10,0,0,0,0,7,-1",
            "1,sample_a,node,11,2,0,0,0,-1,-1",
            "2,sample_a,edge,-1,-1,-1,-1,-1,10,11",
            "3,sample_a,edge,-1,-1,-1,-1,-1,10,99",
        ],
    )

    assert report.passed is False
    assert any("node rows must use" in error for error in report.errors)
    assert any("missing target nodes" in error for error in report.errors)
    assert any("connect t to t+1" in error for error in report.errors)


def test_biohub_rejects_nonconsecutive_ids_and_duplicate_graph_rows(
    tmp_path: Path,
) -> None:
    report = validate_biohub(
        tmp_path,
        [
            BIOHUB_HEADER,
            "0,sample_a,node,10,0,0,0,0,-1,-1",
            "2,sample_a,node,10,1,0,0,0,-1,-1",
            "3,sample_a,edge,-1,-1,-1,-1,-1,10,10",
            "4,sample_a,edge,-1,-1,-1,-1,-1,10,10",
        ],
    )

    assert report.passed is False
    assert any("id must be consecutive" in error for error in report.errors)
    assert any("duplicate node_id" in error for error in report.errors)
    assert any("duplicate edges" in error for error in report.errors)
    assert any("self-loop" in error for error in report.errors)


def test_biohub_checks_dataset_coverage_and_coordinate_bounds(tmp_path: Path) -> None:
    report = validate_biohub(
        tmp_path,
        [
            BIOHUB_HEADER,
            "0,sample_a,node,10,0,0,0,5,-1,-1",
        ],
        expected_shapes={"sample_a": (2, 3, 4, 5), "sample_b": None},
    )

    assert report.passed is False
    assert any("missing test datasets" in error for error in report.errors)
    assert any("sample_a: x is outside" in error for error in report.errors)


def test_load_biohub_test_shapes_reads_zarr_v3_metadata(tmp_path: Path) -> None:
    test_dir = tmp_path / "test"
    metadata_dir = test_dir / "sample_a.zarr" / "0"
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "zarr.json").write_text(json.dumps({"shape": [2, 3, 4, 5]}))
    (test_dir / "sample_b.zarr").mkdir()

    shapes = VALIDATOR.load_biohub_test_shapes(test_dir)

    assert shapes == {"sample_a": (2, 3, 4, 5), "sample_b": None}


def test_build_report_dispatches_biohub_without_local_sample(
    tmp_path: Path, monkeypatch
) -> None:
    submission = tmp_path / "submission.csv"
    write_csv(
        submission,
        [BIOHUB_HEADER, "0,sample_a,node,10,0,0,0,0,-1,-1"],
    )
    config = {
        "competition": {"slug": VALIDATOR.BIOHUB_COMPETITION_SLUG},
        "data": {"test_dir": str(tmp_path / "missing-test")},
        "submission": {
            "sample_file": str(tmp_path / "missing-sample.csv"),
            "id_column": "id",
            "target_columns": VALIDATOR.BIOHUB_TARGET_COLUMNS,
            "allow_extra_columns": False,
        },
    }
    monkeypatch.setattr(VALIDATOR, "ROOT", tmp_path)
    monkeypatch.setattr(VALIDATOR, "load_project_config", lambda: config)
    args = SimpleNamespace(sample=None, submission=str(submission), test_dir=None)

    report = VALIDATOR.build_report(args)

    assert report.passed is True
    assert report.sample_row_count is None


def test_validation_evidence_is_recorded_in_metrics(tmp_path: Path, monkeypatch) -> None:
    report = validate(tmp_path, ["id,prediction", "a,1", "b,2", "c,3"])
    experiment = tmp_path / "experiments" / "exp999_check"
    experiment.mkdir(parents=True)
    (experiment / "metrics.json").write_text(
        json.dumps(
            {
                "experiment": "exp999_check",
                "evidence": {
                    "artifacts": {},
                    "submission_validation": {"fallback_rows": 7},
                },
            }
        )
    )
    monkeypatch.setattr(VALIDATOR, "ROOT", tmp_path)

    VALIDATOR.record_validation("exp999_check", report, regenerate_summary=False)

    metrics = json.loads((experiment / "metrics.json").read_text())
    validation = metrics["evidence"]["submission_validation"]
    assert validation["passed"] is True
    assert validation["duplicate_id_count"] == 0
    assert validation["infinite_value_count"] == 0
    assert validation["fallback_rows"] == 7
    assert metrics["evidence"]["artifacts"]["submission_sha"] == report.submission_sha256


def test_build_report_rejects_string_allow_extra_columns(monkeypatch) -> None:
    config = {
        "submission": {
            "sample_file": "sample_submission.csv",
            "id_column": "id",
            "target_columns": ["prediction"],
            "allow_extra_columns": "false",
        }
    }
    monkeypatch.setattr(VALIDATOR, "load_project_config", lambda: config)
    args = SimpleNamespace(sample=None, submission="submission.csv")

    with pytest.raises(ValueError, match="allow_extra_columns must be true or false"):
        VALIDATOR.build_report(args)
