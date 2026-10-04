from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import record_experiment  # noqa: E402
import record_submission  # noqa: E402
from record_experiment import (  # noqa: E402
    apply_evidence_assignments,
    parse_evidence_value,
    record_scores,
)
from update_experiment_summary import display_cv  # noqa: E402


def test_parse_evidence_value_preserves_types() -> None:
    assert parse_evidence_value("true") is True
    assert parse_evidence_value("3600") == 3600
    assert parse_evidence_value('["owner/source"]') == ["owner/source"]
    assert parse_evidence_value("owner/slug") == "owner/slug"


def test_apply_evidence_assignments_updates_nested_schema() -> None:
    metrics = {
        "evidence": {
            "kaggle": {"kernel_id": None, "kernel_version": None},
            "submission_validation": {"passed": None},
        }
    }

    apply_evidence_assignments(
        metrics,
        [
            "kaggle.kernel_id=owner/slug",
            "kaggle.kernel_version=2",
            "submission_validation.passed=true",
        ],
    )

    assert metrics["evidence"]["kaggle"] == {
        "kernel_id": "owner/slug",
        "kernel_version": 2,
    }
    assert metrics["evidence"]["submission_validation"]["passed"] is True


def test_apply_evidence_assignments_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="expected KEY=VALUE"):
        apply_evidence_assignments({}, ["kaggle.kernel_id"])

    with pytest.raises(ValueError, match="is not an object"):
        apply_evidence_assignments(
            {"evidence": {"kaggle": "invalid"}},
            ["kaggle.kernel_id=owner/slug"],
        )


def test_display_cv_extracts_overall_score_from_structured_result() -> None:
    assert display_cv(0.42) == "0.42"
    assert (
        display_cv(
            {
                "overall": {"score": 0.12490551617521294},
                "by_embryo": {"44b6": {"score": 0.6247928949700339}},
            }
        )
        == "0.12490551617521294"
    )


def score_args(**overrides) -> Namespace:
    return Namespace(
        **{
            "experiment": "exp123_test",
            "submission_ref": "",
            "submission_status": "",
            "submitted_at": "",
            "cv": "",
            "public_lb": "",
            "private_lb": "",
            **overrides,
        }
    )


def test_submission_score_update_preserves_experiment_and_other_refs() -> None:
    metrics = {"status": "discarded", "cv": 0.4, "public_lb": 0.938, "private_lb": None}
    original = metrics.copy()
    record_scores(metrics, score_args(submission_ref="111", public_lb="0.942"))
    record_scores(metrics, score_args(submission_ref="222", public_lb="0.943"))
    record_scores(
        metrics, score_args(submission_ref="333", submission_status="runtime_limit_exceeded")
    )
    record_scores(metrics, score_args(submission_ref="222", private_lb="0.919"))
    assert {key: metrics[key] for key in original} == original
    assert metrics["submissions"]["111"] == {
        "cv": None,
        "public_lb": 0.942,
        "private_lb": None,
    }
    assert metrics["submissions"]["222"] == {
        "cv": None,
        "public_lb": 0.943,
        "private_lb": 0.919,
    }
    assert metrics["submissions"]["333"]["public_lb"] is None


def test_empty_ref_keeps_experiment_score_workflow() -> None:
    metrics = {"public_lb": 0.1}
    record_scores(metrics, score_args(public_lb="0.2"))
    assert metrics == {"public_lb": 0.2}


def test_failed_submission_cannot_inherit_or_receive_a_score() -> None:
    metrics = {"public_lb": 0.95}
    record_scores(
        metrics,
        score_args(
            submission_ref="333",
            submission_status="notebook_unhandled_error",
            submitted_at="2026-09-28T02:12:13.880000Z",
        ),
    )
    assert metrics["submissions"]["333"]["public_lb"] is None
    with pytest.raises(ValueError, match="failed submissions"):
        record_scores(metrics, score_args(submission_ref="333", public_lb="0.95"))


def test_submission_metadata_requires_ref_and_timezone() -> None:
    with pytest.raises(ValueError, match="require --submission-ref"):
        record_scores({}, score_args(submission_status="complete"))
    with pytest.raises(ValueError, match="timezone"):
        record_scores({}, score_args(submission_ref="111", submitted_at="2026-09-28T01:00:00"))


def test_record_exp_cli_updates_only_requested_submission(tmp_path, monkeypatch) -> None:
    experiment = tmp_path / "exp123_test"
    experiment.mkdir()
    path = experiment / "metrics.json"
    original = {"experiment": experiment.name, "status": "discarded", "public_lb": 0.938}
    path.write_text(json.dumps(original))
    monkeypatch.setattr(record_experiment, "EXPERIMENTS_DIR", tmp_path)
    monkeypatch.setattr(record_experiment, "ROOT", tmp_path)
    base = [
        "record_experiment.py",
        "--experiment",
        experiment.name,
        "--submission-ref",
        "111",
        "--no-summary",
    ]
    monkeypatch.setattr(
        sys, "argv", base + ["--public-lb", "0.942", "--submission-status", "complete"]
    )
    record_experiment.main()
    monkeypatch.setattr(sys, "argv", base + ["--private-lb", "0.919"])
    record_experiment.main()
    metrics = json.loads(path.read_text())
    assert {key: metrics[key] for key in original} == original
    assert metrics["submissions"]["111"] == {
        "cv": None,
        "public_lb": 0.942,
        "private_lb": 0.919,
        "submission_status": "complete",
    }
    before = path.read_bytes()
    monkeypatch.setattr(sys, "argv", base + ["--status", "completed"])
    with pytest.raises(SystemExit, match="record it separately"):
        record_experiment.main()
    assert path.read_bytes() == before


def test_first_private_update_migrates_same_ref_history_without_erasing_scores(
    tmp_path, monkeypatch
) -> None:
    history = tmp_path / "SUBMISSIONS.md"
    history.write_text(
        record_submission.render_table_row(
            [
                "v001",
                "2026-09-28",
                "exp123_test",
                "submission.csv",
                "-",
                "-",
                "-",
                "0.45",
                "0.942",
                "-",
                "111",
                "submission_status=complete",
            ]
        )
        + "\n"
    )
    monkeypatch.setattr(record_submission, "SUBMISSIONS_PATH", history)
    # The existing ref snapshot takes precedence even if the experiment summary differs.
    metrics = {"cv": 0.4, "public_lb": 0.938, "private_lb": None}
    record_scores(metrics, score_args(submission_ref="111", private_lb="0.919"))
    assert metrics["submissions"]["111"] == {
        "cv": 0.45,
        "public_lb": 0.942,
        "private_lb": 0.919,
    }
    assert metrics["public_lb"] == 0.938
    record_scores(metrics, score_args(submission_ref="111", private_lb="0.920"))
    assert metrics["submissions"]["111"]["public_lb"] == 0.942
    assert metrics["submissions"]["111"]["private_lb"] == 0.920
    experiment = tmp_path / "exp123_test"
    experiment.mkdir()
    (experiment / "metrics.json").write_text(json.dumps(metrics))
    monkeypatch.setattr(record_submission, "EXPERIMENTS_DIR", tmp_path)
    monkeypatch.setattr(
        record_submission,
        "parse_args",
        lambda: Namespace(
            experiment="exp123_test",
            submission_ref="111",
            file=str(tmp_path / "absent.csv"),
            notes=None,
            version=None,
            allow_missing_file=True,
        ),
    )
    record_submission.main()
    row = record_submission.find_submission_row(history.read_text().splitlines(), "111")[1]
    assert row[7:10] == ["0.45", "0.942", "0.92"]


def test_first_private_update_can_migrate_verified_unique_legacy_ref(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(record_submission, "SUBMISSIONS_PATH", tmp_path / "absent.md")
    metrics = {"cv": 0.45, "public_lb": 0.942, "evidence": {"submission_ref": 111}}
    record_scores(metrics, score_args(submission_ref="111", private_lb="0.919"))
    assert metrics["submissions"]["111"] == {
        "cv": 0.45,
        "public_lb": 0.942,
        "private_lb": 0.919,
    }


@pytest.mark.parametrize("known_refs", [[], [111], [111, 222]])
def test_new_ref_does_not_inherit_other_submission_scores(
    tmp_path, monkeypatch, known_refs
) -> None:
    monkeypatch.setattr(record_submission, "SUBMISSIONS_PATH", tmp_path / "absent.md")
    metrics = {
        "cv": 0.45,
        "public_lb": 0.942,
        "evidence": [{"submission_ref": ref} for ref in known_refs],
    }
    record_scores(metrics, score_args(submission_ref="333", private_lb="0.919"))
    assert metrics["submissions"]["333"] == {
        "cv": None,
        "public_lb": None,
        "private_lb": 0.919,
    }


def test_multiple_legacy_refs_cannot_supply_individual_scores(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(record_submission, "SUBMISSIONS_PATH", tmp_path / "absent.md")
    metrics = {"public_lb": 0.942, "evidence": [{"submission_ref": 111}, {"submission_ref": 222}]}
    record_scores(metrics, score_args(submission_ref="111", private_lb="0.919"))
    assert metrics["submissions"]["111"]["public_lb"] is None
