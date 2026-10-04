from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / ".agents"
    / "skills"
    / "kaggle-submit-monitor"
    / "scripts"
    / "monitor_submission.py"
)
SPEC = importlib.util.spec_from_file_location("kaggle_submit_monitor", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MONITOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MONITOR)


def test_explicit_monitor_log_path_is_preserved(tmp_path: Path) -> None:
    path = tmp_path / "monitor.log"
    assert MONITOR.resolve_log_path("exp001_baseline", str(path), None) == path


def test_missing_experiment_uses_temporary_directory(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(MONITOR, "REPO_ROOT", tmp_path)
    path = MONITOR.resolve_log_path("exp001_baseline", None, None)
    assert path == (
        Path(tempfile.gettempdir()) / "kaggle-submission-monitor" / "submission_exp001_baseline.log"
    )


def test_unknown_experiment_uses_temporary_directory() -> None:
    path = MONITOR.resolve_log_path("unknown submission", None, None)
    assert path == (
        Path(tempfile.gettempdir())
        / "kaggle-submission-monitor"
        / "submission_unknown-submission.log"
    )


def test_monitor_uses_configured_experiments_directory(tmp_path: Path, monkeypatch) -> None:
    experiment = tmp_path / "runs" / "exp001_baseline"
    experiment.mkdir(parents=True)
    (tmp_path / "project.yml").write_text("paths:\n  experiments_dir: runs\n")
    monkeypatch.setattr(MONITOR, "REPO_ROOT", tmp_path)

    assert MONITOR.resolve_log_path("exp001_baseline", None, None) == (
        experiment / "artifacts" / "submission-monitor.log"
    )


def test_submission_query_uses_kaggle_from_current_interpreter(monkeypatch) -> None:
    observed: list[str] = []

    def fake_run(command, **kwargs):
        observed.extend(command)
        return subprocess.CompletedProcess(command, 0, stdout="ref,status\n1,complete\n")

    monkeypatch.setattr(MONITOR.subprocess, "run", fake_run)

    rows = MONITOR.run_submissions("example-competition")

    assert observed[:3] == [sys.executable, "-m", "kaggle"]
    assert rows == [{"ref": "1", "status": "complete"}]


def test_monitor_reads_default_competition_from_project_yml(tmp_path: Path) -> None:
    project_path = tmp_path / "project.yml"
    project_path.write_text("competition:\n  slug: configured-competition\n")

    assert MONITOR.resolve_competition(None, project_path) == "configured-competition"


def test_explicit_competition_overrides_project_yml(tmp_path: Path) -> None:
    project_path = tmp_path / "project.yml"
    project_path.write_text("competition:\n  slug: configured-competition\n")

    assert MONITOR.resolve_competition("other-competition", project_path) == ("other-competition")


def test_monitor_rejects_unconfigured_project_competition(tmp_path: Path) -> None:
    project_path = tmp_path / "project.yml"
    project_path.write_text("competition:\n  slug: TODO\n")

    try:
        MONITOR.resolve_competition(None, project_path)
    except ValueError as exc:
        assert "competition.slug" in str(exc)
    else:
        raise AssertionError("unconfigured competition slug must be rejected")


def test_submission_summary_distinguishes_scoring_elapsed_from_runtime() -> None:
    complete, summary = MONITOR.row_summary(
        "exp001_baseline",
        {"status": "complete", "publicScore": "1.23"},
        time.time() - 120,
    )

    assert complete is True
    assert "scoring-elapsed: 2 min" in summary
    assert "submission-status: complete" in summary
    assert "run-time:" not in summary


def test_monitor_selects_only_the_requested_submission_ref() -> None:
    rows = [
        {"ref": "newer", "status": "complete", "publicScore": "1.0"},
        {"ref": "target", "status": "pending", "publicScore": ""},
    ]

    assert MONITOR.select_submission(rows, "target") == rows[1]
    assert MONITOR.select_submission(rows, "missing") is None


def test_monitor_rejects_ambiguous_submission_ref() -> None:
    rows = [
        {"ref": "target", "status": "pending"},
        {"ref": "target", "status": "complete"},
    ]

    try:
        MONITOR.select_submission(rows, "target")
    except ValueError as exc:
        assert "ambiguous" in str(exc)
    else:
        raise AssertionError("ambiguous submission refs must be rejected")


@pytest.mark.parametrize(
    "status,error,expected_code",
    [
        ("SubmissionStatus.ERROR", "Notebook exceeded allowed runtime", 3),
        ("runtime_limit_exceeded", "", 3),
        ("notebook_unhandled_error", "", 3),
        ("failed", "", 3),
        ("cancelled", "", 3),
        ("canceled", "", 3),
        ("", "Notebook failed", 3),
        ("complete", "Notebook failed", 3),
        ("SubmissionStatus.COMPLETE", "", 0),
    ],
)
def test_monitor_stops_on_terminal_result_after_pending(
    tmp_path, monkeypatch, status, error, expected_code
) -> None:
    log_path = tmp_path / "monitor.log"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "exp123_test",
            "--competition",
            "test-competition",
            "--submission-ref",
            "111",
            "--log-file",
            str(log_path),
        ],
    )
    responses = iter(
        [
            [{"ref": "111", "status": "pending"}],
            [
                {
                    "ref": "111",
                    "status": status,
                    "errorDescription": error,
                    "publicScore": "0.9" if expected_code == 0 else "",
                }
            ],
        ]
    )
    monkeypatch.setattr(MONITOR, "run_submissions", lambda competition: next(responses))
    sleeps = []
    monkeypatch.setattr(MONITOR.time, "sleep", sleeps.append)

    assert MONITOR.main() == expected_code
    assert sleeps == [300]
    log = log_path.read_text()
    assert "submission-status: pending" in log
    assert "timeout" not in log
    if error:
        assert f"error: {error}" in log


def test_monitor_does_not_stop_for_another_refs_failure(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "exp123_test",
            "--competition",
            "test-competition",
            "--submission-ref",
            "111",
            "--log-file",
            str(tmp_path / "monitor.log"),
        ],
    )
    responses = iter(
        [
            [{"ref": "222", "status": "error"}],
            [{"ref": "111", "status": "complete", "publicScore": "0.9"}],
        ]
    )
    monkeypatch.setattr(MONITOR, "run_submissions", lambda competition: next(responses))
    sleeps = []
    monkeypatch.setattr(MONITOR.time, "sleep", sleeps.append)

    assert MONITOR.main() == 0
    assert sleeps == [300]
