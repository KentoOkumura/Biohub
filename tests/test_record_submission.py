import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import record_submission
from scripts.config_utils import ROOT, load_project_config, project_path


def write_history(path: Path, versions: list[str]) -> None:
    rows = [f"| {version} | 2026-01-01 | exp001 | file.csv |" for version in versions]
    path.write_text("# 提出履歴\n\n" + "\n".join(rows) + "\n")


def test_submission_history_path_comes_from_project_config() -> None:
    expected = project_path(load_project_config(), "paths.submissions_file")

    assert expected == ROOT / "SUBMISSIONS.md"
    assert record_submission.SUBMISSIONS_PATH == expected


def test_next_version_uses_maximum_existing_number(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    history = tmp_path / "SUBMISSIONS.md"
    write_history(history, ["v001", "v003"])
    monkeypatch.setattr(record_submission, "SUBMISSIONS_PATH", history)

    assert record_submission.next_version() == "v004"


def test_duplicate_existing_versions_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    history = tmp_path / "SUBMISSIONS.md"
    write_history(history, ["v002", "v002"])
    monkeypatch.setattr(record_submission, "SUBMISSIONS_PATH", history)

    with pytest.raises(ValueError, match="duplicate submission versions: v002"):
        record_submission.existing_versions()


def test_explicit_duplicate_version_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    history = tmp_path / "SUBMISSIONS.md"
    write_history(history, ["v001"])
    monkeypatch.setattr(record_submission, "SUBMISSIONS_PATH", history)

    with pytest.raises(SystemExit, match="already exists"):
        record_submission.validate_new_version("v001")


def test_submission_ref_must_be_one_exact_numeric_ref() -> None:
    assert record_submission.validate_submission_ref(" 12345678 ") == "12345678"

    with pytest.raises(SystemExit, match="exact numeric Kaggle ref"):
        record_submission.validate_submission_ref("123,456")


def test_submission_notes_require_unambiguous_status_and_runtime_keys() -> None:
    notes = "submission_status=COMPLETE; scoring_elapsed_minutes=12"

    assert record_submission.validate_notes(notes) == notes
    assert record_submission.validate_notes(None) is None
    with pytest.raises(SystemExit, match="submission_status"):
        record_submission.validate_notes("scoring completed")
    with pytest.raises(SystemExit, match="submission_status"):
        record_submission.validate_notes("status=COMPLETE")
    with pytest.raises(SystemExit, match="Markdown table cell"):
        record_submission.validate_notes("first line\nsecond line")


def test_new_submission_history_includes_record_source_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    history = tmp_path / "SUBMISSIONS.md"
    monkeypatch.setattr(record_submission, "SUBMISSIONS_PATH", history)

    record_submission.ensure_table()

    content = history.read_text()
    assert "詳細な時系列の正" in content
    assert "CV/LBとNotebook実行時間の正" in content
    assert "`submission_status`" in content
    assert record_submission.TABLE_HEADER in content


def test_current_submission_notes_follow_note_contract() -> None:
    for line in record_submission.SUBMISSIONS_PATH.read_text().splitlines():
        cells = record_submission.parse_table_row(line)
        if cells is not None and cells[11] != "-":
            assert record_submission.validate_notes(cells[11]) == cells[11]


def test_submission_history_has_a_dedicated_ref_column() -> None:
    assert "| submission ref |" in record_submission.TABLE_HEADER
    assert record_submission.TABLE_HEADER.count("|") == record_submission.TABLE_SEPARATOR.count("|")

    lines = record_submission.SUBMISSIONS_PATH.read_text().splitlines()
    table_lines = [line for line in lines if line.startswith("|")]
    expected_columns = record_submission.TABLE_HEADER.count("|")
    assert all(line.count("|") == expected_columns for line in table_lines)
    for row in table_lines[2:]:
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        assert re.fullmatch(r"\d+", cells[-2]), row


def test_submission_scores_are_read_from_experiment_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiments_dir = tmp_path / "experiments"
    experiment_dir = experiments_dir / "exp123_test"
    experiment_dir.mkdir(parents=True)
    (experiment_dir / "metrics.json").write_text(
        json.dumps({"cv": 0.1234, "public_lb": 0.12, "private_lb": None})
    )
    monkeypatch.setattr(record_submission, "EXPERIMENTS_DIR", experiments_dir)

    assert record_submission.experiment_scores("exp123_test", "12345678") == (
        "0.1234",
        "0.12",
        "-",
    )


def test_existing_submission_ref_is_updated_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    history = tmp_path / "SUBMISSIONS.md"
    experiments_dir = tmp_path / "experiments"
    experiment_dir = experiments_dir / "exp123_test"
    experiment_dir.mkdir(parents=True)
    metrics_path = experiment_dir / "metrics.json"
    metrics_path.write_text(json.dumps({"cv": 0.1234, "public_lb": 0.12, "private_lb": None}))
    submission = tmp_path / "submission.csv"
    submission.write_text("id,target\n1,2.0\n")
    monkeypatch.setattr(record_submission, "SUBMISSIONS_PATH", history)
    monkeypatch.setattr(record_submission, "EXPERIMENTS_DIR", experiments_dir)

    args = SimpleNamespace(
        experiment="exp123_test",
        file=str(submission),
        submission_ref="12345678",
        version=None,
        notes="public score",
        allow_missing_file=False,
    )
    monkeypatch.setattr(record_submission, "parse_args", lambda: args)
    record_submission.main()

    original_row = next(line for line in history.read_text().splitlines() if line.startswith("| v"))
    metrics_path.write_text(json.dumps({"cv": 0.1234, "public_lb": 0.12, "private_lb": 0.11}))
    args.notes = None
    record_submission.main()

    rows = [line for line in history.read_text().splitlines() if line.startswith("| v")]
    assert len(rows) == 1
    assert rows[0].startswith("| v001 |")
    assert "| 0.1234 | 0.12 | 0.11 | 12345678 | public score |" in rows[0]
    assert rows[0].split(" | ")[1:7] == original_row.split(" | ")[1:7]


def test_missing_local_code_submission_can_be_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    history = tmp_path / "SUBMISSIONS.md"
    experiments_dir = tmp_path / "experiments"
    experiment_dir = experiments_dir / "exp123_test"
    experiment_dir.mkdir(parents=True)
    (experiment_dir / "metrics.json").write_text(
        json.dumps({"cv": 0.1234, "public_lb": 0.12, "private_lb": None})
    )
    monkeypatch.setattr(record_submission, "SUBMISSIONS_PATH", history)
    monkeypatch.setattr(record_submission, "EXPERIMENTS_DIR", experiments_dir)
    monkeypatch.setattr(
        record_submission,
        "parse_args",
        lambda: SimpleNamespace(
            experiment="exp123_test",
            file="submission.csv",
            submission_ref="12345678",
            version=None,
            notes=None,
            allow_missing_file=True,
        ),
    )

    record_submission.main()

    row = next(line for line in history.read_text().splitlines() if line.startswith("| v"))
    assert "| submission.csv | - | - | - |" in row


def test_legacy_grouped_submission_ref_must_be_split_before_update() -> None:
    lines = [
        "| v001 | 2026-01-01 | exp123_test | submission.csv | 1 | id,target | "
        "abc | 0.1 | 0.1 | - | 123,456 | legacy |"
    ]

    with pytest.raises(SystemExit, match="legacy grouped row"):
        record_submission.find_submission_row(lines, "123")


def test_multiple_refs_keep_distinct_scores_and_failed_ref_null(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    history = tmp_path / "SUBMISSIONS.md"
    experiments = tmp_path / "experiments"
    experiment = experiments / "exp123_test"
    experiment.mkdir(parents=True)
    metrics = {
        "public_lb": 0.938,
        "submissions": {
            "111": {"cv": None, "public_lb": 0.942, "private_lb": None},
            "222": {"cv": None, "public_lb": 0.943, "private_lb": None},
            "333": {
                "cv": None,
                "public_lb": None,
                "private_lb": None,
                "submission_status": "runtime_limit_exceeded",
                "submitted_at": "2026-09-28T02:12:13.880000+00:00",
            },
        },
    }
    metrics_path = experiment / "metrics.json"
    metrics_path.write_text(json.dumps(metrics))
    monkeypatch.setattr(record_submission, "SUBMISSIONS_PATH", history)
    monkeypatch.setattr(record_submission, "EXPERIMENTS_DIR", experiments)
    args = SimpleNamespace(
        experiment=experiment.name,
        file=str(tmp_path / "absent.csv"),
        submission_ref="111",
        version=None,
        notes=None,
        allow_missing_file=True,
    )
    monkeypatch.setattr(record_submission, "parse_args", lambda: args)
    for _ in range(2):
        for ref in metrics["submissions"]:
            args.submission_ref = ref
            record_submission.main()
    lines = history.read_text().splitlines()
    for ref, expected in (("111", "0.942"), ("222", "0.943"), ("333", "-")):
        row = record_submission.find_submission_row(lines, ref)[1]
        assert row[8] == expected
    assert sum(line.startswith("| v") for line in lines) == 3
    assert record_submission.find_submission_row(lines, "333")[1][1] == "2026-09-28"

    metrics["submissions"]["222"]["private_lb"] = 0.919
    metrics_path.write_text(json.dumps(metrics))
    args.submission_ref = "222"
    record_submission.main()
    lines = history.read_text().splitlines()
    assert record_submission.find_submission_row(lines, "222")[1][7:10] == ["-", "0.943", "0.919"]
    assert record_submission.find_submission_row(lines, "111")[1][9] == "-"
    assert record_submission.find_submission_row(lines, "333")[1][8:10] == ["-", "-"]


def test_multiple_legacy_refs_reject_experiment_wide_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(record_submission, "SUBMISSIONS_PATH", tmp_path / "absent.md")
    metrics = {
        "public_lb": 0.938,
        "evidence": {
            "a": {"submission_ref": 111},
            "b": {"competition_submission_ref": 222},
        },
    }
    with pytest.raises(SystemExit, match="multiple or different"):
        record_submission.submission_record("exp123_test", "111", metrics)


def test_missing_canonical_ref_never_falls_back_to_experiment_score() -> None:
    with pytest.raises(SystemExit, match="no score record"):
        record_submission.submission_record(
            "exp123_test", "222", {"public_lb": 0.938, "submissions": {"111": {}}}
        )


@pytest.mark.parametrize(
    "status", ["runtime_limit_exceeded", "SubmissionStatus.ERROR", " CANCELLED "]
)
def test_failed_legacy_ref_rejects_unrelated_experiment_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str
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
                "-",
                "-",
                "-",
                "111",
                f"submission_status={status}",
            ]
        )
        + "\n"
    )
    monkeypatch.setattr(record_submission, "SUBMISSIONS_PATH", history)
    with pytest.raises(SystemExit, match="failed submission"):
        record_submission.submission_record("exp123_test", "111", {"public_lb": 0.938})


def test_current_multiple_submission_scores_match_history() -> None:
    for line in record_submission.SUBMISSIONS_PATH.read_text().splitlines():
        cells = record_submission.parse_table_row(line)
        if cells is not None:
            metrics = record_submission.experiment_metrics(cells[2])
            if "submissions" in metrics:
                assert record_submission.experiment_scores(cells[2], cells[10]) == tuple(
                    cells[7:10]
                )
