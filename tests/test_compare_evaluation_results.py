import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts import compare_evaluation_results as comparison

BASE_ROWS = [
    ("u1", "a", 0.5, 1.0),
    ("u2", "a", 0.6, 2.0),
    ("u3", "b", 0.8, 3.0),
]


def make_manifest(tmp_path: Path, name: str, rows: list | None = None) -> Path:
    directory = tmp_path / name
    directory.mkdir()
    table = directory / "evaluation.csv"
    with table.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["unit", "group", "score", "loss"])
        writer.writerows(BASE_ROWS if rows is None else rows)
    manifest = {
        "schema_version": 1,
        "table": {"path": table.name, "sha256": hashlib.sha256(table.read_bytes()).hexdigest()},
        "unit_column": "unit",
        "group_column": "group",
        "metrics": {"score": "max", "loss": "min"},
        "evaluation_kind": "conditional",
        "expected_unit_count": 3,
        "fixed_conditions": {key: f"fixed-{key}-v1" for key in comparison.REQUIRED_CONDITIONS},
    }
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def edit_manifest(path: Path, **changes) -> None:
    content = json.loads(path.read_text())
    content.update(changes)
    path.write_text(json.dumps(content))


def rewrite_table(path: Path, content: str) -> None:
    manifest = json.loads(path.read_text())
    table = path.parent / manifest["table"]["path"]
    table.write_text(content)
    manifest["table"]["sha256"] = hashlib.sha256(table.read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest))


def test_pairs_by_id_and_reports_metric_direction_and_groups(tmp_path: Path) -> None:
    baseline = make_manifest(tmp_path, "baseline")
    candidate = make_manifest(
        tmp_path,
        "candidate",
        [("u3", "b", 0.7, 4.0), ("u1", "a", 0.6, 0.5), ("u2", "a", 0.6, 2.0)],
    )
    report, pairs = comparison.compare_evaluations(baseline, candidate)
    assert report["comparison"] == "comparable"
    assert report["reasons"] == []
    assert "status" not in report
    assert report["evaluation_kind"] == "conditional"
    assert len(pairs) == 6
    loss = report["diagnostics"]["loss"]
    assert loss["overall"]["improved"] == 1
    assert loss["overall"]["worsened"] == 1
    assert loss["overall"]["unchanged"] == 1
    assert loss["overall"]["unit_mean_delta"] == pytest.approx(1 / 6)
    assert loss["overall"]["unit_mean_oriented_delta"] == pytest.approx(-1 / 6)
    assert loss["by_group"]["a"]["unit_count"] == 2
    assert loss["by_group"]["b"]["worsened"] == 1
    improved_loss = next(row for row in pairs if row["unit_id"] == "u1" and row["metric"] == "loss")
    assert improved_loss["delta"] == -0.5
    assert improved_loss["oriented_delta"] == 0.5
    assert improved_loss["change"] == "improved"


@pytest.mark.parametrize("field", comparison.REQUIRED_CONDITIONS)
@pytest.mark.parametrize("missing", [True, False])
def test_missing_or_empty_required_condition_is_not_a_match(
    tmp_path: Path, field: str, missing: bool
) -> None:
    paths = [make_manifest(tmp_path, name) for name in ("baseline", "candidate")]
    for path in paths:
        conditions = json.loads(path.read_text())["fixed_conditions"]
        if missing:
            del conditions[field]
        else:
            conditions[field] = " "
        edit_manifest(path, fixed_conditions=conditions)
    report, pairs = comparison.compare_evaluations(*paths)
    assert report["comparison"] == "not_comparable"
    assert field in " ".join(report["reasons"])
    assert "diagnostics" not in report
    assert pairs == []


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"evaluation_kind": "cv"}, "evaluation_kind"),
        ({"metrics": {"score": "min", "loss": "min"}}, "metrics"),
        ({"group_column": None}, "group_column"),
        ({"expected_unit_count": 4}, "unit count mismatch"),
        ({"expected_unit_count": True}, "positive integer"),
        ({"metrics": {"score": ["max"]}}, "direction"),
        ({"schema_version": True}, "schema_version"),
    ],
)
def test_incompatible_or_invalid_manifest_has_no_differences(
    tmp_path: Path, changes: dict, reason: str
) -> None:
    baseline = make_manifest(tmp_path, "baseline")
    candidate = make_manifest(tmp_path, "candidate")
    edit_manifest(candidate, **changes)
    report, pairs = comparison.compare_evaluations(baseline, candidate)
    assert report["comparison"] == "not_comparable"
    assert reason in " ".join(report["reasons"])
    assert pairs == []


def test_extra_or_changed_fixed_condition_prevents_comparison(tmp_path: Path) -> None:
    baseline = make_manifest(tmp_path, "baseline")
    candidate = make_manifest(tmp_path, "candidate")
    conditions = json.loads(candidate.read_text())["fixed_conditions"]
    conditions.update(labels="other-labels", threshold="same-threshold")
    edit_manifest(candidate, fixed_conditions=conditions)
    report, pairs = comparison.compare_evaluations(baseline, candidate)
    assert report["reasons"] == [
        "fixed condition mismatch: labels",
        "fixed condition mismatch: threshold",
    ]
    assert pairs == []


@pytest.mark.parametrize("value", ["TODO", "unknown", "N/A", "not_applicable", "TODO: fill in SHA"])
def test_matching_placeholders_are_not_valid_conditions(tmp_path: Path, value: str) -> None:
    paths = [make_manifest(tmp_path, name) for name in ("baseline", "candidate")]
    for path in paths:
        conditions = json.loads(path.read_text())["fixed_conditions"]
        conditions["fixed_inputs"] = value
        edit_manifest(path, fixed_conditions=conditions)
    report, pairs = comparison.compare_evaluations(*paths)
    assert report["comparison"] == "not_comparable"
    assert "fixed_inputs" in " ".join(report["reasons"])
    assert pairs == []


def test_explained_non_applicability_is_permitted(tmp_path: Path) -> None:
    paths = [make_manifest(tmp_path, name) for name in ("baseline", "candidate")]
    for path in paths:
        conditions = json.loads(path.read_text())["fixed_conditions"]
        conditions["split"] = "N/A: unlabelled diagnostic with no train/evaluation split"
        edit_manifest(path, fixed_conditions=conditions)
    report, _ = comparison.compare_evaluations(*paths)
    assert report["comparison"] == "comparable"


@pytest.mark.parametrize(
    ("rows", "reason"),
    [
        (BASE_ROWS[:2], "unit count mismatch"),
        (BASE_ROWS[:2] + [BASE_ROWS[0]], "duplicate unit ID"),
        ([(" ", "a", 0.5, 1.0), *BASE_ROWS[1:]], "unit ID"),
        ([("u1", "", 0.5, 1.0), *BASE_ROWS[1:]], "group"),
        ([("u1", "b", 0.5, 1.0), *BASE_ROWS[1:]], "group mismatch"),
        ([("u4", "a", 0.5, 1.0), *BASE_ROWS[1:]], "units missing from candidate"),
        ([], "unit count mismatch"),
    ],
)
def test_unit_coverage_identity_and_group_checks(tmp_path: Path, rows: list, reason: str) -> None:
    baseline = make_manifest(tmp_path, "baseline")
    candidate = make_manifest(tmp_path, "candidate", rows)
    report, pairs = comparison.compare_evaluations(baseline, candidate)
    assert report["comparison"] == "not_comparable"
    assert reason in " ".join(report["reasons"])
    assert pairs == []


def test_matching_partial_tables_do_not_pass_expected_coverage(tmp_path: Path) -> None:
    baseline = make_manifest(tmp_path, "baseline", BASE_ROWS[:2])
    candidate = make_manifest(tmp_path, "candidate", BASE_ROWS[:2])
    report, pairs = comparison.compare_evaluations(baseline, candidate)
    assert len(report["reasons"]) == 2
    assert all("expected 3, got 2" in reason for reason in report["reasons"])
    assert pairs == []


@pytest.mark.parametrize("bad_value", ["NaN", "inf", "-inf", "1e309", "", "null"])
def test_nonfinite_and_missing_metrics_are_rejected(tmp_path: Path, bad_value: str) -> None:
    baseline = make_manifest(tmp_path, "baseline")
    candidate = make_manifest(tmp_path, "candidate", [("u1", "a", bad_value, 1), *BASE_ROWS[1:]])
    report, pairs = comparison.compare_evaluations(baseline, candidate)
    assert "finite number" in " ".join(report["reasons"])
    assert pairs == []


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        ("unit,group,score,score,loss\n", "duplicate column headers"),
        ("unit,group,score,\n", "nonempty column headers"),
        ("unit,group,loss\n", "missing columns"),
        ("unit,group,score,loss\nu1,a,0.5\n", "number of fields"),
    ],
)
def test_malformed_csv_is_rejected(tmp_path: Path, content: str, reason: str) -> None:
    baseline = make_manifest(tmp_path, "baseline")
    candidate = make_manifest(tmp_path, "candidate")
    rewrite_table(candidate, content)
    report, pairs = comparison.compare_evaluations(baseline, candidate)
    assert reason in " ".join(report["reasons"])
    assert pairs == []


def test_actual_table_hash_is_checked(tmp_path: Path) -> None:
    baseline = make_manifest(tmp_path, "baseline")
    candidate = make_manifest(tmp_path, "candidate")
    (candidate.parent / "evaluation.csv").write_text("changed data\n")
    report, pairs = comparison.compare_evaluations(baseline, candidate)
    assert "SHA256 mismatch" in " ".join(report["reasons"])
    assert (
        report["inputs"]["candidate"]["table_sha256"]
        == hashlib.sha256(b"changed data\n").hexdigest()
    )
    assert pairs == []


def test_cli_writes_diagnostics_and_removes_stale_pairs_on_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    baseline = make_manifest(tmp_path, "baseline")
    candidate = make_manifest(tmp_path, "candidate")
    output = tmp_path / "comparison"
    args = ["--baseline", str(baseline), "--candidate", str(candidate), "--output-dir", str(output)]
    assert comparison.main(args) == 0
    assert (output / "paired.csv").is_file()
    first = json.loads((output / "comparison.json").read_text())
    assert first["diagnostics"]["score"]["overall"]["unchanged"] == 3
    edit_manifest(candidate, evaluation_kind="diagnostic")
    assert comparison.main(args) == 2
    assert not (output / "paired.csv").exists()
    second = json.loads((output / "comparison.json").read_text())
    assert second["comparison"] == "not_comparable"
    assert "diagnostics" not in second
    assert "not_comparable" in capsys.readouterr().out


def test_tolerance_is_only_a_diagnostic_threshold_and_ungrouped_tables_work(tmp_path: Path) -> None:
    baseline = make_manifest(tmp_path, "baseline")
    candidate = make_manifest(tmp_path, "candidate", [("u1", "a", 0.75, 1), *BASE_ROWS[1:]])
    for path in (baseline, candidate):
        edit_manifest(path, group_column=None)
    report, pairs = comparison.compare_evaluations(baseline, candidate, tolerance=0.25)
    assert report["diagnostics"]["score"]["overall"]["unchanged"] == 3
    assert report["diagnostics"]["score"]["overall"]["unit_mean_delta"] == pytest.approx(0.25 / 3)
    assert report["diagnostics"]["score"]["by_group"] == {}
    assert pairs[0]["delta"] == 0.25


@pytest.mark.parametrize("tolerance", ["-1", "nan", "inf"])
def test_cli_rejects_invalid_tolerance(tmp_path: Path, tolerance: str) -> None:
    with pytest.raises(SystemExit) as exc:
        comparison.main(
            ["--baseline", "unused", "--candidate", "unused", f"--tolerance={tolerance}"]
        )
    assert exc.value.code == 2


def test_finite_values_with_overflowing_difference_are_not_compared(tmp_path: Path) -> None:
    baseline = make_manifest(tmp_path, "baseline", [("u1", "a", -1e308, 1), *BASE_ROWS[1:]])
    candidate = make_manifest(tmp_path, "candidate", [("u1", "a", 1e308, 1), *BASE_ROWS[1:]])
    report, pairs = comparison.compare_evaluations(baseline, candidate)
    assert "overflow" in " ".join(report["reasons"])
    assert pairs == []


@pytest.mark.parametrize("schema_version", [1, 99, True, None])
@pytest.mark.parametrize("filename", ["paired.csv", "comparison.json"])
def test_output_cannot_overwrite_input_even_when_schema_or_hash_is_invalid(
    tmp_path: Path, schema_version, filename: str
) -> None:
    baseline = make_manifest(tmp_path, "baseline")
    candidate = make_manifest(tmp_path, "candidate")
    manifest = json.loads(candidate.read_text())
    manifest["table"]["path"] = filename
    manifest["schema_version"] = schema_version
    candidate.write_text(json.dumps(manifest))
    input_table = candidate.parent / filename
    input_table.write_text("invalid table with the wrong SHA\n")
    with pytest.raises(SystemExit) as exc:
        comparison.main(
            [
                "--baseline",
                str(baseline),
                "--candidate",
                str(candidate),
                "--output-dir",
                str(candidate.parent),
            ]
        )
    assert exc.value.code == 2
    assert input_table.read_text() == "invalid table with the wrong SHA\n"


@pytest.mark.parametrize("bad_manifest", ['{"table":', '{"table": {}, "table": {}}'])
def test_unreadable_manifest_does_not_mutate_output_directory(
    tmp_path: Path, bad_manifest: str, capsys: pytest.CaptureFixture
) -> None:
    baseline = make_manifest(tmp_path, "baseline")
    candidate = make_manifest(tmp_path, "candidate")
    candidate.write_text(bad_manifest)
    output = tmp_path / "output"
    output.mkdir()
    (output / "paired.csv").write_text("preserve possible input data")
    (output / "comparison.json").write_text("preserve earlier report")
    assert (
        comparison.main(
            [
                "--baseline",
                str(baseline),
                "--candidate",
                str(candidate),
                "--output-dir",
                str(output),
            ]
        )
        == 2
    )
    assert (output / "paired.csv").read_text() == "preserve possible input data"
    assert (output / "comparison.json").read_text() == "preserve earlier report"
    assert "output not written" in capsys.readouterr().out
