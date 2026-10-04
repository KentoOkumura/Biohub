from __future__ import annotations

import subprocess

from scripts import run_tests


def test_experiments_with_identical_module_names_are_isolated(tmp_path):
    groups = []
    for number in (1, 2):
        group = tmp_path / f"exp{number:03d}/tests"
        group.mkdir(parents=True)
        (group / "helper.py").write_text(f"VALUE = {number}\n")
        (group / "test_contract.py").write_text(
            f"import helper\ndef test_identity():\n    assert helper.VALUE == {number}\n"
        )
        groups.append(group)
    assert run_tests.run_groups(tmp_path, groups) == 0


def test_failure_is_reported_and_does_not_skip_later_groups(tmp_path, monkeypatch):
    groups = [tmp_path.parent / "external-tests", tmp_path / "second"]
    observed = []

    def fake_run(command, **kwargs):
        observed.append(command[-1])
        return subprocess.CompletedProcess(command, 1 if len(observed) == 1 else 0)

    monkeypatch.setattr(run_tests.subprocess, "run", fake_run)
    assert run_tests.run_groups(tmp_path, groups) == 1
    assert observed == [str(group) for group in groups]
