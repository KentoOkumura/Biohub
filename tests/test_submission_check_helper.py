from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest

from scripts.prepare_kaggle_notebooks import make_support_cell

HELPER_PATH = (
    Path(__file__).resolve().parents[1]
    / ".agents/skills/kaggle-submit-check/scripts/check_submission.py"
)
spec = importlib.util.spec_from_file_location("submission_check_helper", HELPER_PATH)
assert spec and spec.loader
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def write_package(package: Path) -> dict:
    project = b"runtime:\n  kaggle:\n    enable_gpu: false\n    enable_internet: false\n"
    (package / "project.yml").write_bytes(project)
    notebook = {
        "cells": [make_support_cell({"project.yml": project})],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (package / "notebook.ipynb").write_text(json.dumps(notebook))
    metadata = {
        "id": "owner/check-notebook",
        "title": "check notebook",
        "code_file": "notebook.ipynb",
        "competition_sources": ["test-competition"],
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
    }
    (package / "kernel-metadata.json").write_text(json.dumps(metadata))
    return metadata


def test_helper_accepts_package_verified_by_canonical_validator(tmp_path):
    write_package(tmp_path)
    reporter = helper.Reporter()
    helper.check_metadata(tmp_path / "kernel-metadata.json", reporter)
    assert not reporter.failures
    assert any("package validation passed" in message for message in reporter.passes)


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ({"code_file": "missing.ipynb"}, "code_file does not exist"),
        ({"title": "different title"}, "id/title slug mismatch"),
        ({"enable_tpu": True}, "enable_tpu is unsupported"),
        ({"enable_gpu": True}, "enable_gpu does not match"),
        ({"competition_sources": []}, "competition_sources is empty"),
    ],
)
def test_helper_rejects_invalid_metadata_without_reporting_pass(tmp_path, change, error):
    metadata = write_package(tmp_path)
    metadata.update(change)
    metadata_path = tmp_path / "kernel-metadata.json"
    metadata_path.write_text(json.dumps(metadata))
    reporter = helper.Reporter()
    helper.check_metadata(metadata_path, reporter)
    assert any(error in message for message in reporter.failures)
    assert not reporter.passes


@pytest.mark.parametrize("missing", ["project.yml", "notebook.ipynb"])
def test_helper_rejects_incomplete_package(tmp_path, missing):
    write_package(tmp_path)
    (tmp_path / missing).unlink()
    reporter = helper.Reporter()
    helper.check_metadata(tmp_path / "kernel-metadata.json", reporter)
    assert reporter.failures
    assert not reporter.passes


def test_helper_rejects_non_object_metadata_and_returns_failure(tmp_path, monkeypatch):
    write_package(tmp_path)
    metadata = tmp_path / "kernel-metadata.json"
    metadata.write_text("[]")
    monkeypatch.setattr(sys, "argv", [str(HELPER_PATH), str(metadata)])
    assert helper.main() == 1


@pytest.mark.parametrize("target_kind", ["empty_directory", "unknown_file", "ambiguous_csv"])
def test_helper_fails_when_it_cannot_identify_what_to_validate(tmp_path, monkeypatch, target_kind):
    target = tmp_path
    if target_kind == "unknown_file":
        target = tmp_path / "notes.txt"
        target.write_text("not a submission")
    elif target_kind == "ambiguous_csv":
        (target / "one.csv").write_text("id,target\n1,0\n")
        (target / "two.csv").write_text("id,target\n1,1\n")
    monkeypatch.setattr(sys, "argv", [str(HELPER_PATH), str(target)])
    assert helper.main() == 1


def test_helper_checks_every_zip_in_the_selected_directory(tmp_path, monkeypatch):
    for index in range(3):
        with zipfile.ZipFile(tmp_path / f"{index}.zip", "w") as archive:
            archive.writestr("submission.csv", "id,target\n1,0\n")
    (tmp_path / "3.zip").write_text("invalid archive")
    monkeypatch.setattr(helper, "check_csv", lambda path, reporter, sample: reporter.ok(str(path)))
    reporter = helper.Reporter()
    helper.check_dir(tmp_path, reporter)
    assert any("3.zip: invalid zip archive" in message for message in reporter.failures)
