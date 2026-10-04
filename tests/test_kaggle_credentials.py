from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from kagglesdk import kaggle_env

ROOT = Path(__file__).resolve().parents[1]
PLATFORM = ROOT / ".agents/skills/kaggle-platform"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


checker = load_module("credential_capabilities", PLATFORM / "shared/check_all_credentials.py")
report = load_module("report_credentials", PLATFORM / "modules/comp-report/scripts/utils.py")
badge = load_module("badge_credentials", PLATFORM / "modules/badge-collector/scripts/utils.py")


@pytest.fixture(autouse=True)
def isolated_credentials(tmp_path, monkeypatch):
    """Every test uses dummy credentials and never imports/authenticates Kaggle."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(
        kaggle_env.os.path,
        "expanduser",
        lambda value: str(tmp_path / value[2:]) if value.startswith("~/") else value,
    )
    monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising=False)
    for name in ("KAGGLE_API_TOKEN", "KAGGLE_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"):
        monkeypatch.delenv(name, raising=False)
    directory = tmp_path / ".kaggle"
    directory.mkdir()
    return directory


@pytest.mark.parametrize("requirement", ["any", "cli", "python-api"])
def test_oauth_only_works_for_cli_and_kaggle_python_api(isolated_credentials, requirement, capsys):
    token = "dummy-refresh-secret-never-display"
    (isolated_credentials / "credentials.json").write_text(
        json.dumps({"refresh_token": token, "access_token_expiration": "2000-01-01T00:00:00+00:00"})
    )
    assert checker.check_all_credentials(requirement=requirement)
    output = capsys.readouterr().out
    assert token not in output


@pytest.mark.parametrize("requirement", ["api-token", "kagglehub"])
def test_oauth_only_does_not_satisfy_other_clients(isolated_credentials, requirement):
    (isolated_credentials / "credentials.json").write_text(
        json.dumps({"refresh_token": "dummy-refresh"})
    )
    assert not checker.check_all_credentials(requirement=requirement)


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        "[]",
        "{}",
        '{"refresh_token": ""}',
        '{"refresh_token": null}',
        '{"refresh_token": 123}',
        '{"refresh_token": "dummy", "access_token_expiration": "bad date"}',
        '{"refresh_token": "dummy", "access_token_expiration": "2026-01-01T00:00:00"}',
    ],
)
def test_file_presence_is_not_valid_oauth_configuration(isolated_credentials, content):
    (isolated_credentials / "credentials.json").write_text(content)
    assert not checker.check_all_credentials(requirement="python-api")


@pytest.mark.parametrize("requirement", ["cli", "python-api", "kagglehub", "api-token"])
def test_api_token_supports_each_client(monkeypatch, requirement):
    monkeypatch.setenv("KAGGLE_API_TOKEN", "dummy-api-token")
    assert checker.check_all_credentials(requirement=requirement)


@pytest.mark.parametrize("requirement", ["cli", "python-api", "kagglehub", "api-token"])
@pytest.mark.parametrize("contents, expected", [("dummy-file-token\n", True), ("  \n", False)])
def test_environment_token_file_uses_contents_without_fallback(
    isolated_credentials, monkeypatch, requirement, contents, expected, capsys
):
    selected = isolated_credentials / "selected-token"
    selected.write_text(contents)
    (isolated_credentials / "access_token").write_text("must-not-use-default-token")
    monkeypatch.setenv("KAGGLE_API_TOKEN", str(selected))
    assert checker.check_all_credentials(requirement=requirement) is expected
    assert "dummy-file-token" not in capsys.readouterr().out


@pytest.mark.parametrize("requirement", ["cli", "python-api", "kagglehub", "api-token"])
@pytest.mark.parametrize("primary", [None, "  \n"])
def test_token_txt_fallback(isolated_credentials, requirement, primary):
    if primary is not None:
        (isolated_credentials / "access_token").write_text(primary)
    (isolated_credentials / "access_token.txt").write_text("dummy-txt-token")
    assert checker.check_all_credentials(requirement=requirement)


def test_primary_token_file_precedes_txt_fallback(isolated_credentials):
    (isolated_credentials / "access_token").write_text("dummy-primary-token")
    (isolated_credentials / "access_token.txt").write_text("dummy-fallback-token")
    assert checker._resolve_api_token()[0] == "dummy-primary-token"


@pytest.mark.parametrize("requirement", ["cli", "python-api", "kagglehub"])
def test_legacy_pair_supports_each_non_mcp_client(monkeypatch, requirement):
    monkeypatch.setenv("KAGGLE_USERNAME", "dummy-user")
    monkeypatch.setenv("KAGGLE_KEY", "dummy-legacy-key")
    assert checker.check_all_credentials(requirement=requirement)


@pytest.mark.parametrize(
    "environment, file_data",
    [
        ({"KAGGLE_USERNAME": "dummy-env-user"}, {"key": "dummy-file-key"}),
        ({"KAGGLE_KEY": "dummy-env-key"}, {"username": "dummy-file-user"}),
    ],
)
def test_kagglehub_does_not_merge_incomplete_legacy_sources(
    isolated_credentials, monkeypatch, environment, file_data
):
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    (isolated_credentials / "kaggle.json").write_text(json.dumps(file_data))
    assert checker.check_all_credentials(requirement="python-api")
    assert not checker.check_all_credentials(requirement="kagglehub")


def test_kagglehub_uses_complete_file_when_environment_pair_is_incomplete(
    isolated_credentials, monkeypatch, capsys
):
    monkeypatch.setenv("KAGGLE_USERNAME", "dummy-env-user")
    (isolated_credentials / "kaggle.json").write_text(
        '{"username": "dummy-file-user", "key": "dummy-file-key"}'
    )
    assert checker.check_all_credentials(requirement="kagglehub")
    output = capsys.readouterr().out
    assert "KAGGLE_USERNAME: dummy-file-user (from kaggle.json)" in output
    assert "dummy-env-user" not in output


def test_competition_report_authenticates_oauth_only_configuration(
    isolated_credentials, monkeypatch
):
    (isolated_credentials / "credentials.json").write_text(
        json.dumps({"refresh_token": "dummy-refresh"})
    )
    calls = []

    def fake_get_api():
        calls.append("authenticate")
        return SimpleNamespace(competitions_list=lambda **kwargs: [])

    monkeypatch.setattr(report, "get_api", fake_get_api)
    assert report.check_credentials()
    assert calls == ["authenticate"]


def test_competition_report_rejects_empty_oauth_without_calling_api(
    isolated_credentials, monkeypatch
):
    (isolated_credentials / "credentials.json").write_text("{}")
    monkeypatch.setattr(report, "get_api", lambda: pytest.fail("Must not authenticate"))
    assert not report.check_credentials()


def test_competition_report_auth_failure_returns_false(isolated_credentials, monkeypatch):
    (isolated_credentials / "credentials.json").write_text(
        json.dumps({"refresh_token": "dummy-refresh"})
    )

    def failed_authentication():
        raise SystemExit(1)

    monkeypatch.setattr(report, "get_api", failed_authentication)
    assert not report.check_credentials()


@pytest.mark.parametrize("phases, requirement", [([1], "python-api"), ([2, 3, 5], "cli")])
def test_badge_uses_tracked_canonical_checker(monkeypatch, phases, requirement):
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="configured")

    monkeypatch.setattr(badge.subprocess, "run", fake_run)
    assert badge.REPO_ROOT == ROOT
    assert badge.check_credentials(phases)
    assert calls == [
        [
            sys.executable,
            str(PLATFORM / "shared/check_all_credentials.py"),
            "--require",
            requirement,
        ]
    ]


def test_badge_missing_tracked_checker_is_not_hidden_by_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(badge, "SHARED_SCRIPTS", tmp_path)
    with pytest.raises(FileNotFoundError, match="Credential checker is missing"):
        badge.check_credentials([1])
