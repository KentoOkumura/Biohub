from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app import oof_analysis_app


def test_oof_discovery_includes_nested_artifacts_and_legacy_paths(tmp_path, monkeypatch):
    expected = [
        tmp_path / "exp001_demo/artifacts/oof.csv",
        tmp_path / "exp001_demo/artifacts/train/fold0/validation_oof.csv",
        tmp_path / "exp002_legacy/features/oof.csv",
        tmp_path / "exp002_legacy/oof_predictions.csv",
    ]
    other = tmp_path / "exp001_demo/artifacts/train/submission.csv"
    for path in [*expected, other]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("id,prediction\n1,0.5\n")
    monkeypatch.setattr(oof_analysis_app, "EXPERIMENTS_DIR", tmp_path)

    assert oof_analysis_app.list_oof_files() == sorted(expected)


@pytest.mark.parametrize("name", ["streamlit_app.py", "oof_analysis_app.py"])
def test_app_starts_without_experiments(tmp_path, monkeypatch, name):
    from scripts import config_utils

    monkeypatch.setattr(config_utils, "ROOT", tmp_path / "project")
    monkeypatch.setattr(
        config_utils,
        "load_project_config",
        lambda: {
            "paths": {
                "experiments_dir": str(tmp_path / "external-runs"),
                "submissions_file": str(tmp_path / "SUBMISSIONS.md"),
            }
        },
    )
    source = Path(__file__).resolve().parents[1] / "app" / name
    app = AppTest.from_file(str(source)).run(timeout=15)

    assert not app.exception
    assert len(app.info) == 1
