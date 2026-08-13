from __future__ import annotations

from pathlib import Path

import pytest

from scripts.config_utils import ROOT, configured_project_path


def configured_experiments_dir(root: Path = ROOT) -> Path:
    return configured_project_path("paths.experiments_dir", "experiments", root=root)


def pytest_configure(config: pytest.Config) -> None:
    """Add configured experiment tests only for pytest's default collection."""
    if config.getoption("file_or_dir"):
        return
    experiments_dir = configured_experiments_dir()
    if experiments_dir.is_dir():
        config.args.append(str(experiments_dir))
