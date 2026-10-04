"""Run common tests and each experiment in separate Python processes.

Historical experiments intentionally reuse module and test filenames. Sharing
sys.modules between them can silently test another experiment's implementation.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

try:
    from .config_utils import ROOT, configured_project_path
except ImportError:
    from config_utils import ROOT, configured_project_path


def test_groups(root: Path) -> list[Path]:
    experiments = configured_project_path("paths.experiments_dir", "experiments", root=root)
    groups = [root / "tests"] if (root / "tests").is_dir() else []
    groups.extend(sorted(path for path in experiments.glob("*/tests") if path.is_dir()))
    return groups


def run_groups(root: Path, groups: list[Path]) -> int:
    failed = []
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    for group in groups:
        label = group.relative_to(root).as_posix() if group.is_relative_to(root) else str(group)
        print(f"\nTesting {label}", flush=True)
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(group)],
            cwd=root,
            env=env,
            check=False,
        )
        if result.returncode:
            failed.append(label)
    print(f"\nTest groups: {len(groups) - len(failed)} passed, {len(failed)} failed", flush=True)
    for group in failed:
        print(f"FAILED {group}", flush=True)
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(run_groups(ROOT, test_groups(ROOT)))
