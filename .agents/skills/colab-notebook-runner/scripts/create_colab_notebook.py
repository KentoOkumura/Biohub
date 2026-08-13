#!/usr/bin/env python3
"""Create a generic Google Colab runner notebook for one experiment."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

EXPERIMENT_RE = re.compile(r"exp\d+_[a-z0-9_]+")
REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from config_utils import configured_project_path  # noqa: E402


def markdown_cell(source: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code_cell(source: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def build_notebook(args: argparse.Namespace) -> dict[str, Any]:
    cache_sources = [str(path) for path in args.cache_source]
    resolved_experiments_path = configured_project_path(
        "paths.experiments_dir", "experiments", root=REPO_ROOT
    )
    try:
        experiments_path = resolved_experiments_path.relative_to(REPO_ROOT)
    except ValueError:
        experiments_path = resolved_experiments_path
    configuration = "\n".join(
        [
            "from pathlib import Path",
            "import json",
            "import os",
            "import shlex",
            "import shutil",
            "import subprocess",
            "import sys",
            "",
            f"EXPERIMENT = {args.experiment!r}",
            f"DRIVE_ROOT = Path({str(args.drive_root)!r})",
            f"RUN_COMMAND = {args.run_command!r}",
            f"CACHE_SOURCES = {cache_sources!r}",
            f"LOCAL_CACHE_DIR = Path({str(args.local_cache_dir)!r}) / EXPERIMENT",
            f"EXPERIMENTS_PATH = Path({str(experiments_path)!r})",
            "EXPERIMENTS_DIR = ("
            "EXPERIMENTS_PATH if EXPERIMENTS_PATH.is_absolute() "
            "else DRIVE_ROOT / EXPERIMENTS_PATH"
            ")",
            "RUN_DIR = EXPERIMENTS_DIR / EXPERIMENT / 'artifacts' / 'colab_runs'",
        ]
    )
    checks = """required = [DRIVE_ROOT / "project.yml", EXPERIMENTS_DIR / EXPERIMENT]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise FileNotFoundError(f"Missing required project paths: {missing}")

print("python", sys.version)
print("cpu_count", os.cpu_count())
try:
    import psutil
    print("ram_gb", round(psutil.virtual_memory().total / 1024**3, 2))
except ImportError:
    print("ram_gb", "install psutil to inspect")
try:
    import torch
    print("cuda", torch.cuda.is_available())
    print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
except ImportError:
    print("gpu", "torch is not installed")
"""
    copy_caches = """LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
local_cache_paths = []
for relative in CACHE_SOURCES:
    source = DRIVE_ROOT / relative
    if not source.is_file():
        raise FileNotFoundError(f"Missing cache source: {source}")
    destination = LOCAL_CACHE_DIR / source.name
    shutil.copy2(source, destination)
    local_cache_paths.append(str(destination))
    print("copied", source, "->", destination)
print("local_cache_paths", json.dumps(local_cache_paths, indent=2))
"""
    run = """RUN_DIR.mkdir(parents=True, exist_ok=True)
log_path = RUN_DIR / "latest.log"
status_path = RUN_DIR / "latest_status.json"
status_path.write_text(
    json.dumps(
        {"experiment": EXPERIMENT, "status": "running", "command": RUN_COMMAND},
        indent=2,
    )
)

with log_path.open("w") as log:
    completed = subprocess.run(
        shlex.split(RUN_COMMAND),
        cwd=DRIVE_ROOT,
        stdout=log,
        stderr=subprocess.STDOUT,
        check=False,
    )

status = "completed" if completed.returncode == 0 else "failed"
status_path.write_text(
    json.dumps(
        {
            "experiment": EXPERIMENT,
            "status": status,
            "returncode": completed.returncode,
            "command": RUN_COMMAND,
        },
        indent=2,
    )
)
print(status_path.read_text())
print(log_path.read_text()[-4000:])
if completed.returncode != 0:
    raise RuntimeError(f"Experiment command failed with exit code {completed.returncode}")
"""
    return {
        "cells": [
            markdown_cell(f"# {args.experiment} Colab runner\n"),
            markdown_cell("## 1. Mount Google Drive\n"),
            code_cell("from google.colab import drive\ndrive.mount('/content/drive')\n"),
            markdown_cell("## 2. Configuration\n"),
            code_cell(configuration + "\n"),
            markdown_cell("## 3. Runtime and input checks\n"),
            code_cell(checks),
            markdown_cell("## 4. Copy declared caches to local storage\n"),
            code_cell(copy_caches),
            markdown_cell("## 5. Execute and persist status\n"),
            code_cell(run),
        ],
        "metadata": {
            "colab": {"name": args.output.name, "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--drive-root", required=True, type=Path)
    parser.add_argument("--run-command", required=True)
    parser.add_argument("--cache-source", action="append", default=[], type=Path)
    parser.add_argument("--local-cache-dir", default="/content/kaggle_cache", type=Path)
    args = parser.parse_args()
    if EXPERIMENT_RE.fullmatch(args.experiment) is None:
        parser.error("--experiment must match exp<digits>_<lowercase_name>")
    return args


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(build_notebook(args), indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
