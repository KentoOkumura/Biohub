"""Build the self-contained diagnostic source from the exact inference helpers."""

from __future__ import annotations

import ast
from pathlib import Path

EXP = Path(__file__).resolve().parent


def extract(path: Path, roots: list[str]) -> str:
    source = path.read_text()
    tree = ast.parse(source)
    definitions = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            definitions[node.name] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    definitions[target.id] = node
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            definitions[node.target.id] = node
    selected = set(roots)
    pending = list(roots)
    while pending:
        name = pending.pop()
        node = definitions[name]
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and child.id in definitions and child.id not in selected:
                selected.add(child.id)
                pending.append(child.id)
    lines = source.splitlines()
    selected_nodes = {id(definitions[name]): definitions[name] for name in selected}
    blocks = []
    for node in sorted(selected_nodes.values(), key=lambda item: item.lineno):
        first = min([node.lineno, *[d.lineno for d in getattr(node, "decorator_list", [])]])
        blocks.append("\n".join(lines[first - 1 : node.end_lineno]))
    return "\n\n\n".join(blocks)


def main() -> None:
    destination = EXP / "exp027_multi_frame_tracker_diagnostic.py"
    execution_marker = "# %% [markdown]\n# ## 4. Verify saved inputs"
    source = destination.read_text()
    execution = execution_marker + source.split(execution_marker, 1)[1]
    header = """# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # exp027: saved-model previous-context diagnostic
#
# 1. Imports and offline runtime
# 2. Exact cache, teacher, and model helpers
# 3. Paired diagnostic definitions
# 4. Verify saved inputs and fixed evaluation windows
# 5. Run three inference conditions on each embryo
# 6. Reproduce pilot metrics and save paired results
#
# Use the same three-frame weights with previous points included or masked.
# The separately trained two-frame model is a reference. No weights are trained.
# Threshold curves describe sparse-label predictions; they are not official scores.

# %%
from __future__ import annotations

# ruff: noqa: E402,I001
import csv
import hashlib
import importlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

EXPERIMENT = "exp027_multi_frame_tracker"
COMPETITION = "biohub-cell-tracking-during-development"
WORKING_ROOT = Path.cwd()
NOTEBOOK_STARTED = time.perf_counter()
if not Path('/kaggle/input').is_dir():
    raise RuntimeError('The authoritative diagnostic must run on Kaggle')
config = yaml.safe_load((WORKING_ROOT / 'config.yaml').read_text())
diag_cfg = config['diagnostic']
cache_cfg = config['data']['cache']
public_cfg = config['model']['public_source']
params_cfg = config['model']['params']
teacher_cfg = config['model']['teacher']
loss_cfg = config['model']['loss']
METRICS_PATH = WORKING_ROOT / 'metrics.json'

# %% [markdown]
# ## 1. Offline GEFF runtime
# Install the same offline dependencies as train v4 before reading annotations.

# %%
"""
    train = EXP / "exp027_multi_frame_tracker_train.py"
    offline = extract(train, ["ensure_geff_runtime_dependencies"])
    core = extract(
        EXP / "frozen_tracker.py",
        [
            "file_sha256",
            "validate_cache_summary",
            "discover_cache_paths",
            "recompute_cache_identity_sha256",
            "load_annotation_graph",
            "build_three_frame_example",
            "collate_three_frame_examples",
            "move_batch_to_device",
            "legacy_focal_bce",
            "canonical_state_sha256",
            "seed_everything",
        ],
    )
    model = extract(EXP / "local_tracker_model.py", ["LocalThreeFrameTracker"])
    diagnostic = extract(
        EXP / "context_diagnostic.py",
        [
            "diagnose_pair",
            "aggregate_diagnostics",
            "compare_known_edges",
            "paired_video_bootstrap",
        ],
    )
    resolve = extract(train, ["resolve_cache_output_root", "resolve_train_dir", "update_metrics"])
    body = (
        header + offline + "\n\nensure_geff_runtime_dependencies()\n"
        "import torch\nfrom torch import nn\n"
        "from torch.utils.checkpoint import checkpoint as grad_checkpoint\n\n"
        "# %% [markdown]\n# ## 2. Exact cache, teacher, and model helpers\n\n# %%\n"
        + core
        + "\n\n"
        + model
        + "\n\n"
        + resolve
        + "\n\n# %% [markdown]\n# ## 3. Paired diagnostic definitions\n\n# %%\n"
        + diagnostic
        + "\n\n"
        + execution
    )
    destination = EXP / "exp027_multi_frame_tracker_diagnostic.py"
    destination.write_text(body)
    print(f"Wrote {destination.name}: {len(body.splitlines())} lines")


if __name__ == "__main__":
    main()
