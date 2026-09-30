from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

SOURCE = Path(__file__).resolve().parents[1] / "exp046_x138_author_head_comparison_inference.py"


def notebook_function(name: str, **scope: object):
    tree = ast.parse(SOURCE.read_text())
    node = next(
        item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == name
    )
    namespace = {"__builtins__": __builtins__, **scope}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace[name]


def checkpoint(path: Path, *, bad_scale: bool = False) -> None:
    state = {
        "state_dict": {
            "0.weight": torch.zeros(32, 224),
            "0.bias": torch.zeros(32),
            "2.weight": torch.zeros(3, 32),
            "2.bias": torch.zeros(3),
        },
        "mean": torch.zeros(224),
        "scale": torch.ones(224),
    }
    if bad_scale:
        state["scale"][5] = 0
    torch.save(state, path)


def test_checkpoint_bundle_keeps_its_own_normalization(tmp_path: Path) -> None:
    check = notebook_function("_check_head_bundle", _head_torch=torch, _self_hashlib=hashlib)
    valid = tmp_path / "valid.pt"
    checkpoint(valid)
    assert check(valid)["parameter_count"] == 7299
    invalid = tmp_path / "invalid.pt"
    checkpoint(invalid, bad_scale=True)
    with pytest.raises(RuntimeError, match="nonpositive normalization scale"):
        check(invalid)


def test_original_candidates_use_native_voxels(tmp_path: Path) -> None:
    audit_root = tmp_path
    raw = audit_root / "self" / "raw_detector_coordinates" / "movie"
    raw.mkdir(parents=True)
    np.savez(raw / "0000.npz", coords=np.array([[0, 2, 3, 4]], dtype=np.float32))
    np.savez(raw / "0001.npz", coords=np.array([[1, 5, 6, 7]], dtype=np.float32))
    load = notebook_function("_audit_original_candidates", np=np, AUDIT_ROOT=audit_root)
    assert load("movie").tolist() == [[0, 2, 12, 16], [1, 5, 24, 28]]


def test_each_arm_receives_separate_checkpoint_path() -> None:
    source = SOURCE.read_text()
    assert '"V1284_MODE": "candidate"' in source
    assert '"V1284_HEAD": str(_HEAD_PATHS[arm])' in source
    assert 'AUDIT_ARMS = ("self", "author")' in source
