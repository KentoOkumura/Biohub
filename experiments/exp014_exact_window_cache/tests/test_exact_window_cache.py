from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
INFERENCE_SOURCE = ROOT / "exp014_exact_window_cache_inference.py"
CONFIG_PATH = ROOT / "config.yaml"
REFERENCE_NOTEBOOK = ROOT / "assets" / "reference_notebook" / "biohub-cell-tracking-0-947-lb.ipynb"
EXPECTED_REFERENCE_SHA256 = "ae8e01a262211045161984e469e8be23e3386bab9140fe12df503dc6a1e010e6"

sys.path.insert(0, str(ROOT))
from window_cache import (  # noqa: E402
    assert_exact_arrays,
    read_window_cache,
    write_window_cache,
)


def test_window_cache_round_trip_preserves_dtype_shape_and_values(tmp_path: Path) -> None:
    arrays = {
        "candidate_ids_src": np.array([10, 11], dtype=np.int64),
        "coords_src_physical": np.array([[1.625, 2.0, 3.0]], dtype=np.float32),
        "primary_features_src": np.arange(64, dtype=np.float16).reshape(2, 32),
        "secondary_features_src": np.arange(64, dtype=np.float32).reshape(2, 32),
    }
    metadata = {
        "experiment": "exp014_exact_window_cache",
        "dataset": "movie",
        "window_frames": [3, 4],
    }
    path = tmp_path / "movie__000003__000004.npz"
    written = write_window_cache(path, metadata=metadata, arrays=arrays)
    loaded, read = read_window_cache(path, expected_metadata=metadata)
    assert_exact_arrays(arrays, loaded)
    assert written["content_sha256"] == read["content_sha256"]
    assert written["schema_sha256"] == read["schema_sha256"]
    assert written["bytes"] == path.stat().st_size


def test_window_cache_rejects_wrong_window_identity(tmp_path: Path) -> None:
    path = tmp_path / "cache.npz"
    write_window_cache(
        path,
        metadata={"dataset": "movie", "window_frames": [0, 1]},
        arrays={"features": np.zeros((1, 32), dtype=np.float32)},
    )
    with pytest.raises(ValueError, match="cache_metadata_mismatch"):
        read_window_cache(
            path,
            expected_metadata={"dataset": "movie", "window_frames": [1, 2]},
        )


def test_reference_notebook_and_model_contract_are_pinned() -> None:
    assert hashlib.sha256(REFERENCE_NOTEBOOK.read_bytes()).hexdigest() == EXPECTED_REFERENCE_SHA256
    config = CONFIG_PATH.read_text(encoding="utf-8")
    assert "parent: exp013_public_notebook_replay" in config
    assert "hypothesis_id: HYP-20260910-12" in config
    assert "backlog_candidate: exact_window_cache" in config
    assert "primary: 12f6881e" in config
    assert "secondary: 9bac2fa0" in config
    assert "expected_submission_sha256: 0319ba6d" in config


def test_inference_installs_cache_round_trip_and_exact_guards() -> None:
    source = INFERENCE_SOURCE.read_text(encoding="utf-8")
    for marker in (
        "EXP014_WINDOW_CACHE_ADDITION_START",
        "write_window_cache",
        "read_window_cache",
        "assert_exact_arrays",
        "WINDOW_CACHE_SCORE_MISMATCH",
        "window_cache_manifest_*.jsonl",
        "_EXP014_BASELINE_SUBMISSION_SHA256",
    ):
        assert marker in source
    assert "np.savez_compressed" not in source
    assert "torch.use_deterministic_algorithms" not in source
