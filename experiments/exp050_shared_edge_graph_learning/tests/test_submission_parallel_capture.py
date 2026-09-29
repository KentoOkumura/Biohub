"""Validate exp050 test-video feature cache sharding before graph decoding."""

from __future__ import annotations

import ast
import subprocess
import time
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "exp050_shared_edge_graph_learning_submission_gnn.py"


def load_capture_function(name):
    module = ast.parse(SOURCE.read_text())
    function = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == name
    )
    namespace = {"subprocess": subprocess, "time": time}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace[name]


def test_interleaved_gpu_caches_merge_without_changing_payload(tmp_path):
    stems = ["a", "b", "c", "d"]
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    shard_roots = [tmp_path / "cache_gpu0", tmp_path / "cache_gpu1"]
    for root in shard_roots:
        root.mkdir()
    for index, stem in enumerate(stems):
        (shard_roots[index % 2] / f"{stem}.npz").write_bytes(stem.encode())

    load_capture_function("_merge_feature_capture_shards")(cache_root, shard_roots, stems)

    assert {path.stem for path in cache_root.iterdir()} == set(stems)
    assert all((cache_root / f"{stem}.npz").read_bytes() == stem.encode() for stem in stems)
    assert all(not root.exists() for root in shard_roots)


def test_missing_video_is_rejected_before_moving_cache(tmp_path):
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    shard_roots = [tmp_path / "cache_gpu0", tmp_path / "cache_gpu1"]
    for root in shard_roots:
        root.mkdir()
    (shard_roots[0] / "a.npz").write_bytes(b"a")

    with pytest.raises(RuntimeError, match="feature mismatch"):
        load_capture_function("_merge_feature_capture_shards")(cache_root, shard_roots, ["a", "b"])

    assert (shard_roots[0] / "a.npz").read_bytes() == b"a"
    assert not any(cache_root.iterdir())
    assert not (tmp_path / "cache_merge_staging").exists()


class FakeProcess:
    def __init__(self, return_code):
        self.return_code = return_code
        self.terminated = False

    def poll(self):
        return self.return_code

    def terminate(self):
        self.terminated = True
        self.return_code = -15

    def wait(self, timeout=None):
        return self.return_code


def test_failed_gpu_shard_stops_other_shard():
    failed = FakeProcess(2)
    running = FakeProcess(None)
    with pytest.raises(subprocess.CalledProcessError):
        load_capture_function("_wait_for_feature_capture_shards")(
            {0: failed, 1: running},
            {0: ["first"], 1: ["second"]},
            time.monotonic() + 30,
        )
    assert running.terminated
