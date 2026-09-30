from __future__ import annotations

import json

import numpy as np
import torch

from diagnose_pair_effects import (
    VARIANTS,
    add_pair,
    empty_counts,
    restore_counts,
    save_json_atomic,
    serialize_counts,
)


def test_counts_roundtrip_for_resume(tmp_path) -> None:
    counts = {variant: empty_counts() for variant in VARIANTS}
    truth = np.array([[True, False], [False, True]])
    add_pair(counts["public_no_attention"], torch.tensor([[2.0, 0.0], [0.0, 2.0]]), truth)
    path = tmp_path / "progress.json"
    save_json_atomic(path, {"counts": serialize_counts(counts)})
    assert not (tmp_path / "progress.json.tmp").exists()
    restored = restore_counts(json.loads(path.read_text())["counts"])
    for variant in VARIANTS:
        for key, expected in counts[variant].items():
            actual = restored[variant][key]
            if isinstance(expected, np.ndarray):
                assert np.array_equal(actual, expected)
            else:
                assert actual == expected
