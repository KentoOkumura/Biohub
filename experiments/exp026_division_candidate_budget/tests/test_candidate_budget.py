import sys
from pathlib import Path
from runpy import run_path
from types import SimpleNamespace

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1] / "exp026_division_candidate_budget_diagnostic.py"
FUNCTIONS = run_path(str(SOURCE), run_name="budget_test")


def test_unordered_geometry_and_equal_window_budget():
    enumerate_triplets = FUNCTIONS["enumerate_triplets"]
    select = FUNCTIONS["budget_indices"]
    sources = np.array([7, 5])
    source_xyz = np.array([[10.0, 0, 0], [0.0, 0, 0]])
    targets = np.array([12, 11, 10])
    target_xyz = np.array([[12.0, 0, 0], [0, 1.0, 0], [1.0, 0, 0]])
    triples, distances = enumerate_triplets(sources, source_xyz, targets, target_xyz, 3.0, 3.0, 10)
    reversed_triples, _ = enumerate_triplets(
        sources, source_xyz, targets[::-1], target_xyz[::-1], 3.0, 3.0, 10
    )
    assert set(map(tuple, triples)) == {(5, 10, 11)}
    assert set(map(tuple, reversed_triples)) == set(map(tuple, triples))
    assert len(select(triples, distances, 0.1)) == len(select(triples, -distances, 0.1)) == 1
    with pytest.raises(ValueError, match="guard"):
        enumerate_triplets(sources, source_xyz, targets, target_xyz, 3.0, 3.0, 0)


def test_ties_are_determined_by_ids_and_budget_is_exact():
    select = FUNCTIONS["budget_indices"]
    triples = np.array([[9, 12, 13], [2, 5, 6], [9, 10, 11], [2, 5, 7]])
    order = select(triples, np.ones(4), 0.5)
    assert triples[order].tolist() == [[2, 5, 6], [2, 5, 7]]
    assert len(select(triples, np.arange(4.0), 0.25)) == 1
    with pytest.raises(ValueError, match="nonfinite"):
        select(triples, np.array([1.0, 2.0, np.nan, 4.0]), 0.5)


def test_historyless_mother_uses_only_image_cost(monkeypatch):
    class Tensor:
        def __init__(self, array):
            self.array = np.asarray(array)

        def __getitem__(self, index):
            return Tensor(self.array[index])

        def float(self):
            return self

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.array

    def softmax(value, dim):
        raw = np.exp(value.array - value.array.max(axis=dim, keepdims=True))
        return Tensor(raw / raw.sum(axis=dim, keepdims=True))

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(softmax=softmax))
    costs = FUNCTIONS["triplet_costs"]
    state = FUNCTIONS["State"]
    arrays = {
        "candidate_ids_src": np.array([1, 2]),
        "candidate_ids_tgt": np.array([3, 4]),
        "coords_tgt_physical": np.array([[1.0, 0, 0], [2.0, 0, 0]]),
    }
    triples = np.array([[1, 3, 4], [2, 3, 4]])
    saved = {
        1: state(1, np.zeros(6), np.eye(6), 1),
        2: state(2, np.zeros(6), np.eye(6), 3),
    }
    noise = {
        "acceleration_variance": [1.0, 1.0, 1.0],
        "observation_variance": [1.0, 1.0, 1.0],
        "nll_center": 0.0,
        "nll_scale": 1.0,
    }
    image, motion, fallback = costs(arrays, triples, Tensor(np.zeros((1, 2, 2))), saved, noise, 0.5)
    assert fallback == 1
    assert np.allclose(image, 2 * np.log(2))
    assert motion[0] == 0
    assert np.isfinite(motion[1]) and motion[1] > 0
    with pytest.raises(ValueError, match="miss"):
        costs(arrays, triples, Tensor(np.zeros((1, 2, 2))), {1: saved[1]}, noise, 0.5)


def test_budget_uses_only_training_embryo():
    choose = FUNCTIONS["choose_training_budget"]
    rows = []
    for embryo, recovered in (("44b6", (4, 8, 9, 10)), ("6bba", (1, 2, 3, 10))):
        for fraction, count in zip((0.1, 0.25, 0.5, 1.0), recovered, strict=True):
            rows.append(
                {"embryo": embryo, "method": "distance", "fraction": fraction, "recovered": count}
            )
    assert choose(rows, [0.1, 0.25, 0.5, 1.0], 0.9, "44b6") == 0.5
    assert choose(rows, [0.1, 0.25, 0.5, 1.0], 0.9, "6bba") == 1.0
