from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def diagnostic(monkeypatch: pytest.MonkeyPatch):
    source = Path(__file__).resolve().parents[1] / "diagnose_divisions.py"
    monkeypatch.syspath_prepend(str(source.parent))
    spec = importlib.util.spec_from_file_location("exp029_division_diagnostic", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dominance_requires_a_subset_not_any_better_singleton(diagnostic) -> None:
    options = [(), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2)]
    scores = np.array([0.0, 1.0, 1.0, 10.0, 2.0, 0.0, 0.0])
    row = diagnostic.inspect_parent(scores, options, [0, 1], np.array([10, 20, 30]), 0.0, 16)
    assert not row["true_subset_dominated"]
    assert row["true_minus_best_singleton"] == -8.0
    assert row["true_minus_best_own_subset"] == 1.0
    scores[1] = 3.0
    row = diagnostic.inspect_parent(scores, options, [0, 1], np.array([10, 20, 30]), 0.0, 16)
    assert row["true_subset_dominated"]
    assert row["true_minus_best_own_subset"] == -1.0


def test_keep_budget_includes_empty_and_rank_is_one_based(diagnostic) -> None:
    options = [(), (0,), (1,), (0, 1)]
    scores = np.array([0.0, 10.0, 1.0, 5.0])
    args = (scores, options, [0, 1], np.array([10, 20]), 0.0)
    assert diagnostic.inspect_parent(*args, 3)["true_set_retained"]
    assert not diagnostic.inspect_parent(*args, 2)["true_set_retained"]
    assert diagnostic.inspect_parent(*args, 3)["true_set_rank_nonempty"] == 2


def test_missing_truth_has_no_rank_or_score_margin(diagnostic) -> None:
    row = diagnostic.inspect_parent(
        np.array([0.0, 1.0]), [(), (0,)], [0, 1], np.array([10, 20]), 0.0, 16
    )
    assert not row["candidate_covered"]
    assert row["true_set_rank_nonempty"] is None
    assert row["true_minus_best_own_subset"] is None


def test_known_two_daughters_send_gradient_to_pair_head(diagnostic) -> None:
    torch = pytest.importorskip("torch")
    model = diagnostic.TRAIN.make_model(
        {"input_feature_dim": 65, "hidden_dim": 8, "n_heads": 2, "n_layers": 1, "dropout": 0.0}
    )
    batch = {
        "features_src": torch.zeros((1, 1, 65)),
        "features_tgt": torch.zeros((1, 2, 65)),
        "coords_src": torch.zeros((1, 1, 3)),
        "coords_tgt": torch.zeros((1, 2, 3)),
        "source_mask": torch.ones((1, 1), dtype=torch.bool),
        "target_mask": torch.ones((1, 2), dtype=torch.bool),
        "set_children": torch.tensor([[[[-1, -1], [0, -1], [1, -1], [0, 1]]]]),
        "set_mask": torch.ones((1, 1, 4), dtype=torch.bool),
    }
    logits = model(batch)
    loss = diagnostic.TRAIN.partial_set_loss(
        logits, torch.tensor([[[False, False, False, True]]]), torch.tensor([[True]])
    )
    loss.backward()
    assert model.pair[-1].bias.grad.item() < 0
    assert model.singleton[-1].bias.grad.item() > 0
    assert model.pair[0].weight.grad.abs().sum().item() > 0
