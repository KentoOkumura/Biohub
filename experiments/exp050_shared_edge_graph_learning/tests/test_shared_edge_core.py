import numpy as np
import pytest
import torch

from experiments.exp047_x138_edge_candidates.candidate_policy import select_candidate_edges
from experiments.exp050_shared_edge_graph_learning.shared_edge_core import (
    KNOWN_NO_PARENT,
    SharedEdgeGNN,
    enumerate_sets,
    partial_teacher_loss,
    select_expanded_edges,
    solve_set_ilp,
)


def example():
    coords = np.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0]],
        dtype=np.float32,
    )
    edges = np.array([[0, 2, 0.9, 0], [0, 3, 0.8, 1], [1, 3, 0.2, 0]], dtype=np.float64)
    catalog, by_parent = enumerate_sets(coords, edges)
    return coords, edges, catalog, by_parent


def test_candidate_rule_matches_exp047_on_fixed_scores():
    coords = np.array([[0, i, 0, 0] for i in range(5)] + [[1, 0, 0, 0]], dtype=np.float32)
    src = np.arange(5)
    tgt = np.full(5, 5)
    prob = np.array([0.11, 0.20, 0.20, 0.12, 0.49])
    admitted = np.array([[4, 5, 0.49, 4.0]])
    expected = select_candidate_edges(coords, src, tgt, prob, admitted).expanded
    actual = select_expanded_edges(coords, src, tgt, prob, admitted)
    assert np.array_equal(actual, expected)


def test_model_shares_edge_score_and_pair_is_unordered_at_initialization():
    coords, edges, catalog, _ = example()
    model = SharedEdgeGNN(feature_width=3)
    model.eval()
    feature = torch.arange(12, dtype=torch.float32).reshape(4, 3) / 10
    p, penalty, scores = model(
        feature, torch.from_numpy(coords), torch.tensor(edges, dtype=torch.float32), catalog
    )
    assert torch.allclose(p, torch.tensor(edges[:, 2], dtype=torch.float32), atol=1e-6)
    assert torch.allclose(penalty, torch.tensor([1.2]), atol=1e-6)
    pair_index = next(i for i, option in enumerate(catalog) if option.daughters == (2, 3))
    assert torch.allclose(scores[pair_index], p[0] + p[1] - penalty[0])


def test_partial_teacher_preserves_unknown_second_daughter_and_trains_known_pair():
    _, edges, catalog, by_parent = example()
    probability = torch.tensor(edges[:, 2], requires_grad=True, dtype=torch.float32)
    scores = []
    for option in catalog:
        if not option.edge_ids:
            scores.append(probability.sum() * 0)
        elif len(option.edge_ids) == 1:
            scores.append(probability[option.edge_ids[0]])
        else:
            scores.append(probability[list(option.edge_ids)].sum() - 0.2)
    scores = torch.stack(scores)
    one_loss, one_stats = partial_teacher_loss(
        scores, probability, catalog, by_parent, edges, {2: 0, 3: KNOWN_NO_PARENT}
    )
    assert one_stats["known_divisions_with_candidates"] == 0
    assert one_stats["informative_mothers"] > 0
    assert torch.isfinite(one_loss)
    two_loss, two_stats = partial_teacher_loss(
        scores, probability, catalog, by_parent, edges, {2: 0, 3: 0}
    )
    assert two_stats["known_divisions_with_candidates"] == 1
    assert two_stats["division_terms"] == 1
    two_loss.backward()
    assert probability.grad is not None


def test_unannotated_second_daughter_is_not_a_negative_label():
    _, edges, catalog, by_parent = example()
    pair_index = next(i for i, option in enumerate(catalog) if option.daughters == (2, 3))
    original = torch.zeros(len(catalog))
    preferred_pair = original.clone()
    preferred_pair[pair_index] = 5.0
    edge_probability = torch.tensor(edges[:, 2], dtype=torch.float32)
    unknown_low, _ = partial_teacher_loss(
        original, edge_probability, catalog, by_parent, edges, {2: 0}
    )
    unknown_high, _ = partial_teacher_loss(
        preferred_pair, edge_probability, catalog, by_parent, edges, {2: 0}
    )
    known_absent_low, _ = partial_teacher_loss(
        original, edge_probability, catalog, by_parent, edges, {2: 0, 3: KNOWN_NO_PARENT}
    )
    known_absent_high, _ = partial_teacher_loss(
        preferred_pair, edge_probability, catalog, by_parent, edges, {2: 0, 3: KNOWN_NO_PARENT}
    )
    assert unknown_high < unknown_low
    assert known_absent_high > known_absent_low


def test_set_ilp_selects_pair_only_once_and_no_double_division_cost():
    coords, edges, catalog, by_parent = example()
    scores = np.array(
        [
            0
            if not option.edge_ids
            else sum(edges[e, 2] for e in option.edge_ids)
            - (0.1 if len(option.edge_ids) == 2 else 0)
            for option in catalog
        ],
        dtype=np.float64,
    )
    selected, info = solve_set_ilp(coords, edges, catalog, by_parent, scores, disappearance_cost=0)
    assert {(int(row[0]), int(row[1])) for row in selected} == {(0, 2), (0, 3)}
    assert info["selected_divisions"] == 1
    assert info["optimal"] is True


def test_duplicate_edge_is_rejected():
    coords, edges, _, _ = example()
    with pytest.raises(ValueError, match="duplicate"):
        enumerate_sets(coords, np.concatenate([edges, edges[:1]]))


def test_window_scores_align_to_global_sets_and_keep_original_ids():
    from experiments.exp050_shared_edge_graph_learning.shared_edge_core import (
        score_video_sets,
        split_windows,
    )

    coords = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0]], dtype=np.float32)
    features = np.zeros((3, 2), dtype=np.float32)
    edges = np.array([[0, 1, 0.8, 0], [1, 2, 0.7, 0]], dtype=np.float64)
    windows = split_windows(coords, features, edges)
    assert [window.time for window in windows] == [0, 1]
    model = SharedEdgeGNN(feature_width=2)
    model.eval()
    catalog, by_parent, scores, probability = score_video_sets(model, coords, features, edges)
    assert np.allclose(probability, edges[:, 2], atol=1e-6)
    assert np.allclose(
        [scores[i] for i, option in enumerate(catalog) if len(option.edge_ids) == 1],
        edges[:, 2],
        atol=1e-6,
    )
    selected, info = solve_set_ilp(coords, edges, catalog, by_parent, scores, disappearance_cost=0)
    assert len(selected) == 2
    assert info["selected_node_ids"] == [0, 1, 2]
