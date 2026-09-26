from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

EXPERIMENT = Path(__file__).resolve().parents[1]
REPO = EXPERIMENT.parents[1]
PUBLIC_SOURCE = REPO / "experiments/exp004_embryo_holdout_baseline/official_source/src"
sys.path.insert(0, str(EXPERIMENT))
sys.path.insert(0, str(PUBLIC_SOURCE))

attention_module = importlib.import_module("primary_pair_attention")
PrimaryPairFeatureAttention = attention_module.PrimaryPairFeatureAttention
select_history = attention_module.select_history
SimpleNodeTransformer = importlib.import_module(
    "tracking_cellmot.models.simple_node_transformer"
).SimpleNodeTransformer
capture_module = importlib.import_module("x138_data")
aligned_previous_features = capture_module.aligned_previous_features
read_capture = capture_module.read_capture


def _base() -> SimpleNodeTransformer:
    return SimpleNodeTransformer(
        feat_dim=64,
        hidden_dim=128,
        n_heads=4,
        n_blocks=1,
        dropout=0.0,
        pair_chunk_size=32,
    )


def test_initial_forward_reverse_parity_and_no_history() -> None:
    torch.manual_seed(7)
    base = _base().eval()
    model = PrimaryPairFeatureAttention(base, source_chunk_size=2, target_chunk_size=2).eval()
    parent = torch.randn(1, 3, 64)
    child = torch.randn(1, 2, 64)
    parent_coords = torch.randn(1, 3, 3)
    child_coords = torch.randn(1, 2, 3)
    prev_coords = torch.randn(1, 4, 3)
    kwargs = {
        "features_previous": torch.randn(1, 4, 64),
        "coordinates_previous_physical": prev_coords,
        "ids_previous": torch.tensor([[10, 11, 12, 13]]),
        "mask_previous": torch.ones(1, 4, dtype=torch.bool),
        "coordinates_parent_physical": parent_coords,
        "coordinates_child_physical": child_coords,
    }
    with torch.no_grad():
        forward, reverse = model(parent, child, parent_coords, child_coords, **kwargs)
        expected_forward = base(parent, child, parent_coords, child_coords)
        expected_reverse = base(child, parent, child_coords, parent_coords)
    torch.testing.assert_close(forward, expected_forward, atol=1e-6, rtol=0)
    torch.testing.assert_close(reverse, expected_reverse, atol=1e-6, rtol=0)

    with torch.no_grad():
        model.pair_adapter.weight.fill_(0.01)
        used_forward, used_reverse = model(parent, child, parent_coords, child_coords, **kwargs)
        kwargs["mask_previous"] = torch.zeros_like(kwargs["mask_previous"])
        masked_forward, masked_reverse = model(parent, child, parent_coords, child_coords, **kwargs)
    torch.testing.assert_close(masked_forward, expected_forward, atol=1e-6, rtol=0)
    torch.testing.assert_close(masked_reverse, expected_reverse, atol=1e-6, rtol=0)
    assert not torch.allclose(used_forward, masked_forward)
    assert not torch.allclose(used_reverse, masked_reverse)
    kwargs["mask_previous"] = torch.ones_like(kwargs["mask_previous"])
    inspected = model.inspect_pair(
        parent,
        child,
        parent_coords,
        child_coords,
        **kwargs,
        parent_index=0,
        child_index=0,
    )
    for direction in ("forward", "reverse"):
        row = inspected[direction]
        assert len(row["candidate_ids"]) == 4
        assert len(row["mean_head_attention_weight"]) == 4
        assert 0 <= row["null_attention_weight"] <= 1
        assert row["adapter_residual_norm"] > 0


def test_history_selection_uses_physical_distance_and_id_ties() -> None:
    features = torch.zeros(1, 3, 64)
    features[0, :, 0] = torch.tensor([1.0, 2.0, 3.0])
    coordinates = torch.tensor([[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [5.0, 0.0, 0.0]]])
    ids = torch.tensor([[20, 10, 30]])
    selected = select_history(
        features,
        coordinates,
        ids,
        torch.tensor([[True, True, False]]),
        torch.zeros(1, 1, 3),
        3,
    )
    assert selected.candidate_ids[0, 0].tolist() == [10, 20, 0]
    assert selected.mask[0, 0].tolist() == [True, True, False]
    assert selected.features[0, 0, 0, 0] == 2
    assert selected.features[0, 0, 2].count_nonzero() == 0
    padded = select_history(
        features,
        coordinates,
        ids,
        torch.tensor([[True, True, False]]),
        torch.zeros(1, 1, 3),
        5,
    )
    assert padded.mask[0, 0].tolist() == [True, True, False, False, False]
    assert padded.features.shape == (1, 1, 5, 64)
    permutation = torch.tensor([2, 0, 1])
    reordered = select_history(
        features[:, permutation],
        coordinates[:, permutation],
        ids[:, permutation],
        torch.tensor([[False, True, True]]),
        torch.zeros(1, 1, 3),
        5,
    )
    torch.testing.assert_close(reordered.features, padded.features)
    torch.testing.assert_close(reordered.candidate_ids, padded.candidate_ids)
    torch.testing.assert_close(reordered.mask, padded.mask)


def _capture(
    path: Path,
    frame: int,
    ids_src: list[int],
    coords_src: np.ndarray,
    ids_prev: list[int] | None = None,
    coords_prev: np.ndarray | None = None,
) -> None:
    n_src = len(ids_src)
    n_prev = len(ids_prev or [])
    if coords_prev is None:
        coords_prev = np.empty((0, 3), dtype=np.float32)
    arrays = {
        "window_frames": np.array([frame, frame + 1], dtype=np.int32),
        "candidate_ids_prev": np.array(ids_prev or [], dtype=np.int64),
        "candidate_ids_src": np.array(ids_src, dtype=np.int64),
        "candidate_ids_tgt": np.array([99], dtype=np.int64),
        "coords_prev_grid": np.asarray(coords_prev, dtype=np.float32).reshape(n_prev, 3),
        "coords_src_grid": np.asarray(coords_src, dtype=np.float32).reshape(n_src, 3),
        "coords_tgt_grid": np.zeros((1, 3), dtype=np.float32),
        "primary_features_src": np.arange(n_src * 32, dtype=np.float32).reshape(n_src, 32),
        "primary_features_tgt": np.zeros((1, 32), dtype=np.float32),
        "position_features_src": np.ones((n_src, 32), dtype=np.float32),
        "position_features_tgt": np.zeros((1, 32), dtype=np.float32),
        "primary_forward_logits": np.zeros((n_src, 1), dtype=np.float32),
        "primary_reverse_logits": np.zeros((1, n_src), dtype=np.float32),
        "secondary_logits": np.zeros((n_src, 1), dtype=np.float32),
        "x138_fused_logits": np.zeros((n_src, 1), dtype=np.float32),
    }
    np.savez(path, **arrays)


def test_previous_window_features_align_by_candidate_id(tmp_path: Path) -> None:
    prior = tmp_path / "000000_000001.npz"
    current = tmp_path / "000001_000002.npz"
    _capture(prior, 0, [7, 4], np.array([[1, 0, 0], [2, 0, 0]]))
    _capture(
        current,
        1,
        [20],
        np.array([[3, 0, 0]]),
        [4, 7],
        np.array([[2, 0, 0], [1, 0, 0]]),
    )
    aligned = aligned_previous_features(current, read_capture(current))
    assert aligned.shape == (2, 64)
    assert aligned[0, 0] == 32
    assert aligned[1, 0] == 0
    assert np.all(aligned[:, 32:] == 1)
    prior.unlink()
    with pytest.raises(FileNotFoundError, match="missing previous feature window"):
        aligned_previous_features(current, read_capture(current))


def test_joint_training_updates_primary_and_history_attention() -> None:
    torch.manual_seed(19)
    model = PrimaryPairFeatureAttention(_base(), source_chunk_size=2, target_chunk_size=2).train()
    parent = torch.randn(1, 3, 64)
    child = torch.randn(1, 2, 64)
    parent_coords = torch.randn(1, 3, 3)
    child_coords = torch.randn(1, 2, 3)
    kwargs = {
        "features_previous": torch.randn(1, 3, 64),
        "coordinates_previous_physical": torch.randn(1, 3, 3),
        "ids_previous": torch.tensor([[5, 6, 7]]),
        "mask_previous": torch.ones(1, 3, dtype=torch.bool),
        "coordinates_parent_physical": parent_coords,
        "coordinates_child_physical": child_coords,
    }
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    first, reverse = model(parent, child, parent_coords, child_coords, **kwargs)
    (first.square().mean() + reverse.square().mean()).backward()
    assert model.base_tracker.pair_mlp[0].weight.grad is not None
    assert model.base_tracker.pair_mlp[0].weight.grad.abs().sum() > 0
    assert model.pair_adapter.weight.grad is not None
    assert model.pair_adapter.weight.grad.abs().sum() > 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    second, reverse = model(parent, child, parent_coords, child_coords, **kwargs)
    (second.square().mean() + reverse.square().mean()).backward()
    assert model.query_projection.weight.grad is not None
    assert model.query_projection.weight.grad.abs().sum() > 0
    assert model.key_projection.weight.grad is not None
    assert model.key_projection.weight.grad.abs().sum() > 0
    assert model.value_projection.weight.grad is not None
    assert model.value_projection.weight.grad.abs().sum() > 0


def test_chunk_sizes_preserve_logits_and_gradients() -> None:
    torch.manual_seed(23)
    small = PrimaryPairFeatureAttention(_base(), source_chunk_size=2, target_chunk_size=2).train()
    large = PrimaryPairFeatureAttention(_base(), source_chunk_size=8, target_chunk_size=8).train()
    with torch.no_grad():
        small.pair_adapter.weight.normal_(mean=0, std=0.01)
    large.load_state_dict(small.state_dict())
    parent = torch.randn(1, 3, 64)
    child = torch.randn(1, 2, 64)
    parent_coords = torch.randn(1, 3, 3)
    child_coords = torch.randn(1, 2, 3)
    kwargs = {
        "features_previous": torch.randn(1, 4, 64),
        "coordinates_previous_physical": torch.randn(1, 4, 3),
        "ids_previous": torch.tensor([[9, 7, 8, 6]]),
        "mask_previous": torch.tensor([[True, True, True, False]]),
        "coordinates_parent_physical": parent_coords,
        "coordinates_child_physical": child_coords,
    }
    outputs = []
    for model in (small, large):
        forward, reverse = model(parent, child, parent_coords, child_coords, **kwargs)
        (forward.square().mean() + reverse.square().mean()).backward()
        outputs.append((forward.detach(), reverse.detach()))
    torch.testing.assert_close(outputs[0][0], outputs[1][0], atol=2e-6, rtol=0)
    torch.testing.assert_close(outputs[0][1], outputs[1][1], atol=2e-6, rtol=0)
    for key in (
        "base_tracker.pair_mlp.0.weight",
        "pair_adapter.weight",
        "query_projection.weight",
        "key_projection.weight",
        "value_projection.weight",
    ):
        left = dict(small.named_parameters())[key].grad
        right = dict(large.named_parameters())[key].grad
        torch.testing.assert_close(left, right, atol=2e-6, rtol=1e-5)


def test_runtime_forecast_uses_size_buckets_instead_of_all_largest_pairs() -> None:
    forecast = importlib.import_module("attention_train_pipeline")._forecast_bucket_specs
    paths = [Path(f"{number:03d}.npz") for number in range(1, 101)]
    workload = {path: number for number, path in enumerate(paths, start=1)}
    buckets = forecast(
        workload,
        paths,
        paths[-1],
        (0.0, 0.5, 0.75, 0.9, 0.99, 1.0),
    )
    assert [row["fit_batches"] for row in buckets] == [50, 25, 15, 9, 1]
    assert [row["maximum_workload"] for row in buckets] == [50, 75, 90, 99, 100]
    assert buckets[-1]["representative_path"] == str(paths[-1])
