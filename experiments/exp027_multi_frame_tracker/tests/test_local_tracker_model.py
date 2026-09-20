from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))
from local_tracker_model import LocalThreeFrameTracker  # noqa: E402


def model_batch(previous_position: float) -> dict[str, torch.Tensor]:
    torch.manual_seed(7)
    return {
        "features_prev": torch.randn(1, 1, 64),
        "features_src": torch.randn(1, 2, 64),
        "features_tgt": torch.randn(1, 2, 64),
        "coords_prev_physical": torch.tensor([[[previous_position, 0.0, 0.0]]]),
        "coords_src_physical": torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]),
        "coords_tgt_physical": torch.tensor([[[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]]),
        "coords_src": torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]),
        "coords_tgt": torch.tensor([[[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]]),
        "prev_mask": torch.ones(1, 1, dtype=torch.bool),
        "source_mask": torch.ones(1, 2, dtype=torch.bool),
        "target_mask": torch.ones(1, 2, dtype=torch.bool),
    }


def test_local_attention_respects_previous_mask_and_radius() -> None:
    torch.manual_seed(8)
    model = LocalThreeFrameTracker(
        feat_dim=64,
        hidden_dim=16,
        n_heads=4,
        n_blocks=2,
        dropout=0.0,
        pair_chunk_size=2,
        attention_radius_um=15.0,
        max_neighbors=5,
    ).eval()
    near = model_batch(0.5)
    far = dict(near)
    far["coords_prev_physical"] = torch.tensor([[[1000.0, 0.0, 0.0]]])
    with torch.no_grad():
        masked = model(near, include_previous=False)
        masked_far = model(far, include_previous=False)
        observed = model(near, include_previous=True)
        ignored = model(far, include_previous=True)
        reverse = model(near, include_previous=True, reverse=True)
    torch.testing.assert_close(masked, masked_far)
    torch.testing.assert_close(masked, ignored)
    assert not torch.allclose(observed, masked)
    assert observed.shape == (1, 2, 2)
    assert reverse.shape == (1, 2, 2)
    assert torch.isfinite(reverse).all()
