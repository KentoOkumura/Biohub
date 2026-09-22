from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))

from velocity_tracker import (  # noqa: E402
    VELOCITY_PAIR_FEATURE_NAMES,
    VelocityAugmentedTracker,
    build_velocity_pair_features,
)


def test_velocity_pair_features_match_two_point_motion_contract() -> None:
    features = build_velocity_pair_features(
        velocity_src=torch.tensor([[[0.0, 5.0, 0.0]]]),
        history_present_src=torch.tensor([[True]]),
        history_length_src=torch.tensor([[4]]),
        history_confidence_src=torch.tensor([[0.8]]),
        coords_src_physical=torch.tensor([[[1.0, 2.0, 3.0]]]),
        coords_tgt_physical=torch.tensor([[[1.0, 7.0, 3.0], [6.0, 2.0, 3.0]]]),
        vector_scale=5.0,
        vector_clip_abs=4.0,
        history_length_cap=8,
    )
    assert features.shape == (1, 1, 2, len(VELOCITY_PAIR_FEATURE_NAMES))
    expected_first = torch.tensor(
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0.5, 0.8],
        dtype=torch.float32,
    )
    assert torch.allclose(features[0, 0, 0], expected_first, atol=1e-6)
    assert features[0, 0, 1, 11].item() == pytest.approx(0.0)


def test_missing_history_zeroes_every_velocity_pair_feature() -> None:
    features = build_velocity_pair_features(
        velocity_src=torch.tensor([[[5.0, 5.0, 5.0]]]),
        history_present_src=torch.tensor([[False]]),
        history_length_src=torch.tensor([[7]]),
        history_confidence_src=torch.tensor([[0.9]]),
        coords_src_physical=torch.zeros(1, 1, 3),
        coords_tgt_physical=torch.ones(1, 2, 3),
        vector_scale=5.0,
        vector_clip_abs=4.0,
        history_length_cap=8,
    )
    assert torch.count_nonzero(features).item() == 0


class DummyBase(nn.Module):
    def forward(
        self,
        feat_t: torch.Tensor,
        feat_t1: torch.Tensor,
        coords_t: torch.Tensor,
        coords_t1: torch.Tensor,
        mask_t: torch.Tensor | None = None,
        mask_t1: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return feat_t[..., :1] + feat_t1[..., :1].transpose(-1, -2)


def test_zero_initialized_velocity_branch_exactly_matches_base_logits() -> None:
    base = DummyBase()
    model = VelocityAugmentedTracker(
        base,
        pair_feature_dim=len(VELOCITY_PAIR_FEATURE_NAMES),
        branch_hidden_dim=8,
        branch_dropout=0.0,
        pair_chunk_size=1,
        vector_scale=5.0,
        vector_clip_abs=4.0,
        history_length_cap=8,
    )
    feat_src = torch.tensor([[1.0, 0.0], [2.0, 0.0]])
    feat_tgt = torch.tensor([[3.0, 0.0], [4.0, 0.0], [5.0, 0.0]])
    coords_src = torch.zeros(2, 3)
    coords_tgt = torch.zeros(3, 3)
    mask_src = torch.tensor([True, True])
    mask_tgt = torch.tensor([True, True, False])
    expected = base(feat_src, feat_tgt, coords_src, coords_tgt, mask_src, mask_tgt)
    actual = model(
        feat_src,
        feat_tgt,
        coords_src,
        coords_tgt,
        mask_src,
        mask_tgt,
        velocity_src=torch.ones(2, 3),
        history_present_src=torch.tensor([True, True]),
        history_length_src=torch.tensor([1, 2]),
        history_confidence_src=torch.tensor([0.7, 0.8]),
        coords_src_physical=torch.zeros(2, 3),
        coords_tgt_physical=torch.ones(3, 3),
    )
    assert torch.equal(actual, expected)
