from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))

from past_candidate_attention import (  # noqa: E402
    PAST_CANDIDATE_FEATURE_NAMES,
    PastCandidateAttentionTracker,
    build_past_candidate_features,
)


class DummyBaseTracker(nn.Module):
    def forward(
        self,
        feat_t: torch.Tensor,
        feat_t1: torch.Tensor,
        coords_t: torch.Tensor,
        coords_t1: torch.Tensor,
        mask_t: torch.Tensor | None = None,
        mask_t1: torch.Tensor | None = None,
    ) -> torch.Tensor:
        unbatched = coords_t.ndim == 2
        if unbatched:
            coords_t = coords_t.unsqueeze(0)
            coords_t1 = coords_t1.unsqueeze(0)
        logits = coords_t[..., 0].unsqueeze(-1) - coords_t1[..., 0].unsqueeze(1)
        return logits.squeeze(0) if unbatched else logits


def new_tracker(
    *,
    past_chunk: int,
    target_chunk: int = 2,
    gradient_checkpointing: bool = False,
) -> PastCandidateAttentionTracker:
    return PastCandidateAttentionTracker(
        DummyBaseTracker(),
        feature_dim=13,
        hidden_dim=32,
        source_chunk_size=2,
        target_chunk_size=target_chunk,
        past_candidate_chunk_size=past_chunk,
        gradient_checkpointing=gradient_checkpointing,
        vector_scale_um=5.0,
        cosine_epsilon=1e-8,
    )


def inputs() -> dict[str, torch.Tensor]:
    return {
        "feat_t": torch.zeros(1, 3, 2),
        "feat_t1": torch.zeros(1, 2, 2),
        "coords_t": torch.tensor([[[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]]]),
        "coords_t1": torch.tensor([[[0.5, 0, 0], [1.5, 0, 0]]]),
        "mask_t": torch.tensor([[True, True, True]]),
        "mask_t1": torch.tensor([[True, True]]),
        "coords_prev_physical": torch.tensor(
            [[[0.0, 0, 0], [0.0, 5, 0], [1.0, 0, 0], [2.0, 0, 0]]]
        ),
        "coords_src_physical": torch.tensor([[[0.0, 5, 0], [1.0, 5, 0], [2.0, 5, 0]]]),
        "coords_tgt_physical": torch.tensor([[[0.0, 10, 0], [2.0, 10, 0]]]),
        "prev_mask": torch.tensor([[True, True, True, True]]),
    }


def test_feature_contract_uses_three_displacements_lengths_and_cosine() -> None:
    features = build_past_candidate_features(
        coords_prev_physical=torch.tensor([[[0.0, 0.0, 0.0]]]),
        coords_src_physical=torch.tensor([[[0.0, 5.0, 0.0]]]),
        coords_tgt_physical=torch.tensor([[[0.0, 10.0, 0.0]]]),
        vector_scale_um=5.0,
        cosine_epsilon=1e-8,
    )
    assert features.shape == (1, 1, 1, 1, len(PAST_CANDIDATE_FEATURE_NAMES))
    assert features.flatten().tolist() == pytest.approx([0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1])
    changed_target = build_past_candidate_features(
        coords_prev_physical=torch.tensor([[[0.0, 0.0, 0.0]]]),
        coords_src_physical=torch.tensor([[[0.0, 5.0, 0.0]]]),
        coords_tgt_physical=torch.tensor([[[5.0, 5.0, 0.0]]]),
        vector_scale_um=5.0,
        cosine_epsilon=1e-8,
    )
    assert not torch.equal(features, changed_target)


def test_zero_motion_cosine_is_zero_and_features_are_not_clipped() -> None:
    features = build_past_candidate_features(
        coords_prev_physical=torch.tensor([[[0.0, 0.0, 0.0]]]),
        coords_src_physical=torch.tensor([[[0.0, 0.0, 0.0]]]),
        coords_tgt_physical=torch.tensor([[[100.0, 0.0, 0.0]]]),
        vector_scale_um=5.0,
        cosine_epsilon=1e-8,
    )
    assert features[..., -1].item() == 0.0
    assert features[..., 3].item() == 20.0


def test_added_parameter_count_and_zero_initialized_output_match_base() -> None:
    tracker = new_tracker(past_chunk=2)
    assert sum(parameter.numel() for parameter in tracker.parameters()) == 1570
    batch = inputs()
    output = tracker(**batch)
    base = tracker.base_tracker(
        batch["feat_t"],
        batch["feat_t1"],
        batch["coords_t"],
        batch["coords_t1"],
        batch["mask_t"],
        batch["mask_t1"],
    )
    torch.testing.assert_close(output, base, rtol=0, atol=0)


def test_empty_past_set_keeps_zero_delta_after_training_changes() -> None:
    tracker = new_tracker(past_chunk=2)
    nn.init.normal_(tracker.delta_output.weight)
    batch = inputs()
    batch["coords_prev_physical"] = torch.zeros(1, 0, 3)
    batch["prev_mask"] = torch.zeros(1, 0, dtype=torch.bool)
    output = tracker(**batch)
    base = tracker.base_tracker(
        batch["feat_t"],
        batch["feat_t1"],
        batch["coords_t"],
        batch["coords_t1"],
        batch["mask_t"],
        batch["mask_t1"],
    )
    torch.testing.assert_close(output, base, rtol=0, atol=0)


def test_candidate_order_and_padding_do_not_change_output() -> None:
    torch.manual_seed(7)
    tracker = new_tracker(past_chunk=2)
    nn.init.normal_(tracker.delta_output.weight)
    batch = inputs()
    reference = tracker(**batch)
    order = torch.tensor([2, 0, 3, 1])
    permuted = dict(batch)
    permuted["coords_prev_physical"] = batch["coords_prev_physical"][:, order]
    permuted["prev_mask"] = batch["prev_mask"][:, order]
    torch.testing.assert_close(tracker(**permuted), reference)
    padded = dict(batch)
    padded["coords_prev_physical"] = torch.cat(
        [batch["coords_prev_physical"], torch.tensor([[[999.0, 999.0, 999.0]]])], dim=1
    )
    padded["prev_mask"] = torch.cat([batch["prev_mask"], torch.tensor([[False]])], dim=1)
    torch.testing.assert_close(tracker(**padded), reference)


def test_chunked_and_unchunked_outputs_and_gradients_match() -> None:
    torch.manual_seed(11)
    chunked = new_tracker(past_chunk=2, target_chunk=1, gradient_checkpointing=True)
    nn.init.normal_(chunked.delta_output.weight)
    full = new_tracker(past_chunk=64, target_chunk=64)
    full.load_state_dict(copy.deepcopy(chunked.state_dict()))
    batch = inputs()
    chunked_output = chunked(**batch)
    full_output = full(**batch)
    torch.testing.assert_close(chunked_output, full_output, rtol=1e-5, atol=1e-6)
    chunked_output.sum().backward()
    full_output.sum().backward()
    for (chunked_name, chunked_parameter), (full_name, full_parameter) in zip(
        chunked.named_parameters(), full.named_parameters(), strict=True
    ):
        assert chunked_name == full_name
        torch.testing.assert_close(
            chunked_parameter.grad,
            full_parameter.grad,
            rtol=2e-5,
            atol=2e-6,
        )
