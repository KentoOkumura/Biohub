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
    select_past_candidates,
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
        "candidate_ids_prev": torch.tensor([[8, 2, 7, 3]]),
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
    batch["candidate_ids_prev"] = torch.zeros(1, 0, dtype=torch.int64)
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
    permuted["candidate_ids_prev"] = batch["candidate_ids_prev"][:, order]
    torch.testing.assert_close(tracker(**permuted), reference)
    padded = dict(batch)
    padded["coords_prev_physical"] = torch.cat(
        [batch["coords_prev_physical"], torch.tensor([[[999.0, 999.0, 999.0]]])], dim=1
    )
    padded["prev_mask"] = torch.cat([batch["prev_mask"], torch.tensor([[False]])], dim=1)
    padded["candidate_ids_prev"] = torch.cat(
        [batch["candidate_ids_prev"], torch.tensor([[99]])], dim=1
    )
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


def test_knn_ties_padding_and_source_specific_selection() -> None:
    previous = torch.tensor([[[1.0, 0, 0], [-1.0, 0, 0], [99.0, 0, 0], [0.0, 0, 0]]])
    source = torch.tensor([[[0.0, 0, 0], [100.0, 0, 0]]])
    ids = torch.tensor([[9, 2, 3, 0]])
    valid = torch.tensor([[True, True, True, False]])
    selected, mask, selected_ids = select_past_candidates(previous, source, ids, valid, 2)
    assert selected_ids.tolist() == [[[2, 9], [3, 9]]]
    assert mask.all() and selected.shape == (1, 2, 2, 3)
    order = torch.tensor([3, 1, 0, 2])
    permuted = select_past_candidates(previous[:, order], source, ids[:, order], valid[:, order], 2)
    torch.testing.assert_close(selected_ids, permuted[2])


@pytest.mark.parametrize("count", [4, 12])
def test_dense_selected_reference_outputs_and_gradients(count: int) -> None:
    torch.manual_seed(61)
    model = new_tracker(past_chunk=2, gradient_checkpointing=True)
    nn.init.normal_(model.delta_output.weight)
    reference = copy.deepcopy(model)
    batch = inputs()
    batch["coords_prev_physical"] = torch.randn(1, count, 3)
    batch["prev_mask"] = torch.ones(1, count, dtype=torch.bool)
    batch["candidate_ids_prev"] = torch.arange(count)[None]
    output = model(**batch)
    features = build_past_candidate_features(
        coords_prev_physical=batch["coords_prev_physical"],
        coords_src_physical=batch["coords_src_physical"],
        coords_tgt_physical=batch["coords_tgt_physical"],
        vector_scale_um=5,
        cosine_epsilon=1e-8,
    )
    hidden = reference.candidate_mlp(features)
    scores = reference.attention_score(hidden).squeeze(-1)
    _, valid, selected_ids = select_past_candidates(
        batch["coords_prev_physical"],
        batch["coords_src_physical"],
        batch["candidate_ids_prev"],
        batch["prev_mask"],
        8,
    )
    selected = (torch.arange(count)[None, None, :, None] == selected_ids[:, :, None]).any(-1)
    scores = scores.masked_fill(~selected[:, :, None], float("-inf"))
    no_past = reference.no_past_score.expand(*scores.shape[:-1], 1)
    weights = torch.softmax(torch.cat([scores, no_past], dim=-1), dim=-1)[..., :-1]
    delta = reference.delta_output((hidden * weights[..., None]).sum(-2)).squeeze(-1)
    expected = (
        reference.base_tracker(
            batch["feat_t"], batch["feat_t1"], batch["coords_t"], batch["coords_t1"]
        )
        + delta
    )
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)
    output.sum().backward()
    expected.sum().backward()
    for p, q in zip(model.parameters(), reference.parameters(), strict=True):
        assert torch.isfinite(p.grad).all()
        torch.testing.assert_close(p.grad, q.grad, rtol=3e-5, atol=3e-6)


def test_all_invalid_past_has_zero_delta_and_finite_gradients() -> None:
    model = new_tracker(past_chunk=2, gradient_checkpointing=True)
    nn.init.normal_(model.delta_output.weight)
    batch = inputs()
    batch["prev_mask"].fill_(False)
    batch["coords_prev_physical"].fill_(float("nan"))
    result = model(**batch)
    base = model.base_tracker(
        batch["feat_t"], batch["feat_t1"], batch["coords_t"], batch["coords_t1"]
    )
    torch.testing.assert_close(result, base, rtol=0, atol=0)
    result.sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
