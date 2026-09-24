from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP))
sys.path.insert(
    0,
    str(ROOT / "experiments/exp001_temporal_unet3d_baseline/official_source/src"),
)

from past_feature_attention import (  # noqa: E402
    PastFeatureAttention,
    PastFeatureCrossAttentionTracker,
)
from tracking_cellmot.models.simple_node_transformer import (  # noqa: E402
    SimpleNodeTransformer,
)


def config(**changes: object) -> dict[str, object]:
    result: dict[str, object] = {
        "hidden_dim": 128,
        "heads": 4,
        "position_time_mlp_hidden_dim": 32,
        "position_scale_um": 5.0,
        "query_chunk_size": 2,
        "key_chunk_size": 2,
        "gradient_checkpointing": False,
    }
    result.update(changes)
    return result


def attention_inputs() -> dict[str, torch.Tensor]:
    torch.manual_seed(3)
    return {
        "query": torch.randn(1, 3, 128),
        "past": torch.randn(1, 4, 128),
        "query_coords": torch.tensor([[[0.0, 0, 0], [1.0, 0, 0], [3.0, 0, 0]]]),
        "past_coords": torch.tensor([[[0.0, 0, 0], [5.0, 0, 0], [10.0, 0, 0], [100.0, 0, 0]]]),
        "query_frames": torch.full((1, 3), 5.0),
        "past_frames": torch.tensor([[2.0, 3.0, 4.0, 2.0]]),
        "past_mask": torch.tensor([[True, True, True, False]]),
        "query_mask": torch.tensor([[True, True, False]]),
    }


def activated_attention(**changes: object) -> PastFeatureAttention:
    module = PastFeatureAttention(**config(**changes))
    nn.init.normal_(module.output.weight, std=0.03)
    module.eval()
    return module


def test_chunked_softmax_matches_full_output_and_gradient() -> None:
    chunked = activated_attention(query_chunk_size=1, key_chunk_size=1)
    full = activated_attention(query_chunk_size=16, key_chunk_size=16)
    full.load_state_dict(copy.deepcopy(chunked.state_dict()))
    first = attention_inputs()
    second = {key: value.clone() for key, value in first.items()}
    first["query"].requires_grad_()
    first["past"].requires_grad_()
    second["query"].requires_grad_()
    second["past"].requires_grad_()
    result_chunked = chunked(**first)
    result_full = full(**second)
    torch.testing.assert_close(result_chunked, result_full, rtol=1e-5, atol=1e-5)
    result_chunked.square().sum().backward()
    result_full.square().sum().backward()
    for name in ("query", "past"):
        torch.testing.assert_close(first[name].grad, second[name].grad, rtol=1e-5, atol=1e-5)
    for (name_a, parameter_a), (name_b, parameter_b) in zip(
        chunked.named_parameters(), full.named_parameters(), strict=True
    ):
        assert name_a == name_b
        torch.testing.assert_close(parameter_a.grad, parameter_b.grad, rtol=1e-5, atol=1e-5)


def test_order_padding_empty_and_relative_time() -> None:
    module = activated_attention()
    inputs = attention_inputs()
    expected = module(**inputs)
    order = torch.tensor([2, 0, 3, 1])
    permuted = dict(inputs)
    for name in ("past", "past_coords", "past_frames", "past_mask"):
        permuted[name] = inputs[name][:, order]
    torch.testing.assert_close(module(**permuted), expected)
    assert torch.equal(expected[:, 2], inputs["query"][:, 2])
    masked = dict(inputs, past_mask=torch.zeros_like(inputs["past_mask"]))
    torch.testing.assert_close(module(**masked), inputs["query"], rtol=0, atol=0)
    changed = dict(inputs, past_frames=inputs["past_frames"] - 2)
    assert not torch.allclose(module(**changed), expected)
    changed_coords = dict(inputs, past_coords=inputs["past_coords"] + 2)
    assert not torch.allclose(module(**changed_coords), expected)
    assert torch.isfinite(expected).all()


def test_initial_logits_match_public_tracker_and_empty_past_after_training() -> None:
    torch.manual_seed(7)
    base = SimpleNodeTransformer(
        feat_dim=64,
        hidden_dim=128,
        n_heads=4,
        n_blocks=1,
        dropout=0.0,
        pair_chunk_size=2,
    )
    wrapper = PastFeatureCrossAttentionTracker(base, attention_config=config())
    assert sum(parameter.numel() for parameter in wrapper.past_attention.parameters()) == 65864
    wrapper.eval()
    base.eval()
    source = torch.randn(1, 3, 64)
    target = torch.randn(1, 2, 64)
    source_coords = torch.randn(1, 3, 3)
    target_coords = torch.randn(1, 2, 3)
    past = torch.randn(1, 4, 64)
    past_coords = torch.randn(1, 4, 3)
    mask_src = torch.ones(1, 3, dtype=torch.bool)
    mask_tgt = torch.ones(1, 2, dtype=torch.bool)
    kwargs = {
        "features_past": past,
        "coords_past_physical": past_coords,
        "frames_past": torch.tensor([[1.0, 2.0, 3.0, 3.0]]),
        "coords_src_physical": source_coords,
        "coords_tgt_physical": target_coords,
        "frames_src": torch.full((1, 3), 4.0),
        "frames_tgt": torch.full((1, 2), 5.0),
        "past_mask": torch.ones(1, 4, dtype=torch.bool),
    }
    with torch.no_grad():
        expected = base(source, target, source_coords, target_coords, mask_src, mask_tgt)
        initial = wrapper(
            source, target, source_coords, target_coords, mask_src, mask_tgt, **kwargs
        )
    torch.testing.assert_close(initial, expected, rtol=0, atol=0)
    nn.init.normal_(wrapper.past_attention.output.weight, std=0.03)
    with torch.no_grad():
        without_past = wrapper(
            source,
            target,
            source_coords,
            target_coords,
            mask_src,
            mask_tgt,
            **dict(kwargs, past_mask=torch.zeros_like(kwargs["past_mask"])),
        )
        same_weight_base = wrapper.base_tracker(
            source, target, source_coords, target_coords, mask_src, mask_tgt
        )
    torch.testing.assert_close(without_past, same_weight_base, rtol=0, atol=0)
    with torch.no_grad():
        reverse = wrapper.reverse_logits(
            source, target, source_coords, target_coords, mask_src, mask_tgt, **kwargs
        )
        manual_reverse = wrapper(
            target,
            source,
            target_coords,
            source_coords,
            mask_tgt,
            mask_src,
            **dict(
                kwargs,
                coords_src_physical=target_coords,
                coords_tgt_physical=source_coords,
                frames_src=kwargs["frames_tgt"],
                frames_tgt=kwargs["frames_src"],
            ),
        ).transpose(-2, -1)
    torch.testing.assert_close(reverse, manual_reverse, rtol=0, atol=0)


def test_single_window_logits_match_batched_path() -> None:
    torch.manual_seed(17)
    base = SimpleNodeTransformer(
        feat_dim=64,
        hidden_dim=128,
        n_heads=4,
        n_blocks=1,
        dropout=0.0,
        pair_chunk_size=2,
    )
    wrapper = PastFeatureCrossAttentionTracker(base, attention_config=config())
    nn.init.normal_(wrapper.past_attention.output.weight, std=0.03)
    wrapper.eval()
    source = torch.randn(3, 64)
    target = torch.randn(2, 64)
    source_coords = torch.randn(3, 3)
    target_coords = torch.randn(2, 3)
    past = torch.randn(4, 64)
    past_coords = torch.randn(4, 3)
    kwargs = {
        "features_past": past,
        "coords_past_physical": past_coords,
        "frames_past": torch.tensor([1.0, 2.0, 3.0, 3.0]),
        "coords_src_physical": source_coords,
        "coords_tgt_physical": target_coords,
        "frames_src": torch.full((3,), 4.0),
        "frames_tgt": torch.full((2,), 5.0),
        "past_mask": torch.ones(4, dtype=torch.bool),
    }
    with torch.no_grad():
        single = wrapper(source, target, source_coords, target_coords, **kwargs)
        batched = wrapper(
            source[None],
            target[None],
            source_coords[None],
            target_coords[None],
            **{key: value[None] for key, value in kwargs.items()},
        )
    torch.testing.assert_close(single, batched[0], rtol=0, atol=0)


def test_batched_training_step_backpropagates_through_both_attention_calls() -> None:
    torch.manual_seed(23)
    base = SimpleNodeTransformer(
        feat_dim=64,
        hidden_dim=128,
        n_heads=4,
        n_blocks=1,
        dropout=0.0,
        pair_chunk_size=2,
    )
    wrapper = PastFeatureCrossAttentionTracker(
        base, attention_config=config(gradient_checkpointing=True)
    )
    wrapper.train()
    source = torch.randn(2, 3, 64)
    target = torch.randn(2, 2, 64)
    source_coords = torch.randn(2, 3, 3)
    target_coords = torch.randn(2, 2, 3)
    logits = wrapper(
        source,
        target,
        source_coords,
        target_coords,
        torch.ones(2, 3, dtype=torch.bool),
        torch.ones(2, 2, dtype=torch.bool),
        features_past=torch.randn(2, 4, 64),
        coords_past_physical=torch.randn(2, 4, 3),
        frames_past=torch.tensor([[1.0, 2.0, 3.0, 3.0]] * 2),
        coords_src_physical=source_coords,
        coords_tgt_physical=target_coords,
        frames_src=torch.full((2, 3), 4.0),
        frames_tgt=torch.full((2, 2), 5.0),
        past_mask=torch.ones(2, 4, dtype=torch.bool),
    )
    logits.square().mean().backward()
    for parameter in (wrapper.past_attention.output.weight, base.proj.weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    assert wrapper.past_attention.output.weight.grad.abs().sum() > 0
