from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP))
from frame_attention_model import (  # noqa: E402
    SimpleNodeTransformer,
    load_public_initialization,
)

PUBLIC_SOURCE = (
    ROOT / "experiments/exp001_temporal_unet3d_baseline/official_source/"
    "src/tracking_cellmot/models/simple_node_transformer.py"
)
PARAMS = {
    "feat_dim": 64,
    "hidden_dim": 32,
    "n_heads": 4,
    "n_blocks": 2,
    "mlp_ratio": 2.0,
    "dropout": 0.0,
    "pair_chunk_size": 2,
}


def inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(7)
    return (
        torch.randn(1, 4, 64),
        torch.randn(1, 5, 64),
        torch.randn(1, 4, 3),
        torch.randn(1, 5, 3),
        torch.ones(1, 4, dtype=torch.bool),
        torch.ones(1, 5, dtype=torch.bool),
    )


def test_legacy_state_and_logits_match() -> None:
    spec = importlib.util.spec_from_file_location("legacy_tracker_for_exp030", PUBLIC_SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    old = module.SimpleNodeTransformer(**PARAMS).eval()
    new = SimpleNodeTransformer(**PARAMS).eval()
    assert old.state_dict().keys() == new.state_dict().keys()
    new.load_state_dict(old.state_dict(), strict=True)
    with torch.no_grad():
        assert torch.equal(old(*inputs()), new(*inputs()))


@pytest.mark.parametrize("variant", ["model_a", "model_b"])
def test_mask_padding_permutation_gradient_and_roundtrip(variant: str) -> None:
    flags = {
        "use_temporal_self_attention": True,
        "use_cross_attention": variant == "model_b",
    }
    model = SimpleNodeTransformer(**PARAMS, **flags).eval()
    x = inputs()
    base = model(*x)
    # Padding may carry arbitrary values and must not change any valid pair.
    padded = (
        torch.cat([x[0], torch.full((1, 2, 64), 999.0)], dim=1),
        torch.cat([x[1], torch.full((1, 1, 64), -999.0)], dim=1),
        torch.cat([x[2], torch.full((1, 2, 3), 999.0)], dim=1),
        torch.cat([x[3], torch.full((1, 1, 3), -999.0)], dim=1),
        torch.tensor([[True, True, True, True, False, False]]),
        torch.tensor([[True, True, True, True, True, False]]),
    )
    torch.testing.assert_close(model(*padded)[:, :4, :5], base, atol=1e-6, rtol=1e-6)
    src_order = torch.tensor([2, 0, 3, 1])
    tgt_order = torch.tensor([3, 1, 4, 0, 2])
    permuted = (
        x[0][:, src_order],
        x[1][:, tgt_order],
        x[2][:, src_order],
        x[3][:, tgt_order],
        x[4][:, src_order],
        x[5][:, tgt_order],
    )
    torch.testing.assert_close(
        model(*permuted), base[:, src_order][:, :, tgt_order], atol=1e-6, rtol=1e-6
    )
    assert torch.isfinite(base).all()
    base.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert any(
        p.grad is not None
        for name, p in model.named_parameters()
        if name.startswith("self_encoder.")
    )
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = SimpleNodeTransformer(**PARAMS, **flags).eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
    torch.testing.assert_close(restored(*x), base)


def test_empty_and_all_padding_are_finite() -> None:
    model = SimpleNodeTransformer(**PARAMS, use_temporal_self_attention=True).eval()
    x = inputs()
    all_padding = (x[0], x[1], x[2], x[3], torch.zeros_like(x[4]), torch.zeros_like(x[5]))
    assert torch.isfinite(model(*all_padding)).all()
    empty = (x[0][:, :0], x[1], x[2][:, :0], x[3], x[4][:, :0], x[5])
    logits = model(*empty)
    assert logits.shape == (1, 0, 5)
    logits.sum().backward()
    with pytest.raises(ValueError, match="node mask"):
        model(*x[:4], torch.ones(1, 4), x[5])


def test_public_initialization_and_embedded_source() -> None:
    public = SimpleNodeTransformer(**PARAMS).state_dict()
    model = SimpleNodeTransformer(**PARAMS, use_temporal_self_attention=True)
    load_public_initialization(model, public)
    assert all(torch.equal(model.state_dict()[key], value) for key, value in public.items())
    with pytest.raises(ValueError):
        load_public_initialization(model, {"proj.weight": public["proj.weight"]})
    source = (EXP / "frame_attention_model.py").read_text(encoding="utf-8").rstrip()
    notebook = (EXP / "exp030_frame_self_attention_diagnostics_diagnostic.py").read_text(
        encoding="utf-8"
    )
    embedded = (
        notebook.split("# BEGIN EMBEDDED MODEL\n", 1)[1]
        .split("\n# END EMBEDDED MODEL", 1)[0]
        .rstrip()
    )
    assert embedded == source
