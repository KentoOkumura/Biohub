from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

HERE = Path(__file__).resolve().parents[1]
PUBLIC = (
    HERE.parent
    / "exp001_temporal_unet3d_baseline"
    / "official_source/src/tracking_cellmot/models/simple_node_transformer.py"
)


def load_model(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SimpleNodeTransformer


New = load_model(HERE / "simple_node_transformer.py", "spatial_tracker")
Old = load_model(PUBLIC, "public_tracker")
PARAMS = dict(feat_dim=8, hidden_dim=16, n_heads=4, n_blocks=2, dropout=0.0)


def example():
    torch.manual_seed(17)
    return (torch.randn(2, 4, 8), torch.randn(2, 5, 8), torch.randn(2, 4, 3), torch.randn(2, 5, 3))


def test_legacy_strict_checkpoint_and_logits():
    torch.manual_seed(3)
    old = Old(**PARAMS).eval()
    new = New(**PARAMS).eval()
    new.load_state_dict(old.state_dict(), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(new(*example()), old(*example()), rtol=0, atol=0)
    assert not any(key.startswith("self_encoder.") for key in new.state_dict())


@pytest.mark.parametrize("cross", [False, True])
def test_self_attention_padding_order_and_round_trip(cross: bool):
    model = New(
        **PARAMS, use_temporal_self_attention=True, use_cross_attention=cross, n_self_blocks=2
    ).eval()
    x, y, cx, cy = example()
    mx = torch.tensor([[True, True, False, False], [True, True, True, False]])
    my = torch.tensor([[True, True, True, False, False], [True, True, False, False, False]])
    with torch.no_grad():
        original = model(x, y, cx, cy, mx, my)
        changed_x, changed_y = x.clone(), y.clone()
        changed_x[~mx] = torch.randn_like(changed_x[~mx]) * 1e4
        changed_y[~my] = torch.randn_like(changed_y[~my]) * 1e4
        changed = model(changed_x, changed_y, cx, cy, mx, my)
        torch.testing.assert_close(
            original[mx[:, :, None] & my[:, None, :]], changed[mx[:, :, None] & my[:, None, :]]
        )
        perm_x = torch.tensor([1, 0, 2, 3])
        perm_y = torch.tensor([2, 0, 1, 3, 4])
        reordered = model(
            x[:, perm_x], y[:, perm_y], cx[:, perm_x], cy[:, perm_y], mx[:, perm_x], my[:, perm_y]
        )
        torch.testing.assert_close(original[:, perm_x][:, :, perm_y], reordered)
        restored = New(
            **PARAMS, use_temporal_self_attention=True, use_cross_attention=cross, n_self_blocks=2
        ).eval()
        restored.load_state_dict(model.state_dict(), strict=True)
        torch.testing.assert_close(original, restored(x, y, cx, cy, mx, my), rtol=0, atol=0)


def test_frame_separation_and_cell_context():
    model = New(**PARAMS, use_temporal_self_attention=True, use_cross_attention=False).eval()
    x, y, cx, cy = example()
    with torch.no_grad():
        projected = model.norm_in(model.proj(x))
        first = model._encode_frame(projected, torch.ones(2, 4, dtype=torch.bool))
        altered = projected.clone()
        altered[:, 1, 0] += 4
        second = model._encode_frame(altered, torch.ones(2, 4, dtype=torch.bool))
        assert not torch.allclose(first[:, 0], second[:, 0])
        t_first = model._encode_frame(projected, torch.ones(2, 4, dtype=torch.bool))
        y = y + 30
        t_second = model._encode_frame(projected, torch.ones(2, 4, dtype=torch.bool))
        torch.testing.assert_close(t_first, t_second, rtol=0, atol=0)


def test_empty_and_all_padding_are_finite_and_backward():
    model = New(**PARAMS, use_temporal_self_attention=True)
    x, y, cx, cy = example()
    empty = model(x[:, :0], y, cx[:, :0], cy)
    assert empty.shape == (2, 0, 5)
    empty.sum().backward()
    assert model.proj.weight.grad is not None
    model.zero_grad()
    mx = torch.tensor([[False] * 4, [True, True, False, False]])
    my = torch.tensor([[True] * 5, [False] * 5])
    output = model(x, y, cx, cy, mx, my)
    assert torch.isfinite(output).all()
    output.sum().backward()
    assert torch.isfinite(model.proj.weight.grad).all()


def test_invalid_configuration_and_masks():
    with pytest.raises(ValueError):
        New(**{**PARAMS, "n_heads": 3})
    with pytest.raises(ValueError):
        New(**PARAMS, use_temporal_self_attention=True, n_self_blocks=0)
    model = New(**PARAMS)
    x, y, cx, cy = example()
    with pytest.raises(ValueError):
        model(x, y, cx, cy, mask_t=torch.ones(2, 4))
    with pytest.raises(ValueError):
        model(x, y, cx, cy, mask_t=torch.ones(3, 4, dtype=torch.bool))
