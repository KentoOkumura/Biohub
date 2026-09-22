from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path

import pytest
import yaml

torch = pytest.importorskip("torch")

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP))
from frozen_tracker import batch_legacy_focal_bce  # noqa: E402
from simple_node_transformer import (  # noqa: E402
    SimpleNodeTransformer,
    SpatialSelfAttentionLayer,
)


def _inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(8)
    feat_t = torch.randn(2, 4, 4)
    feat_t1 = torch.randn(2, 3, 4)
    coords_t = torch.randn(2, 4, 3) * 10
    coords_t1 = torch.randn(2, 3, 3) * 10
    mask_t = torch.tensor([[True, True, False, True], [True, False, False, False]])
    mask_t1 = torch.tensor([[True, False, True], [True, True, False]])
    return feat_t, feat_t1, coords_t, coords_t1, mask_t, mask_t1


def _model(
    self_attention: bool,
    cross_attention: bool,
    *,
    spatial: bool = False,
    identity: bool = False,
) -> SimpleNodeTransformer:
    return SimpleNodeTransformer(
        feat_dim=4,
        hidden_dim=16,
        n_heads=4,
        n_blocks=1,
        mlp_ratio=2.0,
        dropout=0.0,
        pair_chunk_size=2,
        use_temporal_self_attention=self_attention,
        use_cross_attention=cross_attention,
        n_self_blocks=2,
        identity_init_self_attention=identity,
        use_spatial_distance_bias=spatial,
        spatial_coordinate_scale_zyx_um=(2.0, 0.5, 0.25),
        spatial_distance_scale_um=4.0 if spatial else None,
        spatial_distance_bias_initial_coefficient=1.0,
    )


def test_legacy_state_and_output_match_saved_source() -> None:
    source = (
        ROOT
        / "experiments/exp001_temporal_unet3d_baseline/official_source/src/"
        / "tracking_cellmot/models/simple_node_transformer.py"
    )
    spec = importlib.util.spec_from_file_location("saved_public_node_transformer", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(12)
    saved = module.SimpleNodeTransformer(
        feat_dim=4, hidden_dim=16, n_heads=4, n_blocks=1, dropout=0.0, pair_chunk_size=2
    ).eval()
    current = _model(False, True).eval()
    current.load_state_dict(saved.state_dict(), strict=True)
    args = _inputs()
    with torch.no_grad():
        expected = saved(*args)
        actual = current(*args)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not any(key.startswith("self_encoder.") for key in current.state_dict())


@pytest.mark.parametrize("spatial", [False, True])
def test_identity_initialized_branch_starts_from_public_logits_and_can_learn(
    spatial: bool,
) -> None:
    torch.manual_seed(123)
    public = _model(False, True).eval()
    identity = _model(True, True, spatial=spatial, identity=True).eval()
    incompatible = identity.load_state_dict(public.state_dict(), strict=False)
    assert set(incompatible.missing_keys) == {
        f"self_encoder.{name}" for name in identity.self_encoder.state_dict()
    }
    assert not incompatible.unexpected_keys
    args = _inputs()
    with torch.no_grad():
        expected = public(*args)
        actual = identity(*args)
    for batch in range(2):
        rows = args[4][batch].nonzero(as_tuple=True)[0]
        cols = args[5][batch].nonzero(as_tuple=True)[0]
        torch.testing.assert_close(
            actual[batch][rows][:, cols], expected[batch][rows][:, cols], rtol=0, atol=0
        )

    identity.train()
    optimizer = torch.optim.SGD(identity.parameters(), lr=0.05)
    for step in range(2):
        optimizer.zero_grad()
        loss = identity(*args)[0, :2, :2].sum()
        loss.backward()
        assert any(
            layer.self_attn.out_proj.weight.grad is not None
            and bool(layer.self_attn.out_proj.weight.grad.abs().sum() > 0)
            for layer in identity.self_encoder.layers
        )
        if spatial and step == 1:
            assert any(
                layer.raw_distance_coefficient.grad is not None
                and bool(layer.raw_distance_coefficient.grad.abs().sum() > 0)
                for layer in identity.self_encoder.layers
            )
        optimizer.step()

    with pytest.raises(ValueError, match="requires self-attention"):
        SimpleNodeTransformer(identity_init_self_attention=True)
    with pytest.raises(ValueError, match="requires self-attention"):
        SimpleNodeTransformer(use_spatial_distance_bias=True)


def test_distance_bias_uses_squared_physical_distance_per_head_and_masks_keys() -> None:
    layer = SpatialSelfAttentionLayer(
        hidden_dim=8,
        n_heads=2,
        mlp_ratio=2.0,
        dropout=0.0,
        distance_scale_um=2.0,
        initial_distance_coefficient=1.0,
    )
    coordinates_um = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 4.0, 0.0]]])
    mask = torch.tensor([[True, True, False]])
    bias = layer.distance_attention_bias(coordinates_um, mask).reshape(1, 2, 3, 3)
    torch.testing.assert_close(layer.distance_coefficient, torch.ones(2))
    expected = torch.tensor(
        [[0.0, -1.0, float("-inf")], [-1.0, 0.0, float("-inf")], [-4.0, -5.0, float("-inf")]]
    )
    for head in range(2):
        torch.testing.assert_close(bias[0, head], expected)


def test_model_converts_original_voxels_only_for_spatial_attention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model(True, False, spatial=True).eval()
    layer = model.self_encoder.layers[0]
    observed: list[torch.Tensor] = []
    original = layer.distance_attention_bias

    def capture(
        coordinates_um: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        *,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        observed.append(coordinates_um.detach().clone())
        return original(coordinates_um, valid_mask, dtype=dtype)

    monkeypatch.setattr(layer, "distance_attention_bias", capture)
    args = _inputs()
    with torch.no_grad():
        model(*args)
    expected_source = args[2] * torch.tensor([2.0, 0.5, 0.25])
    expected_target = args[3] * torch.tensor([2.0, 0.5, 0.25])
    torch.testing.assert_close(observed[0], expected_source)
    torch.testing.assert_close(observed[1], expected_target)


@pytest.mark.parametrize(
    "self_attention,cross_attention,spatial",
    [(True, False, False), (True, True, False), (True, False, True), (True, True, True)],
)
def test_padding_and_order_are_respected(
    self_attention: bool, cross_attention: bool, spatial: bool
) -> None:
    torch.manual_seed(22)
    model = _model(self_attention, cross_attention, spatial=spatial).eval()
    inputs = list(_inputs())
    with torch.no_grad():
        reference = model(*inputs)
        changed = [value.clone() for value in inputs]
        changed[0][~inputs[4]] += 900.0
        changed[1][~inputs[5]] -= 900.0
        changed[2][~inputs[4]] += 900.0
        changed[3][~inputs[5]] -= 900.0
        actual = model(*changed)
        for batch in range(2):
            rows = inputs[4][batch].nonzero(as_tuple=True)[0]
            cols = inputs[5][batch].nonzero(as_tuple=True)[0]
            torch.testing.assert_close(
                actual[batch][rows][:, cols], reference[batch][rows][:, cols]
            )

        order_t = torch.tensor([3, 1, 0, 2])
        order_t1 = torch.tensor([2, 0, 1])
        permuted = model(
            inputs[0][:, order_t],
            inputs[1][:, order_t1],
            inputs[2][:, order_t],
            inputs[3][:, order_t1],
            inputs[4][:, order_t],
            inputs[5][:, order_t1],
        )
        torch.testing.assert_close(permuted, reference[:, order_t][:, :, order_t1])


@pytest.mark.parametrize("spatial", [False, True])
def test_loss_gradients_and_checkpoint_roundtrip(spatial: bool) -> None:
    torch.manual_seed(40)
    model = _model(True, True, spatial=spatial)
    args = list(_inputs())
    args[4] = torch.tensor([[True, True, True, False], [True, True, False, False]])
    args[5] = torch.tensor([[True, True, False], [True, True, False]])
    logits = model(*args)
    assert logits.shape == (2, 4, 3)
    assert torch.isfinite(logits).all()
    target = torch.zeros_like(logits)
    target[0, 0, 0] = 1.0
    target[1, 1, 1] = 1.0
    loss = batch_legacy_focal_bce(logits, target, args[4], args[5])
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.self_encoder.parameters())
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    model.eval()
    payload = io.BytesIO()
    torch.save(model.state_dict(), payload)
    payload.seek(0)
    restored = _model(True, True, spatial=spatial).eval()
    restored.load_state_dict(torch.load(payload, weights_only=True), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(restored(*args), model(*args), rtol=0, atol=0)


@pytest.mark.parametrize(
    "self_attention,cross_attention,spatial",
    [(False, True, False), (True, False, False), (True, True, True)],
)
def test_empty_and_all_padding_frames_are_finite(
    self_attention: bool, cross_attention: bool, spatial: bool
) -> None:
    model = _model(self_attention, cross_attention, spatial=spatial).eval()
    inputs = _inputs()
    mask_t = torch.tensor([[False] * 4, [True, False, False, False]])
    mask_t1 = torch.tensor([[True, False, False], [False] * 3])
    with torch.no_grad():
        logits = model(*inputs[:4], mask_t, mask_t1)
        assert logits.shape == (2, 4, 3)
        assert torch.isfinite(logits).all()
        empty = model(inputs[0][:, :0], inputs[1], inputs[2][:, :0], inputs[3])
        assert empty.shape == (2, 0, 3)
        unbatched = model(*[value[0] for value in inputs])
        assert unbatched.shape == (4, 3)
        assert torch.isfinite(unbatched).all()


def test_experiment_contract_has_one_fold_specific_spatial_variant() -> None:
    config = yaml.safe_load((EXP / "config.yaml").read_text())
    training = config["model"]["training"]
    assert training["active_variants"] == ["model_b_spatial_distance_bias"]
    assert training["epochs"] == 3
    assert config["validation"]["n_folds"] == 2
    assert config["model"]["output"]["model_count"] == 2
    assert config["model"]["control"]["retrain"] is False
    architecture = config["model"]["architectures"]["model_b_spatial_distance_bias"]
    assert architecture["identity_init_self_attention"] is True
    assert architecture["use_spatial_distance_bias"] is True
    assert architecture["spatial_distance_scale_um_by_fold"] == {
        "0": 8.125,
        "1": 8.285906791687012,
    }
    assert config["model"]["inference"]["implemented"] is True
    assert config["model"]["inference"]["selected_variant"] == "model_b_spatial_distance_bias"
    assert "early_pair_gate" not in config["validation"]
    assert (
        config["validation"]["final_graph_evaluation"]["metric"]
        == "official_adjusted_edge_jaccard_plus_0.1_division_jaccard"
    )


def test_graph_replay_restores_fold_specific_spatial_tracker(tmp_path: Path) -> None:
    import graph_inference as gi

    torch.manual_seed(77)
    model = _model(True, True, spatial=True).eval()
    checkpoint = tmp_path / "spatial_tracker.pth"
    torch.save({"state_dict": model.state_dict()}, checkpoint)
    params = {
        "feature_dim": 4,
        "hidden_dim": 16,
        "n_heads": 4,
        "n_blocks": 1,
        "mlp_ratio": 2.0,
        "dropout": 0.0,
        "pair_chunk_size": 2,
        "n_self_blocks": 2,
        "spatial_coordinate_scale_zyx_um": [2.0, 0.5, 0.25],
        "spatial_distance_bias_initial_coefficient": 1.0,
    }
    architecture = {
        "use_temporal_self_attention": True,
        "use_cross_attention": True,
        "identity_init_self_attention": True,
        "use_spatial_distance_bias": True,
        "spatial_distance_scale_um": 4.0,
    }
    restored = gi._load_tracker(checkpoint, torch.device("cpu"), params, architecture)
    assert restored.self_encoder.layers[0].distance_scale_um == 4.0
    args = _inputs()
    with torch.no_grad():
        torch.testing.assert_close(restored(*args), model(*args), rtol=0, atol=0)


def test_fold_architecture_resolution_uses_configured_scale() -> None:
    import graph_inference as gi

    config = yaml.safe_load((EXP / "config.yaml").read_text())
    architecture = config["model"]["architectures"]["model_b_spatial_distance_bias"]
    fold_zero = gi.resolve_fold_architecture(architecture, 0)
    fold_one = gi.resolve_fold_architecture(architecture, 1)
    assert "spatial_distance_scale_um_by_fold" not in fold_zero
    assert fold_zero["spatial_distance_scale_um"] == 8.125
    assert fold_one["spatial_distance_scale_um"] == 8.285906791687012
