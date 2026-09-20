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
from frozen_tracker import batch_legacy_focal_bce  # noqa: E402
from simple_node_transformer import SimpleNodeTransformer  # noqa: E402


def _inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(8)
    feat_t = torch.randn(2, 4, 4)
    feat_t1 = torch.randn(2, 3, 4)
    coords_t = torch.randn(2, 4, 3) * 10
    coords_t1 = torch.randn(2, 3, 3) * 10
    mask_t = torch.tensor([[True, True, False, True], [True, False, False, False]])
    mask_t1 = torch.tensor([[True, False, True], [True, True, False]])
    return feat_t, feat_t1, coords_t, coords_t1, mask_t, mask_t1


def _model(self_attention: bool, cross_attention: bool) -> SimpleNodeTransformer:
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


def test_identity_initialized_model_b_starts_from_public_logits_and_can_learn(
    tmp_path: Path,
) -> None:
    torch.manual_seed(123)
    public = _model(False, True).eval()
    identity = SimpleNodeTransformer(
        feat_dim=4,
        hidden_dim=16,
        n_heads=4,
        n_blocks=1,
        mlp_ratio=2.0,
        dropout=0.0,
        pair_chunk_size=2,
        use_temporal_self_attention=True,
        use_cross_attention=True,
        n_self_blocks=2,
        identity_init_self_attention=True,
    ).eval()
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
    loss = identity(*args)[0, :2, :2].sum()
    loss.backward()
    assert any(
        layer.self_attn.out_proj.weight.grad is not None
        and bool(layer.self_attn.out_proj.weight.grad.abs().sum() > 0)
        for layer in identity.self_encoder.layers
    )
    assert any(
        layer.linear2.weight.grad is not None and bool(layer.linear2.weight.grad.abs().sum() > 0)
        for layer in identity.self_encoder.layers
    )
    import graph_inference as gi

    checkpoint = tmp_path / "identity_tracker.pth"
    torch.save({"state_dict": identity.state_dict()}, checkpoint)
    params = {
        "feature_dim": 4,
        "hidden_dim": 16,
        "n_heads": 4,
        "n_blocks": 1,
        "mlp_ratio": 2.0,
        "dropout": 0.0,
        "pair_chunk_size": 2,
        "n_self_blocks": 2,
    }
    architecture = {
        "use_temporal_self_attention": True,
        "use_cross_attention": True,
        "identity_init_self_attention": True,
    }
    restored = gi._load_tracker(checkpoint, torch.device("cpu"), params, architecture)
    identity.eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(*args), identity(*args), rtol=0, atol=0)
    with pytest.raises(ValueError, match="requires self-attention"):
        SimpleNodeTransformer(identity_init_self_attention=True)


@pytest.mark.parametrize("self_attention,cross_attention", [(True, False), (True, True)])
def test_padding_and_order_are_respected(self_attention: bool, cross_attention: bool) -> None:
    torch.manual_seed(22)
    model = _model(self_attention, cross_attention).eval()
    inputs = list(_inputs())
    with torch.no_grad():
        reference = model(*inputs)
        changed = [x.clone() for x in inputs]
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


@pytest.mark.parametrize("self_attention,cross_attention", [(True, False), (True, True)])
def test_loss_gradients_and_checkpoint_roundtrip(
    self_attention: bool, cross_attention: bool
) -> None:
    torch.manual_seed(40)
    model = _model(self_attention, cross_attention)
    args = list(_inputs())
    # The real collate path packs valid cells first before the loss slices them.
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
    restored = _model(self_attention, cross_attention).eval()
    restored.load_state_dict(torch.load(payload, weights_only=True), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(restored(*args), model(*args), rtol=0, atol=0)


@pytest.mark.parametrize(
    "self_attention,cross_attention", [(False, True), (True, False), (True, True)]
)
def test_empty_and_all_padding_frames_are_finite(
    self_attention: bool, cross_attention: bool
) -> None:
    model = _model(self_attention, cross_attention).eval()
    inputs = _inputs()
    mask_t = torch.tensor([[False] * 4, [True, False, False, False]])
    mask_t1 = torch.tensor([[True, False, False], [False] * 3])
    with torch.no_grad():
        logits = model(*inputs[:4], mask_t, mask_t1)
        assert logits.shape == (2, 4, 3)
        assert torch.isfinite(logits).all()
        empty = model(inputs[0][:, :0], inputs[1], inputs[2][:, :0], inputs[3])
        assert empty.shape == (2, 0, 3)
        unbatched = model(*[x[0] for x in inputs])
        assert unbatched.shape == (4, 3)
        assert torch.isfinite(unbatched).all()


def test_train_manifest_and_inference_loader_share_model_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    import graph_inference as gi

    monkeypatch.chdir(EXP)
    public = _model(False, True).eval()
    public_file = tmp_path / "repo" / "weights" / "primary.pth"
    public_file.parent.mkdir(parents=True)
    torch.save(
        {f"transformer.{key}": value for key, value in public.state_dict().items()}, public_file
    )
    monkeypatch.setattr(
        gi, "EXPECTED_PUBLIC_PRIMARY_CHECKPOINT_SHA256", gi.sha256_file(public_file)
    )
    train_output = tmp_path / "train"
    records = []
    for fold in (0, 1):
        tracker = _model(True, False).eval()
        checkpoint_path = (
            train_output / "models" / "model_a" / f"fold_{fold}" / "primary_tracker_best.pth"
        )
        checkpoint_path.parent.mkdir(parents=True)
        state = tracker.state_dict()
        torch.save(
            {
                "variant": "model_a",
                "fold": fold,
                "architecture": {
                    "use_temporal_self_attention": True,
                    "use_cross_attention": False,
                },
                "model_source_sha256": gi.sha256_file(EXP / "simple_node_transformer.py"),
                "state_dict": state,
            },
            checkpoint_path,
        )
        records.append(
            {
                "variant": "model_a",
                "fold": fold,
                "path": checkpoint_path.relative_to(train_output).as_posix(),
                "file_sha256": gi.sha256_file(checkpoint_path),
                "canonical_state_sha256": gi.canonical_state_sha256(state),
                "evaluation_embryo": "6bba" if fold == 0 else "44b6",
            }
        )
    params = {
        "feature_dim": 4,
        "hidden_dim": 16,
        "n_heads": 4,
        "n_blocks": 1,
        "mlp_ratio": 2.0,
        "dropout": 0.0,
        "pair_chunk_size": 2,
        "n_self_blocks": 2,
    }
    manifest = {
        "model_params": params,
        "experiment": "exp025_frame_self_attention",
        "model_source_sha256": gi.sha256_file(EXP / "simple_node_transformer.py"),
        "active_variants": ["model_a"],
        "architectures": {
            "model_a": {"use_temporal_self_attention": True, "use_cross_attention": False}
        },
        "models": records,
    }
    manifest_path = train_output / "model_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    selected, selection = gi.build_hybrid_checkpoints(
        repo_dir=tmp_path / "repo",
        method="unet_transformer",
        public_weights_relative="weights/primary.pth",
        train_output=train_output,
        output_manifest_path=tmp_path / "selection.json",
        selected_variant="model_a",
        expected_manifest_sha256=gi.sha256_file(manifest_path),
        expected_architecture=manifest["architectures"]["model_a"],
        expected_model_params=params,
    )
    assert selection["selected_variant"] == "model_a"
    assert len(selected) == 2
    loaded = gi._load_tracker(
        Path(selected[0]), torch.device("cpu"), params, manifest["architectures"]["model_a"]
    )
    assert loaded.self_encoder is not None
    assert loaded.use_cross_attention is False
    legacy = gi._load_tracker(public_file, torch.device("cpu"), params)
    assert legacy.self_encoder is None
    with pytest.raises(RuntimeError, match="absent"):
        gi.build_hybrid_checkpoints(
            repo_dir=tmp_path / "repo",
            method="unet_transformer",
            public_weights_relative="weights/primary.pth",
            train_output=train_output,
            output_manifest_path=tmp_path / "wrong.json",
            selected_variant="model_b",
            expected_manifest_sha256=gi.sha256_file(manifest_path),
            expected_architecture=manifest["architectures"]["model_a"],
            expected_model_params=params,
        )
