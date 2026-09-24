from __future__ import annotations

import ast
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_variant_contract_and_notebook_sources():
    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    model = config["model"]
    assert model["training"]["active_variants"] == ["legacy", "self_only", "self_cross"]
    assert model["variants"] == {
        "legacy": {"use_temporal_self_attention": False, "use_cross_attention": True},
        "self_only": {"use_temporal_self_attention": True, "use_cross_attention": False},
        "self_cross": {"use_temporal_self_attention": True, "use_cross_attention": True},
    }
    assert model["training"]["epochs"] == 3
    assert config["validation"]["n_folds"] == 2
    assert model["output"]["model_count"] == 6
    assert config["lineage"]["hypothesis_id"] == "HYP-20260920-02"
    parent = yaml.safe_load((ROOT.parent / "exp016_frozen_image_encoder/config.yaml").read_text())
    assert config["data"] == parent["data"]
    assert config["validation"] == parent["validation"]
    assert model["teacher"] == parent["model"]["teacher"]
    assert model["loss"] == parent["model"]["loss"]
    assert model["training"]["epochs"] == parent["model"]["training"]["epochs"]
    assert model["training"]["optimizer"] == parent["model"]["training"]["optimizer"]
    assert (
        model["inference"]["replay"]["edge_threshold"]
        == parent["model"]["inference"]["replay"]["edge_threshold"]
    )
    for filename in (
        "simple_node_transformer.py",
        "exp031_frame_self_attention_spatial_train.py",
        "exp031_frame_self_attention_spatial_inference.py",
    ):
        ast.parse((ROOT / filename).read_text(), filename=filename)


def test_graph_replay_patch_keeps_variant_selection_out_of_fixed_setup():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "exp031_graph_inference", ROOT / "graph_inference.py"
    )
    assert spec is not None and spec.loader is not None
    graph = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(graph)
    source = (
        ROOT.parent / "exp015_oracle_stage_limits/exp015_oracle_stage_limits_inference.py"
    ).read_text()
    patched = graph.patch_exp015_source(source)
    setup, replay = patched.split("# EXP016_CACHE_REPLAY_START", 1)
    assert "EXP031_VARIANT" not in setup
    assert "variant=EXP031_VARIANT" in replay
    compile(setup, "fixed_setup.py", "exec")
    compile(replay, "variant_replay.py", "exec")
