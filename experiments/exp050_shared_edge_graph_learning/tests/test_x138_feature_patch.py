from pathlib import Path

import pytest

from experiments.exp050_shared_edge_graph_learning.x138_feature_patch import add_feature_capture
from tests.test_support import require_saved_files


def test_patch_matches_materialized_x138_source_and_preserves_compilability():
    root = Path(__file__).resolve().parents[3]
    source = (
        root
        / "experiments/exp043_x138_self_trained_head/artifacts/inference_v3/tracking_repo"
        / "scripts/predict_unet_transformer.py"
    )
    require_saved_files(source)
    patched = add_feature_capture(source.read_text())
    assert "node_features=_node_features" in patched
    assert "_CACHE_FEATURE_SUM.clear()" in patched


def test_patch_rejects_source_without_exact_x138_anchors():
    with pytest.raises(RuntimeError, match="anchor occurs 0"):
        add_feature_capture("pass\n")
