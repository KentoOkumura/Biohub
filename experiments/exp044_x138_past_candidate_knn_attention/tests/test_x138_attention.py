from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from capture_patch import patch_x138_pair_capture
from frozen_tracker import AnnotationGraph, FrameAnnotation
from past_candidate_attention import PastCandidateAttentionTracker
from x138_data import X138PairDataset, read_capture
from x138_tracking import X138AttentionTracker, x138_fuse_logits

ROOT = Path(__file__).resolve().parents[1]


def test_notebook_uses_pinned_exp043_setup_and_train_only() -> None:
    notebook = json.loads(
        (ROOT / "exp044_x138_past_candidate_knn_attention_train.ipynb").read_text()
    )
    code = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    assert 'TEST_DIR = COMP_DIR / "train"' in code
    assert (
        "_SELF_EXPECTED_SHA = '32d6c62f738c3dfe4862e3df5272850312e81b622e57d9ba45209ebb382824dc'"
        in code
    )
    assert "patch_x138_pair_capture" in code
    assert "train_from_capture(" in code
    assert "if _exp044_projected_minutes > _exp044_cap" in code
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"cell_{index}", "exec")


def test_capture_patch_requires_all_exp043_anchors() -> None:
    source = """import os
def f():
    for _ in (0,):
        if True:
            _bidirectional_weight = float(
                0.15
            )
            if True:
                reverse_logits_pair = reverse_logits_native.transpose(1, 2)
                if secondary_link_mode == "raw":
                    pass
            raw = edge_logits_pair[0]
        graph = build_graph(coords, edges)
"""
    patched = patch_x138_pair_capture(source)
    assert "EXP044_TRACKER_CAPTURE_DIR" in patched
    assert "continue  # Pair diagnostics" in patched
    with pytest.raises(ValueError, match="anchor count=0"):
        patch_x138_pair_capture(source.replace("_bidirectional_weight", "_other_weight"))


def test_x138_fusion_zero_weights_is_identity() -> None:
    forward = torch.tensor([[[0.2, 0.7], [0.9, -0.3]]], requires_grad=True)
    reverse = torch.zeros(1, 2, 2)
    secondary = torch.ones(1, 2, 2)
    fused = x138_fuse_logits(
        forward,
        reverse,
        secondary,
        bidirectional_weight=0.0,
        secondary_edge_weight=0.0,
    )
    assert torch.equal(fused, forward)
    fused.sum().backward()
    assert forward.grad is not None


class TinyTracker(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, feat_t, feat_t1, coords_t, coords_t1, mask_t=None, mask_t1=None):
        return self.weight * (feat_t[..., :1] - feat_t1[..., :1].transpose(1, 2))


def test_primary_and_attention_are_trainable() -> None:
    attention = PastCandidateAttentionTracker(
        TinyTracker(),
        feature_dim=13,
        hidden_dim=32,
        source_chunk_size=2,
        target_chunk_size=2,
        past_candidate_chunk_size=8,
        gradient_checkpointing=False,
        vector_scale_um=5.0,
        cosine_epsilon=1e-8,
        max_past_candidates=8,
    )
    model = X138AttentionTracker(attention)
    features_src = torch.randn(1, 2, 64)
    features_tgt = torch.randn(1, 2, 64)
    coords_src = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
    coords_tgt = coords_src + 0.5
    coords_prev = coords_src - 0.5
    fused = model(
        features_src,
        features_tgt,
        coords_src,
        coords_tgt,
        coords_prev,
        coords_src,
        coords_tgt,
        torch.ones(1, 2, dtype=torch.bool),
        torch.tensor([[10, 11]]),
        torch.zeros(1, 2, 2),
    )
    assert fused.shape == (1, 2, 2)
    fused.sum().backward()
    assert attention.base_tracker.weight.grad is not None
    assert attention.delta_output.weight.grad is not None
    assert attention.delta_output.weight.grad.abs().sum() > 0


def test_capture_schema_and_sparse_teacher(tmp_path: Path) -> None:
    video = tmp_path / "44b6_test"
    video.mkdir()
    path = video / "000001_000002.npz"
    np.savez_compressed(
        path,
        window_frames=np.array([1, 2], np.int32),
        candidate_ids_prev=np.array([0], np.int64),
        candidate_ids_src=np.array([1, 2], np.int64),
        candidate_ids_tgt=np.array([3, 4], np.int64),
        coords_prev_grid=np.array([[0, 0, 0]], np.float32),
        coords_src_grid=np.array([[1, 0, 0], [9, 0, 0]], np.float32),
        coords_tgt_grid=np.array([[2, 0, 0], [10, 0, 0]], np.float32),
        primary_features_src=np.ones((2, 32), np.float32),
        primary_features_tgt=np.ones((2, 32), np.float32),
        position_features_src=np.zeros((2, 32), np.float32),
        position_features_tgt=np.zeros((2, 32), np.float32),
        primary_forward_logits=np.zeros((2, 2), np.float32),
        primary_reverse_logits=np.zeros((2, 2), np.float32),
        secondary_logits=np.zeros((2, 2), np.float32),
        x138_fused_logits=np.zeros((2, 2), np.float32),
    )
    arrays = read_capture(path)
    annotation = AnnotationGraph(
        frames={
            1: FrameAnnotation(np.array([100]), np.array([[1.625, 0, 0]], np.float32)),
            2: FrameAnnotation(np.array([101]), np.array([[3.25, 0, 0]], np.float32)),
        },
        edges=frozenset({(100, 101)}),
        content_sha256="test",
        outgoing_edges={100: (101,)},
    )
    example = X138PairDataset([path], {video.name: annotation})[0]
    assert arrays["primary_forward_logits"].shape == (2, 2)
    assert example["target"].shape == (2, 2)
    assert example["target"][0, 0] == 1
    assert example["features_src"].shape == (2, 64)
