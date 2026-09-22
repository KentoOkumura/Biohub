from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint

VELOCITY_PAIR_FEATURE_NAMES = (
    "velocity_z",
    "velocity_y",
    "velocity_x",
    "candidate_displacement_z",
    "candidate_displacement_y",
    "candidate_displacement_x",
    "predicted_position_residual_z",
    "predicted_position_residual_y",
    "predicted_position_residual_x",
    "speed",
    "residual_norm",
    "direction_cosine",
    "history_present",
    "history_length",
    "previous_edge_probability",
)


def build_velocity_pair_features(
    *,
    velocity_src: torch.Tensor,
    history_present_src: torch.Tensor,
    history_length_src: torch.Tensor,
    history_confidence_src: torch.Tensor,
    coords_src_physical: torch.Tensor,
    coords_tgt_physical: torch.Tensor,
    vector_scale: float,
    vector_clip_abs: float,
    history_length_cap: int,
) -> torch.Tensor:
    if vector_scale <= 0 or vector_clip_abs <= 0 or history_length_cap < 1:
        raise ValueError("velocity normalization settings must be positive")
    if velocity_src.ndim != 3 or velocity_src.shape[-1] != 3:
        raise ValueError("velocity_src must have shape (B, N_source, 3)")
    batch_size, source_count, _ = velocity_src.shape
    if coords_src_physical.shape != (batch_size, source_count, 3):
        raise ValueError("coords_src_physical does not align with velocity_src")
    if coords_tgt_physical.ndim != 3 or coords_tgt_physical.shape[0] != batch_size:
        raise ValueError("coords_tgt_physical must have shape (B, N_target, 3)")
    expected_node_shape = (batch_size, source_count)
    for name, value in (
        ("history_present_src", history_present_src),
        ("history_length_src", history_length_src),
        ("history_confidence_src", history_confidence_src),
    ):
        if value.shape != expected_node_shape:
            raise ValueError(f"{name} must have shape {expected_node_shape}")

    present = history_present_src.to(dtype=velocity_src.dtype)
    velocity = torch.clamp(
        velocity_src / float(vector_scale),
        min=-float(vector_clip_abs),
        max=float(vector_clip_abs),
    )
    displacement = (coords_tgt_physical.unsqueeze(1) - coords_src_physical.unsqueeze(2)) / float(
        vector_scale
    )
    displacement = torch.clamp(
        displacement,
        min=-float(vector_clip_abs),
        max=float(vector_clip_abs),
    )
    velocity_pair = velocity.unsqueeze(2).expand(-1, -1, displacement.shape[2], -1)
    residual = torch.clamp(
        displacement - velocity_pair,
        min=-float(vector_clip_abs),
        max=float(vector_clip_abs),
    )
    speed = torch.linalg.vector_norm(velocity_pair, dim=-1, keepdim=True)
    displacement_norm = torch.linalg.vector_norm(displacement, dim=-1, keepdim=True)
    residual_norm = torch.linalg.vector_norm(residual, dim=-1, keepdim=True)
    dot = (velocity_pair * displacement).sum(dim=-1, keepdim=True)
    cosine = dot / torch.clamp(speed * displacement_norm, min=1e-6)
    cosine = torch.where(
        (speed > 1e-6) & (displacement_norm > 1e-6),
        cosine,
        torch.zeros_like(cosine),
    )
    target_count = displacement.shape[2]
    present_pair = present.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, target_count, -1)
    length = (
        torch.clamp(history_length_src, min=0, max=history_length_cap)
        .to(dtype=velocity_src.dtype)
        .div(float(history_length_cap))
        .unsqueeze(-1)
        .unsqueeze(-1)
        .expand(-1, -1, target_count, -1)
    )
    confidence = (
        torch.clamp(history_confidence_src, min=0.0, max=1.0)
        .to(dtype=velocity_src.dtype)
        .unsqueeze(-1)
        .unsqueeze(-1)
        .expand(-1, -1, target_count, -1)
    )
    features = torch.cat(
        [
            velocity_pair,
            displacement,
            residual,
            speed,
            residual_norm,
            cosine,
            present_pair,
            length,
            confidence,
        ],
        dim=-1,
    )
    if features.shape[-1] != len(VELOCITY_PAIR_FEATURE_NAMES):
        raise RuntimeError("velocity pair feature width changed unexpectedly")
    features = features * present_pair
    if not torch.isfinite(features).all():
        raise FloatingPointError("velocity pair features contain non-finite values")
    return features


class VelocityAugmentedTracker(nn.Module):
    """Public primary tracker plus a learned delta from fixed prediction history."""

    requires_velocity_features = True

    def __init__(
        self,
        base_tracker: nn.Module,
        *,
        pair_feature_dim: int,
        branch_hidden_dim: int,
        branch_dropout: float,
        pair_chunk_size: int,
        vector_scale: float,
        vector_clip_abs: float,
        history_length_cap: int,
    ) -> None:
        super().__init__()
        if pair_feature_dim != len(VELOCITY_PAIR_FEATURE_NAMES):
            raise ValueError("pair_feature_dim does not match the velocity feature contract")
        self.base_tracker = base_tracker
        self.pair_chunk_size = int(pair_chunk_size)
        self.vector_scale = float(vector_scale)
        self.vector_clip_abs = float(vector_clip_abs)
        self.history_length_cap = int(history_length_cap)
        self.velocity_pair_mlp = nn.Sequential(
            nn.Linear(pair_feature_dim, branch_hidden_dim),
            nn.GELU(),
            nn.Dropout(branch_dropout),
            nn.Linear(branch_hidden_dim, 1),
        )
        final = self.velocity_pair_mlp[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(
        self,
        feat_t: torch.Tensor,
        feat_t1: torch.Tensor,
        coords_t: torch.Tensor,
        coords_t1: torch.Tensor,
        mask_t: torch.Tensor | None = None,
        mask_t1: torch.Tensor | None = None,
        *,
        velocity_src: torch.Tensor,
        history_present_src: torch.Tensor,
        history_length_src: torch.Tensor,
        history_confidence_src: torch.Tensor,
        coords_src_physical: torch.Tensor,
        coords_tgt_physical: torch.Tensor,
    ) -> torch.Tensor:
        unbatched = feat_t.ndim == 2
        base_logits = self.base_tracker(
            feat_t,
            feat_t1,
            coords_t,
            coords_t1,
            mask_t,
            mask_t1,
        )
        if unbatched:
            velocity_src = velocity_src.unsqueeze(0)
            history_present_src = history_present_src.unsqueeze(0)
            history_length_src = history_length_src.unsqueeze(0)
            history_confidence_src = history_confidence_src.unsqueeze(0)
            coords_src_physical = coords_src_physical.unsqueeze(0)
            coords_tgt_physical = coords_tgt_physical.unsqueeze(0)
            base_logits_batched = base_logits.unsqueeze(0)
        else:
            base_logits_batched = base_logits

        chunks: list[torch.Tensor] = []
        source_count = velocity_src.shape[1]
        for start in range(0, source_count, self.pair_chunk_size):
            stop = start + self.pair_chunk_size
            feature_chunk = build_velocity_pair_features(
                velocity_src=velocity_src[:, start:stop],
                history_present_src=history_present_src[:, start:stop],
                history_length_src=history_length_src[:, start:stop],
                history_confidence_src=history_confidence_src[:, start:stop],
                coords_src_physical=coords_src_physical[:, start:stop],
                coords_tgt_physical=coords_tgt_physical,
                vector_scale=self.vector_scale,
                vector_clip_abs=self.vector_clip_abs,
                history_length_cap=self.history_length_cap,
            )

            def score_velocity(features: torch.Tensor) -> torch.Tensor:
                return self.velocity_pair_mlp(features).squeeze(-1)

            if self.training and torch.is_grad_enabled():
                delta = grad_checkpoint(score_velocity, feature_chunk, use_reentrant=False)
            else:
                delta = score_velocity(feature_chunk)
            chunks.append(delta)

        delta_logits = torch.cat(chunks, dim=1)
        if mask_t is not None and mask_t1 is not None:
            if unbatched:
                mask_t = mask_t.unsqueeze(0)
                mask_t1 = mask_t1.unsqueeze(0)
            valid_pairs = mask_t.unsqueeze(-1) & mask_t1.unsqueeze(1)
            delta_logits = delta_logits.masked_fill(~valid_pairs, 0.0)
        logits = base_logits_batched + delta_logits
        return logits.squeeze(0) if unbatched else logits

    def load_public_tracker_state(self, state: dict[str, Any]) -> None:
        self.base_tracker.load_state_dict(state, strict=True)

    def base_state_dict(self) -> dict[str, Any]:
        return self.base_tracker.state_dict()
