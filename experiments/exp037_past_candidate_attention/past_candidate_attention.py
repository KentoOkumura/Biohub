from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

PAST_CANDIDATE_FEATURE_NAMES = (
    "past_displacement_z",
    "past_displacement_y",
    "past_displacement_x",
    "current_displacement_z",
    "current_displacement_y",
    "current_displacement_x",
    "displacement_change_z",
    "displacement_change_y",
    "displacement_change_x",
    "past_displacement_norm",
    "current_displacement_norm",
    "displacement_change_norm",
    "direction_cosine",
)


def build_past_candidate_features(
    *,
    coords_prev_physical: torch.Tensor,
    coords_src_physical: torch.Tensor,
    coords_tgt_physical: torch.Tensor,
    vector_scale_um: float,
    cosine_epsilon: float,
) -> torch.Tensor:
    """Build the agreed 13-D feature for every (source, target, past) triple."""
    if vector_scale_um <= 0 or cosine_epsilon <= 0:
        raise ValueError("vector scale and cosine epsilon must be positive")
    if coords_prev_physical.ndim != 3 or coords_prev_physical.shape[-1] != 3:
        raise ValueError("coords_prev_physical must have shape (B, N_past, 3)")
    if coords_src_physical.ndim != 3 or coords_src_physical.shape[-1] != 3:
        raise ValueError("coords_src_physical must have shape (B, N_source, 3)")
    if coords_tgt_physical.ndim != 3 or coords_tgt_physical.shape[-1] != 3:
        raise ValueError("coords_tgt_physical must have shape (B, N_target, 3)")
    batch_size = coords_prev_physical.shape[0]
    if coords_src_physical.shape[0] != batch_size or coords_tgt_physical.shape[0] != batch_size:
        raise ValueError("past, source, and target coordinates must share a batch dimension")

    scale = float(vector_scale_um)
    past_displacement = (
        coords_src_physical[:, :, None, None, :] - coords_prev_physical[:, None, None, :, :]
    ) / scale
    current_displacement = (
        coords_tgt_physical[:, :, None, :] - coords_src_physical[:, None, :, :]
    ).transpose(1, 2) / scale
    current_displacement = current_displacement.unsqueeze(3)
    past_displacement = past_displacement.expand(-1, -1, coords_tgt_physical.shape[1], -1, -1)
    current_displacement = current_displacement.expand(
        -1, -1, -1, coords_prev_physical.shape[1], -1
    )
    displacement_change = current_displacement - past_displacement
    past_norm = torch.linalg.vector_norm(past_displacement, dim=-1, keepdim=True)
    current_norm = torch.linalg.vector_norm(current_displacement, dim=-1, keepdim=True)
    change_norm = torch.linalg.vector_norm(displacement_change, dim=-1, keepdim=True)
    dot = (past_displacement * current_displacement).sum(dim=-1, keepdim=True)
    denominator = past_norm * current_norm
    cosine = dot / torch.clamp(denominator, min=float(cosine_epsilon))
    cosine = torch.where(denominator > float(cosine_epsilon), cosine, torch.zeros_like(cosine))
    features = torch.cat(
        [
            past_displacement,
            current_displacement,
            displacement_change,
            past_norm,
            current_norm,
            change_norm,
            cosine,
        ],
        dim=-1,
    )
    if features.shape[-1] != len(PAST_CANDIDATE_FEATURE_NAMES):
        raise RuntimeError("past-candidate feature width changed unexpectedly")
    if not torch.isfinite(features).all():
        raise FloatingPointError("past-candidate features contain non-finite values")
    return features


class PastCandidateAttentionTracker(nn.Module):
    """Public primary tracker plus attention over all candidates one frame earlier."""

    requires_past_candidate_attention = True

    def __init__(
        self,
        base_tracker: nn.Module,
        *,
        feature_dim: int,
        hidden_dim: int,
        source_chunk_size: int,
        target_chunk_size: int,
        past_candidate_chunk_size: int,
        gradient_checkpointing: bool,
        vector_scale_um: float,
        cosine_epsilon: float,
    ) -> None:
        super().__init__()
        if feature_dim != len(PAST_CANDIDATE_FEATURE_NAMES):
            raise ValueError("feature_dim does not match the past-candidate feature contract")
        if (
            hidden_dim < 1
            or source_chunk_size < 1
            or target_chunk_size < 1
            or past_candidate_chunk_size < 1
        ):
            raise ValueError("hidden and chunk sizes must be positive")
        self.base_tracker = base_tracker
        self.source_chunk_size = int(source_chunk_size)
        self.target_chunk_size = int(target_chunk_size)
        self.past_candidate_chunk_size = int(past_candidate_chunk_size)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.vector_scale_um = float(vector_scale_um)
        self.cosine_epsilon = float(cosine_epsilon)
        self.candidate_mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.GELU(),
        )
        self.attention_score = nn.Linear(hidden_dim, 1, bias=True)
        self.no_past_score = nn.Parameter(torch.zeros(()))
        self.delta_output = nn.Linear(hidden_dim, 1, bias=False)
        nn.init.zeros_(self.delta_output.weight)

    def _aggregate_source_chunk(
        self,
        coords_prev_physical: torch.Tensor,
        coords_src_physical: torch.Tensor,
        coords_tgt_physical: torch.Tensor,
        prev_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, source_count, _ = coords_src_physical.shape
        target_count = coords_tgt_physical.shape[1]
        hidden_dim = self.delta_output.in_features
        running_max = self.no_past_score.expand(batch_size, source_count, target_count)
        denominator = torch.ones_like(running_max)
        numerator = torch.zeros(
            batch_size,
            source_count,
            target_count,
            hidden_dim,
            device=coords_src_physical.device,
            dtype=coords_src_physical.dtype,
        )
        for start in range(0, coords_prev_physical.shape[1], self.past_candidate_chunk_size):
            stop = min(start + self.past_candidate_chunk_size, coords_prev_physical.shape[1])
            features = build_past_candidate_features(
                coords_prev_physical=coords_prev_physical[:, start:stop],
                coords_src_physical=coords_src_physical,
                coords_tgt_physical=coords_tgt_physical,
                vector_scale_um=self.vector_scale_um,
                cosine_epsilon=self.cosine_epsilon,
            )
            candidate_hidden = self.candidate_mlp(features)
            scores = self.attention_score(candidate_hidden).squeeze(-1)
            valid = prev_mask[:, None, None, start:stop]
            scores = scores.masked_fill(~valid, float("-inf"))
            chunk_max = scores.max(dim=-1).values
            next_max = torch.maximum(running_max, chunk_max)
            old_scale = torch.exp(running_max - next_max)
            weights = torch.exp(scores - next_max.unsqueeze(-1))
            weights = torch.where(valid, weights, torch.zeros_like(weights))
            denominator = denominator * old_scale + weights.sum(dim=-1)
            numerator = numerator * old_scale.unsqueeze(-1) + (
                weights.unsqueeze(-1) * candidate_hidden
            ).sum(dim=-2)
            running_max = next_max
        return numerator / denominator.unsqueeze(-1)

    def forward(
        self,
        feat_t: torch.Tensor,
        feat_t1: torch.Tensor,
        coords_t: torch.Tensor,
        coords_t1: torch.Tensor,
        mask_t: torch.Tensor | None = None,
        mask_t1: torch.Tensor | None = None,
        *,
        coords_prev_physical: torch.Tensor,
        coords_src_physical: torch.Tensor,
        coords_tgt_physical: torch.Tensor,
        prev_mask: torch.Tensor,
    ) -> torch.Tensor:
        unbatched = feat_t.ndim == 2
        base_logits = self.base_tracker(feat_t, feat_t1, coords_t, coords_t1, mask_t, mask_t1)
        if unbatched:
            coords_prev_physical = coords_prev_physical.unsqueeze(0)
            coords_src_physical = coords_src_physical.unsqueeze(0)
            coords_tgt_physical = coords_tgt_physical.unsqueeze(0)
            prev_mask = prev_mask.unsqueeze(0)
            base_logits = base_logits.unsqueeze(0)
            if mask_t is not None:
                mask_t = mask_t.unsqueeze(0)
            if mask_t1 is not None:
                mask_t1 = mask_t1.unsqueeze(0)
        if coords_src_physical.shape[1] == 0 or coords_tgt_physical.shape[1] == 0:
            return base_logits.squeeze(0) if unbatched else base_logits
        source_delta_chunks: list[torch.Tensor] = []
        for source_start in range(0, coords_src_physical.shape[1], self.source_chunk_size):
            source_stop = min(source_start + self.source_chunk_size, coords_src_physical.shape[1])
            target_delta_chunks: list[torch.Tensor] = []
            for target_start in range(0, coords_tgt_physical.shape[1], self.target_chunk_size):
                target_stop = min(
                    target_start + self.target_chunk_size, coords_tgt_physical.shape[1]
                )
                aggregate_args = (
                    coords_prev_physical,
                    coords_src_physical[:, source_start:source_stop],
                    coords_tgt_physical[:, target_start:target_stop],
                    prev_mask,
                )
                if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                    aggregated = checkpoint(
                        self._aggregate_source_chunk,
                        *aggregate_args,
                        use_reentrant=False,
                    )
                else:
                    aggregated = self._aggregate_source_chunk(*aggregate_args)
                target_delta_chunks.append(self.delta_output(aggregated).squeeze(-1))
            source_delta_chunks.append(torch.cat(target_delta_chunks, dim=2))
        delta_logits = torch.cat(source_delta_chunks, dim=1)
        if mask_t is not None and mask_t1 is not None:
            valid_pairs = mask_t.unsqueeze(-1) & mask_t1.unsqueeze(1)
            delta_logits = delta_logits.masked_fill(~valid_pairs, 0.0)
        logits = base_logits + delta_logits
        return logits.squeeze(0) if unbatched else logits

    def load_public_tracker_state(self, state: dict[str, Any]) -> None:
        self.base_tracker.load_state_dict(state, strict=True)

    def base_state_dict(self) -> dict[str, Any]:
        return self.base_tracker.state_dict()
