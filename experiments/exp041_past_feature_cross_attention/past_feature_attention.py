"""Past feature cross-attention before the public two-frame tracker blocks."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class PastFeatureAttention(nn.Module):
    """Update current detections from all valid points in three earlier frames."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        heads: int,
        position_time_mlp_hidden_dim: int,
        position_scale_um: float,
        query_chunk_size: int,
        key_chunk_size: int,
        gradient_checkpointing: bool,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or heads <= 0 or hidden_dim % heads:
            raise ValueError("hidden_dim must be a positive multiple of heads")
        if position_scale_um <= 0 or min(query_chunk_size, key_chunk_size) <= 0:
            raise ValueError("position scale and chunk sizes must be positive")
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.position_scale_um = float(position_scale_um)
        self.query_chunk_size = int(query_chunk_size)
        self.key_chunk_size = int(key_chunk_size)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.position_time_mlp = nn.Sequential(
            nn.Linear(5, position_time_mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(position_time_mlp_hidden_dim, heads),
        )
        self.no_past_score = nn.Parameter(torch.zeros(heads))
        self.output = nn.Linear(hidden_dim, hidden_dim, bias=False)
        for layer in (
            self.query_proj,
            self.key_proj,
            self.value_proj,
            self.position_time_mlp[0],
            self.position_time_mlp[2],
        ):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.output.weight)
        self.null_mass_sum: torch.Tensor | None = None
        self.null_mass_count: torch.Tensor | None = None

    def reset_null_mass(self) -> None:
        self.null_mass_sum = None
        self.null_mass_count = None

    def null_mass_fraction(self) -> float | None:
        if self.null_mass_count is None or not bool(self.null_mass_count.item()):
            return None
        return float((self.null_mass_sum / self.null_mass_count).item())

    def _record_null_mass(self, probability: torch.Tensor, mask: torch.Tensor | None) -> None:
        if self.training:
            return
        with torch.no_grad():
            valid = (
                torch.ones_like(probability, dtype=torch.bool)
                if mask is None
                else mask[:, None, :].expand_as(probability)
            )
            total = probability.masked_select(valid).sum()
            count = valid.sum()
            self.null_mass_sum = total if self.null_mass_sum is None else self.null_mass_sum + total
            self.null_mass_count = (
                count if self.null_mass_count is None else self.null_mass_count + count
            )

    def _aggregate_chunk(
        self,
        query: torch.Tensor,
        past: torch.Tensor,
        query_coords: torch.Tensor,
        past_coords: torch.Tensor,
        query_frames: torch.Tensor,
        past_frames: torch.Tensor,
        past_mask: torch.Tensor,
        query_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, query_count, _ = query.shape
        if past.shape[1] == 0 or not bool(past_mask.any()):
            self._record_null_mass(query.new_ones(batch_size, self.heads, query_count), query_mask)
            return torch.zeros_like(query)
        q = self.query_proj(query).reshape(batch_size, query_count, self.heads, self.head_dim)
        running_max = self.no_past_score[None, :, None].expand(batch_size, -1, query_count)
        denominator = torch.ones_like(running_max)
        numerator = q.new_zeros(batch_size, self.heads, query_count, self.head_dim)
        for start in range(0, past.shape[1], self.key_chunk_size):
            stop = min(start + self.key_chunk_size, past.shape[1])
            keys = self.key_proj(past[:, start:stop]).reshape(
                batch_size, stop - start, self.heads, self.head_dim
            )
            values = self.value_proj(past[:, start:stop]).reshape(
                batch_size, stop - start, self.heads, self.head_dim
            )
            displacement = (
                past_coords[:, None, start:stop] - query_coords[:, :, None]
            ) / self.position_scale_um
            distance = torch.linalg.vector_norm(displacement, dim=-1, keepdim=True)
            frame_delta = (past_frames[:, None, start:stop] - query_frames[:, :, None]).unsqueeze(
                -1
            )
            position_time = torch.cat((displacement, distance, frame_delta), dim=-1)
            bias = self.position_time_mlp(position_time).permute(0, 3, 1, 2)
            scores = torch.einsum("bqhd,bkhd->bhqk", q, keys) / math.sqrt(self.head_dim)
            scores = (scores + bias).masked_fill(
                ~past_mask[:, None, None, start:stop], float("-inf")
            )
            next_max = torch.maximum(running_max, scores.max(dim=-1).values)
            old_scale = torch.exp(running_max - next_max)
            weights = torch.exp(scores - next_max.unsqueeze(-1))
            denominator = denominator * old_scale + weights.sum(dim=-1)
            numerator = numerator * old_scale.unsqueeze(-1) + torch.einsum(
                "bhqk,bkhd->bhqd", weights, values
            )
            running_max = next_max
        null_probability = torch.exp(self.no_past_score[None, :, None] - running_max) / denominator
        self._record_null_mass(null_probability, query_mask)
        aggregate = numerator / denominator.unsqueeze(-1)
        return self.output(
            aggregate.permute(0, 2, 1, 3).reshape(batch_size, query_count, self.hidden_dim)
        )

    def forward(
        self,
        query: torch.Tensor,
        past: torch.Tensor,
        query_coords: torch.Tensor,
        past_coords: torch.Tensor,
        query_frames: torch.Tensor,
        past_frames: torch.Tensor,
        past_mask: torch.Tensor,
        query_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if (
            query.ndim != 3
            or past.ndim != 3
            or query.shape[-1] != self.hidden_dim
            or past.shape[-1] != self.hidden_dim
        ):
            raise ValueError("query and past must be batched hidden features")
        batch_size, query_count = query.shape[:2]
        if past_mask.shape != past.shape[:2] or past_mask.dtype != torch.bool:
            raise ValueError("past_mask must be boolean with shape (B, N_past)")
        if query_frames.shape != (batch_size, query_count) or past_frames.shape != past.shape[:2]:
            raise ValueError("frame arrays must match query and past point counts")
        if query_coords.shape != (batch_size, query_count, 3) or past_coords.shape != (
            *past.shape[:2],
            3,
        ):
            raise ValueError("physical coordinates must match point counts")
        if query_mask is not None and (
            query_mask.shape != (batch_size, query_count) or query_mask.dtype != torch.bool
        ):
            raise ValueError("query_mask must be boolean with shape (B, N_query)")
        if query_count == 0:
            return query
        if past.shape[1] == 0 or not bool(past_mask.any()):
            self._record_null_mass(query.new_ones(batch_size, self.heads, query_count), query_mask)
            return query
        parts = []
        for start in range(0, query_count, self.query_chunk_size):
            stop = min(start + self.query_chunk_size, query_count)
            args = (
                query[:, start:stop],
                past,
                query_coords[:, start:stop],
                past_coords,
                query_frames[:, start:stop],
                past_frames,
                past_mask,
                None if query_mask is None else query_mask[:, start:stop],
            )
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                residual = checkpoint(self._aggregate_chunk, *args, use_reentrant=False)
            else:
                residual = self._aggregate_chunk(*args)
            if query_mask is not None:
                residual = residual.masked_fill(~query_mask[:, start:stop, None], 0.0)
            parts.append(query[:, start:stop] + residual)
        return torch.cat(parts, dim=1)


class PastFeatureCrossAttentionTracker(nn.Module):
    """Public primary tracker with one shared past-attention layer before its blocks."""

    requires_past_feature_attention = True

    def __init__(self, base_tracker: nn.Module, *, attention_config: dict[str, Any]) -> None:
        super().__init__()
        self.base_tracker = base_tracker
        self.past_attention = PastFeatureAttention(
            hidden_dim=int(attention_config["hidden_dim"]),
            heads=int(attention_config["heads"]),
            position_time_mlp_hidden_dim=int(attention_config["position_time_mlp_hidden_dim"]),
            position_scale_um=float(attention_config["position_scale_um"]),
            query_chunk_size=int(attention_config["query_chunk_size"]),
            key_chunk_size=int(attention_config["key_chunk_size"]),
            gradient_checkpointing=bool(attention_config["gradient_checkpointing"]),
        )

    def load_public_tracker_state(self, state: dict[str, Any]) -> None:
        self.base_tracker.load_state_dict(state, strict=True)

    def base_state_dict(self) -> dict[str, Any]:
        return self.base_tracker.state_dict()

    def forward(
        self,
        feat_t: torch.Tensor,
        feat_t1: torch.Tensor,
        coords_t: torch.Tensor,
        coords_t1: torch.Tensor,
        mask_t: torch.Tensor | None = None,
        mask_t1: torch.Tensor | None = None,
        *,
        features_past: torch.Tensor,
        coords_past_physical: torch.Tensor,
        frames_past: torch.Tensor,
        coords_src_physical: torch.Tensor,
        coords_tgt_physical: torch.Tensor,
        frames_src: torch.Tensor,
        frames_tgt: torch.Tensor,
        past_mask: torch.Tensor,
    ) -> torch.Tensor:
        base = self.base_tracker
        unbatched = feat_t.ndim == 2
        if unbatched:
            feat_t, feat_t1, coords_t, coords_t1 = (
                value.unsqueeze(0) for value in (feat_t, feat_t1, coords_t, coords_t1)
            )
            features_past, coords_past_physical, frames_past = (
                value.unsqueeze(0) for value in (features_past, coords_past_physical, frames_past)
            )
            coords_src_physical, coords_tgt_physical, frames_src, frames_tgt, past_mask = (
                value.unsqueeze(0)
                for value in (
                    coords_src_physical,
                    coords_tgt_physical,
                    frames_src,
                    frames_tgt,
                    past_mask,
                )
            )
            mask_t = None if mask_t is None else mask_t.unsqueeze(0)
            mask_t1 = None if mask_t1 is None else mask_t1.unsqueeze(0)
        batch_size, n_t = feat_t.shape[:2]
        n_t1 = feat_t1.shape[1]
        if n_t == 0 or n_t1 == 0:
            logits = feat_t.new_zeros((batch_size, n_t, n_t1))
            return logits.squeeze(0) if unbatched else logits
        q = base.norm_in(base.proj(feat_t))
        k = base.norm_in(base.proj(feat_t1))
        past = (
            base.norm_in(base.proj(features_past))
            if features_past.shape[1]
            else q.new_zeros(batch_size, 0, q.shape[-1])
        )
        q = self.past_attention(
            q,
            past,
            coords_src_physical,
            coords_past_physical,
            frames_src,
            frames_past,
            past_mask,
            mask_t,
        )
        k = self.past_attention(
            k,
            past,
            coords_tgt_physical,
            coords_past_physical,
            frames_tgt,
            frames_past,
            past_mask,
            mask_t1,
        )
        for block in base.blocks:

            def apply_block(
                current: torch.Tensor,
                other: torch.Tensor,
                mask: torch.Tensor | None,
                *,
                active_block: nn.Module = block,
            ) -> torch.Tensor:
                return active_block(current, other, kv_mask=mask)

            if torch.is_grad_enabled():
                q = checkpoint(apply_block, q, k, mask_t1, use_reentrant=False)
                k = checkpoint(apply_block, k, q, mask_t, use_reentrant=False)
            else:
                q = apply_block(q, k, mask_t1)
                k = apply_block(k, q, mask_t)
        q, k = base.norm_out(q), base.norm_out(k)
        chunk = base.pair_chunk_size or n_t
        chunks = []
        for start in range(0, n_t, chunk):
            q_part = q[:, start : start + chunk]
            coord_part = coords_t[:, start : start + chunk]

            def score_pairs(
                qc: torch.Tensor, kk: torch.Tensor, cc: torch.Tensor, cc1: torch.Tensor
            ) -> torch.Tensor:
                count = qc.shape[1]
                target_count = kk.shape[1]
                rel = (cc[:, :, None] - cc1[:, None, :]) / 100.0
                joined = torch.cat(
                    (
                        qc[:, :, None, :].expand(-1, -1, target_count, -1),
                        kk[:, None, :, :].expand(-1, count, -1, -1),
                        rel,
                    ),
                    dim=-1,
                )
                return base.pair_mlp(joined).squeeze(-1)

            if torch.is_grad_enabled():
                chunks.append(
                    checkpoint(score_pairs, q_part, k, coord_part, coords_t1, use_reentrant=False)
                )
            else:
                chunks.append(score_pairs(q_part, k, coord_part, coords_t1))
        logits = torch.cat(chunks, dim=1)
        return logits.squeeze(0) if unbatched else logits

    def reverse_logits(
        self,
        feat_t: torch.Tensor,
        feat_t1: torch.Tensor,
        coords_t: torch.Tensor,
        coords_t1: torch.Tensor,
        mask_t: torch.Tensor | None = None,
        mask_t1: torch.Tensor | None = None,
        *,
        features_past: torch.Tensor,
        coords_past_physical: torch.Tensor,
        frames_past: torch.Tensor,
        coords_src_physical: torch.Tensor,
        coords_tgt_physical: torch.Tensor,
        frames_src: torch.Tensor,
        frames_tgt: torch.Tensor,
        past_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Score t+1 to t with the same real-time past, then align pair axes."""
        reverse = self.forward(
            feat_t1,
            feat_t,
            coords_t1,
            coords_t,
            mask_t1,
            mask_t,
            features_past=features_past,
            coords_past_physical=coords_past_physical,
            frames_past=frames_past,
            coords_src_physical=coords_tgt_physical,
            coords_tgt_physical=coords_src_physical,
            frames_src=frames_tgt,
            frames_tgt=frames_src,
            past_mask=past_mask,
        )
        return reverse.transpose(-2, -1)
