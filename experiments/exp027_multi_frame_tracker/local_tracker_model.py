from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint


class LocalAttentionBlock(nn.Module):
    """Public block parameters applied to gathered spatial neighbors."""

    def __init__(self, hidden_dim: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, n_heads, batch_first=True, dropout=dropout
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self, hidden: torch.Tensor, indices: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        batch_size, node_count, hidden_dim = hidden.shape
        batch_index = torch.arange(batch_size, device=hidden.device)[:, None, None]
        neighbors = hidden[batch_index, indices]
        queries = self.norm1(hidden).reshape(batch_size * node_count, 1, hidden_dim)
        keys = self.norm1(neighbors).reshape(batch_size * node_count, indices.shape[-1], hidden_dim)
        output, _ = self.cross_attn(
            queries,
            keys,
            keys,
            key_padding_mask=~valid.reshape(batch_size * node_count, -1),
            need_weights=False,
        )
        hidden = hidden + output.reshape(batch_size, node_count, hidden_dim)
        return hidden + self.mlp(self.norm2(hidden))


class LocalThreeFrameTracker(nn.Module):
    """Local attention across three point sets, with central-pair edge logits."""

    def __init__(
        self,
        *,
        feat_dim: int,
        hidden_dim: int,
        n_heads: int,
        n_blocks: int,
        dropout: float,
        pair_chunk_size: int,
        attention_radius_um: float,
        max_neighbors: int,
    ) -> None:
        super().__init__()
        if attention_radius_um <= 0 or max_neighbors < 1:
            raise ValueError("local attention radius and neighbor count must be positive")
        self.pair_chunk_size = pair_chunk_size
        self.attention_radius_um = attention_radius_um
        self.max_neighbors = max_neighbors
        self.max_observed_neighbors = 0
        self.proj = nn.Linear(feat_dim, hidden_dim)
        self.norm_in = nn.LayerNorm(hidden_dim)
        self.blocks = nn.ModuleList(
            LocalAttentionBlock(hidden_dim, n_heads, dropout) for _ in range(n_blocks)
        )
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.pair_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.time_embedding = nn.Embedding(3, hidden_dim)
        nn.init.zeros_(self.time_embedding.weight)

    def load_public_tracker_state(self, state: dict[str, Any]) -> None:
        result = self.load_state_dict(state, strict=False)
        if result.missing_keys != ["time_embedding.weight"] or result.unexpected_keys:
            raise ValueError(
                {
                    "missing_public_parameters": result.missing_keys,
                    "unexpected_public_parameters": result.unexpected_keys,
                }
            )

    def _neighbors(
        self, coords: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            distance = torch.cdist(coords.float(), coords.float())
            distance = distance.masked_fill(~mask[:, None, :], float("inf"))
            neighborhood_size = (distance <= self.attention_radius_um).sum(dim=-1)
            observed_neighbors = int(neighborhood_size[mask].max().item()) if mask.any() else 1
            self.max_observed_neighbors = max(self.max_observed_neighbors, observed_neighbors)
            if observed_neighbors > self.max_neighbors:
                raise RuntimeError(
                    f"neighborhood {observed_neighbors} exceeds limit {self.max_neighbors}"
                )
            k = min(observed_neighbors, coords.shape[1])
            values, indices = torch.topk(distance, k=k, dim=-1, largest=False)
            valid = values <= self.attention_radius_um
            invalid_queries = ~mask
            if invalid_queries.any():
                indices[:, :, 0] = torch.where(
                    invalid_queries, torch.zeros_like(indices[:, :, 0]), indices[:, :, 0]
                )
                valid[:, :, 0] |= invalid_queries
            if not valid.any(dim=-1).all():
                raise RuntimeError("a detection has no local attention neighbor")
            return indices, valid

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        include_previous: bool,
        reverse: bool = False,
    ) -> torch.Tensor:
        prev_mask = batch["prev_mask"] if include_previous else torch.zeros_like(batch["prev_mask"])
        n_prev = batch["features_prev"].shape[1]
        n_src = batch["features_src"].shape[1]
        features = torch.cat(
            [batch["features_prev"], batch["features_src"], batch["features_tgt"]], dim=1
        )
        physical = torch.cat(
            [
                batch["coords_prev_physical"],
                batch["coords_src_physical"],
                batch["coords_tgt_physical"],
            ],
            dim=1,
        )
        mask = torch.cat([prev_mask, batch["source_mask"], batch["target_mask"]], dim=1)
        time_ids = torch.cat(
            [
                torch.zeros(n_prev, device=features.device, dtype=torch.long),
                torch.ones(n_src, device=features.device, dtype=torch.long),
                torch.full(
                    (batch["features_tgt"].shape[1],), 2, device=features.device, dtype=torch.long
                ),
            ]
        )
        hidden = self.norm_in(self.proj(features) + self.time_embedding(time_ids))
        hidden = hidden * mask.unsqueeze(-1)
        indices, valid = self._neighbors(physical, mask)
        for block in self.blocks:
            if self.training and torch.is_grad_enabled():
                hidden = grad_checkpoint(block, hidden, indices, valid, use_reentrant=False)
            else:
                hidden = block(hidden, indices, valid)
            hidden = hidden * mask.unsqueeze(-1)
        hidden = self.norm_out(hidden)
        src = hidden[:, n_prev : n_prev + n_src]
        tgt = hidden[:, n_prev + n_src :]
        coords_src, coords_tgt = batch["coords_src"], batch["coords_tgt"]
        if reverse:
            src, tgt = tgt, src
            coords_src, coords_tgt = coords_tgt, coords_src
        chunks = []
        for start in range(0, src.shape[1], self.pair_chunk_size):
            src_part = src[:, start : start + self.pair_chunk_size]
            coords_part = coords_src[:, start : start + self.pair_chunk_size]

            def score_pairs(
                q: torch.Tensor, k: torch.Tensor, cq: torch.Tensor, ck: torch.Tensor
            ) -> torch.Tensor:
                query = q.unsqueeze(2).expand(-1, -1, k.shape[1], -1)
                key = k.unsqueeze(1).expand(-1, q.shape[1], -1, -1)
                relative = (cq.unsqueeze(2) - ck.unsqueeze(1)) / 100.0
                return self.pair_mlp(torch.cat([query, key, relative], dim=-1)).squeeze(-1)

            if self.training and torch.is_grad_enabled():
                chunk = grad_checkpoint(
                    score_pairs, src_part, tgt, coords_part, coords_tgt, use_reentrant=False
                )
            else:
                chunk = score_pairs(src_part, tgt, coords_part, coords_tgt)
            chunks.append(chunk)
        return torch.cat(chunks, dim=1)
