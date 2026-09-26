"""Pair-conditioned history attention inside the public primary pair MLP."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from past_candidate_attention import build_past_candidate_features


class SelectedHistory(NamedTuple):
    features: torch.Tensor
    coordinates: torch.Tensor
    candidate_ids: torch.Tensor
    mask: torch.Tensor


def select_history(
    features: torch.Tensor,
    coordinates: torch.Tensor,
    candidate_ids: torch.Tensor,
    mask: torch.Tensor,
    parent_coordinates: torch.Tensor,
    k: int,
) -> SelectedHistory:
    """Physical-distance kNN per chronological parent; IDs break distance ties."""
    if k < 1:
        raise ValueError("k must be positive")
    if features.ndim != 3 or features.shape[-1] != 64:
        raise ValueError("previous features must have shape (B, N, 64)")
    batch, previous_count, _ = features.shape
    if coordinates.shape != (batch, previous_count, 3):
        raise ValueError("previous coordinate shape differs")
    if candidate_ids.shape != mask.shape or mask.shape != (batch, previous_count):
        raise ValueError("previous IDs and mask shape differs")
    if parent_coordinates.ndim != 3 or parent_coordinates.shape[0] != batch:
        raise ValueError("parent coordinates must have shape (B, N, 3)")
    parents = parent_coordinates.shape[1]
    count = min(k, previous_count)
    if count == 0:
        return SelectedHistory(
            features.new_zeros((batch, parents, k, 64)),
            coordinates.new_zeros((batch, parents, k, 3)),
            candidate_ids.new_zeros((batch, parents, k)),
            mask.new_zeros((batch, parents, k)),
        )
    id_order = torch.argsort(candidate_ids, dim=1, stable=True)
    ordered_coords = torch.gather(coordinates, 1, id_order[..., None].expand(-1, -1, 3))
    ordered_mask = torch.gather(mask, 1, id_order)
    distances = ((parent_coordinates[:, :, None] - ordered_coords[:, None]) ** 2).sum(dim=-1)
    distances = distances.masked_fill(~ordered_mask[:, None], float("inf"))
    nearest = torch.argsort(distances, dim=-1, stable=True)[..., :count]
    indices = torch.gather(id_order[:, None].expand(-1, parents, -1), 2, nearest)
    expanded_features = features[:, None].expand(-1, parents, -1, -1)
    expanded_coords = coordinates[:, None].expand(-1, parents, -1, -1)
    expanded_ids = candidate_ids[:, None].expand(-1, parents, -1)
    expanded_mask = mask[:, None].expand(-1, parents, -1)
    chosen_mask = torch.gather(expanded_mask, 2, indices)
    chosen_features = torch.gather(expanded_features, 2, indices[..., None].expand(-1, -1, -1, 64))
    chosen_coords = torch.gather(expanded_coords, 2, indices[..., None].expand(-1, -1, -1, 3))
    chosen_ids = torch.gather(expanded_ids, 2, indices)
    chosen_features = chosen_features.masked_fill(~chosen_mask[..., None], 0)
    chosen_coords = chosen_coords.masked_fill(~chosen_mask[..., None], 0)
    chosen_ids = chosen_ids.masked_fill(~chosen_mask, 0)
    if count < k:
        pad = k - count
        chosen_features = torch.cat(
            (chosen_features, features.new_zeros((batch, parents, pad, 64))), dim=2
        )
        chosen_coords = torch.cat(
            (chosen_coords, coordinates.new_zeros((batch, parents, pad, 3))), dim=2
        )
        chosen_ids = torch.cat((chosen_ids, candidate_ids.new_zeros((batch, parents, pad))), dim=2)
        chosen_mask = torch.cat((chosen_mask, mask.new_zeros((batch, parents, pad))), dim=2)
    return SelectedHistory(chosen_features, chosen_coords, chosen_ids, chosen_mask)


class PrimaryPairFeatureAttention(nn.Module):
    """Public SimpleNodeTransformer with history entering its first pair-MLP layer."""

    def __init__(
        self,
        base_tracker: nn.Module,
        *,
        max_past_candidates: int = 8,
        hidden_dim: int = 128,
        n_heads: int = 4,
        source_chunk_size: int = 32,
        target_chunk_size: int = 32,
        vector_scale_um: float = 5.0,
        cosine_epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        if hidden_dim != 128 or n_heads != 4 or hidden_dim % n_heads:
            raise ValueError("The approved architecture uses width 128 and 4 heads")
        if min(max_past_candidates, source_chunk_size, target_chunk_size) < 1:
            raise ValueError("K and chunk sizes must be positive")
        self.base_tracker = base_tracker
        self.max_past_candidates = max_past_candidates
        self.source_chunk_size = source_chunk_size
        self.target_chunk_size = target_chunk_size
        self.vector_scale_um = vector_scale_um
        self.cosine_epsilon = cosine_epsilon
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.query_projection = nn.Linear(259, hidden_dim)
        self.key_projection = nn.Linear(77, hidden_dim)
        self.value_projection = nn.Linear(77, hidden_dim)
        self.pair_adapter = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.zeros_(self.pair_adapter.weight)

    def _embeddings(
        self, source_features: torch.Tensor, target_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        base = self.base_tracker
        q = base.norm_in(base.proj(source_features))
        k = base.norm_in(base.proj(target_features))
        for block in base.blocks:
            if torch.is_grad_enabled():

                def q_fn(q_value: torch.Tensor, kv_value: torch.Tensor, _block=block):
                    return _block(q_value, kv_value)

                q = checkpoint(q_fn, q, k, use_reentrant=False)

                def k_fn(k_value: torch.Tensor, kv_value: torch.Tensor, _block=block):
                    return _block(k_value, kv_value)

                k = checkpoint(k_fn, k, q, use_reentrant=False)
            else:
                q = block(q, k)
                k = block(k, q)
        return base.norm_out(q), base.norm_out(k)

    def _history_context(
        self,
        pair_input: torch.Tensor,
        selected: SelectedHistory,
        parent_coordinates: torch.Tensor,
        child_coordinates: torch.Tensor,
        *,
        reverse: bool,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch, nq, nk = pair_input.shape[:3]
        count = selected.features.shape[2]
        if count == 0:
            return pair_input.new_zeros((batch, nq, nk, 128))
        geometry = build_past_candidate_features(
            coords_prev_physical=selected.coordinates,
            coords_src_physical=parent_coordinates,
            coords_tgt_physical=child_coordinates,
            vector_scale_um=self.vector_scale_um,
            cosine_epsilon=self.cosine_epsilon,
        )
        past_features = selected.features[:, :, None].expand(
            -1, -1, child_coordinates.shape[1], -1, -1
        )
        past_mask = selected.mask[:, :, None].expand(-1, -1, child_coordinates.shape[1], -1)
        if reverse:
            geometry = geometry.transpose(1, 2)
            past_features = past_features.transpose(1, 2)
            past_mask = past_mask.transpose(1, 2)
        token = torch.cat((past_features, geometry), dim=-1)
        query = self.query_projection(pair_input).reshape(
            batch, nq, nk, self.n_heads, self.head_dim
        )
        key = self.key_projection(token).reshape(batch, nq, nk, count, self.n_heads, self.head_dim)
        value = self.value_projection(token).reshape(
            batch, nq, nk, count, self.n_heads, self.head_dim
        )
        score = (query.unsqueeze(3) * key).sum(-1).permute(0, 1, 2, 4, 3)
        score = score / (self.head_dim**0.5)
        score = score.masked_fill(~past_mask.unsqueeze(3), float("-inf"))
        null_score = score.new_zeros((*score.shape[:-1], 1))
        weights = torch.softmax(torch.cat((score, null_score), dim=-1), dim=-1)[..., :count]
        context = (weights.permute(0, 1, 2, 4, 3).unsqueeze(-1) * value).sum(dim=3)
        context = context.reshape(batch, nq, nk, -1)
        if return_weights:
            return context, weights
        return context

    def _score_chunk(
        self,
        pair_input: torch.Tensor,
        history: SelectedHistory,
        parent_coordinates: torch.Tensor,
        child_coordinates: torch.Tensor,
        *,
        reverse: bool,
    ) -> torch.Tensor:
        context = self._history_context(
            pair_input,
            history,
            parent_coordinates,
            child_coordinates,
            reverse=reverse,
        )
        first = self.base_tracker.pair_mlp[0](pair_input)
        first = first + self.pair_adapter(context)
        return self.base_tracker.pair_mlp[1:](first).squeeze(-1)

    def _score(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        source_coordinates: torch.Tensor,
        target_coordinates: torch.Tensor,
        parent_coordinates: torch.Tensor,
        child_coordinates: torch.Tensor,
        selected: SelectedHistory,
        *,
        reverse: bool,
    ) -> torch.Tensor:
        q, k = self._embeddings(source_features, target_features)
        outputs = []
        for q_start in range(0, q.shape[1], self.source_chunk_size):
            q_stop = q_start + self.source_chunk_size
            q_chunk = q[:, q_start:q_stop]
            q_coord = source_coordinates[:, q_start:q_stop]
            row_outputs = []
            for k_start in range(0, k.shape[1], self.target_chunk_size):
                k_stop = k_start + self.target_chunk_size
                k_chunk = k[:, k_start:k_stop]
                k_coord = target_coordinates[:, k_start:k_stop]
                pair_input = torch.cat(
                    (
                        q_chunk[:, :, None].expand(-1, -1, k_chunk.shape[1], -1),
                        k_chunk[:, None].expand(-1, q_chunk.shape[1], -1, -1),
                        (q_coord[:, :, None] - k_coord[:, None]) / 100.0,
                    ),
                    dim=-1,
                )
                if reverse:
                    history = SelectedHistory(
                        selected.features[:, k_start:k_stop],
                        selected.coordinates[:, k_start:k_stop],
                        selected.candidate_ids[:, k_start:k_stop],
                        selected.mask[:, k_start:k_stop],
                    )
                    parent_chunk = parent_coordinates[:, k_start:k_stop]
                    child_chunk = child_coordinates[:, q_start:q_stop]
                else:
                    history = SelectedHistory(
                        selected.features[:, q_start:q_stop],
                        selected.coordinates[:, q_start:q_stop],
                        selected.candidate_ids[:, q_start:q_stop],
                        selected.mask[:, q_start:q_stop],
                    )
                    parent_chunk = parent_coordinates[:, q_start:q_stop]
                    child_chunk = child_coordinates[:, k_start:k_stop]
                if torch.is_grad_enabled():

                    def chunk_fn(
                        pair_value: torch.Tensor,
                        past_value: torch.Tensor,
                        past_coords: torch.Tensor,
                        past_ids: torch.Tensor,
                        past_mask: torch.Tensor,
                        parent_value: torch.Tensor,
                        child_value: torch.Tensor,
                        _reverse: bool = reverse,
                    ) -> torch.Tensor:
                        return self._score_chunk(
                            pair_value,
                            SelectedHistory(past_value, past_coords, past_ids, past_mask),
                            parent_value,
                            child_value,
                            reverse=_reverse,
                        )

                    chunk_logits = checkpoint(
                        chunk_fn,
                        pair_input,
                        history.features,
                        history.coordinates,
                        history.candidate_ids,
                        history.mask,
                        parent_chunk,
                        child_chunk,
                        use_reentrant=False,
                    )
                else:
                    chunk_logits = self._score_chunk(
                        pair_input,
                        history,
                        parent_chunk,
                        child_chunk,
                        reverse=reverse,
                    )
                row_outputs.append(chunk_logits)
            outputs.append(torch.cat(row_outputs, dim=2))
        return torch.cat(outputs, dim=1)

    @torch.no_grad()
    def inspect_pair(
        self,
        features_parent: torch.Tensor,
        features_child: torch.Tensor,
        coordinates_parent: torch.Tensor,
        coordinates_child: torch.Tensor,
        *,
        features_previous: torch.Tensor,
        coordinates_previous_physical: torch.Tensor,
        ids_previous: torch.Tensor,
        mask_previous: torch.Tensor,
        coordinates_parent_physical: torch.Tensor,
        coordinates_child_physical: torch.Tensor,
        parent_index: int,
        child_index: int,
    ) -> dict:
        """Explain one annotated pair without changing prediction or training."""
        selected = select_history(
            features_previous,
            coordinates_previous_physical,
            ids_previous,
            mask_previous,
            coordinates_parent_physical,
            self.max_past_candidates,
        )
        history = SelectedHistory(
            *(value[:, parent_index : parent_index + 1] for value in selected)
        )
        parent_physical = coordinates_parent_physical[:, parent_index : parent_index + 1]
        child_physical = coordinates_child_physical[:, child_index : child_index + 1]
        report = {}
        for direction in ("forward", "reverse"):
            reverse = direction == "reverse"
            if reverse:
                q, k = self._embeddings(features_child, features_parent)
                q = q[:, child_index : child_index + 1]
                k = k[:, parent_index : parent_index + 1]
                q_coord = coordinates_child[:, child_index : child_index + 1]
                k_coord = coordinates_parent[:, parent_index : parent_index + 1]
            else:
                q, k = self._embeddings(features_parent, features_child)
                q = q[:, parent_index : parent_index + 1]
                k = k[:, child_index : child_index + 1]
                q_coord = coordinates_parent[:, parent_index : parent_index + 1]
                k_coord = coordinates_child[:, child_index : child_index + 1]
            pair_input = torch.cat(
                (q[:, :, None], k[:, None], (q_coord[:, :, None] - k_coord[:, None]) / 100.0),
                dim=-1,
            )
            context, weights = self._history_context(
                pair_input,
                history,
                parent_physical,
                child_physical,
                reverse=reverse,
                return_weights=True,
            )
            baseline = self.base_tracker.pair_mlp(pair_input).squeeze(-1)
            adjusted = self._score_chunk(
                pair_input,
                history,
                parent_physical,
                child_physical,
                reverse=reverse,
            )
            valid = history.mask[0, 0]
            report[direction] = {
                "candidate_ids": history.candidate_ids[0, 0, valid].cpu().tolist(),
                "mean_head_attention_weight": weights[0, 0, 0][:, valid].mean(dim=0).cpu().tolist(),
                "null_attention_weight": float(1.0 - weights[0, 0, 0].sum(dim=-1).mean()),
                "adapter_residual_norm": float(self.pair_adapter(context).norm()),
                "pair_logit_delta": float((adjusted - baseline)[0, 0, 0]),
            }
        return report

    def forward(
        self,
        features_parent: torch.Tensor,
        features_child: torch.Tensor,
        coordinates_parent: torch.Tensor,
        coordinates_child: torch.Tensor,
        *,
        features_previous: torch.Tensor,
        coordinates_previous_physical: torch.Tensor,
        ids_previous: torch.Tensor,
        mask_previous: torch.Tensor,
        coordinates_parent_physical: torch.Tensor,
        coordinates_child_physical: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        selected = select_history(
            features_previous,
            coordinates_previous_physical,
            ids_previous,
            mask_previous,
            coordinates_parent_physical,
            self.max_past_candidates,
        )
        forward = self._score(
            features_parent,
            features_child,
            coordinates_parent,
            coordinates_child,
            coordinates_parent_physical,
            coordinates_child_physical,
            selected,
            reverse=False,
        )
        reverse = self._score(
            features_child,
            features_parent,
            coordinates_child,
            coordinates_parent,
            coordinates_parent_physical,
            coordinates_child_physical,
            selected,
            reverse=True,
        )
        return forward, reverse
