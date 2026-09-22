"""Transformer operating on nodes to predict edges between nodes."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_ckpt


class CrossAttentionBlock(nn.Module):
    """A single cross-attention block with MLP and residual connections."""

    def __init__(
        self,
        hidden_dim: int = 64,
        n_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, n_heads, batch_first=True, dropout=dropout
        )

        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        kv_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        key_padding_mask = ~kv_mask if kv_mask is not None else None
        attn_out, _ = self.cross_attn(
            self.norm1(q),
            self.norm1(kv),
            self.norm1(kv),
            key_padding_mask=key_padding_mask,
        )
        q = q + attn_out
        q = q + self.mlp(self.norm2(q))
        return q


class SpatialSelfAttentionLayer(nn.Module):
    """Pre-norm self-attention with one learned physical-distance weight per head."""

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        mlp_ratio: float,
        dropout: float,
        distance_scale_um: float,
        initial_distance_coefficient: float,
    ):
        super().__init__()
        if distance_scale_um <= 0:
            raise ValueError("spatial distance scale must be positive")
        if initial_distance_coefficient <= 0:
            raise ValueError("initial spatial distance coefficient must be positive")
        self.n_heads = n_heads
        self.distance_scale_um = float(distance_scale_um)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, n_heads, batch_first=True, dropout=dropout
        )
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.linear1 = nn.Linear(hidden_dim, mlp_hidden)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(mlp_hidden, hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()
        raw_initial = math.log(math.expm1(float(initial_distance_coefficient)))
        self.raw_distance_coefficient = nn.Parameter(torch.full((n_heads,), raw_initial))

    @property
    def distance_coefficient(self) -> torch.Tensor:
        return F.softplus(self.raw_distance_coefficient)

    def distance_attention_bias(
        self,
        coordinates_um: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        *,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Return additive attention bias with shape (batch * heads, cells, cells)."""
        if coordinates_um.ndim != 3 or coordinates_um.shape[-1] != 3:
            raise ValueError("spatial coordinates must have shape (B, N, 3)")
        batch_size, cell_count, _ = coordinates_um.shape
        normalized = coordinates_um / self.distance_scale_um
        delta = normalized.unsqueeze(2) - normalized.unsqueeze(1)
        squared_distance = delta.square().sum(dim=-1)
        coefficient = self.distance_coefficient.to(
            device=coordinates_um.device,
            dtype=coordinates_um.dtype,
        )
        bias = -coefficient.view(1, self.n_heads, 1, 1) * squared_distance.unsqueeze(1)
        if valid_mask is not None:
            if valid_mask.dtype != torch.bool or valid_mask.shape != (batch_size, cell_count):
                raise ValueError("spatial valid mask must be boolean with shape (B, N)")
            bias = bias.masked_fill(~valid_mask[:, None, None, :], float("-inf"))
        if dtype is not None:
            bias = bias.to(dtype=dtype)
        return bias.reshape(batch_size * self.n_heads, cell_count, cell_count)

    def forward(
        self,
        features: torch.Tensor,
        coordinates_um: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        normalized_features = self.norm1(features)
        attention_bias = self.distance_attention_bias(
            coordinates_um, valid_mask, dtype=normalized_features.dtype
        )
        attention_output, _ = self.self_attn(
            normalized_features,
            normalized_features,
            normalized_features,
            attn_mask=attention_bias,
            need_weights=False,
        )
        features = features + self.dropout1(attention_output)
        feed_forward = self.linear2(
            self.dropout(self.activation(self.linear1(self.norm2(features))))
        )
        return features + self.dropout2(feed_forward)


class SpatialSelfAttentionEncoder(nn.Module):
    """Stack of distance-biased self-attention layers."""

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        mlp_ratio: float,
        dropout: float,
        n_layers: int,
        distance_scale_um: float,
        initial_distance_coefficient: float,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                SpatialSelfAttentionLayer(
                    hidden_dim=hidden_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    distance_scale_um=distance_scale_um,
                    initial_distance_coefficient=initial_distance_coefficient,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(
        self,
        features: torch.Tensor,
        coordinates_um: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        valid_mask = ~src_key_padding_mask if src_key_padding_mask is not None else None
        for layer in self.layers:
            features = layer(features, coordinates_um, valid_mask)
        return features


class SimpleNodeTransformer(nn.Module):
    """Transformer for predicting edges between cell detections."""

    def __init__(
        self,
        feat_dim: int = 33,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_blocks: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.3,
        pair_chunk_size: int | None = 32,
        *,
        use_temporal_self_attention: bool = False,
        use_cross_attention: bool = True,
        n_self_blocks: int = 2,
        identity_init_self_attention: bool = False,
        use_spatial_distance_bias: bool = False,
        spatial_coordinate_scale_zyx_um: Sequence[float] = (1.625, 0.40625, 0.40625),
        spatial_distance_scale_um: float | None = None,
        spatial_distance_bias_initial_coefficient: float = 1.0,
    ):
        super().__init__()
        if hidden_dim % n_heads:
            raise ValueError("hidden_dim must be divisible by n_heads")
        if identity_init_self_attention and not use_temporal_self_attention:
            raise ValueError("identity initialization requires self-attention")
        if use_spatial_distance_bias and not use_temporal_self_attention:
            raise ValueError("spatial distance bias requires self-attention")
        if use_temporal_self_attention and n_self_blocks < 1:
            raise ValueError("n_self_blocks must be positive when self-attention is enabled")
        if use_cross_attention and n_blocks < 1:
            raise ValueError("n_blocks must be positive when cross-attention is enabled")
        coordinate_scale = tuple(float(value) for value in spatial_coordinate_scale_zyx_um)
        if len(coordinate_scale) != 3 or any(value <= 0 for value in coordinate_scale):
            raise ValueError("spatial coordinate scale must contain three positive values")
        if use_spatial_distance_bias and (
            spatial_distance_scale_um is None or spatial_distance_scale_um <= 0
        ):
            raise ValueError("spatial distance bias requires a positive distance scale")

        self.pair_chunk_size = pair_chunk_size
        self.use_temporal_self_attention = use_temporal_self_attention
        self.use_cross_attention = use_cross_attention
        self.use_spatial_distance_bias = use_spatial_distance_bias
        self.register_buffer(
            "spatial_coordinate_scale_zyx_um",
            torch.tensor(coordinate_scale, dtype=torch.float32),
            persistent=False,
        )
        self.proj = nn.Linear(feat_dim, hidden_dim)
        self.norm_in = nn.LayerNorm(hidden_dim)

        self.blocks = nn.ModuleList(
            [CrossAttentionBlock(hidden_dim, n_heads, mlp_ratio, dropout) for _ in range(n_blocks)]
        )

        if use_temporal_self_attention:
            if use_spatial_distance_bias:
                self.self_encoder = SpatialSelfAttentionEncoder(
                    hidden_dim=hidden_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    n_layers=n_self_blocks,
                    distance_scale_um=float(spatial_distance_scale_um),
                    initial_distance_coefficient=spatial_distance_bias_initial_coefficient,
                )
            else:
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=n_heads,
                    dim_feedforward=int(hidden_dim * mlp_ratio),
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.self_encoder = nn.TransformerEncoder(
                    encoder_layer, num_layers=n_self_blocks, enable_nested_tensor=False
                )
            for layer in self.self_encoder.layers:
                for parameter in layer.parameters():
                    if parameter.dim() > 1:
                        nn.init.xavier_uniform_(parameter)
                if identity_init_self_attention:
                    nn.init.zeros_(layer.self_attn.out_proj.weight)
                    nn.init.zeros_(layer.self_attn.out_proj.bias)
                    nn.init.zeros_(layer.linear2.weight)
                    nn.init.zeros_(layer.linear2.bias)
        else:
            self.self_encoder = None

        self.norm_out = nn.LayerNorm(hidden_dim)
        self.pair_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    @staticmethod
    def _normalize_mask(
        mask: torch.Tensor | None, batch_size: int, node_count: int, *, unbatched: bool
    ) -> torch.Tensor | None:
        if mask is None:
            return None
        if unbatched and mask.ndim == 1:
            mask = mask.unsqueeze(0)
        if mask.dtype != torch.bool or mask.shape != (batch_size, node_count):
            raise ValueError("cell mask must be boolean with shape (B, N)")
        return mask

    def _encode_frame(
        self,
        features: torch.Tensor,
        coordinates_zyx: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.self_encoder is None or features.shape[1] == 0:
            return features
        nonempty = (
            torch.ones(features.shape[0], dtype=torch.bool, device=features.device)
            if mask is None
            else mask.any(dim=1)
        )
        if not bool(nonempty.any()):
            return torch.zeros_like(features)
        selected_features = features[nonempty]
        selected_mask = mask[nonempty] if mask is not None else None
        if self.use_spatial_distance_bias:
            scale = self.spatial_coordinate_scale_zyx_um.to(
                device=coordinates_zyx.device,
                dtype=coordinates_zyx.dtype,
            )
            coordinates_um = coordinates_zyx[nonempty] * scale
            selected_encoded = self.self_encoder(
                selected_features,
                coordinates_um,
                src_key_padding_mask=~selected_mask if selected_mask is not None else None,
            )
        else:
            selected_encoded = self.self_encoder(
                selected_features,
                src_key_padding_mask=~selected_mask if selected_mask is not None else None,
            )
        if mask is None:
            return selected_encoded
        encoded = torch.zeros_like(features)
        encoded[nonempty] = selected_encoded
        return encoded.masked_fill(~mask.unsqueeze(-1), 0.0)

    def forward(
        self,
        feat_t: torch.Tensor,
        feat_t1: torch.Tensor,
        coords_t: torch.Tensor,
        coords_t1: torch.Tensor,
        mask_t: torch.Tensor | None = None,
        mask_t1: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict edge logits between detections at consecutive frames."""
        unbatched = feat_t.ndim == 2
        if unbatched:
            feat_t = feat_t.unsqueeze(0)
            feat_t1 = feat_t1.unsqueeze(0)
            coords_t = coords_t.unsqueeze(0)
            coords_t1 = coords_t1.unsqueeze(0)

        batch_size, n_t = feat_t.shape[:2]
        n_t1 = feat_t1.shape[1]
        mask_t = self._normalize_mask(mask_t, batch_size, n_t, unbatched=unbatched)
        mask_t1 = self._normalize_mask(mask_t1, batch_size, n_t1, unbatched=unbatched)
        if n_t == 0 or n_t1 == 0:
            logits = feat_t.new_zeros((batch_size, n_t, n_t1))
            return logits.squeeze(0) if unbatched else logits

        q = self.norm_in(self.proj(feat_t))
        k = self.norm_in(self.proj(feat_t1))
        q = self._encode_frame(q, coords_t, mask_t)
        k = self._encode_frame(k, coords_t1, mask_t1)

        for block in self.blocks if self.use_cross_attention else ():

            def _q_fn(
                q: torch.Tensor,
                kv: torch.Tensor,
                mask: torch.Tensor | None,
                _b: CrossAttentionBlock = block,
            ) -> torch.Tensor:
                return _b(q, kv, kv_mask=mask)

            def _k_fn(
                k: torch.Tensor,
                kv: torch.Tensor,
                mask: torch.Tensor | None,
                _b: CrossAttentionBlock = block,
            ) -> torch.Tensor:
                return _b(k, kv, kv_mask=mask)

            valid_k = (
                torch.ones(batch_size, dtype=torch.bool, device=k.device)
                if mask_t1 is None
                else mask_t1.any(dim=1)
            )
            valid_q = (
                torch.ones(batch_size, dtype=torch.bool, device=q.device)
                if mask_t is None
                else mask_t.any(dim=1)
            )
            if bool(valid_k.all()):
                if torch.is_grad_enabled():
                    q = grad_ckpt(_q_fn, q, k, mask_t1, use_reentrant=False)
                else:
                    q = _q_fn(q, k, mask_t1)
            elif bool(valid_k.any()):
                updated_q = _q_fn(
                    q[valid_k], k[valid_k], mask_t1[valid_k] if mask_t1 is not None else None
                )
                q = q.index_copy(0, valid_k.nonzero(as_tuple=True)[0], updated_q)
            if self.use_temporal_self_attention and mask_t is not None:
                q = q.masked_fill(~mask_t.unsqueeze(-1), 0.0)
            if bool(valid_q.all()):
                if torch.is_grad_enabled():
                    k = grad_ckpt(_k_fn, k, q, mask_t, use_reentrant=False)
                else:
                    k = _k_fn(k, q, mask_t)
            elif bool(valid_q.any()):
                updated_k = _k_fn(
                    k[valid_q], q[valid_q], mask_t[valid_q] if mask_t is not None else None
                )
                k = k.index_copy(0, valid_q.nonzero(as_tuple=True)[0], updated_k)
            if self.use_temporal_self_attention and mask_t1 is not None:
                k = k.masked_fill(~mask_t1.unsqueeze(-1), 0.0)

        q = self.norm_out(q)
        k = self.norm_out(k)

        node_count = q.shape[1]
        chunk = self.pair_chunk_size or node_count
        chunks = []
        pair_mlp = self.pair_mlp

        for index in range(0, node_count, chunk):
            q_chunk = q[:, index : index + chunk, :]
            coords_chunk = coords_t[:, index : index + chunk, :]

            def _chunk_fn(
                query: torch.Tensor,
                keys: torch.Tensor,
                query_coords: torch.Tensor,
                key_coords: torch.Tensor,
                _pair_mlp: nn.Module = pair_mlp,
            ) -> torch.Tensor:
                chunk_count = query.shape[1]
                target_count = keys.shape[1]
                expanded_query = query.unsqueeze(2).expand(-1, -1, target_count, -1)
                expanded_keys = keys.unsqueeze(1).expand(-1, chunk_count, -1, -1)
                relative = (query_coords.unsqueeze(2) - key_coords.unsqueeze(1)) / 100.0
                return _pair_mlp(
                    torch.cat([expanded_query, expanded_keys, relative], dim=-1)
                ).squeeze(-1)

            if torch.is_grad_enabled():
                output = grad_ckpt(
                    _chunk_fn,
                    q_chunk,
                    k,
                    coords_chunk,
                    coords_t1,
                    use_reentrant=False,
                )
            else:
                output = _chunk_fn(q_chunk, k, coords_chunk, coords_t1)
            chunks.append(output)

        logits = torch.cat(chunks, dim=1)
        return logits.squeeze(0) if unbatched else logits
