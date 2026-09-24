"""Predict cell edges with optional within-frame self-attention."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_ckpt


class CrossAttentionBlock(nn.Module):
    """The unchanged public cross-attention block and parameter names."""

    def __init__(self, hidden_dim=64, n_heads=4, mlp_ratio=2.0, dropout=0.1):
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

    def forward(self, q, kv, kv_mask=None):
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


class SimpleNodeTransformer(nn.Module):
    """Score all pairs of cells in adjacent frames.

    ``use_temporal_self_attention`` is the retained configuration key from the
    approved design. It applies attention over cells *within* each frame.
    """

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
    ):
        super().__init__()
        if hidden_dim % n_heads:
            raise ValueError("hidden_dim must be divisible by n_heads")
        if use_temporal_self_attention and n_self_blocks < 1:
            raise ValueError("n_self_blocks must be positive when self-attention is enabled")
        if use_cross_attention and n_blocks < 1:
            raise ValueError("n_blocks must be positive when cross-attention is enabled")
        if pair_chunk_size is not None and pair_chunk_size < 1:
            raise ValueError("pair_chunk_size must be positive or None")
        self.pair_chunk_size = pair_chunk_size
        self.use_temporal_self_attention = use_temporal_self_attention
        self.use_cross_attention = use_cross_attention
        self.proj = nn.Linear(feat_dim, hidden_dim)
        self.norm_in = nn.LayerNorm(hidden_dim)
        self.blocks = nn.ModuleList(
            [CrossAttentionBlock(hidden_dim, n_heads, mlp_ratio, dropout) for _ in range(n_blocks)]
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
        # No new state keys in legacy mode, so strict public checkpoint loading works.
        self.self_encoder: nn.TransformerEncoder | None = None
        if use_temporal_self_attention:
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=n_heads,
                dim_feedforward=int(hidden_dim * mlp_ratio),
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.self_encoder = nn.TransformerEncoder(
                layer, num_layers=n_self_blocks, enable_nested_tensor=False
            )
            # Encoder clones a layer with identical initial values. Reinitialize
            # only the added layers with independent RNG draws.
            for encoder_layer in self.self_encoder.layers:
                for module in encoder_layer.modules():
                    if isinstance(module, nn.MultiheadAttention):
                        nn.init.xavier_uniform_(module.in_proj_weight)
                        nn.init.xavier_uniform_(module.out_proj.weight)
                        if module.in_proj_bias is not None:
                            nn.init.zeros_(module.in_proj_bias)
                        if module.out_proj.bias is not None:
                            nn.init.zeros_(module.out_proj.bias)
                    elif (
                        isinstance(module, nn.Linear)
                        and module is not encoder_layer.self_attn.out_proj
                    ):
                        nn.init.xavier_uniform_(module.weight)
                        if module.bias is not None:
                            nn.init.zeros_(module.bias)

    @staticmethod
    def _node_mask(mask: torch.Tensor | None, features: torch.Tensor) -> torch.Tensor:
        expected = features.shape[:2]
        if mask is None:
            return torch.ones(expected, dtype=torch.bool, device=features.device)
        if mask.ndim == 1 and features.shape[0] == 1:
            mask = mask.unsqueeze(0)
        if mask.shape != expected or mask.dtype != torch.bool:
            raise ValueError(f"node mask must be boolean with shape {tuple(expected)}")
        return mask.to(device=features.device)

    def _encode_frame(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.self_encoder is None:
            return x
        active = mask.any(dim=1)
        if not bool(active.any()):
            return torch.zeros_like(x)
        selected = x[active].masked_fill(~mask[active, :, None], 0)
        padding_mask = ~mask[active]
        for layer in self.self_encoder.layers:

            def run(src: torch.Tensor, pad: torch.Tensor, *, _layer=layer) -> torch.Tensor:
                return _layer(src, src_key_padding_mask=pad)

            if torch.is_grad_enabled():
                selected = grad_ckpt(run, selected, padding_mask, use_reentrant=False)
            else:
                selected = run(selected, padding_mask)
        if self.self_encoder.norm is not None:
            selected = self.self_encoder.norm(selected)
        selected = selected.masked_fill(~mask[active, :, None], 0)
        result = torch.zeros_like(x)
        return result.index_copy(0, active.nonzero(as_tuple=True)[0], selected)

    def forward(
        self,
        feat_t: torch.Tensor,
        feat_t1: torch.Tensor,
        coords_t: torch.Tensor,
        coords_t1: torch.Tensor,
        mask_t: torch.Tensor | None = None,
        mask_t1: torch.Tensor | None = None,
    ) -> torch.Tensor:
        unbatched = feat_t.ndim == 2
        if unbatched:
            feat_t = feat_t.unsqueeze(0)
            feat_t1 = feat_t1.unsqueeze(0)
            coords_t = coords_t.unsqueeze(0)
            coords_t1 = coords_t1.unsqueeze(0)
        if feat_t.ndim != 3 or feat_t1.ndim != 3 or feat_t.shape[0] != feat_t1.shape[0]:
            raise ValueError("features must have shape [B,N,F] with a shared batch size")
        if coords_t.shape != (*feat_t.shape[:2], 3) or coords_t1.shape != (*feat_t1.shape[:2], 3):
            raise ValueError("coordinates must have shape [B,N,3]")
        mask_t_was_none = mask_t is None
        mask_t1_was_none = mask_t1 is None
        mask_t = self._node_mask(mask_t, feat_t)
        mask_t1 = self._node_mask(mask_t1, feat_t1)
        cross_mask_t = None if mask_t_was_none else mask_t
        cross_mask_t1 = None if mask_t1_was_none else mask_t1
        if feat_t.shape[1] == 0 or feat_t1.shape[1] == 0:
            logits = feat_t.new_zeros((feat_t.shape[0], feat_t.shape[1], feat_t1.shape[1]))
            logits = logits + self.proj.weight.sum() * 0
            return logits.squeeze(0) if unbatched else logits

        q = self.norm_in(self.proj(feat_t))
        k = self.norm_in(self.proj(feat_t1))
        if self.use_temporal_self_attention:
            q = self._encode_frame(q, mask_t)
            k = self._encode_frame(k, mask_t1)

        both_active = mask_t.any(dim=1) & mask_t1.any(dim=1)
        for block in self.blocks if self.use_cross_attention else ():

            def q_fn(q_: torch.Tensor, kv: torch.Tensor, mask: torch.Tensor | None, *, _b=block):
                return _b(q_, kv, kv_mask=mask)

            def k_fn(k_: torch.Tensor, kv: torch.Tensor, mask: torch.Tensor | None, *, _b=block):
                return _b(k_, kv, kv_mask=mask)

            if bool(both_active.all()):
                # Preserve public model operation order for normal legacy inputs.
                if torch.is_grad_enabled():
                    q = grad_ckpt(q_fn, q, k, cross_mask_t1, use_reentrant=False)
                    k = grad_ckpt(k_fn, k, q, cross_mask_t, use_reentrant=False)
                else:
                    q = q_fn(q, k, cross_mask_t1)
                    k = k_fn(k, q, cross_mask_t)
            elif bool(both_active.any()):
                idx = both_active.nonzero(as_tuple=True)[0]
                qa, ka = q[idx], k[idx]
                mt = None if cross_mask_t is None else cross_mask_t[idx]
                mt1 = None if cross_mask_t1 is None else cross_mask_t1[idx]
                if torch.is_grad_enabled():
                    qa = grad_ckpt(q_fn, qa, ka, mt1, use_reentrant=False)
                    ka = grad_ckpt(k_fn, ka, qa, mt, use_reentrant=False)
                else:
                    qa = q_fn(qa, ka, mt1)
                    ka = k_fn(ka, qa, mt)
                q = q.index_copy(0, idx, qa)
                k = k.index_copy(0, idx, ka)
            if self.use_temporal_self_attention:
                q = q.masked_fill(~mask_t[:, :, None], 0)
                k = k.masked_fill(~mask_t1[:, :, None], 0)

        q = self.norm_out(q)
        k = self.norm_out(k)
        n_t = q.shape[1]
        chunk = self.pair_chunk_size or n_t
        outputs = []
        pair_mlp = self.pair_mlp
        for i in range(0, n_t, chunk):
            q_c = q[:, i : i + chunk, :]
            coords_c = coords_t[:, i : i + chunk, :]

            def chunk_fn(
                qc: torch.Tensor,
                kk: torch.Tensor,
                cc: torch.Tensor,
                cc1: torch.Tensor,
                *,
                _pm=pair_mlp,
            ) -> torch.Tensor:
                nc_i = qc.shape[1]
                n1 = kk.shape[1]
                qe = qc.unsqueeze(2).expand(-1, -1, n1, -1)
                ke = kk.unsqueeze(1).expand(-1, nc_i, -1, -1)
                rel = (cc.unsqueeze(2) - cc1.unsqueeze(1)) / 100.0
                return _pm(torch.cat([qe, ke, rel], dim=-1)).squeeze(-1)

            if torch.is_grad_enabled():
                out = grad_ckpt(chunk_fn, q_c, k, coords_c, coords_t1, use_reentrant=False)
            else:
                out = chunk_fn(q_c, k, coords_c, coords_t1)
            outputs.append(out)
        logits = torch.cat(outputs, dim=1)
        return logits.squeeze(0) if unbatched else logits
