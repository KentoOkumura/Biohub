"""Transformer operating on nodes to predict edges between nodes."""

from __future__ import annotations

import torch
import torch.nn as nn
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
        """Cross-attention with residual.

        Parameters
        ----------
        q : torch.Tensor
            Query tensor, shape (B, N_q, D).
        kv : torch.Tensor
            Key/value tensor, shape (B, N_kv, D).
        kv_mask : torch.Tensor, optional
            Boolean mask for kv positions, shape (B, N_kv).
            True = real position, False = padding (will be ignored).
        """
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
    ):
        super().__init__()
        if hidden_dim % n_heads:
            raise ValueError("hidden_dim must be divisible by n_heads")
        if identity_init_self_attention and not use_temporal_self_attention:
            raise ValueError("identity initialization requires self-attention")
        if use_temporal_self_attention and n_self_blocks < 1:
            raise ValueError("n_self_blocks must be positive when self-attention is enabled")
        if use_cross_attention and n_blocks < 1:
            raise ValueError("n_blocks must be positive when cross-attention is enabled")
        self.pair_chunk_size = pair_chunk_size
        self.use_temporal_self_attention = use_temporal_self_attention
        self.use_cross_attention = use_cross_attention
        self.proj = nn.Linear(feat_dim, hidden_dim)
        self.norm_in = nn.LayerNorm(hidden_dim)

        self.blocks = nn.ModuleList(
            [CrossAttentionBlock(hidden_dim, n_heads, mlp_ratio, dropout) for _ in range(n_blocks)]
        )

        # The same encoder weights contextualize each frame independently.
        # Keep it absent in legacy mode so old checkpoints load with strict=True.
        if use_temporal_self_attention:
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
            # TransformerEncoder clones layers; initialize each clone independently.
            for layer in self.self_encoder.layers:
                for parameter in layer.parameters():
                    if parameter.dim() > 1:
                        nn.init.xavier_uniform_(parameter)
                if identity_init_self_attention:
                    # This pre-norm layer adds attention and feed-forward residuals.
                    # Zero their final projections so the pretrained tracker
                    # initially receives unchanged features from both frames.
                    nn.init.zeros_(layer.self_attn.out_proj.weight)
                    nn.init.zeros_(layer.self_attn.out_proj.bias)
                    nn.init.zeros_(layer.linear2.weight)
                    nn.init.zeros_(layer.linear2.bias)
        else:
            self.self_encoder = None

        self.norm_out = nn.LayerNorm(hidden_dim)

        # MLP for pairwise scoring: concatenated features + relative position
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

    def _encode_frame(self, features: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if self.self_encoder is None or features.shape[1] == 0:
            return features
        if mask is None:
            return self.self_encoder(features)
        # PyTorch attention is undefined when every key is padding. Encode only
        # nonempty samples, then return zeros for padded queries and empty frames.
        nonempty = mask.any(dim=1)
        if not bool(nonempty.any()):
            return torch.zeros_like(features)
        encoded = torch.zeros_like(features)
        encoded[nonempty] = self.self_encoder(
            features[nonempty], src_key_padding_mask=~mask[nonempty]
        )
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
        """
        Predict edge logits between detections at consecutive frames.

        Accepts both unbatched (N, D) and batched (B, N, D) inputs.
        When unbatched, a batch dimension is added and removed automatically.

        Parameters
        ----------
        feat_t : torch.Tensor
            Features at time t, shape (N_t, D) or (B, N_t, D).
        feat_t1 : torch.Tensor
            Features at time t+1, shape (N_t1, D) or (B, N_t1, D).
        coords_t : torch.Tensor
            Coordinates (z, y, x) at time t, shape (N_t, 3) or (B, N_t, 3).
        coords_t1 : torch.Tensor
            Coordinates (z, y, x) at time t+1, shape (N_t1, 3) or (B, N_t1, 3).
        mask_t : torch.Tensor, optional
            Boolean mask for t nodes, shape (B, N_t). True = real, False = pad.
        mask_t1 : torch.Tensor, optional
            Boolean mask for t+1 nodes, shape (B, N_t1). True = real, False = pad.
            Unbatched inputs also accept masks with shape (N,).

        Returns
        -------
        torch.Tensor
            Edge logits, shape (N_t, N_t1) or (B, N_t, N_t1).
        """
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

        # 1. Project both frames with the unchanged shared input projection.
        q = self.norm_in(self.proj(feat_t))  # (B, N_t, hidden)
        k = self.norm_in(self.proj(feat_t1))  # (B, N_t1, hidden)
        # 2. Optional NFL-inspired branch: contextualize cells *within* each
        # frame independently, using one shared TransformerEncoder.
        q = self._encode_frame(q, mask_t)
        k = self._encode_frame(k, mask_t1)

        # 3. Optional existing cross-frame fusion. Model B preserves the
        # sequential update: reverse attention sees the newly updated q.
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

            # Skip any batch item whose key/value frame is entirely padding;
            # MultiheadAttention would otherwise emit NaNs for that item.
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

        q = self.norm_out(q)  # (B, N_t, hidden)
        k = self.norm_out(k)  # (B, N_t1, hidden)

        # 4. Score every cross-frame cell pair with the unchanged relative
        # coordinate and Pair MLP path. The output remains raw edge logits.
        # Build pairwise logits in chunks over N_t to avoid O(N²) peak allocation.
        # Full tensor (B, N_t, N_t1, 2*hidden+3) can be tens of GB for large N.
        # Each chunk is grad-checkpointed: forward peak = B×chunk×N_t1×(2H+3),
        # backward only re-stores tiny q_c / coords slice instead of all activations.
        N_t = q.shape[1]
        chunk = self.pair_chunk_size or N_t
        chunks = []
        pair_mlp = self.pair_mlp

        for i in range(0, N_t, chunk):
            q_c = q[:, i : i + chunk, :]
            coords_c = coords_t[:, i : i + chunk, :]

            def _chunk_fn(
                qc: torch.Tensor,
                kk: torch.Tensor,
                cc: torch.Tensor,
                cc1: torch.Tensor,
                _pm: nn.Module = pair_mlp,
            ) -> torch.Tensor:
                nc_i = qc.shape[1]
                n1 = kk.shape[1]
                qe = qc.unsqueeze(2).expand(-1, -1, n1, -1)
                ke = kk.unsqueeze(1).expand(-1, nc_i, -1, -1)
                rel = (cc.unsqueeze(2) - cc1.unsqueeze(1)) / 100.0
                return _pm(torch.cat([qe, ke, rel], dim=-1)).squeeze(-1)

            if torch.is_grad_enabled():
                out = grad_ckpt(_chunk_fn, q_c, k, coords_c, coords_t1, use_reentrant=False)
            else:
                out = _chunk_fn(q_c, k, coords_c, coords_t1)

            chunks.append(out)

        logits = torch.cat(chunks, dim=1)  # (B, N_t, N_t1)

        if unbatched:
            logits = logits.squeeze(0)

        return logits
