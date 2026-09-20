"""Cell-pair tracker with optional within-frame self-attention.

The original module names and legacy forward order are retained so public
SimpleNodeTransformer weights load strictly in the control configuration.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as grad_ckpt


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 64,
        n_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
    ) -> None:
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
        self, q: torch.Tensor, kv: torch.Tensor, kv_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        key_padding_mask = ~kv_mask if kv_mask is not None else None
        attn_out, _ = self.cross_attn(
            self.norm1(q),
            self.norm1(kv),
            self.norm1(kv),
            key_padding_mask=key_padding_mask,
        )
        q = q + attn_out
        return q + self.mlp(self.norm2(q))


class SimpleNodeTransformer(nn.Module):
    """Score all pairs between two cell sets without changing their order."""

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
    ) -> None:
        super().__init__()
        if hidden_dim % n_heads:
            raise ValueError("hidden_dim must be divisible by n_heads")
        if use_temporal_self_attention and n_self_blocks < 1:
            raise ValueError("n_self_blocks must be positive when self-attention is enabled")
        if use_cross_attention and n_blocks < 1:
            raise ValueError("n_blocks must be positive when cross-attention is enabled")
        if pair_chunk_size is not None and pair_chunk_size < 1:
            raise ValueError("pair_chunk_size must be positive")
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
            # TransformerEncoder clones its layer. Initialize every clone
            # independently without touching the public checkpoint modules.
            for block in self.self_encoder.layers:
                nn.init.xavier_uniform_(block.self_attn.in_proj_weight)
                nn.init.xavier_uniform_(block.self_attn.out_proj.weight)
                nn.init.xavier_uniform_(block.linear1.weight)
                nn.init.xavier_uniform_(block.linear2.weight)
                for module in (block.self_attn, block.linear1, block.linear2):
                    for name, parameter in module.named_parameters():
                        if name.endswith("bias"):
                            nn.init.zeros_(parameter)

    @staticmethod
    def _mask(
        mask: torch.Tensor | None, batch: int, count: int, device: torch.device
    ) -> torch.Tensor:
        if mask is None:
            return torch.ones((batch, count), dtype=torch.bool, device=device)
        if mask.ndim == 1 and batch == 1:
            mask = mask.unsqueeze(0)
        if mask.shape != (batch, count) or mask.dtype != torch.bool:
            raise ValueError(f"node mask must be boolean with shape {(batch, count)}")
        return mask.to(device)

    def _self_encode(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.self_encoder is None:
            return x
        active = mask.any(dim=1)
        if not bool(active.any()):
            return torch.zeros_like(x)
        selected = x[active].masked_fill(~mask[active, :, None], 0)
        selected_mask = mask[active]

        def encode(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
            assert self.self_encoder is not None
            return self.self_encoder(value, src_key_padding_mask=~valid)

        if torch.is_grad_enabled():
            encoded = grad_ckpt(
                encode,
                selected,
                selected_mask,
                use_reentrant=False,
                preserve_rng_state=True,
            )
        else:
            encoded = encode(selected, selected_mask)
        encoded = encoded.masked_fill(~selected_mask[:, :, None], 0)
        result = torch.zeros_like(x)
        result[active] = encoded
        return result

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
            feat_t, feat_t1 = feat_t.unsqueeze(0), feat_t1.unsqueeze(0)
            coords_t, coords_t1 = coords_t.unsqueeze(0), coords_t1.unsqueeze(0)
        if feat_t.ndim != 3 or feat_t1.ndim != 3:
            raise ValueError("features must have shape [N,F] or [B,N,F]")
        batch, n_t, _ = feat_t.shape
        if feat_t1.shape[0] != batch:
            raise ValueError("feature batch sizes differ")
        n_t1 = feat_t1.shape[1]
        if coords_t.shape != (batch, n_t, 3) or coords_t1.shape != (batch, n_t1, 3):
            raise ValueError("coordinates must match feature counts and have three axes")
        valid_t = self._mask(mask_t, batch, n_t, feat_t.device)
        valid_t1 = self._mask(mask_t1, batch, n_t1, feat_t1.device)
        if n_t == 0 or n_t1 == 0:
            logits = self.proj.weight.sum() * 0
            logits = logits + feat_t.new_zeros((batch, n_t, n_t1))
            return logits.squeeze(0) if unbatched else logits

        q = self.norm_in(self.proj(feat_t))
        k = self.norm_in(self.proj(feat_t1))
        if self.use_temporal_self_attention:
            q = self._self_encode(q, valid_t)
            k = self._self_encode(k, valid_t1)

        active = valid_t.any(dim=1) & valid_t1.any(dim=1)
        if self.use_cross_attention and bool(active.any()):
            qa, ka = q[active], k[active]
            ma, mb = valid_t[active], valid_t1[active]
            for block in self.blocks:

                def q_step(
                    value: torch.Tensor,
                    other: torch.Tensor,
                    mask: torch.Tensor,
                    current: CrossAttentionBlock = block,
                ) -> torch.Tensor:
                    return current(value, other, kv_mask=mask)

                def k_step(
                    value: torch.Tensor,
                    other: torch.Tensor,
                    mask: torch.Tensor,
                    current: CrossAttentionBlock = block,
                ) -> torch.Tensor:
                    return current(value, other, kv_mask=mask)

                if torch.is_grad_enabled():
                    qa = grad_ckpt(q_step, qa, ka, mb, use_reentrant=False)
                    qa = qa.masked_fill(~ma[:, :, None], 0)
                    ka = grad_ckpt(k_step, ka, qa, ma, use_reentrant=False)
                    ka = ka.masked_fill(~mb[:, :, None], 0)
                else:
                    qa = q_step(qa, ka, mb).masked_fill(~ma[:, :, None], 0)
                    ka = k_step(ka, qa, ma).masked_fill(~mb[:, :, None], 0)
            q = q.clone()
            k = k.clone()
            q[active], k[active] = qa, ka

        q = self.norm_out(q)
        k = self.norm_out(k)
        chunk = self.pair_chunk_size or n_t
        chunks = []
        pair_mlp = self.pair_mlp
        for start in range(0, n_t, chunk):
            q_c = q[:, start : start + chunk, :]
            coords_c = coords_t[:, start : start + chunk, :]

            def pair_score(
                qc: torch.Tensor,
                kk: torch.Tensor,
                cc: torch.Tensor,
                cc1: torch.Tensor,
                scoring: nn.Module = pair_mlp,
            ) -> torch.Tensor:
                n_c, n_other = qc.shape[1], kk.shape[1]
                qe = qc.unsqueeze(2).expand(-1, -1, n_other, -1)
                ke = kk.unsqueeze(1).expand(-1, n_c, -1, -1)
                relative = (cc.unsqueeze(2) - cc1.unsqueeze(1)) / 100.0
                return scoring(torch.cat([qe, ke, relative], dim=-1)).squeeze(-1)

            if torch.is_grad_enabled():
                output = grad_ckpt(pair_score, q_c, k, coords_c, coords_t1, use_reentrant=False)
            else:
                output = pair_score(q_c, k, coords_c, coords_t1)
            chunks.append(output)
        logits = torch.cat(chunks, dim=1)
        logits = logits.masked_fill(~active[:, None, None], 0)
        return logits.squeeze(0) if unbatched else logits


def load_public_initialization(
    model: SimpleNodeTransformer, public_state: dict[str, torch.Tensor]
) -> None:
    """Load original weights, allowing only the new encoder to be absent."""
    missing, unexpected = model.load_state_dict(public_state, strict=False)
    expected = {key for key in model.state_dict() if key.startswith("self_encoder.")}
    if set(missing) != expected or unexpected:
        raise ValueError({"missing": missing, "unexpected": unexpected})
