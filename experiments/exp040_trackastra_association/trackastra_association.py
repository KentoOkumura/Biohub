# ruff: noqa: B905, B007, E501
"""Trackastra association model and Biohub fixed-cache adaptation.

Architecture excerpts from weigertlab/trackastra revision
 aa57a95160002e0fc70b915ab74178b39c99fd6a, BSD-3-Clause.
See vendor/LICENSE. Biohub-specific teacher, loss and inference helpers follow below.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn

CACHE_SCHEMA_VERSION = 1
CACHE_METADATA_KEY = "__metadata_json__"
REQUIRED_CACHE_ARRAYS = (
    "coords_src_grid",
    "coords_tgt_grid",
    "coords_src_physical",
    "coords_tgt_physical",
    "position_features_src",
    "position_features_tgt",
    "candidate_mask_src",
    "candidate_mask_tgt",
    "primary_features_src",
    "primary_features_tgt",
)

logger = logging.getLogger(__name__)


def _bin_init_exp(cutoff: float, n: int):
    return torch.exp(torch.linspace(0, math.log(cutoff + 1), n))


def _bin_init_linear(cutoff: float, n: int):
    return torch.linspace(-cutoff, cutoff, n)


def _rope_pos_embed_fourier1d_init(cutoff: float = 128, n: int = 32):
    # Maximum initial frequency is 1
    return torch.exp(torch.linspace(0, -math.log(cutoff), n)).unsqueeze(0).unsqueeze(0)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate pairs of scalars as 2d vectors by pi/2.
    Refer to eq 34 in https://arxiv.org/pdf/2104.09864.pdf.
    """
    x = x.unflatten(-1, (-1, 2))
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


class RotaryPositionalEncoding(nn.Module):
    def __init__(self, cutoffs: tuple[float] = (256,), n_pos: tuple[int] = (32,)):
        """Rotary positional encoding with given cutoff and number of frequencies for each dimension.
        number of dimension is inferred from the length of cutoffs and n_pos.

        see
        https://arxiv.org/pdf/2104.09864.pdf
        """
        super().__init__()
        assert len(cutoffs) == len(n_pos)
        if not all(n % 2 == 0 for n in n_pos):
            raise ValueError("n_pos must be even")

        self._n_dim = len(cutoffs)
        # theta in RoFormer https://arxiv.org/pdf/2104.09864.pdf
        self.freqs = nn.ParameterList(
            [
                nn.Parameter(_rope_pos_embed_fourier1d_init(cutoff, n // 2))
                for cutoff, n in zip(cutoffs, n_pos)
            ]
        )

    def get_co_si(self, coords: torch.Tensor):
        _B, _N, D = coords.shape
        assert D == len(self.freqs)
        co = torch.cat(
            tuple(
                torch.cos(0.5 * math.pi * x.unsqueeze(-1) * freq) / math.sqrt(len(freq))
                for x, freq in zip(coords.moveaxis(-1, 0), self.freqs)
            ),
            axis=-1,
        )
        si = torch.cat(
            tuple(
                torch.sin(0.5 * math.pi * x.unsqueeze(-1) * freq) / math.sqrt(len(freq))
                for x, freq in zip(coords.moveaxis(-1, 0), self.freqs)
            ),
            axis=-1,
        )

        return co, si

    def forward(self, q: torch.Tensor, k: torch.Tensor, coords: torch.Tensor):
        _B, _N, D = coords.shape
        _B, _H, _N, _C = q.shape

        if not D == self._n_dim:
            raise ValueError(f"coords must have {self._n_dim} dimensions, got {D}")

        co, si = self.get_co_si(coords)

        co = co.unsqueeze(1).repeat_interleave(2, dim=-1)
        si = si.unsqueeze(1).repeat_interleave(2, dim=-1)
        q2 = q * co + _rotate_half(q) * si
        k2 = k * co + _rotate_half(k) * si

        return q2, k2


def _pos_embed_fourier1d_init(cutoff: float = 256, n: int = 32, cutoff_start: float = 1):
    return (
        torch.exp(torch.linspace(-math.log(cutoff_start), -math.log(cutoff), n))
        .unsqueeze(0)
        .unsqueeze(0)
    )


class FeedForward(nn.Module):
    def __init__(self, d_model, expand: float = 2, bias: bool = True):
        super().__init__()
        self.fc1 = nn.Linear(d_model, int(d_model * expand))
        self.fc2 = nn.Linear(int(d_model * expand), d_model, bias=bias)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        cutoffs: tuple[float] = (256,),
        n_pos: tuple[int] = (32,),
        cutoffs_start=None,
    ):
        """Positional encoding with given cutoff and number of frequencies for each dimension.
        number of dimension is inferred from the length of cutoffs and n_pos.
        """
        super().__init__()
        if cutoffs_start is None:
            cutoffs_start = (1,) * len(cutoffs)

        assert len(cutoffs) == len(n_pos)
        self.freqs = nn.ParameterList(
            [
                nn.Parameter(_pos_embed_fourier1d_init(cutoff, n // 2))
                for cutoff, n, cutoff_start in zip(cutoffs, n_pos, cutoffs_start)
            ]
        )

    def forward(self, coords: torch.Tensor):
        _B, _N, D = coords.shape
        assert D == len(self.freqs)
        embed = torch.cat(
            tuple(
                torch.cat(
                    (
                        torch.sin(0.5 * math.pi * x.unsqueeze(-1) * freq),
                        torch.cos(0.5 * math.pi * x.unsqueeze(-1) * freq),
                    ),
                    axis=-1,
                )
                / math.sqrt(len(freq))
                for x, freq in zip(coords.moveaxis(-1, 0), self.freqs)
            ),
            axis=-1,
        )

        return embed


class RelativePositionalBias(nn.Module):
    def __init__(
        self,
        n_head: int,
        cutoff_spatial: float,
        cutoff_temporal: float,
        n_spatial: int = 32,
        n_temporal: int = 16,
    ):
        """Learnt relative positional bias to add to self-attention matrix.

        Spatial bins are exponentially spaced, temporal bins are linearly spaced.

        Args:
            n_head (int): Number of pos bias heads. Equal to number of attention heads
            cutoff_spatial (float): Maximum distance in space.
            cutoff_temporal (float): Maxium distance in time. Equal to window size of transformer.
            n_spatial (int, optional): Number of spatial bins.
            n_temporal (int, optional): Number of temporal bins in each direction. Should be equal to window size. Total = 2 * n_temporal + 1. Defaults to 16.
        """
        super().__init__()
        self._spatial_bins = _bin_init_exp(cutoff_spatial, n_spatial)
        self._temporal_bins = _bin_init_linear(cutoff_temporal, 2 * n_temporal + 1)
        self.register_buffer("spatial_bins", self._spatial_bins)
        self.register_buffer("temporal_bins", self._temporal_bins)
        self.n_spatial = n_spatial
        self.n_head = n_head
        self.bias = nn.Parameter(-0.5 + torch.rand((2 * n_temporal + 1) * n_spatial, n_head))

    def forward(self, coords: torch.Tensor):
        _B, _N, _D = coords.shape
        t = coords[..., 0]
        yx = coords[..., 1:]
        temporal_dist = t.unsqueeze(-1) - t.unsqueeze(-2)
        spatial_dist = torch.cdist(yx, yx)

        spatial_idx = torch.bucketize(spatial_dist, self.spatial_bins)
        torch.clamp_(spatial_idx, max=len(self.spatial_bins) - 1)
        temporal_idx = torch.bucketize(temporal_dist, self.temporal_bins)
        torch.clamp_(temporal_idx, max=len(self.temporal_bins) - 1)

        # do some index gymnastics such that backward is not super slow
        # https://discuss.pytorch.org/t/how-to-select-multiple-indexes-over-multiple-dimensions-at-the-same-time/98532/2
        idx = spatial_idx.flatten() + temporal_idx.flatten() * self.n_spatial
        bias = self.bias.index_select(0, idx).view((*spatial_idx.shape, self.n_head))
        # -> B, nH, N, N
        bias = bias.transpose(-1, 1)
        return bias


class RelativePositionalAttention(nn.Module):
    def __init__(
        self,
        coord_dim: int,
        embed_dim: int,
        n_head: int,
        cutoff_spatial: float = 256,
        cutoff_temporal: float = 16,
        n_spatial: int = 32,
        n_temporal: int = 16,
        dropout: float = 0.0,
        mode: Literal["bias", "rope", "none"] = "bias",
        attn_dist_mode: str = "v0",
    ):
        super().__init__()

        if not embed_dim % (2 * n_head) == 0:
            raise ValueError(
                f"embed_dim {embed_dim} must be divisible by 2 times n_head {2 * n_head}"
            )

        # qkv projection
        self.q_pro = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_pro = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_pro = nn.Linear(embed_dim, embed_dim, bias=True)

        # output projection
        self.proj = nn.Linear(embed_dim, embed_dim)
        # regularization
        self.dropout = dropout
        self.n_head = n_head
        self.embed_dim = embed_dim
        self.cutoff_spatial = cutoff_spatial
        self.attn_dist_mode = attn_dist_mode

        if mode == "bias" or mode is True:
            self.pos_bias = RelativePositionalBias(
                n_head=n_head,
                cutoff_spatial=cutoff_spatial,
                cutoff_temporal=cutoff_temporal,
                n_spatial=n_spatial,
                n_temporal=n_temporal,
            )
        elif mode == "rope":
            # each part needs to be divisible by 2
            n_split = 2 * (embed_dim // (2 * (coord_dim + 1) * n_head))

            self.rot_pos_enc = RotaryPositionalEncoding(
                cutoffs=((cutoff_temporal,) + (cutoff_spatial,) * coord_dim),
                n_pos=(embed_dim // n_head - coord_dim * n_split,) + (n_split,) * coord_dim,
            )
        elif mode == "none":
            pass
        elif mode is None or mode is False:
            logger.warning("attn_positional_bias is not set (None or False), no positional bias.")
            pass
        else:
            raise ValueError(f"Unknown mode {mode}")

        self._mode = mode

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        coords: torch.Tensor,
        padding_mask: torch.Tensor = None,
    ):
        B, N, D = query.size()
        q = self.q_pro(query)  # (B, N, D)
        k = self.k_pro(key)  # (B, N, D)
        v = self.v_pro(value)  # (B, N, D)
        # (B, nh, N, hs)
        k = k.view(B, N, self.n_head, D // self.n_head).transpose(1, 2)
        q = q.view(B, N, self.n_head, D // self.n_head).transpose(1, 2)
        v = v.view(B, N, self.n_head, D // self.n_head).transpose(1, 2)

        attn_mask = torch.zeros((B, self.n_head, N, N), device=query.device, dtype=q.dtype)

        # add negative value but not too large to keep mixed precision loss from becoming nan
        attn_ignore_val = -1e3

        # spatial cutoff
        yx = coords[..., 1:]
        spatial_dist = torch.cdist(yx, yx)
        spatial_mask = (spatial_dist > self.cutoff_spatial).unsqueeze(1)
        attn_mask.masked_fill_(spatial_mask, attn_ignore_val)

        # dont add positional bias to self-attention if coords is None
        if coords is not None:
            if self._mode == "bias":
                attn_mask = attn_mask + self.pos_bias(coords)
            elif self._mode == "rope":
                q, k = self.rot_pos_enc(q, k, coords)
            else:
                pass

            if self.attn_dist_mode == "v0":
                dist = torch.cdist(coords, coords, p=2)
                attn_mask += torch.exp(-0.1 * dist.unsqueeze(1))
            elif self.attn_dist_mode == "v1":
                attn_mask += torch.exp(-5 * spatial_dist.unsqueeze(1) / self.cutoff_spatial)
            else:
                raise ValueError(f"Unknown attn_dist_mode {self.attn_dist_mode}")

        # if given key_padding_mask = (B,N) then ignore those tokens (e.g. padding tokens)
        if padding_mask is not None:
            ignore_mask = torch.logical_or(
                padding_mask.unsqueeze(1), padding_mask.unsqueeze(2)
            ).unsqueeze(1)
            attn_mask.masked_fill_(ignore_mask, attn_ignore_val)

        # self.attn_mask = attn_mask.clone()

        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0
        )

        y = y.transpose(1, 2).contiguous().view(B, N, D)
        # output projection
        y = self.proj(y)

        return y


class EncoderLayer(nn.Module):
    def __init__(
        self,
        coord_dim: int = 2,
        d_model=256,
        num_heads=4,
        dropout=0.1,
        cutoff_spatial: int = 256,
        window: int = 16,
        positional_bias: Literal["bias", "rope", "none"] = "bias",
        positional_bias_n_spatial: int = 32,
        attn_dist_mode: str = "v0",
    ):
        super().__init__()
        self.positional_bias = positional_bias
        self.attn = RelativePositionalAttention(
            coord_dim,
            d_model,
            num_heads,
            cutoff_spatial=cutoff_spatial,
            n_spatial=positional_bias_n_spatial,
            cutoff_temporal=window,
            n_temporal=window,
            dropout=dropout,
            mode=positional_bias,
            attn_dist_mode=attn_dist_mode,
        )
        self.mlp = FeedForward(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        padding_mask: torch.Tensor = None,
    ):
        x = self.norm1(x)

        # setting coords to None disables positional bias
        a = self.attn(
            x,
            x,
            x,
            coords=coords if self.positional_bias else None,
            padding_mask=padding_mask,
        )

        x = x + a
        x = x + self.mlp(self.norm2(x))

        return x


class DecoderLayer(nn.Module):
    def __init__(
        self,
        coord_dim: int = 2,
        d_model=256,
        num_heads=4,
        dropout=0.1,
        window: int = 16,
        cutoff_spatial: int = 256,
        positional_bias: Literal["bias", "rope", "none"] = "bias",
        positional_bias_n_spatial: int = 32,
        attn_dist_mode: str = "v0",
    ):
        super().__init__()
        self.positional_bias = positional_bias
        self.attn = RelativePositionalAttention(
            coord_dim,
            d_model,
            num_heads,
            cutoff_spatial=cutoff_spatial,
            n_spatial=positional_bias_n_spatial,
            cutoff_temporal=window,
            n_temporal=window,
            dropout=dropout,
            mode=positional_bias,
            attn_dist_mode=attn_dist_mode,
        )

        self.mlp = FeedForward(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        coords: torch.Tensor,
        padding_mask: torch.Tensor = None,
    ):
        x = self.norm1(x)
        y = self.norm2(y)
        # cross attention
        # setting coords to None disables positional bias
        a = self.attn(
            x,
            y,
            y,
            coords=coords if self.positional_bias else None,
            padding_mask=padding_mask,
        )

        x = x + a
        x = x + self.mlp(self.norm3(x))

        return x


class TrackingTransformer(torch.nn.Module):
    def __init__(
        self,
        coord_dim: int = 3,
        feat_dim: int = 0,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dropout: float = 0.1,
        pos_embed_per_dim: int = 32,
        feat_embed_per_dim: int = 1,
        window: int = 6,
        spatial_pos_cutoff: int = 256,
        attn_positional_bias: Literal["bias", "rope", "none"] = "rope",
        attn_positional_bias_n_spatial: int = 16,
        causal_norm: Literal["none", "linear", "softmax", "quiet_softmax"] = "quiet_softmax",
        attn_dist_mode: str = "v0",
    ):
        super().__init__()

        self.config = dict(
            coord_dim=coord_dim,
            feat_dim=feat_dim,
            pos_embed_per_dim=pos_embed_per_dim,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            window=window,
            dropout=dropout,
            attn_positional_bias=attn_positional_bias,
            attn_positional_bias_n_spatial=attn_positional_bias_n_spatial,
            spatial_pos_cutoff=spatial_pos_cutoff,
            feat_embed_per_dim=feat_embed_per_dim,
            causal_norm=causal_norm,
            attn_dist_mode=attn_dist_mode,
        )

        # TODO remove, alredy present in self.config
        # self.window = window
        # self.feat_dim = feat_dim
        # self.coord_dim = coord_dim

        self.proj = nn.Linear(
            (1 + coord_dim) * pos_embed_per_dim + feat_dim * feat_embed_per_dim, d_model
        )
        self.norm = nn.LayerNorm(d_model)

        self.encoder = nn.ModuleList(
            [
                EncoderLayer(
                    coord_dim,
                    d_model,
                    nhead,
                    dropout,
                    window=window,
                    cutoff_spatial=spatial_pos_cutoff,
                    positional_bias=attn_positional_bias,
                    positional_bias_n_spatial=attn_positional_bias_n_spatial,
                    attn_dist_mode=attn_dist_mode,
                )
                for _ in range(num_encoder_layers)
            ]
        )
        self.decoder = nn.ModuleList(
            [
                DecoderLayer(
                    coord_dim,
                    d_model,
                    nhead,
                    dropout,
                    window=window,
                    cutoff_spatial=spatial_pos_cutoff,
                    positional_bias=attn_positional_bias,
                    positional_bias_n_spatial=attn_positional_bias_n_spatial,
                    attn_dist_mode=attn_dist_mode,
                )
                for _ in range(num_decoder_layers)
            ]
        )

        self.head_x = FeedForward(d_model)
        self.head_y = FeedForward(d_model)

        if feat_embed_per_dim > 1:
            self.feat_embed = PositionalEncoding(
                cutoffs=(1000,) * feat_dim,
                n_pos=(feat_embed_per_dim,) * feat_dim,
                cutoffs_start=(0.01,) * feat_dim,
            )
        else:
            self.feat_embed = nn.Identity()

        self.pos_embed = PositionalEncoding(
            cutoffs=(window,) + (spatial_pos_cutoff,) * coord_dim,
            n_pos=(pos_embed_per_dim,) * (1 + coord_dim),
        )

        # self.pos_embed = NoPositionalEncoding(d=pos_embed_per_dim * (1 + coord_dim))

    def forward(self, coords, features=None, padding_mask=None):
        assert coords.ndim == 3 and coords.shape[-1] in (3, 4)
        _B, _N, _D = coords.shape

        # disable padded coords (such that it doesnt affect minimum)
        if padding_mask is not None:
            coords = coords.clone()
            coords[padding_mask] = coords.max()

        # remove temporal offset
        min_time = coords[:, :, :1].min(dim=1, keepdims=True).values
        coords = torch.cat((coords[..., :1] - min_time, coords[..., 1:]), dim=-1)

        pos = self.pos_embed(coords)

        if features is None or features.numel() == 0:
            features = pos
        else:
            features = self.feat_embed(features)
            features = torch.cat((pos, features), axis=-1)

        features = self.proj(features)
        features = self.norm(features)

        x = features

        # encoder
        for enc in self.encoder:
            x = enc(x, coords=coords, padding_mask=padding_mask)

        y = features
        # decoder w cross attention
        for dec in self.decoder:
            y = dec(y, x, coords=coords, padding_mask=padding_mask)
            # y = dec(y, y, coords=coords, padding_mask=padding_mask)

        x = self.head_x(x)
        y = self.head_y(y)

        # outer product is the association matrix (logits)
        A = torch.einsum("bnd,bmd->bnm", x, y)

        return A

    def normalize_output(
        self,
        A: torch.FloatTensor,
        timepoints: torch.LongTensor,
        coords: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Apply (parental) softmax, or elementwise sigmoid.

        Args:
            A: Tensor of shape B, N, N
            timepoints: Tensor of shape B, N
            coords: Tensor of shape B, N, (time + n_spatial)
        """
        assert A.ndim == 3
        assert timepoints.ndim == 2
        assert coords.ndim == 3
        assert coords.shape[2] == 1 + self.config["coord_dim"]

        # spatial distances
        dist = torch.cdist(coords[:, :, 1:], coords[:, :, 1:])
        invalid = dist > self.config["spatial_pos_cutoff"]
        invalid = invalid | (timepoints.unsqueeze(1) == -1) | (timepoints.unsqueeze(2) == -1)

        if self.config["causal_norm"] == "none":
            # Spatially distant entries are set to zero
            A = torch.sigmoid(A)
            A[invalid] = 0
        else:
            return torch.stack(
                [
                    official_blockwise_causal_norm(
                        _A, _t, mode=self.config["causal_norm"], mask_invalid=_m
                    )
                    for _A, _t, _m in zip(A, timepoints, invalid)
                ]
            )
        return A

    def save(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        yaml.safe_dump(self.config, open(folder / "config.yaml", "w"))
        torch.save(self.state_dict(), folder / "model.pt")

    @staticmethod
    def create(config):
        model_classes = {
            "default": TrackingTransformer,
        }
        try:
            from trackastra_pretrained_feats import TrackingTransformerwPretrainedFeats

            PRETRAINED_FEATS_INSTALLED = True
        except ImportError:
            PRETRAINED_FEATS_INSTALLED = False
        if PRETRAINED_FEATS_INSTALLED:
            model_classes["pretrained_feats"] = TrackingTransformerwPretrainedFeats

        model_type = "pretrained_feats" if "pretrained_feat_dim" in config else "default"
        # TODO instead we could add explicit field in config to dispatch to different model classes rather than a train arg

        if model_type == "pretrained_feats" and not PRETRAINED_FEATS_INSTALLED:
            raise ImportError(
                "Model was trained with pretrained features, but trackastra_pretrained_feats is not installed. "
                "Please install it with `pip install trackastra[etultra]`."
            )

        return model_classes[model_type](**config)

    @classmethod
    def from_folder(cls, folder, map_location=None, args=None, checkpoint_path: str = "model.pt"):
        folder = Path(folder)

        config = yaml.load(open(folder / "config.yaml"), Loader=yaml.FullLoader)
        if args:
            args = vars(args)
            for k, v in config.items():
                errors = []
                if k in args:
                    if config[k] != args[k]:
                        errors.append(
                            f"Loaded model config {k}={config[k]}, but current argument"
                            f" {k}={args[k]}."
                        )
            if errors:
                raise ValueError("\n".join(errors))
        model = cls.create(config)

        # try:
        #     # Try to load from lightning checkpoint first
        #     v_folder = sorted((folder / "tb").glob("version_*"))[version]
        #     checkpoint = sorted((v_folder / "checkpoints").glob("*epoch*.ckpt"))[0]
        #     pl_state_dict = torch.load(checkpoint, map_location=map_location)[
        #         "state_dict"
        #     ]
        #     state_dict = OrderedDict()

        #     # Hack
        #     for k, v in pl_state_dict.items():
        #         if k.startswith("model."):
        #             state_dict[k[6:]] = v
        #         else:
        #             raise ValueError(f"Unexpected key {k} in state_dict")

        #     model.load_state_dict(state_dict)
        #     logger.info(f"Loaded model from {checkpoint}")
        # except:
        #     # Default: Load manually saved model (legacy)

        fpath = folder / checkpoint_path
        logger.info(f"Loading model state from {fpath}")

        state = torch.load(fpath, map_location=map_location, weights_only=True)
        # if state is a checkpoint, we have to extract state_dict
        if "state_dict" in state:
            state = state["state_dict"]
            state = OrderedDict((k[6:], v) for k, v in state.items() if k.startswith("model."))
        model.load_state_dict(state)

        return model


# Adapted from exp016 frozen_tracker.py; fixed cache and GEFF contracts.


@dataclass(frozen=True)
class FrameAnnotation:
    node_ids: np.ndarray
    coords_physical: np.ndarray


@dataclass(frozen=True)
class AnnotationGraph:
    frames: dict[int, FrameAnnotation]
    edges: frozenset[tuple[int, int]]
    content_sha256: str
    outgoing_edges: dict[int, tuple[int, ...]] | None = None


def json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_schema(arrays: dict[str, np.ndarray]) -> list[dict[str, object]]:
    return [
        {
            "name": name,
            "dtype": np.asarray(arrays[name]).dtype.str,
            "shape": list(np.asarray(arrays[name]).shape),
        }
        for name in sorted(arrays)
    ]


def array_content_sha256(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for item in array_schema(arrays):
        name = str(item["name"])
        array = np.ascontiguousarray(arrays[name])
        digest.update(json.dumps(item, separators=(",", ":"), sort_keys=True).encode())
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def validate_cache_summary(path: Path, cache_cfg: dict[str, Any]) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"cache summary must be an object: {path}")
    recorded_sha = value.get("summary_sha256")
    unsigned = dict(value)
    unsigned.pop("summary_sha256", None)
    observed_sha = json_sha256(unsigned)
    if recorded_sha != observed_sha:
        raise ValueError(f"cache summary self-check failed: {path}")
    expected = {
        "schema_version": int(cache_cfg["schema_version"]),
        "dataset_count": int(cache_cfg["expected_dataset_count"]),
        "window_count": int(cache_cfg["expected_window_count"]),
        "cache_identity_sha256": str(cache_cfg["identity_sha256"]),
        "summary_sha256": str(cache_cfg["summary_sha256"]),
    }
    observed = {key: value.get(key) for key in expected}
    if observed != expected:
        raise ValueError({"cache_summary_mismatch": {"expected": expected, "actual": observed}})
    return value


def discover_cache_paths(cache_root: Path, cache_cfg: dict[str, Any]) -> list[Path]:
    paths = sorted(cache_root.glob("*/*.npz"))
    samples = {path.parent.name for path in paths}
    expected_windows = int(cache_cfg["expected_window_count"])
    expected_samples = int(cache_cfg["expected_dataset_count"])
    if len(paths) != expected_windows or len(samples) != expected_samples:
        raise ValueError(
            {
                "cache_coverage_mismatch": {
                    "expected_samples": expected_samples,
                    "actual_samples": len(samples),
                    "expected_windows": expected_windows,
                    "actual_windows": len(paths),
                }
            }
        )
    return paths


def cache_identity_record(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as saved:
        if CACHE_METADATA_KEY not in saved.files:
            raise ValueError(f"cache metadata missing: {path}")
        metadata = json.loads(saved[CACHE_METADATA_KEY].tobytes().decode("utf-8"))
    if not isinstance(metadata, dict):
        raise TypeError(f"cache metadata must be an object: {path}")
    schema = metadata.get("array_schema")
    if not isinstance(schema, list):
        raise ValueError(f"cache array schema missing: {path}")
    schema_by_name = {str(item.get("name")): item for item in schema}
    feature_names = (
        "primary_features_src",
        "primary_features_tgt",
        "secondary_features_src",
        "secondary_features_tgt",
    )
    if any(name not in schema_by_name for name in feature_names):
        raise ValueError(f"cache feature schema incomplete: {path}")
    frames = metadata.get("window_frames")
    if metadata.get("dataset") != path.parent.name or not isinstance(frames, list):
        raise ValueError(f"cache identity and path differ: {path}")
    return {
        "dataset": path.parent.name,
        "window_frames": [int(value) for value in frames],
        "candidate_count_src": int(schema_by_name["candidate_mask_src"]["shape"][0]),
        "candidate_count_tgt": int(schema_by_name["candidate_mask_tgt"]["shape"][0]),
        "feature_values": int(
            sum(np.prod(schema_by_name[name]["shape"], dtype=np.int64) for name in feature_names)
        ),
        "cache_bytes": int(path.stat().st_size),
        "cache_schema_sha256": json_sha256(schema),
        "cache_content_sha256": str(metadata.get("array_content_sha256")),
    }


def recompute_cache_identity_sha256(paths: list[Path]) -> str:
    return json_sha256([cache_identity_record(path) for path in sorted(paths)])


def embryo_id(sample_name: str) -> str:
    embryo, separator, _ = sample_name.partition("_")
    if not separator or not embryo:
        raise ValueError(f"sample name has no embryo prefix: {sample_name}")
    return embryo


def build_embryo_splits(
    sample_names: list[str],
    outer_specs: list[dict[str, Any]],
    split_seed: int,
) -> list[dict[str, Any]]:
    sample_set = set(sample_names)
    if len(sample_set) != len(sample_names):
        raise ValueError("duplicate sample names")
    records: list[dict[str, Any]] = []
    outer_evaluation_seen: set[str] = set()
    for spec in outer_specs:
        fold = int(spec["fold"])
        train_embryo = str(spec["train_embryo"])
        evaluation_embryo = str(spec["evaluation_embryo"])
        train_pool = sorted(name for name in sample_names if embryo_id(name) == train_embryo)
        evaluation = sorted(name for name in sample_names if embryo_id(name) == evaluation_embryo)
        shuffled = list(train_pool)
        random.Random(split_seed).shuffle(shuffled)
        n_internal = max(1, len(shuffled) // 10)
        internal_validation = shuffled[:n_internal]
        gradient_update = shuffled[n_internal:]
        record = {
            "fold": fold,
            "train_embryo": train_embryo,
            "evaluation_embryo": evaluation_embryo,
            "gradient_update": gradient_update,
            "internal_validation": internal_validation,
            "outer_evaluation": evaluation,
        }
        expected = {
            "gradient_update": int(spec["expected_train_sample_count"]),
            "internal_validation": int(spec["expected_internal_validation_sample_count"]),
            "outer_evaluation": int(spec["expected_evaluation_sample_count"]),
        }
        actual = {key: len(record[key]) for key in expected}
        if actual != expected:
            raise ValueError({"fold": fold, "expected": expected, "actual": actual})
        train_names = set(gradient_update) | set(internal_validation)
        evaluation_names = set(evaluation)
        if set(gradient_update) & set(internal_validation):
            raise ValueError(f"fold {fold} internal split overlaps")
        if train_names != set(train_pool):
            raise ValueError(f"fold {fold} internal split does not cover the training embryo")
        if train_names & evaluation_names:
            raise ValueError(f"fold {fold} outer evaluation leaked into training")
        if outer_evaluation_seen & evaluation_names:
            raise ValueError("an outer evaluation sample appears in multiple folds")
        outer_evaluation_seen.update(evaluation_names)
        records.append(record)
    if outer_evaluation_seen != sample_set:
        raise ValueError("outer evaluation folds do not cover every sample exactly once")
    return records


def graph_from_geff(path: Path) -> Any:
    import tracksdata as td

    graph = td.graph.IndexedRXGraph.from_geff(path)
    return graph[0] if isinstance(graph, tuple) else graph


def load_annotation_graph(
    path: Path,
    voxel_scale_zyx_um: tuple[float, float, float],
) -> AnnotationGraph:
    graph = graph_from_geff(path)
    scale = np.asarray(voxel_scale_zyx_um, dtype=np.float64)
    rows_by_frame: dict[int, list[tuple[int, np.ndarray]]] = {}
    canonical_nodes: list[list[int | float]] = []
    seen: set[int] = set()
    for row in graph.node_attrs().iter_rows(named=True):
        node_id = int(row["node_id"])
        if node_id in seen:
            raise ValueError(f"duplicate node id in {path}: {node_id}")
        seen.add(node_id)
        frame = int(row["t"])
        raw = np.asarray([row["z"], row["y"], row["x"]], dtype=np.float64)
        physical = raw * scale
        rows_by_frame.setdefault(frame, []).append((node_id, physical))
        canonical_nodes.append([node_id, frame, *map(float, raw)])
    edges = frozenset(
        (int(row["source_id"]), int(row["target_id"]))
        for row in graph.edge_attrs().iter_rows(named=True)
    )
    if any(source not in seen or target not in seen for source, target in edges):
        raise ValueError(f"dangling edge in {path}")
    frames: dict[int, FrameAnnotation] = {}
    for frame, entries in rows_by_frame.items():
        entries.sort(key=lambda item: item[0])
        frames[frame] = FrameAnnotation(
            node_ids=np.asarray([item[0] for item in entries], dtype=np.int64),
            coords_physical=np.asarray([item[1] for item in entries], dtype=np.float32),
        )
    canonical = {
        "nodes": sorted(canonical_nodes, key=lambda row: int(row[0])),
        "edges": [list(edge) for edge in sorted(edges)],
    }
    outgoing: dict[int, list[int]] = {}
    for source, target in edges:
        outgoing.setdefault(source, []).append(target)
    return AnnotationGraph(
        frames=frames,
        edges=edges,
        content_sha256=json_sha256(canonical),
        outgoing_edges={source: tuple(sorted(targets)) for source, targets in outgoing.items()},
    )


def greedy_match_candidates(
    candidate_coords_physical: np.ndarray,
    gt_node_ids: np.ndarray,
    gt_coords_physical: np.ndarray,
    max_distance_um: float,
) -> tuple[np.ndarray, np.ndarray]:
    candidates = np.asarray(candidate_coords_physical, dtype=np.float64)
    gt_ids = np.asarray(gt_node_ids, dtype=np.int64)
    gt_coords = np.asarray(gt_coords_physical, dtype=np.float64)
    if candidates.ndim != 2 or candidates.shape[1:] != (3,):
        raise ValueError("candidate physical coordinates must have shape (N, 3)")
    if gt_coords.ndim != 2 or gt_coords.shape[1:] != (3,) or len(gt_coords) != len(gt_ids):
        raise ValueError("GT ids and coordinates have incompatible shapes")
    matched_ids = np.full(len(candidates), -1, dtype=np.int64)
    matched_distances = np.full(len(candidates), np.nan, dtype=np.float32)
    if len(candidates) == 0 or len(gt_ids) == 0:
        return matched_ids, matched_distances
    distances = np.linalg.norm(candidates[:, None, :] - gt_coords[None, :, :], axis=2)
    nearest_index = distances.argmin(axis=1)
    nearest_distance = distances[np.arange(len(candidates)), nearest_index]
    order = np.argsort(nearest_distance, kind="stable")
    gt_taken = np.zeros(len(gt_ids), dtype=bool)
    for candidate_index in order:
        distance = float(nearest_distance[candidate_index])
        if distance > max_distance_um:
            break
        gt_index = int(nearest_index[candidate_index])
        if gt_taken[gt_index]:
            continue
        gt_taken[gt_index] = True
        matched_ids[candidate_index] = gt_ids[gt_index]
        matched_distances[candidate_index] = distance
    return matched_ids, matched_distances


def _read_cache_payload(
    path: Path,
    *,
    load_all_arrays: bool,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as saved:
        if CACHE_METADATA_KEY not in saved.files:
            raise ValueError(f"cache metadata missing: {path}")
        metadata = json.loads(saved[CACHE_METADATA_KEY].tobytes().decode("utf-8"))
        selected_names = (
            [name for name in saved.files if name != CACHE_METADATA_KEY]
            if load_all_arrays
            else [name for name in REQUIRED_CACHE_ARRAYS if name in saved.files]
        )
        arrays = {name: np.ascontiguousarray(saved[name]) for name in selected_names}
    if not isinstance(metadata, dict):
        raise TypeError(f"cache metadata must be an object: {path}")
    return arrays, metadata


def validate_window_cache(
    path: Path,
    *,
    feature_channels: int,
    expected_primary_checkpoint_sha256: str,
    verify_content: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays, metadata = _read_cache_payload(path, load_all_arrays=verify_content)
    missing = sorted(set(REQUIRED_CACHE_ARRAYS) - set(arrays))
    if missing:
        raise ValueError({"cache_arrays_missing": missing, "path": str(path)})
    expected_metadata = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "experiment": "exp015_oracle_stage_limits",
        "dataset": path.parent.name,
        "primary_checkpoint_sha256": expected_primary_checkpoint_sha256,
    }
    observed_metadata = {key: metadata.get(key) for key in expected_metadata}
    if observed_metadata != expected_metadata:
        raise ValueError(
            {
                "cache_metadata_mismatch": {
                    "expected": expected_metadata,
                    "actual": observed_metadata,
                    "path": str(path),
                }
            }
        )
    frames = metadata.get("window_frames")
    if not isinstance(frames, list) or len(frames) != 2 or int(frames[1]) != int(frames[0]) + 1:
        raise ValueError(f"cache window is not an adjacent frame pair: {path}")
    expected_name = f"{int(frames[0]):06d}_{int(frames[1]):06d}.npz"
    if path.name != expected_name:
        raise ValueError(f"cache filename and window metadata differ: {path}")
    for side in ("src", "tgt"):
        mask = np.asarray(arrays[f"candidate_mask_{side}"], dtype=bool)
        count = len(mask)
        shapes = {
            "grid": np.asarray(arrays[f"coords_{side}_grid"]).shape,
            "physical": np.asarray(arrays[f"coords_{side}_physical"]).shape,
            "position": np.asarray(arrays[f"position_features_{side}"]).shape,
            "primary": np.asarray(arrays[f"primary_features_{side}"]).shape,
        }
        expected_shapes = {
            "grid": (count, 3),
            "physical": (count, 3),
            "position": (count, 32),
            "primary": (count, feature_channels),
        }
        if shapes != expected_shapes:
            raise ValueError(
                {"cache_shape_mismatch": side, "expected": expected_shapes, "actual": shapes}
            )
        if not mask.all():
            raise ValueError(f"cache contains empty or padded candidate rows: {path} {side}")
        for key in (
            f"coords_{side}_grid",
            f"coords_{side}_physical",
            f"position_features_{side}",
            f"primary_features_{side}",
        ):
            if not np.isfinite(arrays[key]).all():
                raise ValueError(f"cache contains non-finite values: {path} {key}")
    recorded_schema = {str(item.get("name")): item for item in metadata.get("array_schema", [])}
    actual_schema = {str(item["name"]): item for item in array_schema(arrays)}
    for name in REQUIRED_CACHE_ARRAYS:
        if recorded_schema.get(name) != actual_schema.get(name):
            raise ValueError(f"cache array schema mismatch: {path} {name}")
    if verify_content and metadata.get("array_content_sha256") != array_content_sha256(arrays):
        raise ValueError(f"cache array content mismatch: {path}")
    return arrays, metadata


def build_legacy_edge_target(
    source_matches: np.ndarray,
    target_matches: np.ndarray,
    annotated_edges: frozenset[tuple[int, int]] | set[tuple[int, int]],
    outgoing_edges: dict[int, tuple[int, ...]] | None = None,
) -> np.ndarray:
    source = np.asarray(source_matches, dtype=np.int64)
    target = np.asarray(target_matches, dtype=np.int64)
    matrix = np.zeros((len(source), len(target)), dtype=np.float32)
    target_columns: dict[int, list[int]] = {}
    for column, node_id in enumerate(target):
        if node_id >= 0:
            target_columns.setdefault(int(node_id), []).append(column)
    if outgoing_edges is None:
        temporary: dict[int, list[int]] = {}
        for edge_source, edge_target in annotated_edges:
            temporary.setdefault(int(edge_source), []).append(int(edge_target))
        outgoing_edges = {key: tuple(values) for key, values in temporary.items()}
    for row, source_id in enumerate(source):
        if source_id < 0:
            continue
        for edge_target in outgoing_edges.get(int(source_id), ()):
            for column in target_columns.get(edge_target, []):
                matrix[row, column] = 1.0
    return matrix


def legacy_active_pair_mask(target: np.ndarray) -> np.ndarray:
    matrix = np.asarray(target)
    if matrix.ndim != 2:
        raise ValueError("target must be a matrix")
    active_rows = matrix.sum(axis=1) > 0
    active_cols = matrix.sum(axis=0) > 0
    return active_rows[:, None] | active_cols[None, :]


def filter_nonempty_gt_window_paths(
    paths: list[Path],
    annotations: dict[str, AnnotationGraph],
) -> tuple[list[Path], dict[str, Any]]:
    """Match the public trainer's exclusion of windows containing an empty GT frame."""
    eligible: list[Path] = []
    skipped: list[dict[str, Any]] = []
    skipped_by_sample: dict[str, int] = {}
    for path in paths:
        sample = path.parent.name
        if sample not in annotations:
            raise ValueError(f"annotation is missing for cache sample: {path}")
        parts = path.stem.split("_")
        if len(parts) != 2:
            raise ValueError(f"cache filename does not encode two frames: {path}")
        source_frame, target_frame = map(int, parts)
        if target_frame != source_frame + 1:
            raise ValueError(f"cache filename is not an adjacent frame pair: {path}")
        missing_frames = [
            frame
            for frame in (source_frame, target_frame)
            if frame not in annotations[sample].frames
        ]
        if not missing_frames:
            eligible.append(path)
            continue
        skipped.append(
            {
                "sample": sample,
                "window_frames": [source_frame, target_frame],
                "empty_gt_frames": missing_frames,
            }
        )
        skipped_by_sample[sample] = skipped_by_sample.get(sample, 0) + 1
    audit = {
        "policy": "skip_window_if_any_frame_has_zero_gt_nodes",
        "public_source_function": "get_window_data",
        "input_window_count": len(paths),
        "eligible_window_count": len(eligible),
        "skipped_window_count": len(skipped),
        "skipped_by_sample": dict(sorted(skipped_by_sample.items())),
        "skipped_windows": skipped,
    }
    return eligible, audit


def extract_public_tracker_state(full_state: dict[str, Any]) -> dict[str, Any]:
    prefixes = ("transformer.", "module.transformer.")
    for prefix in prefixes:
        selected = {
            key[len(prefix) :]: value for key, value in full_state.items() if key.startswith(prefix)
        }
        if selected:
            return selected
    expected_tracker_roots = {"proj", "norm_in", "blocks", "norm_out", "pair_mlp"}
    observed_roots = {key.split(".", 1)[0] for key in full_state}
    if expected_tracker_roots.issubset(observed_roots):
        return dict(full_state)
    raise ValueError("checkpoint does not contain a SimpleNodeTransformer state")


def canonical_state_sha256(state: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        descriptor = {
            "name": name,
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
        }
        digest.update(json.dumps(descriptor, separators=(",", ":"), sort_keys=True).encode())
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Official blockwise normalization retained for source parity checks.


def blockwise_sum(A: torch.Tensor, timepoints: torch.Tensor, dim: int = 0, reduce: str = "sum"):
    if not A.shape[dim] == len(timepoints):
        raise ValueError(
            f"Dimension {dim} of A ({A.shape[dim]}) must match length of timepoints"
            f" ({len(timepoints)})"
        )

    A = A.transpose(dim, 0)

    if len(timepoints) == 0:
        logger.warning("Empty timepoints in block_sum. Returning zero tensor.")
        return A
    # -1 is the filling value for padded/invalid timepoints
    min_t = timepoints[timepoints >= 0]
    if len(min_t) == 0:
        logger.warning("All timepoints are -1 in block_sum. Returning zero tensor.")
        return A

    min_t = min_t.min()
    # after that, valid timepoints start with 1 (padding timepoints will be mapped to 0)
    ts = torch.clamp(timepoints - min_t + 1, min=0)
    index = ts.unsqueeze(1).expand(-1, len(ts))
    blocks = ts.max().long() + 1
    out = torch.zeros((blocks, A.shape[1]), device=A.device, dtype=A.dtype)
    out = torch.scatter_reduce(out, 0, index, A, reduce=reduce)
    B = out[ts]
    B = B.transpose(0, dim)

    return B


def official_blockwise_causal_norm(
    A: torch.Tensor,
    timepoints: torch.Tensor,
    mode: str = "quiet_softmax",
    mask_invalid: torch.BoolTensor = None,
    eps: float = 1e-6,
):
    """Normalization over the causal dimension of A.

    For each block of constant timepoints, normalize the corresponding block of A
    such that the sum over the causal dimension is 1.

    Args:
        A (torch.Tensor): input tensor
        timepoints (torch.Tensor): timepoints for each element in the causal dimension
        mode: normalization mode.
            `linear`: Simple linear normalization.
            `softmax`: Apply exp to A before normalization.
            `quiet_softmax`: Apply exp to A before normalization, and add 1 to the denominator of each row/column.
        mask_invalid: Values that should not influence the normalization.
        eps (float, optional): epsilon for numerical stability.
    """
    assert A.ndim == 2 and A.shape[0] == A.shape[1]
    A = A.clone()

    if mode in ("softmax", "quiet_softmax"):
        # Subtract max for numerical stability
        # https://stats.stackexchange.com/questions/338285/how-does-the-subtraction-of-the-logit-maximum-improve-learning
        # TODO test without this subtraction

        if mask_invalid is not None:
            assert mask_invalid.shape == A.shape
            A[mask_invalid] = -torch.inf
        # TODO set to min, then to 0 after exp

        # Blockwise max
        with torch.no_grad():
            ma0 = blockwise_sum(A, timepoints, dim=0, reduce="amax")
            ma1 = blockwise_sum(A, timepoints, dim=1, reduce="amax")

        u0 = torch.exp(A - ma0)
        u1 = torch.exp(A - ma1)

    elif mode == "linear":
        A = torch.sigmoid(A)
        if mask_invalid is not None:
            assert mask_invalid.shape == A.shape
            A[mask_invalid] = 0

        u0, u1 = A, A
        ma0 = ma1 = 0
    else:
        raise NotImplementedError(f"Mode {mode} not implemented")

    # get block boundaries and normalize within blocks
    # bounds = _bounds_from_timepoints(timepoints)
    # u0_sum = _blockwise_sum_with_bounds(u0, bounds, dim=0) + eps
    # u1_sum = _blockwise_sum_with_bounds(u1, bounds, dim=1) + eps

    u0_sum = blockwise_sum(u0, timepoints, dim=0) + eps
    u1_sum = blockwise_sum(u1, timepoints, dim=1) + eps

    if mode == "quiet_softmax":
        # Add 1 to the denominator of the softmax. With this, the softmax outputs can be all 0, if the logits are all negative.
        # If the logits are positive, the softmax outputs will sum to 1.
        # Trick: With maximum subtraction, this is equivalent to adding 1 to the denominator
        u0_sum += torch.exp(-ma0)
        u1_sum += torch.exp(-ma1)

    mask0 = timepoints.unsqueeze(0) > timepoints.unsqueeze(1)
    # mask1 = timepoints.unsqueeze(0) < timepoints.unsqueeze(1)
    # Entries with t1 == t2 are always masked out in final loss
    mask1 = ~mask0

    # blockwise diagonal will be normalized along dim=0
    res = mask0 * u0 / u0_sum + mask1 * u1 / u1_sum
    res = torch.clamp(res, 0, 1)

    return res


# Biohub adaptation. All times below are actual frame indices; feature sources remain
# the two-frame caches that produced them.


def load_video_cache(
    paths: list[Path], *, feature_channels: int, checkpoint_sha256: str
) -> dict[str, Any]:
    if not paths:
        raise ValueError("video has no adjacent cache windows")
    pairs = []
    sample = paths[0].parent.name
    content_records = []
    for path in sorted(paths):
        if path.parent.name != sample:
            raise ValueError("a video cache contains multiple samples")
        arrays, metadata = validate_window_cache(
            path,
            feature_channels=feature_channels,
            expected_primary_checkpoint_sha256=checkpoint_sha256,
            verify_content=True,
        )
        frames = tuple(int(frame) for frame in metadata["window_frames"])
        if pairs and frames != (pairs[-1]["frames"][1], pairs[-1]["frames"][1] + 1):
            raise ValueError(f"non-consecutive pair cache: {path}")
        if pairs:
            prev = pairs[-1]["arrays"]
            for side, left, right in (
                ("candidate ids", "candidate_ids_tgt", "candidate_ids_src"),
                ("grid coordinates", "coords_tgt_grid", "coords_src_grid"),
                ("physical coordinates", "coords_tgt_physical", "coords_src_physical"),
            ):
                if not np.array_equal(prev[left], arrays[right]):
                    raise ValueError(f"overlapping {side} disagree at frame {frames[0]}")
        pairs.append({"frames": frames, "arrays": arrays, "path": path})
        content_records.append({"pair": frames, "sha256": str(metadata["array_content_sha256"])})
    frames = [pairs[0]["frames"][0]] + [pair["frames"][1] for pair in pairs]
    registry = {}
    for index, frame in enumerate(frames):
        pair = pairs[min(index, len(pairs) - 1)]
        side = "src" if index < len(pairs) else "tgt"
        arrays = pair["arrays"]
        ids = np.asarray(arrays[f"candidate_ids_{side}"], dtype=np.int64)
        if len(ids) != len(set(ids.tolist())):
            raise ValueError(f"duplicate candidate id in {sample} frame {frame}")
        registry[frame] = {
            "ids": ids,
            "physical": np.asarray(arrays[f"coords_{side}_physical"], dtype=np.float32),
            "grid": np.asarray(arrays[f"coords_{side}_grid"], dtype=np.float32),
        }
    return {
        "sample": sample,
        "pairs": pairs,
        "frames": frames,
        "registry": registry,
        "content_identity_sha256": json_sha256(content_records),
    }


def context_window_starts(frame_count: int, window_size: int = 6) -> list[int]:
    if frame_count < 2:
        return []
    return list(range(max(1, frame_count - window_size + 1)))


def matched_candidate_frames(
    video: dict[str, Any], annotation: AnnotationGraph, max_distance_um: float
) -> dict[int, np.ndarray]:
    matches = {}
    for frame in video["frames"]:
        candidate = video["registry"][frame]
        gt = annotation.frames.get(frame)
        if gt is None:
            matches[frame] = np.full(len(candidate["ids"]), -1, dtype=np.int64)
        else:
            matches[frame], _ = greedy_match_candidates(
                candidate["physical"], gt.node_ids, gt.coords_physical, max_distance_um
            )
    return matches


def _geff_parent_map(
    annotation: AnnotationGraph,
) -> tuple[dict[int, int], dict[int, int], dict[int, int]]:
    node_times = {
        int(node_id): int(frame)
        for frame, nodes in annotation.frames.items()
        for node_id in nodes.node_ids
    }
    parents: dict[int, int] = {}
    children: dict[int, int] = {}
    for parent, child in annotation.edges:
        if parent not in node_times or child not in node_times:
            raise ValueError("GEFF edge has a missing endpoint")
        if node_times[child] <= node_times[parent]:
            raise ValueError("GEFF edge is simultaneous or backward")
        if child in parents:
            raise ValueError("GEFF child has multiple parents")
        parents[child] = parent
        children[parent] = children.get(parent, 0) + 1
    return node_times, parents, children


def build_sparse_teacher(
    times: np.ndarray,
    physical: np.ndarray,
    matched_ids: np.ndarray,
    annotation: AnnotationGraph,
    *,
    cutoff_um: float,
) -> dict[str, np.ndarray | dict[str, int]]:
    """Direct BCE only where a detected true ancestor and wrong matched parents are known."""
    times = np.asarray(times, dtype=np.int64)
    physical = np.asarray(physical, dtype=np.float32)
    matched_ids = np.asarray(matched_ids, dtype=np.int64)
    n = len(times)
    if physical.shape != (n, 3) or matched_ids.shape != (n,):
        raise ValueError("teacher token arrays have different lengths")
    node_times, parent_map, child_count = _geff_parent_map(annotation)
    target = np.zeros((n, n), dtype=np.float32)
    mask = np.zeros((n, n), dtype=bool)
    weight = np.zeros((n, n), dtype=np.float32)
    stats = {
        "positive_dt1": 0,
        "positive_dt2": 0,
        "known_negative": 0,
        "division_positive": 0,
        "unknown_pairs": 0,
        "out_of_cutoff_positive": 0,
    }
    for child in range(n):
        gt_child = int(matched_ids[child])
        if gt_child < 0:
            continue
        ancestor = gt_child
        path_divides = False
        for hop in (1, 2):
            previous = parent_map.get(ancestor)
            if previous is None or node_times[ancestor] != node_times[previous] + 1:
                break
            path_divides |= child_count.get(previous, 0) >= 2
            ancestor = previous
            frame = int(times[child] - hop)
            if node_times[ancestor] != frame:
                break
            parent_rows = np.flatnonzero(times == frame)
            true_rows = parent_rows[matched_ids[parent_rows] == ancestor]
            if len(true_rows) == 0:
                continue
            if len(true_rows) != 1:
                raise ValueError("candidate matching is not one-to-one")
            correct = int(true_rows[0])
            if np.linalg.norm(physical[correct] - physical[child]) > cutoff_um:
                stats["out_of_cutoff_positive"] += 1
                continue
            for parent in parent_rows:
                gt_parent = int(matched_ids[parent])
                if gt_parent < 0:
                    continue
                if np.linalg.norm(physical[parent] - physical[child]) > cutoff_um:
                    continue
                mask[parent, child] = True
                if gt_parent == ancestor:
                    target[parent, child] = 1.0
                    weight[parent, child] = 11.0 if path_divides else 2.0
                    stats[f"positive_dt{hop}"] += 1
                    stats["division_positive"] += int(path_divides)
                else:
                    weight[parent, child] = 1.0
                    stats["known_negative"] += 1
    valid_time = (times[None, :] - times[:, None] >= 1) & (times[None, :] - times[:, None] <= 2)
    stats["unknown_pairs"] = int(np.count_nonzero(valid_time & ~mask))
    if stats["out_of_cutoff_positive"]:
        raise ValueError({"positive_association_outside_spatial_cutoff": stats})
    return {"target": target, "mask": mask, "weight": weight, "stats": stats}


def build_context_window(
    video: dict[str, Any],
    start_index: int,
    *,
    window_size: int = 6,
    annotation: AnnotationGraph | None = None,
    matches: dict[int, np.ndarray] | None = None,
    cutoff_um: float = 256.0,
) -> dict[str, Any]:
    frames = video["frames"]
    length = min(window_size, len(frames))
    if start_index < 0 or start_index + length > len(frames):
        raise IndexError("context window is outside the video")
    window_frames = frames[start_index : start_index + length]
    chunks = []
    source_records = []
    for offset, frame in enumerate(window_frames):
        index = start_index + offset
        pair_index = index if offset < length - 1 else index - 1
        pair = video["pairs"][pair_index]
        side = "src" if offset < length - 1 else "tgt"
        registry = video["registry"][frame]
        candidate_ids = np.asarray(pair["arrays"][f"candidate_ids_{side}"], dtype=np.int64)
        if not np.array_equal(candidate_ids, registry["ids"]):
            raise ValueError("window feature candidate IDs disagree with video registry")
        chunks.append(
            {
                "time": np.full(len(candidate_ids), frame, dtype=np.int64),
                "ids": candidate_ids,
                "physical": registry["physical"],
                "grid": registry["grid"],
                "features": np.asarray(
                    pair["arrays"][f"primary_features_{side}"], dtype=np.float32
                ),
                "matches": matches[frame]
                if matches is not None
                else np.full(len(candidate_ids), -1),
            }
        )
        source_records.append({"frame": frame, "pair": pair["frames"], "side": side})
    result = {
        "sample": video["sample"],
        "frames": window_frames,
        "times": np.concatenate([item["time"] for item in chunks]),
        "candidate_ids": np.concatenate([item["ids"] for item in chunks]),
        "physical": np.concatenate([item["physical"] for item in chunks]),
        "grid": np.concatenate([item["grid"] for item in chunks]),
        "features": np.concatenate([item["features"] for item in chunks]),
        "source_records": source_records,
        "frame_counts": [len(item["ids"]) for item in chunks],
    }
    if len(result["times"]):
        result["coords"] = np.concatenate(
            ((result["times"] - window_frames[0])[:, None], result["physical"]), axis=1
        ).astype(np.float32)
    else:
        result["coords"] = np.empty((0, 4), dtype=np.float32)
    if annotation is not None:
        result["teacher"] = build_sparse_teacher(
            result["times"],
            result["physical"],
            np.concatenate([item["matches"] for item in chunks]),
            annotation,
            cutoff_um=cutoff_um,
        )
    return result


def parental_softmax(
    logits: torch.Tensor, times: torch.Tensor, physical: torch.Tensor, *, cutoff_um: float
) -> torch.Tensor:
    """Normalize each child over one parent frame plus a fixed zero-logit abstention."""
    return parental_log_probabilities(logits, times, physical, cutoff_um=cutoff_um).exp()


def parental_log_probabilities(
    logits: torch.Tensor,
    times: torch.Tensor,
    physical: torch.Tensor,
    *,
    cutoff_um: float,
    include_complement: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Keep log probabilities for a stable BCE, including near-certain mistakes."""
    if logits.ndim != 2 or logits.shape[0] != logits.shape[1]:
        raise ValueError("association logits must be square")
    n = logits.shape[0]
    if times.shape != (n,) or physical.shape != (n, 3):
        raise ValueError("time and physical coordinates must match logits")
    log_probs = torch.full_like(logits, -torch.inf, dtype=torch.float32)
    log_complements = torch.zeros_like(log_probs) if include_complement else None
    if n == 0:
        return (log_probs, log_complements) if include_complement else log_probs
    values = logits.float()
    for child_frame in torch.unique(times).tolist():
        children = torch.where(times == child_frame)[0]
        for hop in (1, 2):
            parents = torch.where(times == child_frame - hop)[0]
            if not len(parents) or not len(children):
                continue
            block = values[parents[:, None], children[None, :]]
            dist = torch.cdist(physical[parents].float(), physical[children].float())
            block = block.masked_fill(dist > cutoff_um, -torch.inf)
            null = torch.zeros((1, len(children)), device=logits.device)
            denom = torch.logsumexp(torch.cat((block, null), dim=0), dim=0)
            log_block = block - denom
            log_probs[parents[:, None], children[None, :]] = log_block
            if log_complements is not None:
                log_other = torch.empty_like(log_block)
                low = log_block < -math.log(2)
                middle = (~low) & (log_block < 0)
                log_other[low] = torch.log1p(-torch.exp(log_block[low]))
                log_other[middle] = torch.log(-torch.expm1(log_block[middle]))
                saturated = (~low) & (~middle)
                if saturated.any():
                    parent_rows, child_columns = torch.where(saturated)
                    other_logits = block[:, child_columns].clone()
                    other_logits[
                        parent_rows, torch.arange(len(parent_rows), device=logits.device)
                    ] = -torch.inf
                    other_denominator = torch.logsumexp(
                        torch.cat((other_logits, null[:, child_columns]), dim=0), dim=0
                    )
                    log_other[parent_rows, child_columns] = other_denominator - denom[child_columns]
                log_complements[parents[:, None], children[None, :]] = log_other
    return (log_probs, log_complements) if include_complement else log_probs


def masked_association_loss(
    logits: torch.Tensor,
    log_probabilities: torch.Tensor,
    log_complements: torch.Tensor,
    teacher: dict[str, Any],
    *,
    auxiliary_weight: float = 0.01,
) -> torch.Tensor | None:
    mask = torch.as_tensor(teacher["mask"], dtype=torch.bool, device=logits.device)
    if not mask.any():
        return None
    target = torch.as_tensor(teacher["target"], dtype=torch.float32, device=logits.device)
    weight = torch.as_tensor(teacher["weight"], dtype=torch.float32, device=logits.device)
    normalized = torch.where(target[mask] > 0.5, -log_probabilities[mask], -log_complements[mask])
    auxiliary = F.binary_cross_entropy_with_logits(
        logits.float()[mask], target[mask], reduction="none"
    )
    return torch.sum(weight[mask] * (normalized + auxiliary_weight * auxiliary)) / mask.sum()


def score_context_window(
    model: TrackingTransformer, window: dict[str, Any], device: torch.device, cutoff_um: float
) -> np.ndarray:
    coords = torch.as_tensor(window["coords"], device=device).unsqueeze(0)
    features = torch.as_tensor(window["features"], device=device).unsqueeze(0)
    if coords.shape[1] == 0:
        return np.empty((0, 0), dtype=np.float32)
    with torch.no_grad():
        logits = model(coords, features)[0]
        probs = parental_softmax(
            logits,
            torch.as_tensor(window["times"], device=device),
            torch.as_tensor(window["physical"], device=device),
            cutoff_um=cutoff_um,
        )
    if not torch.isfinite(probs).all():
        raise FloatingPointError("non-finite association probabilities")
    return probs.detach().cpu().numpy()


def accumulate_adjacent_scores(
    sums: dict[tuple[int, int, int], tuple[float, int]],
    window: dict[str, Any],
    probabilities: np.ndarray,
) -> None:
    times = window["times"]
    ids = window["candidate_ids"]
    for frame in window["frames"][:-1]:
        parents = np.flatnonzero(times == frame)
        children = np.flatnonzero(times == frame + 1)
        for i in parents:
            for j in children:
                key = (int(frame), int(ids[i]), int(ids[j]))
                previous_sum, count = sums.get(key, (0.0, 0))
                sums[key] = (previous_sum + float(probabilities[i, j]), count + 1)


def mean_adjacent_scores(
    sums: dict[tuple[int, int, int], tuple[float, int]],
) -> dict[tuple[int, int, int], float]:
    return {key: total / count for key, (total, count) in sums.items() if count > 0}


def select_internal_threshold(negative_scores: np.ndarray, baseline_count: int) -> float:
    """Smallest strict-greater threshold with no more false alarms than control."""
    scores = np.asarray(negative_scores, dtype=np.float64)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("internal teacher-negative scores are missing or non-finite")
    if baseline_count < 0:
        raise ValueError("baseline negative count must be nonnegative")
    candidates = np.unique(np.concatenate(([0.0], scores, [1.0])))
    return float(next(t for t in candidates if np.count_nonzero(scores > t) <= baseline_count))


def probability_candidate_edges(
    scores: dict[tuple[int, int, int], float], video: dict[str, Any], threshold: float
) -> list[tuple[int, int, float, float]]:
    """Return the existing graph input tuple without a second softmax."""
    result = []
    for (frame, source, target), probability in scores.items():
        if not np.isfinite(probability):
            raise FloatingPointError("non-finite edge probability")
        if probability <= threshold:
            continue
        parents = video["registry"][frame]
        children = video["registry"][frame + 1]
        src_index = np.flatnonzero(parents["ids"] == source)
        dst_index = np.flatnonzero(children["ids"] == target)
        if len(src_index) != 1 or len(dst_index) != 1:
            raise ValueError("aggregated edge endpoints do not match fixed candidates")
        distance = float(
            np.linalg.norm(parents["grid"][src_index[0]] - children["grid"][dst_index[0]])
        )
        result.append((source, target, float(probability), distance))
    return sorted(result, key=lambda edge: (-edge[2], edge[0], edge[1]))


def new_trackastra_model(params: dict[str, Any]) -> TrackingTransformer:
    return TrackingTransformer(
        coord_dim=int(params["coord_dim"]),
        feat_dim=int(params["feat_dim"]),
        d_model=int(params["d_model"]),
        nhead=int(params["nhead"]),
        num_encoder_layers=int(params["num_encoder_layers"]),
        num_decoder_layers=int(params["num_decoder_layers"]),
        dropout=float(params["dropout"]),
        pos_embed_per_dim=int(params["pos_embed_per_dim"]),
        feat_embed_per_dim=int(params["feat_embed_per_dim"]),
        window=int(params["window"]),
        spatial_pos_cutoff=float(params["spatial_pos_cutoff_um"]),
        attn_positional_bias=str(params["attn_positional_bias"]),
        attn_positional_bias_n_spatial=int(params["attn_positional_bias_n_spatial"]),
        causal_norm=str(params["causal_norm"]),
        attn_dist_mode=str(params["attn_dist_mode"]),
    )


def window_loss(
    model: TrackingTransformer,
    window: dict[str, Any],
    device: torch.device,
    *,
    cutoff_um: float,
    auxiliary_weight: float,
) -> torch.Tensor | None:
    teacher = window.get("teacher")
    if teacher is None or not np.any(teacher["mask"]):
        return None
    coords = torch.as_tensor(window["coords"], device=device).unsqueeze(0)
    features = torch.as_tensor(window["features"], device=device).unsqueeze(0)
    logits = model(coords, features)[0]
    log_probabilities, log_complements = parental_log_probabilities(
        logits,
        torch.as_tensor(window["times"], device=device),
        torch.as_tensor(window["physical"], device=device),
        cutoff_um=cutoff_um,
        include_complement=True,
    )
    loss = masked_association_loss(
        logits,
        log_probabilities,
        log_complements,
        teacher,
        auxiliary_weight=auxiliary_weight,
    )
    if loss is None or not torch.isfinite(loss):
        raise FloatingPointError("non-finite sparse association loss")
    return loss


def predict_video_pair_matrices(
    model: TrackingTransformer,
    video: dict[str, Any],
    device: torch.device,
    *,
    window_size: int,
    cutoff_um: float,
) -> dict[int, np.ndarray]:
    matrices = {}
    counts = {}
    for pair in video["pairs"]:
        frame = pair["frames"][0]
        n_src = len(pair["arrays"]["candidate_ids_src"])
        n_tgt = len(pair["arrays"]["candidate_ids_tgt"])
        matrices[frame] = np.zeros((n_src, n_tgt), dtype=np.float64)
        counts[frame] = 0
    model.eval()
    for start in context_window_starts(len(video["frames"]), window_size):
        window = build_context_window(video, start, window_size=window_size)
        probs = score_context_window(model, window, device, cutoff_um)
        offsets = np.cumsum([0, *window["frame_counts"]])
        for index, frame in enumerate(window["frames"][:-1]):
            block = probs[
                offsets[index] : offsets[index + 1], offsets[index + 1] : offsets[index + 2]
            ]
            matrices[frame] += block
            counts[frame] += 1
    if any(count == 0 for count in counts.values()):
        raise ValueError("some adjacent pairs were never scored")
    return {
        frame: (matrix / counts[frame]).astype(np.float32) for frame, matrix in matrices.items()
    }


def predict_control_pair_matrices(
    tracker: Any,
    video: dict[str, Any],
    device: torch.device,
    *,
    downsample_zyx: tuple[float, float, float] = (1.0, 4.0, 4.0),
) -> dict[int, np.ndarray]:
    scale = torch.as_tensor(downsample_zyx, device=device, dtype=torch.float32)
    matrices = {}
    tracker.eval()
    for pair in video["pairs"]:
        arrays = pair["arrays"]

        def tensor(
            name: str, dtype: torch.dtype = torch.float32, *, cache_arrays: dict[str, Any] = arrays
        ) -> torch.Tensor:
            return torch.as_tensor(cache_arrays[name], device=device, dtype=dtype).unsqueeze(0)

        src = torch.cat((tensor("primary_features_src"), tensor("position_features_src")), -1)
        tgt = torch.cat((tensor("primary_features_tgt"), tensor("position_features_tgt")), -1)
        with torch.no_grad():
            logits = tracker(
                src,
                tgt,
                tensor("coords_src_grid") * scale,
                tensor("coords_tgt_grid") * scale,
                tensor("candidate_mask_src", torch.bool),
                tensor("candidate_mask_tgt", torch.bool),
            )[0]
            if logits.shape[0] and logits.shape[1]:
                probs = torch.softmax(logits.float(), dim=0)
            else:
                probs = logits.float().new_zeros(logits.shape)
        matrices[pair["frames"][0]] = probs.cpu().numpy()
    return matrices


def eligible_pair_frames(video: dict[str, Any], annotation: AnnotationGraph) -> set[int]:
    """Use the same adjacent-pair GT availability rule as the saved exp016 control."""
    return {
        int(pair["frames"][0])
        for pair in video["pairs"]
        if all(frame in annotation.frames for frame in pair["frames"])
    }


def pair_label_matrices(
    video: dict[str, Any],
    annotation: AnnotationGraph,
    matches: dict[int, np.ndarray],
    *,
    cutoff_um: float,
) -> dict[int, dict[str, np.ndarray]]:
    labels = {}
    for pair in video["pairs"]:
        frame, next_frame = pair["frames"]
        target = build_legacy_edge_target(
            matches[frame],
            matches[next_frame],
            annotation.edges,
            outgoing_edges=annotation.outgoing_edges,
        )
        active = legacy_active_pair_mask(target)
        left = video["registry"][frame]
        right = video["registry"][next_frame]
        times = np.concatenate(
            (np.full(len(left["ids"]), frame), np.full(len(right["ids"]), next_frame))
        )
        physical = np.concatenate((left["physical"], right["physical"]))
        ids = np.concatenate((matches[frame], matches[next_frame]))
        safe = build_sparse_teacher(times, physical, ids, annotation, cutoff_um=cutoff_um)
        n_left = len(left["ids"])
        labels[frame] = {
            "target": target,
            "legacy_negative": active & (target == 0),
            "known_negative": safe["mask"][:n_left, n_left:] & (target == 0),
            "source_matches": matches[frame],
            "target_matches": matches[next_frame],
        }
    return labels


def summarize_pair_predictions(
    scores: dict[int, np.ndarray],
    labels: dict[int, dict[str, np.ndarray]],
    annotation: AnnotationGraph,
    threshold: float,
) -> dict[str, int | float]:
    counts = {
        "known_edges": 0,
        "recovered_edges": 0,
        "known_parent_children": 0,
        "correct_parent_top1": 0,
        "legacy_negative_denominator": 0,
        "legacy_negative_predictions": 0,
        "known_negative_denominator": 0,
        "known_negative_predictions": 0,
        "division_parents": 0,
        "recovered_division_parents": 0,
    }
    for frame, score in scores.items():
        label = labels[frame]
        positive = label["target"] > 0
        if score.shape != positive.shape:
            raise ValueError("pair score and label shapes disagree")
        if not np.isfinite(score).all():
            raise FloatingPointError("non-finite pair score")
        selected = score > threshold
        counts["known_edges"] += int(positive.sum())
        counts["recovered_edges"] += int((positive & selected).sum())
        for column in np.flatnonzero(positive.any(axis=0)):
            counts["known_parent_children"] += 1
            if score.shape[0] and positive[int(np.argmax(score[:, column])), column]:
                counts["correct_parent_top1"] += 1
        for kind in ("legacy_negative", "known_negative"):
            mask = label[kind]
            counts[kind + "_denominator"] += int(mask.sum())
            counts[kind + "_predictions"] += int((mask & selected).sum())
        source = label["source_matches"]
        target = label["target_matches"]
        for parent in set(int(x) for x in source if x >= 0):
            daughters = (
                annotation.outgoing_edges.get(parent, ()) if annotation.outgoing_edges else ()
            )
            if len(daughters) != 2:
                continue
            row_indices = np.flatnonzero(source == parent)
            daughter_cols = [np.flatnonzero(target == daughter) for daughter in daughters]
            if len(row_indices) != 1 or any(len(cols) != 1 for cols in daughter_cols):
                continue
            counts["division_parents"] += 1
            row = int(row_indices[0])
            counts["recovered_division_parents"] += int(
                all(selected[row, int(cols[0])] for cols in daughter_cols)
            )
    counts["edge_recall"] = (
        counts["recovered_edges"] / counts["known_edges"] if counts["known_edges"] else 0.0
    )
    counts["parent_top1_rate"] = (
        counts["correct_parent_top1"] / counts["known_parent_children"]
        if counts["known_parent_children"]
        else 0.0
    )
    return counts


def internal_negative_scores(
    scores: dict[int, np.ndarray], labels: dict[int, dict[str, np.ndarray]]
) -> np.ndarray:
    if not scores:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(
        [scores[frame][labels[frame]["legacy_negative"]] for frame in sorted(scores)]
    )


def pair_gate(control: dict[str, float], candidate: dict[str, float]) -> dict[str, Any]:
    required = (
        "known_edges",
        "division_parents",
        "legacy_negative_denominator",
        "known_negative_denominator",
    )
    if any(control[key] == 0 or candidate[key] == 0 for key in required):
        return {"passed": False, "reason": "missing effective denominator"}
    checks = {
        "known_edge_recall_strict": candidate["edge_recall"] > control["edge_recall"],
        "parent_top1_nondecrease": candidate["parent_top1_rate"] >= control["parent_top1_rate"],
        "division_nondecrease": candidate["recovered_division_parents"]
        >= control["recovered_division_parents"],
        "legacy_negative_5_percent": candidate["legacy_negative_predictions"]
        <= max(0, 1.05 * control["legacy_negative_predictions"]),
        "known_negative_5_percent": candidate["known_negative_predictions"]
        <= max(0, 1.05 * control["known_negative_predictions"]),
    }
    return {"passed": all(checks.values()), "checks": checks}


def training_window_schedule(
    manifest: list[dict[str, int | str]],
    *,
    windows_per_epoch: int,
    epochs: int,
    seed: int,
) -> dict[str, Any]:
    if epochs < 1 or windows_per_epoch < 1:
        raise ValueError("epochs and windows_per_epoch must be positive")
    active = [
        row
        for row in manifest
        if int(row["positive_dt1"]) + int(row["positive_dt2"]) + int(row["known_negative"]) > 0
    ]
    keys = [(str(row["sample"]), int(row["start"])) for row in active]
    if not active or len(keys) != len(set(keys)):
        raise ValueError("effective window manifest is empty or contains duplicate windows")
    if windows_per_epoch > len(active):
        raise ValueError("windows_per_epoch exceeds the effective training population")
    cuts = np.quantile(
        np.asarray([int(row["tokens"]) for row in active]),
        [0.25, 0.5, 0.75],
    )
    groups: dict[tuple[str, int], list[dict[str, int | str]]] = {}
    for row in active:
        if int(row["division_positive"]) > 0:
            teacher_type = "division"
        elif int(row["positive_dt2"]) > 0:
            teacher_type = "two_step"
        elif int(row["positive_dt1"]) > 0:
            teacher_type = "adjacent"
        else:
            teacher_type = "known_negative_only"
        band = int(np.searchsorted(cuts, int(row["tokens"]), side="right"))
        groups.setdefault((teacher_type, band), []).append(row)
    ordered_groups = sorted(groups)
    allocation = {}
    remainders = []
    for key in ordered_groups:
        exact = windows_per_epoch * len(groups[key]) / len(active)
        allocation[key] = int(math.floor(exact))
        remainders.append((exact - allocation[key], key))
    remaining = windows_per_epoch - sum(allocation.values())
    for _, key in sorted(remainders, key=lambda item: (-item[0], item[1]))[:remaining]:
        allocation[key] += 1
    teacher_types = sorted({key[0] for key in ordered_groups})
    if windows_per_epoch < len(teacher_types):
        raise ValueError("sampling capacity is smaller than the number of supervision types")
    for teacher_type in teacher_types:
        if sum(value for key, value in allocation.items() if key[0] == teacher_type):
            continue
        receiver = min(
            (key for key in ordered_groups if key[0] == teacher_type),
            key=lambda key: (-len(groups[key]), key),
        )
        donors = [
            key
            for key in ordered_groups
            if allocation[key] > 0
            and sum(value for other, value in allocation.items() if other[0] == key[0]) > 1
        ]
        if not donors:
            raise ValueError("no window can be reassigned to a missing supervision type")
        donor = min(donors, key=lambda key: (-allocation[key], key))
        allocation[donor] -= 1
        allocation[receiver] += 1
    orders = {
        key: sorted(
            groups[key],
            key=lambda row: hashlib.sha256(
                f"{seed}:{key[0]}:{key[1]}:{row['sample']}:{row['start']}".encode()
            ).hexdigest(),
        )
        for key in ordered_groups
    }
    positions = {key: 0 for key in ordered_groups}
    schedule = []
    for epoch in range(epochs):
        selected = []
        for key in ordered_groups:
            rows = orders[key]
            quota = allocation[key]
            for offset in range(quota):
                row = rows[(positions[key] + offset) % len(rows)]
                selected.append(
                    {
                        "sample": str(row["sample"]),
                        "start": int(row["start"]),
                        "tokens": int(row["tokens"]),
                        "teacher_type": key[0],
                        "density_band": key[1],
                    }
                )
            positions[key] = (positions[key] + quota) % len(rows)
        if len(selected) != windows_per_epoch:
            raise AssertionError("sampling allocation did not fill the epoch")
        if len({(row["sample"], row["start"]) for row in selected}) != len(selected):
            raise AssertionError("a training window repeated within an epoch")
        schedule.append(selected)
    return {
        "seed": seed,
        "epochs": schedule,
        "population_count": len(active),
        "token_quartile_boundaries": [float(value) for value in cuts],
        "stratum_counts": {f"{key[0]}:{key[1]}": len(groups[key]) for key in ordered_groups},
        "stratum_windows_per_epoch": {
            f"{key[0]}:{key[1]}": allocation[key] for key in ordered_groups
        },
    }
