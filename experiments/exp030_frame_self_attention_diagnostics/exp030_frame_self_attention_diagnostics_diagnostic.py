# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # exp030: within-frame Self-Attention diagnostics
#
# This notebook checks the public tracker checkpoint against the unchanged
# Cross-Attention path and measures forward and backward cost for the current,
# Model A and Model B architectures on fixed exp015 cache windows. Its gradient
# benchmark uses a synthetic scalar; it neither trains a tracker nor estimates
# an edge or official graph score.

# %% [markdown]
# ## Contents
# 1. Model implementation and public initialization contract
# 2. Configuration and fixed inputs
# 3. Cache identity and window selection
# 4. Legacy output and checkpoint checks
# 5. GPU time and peak memory
# 6. Diagnostic record

# %% [markdown]
# ## 1. Model implementation and public initialization contract
#
# The following source is identical to frame_attention_model.py in this
# experiment at packaging time. It is embedded so the Kaggle notebook does
# not import an experiment-local helper module.

# %%
# ruff: noqa: E402
# BEGIN EMBEDDED MODEL
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


# END EMBEDDED MODEL

# %% [markdown]
# ## 2. Configuration and fixed inputs

# %%
import hashlib
import io
import json
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml

WORKING_ROOT = Path.cwd()
if not Path("/kaggle/input").is_dir() or not Path("/kaggle/working").is_dir():
    raise RuntimeError("Run the authoritative diagnostic in a Kaggle Notebook")
if not torch.cuda.is_available():
    raise RuntimeError("A Kaggle GPU is required for the resource diagnostic")
config = yaml.safe_load((WORKING_ROOT / "config.yaml").read_text(encoding="utf-8"))
cache_cfg = config["data"]
model_cfg = config["model"]
params = model_cfg["params"]
variants = model_cfg["variants"]
if list(variants) != ["current", "model_a", "model_b"]:
    raise RuntimeError("Unexpected architecture comparison")
if model_cfg["epochs"] != 0:
    raise RuntimeError("This notebook must not train checkpoints")
assert config["lineage"]["parent"] == "exp016_frozen_image_encoder"
STARTED = time.perf_counter()
DEVICE = torch.device("cuda")
torch.manual_seed(int(config["reproducibility"]["seed"]))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def one_path(paths: list[Path], label: str) -> Path:
    selected = sorted(set(path.resolve() for path in paths))
    if len(selected) != 1:
        raise RuntimeError(f"Expected one {label}; found {selected}")
    return selected[0]


input_root = Path("/kaggle/input")
cache_slug = str(cache_cfg["cache_kernel_source"]).split("/", 1)[1]
cache_summary_path = one_path(
    [
        path
        for path in input_root.rglob("window_cache_summary.json")
        if cache_slug in path.as_posix()
    ],
    "fixed exp015 cache summary",
)
summary = json.loads(cache_summary_path.read_text(encoding="utf-8"))
unsigned_summary = dict(summary)
recorded_summary_sha = unsigned_summary.pop("summary_sha256", None)
if (
    recorded_summary_sha != cache_cfg["cache_summary_sha256"]
    or json_sha256(unsigned_summary) != recorded_summary_sha
):
    raise RuntimeError("Fixed cache summary checksum mismatch")
if (
    summary.get("cache_identity_sha256") != cache_cfg["cache_identity_sha256"]
    or summary.get("window_count") != cache_cfg["expected_window_count"]
):
    raise RuntimeError("Fixed cache summary coverage mismatch")
cache_root = cache_summary_path.parent / cache_cfg["cache_directory"]
paths = sorted(cache_root.glob("*/*.npz"))
if len(paths) != cache_cfg["expected_window_count"]:
    raise RuntimeError("Fixed cache window count mismatch")

public_source_path = one_path(
    list(input_root.rglob(str(cache_cfg["public_model_source"]))),
    "public model source",
)
public_root = public_source_path.parents[4]
public_checkpoint_path = public_root / str(cache_cfg["public_checkpoint"])
if sha256_file(public_source_path) != cache_cfg["public_model_source_sha256"]:
    raise RuntimeError("Public model source SHA mismatch")
if sha256_file(public_checkpoint_path) != cache_cfg["public_checkpoint_sha256"]:
    raise RuntimeError("Public tracker checkpoint SHA mismatch")

sys.path.insert(0, str(public_root / "repo" / "src"))
from biohub_tracking.models.simple_node_transformer import (
    SimpleNodeTransformer as PublicSimpleNodeTransformer,
)

full_state = torch.load(public_checkpoint_path, map_location="cpu", weights_only=True)
public_state = {}
for prefix in ("transformer.", "module.transformer."):
    public_state = {
        key[len(prefix) :]: value for key, value in full_state.items() if key.startswith(prefix)
    }
    if public_state:
        break
if not public_state:
    raise RuntimeError("Public checkpoint has no primary tracker parameters")

# %% [markdown]
# ## 3. Cache identity and window selection
#
# Scan metadata only. Select the median and largest cell-pair window by
# N_t * N_t1, with a stable file-path tie break. Do not truncate cells.

# %%
scan_started = time.perf_counter()
identities = []
window_sizes = []
for path in paths:
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(saved["__metadata_json__"].tobytes().decode("utf-8"))
    schema = metadata["array_schema"]
    by_name = {str(item["name"]): item for item in schema}
    n_src = int(by_name["candidate_mask_src"]["shape"][0])
    n_tgt = int(by_name["candidate_mask_tgt"]["shape"][0])
    feature_names = (
        "primary_features_src",
        "primary_features_tgt",
        "secondary_features_src",
        "secondary_features_tgt",
    )
    identities.append(
        {
            "dataset": path.parent.name,
            "window_frames": [int(value) for value in metadata["window_frames"]],
            "candidate_count_src": n_src,
            "candidate_count_tgt": n_tgt,
            "feature_values": int(
                sum(np.prod(by_name[name]["shape"], dtype=np.int64) for name in feature_names)
            ),
            "cache_bytes": int(path.stat().st_size),
            "cache_schema_sha256": json_sha256(schema),
            "cache_content_sha256": str(metadata.get("array_content_sha256")),
        }
    )
    if n_src > 0 and n_tgt > 0:
        window_sizes.append((n_src * n_tgt, path.as_posix(), n_src, n_tgt))
if json_sha256(identities) != cache_cfg["cache_identity_sha256"]:
    raise RuntimeError("Fixed cache identity differs from exp015")
if not window_sizes:
    raise RuntimeError("Cache has no nonempty windows")
window_sizes.sort()
selected = {
    "representative": window_sizes[len(window_sizes) // 2],
    "largest": window_sizes[-1],
}
scan_seconds = time.perf_counter() - scan_started


def window_input(record: tuple[int, str, int, int]) -> tuple[torch.Tensor, ...]:
    _, path_string, _, _ = record
    with np.load(path_string, allow_pickle=False) as saved:
        arrays = {
            key: saved[key]
            for key in (
                "primary_features_src",
                "primary_features_tgt",
                "position_features_src",
                "position_features_tgt",
                "coords_src_grid",
                "coords_tgt_grid",
                "candidate_mask_src",
                "candidate_mask_tgt",
            )
        }
    scale = np.asarray(config["runtime"]["downsample_zyx"], dtype=np.float32)
    src = np.concatenate(
        [arrays["primary_features_src"], arrays["position_features_src"]], axis=1
    ).astype(np.float32)
    tgt = np.concatenate(
        [arrays["primary_features_tgt"], arrays["position_features_tgt"]], axis=1
    ).astype(np.float32)
    return (
        torch.from_numpy(src).unsqueeze(0).to(DEVICE),
        torch.from_numpy(tgt).unsqueeze(0).to(DEVICE),
        torch.from_numpy(arrays["coords_src_grid"].astype(np.float32) * scale)
        .unsqueeze(0)
        .to(DEVICE),
        torch.from_numpy(arrays["coords_tgt_grid"].astype(np.float32) * scale)
        .unsqueeze(0)
        .to(DEVICE),
        torch.from_numpy(arrays["candidate_mask_src"].astype(np.bool_)).unsqueeze(0).to(DEVICE),
        torch.from_numpy(arrays["candidate_mask_tgt"].astype(np.bool_)).unsqueeze(0).to(DEVICE),
    )


# %% [markdown]
# ## 4. Legacy output and checkpoint checks

# %%
legacy_kwargs = {
    key: params[key]
    for key in (
        "feat_dim",
        "hidden_dim",
        "n_heads",
        "n_blocks",
        "mlp_ratio",
        "dropout",
        "pair_chunk_size",
    )
}
legacy = PublicSimpleNodeTransformer(**legacy_kwargs).to(DEVICE)
legacy.load_state_dict(public_state, strict=True)
legacy.eval()
new_models = {}
encoder_states = {}
for name, flags in variants.items():
    torch.manual_seed(int(config["reproducibility"]["seed"]))
    model = SimpleNodeTransformer(
        **legacy_kwargs, n_self_blocks=params["n_self_blocks"], **flags
    ).to(DEVICE)
    load_public_initialization(model, public_state)
    model.eval()
    if flags["use_temporal_self_attention"]:
        encoder_states[name] = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
            if key.startswith("self_encoder.")
        }
    new_models[name] = model
if encoder_states["model_a"].keys() != encoder_states["model_b"].keys():
    raise RuntimeError("Model A/B encoder keys differ")
if any(
    not torch.equal(value, encoder_states["model_b"][key])
    for key, value in encoder_states["model_a"].items()
):
    raise RuntimeError("Model A/B encoder initialization differs")

representative_input = window_input(selected["representative"])
with torch.no_grad():
    expected = legacy(*representative_input)
    observed = new_models["current"](*representative_input)
    legacy_max_abs_diff = float((expected - observed).abs().max().item())
if not torch.equal(expected, observed):
    raise RuntimeError(f"Legacy checkpoint logits changed: {legacy_max_abs_diff}")
for name in ("model_a", "model_b"):
    model = new_models[name]
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = SimpleNodeTransformer(
        **legacy_kwargs, n_self_blocks=params["n_self_blocks"], **variants[name]
    ).to(DEVICE)
    restored_state = torch.load(buffer, map_location=DEVICE, weights_only=True)
    restored.load_state_dict(restored_state, strict=True)
    restored.eval()
    with torch.no_grad():
        before = model(*representative_input)
        after = restored(*representative_input)
    if not torch.equal(before, after):
        raise RuntimeError(f"{name} checkpoint round-trip changed logits")
    del restored

# %% [markdown]
# ## 5. Mask, empty-set, order, and gradient contracts
#
# These checks run against the same public-initialized Model A/B modules that
# enter the resource benchmark. Padding and cell permutation must preserve
# the valid pair scores.

# %%
with torch.no_grad():
    small = (
        torch.randn((1, 4, int(params["feat_dim"])), device=DEVICE),
        torch.randn((1, 4, int(params["feat_dim"])), device=DEVICE),
        torch.randn((1, 4, 3), device=DEVICE),
        torch.randn((1, 4, 3), device=DEVICE),
        torch.ones((1, 4), dtype=torch.bool, device=DEVICE),
        torch.ones((1, 4), dtype=torch.bool, device=DEVICE),
    )
    src_order = torch.tensor([2, 0, 3, 1], device=DEVICE)
    tgt_order = torch.tensor([3, 1, 0, 2], device=DEVICE)
    for name in ("model_a", "model_b"):
        model = new_models[name]
        base = model(*small)
        padded = (
            torch.cat([small[0], torch.full((1, 1, 64), 999.0, device=DEVICE)], dim=1),
            torch.cat([small[1], torch.full((1, 1, 64), -999.0, device=DEVICE)], dim=1),
            torch.cat([small[2], torch.full((1, 1, 3), 999.0, device=DEVICE)], dim=1),
            torch.cat([small[3], torch.full((1, 1, 3), -999.0, device=DEVICE)], dim=1),
            torch.cat([small[4], torch.zeros((1, 1), dtype=torch.bool, device=DEVICE)], dim=1),
            torch.cat([small[5], torch.zeros((1, 1), dtype=torch.bool, device=DEVICE)], dim=1),
        )
        torch.testing.assert_close(model(*padded)[:, :4, :4], base, atol=1e-5, rtol=1e-5)
        permuted = (
            small[0][:, src_order],
            small[1][:, tgt_order],
            small[2][:, src_order],
            small[3][:, tgt_order],
            small[4][:, src_order],
            small[5][:, tgt_order],
        )
        torch.testing.assert_close(
            model(*permuted), base[:, src_order][:, :, tgt_order], atol=1e-5, rtol=1e-5
        )
        all_padding = (
            small[0],
            small[1],
            small[2],
            small[3],
            torch.zeros_like(small[4]),
            torch.zeros_like(small[5]),
        )
        if not torch.isfinite(model(*all_padding)).all():
            raise RuntimeError(f"{name} produced nonfinite all-padding logits")
        unbatched = tuple(value.squeeze(0) for value in small)
        if model(*unbatched).shape != (4, 4):
            raise RuntimeError(f"{name} unbatched shape changed")
        if model(*unbatched[:4], unbatched[4], unbatched[5]).shape != (4, 4):
            raise RuntimeError(f"{name} one-dimensional masks failed")
    empty = (
        small[0][:, :0],
        small[1],
        small[2][:, :0],
        small[3],
        small[4][:, :0],
        small[5],
    )
empty_logits = new_models["model_b"](*empty)
if empty_logits.shape != (1, 0, 4):
    raise RuntimeError("Empty cell-set shape changed")
empty_logits.sum().backward()
new_models["model_b"].zero_grad(set_to_none=True)
print("Mask, empty-set, order and checkpoint contracts passed", flush=True)

# %% [markdown]
# ## 6. GPU time and peak memory
#
# The backward pass differentiates mean squared logits to measure resource
# use. No optimizer step, label construction, or checkpoint selection occurs.


# %%
def benchmark(model: SimpleNodeTransformer, sample: tuple[torch.Tensor, ...]) -> dict[str, object]:
    result = {}
    for mode in ("forward", "forward_backward"):
        model.train(mode == "forward_backward")
        times = []
        peaks = []
        peak_increments = []
        for iteration in range(
            config["runtime"]["benchmark_warmup"] + config["runtime"]["benchmark_repeats"]
        ):
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            baseline_allocated = torch.cuda.memory_allocated()
            torch.cuda.synchronize()
            started = time.perf_counter()
            if mode == "forward":
                with torch.no_grad():
                    logits = model(*sample)
            else:
                logits = model(*sample)
                logits.square().mean().backward()
                if not all(
                    torch.isfinite(param.grad).all()
                    for param in model.parameters()
                    if param.grad is not None
                ):
                    raise RuntimeError("Nonfinite model gradient")
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            peak = torch.cuda.max_memory_allocated()
            if not torch.isfinite(logits).all():
                raise RuntimeError("Nonfinite model logits")
            if iteration >= config["runtime"]["benchmark_warmup"]:
                times.append(seconds)
                peaks.append(peak)
                peak_increments.append(max(0, peak - baseline_allocated))
        result[mode] = {
            "median_seconds": statistics.median(times),
            "max_peak_allocated_bytes": max(peaks),
            "max_peak_increment_bytes": max(peak_increments),
            "repeats": len(times),
        }
    model.eval()
    return result


benchmarks = {}
for window_name, window_record in selected.items():
    sample = window_input(window_record)
    benchmarks[window_name] = {}
    for variant_name, model in new_models.items():
        try:
            benchmarks[window_name][variant_name] = benchmark(model, sample)
        except torch.cuda.OutOfMemoryError as exc:
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            benchmarks[window_name][variant_name] = {
                "error": type(exc).__name__,
                "message": str(exc)[:500],
            }
        print(window_name, variant_name, benchmarks[window_name][variant_name], flush=True)
    del sample

# %% [markdown]
# ## 7. Diagnostic record

# %%
report = {
    "experiment": config["experiment"]["name"],
    "created_at": datetime.now(UTC).isoformat(),
    "cache_summary_sha256": recorded_summary_sha,
    "cache_identity_sha256": cache_cfg["cache_identity_sha256"],
    "public_model_source_sha256": sha256_file(public_source_path),
    "public_checkpoint_sha256": sha256_file(public_checkpoint_path),
    "embedded_model_source_sha256": sha256_file(WORKING_ROOT / "frame_attention_model.py"),
    "device": torch.cuda.get_device_name(DEVICE),
    "window_scan_seconds": scan_seconds,
    "windows": {
        name: {"path": record[1], "n_src": record[2], "n_tgt": record[3]}
        for name, record in selected.items()
    },
    "legacy_max_abs_logit_diff": legacy_max_abs_diff,
    "legacy_exact_match": True,
    "model_a_b_same_encoder_initialization": True,
    "variant_flags": variants,
    "parameters": {
        name: {
            "total": sum(p.numel() for p in model.parameters()),
            "forward_used": sum(
                p.numel()
                for key, p in model.named_parameters()
                if not (name == "model_a" and key.startswith("blocks."))
            ),
        }
        for name, model in new_models.items()
    },
    "benchmarks": benchmarks,
    "notebook_elapsed_seconds": time.perf_counter() - STARTED,
    "epochs_trained": 0,
    "official_graph_score_computed": False,
    "submission_created": False,
}
output = WORKING_ROOT / "frame_self_attention_diagnostic.json"
output.write_text(
    json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
)
print("Diagnostic record:", output, sha256_file(output), flush=True)
