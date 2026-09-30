"""Trainable primary tracker with x138's fixed logit fusion and K=8 history."""

from __future__ import annotations

import torch
from torch import nn

from past_candidate_attention import PastCandidateAttentionTracker


def x138_fuse_logits(
    forward: torch.Tensor,
    reverse_native: torch.Tensor,
    secondary: torch.Tensor,
    *,
    bidirectional_weight: float = 0.15,
    secondary_edge_weight: float = 0.15,
    secondary_low_margin_max: float = 0.35,
    secondary_mix_temperature: float = 1.0,
) -> torch.Tensor:
    """The exp043 harmonic/low-margin-consensus calculation, before source softmax."""
    if forward.ndim != 3 or reverse_native.shape != forward.transpose(1, 2).shape:
        raise ValueError("forward/reverse logit shapes differ")
    if secondary.shape != forward.shape:
        raise ValueError("secondary logit shape differs")
    if not 0.0 <= bidirectional_weight <= 1.0:
        raise ValueError("invalid bidirectional weight")
    logits = forward
    if bidirectional_weight:
        reverse = reverse_native.transpose(1, 2)
        forward_center = logits.mean(dim=1, keepdim=True)
        forward_scale = logits.float().std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)
        reverse_center = reverse.mean(dim=1, keepdim=True)
        reverse_scale = reverse.float().std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)
        ratio = (forward_scale / reverse_scale).clamp(0.5, 2.0).to(reverse.dtype)
        reverse_aligned = (reverse - reverse_center) * ratio + forward_center
        forward_prob = torch.softmax(logits.float(), dim=1).clamp_min(1e-8)
        reverse_prob = torch.softmax(reverse_aligned.float(), dim=1).clamp_min(1e-8)
        harmonic_prob = 1.0 / (
            (1.0 - bidirectional_weight) / forward_prob + bidirectional_weight / reverse_prob
        )
        harmonic_prob = harmonic_prob / harmonic_prob.sum(dim=1, keepdim=True).clamp_min(1e-8)
        harmonic_logits = torch.log(harmonic_prob.clamp_min(1e-8))
        center = harmonic_logits.mean(dim=1, keepdim=True)
        scale = harmonic_logits.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)
        ratio = (forward_scale / scale).clamp(0.5, 2.0)
        logits = ((harmonic_logits - center) * ratio + forward_center).to(reverse.dtype)

    primary_center = logits.mean(dim=1, keepdim=True)
    primary_scale = logits.float().std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)
    secondary_center = secondary.mean(dim=1, keepdim=True)
    secondary_scale = secondary.float().std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)
    ratio = (primary_scale / secondary_scale).clamp(0.5, 2.0)
    secondary_aligned = (secondary - secondary_center) * ratio + primary_center
    if logits.shape[1] >= 2:
        primary_probs = torch.softmax(logits[0], dim=0)
        secondary_probs = torch.softmax(secondary_aligned[0], dim=0)
        primary_top2 = torch.topk(primary_probs, k=2, dim=0)
        secondary_top2 = torch.topk(secondary_probs, k=2, dim=0)
        primary_margin = primary_top2.values[0] - primary_top2.values[1]
        same_parent = primary_top2.indices[0].eq(secondary_top2.indices[0])
        uncertainty = (
            (secondary_low_margin_max - primary_margin) / secondary_low_margin_max
        ).clamp(0, 1)
        local_weight = secondary_edge_weight * uncertainty
        local_weight = torch.where(same_parent, local_weight, torch.zeros_like(local_weight))
        blend_weight: torch.Tensor | float = local_weight.view(1, 1, -1)
    else:
        blend_weight = 0.0
    mixed = (1.0 - blend_weight) * logits + blend_weight * secondary_aligned
    if secondary_mix_temperature != 1.0:
        center = mixed.mean(dim=1, keepdim=True)
        mixed = center + (mixed - center) / secondary_mix_temperature
    return mixed


class X138AttentionTracker(nn.Module):
    """Update primary forward/reverse weights and forward-only past-candidate attention."""

    def __init__(
        self,
        attention_tracker: PastCandidateAttentionTracker,
        *,
        fusion_kwargs: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.attention_tracker = attention_tracker
        self.fusion_kwargs = dict(fusion_kwargs or {})

    def forward(
        self,
        features_src: torch.Tensor,
        features_tgt: torch.Tensor,
        coords_src: torch.Tensor,
        coords_tgt: torch.Tensor,
        coords_prev_physical: torch.Tensor,
        coords_src_physical: torch.Tensor,
        coords_tgt_physical: torch.Tensor,
        prev_mask: torch.Tensor,
        candidate_ids_prev: torch.Tensor,
        secondary_logits: torch.Tensor,
    ) -> torch.Tensor:
        forward = self.attention_tracker(
            features_src,
            features_tgt,
            coords_src,
            coords_tgt,
            coords_prev_physical=coords_prev_physical,
            coords_src_physical=coords_src_physical,
            coords_tgt_physical=coords_tgt_physical,
            prev_mask=prev_mask,
            candidate_ids_prev=candidate_ids_prev,
        )
        reverse = self.attention_tracker.base_tracker(
            features_tgt, features_src, coords_tgt, coords_src
        )
        return x138_fuse_logits(forward, reverse, secondary_logits, **self.fusion_kwargs)
