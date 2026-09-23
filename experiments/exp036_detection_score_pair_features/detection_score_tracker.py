from __future__ import annotations

import torch


def build_detection_score_pair_features(
    source_scores: torch.Tensor,
    target_scores: torch.Tensor,
    *,
    use_detection_score_values: bool,
) -> torch.Tensor:
    if source_scores.ndim != 2 or target_scores.ndim != 2:
        raise ValueError("batched detection scores must have shape [batch, candidates]")
    if source_scores.shape[0] != target_scores.shape[0]:
        raise ValueError("source and target detection-score batches differ")
    source = source_scores.unsqueeze(2).expand(-1, -1, target_scores.shape[1])
    target = target_scores.unsqueeze(1).expand(-1, source_scores.shape[1], -1)
    features = torch.stack(
        (source, target, torch.minimum(source, target), torch.abs(source - target)),
        dim=-1,
    )
    return features if use_detection_score_values else torch.zeros_like(features)


class DetectionScorePairTracker(torch.nn.Module):
    def __init__(
        self,
        base_tracker: torch.nn.Module,
        *,
        pair_feature_dim: int,
        head_hidden_dim: int,
        use_detection_score_values: bool,
    ) -> None:
        super().__init__()
        if pair_feature_dim != 4:
            raise ValueError("the approved detection-score pair contract requires four features")
        self.base_tracker = base_tracker
        self.use_detection_score_values = bool(use_detection_score_values)
        self.score_pair_head = torch.nn.Sequential(
            torch.nn.Linear(pair_feature_dim, head_hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(head_hidden_dim, 1),
        )
        torch.nn.init.zeros_(self.score_pair_head[-1].weight)
        torch.nn.init.zeros_(self.score_pair_head[-1].bias)

    def forward(
        self,
        features_src: torch.Tensor,
        features_tgt: torch.Tensor,
        coords_src: torch.Tensor,
        coords_tgt: torch.Tensor,
        source_mask: torch.Tensor,
        target_mask: torch.Tensor,
        detection_scores_src: torch.Tensor,
        detection_scores_tgt: torch.Tensor,
    ) -> torch.Tensor:
        base_logits = self.base_tracker(
            features_src,
            features_tgt,
            coords_src,
            coords_tgt,
            source_mask,
            target_mask,
        )
        pair_features = build_detection_score_pair_features(
            detection_scores_src,
            detection_scores_tgt,
            use_detection_score_values=self.use_detection_score_values,
        )
        residual = self.score_pair_head(pair_features).squeeze(-1)
        valid_pairs = source_mask.unsqueeze(2) & target_mask.unsqueeze(1)
        return base_logits + residual.masked_fill(~valid_pairs, 0.0)
