"""Kaggle-only exp043 pair-cache audit and exp044 tracker training."""

from __future__ import annotations

import hashlib
import importlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

import frozen_tracker as labels
from past_candidate_attention import PastCandidateAttentionTracker
from x138_data import X138PairDataset, known_parent_retention, read_capture
from x138_tracking import X138AttentionTracker, x138_fuse_logits


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=float) + "\n")


def _as_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _model_logits(model: X138AttentionTracker, batch: dict[str, Any]) -> torch.Tensor:
    return model(
        batch["features_src"],
        batch["features_tgt"],
        batch["coords_src"],
        batch["coords_tgt"],
        batch["coords_prev_physical"],
        batch["coords_src_physical"],
        batch["coords_tgt_physical"],
        batch["prev_mask"],
        batch["candidate_ids_prev"],
        batch["secondary_logits"],
    )


def _metrics(
    dataset: X138PairDataset,
    model: X138AttentionTracker | None,
    device: torch.device,
    threshold: float,
) -> dict[str, Any]:
    counts = {
        key: 0
        for key in (
            "windows",
            "active_pairs",
            "active_pair_errors",
            "known_edges",
            "recovered_known_edges",
            "false_edges_on_active_pairs",
            "division_parents",
            "recovered_division_parents",
        )
    }
    if model is not None:
        model.eval()
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0):
            batch = _as_device(batch, device)
            logits = batch["x138_fused_logits"] if model is None else _model_logits(model, batch)
            truth = batch["target"][0].bool()
            predicted = torch.softmax(logits[0].float(), dim=0) > threshold
            active_rows = truth.any(dim=1)
            active_cols = truth.any(dim=0)
            active = active_rows[:, None] | active_cols[None, :]
            division = truth.sum(dim=1) > 1
            counts["windows"] += 1
            counts["active_pairs"] += int(active.sum())
            counts["active_pair_errors"] += int(((predicted != truth) & active).sum())
            counts["known_edges"] += int(truth.sum())
            counts["recovered_known_edges"] += int((predicted & truth).sum())
            counts["false_edges_on_active_pairs"] += int((predicted & ~truth & active).sum())
            counts["division_parents"] += int(division.sum())
            counts["recovered_division_parents"] += int(
                ((predicted | ~truth).all(dim=1) & division).sum()
            )
    counts["known_edge_recall"] = labels.safe_ratio(
        counts["recovered_known_edges"], counts["known_edges"]
    )
    counts["active_pair_error_rate"] = labels.safe_ratio(
        counts["active_pair_errors"], counts["active_pairs"]
    )
    counts["division_parent_recall"] = labels.safe_ratio(
        counts["recovered_division_parents"], counts["division_parents"]
    )
    return counts


def train_from_capture(
    *,
    config: dict[str, Any],
    cache_root: Path,
    train_dir: Path,
    public_repo: Path,
    working_dir: Path,
    groups: dict[str, list[str]],
    capture_seconds: float,
) -> None:
    if not Path("/kaggle/input").is_dir() or not torch.cuda.is_available():
        raise RuntimeError("The authoritative train run requires Kaggle GPU")
    training = config["model"]["training"]
    attention = config["model"]["attention"]
    validation = config["validation"]
    stems = sorted(stem for group in groups.values() for stem in group)
    paths_by_stem = {stem: sorted((cache_root / stem).glob("*.npz")) for stem in stems}
    if any(not paths for paths in paths_by_stem.values()):
        raise RuntimeError("One or more selected videos have no captured pairs")
    capture_digest = hashlib.sha256()
    for stem in stems:
        for path in paths_by_stem[stem]:
            capture_digest.update(stem.encode())
            capture_digest.update(path.name.encode())
            capture_digest.update(labels.file_sha256(path).encode())
    capture_sha256 = capture_digest.hexdigest()
    annotations = {
        stem: labels.load_annotation_graph(
            train_dir / f"{stem}.geff",
            tuple(float(x) for x in config["data"]["native_spacing_zyx_um"]),
        )
        for stem in stems
    }
    eligible = {
        stem: [
            path
            for path in paths_by_stem[stem]
            if all(
                int(frame) in annotations[stem].frames
                for frame in read_capture(path)["window_frames"]
            )
        ]
        for stem in stems
    }
    if any(not paths for paths in eligible.values()):
        raise RuntimeError("No annotated pair windows for one or more videos")
    retention = {}
    for embryo, group in groups.items():
        recovered = total = 0
        for stem in group:
            for path in eligible[stem]:
                found, n = known_parent_retention(
                    read_capture(path),
                    annotations[stem],
                    float(config["data"]["candidate_match_radius_um"]),
                    int(attention["max_past_candidates"]),
                )
                recovered += found
                total += n
        retention[embryo] = {
            "recovered": recovered,
            "eligible": total,
            "recall": labels.safe_ratio(recovered, total),
        }
    _write_json(working_dir / "candidate_retention.json", retention)
    if any(
        not row["eligible"] or row["recall"] < float(validation["candidate_retention_minimum"])
        for row in retention.values()
    ):
        raise RuntimeError("K=8 known-parent retention failed before training")

    source_path = public_repo / "src"
    sys.path.insert(0, str(source_path))
    Transformer = importlib.import_module(
        "biohub_tracking.models.simple_node_transformer"
    ).SimpleNodeTransformer
    checkpoint = public_repo / "weights/unet_transformer/split_0/edge_predictor_best.pth"
    actual_sha = labels.file_sha256(checkpoint)
    if actual_sha != config["model"]["primary_checkpoint_sha256"]:
        raise RuntimeError("Public primary checkpoint SHA differs")
    state = labels.extract_public_tracker_state(
        torch.load(checkpoint, map_location="cpu", weights_only=True)
    )
    device = torch.device("cuda:0")
    seed = int(training["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    def new_model() -> X138AttentionTracker:
        params = config["model"]["primary_params"]
        base = Transformer(**params)
        base.load_state_dict(state, strict=True)
        branch = PastCandidateAttentionTracker(
            base,
            feature_dim=int(attention["feature_dim"]),
            hidden_dim=int(attention["hidden_dim"]),
            source_chunk_size=int(attention["source_chunk_size"]),
            target_chunk_size=int(attention["target_chunk_size"]),
            past_candidate_chunk_size=int(attention["past_candidate_chunk_size"]),
            gradient_checkpointing=bool(attention["gradient_checkpointing"]),
            vector_scale_um=float(attention["vector_scale_um"]),
            cosine_epsilon=float(attention["cosine_epsilon"]),
            max_past_candidates=int(attention["max_past_candidates"]),
        )
        added = sum(p.numel() for p in branch.parameters()) - sum(
            p.numel() for p in base.parameters()
        )
        if added != int(attention["expected_added_parameter_count"]):
            raise RuntimeError(f"attention parameter count differs: {added}")
        fusion = {
            key: float(value)
            for key, value in config["model"]["fusion"].items()
            if key != "secondary_link_mode"
        }
        if config["model"]["fusion"]["secondary_link_mode"] != "low_margin_consensus":
            raise RuntimeError("x138 secondary fusion mode changed")
        return X138AttentionTracker(branch, fusion_kwargs=fusion).to(device)

    reference = new_model().eval()
    parity = {
        key: 0.0
        for key in (
            "primary_forward_max_abs",
            "primary_reverse_max_abs",
            "fusion_replay_max_abs",
            "zero_delta_full_max_abs",
        )
    }
    parity["paths"] = []
    for group in groups.values():
        stem = group[0]
        choices = eligible[stem]
        for path in dict.fromkeys((choices[0], choices[len(choices) // 2], choices[-1])):
            sample = X138PairDataset([path], annotations)
            batch = _as_device(next(iter(DataLoader(sample, batch_size=1))), device)
            with torch.no_grad():
                forward = reference.attention_tracker.base_tracker(
                    batch["features_src"],
                    batch["features_tgt"],
                    batch["coords_src"],
                    batch["coords_tgt"],
                )
                reverse = reference.attention_tracker.base_tracker(
                    batch["features_tgt"],
                    batch["features_src"],
                    batch["coords_tgt"],
                    batch["coords_src"],
                )
                fused = _model_logits(reference, batch)
                replay = x138_fuse_logits(
                    batch["primary_forward_logits"],
                    batch["primary_reverse_logits"],
                    batch["secondary_logits"],
                    **reference.fusion_kwargs,
                )
            observed = {
                "primary_forward_max_abs": float(
                    (forward - batch["primary_forward_logits"]).abs().max()
                ),
                "primary_reverse_max_abs": float(
                    (reverse - batch["primary_reverse_logits"]).abs().max()
                ),
                "fusion_replay_max_abs": float((replay - batch["x138_fused_logits"]).abs().max()),
                "zero_delta_full_max_abs": float((fused - batch["x138_fused_logits"]).abs().max()),
            }
            for key, value in observed.items():
                parity[key] = max(parity[key], value)
            parity["paths"].append(str(path))
    _write_json(working_dir / "baseline_parity.json", parity)
    if max(value for key, value in parity.items() if key.endswith("max_abs")) > float(
        validation["baseline_logit_max_abs_error"]
    ):
        raise RuntimeError("Exp043 pair-logit replay failed before training")

    started = time.monotonic()
    deadline = started + 60 * float(training["budget_minutes"])
    records = {}
    output_models = working_dir / "models"
    output_models.mkdir(exist_ok=True)
    for fold, held_out in enumerate(groups):
        train_stems = sorted(set(stems) - set(groups[held_out]))
        internal_count = int(training["internal_validation_videos_per_embryo"])
        internal_stems = train_stems[:internal_count]
        fit_stems = train_stems[internal_count:]
        if not fit_stems or not internal_stems:
            raise RuntimeError("Invalid internal video split")

        def make_dataset(subset: list[str]) -> X138PairDataset:
            return X138PairDataset(
                [path for stem in subset for path in eligible[stem]],
                annotations,
                match_radius_um=float(config["data"]["candidate_match_radius_um"]),
            )

        fit_data = make_dataset(fit_stems)
        internal_data = make_dataset(internal_stems)
        held_data = make_dataset(groups[held_out])
        model = new_model()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
        )
        best_score = float("inf")
        best_state = None
        epochs = []
        for epoch in range(int(training["epochs"])):
            model.train()
            losses = []
            generator = torch.Generator().manual_seed(seed + fold * 100 + epoch)
            loader = DataLoader(
                fit_data, batch_size=1, shuffle=True, generator=generator, num_workers=0
            )
            for raw_batch in loader:
                if time.monotonic() > deadline:
                    raise TimeoutError("Tracker training budget exceeded")
                item = _as_device(raw_batch, device)
                optimizer.zero_grad(set_to_none=True)
                logits = _model_logits(model, item)
                loss = labels.legacy_focal_bce(
                    logits[0].float(),
                    item["target"][0].float(),
                    gamma=float(training["focal_gamma"]),
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite tracker loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(training["gradient_clip_norm"])
                )
                optimizer.step()
                losses.append(float(loss.detach()))
            internal = _metrics(
                internal_data, model, device, float(validation["edge_probability_threshold"])
            )
            score = internal["active_pair_error_rate"]
            if score <= best_score:
                best_score = score
                best_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
            epochs.append(
                {"epoch": epoch, "mean_train_loss": float(np.mean(losses)), "internal": internal}
            )
            _write_json(working_dir / f"fold_{fold}_progress.json", epochs)
        if best_state is None:
            raise RuntimeError("No tracker checkpoint selected")
        model.load_state_dict(best_state)
        model_path = output_models / f"fold_{fold}.pt"
        torch.save(
            {
                "state_dict": best_state,
                "held_out_embryo": held_out,
                "primary_checkpoint_sha256": actual_sha,
            },
            model_path,
        )
        records[held_out] = {
            "fold": fold,
            "fit_videos": fit_stems,
            "internal_videos": internal_stems,
            "outer_videos": groups[held_out],
            "control": _metrics(
                held_data, None, device, float(validation["edge_probability_threshold"])
            ),
            "attention": _metrics(
                held_data, model, device, float(validation["edge_probability_threshold"])
            ),
            "checkpoint_sha256": labels.file_sha256(model_path),
            "epochs": epochs,
        }
        _write_json(working_dir / "pair_metrics.json", records)
    receipt = {
        "experiment": config["experiment"]["name"],
        "capture_seconds": capture_seconds,
        "capture_sha256": capture_sha256,
        "training_seconds": time.monotonic() - started,
        "videos": stems,
        "window_counts": {stem: len(eligible[stem]) for stem in stems},
        "primary_checkpoint_sha256": actual_sha,
        "coordinate_head_sha256": config["model"]["coordinate_head_sha256"],
        "candidate_retention": retention,
        "baseline_parity": parity,
        "pair_metrics": records,
        "official_score": None,
        "graph_inference_performed": False,
    }
    _write_json(working_dir / "train_receipt.json", receipt)
