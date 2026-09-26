"""CPU-only 2x2 ablation of the existing exp044 fold checkpoints on pair captures.

The public-primary + attention cell reuses the co-trained attention delta; it is
a counterfactual at evaluation time, not an independently trained model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
import zarr

import frozen_tracker as labels
from past_candidate_attention import PastCandidateAttentionTracker
from x138_data import X138PairDataset
from x138_tracking import X138AttentionTracker, x138_fuse_logits

VARIANTS = (
    "public_no_attention",
    "public_with_trained_attention",
    "trained_no_attention",
    "trained_with_attention",
)
THRESHOLDS = [index / 100 for index in range(5, 96)]


def read_geff_direct(path: Path, spacing: tuple[float, float, float]) -> labels.AnnotationGraph:
    """Read the GEFF node/edge arrays without optional tracksdata dependencies."""
    group = zarr.open_group(str(path), mode="r")
    ids = np.asarray(group["nodes/ids"][:], dtype=np.int64)
    frames_raw = np.asarray(group["nodes/props/t/values"][:], dtype=np.int64)
    coords_raw = np.stack(
        [np.asarray(group[f"nodes/props/{axis}/values"][:], dtype=np.float64) for axis in "zyx"],
        axis=1,
    )
    edge_array = np.asarray(group["edges/ids"][:], dtype=np.int64)
    if len(np.unique(ids)) != len(ids) or coords_raw.shape != (len(ids), 3):
        raise ValueError(f"Invalid GEFF nodes: {path}")
    edges = frozenset((int(source), int(target)) for source, target in edge_array)
    known_ids = set(ids.tolist())
    if any(source not in known_ids or target not in known_ids for source, target in edges):
        raise ValueError(f"Dangling GEFF edge: {path}")
    native_scale = np.asarray(spacing, dtype=np.float64)
    annotations = {}
    for frame in np.unique(frames_raw):
        selected = np.where(frames_raw == frame)[0]
        selected = selected[np.argsort(ids[selected])]
        annotations[int(frame)] = labels.FrameAnnotation(
            node_ids=ids[selected],
            coords_physical=np.asarray(coords_raw[selected] * native_scale, dtype=np.float32),
        )
    outgoing: dict[int, list[int]] = defaultdict(list)
    for source, target in edges:
        outgoing[source].append(target)
    canonical = {
        "nodes": sorted(
            (
                [int(node_id), int(frame), *map(float, raw)]
                for node_id, frame, raw in zip(ids, frames_raw, coords_raw, strict=True)
            ),
            key=lambda row: row[0],
        ),
        "edges": [list(edge) for edge in sorted(edges)],
    }
    return labels.AnnotationGraph(
        frames=annotations,
        edges=edges,
        content_sha256=labels.json_sha256(canonical),
        outgoing_edges={source: tuple(sorted(targets)) for source, targets in outgoing.items()},
    )


def build_model(config: dict, checkpoint: Path) -> X138AttentionTracker:
    from biohub_tracking.models.simple_node_transformer import SimpleNodeTransformer

    attention = config["model"]["attention"]
    base = SimpleNodeTransformer(**config["model"]["primary_params"])
    branch = PastCandidateAttentionTracker(
        base,
        feature_dim=int(attention["feature_dim"]),
        hidden_dim=int(attention["hidden_dim"]),
        source_chunk_size=int(attention["source_chunk_size"]),
        target_chunk_size=int(attention["target_chunk_size"]),
        past_candidate_chunk_size=int(attention["past_candidate_chunk_size"]),
        gradient_checkpointing=False,
        vector_scale_um=float(attention["vector_scale_um"]),
        cosine_epsilon=float(attention["cosine_epsilon"]),
        max_past_candidates=int(attention["max_past_candidates"]),
    )
    fusion = {
        key: float(value)
        for key, value in config["model"]["fusion"].items()
        if key != "secondary_link_mode"
    }
    model = X138AttentionTracker(branch, fusion_kwargs=fusion)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(saved["state_dict"], strict=True)
    model.eval()
    return model


def logits_by_variant(model: X138AttentionTracker, item: dict) -> dict[str, torch.Tensor]:
    tensor = {
        key: value.unsqueeze(0) for key, value in item.items() if isinstance(value, torch.Tensor)
    }
    branch = model.attention_tracker
    with torch.inference_mode():
        captured: list[torch.Tensor] = []
        hook = branch.base_tracker.register_forward_hook(
            lambda _module, _inputs, output: captured.append(output)
        )
        try:
            attention_forward = branch(
                tensor["features_src"],
                tensor["features_tgt"],
                tensor["coords_src"],
                tensor["coords_tgt"],
                coords_prev_physical=tensor["coords_prev_physical"],
                coords_src_physical=tensor["coords_src_physical"],
                coords_tgt_physical=tensor["coords_tgt_physical"],
                prev_mask=tensor["prev_mask"],
                candidate_ids_prev=tensor["candidate_ids_prev"],
            )
        finally:
            hook.remove()
        if len(captured) != 1:
            raise RuntimeError("Attention branch did not call its base tracker exactly once")
        trained_forward = captured[0]
        trained_reverse = branch.base_tracker(
            tensor["features_tgt"],
            tensor["features_src"],
            tensor["coords_tgt"],
            tensor["coords_src"],
        )
        delta = attention_forward - trained_forward
        captured_forward = tensor["primary_forward_logits"]
        captured_reverse = tensor["primary_reverse_logits"]
        secondary = tensor["secondary_logits"]
        fusion = model.fusion_kwargs
        return {
            "public_no_attention": tensor["x138_fused_logits"][0],
            "public_with_trained_attention": x138_fuse_logits(
                captured_forward + delta, captured_reverse, secondary, **fusion
            )[0],
            "trained_no_attention": x138_fuse_logits(
                trained_forward, trained_reverse, secondary, **fusion
            )[0],
            "trained_with_attention": x138_fuse_logits(
                attention_forward, trained_reverse, secondary, **fusion
            )[0],
        }


def empty_counts() -> dict:
    return {
        "windows": 0,
        "active_pairs": 0,
        "known_edges": 0,
        "division_parents": 0,
        "recovered_division_parents": 0,
        "true_positives_by_threshold": np.zeros(len(THRESHOLDS), dtype=np.int64),
        "false_positives_by_threshold": np.zeros(len(THRESHOLDS), dtype=np.int64),
    }


def add_pair(counts: dict, logits: torch.Tensor, truth: np.ndarray) -> np.ndarray:
    probability = torch.softmax(logits.float(), dim=0).numpy()
    active = truth.any(axis=1)[:, None] | truth.any(axis=0)[None, :]
    positive = np.sort(probability[truth])
    negative = np.sort(probability[active & ~truth])
    thresholds = np.asarray(THRESHOLDS, dtype=np.float32)
    counts["windows"] += 1
    counts["active_pairs"] += int(active.sum())
    counts["known_edges"] += len(positive)
    counts["true_positives_by_threshold"] += len(positive) - np.searchsorted(
        positive, thresholds, side="right"
    )
    counts["false_positives_by_threshold"] += len(negative) - np.searchsorted(
        negative, thresholds, side="right"
    )
    division = truth.sum(axis=1) > 1
    counts["division_parents"] += int(division.sum())
    predicted = probability > np.float32(0.48)
    counts["recovered_division_parents"] += int(((predicted | ~truth).all(axis=1) & division).sum())
    return predicted


def summarize(counts: dict) -> dict:
    curve = []
    for index, threshold in enumerate(THRESHOLDS):
        tp = int(counts["true_positives_by_threshold"][index])
        fp = int(counts["false_positives_by_threshold"][index])
        known = counts["known_edges"]
        curve.append(
            {
                "threshold": threshold,
                "recovered_known_edges": tp,
                "known_edge_recall": labels.safe_ratio(tp, known),
                "false_edges_on_active_pairs": fp,
                "active_pair_errors": known - tp + fp,
            }
        )
    fixed = curve[THRESHOLDS.index(0.48)]
    return {
        "windows": counts["windows"],
        "active_pairs": counts["active_pairs"],
        "known_edges": counts["known_edges"],
        "division_parents": counts["division_parents"],
        "recovered_division_parents": counts["recovered_division_parents"],
        "at_threshold_0_48": {
            **fixed,
            "active_pair_error_rate": labels.safe_ratio(
                fixed["active_pair_errors"], counts["active_pairs"]
            ),
            "division_parent_recall": labels.safe_ratio(
                counts["recovered_division_parents"], counts["division_parents"]
            ),
        },
        "threshold_curve": curve,
    }


def save_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def serialize_counts(counts: dict[str, dict]) -> dict[str, dict]:
    return {
        variant: {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in row.items()
        }
        for variant, row in counts.items()
    }


def restore_counts(saved: dict[str, dict]) -> dict[str, dict]:
    counts = {variant: empty_counts() for variant in VARIANTS}
    for variant in VARIANTS:
        for key, value in saved[variant].items():
            counts[variant][key] = (
                np.asarray(value, dtype=np.int64) if key.endswith("_by_threshold") else value
            )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--geff", type=Path, required=True)
    parser.add_argument(
        "--public-source",
        type=Path,
        required=True,
        help="Directory containing biohub_tracking package",
    )
    parser.add_argument("--public-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--max-pairs-per-embryo",
        type=int,
        default=0,
        help="Smoke test only; 0 requires complete capture",
    )
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    experiment = args.experiment.resolve()
    config = yaml.safe_load((experiment / "config.yaml").read_text())
    receipt = json.loads((experiment / "artifacts/kaggle_v1/train_receipt.json").read_text())
    prior = json.loads((experiment / "artifacts/kaggle_v1/pair_metrics.json").read_text())
    if labels.file_sha256(args.public_checkpoint) != config["model"]["primary_checkpoint_sha256"]:
        raise RuntimeError("Public checkpoint SHA differs")
    sys.path.insert(0, str(args.public_source.resolve()))
    selected = json.loads(
        (experiment / "artifacts/kaggle_v1/selected_train_videos.json").read_text()
    )
    groups = selected["groups"] if "groups" in selected else selected
    stems = sorted(stem for stems in groups.values() for stem in stems)
    paths_by_stem = {stem: sorted((args.capture / stem).glob("*.npz")) for stem in stems}
    if not args.max_pairs_per_embryo:
        if any(not paths for paths in paths_by_stem.values()):
            raise RuntimeError("Capture missing for at least one selected video")
        digest = hashlib.sha256()
        for stem in stems:
            for path in paths_by_stem[stem]:
                digest.update(stem.encode())
                digest.update(path.name.encode())
                digest.update(labels.file_sha256(path).encode())
        if digest.hexdigest() != receipt["capture_sha256"]:
            raise RuntimeError("Capture SHA differs from Kaggle train receipt")
    spacing = tuple(float(value) for value in config["data"]["native_spacing_zyx_um"])
    annotations = {stem: read_geff_direct(args.geff / f"{stem}.geff", spacing) for stem in stems}
    output = {
        "capture_sha256": receipt["capture_sha256"] if not args.max_pairs_per_embryo else None,
        "public_checkpoint_sha256": config["model"]["primary_checkpoint_sha256"],
        "public_source_sha256": labels.file_sha256(
            args.public_source / "biohub_tracking/models/simple_node_transformer.py"
        ),
        "threshold": 0.48,
        "counterfactual_warning": (
            "public_with_trained_attention applies a co-trained delta, "
            "not a separately trained public-primary attention model"
        ),
        "embryos": {},
    }
    if args.output.is_file():
        previous = json.loads(args.output.read_text())
        for key in ("capture_sha256", "public_checkpoint_sha256", "public_source_sha256"):
            if previous.get(key) != output[key]:
                raise RuntimeError(f"Existing diagnostic has different {key}")
        output = previous
    progress_path = args.output.with_name(args.output.name + ".progress.json")
    progress = json.loads(progress_path.read_text()) if progress_path.is_file() else None
    for embryo, embryo_stems in groups.items():
        paths = [
            path
            for stem in embryo_stems
            for path in paths_by_stem[stem]
            if all(int(frame) in annotations[stem].frames for frame in path.stem.split("_"))
        ]
        if args.max_pairs_per_embryo:
            paths = paths[: args.max_pairs_per_embryo]
        elif len(paths) != int(prior[embryo]["control"]["windows"]):
            raise RuntimeError(f"Eligible pair count differs for {embryo}: {len(paths)}")
        paths_sha256 = labels.json_sha256([str(path.relative_to(args.capture)) for path in paths])
        if embryo in output["embryos"]:
            print(f"{embryo}: already complete; skipping", flush=True)
            continue
        model = build_model(
            config, experiment / f"artifacts/kaggle_v1/models/fold_{prior[embryo]['fold']}.pt"
        )
        counts = {variant: empty_counts() for variant in VARIANTS}
        effects = {
            key: {
                "changed_active_pairs": 0,
                "gained_known_edges": 0,
                "lost_known_edges": 0,
                "gained_false_edges": 0,
                "lost_false_edges": 0,
            }
            for key in ("primary_only", "attention_on_public", "attention_on_trained")
        }
        resume_index = 0
        if progress is not None and progress.get("embryo") == embryo:
            if (
                progress.get("capture_sha256") != output["capture_sha256"]
                or progress.get("public_source_sha256") != output["public_source_sha256"]
                or progress.get("paths_sha256") != paths_sha256
                or progress.get("total_paths") != len(paths)
            ):
                raise RuntimeError(f"Saved progress has different inputs for {embryo}")
            resume_index = int(progress["processed"])
            if not 0 <= resume_index <= len(paths):
                raise RuntimeError("Invalid saved progress index")
            counts = restore_counts(progress["counts"])
            effects = progress["effects"]
            print(f"{embryo}: resumed at {resume_index}/{len(paths)}", flush=True)

        dataset = X138PairDataset(
            paths, annotations, match_radius_um=float(config["data"]["candidate_match_radius_um"])
        )
        for offset in range(resume_index, len(dataset)):
            index = offset + 1
            item = dataset[offset]
            truth = item["target"].numpy().astype(bool)
            decisions = {}
            for variant, logits in logits_by_variant(model, item).items():
                decisions[variant] = add_pair(counts[variant], logits, truth)
            active = truth.any(axis=1)[:, None] | truth.any(axis=0)[None, :]
            for key, before_name, after_name in (
                ("primary_only", "public_no_attention", "trained_no_attention"),
                ("attention_on_public", "public_no_attention", "public_with_trained_attention"),
                ("attention_on_trained", "trained_no_attention", "trained_with_attention"),
            ):
                before, after = decisions[before_name], decisions[after_name]
                gained, lost = ~before & after, before & ~after
                effects[key]["changed_active_pairs"] += int(((gained | lost) & active).sum())
                effects[key]["gained_known_edges"] += int((gained & truth).sum())
                effects[key]["lost_known_edges"] += int((lost & truth).sum())
                effects[key]["gained_false_edges"] += int((gained & active & ~truth).sum())
                effects[key]["lost_false_edges"] += int((lost & active & ~truth).sum())
            if index % 50 == 0:
                save_json_atomic(
                    progress_path,
                    {
                        "embryo": embryo,
                        "processed": index,
                        "total_paths": len(paths),
                        "paths_sha256": paths_sha256,
                        "capture_sha256": output["capture_sha256"],
                        "public_source_sha256": output["public_source_sha256"],
                        "counts": serialize_counts(counts),
                        "effects": effects,
                    },
                )
                print(f"{embryo}: {index}/{len(dataset)}", flush=True)
        result = {variant: summarize(value) for variant, value in counts.items()}
        result["decision_changes_at_threshold_0_48"] = effects
        mismatches = []
        if not args.max_pairs_per_embryo:
            for variant, expected_key in (
                ("public_no_attention", "control"),
                ("trained_with_attention", "attention"),
            ):
                row = result[variant]
                fixed = row["at_threshold_0_48"]
                expected = prior[embryo][expected_key]
                for key in (
                    "windows",
                    "active_pairs",
                    "known_edges",
                    "recovered_known_edges",
                    "false_edges_on_active_pairs",
                    "active_pair_errors",
                    "division_parents",
                    "recovered_division_parents",
                ):
                    observed = row[key] if key in row else fixed[key]
                    if observed != expected[key]:
                        mismatches.append(
                            {
                                "variant": variant,
                                "metric": key,
                                "observed": observed,
                                "expected": expected[key],
                            }
                        )
        result["baseline_parity"] = {
            "passed": None if args.max_pairs_per_embryo else not mismatches,
            "mismatches": mismatches,
        }
        output["embryos"][embryo] = result
        save_json_atomic(args.output, output)
        print(f"{embryo}: complete, parity_mismatches={len(mismatches)}", flush=True)
    if any(row["baseline_parity"]["passed"] is False for row in output["embryos"].values()):
        raise RuntimeError("Full-pair diagnostic differs from Kaggle pair metrics; inspect output")


if __name__ == "__main__":
    main()
