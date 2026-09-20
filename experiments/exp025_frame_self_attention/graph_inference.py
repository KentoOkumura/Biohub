from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

EXPECTED_EXP015_ORACLE_MANIFEST_SHA256 = (
    "5a511b8d5c8b25f6cab76563a32a63cde8b1bd74b83d5d38cacd64034396b28e"
)
EXPECTED_EXP015_CACHE_SUMMARY_SHA256 = (
    "040d1f6437e27149e0e34ad4aac8bba3c8cf63dc266d33df497acd969972194c"
)
EXPECTED_EXP015_CACHE_IDENTITY_SHA256 = (
    "440891c4550adf8540e3c47f9784e68b6ca1b64bbe56dbb93a25eb87c46bc1ee"
)
EXPECTED_PUBLIC_PRIMARY_CHECKPOINT_SHA256 = (
    "12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771"
)
EXPECTED_CACHE_SAMPLE_COUNT = 199
EXPECTED_CACHE_WINDOW_COUNT = 19_701
REQUIRED_REPLAY_ARRAYS = (
    "candidate_ids_src",
    "candidate_ids_tgt",
    "coords_src_grid",
    "coords_tgt_grid",
    "detection_scores_src",
    "detection_scores_tgt",
    "position_features_src",
    "position_features_tgt",
    "candidate_mask_src",
    "candidate_mask_tgt",
    "primary_features_src",
    "primary_features_tgt",
    "secondary_features_src",
    "secondary_features_tgt",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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


def resolve_verified_train_output(input_root: Path, expected_sha256: str | None) -> Path:
    if not expected_sha256 or len(expected_sha256) != 64:
        raise RuntimeError("record the exp025 train manifest SHA before graph inference")
    matches = [
        path
        for path in input_root.rglob("model_manifest.json")
        if sha256_file(path) == expected_sha256
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one verified exp025 train output, found {matches}")
    manifest = json.loads(matches[0].read_text(encoding="utf-8"))
    if manifest.get("experiment") != "exp025_frame_self_attention":
        raise RuntimeError("verified model manifest has the wrong experiment")
    return matches[0].parent


def resolve_verified_exp015_output(input_root: Path) -> Path:
    matches: list[Path] = []
    for path in input_root.rglob("oracle_inference_manifest.json"):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("manifest_sha256") == EXPECTED_EXP015_ORACLE_MANIFEST_SHA256:
            matches.append(path)
    if len(matches) != 1:
        raise RuntimeError(f"expected one verified exp015 inference output, found {matches}")
    root = matches[0].parent
    final_graphs = sorted((root / "oracle_final_graphs").glob("*.npz"))
    if len(final_graphs) != 199:
        raise RuntimeError(f"expected 199 exp015 final graphs, found {len(final_graphs)}")
    return root


def _extract_tracker_state(full_state: dict[str, Any]) -> dict[str, Any]:
    nested_state = full_state.get("state_dict")
    if isinstance(nested_state, dict):
        return _extract_tracker_state(nested_state)
    for prefix in ("transformer.", "module.transformer."):
        selected = {
            key[len(prefix) :]: value for key, value in full_state.items() if key.startswith(prefix)
        }
        if selected:
            return selected
    tracker_roots = {"proj", "norm_in", "blocks", "norm_out", "pair_mlp"}
    observed_roots = {key.split(".", 1)[0] for key in full_state}
    if tracker_roots.issubset(observed_roots):
        return dict(full_state)
    raise ValueError("checkpoint does not contain a SimpleNodeTransformer state")


def _load_tracker(
    checkpoint_path: Path,
    device: Any,
    model_params: dict[str, Any],
    architecture: dict[str, bool] | None = None,
) -> Any:
    import torch
    from simple_node_transformer import SimpleNodeTransformer

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"tracker checkpoint is not a state dictionary: {checkpoint_path}")
    tracker_state = _extract_tracker_state(state)
    flags = architecture or {"use_temporal_self_attention": False, "use_cross_attention": True}
    tracker = SimpleNodeTransformer(
        feat_dim=int(model_params["feature_dim"]),
        hidden_dim=int(model_params["hidden_dim"]),
        n_heads=int(model_params["n_heads"]),
        n_blocks=int(model_params["n_blocks"]),
        mlp_ratio=float(model_params.get("mlp_ratio", 2.0)),
        dropout=float(model_params["dropout"]),
        pair_chunk_size=int(model_params["pair_chunk_size"]),
        use_temporal_self_attention=bool(flags["use_temporal_self_attention"]),
        use_cross_attention=bool(flags["use_cross_attention"]),
        n_self_blocks=int(model_params.get("n_self_blocks", 2)),
        identity_init_self_attention=bool(flags.get("identity_init_self_attention", False)),
    )
    tracker.load_state_dict(tracker_state, strict=True)
    tracker.to(device)
    tracker.eval()
    return tracker


def _load_predict_module(repo_dir: Path) -> Any:
    source = repo_dir / "scripts" / "predict_unet_transformer.py"
    spec = importlib.util.spec_from_file_location("exp025_cache_predict_module", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load fixed prediction module: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _read_cache_window(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np
    from window_cache import read_window_cache

    source_frame, target_frame = map(int, path.stem.split("_"))
    arrays, receipt = read_window_cache(
        path,
        expected_metadata={
            "experiment": "exp015_oracle_stage_limits",
            "dataset": path.parent.name,
            "window_frames": [source_frame, target_frame],
            "primary_checkpoint_sha256": EXPECTED_PUBLIC_PRIMARY_CHECKPOINT_SHA256,
        },
    )
    missing = sorted(set(REQUIRED_REPLAY_ARRAYS) - set(arrays))
    if missing:
        raise ValueError({"cache_arrays_missing": missing, "path": str(path)})
    for side in ("src", "tgt"):
        candidate_ids = np.asarray(arrays[f"candidate_ids_{side}"], dtype=np.int64)
        coords = np.asarray(arrays[f"coords_{side}_grid"])
        mask = np.asarray(arrays[f"candidate_mask_{side}"], dtype=bool)
        if candidate_ids.ndim != 1 or coords.shape != (len(candidate_ids), 3):
            raise ValueError(f"invalid cached candidates: {path} {side}")
        if mask.shape != (len(candidate_ids),) or not mask.all():
            raise ValueError(f"cached candidates contain padding: {path} {side}")
        for name in (
            f"position_features_{side}",
            f"primary_features_{side}",
            f"secondary_features_{side}",
        ):
            if np.asarray(arrays[name]).shape != (len(candidate_ids), 32):
                raise ValueError(f"invalid cached feature shape: {path} {name}")
            if not np.isfinite(arrays[name]).all():
                raise ValueError(f"non-finite cached feature: {path} {name}")
    return arrays, receipt


def _cached_tracker_logits(
    tracker: Any,
    arrays: dict[str, Any],
    *,
    feature_prefix: str,
    reverse: bool,
    device: Any,
    downsample_zyx: tuple[float, float, float],
) -> Any:
    import numpy as np
    import torch

    source_side, target_side = ("tgt", "src") if reverse else ("src", "tgt")

    def tensor(name: str, *, dtype: Any | None = None) -> Any:
        value = np.asarray(arrays[name])
        result = torch.from_numpy(value).unsqueeze(0).to(device)
        return result.to(dtype=dtype) if dtype is not None else result

    features_source = torch.cat(
        [
            tensor(f"{feature_prefix}_features_{source_side}", dtype=torch.float32),
            tensor(f"position_features_{source_side}", dtype=torch.float32),
        ],
        dim=-1,
    )
    features_target = torch.cat(
        [
            tensor(f"{feature_prefix}_features_{target_side}", dtype=torch.float32),
            tensor(f"position_features_{target_side}", dtype=torch.float32),
        ],
        dim=-1,
    )
    scale = torch.as_tensor(downsample_zyx, dtype=torch.float32, device=device)
    coords_source = tensor(f"coords_{source_side}_grid", dtype=torch.float32) * scale
    coords_target = tensor(f"coords_{target_side}_grid", dtype=torch.float32) * scale
    mask_source = tensor(f"candidate_mask_{source_side}", dtype=torch.bool)
    mask_target = tensor(f"candidate_mask_{target_side}", dtype=torch.bool)
    with torch.no_grad():
        return tracker(
            features_source,
            features_target,
            coords_source,
            coords_target,
            mask_source,
            mask_target,
        )


def fuse_cached_edge_logits(
    primary_tracker: Any,
    secondary_tracker: Any,
    arrays: dict[str, Any],
    *,
    secondary_logits: Any | None = None,
    device: Any,
    downsample_zyx: tuple[float, float, float],
    bidirectional_weight: float,
    secondary_edge_weight: float,
    secondary_low_margin_max: float,
    secondary_mix_temperature: float,
) -> Any:
    import torch

    edge_logits = _cached_tracker_logits(
        primary_tracker,
        arrays,
        feature_prefix="primary",
        reverse=False,
        device=device,
        downsample_zyx=downsample_zyx,
    )
    if bidirectional_weight > 0.0:
        reverse_native = _cached_tracker_logits(
            primary_tracker,
            arrays,
            feature_prefix="primary",
            reverse=True,
            device=device,
            downsample_zyx=downsample_zyx,
        )
        reverse_logits = reverse_native.transpose(1, 2)
        forward_center = edge_logits.mean(dim=1, keepdim=True)
        forward_scale = edge_logits.float().std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)
        reverse_center = reverse_logits.mean(dim=1, keepdim=True)
        reverse_scale = (
            reverse_logits.float().std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)
        )
        reverse_ratio = (forward_scale / reverse_scale).clamp(0.5, 2.0).to(reverse_logits.dtype)
        reverse_aligned = (reverse_logits - reverse_center) * reverse_ratio + forward_center
        forward_prob = torch.softmax(edge_logits.float(), dim=1).clamp_min(1e-8)
        reverse_prob = torch.softmax(reverse_aligned.float(), dim=1).clamp_min(1e-8)
        harmonic_prob = 1.0 / (
            (1.0 - bidirectional_weight) / forward_prob + bidirectional_weight / reverse_prob
        )
        harmonic_prob = harmonic_prob / harmonic_prob.sum(dim=1, keepdim=True).clamp_min(1e-8)
        harmonic_logits = torch.log(harmonic_prob.clamp_min(1e-8))
        harmonic_center = harmonic_logits.mean(dim=1, keepdim=True)
        harmonic_scale = harmonic_logits.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)
        harmonic_ratio = (forward_scale / harmonic_scale).clamp(0.5, 2.0)
        edge_logits = ((harmonic_logits - harmonic_center) * harmonic_ratio + forward_center).to(
            reverse_aligned.dtype
        )

    if secondary_logits is None:
        secondary_logits = _cached_tracker_logits(
            secondary_tracker,
            arrays,
            feature_prefix="secondary",
            reverse=False,
            device=device,
            downsample_zyx=downsample_zyx,
        )
    primary_center = edge_logits.mean(dim=1, keepdim=True)
    primary_scale = edge_logits.float().std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)
    secondary_center = secondary_logits.mean(dim=1, keepdim=True)
    secondary_scale = (
        secondary_logits.float().std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)
    )
    secondary_ratio = (primary_scale / secondary_scale).clamp(0.5, 2.0)
    secondary_aligned = (secondary_logits - secondary_center) * secondary_ratio + primary_center
    n_source = int(edge_logits.shape[1])
    if n_source >= 2:
        primary_probs = torch.softmax(edge_logits[0], dim=0)
        secondary_probs = torch.softmax(secondary_aligned[0], dim=0)
        primary_top2 = torch.topk(primary_probs, k=2, dim=0)
        secondary_top2 = torch.topk(secondary_probs, k=2, dim=0)
        primary_margin = primary_top2.values[0] - primary_top2.values[1]
        same_parent = primary_top2.indices[0].eq(secondary_top2.indices[0])
        uncertainty = (
            (secondary_low_margin_max - primary_margin) / secondary_low_margin_max
        ).clamp(0.0, 1.0)
        local_weight = secondary_edge_weight * uncertainty
        local_weight = torch.where(same_parent, local_weight, torch.zeros_like(local_weight))
        blend_weight: Any = local_weight.view(1, 1, -1)
    else:
        blend_weight = 0.0
    edge_logits = (1.0 - blend_weight) * edge_logits + blend_weight * secondary_aligned
    if secondary_mix_temperature != 1.0:
        mixed_center = edge_logits.mean(dim=1, keepdim=True)
        edge_logits = mixed_center + (edge_logits - mixed_center) / secondary_mix_temperature
    return edge_logits


def select_cached_candidate_edges(
    edge_logits: Any,
    arrays: dict[str, Any],
    *,
    threshold: float,
) -> list[tuple[int, int, float, float]]:
    import numpy as np
    import torch

    probabilities = torch.softmax(edge_logits[0], dim=0).detach().cpu().numpy()
    source_ids = np.asarray(arrays["candidate_ids_src"], dtype=np.int64)
    target_ids = np.asarray(arrays["candidate_ids_tgt"], dtype=np.int64)
    source_coords = np.asarray(arrays["coords_src_grid"], dtype=np.float32)
    target_coords = np.asarray(arrays["coords_tgt_grid"], dtype=np.float32)
    ranked = sorted(
        [
            (probabilities[i, j], i, j)
            for i in range(len(source_ids))
            for j in range(len(target_ids))
            if probabilities[i, j] > threshold
        ],
        reverse=True,
    )
    return [
        (
            int(source_ids[i]),
            int(target_ids[j]),
            float(probability),
            float(np.linalg.norm(source_coords[i] - target_coords[j])),
        )
        for probability, i, j in ranked
    ]


def _register_cached_frame(
    registry: dict[int, tuple[int, int, int, int]],
    *,
    frame: int,
    candidate_ids: Any,
    coords_grid: Any,
    downsample_zyx: tuple[int, int, int],
) -> None:
    import numpy as np

    ids = np.asarray(candidate_ids, dtype=np.int64)
    grid = np.asarray(coords_grid, dtype=np.float32)
    scaled = (grid * np.asarray(downsample_zyx, dtype=np.float32)).astype(np.int16)
    for node_id, zyx in zip(ids, scaled, strict=True):
        record = (int(frame), int(zyx[0]), int(zyx[1]), int(zyx[2]))
        previous = registry.setdefault(int(node_id), record)
        if previous != record:
            raise ValueError({"candidate_id_changed_across_windows": int(node_id)})


def _coords_from_registry(registry: dict[int, tuple[int, int, int, int]]) -> Any:
    import numpy as np

    ids = sorted(registry)
    if ids != list(range(len(ids))):
        raise ValueError("cached candidate IDs are not contiguous from zero")
    return np.asarray([registry[index] for index in ids], dtype=np.int16).reshape((-1, 4))


def _graph_payload(graph: Any) -> dict[str, Any]:
    nodes = sorted(
        (
            int(row["node_id"]),
            int(row["t"]),
            float(row["z"]),
            float(row["y"]),
            float(row["x"]),
        )
        for row in graph.node_attrs().iter_rows(named=True)
    )
    edges = sorted(
        (
            int(row["source_id"]),
            int(row["target_id"]),
            float(row["edge_prob"]),
            float(row["edge_dist"]),
        )
        for row in graph.edge_attrs().iter_rows(named=True)
    )
    return {"nodes": nodes, "edges": edges}


def compare_control_candidate_graph(candidate_graph: Any, baseline_path: Path) -> dict[str, Any]:
    import tracksdata as td

    baseline_result = td.graph.IndexedRXGraph.from_geff(baseline_path)
    baseline_graph = baseline_result[0] if isinstance(baseline_result, tuple) else baseline_result
    observed = _graph_payload(candidate_graph)
    expected = _graph_payload(baseline_graph)
    if observed != expected:
        observed_edges = {(row[0], row[1]): row[2:] for row in observed["edges"]}
        expected_edges = {(row[0], row[1]): row[2:] for row in expected["edges"]}
        shared = sorted(set(observed_edges) & set(expected_edges))
        raise RuntimeError(
            {
                "cache_control_candidate_graph_mismatch": baseline_path.stem,
                "nodes_exact": observed["nodes"] == expected["nodes"],
                "missing_edges": sorted(set(expected_edges) - set(observed_edges))[:20],
                "extra_edges": sorted(set(observed_edges) - set(expected_edges))[:20],
                "shared_edge_value_mismatches": [
                    edge for edge in shared if observed_edges[edge] != expected_edges[edge]
                ][:20],
            }
        )
    return {
        "sample": baseline_path.stem,
        "node_count": len(observed["nodes"]),
        "edge_count": len(observed["edges"]),
        "content_sha256": json_sha256(observed),
        "exact": True,
    }


def _run_cache_worker(spec_path: Path) -> None:
    import numpy as np
    import torch

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    repo_dir = Path(spec["repo_dir"])
    cache_root = Path(spec["cache_root"])
    baseline_candidate_root = Path(spec["baseline_candidate_root"])
    candidate_output_root = Path(spec["candidate_output_root"])
    prediction_output_root = Path(spec["prediction_output_root"])
    candidate_output_root.mkdir(parents=True, exist_ok=True)
    prediction_output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("cache replay worker requires a visible CUDA device")
    predict_module = _load_predict_module(repo_dir)
    model_params = spec["model_params"]
    public_tracker = _load_tracker(Path(spec["public_weights"]), device, model_params)
    secondary_tracker = _load_tracker(Path(spec["secondary_weights"]), device, model_params)
    fold_trackers = {
        int(fold): _load_tracker(Path(path), device, model_params, spec["variant_architecture"])
        for fold, path in spec["fold_weights"].items()
    }
    downsample = tuple(int(value) for value in spec["downsample_zyx"])
    replay_cfg = spec["replay"]
    sample_records: list[dict[str, Any]] = []
    total_windows = 0
    read_seconds = 0.0
    started = time.perf_counter()

    for sample in spec["samples"]:
        embryo = sample.split("_", 1)[0]
        fold = int(spec["fold_model_by_embryo"][embryo])
        sample_paths = sorted((cache_root / sample).glob("*.npz"))
        if len(sample_paths) != int(spec["expected_windows_per_sample"]):
            raise RuntimeError(
                {
                    "sample": sample,
                    "expected_windows": spec["expected_windows_per_sample"],
                    "actual": len(sample_paths),
                }
            )
        registry: dict[int, tuple[int, int, int, int]] = {}
        public_edges: list[tuple[int, int, float, float]] = []
        retrained_edges: list[tuple[int, int, float, float]] = []
        for window_index, path in enumerate(sample_paths):
            source_frame, target_frame = map(int, path.stem.split("_"))
            if (source_frame, target_frame) != (window_index, window_index + 1):
                raise RuntimeError({"non_contiguous_cache_window": str(path)})
            arrays, cache_receipt = _read_cache_window(path)
            read_seconds += float(cache_receipt["read_seconds"])
            _register_cached_frame(
                registry,
                frame=source_frame,
                candidate_ids=arrays["candidate_ids_src"],
                coords_grid=arrays["coords_src_grid"],
                downsample_zyx=downsample,
            )
            _register_cached_frame(
                registry,
                frame=target_frame,
                candidate_ids=arrays["candidate_ids_tgt"],
                coords_grid=arrays["coords_tgt_grid"],
                downsample_zyx=downsample,
            )
            secondary_logits = _cached_tracker_logits(
                secondary_tracker,
                arrays,
                feature_prefix="secondary",
                reverse=False,
                device=device,
                downsample_zyx=downsample,
            )
            public_logits = fuse_cached_edge_logits(
                public_tracker,
                secondary_tracker,
                arrays,
                secondary_logits=secondary_logits,
                device=device,
                downsample_zyx=downsample,
                bidirectional_weight=float(replay_cfg["bidirectional_weight"]),
                secondary_edge_weight=float(replay_cfg["secondary_edge_weight"]),
                secondary_low_margin_max=float(replay_cfg["secondary_low_margin_max"]),
                secondary_mix_temperature=float(replay_cfg["secondary_mix_temperature"]),
            )
            retrained_logits = fuse_cached_edge_logits(
                fold_trackers[fold],
                secondary_tracker,
                arrays,
                secondary_logits=secondary_logits,
                device=device,
                downsample_zyx=downsample,
                bidirectional_weight=float(replay_cfg["bidirectional_weight"]),
                secondary_edge_weight=float(replay_cfg["secondary_edge_weight"]),
                secondary_low_margin_max=float(replay_cfg["secondary_low_margin_max"]),
                secondary_mix_temperature=float(replay_cfg["secondary_mix_temperature"]),
            )
            public_edges.extend(
                select_cached_candidate_edges(
                    public_logits, arrays, threshold=float(replay_cfg["edge_threshold"])
                )
            )
            retrained_edges.extend(
                select_cached_candidate_edges(
                    retrained_logits, arrays, threshold=float(replay_cfg["edge_threshold"])
                )
            )
            total_windows += 1
        coords = _coords_from_registry(registry)
        public_graph = predict_module.build_graph(coords, public_edges)
        control_receipt = compare_control_candidate_graph(
            public_graph, baseline_candidate_root / f"{sample}.geff"
        )
        retrained_graph = predict_module.build_graph(coords, retrained_edges)
        predict_module.save_graph(retrained_graph, candidate_output_root / f"{sample}.geff")
        if bool(replay_cfg["use_ilp"]) and retrained_graph.num_edges() > 0:
            solver = predict_module.td.solvers.ILPSolver(
                edge_weight=float(replay_cfg["ilp_edge_weight"])
                * predict_module.td.EdgeAttr("edge_prob"),
                appearance_weight=float(replay_cfg["ilp_appearance_weight"]),
                disappearance_weight=float(replay_cfg["ilp_disappearance_weight"]),
                division_weight=float(replay_cfg["ilp_division_weight"]),
            )
            with predict_module.suppress_output():
                retrained_graph = solver.solve(retrained_graph)
        predict_module.save_graph(retrained_graph, prediction_output_root / f"{sample}.geff")
        coordinate_bytes = np.ascontiguousarray(coords.astype("<i2", copy=False)).tobytes(order="C")
        sample_records.append(
            {
                **control_receipt,
                "embryo": embryo,
                "fold": fold,
                "window_count": len(sample_paths),
                "candidate_coordinate_sha256": hashlib.sha256(coordinate_bytes).hexdigest(),
                "retrained_candidate_edge_count": len(retrained_edges),
            }
        )
        print(
            f"CACHE_REPLAY sample={sample} fold={fold} nodes={len(coords)} "
            f"control_edges={len(public_edges)} retrained_edges={len(retrained_edges)}",
            flush=True,
        )

    receipt = {
        "worker": int(spec["worker"]),
        "samples": sample_records,
        "sample_count": len(sample_records),
        "window_count": total_windows,
        "cache_read_seconds": read_seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "main_image_encoder_forward_count": 0,
        "tracker_forward_count_per_window": 5,
        "public_control_candidate_graphs_exact": all(row["exact"] for row in sample_records),
    }
    receipt["receipt_sha256"] = json_sha256(receipt)
    Path(spec["receipt_path"]).write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _visible_cuda_tokens(count: int) -> list[str]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if raw and raw != "-1":
        tokens = [token.strip() for token in raw.split(",") if token.strip()]
        if len(tokens) < count:
            raise RuntimeError(f"expected {count} visible CUDA tokens, got {raw!r}")
        return tokens[:count]
    return [str(index) for index in range(count)]


def run_cached_graph_replay(
    *,
    repo_dir: Path,
    baseline_root: Path,
    public_weights: Path,
    secondary_weights: Path,
    fold_weights: dict[int, str],
    sample_names: list[str],
    method: str,
    working_dir: Path,
    model_params: dict[str, Any],
    replay_config: dict[str, Any],
    worker_count: int = 2,
    runtime_gate_seconds: float = 12 * 60 * 60,
) -> dict[str, Any]:
    from frozen_tracker import (
        discover_cache_paths,
        recompute_cache_identity_sha256,
        validate_cache_summary,
    )

    if len(sample_names) != EXPECTED_CACHE_SAMPLE_COUNT or len(set(sample_names)) != len(
        sample_names
    ):
        raise RuntimeError("cache replay requires 199 unique training samples")
    if worker_count != 2:
        raise ValueError("authoritative cache replay requires exactly two GPU workers")
    cache_root = baseline_root / "window_cache"
    summary_path = baseline_root / "window_cache_summary.json"
    cache_cfg = {
        "schema_version": 1,
        "expected_dataset_count": EXPECTED_CACHE_SAMPLE_COUNT,
        "expected_window_count": EXPECTED_CACHE_WINDOW_COUNT,
        "identity_sha256": EXPECTED_EXP015_CACHE_IDENTITY_SHA256,
        "summary_sha256": EXPECTED_EXP015_CACHE_SUMMARY_SHA256,
    }
    cache_summary = validate_cache_summary(summary_path, cache_cfg)
    cache_paths = discover_cache_paths(cache_root, cache_cfg)
    if recompute_cache_identity_sha256(cache_paths) != EXPECTED_EXP015_CACHE_IDENTITY_SHA256:
        raise RuntimeError("exp015 cache identity changed")
    baseline_candidate_root = baseline_root / "oracle_candidate_graphs"
    baseline_candidates = sorted(baseline_candidate_root.glob("*.geff"))
    if [path.stem for path in baseline_candidates] != sorted(sample_names):
        raise RuntimeError("exp015 candidate graph coverage changed")
    if sha256_file(public_weights) != EXPECTED_PUBLIC_PRIMARY_CHECKPOINT_SHA256:
        raise RuntimeError("public primary checkpoint changed before cache replay")

    prediction_output_root = repo_dir / "predictions" / "exp025" / method / "split_0"
    candidate_output_root = working_dir / "oracle_candidate_graphs"
    worker_root = working_dir / "cache_replay_workers"
    for path in (prediction_output_root, candidate_output_root, worker_root):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
    cuda_tokens = _visible_cuda_tokens(worker_count)
    worker_specs: list[Path] = []
    processes: list[subprocess.Popen[Any]] = []
    for worker in range(worker_count):
        assigned = sorted(sample_names)[worker::worker_count]
        receipt_path = worker_root / f"worker_{worker}_receipt.json"
        spec = {
            "worker": worker,
            "repo_dir": str(repo_dir),
            "cache_root": str(cache_root),
            "baseline_candidate_root": str(baseline_candidate_root),
            "candidate_output_root": str(candidate_output_root),
            "prediction_output_root": str(prediction_output_root),
            "public_weights": str(public_weights),
            "secondary_weights": str(secondary_weights),
            "fold_weights": {str(fold): path for fold, path in sorted(fold_weights.items())},
            "fold_model_by_embryo": {"6bba": 0, "44b6": 1},
            "samples": assigned,
            "expected_windows_per_sample": 99,
            "downsample_zyx": [1, 4, 4],
            "model_params": model_params,
            "variant_architecture": model_params["architecture"],
            "replay": replay_config,
            "receipt_path": str(receipt_path),
        }
        spec_path = worker_root / f"worker_{worker}_spec.json"
        spec_path.write_text(
            json.dumps(spec, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        python_paths = [str(repo_dir / "src"), str(repo_dir / "scripts")]
        inherited_pythonpath = os.environ.get("PYTHONPATH", "").strip()
        if inherited_pythonpath:
            python_paths.append(inherited_pythonpath)
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": cuda_tokens[worker],
            "PYTHONPATH": os.pathsep.join(python_paths),
        }
        command = [sys.executable, str(Path(__file__).resolve()), "cache-worker", str(spec_path)]
        print(f"EXP025 cache worker {worker}: {' '.join(command)}", flush=True)
        worker_specs.append(spec_path)
        processes.append(subprocess.Popen(command, cwd=repo_dir, env=env))

    deadline = time.monotonic() + runtime_gate_seconds
    while processes:
        for process in list(processes):
            return_code = process.poll()
            if return_code is None:
                continue
            processes.remove(process)
            if return_code != 0:
                for pending in processes:
                    pending.terminate()
                raise RuntimeError(f"cache replay worker failed with exit code {return_code}")
        if processes and time.monotonic() >= deadline:
            for process in processes:
                process.terminate()
            raise TimeoutError("cache replay exceeded the 12-hour runtime gate")
        if processes:
            time.sleep(1.0)

    receipts = [
        json.loads((worker_root / f"worker_{worker}_receipt.json").read_text(encoding="utf-8"))
        for worker in range(worker_count)
    ]
    records = sorted(
        [record for receipt in receipts for record in receipt["samples"]],
        key=lambda row: row["sample"],
    )
    predicted = sorted(prediction_output_root.glob("*.geff"))
    candidates = sorted(candidate_output_root.glob("*.geff"))
    if [path.stem for path in predicted] != sorted(sample_names):
        raise RuntimeError("cache replay ILP graph coverage mismatch")
    if [path.stem for path in candidates] != sorted(sample_names):
        raise RuntimeError("cache replay candidate graph coverage mismatch")
    if len(records) != EXPECTED_CACHE_SAMPLE_COUNT or not all(row["exact"] for row in records):
        raise RuntimeError("public control candidate graph replay is not exact")
    if sum(int(receipt["window_count"]) for receipt in receipts) != EXPECTED_CACHE_WINDOW_COUNT:
        raise RuntimeError("cache replay window coverage mismatch")
    summary = {
        "experiment": "exp025_frame_self_attention",
        "stage": "fixed_candidate_cache_graph_replay",
        "cache_summary_sha256": cache_summary["summary_sha256"],
        "cache_identity_sha256": EXPECTED_EXP015_CACHE_IDENTITY_SHA256,
        "sample_count": len(records),
        "window_count": EXPECTED_CACHE_WINDOW_COUNT,
        "main_image_encoder_forward_count": 0,
        "tracker_forward_count_per_window": 5,
        "tracker_forward_count": 5 * EXPECTED_CACHE_WINDOW_COUNT,
        "public_control_candidate_graphs_exact": True,
        "public_control_candidate_graph_content_sha256": json_sha256(
            [[row["sample"], row["content_sha256"]] for row in records]
        ),
        "candidate_coordinate_content_sha256": json_sha256(
            [[row["sample"], row["candidate_coordinate_sha256"]] for row in records]
        ),
        "fold_model_by_evaluation_embryo": {"6bba": 0, "44b6": 1},
        "worker_receipts": receipts,
        "worker_count": worker_count,
        "additional_training": False,
        "submission_created": False,
    }
    summary["summary_sha256"] = json_sha256(summary)
    (working_dir / "cache_graph_replay_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def build_hybrid_checkpoints(
    *,
    repo_dir: Path,
    method: str,
    public_weights_relative: str,
    train_output: Path,
    output_manifest_path: Path,
    selected_variant: str,
    expected_manifest_sha256: str,
    expected_architecture: dict[str, bool],
    expected_model_params: dict[str, Any],
) -> tuple[dict[int, str], dict[str, Any]]:
    import torch

    public_path = repo_dir / public_weights_relative
    if sha256_file(public_path) != EXPECTED_PUBLIC_PRIMARY_CHECKPOINT_SHA256:
        raise RuntimeError("public checkpoint SHA changed")
    manifest_path = train_output / "model_manifest.json"
    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise RuntimeError("train model manifest SHA changed")
    train_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model_source = Path("simple_node_transformer.py")
    if sha256_file(model_source) != train_manifest["model_source_sha256"]:
        raise RuntimeError("train and inference model source differ")
    if selected_variant not in train_manifest["active_variants"]:
        raise RuntimeError("selected variant is absent from train manifest")
    selected_architecture = train_manifest["architectures"][selected_variant]
    if selected_architecture != expected_architecture:
        raise RuntimeError("inference architecture differs from train manifest")
    if train_manifest["model_params"] != expected_model_params:
        raise RuntimeError("inference model parameters differ from train manifest")
    records = []
    relative_paths = {}
    for fold in (0, 1):
        matches = [
            item
            for item in train_manifest["models"]
            if item["variant"] == selected_variant and int(item["fold"]) == fold
        ]
        if len(matches) != 1:
            raise RuntimeError(f"expected one {selected_variant} model for fold {fold}")
        item = matches[0]
        model_path = train_output / item["path"]
        if sha256_file(model_path) != item["file_sha256"]:
            raise RuntimeError(f"{selected_variant} fold {fold} model SHA mismatch")
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        if checkpoint.get("variant") != selected_variant or checkpoint.get("fold") != fold:
            raise RuntimeError("checkpoint variant/fold metadata mismatch")
        if checkpoint.get("architecture") != selected_architecture:
            raise RuntimeError("checkpoint architecture differs from manifest")
        if checkpoint.get("model_source_sha256") != train_manifest["model_source_sha256"]:
            raise RuntimeError("checkpoint model source SHA differs from manifest")
        state = checkpoint.get("state_dict")
        if (
            not isinstance(state, dict)
            or canonical_state_sha256(state) != item["canonical_state_sha256"]
        ):
            raise RuntimeError("checkpoint state SHA differs from manifest")
        relative_paths[fold] = str(model_path)
        records.append(
            {
                "variant": selected_variant,
                "fold": fold,
                "evaluation_embryo": item["evaluation_embryo"],
                "source_tracker_file_sha256": item["file_sha256"],
                "source_tracker_state_sha256": item["canonical_state_sha256"],
                "path": item["path"],
            }
        )
    result = {
        "experiment": "exp025_frame_self_attention",
        "stage": "fold_specific_primary_tracker_selection",
        "selected_variant": selected_variant,
        "architecture": selected_architecture,
        "source_model_manifest_sha256": expected_manifest_sha256,
        "model_source_sha256": train_manifest["model_source_sha256"],
        "public_checkpoint_sha256": sha256_file(public_path),
        "fold_models": records,
        "model_count": 2,
        "additional_training": False,
    }
    result["manifest_sha256"] = json_sha256(result)
    output_manifest_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return relative_paths, result


_HYBRID_INJECTION = r"""
# EXP025_FOLD_TRACKER_INJECTION_START
_exp025_train_output = resolve_verified_train_output(
    Path('/kaggle/input/notebooks'),
    exp025_config['model']['inference']['train_output_manifest_sha256'],
)
WEIGHTS_RELATIVE_BY_FOLD, _exp025_hybrid_manifest = build_hybrid_checkpoints(
    repo_dir=REPO_DIR,
    method=METHOD,
    public_weights_relative=WEIGHTS_RELATIVE,
    train_output=_exp025_train_output,
    output_manifest_path=WORKING_DIR / 'fold_tracker_inference_manifest.json',
    selected_variant=exp025_config['model']['inference']['selected_variant'],
    expected_manifest_sha256=exp025_config['model']['inference']['train_output_manifest_sha256'],
    expected_architecture=exp025_config['model']['architectures'][
        exp025_config['model']['inference']['selected_variant']
    ],
    expected_model_params=exp025_config['model']['params'],
)
print('EXP025 fold-specific tracker checkpoints:', WEIGHTS_RELATIVE_BY_FOLD)
# EXP025_FOLD_TRACKER_INJECTION_END

"""


_CUSTOM_PREDICTION = r"""# EXP025_CACHE_REPLAY_START
start_time = time.time()
available_gpu_count = _torch.cuda.device_count()
if available_gpu_count < 2:
    raise RuntimeError(f'exp025 cache replay requires two T4 GPUs, found {available_gpu_count}')
if SLICE:
    raise RuntimeError('exp025 authoritative graph inference does not permit a partial slice')

_exp025_gate_seconds = 12 * 60 * 60
_exp025_elapsed_before_replay = time.monotonic() - started
_exp025_replay_gate_seconds = _exp025_gate_seconds - _exp025_elapsed_before_replay
if _exp025_replay_gate_seconds <= 0:
    raise TimeoutError('exp025 exhausted the 12-hour gate before cache replay')
(WORKING_DIR / 'inference_runtime_preflight.json').write_text(
    json.dumps({
        'route': 'fixed_candidate_cache_graph_replay',
        'superseded_raw_image_prediction_seconds': 23217.361345529556,
        'main_image_encoder_forward_count': 0,
        'cache_sample_count': 199,
        'cache_window_count': 19701,
        'runtime_gate_seconds': _exp025_gate_seconds,
        'visible_gpu_count': available_gpu_count,
    }, indent=2, sort_keys=True) + '\n'
)

_exp025_baseline_root = resolve_verified_exp015_output(Path('/kaggle/input/notebooks'))
_exp025_cache_replay_summary = run_cached_graph_replay(
    repo_dir=REPO_DIR,
    baseline_root=_exp025_baseline_root,
    public_weights=REPO_DIR / WEIGHTS_RELATIVE,
    secondary_weights=Path(SECONDARY_WEIGHTS_PATH),
    fold_weights=WEIGHTS_RELATIVE_BY_FOLD,
    sample_names=test_stems,
    method=METHOD,
    working_dir=WORKING_DIR,
    model_params={
        **exp025_config['model']['params'],
        'architecture': exp025_config['model']['architectures'][
            exp025_config['model']['inference']['selected_variant']
        ],
    },
    replay_config={
        'bidirectional_weight': float(os.environ['BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT']),
        'secondary_edge_weight': float(os.environ['BIOHUB_SECONDARY_EDGE_WEIGHT']),
        'secondary_low_margin_max': float(os.environ['BIOHUB_SECONDARY_LOW_MARGIN_MAX']),
        'secondary_mix_temperature': float(os.environ['BIOHUB_SECONDARY_MIX_TEMPERATURE']),
        'edge_threshold': float(os.environ['BIOHUB_DUAL_SEED_EDGE_THRESHOLD']),
        'use_ilp': bool(USE_ILP),
        'ilp_edge_weight': float(ILP_EDGE_WEIGHT),
        'ilp_appearance_weight': float(ILP_APPEARANCE_WEIGHT),
        'ilp_disappearance_weight': float(ILP_DISAPPEARANCE_WEIGHT),
        'ilp_division_weight': float(ILP_DIVISION_WEIGHT),
    },
    worker_count=2,
    runtime_gate_seconds=_exp025_replay_gate_seconds,
)
_exp025_assignment_rows = [
    {
        'sample': name,
        'embryo': name.split('_', 1)[0],
        'fold': 0 if name.startswith('6bba_') else 1,
    }
    for name in test_stems
]
(WORKING_DIR / 'fold_assignment_manifest.json').write_text(
    json.dumps(sorted(_exp025_assignment_rows, key=lambda row: row['sample']), indent=2) + '\n'
)
predict_seconds = time.time() - start_time
print(f'EXP025 cache graph replay completed in {predict_seconds / 60:.2f} minutes')
# EXP025_CACHE_REPLAY_END"""


def patch_exp015_source(source: str) -> str:
    injection_marker = (
        "# Save the thresholded detector/association graph before ILP without changing it."
    )
    if source.count(injection_marker) != 1:
        raise RuntimeError("exp015 hybrid-checkpoint injection marker changed")
    source = source.replace(injection_marker, _HYBRID_INJECTION + injection_marker, 1)

    start_marker = "start_time = time.time()\navailable_gpu_count = _torch.cuda.device_count()"
    end_marker = "print(f'Prediction completed in {predict_seconds / 60:.2f} minutes')"
    if source.count(start_marker) != 1 or source.count(end_marker) != 1:
        raise RuntimeError("exp015 prediction execution markers changed")
    start = source.index(start_marker)
    end = source.index(end_marker, start) + len(end_marker)
    source = source[:start] + _CUSTOM_PREDICTION + source[end:]
    repair_end_marker = "display(pd.read_csv(FINAL_GRAPH_ROWS_PATH, nrows = 8))"
    if source.count(repair_end_marker) != 1:
        raise RuntimeError("exp015 final graph repair marker changed")
    repair_end = source.index(repair_end_marker) + len(repair_end_marker)
    source = source[:repair_end] + "\nprint('EXP025 cache replay graph repair complete')\n"
    repair_loop_marker = "    for geff_path in geffs:\n        dataset = geff_path.stem"
    repair_loop_replacement = """    for geff_path in geffs:
        if time.monotonic() - started >= _exp025_gate_seconds:
            raise TimeoutError('exp025 cache replay and graph repair exceeded the 12-hour gate')
        dataset = geff_path.stem"""
    if source.count(repair_loop_marker) != 1:
        raise RuntimeError("exp015 graph repair loop marker changed")
    source = source.replace(repair_loop_marker, repair_loop_replacement, 1)
    source = source.replace(
        "print('Submission path:', FINAL_GRAPH_ROWS_PATH)",
        "print('Temporary row-form graph path:', FINAL_GRAPH_ROWS_PATH)",
    )
    if "EXP025_FOLD_TRACKER_INJECTION_START" not in source:
        raise RuntimeError("exp025 hybrid-checkpoint patch did not persist")
    if "EXP025_CACHE_REPLAY_START" not in source:
        raise RuntimeError("exp025 cache replay patch did not persist")
    compile(source, "exp015_inference_base_patched.py", "exec")
    return source


def compact_graph_content_sha256(root: Path) -> str:
    records = [[path.stem, sha256_file(path)] for path in sorted(root.glob("*.npz"))]
    if len(records) != 199:
        raise RuntimeError(f"expected 199 compact final graphs under {root}, found {len(records)}")
    return json_sha256(records)


def load_compact_graph_arrays(
    path: Path,
) -> tuple[list[dict[str, float | int]], list[tuple[int, int]]]:
    import numpy as np

    with np.load(path, allow_pickle=False) as payload:
        old_ids = np.asarray(payload["node_ids"], dtype=np.int64)
        tzyx = np.asarray(payload["node_tzyx"], dtype=np.float64)
        old_edges = np.asarray(payload["edges"], dtype=np.int64)
    if old_ids.ndim != 1 or tzyx.shape != (len(old_ids), 4):
        raise ValueError(f"invalid compact node arrays: {path}")
    if old_edges.shape != (len(old_edges), 2):
        raise ValueError(f"invalid compact edge array: {path}")
    if len(set(int(value) for value in old_ids)) != len(old_ids):
        raise ValueError(f"duplicate compact node IDs: {path}")
    old_to_new = {int(old_id): index for index, old_id in enumerate(old_ids)}
    nodes = [
        {
            "t": int(values[0]),
            "z": max(0, int(round(float(values[1])))),
            "y": max(0, int(round(float(values[2])))),
            "x": max(0, int(round(float(values[3])))),
        }
        for values in tzyx
    ]
    edges: list[tuple[int, int]] = []
    for source, target in old_edges:
        if int(source) not in old_to_new or int(target) not in old_to_new:
            raise ValueError(f"dangling compact edge: {path}")
        edges.append((old_to_new[int(source)], old_to_new[int(target)]))
    return nodes, edges


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "cache-worker":
        _run_cache_worker(Path(sys.argv[2]))
        return
    raise SystemExit("usage: graph_inference.py cache-worker SPEC.json")


if __name__ == "__main__":
    _main()
