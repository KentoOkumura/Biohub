from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import graph_inference as gi

EXPECTED_SECONDARY_CHECKPOINT_SHA256 = (
    "9bac2fa0dadc4a6fc1899e0caf187f4b553e0a7cd90ba1261a68b35ffe9e305f"
)
MODEL_PARAMS = {
    "feature_dim": 64,
    "hidden_dim": 128,
    "n_heads": 4,
    "n_blocks": 4,
    "dropout": 0.3,
    "pair_chunk_size": 32,
}
REPLAY_CONFIG = {
    "bidirectional_weight": 0.15,
    "secondary_edge_weight": 0.15,
    "secondary_low_margin_max": 0.35,
    "secondary_mix_temperature": 1.0,
    "edge_threshold": 0.48,
    "use_ilp": True,
    "ilp_edge_weight": -1.0,
    "ilp_appearance_weight": 0.0,
    "ilp_disappearance_weight": 2.0,
    "ilp_division_weight": 1.2,
}


def _require_file_sha(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    actual = gi.sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"{label} SHA changed: expected {expected}, got {actual}")


def _validate_fold_model(path: Path, fold: int) -> None:
    expected = gi.EXPECTED_FOLD_MODELS[fold]["file_sha256"]
    _require_file_sha(path, expected, f"fold {fold} tracker")


def _sample_names(cache_root: Path, requested: list[str], limit: int | None) -> list[str]:
    available = sorted(path.name for path in cache_root.iterdir() if path.is_dir())
    names = requested or available
    unknown = sorted(set(names) - set(available))
    if unknown:
        raise FileNotFoundError({"requested_cache_samples_missing": unknown})
    names = sorted(dict.fromkeys(names))
    if limit is not None:
        names = names[:limit]
    if not names:
        raise RuntimeError("no cache samples selected")
    return names


def _validate_sample_inputs(baseline_root: Path, names: list[str]) -> None:
    cache_root = baseline_root / "window_cache"
    candidate_root = baseline_root / "oracle_candidate_graphs"
    failures: list[dict[str, Any]] = []
    for name in names:
        windows = sorted((cache_root / name).glob("*.npz"))
        candidate = candidate_root / f"{name}.geff"
        if len(windows) != 99 or not candidate.exists():
            failures.append(
                {
                    "sample": name,
                    "window_count": len(windows),
                    "candidate_exists": candidate.exists(),
                }
            )
    if failures:
        raise RuntimeError({"invalid_sample_inputs": failures})


def _validate_full_cache(baseline_root: Path, names: list[str]) -> dict[str, Any]:
    from frozen_tracker import (
        discover_cache_paths,
        recompute_cache_identity_sha256,
        validate_cache_summary,
    )

    if len(names) != gi.EXPECTED_CACHE_SAMPLE_COUNT:
        raise RuntimeError("full Colab replay requires all 199 samples")
    cache_cfg = {
        "schema_version": 1,
        "expected_dataset_count": gi.EXPECTED_CACHE_SAMPLE_COUNT,
        "expected_window_count": gi.EXPECTED_CACHE_WINDOW_COUNT,
        "identity_sha256": gi.EXPECTED_EXP015_CACHE_IDENTITY_SHA256,
        "summary_sha256": gi.EXPECTED_EXP015_CACHE_SUMMARY_SHA256,
    }
    summary = validate_cache_summary(baseline_root / "window_cache_summary.json", cache_cfg)
    paths = discover_cache_paths(baseline_root / "window_cache", cache_cfg)
    identity = recompute_cache_identity_sha256(paths)
    if identity != gi.EXPECTED_EXP015_CACHE_IDENTITY_SHA256:
        raise RuntimeError("exp015 cache identity changed")
    baseline_names = sorted(
        path.name.removesuffix(".geff")
        for path in (baseline_root / "oracle_candidate_graphs").glob("*.geff")
    )
    if baseline_names != names:
        raise RuntimeError("exp015 candidate graph coverage changed")
    return summary


def _valid_completed_batch(
    receipt_path: Path, expected_names: list[str], output_root: Path
) -> bool:
    if not receipt_path.is_file():
        return False
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    observed = sorted(str(row["sample"]) for row in receipt.get("samples", []))
    if observed != sorted(expected_names):
        return False
    if not receipt.get("public_control_candidate_graphs_exact"):
        return False
    archive_name = receipt.get("archive_name")
    archive_sha256 = receipt.get("archive_sha256")
    if not isinstance(archive_name, str) or not isinstance(archive_sha256, str):
        return False
    archive_path = output_root / "batch_archives" / archive_name
    return archive_path.is_file() and gi.sha256_file(archive_path) == archive_sha256


def _archive_batch(batch_root: Path, archive_path: Path) -> None:
    paths = sorted(path for path in batch_root.rglob("*") if path.is_file())
    if not paths:
        raise RuntimeError(f"batch produced no graph files: {batch_root}")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.relative_to(batch_root).as_posix())


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    started = time.monotonic()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"Colab replay expects exactly one visible CUDA GPU, found {torch.cuda.device_count()}"
        )
    baseline_root = args.baseline_root.resolve()
    repo_dir = args.repo_dir.resolve()
    output_root = args.output_root.resolve()
    cache_root = baseline_root / "window_cache"
    if not cache_root.is_dir():
        raise FileNotFoundError(f"window cache is missing: {cache_root}")
    public_weights = args.public_weights.resolve()
    secondary_weights = args.secondary_weights.resolve()
    fold_weights = {
        0: args.fold_0_weights.resolve(),
        1: args.fold_1_weights.resolve(),
    }
    _require_file_sha(
        public_weights,
        gi.EXPECTED_PUBLIC_PRIMARY_CHECKPOINT_SHA256,
        "public primary tracker",
    )
    _require_file_sha(
        secondary_weights,
        EXPECTED_SECONDARY_CHECKPOINT_SHA256,
        "secondary tracker",
    )
    for fold, path in fold_weights.items():
        _validate_fold_model(path, fold)
    names = _sample_names(cache_root, args.sample, args.sample_limit)
    _validate_sample_inputs(baseline_root, names)
    cache_summary = _validate_full_cache(baseline_root, names) if args.require_full else None

    for path in (
        repo_dir / "src",
        repo_dir / "scripts",
        args.code_root.resolve(),
    ):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    output_root.mkdir(parents=True, exist_ok=True)
    receipt_root = output_root / "batch_receipts"
    archive_root = output_root / "batch_archives"
    scratch_root = args.scratch_root.resolve()
    for path in (receipt_root, archive_root, scratch_root):
        path.mkdir(parents=True, exist_ok=True)

    batches = [
        names[index : index + args.batch_size] for index in range(0, len(names), args.batch_size)
    ]
    for batch_index, batch_names in enumerate(batches):
        if time.monotonic() - started >= args.runtime_gate_hours * 60 * 60:
            raise TimeoutError("Colab cache replay exceeded the configured runtime gate")
        receipt_path = receipt_root / f"batch_{batch_index:03d}.json"
        if args.resume and _valid_completed_batch(receipt_path, batch_names, output_root):
            print(f"COLAB_REPLAY resume batch={batch_index} samples={len(batch_names)}", flush=True)
            continue
        batch_root = scratch_root / f"batch_{batch_index:03d}"
        if batch_root.exists():
            shutil.rmtree(batch_root)
        candidate_root = batch_root / "oracle_candidate_graphs"
        prediction_root = batch_root / "ilp_graphs"
        candidate_root.mkdir(parents=True)
        prediction_root.mkdir(parents=True)
        worker_receipt_path = batch_root / "worker_receipt.json"
        spec = {
            "worker": 0,
            "repo_dir": str(repo_dir),
            "cache_root": str(cache_root),
            "baseline_candidate_root": str(baseline_root / "oracle_candidate_graphs"),
            "candidate_output_root": str(candidate_root),
            "prediction_output_root": str(prediction_root),
            "public_weights": str(public_weights),
            "secondary_weights": str(secondary_weights),
            "fold_weights": {str(fold): str(path) for fold, path in fold_weights.items()},
            "fold_model_by_embryo": {"6bba": 0, "44b6": 1},
            "samples": batch_names,
            "expected_windows_per_sample": 99,
            "downsample_zyx": [1, 4, 4],
            "model_params": MODEL_PARAMS,
            "replay": REPLAY_CONFIG,
            "receipt_path": str(worker_receipt_path),
        }
        spec_path = batch_root / "worker_spec.json"
        spec_path.write_text(
            json.dumps(spec, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        gi._run_cache_worker(spec_path)
        worker_receipt = json.loads(worker_receipt_path.read_text(encoding="utf-8"))
        if sorted(str(row["sample"]) for row in worker_receipt.get("samples", [])) != sorted(
            batch_names
        ):
            raise RuntimeError(f"batch {batch_index} worker receipt coverage changed")
        if not worker_receipt.get("public_control_candidate_graphs_exact"):
            raise RuntimeError(f"batch {batch_index} public control exactness failed")
        for name in batch_names:
            if (
                not (candidate_root / f"{name}.geff").exists()
                or not (prediction_root / f"{name}.geff").exists()
            ):
                raise RuntimeError(f"batch {batch_index} graph output missing for {name}")
        local_archive = scratch_root / f"batch_{batch_index:03d}.zip"
        if local_archive.exists():
            local_archive.unlink()
        _archive_batch(batch_root, local_archive)
        archive_path = archive_root / local_archive.name
        archive_partial = archive_path.with_suffix(".zip.part")
        shutil.copy2(local_archive, archive_partial)
        archive_partial.replace(archive_path)
        persistent_receipt = {
            **worker_receipt,
            "archive_name": archive_path.name,
            "archive_bytes": archive_path.stat().st_size,
            "archive_sha256": gi.sha256_file(archive_path),
        }
        receipt_partial = receipt_path.with_suffix(".json.part")
        receipt_partial.write_text(
            json.dumps(persistent_receipt, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        receipt_partial.replace(receipt_path)
        if not _valid_completed_batch(receipt_path, batch_names, output_root):
            raise RuntimeError(f"batch {batch_index} did not produce a valid persistent receipt")
        shutil.rmtree(batch_root)
        local_archive.unlink()
        if time.monotonic() - started >= args.runtime_gate_hours * 60 * 60:
            raise TimeoutError("Colab cache replay exceeded the configured runtime gate")

    receipts = [
        json.loads((receipt_root / f"batch_{index:03d}.json").read_text(encoding="utf-8"))
        for index in range(len(batches))
    ]
    rows = sorted(
        [row for receipt in receipts for row in receipt["samples"]],
        key=lambda row: row["sample"],
    )
    if [row["sample"] for row in rows] != names:
        raise RuntimeError("combined Colab replay receipt coverage changed")
    if not all(row["exact"] for row in rows):
        raise RuntimeError("public tracker did not exactly reproduce an exp015 candidate graph")
    summary = {
        "experiment": "exp016_frozen_image_encoder",
        "stage": "colab_cached_tracker_and_ilp_replay",
        "status": "complete",
        "authoritative_full_replay": args.require_full,
        "sample_count": len(rows),
        "window_count": sum(int(row["window_count"]) for row in rows),
        "samples": rows,
        "public_control_candidate_graphs_exact": True,
        "main_image_encoder_forward_count": 0,
        "tracker_forward_count_per_window": 5,
        "worker_count": 1,
        "resume_batch_size": args.batch_size,
        "persistent_batch_archive_count": len(batches),
        "runtime_gate_hours": args.runtime_gate_hours,
        "elapsed_seconds": time.monotonic() - started,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "gpu_name": torch.cuda.get_device_name(0),
        "input_cache_summary": cache_summary,
        "checkpoint_sha256": {
            "public_primary": gi.sha256_file(public_weights),
            "secondary": gi.sha256_file(secondary_weights),
            "fold_0": gi.sha256_file(fold_weights[0]),
            "fold_1": gi.sha256_file(fold_weights[1]),
        },
    }
    summary["summary_sha256"] = gi.json_sha256(summary)
    (output_root / "colab_cache_replay_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "samples"}, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--public-weights", type=Path, required=True)
    parser.add_argument("--secondary-weights", type=Path, required=True)
    parser.add_argument("--fold-0-weights", type=Path, required=True)
    parser.add_argument("--fold-1-weights", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--sample", action="append", default=[])
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--runtime-gate-hours", type=float, default=12.0)
    parser.add_argument("--require-full", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.sample_limit is not None and args.sample_limit <= 0:
        parser.error("--sample-limit must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.runtime_gate_hours <= 0:
        parser.error("--runtime-gate-hours must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
