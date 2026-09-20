"""Read saved exp027 metrics and float32 checkpoint tensors without importing torch."""

from __future__ import annotations

import hashlib
import io
import json
import pickle
import zipfile
from collections import OrderedDict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments/exp027_multi_frame_tracker"
OUTPUT = EXP / "artifacts/kaggle_train_v4"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rebuild_tensor(storage, offset, shape, stride, requires_grad, hooks):
    expected = []
    step = 1
    for size in reversed(shape):
        expected.append(step)
        step *= size
    if tuple(reversed(expected)) != tuple(stride):
        raise ValueError("Only contiguous saved tensors are supported")
    if offset < 0 or offset + step > len(storage):
        raise ValueError("Invalid storage range")
    return storage[offset : offset + step].reshape(shape)


class FloatCheckpointReader(pickle.Unpickler):
    def __init__(self, archive, prefix):
        self.archive = archive
        self.prefix = prefix
        super().__init__(io.BytesIO(archive.read(prefix + "data.pkl")))

    def find_class(self, module, name):
        allowed = {
            ("collections", "OrderedDict"): OrderedDict,
            ("torch", "FloatStorage"): np.dtype("<f4"),
            ("torch._utils", "_rebuild_tensor_v2"): rebuild_tensor,
        }
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f"Unsupported global {module}.{name}")
        return allowed[module, name]

    def persistent_load(self, pid):
        kind, dtype, key, device, count = pid
        if kind != "storage" or dtype != np.dtype("<f4"):
            raise pickle.UnpicklingError("Only float32 storage is supported")
        data = self.archive.read(self.prefix + "data/" + key)
        if len(data) != count * dtype.itemsize:
            raise ValueError("Storage length mismatch")
        return np.frombuffer(data, dtype=dtype)


def load_checkpoint(path):
    with zipfile.ZipFile(path) as archive:
        prefix = next(
            name[: -len("data.pkl")] for name in archive.namelist() if name.endswith("/data.pkl")
        )
        if archive.read(prefix + "byteorder") != b"little":
            raise ValueError("Only little-endian tensors are supported")
        return FloatCheckpointReader(archive, prefix).load()


def main():
    manifest = json.loads((OUTPUT / "model_manifest.json").read_text())
    metrics = json.loads((EXP / "metrics.json").read_text())
    selection = json.loads((OUTPUT / "pilot_outer_window_selection.json").read_text())
    records = []
    for fold in metrics["train_stage"]["folds"]:
        path = OUTPUT / fold["model_file"]
        assert sha256(path) == fold["model_file_sha256"]
        checkpoint = load_checkpoint(path)
        assert checkpoint["fold"] == fold["fold"]
        assert checkpoint["variant"] == fold["variant"]
        state = checkpoint["state_dict"]
        embeddings = state["time_embedding.weight"].astype(np.float64)
        assert embeddings.shape == (3, 128)
        internal = fold["epochs"][0]["internal_validation"]
        outer = fold["trained_outer_evaluation"]
        record = {
            "fold": fold["fold"],
            "variant": fold["variant"],
            "train_embryo": fold["train_embryo"],
            "evaluation_embryo": fold["evaluation_embryo"],
            "checkpoint_sha256": sha256(path),
            "time_embedding_l2_by_frame": np.linalg.norm(embeddings, axis=1).tolist(),
            "time_embedding_pairwise_l2": np.linalg.norm(
                embeddings[:, None] - embeddings[None, :], axis=-1
            ).tolist(),
            "all_tensors_finite": all(np.isfinite(v).all() for v in state.values()),
            "parameter_count": sum(v.size for v in state.values()),
            "optimizer_steps": int(fold["epochs"][0]["train"]["batch_count"]),
            "internal": internal,
            "outer": outer,
            "train_loss": fold["epochs"][0]["train"]["legacy_mask_loss"],
        }
        records.append(record)
    comparisons = []
    for fold_id in (0, 1):
        two = next(r for r in records if r["fold"] == fold_id and r["variant"] == "two_frame_local")
        three = next(
            r for r in records if r["fold"] == fold_id and r["variant"] == "three_frame_local"
        )
        for split in ("internal", "outer"):
            a, b = two[split], three[split]
            comparisons.append(
                {
                    "fold": fold_id,
                    "split": split,
                    "train_embryo": two["train_embryo"],
                    "evaluation_embryo": two["evaluation_embryo"]
                    if split == "outer"
                    else two["train_embryo"],
                    "recall_delta_percentage_points": 100
                    * (b["positive_edge_recall"] - a["positive_edge_recall"]),
                    "true_positive_delta": round(
                        b["positive_edge_recall"] * b["positive_edge_count"]
                    )
                    - round(a["positive_edge_recall"] * a["positive_edge_count"]),
                    "teacher_negative_positive_prediction_delta": b[
                        "false_positive_active_pair_count"
                    ]
                    - a["false_positive_active_pair_count"],
                    "loss_relative_change_percent": 100
                    * (b["legacy_mask_loss"] / a["legacy_mask_loss"] - 1),
                    "positive_fraction_active_pairs": a["positive_edge_count"]
                    / a["active_pair_count"],
                }
            )
    boundary = {
        f["evaluation_embryo"]: sum(
            Path(p).name.startswith("000000_") for p in f["selected_windows"]
        )
        for f in selection["folds"]
    }
    source_match = {
        name: sha256(EXP / name) == sha256(OUTPUT / name)
        for name in (
            "local_tracker_model.py",
            "frozen_tracker.py",
            "exp027_multi_frame_tracker_train.py",
        )
    }
    result = {
        "executed_sources_match_local": source_match,
        "source_metrics_sha256": sha256(EXP / "metrics.json"),
        "source_model_manifest_sha256": sha256(OUTPUT / "model_manifest.json"),
        "manifest_stage": manifest["stage"],
        "method": (
            "Whitelist-only pickle reader for verified float32 saved tensors; "
            "no model inference."
        ),
        "checkpoint_audit": records,
        "comparisons": comparisons,
        "outer_windows_without_previous_frame": boundary,
    }
    output = Path(__file__).with_name("audit.json")
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "comparisons": comparisons,
                "boundary": boundary,
                "time_embeddings": [
                    {
                        k: r[k]
                        for k in (
                            "fold",
                            "variant",
                            "time_embedding_l2_by_frame",
                            "time_embedding_pairwise_l2",
                            "all_tensors_finite",
                            "optimizer_steps",
                        )
                    }
                    for r in records
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
