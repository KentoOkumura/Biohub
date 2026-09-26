from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

HistoryFrame = dict[str, np.ndarray]
HistoryByFrame = dict[tuple[str, int], HistoryFrame]


def _validated_candidate_inputs(
    candidate_ids: np.ndarray,
    coords_physical: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(candidate_ids, dtype=np.int64)
    coords = np.asarray(coords_physical, dtype=np.float32)
    if ids.ndim != 1 or coords.shape != (len(ids), 3):
        raise ValueError("candidate ids and physical coordinates do not align")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("candidate ids must be unique within a frame")
    if not np.isfinite(coords).all():
        raise ValueError("candidate coordinates contain non-finite values")
    return ids, coords


def empty_history_frame(
    candidate_ids: np.ndarray,
    coords_physical: np.ndarray,
) -> HistoryFrame:
    ids, coords = _validated_candidate_inputs(candidate_ids, coords_physical)
    return {
        "candidate_ids": ids.copy(),
        "coords_physical": coords.copy(),
        "velocity": np.zeros((len(ids), 3), dtype=np.float32),
        "present": np.zeros(len(ids), dtype=bool),
        "length": np.zeros(len(ids), dtype=np.int64),
        "confidence": np.zeros(len(ids), dtype=np.float32),
    }


def align_history_frame(
    history: HistoryFrame,
    candidate_ids: np.ndarray,
    coords_physical: np.ndarray,
    *,
    coordinate_atol: float = 1e-5,
) -> HistoryFrame:
    ids, coords = _validated_candidate_inputs(candidate_ids, coords_physical)
    history_ids, history_coords = _validated_candidate_inputs(
        history["candidate_ids"],
        history["coords_physical"],
    )
    if set(map(int, ids)) != set(map(int, history_ids)):
        raise ValueError("history and cache candidate ids differ")
    index = {int(candidate_id): idx for idx, candidate_id in enumerate(history_ids)}
    order = np.asarray([index[int(candidate_id)] for candidate_id in ids], dtype=np.int64)
    if not np.allclose(history_coords[order], coords, rtol=0.0, atol=coordinate_atol):
        raise ValueError("history and cache candidate coordinates differ")
    aligned = {
        "candidate_ids": ids.copy(),
        "coords_physical": coords.copy(),
        "velocity": np.asarray(history["velocity"], dtype=np.float32)[order].copy(),
        "present": np.asarray(history["present"], dtype=bool)[order].copy(),
        "length": np.asarray(history["length"], dtype=np.int64)[order].copy(),
        "confidence": np.asarray(history["confidence"], dtype=np.float32)[order].copy(),
    }
    expected_shapes = {
        "velocity": (len(ids), 3),
        "present": (len(ids),),
        "length": (len(ids),),
        "confidence": (len(ids),),
    }
    observed_shapes = {name: aligned[name].shape for name in expected_shapes}
    if observed_shapes != expected_shapes:
        raise ValueError(
            {"history_shape_mismatch": {"expected": expected_shapes, "actual": observed_shapes}}
        )
    for name in ("velocity", "confidence"):
        if not np.isfinite(aligned[name]).all():
            raise ValueError(f"history contains non-finite {name}")
    return aligned


def propagate_history_frame(
    source_history: HistoryFrame,
    *,
    candidate_ids_src: np.ndarray,
    candidate_ids_tgt: np.ndarray,
    coords_src_physical: np.ndarray,
    coords_tgt_physical: np.ndarray,
    source_axis_probabilities: np.ndarray,
    edge_probability_threshold: float,
) -> tuple[HistoryFrame, dict[str, int]]:
    if not 0.0 <= edge_probability_threshold <= 1.0:
        raise ValueError("edge probability threshold must be in [0, 1]")
    source_ids, source_coords = _validated_candidate_inputs(
        candidate_ids_src,
        coords_src_physical,
    )
    target_ids, target_coords = _validated_candidate_inputs(
        candidate_ids_tgt,
        coords_tgt_physical,
    )
    aligned_source = align_history_frame(source_history, source_ids, source_coords)
    probabilities = np.asarray(source_axis_probabilities, dtype=np.float32)
    if probabilities.shape != (len(source_ids), len(target_ids)):
        raise ValueError("probability matrix does not align with candidates")
    if not np.isfinite(probabilities).all():
        raise ValueError("probability matrix contains non-finite values")
    if len(source_ids) == 0:
        return empty_history_frame(target_ids, target_coords), {
            "selected_edge_count": 0,
            "predicted_division_source_count": 0,
            "division_reset_target_count": 0,
            "history_present_target_count": 0,
        }

    best_sources = probabilities.argmax(axis=0)
    best_probabilities = probabilities[best_sources, np.arange(len(target_ids))]
    selected = best_probabilities >= float(edge_probability_threshold)
    selected_counts = np.bincount(best_sources[selected], minlength=len(source_ids))
    division_sources = selected_counts > 1

    target_history = empty_history_frame(target_ids, target_coords)
    reset_targets = 0
    for target_index in np.flatnonzero(selected):
        source_index = int(best_sources[target_index])
        if division_sources[source_index]:
            reset_targets += 1
            continue
        target_history["velocity"][target_index] = (
            target_coords[target_index] - source_coords[source_index]
        )
        target_history["present"][target_index] = True
        previous_length = (
            int(aligned_source["length"][source_index])
            if bool(aligned_source["present"][source_index])
            else 0
        )
        target_history["length"][target_index] = previous_length + 1
        target_history["confidence"][target_index] = best_probabilities[target_index]

    audit = {
        "selected_edge_count": int(selected.sum()),
        "predicted_division_source_count": int(division_sources.sum()),
        "division_reset_target_count": int(reset_targets),
        "history_present_target_count": int(target_history["present"].sum()),
    }
    return target_history, audit


def history_content_sha256(history_by_frame: HistoryByFrame) -> str:
    digest = hashlib.sha256()
    for sample, frame in sorted(history_by_frame):
        record = history_by_frame[(sample, frame)]
        digest.update(
            json.dumps(
                {"sample": sample, "frame": int(frame)},
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        )
        digest.update(b"\0")
        for name in (
            "candidate_ids",
            "coords_physical",
            "velocity",
            "present",
            "length",
            "confidence",
        ):
            array = np.ascontiguousarray(record[name])
            digest.update(
                json.dumps(
                    {"name": name, "dtype": array.dtype.str, "shape": list(array.shape)},
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            )
            digest.update(b"\0")
            digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def aggregate_history_audits(records: list[dict[str, Any]]) -> dict[str, int]:
    keys = (
        "window_count",
        "selected_edge_count",
        "predicted_division_source_count",
        "division_reset_target_count",
        "history_present_source_count",
        "history_source_count",
    )
    return {key: int(sum(int(record.get(key, 0)) for record in records)) for key in keys}
