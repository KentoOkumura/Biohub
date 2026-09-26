"""Run exp045 fixed-ID readouts from saved version 1 output and local GT GEFF.

This local recovery uses only the already generated candidate caches and stage
files. The public official evaluator runs separately on Kaggle.
"""

from __future__ import annotations

import argparse
import hashlib as _audit_hashlib
import json
import math as _audit_math
import time
from pathlib import Path

import numpy as np
import zarr

from audit_core import (
    NATIVE_SPACING_UM,
    known_edges,
    match_known_centers,
    readout_stages,
    scored_known_edge_ranks,
    trace_original_edges,
    validate_initial_graph_ids,
)

EXPECTED_SELECTION_SHA = "ba7eb44348be9ee18a0a15180276dbb92399a5c5995e26ae3b3d667b8cc30daa"
EXPECTED_RUNTIME_GATE_SHA = "b5b53aacfae8c67a81bb75be00aaeda6c1809ca2faceaf80f1a3bc75b7ebd85c"
EXPECTED_HEAD_SHA = "32d6c62f738c3dfe4862e3df5272850312e81b622e57d9ba45209ebb382824dc"
AUDIT_ARMS = ("zero", "refined")


def _audit_sha256(path: Path) -> str:
    digest = _audit_hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _audit_jsonable(value):
    if isinstance(value, dict):
        return {str(key): _audit_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_audit_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return _audit_jsonable(value.item())
    if isinstance(value, float) and not _audit_math.isfinite(value):
        return None
    return value


class _AttributeRows:
    def __init__(self, names, arrays):
        self.names = names
        self.arrays = arrays

    def iter_rows(self, *, named: bool):
        if not named:
            raise ValueError("named rows are required")
        for values in zip(*self.arrays, strict=True):
            yield dict(zip(self.names, values, strict=True))


class _GeffGraph:
    def __init__(self, path: Path):
        group = zarr.open_group(path, mode="r")
        self.nodes = _AttributeRows(
            ("node_id", "t", "z", "y", "x"),
            tuple(
                np.asarray(group[key])
                for key in (
                    "nodes/ids",
                    "nodes/props/t/values",
                    "nodes/props/z/values",
                    "nodes/props/y/values",
                    "nodes/props/x/values",
                )
            ),
        )
        edges = np.asarray(group["edges/ids"])
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise RuntimeError(f"unexpected GEFF edge shape: {path}")
        self.edges = _AttributeRows(("source_id", "target_id"), (edges[:, 0], edges[:, 1]))

    def node_attrs(self):
        return self.nodes

    def edge_attrs(self):
        return self.edges


def graph_from_geff(path: Path):
    return _GeffGraph(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-root", required=True, type=Path)
    parser.add_argument("--gt-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    global AUDIT_ROOT, AUDIT_STAGE_ROOT, AUDIT_OUT, TRAIN_DIR, test_stems
    AUDIT_ROOT = args.prediction_root.resolve()
    AUDIT_STAGE_ROOT = AUDIT_ROOT / "stages"
    TRAIN_DIR = args.gt_root.resolve()
    AUDIT_OUT = args.output_root.resolve()
    AUDIT_OUT.mkdir(parents=True, exist_ok=True)
    selection_path = AUDIT_ROOT.parent / "exp045_selection.json"
    gate_path = AUDIT_ROOT / "runtime_gate.json"
    if _audit_sha256(selection_path) != EXPECTED_SELECTION_SHA:
        raise RuntimeError("selection manifest SHA changed")
    if _audit_sha256(gate_path) != EXPECTED_RUNTIME_GATE_SHA:
        raise RuntimeError("runtime gate SHA changed")
    selection_receipt = json.loads(selection_path.read_text())
    selection = selection_receipt["selected"]
    runtime_gate = json.loads(gate_path.read_text())
    test_stems = sorted(stem for group in selection.values() for stem in group)
    if (
        set(selection) != {"44b6", "6bba"}
        or any(len(group) != 10 for group in selection.values())
        or len(set(test_stems)) != 20
        or set(test_stems) & set(selection_receipt["trained"])
        or selection_receipt["head_sha256"] != EXPECTED_HEAD_SHA
        or runtime_gate["pilot"] != sorted(group[0] for group in selection.values())
    ):
        raise RuntimeError("saved selection, head, and runtime gate disagree")
    for arm in AUDIT_ARMS:
        if {p.stem for p in (AUDIT_ROOT / arm / "candidate_cache").glob("*.npz")} != set(
            test_stems
        ):
            raise RuntimeError(f"{arm}: candidate caches are incomplete")
        if {p.stem for p in (AUDIT_ROOT / arm / "final_graphs").glob("*.geff")} != set(test_stems):
            raise RuntimeError(f"{arm}: final graphs are incomplete")
        for stem in test_stems:
            if len(list((AUDIT_STAGE_ROOT / arm / stem).glob("*.npz"))) != 9:
                raise RuntimeError(f"{arm}/{stem}: stage artifacts are incomplete")
    if any(not (TRAIN_DIR / f"{stem}.geff").is_dir() for stem in test_stems):
        raise RuntimeError("GT GEFF is missing")
    start = time.time()
    print("Fixed-ID local recovery: 20 videos, 2 arms, GT only after prediction", flush=True)

    def _audit_check_identity(stems: list[str]) -> dict[str, object]:
        by_video = {}
        for stem in stems:
            zero_raw = AUDIT_ROOT / "zero" / "raw_detector_coordinates" / stem
            refined_raw = AUDIT_ROOT / "refined" / "raw_detector_coordinates" / stem
            names_zero = sorted(path.name for path in zero_raw.glob("*.npz"))
            names_refined = sorted(path.name for path in refined_raw.glob("*.npz"))
            if not names_zero or names_zero != names_refined:
                raise RuntimeError(f"{stem}: detector frame coverage differs or is empty")
            for name in names_zero:
                with np.load(zero_raw / name, allow_pickle=False) as left:
                    with np.load(refined_raw / name, allow_pickle=False) as right:
                        if not np.array_equal(left["coords"], right["coords"]):
                            raise RuntimeError(f"{stem}/{name}: raw detections differ")
            coords_by_arm = {}
            initial_nodes_by_arm = {}
            for arm in AUDIT_ARMS:
                with np.load(
                    AUDIT_ROOT / arm / "candidate_cache" / f"{stem}.npz", allow_pickle=False
                ) as payload:
                    coords_by_arm[arm] = np.asarray(payload["coords"], dtype=np.float64)
                graph = graph_from_geff(AUDIT_ROOT / arm / "raw_ilp_graphs" / f"{stem}.geff")
                graph_rows = np.asarray(
                    [
                        [
                            int(row["node_id"]),
                            int(row["t"]),
                            float(row["z"]),
                            float(row["y"]),
                            float(row["x"]),
                        ]
                        for row in graph.node_attrs().iter_rows(named=True)
                    ],
                    dtype=np.float64,
                ).reshape(-1, 5)
                initial_ids = validate_initial_graph_ids(coords_by_arm[arm], graph_rows)
                initial_nodes_by_arm[arm] = len(initial_ids)
            zero = coords_by_arm["zero"]
            refined = coords_by_arm["refined"]
            if zero.shape != refined.shape or not np.array_equal(zero[:, 0], refined[:, 0]):
                raise RuntimeError(f"{stem}: candidate count or time order differs")
            shifts = np.linalg.norm((refined[:, 1:4] - zero[:, 1:4]) * NATIVE_SPACING_UM, axis=1)
            if len(shifts) and (not np.isfinite(shifts).all() or float(shifts.max()) > 2.00001):
                raise RuntimeError(f"{stem}: coordinate shift violates the head bound")
            by_video[stem] = {
                "candidates": len(zero),
                "initial_graph_nodes": initial_nodes_by_arm,
                "raw_frames": len(names_zero),
                "max_shift_um": 0.0 if len(shifts) == 0 else float(shifts.max()),
                "raw_detector_sha256": _audit_hashlib.sha256(
                    b"".join((zero_raw / name).read_bytes() for name in names_zero)
                ).hexdigest(),
            }
        return by_video

    _identity = _audit_check_identity(test_stems)
    (AUDIT_OUT / "identity.json").write_text(
        json.dumps(_audit_jsonable(_identity), indent=2, sort_keys=True) + "\n"
    )
    print("All 20 paired candidate IDs: PASS", flush=True)

    def _audit_load_stage_edges(
        arm: str, stem: str, stage: str, original_ids: set[int] | None = None
    ) -> set[tuple[int, int]]:
        path = AUDIT_STAGE_ROOT / arm / stem / f"{stage}.npz"
        with np.load(path, allow_pickle=False) as payload:
            edges = np.asarray(payload["edges"], dtype=np.float64).reshape(-1, 3)
        pairs = {(int(row[0]), int(row[1])) for row in edges}
        return pairs if original_ids is None else trace_original_edges(pairs, original_ids)

    def _audit_gt_graph(stem: str) -> tuple[np.ndarray, set[tuple[int, int]]]:
        graph = graph_from_geff(TRAIN_DIR / f"{stem}.geff")
        node_rows = np.asarray(
            [
                [
                    int(row["node_id"]),
                    int(row["t"]),
                    float(row["z"]),
                    float(row["y"]),
                    float(row["x"]),
                ]
                for row in graph.node_attrs().iter_rows(named=True)
            ],
            dtype=np.float64,
        ).reshape(-1, 5)
        edge_set = {
            (int(row["source_id"]), int(row["target_id"]))
            for row in graph.edge_attrs().iter_rows(named=True)
        }
        return node_rows, edge_set

    _DIAGNOSTIC = {}
    _STAGE_NAMES = (
        "edge_filter",
        "relink",
        "gap1",
        "gap2",
        "low_detection",
        "division",
        "short_track",
        "final",
    )
    for _stem in test_stems:
        _known_nodes, _known_gt_edges = _audit_gt_graph(_stem)
        with np.load(
            AUDIT_ROOT / "zero" / "candidate_cache" / f"{_stem}.npz", allow_pickle=False
        ) as _zero_payload:
            _zero_coords = np.asarray(_zero_payload["coords"], dtype=np.float64)
        _candidate_rows = np.column_stack((np.arange(len(_zero_coords)), _zero_coords))
        _fixed_match = match_known_centers(_candidate_rows, _known_nodes, radius_um=7.0)
        _, _mapped_known_edges = known_edges(_known_gt_edges, _fixed_match)
        _video_report = {
            "embryo": _stem.split("_", 1)[0],
            "candidate_count": len(_zero_coords),
            "matched_gt_nodes": len(_fixed_match.gt_to_candidate),
            "ambiguous_gt": _fixed_match.ambiguous_gt,
            "ambiguous_candidate": _fixed_match.ambiguous_candidate,
            "arms": {},
        }
        for _arm in AUDIT_ARMS:
            _cache_path = AUDIT_ROOT / _arm / "candidate_cache" / f"{_stem}.npz"
            with np.load(_cache_path, allow_pickle=False) as _cache:
                _edge_src = np.asarray(_cache["edge_src"], dtype=np.int32)
                _edge_tgt = np.asarray(_cache["edge_tgt"], dtype=np.int32)
                _edge_prob = np.asarray(_cache["edge_prob"], dtype=np.float32)
                _admitted = np.asarray(_cache["admitted"], dtype=np.float64)
                if _admitted.size == 0:
                    _admitted = _admitted.reshape(0, 4)
                elif _admitted.ndim != 2 or _admitted.shape[1] != 4:
                    raise RuntimeError(f"{_stem}/{_arm}: expected 4-column admitted edge cache")
                _coords = np.asarray(_cache["coords"], dtype=np.float64)
            if not (len(_edge_src) == len(_edge_tgt) == len(_edge_prob)):
                raise RuntimeError(f"{_stem}/{_arm}: score array lengths differ")
            if len(_admitted) and (
                _admitted[:, :2].min() < 0 or _admitted[:, :2].max() >= len(_coords)
            ):
                raise RuntimeError(f"{_stem}/{_arm}: candidate edge IDs outside node range")
            if not np.isfinite(_edge_prob).all():
                raise RuntimeError(f"{_stem}/{_arm}: nonfinite captured pair score")
            # The public tracker captures millions of pairs per video. The
            # pure readout already ignores every source without a known GT
            # edge; apply that same filter in NumPy before Python iteration.
            _known_sources = np.fromiter(
                {source for source, _ in _mapped_known_edges}, dtype=np.int32
            )
            _known_source_mask = np.isin(_edge_src, _known_sources)
            _ranks = scored_known_edge_ranks(
                zip(
                    _edge_src[_known_source_mask],
                    _edge_tgt[_known_source_mask],
                    _edge_prob[_known_source_mask],
                    strict=True,
                ),
                _mapped_known_edges,
            )
            _ilp_stage_path = AUDIT_STAGE_ROOT / _arm / _stem / "ilp.npz"
            with np.load(_ilp_stage_path, allow_pickle=False) as _ilp_payload:
                _ilp_nodes = np.asarray(_ilp_payload["nodes"], dtype=np.float64).reshape(-1, 5)
            _initial_ids = validate_initial_graph_ids(_coords, _ilp_nodes)
            _stages = {
                "scored": {pair for pair, (score, _) in _ranks.items() if score is not None},
                "candidate": {(int(row[0]), int(row[1])) for row in _admitted},
                "ilp": _audit_load_stage_edges(_arm, _stem, "ilp", _initial_ids),
            }
            _stages.update(
                {
                    stage: _audit_load_stage_edges(_arm, _stem, stage, _initial_ids)
                    for stage in _STAGE_NAMES
                }
            )
            _stage_counts = readout_stages(_known_gt_edges, _fixed_match, _stages)
            _stage_nodes = {}
            for _stage in ("ilp", *_STAGE_NAMES):
                _stage_path = AUDIT_STAGE_ROOT / _arm / _stem / f"{_stage}.npz"
                with np.load(_stage_path, allow_pickle=False) as _payload:
                    _stage_nodes[_stage] = np.asarray(_payload["nodes"], dtype=np.float64).reshape(
                        -1, 5
                    )
                _original = _stage_nodes[_stage][
                    np.isin(_stage_nodes[_stage][:, 0], list(_initial_ids))
                ]
                if len(_original) and np.any(
                    _original[:, 1] != _coords[_original[:, 0].astype(int), 0]
                ):
                    raise RuntimeError(f"{_stem}/{_arm}/{_stage}: original candidate time changed")
            _known_edge_fate = {
                f"{source}:{target}": {
                    "score": _ranks[source, target][0],
                    "rank": _ranks[source, target][1],
                    "selected": {
                        stage: (source, target) in pairs for stage, pairs in _stages.items()
                    },
                }
                for source, target in sorted(_mapped_known_edges)
            }
            _distance_um = [
                float(
                    np.linalg.norm(
                        (_coords[_candidate_id, 1:4] * NATIVE_SPACING_UM)
                        - (_known_nodes[_known_nodes[:, 0] == _gt_id][0, 2:5] * NATIVE_SPACING_UM)
                    )
                )
                for _gt_id, _candidate_id in _fixed_match.gt_to_candidate.items()
            ]
            _video_report["arms"][_arm] = {
                "mean_fixed_match_center_distance_um": (
                    None if not _distance_um else float(np.mean(_distance_um))
                ),
                "stage_counts": _stage_counts,
                "known_edge_fate": _known_edge_fate,
                "initial_graph_node_count": len(_initial_ids),
                "captured_pair_score_count": len(_edge_src),
                "score_capture_threshold": 1.0e-8,
                "scored_stage_contains_only_matched_known_edges": True,
                "stage_node_count": {stage: len(nodes) for stage, nodes in _stage_nodes.items()},
                "stage_added_node_count": {
                    stage: sum(int(row[0]) not in _initial_ids for row in nodes)
                    for stage, nodes in _stage_nodes.items()
                },
                "candidate_cache_sha256": _audit_sha256(_cache_path),
                "final_geff_exists": (
                    AUDIT_ROOT / _arm / "final_graphs" / f"{_stem}.geff"
                ).is_dir(),
            }
        _DIAGNOSTIC[_stem] = _video_report
        print(f"{_stem} fixed-ID complete", flush=True)
    (AUDIT_OUT / "fixed_id_diagnostic.json").write_text(
        json.dumps(_audit_jsonable(_DIAGNOSTIC), indent=2, sort_keys=True) + "\n"
    )
    _EMBRYO_DIAGNOSTIC = {}
    for _embryo in ("44b6", "6bba"):
        _group = [report for report in _DIAGNOSTIC.values() if report["embryo"] == _embryo]
        if len(_group) != 10:
            raise RuntimeError(f"{_embryo}: fixed-ID diagnostic video count changed")
        _EMBRYO_DIAGNOSTIC[_embryo] = {
            "videos": len(_group),
            "candidate_count": sum(report["candidate_count"] for report in _group),
            "matched_gt_nodes": sum(report["matched_gt_nodes"] for report in _group),
            "ambiguous_gt": sum(report["ambiguous_gt"] for report in _group),
            "ambiguous_candidate": sum(report["ambiguous_candidate"] for report in _group),
            "arms": {
                arm: {
                    stage: {
                        key: sum(
                            report["arms"][arm]["stage_counts"][stage][key] for report in _group
                        )
                        for key in _group[0]["arms"][arm]["stage_counts"][stage]
                    }
                    for stage in _group[0]["arms"][arm]["stage_counts"]
                }
                for arm in AUDIT_ARMS
            },
        }
    (AUDIT_OUT / "fixed_id_by_embryo.json").write_text(
        json.dumps(_audit_jsonable(_EMBRYO_DIAGNOSTIC), indent=2, sort_keys=True) + "\n"
    )

    receipt = {
        "experiment": "exp045_x138_coordinate_effect_audit",
        "prediction_kernel_id": "kentookumura/exp045-x138-coordinate-effect-audit-inference",
        "prediction_kernel_version": 1,
        "resume_mode": "local_fixed_id_from_saved_prediction",
        "selected_videos": test_stems,
        "selected_by_embryo": selection,
        "arms": list(AUDIT_ARMS),
        "head_sha256": EXPECTED_HEAD_SHA,
        "selection_manifest_sha256": EXPECTED_SELECTION_SHA,
        "runtime_gate_sha256": EXPECTED_RUNTIME_GATE_SHA,
        "fixed_id_by_embryo": _EMBRYO_DIAGNOSTIC,
        "identity": _identity,
        "diagnostic_sha256": _audit_sha256(AUDIT_OUT / "fixed_id_diagnostic.json"),
        "embryo_summary_sha256": _audit_sha256(AUDIT_OUT / "fixed_id_by_embryo.json"),
        "runtime_seconds": time.time() - start,
        "official_metric": False,
        "prediction_rerun": False,
        "competition_submission_created": False,
    }
    (AUDIT_OUT / "local_fixed_id_receipt.json").write_text(
        json.dumps(_audit_jsonable(receipt), indent=2, sort_keys=True) + "\n"
    )
    print("Fixed-ID local receipt:", AUDIT_OUT / "local_fixed_id_receipt.json", flush=True)


if __name__ == "__main__":
    main()
