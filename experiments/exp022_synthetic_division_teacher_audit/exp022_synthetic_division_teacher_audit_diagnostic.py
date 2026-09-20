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
# # Public synthetic lineage: division triplet teacher diagnostic
#
# This reads the published synthetic output. It does not generate images, train a
# model, or claim that synthetic labels are confirmed labels for real Biohub data.

# %%
from __future__ import annotations

import csv
import hashlib
import json
import os
import time
import zipfile
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.spatial import cKDTree

EXPERIMENT = "exp022_synthetic_division_teacher_audit"
COUNT_FIELDS = (
    "nodes",
    "edges",
    "division_mothers",
    "continuation_mothers",
    "division_recovered",
    "division_with_wrong_triplet",
    "recovered_division_with_wrong_triplet",
    "continuation_with_triplet",
    "positive_triplets",
    "wrong_division_triplets",
    "continuation_triplets",
    "candidate_triplets",
    "max_triplets_per_mother",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def find_config() -> Path:
    candidates = [
        Path.cwd() / "config.yaml",
        Path.cwd() / "experiments" / EXPERIMENT / "config.yaml",
    ]
    found = [p for p in candidates if p.is_file()]
    if len(found) != 1:
        raise RuntimeError(f"expected one experiment config: {found}")
    return found[0]


def find_source(config: dict[str, Any]) -> Path:
    override = os.environ.get("SYNTHETIC_SOURCE_ROOT")
    if override:
        source = Path(override)
        if not (source / "manifest.json").is_file():
            raise FileNotFoundError(source / "manifest.json")
        return source
    slug = config["data"]["kernel_source"].split("/", 1)[1]
    root = Path("/kaggle/input/notebooks")
    found = [
        p.parent
        for p in root.rglob("manifest.json")
        if p.parent.name == config["data"]["output_directory"] and slug in str(p)
    ]
    if len(found) != 1:
        raise RuntimeError(f"expected one public synthetic output: {found}")
    return found[0]


def read_array_header(archive: zipfile.ZipFile, name: str) -> tuple[tuple[int, ...], str]:
    with archive.open(name + ".npy") as member:
        version = np.lib.format.read_magic(member)
        if version == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(member)
        elif version == (2, 0):
            shape, _, dtype = np.lib.format.read_array_header_2_0(member)
        else:
            raise ValueError(f"unsupported npy header version: {version}")
    return tuple(shape), str(dtype)


def validate_graph(
    nodes: np.ndarray, edges: np.ndarray, divisions: np.ndarray, expected_t: int
) -> tuple[np.ndarray, list[list[int]]]:
    if nodes.ndim != 2 or nodes.shape[1] != 5 or nodes.dtype != np.float32:
        raise ValueError("nodes must be float32[N,5]")
    if edges.ndim != 2 or edges.shape[1] != 2 or edges.dtype != np.int32:
        raise ValueError("edges must be int32[E,2]")
    if divisions.ndim != 1 or divisions.dtype != np.int32:
        raise ValueError("divisions must be int32[D]")
    if not np.isfinite(nodes).all():
        raise ValueError("nonfinite node values")
    times = nodes[:, 0]
    if (
        not np.array_equal(times, times.astype(np.int32))
        or not ((times >= 0) & (times < expected_t)).all()
    ):
        raise ValueError("invalid node time")
    if set(times.astype(int)) != set(range(expected_t)):
        raise ValueError("missing frame")
    if np.any(edges < 0) or np.any(edges >= len(nodes)):
        raise ValueError("edge index outside node array")
    if not (times[edges[:, 1]] == times[edges[:, 0]] + 1).all():
        raise ValueError("edge must join adjacent frames")
    if len(np.unique(edges, axis=0)) != len(edges):
        raise ValueError("duplicate edge")
    out = np.bincount(edges[:, 0], minlength=len(nodes))
    incoming = np.bincount(edges[:, 1], minlength=len(nodes))
    if not (out[times < expected_t - 1] >= 1).all() or not (out[times < expected_t - 1] <= 2).all():
        raise ValueError("interior mother must have one or two daughters")
    if (
        np.any(out[times == expected_t - 1])
        or np.any(incoming[times == 0])
        or not (incoming[times > 0] == 1).all()
    ):
        raise ValueError("lineage endpoint/incoming inconsistency")
    if not np.array_equal(np.sort(divisions), np.flatnonzero(out == 2)):
        raise ValueError("divisions do not match two-daughter mothers")
    children: list[list[int]] = [[] for _ in range(len(nodes))]
    for parent, child in edges:
        children[int(parent)].append(int(child))
    return out, children


def audit_sequence(
    path: Path, row: dict[str, Any], config: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, list[float]]]:
    valid = config["validation"]
    if path.stat().st_size != row["bytes"]:
        raise ValueError(f"manifest size mismatch: {path}")
    with zipfile.ZipFile(path) as archive:
        for key, shape, dtype in (
            ("nodes", (row["n_nodes"], 5), "float32"),
            ("edges", (row["n_edges"], 2), "int32"),
            ("divisions", (row["n_divisions"],), "int32"),
            ("volumes", (row["T"], 64, 64, 64), "uint16"),
            ("voxel_um_pooled", (3,), "float32"),
        ):
            actual = read_array_header(archive, key)
            if actual != (shape, dtype):
                raise ValueError(f"{path.name}: {key} header {actual} != {(shape, dtype)}")
    with np.load(path, allow_pickle=False) as archive:
        nodes = archive["nodes"]
        edges = archive["edges"]
        divisions = archive["divisions"]
        if not np.allclose(archive["voxel_um_pooled"], valid["expected_pooled_voxel_zyx_um"]):
            raise ValueError("pooled voxel mismatch")
    if row["T"] != valid["expected_sequence_length"]:
        raise ValueError("unexpected sequence length")
    out, children = validate_graph(nodes, edges, divisions, row["T"])
    xyz = nodes[:, 1:4].astype(np.float64) * np.asarray(valid["native_voxel_zyx_um"])
    mother_radius = float(valid["parent_daughter_max_um"])
    sister_radius = float(valid["sister_max_um"])
    guard = int(valid["max_triplets_per_parent_guard"])
    counts: Counter[str] = Counter()
    distances: dict[str, list[float]] = {"mother_daughter_um": [], "sister_um": []}
    counts.update(
        nodes=len(nodes),
        edges=len(edges),
        division_mothers=len(divisions),
        continuation_mothers=int(np.count_nonzero(out == 1)),
    )
    for t in range(row["T"] - 1):
        parents = np.flatnonzero(nodes[:, 0] == t)
        next_ids = np.flatnonzero(nodes[:, 0] == t + 1)
        tree = cKDTree(xyz[next_ids])
        nearby = tree.query_ball_point(xyz[parents], mother_radius)
        for parent, local_ids in zip(parents, nearby, strict=True):
            mother = int(parent)
            candidates = sorted(int(next_ids[i]) for i in local_ids)
            true_children = children[mother]
            if len(true_children) == 2:
                distances["mother_daughter_um"].extend(
                    float(np.linalg.norm(xyz[mother] - xyz[child])) for child in true_children
                )
                distances["sister_um"].append(
                    float(np.linalg.norm(xyz[true_children[0]] - xyz[true_children[1]]))
                )
            triplets = 0
            wrong = 0
            positive = 0
            for first, second in combinations(candidates, 2):
                if np.linalg.norm(xyz[first] - xyz[second]) > sister_radius:
                    continue
                triplets += 1
                if triplets > guard:
                    raise RuntimeError(f"candidate guard exceeded: {path.name} mother {mother}")
                if len(true_children) == 2:
                    if {first, second} == set(true_children):
                        positive += 1
                    else:
                        wrong += 1
                else:
                    counts["continuation_triplets"] += 1
            counts["candidate_triplets"] += triplets
            counts["max_triplets_per_mother"] = max(counts["max_triplets_per_mother"], triplets)
            if len(true_children) == 2:
                counts["positive_triplets"] += positive
                counts["wrong_division_triplets"] += wrong
                counts["division_recovered"] += int(positive == 1)
                counts["division_with_wrong_triplet"] += int(wrong > 0)
                counts["recovered_division_with_wrong_triplet"] += int(positive == 1 and wrong > 0)
            else:
                counts["continuation_with_triplet"] += int(triplets > 0)
    if counts["candidate_triplets"] != (
        counts["positive_triplets"]
        + counts["wrong_division_triplets"]
        + counts["continuation_triplets"]
    ):
        raise AssertionError("triplet partition mismatch")
    if counts["positive_triplets"] != counts["division_recovered"]:
        raise AssertionError("one positive per division mother violated")
    result = {
        "sequence": row["file"],
        "file_sha256": sha256_file(path),
        **{k: counts[k] for k in COUNT_FIELDS},
    }
    return result, distances


def describe(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "p50": None, "p90": None, "p95": None, "p99": None, "max": None}
    arr = np.asarray(values)
    return {
        "count": len(values),
        **{f"p{p}": float(np.percentile(arr, p)) for p in (50, 90, 95, 99)},
        "max": float(arr.max()),
    }


def main() -> None:
    started = time.monotonic()
    cfg_path = find_config()
    config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    source = find_source(config)
    valid = config["validation"]
    manifest_path, metadata_path = source / "manifest.json", source / "metadata.json"
    if (
        sha256_file(manifest_path) != valid["expected_manifest_sha256"]
        or sha256_file(metadata_path) != valid["expected_metadata_sha256"]
    ):
        raise ValueError("public source manifest/metadata SHA mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    sequence_rows = manifest["sequences"]
    for key, actual in (
        ("expected_source_sequences", len(sequence_rows)),
        ("expected_source_divisions", metadata["total_divisions"]),
        ("expected_source_nodes", metadata["total_nodes"]),
    ):
        if actual != valid[key]:
            raise ValueError(f"source count mismatch: {key}: {actual}")
    if metadata["seq_len"] != valid["expected_sequence_length"] or not np.allclose(
        metadata["voxel_native_um"], valid["native_voxel_zyx_um"]
    ):
        raise ValueError("source geometry metadata mismatch")
    if (
        sum(row["n_divisions"] for row in sequence_rows) != metadata["total_divisions"]
        or sum(row["n_nodes"] for row in sequence_rows) != metadata["total_nodes"]
    ):
        raise ValueError("manifest count totals disagree with metadata")
    selected = sequence_rows[: valid["sample_count"]]
    results = []
    all_distances: dict[str, list[float]] = {"mother_daughter_um": [], "sister_um": []}
    for index, row in enumerate(selected, 1):
        path = source / row["file"]
        result, distances = audit_sequence(path, row, config)
        results.append(result)
        for key in all_distances:
            all_distances[key].extend(distances[key])
        print(
            f"{index}/{len(selected)} {path.name}: "
            f"{result['division_recovered']}/{result['division_mothers']} divisions, "
            f"{result['candidate_triplets']} triplets",
            flush=True,
        )
    totals = {
        key: (
            max(row[key] for row in results)
            if key == "max_triplets_per_mother"
            else sum(row[key] for row in results)
        )
        for key in COUNT_FIELDS
    }
    output_dir = (
        Path("/kaggle/working" if Path("/kaggle/working").is_dir() else Path.cwd())
        / "synthetic_teacher_audit"
    )
    output_dir.mkdir(exist_ok=True)
    csv_path = output_dir / "per_sequence.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sequence", "file_sha256", *COUNT_FIELDS])
        writer.writeheader()
        writer.writerows(results)
    summary = {
        "experiment": EXPERIMENT,
        "source_kernel": config["data"]["kernel_source"],
        "source_notebook_sha256": valid["source_notebook_sha256"],
        "source_manifest_sha256": sha256_file(manifest_path),
        "source_metadata_sha256": sha256_file(metadata_path),
        "sample_selection": valid["sample_selection"],
        "selected_sequences": len(results),
        "geometry_um": {
            "parent_daughter": valid["parent_daughter_max_um"],
            "sister": valid["sister_max_um"],
        },
        "totals": totals,
        "division_recovery_rate": totals["division_recovered"] / totals["division_mothers"],
        "division_mother_with_wrong_rate": totals["division_with_wrong_triplet"]
        / totals["division_mothers"],
        "recovered_division_with_wrong_rate": totals["recovered_division_with_wrong_triplet"]
        / totals["division_recovered"],
        "distance_um": {key: describe(values) for key, values in all_distances.items()},
        "seconds": time.monotonic() - started,
        "interpretation_limit": (
            "Complete labels apply only to generated lineages at oracle node centers, "
            "not real Biohub detections."
        ),
    }
    summary_path = output_dir / "summary.json"
    write_json(summary_path, summary)
    write_json(
        output_dir / "manifest.json",
        {
            "input": {
                "manifest_sha256": summary["source_manifest_sha256"],
                "metadata_sha256": summary["source_metadata_sha256"],
                "selected_sequence_files": [
                    {"file": r["sequence"], "sha256": r["file_sha256"]} for r in results
                ],
            },
            "output": {
                "per_sequence_sha256": sha256_file(csv_path),
                "summary_sha256": sha256_file(summary_path),
            },
        },
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


# %%
if __name__ == "__main__":
    main()
