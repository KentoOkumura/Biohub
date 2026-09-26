"""Reduce the saved exp045 refined score caches for exp047 Kaggle replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from candidate_policy import EXPANDED_THRESHOLD, select_candidate_edges


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    stems = sorted(stem for group in selection["selected"].values() for stem in group)
    if len(stems) != 20 or len(set(stems)) != 20:
        raise ValueError("expected 20 unique head-unseen videos")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": "exp045 refined v1 recovered Kaggle cache",
        "strict_threshold": EXPANDED_THRESHOLD,
        "selection_sha256": sha256(args.selection),
        "videos": {},
    }
    for stem in stems:
        source = args.source / f"{stem}.npz"
        destination = args.output / source.name
        with np.load(source, allow_pickle=False) as payload:
            coords = np.asarray(payload["coords"], dtype=np.float32)
            src = np.asarray(payload["edge_src"], dtype=np.int32)
            tgt = np.asarray(payload["edge_tgt"], dtype=np.int32)
            prob = np.asarray(payload["edge_prob"], dtype=np.float32)
            admitted = np.asarray(payload["admitted"], dtype=np.float64).reshape(-1, 4)
            low_coords = np.asarray(payload["low_coords"], dtype=np.int16)
            low_score = np.asarray(payload["low_score"], dtype=np.float32)
        keep = prob > EXPANDED_THRESHOLD
        src, tgt, prob = src[keep], tgt[keep], prob[keep]
        edges = select_candidate_edges(coords, src, tgt, prob, admitted)
        np.savez_compressed(
            destination,
            coords=coords,
            edge_src=src,
            edge_tgt=tgt,
            edge_prob=prob,
            admitted=admitted,
            low_coords=low_coords,
            low_score=low_score,
        )
        with np.load(destination, allow_pickle=False) as check:
            replay = select_candidate_edges(
                check["coords"],
                check["edge_src"],
                check["edge_tgt"],
                check["edge_prob"],
                check["admitted"],
            )
        if not np.array_equal(replay.expanded, edges.expanded):
            raise RuntimeError(f"{stem}: reduced cache changed candidate edges")
        manifest["videos"][stem] = {
            "source_sha256": sha256(source),
            "reduced_sha256": sha256(destination),
            "source_score_count": int(len(keep)),
            "reduced_score_count": int(len(src)),
            "baseline_candidates": len(edges.baseline),
            "added_candidates": len(edges.added),
            "expanded_candidates": len(edges.expanded),
            "minimum_reduced_probability": float(prob.min()) if len(prob) else None,
            "nodes": len(coords),
            "low_detection_peaks": len(low_coords),
        }
        print(stem, manifest["videos"][stem]["reduced_score_count"], flush=True)
    (args.output / "cache_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
