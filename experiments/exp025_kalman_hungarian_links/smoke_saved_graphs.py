"""Exercise eight frames from each embryo; never run official scoring locally."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=8)
    args = parser.parse_args()
    if not 2 <= args.frames <= 10:
        raise ValueError("local smoke is restricted to 2-10 frames")
    exp = Path(__file__).resolve().parent
    source = exp / f"{exp.name}_diagnostic.py"
    spec = importlib.util.spec_from_file_location("exp025_smoke_module", source)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    cfg = yaml.safe_load((exp / "config.yaml").read_text())
    manifest = json.loads(
        m.require_sha(
            exp / cfg["data"]["manifest_file"], cfg["data"]["manifest_sha256"]
        ).read_text()
    )
    sample_names = sorted(s for r in manifest["archives"] for s in r["samples"])
    selected = [
        next(s for s in sample_names if s.startswith(e + "_"))
        for e in sorted(cfg["validation"]["expected_embryo_counts"])
    ]
    archives = [r for r in manifest["archives"] if set(r["samples"]) & set(selected)]
    subset = {
        "archives": archives,
        "expected_sample_count": sum(len(r["samples"]) for r in archives),
    }
    candidates = m.materialize_candidates(args.archive_root, args.output, subset)
    scale = np.asarray(cfg["validation"]["scale_zyx_um"])
    params = cfg["model"]["params"]
    # Deliberately uncalibrated numeric fixture: this smoke makes no accuracy claim.
    noise = {
        "observation_variance": [params["observation_variance_floor"]] * 3,
        "acceleration_variance": [params["acceleration_variance_floor"]] * 3,
        "initial_velocity_variance": [params["velocity_variance_floor"]] * 3,
        "daughter_covariance_multiplier": params["daughter_covariance_multiplier"],
        "nll_center": 0.0,
        "nll_scale": params["motion_nll_scale"],
    }
    records = []
    reference_sha = dict(manifest["reference_graphs"])
    for sample in selected:
        candidate = m.load_geff(candidates[sample], scale)
        name = f"{sample}.npz"
        reference = m.load_compact(
            m.require_sha(args.reference_root / name, reference_sha[name]), scale
        )
        graph, reserved, receipt = m.reserve_divisions(candidate, reference)
        mask = graph.frames < args.frames
        ids = set(graph.ids[mask])
        edge_mask = np.asarray([a in ids and b in ids for a, b in graph.edges])
        graph = m.Graph(
            graph.ids[mask],
            graph.frames[mask],
            graph.positions[mask],
            graph.edges[edge_mask],
            graph.probabilities[edge_mask],
        )
        reserved = np.asarray(
            [(a, b) for a, b in reserved if a in ids and b in ids], dtype=np.int64
        ).reshape(-1, 2)
        digest = graph.digest()
        for variant, weight in (("image_only", 0.0), ("kalman", params["motion_weight_grid"][0])):
            started = time.monotonic()
            output = m.decode(
                graph,
                reserved,
                noise,
                motion_weight=weight,
                no_match_cost=params["no_match_cost_grid"][1],
                cfg=params,
            )
            assert graph.digest() == digest
            m.validate_output(graph, reserved, output["edges"])
            assert np.isfinite(output["covariances"]).all()
            records.append(
                {
                    "sample": sample,
                    "variant": variant,
                    "frames": args.frames,
                    "nodes": len(graph.ids),
                    "edges": len(output["edges"]),
                    "reserved_divisions": len(reserved) // 2,
                    "elapsed_seconds": time.monotonic() - started,
                    "input_sha256": digest,
                    "output_edges_sha256": m.json_sha(output["edges"].tolist()),
                }
            )
    result = {
        "source_sha256": m.file_sha(source),
        "records": records,
        "calibrated": False,
        "official_metric_computed": False,
        "full_graph_inference": False,
        "passed": True,
    }
    m.write_json(args.output / "smoke_result.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
