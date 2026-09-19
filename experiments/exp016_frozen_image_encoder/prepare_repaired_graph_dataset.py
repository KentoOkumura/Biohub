from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import graph_inference as gi


def prepare_dataset(source: Path, destination: Path, dataset_id: str) -> dict[str, object]:
    graph_source = source / "oracle_final_graphs"
    graph_paths = sorted(graph_source.glob("*.npz"))
    if len(graph_paths) != 199 or len({path.stem for path in graph_paths}) != 199:
        raise RuntimeError(f"expected 199 unique repaired graphs under {graph_source}")
    required_files = [
        source / "run_stats.csv",
        source / "recovered_ilp_input_receipt.json",
    ]
    for required in required_files:
        if not required.is_file():
            raise FileNotFoundError(required)
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError(f"destination must be empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    graph_destination = destination / "oracle_final_graphs"
    graph_destination.mkdir()
    for path in graph_paths:
        shutil.copy2(path, graph_destination / path.name)
    for path in required_files:
        shutil.copy2(path, destination / path.name)

    graph_records = [[path.name, gi.sha256_file(path)] for path in graph_paths]
    manifest: dict[str, object] = {
        "experiment": "exp016_frozen_image_encoder",
        "source_kernel_ref": "kentookumura/exp016-frozen-image-encoder-official-eval",
        "source_kernel_version": 5,
        "graph_repair_complete": True,
        "graph_count": 199,
        "graph_names": [path.stem for path in graph_paths],
        "graph_files_sha256": graph_records,
        "compact_graph_content_sha256": gi.compact_graph_content_sha256(graph_source),
        "run_stats_sha256": gi.sha256_file(source / "run_stats.csv"),
        "recovered_ilp_input_receipt_sha256": gi.sha256_file(
            source / "recovered_ilp_input_receipt.json"
        ),
        "submission_created": False,
    }
    manifest["manifest_sha256"] = gi.json_sha256(manifest)
    (destination / "repaired_graph_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metadata = {
        "title": "exp016 repaired graphs from official eval v5",
        "id": dataset_id,
        "licenses": [{"name": "CC0-1.0"}],
    }
    (destination / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    args = parser.parse_args()
    manifest = prepare_dataset(args.source, args.destination, args.dataset_id)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
