"""Package audited version-2 graphs for a private Kaggle input Dataset."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "artifacts/kaggle_evaluate"
OUTPUT = EVIDENCE / "v2_output"
MANIFEST = ROOT / "assets/reused_graph_manifest.json"
DESTINATION = EVIDENCE / "reuse_dataset_upload"
DATASET_ID = "kentookumura/exp051-x138-clipping-v2-graphs"


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    index = json.loads((EVIDENCE / "v2_remote_output_index.json").read_text())
    if index["kernel_id"] != manifest["source_kernel_id"] or index["kernel_version"] != 2:
        raise ValueError("published output provenance differs")
    names = sorted(p for p in index["files"] if "/final_graphs/" in p or "/stages/" in p)
    if len(names) != 1178:
        raise ValueError(f"expected 1178 graph/stage files, got {len(names)}")
    DESTINATION.mkdir(parents=True, exist_ok=True)
    archive = DESTINATION / "reused_graphs.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for name in names:
            source = OUTPUT / name
            if not source.is_file():
                raise ValueError(f"missing published payload: {name}")
            zf.write(source, name)
        zf.write(OUTPUT / "config.yaml", "config.yaml")
        zf.write(EVIDENCE / "v2_metadata/exp051_selection.json", "exp051_selection.json")
        zf.write(MANIFEST, "reused_graph_manifest.json")
    with zipfile.ZipFile(archive) as zf:
        if len(zf.namelist()) != 1181 or zf.testzip() is not None:
            raise ValueError("archive verification failed")
    metadata = {
        "title": "exp051 audited evaluate v2 graph output",
        "id": DATASET_ID,
        "licenses": [{"name": "CC0-1.0"}],
    }
    (DESTINATION / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "dataset_id": DATASET_ID,
                "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "archive_bytes": archive.stat().st_size,
                "payload_file_count": len(names),
                "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
