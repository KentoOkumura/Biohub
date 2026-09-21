from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

COMPETITION = "biohub-cell-tracking-during-development"
OUTPUT = Path("/kaggle/working/exp032_train_geff.zip")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    candidates = (
        Path("/kaggle/input/competitions") / COMPETITION / "train",
        Path("/kaggle/input") / COMPETITION / "train",
    )
    source = next((path for path in candidates if path.is_dir()), None)
    if source is None:
        raise FileNotFoundError(f"Competition train GEFF mount missing: {candidates}")
    samples = sorted(source.glob("*.geff"))
    files = sorted(path for sample in samples for path in sample.rglob("*") if path.is_file())
    if len(samples) != 199 or len(files) != 4179:
        raise RuntimeError(
            f"Unexpected GEFF collection: {len(samples)} samples, {len(files)} files"
        )
    manifest = {
        "competition": COMPETITION,
        "samples": len(samples),
        "files": [
            {
                "path": "train/" + path.relative_to(source).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in files
        ],
    }
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for row, path in zip(manifest["files"], files, strict=True):
            archive.write(path, row["path"])
        archive.writestr(
            "GEFF_MANIFEST.json", json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
        )
    with zipfile.ZipFile(OUTPUT) as archive:
        if set(archive.namelist()) != {row["path"] for row in manifest["files"]} | {
            "GEFF_MANIFEST.json"
        }:
            raise RuntimeError("GEFF archive member list differs from manifest")
    print(
        json.dumps(
            {
                "archive": str(OUTPUT),
                "bytes": OUTPUT.stat().st_size,
                "sha256": sha256(OUTPUT),
                "samples": len(samples),
                "files": len(files),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
