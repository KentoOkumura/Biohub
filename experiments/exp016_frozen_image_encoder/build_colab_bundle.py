from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

EXPERIMENT = "exp016_frozen_image_encoder"
REPO_ROOT = Path(__file__).resolve().parents[2]
EXP_ROOT = REPO_ROOT / "experiments" / EXPERIMENT
FILES = {
    EXP_ROOT / "colab_graph_replay.py": "code/colab_graph_replay.py",
    EXP_ROOT / "frozen_tracker.py": "code/frozen_tracker.py",
    EXP_ROOT / "graph_inference.py": "code/graph_inference.py",
    REPO_ROOT / "experiments/exp015_oracle_stage_limits/window_cache.py": "code/window_cache.py",
    EXP_ROOT / "config.yaml": "metadata/config.yaml",
    EXP_ROOT / "artifacts/train_v3/model_manifest.json": "metadata/model_manifest.json",
    EXP_ROOT / "artifacts/train_v3/models/fold_0/primary_tracker_best.pth": (
        "models/fold_0/primary_tracker_best.pth"
    ),
    EXP_ROOT / "artifacts/train_v3/models/fold_1/primary_tracker_best.pth": (
        "models/fold_1/primary_tracker_best.pth"
    ),
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info


def build_bundle(output: Path) -> dict[str, object]:
    missing = [str(path) for path in FILES if not path.is_file()]
    if missing:
        raise FileNotFoundError({"colab_bundle_inputs_missing": missing})
    payloads = {destination: source.read_bytes() for source, destination in FILES.items()}
    if any("kaggle.json" in name or "access_token" in name for name in payloads):
        raise RuntimeError("credential-like file found in Colab bundle")
    manifest: dict[str, object] = {
        "bundle_schema_version": 1,
        "experiment": EXPERIMENT,
        "stage": "cached_tracker_and_ilp_replay",
        "contains_credentials": False,
        "contains_competition_images": False,
        "files": [
            {
                "path": name,
                "bytes": len(payloads[name]),
                "sha256": sha256_bytes(payloads[name]),
            }
            for name in sorted(payloads)
        ],
    }
    unsigned = json.dumps(
        manifest, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    manifest["manifest_sha256"] = sha256_bytes(unsigned)
    manifest_bytes = (
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr(zip_info("BUNDLE_MANIFEST.json"), manifest_bytes)
        for name in sorted(payloads):
            archive.writestr(zip_info(name), payloads[name])
    result = {
        **manifest,
        "bundle_path": str(output),
        "bundle_bytes": output.stat().st_size,
        "bundle_sha256": sha256_bytes(output.read_bytes()),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=EXP_ROOT / "artifacts/colab_bundle/exp016_colab_bundle.zip",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_bundle(args.output.resolve())
