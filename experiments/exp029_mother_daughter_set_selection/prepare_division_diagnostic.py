from __future__ import annotations

import argparse
import hashlib
import json
import time
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
import yaml

EXP = Path(__file__).resolve().parent
OUTPUT = EXP / "artifacts/division_diagnostic_v1"
GEFF_SHA = "917b7d354bab6346cfb94dff6ccc478ad52f94a82effafeb2b7e543b5d344663"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_annotations(train_dir: Path, scale: list[float]) -> dict:
    import zarr

    result = {}
    for path in sorted(train_dir.glob("*.geff")):

        def array(relative: str, root: Path = path) -> np.ndarray:
            return np.asarray(zarr.open_array(str(root / relative), mode="r")[:])

        ids = array("nodes/ids").astype(np.int64)
        frames = array("nodes/props/t/values").astype(np.int64)
        coords = np.column_stack([array(f"nodes/props/{axis}/values") for axis in "zyx"])
        coords = coords.astype(np.float64) * np.asarray(scale)
        edges = array("edges/ids").astype(np.int64)
        by_frame = defaultdict(list)
        node_frames = {}
        incoming = defaultdict(list)
        outgoing = defaultdict(list)
        assert len(set(ids.tolist())) == len(ids)
        for node, frame, coord in zip(ids, frames, coords, strict=True):
            by_frame[int(frame)].append((int(node), coord))
            node_frames[int(node)] = int(frame)
        for parent, child in edges:
            assert int(parent) in node_frames and int(child) in node_frames
            incoming[int(child)].append(int(parent))
            outgoing[int(parent)].append(int(child))
        division_frames = set()
        divisions = 0
        for parent, children in outgoing.items():
            adjacent = [
                child for child in children if node_frames[child] == node_frames[parent] + 1
            ]
            if len(adjacent) >= 2:
                division_frames.add(node_frames[parent])
                divisions += 1
        result[path.stem] = {
            "frames": dict(by_frame),
            "incoming": dict(incoming),
            "edge_count": len(edges),
            "division_frames": sorted(division_frames),
            "adjacent_gt_divisions": divisions,
        }
    return result


def extract_geff(archive_path: Path) -> None:
    assert sha(archive_path) == GEFF_SHA, "GEFF archive SHA mismatch"
    root = OUTPUT / "input"
    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read("GEFF_MANIFEST.json"))
        assert manifest["samples"] == 199 and len(manifest["files"]) == 4179
        for row in manifest["files"]:
            relative = Path(row["path"])
            assert not relative.is_absolute() and ".." not in relative.parts
            payload = archive.read(row["path"])
            assert hashlib.sha256(payload).hexdigest() == row["sha256"]
            dest = root / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(payload)
        (root / "GEFF_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")


def download_cache(wanted: set[str]) -> None:
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

    api = KaggleApi()
    api.authenticate()
    checkpoint_path = Path("/tmp/exp029-cache-listing.json")
    previous = json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else {}
    found = previous.get("found", {})
    token = previous.get("token")
    with api.build_kaggle_client() as client:
        for page in range(300):
            req = ApiListKernelSessionOutputRequest()
            req.user_name = "kentookumura"
            req.kernel_slug = "exp015-oracle-stage-limits-inference"
            req.page_size = 200
            if token:
                req.page_token = token
            for attempt in range(8):
                try:
                    response = client.kernels.kernels_api_client.list_kernel_session_output(req)
                    break
                except requests.HTTPError as exc:
                    status = exc.response.status_code if exc.response is not None else None
                    if status not in (429, 500, 502, 503, 504) or attempt == 7:
                        raise
                    print(f"Cache listing HTTP {status}; retrying after 30 seconds", flush=True)
                    time.sleep(30)
            for item in response.files or []:
                name = str(item.file_name)
                if name in wanted:
                    found[name] = str(item.url)
            token = response.next_page_token or None
            checkpoint_path.write_text(json.dumps({"found": found, "token": token}))
            checkpoint_path.chmod(0o600)
            time.sleep(0.7)
            if page % 20 == 0:
                print(
                    f"Cache listing page {page + 1}: found {len(found)}/{len(wanted)}", flush=True
                )
            if not token:
                break
    missing = wanted - set(found)
    assert not missing, f"Missing cache files: {sorted(missing)}"

    def fetch(item: tuple[str, str]) -> dict:
        name, url = item
        path = OUTPUT / "input" / name
        if not path.exists():
            for attempt in range(4):
                response = requests.get(url, timeout=90)
                if response.status_code == 200:
                    break
                if response.status_code not in (429, 500, 502, 503, 504):
                    raise RuntimeError(f"Download failed for {name}: HTTP {response.status_code}")
                time.sleep(2 ** (attempt + 1))
            if response.status_code != 200:
                raise RuntimeError(f"Download retry failed for {name}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(response.content)
        return {"path": name, "bytes": path.stat().st_size, "sha256": sha(path)}

    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(fetch, sorted(found.items())))
    (OUTPUT / "input_files.json").write_text(
        json.dumps(
            {
                "kernel": "kentookumura/exp015-oracle-stage-limits-inference",
                "version": None,
                "geff_archive_sha256": GEFF_SHA,
                "files": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Downloaded {len(rows)} cache files ({sum(x['bytes'] for x in rows)} bytes)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geff-archive", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    extract_geff(args.geff_archive)
    config = yaml.safe_load((EXP / "config.yaml").read_text())
    annotations = read_annotations(
        OUTPUT / "input/train", config["data"]["annotation"]["voxel_scale_zyx_um"]
    )
    assert len(annotations) == 199
    wanted = {"window_cache_summary.json"}
    for sample, annotation in annotations.items():
        for frame in annotation["division_frames"]:
            wanted.add(f"window_cache/{sample}/{frame:06d}_{frame + 1:06d}.npz")
    print(
        f"Selected {len(wanted) - 1} windows containing "
        f"{sum(a['adjacent_gt_divisions'] for a in annotations.values())} GT divisions",
        flush=True,
    )
    (OUTPUT / "requested_windows.json").write_text(
        json.dumps(sorted(wanted), indent=2) + "\n", encoding="utf-8"
    )
    if args.download:
        download_cache(wanted)


if __name__ == "__main__":
    main()
