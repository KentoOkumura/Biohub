from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
from requests.exceptions import HTTPError

EXPERIMENT = "exp032_three_frame_ten_epoch_training"
ROOT = Path(__file__).resolve().parents[2]
PREDICTION_MANIFEST = (
    ROOT / "experiments/exp027_multi_frame_tracker/artifacts/kaggle_diagnostic_v1/"
    "context_diagnostic/prediction_manifest.json"
)
CACHE_KERNEL = "exp015-oracle-stage-limits-inference"
GEFF_KERNEL = "exp032-train-geff-export"
GEFF_EXPORT_SHA256 = "917b7d354bab6346cfb94dff6ccc478ad52f94a82effafeb2b7e543b5d344663"
GEFF_EXPORT_BYTES = 2387014
PREDICTION_KERNEL = "exp027-multi-frame-tracker-diagnostic"


def retry(call: Any) -> Any:
    for attempt in range(9):
        try:
            return call()
        except HTTPError as exc:
            if exc.response is None or exc.response.status_code not in {429, 500, 502, 503, 504}:
                raise
            if attempt == 8:
                raise
            retry_after = exc.response.headers.get("Retry-After")
            delay = (
                max(2, min(90, int(retry_after)))
                if retry_after and retry_after.isdigit()
                else min(90, 2 ** (attempt + 1))
            )
            print(f"Kaggle HTTP {exc.response.status_code}; retry in {delay}s", flush=True)
            time.sleep(delay)
    raise AssertionError("unreachable")


def kernel_files(api: KaggleApi, slug: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    token = None
    with api.build_kaggle_client() as kaggle:
        for _ in range(300):
            request = ApiListKernelSessionOutputRequest()
            request.user_name = "kentookumura"
            request.kernel_slug = slug
            request.page_size = 200
            if token:
                request.page_token = token
            response = retry(
                lambda request=request: (
                    kaggle.kernels.kernels_api_client.list_kernel_session_output(request)
                )
            )
            for item in response.files or []:
                output.append(
                    {"path": str(item.file_name), "url": str(item.url), "size_bytes": None}
                )
            token = response.next_page_token or None
            if not token:
                break
        else:
            raise RuntimeError(f"Kaggle output pagination exceeded 300 pages: {slug}")
    names = [row["path"] for row in output]
    if len(names) != len(set(names)) or any(
        not row["url"].startswith("https://") for row in output
    ):
        raise RuntimeError(f"Kaggle output has duplicates or missing HTTPS URLs: {slug}")
    return output


def cache_files(api: KaggleApi) -> list[dict[str, Any]]:
    output = [
        row
        for row in kernel_files(api, CACHE_KERNEL)
        if row["path"] == "window_cache_summary.json" or row["path"].startswith("window_cache/")
    ]
    windows = [
        row
        for row in output
        if row["path"].startswith("window_cache/") and row["path"].endswith(".npz")
    ]
    if len(windows) != 19701 or not any(
        row["path"] == "window_cache_summary.json" for row in output
    ):
        raise RuntimeError("Kaggle exp015 cache listing does not contain 19701 windows and summary")
    return output


def geff_files(api: KaggleApi) -> list[dict[str, Any]]:
    output = [
        row for row in kernel_files(api, GEFF_KERNEL) if row["path"] == "exp032_train_geff.zip"
    ]
    if len(output) != 1:
        raise RuntimeError("Kaggle GEFF export is not complete")
    return [
        {
            **output[0],
            "size_bytes": GEFF_EXPORT_BYTES,
            "sha256": GEFF_EXPORT_SHA256,
        }
    ]


def prediction_files(api: KaggleApi) -> list[dict[str, Any]]:
    declared = json.loads(PREDICTION_MANIFEST.read_text(encoding="utf-8"))
    if len(declared) != 128:
        raise RuntimeError("Expected 128 saved exp027 predictions")
    expected = {"context_diagnostic/" + row["path"]: row["file_sha256"] for row in declared}
    if len(expected) != 128:
        raise RuntimeError("Duplicate saved exp027 prediction paths")
    output = [
        {**row, "sha256": expected[row["path"]]}
        for row in kernel_files(api, PREDICTION_KERNEL)
        if row["path"] in expected
    ]
    if {row["path"] for row in output} != set(expected):
        raise RuntimeError("Kaggle exp027 output is missing saved predictions")
    return output


def write_manifest(path: Path, source: str, files: list[dict[str, Any]]) -> None:
    if not files or len(files) != len({row["path"] for row in files}):
        raise RuntimeError(f"{source} manifest is empty or duplicated")
    payload = {
        "experiment": EXPERIMENT,
        "source": source,
        "created_at": datetime.now(UTC).isoformat(),
        "files": files,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    path.chmod(0o600)
    print(f"Prepared {source}: {len(files)} files", flush=True)


def prepare(stage: str, cache_path: Path, geff_path: Path, prediction_path: Path | None) -> None:
    api = KaggleApi()
    api.authenticate()
    write_manifest(geff_path, "exp032_geff_export_v1", geff_files(api))
    if stage == "diagnostic":
        if prediction_path is None:
            raise RuntimeError("Diagnostic needs the saved prediction manifest path")
        write_manifest(prediction_path, "exp027_saved_predictions_v1", prediction_files(api))
    write_manifest(cache_path, "exp015_kernel_output_v1", cache_files(api))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("train", "diagnostic"), required=True)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--geff-manifest", type=Path, required=True)
    parser.add_argument("--prediction-manifest", type=Path)
    arguments = parser.parse_args()
    prepare(
        arguments.stage,
        arguments.cache_manifest,
        arguments.geff_manifest,
        arguments.prediction_manifest,
    )
