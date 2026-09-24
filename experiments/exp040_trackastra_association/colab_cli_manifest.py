"""Prepare short-lived Kaggle output URLs without transferring credentials."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
from requests.exceptions import HTTPError

EXPERIMENT = "exp040_trackastra_association"
GEFF_SHA256 = "917b7d354bab6346cfb94dff6ccc478ad52f94a82effafeb2b7e543b5d344663"
GEFF_BYTES = 2387014


def kernel_files(api: KaggleApi, slug: str) -> list[dict[str, Any]]:
    output = []
    token = None
    with api.build_kaggle_client() as client:
        for _ in range(300):
            request = ApiListKernelSessionOutputRequest()
            request.user_name = "kentookumura"
            request.kernel_slug = slug
            request.page_size = 200
            if token:
                request.page_token = token
            for attempt in range(9):
                try:
                    response = client.kernels.kernels_api_client.list_kernel_session_output(request)
                    break
                except HTTPError as exc:
                    if (
                        exc.response is None
                        or exc.response.status_code not in {429, 500, 502, 503, 504}
                        or attempt == 8
                    ):
                        raise
                    time.sleep(min(90, 2 ** (attempt + 1)))
            output.extend(
                {"path": str(row.file_name), "url": str(row.url)} for row in response.files or []
            )
            token = response.next_page_token or None
            if not token:
                break
        else:
            raise RuntimeError("Kaggle output pagination exceeded 300 pages")
    if len(output) != len({row["path"] for row in output}):
        raise ValueError(f"duplicate Kaggle output path in {slug}")
    if any(not row["url"].startswith("https://") for row in output):
        raise ValueError(f"non-HTTPS Kaggle output URL in {slug}")
    return output


def prepare(cache_path: Path, geff_path: Path) -> None:
    api = KaggleApi()
    api.authenticate()
    cache = [
        row
        for row in kernel_files(api, "exp015-oracle-stage-limits-inference")
        if row["path"] == "window_cache_summary.json" or row["path"].startswith("window_cache/")
    ]
    if len(cache) != 19702 or sum(row["path"].endswith(".npz") for row in cache) != 19701:
        raise RuntimeError("exp015 output does not list the fixed 19701 cache windows")
    geff = [
        row
        for row in kernel_files(api, "exp032-train-geff-export")
        if row["path"] == "exp032_train_geff.zip"
    ]
    if len(geff) != 1:
        raise RuntimeError("GEFF export is unavailable")
    geff[0].update({"size_bytes": GEFF_BYTES, "sha256": GEFF_SHA256})
    for path, source, rows in (
        (cache_path, "exp015_kernel_output_v1", cache),
        (geff_path, "exp032_geff_export_v1", geff),
    ):
        path.write_text(
            json.dumps(
                {
                    "experiment": EXPERIMENT,
                    "source": source,
                    "created_at": datetime.now(UTC).isoformat(),
                    "files": rows,
                },
                separators=(",", ":"),
            )
            + "\n"
        )
        path.chmod(0o600)
        print(f"Prepared {source}: {len(rows)} files", flush=True)
