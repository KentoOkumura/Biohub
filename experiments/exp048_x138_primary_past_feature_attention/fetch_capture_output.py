"""Download the already-completed exp044 v1 capture without starting a kernel."""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

import requests
from kaggle.api.kaggle_api_extended import ApiListKernelSessionOutputRequest, KaggleApi

OWNER = "kentookumura"
SLUG = "exp044-x138-past-candidate-knn-attention-train"
PREFIX = "tracker_capture/"


def _fetch(item: object, root: Path) -> tuple[str, int, bool]:
    name = str(item.file_name)
    relative = PurePosixPath(name)
    if not name.startswith(PREFIX) or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unexpected output path: {name}")
    target = root.joinpath(*relative.parts)
    if target.is_file() and target.stat().st_size:
        return name, target.stat().st_size, False
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".download")
    for attempt in range(6):
        try:
            with requests.get(str(item.url), stream=True, timeout=(30, 180)) as response:
                response.raise_for_status()
                expected = response.headers.get("Content-Length")
                actual = 0
                with temporary.open("wb") as output:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        output.write(chunk)
                        actual += len(chunk)
                if expected is not None and actual != int(expected):
                    raise OSError(f"truncated {name}: {actual} != {expected}")
            os.replace(temporary, target)
            return name, actual, True
        except (requests.RequestException, OSError) as error:
            if attempt == 5:
                raise RuntimeError(f"Download failed after retries: {name}") from error
            time.sleep(min(2**attempt, 16))
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--page-size", type=int, default=200)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("workers must be between 1 and 16")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    token = None
    pages = files = downloaded = total_bytes = 0
    with api.build_kaggle_client() as kaggle:
        while True:
            request = ApiListKernelSessionOutputRequest()
            request.user_name = OWNER
            request.kernel_slug = SLUG
            api._set_paging(request, args.page_size, token)
            for attempt in range(6):
                try:
                    response = kaggle.kernels.kernels_api_client.list_kernel_session_output(request)
                    break
                except (requests.RequestException, OSError) as error:
                    if attempt == 5:
                        raise RuntimeError("Could not list existing notebook output") from error
                    time.sleep(min(2**attempt, 16))
            selected = [item for item in response.files or [] if item.file_name.startswith(PREFIX)]
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(_fetch, item, root) for item in selected]
                for future in as_completed(futures):
                    _, count, fresh = future.result()
                    files += 1
                    downloaded += int(fresh)
                    total_bytes += count
            pages += 1
            token = response.next_page_token
            print(
                f"pages={pages} files={files} downloaded={downloaded} "
                f"bytes={total_bytes} more={bool(token)}",
                flush=True,
            )
            if not token:
                break


if __name__ == "__main__":
    main()
