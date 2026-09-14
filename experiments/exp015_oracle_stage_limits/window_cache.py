from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

CACHE_SCHEMA_VERSION = 1
METADATA_KEY = "__metadata_json__"


def json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_schema(arrays: dict[str, np.ndarray]) -> list[dict[str, object]]:
    return [
        {
            "name": name,
            "dtype": np.asarray(arrays[name]).dtype.str,
            "shape": list(np.asarray(arrays[name]).shape),
        }
        for name in sorted(arrays)
    ]


def array_content_sha256(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for item in array_schema(arrays):
        name = str(item["name"])
        array = np.ascontiguousarray(arrays[name])
        digest.update(json.dumps(item, separators=(",", ":"), sort_keys=True).encode("utf-8"))
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _normalise_arrays(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if not arrays:
        raise ValueError("window cache requires at least one array")
    if METADATA_KEY in arrays:
        raise ValueError(f"reserved cache key: {METADATA_KEY}")
    normalised: dict[str, np.ndarray] = {}
    for name, value in arrays.items():
        if not name or not isinstance(name, str):
            raise TypeError(f"invalid cache array name: {name!r}")
        array = np.ascontiguousarray(value)
        if array.dtype.hasobject:
            raise TypeError(f"object dtype is not allowed in cache array {name!r}")
        normalised[name] = array
    return normalised


def write_window_cache(
    path: Path,
    *,
    metadata: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> dict[str, object]:
    normalised = _normalise_arrays(arrays)
    cache_metadata = {
        **metadata,
        "schema_version": CACHE_SCHEMA_VERSION,
        "array_schema": array_schema(normalised),
        "array_content_sha256": array_content_sha256(normalised),
    }
    metadata_bytes = json.dumps(
        cache_metadata,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    np.savez(
        path,
        **normalised,
        **{METADATA_KEY: np.frombuffer(metadata_bytes, dtype=np.uint8)},
    )
    write_seconds = time.perf_counter() - started
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "file_sha256": file_sha256(path),
        "schema_sha256": json_sha256(cache_metadata["array_schema"]),
        "content_sha256": cache_metadata["array_content_sha256"],
        "write_seconds": float(write_seconds),
        "metadata": cache_metadata,
    }


def read_window_cache(
    path: Path,
    *,
    expected_metadata: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    started = time.perf_counter()
    with np.load(path, allow_pickle=False) as saved:
        if METADATA_KEY not in saved.files:
            raise ValueError(f"cache metadata missing: {path}")
        metadata = json.loads(saved[METADATA_KEY].tobytes().decode("utf-8"))
        arrays = {
            name: np.ascontiguousarray(saved[name]) for name in saved.files if name != METADATA_KEY
        }
    read_seconds = time.perf_counter() - started
    if not isinstance(metadata, dict):
        raise TypeError(f"cache metadata must be an object: {path}")
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(
                {
                    "cache_metadata_mismatch": key,
                    "expected": expected,
                    "actual": metadata.get(key),
                    "path": str(path),
                }
            )
    if metadata.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError(f"unsupported cache schema: {metadata.get('schema_version')}")
    actual_schema = array_schema(arrays)
    actual_content_sha = array_content_sha256(arrays)
    if metadata.get("array_schema") != actual_schema:
        raise ValueError(f"cache array schema mismatch: {path}")
    if metadata.get("array_content_sha256") != actual_content_sha:
        raise ValueError(f"cache array content mismatch: {path}")
    return arrays, {
        "read_seconds": float(read_seconds),
        "schema_sha256": json_sha256(actual_schema),
        "content_sha256": actual_content_sha,
    }


def assert_exact_arrays(
    expected: dict[str, np.ndarray],
    actual: dict[str, np.ndarray],
) -> None:
    if sorted(expected) != sorted(actual):
        raise AssertionError(
            {"expected_array_keys": sorted(expected), "actual_array_keys": sorted(actual)}
        )
    for name in sorted(expected):
        left = np.asarray(expected[name])
        right = np.asarray(actual[name])
        if left.dtype != right.dtype or left.shape != right.shape:
            raise AssertionError(
                {
                    "array": name,
                    "expected_dtype": left.dtype.str,
                    "actual_dtype": right.dtype.str,
                    "expected_shape": list(left.shape),
                    "actual_shape": list(right.shape),
                }
            )
        if not np.array_equal(left, right, equal_nan=True):
            raise AssertionError(f"cache round-trip changed array {name!r}")


def append_jsonl(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
