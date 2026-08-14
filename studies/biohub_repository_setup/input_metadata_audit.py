# %% [markdown]
# # Biohub input metadata audit

# %% [markdown]
# ## Contents
#
# 1. Imports and constants
# 2. Metadata helpers
# 3. Competition input checks
# 4. Train and test inventory
# 5. Sample submission audit
# 6. Audit output

# %% [markdown]
# ## 1. Imports and constants

# %%
from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

import pandas as pd

COMPETITION_SLUG = "biohub-cell-tracking-during-development"
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
OUTPUT_PATH = Path("input_metadata_audit.json")

# %% [markdown]
# ## 2. Metadata helpers


# %%
def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def nested_value(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        if key in value:
            return value[key]
        for child in value.values():
            found = nested_value(child, key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = nested_value(child, key)
            if found is not None:
                return found
    return None


def array_metadata(path: Path) -> dict[str, Any] | None:
    metadata = read_json(path / "zarr.json")
    if metadata is None:
        return None
    chunk_grid = metadata.get("chunk_grid", {})
    chunk_configuration = (
        chunk_grid.get("configuration", {}) if isinstance(chunk_grid, dict) else {}
    )
    return {
        "shape": metadata.get("shape"),
        "data_type": metadata.get("data_type", metadata.get("dtype")),
        "chunk_shape": chunk_configuration.get("chunk_shape"),
    }


def first_dimension(metadata: dict[str, Any] | None) -> int | None:
    if not metadata:
        return None
    shape = metadata.get("shape")
    if isinstance(shape, list) and shape and isinstance(shape[0], int):
        return shape[0]
    return None


def numeric_summary(values: list[int]) -> dict[str, int | float | None]:
    if not values:
        return {"count": 0, "sum": None, "min": None, "median": None, "max": None}
    return {
        "count": len(values),
        "sum": sum(values),
        "min": min(values),
        "median": median(values),
        "max": max(values),
    }


def shape_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        shape = record.get("image_metadata", {}).get("shape")
        key = "missing" if shape is None else "x".join(str(value) for value in shape)
        counts[key] += 1
    return dict(sorted(counts.items()))


def embryo_id(sample_id: str) -> str:
    return sample_id.split("_", maxsplit=1)[0]


def find_competition_root() -> Path:
    direct = KAGGLE_INPUT_ROOT / COMPETITION_SLUG
    if direct.is_dir():
        return direct
    matches = sorted(
        path for path in KAGGLE_INPUT_ROOT.glob(f"**/{COMPETITION_SLUG}") if path.is_dir()
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one competition input root for {COMPETITION_SLUG}, found {matches}"
        )
    return matches[0]


# %% [markdown]
# ## 3. Competition input checks

# %%
competition_root = find_competition_root()
train_root = competition_root / "train"
test_root = competition_root / "test"
sample_submission_path = competition_root / "sample_submission.csv"

for required_path in (train_root, test_root, sample_submission_path):
    if not required_path.exists():
        raise FileNotFoundError(f"required competition input is missing: {required_path}")

print(f"competition_root={competition_root}", flush=True)

# %% [markdown]
# ## 4. Train and test inventory

# %%
train_images = {
    path.name.removesuffix(".zarr"): path
    for path in train_root.iterdir()
    if path.is_dir() and path.name.endswith(".zarr")
}
train_graphs = {
    path.name.removesuffix(".geff"): path
    for path in train_root.iterdir()
    if path.is_dir() and path.name.endswith(".geff")
}
test_images = {
    path.name.removesuffix(".zarr"): path
    for path in test_root.iterdir()
    if path.is_dir() and path.name.endswith(".zarr")
}

train_records: list[dict[str, Any]] = []
for sample_id in sorted(set(train_images) | set(train_graphs)):
    image_path = train_images.get(sample_id)
    graph_path = train_graphs.get(sample_id)
    graph_root_metadata = read_json(graph_path / "zarr.json") if graph_path else None
    node_metadata = array_metadata(graph_path / "nodes" / "ids") if graph_path else None
    edge_metadata = array_metadata(graph_path / "edges" / "ids") if graph_path else None
    train_records.append(
        {
            "sample_id": sample_id,
            "embryo_id": embryo_id(sample_id),
            "has_image": image_path is not None,
            "has_graph": graph_path is not None,
            "image_metadata": array_metadata(image_path / "0") if image_path else None,
            "node_count": first_dimension(node_metadata),
            "edge_count": first_dimension(edge_metadata),
            "estimated_number_of_nodes": (
                nested_value(graph_root_metadata, "estimated_number_of_nodes")
                if graph_root_metadata
                else None
            ),
        }
    )

test_records = [
    {
        "sample_id": sample_id,
        "embryo_id": embryo_id(sample_id),
        "image_metadata": array_metadata(path / "0"),
    }
    for sample_id, path in sorted(test_images.items())
]

samples_per_embryo = Counter(record["embryo_id"] for record in train_records)
node_counts = [
    record["node_count"] for record in train_records if isinstance(record["node_count"], int)
]
edge_counts = [
    record["edge_count"] for record in train_records if isinstance(record["edge_count"], int)
]
estimated_node_counts = [
    record["estimated_number_of_nodes"]
    for record in train_records
    if isinstance(record["estimated_number_of_nodes"], int)
]

inventory_summary = {
    "train_image_count": len(train_images),
    "train_graph_count": len(train_graphs),
    "paired_train_sample_count": len(set(train_images) & set(train_graphs)),
    "train_images_without_graph": sorted(set(train_images) - set(train_graphs)),
    "train_graphs_without_image": sorted(set(train_graphs) - set(train_images)),
    "train_embryo_count": len(samples_per_embryo),
    "train_samples_per_embryo": dict(sorted(samples_per_embryo.items())),
    "train_image_shape_counts": shape_counts(train_records),
    "train_node_count_summary": numeric_summary(node_counts),
    "train_edge_count_summary": numeric_summary(edge_counts),
    "estimated_node_count_summary": numeric_summary(estimated_node_counts),
    "test_image_count": len(test_images),
    "test_image_shape_counts": shape_counts(test_records),
    "test_sample_ids": sorted(test_images),
}

# %% [markdown]
# ## 5. Sample submission audit

# %%
sample_submission = pd.read_csv(sample_submission_path)
sample_submission_summary = {
    "file_size_bytes": sample_submission_path.stat().st_size,
    "row_count": len(sample_submission),
    "columns": list(sample_submission.columns),
    "dtypes": {column: str(dtype) for column, dtype in sample_submission.dtypes.items()},
    "row_type_counts": {
        str(key): int(value)
        for key, value in sample_submission["row_type"].value_counts(dropna=False).items()
    },
    "dataset_counts": {
        str(key): int(value)
        for key, value in sample_submission["dataset"].value_counts(dropna=False).items()
    },
    "preview": json.loads(sample_submission.head(10).to_json(orient="records")),
}

# %% [markdown]
# ## 6. Audit output

# %%
audit = {
    "schema_version": 1,
    "captured_at_utc": datetime.now(UTC).isoformat(),
    "competition_slug": COMPETITION_SLUG,
    "inventory_summary": inventory_summary,
    "sample_submission": sample_submission_summary,
    "train_samples": train_records,
    "test_samples": test_records,
}

OUTPUT_PATH.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")

print("INPUT_AUDIT_SUMMARY_START", flush=True)
print(
    json.dumps(
        {
            "inventory_summary": inventory_summary,
            "sample_submission": sample_submission_summary,
            "output_path": str(OUTPUT_PATH),
        },
        indent=2,
        ensure_ascii=False,
    ),
    flush=True,
)
print("INPUT_AUDIT_SUMMARY_END", flush=True)
