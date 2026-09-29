# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # exp053: train a four-frame score for adding a second daughter

# %% [markdown]
# ## Contents
# 1. Imports and runtime checks
# 2. Fixed video split and GEFF graph reading
# 3. Division candidate and partial-label generation
# 4. Fit four-frame and context-masked controls
# 5. Internal threshold, outer diagnostics, and model artifact

# %% [markdown]
# ## 1. Imports and runtime checks

# %%
import hashlib
import importlib
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from division_features import (
    FEATURE_NAMES,
    build_edge_index,
    feature_vector,
    mask_context,
    position_um,
)
from scipy.spatial import cKDTree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

START = time.time()
WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
COMPETITION = "biohub-cell-tracking-during-development"
COMP = next(
    (
        p
        for p in (INPUT / "competitions" / COMPETITION, INPUT / COMPETITION)
        if (p / "train").is_dir() and (p / "sample_submission.csv").exists()
    ),
    None,
)
if COMP is None:
    raise FileNotFoundError("competition train input is unavailable")
TRAIN = COMP / "train"
print("competition input:", COMP)

# %%
try:
    import zarr

    if int(str(zarr.__version__).split(".", 1)[0]) < 3:
        raise ImportError("zarr v3 required")
except (ImportError, AttributeError):
    support_roots = (
        INPUT / "biohub-tracking-support-pack-50ep-v1",
        INPUT / "datasets" / "pilkwang" / "biohub-tracking-support-pack-50ep-v1",
    )
    wheel_dirs = sorted(
        {
            wheel.parent
            for root in support_roots
            if root.is_dir()
            for wheel in root.rglob("zarr-*.whl")
        }
    )
    if not wheel_dirs:
        raise ImportError("zarr v3 offline wheels unavailable") from None
    command = [sys.executable, "-m", "pip", "install", "--no-index"]
    for directory in wheel_dirs:
        command += ["--find-links", str(directory)]
    command += ["zarr>=3.0.10,<4"]
    subprocess.run(command, check=True)
    importlib.invalidate_caches()
    import zarr

# %% [markdown]
# ## 2. Fixed video split and GEFF graph reading


# %%
def rank(stem):
    return hashlib.sha256(f"42:{stem}".encode()).hexdigest(), stem


stems = sorted(p.stem for p in TRAIN.glob("*.geff") if p.is_dir())
splits = {"fit": [], "inner": [], "outer": []}
for embryo in ("44b6", "6bba"):
    ranked = sorted((s for s in stems if s.startswith(embryo + "_")), key=rank)
    if len(ranked) < 35:
        raise RuntimeError(f"{embryo}: fewer than 35 labeled videos")
    splits["outer"] += ranked[20:30]
    splits["inner"] += ranked[30:35]
    splits["fit"] += ranked[:20] + ranked[35:]
if len(set(sum(splits.values(), []))) != len(stems):
    raise RuntimeError("video split overlap or incomplete split")
print({key: Counter(s[:4] for s in values) for key, values in splits.items()})


def read_geff(stem):
    graph = zarr.open_group(str(TRAIN / f"{stem}.geff"), mode="r")
    ids = np.asarray(graph["nodes/ids"][:], dtype=np.int64)
    coordinates = {
        axis: np.asarray(graph[f"nodes/props/{axis}/values"][:]) for axis in ("t", "z", "y", "x")
    }
    if any(len(values) != len(ids) for values in coordinates.values()):
        raise ValueError(f"{stem}: node property mismatch")
    nodes = {
        int(node_id): {axis: float(values[i]) for axis, values in coordinates.items()}
        for i, node_id in enumerate(ids)
    }
    edges = np.asarray(graph["edges/ids"][:], dtype=np.int64).reshape(-1, 2)
    return nodes, edges


# %% [markdown]
# ## 3. Division candidates and partial labels

# %%
PARENT_MAX_UM = 9.0
SISTER_MAX_UM = 14.0
FIRST_MAX_UM = 10.0
MAX_NEGATIVES = 8


def labeled_candidates(stem):
    nodes, raw_edges = read_geff(stem)
    edges = np.asarray(
        [
            (int(a), int(b))
            for a, b in raw_edges
            if a in nodes and b in nodes and int(nodes[int(b)]["t"]) == int(nodes[int(a)]["t"]) + 1
        ],
        dtype=np.int64,
    ).reshape(-1, 2)
    incoming, outgoing = build_edge_index(edges)
    frame_ids = {}
    for node_id, node in nodes.items():
        frame_ids.setdefault(int(node["t"]), []).append(node_id)
    trees = {}
    for t, ids in frame_ids.items():
        ids.sort()
        trees[t] = (ids, cKDTree(np.stack([position_um(nodes[node_id]) for node_id in ids])))
    features = []
    labels = []
    counts = Counter()
    for mother_id, daughters in outgoing.items():
        if mother_id not in nodes or len(daughters) not in (1, 2):
            continue
        t = int(nodes[mother_id]["t"])
        if t + 1 not in trees:
            continue
        daughters = [
            child for child in daughters if child in nodes and int(nodes[child]["t"]) == t + 1
        ]
        if len(daughters) not in (1, 2):
            continue
        nearby_ids, tree = trees[t + 1]
        mother_pos = position_um(nodes[mother_id])
        for first_id in daughters:
            first_pos = position_um(nodes[first_id])
            if np.linalg.norm(first_pos - mother_pos) > FIRST_MAX_UM:
                continue
            options = []
            for index in tree.query_ball_point(mother_pos, r=PARENT_MAX_UM):
                second_id = nearby_ids[int(index)]
                if second_id == first_id:
                    continue
                if np.linalg.norm(position_um(nodes[second_id]) - first_pos) > SISTER_MAX_UM:
                    continue
                known_parent = incoming.get(second_id)
                if known_parent is None:
                    counts["unknown_second_parent"] += 1
                    continue
                label = int(known_parent == mother_id)
                options.append(
                    (
                        label,
                        float(np.linalg.norm(position_um(nodes[second_id]) - mother_pos)),
                        second_id,
                    )
                )
            positives = [item for item in options if item[0] == 1]
            negatives = sorted(
                (item for item in options if item[0] == 0), key=lambda item: (item[1], item[2])
            )[:MAX_NEGATIVES]
            for label, _, second_id in positives + negatives:
                features.append(
                    feature_vector(nodes, incoming, outgoing, mother_id, first_id, second_id)
                )
                labels.append(label)
                counts["positive" if label else "negative"] += 1
    if features:
        return np.stack(features), np.asarray(labels, dtype=np.int8), counts
    return np.empty((0, len(FEATURE_NAMES)), np.float32), np.empty(0, np.int8), counts


sets = {}
video_counts = {}
for part, selected in splits.items():
    features, labels, video_ids = [], [], []
    for stem in selected:
        x, y, counts = labeled_candidates(stem)
        video_counts[stem] = dict(counts)
        features.append(x)
        labels.append(y)
        video_ids.extend([stem] * len(y))
    sets[part] = (
        np.concatenate(features),
        np.concatenate(labels),
        np.asarray(video_ids),
    )
    print(part, "candidate rows", len(sets[part][1]), "known positives", int(sets[part][1].sum()))
if len(np.unique(sets["fit"][1])) != 2:
    raise RuntimeError("fit lacks positive or explicit negative division candidates")
for embryo in ("44b6", "6bba"):
    embryo_rows = np.array([s.startswith(embryo + "_") for s in sets["fit"][2]])
    if not np.any(sets["fit"][1][embryo_rows] == 1) or not np.any(sets["fit"][1][embryo_rows] == 0):
        raise RuntimeError(f"{embryo}: insufficient known labels")

table_path = WORK / "division_training_table.npz"
np.savez_compressed(
    table_path,
    fit_x=sets["fit"][0],
    fit_y=sets["fit"][1],
    fit_video=sets["fit"][2],
    inner_x=sets["inner"][0],
    inner_y=sets["inner"][1],
    inner_video=sets["inner"][2],
    outer_x=sets["outer"][0],
    outer_y=sets["outer"][1],
    outer_video=sets["outer"][2],
)
print("training table SHA256", hashlib.sha256(table_path.read_bytes()).hexdigest())


# %% [markdown]
# ## 4. Fit four-frame and context-masked controls

# %%
variants = {}
for name, transform in (("four_frame", lambda x: x), ("context_masked", mask_context)):
    x_fit, y_fit, _ = sets["fit"]
    scaler = StandardScaler().fit(transform(x_fit))
    model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42).fit(
        scaler.transform(transform(x_fit)), y_fit
    )
    variants[name] = (scaler, model, transform)
    print(
        name,
        "converged",
        int(model.n_iter_[0]),
        "coef finite",
        bool(np.isfinite(model.coef_).all()),
    )


def predict(part, name):
    scaler, model, transform = variants[name]
    x, _, _ = sets[part]
    return model.predict_proba(scaler.transform(transform(x)))[:, 1]


# %% [markdown]
# ## 5. Internal threshold, outer diagnostics, and model artifact

# %%
inner_y = sets["inner"][1]
inner_score = predict("inner", "four_frame")
if not np.any(inner_y == 1) or not np.any(inner_y == 0):
    raise RuntimeError("inner split lacks positive or explicit negative candidates")
thresholds = np.unique(np.r_[0.5, inner_score])
threshold_rows = []
for threshold in thresholds:
    pred = inner_score >= threshold
    precision, recall, fbeta, _ = precision_recall_fscore_support(
        inner_y, pred, beta=0.5, average="binary", zero_division=0
    )
    threshold_rows.append((float(precision), float(fbeta), float(recall), float(threshold)))
eligible = [row for row in threshold_rows if row[0] >= 0.8]
if not eligible:
    raise RuntimeError("no inner threshold achieves precision >= 0.8")
selected = max(eligible, key=lambda row: (row[1], row[2], row[3]))
threshold = selected[3]
print("selected threshold:", threshold, "inner precision/f0.5/recall:", selected[:3])

diagnostics = {}
for part in ("inner", "outer"):
    y = sets[part][1]
    ids = sets[part][2]
    diagnostics[part] = {}
    for embryo in ("44b6", "6bba"):
        mask = np.array([stem.startswith(embryo + "_") for stem in ids])
        if not mask.any():
            continue
        diagnostics[part][embryo] = {}
        for name in variants:
            score = predict(part, name)[mask]
            truth = y[mask]
            pred = score >= threshold
            precision, recall, fbeta, _ = precision_recall_fscore_support(
                truth, pred, beta=0.5, average="binary", zero_division=0
            )
            diagnostics[part][embryo][name] = {
                "rows": int(mask.sum()),
                "positives": int(truth.sum()),
                "average_precision": float(average_precision_score(truth, score))
                if truth.sum()
                else None,
                "precision_at_four_frame_threshold": float(precision),
                "recall_at_four_frame_threshold": float(recall),
                "f0_5_at_four_frame_threshold": float(fbeta),
                "predicted_positive": int(pred.sum()),
            }
print(json.dumps(diagnostics, indent=2, sort_keys=True))

scaler, model, _ = variants["four_frame"]
artifact = {
    "experiment": "exp053_x138_postlink_division_score",
    "selected_variant": "four_frame",
    "feature_names": FEATURE_NAMES,
    "mean": scaler.mean_.tolist(),
    "scale": scaler.scale_.tolist(),
    "coef": model.coef_[0].tolist(),
    "intercept": float(model.intercept_[0]),
    "threshold": threshold,
    "selection_seed": 42,
    "split": splits,
    "diagnostics": diagnostics,
    "video_counts": video_counts,
    "training_seconds": time.time() - START,
}
model_path = WORK / "division_model.json"
model_path.write_text(json.dumps(artifact, sort_keys=True, separators=(",", ":")))
receipt = {
    "experiment": artifact["experiment"],
    "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
    "feature_schema_sha256": hashlib.sha256(
        json.dumps(FEATURE_NAMES, separators=(",", ":")).encode()
    ).hexdigest(),
    "candidate_counts": {
        part: {"rows": len(sets[part][1]), "positives": int(sets[part][1].sum())} for part in sets
    },
    "threshold": threshold,
    "elapsed_seconds": time.time() - START,
}
(WORK / "train_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True))
print("EXP053_TRAIN_COMPLETE", json.dumps(receipt, sort_keys=True))
