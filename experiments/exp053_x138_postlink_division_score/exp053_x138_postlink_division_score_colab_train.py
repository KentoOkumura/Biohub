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
# # exp053: Colab CLI fit of the four-frame division scorer

# %% [markdown]
# ## Contents
# 1. Imports and verified input
# 2. Fit four-frame and context-masked controls
# 3. Select the threshold on inner videos
# 4. Evaluate fixed outer videos and export coefficients

# %% [markdown]
# ## 1. Imports and verified input

# %%
import hashlib
import json
import platform
import time
import zipfile
from pathlib import Path

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

START = time.time()
SOURCE_PATH = next(
    (
        path
        for path in (
            Path("/division_training_table.npz"),
            Path("/content/division_training_table.npz"),
        )
        if path.exists()
    ),
    None,
)
if SOURCE_PATH is None:
    raise FileNotFoundError("verified division_training_table.npz is missing")
SOURCE_SHA256 = hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest()
OUTPUT_DIR = Path("/content/exp053_colab_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_NAMES = (
    "mother_to_first_um",
    "mother_to_second_um",
    "sister_distance_um",
    "daughter_distance_asymmetry_um",
    "daughter_displacement_cosine",
    "predecessor_present",
    "predecessor_step_um",
    "predecessor_to_first_cosine",
    "predecessor_to_second_cosine",
    "predecessor_extrapolation_first_um",
    "predecessor_extrapolation_second_um",
    "first_successor_present",
    "second_successor_present",
    "first_successor_step_um",
    "second_successor_step_um",
    "first_straightness_cosine",
    "second_straightness_cosine",
    "next_sister_distance_um",
    "daughter_divergence_um",
)

table = np.load(SOURCE_PATH, allow_pickle=False)
sets = {
    part: (table[f"{part}_x"], table[f"{part}_y"], table[f"{part}_video"])
    for part in ("fit", "inner", "outer")
}
for part, (features, labels, videos) in sets.items():
    if (
        features.ndim != 2
        or features.shape[1] != len(FEATURE_NAMES)
        or len(features) != len(labels)
        or len(labels) != len(videos)
        or not np.isfinite(features).all()
    ):
        raise ValueError(f"invalid {part} training table")
    print(part, "rows", len(labels), "known positive", int(labels.sum()))
if len(np.unique(sets["fit"][1])) != 2:
    raise ValueError("fit lacks both labels")
print("input SHA256:", SOURCE_SHA256)

# %% [markdown]
# ## 2. Fit four-frame and context-masked controls


# %%
def mask_context(features):
    masked = features.copy()
    masked[:, 5:] = 0.0
    return masked


variants = {}
for name, transform in (("four_frame", lambda x: x), ("context_masked", mask_context)):
    x_fit, y_fit, _ = sets["fit"]
    scaler = StandardScaler().fit(transform(x_fit))
    model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42).fit(
        scaler.transform(transform(x_fit)), y_fit
    )
    if not np.isfinite(model.coef_).all():
        raise ValueError(f"{name} has non-finite coefficients")
    variants[name] = (scaler, model, transform)
    print(name, "iterations", int(model.n_iter_[0]))


def predict(part, name):
    scaler, model, transform = variants[name]
    return model.predict_proba(scaler.transform(transform(sets[part][0])))[:, 1]


# %% [markdown]
# ## 3. Select the threshold on inner videos

# %%
inner_y = sets["inner"][1]
inner_score = predict("inner", "four_frame")
if not np.any(inner_y == 1) or not np.any(inner_y == 0):
    raise ValueError("inner split lacks both labels")
threshold_rows = []
for threshold in np.unique(np.r_[0.5, inner_score]):
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
print("threshold", threshold, "inner precision/f0.5/recall", selected[:3])

# %% [markdown]
# ## 4. Evaluate fixed outer videos and export coefficients

# %%
diagnostics = {}
for part in ("inner", "outer"):
    y, ids = sets[part][1:]
    diagnostics[part] = {}
    for embryo in ("44b6", "6bba"):
        mask = np.char.startswith(ids.astype(str), embryo + "_")
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
print(json.dumps(diagnostics, sort_keys=True, indent=2))

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
    "input_table_sha256": SOURCE_SHA256,
    "diagnostics": diagnostics,
}
model_path = OUTPUT_DIR / "division_model.json"
model_path.write_text(json.dumps(artifact, sort_keys=True, separators=(",", ":")))
receipt = {
    "experiment": artifact["experiment"],
    "source_table_sha256": SOURCE_SHA256,
    "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
    "feature_schema_sha256": hashlib.sha256(
        json.dumps(FEATURE_NAMES, separators=(",", ":")).encode()
    ).hexdigest(),
    "candidate_counts": {
        part: {"rows": len(sets[part][1]), "positives": int(sets[part][1].sum())} for part in sets
    },
    "threshold": threshold,
    "elapsed_seconds": time.time() - START,
    "python_version": platform.python_version(),
    "numpy_version": np.__version__,
    "sklearn_version": sklearn.__version__,
}
receipt_path = OUTPUT_DIR / "colab_train_receipt.json"
receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2))
archive_path = Path("/content/exp053_colab_output.zip")
with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    archive.write(model_path, model_path.name)
    archive.write(receipt_path, receipt_path.name)
print("EXP053_COLAB_TRAIN_COMPLETE", json.dumps(receipt, sort_keys=True), flush=True)
print("OUTPUT_ARCHIVE", archive_path, archive_path.stat().st_size, flush=True)
