"""Derive the frozen exp050 Kaggle submission notebooks from validated sources."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAME = ROOT.name
EVALUATION = (ROOT / f"{NAME}_inference.py").read_text()
CAPTURE = (ROOT / f"{NAME}_capture0.py").read_text()


def section(source: str, start: str, end: str) -> str:
    assert source.count(start) == 1, start
    assert source.count(end) == 1, end
    return source[source.index(start) : source.index(end)]


PREFIX = section(EVALUATION, "# %%\n# ruff: noqa\n", "# Label-blind split.")
PREDICTOR_PATCH = section(
    EVALUATION,
    "predict_cmd = [\n",
    "# %% [markdown]\n# ## 4. Original graph postprocessing and output checks\n",
)
FEATURE_PATCH = section(
    CAPTURE,
    '# %%\n# ruff: noqa: E501\n"""Patch the materialized x138 predictor',
    "# %% [markdown]\n# ## 4. Capture fixed image and tracker features",
)
POSTPROCESS = section(
    EVALUATION,
    "# %% [markdown]\n# ## 4. Original graph postprocessing and output checks\n",
    "# %% [markdown]\n# ## Shared edge catalog and exact set ILP\n",
)
LEARNED = section(
    EVALUATION,
    "# %% [markdown]\n# ## Shared edge catalog and exact set ILP\n",
    "# %% [markdown]\n# ## 5. Verify train cache and the fixed inner selection\n",
)
CONFIG_PART = section(
    EVALUATION,
    "CONFIG = json.loads(\n",
    'EXP050_ROOT = WORKING_DIR / "exp050_inference"',
)
LEGACY = section(
    EVALUATION,
    "# %% [markdown]\n# ## 6. Exact original-cost ILP and fixed postprocessing\n",
    "def _audit_save_official_graph(",
)

COMMON_SETUP = """# %% [markdown]
# ## Test inputs and frozen training split

# %%
TRAIN_DIR = COMP_DIR / "train"
test_stems = list_test_stems()
if any(stem.split("_", 1)[0] not in {"44b6", "6bba"} for stem in test_stems):
    raise RuntimeError("Unknown embryo prefix; no frozen cross-embryo model or costs")
_selection_sha = "20a67ce13cd208430805b9c6221e1f6e21201b469a3938aae58a33a52e2cdcf3"
splits_path = REPO_DIR / "kaggle_test_splits_50ep.json"
splits_path.parent.mkdir(parents=True, exist_ok=True)
splits_path.write_text(json.dumps([{"split": 0, "train": [], "test": test_stems}], indent=2))
print("Test videos:", test_stems, flush=True)

"""

CAPTURE_RUN = """# %% [markdown]
# ## Capture fixed x138 node and candidate features on the current test set

# %%
_ps.write_text(add_feature_capture(_ps.read_text()))
print("exp050 feature capture installed", flush=True)

# %%
EXP050_ROOT = WORKING_DIR / f"exp050_submission_{VARIANT}"
"""
CAPTURE_RUN += """
CACHE_ROOT = EXP050_ROOT / "cache"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
if any(CACHE_ROOT.iterdir()):
    raise RuntimeError(f"Refusing to reuse a nonempty feature cache: {CACHE_ROOT}")
env = {
    **os.environ,
    "PYTHONPATH": "src",
    "V1284_MODE": "candidate",
    "V1284_HEAD": str(_SELF_HEAD_PATH),
    "BIOHUB_CACHE_DIR": str(CACHE_ROOT),
    "BIOHUB_CACHE_EDGE_THRESHOLD": "0.10",
    "BIOHUB_SKIP_CACHE_ILP": "1",
    "BIOHUB_DIAGNOSTIC_ARM": f"exp050_submission_{VARIANT}",
}


def _wait_for_feature_capture_shards(processes, commands, deadline):
    active = dict(processes)
    try:
        while active:
            for shard_index, process in list(active.items()):
                return_code = process.poll()
                if return_code is None:
                    continue
                del active[shard_index]
                if return_code != 0:
                    raise subprocess.CalledProcessError(return_code, commands[shard_index])
            if active:
                if time.monotonic() >= deadline:
                    raise subprocess.TimeoutExpired(commands[min(active)], 11.5 * 3600)
                time.sleep(1.0)
    except BaseException:
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        for process in processes.values():
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise


def _merge_feature_capture_shards(cache_root, shard_roots, stems):
    expected_all = set(stems)
    if len(expected_all) != len(stems):
        raise RuntimeError("Duplicate test video names")
    if any(cache_root.iterdir()):
        raise RuntimeError(f"Refusing to overwrite feature cache: {cache_root}")
    shard_files = []
    seen = set()
    for shard_index, shard_root in enumerate(shard_roots):
        paths = sorted(shard_root.iterdir())
        if any(not path.is_file() or path.suffix != ".npz" for path in paths):
            raise RuntimeError(f"Unexpected output in GPU shard cache: {shard_root}")
        found = {path.stem for path in paths}
        expected = set(stems[shard_index::len(shard_roots)])
        if found != expected or len(paths) != len(expected):
            raise RuntimeError(
                f"GPU shard {shard_index} feature mismatch: "
                f"missing={sorted(expected - found)}, extra={sorted(found - expected)}"
            )
        if seen & found:
            raise RuntimeError(f"Duplicate feature cache videos: {sorted(seen & found)}")
        seen.update(found)
        shard_files.append(paths)
    if seen != expected_all:
        raise RuntimeError(f"Feature cache coverage mismatch: {sorted(expected_all - seen)}")

    staging_cache = cache_root.with_name("cache_merge_staging")
    if staging_cache.exists():
        raise RuntimeError(f"Refusing to reuse staged feature cache: {staging_cache}")
    staging_cache.mkdir()
    for paths in shard_files:
        for path in paths:
            path.replace(staging_cache / path.name)
    cache_root.rmdir()
    staging_cache.replace(cache_root)
    for shard_root in shard_roots:
        shard_root.rmdir()


capture_start = time.monotonic()
capture_workers = min(2, _torch.cuda.device_count(), len(test_stems))
if capture_workers >= 2 and not SLICE:
    cuda_tokens = _visible_cuda_tokens(capture_workers)
    processes = {}
    commands = {}
    shard_roots = []
    try:
        print(
            f"Launching {capture_workers} independent feature-capture shards "
            f"on CUDA devices {cuda_tokens}",
            flush=True,
        )
        for shard_index in range(capture_workers):
            shard_root = EXP050_ROOT / f"cache_gpu{shard_index}"
            shard_root.mkdir(parents=True, exist_ok=True)
            if any(shard_root.iterdir()):
                raise RuntimeError(f"Refusing to reuse a nonempty GPU shard cache: {shard_root}")
            shard_roots.append(shard_root)
            shard_command = [
                *predict_cmd,
                "--method", f"exp050_submission_{VARIANT}_gpu{shard_index}",
                "--slice", f"{shard_index}::{capture_workers}",
            ]
            shard_env = {
                **env,
                "CUDA_VISIBLE_DEVICES": cuda_tokens[shard_index],
                "BIOHUB_CACHE_DIR": str(shard_root),
                "BIOHUB_GPU_SHARD": f"{shard_index}/{capture_workers}",
            }
            print(f"GPU shard {shard_index}: {' '.join(shard_command)}", flush=True)
            commands[shard_index] = shard_command
            processes[shard_index] = subprocess.Popen(
                shard_command, cwd=REPO_DIR, env=shard_env,
            )
        _wait_for_feature_capture_shards(processes, commands, capture_start + 11.5 * 3600)
    except BaseException:
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        for process in processes.values():
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise
    _merge_feature_capture_shards(CACHE_ROOT, shard_roots, test_stems)
else:
    reason = "SLICE is active" if SLICE else f"only {capture_workers} CUDA device(s) available"
    print(f"Using single-process feature capture because {reason}.", flush=True)
    command = [*predict_cmd, "--method", f"exp050_submission_{VARIANT}"]
    subprocess.run(command, cwd=REPO_DIR, env=env, check=True, timeout=11.5 * 3600)
if {p.stem for p in CACHE_ROOT.glob("*.npz")} != set(test_stems):
    raise RuntimeError("Incomplete public or hidden test feature capture")
capture_elapsed_seconds = time.monotonic() - capture_start
print("Captured", len(test_stems), "videos in", capture_elapsed_seconds, "s", flush=True)

"""

MODEL_SETUP = """# %% [markdown]
# ## Verify frozen training output and select the cross-embryo fold

# %%
import hashlib
from pathlib import Path

CONFIG_PART_PLACEHOLDER
_train_receipts = sorted(Path("/kaggle/input").rglob("train_receipt.json"))
if len(_train_receipts) != 1:
    raise RuntimeError(f"Expected one frozen train receipt: {_train_receipts}")
TRAIN_OUTPUT = _train_receipts[0].parent
train_receipt = json.loads(_train_receipts[0].read_text())
if (train_receipt["selection_sha256"] != _selection_sha
        or train_receipt["config_sha256"] != CONFIG_SHA256
        or len(train_receipt["folds"]) != 2):
    raise RuntimeError("Frozen training receipt differs from exp050 contract")
REPAIR_DEADLINE_S = float("inf")
AUDIT_ARM = None

""".replace("CONFIG_PART_PLACEHOLDER", CONFIG_PART.rstrip())

GNN_SETUP = """models = {}
for fold in (0, 1):
    path = TRAIN_OUTPUT / f"fold_{fold}.pt"
    if hashlib.sha256(path.read_bytes()).hexdigest() != train_receipt["folds"][fold]["sha256"]:
        raise RuntimeError(f"fold {fold}: model SHA mismatch")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["selection_sha256"] != _selection_sha or saved["config_sha256"] != CONFIG_SHA256:
        raise RuntimeError(f"fold {fold}: model split/config mismatch")
    model = SharedEdgeGNN(
        saved["feature_width"],
        hidden=int(CONFIG["model"]["params"]["hidden_dim"]),
        dropout=float(CONFIG["model"]["params"]["dropout"]),
    )
    model.load_state_dict(saved["state_dict"])
    model.eval()
    if not torch.cuda.is_available():
        raise RuntimeError("exp050 submission expects Kaggle T4")
    models[fold] = model.to("cuda")


def score_submission_gnn_sets(models, embryo, coords, features, edges, *, device="cuda"):
    if embryo == "44b6":
        model_folds = (1,)
    elif embryo == "6bba":
        model_folds = (0,)
    else:
        model_folds = (0, 1)
    catalog = by_parent = scores = None
    for model_fold in model_folds:
        next_catalog, next_by_parent, next_scores, _ = score_video_sets(
            models[model_fold], coords, features, edges, device=device,
        )
        if catalog is None:
            catalog, by_parent, scores = next_catalog, next_by_parent, next_scores
        else:
            if next_catalog != catalog or next_by_parent != by_parent:
                raise RuntimeError("GNN folds produced different candidate set catalogs")
            scores = 0.5 * scores + 0.5 * next_scores
    if not np.isfinite(scores).all():
        raise RuntimeError("Nonfinite averaged GNN set score")
    return catalog, by_parent, scores, model_folds

"""

OPTUNA_SETUP = """_tune_receipts = sorted(Path("/kaggle/input").rglob("tune_receipt.json"))
if len(_tune_receipts) != 1:
    raise RuntimeError(f"Expected one frozen tune receipt: {_tune_receipts}")
tune_receipt = json.loads(_tune_receipts[0].read_text())
if (tune_receipt["selection_sha256"] != _selection_sha
        or tune_receipt["config_sha256"] != CONFIG_SHA256
        or tune_receipt["train_receipt_sha256"]
        != hashlib.sha256(_train_receipts[0].read_bytes()).hexdigest()):
    raise RuntimeError("Optuna receipt differs from frozen training run")
if set(tune_receipt["folds"]) != {"0", "1"}:
    raise RuntimeError("Missing tuned fold")
print("Frozen selected trials:",
      {fold: tune_receipt["folds"][str(fold)]["selected_trial"] for fold in (0, 1)},
      flush=True)

"""

SUBMISSION = """# %% [markdown]
# ## Solve one frozen method per test video and write competition CSV

# %%
import csv

CSV_COLUMNS = ["id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]
SUBMISSION_PATH = WORKING_DIR / "submission.csv"
video_rows = {}
row_id = 0
run_start = time.monotonic()
with SUBMISSION_PATH.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    for stem in test_stems:
        embryo = stem.split("_", 1)[0]
        fold = 1 if embryo == "44b6" else 0
        with np.load(CACHE_ROOT / f"{stem}.npz", allow_pickle=False) as data:
            coords = np.asarray(data["coords"], dtype=np.float32)
            features = np.asarray(data["node_features"], dtype=np.float32)
            baseline = np.asarray(data["admitted"], dtype=np.float64).reshape(-1, 4)
            expanded = select_expanded_edges(
                coords, data["edge_src"], data["edge_tgt"], data["edge_prob"], baseline,
            )
        if features.shape != (len(coords), 65) or not np.isfinite(features).all():
            raise RuntimeError(f"{stem}: invalid frozen feature width or values")
        METHOD_SOLVE_PLACEHOLDER
        os.environ["BIOHUB_CACHE_DIR"] = str(CACHE_ROOT)
        final_nodes, final_edges, stats = filter_output_graph(
            nodes, selected, dataset=stem, deepcenter_bundle=DEEPCENTER_VETO_DETECTOR,
        )
        if _deadline_degraded or not final_nodes:
            raise RuntimeError(f"{stem}: fixed postprocessing degraded or empty")
        for node_id in sorted(final_nodes):
            node = final_nodes[node_id]
            writer.writerow({
                "id": row_id, "dataset": stem, "row_type": "node", "node_id": int(node_id),
                "t": int(node["t"]),
                "z": max(0, int(round(float(node["z"])))),
                "y": max(0, int(round(float(node["y"])))),
                "x": max(0, int(round(float(node["x"])))),
                "source_id": -1, "target_id": -1,
            })
            row_id += 1
        for edge in final_edges:
            source, target = int(edge["source_id"]), int(edge["target_id"])
            if source not in final_nodes or target not in final_nodes:
                raise RuntimeError(f"{stem}: dangling final edge")
            writer.writerow({
                "id": row_id, "dataset": stem, "row_type": "edge", "node_id": -1,
                "t": -1, "z": -1, "y": -1, "x": -1,
                "source_id": source, "target_id": target,
            })
            row_id += 1
        video_rows[stem] = {
            "fold": fold, "candidate_edges": len(expanded), "solve": solve_info,
            "postprocess": stats, "nodes": len(final_nodes), "edges": len(final_edges),
        }
        print(stem, VARIANT, "rows", row_id, "solve", solve_info, flush=True)
        if time.monotonic() - run_start > 11.5 * 3600:
            raise RuntimeError("Approaching Kaggle notebook 12h limit")

# %% [markdown]
# ## Validate all datasets and graph constraints before publishing

# %%
frame = pd.read_csv(SUBMISSION_PATH)
if frame.empty or frame.columns.tolist() != CSV_COLUMNS:
    raise RuntimeError("Submission schema or row count invalid")
if frame["id"].tolist() != list(range(len(frame))):
    raise RuntimeError("Submission IDs are not contiguous")
if set(frame.dataset.astype(str)) != set(test_stems):
    raise RuntimeError("Public or hidden test video coverage mismatch")
if frame.isna().any().any():
    raise RuntimeError("Submission contains missing values")
for stem, group in frame.groupby("dataset"):
    nodes = group[group.row_type.eq("node")]
    edges = group[group.row_type.eq("edge")]
    if nodes.node_id.duplicated().any():
        raise RuntimeError(f"{stem}: duplicate node ID")
    by_id = dict(zip(nodes.node_id.astype(int), nodes.t.astype(int), strict=True))
    if any(by_id.get(int(src), -100) + 1 != by_id.get(int(tgt), -200)
           for src, tgt in zip(edges.source_id, edges.target_id, strict=True)):
        raise RuntimeError(f"{stem}: invalid edge time or endpoint")
    if not edges.empty and (edges.target_id.value_counts().max() > 1
                            or edges.source_id.value_counts().max() > 2):
        raise RuntimeError(f"{stem}: invalid graph degree")
receipt = {
    "experiment": "exp050_shared_edge_graph_learning", "variant": VARIANT,
    "selection_sha256": _selection_sha, "config_sha256": CONFIG_SHA256,
    "test_videos": test_stems, "rows": len(frame),
    "submission_sha256": hashlib.sha256(SUBMISSION_PATH.read_bytes()).hexdigest(),
    "train_receipt_sha256": hashlib.sha256(_train_receipts[0].read_bytes()).hexdigest(),
    "elapsed_seconds": time.monotonic() - run_start,
    "feature_capture_seconds": capture_elapsed_seconds,
    "feature_capture_workers": capture_workers,
    "videos": video_rows,
}
if VARIANT == "optuna":
    receipt["tune_receipt_sha256"] = hashlib.sha256(_tune_receipts[0].read_bytes()).hexdigest()
(WORKING_DIR / f"exp050_submission_{VARIANT}_receipt.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\\n"
)
print("SUBMISSION_READY", VARIANT, len(frame), receipt["submission_sha256"], flush=True)
"""

GNN_SOLVE = """catalog, by_parent, scores, model_folds = score_submission_gnn_sets(
            models, embryo, coords, features, expanded, device="cuda",
        )
        fold = model_folds[0] if len(model_folds) == 1 else None
        chosen, solve_info = solve_set_ilp(
            coords, expanded, catalog, by_parent, scores,
            appearance_cost=float(CONFIG["model"]["decoding"]["appearance_cost"]),
            disappearance_cost=float(CONFIG["model"]["decoding"]["disappearance_cost"]),
            timeout_seconds=float(CONFIG["model"]["decoding"]["time_limit_seconds_per_video"]),
        )
        selected_node_ids = set(solve_info["selected_node_ids"])
        nodes = {
            i: {"node_id": i, "t": int(t), "z": float(z), "y": float(y), "x": float(x)}
            for i, (t, z, y, x) in enumerate(coords)
            if i in selected_node_ids
        }
        selected = [
            {"source_id": int(s), "target_id": int(t), "edge_prob": float(p)}
            for s, t, p, _ in chosen
        ]"""

OPTUNA_SOLVE = """costs = tuple(tune_receipt["folds"][str(fold)]["selected_costs"])
        nodes, selected, solve_info = legacy_solve(coords, expanded, costs, stem)"""

FIXED_COSTS = """costs = (
            float(CONFIG["model"]["decoding"]["appearance_cost"]),
            float(CONFIG["model"]["decoding"]["disappearance_cost"]),
            float(CONFIG["model"]["params"]["division_cost_initial"]),
        )
        nodes, selected, solve_info = legacy_solve(coords, CANDIDATES, costs, stem)"""

SOLVES = {
    "gnn": GNN_SOLVE,
    "optuna": OPTUNA_SOLVE,
    "exp043": FIXED_COSTS.replace("CANDIDATES", "baseline"),
    "old_cost": FIXED_COSTS.replace("CANDIDATES", "expanded"),
}

for variant, solve in SOLVES.items():
    if len(sys.argv) > 1 and variant != sys.argv[1]:
        continue
    submission = SUBMISSION
    if variant == "gnn":
        submission = submission.replace('        fold = 1 if embryo == "44b6" else 0\n', "")
        submission = submission.replace(
            '"fold": fold, "candidate_edges": len(expanded)',
            '"fold": fold, "gnn_model_folds": model_folds, "candidate_edges": len(expanded)',
        )
    if variant == "old_cost":
        # Fixed costs use no trained fold and apply to every test embryo.
        submission = submission.replace(
            '        embryo = stem.split("_", 1)[0]\n        fold = 1 if embryo == "44b6" else 0\n',
            "",
        )
        submission = submission.replace('"fold": fold, "candidate_edges"', '"candidate_edges"')
    if variant != "optuna":
        tune_receipt_line = (
            'if VARIANT == "optuna":\n'
            '    receipt["tune_receipt_sha256"] = '
            "hashlib.sha256(_tune_receipts[0].read_bytes()).hexdigest()\n"
        )
        assert submission.count(tune_receipt_line) == 1
        submission = submission.replace(tune_receipt_line, "")
    body = (
        PREFIX.replace(
            "# # exp050: four-arm outer graph evaluation on the fixed x138 cache",
            f"# # exp050: frozen {variant} test inference and competition CSV",
            1,
        )
        + (
            COMMON_SETUP.replace(
                'if any(stem.split("_", 1)[0] not in {"44b6", "6bba"} for stem in test_stems):\n'
                '    raise RuntimeError("Unknown embryo prefix; '
                'no frozen cross-embryo model or costs")\n',
                "",
            )
            if variant in {"gnn", "old_cost"}
            else COMMON_SETUP
        )
        + PREDICTOR_PATCH
        + FEATURE_PATCH
        + CAPTURE_RUN.replace("VARIANT", repr(variant))
        + POSTPROCESS
        + LEARNED
        + "# %% [markdown]\n# ## Frozen configuration and solver\n\n# %%\n"
        + (
            "import hashlib\nimport json\nimport time\n"
            "from pathlib import Path\nimport pandas as pd\n"
        )
        + MODEL_SETUP.replace("VARIANT", repr(variant))
        + ("import polars as _audit_pl\n" if variant != "gnn" else "")
        + LEGACY
        + (GNN_SETUP if variant == "gnn" else OPTUNA_SETUP if variant == "optuna" else "")
        + submission.replace("METHOD_SOLVE_PLACEHOLDER", solve)
        .replace("VARIANT", repr(variant))
        .replace(
            '"candidate_edges": len(expanded)',
            '"candidate_edges": len(baseline)'
            if variant == "exp043"
            else '"candidate_edges": len(expanded)',
        )
    )
    output = ROOT / f"{NAME}_submission_{variant}.py"
    compile(body, str(output), "exec")
    output.write_text(body)
    print(output, len(body), "bytes")
