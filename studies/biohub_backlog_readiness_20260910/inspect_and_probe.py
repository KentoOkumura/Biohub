"""Small, reproducible source and arithmetic checks; no competition images or weights."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import math
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / "experiments/exp002_unet3d_expandable_segments/official_source"


def save(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def extract(path, names, dest):
    source = path.read_text()
    tree = ast.parse(source)
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in selected} == set(names)
    text = "from __future__ import annotations\n\n" + "\n\n".join(ast.get_source_segment(source, node) for node in selected) + "\n"
    destination = OUT / dest
    destination.write_text(text)
    namespace = {}
    exec(compile(text, str(destination), "exec"), namespace)
    return namespace, {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "functions": names, "extracted_path": dest}


class Graph:
    def __init__(self, edges):
        self.edges = edges

    def predecessors(self, node):
        return [src for src, dst in self.edges if dst == node]

    def successors(self, node):
        return [dst for src, dst in self.edges if src == node]


def focal(logits, target, mask, denominator_mask=None):
    values = logits.copy()
    if denominator_mask is not None:
        values[~denominator_mask] = -np.inf
    exps = np.exp(values - values.max(axis=0, keepdims=True))
    probs = exps / exps.sum(axis=0, keepdims=True)
    probs = np.clip(probs, 1e-12, 1 - 1e-12)
    bce = -(target * np.log(probs) + (1 - target) * np.log(1 - probs))
    p_t = probs * target + (1 - probs) * (1 - target)
    return float((((1 - p_t) ** 2) * bce)[mask].mean())


def gradient(logits, target, mask, denominator_mask=None):
    grad = np.empty_like(logits)
    for idx in np.ndindex(logits.shape):
        hi, lo = logits.copy(), logits.copy()
        hi[idx] += 1e-6
        lo[idx] -= 1e-6
        grad[idx] = (focal(hi, target, mask, denominator_mask) - focal(lo, target, mask, denominator_mask)) / 2e-6
    return grad.tolist()


def fetch(item):
    name, url = item
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Biohub-backlog-source-audit"})
        with urllib.request.urlopen(request, timeout=25) as response:
            body = response.read(2_000_001)
        if len(body) > 2_000_000:
            raise ValueError("source exceeds small-source limit")
        (OUT / name).write_bytes(body)
        return {"name": name, "url": url, "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body), "ok": True}
    except Exception as error:
        return {"name": name, "url": url, "ok": False, "error": f"{type(error).__name__}: {error}"}


def main():
    official, official_source = extract(SOURCE / "src/tracking_cellmot/division_metrics.py", ["_is_strongly_connected_division", "_bipartite_max_matching"], "official_division_helpers.py")
    proxy_path = Path("/tmp/biohub-baselines-20260910/dctta-source.txt")
    if not proxy_path.exists():
        proxy_path = OUT / "public_division_helpers.py"
    proxy, proxy_source = extract(proxy_path, ["weakly_connected_components", "compute_division_confusion"], "public_division_helpers.py")
    gt_edges = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]
    cases = [
        ("correct_local_fork", [(10, 11), (10, 12)], {10: 0, 11: 1, 12: 2}, 10, {10}, [{11}, {12}], True),
        ("distant_fork_and_descendants", [(10, 20), (20, 21), (21, 11), (21, 12)], {10: 0, 11: 5, 12: 6}, 21, {10}, [set(), set()], False),
        ("same_predicted_child_branch", [(10, 11), (10, 12), (11, 13), (11, 14)], {10: 0, 13: 3, 14: 4}, 10, {10}, [{13}, {14}], False),
    ]
    divisions = []
    for name, edges, mapping, fork, parents, daughters, expected in cases:
        nodes = {n: {} for edge in edges for n in edge}
        result = proxy["compute_division_confusion"](nodes, edges, {n: {} for edge in gt_edges for n in edge}, gt_edges, mapping, {v: k for k, v in mapping.items()})
        accepted = official["_is_strongly_connected_division"](Graph(edges), fork, parents, daughters)
        assert accepted is expected
        assert result[0] == 1
        divisions.append({"case": name, "pred_edges": edges, "gt_edges": gt_edges, "pred_to_gt": mapping, "official_checked_fork": fork, "official_parent_ids": sorted(parents), "official_daughter_ids": [sorted(s) for s in daughters], "public_proxy_tp_fp_fn": result, "official_local_topology_accepts": accepted})
    target = np.array([[1., 0.], [0., 0.]])
    original_mask = (target.sum(axis=1) > 0)[:, None] | (target.sum(axis=0) > 0)[None, :]
    known_child_mask = np.broadcast_to((target.sum(axis=0) > 0)[None, :], target.shape)
    current_grad = gradient(np.zeros_like(target), target, original_mask)
    child_mask_grad = gradient(np.zeros_like(target), target, known_child_mask)
    assert current_grad[0][1] > 0 and child_mask_grad[0][1] == 0
    target2 = np.array([[1.], [0.], [0.]])
    known_terms = np.array([[True], [True], [False]])
    denominator_grad = gradient(np.zeros_like(target2), target2, known_terms)
    restricted_grad = gradient(np.zeros_like(target2), target2, known_terms, known_terms)
    assert denominator_grad[2][0] > 0 and restricted_grad[2][0] == 0
    # Nearest-only greedy can fail to assign a valid second nearest GT.
    distances = abs(np.array([0.1, 1.4])[:, None] - np.array([0., 3.])[None, :])
    taken, greedy = set(), []
    for pred in np.argsort(distances.min(axis=1)):
        gt = int(distances[pred].argmin())
        if distances[pred, gt] <= 5 and gt not in taken:
            greedy.append((int(pred), gt))
            taken.add(gt)
    maximum = official["_bipartite_max_matching"]([0, 1], {i: set(np.flatnonzero(distances[i] <= 5).tolist()) for i in range(2)})
    assert len(greedy) == 1 and len(maximum) == 2
    pool, pool_source = extract(SOURCE / "scripts/predict_unet_transformer.py", ["pool_kernel_from_um"], "official_pool_helper.py")
    kernels = {str(um): pool["pool_kernel_from_um"](um, (1.625,) * 3) for um in (3., 5.)}
    assert kernels["3.0"] == kernels["5.0"] == (3, 3, 3)
    save("synthetic_checks.json", {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Exact extracted topology/helper functions; NumPy loss arithmetic and finite differences. No full official scoring, Torch autograd, real image evaluation, or accuracy improvement is claimed.",
        "sources": [official_source, proxy_source, pool_source],
        "division_cases": divisions,
        "one_fork_two_gt_maximum_matches": len(official["_bipartite_max_matching"]([0, 1], {0: {10}, 1: {10}})),
        "edge_gradients": {"logits": "all zero", "partly_annotated_daughter_target": target.tolist(), "current_or_mask": current_grad, "known_child_columns_only": child_mask_grad, "unknown_parent_target": target2.tolist(), "known_terms_only_softmax_all_parents": denominator_grad, "known_terms_and_known_denominator": restricted_grad, "caveat": "Removing ambiguous parents from softmax is an arithmetic diagnostic, not proof that the resulting supervision is identifiable or accurate."},
        "nearest_only_matching": {"distances_um": distances.tolist(), "greedy": greedy, "maximum": maximum},
        "thresholds": {"train_raw_logit": 0.3, "train_sigmoid_equivalent": 1 / (1 + math.exp(-0.3)), "infer_sigmoid": 0.99, "infer_logit_equivalent": math.log(99), "nms_kernel_after_downsampling": kernels},
        "feature_memory_assumptions": {"spatial_shape": [64, 64, 64], "channels": 32, "bytes_per_value": 2, "frames": 100, "videos": 199, "frame_MiB": 16, "one_feature_per_frame_GiB": 100 * 199 * 16 / 1024, "separate_99_two_frame_windows_GiB": 99 * 2 * 199 * 16 / 1024, "caveat": "Arithmetic only; each frame's contextual feature changes with window, so one-feature-per-frame caching is generally not equivalent. Excludes metadata, augmentation variants, disk overhead."},
    })
    save("local_availability.json", {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "packages": {name: importlib.util.find_spec(name) is not None for name in ["numpy", "scipy", "torch", "polars", "tracksdata"]}, "local_models": [str(p.relative_to(ROOT)) for p in (ROOT / "experiments/exp002_unet3d_expandable_segments/artifacts").rglob("*.pth")]})
    requests = [
        ("official_head.json", "https://api.github.com/repos/royerlab/kaggle-cell-tracking-competition/commits/main"),
        ("focus_readme.md.txt", "https://raw.githubusercontent.com/yu-lab-vt/FOCUS-3D/main/README.md"),
        ("trackastra_pretrained.json", "https://raw.githubusercontent.com/weigertlab/trackastra/main/trackastra/model/pretrained.json"),
        ("hoct_models.py", "https://raw.githubusercontent.com/royerlab/hoct/main/src/hoct/_models.py"),
    ]
    with ThreadPoolExecutor(max_workers=4) as executor:
        fetched = list(executor.map(fetch, requests))
    if (OUT / "official_head.json").exists():
        commit = json.loads((OUT / "official_head.json").read_text())["sha"]
        fetched.append(fetch(("latest_division_metrics.py", f"https://raw.githubusercontent.com/royerlab/kaggle-cell-tracking-competition/{commit}/src/tracking_cellmot/division_metrics.py")))
        fetched.append(fetch(("latest_metrics.md.txt", f"https://raw.githubusercontent.com/royerlab/kaggle-cell-tracking-competition/{commit}/metrics.md")))
    save("external_sources.json", {"retrieved_at_utc": datetime.now(timezone.utc).isoformat(), "sources": fetched})
    print(json.dumps({"division_checks": divisions, "loss_gradients": {"unannotated_daughter_original": current_grad[0][1], "unannotated_daughter_masked": child_mask_grad[0][1], "unknown_parent_bce_mask_only": denominator_grad[2][0], "unknown_parent_denominator_excluded": restricted_grad[2][0]}, "nms": kernels, "source_fetches": fetched}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
