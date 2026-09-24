"""Build the x138 feature-capture and coordinate-head training notebook."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
REFERENCE = (
    ROOT.parents[1]
    / "docs/notebooks"
    / "biohub-cell-tracking-during-development"
    / "anvithpothula__biohub-x138"
    / "biohub-x138.ipynb"
)
OUTPUT = ROOT / "exp043_x138_self_trained_head_train.py"
REFERENCE_SHA = "6b655e39bbfd2d3d6c762badea69847d3f00f5b548f385cb01b07ee2600fde6d"
SPLIT_ANCHOR = "start_time = time.time()\n"

SETUP = r"""
import hashlib as _self_hashlib
import json as _self_json
import yaml as _self_yaml
from pathlib import Path as _SelfPath

SELF_CONFIG = _self_yaml.safe_load(_SelfPath('config.yaml').read_text())
TRAIN_IMAGE_DIR = COMP_DIR / 'train'
PUBLIC_TEST_DIR = COMP_DIR / 'test'
if not TRAIN_IMAGE_DIR.is_dir() or not PUBLIC_TEST_DIR.is_dir():
    raise FileNotFoundError((TRAIN_IMAGE_DIR, PUBLIC_TEST_DIR))
if _self_hashlib.sha256(_SelfPath('config.yaml').read_bytes()).hexdigest() == '':
    raise RuntimeError('unreachable config guard')
TEST_DIR = TRAIN_IMAGE_DIR
print('Training images and GEFF:', TRAIN_IMAGE_DIR)
"""

SELECTION = r"""
def _stable_video_rank(stem: str) -> str:
    seed = SELF_CONFIG['data']['selection_seed']
    return _self_hashlib.sha256(f'{seed}:{stem}'.encode()).hexdigest()

_all_train_stems = sorted(path.stem for path in TRAIN_IMAGE_DIR.glob('*.zarr'))
_selected_by_embryo = {}
for _embryo in SELF_CONFIG['data']['expected_embryos']:
    _eligible = [stem for stem in _all_train_stems
                 if stem.split('_', 1)[0] == str(_embryo)
                 and (TRAIN_IMAGE_DIR / f'{stem}.geff').is_dir()]
    _eligible.sort(key=lambda stem: (_stable_video_rank(stem), stem))
    _required = int(SELF_CONFIG['data']['videos_per_embryo'])
    if len(_eligible) < _required:
        raise RuntimeError({'embryo': _embryo, 'eligible': len(_eligible), 'required': _required})
    _selected_by_embryo[str(_embryo)] = _eligible[:_required]
test_stems = sorted(stem for group in _selected_by_embryo.values() for stem in group)
if len(test_stems) != 2 * int(SELF_CONFIG['data']['videos_per_embryo']):
    raise RuntimeError('Unexpected train video count')
(WORKING_DIR / 'selected_train_videos.json').write_text(
    _self_json.dumps(_selected_by_embryo, indent=2, sort_keys=True) + '\n')
print('Selected train videos:', _selected_by_embryo, flush=True)
"""

CAPTURE = r"""
# The original x138 source has already been materialized and patched. The only
# change to the V1284 module is its supported capture mode instead of a private head.
_CAPTURE_ROOT = WORKING_DIR / 'v1284_capture'
_CAPTURE_ROOT.mkdir(exist_ok=True)
os.environ['V1284_MODE'] = 'capture'
os.environ['V1284_CAPTURE'] = str(_CAPTURE_ROOT)


def _run_capture_subset(stems: list[str], label: str) -> float:
    splits_path.write_text(json.dumps([{'split': 0, 'train': [], 'test': stems}], indent=2))
    worker_count = min(2, _torch.cuda.device_count(), len(stems))
    if worker_count != 2:
        raise RuntimeError(f'Expected two T4 GPUs, got {_torch.cuda.device_count()}')
    tokens = _visible_cuda_tokens(worker_count)
    processes = {}
    commands = {}
    started = time.time()
    for shard in range(worker_count):
        command = [*predict_cmd, '--method', f'{METHOD}_capture_{label}_gpu{shard}',
                   '--slice', f'{shard}::{worker_count}']
        child_env = {**os.environ, 'PYTHONPATH': 'src',
                     'CUDA_VISIBLE_DEVICES': tokens[shard],
                     'BIOHUB_GPU_SHARD': f'{shard}/{worker_count}'}
        commands[shard] = command
        processes[shard] = subprocess.Popen(command, cwd=REPO_DIR, env=child_env)
    _wait_for_prediction_shards(processes, commands)
    elapsed = time.time() - started
    for stem in stems:
        folder = _CAPTURE_ROOT / stem
        if not folder.is_dir() or not any(folder.glob('*.npz')):
            raise RuntimeError(f'No coordinate features captured for {stem}')
    print(f'Captured {label}: {len(stems)} videos, {elapsed / 60:.2f} min', flush=True)
    return elapsed

_pilot = sorted(group[0] for group in _selected_by_embryo.values())
_pilot_seconds = _run_capture_subset(_pilot, 'pilot')
_expected_capture_minutes = (_pilot_seconds / 60) * len(test_stems) / len(_pilot)
_budget_minutes = float(SELF_CONFIG['runtime']['capture_budget_minutes'])
print('Projected capture minutes:', round(_expected_capture_minutes, 2),
      'budget:', _budget_minutes, flush=True)
if _expected_capture_minutes > _budget_minutes:
    raise RuntimeError('Pilot projection exceeds capture budget; remaining videos were not started')
_remaining = [stem for stem in test_stems if stem not in set(_pilot)]
_remaining_seconds = _run_capture_subset(_remaining, 'remaining')
_capture_seconds = _pilot_seconds + _remaining_seconds
if _capture_seconds / 60 > _budget_minutes:
    raise RuntimeError('Actual capture exceeded configured budget')
"""

ASSEMBLE = r"""
import numpy as np
import torch
import tracksdata as td
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

_GRID_SPACING = np.asarray(SELF_CONFIG['data']['feature_grid_spacing_zyx_um'], dtype=np.float32)
_NATIVE_SPACING = np.asarray(SELF_CONFIG['data']['native_spacing_zyx_um'], dtype=np.float32)
_MATCH_RADIUS = float(SELF_CONFIG['data']['candidate_match_radius_um'])
_FEATURE_WIDTH = int(SELF_CONFIG['data']['feature_width'])
_movie_arrays = {}
_capture_digest = _self_hashlib.sha256()
for stem in test_stems:
    graph = td.graph.IndexedRXGraph.from_geff(TRAIN_IMAGE_DIR / f'{stem}.geff')
    if isinstance(graph, tuple):
        graph = graph[0]
    gt_by_t = {}
    for row in graph.node_attrs().iter_rows(named=True):
        gt_by_t.setdefault(int(row['t']), []).append(
            np.asarray([row['z'], row['y'], row['x']], dtype=np.float32) * _NATIVE_SPACING)
    x_parts, y_parts, before_parts, frame_parts = [], [], [], []
    for path in sorted((_CAPTURE_ROOT / stem).glob('*.npz')):
        t = int(path.stem)
        with np.load(path, allow_pickle=False) as payload:
            coords = np.asarray(payload['coords'], dtype=np.float32)
            features = np.asarray(payload['features'], dtype=np.float32)
        valid_shape = coords.ndim == 2 and coords.shape[1] == 4
        valid_shape &= features.shape == (len(coords), _FEATURE_WIDTH)
        if not valid_shape:
            raise RuntimeError({'file': str(path),
                                'coords': coords.shape, 'features': features.shape})
        if not np.all(coords[:, 0] == t) or not np.isfinite(features).all():
            raise RuntimeError(f'Bad frame/features: {path}')
        _capture_digest.update(stem.encode())
        _capture_digest.update(path.name.encode())
        _capture_digest.update(np.ascontiguousarray(coords).tobytes())
        _capture_digest.update(np.ascontiguousarray(features).tobytes())
        gt = np.asarray(gt_by_t.get(t, []), dtype=np.float32).reshape(-1, 3)
        if not len(gt) or not len(coords):
            continue
        candidate_um = coords[:, 1:] * _GRID_SPACING
        cost = cdist(candidate_um, gt)
        rows, cols = linear_sum_assignment(np.where(cost <= _MATCH_RADIUS, cost, 1e6))
        valid = cost[rows, cols] <= _MATCH_RADIUS
        rows, cols = rows[valid], cols[valid]
        if len(rows):
            x_parts.append(features[rows])
            y_parts.append(gt[cols] - candidate_um[rows])
            before_parts.append(cost[rows, cols].astype(np.float32))
            frame_parts.append(np.full(len(rows), t, dtype=np.int16))
    if not x_parts:
        raise RuntimeError(f'No matched known centers for {stem}')
    _movie_arrays[stem] = {
        'x': np.concatenate(x_parts), 'y': np.concatenate(y_parts),
        'before': np.concatenate(before_parts), 't': np.concatenate(frame_parts),
        'embryo': stem.split('_', 1)[0],
    }
    print('Matched', stem, len(_movie_arrays[stem]['x']), flush=True)
if sum(len(row['x']) for row in _movie_arrays.values()) < 100:
    raise RuntimeError('Too few matched known centers')
_CAPTURE_SHA = _capture_digest.hexdigest()
print('Capture content SHA256:', _CAPTURE_SHA, flush=True)
"""

TRAIN = r"""
import copy
import random
import torch.nn.functional as F

_PARAMS = SELF_CONFIG['model']['params']
_SEED = int(_PARAMS['seed'])
_FOLDS = int(_PARAMS['validation_folds'])
if _FOLDS != 5 or len(test_stems) != 20:
    raise RuntimeError('Five-fold 20-video contract changed')
random.seed(_SEED)
np.random.seed(_SEED)
torch.manual_seed(_SEED)
torch.set_num_threads(2)


def _head():
    model = torch.nn.Sequential(torch.nn.Linear(_FEATURE_WIDTH, 32),
                                torch.nn.SiLU(), torch.nn.Linear(32, 3))
    torch.nn.init.zeros_(model[-1].weight)
    torch.nn.init.zeros_(model[-1].bias)
    return model


def _bounded(model, x):
    delta = model(x)
    norm = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
    return float(_PARAMS['max_shift_um']) * delta / (1 + norm)


def _fit_head(train_stems: list[str], seed: int):
    torch.manual_seed(seed)
    x = np.concatenate([_movie_arrays[s]['x'] for s in train_stems]).astype(np.float32)
    y = np.concatenate([_movie_arrays[s]['y'] for s in train_stems]).astype(np.float32)
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 1e-6)
    xx = torch.from_numpy((x - mean) / scale)
    yy = torch.from_numpy(y)
    model = _head()
    opt = torch.optim.AdamW(model.parameters(), lr=float(_PARAMS['learning_rate']),
                            weight_decay=float(_PARAMS['weight_decay']))
    batch_size = int(_PARAMS['batch_size'])
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(int(_PARAMS['epochs'])):
        for batch in torch.randperm(len(xx), generator=generator).split(batch_size):
            prediction = _bounded(model, xx[batch])
            loss = F.huber_loss(prediction, yy[batch], delta=float(_PARAMS['huber_delta_um']))
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite train loss at epoch {epoch}')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    model.eval()
    return model, mean, scale


def _predict(model, mean, scale, stems):
    with torch.no_grad():
        return {stem: _bounded(model, torch.from_numpy(
            (_movie_arrays[stem]['x'] - mean) / scale)).numpy() for stem in stems}

_fold_stems = {fold: [] for fold in range(_FOLDS)}
for embryo, stems in _selected_by_embryo.items():
    ordered = sorted(stems, key=lambda s: (_stable_video_rank(s), s))
    for i, stem in enumerate(ordered):
        _fold_stems[i % _FOLDS].append(stem)
_oof_shifts = {}
for fold in range(_FOLDS):
    valid_stems = sorted(_fold_stems[fold])
    train_stems = sorted(set(test_stems) - set(valid_stems))
    model, mean, scale = _fit_head(train_stems, _SEED + fold)
    _oof_shifts.update(_predict(model, mean, scale, valid_stems))
    print('Fold', fold, 'train', len(train_stems), 'valid', valid_stems, flush=True)
if set(_oof_shifts) != set(test_stems):
    raise RuntimeError('Video-fold predictions incomplete')

_per_movie = []
_oof_digest = _self_hashlib.sha256()
for stem in test_stems:
    row = _movie_arrays[stem]
    after = np.linalg.vector_norm(row['y'] - _oof_shifts[stem], axis=1)
    before = row['before']
    _oof_digest.update(stem.encode())
    _oof_digest.update(np.ascontiguousarray(_oof_shifts[stem], dtype=np.float32).tobytes())
    _per_movie.append({'video': stem, 'embryo': row['embryo'], 'pairs': len(before),
                       'before_mean_um': float(before.mean()),
                       'after_mean_um': float(after.mean()),
                       'improved': bool(after.mean() < before.mean())})

_readout = {
    'experiment': 'exp043_x138_self_trained_head',
    'evaluation': 'video-grouped, conditional on pretrained public image models',
    'capture_sha256': _CAPTURE_SHA,
    'oof_prediction_sha256': _oof_digest.hexdigest(),
    'videos': len(_per_movie),
    'pairs': int(sum(row['pairs'] for row in _per_movie)),
    'per_movie': _per_movie,
    'by_embryo': {},
}
for embryo in sorted(_selected_by_embryo):
    stems = _selected_by_embryo[embryo]
    base = np.concatenate([_movie_arrays[s]['before'] for s in stems])
    refined = np.concatenate([np.linalg.vector_norm(
        _movie_arrays[s]['y'] - _oof_shifts[s], axis=1) for s in stems])
    _readout['by_embryo'][embryo] = {
        'videos': len(stems), 'pairs': len(base),
        'before_mean_um': float(base.mean()), 'after_mean_um': float(refined.mean()),
        'improved_videos': sum(row['improved'] for row in _per_movie if row['embryo'] == embryo),
    }
(WORKING_DIR / 'coordinate_validation.json').write_text(_self_json.dumps(_readout, indent=2) + '\n')
print(_self_json.dumps(_readout['by_embryo'], indent=2), flush=True)

_final_model, _final_mean, _final_scale = _fit_head(test_stems, _SEED + _FOLDS)
_HEAD_PATH = WORKING_DIR / 'self_trained_v1284_head.pt'
torch.save({'state_dict': {k: v.cpu() for k, v in _final_model.state_dict().items()},
            'mean': torch.from_numpy(_final_mean),
            'scale': torch.from_numpy(_final_scale)}, _HEAD_PATH)
_HEAD_SHA = _self_hashlib.sha256(_HEAD_PATH.read_bytes()).hexdigest()
_manifest = {
    'checkpoint': _HEAD_PATH.name, 'checkpoint_sha256': _HEAD_SHA,
    'reference_notebook_sha256': SELF_CONFIG['source']['reference_notebook_sha256'],
    'capture_content_sha256': _CAPTURE_SHA,
    'oof_prediction_sha256': _readout['oof_prediction_sha256'],
    'training_videos': test_stems, 'pairs': _readout['pairs'],
    'feature_width': _FEATURE_WIDTH, 'feature_grid_spacing_um': _GRID_SPACING.tolist(),
    'model': 'Linear(224,32)-SiLU-Linear(32,3)',
    'shift_bound_um': float(_PARAMS['max_shift_um']),
}
(WORKING_DIR / 'self_trained_head_manifest.json').write_text(
    _self_json.dumps(_manifest, indent=2, sort_keys=True) + '\n')
print('Final self-trained head:', _HEAD_PATH, _HEAD_SHA, flush=True)
"""

RECEIPT = r"""
_validation_path = WORKING_DIR / 'coordinate_validation.json'
_manifest_path = WORKING_DIR / 'self_trained_head_manifest.json'
for _required_path in (_validation_path, _manifest_path, _HEAD_PATH):
    if not _required_path.is_file():
        raise FileNotFoundError(_required_path)
_saved_validation = _self_json.loads(_validation_path.read_text())
_saved_manifest = _self_json.loads(_manifest_path.read_text())
if _saved_validation['capture_sha256'] != _CAPTURE_SHA:
    raise RuntimeError('Saved validation capture SHA differs from the current run')
_actual_head_sha = _self_hashlib.sha256(_HEAD_PATH.read_bytes()).hexdigest()
if _saved_manifest['checkpoint_sha256'] != _actual_head_sha:
    raise RuntimeError('Saved checkpoint SHA differs from the manifest')
if (_saved_validation['pairs'] != _saved_manifest['pairs']
        or _saved_validation['videos'] != len(test_stems)):
    raise RuntimeError('Saved training coverage differs from the manifest')
_receipt = {
    'experiment': 'exp043_x138_self_trained_head',
    'training_complete': True,
    'head_sha256': _saved_manifest['checkpoint_sha256'],
    'capture_sha256': _CAPTURE_SHA,
    'oof_prediction_sha256': _saved_validation['oof_prediction_sha256'],
    'training_videos': len(test_stems),
    'matched_pairs': _saved_validation['pairs'],
    'capture_seconds': _capture_seconds,
    'official_score': None,
    'competition_submission_created': False,
}
(WORKING_DIR / 'self_trained_head_train_receipt.json').write_text(
    _self_json.dumps(_receipt, indent=2, sort_keys=True) + '\n')
print(_self_json.dumps(_receipt, indent=2, sort_keys=True), flush=True)
"""


def build() -> None:
    if hashlib.sha256(REFERENCE.read_bytes()).hexdigest() != REFERENCE_SHA:
        raise RuntimeError("Pinned x138 notebook SHA256 mismatch")
    notebook = json.loads(REFERENCE.read_text())
    source = ["".join(cell["source"]) for cell in notebook["cells"]]
    if len(source) != 12 or source[4].count(SPLIT_ANCHOR) != 1:
        raise RuntimeError("x138 cell structure changed")
    before, _execution = source[4].split(SPLIT_ANCHOR)
    guard_start = before.index("_myhead = sorted(Path('/kaggle/input').rglob(")
    guard_end = before.index("\n_trial_source = _ps.read_text()", guard_start)
    before = (
        before[:guard_start]
        + (
            "os.environ['V1284_MODE']='capture'\n"
            "os.environ['V1284_CAPTURE']=str(WORKING_DIR / 'v1284_capture')\n"
        )
        + before[guard_end:]
    )
    original_selection = "test_stems = list_test_stems()\n"
    if before.count(original_selection) != 1:
        raise RuntimeError("x138 video selection anchor mismatch")
    before = before.replace(original_selection, SELECTION + "\n")
    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    if config["source"]["reference_notebook_sha256"] != REFERENCE_SHA:
        raise RuntimeError("Config reference SHA mismatch")
    cells = [
        (
            "markdown",
            "# exp043 x138 self-trained coordinate head\n\n"
            "Fixed public image models and tracker; train the coordinate head from GEFF.",
        ),
        ("markdown", "## 1. Public x138 configuration and fixed artifacts"),
        ("code", source[0]),
        ("code", source[1]),
        ("code", source[2]),
        ("code", source[3]),
        ("markdown", "## 2. Train input and feature capture"),
        ("code", SETUP),
        ("code", before),
        ("code", CAPTURE),
        ("markdown", "## 3. Match known centers and assemble supervised examples"),
        ("code", ASSEMBLE),
        ("markdown", "## 4. Video-grouped evaluation and final head training"),
        ("code", TRAIN),
        ("markdown", "## 5. Verify saved training artifacts"),
        ("code", RECEIPT),
    ]
    lines = [
        "# ---",
        "# jupyter:",
        "#   jupytext:",
        "#     formats: py:percent",
        "#   kernelspec:",
        "#     display_name: Python 3",
        "#     language: python",
        "#     name: python3",
        "# ---",
        "",
    ]
    for kind, value in cells:
        if kind == "markdown":
            lines.append("# %% [markdown]")
            lines.extend("# " + line if line else "#" for line in value.splitlines())
        else:
            lines.append("# %%")
            lines.append(value.rstrip())
        lines.append("")
    OUTPUT.write_text("\n".join(lines))
    print(OUTPUT)
    print("cells:", len(cells), "bytes:", OUTPUT.stat().st_size)


if __name__ == "__main__":
    build()
