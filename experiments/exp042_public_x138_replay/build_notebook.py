"""Build the x138 replay notebook from the pinned public notebook.

The original code cells are copied in order. Added cells only verify inputs,
record coordinate hashes, and summarize outputs; they do not select predictions.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / "assets/reference_notebook/biohub-x138.ipynb"
OUTPUT = ROOT / "exp042_public_x138_replay_inference.py"
REFERENCE_SHA256 = "6b655e39bbfd2d3d6c762badea69847d3f00f5b548f385cb01b07ee2600fde6d"
SPLIT_ANCHOR = "start_time = time.time()\n"

PREFLIGHT = r"""
# EXP042_ADDITION_START: reject a missing or different coordinate head before inference.
import hashlib as _x138_hashlib
import json as _x138_json
import os as _x138_os
import platform as _x138_platform
import sys as _x138_sys
from pathlib import Path as _X138Path

_X138_HEAD_SHA256 = __HEAD_SHA__
_X138_HEAD_DATASET_REF = __HEAD_REF__
_X138_HEAD_DATASET_VERSION_ID = __HEAD_VERSION_ID__
_X138_REFERENCE_SHA256 = __REFERENCE_SHA__
_X138_DOCKER_IMAGE_SHA256 = '37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461'
_X138_CANONICAL_MANIFESTS = __CANONICAL_MANIFESTS__
_X138_CHECKPOINT_SHA256 = __CHECKPOINT_SHA__
_X138_SUPPORT_SOURCE_SHA256 = __SUPPORT_SOURCE_SHA__

def _x138_file_sha(path):
    digest = _x138_hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def _x138_json_sha(value):
    return _x138_hashlib.sha256(
        _x138_json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(',', ':')).encode('utf-8')
    ).hexdigest()

if not _X138_HEAD_SHA256 or not _X138_HEAD_DATASET_REF or not _X138_HEAD_DATASET_VERSION_ID:
    raise RuntimeError('V1284 checkpoint dataset ref and SHA256 must be pinned in config.yaml')
if len(_X138_HEAD_SHA256) != 64:
    raise RuntimeError('Invalid pinned V1284 checkpoint SHA256')
_x138_matches = sorted(
    _X138Path('/kaggle/input').rglob('biohub-v1284-head-s075/v1284_head.pt')
)
if len(_x138_matches) != 1:
    raise RuntimeError({'v1284_checkpoint_matches': [str(p) for p in _x138_matches]})
_x138_head_path = _x138_matches[0]
_x138_slug = _X138_HEAD_DATASET_REF.split('/')[-1]
if _x138_slug not in _x138_head_path.parts:
    raise RuntimeError({'v1284_dataset_ref': _X138_HEAD_DATASET_REF,
                        'actual_path': str(_x138_head_path)})
_x138_head_actual_sha = _x138_file_sha(_x138_head_path)
if _x138_head_actual_sha != _X138_HEAD_SHA256:
    raise RuntimeError({'v1284_expected_sha': _X138_HEAD_SHA256,
                        'v1284_actual_sha': _x138_head_actual_sha})
print('Pinned V1284 checkpoint:', _x138_head_path, _x138_head_actual_sha)
# EXP042_ADDITION_END
"""

INPUT_MANIFEST = r"""
# EXP042_ADDITION_START: verify public artifacts and describe the dynamic test input.
import torch as _x138_torch
_x138_gpu_names = [_x138_torch.cuda.get_device_name(i)
                   for i in range(_x138_torch.cuda.device_count())]
if len(_x138_gpu_names) != 2 or any('T4' not in name for name in _x138_gpu_names):
    raise RuntimeError({'required_gpu': '2 x T4', 'actual': _x138_gpu_names})
if not TEST_DIR.is_dir():
    raise FileNotFoundError(TEST_DIR)
_x138_artifacts = {}
for _x138_slug, _x138_expected in _X138_CANONICAL_MANIFESTS.items():
    _x138_roots = [
        _X138Path('/kaggle/input/datasets/pilkwang') / _x138_slug,
        _X138Path('/kaggle/input') / _x138_slug,
    ]
    _x138_root = next((root for root in _x138_roots if root.is_dir()), None)
    if _x138_root is None:
        raise FileNotFoundError({'artifact': _x138_slug, 'searched': [str(p) for p in _x138_roots]})
    _x138_manifest_path = _x138_root / 'ARTIFACT_MANIFEST.json'
    _x138_actual = _x138_file_sha(_x138_manifest_path)
    if _x138_actual != _x138_expected:
        raise RuntimeError({'artifact': _x138_slug,
                            'expected': _x138_expected, 'actual': _x138_actual})
    _x138_artifacts[_x138_slug] = {'root': str(_x138_root),
                                   'manifest_sha256': _x138_actual}
_x138_primary = _X138Path(_x138_artifacts['biohub-tracking-support-pack-50ep-v1']['root'])
_x138_wheels = [
    {'path': path.relative_to(_x138_primary).as_posix(),
     'bytes': path.stat().st_size, 'sha256': _x138_file_sha(path)}
    for path in sorted(_x138_primary.rglob('*.whl'))
]
if not _x138_wheels:
    raise RuntimeError('Offline wheel set is empty')
_x138_test_movies = sorted(TEST_DIR.glob('*.zarr'))
if not _x138_test_movies:
    raise RuntimeError('No test movies')
_x138_test = {}
for _x138_movie in _x138_test_movies:
    _x138_files = sorted(path for path in _x138_movie.rglob('*') if path.is_file())
    _x138_metadata = [
        {'path': path.relative_to(_x138_movie).as_posix(),
         'bytes': path.stat().st_size, 'sha256': _x138_file_sha(path)}
        for path in _x138_files if path.name in {'zarr.json', '.zarray', '.zattrs', '.zgroup'}
    ]
    _x138_tree = [[path.relative_to(_x138_movie).as_posix(), path.stat().st_size]
                  for path in _x138_files]
    _x138_test[_x138_movie.stem] = {
        'files': len(_x138_files), 'bytes': sum(row[1] for row in _x138_tree),
        'path_size_manifest_sha256': _x138_json_sha(_x138_tree),
        'metadata': _x138_metadata,
    }
_X138_INPUT_MANIFEST = {
    'schema_version': 1,
    'reference_notebook_sha256': _X138_REFERENCE_SHA256,
    'expected_docker_image_sha256': _X138_DOCKER_IMAGE_SHA256,
    'python': _x138_sys.version, 'platform': _x138_platform.platform(),
    'torch': _x138_torch.__version__, 'cuda': _x138_torch.version.cuda,
    'gpu_names': _x138_gpu_names, 'canonical_artifacts': _x138_artifacts,
    'v1284_head': {'dataset_ref': _X138_HEAD_DATASET_REF,
                   'dataset_version_id': _X138_HEAD_DATASET_VERSION_ID,
                   'path': str(_x138_head_path), 'sha256': _x138_head_actual_sha},
    'offline_wheels': _x138_wheels, 'wheel_manifest_sha256': _x138_json_sha(_x138_wheels),
    'test_movies': _x138_test,
}
_X138_INPUT_MANIFEST['input_manifest_sha256'] = _x138_json_sha(_X138_INPUT_MANIFEST)
(WORKING_DIR / 'replay_input_manifest.json').write_text(
    _x138_json.dumps(_X138_INPUT_MANIFEST, indent=2, sort_keys=True) + '\n'
)
print('Replay input manifest:', _X138_INPUT_MANIFEST['input_manifest_sha256'])
# EXP042_ADDITION_END
"""

DIAGNOSTIC = r'''
# EXP042_ADDITION_START: instrument the generated V1284 module without changing its return.
_x138_refine_module = _ps.parent / 'v1284_coordinate_refinement.py'
_x138_diagnostic_source = """
import hashlib as _diag_hashlib
import json as _diag_json
_original_refine_for_replay = refine

def refine(ds_path, t, arr, feature):
    before = np.ascontiguousarray(arr.astype('<f4', copy=False))
    result = _original_refine_for_replay(ds_path, t, arr, feature)
    after = np.ascontiguousarray(result.astype('<f4', copy=False))
    record = {
        'dataset': ds_path.stem,
        't': int(t),
        'rows': int(len(arr)),
        'before_sha256': _diag_hashlib.sha256(before.tobytes()).hexdigest(),
        'after_sha256': _diag_hashlib.sha256(after.tobytes()).hexdigest(),
        'changed_rows': int(np.any(before != after, axis=1).sum()),
    }
    shard = os.environ.get('BIOHUB_GPU_SHARD', 'single').replace('/', '_')
    path = Path('/kaggle/working') / f'v1284_coordinates_{shard}.jsonl'
    with path.open('a') as handle:
        handle.write(_diag_json.dumps(record, sort_keys=True) + '\\n')
    return result
"""
with _x138_refine_module.open('a') as _x138_handle:
    _x138_handle.write('\n' + _x138_diagnostic_source)
print('V1284 pre/post-coordinate diagnostics installed')
# EXP042_ADDITION_END
'''

RECEIPT = r"""
# EXP042_ADDITION_START: read only final outputs and write a replay receipt.
import csv as _x138_csv

def _x138_records(pattern):
    records = []
    for path in sorted(WORKING_DIR.glob(pattern)):
        for line in path.read_text().splitlines():
            if line.strip():
                records.append(_x138_json.loads(line))
    return records

_x138_coordinate_records = sorted(
    _x138_records('v1284_coordinates_*.jsonl'),
    key=lambda row: (row['dataset'], int(row['t']), row['before_sha256'], row['after_sha256']),
)
_x138_expected_movies = sorted(_X138_INPUT_MANIFEST['test_movies'])
if not _x138_coordinate_records:
    raise RuntimeError('V1284 coordinate diagnostics are missing')
if sorted({row['dataset'] for row in _x138_coordinate_records}) != _x138_expected_movies:
    raise RuntimeError('V1284 diagnostics do not cover all test movies')
if sum(row['changed_rows'] for row in _x138_coordinate_records) <= 0:
    raise RuntimeError('V1284 candidate mode did not change any coordinate')
if _x138_os.environ.get('V1284_MODE') != 'candidate':
    raise RuntimeError('V1284 candidate mode was not active')

_x138_detector_records = sorted(
    _x138_records('detector_coordinates_*.jsonl'),
    key=lambda row: (row['dataset'], row['coordinate_sha256']),
)
if sorted({row['dataset'] for row in _x138_detector_records}) != _x138_expected_movies:
    raise RuntimeError('Detector coordinate manifests do not cover all test movies')
_x138_integrity = _x138_json.loads(
    (WORKING_DIR / 'bidirectional_production_runtime_integrity.json').read_text()
)
if _x138_integrity['checkpoint_sha256'] != _X138_CHECKPOINT_SHA256:
    raise RuntimeError('Public checkpoint SHA mismatch')
if _x138_integrity['support_repo_python_manifest_sha256'] != _X138_SUPPORT_SOURCE_SHA256:
    raise RuntimeError('Support source SHA mismatch')
_x138_guard = _x138_json.loads(
    (WORKING_DIR / 'dual_seed_frame_retention_guard_report.json').read_text()
)
_x138_submission = WORKING_DIR / 'submission.csv'
_x138_submission_sha = _x138_file_sha(_x138_submission)
if _x138_guard['submission']['sha256'] != _x138_submission_sha:
    raise RuntimeError('Public guard SHA differs from final submission')
_x138_stats = []
with RUN_STATS_PATH.open(newline='') as handle:
    for row in _x138_csv.DictReader(handle):
        _x138_stats.append({
            key: value for key, value in row.items()
            if key not in {'predict_minutes_total', 'repair_seconds', 'kernel_elapsed_seconds'}
        })
_x138_stats.sort(key=lambda row: row['dataset'])
_x138_receipt = {
    'schema_version': 1, 'experiment': 'exp042_public_x138_replay',
    'stage': 'public_test_full_replay',
    'reference_kernel': 'anvithpothula/biohub-x138',
    'reference_notebook_sha256': _X138_REFERENCE_SHA256,
    'input_manifest_sha256': _X138_INPUT_MANIFEST['input_manifest_sha256'],
    'wheel_manifest_sha256': _X138_INPUT_MANIFEST['wheel_manifest_sha256'],
    'v1284_head_sha256': _x138_head_actual_sha,
    'checkpoint_sha256': _x138_integrity['checkpoint_sha256'],
    'support_source_manifest_sha256': _x138_integrity['support_repo_python_manifest_sha256'],
    'test_datasets': _x138_expected_movies,
    'pre_refinement_coordinates_sha256': _x138_json_sha([
        {key: row[key] for key in ('dataset', 't', 'rows', 'before_sha256')}
        for row in _x138_coordinate_records
    ]),
    'post_refinement_coordinates_sha256': _x138_json_sha([
        {key: row[key] for key in ('dataset', 't', 'rows', 'after_sha256')}
        for row in _x138_coordinate_records
    ]),
    'v1284_changed_rows': sum(row['changed_rows'] for row in _x138_coordinate_records),
    'detector_coordinates_sha256': _x138_json_sha(_x138_detector_records),
    'graph_topology_sha256': _x138_json_sha(_x138_guard['topology']),
    'deterministic_run_statistics_sha256': _x138_json_sha(_x138_stats),
    'submission_sha256': _x138_submission_sha,
    'submission_rows': int(_x138_guard['submission']['rows']),
    'comparison_fields': [
        'reference_notebook_sha256', 'input_manifest_sha256',
        'wheel_manifest_sha256', 'v1284_head_sha256',
        'checkpoint_sha256', 'support_source_manifest_sha256', 'test_datasets',
        'pre_refinement_coordinates_sha256', 'post_refinement_coordinates_sha256',
        'v1284_changed_rows', 'detector_coordinates_sha256',
        'graph_topology_sha256', 'deterministic_run_statistics_sha256',
        'submission_sha256', 'submission_rows',
    ],
}
_x138_receipt['receipt_sha256'] = _x138_json_sha(_x138_receipt)
(WORKING_DIR / 'replay_receipt.json').write_text(
    _x138_json.dumps(_x138_receipt, indent=2, sort_keys=True) + '\n'
)
print(_x138_json.dumps(_x138_receipt, indent=2, sort_keys=True))
# EXP042_ADDITION_END
"""


def cell_source(cell: dict[str, object]) -> str:
    value = cell["source"]
    return "".join(value) if isinstance(value, list) else str(value)


def markdown(title: str) -> str:
    return (
        "# %% [markdown]\n"
        + "\n".join("# " + line if line else "#" for line in title.splitlines())
        + "\n\n"
    )


def code(source: str) -> str:
    if not source.endswith("\n"):
        source += "\n"
    return "# %%\n" + source + "\n"


def main() -> None:
    if hashlib.sha256(REFERENCE.read_bytes()).hexdigest() != REFERENCE_SHA256:
        raise RuntimeError("Pinned x138 reference notebook SHA256 changed")
    notebook = json.loads(REFERENCE.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    if len(cells) != 12 or any(cell["cell_type"] != "code" for cell in cells):
        raise RuntimeError("Unexpected x138 reference notebook cell structure")
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    head = config["data"]["v1284_head"]
    canonical = {
        item["ref"].split("/")[-1]: item["manifest_sha256"]
        for item in config["data"]["canonical_artifacts"]
    }
    preflight = (
        PREFLIGHT.replace("__HEAD_SHA__", repr(head["sha256"] or ""))
        .replace("__HEAD_REF__", repr(head["dataset_ref"] or ""))
        .replace("__HEAD_VERSION_ID__", repr(head["version_id"] or ""))
        .replace("__REFERENCE_SHA__", repr(REFERENCE_SHA256))
        .replace("__CANONICAL_MANIFESTS__", repr(canonical))
        .replace("__CHECKPOINT_SHA__", repr(config["model"]["checkpoint_sha256"]))
        .replace("__SUPPORT_SOURCE_SHA__", repr(config["source"]["support_source_manifest_sha256"]))
    )
    pieces = [
        markdown(
            "# exp042 public x138 replay\n\n"
            "## Contents\n"
            "1. Input and V1284 checkpoint guard\n"
            "2. Public x138 configuration and runtime checks\n"
            "3. Detector, tracker, and V1284 coordinate refinement\n"
            "4. Graph repair and submission generation\n"
            "5. Replay input and output receipts"
        ),
        code(preflight),
    ]
    sections = {
        0: "## 2. Public x138 configuration and runtime checks",
        2: "## 5. Replay input manifest",
        4: "## 3. Detector, tracker, and V1284 coordinate refinement",
        6: "## 4. Graph repair and submission generation",
    }
    for index, cell in enumerate(cells):
        if index in sections:
            pieces.append(markdown(sections[index]))
        source = cell_source(cell)
        if "\n# %%" in source:
            raise RuntimeError(f"Embedded Jupytext cell marker in public cell {index}")
        if index == 4:
            if source.count(SPLIT_ANCHOR) != 1:
                raise RuntimeError("V1284 diagnostic insertion anchor changed")
            before, after = source.split(SPLIT_ANCHOR, 1)
            pieces.extend([code(before), code(DIAGNOSTIC), code(SPLIT_ANCHOR + after)])
        else:
            pieces.append(code(source))
        if index == 2:
            pieces.append(code(INPUT_MANIFEST))
    pieces.extend([markdown("## 5. Replay output receipt"), code(RECEIPT)])
    OUTPUT.write_text("".join(pieces), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
