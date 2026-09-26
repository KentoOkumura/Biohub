"""Build exp048 train from exp043's pinned x138 inference setup."""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
PARENT = (
    ROOT.parent / "exp043_x138_self_trained_head" / "exp043_x138_self_trained_head_inference.py"
)
OUTPUT = ROOT / "exp048_x138_primary_past_feature_attention_train.py"

SELECTION = """
def _exp048_rank(stem):
    seed_text = f"{EXP048_CONFIG['data']['selection_seed']}:{stem}"
    return _self_hashlib.sha256(seed_text.encode()).hexdigest()

_exp048_groups = {}
for _embryo in EXP048_CONFIG['data']['expected_embryos']:
    _stems = [p.stem for p in TEST_DIR.glob('*.zarr')
              if p.stem.split('_', 1)[0] == _embryo
              and (TEST_DIR / f'{p.stem}.geff').is_dir()]
    _stems.sort(key=lambda stem: (_exp048_rank(stem), stem))
    _required = int(EXP048_CONFIG['data']['videos_per_embryo'])
    if len(_stems) < _required:
        raise RuntimeError({'embryo': _embryo, 'eligible': len(_stems)})
    _exp048_groups[_embryo] = _stems[:_required]
test_stems = sorted(stem for group in _exp048_groups.values() for stem in group)
(WORKING_DIR / 'selected_train_videos.json').write_text(
    json.dumps(_exp048_groups, indent=2, sort_keys=True) + '\\n')
"""

TRAILER = """
# %% [markdown]
# ## 4. Capture exp043 pair inputs before graph inference

# %%
from capture_patch import patch_x138_pair_capture
from attention_train_pipeline import train_from_capture

_ps.write_text(patch_x138_pair_capture(_ps.read_text()))
_exp048_cache = WORKING_DIR / 'tracker_capture'
_exp048_cache.mkdir(exist_ok=True)
os.environ['EXP048_TRACKER_CAPTURE_DIR'] = str(_exp048_cache)
os.environ['BIOHUB_CACHE_DIR'] = ''
os.environ['BIOHUB_DIAGNOSTIC_ARM'] = ''

def _exp048_capture(stems, label):
    splits_path.write_text(json.dumps([{'split': 0, 'train': [], 'test': stems}], indent=2))
    count = min(2, _torch.cuda.device_count(), len(stems))
    if count != 2:
        raise RuntimeError('exp048 capture requires two Kaggle GPUs')
    tokens = _visible_cuda_tokens(count)
    processes, commands = {}, {}
    started = time.time()
    for shard in range(count):
        command = [*predict_cmd, '--method', f'{METHOD}_exp048_{label}_gpu{shard}',
                   '--slice', f'{shard}::{count}']
        env = {**os.environ, 'PYTHONPATH': 'src', 'CUDA_VISIBLE_DEVICES': tokens[shard],
               'BIOHUB_GPU_SHARD': f'{shard}/{count}'}
        processes[shard] = subprocess.Popen(command, cwd=REPO_DIR, env=env)
        commands[shard] = command
    _wait_for_prediction_shards(processes, commands)
    for stem in stems:
        if not any((_exp048_cache / stem).glob('*.npz')):
            raise RuntimeError(f'No pair windows captured for {stem}')
    return time.time() - started

_exp048_pilot = sorted(group[0] for group in _exp048_groups.values())
_exp048_pilot_seconds = _exp048_capture(_exp048_pilot, 'pilot')
_exp048_projected_minutes = _exp048_pilot_seconds / 60 * len(test_stems) / len(_exp048_pilot)
_exp048_cap = float(EXP048_CONFIG['runtime']['capture_budget_minutes'])
print('Capture projected minutes:', _exp048_projected_minutes, 'limit:', _exp048_cap, flush=True)
if _exp048_projected_minutes > _exp048_cap:
    raise RuntimeError('Exp048 capture projection exceeds configured limit')
_exp048_remaining = [stem for stem in test_stems if stem not in _exp048_pilot]
_exp048_capture_seconds = _exp048_pilot_seconds + _exp048_capture(_exp048_remaining, 'remaining')
if _exp048_capture_seconds / 60 > _exp048_cap:
    raise RuntimeError('Exp048 capture exceeded configured limit')

# %% [markdown]
# ## 5. Baseline parity, K=8 audit, two training modes and pair evaluation

# %%
_exp048_modes = EXP048_CONFIG['model']['training']['active_modes']
assert _exp048_modes == ['primary_only', 'primary_with_past_feature_attention']
print('Parent:', EXP048_CONFIG['lineage']['parent'], flush=True)
print('Held-out embryos:', list(_exp048_groups), flush=True)
print('Training modes:', _exp048_modes, flush=True)
print('Epochs:', EXP048_CONFIG['model']['training']['epochs'], flush=True)
print('Pair capture root:', _exp048_cache, flush=True)
train_from_capture(
    config=EXP048_CONFIG,
    cache_root=_exp048_cache,
    train_dir=TEST_DIR,
    public_repo=REPO_DIR,
    working_dir=WORKING_DIR,
    groups=_exp048_groups,
    capture_seconds=_exp048_capture_seconds,
)
for _exp048_name in ('candidate_retention.json', 'baseline_parity.json',
                     'runtime_forecast.json', 'pair_metrics.json',
                     'graph_progression_gate.json', 'train_receipt.json'):
    _exp048_artifact = WORKING_DIR / _exp048_name
    print(_exp048_name, _exp048_artifact.exists(), flush=True)
"""


def build() -> None:
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    source = PARENT.read_text(encoding="utf-8")
    if hashlib.sha256(source.encode()).hexdigest() != config["source"]["exp043_inference_sha256"]:
        raise RuntimeError("Pinned exp043 inference source SHA differs")
    anchor = "start_time = time.time()\n"
    if source.count(anchor) != 1:
        raise RuntimeError("exp043 execution boundary changed")
    prefix = source.split(anchor, 1)[0]
    replacements = {
        "# # exp043 x138 inference with a self-trained coordinate head": (
            "# # exp048 x138 primary pair-MLP past-feature attention with matched control\n"
            "#\n"
            "# ## Contents\n"
            "# 1. Public x138 configuration and fixed artifacts\n"
            "# 2. Verify the self-trained coordinate head\n"
            "# 3. Prepare x138 predictor\n"
            "# 4. Capture exp043 pair inputs\n"
            "# 5. Baseline parity, two training modes, pair evaluation and artifacts"
        ),
        'TEST_DIR = COMP_DIR / "test"': 'TEST_DIR = COMP_DIR / "train"',
        "test_stems = list_test_stems()": SELECTION,
    }
    for old, new in replacements.items():
        if prefix.count(old) != 1:
            raise RuntimeError(f"exp043 anchor count differs: {old[:70]}")
        prefix = prefix.replace(old, new, 1)
    config_load = "import yaml\nEXP048_CONFIG = yaml.safe_load(Path('config.yaml').read_text())\n"
    # The x138 setup imports yaml before source patching.
    marker = "def _exp048_rank(stem):"
    if prefix.count(marker) != 1:
        raise RuntimeError("selection insertion failed")
    prefix = prefix.replace(marker, config_load + "\n" + marker, 1)
    OUTPUT.write_text(prefix + TRAILER, encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    build()
