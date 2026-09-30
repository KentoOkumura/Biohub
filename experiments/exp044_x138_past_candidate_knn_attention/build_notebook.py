"""Build exp044 train from exp043's pinned x138 inference setup."""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
PARENT = (
    ROOT.parent / "exp043_x138_self_trained_head" / "exp043_x138_self_trained_head_inference.py"
)
OUTPUT = ROOT / "exp044_x138_past_candidate_knn_attention_train.py"

SELECTION = """
def _exp044_rank(stem):
    seed_text = f"{EXP044_CONFIG['data']['selection_seed']}:{stem}"
    return _self_hashlib.sha256(seed_text.encode()).hexdigest()

_exp044_groups = {}
for _embryo in EXP044_CONFIG['data']['expected_embryos']:
    _stems = [p.stem for p in TEST_DIR.glob('*.zarr')
              if p.stem.split('_', 1)[0] == _embryo
              and (TEST_DIR / f'{p.stem}.geff').is_dir()]
    _stems.sort(key=lambda stem: (_exp044_rank(stem), stem))
    _required = int(EXP044_CONFIG['data']['videos_per_embryo'])
    if len(_stems) < _required:
        raise RuntimeError({'embryo': _embryo, 'eligible': len(_stems)})
    _exp044_groups[_embryo] = _stems[:_required]
test_stems = sorted(stem for group in _exp044_groups.values() for stem in group)
(WORKING_DIR / 'selected_train_videos.json').write_text(
    json.dumps(_exp044_groups, indent=2, sort_keys=True) + '\\n')
"""

TRAILER = """
# %% [markdown]
# ## 2. Capture exp043 pair inputs before graph inference

# %%
from capture_patch import patch_x138_pair_capture
from attention_train_pipeline import train_from_capture

_ps.write_text(patch_x138_pair_capture(_ps.read_text()))
_exp044_cache = WORKING_DIR / 'tracker_capture'
_exp044_cache.mkdir(exist_ok=True)
os.environ['EXP044_TRACKER_CAPTURE_DIR'] = str(_exp044_cache)
os.environ['BIOHUB_CACHE_DIR'] = ''
os.environ['BIOHUB_DIAGNOSTIC_ARM'] = ''

def _exp044_capture(stems, label):
    splits_path.write_text(json.dumps([{'split': 0, 'train': [], 'test': stems}], indent=2))
    count = min(2, _torch.cuda.device_count(), len(stems))
    if count != 2:
        raise RuntimeError('exp044 capture requires two Kaggle GPUs')
    tokens = _visible_cuda_tokens(count)
    processes, commands = {}, {}
    started = time.time()
    for shard in range(count):
        command = [*predict_cmd, '--method', f'{METHOD}_exp044_{label}_gpu{shard}',
                   '--slice', f'{shard}::{count}']
        env = {**os.environ, 'PYTHONPATH': 'src', 'CUDA_VISIBLE_DEVICES': tokens[shard],
               'BIOHUB_GPU_SHARD': f'{shard}/{count}'}
        processes[shard] = subprocess.Popen(command, cwd=REPO_DIR, env=env)
        commands[shard] = command
    _wait_for_prediction_shards(processes, commands)
    for stem in stems:
        if not any((_exp044_cache / stem).glob('*.npz')):
            raise RuntimeError(f'No pair windows captured for {stem}')
    return time.time() - started

_exp044_pilot = sorted(group[0] for group in _exp044_groups.values())
_exp044_pilot_seconds = _exp044_capture(_exp044_pilot, 'pilot')
_exp044_projected_minutes = _exp044_pilot_seconds / 60 * len(test_stems) / len(_exp044_pilot)
_exp044_cap = float(EXP044_CONFIG['runtime']['capture_budget_minutes'])
print('Capture projected minutes:', _exp044_projected_minutes, 'limit:', _exp044_cap, flush=True)
if _exp044_projected_minutes > _exp044_cap:
    raise RuntimeError('Exp044 capture projection exceeds configured limit')
_exp044_remaining = [stem for stem in test_stems if stem not in _exp044_pilot]
_exp044_capture_seconds = _exp044_pilot_seconds + _exp044_capture(_exp044_remaining, 'remaining')
if _exp044_capture_seconds / 60 > _exp044_cap:
    raise RuntimeError('Exp044 capture exceeded configured limit')

# %% [markdown]
# ## 3. Baseline parity, K=8 audit, tracker training and pair evaluation

# %%
train_from_capture(
    config=EXP044_CONFIG,
    cache_root=_exp044_cache,
    train_dir=TEST_DIR,
    public_repo=REPO_DIR,
    working_dir=WORKING_DIR,
    groups=_exp044_groups,
    capture_seconds=_exp044_capture_seconds,
)
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
            "# # exp044 x138 primary tracker and K=8 past-candidate attention"
        ),
        'TEST_DIR = COMP_DIR / "test"': 'TEST_DIR = COMP_DIR / "train"',
        "test_stems = list_test_stems()": SELECTION,
    }
    for old, new in replacements.items():
        if prefix.count(old) != 1:
            raise RuntimeError(f"exp043 anchor count differs: {old[:70]}")
        prefix = prefix.replace(old, new, 1)
    config_load = "import yaml\nEXP044_CONFIG = yaml.safe_load(Path('config.yaml').read_text())\n"
    # The x138 setup imports yaml before source patching.
    marker = "def _exp044_rank(stem):"
    if prefix.count(marker) != 1:
        raise RuntimeError("selection insertion failed")
    prefix = prefix.replace(marker, config_load + "\n" + marker, 1)
    OUTPUT.write_text(prefix + TRAILER, encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    build()
