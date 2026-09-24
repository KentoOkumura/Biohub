"""Build x138 inference with the SHA-pinned self-trained V1284-shaped head."""

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
OUTPUT = ROOT / "exp043_x138_self_trained_head_inference.py"
REFERENCE_SHA = "6b655e39bbfd2d3d6c762badea69847d3f00f5b548f385cb01b07ee2600fde6d"

PREFLIGHT = r"""
import hashlib as _self_hashlib
import json as _self_json
from pathlib import Path as _SelfPath

_SELF_EXPECTED_SHA = '__HEAD_SHA__'
_SELF_HEAD_MATCHES = sorted(_SelfPath('/kaggle/input').rglob('self_trained_v1284_head.pt'))
if len(_SELF_HEAD_MATCHES) != 1:
    raise RuntimeError({'self_trained_head_matches': [str(p) for p in _SELF_HEAD_MATCHES]})
_SELF_HEAD_PATH = _SELF_HEAD_MATCHES[0]
_SELF_HEAD_SHA = _self_hashlib.sha256(_SELF_HEAD_PATH.read_bytes()).hexdigest()
if _SELF_HEAD_SHA != _SELF_EXPECTED_SHA:
    raise RuntimeError({'expected_head_sha': _SELF_EXPECTED_SHA, 'actual': _SELF_HEAD_SHA})
print('Self-trained coordinate head:', _SELF_HEAD_PATH, _SELF_HEAD_SHA, flush=True)
"""

RECEIPT = r"""
import pandas as _self_pd
_self_submission = _SelfPath('/kaggle/working/submission.csv')
if not _self_submission.is_file():
    raise FileNotFoundError(_self_submission)
_self_frame = _self_pd.read_csv(_self_submission)
_self_columns = ['id', 'dataset', 'row_type', 'node_id', 't', 'z', 'y', 'x',
                 'source_id', 'target_id']
if _self_frame.columns.tolist() != _self_columns:
    raise RuntimeError('Submission columns changed')
if sorted(_self_frame.dataset.unique().tolist()) != sorted(test_stems):
    raise RuntimeError('Public test video coverage changed')
if _self_frame['id'].tolist() != list(range(len(_self_frame))):
    raise RuntimeError('Non-contiguous submission IDs')
_self_receipt = {
    'experiment': 'exp043_x138_self_trained_head',
    'head_sha256': _SELF_HEAD_SHA,
    'public_test_video_count': len(test_stems),
    'public_test_rows': len(_self_frame),
    'submission_sha256': _self_hashlib.sha256(_self_submission.read_bytes()).hexdigest(),
    'prediction_seconds': predict_seconds,
    'official_score': None,
    'competition_submission_created': False,
}
(_SelfPath('/kaggle/working') / 'self_trained_x138_receipt.json').write_text(
    _self_json.dumps(_self_receipt, indent=2, sort_keys=True) + '\n')
print(_self_json.dumps(_self_receipt, indent=2, sort_keys=True), flush=True)
"""


def build() -> None:
    if hashlib.sha256(REFERENCE.read_bytes()).hexdigest() != REFERENCE_SHA:
        raise RuntimeError("Pinned x138 notebook SHA256 mismatch")
    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    head_sha = config["model"]["params"]["checkpoint_sha256"]
    if len(head_sha) != 64:
        raise RuntimeError("Self-trained head SHA is not pinned")
    notebook = json.loads(REFERENCE.read_text())
    source = ["".join(cell["source"]) for cell in notebook["cells"]]
    if len(source) != 12:
        raise RuntimeError("x138 notebook cell structure changed")
    patched = source[4]
    guard_start = patched.index("_myhead = sorted(Path('/kaggle/input').rglob(")
    guard_end = patched.index("\n_trial_source = _ps.read_text()", guard_start)
    patched = (
        patched[:guard_start]
        + ("os.environ['V1284_MODE']='candidate'\nos.environ['V1284_HEAD']=str(_SELF_HEAD_PATH)\n")
        + patched[guard_end:]
    )
    cells = [
        ("markdown", "# exp043 x138 inference with a self-trained coordinate head"),
        ("markdown", "## 1. Public x138 configuration and fixed artifacts"),
        *(("code", cell) for cell in source[:4]),
        ("markdown", "## 2. Verify the self-trained head input"),
        ("code", PREFLIGHT.replace("__HEAD_SHA__", head_sha)),
        ("markdown", "## 3. Run x138 with the self-trained coordinate head"),
        ("code", patched),
        ("markdown", "## 4. Original graph postprocessing and output checks"),
        *(("code", cell) for cell in source[5:]),
        ("markdown", "## 5. Output receipt"),
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
