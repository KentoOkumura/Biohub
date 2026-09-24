"""The self-trained head changes the missing V1284 input, not x138 graph code."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = (
    ROOT.parents[1]
    / "docs/notebooks"
    / "biohub-cell-tracking-during-development"
    / "anvithpothula__biohub-x138"
    / "biohub-x138.ipynb"
)
TRAIN = ROOT / "exp043_x138_self_trained_head_train.ipynb"
INFERENCE = ROOT / "exp043_x138_self_trained_head_inference.ipynb"


def code(path: Path) -> list[str]:
    notebook = json.loads(path.read_text())
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


def test_train_only_captures_features_and_saves_the_head() -> None:
    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    assert (
        hashlib.sha256(REFERENCE.read_bytes()).hexdigest()
        == config["source"]["reference_notebook_sha256"]
    )
    reference = code(REFERENCE)
    generated = code(TRAIN)
    assert [cell.rstrip() for cell in generated[:4]] == [cell.rstrip() for cell in reference[:4]]
    assert len(generated) == 10
    capture_source = generated[5]
    assert "os.environ['V1284_MODE']='capture'" in capture_source
    assert "v1284_coordinate_refinement.py" in capture_source
    assert "_myhead = sorted" not in capture_source
    assert "_v1284_refine(ds_path, t, arr, unet_out[:, f_idx])" in capture_source
    assert "self_trained_v1284_head.pt" in generated[8]
    assert "self_trained_head_train_receipt.json" in generated[-1]
    assert all("TEST_DIR = PUBLIC_TEST_DIR" not in cell for cell in generated)
    assert all("V1284_MODE'] = 'candidate'" not in cell for cell in generated)


def test_inference_uses_pinned_self_trained_head_and_public_graph_cells() -> None:
    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    reference = code(REFERENCE)
    generated = code(INFERENCE)
    assert [cell.rstrip() for cell in generated[:4]] == [cell.rstrip() for cell in reference[:4]]
    assert [cell.rstrip() for cell in generated[-8:-1]] == [cell.rstrip() for cell in reference[5:]]
    assert config["model"]["params"]["checkpoint_sha256"] in generated[4]
    assert "_myhead = sorted" not in generated[5]
    assert "os.environ['V1284_MODE']='candidate'" in generated[5]
    assert "os.environ['V1284_HEAD']=str(_SELF_HEAD_PATH)" in generated[5]
    assert "self_trained_x138_receipt.json" in generated[-1]
