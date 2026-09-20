from __future__ import annotations

import json
from pathlib import Path

import yaml

EXP = Path(__file__).resolve().parents[1]
NOTEBOOK_STEM = "exp030_frame_self_attention_diagnostics_diagnostic"


def test_diagnostic_contract_and_embedded_model_match() -> None:
    config = yaml.safe_load((EXP / "config.yaml").read_text(encoding="utf-8"))
    assert config["experiment"]["notebooks"] == ["diagnostic"]
    assert config["model"]["epochs"] == 0
    assert config["model"]["variants"] == {
        "current": {
            "use_temporal_self_attention": False,
            "use_cross_attention": True,
        },
        "model_a": {
            "use_temporal_self_attention": True,
            "use_cross_attention": False,
        },
        "model_b": {
            "use_temporal_self_attention": True,
            "use_cross_attention": True,
        },
    }
    source = (EXP / "frame_attention_model.py").read_text(encoding="utf-8").strip()
    notebook_source = (EXP / f"{NOTEBOOK_STEM}.py").read_text(encoding="utf-8")
    embedded = (
        notebook_source.split("# BEGIN EMBEDDED MODEL\n", 1)[1]
        .split("\n# END EMBEDDED MODEL", 1)[0]
        .strip()
    )
    assert embedded == source
    notebook = json.loads((EXP / f"{NOTEBOOK_STEM}.ipynb").read_text(encoding="utf-8"))
    assert any(
        "class SimpleNodeTransformer" in "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert "frame_self_attention_diagnostic.json" in notebook_source
    assert "official_graph_score_computed" in notebook_source
