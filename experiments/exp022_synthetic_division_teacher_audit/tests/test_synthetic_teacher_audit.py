from pathlib import Path
from runpy import run_path

import numpy as np
import pytest
import yaml

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
FUNCTIONS = run_path(
    str(EXPERIMENT_DIR / "exp022_synthetic_division_teacher_audit_diagnostic.py"),
    run_name="synthetic_teacher_test",
)


def small_complete_lineage():
    # One division and one continuation at t=0; all three descendants continue.
    nodes = []
    edges = []
    nodes.extend([[0, 0, 0, 0, 1], [0, 0, 12, 0, 2]])
    previous = [0, 1]
    for t in range(1, 6):
        start = len(nodes)
        nodes.extend([[t, 0, -2, 0, 1], [t, 0, 2, 0, 1], [t, 0, 12, 0, 2]])
        current = [start, start + 1, start + 2]
        if t == 1:
            edges.extend([[0, current[0]], [0, current[1]], [1, current[2]]])
        else:
            edges.extend(zip(previous, current, strict=True))
        previous = current
    return (
        np.asarray(nodes, dtype=np.float32),
        np.asarray(edges, dtype=np.int32),
        np.array([0], dtype=np.int32),
    )


def test_complete_lineage_labels_division_and_continuation_differently(tmp_path):
    nodes, edges, divisions = small_complete_lineage()
    path = tmp_path / "seq_0000.npz"
    np.savez_compressed(
        path,
        nodes=nodes,
        edges=edges,
        divisions=divisions,
        volumes=np.zeros((6, 64, 64, 64), dtype=np.uint16),
        voxel_um_pooled=np.array([1.625, 1.625, 1.625], dtype=np.float32),
    )
    row = {
        "file": path.name,
        "T": 6,
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "n_divisions": len(divisions),
        "bytes": path.stat().st_size,
    }
    config = yaml.safe_load((EXPERIMENT_DIR / "config.yaml").read_text())
    result, distances = FUNCTIONS["audit_sequence"](path, row, config)
    assert result["division_mothers"] == 1
    assert result["division_recovered"] == result["positive_triplets"] == 1
    assert result["division_with_wrong_triplet"] == 1
    assert result["recovered_division_with_wrong_triplet"] == 1
    assert result["wrong_division_triplets"] == 2
    assert result["continuation_mothers"] == 13
    assert result["continuation_triplets"] > 0
    assert result["candidate_triplets"] == sum(
        result[k] for k in ("positive_triplets", "wrong_division_triplets", "continuation_triplets")
    )
    assert len(distances["mother_daughter_um"]) == 2
    assert len(distances["sister_um"]) == 1


def test_lineage_validation_rejects_missing_or_duplicate_parent():
    nodes, edges, divisions = small_complete_lineage()
    validate = FUNCTIONS["validate_graph"]
    with pytest.raises(ValueError, match="incoming|one or two daughters"):
        validate(nodes, edges[:-1], divisions, 6)
    broken = edges.copy()
    broken[-1, 0] = broken[-2, 0]
    with pytest.raises(ValueError, match="incoming|one or two daughters|two-daughter"):
        validate(nodes, broken, divisions, 6)
