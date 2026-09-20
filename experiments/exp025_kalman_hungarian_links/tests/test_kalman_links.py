from __future__ import annotations

import importlib.util
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

EXP = Path(__file__).resolve().parents[1]
SOURCE = EXP / f"{EXP.name}_diagnostic.py"
spec = importlib.util.spec_from_file_location("exp025_diagnostic", SOURCE)
assert spec and spec.loader
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)
CFG = yaml.safe_load((EXP / "config.yaml").read_text())
PARAMS = {**CFG["model"]["params"], **CFG["validation"]}
EMPTY = np.empty((0, 2), dtype=np.int64)


def noise():
    return {
        "observation_variance": [0.01] * 3,
        "acceleration_variance": [0.01] * 3,
        "initial_velocity_variance": [10.0] * 3,
        "daughter_covariance_multiplier": 4.0,
        "nll_center": 0.0,
        "nll_scale": 1.5,
    }


def graph(frames, positions, edges, probabilities=None, ids=None):
    positions = np.asarray(positions, dtype=float)
    if positions.ndim == 1:
        positions = np.column_stack([positions, np.zeros((len(positions), 2))])
    return m.Graph(
        np.asarray(ids if ids is not None else range(len(frames)), dtype=np.int64),
        np.asarray(frames),
        positions,
        np.asarray(edges, dtype=np.int64).reshape(-1, 2),
        np.asarray(probabilities if probabilities is not None else [0.9] * len(edges)),
    )


def decode(g, reserved=EMPTY, weight=0.0, null=2.0):
    return m.decode(g, reserved, noise(), motion_weight=weight, no_match_cost=null, cfg=PARAMS)


def test_assignment_matches_bruteforce_and_never_uses_absent_edges():
    edges = [(0, 10), (0, 11), (1, 10), (2, 12)]
    costs = [0.1, 0.2, 0.3, 2.0]
    null = 0.8
    result = m.assign_with_no_match(edges, costs, null, 100000)

    def objective(selected):
        return sum(costs[edges.index(e)] for e in selected) + (3 - len(selected)) * null

    feasible = []
    for n in range(5):
        for chosen in itertools.combinations(edges, n):
            if len({a for a, _ in chosen}) == n and len({b for _, b in chosen}) == n:
                feasible.append(chosen)
    assert objective(result) == pytest.approx(min(map(objective, feasible)))
    assert result == [(0, 11), (1, 10)]


def test_assignment_rectangular_disconnected_empty_and_memory_gate():
    assert m.assign_with_no_match([], [], 1, 8) == []
    assert m.assign_with_no_match([(3, 7), (3, 8)], [0.1, 0.2], 1, 1000) == [(3, 7)]
    with pytest.raises(MemoryError):
        m.assign_with_no_match([(3, 7)], [0.1], 1, 1)
    with pytest.raises(ValueError):
        m.assign_with_no_match([(3, 7)], [np.nan], 1, 1000)


def test_logdet_penalizes_uncertainty_at_same_predicted_center():
    state = m.initial_state(np.zeros(3), 0, noise())
    broad = m.State(0, state.mean.copy(), state.covariance * 100, 1)
    assert m.innovation_cost(broad, np.zeros(3), noise()) > m.innovation_cost(
        state, np.zeros(3), noise()
    )


def test_kalman_updates_velocity_and_stays_positive_definite():
    state = m.initial_state(np.zeros(3), 0, noise())
    for frame in range(1, 100):
        prediction = m.predict(state, noise())
        state = m.update(prediction, np.array([frame * 2.0, frame * -0.5, 0]), noise())
        assert np.linalg.eigvalsh(state.covariance).min() > 0
    np.testing.assert_allclose(state.mean[3:], [2, -0.5, 0], atol=1e-6)
    assert state.history == 100


def test_crossing_uses_self_predicted_velocity_and_preserves_centers():
    g = graph(
        [0, 0, 1, 1, 2, 2],
        [-3, 3, -1, 1, -0.9, 0.9],
        [(0, 2), (1, 3), (2, 4), (2, 5), (3, 4), (3, 5)],
        [0.99, 0.99, 0.9, 0.8, 0.8, 0.9],
    )
    before = g.digest()
    image = decode(g)
    motion = decode(g, weight=0.1, null=5)
    assert (2, 4) in set(map(tuple, image["edges"]))
    assert {(2, 5), (3, 4)} <= set(map(tuple, motion["edges"]))
    assert motion["track_ids"][0] == motion["track_ids"][5]
    assert motion["histories"][5] == 3
    assert g.digest() == before


def test_no_match_terminates_and_does_not_reconnect_over_missing_frame():
    g = graph([0, 1, 3], [0, 20, 0], [(0, 1)], [0.1])
    result = decode(g, null=0.1)
    assert len(result["edges"]) == 0
    assert len(set(result["track_ids"])) == 3
    np.testing.assert_array_equal(result["histories"], [1, 1, 1])
    np.testing.assert_array_equal(result["means"][:, 3:], np.zeros((3, 3)))
    assert decode(g)["track_ids"][0] == 0  # new video resets identity


def test_reserved_division_owns_capacity_and_daughters_inherit_velocity():
    g = graph(
        [0, 1, 2, 2, 2, 3],
        [0, 1, 1.8, 2.2, 10, 3],
        [(0, 1), (1, 2), (1, 3), (1, 4), (2, 5), (3, 5)],
    )
    reserved = np.asarray([[1, 2], [1, 3]], dtype=np.int64)
    result = decode(g, reserved)
    edges = set(map(tuple, result["edges"]))
    assert {(1, 2), (1, 3)} <= edges and (1, 4) not in edges
    assert len({result["track_ids"][i] for i in (1, 2, 3)}) == 3
    np.testing.assert_allclose(result["means"][2, 3:], result["means"][1, 3:])
    assert result["covariances"][2, 3, 3] > result["covariances"][1, 3, 3]
    assert sum(b == 5 for _, b in edges) == 1
    with pytest.raises(ValueError, match="lost reserved"):
        m.validate_output(g, reserved, np.asarray([[1, 2]]))


def test_reserved_only_saved_node_is_not_added_to_ordinary_support():
    candidate = graph([0, 1], [0, 1], [(0, 1)])
    reference = graph([0, 1, 1], [0, 1, -1], [(0, 1), (0, 2)])
    combined, reserved, receipt = m.reserve_divisions(candidate, reference)
    np.testing.assert_array_equal(combined.edges, candidate.edges)
    np.testing.assert_array_equal(combined.positions[:2], candidate.positions)
    assert receipt["reserved_only_node_ids"] == [2]
    assert receipt["reserved_edges_outside_ordinary_support"] == 1
    result = decode(combined, reserved)
    assert set(map(tuple, result["edges"])) == {(0, 1), (0, 2)}


def test_reservation_rejects_multi_parent_or_nonadjacent_edges():
    g = graph([0, 1, 1], [0, 1, 2], [(0, 1), (0, 2)])
    with pytest.raises(ValueError):
        m.validate_output(g, EMPTY, np.asarray([[0, 1], [0, 2]]))
    invalid = graph([0, 2], [0, 2], [(0, 1)])
    with pytest.raises(ValueError, match="non-adjacent"):
        invalid.validate()


def test_noise_uses_only_training_side_and_requires_evidence():
    truth = graph(range(5), [0, 1, 3, 6, 10], [(0, 1), (1, 2), (2, 3), (3, 4)])
    candidate = graph(range(5), [0.1, 1.2, 3.3, 6.4, 10.5], truth.edges)
    cfg = {**PARAMS, "minimum_noise_observations": 2}
    fitted = m.fit_noise([("44b6_one", candidate, truth)], "44b6", cfg)
    assert fitted["observation_variance"][0] == pytest.approx(
        np.mean(np.square([0.1, 0.2, 0.3, 0.4, 0.5]))
    )
    assert fitted["acceleration_variance"][0] == pytest.approx(1)
    assert fitted["counts"]["accelerations"] == 3
    with pytest.raises(ValueError, match="outer embryo"):
        m.fit_noise([("6bba_one", candidate, truth)], "44b6", cfg)
    with pytest.raises(ValueError, match="insufficient"):
        m.fit_noise([], "44b6", cfg)


def test_parent_split_and_asset_checksums_are_pinned():
    validation, data = CFG["validation"], CFG["data"]
    manifest_path = m.require_sha(EXP / data["manifest_file"], data["manifest_sha256"])
    manifest = json.loads(manifest_path.read_text())
    splits = json.loads(
        m.require_sha(EXP / validation["split_file"], validation["split_sha256"]).read_text()
    )
    samples = [s for a in manifest["archives"] for s in a["samples"]]
    for split in splits:
        m.validate_split(split, samples)
    assert len(samples) == 199 and len(set(samples)) == 199
    splits[0]["internal_validation"].append(splits[0]["outer_evaluation"][0])
    with pytest.raises(ValueError):
        m.validate_split(splits[0], samples)


def test_calibration_keeps_no_match_fixed_between_variants():
    cfg = {"no_match_cost_grid": [0.4, 0.8], "motion_weight_grid": [0.25, 0.5]}
    rows = [
        {"motion_weight": 0, "no_match_cost": 0.4, "combined_score": 0.8},
        {"motion_weight": 0, "no_match_cost": 0.8, "combined_score": 0.9},
        {"motion_weight": 0.25, "no_match_cost": 0.8, "combined_score": 0.95},
        {"motion_weight": 0.5, "no_match_cost": 0.8, "combined_score": 0.95},
    ]
    assert m.choose_costs(rows, cfg) == {"no_match_cost": 0.8, "motion_weight": 0.25}
    with pytest.raises(ValueError, match="incomplete"):
        m.choose_costs(rows[:-1], cfg)


def test_unknown_annotation_is_not_counted_as_known_wrong():
    g = graph([0, 1, 1], [0, 1, 20], [(0, 2)])
    truth = graph([0, 1], [0, 1], [(0, 1)])
    rows = m.diagnostic_counts(g, truth, decode(g), PARAMS)
    assert sum(r.get("unknown_selected_edges", 0) for r in rows) == 1
    assert sum(r.get("known_wrong_edges", 0) for r in rows) == 0


def test_full_execution_guard_and_promotion_require_both_embryos():
    with pytest.raises(RuntimeError, match="Kaggle"):
        m.require_kaggle_execution()
    summaries = {"image_only": {}, "kalman": {}}
    for embryo, delta in (("44b6", 0.01), ("6bba", -0.01)):
        summaries["image_only"][embryo] = {"combined_score": 0.9, "division_jaccard": 0.5}
        summaries["kalman"][embryo] = {"combined_score": 0.9 + delta, "division_jaccard": 0.5}
    result = m.promotion_decision(summaries, ["44b6", "6bba"])
    assert not result["conditions_met"]
    assert result["adoption_decision"] == "pending_user"


def test_official_summary_normalizes_source_keys_and_rejects_skips():
    class Metric:
        @staticmethod
        def summarise(rows):
            return {
                "score": 0.91,
                "adj_edge_jaccard": 0.9,
                "division_jaccard": 0.1,
                "n": len(rows),
                "n_adj": len(rows),
            }

    result = m.summarize([{}], Metric, 1)
    assert result["combined_score"] == 0.91
    assert result["adjusted_edge_jaccard"] == 0.9
    with pytest.raises(ValueError, match="skipped"):
        m.summarize([{}], Metric, 2)


def test_truth_resolver_uses_competition_train_not_prediction_geff(tmp_path):
    competition = "biohub-cell-tracking-during-development"
    sample = "44b6_one"
    expected = tmp_path / "competitions" / competition / "train" / f"{sample}.geff"
    expected.mkdir(parents=True)
    (tmp_path / "some-prediction-dataset" / f"{sample}.geff").mkdir(parents=True)
    assert m.resolve_truth_paths(tmp_path, [sample], competition) == {sample: expected}
    with pytest.raises(ValueError):
        m.resolve_truth_paths(tmp_path, [sample, "6bba_absent"], competition)


def test_public_archive_fallback_is_checksum_verified(tmp_path):
    import hashlib
    import zipfile

    payload = b"# official fixture\n"
    digest = hashlib.sha256(payload).hexdigest()
    with zipfile.ZipFile(tmp_path / "repo.zip", "w") as archive:
        archive.writestr("repo/scripts/evaluate.py", payload)
        archive.writestr("repo/src/module.py", "VALUE = 1\n")
    result = m.materialize_public_repository(tmp_path, tmp_path / "working", digest)
    assert result.read_bytes() == payload
    assert (result.parents[1] / "src/module.py").exists()
    with pytest.raises(FileNotFoundError):
        m.materialize_public_repository(tmp_path, tmp_path / "working", "wrong")


def test_fold_checkpoint_rejects_wrong_embryo_provenance(tmp_path):
    content = b"checkpoint fixture"
    checkpoint = tmp_path / "tracker.pth"
    checkpoint.write_bytes(content)
    row = {
        "fold": 0,
        "train_embryo": "44b6",
        "evaluation_embryo": "6bba",
        "file_sha256": m.file_sha(checkpoint),
        "path": "tracker.pth",
    }
    manifest = tmp_path / "model_manifest.json"
    manifest.write_text(json.dumps({"models": [row]}))
    cfg = {
        "data": {"model_manifest_sha256": m.file_sha(manifest)},
        "model": {
            "fold_checkpoints": {
                "0": {"relative_path": "tracker.pth", "sha256": m.file_sha(checkpoint)}
            }
        },
    }
    split = {"fold": 0, "train_embryo": "44b6", "evaluation_embryo": "6bba"}
    assert m.fold_tracker_checkpoint(tmp_path, split, cfg) == checkpoint
    with pytest.raises(ValueError, match="wrong fold"):
        m.fold_tracker_checkpoint(tmp_path, {**split, "train_embryo": "6bba"}, cfg)


def test_tracksdata_conversion_maps_noncontiguous_ids(monkeypatch):
    from types import SimpleNamespace

    class FakeGraph:
        def __init__(self):
            self.edges = []

        def add_node_attr_key(self, *args):
            pass

        def bulk_add_nodes(self, rows):
            self.nodes = rows
            return list(range(100, 100 + len(rows)))

        def bulk_add_edges(self, rows):
            self.edges.extend(rows)

    monkeypatch.setitem(sys.modules, "polars", SimpleNamespace(Float64=float))
    monkeypatch.setitem(
        sys.modules, "tracksdata", SimpleNamespace(graph=SimpleNamespace(InMemoryGraph=FakeGraph))
    )
    g = graph([1, 0], [1, 0], [(42, 9)], ids=[9, 42])
    converted, mapping = m.as_tracksdata(
        g, g.edges, np.ones(3), rounding=False, return_mapping=True
    )
    assert mapping == {9: 100, 42: 101}
    assert converted.edges == [{"source_id": 101, "target_id": 100}]


def test_expanded_candidates_match_archive_and_reject_corruption(tmp_path):
    import zipfile

    input_root = tmp_path / "input"
    input_root.mkdir()
    archive_path = input_root / "batch_000.zip"
    relative = "oracle_candidate_graphs/44b6_test.geff/nodes/zarr.json"
    with zipfile.ZipFile(archive_path, "w") as z:
        z.writestr(relative, b'{"same":true}')
    receipt = input_root / "batch_000.json"
    receipt.write_text("{}")
    digest = m.json_sha([["nodes/zarr.json", m.hashlib.sha256(b'{"same":true}').hexdigest()]])
    manifest = {
        "expected_sample_count": 1,
        "archives": [
            {
                "name": archive_path.name,
                "sha256": m.file_sha(archive_path),
                "bytes": archive_path.stat().st_size,
                "samples": ["44b6_test"],
                "receipt_name": receipt.name,
                "receipt_sha256": m.file_sha(receipt),
                "candidate_tree_shas": {"44b6_test": digest},
            }
        ],
    }
    result = m.materialize_candidates(input_root, tmp_path / "zip_output", manifest)
    assert m.candidate_tree_sha(result["44b6_test"]) == digest
    with zipfile.ZipFile(archive_path) as z:
        z.extractall(input_root / "batch_000")
    archive_path.unlink()
    result = m.materialize_candidates(input_root, tmp_path / "expanded_output", manifest)
    assert m.candidate_tree_sha(result["44b6_test"]) == digest
    (result["44b6_test"] / "nodes/zarr.json").write_text('{"same":false}')
    with pytest.raises(ValueError, match="content changed"):
        m.materialize_candidates(input_root, tmp_path / "corrupt_output", manifest)


def test_artifact_discovery_skips_competition_and_chunk_stores(tmp_path):
    for directory in [
        "datasets/owner/support/repo",
        "competitions/train",
        "legacy/train/sample.zarr",
        "data/sample.geff",
        "cache/window_cache",
    ]:
        root = tmp_path / directory
        root.mkdir(parents=True)
        (root / "evaluate.py").write_text("same_name")
    assert list(m.artifact_paths(tmp_path, "evaluate.py")) == [
        tmp_path / "datasets/owner/support/repo/evaluate.py"
    ]
