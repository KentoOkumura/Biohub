import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "assets" / "public_detector_selection.json"


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_reference_notebook_is_content_pinned_and_adopted() -> None:
    manifest = load_manifest()
    notebook = manifest["reference_notebook"]

    assert manifest["schema_version"] == 1
    assert manifest["selection_state"] == "adopted"
    assert manifest["adoption"] == {
        "decided_at": "2026-09-12",
        "user_message": "採用でいいです",
    }
    assert notebook["kernel_ref"] == "reyhanksatria/biohub-cell-tracking-0-946-lb"
    assert notebook["kernel_id_no"] == 133199516
    assert len(notebook["retrieved_notebook_sha256"]) == 64
    int(notebook["retrieved_notebook_sha256"], 16)
    assert "not independently retrieved" in notebook["score_evidence"]
    assert notebook["code_license"].startswith("not_declared")


def test_canonical_artifacts_have_fixed_versions_and_unique_shas() -> None:
    artifacts = load_manifest()["canonical_public_artifacts"]
    expected = {
        "pilkwang/biohub-tracking-support-pack-50ep-v1": 10,
        "pilkwang/biohub-temporal-unet3d-seed314159-v1": 2,
        "pilkwang/biohub-deepcenter-unet3d-center-prior-v1": 5,
    }

    assert len(artifacts) == 3
    assert {item["dataset_ref"]: item["current_dataset_version"] for item in artifacts} == expected
    assert {item["license"] for item in artifacts} == {"CC0-1.0"}
    checkpoint_shas = [item["checkpoint_sha256"] for item in artifacts]
    assert len(set(checkpoint_shas)) == 3
    for sha in checkpoint_shas:
        assert len(sha) == 64
        int(sha, 16)


def test_feature_and_tracker_contract_preserves_frozen_image_policy() -> None:
    manifest = load_manifest()
    temporal_models = manifest["canonical_public_artifacts"][:2]
    contract = manifest["pipeline_contract"]

    assert {item["window_size"] for item in temporal_models} == {2}
    assert {tuple(item["downsample_zyx"]) for item in temporal_models} == {(1, 4, 4)}
    assert {item["unet_output_channels"] for item in temporal_models} == {32}
    assert "32-channel" in contract["primary_candidate_feature"]
    assert "32-channel" in contract["secondary_candidate_feature"]
    recommendation = contract["downstream_training_recommendation"]
    assert "primary SimpleNodeTransformer" in recommendation
    assert "Freeze the primary TemporalUNet3D" in recommendation
    assert "complete secondary branch" in recommendation
    assert "DeepCenter" in recommendation
