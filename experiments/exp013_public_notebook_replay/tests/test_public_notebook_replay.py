import ast
import hashlib
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_NOTEBOOK = ROOT / "assets" / "reference_notebook" / "biohub-cell-tracking-0-947-lb.ipynb"
INFERENCE_SOURCE = ROOT / "exp013_public_notebook_replay_inference.py"
CONFIG_PATH = ROOT / "config.yaml"
EXPECTED_REFERENCE_SHA256 = "ae8e01a262211045161984e469e8be23e3386bab9140fe12df503dc6a1e010e6"

sys.path.insert(0, str(ROOT))
from compare_replays import compare_run_dirs, json_sha256  # noqa: E402


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def reference_code_source() -> str:
    notebook = json.loads(REFERENCE_NOTEBOOK.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(code_cells) == 1
    source = code_cells[0]["source"]
    return "".join(source) if isinstance(source, list) else source


def public_code_from_jupytext() -> str:
    source = INFERENCE_SOURCE.read_text(encoding="utf-8")
    code_marker = (
        '# %% _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5" '
        '_cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19"\n'
    )
    public_code = source.split(code_marker, maxsplit=1)[1]
    public_code = re.sub(
        r"(?ms)^# EXP013_REPLAY_ADDITION_START\n.*?^# EXP013_REPLAY_ADDITION_END\n?",
        "",
        public_code,
    )
    for added_import in (
        "import hashlib as _replay_hashlib\n",
        "import json as _replay_json\n",
        "import platform as _replay_platform\n",
        "import sys as _replay_sys\n",
        "from pathlib import Path as _ReplayPath\n",
    ):
        public_code = public_code.replace(added_import, "", 1)

    deepcenter_assignment = (
        "os.environ['BIOHUB_DEEPCENTER_CHECKPOINT'] = "
        "str(_REPLAY_DEEPCENTER_ROOT / 'weights/full_frame_center/best.pt')"
    )
    secondary_manifest_assignment = (
        "os.environ['BIOHUB_SECONDARY_ARTIFACT_MANIFEST'] = "
        "str(_REPLAY_SECONDARY_ROOT / 'ARTIFACT_MANIFEST.json')"
    )
    replacements = {
        "os.environ['BIOHUB_MODEL_ARTIFACTS'] = str(_REPLAY_PRIMARY_ROOT)": (
            "os.environ['BIOHUB_MODEL_ARTIFACTS'] = "
            "'/kaggle/input/datasets/reyhanksatria/biohub-tracking-support-pack'"
        ),
        "os.environ['BIOHUB_TARGET_ARTIFACT_SLUG'] = 'biohub-tracking-support-pack-50ep-v1'": (
            "os.environ['BIOHUB_TARGET_ARTIFACT_SLUG'] = 'biohub-tracking-support-pack'"
        ),
        deepcenter_assignment: (
            "os.environ['BIOHUB_DEEPCENTER_CHECKPOINT'] = "
            "'/kaggle/input/datasets/reyhanksatria/"
            "biohub-deepcenterunet3d-center-prior-v1/weights/full_frame_center/best.pt'"
        ),
        secondary_manifest_assignment: (
            "os.environ['BIOHUB_SECONDARY_ARTIFACT_MANIFEST'] = "
            "'/kaggle/input/datasets/reyhanksatria/"
            "biohub-temporalunet3d-seed-314159-v1/ARTIFACT_MANIFEST.json'"
        ),
    }
    for current, reference in replacements.items():
        assert current in public_code
        public_code = public_code.replace(current, reference)
    return public_code


def write_run(root: Path, *, candidate_sha: str = "candidate-a", runtime: float = 1.0) -> None:
    root.mkdir(parents=True)
    submission = b"id,dataset,row_type\n0,movie,node\n"
    (root / "submission.csv").write_bytes(submission)
    input_manifest = {
        "schema_version": 1,
        "wheel_manifest_sha256": "wheel-a",
        "test_movies": {"movie": {"file_count": 1}},
    }
    input_manifest_sha = json_sha256(input_manifest)
    (root / "replay_input_manifest.json").write_text(
        json.dumps(
            {**input_manifest, "input_manifest_sha256": input_manifest_sha},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    comparison_fields = [
        "input_manifest_sha256",
        "candidate_coordinate_content_sha256",
        "submission_sha256",
    ]
    receipt = {
        "comparison_fields": comparison_fields,
        "input_manifest_sha256": input_manifest_sha,
        "candidate_coordinate_content_sha256": candidate_sha,
        "submission_sha256": sha256_bytes(submission),
        "prediction_seconds_observed": runtime,
    }
    receipt["receipt_sha256"] = json_sha256(receipt)
    (root / "replay_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_reference_notebook_is_exactly_pinned() -> None:
    assert sha256_bytes(REFERENCE_NOTEBOOK.read_bytes()) == EXPECTED_REFERENCE_SHA256


def test_prediction_code_matches_reference_except_approved_mount_and_guards() -> None:
    reference_ast = ast.dump(ast.parse(reference_code_source()), include_attributes=False)
    implementation_ast = ast.dump(
        ast.parse(public_code_from_jupytext()),
        include_attributes=False,
    )
    assert implementation_ast == reference_ast


def test_notebook_has_required_sections_and_no_mirror_input_path() -> None:
    source = INFERENCE_SOURCE.read_text(encoding="utf-8")
    for section in (
        "## Contents",
        "## 1. Replay contract and reference source",
        "## 2. Public inference configuration",
        "## 3. Runtime, input, and dependency checks",
        "## 4. Source and checkpoint integrity checks",
        "## 5. Public inference and GPU sharding",
        "## 6. Graph repair and submission generation",
        "## 7. Replay metrics and generated artifacts",
    ):
        assert section in source
    assert "/kaggle/input/datasets/reyhanksatria" not in source
    assert "torch.use_deterministic_algorithms" not in source
    assert "CUBLAS_WORKSPACE_CONFIG" not in source
    assert "replay_input_manifest.json" in source
    assert "replay_receipt.json" in source


def test_config_pins_lineage_runtime_and_canonical_datasets() -> None:
    config = CONFIG_PATH.read_text(encoding="utf-8")
    assert "hypothesis_id: HYP-20260910-12" in config
    assert "backlog_candidate: public_notebook_replay" in config
    assert "machine_shape: NvidiaTeslaT4" in config
    assert "repeats: 2" in config
    for dataset in (
        "pilkwang/biohub-tracking-support-pack-50ep-v1",
        "pilkwang/biohub-temporal-unet3d-seed314159-v1",
        "pilkwang/biohub-deepcenter-unet3d-center-prior-v1",
    ):
        assert dataset in config


def test_compare_replays_ignores_runtime_but_requires_contract_fields(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    rerun = tmp_path / "rerun"
    write_run(reference, runtime=10.0)
    write_run(rerun, runtime=12.5)
    report = compare_run_dirs(reference, rerun)
    assert report["byte_identical_to_reference"] is True
    assert report["mismatches"] == []


def test_compare_replays_reports_candidate_difference(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    rerun = tmp_path / "rerun"
    write_run(reference)
    write_run(rerun, candidate_sha="candidate-b")
    report = compare_run_dirs(reference, rerun)
    assert report["byte_identical_to_reference"] is False
    assert report["mismatches"] == ["candidate_coordinate_content_sha256"]


def test_compare_replays_rejects_tampered_submission(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    rerun = tmp_path / "rerun"
    write_run(reference)
    write_run(rerun)
    (rerun / "submission.csv").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="submission_sha_mismatch"):
        compare_run_dirs(reference, rerun)
