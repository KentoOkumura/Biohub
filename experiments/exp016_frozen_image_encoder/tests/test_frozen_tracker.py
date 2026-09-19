from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
TRAIN_SOURCE = EXP / "exp016_frozen_image_encoder_train.py"
INFERENCE_SOURCE = EXP / "exp016_frozen_image_encoder_inference.py"
OFFICIAL_EVAL_SOURCE = EXP / "exp016_frozen_image_encoder_official_eval.py"
COLAB_BENCHMARK_SOURCE = EXP / "colab_benchmark_execute.py"
EXP015_INFERENCE_SOURCE = (
    EXP.parent / "exp015_oracle_stage_limits" / "exp015_oracle_stage_limits_inference.py"
)

sys.path.insert(0, str(EXP))
import colab_graph_replay as cgr  # noqa: E402
from build_colab_bundle import FILES as COLAB_BUNDLE_FILES  # noqa: E402
from frozen_tracker import (  # noqa: E402
    AnnotationGraph,
    FrameAnnotation,
    array_content_sha256,
    array_schema,
    build_embryo_splits,
    build_legacy_edge_target,
    build_window_example,
    cache_identity_record,
    extract_public_tracker_state,
    filter_nonempty_gt_window_paths,
    greedy_match_candidates,
    json_sha256,
    legacy_active_pair_mask,
    recompute_cache_identity_sha256,
)
from graph_inference import (  # noqa: E402
    EXPECTED_FOLD_MODELS,
    _coords_from_registry,
    _extract_tracker_state,
    _register_cached_frame,
    load_compact_graph_arrays,
    patch_exp015_source,
    select_cached_candidate_edges,
    sha256_file,
)
from official_evaluation import (  # noqa: E402
    force_public_evaluator_metadata_only,
    patch_exp015_source_for_recovered_ilp,
    patch_exp015_source_for_repaired_graph_evaluation,
)

COLAB_MODEL_PARAMS = cgr.MODEL_PARAMS
COLAB_REPLAY_CONFIG = cgr.REPLAY_CONFIG
colab_sample_names = cgr._sample_names
valid_colab_completed_batch = cgr._valid_completed_batch


def load_config() -> dict:
    return yaml.safe_load((EXP / "config.yaml").read_text(encoding="utf-8"))


def test_contract_freezes_image_side_and_trains_one_tracker_only() -> None:
    config = load_config()
    assert config["experiment"]["notebooks"] == ["train", "inference", "official_eval"]
    assert config["lineage"] == {
        "parent": "exp011_public_detector_selection",
        "hypothesis_id": "HYP-20260910-12",
        "backlog_candidate": "frozen_image_encoder",
        "diff_summary": config["lineage"]["diff_summary"],
    }
    model = config["model"]
    assert model["trainable_components"] == ["primary_SimpleNodeTransformer"]
    assert {
        "primary_TemporalUNet3D",
        "primary_detection_head",
        "secondary_TemporalUNet3D",
        "secondary_SimpleNodeTransformer",
        "detector_candidates",
        "decode_and_graph_repair",
    }.issubset(model["frozen_components"])
    assert model["training"]["active_variants"] == ["public_source_teacher"]
    assert model["training"]["epochs"] == 3
    assert model["training"]["runtime_gate_hours"] == 12.0
    assert model["control"]["retrain"] is False
    assert model["inference"]["implemented"] is True
    assert model["inference"]["fold_model_by_evaluation_embryo"] == {"6bba": 0, "44b6": 1}
    assert model["inference"]["active_variants"] == [
        "fixed_public_control",
        "fold_specific_retrained_primary_tracker",
    ]
    assert model["inference"]["mode"] == "fold_specific_primary_tracker_cached_graph_replay"
    assert model["inference"]["main_image_encoder_forward_count"] == 0
    assert model["inference"]["post_retention_candidates_reused_from_exp015"] is True
    assert model["inference"]["require_exact_public_control_candidate_graphs"] is True
    assert (
        model["inference"]["official_evaluator_preflight"]
        == "saved_public_control_graph_end_to_end"
    )
    assert model["inference"]["runtime_gate_scope"] == (
        "notebook_setup_cache_replay_and_graph_repair"
    )
    assert model["inference"]["replay"] == {
        "worker_count": 2,
        "expected_sample_count": 199,
        "expected_window_count": 19_701,
        "tracker_forward_count_per_window": 5,
        "expected_tracker_forward_count": 98_505,
        "downsample_zyx": [1, 4, 4],
        "edge_threshold": 0.48,
        "bidirectional_weight": 0.15,
        "secondary_edge_weight": 0.15,
        "secondary_link_mode": "low_margin_consensus",
        "secondary_low_margin_max": 0.35,
        "secondary_mix_temperature": 1.0,
        "use_ilp": True,
        "ilp_edge_weight": -1.0,
        "ilp_appearance_weight": 0.0,
        "ilp_disappearance_weight": 2.0,
        "ilp_division_weight": 1.2,
    }
    assert model["inference"]["retrain_models"] == 0
    assert model["inference"]["submission_created"] is False
    assert model["output"]["model_count"] == 2


def test_colab_route_is_one_gpu_resumable_cache_replay() -> None:
    colab = load_config()["runtime"]["colab"]
    assert colab == {
        "notebook": "exp016_frozen_image_encoder_colab_inference.ipynb",
        "stage": "cached_tracker_and_ilp_replay",
        "gpu_count": 1,
        "observed_gpu": "Tesla T4",
        "observed_python": "3.13.15",
        "input_modes": [
            "kaggle_api_token_colab_secret",
            "manual_google_drive",
        ],
        "run_mode": "benchmark_then_full",
        "benchmark_sample_count": 2,
        "full_sample_count": 199,
        "persistent_batch_size": 5,
        "resume": True,
        "runtime_gate_hours": 12.0,
        "main_image_encoder_forward_count": 0,
        "official_graph_evaluation_in_this_stage": False,
        "bundle_path": "artifacts/colab_bundle/exp016_colab_bundle.zip",
    }
    assert COLAB_MODEL_PARAMS == {
        "feature_dim": 64,
        "hidden_dim": 128,
        "n_heads": 4,
        "n_blocks": 4,
        "dropout": 0.3,
        "pair_chunk_size": 32,
    }
    canonical_replay = load_config()["model"]["inference"]["replay"]
    assert COLAB_REPLAY_CONFIG == {key: canonical_replay[key] for key in COLAB_REPLAY_CONFIG}


def test_kaggle_inference_remains_the_canonical_benchmark() -> None:
    runtime = load_config()["runtime"]
    assert runtime["benchmark"] == {
        "canonical_environment": "kaggle",
        "canonical_source": "exp016_frozen_image_encoder_inference.py",
        "canonical_notebook": "exp016_frozen_image_encoder_inference.ipynb",
        "canonical_stage": "cached_tracker_graph_repair_official_evaluation",
        "expected_sample_count": 199,
        "includes_graph_repair": True,
        "includes_official_graph_evaluation": True,
        "colab_role": "alternative_cached_tracker_and_ilp_runtime",
        "cross_environment_exact_fields": [
            "public_control_candidate_graph_content_sha256",
            "candidate_coordinate_content_sha256",
        ],
    }
    kaggle_inference = runtime["kaggle"]["inference"]
    assert kaggle_inference["role"] == "canonical_benchmark"
    assert kaggle_inference["source"] == "exp016_frozen_image_encoder_inference.py"
    assert kaggle_inference["notebook"] == "exp016_frozen_image_encoder_inference.ipynb"
    source = INFERENCE_SOURCE.read_text(encoding="utf-8")
    assert 'EXECUTION_ENVIRONMENT = "kaggle"' in source
    assert 'BENCHMARK_ROLE = "canonical"' in source
    assert '"execution_environment": EXECUTION_ENVIRONMENT' in source
    assert '"benchmark_role": BENCHMARK_ROLE' in source


def test_colab_notebook_uses_secrets_without_persisting_credentials() -> None:
    source_path = EXP / "exp016_frozen_image_encoder_colab_inference.py"
    notebook_path = EXP / "exp016_frozen_image_encoder_colab_inference.ipynb"
    source = source_path.read_text(encoding="utf-8")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    for marker in (
        'RUN_MODE = "benchmark_then_full"',
        'INPUT_MODE = "kaggle"',
        'userdata.get("KAGGLE_API_TOKEN")',
        '"--require-full"',
        '"--resume"',
        '"official_graph_evaluation_complete": False',
        "projected_full_seconds_with_25pct_reserve",
        "cache_replay_complete.json",
    ):
        assert marker in source
    assert "__file__" not in source
    assert "kaggle.json" not in source
    assert "access_token" not in source
    assert (
        'public_weight = primary_root / "weights/unet_transformer/split_0/edge_predictor_best.pth"'
    ) in source
    assert 'public_weight = repo_dir / "weights/' not in source
    assert all(not cell.get("outputs") for cell in notebook["cells"] if cell["cell_type"] == "code")


def test_colab_benchmark_helper_enforces_the_two_sample_gate() -> None:
    source = COLAB_BENCHMARK_SOURCE.read_text(encoding="utf-8")
    for marker in (
        'SAMPLES = ["44b6_0113de3b", "6bba_05b6850b"]',
        "RUNTIME_GATE_SECONDS = 12 * 60 * 60",
        "PROJECTION_RESERVE = 1.25",
        '"both_samples_exact"',
        '"fold_mapping_exact"',
        '"main_image_encoder_not_run"',
        '"tracker_forward_count_five"',
        '"within_12_hour_gate"',
    ):
        assert marker in source


def test_colab_bundle_contains_only_code_metadata_and_fold_trackers() -> None:
    destinations = sorted(COLAB_BUNDLE_FILES.values())
    assert destinations == [
        "code/colab_graph_replay.py",
        "code/frozen_tracker.py",
        "code/graph_inference.py",
        "code/window_cache.py",
        "metadata/config.yaml",
        "metadata/model_manifest.json",
        "models/fold_0/primary_tracker_best.pth",
        "models/fold_1/primary_tracker_best.pth",
    ]
    assert not any("kaggle.json" in path or "access_token" in path for path in destinations)


def test_colab_replay_resume_requires_receipt_and_batch_archive(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    for name in ("44b6_b", "6bba_a"):
        (cache_root / name).mkdir(parents=True)
    assert colab_sample_names(cache_root, [], None) == ["44b6_b", "6bba_a"]
    assert colab_sample_names(cache_root, ["6bba_a", "44b6_b"], 1) == ["44b6_b"]

    output_root = tmp_path / "output"
    sample = "44b6_b"
    receipt_path = output_root / "batch_receipts/batch_000.json"
    receipt_path.parent.mkdir(parents=True)
    archive_path = output_root / "batch_archives/batch_000.zip"
    archive_path.parent.mkdir(parents=True)
    archive_path.write_bytes(b"batch archive")
    receipt_path.write_text(
        json.dumps(
            {
                "samples": [{"sample": sample, "exact": True}],
                "public_control_candidate_graphs_exact": True,
                "archive_name": archive_path.name,
                "archive_sha256": sha256_file(archive_path),
            }
        ),
        encoding="utf-8",
    )
    assert valid_colab_completed_batch(receipt_path, [sample], output_root)
    archive_path.write_bytes(b"corrupted")
    assert not valid_colab_completed_batch(receipt_path, [sample], output_root)


def test_tracker_state_loader_accepts_training_checkpoint_wrapper() -> None:
    tracker_state = {
        "proj.weight": 1,
        "norm_in.weight": 2,
        "blocks.0.weight": 3,
        "norm_out.weight": 4,
        "pair_mlp.0.weight": 5,
    }
    checkpoint = {"fold": 1, "state_dict": tracker_state}
    assert _extract_tracker_state(checkpoint) == tracker_state


def test_public_teacher_and_loss_contract_are_exactly_pinned() -> None:
    model = load_config()["model"]
    assert model["teacher"] == {
        "matching": "greedy_distance_order_one_to_one",
        "max_matching_distance_um": 5.0,
        "target": "annotated_adjacent_frame_geff_edges",
        "unmatched_candidate_label": "unknown_but_legacy_mask_can_include_as_negative",
    }
    assert model["loss"] == {
        "edge_activation": "source_axis_softmax",
        "function": "focal_weighted_binary_cross_entropy",
        "focal_gamma": 2.0,
        "mask": "pair_touches_source_or_target_with_positive_annotated_edge",
        "division_weight": 1.0,
        "detection_loss": "disabled",
    }
    source = (EXP / "frozen_tracker.py").read_text(encoding="utf-8")
    assert "torch.softmax(logits, dim=0)" in source
    assert "active_rows.unsqueeze(1) | active_cols.unsqueeze(0)" in source
    assert "((1 - p_t) ** gamma) * bce" in source


def test_split_builder_produces_two_embryo_disjoint_directions() -> None:
    names_44b6 = [f"44b6_{index:08d}" for index in range(71)]
    names_6bba = [f"6bba_{index:08d}" for index in range(128)]
    names = names_44b6 + names_6bba
    records = build_embryo_splits(
        names,
        load_config()["validation"]["outer_folds"],
        split_seed=0,
    )
    assert len(records) == 2
    assert [
        (
            len(record["gradient_update"]),
            len(record["internal_validation"]),
            len(record["outer_evaluation"]),
        )
        for record in records
    ] == [(64, 7, 128), (116, 12, 71)]
    for record in records:
        training = set(record["gradient_update"]) | set(record["internal_validation"])
        evaluation = set(record["outer_evaluation"])
        assert not training & evaluation
        assert {name.split("_", 1)[0] for name in training} == {record["train_embryo"]}
        assert {name.split("_", 1)[0] for name in evaluation} == {record["evaluation_embryo"]}
    assert set(records[0]["outer_evaluation"]) | set(records[1]["outer_evaluation"]) == set(names)


def test_greedy_matching_uses_distance_order_and_one_to_one_gt_claims() -> None:
    candidates = np.asarray([[0.2, 0.0, 0.0], [0.1, 0.0, 0.0], [9.0, 0.0, 0.0]])
    gt_ids = np.asarray([101, 102])
    gt_coords = np.asarray([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    matches, distances = greedy_match_candidates(candidates, gt_ids, gt_coords, 5.0)
    assert matches.tolist() == [-1, 101, 102]
    assert np.isnan(distances[0])
    assert np.allclose(distances[1:], [0.1, 1.0])


def test_division_target_and_legacy_or_mask_include_unknown_endpoint_pairs() -> None:
    source_matches = np.asarray([1, -1])
    target_matches = np.asarray([2, 3, -1])
    target = build_legacy_edge_target(
        source_matches,
        target_matches,
        frozenset({(1, 2), (1, 3)}),
    )
    assert target.tolist() == [[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
    mask = legacy_active_pair_mask(target)
    assert mask.tolist() == [[True, True, True], [True, True, False]]
    unknown = (source_matches < 0)[:, None] | (target_matches < 0)[None, :]
    assert int((mask & unknown).sum()) == 3


def test_window_example_uses_physical_coordinates_only_for_teacher(tmp_path: Path) -> None:
    sample = "44b6_example"
    path = tmp_path / sample / "000000_000001.npz"
    path.parent.mkdir()
    arrays = {
        "coords_src_grid": np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32),
        "coords_tgt_grid": np.asarray([[2.0, 3.0, 4.0]], dtype=np.float32),
        "coords_src_physical": np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        "coords_tgt_physical": np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32),
        "position_features_src": np.ones((1, 32), dtype=np.float32),
        "position_features_tgt": np.full((1, 32), 2.0, dtype=np.float32),
        "candidate_mask_src": np.asarray([True]),
        "candidate_mask_tgt": np.asarray([True]),
        "primary_features_src": np.full((1, 32), 3.0, dtype=np.float32),
        "primary_features_tgt": np.full((1, 32), 4.0, dtype=np.float32),
        "secondary_features_src": np.full((1, 32), 5.0, dtype=np.float32),
        "secondary_features_tgt": np.full((1, 32), 6.0, dtype=np.float32),
    }
    metadata = {
        "schema_version": 1,
        "experiment": "exp015_oracle_stage_limits",
        "dataset": sample,
        "window_frames": [0, 1],
        "primary_checkpoint_sha256": "checkpoint",
        "array_schema": array_schema(arrays),
        "array_content_sha256": array_content_sha256(arrays),
    }
    np.savez(
        path,
        **arrays,
        __metadata_json__=np.frombuffer(
            json.dumps(metadata, separators=(",", ":"), sort_keys=True).encode(),
            dtype=np.uint8,
        ),
    )
    annotation = AnnotationGraph(
        frames={
            0: FrameAnnotation(np.asarray([10]), np.asarray([[0.0, 0.0, 0.0]])),
            1: FrameAnnotation(np.asarray([11]), np.asarray([[0.0, 1.0, 0.0]])),
        },
        edges=frozenset({(10, 11)}),
        content_sha256="annotation",
    )
    example = build_window_example(
        path,
        annotation,
        feature_channels=32,
        expected_primary_checkpoint_sha256="checkpoint",
        max_matching_distance_um=5.0,
        downsample_zyx=(1.0, 4.0, 4.0),
        verify_content=True,
    )
    assert example["target"].tolist() == [[1.0]]
    assert example["features_src"].shape == (1, 64)
    assert example["features_src"][0, :32].tolist() == [3.0] * 32
    assert example["features_src"][0, 32:].tolist() == [1.0] * 32
    assert example["coords_src"].tolist() == [[1.0, 8.0, 12.0]]
    identity_record = cache_identity_record(path)
    assert identity_record["feature_values"] == 128
    assert recompute_cache_identity_sha256([path]) == json_sha256([identity_record])


def test_empty_gt_frames_are_filtered_like_the_public_trainer() -> None:
    sample = "6bba_example"
    paths = [
        Path("/cache") / sample / "000038_000039.npz",
        Path("/cache") / sample / "000039_000040.npz",
        Path("/cache") / sample / "000040_000041.npz",
    ]
    annotations = {
        sample: AnnotationGraph(
            frames={
                38: FrameAnnotation(np.asarray([1]), np.zeros((1, 3))),
                39: FrameAnnotation(np.asarray([2]), np.zeros((1, 3))),
                41: FrameAnnotation(np.asarray([3]), np.zeros((1, 3))),
            },
            edges=frozenset({(1, 2)}),
            content_sha256="annotation",
        )
    }
    eligible, audit = filter_nonempty_gt_window_paths(paths, annotations)
    assert eligible == [paths[0]]
    assert audit == {
        "policy": "skip_window_if_any_frame_has_zero_gt_nodes",
        "public_source_function": "get_window_data",
        "input_window_count": 3,
        "eligible_window_count": 1,
        "skipped_window_count": 2,
        "skipped_by_sample": {sample: 2},
        "skipped_windows": [
            {
                "sample": sample,
                "window_frames": [39, 40],
                "empty_gt_frames": [40],
            },
            {
                "sample": sample,
                "window_frames": [40, 41],
                "empty_gt_frames": [40],
            },
        ],
    }


def test_public_tracker_state_extraction_accepts_full_and_tracker_only_states() -> None:
    full = {"unet.layer": 1, "transformer.proj.weight": 2, "transformer.norm_in.weight": 3}
    assert extract_public_tracker_state(full) == {"proj.weight": 2, "norm_in.weight": 3}
    tracker_only = {
        "proj.weight": 1,
        "norm_in.weight": 2,
        "blocks.0.weight": 3,
        "norm_out.weight": 4,
        "pair_mlp.0.weight": 5,
    }
    assert extract_public_tracker_state(tracker_only) == tracker_only


def test_train_source_has_full_training_and_no_submission_path() -> None:
    source = TRAIN_SOURCE.read_text(encoding="utf-8")
    for marker in (
        "validate_cache_summary",
        "build_embryo_splits",
        "benchmark_summary.json",
        "runtime_gate_seconds",
        "new_optimizer",
        "optimizer parameters differ from primary tracker parameters",
        "baseline_outer = evaluate_tracker",
        'for epoch in range(int(train_cfg["epochs"]))',
        "primary_tracker_best.pth",
        "teacher_audit.json",
        "model_manifest",
        '"official_metric_computed": False',
        '"submission_created": False',
    ):
        assert marker in source
    assert "kaggle competitions submit" not in source
    assert "TemporalUNet3D(" not in source
    assert "detection_loss" not in source


def test_train_notebook_is_synced_and_backlog_contract_has_migrated() -> None:
    notebook = json.loads((EXP / "exp016_frozen_image_encoder_train.ipynb").read_text())
    notebook_source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    assert "benchmark_summary.json" in notebook_source
    assert "primary_tracker_best.pth" in notebook_source
    direction = (ROOT / "backlog" / "KAGGLE_DIRECTION.md").read_text(encoding="utf-8")
    assert "`exp016_frozen_image_encoder`" in direction
    assert "[frozen_image_encoder](frozen_image_encoder.md)" not in direction
    assert not (ROOT / "backlog" / "frozen_image_encoder.md").exists()


def test_inference_patches_checkpoint_injection_and_cache_graph_replay() -> None:
    config = load_config()
    inference_cfg = config["model"]["inference"]
    assert sha256_file(EXP015_INFERENCE_SOURCE) == inference_cfg["base_pipeline_source"]["sha256"]
    dependency_files = config["runtime"]["kaggle"]["inference"]["bootstrap_dependency_files"]
    assert {item["destination"] for item in dependency_files} == {
        "exp015_inference_base.py",
        "window_cache.py",
    }
    patched = patch_exp015_source(EXP015_INFERENCE_SOURCE.read_text(encoding="utf-8"))
    assert patched.count("EXP016_FOLD_TRACKER_INJECTION_START") == 1
    assert patched.count("EXP016_CACHE_REPLAY_START") == 1
    custom_prediction = patched.split("EXP016_CACHE_REPLAY_START", 1)[1].split(
        "EXP016_CACHE_REPLAY_END", 1
    )[0]
    assert "run_cached_graph_replay" in custom_prediction
    assert "main_image_encoder_forward_count': 0" in custom_prediction
    assert "predict_unet_transformer.py" not in custom_prediction
    assert "subprocess.Popen" not in custom_prediction
    assert "No frame-retention diagnostics were produced" not in patched
    assert patched.count("EXP013_REPLAY_ADDITION_START") == 2
    assert "def _replay_read_json" not in patched
    assert "EXP016 cache replay graph repair complete" in patched
    assert "time.monotonic() - started >= _exp016_gate_seconds" in patched
    helper_source = (EXP / "graph_inference.py").read_text(encoding="utf-8")
    assert 'str(repo_dir / "scripts")' in helper_source
    assert 'str(repo_dir / "src")' in helper_source
    compile(patched, "patched_exp015.py", "exec")


def test_official_evaluation_uses_recovered_ilp_without_prediction() -> None:
    config = load_config()
    runtime = config["runtime"]["kaggle"]["official_eval"]
    assert runtime["enable_gpu"] is False
    assert runtime["enable_internet"] is False
    assert runtime["machine_shape"] is None
    assert runtime["kernel_sources"] == ["kentookumura/exp015-oracle-stage-limits-inference"]
    assert "kentookumura/exp016-repaired-graphs-v5" in runtime["dataset_sources"]

    patched = patch_exp015_source_for_recovered_ilp(
        EXP015_INFERENCE_SOURCE.read_text(encoding="utf-8")
    )
    assert "EXP016_RECOVERED_ILP_INPUT_START" in patched
    assert "recovered_ilp_input_receipt.json" in patched
    assert "public_control_candidate_graphs_exact': True" in patched
    assert "CUDA GPU is required for this notebook" not in patched
    assert "DeepCenter uses CPU" in patched
    assert "kaggle_auto_expanded_archive" in patched
    assert "checksum_pinned_raw_fallback" in patched
    assert "_exp016_persistent_field" in patched
    assert "shutil.copytree(_exp016_graph, _exp016_destination)" in patched
    recovered_stage = patched.split("EXP016_RECOVERED_ILP_INPUT_START", 1)[1].split(
        "EXP016_RECOVERED_ILP_INPUT_END", 1
    )[0]
    assert "predict_unet_transformer.py" not in recovered_stage
    assert "subprocess.Popen" not in recovered_stage
    assert "EXP016 recovered-ILP graph repair complete" in patched
    assert "time.monotonic() - started >= OFFICIAL_EVAL_RUNTIME_GATE_SECONDS" in patched
    compile(patched, "patched_exp015_official_eval.py", "exec")

    source = OFFICIAL_EVAL_SOURCE.read_text(encoding="utf-8")
    for marker in (
        "patch_exp015_source_for_repaired_graph_evaluation",
        "RECOVERED_REPAIRED_MANIFEST_FILE_SHA256",
        "graph_repair_complete",
        "official_metric_computed",
        "force_public_evaluator_metadata_only",
        "public_evaluator.evaluate_run",
        "delta_retrained_minus_control",
        '"submission_created": False',
    ):
        assert marker in source
    assert "submission.csv" not in source
    assert "kaggle competitions submit" not in source


def test_repaired_graph_evaluation_setup_skips_prediction_and_repair() -> None:
    patched = patch_exp015_source_for_repaired_graph_evaluation(
        EXP015_INFERENCE_SOURCE.read_text(encoding="utf-8")
    )
    assert "Skipping prediction: using checksum-verified repaired compact graphs" in patched
    assert "Skipping graph repair: using checksum-verified Kaggle v5 outputs" in patched
    assert "subprocess.run(predict_cmd" not in patched
    assert "DEEPCENTER_VETO_DETECTOR = load_deepcenter_veto_detector()" not in patched
    assert "CUDA GPU is required for this notebook" not in patched
    compile(patched, "repaired_graph_evaluation_setup.py", "exec")


def test_public_evaluator_metadata_only_loader_disables_image_loading() -> None:
    calls: list[dict[str, object]] = []

    def open_dataset(*args: object, **kwargs: object) -> SimpleNamespace:
        calls.append(dict(kwargs))
        return SimpleNamespace(image=None, scale=(1.0, 1.0, 1.0), tracks=object())

    evaluator = SimpleNamespace(open_dataset=open_dataset)
    force_public_evaluator_metadata_only(evaluator)
    dataset = evaluator.open_dataset(Path("sample"), require_tracks=True)
    assert dataset.image is None
    assert calls == [{"require_tracks": True, "load_image": False}]


def test_cached_candidate_registry_preserves_global_ids_and_downsample() -> None:
    registry: dict[int, tuple[int, int, int, int]] = {}
    _register_cached_frame(
        registry,
        frame=0,
        candidate_ids=np.asarray([0, 1], dtype=np.int64),
        coords_grid=np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        downsample_zyx=(1, 4, 4),
    )
    _register_cached_frame(
        registry,
        frame=1,
        candidate_ids=np.asarray([2], dtype=np.int64),
        coords_grid=np.asarray([[7, 8, 9]], dtype=np.float32),
        downsample_zyx=(1, 4, 4),
    )
    assert _coords_from_registry(registry).tolist() == [
        [0, 1, 8, 12],
        [0, 4, 20, 24],
        [1, 7, 32, 36],
    ]


def test_cached_edge_selection_uses_source_softmax_and_global_candidate_ids(monkeypatch) -> None:
    class FakeTensor:
        def __init__(self, value: np.ndarray) -> None:
            self.value = value

        def detach(self) -> FakeTensor:
            return self

        def cpu(self) -> FakeTensor:
            return self

        def numpy(self) -> np.ndarray:
            return self.value

    def softmax(value: np.ndarray, *, dim: int) -> FakeTensor:
        shifted = value - value.max(axis=dim, keepdims=True)
        numerator = np.exp(shifted)
        return FakeTensor(numerator / numerator.sum(axis=dim, keepdims=True))

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(softmax=softmax))

    logits = np.asarray([[[2.0, 0.0], [0.0, 2.0]]], dtype=np.float32)
    arrays = {
        "candidate_ids_src": np.asarray([10, 11], dtype=np.int64),
        "candidate_ids_tgt": np.asarray([20, 21], dtype=np.int64),
        "coords_src_grid": np.asarray([[0, 0, 0], [0, 3, 4]], dtype=np.float32),
        "coords_tgt_grid": np.asarray([[0, 0, 1], [0, 6, 8]], dtype=np.float32),
    }
    edges = select_cached_candidate_edges(logits, arrays, threshold=0.8)
    assert [(source, target) for source, target, _, _ in edges] == [(11, 21), (10, 20)]
    assert all(probability > 0.8 for _, _, probability, _ in edges)
    assert sorted(distance for _, _, _, distance in edges) == [1.0, 5.0]


def test_fold_checkpoints_are_pinned_to_the_held_out_embryo_mapping() -> None:
    assert EXPECTED_FOLD_MODELS == {
        0: {
            "evaluation_embryo": "6bba",
            "file_sha256": "c775ddabc2df623b08c39e548bebc887a73e790cc89bd9a9e1cb2c3506eff9d2",
            "canonical_state_sha256": (
                "b080c8dc374454f0f97403ec1e27388ef9a5a76c4c658b0ccaeb247ab136ab6c"
            ),
        },
        1: {
            "evaluation_embryo": "44b6",
            "file_sha256": "ef8b60dc1a34e336942f9abaefdd47606a457e0f569363434619992980d0c5e9",
            "canonical_state_sha256": (
                "41b03b82d30e9b22d73df6924a70d56ca2b292bb3c41b2678acb5d33990fb3ea"
            ),
        },
    }
    helper_source = (EXP / "graph_inference.py").read_text(encoding="utf-8")
    assert "destination = output_manifest_path.parent / relative" in helper_source
    assert 'Path("hybrid_primary_models")' in helper_source


def test_compact_final_graph_conversion_rounds_and_reindexes(tmp_path: Path) -> None:
    path = tmp_path / "sample.npz"
    np.savez_compressed(
        path,
        node_ids=np.asarray([4, 9, 12], dtype=np.int64),
        node_tzyx=np.asarray(
            [[0, -0.4, 1.6, 2.4], [1, 2.5, 3.49, 4.51], [2, 5, 6, 7]],
            dtype=np.float64,
        ),
        edges=np.asarray([[4, 9], [9, 12]], dtype=np.int64),
    )
    nodes, edges = load_compact_graph_arrays(path)
    assert nodes == [
        {"t": 0, "z": 0, "y": 2, "x": 2},
        {"t": 1, "z": 2, "y": 3, "x": 5},
        {"t": 2, "z": 5, "y": 6, "x": 7},
    ]
    assert edges == [(0, 1), (1, 2)]


def test_inference_source_runs_official_metric_without_submission() -> None:
    evaluator_cfg = load_config()["model"]["inference"]["public_evaluator_source"]
    assert evaluator_cfg == {
        "dataset_ref": "pilkwang/biohub-tracking-support-pack-50ep-v1",
        "path": "repo/scripts/evaluate.py",
        "sha256": "614813cc51c3581c6ccda4bb20725a19da8ecac4a27620654bfca58319cffa3c",
        "entrypoint": "evaluate_run",
        "metrics_module": "biohub_tracking.metrics",
    }
    source = INFERENCE_SOURCE.read_text(encoding="utf-8")
    for marker in (
        "resolve_verified_exp015_output",
        "run_cached_graph_replay",
        "cache_replay_contract_changed",
        "main_image_encoder_forward_count",
        "public_control_candidate_graphs_exact",
        "PUBLIC_EVALUATOR_SHA256",
        "public_evaluator.evaluate_run",
        "from biohub_tracking.metrics import summarise",
        "summarise",
        "delta_retrained_minus_control",
        '"submission_created": False',
        "official_graph_evaluation.json",
    ):
        assert marker in source
    assert "submission.csv" not in source
    assert "kaggle competitions submit" not in source
    assert evaluator_cfg["sha256"] in source
    assert "evaluate_pairs" not in source
    assert "tracking_cellmot.metrics" not in source
    assert source.index("PUBLIC_EVALUATOR_SHA256") < source.index(
        "compile(cache_replay_marker + replay_source"
    )
    assert source.index("preflight_rows = public_evaluator.evaluate_run") < source.index(
        "compile(cache_replay_marker + replay_source"
    )
