from __future__ import annotations

from typing import Any

_RECOVERED_ILP_STAGE = r"""# EXP016_RECOVERED_ILP_INPUT_START
start_time = time.time()
_exp016_recovered_summary_path = RECOVERED_ILP_ROOT / 'colab_cache_replay_summary.json'
if not _exp016_recovered_summary_path.is_file():
    raise FileNotFoundError(_exp016_recovered_summary_path)
_exp016_recovered_summary = json.loads(
    _exp016_recovered_summary_path.read_text(encoding='utf-8')
)
_exp016_required_summary = {
    'authoritative_full_replay': True,
    'sample_count': 199,
    'window_count': 19701,
    'persistent_batch_archive_count': 40,
    'main_image_encoder_forward_count': 0,
    'public_control_candidate_graphs_exact': True,
}
_exp016_observed_summary = {
    key: _exp016_recovered_summary.get(key) for key in _exp016_required_summary
}
if _exp016_observed_summary != _exp016_required_summary:
    raise RuntimeError({
        'recovered_ilp_summary_changed': {
            'expected': _exp016_required_summary,
            'actual': _exp016_observed_summary,
        }
    })
_exp016_summary_samples = [
    str(row['sample']) for row in _exp016_recovered_summary.get('samples', [])
]
if (
    len(_exp016_summary_samples) != 199
    or len(set(_exp016_summary_samples)) != 199
    or sorted(_exp016_summary_samples) != sorted(test_stems)
    or not all(bool(row.get('exact')) for row in _exp016_recovered_summary['samples'])
):
    raise RuntimeError('recovered ILP summary sample contract changed')

_exp016_archives = sorted(RECOVERED_ILP_ROOT.glob('batch_*.zip'))
_exp016_batch_dirs = sorted(
    path for path in RECOVERED_ILP_ROOT.glob('batch_*') if path.is_dir()
)
_exp016_fallback_archives = sorted(RECOVERED_ILP_ROOT.glob('batch_*_archive.bin'))
_exp016_receipts = sorted(RECOVERED_ILP_ROOT.glob('batch_*.json'))
if len(_exp016_receipts) != 40:
    raise RuntimeError({
        'expected_batch_count': 40,
        'archives': len(_exp016_archives),
        'expanded_batch_dirs': len(_exp016_batch_dirs),
        'fallback_archives': len(_exp016_fallback_archives),
        'receipts': len(_exp016_receipts),
    })
_exp016_batches = []
for _exp016_receipt_path in _exp016_receipts:
    _exp016_batch_stem = _exp016_receipt_path.stem
    _exp016_archive_path = RECOVERED_ILP_ROOT / (_exp016_batch_stem + '.zip')
    _exp016_expanded_path = RECOVERED_ILP_ROOT / _exp016_batch_stem
    _exp016_fallback_path = RECOVERED_ILP_ROOT / (_exp016_batch_stem + '_archive.bin')
    if _exp016_archive_path.is_file():
        _exp016_batches.append(('source_zip_archive', _exp016_archive_path, _exp016_receipt_path))
    elif _exp016_expanded_path.is_dir():
        _exp016_batches.append((
            'kaggle_auto_expanded_archive',
            _exp016_expanded_path,
            _exp016_receipt_path,
        ))
    elif _exp016_fallback_path.is_file():
        _exp016_batches.append((
            'checksum_pinned_raw_fallback',
            _exp016_fallback_path,
            _exp016_receipt_path,
        ))
    else:
        raise RuntimeError({'missing_recovered_batch': _exp016_batch_stem})
_exp016_input_layout = sorted({mode for mode, _, _ in _exp016_batches})

_exp016_extract_root = WORKING_DIR / 'recovered_ilp_graphs'
if _exp016_extract_root.exists():
    shutil.rmtree(_exp016_extract_root)
_exp016_extract_root.mkdir(parents=True)
_exp016_archive_records = []
_exp016_receipt_samples = []
for _exp016_mode, _exp016_batch, _exp016_receipt_path in _exp016_batches:
    _exp016_batch_stem = _exp016_receipt_path.stem
    _exp016_receipt = json.loads(_exp016_receipt_path.read_text(encoding='utf-8'))
    _exp016_receipt_sha = str(_exp016_receipt.get('receipt_sha256', ''))
    _exp016_receipt_payload = dict(_exp016_receipt)
    for _exp016_persistent_field in (
        'receipt_sha256',
        'archive_name',
        'archive_bytes',
        'archive_sha256',
    ):
        _exp016_receipt_payload.pop(_exp016_persistent_field, None)
    if _replay_json_sha256(_exp016_receipt_payload) != _exp016_receipt_sha:
        raise RuntimeError({'invalid_recovered_receipt_sha': _exp016_receipt_path.name})
    _exp016_batch_sample_rows = list(_exp016_receipt.get('samples', []))
    _exp016_batch_sample_names = sorted(
        str(row['sample']) for row in _exp016_batch_sample_rows
    )
    if (
        len(_exp016_batch_sample_names) != int(_exp016_receipt.get('sample_count', -1))
        or len(set(_exp016_batch_sample_names)) != len(_exp016_batch_sample_names)
        or not all(bool(row.get('exact')) for row in _exp016_batch_sample_rows)
    ):
        raise RuntimeError({'invalid_recovered_receipt_samples': _exp016_receipt_path.name})
    _exp016_receipt_samples.extend(_exp016_batch_sample_names)
    _exp016_expected_archive_name = _exp016_batch_stem + '.zip'
    if (
        _exp016_receipt.get('archive_name') != _exp016_expected_archive_name
        or not str(_exp016_receipt.get('archive_sha256', ''))
        or int(_exp016_receipt.get('archive_bytes', -1)) <= 0
    ):
        raise RuntimeError({'invalid_recovered_archive_receipt': _exp016_batch_stem})
    if _exp016_mode != 'kaggle_auto_expanded_archive':
        _exp016_archive_sha = _replay_sha256_file(_exp016_batch)
        if (
            _exp016_receipt.get('archive_sha256') != _exp016_archive_sha
            or int(_exp016_receipt['archive_bytes']) != _exp016_batch.stat().st_size
        ):
            raise RuntimeError({'invalid_recovered_archive_bytes': _exp016_batch.name})
        with zipfile.ZipFile(_exp016_batch) as _exp016_zip:
            _exp016_bad_member = _exp016_zip.testzip()
            if _exp016_bad_member is not None:
                raise RuntimeError({
                    'corrupt_recovered_ilp_archive': _exp016_batch.name,
                    'member': _exp016_bad_member,
                })
            for _exp016_info in _exp016_zip.infolist():
                _exp016_relative = Path(_exp016_info.filename)
                if _exp016_relative.is_absolute() or '..' in _exp016_relative.parts:
                    raise RuntimeError({'unsafe_recovered_ilp_member': _exp016_info.filename})
                if (
                    not _exp016_relative.parts
                    or _exp016_relative.parts[0] != 'ilp_graphs'
                    or _exp016_info.is_dir()
                ):
                    continue
                _exp016_destination = _exp016_extract_root.joinpath(
                    *_exp016_relative.parts[1:]
                )
                _exp016_destination.parent.mkdir(parents=True, exist_ok=True)
                if _exp016_destination.exists():
                    raise RuntimeError({
                        'duplicate_recovered_ilp_member': str(_exp016_relative)
                    })
                with (
                    _exp016_zip.open(_exp016_info) as _exp016_source,
                    _exp016_destination.open('wb') as _exp016_target,
                ):
                    shutil.copyfileobj(_exp016_source, _exp016_target)
    else:
        _exp016_graph_root = _exp016_batch / 'ilp_graphs'
        _exp016_expanded_graphs = sorted(_exp016_graph_root.glob('*.geff'))
        if [path.stem for path in _exp016_expanded_graphs] != _exp016_batch_sample_names:
            raise RuntimeError({'expanded_recovered_batch_changed': _exp016_batch.name})
        for _exp016_graph in _exp016_expanded_graphs:
            _exp016_destination = _exp016_extract_root / _exp016_graph.name
            if _exp016_destination.exists():
                raise RuntimeError({'duplicate_recovered_ilp_graph': _exp016_graph.name})
            shutil.copytree(_exp016_graph, _exp016_destination)
    _exp016_archive_records.append([
        str(_exp016_receipt['archive_name']),
        str(_exp016_receipt['archive_sha256']),
        int(_exp016_receipt['archive_bytes']),
    ])

if sorted(_exp016_receipt_samples) != sorted(test_stems):
    raise RuntimeError('recovered ILP receipt sample coverage changed')

_exp016_graphs = sorted(_exp016_extract_root.glob('*.geff'))
_exp016_graph_names = [path.stem for path in _exp016_graphs]
if (
    len(_exp016_graph_names) != 199
    or len(set(_exp016_graph_names)) != 199
    or _exp016_graph_names != sorted(test_stems)
):
    raise RuntimeError({
        'recovered_ilp_graph_count': len(_exp016_graph_names),
        'names_match': _exp016_graph_names == sorted(test_stems),
    })

_exp016_prediction_root = REPO_DIR / 'predictions'
if _exp016_prediction_root.exists():
    shutil.rmtree(_exp016_prediction_root)
for _exp016_graph in _exp016_graphs:
    _exp016_destination = (
        _exp016_prediction_root
        / _exp016_graph.stem
        / METHOD
        / 'split_0'
        / _exp016_graph.name
    )
    _exp016_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(_exp016_graph), str(_exp016_destination))
shutil.rmtree(_exp016_extract_root)

predict_seconds = time.time() - start_time
_exp016_recovered_input_receipt = {
    'stage': 'recovered_colab_ilp_graph_input',
    'sample_count': 199,
    'archive_count': 40,
    'input_layout': _exp016_input_layout,
    'window_count': 19701,
    'main_image_encoder_forward_count': 0,
    'public_control_candidate_graphs_exact': True,
    'source_full_summary_sha256': _replay_sha256_file(_exp016_recovered_summary_path),
    'archive_manifest_sha256': _replay_json_sha256(_exp016_archive_records),
    'elapsed_seconds': predict_seconds,
}
_exp016_recovered_input_receipt['receipt_sha256'] = _replay_json_sha256(
    _exp016_recovered_input_receipt
)
(WORKING_DIR / 'recovered_ilp_input_receipt.json').write_text(
    json.dumps(_exp016_recovered_input_receipt, indent=2, sort_keys=True) + '\n',
    encoding='utf-8',
)
print(f'Validated and materialized {len(_exp016_graph_names)} recovered ILP graphs.')
# EXP016_RECOVERED_ILP_INPUT_END"""


def patch_exp015_source_for_recovered_ilp(source: str) -> str:
    input_manifest_marker = (
        "_REPLAY_INPUT_MANIFEST = _replay_write_input_manifest(TEST_DIR, WORKING_DIR)"
    )
    if source.count(input_manifest_marker) != 1:
        raise RuntimeError("exp015 input-manifest marker changed")
    source = source.replace(
        input_manifest_marker,
        "_REPLAY_INPUT_MANIFEST = {'stage': 'recovered_ilp_graph_postprocess'}",
        1,
    )

    cuda_guard = (
        "# Require a CUDA device before launching GPU inference\n"
        "if not _torch.cuda.is_available():\n"
        "    raise RuntimeError('CUDA GPU is required for this notebook. "
        "Enable a Kaggle GPU accelerator and commit again.')\n\n"
        "print('CUDA device:', _torch.cuda.get_device_name(0))"
    )
    cpu_postprocess_guard = """# Recovered-ILP postprocessing does not launch tracker inference.
print('CUDA device: not required for recovered-ILP graph repair; DeepCenter uses CPU')"""
    if source.count(cuda_guard) != 1:
        raise RuntimeError("exp015 CUDA preflight marker changed")
    source = source.replace(cuda_guard, cpu_postprocess_guard, 1)

    prediction_start_marker = (
        "start_time = time.time()\navailable_gpu_count = _torch.cuda.device_count()"
    )
    prediction_end_marker = "print(f'Prediction completed in {predict_seconds / 60:.2f} minutes')"
    if source.count(prediction_start_marker) != 1 or source.count(prediction_end_marker) != 1:
        raise RuntimeError("exp015 prediction execution markers changed")
    prediction_start = source.index(prediction_start_marker)
    prediction_end = source.index(prediction_end_marker, prediction_start) + len(
        prediction_end_marker
    )
    source = source[:prediction_start] + _RECOVERED_ILP_STAGE + source[prediction_end:]

    repair_loop_marker = "    for geff_path in geffs:\n        dataset = geff_path.stem"
    repair_loop_replacement = """    for geff_path in geffs:
        if time.monotonic() - started >= OFFICIAL_EVAL_RUNTIME_GATE_SECONDS:
            raise TimeoutError('exp016 recovered-ILP graph repair exceeded the runtime gate')
        dataset = geff_path.stem"""
    if source.count(repair_loop_marker) != 1:
        raise RuntimeError("exp015 graph repair loop marker changed")
    source = source.replace(repair_loop_marker, repair_loop_replacement, 1)

    repair_end_marker = "display(pd.read_csv(FINAL_GRAPH_ROWS_PATH, nrows = 8))"
    if source.count(repair_end_marker) != 1:
        raise RuntimeError("exp015 final graph repair marker changed")
    repair_end = source.index(repair_end_marker) + len(repair_end_marker)
    source = source[:repair_end] + "\nprint('EXP016 recovered-ILP graph repair complete')\n"
    source = source.replace(
        "print('Submission path:', FINAL_GRAPH_ROWS_PATH)",
        "print('Temporary row-form graph path:', FINAL_GRAPH_ROWS_PATH)",
    )
    if "EXP016_RECOVERED_ILP_INPUT_START" not in source:
        raise RuntimeError("exp016 recovered-ILP input patch did not persist")
    compile(source, "exp015_recovered_ilp_postprocess.py", "exec")
    return source


def patch_exp015_source_for_repaired_graph_evaluation(source: str) -> str:
    """Keep the pinned environment setup but reuse completed graph-repair output."""
    input_manifest_marker = (
        "_REPLAY_INPUT_MANIFEST = _replay_write_input_manifest(TEST_DIR, WORKING_DIR)"
    )
    if source.count(input_manifest_marker) != 1:
        raise RuntimeError("exp015 input-manifest marker changed")
    source = source.replace(
        input_manifest_marker,
        "_REPLAY_INPUT_MANIFEST = {'stage': 'recovered_repaired_graph_evaluation'}",
        1,
    )

    cuda_guard = (
        "# Require a CUDA device before launching GPU inference\n"
        "if not _torch.cuda.is_available():\n"
        "    raise RuntimeError('CUDA GPU is required for this notebook. "
        "Enable a Kaggle GPU accelerator and commit again.')\n\n"
        "print('CUDA device:', _torch.cuda.get_device_name(0))"
    )
    cpu_evaluation_guard = """# Saved graph evaluation does not launch GPU inference.
print('CUDA device: not required for recovered repaired-graph evaluation')"""
    if source.count(cuda_guard) != 1:
        raise RuntimeError("exp015 CUDA preflight marker changed")
    source = source.replace(cuda_guard, cpu_evaluation_guard, 1)

    prediction_start_marker = (
        "start_time = time.time()\navailable_gpu_count = _torch.cuda.device_count()"
    )
    prediction_end_marker = "print(f'Prediction completed in {predict_seconds / 60:.2f} minutes')"
    if source.count(prediction_start_marker) != 1 or source.count(prediction_end_marker) != 1:
        raise RuntimeError("exp015 prediction execution markers changed")
    prediction_start = source.index(prediction_start_marker)
    prediction_end = source.index(prediction_end_marker, prediction_start) + len(
        prediction_end_marker
    )
    recovered_prediction = """start_time = time.time()
predict_seconds = 0.0
print('Skipping prediction: using checksum-verified repaired compact graphs')"""
    source = source[:prediction_start] + recovered_prediction + source[prediction_end:]

    repair_start_marker = "DEEPCENTER_VETO_DETECTOR = load_deepcenter_veto_detector()"
    if source.count(repair_start_marker) != 1:
        raise RuntimeError("exp015 graph-repair execution marker changed")
    repair_start = source.index(repair_start_marker)
    source = source[:repair_start] + (
        "print('Skipping graph repair: using checksum-verified Kaggle v5 outputs')\n"
    )
    compile(source, "exp015_repaired_graph_evaluation_setup.py", "exec")
    return source


def force_public_evaluator_metadata_only(public_evaluator: Any) -> None:
    """Prevent the graph-only public evaluator from loading image tensors.

    The pinned evaluator only reads ``Dataset.scale`` and ``Dataset.tracks``.
    Its default ``open_dataset`` call nevertheless loads and pins the full image,
    which fails on a CPU-only Kaggle runtime. Passing ``load_image=False`` keeps
    those two graph-evaluation inputs unchanged.
    """
    original_open_dataset = public_evaluator.open_dataset

    def open_dataset_metadata_only(*args: object, **kwargs: object) -> object:
        if "load_image" in kwargs and kwargs["load_image"] is not False:
            raise RuntimeError("public evaluator attempted to override metadata-only loading")
        kwargs["load_image"] = False
        dataset = original_open_dataset(*args, **kwargs)
        if getattr(dataset, "image", None) is not None:
            raise RuntimeError("metadata-only public evaluator unexpectedly loaded image data")
        return dataset

    public_evaluator.open_dataset = open_dataset_metadata_only
