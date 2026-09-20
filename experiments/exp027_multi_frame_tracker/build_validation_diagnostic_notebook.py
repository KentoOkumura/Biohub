"""Build a self-contained internal validation diagnostic from audited inference."""

import ast
import subprocess
import sys
from pathlib import Path

# ruff: noqa: E501
from build_diagnostic_notebook import extract

EXP = Path(__file__).resolve().parent


def main() -> None:
    source = (EXP / "exp027_multi_frame_tracker_diagnostic.py").read_text()
    source = source.split("# %% [markdown]\n# ## 6. Reproduce pilot metrics")[0]
    source = source.replace(
        "saved-model previous-context diagnostic", "internal-validation probability diagnostic"
    )
    source = source.replace(
        "# 6. Reproduce pilot metrics and save paired results",
        "# 6. Reproduce internal metrics and choose thresholds",
    )
    source = source.replace(
        'diag_cfg = config["diagnostic"]',
        'diag_cfg = config["diagnostic"]\nvalidation_cfg = config["validation_diagnostic"]',
    )
    begin = source.index("train_candidates = [")
    end = source.index("model_manifest = ", begin)
    source = (
        source[:begin]
        + """def resolve_notebook_root(reference: str, filename: str) -> Path:
    owner, slug = reference.split('/')
    roots = [Path('/kaggle/input') / slug, Path('/kaggle/input/notebooks') / owner / slug]
    return unique_existing([root for root in roots if (root / filename).is_file()], reference)


trained_root = resolve_notebook_root(diag_cfg['source_kernel'], 'model_manifest.json')
require_sha(trained_root / 'model_manifest.json', diag_cfg['model_manifest_sha256'])
"""
        + source[end:]
    )
    source = source.replace(
        "cache_output_root = resolve_cache_output_root()",
        "cache_output_root = resolve_notebook_root(cache_cfg['kernel_source'], cache_cfg['summary_file'])",
    )
    marker = 'OUTPUT = WORKING_ROOT / "context_diagnostic"'
    source = source.replace(
        marker,
        """require_sha(trained_root / 'split_manifest.json', validation_cfg['split_manifest_sha256'])
splits = json.loads((trained_root / 'split_manifest.json').read_text())
eligible_paths, _ = filter_nonempty_gt_window_paths(cache_paths, annotations)
selection = {'folds': []}
for record in splits:
    samples = set(record['internal_validation'])
    if samples & set(record['gradient_update']) or samples & set(record['outer_evaluation']):
        raise RuntimeError('internal validation overlaps gradient or outer samples')
    paths = [path for path in eligible_paths if path.parent.name in samples]
    if len(paths) != validation_cfg['expected_window_counts'][record['fold']]:
        raise RuntimeError('internal window count differs from train v4')
    selection['folds'].append({**record, 'evaluation_embryo': record['train_embryo'],
        'outer_embryo': record['evaluation_embryo'],
        'selected_windows': [p.relative_to(cache_root).as_posix() for p in paths]})
OUTPUT = WORKING_ROOT / 'validation_diagnostic' """,
    )
    source = source.replace(
        '    trained_root / "pilot_outer_window_selection.json", OUTPUT / "pilot_outer_window_selection.json"',
        '    trained_root / "split_manifest.json", OUTPUT / "split_manifest.json"',
    )
    source = source.replace(
        "Same pilot windows; no training; conditions:",
        "All internal validation windows; no training; conditions:",
    )
    begin = source.index('    if len(paths) != diag_cfg["windows_per_embryo"]')
    end = source.index("    if any(not path.parent.name.startswith", begin)
    source = source[:begin] + source[end:]
    source = source.replace(
        'payload["evaluation_embryo"] != embryo',
        'payload["evaluation_embryo"] != fold["outer_embryo"]\n            or payload["train_embryo"] != embryo',
    )
    source = source.replace(
        "mode_predictions.setdefault(absolute_index, {})[mode_name] = logits.cpu().numpy()",
        "mode_predictions.setdefault(absolute_index, {})[mode_name] = probabilities.cpu().numpy()",
    )
    source = source.replace('OUTPUT / "pair_logits"', 'OUTPUT / "active_scores"')
    begin = source.index('            payload = {f"logits_{name}"')
    end = source.index("            np.savez_compressed", begin)
    source = (
        source[:begin]
        + """            payload = {
                f'{key}_{name}': values
                for name, probability in predictions.items()
                for key, values in active_scores(probability, example['target']).items()
            }
"""
        + source[end:]
    )
    source = source.replace(
        'row["trained_outer_evaluation"]', 'row["epochs"][row["best_epoch"]]["internal_validation"]'
    )
    source = source.replace(
        "processed_windows = 0",
        "processed_windows = 0\ninference_seconds = 0.0\nsetup_seconds = 0.0\nloaded_windows = 0",
    )
    source = source.replace(
        'for fold in selection["folds"]:',
        'for fold in selection["folds"]:\n    fold_setup_started = time.perf_counter()',
    )
    source = source.replace(
        "    for offset in range(0, len(examples)",
        "    setup_seconds += time.perf_counter() - fold_setup_started\n    loaded_windows += len(examples)\n    for offset in range(0, len(examples)",
    )
    source = source.replace(
        "        chosen = examples[offset",
        "        batch_started = time.perf_counter()\n        chosen = examples[offset",
    )
    source = source.replace(
        "        processed_windows += len(chosen)",
        "        inference_seconds += time.perf_counter() - batch_started\n        processed_windows += len(chosen)",
    )
    source = source.replace(
        "                + elapsed\n                / processed_windows",
        "                + (inference_seconds / processed_windows + setup_seconds / loaded_windows)",
    )
    helper = extract(EXP / "frozen_tracker.py", ["filter_nonempty_gt_window_paths"])
    helper = ast.get_source_segment(
        helper,
        next(
            node
            for node in ast.parse(helper).body
            if isinstance(node, ast.FunctionDef) and node.name == "filter_nonempty_gt_window_paths"
        ),
    )
    analysis = extract(
        EXP / "validation_diagnostic.py",
        ["active_scores", "select_negative_budget_threshold", "describe_scores"],
    )
    source = source.replace(
        "# %% [markdown]\n# ## 4. Verify saved inputs",
        helper + "\n\n" + analysis + "\n\n# %% [markdown]\n# ## 4. Verify saved inputs",
    )
    source += """# %% [markdown]
# ## 6. Reproduce internal metrics and choose thresholds
# The threshold is a negative-score order statistic from internal videos only.
# Known positive recall and outer labels are never used to choose it.

# %%
reproduced = all(all(row['checks'].values()) for row in reference_checks)
write_json(OUTPUT / 'reference_reproduction.json', reference_checks)
if not reproduced:
    raise RuntimeError({'internal_reproduction_failed': reference_checks})
selected = {}
for fold in selection['folds']:
    fold_id, embryo = fold['fold'], fold['evaluation_embryo']
    scores_by_mode = {}
    for mode in mode_names:
        collected = {key: [] for key in ('positive', 'negative', 'known_child_confidence', 'known_child_correct')}
        for record in prediction_records:
            if Path(record['path']).parent.name.startswith(embryo + '_'):
                with np.load(OUTPUT / record['path'], allow_pickle=False) as data:
                    for key in collected:
                        collected[key].append(data[f'{key}_{mode}'])
        scores_by_mode[mode] = {key: np.concatenate(values) for key, values in collected.items()}
    budget = int(np.count_nonzero(scores_by_mode[validation_cfg['baseline_mode']]['negative'] > validation_cfg['baseline_threshold']))
    selected[str(fold_id)] = {'internal_embryo': embryo, 'outer_embryo': fold['outer_embryo'], 'negative_budget': budget, 'modes': {}}
    for mode, scores in scores_by_mode.items():
        threshold = select_negative_budget_threshold(scores['negative'], budget)
        selected[str(fold_id)]['modes'][mode] = {
            'selected_threshold': threshold,
            'fixed': describe_scores(scores, validation_cfg['baseline_threshold'], validation_cfg['probability_quantiles'], validation_cfg['confidence_bin_edges']),
            'selected': describe_scores(scores, threshold, validation_cfg['probability_quantiles'], validation_cfg['confidence_bin_edges'])}
        if selected[str(fold_id)]['modes'][mode]['selected']['false_positive_active_pair_count'] > budget:
            raise RuntimeError('negative budget violation')
write_json(OUTPUT / 'selected_thresholds.json', selected)
write_json(OUTPUT / 'prediction_manifest.json', prediction_records)
write_json(OUTPUT / 'input_manifest.json', input_records)
write_json(OUTPUT / 'window_diagnostics.json', all_window_rows)
write_csv(OUTPUT / 'known_edge_predictions.csv', all_edge_rows)
summary = {
    'reference_reproduced': reproduced, 'model_training_count': 0,
    'source_kernel_version': diag_cfg['source_kernel_version'],
    'split_manifest_sha256': validation_cfg['split_manifest_sha256'],
    'model_manifest_sha256': diag_cfg['model_manifest_sha256'],
    'weight_evidence': weight_evidence, 'summaries': all_summaries,
    'selected_thresholds': selected,
    'selected_thresholds_sha256': file_sha256(OUTPUT / 'selected_thresholds.json'),
    'prediction_manifest_sha256': file_sha256(OUTPUT / 'prediction_manifest.json'),
    'input_manifest_sha256': file_sha256(OUTPUT / 'input_manifest.json'),
    'known_edge_predictions_sha256': file_sha256(OUTPUT / 'known_edge_predictions.csv'),
    'prediction_content_sha256': json_sha256([{'path': r['path'], 'array_content_sha256': r['array_content_sha256']} for r in prediction_records]),
    'notebook_runtime_seconds': time.perf_counter() - NOTEBOOK_STARTED,
}
write_json(OUTPUT / 'diagnostic_summary.json', summary)
update_metrics(METRICS_PATH, {'status': 'debug_completed', 'validation_diagnostic': summary})
print(json.dumps({'reference_reproduced': reproduced, 'selected_thresholds': selected}, indent=2), flush=True)
print('Internal validation probability diagnostic complete.', flush=True)
"""
    destination = EXP / "exp027_multi_frame_tracker_validation_diagnostic.py"
    destination.write_text(source)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--fix",
            "--ignore=E501",
            "--cache-dir",
            "/tmp/ruff-cache",
            str(destination),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "format",
            "--cache-dir",
            "/tmp/ruff-cache",
            str(destination),
        ],
        check=True,
    )
    print(f"Wrote {destination.name}: {len(source.splitlines())} lines")


if __name__ == "__main__":
    main()
