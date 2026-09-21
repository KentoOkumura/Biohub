from __future__ import annotations

import csv
import importlib.util
import json
import platform
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from prepare_division_diagnostic import OUTPUT, read_annotations, sha

EXP = Path(__file__).resolve().parent
SOURCE = EXP / "exp029_mother_daughter_set_selection_train.py"
SPEC = importlib.util.spec_from_file_location("exp029_saved_train", SOURCE)
assert SPEC is not None and SPEC.loader is not None
TRAIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAIN)


def inspect_parent(
    scores: np.ndarray,
    options: list[tuple[int, ...]],
    children: list[int],
    target_ids: np.ndarray,
    edge_cost: float,
    capacity: int,
) -> dict:
    truth = set(children)
    index = next((k for k, option in enumerate(options) if set(option) == truth), None)
    adjusted = np.asarray(
        [scores[k] - edge_cost * len(option) for k, option in enumerate(options)], dtype=np.float64
    )
    ranked = sorted(
        range(1, len(options)),
        key=lambda k: (-float(adjusted[k]), tuple(int(target_ids[j]) for j in options[k])),
    )
    kept = {0, *ranked[: capacity - 1]}
    best = int(np.argmax(adjusted))
    pair_indices = [k for k, option in enumerate(options) if len(option) == 2]
    single_indices = [k for k, option in enumerate(options) if len(option) == 1]
    shifted = np.exp(adjusted - adjusted.max())
    probabilities = shifted / shifted.sum()
    row = {
        "candidate_covered": index is not None,
        "candidate_set_count": len(options),
        "independent_choice": list(options[best]),
        "independent_cardinality": len(options[best]),
        "independent_correct": set(options[best]) == truth,
        "true_set_rank_nonempty": None,
        "true_set_retained": False,
        "true_set_probability": None,
        "pair_probability_mass": float(probabilities[pair_indices].sum()),
        "true_minus_best_own_subset": None,
        "true_minus_best_singleton": None,
        "true_minus_best_pair": None,
        "true_subset_dominated": False,
        "true_minus_empty": None,
    }
    if index is not None:
        own_subsets = [
            k for k, option in enumerate(options) if len(option) < 2 and set(option).issubset(truth)
        ]
        margin = float(adjusted[index] - adjusted[own_subsets].max())
        row.update(
            {
                "true_set_rank_nonempty": ranked.index(index) + 1,
                "true_set_retained": index in kept,
                "true_set_probability": float(probabilities[index]),
                "true_minus_best_own_subset": margin,
                "true_subset_dominated": margin < -1e-6,
                "true_minus_best_singleton": float(
                    adjusted[index] - adjusted[single_indices].max()
                ),
                "true_minus_best_pair": float(adjusted[index] - adjusted[pair_indices].max()),
                "true_minus_empty": float(adjusted[index] - adjusted[0]),
            }
        )
    return row


def aggregate(rows: list[dict]) -> dict:
    result = {
        "known_divisions": len(rows),
        "candidate_covered": sum(r["candidate_covered"] for r in rows),
        "true_set_retained": sum(r["true_set_retained"] for r in rows),
        "independent_correct": sum(r["independent_correct"] for r in rows),
        "true_subset_dominated": sum(r["true_subset_dominated"] for r in rows),
        "decoded_correct": sum(r["decoded_correct"] for r in rows),
        "full_catalog_decoded_correct": sum(
            r["full_catalog_decoded_correct"] is True for r in rows
        ),
        "full_catalog_evaluated": sum(r["full_catalog_decoded_correct"] is not None for r in rows),
        "independent_cardinality": dict(Counter(r["independent_cardinality"] for r in rows)),
        "decoded_cardinality": dict(Counter(r["decoded_cardinality"] for r in rows)),
        "decoded_correct_daughters": dict(Counter(r["decoded_correct_daughters"] for r in rows)),
    }
    for column in (
        "true_set_rank_nonempty",
        "true_set_probability",
        "true_minus_best_own_subset",
        "true_minus_best_singleton",
        "true_minus_best_pair",
        "pair_probability_mass",
    ):
        values = [r[column] for r in rows if r[column] is not None]
        result[column] = (
            dict(
                zip(
                    ("min", "p25", "median", "p75", "max"),
                    map(float, np.quantile(values, [0, 0.25, 0.5, 0.75, 1])),
                    strict=True,
                )
            )
            if values
            else None
        )
    return result


def main() -> None:
    import torch

    started = time.perf_counter()
    torch.set_num_threads(4)
    config = yaml.safe_load((EXP / "config.yaml").read_text(encoding="utf-8"))
    manifest_path = EXP / "artifacts/train_v4/model_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    files = json.loads((OUTPUT / "input_files.json").read_text())
    for row in files["files"]:
        assert sha(OUTPUT / "input" / row["path"]) == row["sha256"]
    cache_summary = json.loads((OUTPUT / "input/window_cache_summary.json").read_text())
    unsigned = {k: v for k, v in cache_summary.items() if k != "summary_sha256"}
    assert cache_summary["summary_sha256"] == TRAIN.json_sha256(unsigned)
    assert cache_summary["summary_sha256"] == config["data"]["cache"]["summary_sha256"]
    assert cache_summary["cache_identity_sha256"] == config["data"]["cache"]["identity_sha256"]
    annotations = read_annotations(
        OUTPUT / "input/train", config["data"]["annotation"]["voxel_scale_zyx_um"]
    )
    paths = sorted((OUTPUT / "input/window_cache").glob("*/*.npz"))
    cache_source = EXP.parent / "exp015_oracle_stage_limits/window_cache.py"
    cache_spec = importlib.util.spec_from_file_location("exp015_cache_verify", cache_source)
    assert cache_spec is not None and cache_spec.loader is not None
    cache_module = importlib.util.module_from_spec(cache_spec)
    cache_spec.loader.exec_module(cache_module)
    for path in paths:
        cache_module.read_window_cache(
            path,
            expected_metadata={
                "dataset": path.parent.name,
                "primary_checkpoint_sha256": config["data"]["cache"]["primary_checkpoint_sha256"],
            },
        )
    models = {}
    splits = {}
    histories = {}
    for fold in config["validation"]["outer_folds"]:
        number = fold["fold"]
        path = EXP / "artifacts/train_v4/models" / manifest["folds"][number]["model_file"]
        assert sha(path) == manifest["folds"][number]["model_sha256"]
        model = TRAIN.make_model(config["model"])
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        model.eval()
        models[number] = model
        names = TRAIN.split_samples(
            sorted(annotations),
            fold["train_embryo"],
            fold["evaluation_embryo"],
            config["validation"]["internal_split_seed"],
            config["validation"]["internal_selection_fraction"],
        )
        splits[number] = {sample: split for split, samples in names.items() for sample in samples}
        hist = json.loads((EXP / f"artifacts/train_v4/fold_{number}_summary.json").read_text())
        histories[str(number)] = [
            {
                "epoch": h["epoch"],
                "loss": h["train"]["loss"],
                "train_teacher": h["train"]["set_teacher"],
                "internal_selection_accuracy": h["internal"]["selected"]["selection_accuracy"],
                "internal_known_divisions": h["internal"]["selected"]["known_division_parents"],
                "internal_recovered_divisions": h["internal"]["selected"][
                    "recovered_division_parents"
                ],
            }
            for h in hist["history"]
        ]
    rows = []
    window_counts = defaultdict(Counter)
    checks = Counter()
    with torch.inference_mode():
        for step, path in enumerate(paths):
            item = TRAIN.make_example(
                path,
                annotations[path.parent.name],
                config["data"]["cache"]["primary_checkpoint_sha256"],
                config["data"]["teacher"]["max_matching_distance_um"],
                config["model"]["candidate_sets"],
            )
            by_parent = defaultdict(list)
            for child, parent in enumerate(item["labels"]):
                if 0 <= parent < len(item["candidate_ids_src"]):
                    by_parent[int(parent)].append(child)
            divisions = {
                parent: children for parent, children in by_parent.items() if len(children) >= 2
            }
            checks["windows_with_gt_division"] += 1
            if not divisions:
                continue
            checks["windows_with_matched_division"] += 1
            batch = TRAIN.collate_examples([item])
            for number, model in models.items():
                split = splits[number][item["sample"]]
                logits = model(batch)[0].numpy()
                edge_cost = manifest["folds"][number]["edge_cost"]
                kwargs = {
                    "edge_cost": edge_cost,
                    "time_limit_seconds": config["model"]["decoding"][
                        "time_limit_seconds_per_window"
                    ],
                    "relative_gap": config["model"]["decoding"]["relative_gap"],
                }
                choices, _ = TRAIN.decode_daughter_sets(
                    logits,
                    item["daughter_sets"],
                    item["candidate_ids_src"],
                    item["candidate_ids_tgt"],
                    maximum_decode_sets_per_mother=16,
                    **kwargs,
                )
                full_choices = None
                if split == "outer":
                    full_choices, _ = TRAIN.decode_daughter_sets(
                        logits,
                        item["daughter_sets"],
                        item["candidate_ids_src"],
                        item["candidate_ids_tgt"],
                        maximum_decode_sets_per_mother=37,
                        **kwargs,
                    )
                group = f"fold_{number}_{split}"
                window_counts[group]["windows"] += 1
                selected = Counter(int(x) for x in choices if x >= 0)
                window_counts[group]["mothers"] += len(item["daughter_sets"])
                window_counts[group]["decoded_two_daughter_mothers"] += sum(
                    value == 2 for value in selected.values()
                )
                for parent, options in enumerate(item["daughter_sets"]):
                    best = max(
                        range(len(options)),
                        key=lambda k: float(logits[parent, k]) - edge_cost * len(options[k]),
                    )
                    window_counts[group]["independent_two_daughter_mothers"] += (
                        len(options[best]) == 2
                    )
                for parent, children in divisions.items():
                    assert len(children) == 2
                    detail = inspect_parent(
                        logits[parent],
                        item["daughter_sets"][parent],
                        children,
                        item["candidate_ids_tgt"],
                        edge_cost,
                        16,
                    )
                    chosen_children = np.flatnonzero(choices == parent).tolist()
                    detail.update(
                        {
                            "fold": number,
                            "split": split,
                            "embryo": item["sample"][:4],
                            "sample": item["sample"],
                            "source_frame": item["frames"][0],
                            "mother_candidate_id": int(item["candidate_ids_src"][parent]),
                            "mother_gt_id": int(item["source_matches"][parent]),
                            "true_daughter_indices": children,
                            "true_daughter_candidate_ids": [
                                int(item["candidate_ids_tgt"][c]) for c in children
                            ],
                            "decoded_daughters": chosen_children,
                            "decoded_cardinality": len(chosen_children),
                            "decoded_correct_daughters": len(set(children) & set(chosen_children)),
                            "decoded_correct": set(children) == set(chosen_children),
                            "full_catalog_decoded_correct": (
                                all(full_choices[c] == parent for c in children)
                                if full_choices is not None
                                else None
                            ),
                        }
                    )
                    rows.append(detail)
            if step % 10 == 0:
                print(
                    f"Analyzed {step + 1}/{len(paths)} windows; {len(rows)} division records",
                    flush=True,
                )
    groups = defaultdict(list)
    for row in rows:
        groups[f"fold_{row['fold']}_{row['split']}"].append(row)
    aggregated = {key: aggregate(value) for key, value in sorted(groups.items())}
    for number, expected_known, expected_covered in ((0, 108, 107), (1, 22, 21)):
        outer = aggregated[f"fold_{number}_outer"]
        assert outer["known_divisions"] == expected_known
        assert outer["candidate_covered"] == expected_covered
        assert outer["decoded_correct"] == 0, "CPU replay differs from saved Kaggle divisions"
    summary = {
        "experiment": config["experiment"]["name"],
        "diagnostic_only": True,
        "scope": "all adjacent GT-division windows, both saved models; no training",
        "elapsed_seconds": time.perf_counter() - started,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
            "threads": torch.get_num_threads(),
        },
        "sources": {
            "model_manifest_sha256": sha(manifest_path),
            "train_source_sha256": sha(SOURCE),
            "diagnostic_source_sha256": sha(Path(__file__)),
            "config_sha256": sha(EXP / "config.yaml"),
            "input_manifest_sha256": sha(OUTPUT / "input_files.json"),
            "cache_summary_sha256": cache_summary["summary_sha256"],
            "geff_archive_sha256": files["geff_archive_sha256"],
        },
        "coverage": dict(checks),
        "by_fold_split": aggregated,
        "window_counts": {key: dict(value) for key, value in sorted(window_counts.items())},
        "training_history": histories,
        "limitations": [
            "CPU replay of saved weights; no new official graph score.",
            "All-mother cardinalities cover only windows with known divisions.",
            "Training and internal splits are diagnostic only, not held-out performance.",
            "Dominance identifies an inference-stage obstruction, not the training cause.",
        ],
    }
    row_path = OUTPUT / "division_rows.json"
    row_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    summary["division_rows_sha256"] = sha(row_path)
    with (OUTPUT / "division_rows.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "elapsed_seconds": summary["elapsed_seconds"],
                "by_fold_split": aggregated,
                "window_counts": summary["window_counts"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
