# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # exp037 past-candidate attention training and pair diagnostics
#
# exp015の固定候補・画像特徴とexp016の教師・lossを維持し、中央の各接続候補に
# 対して直前時点の全候補との3点座標関係を学習可能なattentionで集約する。
# 保存済みexp016 fold modelを同じ外側windowで再評価し、新規model 2個と比較する。
# このNotebookは全graph推論、公式評価、submissionを実行しない。

# %% [markdown]
# ## Contents
#
# 1. Imports and runtime
# 2. Configuration and fixed contract
# 3. Input and saved-control checks
# 4. Training, runtime gate, and outer diagnostics
# 5. Metrics and generated artifacts

# %% [markdown]
# ## 1. Imports and runtime

# %%
from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml
from attention_train_pipeline import main
from runtime_helpers import (
    resolve_cache_output_root,
    resolve_control_output_root,
    resolve_public_artifact_root,
)

WORKING_ROOT = Path.cwd()
CONFIG_PATH = WORKING_ROOT / "config.yaml"
METRICS_PATH = WORKING_ROOT / "metrics.json"
print({"working_root": str(WORKING_ROOT), "cuda": torch.cuda.is_available()})

# %% [markdown]
# ## 2. Configuration and fixed contract

# %%
config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
contract = {
    "experiment": config["experiment"]["name"],
    "parent": config["lineage"]["parent"],
    "hypothesis_id": config["lineage"]["hypothesis_id"],
    "backlog_candidate": config["lineage"]["backlog_candidate"],
    "active_variants": config["model"]["training"]["active_variants"],
    "folds": config["validation"]["n_folds"],
    "epochs": config["model"]["training"]["epochs"],
    "control_retrain": config["model"]["control"]["retrain"],
    "added_parameter_count": config["model"]["past_candidate_attention"][
        "expected_added_parameter_count"
    ],
    "graph_gate": config["validation"]["graph_progression_gate"],
}
print(json.dumps(contract, indent=2, ensure_ascii=False))
assert contract["active_variants"] == ["past_candidate_attention"]
assert contract["folds"] == 2 and contract["epochs"] == 3
assert contract["control_retrain"] is False
assert contract["added_parameter_count"] == 1570

# %% [markdown]
# ## 3. Input and saved-control checks
#
# exp015 cache、exp016 model manifest、公開checkpointをKaggle inputから一意に解決する。
# 本処理では各中央windowと直前windowの重複時点についてcandidate ID・grid座標・
# 物理座標の完全一致を検査し、不一致なら学習を停止する。

# %%
cache_root = resolve_cache_output_root(config["data"]["cache"])
control_root = resolve_control_output_root(config["model"]["control"])
public_root = resolve_public_artifact_root(config["model"]["public_source"])
print(
    json.dumps(
        {
            "cache_root": str(cache_root),
            "control_root": str(control_root),
            "public_root": str(public_root),
        },
        indent=2,
    )
)

# %% [markdown]
# ## 4. Training, runtime gate, and outer diagnostics
#
# 64 windowと候補積の大きいwindowでforward/backwardを測定し、12時間gateを先に判定する。
# gate内なら、外側2foldで各3 epoch学習する。直前時点の候補を距離で制限せず、
# 全有効候補と「過去情報を使わない」選択肢を同じsoftmax分母へ含める。

# %%
main()

# %% [markdown]
# ## 5. Metrics and generated artifacts
#
# `training_summary.json`に両胚別のexp016対照、新model、差分、進行条件を保存する。
# model、manifest、入力・feature schema・予測のSHAは同じ実験のmetricsへ記録する。

# %%
summary = json.loads((WORKING_ROOT / "training_summary.json").read_text(encoding="utf-8"))
for fold in summary["folds"]:
    print(
        json.dumps(
            {
                "fold": fold["fold"],
                "evaluation_embryo": fold["evaluation_embryo"],
                "outer_metric_delta": fold["outer_metric_delta"],
                "graph_progression_gate": fold["graph_progression_gate"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
print(json.dumps(summary["graph_progression_gate"], indent=2, ensure_ascii=False))
