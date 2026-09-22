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
# # exp035 velocity features: nested-history tracker training
#
# exp015の固定候補・画像特徴とexp016の教師・lossを維持し、対象sampleを
# 学習に使わないfixed first-pass trackerの予測履歴から速度pair特徴を作る。
# inner cross-fittingで履歴modelを4個、外側胚分割でvelocity modelを2個学習し、
# 保存済みexp016 fold modelと同じ外側windowで比較する。
#
# この段階は隣接2-frameのpair診断だけを計算する。進行条件を満たすまで
# 全graph推論・公式評価へ進まず、Kaggle submissionも作成しない。

# %% [markdown]
# ## Run the audited training pipeline

# %%
from __future__ import annotations

from train_pipeline import main

main()
