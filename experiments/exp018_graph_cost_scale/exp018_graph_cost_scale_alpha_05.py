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

# %%
from exp018_graph_cost_scale_diagnostic import run_diagnostic

if __name__ == "__main__":
    SUMMARY = run_diagnostic(shard_name="alpha_05")
