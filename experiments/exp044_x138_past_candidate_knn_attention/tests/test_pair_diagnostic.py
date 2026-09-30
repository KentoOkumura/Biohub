from __future__ import annotations

import numpy as np
import torch

from diagnose_pair_effects import THRESHOLDS, add_pair, empty_counts, summarize


def test_threshold_sweep_matches_direct_pair_decoding() -> None:
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [-1.0, -1.0]])
    truth = np.array([[True, False], [False, True], [False, False]])
    counts = empty_counts()
    predicted = add_pair(counts, logits, truth)
    direct = torch.softmax(logits, dim=0).numpy() > np.float32(0.48)
    assert np.array_equal(predicted, direct)
    report = summarize(counts)
    fixed = report["at_threshold_0_48"]
    assert fixed["recovered_known_edges"] == int((direct & truth).sum())
    assert fixed["false_edges_on_active_pairs"] == int((direct & ~truth).sum())
    assert fixed["active_pair_errors"] == int((direct != truth).sum())
    assert len(report["threshold_curve"]) == len(THRESHOLDS)
    recalls = [row["known_edge_recall"] for row in report["threshold_curve"]]
    assert recalls == sorted(recalls, reverse=True)
