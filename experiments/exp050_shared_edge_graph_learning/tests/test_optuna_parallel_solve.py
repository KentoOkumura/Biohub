"""The CPU ILP pool must preserve test-video order and keep work bounded."""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import get_context
from threading import Barrier

from experiments.exp050_shared_edge_graph_learning.exp050_optuna_solver_worker import (
    ordered_bounded_solve,
)


def _return_video(stem: str, payload: str) -> str:
    if stem == "first":
        time.sleep(0.1)
    return payload


def test_spawned_workers_return_in_original_video_order():
    stems = ["first", "second", "third"]
    with ProcessPoolExecutor(max_workers=2, mp_context=get_context("spawn")) as executor:
        result = list(
            ordered_bounded_solve(
                stems,
                lambda stem: stem.upper(),
                _return_video,
                executor,
                max_in_flight=2,
            )
        )
    assert result == [(stem, stem.upper()) for stem in stems]


def test_two_videos_start_before_first_result_is_consumed():
    both_started = Barrier(2)

    def solve(stem: str, payload: str) -> str:
        both_started.wait(timeout=5)
        return payload

    with ThreadPoolExecutor(max_workers=2) as executor:
        result = list(
            ordered_bounded_solve(["a", "b"], lambda stem: stem, solve, executor, max_in_flight=2)
        )
    assert result == [("a", "a"), ("b", "b")]
