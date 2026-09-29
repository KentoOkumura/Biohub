"""CPU-only legacy ILP worker for exp050's frozen Optuna-cost submission."""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Executor
from typing import TypeVar

import numpy as np

_Input = TypeVar("_Input")
_Output = TypeVar("_Output")


class SolverWarnings(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno >= logging.WARNING:
            self.messages.append(record.getMessage())


def _solver_dependencies():
    import polars as pl
    import tracksdata as td
    from tracksdata.utils._logging import LOG

    return td, pl, LOG


def legacy_solve(
    coords: np.ndarray,
    edges: np.ndarray,
    costs: tuple[float, float, float],
    stem: str,
    timeout_seconds: float,
) -> tuple[dict[int, dict[str, object]], list[dict[str, object]], dict[str, object]]:
    """Run the original tracksdata objective without any GPU or postprocessing."""
    td, pl, LOG = _solver_dependencies()

    graph = td.graph.InMemoryGraph()
    for key in ("z", "y", "x"):
        graph.add_node_attr_key(key, pl.Float64, -999999.0)
    ids = graph.bulk_add_nodes(
        [{"t": int(t), "z": float(z), "y": float(y), "x": float(x)} for t, z, y, x in coords]
    )
    graph.add_edge_attr_key("edge_prob", pl.Float64, 0.0)
    graph.add_edge_attr_key("edge_dist", pl.Float64, 0.0)
    if len(edges):
        graph.bulk_add_edges(
            [
                {
                    "source_id": ids[int(s)],
                    "target_id": ids[int(t)],
                    "edge_prob": float(p),
                    "edge_dist": float(d),
                }
                for s, t, p, d in edges
            ]
        )
    solver = td.solvers.ILPSolver(
        edge_weight=-1.0 * td.EdgeAttr("edge_prob"),
        appearance_weight=float(costs[0]),
        disappearance_weight=float(costs[1]),
        division_weight=float(costs[2]),
        timeout=float(timeout_seconds),
    )
    handler = SolverWarnings()
    LOG.addHandler(handler)
    start = time.monotonic()
    try:
        solution = solver.solve(graph)
    finally:
        LOG.removeHandler(handler)
    elapsed = time.monotonic() - start
    nonoptimal_warnings = [w for w in handler.messages if "did not converge" in w]
    timeout_incumbent = bool(nonoptimal_warnings) and all(
        "SolverStatus.TIMELIMIT" in warning for warning in nonoptimal_warnings
    )
    if (
        solution is None
        or any("Trivial solution" in warning for warning in handler.messages)
        or (nonoptimal_warnings and not timeout_incumbent)
    ):
        raise RuntimeError(
            f"{stem}: legacy ILP has no approved feasible solution after {elapsed:.1f}s: "
            f"{handler.messages}"
        )
    nodes = {
        int(row["node_id"]): {
            "node_id": int(row["node_id"]),
            "t": int(row["t"]),
            "z": float(row["z"]),
            "y": float(row["y"]),
            "x": float(row["x"]),
        }
        for row in solution.node_attrs().iter_rows(named=True)
    }
    selected = [
        {
            "source_id": int(row["source_id"]),
            "target_id": int(row["target_id"]),
            "edge_prob": float(row["edge_prob"]),
        }
        for row in solution.edge_attrs().iter_rows(named=True)
    ]
    if not nodes:
        raise RuntimeError(f"{stem}: ILP returned no selected nodes")
    incoming, outgoing = {}, {}
    for edge in selected:
        source, target = edge["source_id"], edge["target_id"]
        if (
            source not in nodes
            or target not in nodes
            or nodes[source]["t"] + 1 != nodes[target]["t"]
        ):
            raise RuntimeError(f"{stem}: ILP returned an invalid edge")
        incoming[target] = incoming.get(target, 0) + 1
        outgoing[source] = outgoing.get(source, 0) + 1
        if incoming[target] > 1 or outgoing[source] > 2:
            raise RuntimeError(f"{stem}: ILP returned an invalid graph degree")
    return (
        nodes,
        selected,
        {
            "seconds": elapsed,
            "solver_limit_seconds": float(timeout_seconds),
            "solver_status": "TIMELIMIT_FEASIBLE" if timeout_incumbent else "OPTIMAL",
            "optimality_proven": not timeout_incumbent,
            "warnings": handler.messages,
            "candidate_edges": len(edges),
            "selected_edges": len(selected),
        },
    )


def solve_prepared(
    stem: str,
    payload: tuple[np.ndarray, np.ndarray],
    costs: tuple[float, float, float],
    timeout_seconds: float,
) -> tuple[dict[int, dict[str, object]], list[dict[str, object]], dict[str, object]]:
    coords, edges = payload
    return legacy_solve(coords, edges, costs, stem, timeout_seconds)


def ordered_bounded_solve(
    stems: Iterable[str],
    prepare: Callable[[str], _Input],
    solve: Callable[[str, _Input], _Output],
    executor: Executor,
    max_in_flight: int,
) -> Iterator[tuple[str, _Output]]:
    """Keep workers busy while yielding completed videos in stable CSV order."""
    if max_in_flight < 1:
        raise ValueError("max_in_flight must be positive")
    remaining = iter(stems)
    pending = deque()

    def fill() -> None:
        while len(pending) < max_in_flight:
            try:
                stem = next(remaining)
            except StopIteration:
                break
            payload = prepare(stem)
            pending.append((stem, executor.submit(solve, stem, payload)))

    fill()
    while pending:
        stem, future = pending.popleft()
        result = future.result()
        fill()
        yield stem, result
