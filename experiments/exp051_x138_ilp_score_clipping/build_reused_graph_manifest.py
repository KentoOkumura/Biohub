"""Verify the published version-2 graph output before resuming evaluation."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "artifacts/kaggle_evaluate"
OUTPUT = EVIDENCE / "v2_output"
GRAPH_ROOT = OUTPUT / "exp051_clipping"
INDEX = EVIDENCE / "v2_remote_output_index.json"
LOG = EVIDENCE / "evaluate_v2.log"
MANIFEST = ROOT / "assets/reused_graph_manifest.json"
ARMS = ("expanded_optuna_cost", "expanded_optuna_cost_clipped")
STAGES = (
    "candidate",
    "ilp",
    "edge_filter",
    "relink",
    "gap1",
    "gap2",
    "low_detection",
    "division",
    "short_track",
    "final",
)
MISSING = "6bba_fe670320"


def content_sha256(folder: Path) -> str:
    files = sorted(path for path in folder.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"empty folder: {folder}")
    return hashlib.sha256(
        b"".join(
            path.relative_to(folder).as_posix().encode()
            + b"\0"
            + hashlib.sha256(path.read_bytes()).digest()
            for path in files
        )
    ).hexdigest()


def main() -> None:
    remote = json.loads(INDEX.read_text())
    if (
        remote["kernel_id"] != "kentookumura/exp051-x138-ilp-score-clipping-evaluate"
        or remote["kernel_version"] != 2
    ):
        raise ValueError("wrong remote kernel version")
    names = set(remote["files"])
    graph_names = {p for p in names if "/final_graphs/" in p}
    stage_names = {p for p in names if "/stages/" in p}
    for published, count in ((graph_names, 798), (stage_names, 380)):
        local = {
            p.relative_to(OUTPUT).as_posix()
            for p in OUTPUT.rglob("*")
            if p.is_file() and p.relative_to(OUTPUT).as_posix() in published
        }
        if len(published) != count or local != published:
            raise ValueError(
                f"published/local file mismatch: {len(published)}, {len(local)}, expected {count}"
            )
    rows = re.findall(r"^OUTER_GRAPH (\S+) (\S+) (\S+) ([0-9.]+)$", LOG.read_text(), re.M)
    if len(rows) != 38 or len(set((stem, arm) for stem, arm, _, _ in rows)) != 38:
        raise ValueError("expected 38 distinct completed graph records")
    stems = sorted(set(stem for stem, _, _, _ in rows))
    if (
        len(stems) != 19
        or MISSING in stems
        or sum(x.startswith("44b6_") for x in stems) != 10
        or sum(x.startswith("6bba_") for x in stems) != 9
    ):
        raise ValueError("completed video coverage differs")
    if {(stem, arm) for stem, arm, _, _ in rows} != {(stem, arm) for stem in stems for arm in ARMS}:
        raise ValueError("completed videos are not paired")
    graphs = {}
    for stem, arm, status, seconds in rows:
        if status not in ("OPTIMAL", "TIMELIMIT"):
            raise ValueError(f"unexpected solver status: {status}")
        graph = GRAPH_ROOT / arm / "final_graphs" / f"{stem}.geff"
        stage = GRAPH_ROOT / "stages" / arm / stem
        if not graph.is_dir() or {p.name for p in stage.iterdir()} != {
            f"{name}.npz" for name in STAGES
        }:
            raise ValueError(f"incomplete graph or stage: {arm}/{stem}")
        graphs[f"{arm}/{stem}"] = {
            "graph_sha256": content_sha256(graph),
            "stages_sha256": content_sha256(stage),
            "solver_status": status,
            "ilp_seconds": float(seconds),
        }
    selection = EVIDENCE / "v2_metadata/exp051_selection.json"
    config = OUTPUT / "config.yaml"
    if not selection.is_file() or not config.is_file():
        raise ValueError("missing published provenance")
    manifest = {
        "source_kernel_id": remote["kernel_id"],
        "source_kernel_version": remote["kernel_version"],
        "selection_sha256": hashlib.sha256(selection.read_bytes()).hexdigest(),
        "source_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "source_log_sha256": hashlib.sha256(LOG.read_bytes()).hexdigest(),
        "remote_output_index_sha256": hashlib.sha256(INDEX.read_bytes()).hexdigest(),
        "missing_stems": [MISSING],
        "reused_stems": stems,
        "graphs": graphs,
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
                "graph_count": len(graphs),
                "stage_file_count": len(stage_names),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
