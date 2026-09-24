"""Build a readable, statically inlined notebook from reachable experiment definitions."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODULES = (
    "velocity_history",
    "frozen_tracker",
    "past_candidate_attention",
    "past_candidate_data",
    "knn_diagnostics",
    "runtime_helpers",
    "settings",
    "attention_train_pipeline",
)


class FlattenImports(ast.NodeTransformer):
    def visit_Import(self, node: ast.Import) -> ast.AST | None:
        names = [name for name in node.names if name.name not in MODULES]
        return ast.Import(names=names) if names else None

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.AST | None:
        return None if node.module in MODULES or node.module == "__future__" else node

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        if isinstance(node.value, ast.Name) and node.value.id in {"base", "runtime"}:
            return ast.copy_location(ast.Name(id=node.attr, ctx=node.ctx), node)
        return self.generic_visit(node)


def references(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


def main() -> None:
    definitions = {}
    ordered = []
    imports = {}
    main_body = []
    for module in MODULES:
        tree = ast.parse((ROOT / f"{module}.py").read_text())
        tree = ast.fix_missing_locations(FlattenImports().visit(tree))
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports[ast.unparse(node)] = node
            elif isinstance(node, ast.FunctionDef) and node.name == "main":
                main_body = node.body
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign)):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    names = [node.name]
                else:
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    names = [target.id for target in targets if isinstance(target, ast.Name)]
                for name in names:
                    if name in definitions:
                        raise ValueError(f"duplicate global definition: {name}")
                    definitions[name] = node
                ordered.append((module, node, names))
    needed = set().union(*(references(node) for node in main_body))
    while True:
        expanded = needed | set().union(
            *(references(definitions[n]) for n in needed if n in definitions)
        )
        if expanded == needed:
            break
        needed = expanded
    header = """# %% [markdown]
# # exp038: nearest-past candidate attention
#
# 固定候補と教師を保ち、物理距離の近傍8点を選択してからattentionを計算する。
# 学習側の前身候補保持率99%、実行費用、両胚の接続診断を順に確認する。
# 全graph推論・公式score・submissionはこのNotebookでは実行しない。
#
# ## Contents
# 1. Imports
# 2. Reachable runtime, cache, teacher, model and diagnostic definitions
# 3. Setup and fixed contract
# 4. Input integrity and embryo splits
# 5. Models and loaders
# 6. Candidate retention and runtime benchmark
# 7. Two-fold training and saved-control comparison
# 8. Metrics, model manifest and artifacts

# %%
from __future__ import annotations
"""
    parts = [header, "\n".join(imports), "\nfrom contextlib import contextmanager\n"]
    last_module = None
    kept = []
    for module, node, names in ordered:
        if not needed.intersection(names):
            continue
        if module != last_module:
            parts.append(f"\n# %% [markdown]\n# ## Definitions: {module}\n\n# %%\n")
            last_module = module
        parts.append(ast.unparse(node) + "\n\n")
        kept.extend(names)
    parts.append("""
# %%
@contextmanager
def recorded_stage(name):
    print({"stage": name}, flush=True)
    try:
        yield
    except Exception as error:
        update_metrics(METRICS_PATH, {
            "status": "failed",
            "evidence": {
                "failure": {"stage": name, "type": type(error).__name__, "message": str(error)},
                "kaggle": {"notebook_runtime_seconds": time.perf_counter() - NOTEBOOK_STARTED}
            },
            "notes": "Stopped at " + name + ": " + str(error),
        })
        raise
""")
    groups = []
    group = []
    stage = "Setup and configuration"
    boundaries = {
        "cache_output_root": "Input cache and source integrity",
        "sample_names": "Annotation and fixed embryo splits",
        "public_src": "Public tracker and device",
        "feature_channels": "Models, loaders and checkpoint selection",
        "inventory": "Candidate counts and retention audit",
        "inventory_by_path": "Stratified runtime and stress benchmark",
        "training_started": "Two-fold training and saved-control comparison",
        "graph_gate_passed": "Graph progression diagnostic",
        "feature_schema": "Model manifest and experiment metrics",
    }
    for node in main_body:
        target = (
            node.targets[0].id
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
            else None
        )
        if target in boundaries:
            groups.append((stage, group))
            stage, group = boundaries[target], []
        group.append(node)
    groups.append((stage, group))
    for name, nodes in groups:
        if not nodes:
            continue
        body = "\n\n".join(ast.unparse(node) for node in nodes)
        parts.append(f"\n# %% [markdown]\n# ## {name}\n\n# %%\nwith recorded_stage({name!r}):\n")
        parts.append("\n".join("    " + line if line else "" for line in body.splitlines()) + "\n")
    text = "# %%\n# ruff: noqa: E501\n" + "\n".join(parts)
    if "__file__" in text:
        raise ValueError("notebook must not depend on __file__")
    ast.parse(text)
    output = ROOT / f"{ROOT.name}_train.py"
    output.write_text(text)
    subprocess.run(
        [sys.executable, "-m", "ruff", "check", str(output), "--fix"],
        check=True,
    )
    subprocess.run([sys.executable, "-m", "ruff", "format", str(output)], check=True)
    manifest = {
        "definitions": kept,
        "total_definitions": len(definitions),
        "inlined_count": len(kept),
        "stages": [name for name, _ in groups],
    }
    (ROOT / "artifacts" / "notebook_build_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps({"output": str(output), "inlined_definitions": len(kept)}, indent=2))


if __name__ == "__main__":
    main()
