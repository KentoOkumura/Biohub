"""Select maintained documents without rewriting archived source material."""

from __future__ import annotations

import subprocess
from pathlib import Path

try:
    from .config_utils import configured_project_path
except ImportError:
    from config_utils import configured_project_path


def is_archived_source(path: Path, root: Path) -> bool:
    docs = configured_project_path("paths.docs_dir", "docs", root=root)
    if path.is_relative_to(docs):
        relative = path.relative_to(docs)
        # Keep the authored indexes in both directories under validation.
        if relative.match("discussions/biohub-*.md"):
            return True
        if relative.parts[0] == "notebooks" and path.suffix == ".ipynb":
            return True
    # These inputs were frozen with hashes for the source-hidden evaluation.
    packet = root / "studies/biohub_source_hidden_eval_20261002/packet"
    return path.is_relative_to(packet)


def maintained_documents(root: Path, extensions: set[str]) -> list[Path]:
    output = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=root,
    )
    paths = {root / name for name in output.decode().split("\0") if name}
    return sorted(
        path
        for path in paths
        if path.suffix in extensions and path.is_file() and not is_archived_source(path, root)
    )
