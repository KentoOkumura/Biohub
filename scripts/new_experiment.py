from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
from collections.abc import Callable
from datetime import date
from pathlib import Path

import yaml

try:
    from .config_utils import ROOT, configured_project_path
except ImportError:  # Direct execution: `uv run python scripts/new_experiment.py`
    from config_utils import ROOT, configured_project_path

GENERATED_DIRS = ("artifacts",)
LEGACY_GENERATED_DIRS = ("features", "variants")
IGNORED_DIRS = (*GENERATED_DIRS, *LEGACY_GENERATED_DIRS, "kaggle")
RESET_RECORD_FILES = (
    "README.md",
    "requirements.md",
    "SESSION_NOTES.md",
    "result.md",
    "metrics.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a new experiment from a template.")
    parser.add_argument("--name", required=True, help="Experiment name, e.g. expXXX_next_idea")
    parser.add_argument(
        "--source",
        default="templates/experiment",
        help="Template directory or existing experiment directory",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing experiment directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the destination without writing files",
    )
    parser.add_argument(
        "--copy-tests",
        action="store_true",
        help="Copy tests from a source experiment; review copied paths and contracts manually",
    )
    return parser.parse_args()


def validate_copy_paths(source: Path, destination: Path, *, force: bool) -> None:
    if not source.is_dir():
        raise NotADirectoryError(f"Template/source is not a directory: {source}")
    source_path = source.resolve()
    destination_path = destination.resolve()
    if source_path.is_relative_to(destination_path) or destination_path.is_relative_to(source_path):
        raise ValueError("source and destination must not be the same or contain one another")
    if (destination.exists() or destination.is_symlink()) and not force:
        raise FileExistsError(f"{destination} already exists. Use --force to overwrite.")


def copy_tree(
    source: Path,
    destination: Path,
    force: bool,
    copy_tests: bool,
    *,
    postprocess: Callable[[Path], None] | None = None,
) -> None:
    validate_copy_paths(source, destination, force=force)
    ignored_names = [
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        *IGNORED_DIRS,
    ]
    experiments_dir = configured_project_path("paths.experiments_dir", "experiments", root=ROOT)
    if source.resolve().parent == experiments_dir.resolve() and not copy_tests:
        ignored_names.append("tests")
    ignore = shutil.ignore_patterns(*ignored_names)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}-", dir=destination.parent
    ) as temp:
        staged = Path(temp) / "experiment"
        shutil.copytree(source, staged, ignore=ignore)
        for dirname in GENERATED_DIRS:
            generated_dir = staged / dirname
            generated_dir.mkdir(parents=True, exist_ok=True)
            (generated_dir / ".gitkeep").touch()
        if postprocess is not None:
            postprocess(staged)

        if not force:
            # copytree creates the destination exclusively. A rename could replace
            # an empty directory created by another writer after our initial check.
            # Leave a partial destination intact on failure; another writer may
            # already have added work to it.
            shutil.copytree(staged, destination)
            return

        # Keep the backup outside automatically cleaned staging, so even an
        # interrupted publication or failed restoration leaves recoverable work.
        backup_dir: Path | None = None
        backup: Path | None = None
        if destination.exists() or destination.is_symlink():
            backup_dir = Path(
                tempfile.mkdtemp(prefix=f".{destination.name}-backup-", dir=destination.parent)
            )
            backup = backup_dir / "previous"
            try:
                destination.rename(backup)
            except BaseException:
                backup_dir.rmdir()
                raise
        try:
            staged.rename(destination)
        except BaseException:
            if backup is not None:
                try:
                    backup.rename(destination)
                except BaseException as exc:
                    raise RuntimeError(
                        "copy publication and restoration failed; "
                        f"original work is preserved at {backup}"
                    ) from exc
                assert backup_dir is not None
                backup_dir.rmdir()
            raise
        else:
            if backup_dir is not None:
                shutil.rmtree(backup_dir)


def replace_text(path: Path, replacements: tuple[tuple[str, str], ...]) -> None:
    for file_path in path.rglob("*"):
        if not file_path.is_file():
            continue
        try:
            text = file_path.read_text()
        except UnicodeDecodeError:
            continue
        updated = text
        for old, new in replacements:
            updated = updated.replace(old, new)
        if updated != text:
            file_path.write_text(updated)


def rename_paths(path: Path, replacements: tuple[tuple[str, str], ...]) -> None:
    for file_path in sorted(path.rglob("*"), key=lambda value: len(value.parts), reverse=True):
        new_name = file_path.name
        for old, new in replacements:
            new_name = new_name.replace(old, new)
        if new_name != file_path.name:
            file_path.rename(file_path.with_name(new_name))


def replace_tokens(path: Path, experiment_name: str) -> None:
    today = date.today().isoformat()
    replacements = (
        ("{{ EXPERIMENT_NAME }}", experiment_name),
        ("{{EXPERIMENT_NAME}}", experiment_name),
        ("{{ TODAY }}", today),
        ("{{TODAY}}", today),
    )
    replace_text(path, replacements)
    rename_paths(path, replacements)


def replace_parent_experiment_identity(
    destination: Path,
    source_experiment: str,
    experiment_name: str,
) -> None:
    replacements = ((source_experiment, experiment_name),)
    replace_text(destination, replacements)
    rename_paths(destination, replacements)


def reset_parent_records(
    destination: Path,
    experiment_name: str,
    source_experiment: str,
) -> None:
    template_dir = ROOT / "templates" / "experiment"
    missing = [name for name in RESET_RECORD_FILES if not (template_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"experiment templates missing: {', '.join(missing)}")

    for filename in RESET_RECORD_FILES:
        text = (template_dir / filename).read_text()
        text = text.replace("{{ EXPERIMENT_NAME }}", experiment_name)
        text = text.replace("{{EXPERIMENT_NAME}}", experiment_name)
        text = text.replace("{{ TODAY }}", date.today().isoformat())
        text = text.replace("{{TODAY}}", date.today().isoformat())
        (destination / filename).write_text(text)

    config_path = destination / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError("source experiment is missing config.yaml")
    config = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(config, dict):
        raise ValueError("source experiment config.yaml must contain a mapping")
    experiment = config.setdefault("experiment", {})
    if not isinstance(experiment, dict):
        raise ValueError("source experiment config.yaml experiment must contain a mapping")
    experiment["name"] = experiment_name
    experiment["description"] = "TODO"
    experiment["created_at"] = date.today().isoformat()
    experiment.pop("updated_at", None)
    experiment.pop("status", None)
    lineage = config.setdefault("lineage", {})
    if not isinstance(lineage, dict):
        raise ValueError("source experiment config.yaml lineage must contain a mapping")
    lineage["parent"] = source_experiment
    lineage["hypothesis_id"] = "TODO"
    lineage["backlog_candidate"] = "TODO"
    lineage["diff_summary"] = "TODO"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))

    metrics = json.loads((destination / "metrics.json").read_text())
    if metrics.get("experiment") != experiment_name or metrics.get("status") != "planned":
        raise RuntimeError("reset metrics.json did not produce a planned child experiment")


def validate_experiment_name(name: str, experiments_dir: Path) -> None:
    match = re.fullmatch(r"(exp[A-Za-z]?\d+)_[a-zA-Z0-9_-]+", name)
    if match is None:
        raise ValueError("experiment name must use an ID and suffix, e.g. exp054_example")
    experiment_id = match.group(1).lower()
    collisions = sorted(
        path.name
        for path in experiments_dir.glob("*")
        if path.is_dir()
        and path.name != name
        and path.name.split("_", 1)[0].lower() == experiment_id
    )
    if collisions:
        raise ValueError(f"experiment ID {experiment_id} already exists: {', '.join(collisions)}")


def display_path(path: Path) -> str:
    return str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)


def main() -> None:
    args = parse_args()
    source = (ROOT / args.source).resolve()
    experiments_dir = configured_project_path(
        "paths.experiments_dir", "experiments", root=ROOT
    ).resolve()
    validate_experiment_name(args.name, experiments_dir)
    destination = experiments_dir / args.name

    validate_copy_paths(source, destination, force=args.force or args.dry_run)

    if args.dry_run:
        print(f"Source: {display_path(source)}")
        print(f"Destination: {display_path(destination)}")
        print(f"Exists: {destination.exists()}")
        return

    def initialize_records(staged: Path) -> None:
        if source.parent == experiments_dir:
            replace_parent_experiment_identity(staged, source.name, args.name)
            reset_parent_records(staged, args.name, source.name)
        else:
            replace_tokens(staged, args.name)

    copy_tree(source, destination, args.force, args.copy_tests, postprocess=initialize_records)

    print(f"Created {display_path(destination)}")
    if source.parent == experiments_dir and not args.copy_tests:
        print("Source experiment tests were not copied. Add tests for the new experiment contract.")
    elif source.parent == experiments_dir and args.copy_tests:
        print("Source experiment tests were copied; review old experiment paths and expectations.")
    if source.parent == experiments_dir:
        print(
            "Parent experiment identity was replaced and execution records were reset to planned. "
            "Review copied input paths and contracts before implementation."
        )
    print(
        "Next: fill requirements.md, then implement and validate the experiment. "
        "Shared competition defaults are inherited from project.yml."
    )


if __name__ == "__main__":
    main()
