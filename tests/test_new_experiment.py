from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

from scripts import new_experiment


def test_existing_number_cannot_be_reused_for_a_different_experiment(tmp_path):
    (tmp_path / "exp025_original").mkdir()
    with pytest.raises(ValueError, match="ID exp025 already exists"):
        new_experiment.validate_experiment_name("exp025_other", tmp_path)
    new_experiment.validate_experiment_name("exp025_original", tmp_path)
    new_experiment.validate_experiment_name("exp026_new", tmp_path)


def test_experiment_name_cannot_escape_the_experiments_directory(tmp_path):
    with pytest.raises(ValueError, match="experiment name"):
        new_experiment.validate_experiment_name("../exp025_elsewhere", tmp_path)


def write_record_templates(root: Path) -> None:
    template = root / "templates" / "experiment"
    template.mkdir(parents=True)
    (template / "README.md").write_text("# {{ EXPERIMENT_NAME }}\n\n## 概要\n")
    (template / "requirements.md").write_text(
        "# {{ EXPERIMENT_NAME }} 要件と実装方法\n\n## 実装方法\n\n- TODO\n"
    )
    (template / "SESSION_NOTES.md").write_text(
        "# {{ EXPERIMENT_NAME }} セッションノート\n\n## 現在の作業\n\n- 次: TODO\n"
    )
    (template / "result.md").write_text(
        "# {{ EXPERIMENT_NAME }} 結果\n\n## 記録の参照先\n\n- 設定: config.yaml\n"
    )
    (template / "metrics.json").write_text(
        '{"experiment": "{{ EXPERIMENT_NAME }}", "status": "planned"}\n'
    )


def test_source_experiment_tests_are_excluded_by_default(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(new_experiment, "ROOT", tmp_path)
    source = tmp_path / "experiments" / "exp001_parent"
    (source / "tests").mkdir(parents=True)
    (source / "tests" / "test_parent.py").write_text("def test_parent(): pass\n")
    (source / "config.yaml").write_text("experiment: exp001_parent\n")

    destination = tmp_path / "experiments" / "exp002_child"
    new_experiment.copy_tree(source, destination, force=False, copy_tests=False)

    assert (destination / "config.yaml").exists()
    assert not (destination / "tests").exists()
    for dirname in new_experiment.GENERATED_DIRS:
        generated_dir = destination / dirname
        assert generated_dir.is_dir()
        assert (generated_dir / ".gitkeep").is_file()


def test_source_experiment_tests_can_be_copied_explicitly(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(new_experiment, "ROOT", tmp_path)
    source = tmp_path / "experiments" / "exp001_parent"
    (source / "tests").mkdir(parents=True)
    (source / "tests" / "test_parent.py").write_text("def test_parent(): pass\n")

    destination = tmp_path / "experiments" / "exp002_child"
    new_experiment.copy_tree(source, destination, force=False, copy_tests=True)

    assert (destination / "tests" / "test_parent.py").exists()


def test_main_uses_configured_experiments_directory(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    source = tmp_path / "templates" / "experiment"
    source.mkdir(parents=True)
    (source / "README.md").write_text("# {{ EXPERIMENT_NAME }}\n")
    (tmp_path / "project.yml").write_text("paths:\n  experiments_dir: runs\n")
    monkeypatch.setattr(new_experiment, "ROOT", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "new_experiment.py",
            "--name",
            "exp002_child",
            "--source",
            "templates/experiment",
            "--dry-run",
        ],
    )

    new_experiment.main()

    assert "runs/exp002_child" in capsys.readouterr().out


def test_parent_copy_replaces_identity_and_resets_execution_records(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(new_experiment, "ROOT", tmp_path)
    write_record_templates(tmp_path)
    source = tmp_path / "experiments" / "exp001_parent"
    source.mkdir(parents=True)
    (source / "README.md").write_text("# exp001_parent\n\nPublic LB: 1.23\n")
    (source / "requirements.md").write_text("# exp001_parent contract\ncompleted\n")
    (source / "SESSION_NOTES.md").write_text("# exp001_parent\ncompleted\n")
    (source / "result.md").write_text("# exp001_parent result\ncompleted\n")
    (source / "metrics.json").write_text(
        '{"experiment": "exp001_parent", "status": "completed", "public_lb": 1.23}\n'
    )
    (source / "config.yaml").write_text(
        "experiment:\n"
        "  name: exp001_parent\n"
        "  description: completed parent\n"
        "  route: sequence_model\n"
        "  status: completed\n"
        "lineage:\n"
        "  parent: exp000_base\n"
        "  hypothesis_id: HYP-19000101-93\n"
        "  backlog_candidate: parent_candidate\n"
        "  diff_summary: parent diff\n"
        "model:\n"
        "  name: retained_model\n"
    )
    (source / "settings.py").write_text('EXPERIMENT_NAME = "exp001_parent"\n')
    (source / "exp001_parent_train.ipynb").write_text(
        '{"cells": [{"source": ["# exp001_parent train"]}]}\n'
    )

    destination = tmp_path / "experiments" / "exp002_child"
    new_experiment.copy_tree(source, destination, force=False, copy_tests=False)
    new_experiment.replace_parent_experiment_identity(destination, "exp001_parent", "exp002_child")
    new_experiment.reset_parent_records(destination, "exp002_child", "exp001_parent")

    assert not (destination / "exp001_parent_train.ipynb").exists()
    assert (destination / "exp002_child_train.ipynb").exists()
    assert "exp002_child" in (destination / "settings.py").read_text()
    assert "Public LB" not in (destination / "README.md").read_text()
    assert "exp001_parent" not in (destination / "README.md").read_text()
    assert "exp002_child" in (destination / "requirements.md").read_text()
    assert "completed" not in (destination / "requirements.md").read_text()
    assert "exp001_parent" not in (destination / "result.md").read_text()
    metrics = json.loads((destination / "metrics.json").read_text())
    assert metrics == {"experiment": "exp002_child", "status": "planned"}
    config = yaml.safe_load((destination / "config.yaml").read_text())
    assert config["experiment"]["name"] == "exp002_child"
    assert config["experiment"]["description"] == "TODO"
    assert config["experiment"]["route"] == "sequence_model"
    assert "status" not in config["experiment"]
    assert config["lineage"]["parent"] == "exp001_parent"
    assert config["lineage"]["hypothesis_id"] == "TODO"
    assert config["lineage"]["backlog_candidate"] == "TODO"
    assert config["lineage"]["diff_summary"] == "TODO"
    assert config["model"]["name"] == "retained_model"
    assert "ルート:" not in (destination / "README.md").read_text()
    assert "状態:" not in (destination / "README.md").read_text()
    assert "Route:" not in (destination / "SESSION_NOTES.md").read_text()
    assert "状態:" not in (destination / "SESSION_NOTES.md").read_text()


@pytest.mark.parametrize(
    "relationship", ["same", "source_contains", "destination_contains", "alias"]
)
def test_overlapping_copy_paths_preserve_existing_work(tmp_path, monkeypatch, relationship):
    monkeypatch.setattr(new_experiment, "ROOT", tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    if relationship == "same":
        destination = source
    elif relationship == "source_contains":
        destination = source / "child"
        destination.mkdir()
    elif relationship == "destination_contains":
        destination = tmp_path
    else:
        alias = tmp_path / "alias"
        alias.symlink_to(tmp_path, target_is_directory=True)
        destination = alias / "source"
    source_record = source / "parent.txt"
    source_record.write_text("parent work")
    destination_record = destination / "child.txt"
    destination_record.write_text("existing child work")

    with pytest.raises(ValueError, match="same or contain"):
        new_experiment.copy_tree(source, destination, force=True, copy_tests=False)

    assert source_record.read_text() == "parent work"
    assert destination_record.read_text() == "existing child work"


def test_file_source_cannot_replace_existing_directory(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("not a directory")
    destination = tmp_path / "exp001_existing"
    destination.mkdir()
    record = destination / "notes.txt"
    record.write_text("keep my work")
    with pytest.raises(NotADirectoryError, match="not a directory"):
        new_experiment.copy_tree(source, destination, force=True, copy_tests=False)
    assert record.read_text() == "keep my work"


@pytest.mark.parametrize("failure_stage", ["copy", "initialize", "publish"])
def test_failed_force_copy_preserves_existing_destination(tmp_path, monkeypatch, failure_stage):
    monkeypatch.setattr(new_experiment, "ROOT", tmp_path)
    source = tmp_path / "templates" / "experiment"
    source.mkdir(parents=True)
    (source / "new.txt").write_text("new work")
    destination = tmp_path / "experiments" / "exp001_existing"
    destination.mkdir(parents=True)
    (destination / "old.txt").write_text("existing work")
    initialize = None

    def fail(*args, **kwargs):
        raise OSError("injected failure")

    if failure_stage == "copy":
        monkeypatch.setattr(new_experiment.shutil, "copytree", fail)
    elif failure_stage == "initialize":
        initialize = fail
    else:
        rename = Path.rename

        def fail_publication(path, target):
            if path.name == "experiment":
                fail()
            return rename(path, target)

        monkeypatch.setattr(Path, "rename", fail_publication)

    with pytest.raises(OSError, match="injected failure"):
        new_experiment.copy_tree(
            source, destination, force=True, copy_tests=False, postprocess=initialize
        )

    assert (destination / "old.txt").read_text() == "existing work"
    assert not (destination / "new.txt").exists()
    assert (source / "new.txt").read_text() == "new work"
    assert list(destination.parent.iterdir()) == [destination]


def test_force_copy_replaces_only_destination_after_initializing(tmp_path, monkeypatch):
    monkeypatch.setattr(new_experiment, "ROOT", tmp_path)
    source = tmp_path / "template"
    source.mkdir()
    (source / "README.md").write_text("# {{ EXPERIMENT_NAME }}\n")
    destination = tmp_path / "exp001_existing"
    destination.mkdir()
    (destination / "old.txt").write_text("existing work")
    new_experiment.copy_tree(
        source,
        destination,
        force=True,
        copy_tests=False,
        postprocess=lambda staged: new_experiment.replace_tokens(staged, destination.name),
    )
    assert (destination / "README.md").read_text() == "# exp001_existing\n"
    assert not (destination / "old.txt").exists()
    assert (source / "README.md").read_text() == "# {{ EXPERIMENT_NAME }}\n"


def test_invalid_parent_config_does_not_replace_existing_child(tmp_path, monkeypatch):
    monkeypatch.setattr(new_experiment, "ROOT", tmp_path)
    write_record_templates(tmp_path)
    source = tmp_path / "experiments" / "exp001_parent"
    source.mkdir(parents=True)
    (source / "config.yaml").write_text("experiment: not-a-mapping\n")
    destination = tmp_path / "experiments" / "exp002_child"
    destination.mkdir()
    (destination / "notes.txt").write_text("existing child work")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "new_experiment.py",
            "--name",
            destination.name,
            "--source",
            str(source.relative_to(tmp_path)),
            "--force",
        ],
    )
    with pytest.raises(ValueError, match="experiment must contain a mapping"):
        new_experiment.main()
    assert (destination / "notes.txt").read_text() == "existing child work"
    assert sorted(path.name for path in destination.iterdir()) == ["notes.txt"]


@pytest.mark.parametrize("restore_fails", [False, True])
def test_interrupted_publication_keeps_original_work(tmp_path, monkeypatch, restore_fails):
    monkeypatch.setattr(new_experiment, "ROOT", tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    (source / "new.txt").write_text("new work")
    destination = tmp_path / "experiments" / "exp001_existing"
    destination.mkdir(parents=True)
    (destination / "old.txt").write_text("original work")
    rename = Path.rename

    def interrupt_publication(path, target):
        if path.name == "experiment":
            raise KeyboardInterrupt("interrupted publication")
        if restore_fails and path.name == "previous":
            raise OSError("restoration unavailable")
        return rename(path, target)

    monkeypatch.setattr(Path, "rename", interrupt_publication)
    if restore_fails:
        with pytest.raises(RuntimeError, match="original work is preserved at") as caught:
            new_experiment.copy_tree(source, destination, force=True, copy_tests=False)
        backups = list(destination.parent.glob(f".{destination.name}-backup-*/previous"))
        assert len(backups) == 1
        assert (backups[0] / "old.txt").read_text() == "original work"
        assert str(backups[0]) in str(caught.value)
        assert not destination.exists()
    else:
        with pytest.raises(KeyboardInterrupt, match="interrupted publication"):
            new_experiment.copy_tree(source, destination, force=True, copy_tests=False)
        assert (destination / "old.txt").read_text() == "original work"
        assert list(destination.parent.iterdir()) == [destination]
    assert (source / "new.txt").read_text() == "new work"


def test_publish_and_restore_io_errors_preserve_backup(tmp_path, monkeypatch):
    monkeypatch.setattr(new_experiment, "ROOT", tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    destination = tmp_path / "experiments" / "exp001_existing"
    destination.mkdir(parents=True)
    (destination / "old.txt").write_text("original work")
    rename = Path.rename

    def fail_publication_and_restoration(path, target):
        if path.name in {"experiment", "previous"}:
            raise OSError("device unavailable")
        return rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_publication_and_restoration)
    with pytest.raises(RuntimeError, match="original work is preserved at") as caught:
        new_experiment.copy_tree(source, destination, force=True, copy_tests=False)
    backup = next(destination.parent.glob(f".{destination.name}-backup-*/previous"))
    assert (backup / "old.txt").read_text() == "original work"
    assert str(backup) in str(caught.value)


@pytest.mark.parametrize("concurrent_target", ["empty_directory", "populated_directory", "file"])
def test_no_force_copy_never_replaces_destination_created_during_staging(
    tmp_path, monkeypatch, concurrent_target
):
    monkeypatch.setattr(new_experiment, "ROOT", tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    (source / "new.txt").write_text("copied content")
    destination = tmp_path / "experiments" / "exp001_existing"

    def concurrent_write(staged):
        if concurrent_target == "file":
            destination.write_text("another writer's work")
        else:
            destination.mkdir()
            if concurrent_target == "populated_directory":
                (destination / "notes.txt").write_text("another writer's work")

    with pytest.raises(FileExistsError):
        new_experiment.copy_tree(
            source, destination, force=False, copy_tests=False, postprocess=concurrent_write
        )
    if concurrent_target == "file":
        assert destination.read_text() == "another writer's work"
    elif concurrent_target == "populated_directory":
        assert (destination / "notes.txt").read_text() == "another writer's work"
        assert sorted(path.name for path in destination.iterdir()) == ["notes.txt"]
    else:
        assert list(destination.iterdir()) == []
    assert list(destination.parent.iterdir()) == [destination]


def test_no_force_partial_publication_failure_does_not_delete_other_writers_work(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(new_experiment, "ROOT", tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    (source / "new.txt").write_text("copied content")
    destination = tmp_path / "experiments" / "exp001_new"
    copytree = new_experiment.shutil.copytree

    def fail_after_exclusive_creation(src, dst, *args, **kwargs):
        if Path(dst) == destination:
            Path(dst).mkdir()
            (destination / "notes.txt").write_text("another writer's work")
            raise OSError("copy interrupted after destination creation")
        return copytree(src, dst, *args, **kwargs)

    monkeypatch.setattr(new_experiment.shutil, "copytree", fail_after_exclusive_creation)
    with pytest.raises(OSError, match="copy interrupted"):
        new_experiment.copy_tree(source, destination, force=False, copy_tests=False)
    assert (destination / "notes.txt").read_text() == "another writer's work"
    assert (source / "new.txt").read_text() == "copied content"


@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize("external_source", [False, True])
def test_main_supports_external_configured_destination_and_source_display(
    tmp_path, monkeypatch, capsys, dry_run, external_source
):
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(new_experiment, "ROOT", root)
    source = tmp_path / "external-template" if external_source else root / "templates/experiment"
    source.mkdir(parents=True)
    (source / "README.md").write_text("# {{ EXPERIMENT_NAME }}\n")
    destination = tmp_path / "external-runs" / "exp001_example"
    (root / "project.yml").write_text(
        yaml.safe_dump({"paths": {"experiments_dir": str(destination.parent)}})
    )
    argv = ["new_experiment.py", "--name", destination.name, "--source", str(source)]
    if dry_run:
        argv.append("--dry-run")
    monkeypatch.setattr(sys, "argv", argv)
    new_experiment.main()
    output = capsys.readouterr().out
    assert str(destination) in output
    if dry_run:
        assert not destination.exists()
        expected_source = str(source) if external_source else "templates/experiment"
        assert f"Source: {expected_source}" in output
    else:
        assert (destination / "README.md").read_text() == "# exp001_example\n"
