import subprocess
import sys
from pathlib import Path

import pytest

from scripts import check_markdown_links


def scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    documents: dict[str, str],
    *,
    sources: list[str] | None = None,
) -> list[str]:
    for name, content in documents.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    selected = (
        sources
        if sources is not None
        else [name for name in documents if Path(name).suffix in {".md", ".markdown"}]
    )
    monkeypatch.setattr(check_markdown_links, "ROOT", tmp_path)
    monkeypatch.setattr(
        check_markdown_links, "markdown_files", lambda: [tmp_path / name for name in selected]
    )
    return check_markdown_links.broken_links()


def test_missing_file_and_missing_fragment_are_distinct(tmp_path, monkeypatch):
    errors = scan(
        tmp_path,
        monkeypatch,
        {
            "README.md": "# Overview\n[missing file](missing.md#section)\n"
            "[missing heading](target.md#old-section)\n",
            "target.md": "# New section\n",
        },
    )

    assert errors == [
        "README.md:2: missing local link target missing.md",
        "README.md:3: missing Markdown fragment #old-section in target.md",
    ]


def test_japanese_encoded_paths_and_fragments_and_link_titles(tmp_path, monkeypatch):
    errors = scan(
        tmp_path,
        monkeypatch,
        {
            "README.md": '[結果](<資料 メモ.md#確認結果> "説明")\n'
            "[encoded](%E8%B3%87%E6%96%99%20%E3%83%A1%E3%83%A2.md"
            "#%E7%A2%BA%E8%AA%8D%E7%B5%90%E6%9E%9C)\n",
            "資料 メモ.md": "## 確認結果\n",
        },
    )

    assert errors == []


def test_same_document_heading_links_include_inline_formatting(tmp_path, monkeypatch):
    errors = scan(
        tmp_path,
        monkeypatch,
        {
            "README.md": "# **日本語** / `API` & [CV](https://example.com)\n"
            "[section](#日本語--api--cv)\n"
            "[top](#)\n",
        },
    )

    assert errors == []


def test_duplicate_and_setext_headings_use_unique_suffixes(tmp_path, monkeypatch):
    errors = scan(
        tmp_path,
        monkeypatch,
        {
            "README.md": "# Same\n# Same-1\nSame\n----\n# Same\n"
            "[first](#same) [second](#same-1) [third](#same-2) [fourth](#same-3)\n"
            "[missing](#same-4)\n",
        },
    )

    assert errors == ["README.md:7: missing Markdown fragment #same-4 in README.md"]


def test_explicit_html_id_and_named_anchors(tmp_path, monkeypatch):
    errors = scan(
        tmp_path,
        monkeypatch,
        {
            "README.md": '<a id="Custom_ID"></a>\n\n<a name="legacy"></a>\n\n'
            '<h2 id="詳細&amp;条件">説明</h2>\n\n'
            "[custom](#Custom_ID) [legacy](#legacy) [entity](#詳細%26条件)\n",
        },
    )

    assert errors == []


def test_inline_fenced_indented_and_quoted_code_are_not_links_or_anchors(tmp_path, monkeypatch):
    errors = scan(
        tmp_path,
        monkeypatch,
        {
            "README.md": "# Real\n\n`[example](missing.md#missing)`\n\n"
            "````md\n# Fake\n[example](missing.md)\n```\n````\n\n"
            '~~~md\n<a id="fake"></a>\n[example](missing.md)\n~~~\n\n'
            "    [example](missing.md)\n\n"
            "> ```md\n> [example](missing.md)\n> ```\n\n"
            "<!-- [example](missing.md) -->\n\n[broken](#fake)\n",
        },
    )

    assert len(errors) == 1
    assert "missing Markdown fragment #fake" in errors[0]


def test_front_matter_is_not_a_heading_or_link(tmp_path, monkeypatch):
    errors = scan(
        tmp_path,
        monkeypatch,
        {
            "README.md": "---\ntitle: [metadata](missing.md)\n---\n# Actual\n[actual](#actual)\n",
        },
    )

    assert errors == []


def test_reference_links_tables_images_and_multiline_locations(tmp_path, monkeypatch):
    errors = scan(
        tmp_path,
        monkeypatch,
        {
            "README.md": "# Actual\n\nA `multiline\ncode span` [missing](#no-section)\n\n"
            "| Link |\n| --- |\n| [target][ref] |\n\n"
            "![missing image](missing.png)\n\n[ref]: target.md#absent\n",
            "target.md": "# Present\n",
        },
    )

    assert errors == [
        "README.md:4: missing Markdown fragment #no-section in README.md",
        "README.md:8: missing Markdown fragment #absent in target.md",
        "README.md:10: missing local link target missing.png",
    ]


def test_parentheses_in_paths_are_parsed_as_part_of_destination(tmp_path, monkeypatch):
    errors = scan(
        tmp_path,
        monkeypatch,
        {
            "README.md": "[nested](notes(draft).md#section)\n",
            "notes(draft).md": "# Section\n",
        },
    )

    assert errors == []


def test_non_markdown_fragments_and_remote_urls_are_not_guessed(tmp_path, monkeypatch):
    errors = scan(
        tmp_path,
        monkeypatch,
        {
            "README.md": "[pdf](report.pdf#page=9) [notebook](example.ipynb#cell-id)\n"
            "[remote](https://example.com/missing.md#missing)\n"
            "[mail](mailto:name@example.com) [absolute](/external/file.md#missing)\n"
            "[custom](custom:missing.md#missing)\n",
            "report.pdf": "unparsed binary format",
            "example.ipynb": "{}",
        },
    )

    assert errors == []


def test_local_url_queries_do_not_bypass_file_and_heading_validation(tmp_path, monkeypatch):
    errors = scan(
        tmp_path,
        monkeypatch,
        {
            "README.md": "[valid](target.md?plain=1#present)\n"
            "[file](missing.md?raw=1)\n[heading](target.md?plain=1#absent)\n"
            "[remote](https://example.com/missing.md?plain=1#absent)\n",
            "target.md": "# Present\n",
        },
    )

    assert errors == [
        "README.md:2: missing local link target missing.md",
        "README.md:3: missing Markdown fragment #absent in target.md",
    ]


def test_archived_destination_headings_are_checked_without_scanning_its_links(
    tmp_path, monkeypatch
):
    errors = scan(
        tmp_path,
        monkeypatch,
        {
            "README.md": "[saved source](archive/source.md#original)\n",
            "archive/source.md": "# Original\n[original relative link](absent.md)\n",
        },
        sources=["README.md"],
    )

    assert errors == []


def git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def artifact_repository(tmp_path: Path, experiments: str = "experiments") -> Path:
    git(tmp_path, "init", "--quiet")
    (tmp_path / "project.yml").write_text(f"paths:\n  experiments_dir: {experiments}\n")
    (tmp_path / ".gitignore").write_text(f"/{experiments}/*/artifacts/*\n*.npy\n")
    experiment = tmp_path / experiments / "exp001_example"
    experiment.mkdir(parents=True)
    (experiment / "config.yaml").write_text("experiment: exp001_example\n")
    git(tmp_path, "add", "project.yml", ".gitignore", str(experiment.relative_to(tmp_path)))
    return experiment


@pytest.mark.parametrize("experiments", ["experiments", "runs/nested"])
def test_missing_ignored_artifacts_are_reported_separately(tmp_path, monkeypatch, experiments):
    artifact_repository(tmp_path, experiments)
    relative = f"{experiments}/exp001_example/artifacts"
    errors = scan(
        tmp_path,
        monkeypatch,
        {
            "README.md": f"[receipt]({relative}/run/receipt.json)\n"
            f"[directory]({relative}/run/)\n[readout]({relative}/readout.md#summary)\n",
        },
    )

    assert errors == []
    _, unverified = check_markdown_links.check_links()
    assert unverified == [
        f"README.md:1: unavailable artifact {relative}/run/receipt.json",
        f"README.md:2: unavailable artifact {relative}/run/",
        f"README.md:3: unavailable artifact {relative}/readout.md",
    ]


@pytest.mark.parametrize("tracked_target", ["receipt.json", "saved/receipt.json", ".gitkeep"])
def test_deleted_tracked_artifacts_and_their_directories_are_errors(
    tmp_path, monkeypatch, tracked_target
):
    experiment = artifact_repository(tmp_path)
    target = experiment / "artifacts" / tracked_target
    target.parent.mkdir(parents=True)
    target.write_text("tracked evidence")
    git(tmp_path, "add", "--force", str(target.relative_to(tmp_path)))
    target.unlink()
    target.parent.rmdir()
    target_link = target.relative_to(tmp_path).as_posix()
    parent_link = target.parent.relative_to(tmp_path).as_posix() + "/"

    errors = scan(
        tmp_path,
        monkeypatch,
        {"README.md": f"[file]({target_link})\n[directory]({parent_link})\n"},
    )

    assert len(errors) == 2
    assert all("missing local link target" in error for error in errors)
    assert check_markdown_links.check_links()[1] == []


def test_ordinary_missing_paths_and_unknown_experiments_are_errors(tmp_path, monkeypatch):
    artifact_repository(tmp_path)
    errors = scan(
        tmp_path,
        monkeypatch,
        {
            "README.md": "[document](docs/missing.md)\n"
            "[source](experiments/exp001_example/source.py)\n"
            "[ignored format](experiments/exp001_example/source.npy)\n"
            "[unknown experiment](experiments/exp999_missing/artifacts/receipt.json)\n"
            "[unknown location](experiments/exp001_example/artifacts2/receipt.npy)\n"
            "[escaped](experiments/exp001_example/artifacts/../source.npy)\n",
        },
    )

    assert len(errors) == 6
    assert check_markdown_links.check_links()[1] == []


def test_changed_configuration_does_not_exempt_old_experiment_location(tmp_path, monkeypatch):
    artifact_repository(tmp_path, "runs")
    old_experiment = tmp_path / "experiments/exp001_example"
    old_experiment.mkdir(parents=True)
    (old_experiment / "config.yaml").write_text("experiment: exp001_example\n")
    with (tmp_path / ".gitignore").open("a") as handle:
        handle.write("/experiments/*/artifacts/*\n")
    errors = scan(
        tmp_path,
        monkeypatch,
        {"README.md": "[old](experiments/exp001_example/artifacts/receipt.json)\n"},
    )

    assert len(errors) == 1
    assert check_markdown_links.check_links()[1] == []


def test_artifact_without_git_ignore_rule_is_still_an_error(tmp_path, monkeypatch):
    artifact_repository(tmp_path)
    (tmp_path / ".gitignore").write_text("")
    errors = scan(
        tmp_path,
        monkeypatch,
        {"README.md": "[receipt](experiments/exp001_example/artifacts/receipt.json)\n"},
    )

    assert len(errors) == 1
    assert check_markdown_links.check_links()[1] == []


def test_absolute_configured_experiment_path_inside_repository_is_supported(tmp_path, monkeypatch):
    artifact_repository(tmp_path, "runs")
    (tmp_path / "project.yml").write_text(f"paths:\n  experiments_dir: {tmp_path / 'runs'}\n")
    errors = scan(
        tmp_path,
        monkeypatch,
        {"README.md": "[receipt](runs/exp001_example/artifacts/receipt.json)\n"},
    )

    assert errors == []
    assert len(check_markdown_links.check_links()[1]) == 1


@pytest.mark.parametrize("target_name", ["source.py", "requirements.md"])
def test_deleted_tracked_source_and_document_are_errors(tmp_path, monkeypatch, target_name):
    experiment = artifact_repository(tmp_path)
    target = experiment / target_name
    target.write_text("tracked file")
    git(tmp_path, "add", str(target.relative_to(tmp_path)))
    target.unlink()
    errors = scan(
        tmp_path,
        monkeypatch,
        {"README.md": f"[required]({target.relative_to(tmp_path).as_posix()})\n"},
    )

    assert len(errors) == 1
    assert check_markdown_links.check_links()[1] == []


def test_present_artifact_fragment_is_checked(tmp_path, monkeypatch):
    artifact_repository(tmp_path)
    errors = scan(
        tmp_path,
        monkeypatch,
        {
            "README.md": "[readout](experiments/exp001_example/artifacts/readout.md#absent)\n",
            "experiments/exp001_example/artifacts/readout.md": "# Actual\n",
        },
        sources=["README.md"],
    )

    assert len(errors) == 1
    assert "missing Markdown fragment #absent" in errors[0]


@pytest.mark.parametrize("symlink_destination", ["outside", "artifacts"])
def test_symlink_cannot_hide_missing_tracked_or_external_target(
    tmp_path, monkeypatch, symlink_destination
):
    experiment = artifact_repository(tmp_path)
    destination = experiment / symlink_destination
    destination.mkdir()
    target = experiment / "artifacts"
    if symlink_destination == "outside":
        target.symlink_to(destination, target_is_directory=True)
        linked = target / "receipt.json"
    else:
        linked = target / "tracked.json"
        linked.symlink_to(target / "absent.json")
    git(
        tmp_path,
        "add",
        "--force",
        str((target if target.is_symlink() else linked).relative_to(tmp_path)),
    )
    errors = scan(
        tmp_path,
        monkeypatch,
        {"README.md": f"[evidence]({linked.relative_to(tmp_path).as_posix()})\n"},
    )

    assert len(errors) == 1
    assert check_markdown_links.check_links()[1] == []


def test_git_failure_does_not_hide_missing_artifacts(tmp_path, monkeypatch):
    artifact_repository(tmp_path)

    def failed_git(*args, **kwargs):
        raise subprocess.CalledProcessError(128, "git")

    monkeypatch.setattr(check_markdown_links.subprocess, "check_output", failed_git)
    errors = scan(
        tmp_path,
        monkeypatch,
        {"README.md": "[receipt](experiments/exp001_example/artifacts/receipt.json)\n"},
    )

    assert len(errors) == 1


@pytest.mark.parametrize("strict", [False, True])
def test_cli_reports_unverified_artifacts_and_can_require_them(
    tmp_path, monkeypatch, capsys, strict
):
    artifact_repository(tmp_path)
    scan(
        tmp_path,
        monkeypatch,
        {"README.md": "[receipt](experiments/exp001_example/artifacts/receipt.json)\n"},
    )
    monkeypatch.setattr(
        sys, "argv", ["check_markdown_links.py", *(["--require-generated"] if strict else [])]
    )

    if strict:
        with pytest.raises(SystemExit, match="Generated evidence is required"):
            check_markdown_links.main()
    else:
        check_markdown_links.main()
    output = capsys.readouterr().out
    assert "Uncollected generated evidence (1 links; targets not verified)" in output
    assert "README.md:1: unavailable artifact" in output
    assert ("Markdown local links passed" in output) is not strict
    assert ("; 1 generated links unverified" in output) is not strict
