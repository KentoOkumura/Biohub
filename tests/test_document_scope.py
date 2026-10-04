from scripts import check_markdown_links, document_scope


def test_archives_are_excluded_but_authored_indexes_and_notes_are_checked(tmp_path, monkeypatch):
    (tmp_path / "project.yml").write_text("paths:\n  docs_dir: knowledge\n")
    names = [
        "knowledge/discussions/biohub-example.md",
        "knowledge/discussions/README.md",
        "knowledge/notebooks/author/example.ipynb",
        "knowledge/notebooks/README.md",
        "studies/biohub_source_hidden_eval_20261002/packet/idea_forge_skill.md",
        "studies/biohub_source_hidden_eval_20261002/README.md",
        "experiments/exp001_example/requirements.md",
    ]
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("[missing](missing.md)\n")
    monkeypatch.setattr(
        document_scope.subprocess, "check_output", lambda *a, **kw: "\0".join(names).encode()
    )
    selected = document_scope.maintained_documents(tmp_path, {".md", ".ipynb"})
    assert {path.relative_to(tmp_path).as_posix() for path in selected} == {
        names[1],
        names[3],
        names[5],
        names[6],
    }
    monkeypatch.setattr(check_markdown_links, "ROOT", tmp_path)
    errors = check_markdown_links.broken_links()
    assert len(errors) == 4
    assert any("requirements.md" in error for error in errors)
