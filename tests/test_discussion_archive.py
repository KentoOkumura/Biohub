from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from kagglesdk.discussions.types.discussions_api_service import (
    ApiDiscussionComment,
    ApiDiscussionTopic,
)

ROOT = Path(__file__).resolve().parents[1]
CONVERTER_PATH = ROOT / ".agents/skills/kaggle-discussion-archive/scripts/html_to_discussion_md.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(ROOT / "scripts"))
    return module


converter = load_module("discussion_converter", CONVERTER_PATH)
archive = load_module("discussion_archive", ROOT / "scripts/archive_kaggle_discussions.py")

MARKDOWN = (
    "# Code example\n\n```python\nif x <y and y>0:\n    print(x)\n```\n\n"
    "Source: <https://example.com/method>\n\n"
    "[method](https://example.com/method)\n"
)
COMPETITION_SOURCE = "https://www.kaggle.com/competitions/competition/discussion/12"


def topic(content: str | None = MARKDOWN, topic_id: int = 12):
    return ApiDiscussionTopic.from_dict(
        {"id": topic_id, "title": "Title", "content": content, "authorName": "author"}
    )


def comment(comment_id: int, content: str, replies: list | None = None):
    return ApiDiscussionComment.from_dict(
        {"id": comment_id, "content": content, "replies": replies or []}
    )


class FakeApi:
    def __init__(self, pages):
        self.pages = iter(pages)
        self.calls = []

    def forums_topic_show(self, topic_id, *, page_size, page_token):
        self.calls.append((topic_id, page_size, page_token))
        response = next(self.pages)
        if isinstance(response, Exception):
            raise response
        return response


@pytest.mark.parametrize("source", [MARKDOWN, "    indented code\n", "<div>literal</div>\n"])
def test_text_mode_preserves_source_without_guessing_html(source):
    assert converter.convert(source) == source
    assert converter.convert(source, input_format="text") == source


def test_explicit_html_mode_converts_links_and_code():
    html = '<p>See <a href="https://example.com">method</a>.</p><pre>if x &lt; 2:\n  x += 1</pre>'
    converted = converter.convert(html, input_format="html")
    assert "[method](https://example.com)" in converted
    assert "```text\nif x < 2:\n  x += 1\n```" in converted


def test_converter_cli_default_text_and_explicit_html(tmp_path):
    for input_format, source, expected in (
        (None, MARKDOWN, MARKDOWN),
        ("html", '<a href="https://example.com">method</a>', "[method](https://example.com)"),
    ):
        args = [sys.executable, str(CONVERTER_PATH), "--output-dir", str(tmp_path)]
        if input_format:
            args.extend(["--input-format", input_format])
        subprocess.run(args, input=source, text=True, check=True, capture_output=True)
        assert expected in (tmp_path / "kaggle-discussion.md").read_text()


@pytest.mark.parametrize("authenticated", [False, True])
def test_raw_client_reuses_cli_authentication(monkeypatch, authenticated):
    calls = []
    client = SimpleNamespace(_authenticated=authenticated)

    def authenticate():
        calls.append("same CLI authenticate method")
        client._authenticated = True

    client.authenticate = authenticate
    monkeypatch.setitem(sys.modules, "kaggle", SimpleNamespace(api=client))
    assert archive.discussion_api() is client
    assert calls == ([] if authenticated else ["same CLI authenticate method"])


def test_all_pages_preserve_topic_long_comments_and_replies(tmp_path):
    long_content = "Long comment: " + "unchanged " * 100 + MARKDOWN
    first = comment(1, long_content, [{"id": 2, "content": "First reply"}])
    repeated = comment(1, long_content, [{"id": 3, "content": "Later reply"}])
    api = FakeApi(
        [
            (topic(), [first], "page-two"),
            (topic(), [repeated, comment(4, "<pre>if x &lt; 2:\n  print(x)</pre>")], ""),
        ]
    )
    result = archive.archive_topic(
        "competition", {"id": "12", "title": "Title"}, tmp_path, CONVERTER_PATH, False, api=api
    )
    assert "archived 12" in result
    assert api.calls == [(12, 200, None), (12, 200, "page-two")]
    text = (tmp_path / "competition-12-title.md").read_text()
    assert MARKDOWN in text
    assert long_content in text
    assert "First reply" in text and "Later reply" in text
    assert "- Reply to: 1" in text
    assert text.count("## Comment 1\n") == 1
    assert "<pre>if x &lt; 2:\n  print(x)</pre>" in text
    raw = json.loads((tmp_path / "competition-12-title.json").read_text())
    assert raw["topic"]["content"] == MARKDOWN
    assert raw["comment_pages"][0]["comments"][0]["content"] == long_content
    assert raw["comment_pages"][1]["comments"][0]["replies"][0]["id"] == 3
    assert raw["comment_pages"][-1]["next_page_token"] is None


@pytest.mark.parametrize(
    "pages, error",
    [
        ([(None, [], "")], "missing or mismatched"),
        ([(topic(topic_id=13), [], "")], "missing or mismatched"),
        ([(topic(content=None), [], "")], "does not contain the topic body"),
        ([(topic(), [], "next"), (topic(), [], "next")], "repeated comment page token"),
        ([(topic(), [], "next"), RuntimeError("fetch failed")], "fetch failed"),
    ],
)
def test_failed_fetch_preserves_existing_archive_even_with_force(tmp_path, pages, error):
    output = tmp_path / "competition-12-title.md"
    snapshot = output.with_suffix(".json")
    output.write_text("existing markdown")
    saved_snapshot = json.dumps({"source": COMPETITION_SOURCE})
    snapshot.write_text(saved_snapshot)
    with pytest.raises((ValueError, RuntimeError), match=error):
        archive.archive_topic(
            "competition",
            {"id": "12", "title": "Title"},
            tmp_path,
            CONVERTER_PATH,
            True,
            api=FakeApi(pages),
        )
    assert output.read_text() == "existing markdown"
    assert snapshot.read_text() == saved_snapshot


def test_failed_converter_preserves_existing_archive(tmp_path, monkeypatch):
    output = tmp_path / "competition-12-title.md"
    output.write_text("existing markdown")
    saved_snapshot = json.dumps({"source": COMPETITION_SOURCE})
    output.with_suffix(".json").write_text(saved_snapshot)

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "converter")

    monkeypatch.setattr(archive.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        archive.archive_topic(
            "competition",
            {"id": "12", "title": "Title"},
            tmp_path,
            CONVERTER_PATH,
            True,
            api=FakeApi([(topic(), [], "")]),
        )
    assert output.read_text() == "existing markdown"
    assert output.with_suffix(".json").read_text() == saved_snapshot
    assert not list(tmp_path.glob(".discussion-*"))


def test_existing_archive_is_skipped_without_fetching(tmp_path):
    (tmp_path / "competition-12-title.md").write_text(
        f"# Title\n\n- source: {COMPETITION_SOURCE}\n\nexisting"
    )
    api = FakeApi([])
    assert archive.archive_topic(
        "competition",
        {"id": "12", "title": "Title"},
        tmp_path,
        CONVERTER_PATH,
        False,
        api=api,
    ).startswith("skip existing")
    assert api.calls == []


def test_forum_source_and_single_topic_do_not_replace_bulk_listing(tmp_path, monkeypatch):
    listing = tmp_path / "general_topics_recent.csv"
    listing.write_text("previous listing")
    monkeypatch.setattr(archive, "discussion_api", lambda: FakeApi([(topic(), [], "")]))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "archive_kaggle_discussions.py",
            "--forum",
            "general",
            "--topic-id",
            "12",
            "--output-dir",
            str(tmp_path),
            "--converter",
            str(CONVERTER_PATH),
        ],
    )
    assert archive.main() == 0
    raw = json.loads((tmp_path / "forum-general-12-title.json").read_text())
    assert raw["source"] == "https://www.kaggle.com/discussions/general/12"
    assert listing.read_text() == "previous listing"


@pytest.mark.parametrize("forum", [False, True])
@pytest.mark.parametrize("force", [False, True])
def test_bulk_then_direct_reuses_archive_despite_title_change(tmp_path, forum, force):
    archive.archive_topic(
        "competition",
        {"id": "12", "title": "Listed title"},
        tmp_path,
        CONVERTER_PATH,
        False,
        forum=forum,
        api=FakeApi([(topic(), [], "")]),
    )
    output = next(tmp_path.glob("*.md"))
    original = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    updated = topic(content="Updated body")
    updated.title = "Changed title"
    api = FakeApi([(updated, [], "")] if force else [])
    result = archive.archive_topic(
        "competition", {"id": "12"}, tmp_path, CONVERTER_PATH, force, forum=forum, api=api
    )
    assert str(output) in result
    assert {path.name for path in tmp_path.iterdir()} == set(original)
    if force:
        assert "# Changed title" in output.read_text()
        assert "Updated body" in output.read_text()
        assert (
            json.loads(output.with_suffix(".json").read_text())["topic"]["title"] == "Changed title"
        )
    else:
        assert result.startswith("skip existing")
        assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == original
        assert api.calls == []


@pytest.mark.parametrize("forum", [False, True])
def test_new_archive_name_matches_for_bulk_and_direct(tmp_path, forum):
    names = []
    for mode, listing in (
        ("bulk", {"id": "12", "title": "Outdated title"}),
        ("direct", {"id": "12"}),
    ):
        output_dir = tmp_path / mode
        archive.archive_topic(
            "competition",
            listing,
            output_dir,
            CONVERTER_PATH,
            False,
            forum=forum,
            api=FakeApi([(topic(), [], "")]),
        )
        names.append(next(output_dir.glob("*.md")).name)
    assert names[0] == names[1]
    assert names[0].endswith("-12-title.md")


@pytest.mark.parametrize("evidence", ["markdown", "json"])
def test_legacy_archive_basename_is_preserved(tmp_path, evidence):
    output = tmp_path / "Saved_Existing_Name.md"
    output.write_text(
        f"# Old title\n\n- source: {COMPETITION_SOURCE}\n\nOld body"
        if evidence == "markdown"
        else "Old body"
    )
    if evidence == "json":
        output.with_suffix(".json").write_text(json.dumps({"source": COMPETITION_SOURCE}))
    archive.archive_topic(
        "competition",
        {"id": "12"},
        tmp_path,
        CONVERTER_PATH,
        True,
        api=FakeApi([(topic(), [], "")]),
    )
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "Saved_Existing_Name.json",
        "Saved_Existing_Name.md",
    ]
    assert MARKDOWN in output.read_text()


@pytest.mark.parametrize("force", [False, True])
def test_duplicate_archives_are_rejected_before_fetch(tmp_path, force):
    (tmp_path / "first.md").write_text(f"# First\n\n- source: {COMPETITION_SOURCE}\n\nbody")
    (tmp_path / "second.json").write_text(json.dumps({"source": COMPETITION_SOURCE}))
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    api = FakeApi([])
    with pytest.raises(ValueError, match="ambiguous existing archives") as caught:
        archive.archive_topic("competition", {"id": "12"}, tmp_path, CONVERTER_PATH, force, api=api)
    assert "first.md" in str(caught.value) and "second.md" in str(caught.value)
    assert api.calls == []
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


def test_same_id_in_other_sources_is_not_reused(tmp_path):
    for source, forum in (("competition", False), ("competition", True), ("other", False)):
        archive.archive_topic(
            source,
            {"id": "12"},
            tmp_path,
            CONVERTER_PATH,
            False,
            forum=forum,
            api=FakeApi([(topic(), [], "")]),
        )
    assert len(list(tmp_path.glob("*.md"))) == 3
    assert len({json.loads(path.read_text())["source"] for path in tmp_path.glob("*.json")}) == 3


def test_conflicting_markdown_and_snapshot_sources_are_rejected(tmp_path):
    output = tmp_path / "archive.md"
    output.write_text(f"# Title\n\n- source: {COMPETITION_SOURCE}\n\nbody")
    output.with_suffix(".json").write_text(json.dumps({"source": COMPETITION_SOURCE + "3"}))
    with pytest.raises(ValueError, match="conflicting Markdown/JSON sources"):
        archive.archive_topic(
            "competition", {"id": "12"}, tmp_path, CONVERTER_PATH, True, api=FakeApi([])
        )


def test_force_does_not_overwrite_unknown_source_at_generated_path(tmp_path):
    output = tmp_path / "competition-12-title.md"
    # A source URL quoted in the body is not the archive's source metadata.
    original = f"# Unrelated memo\n\nA quoted source:\n\n- source: {COMPETITION_SOURCE}\n"
    output.write_text(original)
    with pytest.raises(FileExistsError, match="another or unknown source"):
        archive.archive_topic(
            "competition",
            {"id": "12"},
            tmp_path,
            CONVERTER_PATH,
            True,
            api=FakeApi([(topic(), [], "")]),
        )
    assert output.read_text() == original
