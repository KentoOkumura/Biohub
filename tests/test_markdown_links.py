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
