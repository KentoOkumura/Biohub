from __future__ import annotations

import unicodedata
from collections.abc import Callable, Iterator
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

from markdown_it import MarkdownIt
from markdown_it.rules_inline import image, link
from markdown_it.rules_inline.state_inline import StateInline
from markdown_it.token import Token

try:
    from .config_utils import ROOT
    from .document_scope import maintained_documents
except ImportError:  # Direct execution: `uv run python scripts/check_markdown_links.py`
    from config_utils import ROOT
    from document_scope import maintained_documents


MARKDOWN_EXTENSIONS = {".md", ".markdown"}


def _with_source_line(
    rule: Callable[[StateInline, bool], bool], token_type: str
) -> Callable[[StateInline, bool], bool]:
    """Preserve link locations, including multiline paragraphs and code spans."""

    def wrapped(state: StateInline, silent: bool) -> bool:
        start = state.pos
        first_token = len(state.tokens)
        matched = rule(state, silent)
        if matched and not silent:
            for token in state.tokens[first_token:]:
                if token.type == token_type:
                    token.meta["source_line"] = state.src.count("\n", 0, start)
                    break
        return matched

    return wrapped


def markdown_parser() -> MarkdownIt:
    parser = MarkdownIt("commonmark").enable("table")
    parser.inline.ruler.at("link", _with_source_line(link, "link_open"))
    parser.inline.ruler.at("image", _with_source_line(image, "image"))
    return parser


def parse_markdown(text: str) -> list[Token]:
    lines = text.splitlines(keepends=True)
    if lines and lines[0].strip() == "---":
        for index, line in enumerate(lines[1:], 1):
            if line.strip() in {"---", "..."}:
                lines[: index + 1] = ["\n"] * (index + 1)
                break
    return markdown_parser().parse("".join(lines))


def markdown_files() -> list[Path]:
    return maintained_documents(ROOT, MARKDOWN_EXTENSIONS)


def normalized_target(raw_target: str) -> tuple[str, str] | None:
    target = raw_target.strip()
    if not target or target.startswith("/"):
        return None
    parsed = urlsplit(target)
    if parsed.scheme or parsed.netloc or parsed.query:
        return None
    path, fragment = unquote(parsed.path), unquote(parsed.fragment)
    if "{{" in path or "}}" in path or any(char in path for char in "*?"):
        return None
    return path, fragment


def document_links(tokens: list[Token]) -> Iterator[tuple[str, int]]:
    for token in tokens:
        if token.type != "inline" or token.map is None:
            continue
        for child in token.children or []:
            attribute = "href" if child.type == "link_open" else "src"
            if child.type not in {"link_open", "image"}:
                continue
            target = child.attrGet(attribute)
            if isinstance(target, str):
                yield target, token.map[0] + child.meta.get("source_line", 0) + 1


class ExplicitAnchors(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.anchors: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if value is not None and (name == "id" or (tag == "a" and name == "name")):
                self.anchors.add(value)


def _heading_text(tokens: list[Token]) -> str:
    pieces = []
    for token in tokens:
        if token.type in {"text", "code_inline"}:
            pieces.append(token.content)
        elif token.type == "image":
            pieces.append(_heading_text(token.children or []))
        elif token.type in {"softbreak", "hardbreak"}:
            pieces.append(" ")
    return "".join(pieces)


def heading_slug(text: str) -> str:
    """GitHub-style heading IDs: retain Unicode words, spaces, hyphens and underscores."""
    return "".join(
        "-" if char == " " else char
        for char in text.lower()
        if char in " _-" or unicodedata.category(char)[0] in {"L", "M", "N"}
    )


def document_anchors(tokens: list[Token]) -> set[str]:
    anchors: set[str] = set()
    generated: set[str] = set()
    explicit = ExplicitAnchors()
    for index, token in enumerate(tokens):
        if token.type == "heading_open" and index + 1 < len(tokens):
            heading = tokens[index + 1]
            slug = heading_slug(_heading_text(heading.children or []))
            candidate = slug
            suffix = 0
            while candidate in generated:
                suffix += 1
                candidate = f"{slug}-{suffix}"
            generated.add(candidate)
            anchors.add(candidate)
        if token.type == "html_block":
            explicit.feed(token.content)
        for child in token.children or []:
            if child.type == "html_inline":
                explicit.feed(child.content)
    explicit.close()
    return anchors | explicit.anchors


def broken_links() -> list[str]:
    offenders: list[str] = []
    parsed_documents: dict[Path, list[Token]] = {}
    anchors_by_path: dict[Path, set[str]] = {}

    def parsed_document(path: Path) -> list[Token]:
        if path not in parsed_documents:
            parsed_documents[path] = parse_markdown(path.read_text(errors="replace"))
        return parsed_documents[path]

    for path in markdown_files():
        for raw_target, line in document_links(parsed_document(path)):
            target = normalized_target(raw_target)
            if target is None:
                continue
            relative_path, fragment = target
            destination = (
                (path.parent / relative_path).resolve() if relative_path else path.resolve()
            )
            if not destination.exists():
                offenders.append(
                    f"{path.relative_to(ROOT)}:{line}: missing local link target {relative_path}"
                )
            elif fragment and destination.is_file() and destination.suffix in MARKDOWN_EXTENSIONS:
                if destination not in anchors_by_path:
                    anchors_by_path[destination] = document_anchors(parsed_document(destination))
                if fragment in anchors_by_path[destination]:
                    continue
                offenders.append(
                    f"{path.relative_to(ROOT)}:{line}: missing Markdown fragment "
                    f"#{fragment} in {relative_path or path.name}"
                )
    return offenders


def main() -> None:
    offenders = broken_links()
    if offenders:
        raise SystemExit("\n".join(offenders))
    print(f"Markdown local links passed ({len(markdown_files())} files)")


if __name__ == "__main__":
    main()
