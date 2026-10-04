from __future__ import annotations

import argparse
import os
import re
import subprocess
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
    from .config_utils import ROOT, configured_project_path
    from .document_scope import maintained_documents
except ImportError:  # Direct execution: `uv run python scripts/check_markdown_links.py`
    from config_utils import ROOT, configured_project_path
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
    if parsed.scheme or parsed.netloc:
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


def unavailable_generated_paths(destinations: set[Path]) -> set[Path]:
    """Recognize absent, ignored artifacts without hiding missing tracked files."""
    root = ROOT.resolve()
    experiments = configured_project_path(
        "paths.experiments_dir", "experiments", root=root
    ).resolve()
    candidates: dict[str, Path] = {}
    for destination in destinations:
        resolved = destination.resolve()
        if (
            not resolved.is_relative_to(experiments)
            or not resolved.is_relative_to(root)
            or not destination.is_relative_to(root)
        ):
            continue
        parts = resolved.relative_to(experiments).parts
        if (
            len(parts) < 2
            or parts[1] != "artifacts"
            or not re.fullmatch(r"exp[A-Za-z]?\d+_[a-zA-Z0-9_-]+", parts[0])
            or not (experiments / parts[0] / "config.yaml").is_file()
        ):
            continue
        candidates[destination.relative_to(root).as_posix()] = destination
    if not candidates:
        return set()

    # Include parents so a deleted directory containing tracked files cannot be
    # mistaken for optional evidence. Git failures leave every link as an error.
    try:
        tracked = subprocess.check_output(["git", "ls-files", "--cached", "-z"], cwd=root)
        tracked_files = {Path(name) for name in tracked.decode().split("\0") if name}
        protected: set[str] = set()
        for path in tracked_files:
            protected.add(path.as_posix())
            protected.update(parent.as_posix() for parent in path.parents)
        candidates = {
            name: path
            for name, path in candidates.items()
            if name not in protected
            and not any(parent in tracked_files for parent in Path(name).parents)
        }
        if not candidates:
            return set()
        ignored = subprocess.run(
            ["git", "check-ignore", "--stdin", "-z"],
            cwd=root,
            input="\0".join(candidates) + "\0",
            text=True,
            capture_output=True,
            check=False,
        )
    except (OSError, subprocess.CalledProcessError):
        return set()
    if ignored.returncode not in {0, 1}:
        return set()
    return {candidates[name] for name in ignored.stdout.split("\0") if name in candidates}


def check_links() -> tuple[list[str], list[str]]:
    findings: list[tuple[str, Path | None]] = []
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
                Path(os.path.abspath(path.parent / relative_path))
                if relative_path
                else path.absolute()
            )
            if not destination.exists():
                findings.append(
                    (
                        f"{path.relative_to(ROOT)}:{line}: "
                        f"missing local link target {relative_path}",
                        destination,
                    )
                )
            elif fragment and destination.is_file() and destination.suffix in MARKDOWN_EXTENSIONS:
                if destination not in anchors_by_path:
                    anchors_by_path[destination] = document_anchors(parsed_document(destination))
                if fragment in anchors_by_path[destination]:
                    continue
                findings.append(
                    (
                        f"{path.relative_to(ROOT)}:{line}: missing Markdown fragment "
                        f"#{fragment} in {relative_path or path.name}",
                        None,
                    )
                )
    unavailable = unavailable_generated_paths(
        {destination for _, destination in findings if destination is not None}
    )
    offenders, unverified = [], []
    for message, destination in findings:
        if destination in unavailable:
            unverified.append(message.replace("missing local link target", "unavailable artifact"))
        else:
            offenders.append(message)
    return offenders, unverified


def broken_links() -> list[str]:
    return check_links()[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate maintained Markdown local links.")
    parser.add_argument(
        "--require-generated",
        action="store_true",
        help="Fail also when ignored experiment artifacts have not been collected locally.",
    )
    args = parser.parse_args()
    offenders, unverified = check_links()
    if unverified:
        print(f"Uncollected generated evidence ({len(unverified)} links; targets not verified):")
        print("\n".join(unverified))
    if offenders:
        raise SystemExit("\n".join(offenders))
    if args.require_generated and unverified:
        raise SystemExit("Generated evidence is required but has not been collected locally.")
    qualifier = f"; {len(unverified)} generated links unverified" if unverified else ""
    print(f"Markdown local links passed ({len(markdown_files())} files{qualifier})")


if __name__ == "__main__":
    main()
