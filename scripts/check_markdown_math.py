"""Check authored Markdown math; optionally verify GitHub's returned TeX.

The default is an offline, read-only check. Notebook code cells and outputs are
never inspected. GitHub's Markdown API is not a notebook or browser renderer.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path

try:
    from .document_scope import maintained_documents
except ImportError:
    from document_scope import maintained_documents

ROOT = Path(__file__).resolve().parents[1]
EXTENSIONS = {".md", ".markdown", ".mdx", ".ipynb"}
FENCE = re.compile(r"^( {0,3})(`{3,}|~{3,})(.*)$")
QUOTE = re.compile(r"^(?: {0,3}>[ \t]?)+")


@dataclass(frozen=True)
class Formula:
    kind: str
    text: str
    line: int
    offset: int


@dataclass
class Analysis:
    formulas: list[Formula] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def blank(text: str) -> str:
    return "".join("\n" if char == "\n" else " " for char in text)


def escaped(text: str, index: int) -> bool:
    count = 0
    while index > 0 and text[index - 1] == "\\":
        count += 1
        index -= 1
    return count % 2 == 1


def mask_comments(line: str, active: bool) -> tuple[str, bool]:
    """Ignore HTML comments without interpreting comment markers inside code spans."""
    chars = list(line)
    index = 0
    while index < len(line):
        if active:
            end = line.find("-->", index)
            stop = len(line) if end < 0 else end + 3
            chars[index:stop] = blank(line[index:stop])
            active = end < 0
            index = stop
            continue
        token = re.search(r"`+|<!--", line[index:])
        if token is None:
            break
        index += token.start()
        if token[0] == "<!--":
            active = True
            continue
        closing = re.search(
            r"(?<!`)" + re.escape(token[0]) + r"(?!`)", line[index + len(token[0]) :]
        )
        if closing is None:
            break
        index += len(token[0]) + closing.end()
    return "".join(chars), active


def analyze(source: str, *, notebook: bool = False) -> Analysis:
    """Extract math outside code/examples and diagnose known GitHub pitfalls."""
    result = Analysis()
    masked = list(source)
    lines = source.splitlines(keepends=True)
    offset = 0
    fence = None
    content: list[str] = []
    comment = False
    front_matter = bool(lines and lines[0].strip() == "---")
    for line_number, line in enumerate(lines, 1):
        visible = line
        if not fence and not front_matter:
            visible, comment = mask_comments(line, comment)
            masked[offset : offset + len(line)] = visible
        raw = visible.rstrip("\r\n")
        quote = QUOTE.match(raw)
        depth = quote[0].count(">") if quote else 0
        if fence:
            logical = raw
            for _ in range(fence[5]):
                logical = re.sub(r"^ {0,3}>[ \t]?", "", logical, count=1)
        else:
            logical = QUOTE.sub("", raw)
        if front_matter:
            masked[offset : offset + len(line)] = blank(line)
            if line_number > 1 and logical in {"---", "..."}:
                front_matter = False
        elif fence:
            marker, language, start, start_offset, indent, _ = fence
            closing = re.fullmatch(
                r" {0,3}" + re.escape(marker[0]) + "{" + str(len(marker)) + r",}\s*", logical
            )
            if closing:
                if language == "math":
                    result.formulas.append(
                        Formula("display", "\n".join(content), start, start_offset)
                    )
                fence = None
            else:
                remove = min(indent, len(logical) - len(logical.lstrip(" ")))
                content.append(logical[remove:])
            masked[offset : offset + len(line)] = blank(line)
        elif match := FENCE.match(logical):
            language = match[3].strip()
            fence = (match[2], language, line_number, offset, len(match[1]), depth)
            content = []
            masked[offset : offset + len(line)] = blank(line)
        elif logical.startswith(("    ", "\t")):
            masked[offset : offset + len(line)] = blank(line)
        offset += len(line)
    if fence and fence[1] == "math":
        result.errors.append(f"line {fence[2]}: unclosed math code block")
    text = "".join(masked)
    index = 0
    while index < len(text):
        line_number = text.count("\n", 0, index) + 1
        if text[index] == "`" and not escaped(text, index):
            marker = re.match(r"`+", text[index:])[0]
            closing = re.search(
                r"(?<!`)" + re.escape(marker) + r"(?!`)", text[index + len(marker) :]
            )
            index += len(marker) + (closing.end() if closing else 0)
            continue
        if text[index : index + 2] in {r"\(", r"\["} and not escaped(text, index):
            opening = text[index : index + 2]
            closing = r"\)" if opening == r"\(" else r"\]"
            end = text.find(closing, index + 2)
            if not notebook:
                result.errors.append(
                    f"line {line_number}: use dollar inline math or "
                    f"a math code block instead of {opening}"
                )
            if end < 0:
                result.errors.append(f"line {line_number}: unclosed {opening}")
                index += 2
                continue
            kind = "inline" if opening == r"\(" else "display"
            result.formulas.append(Formula(kind, text[index + 2 : end], line_number, index))
            index = end + 2
            continue
        if text[index] != "$" or escaped(text, index):
            index += 1
            continue
        if literal := re.match(r"\${3,}", text[index:]):
            index += len(literal[0])
            continue
        delimiter = "$$" if text.startswith("$$", index) else "$"
        start = index + len(delimiter)
        end = text.find(delimiter, start)
        while end >= 0 and escaped(text, end):
            end = text.find(delimiter, end + len(delimiter))
        if end < 0 or (delimiter == "$" and "\n" in text[start:end]):
            if delimiter == "$$":
                result.errors.append(f"line {line_number}: unclosed display math")
            # A lone dollar can be currency or a shell variable in prose.
            index = start
            continue
        stop = end + len(delimiter)
        if delimiter == "$" and stop < len(text) and text[stop].isdigit():
            index = start
            continue
        body = text[start:end]
        if delimiter == "$" and body and body[0].isdigit() and body[-1].isspace():
            # A price followed by prose can precede a later, unrelated formula.
            index = start
            continue
        kind = "display" if delimiter == "$$" else "inline"
        result.formulas.append(Formula(kind, body, line_number, index))
        if not notebook:
            if kind == "display":
                result.errors.append(
                    f"line {line_number}: use a math code block for display math in Markdown"
                )
            else:
                # GitHub also supports dollar/backtick delimiters; preserve them.
                if body.startswith("`") and body.endswith("`"):
                    result.formulas[-1] = Formula(kind, body[1:-1], line_number, index)
                else:
                    if not body or body != body.strip():
                        result.errors.append(
                            f"line {line_number}: inline delimiters must touch the formula"
                        )
                    before = text[index - 1] if index else " "
                    after = text[stop] if stop < len(text) else " "
                    if (not before.isspace() and before not in "([{") or (
                        not after.isspace() and after not in ".,;:!?)]}"
                    ):
                        result.errors.append(
                            f"line {line_number}: separate inline math "
                            "from adjacent prose with spaces"
                        )
                    if re.search(r"\\[,_{}]|[<>]", body):
                        result.errors.append(
                            f"line {line_number}: inline TeX may be altered by Markdown; "
                            "use TeX commands or display math"
                        )
        index = stop
    result.formulas.sort(key=lambda formula: formula.offset)
    return result


class GitHubMath(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.formulas: list[tuple[str, str]] = []
        self.kind: str | None = None
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "math-renderer":
            classes = dict(attrs).get("class") or ""
            self.kind = "display" if "js-display-math" in classes else "inline"
            self.parts = []

    def handle_data(self, data: str) -> None:
        if self.kind:
            self.parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "math-renderer" and self.kind:
            text = "".join(self.parts).strip()
            delimiter = "$$" if self.kind == "display" else "$"
            if text.startswith(delimiter) and text.endswith(delimiter):
                text = text[len(delimiter) : -len(delimiter)]
            self.formulas.append((self.kind, text.strip()))
            self.kind = None


def verify_github(source: str, analysis: Analysis, context: str | None = None) -> list[str]:
    payload = {"text": source, "mode": "gfm"}
    if context:
        payload["context"] = context
    try:
        response = subprocess.run(
            ["gh", "api", "markdown", "--method", "POST", "--input", "-"],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            timeout=45,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return [f"GitHub conversion unavailable: {error}"]
    if response.returncode:
        return [f"GitHub conversion failed: {response.stderr.strip()[:1000]}"]
    parsed = GitHubMath()
    parsed.feed(response.stdout)
    expected = [(formula.kind, formula.text.strip()) for formula in analysis.formulas]
    if len(parsed.formulas) != len(expected):
        return [f"GitHub recognized {len(parsed.formulas)} of {len(expected)} expected formulas"]
    errors = []
    for formula, actual, wanted in zip(analysis.formulas, parsed.formulas, expected, strict=True):
        if actual != wanted:
            errors.append(f"line {formula.line}: GitHub changed the {formula.kind} TeX content")
    return errors


def document_sources(path: Path) -> list[tuple[str, str]]:
    if path.suffix == ".ipynb":
        notebook = json.loads(path.read_text())
        sources = []
        for index, cell in enumerate(notebook["cells"], 1):
            if cell["cell_type"] == "markdown":
                source = cell["source"]
                sources.append(
                    (f"{path}:cell {index}", source if isinstance(source, str) else "".join(source))
                )
        return sources
    return [(str(path), path.read_text())]


def default_paths() -> list[Path]:
    return maintained_documents(ROOT, EXTENSIONS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Files to check; defaults to repository Markdown and notebooks.",
    )
    parser.add_argument(
        "--github",
        action="store_true",
        help="Send explicitly selected Markdown to gh api markdown and compare TeX. "
        "Requires gh and network access.",
    )
    parser.add_argument("--context", help="Optional OWNER/REPO context for GitHub conversion.")
    args = parser.parse_args(argv)
    if args.github and not args.paths:
        parser.error("--github requires explicit Markdown file paths")
    paths = args.paths or default_paths()
    for path in paths:
        if not path.is_file() or path.suffix not in EXTENSIONS:
            parser.error(f"not a supported document: {path}")
        if args.github and path.suffix == ".ipynb":
            parser.error(
                "--github checks Markdown, not notebook rendering; "
                "check the notebook locally and inspect its preview"
            )
    errors = []
    total = 0
    notebook_count = 0
    for path in paths:
        is_notebook = path.suffix == ".ipynb"
        notebook_count += is_notebook
        try:
            sources = document_sources(path)
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append(f"{path}: invalid document: {error}")
            continue
        for label, source in sources:
            analysis = analyze(source, notebook=is_notebook)
            total += len(analysis.formulas)
            findings = analysis.errors
            if args.github and not findings:
                findings = verify_github(source, analysis, args.context)
            errors.extend(f"{label}: {error}" for error in findings)
    if errors:
        print("\n".join(errors))
        return 1
    if args.github:
        print(
            f"GitHub conversion passed ({len(paths)} files, {total} formulas; "
            "TeX content preserved). Browser display not checked."
        )
    else:
        print(
            f"Local math checks passed ({len(paths)} files, {total} formulas, "
            f"{notebook_count} notebooks; Markdown cells only). "
            "GitHub conversion and visual display not checked."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
