import json
import subprocess

import pytest

from scripts import check_markdown_math as math_check


def test_math_fence_keeps_setext_operators_and_tex_escapes():
    source = (
        "# Metric\n\n```math\n"
        + r"\mathrm{Score}"
        + "\n=\n"
        + r"a+0.1\,b_{\texttt{x\_y}}"
        + "\n```\n\n値は $x_i$ 。\n"
    )
    result = math_check.analyze(source)
    assert not result.errors
    assert [(f.kind, f.text, f.line) for f in result.formulas] == [
        ("display", "\\mathrm{Score}\n=\na+0.1\\,b_{\\texttt{x\\_y}}", 3),
        ("inline", "x_i", 9),
    ]


def test_examples_and_non_math_code_are_excluded():
    source = r"""---
title: '$example$'
---
````markdown
```math
x = y
```
\(example\)
````
~~~python
regex = r"\(x\)"
~~~
    $$shell$$
`\(x\)` and ``$y$``
<!-- $comment$ -->
Actual $z$.
"""
    result = math_check.analyze(source)
    assert not result.errors
    assert [f.text for f in result.formulas] == ["z"]


def test_indented_quoted_math_fence_and_longer_closing_fence():
    result = math_check.analyze(">   ```math\n>   x+y\n>   ````\n")
    assert not result.errors
    assert [f.text for f in result.formulas] == ["x+y"]


@pytest.mark.parametrize(
    "source",
    [
        "$$\nx\n=\ny\n$$\n",
        "$$x=y$$\n",
        r"値は \(x\) です。",
        r"\[x=y\]",
        "胚、$r$ は領域。",
        "$x$に近い。",
        "$ x $",
        r"$a\,b$",
        r"$\texttt{x\_y}$",
        "$x<y$",
        "```math\nx\n",
        "$$\nx\n",
    ],
)
def test_known_markdown_failures_are_reported(source):
    assert math_check.analyze(source).errors


def test_currency_and_escaped_dollars_are_not_math():
    result = math_check.analyze(r"Prices are $10 and $20, or \$30. Use $x$.")
    assert not result.errors
    assert [f.text for f in result.formulas] == ["x"]


def test_notebook_math_keeps_its_own_delimiters():
    result = math_check.analyze("$$\nx\n=\ny\n$$\n\n" + r"Then \(x\) and $y$.", notebook=True)
    assert not result.errors
    assert [f.kind for f in result.formulas] == ["display", "inline", "inline"]


def test_notebook_reads_only_markdown_and_preserves_file(tmp_path):
    path = tmp_path / "example.ipynb"
    payload = {
        "cells": [
            {"cell_type": "code", "source": r"regex = '\\(x\\)'", "outputs": [{"text": "$wrong$"}]},
            {"cell_type": "raw", "source": "$raw$"},
            {"cell_type": "markdown", "source": ["An expression: ", "$x$.\n"]},
            {"cell_type": "markdown", "source": "$$\ny\n$$\n"},
        ]
    }
    path.write_text(json.dumps(payload))
    original = path.read_bytes()
    sources = math_check.document_sources(path)
    assert [text for _, text in sources] == ["An expression: $x$.\n", "$$\ny\n$$\n"]
    assert "cell 3" in sources[0][0]
    assert math_check.main([str(path)]) == 0
    assert path.read_bytes() == original


def fake_response(monkeypatch, html, *, returncode=0, stderr=""):
    requests = []

    def run(command, **kwargs):
        requests.append((command, kwargs))
        return subprocess.CompletedProcess(command, returncode, html, stderr)

    monkeypatch.setattr(math_check.subprocess, "run", run)
    return requests


def test_github_verifies_count_order_and_original_tex(monkeypatch):
    source = "```math\n" + r"\texttt{x\_y} = a\,b" + "\n```\n\n$x$.\n"
    html = (
        '<math-renderer class="js-display-math">$$'
        + r"\texttt{x\_y} = a\,b"
        + '$$</math-renderer><p><math-renderer class="js-inline-math">$x$</math-renderer>.</p>'
    )
    requests = fake_response(monkeypatch, html)
    assert not math_check.verify_github(source, math_check.analyze(source), "owner/repo")
    command, options = requests[0]
    assert command == ["gh", "api", "markdown", "--method", "POST", "--input", "-"]
    assert json.loads(options["input"]) == {"text": source, "mode": "gfm", "context": "owner/repo"}
    assert options["timeout"] <= 60


def test_github_detects_setext_heading_from_actual_failure(monkeypatch):
    source = "$$\n\\mathrm{Score}\n=\na\n$$\n"
    fake_response(monkeypatch, "<h1>$$<br>\\mathrm{Score}</h1><p>a<br>$$</p>")
    errors = math_check.verify_github(source, math_check.analyze(source))
    assert errors == ["GitHub recognized 0 of 1 expected formulas"]


@pytest.mark.parametrize(
    ("source", "rendered"),
    [
        (r"$a\,b$", "$a,b$"),
        (r"$\texttt{x\_y}$", r"$\texttt{x_y}$"),
        (r"$\text{a b}$", r"$\text{ab}$"),
        ("$x<y$", "$x&amp;lt;y$"),
    ],
)
def test_github_detects_tex_damage_even_when_count_matches(monkeypatch, source, rendered):
    fake_response(
        monkeypatch, '<math-renderer class="js-inline-math">' + rendered + "</math-renderer>"
    )
    assert math_check.verify_github(source, math_check.analyze(source))


def test_github_detects_reordered_formulas(monkeypatch):
    source = "$x$ and $y$."
    fake_response(
        monkeypatch, "<math-renderer>$y$</math-renderer><math-renderer>$x$</math-renderer>"
    )
    assert len(math_check.verify_github(source, math_check.analyze(source))) == 2


def test_api_failure_cannot_pass_as_verified(monkeypatch):
    fake_response(monkeypatch, "", returncode=1, stderr="authentication required")
    assert math_check.verify_github("$x$", math_check.analyze("$x$")) == [
        "GitHub conversion failed: authentication required"
    ]


@pytest.mark.parametrize(
    "error", [FileNotFoundError("gh missing"), subprocess.TimeoutExpired("gh", 45)]
)
def test_missing_cli_and_timeout_are_unverified(monkeypatch, error):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(math_check.subprocess, "run", fail)
    assert "unavailable" in math_check.verify_github("$x$", math_check.analyze("$x$"))[0]


def test_github_requires_explicit_files_and_rejects_notebook(tmp_path):
    with pytest.raises(SystemExit) as error:
        math_check.main(["--github"])
    assert error.value.code == 2
    path = tmp_path / "example.ipynb"
    path.write_text('{"cells": []}')
    with pytest.raises(SystemExit) as error:
        math_check.main(["--github", str(path)])
    assert error.value.code == 2


def test_local_mode_never_calls_github(tmp_path, monkeypatch, capsys):
    def unexpected_call(*args, **kwargs):
        raise AssertionError("Local checks must not call GitHub")

    monkeypatch.setattr(math_check.subprocess, "run", unexpected_call)
    path = tmp_path / "note.md"
    path.write_text("```math\nx=y\n```\n")
    assert math_check.main([str(path)]) == 0
    assert "GitHub conversion and visual display not checked" in capsys.readouterr().out


def test_invalid_notebook_fails(tmp_path):
    path = tmp_path / "broken.ipynb"
    path.write_text("invalid JSON")
    assert math_check.main([str(path)]) == 1


def test_default_paths_respect_git_file_list(tmp_path, monkeypatch):
    monkeypatch.setattr(math_check, "ROOT", tmp_path)
    (tmp_path / "note.md").write_text("$x$")
    (tmp_path / "ignored.md").write_text("$x$")
    (tmp_path / "code.py").write_text("x=1")
    monkeypatch.setattr(
        math_check.subprocess,
        "check_output",
        lambda *args, **kwargs: b"note.md\0code.py\0deleted.md\0note.md\0",
    )
    assert math_check.default_paths() == [tmp_path / "note.md"]


def test_literal_dollar_run_is_not_an_unclosed_formula():
    result = math_check.analyze("Featured competition (often $$$).")
    assert not result.errors
    assert not result.formulas


def test_math_fence_preserves_greater_than_and_comment_text():
    source = "```math\nx\n> y\n\\text{<!-- literal -->}\n```\n"
    result = math_check.analyze(source)
    assert not result.errors
    assert result.formulas[0].text == "x\n> y\n\\text{<!-- literal -->}"


def test_commented_out_math_fence_is_not_a_formula():
    source = "<!--\n```math\nx\n```\n-->\n$y$.\n"
    result = math_check.analyze(source)
    assert not result.errors
    assert [f.text for f in result.formulas] == ["y"]


def test_comment_marker_in_inline_code_does_not_hide_math():
    result = math_check.analyze("The marker `<!--` precedes $x$.")
    assert not result.errors
    assert [f.text for f in result.formulas] == ["x"]


def test_explicit_github_check_still_converts_file_with_no_math(tmp_path, monkeypatch):
    path = tmp_path / "note.md"
    path.write_text("Just text.")
    requests = fake_response(monkeypatch, "<p>Just text.</p>")
    assert math_check.main(["--github", str(path)]) == 0
    assert len(requests) == 1
