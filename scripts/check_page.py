#!/usr/bin/env python3
"""Run a page's Python as a reader would, and check that it prints what the page shows.

A user page carries its code in fenced ``python`` blocks, and the lines a
statement prints sit directly under it as comment lines: ``#`` and two spaces,
then the printed line, or ``#`` alone for an empty one.  A remark about an
output is prose, or a comment with one space above the statement::

    python scripts/check_page.py [--update] [--register] [--links] [page.md ...]

Each page runs in a fresh process, from the repository root, with the
checkout's ``src`` first on ``PYTHONPATH``: its ``python`` blocks in order and
in one namespace, each split into its top-level statements, which run one at a
time.  What each statement prints is compared with the output lines under it;
trailing whitespace is ignored and nothing else is.  A block fenced ``python
template`` -- the reader's own data, a file on their disk -- is not run.  A
warning goes to stderr and is not output, and ``sys.exit`` in a page is a
failure: a page shows a refusal by printing it.  The report gives each block's
statements and time, and the page's wall time.

``--update`` rewrites each statement's output lines from what it printed,
keeping every other line as it stands, and says what changed: a page's output
is never typed by hand.  ``--register`` adds the page's count under every rule
of ``tests/test_register.py``, read from that module's own patterns, and any
``§`` anywhere in the page, which a page may not carry.  ``--links`` adds the
check that every relative link resolves to a file in the tree, and every
``#anchor`` to a heading of that file by GitHub's rule.  With no page named,
every user page is checked (:func:`user_pages`).

The exit status is 0 when everything asked for holds, 1 otherwise.
``tests/test_docs.py`` runs every user page through :func:`check` before a
commit, so this file is that test's engine as well as the writer's tool.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import difflib
import html
import importlib.util
import io
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import time
import traceback
import unicodedata
import urllib.parse

ROOT = pathlib.Path(__file__).resolve().parent.parent

#: generous -- a page should run in well under two minutes
TIMEOUT = 600

# ---------------------------------------------------------------------------
# the pages
# ---------------------------------------------------------------------------

#: The folders whose every ``.md``, at any depth, is a page.
PAGE_FOLDERS = ("start", "guide", "background", "reference")


def user_pages(root=ROOT):
    """Every user page, as a path relative to ``root``, read off the tree.

    The root ``README.md``, every ``docs/*.md``, and every
    ``.md`` under the folders of :data:`PAGE_FOLDERS`; ``docs/data`` holds no
    pages.  The list is read off the directories, never
    written out, so a page is checked the day it exists.
    """
    root = pathlib.Path(root)
    docs = root / "docs"
    found = ["README.md"] if (root / "README.md").is_file() else []
    found += sorted(f"docs/{path.name}" for path in docs.glob("*.md"))
    for folder in PAGE_FOLDERS:
        found += sorted(path.relative_to(root).as_posix()
                        for path in (docs / folder).rglob("*.md"))
    return found


def _read(path):
    """A page's text exactly as stored, and the line ending it uses."""
    text = pathlib.Path(path).read_bytes().decode("utf-8")
    return text, "\r\n" if "\r\n" in text else "\n"


def _lines(text):
    """A page's lines, each with its own line ending, as Markdown counts them."""
    return [line for line in re.split(r"(?<=\n)", text) if line]


def _label(path, root=ROOT):
    """A page's name in a report: its path from the root where it has one."""
    path = pathlib.Path(path)
    try:
        return path.resolve().relative_to(pathlib.Path(root).resolve()).as_posix()
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# a page's blocks and statements
# ---------------------------------------------------------------------------

_FENCE = re.compile(r"^(?P<indent>[ \t>]*)(?P<mark>`{3,}|~{3,})(?P<info>.*)$")


def fences(lines):
    """Every fenced block of a page: (opening index, closing index, indent, info).

    A fence closes on the same character, at least as many of it, and nothing
    else on the line; the closing index is ``len(lines)`` for a fence the page
    never closes.
    """
    out = []
    i = 0
    while i < len(lines):
        m = _FENCE.match(lines[i])
        if not m or (m["mark"][0] == "`" and "`" in m["info"]):
            i += 1
            continue
        close = re.compile(rf"^[ \t>]*{re.escape(m['mark'][0])}{{{len(m['mark'])},}}\s*$")
        j = i + 1
        while j < len(lines) and not close.match(lines[j]):
            j += 1
        out.append((i, j, m["indent"], m["info"].strip()))
        i = j + 1
    return out


def _is_output(line):
    """Whether a line of a block is printed output: ``#`` and two spaces, or ``#`` alone."""
    return line.startswith("#  ") or line.rstrip() == "#"


def _shown(line):
    """The printed line one output line shows."""
    return "" if line.rstrip() == "#" else line[3:].rstrip()


def _as_output(printed, indent=""):
    """One printed line, written as the page writes it."""
    return indent + ("#  " + printed if printed else "#")


def _python_blocks(lines):
    """The page's ``python`` blocks, each split into its statements; and what is malformed.

    A statement is a top-level statement of its block, two on one line taken
    together.  Its output lines are the run of them directly under it; an
    output line anywhere else in the gap before the next statement is a stray.
    Indices are into the page's lines, counted from 0; a statement's ``line``
    is the page's line number of its first line.
    """
    blocks, problems = [], []
    for start, close, indent, info in fences(lines):
        words = info.split()
        if not words or words[0] != "python":
            continue
        if words == ["python", "template"]:
            blocks.append({"line": start + 1, "template": True})
            continue
        if words != ["python"]:
            problems.append(f"{start + 1}: a Python block is fenced `python`, or "
                            f"`python template` for the reader's own data, not `{info}`")
            continue
        if close == len(lines):
            problems.append(f"{start + 1}: this block is never closed")
            continue
        if ">" in indent:
            problems.append(f"{start + 1}: a block inside a quotation is not run")
            continue
        width = len(indent)
        body = [line[width:] if not line[:width].strip() else line.lstrip()
                for line in lines[start + 1:close]]
        try:
            tree = ast.parse("\n".join(body))
        except SyntaxError as error:
            problems.append(f"{start + 1 + (error.lineno or 1)}: {error.msg}")
            continue
        spans = []
        for node in tree.body:
            first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", ())])
            if spans and first <= spans[-1][1]:
                spans[-1][1] = max(spans[-1][1], node.end_lineno)
            else:
                spans.append([first, node.end_lineno])
        statements = []
        for k, (first, last) in enumerate(spans):
            stop = spans[k + 1][0] - 1 if k + 1 < len(spans) else len(body)
            gap = range(last, stop)
            outputs = []
            for index in gap:
                if not _is_output(body[index]):
                    break
                outputs.append(index)
            statements.append({
                "line": start + 1 + first,
                "end": start + last,
                "source": "\n".join(body[first - 1:last]),
                "outputs": [start + 1 + index for index in outputs],
                "shown": [_shown(body[index]) for index in outputs],
                "strays": [start + 1 + index for index in gap
                           if _is_output(body[index]) and index not in outputs],
            })
        head = range(spans[0][0] - 1 if spans else len(body))
        blocks.append({
            "line": start + 1,
            "indent": indent,
            "statements": statements,
            "strays": [start + 1 + index for index in head if _is_output(body[index])],
        })
    return blocks, problems


# ---------------------------------------------------------------------------
# running a page
# ---------------------------------------------------------------------------


class PageRun:
    """What one run of one page found, and what it rewrote."""

    def __init__(self, label):
        self.label = label
        #: per block run: (fence line, statements, seconds), or None for one not reached
        self.blocks = []
        self.templates = 0
        self.seconds = 0.0
        #: each a readable paragraph, ``page:line:`` first
        self.problems = []
        #: with ``update``, each statement rewritten, as a readable paragraph
        self.rewritten = []

    @property
    def failed(self):
        return bool(self.problems)

    def report(self):
        """The run, as the command line prints it."""
        out = [self.label]
        statements = 0
        for line, count, seconds in self.blocks:
            if seconds is None:
                out.append(f"  block at line {line}: not run")
                continue
            statements += count
            out.append(f"  block at line {line}: {count} statement{'s' * (count != 1)}, "
                       f"{seconds:.1f} s")
        if self.templates:
            out.append(f"  {self.templates} template block{'s' * (self.templates != 1)}, "
                       f"not run")
        if not self.blocks:
            out.append("  no Python blocks to run")
        else:
            out.append(f"  {len(self.blocks)} block{'s' * (len(self.blocks) != 1)}, "
                       f"{statements} statement{'s' * (statements != 1)}, "
                       f"{self.seconds:.1f} s")
        for paragraph in self.rewritten:
            out.append("  rewritten: " + paragraph.replace("\n", "\n  "))
        for paragraph in self.problems:
            out.append("  FAILED " + paragraph.replace("\n", "\n  "))
        if self.blocks and not self.problems:
            out.append("  every statement prints what the page shows"
                       if not self.rewritten else
                       f"  {len(self.rewritten)} statement"
                       f"{'s' * (len(self.rewritten) != 1)} rewritten")
        return "\n".join(out)


def _diff(shown, printed, context=2):
    """The page's output lines against the printed ones, as a unified diff."""
    return "\n".join(difflib.unified_diff(
        [_as_output(line) for line in shown], [_as_output(line) for line in printed],
        "the page", "printed", lineterm="", n=context))


def check(page, update=False, root=ROOT):
    """Run one page in a fresh process and compare; with ``update``, rewrite what differs.

    Returns a :class:`PageRun`.  A page that is malformed -- a block never
    closed, a fence the runner does not know, a syntax error -- is reported and
    not run.  With ``update`` the page is rewritten only where a statement's
    output lines differ from what it printed, and only when every statement
    ran: a run cut short by an exception rewrites nothing.
    """
    root = pathlib.Path(root)
    path = pathlib.Path(page)
    path = path if path.is_absolute() else root / path
    label = _label(path, root)
    run = PageRun(label)
    text, newline = _read(path)
    raw = _lines(text)
    blocks, malformed = _python_blocks([line.rstrip("\r\n") for line in raw])
    run.problems += [f"{label}:{problem}" for problem in malformed]
    runnable = [block for block in blocks if not block.get("template")]
    run.templates = len(blocks) - len(runnable)
    strays = sorted(index for block in runnable
                    for index in block["strays"] + [i for s in block["statements"]
                                                    for i in s["strays"]])
    record = None
    if runnable and not malformed:
        record, run.seconds, stray_print, failure = _run_page(label, runnable, root)
        if failure:
            run.problems.append(f"{label}: {failure}")
        if record is not None and stray_print.strip():
            run.problems.append(f"{label}: printed past Python's own output, where no "
                                f"statement's lines can show it:\n{stray_print.rstrip()}")
        if record is not None and record["failure"]:
            run.problems.append(record["failure"])
    rewrite = update and record is not None and not record["failure"]
    if not rewrite:
        run.problems += [f"{label}:{index + 1}: an output line that is not directly under "
                         f"a statement" for index in strays]
    ran_blocks = record["blocks"] if record is not None else []

    mismatches = []
    for k, block in enumerate(runnable):
        if k >= len(ran_blocks):
            run.blocks.append((block["line"], len(block["statements"]), None))
            continue
        printed = ran_blocks[k]["printed"]
        run.blocks.append((block["line"], len(printed), ran_blocks[k]["seconds"]))
        if record["failure"] and k == len(ran_blocks) - 1:
            printed = printed[:-1]          # the statement that raised is reported as such
        for statement, out in zip(block["statements"], printed):
            got = [line.rstrip() for line in out.splitlines()]
            if got != statement["shown"]:
                mismatches.append((statement, block["indent"], got))

    deleted, inserted = set(), {}
    for statement, indent, got in mismatches:
        first = statement["source"].splitlines()[0]
        paragraph = f"{label}:{statement['line']}: {first}\n" + _diff(statement["shown"], got)
        if not rewrite:
            run.problems.append(paragraph)
            continue
        run.rewritten.append(paragraph)
        deleted.update(statement["outputs"])
        inserted[statement["end"]] = [_as_output(line, indent) + newline for line in got]
    if rewrite and (inserted or strays):
        run.rewritten += [f"{label}:{index + 1}: a stray output line removed"
                          for index in strays]
        deleted.update(strays)
        out = []
        for index, line in enumerate(raw):
            if index not in deleted:
                out.append(line)
            out += inserted.get(index, [])
        path.write_bytes("".join(out).encode("utf-8"))
    return run


def _run_page(label, blocks, root):
    """Run the blocks in a fresh process at the root.

    Returns what each statement printed (``None`` if the process left no
    record), the wall time, anything printed past Python's own ``sys.stdout``,
    and why the process itself failed, if it did.
    """
    with tempfile.TemporaryDirectory(prefix="tqc_page_") as scratch:
        spec = pathlib.Path(scratch) / "page.json"
        record = pathlib.Path(scratch) / "record.json"
        spec.write_text(json.dumps({
            "page": label,
            "record": str(record),
            "blocks": [[{"line": s["line"], "source": s["source"]} for s in block["statements"]]
                       for block in blocks],
        }, ensure_ascii=False), encoding="utf-8")
        env = dict(os.environ)
        env.update(PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")
        env["PYTHONPATH"] = os.pathsep.join(
            [str(root / "src")] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        began = time.perf_counter()
        try:
            proc = subprocess.run(
                [sys.executable, str(pathlib.Path(__file__).resolve()), "--child", str(spec)],
                cwd=root, env=env, stdout=subprocess.PIPE, text=True, encoding="utf-8",
                errors="replace", timeout=TIMEOUT)
        except subprocess.TimeoutExpired:
            return None, time.perf_counter() - began, "", f"did not finish in {TIMEOUT} s"
        seconds = time.perf_counter() - began
        if not record.exists():
            return (None, seconds, proc.stdout,
                    f"the page's process exited {proc.returncode} and left no record")
        return json.loads(record.read_text(encoding="utf-8")), seconds, proc.stdout, None


def _child(spec_path):
    """The page's own process: run its statements one at a time, keep what each prints.

    Each statement is compiled under the page's name with its own line
    numbers, so a traceback points at the page.  The first statement that
    raises, or calls ``sys.exit``, ends the run.
    """
    spec = json.loads(pathlib.Path(spec_path).read_text(encoding="utf-8"))
    if sys.path and pathlib.Path(sys.path[0]).resolve() == pathlib.Path(__file__).resolve().parent:
        del sys.path[0]                 # this script's folder: the page runs from the root
    namespace = {"__name__": "__main__"}
    record = {"blocks": [], "failure": None}
    for statements in spec["blocks"]:
        began = time.perf_counter()
        printed = []
        record["blocks"].append({"printed": printed, "seconds": 0.0})
        for statement in statements:
            code = compile("\n" * (statement["line"] - 1) + statement["source"],
                           spec["page"], "exec")
            buffer = io.StringIO()
            try:
                with contextlib.redirect_stdout(buffer):
                    exec(code, namespace)
            except SystemExit as stop:
                record["failure"] = (f"{spec['page']}:{statement['line']}: calls "
                                     f"sys.exit({stop.code!r}); a page shows a refusal by "
                                     f"printing it")
            except Exception:
                kind, value, trace = sys.exc_info()
                record["failure"] = (f"{spec['page']}:{statement['line']}: raises\n"
                                     + "".join(traceback.format_exception(kind, value,
                                                                          trace.tb_next)))
            printed.append(buffer.getvalue())
            if record["failure"]:
                break
        record["blocks"][-1]["seconds"] = time.perf_counter() - began
        if record["failure"]:
            break
    pathlib.Path(spec["record"]).write_text(json.dumps(record, ensure_ascii=False),
                                            encoding="utf-8")
    return 0


# ---------------------------------------------------------------------------
# the register, one page at a time
# ---------------------------------------------------------------------------

#: The two rules a page may break, and their counts are reported, not judged:
#: "you" in an instruction, and a title that is a question.
REPORTED_ONLY = frozenset({"voice-you", "voice-question"})

_REGISTERS = {}


def _register(root=ROOT):
    """``tests/test_register.py``, loaded by path: the patterns are its own."""
    root = pathlib.Path(root).resolve()
    if root not in _REGISTERS:
        spec = importlib.util.spec_from_file_location(
            "tqc_test_register", root / "tests" / "test_register.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _REGISTERS[root] = module
    return _REGISTERS[root]


def register_hits(page, root=ROOT):
    """The page's hits under every rule of the register, and every section mark in it.

    The rules are ``tests/test_register.py``'s own -- its patterns, its
    carve-outs, its reading of a Markdown document -- applied to this page
    alone, as one of the user pages.  A section mark counts anywhere, fenced
    blocks included: a page links to the heading it means.
    """
    register = _register(root)
    relative = os.path.relpath(pathlib.Path(page).resolve(), pathlib.Path(root).resolve())
    records = register._markdown_lines(relative)
    rules = {rule: register.scan(rule, records, "pages") for rule in register.WORD_RULES}
    marks = [(lineno, line.strip())
             for lineno, line in enumerate(_lines(_read(page)[0]), 1)
             for _ in re.finditer("§+", line)]
    return rules, marks


def register_report(page, root=ROOT):
    """One page's register counts as the command line prints them, and whether they fail it."""
    rules, marks = register_hits(page, root)
    out = [f"{_label(page, root)}: the register"]
    failed = False
    for rule, found in rules.items():
        note = "   (reported, not judged)" if rule in REPORTED_ONLY else ""
        out.append(f"  {rule:<20} {len(found):>3}{note}")
        failed |= bool(found) and rule not in REPORTED_ONLY
        out += [f"    line {hit.lineno}: [{hit.label}] {hit.text[:100]}" for hit in found]
    out.append(f"  {'section marks':<20} {len(marks):>3}")
    out += [f"    line {lineno}: {line[:100]}" for lineno, line in marks]
    return "\n".join(out), failed or bool(marks)


# ---------------------------------------------------------------------------
# links and anchors
# ---------------------------------------------------------------------------

_ATX = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*?))?(?:[ \t]+#+)?[ \t]*$")
_SETEXT = re.compile(r"^ {0,3}(?:=+|-+)[ \t]*$")
_LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])(?:\s|$)")
_HTML_ANCHOR = re.compile(r"<a\s[^>]*\b(?:id|name)\s*=\s*[\"']([^\"']+)[\"']", re.I)
# a code span: a run of backticks, closed by a run of the same length, within
# one paragraph
_CODE_SPAN = re.compile(r"(?<!`)(`+)(?!`)((?:(?!\n[ \t]*\n).)+?)(?<!`)\1(?!`)", re.S)
_INLINE_LINK = re.compile(
    r"!?\[[^\]]*\]\(\s*(<[^>\n]*>|[^\s)]*)(?:\s+(?:\"[^\"]*\"|'[^']*'))?\s*\)")
# a reference definition, ``[label]: target``; a footnote, ``[^1]: text``, is not one
_REFERENCE = re.compile(r"^ {0,3}\[(?!\^)[^\]]+\]:[ \t]*(<[^>\n]*>|\S+)", re.M)
_SCHEME = re.compile(r"^(?:[A-Za-z][A-Za-z0-9+.-]*:|//)")


def _prose(page):
    """A page's lines, endings dropped, with its fenced blocks blanked."""
    out = [line.rstrip("\r\n") for line in _lines(_read(page)[0])]
    for start, close, _, _ in fences(out):
        for index in range(start, min(close + 1, len(out))):
            out[index] = ""
    return out


def _blank_code(text):
    """The text with every code span's content blanked, positions kept."""
    return _CODE_SPAN.sub(lambda m: m[1] + re.sub(r"[^\n]", " ", m[2]) + m[1], text)


def _plain(text):
    """A heading's text as GitHub renders it: the markup gone, code kept as written."""
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"!?\[([^\]]*)\]\[[^\]]*\]", r"\1", text)
    out, last = [], 0
    for m in _CODE_SPAN.finditer(text):
        out.append(_unmarked(text[last:m.start()]))
        out.append(m[2])
        last = m.end()
    out.append(_unmarked(text[last:]))
    return html.unescape("".join(out)).strip()


def _unmarked(text):
    """Prose with its HTML tags, escapes and emphasis underscores taken out."""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\\(.)", r"\1", text)
    # an underscore inside a word is a character, at a word's edge emphasis
    return re.sub(r"_+", lambda m: m[0] if (
        0 < m.start() and m.end() < len(text)
        and text[m.start() - 1].isalnum() and text[m.end()].isalnum()) else "", text)


def github_slug(text):
    """GitHub's anchor for a heading's rendered text.

    Lower case; every character dropped but a letter, a mark, a number, a
    connector such as ``_``, a hyphen and a space; each space a hyphen.  A
    repeated anchor takes ``-1``, ``-2``, ... in page order (:func:`headings`).
    """
    kept = (c for c in text.lower()
            if c in "- " or unicodedata.category(c)[0] in "LMN"
            or unicodedata.category(c) == "Pc")
    return "".join(kept).replace(" ", "-")


def headings(page):
    """Every heading of a Markdown page: (line, its text as rendered, its anchor)."""
    lines = _prose(page)
    found = []
    for index, line in enumerate(lines):
        m = _ATX.match(line)
        if m:
            found.append((index + 1, _plain(m[2] or "")))
        elif (index and _SETEXT.match(line) and lines[index - 1].strip()
              and not _ATX.match(lines[index - 1]) and not _LIST_ITEM.match(lines[index - 1])
              and not lines[index - 1].lstrip().startswith("|")):
            found.append((index, _plain(lines[index - 1])))
    out, seen = [], {}
    for lineno, text in found:
        base = slug = github_slug(text)
        while slug in seen:
            seen[base] += 1
            slug = f"{base}-{seen[base]}"
        seen[slug] = 0
        out.append((lineno, text, slug))
    return out


_ANCHORS = {}


def anchors(page):
    """Every anchor GitHub gives a Markdown page: its headings', and any HTML anchor."""
    key = pathlib.Path(page).resolve()
    if key not in _ANCHORS:
        _ANCHORS[key] = ({slug for _, _, slug in headings(key)}
                         | set(_HTML_ANCHOR.findall("\n".join(_prose(key)))))
    return _ANCHORS[key]


def page_links(page):
    """Every relative link of a page, as (line, target), in page order.

    Inline links and images, and reference definitions; a link inside code, a
    fenced block or a code span, is code and not a link, and a target with a
    scheme (``https:``, ``mailto:``) is not relative.
    """
    text = _blank_code("\n".join(_prose(page)))
    found = [(m.start(), m[1]) for m in _INLINE_LINK.finditer(text)]
    found += [(m.start(), m[1]) for m in _REFERENCE.finditer(text)]
    out = []
    for position, target in sorted(found):
        target = target.strip("<>")
        if target and not _SCHEME.match(target):
            out.append((text.count("\n", 0, position) + 1, target))
    return out


_LISTINGS = {}


def _exists_as_written(path, root):
    """Whether a path exists, each name below the root spelled as it is on disk.

    Windows and macOS compare file names without case and GitHub with it, so a
    link that differs from a file's name only in case breaks where it is read.
    """
    path = pathlib.Path(os.path.normpath(path))
    if not path.exists():
        return False
    base = pathlib.Path(os.path.normpath(pathlib.Path(root).resolve()))
    try:
        names = path.relative_to(base).parts
    except ValueError:
        return True
    folder = base
    for name in names:
        if folder not in _LISTINGS:
            _LISTINGS[folder] = set(os.listdir(folder))
        if name not in _LISTINGS[folder]:
            return False
        folder = folder / name
    return True


def link_problem(page, target, root=ROOT):
    """Why one relative link of a page does not resolve, or ``None`` when it does.

    A path is read from the page's own folder, or from the root where it
    starts with ``/``, as GitHub reads it.
    """
    page = pathlib.Path(page).resolve()
    path, _, anchor = target.partition("#")
    path, anchor = urllib.parse.unquote(path), urllib.parse.unquote(anchor)
    if path.startswith("/"):
        destination = pathlib.Path(root).resolve() / path.lstrip("/")
    else:
        destination = page.parent / path if path else page
    if not _exists_as_written(destination, root):
        return "no such file"
    if not anchor:
        return None
    if destination.is_dir():
        return "an anchor on a folder"
    if destination.suffix.lower() != ".md":
        return None
    if anchor not in anchors(destination):
        return f"no heading there has the anchor #{anchor}"
    return None


def broken_links(page, root=ROOT):
    """Every relative link of a page that does not resolve: (line, target, why)."""
    out = []
    for lineno, target in page_links(page):
        problem = link_problem(page, target, root)
        if problem:
            out.append((lineno, target, problem))
    return out


def links_report(page, root=ROOT):
    """The page's links, as the command line prints them; and whether any is broken."""
    found = page_links(page)
    broken = broken_links(page, root)
    out = [f"{_label(page, root)}: {len(found)} relative link{'s' * (len(found) != 1)}, "
           f"{len(broken)} broken"]
    out += [f"  line {lineno}: {target} -- {why}" for lineno, target, why in broken]
    return "\n".join(out), bool(broken)


# ---------------------------------------------------------------------------
# the command line
# ---------------------------------------------------------------------------


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["--child"]:
        return _child(argv[1])
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pages", nargs="*", default=[],
                        help="the pages to check (default: every user page)")
    parser.add_argument("--update", action="store_true",
                        help="rewrite each statement's output lines from what it prints")
    parser.add_argument("--register", action="store_true",
                        help="also count the page's hits under every rule of the register")
    parser.add_argument("--links", action="store_true",
                        help="also check that the page's relative links resolve")
    args = parser.parse_args(argv)

    pages = ([pathlib.Path(page).resolve() for page in args.pages]
             or [ROOT / page for page in user_pages()])
    failed = []
    for page in pages:
        if not page.is_file():
            print(f"{page}: no such page")
            failed.append(page)
            continue
        run = check(page, update=args.update)
        print(run.report())
        bad = run.failed
        if args.register:
            text, bad_register = register_report(page)
            print(text)
            bad |= bad_register
        if args.links:
            text, bad_links = links_report(page)
            print(text)
            bad |= bad_links
        if bad:
            failed.append(page)
    if len(pages) > 1:
        print(f"{len(pages)} pages, {len(failed)} failing")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
