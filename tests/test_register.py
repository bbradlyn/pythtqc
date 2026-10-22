"""The register of the text this package puts in front of a reader, as a static scan.

Four classes of text carry the package's words to the physicist using it: the
docstrings and the comments of ``src/pythtqc``, the ten worked examples with
their comments and every string they could print, and the user pages -- ``README.md``
and the pages of ``docs/``, the map ``docs/README.md`` among them.  The pages
are a listing, not one path: the one ``scripts/check_page.py`` runs them
by is the one read here, so a page is
scanned the day it exists, and the class's budgets are sums over its pages.
The register those words obey is one voice -- physics first, present tense,
third person, the implementation out of sight -- and this module measures how
far each class is from it, by pattern, without importing the package or
running any of it.

A **budget** is a measurement, never a licence.  Each rule carries, per class,
the number of hits that class holds; the test fails when the count rises above
the budget, listing every offending ``file:line``, and it fails just as loudly
when the count falls below it, naming the number the budget should now be.  So
a rewrite that removes ten hits cannot leave a stale ten behind in the table:
the table always says what the tree says.

The rules, by name:

``history``
    A sentence says what the code does, not what was decided about it, when, or
    by whom: no dates, no process, no pointers to a planning document kept
    outside this repository.  On a user page every section mark is such a
    pointer, whatever its number: a page links to the heading it means.
``spellings``
    Names the package does not use.
``terms``
    One term, one meaning, one name.  Each pattern is a spelling the package
    does not use for a thing it already names: the noun *reading* for an
    environment, *re-presentation* for a presentation, *readout* / *answer* /
    *result* for a verdict, *group element* for an operation, *HSP* for a
    high-symmetry point, *BR* in prose for an elementary band representation,
    bare *Bilbao* or *BCS index* for the tables, *the suite* for the test suite.
``impl-hard`` / ``impl-soft``
    The implementation register in prose.  A docstring says what a thing *is*
    (a mapping from label to multiplicity; the operations a Hamiltonian obeys),
    not what holds it.  Types belong in a *Parameters* / *Returns* /
    *Attributes* / *Yields* / *Raises* / *Refusals* entry, and a hit inside one
    of those entries is not counted.
``voice-first-person`` / ``voice-hedge`` / ``voice-exclaim``
``voice-question`` / ``voice-you``
    Third person and no chatter.  A page may address its reader as "you" where
    it gives an instruction, and a guide chapter's title is the reader's
    question, so the pages' counts of those two are budgets like any other.
``gates-named``
    Every gate a public function -- or a public method of an exported class --
    can return is named in its own docstring, so a reader meets the verdict
    before the code returns it.  A gate reaches a reader whether the
    ``Refusal`` literal stands in the public body or in a private function,
    of its own module or imported from another module of the package, whose
    result that body hands back as it is, so the scan follows those hand-backs
    transitively and across modules; a private function's refusal the caller
    unpacks, raises on or replaces is not the caller's verdict and is not
    counted.  A gate the walk reaches that no public call can actually
    serve is carved out in ``_UNSERVABLE``, with what was measured -- by
    ``(module, name, gate)`` where one call is the exception, or by the bare
    gate name where no public call anywhere can produce it, which is the case
    of ``lattice_not_descended`` now that the geometry-only identification is
    private and a realized group always has lattice index 1.
``small-literals``
    A number a judgement compares against is named for what it compares, in
    one place: ``src/pythtqc/_tolerance.py``, which says of each what it was
    measured against.  A judgement of the noise in a model's own data reads
    the declared precision through it, and a judgement of exact arithmetic or
    of rounding reads a fixed number named there.  The scan counts the float
    literals below ``1e-2`` that still stand in the code of every other module
    of ``src/pythtqc`` -- a number in a docstring or a comment is text, not
    code -- and the budget falls to zero as each is named.

Code font is removed before any word is counted (``double-backtick`` and
`single-backtick` spans, :role:`targets`, and fenced blocks in the documents),
because a ``dict`` in code font is a type and not prose.  ``history`` and
``spellings`` read the text before that removal: both look for names, and a name
in code font is exactly the case each of them is for.

The carve-outs, in one place.  Each is a phrase the package owns, measured on
this tree, so that a rewrite does not delete good text to satisfy a pattern:

* A type inside a *Parameters*, *Returns*, *Attributes*, *Yields*, *Raises* or
  *Refusals* entry is the honest type of a served record, and ``impl-hard`` and
  ``impl-soft`` do not read those entries.  ``terms`` does read them: a retired
  name is retired in a *Returns* entry too.
* The paper ``models.py`` quotes verbatim keeps its first person.
* "actually obeys" is the package's own phrase and not a hedge, and so is
  "actually has" -- the same claim about the same measurement, said of the
  group instead of the operations.
* A conversion field (``{x!r}``), a comparison (``!=``) and a LaTeX thin space
  (a backslash before the mark) are not exclamations.
* A docstring's first line may be a question -- the four module framing lines
  and the predicates; a question anywhere else is rhetoric.
* ``BR`` is the paper's matrix symbol where a dot binds it (``BR.p``, ``c.BR``)
  or ``ker`` stands before it, and a symbol in a formula stays.
* "Bilbao" carries its qualifier -- Crystallographic Server, website, site,
  server -- which may wrap to the next line, so a "Bilbao" that ends its line is
  read as qualified.
* The verb "answers" stays, and only the noun is counted: singular behind a
  determiner ("a partial answer"), plural behind a counting word ("two
  answers").
* "the test suite", spelled in full, is not "the suite".
"""

from __future__ import annotations

import ast
import importlib.util
import io
import os
import re
import tokenize
from collections import namedtuple

# ---------------------------------------------------------------------------
# the budgets
# ---------------------------------------------------------------------------
#
# One table, keyed (rule, class).  Each number is a count measured on this tree.
# A step that rewrites a class lowers that class's budgets to the count it
# measures on landing -- the failure message carries the number, so the table is
# never a guess.  A budget of 0 is the finished state: the rule holds there.

BUDGETS = {
    # ---- history and process: none, on any surface ----
    ("history", "src docstrings"): 0,
    ("history", "src comments"): 0,
    ("history", "examples"): 0,
    ("history", "pages"): 0,
    # ---- names the package does not use: none, on any surface ----
    ("spellings", "src docstrings"): 0,
    ("spellings", "src comments"): 0,
    ("spellings", "examples"): 0,
    ("spellings", "pages"): 0,
    # ---- one term, one meaning, one name ----
    # What stands in `src` docstrings is a name, a theorem or a plain verb: the
    # parameter `readouts` (`sewing.py`), "the paper's result" -- a theorem of
    # the literature and not a returned value -- and the plain-English "serves"
    # in `groups.py` ("context ... serves"); one more "serves" stands in an
    # `ebr.py` comment ("one detector serves the").
    ("terms", "src docstrings"): 3,
    ("terms", "src comments"): 1,
    ("terms", "examples"): 0,
    ("terms", "pages"): 0,
    # ---- the implementation register in prose ----
    # The one src hit that stands: `tuple(...)` inside a doctest of `groups.py`
    # (code the shown output needs).
    ("impl-hard", "src docstrings"): 1,
    ("impl-hard", "src comments"): 0,
    ("impl-hard", "examples"): 0,
    ("impl-hard", "pages"): 0,
    ("impl-soft", "src docstrings"): 0,
    ("impl-soft", "src comments"): 0,
    ("impl-soft", "examples"): 0,
    ("impl-soft", "pages"): 0,
    # ---- voice ----
    # The two src hedges that stand are physics: "just below one" and "just
    # above zero" in `groups.py`.
    ("voice-first-person", "src docstrings"): 0,
    ("voice-first-person", "src comments"): 0,
    ("voice-first-person", "examples"): 0,
    ("voice-first-person", "pages"): 0,
    ("voice-hedge", "src docstrings"): 2,
    ("voice-hedge", "src comments"): 0,
    ("voice-hedge", "examples"): 0,
    ("voice-hedge", "pages"): 0,
    # Every exclamation mark in `src` and the examples is LaTeX's `\!` or a
    # `!=`.
    ("voice-exclaim", "src docstrings"): 0,
    ("voice-exclaim", "src comments"): 0,
    ("voice-exclaim", "examples"): 0,
    ("voice-exclaim", "pages"): 0,
    ("voice-question", "src docstrings"): 0,
    ("voice-question", "src comments"): 0,
    ("voice-question", "examples"): 0,
    # the titles of the ten chapters of `docs/guide/`, each the question its
    # reader brings
    ("voice-question", "pages"): 10,
    ("voice-you", "src docstrings"): 0,
    ("voice-you", "src comments"): 0,
    ("voice-you", "examples"): 0,
    ("voice-you", "pages"): 0,
    # ---- the gates of a public function, named in its own docstring ----
    ("gates-named", "src docstrings"): 0,
    # ---- the numbers a judgement compares against, named ----
    # Every judgement of the noise in a model's own data reads the declared
    # precision by name.  What stands is exact arithmetic on exact objects --
    # the tables' printed decimals, integer rotations, a pure translation -- the
    # rounding of the package's own arithmetic and its guards, and the
    # identification's own numbers, each to be named in `_tolerance.py`.
    ("small-literals", "src code"): 204,
}

CLASSES = (
    "src docstrings",
    "src comments",
    "examples",
    "pages",
)

WORD_RULES = (
    "history",
    "spellings",
    "terms",
    "impl-hard",
    "impl-soft",
    "voice-first-person",
    "voice-hedge",
    "voice-exclaim",
    "voice-question",
    "voice-you",
)

# ---------------------------------------------------------------------------
# where the text lives
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
_SRC = os.path.join(_ROOT, "src", "pythtqc")
_EXAMPLES = os.path.join(_ROOT, "examples")

#: The tool that runs the user pages, and lists them: the register reads its
#: list rather than a second one, so a page the gate runs is a page scanned.
_PAGE_TOOL = os.path.join(_ROOT, "scripts", "check_page.py")


def _pages():
    """The user pages, as paths relative to the root, by the pages' own listing."""
    spec = importlib.util.spec_from_file_location("tqc_check_page", _PAGE_TOOL)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    return tool.user_pages(_ROOT)


def _read(path):
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def _relative(path):
    return os.path.relpath(path, _ROOT).replace(os.sep, "/")


# ---------------------------------------------------------------------------
# code font
# ---------------------------------------------------------------------------
#
# The spans are removed from the whole block at once, and each is replaced by
# the newlines it spanned, so a line keeps its number and a span that wraps is
# removed whole.  A role target and a double-backtick span may wrap and never
# hold a backtick, so both are bounded by the next backtick; a single-backtick
# span may wrap once and no further, so that one unpaired backtick costs a line
# or two rather than inverting every span after it.

_ROLE = re.compile(r":[A-Za-z_][A-Za-z0-9_.+:-]*:`[^`]*`", re.S)
_DOUBLE_TICK = re.compile(r"``[^`]*``", re.S)
_SINGLE_TICK = re.compile(r"`[^`\n]*(?:\n[^`\n]*)?`")


def _blank(match):
    return "\n" * match.group(0).count("\n")


def _strip_code_font(text):
    """The text with role targets and backtick spans removed, line count kept."""
    text = _ROLE.sub(_blank, text)
    text = _DOUBLE_TICK.sub(_blank, text)
    return _SINGLE_TICK.sub(_blank, text)


# ---------------------------------------------------------------------------
# numpydoc sections
# ---------------------------------------------------------------------------

_UNDERLINE = re.compile(r"^\s*-{3,}\s*$")
_HEADER = re.compile(r"^\s*([A-Z][A-Za-z]*(?: [A-Za-z]+){0,2})\s*$")

#: A hit inside one of these entries is a type or a verdict name, not prose.
_TYPE_SECTIONS = frozenset(
    {"Parameters", "Returns", "Attributes", "Yields", "Raises", "Refusals"}
)


def _sections_of(block_lines):
    """The numpydoc section each line of one docstring sits in ("" for none)."""
    out = []
    current = ""
    for index, line in enumerate(block_lines):
        nxt = block_lines[index + 1] if index + 1 < len(block_lines) else ""
        header = _HEADER.match(line)
        if header and _UNDERLINE.match(nxt):
            current = header.group(1)
        out.append(current)
    return out


# ---------------------------------------------------------------------------
# the records
# ---------------------------------------------------------------------------

#: One line of text: where it is, what it says with and without code font, the
#: numpydoc section it sits in, and whether it opens a docstring.
Line = namedtuple("Line", "path lineno raw prose section first")


def _docstring_nodes(tree):
    """Every module, class and function docstring node in one parsed file."""
    out = []
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        if not body:
            continue
        head = body[0]
        if (
            isinstance(head, ast.Expr)
            and isinstance(head.value, ast.Constant)
            and isinstance(head.value.value, str)
        ):
            out.append(head.value)
    return out


def _docstring_lines(path, source, tree):
    rel = _relative(path)
    source_lines = source.splitlines()
    records = []
    for node in _docstring_nodes(tree):
        start, end = node.lineno, node.end_lineno
        block = source_lines[start - 1 : end]
        sections = _sections_of(block)
        stripped = _strip_code_font("\n".join(block)).splitlines()
        stripped += [""] * (len(block) - len(stripped))
        for offset, raw in enumerate(block):
            records.append(
                Line(
                    path=rel,
                    lineno=start + offset,
                    raw=raw,
                    prose=stripped[offset],
                    section=sections[offset],
                    first=offset == 0,
                )
            )
    return records


def _comment_lines(path, source):
    rel = _relative(path)
    records = []
    reader = io.StringIO(source).readline
    for token in tokenize.generate_tokens(reader):
        if token.type != tokenize.COMMENT:
            continue
        records.append(
            Line(
                path=rel,
                lineno=token.start[0],
                raw=token.string,
                prose=_strip_code_font(token.string),
                section="",
                first=False,
            )
        )
    return records


def _printed_literals(node):
    """The string literals of one expression; a replacement field is code."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        yield node
    elif isinstance(node, ast.JoinedStr):
        for part in node.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                yield part
    elif isinstance(node, ast.BinOp):
        yield from _printed_literals(node.left)
        yield from _printed_literals(node.right)
    elif isinstance(node, ast.IfExp):
        yield from _printed_literals(node.body)
        yield from _printed_literals(node.orelse)


def _literal_lines(path, tree):
    """Every string an example could print: a ``print`` argument, a ledger line's
    label or value, the text a printing function returns.  A literal without a
    space is a key, a mode or a symbol, not prose; a bare string statement is a
    docstring, which ``_docstring_lines`` reads."""
    rel = _relative(path)
    docstrings = {id(node.value) for node in ast.walk(tree)
                  if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)}
    parts = {id(part) for node in ast.walk(tree) if isinstance(node, ast.JoinedStr)
             for part in node.values}
    records = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Constant, ast.JoinedStr)):
            continue
        if id(node) in docstrings or id(node) in parts:
            continue
        for literal in _printed_literals(node):
            for piece in literal.value.splitlines() or [""]:
                if " " not in piece.strip():
                    continue
                records.append(
                    Line(
                        path=rel,
                        lineno=literal.lineno,
                        raw=piece,
                        prose=_strip_code_font(piece),
                        section="",
                        first=False,
                    )
                )
    return records


_FENCE = re.compile(r"^[\s>]*(```|~~~)")


def _markdown_lines(relative):
    """The prose lines of one document; a fenced block is code, not prose.

    A fence is written with backticks of its own, so the fenced lines come out
    of the text *before* code font is stripped from what is left.
    """
    path = os.path.join(_ROOT, relative)
    raw_lines = _read(path).splitlines()
    prose_only = []
    fenced = False
    inside = []
    for raw in raw_lines:
        if _FENCE.match(raw):
            fenced = not fenced
            inside.append(True)
            prose_only.append("")
            continue
        inside.append(fenced)
        prose_only.append("" if fenced else raw)
    stripped = _strip_code_font("\n".join(prose_only)).splitlines()
    stripped += [""] * (len(raw_lines) - len(stripped))
    records = []
    for index, raw in enumerate(raw_lines):
        if inside[index]:
            continue
        records.append(
            Line(
                path=relative.replace(os.sep, "/"),
                lineno=index + 1,
                raw=raw,
                prose=stripped[index],
                section="",
                first=False,
            )
        )
    return records


def _source_files(directory, pattern):
    return sorted(
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if pattern.match(name)
    )


_SRC_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\.py$")
_EXAMPLE_NAME = re.compile(r"^tqc_.*\.py$")


def _build(text_class):
    if text_class == "src docstrings":
        records = []
        for path in _source_files(_SRC, _SRC_NAME):
            source = _read(path)
            records += _docstring_lines(path, source, ast.parse(source))
        return records
    if text_class == "src comments":
        records = []
        for path in _source_files(_SRC, _SRC_NAME):
            records += _comment_lines(path, _read(path))
        return records
    if text_class == "examples":
        records = []
        for path in _source_files(_EXAMPLES, _EXAMPLE_NAME):
            source = _read(path)
            tree = ast.parse(source)
            records += _docstring_lines(path, source, tree)
            records += _comment_lines(path, source)
            records += _literal_lines(path, tree)
        return records
    if text_class == "pages":
        records = []
        for relative in _pages():
            records += _markdown_lines(relative)
        return records
    raise KeyError(f"no text class {text_class!r}")


_TEXT = {}


def text_of(text_class):
    """Every line of one text class, measured once per session."""
    if text_class not in _TEXT:
        _TEXT[text_class] = _build(text_class)
    return _TEXT[text_class]


# ---------------------------------------------------------------------------
# rule 1 -- history and process
# ---------------------------------------------------------------------------
#
# "no longer", a bare "audit" (this package audits its own answers), "rules out"
# and a numbered step of an example's narration are NOT here: each is physics or
# a reading, and the census measured them as false positives.

HISTORY = (
    ("a date", re.compile(r"\b20\d\d-\d\d-\d\d\b")),
    # a pointer to a planning document kept outside this repository; the old
    # spellings stay in the pattern, since a surface that names them names the
    # document just as much
    (
        "the project's record",
        re.compile(
            r"DESIGN_NOTES|\bSTATE\.md\b|\btqc_plan\b"
            r"|\bdocs/design\b|\bdesign/(?:README|history)\b"
        ),
    ),
    ("the owner", re.compile(r"\bthe owner\b|\bowner's ruling\b", re.I)),
    (
        "a ruling",
        re.compile(r"\b(ruled|ruling) (on|that|by)\b|\bby (his|the owner's) ruling\b", re.I),
    ),
    (
        "a wave, session or step of work",
        re.compile(
            r"\bwave [A-Za-z]?\d|\bsince wave\b|\bthis wave\b|\bthe wave\b"
            r"|\bthis session\b|\bstep \d+ landed\b",
            re.I,
        ),
    ),
    (
        "a pass over the project",
        re.compile(r"\bthe readability pass\b|\bthe cleanup wave\b|\bthe API wave\b", re.I),
    ),
    (
        "a spelling kept for an old caller",
        re.compile(r"\bdeprecat|\bcompatibility shim\b|\bshim for the old\b", re.I),
    ),
    (
        "a machine",
        re.compile(
            r"\bOpenBLAS\b|\bthis machine\b|\bApple silicon\b|\bthe M1\b|\bx86\b|\barm64\b",
            re.I,
        ),
    ),
    ("an unfinished note", re.compile(r"\b(TODO|FIXME|XXX|HACK)\b")),
    (
        "what it used to do",
        re.compile(
            r"\bused to (be|read|mean|say|carry|spell|return|answer)\b"
            r"|\bpreviously\b|\bformerly\b"
            r"|\bthe old (name|spelling|default|behaviour|behavior)\b",
            re.I,
        ),
    ),
    (
        "before or after a change",
        re.compile(r"\b(before|after) (the|this) (change|rename|wave|step|commit|fix)\b", re.I),
    ),
    (
        "a commit, a branch, a worktree",
        re.compile(r"\bworktree\b|\bthis commit\b|\bthe commit\b|\bgit (log|commit|branch)\b", re.I),
    ),
    (
        "a baseline being rebuilt",
        re.compile(r"\bthe baseline was\b|\bregenerat|\btripwire\b", re.I),
    ),
    ("a rewrite of the code", re.compile(r"\brefactor", re.I)),
    (
        "a dated measurement",
        re.compile(r"\b(measured|checked|verified|audited|counted) (on )?20\d\d", re.I),
    ),
    # a package's own letter ("package Q4") and a step label's possessive
    # ("S1b's") -- never a bare step label, which collides with an irrep name
    # such as `A1` or `S2`
    (
        "a package label",
        re.compile(r"\bpackage [A-Z]\w*"),
    ),
    (
        "a step label's possessive",
        re.compile(r"\b[A-Z]\d*[a-z]?'s\b"),
    ),
    ("a benchmark recorded", re.compile(r"\bbenchmark recorded\b", re.I)),
    # The two-digit-and-up sections (`§ 36`, `§ 49`, `§ 50`) of a planning
    # document kept outside this repository -- the ones a src surface has no
    # business pointing at.  The pages read every section mark, whatever its
    # number (`_CLASS_PATTERNS`).
    (
        "a design-record section number",
        re.compile(r"§+\s*\d{2,}"),
    ),
)

# ---------------------------------------------------------------------------
# rule 8 -- names the package does not use
# ---------------------------------------------------------------------------
#
# Read against the raw text, code font included.  `n_bands` (a count field),
# `"measured"` (a record key), `tqc.symmetrize` (the function) and
# `irrep_labels` (irreptables' own raw key) are the package's own names and are
# deliberately absent from this list.

SPELLINGS = (
    ("occ_or_bands", re.compile(r"\bocc_or_bands\b")),
    ("reference_wyckoff", re.compile(r"\breference_wyckoff\b")),
    ("reference_letters", re.compile(r"\breference_letters\b")),
    ("spglib_wyckoff", re.compile(r"\bspglib_wyckoff\b")),
    ("wyckoff_letters", re.compile(r"\bwyckoff_letters\b")),
    ("bcs_irrep_labels", re.compile(r"\bbcs_irrep_labels\b")),
    ("labels_tables", re.compile(r"\blabels_tables\b")),
    ("operation_closure", re.compile(r"\boperation_closure\b")),
    ("operation_defect", re.compile(r"\boperation_defect\b")),
    ("svd as a solver word", re.compile(r"""["']svd(_subset)?["']""")),
    ("gram_threshold", re.compile(r"\bgram_threshold\b")),
    ("screw_orbit, screw_bonds, screw_atomic", re.compile(r"\bscrew_(orbit|bonds|atomic)\b")),
    ("su2_bonds", re.compile(r"\bsu2_bonds\b")),
    ("orientation_branch=", re.compile(r"\borientation_branch\s*=\s*(True|False)\b")),
    ("Thresholds.tol", re.compile(r"\bThresholds\.tol\b")),
    ("sewing_matrix(ind=", re.compile(r"\bsewing_matrix\s*\([^)]*\bind\s*=")),
    ("strict= as a keyword", re.compile(r"\bstrict\s*=")),
    ("valence= as a keyword", re.compile(r"\bvalence\s*=")),
    ("GroupData.number", re.compile(r"\bGroupData\.number\b")),
    ("decorations=() as the environment", re.compile(r"\bdecorations\s*=\s*\(\s*\)")),
    (".named as the SRSI attribute", re.compile(r"\.named\b")),
    ('verdict == "measured"', re.compile(r"""\bverdict\s*==\s*["']measured["']""")),
    # retired by the sweep of 2026-09-14; `rep.search` is a kept record name, so
    # that keyword is scanned together with its value
    ("keep_reps", re.compile(r"\bkeep_reps\b")),
    ("ValidatedGroup.reps", re.compile(r"\.reps\b")),
    ('build_rep(search="readoff" | "enumerate")',
     re.compile(r"""\bsearch\s*=\s*["'](readoff|enumerate)["']""")),
    ('decoration["species"]', re.compile(r"""\bdecoration\s*\[\s*["']species["']\s*\]""")),
    ("sg199_ / SG199_ builders", re.compile(r"\b(sg199|SG199)_")),
    ("identify_from_characters(tol=", re.compile(r"\bidentify_from_characters\s*\([^)]*\btol\s*=")),
    ("verdict_trivial", re.compile(r"\bverdict_trivial\b")),
    # retired by the interface package and its ratchet: each confirmed absent
    # on every surface this module scans before joining the list.
    ("ConventionError", re.compile(r"\bConventionError\b")),
    ("descent_failed", re.compile(r"\bdescent_failed\b")),
    ("environment_mismatch", re.compile(r"\benvironment_mismatch\b")),
    ("character_mismatch", re.compile(r"\bcharacter_mismatch\b")),
    ("declaration_needs_complex_twist", re.compile(r"\bdeclaration_needs_complex_twist\b")),
    ("site_group_order", re.compile(r"\bsite_group_order\b")),
    ("DEFAULT_STATUS", re.compile(r"\bDEFAULT_STATUS\b")),
    ("ChannelLayout", re.compile(r"\bChannelLayout\b")),
    ("offset_label", re.compile(r"\boffset_label\b")),
    ("_reference_scalars", re.compile(r"\b_reference_scalars\b")),
    ("rep.layout", re.compile(r"\brep\.layout\b")),
    ('frames="shared"', re.compile(r"""\bframes\s*=\s*["']shared["']""")),
    ("CharacterAssignment", re.compile(r"\bCharacterAssignment\b")),
    ("SiteCharacter", re.compile(r"\bSiteCharacter\b")),
    ("atomic_candidates", re.compile(r"\batomic_candidates\b")),
    ("AtomicCandidate", re.compile(r"\bAtomicCandidate\b")),
    ("candidates_v", re.compile(r"\bcandidates_v\b")),
    ("candidates_c", re.compile(r"\bcandidates_c\b")),
    ("srsi_candidates_truncated", re.compile(r"\bsrsi_candidates_truncated\b")),
    ("zn_moduli", re.compile(r"\bzn_moduli\b")),
    ("GATE_MEANINGS", re.compile(r"\bGATE_MEANINGS\b")),
    ("theta_matrix", re.compile(r"\btheta_matrix\b")),
    ("ebr_transport", re.compile(r"\bebr_transport\b")),
    ("slab_embedding_changed_H", re.compile(r"\bslab_embedding_changed_H\b")),
    ("include_padded_ops", re.compile(r"\binclude_padded_ops\b")),
    ("shared_k_action", re.compile(r"\bshared_k_action\b")),
    ("GAUGE_TOL", re.compile(r"\bGAUGE_TOL\b")),
    ("gauge_tol", re.compile(r"\bgauge_tol\b")),
    ("winding_certificate", re.compile(r"\bwinding_certificate\b")),
    ("n_channels", re.compile(r"\bn_channels\b")),
    ("channel_of_orbital", re.compile(r"\bchannel_of_orbital\b")),
    ("measured_channel_fibers", re.compile(r"\bmeasured_channel_fibers\b")),
    ("_declaration_refusal", re.compile(r"\b_declaration_refusal\b")),
    ("_decline_descent", re.compile(r"\b_decline_descent\b")),
    ("_descent_position_defect", re.compile(r"\b_descent_position_defect\b")),
    ("_settle_twist", re.compile(r"\b_settle_twist\b")),
    ("_twist_weight", re.compile(r"\b_twist_weight\b")),
    ("_free_class_twists", re.compile(r"\b_free_class_twists\b")),
    ("_layout_of", re.compile(r"\b_layout_of\b")),
    ("_channel_layout", re.compile(r"\b_channel_layout\b")),
    ("_gauge_record", re.compile(r"\b_gauge_record\b")),
    ("_environment_mismatch", re.compile(r"\b_environment_mismatch\b")),
    ("served_head", re.compile(r"\bserved_head\b")),
    # retired when the spinor basis became its pairing: each confirmed absent on
    # every surface this module scans.
    ("sz_signs", re.compile(r"\bsz_signs\b")),
    ("spin_pairing_undeclared", re.compile(r"\bspin_pairing_undeclared\b")),
    ("SpinBasis.operator", re.compile(r"\bSpinBasis\.operator\b")),
    # retired when a model periodic in fewer than two directions became one
    # refusal, whatever the dimension of its cell: each confirmed absent on
    # every surface this module scans.
    ("not_a_2d_model", re.compile(r"\bnot_a_2d_model\b")),
    ("no_periodic_dirs", re.compile(r"\bno_periodic_dirs\b")),
    # retired when the pairs were read from the declaration, so that no call
    # reaches it
    ("spin_pair_permutation", re.compile(r"\bspin_pair_permutation\b")),
    # retired when the centres were made exact by one projection, which the
    # public symmetrize_positions reads too, so that no call reaches them
    ("_centre_snap", re.compile(r"\b_centre_snap\b")),
    ("_moved_centres", re.compile(r"\b_moved_centres\b")),
    ("fit_origin", re.compile(r"\bfit_origin\b")),
    # retired when the build took each factor between orbits of multiplets as
    # measured, so that nothing offers it the phases the hoppings pin
    ("_pinned_phases", re.compile(r"\b_pinned_phases\b")),
    # retired when the decoupled stack kept every orbital at its own height, so
    # that a slab is stacked as a flat layer is
    ("layer_not_flat", re.compile(r"\blayer_not_flat\b")),
    ("flat_tol", re.compile(r"\bflat_tol\b")),
    # retired when a content's sums of elementary band representations were put
    # in the tables' order, fewest first, so that nothing ranks the letters by
    # their distance from the orbitals
    ("_nearest_solutions", re.compile(r"\b_nearest_solutions\b")),
    ("_nearest_letters", re.compile(r"\b_nearest_letters\b")),
    ("_letter_places", re.compile(r"\b_letter_places\b")),
    # retired when a multiplet whose commutant the sectors do not read kept the
    # block its intertwiner gives, so that no sign per commutant projector is
    # chosen against the group's table
    ("_block_units", re.compile(r"\b_block_units\b")),
    # retired when the steps realized_group runs, and the record only one of
    # them returned, were made private, the declarations staying on
    # realized_group by name
    ("build_rep", re.compile(r"\bbuild_rep\b")),
    ("validate_group", re.compile(r"\bvalidate_group\b")),
    ("assign_orbitals", re.compile(r"\bassign_orbitals\b")),
    ("find_intertwiners", re.compile(r"\bfind_intertwiners\b")),
    ("IntertwinerSpace", re.compile(r"\bIntertwinerSpace\b")),
    ("phases=", re.compile(r"\bphases\s*=")),
    # retired with the indicator's route for a layer's representation that
    # records no environment, which no call reaches once the build is private
    ('completion "larger group"', re.compile(r"""["']larger group["']""")),
    ('completion["cell_group"]', re.compile(r"\bcell_group\b")),
    # retired when no call returned a representation that leaves an operation
    # out; `ValidatedGroup.undetermined` stays, so the field is scanned with
    # its owner
    ("rep.undetermined", re.compile(r"\b(rep|SpaceGroupRep)\.undetermined\b")),
    ('reason "vacuum direction"', re.compile(r"""["']vacuum direction["']""")),
    # retired when a representation short of its own little group became a
    # defect that raises
    ("little_group_incomplete", re.compile(r"\blittle_group_incomplete\b")),
    # retired when the returned fields were named for the physics: a descent
    # drops operations the Hamiltonian obeys too, so what it lists and counts is
    # what it dropped, by cause; the keys `broken` and `n_broken` stay on the
    # records whose operations are broken, so these two are scanned on the
    # descent
    ('descent["broken"]', re.compile(r"""\.descent\[["']broken["']\]""")),
    ('descent["n_broken"]', re.compile(r"""\.descent\[["']n_broken["']\]""")),
    ("n_declaration", re.compile(r"\bn_declaration\b")),
    ("n_projective", re.compile(r"\bn_projective\b")),
    ("n_centres", re.compile(r"\bn_centres\b")),
    ('descent["candidates"]', re.compile(r"""descent\[["']candidates["']\]""")),
    ('time_reversal["shared"]', re.compile(r"""time_reversal\[["']shared["']\]""")),
    ('site_content["labels"]', re.compile(r"""site_content\[["']labels["']\]""")),
    ('composite["labels"]', re.compile(r"""composite\[["']labels["']\]""")),
    ("integer_probe", re.compile(r"\binteger_probe\b")),
    ('beyond_grey["of"]', re.compile(r"""beyond_grey\[["']of["']\]""")),
    # retired when the build's own record, the detector's and the record of
    # irrep's setting went private, and the vacuum took its own name
    ("rep.operation(ind)", re.compile(r"\.operation\(")),
    ("rep.search", re.compile(r"\b(rep|SpaceGroupRep)\.search\b")),
    ("rep.choice", re.compile(r"\b(rep|SpaceGroupRep)\.choice\b")),
    ("rep.normalization", re.compile(r"\b(rep|SpaceGroupRep)\.normalization\b")),
    ("rep.battery", re.compile(r"\b(rep|SpaceGroupRep)\.battery\b")),
    ("rep.branch_tie", re.compile(r"\b(rep|SpaceGroupRep)\.branch_tie\b")),
    ("rep.time_reversal", re.compile(r"\b(rep|SpaceGroupRep)\.time_reversal\b")),
    ("validated.records", re.compile(r"\b(validated|ValidatedGroup)\.records\b")),
    ("validated.intertwiners", re.compile(r"\b(validated|ValidatedGroup)\.intertwiners\b")),
    ("validated.bases", re.compile(r"\b(validated|ValidatedGroup)\.bases\b")),
    ("validated.nullspace_dims", re.compile(r"\b(validated|ValidatedGroup)\.nullspace_dims\b")),
    ("group.refUC", re.compile(r"\b(group|GroupData)\.refUC\b")),
    ("group.shiftUC", re.compile(r"\b(group|GroupData)\.shiftUC\b")),
    ("group.tables_ready", re.compile(r"\b(group|GroupData)\.tables_ready\b")),
    ("group.setup_note", re.compile(r"\b(group|GroupData)\.setup_note\b")),
    ("group.closure_measured", re.compile(r"\b(group|GroupData)\.closure_measured\b")),
    ("group.closure_repaired", re.compile(r"\b(group|GroupData)\.closure_repaired\b")),
    ("group.lattice_index", re.compile(r"\b(group|GroupData)\.lattice_index\b")),
    ("group.fractional_translations", re.compile(r"\b(group|GroupData)\.fractional_translations\b")),
    ("group.spin_frame", re.compile(r"\b(group|GroupData)\.spin_frame\b")),
    ('tables_status()["dummy"]', re.compile(r"""["']dummy["']""")),
    ("padded_dirs", re.compile(r"\bpadded_dirs\b")),
    ("pad_length", re.compile(r"\bpad_length\b")),
    ("padded, for the vacuum", re.compile(r"\b[Pp]added\b")),
    # retired when a refusal's context named the physics: the build's search
    # stays its own, its refusals carrying the number of candidates tried; a
    # declaration's refusal names the irreps measured and declared, and the
    # readings tied; the cell group's refusal counts what refuted each operation
    # broken
    ('context["search"]', re.compile(r"""context\[["']search["']\]""")),
    ('context["choices"]', re.compile(r"""context\[["']choices["']\]""")),
    ('context["readings"]', re.compile(r"""context\[["']readings["']\]""")),
    ('context["reasons"]', re.compile(r"""context\[["']reasons["']\]""")),
    ("measured_label", re.compile(r"\bmeasured_label\b")),
    ("declared_labels", re.compile(r"\bdeclared_labels\b")),
    # retired when the cell group's refusal counted the centres under the
    # descent's own word
    ('refuted_by "the centres"', re.compile(r"""["']the centres["']""")),
    # retired when a presentation that changes what it only relabels became the
    # package disagreeing with itself, which raises
    ("slab_presentation_changed_H", re.compile(r"\bslab_presentation_changed_H\b")),
    ("stack_changed_H", re.compile(r"\bstack_changed_H\b")),
    ("spinless_presentation_changed_H",
     re.compile(r"\bspinless_presentation_changed_H\b")),
    ("position_gauge_changed_spectrum",
     re.compile(r"\bposition_gauge_changed_spectrum\b")),
    # dropped: the indices a descent step's dropped entries already carry
    ('steps[i]["ghosts"]', re.compile(r"""["']ghosts["']""")),
    # dropped from a descent's tensor-structure entries, whose residual is the
    # dimensionless miss
    ("ansatz_scale", re.compile(r"\bansatz_scale\b")),
    # dropped: the identity counted with the conserved matrices, dim + 1
    ('conserved["nullspace"]', re.compile(r"""conserved\[["']nullspace["']\]""")),
    # dropped: the wall-clock seconds the other environment took
    ('cross_check["cost"]', re.compile(r"""cross_check\[["']cost["']\]""")),
    # dropped: the solver's word for how the signed search ended, which
    # n_solutions and truncated say
    ('signed_combinations["status"]',
     re.compile(r"""signed_combinations\[["']status["']\]""")),
    # dropped: the question every cell is asked, always "grey"
    ('time_reversal["asked"]', re.compile(r"""time_reversal\[["']asked["']\]""")),
    # dropped from the projection's operators record: the private build's own
    # strategy, cost, choice, miss and gate
    ('operators["strategy"] and the four beside it',
     re.compile(r"""operators\[["'](strategy|n_enumeration|choice|ansatz_defect"""
                r"""|internal_tol)["']\]""")),
    ("ansatz_defect", re.compile(r"\bansatz_defect\b")),
    # dropped: a character table is built only where the representation
    # carries its whole little group, and raises otherwise
    ("CharacterTable.not_carried", re.compile(r"\bnot_carried\b")),
    ("CharacterTable.complete", re.compile(r"\bCharacterTable\.complete\b")),
    # retired when the SRSI layer took one loader, `srsi_tables`, and one
    # evaluation, `theta` / `theta_unreduced` on either shape of `p`, with no
    # shim: each confirmed absent on every surface this module scans before
    # joining the list
    ("srsi_table", re.compile(r"\bsrsi_table\b")),
    ("theta_of", re.compile(r"\btheta_of\b")),
    ("theta_of_sites", re.compile(r"\btheta_of_sites\b")),
    ("theta_values", re.compile(r"\btheta_values\b")),
    ("theta_values_of_sites", re.compile(r"\btheta_values_of_sites\b")),
    # the verdicts of a band splitting: a consistent pair of atomic realizations
    # makes any topology of the splitting invisible to the SRSIs, and the
    # criterion is sufficient, never necessary, so no splitting is called either
    # of these
    ("rescued", re.compile(r"\brescued\b", re.I)),
    ("trivializable", re.compile(r"\btrivializable\b", re.I)),
    # retired when the representation's own records beside the covariance it
    # achieves went private: how far each matrix misses its canonical norm and
    # its anchor to time reversal, the generators and their words, the reference
    # orbital and the operations broken; and the grid the detector evaluates H
    # on.  A field named by a common word is scanned with its owner
    ("norm_residuals", re.compile(r"\bnorm_residuals\b")),
    ("anchor_residuals", re.compile(r"\banchor_residuals\b")),
    ("rep.generators", re.compile(r"\b(rep|SpaceGroupRep)\.generators\b")),
    ("rep.words", re.compile(r"\b(rep|SpaceGroupRep)\.words\b")),
    ("rep.reference", re.compile(r"\b(rep|SpaceGroupRep)\.reference\b")),
    ("rep.failed", re.compile(r"\b(rep|SpaceGroupRep)\.failed\b")),
    ("GridHamiltonians", re.compile(r"\bGridHamiltonians\b")),
)

# ---------------------------------------------------------------------------
# rule 2 -- one term, one meaning, one name
# ---------------------------------------------------------------------------
#
# One pattern per retired spelling, read on the prose with code font gone,
# because a name in code font is code: `readouts` is a parameter, `BR` in a
# formula is the paper's matrix, `outcome` is a record key.  A *Parameters* or
# *Returns* entry is not exempt here -- a retired name is retired wherever a
# reader meets it.  Two limits are deliberate.  The bare noun "reading" is not
# scannable: it collides with the gerund a docstring uses throughout ("reading
# a label off a mesh"), so only its environment sense is on the list.  And
# "outcome" is not on the list at all, because the register sheet keeps it for
# the per-operation words `"obeyed"` / `"broken"` / `"undetermined"` and for
# the record key of that name.

TERMS = (
    ("a readout", re.compile(r"\breadouts?\b", re.I)),
    ("a re-presentation", re.compile(r"\bre-presentations?\b", re.I)),
    ("re-present", re.compile(r"\bre-present(s|ed|ing)?\b", re.I)),
    # `BR.p`, `c.BR` and `ker BR` are the paper's matrix symbol, which stays.
    ("BR in prose", re.compile(r"(?<!\.)(?<!\bker )\bBR\b(?!\.\w)")),
    ("a group element", re.compile(r"\bgroup elements?\b", re.I)),
    ("HSP", re.compile(r"\bHSPs?\b")),
    # The qualifier may wrap, so a "Bilbao" that ends its line is read as
    # carrying it.
    (
        "Bilbao without its server",
        re.compile(r"\bBilbao\b(?!\s*(?:$|Crystallographic|website|site|server))", re.I),
    ),
    ("the suite", re.compile(r"\bthe suite\b", re.I)),
    # The bare noun is the gerund far more often than the environment; this is
    # the sense the sheet retires, and it is the one that can be caught.
    (
        "a reading of an environment",
        re.compile(
            r"\b(substrate|free-standing|2D|3D|environment|other|both|each|two)"
            r"\s+readings?\b",
            re.I,
        ),
    ),
    ("the BCS index", re.compile(r"\bBCS index\b")),
    # The verb "answers" stays: the noun is singular behind a determiner, or
    # plural behind a counting word.
    (
        "an answer",
        re.compile(
            r"\b(an?|the|its)\s+(\w+\s+)?answer\b"
            r"|\b(two|both|different|these|those)\s+answers\b",
            re.I,
        ),
    ),
    ("a result", re.compile(r"\bresults?\b", re.I)),
    # the served -> returned sweep's own word; a plain-English "serves" may
    # still stand, so this is a budget, not a zero
    ("served", re.compile(r"\bserv(e|es|ed|ing)\b", re.I)),
)

# ---------------------------------------------------------------------------
# rule 7 -- the implementation register in prose
# ---------------------------------------------------------------------------

IMPL_HARD = (
    ("dict", re.compile(r"\bdicts?\b", re.I)),
    ("tuple", re.compile(r"\btuples?\b", re.I)),
    ("kwargs", re.compile(r"\bkwargs?\b", re.I)),
    ("helper", re.compile(r"\bhelpers?\b", re.I)),
    ("passthrough", re.compile(r"\bpass-?throughs?\b", re.I)),
    ("wrapper", re.compile(r"\bwrapp?ers?\b", re.I)),
    ("bookkeeping", re.compile(r"\bbook-?keeping\b", re.I)),
    ("sentinel", re.compile(r"\bsentinels?\b", re.I)),
    ("the routine", re.compile(r"\bthe routine\b", re.I)),
    ("the record says", re.compile(r"\bthe record (says|reads)\b", re.I)),
)

IMPL_SOFT = (
    ("caller", re.compile(r"\bcallers?\b", re.I)),
    ("this module", re.compile(r"\bthis module\b", re.I)),
    ("the function", re.compile(r"\b(the|this) function\b", re.I)),
    ("the code", re.compile(r"\bthe code\b", re.I)),
    ("internally", re.compile(r"\binternally\b", re.I)),
    ("under the hood", re.compile(r"\bunder the hood\b", re.I)),
)

# ---------------------------------------------------------------------------
# rule 6 -- voice
# ---------------------------------------------------------------------------

FIRST_PERSON = (("we, our, us", re.compile(r"\b(we|ours|our|us)\b", re.I)),)

HEDGES = (
    ("basically", re.compile(r"\bbasically\b", re.I)),
    ("simply", re.compile(r"\bsimply\b", re.I)),
    ("just", re.compile(r"\bjust\b", re.I)),
    ("obviously", re.compile(r"\bobviously\b", re.I)),
    ("of course", re.compile(r"\bof course\b", re.I)),
    ("note that", re.compile(r"\bnote that\b", re.I)),
    ("importantly", re.compile(r"\bimportantly\b", re.I)),
    ("actually", re.compile(r"\bactually\b", re.I)),
    ("in fact", re.compile(r"\bin fact\b", re.I)),
    ("really", re.compile(r"\breally\b", re.I)),
)

EXCLAIM = (("an exclamation", re.compile(r"(?<!\\)!(?![=rsa])")),)

QUESTION = (("a rhetorical question", re.compile(r"\?\s*$")),)

YOU = (("you, your", re.compile(r"\b(you|yours|your)\b", re.I)),)

# ---------------------------------------------------------------------------
# the carve-outs
# ---------------------------------------------------------------------------
#
# Each is a phrase the package owns, measured on this tree, so that a rewrite
# does not delete good text to satisfy a pattern.

#: ``models.py`` quotes a paper verbatim, first person and all: the entry is
#: the line that opens the quotation and the line that closes it.
_QUOTATIONS = (
    (re.compile(r'"By using s-orbitals as our basis functions'), re.compile(r'trivially\."')),
)

#: "which operations the Hamiltonian *actually* obeys" is the package's phrase,
#: and so is "the space group this Hamiltonian actually has" -- the owner's own
#: first line for `realized_group` in the physics-first wave.  The two say the
#: same thing about the same measurement, one of the operations and one of the
#: group they make up, and neither is a hedge.  Every other "actually" is.
_ACTUALLY_OBEYS = re.compile(r"\s*(\*\s*)?(obey(s|ed)?|has)\b", re.I)


def _verbatim_quote_lines():
    """The line numbers of the quotations ``models.py`` carries verbatim."""
    path = os.path.join(_SRC, "models.py")
    lines = _read(path).splitlines()
    out = set()
    for opens, closes in _QUOTATIONS:
        for index, line in enumerate(lines):
            if opens.search(line):
                end = index
                while end < len(lines) - 1 and not closes.search(lines[end]):
                    end += 1
                out.update(range(index + 1, end + 2))
                break
    return out


_QUOTE_LINES = None


def _in_verbatim_quote(record):
    global _QUOTE_LINES
    if _QUOTE_LINES is None:
        _QUOTE_LINES = _verbatim_quote_lines()
    return record.path.endswith("models.py") and record.lineno in _QUOTE_LINES


def _carved(rule, record, match):
    """True when this hit is a phrase the package keeps."""
    if rule == "voice-first-person":
        return _in_verbatim_quote(record)
    if rule == "voice-hedge":
        if match.group(0).lower() == "actually":
            return bool(_ACTUALLY_OBEYS.match(record.prose, match.end()))
        return False
    if rule == "voice-question":
        return record.first
    return False


# ---------------------------------------------------------------------------
# the scan
# ---------------------------------------------------------------------------

#: (patterns, which field they read, whether a type entry exempts a hit)
_RULES = {
    # `history` reads the raw text for the same reason `spellings` does: a
    # pointer to a planning document kept outside this repository is a file
    # name and would be written in code font, where stripping it would hide
    # exactly what the rule forbids.
    "history": (HISTORY, "raw", False),
    "spellings": (SPELLINGS, "raw", False),
    "terms": (TERMS, "prose", False),
    "impl-hard": (IMPL_HARD, "prose", True),
    "impl-soft": (IMPL_SOFT, "prose", True),
    "voice-first-person": (FIRST_PERSON, "prose", False),
    "voice-hedge": (HEDGES, "prose", False),
    "voice-exclaim": (EXCLAIM, "prose", False),
    "voice-question": (QUESTION, "prose", False),
    "voice-you": (YOU, "prose", False),
}

#: Where a class reads a rule its own way.  A user page links to the heading
#: it means, so every section mark on one -- one digit or three -- is a pointer
#: its reader cannot follow: the pages read ``history`` with the two-digit
#: pattern widened to every mark.
_CLASS_PATTERNS = {
    ("history", "pages"): tuple(
        entry for entry in HISTORY if entry[0] != "a design-record section number"
    ) + (("a section mark", re.compile(r"§+")),),
}

#: one hit: where it is, which pattern caught it, what it caught, the line
Hit = namedtuple("Hit", "path lineno label word text")


def hits(rule, text_class):
    """Every hit of one rule on one text class, in file and line order."""
    return scan(rule, text_of(text_class), text_class)


def scan(rule, records, text_class):
    """Every hit of one rule on some lines of one class, in file and line order.

    The lines need not be the whole class: ``scripts/check_page.py`` hands in
    one page's, so that a writer sees that page's counts by these patterns and
    no copy of them.
    """
    patterns, field, exempt_types = _RULES[rule]
    patterns = _CLASS_PATTERNS.get((rule, text_class), patterns)
    found = []
    for record in records:
        if exempt_types and record.section in _TYPE_SECTIONS:
            continue
        subject = record.raw if field == "raw" else record.prose
        if not subject.strip():
            continue
        for label, pattern in patterns:
            for match in pattern.finditer(subject):
                if _carved(rule, record, match):
                    continue
                found.append(
                    Hit(record.path, record.lineno, label, match.group(0), record.raw.strip())
                )
    found.sort(key=lambda hit: (hit.path, hit.lineno))
    return found


def _listing(found, limit=60):
    """The offending lines, so the next reader works from this message."""
    shown = [
        f"  {hit.path}:{hit.lineno}: [{hit.label}] {hit.text[:110]}" for hit in found[:limit]
    ]
    if len(found) > limit:
        shown.append(f"  ... and {len(found) - limit} more")
    return "\n".join(shown)


# ---------------------------------------------------------------------------
# rule 4 -- every gate named in its own docstring
# ---------------------------------------------------------------------------


def _exported(tree):
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            getattr(target, "id", None) == "__all__" for target in node.targets
        ):
            try:
                return set(ast.literal_eval(node.value))
            except (ValueError, SyntaxError):
                return set()
    return set()


def _gates_in(node):
    """The gate names of the ``Refusal(...)`` calls lexically inside a body.

    The gate is the first positional argument, ``Refusal("<gate>", ...)``, or
    the ``gate=`` keyword where a call spells it that way.
    """
    out = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        func = child.func
        named = (isinstance(func, ast.Name) and func.id == "Refusal") or (
            isinstance(func, ast.Attribute) and func.attr == "Refusal"
        )
        if not named:
            continue
        gate = child.args[0] if child.args else next(
            (kw.value for kw in child.keywords if kw.arg == "gate"), None)
        if isinstance(gate, ast.Constant) and isinstance(gate.value, str):
            out.add(gate.value)
    return out


def _private_functions(tree):
    """Every module-level function whose name marks it private, by name."""
    return {node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("_")}


def _imported_privates(tree):
    """Every private name a module imports from another module of the package: bound name -> (module, name)."""
    out = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
            for alias in node.names:
                if alias.name.startswith("_"):
                    out[alias.asname or alias.name] = (node.module.split(".")[0], alias.name)
    return out


_MODULES = {}


def _modules():
    """Every module of the package, parsed once: its path, its tree, its private functions and the private names it imports."""
    if not _MODULES:
        for path in _source_files(_SRC, _SRC_NAME):
            tree = ast.parse(_read(path))
            _MODULES[os.path.basename(path)[:-3]] = (
                path, tree, _private_functions(tree), _imported_privates(tree))
    return _MODULES


def _resolve(module, name):
    """The definition a private name stands for in ``module``: ``(module, node)``, through its imports; ``None`` for a name that is no function."""
    seen = set()
    while (module, name) not in seen:
        seen.add((module, name))
        _, _, own, imported = _modules()[module]
        if name in own:
            return module, own[name]
        if name in imported:
            module, name = imported[name]
            continue
        return None
    return None


def _result_positions(value):
    """The expressions a ``return`` hands back as they are.

    The value itself; either branch of a conditional expression; any operand of
    an ``or`` / ``and``.  Anything else -- an index, an attribute, an
    arithmetic result -- is a new value, not the callee's.
    """
    if isinstance(value, ast.IfExp):
        return _result_positions(value.body) + _result_positions(value.orelse)
    if isinstance(value, ast.BoolOp):
        return [x for operand in value.values for x in _result_positions(operand)]
    return [value]


def _is_private_call(value, privates):
    """Whether ``value`` is a bare-name call of a private function the module can see."""
    return (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
            and value.func.id in privates)


def _privates_handed_back(node, privates):
    """The private functions whose result a body returns as it is.

    Two idioms hand a callee's verdict back unchanged, and only they count:
    ``return _f(...)``, and ``x = _f(...)`` with a ``return x`` somewhere in the
    body -- ``gate = _lattice_not_descended(group)`` followed by
    ``if gate is not None: return gate`` is the pattern.  A result that is
    unpacked, indexed, iterated, raised on or replaced never reaches the
    caller's reader as a refusal (``_maximal_basis(...)[1]``; ``_stable_pairs``
    answering ``()`` on it), so its gates are not this function's gates.
    ``privates`` is every private name the module can call by its bare name:
    its own functions and the ones it imports.
    """
    handed, bound, returned = set(), {}, set()
    for child in ast.walk(node):
        if isinstance(child, ast.Return) and child.value is not None:
            for value in _result_positions(child.value):
                if _is_private_call(value, privates):
                    handed.add(value.func.id)
                elif isinstance(value, ast.Name):
                    returned.add(value.id)
        elif (isinstance(child, ast.Assign) and len(child.targets) == 1
              and isinstance(child.targets[0], ast.Name)
              and _is_private_call(child.value, privates)):
            bound.setdefault(child.targets[0].id, set()).add(child.value.func.id)
        elif isinstance(child, ast.NamedExpr) and _is_private_call(child.value, privates):
            bound.setdefault(child.target.id, set()).add(child.value.func.id)
    for name in returned:
        handed |= bound.get(name, set())
    return handed


def _visible_privates(module):
    """Every private name ``module`` can call by its bare name: its own functions and the ones it imports."""
    _, _, own, imported = _modules()[module]
    return set(own) | set(imported)


def _reachable_gates(node, module):
    """Every gate a body in ``module`` can return: its own literals, and those of
    every private function whose result it hands back, in its own module or in
    the module it imports the function from.

    A gate reaches a reader through whichever public name returns it, and a
    public name that hands a private function's verdict straight back returns
    that gate just as much as if the literal stood in its own body --
    ``lattice_not_descended`` through ``_lattice_not_descended`` is the
    pattern.  Only a result handed back as it is counts
    (:func:`_privates_handed_back`): a private function whose refusal the
    caller unpacks, raises on or replaces cannot put that verdict in front of
    the caller's reader, so its literals are not collected.  The walk applies
    the same rule at every step, each callee read in the module that defines
    it, with a visited set, so a chain of hand-backs is measured and a chain
    broken anywhere stops there.
    """
    gates = set(_gates_in(node))
    seen = set()
    todo = [(module, name) for name in sorted(_privates_handed_back(node, _visible_privates(module)))]
    while todo:
        target = _resolve(*todo.pop())
        if target is None or target in seen:
            continue
        seen.add(target)
        defining, body = target
        gates |= _gates_in(body)
        todo.extend((defining, name)
                    for name in sorted(_privates_handed_back(body, _visible_privates(defining))))
    return gates


def _public_names(tree, exported):
    """Every exported function and public method of an exported class, by name."""
    out = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in exported:
                out.append((node.name, node))
        elif isinstance(node, ast.ClassDef) and node.name in exported:
            for member in node.body:
                if not isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if member.name.startswith("_"):
                    continue
                out.append((f"{node.name}.{member.name}", member))
    return out


#: What the walk reaches and no public call can serve: a
#: `(module, public name, gate)` triple where one call is the exception, or a
#: bare gate name where no public call anywhere can produce it.
#:
#: `realized_group` hands `_group_of`'s refusal straight back, so the walk
#: collects every gate the private identification can return; it passes it no
#: `magmom`, so the group found is always magnetic type I or II and that one
#: gate cannot fire.  The others the walk finds on the way ARE served and
#: `realized_group` names them -- measured on this tree: a model periodic in
#: fewer than two directions, in a cell of any dimension, gives
#: `fewer_than_two_periodic_directions` before any cell is identified,
#: `spinor=False` on a spinful 3D model gives `single_valued_spinful`, and
#: `irrep_tables` fires wherever `irreptables` is absent.
#:
#: `lattice_not_descended` is the gate a *supercell's* group trips, and only a
#: `GroupData` with `_lattice_index > 1` trips it.  `_group_of` is the sole
#: producer of one, and it is private: `realized_group` either refuses
#: `supercell` (the Hamiltonian obeys the pure translation) or makes the
#: lattice descent its first step, and `tests/test_lattice_descent.py` asserts
#: `r.group.lattice_index == 1` afterwards.  So the public names whose walk
#: reaches it can never serve it, and it is named for the reader in
#: `docs/guide/group.md` as the `supercell` refusal of `realized_group`, which
#: is the verdict a user can actually meet.
_UNSERVABLE = frozenset({
    ("src/pythtqc/_realized.py", "realized_group", "magnetic_type_out_of_scope"),
    "lattice_not_descended",
})


def _public_gates():
    """Every gate a public name can return: (module, line, name, gate, its docstring).

    The gates are the ones :func:`_reachable_gates` finds, less what
    ``_UNSERVABLE`` names: what the walk reaches and no public call can serve.
    """
    for module, (path, tree, own, imported) in _modules().items():
        for name, node in _public_names(tree, _exported(tree)):
            doc = ast.get_docstring(node, clean=False) or ""
            for gate in sorted(_reachable_gates(node, module)):
                if gate in _UNSERVABLE or (_relative(path), name, gate) in _UNSERVABLE:
                    continue
                yield _relative(path), node.lineno, name, gate, doc


def unnamed_gates():
    """Every (module, name, gate) a public name can return and does not name.

    A public name is an exported function or a public method of an exported
    class -- ``SpaceGroupRep.sewing_operator`` refuses, and a reader calling it
    meets its gate in its own docstring, not in the module's.  The gates counted
    are the ones :func:`_reachable_gates` finds: the literals in the body, and
    the literals of every private function whose result the body hands back as
    it is, wherever in the package that function is defined.  ``_UNSERVABLE``
    names what the walk reaches
    and no public call can serve, by triple or by bare gate name.
    """
    return [(path, lineno, name, gate)
            for path, lineno, name, gate, doc in _public_gates() if gate not in doc]


def returned_gates():
    """Every gate some public name can return, by name, in sorted order.

    The same walk :func:`unnamed_gates` reads the docstrings against, so the
    refusal reference (``tests/test_docs.py``) is held to the list the
    docstrings are.
    """
    return sorted({gate for _, _, _, gate, _ in _public_gates()})


# ---------------------------------------------------------------------------
# rule 5 -- every number a judgement compares against, named
# ---------------------------------------------------------------------------

#: The one module that names the numbers the package's judgements compare
#: against, each with what it was measured against.
_TOLERANCE_MODULE = "_tolerance.py"

#: A float literal smaller than this is a tolerance: a number a measured or
#: computed value is judged against.
_SMALL = 1e-2


def small_literals():
    """Every float literal ``0 < x < 1e-2`` in the code of ``src/pythtqc``, outside the tolerance module.

    Read off the token stream, so a number in a docstring or a comment is text
    and is not counted.  The tolerance module names each such number, and is
    not counted either.
    """
    found = []
    for path in _source_files(_SRC, _SRC_NAME):
        if os.path.basename(path) == _TOLERANCE_MODULE:
            continue
        source = _read(path)
        lines = source.splitlines()
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type != tokenize.NUMBER:
                continue
            text = token.string.replace("_", "").lower()
            if text.endswith("j") or text.startswith(("0x", "0o", "0b")):
                continue
            if "." not in text and "e" not in text:
                continue
            if 0.0 < float(text) < _SMALL:
                lineno = token.start[0]
                found.append(Hit(_relative(path), lineno, "a small literal", token.string,
                                 lines[lineno - 1].strip()))
    return found


# ---------------------------------------------------------------------------
# the tests
# ---------------------------------------------------------------------------

_WORD_KEYS = [key for key in sorted(BUDGETS) if key[0] in _RULES]


def test_register():
    """Every rule on every class: each count is its budget, neither over nor under.

    Two preconditions come first, so that a green run is never an empty scan and
    never a table with a count missing: each class resolves to real lines, and
    the table carries one number per rule per class and no number more.  Then
    every count is measured -- the words of every class, and the small numbers
    standing in the code -- and every drift -- a count over its budget, with the
    offending lines, or under it, with the number the budget should now be -- is
    reported together in the one failure.
    """
    for text_class in CLASSES:
        assert len(text_of(text_class)) > 40, text_class
    expected = {(rule, cls) for rule in WORD_RULES for cls in CLASSES}
    expected.add(("gates-named", "src docstrings"))
    expected.add(("small-literals", "src code"))
    assert set(BUDGETS) == expected, sorted(set(BUDGETS) ^ expected)

    measured = [((rule, text_class), hits(rule, text_class))
                for rule, text_class in _WORD_KEYS]
    measured.append((("small-literals", "src code"), small_literals()))
    drifts = []
    for (rule, text_class), found in measured:
        budget = BUDGETS[(rule, text_class)]
        if len(found) > budget:
            drifts.append(f"{rule} on {text_class}: {len(found)} hits against a budget "
                          f"of {budget}\n" + _listing(found))
        elif len(found) < budget:
            drifts.append(f"lower the budget for ({rule!r}, {text_class!r}) to {len(found)}")
    assert not drifts, "\n".join(drifts)


def test_gates_named():
    """Every gate a public function or method returns is named in its docstring."""
    found = unnamed_gates()
    budget = BUDGETS[("gates-named", "src docstrings")]
    listing = "\n".join(
        f"  {path}:{lineno}: {name} does not name {gate!r}"
        for path, lineno, name, gate in found
    )
    assert len(found) <= budget, (
        f"gates-named: {len(found)} gates unnamed against a budget of {budget}\n" + listing
    )
    assert len(found) >= budget, (
        f"lower the budget for ('gates-named', 'src docstrings') to {len(found)}"
    )
