#!/usr/bin/env python3
"""Write ``docs/reference/functions.md`` from the docstrings of the functions ``pythtqc`` exports.

The page is the docstrings, rendered: for each function in ``pythtqc.__all__``,
its signature, its summary paragraph, and its *Parameters*, *Returns* and
*Refusals* sections, grouped by task as :data:`GROUPS` lists them.  It is
written by this script and never by hand, so the docstrings stay the one
source, and ``tests/test_docs.py`` fails when the page in the tree is not what
the docstrings make it now::

    python scripts/build_reference.py

The docstrings are numpydoc in reStructuredText; the page is Markdown as GitHub
renders it.

* A Sphinx role naming one of the exported functions,
  ``:func:`~.groups.realized_group```, becomes a link to that function's
  entry; any other role becomes code font, a ``~`` keeping its last component
  only, and a module's name dropped where what follows is exported.
* ``:math:`` becomes GitHub's inline mathematics, ``$`...`$``, and a
  ``.. math::`` block a ``math`` fence.
* A numpydoc heading becomes a bold label, and each entry under it a list
  item: the name and the type in code font, then the description.
* A double-backtick literal becomes code font, and an escaped space nothing.

A signature is the one ``help`` prints, less its annotations, which the
*Parameters* entries carry, and less any parameter whose name starts with an
underscore, which is not part of the call.  An exported function that no group
holds, or a group naming a function the package does not export, stops the
script with the names: the page lists every function or it is not written.
"""
from __future__ import annotations

import inspect
import pathlib
import re
import sys
import textwrap

ROOT = pathlib.Path(__file__).resolve().parent.parent
PAGE = ROOT / "docs" / "reference" / "functions.md"

#: The groups, in page order: each a title, what its functions are for, and the
#: functions, as the task index of ``docs/README.md`` groups them.
GROUPS = (
    ("The group",
     "the space group a Hamiltonian has and which operations it obeys, the "
     "orbitals' site orbits, and the operations by index and Seitz symbol",
     ("realized_group", "operations", "seitz", "operation_index")),
    ("Declaring what the hoppings cannot supply",
     "the spin a spinless model carries in its orbital index",
     ("spin_basis", "spin_operator", "spinless_presentation")),
    ("Group data in this cell's coordinates",
     "the tables' k-points, little groups, character tables, Wyckoff positions "
     "and elementary band representations, in the model's own cell",
     ("high_symmetry_points", "little_group", "corep_table", "wyckoff_sites_tables",
      "k_grid", "ebr_data", "tables_status")),
    ("Labelling the bands",
     "the states at one k-point, and the sewing matrices, characters and corep "
     "content of a set of bands, at one high-symmetry point or at all of them",
     ("solve_at", "sewing_matrix", "characters", "corep_content", "hsp_table",
      "format_counts", "multiplets")),
    ("Atomic limits",
     "whether a symmetry vector is a sum of elementary band representations, "
     "and the counting beside it",
     ("decompose", "subtract", "completeness_check", "regular_completion",
      "ebr_rows", "in_plane_restriction")),
    ("Stable invariants",
     "the symmetry-indicator class of a band set, and its stable real-space "
     "invariants, from the bands or from the orbitals",
     ("symmetry_indicator", "stack_presentation", "srsi_tables", "srsi_of",
      "srsi_of_columns", "srsi_of_bands", "srsi_of_model", "splitting",
      "nsit_criterion", "atomic_realizations", "stably_equivalent_pairs")),
    ("Numerical and `wannier90` models",
     "a model symmetric only to its own precision: its orbital centres moved onto "
     "the nearest positions the group carries, its hoppings onto the group's own "
     "average, and each operation classified",
     ("symmetrize_positions", "symmetrize", "classify_operations")),
)

#: the docstring sections the page carries, after the summary paragraph
SECTIONS = ("Parameters", "Returns", "Refusals")

#: the widest a signature runs on one line before it wraps
WIDTH = 88

INTRODUCTION = """\
# Functions

Every function `pythtqc` exports, by task: its signature, what it is for, the
arguments it takes, what it returns, and the checks at which it refuses. The
page is written from the docstrings by `scripts/build_reference.py` and is not
edited by hand. A docstring holds more than its entry here, notes and examples
among it, and `help(tqc.<name>)` prints the whole of one, with
`import pythtqc as tqc`: `help(tqc.realized_group)`, for one.

A call that cannot complete returns a `Refusal` in place of its record: a
falsy record that names the check that stopped it and what was measured there.
Each entry's **Refusals** are the checks at which that call can refuse.

The functions fall into seven groups:
"""


# ---------------------------------------------------------------------------
# the package, read
# ---------------------------------------------------------------------------


def _package():
    """``pythtqc`` from the checkout's own ``src``, whatever is installed."""
    src = str(ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    import pythtqc

    where = pathlib.Path(pythtqc.__file__).resolve().parent
    if where != (ROOT / "src" / "pythtqc").resolve():
        raise RuntimeError(f"pythtqc is imported from {where}, not from this checkout")
    return pythtqc


class _Names:
    """What a role's target can name: the functions, every exported name, the modules."""

    def __init__(self, package):
        exported = [getattr(package, name) for name in package.__all__]
        self.functions = {name for name, obj in zip(package.__all__, exported)
                          if inspect.isfunction(obj)}
        self.modules = {name for name, obj in zip(package.__all__, exported)
                        if inspect.ismodule(obj)}
        self.exported = set(package.__all__)

    def short(self, dotted):
        """A dotted name as the page writes it: a module dropped where the rest is exported."""
        head, _, rest = dotted.partition(".")
        if rest and head in self.modules and rest.split(".")[0] in self.exported:
            return rest
        return dotted


# ---------------------------------------------------------------------------
# a docstring, taken apart
# ---------------------------------------------------------------------------

_UNDERLINE = re.compile(r"^\s*-{3,}\s*$")
_HEADER = re.compile(r"^\s*([A-Z][A-Za-z]*(?: [A-Za-z]+){0,2})\s*$")


def _parts(function):
    """A docstring's summary paragraph, and its sections by name."""
    lines = inspect.getdoc(function).splitlines()
    end = next((i for i, line in enumerate(lines) if not line.strip()), len(lines))
    sections, current = {}, None
    i = 0
    while i < len(lines):
        if (i + 1 < len(lines) and _HEADER.match(lines[i])
                and _UNDERLINE.match(lines[i + 1])):
            current = _HEADER.match(lines[i]).group(1)
            sections[current] = []
            i += 2
            continue
        if current is not None:
            sections[current].append(lines[i])
        i += 1
    return lines[:end], sections


def _items(body, refusals):
    """A section's entries and its free paragraphs, in order.

    An entry is ``("entry", head, description lines)``: its head a line at the
    section's own indentation -- in *Refusals* a head runs on over lines that
    end in ``,`` or ``/`` -- and its description the indented lines under it.
    In *Refusals* a line at the section's indentation with nothing indented
    under it is a sentence, and the sentences run into ``("text", lines)``.
    """
    items, sentence = [], False
    i = 0
    while i < len(body):
        line = body[i]
        if not line.strip():
            if items and items[-1][0] == "entry":
                items[-1][2].append("")
            sentence = False
        elif line[:1].isspace():
            if not items or items[-1][0] != "entry":
                raise RuntimeError(f"an indented line under no entry: {line.strip()!r}")
            items[-1][2].append(line)
        else:
            head = [line]
            while (refusals and head[-1].rstrip().endswith((",", "/")) and i + 1 < len(body)
                   and body[i + 1].strip() and not body[i + 1][:1].isspace()):
                i += 1
                head.append(body[i])
            following = body[i + 1] if i + 1 < len(body) else ""
            if not refusals or (following.strip() and following[:1].isspace()):
                items.append(("entry", " ".join(part.strip() for part in head), []))
                sentence = False
            elif sentence:
                items[-1][1].extend(head)
            else:
                items.append(("text", head))
                sentence = True
        i += 1
    return items


# ---------------------------------------------------------------------------
# reStructuredText, as Markdown
# ---------------------------------------------------------------------------

_INLINE = re.compile(
    r":(?P<role>[a-z]+(?::[a-z]+)?):`(?P<target>[^`]*)`"
    r"|``(?P<literal>.+?)``"
    r"|`(?P<interpreted>[^`]+)`"
    r"|\\ ",
    re.S,
)

# a prose line Markdown would read as the start of a block: a list item, a
# heading, a quotation, a fence, a table row, a heading's underline
_BLOCK_START = re.compile(r"^(?:[-+*](?=\s|$)|#|>|\d+[.)](?=\s|$)|[=-]+\s*$|`{3}|~{3}|\|)")


def _code(text):
    """Code font for any text, on one line."""
    text = " ".join(text.split("\n"))
    fence = "``" if "`" in text else "`"
    pad = " " if "`" in text else ""
    return f"{fence}{pad}{text}{pad}{fence}"


def _role(role, target, names):
    """One role as a segment: a link to an entry, mathematics, or code."""
    if role == "math":
        return "math", target
    shown = target.lstrip("~").lstrip(".")
    if target.startswith("~"):
        shown = shown.rsplit(".", 1)[-1]
    shown = names.short(shown)
    if role == "func" and shown in names.functions:
        return "link", shown
    return "code", shown


def _inline(text, names):
    """One paragraph's inline markup, reStructuredText to Markdown."""
    segments, last = [], 0
    for m in _INLINE.finditer(text):
        segments.append(("text", text[last:m.start()]))
        if m["role"]:
            segments.append(_role(m["role"], m["target"], names))
        elif m["literal"] is not None:
            segments.append(("code", m["literal"]))
        elif m["interpreted"] is not None:
            segments.append(("code", m["interpreted"]))
        last = m.end()
    segments.append(("text", text[last:]))
    # code side by side is one span: two written so would read as a double backtick
    merged = []
    for kind, value in segments:
        if kind == "text" and not value:
            continue
        if kind == "code" and merged and merged[-1][0] == "code":
            merged[-1] = ("code", merged[-1][1] + value)
        else:
            merged.append((kind, value))
    out = []
    for kind, value in merged:
        if kind == "text":
            out.append(value.replace("<", "&lt;"))
        elif kind == "code":
            out.append(_code(value))
        elif kind == "link":
            out.append(f"[{_code(value)}](#{value})")
        else:
            out.append("$" + _code(" ".join(value.split())) + "$")
    return "".join(out)


def _escaped(line):
    """A prose line, with a start Markdown would read as a block's escaped."""
    if not _BLOCK_START.match(line):
        return line
    if line[0].isdigit():
        return re.sub(r"^(\d+)([.)])", r"\1\\\2", line)
    return "\\" + line


def _paragraph(lines, names):
    text = _inline("\n".join(line.strip() for line in lines), names)
    return [_escaped(line) for line in text.split("\n")]


def _bullets(lines, names):
    """A bullet list, each item's lines under its first."""
    items = []
    for line in lines:
        if re.match(r"[*+-] ", line):
            items.append([line[2:]])
        else:
            items[-1].append(line.strip())
    out = []
    for item in items:
        text = _inline("\n".join(item), names).split("\n")
        out.append("* " + text[0])
        out += ["  " + _escaped(line) for line in text[1:]]
    return out


def _blocks(lines, names):
    """A description as Markdown blocks: paragraphs, lists, mathematics, literal text."""
    text = textwrap.dedent("\n".join(lines)).strip("\n")
    chunks = re.split(r"\n[ \t]*\n", text) if text else []
    out = []
    i = 0
    while i < len(chunks):
        chunk = chunks[i].split("\n")
        if chunk[0].startswith(".. math::"):
            formula = [chunk[0][len(".. math::"):]] + chunk[1:]
            if not "".join(formula).strip() and i + 1 < len(chunks):
                i += 1
                formula = chunks[i].split("\n")
            out.append(["```math"] + [line.strip() for line in formula if line.strip()]
                       + ["```"])
        elif chunk[0][:1].isspace():
            if out and out[-1] and out[-1][-1].endswith("::"):
                out[-1][-1] = out[-1][-1][:-1]
            out.append(["```text"] + textwrap.dedent("\n".join(chunk)).split("\n") + ["```"])
        elif re.match(r"[*+-] ", chunk[0]):
            out.append(_bullets(chunk, names))
        else:
            out.append(_paragraph(chunk, names))
        i += 1
    return out


def _types(spec, names):
    """A type specification as the page writes it, the module names dropped."""
    return re.sub(r"\.{0,2}\b([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+)",
                  lambda m: names.short(m[1]), spec.strip())


def _head(section, head, names):
    """An entry's head: the names and type of a parameter, a return's type, a gate."""
    if section == "Refusals":
        return "**" + _inline(head, names) + "**"
    label, colon, spec = head.partition(" : ")
    if section == "Returns" and not colon:
        return _code(_types(head, names))
    shown = "**" + ", ".join(_code(name.strip()) for name in label.split(",")) + "**"
    return shown + (f" ({_code(_types(spec, names))})" if spec.strip() else "")


def _section(section, body, names):
    """One numpydoc section as Markdown lines, under its bold label."""
    out = [f"**{section}**", ""]
    for item in _items(body, refusals=section == "Refusals"):
        if item[0] == "text":
            out += _paragraph(item[1], names) + [""]
            continue
        _, head, description = item
        first = _head(section, head, names)
        blocks = _blocks(description, names)
        if blocks and not blocks[0][0].startswith(("```", "* ")):
            out.append(f"- {first}: {blocks[0][0]}")
            out += ["  " + line for line in blocks[0][1:]]
            blocks = blocks[1:]
        else:
            out.append(f"- {first}")
        for block in blocks:
            out.append("")
            out += ["  " + line if line else "" for line in block]
        out.append("")
    return out


# ---------------------------------------------------------------------------
# the page
# ---------------------------------------------------------------------------


def _signature(name, function):
    """The signature ``help`` prints, less annotations and private parameters, wrapped."""
    parts, star = [], False
    params = [p for p in inspect.signature(function).parameters.values()
              if not p.name.startswith("_")]
    for k, param in enumerate(params):
        if param.kind is param.KEYWORD_ONLY and not star:
            parts.append("*")
        star |= param.kind in (param.KEYWORD_ONLY, param.VAR_POSITIONAL)
        parts.append(str(param.replace(annotation=inspect.Parameter.empty)))
        if (param.kind is param.POSITIONAL_ONLY
                and (k + 1 == len(params) or params[k + 1].kind is not param.POSITIONAL_ONLY)):
            parts.append("/")
    flat = f"{name}({', '.join(parts)})"
    if len(flat) <= WIDTH:
        return [flat]
    lines, current = [f"{name}("], ""
    for k, part in enumerate(parts):
        piece = part + ("," if k + 1 < len(parts) else "")
        if current and 4 + len(current) + 1 + len(piece) > WIDTH:
            lines.append("    " + current)
            current = ""
        current += (" " if current else "") + piece
    return lines + ["    " + current, ")"]


def render():
    """The page, as the docstrings make it now."""
    package = _package()
    names = _Names(package)
    grouped = [name for _, _, members in GROUPS for name in members]
    missing = sorted(names.functions - set(grouped))
    unknown = sorted(set(grouped) - names.functions)
    twice = sorted({name for name in grouped if grouped.count(name) > 1})
    if missing or unknown or twice:
        raise RuntimeError(
            "GROUPS does not hold every exported function once:"
            + (f"\n  in no group: {', '.join(missing)}" if missing else "")
            + (f"\n  not exported: {', '.join(unknown)}" if unknown else "")
            + (f"\n  in two groups: {', '.join(twice)}" if twice else ""))

    out = INTRODUCTION.split("\n")
    for title, holds, _ in GROUPS:
        out.append(f"- [{title}](#{_slug(title)}) — {holds}.")
    for title, _, members in GROUPS:
        out += ["", f"## {title}", ""]
        for name in members:
            summary, _ = _parts(getattr(package, name))
            out.append(f"- [{_code(name)}](#{name}) — "
                       + " ".join(_paragraph(summary, names)))
        for name in members:
            function = getattr(package, name)
            summary, sections = _parts(function)
            out += ["", f"### {_code(name)}", "", "```text"]
            out += _signature(name, function)
            out += ["```", ""] + _paragraph(summary, names) + [""]
            for section in SECTIONS:
                body = sections.get(section)
                if body and any(line.strip() for line in body):
                    out += _section(section, body, names)
            while out[-1] == "":
                out.pop()
    return "\n".join(out) + "\n"


def _slug(title):
    """The anchor GitHub gives a group's heading.

    The title in lower case, its code font and punctuation dropped and its
    spaces hyphens; the pages' link check holds every such anchor to GitHub's
    rule (``scripts/check_page.py``).
    """
    kept = (c for c in title.lower() if c.isalnum() or c in "-_ ")
    return "".join(kept).replace(" ", "-")


def main():
    try:
        text = render()
    except RuntimeError as error:
        print(error)
        return 1
    before = PAGE.read_bytes().decode("utf-8").replace("\r\n", "\n") if PAGE.exists() else None
    if text != before:
        PAGE.parent.mkdir(parents=True, exist_ok=True)
        PAGE.write_text(text, encoding="utf-8", newline="\n")
    count = sum(len(members) for _, _, members in GROUPS)
    state = "unchanged" if text == before else ("written" if before is None else "rewritten")
    print(f"{PAGE.relative_to(ROOT).as_posix()}: {count} functions in {len(GROUPS)} groups, "
          f"{text.count(chr(10))} lines, {state}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
