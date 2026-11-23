#!/usr/bin/env python3
"""Write the two reference pages from the docstrings of what ``pythtqc`` exports.

``docs/reference/functions.md`` is the functions' docstrings, rendered: for each
function in ``pythtqc.__all__``, its signature, its summary line, its lead
paragraphs (what it computes), and its *Parameters*, *Returns* and *Refusals*
sections, grouped by task as :data:`GROUPS` lists them.

``docs/reference/records.md`` is the records' docstrings, rendered: for each
class a public module's ``__all__`` lists, its summary line, its lead, the
calls and methods that return it (**Returned by**) and the fields that hold it
(**Held in**), the modules that export it, its *Attributes* and its public
methods, grouped as :data:`RECORD_GROUPS` lists them, each record held in a
field of another one heading level under it (no deeper than a sixth-level
heading).  What returns a record is read off the return annotations and the
*Returns* of the public functions and methods; what holds it, off the
annotations and the *Attributes* types of the records' fields;
:data:`HELD_BY_HAND`, :data:`HELD_IN_EVERY` and :data:`RETURNED_BY_HAND` say it
where none of those does, and the script stops where one of their lines names
a record or a field the package no longer has.

Both are written by this script and never by hand, so the docstrings stay the
one source, and ``tests/test_docs.py`` fails when a page in the tree is not
what the docstrings make it now::

    python scripts/build_reference.py

The docstrings are numpydoc in reStructuredText; the page is Markdown as GitHub
renders it.

* A Sphinx role naming one of the exported functions,
  ``:func:`~.groups.realized_group```, becomes a link to that function's
  entry; on the records' page a role naming a record, or a field or method of
  one, becomes a link to the record's entry.  Any other role becomes code
  font, a ``~`` keeping its last component only, and a module's name dropped
  where what follows is exported.  A role written against the private module
  that defines an exported name names it as the package exports it.
* ``:math:`` becomes GitHub's inline mathematics, ``$`...`$``, and a
  ``.. math::`` block a ``math`` fence.
* A numpydoc heading becomes a bold label, and each entry under it a list
  item: the name and the type in code font, then the description.
* A double-backtick literal becomes code font, and an escaped space nothing.

A signature is the one ``help`` prints, less its annotations, which the
*Parameters* entries carry, and less any parameter whose name starts with an
underscore, which is not part of the call.  An exported function that no group
holds, or a group naming a function the package does not export, stops the
script with the names: the page lists every function or it is not written.  So
does an exported record no group holds, one placed twice or under a record that
holds none of it, a field with no *Attributes* entry, a record nothing returns,
and two headings with one anchor.  A heading's anchor is the one the site gives
it, by the rule ``mkdocs.yml`` sets (:func:`_site_slug`).
"""
from __future__ import annotations

import dataclasses
import inspect
import pathlib
import re
import sys
import textwrap
import unicodedata

ROOT = pathlib.Path(__file__).resolve().parent.parent
PAGE = ROOT / "docs" / "reference" / "functions.md"

#: The groups, in page order: each a title, what its functions are for, and the
#: functions, as the task index of ``docs/README.md`` groups them.
GROUPS = (
    ("The group",
     "the space group a Hamiltonian has and which operations it obeys, the "
     "orbitals' Wyckoff orbits, and the operations by index and Seitz symbol",
     ("realized_group", "operations", "operation_index")),
    ("Declaring what the hoppings cannot supply",
     "the spin a spinless model carries in its orbital index",
     ("spin_basis", "spin_operator", "spinless_presentation")),
    ("Group data in this cell's coordinates",
     "the tables' k-points, little groups, character tables, Wyckoff positions "
     "and elementary band representations, in the model's own cell",
     ("high_symmetry_points", "little_group", "corep_table", "wyckoff_sites_tables",
      "k_grid", "ebr_data", "tables_status")),
    ("Labeling the bands",
     "the states at one k-point, and the sewing matrices, characters and coreps "
     "of a set of bands, at one high-symmetry point or at all of them",
     ("solve_at", "sewing_matrix", "characters", "coreps_at", "hsp_table",
      "format_coreps", "multiplets")),
    ("Atomic limits",
     "whether a symmetry data vector is a sum of elementary band representations, "
     "and the counting beside it",
     ("decompose", "subtract", "regular_completion",
      "ebr_rows", "in_plane_restriction")),
    ("Stable invariants",
     "the symmetry-indicator class of a band set, and its stable real-space "
     "invariants, from the bands or from the orbitals",
     ("symmetry_indicator", "stack_presentation", "srsi_tables", "srsi_of_bands",
      "srsi_of_model", "splitting", "nsit_criterion", "atomic_realizations")),
)

#: the docstring sections the page carries, after the summary paragraph
SECTIONS = ("Parameters", "Returns", "Refusals")

#: the widest a signature runs on one line before it wraps
WIDTH = 88

INTRODUCTION = """\
# Functions

Every function `pythtqc` exports, by task: its signature, what it is for, what
it computes, the arguments it takes, what it returns, and the checks at which it
refuses. The page is written from the docstrings by `scripts/build_reference.py`
and is not edited by hand. A docstring holds more than its entry here, its
examples among it, and `help(tqc.<name>)` prints the whole of one, with
`import pythtqc as tqc`: `help(tqc.realized_group)`, for one.

A call that cannot complete returns a `Refusal` in place of its record: a
falsy record that names the check that stopped it and what was measured there.
Each entry's **Refusals** are the checks at which that call can refuse.

The functions fall into six groups:
"""

RECORDS_PAGE = ROOT / "docs" / "reference" / "records.md"

#: the modules whose ``__all__`` lists the records a user meets
RECORD_MODULES = ("pythtqc", "pythtqc.groups", "pythtqc.srsi", "pythtqc.ebr",
                  "pythtqc.sewing", "pythtqc.spin", "pythtqc.symmetry",
                  "pythtqc.symmetrization", "pythtqc.refusal")

#: The records' groups, in page order: each a title, a line for the page's
#: index and the sentences that open the group, and its records, one a line,
#: each record held in a field of another indented under it.  The records under
#: one record stand in the order of its *Attributes*.
RECORD_GROUPS = (
    ("The group",
     ("what `realized_group` returns and the records it holds, the site "
      "multiplets it can take, and an operation's matrix in its canonical phase.",
      """\
`realized_group` returns a `RealizedGroup`. It holds the space group the
Hamiltonian obeys, how each operation of the cell was measured, the atomic-limit
table of the orbitals, and the representation of the group on them. Each record
it holds stands under it. After it stand the site multiplets `realized_group`
can take, and the records of `symmetry.normalize_rep`."""),
     """\
RealizedGroup
  GroupData
  OperationTable
    Operation
  AtomicLimitTable
    WyckoffOrbit
      Term
      BandRepresentation
      WyckoffOrbitNumerical
    AtomicLimitTableNumerical
      DeclarationCheck
      AccidentalDegeneracy
  Descent
    NamedGroup
    DroppedOperation
    DescentCell
    DescentStep
      StepDeclaration
      StepCenters
      StepTie
      StepProjective
        ProjectivePair
        ProjectiveMirror
        ProjectiveReading
    DescentLattice
      LatticeTranslation
      TranslationDressing
      RefutedTranslation
  TimeReversal
    TimeReversalSquare
  BeyondGrey
    BeyondGreyOperation
  Conserved
    ConservedSector
  LayerEnvironment
    CrossCheck
      CrossCheckRow
      CrossCheckOperation
      CrossCheckRefusal
    Presentation
      PresentationTilt
  Decoration
    DeclaredReduction
      ExcludedOperation
  SpaceGroupRep
    SpaceGroupRepNumerical
  SpinBasis
  RealizedGroupNumerical
    Precision
    Positions
    Symmetrization
      SymmetrizationChecks
      SymmetrizationVerification
      SymmetrizationSiteMultiplets
      SymmetrizationParts
      SymmetrizationCopies
      SymmetrizationShares
    ValidatedGroup
      Discrepancy
SiteMultiplets
NormalizedRep
RepAnchor
"""),
    ("Group data in this cell's coordinates",
     ("the tables' k-points, little groups, Wyckoff positions, k-point meshes and "
      "elementary band representations, in the model's own cell.",
      """\
The records of the tables of a space group, written in the cell of the model's
pythtb Lattice: its high-symmetry points, the little group of a k-point, the
Wyckoff position of a site, a mesh of k-points, its elementary band
representations, and the little-group representations read off measured
characters."""),
     """\
HighSymmetryPoints
LittleGroup
SiteReference
KGrid
EBRData
  EBRConvention
CorepIdentification
"""),
    ("Labeling the bands",
     ("the sewing matrices, characters and little-group representations of a set "
      "of bands, at one high-symmetry point or at all of them.",
      """\
The records of the calls that label a set of bands: the sewing matrix of one
operation, the characters and the little-group representations of the bands at
one k-point, and their symmetry data vector over every high-symmetry point."""),
     """\
SewingMatrix
  SewingMatrixNumerical
BandCharacters
  MultipletCharacters
  BandCharactersNumerical
    Thresholds
CorepsAt
  Multiplet
  CorepsAtNumerical
HSPTable
"""),
    ("Atomic limits",
     ("the sums of elementary band representations that carry a symmetry data "
      "vector, and the completion to copies of the regular band representation.",
      """\
The records of the calls that compare a symmetry data vector with the
elementary band representations of the group: the sums of them that carry it,
and the bands that complete a band set to copies of the regular band
representation."""),
     """\
EBRDecomposition
  SignedCombinations
  SiteContent
  EBRDecompositionNumerical
    Enumeration
    Completeness
    Restriction
RegularCompletion
"""),
    ("Stable invariants",
     ("the symmetry-indicator class of a band set, and the stable real-space "
      "invariants of a group, a band set and a model.",
      """\
The records of the symmetry-indicator class of a band set and of the stable
real-space invariants (SRSIs): the tables of one group, the integer-valued
SRSIs (ZSRSIs) of a band set, the SRSIs of a model, the no-go test of a gapped
splitting, and a pair of site irreps that no stable real-space invariant tells
apart."""),
     """\
SymmetryIndicator
  SymmetryIndicatorNumerical
    SmithVerification
    LatticeCheck
    StackCompletion
      StackInput
      StackGroup
      StackChecks
        HamiltonianIdentity
        StackHoppings
        RepresentationCheck
        CharactersCheck
        StackFiber
    TwoDimensionalReading
      IndicatorComponent
SRSITables
  SiteIrrep
  SplitEBR
  Provenance
    ProvenanceCounts
  SRSITablesNumerical
    IndicatorCheck
  Invariants
    ModularValue
BandSRSIs
  BandSRSIsNumerical
ModelSRSIs
  WyckoffOrbitSRSIs
    WyckoffOrbitSRSIsNumerical
      Localization
Splitting
  Realizations
    AtomicRealization
    RealizationsNumerical
StablePair
"""),
    ("Refusals",
     ("what a call returns where it cannot complete, the exception that carries "
      "it, the error raised for a defect, and the one warning.",
      """\
What a call returns in place of its record where it cannot complete, the
exception that carries it, the error raised where the package disagrees with
itself, and the one warning the package issues."""),
     """\
Refusal
RefusalError
InternalError
MagneticScopeWarning
"""),
)

#: Records held where no annotation or type says so: (owner, as written).
HELD_BY_HAND = {
    "Operation": (("OperationTable", "OperationTable[index]"),),
    "WyckoffOrbit": (("AtomicLimitTable", "AtomicLimitTable[j]"),),
    "RefutedTranslation": (("DescentLattice", "DescentLattice.refuted"),),
    "IndicatorCheck": (("SRSITablesNumerical", "SRSITablesNumerical.indicators"),),
    "WyckoffOrbitSRSIsNumerical": (("WyckoffOrbitSRSIs", "WyckoffOrbitSRSIs.numerical"),),
}

#: Records that are the values of a field, wherever a record has that field.
HELD_IN_EVERY = {"ModularValue": "theta_Zn"}

#: Records no annotation reaches: their lines, each a label and its text as
#: reStructuredText.  A call is named by a role, so that the script can check it.
RETURNED_BY_HAND = {
    "SiteMultiplets": (("Taken by", ":func:`realized_group`, as ``site_multiplets=``"),),
    "RepAnchor": (("Taken by", ":func:`symmetry.normalize_rep`, as ``anchor=``"),),
    "Refusal": (("Returned by", "a call that cannot complete, in place of its record"),
                ("Held in", ":attr:`RefusalError.refusal`")),
    "RefusalError": (("Raised by", ":meth:`Refusal.raise_`"),),
    "InternalError": (("Raised by", "any call, where the package disagrees with itself"),),
    "MagneticScopeWarning": (("Issued by", ":func:`realized_group`, where time reversal is "
                              "measured broken"),),
}

RECORDS_INTRODUCTION = """\
# Returned records

Every record a call of `pythtqc` returns, by task: what it holds, the call that
returns it or the field that holds it, each of its fields, and its methods. A call that
cannot complete returns a [`Refusal`](#refusal) in place of its record. The page
is written from the records' docstrings by `scripts/build_reference.py` and is
not edited by hand. `help` prints the whole docstring of a record, with
`import pythtqc as tqc`: `help(tqc.RealizedGroup)`, or
`help(tqc.groups.Descent)` for a record exported by a module of the package
alone. The `summary()` of most records prints its parts, the line that opens
each part ending with the field or fields that hold it.

The arguments of each call are on the page [Functions](functions.md), what each
refusal means and what to do about it on [Refusals](refusals.md), and the words
in the [Glossary](glossary.md). A record held in a field of another follows it,
one heading level lower down to the sixth, and its line **Held in** names the
field. The records fall into six groups:
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
    """What a role's target can name: the functions, every exported name, the modules.

    ``records`` is empty on the functions' page, where a role links only to a
    function's entry on the same page.  On the records' page it holds every
    record's name, a role naming a record links to the record's entry, and a
    role naming a function links to the functions' page.
    """

    def __init__(self, package, records=()):
        exported = [getattr(package, name) for name in package.__all__]
        self.functions = {name for name, obj in zip(package.__all__, exported)
                          if inspect.isfunction(obj)}
        self.modules = {name for name, obj in zip(package.__all__, exported)
                        if inspect.ismodule(obj)}
        self.exported = set(package.__all__)
        self.records = set(records)
        #: on the records' page, each function a module of the package alone
        #: exports, by its name, as the page writes it: ``groups.decompose_ebrs``
        self.public = {}

    def short(self, dotted):
        """A dotted name as the page writes it: a module dropped where the rest is exported."""
        head, _, rest = dotted.partition(".")
        if rest and head in self.modules and rest.split(".")[0] in self.exported:
            return rest
        return dotted

    def target(self, role, shown):
        """Where a role's link goes, or ``None`` where it is code font alone."""
        if not self.records:
            return f"#{shown}" if role == "func" and shown in self.functions else None
        if role == "func" and shown in self.functions:
            return f"functions.md#{shown}"
        parts = shown.split(".")
        if role in ("class", "exc") and parts[-1] in self.records:
            return f"#{_site_slug(parts[-1])}"
        if role in ("attr", "meth") and len(parts) > 1 and parts[-2] in self.records:
            return f"#{_site_slug(parts[-2])}"
        return None


# ---------------------------------------------------------------------------
# a docstring, taken apart
# ---------------------------------------------------------------------------

_UNDERLINE = re.compile(r"^\s*-{3,}\s*$")
_HEADER = re.compile(r"^\s*([A-Z][A-Za-z]*(?: [A-Za-z]+){0,2})\s*$")


def _parts(function):
    """A docstring's summary paragraph, its lead, and its sections by name.

    The lead is the text between the summary paragraph and the first section.
    """
    lines = inspect.getdoc(function).splitlines()
    end = next((i for i, line in enumerate(lines) if not line.strip()), len(lines))
    sections, current, lead = {}, None, []
    i = end
    while i < len(lines):
        if (i + 1 < len(lines) and _HEADER.match(lines[i])
                and _UNDERLINE.match(lines[i + 1])):
            current = _HEADER.match(lines[i]).group(1)
            sections[current] = []
            i += 2
            continue
        if current is not None:
            sections[current].append(lines[i])
        else:
            lead.append(lines[i])
        i += 1
    return lines[:end], lead, sections


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
    # a role written against the module that defines an exported name, such
    # as ``.._tables.little_group``, names it as the package exports it
    last = shown.rsplit(".", 1)[-1]
    if any(part.startswith("_") for part in shown.split(".")[:-1]):
        if last in names.public:
            shown = names.public[last]
        elif last in names.functions or last in names.records:
            shown = last
    target = names.target(role, shown)
    if target:
        return "link", (shown, target)
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
            out.append(f"[{_code(value[0])}]({value[1]})")
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


def _signature(name, function, drop_first=False):
    """The signature ``help`` prints, less annotations and private parameters, wrapped.

    With ``drop_first``, a method's first parameter, ``self``, is left out too.
    """
    parts, star = [], False
    params = [p for p in inspect.signature(function).parameters.values()
              if not p.name.startswith("_")][1 if drop_first else 0:]
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
        out.append(f"- [{title}](#{_slug(title)}): {holds}.")
    for title, _, members in GROUPS:
        out += ["", f"## {title}", ""]
        for name in members:
            summary, _, _ = _parts(getattr(package, name))
            out.append(f"- [{_code(name)}](#{name}): "
                       + " ".join(_paragraph(summary, names)))
        for name in members:
            function = getattr(package, name)
            summary, lead, sections = _parts(function)
            out += ["", f"### {_code(name)}", "", "```text"]
            out += _signature(name, function)
            out += ["```", ""] + _paragraph(summary, names) + [""]
            for block in _blocks(lead, names):
                out += block + [""]
            for section in SECTIONS:
                body = sections.get(section)
                if body and any(line.strip() for line in body):
                    out += _section(section, body, names)
            while out[-1] == "":
                out.pop()
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# the records' page
# ---------------------------------------------------------------------------


def _record_classes():
    """Every class a public module exports, by name, with the modules that export it."""
    import importlib

    _package()
    found = {}
    for module_name in RECORD_MODULES:
        module = importlib.import_module(module_name)
        for name in module.__all__:
            obj = getattr(module, name)
            if inspect.isclass(obj):
                if name in found and found[name][0] is not obj:
                    raise RuntimeError(f"two classes are exported as {name}")
                found.setdefault(name, (obj, []))[1].append(module_name)
    return found


def _public_functions():
    """Every function a public module exports, as (the name the page writes, the function)."""
    import importlib

    package = _package()
    seen, out = set(), []
    for module_name in RECORD_MODULES:
        module = importlib.import_module(module_name)
        for name in module.__all__:
            obj = getattr(module, name)
            if inspect.isfunction(obj) and obj not in seen:
                seen.add(obj)
                shown = (name if name in package.__all__
                         else f"{module_name.rpartition('.')[2]}.{name}")
                out.append((shown, obj))
    return out


def _named(spec, records):
    """The records a type specification names, quoted values left out, in order."""
    spec = re.sub(r"\"[^\"]*\"|'[^']*'", "", spec)
    out = []
    for token in re.findall(r"[A-Za-z_][\w.]*", spec):
        name = token.split(".")[-1]
        if name in records and name != "Refusal" and name not in out:
            out.append(name)
    return out


def _attributes(cls):
    """A record's *Attributes* entries, as (field names, type specification, lines)."""
    _, _, sections = _parts(cls)
    out = []
    for item in _items(sections.get("Attributes", []), refusals=False):
        if item[0] != "entry":
            continue
        label, _, spec = item[1].partition(" : ")
        out.append(([name.strip() for name in label.split(",")], spec.strip(), item[2]))
    return out


def _methods(cls):
    """A record's public methods, in the order they are defined, its own and inherited.

    A property is a field, and stands among the *Attributes*; a method a
    built-in base gives, such as a mapping's ``keys``, is not the record's.
    """
    out, seen = [], set()
    for klass in reversed([k for k in cls.__mro__ if k.__module__.startswith("pythtqc")]):
        for name, value in vars(klass).items():
            if name.startswith("_") or isinstance(value, property):
                continue
            if isinstance(value, (staticmethod, classmethod)):
                value = value.__func__
            if inspect.isfunction(value):
                if name not in seen:
                    out.append(name)
                seen.add(name)
    return [(name, getattr(cls, name)) for name in out]


def _returns_spec(function):
    """The type a function's or method's *Returns* names, and its annotation, as one string."""
    _, _, sections = _parts(function)
    heads = [item[1] for item in _items(sections.get("Returns", []), refusals=False)
             if item[0] == "entry"]
    annotation = inspect.signature(function).return_annotation
    annotation = "" if annotation is inspect.Signature.empty else str(annotation)
    return " ".join(heads[:1] + [annotation])


def _tree(text):
    """A group's records, written one a line, indented under the record that holds them."""
    out, stack = [], []
    for line in text.strip("\n").split("\n"):
        depth = (len(line) - len(line.lstrip(" "))) // 2
        node = (line.strip(), [])
        while len(stack) > depth:
            stack.pop()
        if depth > len(stack):
            raise RuntimeError(f"{line.strip()} is indented under no record")
        (stack[-1][1] if stack else out).append(node)
        stack.append(node)
    return out


class _Records:
    """The records, what holds and returns each, and the page's order of them."""

    def __init__(self):
        self.package = _package()
        found = _record_classes()
        self.classes = {name: cls for name, (cls, _) in found.items()}
        self.modules = {name: modules for name, (_, modules) in found.items()}
        self.names = _Names(self.package, records=self.classes)
        records = set(self.classes)
        # what names each record, in the order of the record's own entries:
        # held[child] is a list of (owner, how it is written, position in owner)
        self.held = {name: [] for name in records}
        for owner, cls in self.classes.items():
            fields = ({f.name: str(f.type) for f in dataclasses.fields(cls)}
                      if dataclasses.is_dataclass(cls) else {})
            position = 0
            for names, spec, _ in _attributes(cls):
                for field in names:
                    for child in _named(spec + " " + fields.get(field, ""), records):
                        self.held[child].append((owner, f"{owner}.{field}", position))
                position += 1
            for method, function in _methods(cls):
                for child in _named(_returns_spec(function), records):
                    self.held[child].append((owner, f"{owner}.{method}()", position))
                position += 1
        # a field written by hand stands where its entry does; a row or a
        # value of a mapping, ``table[j]``, before every field
        stale = [name for name in RETURNED_BY_HAND if name not in self.classes]
        for child, owners in HELD_BY_HAND.items():
            for owner, written in owners:
                if child not in self.classes or owner not in self.classes:
                    stale.append(f"{owner} holding {child}")
                    continue
                field = written.partition(".")[2]
                spots = [p for p, (names, _, _) in enumerate(_attributes(self.classes[owner]))
                         if field in names]
                if field and not spots and not (
                        dataclasses.is_dataclass(self.classes[owner])
                        and field in {f.name for f in dataclasses.fields(self.classes[owner])}):
                    stale.append(f"{written}, no field of {owner}")
                self.held[child].append((owner, written, spots[0] if spots else -1))
        if stale:
            raise RuntimeError("a line written by hand names what the package does not export "
                               "or hold:\n  " + "\n  ".join(stale))
        # each entry of the field must name the record it is said to hold
        for child, field in HELD_IN_EVERY.items():
            if child not in self.classes:
                stale.append(f"{child}, the values of every {field}")
                continue
            found = False
            for owner, cls in self.classes.items():
                for position, (names, spec, lines) in enumerate(_attributes(cls)):
                    if field in names:
                        found = True
                        if child not in spec + " " + " ".join(lines):
                            stale.append(f"{owner}.{field}, whose entry does not name {child}")
                        self.held[child].append((owner, f"{owner}.{field}", position))
            if not found:
                stale.append(f"{field}, which no record holds")
        # a call or a method a hand line names must be one the package exports
        public = set(self.names.functions) | {shown for shown, _ in _public_functions()}
        for name, hand_lines in RETURNED_BY_HAND.items():
            text = " ".join(line for _, line in hand_lines)
            for role, target in re.findall(r":(func|meth):`([^`]*)`", text):
                shown = self.names.short(target.lstrip("~").lstrip("."))
                owner, _, method = shown.rpartition(".")
                if role == "func" and shown not in public:
                    stale.append(f"{name}'s line names {shown}, which no public module exports")
                if role == "meth" and not (owner in self.classes
                                           and callable(getattr(self.classes[owner], method, None))):
                    stale.append(f"{name}'s line names {shown}, which no record has")
        if stale:
            raise RuntimeError("a line written by hand names what the package does not export "
                               "or hold:\n  " + "\n  ".join(stale))
        self.returned = {name: [] for name in records}
        self.names.public = {function.__name__: shown for shown, function in _public_functions()
                             if shown != function.__name__}
        for shown, function in _public_functions():
            for child in _named(_returns_spec(function), records):
                self.returned[child].append(shown)

    def placed(self):
        """Each group's records in page order, as (name, depth); stops unless each is once."""
        order, problems = [], []

        def walk(node, depth, parent):
            name, children = node
            if name not in self.classes:
                problems.append(f"not exported by a public module: {name}")
                return
            if parent and not any(owner == parent for owner, _, _ in self.held[name]):
                problems.append(f"{name} stands under {parent}, which holds no {name}")
            order.append((name, depth))
            rank = {}
            for child, _ in children:
                spots = [p for owner, _, p in self.held.get(child, ()) if owner == name]
                rank[child] = min(spots) if spots else 10 ** 6
            for child in sorted(children, key=lambda c: rank[c[0]]):
                walk(child, depth + 1, name)

        groups = []
        for title, _, text in RECORD_GROUPS:
            start = len(order)
            for node in _tree(text):
                walk(node, 0, None)
            groups.append((title, order[start:]))
        placed = [name for name, _ in order]
        missing = sorted(set(self.classes) - set(placed))
        twice = sorted({name for name in placed if placed.count(name) > 1})
        if missing:
            problems.append(f"in no group: {', '.join(missing)}")
        if twice:
            problems.append(f"in two places: {', '.join(twice)}")
        if problems:
            raise RuntimeError("RECORD_GROUPS does not hold every exported record once:\n  "
                               + "\n  ".join(problems))
        return groups


def _site_slug(text):
    """The anchor the documentation site gives a heading's text.

    ``pymdownx.slugs.slugify(case="lower")``, as ``mkdocs.yml`` sets it: the
    text in lower case, every character dropped but a letter, a digit, an
    underscore, a hyphen and a space, and each space a hyphen.  A run of
    hyphens is kept as it is.
    """
    text = unicodedata.normalize("NFC", re.sub(r"</?[^>]*>", "", text)).strip().lower()
    return re.sub(r"[^\w\- ]", "", text).replace(" ", "-")


def _rendered(heading):
    """A heading's text as the site renders it: code font's backticks gone."""
    return re.sub(r"`+([^`]*)`+", r"\1", heading)


def _returned_by(name, records):
    """What returns or holds a record: lines of (label, Markdown).

    **Returned by** names the calls and the methods that return it, a call
    linked to its entry on the functions' page where ``pythtqc`` exports it;
    **Held in** names the fields that hold it, each linked to the entry of the
    record that has the field.  A record no annotation reaches has its line in
    :data:`RETURNED_BY_HAND`.
    """
    if name in RETURNED_BY_HAND:
        return [(label, " ".join(_paragraph([text], records.names)))
                for label, text in RETURNED_BY_HAND[name]]
    calls, fields = [], []
    for shown in records.returned[name]:
        calls.append(f"[{_code(shown)}](functions.md#{shown})"
                     if shown in records.names.functions else _code(shown))
    for owner, written, _ in records.held[name]:
        link = f"[{_code(written)}](#{_site_slug(owner)})"
        (calls if written.endswith("()") else fields).append(link)
    if not calls and not fields:
        raise RuntimeError(f"nothing returns or holds {name}: give it a line in RETURNED_BY_HAND")
    return ([("Returned by", ", ".join(calls))] if calls else []) + (
        [("Held in", ", ".join(fields))] if fields else [])


def _record_entry(name, depth, records):
    """One record's entry: its heading, docstring, what returns it, fields and methods."""
    cls = records.classes[name]
    names = records.names
    summary, lead, sections = _parts(cls)
    level = "#" * min(3 + depth, 6)
    out = ["", f"{level} {_code(name)}", ""]
    out += _paragraph(summary, names) + [""]
    for block in _blocks(lead, names):
        out += block + [""]
    for label, text in _returned_by(name, records):
        out += [f"**{label}** {text}.", ""]
    modules = records.modules[name]
    out += ["**Exported by** " + ", ".join(_code(m) for m in modules) + ".", ""]
    body = sections.get("Attributes")
    if body and any(line.strip() for line in body):
        out += _section("Attributes", body, names)
    documented = {field for names_, _, _ in _attributes(cls) for field in names_}
    fields = ([f.name for f in dataclasses.fields(cls) if not f.name.startswith("_")]
              if dataclasses.is_dataclass(cls) else [])
    undocumented = [field for field in fields if field not in documented]
    # a field of the base whose value here is another record, as a line of
    # HELD_BY_HAND says, is not the base's: its record stands under this one
    retyped = {written.partition(".")[2] for owners in HELD_BY_HAND.values()
               for owner, written in owners if owner == name}
    undocumented = [field for field in undocumented if field not in retyped]
    if undocumented:
        bases = [base.__name__ for base in cls.__mro__[1:] if base.__name__ in records.classes]
        base_fields = {field for base in bases
                       for names_, _, _ in _attributes(records.classes[base])
                       for field in names_}
        if not bases or not set(undocumented) <= base_fields:
            raise RuntimeError(f"{name} has fields with no entry: {', '.join(undocumented)}")
        out += [f"The fields {', '.join(_code(f) for f in undocumented)} are those of "
                f"[{_code(bases[0])}](#{_site_slug(bases[0])}), "
                "with the entries there.", ""]
    methods = _methods(cls)
    if methods:
        out += ["**Methods**", ""]
        for method, function in methods:
            first, _, _ = _parts(function) if inspect.getdoc(function) else ([], [], {})
            signature = _signature(method, function, drop_first=True)
            head = _code(" ".join(line.strip() for line in signature).replace("( ", "(")
                         .replace(" )", ")"))
            text = " ".join(_paragraph(first, names)) if first else ""
            out.append(f"- **{head}**" + (f": {text}" if text else ""))
        out.append("")
    while out[-1] == "":
        out.pop()
    return out


def render_records():
    """The records' page, as the docstrings make it now."""
    records = _Records()
    groups = records.placed()
    names = records.names
    out = RECORDS_INTRODUCTION.split("\n")
    seen = {}
    for (title, opening, _), (_, placed) in zip(RECORD_GROUPS, groups):
        out.append(f"- [{title}](#{_site_slug(title)}): {opening[0]}")
    for (title, opening, _), (_, placed) in zip(RECORD_GROUPS, groups):
        out += ["", f"## {title}", ""] + opening[1].strip("\n").split("\n") + [""]
        for name, depth in placed:
            if depth == 0:
                summary, _, _ = _parts(records.classes[name])
                out.append(f"- [{_code(name)}](#{_site_slug(name)}): "
                           + " ".join(_paragraph(summary, names)))
        for name, depth in placed:
            out += _record_entry(name, depth, records)
    for line in out:
        if line.startswith("#"):
            slug = _site_slug(_rendered(line.lstrip("#").strip()))
            if slug in seen:
                raise RuntimeError(f"two headings have the anchor #{slug}")
            seen[slug] = line
    return "\n".join(out) + "\n"


def anchors_of(text):
    """Every heading's anchor on a page this script writes, in page order."""
    return [_site_slug(_rendered(line.lstrip("#").strip()))
            for line in text.split("\n") if re.match(r"#{1,6} ", line)]


def _slug(title):
    """The anchor GitHub gives a group's heading.

    The title in lower case, its code font and punctuation dropped and its
    spaces hyphens; the pages' link check holds every such anchor to GitHub's
    rule (``scripts/check_page.py``).
    """
    kept = (c for c in title.lower() if c.isalnum() or c in "-_ ")
    return "".join(kept).replace(" ", "-")


def _write(page, text):
    """Write a page where it differs from the tree's, and say whether it did."""
    before = page.read_bytes().decode("utf-8").replace("\r\n", "\n") if page.exists() else None
    if text != before:
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text(text, encoding="utf-8", newline="\n")
    return "unchanged" if text == before else ("written" if before is None else "rewritten")


def main():
    try:
        functions = render()
        records = render_records()
    except RuntimeError as error:
        print(error)
        return 1
    state = _write(PAGE, functions)
    count = sum(len(members) for _, _, members in GROUPS)
    print(f"{PAGE.relative_to(ROOT).as_posix()}: {count} functions in {len(GROUPS)} groups, "
          f"{functions.count(chr(10))} lines, {state}")
    state = _write(RECORDS_PAGE, records)
    count = len(re.findall(r"^#{3,6} `", records, re.M))
    print(f"{RECORDS_PAGE.relative_to(ROOT).as_posix()}: {count} records in "
          f"{len(RECORD_GROUPS)} groups, {records.count(chr(10))} lines, {state}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
