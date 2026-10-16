r"""Convert the stable-RSI (SRSI) supplement tables of Hwang et al. to JSON.

    python scripts/convert_srsi_tables.py "<supple info folder>" --out src/pythtqc/data

The source is the TeX of the supplement of

    Hwang et al., "Stable real-space invariants and topology beyond symmetry
    indicators", Nat. Commun. 17, 8564 (2026),
    https://doi.org/10.1038/s41467-026-74844-w

specifically the six ``longtable`` files of its supplementary information
folder ``260624 updated files/supple info``:

===================================  =====================================
``rsi_tables_single.tex``            the SRSIs :math:`\theta_i` without SOC
``rsi_tables_double.tex``            the SRSIs with SOC
``stableindex_tables_single.tex``    the symmetry indicators (SIs) as
``stableindex_tables_double.tex``    integer/rational combinations of the
                                     :math:`\theta_i`, mod :math:`n`
``ebr_tables_single.tex``            the symmetry-indicated split EBRs
``ebr_tables_double.tex``            :math:`\Lambda = V \oplus C`
===================================  =====================================

**The TeX is the paper's, and is not shipped with this package**: the folder is
handed to this script on the command line, and only the converted JSON (under
``src/pythtqc/data``) and this script itself live in the repository.  The
SHA-256 of every source file read is recorded in the JSON's provenance block, so
a table can always be traced back to the bytes it came from.

What the conversion does, and what it refuses to do
---------------------------------------------------
Every ``(site-symmetry irrep, Wyckoff letter)`` term of every :math:`\theta` is
**resolved** against ``pythtqc.groups.ebr_data`` of the group's **grey**
(type-II) BNS entry — the paper's tables are for time-reversal-symmetric
systems — and the resolved EBR column index is stored beside the term.  A term
whose Wyckoff letter *is* one of the group's EBR (maximal) positions and whose
label does not resolve to exactly one EBR there is a **hard error naming the
row**: a conversion that guesses is worse than none.

The paper's site-symmetry representation vector :math:`p` runs over **all**
Wyckoff positions, maximal and not (SN "Atomic insulators": *"we considered all
WPs to define the site-symmetry representation vector"*), while an EBR basis
carries only the maximal ones.  Terms at non-maximal positions are therefore
recorded in full — coefficient, label, letter — with ``"column": null`` and
``"maximal": false``.  A *maximal* position can also carry a site irrep whose
induced band representation is **composite** rather than elementary, which no
EBR list carries either; those terms are stamped ``"maximal": true,
"ebr": false`` and, measured, they are ``{}^{1}E{}^{2}E`` at site symmetry
:math:`4` and ``E`` at :math:`\bar 4 m 2` / :math:`\bar 4 2 m`, 39 positions in
the single-valued table and none in the double-valued one.  Nothing is dropped
from the JSON; what the loader builds its :math:`\Theta` matrix from is the EBR
block, which is exact for any :math:`p` supported on EBR columns — which is
every :math:`p` this package builds.

The grey BNS number of each space group is **measured** from ``irreptables``'
own tables (the unique magnetic entry of that space group whose antiunitary
coset contains :math:`\{1|0\}\cdot T`, the same detector
``pythtqc.groups._magnetic_type`` uses), never typed.

Output
------
``srsi_single.json`` and ``srsi_double.json``, each

.. code-block:: text

    {"provenance": {...}, "groups": {"25": {...}, ...}}

with one entry per space group present in the tables, holding ``srsis``
(the :math:`\theta_i` with their moduli and resolved terms), ``indicators``
(the named SIs as rational combinations of the :math:`\theta_i`, mod n),
``split_ebrs`` (where the group has any) and the group's EBR column key.

Self-checks run on every conversion, and reported at the end
-----------------------------------------------------------
* every term of every :math:`\theta` resolves, or says which of the two
  no-EBR cases it is;
* the row, term and modulus counts;
* **the kernel test** — ``ker(BR) subset ker(Theta^(0))`` on the EBR columns,
  over the rationals and exactly, with the ranks compared as well.  This is the
  paper's own structural claim (the one-to-one map between the
  :math:`\mathbb{Z}`-SRSIs and the symmetry-data vector), and it is the
  sharpest independent check the conversion has of the label map, the Wyckoff
  letters and the column indices *together*: one mis-resolved term generically
  breaks it.  Measured 2026-09-02: zero violations in all 217 + 217 groups;
* **the SI check** — every tabulated symmetry indicator must evaluate to
  ``0 mod n`` on every EBR column, because an EBR is an atomic limit.  This one
  is *recorded* rather than enforced: it holds for all 320 double-valued
  expressions and for 66 of the 70 single-valued ones, and fails for every
  expression of SGs 147, 162 and 163 without SOC — the only three groups whose
  indicator rows carry fractional :math:`\theta`-coefficients.  Those rows
  ship flagged ``"vanishes_on_ebrs": false`` with the columns they fail on;
* **the derivation** (:func:`derive_indicators`) — every printed indicator is
  *re-derived* from the momentum-space side, as the unique rational ``a`` with
  :math:`a\cdot\Theta^{(0)} = c\cdot\mathrm{BR}` for ``irreptables``' own named
  form ``c`` (or, in a group with a single elementary divisor, its own Smith
  coordinate).  The derived row ships beside the printed one with a
  ``relation`` — measured 2026-09-03: 320 of 320 double-valued rows *equal*;
  54 equal, 3 equivalent and 4 different of the 61 single-valued rows with a
  counterpart, with 9 more having none.  The four that differ are exactly the
  four the SI check flags, so the two independent measurements agree about the
  defect and the derivation supplies the correction;
* the injectivity of the label map inside each group.
"""

from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
import os
import re
import sys
from fractions import Fraction

import sympy

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

from pythtqc.groups import ebr_data, tables_status          # noqa: E402
from pythtqc.ebr import (_named_forms, _smith_system,       # noqa: E402
                         _unimodular_inverse)

DOI = "https://doi.org/10.1038/s41467-026-74844-w"
PAPER = ("Hwang et al., 'Stable real-space invariants and topology beyond "
         "symmetry indicators', Nat. Commun. 17, 8564 (2026)")

#: The caption's own rule for the space groups absent from the SRSI tables.
OMITTED_REASON = (
    "space groups with only one site-symmetry irrep in a unit cell, for which the "
    "unique SRSI is determined by the number of bands (the caption of the "
    "supplement's SRSI tables states the omission and this reason)")

SRSI_FILES = {"single": "rsi_tables_single.tex", "double": "rsi_tables_double.tex"}
SI_FILES = {"single": "stableindex_tables_single.tex",
            "double": "stableindex_tables_double.tex"}
EBR_FILES = {"single": "ebr_tables_single.tex", "double": "ebr_tables_double.tex"}


class ConversionError(RuntimeError):
    """A table row this converter will not guess at.

    Raised, not returned: an unparseable row or an unresolved site-symmetry
    irrep is a defect of the conversion (or of the source), not a measurement
    about a model, so there is nothing for a caller to branch on.
    """


# ---------------------------------------------------------------------------
# the grey (type-II) BNS number of each space group, measured
# ---------------------------------------------------------------------------

def grey_numbers() -> dict[int, str]:
    r"""``{sg: "<sg>.<bns>"}`` for the grey (type-II) entry of every space group.

    Measured from ``irreptables``' shipped correpresentation tables, with the
    same test :func:`pythtqc.groups._magnetic_type` applies to a measured
    operation list: a group is grey exactly when its antiunitary coset contains
    time reversal itself, :math:`\{1|0\}\cdot T` up to a lattice vector.  Each
    ``irreps-SG=<n>.<m>-scal.dat`` header lists the operations as
    ``R(9) t(3) antiunitary(±1)``, which is all the test needs.

    Raises :class:`ConversionError` when a space group does not have exactly one
    such entry — the BNS listing says it must.
    """
    import irreptables

    root = os.path.join(os.path.dirname(irreptables.__file__), "data", "correptables")
    ident = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    entries: dict[int, list[int]] = collections.defaultdict(list)
    for fname in os.listdir(root):
        m = re.match(r"^irreps-SG=(\d+)\.(\d+)-scal\.dat$", fname)
        if m:
            entries[int(m.group(1))].append(int(m.group(2)))

    out = {}
    for sg, bns_numbers in entries.items():
        hits = []
        for bns in sorted(bns_numbers):
            path = os.path.join(root, f"irreps-SG={sg}.{bns}-scal.dat")
            with open(path, "r", errors="replace") as handle:
                lines = handle.read().splitlines()
            start = next(i for i, ln in enumerate(lines)
                         if ln.strip().startswith("symmetries=")) + 1
            for line in lines[start:]:
                tok = line.split()
                if len(tok) != 13:
                    break
                rot = [int(t) for t in tok[:9]]
                tau = [Fraction(t) for t in tok[9:12]]
                if (int(tok[12]) == -1 and rot == ident
                        and all(t.denominator == 1 for t in tau)):
                    hits.append(bns)
                    break
        if len(hits) != 1:
            raise ConversionError(
                f"space group {sg}: {len(hits)} grey (type-II) BNS entries {hits} in "
                "irreptables' correpresentation tables; exactly one was expected")
        out[sg] = f"{sg}.{hits[0]}"
    return out


# ---------------------------------------------------------------------------
# the label map: the supplement's TeX Mulliken -> irreptables' spelling
# ---------------------------------------------------------------------------

_SUPERSCRIPT = re.compile(r"\^\{?([12])\}?")
_SUBSCRIPT_BRACED = re.compile(r"_\{([^{}]*)\}")
_SUBSCRIPT_BARE = re.compile(r"_([A-Za-z0-9])")


def tex_label(tex: str) -> str:
    r"""The ``irreptables`` spelling of a supplement site-symmetry irrep label.

    Purely mechanical, and every rule is forced by a spelling that occurs in the
    tables:

    * ``\bar{X}`` marks a **double-valued** label; ``irreptables`` writes that as
      a single leading ``-`` on the whole label, however many bars the TeX
      carries (``\bar{A}_g\bar{A}_g`` -> ``-AgAg``);
    * the empty group ``{}`` that TeX needs before a left superscript is
      dropped, and ``^{1}`` / ``^1`` becomes a bare ``1``
      (``{}^{1}E{}^{2}E`` -> ``1E2E``, ``{}^1\bar{E}{}^2\bar{E}`` -> ``-1E2E``);
    * subscripts lose their braces (``A_{1g}`` -> ``A1g``, ``\bar{E}_1`` ->
      ``-E1``);
    * primes are kept as written (``A'``, ``A''``, ``{}^{1}E'{}^{2}E'`` ->
      ``1E'2E'``).

    The map is not trusted on its word: the conversion **resolves** every
    ``(label, Wyckoff letter)`` pair against the group's own EBR list, and a
    maximal-position pair that does not land on exactly one EBR stops the run.
    """
    bar = "\\bar" in tex
    out = tex.replace("{}", "").replace("\\bar", "")
    out = _SUPERSCRIPT.sub(r"\1", out)
    out = _SUBSCRIPT_BRACED.sub(r"\1", out)
    out = _SUBSCRIPT_BARE.sub(r"\1", out)
    out = out.replace("{", "").replace("}", "")
    if "\\" in out:
        raise ConversionError(f"unconverted TeX left in site-symmetry irrep {tex!r}: "
                              f"{out!r}")
    return ("-" + out) if bar else out


_NAME_TOKENS = {"\\theta": "theta", "\\pi": "pi", "\\delta": "delta",
                "\\gamma": "gamma", "\\omega": "omega", "\\phi": "phi",
                "\\Theta": "Theta", "\\alpha": "alpha", "\\beta": "beta",
                "\\eta": "eta", "\\zeta": "zeta", "\\kappa": "kappa",
                "\\mu": "mu", "\\nu": "nu", "\\chi": "chi", "\\lambda": "lambda",
                "\\sigma": "sigma", "\\tau": "tau", "\\rho": "rho"}


def indicator_name(tex: str) -> str:
    r"""The flat spelling of a symmetry-indicator name: ``z_{2w,1}`` -> ``z2w,1``.

    Greek control sequences become their names (``\delta_2'`` -> ``delta2'``,
    ``\omega_{2,\pi}`` -> ``omega2,pi``), subscript braces are dropped and
    primes are kept.  These are the labels of SRef. Song et al. 2018, which the
    supplement's caption names as its convention; where one of them coincides
    with a form ``irreptables`` ships under the same name the two are compared
    in the test suite, and where it does not, the mismatch is reported rather
    than papered over.
    """
    out = tex.strip()
    for tok, word in _NAME_TOKENS.items():
        out = out.replace(tok, word)
    out = out.replace("_", "").replace("{", "").replace("}", "")
    out = re.sub(r"\s+", "", out)
    if "\\" in out:
        raise ConversionError(f"unconverted TeX left in indicator name {tex!r}: {out!r}")
    return out


# ---------------------------------------------------------------------------
# the longtable grammar
# ---------------------------------------------------------------------------

def _body(text: str, where: str) -> str:
    """The rows of a ``longtable``, with caption and header removed."""
    marker = r"\hline \hline"
    if marker not in text:
        raise ConversionError(f"{where}: no '\\hline \\hline' header terminator found")
    body = text[text.index(marker) + len(marker):]
    if r"\end{longtable}" not in body:
        raise ConversionError(f"{where}: no \\end{{longtable}}")
    return body[:body.rindex(r"\end{longtable}")]


def _rows(text: str, where: str) -> list[tuple[int, str]]:
    """``[(space-group number, the row's second cell), ...]``, in file order."""
    out = []
    for chunk in re.split(r"\\\\\s*\\hline", _body(text, where)):
        chunk = chunk.strip()
        if not chunk:
            continue
        m = re.match(r"^(\d+)\s*&(.*)$", chunk, re.S)
        if not m:
            raise ConversionError(f"{where}: row does not start with 'N&': "
                                  f"{chunk[:120]!r}")
        out.append((int(m.group(1)), m.group(2)))
    if len({sg for sg, _ in out}) != len(out):
        seen = collections.Counter(sg for sg, _ in out)
        dupes = [sg for sg, n in seen.items() if n > 1]
        raise ConversionError(f"{where}: repeated space groups {dupes}")
    return out


_TERM = re.compile(
    r"(?P<sign>[+-])\s*(?P<coef>\d+)?\s*"
    r"m\[\((?P<irrep>.+?)\)_\{\\mathrm\{(?P<w>[a-zA-Z])\}\}\]")
_MOD = re.compile(r"\\mod\s+(\d+)")


def parse_srsi_row(sg: int, cell: str, where: str) -> list[dict]:
    r"""One space group's ``\theta_i`` definitions, from its table cell.

    The cell is a sequence of ``$\theta_{i}= <terms>$`` boxes, broken across
    lines with ``\\ &`` and repeating ``\theta_{i}=`` inside a ``\hphantom{}``
    on every continuation (which is spacing, and is dropped).  A ``\mod n``
    suffix marks a :math:`\mathbb{Z}_n`-valued SRSI; its absence marks a
    :math:`\mathbb{Z}`-valued one.

    After every term, every modulus and the row's own scaffolding have been
    consumed, what is left must be blank — anything else is a grammar this
    converter has not measured, and stops the run.
    """
    flat = cell.replace("$", " ")
    flat = re.sub(r"\\\\\s*&", " ", flat)
    flat = re.sub(r"\\hphantom\{\\theta_\{\d+\}=\}", " ", flat)
    parts = re.split(r"\\theta_\{(\d+)\}\s*=", flat)
    if parts[0].strip():
        raise ConversionError(f"{where} SG {sg}: text before the first theta: "
                              f"{parts[0][:120]!r}")
    if len(parts) < 3:
        raise ConversionError(f"{where} SG {sg}: no theta definition in the row")

    out = []
    walk = iter(parts[1:])
    for index, expr in zip(walk, walk):
        modulus = None
        found = _MOD.search(expr)
        if found:
            modulus = int(found.group(1))
            expr = expr[:found.start()] + expr[found.end():]
        terms = []
        for hit in _TERM.finditer(expr):
            coef = int(hit.group("coef") or 1)
            terms.append({"coef": -coef if hit.group("sign") == "-" else coef,
                          "irrep_tex": hit.group("irrep"),
                          "irrep": tex_label(hit.group("irrep")),
                          "wyckoff": hit.group("w")})
        rest = _TERM.sub(" ", expr).strip()
        if rest:
            raise ConversionError(
                f"{where} SG {sg} theta_{index}: unparsed text {rest[:120]!r}")
        if not terms:
            raise ConversionError(f"{where} SG {sg} theta_{index}: no terms")
        seen = collections.Counter((t["irrep"], t["wyckoff"]) for t in terms)
        repeated = [k for k, n in seen.items() if n > 1]
        if repeated:
            raise ConversionError(
                f"{where} SG {sg} theta_{index}: the site-symmetry irreps {repeated} "
                "occur twice in one linear form")
        out.append({"index": int(index), "name": f"theta_{int(index)}",
                    "mod": modulus, "terms": terms})
    if [t["index"] for t in out] != list(range(1, len(out) + 1)):
        raise ConversionError(f"{where} SG {sg}: theta indices are not 1..n: "
                              f"{[t['index'] for t in out]}")
    return out


_SI_TERM = re.compile(
    r"(?P<sign>[+-])?\s*"
    r"(?:\((?P<num>\d+)/(?P<den>\d+)\)|(?P<int>\d+))?\s*"
    r"\\theta_\{(?P<i>\d+)\}")


def parse_si_row(sg: int, cell: str, where: str) -> list[dict]:
    r"""One space group's symmetry indicators, from its table cell.

    Each indicator is an ``\begin{aligned}NAME=& <terms> \mod n\end{aligned}``
    box; the terms are rational multiples of the :math:`\theta_i` of the same
    space group and valuedness, written ``+4\theta_{2}``, ``-\theta_{5}`` or
    ``+(5/2)\theta_{4}``.  Fractional coefficients are real and frequent (SG 147
    and SG 162 without SOC, for instance), so a coefficient is stored as an
    exact ``[numerator, denominator]`` pair and never rounded.
    """
    out = []
    for box in re.finditer(r"\\begin\{aligned\}(.*?)\\end\{aligned\}", cell, re.S):
        inner = box.group(1)
        if "=&" not in inner:
            raise ConversionError(f"{where} SG {sg}: aligned box without '=&': "
                                  f"{inner[:120]!r}")
        name_tex, expr = inner.split("=&", 1)
        found = _MOD.search(expr)
        if not found:
            raise ConversionError(f"{where} SG {sg} {name_tex.strip()!r}: no modulus")
        modulus = int(found.group(1))
        expr = expr[:found.start()] + expr[found.end():]
        expr = re.sub(r"\\\\\s*&", " ", expr)
        coefficients: dict[str, list[int]] = {}
        for hit in _SI_TERM.finditer(expr):
            if hit.group("num"):
                value = Fraction(int(hit.group("num")), int(hit.group("den")))
            else:
                value = Fraction(int(hit.group("int") or 1))
            if hit.group("sign") == "-":
                value = -value
            key = hit.group("i")
            if key in coefficients:
                raise ConversionError(
                    f"{where} SG {sg} {name_tex.strip()!r}: theta_{key} appears twice")
            coefficients[key] = [value.numerator, value.denominator]
        rest = _SI_TERM.sub(" ", expr).strip()
        if rest:
            raise ConversionError(f"{where} SG {sg} {name_tex.strip()!r}: unparsed "
                                  f"text {rest[:120]!r}")
        if not coefficients:
            raise ConversionError(f"{where} SG {sg} {name_tex.strip()!r}: no terms")
        out.append({"name_tex": name_tex.strip(), "name": indicator_name(name_tex),
                    "mod": modulus, "theta": coefficients})
    if not out:
        raise ConversionError(f"{where} SG {sg}: no aligned indicator boxes in the row")
    leftover = re.sub(r"\\begin\{aligned\}.*?\\end\{aligned\}", " ", cell, flags=re.S)
    leftover = re.sub(r"[\$\s]|\\\\|&", "", leftover)
    if leftover:
        raise ConversionError(f"{where} SG {sg}: text outside the aligned boxes: "
                              f"{leftover[:120]!r}")
    names = [entry["name"] for entry in out]
    if len(set(names)) != len(names):
        raise ConversionError(f"{where} SG {sg}: repeated indicator names {names}")
    return out


_SITE = re.compile(r"\((?P<irrep>.+?)\)_\{\\mathrm\{(?P<w>[a-zA-Z])\}\}")


def parse_split_row(sg: int, cell: str, where: str) -> list[dict]:
    r"""One space group's symmetry-indicated split EBRs.

    Each entry is ``$(\Lambda)_y = (V)_{y'} \oplus (C)_{y''} \quad
    \text{\xmark}$``: a band representation that the symmetry indicators allow
    to split into a sum of two or more EBRs.  The flag is the supplement's own:
    ``\xmark`` where the SRSIs of the two sides **mismatch** (which the paper
    proves forces topology) and ``\cmark`` where they match.
    """
    out = []
    for piece in re.split(r"\\\\\s*&", cell):
        piece = piece.strip()
        if not piece:
            continue
        if "\\xmark" in piece:
            mismatch = True
        elif "\\cmark" in piece:
            mismatch = False
        else:
            raise ConversionError(f"{where} SG {sg}: split EBR with neither \\xmark "
                                  f"nor \\cmark: {piece[:120]!r}")
        sites = [{"irrep_tex": m.group("irrep"), "irrep": tex_label(m.group("irrep")),
                  "wyckoff": m.group("w")} for m in _SITE.finditer(piece)]
        if len(sites) < 3:
            raise ConversionError(f"{where} SG {sg}: split EBR with fewer than a parent "
                                  f"and two children: {piece[:120]!r}")
        rest = _SITE.sub(" ", piece)
        for token in ("$", "=", "\\oplus", "\\quad", "\\text", "\\xmark", "\\cmark",
                      "{", "}"):
            rest = rest.replace(token, " ")
        if rest.strip():
            raise ConversionError(f"{where} SG {sg}: unparsed text in a split EBR: "
                                  f"{rest.strip()[:120]!r}")
        out.append({"parent": sites[0], "children": sites[1:],
                    "srsi_mismatch": mismatch})
    if not out:
        raise ConversionError(f"{where} SG {sg}: no split EBRs in the row")
    return out


# ---------------------------------------------------------------------------
# resolution against the group's own EBR list
# ---------------------------------------------------------------------------

class _GreyKey:
    """The three fields :func:`pythtqc.groups.ebr_data` reads off a group record.

    ``ebr_data`` needs a ``number_str``, a ``spinor`` flag and a ``magnetic``
    flag and nothing else — no lattice, no operations, no model — so the
    conversion, which has no Hamiltonian to hand, supplies exactly those.  The
    loader in ``pythtqc.srsi`` is handed real
    :class:`~pythtqc.groups.GroupData` records instead.
    """

    def __init__(self, number_str: str, spinor: bool):
        self.number_str = number_str
        self.spinor = bool(spinor)
        self.magnetic = True


def ebr_index(number_str: str, spinor: bool):
    """``(key list, {(label, letter): columns}, letters, site symbols)`` for a grey group.

    The key list is ``[(label, Wyckoff letter), ...]`` in ``ebr_data``'s own
    column order, which is what the stored column indices index and what the
    loader re-checks the installed tables against.  The site symbols come from
    the position strings (``1a(2221',222)`` -> ``2221',222``) and are what the
    label-elimination report is keyed by.
    """
    data = ebr_data(_GreyKey(number_str, spinor))
    if not data:
        raise ConversionError(f"#{number_str} (spinor={spinor}): ebr_data refuses, "
                              f"so its terms cannot be resolved -- {data}")
    keys, index, symbols = [], {}, {}
    for column, (name, position) in enumerate(data["ebrs"]):
        label = str(name).split("\u2191")[0]
        m = re.match(r"^(\d+)([a-zA-Z])\((.*)\)$", str(position))
        if not m:
            raise ConversionError(f"#{number_str}: EBR position {position!r} is not "
                                  "'<multiplicity><letter>(<site symbols>)'")
        key = (label, m.group(2))
        keys.append(key)
        index.setdefault(key, []).append(column)
        symbols[m.group(2)] = m.group(3)
    return keys, index, {letter for _label, letter in keys}, symbols


def resolve_labels(paper_labels, table_labels, site, where):
    r"""Match one Wyckoff position's site-irrep spellings, **by elimination**.

    Returns ``(alias, absent)``: the paper spellings that had to be identified
    with a differently-spelled EBR label, and the paper spellings that name a
    site irrep the group's EBR list does not carry at all.

    Two conventions differ between the supplement and ``irreptables``, and both
    are *derived* here rather than typed:

    * for the site group :math:`222` the supplement writes the totally symmetric
      irrep ``A_1`` where ``irreptables`` writes ``A`` (both are current
      Mulliken practice for :math:`D_2`).  At such a position every other
      spelling matches directly, so exactly one label is left over on each side
      and the identification is **forced**; the converter reports every alias it
      derived, and refuses if more than one is ever left over on either side;
    * a site irrep whose induced band representation is **not elementary** is
      absent from an EBR list by construction, so a paper spelling can be left
      over with nothing on the table's side to pair it with.  Measured, that is
      ``{}^{1}E{}^{2}E`` at site symmetry :math:`4` and ``E`` at
      :math:`\bar{4}m2` / :math:`\bar{4}2m`; those terms keep their coefficient
      and their Wyckoff letter and are stamped ``ebr=false``.

    Anything else — two unmatched spellings against one unmatched label, say —
    is a grammar this converter has not measured, and stops the run.
    """
    only_paper = sorted(set(paper_labels) - set(table_labels))
    only_table = sorted(set(table_labels) - set(paper_labels))
    if not only_paper:
        return {}, set()
    if not only_table:
        return {}, set(only_paper)
    if len(only_paper) == 1 and len(only_table) == 1:
        return {only_paper[0]: only_table[0]}, set()
    raise ConversionError(
        f"{where}: at site {site!r} the supplement's spellings {only_paper} and the "
        f"EBR labels {only_table} cannot be matched by elimination -- {len(only_paper)} "
        f"against {len(only_table)} left over, and nothing forces the pairing")


def resolve_group(srsis, keys, index, letters, site_symbols, where):
    """Stamp every term of every SRSI of one group with its EBR column, or its reason.

    Three outcomes, all recorded on the term itself:

    ``maximal=True, ebr=True``
        the site irrep induces an EBR, and ``column`` is its index into
        ``ebr_data(group)["ebrs"]``;
    ``maximal=True, ebr=False``
        a maximal Wyckoff position whose site irrep induces a **composite** band
        representation, which no EBR list carries (``column`` is ``null``);
    ``maximal=False, ebr=False``
        a non-maximal Wyckoff position — the paper's :math:`p` runs over all of
        them and an EBR basis over none (``column`` is ``null``).
    """
    used = collections.defaultdict(set)
    for srsi in srsis:
        for term in srsi["terms"]:
            if term["wyckoff"] in letters:
                used[term["wyckoff"]].add(term["irrep"])

    aliases, absent = {}, {}
    for letter, paper in used.items():
        table = {label for label, own in keys if own == letter}
        alias, gone = resolve_labels(paper, table, site_symbols[letter],
                                     f"{where} Wyckoff {letter}")
        aliases[letter] = alias
        absent[letter] = gone

    for srsi in srsis:
        for term in srsi["terms"]:
            letter = term["wyckoff"]
            if letter not in letters:
                term.update(column=None, maximal=False, ebr=False)
                continue
            label = aliases[letter].get(term["irrep"], term["irrep"])
            if term["irrep"] in absent[letter]:
                term.update(column=None, maximal=True, ebr=False)
                continue
            hits = index.get((label, letter), [])
            if len(hits) != 1:
                raise ConversionError(
                    f"{where} theta_{srsi['index']}: the site-symmetry irrep "
                    f"({term['irrep_tex']!r} -> {label!r}) at maximal Wyckoff {letter} "
                    f"resolves to {len(hits)} EBRs instead of exactly one")
            term.update(column=int(hits[0]), maximal=True, ebr=True)
            if label != term["irrep"]:
                term["ebr_label"] = label
    return {letter: dict(alias) for letter, alias in aliases.items() if alias}, \
           {letter: sorted(gone) for letter, gone in absent.items() if gone}


# ---------------------------------------------------------------------------
# the conversion
# ---------------------------------------------------------------------------

def _sha256(path):
    with open(path, "rb") as handle:
        raw = handle.read()
    return hashlib.sha256(raw).hexdigest(), len(raw)


def _read(folder, name):
    path = os.path.join(folder, name)
    if not os.path.isfile(path):
        raise ConversionError(f"the supplement folder has no {name}: {path}")
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def convert(folder: str, valuedness: str, *, check_kernel: bool = True) -> dict:
    """Parse, resolve and self-check one valuedness; return the JSON document."""
    spinor = valuedness == "double"
    grey = grey_numbers()
    srsi_file = SRSI_FILES[valuedness]
    si_file = SI_FILES[valuedness]
    ebr_file = EBR_FILES[valuedness]
    srsi_rows = _rows(_read(folder, srsi_file), srsi_file)
    si_rows = _rows(_read(folder, si_file), si_file)
    split_rows = _rows(_read(folder, ebr_file), ebr_file)

    si_by_sg = {sg: parse_si_row(sg, cell, si_file) for sg, cell in si_rows}
    split_by_sg = {sg: parse_split_row(sg, cell, ebr_file) for sg, cell in split_rows}
    tabulated = {sg for sg, _ in srsi_rows}
    stray = sorted((set(si_by_sg) | set(split_by_sg)) - tabulated)
    if stray:
        raise ConversionError(
            f"{valuedness}: space groups {stray} carry indicator or split-EBR rows but "
            "no SRSI row, so their theta indices refer to nothing")

    groups, label_map = {}, {}
    alias_report = collections.Counter()
    absent_report = collections.Counter()
    stats = {"terms": 0, "terms_ebr": 0, "terms_composite": 0, "terms_nonmaximal": 0,
             "srsi_rows": 0, "zn_rows": collections.Counter(),
             "si_expressions": 0, "si_moduli": collections.Counter(),
             "split_rows": 0, "splittings": 0, "split_mismatch": 0,
             "split_match": 0, "ebr_columns": 0, "si_checks": 0,
             "alias_positions": 0, "composite_positions": 0,
             "si_not_vanishing": 0, "si_failing_groups": set(), "kernel_checks": 0,
             "derived": collections.Counter()}

    for sg, cell in srsi_rows:
        number_str = grey[sg]
        keys, index, letters, symbols = ebr_index(number_str, spinor)
        where = f"{srsi_file} SG {sg}"
        srsis = parse_srsi_row(sg, cell, where)
        aliases, absent = resolve_group(srsis, keys, index, letters, symbols, where)
        for letter, alias in aliases.items():
            for mechanical, ebr_label in alias.items():
                alias_report[(symbols[letter], mechanical, ebr_label)] += 1
                stats["alias_positions"] += 1
        for letter, gone in absent.items():
            for label in gone:
                absent_report[(symbols[letter], label)] += 1
                stats["composite_positions"] += 1
        for srsi in srsis:
            for term in srsi["terms"]:
                label_map.setdefault(term["irrep_tex"], term["irrep"])
                if label_map[term["irrep_tex"]] != term["irrep"]:
                    raise ConversionError(f"{where}: the label map is not a function")
                stats["terms"] += 1
                if term["ebr"]:
                    stats["terms_ebr"] += 1
                elif term["maximal"]:
                    stats["terms_composite"] += 1
                else:
                    stats["terms_nonmaximal"] += 1
            stats["srsi_rows"] += 1
            if srsi["mod"] is not None:
                stats["zn_rows"][srsi["mod"]] += 1

        indicators = si_by_sg.get(sg, [])
        n_theta = len(srsis)
        for entry in indicators:
            for key in entry["theta"]:
                if not 1 <= int(key) <= n_theta:
                    raise ConversionError(
                        f"{si_file} SG {sg} {entry['name']!r}: refers to theta_{key} "
                        f"but the group has only {n_theta} SRSIs")
            stats["si_expressions"] += 1
            stats["si_moduli"][entry["mod"]] += 1

        splits = split_by_sg.get(sg, [])
        for entry in splits:
            for site in [entry["parent"]] + list(entry["children"]):
                label_map.setdefault(site["irrep_tex"], site["irrep"])
                letter = site["wyckoff"]
                label = aliases.get(letter, {}).get(site["irrep"], site["irrep"])
                hits = index.get((label, letter), [])
                if len(hits) != 1:
                    raise ConversionError(
                        f"{ebr_file} SG {sg}: the split-EBR constituent "
                        f"({site['irrep']!r}, {letter!r}) resolves to {len(hits)} EBRs; "
                        "the caption states every constituent is an EBR")
                site["column"] = int(hits[0])
                if label != site["irrep"]:
                    site["ebr_label"] = label
            stats["splittings"] += 1
            stats["split_mismatch" if entry["srsi_mismatch"] else "split_match"] += 1
        if splits:
            stats["split_rows"] += 1

        # -- the label map must stay injective inside one group -------------
        for letter in letters:
            spellings = collections.defaultdict(set)
            for srsi in srsis:
                for term in srsi["terms"]:
                    if term["wyckoff"] == letter:
                        spellings[term["irrep"]].add(term["irrep_tex"])
            clashes = {lab: sorted(tex) for lab, tex in spellings.items() if len(tex) > 1}
            if clashes:
                raise ConversionError(
                    f"{where}: two TeX spellings collapse onto one label at Wyckoff "
                    f"{letter}: {clashes}")

        theta_matrix = _theta_matrix(srsis, len(keys))
        _measure_si_on_ebrs(sg, indicators, theta_matrix, len(keys), stats)
        if check_kernel:
            data = ebr_data(_GreyKey(number_str, spinor))
            _check_kernel(sg, srsis, theta_matrix,
                          [[int(x) for x in row] for row in data["matrix"]], where)
            stats["kernel_checks"] += 1
        if indicators:
            derive_indicators(number_str, spinor, srsis, indicators, theta_matrix,
                              ebr_data(_GreyKey(number_str, spinor)),
                              f"{si_file} SG {sg}", stats)
        stats["ebr_columns"] += len(keys)

        groups[str(sg)] = {
            "sg": sg, "number_str": number_str, "valuedness": valuedness,
            "ebrs": [[label, letter] for label, letter in keys],
            "srsis": srsis, "indicators": indicators, "split_ebrs": splits,
        }

    omitted = sorted(set(range(1, 231)) - tabulated)
    sources = []
    for name in (srsi_file, si_file, ebr_file):
        digest, size = _sha256(os.path.join(folder, name))
        sources.append({"file": name, "sha256": digest, "bytes": size})
    status = tables_status()
    provenance = {
        "paper": PAPER, "doi": DOI,
        "valuedness": valuedness,
        "converted": datetime.date.today().isoformat(),
        "converter": "scripts/convert_srsi_tables.py",
        "sources": sources,
        "irrep_version": status["irrep_version"],
        "irreptables_version": status["irreptables_version"],
        "omitted_groups": omitted,
        "omitted_reason": OMITTED_REASON,
        "label_map": dict(sorted(label_map.items())),
        "label_aliases": [
            {"site": site, "supplement": mechanical, "ebr_label": ebr_label,
             "positions": count}
            for (site, mechanical, ebr_label), count in sorted(alias_report.items())],
        "composite_site_irreps": [
            {"site": site, "supplement": label, "positions": count}
            for (site, label), count in sorted(absent_report.items())],
        "counts": {
            "groups": len(groups),
            "srsi_rows": stats["srsi_rows"],
            "zn_rows": dict(sorted(stats["zn_rows"].items())),
            "terms": stats["terms"],
            "terms_on_ebr_columns": stats["terms_ebr"],
            "terms_at_maximal_wyckoff_without_an_ebr": stats["terms_composite"],
            "terms_at_nonmaximal_wyckoff": stats["terms_nonmaximal"],
            "alias_positions": stats["alias_positions"],
            "composite_positions": stats["composite_positions"],
            "si_groups": len(si_by_sg),
            "si_expressions": stats["si_expressions"],
            "si_moduli": dict(sorted(stats["si_moduli"].items())),
            "split_groups": stats["split_rows"],
            "splittings": stats["splittings"],
            "splittings_srsi_mismatch": stats["split_mismatch"],
            "splittings_srsi_match": stats["split_match"],
            "ebr_columns": stats["ebr_columns"],
            "si_on_ebr_checks": stats["si_checks"],
            "si_expressions_not_vanishing_on_ebrs": stats["si_not_vanishing"],
            "si_failing_groups": sorted(stats["si_failing_groups"]),
            "kernel_checks": stats["kernel_checks"],
            "derived_indicator_rows": dict(sorted(stats["derived"].items())),
        },
        "notes": [
            "The site-symmetry representation vector p of the paper runs over ALL "
            "Wyckoff positions; an EBR basis carries only the maximal ones, and only "
            "those maximal site irreps whose induced band representation is elementary. "
            "Terms with no EBR column are kept in full with column=null, and say which "
            "case they are: maximal=false is a non-maximal Wyckoff position, "
            "maximal=true with ebr=false is a maximal position whose induced band "
            "representation is composite. Restricting Theta to the EBR columns is exact "
            "for any p supported there, which is every p this package builds.",
            "The label elimination: at each maximal position the supplement's "
            "spellings and irreptables' EBR labels are matched directly, and what is "
            "left over is paired only when exactly one label remains on each side. "
            "`label_aliases` reports every pairing that was forced this way and "
            "`composite_site_irreps` the supplement spellings that had nothing to pair "
            "with. Both are measured, not declared.",
            "Column indices index ebr_data(grey group)['ebrs'] of pythtqc, whose rows "
            "are transported into this package's e^{-ik.t} convention; the (label, "
            "Wyckoff letter) key list is stored beside them so a loader can check the "
            "installed tables still agree.",
            "Indicator coefficients are exact [numerator, denominator] pairs: the "
            "supplement's SI rows carry fractional coefficients (for example SG 147 "
            "and SG 162 without SOC).",
            "Every indicator carries a DERIVED row beside the printed one: the unique "
            "rational a with a.Theta0 = c.BR, c being irreptables' named form of the "
            "same name (transported and checked to vanish on this group's EBRs) or, "
            "for a group with a single elementary divisor, its own Smith coordinate. "
            "`relation` says how the printed row compares -- equal, equivalent (same "
            "function on a Z-basis of {BS}, mod n) or different. `derived` is null for "
            "the rows whose name has no counterpart in either vocabulary, with the "
            "reason on `derived_note`.",
        ],
    }
    return {"provenance": provenance, "groups": groups}


def _theta_matrix(srsis, n_ebr):
    """``Theta`` restricted to the EBR columns, one row per SRSI."""
    rows = []
    for srsi in srsis:
        row = [0] * n_ebr
        for term in srsi["terms"]:
            if term["column"] is not None:
                row[term["column"]] += term["coef"]
        rows.append(row)
    return rows


def _measure_si_on_ebrs(sg, indicators, theta, n_ebr, stats):
    """Does each tabulated SI vanish mod n on every EBR column?  Measured, not assumed.

    An EBR *is* an atomic limit, so its symmetry indicators are zero; written
    through the SRSIs that is ``sum_i c_i theta_i(e_j) = 0 mod n`` for every EBR
    column ``e_j``.  This is the statement of the paper that the conversion can
    check on its own output with no further input, and it is checked on every
    column of every group.

    It is **recorded, not enforced**: measured 2026-09-02 it holds for all 320
    indicator expressions of the double-valued table and for 66 of the 70
    single-valued ones, and fails for every expression of exactly three groups —
    SGs 147, 162 and 163 without SOC, which are also the only three whose
    indicator rows carry fractional theta-coefficients.  A row that parses and
    resolves cleanly but fails a physics check is a measurement about the
    source, not a malformed row, so it ships with
    ``"vanishes_on_ebrs": false`` and the count of columns it fails on, and the
    conversion says so loudly rather than dropping it or silently repairing it.
    """
    for entry in indicators:
        modulus = entry["mod"]
        coefficients = {int(k): Fraction(v[0], v[1]) for k, v in entry["theta"].items()}
        failures = []
        for column in range(n_ebr):
            total = sum(c * theta[i - 1][column] for i, c in coefficients.items())
            stats["si_checks"] += 1
            if total.denominator != 1 or int(total) % modulus:
                failures.append([column, str(total)])
        entry["vanishes_on_ebrs"] = not failures
        if failures:
            entry["ebr_failures"] = failures
            stats["si_not_vanishing"] += 1
            stats["si_failing_groups"].add(sg)


# ---------------------------------------------------------------------------
# the derived indicator rows
# ---------------------------------------------------------------------------

#: ``irreptables`` spells the Song et al. 2018 indicator names with Greek
#: letters and a typographic prime; the supplement's TeX transliterates them.
#: Same forms, different bytes -- and the *only* thing this map is used for is
#: finding a counterpart to derive against, never for asserting that two names
#: mean the same thing (a canonical vocabulary is a later ruling).
_TRANSLITERATION = {"π": "pi", "′": "'", "δ": "delta",
                    "γ": "gamma", "ω": "omega", "φ": "phi",
                    "Θ": "Theta", "α": "alpha", "β": "beta",
                    "η": "eta", "θ": "theta"}


def _ascii(name: str) -> str:
    for greek, word in _TRANSLITERATION.items():
        name = name.replace(greek, word)
    return name


def _rational_p(u, v, d, y):
    """A rational ``p`` with ``BR . p = y``, from the Smith factors, exactly."""
    n_col = len(v)
    y_prime = [sum(Fraction(u[i][j]) * y[j] for j in range(len(y)))
               for i in range(len(u))]
    z = [y_prime[i] / d[i] if i < len(d) and d[i] else Fraction(0)
         for i in range(n_col)]
    return [sum((Fraction(v[j][i]) * z[i] for i in range(n_col)), Fraction(0))
            for j in range(n_col)]


def _equivalent(theta, u, v, d, printed, derived, modulus) -> bool:
    r"""Do two forms in the :math:`\theta` agree mod ``n`` on a Z-basis of ``{BS}``?

    Two indicator expressions can be different vectors and the same *function*
    on band structures, because they only ever have to agree on ``{BS}`` — the
    lattice of symmetry vectors satisfying the compatibility relations — and
    modulo ``n``.  Its :math:`\mathbb{Z}`-basis is :math:`B_i = u^{-1}e_i` over
    the rows where ``R``'s diagonal is nonzero (the empty rows are the
    compatibility relations, and are zero on every band structure); each
    :math:`B_i` is pulled back to a rational ``p`` by the Smith factors and both
    forms are evaluated there.
    """
    n_row = len(u)
    u_inv = _unimodular_inverse(u)
    for i, di in enumerate(d):
        if di == 0:
            continue
        y = [Fraction(int(u_inv[r][i])) for r in range(n_row)]
        p = _rational_p(u, v, d, y)
        raw = [sum((Fraction(c) * pj for c, pj in zip(row, p)), Fraction(0))
               for row in theta]
        left = sum((c * raw[k - 1] for k, c in printed.items()), Fraction(0))
        right = sum((c * raw[k - 1] for k, c in derived.items()), Fraction(0))
        gap = left - right
        if gap.denominator != 1 or int(gap) % modulus:
            return False
    return True


def derive_indicators(number_str, spinor, srsis, indicators, theta, data, where, stats):
    r"""Derive each printed indicator from the momentum-space side, and compare.

    **The derivation.**  An indicator is a linear form :math:`\kappa = c\cdot B`
    in the *momentum-space* symmetry vector — ``irreptables`` ships one per name,
    and :func:`pythtqc.ebr._named_forms` transports it into this package's
    convention and checks that it vanishes on every EBR of the group, which is
    what makes it an indicator at all.  With :math:`B = \mathrm{BR}\cdot p` and
    :math:`\theta_{\mathbb Z} = \Theta^{(0)}\cdot p`, and with
    :math:`\ker\mathrm{BR} = \ker\Theta^{(0)}` (the conversion's own kernel
    check), the two matrices have the same row space, so there is **exactly
    one** rational ``a`` with :math:`a\cdot\Theta^{(0)} = c\cdot\mathrm{BR}` —
    :math:`\Theta^{(0)}` has full row rank, asserted here — and then
    :math:`\kappa = a\cdot\theta_{\mathbb Z}` on every band structure, exactly.
    That is a *derivation* of the printed row from data the supplement does not
    supply, and it is done in exact rational arithmetic.

    **The counterpart.**  A printed row is derived against the ``irreptables``
    form of the same name and modulus (after undoing the transliteration), or —
    when the group has exactly one elementary divisor and it equals the row's
    modulus — against the **Smith coordinate** ``u[slot] . BR`` itself, which is
    then the unique nonzero homomorphism onto :math:`\mathbb{Z}_n` and so pins
    the counterpart with no name matching at all.  That is how SGs 162 and 163
    resolve.  Where neither exists the row records ``"derived": null`` with the
    reason; measured, that is 9 single-valued rows in SGs 10, 83, 84, 87 and 175,
    where the two vocabularies order the qualifiers differently
    (``delta2+,pi`` against ``delta2,pi+``) and pairing them would be typing a
    name correspondence rather than measuring one.

    **The comparison**, recorded as ``relation``:

    ``"equal"``
        the derived coefficients *are* the printed ones, term for term;
    ``"equivalent"``
        they differ as vectors but agree modulo ``n`` on a
        :math:`\mathbb{Z}`-basis of ``{BS}`` (:func:`_equivalent`), so the two
        rows are the same function of a band structure;
    ``"different"``
        neither.

    Measured 2026-09-03 over both shipped tables: **320 equal** of 320
    double-valued rows; **54 equal, 3 equivalent, 4 different** of the 61
    single-valued rows that have a counterpart, and 9 without one.  The four
    that differ are exactly the four the S1 conversion had already flagged
    ``vanishes_on_ebrs=False`` — SG 147's ``z2,3`` and ``z4`` and SG 162's and
    163's ``z2'``, all without SOC — which is an independent confirmation of
    that defect *and* the correction for it.
    """
    n_row, n_col = len(data["matrix"]), len(data["matrix"][0])
    u, v, d, matrix = _smith_system(data, number_str)
    z_rows = [srsi["index"] for srsi in srsis if srsi["mod"] is None]
    theta_z = [theta[i - 1] for i in z_rows]
    if sympy.Matrix(theta_z).rank() != len(z_rows):
        raise ConversionError(
            f"{where}: the Z-valued SRSI rows do not have full row rank on the EBR "
            f"columns ({sympy.Matrix(theta_z).rank()} of {len(z_rows)}), so the "
            "rational a with a.Theta0 = c.BR is not unique and no indicator can be "
            "derived from the momentum-space side")

    key = _GreyKey(number_str, spinor)
    _values, _touched, _note, coeffs = _named_forms(key, data, matrix,
                                                    [0] * n_row, "II")
    shipped_mod = {}
    try:
        from irreptables.symmetry_indicators import load_si_table
        entry = load_si_table(spinor=spinor, magnetic=True).get(str(number_str), {})
        for name, spec in entry.get("indicators", {}).items():
            shipped_mod[_ascii(str(name))] = int(round(float(spec["mod"])))
    except Exception as exc:                   # pragma: no cover - table defect
        raise ConversionError(f"{where}: irreptables' indicator table could not be "
                              f"read: {exc}")
    by_name = {_ascii(str(name)): tuple(coef) for name, coef in (coeffs or {}).items()}
    slots = [i for i, di in enumerate(d) if di > 1]
    lone = len(slots) == 1

    for entry in indicators:
        name, modulus = _ascii(str(entry["name"])), int(entry["mod"])
        if name in by_name and shipped_mod.get(name) == modulus:
            c_vector = [Fraction(x).limit_denominator(10**6)
                        for x in by_name[name]]
            source = f"irreptables:{name}"
        elif lone and int(d[slots[0]]) == modulus:
            c_vector = [Fraction(int(u[slots[0]][i])) for i in range(n_row)]
            source = "smith"
        else:
            entry["derived"] = None
            entry["derived_note"] = (
                f"no counterpart to derive against: irreptables ships no named form "
                f"{entry['name']!r} of modulus {modulus} for #{number_str}, and the "
                f"group has {len(slots)} elementary divisors, so no Smith coordinate "
                "pins one either. The two vocabularies order this family's qualifiers "
                "differently and pairing them would be typing a name correspondence, "
                "not measuring one")
            stats["derived"]["none"] += 1
            continue

        target = [sum(c_vector[i] * Fraction(int(matrix[i][j]))
                      for i in range(n_row)) for j in range(n_col)]
        left = sympy.Matrix([[sympy.Integer(int(x)) for x in row]
                             for row in theta_z]).T
        rhs = sympy.Matrix([sympy.Rational(x.numerator, x.denominator)
                            for x in target])
        a = left.solve_least_squares(rhs)
        residual = left * a - rhs
        if any(sympy.simplify(x) != 0 for x in residual):
            raise ConversionError(
                f"{where} {entry['name']!r}: no rational a solves a.Theta0 = c.BR, "
                "though ker BR = ker Theta0 forces one to exist. Either the kernel "
                "check did not run or the transported coefficients are not on this "
                "group's EBR basis")
        derived = {}
        for i, value in enumerate(a):
            q = sympy.Rational(value)
            if q != 0:
                derived[z_rows[i]] = Fraction(int(q.p), int(q.q))
        printed = {int(k): Fraction(v[0], v[1]) for k, v in entry["theta"].items()
                   if Fraction(v[0], v[1]) != 0}
        if derived == printed:
            relation = "equal"
        elif _equivalent(theta, u, v, d, printed, derived, modulus):
            relation = "equivalent"
        else:
            relation = "different"
        entry["derived"] = {
            "theta": {str(k): [value.numerator, value.denominator]
                      for k, value in sorted(derived.items())},
            "source": source, "relation": relation}
        entry["derived_note"] = (
            "the unique rational a with a.Theta0 = c.BR, c being "
            + ("irreptables' named form of the same name, transported and checked to "
               "vanish on every EBR of this group"
               if source != "smith" else
               "the group's own single Smith coordinate u[slot].BR, which is the unique "
               "nonzero homomorphism onto Z_n and needs no name matching")
            + f"; the printed row is {relation}")
        stats["derived"][relation] += 1


def _check_kernel(sg, srsis, theta, matrix, where):
    r"""``ker(EBR) subset ker(Theta_Z)``, over the rationals, exactly.

    The paper's central structural claim is that the :math:`\mathbb{Z}`-valued
    SRSIs and the symmetry-data vector determine each other:
    :math:`B = \mathrm{BR}\cdot[\Theta^{(0)}]^\ddagger\cdot\theta_{\mathbb{Z}}`,
    which forces :math:`\ker\Theta^{(0)} = \ker \mathrm{BR}`.  Restricted to the
    EBR columns — a coordinate subspace — both kernels restrict with it, so the
    inclusion is checkable here against ``irreptables``' own EBR matrix, and it
    is the sharpest independent check the conversion has of the label map, the
    Wyckoff letters and the resolved column indices together: a single
    mis-resolved term would generically break it.

    Measured 2026-09-02 it holds with zero violations for all 217 + 217 groups,
    and the ranks agree as well (:math:`\operatorname{rank}\Theta^{(0)} =
    \operatorname{rank}\mathrm{BR}` on those columns), so the inclusion is an
    equality.  A violation is a hard error.
    """
    import sympy

    z_rows = [theta[srsi["index"] - 1] for srsi in srsis if srsi["mod"] is None]
    if not z_rows:
        raise ConversionError(f"{where}: no Z-valued SRSI in the row")
    for vector in sympy.Matrix(matrix).nullspace():
        for i, row in enumerate(z_rows):
            total = sum(sympy.Integer(int(row[j])) * vector[j] for j in range(len(row)))
            if total != 0:
                raise ConversionError(
                    f"{where}: a vector in the kernel of the EBR matrix is not in the "
                    f"kernel of the Z-valued SRSI row {i + 1} (it evaluates to {total}), "
                    "so the resolved SRSI matrix is not the paper's Theta^(0)")
    rank_ebr = sympy.Matrix(matrix).rank()
    rank_theta = sympy.Matrix(z_rows).rank()
    if rank_ebr != rank_theta:
        raise ConversionError(
            f"{where}: rank(BR) = {rank_ebr} but rank(Theta_Z) = {rank_theta} on the "
            "EBR columns; the paper's one-to-one map between Z-SRSIs and symmetry-data "
            "vectors requires them to agree")
    return int(rank_ebr)


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------

def dumps(document: dict) -> str:
    """Pretty provenance, one compact line per space group (small, and diffable)."""
    head = json.dumps(document["provenance"], indent=1, sort_keys=False,
                      ensure_ascii=False)
    lines = ['{', '"provenance": ' + head + ',', '"groups": {']
    keys = sorted(document["groups"], key=int)
    for i, key in enumerate(keys):
        body = json.dumps(document["groups"][key], separators=(",", ":"),
                          ensure_ascii=False)
        lines.append(f'{json.dumps(key)}: {body}' + ("," if i + 1 < len(keys) else ""))
    lines += ['}', '}', '']
    return "\n".join(lines)


def report(document: dict, valuedness: str) -> str:
    counts = document["provenance"]["counts"]
    lines = [f"--- {valuedness} ---"]
    for key, value in counts.items():
        lines.append(f"  {key}: {value}")
    lines.append(f"  omitted groups: {document['provenance']['omitted_groups']}")
    lines.append(f"  distinct site-symmetry irrep spellings: "
                 f"{len(document['provenance']['label_map'])}")
    for alias in document["provenance"]["label_aliases"]:
        lines.append(f"  alias by elimination: {alias['supplement']!r} -> "
                     f"{alias['ebr_label']!r} at site {alias['site']!r} "
                     f"({alias['positions']} positions)")
    for entry in document["provenance"]["composite_site_irreps"]:
        lines.append(f"  no EBR (composite induced BR): {entry['supplement']!r} at site "
                     f"{entry['site']!r} ({entry['positions']} positions)")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", help="the supplement's 'supple info' folder")
    parser.add_argument("--out", default=None,
                        help="directory to write srsi_{single,double}.json into")
    parser.add_argument("--dry-run", action="store_true",
                        help="parse, resolve and self-check, but write nothing")
    args = parser.parse_args(argv)

    status = tables_status()
    if not status["tables"]:
        print(f"irreptables is not loaded: {status['reason']}", file=sys.stderr)
        return 2

    written = []
    for valuedness in ("single", "double"):
        document = convert(args.folder, valuedness)
        print(report(document, valuedness))
        if args.dry_run:
            continue
        out_dir = args.out or os.path.join(_HERE, "..", "src", "pythtqc", "data")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"srsi_{valuedness}.json")
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(dumps(document))
        written.append(os.path.normpath(path))
    for path in written:
        print(f"wrote {path} ({os.path.getsize(path)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
