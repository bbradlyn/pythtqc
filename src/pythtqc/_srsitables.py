"""The tables of the stable real-space invariants of Hwang et al., as shipped:
each grey group's invariants over the site irreps at its Wyckoff positions, and
the paper's names for the symmetry indicators against ``irreptables``' own.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from fractions import Fraction
import functools
import itertools
import json
import os
import re
from typing import Any

import numpy as np

from . import _smith
from . import _spacegroup
from . import _tables
from .refusal import InternalError, Refusal

__all__ = ["DATA_DIR", "GATES", "SRSITables", "SRSITablesNumerical", "IndicatorCheck",
           "Provenance", "ProvenanceCounts", "SiteIrrep", "SplitEBR", "StablePair",
           "Invariants", "ModularValue", "srsi_status", "srsi_tables",
           "canonical_determinism"]


#: Where the converted tables live inside the installed package.
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

#: ``ebr_data`` position strings: ``"12d(31',3)"`` -> multiplicity, letter, symbols.
_POSITION = re.compile(r"^(\d+)([a-zA-Z])\((.*)\)$")

#: A Wyckoff position as a call writes it, ``"12d"``: multiplicity, letter.
_WYCKOFF = re.compile(r"^(\d+)([a-zA-Z])$")

#: The paper the shipped tables come from; the converted files repeat it in
#: their own provenance block, and every refusal that names a table carries it.
_DOI = "https://doi.org/10.1038/s41467-026-74844-w"

#: Every refusal the SRSI calls return, with what it measures, in the order a
#: calculation meets them: the tables, the arithmetic, the two routes onto a
#: model, the criterion.  Those passed on from elsewhere are named where they
#: are measured: ``ebr_tables``, ``irrep_tables`` and ``ebr_convention`` by
#: :func:`..groups.ebr_data`, and on the bands' route those of
#: :func:`..ebr.symmetry_indicator` and :func:`..sewing.hsp_table`.
GATES = {
    "srsi_data":
        "the converted SRSI tables are not installed -- see srsi_status()",
    "srsi_scope":
        "the group is not grey (type II), or the number is outside 1..230; the "
        "shipped tables are the time-reversal-symmetric ones",
    "srsi_table_absent":
        "one of the thirteen space groups the tables omit (1, 4, 7, 9, 19, 29, 33, "
        "76, 78, 144, 145, 169, 170), whose only SRSI is the band count",
    "srsi_ebr_basis":
        "the installed irreptables carries a different EBR list than the conversion "
        "resolved against, so the stored column indices would name the wrong EBRs",
    "srsi_realizations_truncated":
        "the enumeration of atomic realizations hit its cap, and a no-go is asserted "
        "only on a complete enumeration",
    "wyckoff_orbit_content_incomplete":
        "a high-symmetry point refused for a reason other than a trivial little "
        "group, so the symmetry data vector measured on a Wyckoff orbit is not whole",
    "wyckoff_orbit_not_localizable":
        "the symmetry data vector of a Wyckoff orbit is a band representation of no "
        "position on the closure of its variety, and no composite site irrep "
        "explains it either",
    "wyckoff_setting":
        "geometry and the symmetry data vector name different Wyckoff positions "
        "(srsi_of_model), or pyxtal's list is not in the tables' setting "
        "(wyckoff_sites_tables)",
    "wyckoff_unresolved":
        "a position lies on no Wyckoff variety within tol -- a guard, since the "
        "general position's variety is the whole cell",
    "tables_not_ready":
        "the transformation to the tables' conventional cell was never found, so no "
        "position can be carried into the reference cell the tables are written in",
    "irreptables_indicator_tables":
        "irreptables' own indicator table could not be read, so the two "
        "vocabularies cannot be paired for this group",
}


# ---------------------------------------------------------------------------
# how a summary writes a group, a set of invariants, a sum of site irreps
# ---------------------------------------------------------------------------

def _count(n, noun) -> str:
    """``"1 row"``, ``"3 rows"``: a count and its noun, in the number the count takes."""
    return f"{n} {noun}" + ("" if n == 1 else "s")


def _valuedness(spinor) -> str:
    """Which of a group's two tables: ``"double-valued"`` with spin, else ``"single-valued"``."""
    return "double-valued" if spinor else "single-valued"


@functools.lru_cache(maxsize=None)
def _grey_symbol(number_str: str, spinor: bool) -> str:
    """The grey group's Hermann-Mauguin symbol, ``"I4321'"``, as irreptables' table of it names it.

    Empty where that table cannot be read, and a summary then names the group by
    its number alone.
    """
    try:
        return str(_spacegroup._irrep_table(number_str, spinor, True).name)
    except (ImportError, OSError):              # pragma: no cover - tables not installed
        return ""


def _label(group) -> str:
    """``P6mm1' #183.186``: a group as a summary names it, by the tables' symbol and its number."""
    symbol = _grey_symbol(str(group.number_str), bool(group.spinor)) or str(group.name or "")
    return (f"{symbol} " if symbol else "") + f"#{group.number_str}"


def _group(group) -> str:
    """``P6mm1' #183.186, double-valued``: the group a summary is of, and which of its tables."""
    return f"{_label(group)}, {_valuedness(group.spinor)}"


def _values(values) -> str:
    """``(1, -1/2, 0)``: invariants as a summary prints them, a lone one ``(0)``."""
    return "(" + ", ".join(str(v) for v in values) + ")"


def _span(names) -> str:
    """``theta_1 to theta_3``, ``theta_4, theta_5``, ``theta_5``: consecutive invariants named."""
    names = list(names)
    if len(names) >= 3:
        return f"{names[0]} to {names[-1]}"
    return ", ".join(names)


def _and(items) -> str:
    """``a``, ``a and b``, ``a, b and c``."""
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _zn_groups(theta_zn) -> list:
    """``[(n, names, values), ...]``: the invariants valued modulo n, a group per modulus, in the tables' order."""
    out: list = []
    for name, entry in theta_zn.items():
        if out and out[-1][0] == int(entry.modulus):
            out[-1][1].append(name)
            out[-1][2].append(entry.value)
        else:
            out.append((int(entry.modulus), [name], [entry.value]))
    return out


def _zn_text(theta_zn) -> str:
    """``theta_4, theta_5 = (0, 1) mod 2``: each modulus its own clause."""
    return "; ".join(f"{_span(names)} = {_values(values)} mod {n}"
                     for n, names, values in _zn_groups(theta_zn))


def _invariants_text(theta_z, theta_zn=None) -> str:
    """``theta_1 to theta_3 = (2, 2, 0); theta_4, theta_5 = (0, 1) mod 2``."""
    text = f"{_span(theta_z)} = {_values(theta_z.values())}"
    return text + (f"; {_zn_text(theta_zn)}" if theta_zn else "")


def _zn_header(theta_zn) -> str:
    """``(theta_4, theta_5) mod 2``: the invariants valued modulo n a column of values holds."""
    return _and([f"({_span(names)}) mod {n}" for n, names, _v in _zn_groups(theta_zn)])


def _zn_values(theta_zn) -> str:
    """``(1, 1)``: the values under :func:`_zn_header`, a group per modulus."""
    return "; ".join(_values(values) for _n, _names, values in _zn_groups(theta_zn))


def _indicator_line(indicator, whose) -> str:
    """A summary's line for the symmetry-indicator class of a band set."""
    if not indicator.divisors:
        return f"  {whose} symmetry-indicator group is trivial   (.indicator)"
    if not any(indicator.values):
        return (f"  {whose} symmetry indicator: class zero in {indicator.indicator_group}"
                "   (.indicator)")
    return (f"  {whose} symmetry indicator: class {indicator.values} in "
            f"{indicator.indicator_group}, of order {indicator.order}   (.indicator)")


def _pair_text(pair) -> str:
    """``B2 and B3 at 12d``: a pair no invariant tells apart."""
    return f"{' and '.join(pair.irreps)} at {pair.wyckoff}"


def _signed(terms) -> str:
    """``2 x - 4 y + 1/2 z``: a sum of ``(coefficient, symbol)``, its signs written out."""
    out = []
    for coefficient, symbol in terms:
        coefficient = Fraction(coefficient)
        sign = ("-" if coefficient < 0 else "") if not out else (
            " - " if coefficient < 0 else " + ")
        out.append(f"{sign}{abs(coefficient)} {symbol}")
    return "".join(out) or "0"


def _formula(row) -> str:
    """``2 (-1E2E)_e + 1 (-E)_a``: an invariant's terms, each a site irrep at its Wyckoff letter times its multiplicity.

    A site irrep is spelled as the list of elementary band representations
    spells it where it names one, ``A`` at a ``222`` site.
    """
    return _signed((term["coef"], f"({term.get('ebr_label') or term['irrep']})_{term['letter']}")
                   for term in row.terms)


def _is_sum(p) -> bool:
    """Whether ``p`` is a sum of term rows: an iterable, not a mapping or a string, of rows with ``multiplicity``, ``irrep`` and ``wyckoff``."""
    if isinstance(p, (Mapping, str, bytes)):
        return False
    try:
        rows = list(p)
    except TypeError:
        return False
    return all(hasattr(t, "multiplicity") and hasattr(t, "irrep") and hasattr(t, "wyckoff")
               for t in rows)


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModularValue:
    """One invariant valued modulo n: its value and the modulus.

    An entry of every ``theta_Zn``.

    Attributes
    ----------
    value : int or Fraction
        The value, reduced modulo :attr:`modulus`; a fraction, unreduced, only
        where :meth:`SRSITables.theta` was handed a rational site-symmetry
        vector and the value is not a whole number.
    modulus : int
        The modulus n.
    """

    value: Any
    modulus: int


@dataclass(frozen=True)
class Invariants:
    r"""The stable real-space invariants of a site-symmetry vector, by name.

    What :meth:`SRSITables.theta` returns.

    Attributes
    ----------
    theta_Z : dict
        ``{name: value}``: each integer-valued invariant, an exact fraction, in
        the tables' order; whole numbers exactly where the symmetry indicator is
        zero.
    theta_Zn : dict
        ``{name: ModularValue}``: each invariant valued modulo n.  They belong to
        the site-symmetry vector handed in, not to its symmetry data vector.
    """

    theta_Z: dict
    theta_Zn: dict

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return f"Invariants({_invariants_text(self.theta_Z, self.theta_Zn)})"


@dataclass(frozen=True)
class _SRSIRow:
    r"""One invariant :math:`\theta_i` of a group: its formula over the site irreps.

    ``index`` is the paper's, from 1; ``mod`` is ``n`` for one valued modulo n
    and ``None`` for an integer-valued one; ``coefficients`` is the formula on
    the elementary band representations, one per entry of ``tables._ebrs``;
    ``terms`` is every term of the formula, ``{"coef", "irrep", "irrep_tex",
    "letter", "column", "maximal", "ebr"}``, the site irrep as the paper spells
    it and ``ebr_label`` where the list spells it otherwise, ``column`` ``None``
    for a site irrep that induces no elementary band representation.
    """

    index: int
    name: str
    mod: int | None
    coefficients: tuple
    terms: tuple = field(repr=False)


@dataclass(frozen=True)
class _SRSIIndicator:
    r"""One symmetry indicator as the paper tabulates it: a form in the :math:`\theta_i`, modulo its own modulus.

    ``theta`` is the printed form ``{i: coefficient}``; ``derived``, where there
    is one, the form equal to ``irreptables``' of the same name or to the
    group's one Smith coordinate, with ``relation`` and ``source``;
    ``ebr_failures`` the ``(column, value)`` of each elementary band
    representation the printed form does not vanish on.
    """

    name: str
    name_tex: str
    mod: int
    theta: dict
    vanishes_on_ebrs: bool = True
    ebr_failures: tuple = ()
    derived: dict | None = None
    relation: str | None = None
    source: str | None = None
    derived_note: str = ""

    @property
    def coefficients(self) -> dict:
        """The form :meth:`SRSITables.indicators_of` evaluates: the printed one, or the derived one where :attr:`relation` is ``"different"``."""
        if self.relation == "different" and self.derived is not None:
            return dict(self.derived)
        return dict(self.theta)


@dataclass(frozen=True)
class IndicatorCheck:
    """One symmetry indicator of the tables held against the group's elementary band representations and against irreptables.

    An entry of ``tables.numerical.indicators``.

    Attributes
    ----------
    relation : str or None
        The printed form against the derived one: ``"equal"``, ``"equivalent"``
        (other coefficients, the same values on every band set modulo the
        indicator's modulus) or ``"different"``; ``None`` where none is derived.
    source : str or None
        What the derived form was derived from, ``"irreptables:<name>"`` or
        ``"smith"``.
    derived : str or None
        The derived form in the invariants, written as ``tables.indicators``
        writes the printed one.  Where :attr:`relation` is ``"different"`` it is
        the form ``indicators_of`` evaluates.
    derived_note : str
        The derivation in a sentence, or why there is none.
    ebr_failures : tuple
        ``(site irrep, Wyckoff position, value)`` of each elementary band
        representation the printed form does not vanish on,
        ``("A1", "1a", "1/2")``.
    vanishes_on_ebrs : bool
        Whether the printed form is zero modulo its modulus on every elementary
        band representation of the group, as an indicator is.
    """

    relation: str | None
    source: str | None
    derived: str | None
    derived_note: str
    ebr_failures: tuple
    vanishes_on_ebrs: bool


@dataclass(frozen=True)
class SRSITablesNumerical:
    """How each symmetry indicator of the tables was checked.

    ``tables.numerical``.

    Attributes
    ----------
    indicators : dict
        ``{name: IndicatorCheck}`` for each symmetry indicator.
    """

    indicators: dict


@dataclass(frozen=True)
class ProvenanceCounts:
    """What the conversion of the paper's tables counted, for one valuedness.

    ``tables.provenance.counts``.  Each field is a count over the 217 tabulated
    groups of that valuedness, ``zn_rows``, ``si_moduli`` and
    ``derived_indicator_rows`` by modulus or relation.
    """

    groups: int
    srsi_rows: int
    zn_rows: dict
    terms: int
    terms_on_ebr_columns: int
    terms_at_maximal_wyckoff_without_an_ebr: int
    terms_at_nonmaximal_wyckoff: int
    alias_positions: int
    composite_positions: int
    si_groups: int
    si_expressions: int
    si_moduli: dict
    split_groups: int
    splittings: int
    splittings_srsi_mismatch: int
    splittings_srsi_match: int
    ebr_columns: int
    si_on_ebr_checks: int
    si_expressions_not_vanishing_on_ebrs: int
    si_failing_groups: tuple
    kernel_checks: int
    derived_indicator_rows: dict


@dataclass(frozen=True)
class Provenance:
    """Where the tables come from: the paper, and the versions of the tables they were converted against.

    ``tables.provenance``.

    Attributes
    ----------
    paper, doi : str
        The paper the tables are converted from, and its DOI.
    valuedness : str
        ``"single"`` or ``"double"``.
    converted : str
        When the tables were converted.
    irrep_version, irreptables_version : str
        The versions of ``irrep`` and ``irreptables`` the elementary band
        representations were resolved against.
    omitted_groups : tuple of int
        The thirteen space groups the tables leave out.
    omitted_reason : str
        Why, in the words of the paper's caption.
    label_map : dict
        ``{TeX spelling: spelling}``: the paper's site irreps as the package
        spells them.
    label_aliases, composite_site_irreps : tuple
        The site irreps the paper and the list spell differently, and those
        whose band representation is composite.
    counts : ProvenanceCounts
        What the conversion counted.
    notes : tuple of str
        What the conversion says about the tables, a sentence each.
    """

    paper: str
    doi: str
    valuedness: str
    converted: str
    irrep_version: str
    irreptables_version: str
    omitted_groups: tuple
    omitted_reason: str
    label_map: dict
    label_aliases: tuple
    composite_site_irreps: tuple
    counts: ProvenanceCounts
    notes: tuple


@dataclass(frozen=True)
class SiteIrrep:
    r"""A site irrep at a Wyckoff position: one coordinate of the site-symmetry vector ``p``.

    A site irrep at a maximal position induces an elementary band
    representation, unless that band representation is a sum of others
    (:math:`{}^1E{}^2E` at site symmetry :math:`\bar4`, ``E`` at
    :math:`\bar4m2` and :math:`\bar42m`); one at a non-maximal position induces
    a composite one.  An entry of ``tables.site_irreps``.

    Attributes
    ----------
    irrep : str
        The site irrep, as the list of elementary band representations spells
        it where it names one, ``A`` where the paper writes ``A1`` at a ``222``
        site, and as the paper spells it elsewhere.
    wyckoff : str
        The Wyckoff position in the setting the tables are written in, with its
        multiplicity in the conventional cell, ``"4d"``.
    maximal : bool
        Whether the position is maximal.
    ebr : bool
        Whether the site irrep induces an elementary band representation.
    """

    irrep: str
    wyckoff: str
    maximal: bool
    ebr: bool
    # the paper's spelling, the letter alone, and the column of the elementary
    # band representation in the group's list
    _paper: str = field(default="", repr=False)
    _letter: str = field(default="", repr=False)
    _column: int | None = field(default=None, repr=False)

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return (f"SiteIrrep({self.irrep} at {self.wyckoff}, "
                f"{'maximal' if self.maximal else 'not maximal'}, "
                + ("an elementary band representation" if self.ebr
                   else "no elementary band representation") + ")")


@dataclass(frozen=True)
class SplitEBR:
    r"""One splitting of an elementary band representation into others, as the paper lists it.

    The paper lists the splittings the symmetry indicators allow, every part an
    elementary band representation, and marks whether the invariants valued
    modulo n of the parts add up to the whole's.  An entry of
    ``tables.split_ebrs``; :meth:`summary` prints it.

    Attributes
    ----------
    parent : SiteIrrep
        The whole, the same entry as in ``tables.site_irreps``.
    children : tuple of SiteIrrep
        The parts, two or more.
    label : str
        The splitting as the paper writes it, ``"(-E)_c = (-E2)_a + (-E1)_a"``.
    srsi_mismatch : bool
        The paper's mark: ``True`` where the parts' invariants valued modulo n
        do not add up to the whole's.
    """

    parent: SiteIrrep
    children: tuple
    label: str
    srsi_mismatch: bool

    def summary(self) -> str:
        """The splitting, and whether the invariants of its parts add up to the whole's, a line each."""
        return (f"{self.label}, a splitting the symmetry indicators allow, each term the "
                "elementary band representation of a site irrep at a Wyckoff position"
                "   (.label)\n"
                "  the invariants valued modulo n of the parts "
                f"{'do not add' if self.srsi_mismatch else 'add'} up to the whole's"
                "   (.srsi_mismatch)")

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return f"SplitEBR({self.label}, srsi_mismatch={self.srsi_mismatch})"


@dataclass(frozen=True)
class StablePair:
    """Two site irreps at one maximal Wyckoff position that no stable real-space invariant distinguishes.

    Their elementary band representations have the same symmetry data vector,
    and they carry the same invariants, so nothing the package measures tells
    them apart; the paper's example is ``B2`` and ``B3`` at ``12d`` of
    ``I432``.  An entry of ``tables.stably_equivalent``; :meth:`summary` prints
    it.

    Attributes
    ----------
    irreps : tuple of str
        The two site irreps, as the list of elementary band representations
        spells them.
    wyckoff : str
        The Wyckoff position, ``"12d"``.
    symmetry_data : dict
        ``{corep: multiplicity}``: the symmetry data vector both induce.
    theta_Z : dict
        ``{name: value}``: the integer-valued invariants both carry.
    theta_Zn : dict
        ``{name: ModularValue}``: the invariants valued modulo n both carry.
    """

    irreps: tuple
    wyckoff: str
    symmetry_data: dict = field(repr=False, default_factory=dict)
    theta_Z: dict = field(default_factory=dict)
    theta_Zn: dict = field(default_factory=dict)
    # the two elementary band representations' names and columns in the list
    _names: tuple = field(default=(), repr=False)
    _columns: tuple = field(default=(), repr=False)

    def summary(self) -> str:
        """The pair, then what both members share, a line each."""
        return "\n".join([
            f"{_pair_text(self)}, which no stable real-space invariant distinguishes"
            "   (.irreps, .wyckoff)",
            "  their elementary band representations have the same symmetry data vector"
            "   (.symmetry_data)",
            f"  the invariants both carry: {_invariants_text(self.theta_Z, self.theta_Zn)}"
            + ("   (.theta_Z, .theta_Zn)" if self.theta_Zn else "   (.theta_Z)")])

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return f"StablePair({'/'.join(self.irreps)} at {self.wyckoff})"


@dataclass(frozen=True)
class SRSITables:
    r"""The stable real-space invariants of one grey space group, as the paper tabulates them.

    What :func:`srsi_tables` returns; :meth:`summary` prints the invariants, a
    line each, and :meth:`formulas` each one's formula over the site irreps.
    :meth:`theta` evaluates them on a site-symmetry vector ``p``, and
    :meth:`indicators_of` evaluates the paper's symmetry indicators.

    Attributes
    ----------
    group : ..groups.GroupData
        The group the tables were read for: the measured group handed to
        :func:`srsi_tables`, or, from a space-group number, the grey group with
        its name, number, valuedness and magnetic type, and no cell
        (``periodic_dirs`` is ``None``).
    invariants : dict
        ``{name: formula}``: each invariant, in the paper's order, as a sum of
        terms ``c (irrep)_w``, ``c`` times the multiplicity of a site irrep at
        Wyckoff position ``w``: ``tables.invariants["theta_4"]`` is
        ``"1 (-E2)_b"``.
    indicators : dict
        ``{name: form}``: each of the paper's symmetry indicators as a form in
        the invariants, ``"-2 theta_1 + 2 theta_2 + 2 theta_5"``, as the paper
        prints it.
    moduli : dict
        ``{name: modulus}``: each invariant's and each indicator's modulus,
        ``None`` for an integer-valued invariant.
    site_irreps : tuple of SiteIrrep
        Every site irrep at a Wyckoff position the formulas count, maximal or
        not: the coordinates of ``p``.
    split_ebrs : tuple of SplitEBR
        The paper's splittings of an elementary band representation into
        others.
    stably_equivalent : tuple of StablePair
        The site irreps at one maximal position that no invariant tells apart.
    provenance : Provenance
        Where the tables come from.
    numerical : SRSITablesNumerical
        How each symmetry indicator was checked.
    """

    group: Any = field(compare=False)
    invariants: dict
    indicators: dict
    moduli: dict
    site_irreps: tuple
    split_ebrs: tuple
    provenance: Any = field(repr=False)
    numerical: Any = field(repr=False)
    # the space-group number, the tables' label for the grey group and which
    # of its tables; the invariants as rows; the formulas on the columns, the
    # elementary band representations of the group's list as (irrep, letter);
    # the indicators as rows; the terms on site irreps that induce no
    # elementary band representation; each letter's multiplicity
    _sg: int = field(repr=False)
    _number_str: str = field(repr=False)
    _spinor: bool = field(repr=False)
    _rows: tuple = field(repr=False)
    _theta_on_columns: tuple = field(repr=False)
    _ebrs: tuple = field(repr=False)
    _indicator_rows: tuple = field(repr=False)
    _terms_without_ebr_column: tuple = field(repr=False)
    _multiplicities: dict = field(repr=False, compare=False)

    # -- the shape of the invariant group ----------------------------------
    @property
    def _names(self) -> tuple:
        """Every invariant's name, in the paper's order."""
        return tuple(row.name for row in self._rows)

    @property
    def _z_rows(self) -> tuple:
        r"""Where the integer-valued invariants are in :attr:`_rows`."""
        return tuple(i for i, row in enumerate(self._rows) if row.mod is None)

    @property
    def _zn_rows(self) -> dict:
        r"""``{n: (row indices,)}``: where the invariants valued modulo n are in :attr:`_rows`, by modulus."""
        out: dict[int, list] = {}
        for i, row in enumerate(self._rows):
            if row.mod is not None:
                out.setdefault(int(row.mod), []).append(i)
        return {n: tuple(v) for n, v in sorted(out.items())}

    @property
    def stably_equivalent(self) -> tuple:
        """The site irreps at one maximal Wyckoff position that no stable real-space invariant distinguishes."""
        return _stable_pairs(int(self._sg), bool(self._spinor))

    # -- the arithmetic ----------------------------------------------------
    def theta(self, p) -> Invariants:
        r"""The stable real-space invariants of a site-symmetry vector ``p``.

        Parameters
        ----------
        p : mapping or sum of term rows
            ``{(site irrep, Wyckoff position): multiplicity}`` over any site
            irreps of the group, at maximal positions or not, the position
            written with its multiplicity, ``("-E", "2c")``, and the site irrep
            spelled as the list of elementary band representations or as the
            paper spells it; or a sum of term rows, each with ``multiplicity``,
            ``irrep`` and ``wyckoff``, such as a candidate sum
            :func:`..ebr.decompose` returns, an atomic realization or a Wyckoff
            orbit's ``terms``.  Integer multiplicities for an atomic
            configuration, rational ones for a band set with a nonzero
            symmetry indicator.

        Returns
        -------
        Invariants
            ``out.theta_Z`` holds the integer-valued invariants by name, exact
            fractions, whole numbers exactly where the symmetry indicator is
            zero; ``out.theta_Zn`` the invariants valued modulo n.  These belong
            to ``p``, not to its symmetry data vector: they are a band set's
            only for a configuration that is known.

        Raises
        ------
        TypeError
            ``p`` is neither a mapping nor a sum of term rows.
        ValueError
            A position written without its multiplicity or with another one
            than the tables give it, or a site irrep the tables do not carry
            there; the message lists those they carry.

        Examples
        --------
        >>> tables.theta({("-E", "2c"): 1})                    # doctest: +SKIP
        Invariants(theta_1 to theta_3 = (1, 1, 0); theta_4, theta_5 = (0, 1) mod 2)
        """
        return self._invariants(self._theta_of_sites("theta", p))

    def theta_unreduced(self, p) -> dict:
        r"""Every invariant of ``p``, by name, those valued modulo n not reduced.

        Parameters
        ----------
        p : mapping or sum of term rows
            As :meth:`theta` takes it.

        Returns
        -------
        dict
            ``{name: value}`` for every invariant, in the paper's order, exact
            fractions.

        Raises
        ------
        TypeError, ValueError
            As :meth:`theta` raises them.
        """
        raw = self._theta_of_sites("theta_unreduced", p)
        return {row.name: value for row, value in zip(self._rows, raw)}

    def indicators_of(self, p) -> dict:
        """The paper's symmetry indicators of ``p``, each modulo its own modulus.

        Each form is the one ``tables.indicators`` prints, but for the four
        forms of space groups 147, 162 and 163 without spin-orbit coupling that
        do not vanish on their own group's elementary band representations,
        where it is the derived one ``tables.numerical.indicators`` holds.

        Parameters
        ----------
        p : mapping or sum of term rows
            As :meth:`theta` takes it.

        Returns
        -------
        dict
            ``{name: value}``: an integer modulo the indicator's modulus, or an
            unreduced fraction where the value is not a whole number.

        Raises
        ------
        TypeError, ValueError
            As :meth:`theta` raises them.
        """
        return self._indicators_of_raw(self._theta_of_sites("indicators_of", p))

    def _invariants(self, raw) -> Invariants:
        """The :class:`Invariants` of the unreduced rows ``raw``."""
        theta_z = {self._rows[i].name: raw[i] for i in self._z_rows}
        theta_zn = {}
        for i, row in enumerate(self._rows):
            if row.mod is not None:
                n = int(row.mod)
                theta_zn[row.name] = ModularValue(
                    int(raw[i]) % n if raw[i].denominator == 1 else raw[i], n)
        return Invariants(theta_Z=theta_z, theta_Zn=theta_zn)

    def _theta_columns(self, p) -> Invariants:
        """:meth:`theta` on a ``p`` over the columns, the group's elementary band representations in the order of ``tables._ebrs``."""
        return self._invariants(self._theta_of_columns(p))

    def _theta_unreduced_columns(self, p) -> tuple:
        """Every invariant of a ``p`` over the columns, unreduced, one per row in the paper's order."""
        return self._theta_of_columns(p)

    def _indicators_of_columns(self, p) -> dict:
        """:meth:`indicators_of` on a ``p`` over the columns."""
        return self._indicators_of_raw(self._theta_of_columns(p))

    def _theta_of_columns(self, p) -> tuple:
        """The unreduced rows of a ``p`` over the columns."""
        vec = _as_fractions(p, len(self._ebrs))
        return tuple(sum((Fraction(c) * v for c, v in zip(row, vec)), Fraction(0))
                     for row in self._theta_on_columns)

    def _indicators_of_raw(self, raw) -> dict:
        """:meth:`indicators_of`' arithmetic on the unreduced rows."""
        out = {}
        for form in self._indicator_rows:
            total = sum((c * raw[i - 1] for i, c in form.coefficients.items()),
                        Fraction(0))
            out[form.name] = (int(total) % form.mod if total.denominator == 1
                              else total)
        return out

    # -- the full site-symmetry vector -------------------------------------
    def _site_lookup(self) -> dict:
        """``{(spelling, letter): (paper's spelling, letter)}``: both spellings of a site irrep."""
        out = {}
        for site in self.site_irreps:
            out[(site._paper, site._letter)] = (site._paper, site._letter)
            out[(site.irrep, site._letter)] = (site._paper, site._letter)
        return out

    def _position(self, call, position) -> str:
        """The letter of ``position``, written ``"1a"``; ``ValueError`` where it is not one of the tables' positions."""
        position = str(position)
        letters = self._multiplicities
        match = _WYCKOFF.match(position)
        if match is None:
            if re.fullmatch(r"[a-zA-Z]", position):
                written = (f"'{letters[position]}{position}'" if position in letters
                           else "such as '1a'")
                raise ValueError(f"{call}(): the position {position!r} is written with its "
                                 f"multiplicity, {written}")
            raise ValueError(f"{call}(): {position!r} is no Wyckoff position; a position is "
                             "written with its multiplicity, such as '1a'")
        multiplicity, letter = int(match.group(1)), match.group(2)
        carried = sorted({site._letter for site in self.site_irreps})
        if letter not in carried:
            raise ValueError(
                f"{call}(): #{self._number_str} has no Wyckoff position {position} in its "
                "tables; they carry "
                + ", ".join(f"{letters.get(c, '')}{c}" for c in carried))
        if letters.get(letter) != multiplicity:
            raise ValueError(f"{call}(): #{self._number_str} has Wyckoff position "
                             f"{letters.get(letter)}{letter}, not {position}")
        return letter

    def _theta_of_sites(self, call, p) -> tuple:
        """The unreduced rows of ``p``, a mapping keyed ``(site irrep, "1a")`` or a sum of term rows."""
        if isinstance(p, Mapping):
            items = list(p.items())
        elif _is_sum(p):
            items = [((t.irrep, t.wyckoff), t.multiplicity) for t in p]
        else:
            raise TypeError(
                f"{call}(): p is a mapping {{(site irrep, Wyckoff position): multiplicity}}, "
                "the position written '1a', or a sum of term rows; got a "
                f"{type(p).__name__}")
        lookup = self._site_lookup()
        vector: dict = {}
        for key, value in items:
            if not (isinstance(key, tuple) and len(key) == 2):
                raise ValueError(
                    f"{call}(): the site-symmetry vector is keyed by (site irrep, Wyckoff "
                    "position) pairs, the position written '1b', or is a sum of term rows; "
                    f"got {key!r}")
            letter = self._position(call, key[1])
            resolved = lookup.get((str(key[0]), letter))
            if resolved is None:
                here = sorted({site.irrep for site in self.site_irreps
                               if site._letter == letter})
                raise ValueError(
                    f"{call}(): #{self._number_str} ({_valuedness(self._spinor)}) has no "
                    f"site irrep {str(key[0])!r} at {key[1]}; at {key[1]} the tables carry "
                    f"{here}")
            vector[resolved] = vector.get(resolved, Fraction(0)) + Fraction(value)
        return self._raw_of(vector)

    def _theta_at(self, irreps, letter) -> Invariants:
        """:meth:`theta` of ``{irrep: multiplicity}`` at one letter, the irreps in either spelling."""
        lookup = self._site_lookup()
        vector: dict = {}
        for irrep, value in dict(irreps).items():
            resolved = lookup.get((str(irrep), str(letter)))
            if resolved is None:
                raise ValueError(
                    f"#{self._number_str} ({_valuedness(self._spinor)}) has no site irrep "
                    f"{str(irrep)!r} at Wyckoff letter {str(letter)!r}")
            vector[resolved] = vector.get(resolved, Fraction(0)) + Fraction(value)
        return self._invariants(self._raw_of(vector))

    def _raw_of(self, vector) -> tuple:
        """The unreduced rows of ``{(paper's spelling, letter): multiplicity}``."""
        out = []
        for row in self._rows:
            total = Fraction(0)
            for term in row.terms:
                got = vector.get((str(term["irrep"]), str(term["letter"])))
                if got:
                    total += Fraction(int(term["coef"])) * got
            out.append(total)
        return tuple(out)

    # -- the basis of the atomic realizations: the maximal positions -------
    @property
    def _maximal_keys(self) -> tuple:
        r"""Every maximal site irrep, with the bands one copy of it carries: the coordinates :func:`..srsi.atomic_realizations` lists configurations over.

        The site irreps with an elementary band representation come first, in
        the order of ``tables._ebrs``, then the maximal ones whose band
        representation is a sum of others.
        """
        return _maximal_basis(int(self._sg), bool(self._spinor))[0]

    @property
    def _band_count_form(self) -> tuple:
        r"""The band count as a form in the integer-valued invariants: the ``a`` with ``a . theta_Z`` the number of bands, one entry per integer-valued invariant."""
        return _maximal_basis(int(self._sg), bool(self._spinor))[2]

    def summary(self) -> str:
        """The invariants, a line each with its modulus, and what else the tables hold, a line each."""
        lines = [f"{_group(self.group)}: "
                 f"{_count(len(self.invariants), 'stable real-space invariant')}"
                 "   (.group, .invariants)",
                 "  each invariant, and its modulus where it is valued modulo n"
                 "   (.invariants, .moduli)"]
        lines += [f"  {name:>9s}  "
                  f"{'Z' if self.moduli[name] is None else 'Z%d' % self.moduli[name]:>3s}"
                  for name in self.invariants]
        n_ebr = sum(1 for site in self.site_irreps if site.ebr)
        lines.append(f"  {_count(len(self.site_irreps), 'site irrep')} at Wyckoff positions, "
                     f"{n_ebr} of them inducing an elementary band representation"
                     "   (.site_irreps)")
        lines.append("  " + (f"{_count(len(self.indicators), 'symmetry indicator')}, each "
                             "a form in the theta_i: "
                             + ", ".join(f"{name} (mod {self.moduli[name]})"
                                         for name in self.indicators)
                             + "   (.indicators, .moduli)"
                             if self.indicators else
                             "no symmetry indicator, the group's indicator group being "
                             "trivial   (.indicators)"))
        if self.split_ebrs:
            lines.append(f"  {len(self.split_ebrs)} splitting"
                         f"{'' if len(self.split_ebrs) == 1 else 's'} of an elementary band "
                         "representation into others that the symmetry indicators allow"
                         "   (.split_ebrs)")
        pairs = self.stably_equivalent
        if pairs:
            lines.append("  site irreps no stable real-space invariant distinguishes: "
                         + "; ".join(_pair_text(pair) for pair in pairs)
                         + "   (.stably_equivalent)")
        return "\n".join(lines)

    def formulas(self) -> str:
        r"""Every invariant's formula, a line each, as the paper tabulates them, under a line saying what a term is.

        ``theta_2 (Z2) = 1 (-E)_b``: the invariant and its modulus, then its
        terms, each a site irrep at its Wyckoff letter times its multiplicity.
        """
        heads = [f"{name} ({'Z' if self.moduli[name] is None else f'Z{self.moduli[name]}'})"
                 for name in self.invariants]
        width = max((len(head) for head in heads), default=0)
        return "\n".join(
            ["each invariant, Z or Z_n by its modulus, as a sum of terms c (irrep)_w, c times "
             "the multiplicity of a site irrep at Wyckoff position w   (.invariants, .moduli)"]
            + [f"  {head:<{width}} = {formula}"
               for head, formula in zip(heads, self.invariants.values())])

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return (f"SRSITables({_label(self.group)}, {_valuedness(self._spinor)}, "
                f"{_count(len(self.invariants), 'invariant')}, "
                f"{_count(len(self.indicators), 'indicator')}, "
                f"{_count(len(self.split_ebrs), 'split EBR')})")


def _split(tables, invariants) -> tuple:
    """``(theta_Z, {n: values})`` of :class:`Invariants`, each in the tables' order, the values valued modulo n grouped by modulus."""
    theta_z = tuple(invariants.theta_Z.values())
    theta_zn = {n: tuple(invariants.theta_Zn[tables._rows[i].name].value for i in rows)
                for n, rows in tables._zn_rows.items()}
    return theta_z, theta_zn


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def srsi_status() -> dict[str, Any]:
    """Whether the stable real-space invariant tables are installed, and where.

    They ship with the package; an installation without them refuses
    ``srsi_data`` on every call here.

    Returns
    -------
    dict
        ``{"data", "single", "double", "dir", "doi", "reason"}``: ``data`` is
        ``True`` when both tables are in place, ``single`` and ``double`` are their
        paths, and ``reason`` says what is missing.
    """
    info: dict[str, Any] = {"data": False, "single": None, "double": None,
                            "dir": DATA_DIR, "doi": _DOI, "reason": None}
    missing = []
    for valuedness in ("single", "double"):
        path = os.path.join(DATA_DIR, f"srsi_{valuedness}.json")
        if os.path.isfile(path):
            info[valuedness] = path
        else:
            missing.append(os.path.basename(path))
    if missing:
        info["reason"] = (
            f"the SRSI tables {missing} are not installed in {DATA_DIR}. They ship "
            "with the package as its data, so this installation is missing them: "
            "reinstall pythTQC")
    else:
        info["data"] = True
    return info


@functools.lru_cache(maxsize=2)
def _document(valuedness: str) -> dict:
    path = os.path.join(DATA_DIR, f"srsi_{valuedness}.json")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _refuse_no_data(status) -> Refusal:
    return Refusal("srsi_data", status["reason"],
                   context={k: v for k, v in status.items() if k != "reason"})


def _as_fractions(p, n) -> tuple:
    values = list(p)
    if len(values) != n:
        raise ValueError(f"the site-symmetry vector has {len(values)} entries against "
                         f"{n} elementary band representations; it is indexed by the "
                         "group's own list of elementary band representations")
    return tuple(Fraction(v) for v in values)


def _lettered(entry: dict) -> dict:
    """One term or site of the converted data, its Wyckoff letter under ``letter``."""
    return {("letter" if key == "wyckoff" else key): value for key, value in entry.items()}


def _tupled(value):
    """A list of the converted data as an immutable sequence, anything else as it is."""
    return tuple(value) if isinstance(value, list) else value


@functools.lru_cache(maxsize=2)
def _provenance(valuedness: str) -> Provenance:
    """The provenance block of one of the two documents, as its record."""
    block = dict(_document(valuedness)["provenance"])
    counts = ProvenanceCounts(**{key: _tupled(value)
                                 for key, value in dict(block.pop("counts")).items()})
    return Provenance(counts=counts, **{key: _tupled(value) for key, value in block.items()})


def _build(record: dict, provenance: Provenance, group) -> SRSITables:
    n_ebr = len(record["ebrs"])
    multiplicities = {str(p["letter"]): int(p["multiplicity"])
                      for p in _spacegroup._pyxtal_wyckoff(int(record["sg"]))[0]}
    rows, matrix, orphans = [], [], []
    for entry in record["srsis"]:
        coefficients = [0] * n_ebr
        terms = []
        for term in entry["terms"]:
            terms.append(_lettered(term))
            if term["column"] is None:
                orphans.append(dict(_lettered(term), srsi=entry["index"]))
            else:
                coefficients[term["column"]] += int(term["coef"])
        rows.append(_SRSIRow(index=int(entry["index"]), name=str(entry["name"]),
                             mod=None if entry["mod"] is None else int(entry["mod"]),
                             coefficients=tuple(coefficients), terms=tuple(terms)))
        matrix.append(tuple(coefficients))
    indicator_rows = tuple(
        _SRSIIndicator(name=str(e["name"]), name_tex=str(e["name_tex"]),
                       mod=int(e["mod"]),
                       theta={int(k): Fraction(v[0], v[1]) for k, v in e["theta"].items()},
                       vanishes_on_ebrs=bool(e.get("vanishes_on_ebrs", True)),
                       ebr_failures=tuple((int(c), str(v))
                                          for c, v in e.get("ebr_failures", ())),
                       derived=(None if e.get("derived") is None else
                                {int(k): Fraction(v[0], v[1])
                                 for k, v in e["derived"]["theta"].items()}),
                       relation=(None if e.get("derived") is None
                                 else str(e["derived"]["relation"])),
                       source=(None if e.get("derived") is None
                               else str(e["derived"]["source"])),
                       derived_note=str(e.get("derived_note", "")))
        for e in record["indicators"])
    ebrs = tuple((str(a), str(b)) for a, b in record["ebrs"])

    # every site irrep the formulas count, in the order of the paper's spelling
    # and letter
    seen = {}
    for row in rows:
        for term in row.terms:
            key = (str(term["irrep"]), str(term["letter"]))
            if key not in seen:
                seen[key] = SiteIrrep(
                    irrep=str(term.get("ebr_label") or term["irrep"]),
                    wyckoff=f"{multiplicities[key[1]]}{key[1]}",
                    maximal=bool(term["maximal"]), ebr=bool(term["ebr"]),
                    _paper=key[0], _letter=key[1],
                    _column=None if term["column"] is None else int(term["column"]))
    site_irreps = tuple(seen[k] for k in sorted(seen))

    def site(entry) -> SiteIrrep:
        entry = _lettered(entry)
        key = (str(entry["irrep"]), str(entry["letter"]))
        if key in seen:
            return seen[key]
        column = entry.get("column")
        return SiteIrrep(irrep=key[0], wyckoff=f"{multiplicities[key[1]]}{key[1]}",
                         maximal=True, ebr=column is not None, _paper=key[0],
                         _letter=key[1], _column=None if column is None else int(column))

    def written(s) -> str:
        return f"({s.irrep})_{s._letter}"

    splits = []
    for e in record["split_ebrs"]:
        parent, children = site(e["parent"]), tuple(site(c) for c in e["children"])
        splits.append(SplitEBR(
            parent=parent, children=children,
            label=f"{written(parent)} = {' + '.join(written(c) for c in children)}",
            srsi_mismatch=bool(e["srsi_mismatch"])))

    def form(row) -> str:
        return _signed((c, f"theta_{i}") for i, c in sorted(row.items()))

    checks = {}
    for f in indicator_rows:
        checks[f.name] = IndicatorCheck(
            relation=f.relation, source=f.source,
            derived=None if f.derived is None else form(f.derived),
            derived_note=f.derived_note,
            ebr_failures=tuple((ebrs[c][0], f"{multiplicities[ebrs[c][1]]}{ebrs[c][1]}", v)
                               for c, v in f.ebr_failures),
            vanishes_on_ebrs=f.vanishes_on_ebrs)
    moduli = {row.name: row.mod for row in rows}
    moduli.update({f.name: f.mod for f in indicator_rows})
    return SRSITables(
        group=group,
        invariants={row.name: _formula(row) for row in rows},
        indicators={f.name: form(f.theta) for f in indicator_rows},
        moduli=moduli, site_irreps=site_irreps, split_ebrs=tuple(splits),
        provenance=provenance, numerical=SRSITablesNumerical(indicators=checks),
        _sg=int(record["sg"]), _number_str=str(record["number_str"]),
        _spinor=record["valuedness"] == "double", _rows=tuple(rows),
        _theta_on_columns=tuple(matrix), _ebrs=ebrs, _indicator_rows=indicator_rows,
        _terms_without_ebr_column=tuple(orphans), _multiplicities=multiplicities)


def _table_of_number(sg: int, spinor: bool) -> SRSITables | Refusal:
    """The tables of a space-group **number** -- :func:`srsi_tables`' second shape.

    Their group is the grey group of the number, with no cell.  Refuses
    ``srsi_data`` (the converted files are not installed), ``srsi_scope``
    (``sg`` outside 1..230) or ``srsi_table_absent`` (one of the thirteen groups
    the tables omit, with the caption's reason).
    """
    if not 1 <= int(sg) <= 230:
        return Refusal(
            "srsi_scope",
            f"space group {sg} is not one of the 230; the SRSI tables run over those",
            value=float(sg), context={"sg": int(sg)})
    status = srsi_status()
    if not status["data"]:
        return _refuse_no_data(status)
    valuedness = "double" if spinor else "single"
    document = _document(valuedness)
    provenance = document["provenance"]
    record = document["groups"].get(str(int(sg)))
    if record is None:
        return Refusal(
            "srsi_table_absent",
            f"the SRSI tables carry no row for space group {sg}: "
            f"{provenance['omitted_reason']}. Its one stable real-space invariant is the "
            "band count, which needs no table",
            context={"sg": int(sg), "spinor": bool(spinor),
                     "omitted_groups": tuple(provenance["omitted_groups"]),
                     "doi": provenance["doi"]})
    number_str = str(record["number_str"])
    group = _spacegroup._group_of_number(int(sg), number_str, bool(spinor),
                                         _grey_symbol(number_str, bool(spinor)))
    return _build(record, _provenance(valuedness), group)


def srsi_tables(group_or_number, *, spinor=None) -> SRSITables | Refusal:
    """The stable real-space invariants of a grey space group, as the paper tabulates them.

    From a measured group the tables come with the group, and every call that
    reads a model's site irreps reads them in these tables; from a space-group
    number with ``spinor=`` they need no model.

    Parameters
    ----------
    group_or_number : ..groups.GroupData or int
        A grey group a model was measured to have, ``r.group``, or a
        space-group number from 1 to 230.
    spinor : bool, optional
        With a number, and only then: the tables with spin-orbit coupling
        (``True``) or without.

    Returns
    -------
    SRSITables or Refusal
        ``tables.summary()`` prints the invariants; ``tables.theta(p)`` evaluates
        them on a configuration.  From a number, ``tables.group`` holds the grey
        group's name, number and valuedness and no cell.  A ``Refusal`` in its
        place is falsy.

    Raises
    ------
    TypeError
        Neither a group nor a number: what :func:`..groups.realized_group`
        returns, or its representation, in place of its group.
    ValueError
        A number without ``spinor=``, or a group with it.

    Refusals
    --------
    ``srsi_scope``
        the group is not grey, or the number is outside 1 to 230.
    ``srsi_table_absent``
        one of the thirteen groups whose one invariant is the band count.
    ``srsi_data``
        the tables are not installed (:func:`srsi_status`).
    ``srsi_ebr_basis``
        the installed ``irreptables`` lists other elementary band representations
        than the tables were converted against; ``context["installed"]`` holds its
        list.
    ``ebr_tables``, ``irrep_tables``, ``ebr_convention``
        from :func:`..groups.ebr_data`.

    Examples
    --------
    >>> srsi_tables(25, spinor=True).moduli["theta_2"]     # doctest: +SKIP
    2
    """
    if isinstance(group_or_number, (int, np.integer)) and not isinstance(group_or_number,
                                                                          bool):
        if spinor is None:
            raise ValueError(
                f"srsi_tables(): {int(group_or_number)} is a space-group number, and the "
                "tables come in two valuednesses: pass spinor=True or spinor=False. "
                "A measured group carries its own")
        return _table_of_number(int(group_or_number), bool(spinor))
    group = group_or_number
    if not _spacegroup._is_number_group(group):
        _spacegroup._require_group("srsi_tables", group)
    # The grammar check comes FIRST and stays a hard error: declaring the
    # valuedness twice is a mistake in the call, not a fact about the crystal,
    # and it is wrong whatever lattice the group is on.  The lattice gate is a
    # physics verdict and follows.
    if spinor is not None and not isinstance(group, Refusal):
        own = bool(getattr(group, "spinor", False))
        raise ValueError(
            f"srsi_tables(): a group was given with spinor={spinor!r}. A group already "
            f"declares its valuedness (spinor={own!r} here), and it decides which table "
            "is read; pass spinor= only with a space-group number")
    gate = _spacegroup._lattice_not_descended(group)
    if gate is not None:
        return gate
    if _spacegroup._is_number_group(group):
        return _table_of_number(int(group._sg_number), bool(group.spinor))
    mtype = _spacegroup._magnetic_type(group)
    if mtype != "II":
        return Refusal(
            "srsi_scope",
            f"#{group.number_str} is a magnetic group of type {mtype}, and the shipped "
            "SRSI tables are the time-reversal-symmetric (grey, type-II) ones. A group "
            "is returned grey exactly where the Hamiltonian obeys time reversal, and "
            "for a model that breaks it no SRSI tables exist",
            context={"number_str": str(group.number_str), "magnetic_type": mtype})
    table = _table_of_number(int(group._sg_number), bool(group.spinor))
    if not table:
        return table

    data = _tables._ebr_data(group)
    if not data:
        return data
    installed = tuple(_ebr_key(name, position) for name, position in data["ebrs"])
    if installed != table._ebrs or str(group.number_str) != table._number_str:
        return Refusal(
            "srsi_ebr_basis",
            f"the EBR list installed for #{group.number_str} is not the one the SRSI "
            f"conversion resolved against (#{table._number_str}, {len(table._ebrs)} "
            f"columns against {len(installed)}), so its stored column indices would "
            "name the wrong EBRs. The tables were converted against irreptables "
            f"{table.provenance.irreptables_version}: install that version",
            value=float(len(installed)), threshold=float(len(table._ebrs)),
            context={"number_str": str(group.number_str),
                     "converted_against": table._number_str,
                     "installed": tuple(installed), "expected": table._ebrs,
                     "irreptables_version": table.provenance.irreptables_version})
    return replace(table, group=group)


def _ebr_key(name, position) -> tuple:
    """``("-E2", "b")`` from ``ebr_data``'s ``("-E2↑G(2)", "1b(4mm1',4mm)")``.

    The same reduction the conversion keyed its column indices on: the label
    before the induction arrow, and the Wyckoff letter of the position.
    """
    match = _POSITION.match(str(position))
    if match is None:
        raise ValueError(f"EBR position {position!r} is not "
                         "'<multiplicity><letter>(<site symbols>)'")
    return str(name).split("↑")[0], match.group(2)


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def _rational_p(u, v, d, y) -> list:
    r"""A rational ``p`` with :math:`\mathrm{BR}\cdot p = y`, from the Smith factors.

    ``u M v = R`` with ``R`` diagonal, so ``M p = y`` is ``R z = u y`` in
    ``z = v^{-1} p``: divide by the nonzero divisors, take zero on the kernel
    directions, come back with ``p = v z``.  Exact over
    :class:`~fractions.Fraction`.  The solution is not unique — two differ by an
    element of :math:`\ker\mathrm{BR}` — and the :math:`\mathbb{Z}`-SRSIs are
    the same on all of them, which is the conversion's kernel check.
    """
    n_col = len(v)
    y_prime = [sum((Fraction(u[i][j]) * y[j] for j in range(len(y))), Fraction(0))
               for i in range(len(u))]
    z = [y_prime[i] / d[i] if i < len(d) and d[i] else Fraction(0)
         for i in range(n_col)]
    return [sum((Fraction(v[j][i]) * z[i] for i in range(n_col)), Fraction(0))
            for j in range(n_col)]


def _no_bare_number(group_or_tables, call) -> None:
    """Raise :class:`TypeError` on a bare space-group number, naming the loader."""
    if isinstance(group_or_tables, (int, np.integer)):
        n = int(group_or_tables)
        raise TypeError(
            f"{call}() takes a measured GroupData or the SRSITables of one. A bare "
            "space-group number carries no valuedness, so load the table first -- "
            f"srsi_tables({n}, spinor=True) or srsi_tables({n}, spinor=False) -- and "
            "pass that")


@functools.lru_cache(maxsize=None)
def _stable_pairs(sg: int, spinor: bool) -> tuple:
    """The pairs of site irreps at one maximal position no invariant tells apart, in one system: ``tables.stably_equivalent``."""
    tables = _table_of_number(int(sg), bool(spinor))
    if not tables:                            # pragma: no cover - gated above
        return ()
    data = _tables._ebr_data(_TableKey(tables._number_str, spinor))
    if not data:                              # pragma: no cover - install-dependent
        return ()
    matrix = np.asarray(data["matrix"], dtype=int)
    labels = list(data["labels_physics_convention"])
    by_content: dict = {}
    for j, (_name, position) in enumerate(data["ebrs"]):
        match = _POSITION.match(str(position))
        by_content.setdefault((match.group(2), tuple(int(x) for x in matrix[:, j])),
                              []).append((j, match.group(1) + match.group(2)))
    out = []
    for (_letter, vector), entries in sorted(by_content.items()):
        if len(entries) < 2:
            continue
        columns = [j for j, _w in entries]
        got = tables._theta_columns(
            [1 if j == columns[0] else 0 for j in range(len(tables._ebrs))])
        out.append(StablePair(
            irreps=tuple(sorted(tables._ebrs[j][0] for j in columns)),
            wyckoff=entries[0][1],
            symmetry_data=_tables.ordered_content(
                {labels[i]: int(x) for i, x in enumerate(vector) if x}),
            theta_Z=got.theta_Z, theta_Zn=got.theta_Zn,
            _names=tuple(sorted(str(data["ebrs"][j][0]) for j in columns)),
            _columns=tuple(columns)))
    return tuple(out)


# ---------------------------------------------------------------------------
# the two vocabularies: the paper's names, and irreptables' as the backend
# ---------------------------------------------------------------------------
#
# The paper's vocabulary is canonical and `irreptables`' is the backend; what
# follows is the conversion between them.  Both vocabularies follow Song et al.
# 2018 and they differ only in how they are *spelled*; the three differences are
# measured, not typed:
#
#   1. the alphabet -- `irreptables` writes the Greek letters and a typographic
#      prime, the supplement's TeX transliterates them (`z3m,pi` / `z3m,π`);
#   2. the order of the qualifiers on one family -- the supplement writes the
#      sign immediately after the stem, `irreptables` writes it last
#      (`delta2+,pi` / `δ2,π+`, `z2,+` / `z2+`);
#   3. one letter's case -- `\Theta_2` against `θ2`.
#
# The map is a **convention**, and it is gated by the measured agreement of the
# two sides' VALUES on every name it pairs (`tests/test_vocabulary.py`, the
# 7295-comparison sweep).  A pairing that failed that gate would be a bug in the
# map and never a tolerance.

#: ``irreptables``' glyphs, and the words the supplement's TeX transliterates
#: them to.  Both cases of theta map to ``Theta``: the supplement writes
#: ``\Theta_2`` for the form ``irreptables`` spells ``θ2``, and neither
#: vocabulary carries a second, lowercase one (measured over the shipped
#: tables: 34 distinct names on each side).
_GREEK_TO_ASCII = {"π": "pi", "′": "'", "δ": "delta", "γ": "gamma", "ω": "omega",
                   "φ": "phi", "α": "alpha", "β": "beta", "η": "eta",
                   "θ": "Theta", "Θ": "Theta"}

#: The inverse alphabet.  ``Theta`` goes back to the lowercase glyph, which is
#: the one ``irreptables`` ships.
_ASCII_TO_GREEK = {"pi": "π", "'": "′", "delta": "δ", "gamma": "γ",
                   "omega": "ω", "phi": "φ", "alpha": "α", "beta": "β",
                   "eta": "η", "Theta": "θ"}

#: How a paper name and an ``irreptables`` name are related.  The first three
#: are pairings and the last two are the honest "there is nothing to pair with"
#: on that side.  ``"reordered"`` **subsumes** the transliteration where both
#: apply: ``delta2+,pi`` -> ``δ2,π+`` moves the qualifier *and* changes the
#: alphabet, ``z2,+`` -> ``z2+`` only moves the qualifier, and both are reported
#: as ``"reordered"`` because the move is the difference worth naming.
NAME_RELATIONS = ("literal", "transliterated", "reordered", "paper_only",
                  "irreptables_only")

# longest alternative first, so ``Theta`` is never eaten by ``eta``
_GREEK_RE = re.compile("|".join(sorted((re.escape(g) for g in _GREEK_TO_ASCII),
                                       key=len, reverse=True)))

_ASCII_RE = re.compile("|".join(sorted((re.escape(w) for w in _ASCII_TO_GREEK),
                                       key=len, reverse=True)))

#: ``delta2+,pi`` and ``z2,+``: a stem, a sign, and an optional qualifier after
#: it.  The sign is what moves.
_PAPER_SIGN = re.compile(r"^([A-Za-z0-9]+?),?([+-])(?:,(.+))?$")

#: ``δ2,π+`` and ``z2+`` transliterated: the same name with the sign last.
_TABLE_SIGN = re.compile(r"^([A-Za-z0-9]+?)(?:,(.+?))?([+-])$")


def _spelling(name: str) -> tuple:
    """``(irreptables spelling, relation)`` of one paper name, by the rules."""
    match = _PAPER_SIGN.match(name)
    if match is not None:
        stem, sign, qualifier = match.groups()
        moved = f"{stem},{qualifier}{sign}" if qualifier else f"{stem}{sign}"
        return _ASCII_RE.sub(lambda m: _ASCII_TO_GREEK[m.group(0)], moved), "reordered"
    greek = _ASCII_RE.sub(lambda m: _ASCII_TO_GREEK[m.group(0)], name)
    return greek, ("literal" if greek == name else "transliterated")


def _unspelling(name: str) -> tuple:
    """``(paper spelling, relation)`` of one ``irreptables`` name, by the rules."""
    plain = _GREEK_RE.sub(lambda m: _GREEK_TO_ASCII[m.group(0)], name)
    match = _TABLE_SIGN.match(plain)
    if match is not None:
        stem, qualifier, sign = match.groups()
        return (f"{stem}{sign},{qualifier}" if qualifier
                else f"{stem},{sign}"), "reordered"
    return plain, ("literal" if plain == name else "transliterated")


@functools.lru_cache(maxsize=1)
def indicator_vocabularies() -> tuple:
    """The paper's names for the symmetry indicators of every tabulated group, and ``irreptables``'.

    Returns
    -------
    tuple of frozenset
        ``(paper names, irreptables names)``, 34 on each side with
        ``irreptables`` 3.1.0.  Both are empty where the tables are not installed
        (:func:`srsi_status`), and the second where ``irreptables``' indicator
        tables cannot be read; :func:`to_irreptables_name` and
        :func:`from_irreptables_name` then return ``None``.
    """
    if not srsi_status()["data"]:
        return frozenset(), frozenset()
    try:
        from irreptables.symmetry_indicators import load_si_table
    except Exception:                          # pragma: no cover - table defect
        load_si_table = None
    paper, shipped = set(), set()
    for valuedness in ("single", "double"):
        document = _document(valuedness)
        table = ({} if load_si_table is None
                 else _spacegroup._indicator_table(valuedness == "double", True))
        for record in document["groups"].values():
            paper |= {str(e["name"]) for e in record["indicators"]}
            entry = table.get(str(record["number_str"]), {})
            shipped |= {str(n) for n in entry.get("indicators", {})}
    return frozenset(paper), frozenset(shipped)


def to_irreptables_name(name) -> str | None:
    """The ``irreptables`` spelling of one of the paper's indicator names.

    ``"z3m,pi"`` is ``"z3m,π"``, ``"delta2+,pi"`` is ``"δ2,π+"``, ``"Theta2"`` is
    ``"θ2"``, and ``"z4"`` is ``"z4"``; every one of the paper's names has one.

    Parameters
    ----------
    name : str
        One of the paper's names, as :func:`indicator_vocabularies` lists them.

    Returns
    -------
    str or None
        The spelling, or ``None`` where ``name`` is not one of the paper's names
        or ``irreptables`` ships no such form; :func:`name_correspondence` says
        why, group by group.
    """
    name = str(name)
    paper, shipped = indicator_vocabularies()
    if name not in paper:
        return None
    spelled, _relation = _spelling(name)
    return spelled if spelled in shipped else None


def from_irreptables_name(name) -> str | None:
    """The paper's spelling of one of ``irreptables``' indicator names, the inverse of :func:`to_irreptables_name`.

    ``"δ2,π+"`` is ``"delta2+,pi"``, and ``"θ2"`` is ``"Theta2"``.

    Parameters
    ----------
    name : str
        One of ``irreptables``' names.

    Returns
    -------
    str or None
        The spelling, or ``None`` where the paper has no such form.
    """
    name = str(name)
    paper, shipped = indicator_vocabularies()
    if name not in shipped:
        return None
    back, _relation = _unspelling(name)
    return back if back in paper else None


@dataclass(frozen=True)
class NamePair:
    """One symmetry indicator of one group, named in the paper's vocabulary and in ``irreptables``'.

    Attributes
    ----------
    paper, irreptables : str or None
        The two spellings; one is ``None`` where the other vocabulary has no such
        form.
    relation : str
        How the two are related, one of :data:`NAME_RELATIONS`.
    mod : int or None
        The modulus, the same on both sides of a pair.
    reason : str
        Why there is no pair, and empty for one.
    """

    paper: str | None
    irreptables: str | None
    relation: str
    mod: int | None = None
    reason: str = ""

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return (f"NamePair({self.paper} <-> {self.irreptables}, "
                f"{self.relation}, mod {self.mod})")


def name_correspondence(group_or_tables):
    """The paper's names for one group's symmetry indicators, against ``irreptables``'.

    The two vocabularies differ in spelling alone: the two forms of a pair are
    measured to take the same value on every band set.

    Parameters
    ----------
    group_or_tables : ..groups.GroupData or SRSITables
        A measured grey group, or its tables.

    Returns
    -------
    tuple of NamePair or Refusal
        One per form of either vocabulary, the paper's first.  A ``Refusal`` in
        its place is falsy.

    Raises
    ------
    TypeError
        A bare space-group number, which carries no valuedness:
        :func:`srsi_tables` takes it with ``spinor=``.

    Refusals
    --------
    ``irreptables_indicator_tables``
        ``irreptables``' indicator table cannot be read for this group.
    ``srsi_data``, ``srsi_scope``, ``srsi_table_absent``, ``srsi_ebr_basis``
        from :func:`srsi_tables`: the group has no tables.
    """
    _no_bare_number(group_or_tables, "name_correspondence")
    gate = _spacegroup._lattice_not_descended(group_or_tables)
    if gate is not None:
        return gate
    tables = (group_or_tables if isinstance(group_or_tables, SRSITables)
              else srsi_tables(group_or_tables))
    if not tables:
        return tables
    try:
        entry = _spacegroup._indicator_table(tables._spinor, True).get(
            tables._number_str, {})
        forms = dict(entry.get("indicators", {}))
    except Exception as exc:                   # pragma: no cover - table defect
        return Refusal("irreptables_indicator_tables",
                       f"irreptables' indicator table could not be read: {exc}",
                       context={"number_str": tables._number_str})
    out, paired = [], set()
    for form in tables._indicator_rows:
        spelled, relation = _spelling(form.name)
        if spelled not in forms:
            out.append(NamePair(form.name, None, "paper_only", form.mod,
                                f"irreptables ships no form spelled {spelled!r} for "
                                f"#{tables._number_str}"))
            continue
        mod = float(forms[spelled]["mod"])
        if abs(mod - round(mod)) > 1e-9 or int(round(mod)) != form.mod:
            out.append(NamePair(form.name, None, "paper_only", form.mod,
                                f"irreptables' {spelled!r} is mod {mod}, the paper's "
                                f"{form.name!r} is mod {form.mod}; the spelling matches "
                                "and the form does not, so this is not a pairing"))
            out.append(NamePair(None, spelled, "irreptables_only", None,
                                f"the modulus disagrees with the paper's {form.name!r}"))
            paired.add(spelled)
            continue
        paired.add(spelled)
        out.append(NamePair(form.name, spelled, relation, form.mod))
    for spelled in sorted(set(forms) - paired):
        mod = float(forms[spelled]["mod"])
        out.append(NamePair(None, spelled, "irreptables_only",
                            int(round(mod)) if abs(mod - round(mod)) < 1e-9 else None,
                            f"the paper's table for #{tables._number_str} carries no "
                            f"form spelled {_unspelling(spelled)[0]!r}"))
    return tuple(out)


# ---------------------------------------------------------------------------
# the canonical class: the paper's names, and the gate that lets them speak it
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def _canonical_gate(number_str: str, spinor: bool) -> dict:
    r"""The two convention checks behind :attr:`..ebr.SymmetryIndicator.indicators`.

    Computed once per system and cached, because both are properties of the
    *tables* and of nothing else.

    **Every returned form is an indicator of this group.**  A linear form descends
    to :math:`\{BS\}/\{AI\}` exactly when it vanishes modulo its own modulus on
    every EBR column -- an EBR is an atomic limit -- and that is checked here on
    the row :meth:`SRSITables.indicators_of` evaluates
    (``_SRSIIndicator.coefficients``: the printed row, or the derived one
    for the four rows of SGs 147, 162 and 163 without SOC where the printed one
    is not an indicator).  Measured over all 434 tabulated systems: zero
    failures.

    **The paper's names determine the class.**  Speaking a class in a vocabulary
    is only honest if the vocabulary can tell the classes apart, so every
    element of :math:`\{BS\}/\{AI\}` is enumerated -- one representative per
    class, :math:`y = \sum_i c_i\,u^{-1}e_{\text{slot}_i}` -- and the map
    class -> canonical values is checked to be injective.  The values on a
    general class are the generators' values combined modulo each form's own
    modulus, which is exact because an indicator is a homomorphism
    :math:`\{BS\}/\{AI\} \to \mathbb{Z}_n`; the test suite re-runs the whole
    enumeration through the rational solve instead and gets the same 1355
    values.  Measured over the 170 systems with an indicator group: 263
    generators, 1355 classes, **injective in every one**.

    Returns
    -------
    dict
        The two measurements, keyed as :func:`canonical_determinism` returns
        them.  ``available`` is ``False`` with a ``reason`` when the tables or
        the EBR data are not available at all.
    """
    tables = _table_of_number(int(str(number_str).split(".")[0]), bool(spinor))
    if not tables:
        return {"available": False, "gate": tables.gate, "reason": tables.reason}
    data = _tables._ebr_data(_TableKey(number_str, spinor))
    if not data:
        return {"available": False, "gate": data.gate, "reason": data.reason}
    u, v, d, matrix = _smith._smith_system(data, str(number_str))
    n_row, n_col = len(matrix), len(matrix[0])
    slots = tuple(i for i, x in enumerate(d) if x > 1)
    divisors = tuple(int(d[i]) for i in slots)

    failures = []
    for form in tables._indicator_rows:
        row = form.coefficients
        for column in range(n_col):
            total = sum((c * Fraction(tables._theta_on_columns[i - 1][column])
                         for i, c in row.items()), Fraction(0))
            if total.denominator != 1 or int(total) % form.mod:
                failures.append((str(form.name), int(column), str(total)))

    generators, collisions, fractional = (), (), ()
    n_classes = 0
    if divisors and tables._indicator_rows:
        u_inv = _smith._unimodular_inverse(u)
        values_of = []
        for slot in slots:
            y = [Fraction(int(u_inv[i][slot])) for i in range(n_row)]
            values_of.append(tables._indicators_of_columns(_rational_p(u, v, d, y)))
        generators = tuple(dict(v2) for v2 in values_of)
        fractional = tuple((str(name), str(value))
                           for record in values_of
                           for name, value in record.items()
                           if not isinstance(value, int))
        if not fractional:
            seen, hits = {}, []
            for combination in itertools.product(*(range(x) for x in divisors)):
                reading = tuple(
                    (str(form.name),
                     sum(c * values_of[g][form.name]
                         for g, c in enumerate(combination)) % form.mod)
                    for form in tables._indicator_rows)
                n_classes += 1
                if reading in seen:
                    hits.append((seen[reading], combination, dict(reading)))
                seen[reading] = combination
            collisions = tuple(hits)
    return {"available": True, "number_str": str(number_str), "spinor": bool(spinor),
            "divisors": divisors, "slots": slots,
            "names": tuple(str(f.name) for f in tables._indicator_rows),
            "moduli": tuple(int(f.mod) for f in tables._indicator_rows),
            "vanishes": not failures, "ebr_failures": tuple(failures),
            "integral": not fractional, "fractional": fractional,
            "n_classes": int(n_classes), "injective": not collisions,
            "collisions": collisions, "generators": generators}


def canonical_determinism(group_or_tables) -> dict | Refusal:
    """Whether the paper's indicator names tell apart every class of a group's symmetry-indicator group.

    What :attr:`..ebr.SymmetryIndicator.indicators` rests on: every form vanishes on
    the group's elementary band representations, and no two classes share every
    value.  Both hold in all 170 systems with an indicator group.

    Parameters
    ----------
    group_or_tables : ..groups.GroupData or SRSITables
        A measured grey group, or its tables.

    Returns
    -------
    dict or Refusal
        ``available``, ``False`` with a ``gate`` and a ``reason`` where the tables
        cannot be read; ``divisors`` and ``n_classes``, the indicator group and
        its size; ``names`` and ``moduli``, the forms; ``injective``, whether the
        names tell every class apart, and ``collisions``, any two they do not;
        ``vanishes`` and ``ebr_failures``, whether every form vanishes on the
        elementary band representations; ``generators``, each form's values on
        the generators of the indicator group, and ``integral``, whether they are
        integers.  A ``Refusal`` in its place is falsy.

    Raises
    ------
    TypeError
        A bare space-group number, which carries no valuedness:
        :func:`srsi_tables` takes it with ``spinor=``.

    Refusals
    --------
    ``srsi_data``, ``srsi_scope``, ``srsi_table_absent``, ``srsi_ebr_basis``
        from :func:`srsi_tables`: the group has no tables.
    """
    _no_bare_number(group_or_tables, "canonical_determinism")
    gate = _spacegroup._lattice_not_descended(group_or_tables)
    if gate is not None:
        return gate
    tables = (group_or_tables if isinstance(group_or_tables, SRSITables)
              else srsi_tables(group_or_tables))
    if not tables:
        return tables
    return dict(_canonical_gate(tables._number_str, tables._spinor))


def _canonical_reading(group, magnetic_type, u, v, d, y, divisors) -> tuple:
    r"""``(canonical, note, tex)`` -- the class in the paper's names.

    What :func:`..ebr.symmetry_indicator` calls to fill
    :attr:`..ebr.SymmetryIndicator.indicators`; ``tex`` is the names in LaTeX.  ``y`` is the returned symmetry data
    vector over the full EBR basis and ``u``, ``v``, ``d`` are the Smith factors
    it was read with, so the rational ``p`` here is the same one
    :func:`..srsi.srsi_of_bands` solves for -- one arithmetic, two records.

    Three states raise :class:`~.refusal.InternalError` rather than returning a
    ``None``, because each of them is this package failing to apply a convention
    the tables fix outright:

    * a grey group with a nontrivial Smith form and **no** paper row (the
      shipped tables cover exactly the 170 systems that have one -- measured, so
      this cannot happen with the shipped data);
    * a returned form that does not vanish on the group's own EBRs, so it is not
      an indicator and its value is not a class;
    * a group whose classes the paper's names do **not** separate: the
      vocabulary cannot speak that class, and printing a number that two
      different classes share would be worse than saying so.
    """

    if magnetic_type != "II":
        return None, (
            "the paper's named indicators are the SRSI paper's, whose shipped tables are "
            f"the time-reversal-symmetric (grey, type-II) ones; #{group.number_str} is "
            f"a magnetic group of type {magnetic_type}, so none of the paper's named "
            "indicators is returned. The Smith class above is unaffected: it is computed from this "
            f"group's own EBR matrix ({_DOI})"), {}
    status = srsi_status()
    if not status["data"]:
        return None, ("the converted SRSI tables are not installed, so the paper's "
                      f"vocabulary cannot be returned: {status['reason']}"), {}
    tables = srsi_tables(group)
    if not tables:
        if divisors and tables.gate == "srsi_table_absent":
            raise InternalError(
                f"#{group.number_str}: the indicator group is {divisors} and the SRSI "
                "tables carry no row for this space group, so its class cannot be "
                "spoken in the paper's names at all. The shipped tables cover exactly "
                "the 170 grey systems with a nontrivial Smith form -- measured, and "
                "asserted in the test suite -- so a group reaching here is this package "
                "applying the wrong table and not a property of the model.")
        return None, ("the paper's named indicators are not returned here: "
                      f"{tables.reason}"), {}
    if not divisors:
        return None, (
            f"the indicator group of #{group.number_str} is trivial, so there is no "
            "class to read in the paper's named indicators; the paper's tables carry an indicator "
            "row for exactly the 170 grey systems that have one, and this is not one "
            "of them"), {}
    if not tables._indicator_rows:
        raise InternalError(
            f"#{group.number_str}: the indicator group is {divisors} and the paper's "
            "table for this system carries no indicator row, so the paper's named "
            "indicators have no value for this class. The two lists were measured to "
            "coincide exactly (170 systems), so this is a defect in the shipped data "
            "or in this package's use of it.")
    gate = _canonical_gate(tables._number_str, tables._spinor)
    if not gate["available"]:
        raise InternalError(
            f"#{group.number_str}: the paper's forms could not be checked as indicators "
            f"of this group that tell its classes apart ({gate['gate']}: "
            f"{gate['reason']}), though the tables returned a record. "
            "Two reads of the same tables disagreeing is arithmetic here failing.")
    if tuple(gate["divisors"]) != tuple(divisors):
        raise InternalError(
            f"#{group.number_str}: the indicator group read to check the paper's forms is "
            f"{tuple(gate['divisors'])} against {tuple(divisors)} here, from the same "
            "EBR matrix. The Smith form is a property of that matrix alone.")
    if not gate["vanishes"]:
        broken = sorted({name for name, _column, _value in gate["ebr_failures"]})
        raise InternalError(
            f"#{group.number_str}: the paper's forms {broken} do not vanish on every "
            "EBR column of this group, so they are not indicators of it and no value "
            f"of them is a class: {gate['ebr_failures'][:4]}. The derived rows cover "
            "the four known cases (SGs 147, 162, 163 without SOC); a fifth is a defect "
            "to surface, not a number to return.")
    if not gate["integral"]:
        raise InternalError(
            f"#{group.number_str}: the paper's forms take non-integer values "
            f"{gate['fractional'][:4]} on the generators of the indicator group, where "
            "an indicator is a homomorphism onto Z_n and must be integral.")
    if not gate["injective"]:
        raise InternalError(
            f"#{group.number_str}: the paper's names do not determine the class -- "
            f"{len(gate['collisions'])} of the {gate['n_classes']} elements of "
            f"{divisors} share every canonical value with another "
            f"({gate['collisions'][0][:2]}). The paper's named indicators cannot tell "
            "this group's classes apart, which is a finding about the tables and not a "
            "number to print.")
    p = _rational_p(u, v, d, [Fraction(int(x)) for x in y])
    canonical = tables._indicators_of_columns(p)
    tex = {str(f.name): str(f.name_tex) for f in tables._indicator_rows}
    note = (
        f"the class in the paper's named indicators, of the SRSI paper ({_DOI}), evaluated "
        f"as the tabulated forms in the Z-SRSIs on the exact rational p of this "
        f"symmetry data vector. The paper's names determine the class here: all "
        f"{gate['n_classes']} elements of {'x'.join('Z%d' % x for x in divisors)} have "
        f"different canonical values, and every one of the {len(tables._indicator_rows)} "
        "forms vanishes on every EBR column of this group")
    return canonical, note, tex


@dataclass(frozen=True)
class _MaximalKey:
    """One maximal site irrep, with the bands one copy of it carries: a coordinate of the atomic realizations.

    ``irrep`` is the paper's spelling and ``letter`` the Wyckoff letter; ``dim``
    the bands one copy carries, its elementary band representation's or, for a
    site irrep whose band representation is a sum of others, the band-count
    form's value on it; ``column`` and ``ebr_name`` the column and name in the
    group's list, and ``ebr_label`` the list's spelling where it differs;
    ``term`` the site irrep as a term of multiplicity one, as the list writes
    its position and site groups.  The keys of a group are
    ``tables._maximal_keys``.
    """

    irrep: str
    letter: str
    dim: int
    ebr: bool
    column: int | None = None
    ebr_name: str | None = None
    ebr_label: str | None = None
    term: Any = None

    @property
    def label(self) -> str:
        """``"(B2)_d"``, the paper's way of writing the key."""
        return f"({self.irrep})_{self.letter}"

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        kind = (f"column {self.column}" if self.ebr
                else "composite band representation")
        return f"_MaximalKey({self.label}, {self.dim} bands, {kind})"


class _TableKey:
    """The three fields :func:`.._tables._ebr_data` reads, from a table's own record.

    The SRSI tables are keyed by a grey BNS number and a valuedness, which is
    exactly what :func:`.._tables._ebr_data` needs to return the columns the
    conversion resolved against -- so a table in hand can fetch its own EBR list
    without a measured ``GroupData``, which is what
    ``tables._maximal_keys`` and a sweep over all 230 space groups both
    need.
    """

    def __init__(self, number_str, spinor):
        self.number_str = str(number_str)
        self.spinor = bool(spinor)
        self._magnetic = True


def _ebr_dimension(name) -> int:
    """``4`` from ``ebr_data``'s ``'-E2^G(4)'`` -- the EBR's own band count."""
    match = re.search(r"\((\d+)\)\s*$", str(name))
    if match is None:
        raise ValueError(f"EBR name {name!r} does not end in its band count '(n)'")
    return int(match.group(1))


def _solve_left(rows, rhs):
    r"""The rational ``a`` with :math:`a\cdot\text{rows} = \text{rhs}`, or ``None``.

    ``rows`` is a list of rows (here :math:`\Theta^{(0)}` over the EBR columns)
    and ``rhs`` a vector over the same columns; the system is over-determined,
    solved exactly over :class:`~fractions.Fraction`, and the solution is
    **verified** on every column before it is returned.
    """
    n_unknown, n_eq = len(rows), len(rhs)
    augmented = [[Fraction(rows[i][j]) for i in range(n_unknown)] + [Fraction(rhs[j])]
                 for j in range(n_eq)]
    pivots, r = [], 0
    for c in range(n_unknown):
        p = next((i for i in range(r, n_eq) if augmented[i][c]), None)
        if p is None:
            continue
        augmented[r], augmented[p] = augmented[p], augmented[r]
        scale = augmented[r][c]
        augmented[r] = [x / scale for x in augmented[r]]
        for i in range(n_eq):
            if i != r and augmented[i][c]:
                factor = augmented[i][c]
                augmented[i] = [x - factor * y
                                for x, y in zip(augmented[i], augmented[r])]
        pivots.append(c)
        r += 1
        if r == n_eq:
            break
    a = [Fraction(0)] * n_unknown
    for i, c in enumerate(pivots):
        a[c] = augmented[i][n_unknown]
    for j in range(n_eq):
        total = sum((a[i] * Fraction(rows[i][j]) for i in range(n_unknown)),
                    Fraction(0))
        if total != Fraction(rhs[j]):
            return None
    return tuple(a)


@functools.lru_cache(maxsize=None)
def _maximal_basis(sg: int, spinor: bool) -> tuple:
    r"""``(keys, columns, band_form)`` for one system, computed once.

    ``columns[j]`` is the **full** :math:`\Theta` column of ``keys[j]`` (every
    row, in table order) as integers; ``band_form`` is the rational ``a`` with
    :math:`a\cdot\Theta^{(0)} = \nu` on the EBR columns, :math:`\nu_j` being
    that EBR's band count.

    Raises
    ------
    ..refusal.InternalError
        The installed EBR list is not the one the conversion resolved against
        (:func:`srsi_tables` refuses ``srsi_ebr_basis`` for the same reason), or
        the band count is not a :math:`\mathbb{Z}`-linear form in the
        :math:`\mathbb{Z}`-SRSIs, or a composite key's band count comes out
        non-integral or non-positive.  All three would mean the arithmetic here
        cannot be applied to the shipped table at all, which is a bug to surface
        rather than a verdict to hand back.
    """

    tables = _table_of_number(int(sg), bool(spinor))
    if not tables:                            # pragma: no cover - gated above
        raise ValueError(f"space group {sg} has no shipped SRSI table: {tables}")
    data = _tables._ebr_data(_TableKey(tables._number_str, spinor))
    if not data:                              # pragma: no cover - install-dependent
        raise InternalError(
            f"#{tables._number_str}: the EBR list this table's columns index into "
            f"cannot be returned ({data}), so the maximal-position basis cannot be "
            "built.")
    installed = tuple(_ebr_key(name, position) for name, position in data["ebrs"])
    if installed != tables._ebrs:              # pragma: no cover - install-dependent
        raise InternalError(
            f"#{tables._number_str}: the installed EBR list is not the one the SRSI "
            f"conversion resolved against ({len(installed)} columns against "
            f"{len(tables._ebrs)}), so the stored column indices name the wrong EBRs "
            "and the atomic realizations would be written in the wrong coordinates. "
            "srsi_tables() refuses 'srsi_ebr_basis' for exactly this; the tables were "
            "converted against irreptables "
            f"{tables.provenance.irreptables_version}: install that version.")

    by_column, composite = {}, []
    for key in tables.site_irreps:
        if not key.maximal:
            continue
        if key.ebr:
            by_column[int(key._column)] = key
        else:
            composite.append(key)
    # the list's entry at each letter, which writes a position's site groups
    entry_at = {}
    for name, position in data["ebrs"]:
        entry_at.setdefault(_ebr_key(name, position)[1], _tables._ebr_term(name, position))
    if set(by_column) != set(range(len(tables._ebrs))):
        raise InternalError(          # pragma: no cover - table-dependent
            f"#{tables._number_str}: the table's rows touch {len(by_column)} of the "
            f"{len(tables._ebrs)} EBR columns. Every column must appear, because the "
            "band count is a Z-linear form in the Z-SRSIs and an EBR carries bands.")

    nu = [_ebr_dimension(name) for name, _position in data["ebrs"]]
    theta_zero = [[tables._theta_on_columns[i][j] for j in range(len(tables._ebrs))]
                  for i in tables._z_rows]
    band_form = _solve_left(theta_zero, nu)
    if band_form is None:
        raise InternalError(          # pragma: no cover - table-dependent
            f"#{tables._number_str}: the band count is not a Z-linear form in the "
            "Z-SRSIs on this group's EBR columns, though it is one in all 434 "
            "tabulated systems as measured -- the band count is a stable real-space "
            "invariant, so it has to be. Either the shipped Theta or the EBR band "
            "counts are not what they are taken to be.")

    def alias(key):
        return key.irrep if key.irrep != key._paper else None

    keys, columns = [], []
    for j in range(len(tables._ebrs)):
        key = by_column[j]
        keys.append(_MaximalKey(irrep=key._paper, letter=key._letter, dim=int(nu[j]),
                                ebr=True, column=j,
                                ebr_name=str(data["ebrs"][j][0]),
                                ebr_label=alias(key),
                                term=_tables._ebr_term(*data["ebrs"][j])))
        columns.append(tuple(int(row[j]) for row in tables._theta_on_columns))
    for key in composite:
        raw = tables._raw_of({(key._paper, key._letter): Fraction(1)})
        if any(value.denominator != 1 for value in raw):
            raise InternalError(      # pragma: no cover - table-dependent
                f"#{tables._number_str}: the Theta column of the composite site irrep "
                f"({key._paper})_{key._letter} is not integral.")
        dim = sum((band_form[i] * raw[r] for i, r in enumerate(tables._z_rows)),
                  Fraction(0))
        if dim.denominator != 1 or dim <= 0:
            raise InternalError(      # pragma: no cover - table-dependent
                f"#{tables._number_str}: the band-count form gives {dim} bands for the "
                f"composite site irrep ({key._paper})_{key._letter}, which is not a "
                "positive integer. A site irrep whose induced band representation is "
                "composite still induces a definite number of bands.")
        listed = entry_at[key._letter]
        keys.append(_MaximalKey(irrep=key._paper, letter=key._letter, dim=int(dim),
                                ebr=False, column=None, ebr_name=None,
                                ebr_label=alias(key),
                                term=replace(listed, irrep=key.irrep)))
        columns.append(tuple(int(value) for value in raw))
    return tuple(keys), tuple(columns), band_form
