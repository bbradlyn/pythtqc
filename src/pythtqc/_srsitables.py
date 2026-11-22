"""The tables of the stable real-space invariants of Hwang et al., as shipped.

For each grey space group they tabulate, the tables give every invariant as a
linear form in the site-irrep multiplicities at the Wyckoff positions, and the
paper's symmetry indicators as forms in the invariants.  The paper's names for
the symmetry indicators are paired here with those of the irreptables package.
The list of elementary band representations the tables are read against is
the one the irreptables package ships.
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

#: The DOI of the paper the shipped tables come from.  The converted files
#: hold it in their provenance block, and srsi_status() returns it.
_DOI = "https://doi.org/10.1038/s41467-026-74844-w"

#: Every refusal the SRSI calls return, with what it measures.  Those passed
#: on from elsewhere are named where they are measured: ``ebr_tables``,
#: ``irrep_tables`` and ``ebr_convention`` by :func:`..groups.ebr_data`, and
#: on the bands' route those of
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
    """A count ``n`` and its ``noun``, plural unless ``n`` is 1: ``"1 row"``, ``"3 rows"``."""
    return f"{n} {noun}" + ("" if n == 1 else "s")


def _valuedness(spinor) -> str:
    """The tables ``spinor`` selects: ``"double-valued"`` if it is true, else ``"single-valued"``."""
    return "double-valued" if spinor else "single-valued"


@functools.lru_cache(maxsize=None)
def _grey_symbol(number_str: str, spinor: bool) -> str:
    """A grey group's Hermann-Mauguin symbol, as the irreptables package names it.

    ``number_str`` is the group's number as the tables write it, ``"183.186"``,
    and ``spinor`` selects its double-valued table.  Returns the symbol,
    ``"P6mm1'"``, or ``""`` where that table cannot be read.
    """
    try:
        return str(_spacegroup._irrep_table(number_str, spinor, True).name)
    except (ImportError, OSError):              # pragma: no cover - tables not installed
        return ""


def _label(group) -> str:
    """A group as a summary names it, by symbol and number: ``P6mm1' #183.186``.

    ``group`` is a :class:`..groups.GroupData`; its ``number_str``, ``spinor``
    and ``name`` are read.  The symbol is :func:`_grey_symbol`'s, or
    ``group.name`` where that is empty.  Where both are empty, the number
    stands alone.
    """
    symbol = _grey_symbol(str(group.number_str), bool(group.spinor)) or str(group.name or "")
    return (f"{symbol} " if symbol else "") + f"#{group.number_str}"


def _group(group) -> str:
    """A group and which of its tables, as a summary heads them: ``P6mm1' #183.186, double-valued``.

    ``group`` is as :func:`_label` takes it.
    """
    return f"{_label(group)}, {_valuedness(group.spinor)}"


def _values(values) -> str:
    """The invariants' ``values`` as a summary prints them: ``(1, -1/2, 0)``, a lone one ``(0)``."""
    return "(" + ", ".join(str(v) for v in values) + ")"


def _span(names) -> str:
    """The invariants ``names``, consecutive in the paper's order, written short.

    Three or more are written ``theta_1 to theta_3``; one or two are listed,
    ``theta_4, theta_5``.
    """
    names = list(names)
    if len(names) >= 3:
        return f"{names[0]} to {names[-1]}"
    return ", ".join(names)


def _and(items) -> str:
    """The strings ``items``, at least one, joined as a sentence lists them: ``a, b and c``."""
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _zn_groups(theta_zn) -> list:
    """The invariants valued modulo n, grouped by modulus.

    ``theta_zn`` maps each name to its :class:`ModularValue`, in the paper's
    order.  Returns ``[(n, names, values), ...]`` in that order, consecutive
    invariants of one modulus forming one group.
    """
    out: list = []
    for name, entry in theta_zn.items():
        if out and out[-1][0] == int(entry.modulus):
            out[-1][1].append(name)
            out[-1][2].append(entry.value)
        else:
            out.append((int(entry.modulus), [name], [entry.value]))
    return out


def _zn_text(theta_zn) -> str:
    """The invariants valued modulo n as a summary writes them: ``theta_4, theta_5 = (0, 1) mod 2``.

    ``theta_zn`` is as :func:`_zn_groups` takes it.  Each of its groups is one
    clause, the clauses joined by ``"; "``.
    """
    return "; ".join(f"{_span(names)} = {_values(values)} mod {n}"
                     for n, names, values in _zn_groups(theta_zn))


def _invariants_text(theta_z, theta_zn=None) -> str:
    """The invariants as a summary writes them.

    ``theta_z`` maps each integer-valued invariant's name to its value, in the
    paper's order.  ``theta_zn``, where given and nonempty, adds the invariants
    valued modulo n as :func:`_zn_text` writes them:
    ``theta_1 to theta_3 = (2, 2, 0); theta_4, theta_5 = (0, 1) mod 2``.
    """
    text = f"{_span(theta_z)} = {_values(theta_z.values())}"
    return text + (f"; {_zn_text(theta_zn)}" if theta_zn else "")


def _zn_header(theta_zn) -> str:
    """The heading of a column of values modulo n: ``(theta_4, theta_5) mod 2``.

    ``theta_zn`` is as :func:`_zn_groups` takes it.  Each of its groups is one
    parenthesis, and the groups are joined as :func:`_and` joins them.
    """
    return _and([f"({_span(names)}) mod {n}" for n, names, _v in _zn_groups(theta_zn)])


def _zn_values(theta_zn) -> str:
    """The values of ``theta_zn`` as a row under :func:`_zn_header`'s heading: ``(1, 1)``.

    ``theta_zn`` is as :func:`_zn_groups` takes it.  Each of its groups is one
    parenthesis, the parentheses joined by ``"; "``.
    """
    return "; ".join(_values(values) for _n, _names, values in _zn_groups(theta_zn))


def _indicator_line(indicator, whose) -> str:
    """A summary's line for the symmetry indicator of a band set.

    ``indicator`` is a :class:`..ebr.SymmetryIndicator`, and ``whose`` the
    possessive the line opens on, ``"their"``.  The line says that the
    indicator group is trivial, or gives the class, with its order where the
    class is not zero.
    """
    if not indicator.divisors:
        return f"  {whose} symmetry-indicator group is trivial   (.indicator)"
    if not any(indicator.values):
        return (f"  {whose} symmetry indicator: class zero in {indicator.indicator_group}"
                "   (.indicator)")
    return (f"  {whose} symmetry indicator: class {indicator.values} in "
            f"{indicator.indicator_group}, of order {indicator.order}   (.indicator)")


def _pair_text(pair) -> str:
    """The :class:`StablePair` ``pair`` as a summary names it: ``B2 and B3 at 12d``."""
    return f"{' and '.join(pair.irreps)} at {pair.wyckoff}"


def _signed(terms) -> str:
    """A sum of ``terms`` with its signs written out: ``2 x - 4 y + 1/2 z``.

    ``terms`` holds ``(coefficient, symbol)`` pairs, each coefficient an
    integer or a fraction.  Returns ``"0"`` where there are none.
    """
    out = []
    for coefficient, symbol in terms:
        coefficient = Fraction(coefficient)
        sign = ("-" if coefficient < 0 else "") if not out else (
            " - " if coefficient < 0 else " + ")
        out.append(f"{sign}{abs(coefficient)} {symbol}")
    return "".join(out) or "0"


def _formula(row) -> str:
    """The formula of the invariant ``row``: ``2 (-1E2E)_e + 1 (-E)_a``.

    ``row`` is an :class:`_SRSIRow`.  Each term is a coefficient times the
    multiplicity of a site irrep at the Wyckoff position of a letter.  The site
    irrep is spelled as the list of elementary band representations spells it
    where that differs from the paper: ``A`` for ``A1`` at a ``222`` site.
    """
    return _signed((term["coef"], f"({term.get('ebr_label') or term['irrep']})_{term['letter']}")
                   for term in row.terms)


def _is_sum(p) -> bool:
    """Whether ``p`` is a sum of terms, as :meth:`SRSITables.theta` takes one.

    A sum is an iterable, not a mapping or a string, whose every item has
    ``multiplicity``, ``irrep`` and ``wyckoff``.  An empty one counts.
    """
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
    """One stable real-space invariant valued modulo n, with its modulus.

    Every ``theta_Zn`` field maps an invariant's name to one.

    Attributes
    ----------
    value : int or Fraction
        The value, reduced modulo :attr:`modulus` where it is an integer.  A
        value that is not an integer, which a rational site-symmetry vector
        can give, is not reduced.
    modulus : int
        The modulus n.
    """

    value: Any
    modulus: int


@dataclass(frozen=True)
class Invariants:
    r"""The stable real-space invariants of a site-symmetry vector, by name.

    The record :meth:`SRSITables.theta` returns.

    Attributes
    ----------
    theta_Z : dict
        ``{name: value}``: each integer-valued invariant (ZSRSI), in the
        paper's order, as an exact rational number.  For a rational
        site-symmetry vector whose symmetry data vector has a nonzero
        symmetry indicator, some are not integers.
    theta_Zn : dict
        ``{name: ModularValue}``: each invariant valued modulo n.  Two
        site-symmetry vectors with the same symmetry data vector can differ in
        them.
    """

    theta_Z: dict
    theta_Zn: dict

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return f"Invariants({_invariants_text(self.theta_Z, self.theta_Zn)})"


@dataclass(frozen=True)
class _SRSIRow:
    r"""One invariant :math:`\theta_i` of a group, with its formula over the site irreps.

    Attributes
    ----------
    index : int
        The paper's index of the invariant, from 1.
    name : str
        The invariant's name, ``"theta_<index>"``.
    mod : int or None
        The modulus n of an invariant valued modulo n, ``None`` for an
        integer-valued one.
    coefficients : tuple of int
        The formula on the elementary band representations, one coefficient
        per entry of ``SRSITables._ebrs``.  Terms on a site irrep that induces
        none do not enter it.
    terms : tuple of dict
        Every term of the formula.  ``"coef"`` is its coefficient,
        ``"irrep"`` and ``"irrep_tex"`` the site irrep in the paper's spelling
        and in TeX, and ``"letter"`` its Wyckoff letter.  ``"column"`` is the
        index into ``SRSITables._ebrs`` of the elementary band representation
        it induces, ``None`` where it induces none.  ``"maximal"`` and
        ``"ebr"`` are as :class:`SiteIrrep` holds them.  ``"ebr_label"``,
        where present, is the spelling of the list of elementary band
        representations, where it differs from the paper's.
    """

    index: int
    name: str
    mod: int | None
    coefficients: tuple
    terms: tuple = field(repr=False)


@dataclass(frozen=True)
class _SRSIIndicator:
    r"""One symmetry indicator as the paper tabulates it, a form in the :math:`\theta_i`.

    Attributes
    ----------
    name, name_tex : str
        The paper's name for the indicator, ``"z4"``, and its TeX.
    mod : int
        The modulus the indicator is valued in.
    theta : dict
        ``{i: coefficient}``: the form the paper prints, over the invariants by
        their index from 1, each coefficient a fraction.
    vanishes_on_ebrs : bool
        Whether the printed form is zero modulo :attr:`mod` on every elementary
        band representation of the group.
    ebr_failures : tuple
        If nonempty, ``(column, value)`` of each elementary band
        representation the printed form does not vanish on: the column the
        index of an entry of ``SRSITables._ebrs``, the value a string such as
        ``"1/2"``.
    derived : dict or None
        If not ``None``, the derived form, keyed as :attr:`theta`.
    relation, source : str or None
        As :class:`IndicatorCheck` holds them, ``None`` where no form is
        derived.
    derived_note : str
        As :class:`IndicatorCheck` holds it.
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
        """The form :meth:`SRSITables.indicators_of` evaluates, keyed as :attr:`theta`.

        It is :attr:`derived` where :attr:`relation` is ``"different"``, and the
        printed :attr:`theta` otherwise.
        """
        if self.relation == "different" and self.derived is not None:
            return dict(self.derived)
        return dict(self.theta)


@dataclass(frozen=True)
class IndicatorCheck:
    """How one symmetry indicator of the tables was checked.

    The form the paper prints for the indicator, a linear combination of the
    invariants, is evaluated on every elementary band representation of the
    group.  An indicator vanishes on each of them, since each is an atomic
    limit.  Where it can be, a second form in the invariants is derived and
    compared with the printed one.  On every band set the derived form takes,
    modulo the indicator's modulus, the value of the indicator of the same
    name that the irreptables package ships.  Where irreptables ships none
    and the symmetry-indicator group is cyclic, it takes instead the band
    set's symmetry-indicator class, a single number modulo the order of that
    cyclic group, read off the Smith normal form of the matrix of elementary
    band representations.  An entry of ``tables.numerical.indicators``.

    Attributes
    ----------
    relation : str or None
        If not ``None``, how the printed form compares with the derived one.
        ``"equal"``: the same coefficients.  ``"equivalent"``: other
        coefficients, with the same value modulo the indicator's modulus on
        every band set.  ``"different"``: a value that differs on some band
        set.
    source : str or None
        If not ``None``, what the form is derived from:
        ``"irreptables:<name>"``, the indicator of that name the irreptables
        package ships, or ``"smith"``, the symmetry-indicator class read off
        the Smith normal form.
    derived : str or None
        If not ``None``, the derived form in the invariants, written as
        ``tables.indicators`` writes the printed one.
    derived_note : str
        The derivation in a sentence, or why there is none.
    ebr_failures : tuple
        If nonempty, each elementary band representation the printed form
        does not vanish on, as ``(site irrep, Wyckoff position, value)``:
        ``("1Eg2Eg", "1a", "1/2")`` for ``z2,3`` of space group 147 without
        spin-orbit coupling.
    vanishes_on_ebrs : bool
        Whether the printed form is zero modulo its modulus on every
        elementary band representation of the group.
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

    The ``numerical`` of an :class:`SRSITables`.

    Attributes
    ----------
    indicators : dict
        ``{name: IndicatorCheck}``: the check of each symmetry indicator, by
        the paper's name.
    """

    indicators: dict


@dataclass(frozen=True)
class ProvenanceCounts:
    """What the conversion of the paper's tables counted, for one valuedness.

    The ``counts`` of a :class:`Provenance`.  Each count runs over the 217
    space groups the tables of that valuedness hold.

    Attributes
    ----------
    groups : int
        The number of space groups tabulated.
    srsi_rows : int
        The number of invariants.
    zn_rows : dict
        ``{modulus: count}``: the number of invariants valued modulo n, by n,
        the modulus written as a string, ``{"2": 62}``.
    terms : int
        The number of terms in the invariants' formulas.
    terms_on_ebr_columns : int
        The number of terms on a site irrep that induces an elementary band
        representation.
    terms_at_maximal_wyckoff_without_an_ebr : int
        The number of terms on a site irrep at a maximal Wyckoff position
        whose band representation is a sum of others.
    terms_at_nonmaximal_wyckoff : int
        The number of terms on a site irrep at a non-maximal Wyckoff position.
    alias_positions : int
        The number of site irreps the paper and the irreptables package's list
        of elementary band representations spell differently, counted at each
        Wyckoff position where one occurs.
    composite_positions : int
        The number of site irreps at a maximal Wyckoff position whose band
        representation is a sum of others, counted the same way.
    si_groups : int
        The number of space groups with a symmetry indicator.
    si_expressions : int
        The number of symmetry indicators.
    si_moduli : dict
        ``{modulus: count}``: the number of symmetry indicators, by modulus,
        the modulus written as a string.
    split_groups : int
        The number of space groups with a splitting of an elementary band
        representation.
    splittings : int
        The number of splittings.
    splittings_srsi_mismatch, splittings_srsi_match : int
        The number of splittings whose parts' invariants valued modulo n do
        not add up to the whole's, and the number whose do.
    ebr_columns : int
        The number of elementary band representations.
    si_on_ebr_checks : int
        The number of evaluations of a symmetry indicator on an elementary
        band representation.
    si_expressions_not_vanishing_on_ebrs : int
        The number of forms the paper prints for a symmetry indicator that do
        not vanish on every elementary band representation of their group.
    si_failing_groups : tuple of int
        The space groups those forms belong to.
    kernel_checks : int
        The number of space groups in which the integer-valued invariants
        were checked to vanish on every rational combination of elementary
        band representations with a zero symmetry data vector.
    derived_indicator_rows : dict
        ``{relation: count}``: the number of symmetry indicators by how the
        printed form compares with the derived one, ``"none"`` where no form
        is derived.
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
    """The paper the tables come from, and what the conversion read and counted.

    The ``provenance`` of an :class:`SRSITables`, the same for every group of
    one valuedness.

    Attributes
    ----------
    paper, doi : str
        The paper the tables are converted from, and its DOI.
    valuedness : str
        ``"single"`` or ``"double"``: the single-valued or the double-valued
        tables.
    converted : str
        The date the tables were converted, ``"YYYY-MM-DD"``.
    irrep_version, irreptables_version : str
        The versions of the irrep and irreptables packages whose elementary
        band representations the conversion read.
    omitted_groups : tuple of int
        The thirteen space groups the tables leave out.
    omitted_reason : str
        Why they are left out, as the caption of the paper's tables states it.
    label_map : dict
        ``{TeX spelling: spelling}``: each site irrep of the paper, as the
        package spells it, ``{"A_{1g}": "A1g"}``.
    label_aliases : tuple of dict
        If nonempty, each site irrep the paper and the irreptables package's
        list of elementary band representations spell differently: its site
        symmetry (``site``), the paper's spelling (``supplement``), the list's
        (``ebr_label``), and the number of Wyckoff positions over all the
        groups where it occurs (``positions``).
    composite_site_irreps : tuple of dict
        If nonempty, each site irrep at a maximal Wyckoff position whose band
        representation is a sum of others, with its site symmetry, the
        paper's spelling and its number of Wyckoff positions, keyed as in
        :attr:`label_aliases`.
    counts : ProvenanceCounts
        What the conversion counted.
    notes : tuple of str
        What the conversion says about the tables, a paragraph each.
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
    r"""A site irrep at a Wyckoff position: one coordinate of the site-symmetry vector.

    A site irrep at a maximal Wyckoff position induces an elementary band
    representation, unless the band representation it induces is a sum of
    others (:math:`{}^1E{}^2E` at site symmetry :math:`\bar4`, ``E`` at
    :math:`\bar4m2` and :math:`\bar42m`).  One at a non-maximal position
    induces a band representation equivalent to a sum of those induced at a
    maximal position whose site-symmetry group contains its own.  An entry of
    ``tables.site_irreps``.

    Attributes
    ----------
    irrep : str
        The site irrep, spelled as the irreptables package's list of
        elementary band representations spells it where the list has it, and
        as the paper spells it elsewhere.  At a ``222``
        site the list writes ``A`` where the paper writes ``A1``.
    wyckoff : str
        The Wyckoff position in the setting the tables are written in, with
        its multiplicity in the conventional cell, ``"4d"``.
    maximal : bool
        Whether the Wyckoff position is maximal.
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
    r"""One splitting of an elementary band representation, as the paper lists it.

    The symmetry data vector of the whole is the sum of the parts', every
    part an elementary band representation.  The paper marks whether the
    invariants valued modulo n of the parts, each counted once, add up to the
    whole's.  An entry of ``tables.split_ebrs``; :meth:`summary` prints it.

    Attributes
    ----------
    parent : SiteIrrep
        The site irrep that induces the whole.
    children : tuple of SiteIrrep
        The site irreps that induce the parts, two or more.
    label : str
        The splitting as the paper writes it: ``"(-E)_c = (-E2)_a + (-E1)_a"``
        in space group 99 with spin-orbit coupling.
    srsi_mismatch : bool
        The paper's mark: whether the parts' invariants valued modulo n do not
        add up to the whole's.
    """

    parent: SiteIrrep
    children: tuple
    label: str
    srsi_mismatch: bool

    def summary(self) -> str:
        """The splitting, and whether its parts' invariants add up, a line each.

        Each line ends with the attribute that holds what it prints.
        """
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
    """Two site irreps at one Wyckoff position that no invariant distinguishes.

    The position is maximal, and the elementary band representations the two
    induce have the same symmetry data vector.  Atomic limits that differ
    only by one of the two for the other are stably equivalent.  The paper's
    example is ``B2`` and ``B3`` at ``12d`` of ``I432``.  An entry of
    ``tables.stably_equivalent``; :meth:`summary` prints it.

    Attributes
    ----------
    irreps : tuple of str
        The two site irreps, as the irreptables package's list of elementary
        band representations spells them.
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
        """The pair, then the symmetry data vector and invariants both share.

        Each line ends with the attributes that hold what it prints.
        """
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
    r"""The stable real-space invariants of one grey space group, as tabulated.

    The record :func:`srsi_tables` returns.  The tables are those of Hwang et
    al., Stable real-space invariants and topology beyond symmetry indicators,
    Nat. Commun. 17, 8564 (2026).  Each invariant is a linear function of the
    site-symmetry vector ``p``, the multiplicity of each site irrep at each
    Wyckoff position.  :meth:`summary` prints the invariants and
    :meth:`formulas` each one's formula.  :meth:`theta` evaluates them on a
    site-symmetry vector, and :meth:`indicators_of` evaluates the paper's
    symmetry indicators.

    Attributes
    ----------
    group : ..groups.GroupData
        The group the tables are of.  Where :func:`srsi_tables` is passed a
        measured group, it is that group.  Where it is passed a space-group
        number, it is the grey group of that number, with its name, number,
        valuedness and magnetic type, and no cell: ``periodic_dirs`` is
        ``None``.
    invariants : dict
        ``{name: formula}``: each invariant in the paper's order, as a sum of
        terms ``c (irrep)_w``.  A term is ``c`` times the multiplicity of the
        site irrep ``irrep`` at the Wyckoff position of letter ``w``.  In space
        group 99 with spin-orbit coupling, ``invariants["theta_4"]`` is
        ``"1 (-E2)_b"``.
    indicators : dict
        If nonempty, ``{name: form}``: each of the paper's symmetry indicators
        as the paper prints it, a linear combination of the invariants called
        its form.  In space group 166 with spin-orbit coupling,
        ``indicators["z4"]`` is
        ``"-4 theta_1 + 4 theta_2 - 4 theta_3 + 4 theta_4"``.  Empty where the
        group's symmetry-indicator group is trivial.
    moduli : dict
        ``{name: modulus}``: the modulus of each invariant and each
        indicator, ``None`` for an integer-valued invariant.
    site_irreps : tuple of SiteIrrep
        Every site irrep the formulas count, at a maximal Wyckoff position or
        not: the coordinates of ``p``.
    split_ebrs : tuple of SplitEBR
        If nonempty, each splitting the paper lists of an elementary band
        representation into others, one SplitEBR each.
    stably_equivalent : tuple of StablePair
        If nonempty, each pair of site irreps at one maximal Wyckoff position
        that no invariant distinguishes.
    provenance : Provenance
        The paper the tables come from, and what the conversion read and
        counted.
    numerical : SRSITablesNumerical
        How each symmetry indicator of the tables was checked, by name.
    """

    group: Any = field(compare=False)
    invariants: dict
    indicators: dict
    moduli: dict
    site_irreps: tuple
    split_ebrs: tuple
    provenance: Any = field(repr=False)
    numerical: Any = field(repr=False)
    # the space-group number, the BNS number of the grey group and which of
    # its tables; the invariants as rows; the formulas on the columns, the
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
        r"""The positions of the integer-valued invariants in :attr:`_rows`."""
        return tuple(i for i, row in enumerate(self._rows) if row.mod is None)

    @property
    def _zn_rows(self) -> dict:
        r"""The positions of the invariants valued modulo n in :attr:`_rows`, ``{n: (i, ...)}`` by ascending n."""
        out: dict[int, list] = {}
        for i, row in enumerate(self._rows):
            if row.mod is not None:
                out.setdefault(int(row.mod), []).append(i)
        return {n: tuple(v) for n, v in sorted(out.items())}

    @property
    def stably_equivalent(self) -> tuple:
        """The pairs of site irreps that no stable real-space invariant distinguishes."""
        return _stable_pairs(int(self._sg), bool(self._spinor))

    # -- the arithmetic ----------------------------------------------------
    def theta(self, p) -> Invariants:
        r"""The stable real-space invariants of a site-symmetry vector.

        Each invariant's formula, :attr:`invariants`, is evaluated on ``p``,
        and the invariants valued modulo n are reduced modulo n.

        Parameters
        ----------
        p : mapping or sum of terms
            The site-symmetry vector.  Pass ``{(site irrep, Wyckoff position):
            multiplicity}``, the position written with its number of sites in
            the conventional cell, ``("-E", "2c")``.  Or pass a sum of terms,
            each with ``multiplicity``, ``irrep`` and ``wyckoff``: a candidate
            sum :func:`..ebr.decompose` returns, an atomic realization
            :func:`..srsi.atomic_realizations` returns, a localization of a
            row :func:`..srsi.srsi_of_model` returns, or the terms of a row of
            ``r.wyckoff_orbits``, with ``r`` what
            :func:`..groups.realized_group` returns.  A site irrep may be at
            any Wyckoff position of the tables, maximal or not, and spelled as
            the irreptables package's list of elementary band representations
            spells it or as the paper does.  The multiplicities are
            integers for an atomic limit, and may be rational, as for the
            site-symmetry vector :func:`..srsi.srsi_of_bands` evaluates for a
            band set.

        Returns
        -------
        Invariants
            The integer-valued invariants as exact fractions, and the
            invariants valued modulo n with their modulus.

        Examples
        --------
        >>> tables = srsi_tables(99, spinor=True)              # doctest: +SKIP
        >>> tables.theta({("-E", "2c"): 1})                    # doctest: +SKIP
        Invariants(theta_1 to theta_3 = (1, 1, 0); theta_4, theta_5 = (0, 1) mod 2)

        Raises
        ------
        TypeError
            ``p`` is neither a mapping nor a sum of terms.
        ValueError
            A key of ``p`` is not a (site irrep, Wyckoff position) pair.  A
            position is not a Wyckoff position of the tables, or is written
            without its number of sites or with a number other than the tables
            give it.  The tables carry no such site irrep at that position.
            The message says what to write.
        """
        return self._invariants(self._theta_of_sites("theta", p))

    def theta_unreduced(self, p) -> dict:
        r"""Every invariant of a site-symmetry vector, none reduced modulo n.

        Each invariant's formula is evaluated on ``p`` as :meth:`theta`
        evaluates it.

        Parameters
        ----------
        p : mapping or sum of terms
            The site-symmetry vector, as :meth:`theta` takes it.

        Returns
        -------
        dict
            ``{name: value}``: every invariant, in the paper's order, as an
            exact fraction.

        Raises
        ------
        TypeError, ValueError
            As :meth:`theta` raises them.
        """
        raw = self._theta_of_sites("theta_unreduced", p)
        return {row.name: value for row, value in zip(self._rows, raw)}

    def indicators_of(self, p) -> dict:
        """The paper's symmetry indicators of a site-symmetry vector.

        Each indicator's form in the invariants, :attr:`indicators`, is
        evaluated on ``p`` and reduced modulo the indicator's modulus.  Four
        printed forms, of space groups 147, 162 and 163 without spin-orbit
        coupling, are nonzero on some elementary band representation of their
        group.  For these the form evaluated is the derived one,
        ``numerical.indicators[name].derived``, which is equal on every band
        set to the indicator of the same name that the irreptables package
        ships.

        Parameters
        ----------
        p : mapping or sum of terms
            The site-symmetry vector, as :meth:`theta` takes it.

        Returns
        -------
        dict
            ``{name: value}``: each indicator's value, reduced modulo its
            modulus where it is an integer and not reduced where it is not.
            Empty where the group's symmetry-indicator group is trivial.

        Raises
        ------
        TypeError, ValueError
            As :meth:`theta` raises them.
        """
        return self._indicators_of_raw(self._theta_of_sites("indicators_of", p))

    def _invariants(self, raw) -> Invariants:
        """The :class:`Invariants` of the unreduced values ``raw``.

        ``raw`` holds every invariant's value as a fraction, one per entry of
        :attr:`_rows`.  An invariant valued modulo n is reduced modulo n where
        its value is an integer, and kept as it is otherwise.
        """
        theta_z = {self._rows[i].name: raw[i] for i in self._z_rows}
        theta_zn = {}
        for i, row in enumerate(self._rows):
            if row.mod is not None:
                n = int(row.mod)
                theta_zn[row.name] = ModularValue(
                    int(raw[i]) % n if raw[i].denominator == 1 else raw[i], n)
        return Invariants(theta_Z=theta_z, theta_Zn=theta_zn)

    def _theta_columns(self, p) -> Invariants:
        """The :class:`Invariants` of a vector ``p`` over the elementary band representations.

        ``p`` holds one multiplicity, an integer or a fraction, per entry of
        :attr:`_ebrs`.  The invariants returned are those of the site-symmetry
        vector with that multiplicity on the site irrep inducing each
        elementary band representation, and zero on every other site irrep.
        Raises ``ValueError`` where ``p`` has another length.
        """
        return self._invariants(self._theta_of_columns(p))

    def _theta_unreduced_columns(self, p) -> tuple:
        """Every invariant of ``p``, taken as :meth:`_theta_columns` takes it, none reduced modulo n.

        Returns one fraction per entry of :attr:`_rows`, in the paper's order.
        """
        return self._theta_of_columns(p)

    def _indicators_of_columns(self, p) -> dict:
        """The paper's symmetry indicators of ``p``, taken as :meth:`_theta_columns` takes it.

        Returns ``{name: value}``, as :meth:`indicators_of` does.
        """
        return self._indicators_of_raw(self._theta_of_columns(p))

    def _theta_of_columns(self, p) -> tuple:
        """The unreduced invariants of ``p``, taken as :meth:`_theta_columns` takes it.

        Returns one fraction per entry of :attr:`_rows`, none reduced modulo
        n.
        """
        vec = _as_fractions(p, len(self._ebrs))
        return tuple(sum((Fraction(c) * v for c, v in zip(row, vec)), Fraction(0))
                     for row in self._theta_on_columns)

    def _indicators_of_raw(self, raw) -> dict:
        """The paper's symmetry indicators of the unreduced invariants ``raw``.

        ``raw`` holds every invariant's value as a fraction, one per entry of
        :attr:`_rows`.  Each indicator's form,
        :attr:`_SRSIIndicator.coefficients`, is evaluated on it and reduced
        modulo the indicator's modulus where the value is an integer; a
        fraction is kept as it is.  Returns ``{name: value}``.
        """
        out = {}
        for form in self._indicator_rows:
            total = sum((c * raw[i - 1] for i, c in form.coefficients.items()),
                        Fraction(0))
            out[form.name] = (int(total) % form.mod if total.denominator == 1
                              else total)
        return out

    # -- the full site-symmetry vector -------------------------------------
    def _site_lookup(self) -> dict:
        """The paper's spelling of each site irrep, keyed by either spelling and letter.

        Returns ``{(irrep, letter): (paper's spelling, letter)}``, ``letter``
        the Wyckoff letter, with a key for the paper's spelling of each site
        irrep of :attr:`site_irreps` and one for the spelling of the list of
        elementary band representations.
        """
        out = {}
        for site in self.site_irreps:
            out[(site._paper, site._letter)] = (site._paper, site._letter)
            out[(site.irrep, site._letter)] = (site._paper, site._letter)
        return out

    def _position(self, call, position) -> str:
        """The Wyckoff letter of ``position``, a position of the tables.

        ``position`` is written with its multiplicity in the conventional
        cell, ``"2c"``.  ``call`` is the name of the public call, which opens
        each message.  Returns the letter, ``"c"``.  Raises ``ValueError``
        where ``position`` is a bare letter or no Wyckoff position, where no
        site irrep of the tables sits at its letter, or where its
        multiplicity is not the one the group gives that letter.
        """
        position = str(position)
        letters = self._multiplicities
        match = _WYCKOFF.match(position)
        if match is None:
            if re.fullmatch(r"[a-zA-Z]", position):
                written = (f"'{letters[position]}{position}'" if position in letters
                           else "such as '1a'")
                raise ValueError(f"{call}(): the position {position!r} is written without "
                                 f"its multiplicity. Write it as its multiplicity and letter, "
                                 f"{written}.")
            raise ValueError(f"{call}(): {position!r} is not a Wyckoff position. Write a "
                             "position as its multiplicity and letter, such as '1a'.")
        multiplicity, letter = int(match.group(1)), match.group(2)
        carried = sorted({site._letter for site in self.site_irreps})
        if letter not in carried:
            raise ValueError(
                f"{call}(): the tables of #{self._number_str} have no Wyckoff position "
                f"{position}. Write one of the positions they carry: "
                + ", ".join(f"{letters.get(c, '')}{c}" for c in carried))
        if letters.get(letter) != multiplicity:
            raise ValueError(f"{call}(): #{self._number_str} has Wyckoff position "
                             f"{letters.get(letter)}{letter}, not {position}. Write "
                             f"{letters.get(letter)}{letter}.")
        return letter

    def _theta_of_sites(self, call, p) -> tuple:
        """Every invariant of the site-symmetry vector ``p``, none reduced modulo n.

        ``p`` is ``{(site irrep, Wyckoff position): multiplicity}`` or a sum of
        terms, as :meth:`theta` takes it, each site irrep in either spelling;
        multiplicities given twice for one site irrep are added.  ``call`` is
        the name of the public call, which opens each message.  Returns one
        fraction per entry of :attr:`_rows`.  Raises ``TypeError`` where ``p``
        is neither, and ``ValueError`` on a key that is not a pair, on a
        position :meth:`_position` raises on, or on a site irrep the tables do
        not carry at that position.
        """
        if isinstance(p, Mapping):
            items = list(p.items())
        elif _is_sum(p):
            items = [((t.irrep, t.wyckoff), t.multiplicity) for t in p]
        else:
            raise TypeError(
                f"{call}(): p is a {type(p).__name__}. Pass "
                "{(site irrep, Wyckoff position): multiplicity} or a sum of terms.")
        lookup = self._site_lookup()
        vector: dict = {}
        for key, value in items:
            if not (isinstance(key, tuple) and len(key) == 2):
                raise ValueError(
                    f"{call}(): the key {key!r} of p is not a pair. Key each "
                    "multiplicity by (site irrep, Wyckoff position), the position "
                    "written as its multiplicity and letter, such as '1b'.")
            letter = self._position(call, key[1])
            resolved = lookup.get((str(key[0]), letter))
            if resolved is None:
                here = sorted({site.irrep for site in self.site_irreps
                               if site._letter == letter})
                raise ValueError(
                    f"{call}(): #{self._number_str} ({_valuedness(self._spinor)}) has no "
                    f"site irrep {str(key[0])!r} at {key[1]}. Pass one the tables carry "
                    f"there: {here}")
            vector[resolved] = vector.get(resolved, Fraction(0)) + Fraction(value)
        return self._raw_of(vector)

    def _theta_at(self, irreps, letter) -> Invariants:
        """The invariants of site irreps at one Wyckoff position, as :meth:`theta` returns them.

        ``irreps`` is ``{site irrep: multiplicity}``, each site irrep in either
        spelling, and ``letter`` the position's Wyckoff letter alone, ``"c"``.
        Returns the :class:`Invariants`.  Raises ``ValueError`` on a site irrep
        the tables do not carry at that letter.
        """
        lookup = self._site_lookup()
        vector: dict = {}
        for irrep, value in dict(irreps).items():
            resolved = lookup.get((str(irrep), str(letter)))
            if resolved is None:
                raise ValueError(
                    f"#{self._number_str} ({_valuedness(self._spinor)}): the tables carry "
                    f"no site irrep {str(irrep)!r} at the Wyckoff letter {str(letter)!r}.")
            vector[resolved] = vector.get(resolved, Fraction(0)) + Fraction(value)
        return self._invariants(self._raw_of(vector))

    def _raw_of(self, vector) -> tuple:
        """Every invariant of a site-symmetry vector, none reduced modulo n.

        ``vector`` is ``{(paper's spelling, letter): multiplicity}``, each
        multiplicity a fraction; a site irrep it leaves out has multiplicity
        zero.  Returns one fraction per entry of :attr:`_rows`.
        """
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
        r"""Every maximal site irrep, with the bands one copy of it carries.

        These are the coordinates over which :func:`..srsi.atomic_realizations`
        lists configurations.  Returns one :class:`_MaximalKey` per maximal
        site irrep: those that induce an elementary band representation, in
        the order of :attr:`_ebrs`, then those whose band representation is a
        sum of others, in the order of :attr:`site_irreps`.
        """
        return _maximal_basis(int(self._sg), bool(self._spinor))[0]

    @property
    def _band_count_form(self) -> tuple:
        r"""The band count as a linear form in the integer-valued invariants.

        Returns ``a``, one fraction per integer-valued invariant in the order
        of :attr:`_z_rows`, solved so that ``a . theta_Z`` is the number of
        bands of each elementary band representation of the group.  Every
        coefficient is an integer in the shipped tables.
        """
        return _maximal_basis(int(self._sg), bool(self._spinor))[2]

    def summary(self) -> str:
        """The invariants and what else the tables hold, a line each.

        Each invariant's line gives its modulus.  Each line ends with the
        attributes that hold what it prints.
        """
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
        r"""Every invariant's formula, a line each, under a line saying what a term is.

        In space group 25 with spin-orbit coupling, ``theta_2 (Z2) = 1 (-E)_b``
        is the invariant ``theta_2``, valued modulo 2, as one times the
        multiplicity of the site irrep ``-E`` at the Wyckoff position of letter
        ``b``.
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
    """The values of ``invariants``, the integer-valued ones apart from the rest.

    ``invariants`` is an :class:`Invariants` of ``tables``.  Returns
    ``(theta_Z, {n: values})``: the integer-valued values, and the values
    valued modulo n grouped by ascending n, each in the paper's order.
    """
    theta_z = tuple(invariants.theta_Z.values())
    theta_zn = {n: tuple(invariants.theta_Zn[tables._rows[i].name].value for i in rows)
                for n, rows in tables._zn_rows.items()}
    return theta_z, theta_zn


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def srsi_status() -> dict[str, Any]:
    """Whether the tables of stable real-space invariants are installed, and where.

    The tables ship with the package.  Where they are missing, every call that
    reads them refuses ``srsi_data``.

    Returns
    -------
    dict
        ``data``: whether both tables are installed.  ``single`` and
        ``double``: the paths of the single-valued and the double-valued
        tables, ``None`` for one that is missing.  ``dir``: the directory they
        are installed in.  ``doi``: the DOI of the paper they are converted
        from.  ``reason``: ``None`` where both are installed, and otherwise
        which are missing.
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
            f"the SRSI tables {missing} are not installed in {DATA_DIR}. "
            "Reinstall pythTQC.")
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
        raise ValueError(f"the vector over the elementary band representations has "
                         f"{len(values)} entries, and the group lists {n}. Pass one "
                         "multiplicity for each, in the group's order.")
    return tuple(Fraction(v) for v in values)


def _lettered(entry: dict) -> dict:
    """A term or a site of the converted data, its Wyckoff letter keyed ``"letter"``.

    ``entry`` is the mapping as the data holds it, the letter keyed
    ``"wyckoff"``.  Returns a new mapping, ``entry`` left as it is.
    """
    return {("letter" if key == "wyckoff" else key): value for key, value in entry.items()}


def _tupled(value):
    """One value of the converted data, a list made immutable.

    Returns ``value`` as an immutable sequence where it is a list, and as it
    is otherwise.
    """
    return tuple(value) if isinstance(value, list) else value


@functools.lru_cache(maxsize=2)
def _provenance(valuedness: str) -> Provenance:
    """The provenance block of one of the two converted documents, as a :class:`Provenance`.

    ``valuedness`` is ``"single"`` or ``"double"``, the document of the
    single-valued or the double-valued tables.
    """
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
    """The tables of the grey group of space-group number ``sg``.

    ``spinor`` selects the double-valued tables.  Returns the
    :class:`SRSITables`, whose group is the grey group of the number with no
    cell, as :func:`srsi_tables` returns them for a number.

    Refusals
    --------
    ``srsi_scope``
        ``sg`` is outside 1 to 230.
    ``srsi_data``
        the converted tables are not installed.
    ``srsi_table_absent``
        ``sg`` is one of the thirteen space groups the tables leave out; the
        reason carries the paper's own for leaving them out.
    """
    if not 1 <= int(sg) <= 230:
        return Refusal(
            "srsi_scope",
            f"there is no space group {sg}. Pass a number from 1 to 230.",
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
            f"space group {sg} is not in the SRSI tables, which leave out "
            f"{provenance['omitted_reason']}.",
            context={"sg": int(sg), "spinor": bool(spinor),
                     "omitted_groups": tuple(provenance["omitted_groups"]),
                     "doi": provenance["doi"]})
    number_str = str(record["number_str"])
    group = _spacegroup._group_of_number(int(sg), number_str, bool(spinor),
                                         _grey_symbol(number_str, bool(spinor)))
    return _build(record, _provenance(valuedness), group)


def srsi_tables(group_or_number, *, spinor=None) -> SRSITables | Refusal:
    """The stable real-space invariants of a grey space group, as tabulated.

    The tables are those of Hwang et al., Stable real-space invariants and
    topology beyond symmetry indicators, Nat. Commun. 17, 8564 (2026).  They
    are read for a group a model was measured to have, or for a space-group
    number, which needs no model.

    Parameters
    ----------
    group_or_number : ..groups.GroupData or int
        The group.  Pass ``r.group`` of what :func:`..groups.realized_group`
        returns, or a space-group number from 1 to 230.
    spinor : bool, optional
        Whether the double-valued tables, those with spin-orbit coupling, are
        read.  Pass it with a space-group number, and only then.

    Returns
    -------
    SRSITables or Refusal
        The tables of the group.  A ``Refusal`` in its place is falsy.

    Examples
    --------
    >>> srsi_tables(25, spinor=True).moduli["theta_2"]     # doctest: +SKIP
    2

    Refusals
    --------
    A refusal handed in as ``group_or_number`` is returned as it is.

    ``srsi_scope``
        the group is not grey, or the number is outside 1 to 230.
    ``srsi_table_absent``
        the group is one of the thirteen space groups the tables leave out.
        In each, the general position is the only Wyckoff position, and it has
        one site irrep.  The one invariant is the band count.
    ``srsi_data``
        the tables are not installed.  :func:`srsi_status` says which is
        missing.
    ``srsi_ebr_basis``
        the installed irreptables package lists elementary band
        representations of the measured group other than those the tables
        were converted against.  ``context["installed"]`` holds its list.
    ``ebr_tables`` / ``irrep_tables`` / ``ebr_convention``
        the tables installed hold no elementary band representations of the
        measured group that can be read in this package's convention.

    Raises
    ------
    TypeError
        ``group_or_number`` is neither a group nor a number: what
        :func:`..groups.realized_group` returns, or its representation, in
        place of its group.
    ValueError
        A number without ``spinor=``, or a group with it.
    """
    if isinstance(group_or_number, (int, np.integer)) and not isinstance(group_or_number,
                                                                          bool):
        if spinor is None:
            raise ValueError(
                f"srsi_tables(): the space-group number {int(group_or_number)} is passed "
                "without spinor=. Pass spinor=True for the double-valued tables or "
                "spinor=False for the single-valued ones.")
        return _table_of_number(int(group_or_number), bool(spinor))
    group = group_or_number
    if not _spacegroup._is_number_group(group):
        _spacegroup._require_group("srsi_tables", group)
    # A group passed with spinor= raises before its lattice is checked: the
    # valuedness given twice is a mistake in the call, whatever the lattice.
    if spinor is not None and not isinstance(group, Refusal):
        own = bool(getattr(group, "spinor", False))
        raise ValueError(
            f"srsi_tables(): spinor={spinor!r} is passed with a group, which has "
            f"spinor={own!r} of its own. Pass spinor= only with a space-group number.")
    gate = _spacegroup._lattice_not_descended(group)
    if gate is not None:
        return gate
    if _spacegroup._is_number_group(group):
        return _table_of_number(int(group._sg_number), bool(group.spinor))
    mtype = _spacegroup._magnetic_type(group)
    if mtype != "II":
        return Refusal(
            "srsi_scope",
            f"#{group.number_str} is a magnetic group of type {mtype}, and the SRSI "
            "tables are of grey (type-II) groups alone.",
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
            f"the installed irreptables lists {len(installed)} elementary band "
            f"representations for #{group.number_str}, other than the "
            f"{len(table._ebrs)} of #{table._number_str} that the SRSI tables were "
            "converted against. Install irreptables "
            f"{table.provenance.irreptables_version}.",
            value=float(len(installed)), threshold=float(len(table._ebrs)),
            context={"number_str": str(group.number_str),
                     "converted_against": table._number_str,
                     "installed": tuple(installed), "expected": table._ebrs,
                     "irreptables_version": table.provenance.irreptables_version})
    return replace(table, group=group)


def _ebr_key(name, position) -> tuple:
    """The key of an elementary band representation: its site irrep and Wyckoff letter.

    ``name`` and ``position`` are one entry of the list of elementary band
    representations, ``("-E2↑G(2)", "1b(4mm1',4mm)")``.  Returns the site
    irrep before the induction arrow and the letter of the position,
    ``("-E2", "b")``, as :attr:`SRSITables._ebrs` holds each column.  Raises
    ``ValueError`` where ``position`` is not written
    ``<multiplicity><letter>(<site symmetries>)``.
    """
    match = _POSITION.match(str(position))
    if match is None:
        raise ValueError(f"the position {position!r} in the list of elementary band "
                         "representations is not written "
                         "'<multiplicity><letter>(<site symmetries>)'.")
    return str(name).split("↑")[0], match.group(2)


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def _rational_p(u, v, d, y) -> list:
    r"""A rational solution ``p`` of ``M p = y``, from the Smith factors of ``M``.

    ``M`` is the matrix of elementary band representations of a group, one
    column each.  ``u``, ``v`` and ``d`` are its Smith factors, as
    :func:`.._smith._smith_system` returns them: ``u M v`` is diagonal with
    entries ``d``.  ``y`` is a vector over the rows of ``M``, a symmetry data
    vector.  Writing ``p = v z``, the equation is ``d_i z_i = (u y)_i``; each
    ``z_i`` is solved where ``d_i`` is nonzero and is zero elsewhere.
    Returns ``p`` as a list of fractions, one per column of ``M``.  It solves
    ``M p = y`` where ``y`` lies in the rational span of the columns, as a
    symmetry data vector that satisfies the compatibility relations does.
    Two solutions differ by a vector in the kernel of ``M``, and the
    integer-valued invariants take the same values on both in the shipped
    tables.
    """
    n_col = len(v)
    y_prime = [sum((Fraction(u[i][j]) * y[j] for j in range(len(y))), Fraction(0))
               for i in range(len(u))]
    z = [y_prime[i] / d[i] if i < len(d) and d[i] else Fraction(0)
         for i in range(n_col)]
    return [sum((Fraction(v[j][i]) * z[i] for i in range(n_col)), Fraction(0))
            for j in range(n_col)]


def _no_bare_number(group_or_tables, call) -> None:
    """Raise ``TypeError`` where ``group_or_tables`` is a bare space-group number.

    ``call`` is the name of the public call, which opens the message.  The
    message names :func:`srsi_tables` with ``spinor=`` as the call that
    takes a number.
    """
    if isinstance(group_or_tables, (int, np.integer)):
        n = int(group_or_tables)
        raise TypeError(
            f"{call}(): {n} is a space-group number. Pass srsi_tables({n}, spinor=True) "
            f"or srsi_tables({n}, spinor=False), or the group realized_group returns, "
            "r.group.")


@functools.lru_cache(maxsize=None)
def _stable_pairs(sg: int, spinor: bool) -> tuple:
    """The pairs of site irreps that no stable real-space invariant distinguishes.

    ``sg`` is a space-group number and ``spinor`` selects the double-valued
    tables.  A pair is two site irreps at one maximal Wyckoff position whose
    elementary band representations have the same symmetry data vector.
    Returns one :class:`StablePair` per pair, ordered by Wyckoff letter, with
    the invariants of the first site irrep of each; in the shipped tables
    both carry the same.  Returns ``()`` where the tables or the
    elementary band representations cannot be read.
    """
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
# the two vocabularies: the paper's names for the indicators, and irreptables'
# ---------------------------------------------------------------------------
#
# What follows converts between the paper's names for the symmetry indicators
# and the names the irreptables package gives the same forms.  Both follow Song
# et al. 2018 and differ only in spelling, in three ways:
#
#   1. the alphabet: the paper's supplement spells out the Greek letters that
#      irreptables writes as glyphs, and writes a plain prime where
#      irreptables writes a typographic one (`z3m,pi` and `z3m,π`);
#   2. where the sign stands: the supplement writes it right after the stem,
#      irreptables writes it last (`delta2+,pi` and `δ2,π+`, `z2,+` and `z2+`);
#   3. the case of one letter: `\Theta_2` and `θ2`.
#
# The map is a convention of spelling.  `tests/test_vocabulary.py` checks, in
# every tabulated group, that the two names of every pair the map makes take
# the same value on every elementary band representation, on random atomic
# limits and on every generator of the symmetry-indicator group.

#: ``irreptables``' glyphs, and what the supplement's TeX writes for each: a
#: word for a Greek letter, a plain prime for the typographic one.  Both cases
#: of theta map to ``Theta``: the supplement writes
#: ``\Theta_2`` for the form ``irreptables`` spells ``θ2``, and neither
#: vocabulary has two names that differ only in the case of theta.
_GREEK_TO_ASCII = {"π": "pi", "′": "'", "δ": "delta", "γ": "gamma", "ω": "omega",
                   "φ": "phi", "α": "alpha", "β": "beta", "η": "eta",
                   "θ": "Theta", "Θ": "Theta"}

#: The inverse alphabet.  ``Theta`` goes back to the lowercase glyph, which is
#: the one ``irreptables`` ships.
_ASCII_TO_GREEK = {"pi": "π", "'": "′", "delta": "δ", "gamma": "γ",
                   "omega": "ω", "phi": "φ", "alpha": "α", "beta": "β",
                   "eta": "η", "Theta": "θ"}

#: How a name of the paper and a name of ``irreptables`` are related.  The
#: first three pair the two names; the last two mark a name with no
#: counterpart on the other side.  ``"reordered"`` is reported wherever the
#: sign moves, whether or not the alphabet changes too: ``delta2+,pi`` and
#: ``δ2,π+``, ``z2,+`` and ``z2+``.
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
    """The ``irreptables`` spelling of one of the paper's indicator names.

    ``name`` is spelled as the paper spells it.  Returns ``(spelling,
    relation)``: ``relation`` is ``"reordered"`` where the sign moves,
    ``"transliterated"`` where only the alphabet changes, and ``"literal"``
    where nothing does.  The spelling follows from the rules of spelling
    alone; whether ``irreptables`` ships it is not checked.
    """
    match = _PAPER_SIGN.match(name)
    if match is not None:
        stem, sign, qualifier = match.groups()
        moved = f"{stem},{qualifier}{sign}" if qualifier else f"{stem}{sign}"
        return _ASCII_RE.sub(lambda m: _ASCII_TO_GREEK[m.group(0)], moved), "reordered"
    greek = _ASCII_RE.sub(lambda m: _ASCII_TO_GREEK[m.group(0)], name)
    return greek, ("literal" if greek == name else "transliterated")


def _unspelling(name: str) -> tuple:
    """The paper's spelling of one of the ``irreptables`` indicator names.

    ``name`` is spelled as ``irreptables`` spells it.  Returns ``(spelling,
    relation)``, the inverse of :func:`_spelling`.  Whether the paper has the
    spelling is not checked.
    """
    plain = _GREEK_RE.sub(lambda m: _GREEK_TO_ASCII[m.group(0)], name)
    match = _TABLE_SIGN.match(plain)
    if match is not None:
        stem, qualifier, sign = match.groups()
        return (f"{stem}{sign},{qualifier}" if qualifier
                else f"{stem},{sign}"), "reordered"
    return plain, ("literal" if plain == name else "transliterated")


@functools.lru_cache(maxsize=1)
def indicator_vocabularies() -> tuple:
    """The names of the symmetry indicators in the paper and in ``irreptables``.

    Each is the set of names over every group the tables carry, with and
    without spin-orbit coupling.

    Returns
    -------
    tuple of frozenset
        ``(paper names, irreptables names)``, 34 on each side with
        ``irreptables`` 3.1.0.  Both are empty where the tables are not
        installed (:func:`srsi_status`), and the second where the indicator
        tables of ``irreptables`` cannot be read.
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
    """The paper's spelling of one of the ``irreptables`` indicator names.

    It is the inverse of :func:`to_irreptables_name`: ``"δ2,π+"`` is
    ``"delta2+,pi"``, and ``"θ2"`` is ``"Theta2"``.

    Parameters
    ----------
    name : str
        One of the ``irreptables`` names, as :func:`indicator_vocabularies`
        lists them.

    Returns
    -------
    str or None
        The spelling, or ``None`` where ``name`` is not one of the
        ``irreptables`` names or the paper has no such form.
    """
    name = str(name)
    paper, shipped = indicator_vocabularies()
    if name not in shipped:
        return None
    back, _relation = _unspelling(name)
    return back if back in paper else None


@dataclass(frozen=True)
class NamePair:
    """One symmetry indicator of one group, named by the paper and by ``irreptables``.

    Attributes
    ----------
    paper, irreptables : str or None
        The two spellings.  One is ``None`` where the other vocabulary has no
        such form.
    relation : str
        How the two are related, one of :data:`NAME_RELATIONS`.
    mod : int or None
        The modulus, the same on both sides of a pair.  ``None`` for a name of
        ``irreptables`` alone whose modulus is not an integer or disagrees
        with the paper's.
    reason : str
        If nonempty, why the name has no pair.
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
    """The names of one group's symmetry indicators in the paper and in ``irreptables``.

    The two vocabularies differ in spelling alone, and the two names of a pair
    take the same value on every band set.

    Parameters
    ----------
    group_or_tables : ..groups.GroupData or SRSITables
        A measured grey group, or its tables.

    Returns
    -------
    tuple of NamePair or Refusal
        One per form of either vocabulary.  The paper's names come first.
        Where the two moduli disagree, the name of ``irreptables`` with the
        same spelling follows the paper's, unpaired.  The other names of
        ``irreptables`` alone come last, in alphabetical order.  A
        ``Refusal`` in its place is falsy.

    Refusals
    --------
    A refusal handed in as ``group_or_tables`` is returned as it is.

    ``irreptables_indicator_tables``
        the indicator table of ``irreptables`` cannot be read for this group.
    ``srsi_scope`` / ``srsi_table_absent`` / ``srsi_data`` / ``srsi_ebr_basis``
        from :func:`srsi_tables`: the group has no tables that can be read.
    ``ebr_tables`` / ``irrep_tables`` / ``ebr_convention``
        from :func:`srsi_tables`: the tables installed hold no elementary band
        representations of the group that can be read in this package's
        convention.

    Raises
    ------
    TypeError
        A bare space-group number, which does not say whether the single- or
        the double-valued tables are meant: :func:`srsi_tables` takes it with
        ``spinor=``.  Or anything that is neither a group nor its tables.
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
                       f"the indicator table of irreptables cannot be read: {exc}. "
                       "Check the irreptables installation.",
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
# the class in the paper's named indicators, and the checks behind it
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def _canonical_gate(number_str: str, spinor: bool) -> dict:
    r"""The two checks on the paper's named indicators of one system.

    :func:`_canonical_reading` reads them before it fills
    :attr:`..ebr.SymmetryIndicator.indicators`, and raises where one fails.
    ``number_str`` is the BNS number of the grey group, ``"99.164"``, and
    ``spinor`` selects the double-valued tables.

    The first check evaluates each named form,
    :attr:`_SRSIIndicator.coefficients`, on every column of the matrix of
    elementary band representations.  A form is an indicator of the group
    where each value is an integer that vanishes modulo the form's modulus.
    The second check reads the Smith normal form of that matrix, whose
    diagonal entries above 1, ``d_i``, stand at the places ``slot_i``.  It
    evaluates the forms on each generator of the symmetry-indicator group,
    the column ``slot_i`` of ``u^{-1}`` for the Smith factor ``u``, through
    :func:`_rational_p`.  Each class, ``sum_i c_i`` times the generators
    with ``0 <= c_i < d_i``, takes the generators' values combined modulo
    each form's modulus.  The check records each two classes that take the
    same value of every form.

    Returns
    -------
    dict
        Keyed as :func:`canonical_determinism` returns it.  ``available`` is
        ``False``, with ``gate`` and ``reason`` and no other key, where the
        tables or the elementary band representations cannot be read.  No
        class is enumerated where the symmetry-indicator group is trivial,
        where the tables carry no named form, or where a form takes a value
        on a generator that is not an integer: ``n_classes`` is then ``0``
        and ``injective`` is ``True``.
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
    """Whether the paper's named indicators tell every symmetry-indicator class apart.

    :func:`..ebr.symmetry_indicator` returns, as ``indicators``, a band set's
    value of each of the paper's named symmetry indicators.  These values
    name the band set's class, its element of the symmetry-indicator group,
    where two properties of the tables hold.  First, each indicator's form,
    the linear combination of the invariants :meth:`SRSITables.indicators_of`
    evaluates, vanishes on every elementary band representation of the
    group.  For the four printed forms, of space groups 147, 162 and 163
    without spin-orbit coupling, that do not, the form checked is the derived
    one.  Second, no two elements of the symmetry-indicator group take the
    same value of every indicator.  The forms are evaluated on the
    generators of the symmetry-indicator group, and every element is
    enumerated from those values.  Both properties hold in all 170 tabulated
    systems with a nontrivial symmetry-indicator group, a system being a grey
    group with its single-valued or its double-valued tables.

    Parameters
    ----------
    group_or_tables : ..groups.GroupData or SRSITables
        The group.  Pass ``r.group`` of what :func:`..groups.realized_group`
        returns, or the tables :func:`srsi_tables` returns.

    Returns
    -------
    dict or Refusal
        ``available`` is ``True``, except where tables read for a
        space-group number are handed in and the elementary band
        representations of their group cannot be read.  There it is
        ``False``, with ``gate`` and ``reason`` and no other key; with a
        measured group, :func:`srsi_tables` refuses that case instead.
        ``number_str`` and ``spinor`` name the system.  ``divisors`` are the
        orders of the cyclic factors of the symmetry-indicator group, and
        ``slots`` their places on the diagonal of the Smith normal form of
        the matrix of elementary band representations.  ``names`` and
        ``moduli`` are the names of the paper's indicators and their moduli.
        ``vanishes`` says whether every form vanishes on every elementary
        band representation.  ``ebr_failures`` holds ``(name, column,
        value)`` where one does not, ``column`` being the elementary band
        representation's column in the matrix of elementary band
        representations.  ``generators`` holds each indicator's values on
        each generator of the symmetry-indicator group.  ``integral`` says
        whether all of them are integers, and ``fractional`` holds each
        ``(name, value)`` that is not.  ``n_classes`` is the number of
        elements enumerated, the order of the symmetry-indicator group, or
        ``0`` where none is: a trivial group, no named indicator, or a value
        on a generator that is not an integer.  ``injective`` says whether
        no two elements take the same value of every indicator, and
        ``collisions`` holds each two that do, with the values they share.  A
        ``Refusal`` in its place is falsy.

    Refusals
    --------
    A refusal handed in as ``group_or_tables`` is returned as it is.

    ``srsi_scope`` / ``srsi_table_absent`` / ``srsi_data`` / ``srsi_ebr_basis``
        from :func:`srsi_tables`: the group has no tables that can be read.
    ``ebr_tables`` / ``irrep_tables`` / ``ebr_convention``
        from :func:`srsi_tables`: the tables installed hold no elementary band
        representations of the group that can be read in this package's
        convention.

    Raises
    ------
    TypeError
        A space-group number alone, which does not say whether the single- or
        the double-valued tables are meant, or anything that is neither a
        group nor its tables.
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
    r"""The symmetry-indicator class of a set of bands, in the paper's named indicators.

    :func:`..ebr.symmetry_indicator` calls it to fill
    :attr:`..ebr.SymmetryIndicator.indicators`.  ``group`` is the measured
    group and ``magnetic_type`` its magnetic type, ``"II"`` for a grey group.
    ``y`` is the symmetry data vector over the rows of the group's matrix of
    elementary band representations, and ``u``, ``v`` and ``d`` are the
    Smith factors of that matrix, as :func:`_rational_p` takes them.
    ``divisors`` are the orders of the cyclic factors of the indicator group.
    The named forms are evaluated on the rational solution ``p`` of
    :func:`_rational_p`, the one :func:`..srsi.srsi_of_bands` makes from the
    same vector.

    Returns ``(indicators, note, tex)``: each form's value ``{name: value}``,
    a sentence on how the values were read, and each name in TeX.
    ``indicators`` is ``None`` and ``tex`` empty, with the reason in
    ``note``, for a group that is not grey, where the tables are not
    installed or refuse the group, and where the indicator group is trivial.

    Raises
    ------
    ..refusal.InternalError
        The indicator group is nontrivial and the tables leave the group out
        or carry no named form for it.  The checks of :func:`_canonical_gate`
        cannot be read, or read other divisors.  Or they fail: a form does
        not vanish on an elementary band representation, a form takes a value
        that is not an integer on a generator, or two classes take the same
        value of every form.
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
                f"#{group.number_str}: the indicator group is {divisors}, and the SRSI "
                "tables carry no row for this space group. Report it, with the model "
                "and the call.")
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
            f"#{group.number_str}: the indicator group is {divisors}, and the paper's "
            "table for this system carries no indicator row. Report it, with the model "
            "and the call.")
    gate = _canonical_gate(tables._number_str, tables._spinor)
    if not gate["available"]:
        raise InternalError(
            f"#{group.number_str}: the checks on the paper's forms cannot be made "
            f"({gate['gate']}: {gate['reason']}), though srsi_tables returns the "
            "tables. Report it, with the model and the call.")
    if tuple(gate["divisors"]) != tuple(divisors):
        raise InternalError(
            f"#{group.number_str}: the indicator group is {tuple(gate['divisors'])} in "
            f"the checks on the paper's forms and {tuple(divisors)} in the symmetry "
            "indicator of the bands, both read from one EBR matrix. Report it, with "
            "the model and the call.")
    if not gate["vanishes"]:
        broken = sorted({name for name, _column, _value in gate["ebr_failures"]})
        raise InternalError(
            f"#{group.number_str}: the paper's forms {broken} do not vanish on every "
            f"EBR column of this group: {gate['ebr_failures'][:4]}. Report it, with the "
            "model and the call.")
    if not gate["integral"]:
        raise InternalError(
            f"#{group.number_str}: the paper's forms take values that are not integers "
            f"on the generators of the indicator group: {gate['fractional'][:4]}. "
            "Report it, with the model and the call.")
    if not gate["injective"]:
        raise InternalError(
            f"#{group.number_str}: the paper's names do not determine the class. "
            f"{len(gate['collisions'])} of the {gate['n_classes']} elements of "
            f"{divisors} take the same value of every named indicator as another "
            f"element, such as {gate['collisions'][0][:2]}. Report it, with the model and the "
            "call.")
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
    """One maximal site irrep, with the bands one copy of it carries.

    A coordinate of the atomic realizations, an entry of
    ``tables._maximal_keys``.

    Attributes
    ----------
    irrep : str
        The site irrep, as the paper spells it.
    letter : str
        The letter of its Wyckoff position.
    dim : int
        The number of bands one copy carries: those of its elementary band
        representation, or, where its band representation is a sum of
        others, the value of ``tables._band_count_form`` on it.
    ebr : bool
        Whether it induces an elementary band representation.
    column, ebr_name : int or None, str or None
        The column and the name of that elementary band representation in
        the irreptables package's list of elementary band representations,
        ``None`` where there is none.
    ebr_label : str or None
        The list's spelling of the site irrep, ``None`` where it is the
        paper's.
    term : ..groups.Term
        The site irrep as a term of multiplicity one, its position and site
        symmetries as the list writes them.
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
        """The site irrep and its Wyckoff letter as the paper writes them, ``"(B2)_d"``."""
        return f"({self.irrep})_{self.letter}"

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        kind = (f"column {self.column}" if self.ebr
                else "composite band representation")
        return f"_MaximalKey({self.label}, {self.dim} bands, {kind})"


class _TableKey:
    """The three fields :func:`.._tables._ebr_data` reads, made from the tables alone.

    ``number_str`` is the BNS number of the grey group, ``"99.164"``, and
    ``spinor`` selects the double-valued tables; ``_magnetic`` is ``True``.
    With it, the elementary band representations of the tables' own group
    are read without a measured group, as ``tables._maximal_keys`` and
    ``tables.stably_equivalent`` read them.
    """

    def __init__(self, number_str, spinor):
        self.number_str = str(number_str)
        self.spinor = bool(spinor)
        self._magnetic = True


def _ebr_dimension(name) -> int:
    """The number of bands an elementary band representation carries, from its name.

    ``name`` is the name in the list of elementary band representations,
    ``"-E1↑G(2)"``.  Returns the count in its closing parentheses, ``2``.
    Raises ``ValueError`` where the name does not end in one.
    """
    match = re.search(r"\((\d+)\)\s*$", str(name))
    if match is None:
        raise ValueError(f"the name {name!r} in the list of elementary band "
                         "representations does not end in its number of bands, '(n)'.")
    return int(match.group(1))


def _solve_left(rows, rhs):
    r"""The rational ``a`` with ``a . rows = rhs``, or ``None`` where there is none.

    ``rows`` is a list of rows over a set of columns, here the integer-valued
    invariants on the elementary band representations, and ``rhs`` a vector
    over the same columns.  The system is solved by elimination in exact
    fractions, an unknown left free taken as zero.  Returns ``a``, one
    fraction per row, once it is checked to solve every column.
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
    r"""The maximal site irreps of one system, their invariants, and the band count.

    ``sg`` is a space-group number the tables carry, and ``spinor`` selects
    the double-valued tables.  Returns ``(keys, columns, band_form)``:
    ``keys`` as ``tables._maximal_keys`` returns them; ``columns[j]`` every
    invariant of one copy of ``keys[j]``, none reduced modulo n, as integers
    in the paper's order; ``band_form`` as ``tables._band_count_form``
    returns it.

    Raises
    ------
    ValueError
        No table of ``sg`` is returned: ``sg`` is outside 1 to 230 or left
        out, or the tables are not installed.
    ..refusal.InternalError
        The installed list of elementary band representations cannot be read,
        or is not the one the tables were converted against
        (:func:`srsi_tables` refuses ``srsi_ebr_basis`` there).  The formulas
        leave a column of the list out.  The band count is not a linear form
        in the integer-valued invariants on the elementary band
        representations.  Or a site irrep whose band representation is a sum
        of others has an invariant that is not an integer, or a band count
        that is not a positive integer.
    """

    tables = _table_of_number(int(sg), bool(spinor))
    if not tables:                            # pragma: no cover - gated above
        raise ValueError(f"no SRSI table is returned for space group {sg}: {tables}")
    data = _tables._ebr_data(_TableKey(tables._number_str, spinor))
    if not data:                              # pragma: no cover - install-dependent
        raise InternalError(
            f"#{tables._number_str}: the list of elementary band representations the "
            f"table's columns index cannot be read: {data}. Report it, with the model "
            "and the call.")
    installed = tuple(_ebr_key(name, position) for name, position in data["ebrs"])
    if installed != tables._ebrs:              # pragma: no cover - install-dependent
        raise InternalError(
            f"#{tables._number_str}: the {len(installed)} elementary band "
            "representations the installed irreptables lists differ from the "
            f"{len(tables._ebrs)} that the SRSI tables were converted against. "
            f"Install irreptables {tables.provenance.irreptables_version}.")

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
            f"#{tables._number_str}: the site irreps the table's formulas count induce "
            f"{len(by_column)} of the {len(tables._ebrs)} elementary band representations "
            "of the list. Report it, with the model and the call.")

    nu = [_ebr_dimension(name) for name, _position in data["ebrs"]]
    theta_zero = [[tables._theta_on_columns[i][j] for j in range(len(tables._ebrs))]
                  for i in tables._z_rows]
    band_form = _solve_left(theta_zero, nu)
    if band_form is None:
        raise InternalError(          # pragma: no cover - table-dependent
            f"#{tables._number_str}: on this group's EBR columns, the number of bands "
            "is not a linear form in the ZSRSIs. Report it, with the model and the "
            "call.")

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
                f"#{tables._number_str}: the invariants of one copy of the site irrep "
                f"({key._paper})_{key._letter}, whose band representation is a sum of "
                "others, are not all integers. Report it, with the model and the call.")
        dim = sum((band_form[i] * raw[r] for i, r in enumerate(tables._z_rows)),
                  Fraction(0))
        if dim.denominator != 1 or dim <= 0:
            raise InternalError(      # pragma: no cover - table-dependent
                f"#{tables._number_str}: the band count the ZSRSIs give for the site irrep "
                f"({key._paper})_{key._letter}, whose band representation is a sum of "
                f"others, is {dim}, which is not a positive integer. Report it, with the "
                "model and the call.")
        listed = entry_at[key._letter]
        keys.append(_MaximalKey(irrep=key._paper, letter=key._letter, dim=int(dim),
                                ebr=False, column=None, ebr_name=None,
                                ebr_label=alias(key),
                                term=replace(listed, irrep=key.irrep)))
        columns.append(tuple(int(value) for value in raw))
    return tuple(keys), tuple(columns), band_form
