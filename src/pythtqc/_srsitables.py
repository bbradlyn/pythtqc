"""The tables of the stable real-space invariants of Hwang et al., as shipped:
each grey group's rows over the site keys they are written on, and the paper's
names for the symmetry indicators against ``irreptables``' own.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
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

__all__ = ["DATA_DIR", "GATES", "SRSIRow", "SRSIIndicator", "SiteKey", "SplitEBR",
           "SRSITables", "srsi_status", "srsi_tables", "NAME_RELATIONS",
           "indicator_vocabularies", "to_irreptables_name", "from_irreptables_name",
           "NamePair", "name_correspondence", "canonical_determinism", "MaximalKey"]


#: Where the converted tables live inside the installed package.
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

#: ``ebr_data`` position strings: ``"12d(31',3)"`` -> multiplicity, letter, symbols.
_POSITION = re.compile(r"^(\d+)([a-zA-Z])\((.*)\)$")

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
    "srsi_not_atomic":
        "theta_Z is fractional, so no atomic insulator has this symmetry data and "
        "there is nothing to enumerate",
    "srsi_realizations_truncated":
        "the enumeration of atomic realizations hit its cap, and a no-go is asserted "
        "only on a complete enumeration",
    "orbit_content_incomplete":
        "a high-symmetry point refused for a reason other than a trivial little "
        "group, so the orbit's measured content is not the whole vector",
    "orbit_not_localizable":
        "the orbit's content is a band representation of nothing on the closure of "
        "its variety, and no composite site irrep explains it either",
    "wyckoff_setting":
        "geometry and content name different Wyckoff positions (srsi_of_model), or "
        "pyxtal's list is not in the tables' setting (wyckoff_sites_tables)",
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
        from irreptables.irreps import IrrepTable
        return str(IrrepTable(str(number_str), bool(spinor), magnetic=True).name)
    except (ImportError, OSError):              # pragma: no cover - tables not installed
        return ""


def _group(number_str, spinor) -> str:
    """``P6mm1' #183.186, double-valued``: the group a summary is of, and which of its tables."""
    symbol = _grey_symbol(str(number_str), bool(spinor))
    return (f"{symbol} " if symbol else "") + f"#{number_str}, {_valuedness(spinor)}"


def _values(values) -> str:
    """``(1, -1/2, 0)``: invariants as a summary prints them."""
    return "(" + ", ".join(str(v) for v in values) + ")"


def _by_modulus(theta_zn) -> str:
    """``mod 2 (0, 1), mod 4 (3)``: the Z_n-valued invariants, by modulus."""
    return ", ".join(f"mod {n} {_values(values)}" for n, values in sorted(theta_zn.items()))


def _indicator_line(indicator, whose) -> str:
    """A summary's line for the symmetry-indicator class of a band set."""
    if not indicator.divisors:
        return f"  {whose} symmetry-indicator group is trivial   (.indicator)"
    if not indicator.nontrivial:
        return (f"  {whose} symmetry-indicator class is zero, in {indicator.structure}"
                "   (.indicator)")
    return (f"  {whose} symmetry-indicator class: {indicator.values} in "
            f"{indicator.structure}, of order {indicator.order}   (.indicator)")


def _site_irreps(irreps, letter) -> str:
    """``B2 and B3 at Wyckoff position d``: site irreps at one position."""
    return f"{' and '.join(irreps)} at Wyckoff position {letter}"


#: How a summary reads a configuration of orbitals, ``1 (B2)_d + 2 (A1)_a``.


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
    """``2 (-1E2E)_e + 1 (-E)_a``: an invariant's terms, each a site irrep at its Wyckoff letter times its multiplicity."""
    return _signed((term["coef"], f"({term['irrep']})_{term['letter']}") for term in row.terms)


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SRSIRow:
    r"""One stable real-space invariant :math:`\theta_i` of a group: its formula over the site keys.

    :meth:`summary` prints the formula.

    Attributes
    ----------
    index : int
        The paper's index, from 1.
    name : str
        ``"theta_3"``.
    mod : int or None
        ``n`` for a :math:`\mathbb{Z}_n`-valued invariant, ``None`` for a
        :math:`\mathbb{Z}`-valued one.
    coefficients : tuple of int
        The formula on the group's elementary band representations, one
        coefficient per entry of :attr:`SRSITables.ebrs`.
    terms : tuple of dict
        Every term of the formula, ``{"coef", "irrep", "irrep_tex", "letter",
        "column", "maximal", "ebr"}``: the coefficient, the site irrep and its
        TeX, the Wyckoff letter, and the column of the elementary band
        representation the key induces, ``None`` where it induces a composite one (``ebr``),
        at a non-maximal position or at a maximal one whose band representation is
        a sum of others (``maximal`` says which).
    """

    index: int
    name: str
    mod: int | None
    coefficients: tuple
    terms: tuple = field(repr=False)

    def summary(self) -> str:
        """The invariant, then its formula, a line each."""
        return (f"{self.name}, {'Z' if self.mod is None else f'Z{self.mod}'}-valued: a sum "
                "of terms c (irrep)_w, c times the multiplicity of a site irrep at Wyckoff "
                "position w   (.name, .mod)\n"
                "  " + _formula(self) + "   (.terms)")

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        kind = "Z" if self.mod is None else f"Z{self.mod}"
        return f"SRSIRow({self.name}, {kind}, {_count(len(self.terms), 'term')})"


@dataclass(frozen=True)
class SRSIIndicator:
    r"""One symmetry indicator as the paper tabulates it: a form in the :math:`\theta_i`, modulo its own modulus.

    :meth:`summary` prints the form evaluated and where it comes from.

    Attributes
    ----------
    name, name_tex : str
        The paper's name, ``"z2w,1"``, and its TeX, ``"z_{2w,1}"``, in the naming
        of Song et al. 2018.
    mod : int
        The modulus.
    theta : dict
        ``{i: coefficient}``: the printed coefficient of each :math:`\theta_i`, an
        exact fraction.
    vanishes_on_ebrs : bool
        Whether the printed form is ``0`` modulo :attr:`mod` on every elementary
        band representation of the group, as an indicator is; ``False`` for four
        forms, all of SGs 147, 162 and 163 without spin-orbit coupling.
    ebr_failures : tuple
        ``(column, value)`` of each elementary band representation the printed
        form does not vanish on.
    derived : dict or None
        ``{i: coefficient}``: the form in the :math:`\theta_i` that equals, on
        every band set, ``irreptables``' momentum-space form of the same name, or
        the group's one Smith coordinate.  ``None`` where ``irreptables`` has no
        form of the name and the group no single Smith coordinate, as
        :attr:`derived_note` says.
    relation : str or None
        The printed form against the derived one: ``"equal"``, ``"equivalent"``
        (other coefficients, the same values on every band set modulo
        :attr:`mod`) or ``"different"``.
    source : str or None
        What the derived form was derived from, ``"irreptables:<name>"`` or
        ``"smith"``.
    derived_note : str
        The derivation in a sentence, or why there is none.
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
        """The form :meth:`SRSITables.indicators_of` evaluates: the printed one, or the derived one where :attr:`relation` is ``"different"``.

        Those are the four forms of SGs 147, 162 and 163 without spin-orbit coupling,
        which do not vanish on their own group's elementary band representations and
        so are not indicators as printed.
        """
        if self.relation == "different" and self.derived is not None:
            return dict(self.derived)
        return dict(self.theta)

    def summary(self) -> str:
        """The indicator, the form evaluated, and where that form comes from, a line each."""
        def form(row):
            return _signed((c, f"theta_{i}") for i, c in sorted(row.items()))

        source = ("the group's one Smith coordinate" if self.source == "smith" else
                  f"irreptables' formula named {str(self.source).partition(':')[2]}")
        whence = {
            "equal": f"the form the tables print, equal to the one derived from {source}",
            "equivalent": (f"the form the tables print, with the values modulo {self.mod} "
                           f"of the one derived from {source}"),
            "different": (f"the form derived from {source}: the one the tables print, "
                          f"{form(self.theta)}, does not vanish on every elementary band "
                          "representation, and is no indicator"),
        }.get(self.relation, "the form the tables print, with nothing to derive it from")
        attributes = {"equal": ".relation, .source", "equivalent": ".relation, .source",
                      "different": ".relation, .theta, .vanishes_on_ebrs"}.get(
                          self.relation, ".derived_note")
        return (f"{self.name}, modulo {self.mod}: a form in the stable real-space "
                "invariants theta_i   (.name, .mod)\n"
                f"  {form(self.coefficients)}   (.coefficients)\n"
                f"  {whence}   ({attributes})")

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        flag = "" if self.vanishes_on_ebrs else ", printed row not an indicator"
        rel = f", {self.relation}" if self.relation else ""
        return f"SRSIIndicator({self.name}, mod {self.mod}{rel}{flag})"


@dataclass(frozen=True)
class SiteKey:
    r"""One site key: a site irrep at a Wyckoff position, one coordinate of the site-symmetry vector ``p``.

    A key at a maximal position induces an elementary band representation, unless
    that band representation is a sum of others (:math:`{}^1E{}^2E` at site
    symmetry :math:`\bar4`, ``E`` at :math:`\bar4m2` and :math:`\bar42m`); a key at
    a non-maximal position induces a composite one.  A group's keys are
    :attr:`SRSITables.site_keys`.

    Attributes
    ----------
    irrep : str
        The site irrep, as the paper spells it.
    letter : str
        The Wyckoff letter, in the setting the tables are written in.
    maximal, ebr : bool
        Whether the position is maximal, and whether the key induces an
        elementary band representation.
    column : int or None
        That representation's column in ``ebr_data(group)["ebrs"]``.
    ebr_label : str or None
        That list's spelling of the irrep where it differs, ``A`` for the paper's
        ``A_1`` at a ``222`` site; :meth:`SRSITables.theta` takes either spelling,
        or the key itself.
    """

    irrep: str
    letter: str
    maximal: bool
    ebr: bool
    column: int | None = None
    ebr_label: str | None = None

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        kind = ("EBR column %d" % self.column if self.ebr
                else "maximal, composite band representation" if self.maximal
                else "non-maximal")
        return f"SiteKey(({self.irrep})_{self.letter}, {kind})"


@dataclass(frozen=True)
class SplitEBR:
    r"""One splitting of an elementary band representation into others, as the paper lists it.

    The paper lists the splittings the symmetry indicators allow, every part an
    elementary band representation, and marks whether the
    :math:`\mathbb{Z}_n`-valued invariants of the parts add up to the whole's.  The
    paper's result is that all of them are topological; :func:`..srsi.nsit_criterion`
    decides one over every atomic realization of its parts.  :meth:`summary`
    prints it.

    Attributes
    ----------
    parent : dict
        The whole, ``{"irrep", "irrep_tex", "letter", "column"}``: its site irrep,
        Wyckoff letter and column.
    children : tuple of dict
        The parts, two or more, each written the same way.
    srsi_mismatch : bool
        The paper's mark: ``True`` where the parts' :math:`\mathbb{Z}_n`-valued
        invariants do not add up to the whole's.
    """

    parent: dict
    children: tuple
    srsi_mismatch: bool

    def summary(self) -> str:
        """The splitting, and whether the invariants of its parts add up to the whole's, a line each."""
        def key(entry):
            return f"({entry['irrep']})_{entry['letter']}"

        return (f"{key(self.parent)} = {' + '.join(key(c) for c in self.children)}, a "
                "splitting the symmetry indicators allow, each term the elementary band "
                "representation of a site irrep at a Wyckoff position"
                "   (.parent, .children)\n"
                "  the Z_n-valued stable real-space invariants of the parts "
                f"{'do not add' if self.srsi_mismatch else 'add'} up to the whole's"
                "   (.srsi_mismatch)")


@dataclass(frozen=True)
class SRSITables:
    r"""The stable real-space invariants of one grey space group, as the paper tabulates them.

    What :func:`srsi_tables` returns; :meth:`summary` prints the invariants, a line
    each, and :meth:`formulas` each one's formula over the site irreps.
    :meth:`theta` evaluates them on a site-symmetry vector ``p`` over the
    site keys, :meth:`theta_columns` on one over the group's elementary band
    representations, and :meth:`indicators_of` and :meth:`indicators_of_columns`
    evaluate the paper's symmetry indicators.

    Attributes
    ----------
    sg, number_str : int, str
        The space-group number, and the grey group's label in the tables,
        ``"25.58"``.
    spinor : bool
        Which of the group's two tables: with spin-orbit coupling, double-valued,
        or without.
    rows : tuple of SRSIRow
        The invariants :math:`\theta_i`, in the paper's order.
    names, moduli : tuple
        Their names, and each one's modulus, ``None`` for a
        :math:`\mathbb{Z}`-valued one.
    ebrs : tuple
        ``(site irrep, Wyckoff letter)`` of each elementary band representation, in
        the order of ``ebr_data(group)["ebrs"]``: the columns.
    theta_on_columns : tuple of tuple of int
        :math:`\Theta` on those columns: ``theta_on_columns[i][j]`` is the
        coefficient of column ``j`` in :math:`\theta_{i+1}`.
    indicators : tuple of SRSIIndicator
        The paper's symmetry indicators, as forms in the :math:`\theta_i`.
    split_ebrs : tuple of SplitEBR
        The paper's splittings of an elementary band representation into others.
    terms_without_ebr_column : tuple of dict
        Every term on a site key that induces a composite band representation,
        with the index of its invariant, ``srsi``.
    provenance : dict
        Where the tables come from: the paper's DOI, its source files and their
        SHA-256, the map of its labels.
    """

    sg: int
    number_str: str
    spinor: bool
    rows: tuple
    theta_on_columns: tuple
    moduli: tuple
    names: tuple
    ebrs: tuple
    indicators: tuple
    split_ebrs: tuple
    terms_without_ebr_column: tuple = field(repr=False)
    provenance: dict = field(repr=False, default_factory=dict)

    # -- the shape of the invariant group ----------------------------------
    @property
    def z_rows(self) -> tuple:
        r"""Where the :math:`\mathbb{Z}`-valued invariants are in :attr:`rows`."""
        return tuple(i for i, m in enumerate(self.moduli) if m is None)

    @property
    def zn_rows(self) -> dict:
        r"""``{n: (row indices,)}``: where the :math:`\mathbb{Z}_n`-valued invariants are in :attr:`rows`, by modulus."""
        out: dict[int, list] = {}
        for i, mod in enumerate(self.moduli):
            if mod is not None:
                out.setdefault(int(mod), []).append(i)
        return {n: tuple(v) for n, v in sorted(out.items())}

    @property
    def structure(self) -> str:
        """The group the invariants take values in, ``"Z x Z x Z2"``."""
        parts = ["Z"] * len(self.z_rows)
        for n, rows in self.zn_rows.items():
            parts += [f"Z{n}"] * len(rows)
        return " x ".join(parts) or "trivial"

    # -- the arithmetic ----------------------------------------------------
    def theta_unreduced(self, p) -> tuple:
        r"""Every :math:`\theta_i` of ``p``, in table order, the :math:`\mathbb{Z}_n`-valued ones not reduced modulo ``n``.

        ``p`` is the mapping :meth:`theta` takes; a sequence over the columns is
        :meth:`theta_unreduced_columns`' and raises :class:`TypeError` here.

        Returns
        -------
        tuple of Fraction
            One per row of :attr:`rows`.
        """
        if not isinstance(p, Mapping):
            raise TypeError(
                "SRSITables.theta_unreduced() takes a {(site irrep, Wyckoff "
                "letter): multiplicity} mapping over the site keys; a sequence "
                "indexed by the EBR columns (SRSITables.ebrs) is "
                "theta_unreduced_columns()'s")
        return self._theta_of_sites(p)

    def theta_unreduced_columns(self, p) -> tuple:
        """:meth:`theta_unreduced` on a ``p`` over the columns, :attr:`ebrs`.

        A mapping is :meth:`theta_unreduced`'s and raises :class:`TypeError` here.
        """
        if isinstance(p, Mapping):
            raise TypeError(
                "SRSITables.theta_unreduced_columns() takes a sequence indexed by "
                "the EBR columns (SRSITables.ebrs); a {(site irrep, Wyckoff "
                "letter): multiplicity} mapping is theta_unreduced()'s")
        return self._theta_of_columns(p)

    def theta(self, p) -> tuple:
        r"""``(theta_Z, theta_Zn)``: the invariants of a site-symmetry vector ``p`` over the site keys.

        Parameters
        ----------
        p : mapping
            ``{(site irrep, Wyckoff letter): multiplicity}`` over any site keys of the
            group, maximal or not, a key spelled as the paper or as the list of
            elementary band representations spells it, or one of :attr:`site_keys`.
            Integer for an atomic configuration, rational for a band set with a
            nonzero symmetry indicator.

        Returns
        -------
        tuple
            ``theta_Z``, the :math:`\mathbb{Z}`-valued invariants in table order as
            exact fractions, integral exactly when the symmetry indicator is zero; and
            ``theta_Zn``, ``{n: values}``, each modulo ``n``, or an unreduced fraction
            where it is not an integer.  ``theta_Zn`` belongs to this ``p``, not to its
            symmetry vector: it is the band set's only for a configuration that is
            known, never for one solved back out of a band set's content.

        Raises
        ------
        TypeError
            ``p`` is a sequence, :meth:`theta_columns`' shape.
        ValueError
            A key the tables do not carry; the message lists those at its letter.

        Examples
        --------
        >>> tables.theta({("-E", "a"): 1})                  # doctest: +SKIP
        ((Fraction(1, 1),), {2: (0, 0, 0)})
        """
        if not isinstance(p, Mapping):
            raise TypeError(
                "SRSITables.theta() takes a {(site irrep, Wyckoff letter): "
                "multiplicity} mapping over the site keys; a sequence indexed by "
                "the EBR columns (SRSITables.ebrs) is theta_columns()'s")
        return self._reduced(self._theta_of_sites(p))

    def theta_columns(self, p) -> tuple:
        """:meth:`theta` on a ``p`` over the group's elementary band representations, :attr:`ebrs`.

        The shape a decomposition writes ``p`` in, exact for a configuration at those
        positions alone.  A mapping is :meth:`theta`'s and raises :class:`TypeError`
        here.
        """
        if isinstance(p, Mapping):
            raise TypeError(
                "SRSITables.theta_columns() takes a sequence indexed by the EBR "
                "columns (SRSITables.ebrs); a {(site irrep, Wyckoff letter): "
                "multiplicity} mapping is theta()'s")
        return self._reduced(self._theta_of_columns(p))

    def _reduced(self, raw) -> tuple:
        """``(theta_Z, theta_Zn)`` from the unreduced rows."""
        z_values = tuple(raw[i] for i in self.z_rows)
        zn = {}
        for n, rows in self.zn_rows.items():
            zn[n] = tuple(int(raw[i]) % n if raw[i].denominator == 1 else raw[i]
                          for i in rows)
        return z_values, zn

    def _theta_of_columns(self, p) -> tuple:
        """:meth:`theta_unreduced_columns`' arithmetic, without the type gate."""
        vec = _as_fractions(p, len(self.ebrs))
        return tuple(sum((Fraction(c) * v for c, v in zip(row, vec)), Fraction(0))
                     for row in self.theta_on_columns)

    def indicators_of(self, p) -> dict:
        """``{name: value}``: the paper's symmetry indicators of ``p``, each modulo its own modulus.

        ``p`` is the mapping :meth:`theta` takes; a sequence over the columns is
        :meth:`indicators_of_columns`' and raises :class:`TypeError` here.  A value is
        an integer, or an unreduced fraction where it is not one.  Each form is
        :attr:`SRSIIndicator.coefficients`: the printed one, or the derived one for the
        four forms of SGs 147, 162 and 163 without spin-orbit coupling that are not
        indicators as printed.
        """
        if not isinstance(p, Mapping):
            raise TypeError(
                "SRSITables.indicators_of() takes a {(site irrep, Wyckoff letter): "
                "multiplicity} mapping over the site keys; a sequence indexed by "
                "the EBR columns (SRSITables.ebrs) is indicators_of_columns()'s")
        return self._indicators_of_raw(self._theta_of_sites(p))

    def indicators_of_columns(self, p) -> dict:
        """:meth:`indicators_of` on a ``p`` over the group's elementary band representations, :attr:`ebrs`.

        The shape a ``p`` solved out of a symmetry vector comes in.  A mapping is
        :meth:`indicators_of`'s and raises :class:`TypeError` here.
        """
        if isinstance(p, Mapping):
            raise TypeError(
                "SRSITables.indicators_of_columns() takes a sequence indexed by "
                "the EBR columns (SRSITables.ebrs); a {(site irrep, Wyckoff "
                "letter): multiplicity} mapping is indicators_of()'s")
        return self._indicators_of_raw(self._theta_of_columns(p))

    def _indicators_of_raw(self, raw) -> dict:
        """:meth:`indicators_of`' arithmetic on the unreduced rows."""
        out = {}
        for form in self.indicators:
            total = sum((c * raw[i - 1] for i, c in form.coefficients.items()),
                        Fraction(0))
            out[form.name] = (int(total) % form.mod if total.denominator == 1
                              else total)
        return out

    # -- the full site-symmetry vector -------------------------------------
    @property
    def site_keys(self) -> tuple:
        """Every site key the invariants count, a site irrep at a Wyckoff position, maximal or not: the coordinates of ``p``.

        Returns
        -------
        tuple of SiteKey
            Sorted by Wyckoff letter, then by irrep.
        """
        seen = {}
        for row in self.rows:
            for term in row.terms:
                key = (str(term["irrep"]), str(term["letter"]))
                if key not in seen:
                    seen[key] = SiteKey(
                        irrep=key[0], letter=key[1],
                        maximal=bool(term["maximal"]), ebr=bool(term["ebr"]),
                        column=(None if term["column"] is None
                                else int(term["column"])),
                        ebr_label=term.get("ebr_label"))
        return tuple(seen[k] for k in sorted(seen))

    def _site_lookup(self) -> dict:
        """``{(spelling, letter): (irrep, letter)}`` — both spellings of a key."""
        out = {}
        for key in self.site_keys:
            out[(key.irrep, key.letter)] = (key.irrep, key.letter)
            if key.ebr_label:
                out[(key.ebr_label, key.letter)] = (key.irrep, key.letter)
        return out

    def _theta_of_sites(self, p) -> tuple:
        """:meth:`theta_unreduced` on a ``p`` keyed by ``(site irrep, letter)``."""
        lookup = self._site_lookup()
        vector: dict = {}
        for key, value in dict(p).items():
            if isinstance(key, SiteKey):
                key = (key.irrep, key.letter)
            if not (isinstance(key, tuple) and len(key) == 2):
                raise ValueError(
                    f"the site-symmetry vector is keyed by (site irrep, Wyckoff "
                    f"letter) pairs or by the SiteKey records site_keys returns; "
                    f"got {key!r}")
            resolved = lookup.get((str(key[0]), str(key[1])))
            if resolved is None:
                here = sorted({k.irrep for k in self.site_keys
                               if k.letter == str(key[1])})
                letters = sorted({k.letter for k in self.site_keys})
                raise ValueError(
                    f"#{self.number_str} "
                    f"({'double' if self.spinor else 'single'}-valued) has no site "
                    f"irrep {str(key[0])!r} at Wyckoff letter {str(key[1])!r}. "
                    + (f"At {str(key[1])!r} the tables carry {here}."
                       if here else
                       f"The tables carry no Wyckoff letter {str(key[1])!r} at all; "
                       f"they carry {letters}."))
            vector[resolved] = vector.get(resolved, Fraction(0)) + Fraction(value)
        out = []
        for row in self.rows:
            total = Fraction(0)
            for term in row.terms:
                got = vector.get((str(term["irrep"]), str(term["letter"])))
                if got:
                    total += Fraction(int(term["coef"])) * got
            out.append(total)
        return tuple(out)

    # -- the basis of the atomic realizations: the maximal positions -------
    @property
    def maximal_keys(self) -> tuple:
        r"""Every maximal site key, with the bands one copy of it carries: the coordinates :func:`..srsi.atomic_realizations` lists configurations over.

        The keys with an elementary band representation come first, in the order of
        :attr:`ebrs`, then the maximal ones whose band representation is a sum of
        others.  Maximal positions are enough: an atomic insulator moves onto them
        with its invariants unchanged.

        Returns
        -------
        tuple of MaximalKey

        Raises
        ------
        ..refusal.InternalError
            The installed list of elementary band representations is not the one the
            tables were converted against, or the band count is not a form in the
            :math:`\mathbb{Z}`-valued invariants: a defect to report.
            :func:`srsi_tables` refuses ``srsi_ebr_basis`` for the first before this
            is reached.
        """
        return _maximal_basis(int(self.sg), bool(self.spinor))[0]

    @property
    def band_count_form(self) -> tuple:
        r"""The band count as a form in the :math:`\mathbb{Z}`-valued invariants: the ``a`` with ``a . theta_Z`` the number of bands.

        The band count is itself an invariant, and the form is unique; its value on a
        key whose band representation is a sum of others is that key's band count
        (:attr:`MaximalKey.dim`).  ``#211.57`` reads :math:`\nu = \theta_1 + \theta_2
        + 2\theta_3 + 3\theta_4 + 3\theta_5 + 3\theta_6 + 4\theta_7`.

        Returns
        -------
        tuple of Fraction
            One per :attr:`z_rows`.
        """
        return _maximal_basis(int(self.sg), bool(self.spinor))[2]

    def row(self, name: str) -> SRSIRow | None:
        """The invariant named ``name``, ``"theta_3"``, or ``None``."""
        for row in self.rows:
            if row.name == name:
                return row
        return None

    def summary(self) -> str:
        """The invariants, row by row, and what else the table holds, a line each."""
        lines = [f"{_group(self.number_str, self.spinor)}: stable real-space invariants "
                 f"{self.structure}   (.structure)",
                 "  each invariant, Z- or Z_n-valued, and the number of terms in its "
                 "formula   (.rows)"]
        lines += [f"  {row.name:>9s}  {'Z' if row.mod is None else 'Z%d' % row.mod:>3s}"
                  f"  {_count(len(row.terms), 'term')}" for row in self.rows]
        lines.append(f"  {_count(len(self.site_keys), 'site irrep')} at Wyckoff positions, "
                     f"{len(self.ebrs)} of them inducing an elementary band representation"
                     "   (.site_keys, .ebrs)")
        if self.terms_without_ebr_column:
            lines.append(f"  {_count(len(self.terms_without_ebr_column), 'term')} on site "
                         "irreps that induce a composite band representation"
                         "   (.terms_without_ebr_column)")
        lines.append("  " + (f"{_count(len(self.indicators), 'symmetry indicator')}, each "
                             "a form in the theta_i: "
                             + ", ".join(f"{f.name} (mod {f.mod})" for f in self.indicators)
                             if self.indicators else
                             "no symmetry indicator, the group's indicator group being "
                             "trivial")
                     + "   (.indicators)")
        if self.split_ebrs:
            lines.append(f"  {len(self.split_ebrs)} splitting"
                         f"{'' if len(self.split_ebrs) == 1 else 's'} of an elementary band "
                         "representation into others that the symmetry indicators allow"
                         "   (.split_ebrs)")
        return "\n".join(lines)

    def formulas(self) -> str:
        r"""Every invariant's formula, a line each, as the paper tabulates them.

        ``theta_2 (Z2) = 1 (-E)_b``: the invariant and whether it is
        :math:`\mathbb{Z}`- or :math:`\mathbb{Z}_n`-valued, then its terms, each
        a site irrep at its Wyckoff letter times its multiplicity.
        """
        heads = [f"{row.name} ({'Z' if row.mod is None else f'Z{row.mod}'})"
                 for row in self.rows]
        width = max((len(head) for head in heads), default=0)
        return "\n".join(f"{head:<{width}} = {_formula(row)}"
                         for head, row in zip(heads, self.rows))

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return (f"SRSITables(#{self.number_str}, "
                f"{'double' if self.spinor else 'single'}-valued, "
                f"{len(self.rows)} SRSIs = {self.structure}, "
                f"{len(self.ebrs)} EBR columns, "
                f"{len(self.indicators)} indicators, "
                f"{len(self.split_ebrs)} split EBRs)")


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
        paths, and ``reason`` says what is missing and how to rebuild it.
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
            f"the converted SRSI tables {missing} are not installed in {DATA_DIR}. "
            "They are package data generated from the paper's TeX supplement, which is "
            "not shipped: run scripts/convert_srsi_tables.py '<supple info folder>' "
            "--out src/pythtqc/data")
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
                         f"{n} EBR columns; it must be indexed by the group's own EBR "
                         "list (SRSITables.ebrs)")
    return tuple(Fraction(v) for v in values)


def _lettered(entry: dict) -> dict:
    """One term or site of the converted data, its Wyckoff letter under ``letter``."""
    return {("letter" if key == "wyckoff" else key): value for key, value in entry.items()}


def _build(record: dict, provenance: dict) -> SRSITables:
    n_ebr = len(record["ebrs"])
    rows, matrix, moduli, names, orphans = [], [], [], [], []
    for entry in record["srsis"]:
        coefficients = [0] * n_ebr
        terms = []
        for term in entry["terms"]:
            terms.append(_lettered(term))
            if term["column"] is None:
                orphans.append(dict(_lettered(term), srsi=entry["index"]))
            else:
                coefficients[term["column"]] += int(term["coef"])
        rows.append(SRSIRow(index=int(entry["index"]), name=str(entry["name"]),
                            mod=None if entry["mod"] is None else int(entry["mod"]),
                            coefficients=tuple(coefficients), terms=tuple(terms)))
        matrix.append(tuple(coefficients))
        moduli.append(rows[-1].mod)
        names.append(rows[-1].name)
    indicators = tuple(
        SRSIIndicator(name=str(e["name"]), name_tex=str(e["name_tex"]),
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
    splits = tuple(SplitEBR(parent=_lettered(e["parent"]),
                            children=tuple(_lettered(c) for c in e["children"]),
                            srsi_mismatch=bool(e["srsi_mismatch"]))
                   for e in record["split_ebrs"])
    return SRSITables(
        sg=int(record["sg"]), number_str=str(record["number_str"]),
        spinor=record["valuedness"] == "double",
        rows=tuple(rows), theta_on_columns=tuple(matrix), moduli=tuple(moduli),
        names=tuple(names),
        ebrs=tuple((str(a), str(b)) for a, b in record["ebrs"]),
        indicators=indicators, split_ebrs=splits,
        terms_without_ebr_column=tuple(orphans), provenance=provenance)


def _table_of_number(sg: int, spinor: bool) -> SRSITables | Refusal:
    """The SRSI table of a space-group **number** — :func:`srsi_tables`' second shape.

    Refuses ``srsi_data`` (the converted files are not installed),
    ``srsi_scope`` (``sg`` outside 1..230) or ``srsi_table_absent`` (one of the
    thirteen groups the tables omit, with the caption's reason).
    """
    if not 1 <= int(sg) <= 230:
        return Refusal(
            "srsi_scope",
            f"space group {sg} is not one of the 230; the SRSI tables run over those",
            value=float(sg), context={"sg": int(sg)})
    status = srsi_status()
    if not status["data"]:
        return _refuse_no_data(status)
    document = _document("double" if spinor else "single")
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
    return _build(record, provenance)


def srsi_tables(group_or_number, *, spinor=None) -> SRSITables | Refusal:
    """The stable real-space invariants of a grey space group, as the paper tabulates them.

    From a measured group the tables come in that group's order of elementary band
    representations, so a ``p`` built anywhere in the package drops into
    :meth:`SRSITables.theta`; from a space-group number with ``spinor=`` they need
    no model.

    Parameters
    ----------
    group_or_number : ..groups.GroupData or int
        A grey group a model was measured to have, or a space-group number from
        1 to 230.
    spinor : bool, optional
        With a number, and only then: the tables with spin-orbit coupling
        (``True``) or without.

    Returns
    -------
    SRSITables or Refusal
        ``tables.summary()`` prints the invariants; ``tables.theta(p)`` evaluates
        them on a configuration.  A ``Refusal`` in its place is falsy.

    Raises
    ------
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
    >>> srsi_tables(25, spinor=True).structure             # doctest: +SKIP
    'Z x Z2 x Z2 x Z2'
    """
    if isinstance(group_or_number, (int, np.integer)):
        if spinor is None:
            raise ValueError(
                f"srsi_tables({int(group_or_number)}) is a space-group number, and the "
                "tables come in two valuednesses: pass spinor=True or spinor=False. "
                "With a measured GroupData the flag comes off the group itself")
        return _table_of_number(int(group_or_number), bool(spinor))
    group = group_or_number
    # The two grammar checks come FIRST and stay hard errors: declaring the
    # valuedness twice is a mistake in the call, not a fact about the crystal,
    # and it is wrong whatever lattice the group is on.  The lattice gate is a
    # physics verdict and follows.
    if spinor is not None:
        own = bool(getattr(group, "spinor", False))
        raise ValueError(
            f"srsi_tables() was given a group and spinor={spinor!r}. A GroupData "
            f"already declares its valuedness (spinor={own!r} here), and it is the "
            "group's own flag that decides which table its EBR columns belong to; "
            "pass spinor= only with a bare space-group number")
    gate = _spacegroup._lattice_not_descended(group)
    if gate is not None:
        return gate
    mtype = _spacegroup._magnetic_type(group)
    if mtype != "II":
        return Refusal(
            "srsi_scope",
            f"#{group.number_str} is a magnetic group of type {mtype}, and the shipped "
            "SRSI tables are the time-reversal-symmetric (grey, type-II) ones. A group "
            "is returned grey exactly where the Hamiltonian obeys time reversal, and "
            "for a model that breaks it no SRSI tables exist",
            context={"number_str": str(group.number_str), "magnetic_type": mtype})
    table = _table_of_number(int(group.sg_number), bool(group.spinor))
    if not table:
        return table

    data = _tables.ebr_data(group)
    if not data:
        return data
    installed = tuple(_ebr_key(name, position) for name, position in data["ebrs"])
    if installed != table.ebrs or str(group.number_str) != table.number_str:
        return Refusal(
            "srsi_ebr_basis",
            f"the EBR list installed for #{group.number_str} is not the one the SRSI "
            f"conversion resolved against (#{table.number_str}, {len(table.ebrs)} "
            f"columns against {len(installed)}), so its stored column indices would "
            "name the wrong EBRs. Re-run scripts/convert_srsi_tables.py against this "
            "irreptables installation",
            value=float(len(installed)), threshold=float(len(table.ebrs)),
            context={"number_str": str(group.number_str),
                     "converted_against": table.number_str,
                     "installed": tuple(installed), "expected": table.ebrs,
                     "irreptables_version": table.provenance.get(
                         "irreptables_version")})
    return table


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
                 else load_si_table(spinor=valuedness == "double", magnetic=True))
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
        from irreptables.symmetry_indicators import load_si_table
        entry = load_si_table(spinor=tables.spinor, magnetic=True).get(
            tables.number_str, {})
        forms = dict(entry.get("indicators", {}))
    except Exception as exc:                   # pragma: no cover - table defect
        return Refusal("irreptables_indicator_tables",
                       f"irreptables' indicator table could not be read: {exc}",
                       context={"number_str": tables.number_str})
    out, paired = [], set()
    for form in tables.indicators:
        spelled, relation = _spelling(form.name)
        if spelled not in forms:
            out.append(NamePair(form.name, None, "paper_only", form.mod,
                                f"irreptables ships no form spelled {spelled!r} for "
                                f"#{tables.number_str}"))
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
                            f"the paper's table for #{tables.number_str} carries no "
                            f"form spelled {_unspelling(spelled)[0]!r}"))
    return tuple(out)


# ---------------------------------------------------------------------------
# the canonical class: the paper's names, and the gate that lets them speak it
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def _canonical_gate(number_str: str, spinor: bool) -> dict:
    r"""The two convention checks behind :attr:`..ebr.SymmetryIndicator.canonical`.

    Computed once per system and cached, because both are properties of the
    *tables* and of nothing else.

    **Every returned form is an indicator of this group.**  A linear form descends
    to :math:`\{BS\}/\{AI\}` exactly when it vanishes modulo its own modulus on
    every EBR column -- an EBR is an atomic limit -- and that is checked here on
    the row :meth:`SRSITables.indicators_of` evaluates
    (:attr:`SRSIIndicator.coefficients`: the printed row, or the derived one
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
    data = _tables.ebr_data(_TableKey(number_str, spinor))
    if not data:
        return {"available": False, "gate": data.gate, "reason": data.reason}
    u, v, d, matrix = _smith._smith_system(data, str(number_str))
    n_row, n_col = len(matrix), len(matrix[0])
    slots = tuple(i for i, x in enumerate(d) if x > 1)
    divisors = tuple(int(d[i]) for i in slots)

    failures = []
    for form in tables.indicators:
        row = form.coefficients
        for column in range(n_col):
            total = sum((c * Fraction(tables.theta_on_columns[i - 1][column])
                         for i, c in row.items()), Fraction(0))
            if total.denominator != 1 or int(total) % form.mod:
                failures.append((str(form.name), int(column), str(total)))

    generators, collisions, fractional = (), (), ()
    n_classes = 0
    if divisors and tables.indicators:
        u_inv = _smith._unimodular_inverse(u)
        values_of = []
        for slot in slots:
            y = [Fraction(int(u_inv[i][slot])) for i in range(n_row)]
            values_of.append(tables.indicators_of_columns(_rational_p(u, v, d, y)))
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
                    for form in tables.indicators)
                n_classes += 1
                if reading in seen:
                    hits.append((seen[reading], combination, dict(reading)))
                seen[reading] = combination
            collisions = tuple(hits)
    return {"available": True, "number_str": str(number_str), "spinor": bool(spinor),
            "divisors": divisors, "slots": slots,
            "names": tuple(str(f.name) for f in tables.indicators),
            "moduli": tuple(int(f.mod) for f in tables.indicators),
            "vanishes": not failures, "ebr_failures": tuple(failures),
            "integral": not fractional, "fractional": fractional,
            "n_classes": int(n_classes), "injective": not collisions,
            "collisions": collisions, "generators": generators}


def canonical_determinism(group_or_tables) -> dict | Refusal:
    """Whether the paper's indicator names tell apart every class of a group's symmetry-indicator group.

    What :attr:`..ebr.SymmetryIndicator.canonical` rests on: every form vanishes on
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
    return dict(_canonical_gate(tables.number_str, tables.spinor))


def _canonical_reading(group, magnetic_type, u, v, d, y, divisors) -> tuple:
    r"""``(canonical, note, tex)`` -- the class in the paper's names.

    What :func:`..ebr.symmetry_indicator` calls to fill
    :attr:`..ebr.SymmetryIndicator.canonical`.  ``y`` is the returned symmetry
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
            "the canonical vocabulary is the SRSI paper's, whose shipped tables are "
            f"the time-reversal-symmetric (grey, type-II) ones; #{group.number_str} is "
            f"a magnetic group of type {magnetic_type}, so no canonical name is "
            "returned. The Smith class above is unaffected: it is computed from this "
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
        return None, ("no canonical vocabulary is returned here: "
                      f"{tables.reason}"), {}
    if not divisors:
        return None, (
            f"the indicator group of #{group.number_str} is trivial, so there is no "
            "class to speak in any vocabulary; the paper's tables carry an indicator "
            "row for exactly the 170 grey systems that have one, and this is not one "
            "of them"), {}
    if not tables.indicators:
        raise InternalError(
            f"#{group.number_str}: the indicator group is {divisors} and the paper's "
            "table for this system carries no indicator row, so the canonical "
            "vocabulary has no word for this class. The two lists were measured to "
            "coincide exactly (170 systems), so this is a defect in the shipped data "
            "or in this package's use of it.")
    gate = _canonical_gate(tables.number_str, tables.spinor)
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
            f"({gate['collisions'][0][:2]}). The canonical vocabulary cannot speak "
            "this group's classes, which is a finding about the tables and not a "
            "number to print.")
    p = _rational_p(u, v, d, [Fraction(int(x)) for x in y])
    canonical = tables.indicators_of_columns(p)
    tex = {str(f.name): str(f.name_tex) for f in tables.indicators}
    note = (
        f"the class in the canonical vocabulary of the SRSI paper ({_DOI}), evaluated "
        f"as the tabulated forms in the Z-SRSIs on the exact rational p of this "
        f"symmetry vector. The paper's names determine the class here: all "
        f"{gate['n_classes']} elements of {'x'.join('Z%d' % x for x in divisors)} have "
        f"different canonical values, and every one of the {len(tables.indicators)} "
        "forms vanishes on every EBR column of this group. irreptables' own names for "
        "the same classes are `irreptables_named`, and `as_irreptables()` re-keys these "
        "values into them")
    return canonical, note, tex


@dataclass(frozen=True)
class MaximalKey:
    """One maximal site key, with the bands one copy of it carries: a coordinate of the atomic realizations.

    The keys of a group are :attr:`SRSITables.maximal_keys`.

    Attributes
    ----------
    irrep, letter : str
        The site irrep, as the paper spells it, and the Wyckoff letter.
    dim : int
        The bands one copy carries: its elementary band representation's, or, for
        a key whose band representation is a sum of others, the value of
        :attr:`SRSITables.band_count_form` on it.
    ebr : bool
        Whether it induces an elementary band representation.
    column, ebr_name, ebr_label : int, str, str or None
        That representation's column and name in ``ebr_data(group)["ebrs"]``, and
        that list's spelling of the irrep where it differs.
    """

    irrep: str
    letter: str
    dim: int
    ebr: bool
    column: int | None = None
    ebr_name: str | None = None
    ebr_label: str | None = None

    @property
    def label(self) -> str:
        """``"(B2)_d"``, the paper's way of writing the key."""
        return f"({self.irrep})_{self.letter}"

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        kind = (f"column {self.column}" if self.ebr
                else "composite band representation")
        return f"MaximalKey({self.label}, {self.dim} bands, {kind})"


class _TableKey:
    """The three fields :func:`..groups.ebr_data` reads, from a table's own record.

    The SRSI tables are keyed by a grey BNS number and a valuedness, which is
    exactly what :func:`..groups.ebr_data` needs to return the columns the
    conversion resolved against -- so a table in hand can fetch its own EBR list
    without a measured ``GroupData``, which is what
    :attr:`SRSITables.maximal_keys` and a sweep over all 230 space groups both
    need.
    """

    def __init__(self, number_str, spinor):
        self.number_str = str(number_str)
        self.spinor = bool(spinor)
        self.magnetic = True


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
    data = _tables.ebr_data(_TableKey(tables.number_str, spinor))
    if not data:                              # pragma: no cover - install-dependent
        raise InternalError(
            f"#{tables.number_str}: the EBR list this table's columns index into "
            f"cannot be returned ({data}), so the maximal-position basis cannot be "
            "built.")
    installed = tuple(_ebr_key(name, position) for name, position in data["ebrs"])
    if installed != tables.ebrs:              # pragma: no cover - install-dependent
        raise InternalError(
            f"#{tables.number_str}: the installed EBR list is not the one the SRSI "
            f"conversion resolved against ({len(installed)} columns against "
            f"{len(tables.ebrs)}), so the stored column indices name the wrong EBRs "
            "and the atomic realizations would be written in the wrong coordinates. "
            "srsi_tables() refuses 'srsi_ebr_basis' for exactly this; re-run "
            "scripts/convert_srsi_tables.py against this irreptables installation.")

    by_column, composite = {}, []
    for key in tables.site_keys:
        if not key.maximal:
            continue
        if key.ebr:
            by_column[int(key.column)] = key
        else:
            composite.append(key)
    if set(by_column) != set(range(len(tables.ebrs))):
        raise InternalError(          # pragma: no cover - table-dependent
            f"#{tables.number_str}: the table's rows touch {len(by_column)} of the "
            f"{len(tables.ebrs)} EBR columns. Every column must appear, because the "
            "band count is a Z-linear form in the Z-SRSIs and an EBR carries bands.")

    nu = [_ebr_dimension(name) for name, _position in data["ebrs"]]
    theta_zero = [[tables.theta_on_columns[i][j] for j in range(len(tables.ebrs))]
                  for i in tables.z_rows]
    band_form = _solve_left(theta_zero, nu)
    if band_form is None:
        raise InternalError(          # pragma: no cover - table-dependent
            f"#{tables.number_str}: the band count is not a Z-linear form in the "
            "Z-SRSIs on this group's EBR columns, though it is one in all 434 "
            "tabulated systems as measured -- the band count is a stable real-space "
            "invariant, so it has to be. Either the shipped Theta or the EBR band "
            "counts are not what they are taken to be.")

    keys, columns = [], []
    for j in range(len(tables.ebrs)):
        key = by_column[j]
        keys.append(MaximalKey(irrep=key.irrep, letter=key.letter, dim=int(nu[j]),
                               ebr=True, column=j,
                               ebr_name=str(data["ebrs"][j][0]),
                               ebr_label=key.ebr_label))
        columns.append(tuple(int(row[j]) for row in tables.theta_on_columns))
    for key in composite:
        raw = tables.theta_unreduced({key: 1})
        if any(value.denominator != 1 for value in raw):
            raise InternalError(      # pragma: no cover - table-dependent
                f"#{tables.number_str}: the Theta column of the composite site irrep "
                f"({key.irrep})_{key.letter} is not integral.")
        dim = sum((band_form[i] * raw[r] for i, r in enumerate(tables.z_rows)),
                  Fraction(0))
        if dim.denominator != 1 or dim <= 0:
            raise InternalError(      # pragma: no cover - table-dependent
                f"#{tables.number_str}: the band-count form gives {dim} bands for the "
                f"composite site irrep ({key.irrep})_{key.letter}, which is not a "
                "positive integer. A site irrep whose induced band representation is "
                "composite still induces a definite number of bands.")
        keys.append(MaximalKey(irrep=key.irrep, letter=key.letter, dim=int(dim),
                               ebr=False, column=None, ebr_name=None,
                               ebr_label=key.ebr_label))
        columns.append(tuple(int(value) for value in raw))
    return tuple(keys), tuple(columns), band_form
