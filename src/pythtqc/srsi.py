r"""Stable real-space invariants (SRSIs): the tables, their values on bands and models, and a no-go criterion for a gapped splitting.

An SRSI is a linear form :math:`\theta = \Theta\cdot p` in the site-symmetry
vector ``p``, how many times each site irrep occurs at each Wyckoff position,
that no symmetric deformation of the Wannier functions changes: two atomic
configurations share every SRSI exactly when they deform into each other once
the same atomic orbitals are added to both.  The :math:`\mathbb{Z}`-valued
ones, ``theta_Z``, are a function of the symmetry vector, fractional exactly
when the symmetry indicator is nonzero.  The :math:`\mathbb{Z}_n`-valued ones,
``theta_Zn``, are not, and tell apart atomic limits with the same content at
every high-symmetry point.

The tables are those of Hwang et al., *Stable real-space invariants and
topology beyond symmetry indicators*, Nat. Commun. **17**, 8564 (2026),
https://doi.org/10.1038/s41467-026-74844-w, for the grey space groups with
spin-orbit coupling and without, shipped with the package; the thirteen groups
whose one invariant is the band count have none.

- :func:`srsi_tables`, the tables of one group, and :func:`srsi_of` and
  :func:`srsi_of_columns`, their values on a configuration in hand;
- :func:`srsi_of_bands`, the :math:`\mathbb{Z}`-valued invariants of a band
  set, from its symmetry vector, and :func:`srsi_of_model`, both kinds, from a
  model's orbitals;
- :func:`splitting`, the paper's no-go criterion on a gapped splitting of a
  model's bands, with :func:`nsit_criterion`, the criterion on invariants in
  hand, and :func:`atomic_realizations`, every atomic insulator with given
  invariants;
- :func:`stably_equivalent_pairs`, site irreps no invariant distinguishes;
- :func:`srsi_status`, whether the tables are installed, and :data:`GATES`,
  every refusal the calls here return;
- :func:`to_irreptables_name`, :func:`from_irreptables_name`,
  :func:`name_correspondence` and :func:`canonical_determinism`, the paper's
  names for the symmetry indicators against ``irreptables``'.

``docs/guide/srsi.md`` reads them on models.
"""

from __future__ import annotations

import functools
import itertools
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

import numpy as np

from . import ebr as _ebr
from . import groups as _groups
from .refusal import Refusal
from .symmetry import _index_ranges

__all__ = ["SRSITables", "SRSIRow", "SRSIIndicator", "SiteKey", "SplitEBR",
           "BandSRSIs", "ModelSRSIs", "OrbitSRSIs",
           "MaximalKey", "StablePair", "AtomicRealization", "Realizations",
           "Splitting", "SPLITTING_VERDICTS",
           "GATES",
           "srsi_status", "srsi_tables", "srsi_of", "srsi_of_columns",
           "srsi_of_bands", "srsi_of_model",
           "atomic_realizations", "nsit_criterion", "splitting",
           "stably_equivalent_pairs",
           "NamePair", "NAME_RELATIONS", "indicator_vocabularies",
           "to_irreptables_name", "from_irreptables_name", "name_correspondence",
           "canonical_determinism"]

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
_CONFIGURATION = "written m (irrep)_w, m copies of a site irrep at Wyckoff position w"


def _signed(terms) -> str:
    """``2 x - 4 y + 1/2 z``: a sum of ``(coefficient, symbol)``, its signs written out."""
    out = []
    for coefficient, symbol in terms:
        coefficient = Fraction(coefficient)
        sign = ("-" if coefficient < 0 else "") if not out else (
            " - " if coefficient < 0 else " + ")
        out.append(f"{sign}{abs(coefficient)} {symbol}")
    return "".join(out) or "0"


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
        representation the key induces, ``None`` where it induces none (``ebr``),
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
                "of terms c (irrep)_w, c times the copies of a site irrep at Wyckoff "
                "position w   (.name, .mod)\n"
                "  " + _signed((term["coef"], f"({term['irrep']})_{term['letter']}")
                               for term in self.terms)
                + "   (.terms)")

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
    a non-maximal position induces none.  A group's keys are
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
    paper's result is that all of them are topological; :func:`nsit_criterion`
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
    each.  :meth:`theta` evaluates them on a site-symmetry vector ``p`` over the
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
        Every term on a site key that induces no elementary band representation,
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
        r"""Every maximal site key, with the bands one copy of it carries: the coordinates :func:`atomic_realizations` lists configurations over.

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
                 "  each invariant theta_i, Z- or Z_n-valued, and how many terms its "
                 "formula has, each term counting the copies of one site irrep at one "
                 "Wyckoff position   (.rows)"]
        lines += [f"  {row.name:>9s}  {'Z' if row.mod is None else 'Z%d' % row.mod:>3s}"
                  f"  {_count(len(row.terms), 'term')}" for row in self.rows]
        lines.append(f"  {_count(len(self.site_keys), 'site irrep')} at Wyckoff positions, "
                     f"{len(self.ebrs)} of them inducing an elementary band representation"
                     "   (.site_keys, .ebrs)")
        if self.terms_without_ebr_column:
            lines.append(f"  {_count(len(self.terms_without_ebr_column), 'term')} on site "
                         "irreps that induce none: at a non-maximal Wyckoff position, or at "
                         "a maximal one whose band representation is a sum of others"
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
    gate = _groups._lattice_not_descended(group)
    if gate is not None:
        return gate
    mtype = _groups._magnetic_type(group)
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

    data = _groups.ebr_data(group)
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


# ---------------------------------------------------------------------------
# a band subset -> its Z-SRSIs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BandSRSIs:
    r"""The :math:`\mathbb{Z}`-valued stable real-space invariants of a band set, read off its symmetry vector.

    What :func:`srsi_of_bands` returns; :meth:`summary` prints it.  It holds no
    :math:`\mathbb{Z}_n`-valued invariants: those are no function of the symmetry
    vector.

    Attributes
    ----------
    theta_Z : tuple of Fraction
        The invariants, in table order; fractional exactly when the symmetry
        indicator is nonzero.
    names : tuple of str
        Their names, ``theta_1``, ...
    stable : bool
        Whether any is fractional, the band set stable topological.
    denominator : int
        Their common denominator.
    indicator : ..ebr.SymmetryIndicator
        The symmetry-indicator class measured with them.
    p : tuple of Fraction
        The rational site-symmetry vector over the columns that was evaluated, one
        of many with this symmetry vector.
    n_bands : int or None
        The number of bands.
    number_str, spinor : str, bool
        The group, and which of its two tables.
    """

    number_str: str
    spinor: bool
    theta_Z: tuple
    names: tuple
    stable: bool
    denominator: int
    indicator: Any = field(repr=False, default=None)
    p: tuple = field(repr=False, default=())
    n_bands: int | None = None

    def summary(self) -> str:
        """The invariants, one per line, then what they say, a line each."""
        bands = "the bands" if self.n_bands is None else _count(self.n_bands, "band")
        lines = [f"{_group(self.number_str, self.spinor)}: theta_Z, the Z-valued "
                 f"stable real-space invariants of {bands}   (.theta_Z)"]
        lines += [f"  {name:>9s} = {str(value):>8s}"
                  for name, value in zip(self.names, self.theta_Z)]
        lines.append(f"  fractional, with common denominator {self.denominator}: the bands "
                     "are stable topological   (.stable, .denominator)"
                     if self.stable else
                     "  integral: the bands are not stable topological   (.stable)")
        if self.indicator is not None:
            lines.append(_indicator_line(self.indicator, "their"))
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return (f"BandSRSIs(#{self.number_str}, "
                f"({', '.join(str(v) for v in self.theta_Z)}), "
                f"{'stable' if self.stable else 'integral'})")


def srsi_of_bands(content, group, *, rep=None, points=None,
                  bands=None) -> BandSRSIs | Refusal:
    r"""The :math:`\mathbb{Z}`-valued stable real-space invariants of a band set, from its symmetry vector.

    They are the invariants of a rational site-symmetry vector ``p`` with the band
    set's symmetry vector, every such ``p`` giving the same ones, and they are
    fractional exactly when the symmetry indicator is nonzero.  The
    :math:`\mathbb{Z}_n`-valued invariants are no function of the symmetry vector,
    so none are returned; :func:`srsi_of_model` reads them off a model's orbitals.

    Parameters
    ----------
    content : ..sewing.HSPTable or mapping
        The band set's content at the high-symmetry points, as
        :func:`..ebr.symmetry_indicator` takes it.
    group : ..groups.GroupData
        A grey group with tables.
    rep : ..symmetry.SpaceGroupRep, optional
        The model's representation, for :func:`..ebr.symmetry_indicator`; a layer
        needs it for the rows out of its plane.
    points : sequence of str, optional
        The high-symmetry points to read, for :func:`..ebr.symmetry_indicator`.
    bands : int or sequence of int, optional
        Which bands the content counts: an ``int`` is the lowest *n*, a sequence
        those bands.  An :class:`~..sewing.HSPTable` says so itself, and another
        set raises :class:`ValueError`.

    Returns
    -------
    BandSRSIs or Refusal
        ``out.summary()`` prints the invariants, whether they are fractional, and
        the symmetry-indicator class; ``out.theta_Z``, ``out.stable`` and
        ``out.indicator`` hold them.  A ``Refusal`` in its place is falsy.

    Raises
    ------
    ..refusal.InternalError
        The rational solve does not reproduce the symmetry vector it was built
        from: a defect to report.
    ValueError
        ``bands=`` names other bands than the content was measured on.

    Refusals
    --------
    ``srsi_scope``, ``srsi_table_absent``, ``srsi_data``, ``srsi_ebr_basis``
        from :func:`srsi_tables`: the group has no tables.
    ``incomplete_content``, ``not_a_band_structure``, ``indicator_not_evaluable``
        from :func:`..ebr.symmetry_indicator`, with the refusals of the content it
        reads.
    """
    block = _ebr._bands_of(content, bands, where="srsi_of_bands")
    gate = _groups._lattice_not_descended(group)
    if gate is not None:
        return gate
    tables = srsi_tables(group)          # a measured GroupData, not an SRSITables:
    if not tables:                       # the momentum-space half needs the group
        return tables
    si = _ebr.symmetry_indicator(content, group, rep=rep, points=points,
                                 bands=block)
    if not si:
        return si
    data = _groups.ebr_data(group)
    if not data:
        return data
    u, v, d, matrix = _ebr._smith_system(data, group.number_str)
    p = _rational_p(u, v, d, [Fraction(int(x)) for x in si.y])
    n_row, n_col = len(matrix), len(matrix[0])
    check = [sum((Fraction(matrix[i][j]) * p[j] for j in range(n_col)), Fraction(0))
             for i in range(n_row)]
    if check != [Fraction(int(x)) for x in si.y]:
        from .refusal import InternalError
        raise InternalError(
            f"#{group.number_str}: the rational solve of BR.p = y does not reproduce "
            "the symmetry vector it was built from, though the compatibility relations "
            "hold (symmetry_indicator checked them). The Smith factors and the EBR "
            "matrix this package returns are not a decomposition of each other, which "
            "is arithmetic here failing rather than a measurement about the model.")
    theta_z = tables.theta_columns(p)[0]
    denominator = 1
    for value in theta_z:
        denominator = denominator * value.denominator // _gcd(denominator,
                                                              value.denominator)
    return BandSRSIs(
        number_str=str(tables.number_str), spinor=bool(tables.spinor),
        theta_Z=theta_z, names=tuple(tables.names[i] for i in tables.z_rows),
        stable=any(value.denominator != 1 for value in theta_z),
        denominator=int(denominator), indicator=si, p=tuple(p),
        n_bands=None if block is None else len(block))


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


def srsi_of(group_or_tables, p) -> tuple | Refusal:
    r"""``(theta_Z, theta_Zn)``: the stable real-space invariants of a configuration ``p`` over the site keys.

    :meth:`SRSITables.theta` as a call, on a measured group or its tables.

    Parameters
    ----------
    group_or_tables : ..groups.GroupData or SRSITables
        A measured grey group, or its tables.
    p : mapping
        ``{(site irrep, Wyckoff letter): multiplicity}``: integer for an atomic
        configuration, rational for a band set with a nonzero symmetry indicator.

    Returns
    -------
    tuple or Refusal
        ``theta_Z``, the :math:`\mathbb{Z}`-valued invariants as exact
        fractions in table order, and ``theta_Zn``, ``{n: values}``, the
        :math:`\mathbb{Z}_n`-valued ones modulo ``n``.  ``theta_Zn`` belongs to
        this ``p``: it is a band set's only for a configuration that is known.  A
        ``Refusal`` in its place is falsy.

    Raises
    ------
    TypeError
        A bare space-group number, which carries no valuedness
        (:func:`srsi_tables` takes it with ``spinor=``), or a sequence ``p``,
        which is :func:`srsi_of_columns`' shape.

    Refusals
    --------
    ``srsi_data``, ``srsi_scope``, ``srsi_table_absent``, ``srsi_ebr_basis``
        from :func:`srsi_tables`: the group has no tables.
    """
    _no_bare_number(group_or_tables, "srsi_of")
    if not isinstance(p, Mapping):
        raise TypeError(
            "srsi_of() takes a {(site irrep, Wyckoff letter): multiplicity} mapping "
            "over the site keys; a sequence indexed by the group's EBR columns is "
            "srsi_of_columns()'s")
    gate = _groups._lattice_not_descended(group_or_tables)
    if gate is not None:
        return gate
    tables = (group_or_tables if isinstance(group_or_tables, SRSITables)
              else srsi_tables(group_or_tables))
    if not tables:
        return tables
    return tables.theta(p)


def srsi_of_columns(group_or_tables, p) -> tuple | Refusal:
    """``(theta_Z, theta_Zn)`` of a configuration ``p`` over the group's elementary band representations.

    :meth:`SRSITables.theta_columns` as a call: :func:`srsi_of` on the shape a
    decomposition writes ``p`` in, exact for a configuration at those positions
    alone.

    Parameters
    ----------
    group_or_tables : ..groups.GroupData or SRSITables
        As for :func:`srsi_of`.
    p : sequence
        One multiplicity per entry of :attr:`SRSITables.ebrs`.

    Returns
    -------
    tuple or Refusal
        ``(theta_Z, theta_Zn)``, as :func:`srsi_of` returns them.  A ``Refusal``
        in its place is falsy.

    Raises
    ------
    TypeError
        A bare space-group number, or a mapping ``p``, :func:`srsi_of`'s shape.
    ValueError
        A ``p`` of another length.

    Refusals
    --------
    ``srsi_data``, ``srsi_scope``, ``srsi_table_absent``, ``srsi_ebr_basis``
        from :func:`srsi_tables`: the group has no tables.
    """
    _no_bare_number(group_or_tables, "srsi_of_columns")
    if isinstance(p, Mapping):
        raise TypeError(
            "srsi_of_columns() takes a sequence indexed by the group's EBR columns; "
            "a {(site irrep, Wyckoff letter): multiplicity} mapping is srsi_of()'s")
    gate = _groups._lattice_not_descended(group_or_tables)
    if gate is not None:
        return gate
    tables = (group_or_tables if isinstance(group_or_tables, SRSITables)
              else srsi_tables(group_or_tables))
    if not tables:
        return tables
    return tables.theta_columns(p)


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
    gate = _groups._lattice_not_descended(group_or_tables)
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
    data = _groups.ebr_data(_TableKey(number_str, spinor))
    if not data:
        return {"available": False, "gate": data.gate, "reason": data.reason}
    u, v, d, matrix = _ebr._smith_system(data, str(number_str))
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
        u_inv = _ebr._unimodular_inverse(u)
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
    gate = _groups._lattice_not_descended(group_or_tables)
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
    :func:`srsi_of_bands` solves for -- one arithmetic, two records.

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
    from .refusal import InternalError

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


# ---------------------------------------------------------------------------
# a model -> its SRSIs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OrbitSRSIs:
    """One site orbit of a model: its orbitals, where they sit, what they carry, and its stable real-space invariants.

    Attributes
    ----------
    orbit : int
        Its index in ``assignment.orbits``.
    orbitals, sites : tuple of int
        Its orbitals, and the sites they sit at.
    letter, multiplicity, n_free : str, int, int
        Its Wyckoff position in the setting the tables are written in
        (:func:`..groups.wyckoff_sites_tables`), and that position's free
        parameters.
    closure : tuple of str
        The letters its orbitals can slide onto, on the closure of the set of
        points its position covers, that the tables list elementary band
        representations at; its own among them where they list one there.
    content : dict
        ``{corep: multiplicity}``: the coreps its states carry, measured at
        :attr:`points`.
    points : tuple of str
        The high-symmetry points the content was measured at.
    localizations : tuple of dict
        Every decomposition of the content into elementary band representations
        at a letter of :attr:`closure`, ``{"letter", "terms", "p", "theta_Z",
        "theta_Zn"}``; all give the same invariants.
    chosen : dict
        The one returned.
    agree : bool
        ``True``: localizations that disagree raise
        :class:`~..refusal.InternalError`.
    same_content : tuple
        Elementary band representations the content does not tell apart, which
        carry the same invariants.
    composite : dict or None
        ``{"irreps", "site"}``: the site irreps read off the representation, for
        an orbit whose band representation is a sum no list carries.
    theta_Z, theta_Zn : tuple, dict
        Its invariants.
    """

    orbit: int
    orbitals: tuple
    sites: tuple
    letter: str
    multiplicity: int
    n_free: int
    closure: tuple
    content: dict
    points: tuple
    localizations: tuple
    chosen: dict
    agree: bool
    theta_Z: tuple
    theta_Zn: dict
    same_content: tuple = ()
    composite: dict | None = None

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return (f"OrbitSRSIs(orbitals {list(self.orbitals)} @ "
                f"{self.multiplicity}{self.letter}, "
                f"{len(self.localizations)} localization(s))")


@dataclass(frozen=True)
class ModelSRSIs:
    r"""A model's stable real-space invariants, read off its orbitals, site orbit by site orbit.

    What :func:`srsi_of_model` returns; :meth:`summary` prints it.  A model is an
    atomic configuration: its invariants are integral, and its
    :math:`\mathbb{Z}_n`-valued ones are its own.

    Attributes
    ----------
    theta_Z : tuple of Fraction
        The :math:`\mathbb{Z}`-valued invariants, summed over the orbits; the same
        as :func:`srsi_of_bands` on all the model's bands.
    theta_Zn : dict
        ``{n: values}``: the :math:`\mathbb{Z}_n`-valued ones, modulo ``n``.
    orbits : tuple of OrbitSRSIs
        Each site orbit's.
    names, moduli, structure : tuple, tuple, str
        The names of the :math:`\mathbb{Z}`-valued invariants, every invariant's
        modulus, and the group they take values in.
    tables : SRSITables
        The tables they were read in.
    number_str, spinor : str, bool
        The group, and which of its two tables.
    """

    number_str: str
    spinor: bool
    orbits: tuple
    theta_Z: tuple
    theta_Zn: dict
    names: tuple
    moduli: tuple
    structure: str
    tables: Any = field(repr=False, default=None)

    def summary(self) -> str:
        """The invariants, then each site orbit's, a line each."""
        def both(theta_z, theta_zn):
            return _values(theta_z) + (f", {_by_modulus(theta_zn)}" if theta_zn else "")

        lines = [f"{_group(self.number_str, self.spinor)}: stable real-space invariants "
                 f"{self.structure}   (.structure)",
                 f"  the model's: Z-valued {_values(self.theta_Z)}"
                 + (f", Z_n-valued {_by_modulus(self.theta_Zn)}   (.theta_Z, .theta_Zn)"
                    if self.theta_Zn else "   (.theta_Z)")]
        def slid(orbit):
            return orbit.composite is None and orbit.letter not in orbit.closure

        moved = any(slid(orbit) for orbit in self.orbits)
        lines.append("  each site orbit: its orbitals, its Wyckoff position, and its content, "
                     "written m (irrep)_w, m copies of a site irrep at Wyckoff position w"
                     + (", at a position it slides onto where the tables list no elementary "
                        "band representation at its own" if moved else "")
                     + "; then its invariants   (.orbits)")
        # each site irrep as the invariants' formulas spell it; the list of
        # elementary band representations spells a few otherwise, A for A1 at 222
        spelling = self.tables._site_lookup() if self.tables is not None else {}

        def term(m, irrep, letter):
            irrep, letter = spelling.get((irrep, letter), (irrep, letter))
            return f"{m} ({irrep})_{letter}"

        for orbit in self.orbits:
            terms = " + ".join(term(m, *_ebr_key(name, w))
                               for m, name, w in orbit.chosen.get("terms", ()))
            if orbit.composite is not None:
                terms = " + ".join(term(m, lab, orbit.letter) for lab, m
                                   in sorted(orbit.composite["irreps"].items()))
            where = f"{orbit.multiplicity}{orbit.letter}"
            if slid(orbit):
                # the position it slid onto, as the elementary band representations name it
                onto = next((match.group(1) + match.group(2)
                             for _m, _name, position in orbit.chosen.get("terms", ())
                             for match in [_POSITION.match(str(position))] if match),
                            orbit.chosen.get("letter", ""))
                where += f", slid onto {onto}"
            lines.append(f"    orbitals {_index_ranges(orbit.orbitals)} at {where}: "
                         f"{terms}")
            lines.append(f"      {both(orbit.theta_Z, orbit.theta_Zn)}")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return (f"ModelSRSIs(#{self.number_str}, {len(self.orbits)} orbit(s), "
                f"theta_Z=({', '.join(str(v) for v in self.theta_Z)}))")


def _unpack_realized(realized):
    """``(model, group, assignment, rep)`` from a RealizedGroup or a sequence."""
    if all(hasattr(realized, name)
           for name in ("model", "group", "assignment", "rep")):
        return (realized.model, realized.group, realized.assignment, realized.rep)
    parts = tuple(realized)
    if len(parts) == 5:                      # the RealizedGroup unpacking order
        return parts[0], parts[1], parts[2], parts[4]
    if len(parts) == 4:
        return parts
    raise ValueError(
        "srsi_of_model takes a RealizedGroup, or (model, group, assignment, rep), "
        f"or (model, group, assignment, validated, rep); got {len(parts)} items")


def _orbit_content(group, rep, rows):
    r"""The symmetry content of one block of **states**, measured.

    ``rows`` are state indices (``orbital * nspin + spin``); the content is
    :func:`_orbit_contents` of that one block.

    Returns ``(counts, points, refusals)``; ``counts`` is ``None`` when the
    group has no usable high-symmetry point list, and ``refusals`` is then the
    refusal itself.
    """
    return _orbit_contents(group, rep, [rows])[0]


def _orbit_contents(group, rep, blocks):
    r"""The symmetry content of each block of **states**, measured.

    Each block is a list of state indices (``orbital * nspin + spin``).  At
    every tabulated high-symmetry point with a non-trivial little group, the
    character of a block under each unitary little-group operation is the trace
    of that operation's **sewing operator** restricted to the block,
    ``trace(rep.sewing_operator(ind, k)[rows, rows])``, and
    :func:`..groups.identify_from_characters` turns those characters into corep
    multiplicities against the group's own transported tables.  Each operator
    is built once, and every block is read off it.

    ``sewing.characters`` and ``sewing.hsp_table`` are deliberately *not* used
    here: they gate on the supplied states diagonalizing ``H(k)``, which is what
    makes them honest about a band block — and a block of *orbitals* is not a
    block of bands.  The gate stays where it is; this route computes the same
    trace on a set of states whose meaning is already fixed.  The operators
    check the orbitals' positions at their own tolerance
    (:func:`..symmetry.bloch_operator`'s ``tol``), as every other use of them does.

    Returns one ``(counts, points, refusals)`` per block; ``counts`` is ``None``
    when the group has no usable high-symmetry point list, and ``refusals`` is
    then the refusal itself.
    """
    blocks = [list(rows) for rows in blocks]
    points_map = _groups.high_symmetry_points(group)
    if not points_map:
        return [(None, None, points_map) for _rows in blocks]
    read = [({}, [], {}) for _rows in blocks]
    for name in sorted(points_map):
        k = points_map[name]
        lg = _groups.little_group(group, k)
        why = None
        if isinstance(lg, Refusal):
            why = lg
        elif len(lg.unitary) <= 1:
            why = "trivial little group: no content to identify"
        operators = []
        for sym in (() if why is not None else lg.unitary):
            operator = rep.sewing_operator(int(sym.ind), k)
            if isinstance(operator, Refusal):
                why = operator
                break
            operators.append(np.asarray(operator))
        if why is not None:
            for _counts, _points, refusals in read:
                refusals[name] = why
            continue
        for rows, (counts, points, refusals) in zip(blocks, read):
            chars = [complex(np.trace(operator[np.ix_(rows, rows)]))
                     for operator in operators]
            got = _groups.identify_from_characters(group, k, [chars],
                                                   multiplet_dims=[len(rows)])
            if isinstance(got, Refusal):
                refusals[name] = got
                continue
            points.append(str(got.kpname))
            for label, multiplicity in got.labels[0].items():
                counts[str(label)] = counts.get(str(label), 0) + int(multiplicity)
    return [(counts, tuple(points), refusals) for counts, points, refusals in read]


def _solve_at_letter(data, counts, points, letter, bound, cap=64, bands=None):
    """Nonnegative-integer decompositions of a block's content at one letter.

    :func:`_solve_at_letters` at that one letter.
    """
    return _solve_at_letters(data, counts, points, (letter,), bound, cap, bands)


def _solve_at_letters(data, counts, points, letters, bound, cap=64, bands=None):
    """Nonnegative-integer decompositions of a block's content at some letters.

    The EBR matrix restricted twice: to the columns induced from ``letters``,
    and to the rows at the k-points the content was identified at (a 2D model
    in a 3D group has no out-of-plane rows to compare).  With ``bands``, the
    block's band count is one more equation: the one a content read at no
    point still fixes, in a group whose every little group is the identity
    alone.  ``ebr._cpsat`` enumerates, and reports its own status.
    """
    labels = list(data["labels_physics_convention"])
    keep = [i for i, label in enumerate(labels)
            if _ebr.kpoint_of_label(label) in set(points)]
    columns = [j for j, (_name, position) in enumerate(data["ebrs"])
               if _POSITION.match(str(position)).group(2) in set(letters)]
    if not columns or (not keep and bands is None):
        return "INFEASIBLE", (), columns, keep
    matrix = np.asarray(data["matrix"], dtype=int)[np.ix_(keep, columns)]
    y = np.array([int(counts.get(labels[i], 0)) for i in keep], dtype=int)
    if bands is not None:
        matrix = np.vstack([matrix,
                            [[_ebr._bands_spanned(data["ebrs"][j][0]) for j in columns]]])
        y = np.append(y, int(bands))
    status, solutions, _truncated = _ebr._cpsat(matrix, y, 0, int(bound), cap)
    return status, solutions, columns, keep


def srsi_of_model(realized, *, rep=None) -> "ModelSRSIs | Refusal":
    """A model's stable real-space invariants, read off its orbitals.

    Each site orbit's content is measured on its own orbitals and decomposed into
    elementary band representations at its Wyckoff position or, where the tables
    list none there, at a position it slides onto; the invariants are that
    decomposition's, summed over the orbits.  A model's orbitals are a known
    configuration, so both kinds are returned.

    Parameters
    ----------
    realized : ..groups.RealizedGroup
        What :func:`..groups.realized_group` returned, or its parts
        ``(model, group, assignment, rep)`` or
        ``(model, group, assignment, validated, rep)``.
    rep : ..symmetry.SpaceGroupRep, optional
        A representation to read in place of the one ``realized`` holds.

    Returns
    -------
    ModelSRSIs or Refusal
        ``out.summary()`` prints the invariants and each orbit's; ``out.theta_Z``
        and ``out.theta_Zn`` hold the model's, ``out.orbits`` each orbit's.  A
        ``Refusal`` in its place is falsy.

    Raises
    ------
    ..refusal.InternalError
        Two localizations of one orbit give different invariants: a defect to
        report.

    Refusals
    --------
    ``srsi_scope``, ``srsi_table_absent``, ``srsi_data``, ``srsi_ebr_basis``
        from :func:`srsi_tables`: the group has no tables.
    ``orbit_content_incomplete``
        an orbit's content could not be read at a high-symmetry point.
    ``orbit_not_localizable``
        an orbit's content is a band representation of no position its orbitals
        can slide onto.
    ``wyckoff_setting``
        the content decomposes at a position the geometry says the orbitals are
        not at: the package disagreeing with itself, a case to report.
    ``tables_not_ready``, ``wyckoff_unresolved``, ``ebr_tables``
        from :func:`..groups.wyckoff_sites_tables` and :func:`..groups.ebr_data`.

    Examples
    --------
    >>> print(srsi_of_model(realized).summary())             # doctest: +SKIP
    """
    model, group, assignment, own_rep = _unpack_realized(realized)
    gate = _groups._lattice_not_descended(group)
    if gate is not None:
        return gate
    rep = own_rep if rep is None else rep
    tables = srsi_tables(group)
    if not tables:
        return tables
    data = _groups.ebr_data(group)
    if not data:
        return data
    sites = _groups.wyckoff_sites_tables(group, assignment.site_positions)
    if isinstance(sites, Refusal):
        return sites
    # a free-standing layer's orbitals are moved within the layer only: the
    # positions off it are never a localization
    layer = _ebr._layer_reading(group, data, rep)
    off_layer = (set() if layer is None
                 else {_ebr._letter_only(position) for position in layer["outside"]})

    nspin = int(getattr(model, "nspin", 1))
    orbit_of_orbital = np.asarray(assignment.orbit_of_orbital, dtype=int)
    records = []
    for index, orbit in enumerate(assignment.orbits):
        orbitals = tuple(int(o) for o in range(int(model.norb))
                         if int(orbit_of_orbital[o]) == index)
        rows = [o * nspin + s for o in orbitals for s in range(nspin)]
        counts, points, refusals = _orbit_content(group, rep, rows)
        if counts is None:
            return refusals
        hard = {name: why for name, why in refusals.items()
                if not isinstance(why, str)}
        if hard:
            name, why = sorted(hard.items())[0]
            return Refusal(
                "orbit_content_incomplete",
                f"orbit {index} (orbitals {list(orbitals)}) could not be identified at "
                f"the high-symmetry point {name} of #{group.number_str}: "
                f"{getattr(why, 'reason', why)}. The orbit's symmetry content is "
                "incomplete, and a decomposition of an incomplete vector is a "
                "decomposition of a different vector.",
                context={"orbit": index, "orbitals": orbitals, "kpname": name,
                         "refusal": why, "identified": points,
                         "refusals": tuple(sorted(refusals))})

        geometry = [sites[s] for s in orbit]
        letters = {site.letter for site in geometry}
        if len(letters) != 1:                     # pragma: no cover - guard
            from .refusal import InternalError
            raise InternalError(
                f"#{group.number_str}: the sites of orbit {index} sit on different "
                f"Wyckoff letters {sorted(letters)} in the tables' reference cell, "
                "though they are one orbit of the group's own action. The orbit "
                "structure and the reference-cell letters cannot both be right.")
        here = geometry[0]

        # the letters the tables' EBR columns are keyed by, on the closure of
        # the orbit's variety: its own among them where they list one there,
        # and within the layer on a free-standing one
        localizations, same_content = [], ()
        tried = tuple(letter for letter in here.closure if letter not in off_layer)
        for letter in tried:
            _status, solutions, columns, _keep = _solve_at_letter(
                data, counts, points, letter, len(rows))
            if len(solutions) > 1:
                same_content = tuple(sorted(
                    {str(data["ebrs"][columns[j]][0])
                     for solution in solutions for j, m in enumerate(solution) if m}))
            for solution in solutions:
                vector = [0] * len(data["ebrs"])
                for j, multiplicity in zip(columns, solution):
                    vector[j] = int(multiplicity)
                theta_z, theta_zn = tables.theta_columns(vector)
                localizations.append({
                    "letter": letter,
                    "terms": tuple((int(m), str(data["ebrs"][j][0]),
                                    str(data["ebrs"][j][1]))
                                   for j, m in enumerate(vector) if m),
                    "p": tuple(vector), "theta_Z": theta_z, "theta_Zn": theta_zn})

        composite = None
        if not localizations:
            composite = _composite_localization(tables, group, assignment, rep,
                                                orbit, here)
            if composite is None:
                elsewhere = (_letters_that_do_fit(data, counts, points, len(rows))
                             if tried else ())
                if elsewhere:
                    return _route_disagreement(group, index, orbitals, here,
                                               elsewhere, counts)
                return Refusal(
                    "orbit_not_localizable",
                    f"orbit {index} (orbitals {list(orbitals)}) of #{group.number_str} "
                    f"sits at Wyckoff {here.multiplicity}{here.letter} and "
                    + ("no position the tables list elementary band representations "
                       "at lies on the closure of its variety, so there is nowhere "
                       "to localize it at all"
                       if not tried else
                       f"its measured content {counts} is not a nonnegative-integer "
                       "sum of the EBRs at "
                       + (here.letter if tried == (here.letter,) else str(list(tried))))
                    + ", and its site irreps are not among the composite ones the "
                    "tables carry either. An orbit whose content is not a band "
                    "representation of the position it sits on is a statement about the "
                    "model.",
                    context={"orbit": index, "orbitals": orbitals,
                             "content": dict(counts), "letter": here.letter,
                             "closure": here.closure, "tried": tuple(tried),
                             "points": points,
                             "number_str": str(group.number_str)})
            theta_z, theta_zn = tables.theta(
                {(label, here.letter): m
                 for label, m in composite["irreps"].items()})
            localizations = [{"letter": here.letter, "terms": (), "p": (),
                              "theta_Z": theta_z, "theta_Zn": theta_zn}]

        values = {(entry["theta_Z"],
                   tuple(sorted((n, v) for n, v in entry["theta_Zn"].items())))
                  for entry in localizations}
        if len(values) != 1:
            from .refusal import InternalError
            raise InternalError(
                f"#{group.number_str}: the localizations of orbit {index} give "
                f"{len(values)} different SRSIs ("
                + str([(e["letter"], tuple(str(v) for v in e["theta_Z"]))
                       for e in localizations])
                + "), though every one of them is the same block of orbitals moved "
                "adiabatically within the variety it sits on. An SRSI that is not "
                "invariant under a process the tables themselves list is a defect in "
                "this package's arithmetic or in the shipped table, not a measurement "
                "about the model.")
        chosen = localizations[0]
        records.append(OrbitSRSIs(
            orbit=index, orbitals=orbitals, sites=tuple(int(s) for s in orbit),
            letter=here.letter, multiplicity=here.multiplicity, n_free=here.n_free,
            closure=here.closure, content=dict(counts),
            points=points, localizations=tuple(localizations), chosen=chosen,
            agree=(len(values) == 1), theta_Z=chosen["theta_Z"],
            theta_Zn=chosen["theta_Zn"],
            same_content=same_content, composite=composite))

    total_z = tuple(sum((record.theta_Z[i] for record in records), Fraction(0))
                    for i in range(len(tables.z_rows)))
    total_zn = {}
    for n, rows_n in tables.zn_rows.items():
        total_zn[n] = tuple(
            sum(int(record.theta_Zn[n][i]) for record in records) % n
            for i in range(len(rows_n)))
    return ModelSRSIs(
        number_str=str(tables.number_str), spinor=bool(tables.spinor),
        orbits=tuple(records), theta_Z=total_z, theta_Zn=total_zn,
        names=tuple(tables.names[i] for i in tables.z_rows),
        moduli=tuple(tables.moduli), structure=tables.structure, tables=tables)


def _letters_that_do_fit(data, counts, points, bound) -> tuple:
    """Which letters of the EBR list the block's content *does* decompose at, if any.

    Run only when the geometry's own letters produced nothing, and only to make
    the refusal sharper: "this content is not a band representation of anything"
    and "this content is a band representation of a position the geometry says
    the orbitals are not on" are different diagnoses, and the second one names a
    setting problem.
    """
    letters = sorted({_POSITION.match(str(position)).group(2)
                      for _name, position in data["ebrs"]})
    return tuple(letter for letter in letters
                 if _solve_at_letter(data, counts, points, letter, bound, cap=1)[1])


def _route_disagreement(group, index, orbitals, here, elsewhere, counts) -> Refusal:
    """The one safety gate: geometry and content name different positions."""
    return Refusal(
        "wyckoff_setting",
        f"orbit {index} (orbitals {list(orbitals)}) of #{group.number_str} sits at "
        f"Wyckoff {here.multiplicity}{here.letter} by geometry (carried into the "
        f"tables' reference cell) and its measured content decomposes at "
        f"{list(elsewhere)} instead -- not at that letter"
        + (f", nor at any of {list(here.closure)}, the positions with elementary "
           "band representations on the closure of its variety"
           if tuple(here.closure) != (here.letter,) else "")
        + ". The two routes to a Wyckoff position disagree, so the setting the letters "
        "are read in is not the one the tables are written in, and no SRSI is returned",
        context={"orbit": index, "orbitals": orbitals,
                 "geometry_letter": here.letter, "content_letters": tuple(elsewhere),
                 "closure": here.closure, "reference": here.reference,
                 "content": dict(counts), "number_str": str(group.number_str)})


def _band_representations(rep) -> tuple:
    r"""The band representation each orbit of ``rep`` induces, in elementary band representations.

    The orbit's block of states is read as :func:`srsi_of_model` reads it: its
    symmetry content measured on the states (:func:`_orbit_contents`), and
    decomposed at the positions its orbitals can be moved onto, one position at
    a time, its own first -- induction in stages makes each such position's
    decomposition the orbit's band representation.  Its alternatives are the
    other sums of elementary band representations with the same symmetry
    content and the same stable real-space invariants: the content and those
    invariants do not distinguish them from it.  Where the group has no
    tabulated invariants there are none.  A site irrep whose band
    representation is a sum at other positions -- an irrep of a position the
    tables list none at, or ``E`` at ``3c`` of ``P-43m`` -- is written as the
    sum with its own invariants (:func:`_composite_localization`); without
    those, at the first position on the closure of a variety through its site
    that carries the content, the most special variety first, since a site
    irrep induced from that variety's site group is moved along it; and
    otherwise as the one sum its content has.  On a free-standing layer only
    the layer's own band representations are read, those of the positions
    within it (:func:`..ebr._layer_reading`).
    What cannot be read is said in the orbit's ``note``: this reads a
    representation already built, and nothing in it is a condition for building
    one.

    Returns
    -------
    tuple of ..symmetry.BandRepresentation
        In the order of ``assignment.orbits``, one per orbit that carries
        orbitals; empty where the tables do not reach the cell, and where the
        representation leaves some unitary operation of its group out — a
        ``_build_rep`` handed a group larger than the one the Hamiltonian obeys
        — since the tables are the whole group's.
    """
    from .symmetry import BandRepresentation, _label_terms

    characters = getattr(rep, "characters", None)
    group, assignment, model = rep.group, rep.assignment, rep.model
    if characters is None:
        return ()
    represented = {int(i) for i in rep.indices}
    if any(int(s.ind) not in represented for s in group.symmetries if not s.time_reversal):
        return ()
    sites = _groups.wyckoff_sites_tables(group, assignment.site_positions)
    if isinstance(sites, Refusal):
        return ()
    data = _groups.ebr_data(group)
    if isinstance(data, Refusal):
        return ()
    ebrs = tuple((str(name), str(position)) for name, position in data["ebrs"])
    # a free-standing layer's own band representations, and the positions off
    # it that its orbitals are never moved onto
    layer = _ebr._layer_reading(group, data, rep)
    columns = list(range(len(ebrs))) if layer is None else list(layer["columns"])
    off_layer = (set() if layer is None
                 else {_ebr._letter_only(position) for position in layer["outside"]})
    tables = srsi_tables(group)
    tables = tables if isinstance(tables, SRSITables) else None
    symmetry_of_site ={int(o.site): str(o.site_symmetry) for o in characters.orbitals}
    orbit_of_orbital = np.asarray(assignment.orbit_of_orbital, dtype=int)
    nspin = int(getattr(model, "nspin", 1))
    blocks = []
    for index, orbit in enumerate(assignment.orbits):
        orbitals = tuple(int(o) for o in range(int(model.norb))
                         if int(orbit_of_orbital[o]) == index)
        if orbitals:
            blocks.append((orbit, orbitals,
                           [o * nspin + s for o in orbitals for s in range(nspin)]))
    # The operators check the orbitals' positions against the operations' action
    # on them, at their own tolerance, as every other use of these operators
    # does.  The centres a representation is built on are exact -- moved there
    # under a declared precision, and within that tolerance wherever the
    # hopping projection ran -- so the check reads rounding.  A representation
    # built by hand on centres the operations miss by more reads no content
    # here, as it reads none anywhere else, and each orbit says why.
    from .symmetrization import _BLOCH_TOL, _position_defect

    def where_of(orbit, orbitals):
        here = sites[int(orbit[0])]
        return here, {"orbitals": orbitals,
                      "wyckoff": f"{int(here.multiplicity)}{here.letter}",
                      "site_symmetry": symmetry_of_site.get(int(orbit[0]), "")}

    ops = {int(s.ind): s for s in group.symmetries if int(s.ind) in represented}
    miss = float(_position_defect(model, group, assignment, ops))
    if miss > _BLOCH_TOL:
        note = ("no sum of elementary band representations, the symmetry content could not "
                "be identified: the orbital centres miss "
                f"the operations by {miss:.3g} (reduced), more than the {_BLOCH_TOL:.0e} the "
                "Bloch operators carry them to; move them onto the group's positions first, "
                "with realized_group's precision= or with symmetrize_positions")
        return tuple(BandRepresentation(**where_of(orbit, orbitals)[1], terms=(), note=note)
                     for orbit, orbitals, _rows in blocks)
    measured = _orbit_contents(group, rep, [rows for _orbit, _orbitals, rows in blocks])
    out = []
    for (orbit, orbitals, rows), (counts, points, refusals) in zip(blocks, measured):
        here, where = where_of(orbit, orbitals)
        hard = ({} if counts is None else
                {name: why for name, why in refusals.items() if not isinstance(why, str)})
        if counts is None or hard:
            name, why = (("", refusals) if counts is None else sorted(hard.items())[0])
            out.append(BandRepresentation(
                **where, terms=(),
                note="no sum of elementary band representations, the symmetry content "
                     "could not be identified"
                     + (f" at {name}" if name else "") + f": {getattr(why, 'reason', why)}"))
            continue
        # THE ORBIT'S OWN SUM: its content decomposed at the positions its
        # orbitals can be moved onto, one position at a time, its own first --
        # srsi_of_model's localization, which needs no invariants
        letters = [here.letter] if here.letter in here.closure else []
        letters += [letter for letter in here.closure if letter not in letters]
        named = frozenset(term for o in characters.orbitals if int(o.site) in set(orbit)
                          for term, _k in _label_terms(o.irrep))
        own = _sum_at_first(data, counts, points, ebrs, len(rows), named,
                            [letter for letter in letters if letter not in off_layer])
        # a site irrep whose band representation is a sum at other positions:
        # the sum with its own invariants, where the tables carry them, and
        # otherwise the one at the first position on the closure of a variety
        # through its site, the most special first -- a site irrep induced from
        # that variety's site group is moved along it
        composite = None
        if own is None and tables is not None:
            composite = _composite_localization(tables, group, assignment, rep, orbit, here)
        if own is None and composite is None:
            listed = frozenset(_ebr._letter_only(position) for _name, position in ebrs)
            through = []
            for variety in here.varieties:
                for letter in _groups._letters_on_closure(int(group.sg_number),
                                                          str(variety), listed):
                    if letter not in letters + through and letter not in off_layer:
                        through.append(letter)
            own = _sum_at_first(data, counts, points, ebrs, len(rows), named, through)
        # EVERY SUM WITH THIS CONTENT, over the layer's own band representations:
        # those its invariants do not tell from the orbit's own sum are its
        # alternatives, and a composite orbit's own sum is among them
        every = ()
        if tables is not None or own is None:
            restriction = _ebr.in_plane_restriction(group, points=points,
                                                    require_exact=False, _layer=layer)
            if not isinstance(restriction, Refusal):
                matrix, y, bound = _ebr._content_system(data, counts, restriction.retained)
                # the band count, one more equation: all a content read at no
                # point fixes, and each multiplicity's bound there
                matrix = np.vstack([matrix,
                                    [[_ebr._bands_spanned(name) for name, _at in ebrs]]])
                y = np.append(y, len(rows))
                bound = bound if restriction.retained else len(rows)
                _status, found, _cut = _ebr._cpsat(matrix[:, columns], y, 0, bound,
                                                   _ebr.DEFAULT_ENUMERATION_CAP)
                every = _ebr._ordered(_ebr._lifted(found, columns, len(ebrs)))
        if own is None and composite is not None:
            theirs = tables.theta({(label, here.letter): m
                                   for label, m in composite["irreps"].items()})
            own = next((s for s in every if tables.theta_columns(s) == theirs), None)
        elif own is None and len(every) == 1:
            own = every[0]
        alike = ()
        if own is not None and tables is not None:
            mine = tables.theta_columns(own)
            alike = tuple(s for s in every
                          if s != own and tables.theta_columns(s) == mine)
        if own is None:
            out.append(BandRepresentation(
                **where, terms=(),
                note=("no sum of elementary band representations carries the symmetry "
                      "content" if not every else
                      f"{len(every)} sums of elementary band representations carry the "
                      "symmetry content, and nothing read here says which is induced")))
            continue
        terms = tuple(
            tuple((int(m), name, position)
                  for (name, position), m in zip(ebrs, solution) if m)
            for solution in (own,) + alike)
        out.append(BandRepresentation(**where, terms=terms[0], alternatives=terms[1:]))
    return tuple(out)


def _sum_at_first(data, counts, points, ebrs, bands, named, letters):
    """A block's content as a sum at the first of ``letters`` that carries it, or ``None``.

    Returned as multiplicities over ``ebrs``; ``bands`` is the block's band
    count.  Where two sums at that position carry the content, the one whose
    site irreps are among ``named`` -- the orbit's own rows -- is taken.
    """
    for letter in letters:
        _status, found, at, _rows = _solve_at_letter(data, counts, points, letter, bands,
                                                     bands=bands)
        if found:
            pick = next((s for s in found
                         if {str(ebrs[at[j]][0]).split("↑")[0]
                             for j, m in enumerate(s) if m} <= named), found[0])
            return _ebr._lifted((pick,), at, len(ebrs))[0]
    return None


def _composite_localization(tables, group, assignment, rep, orbit, here):
    r"""The site irreps of an orbit whose induced band representation is composite.

    :math:`{}^1E{}^2E` at site symmetry :math:`\bar4` and ``E`` at :math:`\bar4m2` /
    :math:`\bar42m` are genuine site-symmetry irreps whose induced band
    representation is *not* elementary, so no EBR column carries them, the
    momentum-space decomposition has no solution — **and no EBR entry names
    them**, so the label the record carries at such a position is the site
    point group's own.  Where that name is not the tables' key, what is matched
    is the **content**: a row of the site's own character table is paired with
    the key of :attr:`SRSITables.maximal_keys` whose band count its induction
    reproduces, which is the one number a composite key and a computed row
    share.  A row whose name is a key of the tables at that letter is taken by
    its name.

    Returns ``{"irreps": {irrep: multiplicity}, "site": n}`` or ``None`` — the
    latter whenever anything is not measured cleanly, so :func:`srsi_of_model`
    refuses rather than guessing.  All 39 composite positions are in the
    single-valued table, so a spinful group takes this route not at all.
    """
    characters = getattr(rep, "characters", None)
    if characters is None or tables.spinor:
        return None
    site = int(orbit[0])
    table = (characters.site_tables or {}).get(site)
    known = {name for key in tables.site_keys if key.letter == here.letter
             for name in (key.irrep, key.ebr_label) if name}
    free = [key for key in tables.maximal_keys
            if key.letter == here.letter and not key.ebr]
    labels: dict = {}
    seen = set()
    for entry in characters.orbitals:
        if int(entry.site) != site or entry.multiplet in seen:
            continue
        seen.add(entry.multiplet)
        name = str(entry.irrep) if entry.irrep in known else ""
        if not name:
            if table is None:
                return None
            bands = _ebr._induced_bands(group, assignment, site, entry.character)
            hit = [key for key in free if int(key.dim) == int(bands)] if bands else []
            if len(hit) != 1:
                return None
            name = str(hit[0].irrep)
        # a multiplet of several copies of one irrep names it on every row, and
        # holds as many copies as its orbitals are multiples of the irrep's
        copies = max(1, len(entry.multiplet) // max(int(entry.dim), 1))
        labels[name] = labels.get(name, 0) + copies
    if not labels:
        return None
    return {"irreps": labels, "site": site}


# ---------------------------------------------------------------------------
# the no-go criterion in theta-coordinates
# ---------------------------------------------------------------------------
#: The verdicts a :class:`Splitting` holds.  A pair of atomic realizations that
#: adds up does not make a splitting trivial: it makes any topology of the
#: splitting invisible to the SRSIs, the criterion being sufficient, never
#: necessary.
SPLITTING_VERDICTS = ("diagnosed topological", "invisible to the SRSIs",
                      "no Z_n SRSIs", "stable topological")


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


@dataclass(frozen=True)
class StablePair:
    """Two site irreps at one maximal Wyckoff position that no stable real-space invariant distinguishes.

    Their elementary band representations have the same content at every
    high-symmetry point, and they carry the same invariants, so nothing the
    package measures tells them apart; the paper's example is ``B2`` and ``B3`` at
    ``12d`` of ``I432``.  What :func:`stably_equivalent_pairs` returns;
    :meth:`summary` prints it.

    Attributes
    ----------
    irreps, letter : tuple of str, str
        The two site irreps, and the Wyckoff letter.
    names, columns : tuple
        Their elementary band representations' names and columns in
        ``ebr_data(group)["ebrs"]``.
    symmetry_vector : tuple of int
        The content both induce.
    theta_Z, theta_Zn : tuple, dict
        The invariants both carry.
    number_str, spinor : str, bool
        The group, and which of its two tables.
    """

    number_str: str
    spinor: bool
    letter: str
    names: tuple
    irreps: tuple
    columns: tuple
    symmetry_vector: tuple = field(repr=False, default=())
    theta_Z: tuple = ()
    theta_Zn: dict = field(default_factory=dict)

    def summary(self) -> str:
        """The pair, then what both members share, a line each."""
        return "\n".join([
            f"{_group(self.number_str, self.spinor)}: "
            f"{_site_irreps(self.irreps, self.letter)}, which no stable real-space "
            "invariant distinguishes   (.irreps, .letter)",
            f"  their elementary band representations, columns "
            f"{' and '.join(str(c) for c in self.columns)} of the group's list, have the "
            "same content at every high-symmetry point   (.names, .columns, "
            ".symmetry_vector)",
            f"  the invariants both carry: Z-valued {_values(self.theta_Z)}"
            + (f", Z_n-valued {_by_modulus(self.theta_Zn)}" if self.theta_Zn else
               ", and the group has no Z_n-valued one")
            + "   (.theta_Z, .theta_Zn)"])

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return (f"StablePair(#{self.number_str}, {'/'.join(self.irreps)} @ "
                f"{self.letter})")


@dataclass(frozen=True)
class AtomicRealization:
    r"""One atomic realization: a set of orbitals at maximal Wyckoff positions with a band set's symmetry data.

    Attributes
    ----------
    p : tuple of int
        How many copies of each key of :attr:`Realizations.keys` it holds.
    terms : tuple
        ``((multiplicity, irrep, letter), ...)``: the keys it holds.
    theta_Zn : dict
        ``{n: values}``: its own :math:`\mathbb{Z}_n`-valued invariants, modulo
        ``n``, which two realizations of one symmetry vector need not share.
    n_bands : int
        The number of bands.
    pairs : tuple of StablePair
        Site irreps it holds that no invariant distinguishes from a partner.
    """

    p: tuple
    terms: tuple
    theta_Zn: dict
    n_bands: int
    pairs: tuple = ()

    @property
    def label(self) -> str:
        """``"1 (B2)_d + 2 (A)_a"``: the configuration, as the paper writes it."""
        return " + ".join(f"{m} ({irrep})_{letter}"
                          for m, irrep, letter in self.terms) or "(empty)"

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return f"AtomicRealization({self.label}, {self.n_bands} bands)"


@dataclass(frozen=True)
class Realizations:
    r"""Every atomic realization of a band set's symmetry data, or none.

    What :func:`atomic_realizations` returns; :meth:`summary` prints it.  The list
    is complete, so an empty one is a verdict: with integral
    :math:`\theta_{\mathbb{Z}}`, a fragile band set.  The record is truthy either
    way.

    Attributes
    ----------
    realizations : tuple of AtomicRealization
        Every one.
    atomic_possible : bool
        Whether there is any.
    theta_Z : tuple of Fraction
        The :math:`\mathbb{Z}`-valued invariants they reproduce.
    n_bands : int
        The band count, read off :attr:`theta_Z`.
    keys : tuple of MaximalKey
        The maximal site keys :attr:`AtomicRealization.p` counts.
    bound, cap : int
        The most copies of one key the band count allows, which makes the list
        complete, and the most realizations it may hold.
    note : str
        What the list says, in a sentence.
    number_str, spinor : str, bool
        The group, and which of its two tables.
    """

    number_str: str
    spinor: bool
    theta_Z: tuple
    n_bands: int
    keys: tuple = field(repr=False, default=())
    realizations: tuple = ()
    bound: int = 0
    cap: int = 0
    note: str = ""
    _status: str = field(default="", repr=False, compare=False)

    @property
    def atomic_possible(self) -> bool:
        """Whether any atomic insulator has this symmetry data."""
        return bool(self.realizations)

    def summary(self) -> str:
        """The band set, then every realization of it, a line each."""
        lines = [f"{_group(self.number_str, self.spinor)}: "
                 + (f"every atomic realization, {len(self.realizations)} of them"
                    if self.realizations else "no atomic realization")
                 + "   (.realizations)",
                 "  of " + (f"{_count(self.n_bands, 'band')} with " if self.n_bands else "")
                 + f"theta_Z = {_values(self.theta_Z)}, their Z-valued stable real-space "
                 "invariants   (.n_bands, .theta_Z)"]
        if not self.realizations:
            return "\n".join(lines + [f"  {self.note}   (.note)"])
        with_zn = any(c.theta_Zn for c in self.realizations)
        lines.append("  each a set of orbitals at maximal Wyckoff positions with that "
                     f"symmetry data, {_CONFIGURATION}"
                     + (", with its own Z_n-valued invariants   (.label, .theta_Zn of each)"
                        if with_zn else "   (.label of each)"))
        lines += [f"    {c.label}" + (f"   {_by_modulus(c.theta_Zn)}" if c.theta_Zn else "")
                  for c in self.realizations]
        pairs = []
        for c in self.realizations:
            pairs += [pair for pair in c.pairs if pair not in pairs]
        if pairs:
            lines.append("  site irreps they use that no stable real-space invariant "
                         "distinguishes   (.pairs of each)")
            lines += [f"    {_site_irreps(pair.irreps, pair.letter)}" for pair in pairs]
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return (f"Realizations(#{self.number_str}, "
                f"{_count(len(self.realizations), 'realization')})")


@dataclass(frozen=True)
class Splitting:
    r"""Whether the two sides of a gapped splitting of bands can both be atomic insulators, by the paper's no-go criterion.

    What :func:`splitting` and :func:`nsit_criterion` return; :meth:`summary`
    prints it.  ``V`` is the bands asked for, ``C`` the rest, ``H`` all of them.
    Every verdict is a verdict about the splitting, so the record is truthy.

    Attributes
    ----------
    verdict : str
        One of :data:`SPLITTING_VERDICTS`:

        ``"diagnosed topological"``
            no pair of atomic realizations of ``V`` and ``C`` adds up to the
            :math:`\mathbb{Z}_n`-valued invariants of ``H``, or one side has no
            realization at all: ``V`` and ``C`` cannot both be atomic
            insulators.  The paper calls it non-symmetry-indicated topological
            (NSIT).
        ``"invisible to the SRSIs"``
            some pair adds up: any topology of the splitting is invisible to the
            stable real-space invariants, the criterion being sufficient and not
            necessary.
        ``"no Z_n SRSIs"``
            the group has no :math:`\mathbb{Z}_n`-valued invariant, so there is
            nothing to add up.
        ``"stable topological"``
            the :math:`\mathbb{Z}`-valued invariants of ``V`` are fractional: it
            carries a nonzero symmetry indicator, and no realization is listed.
    nsit : bool
        Whether the verdict is ``"diagnosed topological"``.
    reason : str
        Why, in a sentence.
    theta_Z_total, theta_Zn_total : tuple, dict
        The invariants of ``H``, read off the model's orbitals.
    theta_Z_v, theta_Z_c : tuple of Fraction
        The :math:`\mathbb{Z}`-valued invariants of ``V`` and of ``C``, which add
        up to those of ``H``.
    realizations_v, realizations_c : Realizations or None
        Every atomic realization of each side; ``None`` for
        ``"stable topological"``.
    consistent_pairs : tuple
        ``(index_v, index_c)`` of each pair of realizations whose
        :math:`\mathbb{Z}_n`-valued invariants add up to those of ``H``, modulo
        each ``n``.
    v_atomic_possible, c_atomic_possible : bool
        Whether each side has any realization: a side with none is fragile, which
        tells a fragile side from two atomic sides that do not add up.
    moduli : tuple of int
        The moduli of the group's :math:`\mathbb{Z}_n`-valued invariants.
    indicator : ..ebr.SymmetryIndicator or None
        The symmetry-indicator class of ``V``, where :func:`splitting` measured
        it.
    n_bands_v, n_bands_c : int or None
        The number of bands on each side.
    same_content : tuple of StablePair
        Site irreps a realization holds that no invariant distinguishes.
    number_str, spinor : str, bool
        The group, and which of its two tables.
    """

    number_str: str
    spinor: bool
    verdict: str
    nsit: bool
    reason: str
    theta_Z_total: tuple
    theta_Zn_total: dict
    theta_Z_v: tuple
    theta_Z_c: tuple
    realizations_v: Any = field(repr=False, default=None)
    realizations_c: Any = field(repr=False, default=None)
    consistent_pairs: tuple = ()
    v_atomic_possible: bool = False
    c_atomic_possible: bool = False
    moduli: tuple = ()
    indicator: Any = field(repr=False, default=None)
    n_bands_v: int | None = None
    n_bands_c: int | None = None
    same_content: tuple = ()

    #: How each verdict is spelled at the head of :meth:`summary`: the verdict,
    #: then what it says.
    HEADLINES = {
        "diagnosed topological":
            "diagnosed topological: the two sides of the gap cannot both be atomic "
            "insulators",
        "invisible to the SRSIs":
            "invisible to the SRSIs: the stable real-space invariants (SRSIs) allow both "
            "sides of the gap to be atomic insulators, which does not make them so",
        "no Z_n SRSIs":
            "no Z_n SRSIs: the group has no Z_n-valued stable real-space invariant (SRSI) "
            "to test whether both sides of the gap are atomic insulators",
        "stable topological":
            "stable topological: the bands asked for carry a nonzero symmetry indicator, "
            "so no atomic insulator has their symmetry data",
    }

    def summary(self) -> str:
        """The verdict, then the two sides, their invariants and their realizations, a line each."""
        lines = [f"{_group(self.number_str, self.spinor)}, "
                 f"{self.HEADLINES.get(self.verdict, self.verdict)}   (.verdict)"]
        n_v, n_c = self.n_bands_v, self.n_bands_c
        counted = n_v is not None and n_c is not None
        if self.indicator is not None:
            # what `splitting` returns: the bands asked for, and the rest of the model's
            lines.append("  V, the " + (f"{n_v} " if counted else "")
                         + "bands asked for; C, the model's other "
                         + (f"{n_c}; H, all {n_v + n_c}" if counted else "bands; H, all")
                         + "   (.n_bands_v, .n_bands_c)")
        else:
            lines.append("  V and C, the two sides handed in"
                         + (f", of {n_v} and {n_c} bands" if counted else "")
                         + "; H, the two together   (.n_bands_v, .n_bands_c)")
        lines.append(f"  theta_Z, the Z-valued stable real-space invariants: "
                     f"H {_values(self.theta_Z_total)} = V {_values(self.theta_Z_v)} + "
                     f"C {_values(self.theta_Z_c)}   "
                     "(.theta_Z_total, .theta_Z_v, .theta_Z_c)")
        if self.theta_Zn_total:
            lines.append(f"  theta_Zn, the Z_n-valued ones of H: "
                         f"{_by_modulus(self.theta_Zn_total)}   (.theta_Zn_total)")
        if self.indicator is not None:
            lines.append(_indicator_line(self.indicator, "V's"))
        explained = False
        for tag, got, attr in (("V", self.realizations_v, "realizations_v"),
                               ("C", self.realizations_c, "realizations_c")):
            if got is None:
                continue
            if not got.realizations:
                lines.append(f"  {tag} has no atomic realization: {got.note}   "
                             f"(.{attr}, .{tag.lower()}_atomic_possible)")
                continue
            lines.append(f"  {tag}'s atomic realizations"
                         + ("" if explained else
                            f", every set of orbitals at maximal Wyckoff positions with "
                            f"{tag}'s symmetry data, {_CONFIGURATION}"
                            + (", with its own theta_Zn" if self.moduli else ""))
                         + f"   (.{attr})")
            explained = True
            lines += [f"    [{index}] {realization.label}"
                      + (f"   {_by_modulus(realization.theta_Zn)}"
                         if realization.theta_Zn else "")
                      for index, realization in enumerate(got.realizations)]
        if (self.moduli and self.realizations_v is not None
                and self.realizations_c is not None
                and self.realizations_v.realizations and self.realizations_c.realizations):
            if self.consistent_pairs:
                lines.append("  the pairs of a realization of V and one of C whose theta_Zn "
                             "add up to H's, modulo each n   (.consistent_pairs)")
                lines += [f"    V[{iv}] with C[{ic}]" for iv, ic in self.consistent_pairs]
            else:
                lines.append("  no realization of V with one of C has theta_Zn adding up "
                             "to H's, modulo each n   (.consistent_pairs)")
        if self.same_content:
            lines.append("  site irreps the realizations use that no stable real-space "
                         "invariant distinguishes   (.same_content)")
            lines += [f"    {_site_irreps(pair.irreps, pair.letter)}"
                      for pair in self.same_content]
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return (f"Splitting(#{self.number_str}, {self.verdict}, "
                f"{len(self.consistent_pairs)} consistent pair(s))")


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
    from .refusal import InternalError

    tables = _table_of_number(int(sg), bool(spinor))
    if not tables:                            # pragma: no cover - gated above
        raise ValueError(f"space group {sg} has no shipped SRSI table: {tables}")
    data = _groups.ebr_data(_TableKey(tables.number_str, spinor))
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


@functools.lru_cache(maxsize=None)
def _stable_pairs(sg: int, spinor: bool) -> tuple:
    """The same-position pairs of one system -- :func:`stably_equivalent_pairs`."""
    tables = _table_of_number(int(sg), bool(spinor))
    if not tables:                            # pragma: no cover - gated above
        return ()
    data = _groups.ebr_data(_TableKey(tables.number_str, spinor))
    if not data:                              # pragma: no cover - install-dependent
        return ()
    matrix = np.asarray(data["matrix"], dtype=int)
    by_content: dict = {}
    for j, (_name, position) in enumerate(data["ebrs"]):
        letter = _POSITION.match(str(position)).group(2)
        by_content.setdefault((letter, tuple(int(x) for x in matrix[:, j])),
                              []).append(j)
    out = []
    for (letter, vector), columns in sorted(by_content.items()):
        if len(columns) < 2:
            continue
        theta_z, theta_zn = tables.theta_columns(
            [1 if j == columns[0] else 0 for j in range(len(tables.ebrs))])
        out.append(StablePair(
            number_str=str(tables.number_str), spinor=bool(tables.spinor),
            letter=str(letter),
            names=tuple(sorted(str(data["ebrs"][j][0]) for j in columns)),
            irreps=tuple(sorted(tables.ebrs[j][0] for j in columns)),
            columns=tuple(columns), symmetry_vector=vector,
            theta_Z=theta_z, theta_Zn=theta_zn))
    return tuple(out)


def stably_equivalent_pairs(group_or_tables) -> "tuple | Refusal":
    """Site irreps at one maximal Wyckoff position that no stable real-space invariant distinguishes.

    Their elementary band representations have the same content at every
    high-symmetry point and they carry the same invariants, so nothing the package
    measures tells them apart, and which one a decomposition names moves no
    number it returns.  The paper's example is ``B2`` and ``B3`` at ``12d`` of
    ``I432``.

    Parameters
    ----------
    group_or_tables : ..groups.GroupData or SRSITables
        A measured grey group, or its tables.

    Returns
    -------
    tuple of StablePair or Refusal
        Each pair's ``summary()`` prints it.  Empty for most groups, and then
        falsy, as a ``Refusal`` in its place is: ``isinstance(out, Refusal)``
        tells the two apart.

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
    _no_bare_number(group_or_tables, "stably_equivalent_pairs")
    gate = _groups._lattice_not_descended(group_or_tables)
    if gate is not None:
        return gate
    tables = (group_or_tables if isinstance(group_or_tables, SRSITables)
              else srsi_tables(group_or_tables))
    if not tables:
        return tables
    return _stable_pairs(int(tables.sg), bool(tables.spinor))


def _theta_zn_of_keys(tables, p) -> dict:
    """``{n: (values mod n,)}`` for an integer ``p`` over the maximal keys."""
    columns = _maximal_basis(int(tables.sg), bool(tables.spinor))[1]
    out = {}
    for n, rows in tables.zn_rows.items():
        out[n] = tuple(
            sum(int(columns[j][r]) * int(p[j]) for j in range(len(p)) if p[j]) % n
            for r in rows)
    return out


def atomic_realizations(tables, theta_Z, *, n_bands=None,
                        cap=256) -> "Realizations | Refusal":
    r"""Every atomic insulator with a band set's symmetry data, each a set of orbitals at maximal Wyckoff positions.

    The configurations with nonnegative integer multiplicities over
    :attr:`SRSITables.maximal_keys` whose :math:`\mathbb{Z}`-valued invariants are
    ``theta_Z``: the same invariants are the same symmetry data.  Maximal
    positions are enough, since an atomic insulator moves onto them with its
    invariants unchanged, and the list is complete.

    Parameters
    ----------
    tables : SRSITables
        The group's tables, from :func:`srsi_tables`.
    theta_Z : sequence
        The band set's :math:`\mathbb{Z}`-valued invariants, one per
        :attr:`SRSITables.z_rows`.
    n_bands : int, optional
        The band count, which ``theta_Z`` fixes; another raises
        :class:`ValueError`.
    cap : int, optional
        The most realizations to list; more is refused.  Default 256.

    Returns
    -------
    Realizations or Refusal
        ``out.summary()`` prints every realization with its own
        :math:`\mathbb{Z}_n`-valued invariants; ``out.realizations`` holds them.
        The record is truthy, and an empty list with integral ``theta_Z`` is a
        fragile band set.  A ``Refusal`` in its place is falsy.

    Raises
    ------
    ValueError
        ``theta_Z`` of the wrong length, or an ``n_bands`` it does not imply.

    Refusals
    --------
    ``srsi_not_atomic``
        ``theta_Z`` is fractional: the band set carries a nonzero symmetry
        indicator, and no atomic insulator has its symmetry data.
    ``srsi_realizations_truncated``
        more than ``cap`` realizations; a larger ``cap=`` lists them.

    Examples
    --------
    >>> print(atomic_realizations(tables, (0, 0, 1)).summary())   # doctest: +SKIP
    """
    keys = tables.maximal_keys
    values = tuple(Fraction(v) for v in theta_Z)
    if len(values) != len(tables.z_rows):
        raise ValueError(
            f"#{tables.number_str} has {len(tables.z_rows)} Z-valued SRSIs and "
            f"theta_Z carries {len(values)} entries; it must be indexed by "
            "SRSITables.z_rows, the order the first half of SRSITables.theta "
            "comes back in")
    if any(value.denominator != 1 for value in values):
        return Refusal(
            "srsi_not_atomic",
            f"#{tables.number_str}: theta_Z = "
            f"({', '.join(str(v) for v in values)}) has a fractional entry, so by the "
            "paper's theorem this band set carries a nonzero symmetry indicator and no "
            "atomic insulator has its symmetry data at all. There is nothing to "
            "enumerate, a stronger statement than an empty list of realizations; "
            "splitting() returns it as the verdict 'stable topological'",
            value=float(max(value.denominator for value in values)), threshold=1.0,
            context={"number_str": str(tables.number_str),
                     "theta_Z": tuple(str(v) for v in values),
                     "denominators": tuple(int(v.denominator) for v in values)})

    form = tables.band_count_form
    implied = sum((form[i] * values[i] for i in range(len(values))), Fraction(0))
    dims = [int(key.dim) for key in keys]
    if implied.denominator != 1 or implied < 0:
        return Realizations(
            number_str=str(tables.number_str), spinor=bool(tables.spinor),
            theta_Z=values, n_bands=0, keys=keys, realizations=(),
            bound=0, cap=int(cap),
            note=(f"the band count implied by theta_Z is {implied}, which is not a "
                  "nonnegative integer, so no configuration of orbitals anywhere has "
                  "this symmetry data"),
            _status="INFEASIBLE")
    if n_bands is not None and int(n_bands) != int(implied):
        raise ValueError(
            f"#{tables.number_str}: theta_Z implies {int(implied)} bands (the band "
            f"count is a Z-linear form in the Z-SRSIs) and n_bands={int(n_bands)} was "
            "asserted. No band set has both, so one of the two is an arithmetic "
            "mistake rather than a measurement")
    total = int(implied)

    columns = _maximal_basis(int(tables.sg), bool(tables.spinor))[1]
    rows = [[int(columns[j][r]) for j in range(len(keys))] for r in tables.z_rows]
    matrix = np.array(rows + [dims], dtype=int)
    y = np.array([int(v) for v in values] + [total], dtype=int)
    bound = total // min(dims) if dims else 0
    status, solutions, truncated = _ebr._cpsat(matrix, y, 0, int(bound), int(cap))
    if truncated:
        return Refusal(
            "srsi_realizations_truncated",
            f"#{tables.number_str}: the enumeration of atomic realizations of theta_Z "
            f"= ({', '.join(str(v) for v in values)}) at {total} bands hit the cap of "
            f"{cap} solutions. A no-go is only ever asserted on a complete "
            "enumeration -- with the list truncated, neither 'no pair adds up' nor "
            "'none exists' may be claimed. Only atomic_realizations takes cap=, 256 by "
            "default: called directly, raise it. nsit_criterion and splitting take "
            "none and enumerate each side at that default; atomic_realizations with a "
            "larger cap=, handed these tables and the theta_Z above "
            "(context['theta_Z']), enumerates that side's realizations.",
            value=float(len(solutions)), threshold=float(cap),
            context={"number_str": str(tables.number_str), "cap": int(cap),
                     "n_bands": total, "bound": int(bound),
                     "theta_Z": tuple(str(v) for v in values)})

    pairs = _stable_pairs(int(tables.sg), bool(tables.spinor))
    records = []
    for solution in solutions:
        touched = tuple(
            pair for pair in pairs
            if any(solution[j] for j, key in enumerate(keys)
                   if key.column is not None and key.column in pair.columns))
        records.append(AtomicRealization(
            p=tuple(int(m) for m in solution),
            terms=tuple((int(m), keys[j].irrep, keys[j].letter)
                        for j, m in enumerate(solution) if m),
            theta_Zn=_theta_zn_of_keys(tables, solution),
            n_bands=total, pairs=touched))
    note = ("every set of orbitals at maximal Wyckoff positions with this symmetry data, "
            "enumerated completely"
            if records else
            "theta_Z is integral, so no symmetry indicator sees these bands, and no set "
            "of orbitals at maximal Wyckoff positions has their symmetry data: a fragile "
            "band set")
    return Realizations(
        number_str=str(tables.number_str), spinor=bool(tables.spinor),
        theta_Z=values, n_bands=total, keys=keys, realizations=tuple(records),
        bound=int(bound), cap=int(cap), note=note, _status=str(status))


def nsit_criterion(tables, theta_total, theta_Z_v, theta_Z_c, *,
                   n_bands_v=None, n_bands_c=None) -> "Splitting | Refusal":
    r"""The paper's no-go criterion on a gapped splitting whose invariants are in hand.

    All the bands together, ``H``, are an atomic insulator.  If the two sides,
    ``V`` and ``C``, are atomic insulators too, then the
    :math:`\mathbb{Z}_n`-valued invariants of some atomic realization of each add
    up to those of ``H``, modulo each ``n``; where no pair does, or a side has no
    realization at all, ``V`` and ``C`` cannot both be atomic insulators.  Every
    realization of each side is listed, since one pair that does not add up
    proves nothing.  The criterion is sufficient, never necessary.

    Parameters
    ----------
    tables : SRSITables
        The group's tables.
    theta_total : tuple
        ``(theta_Z, theta_Zn)`` of ``H``, all the bands, as
        :meth:`SRSITables.theta` returns them for the model's orbitals.
    theta_Z_v, theta_Z_c : sequence
        The :math:`\mathbb{Z}`-valued invariants of ``V`` and of ``C``, the two
        sides of the gap; a pair that does not add up to those of ``H`` raises
        :class:`ValueError`.
    n_bands_v, n_bands_c : int, optional
        The bands on each side, for :func:`atomic_realizations`.

    Returns
    -------
    Splitting or Refusal
        ``out.summary()`` prints the verdict, the realizations of each side and
        the pairs that add up; ``out.verdict`` holds the verdict.  The record is
        truthy.  A ``Refusal`` in its place is falsy.

    Raises
    ------
    ValueError
        Invariants of the wrong length, two sides that do not add up to ``H``, or
        a ``theta_Zn`` with other moduli than the group's.

    Refusals
    --------
    ``srsi_not_atomic``, ``srsi_realizations_truncated``
        from :func:`atomic_realizations`, which lists each side at its default
        cap: a side with fractional invariants, or more realizations than the
        cap, which :func:`atomic_realizations` with a larger ``cap=`` lists.
    """
    theta_z_h = tuple(Fraction(v) for v in theta_total[0])
    theta_zn_h = {int(n): tuple(values) for n, values in dict(theta_total[1]).items()}
    values_v = tuple(Fraction(v) for v in theta_Z_v)
    values_c = tuple(Fraction(v) for v in theta_Z_c)
    if not len(theta_z_h) == len(values_v) == len(values_c) == len(tables.z_rows):
        raise ValueError(
            f"#{tables.number_str} has {len(tables.z_rows)} Z-valued SRSIs; got "
            f"{len(theta_z_h)} for the total and {len(values_v)} / {len(values_c)} for "
            "the two sides. All three are indexed by SRSITables.z_rows")
    if tuple(a + b for a, b in zip(values_v, values_c)) != theta_z_h:
        raise ValueError(
            f"#{tables.number_str}: theta_Z[V] + theta_Z[C] = "
            f"({', '.join(str(a + b) for a, b in zip(values_v, values_c))}) does not "
            f"equal theta_Z[H] = ({', '.join(str(v) for v in theta_z_h)}). The "
            "Z-SRSIs are linear in p and the symmetry vectors add, so this holds for "
            "any genuine splitting -- a violation is an arithmetic mistake in what "
            "was handed in, not a measurement about the bands")
    moduli = tuple(sorted(tables.zn_rows))
    if sorted(theta_zn_h) != list(moduli):
        raise ValueError(
            f"#{tables.number_str} has Z_n-valued SRSIs at moduli {list(moduli)} and "
            f"theta_Zn carries {sorted(theta_zn_h)}; hand in the pair "
            "SRSITables.theta returns")

    got_v = atomic_realizations(tables, values_v, n_bands=n_bands_v)
    if not got_v:
        return got_v
    got_c = atomic_realizations(tables, values_c, n_bands=n_bands_c)
    if not got_c:
        return got_c

    consistent = []
    for iv, realization_v in enumerate(got_v.realizations):
        for ic, realization_c in enumerate(got_c.realizations):
            if all((realization_v.theta_Zn[n][i] + realization_c.theta_Zn[n][i]
                    - int(theta_zn_h[n][i])) % n == 0
                   for n in moduli for i in range(len(theta_zn_h[n]))):
                consistent.append((iv, ic))

    if not got_v.atomic_possible or not got_c.atomic_possible:
        side = " and ".join([tag for tag, got in (("V", got_v), ("C", got_c))
                             if not got.atomic_possible])
        verdict, nsit = "diagnosed topological", True
        reason = (
            f"side {side} has integral theta_Z and no atomic realization of it over "
            "the maximal positions, so that side is not an atomic insulator at all. "
            "The splitting is diagnosed topological by the realization half of the "
            "criterion -- a fragile side -- and not by the Z_n additivity")
    elif not moduli:
        verdict, nsit = "no Z_n SRSIs", False
        reason = (
            f"#{tables.number_str} has no Z_n-valued SRSI, so there is no additivity "
            "to test and the criterion cannot see anything about this splitting. Both "
            "sides do have atomic realizations, which is all that was measured")
    elif consistent:
        verdict, nsit = "invisible to the SRSIs", False
        reason = (
            f"{len(consistent)} of the {len(got_v.realizations)} x "
            f"{len(got_c.realizations)} realization pairs reproduce theta_Zn[H] modulo "
            "each row's own n, so the SRSIs are consistent with V and C both being "
            "atomic insulators. Any topology of this splitting is invisible to the "
            "stable real-space invariants")
    else:
        verdict, nsit = "diagnosed topological", True
        reason = (
            f"none of the {len(got_v.realizations)} x {len(got_c.realizations)} "
            "realization pairs reproduces theta_Zn[H], so V and C cannot both be "
            "atomic insulators: the splitting is non-symmetry-indicated topological "
            "(NSIT)")

    same = []
    for got in (got_v, got_c):
        for realization in got.realizations:
            for pair in realization.pairs:
                if pair not in same:
                    same.append(pair)
    return Splitting(
        number_str=str(tables.number_str), spinor=bool(tables.spinor),
        verdict=verdict, nsit=nsit, reason=reason,
        theta_Z_total=theta_z_h, theta_Zn_total=theta_zn_h,
        theta_Z_v=values_v, theta_Z_c=values_c,
        realizations_v=got_v, realizations_c=got_c,
        consistent_pairs=tuple(consistent),
        v_atomic_possible=bool(got_v.atomic_possible),
        c_atomic_possible=bool(got_c.atomic_possible),
        moduli=moduli, indicator=None,
        n_bands_v=int(got_v.n_bands), n_bands_c=int(got_c.n_bands),
        same_content=tuple(same))


def splitting(realized, *, bands=None, rep=None,
              points=None) -> "Splitting | Refusal":
    r"""Whether the bands asked for and the rest of a model's bands can both be atomic insulators.

    The paper's no-go criterion on a gapped splitting of a model's bands: ``V``,
    the bands asked for; ``C``, the rest of the model's bands; ``H``, all of them.
    The invariants of ``H`` are read off the orbitals (:func:`srsi_of_model`), the
    :math:`\mathbb{Z}`-valued ones of ``V`` off its bands
    (:func:`srsi_of_bands`), and those of ``C`` are the difference;
    :func:`nsit_criterion` decides.  The gap is taken as given.

    Parameters
    ----------
    realized : ..groups.RealizedGroup
        What :func:`..groups.realized_group` returned, or the parts
        :func:`srsi_of_model` takes.
    bands : int or sequence of int
        The bands asked for, ``V``: an ``int`` is the lowest *n*, a sequence
        those bands.  Required.
    rep : ..symmetry.SpaceGroupRep, optional
        A representation to read in place of the one ``realized`` holds.
    points : sequence of str, optional
        The high-symmetry points to read the content of ``V`` at.

    Returns
    -------
    Splitting or Refusal
        The verdict on ``V``, the bands asked for, and ``C``, the rest of the
        model's bands, ``H`` being all of them.  ``out.summary()`` prints it and
        what it rests on: the invariants of ``V``, ``C`` and ``H``, and the atomic
        realizations of each side.  ``out.verdict`` is ``"diagnosed
        topological"``, ``"invisible to the SRSIs"``, ``"no Z_n SRSIs"`` or
        ``"stable topological"``, the last where ``V`` carries a nonzero symmetry
        indicator, ``out.indicator``.  The record is truthy.  A ``Refusal`` in its
        place is falsy.

    Raises
    ------
    TypeError
        ``bands`` left out.

    Refusals
    --------
    ``srsi_scope``, ``srsi_table_absent``, ``srsi_data``, ``srsi_ebr_basis``
        from :func:`srsi_tables`: the group has no tables.
    ``orbit_content_incomplete``, ``orbit_not_localizable``, ``wyckoff_setting``
        from :func:`srsi_of_model`, for the orbitals.
    ``incomplete_content``, ``not_a_band_structure``, ...
        from :func:`..sewing.hsp_table` and :func:`srsi_of_bands`, for the bands.
    ``srsi_not_atomic``, ``srsi_realizations_truncated``
        from :func:`nsit_criterion`, which lists each side at the default cap of
        :func:`atomic_realizations`.

    Examples
    --------
    >>> print(splitting(realized, bands=6).summary())       # doctest: +SKIP
    """
    from . import sewing as _sewing

    if bands is None:
        raise TypeError("splitting() missing required argument 'bands'")
    model, group, _assignment, own_rep = _unpack_realized(realized)
    gate = _groups._lattice_not_descended(group)
    if gate is not None:
        return gate
    rep = own_rep if rep is None else rep
    tables = srsi_tables(group)
    if not tables:
        return tables
    whole = srsi_of_model(realized, rep=rep)
    if not whole:
        return whole
    n_valence = (int(bands) if isinstance(bands, (int, np.integer))
                 else len(list(bands)))
    n_total = int(model.norb) * int(getattr(model, "nspin", 1))
    content = _sewing.hsp_table(rep, bands=bands)
    if not content:
        return content
    valence_side = srsi_of_bands(content, group, rep=rep, points=points,
                                 bands=bands)
    if not valence_side:
        return valence_side

    theta_z_v = tuple(valence_side.theta_Z)
    theta_z_h = tuple(whole.theta_Z)
    theta_z_c = tuple(a - b for a, b in zip(theta_z_h, theta_z_v))
    if any(value.denominator != 1 for value in theta_z_v):
        return Splitting(
            number_str=str(tables.number_str), spinor=bool(tables.spinor),
            verdict="stable topological", nsit=False,
            reason=(
                f"theta_Z[V] = ({', '.join(str(v) for v in theta_z_v)}) is fractional, "
                "so by the paper's theorem the valence set carries a nonzero symmetry "
                f"indicator ({valence_side.indicator.values} of "
                f"{valence_side.indicator.structure}, order "
                f"{valence_side.indicator.order}) and no atomic insulator has its "
                "symmetry data. The no-go criterion is not needed: this splitting is "
                "already stable topological"),
            theta_Z_total=theta_z_h, theta_Zn_total=dict(whole.theta_Zn),
            theta_Z_v=theta_z_v, theta_Z_c=theta_z_c,
            realizations_v=None, realizations_c=None, consistent_pairs=(),
            v_atomic_possible=False, c_atomic_possible=False,
            moduli=tuple(sorted(tables.zn_rows)), indicator=valence_side.indicator,
            n_bands_v=n_valence, n_bands_c=n_total - n_valence)

    got = nsit_criterion(tables, (theta_z_h, dict(whole.theta_Zn)),
                         theta_z_v, theta_z_c,
                         n_bands_v=n_valence, n_bands_c=n_total - n_valence)
    if not got:
        return got
    fields = {name: getattr(got, name) for name in got.__dataclass_fields__}
    fields["indicator"] = valence_side.indicator
    return Splitting(**fields)
