r"""Stable real-space invariants (SRSIs): the tables, their values on bands and models, and a no-go criterion for a gapped splitting.

An SRSI is a linear form :math:`\theta = \Theta\cdot p` in the site-symmetry
vector ``p``, how many times each site irrep occurs at each Wyckoff position,
that no symmetric deformation of the Wannier functions changes: two atomic
configurations share every SRSI exactly when they deform into each other once
the same atomic orbitals are added to both.  The integer-valued ones,
``theta_Z``, are a function of the symmetry data vector, fractional exactly
when the symmetry indicator is nonzero.  Those valued modulo n, ``theta_Zn``,
are not, and tell apart atomic limits with the same symmetry data vector.
Every set of invariants is held by name, ``theta_1``, ``theta_2``, ...

The tables are those of Hwang et al., *Stable real-space invariants and
topology beyond symmetry indicators*, Nat. Commun. **17**, 8564 (2026),
https://doi.org/10.1038/s41467-026-74844-w, for the grey space groups with
spin-orbit coupling and without, shipped with the package; the thirteen groups
whose one invariant is the band count have none.

- :func:`srsi_tables`, the tables of one group, whose ``theta`` evaluates them
  on a configuration in hand and whose ``stably_equivalent`` holds the pairs of
  site irreps no invariant tells apart;
- :func:`srsi_of_bands`, the integer-valued invariants of a band set, from its
  symmetry data vector, and :func:`srsi_of_model`, both kinds, from a model's
  orbitals, a row for each Wyckoff orbit;
- :func:`splitting`, the paper's no-go criterion on a gapped splitting of a
  model's bands, with :func:`nsit_criterion`, the criterion on invariants in
  hand, and :func:`atomic_realizations`, every atomic insulator with given
  invariants;
- :func:`srsi_status`, whether the tables are installed, and :data:`GATES`,
  every refusal the calls here return;
- :func:`canonical_determinism`, whether the paper's named indicators tell
  every class of the indicator group apart.

``docs/guide/srsi.md`` reads them on models.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from fractions import Fraction
from typing import Any

import numpy as np

from . import _naming
from . import _smith
from . import _spacegroup
from . import _tables
from . import ebr as _ebr
from . import sewing as _sewing
from .refusal import InternalError, Refusal
from ._text import _label_terms, _orbitals_named
from ._matrices import _BLOCH_TOL
from ._centers import _position_defect
from .symmetry import (BandRepresentation, WyckoffOrbit, WyckoffOrbitNumerical,
                       _OrbitBandRepresentation, _paper_label, _terms_of,
                       _wyckoff_orbits_of)
from ._srsitables import (
    GATES, IndicatorCheck, Invariants, ModularValue, Provenance, ProvenanceCounts,
    SRSITables, SRSITablesNumerical, SiteIrrep, SplitEBR, StablePair, _POSITION,
    _and, _count, _group, _indicator_line, _invariants_text, _label, _maximal_basis,
    _pair_text, _rational_p, _signed, _span, _stable_pairs, _values, _zn_groups,
    _zn_header, _zn_text, _zn_values, canonical_determinism, srsi_status, srsi_tables)

__all__ = ["SRSITables", "SRSITablesNumerical", "IndicatorCheck", "Provenance",
           "ProvenanceCounts", "SiteIrrep", "SplitEBR", "StablePair", "Invariants",
           "ModularValue",
           "BandSRSIs", "BandSRSIsNumerical", "ModelSRSIs", "WyckoffOrbitSRSIs",
           "WyckoffOrbitSRSIsNumerical", "Localization",
           "AtomicRealization", "Realizations", "RealizationsNumerical", "Splitting",
           "GATES",
           "srsi_status", "srsi_tables", "srsi_of_bands", "srsi_of_model",
           "atomic_realizations", "nsit_criterion", "splitting",
           "canonical_determinism"]


def _fractional(theta_z) -> bool:
    """Whether any of a mapping of invariants is not a whole number."""
    return any(Fraction(value).denominator != 1 for value in theta_z.values())


def _fields(record, **override) -> dict:
    """The fields a record is built from, by name, ``override`` replacing some."""
    out = {f.name: getattr(record, f.name) for f in fields(record) if f.init}
    out.update(override)
    return out


# ---------------------------------------------------------------------------
# a band subset -> its integer-valued invariants
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BandSRSIsNumerical:
    """The rational site-symmetry vector the invariants of a band set were evaluated on.

    ``b.numerical`` of what :func:`srsi_of_bands` returns; :meth:`summary`
    prints it.

    Attributes
    ----------
    p : dict
        ``{(site irrep, Wyckoff position): multiplicity}``: one rational
        combination of elementary band representations with the band set's
        symmetry data vector, of many, its nonzero entries in the order of the
        list of elementary band representations, keyed as
        :meth:`SRSITables.theta` takes them.
    """

    p: dict

    def summary(self) -> str:
        """The combination, in the paper's form."""
        return ("one rational combination of elementary band representations with this "
                "symmetry data vector, of many: "
                + _signed((value, f"({irrep})_{wyckoff.lstrip('0123456789')}")
                          for (irrep, wyckoff), value in self.p.items())
                + "   (.p)")


@dataclass(frozen=True)
class BandSRSIs:
    r"""The integer-valued stable real-space invariants of a band set, read off its symmetry data vector.

    What :func:`srsi_of_bands` returns; :meth:`summary` prints it, and :meth:`line`
    puts it on one line.  It holds no invariants valued modulo n: those are no
    function of the symmetry data vector.

    Attributes
    ----------
    theta_Z : dict
        ``{name: value}``: the invariants, exact fractions, in the tables'
        order; fractional exactly where the symmetry indicator is nonzero, the
        band set stable topological.
    indicator : ..ebr.SymmetryIndicator
        The symmetry-indicator class of the same bands.
    group : ..groups.GroupData
        The group the bands were read in.
    numerical : BandSRSIsNumerical
        The rational site-symmetry vector the invariants were evaluated on.
    """

    theta_Z: dict
    indicator: Any = field(repr=False)
    group: Any = field(repr=False)
    numerical: Any = field(repr=False)

    def summary(self) -> str:
        """The invariants, one per line, then whether they are fractional and the class, a line each."""
        lines = [f"{_group(self.group)}: the Z-valued stable real-space invariants of the "
                 "bands   (.group, .theta_Z)"]
        lines += [f"  {name:>9s} = {str(value):>8s}" for name, value in self.theta_Z.items()]
        lines.append("  fractional   (.theta_Z)" if _fractional(self.theta_Z)
                     else "  integral   (.theta_Z)")
        if self.indicator is not None:
            lines.append(_indicator_line(self.indicator, "their"))
        return "\n".join(lines)

    def line(self) -> str:
        """The record on one line, for a table of band sets: the invariants, whether they are fractional, and the class.

        ``theta_1 to theta_8 = (-1/4, 0, 1/2, 1, 1/4, -1/2, -1/4, 1/4),
        fractional; class (0, 1) in Z2 x Z4, of order 4``; a trivial indicator
        group or a zero class is named.
        """
        out = (f"{_invariants_text(self.theta_Z)}, "
               + ("fractional" if _fractional(self.theta_Z) else "integral"))
        if self.indicator is None:
            return out
        if not self.indicator.divisors:
            return out + "; trivial indicator group"
        if not any(self.indicator.values):
            return out + f"; class zero in {self.indicator.indicator_group}"
        return (out + f"; class {self.indicator.values} in {self.indicator.indicator_group}, "
                f"of order {self.indicator.order}")

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return (f"BandSRSIs({_label(self.group)}, {_invariants_text(self.theta_Z)}, "
                f"{'fractional' if _fractional(self.theta_Z) else 'integral'})")


def srsi_of_bands(symmetry_data, realized=None) -> BandSRSIs | Refusal:
    r"""The integer-valued stable real-space invariants of a band set, from its symmetry data vector.

    They are the invariants of a rational site-symmetry vector ``p`` with the band
    set's symmetry data vector, every such ``p`` giving the same ones, and they are
    fractional exactly when the symmetry indicator is nonzero.  The invariants
    valued modulo n are no function of the symmetry data vector, so none are
    returned; :func:`srsi_of_model` reads them off a model's orbitals.

    Parameters
    ----------
    symmetry_data : ..sewing.HSPTable or mapping
        The table :func:`..sewing.hsp_table` returns, which must be complete,
        or ``{corep: multiplicity}`` written out by hand, as
        :func:`..ebr.symmetry_indicator` takes it.
    realized : ..groups.RealizedGroup, optional
        What :func:`..groups.realized_group` returned, ``r``, the measurement
        the coreps came from, in a grey group with tables.  Pass it beside a
        vector written out by hand, and with a table nothing.

    Returns
    -------
    BandSRSIs or Refusal
        ``out.summary()`` prints the invariants, whether they are fractional, and
        the symmetry-indicator class, and ``out.line()`` the same on one line;
        ``out.theta_Z`` holds the invariants by name and ``out.indicator`` the
        class.  A ``Refusal`` in its place is falsy.

    Refusals
    --------
    A refusal handed in as ``realized`` is returned as it is.

    ``srsi_scope``, ``srsi_table_absent``, ``srsi_data``, ``srsi_ebr_basis``
        from :func:`srsi_tables`: the group has no tables.
    ``incomplete_content``, ``not_a_band_structure``, ``indicator_not_evaluable``
        from :func:`..ebr.symmetry_indicator`, with the refusals of the
        symmetry data vector it reads.

    Raises
    ------
    TypeError
        A table with anything beside it, a vector written out by hand with
        nothing beside it, a second argument that is not what
        :func:`..groups.realized_group` returned, or a first that is neither a
        table nor a mapping.
    ..refusal.InternalError
        The rational solve does not reproduce the symmetry data vector it was built
        from: a defect to report.
    """
    source = _ebr._read("srsi_of_bands", symmetry_data, realized)
    if isinstance(source, Refusal):
        return source
    return _band_srsis(source, symmetry_data)


def _srsi_of_bands(symmetry_data, group, *, rep=None) -> BandSRSIs | Refusal:
    """What :func:`srsi_of_bands` returns for a vector read in ``group``, with the site irreps and heights of ``rep`` where it is given; a table is read in what it carries."""
    return _band_srsis(_ebr._source_of(symmetry_data, group, rep), symmetry_data)


def _band_srsis(source, symmetry_data) -> BandSRSIs | Refusal:
    """What :func:`srsi_of_bands` returns for ``symmetry_data`` read in ``source``."""
    group = source.group
    gate = _spacegroup._lattice_not_descended(group)
    if gate is not None:
        return gate
    tables = srsi_tables(group)          # a measured GroupData, not an SRSITables:
    if not tables:                       # the momentum-space half needs the group
        return tables
    si = _ebr._indicator_in(source, symmetry_data)
    if not si:
        return si
    data = _tables._ebr_data(group)
    if not data:
        return data
    u, v, d, matrix = _smith._smith_system(data, group.number_str)
    p = _rational_p(u, v, d, [Fraction(int(x)) for x in si._y])
    n_row, n_col = len(matrix), len(matrix[0])
    check = [sum((Fraction(matrix[i][j]) * p[j] for j in range(n_col)), Fraction(0))
             for i in range(n_row)]
    if check != [Fraction(int(x)) for x in si._y]:
        raise InternalError(
            f"#{group.number_str}: the rational solve of BR.p = y does not reproduce "
            "the symmetry data vector it was built from, though the compatibility relations "
            "hold (symmetry_indicator checked them). The Smith factors and the EBR "
            "matrix this package returns are not a decomposition of each other, which "
            "is arithmetic here failing rather than a measurement about the model.")
    got = tables._theta_columns(p)
    named = {}
    for (name, position), value in zip(data["ebrs"], p):
        if value:
            term = _tables._ebr_term(name, position)
            named[(term.irrep, term.wyckoff)] = value
    return BandSRSIs(theta_Z=got.theta_Z, indicator=si, group=group,
                     numerical=BandSRSIsNumerical(p=named))


# ---------------------------------------------------------------------------
# a model -> its invariants, a row for each Wyckoff orbit
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False, repr=False)
class Localization(BandRepresentation):
    """One decomposition of a Wyckoff orbit's symmetry data vector into elementary band representations at one position.

    A sum of term rows, as :class:`..symmetry.BandRepresentation` is: iterating,
    indexing and ``len()`` go over :attr:`terms`, and two sums are equal where
    their terms are.  An entry of ``o.numerical.localizations``.

    Attributes
    ----------
    terms : tuple of ..groups.Term
        The elementary band representations, each a site irrep at
        :attr:`wyckoff`; empty for a row whose site irreps induce a composite
        band representation, whose site irreps ``o.numerical.composite``
        holds.
    label : str
        The sum as the paper writes it, ``"1 (-E1g)_a + 1 (-E1u)_a"``.
    wyckoff : str
        The position decomposed at, ``"3a"``.
    theta_Z, theta_Zn : dict
        Its invariants, by name; the same for every localization of a row.
    """

    wyckoff: str = field(default="", compare=False)
    theta_Z: dict = field(default_factory=dict, compare=False)
    theta_Zn: dict = field(default_factory=dict, compare=False)


@dataclass(frozen=True, repr=False)
class WyckoffOrbitSRSIsNumerical(WyckoffOrbitNumerical):
    """How a Wyckoff orbit's invariants were read: its symmetry data vector, and every decomposition of it.

    ``o.numerical`` of a row of ``mo.wyckoff_orbits``: what the row of
    ``r.wyckoff_orbits`` holds there, and the fields below.

    Attributes
    ----------
    localizations : tuple of Localization
        Every decomposition of the row's symmetry data vector into elementary
        band representations at a position of ``o.slides_onto``; all give the
        same invariants.
    chosen : Localization
        The one returned, the first of :attr:`localizations`.
    stably_equivalent : tuple of ..srsi.StablePair
        The tables' pairs of site irreps no invariant tells apart that the
        decompositions at one position use.
    composite : tuple of ..groups.Term or None
        The site irreps read off the representation at the row's own
        position, for a row whose band representation is no sum the list of
        elementary band representations carries; ``None`` otherwise.
    agree : bool
        Whether every localization gives the same invariants; localizations
        that disagree raise :class:`~..refusal.InternalError`.
    symmetry_data : dict
        ``{corep: multiplicity}``: the coreps the row's states carry, measured
        at :attr:`kpoints`.
    kpoints : tuple of str
        The high-symmetry points they were measured at.
    n_free : int
        The free parameters of the row's Wyckoff position.
    """

    localizations: tuple = ()
    chosen: Any = None
    stably_equivalent: tuple = ()
    composite: Any = None
    agree: bool = True
    symmetry_data: dict = field(default_factory=dict)
    kpoints: tuple = ()
    n_free: int = 0


@dataclass(frozen=True, repr=False)
class WyckoffOrbitSRSIs(WyckoffOrbit):
    """One Wyckoff orbit of a model's orbitals, with its stable real-space invariants.

    A row of ``mo.wyckoff_orbits``: the row of ``r.wyckoff_orbits`` for the
    same orbitals, its fields the same values, and the fields below.

    Attributes
    ----------
    theta_Z : dict
        ``{name: value}``: the row's integer-valued invariants.
    theta_Zn : dict
        ``{name: ModularValue}``: its invariants valued modulo n.
    slides_onto : tuple of str
        The positions the tables list elementary band representations at that
        the orbitals can slide onto along their variety, ``("3a", "3b")``; the
        row's own among them where the tables list one there.
    """

    theta_Z: dict = field(default_factory=dict)
    theta_Zn: dict = field(default_factory=dict)
    slides_onto: tuple = ()


def _slides(row) -> bool:
    """Whether a row's invariants are read at a position its orbitals slide onto rather than at its own."""
    return row.numerical.composite is None and row.wyckoff not in row.slides_onto


@dataclass(frozen=True)
class ModelSRSIs:
    r"""A model's stable real-space invariants, read off its orbitals, a row for each Wyckoff orbit.

    What :func:`srsi_of_model` returns; :meth:`summary` prints it.  A model is an
    atomic configuration: its invariants are whole numbers, and those valued
    modulo n are its own.

    Attributes
    ----------
    theta_Z : dict
        ``{name: value}``: the integer-valued invariants, summed over the rows;
        the same as :func:`srsi_of_bands` on all the model's bands.
    theta_Zn : dict
        ``{name: ModularValue}``: the invariants valued modulo n, summed modulo
        each n.
    group : ..groups.GroupData
        The group they were read in, ``r.group``.
    wyckoff_orbits : tuple of WyckoffOrbitSRSIs
        A row for each Wyckoff orbit, in the order of ``r.wyckoff_orbits``.
    """

    theta_Z: dict
    theta_Zn: dict
    group: Any = field(repr=False)
    wyckoff_orbits: tuple = ()

    def summary(self) -> str:
        """The invariants, then each Wyckoff orbit's, a line each."""
        lines = [f"{_group(self.group)}: stable real-space invariants   (.group)",
                 f"  the whole model: {_invariants_text(self.theta_Z, self.theta_Zn)}"
                 + ("   (.theta_Z, .theta_Zn)" if self.theta_Zn else "   (.theta_Z)")]
        columns = [_span(self.theta_Z)] + [f"{_span(names)} mod {n}"
                                           for n, names, _v in _zn_groups(self.theta_Zn)]
        slid = any(_slides(row) for row in self.wyckoff_orbits)
        lines.append("  each Wyckoff orbit, its site irreps written m (irrep)_w, with "
                     + _and(columns)
                     + ("   (.wyckoff_orbits, .slides_onto)" if slid
                        else "   (.wyckoff_orbits)"))
        for row in self.wyckoff_orbits:
            text = (f"    {row.wyckoff} ({row.site_symmetry}): "
                    f"{_orbitals_named(row.orbitals)}, {_paper_label(row.terms)}")
            if _slides(row):
                text += f", slides onto {_and(row.slides_onto)}"
            values = _values(row.theta_Z.values()) + "".join(
                f"; {_values(vals)} mod {n}" for n, _names, vals in _zn_groups(row.theta_Zn))
            lines.append(f"{text}: {values}")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return (f"ModelSRSIs({_label(self.group)}, "
                f"{_count(len(self.wyckoff_orbits), 'Wyckoff orbit')}, "
                f"{_invariants_text(self.theta_Z)})")


def _unpack_realized(realized, call):
    """``(model, group, assignment, rep)`` of what :func:`..groups.realized_group` returned.

    ``TypeError`` for anything else, ``call`` naming the call in its message.
    """
    from ._realized import RealizedGroup

    if isinstance(realized, RealizedGroup):
        return (realized.model, realized.group, realized._assignment, realized.rep)
    raise TypeError(f"{call}(): pass what realized_group returns, r, not a tuple of its "
                    "parts" if isinstance(realized, (tuple, list)) else
                    f"{call}(): pass what realized_group returns, r; got a "
                    f"{type(realized).__name__}")


def _list_positions(data) -> dict:
    """``{letter: "3a"}``: each letter the list of elementary band representations names, with its multiplicity."""
    out = {}
    for _name, position in data["ebrs"]:
        match = _POSITION.match(str(position))
        out.setdefault(match.group(2), match.group(1) + match.group(2))
    return out

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
    points_map = _tables._high_symmetry_points(group)
    if not points_map:
        return [(None, None, points_map) for _rows in blocks]
    read = [({}, [], {}) for _rows in blocks]
    for name in points_map:
        k = points_map[name]
        lg = _tables._little_group(group, k)
        why = None
        if isinstance(lg, Refusal):
            why = lg
        elif len(lg.unitary) <= 1:
            why = "trivial little group: no corep to identify"
        operators = []
        for sym in (() if why is not None else lg._unitary_operations):
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
            got = _tables._identify_from_characters(group, k, [chars],
                                                   multiplet_dims=[len(rows)])
            if isinstance(got, Refusal):
                refusals[name] = got
                continue
            points.append(str(got.kpoint))
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
            if _tables.kpoint_of_label(label) in set(points)]
    columns = [j for j, (_name, position) in enumerate(data["ebrs"])
               if _POSITION.match(str(position)).group(2) in set(letters)]
    if not columns or (not keep and bands is None):
        return "INFEASIBLE", (), columns, keep
    matrix = np.asarray(data["matrix"], dtype=int)[np.ix_(keep, columns)]
    y = np.array([int(counts.get(labels[i], 0)) for i in keep], dtype=int)
    if bands is not None:
        matrix = np.vstack([matrix,
                            [[_tables._bands_spanned(data["ebrs"][j][0]) for j in columns]]])
        y = np.append(y, int(bands))
    status, solutions, _truncated = _ebr._cpsat(matrix, y, 0, int(bound), cap)
    return status, solutions, columns, keep

def srsi_of_model(realized) -> "ModelSRSIs | Refusal":
    """A model's stable real-space invariants, read off its orbitals, a row for each Wyckoff orbit.

    Each Wyckoff orbit's symmetry data vector is measured on its own orbitals
    and decomposed into elementary band representations at its Wyckoff position
    or, where the tables list none there, at a position its orbitals slide onto;
    a row's invariants are that decomposition's, and the model's their sum.  A
    model's orbitals are a known configuration, so both kinds are returned.

    Parameters
    ----------
    realized : ..groups.RealizedGroup
        What :func:`..groups.realized_group` returned, ``r``.

    Returns
    -------
    ModelSRSIs or Refusal
        ``out.summary()`` prints the invariants and each row's; ``out.theta_Z``
        and ``out.theta_Zn`` hold the model's, by name, and
        ``out.wyckoff_orbits`` a row for each Wyckoff orbit, the row of
        ``r.wyckoff_orbits`` with its invariants.  A ``Refusal`` in its place is
        falsy.

    Raises
    ------
    TypeError
        ``realized`` is not what :func:`..groups.realized_group` returned.
    ..refusal.InternalError
        Two localizations of one Wyckoff orbit give different invariants: a
        defect to report.

    Refusals
    --------
    ``srsi_scope``, ``srsi_table_absent``, ``srsi_data``, ``srsi_ebr_basis``
        from :func:`srsi_tables`: the group has no tables.
    ``wyckoff_orbit_content_incomplete``
        a Wyckoff orbit's symmetry data vector could not be read at a
        high-symmetry point.
    ``wyckoff_orbit_not_localizable``
        a Wyckoff orbit's symmetry data vector is a band representation of no
        position its orbitals can slide onto.
    ``wyckoff_setting``
        the symmetry data vector decomposes at a position the geometry says the
        orbitals are not at: the package disagreeing with itself, a case to
        report.
    ``tables_not_ready``, ``wyckoff_unresolved``, ``ebr_tables``
        from :func:`..groups.wyckoff_sites_tables` and :func:`..groups.ebr_data`.

    Examples
    --------
    >>> print(srsi_of_model(realized).summary())             # doctest: +SKIP
    """
    if isinstance(realized, Refusal):
        return realized
    model, group, assignment, rep = _unpack_realized(realized, "srsi_of_model")
    gate = _spacegroup._lattice_not_descended(group)
    if gate is not None:
        return gate
    tables = srsi_tables(group)
    if not tables:
        return tables
    data = _tables._ebr_data(group)
    if not data:
        return data
    sites = _tables._sites_in_tables_setting(group, assignment.site_positions)
    if isinstance(sites, Refusal):
        return sites
    # a free-standing layer's orbitals are moved within the layer only: the
    # positions off it are never a localization
    layer = _ebr._layer_reading(group, data, rep)
    off_layer = (set() if layer is None
                 else {_tables._letter_only(position) for position in layer["outside"]})
    positions = _list_positions(data)
    pairs = _stable_pairs(int(tables._sg), bool(tables._spinor))

    nspin = int(getattr(model, "nspin", 1))
    records = []
    for index, row in enumerate(realized.wyckoff_orbits):
        orbitals = tuple(int(o) for o in row.orbitals)
        states = [o * nspin + s for o in orbitals for s in range(nspin)]
        counts, points, refusals = _orbit_content(group, rep, states)
        if counts is None:
            return refusals
        hard = {name: why for name, why in refusals.items()
                if not isinstance(why, str)}
        if hard:
            name, why = sorted(hard.items())[0]
            return Refusal(
                "wyckoff_orbit_content_incomplete",
                f"the Wyckoff orbit of orbitals {list(orbitals)} at {row.wyckoff} could not "
                f"be identified at {name} of #{group.number_str}: "
                f"{getattr(why, 'reason', why)}. Its symmetry data vector is incomplete.",
                context={"orbit": index, "orbitals": orbitals, "kpname": name,
                         "refusal": why, "identified": points,
                         "refusals": tuple(sorted(refusals, key=_tables.point_order))})

        geometry = [sites[int(s)] for s in row._sites]
        letters = {site.letter for site in geometry}
        if len(letters) != 1:                     # pragma: no cover - guard
            raise InternalError(
                f"#{group.number_str}: the sites of the Wyckoff orbit of orbitals "
                f"{list(orbitals)} sit on different Wyckoff letters {sorted(letters)} in "
                "the tables' reference cell, though they are one orbit of the group's own "
                "action. The orbit structure and the reference-cell letters cannot both be "
                "right.")
        here = geometry[0]
        where = f"{here.multiplicity}{here.letter}"

        # the letters the tables' EBR columns are keyed by, on the closure of
        # the row's variety: its own among them where they list one there, and
        # within the layer on a free-standing one
        localizations, same, same_letter = [], (), None
        tried = tuple(letter for letter in here.closure if letter not in off_layer)
        for letter in tried:
            _status, solutions, columns, _keep = _solve_at_letter(
                data, counts, points, letter, len(states))
            if len(solutions) > 1:
                same = tuple(sorted(
                    {str(data["ebrs"][columns[j]][0])
                     for solution in solutions for j, m in enumerate(solution) if m}))
                same_letter = letter
            for solution in solutions:
                vector = [0] * len(data["ebrs"])
                for j, multiplicity in zip(columns, solution):
                    vector[j] = int(multiplicity)
                got = tables._theta_columns(vector)
                terms = _terms_of(tuple((int(m), str(data["ebrs"][j][0]),
                                         str(data["ebrs"][j][1]))
                                        for j, m in enumerate(vector) if m))
                localizations.append(Localization(
                    terms=terms, label=_paper_label(terms), wyckoff=positions[letter],
                    theta_Z=got.theta_Z, theta_Zn=got.theta_Zn))

        composite = None
        if not localizations:
            found = _composite_localization(tables, group, assignment, rep, row._sites,
                                            here, orbitals=orbitals)
            if found is None:
                elsewhere = (_letters_that_do_fit(data, counts, points, len(states))
                             if tried else ())
                if elsewhere:
                    return _route_disagreement(group, index, orbitals, here,
                                               elsewhere, counts, positions)
                return Refusal(
                    "wyckoff_orbit_not_localizable",
                    f"the Wyckoff orbit of orbitals {list(orbitals)} at {where} of "
                    f"#{group.number_str}: "
                    + ("no position the tables list elementary band representations "
                       "at lies on the closure of its variety, so there is nowhere "
                       "to localize it at all"
                       if not tried else
                       f"its measured symmetry data vector {counts} is not a sum of the "
                       "elementary band representations at "
                       + _and([positions[letter] for letter in tried]))
                    + ", and its site irreps are not among the composite ones the "
                    "tables carry either. A Wyckoff orbit whose symmetry data vector is "
                    "not a band representation of the position it sits on is a "
                    "statement about the model.",
                    context={"orbit": index, "orbitals": orbitals,
                             "content": dict(counts), "letter": here.letter,
                             "closure": here.closure, "tried": tuple(tried),
                             "points": points,
                             "number_str": str(group.number_str)})
            composite = tuple(
                _spacegroup.Term(multiplicity=int(m), irrep=str(irrep), wyckoff=where,
                                 site_symmetry=row.site_symmetry,
                                 unitary_site_symmetry=row.unitary_site_symmetry)
                for irrep, m in found["irreps"].items())
            got = tables._theta_at(found["irreps"], here.letter)
            localizations = [Localization(terms=(), label="", wyckoff=where,
                                          theta_Z=got.theta_Z, theta_Zn=got.theta_Zn)]

        values = {(tuple(entry.theta_Z.values()),
                   tuple((name, value.value) for name, value in entry.theta_Zn.items()))
                  for entry in localizations}
        if len(values) != 1:
            raise InternalError(
                f"#{group.number_str}: the localizations of the Wyckoff orbit of orbitals "
                f"{list(orbitals)} give {len(values)} different SRSIs ("
                + str([(entry.wyckoff, tuple(str(v) for v in entry.theta_Z.values()))
                       for entry in localizations])
                + "), though every one of them is the same block of orbitals moved "
                "adiabatically within the variety it sits on. An SRSI that is not "
                "invariant under a process the tables themselves list is a defect in "
                "this package's arithmetic or in the shipped table, not a measurement "
                "about the model.")
        chosen = localizations[0]
        used = {name.split("↑")[0] for name in same}
        stably = tuple(pair for pair in pairs
                       if same_letter is not None
                       and pair.wyckoff == positions[same_letter]
                       and set(pair.irreps) <= used)
        numerical = WyckoffOrbitSRSIsNumerical(
            **_fields(row.numerical), localizations=tuple(localizations), chosen=chosen,
            stably_equivalent=stably, composite=composite, agree=(len(values) == 1),
            symmetry_data=_tables.ordered_content(counts), kpoints=points,
            n_free=int(here.n_free))
        records.append(WyckoffOrbitSRSIs(
            **_fields(row, numerical=numerical), theta_Z=chosen.theta_Z,
            theta_Zn=chosen.theta_Zn,
            slides_onto=tuple(positions[letter] for letter in here.closure)))

    total_z = {tables._rows[i].name: sum((record.theta_Z[tables._rows[i].name]
                                          for record in records), Fraction(0))
               for i in tables._z_rows}
    total_zn = {}
    for entry in tables._rows:
        if entry.mod is not None:
            n = int(entry.mod)
            total_zn[entry.name] = ModularValue(
                sum(int(record.theta_Zn[entry.name].value) for record in records) % n, n)
    return ModelSRSIs(theta_Z=total_z, theta_Zn=total_zn, group=group,
                      wyckoff_orbits=tuple(records))

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

def _route_disagreement(group, index, orbitals, here, elsewhere, counts, positions) -> Refusal:
    """The one safety gate: geometry and the symmetry data vector name different positions."""
    where = f"{here.multiplicity}{here.letter}"
    return Refusal(
        "wyckoff_setting",
        f"the Wyckoff orbit of orbitals {list(orbitals)} of #{group.number_str} sits at "
        f"{where} by geometry (carried into the tables' reference cell), and its measured "
        "symmetry data vector decomposes at "
        f"{_and([positions.get(letter, letter) for letter in elsewhere])} instead, not "
        "at that position"
        + (f", nor at {_and([positions.get(letter, letter) for letter in here.closure])}, "
           "the positions with elementary band representations on the closure of its "
           "variety"
           if tuple(here.closure) != (here.letter,) else "")
        + ". The two routes to a Wyckoff position disagree, so the setting the positions "
        "are read in is not the one the tables are written in, and no invariant is "
        "returned",
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
    tuple of ..symmetry._OrbitBandRepresentation
        One per Wyckoff orbit of orbitals, the orbit of one multiplet under the
        group, in the order the orbits' first orbitals come in the model
        (:func:`..symmetry._wyckoff_orbits_of`); empty where the tables do not
        reach the cell, and where the
        representation leaves some unitary operation of its group out — a
        ``_build_rep`` handed a group larger than the one the Hamiltonian obeys
        — since the tables are the whole group's.
    """

    characters = getattr(rep, "_characters", None)
    group, assignment, model = rep._group, rep._assignment, rep._model
    if characters is None:
        return ()
    represented = {int(i) for i in rep._indices}
    if any(int(s.ind) not in represented for s in group._symmetries if not s.time_reversal):
        return ()
    sites = _tables._sites_in_tables_setting(group, assignment.site_positions)
    if isinstance(sites, Refusal):
        return ()
    data = _tables._ebr_data(group)
    if isinstance(data, Refusal):
        return ()
    ebrs = tuple((str(name), str(position)) for name, position in data["ebrs"])
    # a free-standing layer's own band representations, and the positions off
    # it that its orbitals are never moved onto
    layer = _ebr._layer_reading(group, data, rep)
    columns = list(range(len(ebrs))) if layer is None else list(layer["columns"])
    off_layer = (set() if layer is None
                 else {_tables._letter_only(position) for position in layer["outside"]})
    tables = srsi_tables(group)
    tables = tables if isinstance(tables, SRSITables) else None
    symmetry_of_site ={int(o.site): str(o.site_symmetry) for o in characters.orbitals}
    nspin = int(getattr(model, "nspin", 1))
    blocks = [(orbit, orbitals, [o * nspin + s for o in orbitals for s in range(nspin)])
              for orbit, orbitals in _wyckoff_orbits_of(rep)]
    # The operators check the orbitals' positions against the operations' action
    # on them, at their own tolerance, as every other use of these operators
    # does.  The centres a representation is built on are exact -- moved there
    # under a declared precision, and within that tolerance wherever the
    # hopping projection ran -- so the check reads rounding.  A representation
    # built by hand on centres the operations miss by more reads no content
    # here, as it reads none anywhere else, and each orbit says why.

    def where_of(orbit, orbitals):
        here = sites[int(orbit[0])]
        return here, {"orbitals": orbitals,
                      "wyckoff": f"{int(here.multiplicity)}{here.letter}",
                      "site_symmetry": symmetry_of_site.get(int(orbit[0]), "")}

    ops = {int(s.ind): s for s in group._symmetries if int(s.ind) in represented}
    miss = float(_position_defect(model, group, assignment, ops))
    if miss > _BLOCH_TOL:
        note = ("no sum of elementary band representations, the symmetry data vector could "
                "not be identified: the orbital centers miss "
                f"the operations by {miss:.3g} (reduced), more than the {_BLOCH_TOL:.0e} the "
                "Bloch operators carry them to; move them onto the group's positions first, "
                "with realized_group's precision=")
        return tuple(_OrbitBandRepresentation(**where_of(orbit, orbitals)[1], terms=(), note=note)
                     for orbit, orbitals, _rows in blocks)
    measured = _orbit_contents(group, rep, [rows for _orbit, _orbitals, rows in blocks])
    out = []
    for (orbit, orbitals, rows), (counts, points, refusals) in zip(blocks, measured):
        here, where = where_of(orbit, orbitals)
        hard = ({} if counts is None else
                {name: why for name, why in refusals.items() if not isinstance(why, str)})
        if counts is None or hard:
            name, why = (("", refusals) if counts is None else sorted(hard.items())[0])
            out.append(_OrbitBandRepresentation(
                **where, terms=(),
                note="no sum of elementary band representations, the symmetry data vector "
                     "could not be identified"
                     + (f" at {name}" if name else "") + f": {getattr(why, 'reason', why)}"))
            continue
        # THE ORBIT'S OWN SUM: its content decomposed at the positions its
        # orbitals can be moved onto, one position at a time, its own first --
        # srsi_of_model's localization, which needs no invariants
        letters = [here.letter] if here.letter in here.closure else []
        letters += [letter for letter in here.closure if letter not in letters]
        named = frozenset(term for o in characters.orbitals if int(o.site) in set(orbit)
                          and int(o.orbital) in set(orbitals)
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
            composite = _composite_localization(tables, group, assignment, rep, orbit, here,
                                                orbitals=orbitals)
        if own is None and composite is None:
            listed = frozenset(_tables._letter_only(position) for _name, position in ebrs)
            through = []
            for variety in here.varieties:
                for letter in _spacegroup._letters_on_closure(int(group._sg_number),
                                                          str(variety), listed):
                    if letter not in letters + through and letter not in off_layer:
                        through.append(letter)
            own = _sum_at_first(data, counts, points, ebrs, len(rows), named, through)
        # EVERY SUM WITH THIS CONTENT, over the layer's own band representations:
        # those its invariants do not tell from the orbit's own sum are its
        # alternatives, and a composite orbit's own sum is among them
        every = ()
        if tables is not None or own is None:
            restriction = _ebr.in_plane_restriction(group, kpoints=points,
                                                    require_exact=False, _layer=layer)
            if not isinstance(restriction, Refusal):
                matrix, y, bound = _ebr._content_system(data, counts, restriction.retained)
                # the band count, one more equation: all a content read at no
                # point fixes, and each multiplicity's bound there
                matrix = np.vstack([matrix,
                                    [[_tables._bands_spanned(name) for name, _at in ebrs]]])
                y = np.append(y, len(rows))
                bound = bound if restriction.retained else len(rows)
                _status, found, _cut = _ebr._cpsat(matrix[:, columns], y, 0, bound,
                                                   _ebr.DEFAULT_ENUMERATION_CAP)
                every = _ebr._ordered(_ebr._lifted(found, columns, len(ebrs)))
        if own is None and composite is not None:
            theirs = tables._theta_at(composite["irreps"], here.letter)
            own = next((s for s in every if tables._theta_columns(s) == theirs), None)
        elif own is None and len(every) == 1:
            own = every[0]
        alike = ()
        if own is not None and tables is not None:
            mine = tables._theta_columns(own)
            alike = tuple(s for s in every
                          if s != own and tables._theta_columns(s) == mine)
        if own is None:
            out.append(_OrbitBandRepresentation(
                **where, terms=(),
                note=("no sum of elementary band representations carries the symmetry "
                      "data vector" if not every else
                      f"{len(every)} sums of elementary band representations carry the "
                      "symmetry data vector, and nothing read here says which is induced")))
            continue
        terms = tuple(
            tuple((int(m), name, position)
                  for (name, position), m in zip(ebrs, solution) if m)
            for solution in (own,) + alike)
        out.append(_OrbitBandRepresentation(**where, terms=terms[0], alternatives=terms[1:]))
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

def _composite_localization(tables, group, assignment, rep, orbit, here, *, orbitals=None):
    r"""The site irreps of an orbit whose induced band representation is composite.

    :math:`{}^1E{}^2E` at site symmetry :math:`\bar4` and ``E`` at :math:`\bar4m2` /
    :math:`\bar42m` are genuine site-symmetry irreps whose induced band
    representation is *not* elementary, so no EBR column carries them, the
    momentum-space decomposition has no solution — **and no EBR entry names
    them**, so the label the record carries at such a position is the site
    point group's own.  Where that name is not the tables' key, what is matched
    is the **content**: a row of the site's own character table is paired with
    the maximal site irrep of the tables (``tables._maximal_keys``) whose band
    count its induction reproduces, which is the one number a composite key and a computed row
    share.  A row whose name is a key of the tables at that letter is taken by
    its name.

    ``orbitals``, where given, are the orbit's own, a Wyckoff orbit's: only
    their multiplets at the orbit's first site are read, where every orbital
    at that site is read otherwise.

    Returns ``{"irreps": {irrep: multiplicity}, "site": n}`` or ``None`` — the
    latter whenever anything is not measured cleanly, so :func:`srsi_of_model`
    refuses rather than guessing.  All 39 composite positions are in the
    single-valued table, so a spinful group takes this route not at all.
    """
    characters = getattr(rep, "_characters", None)
    if characters is None or tables._spinor:
        return None
    site = int(orbit[0])
    table = (characters.site_tables or {}).get(site)
    known = {name for key in tables.site_irreps if key._letter == here.letter
             for name in (key.irrep, key._paper) if name}
    free = [key for key in tables._maximal_keys
            if key.letter == here.letter and not key.ebr]
    labels: dict = {}
    seen = set()
    own = None if orbitals is None else {int(o) for o in orbitals}
    for entry in characters.orbitals:
        if int(entry.site) != site or entry.multiplet in seen:
            continue
        if own is not None and int(entry.orbital) not in own:
            continue
        seen.add(entry.multiplet)
        name = str(entry.irrep) if entry.irrep in known else ""
        if not name:
            if table is None:
                return None
            bands = _naming._induced_bands(group, assignment, site, entry.character)
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
# the no-go criterion
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False, repr=False)
class AtomicRealization(BandRepresentation):
    r"""One atomic realization: a set of orbitals at maximal Wyckoff positions with a band set's symmetry data.

    A sum of term rows, as :class:`..symmetry.BandRepresentation` is: iterating,
    indexing and ``len()`` go over :attr:`terms`, and a realization and the same
    sum from :func:`..ebr.decompose` are equal.  An entry of
    ``rz.realizations``.

    Attributes
    ----------
    terms : tuple of ..groups.Term
        The site irreps it holds, each at its maximal Wyckoff position.
    label : str
        The sum as the paper writes it, ``"1 (-E2)_b + 1 (-E)_c"``.
    theta_Zn : dict
        ``{name: ModularValue}``: its own invariants valued modulo n, which two
        realizations of one symmetry data vector need not share.
    stably_equivalent : tuple of ..srsi.StablePair
        The site irreps it holds that no invariant tells apart from a partner.
    """

    theta_Zn: dict = field(default_factory=dict, compare=False)
    stably_equivalent: tuple = field(default=(), compare=False)


@dataclass(frozen=True)
class RealizationsNumerical:
    """How completely the atomic realizations were listed.

    ``rz.numerical`` of what :func:`atomic_realizations` returns.

    Attributes
    ----------
    bound : int
        The largest multiplicity of one site irrep the band count allows, which
        makes the list complete; ``0`` where no enumeration ran.
    cap : int
        The most realizations the list may hold.
    """

    bound: int
    cap: int


@dataclass(frozen=True)
class Realizations:
    r"""Every atomic realization of a band set's symmetry data, or none.

    What :func:`atomic_realizations` returns; :meth:`summary` prints it.  The list
    is complete, so an empty one says something: with whole-number invariants, a
    fragile band set; with fractional ones, a stable topological one.  The
    record is truthy either way.

    Attributes
    ----------
    realizations : tuple of AtomicRealization
        Every one; empty where none exists, and where ``theta_Z`` is
        fractional.
    theta_Z : dict
        ``{name: value}``: the integer-valued invariants they reproduce, as
        handed in.
    group : ..groups.GroupData
        The group of the tables, ``tables.group``.
    numerical : RealizationsNumerical
        How completely they were listed.
    """

    realizations: tuple
    theta_Z: dict
    group: Any = field(repr=False)
    numerical: Any = field(repr=False)
    _status: str = field(default="", repr=False, compare=False)

    def summary(self) -> str:
        """The band set, then every realization of it, a line each."""
        head = (_count(len(self.realizations), "atomic realization") if self.realizations
                else "no atomic realization")
        lines = [f"{_group(self.group)}: {head} with Z-valued stable real-space invariants "
                 f"{_invariants_text(self.theta_Z)}   (.group, .realizations, .theta_Z)"]
        if not self.realizations:
            if _fractional(self.theta_Z):
                lines.append("  fractional   (.theta_Z)")
            return "\n".join(lines)
        zn = self.realizations[0].theta_Zn
        lines.append("  each realization"
                     + (f", with {_zn_header(zn)}   (.label, .theta_Zn of each)" if zn
                        else "   (.label of each)"))
        lines += [f"    {c.label}" + (f"   {_zn_values(c.theta_Zn)}" if c.theta_Zn else "")
                  for c in self.realizations]
        pairs = []
        for c in self.realizations:
            pairs += [pair for pair in c.stably_equivalent if pair not in pairs]
        if pairs:
            lines.append("  site irreps they use that no stable real-space invariant "
                         "distinguishes   (.stably_equivalent of each)")
            lines += [f"    {_pair_text(pair)}" for pair in pairs]
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return (f"Realizations({_label(self.group)}, "
                f"{_count(len(self.realizations), 'realization')})")


@dataclass(frozen=True)
class Splitting:
    r"""Whether the two sides of a gapped splitting of bands can both be atomic insulators, by the paper's no-go criterion.

    What :func:`splitting` and :func:`nsit_criterion` return; :meth:`summary`
    prints it.  The valence bands are the bands asked for, the conduction bands
    the rest; the record is truthy.

    Attributes
    ----------
    nsit : bool
        Whether the splitting is non-symmetry-indicated topological (NSIT):
        both sides have atomic realizations, and no realization of the valence
        bands and of the conduction bands has invariants valued modulo n that
        add up to the whole model's, so the two sides cannot both be atomic
        insulators.  It is false where the valence bands' invariants are
        fractional, where a side has no atomic realization, where the group has
        no invariants valued modulo n, and where some pair adds up: in the
        first two cases the symmetry data vector itself shows a side is not an
        atomic insulator, and in the last two nothing more is seen.
    valence, conduction : Realizations
        Every atomic realization of each side, as :func:`atomic_realizations`
        returns them; empty where that side's invariants are fractional.
    theta_Z : dict
        ``{name: value}``: the whole model's integer-valued invariants, the
        sum of the two sides'.
    theta_Zn : dict
        ``{name: ModularValue}``: the whole model's invariants valued modulo n.
    consistent_pairs : tuple
        Each pair ``(valence, conduction)`` of atomic realizations whose
        invariants valued modulo n add up to the whole model's, modulo each n.
    stably_equivalent : tuple of ..srsi.StablePair
        The site irreps a realization holds that no invariant tells apart.
    group : ..groups.GroupData
        The group of the tables, ``tables.group``.
    """

    nsit: bool
    valence: Any = field(repr=False)
    conduction: Any = field(repr=False)
    theta_Z: dict = field(default_factory=dict)
    theta_Zn: dict = field(default_factory=dict)
    consistent_pairs: tuple = field(default=(), repr=False)
    stably_equivalent: tuple = field(default=(), repr=False)
    group: Any = field(default=None, repr=False)

    def _why(self) -> tuple:
        """The first line's clause, and the fields it rests on."""
        for side, got in (("valence", self.valence), ("conduction", self.conduction)):
            if _fractional(got.theta_Z):
                return (f"not nsit, the {side} bands' invariants being fractional",
                        f"(.group, .nsit, .{side}.theta_Z)")
        empty = [side for side, got in (("valence", self.valence),
                                        ("conduction", self.conduction))
                 if not got.realizations]
        if empty:
            return (f"not nsit, {_and([f'the {side} bands' for side in empty])} having no "
                    "atomic realization",
                    "(.group, .nsit, "
                    + ", ".join(f".{side}.realizations" for side in empty) + ")")
        if not self.theta_Zn:
            return ("not nsit, the group having no invariants valued modulo n",
                    "(.group, .nsit, .theta_Zn)")
        if self.consistent_pairs:
            return (f"not nsit, {_count(len(self.consistent_pairs), 'pair')} of a valence "
                    "and a conduction realization adding up to the whole model's "
                    "invariants valued modulo n", "(.group, .nsit, .consistent_pairs)")
        return ("nsit, no realization of the valence bands and of the conduction bands "
                "adding up to the whole model's invariants valued modulo n",
                "(.group, .nsit)")

    def summary(self) -> str:
        """Whether the splitting is NSIT and why, then the two sides, their invariants and their realizations, a line each."""
        clause, pointers = self._why()
        lines = [f"{_group(self.group)}: {clause}   {pointers}"]
        lines.append(f"  Z-valued stable real-space invariants {_span(self.theta_Z)}: the "
                     f"whole model {_values(self.theta_Z.values())} = valence "
                     f"{_values(self.valence.theta_Z.values())} + conduction "
                     f"{_values(self.conduction.theta_Z.values())}   "
                     "(.theta_Z, .valence.theta_Z, .conduction.theta_Z)")
        if self.theta_Zn:
            lines.append("  invariants valued modulo n of the whole model: "
                         f"{_zn_text(self.theta_Zn)}   (.theta_Zn)")
        for side, got in (("valence", self.valence), ("conduction", self.conduction)):
            if not got.realizations:
                lines.append(f"  the {side} bands have no atomic realization"
                             + (f", their invariants fractional   (.{side}.realizations, "
                                f".{side}.theta_Z)" if _fractional(got.theta_Z)
                                else f"   (.{side}.realizations)"))
                continue
            lines.append(f"  atomic realizations of the {side} bands"
                         + (f", with {_zn_header(self.theta_Zn)}" if self.theta_Zn else "")
                         + f"   (.{side}.realizations)")
            lines += [f"    {realization.label}"
                      + (f"   {_zn_values(realization.theta_Zn)}"
                         if realization.theta_Zn else "")
                      for realization in got.realizations]
        if self.theta_Zn and self.valence.realizations and self.conduction.realizations:
            if self.consistent_pairs:
                lines.append("  pairs of a valence and a conduction realization whose "
                             "invariants valued modulo n add up to the whole model's"
                             "   (.consistent_pairs)")
                lines += [f"    {v.label} with {c.label}" for v, c in self.consistent_pairs]
            else:
                lines.append("  no pair of a valence and a conduction realization has "
                             "invariants valued modulo n adding up to the whole model's"
                             "   (.consistent_pairs)")
        if self.stably_equivalent:
            lines.append("  site irreps the realizations use that no stable real-space "
                         "invariant distinguishes   (.stably_equivalent)")
            lines += [f"    {_pair_text(pair)}" for pair in self.stably_equivalent]
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return (f"Splitting({_label(self.group)}, {'nsit' if self.nsit else 'not nsit'}, "
                f"{_count(len(self.consistent_pairs), 'consistent pair')})")


def _theta_zn_of_keys(tables, p) -> dict:
    """``{name: ModularValue}`` of an integer ``p`` over the maximal site irreps."""
    columns = _maximal_basis(int(tables._sg), bool(tables._spinor))[1]
    out = {}
    for r, row in enumerate(tables._rows):
        if row.mod is not None:
            n = int(row.mod)
            out[row.name] = ModularValue(
                sum(int(columns[j][r]) * int(p[j]) for j in range(len(p)) if p[j]) % n, n)
    return out


def _by_name(call, argument, tables, theta_z) -> dict:
    """``{name: Fraction}`` of integer-valued invariants handed in by name, in the tables' order.

    ``TypeError`` for anything but a mapping, ``ValueError`` for names that are
    not exactly the tables' integer-valued invariants.
    """
    names = [tables._rows[i].name for i in tables._z_rows]
    if not isinstance(theta_z, Mapping):
        raise TypeError(f"{call}(): {argument} is a mapping from each Z-valued invariant's "
                        f"name to its value, such as b.theta_Z; got a "
                        f"{type(theta_z).__name__}")
    if len(theta_z) != len(names) or set(theta_z) != set(names):
        raise ValueError(f"{call}(): {argument} holds "
                         f"{', '.join(str(k) for k in theta_z) or 'nothing'}; "
                         f"#{tables._number_str} has the Z-valued invariants {_span(names)}")
    return {name: Fraction(theta_z[name]) for name in names}


def atomic_realizations(tables, theta_Z, *, cap=256) -> "Realizations | Refusal":
    r"""Every atomic insulator with a band set's symmetry data, each a set of orbitals at maximal Wyckoff positions.

    The configurations with nonnegative integer multiplicities over the maximal
    site irreps whose integer-valued invariants are ``theta_Z``: the same
    invariants are the same symmetry data.  Maximal positions are enough, since
    an atomic insulator moves onto them with its invariants unchanged, and the
    list is complete.  Fractional invariants have none: the band set carries a
    nonzero symmetry indicator, and nothing is enumerated.

    Parameters
    ----------
    tables : SRSITables
        The group's tables, from :func:`srsi_tables`.
    theta_Z : mapping
        ``{name: value}``: the band set's integer-valued invariants, each once,
        as ``b.theta_Z`` holds them.
    cap : int, default 256
        The most realizations to list; more is refused.

    Returns
    -------
    Realizations or Refusal
        ``out.summary()`` prints every realization with its own invariants
        valued modulo n; ``out.realizations`` holds them.  The record is truthy,
        and an empty list with whole-number ``theta_Z`` is a fragile band set.
        A ``Refusal`` in its place is falsy.

    Raises
    ------
    TypeError
        ``theta_Z`` is not a mapping.
    ValueError
        ``theta_Z`` does not name each of the tables' integer-valued invariants
        once.

    Refusals
    --------
    ``srsi_realizations_truncated``
        more than ``cap`` realizations; a larger ``cap=`` lists them.

    Examples
    --------
    >>> print(atomic_realizations(tables, b.theta_Z).summary())   # doctest: +SKIP
    """
    values = _by_name("atomic_realizations", "theta_Z", tables, theta_Z)
    group = tables.group
    if _fractional(values):
        return Realizations(realizations=(), theta_Z=values, group=group,
                            numerical=RealizationsNumerical(bound=0, cap=int(cap)),
                            _status="FRACTIONAL")

    keys = tables._maximal_keys
    ordered = list(values.values())
    form = tables._band_count_form
    implied = sum((form[i] * ordered[i] for i in range(len(ordered))), Fraction(0))
    dims = [int(key.dim) for key in keys]
    if implied.denominator != 1 or implied < 0:
        return Realizations(realizations=(), theta_Z=values, group=group,
                            numerical=RealizationsNumerical(bound=0, cap=int(cap)),
                            _status="INFEASIBLE")
    total = int(implied)

    columns = _maximal_basis(int(tables._sg), bool(tables._spinor))[1]
    rows = [[int(columns[j][r]) for j in range(len(keys))] for r in tables._z_rows]
    matrix = np.array(rows + [dims], dtype=int)
    y = np.array([int(v) for v in ordered] + [total], dtype=int)
    bound = total // min(dims) if dims else 0
    status, solutions, truncated = _ebr._cpsat(matrix, y, 0, int(bound), int(cap))
    if truncated:
        return Refusal(
            "srsi_realizations_truncated",
            f"#{tables._number_str}: the enumeration of atomic realizations of "
            f"{_invariants_text(values)} at {total} bands hit the cap of {cap} "
            "realizations. A no-go is asserted only on a complete enumeration: with the "
            "list cut, neither 'no pair adds up' nor 'none exists' holds. "
            "atomic_realizations(tables, theta_Z, cap=...) with a larger cap= lists them, "
            "theta_Z being context['theta_Z']; nsit_criterion and splitting list each "
            "side at the default cap of 256.",
            value=float(len(solutions)), threshold=float(cap),
            context={"number_str": str(tables._number_str), "cap": int(cap),
                     "n_bands": total, "bound": int(bound), "theta_Z": dict(values)})

    pairs = _stable_pairs(int(tables._sg), bool(tables._spinor))
    records = []
    for solution in solutions:
        touched = tuple(
            pair for pair in pairs
            if any(solution[j] for j, key in enumerate(keys)
                   if key.column is not None and key.column in pair._columns))
        terms = tuple(replace(keys[j].term, multiplicity=int(m))
                      for j, m in enumerate(solution) if m)
        records.append(AtomicRealization(
            terms=terms, label=_paper_label(terms),
            theta_Zn=_theta_zn_of_keys(tables, solution), stably_equivalent=touched))
    return Realizations(
        realizations=tuple(records), theta_Z=values, group=group,
        numerical=RealizationsNumerical(bound=int(bound), cap=int(cap)),
        _status=str(status))


def nsit_criterion(tables, theta_total, theta_Z_v, theta_Z_c) -> "Splitting | Refusal":
    r"""The paper's no-go criterion on a gapped splitting whose invariants are in hand.

    All the bands together are an atomic insulator.  If the two sides, the
    valence and the conduction bands, are atomic insulators too, then the
    invariants valued modulo n of some atomic realization of each add up to the
    whole model's, modulo each n.  Every realization of each side is listed,
    since one pair that does not add up proves nothing.  The splitting is NSIT
    where both sides have realizations and no pair adds up; it is not where the
    valence bands' invariants are fractional or a side has no realization at
    all, which the symmetry data vector already shows, nor where the group has
    no invariants valued modulo n or some pair adds up.

    Parameters
    ----------
    tables : SRSITables
        The group's tables.
    theta_total : Invariants or ModelSRSIs
        The whole model's invariants, with ``theta_Z`` and ``theta_Zn`` by
        name: what ``tables.theta`` returns for the model's orbitals, or what
        :func:`srsi_of_model` returns.
    theta_Z_v, theta_Z_c : mapping
        ``{name: value}``: the integer-valued invariants of the valence and of
        the conduction bands, as ``b.theta_Z`` holds them.

    Returns
    -------
    Splitting or Refusal
        ``out.nsit`` holds whether the splitting is NSIT; ``out.summary()``
        prints it, why, the realizations of each side and the pairs that add
        up.  The record is truthy.  A ``Refusal`` in its place is falsy.

    Raises
    ------
    TypeError
        ``theta_total`` without ``theta_Z`` and ``theta_Zn``, or a side that is
        not a mapping.
    ValueError
        Invariants that are not the tables' own, or two sides that do not add
        up to the whole model's.

    Refusals
    --------
    ``srsi_realizations_truncated``
        from :func:`atomic_realizations`, which lists each side at its default
        cap: more realizations than the cap, which :func:`atomic_realizations`
        with a larger ``cap=`` lists.
    """
    if not (hasattr(theta_total, "theta_Z") and hasattr(theta_total, "theta_Zn")):
        raise TypeError("nsit_criterion(): theta_total holds the whole model's theta_Z and "
                        "theta_Zn, as what tables.theta returns or what srsi_of_model "
                        f"returns; got a {type(theta_total).__name__}")
    theta_z_h = _by_name("nsit_criterion", "theta_total.theta_Z", tables,
                         theta_total.theta_Z)
    values_v = _by_name("nsit_criterion", "theta_Z_v", tables, theta_Z_v)
    values_c = _by_name("nsit_criterion", "theta_Z_c", tables, theta_Z_c)
    zn_names = [row.name for row in tables._rows if row.mod is not None]
    if (len(theta_total.theta_Zn) != len(zn_names)
            or set(theta_total.theta_Zn) != set(zn_names)):
        raise ValueError(
            "nsit_criterion(): theta_total holds the invariants valued modulo n "
            f"{', '.join(str(k) for k in theta_total.theta_Zn) or 'none'}, and "
            f"#{tables._number_str} has {_span(zn_names) or 'none'}; hand in what "
            "tables.theta returns")
    theta_zn_h = {name: getattr(theta_total.theta_Zn[name], "value",
                                theta_total.theta_Zn[name]) for name in zn_names}
    added = {name: values_v[name] + values_c[name] for name in values_v}
    if added != theta_z_h:
        raise ValueError(
            "nsit_criterion(): the valence bands' Z-valued invariants plus the conduction "
            f"bands' are {_invariants_text(added)}, not the whole model's, "
            f"{_values(theta_z_h.values())}. They are linear in p and the symmetry data "
            "vectors add, so they add up for any splitting; what was handed in does not")

    got_v = atomic_realizations(tables, values_v)
    if not got_v:
        return got_v
    got_c = atomic_realizations(tables, values_c)
    if not got_c:
        return got_c

    consistent = []
    for realization_v in got_v.realizations:
        for realization_c in got_c.realizations:
            if all((realization_v.theta_Zn[name].value + realization_c.theta_Zn[name].value
                    - int(theta_zn_h[name])) % realization_v.theta_Zn[name].modulus == 0
                   for name in zn_names):
                consistent.append((realization_v, realization_c))

    # NSIT where both sides have realizations and no pair adds up, and only there
    nsit = (not _fractional(values_v) and not _fractional(values_c)
            and bool(got_v.realizations) and bool(got_c.realizations)
            and bool(zn_names) and not consistent)
    same = []
    for got in (got_v, got_c):
        for realization in got.realizations:
            for pair in realization.stably_equivalent:
                if pair not in same:
                    same.append(pair)
    return Splitting(
        nsit=bool(nsit), valence=got_v, conduction=got_c, theta_Z=theta_z_h,
        theta_Zn={name: ModularValue(theta_zn_h[name], int(row.mod))
                  for row in tables._rows if row.mod is not None for name in (row.name,)},
        consistent_pairs=tuple(consistent), stably_equivalent=tuple(same),
        group=tables.group)


def splitting(realized, *, bands=None) -> "Splitting | Refusal":
    r"""Whether the bands asked for and the rest of a model's bands can both be atomic insulators.

    The paper's no-go criterion on a gapped splitting of a model's bands: the
    valence bands are the bands asked for, the conduction bands the rest.  The
    whole model's invariants are read off its orbitals (:func:`srsi_of_model`),
    the integer-valued ones of the valence bands off their symmetry data
    vector (:func:`srsi_of_bands`), and those of the conduction bands are the
    difference; :func:`nsit_criterion` decides.  The gap is taken as given.

    Parameters
    ----------
    realized : ..groups.RealizedGroup
        What :func:`..groups.realized_group` returned, ``r``.
    bands : int or sequence of int
        The valence bands: an ``int`` is the lowest *n*, a sequence those bands.
        Required.

    Returns
    -------
    Splitting or Refusal
        ``out.nsit`` holds whether the splitting is NSIT, which it is only
        where both sides have atomic realizations and no pair of them adds up
        to the whole model's invariants valued modulo n.  ``out.valence`` and
        ``out.conduction`` hold each side's realizations, empty where its
        invariants are fractional.  ``out.summary()`` prints it and what it
        rests on.  The record is truthy.  A ``Refusal`` in its place is falsy.

    Raises
    ------
    TypeError
        ``bands`` left out, or ``realized`` is not what
        :func:`..groups.realized_group` returned.

    Refusals
    --------
    ``srsi_scope``, ``srsi_table_absent``, ``srsi_data``, ``srsi_ebr_basis``
        from :func:`srsi_tables`: the group has no tables.
    ``wyckoff_orbit_content_incomplete``, ``wyckoff_orbit_not_localizable``, ``wyckoff_setting``
        from :func:`srsi_of_model`, for the orbitals.
    ``incomplete_content``, ``not_a_band_structure``, ...
        from :func:`..sewing.hsp_table` and :func:`srsi_of_bands`, for the bands.
    ``srsi_realizations_truncated``
        from :func:`nsit_criterion`, which lists each side at the default cap of
        :func:`atomic_realizations`.

    Examples
    --------
    >>> print(splitting(realized, bands=6).summary())       # doctest: +SKIP
    """

    if bands is None:
        raise TypeError("splitting(): pass bands=, the number of valence bands")
    if isinstance(realized, Refusal):
        return realized
    _model, group, _assignment, rep = _unpack_realized(realized, "splitting")
    gate = _spacegroup._lattice_not_descended(group)
    if gate is not None:
        return gate
    tables = srsi_tables(group)
    if not tables:
        return tables
    whole = srsi_of_model(realized)
    if not whole:
        return whole
    content = _sewing._hsp_table_rep(rep, bands=bands)
    if not content:
        return content
    valence_side = srsi_of_bands(content)
    if not valence_side:
        return valence_side
    theta_z_v = dict(valence_side.theta_Z)
    theta_z_c = {name: whole.theta_Z[name] - value for name, value in theta_z_v.items()}
    return nsit_criterion(tables, whole, theta_z_v, theta_z_c)
