r"""Stable real-space invariants: the tables, and their values on bands and models.

The tables are those of Hwang et al., *Stable real-space invariants and
topology beyond symmetry indicators*, Nat. Commun. **17**, 8564 (2026),
https://doi.org/10.1038/s41467-026-74844-w, for the grey space groups with
spin-orbit coupling and without.  They ship with the package.  They hold no
entry for the thirteen space groups whose one invariant is the band count.
A record holds the integer-valued stable real-space invariants (ZSRSIs) in
``theta_Z`` and those valued modulo n in ``theta_Zn``, where it has them, each
under its name in the tables, ``theta_1``, ``theta_2`` and so on.

- :func:`srsi_tables`: the tables of one group, an :class:`SRSITables`.  Its
  ``theta`` evaluates them on a site-symmetry vector, and its
  ``stably_equivalent`` holds the pairs of site irreps no invariant tells
  apart.
- :func:`srsi_of_bands`: the ZSRSIs of a band set, from its symmetry data
  vector.
- :func:`srsi_of_model`: every invariant of a model, from its orbitals, a row
  for each Wyckoff orbit.
- :func:`splitting`: the no-go criterion of Hwang et al. on a gapped splitting
  of a model's bands.  :func:`nsit_criterion` applies the criterion to
  invariants in hand, and :func:`atomic_realizations` returns every atomic
  realization of given ZSRSIs.
- :func:`srsi_status`: whether the tables are installed.  :data:`GATES` holds
  the refusals the calls here measure, each with what it measures.
- :func:`canonical_determinism`: whether the named indicators of Hwang et al.
  tell apart every class of a group's symmetry-indicator group.

The guide page "What do the stable real-space invariants say about these
bands?" (``docs/guide/srsi.md``) reads them on models.
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
    """Whether any value of ``theta_z``, invariants by name, is not a whole number."""
    return any(Fraction(value).denominator != 1 for value in theta_z.values())


def _fields(record, **override) -> dict:
    """The values ``record`` is built from, ``{name: value}``, those in ``override`` replaced."""
    out = {f.name: getattr(record, f.name) for f in fields(record) if f.init}
    out.update(override)
    return out


# ---------------------------------------------------------------------------
# a band subset -> its integer-valued invariants
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BandSRSIsNumerical:
    """A rational sum of elementary band representations, for a band set.

    The sum has the symmetry data vector of the band set, and it is one of
    many sums that do.  Held in :attr:`BandSRSIs.numerical`.  ``summary()``
    prints the sum.

    Attributes
    ----------
    p : dict
        The sum, ``{(site irrep, Wyckoff position): multiplicity}``, each
        elementary band representation written as the site irrep inducing it.
        Its multiplicities are rational, and the zeros are dropped.  Its keys
        are those :meth:`SRSITables.theta` takes, in the order of the group's
        list of elementary band representations.
    """

    p: dict

    def summary(self) -> str:
        """The sum on one line.

        Each term is written ``m (irrep)_w``, with ``m`` the multiplicity and
        ``w`` the Wyckoff letter.
        """
        return ("one rational combination of elementary band representations with this "
                "symmetry data vector, of many: "
                + _signed((value, f"({irrep})_{wyckoff.lstrip('0123456789')}")
                          for (irrep, wyckoff), value in self.p.items())
                + "   (.p)")


@dataclass(frozen=True)
class BandSRSIs:
    r"""The integer-valued stable real-space invariants (ZSRSIs) of a band set.

    The record :func:`srsi_of_bands` returns.  ``summary()`` prints the ZSRSIs,
    whether they are fractional, and the symmetry-indicator class.
    :meth:`line` prints the same on one line.  The record holds no stable
    real-space invariants valued modulo n, which the symmetry data vector does
    not fix.

    Attributes
    ----------
    theta_Z : dict
        The ZSRSIs, ``{name: value}``, as exact fractions in the order of the
        tables.
    indicator : .ebr.SymmetryIndicator
        The symmetry-indicator class of the same bands.
    group : .groups.GroupData
        The group the symmetry data vector was read in.
    numerical : BandSRSIsNumerical
        The rational sum of elementary band representations the ZSRSIs were
        computed from.
    """

    theta_Z: dict
    indicator: Any = field(repr=False)
    group: Any = field(repr=False)
    numerical: Any = field(repr=False)

    def summary(self) -> str:
        """The ZSRSIs, whether they are fractional, and the class, a line each."""
        lines = [f"{_group(self.group)}: the Z-valued stable real-space invariants of the "
                 "bands   (.group, .theta_Z)"]
        lines += [f"  {name:>9s} = {str(value):>8s}" for name, value in self.theta_Z.items()]
        lines.append("  fractional   (.theta_Z)" if _fractional(self.theta_Z)
                     else "  integral   (.theta_Z)")
        if self.indicator is not None:
            lines.append(_indicator_line(self.indicator, "their"))
        return "\n".join(lines)

    def line(self) -> str:
        """The record on one line, for a table of band sets.

        The line holds the ZSRSIs, whether they are fractional, and the
        symmetry-indicator class.  A trivial indicator group and a zero class
        are named as such.  For the two lowest bands of Fu, Kane and Mele's
        diamond with one bond detuned, ``theta_1 to theta_8 = (-1/4, 0, 1/2,
        1, 1/4, -1/2, -1/4, 1/4), fractional; class (0, 1) in Z2 x Z4, of
        order 4``.
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
    r"""The integer-valued stable real-space invariants (ZSRSIs) of a band set.

    The ZSRSIs are fixed by the symmetry data vector of the bands.  They are
    computed by writing the symmetry data vector as a rational sum of
    elementary band representations and applying to it the linear map
    :meth:`SRSITables.theta`.  The SRSIs valued modulo n are not fixed by the
    symmetry data vector and are not returned; :func:`srsi_of_model` computes
    them for a full model.

    Parameters
    ----------
    symmetry_data : .sewing.HSPTable or mapping
        The symmetry data vector.  Pass the table :func:`.sewing.hsp_table`
        returns, or ``{corep: multiplicity}`` written out by hand.
    realized : .groups.RealizedGroup, optional
        What :func:`.groups.realized_group` returned for the model the
        symmetry data vector was measured on.  Pass it with a vector written
        out by hand, and nothing with a table.

    Returns
    -------
    BandSRSIs or Refusal
        The ZSRSIs of the band set by name, the symmetry-indicator class of
        the same bands, and the rational sum of elementary band
        representations the ZSRSIs were computed from.

    Examples
    --------
    The six lowest bands of lin3g (:func:`.models.lin3g`):

    >>> print(srsi_of_bands(hsp_table(r, bands=6)).line())        # doctest: +SKIP
    theta_1 to theta_3 = (1, 1, 1), integral; trivial indicator group

    Refusals
    --------
    A refusal handed in as ``realized`` is returned as it is.

    ``srsi_scope`` / ``srsi_table_absent``
        the tables hold no stable real-space invariants of the group: it is
        not grey, or it is one of the thirteen space groups whose one
        invariant is the band count.
    ``srsi_data`` / ``srsi_ebr_basis``
        the tables of the stable real-space invariants are not installed, or
        were converted against another list of elementary band
        representations than the one installed.
    ``incomplete_content``
        the table holds no little-group representations at some high-symmetry
        point, for example where the bands split a degenerate multiplet.
    ``not_a_band_structure``
        the vector fails the compatibility relations of the group.
    ``indicator_not_evaluable``
        the bands are a layer's, and some of their little-group
        representations at ``k_z = 1/2`` could not be read.  For each of these,
        no little-group representation in the layer's plane occurs as often as
        it does in every elementary band representation.  For a table, their
        measurement on the layer's decoupled stack stopped.  For a vector
        written out by hand, the Wyckoff positions of the tables cannot be
        read in the group's cell.
    ``unknown_label``
        the vector names a little-group representation that the group's
        elementary band representations do not carry.
    ``non_integer_multiplicity`` / ``negative_multiplicity``
        an entry of the vector is not a nonnegative integer.
    ``ebr_tables`` / ``irrep_tables`` / ``ebr_convention`` / ``tables_not_ready``
        the tables installed hold no elementary band representations of this
        group that can be read in the model's cell.

    Raises
    ------
    TypeError
        A table with ``realized`` beside it, a vector written out by hand
        without it, a ``realized`` that is not what
        :func:`.groups.realized_group` returned, or a ``symmetry_data`` that is
        neither a table nor a mapping.

    See Also
    --------
    srsi_of_model : the stable real-space invariants of a model's orbitals,
        those valued modulo n among them.
    atomic_realizations : every atomic realization of given ZSRSIs.
    """
    source = _ebr._read("srsi_of_bands", symmetry_data, realized)
    if isinstance(source, Refusal):
        return source
    return _band_srsis(source, symmetry_data)


def _srsi_of_bands(symmetry_data, group, *, rep=None) -> BandSRSIs | Refusal:
    """What :func:`srsi_of_bands` returns, for a vector read in a group given directly.

    ``symmetry_data`` is the table :func:`.sewing.hsp_table` returns, read in
    the group and representation it carries, or ``{corep: multiplicity}``,
    read in ``group``.  ``rep``, beside a vector, gives the site irreps and,
    on a layer, the orbitals' heights.  Returns the :class:`BandSRSIs`, or the
    refusal, as :func:`srsi_of_bands` does.
    """
    return _band_srsis(_ebr._source_of(symmetry_data, group, rep), symmetry_data)


def _band_srsis(source, symmetry_data) -> BandSRSIs | Refusal:
    """What :func:`srsi_of_bands` returns for ``symmetry_data`` read in ``source``.

    ``source`` holds the group and the representation the symmetry data vector
    is read in, as :func:`.ebr._read` or :func:`.ebr._source_of` gives it.
    """
    group = source.group
    gate = _spacegroup._lattice_not_descended(group)
    if gate is not None:
        return gate
    tables = srsi_tables(group)          # read from the measured group, which
    if not tables:                       # the indicator below reads as well
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
            f"#{group.number_str}: the rational sum of elementary band representations "
            "found for the symmetry data vector does not reproduce it. "
            "Report it, with the model and the call.")
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
    """One decomposition of a Wyckoff orbit's symmetry data vector at one position.

    The decomposition is a sum of elementary band representations at one
    Wyckoff position, each written as the site irrep inducing it.  Each row
    :func:`srsi_of_model` returns holds its decompositions in
    ``numerical.localizations``, and they all have the same stable real-space
    invariants.  A localization is a sum of terms, as
    :class:`.symmetry.BandRepresentation` is.  Iterating, indexing and
    ``len()`` go over :attr:`terms`.  Two sums with the same terms compare
    equal.

    Attributes
    ----------
    terms : tuple of .groups.Term
        If nonempty, the elementary band representations, each a site irrep
        at :attr:`wyckoff` with its multiplicity.  Empty where the row's
        stable real-space invariants are read from its own site irreps,
        ``numerical.composite``, which induce a band representation that is
        not elementary.
    label : str
        The sum on one line, ``"1 (-E1g)_a + 1 (-E1u)_a"``.  Each term is
        written ``m (irrep)_w``, with ``m`` the multiplicity and ``w`` the
        Wyckoff letter.
    wyckoff : str
        The Wyckoff position of the terms, ``"3a"``.
    theta_Z : dict
        The integer-valued stable real-space invariants of the sum,
        ``{name: value}``.
    theta_Zn : dict
        If nonempty, the stable real-space invariants valued modulo n of the
        sum, ``{name: ModularValue}``.  Empty where the group has none.
    """

    wyckoff: str = field(default="", compare=False)
    theta_Z: dict = field(default_factory=dict, compare=False)
    theta_Zn: dict = field(default_factory=dict, compare=False)


@dataclass(frozen=True, repr=False)
class WyckoffOrbitSRSIsNumerical(WyckoffOrbitNumerical):
    """The decompositions behind one Wyckoff orbit's stable real-space invariants.

    Held in ``WyckoffOrbitSRSIs.numerical``.  It holds every field of the
    ``numerical`` of the same orbitals' row of the atomic-limit table
    ``r.wyckoff_orbits``, with ``r`` what :func:`.groups.realized_group`
    returned.  It also holds the fields below.

    Attributes
    ----------
    localizations : tuple of Localization
        Every decomposition of :attr:`symmetry_data` into elementary band
        representations at one Wyckoff position the orbitals can slide onto.
        For a row with :attr:`composite`, one entry with no terms.
    chosen : Localization
        The first of :attr:`localizations`.  The row's stable real-space
        invariants are read from it.
    stably_equivalent : tuple of StablePair
        If nonempty, the pairs of site irreps that no stable real-space
        invariant tells apart and whose two members are both used among the
        localizations at a position that has several.
    composite : tuple of .groups.Term or None
        If not ``None``, the row's site irreps at its own position, which
        induce a band representation that is not elementary but a sum of
        elementary ones at other positions.  It is set where no sum of
        elementary band representations at any one position the orbitals can
        slide onto carries the row's symmetry data vector.  The row's stable
        real-space invariants are read from these site irreps.
    agree : bool
        Whether every localization gives the same stable real-space
        invariants, which holds in every record returned.
    symmetry_data : dict
        The symmetry data vector of the row's own orbitals,
        ``{corep: multiplicity}``, measured at :attr:`kpoints`.
    kpoints : tuple of str
        The high-symmetry points at which :attr:`symmetry_data` was measured.
        A point whose little group has no unitary operation besides the
        identity is left out.
    n_free : int
        The number of free coordinates of the row's Wyckoff position.
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

    A row of the ``wyckoff_orbits`` that :func:`srsi_of_model` returns.  It
    holds every field of the same orbitals' row of the atomic-limit table
    ``r.wyckoff_orbits``, with ``r`` what :func:`.groups.realized_group`
    returned, and with the same values.  Among them are ``wyckoff``,
    ``site_symmetry``, ``orbitals`` and ``terms``.  It also holds the fields
    below.

    Attributes
    ----------
    theta_Z : dict
        The row's integer-valued stable real-space invariants,
        ``{name: value}``.
    theta_Zn : dict
        If nonempty, the row's stable real-space invariants valued modulo n,
        ``{name: ModularValue}``.  Empty where the group has none.
    slides_onto : tuple of str
        The Wyckoff positions the orbitals can slide onto, and at which the
        tables list elementary band representations, ``("3a", "3b")``.  The
        orbitals slide over the points the free coordinates of their Wyckoff
        position sweep, and so onto the positions of higher symmetry among
        them.  The row's own position is among those returned where the
        tables list one there.
    """

    theta_Z: dict = field(default_factory=dict)
    theta_Zn: dict = field(default_factory=dict)
    slides_onto: tuple = ()


def _slides(row) -> bool:
    """Whether ``row`` is read at positions its orbitals slide onto, not at its own.

    ``row`` is a :class:`WyckoffOrbitSRSIs`.  True where it is not read through
    composite site irreps and its own position is not in ``row.slides_onto``.
    """
    return row.numerical.composite is None and row.wyckoff not in row.slides_onto


@dataclass(frozen=True)
class ModelSRSIs:
    r"""The stable real-space invariants of a model, a row for each Wyckoff orbit.

    The record :func:`srsi_of_model` returns.  ``summary()`` prints the
    model's stable real-space invariants (SRSIs), then each Wyckoff orbit's.
    The model's orbitals are an atomic configuration, so its integer-valued
    SRSIs are integers.  Its SRSIs valued modulo n are returned too.

    Attributes
    ----------
    theta_Z : dict
        The integer-valued SRSIs, ``{name: value}``, summed over the rows.
        They equal those :func:`srsi_of_bands` returns for all the model's
        bands.
    theta_Zn : dict
        If nonempty, the SRSIs valued modulo n, ``{name: ModularValue}``,
        summed over the rows modulo each n.  Empty where the group has none.
    group : .groups.GroupData
        The group the SRSIs were read in, the one
        :func:`.groups.realized_group` returned.
    wyckoff_orbits : tuple of WyckoffOrbitSRSIs
        A row for each Wyckoff orbit, in the order of the rows of the
        atomic-limit table that :func:`.groups.realized_group` returned.
    """

    theta_Z: dict
    theta_Zn: dict
    group: Any = field(repr=False)
    wyckoff_orbits: tuple = ()

    def summary(self) -> str:
        """The model's SRSIs, then each Wyckoff orbit's, a line each."""
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
    """The model, group, orbital assignment and representation of ``realized``.

    ``realized`` is what :func:`.groups.realized_group` returned, and ``call``
    the name of the call it was handed to.  Returns ``(model, group,
    assignment, rep)``.  Raises ``TypeError``, its message opening on
    ``call``, for anything else.
    """
    from ._realized import RealizedGroup

    if isinstance(realized, RealizedGroup):
        return (realized.model, realized.group, realized._assignment, realized.rep)
    raise TypeError(f"{call}(): realized= is of type {type(realized).__name__}; pass "
                    "what realized_group returns, whole." if isinstance(realized, (tuple, list))
                    else f"{call}(): realized= is of type {type(realized).__name__}; pass "
                    "what realized_group returns.")


def _list_positions(data) -> dict:
    """The positions of the elementary band representations, by Wyckoff letter.

    ``data`` is the group's list of elementary band representations, as
    :func:`._tables._ebr_data` returns it.  Returns ``{letter: position}``,
    each position with its multiplicity, ``{"a": "3a"}``.
    """
    out = {}
    for _name, position in data["ebrs"]:
        match = _POSITION.match(str(position))
        out.setdefault(match.group(2), match.group(1) + match.group(2))
    return out

def _orbit_content(group, rep, rows):
    r"""The symmetry data vector of one block of states, measured.

    :func:`_orbit_contents` of ``group`` and ``rep`` for the one block
    ``rows``, state indices ``orbital * nspin + spin``.  Returns its
    ``(counts, points, refusals)``.
    """
    return _orbit_contents(group, rep, [rows])[0]

def _orbit_contents(group, rep, blocks):
    r"""The symmetry data vector of each block of states, measured.

    ``blocks`` holds lists of state indices, ``orbital * nspin + spin``, and
    ``rep`` is the representation of ``group`` on the states.  At each
    tabulated high-symmetry point whose little group has a unitary operation
    besides the identity, a block's character under each unitary little-group
    operation is the trace of that operation's sewing operator over the
    block's rows.  :func:`._tables._identify_from_characters` turns the
    characters into the multiplicities of the little-group representations.
    The orbitals' centers are assumed to sit where the operations carry them,
    to the tolerance of :func:`._matrices.bloch_operator`, which raises
    ``ValueError`` otherwise.

    Returns one ``(counts, points, refusals)`` per block: ``{corep:
    multiplicity}`` over the points read; those points' names; and
    ``{point: why}`` for each point not read, ``why`` a refusal, or a string
    where the little group is trivial.  Where the group's high-symmetry
    points cannot be read, ``counts`` and ``points`` are ``None`` and
    ``refusals`` is that refusal.
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
    """The decompositions of a block's symmetry data vector at one Wyckoff letter.

    :func:`_solve_at_letters` at the one ``letter``.  ``data``, ``counts``,
    ``points``, ``bound``, ``cap`` and ``bands``, and what is returned, are
    as it has them.
    """
    return _solve_at_letters(data, counts, points, (letter,), bound, cap, bands)

def _solve_at_letters(data, counts, points, letters, bound, cap=64, bands=None):
    """The nonnegative-integer decompositions of a block's symmetry data vector.

    ``data`` is the group's list of elementary band representations
    (:func:`._tables._ebr_data`), ``counts`` the block's symmetry data vector,
    ``{corep: multiplicity}``, and ``points`` the high-symmetry points it was
    read at.  The matrix of elementary band representations is restricted to
    the columns at the Wyckoff letters ``letters`` and to the rows at
    ``points``, so that a layer's rows out of its plane are not compared.
    With ``bands``, the block's number of bands is one more equation.  It is
    the only one where no point is read, as in a group whose little groups
    hold no unitary operation besides the identity.  Each multiplicity runs
    from 0 to ``bound``.  The enumeration stops at ``cap`` decompositions, and
    whether it stopped is not returned.

    Returns ``(status, solutions, columns, keep)``: the solver's status, such
    as ``"INFEASIBLE"``; the decompositions, each the multiplicities over
    ``columns``, the fewest elementary band representations first; the
    indices of those columns; and the indices of the rows kept.  The status is
    ``"INFEASIBLE"``, with no decomposition, where no column is at
    ``letters``, or no row is kept and ``bands`` is ``None``.
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
    """The stable real-space invariants of a model, a row for each Wyckoff orbit.

    The symmetry data vector of each Wyckoff orbit is measured on its own
    orbitals, from the trace of each little-group operation over them at each
    high-symmetry point.  It is decomposed into elementary band
    representations at each Wyckoff position that the orbitals can slide onto
    by varying the free coordinates of their own, and at which the tables
    list elementary band representations.  Their own position is among these
    where the tables list one there.  A row of a spinless model whose
    symmetry data vector has no such decomposition is given the stable
    real-space invariants of its own site irreps, each identified with one
    the tables list at its position, in ``SRSITables.site_irreps``.  The
    stable real-space invariants (SRSIs) of a row are those of its
    decompositions, all of which agree, and the model's are the sum over the
    rows.  The orbitals are a known configuration, so the SRSIs valued modulo
    n are returned with the integer-valued ones.

    Parameters
    ----------
    realized : .groups.RealizedGroup
        What :func:`.groups.realized_group` returned for the model.

    Returns
    -------
    ModelSRSIs or Refusal
        The model's integer-valued SRSIs and its SRSIs valued modulo n, by
        name, and a row for each row of ``realized.wyckoff_orbits`` with its
        own SRSIs and its decompositions.

    Examples
    --------
    lin3g (:func:`.models.lin3g`), with three Wyckoff orbits:

    >>> print(srsi_of_model(r).summary())                     # doctest: +SKIP
    P4mm1' #99.164, double-valued: stable real-space invariants   (.group)
      the whole model: theta_1 to theta_3 = (2, 2, 0); theta_4, theta_5 = (0, 1) mod 2   (.theta_Z, .theta_Zn)
      each Wyckoff orbit, its site irreps written m (irrep)_w, with theta_1 to theta_3 and theta_4, theta_5 mod 2   (.wyckoff_orbits)
        1a (4mm1'): orbital 0, 1 (-E1)_a: (0, 1, 0); (0, 0) mod 2
        1a (4mm1'): orbital 1, 1 (-E2)_a: (1, 0, 0); (0, 0) mod 2
        2c (mm21'): orbitals 2-3, 1 (-E)_c: (1, 1, 0); (0, 1) mod 2

    Refusals
    --------
    A refusal handed in as ``realized`` is returned as it is.

    ``srsi_scope`` / ``srsi_table_absent``
        the tables hold no SRSIs of the group: it is not grey, or it is one of
        the thirteen space groups whose one invariant is the band count.
    ``srsi_data`` / ``srsi_ebr_basis``
        the tables of the SRSIs are not installed, or were converted against
        another list of elementary band representations than the one
        installed.
    ``ebr_tables`` / ``irrep_tables`` / ``ebr_convention`` / ``tables_not_ready``
        the tables installed hold no elementary band representations of this
        group that can be read in the model's cell.
    ``wyckoff_orbit_content_incomplete``
        the symmetry data vector of a Wyckoff orbit could not be read at some
        high-symmetry point.
    ``wyckoff_orbit_not_localizable``
        no sum of elementary band representations at any one position the
        orbitals of a Wyckoff orbit can slide onto carries its symmetry data
        vector.  The model is spinful, or one of the orbit's site irreps could
        not be identified with a site irrep the tables list at its position.
    ``wyckoff_setting`` / ``wyckoff_unresolved``
        the Wyckoff position of a Wyckoff orbit read from its centers differs
        from the one at which its symmetry data vector is a sum of elementary
        band representations, or the positions could not be read in the
        setting of the tables.

    Raises
    ------
    TypeError
        ``realized`` is not what :func:`.groups.realized_group` returned.

    See Also
    --------
    srsi_of_bands : the integer-valued SRSIs of a band set, from its symmetry
        data vector.
    nsit_criterion : the no-go criterion on a gapped splitting, from the SRSIs
        of all the bands and of each side.
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
                f"the symmetry data vector of the Wyckoff orbit of orbitals {list(orbitals)} "
                f"at {row.wyckoff} of #{group.number_str} could not be read at {name}. "
                "The refusal there, in context['refusal'], says: "
                f"{getattr(why, 'reason', why)}",
                context={"orbit": index, "orbitals": orbitals, "kpname": name,
                         "refusal": why, "identified": points,
                         "refusals": tuple(sorted(refusals, key=_tables.point_order))})

        geometry = [sites[int(s)] for s in row._sites]
        letters = {site.letter for site in geometry}
        if len(letters) != 1:                     # pragma: no cover - guard
            raise InternalError(
                f"#{group.number_str}: the sites of the Wyckoff orbit of orbitals "
                f"{list(orbitals)} are at different Wyckoff letters {sorted(letters)} in "
                "the setting of the tables. Report it, with the model and the call.")
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
                    + ("the tables list elementary band representations at no position "
                       "its orbitals can slide onto"
                       if not tried else
                       f"its symmetry data vector {counts} is not a sum of elementary band "
                       "representations at any one position its orbitals can slide onto. "
                       "Its orbitals can slide onto "
                       + _and([positions[letter] for letter in tried]))
                    + ". Its stable real-space invariants could not be read from its site "
                    "irreps either. Check the positions of the orbitals and their site "
                    "irreps.",
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
                f"#{group.number_str}: the decompositions of the Wyckoff orbit of orbitals "
                f"{list(orbitals)} give {len(values)} different sets of stable real-space "
                "invariants ("
                + str([(entry.wyckoff, tuple(str(v) for v in entry.theta_Z.values()))
                       for entry in localizations])
                + "). Report it, with the model and the call.")
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
    """The Wyckoff letters at which a block's symmetry data vector decomposes.

    ``data``, ``counts``, ``points`` and ``bound`` are as
    :func:`_solve_at_letters` takes them.  Each letter of the list of
    elementary band representations is tried alone for one decomposition.
    Returns the letters that have one, in alphabetical order, or ``()``.
    :func:`srsi_of_model` asks it of an orbit with no decomposition at the
    letters on the closure of its own position, to tell ``wyckoff_setting``
    from ``wyckoff_orbit_not_localizable``.
    """
    letters = sorted({_POSITION.match(str(position)).group(2)
                      for _name, position in data["ebrs"]})
    return tuple(letter for letter in letters
                 if _solve_at_letter(data, counts, points, letter, bound, cap=1)[1])

def _route_disagreement(group, index, orbitals, here, elsewhere, counts, positions) -> Refusal:
    """The refusal where an orbit's geometry and its symmetry data vector disagree.

    The refusal is ``wyckoff_setting``.  ``group`` is the group, ``index`` the
    orbit's row in the atomic-limit table, ``orbitals`` its orbitals, and
    ``here`` the Wyckoff position of its first site in the tables' setting,
    as :func:`._tables._sites_in_tables_setting` gives it.  ``elsewhere``
    holds the Wyckoff letters at which its symmetry data vector ``counts``
    decomposes (:func:`_letters_that_do_fit`), and ``positions`` maps each
    letter to its position (:func:`_list_positions`).  Returns the refusal,
    which names the position of the centers and the positions the symmetry
    data vector decomposes at.
    """
    where = f"{here.multiplicity}{here.letter}"
    return Refusal(
        "wyckoff_setting",
        f"the Wyckoff orbit of orbitals {list(orbitals)} of #{group.number_str} is at "
        f"{where} by its centers, read in the setting of the tables, and its symmetry data "
        "vector is a sum of elementary band representations at "
        f"{_and([positions.get(letter, letter) for letter in elsewhere])} instead"
        + (f", and at none of "
           f"{_and([positions.get(letter, letter) for letter in here.closure])}, "
           "the positions its orbitals can slide onto"
           if tuple(here.closure) != (here.letter,) else f", not at {where}")
        + ". Report it, with the model and the call.",
        context={"orbit": index, "orbitals": orbitals,
                 "geometry_letter": here.letter, "content_letters": tuple(elsewhere),
                 "closure": here.closure, "reference": here.reference,
                 "content": dict(counts), "number_str": str(group.number_str)})

def _band_representations(rep) -> tuple:
    r"""The band representation each Wyckoff orbit of ``rep`` induces.

    ``rep`` is a representation of the group on the orbitals, carrying the
    site irreps of the atomic-limit table.  Each orbit's symmetry data vector
    is measured on its states (:func:`_orbit_contents`) and decomposed at the
    positions its orbitals can slide onto, its own first
    (:func:`_sum_at_first`).  Where none carries it, a site irrep whose band
    representation is not elementary is written as the sum, among those
    carrying the symmetry data vector, whose stable real-space invariants
    are those of its site irreps (:func:`_composite_localization`).  Without
    those, the positions on the closure of each variety through the site are
    tried, the most special variety first, and last the sum the vector has
    where it has only one.  The alternatives are the other sums with the
    same symmetry data vector and the same stable real-space invariants,
    none where the group has no tabulated invariants.  On a free-standing
    layer only the band representations of the positions within it are read
    (:func:`.ebr._layer_reading`).  Nothing refuses: an orbit that cannot be
    read has no terms, and its ``note`` says why.

    Returns
    -------
    tuple of .symmetry._OrbitBandRepresentation
        One per Wyckoff orbit of orbitals, in the order of the orbits' first
        orbitals (:func:`.symmetry._wyckoff_orbits_of`), each a sum of
        elementary band representations.  Empty where ``rep`` carries no site
        irreps, where the tables do not reach the cell, and where ``rep``
        leaves out some unitary operation of its group, the tables being the
        whole group's.  Where the orbitals' centers miss the operations by
        more than the tolerance of the Bloch operators, every orbit has no
        terms.
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
    # The Bloch operators check the orbitals' centers against the operations'
    # action on them, to _BLOCH_TOL.  The centers of a representation the
    # package builds sit at their symmetric positions to rounding.  A
    # representation built by hand on centers that miss by more reads no
    # symmetry data vector here, and each orbit's note says why.

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
        # the orbit's own sum: its symmetry data vector decomposed at the
        # positions its orbitals can be moved onto, one at a time, its own
        # first, as srsi_of_model localizes it; this needs no stable
        # real-space invariants
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
        # through its site, the most special first, since a site irrep induced
        # from that variety's site group is moved along it
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
        # every sum with this symmetry data vector, over all the elementary band
        # representations, or over a free-standing layer's own: those its
        # invariants do not tell from the orbit's own sum are its alternatives,
        # and a composite orbit's own sum is among them
        every = ()
        if tables is not None or own is None:
            restriction = _ebr.in_plane_restriction(group, kpoints=points,
                                                    require_exact=False, _layer=layer)
            if not isinstance(restriction, Refusal):
                matrix, y, bound = _ebr._content_system(data, counts, restriction.retained)
                # the band count, one more equation, the only one where no
                # point was read, and there each multiplicity's bound
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
    """A block's symmetry data vector as a sum at the first of ``letters`` that carries it.

    ``data``, ``counts`` and ``points`` are as :func:`_solve_at_letters` takes
    them, and ``ebrs`` holds the list's ``(name, position)`` pairs.
    ``bands``, the block's number of bands, is the extra equation and each
    multiplicity's bound.  ``letters`` are Wyckoff letters, in the order
    tried.  Where several sums at that letter carry the vector, the first
    whose site irreps are all in ``named``, those of the orbit's own rows, is
    taken, and the first of all otherwise.  Returns the sum as the
    multiplicities over ``ebrs``, or ``None`` where no letter carries it.
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
    r"""The site irreps of an orbit whose band representation is not elementary.

    Some site irreps at a maximal position, such as ``E`` at ``3c`` of
    ``P-43m``, induce a sum of elementary band representations at other
    positions, so that no elementary band representation at their own
    position carries them.  ``tables`` are the tables of the stable real-space
    invariants of ``group``, and ``assignment`` and ``rep`` the orbitals'
    assignment and the representation whose site irreps are read.  ``orbit``
    holds the orbit's sites, and ``here`` the Wyckoff position of the first
    in the tables' setting.  Each multiplet at that site, of ``orbitals``
    alone where they are given, is matched to the tables' site irreps at its
    letter by name, and otherwise to the one composite site irrep whose
    number of bands its induction gives.  Returns ``{"irreps": {irrep:
    multiplicity}, "site": site}``, ``site`` the index of the orbit's first
    site.  Returns ``None`` where a multiplet is matched to none, where
    ``rep`` carries no site irreps, and for double-valued tables, which hold
    no composite site irrep.
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
        # holds as many copies as its orbitals divided by the irrep's dimension
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
    r"""One atomic realization: site irreps at maximal Wyckoff positions.

    An atomic realization is a set of site irreps at maximal Wyckoff
    positions, with nonnegative integer multiplicities, whose integer-valued
    stable real-space invariants are those asked for.  Each entry of
    :attr:`Realizations.realizations` is one.  A realization is a sum of
    terms, as :class:`.symmetry.BandRepresentation` is.  Iterating, indexing
    and ``len()`` go over :attr:`terms`.  Two sums with the same terms compare
    equal, whichever call returned them, :func:`.ebr.decompose` among them.

    Attributes
    ----------
    terms : tuple of .groups.Term
        The site irreps, each at its maximal Wyckoff position with its
        multiplicity.
    label : str
        The sum on one line, ``"1 (-E2)_b + 1 (-E)_c"``.  Each term is
        written ``m (irrep)_w``, with ``m`` the multiplicity and ``w`` the
        Wyckoff letter.
    theta_Zn : dict
        If nonempty, the stable real-space invariants valued modulo n of the
        realization, ``{name: ModularValue}``.  Empty where the group has
        none.  Two realizations of one symmetry data vector can differ in
        them.
    stably_equivalent : tuple of StablePair
        If nonempty, the pairs of site irreps that no stable real-space
        invariant tells apart and of which the realization holds at least one
        member.
    """

    theta_Zn: dict = field(default_factory=dict, compare=False)
    stably_equivalent: tuple = field(default=(), compare=False)


@dataclass(frozen=True)
class RealizationsNumerical:
    """The bound and the cap of an enumeration of atomic realizations.

    Held in :attr:`Realizations.numerical`.

    Attributes
    ----------
    bound : int
        The largest multiplicity of one site irrep searched: the most that the
        band count allows, so that no realization is missed.  ``0`` where
        nothing was enumerated.
    cap : int
        The ``cap=`` of :func:`atomic_realizations`, the largest number of
        realizations listed.  The call refuses where there are more.
    """

    bound: int
    cap: int


@dataclass(frozen=True)
class Realizations:
    r"""Every atomic realization of given integer-valued stable real-space invariants.

    The record :func:`atomic_realizations` returns, and each side of a
    :class:`Splitting`.  ``summary()`` prints the integer-valued stable
    real-space invariants (ZSRSIs), then every atomic realization with its
    stable real-space invariants valued modulo n.  The list of realizations
    is complete.  Where it is empty and the ZSRSIs are integers, no atomic
    insulator has these ZSRSIs.  The symmetry data vector of a band set with
    them is then an integer combination of elementary band representations
    and no nonnegative one, as a fragile set's is.  The record is truthy even
    where the list is empty.

    Attributes
    ----------
    realizations : tuple of AtomicRealization
        If nonempty, every atomic realization of :attr:`theta_Z`.  Empty where
        :attr:`theta_Z` is fractional, and where no atomic insulator has it.
    theta_Z : dict
        The ZSRSIs the realizations have, ``{name: value}``, as exact
        fractions.
    group : .groups.GroupData
        The group of the tables, ``tables.group``.
    numerical : RealizationsNumerical
        The bound and the cap of the enumeration.
    """

    realizations: tuple
    theta_Z: dict
    group: Any = field(repr=False)
    numerical: Any = field(repr=False)
    _status: str = field(default="", repr=False, compare=False)

    def summary(self) -> str:
        """The ZSRSIs, then every realization, a line each."""
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
    r"""The no-go test of a gapped splitting of bands.

    The record :func:`splitting` and :func:`nsit_criterion` return.  It says
    whether the splitting is non-symmetry-indicated topological (NSIT).  The
    valence bands are one set of the splitting and the conduction bands the
    other.  ``summary()`` prints whether the splitting is NSIT and why, then
    the atomic realizations of each side.  The record is truthy even where
    ``nsit`` is ``False``.

    Attributes
    ----------
    nsit : bool
        Whether the splitting is NSIT.  It is ``True`` exactly where both
        sides have atomic realizations, the group has stable real-space
        invariants (SRSIs) valued modulo n, and no pair of a valence and a
        conduction realization has SRSIs valued modulo n adding up to those
        of all the bands.  The two sides of such a splitting cannot both be
        atomic insulators.
    valence : Realizations
        Every atomic realization of the valence bands, as
        :func:`atomic_realizations` returns them.  Its list of realizations
        is empty where the valence bands' integer-valued SRSIs are
        fractional, and where no atomic insulator has them.
    conduction : Realizations
        Every atomic realization of the conduction bands, as
        :func:`atomic_realizations` returns them.  Its list of realizations
        is empty where the conduction bands' integer-valued SRSIs are
        fractional, and where no atomic insulator has them.
    theta_Z : dict
        The integer-valued SRSIs of all the bands, ``{name: value}``: the
        valence bands' plus the conduction bands'.
    theta_Zn : dict
        If nonempty, the SRSIs valued modulo n of all the bands,
        ``{name: ModularValue}``.  Empty where the group has none.
    consistent_pairs : tuple
        If nonempty, each pair ``(valence, conduction)`` of atomic
        realizations whose SRSIs valued modulo n add up to those of all the
        bands, modulo each n.
    stably_equivalent : tuple of StablePair
        If nonempty, the pairs of site irreps that no SRSI tells apart and of
        which some realization of either side holds at least one member.
    group : .groups.GroupData
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
        """The clause of the summary's first line, and the fields it rests on.

        Returns ``(clause, fields)``: whether the splitting is NSIT and why, in
        words, and the fields that say so as the summary points at them,
        ``"(.group, .nsit)"``.
        """
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
        """Whether the splitting is NSIT and why, then each side, a line each."""
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
    """The invariants valued modulo n of site irreps at maximal Wyckoff positions.

    ``tables`` are the tables of the group, and ``p`` holds an integer
    multiplicity for each of the tables' maximal site irreps, in the order of
    ``tables._maximal_keys``.  Returns ``{name: ModularValue}``, each value
    reduced modulo its n.
    """
    columns = _maximal_basis(int(tables._sg), bool(tables._spinor))[1]
    out = {}
    for r, row in enumerate(tables._rows):
        if row.mod is not None:
            n = int(row.mod)
            out[row.name] = ModularValue(
                sum(int(columns[j][r]) * int(p[j]) for j in range(len(p)) if p[j]) % n, n)
    return out


def _by_name(call, argument, tables, theta_z) -> dict:
    """Integer-valued invariants handed in by name, as exact fractions in the tables' order.

    ``theta_z`` is what the call named ``call`` was handed as ``argument``,
    which the messages name as written, ``"theta_Z="``, and ``tables`` the
    tables whose integer-valued invariants it names, each once.  Returns
    ``{name: Fraction}``.  Raises ``TypeError`` where ``theta_z`` is not a
    mapping, and ``ValueError`` where its names are not the tables'
    integer-valued invariants, each message opening on ``call``.
    """
    names = [tables._rows[i].name for i in tables._z_rows]
    if not isinstance(theta_z, Mapping):
        raise TypeError(f"{call}(): {argument} is of type {type(theta_z).__name__}; pass "
                        "a mapping from each integer-valued invariant's name to its value, "
                        "such as the theta_Z that srsi_of_bands or srsi_of_model returns.")
    if len(theta_z) != len(names) or set(theta_z) != set(names):
        raise ValueError(f"{call}(): {argument} holds "
                         f"{', '.join(str(k) for k in theta_z) or 'nothing'}; "
                         f"#{tables._number_str} has the integer-valued invariants "
                         f"{_span(names)}; pass each of them once.")
    return {name: Fraction(theta_z[name]) for name in names}


def atomic_realizations(tables, theta_Z, *, cap=256) -> "Realizations | Refusal":
    r"""Every atomic realization of given integer-valued stable real-space invariants.

    An atomic realization is a set of site irreps at maximal Wyckoff
    positions, with nonnegative integer multiplicities.  Those returned are
    every one whose integer-valued stable real-space invariants (ZSRSIs) are
    ``theta_Z``.  They are found by enumerating the nonnegative integer
    solutions of the linear equations that set the ZSRSIs to ``theta_Z`` and
    the band count to the one ``theta_Z`` fixes.  Maximal positions suffice,
    since an atomic insulator moves onto them with its stable real-space
    invariants unchanged.  The list of realizations is complete.  All the
    realizations returned have one symmetry data vector, since two with the
    same ZSRSIs have the same one.  Each realization carries its own stable
    real-space invariants valued modulo n.  Fractional ZSRSIs have no
    realization.

    Parameters
    ----------
    tables : SRSITables
        The tables of the group, what :func:`srsi_tables` returns.
    theta_Z : mapping
        The ZSRSIs, ``{name: value}``, with one entry for each of the tables'
        integer-valued invariants.  Pass the ``theta_Z`` of what
        :func:`srsi_of_bands` returns.
    cap : int, default 256
        The largest number of realizations listed.

    Returns
    -------
    Realizations or Refusal
        Every atomic realization of ``theta_Z``, each with its stable
        real-space invariants valued modulo n.  The list is empty where
        ``theta_Z`` is fractional, and where no atomic insulator has these
        ZSRSIs.  The record is truthy even where the list is empty.

    Examples
    --------
    The six lowest bands of lin3g (:func:`.models.lin3g`), with three
    realizations:

    >>> b = srsi_of_bands(hsp_table(r, bands=6))                  # doctest: +SKIP
    >>> print(atomic_realizations(srsi_tables(r.group), b.theta_Z).summary())  # doctest: +SKIP
    P4mm1' #99.164, double-valued: 3 atomic realizations with Z-valued stable real-space invariants theta_1 to theta_3 = (1, 1, 1)   (.group, .realizations, .theta_Z)
      each realization, with (theta_4, theta_5) mod 2   (.label, .theta_Zn of each)
        1 (-E2)_b + 1 (-E)_c   (1, 1)
        1 (-E1)_b + 2 (-E2)_b   (0, 0)
        1 (-E1)_a + 1 (-E2)_a + 1 (-E2)_b   (1, 0)

    Refusals
    --------
    A refusal handed in as ``tables`` is returned as it is.

    ``srsi_realizations_truncated``
        the enumeration found more than ``cap`` realizations.  A ``cap=`` at
        or above their number lists them all.

    Raises
    ------
    TypeError
        ``theta_Z`` is not a mapping.
    ValueError
        ``theta_Z`` does not hold one entry for each of the tables'
        integer-valued invariants, or ``cap`` is not a positive whole number.

    See Also
    --------
    .ebr.decompose : the sums of elementary band representations that carry a
        symmetry data vector.
    """
    _ebr._require_cap("atomic_realizations", cap)
    if isinstance(tables, Refusal):
        return tables
    values = _by_name("atomic_realizations", "theta_Z=", tables, theta_Z)
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
            f"{_invariants_text(values)} at {total} bands found more than the cap of "
            f"{cap}. atomic_realizations(tables, theta_Z, cap=...) with a cap at or above "
            "their number "
            "lists them all, theta_Z being context['theta_Z'].",
            value=float(int(cap) + 1), threshold=float(cap),
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
    r"""The no-go test of a gapped splitting, from its invariants.

    The record returned says whether the splitting is non-symmetry-indicated
    topological (NSIT).  The criterion is that of Hwang et al., Nat. Commun.
    17, 8564 (2026).  All the bands together are an atomic insulator, whose
    stable real-space invariants (SRSIs) are ``theta_total``.  The valence
    and the conduction bands split them, and their integer-valued SRSIs
    (ZSRSIs) are ``theta_Z_v`` and ``theta_Z_c``.  Where both sides are
    atomic insulators too, some atomic realization of each has SRSIs valued
    modulo n that add up to those of all the bands, modulo each n.  One pair
    that does not add up does not show the splitting topological, so every
    pair of realizations of the two sides, enumerated with
    :func:`atomic_realizations`, is tested.  A side whose ZSRSIs are
    fractional has no realization.  The splitting is NSIT exactly where both
    sides have realizations, the group has SRSIs valued modulo n, and no pair
    adds up.

    Parameters
    ----------
    tables : SRSITables
        The tables of the group, what :func:`srsi_tables` returns.
    theta_total : Invariants or ModelSRSIs
        The SRSIs of all the bands, as a record with the fields ``theta_Z``
        and ``theta_Zn``, each keyed by the invariants' names.  Pass what
        :meth:`SRSITables.theta` returns for the model's orbitals, or what
        :func:`srsi_of_model` returns.
    theta_Z_v, theta_Z_c : mapping
        The ZSRSIs of the valence and of the conduction bands,
        ``{name: value}``.  Pass for each the ``theta_Z`` of what
        :func:`srsi_of_bands` returns.

    Returns
    -------
    Splitting or Refusal
        Whether the splitting is NSIT, every atomic realization of each side,
        and the pairs whose SRSIs valued modulo n add up to those of all the
        bands.  The record is truthy even where ``nsit`` is ``False``.

    Examples
    --------
    lin3g (:func:`.models.lin3g`), its eight bands split into the lowest six
    and the two above them:

    >>> b = srsi_of_bands(hsp_table(r, bands=6))                  # doctest: +SKIP
    >>> whole = srsi_of_model(r)                                  # doctest: +SKIP
    >>> above = {name: whole.theta_Z[name] - value for name, value in b.theta_Z.items()}  # doctest: +SKIP
    >>> nsit_criterion(srsi_tables(r.group), whole, b.theta_Z, above).nsit  # doctest: +SKIP
    True

    Refusals
    --------
    A refusal handed in as ``tables`` or as ``theta_total`` is returned as it
    is.

    ``srsi_realizations_truncated``
        the enumeration of one side found more realizations than the default
        ``cap`` of :func:`atomic_realizations`.

    Raises
    ------
    TypeError
        ``theta_total`` without ``theta_Z`` and ``theta_Zn``, or a side or
        ``theta_total.theta_Z`` that is not a mapping.
    ValueError
        ``theta_total`` that does not hold one entry for each of the tables'
        invariants, a side that does not hold one for each integer-valued
        invariant, or ZSRSIs of the two sides that do not add up to those of
        ``theta_total``.

    See Also
    --------
    splitting : the same criterion on a model and its valence bands.
    """
    if isinstance(tables, Refusal):
        return tables
    if isinstance(theta_total, Refusal):
        return theta_total
    if not (hasattr(theta_total, "theta_Z") and hasattr(theta_total, "theta_Zn")):
        raise TypeError("nsit_criterion(): theta_total= is of type "
                        f"{type(theta_total).__name__}; "
                        "pass what tables.theta or srsi_of_model returns.")
    theta_z_h = _by_name("nsit_criterion", "theta_total.theta_Z", tables,
                         theta_total.theta_Z)
    values_v = _by_name("nsit_criterion", "theta_Z_v=", tables, theta_Z_v)
    values_c = _by_name("nsit_criterion", "theta_Z_c=", tables, theta_Z_c)
    zn_names = [row.name for row in tables._rows if row.mod is not None]
    if (len(theta_total.theta_Zn) != len(zn_names)
            or set(theta_total.theta_Zn) != set(zn_names)):
        raise ValueError(
            "nsit_criterion(): theta_total= holds the invariants valued modulo n "
            f"{', '.join(str(k) for k in theta_total.theta_Zn) or 'none'}, and "
            f"#{tables._number_str} has {_span(zn_names) or 'none'}; pass what "
            "tables.theta or srsi_of_model returns.")
    theta_zn_h = {name: getattr(theta_total.theta_Zn[name], "value",
                                theta_total.theta_Zn[name]) for name in zn_names}
    added = {name: values_v[name] + values_c[name] for name in values_v}
    if added != theta_z_h:
        raise ValueError(
            "nsit_criterion(): theta_Z_v= plus theta_Z_c= is "
            f"{_invariants_text(added)}, and theta_total.theta_Z is "
            f"{_values(theta_z_h.values())}; pass the integer-valued invariants of the "
            "valence and of the conduction bands of the same model as theta_total=.")

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
    r"""The no-go test of a gapped splitting of a model's bands.

    The record returned says whether the splitting is non-symmetry-indicated
    topological (NSIT).  The criterion is that of Hwang et al., Nat. Commun.
    17, 8564 (2026), as :func:`nsit_criterion` applies it.  The valence bands
    are the bands asked for and the conduction bands are the rest.  The
    stable real-space invariants (SRSIs) of all the bands are read off the
    model's orbitals with :func:`srsi_of_model`.  The integer-valued SRSIs of
    the valence bands are read off their symmetry data vector with
    :func:`srsi_of_bands`, and those of the conduction bands are the
    difference.  The splitting is NSIT exactly where both sides have atomic
    realizations, the group has SRSIs valued modulo n, and no pair of
    realizations has SRSIs valued modulo n adding up to those of all the
    bands.  The bands are read at the high-symmetry points alone, so a gap
    between those points is taken as given.

    Parameters
    ----------
    realized : .groups.RealizedGroup
        What :func:`.groups.realized_group` returned for the model.
    bands : int or sequence of int
        The valence bands.  Pass an integer ``n`` for the lowest ``n`` bands,
        or a sequence of band indices.  Required.

    Returns
    -------
    Splitting or Refusal
        Whether the splitting is NSIT, every atomic realization of each side,
        and the pairs whose SRSIs valued modulo n add up to those of all the
        bands.  The record is truthy even where ``nsit`` is ``False``.

    Examples
    --------
    lin3g (:func:`.models.lin3g`), its six lowest bands against the two above
    them:

    >>> print(splitting(r, bands=6).summary())                    # doctest: +SKIP
    P4mm1' #99.164, double-valued: nsit, no realization of the valence bands and of the conduction bands adding up to the whole model's invariants valued modulo n   (.group, .nsit)
      Z-valued stable real-space invariants theta_1 to theta_3: the whole model (2, 2, 0) = valence (1, 1, 1) + conduction (1, 1, -1)   (.theta_Z, .valence.theta_Z, .conduction.theta_Z)
      invariants valued modulo n of the whole model: theta_4, theta_5 = (0, 1) mod 2   (.theta_Zn)
      atomic realizations of the valence bands, with (theta_4, theta_5) mod 2   (.valence.realizations)
        1 (-E2)_b + 1 (-E)_c   (1, 1)
        1 (-E1)_b + 2 (-E2)_b   (0, 0)
        1 (-E1)_a + 1 (-E2)_a + 1 (-E2)_b   (1, 0)
      atomic realizations of the conduction bands, with (theta_4, theta_5) mod 2   (.conduction.realizations)
        1 (-E1)_b   (0, 0)
      no pair of a valence and a conduction realization has invariants valued modulo n adding up to the whole model's   (.consistent_pairs)

    Refusals
    --------
    A refusal handed in as ``realized`` is returned as it is.

    ``srsi_scope`` / ``srsi_table_absent``
        the tables hold no SRSIs of the group: it is not grey, or it is one of
        the thirteen space groups whose one invariant is the band count.
    ``srsi_data`` / ``srsi_ebr_basis``
        the tables of the SRSIs are not installed, or were converted against
        another list of elementary band representations than the one
        installed.
    ``ebr_tables`` / ``irrep_tables`` / ``ebr_convention`` / ``tables_not_ready``
        the tables installed hold no elementary band representations of this
        group that can be read in the model's cell.
    ``wyckoff_orbit_content_incomplete`` / ``wyckoff_orbit_not_localizable`` /
    ``wyckoff_setting`` / ``wyckoff_unresolved``
        the SRSIs of the orbitals could not be read, as :func:`srsi_of_model`
        says.
    ``incomplete_content``
        the valence bands could not be read at some high-symmetry point, for
        example where they split a degenerate multiplet.
    ``not_a_band_structure`` / ``indicator_not_evaluable``
        the integer-valued SRSIs of the valence bands could not be read, as
        :func:`srsi_of_bands` says.
    ``srsi_realizations_truncated``
        the enumeration of one side found more realizations than the default
        ``cap`` of :func:`atomic_realizations`.

    Raises
    ------
    TypeError
        ``bands`` left out, or ``realized`` is not what
        :func:`.groups.realized_group` returned.
    ValueError
        ``bands`` outside the model's bands or naming none.

    See Also
    --------
    nsit_criterion : the same criterion on invariants in hand.
    """

    if bands is None:
        raise TypeError("splitting(): no bands= is passed; pass the valence bands, as "
                        "their number or their indices.")
    if isinstance(realized, Refusal):
        return realized
    model, group, _assignment, rep = _unpack_realized(realized, "splitting")
    # the bands are checked here, so that the message names this call
    _sewing._band_block(int(model.nstate), bands, "splitting")
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
