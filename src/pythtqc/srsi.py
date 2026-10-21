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

from collections.abc import Mapping
from dataclasses import dataclass, field
from fractions import Fraction
import functools
from typing import Any

import numpy as np

from . import _naming
from . import _smith
from . import _spacegroup
from . import _tables
from . import ebr as _ebr
from . import sewing as _sewing
from .refusal import InternalError, Refusal
from ._text import _index_ranges, _label_terms
from ._matrices import _BLOCH_TOL
from ._centres import _position_defect
from .symmetry import BandRepresentation
from ._srsitables import (
    GATES, MaximalKey, NAME_RELATIONS, NamePair, SRSIIndicator, SRSIRow, SRSITables,
    SiteKey, SplitEBR, _CONFIGURATION, _POSITION, _TableKey, _by_modulus, _count,
    _ebr_key, _gcd, _group, _indicator_line, _maximal_basis, _no_bare_number,
    _rational_p, _site_irreps, _table_of_number, _values, canonical_determinism,
    from_irreptables_name, indicator_vocabularies, name_correspondence, srsi_status,
    srsi_tables, to_irreptables_name)

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
        ``bands=`` names other bands than the content was measured on, or the
        group or ``rep=`` is not the one the table was read in, which it
        carries as ``t.rep``.

    Refusals
    --------
    ``srsi_scope``, ``srsi_table_absent``, ``srsi_data``, ``srsi_ebr_basis``
        from :func:`srsi_tables`: the group has no tables.
    ``incomplete_content``, ``not_a_band_structure``, ``indicator_not_evaluable``
        from :func:`..ebr.symmetry_indicator`, with the refusals of the content it
        reads.
    """
    block = _ebr._bands_of(content, bands, where="srsi_of_bands")
    _ebr._table_binding(content, group, rep, where="srsi_of_bands")
    gate = _spacegroup._lattice_not_descended(group)
    if gate is not None:
        return gate
    tables = srsi_tables(group)          # a measured GroupData, not an SRSITables:
    if not tables:                       # the momentum-space half needs the group
        return tables
    si = _ebr.symmetry_indicator(content, group, rep=rep, points=points,
                                 bands=block)
    if not si:
        return si
    data = _tables.ebr_data(group)
    if not data:
        return data
    u, v, d, matrix = _smith._smith_system(data, group.number_str)
    p = _rational_p(u, v, d, [Fraction(int(x)) for x in si.y])
    n_row, n_col = len(matrix), len(matrix[0])
    check = [sum((Fraction(matrix[i][j]) * p[j] for j in range(n_col)), Fraction(0))
             for i in range(n_row)]
    if check != [Fraction(int(x)) for x in si.y]:
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
    gate = _spacegroup._lattice_not_descended(group_or_tables)
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
    gate = _spacegroup._lattice_not_descended(group_or_tables)
    if gate is not None:
        return gate
    tables = (group_or_tables if isinstance(group_or_tables, SRSITables)
              else srsi_tables(group_or_tables))
    if not tables:
        return tables
    return tables.theta_columns(p)

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
    points_map = _tables.high_symmetry_points(group)
    if not points_map:
        return [(None, None, points_map) for _rows in blocks]
    read = [({}, [], {}) for _rows in blocks]
    for name in sorted(points_map):
        k = points_map[name]
        lg = _tables.little_group(group, k)
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
            got = _tables.identify_from_characters(group, k, [chars],
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
    gate = _spacegroup._lattice_not_descended(group)
    if gate is not None:
        return gate
    rep = own_rep if rep is None else rep
    tables = srsi_tables(group)
    if not tables:
        return tables
    data = _tables.ebr_data(group)
    if not data:
        return data
    sites = _tables.wyckoff_sites_tables(group, assignment.site_positions)
    if isinstance(sites, Refusal):
        return sites
    # a free-standing layer's orbitals are moved within the layer only: the
    # positions off it are never a localization
    layer = _ebr._layer_reading(group, data, rep)
    off_layer = (set() if layer is None
                 else {_tables._letter_only(position) for position in layer["outside"]})

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

    characters = getattr(rep, "characters", None)
    group, assignment, model = rep.group, rep.assignment, rep.model
    if characters is None:
        return ()
    represented = {int(i) for i in rep.indices}
    if any(int(s.ind) not in represented for s in group.symmetries if not s.time_reversal):
        return ()
    sites = _tables.wyckoff_sites_tables(group, assignment.site_positions)
    if isinstance(sites, Refusal):
        return ()
    data = _tables.ebr_data(group)
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
            listed = frozenset(_tables._letter_only(position) for _name, position in ebrs)
            through = []
            for variety in here.varieties:
                for letter in _spacegroup._letters_on_closure(int(group.sg_number),
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
                                    [[_tables._bands_spanned(name) for name, _at in ebrs]]])
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
# the no-go criterion in theta-coordinates
# ---------------------------------------------------------------------------
#: The verdicts a :class:`Splitting` holds.  A pair of atomic realizations that
#: adds up does not make a splitting trivial: it makes any topology of the
#: splitting invisible to the SRSIs, the criterion being sufficient, never
#: necessary.
SPLITTING_VERDICTS = ("diagnosed topological", "invisible to the SRSIs",
                      "no Z_n SRSIs", "stable topological")

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

@functools.lru_cache(maxsize=None)
def _stable_pairs(sg: int, spinor: bool) -> tuple:
    """The same-position pairs of one system -- :func:`stably_equivalent_pairs`."""
    tables = _table_of_number(int(sg), bool(spinor))
    if not tables:                            # pragma: no cover - gated above
        return ()
    data = _tables.ebr_data(_TableKey(tables.number_str, spinor))
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
    gate = _spacegroup._lattice_not_descended(group_or_tables)
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

    if bands is None:
        raise TypeError("splitting() missing required argument 'bands'")
    model, group, _assignment, own_rep = _unpack_realized(realized)
    gate = _spacegroup._lattice_not_descended(group)
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
