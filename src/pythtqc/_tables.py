"""The tables of the irreptables package, read in the model's own cell.

The high-symmetry points and the little group of a k-point; the character
table of its little-group representations, and the ones measured characters
carry; the Wyckoff position of each site in the tables' setting; the
elementary band representations and the indicator group; and a mesh that
respects the group.  Characters and labels are carried into the convention
in which a lattice translation ``t`` acts as ``e^{-ik·t}``, where the tables
have ``e^{+ik·t}``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from functools import lru_cache
import itertools
import re
from typing import Any

import numpy as np

from . import _sitechars
from . import _spacegroup
from .refusal import InternalError, Refusal
from ._tolerance import (
    LABEL_TOL, WYCKOFF_REFERENCE_TOL, _CHAR_TOL, _KGRID_KTOL, _LATTICE_TOL,
    _PHASE_REAL_TOL, _SU2_TOL)
from ._text import _point_text
from ._spacegroup import (
    GroupData, CorepIdentification, KGrid, LittleGroup, OrbitalAssignment, SiteReference, Term,
    _EBRTransportError, _GENERIC_PARAMETER, _SiteReading, _centering, _in_tables_cell,
    _is_reciprocal, _is_translation, _lattice_not_descended, _letters_on_closure,
    _magnetic_type, _pad_cell, _pyxtal_wyckoff, _reference_operations, _refuse_no_tables,
    _require_group, _sites_and_typat, _snap_rational, _variety_residual,
    _wyckoff_positions_spglib, _wyckoff_setting_audit, _written_site_symmetry,
    tables_status)

__all__ = ["kpoint_of_label", "ebr_rows", "wyckoff_sites_tables", "little_group",
           "HighSymmetryPoints", "high_symmetry_points", "corep_table",
           "identify_from_characters", "ebr_data", "EBRData", "EBRConvention",
           "indicator_group", "decompose_ebrs", "k_grid"]


def kpoint_of_label(label: str) -> str:
    """The high-symmetry point a label belongs to, ``GM`` for ``-GM6``.

    The point is the run of letters before the first digit, read after the
    ``-`` that marks a double-valued label: ``M`` for ``-M3M4``.

    Parameters
    ----------
    label : str
        The label of a little-group representation, ``"-GM6"``, ``"GM2-"``
        or ``"X1"``.

    Returns
    -------
    str
        The name of the point in the tables, ``"GM"``.
    """
    s = str(label)
    if s.startswith("-"):
        s = s[1:]
    i = 0
    while i < len(s) and not s[i].isdigit():
        i += 1
    return s[:i]


_COREP_TOKEN = re.compile(r"([A-Z]+)(\d+)([+-]?)")


def point_order(name):
    """The sort key that places a high-symmetry point in the order points are listed.

    ``name`` is the point's name in the tables, ``"GM"`` or ``"KA"``.  ``GM``
    comes first, then the other points alphabetically.  A name of more than
    one letter ending in ``A`` is read as the partner ``-k`` of the point
    named without it, and placed right after that point: ``KA`` after ``K``.
    A list aligned with a matrix, as the basis of the elementary band
    representations is, keeps the tables' order instead.
    """
    name = str(name)
    base = name[:-1] if len(name) > 1 and name.endswith("A") else name
    return (base != "GM", base, name != base)


def corep_order(label):
    """The sort key that places a label in the order labels are listed.

    ``label`` is the name of a little-group representation, ``"GM1+"`` or
    ``"-M3M4"``.  Labels are ordered by point (:func:`point_order`), then by
    the numbers of the irreps the label names, ascending: ``GM2`` before
    ``GM10``, ``GM1+`` before ``GM1-``, a label pairing two irreps placed by
    the first it names and then the second.  A single-valued label comes
    before the double-valued one of the same numbers.
    """
    s = str(label)
    double = s.startswith("-")
    tokens = _COREP_TOKEN.findall(s[1:] if double else s)
    if not tokens:
        return ((True, s, False), (), double, s)
    return (point_order(tokens[0][0]),
            tuple((int(n), sign == "-") for _point, n, sign in tokens), double, s)


def ordered_content(counts) -> dict:
    """Multiplicities of little-group representations, reordered by label.

    ``counts`` maps each label to its multiplicity.  Returns a new mapping with
    the same entries, its labels in the order of :func:`corep_order`.
    """
    return {lab: counts[lab] for lab in sorted(counts, key=corep_order)}


def ebr_rows(group) -> dict | Refusal:
    r"""The symmetry data vector of each elementary band representation of the group.

    A row holds the multiplicity of each little-group representation at
    every high-symmetry point of the group.  The labels are in the package's
    convention, a lattice translation ``t`` acting as ``e^{-ik·t}``, where the
    tables of the Bilbao Crystallographic Server have ``e^{+ik·t}``.  The
    label the server prints at ``k`` is the label returned here at ``-k``: in
    the row of ``A1`` at ``1b`` of space group 143, ``K3`` stands where the
    server prints ``K2``.  The Wyckoff positions are the server's.

    Parameters
    ----------
    group : .groups.GroupData
        The group.  Pass ``r.group`` of what :func:`..groups.realized_group`
        returns.

    Returns
    -------
    dict or Refusal
        ``{(irrep, wyckoff): {label: multiplicity}}``, the zeros dropped, in
        the order the tables list the elementary band representations.  A key
        is the site irrep the row is induced from and its Wyckoff position,
        so that ``rows["-E1", "1a"]`` is one row.  A ``Refusal`` in its place
        is falsy.

    Refusals
    --------
    A refusal handed in as ``group`` is returned as it is.

    ``ebr_tables`` / ``irrep_tables`` / ``ebr_convention``
        the tables installed hold no elementary band representations of this
        group that can be read in the package's convention.

    Raises
    ------
    TypeError
        ``group`` is not ``r.group``: ``r`` itself, ``r.rep``, the ``group``
        of :func:`..srsi.srsi_tables` for a space-group number, which has no
        cell, or anything else.
    """
    _require_group("ebr_rows", group)
    rows = _ebr_rows(group)
    if isinstance(rows, Refusal):
        return rows
    return {(_ebr_irrep(key.split("@", 1)[0]), _wyckoff_letter(key.split("@", 1)[1])): content
            for key, content in rows.items()}


def _ebr_rows(group) -> dict | Refusal:
    """The row of each elementary band representation, keyed as the tables write it.

    ``group`` is the group, ``r.group``.  Returns
    ``{"-E1↑G(2)@1a(6mm1',6mm)": {label: multiplicity}}``, the zeros dropped,
    the representations in the tables' order and the labels in the convention
    of :func:`ebr_data`, in the order of :func:`corep_order`.  Returns
    ``lattice_not_descended`` for a lattice supergroup, or a refusal of
    :func:`_ebr_data`.
    """
    gate = _spacegroup._lattice_not_descended(group)
    if gate is not None:
        return gate
    data = _ebr_data(group)
    if isinstance(data, Refusal):
        return data
    out = {}
    labels = data["labels_physics_convention"]
    for (name, wy), col in zip(data["ebrs"], np.asarray(data["matrix"]).T):
        out[f"{name}@{wy}"] = ordered_content(
            {lab: int(v) for lab, v in zip(labels, col) if v})
    return out


def _ebr_irrep(name: str) -> str:
    """The site irrep an elementary band representation is induced from.

    ``name`` is the representation's name in the tables.  Returns the part
    before ``↑``: ``"-E1"`` for ``"-E1↑G(2)"``.
    """
    return str(name).split("↑")[0]


def _wyckoff_letter(position: str) -> str:
    """A Wyckoff position's multiplicity and letter, without its site groups.

    ``position`` is the position as the tables of elementary band
    representations write it.  Returns ``"1a"`` for ``"1a(4mm1',4mm)"``.
    """
    return str(position).split("(")[0]


def _letter_only(position: str) -> str:
    """A Wyckoff position's letter alone.

    ``position`` is the position with or without its site groups.  Returns
    ``"a"`` for ``"1a(4mm1',4mm)"`` and for ``"1a"``.
    """
    return _wyckoff_letter(position).lstrip("0123456789")


def _bands_spanned(name: str) -> int:
    """The number of bands an elementary band representation spans.

    ``name`` is the representation's name in the tables.  Returns the integer
    in its last parentheses: ``4`` for ``"-E1↑G(4)"``.
    """
    name = str(name)
    return int(name[name.rindex("(") + 1:name.rindex(")")])


def wyckoff_sites_tables(
        group: GroupData, positions=None, *,
        position_tol: float = WYCKOFF_REFERENCE_TOL) -> tuple | Refusal:
    """The Wyckoff position of each site, in the setting of the tables.

    The setting is the one the elementary band representations of
    :func:`ebr_data` are listed in, from the tables of the irreptables
    package.  Two settings can give one orbit different letters: the
    Fu-Kane-Mele diamond's orbit is ``8a`` here and ``8b`` in spglib's
    setting.  Two positions can give atomic limits with the same symmetry
    data vector and different stable real-space invariants, so a site's
    letter is read in this setting.  Each point is given the Wyckoff position
    with the fewest free parameters on which it lies within ``position_tol``,
    and of those the one of smallest multiplicity.  For the model's orbitals,
    ``r.wyckoff_orbits`` holds the same positions.

    Parameters
    ----------
    group : GroupData
        The group.  Pass ``r.group`` of what :func:`..groups.realized_group`
        returns.
    positions : array, shape (n, 3), optional
        The points, in reduced coordinates of the group's three-dimensional
        cell.  Leave it out for the sites the group was found on: the
        model's orbital sites, and those of the species added with the
        ``decorations`` of :func:`..groups.realized_group`.
    position_tol : float, default 1e-4
        The largest distance at which a point is read as lying on a Wyckoff
        position, in reduced coordinates of the conventional cell of the
        tables.

    Returns
    -------
    tuple of SiteReference or Refusal
        One for each point: its Wyckoff position, ``"8a"``; its number of
        free parameters; and the positions with elementary band
        representations in the tables that lie on the point, line or plane
        those parameters sweep out, ``("3a", "3b")`` for the line ``6c`` of
        ``R-3m``.  A ``Refusal`` in its place is falsy.

    Examples
    --------
    Fu, Kane and Mele's diamond model, pythtb's ``fu_kane_mele``, whose two
    orbitals lie on one orbit, ``8a`` in the setting of the tables:

    >>> from pythtb.models import fu_kane_mele
    >>> r = realized_group(fu_kane_mele(1.0, 0.125, [0, 0, 0, 0]))
    >>> [s.wyckoff for s in wyckoff_sites_tables(r.group)]
    ['8a', '8a']

    Refusals
    --------
    A refusal handed in as ``group`` is returned as it is.

    ``wyckoff_setting``
        the list of Wyckoff positions of the pyxtal package is not written in
        the setting of the tables.
    ``wyckoff_unresolved``
        a point lies on no Wyckoff position within ``position_tol``.  The
        reason names the nearest and its distance.
    ``ebr_tables`` / ``irrep_tables`` / ``ebr_convention`` / ``tables_not_ready``
        the tables installed hold no elementary band representations of this
        group that can be read in the model's cell.

    Raises
    ------
    TypeError
        ``group`` is not ``r.group``: ``r`` itself, ``r.rep``, the ``group``
        of :func:`..srsi.srsi_tables` for a space-group number, which has no
        cell, or anything else.
    """
    _require_group("wyckoff_sites_tables", group)
    read = _sites_in_tables_setting(group, positions, position_tol=position_tol)
    if isinstance(read, Refusal):
        return read
    multiplicity = {p["letter"]: int(p["multiplicity"])
                    for p in _pyxtal_wyckoff(int(group._sg_number))[0]}
    return tuple(SiteReference(
        wyckoff=f"{int(s.multiplicity)}{s.letter}",
        slides_onto=tuple(f"{multiplicity[letter]}{letter}" for letter in s.closure),
        n_free=s.n_free, position=s.position, _varieties=s.varieties, _index=s.index,
        _reference=s.reference, _residual=s.residual) for s in read)


def _sites_in_tables_setting(group: GroupData, positions=None, *,
                             position_tol: float = WYCKOFF_REFERENCE_TOL
                             ) -> tuple | Refusal:
    """Each point's Wyckoff position in the tables' setting, as the package reads it.

    ``group`` is the group, ``r.group``.  ``positions`` are the points, in
    reduced coordinates of the group's three-dimensional cell, or ``None`` for
    the sites the group was found on.  ``position_tol`` is the largest
    distance, in reduced coordinates of the tables' conventional cell, at which
    a point lies on a position.  Returns a :class:`_SiteReading` for each
    point, or a refusal :func:`wyckoff_sites_tables` documents, or
    ``lattice_not_descended`` for a lattice supergroup.
    """
    gate = _lattice_not_descended(group)
    if gate is not None:
        return gate
    if not group._tables_ready:
        return Refusal("tables_not_ready",
                       group._setup_note or f"the cell of {group.name} "
                       f"#{group.number_str} could not be matched to the setting the "
                       "installed tables are written in. Report the model, with this "
                       "reason.",
                       context={"number_str": group.number_str})
    data = _ebr_data(group)
    if isinstance(data, Refusal):
        return data
    ebr_positions = []
    for _name, position in data["ebrs"]:
        match = re.match(r"^(\d+)([a-zA-Z])\((.*)\)$", str(position))
        if match is None:                              # pragma: no cover: guard
            raise InternalError(
                f"#{group.number_str}: the tables write the Wyckoff position of an "
                f"elementary band representation as {position!r}, which is not "
                "'<multiplicity><letter>(<site groups>)'. Report it, with the model and "
                "the call.")
        ebr_positions.append((int(match.group(1)), match.group(2), match.group(3)))
    listed = frozenset(letter for _m, letter, _s in ebr_positions)

    audit = group._wyckoff_audit.get("audit")
    if audit is None:
        audit = _wyckoff_setting_audit(int(group._sg_number),
                                       _reference_operations(group),
                                       ebr_positions)
        group._wyckoff_audit["audit"] = audit
    if not audit["ok"]:
        bad = audit["offenders"][0]
        letter = bad["letter"]
        position = f"{dict((l, m) for m, l, _s in ebr_positions)[letter]}{letter}"
        if bad["reason"] == "multiplicity":
            failed = (f"the position {letter} has multiplicity {bad['measured']} in the "
                      f"pyxtal package's list, and {bad['expected']} in the tables")
        elif bad["measured"] is None:
            failed = f"the position {position} of the tables is not in the pyxtal package's list"
        elif bad["expected"] is None:
            failed = (f"the conventional cell has "
                      f"{audit['point_group'] * audit['centering']} operations, not a "
                      f"multiple of the multiplicity of the position {position}")
        else:
            failed = (f"the stabilizer of a point of the position {position} has order "
                      f"{bad['measured']}, where {bad['expected']} is expected")
        return Refusal(
            "wyckoff_setting",
            f"the Wyckoff positions the pyxtal package lists for space group "
            f"{group._sg_number} are not written in the setting of the tables of "
            f"{group.name} #{group.number_str}: {failed}. Report it, with the model and the "
            "call.",
            value=None if bad["measured"] is None else float(bad["measured"]),
            threshold=None if bad["expected"] is None else float(bad["expected"]),
            context={"number_str": str(group.number_str), "sg": int(group._sg_number),
                     "offenders": audit["offenders"], "audit": audit})

    pyxtal_positions, _centering = _pyxtal_wyckoff(int(group._sg_number))
    raw = (np.asarray(group._site_positions, dtype=float) if positions is None
           else np.atleast_2d(np.asarray(positions, dtype=float)))
    reference = _in_tables_cell(group, raw)

    residuals = {p["letter"]: _variety_residual(p["arms"], reference)
                 for p in pyxtal_positions}
    out = []
    for i, (x, x_ref) in enumerate(zip(raw, reference)):
        hits = [p for p in pyxtal_positions
                if residuals[p["letter"]][i] <= position_tol]
        if not hits:                     # pragma: no cover: the general position's
            # variety is the whole cell, so this needs a pyxtal list without one
            nearest = min(pyxtal_positions, key=lambda p: residuals[p["letter"]][i])
            return Refusal(
                "wyckoff_unresolved",
                f"the position {np.round(x, 6).tolist()} is "
                f"{np.round(x_ref, 6).tolist()} in the conventional cell of the tables of "
                f"{group.name} #{group.number_str}, and lies on none of its Wyckoff "
                f"positions within the tolerance. The nearest is "
                f"{nearest['multiplicity']}{nearest['letter']}, "
                f"{float(residuals[nearest['letter']][i]):.3g} away in reduced "
                f"coordinates. Check the position.",
                value=float(residuals[nearest["letter"]][i]),
                threshold=float(position_tol),
                context={"site": i, "position": x, "reference": x_ref,
                         "nearest": nearest["letter"],
                         "number_str": str(group.number_str)})
        hits.sort(key=lambda p: (p["n_free"], p["multiplicity"], p["letter"]))
        own = hits[0]
        out.append(_SiteReading(
            index=i, position=x, reference=x_ref, letter=own["letter"],
            multiplicity=own["multiplicity"], n_free=own["n_free"],
            varieties=tuple(p["letter"] for p in hits),
            closure=_letters_on_closure(int(group._sg_number), own["letter"], listed,
                                        float(position_tol)),
            residual=float(residuals[own["letter"]][i])))
    return tuple(out)


def _assign_orbitals(model, group: GroupData, *, typat=None,
                     symprec=None) -> OrbitalAssignment | Refusal:
    """The site orbit each orbital sits on, and how the operations permute the sites.

    ``model`` is the pythtb model whose orbital positions are read, rather
    than the group's sites.  Its cell is the group's three-dimensional cell
    once each vector out of a layer's plane is given the length the group's
    cell gives it.  ``group`` is the group whose operations permute the
    sites.  ``typat`` gives each orbital's species, as
    :func:`.._realized.realized_group` takes it; pass the value given there.
    ``symprec`` is the matching tolerance, a Cartesian length in the model's
    units, by default the group's own.

    Each operation's image ``R·τ + t`` of a site is matched to the nearest
    site of the same species.  The permutation of the sites, with the integer
    lattice shift that fixes the phase of a permuting operation's matrix, is
    what a representation is built from; the orbits and site-symmetry groups
    are read off it.  Beside them stand three sets of Wyckoff letters, each
    left empty where it cannot be read.  The letters in spglib's setting are
    read per orbit.  The letters in the tables' setting are read per site.
    spglib's own letter of each site is read in the group of the sites
    alone.  All three are left empty for a lattice supergroup.

    Returns the :class:`OrbitalAssignment`, held as the ``_assignment`` of
    what :func:`.._realized.realized_group` returns.  Returns
    ``wyckoff_orbit`` where a site's image lies farther than ``symprec`` from
    every site of its species, naming the orbitals, the operation and the
    distance.  Raises ``ValueError`` where ``typat`` has not one entry per
    orbital.  Raises :class:`~.refusal.InternalError` where the model's cell,
    so scaled, is not the group's: the group handed in is always the one
    measured on the model's own cell.
    """
    tol = float(group._symprec if symprec is None else symprec)
    lat, positions, _ = _pad_cell(model, group._vacuum_length, lattice=group._lattice)
    if not np.allclose(lat, group._lattice, atol=1e-9):
        apart = float(np.abs(np.asarray(lat, dtype=float)
                             - np.asarray(group._lattice, dtype=float)).max())
        raise InternalError(f"the model's cell, its vacuum included, differs from the cell "
                            f"of {group.name} #{group.number_str} by up to {apart:.3g} in a "
                            "Cartesian component, and the group is measured on the model's "
                            "own cell. Report it, with the model and the call.")
    pos, site_orbs, species = _sites_and_typat(positions, typat, tol, lat)
    nsite = pos.shape[0]
    nsym = len(group._symmetries)

    perm = np.full((nsym, nsite), -1, dtype=int)
    shift = np.zeros((nsym, nsite, 3), dtype=int)
    worst = 0.0
    for g, sym in enumerate(group._symmetries):
        images = pos @ np.asarray(sym.rotation, dtype=float).T + sym.translation
        for i in range(nsite):
            d = pos - images[i]
            d = d - np.round(d)
            dist = np.linalg.norm(d @ lat, axis=1)
            same = species == species[i]
            dist_masked = np.where(same, dist, np.inf)
            j = int(np.argmin(dist_masked))
            if not np.isfinite(dist_masked[j]) or dist_masked[j] > tol:
                orbs = site_orbs[i]
                return Refusal(
                    "wyckoff_orbit",
                    f"operation #{sym.ind} (rotation {sym.rotation.tolist()}, translation "
                    f"{np.round(sym.translation, 6).tolist()}) carries the orbitals "
                    f"{list(orbs)} at the reduced position {np.round(pos[i], 6)} to a point "
                    f"{float(np.min(dist_masked)):.6g} from the nearest orbital of their "
                    "kind, a Cartesian distance in the units of the lattice vectors. Check "
                    "the orbitals' positions.",
                    value=float(np.min(dist_masked)), threshold=tol,
                    context={"site": i, "orbitals": list(orbs), "operation_ind": int(sym.ind),
                             "position": pos[i], "image": images[i] % 1.0,
                             "distances": dist})
            worst = max(worst, float(dist_masked[j]))
            perm[g, i] = j
            shift[g, i] = np.round(images[i] - pos[j]).astype(int)

    # orbits = connected components of the permutation action
    orbit_of = np.full(nsite, -1, dtype=int)
    orbits: list[tuple] = []
    for i in range(nsite):
        if orbit_of[i] >= 0:
            continue
        members = sorted(set(int(perm[g, i]) for g in range(nsym)))
        idx = len(orbits)
        for j in members:
            orbit_of[j] = idx
        orbits.append(tuple(members))

    site_sym = tuple(tuple(int(group._symmetries[g].ind) for g in range(nsym) if perm[g, i] == i)
                     for i in range(nsite))

    # Wyckoff letters in spglib's setting.  irrep gives one representative of
    # each position, and a site may sit on a symmetry-related copy of it, so
    # every site of the orbit is tried and the match with the fewest free
    # parameters is kept.  A lattice supergroup's letters are the finer
    # lattice's, so none is read.
    supergroup = int(getattr(group, "_lattice_index", 1) or 1) > 1
    letters: list[str] = []
    wy = (Refusal("lattice_not_descended", "") if supergroup
          else _wyckoff_positions_spglib(group))
    for orb in orbits:
        if isinstance(wy, Refusal):
            letters.append("")
            continue
        best, best_free = "", 99
        for w in wy:
            if w.n_free >= best_free:
                continue
            if any(w.distance(pos[i], lat) <= max(tol, 1e-6) for i in orb):
                best, best_free = w.letter, w.n_free
        letters.append(best)

    orbit_of_orbital = np.zeros(model.norb, dtype=int)
    site_of_orbital = np.zeros(model.norb, dtype=int)
    for s, orbs in enumerate(site_orbs):
        for o in orbs:
            orbit_of_orbital[o] = orbit_of[s]
            site_of_orbital[o] = s

    # The letter in the tables' setting, the one a table is read with.  Where
    # it cannot be read the letters are left empty: the orbits above are
    # measured from the operations and do not depend on it.
    ref_letters: tuple = ()
    if not supergroup:
        sites_ref = _sites_in_tables_setting(group, pos)
        if not isinstance(sites_ref, Refusal):
            ref_letters = tuple(s.letter for s in sites_ref)

    # spglib's own per-site letter for the sites alone: a letter of the group of
    # the positions, which can be larger than the group handed in
    spg_letters: tuple = ()
    try:
        import spglib
        ds = None if supergroup else spglib.get_symmetry_dataset(
            (lat, pos, np.asarray(species, dtype=int)), symprec=tol)
        if ds is not None:
            spg_letters = tuple(ds.wyckoffs)
    except Exception:  # pragma: no cover: spglib API drift must not break the record
        pass

    # The name of each site-symmetry group, spelled as the atomic-limit table
    # spells it: the point group of the site's own unitary operations.  The
    # sites of one orbit have conjugate site groups and so one name, which is
    # read once per orbit.

    unitary = {int(s.ind) for s in group._symmetries if not s.time_reversal}
    orbit_names = [_sitechars.point_group_symbol(
        group, [i for i in site_sym[int(orb[0])] if i in unitary]) for orb in orbits]
    site_names = tuple(orbit_names[int(orbit_of[i])] for i in range(nsite))

    return OrbitalAssignment(
        orbits=tuple(orbits), orbit_of_site=orbit_of, orbit_of_orbital=orbit_of_orbital,
        site_of_orbital=site_of_orbital, site_group=site_sym,
        permutations=perm, lattice_shift=shift,
        wyckoff_spglib=tuple(letters), max_distance=worst,
        site_positions=pos, site_orbitals=site_orbs,
        wyckoff_spglib_dataset=spg_letters, site_symmetry=site_names,
        wyckoff_tables=ref_letters)


# --------------------------------------------------------------------------
# little groups and irrep identification
# --------------------------------------------------------------------------

def little_group(group: GroupData, k) -> LittleGroup | Refusal:
    """The operations that leave ``k`` invariant up to a reciprocal lattice vector.

    The unitary operations are listed apart from those combined with time
    reversal.  A character vector at ``k`` has an entry for each unitary
    operation, in the order listed.

    Parameters
    ----------
    group : GroupData
        The group.  Pass ``r.group`` of what :func:`..groups.realized_group`
        returns.
    k : str or array, shape (dim_k,)
        The point.  Pass its name in the tables, ``"X"``, one of those
        :func:`high_symmetry_points` returns, or its reduced coordinates.  A
        layer also takes the three reduced coordinates of a point of its
        plane in its three-dimensional cell.

    Returns
    -------
    LittleGroup or Refusal
        The little group: its operations by index, unitary and combined with
        time reversal; the reciprocal lattice vector each unitary one moves
        ``k`` by; and the name of ``k`` in the tables where it has one.
        ``summary()`` prints it.  A ``Refusal`` in its place is falsy.

    Refusals
    --------
    A refusal handed in as ``group`` is returned as it is.

    ``out_of_plane``
        ``k`` has a component out of a layer's plane, or names a tabulated
        point out of the plane.
    ``tables_not_ready``
        ``k`` is a name, and the model's cell could not be matched to the
        setting the tables are written in.

    Raises
    ------
    TypeError
        ``group`` is not ``r.group``: ``r`` itself, ``r.rep``, the ``group``
        of :func:`..srsi.srsi_tables` for a space-group number, which has no
        cell, or anything else.
    ValueError
        ``k`` is a name the group does not tabulate, or has neither
        ``dim_k`` coordinates nor three.
    """
    _require_group("little_group", group)
    k = _resolve_kpoint("little_group", group, k)
    if isinstance(k, Refusal):
        return k
    return _little_group(group, k)


def _little_group(group: GroupData, k, call: str = "little_group") -> LittleGroup | Refusal:
    """The little group at the coordinates ``k``, as :func:`little_group` returns it.

    ``group`` is the group, ``r.group``.  ``k`` holds ``dim_k`` reduced
    coordinates, or three for a point of a layer's three-dimensional cell.
    ``call`` is the public call's name, which opens the ``ValueError`` raised
    on any other number of coordinates.  Returns the :class:`LittleGroup`, or
    ``out_of_plane`` for three coordinates off a layer's plane, or
    ``lattice_not_descended`` for a lattice supergroup.
    """
    gate = _lattice_not_descended(group)
    if gate is not None:
        return gate
    k = np.asarray(k, dtype=float).ravel()
    if k.size == 3 and group._dim_k != 3:
        if not group._in_plane(k[None, :]):
            return Refusal("out_of_plane",
                           f"k = {np.round(k, 6)} has a component out of the layer's plane, "
                           f"whose periodic directions are {group.periodic_dirs}. Ask a "
                           "point of the plane.",
                           value=float(np.max(np.abs(k[list(group._vacuum_dirs)]))),
                           threshold=0.0, context={"k": k, "periodic_dirs": group.periodic_dirs})
        k3 = k
        k = group._k_from_3d(k3[None, :])[0]
    elif k.size == group._dim_k:
        k3 = group._k_to_3d(k[None, :])[0]
    elif group._dim_k == 3:
        raise ValueError(f"{call}(): the number of coordinates of k=, {k.size}, is not 3. "
                         "Pass 3, or the name of a tabulated point.")
    else:
        raise ValueError(f"{call}(): the number of coordinates of k=, {k.size}, is neither "
                         f"{group._dim_k} nor 3. Pass {group._dim_k}, or 3 for a point of "
                         "the layer's plane in its three-dimensional cell, or the name of a "
                         "tabulated point.")

    uni, anti = [], []
    for sym in group._symmetries:
        G = sym.transform_k(k3) - k3
        if not np.allclose(G, np.round(G), atol=1e-8):
            continue
        (anti if sym.time_reversal else uni).append(sym)
    uni.sort(key=lambda s: s.ind)
    anti.sort(key=lambda s: s.ind)
    return LittleGroup(k=k, kpoint=_kpoint_name(group, k3),
                       unitary=tuple(int(s.ind) for s in uni),
                       antiunitary=tuple(int(s.ind) for s in anti),
                       G={int(s.ind): np.round(s.transform_k(k3) - k3) for s in uni},
                       _k3=k3, _unitary_operations=uni, _antiunitary_operations=anti,
                       _group=group)


def _kpoint_name(group: GroupData, k3):
    """The tables' name of the point ``k3``, or ``None`` where it is none of theirs.

    ``group`` is the group, ``r.group``, and ``k3`` three reduced coordinates
    of its three-dimensional cell.  The name is that of the first point of
    :func:`_table_kpoints` that ``k3`` equals up to a reciprocal lattice
    vector, each coordinate within the default ``ktol`` of
    :func:`_corep_table` (:func:`_same_point`).  Returns ``None`` also where
    the group's tables cannot be read in this cell.
    """
    if not group._tables_ready:
        return None
    ktol = _corep_table.__kwdefaults__["ktol"]
    for name, tabulated in _table_kpoints(group).items():
        if _same_point(k3, tabulated, ktol):
            return name
    return None


def _table_kpoints(group: GroupData) -> dict:
    """The tabulated points, by name, in the group's three-dimensional cell.

    ``group`` is the group, ``r.group``.  The points are those of the irrep
    table of the irreptables package and, for a group without time reversal,
    the partners ``-k`` its elementary band representations list on their own
    (:func:`_partner_points`), each right after its point.  Returns
    ``{name: k}`` in the tables' order, each ``k`` three reduced coordinates
    of the group's three-dimensional cell.  Each coordinate is snapped to the
    fraction it is and reduced into ``[0, 1)``, so that a zero reads ``0``
    whatever sign rounding gave it.
    """
    table = _spacegroup._irrep_table(group.number_str, group.spinor, group._magnetic)
    inv = np.linalg.inv(group._refUC.T)
    partner = ({} if group._magnetic else
               {base: arm for arm, base, _naming
                in _partner_points(group.number_str, group.spinor)})
    listed: dict[str, np.ndarray] = {}
    for irr in table.irreps:
        if irr.kpname in listed:
            continue
        listed[irr.kpname] = np.asarray(irr.k, dtype=float)
        if irr.kpname in partner:                 # its partner -k, next to it
            listed[partner[irr.kpname]] = -listed[irr.kpname]
    return {name: _snap_rational((inv @ _snap_rational(k)) % 1.0) % 1.0 + 0.0
            for name, k in listed.items()}


@lru_cache(maxsize=None)
def _partner_points(number_str: str, spinor: bool) -> tuple:
    r"""The points ``-k`` a group's EBR rows list and its irrep table does not.

    ``number_str`` is the number of a group without time reversal in the
    tables, ``"143"``, and ``spinor`` whether its double-valued tables are
    read.  Where no operation carries ``k`` onto ``-k``, the two are different
    points, and the elementary band representations have rows at both, ``KA``
    beside ``K``, where the irrep table lists only ``K``.  The characters at
    ``-k`` are the conjugates of those at ``k``, so each label at ``-k``
    names the conjugate of a label at ``k``.  Which one is read off the rows:
    the representation induced from a site irrep ``ρ`` carries at ``-k`` the
    conjugate of what the one induced from ``ρ*`` carries at ``k``.  The
    tables' naming, ``KA_n`` the conjugate of ``K_n``, is kept where the rows
    allow it at every label of the point, and otherwise the one correspondence
    the rows allow is taken.  Returns ``(partner, point, pairs)`` for each
    partner, ``pairs`` the ``(label, partner label)`` at that point, sorted,
    the labels in the convention of :func:`ebr_data`.  Returns ``()`` where
    the rows list no such point or the tables hold no rows for the group.
    Raises :class:`~.refusal.InternalError` where a point the rows list is
    not the partner of a listed one with as many labels, or no correspondence
    reproduces the rows, or several do and the tables' naming is not among
    them.  Raises ``ImportError`` where the irreptables package is missing.
    """
    try:
        raw = _spacegroup._ebr_file(number_str, spinor)
    except ImportError:                           # irreptables itself is missing
        raise
    except Exception:                             # no EBR rows: nothing lists a partner
        return ()
    labels = [str(label) for label in raw["basis"]["irrep_labels"]]
    listed = {irr.kpname
              for irr in _spacegroup._irrep_table(number_str, spinor, False).irreps}
    arms = sorted({_label_kpoint(label) for label in labels} - listed)
    if not arms:
        return ()
    physical = _to_physical_convention(raw, number_str, spinor, False)
    labels = [str(label) for label in physical["basis"]["irrep_labels"]]
    matrix = np.array([e["vector"] for e in physical["ebrs"]], dtype=int).T
    partner = list(_ebr_transport(number_str, spinor, False)[2])
    index = {label: i for i, label in enumerate(labels)}
    out = []
    for arm in arms:
        base = arm[:-1] if len(arm) > 1 and arm.endswith("A") else ""
        theirs = [label for label in labels if _label_kpoint(label) == arm]
        ours = [label for label in labels if _label_kpoint(label) == base]
        if base not in listed or len(theirs) != len(ours):
            raise InternalError(
                f"#{number_str}: the elementary band representations of the tables have "
                f"rows at {arm}, which the irrep table does not list, and {arm} is not the "
                "partner -k of a listed point with as many labels. Report it, with the "
                "model and the call.")
        fits = {n: [a for a in theirs
                    if np.array_equal(matrix[index[a]], matrix[index[n], partner])]
                for n in ours}
        named = {n: n.replace(base, arm, 1) for n in ours}
        if all(named[n] in fits[n] for n in ours):
            pi = named
        else:
            every = [dict(zip(ours, pick)) for pick in itertools.product(*fits.values())
                     if len(set(pick)) == len(ours)]
            if len(every) != 1:
                raise InternalError(
                    f"#{number_str} at {arm}: {len(every)} pairings of its labels with the "
                    f"conjugates of the labels at {base} reproduce the rows of the "
                    f"elementary band representations, and the tables' naming, {arm}n the "
                    f"conjugate of {base}n, is not among them. Report it, with the model "
                    "and the call.")
            pi = every[0]
        out.append((arm, base, tuple(sorted(pi.items()))))
    return tuple(out)


class HighSymmetryPoints(dict):
    """The tabulated high-symmetry points of the model's zone, by name.

    What :func:`high_symmetry_points` returns.  It maps each point's name in
    the tables, ``"GM"``, to its reduced coordinates in the model's cell,
    ``dim_k`` of them.  ``summary()`` prints it.

    Attributes
    ----------
    out_of_plane : tuple of str
        If nonempty, the tabulated points out of a layer's plane, which the
        layer's zone does not have: ``A``, ``H`` and ``L`` of a hexagonal
        layer.
    """

    __slots__ = ("out_of_plane",)

    def __init__(self, points=(), *, out_of_plane=()):
        super().__init__(points)
        #: The tabulated names out of the periodic plane.
        self.out_of_plane = tuple(out_of_plane)

    def summary(self) -> str:
        """The points, a line for each, and a line for those out of a layer's plane."""
        lines = [f"{len(self)} point{'' if len(self) == 1 else 's'}, by name, in the model's "
                 "reduced coordinates   ([name])"]
        lines += [f"    {name} {_point_text(k)}" for name, k in self.items()]
        if self.out_of_plane:
            lines.append(f"  out of the layer's plane: {', '.join(self.out_of_plane)}   "
                         "(.out_of_plane)")
        return "\n".join(lines)


def high_symmetry_points(group: GroupData) -> HighSymmetryPoints | Refusal:
    """The tabulated high-symmetry points, in the model's reduced coordinates.

    The points are those of the tables of the irreptables package, carried
    from the conventional cell the tables are written in into the model's
    cell.  Graphene's ``K`` is ``(1/3, 1/3, 0)`` in the tables and
    ``(1/3, 2/3)`` in pythtb's honeycomb cell.  In a group without time
    reversal, a point ``k`` and its partner ``-k`` are inequivalent where no
    operation carries one onto the other, as ``K`` and ``KA`` are, and both
    are returned.  The points are listed ``GM`` first, then alphabetically by
    name, each partner right after its point.

    Parameters
    ----------
    group : GroupData
        The group.  Pass ``r.group`` of what :func:`..groups.realized_group`
        returns.

    Returns
    -------
    HighSymmetryPoints or Refusal
        ``{name: k}``, each ``k`` with ``dim_k`` reduced coordinates.  Its
        ``out_of_plane`` names the tabulated points out of a layer's plane.
        A ``Refusal`` in its place is falsy.

    Refusals
    --------
    A refusal handed in as ``group`` is returned as it is.

    ``tables_not_ready``
        the model's cell could not be matched to the setting the tables are
        written in.

    Raises
    ------
    TypeError
        ``group`` is not ``r.group``: ``r`` itself, ``r.rep``, the ``group``
        of :func:`..srsi.srsi_tables` for a space-group number, which has no
        cell, or anything else.
    """
    _require_group("high_symmetry_points", group)
    return _high_symmetry_points(group)


def _high_symmetry_points(group: GroupData) -> HighSymmetryPoints | Refusal:
    """The points :func:`high_symmetry_points` returns, in the order of :func:`point_order`.

    ``group`` is the group, ``r.group``.  Returns the
    :class:`HighSymmetryPoints`, or a refusal of
    :func:`_points_in_table_order`.
    """
    points = _points_in_table_order(group)
    if isinstance(points, Refusal):
        return points
    return HighSymmetryPoints({name: points[name] for name in sorted(points, key=point_order)},
                              out_of_plane=sorted(points.out_of_plane, key=point_order))


def _points_in_table_order(group: GroupData) -> HighSymmetryPoints | Refusal:
    """The points :func:`high_symmetry_points` returns, in the tables' own order.

    ``group`` is the group, ``r.group``.  Returns the
    :class:`HighSymmetryPoints`, or ``lattice_not_descended`` for a lattice
    supergroup, or ``tables_not_ready`` where the model's cell cannot be
    matched to the tables' setting.  A choice made by walking the points reads
    them in this order: the choice among band labels the
    lowest-angular-momentum rule makes, the naming of a site irrep by the
    little-group representations it induces, and :func:`..ebr.regular_content`,
    which returns the first refusal it meets.
    """
    gate = _lattice_not_descended(group)
    if gate is not None:
        return gate
    if not group._tables_ready:
        return Refusal("tables_not_ready", group._setup_note or f"the cell of {group.name} "
                       f"#{group.number_str} could not be matched to the setting the "
                       "installed tables are written in. Report the model, with this "
                       "reason.", context={"number_str": group.number_str})
    out, out_of_plane = {}, []
    for name, k3 in _table_kpoints(group).items():
        if group._in_plane(k3[None, :], atol=1e-9):
            out[name] = group._k_from_3d(k3[None, :])[0]
        else:
            out_of_plane.append(name)
    return HighSymmetryPoints(out, out_of_plane=out_of_plane)


def _resolve_kpoint(call: str, group: GroupData, k):
    """The coordinates of a k-point given by its tabulated name or its coordinates.

    ``call`` is the public call's name, which opens the ``ValueError`` raised
    for an unknown name.  ``group`` is the group, ``r.group``.  ``k`` is a
    name of :func:`high_symmetry_points` or the point's coordinates.  Returns
    ``k`` as it is where it is not a ``str``, and a name's ``dim_k`` reduced
    coordinates.  Returns the refusal ``out_of_plane`` of :func:`little_group`
    for the name of a tabulated point out of a layer's plane, and a refusal of
    :func:`high_symmetry_points` as it is.  Raises ``ValueError`` for any
    other name, listing the names.
    """
    if not isinstance(k, str):
        return k
    points = _high_symmetry_points(group)
    if isinstance(points, Refusal):
        return points
    if k in points:
        return points[k]
    if k in points.out_of_plane:
        refused = _little_group(group, _table_kpoints(group)[k], call=call)
        if not isinstance(refused, Refusal):        # pragma: no cover: guard
            raise InternalError(
                f"{group.name} #{group.number_str}: {k} is listed as out of the layer's "
                "plane, but its little group was read. Report it, with the model and the "
                "call.")
        return refused
    raise ValueError(f"{call}(): {group.name} #{group.number_str} has no k-point named "
                     f"{k!r}; pass one of {', '.join(points)}, or the point's coordinates.")


def _same_point(k, tabulated, ktol) -> bool:
    """Whether two k-points are one, up to a reciprocal lattice vector.

    ``k`` and ``tabulated`` are reduced coordinates in one cell, and ``ktol``
    the tolerance on each coordinate.  They are one point where their
    difference is a vector of integers, each coordinate within ``ktol``.
    """
    d = np.asarray(k, dtype=float) - np.asarray(tabulated, dtype=float)
    return bool(np.abs(d - np.round(d)).max() <= ktol)


def _resolve_kpname(group: GroupData, lg: LittleGroup, kpname, ktol):
    """The tables' name of a little group's point, or a refusal listing the points.

    ``group`` is the group, ``r.group``, and ``lg`` the :class:`LittleGroup`
    at the point.  ``kpname``, where it is not ``None``, is returned as it
    is.  Otherwise the name is that of the first point of
    :func:`_table_kpoints` that ``lg``'s point equals up to a reciprocal
    lattice vector, each coordinate within ``ktol`` (:func:`_same_point`).
    Returns ``kpoint_not_tabulated`` where there is none, listing the
    tabulated points of the plane in the model's reduced coordinates.  A
    point of the star of a tabulated point, other than its tabulated arm, is
    refused the same way.
    """
    kmap = _table_kpoints(group)
    if kpname is not None:
        return kpname
    for name, k3 in kmap.items():
        if _same_point(lg._k3, k3, ktol):
            return name
    listed = {n: kmap[n] for n in sorted(kmap, key=point_order)}
    avail = {n: (np.round(group._k_from_3d(v[None, :])[0], 6) + 0.0).tolist()
             for n, v in listed.items() if group._in_plane(v[None, :])}
    return Refusal(
        "kpoint_not_tabulated",
        f"k = {(np.round(lg.k, 6) + 0.0).tolist()} is not a tabulated high-symmetry point of "
        f"{group.name} #{group.number_str}; the tables list one member of each point's "
        f"star. The tabulated points, in the model's reduced "
        f"coordinates, are {avail}. Ask one of them, by its name or its coordinates.",
        context={"k": lg.k, "available": avail, "all_points": listed})


def _operations(table) -> tuple:
    """Each operation of an irrep table as ``(R, t, S, antiunitary)``, in its order.

    ``table`` is an irrep table of the irreptables package.  ``R``, ``t`` and
    ``S`` are each operation's rotation, translation and spinor matrix as the
    table holds them, as arrays, and ``antiunitary`` whether it is combined
    with time reversal.
    """
    return tuple((np.asarray(s.R, dtype=float), np.asarray(s.t, dtype=float),
                  np.asarray(s.S, dtype=complex), bool(s.time_reversal))
                 for s in table.symmetries)


def _as_representative(R, t, S, ops, centering, spinor) -> tuple:
    r"""A unitary operation written as ``σ·{E|m}·g_ℓ``, ``g_ℓ`` an operation of the table.

    ``R``, ``t`` and ``S`` are the operation's rotation, translation and
    spinor matrix, ``ops`` the table's operations as :func:`_operations`
    returns them, ``centering`` the cell's centering, which says which ``m``
    is a lattice translation, and ``spinor`` whether the table is
    double-valued.  Returns ``(ℓ, m, σ)``: ``ℓ`` the one-based index of
    ``g_ℓ`` in ``ops``, ``m`` the lattice translation and ``σ = ±1`` the sign
    of the double group, ``+1`` for a single-valued table.  Raises
    ``_EBRTransportError`` where the operation, its spinor matrix up to sign
    included, is none of the table's times a lattice translation.
    """
    for j, (Rj, tj, Sj, anti) in enumerate(ops):
        if anti or np.abs(R - Rj).max() > 1e-6:
            continue
        m = t - tj
        if not _is_translation(m, centering):
            continue
        if not spinor:
            return j + 1, m, 1.0
        if np.abs(S - Sj).max() < _SU2_TOL:
            return j + 1, m, 1.0
        if np.abs(S + Sj).max() < _SU2_TOL:
            return j + 1, m, -1.0
        break
    raise _EBRTransportError(                          # pragma: no cover: guard
        "the operations the tables list are not closed under products, up to lattice "
        "translations.")


def _herring_data(table, k, centering) -> tuple:
    r"""The data of the Wigner-Herring test at ``k`` that hold in either convention.

    ``table`` is an irrep table, ``k`` a point in reduced coordinates of its
    conventional cell, and ``centering`` the table's centering.  Returns
    ``(H, coset, squares, twist)``:

    * ``H``, the one-based indices of the unitary little group of ``k``, the
      operations with ``R^{-T}k ≡ k`` modulo the reciprocal lattice;
    * ``coset``, the indices of the antiunitary ``a`` with ``-R^{-T}k ≡ k``,
      the antiunitary half of the extended little group: as many as ``H``,
      or none where no antiunitary operation fixes ``k``, as at a ``K``
      whose partner ``KA`` is a different point;
    * ``squares``, mapping each ``a`` of ``coset`` to the ``(ℓ, m, σ)`` of
      :func:`_as_representative` that writes ``a^2 = σ·{E|m}·g_ℓ``, ``σ``
      including ``Θ^2 = -1`` for a double-valued table;
    * ``twist``, mapping each ``h`` of ``H`` to ``a_0^{-1} h a_0`` written the
      same way, ``a_0`` the first of ``coset``.

    The convention of the translations enters only where ``m`` becomes the
    phase ``e^{s·2πi k·m}`` (:func:`_corep_gluing`).
    """
    ops = _operations(table)
    spinor = bool(table.spinor)
    k = np.asarray(k, dtype=float)
    H, coset = [], []
    for i, (R, _t, _S, anti) in enumerate(ops):
        d = np.linalg.inv(R).T @ k
        if _is_reciprocal((-d if anti else d) - k, centering):
            (coset if anti else H).append(i + 1)
    squares, twist = {}, {}
    if coset:
        eps = -1.0 if spinor else 1.0                  # Θ^2 = -1 for spin 1/2
        for a in coset:
            R, t, S, _ = ops[a - 1]
            ell, m, sig = _as_representative(R @ R, t + R @ t, S @ S,
                                             ops, centering, spinor)
            squares[a] = (ell, m, eps * sig)
        R0, t0, S0, _ = ops[coset[0] - 1]
        inv = np.linalg.inv(R0)
        for h in H:
            Rh, th, Sh, _ = ops[h - 1]
            ell, m, sig = _as_representative(
                inv @ Rh @ R0, inv @ (Rh @ t0 + th - t0),
                np.linalg.inv(S0) @ Sh @ S0, ops, centering, spinor)
            twist[h] = (ell, m, sig)
    return tuple(H), tuple(coset), squares, twist


def _corep_gluing(chars, k, H, squares, twist, s):
    r"""The coreps at one k-point, glued from the unitary irreps in the convention ``s``.

    ``chars`` maps each irrep label of the unitary little group to its
    characters ``{index: χ}`` in that convention, ``k`` is the point, in
    reduced coordinates of the table's conventional cell, ``H``, ``squares``
    and ``twist`` are as :func:`_herring_data` returns them, and ``s`` is
    ``+1`` for the tables' convention ``e^{+ik·t}`` and ``-1`` for the
    package's ``e^{-ik·t}``.  Each irrep ``D`` has a conjugate ``\bar D`` and
    a Herring indicator ``I_s(D)``,

    .. math::
        \bar D(h) = \overline{D(a_0^{-1} h a_0)},\qquad
        I_s(D) = \frac{1}{|H|}\sum_{a}\chi_D(a^2),

    ``a`` running over the antiunitary coset, ``a_0`` its first element, and
    each ``{E|m}`` contributing ``e^{s·2πi k·m}``.  ``I_s = +1`` with
    ``\bar D ≅ D`` is type ``a``, the corep ``D``; ``I_s = -1`` with
    ``\bar D ≅ D`` is type ``b``, ``D ⊕ D``; ``I_s = 0`` with ``\bar D ≇ D``
    is type ``c``, ``D ⊕ \bar D``.  Returns the sorted list of
    ``(labels, type)``, ``labels`` the irreps each corep glues and ``type``
    its Wigner type, ``"a"``, ``"b"`` or ``"c"``; or ``None`` where the
    characters fix no single conjugate for an irrep, the pairing is not
    mutual, or the indicator disagrees with the conjugate.  Since every
    ``m`` carries the convention, which irrep doubles can differ between
    ``s = +1`` and ``s = -1``, and at some tabulated points does.
    """
    phase = lambda m: np.exp(s * 2j * np.pi * float(np.asarray(k) @ m))  # noqa: E731
    ind = {}
    for name, row in chars.items():
        ind[name] = sum(sig * phase(m) * row[ell]
                        for ell, m, sig in squares.values()) / len(H)
    partner = {}
    for name, row in chars.items():
        bar = {h: np.conj(sig * phase(m) * row[ell])
               for h, (ell, m, sig) in twist.items()}
        hits = [n for n, r in chars.items()
                if max(abs(bar[h] - r[h]) for h in H) < _CHAR_TOL]
        if len(hits) != 1:
            return None
        partner[name] = hits[0]
    out, seen = [], set()
    for name in sorted(chars):
        if name in seen:
            continue
        w = ind[name]
        if partner[name] == name:
            if abs(abs(w) - 1.0) > 0.2 or abs(w.imag) > 0.2:
                return None
            out.append(((name,), "a") if w.real > 0 else ((name, name), "b"))
            seen.add(name)
        else:
            if abs(w) > 0.2 or partner[partner[name]] != name:
                return None
            out.append((tuple(sorted((name, partner[name]))), "c"))
            seen.update((name, partner[name]))
    return sorted(out)


def _corep_name(parts, double: bool) -> str:
    """The name of a corep, spelled from the irreps it glues as the tables do.

    ``parts`` are the labels of the irreps the corep glues, and ``double``
    whether the table is double-valued.  Returns their names without the
    ``-`` marker, joined in lexicographic order, with one ``-`` in front for
    a double-valued table: ``('-A7', '-A8')`` gives ``'-A7A8'``, and
    ``('-A9', '-A11')`` gives ``'-A11A9'``.  The rule gives every corep name
    of the grey tables from that corep's own irreps.
    """
    bare = sorted(p[1:] if str(p).startswith("-") else str(p) for p in parts)
    return ("-" if double else "") + "".join(bare)


def _decompose_into(chars, H, vector):
    """The irreps a character vector is a sum of, each as often as it occurs.

    ``chars`` maps each irrep label to its characters ``{index: χ}``, ``H``
    lists the operation indices read, and ``vector`` is the character vector
    ``{index: χ}`` to decompose.  Returns the labels, each repeated by its
    multiplicity, ``('A', 'A', 'B')``.  Returns ``None`` where the vector is
    no sum of the characters with nonnegative integer multiplicities, within
    ``_CHAR_TOL``.
    """
    names = sorted(chars)
    a = np.array([[complex(chars[n][h]) for h in H] for n in names]).T
    b = np.array([complex(vector[h]) for h in H])
    n, *_ = np.linalg.lstsq(a, b, rcond=None)
    if (np.abs(a @ n - b).max() > _CHAR_TOL
            or np.abs(n - np.round(n.real)).max() > _CHAR_TOL
            or np.round(n.real).min() < -0.5):
        return None
    return tuple(x for name, m in zip(names, np.round(n.real).astype(int))
                 for x in [name] * int(m))


def _parent_table(number_str: str, spinor: bool):
    """The irrep table of a grey group's parent group, without time reversal.

    ``number_str`` is the grey group's number, ``"80.30"``, of which the part
    before the dot is read, and ``spinor`` whether the table is
    double-valued.  Returns the table of the irreptables package for that
    space group without time reversal.
    """
    return _spacegroup._irrep_table(str(number_str).split(".")[0], spinor, False)


@lru_cache(maxsize=None)
def _unitary_conjugation(number_str: str, spinor: bool) -> tuple:
    """What :func:`_table_conjugation` returns for a grey group's parent table.

    ``number_str`` and ``spinor`` are as :func:`_parent_table` takes them.
    What is returned is kept for each pair of arguments and shared between
    calls, so it is read and never changed.
    """
    return _table_conjugation(_parent_table(number_str, spinor))


def _lifted_conjugation(table) -> dict:
    r"""The name each corep of a grey table is returned under, from its irreps.

    ``table`` is an irrep table of the irreptables package.  Returns
    ``{k-point name: {label: label returned}}``.  A corep's name spells the
    irreps of the parent group it glues, and the name returned spells the
    names those irreps are returned under, the map ``c`` of the parent's own
    :func:`_table_conjugation`.  At each tabulated point:

    1. each corep's characters are decomposed into the parent's irreps,
       which must spell its name;
    2. the gluing is checked against the Herring indicators of the tables'
       convention, :func:`_corep_gluing` with ``s = +1``;
    3. each irrep is mapped through ``c`` and the corep named again by
       :func:`_corep_name`;
    4. that gluing is checked against the Herring indicators of the
       package's convention, ``s = -1``.

    Where the name returned is not one the tables ship, the point is re-glued:
    the same irreps are partitioned into coreps otherwise, and an irrep can
    sit in a corep of another dimension.  Returns ``{}`` for a table without
    antiunitary operations, or one whose operations are not twice as many as
    its parent table's, the first half the parent's.
    A point is left out where the parent table does not list it, holds it in
    ``broken``, or has another little group there, or where a step above
    fails.
    """
    if not getattr(table, "au_symmetries", ()):
        return {}
    parent = _parent_table(table.number_str, table.spinor)
    pops, gops = _operations(parent), _operations(table)
    if len(gops) != 2 * len(pops):
        return {}
    for (R1, t1, S1, _), (R2, t2, S2, _) in zip(gops[:len(pops)], pops):
        if (np.abs(R1 - R2).max() > 1e-6 or np.abs(t1 - t2).max() > _LATTICE_TOL
                or np.abs(S1 - S2).max() > _SU2_TOL):
            return {}                        # the first half is not the parent's
    centering = _centering(table)
    double = bool(table.spinor)
    by_kp_u: dict[str, list] = {}
    for irr in parent.irreps:
        by_kp_u.setdefault(irr.kpname, []).append(irr)
    by_kp_g: dict[str, list] = {}
    for irr in table.irreps:
        by_kp_g.setdefault(irr.kpname, []).append(irr)

    out: dict[str, dict] = {}
    c_u, _conj_u, broken_u = _unitary_conjugation(table.number_str, table.spinor)
    for kp, girreps in by_kp_g.items():
        if kp not in by_kp_u or kp in broken_u:
            continue
        uirreps = by_kp_u[kp]
        k = np.asarray(uirreps[0].k, dtype=float)
        H, coset, squares, twist = _herring_data(table, k, centering)
        if sorted(H) != sorted(int(i) for i in uirreps[0].characters):
            continue
        bcs = {irr.name: {int(i): complex(v) for i, v in irr.characters.items()}
               for irr in uirreps}
        parts = {}
        for irr in girreps:
            got = _decompose_into(bcs, H, {int(i): complex(v)
                                           for i, v in irr.characters.items()})
            if got is None or _corep_name(got, double) != irr.name:
                parts = None
                break
            parts[irr.name] = got
        if parts is None:
            continue
        if not coset:                        # nothing glued: coreps are the irreps
            if sorted(parts.values()) != sorted((n,) for n in bcs):
                continue
            out[kp] = {name: _corep_name([c_u[p] for p in got], double)
                       for name, got in parts.items()}
            continue
        if len(coset) != len(H):
            continue
        want = _corep_gluing(bcs, k, H, squares, twist, +1)   # step 2
        if want is None or (sorted(v for v, _ in want)
                            != sorted(parts.values())):
            continue                         # not the gluing the tables' indicators give
        lift = {name: _corep_name([c_u[p] for p in got], double)
                for name, got in parts.items()}               # step 3
        physchars = {name: {h: np.conj(bcs[c_u[name]][h]) for h in H}
                     for name in bcs}
        phys = _corep_gluing(physchars, k, H, squares, twist, -1)   # step 4
        if phys is None or (sorted(_corep_name(v, double) for v, _ in phys)
                            != sorted(lift.values())):
            continue
        out[kp] = lift
    return out


def _matching(chi: dict, image) -> dict | None:
    """The row each row of a table is carried onto by a map of its characters.

    ``chi`` maps each label to its character vector, an array, and ``image``
    is a map of one such vector.  Returns ``{name: other}`` with
    ``chi[other]`` equal to ``image(chi[name])`` within ``_CHAR_TOL``, or
    ``None`` where any row has no such partner or more than one.
    :func:`_table_conjugation` calls it with the complex conjugation, and
    with the conjugation divided by ``φ(g) = e^{-4πi k·t_g}``, which gives
    the tabulated pairing.
    """
    out: dict[str, str] = {}
    for name, vec in chi.items():
        want = image(vec)
        hit = [other for other, w in chi.items()
               if np.abs(w - want).max() < _CHAR_TOL]
        if len(hit) != 1:
            return None
        out[name] = hit[0]
    return out


def _table_conjugation(table) -> tuple[dict, frozenset, tuple]:
    r"""The label each physical row of a table is returned under.

    ``table`` is an irrep table of the irreptables package.  At a point
    ``k``, ``φ(g) = e^{-4πi k·t_g}``, and the tabulated factor system is
    ``ω_0(R, S) = e^{2πi (R^T k - k)·t_S}``.  Returns
    ``(c, conjugated, broken)``:

    * ``c`` maps each tabulated label ``D`` to the label of the physical
      representation whose characters are the conjugates of ``D``'s,
      ``χ_phys[c(D)] = conj(χ_table[D])``.  For a grey table the label can
      be one the tables do not ship (:func:`_lifted_conjugation`);
    * ``conjugated``, a frozenset, holds the points at which ``c`` is not
      the tabulated pairing.  There :func:`_transported_table` returns the
      whole-character conjugate;
    * ``broken`` holds, sorted, the points at which the irreps disagree on
      the operations of their little group, or no rule below applies while
      ``ω_0`` and ``φ`` are both real.

    At each point the first rule that applies names the rows: the names
    :func:`_lifted_conjugation` returns; the shipped label whose characters
    are ``conj(χ_D)``, so that the name follows the printed character; the
    tabulated pairing, the label whose characters are ``conj(χ_D)/φ``; and
    the identity, where ``ω_0`` or ``φ`` is not real.
    """
    t_of = {i + 1: np.asarray(s.t, dtype=float) for i, s in enumerate(table.symmetries)}
    rot_of = {i + 1: np.asarray(s.R, dtype=float) for i, s in enumerate(table.symmetries)}
    by_kp: dict[str, list] = {}
    for irr in table.irreps:
        by_kp.setdefault(irr.kpname, []).append(irr)
    lift = _lifted_conjugation(table)

    conj: dict[str, str] = {}
    conjugated: list[str] = []
    broken: list[str] = []
    for kp, irreps in by_kp.items():
        k = np.asarray(irreps[0].k, dtype=float)
        inds = sorted(irreps[0].characters)
        if any(sorted(irr.characters) != inds for irr in irreps):
            broken.append(kp)                 # ragged little groups within one k
            continue
        phase = np.array([np.exp(-4j * np.pi * k.dot(t_of[i])) for i in inds])
        chi = {irr.name: np.array([complex(irr.characters[i]) for i in inds])
               for irr in irreps}
        pairs = _matching(chi, lambda vec: np.conj(vec) / phase)
        plain = _matching(chi, np.conj)
        if kp in lift:                        # the corep naming the unitary table forces
            here = lift[kp]
        elif plain is not None:               # the name follows the printed character
            here = plain
        elif pairs is not None:               # the tabulated labeling is physical
            here = pairs
        else:
            omega = max(abs(np.sin(2 * np.pi * (rot_of[i].T @ k - k).dot(t_of[j])))
                        for i in inds for j in inds)
            if (omega <= _PHASE_REAL_TOL
                    and np.abs(phase.imag).max() <= _PHASE_REAL_TOL):
                broken.append(kp)             # no convention difference to blame
                continue
            here = {name: name for name in chi}
        conj.update(here)
        if here != pairs:                     # not the tabulated pairing
            conjugated.append(kp)
    return conj, frozenset(conjugated), tuple(sorted(set(broken)))


@lru_cache(maxsize=None)
def _transport_naming(number_str: str, spinor: bool, magnetic: bool) -> tuple:
    """The points of one table conjugated whole, and the label of each row.

    ``number_str``, ``spinor`` and ``magnetic`` pick the table of the
    irreptables package: the group's number, whether the table is
    double-valued, and whether it is a magnetic table.  Returns
    ``(conjugated, pairs)``, ``conjugated`` and ``c`` of
    :func:`_table_conjugation`, ``c`` as sorted ``(label, image)`` pairs.
    What is returned depends on the table alone, and is kept for each table.
    """
    conj, conjugated, _broken = _table_conjugation(
        _spacegroup._irrep_table(number_str, spinor, magnetic))
    return conjugated, tuple(sorted(conj.items()))


def _conjugated_kpoints(number_str: str, spinor: bool, magnetic: bool) -> frozenset:
    """The points of one table whose characters are conjugated whole.

    ``number_str``, ``spinor`` and ``magnetic`` pick the table, as
    :func:`_transport_naming` takes them.  Returns the frozenset of point
    names ``conjugated`` of :func:`_table_conjugation`.
    """
    return _transport_naming(number_str, spinor, magnetic)[0]


@lru_cache(maxsize=None)
def _divergent_kpoints(number_str: str, spinor: bool, magnetic: bool) -> frozenset:
    """The points of one table whose rows returned are not the shipped rows.

    ``number_str``, ``spinor`` and ``magnetic`` pick the table, as
    :func:`_transport_naming` takes them.  Returns the frozenset of point
    names at which the shipped character vectors are not closed under
    complex conjugation.  There the set of rows returned is the entrywise
    conjugate of the shipped set.  The shipped rows describe the other member
    ``-k`` of the star.  A point whose irreps disagree on their little group is
    left out, being ``broken`` in :func:`_table_conjugation`.
    """
    table = _spacegroup._irrep_table(number_str, spinor, magnetic)
    by_kp: dict[str, list] = {}
    for irr in table.irreps:
        by_kp.setdefault(irr.kpname, []).append(irr)
    out = []
    for kp, irreps in by_kp.items():
        inds = sorted(irreps[0].characters)
        if any(sorted(irr.characters) != inds for irr in irreps):
            continue                          # ragged: already `broken`
        chi = {irr.name: np.array([complex(irr.characters[i]) for i in inds])
               for irr in irreps}
        if _matching(chi, np.conj) is None:
            out.append(kp)
    return frozenset(out)


def _transported_table(group: GroupData, kpname: str, k3, ktol) -> dict | Refusal:
    r"""The character table at a tabulated point, in the package's convention.

    ``group`` is the group, ``kpname`` the point's name in the tables, ``k3``
    its three reduced coordinates in the model's three-dimensional cell, and
    ``ktol`` how far ``k3`` may sit from the tabulated point and be it, in
    reduced reciprocal coordinates.  Returns ``{label: {index: character}}``,
    the character of each little-group representation on each unitary
    operation of the little group, by its one-based index in the group's
    operations.

    The tables represent a lattice translation by ``e^{+ik·t}`` and the
    package by ``e^{-ik·t}``, and the difference is in the factor system
    alone.  So the point-group part ``χ_0 = χ_table e^{-2πi k·t_table}`` of
    each tabulated character is kept, and the phase of the operation's
    translation in the model's cell ``t_cell`` is attached,
    ``χ = χ_0 e^{-2πi k·t_cell}``, both translations and ``k`` in the
    conventional cell of the tables.  At a point of ``conjugated`` of
    :func:`_table_conjugation` the whole character is conjugated instead,
    ``χ = conj(χ_0) e^{-2πi k·t_cell}``, under the label ``c(D)`` that ``c``
    of :func:`_table_conjugation` gives the tabulated label ``D``.  Each
    character is multiplied by the operation's ``sign``, which is ``+1`` on
    every group the package builds.  At a partner point ``-k`` of a group
    without time reversal (:func:`_partner_points`), the characters are the
    conjugates of those at the listed point, under the partner labels.

    Returns ``kpoint_not_tabulated`` where ``k3`` is not the tabulated point
    within ``ktol``, or where the table lists no point named ``kpname``.
    """
    # A point -k the irrep table does not list: its little group is the listed
    # point's, and its characters are the listed point's conjugated, each under
    # the partner label the EBR rows give the conjugate (`_partner_points`)
    if not group._magnetic:
        for arm, base, naming in _partner_points(group.number_str, group.spinor):
            if kpname != arm:
                continue
            rows = _transported_table(group, base, (-np.asarray(k3, dtype=float)) % 1.0, ktol)
            if isinstance(rows, Refusal):
                return rows
            naming = dict(naming)
            return {naming[label]: {ind: complex(np.conj(c)) for ind, c in row.items()}
                    for label, row in rows.items()}
    table = _spacegroup._irrep_table(group.number_str, group.spinor, group._magnetic)
    sg = group._sg
    conjugated = kpname in _conjugated_kpoints(group.number_str, group.spinor,
                                               group._magnetic)
    naming = dict(_transport_naming(group.number_str, group.spinor,
                                    group._magnetic)[1])
    k2 = np.round(np.asarray(k3, dtype=float), 5) % 1
    carried = _table_kpoints(group).get(kpname)
    out: dict[str, dict[int, complex]] = {}
    for irr in table.irreps:
        if irr.kpname != kpname:
            continue
        k_ref = np.asarray(irr.k, dtype=float)
        # the point as the tables hold it, carried exactly into the cell as written
        if not _same_point(k3, carried, ktol):
            return Refusal(
                "kpoint_not_tabulated",
                f"k = {np.round(group._k_from_3d(np.asarray(k3, dtype=float)[None, :])[0], 6)} "
                f"is not the tabulated point {kpname} of {group.name} #{group.number_str}, "
                f"which is at {np.round(group._k_from_3d(carried[None, :])[0], 6)} in the "
                f"model's reduced coordinates. Report it, with the model and the call.",
                context={"kpname": kpname, "k": k2,
                         "number_str": str(group.number_str)})
        row: dict[int, complex] = {}
        for i, (sym, sym_table) in enumerate(zip(sg.symmetries, table.symmetries)):
            ind = i + 1
            if ind not in irr.characters:
                continue                      # not in the little group (or antiunitary)
            t_table = np.asarray(sym_table.t, dtype=float)
            t_ours = sym.translation_refUC(group._refUC, group._shiftUC)
            #  `χ_0 = χ_table e^{-2πi k·t_table}`; `χ = χ_0 e^{-2πi k·t_cell}`
            if conjugated:                    # ... with `χ_0 → conj(χ_0)`, name `c(D)`
                row[ind] = (np.conj(complex(irr.characters[ind])
                                    * np.exp(-2j * np.pi * t_table.dot(k_ref)))
                            * sym.sign * np.exp(-2j * np.pi * t_ours.dot(k_ref)))
            else:
                row[ind] = (complex(irr.characters[ind]) * sym.sign
                            * np.exp(-2j * np.pi * (t_table + t_ours).dot(k_ref)))
        out[naming.get(irr.name, irr.name) if conjugated else irr.name] = row
    if not out:
        return Refusal(
            "kpoint_not_tabulated",
            f"the tables list no point named {kpname} for {group.name} "
            f"#{group.number_str}. Ask a point high_symmetry_points returns.",
            context={"kpname": kpname, "k": k2,
                     "number_str": str(group.number_str)})
    return out


def corep_table(group: GroupData, k) -> dict | Refusal:
    """The character table of the little group at a high-symmetry point.

    The table holds the character of each little-group representation at
    ``k`` on each unitary operation of the little group.  The characters are
    those of the tables of the irreptables package, carried onto the
    operations of the model's cell.  They are in the convention where a
    lattice translation ``t`` acts as ``e^{-ik·t}``, the one
    :func:`..sewing.characters` measures in.  At a partner point ``-k`` whose
    characters the tables do not list, they are the complex conjugates of
    those at ``k``.  ``ebr_data(group).convention`` names the points at which
    the labels and characters returned differ from the tables'.

    Parameters
    ----------
    group : GroupData
        The group.  Pass ``r.group`` of what :func:`..groups.realized_group`
        returns.
    k : str or array, shape (dim_k,)
        The point.  Pass its name in the tables, ``"X"``, or its reduced
        coordinates, as :func:`high_symmetry_points` returns them.  A layer
        also takes the three reduced coordinates of a point of its plane in
        its three-dimensional cell.

    Returns
    -------
    dict or Refusal
        ``{label: {index: character}}``, an entry for each little-group
        representation at ``k``, with its character on each unitary operation
        of the little group, keyed by the operation's index.  A ``Refusal`` in
        its place is falsy.

    Refusals
    --------
    A refusal handed in as ``group`` is returned as it is.

    ``tables_not_ready``
        the model's cell could not be matched to the setting the tables are
        written in.
    ``out_of_plane``
        ``k`` has a component out of a layer's plane, or names a tabulated
        point out of the plane.
    ``kpoint_not_tabulated``
        ``k`` is no tabulated point in the model's cell, or is a member of a
        tabulated point's star other than the one tabulated.  The reason
        lists the tabulated points.

    Raises
    ------
    TypeError
        ``group`` is not ``r.group``: ``r`` itself, ``r.rep``, the ``group``
        of :func:`..srsi.srsi_tables` for a space-group number, which has no
        cell, or anything else.
    ValueError
        ``k`` is a name the group does not tabulate, or has neither
        ``dim_k`` coordinates nor three.

    See Also
    --------
    identify_from_characters : the little-group representations each
        multiplet carries, read from its characters against this table.
    """
    _require_group("corep_table", group)
    k = _resolve_kpoint("corep_table", group, k)
    if isinstance(k, Refusal):
        return k
    return _corep_table(group, k, call="corep_table")


def _corep_table(group: GroupData, k, *, kpname=None, ktol=1e-5,
                 call: str = "corep_table") -> dict | Refusal:
    """The character table :func:`corep_table` returns, at coordinates ``k``.

    ``group`` is the group and ``k`` the point's reduced coordinates.
    ``kpname`` is the point's tabulated name, taken as given instead of found
    from ``k``; ``ktol`` how far ``k`` may sit from a tabulated point and be
    it, in reduced reciprocal coordinates; ``call`` names the public call in
    a ``ValueError`` on the number of ``k``'s components.  Returns what
    :func:`corep_table` returns, and ``lattice_not_descended`` for a lattice
    supergroup.
    """
    gate = _lattice_not_descended(group)
    if gate is not None:
        return gate
    if not group._tables_ready:
        return Refusal("tables_not_ready",
                       group._setup_note or f"the cell of {group.name} "
                       f"#{group.number_str} could not be matched to the setting the "
                       "installed tables are written in. Report the model, with this "
                       "reason.",
                       context={"number_str": group.number_str})
    lg = _little_group(group, k, call=call)
    if isinstance(lg, Refusal):
        return lg
    name = _resolve_kpname(group, lg, kpname, ktol)
    if isinstance(name, Refusal):
        return name
    return _transported_table(group, name, lg._k3 % 1.0, ktol)


def identify_from_characters(group: GroupData, k, characters, *, multiplet_dims=None,
                             label_tol=LABEL_TOL) -> CorepIdentification | Refusal:
    r"""The little-group representations each multiplet carries, from its characters.

    Each row of characters ``χ`` is expanded over the little-group
    representations ``ρ`` at ``k``, whose characters ``χ_ρ`` are those
    :func:`corep_table` returns.  The multiplicity of ``ρ`` is
    :math:`m_\rho = \sum_g \chi_\rho(g)\,\chi(g)^* / \sum_g |\chi_\rho(g)|^2`,
    the sums over the unitary operations ``g`` of the little group.  A
    multiplicity within ``label_tol`` of a nonnegative integer is read as that
    integer.  A multiplet in which levels of several little-group
    representations are degenerate reads as the sum of those representations.

    Parameters
    ----------
    group : GroupData
        The group the characters were measured in.  Pass ``r.group`` of what
        :func:`..groups.realized_group` returns.
    k : str or array, shape (dim_k,)
        The point.  Pass its name in the tables, ``"X"``, or its reduced
        coordinates, as :func:`high_symmetry_points` returns them.  A layer
        also takes the three reduced coordinates of a point of its plane in
        its three-dimensional cell.
    characters : array, shape (n_multiplets, n_unitary)
        A row for each degenerate multiplet and a column for each unitary
        operation of the little group, in the order of
        ``little_group(group, k).unitary``.  An entry is the trace of the
        operation's sewing matrix over the multiplet's states, as
        :func:`..sewing.characters` measures it.
    multiplet_dims : array, shape (n_multiplets,), optional
        The degeneracy of each multiplet.
    label_tol : float, default 1e-3
        The tolerance for reading a multiplicity as an integer.

    Returns
    -------
    CorepIdentification or Refusal
        The little-group representations of each multiplet and its
        dimension, and the multiplicities before rounding.  ``summary()``
        prints it.  A ``Refusal`` in its place is falsy.

    Refusals
    --------
    A refusal handed in as ``group`` is returned as it is.

    ``tables_not_ready``
        the model's cell could not be matched to the setting the tables are
        written in.
    ``out_of_plane``
        ``k`` has a component out of a layer's plane, or names a tabulated
        point out of the plane.
    ``kpoint_not_tabulated``
        ``k`` is no tabulated point in the model's cell, or is a member of a
        tabulated point's star other than the one tabulated.  The reason
        lists the tabulated points.
    ``little_group_mismatch``
        a unitary operation of the little group in the model's cell is not
        among those the tables list at the point.
    ``integer_multiplicity``
        a multiplicity is further than ``label_tol`` from a nonnegative
        integer.
    ``multiplet_dimension``
        with ``multiplet_dims``, the dimensions of the representations found
        in a multiplet do not add up to its degeneracy.

    Raises
    ------
    TypeError
        ``group`` is not ``r.group``: ``r`` itself, ``r.rep``, the ``group``
        of :func:`..srsi.srsi_tables` for a space-group number, which has no
        cell, or anything else.
    ValueError
        ``k`` is a name the group does not tabulate, or has neither
        ``dim_k`` coordinates nor three; ``characters`` has not one column for
        each unitary operation of the little group; ``multiplet_dims`` has
        not one entry for each row.

    See Also
    --------
    coreps_at : the little-group representations of a set of bands at a
        point, measured on the model.
    """
    _require_group("identify_from_characters", group)
    k = _resolve_kpoint("identify_from_characters", group, k)
    if isinstance(k, Refusal):
        return k
    return _identify_from_characters(group, k, characters, multiplet_dims=multiplet_dims,
                                     label_tol=label_tol,
                                     call="identify_from_characters")


def _identify_from_characters(group: GroupData, k, characters, *, multiplet_dims=None,
                              kpname=None, label_tol=1e-3, ktol=1e-5,
                              call: str = "identify_from_characters"
                              ) -> CorepIdentification | Refusal:
    """What :func:`identify_from_characters` returns, at coordinates ``k``.

    ``group``, ``characters``, ``multiplet_dims`` and ``label_tol`` are as
    :func:`identify_from_characters` takes them, and ``k`` is the point's
    reduced coordinates.  ``kpname`` is the point's tabulated name, taken as
    given instead of found from ``k``; ``ktol`` how far ``k`` may sit from a
    tabulated point and be it, in reduced reciprocal coordinates; ``call``
    names the public call in a ``ValueError`` on the number of ``k``'s
    components.  Returns what :func:`identify_from_characters` returns, and
    ``lattice_not_descended`` for a lattice supergroup.
    """
    gate = _lattice_not_descended(group)
    if gate is not None:
        return gate
    if not group._tables_ready:
        return Refusal("tables_not_ready",
                       group._setup_note or f"the cell of {group.name} "
                       f"#{group.number_str} could not be matched to the setting the "
                       "installed tables are written in. Report the model, with this "
                       "reason.",
                       context={"number_str": group.number_str})

    lg = _little_group(group, k, call=call)
    if isinstance(lg, Refusal):
        return lg

    chars = np.atleast_2d(np.asarray(characters, dtype=complex))
    if chars.shape[1] != len(lg.unitary):
        raise ValueError(
            f"{call}(): the number of columns of characters=, {chars.shape[1]}, is not "
            f"the number of unitary operations of the little group at "
            f"k = {np.round(lg.k, 6)}, {len(lg.unitary)}: {lg.unitary}. Pass one column "
            "for each, in that order.")

    kpname = _resolve_kpname(group, lg, kpname, ktol)
    if isinstance(kpname, Refusal):
        return kpname
    table = _corep_table(group, lg.k, kpname=kpname, ktol=ktol, call=call)
    if isinstance(table, Refusal):
        return table

    table_inds = set(next(iter(table.values())).keys())
    if not set(lg.unitary) <= table_inds:
        return Refusal(
            "little_group_mismatch",
            f"the little group at {kpname} holds the unitary operations "
            f"{sorted(lg.unitary)}, and the tables list {sorted(table_inds)} there. Report "
            "it, with the model and the call.",
            context={"ours": sorted(lg.unitary), "table": sorted(table_inds), "kpname": kpname})

    label_order = tuple(table)
    mult = np.zeros((chars.shape[0], len(label_order)), dtype=complex)
    dims = np.zeros(len(label_order))
    # the identity is in every little group; its character is the dimension
    identity = next((int(s.ind) for s in lg._unitary_operations
                     if np.array_equal(s.rotation, np.eye(3, dtype=int))
                     and np.abs(s.translation - np.round(s.translation)).max() < 1e-8),
                    min(lg.unitary))
    for c, lab in enumerate(label_order):
        ir = np.array([table[lab][i] for i in lg.unitary], dtype=complex)
        norm = float((np.abs(ir) ** 2).sum())
        dims[c] = abs(table[lab][identity])
        mult[:, c] = ir @ chars.conj().T / norm

    defect = float(np.abs(mult - np.round(mult.real)).max()) if mult.size else 0.0
    rounded = np.round(mult.real).astype(int)
    bad = (np.abs(mult - rounded) > label_tol) | (rounded < 0)
    if bad.any():
        rows = sorted(set(int(r) for r in np.nonzero(bad)[0]))
        return Refusal(
            "integer_multiplicity",
            f"at {kpname}, the multiplicities of the little-group representations in the "
            f"multiplets {rows} are not nonnegative integers within {label_tol:g}. "
            f"The largest distance from an integer is {defect:.3g}.",
            value=defect, threshold=float(label_tol),
            context={"kpname": kpname, "k": lg.k, "indices": lg.unitary,
                     "labels": label_order, "multiplicities": mult,
                     "characters": chars, "rows": rows})

    total_dims = rounded @ dims
    if multiplet_dims is not None:
        want = np.asarray(multiplet_dims, dtype=float).ravel()
        if want.shape[0] != chars.shape[0]:
            raise ValueError(f"{call}(): the number of entries of multiplet_dims=, "
                             f"{want.shape[0]}, is not the number of rows of characters=, "
                             f"{chars.shape[0]}, one for each multiplet. Pass the "
                             "degeneracy of each multiplet.")
        if not np.allclose(total_dims, want, atol=1e-6):
            return Refusal(
                "multiplet_dimension",
                f"at {kpname}, the little-group representations found in the multiplets "
                f"add up to the dimensions {total_dims.tolist()}, where the multiplets' "
                f"degeneracies are {want.tolist()}.",
                value=float(np.abs(total_dims - want).max()), threshold=1e-6,
                context={"kpname": kpname, "got": total_dims, "expected": want})

    labels = tuple(ordered_content({lab: int(rounded[r, c])
                                    for c, lab in enumerate(label_order)
                                    if rounded[r, c] != 0})
                   for r in range(chars.shape[0]))

    part = np.full(chars.shape[0], -1, dtype=int)
    nxt = 0
    for r in range(chars.shape[0]):
        if part[r] >= 0:
            continue
        part[r] = nxt
        for r2 in range(r + 1, chars.shape[0]):
            if part[r2] < 0 and np.abs(chars[r] - chars[r2]).max() <= label_tol:
                part[r2] = nxt
        nxt += 1

    return CorepIdentification(kpoint=kpname, k=lg.k, indices=lg.unitary, labels=labels,
                               multiplicities=mult, label_order=label_order,
                               dims=total_dims, integrality=defect,
                               _partition=tuple(int(p) for p in part),
                               _kind="irreps" if _magnetic_type(group) == "I" else "coreps")


def _label_kpoint(label: str) -> str:
    """The name of a label's point, as :func:`kpoint_of_label` reads it.

    ``label`` is the label of a little-group representation.  Returns the
    characters before its first digit, a leading ``-`` dropped:
    ``'-GM4GM5'`` gives ``'GM'``.
    """
    s = str(label)[1:] if str(label).startswith("-") else str(label)
    i = 0
    while i < len(s) and not s[i].isdigit():
        i += 1
    return s[:i]


def _conjugate_site_name(ebr_name: str) -> str | None:
    """The EBR name with the site irrep replaced by its complex conjugate.

    ``ebr_name`` is a column's name, ``'1E↑G(1)'``.  The tables name a
    complex-conjugate pair of site irreps ``1X`` and ``2X`` (``1E`` and
    ``2E``, ``-1Eg`` and ``-2Eg``), a leading ``-`` marking a double-valued
    one.  Returns the name with ``1`` and ``2`` exchanged at the head of the
    site irrep, ``'2E↑G(1)'``, and ``None`` where the site irrep starts with
    neither.  A grey group's pair glued into one site irrep, ``1E2E``, gives
    ``2E2E``, a name no column of the tables has, so :func:`_ebr_transport`
    reads it as self-conjugate.  :func:`_ebr_transport` checks every pairing
    on the ``GM`` rows.
    """
    site = str(ebr_name).split("\u2191")[0]
    tail = str(ebr_name)[len(site):]
    sign, rest = ("-", site[1:]) if site.startswith("-") else ("", site)
    if rest[:1] == "1":
        return f"{sign}2{rest[1:]}{tail}"
    if rest[:1] == "2":
        return f"{sign}1{rest[1:]}{tail}"
    return None


def _conjugation_involution(table, labels) -> dict:
    r"""The map ``c`` on a table's labels and on those of its EBR basis.

    ``table`` is an irrep table and ``labels`` the labels of the rows of its
    group's EBR matrix.  Returns ``{label: label}``, ``c`` of
    :func:`_table_conjugation` on every label of the table and of
    ``labels``, ``χ_phys[c(D)] = conj(χ_table[D])``.  A label of ``labels``
    at a partner point ``XA`` the table does not list is mapped through the
    name of its partner at ``X``: ``KA2`` goes where ``K2`` goes, with ``K``
    read as ``KA``.  Raises :class:`_EBRTransportError` where the table has
    ``broken`` points, where a label is neither in the table nor at the
    partner of a point that is, where ``c`` is not one-to-one on ``labels``,
    where ``c(c(D))`` is not ``D`` although ``c(D)`` is a label ``c`` maps,
    or where ``c`` moves a label to another point.  At a re-glued point
    ``c(D)`` is a name the tables do not ship, and ``c`` does not map it.
    """
    conj, _conjugated, broken = _table_conjugation(table)
    if broken:
        raise _EBRTransportError(
            f"the characters the tables list for #{table.number_str} at {list(broken)} "
            "are not closed under complex conjugation, and neither the factor system nor "
            "the phase with which time reversal glues irreps into coreps is complex "
            "there.", points=list(broken))

    for label in labels:                       # the untabulated 'A' arms
        if label in conj:
            continue
        kp = _label_kpoint(label)
        # ``len(kp) > 1`` keeps the hexagonal ``A`` point out of the arm rule.
        base = label.replace(kp, kp[:-1]) if len(kp) > 1 and kp.endswith("A") else None
        if base is None or base not in conj:
            raise _EBRTransportError(
                f"the label {label!r} of the elementary band representations, at "
                f"{kp!r}, is not in the irrep table, and is not at the partner -k of a "
                "point that is.", points=[kp])
        conj[label] = conj[base].replace(kp[:-1], kp)

    images = {conj[label] for label in labels}
    if len(images) != len(labels):
        raise _EBRTransportError(
            f"complex conjugation sends the {len(labels)} labels of the elementary band "
            f"representations to {len(images)} labels.")
    for label in labels:
        image = conj[label]
        if image in conj and conj[image] != label:
            raise _EBRTransportError(
                f"complex conjugation of the labels is not an involution at {label!r}: "
                f"{label} → {image} → {conj.get(image)}.",
                points=[_label_kpoint(label)])
        if _label_kpoint(image) != _label_kpoint(label):
            raise _EBRTransportError(          # pragma: no cover: structural
                f"complex conjugation sends the label {label!r} to {image!r}, at "
                "another point.",
                points=[_label_kpoint(label)])
    return conj


@lru_cache(maxsize=None)
def _ebr_transport(number_str: str, spinor: bool, magnetic: bool) -> tuple:
    r"""The relabeling of the tables' EBR rows into the package's convention.

    ``number_str``, ``spinor`` and ``magnetic`` pick the tables, as
    :func:`_transport_naming` takes them.  Returns
    ``(labels, conjugation, partner, changed_labels, new_labels, perm)``:
    the tables' row labels in their order; ``c`` of
    :func:`_conjugation_involution`; for each column, the index of the column
    induced from the complex-conjugate site irrep ``ρ*`` at the same Wyckoff
    position, the column itself for a real site irrep; the labels that are
    renamed or whose row changes; the row labels returned, in the same
    positions; and the row permutation.  The multiplicity of ``c(D)`` in the
    column of ``ρ`` returned is the tables' multiplicity of ``D`` in the
    column of ``ρ*``, so column ``i`` returned is
    ``matrix[perm, partner[i]]``.  At a re-glued point the labels are renamed
    in place and not permuted.  Raises :class:`_EBRTransportError`, and
    returns nothing transported in part, where an entry has no key
    ``ebr_name`` or ``wyckoff_position``, :func:`_conjugation_involution`
    raises, ``c`` changes a dimension, the pairing of the columns is
    ambiguous or not an involution, a column's ``GM`` rows change, or a
    column's dimension at a point changes.
    """
    raw = _spacegroup._ebr_file(number_str, spinor)
    labels = tuple(str(l) for l in raw["basis"]["irrep_labels"])
    try:
        names = [str(e["ebr_name"]) for e in raw["ebrs"]]
        wyck = [str(e["wyckoff_position"]) for e in raw["ebrs"]]
    except KeyError as exc:                       # e.g. 80.31 writes its positions under `wps`
        raise _EBRTransportError(
            f"an elementary band representation in the tables has no entry {exc}.") from exc
    matrix = np.array([e["vector"] for e in raw["ebrs"]], dtype=int).T

    table = _spacegroup._irrep_table(number_str, spinor, magnetic)
    conj = _conjugation_involution(table, labels)

    degen = {l: int(d) for l, d in zip(labels, raw["basis"]["degeneracies"])}
    bad = [l for l in labels
           if conj[l] in degen and degen[l] != degen[conj[l]]]
    if bad:
        raise _EBRTransportError(
            f"complex conjugation changes the dimension of the labels {bad}.",
            points=[_label_kpoint(l) for l in bad])

    partner = []
    for i, (name, pos) in enumerate(zip(names, wyck)):
        want = _conjugate_site_name(name)
        hits = [j for j, (n2, p2) in enumerate(zip(names, wyck))
                if n2 == want and p2 == pos]
        if want is None or not hits:
            partner.append(i)                     # self-conjugate site irrep
        elif len(hits) == 1:
            partner.append(hits[0])
        else:
            raise _EBRTransportError(
                f"the tables list the elementary band representation {want!r}@{pos} "
                f"of #{number_str} more than once, so {name!r}@{pos} has more than one "
                "partner under time reversal.")
    if any(partner[partner[i]] != i for i in range(len(partner))):
        raise _EBRTransportError(
            f"the pairing of the elementary band representations of #{number_str} "
            "under time reversal is not an involution.")

    index = {l: i for i, l in enumerate(labels)}
    #  Per k-point block: where ``c`` maps the block onto itself the labels stay
    #  put and the multiplicities permute (the ordinary case: the returned rows are
    #  then the shipped rows permuted). Where it does not, at a re-glued point,
    #  the labels are renamed in place.
    block: dict[str, list] = {}
    for i, l in enumerate(labels):
        block.setdefault(_label_kpoint(l), []).append(i)
    new_labels, perm = list(labels), list(range(len(labels)))
    for kp, pos in block.items():
        here = [labels[i] for i in pos]
        if {conj[l] for l in here} == set(here):
            for i in pos:
                perm[i] = index[conj[labels[i]]]
        else:
            for i in pos:
                new_labels[i] = conj[labels[i]]
    new_labels, perm = tuple(new_labels), np.array(perm)

    gamma = [l for l in labels if _label_kpoint(l) == "GM"]
    for i in range(matrix.shape[1]):
        for l in gamma:
            if matrix[index[l], i] != matrix[index[conj[l]], partner[i]]:
                raise _EBRTransportError(
                    f"the rows at GM of {names[i]}@{wyck[i]} change when it is carried "
                    "into the package's convention.",
                    points=["GM"])

    #  Dimension conservation, k-point by k-point and column by column: the band
    #  representation has the same number of states at every k in either
    #  convention, whether or not the corep basis there was re-glued.
    deg = np.array([degen[l] for l in labels], dtype=int)
    for i in range(matrix.shape[1]):
        col = matrix[perm, partner[i]]
        for kp, pos in block.items():
            was = int(sum(matrix[j, i] * deg[j] for j in pos))
            now = int(sum(col[j] * deg[j] for j in pos))
            if was != now:
                raise _EBRTransportError(
                    f"carried into the package's convention, {names[i]}@{wyck[i]} has "
                    f"{now} bands at {kp}, where the tables give {was}.", points=[kp])

    moved = tuple(l for l, l2, j in zip(labels, new_labels, perm)
                  if l != l2 or any(matrix[index[l], i] != matrix[j, partner[i]]
                                    for i in range(matrix.shape[1])))
    return labels, conj, tuple(partner), moved, new_labels, tuple(int(p) for p in perm)


def _to_physical_convention(raw: dict, number_str: str, spinor: bool,
                            magnetic: bool) -> dict:
    """The tables' EBR record carried into the package's convention.

    ``raw`` is the record of the irreptables package for the group, and
    ``number_str``, ``spinor`` and ``magnetic`` pick the tables, as
    :func:`_transport_naming` takes them.  Returns a new mapping, ``raw``
    only read and what is not rewritten shared with it.  Rewritten are each
    column's ``vector`` and ``irrep_list``, by :func:`_ebr_transport`; the
    row labels ``irrep_labels``, where a point is re-glued; and the Smith
    factors ``u`` and ``v``, permuted with the rows and the columns.  With
    ``M' = P M Q`` for the permutation ``P`` of the rows and ``Q`` of the
    columns, both involutions, ``u M v = r`` gives ``(u P) M' (Q v) = r``,
    and the diagonal ``r`` is unchanged.  Raises
    :class:`_EBRTransportError` where :func:`_ebr_transport` raises, and
    where a factor to be permuted is not a matrix.
    """
    labels, conj, partner, moved, new_labels, perm = _ebr_transport(
        number_str, spinor, magnetic)
    perm = np.array(perm)
    matrix = np.array([e["vector"] for e in raw["ebrs"]], dtype=int).T

    out = dict(raw)
    out["basis"] = dict(raw["basis"])
    if new_labels != labels:
        out["basis"]["irrep_labels"] = list(new_labels)
    out["ebrs"] = []
    for i, entry in enumerate(raw["ebrs"]):
        col = matrix[perm, partner[i]]
        new = dict(entry)
        new["vector"] = [float(v) for v in col]
        new["irrep_list"] = [l for l, m in zip(new_labels, col)
                             for _ in range(int(m))]
        out["ebrs"].append(new)

    # ``r``, the elementary divisors, is invariant under both permutations, and
    # each of ``u``/``v`` is rewritten only when its permutation is not the
    # identity: ``irreptables`` stores a degenerate Smith factor as a scalar when
    # the group has a single EBR column (SG 4, 7, 9, 29 and others), and there
    # the column permutation is the identity anyway.
    smith = raw.get("smith_form")
    if smith is not None:
        moves = {"u": (list(perm) != list(range(len(labels))), perm, 1),
                 "v": (list(partner) != list(range(len(raw["ebrs"]))),
                       list(partner), 0)}
        if any(m[0] for m in moves.values()):
            out["smith_form"] = dict(smith)
            for key, (needed, order, axis) in moves.items():
                if not needed:
                    continue
                arr = np.asarray(smith[key], dtype=int)
                if arr.ndim != 2:
                    raise _EBRTransportError(
                        f"the factor {key!r} of the Smith normal form the tables hold "
                        f"for #{number_str} is an array of {arr.ndim} dimensions, not a "
                        "matrix.")
                out["smith_form"][key] = (arr[:, order] if axis else
                                          arr[order, :]).tolist()
    return out


def _transport_record(number_str: str, spinor: bool, magnetic: bool) -> dict:
    """The record of what the convention moved in a group's EBR rows.

    ``number_str``, ``spinor`` and ``magnetic`` pick the tables, as
    :func:`_transport_naming` takes them.  Returns a mapping with the keys
    ``name``, ``applied``, ``changed_labels``, ``changed_points``,
    ``conjugated_points``, ``divergent_points``, ``reglued_points``,
    ``conjugation``, ``labels_bcs``, ``labels_physics_convention``,
    ``row_permutation``, ``tr_partner`` and ``note``, from which
    :func:`ebr_data` builds its ``convention``.  Points are in the package's
    order of points.
    """
    labels, conj, partner, moved, new_labels, _perm = _ebr_transport(
        number_str, spinor, magnetic)
    reglued = tuple(sorted({_label_kpoint(l) for l, l2 in zip(labels, new_labels)
                            if l != l2}, key=point_order))
    return {
        "name": "physical (translations as e^{-ik.t})",
        "applied": True,
        "changed_labels": moved,
        "changed_points": tuple(sorted({_label_kpoint(l) for l in moved},
                                       key=point_order)),
        "conjugated_points": tuple(sorted(
            _conjugated_kpoints(number_str, spinor, magnetic), key=point_order)),
        "divergent_points": tuple(sorted(
            _divergent_kpoints(number_str, spinor, magnetic), key=point_order)),
        "reglued_points": reglued,
        "conjugation": {l: conj[l] for l in labels},
        "labels_bcs": labels,
        "labels_physics_convention": new_labels,
        "row_permutation": tuple(int(p) for p in _perm),
        "tr_partner": tuple(partner),
        "note": ("EBR rows transported out of BCS's e^{+ik.t} convention; label "
                 "pairings at non-TRIM points therefore differ from BCS's shipped "
                 "BANDREP output. At `conjugated_points` the little-group "
                 "characters are returned as whole-character conjugates, but under "
                 "the BCS name of the row that comes out, so the printed table is "
                 "BCS's own and only which label a band lands on differs. At "
                 "`divergent_points` not even that is possible -- the shipped page "
                 "there tabulates the other star member -k -- so the printed rows "
                 "are not BCS's; and at `reglued_points` (a subset, always grey) "
                 "the time-reversal gluing itself differs, so the corep basis is a "
                 "different partition of the same unitary irreps and a given irrep "
                 "can sit in a corep of a different dimension. See "
                 "docs/background/conventions.md, 'Reading across to BCS'."),
    }


@lru_cache(maxsize=None)
def _improper_letters(sg: int) -> frozenset:
    r"""The Wyckoff letters of space group ``sg`` whose site group is improper.

    ``sg`` is the space-group number.  Returns a frozenset of letters.  A
    letter's site group is every operation of the group that fixes its
    representative point at a generic value of its free parameters, the point
    and the operations both ``pyxtal``'s in the tables' setting
    (:func:`_pyxtal_wyckoff`: the general position's operations, the
    conventional cell's centering among them).  It is named by
    :func:`.._sitechars.point_group_of`, and improper is that name starting
    with a bar: ``-1``, ``-3``, ``-4``, ``-6``, ``-3m``, ``-42m``, ``-6m2`` or
    ``-43m``.
    """

    positions, _centering = _pyxtal_wyckoff(int(sg))
    general = max(positions, key=lambda p: p["multiplicity"])
    out = set()
    for position in positions:
        A, b, _P = position["arms"][0]
        point = A @ _GENERIC_PARAMETER + b
        fixing = []
        for R, t, _Q in general["arms"]:
            delta = R @ point + t - point
            if float(np.abs(delta - np.rint(delta)).max()) <= 1e-6:
                fixing.append(R)
        if _sitechars.point_group_of(fixing).startswith("-"):
            out.add(position["letter"])
    return frozenset(out)


def _barred(position: str, improper) -> str:
    """An EBR position with the bar written on its improper site group.

    ``position`` is a position as the tables' EBR files write it,
    ``"3a(3m1',3m)"``, and ``improper`` the letters
    :func:`_improper_letters` returns for the group.  The files write every
    improper site group without its bar: ``3m`` for the ``-3m`` at ``3a`` of
    ``R-3m``, ``6m2`` for the ``-6m2`` at ``2c`` of ``P6/mmm``.  Returns the
    position with a bar on each symbol of its site groups where its letter
    is in ``improper``, ``"3a(-3m1',-3m)"``, and the position as it is
    otherwise.  The bar goes on the tables' own symbol, which keeps the
    orientation that symbol writes, and so which irrep is ``B1`` and which
    ``B2``.
    """
    match = re.match(r"^(\d+)([a-zA-Z])\((.*)\)$", position)
    if match is None or match.group(2) not in improper:
        return position
    symbols = ",".join(part if part.startswith("-") else "-" + part
                       for part in match.group(3).split(","))
    return f"{match.group(1)}{match.group(2)}({symbols})"


@dataclass(frozen=True)
class EBRConvention:
    """Where the elementary band representations returned differ from the tables'.

    What ``ebr_data(group).convention`` holds.  The rows returned are labeled
    in the convention where a lattice translation ``t`` acts on a Bloch state
    at ``k`` as ``e^{-ik·t}``.  The tables of the Bilbao Crystallographic
    Server, which the irreptables package ships, use ``e^{+ik·t}``.  The
    label the server prints at ``k`` is the label returned at ``-k``, so the
    two agree at the time-reversal-invariant momenta.  The Wyckoff positions
    are the server's.

    Attributes
    ----------
    name : str
        The convention, ``"physical (translations as e^{-ik.t})"``.
    changed_points : tuple of str
        If nonempty, the points of ``changed_labels``: the high-symmetry
        points at which a label is renamed or its row differs from the
        server's.
    changed_labels : tuple of str
        If nonempty, the server's labels that are renamed here, or whose row
        of the matrix of elementary band representations differs from the
        server's row of that label.
    conjugated_points : tuple of str
        If nonempty, the points at which the characters of
        :func:`corep_table` are the complex conjugates of the server's.  Each
        character vector there carries the server's label of the vector it
        equals, where the server prints one.
    divergent_points : tuple of str
        If nonempty, the points at which the characters the server prints
        are, in the package's convention, those of ``-k``, another point of
        the star of ``k``.  The characters :func:`corep_table` returns there
        are their complex conjugates.  No reordering of the labels makes them
        the server's.
    reglued_points : tuple of str
        If nonempty, the points at which time reversal glues the irreps into
        coreps differently from the server's tables.  They occur in grey
        groups only.
    conjugation : dict
        ``{label: label}``, each of the server's labels with the label
        returned whose characters are the complex conjugates of its own.
    tr_partner : tuple of int
        For each column of the matrix, the column induced from the
        complex-conjugate site irrep at the same Wyckoff position, the column
        itself for a real site irrep.
    corep_names_bcs : tuple of str
        The server's label of each row, position by position with
        ``ebr_data(group).corep_names``.  The two hold the same label at
        every position outside ``reglued_points``.  Outside them, what can
        differ is the row a label holds, given by ``row_permutation``.
    row_permutation : tuple of int
        The permutation ``P`` of the rows.  Row ``i`` of a column returned is
        the server's row ``P[i]`` of the column ``tr_partner`` names.  So a
        linear form whose coefficients are shipped on the server's labels
        has, on row ``i``, its coefficient on ``corep_names_bcs[P[i]]``.
    note : str
        A few sentences on what the convention changes at each kind of point.
    """

    name: str
    changed_points: tuple
    changed_labels: tuple
    conjugated_points: tuple
    divergent_points: tuple
    reglued_points: tuple
    conjugation: dict
    tr_partner: tuple
    corep_names_bcs: tuple
    row_permutation: tuple
    note: str
    _applied: bool = field(default=True, repr=False)


@dataclass(frozen=True)
class EBRData:
    """The elementary band representations of a group, and their matrix.

    What :func:`ebr_data` returns.  The matrix has a column for each
    elementary band representation and a row for each little-group
    representation at each high-symmetry point of the group, those out of a
    layer's plane included.

    Attributes
    ----------
    ebrs : tuple of Term
        The columns, in the order of the tables.  Each names the site irrep
        it is induced from, its Wyckoff position and the position's
        site-symmetry group.  A site-symmetry group whose symbol begins with
        a bar, ``-43m``, is written with it, where the tables leave it out.
    corep_names : tuple of str
        The label of each row, as :func:`identify_from_characters` returns
        labels.
    matrix : ndarray
        The matrix of elementary band representations.  Entry ``(i, j)`` is
        the multiplicity of ``corep_names[i]`` in ``ebrs[j]``.
    smith : tuple of ndarray
        The Smith normal form ``(U, D, V)`` of ``matrix``, with
        ``U @ matrix @ V == D``.  For a group with one elementary band
        representation, ``D`` is a vector and ``V`` a number.
    convention : EBRConvention
        Where the rows differ from those of the Bilbao Crystallographic
        Server.
    group : GroupData
        The group the elementary band representations are read for.
    """

    ebrs: tuple
    corep_names: tuple
    matrix: Any = field(repr=False)
    smith: tuple = field(repr=False)
    convention: EBRConvention = field(repr=False)
    group: Any = field(repr=False)
    _raw: Any = field(default=None, repr=False, compare=False)

    def __repr__(self) -> str:  # pragma: no cover: convenience only
        kind = "irreps" if getattr(self.group, "magnetic_type", None) == "I" else "coreps"
        return (f"EBRData({self.group.name} #{self.group.number_str}: {len(self.ebrs)} "
                f"elementary band representations over {len(self.corep_names)} {kind})")


def _ebr_term(name, position) -> Term:
    """A column of the EBR matrix as a term of multiplicity one.

    ``name`` is the column's name, ``"-E1↑G(2)"``, and ``position`` its
    position with its site groups, ``"1a(4mm1',4mm)"``.  Returns a ``Term``
    with the site irrep, the Wyckoff position, ``"1a"``, the site-symmetry
    group, the first symbol in the parentheses, and its unitary part, the
    last, each group named as :func:`.._spacegroup._written_site_symmetry`
    writes it.
    """
    position = str(position)
    inside = position[position.index("(") + 1:position.rindex(")")].split(",")
    site = _written_site_symmetry(inside[0])
    unitary = _written_site_symmetry(inside[-1])
    return Term(multiplicity=1, irrep=_ebr_irrep(name), wyckoff=_wyckoff_letter(position),
                site_symmetry=site, unitary_site_symmetry=unitary)


def ebr_data(group: GroupData) -> EBRData | Refusal:
    """The elementary band representations of the group, and their matrix.

    Each elementary band representation is the band representation one site
    irrep induces from its Wyckoff position.  Its column holds the
    multiplicity of each little-group representation at every high-symmetry
    point of the group, those out of a layer's plane included.  The columns
    are read from the tables of the Bilbao Crystallographic Server, which the
    irreptables package ships, and carried into the package's convention for
    a lattice translation ``t``, ``e^{-ik·t}``, where the server writes
    ``e^{+ik·t}``.  The label the server prints at ``k`` is the label
    returned at ``-k``.  ``convention`` lists the points at which a label is
    renamed or its row differs from the server's.

    Parameters
    ----------
    group : GroupData
        The group.  Pass ``r.group`` of what :func:`..groups.realized_group`
        returns.

    Returns
    -------
    EBRData or Refusal
        The elementary band representations, their matrix and its Smith
        normal form, and where the rows differ from the server's.  A
        ``Refusal`` in its place is falsy.

    Refusals
    --------
    A refusal handed in as ``group`` is returned as it is.

    ``ebr_tables`` / ``irrep_tables`` / ``ebr_convention``
        the tables installed hold no elementary band representations of this
        group that can be read in the package's convention.

    Raises
    ------
    TypeError
        ``group`` is not ``r.group``: ``r`` itself, ``r.rep``, the ``group``
        of :func:`..srsi.srsi_tables` for a space-group number, which has no
        cell, or anything else.
    """
    _require_group("ebr_data", group)
    data = _ebr_data(group)
    if isinstance(data, Refusal):
        return data
    record = dict(data["convention"])
    convention = EBRConvention(
        name=str(record["name"]), changed_points=tuple(record["changed_points"]),
        changed_labels=tuple(record["changed_labels"]),
        conjugated_points=tuple(record["conjugated_points"]),
        divergent_points=tuple(record["divergent_points"]),
        reglued_points=tuple(record["reglued_points"]),
        conjugation=dict(record["conjugation"]), tr_partner=tuple(record["tr_partner"]),
        corep_names_bcs=tuple(record["labels_bcs"]),
        row_permutation=tuple(record["row_permutation"]), note=str(record["note"]),
        _applied=bool(record["applied"]))
    return EBRData(
        ebrs=tuple(_ebr_term(name, position) for name, position in data["ebrs"]),
        corep_names=tuple(data["labels_physics_convention"]), matrix=data["matrix"],
        smith=tuple(data["smith"]), convention=convention, group=group,
        _raw=data["raw"])


def _ebr_data(group) -> dict | Refusal:
    """The elementary band representations of a group, as a mapping.

    ``group`` is a :class:`GroupData`, or anything that carries its
    ``number_str``, ``spinor`` and ``_magnetic`` (the tables' own key).  Only
    those are read, with ``_lattice_index`` where it is present, and the
    ``name`` and ``_fractional_translations`` a lattice supergroup's refusal
    names.  Keys: ``matrix``; ``labels_physics_convention``,
    the labels of the rows; ``ebrs``, each column as ``(name, position)``,
    the position with its site groups, an improper one with its bar,
    ``"3a(-3m1',-3m)"``; ``smith``, ``(U, D, V)``; ``convention``, the record
    of :func:`_transport_record`; ``raw``, the tables' own record carried
    into this convention.  Returns ``lattice_not_descended`` for a lattice
    supergroup, and otherwise the refusals :func:`ebr_data` documents.
    """
    gate = _lattice_not_descended(group)
    if gate is not None:
        return gate
    status = tables_status()
    if not status["tables"]:
        return _refuse_no_tables(status)
    from irrep.ebrs import get_ebr_matrix, get_ebr_names_and_positions, get_smith_form

    try:
        raw = _spacegroup._ebr_file(group.number_str, group.spinor)
    except (FileNotFoundError, OSError) as exc:
        return Refusal("ebr_tables",
                       f"the tables of the irreptables package hold no "
                       f"{'double' if group.spinor else 'single'}-valued elementary band "
                       f"representations of #{group.number_str}: {exc}",
                       context={"number_str": group.number_str, "spinor": group.spinor})
    try:
        raw = _to_physical_convention(raw, group.number_str, group.spinor,
                                      group._magnetic)
        record = _transport_record(group.number_str, group.spinor, group._magnetic)
    except _EBRTransportError as exc:
        return Refusal(
            "ebr_convention",
            f"the elementary band representations of #{group.number_str} cannot be "
            "carried from the convention of the tables of the irreptables package, "
            f"e^(+ik·t), into the package's, e^(-ik·t): {exc} Report it, with the model "
            "and the call.",
            context={"number_str": group.number_str, "spinor": group.spinor,
                     "magnetic": group._magnetic, "points": exc.points})
    except (FileNotFoundError, OSError) as exc:
        return Refusal("irrep_tables",
                       f"the tables of the irreptables package hold no irrep table of "
                       f"#{group.number_str}: {exc}",
                       context={"number_str": group.number_str, "spinor": group.spinor})
    improper = _improper_letters(int(str(group.number_str).split(".")[0]))
    return {"raw": copy.deepcopy(raw), "matrix": get_ebr_matrix(raw),
            "labels_physics_convention": list(raw["basis"]["irrep_labels"]),
            "ebrs": [(str(name), _barred(str(position), improper))
                     for name, position in get_ebr_names_and_positions(raw)],
            "smith": get_smith_form(raw), "convention": record}


def _form_on_rows(convention, factors, n_rows):
    """A named indicator's coefficients, carried onto the rows of the EBR matrix.

    ``convention`` is the record of :func:`_transport_record`, ``factors`` the
    form's coefficients as the irreptables package ships them, keyed by the
    Bilbao Crystallographic Server's labels, and ``n_rows`` the number of rows.
    Row ``i`` takes the coefficient of ``labels_bcs[P[i]]``, ``P`` the row
    permutation, whatever the label the package gives the row.  Returns the
    coefficients, row by row, or ``None`` where the record does not carry a
    label and a permutation entry for every row, or the form names a label
    that is no row's.
    """
    labels_bcs = list(convention.get("labels_bcs", ()))
    perm = list(convention.get("row_permutation", ()))
    if len(labels_bcs) != n_rows or len(perm) != n_rows or set(factors) - set(labels_bcs):
        return None
    coef = [float(factors.get(lab, 0.0)) for lab in labels_bcs]
    return tuple(coef[perm[i]] for i in range(n_rows))


#: The note :func:`indicator_group` returns for each case: why the formulas are
#: withheld from a group that is not grey, or how a grey group's formulas are read.
_SI_NOTES = {
    "bare_key": (
        "no named form is returned: for a unitary group irreptables' non-magnetic "
        "indicator file is the grey table keyed by the bare space-group number "
        "(single_indicators.json and double_indicators.json are byte for byte the "
        "type-II entries of the magnetic files), so its factors are grey corep "
        "labels -- against a unitary group's own EBR basis only 25 of 70 single and "
        "48 of 320 double forms even have the right label set, and only 39 / 128 "
        "vanish on that group's own EBRs. Summing one over unitary labels is a "
        "silent partial sum, not an indicator. `elementary_divisors` is unaffected: "
        "it is the Smith form of this group's own transported EBR matrix. For "
        "values, use ebr.symmetry_indicator, which computes the class from the "
        "Smith form and returns a named form only after checking it vanishes on "
        "every EBR of the group it is being read on"),
    "unaudited_bns": (
        "no named form is returned: irreptables does ship indicator forms under this "
        "magnetic (BNS) key, but only the grey entries of those files have been "
        "audited here, and the corner that has not is measurably unclean -- five "
        "shipped z2 forms (216.74, 226.122, 226.127, 227.128, 227.130) are not "
        "indicators of their own group at all, being nonzero on real EBRs. "
        "`elementary_divisors` is unaffected: it is the Smith form of this group's "
        "own transported EBR matrix. For values, use ebr.symmetry_indicator"),
    "grey": (
        "irreptables' named forms for this grey group, as shipped: their "
        "coefficients are keyed by BCS labels, and are carried onto the rows by the "
        "convention's labels and row permutation before they are evaluated on a "
        "returned symmetry data vector. "
        "ebr.symmetry_indicator reports the class of a band set, in the named "
        "indicators of Hwang et al. where the group has them"),
    "grey_none": (
        "irreptables ships no named indicator formula for this grey group, a "
        "verdict in itself -- read `elementary_divisors` for the indicator group "
        "itself, and ebr.symmetry_indicator for a class"),
}


def indicator_group(group: GroupData) -> dict | Refusal:
    """A space group's symmetry-indicator group, and the tables' indicator formulas.

    The indicator group is read off the Smith normal form of the group's
    matrix of elementary band representations, on every high-symmetry point
    of the group, those out of a layer's plane included.  The formulas are
    the named indicator formulas the irreptables package ships, returned for
    a grey group only.

    Parameters
    ----------
    group : GroupData
        The group.  Pass ``r.group`` of what :func:`..groups.realized_group`
        returns.

    Returns
    -------
    dict or Refusal
        A mapping with the keys:

        * ``elementary_divisors``: the elementary divisors ``d`` above 1 of
          the Smith normal form; the indicator group has a factor ``Zd`` for
          each.  ``None`` where :func:`ebr_data` refuses for the group.
        * ``indicators``: the formulas, ``{name: {"factors": {label:
          coefficient}, "mod": modulus}}``, the coefficients keyed by the
          labels of the Bilbao Crystallographic Server.  To read a formula on
          a returned symmetry data vector, its coefficients are carried onto
          the rows by the server's label of each row,
          ``ebr_data(group).convention.corep_names_bcs``, and
          ``ebr_data(group).convention.row_permutation``.  Empty
          where the tables name no formula for the group.
        * ``orders``: ``{name: modulus}``.  Two formulas can be dependent, so
          the moduli need not be the elementary divisors.
        * ``withheld``: whether the formulas are left out, as they are for
          every group that is not grey.
        * ``reason``: why they are left out, ``None`` where they are not.
        * ``note``: where the formulas are withheld, ``reason``; otherwise
          how the formulas are evaluated on a symmetry data vector, or that
          the tables name none.

        A ``Refusal`` in its place is falsy.

    Refusals
    --------
    A refusal handed in as ``group`` is returned as it is.

    ``irrep_tables``
        the character tables of the irreptables package are not loaded.

    See Also
    --------
    symmetry_indicator : the symmetry-indicator class of a band set and, for
        a grey group, its values in the named indicators of Hwang et al.
    """
    gate = _lattice_not_descended(group)
    if gate is not None:
        return gate
    status = tables_status()
    if not status["tables"]:
        return _refuse_no_tables(status)
    data = _ebr_data(group)
    divisors = None
    if not isinstance(data, Refusal):
        r = np.asarray(data["smith"][1])
        # single EBR column: R is stored flat, and the vector is then already the
        # diagonal extended by the empty rows (the shape .._smith._smith_system normalizes)
        d = r if r.ndim == 1 else r.diagonal()
        divisors = d[d > 1]

    if _magnetic_type(group) != "II":
        note = _SI_NOTES["unaudited_bns" if group._magnetic else "bare_key"]
        return {"indicators": {}, "orders": {}, "elementary_divisors": divisors,
                "withheld": True, "reason": note, "note": note}

    si = _spacegroup._indicator_table(group.spinor, group._magnetic)
    entry = copy.deepcopy(si.get(group.number_str, {}).get("indicators", {}))
    return {"indicators": entry,
            "orders": {k: v["mod"] for k, v in entry.items()},
            "elementary_divisors": divisors,
            "withheld": False, "reason": None,
            "note": _SI_NOTES["grey" if entry else "grey_none"]}


def _ebr_solve_shim(ebr_matrix, y, original_exc):
    """The integer combinations of EBRs that equal a symmetry data vector.

    ``ebr_matrix`` is the EBR matrix, a row for each label and a column for
    each EBR, ``y`` the vector on its rows, and ``original_exc`` the error
    the irrep package raised, which is not read.  The model and the callback
    are those of ``irrep.ebrs.compute_ebr_decomposition``, the solver's
    status compared with the value ``Solve`` returns in place of
    ``solver.status_name()``, which ortools 9.15 broke.  Returns
    ``(solutions, True)`` with at most five combinations with coefficients
    from 0 to 50; where there are none, ``(solutions, False)`` with
    coefficients from -50 to 50; and ``(None, False)`` where there are none
    either.
    """
    from ortools.sat.python import cp_model
    from irrep.or_solutions_obtainer import varArraySolutionObtainer

    n_ir, n_ebr = ebr_matrix.shape
    ok = (cp_model.OPTIMAL, cp_model.FEASIBLE)

    def solve(bounds):
        model = cp_model.CpModel()
        solver = cp_model.CpSolver()
        x = [model.NewIntVar(bounds[0], bounds[1], f"x{i}") for i in range(n_ebr)]
        obtainer = varArraySolutionObtainer(x)
        for i in range(n_ir):
            model.Add(sum(int(ebr_matrix[i, j]) * x[j] for j in range(n_ebr)) == int(y[i]))
        solver.parameters.enumerate_all_solutions = True
        status = solver.Solve(model, obtainer)
        return obtainer.n_smallest_solutions(5), status

    solutions, status = solve((0, 50))
    if status in ok and solutions:
        return solutions, True
    solutions, status = solve((-50, 50))
    if status in ok and solutions:
        return solutions, False
    return None, False


def decompose_ebrs(group: GroupData, irrep_counts: dict) -> dict | Refusal:
    """A symmetry data vector written in the group's elementary band representations.

    The vector ``y`` holds the multiplicity of the label of each row of the
    group's matrix of elementary band representations.  Its coordinates
    ``U @ y``, with ``U`` of the matrix's Smith normal form, show whether an
    integer combination of elementary band representations equals it.  Where
    one does, the integer solver of the ortools package searches for
    combinations with coefficients from 0 to 50, and, where it finds none,
    from -50 to 50.

    Parameters
    ----------
    group : GroupData
        The group.  Pass ``r.group`` of what :func:`..groups.realized_group`
        returns.
    irrep_counts : dict
        The symmetry data vector, ``{label: multiplicity}``, at every
        high-symmetry point of the group, those out of a layer's plane
        included.  Pass ``HSPTable.symmetry_data``, or a vector written out by
        hand.

    Returns
    -------
    dict or Refusal
        A mapping with the keys:

        * ``classification``: ``"stable topological"`` where no integer
          combination of elementary band representations equals the vector,
          ``"atomic limit"`` where one with coefficients from 0 to 50 does,
          and ``"fragile topological"`` otherwise.
        * ``stable_nontrivial``: whether the vector is stable topological.
        * ``solutions``: at most five combinations, each a list of
          coefficients on ``ebrs``.  ``None`` for a stable topological
          vector, and where no combination with coefficients from -50 to 50
          is found.
        * ``decomposition_strings``: each combination written out, present
          where combinations were found.
        * ``is_positive``: whether a combination with coefficients from 0 to
          50 was found.  ``None`` for a stable topological vector.
        * ``y``: the vector.
        * ``y_prime``: ``U @ y``.
        * ``ebrs``: the columns, each as its name and Wyckoff position,
          ``("-E1↑G(4)", "8a(-43m1',-43m)")``.  They are written as the
          tables write them, but for the bar of a site-symmetry group such as
          ``-43m``, which the tables leave out.
        * ``unused``: the labels handed in that no row of the matrix carries.
        * ``indicators``: for a grey group, the value of each named formula
          of :func:`indicator_group` on the vector, modulo its modulus.  Each
          shipped coefficient is carried onto the row it stands for, by the
          server's label of each row and
          ``ebr_data(group).convention.row_permutation``, and multiplies the
          multiplicity of that row's label.  ``None`` where the formulas are
          withheld, or cannot be carried onto the rows, as
          ``indicators_note`` then says.
        * ``indicators_note``: the ``note`` of :func:`indicator_group`, or why
          the formulas cannot be carried onto the rows.

        The symmetry data vector of an atomic limit is a combination with no
        negative coefficient.  A band set classified ``"atomic limit"`` need
        not be one.  A ``Refusal`` in its place is falsy.

    Refusals
    --------
    A refusal handed in as ``group`` is returned as it is.

    ``incomplete_content``
        the vector holds no label at a high-symmetry point of the group.
        ``context["missing_points"]`` names each.  A layer's
        ``HSPTable.symmetry_data`` holds no point out of its plane.
        :func:`..ebr.decompose` reads such a vector.
    ``ortools``
        the ortools package is not installed, and an integer combination of
        elementary band representations equals the vector.
    ``ebr_tables`` / ``irrep_tables`` / ``ebr_convention``
        the tables installed hold no elementary band representations of this
        group that can be read in the package's convention.

    See Also
    --------
    decompose : the sums of elementary band representations that carry the
        symmetry data vector of a model's bands, read for a layer on the
        points of its plane.
    """
    gate = _lattice_not_descended(group)
    if gate is not None:
        return gate
    data = _ebr_data(group)
    if isinstance(data, Refusal):
        return data
    from irrep.ebrs import (ORTOOLS_AVAILABLE, compose_ebr_string,
                            compute_ebr_decomposition, compute_topological_classification_vector)

    basis = data["labels_physics_convention"]
    unused = sorted(set(irrep_counts) - set(basis), key=corep_order)
    # A vector at every point, or a refusal. `create_symmetry_vector` reads an
    # unmentioned point as zero, and a zero row is a wrong constraint, so a
    # vector with a point missing decomposes into the EBRs of a different band
    # set. A layer meets this: the EBR basis of the 3D space group carries
    # k_z = 1/2 points its zone does not have.
    points = sorted({_label_kpoint(lab) for lab in basis}, key=point_order)
    covered = {_label_kpoint(lab) for lab in irrep_counts}
    missing_points = [p for p in points if p not in covered]
    if missing_points:
        return Refusal(
            "incomplete_content",
            f"the symmetry data vector holds no label at {missing_points}, and the "
            f"elementary band representations of #{group.number_str} have rows at "
            f"{points}. Pass multiplicities at every point. For a layer, whose zone does "
            "not hold the points out of its plane, use decompose.",
            value=float(len(missing_points)), threshold=0.0,
            context={"missing_points": tuple(missing_points), "points": tuple(points)})
    y, y_prime, nontrivial = compute_topological_classification_vector(irrep_counts, data["raw"])

    si = indicator_group(group)
    indicators, indicators_note = None, None
    if not isinstance(si, Refusal):
        # whether the named forms are returned is `withheld`, not the emptiness of
        # the mapping: a grey group can ship no form at all, which is a verdict,
        # and it is returned as the empty sum, `{}`, not as `None`.
        indicators_note = si["reason"] if si["withheld"] else si["note"]
        if not si["withheld"]:
            # each shipped coefficient on the row it stands for, then summed over
            # the rows: the server's label of a row need not be the package's
            rows = [irrep_counts.get(lab, 0) for lab in basis]
            indicators = {}
            for name, spec in si["indicators"].items():
                coef = _form_on_rows(data["convention"], spec["factors"], len(basis))
                if coef is None:
                    indicators, indicators_note = None, (
                        f"no named form is evaluated: the form {name!r} of "
                        f"#{group.number_str} names a label that is no row of the elementary "
                        "band representations, or the record of the change of convention "
                        "does not give each row's label on the Bilbao Crystallographic "
                        "Server, which carries the form's coefficients onto the rows")
                    break
                indicators[name] = sum(c * n for c, n in zip(coef, rows)) % spec["mod"]

    out = {"y": y, "y_prime": y_prime, "stable_nontrivial": bool(nontrivial),
           "indicators": indicators, "indicators_note": indicators_note,
           "ebrs": data["ebrs"], "unused": unused,
           "solutions": None, "is_positive": None}
    if nontrivial:
        out["classification"] = "stable topological"
        return out
    if not ORTOOLS_AVAILABLE:
        return Refusal("ortools",
                       "an integer combination of elementary band representations equals "
                       "the symmetry data vector, and the ortools package is not "
                       "installed. Install it: pip install 'ortools>=9.10'.",
                       context={"y": y, "y_prime": y_prime})
    try:
        solutions, is_positive = compute_ebr_decomposition(data["raw"], y)
    except (TypeError, AttributeError) as exc:
        # Where irrep's call fails on ortools 9.15, whose CpSolver.status_name()
        # raises, run irrep's model with the solver's status compared directly.
        solutions, is_positive = _ebr_solve_shim(data["matrix"], y, exc)
    out["solutions"] = solutions
    out["is_positive"] = bool(is_positive)
    out["classification"] = "atomic limit" if is_positive else "fragile topological"
    if solutions:
        out["decomposition_strings"] = [compose_ebr_string(s, data["ebrs"]) for s in solutions]
    return out


# --------------------------------------------------------------------------
# meshes
# --------------------------------------------------------------------------

def k_grid(group: GroupData, n, *, allow_asymmetric=False) -> KGrid | Refusal:
    """A mesh through ``GM`` that the group carries onto itself.

    The mesh is a ``pythtb.Mesh``, with ``k`` in ``[0, 1)`` along each
    periodic direction.  It can be handed to a ``pythtb.WFArray`` as it is.
    It holds a tabulated high-symmetry point only where the number of
    divisions fits the point's coordinates: ``X`` at ``1/2`` needs an even
    number.

    Parameters
    ----------
    group : GroupData
        The group.  Pass ``r.group`` of what :func:`..groups.realized_group`
        returns.
    n : int or sequence of int
        The number of divisions along each periodic direction, or one number
        for all of them.
    allow_asymmetric : bool, default False
        Whether a mesh the group does not carry onto itself is returned or
        refused.  With ``allow_asymmetric=True``, it is returned, with no
        irreducible points.

    Returns
    -------
    KGrid or Refusal
        The mesh, its irreducible points, and which tabulated high-symmetry
        points it holds.  ``summary()`` prints it.  With
        ``allow_asymmetric=True``, a mesh the group does not carry onto
        itself is returned with no irreducible points.  A ``Refusal`` in its
        place is falsy.

    Refusals
    --------
    A refusal handed in as ``group`` is returned as it is.

    ``asymmetric_grid``
        the group does not carry the mesh onto itself.  With
        ``allow_asymmetric=True``, the call does not refuse.

    Raises
    ------
    TypeError
        ``group`` is not ``r.group``: ``r`` itself, ``r.rep``, the ``group``
        of :func:`..srsi.srsi_tables` for a space-group number, which has no
        cell, or anything else.
    ValueError
        ``n`` is a sequence of other than ``dim_k`` numbers.
    """
    _require_group("k_grid", group)
    gate = _lattice_not_descended(group)
    if gate is not None:
        return gate
    from pythtb import Mesh
    if np.isscalar(n):
        divs = [int(n)] * group._dim_k
    else:
        divs = [int(v) for v in n]
        if len(divs) != group._dim_k:
            raise ValueError(f"k_grid(): the number of entries of n=, {len(divs)}, is not "
                             f"the number of periodic directions, {group._dim_k}. Pass one "
                             "number, or one for each periodic direction.")
    grid3 = [1, 1, 1]
    for j, d in enumerate(group.periodic_dirs):
        grid3[d] = divs[j]

    symmetric = bool(group._sg.is_grid_symmetrical(grid3))
    if not symmetric and not allow_asymmetric:
        return Refusal("asymmetric_grid",
                       f"{group.name} #{group.number_str} does not carry the "
                       f"{tuple(grid3)} mesh onto itself. Pass a mesh the group carries "
                       "onto itself, or allow_asymmetric=True.",
                       context={"grid": tuple(grid3), "number_str": group.number_str})

    mesh = Mesh(["k"] * group._dim_k)
    mesh.build_grid(tuple(divs), gamma_centered=False, k_endpoints=False)
    points = np.asarray(mesh.flat)

    irr = None
    if symmetric:
        irr3 = group._sg.get_irreducible_kpoints_grid(grid3)
        irr = group._k_from_3d(irr3)

    hsp = _high_symmetry_points(group)
    contains = {}
    if not isinstance(hsp, Refusal):
        for name, kk in hsp.items():
            contains[name] = bool(np.any(
                np.abs(((points - kk) + 0.5) % 1.0 - 0.5).max(axis=1) <= _KGRID_KTOL))

    return KGrid(divisions=tuple(grid3), symmetric=symmetric, mesh=mesh, irreducible=irr,
                 contains=contains)
