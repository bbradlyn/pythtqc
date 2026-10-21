"""``irrep``'s tables in the model's own cell: the high-symmetry points, the
little group of a k-point, the character table of its coreps, the coreps that
measured characters carry, the elementary band representations, the indicator
group, and a mesh that respects the group.
"""

from __future__ import annotations

from functools import lru_cache
import itertools
import re

import numpy as np

from . import _sitechars
from . import _spacegroup
from .refusal import InternalError, Refusal
from ._tolerance import (
    WYCKOFF_REFERENCE_TOL, _CHAR_TOL, _LATTICE_TOL, _PHASE_REAL_TOL, _SU2_TOL)
from ._text import _point_text
from ._spacegroup import (
    GroupData, IrrepContent, KGrid, LittleGroup, OrbitalAssignment, SiteReference,
    _EBRTransportError, _GENERIC_PARAMETER, _centering, _is_reciprocal,
    _is_translation, _lattice_not_descended, _letters_on_closure, _magnetic_type,
    _pad_cell, _pyxtal_wyckoff, _reference_operations, _refuse_no_tables,
    _sites_and_typat, _snap_rational, _variety_residual, _wyckoff_setting_audit,
    tables_status, wyckoff_positions_spglib)

__all__ = ["kpoint_of_label", "ebr_rows", "wyckoff_sites_tables", "little_group",
           "HighSymmetryPoints", "high_symmetry_points", "corep_table",
           "identify_from_characters", "ebr_data", "indicator_group",
           "decompose_ebrs", "k_grid"]


def kpoint_of_label(label: str) -> str:
    """The tables' name of the point a corep label belongs to: ``-GM6`` to ``GM``, ``-M3M4`` to ``M``.

    A leading ``-``, which marks a double-valued corep, is dropped, and the
    letters before the first digit are the point.

    Parameters
    ----------
    label : str
        A corep label, ``"-GM6"``, ``"GM2-"``, ``"X1"``.

    Returns
    -------
    str
        The point's name.
    """
    s = str(label)
    if s.startswith("-"):
        s = s[1:]
    i = 0
    while i < len(s) and not s[i].isdigit():
        i += 1
    return s[:i]


def ebr_rows(group) -> dict | Refusal:
    r"""Each elementary band representation of the group as a symmetry vector: the coreps it carries at the high-symmetry points.

    The coreps are in the convention every returned label is, a lattice
    translation acting as ``e^{-ik·t}``, where the Bilbao Crystallographic
    Server's tables have ``e^{+ik·t}``: away from the time-reversal-invariant
    points a row can carry other labels than the server's page, ``K3`` for its
    ``K2`` in the row of ``A1`` at ``1b`` of space group 143, and the Wyckoff
    positions never move.

    Parameters
    ----------
    group : .groups.GroupData
        The group, ``r.group``.

    Returns
    -------
    dict or Refusal
        ``{key: {corep: multiplicity}}``, the zeros dropped, each key an
        elementary band representation at its Wyckoff position,
        ``"-E1↑G(2)@1a(6mm1',6mm)"``.  A ``Refusal`` in its place is falsy and
        names what stopped the call.

    Refusals
    --------
    ``ebr_tables`` / ``irrep_tables`` / ``ebr_convention``
        no elementary band representations of this group in the tables
        installed, or none that can be read in this package's convention.
    """
    gate = _spacegroup._lattice_not_descended(group)
    if gate is not None:
        return gate
    data = ebr_data(group)
    if isinstance(data, Refusal):
        return data
    out = {}
    labels = data["labels_physics_convention"]
    for (name, wy), col in zip(data["ebrs"], np.asarray(data["matrix"]).T):
        out[f"{name}@{wy}"] = {lab: int(v) for lab, v in zip(labels, col) if v}
    return out


def _wyckoff_letter(position: str) -> str:
    """``"1a(4mm1',4mm)" → "1a"`` — the multiplicity+letter, without the site groups."""
    return str(position).split("(")[0]


def _letter_only(position: str) -> str:
    """``"1a(4mm1',4mm)" → "a"`` — the Wyckoff letter alone, without its multiplicity."""
    return _wyckoff_letter(position).lstrip("0123456789")


def _bands_spanned(name: str) -> int:
    """``"-E1↑G(4)" → 4`` — how many bands an elementary band representation spans."""
    name = str(name)
    return int(name[name.rindex("(") + 1:name.rindex(")")])


def wyckoff_sites_tables(
        group: GroupData, positions=None, *,
        position_tol: float = WYCKOFF_REFERENCE_TOL) -> tuple | Refusal:
    """The Wyckoff position of each site, lettered in the tables' setting, the one every table is read with.

    A letter is a table key, and settings disagree about it: the Fu–Kane–Mele
    diamond's orbit is ``8a`` here and ``8b`` in spglib's setting, and a site
    irrep at two positions can carry the same content at every k-point and
    different real-space invariants.  The orbitals' own letters are in
    ``realized_group(model).assignment.wyckoff_tables``.

    Parameters
    ----------
    group : GroupData
        The group, ``realized_group(model).group``.  A grey group's positions
        are those of its unitary subgroup.
    positions : array, shape (n, 3), optional
        Points in reduced coordinates of the group's three-dimensional cell.
        Default: the points the group was found on,
        :attr:`GroupData.site_positions`.
    position_tol : float, optional
        How far a point may sit from a position and lie on it, in reduced
        coordinates of the tables' conventional cell.  Default
        :data:`WYCKOFF_REFERENCE_TOL`, ``1e-4``.

    Returns
    -------
    tuple of SiteReference or Refusal
        One per point: its letter and multiplicity, its free parameters, and
        the positions an orbital confined there can be moved to.  A
        ``Refusal`` in its place is falsy.

    Refusals
    --------
    A refusal handed in as ``group`` is returned as it is.

    ``tables_not_ready``
        The group has no transformation to the tables' conventional cell.
    ``wyckoff_setting``
        ``pyxtal``'s list of the group's positions is not written in the
        tables' setting, so no letter would be a table key.
    ``wyckoff_unresolved``
        A point lies on no position within ``position_tol``; the refusal names
        the nearest and its distance.
    ``ebr_tables``, ``irrep_tables``, ``ebr_convention``
        From :func:`ebr_data`, whose positions the letters are checked
        against.

    Examples
    --------
    >>> from pythtb.models import fu_kane_mele                         # doctest: +SKIP
    >>> r = realized_group(fu_kane_mele(1.0, 0.125, [0, 0, 0, 0]))     # doctest: +SKIP
    >>> [(s.multiplicity, s.letter) for s in wyckoff_sites_tables(r.group)]  # doctest: +SKIP
    [(8, 'a'), (8, 'a')]
    """
    gate = _lattice_not_descended(group)
    if gate is not None:
        return gate
    if not group._tables_ready:
        return Refusal("tables_not_ready",
                       group._setup_note or "irrep tables are not set up for this cell, "
                       "so there is no refUC/shiftUC to write positions in the "
                       "reference cell with",
                       context={"number_str": group.number_str})
    data = ebr_data(group)
    if isinstance(data, Refusal):
        return data
    ebr_positions = []
    for _name, position in data["ebrs"]:
        match = re.match(r"^(\d+)([a-zA-Z])\((.*)\)$", str(position))
        if match is None:                              # pragma: no cover - guard
            raise InternalError(
                f"#{group.number_str}: EBR position {position!r} is not "
                "'<multiplicity><letter>(<site symbols>)', so its Wyckoff letter cannot "
                "be read")
        ebr_positions.append((int(match.group(1)), match.group(2), match.group(3)))
    listed = frozenset(letter for _m, letter, _s in ebr_positions)

    audit = group._wyckoff_audit.get("audit")
    if audit is None:
        audit = _wyckoff_setting_audit(int(group.sg_number),
                                       _reference_operations(group),
                                       ebr_positions)
        group._wyckoff_audit["audit"] = audit
    if not audit["ok"]:
        bad = audit["offenders"][0]
        return Refusal(
            "wyckoff_setting",
            f"pyxtal's Wyckoff list for space group {group.sg_number} is not written in "
            f"the setting #{group.number_str}'s own tables are: position "
            f"{bad['expected'] if bad['reason'] != 'multiplicity' else ''}{bad['letter']} "
            f"fails the {bad['reason']} check (measured {bad['measured']}, expected "
            f"{bad['expected']}). Every Wyckoff letter this function would return is a "
            "key into the wrong setting, so none is returned",
            value=None if bad["measured"] is None else float(bad["measured"]),
            threshold=None if bad["expected"] is None else float(bad["expected"]),
            context={"number_str": str(group.number_str), "sg": int(group.sg_number),
                     "offenders": audit["offenders"], "audit": audit})

    pyxtal_positions, _centering = _pyxtal_wyckoff(int(group.sg_number))
    raw = (np.asarray(group.site_positions, dtype=float) if positions is None
           else np.atleast_2d(np.asarray(positions, dtype=float)))
    inv = np.linalg.inv(np.asarray(group._refUC, dtype=float))
    shift = np.asarray(group._shiftUC, dtype=float)
    reference = np.array([(inv @ (x - shift)) % 1.0 for x in raw])
    reference = np.where(np.abs(reference - 1.0) < 1e-9, 0.0, reference)

    residuals = {p["letter"]: _variety_residual(p["arms"], reference)
                 for p in pyxtal_positions}
    out = []
    for i, (x, x_ref) in enumerate(zip(raw, reference)):
        hits = [p for p in pyxtal_positions
                if residuals[p["letter"]][i] <= position_tol]
        if not hits:                     # pragma: no cover - the general position's
            # variety is the whole cell, so this needs a pyxtal list without one
            nearest = min(pyxtal_positions, key=lambda p: residuals[p["letter"]][i])
            return Refusal(
                "wyckoff_unresolved",
                f"the position {np.round(x, 6).tolist()} is "
                f"{np.round(x_ref, 6).tolist()} in #{group.number_str}'s reference cell "
                f"and lies on no Wyckoff variety of space group {group.sg_number} there: "
                f"the nearest is {nearest['multiplicity']}{nearest['letter']}, "
                f"{float(residuals[nearest['letter']][i]):.3g} away in reduced "
                f"coordinates. A centre that far off every orbit is a modelling defect",
                value=float(residuals[nearest["letter"]][i]),
                threshold=float(position_tol),
                context={"site": i, "position": x, "reference": x_ref,
                         "nearest": nearest["letter"],
                         "number_str": str(group.number_str)})
        hits.sort(key=lambda p: (p["n_free"], p["multiplicity"], p["letter"]))
        own = hits[0]
        out.append(SiteReference(
            index=i, position=x, reference=x_ref, letter=own["letter"],
            multiplicity=own["multiplicity"], n_free=own["n_free"],
            varieties=tuple(p["letter"] for p in hits),
            closure=_letters_on_closure(int(group.sg_number), own["letter"], listed,
                                        float(position_tol)),
            residual=float(residuals[own["letter"]][i])))
    return tuple(out)


def _assign_orbitals(model, group: GroupData, *, typat=None,
                     symprec=None) -> OrbitalAssignment | Refusal:
    """Which site orbit each orbital sits on, and how the operations move them.

    Returned as :attr:`RealizedGroup.assignment`.  Each operation's image
    ``R·τ + t`` is looked up among the
    sites of the same species, and the permutation that comes out — with the
    integer lattice shift beside it, which is what fixes a permuting
    operation's embedding phase — is the data a representation is built from.
    Orbits, site-symmetry groups and each orbit's Wyckoff position are read off
    it.

    Parameters
    ----------
    model : pythtb.TBModel
        The orbital positions are read from **this** model, not from the group,
        so a model whose orbitals have drifted off the ideal positions can be
        checked against a group determined elsewhere (or at a looser
        ``symprec``).  Its lattice must match the group's 3D cell: the
        periodic vectors the same, and a layer's vector along its normal read at
        the length the group's cell gives it.
    group : GroupData
        The group whose operations do the permuting; its ``symprec`` is the
        default matching tolerance.
        A group comes from :func:`.._realized.realized_group`:
        ``realized_group(model, ...).group``.
    typat : sequence, length norb, optional
        Orbital-class labels, with the same meaning and default as
        :func:`.._realized.realized_group`'s: the value given there.
    symprec : float, optional
        Matching tolerance — Cartesian, in the model's length units.  Defaults
        to the group's own ``symprec``.

    Returns
    -------
    OrbitalAssignment or Refusal

    Refusals
    --------
    ``wyckoff_orbit``
        An orbital's image under some operation lands on no orbital — the
        refusal names the orbital, the operation and the **distance** to the
        nearest candidate.  A misplaced orbital is a modelling bug and is never
        silently symmetrized away.

    Raises
    ------
    ValueError
        When ``typat`` does not carry one entry per orbital, or the model's
        cell, its vacuum included, is not the group's: there is no assignment to make
        between a model and a group built on another cell.

    Notes
    -----
    The orbit structure reported here is *measured* from the group's
    operations.  The letters are read beside it, in both settings: from
    :func:`wyckoff_positions_spglib` per orbit and from
    :func:`wyckoff_sites_tables` per site (left empty where the enumeration or
    the reference cell was unavailable), with spglib's own per-site letter for
    the sites alone, in the group of the positions, alongside.
    """
    tol = float(group.symprec if symprec is None else symprec)
    lat, positions, _ = _pad_cell(model, group.vacuum_length, lattice=group.lattice)
    if not np.allclose(lat, group.lattice, atol=1e-9):
        apart = float(np.abs(np.asarray(lat, dtype=float)
                             - np.asarray(group.lattice, dtype=float)).max())
        raise ValueError(f"the model's cell, its vacuum included, is not the group's, "
                         f"{apart:.3g} apart: the group was built for another cell.")
    pos, site_orbs, species = _sites_and_typat(positions, typat, tol, lat)
    nsite = pos.shape[0]
    nsym = len(group.symmetries)

    perm = np.full((nsym, nsite), -1, dtype=int)
    shift = np.zeros((nsym, nsite, 3), dtype=int)
    worst = 0.0
    for g, sym in enumerate(group.symmetries):
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
                    f"orbital(s) {list(orbs)} at reduced position {np.round(pos[i], 6)} are not "
                    f"carried onto any orbital by symmetry operation #{sym.ind} "
                    f"(rotation {sym.rotation.tolist()}, translation "
                    f"{np.round(sym.translation, 6).tolist()}): the nearest orbital of the same "
                    f"class is {float(np.min(dist_masked)):.6g} away. A misplaced orbital is a "
                    f"modelling bug, not something to symmetrize away.",
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

    site_sym = tuple(tuple(int(group.symmetries[g].ind) for g in range(nsym) if perm[g, i] == i)
                     for i in range(nsite))

    # Wyckoff letters: irrep returns ONE representative per position, so a site may
    # sit on a symmetry-related copy of it -- test the whole orbit and keep the most
    # special (fewest free parameters) match.
    # A lattice supergroup's letters are the finer lattice's, so none is read:
    # the geometry above is the whole of what this record can honestly carry.
    supergroup = int(getattr(group, "_lattice_index", 1) or 1) > 1
    letters: list[str] = []
    wy = (Refusal("lattice_not_descended", "") if supergroup
          else wyckoff_positions_spglib(group))
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

    # the letter in the TABLES' setting -- the one a table lookup needs.
    # A refusal here is information -- the letter is left empty -- and not a
    # defect: the orbit structure above is measured from the operations and does
    # not depend on it.
    ref_letters: tuple = ()
    if not supergroup:
        sites_ref = wyckoff_sites_tables(group, pos)
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
    except Exception:  # pragma: no cover - spglib API drift must not break the record
        pass

    # the site-symmetry group's NAME, spelled as the atomic-limit table spells
    # it: the point group of the site's own unitary operations, read off those
    # operations.  The sites of one orbit have conjugate site groups and so one
    # name, which is read once per orbit.

    unitary = {int(s.ind) for s in group.symmetries if not s.time_reversal}
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

    The group a band's corep at ``k`` is a corep of: only these operations act
    within the states there.

    Parameters
    ----------
    group : GroupData
        The group, ``realized_group(model).group``.
    k : array, shape (dim_k,)
        The point, in the model's reduced coordinates.  A layer takes a point
        of its three-dimensional cell too, where it lies in the plane.

    Returns
    -------
    LittleGroup or Refusal
        Its unitary operations in index order, the columns a character vector
        at ``k`` is written in, and those combined with time reversal, which
        carry no characters; ``summary()`` prints it.  A ``Refusal`` in its
        place is falsy.

    Raises
    ------
    ValueError
        ``k`` has neither ``dim_k`` components nor three.

    Refusals
    --------
    A refusal handed in as ``group`` is returned as it is.

    ``out_of_plane``
        A point with a component along a layer's vacuum, a direction its zone
        does not have.
    """
    gate = _lattice_not_descended(group)
    if gate is not None:
        return gate
    k = np.asarray(k, dtype=float).ravel()
    if k.size == 3 and group.dim_k != 3:
        if not group.in_plane(k[None, :]):
            return Refusal("out_of_plane",
                           f"k = {np.round(k, 6)} has a component out of the model's periodic "
                           f"plane (periodic directions {group.periodic_dirs}); a slab model "
                           f"has no dispersion there.",
                           value=float(np.max(np.abs(k[list(group.vacuum_dirs)]))),
                           threshold=0.0, context={"k": k, "periodic_dirs": group.periodic_dirs})
        k3 = k
        k = group.k_from_3d(k3[None, :])[0]
    elif k.size == group.dim_k:
        k3 = group.k_to_3d(k[None, :])[0]
    else:
        raise ValueError(f"k must have {group.dim_k} (or 3) components, got {k.size}")

    uni, anti = [], []
    for sym in group.symmetries:
        G = sym.transform_k(k3) - k3
        if not np.allclose(G, np.round(G), atol=1e-8):
            continue
        (anti if sym.time_reversal else uni).append(sym)
    uni.sort(key=lambda s: s.ind)
    anti.sort(key=lambda s: s.ind)
    Gs = [np.round(s.transform_k(k3) - k3) for s in uni]
    return LittleGroup(k=k, k3=k3, unitary=uni, antiunitary=anti,
                       indices=tuple(int(s.ind) for s in uni),
                       G=np.array(Gs) if Gs else np.zeros((0, 3)))


def _table_kpoints(group: GroupData) -> dict:
    """``{kpname: k in the model's reduced 3D coordinates}`` from ``irreptables``.

    The irrep table's points, and, for a group without time reversal, the
    partners ``−k`` its elementary band representations list on their own
    (:func:`_partner_points`).
    """
    from irreptables.irreps import IrrepTable
    table = IrrepTable(group.number_str, group.spinor, magnetic=group.magnetic)
    inv = np.linalg.inv(group._refUC.T)
    partner = ({} if group.magnetic else
               {base: arm for arm, base, _naming
                in _partner_points(group.number_str, group.spinor)})
    listed: dict[str, np.ndarray] = {}
    for irr in table.irreps:
        if irr.kpname in listed:
            continue
        listed[irr.kpname] = np.asarray(irr.k, dtype=float)
        if irr.kpname in partner:                 # its partner -k, next to it
            listed[partner[irr.kpname]] = -listed[irr.kpname]
    return {name: _snap_rational((inv @ _snap_rational(k)) % 1.0)
            for name, k in listed.items()}


@lru_cache(maxsize=None)
def _partner_points(number_str: str, spinor: bool) -> tuple:
    r"""The points ``−k`` a unitary group's EBR rows list and its irrep table does not.

    Without time reversal, ``k`` and ``−k`` are different points of the zone
    wherever no operation of the group carries one onto the other, and the
    tables' elementary band representations carry rows at both: ``KA``, ``HA``,
    ``PA`` and ``WA`` beside ``K``, ``H``, ``P`` and ``W``.  The irrep table
    lists only the first.  The little group at ``−k`` is the one at ``k`` and
    its characters are the conjugates, the Bloch phases conjugating with them,
    so the partner's table is the listed point's, conjugated
    (:func:`_transported_table`); what is read here is which conjugate each
    partner label names.

    An elementary band representation induced from a site irrep ``ρ`` carries
    at ``−k`` the conjugate of what the one induced from ``ρ*`` carries at
    ``k``: the induced character's phases :math:`e^{-2\pi i\,k\cdot m}`
    conjugate and so does the site character.  So each partner label is the
    conjugate of the listed label whose row, read on the time-reversed columns,
    is its row: :math:`m_j(XA_{\pi(n)}) = m_{\bar j}(X_n)` for every column
    ``j``, ``j̄`` the column of the conjugate site irrep at the same position
    and ``j`` itself for a real one.  The tables name a partner's irreps as the
    conjugates of the listed point's, ``XA_n`` of ``X_n``, and that naming is
    taken wherever the rows hold it, which includes labels whose rows the
    columns do not tell apart; a different correspondence is taken only where
    it is the one the rows hold.

    Returns
    -------
    tuple
        ``((partner, listed point, ((listed label, partner label), ...)), ...)``
        in the physical convention of :func:`ebr_data`; empty for a table whose
        EBR rows list no such point or that ships no EBR rows.

    Raises
    ------
    ..refusal.InternalError
        No correspondence reproduces the rows, or several do and the tables'
        naming is not among them: the shipped rows and their conjugation
        disagree, which is a defect, never a label to guess.
    """
    from irreptables.ebrs import load_ebr_data
    from irreptables.irreps import IrrepTable
    try:
        raw = load_ebr_data(number_str, spinor)
    except Exception:                             # no EBR rows: nothing lists a partner
        return ()
    labels = [str(label) for label in raw["basis"]["irrep_labels"]]
    listed = {irr.kpname for irr in IrrepTable(number_str, spinor, magnetic=False).irreps}
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
                f"#{number_str}: the EBR rows list {arm}, which the irrep table does "
                "not, and it is not the partner -k of a point the table lists with as "
                "many irreps, so its labels cannot be read off any listed point.")
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
                    f"#{number_str} at {arm}: {len(every)} correspondences between its "
                    f"labels and the conjugates of {base}'s reproduce the EBR rows, and "
                    f"the tables' naming ({arm}n the conjugate of {base}n) is not among "
                    "them. The rows at a point -k and their conjugation disagree; that "
                    "is a defect in the shipped rows or in this reading of them, not a "
                    "label to choose.")
            pi = every[0]
        out.append((arm, base, tuple(sorted(pi.items()))))
    return tuple(out)


class HighSymmetryPoints(dict):
    """The tabulated high-symmetry points of the model's zone, by name, in the model's reduced coordinates.

    What :func:`high_symmetry_points` returns, a mapping ``{name: k}``;
    ``summary()`` prints it.

    Attributes
    ----------
    missing : tuple of str
        The tabulated points out of a layer's plane, ``A``, ``H`` and ``L`` of
        a hexagonal group, which its zone does not have at any mesh: a symmetry
        vector read on this mapping alone is partial there.
    """

    __slots__ = ("missing",)

    def __init__(self, points=(), *, missing=()):
        super().__init__(points)
        #: The tabulated names out of the periodic plane, in the tables' order.
        self.missing = tuple(missing)

    def summary(self) -> str:
        """The points, a line for each part, with the attribute that holds it."""
        lines = [f"{len(self)} point{'' if len(self) == 1 else 's'}, by name, in the model's "
                 "reduced coordinates   ([name])"]
        lines += [f"    {name} {_point_text(k)}" for name, k in self.items()]
        if self.missing:
            lines.append(f"  out of the layer's plane: {', '.join(self.missing)}   (.missing)")
        return "\n".join(lines)


def high_symmetry_points(group: GroupData) -> HighSymmetryPoints | Refusal:
    """The tabulated high-symmetry points, in the model's own reduced coordinates.

    The points a symmetry vector is read at.  The tables write them in their
    conventional cell, generally as other numbers: graphene's ``K`` is
    ``(1/3, 1/3, 0)`` there and ``(1/3, 2/3)`` in pythtb's honeycomb cell.
    Without time reversal a point and its partner ``−k`` are two points
    wherever no operation carries one onto the other, ``KA`` beside ``K``, and
    both are returned, the partner's irreps the complex conjugates of the
    point's.

    Parameters
    ----------
    group : GroupData
        The group, ``realized_group(model).group``.

    Returns
    -------
    HighSymmetryPoints or Refusal
        ``{name: k}``, each ``k`` with ``dim_k`` components; its ``missing``
        names the tabulated points out of a layer's plane.  ``summary()``
        prints it.  A ``Refusal`` in its place is falsy.

    Refusals
    --------
    A refusal handed in as ``group`` is returned as it is.

    ``tables_not_ready``
        The group has no transformation to the tables' conventional cell, to
        carry their points into the model's coordinates.
    """
    gate = _lattice_not_descended(group)
    if gate is not None:
        return gate
    if not group._tables_ready:
        return Refusal("tables_not_ready", group._setup_note or "irrep tables are not set up "
                       "for this cell", context={"number_str": group.number_str})
    out, missing = {}, []
    for name, k3 in _table_kpoints(group).items():
        if group.in_plane(k3[None, :], atol=1e-9):
            out[name] = group.k_from_3d(k3[None, :])[0]
        else:
            missing.append(name)
    return HighSymmetryPoints(out, missing=missing)


def _resolve_kpname(group: GroupData, lg: LittleGroup, kpname, ktol):
    """The tabulated label of the model's k, or a Refusal listing the options."""
    kmap = _table_kpoints(group)
    if kpname is not None:
        return kpname
    for name, k3 in kmap.items():
        d = lg.k3 % 1.0 - k3
        d -= np.round(d)
        if np.abs(d).max() <= ktol:
            return name
    avail = {n: np.round(group.k_from_3d(v[None, :])[0], 6)
             for n, v in kmap.items() if group.in_plane(v[None, :])}
    return Refusal(
        "kpoint_not_tabulated",
        f"k = {np.round(lg.k, 6)} is not one of the tabulated high-symmetry points of "
        f"#{group.number_str} in this cell. Tabulated (in this cell's reduced "
        f"coordinates): {avail}. A point in the star of a tabulated one but not the "
        f"tabulated arm carries a conjugate little group and is not transported here.",
        context={"k": lg.k, "available": avail, "all_points": kmap})


def _operations(table) -> tuple:
    """``(R, t, S, antiunitary)`` per tabulated symmetry, ``irreptables``' order."""
    return tuple((np.asarray(s.R, dtype=float), np.asarray(s.t, dtype=float),
                  np.asarray(s.S, dtype=complex), bool(s.time_reversal))
                 for s in table.symmetries)


def _as_representative(R, t, S, ops, centering, spinor) -> tuple:
    r"""Write a unitary ``{R|t}`` (spinor part ``S``) as ``σ·{E|m}·g_ℓ``.

    Returns ``(ℓ, m, σ)`` with ``ℓ`` a **one-based** ``irreptables`` symmetry
    index, ``m`` the lattice translation and ``σ = ±1`` the double-group sign
    (always ``+1`` for a single-valued table).  Raises when the element is not
    in the tabulated group at all, which nothing upstream can provoke.
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
    raise _EBRTransportError(                          # pragma: no cover - guard
        "a product of tabulated operations is not a tabulated operation times a "
        "lattice translation; the shipped symmetry list is not a group.")


def _herring_data(table, k, centering) -> tuple:
    r"""The convention-free half of the Wigner–Herring test at ``k``.

    Returns ``(H, coset, squares, twist)``:

    * ``H`` — one-based indices of the unitary little group of ``k``,
      ``R^{-T}k ≡ k`` modulo the **reciprocal** lattice;
    * ``coset`` — indices of the antiunitary elements with ``-R^{-T}k ≡ k``,
      i.e. of the extended little group's antiunitary coset.  Empty when no
      antiunitary fixes ``k`` at all (the ``K``/``KA`` "other arm" situation),
      and otherwise of size ``|H|``;
    * ``squares`` — ``(ℓ, m, σ)`` per ``a ∈ coset`` with
      ``a² = σ·{E|m}·g_ℓ``.  With ``a = uΘ`` and ``Θ S Θ⁻¹ = S`` (SU(2) is
      pseudo-real) this is ``a² = u²·Θ²``, so ``σ`` carries both the SU(2)
      accumulation of ``u²`` **and** ``Θ² = −1`` for a double-valued table;
    * ``twist`` — ``(ℓ, m, σ)`` per ``h ∈ H`` writing ``a₀⁻¹ h a₀`` (``a₀`` the
      first coset element) the same way.  ``Θ`` drops out of this one entirely,
      so it is the plain conjugation by ``a₀``'s unitary part.

    Nothing here depends on the ``e^{±ik·t}`` convention: the convention enters
    only when ``m`` is turned into the phase ``e^{s·2πi k·m}``.
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
        eps = -1.0 if spinor else 1.0                  # Θ² = −1 for spin ½
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
    r"""The corep list at one k-point, in the convention ``s`` names.

    ``s = +1`` is BCS's convention and ``s = −1`` the physical one; ``chars``
    maps each unitary little-group irrep label to its characters
    ``{ind: χ}`` **in that convention**.  Returns a sorted list of
    ``(constituent labels, type)`` with the Wigner type ``'a'``/``'b'``/``'c'``,
    or ``None`` when the data does not support the classification.

    The two ingredients, both evaluated from the table's own numbers:

    .. math::
        \bar D(h) = \overline{D(a_0^{-1} h a_0)},\qquad
        I_s(D) = \frac{1}{|H|}\sum_{a}\chi_D(a^2),

    with every ``{E|m}`` in either expression contributing ``e^{s·2πi k·m}``.
    ``I_s = +1`` (``\bar D ≅ D``) is type **a**, the corep is ``D`` alone;
    ``I_s = −1`` is type **b**, ``D ⊕ D``; ``I_s = 0`` (``\bar D ≇ D``) is type
    **c**, ``D ⊕ \bar D``.  Because *every* ``m`` here carries the convention,
    the corep list can — and at a minority of the tabulated k-points does — differ
    between ``s = +1`` and ``s = −1``: that is the fourth face of the convention
    artifact, and it changes which irrep doubles, hence the corep
    **dimensions**.
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
    """``('-A7', '-A8') → '-A7A8'`` — ``irreptables``' own composite-name pattern.

    The constituents are concatenated in **lexicographic** order of their bare
    names (which is why the tables print ``-A11A9``, not ``-A9A11``) with the
    double-valued ``-`` marker carried once at the front.  The rule reproduces
    every one of the 2700 shipped grey corep names from its own constituents,
    which the sweep in ``tests/test_groups.py`` checks.
    """
    bare = sorted(p[1:] if str(p).startswith("-") else str(p) for p in parts)
    return ("-" if double else "") + "".join(bare)


def _decompose_into(chars, H, vector):
    """Non-negative integer multiplicities of ``chars`` in ``vector``, or ``None``."""
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


@lru_cache(maxsize=None)
def _parent_table(number_str: str, spinor: bool):
    """The unitary (``magnetic=False``) table of a grey table's parent group."""
    from irreptables.irreps import IrrepTable
    return IrrepTable(str(number_str).split(".")[0], spinor, magnetic=False)


@lru_cache(maxsize=None)
def _unitary_conjugation(number_str: str, spinor: bool) -> tuple:
    """:func:`_table_conjugation` of the parent unitary table, cached and read-only."""
    return _table_conjugation(_parent_table(number_str, spinor))


def _lifted_conjugation(table) -> dict:
    r"""``{k-point: {corep label: physical corep label}}`` for a **grey** table.

    The whole fourth mechanism in one function.  A grey corep is a *composite*
    object — its BCS name spells out the unitary little-group irreps it is glued
    from — so its physical name is not a free choice: it is fixed by the
    physical names of its constituents, which the unitary table's own
    conjugation map ``c`` already decides.  Concretely, per tabulated k-point:

    1. decompose each shipped corep's characters into the parent unitary
       table's irrep characters (BCS convention, both tables' shipped numbers) —
       the gluing, read off rather than assumed;
    2. check it against the **BCS Herring indicators** computed from the table's
       own operations (:func:`_corep_gluing` at ``s = +1``).  A shipped table
       that violates its own convention's indicators is a data defect and is
       reported by returning nothing for that k-point;
    3. map every constituent through the unitary ``c`` and re-glue, naming the
       glued corep by :func:`_corep_name`;
    4. check *that* against the **physical Herring indicators** (``s = −1``),
       which is the independent confirmation that the re-gluing is right.

    Step 3 is exact rather than heuristic because
    ``χ_phys[c(D)] = conj(χ_BCS[D])`` gives, term by term in the Herring sum,
    ``I_{-1}(c(D)) = conj(I_{+1}(D)) = I_{+1}(D)``: the physical corep list is
    always the ``c``-image of the BCS one.  Where the image label is not a
    shipped label, the point has been **re-glued**: the same unitary irreps are
    partitioned differently, so the corep a given irrep sits in has a different
    dimension, and the EBR basis at that point changes with it.

    Because the unitary ``c`` is itself the character-matched naming, this lift
    usually *agrees* with matching the grey characters directly — BCS's own
    corep names spell BCS's own constituents, so the image of a
    conjugation-closed shipped set is a shipped set, which is why the re-glued
    residue is only 7 k-point entries.  It is still consulted **first**, because
    where the two differ the spelling has to win: at SG 203's spinful ``W`` the
    grey rows admit an exchange that the parent's do not, and taking it would
    give a corep a name whose constituents the unitary table does not name.

    Returns ``{}`` for a unitary table, and omits any k-point where the lift is
    not available (an "other arm" point the unitary table does not carry) or
    where a check above fails; :func:`_table_conjugation` then falls back to
    matching character vectors, or reports the point as broken.
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
            return {}                        # not the parent's grey extension
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
            continue                         # shipped gluing ≠ BCS's own indicators
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
    """``{name: other}`` with ``chi[other] == image(chi[name])``, or ``None``.

    The one primitive both naming rules of :func:`_table_conjugation` are built
    from: ``image = conj`` asks which *shipped* row prints what the physical
    convention prints (the name follows the character), ``image = conj/φ`` asks
    which shipped row the tabulated content ``Γ`` carries onto (the tabulated
    labelling).  ``None`` as soon as any row has no unique partner.
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
    r"""Per tabulated k-point: which BCS name each **physical** row is returned under.

    Returns ``(c, conjugated, broken)``.

    * ``c`` maps each tabulated label to the label whose **physical** character
      is the conjugate of its tabulated one, ``χ_phys[c(D)] = conj(χ_BCS[D])``.
      For a grey table the image may be a corep name the tables do not ship —
      see :func:`_lifted_conjugation` and the re-gluing paragraph below.
    * ``conjugated`` is the set of k-point names where :func:`_transported_table`
      returns the whole-character conjugate rather than the phase-flipped
      tabulated row — which it must do wherever ``c`` is not the pairing that
      keeping the tabulated ``Γ`` would realize.
    * ``broken`` lists k-points where neither branch applies, i.e. where the
      tabulated data is inconsistent with itself.

    **The name follows the character.**  Write
    ``w_D = conj(χ_D)`` for the whole-character conjugate; conjugating a
    representation conjugates its factor system *and* carries BCS's
    time-reversal gluing ``{Γ, conj(Γ)φ}`` onto the physical one
    ``{conj(Γ), Γ conj(φ)}`` (``φ(g) = e^{-4πi k·t_g}``), so **the w-set is
    always the set of physical little-group characters**.  The question this
    function answers is only what to *call* each ``w_D``, and the rule is: call
    it by the BCS row it prints.  So ``c(D)`` is the shipped label ``E`` with
    ``χ_E = w_D``, whenever every ``D`` has one.  The returned printed table is
    then literally BCS's printed table — same names against the same numbers,
    so the time-reversal doubling rules read as they do in BCS's book — and the
    convention divergence shows up only in *which* row a physical band lands
    on, which is what the EBR relabelling ``n^phys_ρ[c(D)] = n^BCS_{ρ*}[D]``
    carries (:func:`_ebr_transport`).

    Where the shipped character set is **not closed under conjugation** no such
    naming exists: no permutation of BCS's rows is the physical table, and the
    label has to be anchored to something else — to what the name *spells*,
    i.e. to the constituents the parent unitary table already named
    (:func:`_lifted_conjugation`), falling back to the pairing ``v_{c(D)} =
    w_D`` that keeping the tabulated ``Γ`` realizes (``v_D = χ_D·φ``), and
    finally to the identity.  59 of the 5462 tabulated (space group,
    valuedness, k-point) entries are in that class, every one of them with at
    least one complex character; the other 5403 are returned under BCS's own
    names.

    What the 59 mean is *not* that two conventions disagree about a doubling.
    The entrywise conjugate of the returned table is the shipped table exactly
    (that is ``χ_phys[c(D)] = conj(χ_BCS[D])`` read backwards), and time
    reversal maps the ``k`` sector antiunitarily onto the ``−k`` one, so the
    shipped page there tabulates the content of the **other star member**: at
    SG 80's ``P`` the returned ``{-P3 (1,−i), -P4P4 (2,+2i)}`` conjugates to the
    shipped ``{-P4 (1,+i), -P3P3 (2,−2i)}``.  The doubling itself is
    convention-independent — it is the trace of the zero-translation
    little-group generator on the degenerate level — which is exactly why no
    relabelling is admissible there.

    Three mechanisms put a k-point in the "not the tabulated pairing" branch,
    and all are checked so that a *tabulation* defect is never mistaken for a
    convention artifact:

    1. the tabulated factor system ``ω₀(R,S) = e^{2πi (Rᵀk − k)·t_S}`` is not
       real — a 3₁/4₁/6₁-type screw at a zone-boundary point (SG 91/92/95/96
       ``A``/``Z``/``R``, 98/122 ``N``/``P``, 151-154 and 178-181 ``A``/``H``/
       ``L``, 210 ``L``/``W``, 212/213 ``M``/``R``/``X``), where keeping ``Γ``
       is not a representation of ``ω₀*`` at all;
    2. ``φ`` is not real, so BCS's time-reversal gluing of a **grey** corep is
       not the physical one (this arises only in grey tables: SG 76/78
       ``A``/``Z``, 109/110 ``P``, 144/145 ``A``, 169-172 ``A``/``H``);
    3. the **gluing itself** is convention-sensitive: the Wigner–Herring
       indicators that decide which unitary irrep doubles carry the same
       ``e^{s·2πi k·m}`` phases, so at a minority of the tabulated k-points the
       physical corep list is a *different partition* of the same unitary
       irreps — different constituents, sometimes different dimensions.  A grey
       corep's *name* is therefore never decided by character matching where the
       parent unitary table has already spoken: the
       lift (:func:`_lifted_conjugation`) is consulted first, so that the name
       always spells the constituents the unitary table names.  Where the two
       agree — the common case, because BCS's own names
       spell BCS's own constituents — the lift *is* the character
       matching; where they disagree the lift wins, which is the only way a
       renamed corep can keep meaning what it says (SG 203's spinful ``W`` is
       such a point: the grey rows admit an exchange that the parent's do not,
       so the exchange is refused there).

    A k-point where **no** naming rule applies while ``ω₀`` *and* ``φ`` are both
    real is none of the three: there the two conventions produce identical
    numbers, so a set that is neither conjugation-closed nor ``φ``-paired would
    be a defect of the shipped table and is returned in ``broken``, and
    ``broken`` is empty for all 460 unitary and all 460 grey instances.  SG 80's
    grey spinful ``P`` is the closest call: it is a mechanism-3 point, named by
    the lift, and its shipped ``-P3P3 ⊕ -P4`` is exactly what BCS's own
    indicators require in BCS's convention.
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
        elif pairs is not None:               # the tabulated labelling is physical
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
    """``(conjugated k-points, ((label, image), ...))`` of one table.

    Cached per ``(number_str, spinor, magnetic)``: it is a property of
    ``irreptables``' tables, not of any model or cell.  The pair is returned
    together because both halves come out of the same
    :func:`_table_conjugation` pass, which loads the parent unitary table.
    """
    from irreptables.irreps import IrrepTable
    conj, conjugated, _broken = _table_conjugation(
        IrrepTable(number_str, spinor, magnetic=magnetic))
    return conjugated, tuple(sorted(conj.items()))


def _conjugated_kpoints(number_str: str, spinor: bool, magnetic: bool) -> frozenset:
    """k-point names of one table returned as whole-character conjugates."""
    return _transport_naming(number_str, spinor, magnetic)[0]


@lru_cache(maxsize=None)
def _divergent_kpoints(number_str: str, spinor: bool, magnetic: bool) -> frozenset:
    """k-point names whose returned table is **not** a permutation of BCS's rows.

    The public census of where a reader cross-checking against the Bilbao
    Crystallographic Server will not
    find the returned numbers: everywhere else the name follows the character
    (:func:`_table_conjugation`) and the returned table *is* BCS's printed table,
    so only the label a measured band lands on can differ from BCS's own
    analysis.  Here the shipped character set is not closed under conjugation,
    so no permutation of BCS's rows is the physical table at all — its
    *entrywise conjugate* is, which is the statement that the shipped page
    tabulates the content of the other star member ``−k``.

    Of the 5462 tabulated (space group, valuedness, k-point) entries, 59 are in
    this class — 25 unitary and 34 grey — every one with at least one complex
    character, and all at ``P``/``W`` of a body- or face-centred group
    (SG 24/73/80/88/98/109/110/122/141/142/199/203/206/210/214/220/227/228/230
    and their grey partners).
    """
    from irreptables.irreps import IrrepTable
    table = IrrepTable(number_str, spinor, magnetic=magnetic)
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


def _transported_table(group: GroupData, kpname: str, k3) -> dict | Refusal:
    r"""``{label: {ind: character}}`` on **this cell's** coset representatives.

    ``SpaceGroupIrreps.get_irreps_from_table`` re-implemented, with the
    **factor system conjugated** at the boundary, reading the tabulated
    characters out of ``irreptables``.

    *The convention statement.*  BCS — and therefore ``irrep``/``irreptables`` —
    tabulates little-group representations in the historically **unphysical**
    convention in which a lattice translation is represented by
    :math:`e^{+2\pi i\,k\cdot t}`.  The physical convention, the one
    :mod:`..sewing` measures in (``(g f)(r) = f(g^{-1}r)`` acting on Bloch
    functions :math:`\psi_k \sim e^{+ik\cdot r}`, so
    :math:`\psi_k(r-m) = e^{-2\pi i\,k\cdot m}\psi_k(r)`), represents it by
    :math:`e^{-2\pi i\,k\cdot t}`.  The mismatch is a historical artifact of the
    tables and lives **entirely in the projective structure**: the tabulated
    character factorises as

    .. math::
        \chi_\text{table}(\{R|t_\text{table}\})
            = e^{+2\pi i\,k_\text{ref}\cdot t_\text{table}}\;\Gamma(R),

    where :math:`\Gamma(R)` — the loaded, point-group half — is gauge-free.  The
    boundary rule is therefore *conjugate the factor system and nothing else*:
    keep :math:`\Gamma(R)` exactly as tabulated and re-attach the **physical**
    phase on this cell's own coset representative,

    .. math::
        \chi_\text{phys}(\{R|t_\text{phys}\})
            = e^{-2\pi i\,k_\text{ref}\cdot t_\text{phys}}\;\Gamma(R)
            = \chi_\text{table}\;\texttt{sym.sign}\;
              e^{-2\pi i\,(t_\text{table} + t_\text{phys})\cdot k_\text{ref}},

    with :math:`t_\text{phys}` from ``sym.translation_refUC(refUC, shiftUC)``
    and everything on the
    right in the BCS conventional basis.  ``k`` itself, the star/k-point
    matching, the time-reversal corep pairing and the tabulated point-group
    content are all left untouched: flipping ``k`` instead would repair the same
    multiplicity defect while exchanging distinct stars (SG 199's ``P`` and
    ``PA``, distinct exactly because the group has no inversion) and disturbing
    the corep pairing.

    The setting transformation is ``irrep``'s own.  ``sym.sign`` is ``+1`` on
    every operation of a group this package built: ``irrep`` fits a ``±1`` per
    operation from an unseeded random start and carries it on the *characters*,
    while :func:`.._spacegroup._pin_spinor_branch` computes it from the tables' own spin frame
    (:func:`.._spacegroup._canonical_spinor_signs`) and applies it to the **lifts** instead.
    That is what makes the returned *names* reproducible and the returned *rows*
    BCS's own: the little group's stored lifts then realize the tabulated ℤ₂
    factor system exactly, so the tabulated character needs no bar at all.
    Against
    ``irrep``'s formula (``χ_table · sign · exp(+2πi·δt·k_ref)`` with
    ``δt = t_table − t_phys``) this differs by ``exp(-4πi·t_table·k_ref)``,
    which is 1 whenever ``k·t_table`` is a half-integer — i.e. everywhere except
    a k-point that is not time-reversal invariant *and* whose little group
    carries fractional translations.

    *Where keeping* ``Γ`` *is not available: whole-character conjugation.*  The
    rule above keeps the tabulated ``Γ`` and re-attaches the physical phase,
    which is a
    representation of the physical factor system :math:`\omega_0^*` only where
    :math:`\omega_0` is **real**, and a physical *corep* only where the phase
    :math:`\varphi(g) = e^{-4\pi i k\cdot t_g}` that carries BCS's
    time-reversal gluing into the physical one is real as well.  Both conditions have the
    same visible consequence, and it is the one tested here (via
    :func:`_table_conjugation`): the tabulated set of character vectors is
    closed under conjugation-composed-with-``φ`` exactly when the tabulated
    labelling already names the physical representations.  Where it is not,
    what is returned is the **whole-character conjugate**

    .. math::
        \chi_\text{phys}(\{R|t_\text{phys}\})
            = \overline{\Gamma(R)}\;e^{-2\pi i\,k_\text{ref}\cdot t_\text{phys}}
            = \overline{\chi_\text{table}(\{R|t_\text{phys}\})},

    under the name :math:`c(D)` that :func:`_table_conjugation` supplies.
    :math:`D\mapsto D^*` is the canonical bijection from the
    :math:`\omega_0`-projective representations to the
    :math:`\omega_0^*`-projective ones (conjugating a representation conjugates
    its factor system, and it carries BCS's gluing
    :math:`\{\Gamma,\overline{\Gamma}\varphi\}` onto the physical one
    :math:`\{\overline\Gamma, \Gamma\overline\varphi\}`).

    *And the name follows the character.*  ``c(D)`` is the **BCS label of the
    row returned here** wherever the shipped character set is closed
    under conjugation, so at those points the returned table is BCS's printed
    table verbatim, whichever branch was taken; only where it is not closed is
    the label anchored to what the name spells instead — for a **grey** table
    to whatever the parent unitary table's relabelling forces on the corep's
    constituents, which at the re-glued points is a corep the tables do not
    ship at all (SG 80's ``-P3P3 ⊕ -P4`` becomes ``-P3 ⊕ -P4P4``: same four
    units of dimension, different partition).

    Over the 5462 tabulated (space group, valuedness, k-point) entries — 460
    unitary and 460 grey instances, with ``broken`` empty —
    **685** are returned by conjugating the whole character (398 unitary, 287
    grey), and at **59** of those the printed rows are not a permutation of
    BCS's shipped rows at all (25 unitary, 34 grey) — the only entries where a
    reader cross-checking against the Bilbao Crystallographic Server will not
    find the returned numbers under
    the returned name, and the only ones where a returned corep name can be one the
    tables do not ship (7 entries; SG 80's spinful ``P`` is the anchor).  At the
    other 5403 the printed table is BCS's — **element by element**, because the
    spinor matching sign of :func:`.._spacegroup._canonical_spinor_signs` lives in the lift
    and not on the character, so the row printed under a returned name is a row
    ``irreptables`` ships under that name, entry for entry, with only the
    physical translation phase re-attached.  Carried on the character instead it would
    print a bar wherever the stored lifts realize a different ℤ₂ factor system
    from the tables' (``K`` of ``#183.186``: ``-K4`` as ``(+i, -i, -i)`` over
    the three mirrors against BCS's ``(-i, -i, -i)``).

    *A partner point* ``−k`` *the character table does not list* (``KA`` and
    the like, :func:`_partner_points`) is read as its listed point's table
    conjugated: the little group is the same, and the conjugate of a physical
    irrep at ``k`` is one at ``−k``.

    Refusals
    --------
    ``kpoint_not_tabulated``
        The label and the coordinates disagree, or the label is not tabulated:
        both are verdicts on what the shipped tables carry.
    """
    from irreptables.irreps import IrrepTable
    # A POINT -k THE IRREP TABLE DOES NOT LIST: its little group is the listed
    # point's, and its characters are the listed point's conjugated, each under
    # the partner label the EBR rows give the conjugate (`_partner_points`)
    if not group.magnetic:
        for arm, base, naming in _partner_points(group.number_str, group.spinor):
            if kpname != arm:
                continue
            rows = _transported_table(group, base, (-np.asarray(k3, dtype=float)) % 1.0)
            if isinstance(rows, Refusal):
                return rows
            naming = dict(naming)
            return {naming[label]: {ind: complex(np.conj(c)) for ind, c in row.items()}
                    for label, row in rows.items()}
    table = IrrepTable(group.number_str, group.spinor, magnetic=group.magnetic)
    sg = group._sg
    conjugated = kpname in _conjugated_kpoints(group.number_str, group.spinor,
                                               group.magnetic)
    naming = dict(_transport_naming(group.number_str, group.spinor,
                                    group.magnetic)[1])
    k2 = np.round(np.asarray(k3, dtype=float), 5) % 1
    out: dict[str, dict[int, complex]] = {}
    for irr in table.irreps:
        if irr.kpname != kpname:
            continue
        k_ref = np.asarray(irr.k, dtype=float)
        k1 = np.round(np.linalg.inv(group._refUC.T).dot(k_ref), 5) % 1
        if not np.allclose(k1, k2):
            return Refusal(
                "kpoint_not_tabulated",
                f"the kpoint {k2} does not correspond to the point {kpname} "
                f"({np.round(k_ref, 3)} in refUC / {k1} in primUC) in the table",
                context={"kpname": kpname, "k": k2,
                         "number_str": str(group.number_str)})
        row: dict[int, complex] = {}
        for i, (sym, sym_table) in enumerate(zip(sg.symmetries, table.symmetries)):
            ind = i + 1
            if ind not in irr.characters:
                continue                      # not in the little group (or antiunitary)
            t_table = np.asarray(sym_table.t, dtype=float)
            t_ours = sym.translation_refUC(group._refUC, group._shiftUC)
            #  Γ(R) = χ_table · e^{-2πi k·t_table};  χ_phys = Γ · e^{-2πi k·t_ours}
            if conjugated:                    # ... with Γ → conj(Γ), name c(D)
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
            f"the k-point with name {kpname} is not found in the spacegroup "
            f"{table.number_str}",
            context={"kpname": kpname, "k": k2,
                     "number_str": str(group.number_str)})
    return out


def corep_table(group: GroupData, k, *, kpname=None, ktol=1e-5) -> dict | Refusal:
    """The character table of the little group at a high-symmetry point: each corep's character on each unitary operation.

    The table measured characters are read against, in the package's
    convention for a lattice translation, so characters measured on the model
    compare with it directly; the points where that convention moves a label,
    or changes how time reversal glues the irreps, are named in
    ``ebr_data(group)["convention"]``.  Operations combined with time reversal
    carry no characters.

    Parameters
    ----------
    group : GroupData
        The group, ``realized_group(model).group``.
    k : array, shape (dim_k,)
        A tabulated point, in the model's reduced coordinates, as
        :func:`high_symmetry_points` gives it.
    kpname : str, optional
        The point's tabulated name, taken as given instead of found from
        ``k``.
    ktol : float, optional
        How far ``k`` may sit from a tabulated point and be it, in reduced
        reciprocal coordinates.  Default ``1e-5``.

    Returns
    -------
    dict or Refusal
        ``{corep: {ind: character}}``, one entry per corep of the little group,
        its character on each unitary operation by index.  A ``Refusal`` in
        its place is falsy.

    Refusals
    --------
    A refusal handed in as ``group`` is returned as it is.

    ``tables_not_ready``
        The group has no transformation to the tables' conventional cell.
    ``out_of_plane``
        ``k`` has a component along a layer's vacuum.
    ``kpoint_not_tabulated``
        ``k`` is no tabulated point in the model's coordinates, or not the
        tabulated member of its star; the refusal lists the tabulated points.
    """
    gate = _lattice_not_descended(group)
    if gate is not None:
        return gate
    if not group._tables_ready:
        return Refusal("tables_not_ready",
                       group._setup_note or "irrep tables are not set up for this cell",
                       context={"number_str": group.number_str})
    lg = little_group(group, k)
    if isinstance(lg, Refusal):
        return lg
    name = _resolve_kpname(group, lg, kpname, ktol)
    if isinstance(name, Refusal):
        return name
    return _transported_table(group, name, lg.k3 % 1.0)


def identify_from_characters(group: GroupData, k, characters, *, multiplet_dims=None,
                             kpname=None, label_tol=1e-3, ktol=1e-5) -> IrrepContent | Refusal:
    r"""The coreps each multiplet carries at a high-symmetry point, from its measured characters.

    Each row of characters is expanded over the coreps of the little group,
    ``m = Σ_g χ_corep(g) χ(g)* / Σ_g |χ_corep(g)|²``, and a multiplicity is a
    nonnegative integer or the characters are wrong.  Levels merged into one
    multiplet read as a sum of coreps.  :func:`..sewing.corep_content`
    measures the characters on a set of bands and reads them here.

    Parameters
    ----------
    group : GroupData
        The group the characters were measured in,
        ``realized_group(model).group``.
    k : array, shape (dim_k,)
        A tabulated point, in the model's reduced coordinates, as
        :func:`high_symmetry_points` gives it.
    characters : array, shape (n_multiplets, n_unitary)
        A row per degenerate multiplet and a column per unitary operation of
        the little group in index order, ``little_group(group, k).indices``:
        the trace over the multiplet's states of ``rep.sewing_operator(ind,
        k)``, as :func:`..sewing.characters` measures it.
    multiplet_dims : array, shape (n_multiplets,), optional
        Each multiplet's degeneracy, which the coreps found must add up to.
    kpname : str, optional
        The point's tabulated name, taken as given instead of found from
        ``k``.
    label_tol : float, optional
        How far a multiplicity may sit from an integer, and two rows'
        characters from each other to share a number in the partition,
        dimensionless.  Default ``1e-3``.
    ktol : float, optional
        How far ``k`` may sit from a tabulated point and be it, in reduced
        reciprocal coordinates.  Default ``1e-5``.

    Returns
    -------
    IrrepContent or Refusal
        Each multiplet's coreps and dimension, and the partition of the
        multiplets by their characters, which stands under any labelling and
        is the statement to quote where a label rests on one; ``summary()``
        prints it.  A ``Refusal`` in its place is falsy.

    Raises
    ------
    ValueError
        ``characters`` has not one column per unitary operation of the little
        group, or ``multiplet_dims`` not one entry per row.

    Refusals
    --------
    A refusal handed in as ``group`` is returned as it is.

    ``tables_not_ready``
        The group has no transformation to the tables' conventional cell.
    ``out_of_plane``
        ``k`` has a component along a layer's vacuum.
    ``kpoint_not_tabulated``
        ``k`` is no tabulated point in the model's coordinates, or not the
        tabulated member of its star; the refusal lists the tabulated points.
    ``little_group_mismatch``
        The little group measured in the model's cell is not the table's.
    ``integer_multiplicity``
        A multiplicity further than ``label_tol`` from a nonnegative integer:
        characters in another convention, or one level split across
        multiplets, which reads one half for a two-dimensional corep.
    ``multiplet_dimension``
        The coreps found do not add up to ``multiplet_dims``: the rows were
        grouped into multiplets wrongly.
    """
    gate = _lattice_not_descended(group)
    if gate is not None:
        return gate
    if not group._tables_ready:
        return Refusal("tables_not_ready",
                       group._setup_note or "irrep tables are not set up for this cell",
                       context={"number_str": group.number_str})

    lg = little_group(group, k)
    if isinstance(lg, Refusal):
        return lg

    chars = np.atleast_2d(np.asarray(characters, dtype=complex))
    if chars.shape[1] != len(lg.unitary):
        raise ValueError(
            f"characters must have one column per unitary little-group operation "
            f"({len(lg.unitary)} at k={np.round(lg.k, 6)}, indices {lg.indices}), "
            f"got {chars.shape[1]}")

    kpname = _resolve_kpname(group, lg, kpname, ktol)
    if isinstance(kpname, Refusal):
        return kpname
    table = corep_table(group, lg.k, kpname=kpname, ktol=ktol)
    if isinstance(table, Refusal):
        return table

    table_inds = set(next(iter(table.values())).keys())
    if not set(lg.indices) <= table_inds:
        return Refusal(
            "little_group_mismatch",
            f"the measured little group at {kpname} carries the operation indices "
            f"{sorted(lg.indices)} but the table lists {sorted(table_inds)}; matching the "
            f"operations to the standard setting did not close, so no label would be meaningful.",
            context={"ours": sorted(lg.indices), "table": sorted(table_inds), "kpname": kpname})

    label_order = tuple(table)
    mult = np.zeros((chars.shape[0], len(label_order)), dtype=complex)
    dims = np.zeros(len(label_order))
    # the identity is in every little group; its character is the irrep dimension
    identity = next((int(s.ind) for s in lg.unitary
                     if np.array_equal(s.rotation, np.eye(3, dtype=int))
                     and np.abs(s.translation - np.round(s.translation)).max() < 1e-8),
                    min(lg.indices))
    for c, lab in enumerate(label_order):
        ir = np.array([table[lab][i] for i in lg.indices], dtype=complex)
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
            f"irrep multiplicities at {kpname} are not non-negative integers (worst defect "
            f"{defect:.3g} > {label_tol:g}) for multiplet row(s) {rows}. Either the characters were "
            f"not computed in the convention pythtqc.groups documents (its module docstring), or "
            f"a degeneracy was split into several multiplets by numerical noise.",
            value=defect, threshold=float(label_tol),
            context={"kpname": kpname, "k": lg.k, "indices": lg.indices,
                     "labels": label_order, "multiplicities": mult,
                     "characters": chars, "rows": rows})

    total_dims = rounded @ dims
    if multiplet_dims is not None:
        want = np.asarray(multiplet_dims, dtype=float).ravel()
        if want.shape[0] != chars.shape[0]:
            raise ValueError("multiplet_dims must have one entry per character row")
        if not np.allclose(total_dims, want, atol=1e-6):
            return Refusal(
                "multiplet_dimension",
                f"identified irrep dimensions {total_dims.tolist()} do not match the supplied "
                f"multiplet degeneracies {want.tolist()} at {kpname}; the degeneracy grouping "
                f"and the characters disagree.",
                value=float(np.abs(total_dims - want).max()), threshold=1e-6,
                context={"kpname": kpname, "got": total_dims, "expected": want})

    labels = tuple({lab: int(rounded[r, c]) for c, lab in enumerate(label_order)
                    if rounded[r, c] != 0} for r in range(chars.shape[0]))

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

    return IrrepContent(kpname=kpname, k=lg.k, indices=lg.indices, labels=labels,
                        multiplicities=mult, label_order=label_order, dims=total_dims,
                        partition=tuple(int(p) for p in part), integrality=defect)


def _label_kpoint(label: str) -> str:
    """``'-GM4GM5' → 'GM'`` — the leading run of non-digits after the ``-`` marker.

    The same three lines as :func:`kpoint_of_label`, kept beside the carrying
    of the rows, which reads this one.
    """
    s = str(label)[1:] if str(label).startswith("-") else str(label)
    i = 0
    while i < len(s) and not s[i].isdigit():
        i += 1
    return s[:i]


def _conjugate_site_name(ebr_name: str) -> str | None:
    """``'1E↑G(1)' → '2E↑G(1)'`` — the complex-conjugate *site* irrep, or ``None``.

    BCS names a complex-conjugate pair of site representations ``1X``/``2X``
    (``1E``/``2E``, ``1E1``/``2E1``, ``-1Eg``/``-2Eg``, …), with the ``-``
    marking a double-valued one; every other name is self-conjugate, and in a
    grey group a conjugate pair that time reversal has already glued carries
    both halves in one name (``1E2E``, ``-1E2E1``) and is self-conjugate too.
    Returns ``None`` when the name is self-conjugate by that rule.

    Measured, not assumed: over **all 230 space groups plus their grey
    extensions, single- and double-valued — 10 074 EBR columns — this rule
    agrees with the Γ-block requirement below in every case and disagrees in
    none**.  The Γ check is applied at run time anyway.
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
    r"""``{label: label}`` — the map ``c`` with ``χ_phys[c(D)] = conj(χ_BCS[D])``.

    At most tabulated k the physical-convention table is the tabulated one with
    its factor system conjugated (:func:`_transported_table`), i.e. on the
    table's own coset representatives

    .. math::
        \chi^\text{phys}_D(\{R|t\}) = \chi^\text{BCS}_D(\{R|t\})\,
                                      e^{-4\pi i\,k\cdot t},

    and ``c`` is fixed by matching character vectors — no naming convention is
    consulted.  ``c`` is always an **involution**: writing
    :math:`\varphi(g) = e^{-4\pi i k\cdot t_g}` (unimodular),
    :math:`\chi_{c(D)}\varphi = \overline{\chi_D}` applied twice gives
    :math:`\chi_{c(c(D))}\varphi = \overline{\chi_{c(D)}}
    = \chi_D\overline{\varphi^{-1}} = \chi_D\varphi`, hence ``c(c(D)) = D``.
    That is checked here rather than trusted.

    Labels of an EBR-basis k-point the *irrep* table does not carry — the
    ``KA``/``HA``/``PA``/``WA`` "other arm" points, which the grey (corep)
    tables do carry but the unitary ones mostly do not — are mapped by
    transporting ``c`` from the partner point through the name (``KA2`` ↔
    ``K2``).  That is BCS's own arm convention, and it is verified against the
    grey tables, where both arms are tabulated with
    :math:`\chi_{XA_n} = \overline{\chi_{X_n}}` at :math:`k_{XA} = -k_X`.

    **Where** :func:`_transported_table` **returns the whole-character conjugate**
    — the k-points :func:`_table_conjugation` marks, where the tabulated
    labelling names no physical representation and the physical table is
    :math:`\overline{\chi_\text{BCS}}` — the definition above is unchanged, and
    it is also what fixes the returned *name*: ``c(D)`` is the
    BCS label whose shipped characters are :math:`\overline{\chi_D}` wherever
    one exists, so the EBR relabelling and the returned table move together by
    construction.  Only where no such label exists does ``c`` come from
    somewhere else — for a **grey** table from the corep relabelling lifted from
    the parent unitary table, whose image at the re-glued points is a corep name
    the EBR basis does not contain.  The transport formula is still the same
    relabelling — the derivation of
    :math:`n^\text{phys}_\rho[c(C)] = n^\text{BCS}_{\rho^*}[C]` only uses
    :math:`\chi_\text{phys}[c(C)] = \overline{\chi_\text{BCS}[C]}`, which the
    lift satisfies constituent by constituent — but the *basis itself* moves
    there, so ``c`` is a bijection onto a label set that is not the one it
    started from.  It is therefore checked to be an involution only where the
    image is a label of the same basis, and to preserve dimension always.

    Raises :class:`_EBRTransportError` only for a **tabulation** defect: a
    k-point whose irreps disagree about which operations are in the little
    group, or one where the tabulated character set is not conjugation-closed
    although both ``ω₀`` and ``φ`` are real and no gluing statement can explain
    it either.  No point shipped in ``irreptables`` is of either kind.
    """
    conj, _conjugated, broken = _table_conjugation(table)
    if broken:
        raise _EBRTransportError(
            f"the tabulated little-group characters of #{table.number_str} at "
            f"{list(broken)} are not a consistent set: they are not closed under "
            "complex conjugation, and neither the tabulated factor system nor the "
            "time-reversal gluing phase is complex there, so the e^{+ik.t}/e^{-ik.t} "
            "convention difference cannot account for it. This is a defect of the "
            "shipped table, not of the transport, so the EBR rows are refused "
            "rather than carried across it.", points=list(broken))

    for label in labels:                       # the untabulated 'A' arms
        if label in conj:
            continue
        kp = _label_kpoint(label)
        # ``len(kp) > 1`` keeps the hexagonal ``A`` point out of the arm rule.
        base = label.replace(kp, kp[:-1]) if len(kp) > 1 and kp.endswith("A") else None
        if base is None or base not in conj:
            raise _EBRTransportError(
                f"the EBR-basis label {label!r} (k-point {kp!r}) is not in the "
                "irrep table and is not the 'other arm' of one that is, so its "
                "convention transport is undefined.", points=[kp])
        conj[label] = conj[base].replace(kp[:-1], kp)

    images = {conj[label] for label in labels}
    if len(images) != len(labels):
        raise _EBRTransportError(
            "the conjugation map is not injective on the EBR basis "
            f"({len(labels)} labels, {len(images)} images); the tabulated "
            "characters are not what this transport assumes.")
    for label in labels:
        image = conj[label]
        if image in conj and conj[image] != label:
            raise _EBRTransportError(
                f"the conjugation map is not an involution at {label!r} "
                f"({label} → {image} → {conj.get(image)}); "
                "the tabulated characters are not what this transport assumes.",
                points=[_label_kpoint(label)])
        if _label_kpoint(image) != _label_kpoint(label):
            raise _EBRTransportError(          # pragma: no cover - structural
                f"the conjugation map moves {label!r} to another k-point "
                f"({image!r}); the transport is per k-point by construction.",
                points=[_label_kpoint(label)])
    return conj


@lru_cache(maxsize=None)
def _ebr_transport(number_str: str, spinor: bool, magnetic: bool) -> tuple:
    r"""Permutations carrying ``irreptables``' EBR rows into the physical convention.

    Returns ``(labels, conjugation, partner, changed_labels, new_labels, perm)``:
    the tabulated EBR-basis label order, the per-k conjugation ``c`` of
    :func:`_conjugation_involution`, the index ``ρ ↦ ρ*`` of the time-reversed
    EBR column, which labels move, the **physical** basis labels in the
    same positions, and the row permutation that goes with them.  Cached per
    ``(number_str, spinor, magnetic)`` — it is a property of the tables, not of
    any model.

    The physics in one line.  A band representation induced from the site
    representation ``ρ`` at Wyckoff position ``q`` has, at the k-point ``k``,
    the character

    .. math::
        \chi^{(s)}(\{R|t\}) = \sum_{j:\,g q_j = q_j}
            e^{s\,2\pi i\,k\cdot m_j}\,\chi_\rho(g_j),
        \qquad m_j = R\tau_j + t - \tau_j\in\mathbb{Z}^3 ,

    with ``s = +1`` in BCS's convention (what ``irreptables`` ships) and
    ``s = −1`` in the physical one (what :mod:`..sewing` measures).  Because
    :math:`\chi_\rho(g_j)` and :math:`m_j` are convention-free,

    .. math::
        \chi^{(-)}_\rho = \overline{\chi^{(+)}_{\rho^*}},

    where :math:`\rho^*` is the complex-conjugate site representation — note
    that conjugating only the phase would be wrong for a complex ``1E``/``2E``
    site pair.  Pairing that with the definition of ``c`` gives the whole
    transport as a **relabelling**,

    .. math::
        n^\text{phys}_\rho[\,c(D)\,] = n^\text{BCS}_{\rho^*}[\,D\,],

    which is what comes back.  Two consequences are worth naming:
    the Wyckoff **position never moves** (``ρ*`` sits at ``q`` too), and at
    ``k = 0`` the transport is the identity — asserted below on the Γ block of
    every column, which is the run-time check that ``ρ*`` was identified right.

    *Re-glued k-points.*  Where ``c`` carries a grey corep label to one the
    tables do not ship, the formula above is unchanged but the **basis** at that
    k-point is not: the physical corep list is a
    different partition of the same unitary irreps, so the labels there are
    renamed in place rather than permuted (``-P3P3, -P4 → -P4P4, -P3`` for
    ``80.30``, multiplicities ``1, 2 → 1, 2`` — read positionally, which is what
    ``n^phys[c(C)] = n^BCS[C]`` says).  ``c`` preserves the dimension of every
    constituent, so the ``degeneracies`` array is untouched and the per-k-point
    dimension ``Σ_C n_C · dim C`` is conserved; both are asserted here.

    Raises :class:`_EBRTransportError` (never returns partly transported rows).
    """
    from irreptables.ebrs import load_ebr_data
    from irreptables.irreps import IrrepTable

    raw = load_ebr_data(number_str, spinor)
    labels = tuple(str(l) for l in raw["basis"]["irrep_labels"])
    try:
        names = [str(e["ebr_name"]) for e in raw["ebrs"]]
        wyck = [str(e["wyckoff_position"]) for e in raw["ebrs"]]
    except KeyError as exc:                       # e.g. 80.31 ships no position
        raise _EBRTransportError(
            f"the EBR data of #{number_str} is missing {exc}; it cannot be "
            "carried into the physical convention (nor read by irrep's own "
            "`get_ebr_names_and_positions`).") from exc
    matrix = np.array([e["vector"] for e in raw["ebrs"]], dtype=int).T

    table = IrrepTable(number_str, spinor, magnetic=magnetic)
    conj = _conjugation_involution(table, labels)

    degen = {l: int(d) for l, d in zip(labels, raw["basis"]["degeneracies"])}
    bad = [l for l in labels
           if conj[l] in degen and degen[l] != degen[conj[l]]]
    if bad:
        raise _EBRTransportError(
            f"conjugation changes the dimension of {bad}; the tabulated "
            "degeneracies and characters disagree.",
            points=[_label_kpoint(l) for l in bad])

    partner = []
    for i, (name, pos) in enumerate(zip(names, wyck)):
        want = _conjugate_site_name(name)
        hits = [j for j, (n2, p2) in enumerate(zip(names, wyck))
                if n2 == want and p2 == pos]
        if want is None or not hits:
            partner.append(i)                     # self-conjugate site representation
        elif len(hits) == 1:
            partner.append(hits[0])
        else:
            raise _EBRTransportError(
                f"the EBR data of #{number_str} lists {want!r}@{pos} more than "
                "once, so the time-reversed partner of "
                f"{name!r}@{pos} is ambiguous.")
    if any(partner[partner[i]] != i for i in range(len(partner))):
        raise _EBRTransportError(
            f"the time-reversal pairing of #{number_str}'s EBR columns is not an "
            "involution.")

    index = {l: i for i, l in enumerate(labels)}
    #  Per k-point block: where ``c`` maps the block onto itself the labels stay
    #  put and the multiplicities permute (the ordinary case: the returned rows are
    #  then the shipped rows permuted); where it does not — a re-glued point —
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
                    f"the Γ block of {names[i]}@{wyck[i]} is not invariant under "
                    "the transport (k = 0 carries no site-displacement phase, so "
                    "it must be): the conjugate-site pairing or the tabulated "
                    "Γ characters are not what this transport assumes.",
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
                    f"the transport changes the dimension of {names[i]}@{wyck[i]} "
                    f"at {kp} ({was} → {now}); the re-glued corep basis is not a "
                    "partition of the same unitary irreps.", points=[kp])

    moved = tuple(l for l, l2, j in zip(labels, new_labels, perm)
                  if l != l2 or any(matrix[index[l], i] != matrix[j, partner[i]]
                                    for i in range(matrix.shape[1])))
    return labels, conj, tuple(partner), moved, new_labels, tuple(int(p) for p in perm)


def _to_physical_convention(raw: dict, number_str: str, spinor: bool,
                            magnetic: bool) -> dict:
    """``raw`` with every EBR row rewritten in the physical convention.

    A new ``dict`` (``irreptables`` re-reads its JSON on every call, so nothing
    shared is mutated): the ``vector``/``irrep_list`` of each EBR, the basis
    ``irrep_labels`` where a k-point was re-glued, and the Smith form, which is
    transported exactly rather than recomputed — with ``M' = P M Q`` for the
    label permutation ``P`` and the column permutation ``Q`` (both involutions),
    ``M = U⁻¹RV⁻¹`` gives ``M' = (UP)⁻¹R(QV)⁻¹`` with the elementary divisors
    ``R`` untouched.  A re-glued block contributes the identity to ``P`` (its
    labels are renamed in place, not permuted), so the Smith form is carried
    across it unchanged, which is right: the multiplicity *vector* does not move
    there, only what its entries are called.
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

    # ``r`` — the elementary divisors — is invariant under both permutations, and
    # each of ``u``/``v`` is rewritten only when its permutation is not the
    # identity: ``irreptables`` stores a degenerate Smith factor as a scalar when
    # the group has a single EBR column (measured on SG 4, 7, 9, 29, …), and
    # there the column permutation is the identity anyway.
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
                        f"#{number_str}'s tabulated Smith factor {key!r} is not a "
                        f"matrix (ndim={arr.ndim}), so the transport cannot be "
                        "carried onto it.")
                out["smith_form"][key] = (arr[:, order] if axis else
                                          arr[order, :]).tolist()
    return out


def _transport_record(number_str: str, spinor: bool, magnetic: bool) -> dict:
    """The ``"convention"`` entry :func:`ebr_data` publishes — what moved, and why."""
    labels, conj, partner, moved, new_labels, _perm = _ebr_transport(
        number_str, spinor, magnetic)
    reglued = tuple(sorted({_label_kpoint(l) for l, l2 in zip(labels, new_labels)
                            if l != l2}))
    return {
        "name": "physical (translations as e^{-ik.t})",
        "applied": True,
        "changed_labels": moved,
        "changed_points": tuple(sorted({_label_kpoint(l) for l in moved})),
        "conjugated_points": tuple(sorted(
            _conjugated_kpoints(number_str, spinor, magnetic))),
        "divergent_points": tuple(sorted(
            _divergent_kpoints(number_str, spinor, magnetic))),
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

    A letter's site group is every operation of the group that fixes its
    representative point at a generic value of its free parameters, the point
    and the operations both ``pyxtal``'s in the tables' setting
    (:func:`_pyxtal_wyckoff`: the general position's operations, the
    conventional cell's centring among them).  It is named by
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
    """``"3a(3m1',3m)" → "3a(-3m1',-3m)"``: an EBR position, its site group with its bar.

    The tables' own EBR files write every improper site group without its bar:
    ``3m`` for the ``-3m`` at ``3a`` of ``R-3m``, ``6m2`` for the ``-6m2`` at
    ``2c`` of ``P6/mmm``, ``4`` for ``-4``.  At a letter of ``improper``
    (:func:`_improper_letters`) the bar goes back on the tables' own symbol,
    which keeps the orientation that symbol writes: which axis carries which
    operation, and so which irrep is ``B1`` and which ``B2``.
    """
    match = re.match(r"^(\d+)([a-zA-Z])\((.*)\)$", position)
    if match is None or match.group(2) not in improper:
        return position
    symbols = ",".join(part if part.startswith("-") else "-" + part
                       for part in match.group(3).split(","))
    return f"{match.group(1)}{match.group(2)}({symbols})"


def ebr_data(group: GroupData) -> dict | Refusal:
    """Every elementary band representation of the group: the bands each site irrep at each Wyckoff position induces.

    The atomic limits the group has, and the columns a symmetry vector is
    decomposed over; :func:`..ebr.decompose` and
    :func:`..ebr.symmetry_indicator` read them, a layer's restricted to the
    points of its plane.  The rows are in the package's convention for a
    lattice translation, ``e^{-ik·t}``, where the tables' own pages write
    ``e^{+ik·t}``: away from the time-reversal-invariant momenta a row pairs
    labels with Wyckoff positions otherwise than the Bilbao Crystallographic
    Server's page does, and ``convention`` says where.

    Parameters
    ----------
    group : GroupData
        The group, ``realized_group(model).group``; only its label, its
        valuedness and whether it is grey are read.

    Returns
    -------
    dict or Refusal
        ``matrix``: the EBR matrix, a row per corep at each high-symmetry
        point of the group, those out of a layer's plane included, and a
        column per elementary band representation;
        ``labels_physics_convention``: the rows' coreps, in the labels
        :func:`identify_from_characters` returns;
        ``ebrs``: each column as ``(name, position)``, the position with its
        site group, an improper one with its bar, ``"3a(-3m1',-3m)"``;
        ``smith``: the Smith normal form ``(U, D, V)``, ``U·M·V = D``;
        ``convention``: what the convention moved, by point and by label, with
        a ``note``: ``changed_points`` and ``changed_labels``,
        ``conjugated_points``, ``divergent_points``, ``reglued_points``;
        ``conjugation``, each label's partner, and ``tr_partner``, each
        column's, the column of the complex-conjugate site irrep;
        ``labels_bcs``, the server's names of the rows;
        and ``row_permutation``, ``P`` with ``y_BCS[j] = y[P[j]]``, which
        re-keys a form written in the server's labels;
        ``raw``: the tables' own record, carried into this convention.
        A ``Refusal`` in its place is falsy.

    Refusals
    --------
    A refusal handed in as ``group`` is returned as it is.

    ``irrep_tables``
        ``irreptables`` is not installed.
    ``ebr_tables``
        ``irreptables`` ships no elementary band representations for this
        group and valuedness.
    ``ebr_convention``
        An entry of the shipped file with no Wyckoff position, so its row
        cannot be carried into this convention; every known case is a magnetic
        group of type III, outside the groups returned.
    """
    gate = _lattice_not_descended(group)
    if gate is not None:
        return gate
    status = tables_status()
    if not status["tables"]:
        return _refuse_no_tables(status)
    from irrep.ebrs import get_ebr_matrix, get_ebr_names_and_positions, get_smith_form
    from irreptables.ebrs import load_ebr_data

    try:
        raw = load_ebr_data(group.number_str, group.spinor)
    except (FileNotFoundError, OSError) as exc:
        return Refusal("ebr_tables",
                       f"irreptables has no EBR data for group #{group.number_str} "
                       f"({'double' if group.spinor else 'single'} valued): {exc}",
                       context={"number_str": group.number_str, "spinor": group.spinor})
    try:
        raw = _to_physical_convention(raw, group.number_str, group.spinor,
                                      group.magnetic)
        record = _transport_record(group.number_str, group.spinor, group.magnetic)
    except _EBRTransportError as exc:
        return Refusal(
            "ebr_convention",
            f"the EBR rows of #{group.number_str} cannot be carried out of BCS's "
            f"e^(+ik.t) convention into this package's physical one: {exc}",
            context={"number_str": group.number_str, "spinor": group.spinor,
                     "magnetic": group.magnetic, "points": exc.points})
    except (FileNotFoundError, OSError) as exc:
        return Refusal("irrep_tables",
                       f"irreptables has no irrep table for group #{group.number_str}, "
                       f"so its EBR rows cannot be transported: {exc}",
                       context={"number_str": group.number_str, "spinor": group.spinor})
    improper = _improper_letters(int(str(group.number_str).split(".")[0]))
    return {"raw": raw, "matrix": get_ebr_matrix(raw),
            "labels_physics_convention": list(raw["basis"]["irrep_labels"]),
            "ebrs": [(str(name), _barred(str(position), improper))
                     for name, position in get_ebr_names_and_positions(raw)],
            "smith": get_smith_form(raw), "convention": record}


#: Why :func:`indicator_group` withholds ``irreptables``' named forms from a
#: group that is not grey, one string per case.
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
        "coefficients are keyed by BCS labels and must be re-keyed by "
        "ebr_data(group)['convention']['row_permutation'] before they are evaluated "
        "on a returned symmetry vector (in scope that bites once, the z2 of 220.90). "
        "ebr.symmetry_indicator does that re-keying, and checks the form on the "
        "group's own EBRs, before it reports a value"),
    "grey_none": (
        "irreptables ships no named indicator formula for this grey group, a "
        "verdict in itself -- read `elementary_divisors` for the indicator group "
        "itself, and ebr.symmetry_indicator for a class"),
}


def indicator_group(group: GroupData) -> dict | Refusal:
    """The group's symmetry-indicator group, and the tables' named indicator formulas for it.

    For a band set's class, :func:`..ebr.symmetry_indicator` is the call: it
    reads the class off the Smith normal form, and names it with a formula only
    after checking that the formula vanishes on every elementary band
    representation of the group.  This one returns the group's own data, a
    layer's out-of-plane points included.

    Parameters
    ----------
    group : GroupData
        The group, ``realized_group(model).group``; only its label, its
        valuedness and whether it is grey are read.

    Returns
    -------
    dict or Refusal
        ``elementary_divisors``: the divisors above 1 of the Smith normal form
        of the group's EBR matrix, the indicator group ``⊕ ℤ_d``; ``None``
        where :func:`ebr_data` refuses, and that refusal is the one to read.
        ``indicators``: the tables' named formulas, ``{name: {"factors",
        "mod"}}``, returned for a grey group only; ``orders``,
        ``{name: mod}``, each the order of one formula, so that equal or
        dependent formulas make them something other than the divisors.
        ``withheld``: whether formulas exist and are left out, as for every
        group that is not grey, ``reason`` saying why; ``indicators`` empty
        with ``withheld`` False is a verdict, no formula named for this group.
        ``note``: the caveat to read either way, that a formula's coefficients
        are keyed by the tables' labels and are re-keyed by
        ``ebr_data(group)["convention"]["row_permutation"]`` before they are
        read on a returned symmetry vector.  A ``Refusal`` in its place is
        falsy.

    Refusals
    --------
    A refusal handed in as ``group`` is returned as it is.

    ``irrep_tables``
        ``irreptables`` is not installed.
    """
    gate = _lattice_not_descended(group)
    if gate is not None:
        return gate
    status = tables_status()
    if not status["tables"]:
        return _refuse_no_tables(status)
    from irreptables.symmetry_indicators import load_si_table

    data = ebr_data(group)
    divisors = None
    if not isinstance(data, Refusal):
        r = np.asarray(data["smith"][1])
        # single EBR column: R is stored flat, and the vector is then already the
        # diagonal extended by the empty rows (the shape .._smith._smith_system normalizes)
        d = r if r.ndim == 1 else r.diagonal()
        divisors = d[d > 1]

    if _magnetic_type(group) != "II":
        note = _SI_NOTES["unaudited_bns" if group.magnetic else "bare_key"]
        return {"indicators": {}, "orders": {}, "elementary_divisors": divisors,
                "withheld": True, "reason": note, "note": note}

    si = load_si_table(spinor=group.spinor, magnetic=group.magnetic)
    entry = si.get(group.number_str, {}).get("indicators", {})
    return {"indicators": entry,
            "orders": {k: v["mod"] for k, v in entry.items()},
            "elementary_divisors": divisors,
            "withheld": False, "reason": None,
            "note": _SI_NOTES["grey" if entry else "grey_none"]}


def _ebr_solve_shim(ebr_matrix, y, original_exc):
    """``irrep.ebrs.compute_ebr_decomposition`` with the ortools-9.15 status fix.

    Identical model and callback to ``irrep``'s (``varArraySolutionObtainer`` is
    reused verbatim); only ``solver.status_name()`` — which ortools 9.15 broke —
    is replaced by comparing the enum ``Solve`` returns.
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
    """A symmetry vector written in the group's elementary band representations.

    For a model's band set, :func:`..ebr.decompose` is the call: it reads a
    layer's vector in its plane and lists the decompositions.  This one takes a
    vector over every high-symmetry point of the group and reads it as ``irrep``
    does.

    Parameters
    ----------
    group : GroupData
        The group, ``realized_group(model).group``.
    irrep_counts : dict
        ``{corep: multiplicity}``, the band set's coreps at every high-symmetry
        point of the group, as ``HSPTable.counts`` holds them.

    Returns
    -------
    dict or Refusal
        ``classification``: ``"stable topological"`` where no integer
        combination of elementary band representations reproduces the vector,
        ``"fragile topological"`` where only one with a negative coefficient
        does, and ``"atomic limit"`` where a nonnegative one does, which an
        atomic limit needs and which does not make one; ``stable_nontrivial``,
        whether it is the first; ``solutions``, the combinations found,
        written out in ``decomposition_strings``, and ``is_positive``, whether
        one is nonnegative; ``y`` and ``y_prime``, the vector and its Smith
        coordinates; ``ebrs``, the columns; ``unused``,
        the coreps handed in that no column carries; ``indicators``, the
        tables' named formulas summed over the vector, ``None`` where they are
        withheld, with ``indicators_note`` saying why, or the formulas'
        caveat.  A ``Refusal`` in its place is falsy.

    Refusals
    --------
    A refusal handed in as ``group`` is returned as it is.

    ``incomplete_content``
        A high-symmetry point of the group carries no corep in the vector,
        which would read as no bands there; ``context["missing_points"]`` names
        each.  A layer's vector misses the points out of its plane, and
        :func:`..ebr.decompose` reads it.
    ``ortools``
        Telling an atomic limit from fragile bands needs the integer solver
        ``ortools``, which is not installed.
    ``ebr_tables``, ``irrep_tables``, ``ebr_convention``
        From :func:`ebr_data`.
    """
    gate = _lattice_not_descended(group)
    if gate is not None:
        return gate
    data = ebr_data(group)
    if isinstance(data, Refusal):
        return data
    from irrep.ebrs import (ORTOOLS_AVAILABLE, compose_ebr_string,
                            compute_ebr_decomposition, compute_topological_classification_vector)

    basis = data["labels_physics_convention"]
    unused = sorted(set(irrep_counts) - set(basis))
    # COMPLETE, or nothing.  `create_symmetry_vector` reads an unmentioned point
    # as zero, and a zero row is a constraint -- the wrong
    # one -- so a vector with a point missing decomposes into the EBRs of a
    # different band set.  A slab is where this bites: the EBR basis of the 3D
    # space group carries k_z = 1/2 points its zone does not have.
    points = sorted({_label_kpoint(lab) for lab in basis})
    covered = {_label_kpoint(lab) for lab in irrep_counts}
    missing_points = [p for p in points if p not in covered]
    if missing_points:
        return Refusal(
            "incomplete_content",
            f"the symmetry vector carries no label at {missing_points}, and the EBR "
            f"basis of #{group.number_str} is written over {points}. A decomposition "
            "of a partial symmetry vector is a confident wrong answer: an unmentioned "
            "point is read as zero content there, which is a constraint on a "
            "different band set. Supply counts covering "
            "every point, or -- for a 2D model, whose zone does not contain the "
            "basis's out-of-plane points at all -- use ebr.decompose, which "
            "restricts the EBR matrix to the measurable rows and checks that the "
            "restriction loses nothing.",
            value=float(len(missing_points)), threshold=0.0,
            context={"missing_points": tuple(missing_points), "points": tuple(points)})
    y, y_prime, nontrivial = compute_topological_classification_vector(irrep_counts, data["raw"])

    si = indicator_group(group)
    indicators, indicators_note = None, None
    if not isinstance(si, Refusal):
        # whether the named forms are returned is `withheld`, not the emptiness of
        # the mapping: a grey group can ship no form at all, which is a verdict --
        # and it is returned as the empty sum, `{}`, not as `None`.
        indicators_note = si["reason"] if si["withheld"] else si["note"]
        if not si["withheld"]:
            indicators = {}
            for name, spec in si["indicators"].items():
                total = sum(v * irrep_counts.get(lab, 0)
                            for lab, v in spec["factors"].items())
                indicators[name] = total % spec["mod"]

    out = {"y": y, "y_prime": y_prime, "stable_nontrivial": bool(nontrivial),
           "indicators": indicators, "indicators_note": indicators_note,
           "ebrs": data["ebrs"], "unused": unused,
           "solutions": None, "is_positive": None}
    if nontrivial:
        out["classification"] = "stable topological"
        return out
    if not ORTOOLS_AVAILABLE:
        return Refusal("ortools",
                       "the band set has integer EBR solutions, so it is trivial or fragile, "
                       "but telling them apart needs the ortools integer solve, which is not "
                       "installed. Install it: pip install 'ortools>=9.10'",
                       context={"y": y, "y_prime": y_prime})
    try:
        solutions, is_positive = compute_ebr_decomposition(data["raw"], y)
    except (TypeError, AttributeError) as exc:
        # ortools 9.15 broke CpSolver.status_name(): its internal
        # `_checked_response.status` became an attribute while cp_model.py still
        # calls it, so irrep's ebrs.get_solutions raises. Everything else in the
        # ortools path works, so run irrep's own model with the status enum
        # compared directly. The shim is deliberately minimal and defers to
        # irrep whenever irrep works.
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

def k_grid(group: GroupData, n, *, allow_asymmetric=False, ktol=1e-8) -> KGrid | Refusal:
    """A Γ-centred mesh that respects the group, with the high-symmetry points it holds.

    A mesh holds only the high-symmetry points it happens to contain: ``Γ``
    always, and the zone boundary only with an even number of divisions.  A
    label read off a mesh at a point it misses is no label, so ``contains``
    names the points it holds and ``hsp`` gives every tabulated point's
    coordinates.  The mesh is pythtb's own, ``k`` in ``[0, 1)``, periodic in
    the gauge a ``WFArray`` reads.

    Parameters
    ----------
    group : GroupData
        The group, ``realized_group(model).group``.
    n : int or sequence of int
        The divisions along each periodic direction, or one number for all.
    allow_asymmetric : bool, optional
        Return a mesh the group does not carry onto itself, with no
        irreducible points, instead of refusing it.  Default ``False``.
    ktol : float, optional
        How far a mesh point may sit from a tabulated point and be it, in
        reduced reciprocal coordinates.  Default ``1e-8``.

    Returns
    -------
    KGrid or Refusal
        The mesh, ready for a ``pythtb.WFArray``, with the tabulated points it
        holds and those it misses; ``summary()`` prints it.  A ``Refusal`` in
        its place is falsy.

    Raises
    ------
    ValueError
        ``n`` is a sequence of other than ``dim_k`` numbers.

    Refusals
    --------
    A refusal handed in as ``group`` is returned as it is.

    ``asymmetric_grid``
        The group does not carry the mesh onto itself, so a sum over it would
        weigh points unlike their partners; ``allow_asymmetric=True`` returns
        it all the same.
    """
    gate = _lattice_not_descended(group)
    if gate is not None:
        return gate
    from pythtb import Mesh
    if np.isscalar(n):
        divs = [int(n)] * group.dim_k
    else:
        divs = [int(v) for v in n]
        if len(divs) != group.dim_k:
            raise ValueError(f"n must be a scalar or have {group.dim_k} entries")
    grid3 = [1, 1, 1]
    for j, d in enumerate(group.periodic_dirs):
        grid3[d] = divs[j]

    symmetric = bool(group._sg.is_grid_symmetrical(grid3))
    if not symmetric and not allow_asymmetric:
        return Refusal("asymmetric_grid",
                       f"the {tuple(grid3)} mesh is not symmetric under #{group.number_str}; "
                       f"reading symmetry-related quantities off it double-counts. Pass "
                       f"allow_asymmetric=True to override deliberately.",
                       context={"grid": tuple(grid3), "number_str": group.number_str})

    mesh = Mesh(["k"] * group.dim_k)
    mesh.build_grid(tuple(divs), gamma_centered=False, k_endpoints=False)
    points = np.asarray(mesh.flat)

    irr = None
    if symmetric:
        irr3 = group._sg.get_irreducible_kpoints_grid(grid3)
        irr = group.k_from_3d(irr3)

    hsp = high_symmetry_points(group)
    contains, missing, hsp_out = {}, [], {}
    if not isinstance(hsp, Refusal):
        for name, kk in hsp.items():
            hsp_out[name] = kk
            hit = bool(np.any(np.abs(((points - kk) + 0.5) % 1.0 - 0.5).max(axis=1) <= ktol))
            contains[name] = hit
            if not hit:
                missing.append(name)

    return KGrid(divisions=tuple(grid3), symmetric=symmetric, mesh=mesh, irreducible=irr,
                 contains=contains, missing=tuple(missing), hsp=hsp_out)
