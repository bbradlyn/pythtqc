r"""Site irreps named by the tables of the stable real-space invariants.

Where the list of elementary band representations has no entry at a Wyckoff
position, a row of a site's character table is named by a site irrep of the
tables of the stable real-space invariants (Hwang et al., Nat. Commun. 17,
8564 (2026)).  It is the one whose band representation has the same
invariants as the row's.  Where the row's multiplicities decompose into
elementary band representations, every invariant is compared, and the two
band representations can then be deformed into one another once the same
orbitals are added to both.  Where no decomposition is found, only the
integer-valued invariants are compared.  The names are computed once, by
inducing each row and matching its invariants (:func:`_name_by_invariants`),
and shipped in ``data/site_irrep_names.json``.  Reading a name induces
nothing.

A name is read at a site by carrying the site to its position's own site: the
position's first point in pyxtal's list, in the conventional cell used by the
irreptables package, at a fixed generic value of its free parameters.  The
points of a position with free parameters form a line or a plane, cut into
stretches by the more special positions on it.  A site is carried, by an
operation of the group, onto the stretch holding a reference point without
crossing a more special position, and its operations are carried with it.
The shipped table is keyed by a row's characters on the carried operations.
So every site of an orbit reads one name, whichever site a model lists first,
and a group with time reversal and the same space group without it read the
same names.
"""

from __future__ import annotations

from functools import lru_cache
import itertools
import json
import os

import numpy as np

from . import _spacegroup
from ._tolerance import WYCKOFF_REFERENCE_TOL, _INT_TOL, _ON_VARIETY

#: The parameters a position's reference points are tried at, in order; the
#: first is the generic parameter every Wyckoff position is placed at, so the
#: first reference point is the position's own site.
_CANDIDATES = (
    (0.13, 0.29, 0.41), (0.37, 0.11, 0.23), (0.61, 0.83, 0.07), (0.21, 0.47, 0.17),
    (0.43, 0.19, 0.31), (0.07, 0.39, 0.47), (0.29, 0.07, 0.11), (0.47, 0.41, 0.03),
    (0.87, 0.67, 0.93), (0.79, 0.93, 0.59), (0.93, 0.57, 0.71), (0.03, 0.53, 0.97),
    (0.71, 0.23, 0.89), (0.97, 0.79, 0.37), (0.57, 0.97, 0.79), (0.83, 0.13, 0.53),
    (0.19, 0.89, 0.73), (0.53, 0.61, 0.83))

#: The lattice vectors a carried point is shifted by when it is placed on the
#: variety of the position's first arm.
_NEAR = np.array(list(itertools.product(range(-2, 3), repeat=3)), dtype=float)

_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                     "site_irrep_names.json")


@lru_cache(maxsize=None)
def _document():
    """The shipped table of names, read once.

    Returns the contents of ``data/site_irrep_names.json``, or ``None`` where
    the file is missing or unreadable.
    """
    try:
        with open(_PATH, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


@lru_cache(maxsize=None)
def _operations(sg: int) -> tuple:
    """The operations of space group ``sg`` in the tables' conventional cell.

    They are the operations the irreptables package lists for the space group,
    each times every centering vector of pyxtal's list in the same setting.
    Each rotation is an integer matrix, and each translation is snapped to a
    rational of small denominator.  They depend on the space group alone, and
    so are the same for a group with time reversal and without.  Returns
    ``((R, t), ...)``.
    """
    table = _spacegroup._irrep_table(str(int(sg)), False, False)
    _positions, centering = _spacegroup._pyxtal_wyckoff(int(sg))
    out = []
    for sym in table.symmetries:
        R = np.rint(np.asarray(sym.R, dtype=float)).astype(int)
        t = np.asarray(_spacegroup._snap_rational(np.asarray(sym.t, dtype=float)), dtype=float)
        for c in np.atleast_2d(centering):
            out.append((R, t + np.asarray(c, dtype=float)))
    return tuple(out)


@lru_cache(maxsize=None)
def _first_arm(sg: int, letter: str):
    """The first arm of position ``letter`` of space group ``sg``.

    The arm is ``(A0, b0, P0)`` in the tables' setting: the points of the
    position are ``A0 p + b0`` for its free parameters ``p``, and ``P0``
    projects onto the directions off them
    (:func:`.._spacegroup._pyxtal_wyckoff`).  Returns ``None`` where ``sg`` has
    no position ``letter``.
    """
    positions, _centering = _spacegroup._pyxtal_wyckoff(int(sg))
    for position in positions:
        if position["letter"] == str(letter):
            return position["arms"][0]
    return None


def _fixed_by(ops, point) -> int:
    """How many of ``ops`` fix ``point``, modulo the lattice."""
    count = 0
    for R, t in ops:
        v = R @ point + t - point
        count += int(np.abs(v - np.rint(v)).max() < _ON_VARIETY)
    return count


def _path_clear(z, p0, ops) -> bool:
    """Whether the segment from ``z`` to ``p0`` meets no more special position.

    ``ops`` are the operations ``(R, t)`` of :func:`_operations`, and the
    segment's points are ``z(s) = z + s (p0 - z)`` for ``0 <= s <= 1``.  An
    operation fixes ``z(s)`` where :math:`(R - 1) z(s) + t` is an integer
    vector.  One for which that vector does not vary along the segment fixes
    all of it or none of it, and is passed over.  Returns ``False`` where
    another operation fixes a point of the segment, its ends included.
    """
    one = np.eye(3)
    for R, t in ops:
        a = (R - one) @ z + t
        d = (R - one) @ (p0 - z)
        if np.abs(d).max() < _ON_VARIETY:
            continue
        i = int(np.argmax(np.abs(d)))
        lo, hi = sorted((a[i], a[i] + d[i]))
        for m in range(int(np.floor(lo - _ON_VARIETY)), int(np.ceil(hi + _ON_VARIETY)) + 1):
            s = (m - a[i]) / d[i]
            if s < -_ON_VARIETY or s > 1 + _ON_VARIETY:
                continue
            v = a + s * d
            if np.abs(v - np.rint(v)).max() < _ON_VARIETY:
                return False
    return True


def _carrier(ops, arm, p0, x, tol):
    """The first operation carrying ``x`` onto the stretch of ``p0``.

    ``ops`` are the operations of :func:`_operations`, ``arm`` the position's
    first arm (:func:`_first_arm`), ``p0`` a reference point on it and ``x`` a
    point, all in the tables' conventional cell.  An operation carries ``x``
    there where an image :math:`R x + t + n`, ``n`` a lattice vector of
    :data:`_NEAR`, lies within ``tol`` of the arm's points, and the segment
    from it, placed on them, to ``p0`` crosses no more special position
    (:func:`_path_clear`).  Returns the operation's index in ``ops``, or
    ``None`` where none carries ``x`` there.
    """
    _A0, b0, P0 = arm
    for k, (R, t) in enumerate(ops):
        zs = (R @ x + t)[None, :] + _NEAR
        off = (zs - b0[None, :]) @ P0.T
        on = np.abs(off).max(axis=1) <= tol
        for z, delta in zip(zs[on], off[on]):
            if _path_clear(z - delta, p0, ops):
                return k
    return None


@lru_cache(maxsize=None)
def _stretch_parameters(sg: int, letter: str) -> tuple:
    """The parameters of the reference point of each stretch of a position.

    The position is ``letter`` of space group ``sg``, and the parameters are
    tried in the order of :data:`_CANDIDATES`.  A candidate is kept where its
    point is fixed by as many operations as the generic point is, and no
    operation carries it onto a stretch already kept (:func:`_carrier`).
    Returns the parameters kept, in order, or ``()`` where ``sg`` has no
    position ``letter``.
    """
    arm = _first_arm(int(sg), str(letter))
    if arm is None:
        return ()
    ops = _operations(int(sg))
    A0, b0, _P0 = arm
    generic = _fixed_by(ops, A0 @ np.asarray(_CANDIDATES[0]) + b0)
    kept, points = [], []
    for q in _CANDIDATES:
        p = A0 @ np.asarray(q) + b0
        if _fixed_by(ops, p) != generic:
            continue
        if any(_carrier(ops, arm, r, p, _ON_VARIETY) is not None for r in points):
            continue
        kept.append(q)
        points.append(p)
    return tuple(kept)


@lru_cache(maxsize=None)
def _stretches(sg: int, letter: str) -> tuple:
    """The reference point of each stretch of a position, in the tables' cell.

    The position is ``letter`` of space group ``sg``.  Stretch ``0``'s point is
    the position's own site.  Returns the points in the order of
    :func:`_stretch_parameters`, or ``()`` where ``sg`` has no position
    ``letter``.
    """
    arm = _first_arm(int(sg), str(letter))
    if arm is None:
        return ()
    A0, b0, _P0 = arm
    return tuple(tuple(float(v) for v in A0 @ np.asarray(q) + b0)
                 for q in _stretch_parameters(int(sg), str(letter)))


@lru_cache(maxsize=4096)
def _carry(sg: int, letter: str, x_ref: tuple):
    """The stretch a site lies in, and the operation carrying it there.

    ``x_ref`` is a site of position ``letter`` of space group ``sg``, in the
    tables' conventional cell.  It is carried within ``WYCKOFF_REFERENCE_TOL``
    of the position's points (:func:`_carrier`), the stretches tried in order.
    Returns ``(stretch, k)``, ``k`` the operation's index in
    :func:`_operations`, or ``None`` where ``sg`` has no position ``letter`` or
    nothing carries the site onto a stretch.
    """
    arm = _first_arm(int(sg), str(letter))
    if arm is None:
        return None
    ops = _operations(int(sg))
    x = np.asarray(x_ref, dtype=float)
    for stretch, p0 in enumerate(_stretches(int(sg), str(letter))):
        k = _carrier(ops, arm, np.asarray(p0), x, WYCKOFF_REFERENCE_TOL)
        if k is not None:
            return stretch, k
    return None


def carry_to_own_site(group, letter, position, indices):
    r"""A site carried to its position's own site, with its operations.

    ``group`` is the group the site was measured in, ``letter`` the site's
    Wyckoff letter in the tables' setting, ``position`` the site in reduced
    coordinates of the model's cell, and ``indices`` the ``.ind`` of operations
    fixing it, unitary and antiunitary.  An operation :math:`g` of the space
    group, in the tables' cell, carries the site onto a stretch of its position
    (:func:`_carry`).  Each operation :math:`h_k` of ``indices`` is carried to
    :math:`g h_k g^{-1}`, which fixes that stretch's reference point.  It is
    found among the operations of ``group`` by its rotation and whether it is
    antiunitary, which name one operation of a site group, and where several
    match, by which fixes the reference point most nearly.  Returns
    ``(stretch, carried)``, ``carried[k]`` the ``.ind`` :math:`h_k` is carried
    to.  Returns ``None`` where the tables are not set up for ``group``, the
    letter is not one of its positions, the site reaches no stretch, or an
    operation has no image.  Reads no table of names and needs no tables of
    the invariants.
    """
    if not getattr(group, "_tables_ready", False):
        return None
    sg = int(group._sg_number)
    x_ref = _spacegroup._in_tables_cell(group, position)[0]
    found = _carry(sg, str(letter), tuple(round(float(v), 12) for v in x_ref))
    if found is None:
        return None
    stretch, k = found
    R_g = _operations(sg)[k][0]
    R_g_inv = np.rint(np.linalg.inv(R_g)).astype(int)
    refUC = np.asarray(group._refUC, dtype=float)
    inv = np.linalg.inv(refUC)
    shift = np.asarray(group._shiftUC, dtype=float)
    own = refUC @ np.asarray(_stretches(sg, str(letter))[stretch]) + shift
    by_ind = {int(s.ind): s for s in group._symmetries}
    carried = []
    for ind in indices:
        h = by_ind.get(int(ind))
        if h is None:
            return None
        R_ref = np.rint(inv @ np.asarray(h.rotation, dtype=float) @ refUC).astype(int)
        R_c = refUC @ (R_g @ R_ref @ R_g_inv) @ inv
        best, defect = None, np.inf
        for s in group._symmetries:
            if bool(s.time_reversal) != bool(h.time_reversal):
                continue
            if np.abs(np.asarray(s.rotation, dtype=float) - R_c).max() > _INT_TOL:
                continue
            v = np.asarray(s.rotation, dtype=float) @ own + np.asarray(s.translation) - own
            miss = float(np.abs(v - np.rint(v)).max())
            if miss < defect:
                best, defect = int(s.ind), miss
        if best is None:
            return None
        carried.append(best)
    return stretch, tuple(carried)


def _keyed_rows(group, table, carried):
    r"""A site table's rows as the shipped table keys them.

    ``table`` is a site table of ``group``, and ``carried`` the ``.ind`` each
    of ``table.indices`` is carried to (:func:`carry_to_own_site`).  The key
    is the carried unitary operations' rotations in the tables' cell,
    :math:`\mathrm{refUC}^{-1} R\,\mathrm{refUC}`, each as its nine integers,
    sorted.  Returns ``(rotations, rows)``, ``rows[r]`` row ``r``'s characters
    on those rotations in that order as integers, or ``None`` for a row with a
    character that is not a real integer.
    """
    refUC = np.asarray(group._refUC, dtype=float)
    inv = np.linalg.inv(refUC)
    by_ind = {int(s.ind): s for s in group._symmetries}
    rotation_of = {}
    for ind, c in zip(table.indices, carried):
        R = inv @ np.asarray(by_ind[int(c)].rotation, dtype=float) @ refUC
        rotation_of[int(ind)] = tuple(int(v) for v in np.rint(R).ravel())
    rotations = tuple(sorted(rotation_of.values()))
    place = {R: i for i, R in enumerate(rotations)}
    rows = []
    for row in range(len(table.labels)):
        chi = [0] * len(rotations)
        for ind in table.indices:
            c = table.character_on(row, int(ind))
            if abs(c.imag) > _INT_TOL or abs(c.real - round(c.real)) > _INT_TOL:
                chi = None
                break
            chi[place[rotation_of[int(ind)]]] = int(round(c.real))
        rows.append(None if chi is None else tuple(chi))
    return rotations, tuple(rows)


def table_names(group, assignment, site, table) -> tuple:
    """The shipped table's name of every row of one site's table.

    ``site`` is a site of ``assignment``, the orbitals' assignment to
    ``group``, and ``table`` its site table.  The names are read at the site
    itself, carried to its position's own site (:func:`carry_to_own_site`),
    never at another site of its orbit.  Returns one name per row, ``""`` for a
    row the entry does not name, such as a row two site irreps tie at, or a
    row with a character that is not a real integer (a member of a pair time
    reversal does not join).  Every name is ``""`` where the table of
    names is missing or holds no entry at the group's space group and the
    site's letter, the site has no letter, the site is not carried onto a
    stretch the entry lists (:func:`carry_to_own_site`), or its carried
    operations are not those the entry is written on.
    """
    blank = ("",) * len(table.labels)
    document = _document()
    letters = tuple(getattr(assignment, "wyckoff_tables", ()) or ())
    if document is None or int(site) >= len(letters) or not letters[int(site)]:
        return blank
    letter = letters[int(site)]
    entries = document.get("groups", {}).get(str(int(group._sg_number)), {}).get(letter)
    if not entries:
        return blank
    indices = tuple(int(i) for i in table.indices)
    found = carry_to_own_site(group, letter, assignment.site_positions[int(site)], indices)
    if found is None or found[0] >= len(entries):
        return blank
    stretch, carried = found
    entry = entries[stretch]
    rotations, rows = _keyed_rows(group, table, carried)
    if rotations != tuple(sorted(tuple(int(v) for v in R) for R in entry["rotations"])):
        return blank
    # the entry's characters are on its rotations in their listed order
    order = [rotations.index(tuple(int(v) for v in R)) for R in entry["rotations"]]
    named = {tuple(int(chi[i]) for i in np.argsort(order)): name
             for chi, name in entry["names"]}
    return tuple((named.get(chi) or "") if chi is not None else "" for chi in rows)


def _holds(group, letter) -> bool:
    """Whether the shipped table of names has an entry for a position.

    The position is ``letter`` of the space group of ``group``.
    """
    document = _document()
    if document is None or not letter:
        return False
    return bool(document.get("groups", {}).get(str(int(group._sg_number)), {}).get(letter))


def _name_by_invariants(group, assignment, site, table, row, *, context=None) -> tuple:
    r"""The site irreps with the invariants of one row's band representation.

    ``row`` indexes ``table``, the site table of ``site``, a site of
    ``assignment``, the orbitals' assignment to ``group``.  The row, times the
    spin-1/2 lift on a spinful group, is induced to every high-symmetry point
    the tables list.  Its multiplicities are decomposed into elementary band
    representations at every listed position on the closure of the site's,
    and the stable real-space invariants of each decomposition are read.  The
    integer-valued ones are also read off the rational solution through the
    Smith form.  They are compared with the invariants of each single site
    irrep the tables of the invariants hold at the site's letter with the
    row's number of bands.  Where no decomposition is found, only the
    integer-valued invariants are compared.  ``context`` is a mapping kept
    across calls on one group and assignment, holding what they share.
    Returns the names that match, sorted, in the paper's spelling: one name is
    the row's, two or more a tie, none a row no single site irrep matches.
    The shipped table of names is made with it, and no measurement of a model
    calls it.

    Raises
    ------
    ..refusal.InternalError
        The tables of the invariants, the list of elementary band
        representations, or the sites or points in the tables' setting are not
        read for the group, or the decompositions and the Smith form disagree
        on the invariants.
    """
    from fractions import Fraction

    from . import _naming, _smith, _tables, srsi
    from ._srsitables import _rational_p, _split, srsi_tables
    from .refusal import InternalError, Refusal

    ctx = {} if context is None else context
    if "tables" not in ctx:
        ctx["tables"] = srsi_tables(group)
        ctx["data"] = _tables._ebr_data(group)
        ctx["sites"] = _tables._sites_in_tables_setting(group, assignment.site_positions)
        ctx["points"] = _tables._points_in_table_order(group)
        ctx["plans"] = {}
    tables, data, sites, points = ctx["tables"], ctx["data"], ctx["sites"], ctx["points"]
    for part in (tables, data, sites, points):
        if isinstance(part, Refusal):
            raise InternalError(
                f"{group.name} #{group.number_str}: a table a site irrep is named from "
                f"is refused, {part.gate}: {part.reason}")
    here = sites[int(site)]
    letter = here.letter
    lift = None
    if group.spinor:
        lift = {int(s.ind): complex(np.trace(np.asarray(s.spinor_rotation)))
                for s in group._symmetries}
    chi = {int(i): table.character_on(row, int(i)) * (1.0 if lift is None else lift[int(i)])
           for i in table.indices}
    counts, read = {}, []
    for name in points:
        key = (int(site), tuple(int(i) for i in table.indices), name)
        if key not in ctx["plans"]:
            ctx["plans"][key] = _naming._induction_plan(
                group, assignment, int(site), tuple(table.indices),
                k=None if name == "GM" else points[name], kpname=name)
        plan = ctx["plans"][key]
        content = ({} if plan is None
                   else _naming._induced_content(group, assignment, int(site), chi, plan=plan))
        if not content:
            continue
        read.append(name)
        for label, m in content.items():
            counts[str(label)] = counts.get(str(label), 0) + int(m)
    dim = int(round(float(np.real(table.dims[row]))))
    orbit = len({int(p) for p in np.asarray(assignment.permutations)[:, int(site)]})
    bands = dim * orbit * (2 if group.spinor else 1)

    found = set()
    for closure_letter in tuple(here.closure):
        _status, solutions, columns, _keep = srsi._solve_at_letter(
            data, counts, tuple(read), closure_letter, bands, bands=bands)
        for solution in solutions:
            vector = [0] * len(data["ebrs"])
            for j, m in zip(columns, solution):
                vector[j] = int(m)
            z, zn = _split(tables, tables._theta_columns(vector))
            found.add((tuple(z), tuple(sorted((n, tuple(v)) for n, v in zn.items()))))
    labels = list(data["labels_physics_convention"])
    y = [Fraction(int(counts.get(label, 0))) for label in labels]
    u, v, d, _matrix = _smith._smith_system(data, str(group.number_str))
    z_rational = tuple(tables._theta_columns(_rational_p(u, v, d, y)).theta_Z.values())
    if len(found) > 1 or any(z != z_rational for z, _zn in found):
        raise InternalError(
            f"{group.name} #{group.number_str}: the band representation a site irrep at "
            f"{letter} induces has the invariants {sorted(found)} by its decompositions, "
            f"and the integer-valued invariants {z_rational} by the Smith form. Report it.")
    target = next(iter(found)) if found else None
    form = tables._band_count_form
    out = []
    # the names in the paper's spelling, which the table of names holds
    for name in sorted({k._paper for k in tables.site_irreps if k._letter == letter}):
        z, zn = _split(tables, tables._theta_at({name: 1}, letter))
        if int(sum(Fraction(a) * b for a, b in zip(form, z))) != bands:
            continue
        theta = (tuple(z), tuple(sorted((n, tuple(w)) for n, w in zn.items())))
        if (theta == target) if target is not None else (theta[0] == z_rational):
            out.append(name)
    return tuple(out)
