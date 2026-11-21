r"""Site irreps named by the tables of the stable real-space invariants.

At a Wyckoff position the list of elementary band representations names no
band representation at, a row of a site's character table is named by the site
irrep the tables of the stable real-space invariants (Hwang et al., Nat.
Commun. 17, 8564 (2026)) give the same invariants: the row and that site irrep
induce band representations that can be deformed into one another.  The names
are worked out once, by inducing each row and matching its invariants, and
shipped in ``data/site_irrep_names.json``; reading them induces nothing.

A name is read at a site by carrying the site to the position's own site, the
point the tables list first, in the tables' conventional cell.  The points of a
position with free parameters form a line or a plane, cut into stretches by the
more special positions on it; a site is carried, by an operation of the group,
onto the stretch holding a reference point without crossing a more special
position, and its operations are carried with it.  The characters of a row on
the carried operations are what the shipped table is keyed by, so every site of
an orbit reads one name, whichever site a model lists first, and a group with
time reversal and the same space group without it read the same names.
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
    """The shipped table of names, read once; ``None`` where it is missing or unreadable."""
    try:
        with open(_PATH, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


@lru_cache(maxsize=None)
def _operations(sg: int) -> tuple:
    """``((R, t), ...)``: the space group's operations in the tables' conventional cell.

    The tables' own list of the space group's operations, each rotation an
    integer matrix and each translation snapped to a small-denominator
    rational, times every centering vector of ``pyxtal``'s list in the tables'
    setting.  A property of the space group alone, read alike for a group with
    time reversal and without.
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
    """``(A0, b0, P0)`` of the first arm of a position in the tables' setting, or ``None``."""
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
    """Whether the segment from ``z`` to ``p0`` meets no point fixed by more than its own fixers.

    For each operation :math:`(R, t)` the points of the segment it fixes are
    those where :math:`(R - 1) z(s) + t` is an integer vector; an operation that
    fixes the whole segment is one of the variety's own fixers.
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
    """The index in ``ops`` of the first operation carrying ``x`` onto ``p0``'s stretch, or ``None``.

    The image :math:`R x + t + n`, ``n`` a lattice vector, has to lie within
    ``tol`` of the arm's variety; it is placed on the variety, and the segment
    from it to ``p0`` must cross no more special position.
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
    """The parameters, among :data:`_CANDIDATES`, of each stretch's reference point, in order.

    A candidate is kept when it places a point of the position itself, fixed
    by as many operations as the generic point is, that no operation carries
    onto a stretch already kept.
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
    """The reference point of each stretch of a position, in the tables' conventional cell.

    Stretch ``0``'s is the position's own site.
    """
    arm = _first_arm(int(sg), str(letter))
    if arm is None:
        return ()
    A0, b0, _P0 = arm
    return tuple(tuple(float(v) for v in A0 @ np.asarray(q) + b0)
                 for q in _stretch_parameters(int(sg), str(letter)))


@lru_cache(maxsize=4096)
def _carry(sg: int, letter: str, x_ref: tuple):
    """``(stretch, k)``: the stretch a point of the tables' cell lies in, and the carrying operation."""
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
    r"""``(stretch, carried)``: a site carried to its position's own site in the tables' cell.

    ``group`` is the group a site was measured in, ``letter`` the site's
    Wyckoff letter in the tables' setting, ``position`` the site in reduced
    coordinates of the model's cell, and ``indices`` the ``.ind`` of operations
    fixing it, unitary and antiunitary.  Returned: the stretch of the position
    the site lies in, and ``carried[k]``, the ``.ind`` of the operation
    :math:`g h_k g^{-1}` fixing that stretch's reference point, :math:`g` the
    carrying operation.  An operation is found by its rotation and whether it
    is antiunitary, which name one operation of a site group.  ``None`` where
    the group has no transformation to the tables' cell, the letter is not one
    of the group's positions, or the site reaches no stretch.  Reads no table of
    names and needs no tables of the invariants.
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
    r"""``(rotations, rows)``: a site table's rows as the shipped table keys them.

    ``carried`` is aligned with ``table.indices``.  ``rotations`` are the
    carried unitary operations' rotations in the tables' cell,
    :math:`\mathrm{refUC}^{-1} R\,\mathrm{refUC}`, each as its nine integers,
    sorted; ``rows[r]`` is row ``r``'s characters on them in that order, as
    integers, or ``None`` for a row with a character that is not a real integer.
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
    """The shipped table's name of every row of one site's table, ``""`` where it has none.

    Read at the site by its own carrying to the position's own site, never at
    another site of its orbit.  Every name is ``""`` where the table of names
    is missing, holds no entry at the group's space group and the site's
    letter, the group has no transformation to the tables' cell, the site
    reaches no stretch, or its carried operations are not those the entry is
    written on; a row with a character that is not a real integer, a member of
    a pair time reversal does not join, has none, and so has a row two site
    irreps tie at.
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
    """Whether the shipped table has an entry at the group's space group and ``letter``."""
    document = _document()
    if document is None or not letter:
        return False
    return bool(document.get("groups", {}).get(str(int(group._sg_number)), {}).get(letter))


def _name_by_invariants(group, assignment, site, table, row, *, context=None) -> tuple:
    r"""The site irreps at a site's position with the invariants of one row's band representation.

    The row is induced to every high-symmetry point the tables list; the
    multiplicities found are decomposed into elementary band representations at
    every listed position on the closure of the site's, and each
    decomposition's stable real-space invariants are read; the integer-valued
    ones are also read by the rational route, from the Smith form, and the two
    must agree.  They are compared with the invariants of each single site
    irrep the tables of the invariants hold at the site's letter with the
    row's number of bands.  Returned: the names that match, sorted; one name
    is the row's, two or more a tie, none a row no single site irrep matches.
    ``context`` is a mapping kept across calls on one group and assignment.
    Never called while a model is measured; the table of names is made with it,
    and its test reads it.

    Raises
    ------
    ..refusal.InternalError
        The tables of the invariants or the list of elementary band
        representations are not read for the group, or the decompositions
        disagree on the invariants.
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
                f"#{group.number_str}: the tables a site irrep is named from are not "
                f"read ({part.gate}: {part.reason})")
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
            f"#{group.number_str}: the band representation a row of the site table at "
            f"{letter} induces has invariants {sorted(found)} by its decompositions and "
            f"integer-valued invariants {z_rational} by the Smith form; the readings of "
            "one band representation have to agree")
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
