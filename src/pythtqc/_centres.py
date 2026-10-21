"""The orbital centres against the group's action on them: the sites as the
symmetry finder reads them, where the operations carry each site, how far the
centres miss, and the least move onto a set the operations carry exactly.
"""

from __future__ import annotations

import math

import numpy as np

from ._tolerance import ROUNDING, _EXACT_CLOSURE
from ._spacegroup import (
    _OWN_ORBITALS, _PAD_LENGTH, _TRANSLATION_DENOMINATOR, _pad_cell,
    _sites_and_typat)
from ._matrices import _orb_positions_3d


def _position_defect(model, group, assignment, ops):
    r"""How far this model's own positions are from the group's action on them.

    :func:`..symmetry.bloch_operator` *checks* the integer-shift identity
    ``A·τ_a + t = τ_{g(a)} + s`` rather than assuming it, at a tolerance whose
    default (``1e-9``) is a statement about a model built from exact numbers. A
    **wannier90** model's centres are numerically determined and satisfy it only
    to the Wannierization's own precision — measured **7.7e-9** on a real
    import.  This measures how far they miss it: the largest residual of the
    identity over the operations ``ops`` and every orbital, each component
    modulo the lattice.
    """

    pos = _orb_positions_3d(model, group)
    site_of = np.asarray(assignment.site_of_orbital, dtype=int)
    perm = np.asarray(assignment.permutations, dtype=int)
    shift = np.asarray(assignment.lattice_shift, dtype=float)
    first = {}
    for a in range(int(model.norb)):
        first.setdefault(int(site_of[a]), a)
    pos_of_ind = {int(s.ind): i for i, s in enumerate(group.symmetries)}
    worst = 0.0
    for ind, sym in ops.items():
        p = pos_of_ind[int(ind)]
        A = np.asarray(sym.rotation, dtype=float)
        t = np.asarray(sym.translation, dtype=float)
        for a in range(int(model.norb)):
            s = int(site_of[a])
            b = first[int(perm[p, s])]
            resid = A @ pos[a] + t - shift[p, s] - pos[b]
            worst = max(worst, float(np.abs(resid - np.rint(resid)).max()))
    return worst


# --------------------------------------------------------------------------
# the centres against the group's action: the sites, the moves and the defect
# --------------------------------------------------------------------------

def _cell_sites(cell, cell_typat, symprec, *, n_extra=0) -> dict:
    r"""The cell's sites as the symmetry finder reads them, and which of them may move.

    The cell is set in 3D and its points grouped into sites at the finder's
    distance exactly as :func:`.._spacegroup._group_of` groups them, so the operations found
    on these sites are the ones it finds.  For the centres' projection a site
    sits at its points' position -- their mean, each read on the branch of the
    site's first point, where they differ -- and weighs as many points as it
    holds, since moving a site moves each of them.  The shadow species a
    substrate puts under the sites (:func:`.._layers._decorated_cell`) are no centres:
    they are read with the rest, so that they remove the operations they
    remove, and they are built again under the moved sites.

    Returns ``lattice`` and ``vacuum_dirs`` (:func:`_pad_cell`), ``points`` (every
    point of the cell in the 3D cell's frame), ``first``, ``site_orbs`` and
    ``species`` (:func:`_sites_and_typat`), ``sites`` and ``weights`` (each
    site's position and count), ``scale`` (how a coordinate of the 3D cell
    maps onto the cell's own, per direction), ``own`` (how many points are the model's
    orbitals, which lead the cell) and ``moving`` (the sites that are not
    shadows, the ones the projection moves).
    """
    lattice, points, padded = _pad_cell(cell, _PAD_LENGTH)
    first, site_orbs, species = _sites_and_typat(points, cell_typat, symprec, lattice)
    norb = int(cell.norb)
    own = int(getattr(cell, _OWN_ORBITALS, norb))
    shadows = set(range(own, norb - int(n_extra)))
    moving = [s for s, orbs in enumerate(site_orbs)
              if any(int(o) not in shadows for o in orbs)]
    return _sites_frame(cell, lattice, points, padded, site_orbs, species=species,
                        own=own, moving=moving)


def _sites_frame(model, lattice, points, padded, site_orbs, *, species=None, own=None,
                 moving=None) -> dict:
    """The record :func:`_cell_sites` returns, for points already set in 3D and grouped."""
    points = np.asarray(points, dtype=float)
    first = np.array([points[int(orbs[0])] for orbs in site_orbs])
    sites = []
    for s, orbs in enumerate(site_orbs):
        on = np.array([points[int(o)] - np.rint(points[int(o)] - first[s]) for o in orbs])
        sites.append(first[s] if np.array_equal(on, np.repeat(on[:1], len(on), axis=0))
                     else on.mean(axis=0))
    lat_model = np.asarray(model.lat_vecs, dtype=float)
    scale = np.ones(3)
    for d in range(lat_model.shape[0]):
        scale[d] = float(np.linalg.norm(lattice[d]) / np.linalg.norm(lat_model[d]))
    return {"lattice": np.asarray(lattice, dtype=float), "vacuum_dirs": tuple(padded),
            "points": points, "first": first,
            "site_orbs": tuple(tuple(int(o) for o in orbs) for orbs in site_orbs),
            "species": None if species is None else np.asarray(species),
            "sites": np.array(sites, dtype=float),
            "weights": np.array([float(len(orbs)) for orbs in site_orbs]),
            "scale": scale, "own": int(points.shape[0] if own is None else own),
            "moving": np.array(range(len(site_orbs)) if moving is None else moving,
                               dtype=int)}


def _assigned_sites(model, group, assignment) -> dict:
    """A model's sites as its assignment to ``group`` groups them, read against the group.

    The record :func:`_cell_sites` returns, built from the model alone -- set
    in the group's lattice (:func:`_pad_cell`), every orbital its own and every
    site one the projection moves -- so that the assignment's permutations and
    lattice shifts are the action :func:`_symmetric_sites` reads.
    """
    lattice, points, padded = _pad_cell(model, group.vacuum_length, lattice=group.lattice)
    return _sites_frame(model, lattice, points, padded, assignment.site_orbitals)


def _site_action(frame, rotations, translations):
    r"""Which moving site each operation sends each one to, and by which lattice vector.

    Each image ``R·x + t`` is read against the sites of the same species, the
    nearest by Cartesian distance modulo the lattice, as :func:`.._tables._assign_orbitals`
    reads it -- once, on the positions as they are.  Returns ``(perm, shift)``,
    ``perm[g, s]`` the image of moving site ``s`` among the moving sites and
    ``shift[g, s]`` the lattice vector with ``R·x_s + t ≈ x_perm + shift``, or
    ``None`` where an operation does not permute them.
    """
    lattice = frame["lattice"]
    move = frame["moving"]
    sites = frame["sites"][move]
    species = frame["species"][move]
    same = species[:, None] == species[None, :]
    n = len(sites)
    perm = np.empty((len(rotations), n), dtype=int)
    shift = np.empty((len(rotations), n, 3))
    for g, (R, t) in enumerate(zip(rotations, translations)):
        images = sites @ np.asarray(R, dtype=float).T + np.asarray(t, dtype=float).reshape(-1)
        d = images[:, None, :] - sites[None, :, :]
        whole = np.rint(d)
        dist = np.where(same, np.linalg.norm((d - whole) @ lattice, axis=2), np.inf)
        j = np.argmin(dist, axis=1)
        if len(np.unique(j)) != n:
            return None
        perm[g] = j
        shift[g] = whole[np.arange(n), j]
    return perm, shift


def _generating(rotations, rows) -> list:
    r"""Operations among ``rows`` that carry every constraint all of them carry.

    Every pure translation among them, and as many others as it takes for their
    rotations to generate every rotation there: two operations of one group with
    one rotation differ by a pure translation, so a set of points these carry is
    carried by all of ``rows``.
    """
    ident = np.eye(3, dtype=int)
    rot = {int(g): np.rint(np.asarray(rotations[g], dtype=float)).astype(int) for g in rows}
    out = [int(g) for g in rows if np.array_equal(rot[int(g)], ident)]
    reached = {ident.tobytes()}
    for g in rows:
        if rot[int(g)].tobytes() in reached:
            continue
        out.append(int(g))
        gens = [rot[h] for h in out if not np.array_equal(rot[h], ident)]
        mats, reached = [ident], {ident.tobytes()}
        k = 0
        while k < len(mats):
            for h in gens:
                m = mats[k] @ h
                if m.tobytes() not in reached:
                    reached.add(m.tobytes())
                    mats.append(m)
            k += 1
    return out


def _symmetric_sites(frame, perm, shift, rotations, rows):
    r"""The moving sites carried, by the least move, onto a set the operations ``rows`` carry exactly.

    Each operation ``g`` sends site ``s`` to ``π_g(s)`` by the lattice vector
    ``n_g(s)`` (:func:`_site_action`), so the sites it carries exactly solve
    ``R_g (x_s − x_0) − (x_π(s) − x_π(0)) = n_g(s) − n_g(0)``: the translation
    eliminated through the first site, so that none is read off the positions
    and no table is needed.  The solutions are an affine space, and the sites
    are moved onto its nearest point, the move measured in Cartesian length and
    weighed by each site's points: one linear projection, on the constraints
    of operations that carry all of the others' (:func:`_generating`).  A
    shift of every site together carries the constraints into themselves, so
    the move has no common part: the origin is fitted to the centres, a shift
    of all of them being no displacement.

    Returns ``(sites, translations, closure)``: the moved sites in the 3D
    cell's frame; each operation's translation on them, ``t_g = x_π(0) + n_g(0) − R_g
    x_0``; and how far the moved sites miss being carried by every operation of
    ``rows`` with those translations -- rounding wherever the operations act on
    the sites as a group does.
    """
    lattice = frame["lattice"]
    move = frame["moving"]
    y = frame["sites"][move]
    w = frame["weights"][move]
    n = len(y)
    rows = [int(g) for g in rows]
    inv = np.linalg.inv(lattice)
    x = y.copy()
    gens = _generating(rotations, rows)
    if n > 1 and gens:
        lt = inv.T                                   # a Cartesian move, read as a reduced one
        i, s = np.arange(n - 1), np.arange(1, n)
        blocks, rhs = [], []
        for g in gens:
            R = np.asarray(rotations[g], dtype=float)
            rl = R @ lt
            block = np.zeros((n - 1, 3, n, 3))
            block[i, :, s, :] += rl
            block[i, :, 0, :] -= rl
            block[i, :, perm[g, 1:], :] -= lt
            block[i, :, perm[g, 0], :] += lt
            blocks.append(block.reshape(3 * (n - 1), 3 * n))
            rhs.append(((shift[g, 1:] - shift[g, 0])
                        - ((y[1:] - y[0]) @ R.T - (y[perm[g, 1:]] - y[perm[g, 0]])))
                       .reshape(-1))
        A = np.vstack(blocks)
        b = np.concatenate(rhs)
        if float(np.abs(b).max(initial=0.0)) > 0.0:
            root = np.repeat(np.sqrt(w), 3)
            u = np.linalg.lstsq(A / root, b, rcond=None)[0]
            x = y + (u / root).reshape(n, 3) @ inv
    trans = np.zeros((len(rotations), 3))
    closure = 0.0
    for g in rows:
        R = np.asarray(rotations[g], dtype=float)
        trans[g] = x[perm[g, 0]] + shift[g, 0] - R @ x[0]
        miss = x @ R.T + trans[g] - x[perm[g]] - shift[g]
        closure = max(closure, float(np.abs(miss).max(initial=0.0)))
    return x, trans, closure


def _rational_offset(trans, rotations, rows, periodic):
    r"""The shift of every site after which each translation is a rational of the cell.

    Exact sites fix each operation's translation up to the origin it is written
    about: moved together by ``−δ``, the translation ``t_g`` moves by ``−(1 −
    R_g)·δ``.  About the origin the model's own numbers are written in, every
    translation is a multiple of ``1/24`` (:data:`_TRANSLATION_DENOMINATOR`,
    with a repeating cell's own pure translations' denominators beside it), so
    ``(1 − R_g)·δ = t_g − round(t_g)`` is solved for every operation of ``rows``
    together, across the periodic directions; a direction every rotation fixes
    is left where it is.  Returns ``δ`` where that solution is exact to
    rounding, and ``None`` where the translations are on no such grid about any
    point: a crystal written about an origin the operations do not single out.
    """
    from fractions import Fraction

    per = [int(d) for d in periodic]
    rows = [int(g) for g in rows]
    if not per or not rows:
        return np.zeros(3)
    den = _TRANSLATION_DENOMINATOR
    for g in rows:
        if np.array_equal(np.rint(np.asarray(rotations[g], dtype=float)), np.eye(3)):
            for t in trans[g][per]:
                d = Fraction(float(t) % 1.0).limit_denominator(64).denominator
                den = den * d // math.gcd(den, d)
    mat = np.vstack([(np.eye(3) - np.asarray(rotations[g], dtype=float))[np.ix_(per, per)]
                     for g in rows])
    t = np.array([np.asarray(trans[g], dtype=float)[per] for g in rows])
    rhs = (t - np.rint(t * den) / den).reshape(-1)
    sol = np.linalg.lstsq(mat, rhs, rcond=None)[0]
    if float(np.abs(mat @ sol - rhs).max(initial=0.0)) > _EXACT_CLOSURE:
        return None
    delta = np.zeros(3)
    delta[per] = sol
    # rounding is no shift: centres written in the model's own frame stay
    # exactly where they are
    delta[np.abs(delta) <= ROUNDING] = 0.0
    return delta


def _moves_of(frame, x) -> np.ndarray:
    r"""Each point's move onto its site at ``x``, in the cell's own reduced coordinates.

    A point moves from where it is to the position of the site it is on,
    modulo the lattice: two orbitals of one site written on the two sides of a
    cell edge are one site, and a whole lattice vector is no move.  A move at
    rounding is none, and the shadows the projection does not move stay at
    zero.
    """
    pts = frame["points"]
    out = np.zeros_like(pts)
    for k, s in enumerate(frame["moving"]):
        for o in frame["site_orbs"][int(s)]:
            d = x[k] - pts[int(o)]
            out[int(o)] = (d - np.rint(d)) * frame["scale"]
    out[np.abs(out) <= ROUNDING] = 0.0
    return out


def _frame_defect(frame, perm, rotations, trans, rows, sites=None) -> float:
    r"""How far the centres miss the operations ``rows``, with the translations ``trans``.

    ``max |R·τ_a + t − τ_b|`` modulo the lattice, over the operations and the
    model's own orbitals, ``τ_b`` the first orbital of the site the operation
    sends ``τ_a``'s to, in the 3D cell's frame: the position defect
    (:func:`.._centres._position_defect`), read on the centres as they
    are, or with every orbital on its site at ``sites`` where that is given.
    """
    pts = frame["points"]
    move = [int(s) for s in frame["moving"]]
    if sites is not None:
        pts = pts.copy()
        for k, s in enumerate(move):
            for o in frame["site_orbs"][s]:
                pts[int(o)] = sites[k]
    own = frame["own"]
    worst = 0.0
    for g in rows:
        R = np.asarray(rotations[g], dtype=float)
        for k, s in enumerate(move):
            target = int(frame["site_orbs"][move[int(perm[g, k])]][0])
            for o in frame["site_orbs"][s]:
                if int(o) >= own:
                    continue
                r = R @ pts[int(o)] + trans[g] - pts[target]
                worst = max(worst, float(np.abs(r - np.rint(r)).max()))
    return worst


def _geometry(model, orb):
    """A model's lattice with the centres ``orb``: geometry, carrying no Hamiltonian."""
    from pythtb import Lattice, TBModel

    return TBModel(Lattice(np.asarray(model.lat_vecs, dtype=float),
                           np.asarray(orb, dtype=float).tolist(),
                           periodic_dirs=list(model.periodic_dirs)), spinful=False)


def _moved_extras(extras, moves, start, dim_r) -> tuple:
    """The decorations moved with the centres: point ``start + k`` of the cell is extra ``k``."""
    return tuple((tuple(float(c) for c in (np.asarray(pos, dtype=float).reshape(-1)
                                           + moves[int(start) + k, :dim_r])), lab)
                 for k, (pos, lab) in enumerate(extras))
