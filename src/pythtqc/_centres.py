"""The orbital centers against the group's action on them: the sites as the
symmetry finder reads them, where the operations carry each site, how far the
centers miss, and the least move onto a set the operations carry exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math

import numpy as np

from . import _bloch
from .refusal import Refusal
from ._tolerance import ROUNDING, _EXACT_CLOSURE
from ._spacegroup import (
    _OWN_ORBITALS, _PAD_LENGTH, _TRANSLATION_DENOMINATOR, _pad_cell,
    _sites_and_typat)
from ._matrices import _orb_positions_3d


def _position_defect(model, group, assignment, ops):
    r"""How far the model's orbital centers miss the group's action on them.

    ``ops`` maps an operation's index in ``group`` to the operation, and
    ``assignment`` holds each operation's permutation of the sites and its
    lattice shifts.  Returns the largest residual of
    ``A·τ_a + t = τ_{g(a)} + s`` over ``ops`` and every orbital, each component
    taken modulo the lattice, in reduced coordinates of the group's 3D cell.
    """

    pos = _orb_positions_3d(model, group)
    site_of = np.asarray(assignment.site_of_orbital, dtype=int)
    perm = np.asarray(assignment.permutations, dtype=int)
    shift = np.asarray(assignment.lattice_shift, dtype=float)
    first = {}
    for a in range(int(model.norb)):
        first.setdefault(int(site_of[a]), a)
    pos_of_ind = {int(s.ind): i for i, s in enumerate(group._symmetries)}
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


def _centre_misses(model, group, assignment, ops):
    r"""Each orbital's miss of each operation, modulo the lattice.

    The miss is ``A·τ_a + t − s − τ_{g(a)}``, the identity
    :func:`_position_defect` measures, kept per operation and orbital.  It is
    evaluated in rational arithmetic on the stored numbers, so that centers at
    their symmetric positions miss by the rounding of their own storage alone.
    Returns a mapping from each operation's index in ``ops`` to an array of
    shape ``(norb, 3)``, in reduced coordinates of the group's 3D cell.
    """
    pos = _orb_positions_3d(model, group)
    exact_pos = [[Fraction(float(x)) for x in row] for row in pos]
    site_of = np.asarray(assignment.site_of_orbital, dtype=int)
    perm = np.asarray(assignment.permutations, dtype=int)
    shift = np.asarray(assignment.lattice_shift, dtype=float)
    first = {}
    for a in range(int(model.norb)):
        first.setdefault(int(site_of[a]), a)
    pos_of_ind = {int(s.ind): i for i, s in enumerate(group._symmetries)}
    out = {}
    for ind, sym in ops.items():
        p = pos_of_ind[int(ind)]
        A = [[Fraction(float(x)) for x in row] for row in np.asarray(sym.rotation, dtype=float)]
        t = [Fraction(float(x)) for x in np.asarray(sym.translation, dtype=float)]
        miss = np.zeros((int(model.norb), 3))
        for a in range(int(model.norb)):
            s = int(site_of[a])
            b = first[int(perm[p, s])]
            for r in range(3):
                x = (sum(A[r][c] * exact_pos[a][c] for c in range(3)) + t[r]
                     - Fraction(float(shift[p, s, r])) - exact_pos[b][r])
                miss[a, r] = float(x - round(x))
        out[int(ind)] = miss
    return out


def _miss_reach(model, group, assignment, ops) -> float:
    r"""The most the centers' miss of ``ops`` can move ``H``, in its units.

    For an operation ``g``, the misses ``e_a`` of :func:`_centre_misses` put
    a phase error ``2π k·W(e_j − e_i)`` on the hopping from ``i`` to ``j`` at
    the image point ``k W``.  For ``k`` in ``[-1, 1]`` along each periodic
    direction that error is at most ``2π |W(e_j − e_i)|``, the components'
    magnitudes summed.  The bound is the size of each hopping times that
    error, summed along a row of ``H``.  It bounds ``D H(k) D† − H(k W)`` for
    ``D`` the operation's matrix on the orbitals, the one exact for centers at
    their symmetric positions, whatever unitary it is within each site.  It
    holds only for points and images in ``[-1, 1]``, which holds every one
    the covariance and the projection read.  Returns the largest bound over
    ``ops``, ``0.0`` for a model with no hoppings.
    """
    bloch = _bloch.evaluator(model)
    if not bloch.n_hops:
        return 0.0
    M = bloch.nstate
    size = np.abs(bloch.blocks).reshape(bloch.n_R, M, M)
    whole = (size + size.swapaxes(-2, -1)).sum(axis=0)   # each hopping and its conjugate
    per = [int(d) for d in group.periodic_dirs]
    worst = 0.0
    for ind, miss in _centre_misses(model, group, assignment, ops).items():
        act = group._k_action_of(ops[ind])
        if act is None:
            continue
        W = act(np.eye(int(group._dim_k)))               # rows: k -> k W
        w = miss[:, per] @ W.T                           # W e_a, one row per orbital
        if bloch.nspin != 1:
            w = np.repeat(w, bloch.nspin, axis=0)
        phase = np.abs(w[None, :, :] - w[:, None, :]).sum(axis=-1)
        worst = max(worst, float(2.0 * np.pi * (whole * phase).sum(axis=-1).max()))
    return worst


def _centres_account_for(model, group, assignment, ops, *, failure, gate, check,
                         declared=False):
    r"""The refusal where the centers' miss accounts for a failed check.

    The refusal is ``positions_not_exactly_symmetric``.  ``failure`` is what
    a check of exact symmetry measured on ``model`` and ``gate`` what it holds
    that to, both in the units of ``H``.  ``check`` names the check in a few
    words for the reason.  ``declared`` says whether ``precision=`` or
    ``position_tol=`` was declared.  The miss accounts for the failure where
    its reach (:func:`_miss_reach`) exceeds the gate and the failure is at
    most the gate plus twice the reach.  Returns the refusal, the group named
    in its reason and carried in its context.  Returns ``None`` where a
    tolerance for the centers was declared, or where the miss does not
    account for the failure.
    """
    # The refusal is for centers no declaration speaks for, as the one made
    # before the detector is, and its reason asks for a precision; under a
    # declared one the centers were moved within it, and the check's own
    # verdict is returned.
    if declared:
        return None
    failure, gate = float(failure), float(gate)
    reach = _miss_reach(model, group, assignment, ops)
    if not (reach > gate and failure <= gate + 2.0 * reach):
        return None
    delta = float(_position_defect(model, group, assignment, ops))
    return Refusal(
        "positions_not_exactly_symmetric",
        f"the orbital centers miss the positions the group puts them at by {delta:.3g} "
        "(reduced): declare this model's precision, `realized_group(..., precision=)`, "
        "or `position_tol=` for the centers alone, and centers within it are moved onto "
        "those positions. Undeclared, the miss alone moves H(k) by up to "
        f"{reach:.3g}, more than the {gate:.3g} that {check} is held to, where "
        f"{failure:.3g} is measured. The group identified is {group.name} "
        f"#{group.number_str}, every operation of it obeyed.",
        value=delta, threshold=delta * gate / reach,
        context={"group": group, "number_str": str(group.number_str),
                 "position_defect": delta, "reach": reach, "gate": gate,
                 "failure": failure})


def _centres_account_for_closure(refusal):
    r"""The refusal where the centers' miss accounts for operations that do not close.

    ``refusal`` is a refusal of the cell's identification, made with no
    tolerance for the centers declared.  Only ``operations_not_consistent``
    whose context carries ``site_miss`` (:func:`.._spacegroup._site_miss`) and
    ``rotation_size`` is read.  The miss accounts for the closure where its
    reach, ``(2 + rotation_size) · site_miss``, exceeds the gate and the
    closure is at most the gate plus the reach.  Returns
    ``positions_not_exactly_symmetric``, naming the cell's group as an upper
    bound, with its name and number but no ``"group"`` in its context.
    Returns ``None`` for any other refusal, or where the miss does not account
    for the closure.
    """
    ctx = refusal.context
    if refusal.gate != "operations_not_consistent" or "site_miss" not in ctx:
        return None
    miss, closure, gate = float(ctx["site_miss"]), float(refusal.value), float(refusal.threshold)
    reach = (2.0 + float(ctx["rotation_size"])) * miss
    if not (reach > gate and closure <= gate + reach):
        return None
    return Refusal(
        "positions_not_exactly_symmetric",
        f"the orbital centers miss the operations of the cell's group by {miss:.3g} (reduced): "
        "declare this model's precision, `realized_group(..., precision=)`, or "
        "`position_tol=` for the centers alone, and centers within it are moved onto the "
        "positions the group puts them at. Undeclared, the operations read off these centers "
        f"close only to {closure:.3g}, more than the {gate:.3g} a cell built from exact "
        f"numbers closes to, and a miss of {miss:.3g} can leave them open by up to "
        f"{reach:.3g}, so there is no exactly symmetric configuration to measure H against. "
        f"Nothing was built. The cell's group is {ctx['name']} #{ctx['number_str']}, an upper "
        "bound on what this Hamiltonian obeys.",
        value=miss, threshold=miss * gate / reach,
        context={"number_str": str(ctx["number_str"]), "name": str(ctx["name"]),
                 "position_defect": miss, "reach": reach, "gate": gate,
                 "failure": closure})


# --------------------------------------------------------------------------
# the centres against the group's action: the sites, the moves and the defect
# --------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class _SiteFrame:
    """A cell's sites, set in 3D, and which of them may move.

    The sites are grouped as the symmetry finder groups them.  Coordinates
    are reduced coordinates of the 3D cell unless said otherwise.

    Attributes
    ----------
    lattice : ndarray, shape (3, 3)
        The 3D cell's lattice vectors, Cartesian, in the model's unit of length.
    vacuum_dirs : tuple of int
        The directions along which the cell is set in vacuum (:func:`_pad_cell`).
    points : ndarray, shape (n, 3)
        Every point of the cell.
    first : ndarray, shape (n_sites, 3)
        Each site's first point.
    site_orbs : tuple of tuple of int
        The points of each site.
    species : ndarray or None
        Each site's species; ``None`` in a frame from :func:`_assigned_sites`,
        which :func:`_site_action` cannot read.
    sites : ndarray, shape (n_sites, 3)
        Each site's position: its points' mean, each read on the branch of the
        site's first point.
    weights : ndarray
        How many points each site holds.
    scale : ndarray, shape (3,)
        How a coordinate of the 3D cell maps onto the cell's own, per direction.
    own : int
        How many points are the model's orbitals, which lead the cell.
    moving : ndarray of int
        The sites that are not shadows, the ones a projection moves.
    """

    lattice: object
    vacuum_dirs: tuple
    points: object
    first: object
    site_orbs: tuple
    species: object
    sites: object
    weights: object
    scale: object
    own: int
    moving: object


def _cell_sites(cell, cell_typat, symprec, *, n_extra=0) -> _SiteFrame:
    r"""The cell's sites as the symmetry finder reads them, and which may move.

    ``cell_typat`` gives each point's species.  The cell is set in 3D and its
    points grouped into sites at the distance ``symprec``, by
    :func:`_sites_and_typat`, as :func:`.._spacegroup._group_of` groups them,
    so that the operations found on these sites are the ones it finds.  The
    shadow species a substrate puts under the sites
    (:func:`.._layers._decorated_cell`) are read with the rest and do not
    move.  The last ``n_extra`` points of ``cell`` are declared decorations,
    which move with the centers.  Returns the :class:`_SiteFrame`.
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
                 moving=None) -> _SiteFrame:
    """The record :func:`_cell_sites` returns, for points set in 3D and grouped.

    The arguments are the fields of :class:`_SiteFrame` of the same names,
    with ``vacuum_dirs`` passed fourth; ``model`` gives the lattice the
    ``scale`` is read against.  ``own`` defaults to every point, ``moving``
    to every site.
    """
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
    return _SiteFrame(lattice=np.asarray(lattice, dtype=float), vacuum_dirs=tuple(padded),
                      points=points, first=first,
                      site_orbs=tuple(tuple(int(o) for o in orbs) for orbs in site_orbs),
                      species=None if species is None else np.asarray(species),
                      sites=np.array(sites, dtype=float),
                      weights=np.array([float(len(orbs)) for orbs in site_orbs]),
                      scale=scale, own=int(points.shape[0] if own is None else own),
                      moving=np.array(range(len(site_orbs)) if moving is None else moving,
                                      dtype=int))


def _assigned_sites(model, group, assignment) -> _SiteFrame:
    """A model's sites as its assignment to ``group`` groups them.

    The record :func:`_cell_sites` returns, built from the model alone: set in
    the group's lattice (:func:`_pad_cell`), every point an orbital of the
    model and every site one the projection moves; its ``species`` is
    ``None``.  The assignment's permutations and lattice shifts are then the
    action :func:`_symmetric_sites` reads.
    """
    lattice, points, padded = _pad_cell(model, group._vacuum_length, lattice=group._lattice)
    return _sites_frame(model, lattice, points, padded, assignment.site_orbitals)


def _site_action(frame, rotations, translations):
    r"""Where each operation sends each moving site, and by which lattice vector.

    Each image ``R·x + t`` is read against the sites of the same species, the
    nearest by Cartesian distance modulo the lattice, as
    :func:`.._tables._assign_orbitals` reads it, once, on the positions as they
    are.  ``frame.species`` must not be ``None``.  Returns ``(perm, shift)``:
    ``perm[g, s]`` the image of moving site ``s`` among the moving sites, and
    ``shift[g, s]`` the lattice vector with ``R·x_s + t ≈ x_perm + shift``.
    Returns ``None`` where an operation does not permute them.
    """
    lattice = frame.lattice
    move = frame.moving
    sites = frame.sites[move]
    species = frame.species[move]
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
    r"""A subset of the operations ``rows`` whose constraints imply all of theirs.

    ``rotations`` is indexed by the operations in ``rows``.  Returns the
    indices of every pure translation among ``rows``, and of as many others as
    it takes for their rotations to generate every rotation there.
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
    r"""The moving sites moved onto the nearest set the operations ``rows`` carry.

    ``perm`` and ``shift`` are each operation's permutation ``π_g`` of the
    moving sites ``x_s`` and its lattice vectors ``n_g``
    (:func:`_site_action`).
    The constraints are written relative to the first site, so no translation
    enters them.  The move is the least in Cartesian length, weighed by each
    site's points, and has no part common to every site.  Returns
    ``(sites, translations, closure)``: the moved sites in the 3D cell's frame;
    each operation's translation read off them,
    ``t_g = x_π(0) + n_g(0) − R_g x_0``; and the largest miss of the moved
    sites under the operations of ``rows`` with those translations, at rounding
    wherever the operations act on the sites as a group does.
    """
    lattice = frame.lattice
    move = frame.moving
    y = frame.sites[move]
    w = frame.weights[move]
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
    r"""The ``δ`` such that the sites at ``x − δ`` have rational translations.

    ``trans`` holds each operation's translation ``t_g`` on the exact sites
    (:func:`_symmetric_sites`).  Moving every site by ``−δ`` moves ``t_g`` by
    ``−(1 − R_g)·δ``.  So ``(1 − R_g)·δ = t_g − round(t_g)`` is solved by least
    squares, for every operation of ``rows`` together, over the periodic
    directions.  ``round`` goes to the nearest multiple of ``1/d``.  ``d`` is
    the least common multiple of :data:`_TRANSLATION_DENOMINATOR` and the
    denominators of the pure translations among ``rows``.  Returns ``δ`` where
    the solution is exact to ``_EXACT_CLOSURE``, and ``None`` where it is not.
    Components at rounding are set to zero, and so is every direction that
    every rotation fixes.
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
    r"""Each point's move onto its site at ``x``.

    The moves are in the cell's own reduced coordinates.  ``x`` holds the
    moved position of each moving site.  A point moves from where it is to
    the position of its site, modulo the lattice.  Where every move is at
    rounding, none is made.  Otherwise a move at the rounding of the largest
    is set to zero, and every other move is made, however small.  The points
    of sites that do not move stay at zero.
    """
    pts = frame.points
    out = np.zeros_like(pts)
    for k, s in enumerate(frame.moving):
        for o in frame.site_orbs[int(s)]:
            d = x[k] - pts[int(o)]
            out[int(o)] = (d - np.rint(d)) * frame.scale
    largest = float(np.abs(out).max(initial=0.0))
    out[np.abs(out) <= (ROUNDING if largest <= ROUNDING else ROUNDING * largest)] = 0.0
    return out


def _frame_defect(frame, perm, rotations, trans, rows, sites=None) -> float:
    r"""How far the centers miss the operations ``rows`` with translations ``trans``.

    ``max |R·τ_a + t − τ_b|`` modulo the lattice, over the operations and the
    model's own orbitals, ``τ_b`` the first orbital of the site the operation
    sends ``τ_a``'s to, in the 3D cell's frame: the position defect
    (:func:`_position_defect`), read on the centers as they are, or with every
    orbital on its site at ``sites`` where that is given.
    """
    pts = frame.points
    move = [int(s) for s in frame.moving]
    if sites is not None:
        pts = pts.copy()
        for k, s in enumerate(move):
            for o in frame.site_orbs[s]:
                pts[int(o)] = sites[k]
    own = frame.own
    worst = 0.0
    for g in rows:
        R = np.asarray(rotations[g], dtype=float)
        for k, s in enumerate(move):
            target = int(frame.site_orbs[move[int(perm[g, k])]][0])
            for o in frame.site_orbs[s]:
                if int(o) >= own:
                    continue
                r = R @ pts[int(o)] + trans[g] - pts[target]
                worst = max(worst, float(np.abs(r - np.rint(r)).max()))
    return worst


def _geometry(model, orb):
    """A model's lattice with the centers ``orb``, carrying no Hamiltonian."""
    from pythtb import Lattice, TBModel

    return TBModel(Lattice(np.asarray(model.lat_vecs, dtype=float),
                           np.asarray(orb, dtype=float).tolist(),
                           periodic_dirs=list(model.periodic_dirs)), spinful=False)


def _moved_extras(extras, moves, start, dim_r) -> tuple:
    """The decorations moved with the centers.

    ``extras`` holds each decoration's position and label, ``moves`` each
    point's move (:func:`_moves_of`), and point ``start + k`` of the cell is
    extra ``k``; the first ``dim_r`` components of its move are applied.
    Returns ``extras`` in the same form, at the moved positions.
    """
    return tuple((tuple(float(c) for c in (np.asarray(pos, dtype=float).reshape(-1)
                                           + moves[int(start) + k, :dim_r])), lab)
                 for k, (pos, lab) in enumerate(extras))
