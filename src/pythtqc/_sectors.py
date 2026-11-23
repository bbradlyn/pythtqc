r"""The sectors a Hamiltonian's conserved matrices split its states into.

The matrices that commute with ``H(k)`` at every ``k`` and act within sites
form an algebra, and the algebra is a sum of sectors (:class:`Sector`).  A
sector holds ``m`` identical copies of a set of ``d`` states, and on it the
algebra is every ``m × m`` matrix on the copies times the identity on each
copy's states.  On states that carry a spin, native or declared in pairs, the
algebra is read on the orbital factor ``O`` of its members ``O ⊗ 1``, whose
indices are the orbitals, or the declared pairs.  On states with no spin the
orbital factor is the states themselves.  Every ``tol`` here but
:class:`SectorSearch`'s is the declared tolerance relative to ``max|H|``,
``None`` where nothing is declared.  A threshold reads it where it is larger
than its exact default (:func:`.._tolerance.noise_floor`).

The sectors are read once, where a model enters (:func:`read_sectors`), and
then what each operation does to them (:func:`sector_action`).  On each orbit
of sectors under a subgroup, the representations the hoppings leave open are
induced from the stabilizer of one sector.  :class:`SectorSearch` weighs them
against a declaration on any subgroup, :func:`induced_orbits` builds them, and
:func:`tables_gauge` writes them in the tables' matrices.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools

import numpy as np

from . import _detector
from . import _matrices
from . import _naming
from ._tolerance import (
    ASSEMBLED, CHARACTER, CLUSTER, ZERO, _FACTOR_TOL, _IN_BLOCKS, _LATTICE_TOL,
    noise_floor)

__all__ = ["Sector", "SectorReading", "SectorAction", "read_sectors",
           "carried_reading", "joined", "sector_action", "SectorSearch",
           "atom_sites", "chan_rows", "spin_places", "in_states", "held_shares",
           "aligned", "InducedOrbit", "SharedOrbits", "held_sectors", "held_on",
           "crossed", "sector_orbits", "induced_orbits", "tables_gauge"]

#: The draws the sectors are read from, in turn, until one reads them as the
#: sum of matrix algebras the conserved matrices are: a generic draw does at
#: the first, and each later one is a fresh look at the same space.
_SEEDS = (7, 8, 9, 10)

@dataclass(frozen=True, eq=False)
class Sector:
    """One sector: ``copies`` identical copies of a set of ``dim`` states.

    Attributes
    ----------
    basis : numpy.ndarray, shape (n, copies · dim)
        Orthonormal columns in the orbital factor, column ``a · dim + w`` the
        state ``w`` of copy ``a``.  On the sector every conserved matrix
        acting within sites is ``basis (c ⊗ 1) basis†``, ``c`` acting on the
        copies.
    copies : int
        The number of identical copies, ``m``.
    dim : int
        The number of states in one copy, ``d``, counted in the orbital factor.
    atoms : tuple of int
        The indices of the orbital factor the sector's states lie on.
    """

    basis: np.ndarray
    copies: int
    dim: int
    atoms: tuple

@dataclass(frozen=True, eq=False)
class SectorReading:
    """The sectors of one Hamiltonian, read once where it enters.

    Attributes
    ----------
    algebra : numpy.ndarray, shape (k, n, n)
        An orthonormal basis of the conserved matrices acting within sites,
        in the orbital factor.
    sectors : tuple of Sector
        The sectors, ordered by the lowest index each touches.
    atoms : tuple of tuple of int
        The orbitals each index of the orbital factor stands for: one orbital,
        or a declared spinor pair's two.
    su2 : bool
        Whether the states carry a spin, native or declared, so that the
        orbital factor is read off the members ``O ⊗ 1``.
    defect : float
        How far the conserved matrices miss the sectors' form, relative to
        their scale: rounding where the reading is exact, and up to the
        declared tolerance where the matrices were measured at it.  Infinite
        where no draw read the sectors, all the states then forming one
        sector of one copy.
    tol : float or None
        The declared tolerance the reading was made at.
    """

    algebra: np.ndarray
    sectors: tuple
    atoms: tuple
    su2: bool
    defect: float
    tol: float | None = None

@dataclass(frozen=True, eq=False)
class SectorAction:
    """What the operations of one group do to the sectors.

    Attributes
    ----------
    images : dict
        Per operation, by its index, a tuple whose entry ``j`` is the sector
        the operation carries sector ``j`` onto.
    factors : dict
        Per ``(ind, j)``, the ``d × d`` factor ``w``, unitary and of
        determinant one, that every matrix of operation ``ind`` carries on
        each copy's states from sector ``j`` to its image.  Those matrices are
        ``c ⊗ w`` there, ``c`` acting on the copies.
    defect : float
        How far the operations' matrices miss that form, relative to their
        scale.
    """

    images: dict
    factors: dict
    defect: float

def _atoms_of(model, spin):
    """The indices of the orbital factor, and the states' spin order.

    ``model`` is the pythtb model and ``spin`` its declared spinor basis
    (:class:`..spin.SpinBasis`), ``None`` where none is declared.  Returns
    ``(atoms, order, su2)``: ``atoms`` holds the orbitals of each index of the
    orbital factor, one orbital or a declared pair's two; ``order`` puts the
    states in pair order, the spin fast, and is ``None`` on states with no
    spin; ``su2`` says whether the states carry a spin.
    """
    if int(model.nspin) == 2:
        atoms = tuple((int(o),) for o in range(int(model.norb)))
        return atoms, np.arange(int(model.nstate)), True
    if spin is not None:
        return (tuple(tuple(int(o) for o in pair) for pair in spin.pairs),
                np.asarray(spin._interleaved_index, dtype=int), True)
    return tuple((int(o),) for o in range(int(model.norb))), None, False

def _orthonormal(members, tol):
    """An orthonormal basis of the span of ``members``.

    ``members`` has shape ``(k, n, n)``, and the inner product is the trace
    one.  ``tol`` is the declared tolerance.  A direction is kept where its
    singular value exceeds :data:`.._tolerance.ZERO`, or ``tol`` where
    larger, relative to the largest singular value or to one where that is
    smaller.  Returns an array of shape ``(r, n, n)``.
    """
    A = np.asarray(members, dtype=complex)
    if not len(A):
        return A
    n = int(A.shape[1])
    # the right singular vectors span the members' own span; the conjugate span
    # would commute with H(k)* and not with H(k)
    _u, s, vh = np.linalg.svd(A.reshape(len(A), -1), full_matrices=False)
    keep = s > noise_floor(ZERO, tol) * max(float(s[0]), 1.0)
    return vh[keep].reshape(int(keep.sum()), n, n)

def _orbital_factor(members, sym, order, su2, tol):
    """The orbital factors ``O`` of members ``O ⊗ S(g)``, orthonormal.

    ``members`` is a solution space of the operation ``sym``, read in its own
    form (:func:`.._multiplets._own_form_spaces`), and ``S(g)`` is that
    operation's spin lift (:func:`.._matrices._lift_of`).  ``order`` and
    ``su2`` are the states' spin order and whether they carry a spin
    (:func:`_atoms_of`), and ``tol`` is the declared tolerance.  The entry
    ``O_ab`` is ``tr(S(g)† M_ab) / 2``, ``M_ab`` a member's spin block between
    indices ``a`` and ``b``.  On states with no spin the members are their own
    orbital factors.  Returns an orthonormal basis of the factors' span
    (:func:`_orthonormal`).
    """
    M = np.asarray(members, dtype=complex)
    if not su2 or not len(M):
        return _orthonormal(M, tol)
    npair = int(M.shape[1]) // 2
    paired = M[:, order][:, :, order].reshape(len(M), npair, 2, npair, 2)
    lift = _matrices._lift_of(sym, True)
    O = 0.5 * np.einsum("jasbu,su->jab", paired, lift.conj())
    return _orthonormal(O, tol)

def _site_blocks(model, assignment, spin, image=None):
    """Per site, the block a matrix carrying it onto its image may fill.

    ``model`` and ``assignment`` give each site's states, and ``spin`` is not
    read.  ``image[s]`` is the site the operation carries site ``s`` onto;
    with ``image=None`` every site is its own, the blocks of a matrix acting
    within sites.  Returns ``[(rows, cols)]``, the image's states and the
    site's.
    """
    nspin = int(model.nspin)
    states = [np.asarray([o * nspin + t for o in orbs for t in range(nspin)], dtype=int)
              for orbs in assignment.site_orbitals]
    return [(states[s if image is None else int(image[s])], states[s])
            for s in range(len(states))]

def _within_sites(basis, blocks, tol):
    """The members of the solution space ``basis`` that lie within ``blocks``.

    ``blocks`` are a matrix's allowed blocks (:func:`_site_blocks`), and
    ``tol`` is the declared tolerance at which a member is read as lying
    within them (:func:`.._matrices._block_members`).
    """
    members, _reach = _matrices._block_members(basis, blocks,
                                               tol=noise_floor(_IN_BLOCKS, tol))
    return members

def _draw(B, rng):
    """One member of the span of ``B``, its coefficients drawn from ``rng``."""
    c = rng.normal(size=len(B)) + 1j * rng.normal(size=len(B))
    return np.einsum("j,jab->ab", c, B)

def _tensor_defect(Z, m, d):
    """How far one block ``Z`` on a sector misses ``c ⊗ 1``.

    ``Z`` acts on the sector's ``m`` copies of ``d`` states, copy-major.  The
    miss is relative to the block's scale.  ``Z`` is rearranged so that a
    product ``c ⊗ w`` has rank one.  Returns the larger of two numbers: the
    ratio of the second to the first singular value of that rearrangement,
    and how far the factor ``w`` of the first, normalized, misses a multiple
    of the identity.  Returns ``0.0`` for a zero block.
    """
    R = Z.reshape(m, d, m, d).transpose(0, 2, 1, 3).reshape(m * m, d * d)
    _u, s, vh = np.linalg.svd(R, full_matrices=False)
    if not s.size or float(s[0]) <= 0.0:
        return 0.0
    worst = float(s[1] / s[0]) if s.size > 1 else 0.0
    w = vh[0].reshape(d, d)
    trace = complex(np.trace(w)) / d
    return max(worst, float(np.abs(w - trace * np.eye(d)).max()))

def _sectors_once(B, rng, tol):
    """One reading of the sectors of the algebra ``B``.

    ``B`` is an orthonormal basis of the algebra, ``rng`` draws its members,
    and ``tol`` is the declared tolerance.  The eigenspaces of one drawn
    Hermitian member are the copies, one eigenvalue to each copy of each
    sector.  Two copies are of one sector where another drawn member carries
    one into the other.  That member, carried from a sector's first copy along
    the links found, puts every copy in one frame, so that each member of
    ``B`` acts on the sector as ``c ⊗ 1``.  Two more draws give the defect:
    what either carries between sectors, and how far either misses ``c ⊗ 1``
    on a sector (:func:`_tensor_defect`).  Returns ``(sectors, defect)``;
    ``None`` where the copies of one sector differ in dimension, or the links
    do not reach every copy.
    """
    n = int(B.shape[1])
    cluster, zero = noise_floor(CLUSTER, tol), noise_floor(ZERO, tol)
    X = _draw(B, rng)
    w, V = np.linalg.eigh(X + X.conj().T)
    scale = max(float(np.abs(w).max()) if w.size else 0.0, 1.0 / n)
    copies, start = [], 0
    for i in range(1, n + 1):
        if i == n or float(w[i] - w[i - 1]) > cluster * scale:
            copies.append(V[:, start:i])
            start = i
    links = [_draw(B, rng), _draw(B, rng)]
    # which copies a drawn member carries into one another: one sector each
    parent = list(range(len(copies)))

    def _find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    edges = {}
    for Y in links:
        floor = zero * max(float(np.abs(Y).max()), 1.0)
        for a in range(len(copies)):
            for b in range(len(copies)):
                if a == b:
                    continue
                blk = copies[b].conj().T @ Y @ copies[a]
                if float(np.abs(blk).max()) > floor:
                    edges.setdefault((a, b), Y)
                    ra, rb = _find(a), _find(b)
                    if ra != rb:
                        parent[max(ra, rb)] = min(ra, rb)
    groups = {}
    for a in range(len(copies)):
        groups.setdefault(_find(a), []).append(a)
    sectors = []
    for members in groups.values():
        d = int(copies[members[0]].shape[1])
        if any(int(copies[a].shape[1]) != d for a in members):
            return None                   # copies of one sector differ: no algebra here
        frame = {members[0]: copies[members[0]]}
        frontier = [members[0]]
        while frontier:
            a = frontier.pop()
            for b in members:
                if b in frame or (a, b) not in edges:
                    continue
                img = copies[b] @ (copies[b].conj().T @ edges[(a, b)] @ frame[a])
                u, sv, vh = np.linalg.svd(img, full_matrices=False)
                # the copy's states in the frame the link carries the first
                # copy's into, made orthonormal
                frame[b] = u @ vh
                frontier.append(b)
        if len(frame) != len(members):
            return None
        Q = np.concatenate([frame[a] for a in members], axis=1)
        atoms = tuple(int(i) for i in np.flatnonzero(
            np.linalg.norm(Q, axis=1) > zero * max(float(np.abs(Q).max()), 1.0)))
        sectors.append(Sector(basis=Q, copies=len(members), dim=d, atoms=atoms))
    sectors.sort(key=lambda s: s.atoms[0] if s.atoms else n)
    defect = 0.0
    for Z in (_draw(B, rng), _draw(B, rng)):
        size = max(float(np.abs(Z).max()), 1.0 / n)
        for i, si in enumerate(sectors):
            for j, sj in enumerate(sectors):
                blk = si.basis.conj().T @ Z @ sj.basis
                if i != j:
                    defect = max(defect, float(np.abs(blk).max()) / size)
                else:
                    defect = max(defect, _tensor_defect(blk, sj.copies, sj.dim))
    return tuple(sectors), defect

def read_sectors(model, group, assignment, spin, validated, *, tol=None):
    """The sectors of one Hamiltonian, read off the identity's solution space.

    ``model`` is the pythtb model, ``group`` the group it is measured on
    (:class:`.._spacegroup.GroupData`), ``assignment`` the sites of its
    orbitals (:class:`.._spacegroup.OrbitalAssignment`), ``spin`` its
    declared spinor basis or ``None``, and ``validated`` the detector's
    verdict, whose ``_bases`` hold each operation's solution space
    (:class:`.._measure.ValidatedGroup`).  ``tol`` is the declared tolerance.
    The identity's space is every matrix that commutes with ``H(k)`` at every
    ``k``, and its members acting within sites are read in the orbital factor
    (:func:`_orbital_factor`).  A coupling below the declared tolerance leaves
    the pieces it joins apart, and one above it joins them.  Returns a
    :class:`SectorReading`; ``None`` where the identity's space is not held,
    is past the detector's memory ceiling for reading a space off, or has no
    member acting within sites.
    """
    syms = group._symmetries
    ident = _matrices._identity_position(group)
    basis = (getattr(validated, "_bases", None) or {}).get(int(syms[ident].ind))
    if basis is None:
        return None
    basis = np.asarray(basis, dtype=complex)
    if basis.ndim != 3 or not len(basis):
        return None
    nstate = int(basis.shape[1])
    if basis.shape[0] * nstate * nstate * 16 > _detector._READOFF_MAX_BYTES:
        return None
    atoms, order, su2 = _atoms_of(model, spin)
    members = _within_sites(basis, _site_blocks(model, assignment, spin), tol)
    B = _orbital_factor(members, syms[ident], order, su2, tol)
    if not len(B):
        return None
    sectors, defect = _sectors_of(B, tol)
    return SectorReading(algebra=B, sectors=sectors, atoms=atoms, su2=su2,
                         defect=float(defect), tol=tol)

def _sectors_of(B, tol):
    """The sectors of the algebra ``B``, the best of the draws :data:`_SEEDS`.

    ``tol`` is the declared tolerance.  Each seed reads the sectors once
    (:func:`_sectors_once`), the reading with the smallest defect is kept, and
    the draws stop at one within :data:`.._tolerance._FACTOR_TOL`, or ``tol``
    where larger.  Returns ``(sectors, defect)``.  Where no draw reads them,
    all the states form one sector of one copy, with an infinite defect.
    """
    best = None
    for seed in _SEEDS:
        got = _sectors_once(B, np.random.default_rng(seed), tol)
        if got is not None and (best is None or got[1] < best[1]):
            best = got
        if best is not None and best[1] <= noise_floor(_FACTOR_TOL, tol):
            break
    if best is None:
        # no draw read an algebra: the conserved matrices are read as one sector
        # of one copy, the crystal's whole set of states, which assumes nothing
        Q = np.eye(int(B.shape[1]), dtype=complex)
        best = ((Sector(basis=Q, copies=1, dim=int(B.shape[1]),
                        atoms=tuple(range(int(B.shape[1])))),), float("inf"))
    return best

def carried_reading(reading, layout, *, tol=None):
    r"""The sectors as the multiplets carry them.

    ``reading`` is the :class:`SectorReading`, ``layout`` the multiplets the
    representation is written on (:class:`..symmetry.SiteMultiplets`), and
    ``tol`` the declared tolerance the reading was made at.  Where every
    conserved matrix acting within sites acts within each multiplet,
    ``reading`` itself is returned.  Where the multiplets keep identical copies
    apart, the matrices exchanging the copies reach between multiplets.  The
    conserved matrices that do not are then read on their own, each copy a
    sector of one copy, and their :class:`SectorReading` is returned.
    """
    index = {int(o): a for a, orbs in enumerate(reading.atoms) for o in orbs}
    B = reading.algebra
    n = int(B.shape[1])
    inside = np.zeros((n, n), dtype=bool)
    for orbs in layout._chan_orbs:
        idx = sorted({index[int(o)] for o in orbs})
        inside[np.ix_(idx, idx)] = True
    outside = B[:, ~inside]
    floor = noise_floor(_IN_BLOCKS, tol)
    if not outside.size or float(np.abs(outside).max()) <= floor:
        return reading
    # the members of the algebra with nothing between two multiplets
    _u, s, vh = np.linalg.svd(outside.T, full_matrices=True)
    top = max(float(s[0]), 1.0) if s.size else 1.0
    keep = [i for i in range(len(B)) if i >= len(s) or float(s[i]) <= floor * top]
    members = np.asarray([np.einsum("j,jab->ab", vh[i].conj(), B) for i in keep])
    carried = _orthonormal(members, tol)
    sectors, defect = _sectors_of(carried, tol)
    return SectorReading(algebra=carried, sectors=sectors, atoms=reading.atoms,
                         su2=reading.su2, defect=float(defect), tol=reading.tol)

def joined(reading, site_atoms):
    r"""How far the conserved matrices join each pair of atoms of one site.

    ``reading`` is the :class:`SectorReading`, and ``site_atoms`` are the
    site's atoms, each the orbitals of one index of the orbital factor.
    Returns the matrix ``W``.  ``W[i, j]``, how far atom ``i`` is joined to
    atom ``j``, is the largest, over the sectors, of the reach between the two
    atoms of the sector's projector and of the matrices exchanging its
    identical copies, the latter summed in square over every pair of copies.
    The projector's reach is zero to rounding where each sector holds each
    atom wholly or not at all, and of order one where it holds part of each:
    a square's ``p_x, p_y`` pair that nothing couples, written as
    ``(p_x ± p_y)/√2``, is joined by a half.  The exchange's reach is the same
    in any basis of the copies, and joins two identical ``s`` orbitals by one,
    with a spin or without.  The diagonal is zero.
    """
    index = {int(o): a for a, orbs in enumerate(reading.atoms) for o in orbs}
    idx = np.asarray([index[int(atom[0])] for atom in site_atoms], dtype=int)
    W = np.zeros((len(idx), len(idx)))
    for sector in reading.sectors:
        rows = sector.basis[idx, :]
        W = np.maximum(W, np.abs(rows @ rows.conj().T))
        m, d = int(sector.copies), int(sector.dim)
        if m > 1:
            # copy a's states at these atoms, against copy b's: the matrix
            # carrying copy b onto copy a, between the two atoms
            per = rows.reshape(len(idx), m, d)
            reach = np.einsum("iaw,jbw->ijab", per, per.conj())
            W = np.maximum(W, np.sqrt(np.sum(np.abs(reach) ** 2, axis=(2, 3))))
    np.fill_diagonal(W, 0.0)
    return W

def sector_action(reading, model, group, assignment, spin, validated, *, tol=None):
    """What the operations of ``group`` do to the sectors of ``reading``.

    ``reading`` is the :class:`SectorReading`; ``model``, ``group``,
    ``assignment``, ``spin`` and ``validated`` are as :func:`read_sectors`
    takes them, and ``tol`` is the declared tolerance.  Each operation's
    solution space is read within the sites it carries onto one another
    (:func:`.._matrices._site_spaces`), in the orbital factor.  One member
    drawn from it gives the sector each sector is carried onto, the one its
    weight lies on.  Two members give the factor ``w`` on each copy's states
    between a sector and its image: the common factor of their ``c ⊗ w``
    form, made unitary and of determinant one, so that two factors compose to
    a third times a ``d``-th root of unity.  An antiunitary operation's
    matrices act on the conjugate states.  Returns a :class:`SectorAction`.
    It leaves out an operation the detector kept no solution space for, one
    too large to read, one none of whose members acts within sites, and one
    that does not carry each sector wholly onto a sector of the same number
    of copies and states per copy.
    """
    syms = group._symmetries
    pos_of = {int(s.ind): p for p, s in enumerate(syms)}
    _atoms, order, su2 = _atoms_of(model, spin)
    spaces, _layout, _mixing = _matrices._site_spaces(model, group, assignment, spin,
                                                      validated, tol=tol)
    zero = noise_floor(ZERO, tol)
    sectors = reading.sectors
    images, factors, defect = {}, {}, 0.0
    rng = np.random.default_rng(_SEEDS[0])
    for ind in sorted(spaces):
        p = pos_of[int(ind)]
        O = _orbital_factor(spaces[ind], syms[p], order, su2, tol)
        if not len(O) or int(O.shape[1]) != int(reading.algebra.shape[1]):
            continue
        au = bool(syms[p].time_reversal)
        X, Y = _draw(O, rng), _draw(O, rng)
        here, ok = [], True
        for sj in sectors:
            src = sj.basis.conj() if au else sj.basis
            weight = np.asarray([float(np.linalg.norm(si.basis.conj().T @ X @ src))
                                 for si in sectors])
            i = int(np.argmax(weight))
            total = float(np.linalg.norm(weight))
            if total <= 0.0 or float(np.sqrt(max(total ** 2 - weight[i] ** 2, 0.0))) > (
                    zero * total) or sectors[i].copies != sj.copies or (
                    sectors[i].dim != sj.dim):
                ok = False
                break
            here.append(i)
        if not ok:
            continue
        images[int(ind)] = tuple(here)
        for j, sj in enumerate(sectors):
            si = sectors[here[j]]
            src = sj.basis.conj() if au else sj.basis
            m, d = sj.copies, sj.dim
            R = np.concatenate([
                (si.basis.conj().T @ Z @ src).reshape(m, d, m, d)
                .transpose(0, 2, 1, 3).reshape(m * m, d * d) for Z in (X, Y)], axis=0)
            _u, s, vh = np.linalg.svd(R, full_matrices=False)
            if s.size > 1 and float(s[0]) > 0.0:
                defect = max(defect, float(s[1] / s[0]))
            a, _sv, bh = np.linalg.svd(vh[0].reshape(d, d))
            w = a @ bh
            # of determinant one: two factors then compose to a third times a
            # d-th root of unity, read exactly whatever the noise
            det = complex(np.linalg.det(w))
            if abs(det) > 0.0:
                w = w * np.exp(-1j * np.angle(det) / d)
            factors[(int(ind), j)] = w
    return SectorAction(images=images, factors=factors, defect=float(defect))

# --------------------------------------------------------------------------
# the representations a subgroup leaves open, and what each puts at a site
# --------------------------------------------------------------------------

class _Unread(Exception):
    """The sectors read no representation of the operations asked.

    Raised where the factors the operations carry on the sectors do not
    compose, where an operation an orbit needs has no reading on them, where
    the multiplets hold part of an orbit or split its copies, where the
    tables' spin sign for a product disagrees with the factor the spin lifts
    carry, or where no draw separates a stabilizer's representations with
    the factor system read (:func:`.._matrices._projective_irreps`).
    """

def _composed(wa, wb, wab, *, after_antiunitary=False, tol=None):
    r"""The root of unity ``ρ`` with ``w(a) w(b) = ρ w(ab)``.

    ``wa``, ``wb`` and ``wab`` are the factors of :func:`sector_action` for
    ``a``, ``b`` and ``ab`` on one sector, ``d × d`` and of determinant one,
    so ``ρ^d = 1``.  With ``after_antiunitary=True``, ``a`` is antiunitary and
    ``w(b)*`` stands in place of ``w(b)``.  ``tol`` is the declared tolerance
    the factors were read at.  Returns the ``d``-th root of unity the measured
    ratio sits at, exactly.  Raises :class:`_Unread` where ``w(a) w(b)`` is
    not a multiple of ``w(ab)``, or where the ratio lies farther than a
    quarter of the roots' spacing from the nearest root.
    """
    rho, res = _matrices._proportionality(wa @ (wb.conj() if after_antiunitary else wb), wab)
    if res > noise_floor(_FACTOR_TOL, tol) or abs(rho) <= 0.0:
        raise _Unread
    d = int(np.asarray(wa).shape[0])
    root = np.exp(2j * np.pi * round(float(np.angle(rho)) * d / (2.0 * np.pi)) / d)
    spacing = abs(1.0 - np.exp(2j * np.pi / d)) if d > 1 else 1.0
    if abs(complex(rho) - root) > 0.25 * spacing:
        raise _Unread
    return complex(root)

class _TooMany(Exception):
    """More representations to weigh than ``max_candidates``.

    ``count`` is how many there are, and ``orbitals`` the orbitals they are
    weighed on.  ``sets`` is the number of orbits read together where
    several are (:class:`SharedOrbits`), ``None`` otherwise.  ``copies`` is
    the largest number of identical copies a sector among them holds where
    ``sets`` is ``None``, and ``1``, standing for nothing, where it is given.
    """

    def __init__(self, count, orbitals, copies, sets=None):
        super().__init__(int(count))
        self.count, self.orbitals, self.copies = int(count), tuple(orbitals), int(copies)
        self.sets = None if sets is None else int(sets)

def _extends(rep, positions, a_pos, prod, factor, *, tol=None, seed=17):
    r"""Whether a stabilizer's representation extends to an antiunitary operation.

    ``rep`` holds ``b(u)`` for every unitary ``u`` of ``positions``, a
    representation with the factor system ``f``, which ``factor`` holds keyed
    by pairs of positions.  ``a_pos`` is an antiunitary operation ``a`` fixing
    the same sector, and ``prod`` the group's multiplication table by
    position.  The matrix ``X`` that ``a`` would carry satisfies
    ``X b(u)* = f(a, u) / f(u', a) · b(u') X`` for every ``u``, with
    ``u' a = a u``, and ``X X* = f(a, a) b(a²)``.  For a unitary ``X`` the
    last is the linear ``X = Xᵀ · conj(f(a, a) b(a²))``.  Members of that
    solution space, drawn from ``seed``, are made unitary, and one that meets
    both within the declared tolerance ``tol`` returns ``True``.  One of a
    pair of complex-conjugate representations without its partner is carried
    by no such ``X``.  Returns ``False`` too where ``a²`` or some ``u'`` is
    not among ``positions``.
    """
    positions = [int(p) for p in positions]
    square = int(prod[a_pos, a_pos])
    if square not in rep:
        return False
    k = int(np.asarray(rep[positions[0]]).shape[0])
    eye = np.eye(k)
    conj_of = {}
    for u in positions:
        target = int(prod[a_pos, u])
        mate = [v for v in positions if int(prod[v, a_pos]) == target]
        if len(mate) != 1:
            return False
        conj_of[u] = mate[0]
    ratio = {u: complex(factor[(a_pos, u)]) / complex(factor[(conj_of[u], a_pos)])
             for u in positions}
    rows = [np.kron(np.asarray(rep[u]).conj().T, eye) - ratio[u] * np.kron(eye, rep[conj_of[u]])
            for u in positions]
    swap = np.zeros((k * k, k * k))
    for i in range(k):
        for j in range(k):
            swap[i + j * k, j + i * k] = 1.0
    squared = complex(factor[(a_pos, a_pos)]) * np.asarray(rep[square], dtype=complex)
    rows.append(np.eye(k * k) - np.kron(squared.conj().T, eye) @ swap)
    _u, sv, vh = np.linalg.svd(np.vstack(rows))
    top = max(1.0, float(sv[0])) if sv.size else 1.0
    zero, close = noise_floor(ZERO, tol), noise_floor(ASSEMBLED, tol)
    null = [vh[i].conj() for i in range(len(sv)) if float(sv[i]) <= zero * top]
    null += [vh[i].conj() for i in range(len(sv), k * k)]
    if not null:
        return False
    rng = np.random.default_rng(int(seed))
    for _ in range(4):
        y = sum(complex(rng.normal(), rng.normal()) * v for v in null)
        w, _s, wh = np.linalg.svd(y.reshape(k, k, order="F"))
        X = w @ wh
        if all(float(np.abs(X @ np.asarray(rep[u]).conj()
                            - ratio[u] * np.asarray(rep[conj_of[u]]) @ X).max()) <= close
               for u in positions) and float(np.abs(X @ X.conj() - squared).max()) <= close:
            return True
    return False

class SectorSearch:
    r"""The search for whether a declaration is carried on a subgroup.

    Made once per round of the search for the group.  :meth:`met` says, for
    any subgroup of the round's unitary operations, whether one
    representation per orbit of sectors under it, of those the sectors leave
    open, gives every declared site what was declared.  That is the same
    character where the whole site was declared, a character containing it
    where part of the site was, and the corep multiplicities alone where the
    declaration is written in coreps.  A site's character under each choice
    is the induced character, read without building a matrix.  An orbit time
    reversal carries onto another is chosen once, its partner carrying the
    conjugate.  Declarations sharing no orbit are weighed apart.  Where the
    sectors of an orbit carry no representation of their stabilizer with the
    tables' factor system, the declarations on them are not weighed.  That
    is a verdict on the states' projective class, given apart from the
    declarations.  Where, on the identity alone, no representation the copies
    admit both extends to time reversal and carries the declarations, they
    are asked of the unitary operations alone.  Once made, ``complete`` is
    false where an operation of the round has no reading on the sectors,
    which then say nothing.

    Parameters
    ----------
    reading : SectorReading
        The sectors as the multiplets carry them (:func:`carried_reading`).
    action : SectorAction
        What the round's operations do to them (:func:`sector_action`).
    group : .._spacegroup.GroupData
        The group of the round.
    assignment : .._spacegroup.OrbitalAssignment
        The sites of the orbitals.
    decls : sequence of dict
        The declarations as the round reads them, each over whole
        multiplets of one site.
    positions : iterable of int
        The round's unitary operations still standing, by position in the
        group's list of operations.
    reversal : int or None, default None
        The position of the pure time reversal where it stands.
    tol : float, default CHARACTER
        How far a character may sit from a declared one and be read as it.
        It also bounds how far the tables' spin sign for a product may sit
        from the factor the spin lifts carry (:meth:`_tau`).
    max_candidates : int, default 2 ** 20
        The most representations weighed at once.
    """

    def __init__(self, reading, action, group, assignment, decls, positions, *,
                 reversal=None, tol=CHARACTER, max_candidates=2 ** 20):
        self.reading, self.group, self.assignment = reading, group, assignment
        self.syms = group._symmetries
        self.prod, _tdiff, self.spf = group._get_product_table(get_diff=True)
        self.ident = _matrices._identity_position(group)
        self.pos_of = {int(s.ind): p for p, s in enumerate(self.syms)}
        self.unitary = sorted(int(p) for p in positions)
        self.reversal = None if reversal is None else int(reversal)
        self.tol, self.max_candidates = float(tol), int(max_candidates)
        self.images = {self.pos_of[int(i)]: tuple(int(j) for j in got)
                       for i, got in action.images.items() if int(i) in self.pos_of}
        self.factor = {(self.pos_of[int(i)], int(j)): w
                       for (i, j), w in action.factors.items() if int(i) in self.pos_of}
        need = self.unitary + ([self.reversal] if self.reversal is not None else [])
        self.complete = all(p in self.images for p in need)
        unitary = set(self.unitary)
        every = [q for q, s in enumerate(self.syms) if not s.time_reversal]
        self.inv = {p: next(q for q in every if int(self.prod[p, q]) == self.ident)
                    for p in every}
        self._lifts, self._target, self._orbit_of, self._theta_of = {}, {}, {}, {}
        index = {int(o): a for a, orbs in enumerate(reading.atoms) for o in orbs}
        zero = noise_floor(ZERO, reading.tol)
        self.items = []
        for d in decls:
            atoms = sorted({index[int(o)] for o in d.get("block", d["orbitals"])})
            site = int(d["site"])
            corep = bool(d.get("corep"))
            self.items.append({
                "atoms": np.asarray(atoms, dtype=int), "site": site,
                "site_ops": frozenset(self.pos_of[int(i)] for i in assignment.site_group[site]
                                      if self.pos_of.get(int(i)) in unitary),
                "want": {self.pos_of[int(i)]: complex(v) for i, v in d["constraints"].items()
                         if int(i) in self.pos_of},
                "partial": bool(d.get("partial")), "corep": corep,
                "touch": {j for j, s in enumerate(reading.sectors)
                          if float(np.linalg.norm(s.basis[atoms, :])) > zero},
                "table_of": _naming._SiteTableOf(group, assignment, site, corep=corep)})
        # Where these states hold time reversal, the declarations are first
        # asked on the identity alone, where the copies as they stand carry
        # every one of them: where none of the representations the copies
        # admit extends to time reversal, its square on the states is not the
        # one the tables ask for, as for a spinor read in single-valued tables.
        # It is then dropped and the declarations are asked of the unitary
        # operations alone; its own verdict says so once those settle.
        if self.reversal is not None and self.complete:
            try:
                if not self.met({int(self.syms[self.ident].ind)}):
                    self.reversal = None
                    self._orbit_of.clear()
                    self._theta_of.clear()
            except _Unread:
                self.complete = False

    # ---- the factors the orbital part of a representation carries --------------
    def _lift(self, p):
        if p not in self._lifts:
            self._lifts[p] = _matrices._lift_of(self.syms[p], self.reading.su2)
        return self._lifts[p]

    def _tau(self, x, y):
        """The factor ``τ`` with ``O(x) O(y) = τ O(xy)`` on the orbital part.

        ``x`` and ``y`` are positions of operations.  ``τ`` is the table's
        spin sign for their product over the factor the spin lifts carry,
        ``S(x) S(y) = λ S(xy)``, ``S(y)*`` standing in place of ``S(y)`` after
        an antiunitary ``x``.  Returns ``1`` where the two agree within the
        search's ``tol``.  Raises :class:`_Unread` where they do not, or where
        ``S(x) S(y)`` is not a multiple of ``S(xy)``.
        """
        key = (int(x), int(y))
        if key not in self._target:
            Sy = self._lift(y)
            lam, res = _matrices._proportionality(
                self._lift(x) @ (Sy.conj() if self.syms[x].time_reversal else Sy),
                self._lift(int(self.prod[x, y])))
            if (res > noise_floor(ASSEMBLED, self.reading.tol) or abs(lam) <= 0.0
                    or abs(complex(self.spf[x, y]) / lam - 1.0) > self.tol):
                raise _Unread
            self._target[key] = 1.0 + 0.0j
        return self._target[key]

    def _rho(self, wa, wb, wab, *, after_antiunitary=False):
        """The root of unity ``ρ`` with ``w(a) w(b) = ρ w(ab)`` on one sector.

        ``wa``, ``wb`` and ``wab`` are the factors of ``a``, ``b`` and ``ab``;
        with ``after_antiunitary=True``, ``w(b)*`` stands in place of
        ``w(b)``.  Read by :func:`_composed` at the reading's tolerance.
        """
        return _composed(wa, wb, wab, after_antiunitary=after_antiunitary,
                         tol=self.reading.tol)

    # ---- one orbit of sectors under a subgroup ---------------------------------
    def _orbit(self, orbit, H):
        """The stabilizer, frames and representations of one orbit under ``H``.

        ``orbit`` lists the orbit's sectors, the first its reference, and ``H``
        the subgroup's unitary operations by position.  Returns a mapping,
        cached: ``orbit``, the sectors as handed in; ``rep[t]``, an operation
        carrying the first sector onto ``t``; ``K``, the first sector's
        stabilizer, and ``w``, the factors on it; ``frames[t]``, the basis of
        ``t`` in the frame ``rep[t]`` carries the first sector's into; ``m``
        and ``d``; ``kind``, how time reversal meets the orbit, ``"fixes"``,
        ``"inside"``, ``"pairs"`` or ``None``; ``options``, the character on
        ``K`` of each representation of ``K`` on the copies with the factor
        system the sectors leave; of these, where an antiunitary operation
        fixes the first sector, only those that extend to it; ``orbitals``;
        and ``obstructed``, true where that factor system admits
        no representation on the copies.  Raises :class:`_Unread` where ``H``
        does not carry the first sector onto every sector of ``orbit``, where
        an operation the orbit needs has no factor on the sectors, where the
        factors do not compose (:meth:`_tau`, :meth:`_rho`), or where no draw
        separates the stabilizer's representations.  Raises
        :class:`_TooMany` where the representations on the copies number
        more than ``max_candidates``.
        """
        key = (tuple(orbit), frozenset(H))
        if key in self._orbit_of:
            return self._orbit_of[key]
        prod = self.prod
        j0 = int(orbit[0])
        rep, frontier = {j0: self.ident}, [j0]
        while frontier:
            a = frontier.pop()
            for h in H:
                b = self.images[h][a]
                if b not in rep:
                    rep[b] = int(prod[h, rep[a]])
                    frontier.append(b)
        if set(rep) != set(orbit):
            raise _Unread
        K = sorted(h for h in H if self.images[h][j0] == j0)
        sector = self.reading.sectors[j0]
        m, d = int(sector.copies), int(sector.dim)

        def _w(p):
            got = self.factor.get((int(p), j0))
            if got is None:
                raise _Unread
            return got

        w = {k: _w(k) for k in K}
        nu = {}
        for a in K:
            for b in K:
                ab = int(prod[a, b])
                nu[(a, b)] = self._tau(a, b) / self._rho(w[a], w[b], _w(ab))
        frames = {}
        for t in orbit:
            wt = (np.eye(d, dtype=complex) if t == j0
                  else self.factor.get((rep[t], j0)))
            if wt is None:
                raise _Unread
            frames[t] = self.reading.sectors[t].basis @ np.kron(np.eye(m), wt)
        # time reversal on the orbit: fixing its first sector, carrying it inside
        # the orbit, or onto another orbit, whose representation is then the
        # conjugate of this one's
        kind, a_pos, f = None, None, dict(nu)
        T = self.reversal
        if T is not None:
            image = self.images[T][j0]
            if image == j0:
                kind, a_pos = "fixes", T
            elif image in rep:
                kind, a_pos = "inside", int(prod[self.inv[rep[image]], T])
                if a_pos not in self.images or self.images[a_pos][j0] != j0:
                    raise _Unread
            else:
                kind = "pairs"
        if a_pos is not None:
            wa = _w(a_pos)
            for k in K:
                ak, ka = int(prod[a_pos, k]), int(prod[k, a_pos])
                f[(a_pos, k)] = self._tau(a_pos, k) / self._rho(wa, w[k], _w(ak),
                                                                 after_antiunitary=True)
                f[(k, a_pos)] = self._tau(k, a_pos) / self._rho(w[k], wa, _w(ka))
            aa = int(prod[a_pos, a_pos])
            f[(a_pos, a_pos)] = self._tau(a_pos, a_pos) / self._rho(wa, wa, _w(aa),
                                                                     after_antiunitary=True)
        # the factor system is read exactly (:func:`_composed`), so the
        # representations are read at the exact defaults, never merged by a
        # declared precision
        irreps = _matrices._projective_irreps(K, prod, nu)
        if irreps is None:
            raise _Unread
        dims = [int(np.asarray(r[K[0]]).shape[0]) for r in irreps]
        orbitals = tuple(sorted(int(o) for t in orbit for a in self.reading.sectors[t].atoms
                                for o in self.reading.atoms[int(a)]))
        count = _matrices._count_dimension_sums(dims, m)
        if count > self.max_candidates:
            raise _TooMany(count, orbitals, m)
        options = []
        # no representation of the stabilizer on the copies at all: the states
        # of the orbit carry none with the tables' factor system, which is their
        # projective class, a verdict of its own and not the declaration's
        obstructed = count == 0
        for combo in _matrices._dimension_sums(dims, m):
            b = {}
            for k in K:
                M = np.zeros((m, m), dtype=complex)
                at = 0
                for i in combo:
                    blk = np.asarray(irreps[i][k])
                    M[at:at + blk.shape[0], at:at + blk.shape[0]] = blk
                    at += blk.shape[0]
                b[k] = M
            if a_pos is not None and not _extends(b, K, a_pos, prod, f, tol=self.reading.tol):
                continue
            options.append({k: complex(np.trace(b[k])) for k in K})
        got = {"orbit": tuple(int(t) for t in orbit), "rep": rep, "K": K, "w": w,
               "frames": frames, "m": m, "d": d, "kind": kind, "options": options,
               "orbitals": orbitals, "obstructed": obstructed}
        self._orbit_of[key] = got
        return got

    def _contributions(self, i, rec, H):
        """What one orbit puts at declared site ``i`` under each operation.

        ``rec`` is the orbit's record (:meth:`_orbit`) and ``H`` the subgroup
        by position.  An operation ``h`` of the site group fixing a sector
        ``t`` of the orbit acts there as ``k = r_t⁻¹ h r_t`` of the stabilizer,
        ``r_t`` carrying the orbit's first sector onto ``t``.  The site's share
        of the induced character is ``tr b(k)`` times ``θ = tr(π_t w(k))``,
        ``π_t`` the projector onto the site's indices, compressed to the first
        copy of ``t``, in the frame of ``rec``.  Returns ``{h: [(k, θ)]}``,
        cached.
        """
        key = (int(i), rec["orbit"], frozenset(H))
        if key in self._theta_of:
            return self._theta_of[key]
        item, prod, d = self.items[i], self.prod, rec["d"]
        theta = {}
        for t in rec["orbit"]:
            F = rec["frames"][t][item["atoms"], :]
            pi = (F.conj().T @ F)[:d, :d]
            theta[t] = {k: complex(np.trace(pi @ rec["w"][k])) for k in rec["K"]}
        out = {}
        for h in H:
            if h not in item["site_ops"]:
                continue
            here = []
            for t in rec["orbit"]:
                if self.images[h][t] != t:
                    continue
                r = rec["rep"][t]
                k = int(prod[self.inv[r], int(prod[h, r])])
                here.append((k, theta[t][k]))
            out[h] = here
        self._theta_of[key] = out
        return out

    # ---- whether the declarations are carried -----------------------------------
    def _holds(self, i, content, H):
        """Whether declared site ``i`` holds what was declared.

        ``content`` maps operations of ``H``, by position, to the site's
        character.  A whole site's declaration holds where the characters
        agree within the search's ``tol``; one written in coreps is not
        compared where the operation's spin lift has zero trace.  A
        declaration of part of the site holds where the character contains it
        on the site group ``H`` leaves (:func:`.._naming._contained`).  Where
        the character is not read on every class of that site group, ``True``
        is returned.
        """
        item = self.items[i]
        ops = [h for h in H if h in content]
        if not item["partial"]:
            for h in ops:
                if h not in item["want"]:
                    continue
                if item["corep"] and abs(complex(np.trace(self._lift(h)))) <= ZERO:
                    continue                  # a corep declaration says nothing here
                if abs(content[h] - item["want"][h]) > self.tol:
                    return False
            return True
        # part of what the site holds: contained in it, on the site group H leaves
        table = item["table_of"](tuple(int(self.syms[h].ind) for h in ops))
        held, want, asked = [], [], []
        for cls in table.classes:
            h = self.pos_of[int(cls[0])]
            if h not in content or h not in item["want"]:
                return True                   # unread: nothing said either way
            held.append(content[h])
            want.append(item["want"][h])
            asked.append(abs(complex(np.trace(self._lift(h)))) > ZERO if item["corep"] else True)
        return _naming._contained(table, held, want, corep=item["corep"], asked=asked)

    def met(self, sub):
        """Whether the subgroup ``sub`` carries every declaration.

        ``sub`` holds the indices of the subgroup's unitary operations.
        Raises :class:`_TooMany` where the choices to weigh pass
        ``max_candidates``, and :class:`_Unread` where the sectors read
        nothing on ``sub``.
        """
        H = sorted(self.pos_of[int(i)] for i in sub if int(i) in self.pos_of)
        sectors = self.reading.sectors
        parent = list(range(len(sectors)))

        def _find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for h in H:
            for j, image in enumerate(self.images[h]):
                ra, rb = _find(j), _find(int(image))
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
        orbits = {}
        for j in range(len(sectors)):
            orbits.setdefault(_find(j), []).append(j)
        orbit_of = {j: root for root, members in orbits.items() for j in members}
        touched = sorted({orbit_of[j] for item in self.items for j in item["touch"]})
        recs = {root: self._orbit(orbits[root], H) for root in touched}
        # an orbit time reversal carries onto another touched one is chosen once,
        # its partner carrying the conjugate
        partner = {}
        if self.reversal is not None:
            for root in touched:
                if recs[root]["kind"] == "pairs":
                    partner[root] = orbit_of[self.images[self.reversal][orbits[root][0]]]
        chosen = {root: (partner[root] if root in partner and partner[root] in recs
                         and partner[root] < root else root) for root in touched}
        doubled = {root: (root in partner and partner[root] in recs) for root in touched}
        units_of = [sorted({chosen[orbit_of[j]] for j in item["touch"]}) for item in self.items]
        # a declaration on states that carry no representation of their
        # stabilizer here is not weighed on this subgroup: that is the states'
        # projective class, whose own verdict speaks
        blocked = {root for root in touched if recs[root]["obstructed"]}
        weighed = [i for i, item in enumerate(self.items)
                   if not {orbit_of[j] for j in item["touch"]} & blocked]
        # the declarations sharing no orbit are weighed apart
        link = {u: u for i in weighed for u in units_of[i]}

        def _root(u):
            while link[u] != u:
                u = link[u]
            return u

        for i in weighed:
            units = units_of[i]
            for u in units[1:]:
                a, b = _root(units[0]), _root(u)
                if a != b:
                    link[max(a, b)] = min(a, b)
        components = {}
        for u in sorted(link):
            components.setdefault(_root(u), []).append(u)
        for units in components.values():
            items = [i for i in weighed if set(units_of[i]) & set(units)]
            order = sorted(units, key=lambda u: (len(recs[u]["options"]), u))
            size = 1
            for u in order:
                size *= max(len(recs[u]["options"]), 1)
            if size > self.max_candidates:
                raise _TooMany(size, tuple(sorted(o for u in order for o in recs[u]["orbitals"])),
                               max(recs[u]["m"] for u in order))
            parts = {(i, u): self._contributions(i, recs[u], H)
                     for i in items for u in units_of[i]}
            found = False
            for combo in itertools.product(*(range(len(recs[u]["options"])) for u in order)):
                pick = dict(zip(order, combo))
                ok = True
                for i in items:
                    content = {}
                    for u in units_of[i]:
                        chi = recs[u]["options"][pick[u]]
                        for h, here in parts[(i, u)].items():
                            v = sum((chi[k] * theta for k, theta in here), 0j)
                            if doubled[u]:
                                v = v + np.conj(v)
                            content[h] = content.get(h, 0j) + v
                    if not self._holds(i, content, H):
                        ok = False
                        break
                if ok:
                    found = True
                    break
            if not found:
                return False
        return True

# --------------------------------------------------------------------------
# the representations an orbit of sectors carries, built by induction
# --------------------------------------------------------------------------

def atom_sites(reading, layout):
    """The site of each index of ``reading``'s orbital factor.

    ``layout`` holds the multiplets, and an index lies at the site of the
    multiplet holding its first orbital.  Returns an array of sites.
    """
    chan_of = np.asarray(layout._chan_of, dtype=int)
    return np.asarray([int(layout._chan_site[int(chan_of[int(orbs[0])])])
                       for orbs in reading.atoms], dtype=int)

def chan_rows(reading, layout):
    """Per multiplet of ``layout``, the indices of the orbital factor it holds.

    ``reading`` is the :class:`SectorReading` whose orbital factor is meant.
    On states read without spin they are the multiplet's states, in order.
    On states that carry a spin, native or declared in pairs, each index is
    listed once, in the order the multiplet's states first reach it.
    """
    if not reading.su2:
        return [tuple(int(a) for a in states) for states in layout._chan_states]
    index = {int(o): a for a, orbs in enumerate(reading.atoms) for o in orbs}
    out = []
    for orbs in layout._chan_orbs:
        rows = []
        for o in orbs:
            if index[int(o)] not in rows:
                rows.append(index[int(o)])
        out.append(tuple(rows))
    return out

def spin_places(reading, states):
    """Where each of ``states`` lies in the orbital factor, and its spin.

    ``reading`` is the :class:`SectorReading` whose orbital factor is meant.
    A native spinful model stores state ``2·orbital + spin``.  A declared
    spinor basis holds each pair's ``(up, down)`` orbitals, and a state's spin
    is its place in its pair.  Returns ``[(index, spin)]``, ``spin`` ``0`` up
    and ``1`` down.
    """
    index = {int(o): a for a, orbs in enumerate(reading.atoms) for o in orbs}
    native = all(len(orbs) == 1 for orbs in reading.atoms)
    out = []
    for x in states:
        if native:
            o, s = divmod(int(x), 2)
            out.append((index[o], int(s)))
        else:
            a = index[int(x)]
            out.append((a, list(reading.atoms[a]).index(int(x))))
    return out

def in_states(reading, basis, nstate):
    r"""A basis written in ``reading``'s orbital factor, written in the states.

    The states carry a spin, native or declared in pairs.  Column ``j`` of
    ``basis`` becomes columns ``2j`` and ``2j + 1``, the same coefficients on
    the up and on the down state.  A basis copy-major in the orbital factor so
    stays copy-major, each copy's states its orbital factor's times the spin.
    ``nstate`` is the model's number of states.  Returns an array of shape
    ``(nstate, 2 · ncols)``, ``ncols`` the columns of ``basis``.
    """
    Q = np.asarray(basis, dtype=complex)
    out = np.zeros((int(nstate), 2 * int(Q.shape[1])), dtype=complex)
    native = all(len(orbs) == 1 for orbs in reading.atoms)
    for a, orbs in enumerate(reading.atoms):
        for s in (0, 1):
            row = 2 * int(orbs[0]) + s if native else int(orbs[s])
            out[row, s::2] = Q[a, :]
    return out

def _snapped(per, sites, tol):
    r"""Copies lying along orbitals within the precision, held exactly along them.

    ``per`` is a sector's basis copy by copy, of shape ``(n, m, d)``: ``n``
    indices of the orbital factor, ``m`` copies of ``d`` states.  ``sites``
    gives the site of each of the ``n`` indices, and ``tol`` is the declared
    tolerance the sector was read at.  Each copy is put wholly on its own
    orbitals, the first made unitary there.  Every later copy is the first
    carried by the matrix between them, kept within each site and made
    unitary, so that every matrix acting on the copies alone acts within
    sites.  Returns an array of the shape of ``per``.  Returns ``None`` where
    a copy does not lie along orbitals of its own, or where the matrix
    carrying the first copy onto another reaches between sites, by more than
    ``tol`` or :data:`.._tolerance._IN_BLOCKS` where that is larger.
    """
    n, m, d = per.shape
    loose = noise_floor(_IN_BLOCKS, tol)
    weight = np.sum(np.abs(per) ** 2, axis=2)
    own = [np.flatnonzero(weight[:, a] > 0.5) for a in range(m)]
    if any(len(o) != d for o in own) or len({int(i) for o in own for i in o}) != m * d:
        return None
    for a in range(m):
        others = np.ones(n, dtype=bool)
        others[own[a]] = False
        if (float(weight[others, a].max(initial=0.0)) > loose
                or float(np.abs(weight[own[a], a] - 1.0).max()) > loose):
            return None
    out = np.zeros_like(per)
    u, _s, vh = np.linalg.svd(per[own[0], 0, :])
    first = u @ vh
    out[own[0], 0, :] = first
    for a in range(1, m):
        # the matrix carrying the first copy's orbitals onto this copy's
        T = per[own[a], a, :] @ per[own[0], 0, :].conj().T
        same = sites[own[a]][:, None] == sites[own[0]][None, :]
        if float(np.abs(T[~same]).max(initial=0.0)) > loose:
            return None
        u, _s, vh = np.linalg.svd(np.where(same, T, 0.0))
        out[own[a], a, :] = (u @ vh) @ first
    return out

def held_shares(reading, sectors, sites, tol, *, on=None):
    r"""Sectors of one copy, each held exactly to its own share of every site.

    ``sectors`` are indices into ``reading.sectors``, each of one copy, and
    ``sites`` gives the site of each index of the orbital factor.  ``tol``,
    the declared tolerance, is not read.  ``on`` holds the indices of the
    orbital factor the multiplets holding the sectors lie on, and the shares
    are read within them; ``None`` reads every index.  At each site the
    sectors' shares are read together, as the eigenspaces of one Hermitian
    matrix that weights each sector's projector by a number of its own, so
    that they are exact and orthogonal.  Each sector's basis is then the
    nearest to its own that lies in its shares.  A basis lying along
    orbitals of its own to rounding is held exactly along them; one turned
    off them by more keeps its turn.  Returns ``{sector: basis}``, or
    ``None`` where a sector's shares hold a number of states other than its
    own.
    """
    bases = {int(t): np.asarray(reading.sectors[int(t)].basis, dtype=complex)
             for t in sectors}
    n = int(next(iter(bases.values())).shape[0])
    held = {t: np.zeros((n, n), dtype=complex) for t in bases}
    inside = (np.ones(n, dtype=bool) if on is None
              else np.isin(np.arange(n), np.asarray(sorted(int(a) for a in on), dtype=int)))
    for s in sorted({int(x) for x in sites}):
        idx = np.flatnonzero((np.asarray(sites, dtype=int) == s) & inside)
        if not idx.size:
            continue
        X = sum((i + 1) * (Q[idx, :] @ Q[idx, :].conj().T)
                for i, Q in enumerate(bases.values()))
        w, V = np.linalg.eigh(0.5 * (X + X.conj().T))
        for i, t in enumerate(bases):
            near = V[:, np.abs(w - (i + 1)) < 0.5]
            held[t][np.ix_(idx, idx)] = near @ near.conj().T
    out = {}
    for t, Q in bases.items():
        if int(round(float(np.real(np.trace(held[t]))))) != int(Q.shape[1]):
            return None
        u, _s, vh = np.linalg.svd(held[t] @ Q, full_matrices=False)
        B = u @ vh
        own = np.flatnonzero(np.sum(np.abs(B) ** 2, axis=1) > 0.5)
        if (len(own) == int(B.shape[1])
                and float(np.abs(np.delete(B, own, axis=0)).max(initial=0.0)) <= _IN_BLOCKS):
            # along orbitals of its own to rounding: held exactly along them
            u, _s, vh = np.linalg.svd(B[own, :])
            B = np.zeros_like(B)
            B[own, :] = u @ vh
        out[t] = B
    return out

def aligned(sector, tol=None, sites=None):
    r"""A sector's basis, its copies put along the orbitals as written.

    ``sector`` is a :class:`Sector`.  Its copies are read orbital by orbital,
    in order: at each orbital in turn, the combinations of copies carrying
    its weight and not already chosen are added, the largest weight first,
    until there are ``copies`` of them.  So copies lying along orbitals of
    their own are each one copy as written, the one holding the lowest
    orbital first.  Each copy after the first is put in the phase that makes
    the largest entry of the matrix carrying the first copy onto it real and
    positive.  ``tol`` is the declared tolerance the sector was read at, and
    ``sites`` the site of each index of the orbital factor.  Where both are
    given, copies lying along the orbitals within ``tol`` are held exactly
    along them (:func:`_snapped`).  Returns the basis, of the shape of
    ``sector.basis``; ``sector.basis`` itself where the sector holds one copy,
    or where the orbitals' weights give fewer than ``copies`` combinations.
    """
    m, d = int(sector.copies), int(sector.dim)
    Q = np.asarray(sector.basis, dtype=complex)
    if m < 2:
        return Q
    n = int(Q.shape[0])
    per = Q.reshape(n, m, d)
    zero = noise_floor(ZERO, tol)
    chosen = np.zeros((m, 0), dtype=complex)
    for i in range(n):
        U = per[i]
        # orbital i's weight on the copies, outside the copies already taken
        G = U.conj() @ U.T
        if chosen.shape[1]:
            P = np.eye(m) - chosen @ chosen.conj().T
            G = P @ G @ P
        w, V = np.linalg.eigh(0.5 * (G + G.conj().T))
        for j in range(m - 1, -1, -1):
            if float(w[j]) <= zero or chosen.shape[1] == m:
                break
            chosen = np.concatenate([chosen, V[:, j:j + 1]], axis=1)
        if chosen.shape[1] == m:
            break
    if chosen.shape[1] != m:
        return Q
    per = np.einsum("iaw,ab->ibw", per, chosen)
    first = per[:, 0, :]
    for a in range(1, m):
        z, _where = _matrices._principal_entry(per[:, a, :] @ first.conj().T)
        if abs(z) > 0.0:
            per[:, a, :] = per[:, a, :] * (np.conj(z) / abs(z))
    if tol is not None and sites is not None:
        held = _snapped(per, np.asarray(sites, dtype=int), tol)
        if held is not None:
            per = held
    return per.reshape(n, m * d)

def _first_sector(reading, sectors, bases=None):
    r"""The sector of an orbit the others are carried from.

    ``sectors`` are the orbit's sectors, indices into ``reading.sectors``.
    The first is read off the orbitals as written: the sector whose
    projector holds the most of the lowest orbital the orbit holds, then of
    the next, and so on.  Between sectors that hold every orbital alike, as
    a square's pair written as ``(p_x ± p_y)/√2`` does, the one whose
    projector joins the orbitals with the largest real part, pair by pair,
    comes first: the sum before the difference.  Where all still tie, the
    lowest index does.  ``bases``, where given, maps a sector to its basis
    held to its shares (:func:`held_shares`), read in place of the
    reading's own.  Returns the sector's index.
    """
    live = sorted(int(t) for t in sectors)
    proj = {t: np.asarray(bases[t] if bases is not None and t in bases
                          else reading.sectors[t].basis, dtype=complex) for t in live}
    proj = {t: Q @ Q.conj().T for t, Q in proj.items()}
    held = np.flatnonzero(sum(np.real(np.diag(P)) for P in proj.values()) > ZERO)
    keys = ([(int(i), int(i)) for i in held]
            + [(int(i), int(j)) for a, i in enumerate(held) for j in held[a + 1:]])
    for i, j in keys:
        if len(live) < 2:
            break
        top = max(float(np.real(proj[t][i, j])) for t in live)
        live = [t for t in live if float(np.real(proj[t][i, j])) >= top - ZERO]
    return live[0]

def _pure_reversal(sym):
    """Whether ``sym`` is time reversal alone: no rotation and no translation."""
    t = np.asarray(sym.translation, dtype=float)
    return (bool(sym.time_reversal)
            and np.array_equal(np.rint(np.asarray(sym.rotation)).astype(int),
                               np.eye(3, dtype=int))
            and float(np.abs(t - np.rint(t)).max()) <= _LATTICE_TOL)

class _InducedCandidate:
    """One representation of an orbit of sectors, its matrices built when asked for.

    ``orbit`` is the :class:`InducedOrbit`.  ``combo`` lists the irreducible
    representations of the stabilizer the copies carry, as indices into the
    orbit's ``irreps``: one, one-dimensional, where each sector holds one
    copy.  ``character`` is their character on the stabilizer's unitary
    operations, by position.  ``X`` is the matrix the copies take under the
    antiunitary operation fixing the orbit's first sector, ``None`` where
    there is none.  Asked by key, ``"W"`` gives the matrices on the orbit's
    states (:meth:`InducedOrbit.matrices`), built the first time;
    ``"contents"`` what each multiplet holds (:meth:`InducedOrbit.contents`);
    ``"character"`` the character keyed by operation index.
    """

    def __init__(self, orbit, combo, X, character):
        self.orbit, self.combo, self.X = orbit, tuple(int(i) for i in combo), X
        self.character = character
        self._W, self._contents = None, None

    def __getitem__(self, key):
        if key == "W":
            if self._W is None:
                self._W = self.orbit.matrices(self)
            return self._W
        if key == "contents":
            if self._contents is None:
                self._contents = self.orbit.contents(self)
            return self._contents
        if key == "character":
            return {int(self.orbit.syms[k].ind): v for k, v in self.character.items()}
        raise KeyError(key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def release(self):
        """Drop the matrices built, keeping what was read off them."""
        self._W = None

@dataclass(frozen=True)
class _OrbitRecord:
    """An orbit of sectors built by induction, and the representations the hoppings leave.

    Attributes
    ----------
    multiplets : tuple of int
        The multiplets the orbit lies on.
    orbitals : tuple of int
        The orbitals of its sectors, in order.
    lead : int
        The multiplet the orbit is read at: the reference multiplet where the
        orbit holds it, its first otherwise.
    site : int
        The site of that multiplet.
    dim : int
        How many identical copies each sector holds; one for orbits read
        together.
    n_candidates : int
        How many representations of the orbit were counted.
    n_admitted : int
        How many of them extend to the antiunitary operations and are kept.
    states : tuple of int
        The states of the orbit's multiplets, in order; every candidate's
        matrices are written on them.
    candidates : tuple
        The admitted representations, in order, each giving by key its
        matrices, what it puts at each multiplet and its own character.
    induced : InducedOrbit or SharedOrbits
        The orbit itself.
    """

    multiplets: tuple
    orbitals: tuple
    lead: int
    site: int
    dim: int
    n_candidates: int
    n_admitted: int
    states: tuple
    candidates: tuple
    induced: object


class InducedOrbit:
    r"""One orbit of sectors, and every representation the hoppings leave on it.

    Every representation is induced from the stabilizer ``K`` of the orbit's
    first sector.  On each copy's states an operation carries the factor
    ``w`` the sectors fix (:func:`sector_action`), and on the copies a
    representation ``b`` of ``K``, which ``H`` leaves open.  The candidates
    are the sums ``b`` of irreducible representations of ``K`` for which
    ``b ⊗ w`` carries the orbital part's factor system
    (:meth:`_orbital_table`): the table's spin signs on states read without
    spin, and trivial on states that carry one.  Where ``K`` holds an
    antiunitary operation, only the sums that extend to it are kept
    (:meth:`_extension`).  On an operation ``g`` and a sector ``t`` the
    matrix is ``F_{gt} (b(k) ⊗ w(k)) F_t^†``, conjugated where ``r_{gt}`` is
    antiunitary.  Here ``gt`` is the sector ``g`` carries ``t`` onto.
    ``r_t`` is the coset representative carrying the first sector onto
    ``t``.  ``F_t`` is the frame of ``t``, its copies along the orbitals
    (:func:`aligned`), or held to its shares where ``bases`` is given, and
    carried from the first sector's by ``r_t``.  ``k`` is
    ``r_{gt}⁻¹ g r_t``.  At a declared tolerance ``w`` is first replaced by
    the exact representation of ``K`` nearest it, where there is one.  The
    candidates are counted before anything is built, and what each multiplet
    holds under one is read without building a matrix (:meth:`contents`).

    Parameters
    ----------
    reading : SectorReading
        The sectors.
    action : SectorAction
        What the group's operations do to them.
    group : .._spacegroup.GroupData
        The group.
    layout : ..symmetry.SiteMultiplets
        The multiplets the representation is written on.
    kept : iterable of int
        The operations represented, unitary and antiunitary, by position in
        the group's list of operations.
    cperm : mapping
        Per position, where the operation carries each multiplet.
    sectors : sequence of int
        The orbit's sectors, indices into ``reading.sectors``.  The first is
        the one its orbitals put first (:func:`_first_sector`).
    reference : int, default None
        The multiplet the orbit is read at, where the orbit holds it.
    tol : float, default None
        The declared tolerance.
    max_candidates : int, default 2 ** 20
        The most candidates an orbit of several copies may have.
    shared : sequence of int, default None
        The multiplets of the orbits read together with this one
        (:class:`SharedOrbits`), on which it is written, holding its share of
        each.  ``None`` reads the multiplets the orbit lies on.
    bases : dict, default None
        Each sector's basis held to its shares (:func:`held_shares`), read
        in place of the reading's own.

    Raises
    ------
    _TooMany
        Where an orbit of several copies has more than ``max_candidates``
        candidates.
    _Unread
        Where the sectors do not read the orbit: an operation of ``kept``
        with no reading on them, factors that do not compose, or a multiplet
        holding part of the orbit or splitting its copies.
    """

    def __init__(self, reading, action, group, layout, kept, cperm, sectors, *,
                 reference=None, tol=None, max_candidates=2 ** 20, shared=None, bases=None):
        self.reading, self.group, self.layout, self.cperm = reading, group, layout, cperm
        self.syms = group._symmetries
        self.prod, _tdiff, self.spf = group._get_product_table(get_diff=True)
        self.ident = _matrices._identity_position(group)
        self.tol = tol
        self.kept = sorted(int(p) for p in kept)
        pos_of = {int(s.ind): p for p, s in enumerate(self.syms)}
        self.images = {pos_of[int(i)]: tuple(int(j) for j in got)
                       for i, got in action.images.items() if int(i) in pos_of}
        self.factor = {(pos_of[int(i)], int(j)): np.asarray(w, dtype=complex)
                       for (i, j), w in action.factors.items() if int(i) in pos_of}
        if any(p not in self.images for p in self.kept):
            raise _Unread
        self.unitary = [p for p in self.kept if not self.syms[p].time_reversal]
        self.anti = [p for p in self.kept if self.syms[p].time_reversal]
        prod, ident = self.prod, self.ident
        self.inv = {p: next(q for q in self.kept if int(prod[p, q]) == ident)
                    for p in self.kept}
        j0 = _first_sector(reading, sectors, bases)
        self.sectors = (j0,) + tuple(int(t) for t in sectors if int(t) != j0)
        first = reading.sectors[j0]
        self.m, self.d = int(first.copies), int(first.dim)
        # the coset representatives, a unitary one wherever one reaches
        rep = {j0: ident}
        for ops in (self.unitary, self.kept):
            frontier = sorted(rep)
            while frontier:
                a = frontier.pop(0)
                for h in ops:
                    b = self.images[h][a]
                    if b not in rep:
                        rep[b] = int(prod[h, rep[a]])
                        frontier.append(b)
        if set(rep) != set(self.sectors):
            raise _Unread
        self.rep = rep
        self.K = [g for g in self.kept if self.images[g][j0] == j0]
        self.Ku = [g for g in self.K if not self.syms[g].time_reversal]
        antis = [g for g in self.K if self.syms[g].time_reversal]
        pure = [g for g in antis if _pure_reversal(self.syms[g])]
        self.a_pos = (pure or antis or [None])[0]
        # the factor w every operation of K fixes on the first sector's copies
        self.w = {}
        for g in self.K:
            got = self.factor.get((g, j0))
            if got is None:
                raise _Unread
            self.w[g] = got
        if tol is not None:
            # at a declared precision the factors compose only within it: they
            # are replaced by the exact representation of the stabilizer
            # nearest them, where there is one, so that what is induced from
            # them is a representation of the group to rounding, as a
            # projection onto the symmetric part needs
            settled = _matrices._nearest_representation(self.w, self.K, self.syms, self.prod)
            if settled is not None:
                self.w = {int(g): settled[int(g)] for g in self.K}
        # the frames: each sector's copies along the orbitals, carried from the first
        self.frames = {}
        sites = atom_sites(reading, layout)
        for t, r in rep.items():
            wt = np.eye(self.d, dtype=complex) if t == j0 else self.factor.get((r, j0))
            if wt is None:
                raise _Unread
            basis = (bases[t] if bases is not None and t in bases
                     else aligned(reading.sectors[t], tol, sites=sites))
            self.frames[t] = basis @ np.kron(np.eye(self.m), wt)
        # each sector after the first is put in the phase that makes the
        # largest entry of the matrix carrying the first sector's states onto
        # its own real and positive, so that the orbitals are written as one
        # another's images under the coset representatives, as the tables
        # write the sites of a Wyckoff orbit
        first = self.frames[j0]
        for t, r in rep.items():
            if t == j0:
                continue
            carry = self.frames[t] @ (first.T if self.syms[r].time_reversal
                                      else first.conj().T)
            z, _where = _matrices._principal_entry(carry)
            if abs(z) > 0.0:
                self.frames[t] = self.frames[t] * (np.conj(z) / abs(z))
        # the factor system b carries on K: the orbital part's own over the one
        # w carries, a root of unity read exactly (:func:`_composed`)
        self.nu = {}
        for x in self.K:
            for y in self.K:
                xy = int(prod[x, y])
                if xy not in self.w:
                    raise _Unread
                rho = _composed(self.w[x], self.w[y], self.w[xy],
                                after_antiunitary=bool(self.syms[x].time_reversal), tol=tol)
                self.nu[(x, y)] = self._orbital_table(x, y) / rho
        irreps = _matrices._projective_irreps(self.Ku, prod, self.nu)
        if irreps is None:
            raise _Unread
        # each matrix made unitary: read off eigenvectors of the regular
        # representation, the irreps are unitary only to rounding, which the
        # induced matrices and their products would carry
        self.irreps = [{k: _matrices._unitarized(r[k]) for k in r} for r in irreps]
        dims = [int(np.asarray(r[self.Ku[0]]).shape[0]) for r in self.irreps]
        self.orbitals = tuple(sorted(int(o) for t in self.sectors
                                     for a in reading.sectors[t].atoms
                                     for o in reading.atoms[int(a)]))
        # counted before anything is built: every sum of the stabilizer's
        # irreducible representations the copies could carry, its
        # one-dimensional ones where each sector holds one copy
        self.n_candidates = _matrices._count_dimension_sums(dims, self.m)
        if self.m > 1 and self.n_candidates > int(max_candidates):
            raise _TooMany(self.n_candidates, self.orbitals, self.m)
        self._members(reference, shared)
        self.candidates = []
        chars = [{k: complex(np.trace(r[k])) for k in self.Ku} for r in self.irreps]
        for combo in _matrices._dimension_sums(dims, self.m):
            X = None
            if self.a_pos is not None:
                X = self._extension(combo)
                if X is None:
                    continue
            character = {k: sum((chars[i][k] for i in combo), 0j) for k in self.Ku}
            self.candidates.append(_InducedCandidate(self, combo, X, character))

    # ---- the orbital part's own factor system ------------------------------------
    def _orbital_table(self, x, y):
        """The factor ``τ`` with ``O(x) O(y) = τ O(xy)`` on the orbital part.

        ``x`` and ``y`` are positions of operations.  On states read without
        spin the orbital part is the whole matrix, and ``τ`` is the group
        table's spin sign for the pair.  On states that carry a spin every
        matrix is ``O ⊗ S(g)``, and ``τ`` is that sign over the factor
        ``λ`` of the spin lifts, ``S(x) S(y) = λ S(xy)``, with ``S(y)*`` in
        place of ``S(y)`` after an antiunitary ``x``.  There it returns ``1``
        where the two agree within :data:`.._tolerance.CHARACTER`, and
        raises :class:`_Unread` where they do not or where ``S(x) S(y)`` is
        not a multiple of ``S(xy)``.
        """
        if not self.reading.su2:
            return complex(self.spf[x, y])
        Sx, Sy = (_matrices._lift_of(self.syms[int(p)], True) for p in (x, y))
        lam, res = _matrices._proportionality(
            Sx @ (Sy.conj() if self.syms[int(x)].time_reversal else Sy),
            _matrices._lift_of(self.syms[int(self.prod[x, y])], True))
        if (res > noise_floor(ASSEMBLED, self.tol) or abs(lam) <= 0.0
                or abs(complex(self.spf[x, y]) / lam - 1.0) > CHARACTER):
            raise _Unread
        return 1.0 + 0.0j

    # ---- the multiplets the orbit lies on -----------------------------------------
    def _members(self, reference, shared=None):
        layout = self.layout
        loose = noise_floor(_IN_BLOCKS, self.tol) * 10.0
        # each multiplet's indices in the orbital factor the frames are written
        # in: its states, or on states carrying a spin each orbital (or pair) once
        rows_of = chan_rows(self.reading, layout)
        if shared is not None:
            # read together with the other orbits its multiplets hold shares of
            members = [int(c) for c in shared]
        else:
            weight = np.zeros(int(next(iter(self.frames.values())).shape[0]))
            for F in self.frames.values():
                weight = weight + np.sum(np.abs(F) ** 2, axis=1)
            members = []
            for c, held in enumerate(rows_of):
                here = weight[np.asarray(held, dtype=int)]
                if float(here.max()) <= loose:
                    continue
                if float(np.abs(here - 1.0).max()) > loose:
                    raise _Unread            # a multiplet the orbit holds part of
                members.append(int(c))
            if sum(len(rows_of[c]) for c in members) != self.m * self.d * len(self.sectors):
                raise _Unread
        self.members = tuple(members)
        self.states = tuple(int(a) for c in members for a in layout._chan_states[c])
        self.local = {a: i for i, a in enumerate(self.states)}
        # the orbit's indices in the orbital factor, and on states carrying a
        # spin where each state lies there and its spin: the matrices are built
        # on the orbital factor and carry each operation's own lift on the spin
        self.atoms = tuple(int(a) for c in members for a in rows_of[c])
        self.alocal = {a: i for i, a in enumerate(self.atoms)}
        self.places = (None if not self.reading.su2 else
                       [(self.alocal[a], s) for a, s in spin_places(self.reading, self.states)])
        rows = np.asarray(self.atoms, dtype=int)
        self.loc = {t: F[rows, :] for t, F in self.frames.items()}
        self.lead = (int(reference) if reference is not None and int(reference) in members
                     else int(members[0]))
        self.site = int(layout._chan_site[self.lead])
        # what each multiplet holds under an operation carrying it onto itself
        # (``cperm``): per sector of its own the operation fixes, the
        # stabilizer's element there and the trace of the multiplet's share of
        # the frame against w
        m, d = self.m, self.d
        self.theta = {}
        for c in members:
            idx = [self.alocal[int(a)] for a in rows_of[c]]
            on = [t for t in self.sectors if float(np.abs(self.loc[t][idx, :]).max()) > loose]
            per = {}
            for h in self.unitary:
                if int(self.cperm[h][c]) != c:
                    continue
                here = []
                for t in on:
                    if self.images[h][t] != t:
                        continue
                    r = self.rep[t]
                    k = int(self.prod[self.inv[r], int(self.prod[h, r])])
                    F = self.loc[t][idx, :]
                    G = F.conj().T @ F
                    block = G[:d, :d]
                    if float(np.abs(G - np.kron(np.eye(m), block)).max()) > loose:
                        raise _Unread    # the multiplet splits the copies
                    conj = bool(self.syms[r].time_reversal)
                    w = self.w[k].conj() if conj else self.w[k]
                    here.append((k, complex(np.trace(block @ w)), conj))
                per[h] = here
            self.theta[c] = per

    # ---- the stabilizer's representations -----------------------------------------
    def _b(self, combo):
        m = self.m
        out = {}
        for k in self.Ku:
            M = np.zeros((m, m), dtype=complex)
            at = 0
            for i in combo:
                blk = np.asarray(self.irreps[i][k])
                M[at:at + blk.shape[0], at:at + blk.shape[0]] = blk
                at += blk.shape[0]
            out[k] = M
        return out

    def _extension(self, combo, seed=17):
        r"""The copies' matrix ``X`` under the antiunitary operation, or ``None``.

        ``combo`` names the candidate's irreducible representations of the
        stabilizer, which make ``b`` (:meth:`_b`), and ``a`` is the orbit's
        antiunitary operation fixing the first sector, time reversal itself
        where the stabilizer holds it.  ``X`` is unitary and solves
        ``X b(u)* = ν(a, u)/ν(u', a) · b(u') X`` for every unitary ``u`` of the
        stabilizer, with ``u' a = a u``, and ``X X* = ν(a, a) b(a²)``, ``ν``
        the orbit's factor system.  The solution nearest the identity is tried
        first, where the identity reaches their space, then four drawn from it
        with the seed ``seed``, each made unitary.  Returns the first that
        meets both equations within :data:`.._tolerance.ASSEMBLED`, or the
        declared tolerance where larger; ``None`` where none does.
        """
        a, prod = self.a_pos, self.prod
        b = self._b(combo)
        Ku, m = self.Ku, self.m
        square = int(prod[a, a])
        if square not in b:
            return None
        eye = np.eye(m)
        conj_of = {}
        for u in Ku:
            target = int(prod[a, u])
            mate = [v for v in Ku if int(prod[v, a]) == target]
            if len(mate) != 1:
                return None
            conj_of[u] = mate[0]
        ratio = {u: self.nu[(a, u)] / self.nu[(conj_of[u], a)] for u in Ku}
        rows = [np.kron(b[u].conj().T, eye) - ratio[u] * np.kron(eye, b[conj_of[u]])
                for u in Ku]
        swap = np.zeros((m * m, m * m))
        for i in range(m):
            for j in range(m):
                swap[i + j * m, j + i * m] = 1.0
        squared = self.nu[(a, a)] * b[square]
        rows.append(np.eye(m * m) - np.kron(squared.conj().T, eye) @ swap)
        _u, sv, vh = np.linalg.svd(np.vstack(rows))
        top = max(1.0, float(sv[0])) if sv.size else 1.0
        zero, close = noise_floor(ZERO, self.tol), noise_floor(ASSEMBLED, self.tol)
        null = [vh[i].conj() for i in range(len(sv)) if float(sv[i]) <= zero * top]
        null += [vh[i].conj() for i in range(len(sv), m * m)]
        if not null:
            return None
        N = np.asarray(null).T

        def _closes(X):
            return (all(float(np.abs(X @ b[u].conj() - ratio[u] * b[conj_of[u]] @ X).max())
                        <= close for u in Ku)
                    and float(np.abs(X @ X.conj() - squared).max()) <= close)

        # the one nearest the identity, where the identity reaches the space
        y = N @ (N.conj().T @ eye.reshape(-1, order="F"))
        tries = [y] if float(np.abs(y).max()) > zero else []
        rng = np.random.default_rng(int(seed))
        tries += [N @ (rng.normal(size=N.shape[1]) + 1j * rng.normal(size=N.shape[1]))
                  for _ in range(4)]
        for y in tries:
            w, _s, wh = np.linalg.svd(y.reshape(m, m, order="F"))
            X = w @ wh
            if _closes(X):
                return X
        return None

    def _beta(self, cand):
        """The matrices ``b(k) ⊗ w(k)`` of the candidate ``cand`` on the stabilizer.

        Returns ``{k: matrix}`` over every operation ``k`` of the stabilizer,
        by position.  ``a`` is the orbit's antiunitary operation fixing the
        first sector, and ``ν`` the orbit's factor system.  An antiunitary
        ``k = u a`` takes ``b(u) X / ν(u, a)`` in place of ``b(k)``, ``X`` the
        candidate's matrix under ``a``.  Raises
        :class:`_Unread` where ``k a⁻¹`` is not a unitary operation of the
        stabilizer.
        """
        b = self._b(cand.combo)
        out = {k: np.kron(b[k], self.w[k]) for k in self.Ku}
        a = self.a_pos
        if a is not None:
            inv_a = self.inv[a]
            for k in self.K:
                if not self.syms[k].time_reversal:
                    continue
                u = int(self.prod[k, inv_a])
                if u not in b or int(self.prod[u, a]) != k:
                    raise _Unread
                out[k] = np.kron(b[u] @ cand.X / self.nu[(u, a)], self.w[k])
        return out

    # ---- what a candidate gives -----------------------------------------------------
    def contents(self, cand):
        """What each multiplet holds under the candidate ``cand``, as a character.

        Returns ``{multiplet: {ind: χ}}`` over the unitary operations carrying
        the multiplet onto itself, read off the induced character without
        building a matrix.  ``χ`` sums, over the sectors on the multiplet that
        the operation fixes, the candidate's character at the stabilizer's
        element there times the trace of the multiplet's share of one copy's
        frame against ``w``.  The character is conjugated where the sector's
        coset representative is antiunitary.
        """
        out = {}
        for c, per in self.theta.items():
            out[c] = {int(self.syms[h].ind): sum(
                ((np.conj(cand.character[k]) if conj else cand.character[k]) * th
                 for k, th, conj in here), 0j) for h, here in per.items()}
        return out

    def matrices(self, cand):
        """The candidate ``cand``'s matrices on the orbit's states.

        Returns ``{position: matrix}`` over the operations represented.  Each
        matrix carries every multiplet onto its image and holds nothing else.
        On states that carry a spin the matrix built on the orbital factor is
        multiplied by the operation's own spin lift, ``O ⊗ S(g)``.  The
        antiunitary operations' matrices are put in one phase, the largest
        entry of the first one's real and positive.
        """
        beta = self._beta(cand)
        n = len(self.atoms)
        layout = self.layout
        W = {}
        for p in self.kept:
            M = np.zeros((n, n), dtype=complex)
            au = bool(self.syms[p].time_reversal)
            for t in self.sectors:
                s = self.images[p][t]
                k = int(self.prod[self.inv[self.rep[s]], int(self.prod[p, self.rep[t]])])
                B = beta.get(k)
                if B is None:
                    raise _Unread
                if self.syms[self.rep[s]].time_reversal:
                    B = B.conj()
                M += self.loc[s] @ B @ (self.loc[t].T if au else self.loc[t].conj().T)
            if self.places is not None:
                at = [a for a, _s in self.places]
                spin = [s for _a, s in self.places]
                M = M[np.ix_(at, at)] * _matrices._lift_of(self.syms[p], True)[np.ix_(spin, spin)]
            kept = np.zeros_like(M)
            for c in self.members:
                rows = [self.local[int(a)] for a in layout._chan_states[int(self.cperm[p][c])]]
                cols = [self.local[int(a)] for a in layout._chan_states[int(c)]]
                kept[np.ix_(rows, cols)] = M[np.ix_(rows, cols)]
            W[p] = kept
        # the antiunitary operations in one phase: the largest entry of the
        # first one's matrix real and positive
        if self.anti:
            z, _where = _matrices._principal_entry(W[self.anti[0]])
            if abs(z) > 0.0:
                for p in self.anti:
                    W[p] = W[p] * (np.conj(z) / abs(z))
        return W

    def check(self, W, bases):
        """How far ``W`` misses the group's table and the solution spaces.

        ``W`` maps each operation represented, by position, to its matrix on
        the orbit's states, and ``bases`` maps each operation's index to its
        solution space.  Returns the larger of the residual of the group's
        multiplication table and, per operation, the distance of its matrix
        from the solution space's block on the orbit's states, relative to the
        larger of one and the matrix's largest entry.  Returns infinity where
        an operation has no solution space or none on those states.
        """
        worst = _matrices._product_table_residual(W, self.kept, self.syms, self.prod, self.spf)
        rows = np.asarray(self.states, dtype=int)
        zero = noise_floor(ZERO, self.tol)
        for p in self.kept:
            basis = bases.get(int(self.syms[p].ind))
            if basis is None or not len(np.asarray(basis)):
                return float("inf")
            space, _scale = _matrices._block_space(np.asarray(basis, dtype=complex),
                                                   rows, rows, tol=zero)
            if not space:
                return float("inf")
            approx = sum(complex(np.vdot(M, W[p])) / complex(np.vdot(M, M)) * np.asarray(M)
                         for M in space)
            worst = max(worst, float(np.abs(W[p] - approx).max())
                        / max(1.0, float(np.abs(W[p]).max())))
        return worst

    def sector_frames(self):
        """Each sector's frame on the orbit's states, the first sector first.

        Returns ``[(F, copies, dim, first, rep)]``, one entry per sector:
        ``F`` the frame, copy-major; ``copies`` and ``dim`` the sector's;
        ``first`` the place of the orbit's first sector in the list, here
        ``0``; ``rep`` the position of the coset representative carrying the
        first sector onto this one.  On states that carry a spin each copy's
        states are its orbital factor's times the spin, and ``dim`` counts
        them.
        """
        if self.places is None:
            return [(self.loc[t], self.m, self.d, 0, int(self.rep[t])) for t in self.sectors]
        at = np.asarray([a for a, _s in self.places], dtype=int)
        spin = np.asarray([s for _a, s in self.places], dtype=int)
        out = []
        for t in self.sectors:
            F = self.loc[t][at, :]
            frame = np.zeros((len(at), 2 * F.shape[1]), dtype=complex)
            for s in (0, 1):
                frame[spin == s, s::2] = F[spin == s, :]
            out.append((frame, self.m, 2 * self.d, 0, int(self.rep[t])))
        return out

    def record(self):
        """The orbit's :class:`_OrbitRecord`, its admitted candidates in order."""
        return _OrbitRecord(multiplets=self.members, orbitals=self.orbitals, lead=self.lead,
                            site=self.site, dim=int(self.m),
                            n_candidates=int(self.n_candidates),
                            n_admitted=len(self.candidates), states=self.states,
                            candidates=tuple(self.candidates), induced=self)

class _SharedCandidate:
    """One representation of orbits read together, one candidate of each.

    ``shared`` is the :class:`SharedOrbits`, and ``picks`` holds one
    candidate of each of its orbits.  Asked by key, ``"W"`` gives the sum of
    their matrices, built the first time, the picks' own then dropped;
    ``"contents"`` and ``"character"`` the sums of theirs; ``"parts"`` each
    pick's ``"contents"``, orbit by orbit.
    """

    def __init__(self, shared, picks):
        self.shared, self.picks = shared, tuple(picks)
        self._W, self._contents = None, None

    def __getitem__(self, key):
        if key == "W":
            if self._W is None:
                parts = [pick["W"] for pick in self.picks]
                self._W = {p: sum(W[p] for W in parts) for p in parts[0]}
                for pick in self.picks:
                    pick.release()
            return self._W
        if key == "contents":
            if self._contents is None:
                out = {}
                for pick in self.picks:
                    for c, chi in pick["contents"].items():
                        row = out.setdefault(int(c), {})
                        for ind, v in chi.items():
                            row[int(ind)] = row.get(int(ind), 0.0) + complex(v)
                self._contents = out
            return self._contents
        if key == "character":
            out = {}
            for pick in self.picks:
                for ind, v in pick["character"].items():
                    out[int(ind)] = out.get(int(ind), 0.0) + complex(v)
            return out
        if key == "parts":
            return tuple(pick["contents"] for pick in self.picks)
        raise KeyError(key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def release(self):
        """Drop the matrices built, keeping what was read off them."""
        self._W = None
        for pick in self.picks:
            pick.release()

class SharedOrbits:
    r"""Orbits of sectors that one multiplet holds sectors of, read together.

    A multiplet holds sectors of several orbits where two orbitals an
    on-site hopping joins make a bonding and an antibonding state that no
    operation exchanges, or where orbitals nothing couples are stated as one
    multiplet.  ``orbits`` are the orbits' :class:`InducedOrbit`, each
    written on the multiplets they share.  What the hoppings leave on each
    orbit is its own, so a candidate is one candidate of each orbit
    (:class:`_SharedCandidate`).  Its matrices are their sum, and what each
    multiplet holds is the sum of the orbits' shares.  Each candidate also
    gives what each orbit puts at each multiplet (``"parts"``), so that a
    labeling reads the orbits one by one.  The candidates are counted before
    any is built, and more than ``max_candidates`` raise :class:`_TooMany`.
    """

    def __init__(self, orbits, *, max_candidates=2 ** 20):
        self.orbits = tuple(orbits)
        first = self.orbits[0]
        self.members, self.states = first.members, first.states
        self.lead, self.site = first.lead, first.site
        self.orbitals = tuple(sorted({int(o) for orbit in self.orbits for o in orbit.orbitals}))
        self.n_candidates = 1
        for orbit in self.orbits:
            self.n_candidates *= int(orbit.n_candidates)
        if self.n_candidates > int(max_candidates):
            raise _TooMany(self.n_candidates, self.orbitals, 1, sets=len(self.orbits))
        self.candidates = [_SharedCandidate(self, picks) for picks in
                           itertools.product(*(orbit.candidates for orbit in self.orbits))]

    def check(self, W, bases):
        """How far ``W`` misses the group's table and the solution spaces ``bases``.

        The orbits share their states, so the first orbit's
        :meth:`InducedOrbit.check` of ``W`` and ``bases`` is returned.
        """
        return self.orbits[0].check(W, bases)

    def sector_frames(self):
        """Every orbit's sector frames on the states they share, orbit after orbit.

        Returns ``[(F, copies, dim, first, rep)]`` as
        :meth:`InducedOrbit.sector_frames` does, each ``first`` the place of
        its own orbit's first sector in the whole list.
        """
        out = []
        for orbit in self.orbits:
            at = len(out)
            out.extend((F, m, d, at + first, rep)
                       for F, m, d, first, rep in orbit.sector_frames())
        return out

    def record(self):
        """The orbits' :class:`_OrbitRecord`, every choice of candidates in order."""
        return _OrbitRecord(multiplets=self.members, orbitals=self.orbitals, lead=self.lead,
                            site=self.site, dim=1, n_candidates=int(self.n_candidates),
                            n_admitted=len(self.candidates), states=self.states,
                            candidates=tuple(self.candidates), induced=self)

def held_sectors(reading, layout, *, tol=None):
    """Per multiplet of ``layout``, the sectors of ``reading`` its states lie on.

    ``tol`` is the declared tolerance the reading was made at.  A multiplet
    lies on a sector where the sector's weight at one of its indices passes
    ten times the larger of :data:`.._tolerance._IN_BLOCKS` and ``tol``.
    Returns a list of frozensets of sector indices, one per multiplet.
    """
    loose = noise_floor(_IN_BLOCKS, tol) * 10.0
    weights = [np.sum(np.abs(np.asarray(s.basis)) ** 2, axis=1) for s in reading.sectors]
    return [frozenset(j for j, w in enumerate(weights)
                      if float(w[np.asarray(rows, dtype=int)].max()) > loose)
            for rows in chan_rows(reading, layout)]

def held_on(reading, layout, held, sectors):
    """The indices the multiplets holding any of ``sectors`` lie on.

    ``held`` is :func:`held_sectors`'s reading of ``layout``, and the indices
    are those of ``reading``'s orbital factor.  Returns a frozenset.
    """
    index = {int(o): a for a, orbs in enumerate(reading.atoms) for o in orbs}
    mine = set(int(t) for t in sectors)
    return frozenset(index[int(o)] for c, h in enumerate(held) if set(h) & mine
                     for o in layout._chan_orbs[c])

def crossed(reading, layout, group, kept, cperm, *, tol=None):
    r"""The sectors an orbit of multiplets lies on differently at different multiplets.

    ``reading`` holds the sectors, and ``layout`` the multiplets of
    ``group``'s representation.  The unitary operations of ``kept``, by
    position, carry the multiplets in orbits, ``cperm`` saying where each
    carries each.  An orbit of multiplets crosses its sectors where its
    multiplets do not all lie on the same set of them (:func:`held_sectors`).
    The ``p_x`` of one site and the ``p_y`` a screw carries it onto, which no
    hopping joins, are such an orbit.  An orbit whose multiplets all lie on
    the same sectors crosses none.  A ``(p_x, p_y)`` pair nothing couples,
    one multiplet holding both, is such an orbit.  ``tol`` is the declared
    tolerance the sectors were read at.  Returns the indices of every crossed
    sector, as a frozenset.
    """
    syms = group._symmetries
    u_pos = [int(p) for p in kept if not syms[int(p)].time_reversal]
    classes, _n = _matrices._channel_classes(cperm, u_pos, layout._nchan)
    by_class = {}
    for c, held in enumerate(held_sectors(reading, layout, tol=tol)):
        if held:
            by_class.setdefault(int(classes[c]), set()).add(held)
    return frozenset(j for sets in by_class.values() if len(sets) > 1
                     for held in sets for j in held)

def _commutators(members, prod, inv, ident):
    """The subgroup the commutators ``x y x⁻¹ y⁻¹`` of ``members`` generate.

    ``members`` are positions of operations, ``prod`` the group's
    multiplication table by position, ``inv`` maps each position to its
    inverse's, and ``ident`` is the identity's position.  Returns the
    subgroup as a set of positions.
    """
    gens = {int(prod[int(prod[int(prod[x, y]), inv[x]]), inv[y]])
            for x in members for y in members}
    got = {int(ident)} | gens
    frontier = list(got)
    while frontier:
        a = frontier.pop()
        for g in gens:
            b = int(prod[a, g])
            if b not in got:
                got.add(b)
                frontier.append(b)
    return got

def _beyond_the_group(K, G, prod, ident):
    r"""Whether ``K`` has a one-dimensional character that no character of ``G`` restricts to.

    ``G`` is a group and ``K`` a subgroup of it, both as positions of
    unitary operations; ``prod`` is the multiplication table by position and
    ``ident`` the identity's position.  A one-dimensional character of ``K``
    is a restriction of one of ``G`` exactly where it is one on every
    commutator of ``G`` that ``K`` holds.  So ``True`` is returned exactly
    where ``K`` holds an element of the commutator subgroup of ``G`` outside
    its own commutator subgroup.
    """
    inv = {int(p): next(int(q) for q in G if int(prod[int(p), int(q)]) == int(ident))
           for p in G}
    return bool((set(int(k) for k in K) & _commutators(G, prod, inv, ident))
                - _commutators(K, prod, inv, ident))

def sector_orbits(reading, action, group, layout, kept, *, tol=None):
    r"""The orbits of sectors under ``kept``, and which are read together.

    ``reading`` holds the sectors, ``action`` what ``group``'s operations do
    to them (:func:`sector_action`), and ``layout`` the multiplets.
    ``kept`` are the operations, by position, and ``tol`` the declared
    tolerance the sectors were read at.  Returns ``(orbits, together)``.
    ``orbits`` is ``{root: [sector, ...]}``, each orbit keyed by its lowest
    sector.  ``together`` is ``[(roots, multiplets), ...]``: the orbits linked
    through every multiplet that lies on sectors of several
    (:func:`held_sectors`), each set with the multiplets it lies on, in the
    order of their lowest roots.  An orbit sharing no multiplet is a set of
    its own.  Returns ``None`` where an operation of ``kept`` has no reading
    on the sectors.
    """
    pos_of = {int(s.ind): p for p, s in enumerate(group._symmetries)}
    images = {pos_of[int(i)]: tuple(int(j) for j in got)
              for i, got in action.images.items() if int(i) in pos_of}
    if any(int(p) not in images for p in kept):
        return None
    parent = list(range(len(reading.sectors)))

    def _find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for p in kept:
        for j, image in enumerate(images[int(p)]):
            ra, rb = _find(j), _find(int(image))
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    orbits = {}
    for j in range(len(reading.sectors)):
        orbits.setdefault(_find(j), []).append(j)
    held = held_sectors(reading, layout, tol=tol)
    link = {root: root for root in orbits}

    def _top(a):
        while link[a] != a:
            link[a] = link[link[a]]
            a = link[a]
        return a

    for h in held:
        roots = sorted({_find(int(j)) for j in h})
        for r in roots[1:]:
            ra, rb = _top(roots[0]), _top(r)
            if ra != rb:
                link[max(ra, rb)] = min(ra, rb)
    sets = {}
    for root in orbits:
        sets.setdefault(_top(root), []).append(root)
    together = []
    for top in sorted(sets):
        roots = sorted(sets[top])
        together.append((roots, [c for c, h in enumerate(held)
                                 if {_find(int(j)) for j in h} & set(roots)]))
    return orbits, together

def induced_orbits(reading, action, group, layout, kept, cperm, *, reference=None, tol=None,
                   max_candidates=2 ** 20):
    r"""Every orbit of sectors whose representations are built by induction.

    Three kinds of orbit are built: those whose sectors hold identical
    copies; those of several sectors of one copy that one multiplet holds
    together, as the square's ``p_x, p_y`` pair nothing couples or three
    lines nothing couples at one site; and those of several sectors of one
    copy that an orbit of multiplets crosses (:func:`crossed`), where the
    stabilizer of a sector has a one-dimensional character that no character
    of the group restricts to (:func:`_beyond_the_group`).  Orbits that one
    multiplet holds sectors of together are read together, as one
    :class:`SharedOrbits`.  Every other orbit of one copy is a coupled part
    of the crystal, whose representations are the group's one-dimensional
    characters times what the hoppings fix, and is not built here.  On states
    that carry a spin the orbits are read on the orbital factor, every matrix
    carrying the operation's own spin lift.

    Parameters
    ----------
    reading : SectorReading or None
        The sectors.  ``None`` returns an empty list.
    action : SectorAction
        What ``group``'s operations do to them (:func:`sector_action`).
    group : .._spacegroup.GroupData
        The group.
    layout : ..symmetry.SiteMultiplets
        The multiplets the representation is written on.
    kept : iterable of int
        The operations represented, unitary and antiunitary, by position in
        the group's list of operations.
    cperm : mapping
        Per position, where the operation carries each multiplet.
    reference : int, default None
        The multiplet an orbit is read at, where the orbit holds it.
    tol : float, default None
        The declared tolerance the sectors were read at.  Where it is given,
        the sectors of one copy of orbits read together, and the several
        sectors of one orbit that one multiplet holds, are held to their
        shares (:func:`held_shares`).
    max_candidates : int, default 2 ** 20
        The most candidates an orbit of several copies, or the orbits read
        together, may have.

    Returns
    -------
    list of InducedOrbit, SharedOrbits or None
        One entry per orbit, or set of orbits read together, that is built;
        ``None`` for one the sectors do not read.  Empty where none is built,
        or where an operation of ``kept`` has no reading on the sectors.

    Raises
    ------
    _TooMany
        Where an orbit of several copies, or a set of orbits read together,
        has more than ``max_candidates`` candidates.
    """
    if reading is None:
        return []
    crossing = crossed(reading, layout, group, kept, cperm, tol=tol)
    held = held_sectors(reading, layout, tol=tol)
    if (not crossing and not any(int(s.copies) > 1 for s in reading.sectors)
            and not any(len(h) > 1 for h in held)):
        return []
    found = sector_orbits(reading, action, group, layout, kept, tol=tol)
    if found is None:
        return []
    orbits, together = found
    pos_of = {int(s.ind): p for p, s in enumerate(group._symmetries)}
    images = {pos_of[int(i)]: tuple(int(j) for j in got)
              for i, got in action.images.items() if int(i) in pos_of}
    prod = group._get_product_table(get_diff=True)[0]
    ident = _matrices._identity_position(group)
    u_pos = [int(p) for p in kept if not group._symmetries[int(p)].time_reversal]
    out = []
    alone = set()
    for roots, shared in together:
        if len(roots) < 2:
            alone |= set(roots)
            continue
        bases = None
        if tol is not None:
            # held to their shares, as the projection onto the symmetric part
            # holds them (:func:`.._labeling._held_sets`)
            members = [t for r in roots for t in orbits[r]]
            if all(int(reading.sectors[t].copies) == 1 for t in members):
                bases = held_shares(reading, members, atom_sites(reading, layout), tol,
                                    on=held_on(reading, layout, held, members))
        try:
            out.append(SharedOrbits(
                [InducedOrbit(reading, action, group, layout, kept, cperm, orbits[r],
                              reference=reference, tol=tol, max_candidates=max_candidates,
                              shared=shared, bases=bases) for r in roots],
                max_candidates=max_candidates))
        except _Unread:
            out.append(None)
    sites = atom_sites(reading, layout)
    for root in sorted(orbits):
        sectors = orbits[root]
        if root not in alone:
            continue                      # read together with the orbits it shares with
        bases = None
        if int(reading.sectors[sectors[0]].copies) < 2:
            if len(sectors) < 2:
                continue
            if any(len(set(h) & set(sectors)) > 1 for h in held):
                # several sectors of the orbit one multiplet holds, held to their
                # shares as the projection holds them (:func:`.._labeling._held_sets`)
                if tol is not None:
                    bases = held_shares(reading, sectors, sites, tol,
                                        on=held_on(reading, layout, held, sectors))
            else:
                if not set(sectors) & crossing:
                    continue
                j0 = int(sectors[0])
                if not _beyond_the_group([p for p in u_pos if images[p][j0] == j0], u_pos,
                                         prod, ident):
                    continue
        try:
            out.append(InducedOrbit(reading, action, group, layout, kept, cperm, sectors,
                                    reference=reference, tol=tol,
                                    max_candidates=max_candidates, bases=bases))
        except _Unread:
            out.append(None)
    return out

def _reaching(lead, frames, on, want, gate):
    r"""The conserved unitary writing the matrices ``on`` as ``want``, or ``None``.

    ``lead`` holds each sector's frame ``E_t`` on the multiplet's states, and
    ``frames`` the sectors' ``(F, copies, dim, first, rep)``
    (:meth:`InducedOrbit.sector_frames`), of which the copies and the
    dimension are read.  ``on`` and ``want`` map each operation, by position,
    to its matrix on the multiplet and to the matrix wanted there.  The
    ``C = Σ_t E_t (c_t ⊗ 1) E_t†`` with ``want(g) C = C on(g)`` form a
    linear space.  Its member nearest the identity is made unitary sector by
    sector.  Where the space is one direction orthogonal to the identity,
    that direction is used.  ``gate`` is the tolerance of every comparison.
    Returns the unitaries ``[c_t]``, one per sector.  Returns ``None`` where
    the space is empty, where that member is singular on a sector, or where
    the unitary made misses ``want`` by more than ``gate``.
    """
    ops = sorted(on)
    units, start = [], []
    for E, f in zip(lead, frames):
        m, d = int(f[1]), int(f[2])
        start.append(len(units))
        for a in range(m):
            for b in range(m):
                e = np.zeros((m, m))
                e[a, b] = 1.0
                units.append(E @ np.kron(e, np.eye(d)) @ E.conj().T)
    A = np.stack([np.concatenate([(want[p] @ X - X @ on[p]).reshape(-1) for p in ops])
                  for X in units], axis=1)
    _u, s, vh = np.linalg.svd(A, full_matrices=True)
    top = max(float(s[0]) if s.size else 0.0, 1.0)
    null = np.asarray([vh[i].conj() for i in range(len(units))
                       if i >= len(s) or float(s[i]) <= gate * top]).T
    if not null.size:
        return None
    one = np.concatenate([np.eye(int(f[1])).reshape(-1) for f in frames]).astype(complex)
    x = null @ (null.conj().T @ one)
    if float(np.abs(x).max()) <= gate and null.shape[1] == 1:
        x = null[:, 0]                    # the identity reaches only its orthogonal complement
    cs = []
    for t, f in enumerate(frames):
        m = int(f[1])
        c = x[start[t]:start[t] + m * m].reshape(m, m)
        u, sv, wh = np.linalg.svd(c)
        if float(sv[-1]) <= gate * max(float(sv[0]), 1.0):
            return None                   # no unitary among them
        cs.append(u @ wh)
    C = sum(E @ np.kron(c, np.eye(int(f[2]))) @ E.conj().T
            for E, c, f in zip(lead, cs, frames))
    if any(float(np.abs(C @ on[p] @ C.conj().T - want[p]).max()) > gate for p in ops):
        return None
    return cs

def tables_gauge(frames, W, target, rows, syms, *, tol=None):
    r"""The matrices ``W``, turned toward ``target`` by a conserved unitary.

    What ``H`` conserves on an orbit of sectors is ``C = Σ_t F_t (c_t ⊗ 1)
    F_t†``, a unitary ``c_t`` on each sector's copies, and conjugating by it
    leaves every character unchanged.  ``frames`` are the sectors' ``(F,
    copies, dim, first, rep)`` on the orbit's states
    (:meth:`InducedOrbit.sector_frames`), and ``W`` the representation,
    ``{position: matrix}`` on those states.  ``rows`` are one multiplet's
    states, as indices into the orbit's states, and ``target`` the matrices
    wanted there, ``{position: matrix}`` over the unitary operations carrying
    the multiplet onto itself.  ``syms`` are the group's operations by
    position, and ``tol`` the declared tolerance the orbit was read at.  ``C``
    solves ``target(g) C = C W(g)`` on the multiplet, unique up to a number
    where the multiplet carries one irreducible representation.  It is the
    solution nearest the identity, made unitary sector by sector
    (:func:`_reaching`).  Where no such ``C`` is found, each sector after its
    orbit's first, both of one copy, is put in
    the phase that brings its coset representative's block from the first
    sector nearest that representative's matrix in ``target``, where the
    block reaches it.  An antiunitary operation's matrix is turned as
    ``C W C^T``, and the antiunitary matrices are then put in one phase, the
    largest entry of the first one's real and positive.  Returns the turned
    matrices, or ``W`` itself where nothing moves.
    """
    rows = np.asarray(rows, dtype=int)
    lead = [np.asarray(f[0], dtype=complex)[rows, :] for f in frames]
    ops = sorted(int(p) for p in target)
    on = {p: np.asarray(W[p], dtype=complex)[np.ix_(rows, rows)] for p in ops}
    want = {p: np.asarray(target[p], dtype=complex) for p in ops}
    gate = noise_floor(ASSEMBLED, tol)
    cs = _reaching(lead, frames, on, want, gate)
    if cs is None:
        # out of reach: each coset representative's block nearest its matrix
        cs = [np.eye(int(f[1]), dtype=complex) for f in frames]
        moved = False
        for t, (_F, m, _d, first, rep) in enumerate(frames):
            if t == first or int(m) > 1 or int(frames[first][1]) > 1 or int(rep) not in want:
                continue
            block = (lead[t] @ lead[t].conj().T @ on[int(rep)]
                     @ lead[first] @ lead[first].conj().T)
            z = complex(np.trace(want[int(rep)].conj().T @ block))
            if abs(z) > gate and abs(z - abs(z)) > gate * abs(z):
                cs[t] = np.asarray([[np.conj(z) / abs(z)]])
                moved = True
        if not moved:
            return W
    if all(float(np.abs(c - np.eye(len(c))).max()) <= gate for c in cs):
        return W                          # the matrices are the tables' already
    full = sum(np.asarray(f[0], dtype=complex) @ np.kron(c, np.eye(int(f[2])))
               @ np.asarray(f[0], dtype=complex).conj().T for c, f in zip(cs, frames))
    out = {p: (full @ np.asarray(M) @ full.T if syms[int(p)].time_reversal
               else full @ np.asarray(M) @ full.conj().T) for p, M in W.items()}
    anti = sorted(int(p) for p in out if syms[int(p)].time_reversal)
    if anti:
        z, _where = _matrices._principal_entry(out[anti[0]])
        if abs(z) > 0.0:
            for p in anti:
                out[p] = out[p] * (np.conj(z) / abs(z))
    return out
