r"""The sectors the Hamiltonian's conserved matrices acting within sites split its states into.

The matrices that commute with ``H(k)`` at every ``k`` and act within sites form
an algebra.  It holds the identity; the projector onto every set of orbitals the
hoppings couple to nothing else; and, where several such sets are identical
copies of one another, the matrices that exchange the copies.  Every such
algebra is a sum of **sectors**: a sector holds ``m`` identical copies of a set
of ``d`` states, and on it the algebra is every ``m × m`` matrix acting on the
copies, times the identity on each copy's states.  A crystal whose hoppings
connect every orbital is one sector of one copy.  Two orbitals at one site that
no hopping reaches, at different energies, are two sectors; a square's
``p_x, p_y`` pair that nothing couples is two sectors whichever basis the pair is
written in, since a sector is a set of states and not of orbitals.  Two identical
copies of a lattice that no hopping joins, on the same sites, are one sector of
two copies: a pseudospin the Hamiltonian conserves.

The sectors are read once, where a model enters, off the identity's solution
space the detector measured -- the matrices that commute with ``H(k)`` at every
``k`` -- at the declared precision: a coupling below it leaves the projectors
onto the pieces it joins in that space, so the pieces are read apart, and one
above it joins them.  On states that carry a spin, native or declared in pairs,
the members read are those that act on the orbitals alone, ``O ⊗ 1``, so a
spin rotation the Hamiltonian conserves is never a sector's.

An operation the Hamiltonian obeys carries each sector onto a sector, and every
matrix it can be acts on each copy's states by one factor ``w(g)``, the
Hamiltonian's own, up to a number: what it does on the copies is the freedom the
hoppings leave.  :func:`sector_action` reads, per operation, the sector each one
is carried onto and that factor.

So the representations of a group ``H`` the hoppings leave open are read off
the sectors alone.  ``H`` carries the sectors in orbits; on each orbit a
representation is induced from the stabilizer of one of its sectors, by any
representation ``b`` of that stabilizer on the ``m`` copies whose product with
the factors ``w`` is a representation of it, extended to time reversal where
``H`` holds it.  An orbit of one sector of one copy -- a coupled part of the
crystal -- is carried by every one-dimensional character of ``H``; an orbit of
several sectors -- uncoupled orbitals a symmetry exchanges -- by every
representation induced from a character of a sector's stabilizer; and an orbit
of identical copies by every representation of the copies the stabilizer
admits.  A smaller ``H`` splits the orbits by itself, and what each declared
site holds under each choice is the induced character, read without building a
matrix (:class:`SectorSearch`).
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
    basis : numpy.ndarray
        ``(n, copies · dim)``, orthonormal columns in the orbital factor,
        column ``a · dim + w`` being state ``w`` of copy ``a``: every conserved
        matrix acting within sites is ``basis (c ⊗ 1) basis†`` on the sector,
        ``c`` acting on the copies.
    copies : int
        How many identical copies the sector holds, ``m``.
    dim : int
        How many states one copy holds, ``d``: orbitals, or declared spinor
        pairs, counted in the orbital factor.
    atoms : tuple of int
        The orbitals (or declared spinor pairs) the sector's states lie on.
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
    algebra : numpy.ndarray
        ``(k, n, n)``, an orthonormal basis of the conserved matrices acting
        within sites, in the orbital factor.
    sectors : tuple of Sector
        The sectors, ordered by the lowest orbital (or pair) each touches.
    atoms : tuple of tuple of int
        The orbitals each index of the orbital factor stands for: one orbital,
        or a declared spinor pair's two.
    su2 : bool
        Whether the states carry a spin, native or declared, so that the
        orbital factor is read off the members ``O ⊗ 1``.
    defect : float
        How far the conserved matrices miss the sectors' form, relative to
        their scale: rounding where the reading is exact, and up to the
        declared tolerance where the matrices were measured at it.
    tol : float or None
        The declared tolerance the reading was made at, relative to ``max|H|``;
        ``None`` where nothing was declared.
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
    images : mapping
        ``{ind: images}``, per operation by index the sector it carries each
        sector onto.
    factors : mapping
        ``{(ind, sector): w}``, the factor every matrix the operation can be
        carries on each copy's states between the sector and its image: unitary,
        of determinant one.
    defect : float
        How far the operations' matrices miss that form, relative to their
        scale.
    """

    images: dict
    factors: dict
    defect: float

def _atoms_of(model, spin):
    """``(atoms, order, su2)`` -- the indices of the orbital factor and the states' spin order.

    An index of the orbital factor is one orbital, or one declared spinor pair;
    ``order`` puts the states in pair order, the spin fast, where they carry a
    spin, and is ``None`` where they do not.
    """
    if int(model.nspin) == 2:
        atoms = tuple((int(o),) for o in range(int(model.norb)))
        return atoms, np.arange(int(model.nstate)), True
    if spin is not None:
        return (tuple(tuple(int(o) for o in pair) for pair in spin.pairs),
                np.asarray(spin.interleaved_index, dtype=int), True)
    return tuple((int(o),) for o in range(int(model.norb))), None, False

def _orthonormal(members, tol):
    """An orthonormal basis of the span of ``members``, under the trace inner product."""
    A = np.asarray(members, dtype=complex)
    if not len(A):
        return A
    n = int(A.shape[1])
    # the right singular vectors span the members' own span -- not its conjugate,
    # which commutes with H(k)* rather than with H(k)
    _u, s, vh = np.linalg.svd(A.reshape(len(A), -1), full_matrices=False)
    keep = s > noise_floor(ZERO, tol) * max(float(s[0]), 1.0)
    return vh[keep].reshape(int(keep.sum()), n, n)

def _orbital_factor(members, sym, order, su2, tol):
    """The orbital factors ``O`` of members ``O ⊗ S(g)``, orthonormal.

    ``S(g)`` is the operation's own spin lift.  Every solution space read here
    was read in its operation's own form (:func:`.._multiplets._own_form_spaces`),
    so each member is ``O ⊗ S(g)`` and ``O`` is its trace against the lift on
    the spin.  On states with no spin the members are their own orbital
    factors.
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
    """``[(rows, cols)]`` per site: where a matrix carrying each site onto its image may be nonzero.

    ``image[s]`` is the site the operation carries site ``s`` onto, every site
    its own where ``None``: the blocks of a matrix acting within sites.
    """
    nspin = int(model.nspin)
    states = [np.asarray([o * nspin + t for o in orbs for t in range(nspin)], dtype=int)
              for orbs in assignment.site_orbitals]
    return [(states[s if image is None else int(image[s])], states[s])
            for s in range(len(states))]

def _within_sites(basis, blocks, tol):
    """The members of a solution space carrying each site onto its image alone."""
    members, _reach = _matrices._block_members(basis, blocks,
                                               tol=noise_floor(_IN_BLOCKS, tol))
    return members

def _draw(B, rng):
    """One member of the span of ``B``, its coefficients drawn from ``rng``."""
    c = rng.normal(size=len(B)) + 1j * rng.normal(size=len(B))
    return np.einsum("j,jab->ab", c, B)

def _tensor_defect(Z, m, d):
    """How far one block ``Z`` on a sector misses ``c ⊗ 1``, relative to its scale."""
    R = Z.reshape(m, d, m, d).transpose(0, 2, 1, 3).reshape(m * m, d * d)
    _u, s, vh = np.linalg.svd(R, full_matrices=False)
    if not s.size or float(s[0]) <= 0.0:
        return 0.0
    worst = float(s[1] / s[0]) if s.size > 1 else 0.0
    w = vh[0].reshape(d, d)
    trace = complex(np.trace(w)) / d
    return max(worst, float(np.abs(w - trace * np.eye(d)).max()))

def _sectors_once(B, rng, tol):
    """``(sectors, defect)`` -- one reading of the sectors of ``B``, or ``None`` where it misses.

    The eigenspaces of one Hermitian member drawn from ``B`` are the copies, one
    eigenvalue to each copy of each sector; two copies are of one sector where
    another drawn member carries one into the other; and that member, carried
    from a sector's first copy into each of the others along the links found,
    puts every copy in one frame, so that each member of ``B`` acts on the
    sector as ``c ⊗ 1``.  Two more draws check the reading: what either
    carries from one sector into another, and how far either misses ``c ⊗ 1`` on
    a sector, is its defect.
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
    """The sectors of one Hamiltonian, read off the identity's solution space; ``None`` where unread.

    ``validated`` holds the detector's solution spaces (``validated._bases``);
    the identity's is every matrix that commutes with ``H(k)`` at every ``k``,
    and its members acting within sites are read, in the orbital factor
    (:func:`_orbital_factor`).  ``tol`` is the declared tolerance relative to
    ``max|H|`` every threshold here reads where it is larger than its exact
    default (:func:`.._tolerance.noise_floor`); ``None`` keeps the defaults.
    ``None`` is returned where the space is not held, or is past the read-off's
    own ceiling.
    """
    syms = group.symmetries
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
    """``(sectors, defect)`` of the algebra ``B``, the best of the draws (:func:`_sectors_once`)."""
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
    r"""The sectors as the multiplets carry them: the conserved matrices block-diagonal on them.

    The multiplets the representation is written on are what a site's content
    is read on.  Where every conserved matrix acting within sites acts within
    each of them -- a model whose identical copies the multiplets hold
    together, or one with none -- the sectors carried are ``reading``'s own,
    and ``reading`` is returned.  Where the multiplets keep identical copies
    apart, each copy written as multiplets of its own, the matrices exchanging
    the copies reach between multiplets, and the sectors carried are those of
    the conserved matrices that do not: each copy a sector of one copy, read
    on its own.  ``layout`` holds the multiplets (:class:`..symmetry.SiteMultiplets`)
    and ``tol`` is the declared tolerance the reading was made at.
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
    r"""``W[i, j]`` -- how far the conserved matrices join atom ``i`` of one site to atom ``j``.

    The largest, over the sectors, of how far a matrix of the sector reaches
    between the two atoms' indices in the orbital factor.  For the sector's
    projector that is zero to rounding where every sector holds each of the two
    wholly or not at all, as it does orbitals written along the sectors, and of
    order one where a sector holds part of each: a square's ``p_x, p_y`` pair
    that nothing couples, written as ``(p_x ± p_y)/√2``, is joined by a half.
    The matrices exchanging a sector's identical copies are read too: their
    reach between the two atoms, summed in square over every pair of copies,
    which is the same in any basis of the copies, joins each state of one copy
    to the same state of every other -- two identical ``s`` orbitals by one,
    with a spin or without.  ``site_atoms`` are the site's atoms, each the
    orbitals of one index of the orbital factor -- one orbital, or a declared
    spinor pair; the diagonal is zero.
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
    """What the operations of ``group`` do to the sectors of ``reading``: a :class:`SectorAction`.

    Each operation's solution space is read within the sites it carries onto
    one another (:func:`.._matrices._site_spaces`) and in the orbital factor, and
    one member drawn from it says which sector it carries each sector onto,
    the sector its weight lies on; two members drawn from it say the factor
    ``w`` they share on each copy's states between a sector and its image, read
    as the common factor of their ``c ⊗ w`` form and made unitary, of
    determinant one, so that two of them compose to a third times a ``d``-th
    root of unity.  An antiunitary operation's matrices act on the conjugate
    states.  An operation the detector kept no solution space for, or none of
    whose members acts within sites, is absent.
    """
    syms = group.symmetries
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
    """The sectors do not read this group: the factors its operations carry do not compose."""

def _composed(wa, wb, wab, *, after_antiunitary=False, tol=None):
    r"""``ρ`` with ``w(a) w(b) = ρ w(ab)``, ``w(b)*`` after an antiunitary ``a``: a root of unity.

    The factors are of determinant one (:func:`sector_action`), so ``ρ^d = 1``
    for ``d × d`` factors, and the measured ratio is read as the ``d``-th root
    of unity it sits at: exactly, whatever noise the factors carry within a
    declared precision, so that the representations it leaves are read off an
    exact factor system.  A ratio that misses that form, or sits nearer the
    middle between two roots than a quarter of their spacing, raises
    :class:`_Unread`.
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
    """More representations to weigh than ``max_candidates``: how many, and on which orbitals.

    ``copies`` is the pseudospin's dimension where one orbit's copies are
    counted; ``sets`` the number of orbits read together where several are
    (:class:`SharedOrbits`), ``None`` otherwise.
    """

    def __init__(self, count, orbitals, copies, sets=None):
        super().__init__(int(count))
        self.count, self.orbitals, self.copies = int(count), tuple(orbitals), int(copies)
        self.sets = None if sets is None else int(sets)

def _extends(rep, positions, a_pos, prod, factor, *, tol=None, seed=17):
    r"""Whether a representation of a sector's stabilizer extends to an antiunitary operation.

    ``rep`` holds ``b(u)`` for every unitary ``u`` of ``positions``, a
    representation with the factor system ``factor``, and ``a_pos`` is an
    antiunitary operation fixing the same sector, ``a²`` among ``positions``.
    The matrix ``X`` it would carry satisfies ``X b(u)* = f(a, u) / f(u', a) ·
    b(u') X`` for every ``u``, ``u' a = a u``, and ``X X* = f(a, a) b(a²)``,
    which for a unitary ``X`` is the linear ``X = Xᵀ · conj(f(a, a) b(a²))``.
    A member of that solution space drawn at random is made unitary and kept
    where it closes both: one of a pair of complex-conjugate representations
    without its partner is carried by no such ``X``.  ``tol`` is the declared
    tolerance the factors were read at (:func:`.._tolerance.noise_floor`).
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
    r"""Whether a declared content is carried on a subgroup, read off the sectors.

    Made once per round, on the sectors the multiplets carry
    (:func:`carried_reading`) and what the round's operations do to them
    (:func:`sector_action`).  ``decls`` are the declared contents as the round
    reads them, each over whole multiplets of one site; ``positions`` the
    round's unitary operations still standing; ``reversal`` the pure time
    reversal where it stands, ``None`` where it does not; ``tol`` how far a
    character may sit from a declared one and be read as it; and
    ``max_candidates`` the most representations weighed at once.

    :meth:`met` answers for any subgroup of those operations: whether one
    representation per orbit of sectors under that subgroup, of those the
    sectors leave open, gives every declared site what was declared -- the same
    content where the whole site was declared, a content holding it where part
    of the site was, and the corep content alone where the words are coreps.
    An orbit time reversal carries onto another is chosen once, its partner
    carrying the conjugate.  Declarations sharing no orbit are weighed apart, so
    that the choices multiply only inside each.  States that carry no
    representation of their stabilizer with the tables' factor system put
    nothing to a declaration there: that is their projective class, a verdict
    of its own.  Time reversal is asked where these states hold it; where none
    of the copies extends to it on the identity, its square on the states is
    not the one the tables ask for, and the declarations are asked of the
    unitary operations alone.  ``complete`` is false where an operation of the
    round has no reading on the sectors, which then answer nothing.
    """

    def __init__(self, reading, action, group, assignment, decls, positions, *,
                 reversal=None, tol=CHARACTER, max_candidates=2 ** 20):
        self.reading, self.group, self.assignment = reading, group, assignment
        self.syms = group.symmetries
        self.prod, _tdiff, self.spf = group.get_product_table(get_diff=True)
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
        # TIME REVERSAL, where these states hold it: on the identity every
        # declaration is carried by the copies as they stand, so a time reversal
        # none of them extends to -- one whose square on the states is not the one
        # the tables ask for, a spinor's read in single-valued tables -- is no
        # operation of these states, which its own verdict says once the
        # unitary operations settle; the declaration is then asked of those alone
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
        """The factor ``O(x) O(y) = τ O(xy)`` the orbital part carries: the table's own over the spin lifts'.

        One wherever the tables and the lifts agree, which every group
        identified here does; a pair where it is not is no group these
        sectors read.
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
        """``ρ`` with ``w(a) w(b) = ρ w(ab)`` on one sector, ``w(b)*`` after an antiunitary ``a``.

        The factors are of determinant one, so ``ρ`` is a root of unity, and
        it is read as the one it sits at.
        """
        return _composed(wa, wb, wab, after_antiunitary=after_antiunitary,
                         tol=self.reading.tol)

    # ---- one orbit of sectors under a subgroup ---------------------------------
    def _orbit(self, orbit, H):
        """The stabilizer, frames and representations of one orbit of sectors under ``H``."""
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
        """``{h: [(k, θ)]}`` -- what one orbit puts at declared site ``i`` under each operation fixing it.

        The induced character: an operation fixing a sector ``t`` of the orbit
        acts there as ``k = r_t⁻¹ h r_t`` of the stabilizer, ``r_t`` carrying the
        orbit's first sector onto ``t``, and the site's share is ``tr b(k)``
        times ``θ = tr(π_t w(k))``, ``π_t`` the site's states in one copy of
        ``t`` read in the frame ``r_t`` carries the first sector's into.
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
        """Whether declared site ``i`` holds what was declared, its content ``{h: χ}`` on ``H``."""
        item = self.items[i]
        ops = [h for h in H if h in content]
        if not item["partial"]:
            for h in ops:
                if h not in item["want"]:
                    continue
                if item["corep"] and abs(complex(np.trace(self._lift(h)))) <= ZERO:
                    continue                  # a corep content says nothing here
                if abs(content[h] - item["want"][h]) > self.tol:
                    return False
            return True
        # part of what the site holds: contained in it, on the site group H leaves
        table = item["table_of"](tuple(int(self.syms[h].ind) for h in ops))
        if table is None:
            return True
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
        """Whether the subgroup of unitary operations ``sub`` (by ``.ind``) carries every declaration."""
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
    """The site each index of ``reading``'s orbital factor lies at, as the multiplets ``layout`` place it."""
    chan_of = np.asarray(layout._chan_of, dtype=int)
    return np.asarray([int(layout._chan_site[int(chan_of[int(orbs[0])])])
                       for orbs in reading.atoms], dtype=int)

def chan_rows(reading, layout):
    """Per multiplet of ``layout``, the indices of ``reading``'s orbital factor it holds, in order.

    On states read without spin they are the multiplet's states; on states that
    carry a spin, native or declared in pairs, each orbital (or pair) once, in
    the order the multiplet's states first reach it.
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
    """``[(index, spin)]`` -- where each state lies in ``reading``'s orbital factor, and its spin.

    A native spinful model stores state ``2·orbital + spin``; a declared spinor
    basis holds each pair's ``(up, dn)`` orbitals, a state's spin its place in
    its pair.
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
    r"""A basis written in ``reading``'s orbital factor, written in the states, each index times the spin.

    On states that carry a spin, native or declared in pairs, an index of the
    orbital factor is one orbital or one declared pair, holding a state of
    either spin.  Column ``j`` of ``basis`` becomes columns ``2j`` and
    ``2j + 1``, the same coefficients on the up and on the down state, so a
    basis copy-major in the orbital factor stays copy-major, each copy's
    states its orbital factor's times the spin.  ``nstate`` is the model's
    number of states.
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
    r"""Copies lying along the orbitals within a declared precision, held exactly along them; or ``None``.

    Read at a declared precision, copies that lie along the orbitals do so only
    within it: a copy's weight reaches the other copies' orbitals, and the
    matrix carrying one copy onto another reaches between sites, by the noise
    the precision allows.  Each copy is put wholly on its own orbitals, the
    first made unitary there, and every copy after it is the first carried by
    the matrix between them, kept within each site and made unitary: the
    copies are then exact, and every matrix acting on the copies alone acts
    within sites, as a conserved one does.  ``per`` is the basis copy by copy,
    ``(n, m, d)``, and ``sites`` the site of each of its ``n`` indices.
    ``None`` where a copy does not lie along orbitals of its own within the
    precision, or the copies' orbitals do not sit at the same sites.
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
    r"""Sectors of one copy read at a declared precision, each held exactly to its own share of every site.

    Read at a declared precision, sectors that split a multiplet between them
    -- the bonding and the antibonding state of two orbitals an on-site
    coupling joins -- are each a set of states only within it: a sector's
    projector reaches between sites, and two sectors' shares of one site
    overlap, by the noise the precision allows.  At each site the shares are
    read off together, as the eigenvectors of one matrix weighting each
    sector's projector by a number of its own, so that they are exact and
    orthogonal; and each sector's basis is the nearest one to its own lying
    in its shares.  ``sectors`` are indices into ``reading.sectors``, each of
    one copy, and ``sites`` the site of each index of the orbital factor;
    ``on`` the indices of the multiplets holding them, where they are known,
    whose states the shares are read within: two lines nothing couples, held
    by a multiplet of their own beside a third line's, reach that third line
    by what the precision allows, and are held off it exactly.  A sector lying
    along orbitals of its own to rounding -- each of those lines -- is held
    exactly along them; one turned off them by more, as an orbital frame turned
    within the precision is, keeps its own turn.  Returns ``{sector: basis}``,
    or ``None`` where a sector's shares do not hold its states.
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
    r"""A sector's basis, its copies put along the orbitals as written wherever they lie so.

    Which basis of its copies a sector is read in is no measurement: every
    unitary acting on the copies alone commutes with ``H``.  The copies are put
    along the orbitals as written -- the copy holding the lowest orbital first,
    then the one holding the lowest orbital the first does not, and so on -- so
    that identical sets of orbitals, each written as orbitals of its own, are
    each one copy; and each copy after the first is put in the phase that makes
    the largest entry of the matrix carrying the first copy onto it real and
    positive, so that copies written alike are carried onto one another by the
    orbitals' own correspondence.  Where no copy lies along the orbitals the
    order is the one their weights give.  ``tol`` is the declared tolerance the
    sector was read at; where one was declared and ``sites`` gives the site of
    each index of the orbital factor, copies lying along the orbitals within
    it are held exactly along them (:func:`_snapped`).
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
    r"""The sector of an orbit the others are carried from: the one its orbitals put first.

    Which sector a reading lists first is the order its draw found them in, no
    measurement, so the first is read off the orbitals as written: the sector
    holding the most of the lowest orbital the orbit holds, then of the next,
    and so on; between sectors holding every orbital alike -- a square's pair
    written as ``(p_x ± p_y)/√2`` -- the one whose projector joins the orbitals
    with the largest real part, entry by entry: the sum before the difference.
    ``bases`` are the sectors' bases held to their shares where a precision was
    declared (:func:`held_shares`), read in place of the reading's own.
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
    """One representation of an orbit of sectors: its stabilizer's part, the rest read lazily.

    ``combo`` names the irreducible representations of the stabilizer the
    copies carry -- one, one-dimensional, where each sector holds one copy --
    ``character`` their character on its unitary operations, and
    ``X`` the matrix the copies take under the antiunitary operation fixing the
    orbit's first sector, where there is one.  The matrices on the orbit's
    states (``"W"``) are built the first time they are asked for, and the
    content each multiplet holds (``"contents"``) is read off the character.
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

class InducedOrbit:
    r"""One orbit of sectors, and every representation the hoppings leave on it.

    ``H`` conserves every matrix acting on the copies of a sector, so what the
    copies carry is not fixed by it; every operation carries each sector of the
    orbit onto another by the factor ``w`` it fixes on each copy's states,
    times any matrix on the copies -- a number, where the sector holds one
    copy, which ``H`` leaves free sector by sector.  A representation of the
    group on the orbit is therefore induced from the stabilizer ``K`` of its
    first sector: a representation ``b`` of ``K`` on the copies whose product
    with ``w`` is one of the group's, extended to the antiunitary operations
    ``K`` holds, and carried round the orbit along the operations taking the
    first sector onto each of the others, every sector's copies put along the
    orbitals as written (:func:`aligned`).  On every operation ``g`` and
    sector ``t`` the matrix is ``F_{gt} (b(k) ⊗ w(k)) F_t^†`` -- ``F_t`` the
    frame the coset representative ``r_t`` carries the first sector's into,
    ``k = r_{gt}⁻¹ g r_t`` -- conjugated where ``r_{gt}`` is antiunitary.
    What a multiplet holds under each is the induced character, read without
    building a matrix; the matrices are built where they are asked for and
    checked against the group's own table and every operation's solution
    space (:meth:`check`).

    ``sectors`` are the orbit's sectors of ``reading``, the first read the one
    its orbitals put first (:func:`_first_sector`);
    ``action`` what the group's operations do to them (:func:`sector_action`);
    ``kept`` the positions represented, unitary and antiunitary, and ``cperm``
    where each carries each multiplet; ``reference`` the multiplet a lead is
    taken at where the orbit holds it; ``tol`` the declared tolerance.  The
    candidates are counted before anything is built, and more than
    ``max_candidates`` raise :class:`_TooMany`; an orbit the sectors do not
    read -- an operation with no reading on them, factors that do not
    compose, a multiplet holding part of the orbit -- raises :class:`_Unread`.
    ``shared`` are the multiplets of orbits read together where they hold
    shares of one another's multiplets (:class:`SharedOrbits`): the orbit is
    written on all of them, holding its share of each, and ``bases`` each
    sector's basis held to its shares where a precision was declared
    (:func:`held_shares`).
    """

    def __init__(self, reading, action, group, layout, kept, cperm, sectors, *,
                 reference=None, tol=None, max_candidates=2 ** 20, shared=None, bases=None):
        self.reading, self.group, self.layout, self.cperm = reading, group, layout, cperm
        self.syms = group.symmetries
        self.prod, _tdiff, self.spf = group.get_product_table(get_diff=True)
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
        # THE COSET REPRESENTATIVES, a unitary one wherever one reaches
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
            # READ AT A DECLARED PRECISION the factors compose only within it,
            # off solution spaces of a model symmetric only within it: they are
            # the exact representation of the stabilizer nearest them, so that
            # what is induced from them is a representation of the group to
            # rounding, as a projection onto the symmetric part needs
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
        # EACH SECTOR THE IMAGE OF THE FIRST: the phase a sector's basis was read
        # in is no measurement, so each sector after the first is put in the one
        # that makes the largest entry of the matrix carrying the first sector's
        # states onto its own real and positive -- the orbitals written as one
        # another's images under the coset representatives, as the tables write
        # the sites of a Wyckoff orbit
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
        # representation, they are unitary only to that reading's rounding, which
        # every matrix induced from them would carry, and products of them more
        self.irreps = [{k: _matrices._unitarized(r[k]) for k in r} for r in irreps]
        dims = [int(np.asarray(r[self.Ku[0]]).shape[0]) for r in self.irreps]
        self.orbitals = tuple(sorted(int(o) for t in self.sectors
                                     for a in reading.sectors[t].atoms
                                     for o in reading.atoms[int(a)]))
        # COUNTED BEFORE ANYTHING IS BUILT: every sum of the stabilizer's
        # irreducible representations the copies could carry -- of one copy,
        # its one-dimensional ones, no more than a coupled part has characters
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
        """The factor ``O(x) O(y) = τ O(xy)`` the orbital part carries: the table's own over the spin lifts'.

        On states read without spin the orbital part is the whole matrix and
        ``τ`` is the table's own; on states that carry a spin every matrix is
        ``O ⊗ S(g)``, the spin lifts carry the table's factor, and the orbital
        part carries one -- wherever the tables and the lifts agree, which
        every group identified here does; a pair where they do not is no group
        these sectors read.
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
        r"""``X`` the copies take under the antiunitary operation fixing the first sector, or ``None``.

        ``X b(u)* = ν(a, u)/ν(u', a) · b(u') X`` for every unitary ``u`` of the
        stabilizer, ``u' a = a u``, and ``X X* = ν(a, a) b(a²)``; of the unitary
        solutions the one nearest the identity, where the identity reaches
        their space, and otherwise one drawn from it.
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
        """``{k: b(k) ⊗ w(k)}`` over the whole stabilizer, its antiunitary part included."""
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
        """``{multiplet: {ind: χ}}`` -- what each multiplet holds under one candidate.

        The induced character: per sector of the multiplet the operation fixes,
        the candidate's character at the stabilizer's element there times the
        trace of the multiplet's share of one copy's frame against ``w``.
        """
        out = {}
        for c, per in self.theta.items():
            out[c] = {int(self.syms[h].ind): sum(
                ((np.conj(cand.character[k]) if conj else cand.character[k]) * th
                 for k, th, conj in here), 0j) for h, here in per.items()}
        return out

    def matrices(self, cand):
        """``{position: matrix}`` -- one candidate's matrices on the orbit's states.

        Each carries every multiplet onto its image and nothing else: what a
        model symmetric only within a declared precision puts between other
        multiplets is that model's noise, not the operation's.  On states that
        carry a spin the matrix built on the orbital factor carries the
        operation's own lift on the spin, ``O ⊗ S(g)``.
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
        """How far ``W`` misses the group's table and every operation's solution space."""
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
        """``[(F, copies, dim, first, rep)]`` -- each sector's frame on the orbit's states, copy-major.

        ``first`` is where in the list the orbit's first sector is, and ``rep``
        the position of the coset representative carrying it onto this one.  On
        states that carry a spin each copy's states are its orbital factor's
        times the spin, which nothing the Hamiltonian conserves on the copies
        acts on.
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
        """The orbit's record, a conserved pseudospin's where it holds copies, its admitted candidates in order."""
        return {"multiplets": self.members, "orbitals": self.orbitals, "lead": self.lead,
                "site": self.site, "dim": int(self.m), "n_candidates": int(self.n_candidates),
                "n_admitted": len(self.candidates), "states": self.states,
                "candidates": tuple(self.candidates), "induced": self}

class _SharedCandidate:
    """One representation of orbits read together: one of each orbit's, their matrices summed."""

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
    r"""Orbits of sectors one multiplet holds sectors of together, read together.

    Two orbitals at a site coupled by a hopping that a symmetry of their own
    conserves -- the same orbital of two layers joined on site, which splits
    into a bonding and an antibonding state -- lie on two sectors no
    operation exchanges, and each orbital holds a share of both, so the
    multiplet the two make holds shares of two orbits; orbitals nothing
    couples, stated as one multiplet, hold one orbit's sector each, whole.
    What the hoppings leave on each orbit is its own (:class:`InducedOrbit`),
    one choice per orbit: the representations of the orbits together are
    every choice of one of each, counted before anything is built, their
    matrices summed and what each multiplet holds the sum of the orbits'
    shares.  Each candidate also says what each orbit puts at each multiplet
    (``"parts"``), so that a labelling reads the orbits one by one.
    ``orbits`` are the orbits' readings, each written on the multiplets they
    share; more than ``max_candidates`` choices raise :class:`_TooMany`.
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
        """How far ``W`` misses the group's table and every operation's solution space."""
        return self.orbits[0].check(W, bases)

    def sector_frames(self):
        """``[(F, copies, dim, first, rep)]`` -- every orbit's sectors' frames on the states they share."""
        out = []
        for orbit in self.orbits:
            at = len(out)
            out.extend((F, m, d, at + first, rep)
                       for F, m, d, first, rep in orbit.sector_frames())
        return out

    def record(self):
        """The orbits' record, their admitted candidates in order."""
        return {"multiplets": self.members, "orbitals": self.orbitals, "lead": self.lead,
                "site": self.site, "dim": 1, "n_candidates": int(self.n_candidates),
                "n_admitted": len(self.candidates), "states": self.states,
                "candidates": tuple(self.candidates), "induced": self}

def held_sectors(reading, layout, *, tol=None):
    """``[sectors]`` -- per multiplet of ``layout``, the sectors of ``reading`` its states lie on."""
    loose = noise_floor(_IN_BLOCKS, tol) * 10.0
    weights = [np.sum(np.abs(np.asarray(s.basis)) ** 2, axis=1) for s in reading.sectors]
    return [frozenset(j for j, w in enumerate(weights)
                      if float(w[np.asarray(rows, dtype=int)].max()) > loose)
            for rows in chan_rows(reading, layout)]

def held_on(reading, layout, held, sectors):
    """The indices of ``reading``'s orbital factor the multiplets holding any of ``sectors`` lie on.

    ``held`` is :func:`held_sectors`'s reading of ``layout``.
    """
    index = {int(o): a for a, orbs in enumerate(reading.atoms) for o in orbs}
    mine = set(int(t) for t in sectors)
    return frozenset(index[int(o)] for c, h in enumerate(held) if set(h) & mine
                     for o in layout._chan_orbs[c])

def crossed(reading, layout, group, kept, cperm, *, tol=None):
    r"""The sectors an orbit of multiplets lies on differently at different multiplets.

    The operations ``kept`` carry the multiplets in orbits, ``cperm`` saying
    where each carries each.  An orbit whose multiplets lie on different
    sectors -- the ``p_x`` of one site and the ``p_y`` of another that a screw
    carries it onto, which no hopping joins -- crosses those sectors, and the
    sectors of every such orbit are returned, as a set of indices.  An orbit
    whose every multiplet lies on the same sectors -- a ``(p_x, p_y)`` pair
    nothing couples, one multiplet holding both of its sectors -- crosses none.
    ``tol`` is the declared tolerance the sectors were read at.
    """
    syms = group.symmetries
    u_pos = [int(p) for p in kept if not syms[int(p)].time_reversal]
    classes, _n = _matrices._channel_classes(cperm, u_pos, layout._nchan)
    by_class = {}
    for c, held in enumerate(held_sectors(reading, layout, tol=tol)):
        if held:
            by_class.setdefault(int(classes[c]), set()).add(held)
    return frozenset(j for sets in by_class.values() if len(sets) > 1
                     for held in sets for j in held)

def _commutators(members, prod, inv, ident):
    """The positions the commutators ``x y x⁻¹ y⁻¹`` of ``members`` generate."""
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

    A character of the subgroup ``K`` is a restriction of one of ``G`` exactly
    where it is one on every commutator of ``G`` that ``K`` holds; every
    character of ``K`` is one on ``K``'s own commutators, so ``K`` has one that
    is not a restriction exactly where ``G``'s commutators in ``K`` are more
    than ``K``'s own.  ``K`` and ``G`` are positions, unitary, ``G`` a group.
    """
    inv = {int(p): next(int(q) for q in G if int(prod[int(p), int(q)]) == int(ident))
           for p in G}
    return bool((set(int(k) for k in K) & _commutators(G, prod, inv, ident))
                - _commutators(K, prod, inv, ident))

def sector_orbits(reading, action, group, layout, kept, *, tol=None):
    r"""The orbits of sectors under the operations ``kept``, and which are read together.

    Returns ``(orbits, together)``: ``{root: [sector, ...]}``, each orbit by
    its lowest sector, and ``[(roots, multiplets), ...]`` -- the orbits one
    multiplet holds sectors of together, linked through every multiplet that
    holds sectors of several, whole or a share of each, each set with the
    multiplets of ``layout`` it lies on; an orbit sharing a multiplet with
    none is a set of its own.  ``None`` where an operation of ``kept`` has no
    reading on the sectors.
    """
    pos_of = {int(s.ind): p for p, s in enumerate(group.symmetries)}
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
    r"""Every orbit of sectors whose representations are built by induction: ``[InducedOrbit or None]``.

    Those are the orbits holding identical copies; the orbits of several
    sectors of one copy that one multiplet holds together -- the square's
    ``p_x, p_y`` pair nothing couples, three lines nothing couples at one
    site; and the orbits of several sectors of one copy that an orbit of
    multiplets crosses (:func:`crossed`) and whose stabilizer has a
    one-dimensional character no character of the group restricts to: a
    ``p_x`` and the ``p_y`` a screw carries it onto, which no hopping joins,
    can carry ``B_{3u}`` and ``B_{2u}`` of the site's ``mmm``, which no
    one-dimensional character of the group times what the hoppings fix gives.
    And orbits one multiplet holds sectors of together -- a bonding and an
    antibonding state no operation exchanges, two orbitals each holding a
    share of both, or orbitals nothing couples stated as one multiplet -- are
    read together, each its own choice (:class:`SharedOrbits`): no one
    character of a part of the crystal made of whole multiplets says what each
    carries.  Every other orbit of one copy is a coupled part of the crystal,
    and the group's one-dimensional characters times what the hoppings fix are
    the representations induced on it: it is not built here.

    The orbits are those of the operations ``kept``, unitary and antiunitary;
    on states that carry a spin they are read on the orbital factor, every
    matrix carrying the operation's own lift on the spin, so that identical
    copies there carry an ordinary representation, the double-valued factor
    system the spin's alone.  An orbit the sectors do not read is ``None``, and
    an empty list is returned where an operation has no reading on the sectors.
    :class:`_TooMany` propagates.
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
    pos_of = {int(s.ind): p for p, s in enumerate(group.symmetries)}
    images = {pos_of[int(i)]: tuple(int(j) for j in got)
              for i, got in action.images.items() if int(i) in pos_of}
    prod = group.get_product_table(get_diff=True)[0]
    ident = _matrices._identity_position(group)
    u_pos = [int(p) for p in kept if not group.symmetries[int(p)].time_reversal]
    out = []
    alone = set()
    for roots, shared in together:
        if len(roots) < 2:
            alone |= set(roots)
            continue
        bases = None
        if tol is not None:
            # held to their shares, as the projection onto the symmetric part
            # holds them (:func:`.._labelling._held_sets`)
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
                # shares as the projection holds them (:func:`.._labelling._held_sets`)
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
    r"""``[c_t]`` -- the conserved unitary writing the multiplet's matrices ``on`` as ``want``; or ``None``.

    ``lead`` is each sector's frame on the multiplet; the ``C = Σ_t E_t (c_t ⊗
    1) E_t†`` with ``want(g) C = C on(g)`` are one linear space, and of them the
    one nearest the identity is made unitary sector by sector.  ``None`` where
    none of them is unitary on every sector the multiplet holds, or the one
    made so misses ``want`` by more than ``gate``.
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
    r"""``W`` turned by the conserved unitary that writes it as ``target`` on one multiplet, where one does.

    What ``H`` conserves on an orbit of sectors is a unitary on each sector's
    copies, ``C = Σ_t F_t (c_t ⊗ 1) F_t†`` -- a phase per sector where each
    holds one copy -- and conjugating a representation by one gives another the
    hoppings obey equally, its characters unchanged: which one is no
    measurement.  It is the one whose matrices on the multiplet of states
    ``rows`` (indices into the orbit's states) are ``target``, ``{position:
    matrix}`` over the unitary operations carrying that multiplet onto itself.
    The ``C`` with ``target(g) C = C W(g)`` there are one linear space; of them
    the one nearest the identity -- the only one, up to a number, where the
    multiplet's content is irreducible -- is made unitary sector by sector, so
    that every direction ``target`` leaves open stays as it was.  Where no
    conserved unitary carries ``W`` onto ``target`` -- the multiplet's orbitals
    not each a row of its content's irreducible representations, as a pair
    turned inside its site, or three lines a rotation permutes, are not -- each
    sector of one copy after its orbit's first is put in the phase that brings
    its coset representative's block from the first sector nearest that
    representative's matrix in ``target``, where the block reaches it, and is
    left as it was otherwise; ``W`` is returned unchanged where nothing moves.
    ``frames`` are ``(F, copies, dim, first, rep)`` per sector on the orbit's
    states, copy-major, with where its orbit's first sector is in the list and
    the position of the coset representative carrying that one onto it
    (:meth:`InducedOrbit.sector_frames`); ``tol`` is the declared tolerance the
    orbit was read at.  An antiunitary operation's matrix is turned as
    ``C W C^T``, and the antiunitary operations are put back in one phase, the
    largest entry of the first one's matrix real and positive.
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
