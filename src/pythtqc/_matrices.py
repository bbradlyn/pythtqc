"""The arithmetic of the matrices of a group's operations on the orbitals.

The matrices are composed by the multiplication table of the group, put in
their canonical phase, and made into Bloch operators.  The module also holds
the record of the multiplets the Hamiltonian binds each site's orbitals into,
and reads the members of each operation's solution space that keep within
the blocks its permutation of the sites allows.  It computes the irreducible
representations of a group with a factor system, the one-dimensional
characters of a group and of its subgroups, and the tolerance a
representation's covariance is held to.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import functools
import inspect
import itertools
import math
from fractions import Fraction
from typing import Any

import numpy as np

from . import _spacegroup
from .refusal import InternalError, Refusal
from ._tolerance import (
    CHARACTER, CLUSTER, EXACT_TOL, SINGULAR, ZERO, _COMPOSITION_GATE, _IN_BLOCKS,
    noise_floor)
from .spin import SpinBasis, spin_basis
from ._text import _orbitals_named
from ._detector import _READOFF_MAX_BYTES

__all__ = ["RepAnchor", "NormalizedRep", "normalize_rep", "bloch_operator",
           "SiteMultiplets", "covariance_gate"]


# ==========================================================================
# the matrices of the operations, and their phases
# ==========================================================================
#
# What follows turns the operations that hold into the matrix of each.  Three
# rules hold throughout.
#
#   * Nothing group-specific is typed.  Orders, required powers, spinor
#     factors and translation differences come from
#     ``GroupData._get_product_table(get_diff=True)``, and site permutations
#     and their lattice shifts from ``_tables._assign_orbitals``.
#   * A discrete choice of representation is searched for, never assumed, and
#     a search that finds none or several is returned as a ``Refusal``.
#   * Every object returned carries the residual of the identity it claims.


def _identity_position(group) -> int:
    """The position of the identity among the operations of ``group``.

    A position is an index into ``group._symmetries``.  Raises
    ``InternalError`` where ``group`` holds no identity.
    """
    eye = np.eye(3, dtype=int)
    for p, sym in enumerate(group._symmetries):
        if (not sym.time_reversal and np.all(np.asarray(sym.rotation, dtype=int) == eye)
                and np.allclose(np.asarray(sym.translation) % 1.0, 0.0, atol=1e-8)):
            return p
    raise InternalError(f"none of the {len(group._symmetries)} operations of the group is "
                        "the identity. Report it, with the model and the call.")


def _order_and_zeta(group, p, prod=None, spf=None):
    r"""The order ``n`` of the operation at ``p``, and the sign ``ζ``.

    ``n`` is the smallest integer with :math:`g^n` the identity modulo lattice
    translations, and

    .. math:: \zeta = \prod_{m=1}^{n-1} \mathrm{spf}(g, g^{m})

    is the product of the spinor factors along the powers, so that a matrix in
    its canonical phase satisfies :math:`D^{n} = \zeta\,\mathbb 1`: ``-1`` for
    a spinful twofold, ``+1`` for a spinless one.  ``group`` is the group and
    ``p`` a position in ``group._symmetries``.  ``prod`` and ``spf`` are its
    multiplication table and spinor factors by position
    (``GroupData._get_product_table``), computed where either is ``None``.
    Returns ``(n, zeta)``, two integers.  Raises ``InternalError`` where no
    power of the operation up to the order of the group is the identity.
    """
    if prod is None or spf is None:
        prod, _, spf = group._get_product_table(get_diff=True)
    ident = _identity_position(group)
    q, n, zeta = p, 1, 1
    limit = len(group._symmetries) + 1
    while q != ident:
        zeta *= int(spf[p, q])
        q = int(prod[p, q])
        n += 1
        if n > limit:  # pragma: no cover - impossible for a genuine group
            raise InternalError(
                f"operation #{group._symmetries[p].ind} of the group has no finite order "
                "in its multiplication table. Report it, with the model and the call.")
    return int(n), int(zeta)


def _compose(D1, antiunitary1, D2):
    """The product ``D1 ∘ D2`` of two operations' matrices.

    ``antiunitary1`` says whether the left operation is antiunitary.  Where it
    is, ``D2`` is conjugated, and the product is ``D1 @ conj(D2)``.
    """
    return D1 @ (np.conj(D2) if antiunitary1 else D2)


def _power(D, antiunitary, n):
    """The ``n``-fold product ``D ∘ D ∘ ... ∘ D`` of an operation's matrix.

    ``antiunitary`` says whether the operation is antiunitary, each product
    then taken as :func:`_compose` takes it.  ``n = 0`` returns the identity.
    """
    out = np.eye(D.shape[0], dtype=complex)
    for _ in range(n):
        out = _compose(D, antiunitary, out)
    return out


#: How many bytes of matrices :func:`_product_table_residual` multiplies at a time.
_RESIDUAL_BLOCK = 2 ** 18


def _product_table_residual(D, positions, syms, prod, spf):
    r"""How far matrices are from multiplying as the group does.

    The residual is ``max |D(g1) ∘ D(g2) − spf(g1, g2)·D(g1 g2)|`` over the
    pairs of ``positions``, the largest entry of each difference.  ``syms``
    is the group's operations.  ``D`` maps each position, an index into
    ``syms``, to its matrix, and holds every one of ``positions``.  ``prod``
    and ``spf`` are the multiplication table and its spinor factors, by
    position.  A pair whose product is not a key of ``D`` is skipped, and a
    pair whose residual is not a number does not count.  Returns the residual
    as a float, ``0.0`` where ``positions`` is empty.
    """
    positions = [int(p) for p in positions]
    if not positions:
        return 0.0
    right = np.stack([D[b] for b in positions])      # D(b) over the positions
    keys = [int(k) for k in D]
    held = np.stack([D[k] for k in keys])            # every matrix, in the order of keys
    where = np.full(max(int(np.max(prod)), *keys) + 1, -1)
    where[keys] = np.arange(len(keys))               # position -> index into held
    cols = np.array(positions)
    # how many D(b) one step takes
    step = max(1, int(_RESIDUAL_BLOCK // max(1, right[0].nbytes)))
    right_conj = None
    worst = 0.0
    for a in positions:
        index = where[np.asarray(prod[a, cols], dtype=int)]
        kept = index >= 0
        if not kept.any():
            continue
        if syms[a].time_reversal:
            if right_conj is None:
                right_conj = np.conj(right)
            side = right_conj
        else:
            side = right
        Da = D[a]
        signs = np.array([int(s) for s in spf[a, cols]], dtype=held.dtype)
        for lo in range(0, len(positions), step):
            part = slice(lo, lo + step)
            here = kept[part]
            if not here.any():
                continue
            lhs = np.matmul(Da, side[part])          # one product of two matrices for each b
            target, sign = index[part], signs[part]
            if not here.all():
                lhs, target, sign = lhs[here], target[here], sign[here]
            per_pair = np.abs(lhs - sign[:, None, None] * held[target]).max(axis=(1, 2))
            # a pair whose residual is not a number does not count
            worst = max(worst, float(np.fmax.reduce(per_pair)))
    return worst


def _composition_defect(D, anchors, positions, syms, prod):
    r"""How far matrices are from composing by the table, up to a phase.

    For each ``a`` of ``anchors`` and ``h`` of ``positions``, ``D(a) ∘ D(h)``
    is compared with ``ρ·D(ah)``, ``ρ`` the best scalar.  The defect is the
    largest residual of those comparisons, or of ``|ρ|`` from ``1``.  It is
    zero to rounding where the matrices form a projective representation of
    the group, whatever their phases, and the noise of a near-symmetric
    model's own blocks otherwise.  ``syms`` is the group's operations.  ``D``
    maps positions, indices into ``syms``, to matrices, and ``prod`` is the
    multiplication table by position; a product outside ``D`` is skipped.
    ``anchors`` are the operations the matrices were propagated from along
    the word tree, the generators and the antiunitary operation ``T``, so
    that these products hold the whole table.  Returns the defect as a float.
    """
    worst = 0.0
    for a in anchors:
        Da, anti = np.asarray(D[a]), bool(syms[a].time_reversal)
        for h in positions:
            c = int(prod[a, h])
            if c not in D or h not in D:
                continue
            rho, res = _proportionality(_compose(Da, anti, np.asarray(D[h])),
                                        np.asarray(D[c]))
            worst = max(worst, float(res), abs(abs(complex(rho)) - 1.0))
    return worst


def _nearest_representation(D, positions, syms, prod, *, tol=1e-12, rounds=12):
    r"""The exact projective representation nearest the matrices given.

    ``syms`` is the group's operations.  ``D`` maps positions, indices into
    ``syms``, to unitary matrices, and those of ``positions`` are read;
    ``prod`` is the multiplication table by position.  In an exact projective
    representation ``D(h) ∘ D(g) = ω(h, g)·D(hg)``, so ``D(g)`` is the
    average over ``h`` of ``ω(h, g)·D(h)^-1 D(hg)``, conjugated where ``h``
    is antiunitary.  Each round reads the phases ``ω`` off the matrices and
    replaces each ``D(g)`` by the unitary part of that average, which squares
    the defect up to a factor of order one.  The matrices returned carry the
    factor system nearest that of the matrices handed in, which
    :func:`normalize_rep` then puts in the tables'.
    Each matrix moves by the defect it had, so a phase read off the model
    stays where it was.  Returns ``{position: matrix}`` once every product
    holds to ``tol``, in the largest entry of its residual, within ``rounds``
    rounds.  Returns ``None`` where the defect stops falling or the rounds
    run out: matrices that are no near-representation of the group.
    """
    Dc = {int(p): np.asarray(D[p], dtype=complex) for p in positions if p in D}
    keys = sorted(Dc)
    previous = np.inf
    for _ in range(int(rounds)):
        new, worst = {}, 0.0
        for g in keys:
            acc = np.zeros_like(Dc[g])
            for h in keys:
                hg = int(prod[h, g])
                if hg not in Dc:
                    continue
                anti = bool(syms[h].time_reversal)
                rho, res = _proportionality(_compose(Dc[h], anti, Dc[g]), Dc[hg])
                worst = max(worst, float(res))
                omega = complex(rho) / abs(rho) if abs(rho) > 1e-12 else 1.0
                term = omega * (Dc[h].conj().T @ Dc[hg])
                acc = acc + (np.conj(term) if anti else term)
            w, _, vh = np.linalg.svd(acc)
            new[g] = w @ vh
        if worst <= tol:
            return Dc
        if worst >= previous:
            return None
        previous, Dc = worst, new
    return None


def _multiplying_roots(D, gens, words, parent, positions, prod, spf, offers, *, tol=1e-6):
    r"""The generators' phases, among those their norms allow, that multiply.

    The normalization ``D(g)^n = ζ·1`` (:func:`normalize_rep`) fixes a
    generator's phase up to an ``n``-th root of unity.  Multiplying generator
    ``i`` by ``w_i`` multiplies each propagated ``D(h)`` by ``W(h)``, the
    product of the ``w``'s along the word of ``h``.  Each relation
    ``D(g_i) D(h) = ω·spf(g_i, h)·D(g_i h)`` is measured once, ``ω`` a phase,
    and holds for the rephased matrices where ``w_i·W(h)·ω = W(g_i h)``.
    Asked of every generator against every operation, these relations are
    the whole multiplication table, since the matrices are defined along the
    word tree.  The combinations are tried in the order offered.  Where each
    generator is offered ``1`` first, as :func:`_table_phases` offers them,
    phases the norms chose that already multiply are the ones returned.

    Parameters
    ----------
    D : dict
        The matrix of each position, an index into the group's operations,
        propagated from the generators along the word tree
        (:func:`_propagated`), each generator in the phase
        :func:`normalize_rep` chose.
    gens : sequence of int
        The positions of the generators, all unitary.
    words : dict
        Each operation's word in the generators, by position
        (:func:`_bfs_words`).
    parent : dict
        The edge ``(i, r)`` of the word tree by which each operation ``q`` is
        reached, ``q = g_i·r``.
    positions : iterable of int
        The operations ``h`` each generator's relations are read against.
    prod : numpy.ndarray of int
        The multiplication table of the group, by position.
    spf : numpy.ndarray of int
        The spinor factor, ``+1`` or ``-1``, of each product, by position.
    offers : sequence of sequence of complex
        For each generator, the multipliers it may take: the roots of unity
        its normalization allows, divided by the phase it was given, so that
        ``1`` is among them.
    tol : float, default 1e-6
        How far a relation may miss, as a difference of scalars, and how far
        ``D(g_i) D(h)`` may sit from a multiple of ``spf(g_i, h)·D(g_i h)``.

    Returns
    -------
    dict
        ``roots``: one multiplier per generator, from the first combination
        that closes every relation to ``tol``, or ``None`` where none does.
        ``projective``: whether every ``D(g_i) D(h)`` is a phase times
        ``spf(g_i, h)·D(g_i h)``, that is whether the matrices form a
        projective representation of the group; where they do not, no
        combination is tried.  ``defect``: the largest departure from that
        proportionality.  ``residual``: the smallest, over the combinations
        tried, of the largest relation residual.  ``tried``: how many
        combinations were tried.
    """
    relations, defect = [], 0.0
    for gi, gp in enumerate(gens):
        for h in positions:
            c = int(prod[gp, int(h)])
            if c not in D or int(h) not in D:
                continue
            rho, res = _proportionality(D[gp] @ D[int(h)], int(spf[gp, int(h)]) * D[c])
            defect = max(defect, float(res), abs(abs(complex(rho)) - 1.0))
            relations.append((gi, int(h), c, complex(rho)))
    out = {"roots": None, "projective": bool(defect <= tol), "defect": float(defect),
           "residual": float("inf"), "tried": 0}
    if not out["projective"]:
        return out
    order = sorted(words, key=lambda q: len(words[q]))
    for combo in itertools.product(*offers):
        W = {}
        for q in order:
            if q in parent:
                gi, r = parent[q]
                W[q] = complex(combo[gi]) * W[r]
            else:
                W[q] = 1.0 + 0.0j
        worst = max((abs(complex(combo[gi]) * W[h] * rho - W[c])
                     for gi, h, c, rho in relations if h in W and c in W), default=0.0)
        out["tried"] += 1
        out["residual"] = min(out["residual"], float(worst))
        if worst <= tol:
            out["roots"] = tuple(complex(w) for w in combo)
            return out
    return out


def _commuting_pair_phases(D, positions, syms, prod, spf, *, tol=1e-6):
    r"""Every commuting pair whose matrices commute up to an extra phase.

    Where ``ab = ba``, the multiplication table gives
    ``D(a)D(b) = spf(a, b)·spf(b, a)·D(b)D(a)``, and a phase on either matrix
    cancels from it.  A measured ratio ``c`` other than ``1`` between the two
    sides is then a projective class that is not the table's, which no choice
    of phases changes.  Spin-1/2 states read with single-valued tables are the
    case: two perpendicular twofolds commute, and their spinor matrices
    anticommute.

    ``syms`` is the group's operations.  ``D`` maps positions, indices into
    ``syms``, to matrices, and the pairs are read among ``positions``.  The
    products are taken without conjugation, as for unitary operations.
    ``prod`` and ``spf`` are the multiplication table and its spinor factors,
    by position.  ``tol`` bounds how far ``D(a)D(b)`` may sit from a multiple
    of the other side, and how close to ``1`` the ratio may come and still
    read as the table's.  Returns ``(a, b, c)`` for each such pair, the two
    positions and the ratio, in the order of the operations' indices, and
    none where every commuting pair commutes as the table says.
    """
    ordered = sorted((int(p) for p in positions), key=lambda p: int(syms[p].ind))
    found = []
    for i, a in enumerate(ordered):
        for b in ordered[i + 1:]:
            if int(prod[a, b]) != int(prod[b, a]) or a not in D or b not in D:
                continue
            rho, res = _proportionality(D[a] @ D[b],
                                        int(spf[a, b]) * int(spf[b, a]) * (D[b] @ D[a]))
            if res <= tol and abs(complex(rho) - 1.0) > tol:
                found.append((a, b, complex(rho)))
    return tuple(found)


def _commuting_pair_phase(D, positions, syms, prod, spf, *, tol=1e-6):
    r"""The first pair :func:`_commuting_pair_phases` returns, or ``None``.

    ``D``, ``positions``, ``syms``, ``prod``, ``spf`` and ``tol`` are as there.
    """
    found = _commuting_pair_phases(D, positions, syms, prod, spf, tol=tol)
    return found[0] if found else None


def _rephasable(D, positions, group, prod, spf, *, tol=1e-6):
    r"""Whether matrices on a subgroup form a representation once rephased.

    ``D`` maps each of ``positions``, positions of the subgroup among the
    operations of ``group``, to a member of that operation's solution space,
    each with its own free phase.  They form a representation with the factor
    system ``spf`` where some phase per operation makes every product
    ``D(g)D(h) = spf(g, h)·D(gh)`` hold, ``prod`` being the multiplication
    table by position.  The question is asked as the representation is
    assembled, by :func:`_rephased`.  ``tol`` is how far a relation may miss
    and still hold, and how far a generator's ``n``-th power may sit from a
    multiple of the identity.  Returns a bool, ``False`` also where no three
    of the subgroup's operations generate it.
    """
    return _rephased(D, positions, group, prod, spf, tol=tol) is not None


def _rephased(D, positions, group, prod, spf, *, tol=1e-6):
    r"""Matrices on a subgroup, rephased into a representation, or ``None``.

    Each generator of the subgroup is put in its normalization
    ``D(g)^n = ζ·1`` and positivity convention (:func:`normalize_rep`), and
    multiplied by the roots of unity that make every relation of the
    multiplication table hold (:func:`_multiplying_roots`).  Every other
    operation is propagated from the generators along their word tree, the
    products taken without conjugation.  ``D`` maps each of ``positions``,
    positions among the operations of ``group``, to a matrix.  ``prod`` is
    the multiplication table by position, and ``spf`` the factor system asked
    for: the group's own spinor factors, or all ones for the trivial one.
    ``tol`` is as in :func:`_rephasable`.  Returns ``{position: matrix}``
    over the subgroup, the identity included.  Returns ``None`` where no
    phases make the matrices a representation with that factor system, or
    where no three of its operations generate the subgroup.
    """
    ident = _identity_position(group)
    positions = sorted({int(p) for p in positions} | {int(ident)})
    gens = _minimal_generators(positions, prod, ident)
    if gens is None:
        return None
    if not gens:
        return {int(ident): np.eye(np.asarray(D[int(ident)]).shape[0], dtype=complex)}
    words, parent = _bfs_words(gens, prod, ident, positions)
    syms = group._symmetries
    records = []
    for gp in gens:
        rec = normalize_rep(D[gp], syms[gp], group, tol=tol, prod=prod, spf=spf)
        if not rec:
            return None
        records.append(rec)
    nstate = np.asarray(D[gens[0]]).shape[0]
    order = sorted(words, key=lambda x: len(words[x]))

    def _propagated(Dgen):
        out = {int(ident): np.eye(nstate, dtype=complex)}
        for q in order:
            if q == ident:
                continue
            gi, r = parent[q]
            out[q] = int(spf[gens[gi], r]) * (Dgen[gi] @ out[r])
        return out

    Dn = _propagated([rec.matrix for rec in records])
    offers = [tuple(complex(lam) / complex(rec.factor) for lam in rec.accepted)
              for rec in records]
    found = _multiplying_roots(Dn, gens, words, parent, positions, prod, spf, offers,
                               tol=tol)
    if not found["projective"] or found["roots"] is None:
        return None
    return _propagated([complex(w) * rec.matrix
                        for w, rec in zip(found["roots"], records)])


def _phase_exponent(n, antiunitary):
    """The power of ``λ`` that ``(λD)`` composed ``n`` times picks up.

    For a unitary operation, ``antiunitary`` false, each factor contributes
    ``λ`` and the power is ``n``.  An antiunitary one alternates ``λ`` and
    ``conj(λ)``, so the power is ``0`` for even ``n`` and ``1`` for odd ``n``.
    An antiunitary operation's order is even, so its ``n``-th power does not
    depend on the phase, which the anchor alone fixes.
    """
    if not antiunitary:
        return n
    return 0 if n % 2 == 0 else 1


def _scalar_of(M):
    """Best scalar ``c`` with ``M ≈ c·1``, and the defect ``max|M − c·1|``."""
    n = M.shape[0]
    c = complex(np.trace(M) / n)
    return c, float(np.abs(M - c * np.eye(n)).max())


def _proportionality(A, B):
    """The least-squares ``rho`` with ``A ≈ rho·B``, and the residual.

    The residual is ``max|A − rho·B|``.  Where ``B`` is zero, ``rho`` is ``0``
    and the residual ``max|A|``.
    """
    denom = float(np.vdot(B, B).real)
    if denom <= 0:
        return 0.0 + 0.0j, float(np.abs(A).max())
    rho = complex(np.vdot(B, A) / denom)
    return rho, float(np.abs(A - rho * B).max())


def _principal_entry(block, atol=1e-9):
    """The entry of ``block`` a positivity condition reads, and its flat index.

    It is the entry of largest modulus, the first in row-major order among
    those within rounding of it.  Returns ``(0, -1)`` where ``block`` is empty
    or no entry's modulus exceeds ``atol``.
    """
    flat = np.asarray(block).ravel()
    if flat.size == 0:
        return 0.0 + 0.0j, -1
    mags = np.abs(flat)
    top = float(mags.max())
    if top <= atol:
        return 0.0 + 0.0j, -1
    idx = int(np.argmax(mags >= top - 1e-12 * max(top, 1.0)))
    return complex(flat[idx]), idx


def _positivity_key(z):
    """The sort key that ranks ``z`` by real part, then by imaginary part.

    Each part is rounded to twelve decimals.
    """
    return (round(float(np.real(z)), 12), round(float(np.imag(z)), 12))


def _table_anchored(candidates, block, internal, *, op_ind, order, zeta, tol=1e-9):
    r"""The candidates making the reference block a real multiple of ``S``.

    ``block`` is the reference block of the operation's matrix, and
    ``internal`` is ``S``, the operation's spin rotation in the tables, or
    ``[[1]]`` without spin (:class:`RepAnchor`).  The reference multiplet
    carries ``+1`` in ``O(g)``, so a matrix ``D(g) = O(g) ⊗ S(g)`` puts ``S``
    in the reference block times one phase ``ρ``.  The tables' spin rotations
    are written in SU(2), which is pseudo-real, so that phase can only be
    real.  ``candidates`` are the phases ``λ`` the normalization allows,
    ``None`` standing for ``(1,)``.  Returns those with ``λρ`` real to
    ``tol``, which counts as at least ``1e-10``.  ``op_ind``, the operation's
    index, and its normalization ``D^order = zeta`` name it in the messages.

    Raises
    ------
    ..refusal.InternalError
        Where ``block`` is not a multiple of ``S``, or no candidate is left.
        The tables fix this phase completely, so either is a defect of this
        package's use of them.
    """
    S = np.asarray(internal, dtype=complex)
    B = np.asarray(block, dtype=complex)
    where = f"operation #{op_ind}"
    if B.shape != S.shape:
        raise InternalError(
            f"{where}: the reference block has shape {B.shape}, and the operation's spin "
            f"rotation S in the tables ([[1]] without spin) has shape {S.shape}. Report it, "
            f"with the model and the call.")
    rho, res = _proportionality(B, S)
    if res > max(tol, 1e-10) or abs(rho) < 0.5:
        raise InternalError(
            f"{where}: the reference block is not a multiple of the operation's spin "
            f"rotation S in the tables ([[1]] without spin). The nearest multiple is "
            f"{rho:.6g} times S, and the block misses it by {res:.3g}, beyond the "
            f"tolerance {max(tol, 1e-10):.3g}. Report it, with the model and the call.\n"
            f"    block = {np.array2string(B, precision=4, max_line_width=200)}\n"
            f"    S(g)  = {np.array2string(S, precision=4, max_line_width=200)}")
    cands = (1.0 + 0.0j,) if candidates is None else tuple(candidates)
    kept = tuple(lam for lam in cands
                 if abs(complex(lam * rho).imag) <= max(tol, 1e-10) * max(abs(lam * rho), 1.0))
    if not kept:
        raise InternalError(
            f"{where}: the normalization D**{order} = {zeta:+d} times the identity allows "
            f"the phases {tuple(f'{c:+.4f}' for c in cands)}, and none of them makes the "
            f"reference block a real multiple of the operation's spin rotation S in the "
            f"tables ([[1]] without spin). The block is {rho:+.6g} times S, and the "
            f"products are "
            f"{tuple(f'{complex(c * rho):+.4f}' for c in cands)}. Report it, with the model "
            f"and the call.")
    return kept


@dataclass(frozen=True)
class RepAnchor:
    """What fixes the phase of an operation's matrix that covariance leaves open.

    :func:`normalize_rep` reads it after the normalization ``(λD)^n = ζ·1``,
    which fixes the phase ``λ`` only up to an ``n``-th root of unity.  Here
    ``n`` is the order of the operation modulo lattice translations, and
    ``ζ``, ``+1`` or ``-1``, the sign the multiplication table of the group
    requires.  The matrix of an antiunitary operation and the operation's
    spin rotation, each where it is given, narrow the roots.  The entry of
    largest modulus of a reference block of the matrix picks among those
    left.  Anchoring a representation at another reference block can
    multiply its matrices by a one-dimensional character of the group.  That
    character is fixed afterwards, where the labeling of the representation
    is chosen.

    Attributes
    ----------
    antiunitary : numpy.ndarray or None
        If not ``None``, the matrix ``T`` of an antiunitary operation of the
        group.  A representation built by :func:`.groups.realized_group` puts
        time reversal here where the group has it, and otherwise another
        antiunitary operation.
    rows : tuple of int
        The rows of the reference block ``D[rows, cols]``: the states of the
        image, under the operation, of the multiplet whose states are
        ``cols``.  Empty to read the whole matrix.
    cols : tuple of int
        The columns of the reference block: the states of one multiplet.
        Empty to read the whole matrix.
    internal : numpy.ndarray or None
        If not ``None``, the operation's spin rotation in the tables,
        ``op.spinor_rotation``, or ``[[1]]`` without spin.  A representation
        built by :func:`.groups.realized_group` gives it only where the group
        has no antiunitary operation, and it then takes the place of
        ``antiunitary``.
    atol : float
        How far the conditions on ``antiunitary`` and ``internal`` may miss,
        and the modulus below which an entry of the matrix is zero.
    """

    antiunitary: np.ndarray | None = None
    rows: tuple = ()
    cols: tuple = ()
    internal: np.ndarray | None = None
    atol: float = 1e-9


@dataclass(frozen=True)
class NormalizedRep:
    """An operation's matrix in its canonical phase, and what fixed the phase.

    The record :func:`normalize_rep` returns.  ``D`` is the matrix handed
    in, ``U`` the unitary matrix nearest it, and ``λU`` the matrix returned.
    ``n`` is the order of the operation modulo lattice translations, and the
    normalization is ``(λU)^n = ζ·1``.  ``T`` is the matrix of an
    antiunitary operation the anchor holds (:class:`RepAnchor`).

    Attributes
    ----------
    matrix : numpy.ndarray
        The matrix in its canonical phase, ``λU``.
    factor : complex
        The phase ``λ`` applied.
    order : int
        The order ``n`` of the operation, modulo lattice translations.
    required : int
        ``ζ``, ``+1`` or ``-1``: the multiple of the identity that the
        multiplication table of the group requires of the ``n``-th power.
    norm_residual : float
        ``max|(λU)^n − ζ·1|``.
    norm_informative : bool
        Whether the normalization narrows the phase.  ``False`` for an
        antiunitary operation, whose order is even, since its ``n``-th power
        does not depend on the phase.
    anchor_residual : float or None
        If not ``None``, ``max|T conj(λU) T† − λU|``.  ``None`` where the
        anchor holds no ``T``.
    anchor_used : bool
        Whether the condition on ``T`` was applied: ``True`` where
        ``T conj(U) T†`` is a multiple of ``U``, even where it removes no root.
    table_anchored : bool
        Whether the operation's spin rotation in the tables narrowed the
        phases.
    product_anchored : bool
        Whether the multiplication table of the group chose the phase among
        the roots the normalization allows.  ``False`` in what
        :func:`normalize_rep` returns: the table chooses only where a whole
        representation is built.
    candidates : tuple of complex
        If nonempty, every ``λ`` the normalization allows.  Empty where it
        allows every phase.
    accepted : tuple of complex
        The ``λ`` left by ``T`` and the spin rotation, among which the
        reference block chose.  ``(1,)`` where the normalization allows every
        phase and nothing narrows it.
    principal_entry : complex
        The entry of the matrix returned on which the choice was read: the
        entry of largest modulus of the reference block, the first in row
        order among equals, or of the whole matrix where that block is zero.
    unitarity : float
        How far ``D`` is from unitary, ``max|D†D − 1|``.
    polar_shift : float
        How far ``U`` is from ``D``, ``max|D − U|``.
    """

    matrix: np.ndarray
    factor: complex
    order: int
    required: int
    norm_residual: float
    norm_informative: bool
    anchor_residual: float | None
    anchor_used: bool
    candidates: tuple
    accepted: tuple
    principal_entry: complex
    unitarity: float
    polar_shift: float
    table_anchored: bool = False
    product_anchored: bool = False

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        anc = "-" if self.anchor_residual is None else f"{self.anchor_residual:.1e}"
        return (f"NormalizedRep(factor={self.factor:.4f}, D^{self.order}="
                f"{self.required:+d}, norm={self.norm_residual:.1e}, anchor={anc})")


def normalize_rep(D, op, group, *, spinful=None, anchor=None, tol=1e-9,
                  prod=None, spf=None) -> NormalizedRep | Refusal:
    r"""An operation's matrix in its canonical phase, and what fixed the phase.

    A matrix carrying ``H(k)`` onto ``H(Rk)`` does so whatever its overall
    phase, and a wrong phase relabels every little-group representation at
    once.  The unitary matrix ``U`` nearest ``D`` is taken, and its phase
    ``λ`` is fixed in three steps.  First, the normalization
    ``(λU)^n = ζ·1`` fixes ``λ`` up to an ``n``-th root of unity.  Here ``n``
    is the order of the operation modulo lattice translations, and ``ζ`` is
    the sign the multiplication table of the group requires.  For an
    antiunitary operation, whose order is even, ``(λU)^n`` does not depend
    on ``λ``, and the normalization leaves ``λ`` open.  Second, each
    condition the anchor holds narrows the roots.  The matrix ``T`` of an
    antiunitary operation, ``anchor.antiunitary``, keeps those with
    ``T conj(λU) T† = λU``, which fixes ``λ`` up to a sign.  Where
    ``T conj(U) T†`` is not a multiple of ``U``, ``T`` keeps them all.  The
    spin rotation ``S``, ``anchor.internal``, keeps those that make ``λ``
    times the reference block ``U[anchor.rows, anchor.cols]`` a real
    multiple of ``S``.  Third, the ``λ`` returned is the root left that gives
    the entry of largest modulus of the reference block, the first in row
    order among equals, the largest real part, a tie going to the larger
    imaginary part.  Where the reference block is zero, the spin rotation is
    not used, and the entry of largest modulus of the whole matrix is read
    instead.  Matrices normalized one by one need not multiply as the group
    does.  A whole representation normalizes its generators and takes the
    other matrices as their products.

    Parameters
    ----------
    D : array, shape (nstate, nstate)
        The matrix of the operation on the states.
    op : irrep.symmetry_operation.SymmetryOperation
        The operation, one of ``group``'s.
    group : .groups.GroupData
        The group, ``r.group``.
    spinful : bool or None, default None
        Whether the model is spinful, checked against ``group.spinor``.  Leave
        it ``None`` to skip the check.
    anchor : RepAnchor or None, default None
        The matrix of an antiunitary operation, the reference block and the
        spin rotation.  Leave it ``None`` to read the entry of largest modulus
        of the whole matrix alone.
    tol : float, default 1e-9
        How far the ``n``-th power of ``U`` may sit from a multiple of the
        identity, in its largest entry.  A value below ``1e-10`` counts as
        ``1e-10``.
    prod : numpy.ndarray or None, default None
        The multiplication table of the group, by position.  Pass it with
        ``spf`` when normalizing many operations of one group, so that the
        table is computed once.  Where either is ``None``, both are computed.
    spf : numpy.ndarray or None, default None
        The spinor factor of each product in ``prod``: the sign, ``+1`` or
        ``-1``, with which the spin rotations of the two operations multiply
        to that of their product.

    Returns
    -------
    NormalizedRep or Refusal
        ``matrix`` is ``λ`` times the unitary matrix nearest ``D``, and
        ``factor`` is ``λ``.  The record also holds the residual of each
        condition for the matrix returned.  A ``Refusal`` returned in its
        place tests false.

    Examples
    --------
    The fourfold rotation of lin3g (:func:`.models.lin3g`), its matrix ``D``
    given the wrong phase ``-i``, with ``anchor`` holding the matrix of time
    reversal and the reference block ``rows=(0, 1), cols=(0, 1)``.  The phase
    is undone:

    >>> rec = normalize_rep(-1j * D, op, group, anchor=anchor)    # doctest: +SKIP
    >>> abs(rec.factor - 1j) < 1e-12                              # doctest: +SKIP
    True
    >>> np.allclose(rec.matrix, D)                                # doctest: +SKIP
    True

    Refusals
    --------
    ``canonical_norm``
        ``U^n`` is not a multiple of the identity.
    ``antiunitary_anchor``
        no root the normalization allows satisfies ``T conj(λU) T† = λU``.

    Raises
    ------
    ValueError
        ``D`` is not square, ``op`` is not one of ``group``'s, or ``spinful``
        disagrees with ``group.spinor``.
    """
    D = np.asarray(D, dtype=complex)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"normalize_rep(): D has shape {D.shape}; pass the square matrix "
                         "of the operation on the states.")
    if spinful is not None and bool(spinful) != bool(group.spinor):
        raise ValueError(f"normalize_rep(): spinful={spinful}, and group.spinor is "
                         f"{group.spinor}. Pass spinful={bool(group.spinor)}, or leave it "
                         "None.")

    pos = {int(s.ind): i for i, s in enumerate(group._symmetries)}
    if int(op.ind) not in pos:
        raise ValueError(f"normalize_rep(): op has index {op.ind}, which is not the index "
                         "of an operation of group. Pass op from group's own operations.")
    p = pos[int(op.ind)]
    anti = bool(op.time_reversal)

    eye = np.eye(D.shape[0])
    unitarity = float(np.abs(D.conj().T @ D - eye).max())
    w, _, vh = np.linalg.svd(D)
    U = w @ vh
    polar_shift = float(np.abs(D - U).max())

    if prod is None or spf is None:
        prod, _, spf = group._get_product_table(get_diff=True)
    n, zeta = _order_and_zeta(group, p, prod=prod, spf=spf)
    p_exp = _phase_exponent(n, anti)

    c, defect = _scalar_of(_power(U, anti, n))
    if defect > max(tol, 1e-10) or abs(c) < 0.5:
        return Refusal(
            "canonical_norm",
            f"operation #{op.ind}, of order {n}: for U the unitary matrix nearest D, "
            f"U**{n} is not a multiple of the identity. "
            f"The multiple nearest it, c times the identity with |c| = {abs(c):.3g}, "
            f"differs from it by {defect:.3g} in its largest entry.",
            value=defect, threshold=float(max(tol, 1e-10)),
            context={"operation_ind": int(op.ind), "order": n, "required": zeta,
                     "scalar": c, "antiunitary": anti})

    if p_exp == 0:
        candidates = None                      # phase-blind: every lam allowed
    else:
        base = complex(zeta / c) ** (1.0 / p_exp)
        candidates = tuple(base * np.exp(2j * np.pi * j / p_exp) for j in range(p_exp))

    anchor = anchor if anchor is not None else RepAnchor()
    accepted, anchor_used, anchor_residual = candidates, False, None
    T = None if anchor.antiunitary is None else np.asarray(anchor.antiunitary, dtype=complex)
    if T is not None:
        rho, prop = _proportionality(T @ np.conj(U) @ T.conj().T, U)
        if prop <= max(anchor.atol, 1e-10) and abs(rho) > 0.5:
            #   T conj(lam U) T† = conj(lam) rho U = (conj(lam) rho / lam)(lam U)
            #   so the anchor holds exactly when lam**2 = rho.
            roots = (np.sqrt(complex(rho)), -np.sqrt(complex(rho)))
            if candidates is None:
                accepted = roots
            else:
                accepted = tuple(lam for lam in candidates
                                 if min(abs(lam - r) for r in roots) <= 1e-6)
            anchor_used = True
            if not accepted:
                return Refusal(
                    "antiunitary_anchor",
                    f"operation #{op.ind}, of order {n}: for U the unitary matrix nearest "
                    f"D, no phase λ gives both (λU)**{n} = {zeta:+d} times the identity "
                    f"and T conj(λU) T† = λU, for T the matrix of time reversal, or of "
                    f"another antiunitary operation where the group has none. The second "
                    f"holds only for λ**2 = {rho:.6g}.",
                    value=float(min(abs(lam - r) for lam in candidates for r in roots)),
                    threshold=1e-6,
                    context={"operation_ind": int(op.ind), "order": n, "required": zeta,
                             "rho": rho, "candidates": candidates, "roots": roots})
        else:
            anchor_residual = prop      # anchor present but not proportional: report it

    rows = np.asarray(anchor.rows, dtype=int)
    cols = np.asarray(anchor.cols, dtype=int)
    block = U[np.ix_(rows, cols)] if (rows.size and cols.size) else U

    # The table anchor: what the operation's spin rotation in the tables still
    # fixes.  It is applied where the reference block has the tensor structure,
    # that is where the block is nonzero.  A candidate whose reference block
    # vanishes is the g^-1 branch of an operation that moves the reference
    # multiplet, which the check that each block is a multiple of the spin
    # rotation, and the product table, discard.
    table_anchored = False
    if anchor.internal is not None and float(np.abs(block).max()) > max(anchor.atol, 1e-10):
        narrowed = _table_anchored(accepted, block, anchor.internal, op_ind=int(op.ind),
                                   order=n, zeta=zeta, tol=max(anchor.atol, 1e-10))
        table_anchored = accepted is None or len(narrowed) < len(accepted)
        accepted = narrowed
    if accepted is None:                # phase-blind and no usable anchor
        accepted = (1.0 + 0.0j,)

    entry, idx = _principal_entry(block, atol=anchor.atol)
    if idx < 0:
        entry, idx = _principal_entry(U, atol=anchor.atol)
    lam = max(accepted, key=lambda l: _positivity_key(l * entry))
    if anchor.internal is not None:
        tied = [l for l in accepted
                if _positivity_key(l * entry) == _positivity_key(lam * entry)]
        if len(tied) > 1:  # pragma: no cover - a guard: two distinct phases cannot
            #                 agree on a nonzero entry, and the entry is nonzero here
            raise InternalError(
                f"operation #{op.ind}: the spin rotation in the tables leaves {len(tied)} "
                f"phases {tuple(f'{c:+.4f}' for c in tied)}, and their products with the "
                f"largest entry of the reference block, {entry:+.4g}, are equal. Report "
                f"it, with the model and the call.")

    Un = lam * U
    norm_residual = float(np.abs(_power(Un, anti, n) - zeta * eye).max())
    if T is not None:
        anchor_residual = float(np.abs(T @ np.conj(Un) @ T.conj().T - Un).max())

    return NormalizedRep(
        matrix=Un, factor=complex(lam), order=n, required=int(zeta),
        norm_residual=norm_residual, norm_informative=bool(p_exp),
        anchor_residual=anchor_residual, anchor_used=anchor_used,
        candidates=tuple(candidates) if candidates is not None else (),
        accepted=tuple(accepted), principal_entry=complex(lam * entry),
        unitarity=unitarity, polar_shift=polar_shift,
        table_anchored=bool(table_anchored))


def _propagated(Dgen, DT, *, gens, words, parent, spf, ident, nstate, T_pos=None,
                au_of=None):
    r"""Every operation's matrix, propagated along the generators' word tree.

    The multiplication table gives ``D(g_i)·D(r) = spf(g_i, r)·D(g_i r)``, and
    ``spf`` is ``+1`` or ``-1``, so each edge ``q = g_i·r`` of the tree sets
    ``D(q) = spf(g_i, r)·D(g_i)·D(r)``.  An antiunitary operation ``p = T·u``
    is set to ``spf(T, u)·D(T)·conj(D(u))``.

    Parameters
    ----------
    Dgen : sequence of numpy.ndarray
        The matrix of each generator, in the order of ``gens``.
    DT : numpy.ndarray or None
        The matrix of the antiunitary operation ``T``, read only where
        ``T_pos`` is given.
    gens : sequence of int
        The positions of the generators, indices into the group's operations.
    words : dict
        Each unitary operation's word in the generators, by position
        (:func:`_bfs_words`).
    parent : dict
        The edge ``(i, r)`` of the word tree by which each unitary operation
        ``q`` is reached, ``q = g_i·r``.
    spf : numpy.ndarray of int
        The spinor factor, ``+1`` or ``-1``, of each product, by position.
    ident : int
        The position of the identity.
    nstate : int
        The number of states, the size of each matrix.
    T_pos : int or None, default None
        The position of the antiunitary operation ``T`` the others are
        written from.  Leave it ``None`` where none is represented.
    au_of : dict or None, default None
        The position of the unitary ``u`` with ``p = T·u``, for each
        antiunitary operation's position ``p``.  Read only where ``T_pos`` is
        given.

    Returns
    -------
    dict
        The matrix of the identity, of each operation of ``words`` and of
        each antiunitary operation of ``au_of``, by position.
    """
    D = {ident: np.eye(nstate, dtype=complex)}
    for q in sorted(words, key=lambda x: len(words[x])):
        if q == ident:
            continue
        gi, r = parent[q]
        D[q] = int(spf[gens[gi], r]) * (Dgen[gi] @ D[r])
    if T_pos is not None:
        for p, u in (au_of or {}).items():
            D[p] = int(spf[T_pos, u]) * (DT @ np.conj(D[u]))
    return D


def _settled(Dsearch, *, group, gens, words, parent, kept, prod, spf, ident, nstate,
             anchor_of, reach, T_pos=None, au_of=None):
    r"""A representation's matrices, put in their canonical phase.

    Each generator, and the antiunitary operation ``T`` where there is one,
    is put in its normalization and the phase its anchor fixes
    (:func:`normalize_rep`), and every other operation is propagated from
    them (:func:`_propagated`).  What is left open is a one-dimensional
    character of the group: a real one where ``T`` anchors the phase, and a
    root of unity per generator where nothing does, which
    :func:`_table_phases` narrows.  Matrices built from a near-symmetric
    model's own blocks compose by the multiplication table only to the
    model's noise, so no normalization can be read off them.  Where their
    composition defect (:func:`_composition_defect`) exceeds
    ``_COMPOSITION_GATE``, ``1e-9``, and is at most ``reach``, they are first
    replaced by the exact representation nearest them
    (:func:`_nearest_representation`), where it converges.

    Parameters
    ----------
    Dsearch : dict
        The matrix of each operation of ``kept``, by position, propagated
        along the word tree, each generator with the free phase of the
        solution space it was read from.
    group : .._spacegroup.GroupData
        The group.
    kept : sequence of int
        The positions of the operations represented.
    prod : numpy.ndarray of int
        The multiplication table of the group, by position.
    anchor_of : callable
        ``anchor_of(p, anchor_T)``, the :class:`RepAnchor` of the operation at
        position ``p``, given ``anchor_T``, the matrix of ``T`` or ``None``.
    reach : float
        The largest composition defect, in the largest entry of a residual,
        that the nearest representation is sought for.
    gens, words, parent, spf, ident, nstate, T_pos, au_of
        The word tree, as :func:`_propagated` takes it.

    Returns
    -------
    tuple
        ``(refusal, D, normalization, anchor_T, DT_n, defect)``.  ``refusal``
        is the refusal :func:`normalize_rep` returned for a generator or
        ``T``, the four after it then ``None``, and ``None`` otherwise.
        ``D`` is the matrix of each operation, by position.
        ``normalization`` is the :class:`NormalizedRep` of each generator and
        of ``T``, by operation index.  ``anchor_T`` is the matrix of ``T`` the
        anchors were read with, and ``DT_n`` that matrix in its canonical
        phase, both ``None`` without ``T``.
        ``defect`` is the composition defect of ``Dsearch`` as handed in.
    """
    syms = group._symmetries
    anchors = list(gens) + ([T_pos] if T_pos is not None else [])
    defect = _composition_defect(Dsearch, anchors, kept, syms, prod)
    if _COMPOSITION_GATE < defect <= reach:
        nearest = _nearest_representation(Dsearch, kept, syms, prod)
        if nearest is not None:
            Dsearch = nearest
    anchor_T = Dsearch[T_pos] if T_pos is not None else None
    normalization, Dgen_n = {}, []
    for gp in gens:
        rec = normalize_rep(Dsearch[gp], syms[gp], group, anchor=anchor_of(gp, anchor_T),
                            prod=prod, spf=spf)
        if not rec:
            return rec, None, None, None, None, defect
        normalization[int(syms[gp].ind)] = rec
        Dgen_n.append(rec.matrix)
    DT_n = None
    if T_pos is not None:
        rec = normalize_rep(Dsearch[T_pos], syms[T_pos], group,
                            anchor=anchor_of(T_pos, anchor_T), prod=prod, spf=spf)
        if not rec:
            return rec, None, None, None, None, defect
        normalization[int(syms[T_pos].ind)] = rec
        DT_n = rec.matrix
    D = _propagated(Dgen_n, DT_n, gens=gens, words=words, parent=parent, spf=spf,
                    ident=ident, nstate=nstate, T_pos=T_pos, au_of=au_of)
    return None, D, normalization, anchor_T, DT_n, defect


def _table_phases(Dn, normalization, DT_n, *, group, gens, words, parent, positions, prod,
                  spf, ident, nstate, T_pos=None, au_of=None):
    r"""The generators' phases the multiplication table chooses, applied.

    Positivity on the reference block picks each generator's phase among the
    roots of unity its normalization and anchor leave.  The phases it picks
    need not multiply in two cases.  In a group with no antiunitary operation
    and a reference multiplet of several orbitals, nothing else narrows the
    roots.  A reference multiplet can carry a character on its site group
    that no one-dimensional character of the group restricts to, as ``p_y``
    at a ``3c`` site of ``P23`` does.  The table then chooses among the roots
    each norm allows, in positivity's order, so that a choice that already
    multiplies is the one kept (:func:`_multiplying_roots`).

    Parameters
    ----------
    Dn : dict
        The matrices :func:`_settled` returns, by position.
    normalization : dict
        The :class:`NormalizedRep` of each generator, by operation index, as
        :func:`_settled` returns it.
    DT_n : numpy.ndarray or None
        The matrix of the antiunitary operation ``T`` in its canonical
        phase, ``None`` without one.
    group : .._spacegroup.GroupData
        The group.
    positions : iterable of int
        The unitary operations each generator's relations are read against.
    prod : numpy.ndarray of int
        The multiplication table of the group, by position.
    gens, words, parent, spf, ident, nstate, T_pos, au_of
        The word tree, as :func:`_propagated` takes it.

    Returns
    -------
    tuple
        ``(found, rephased)``.  ``found`` is the record
        :func:`_multiplying_roots` returns.  ``rephased`` is
        ``(D, normalization)`` with the roots found multiplied in, each
        rephased generator's record marked ``product_anchored``; it is
        ``None`` where no roots were found, or every root found is ``1``.
    """
    syms = group._symmetries
    offers = []
    for gp in gens:
        r = normalization[int(syms[gp].ind)]
        entry = complex(r.principal_entry) / complex(r.factor)
        ranked = sorted(r.accepted, key=lambda lam: _positivity_key(complex(lam) * entry),
                        reverse=True)
        offers.append(tuple(complex(lam) / complex(r.factor) for lam in ranked))
    found = _multiplying_roots(Dn, gens, words, parent, positions, prod, spf, offers)
    roots = found["roots"]
    if roots is None or all(abs(w - 1.0) <= 1e-12 for w in roots):
        return found, None
    Dgen = [roots[gi] * Dn[gp] for gi, gp in enumerate(gens)]
    D2 = _propagated(Dgen, DT_n, gens=gens, words=words, parent=parent, spf=spf,
                     ident=ident, nstate=nstate, T_pos=T_pos, au_of=au_of)
    norm2 = dict(normalization)
    for gi, gp in enumerate(gens):
        if abs(roots[gi] - 1.0) <= 1e-12:
            continue
        ind = int(syms[gp].ind)
        r = normalization[ind]
        M = roots[gi] * np.asarray(r.matrix)
        norm2[ind] = replace(
            r, matrix=M, factor=complex(roots[gi] * r.factor),
            principal_entry=complex(roots[gi] * r.principal_entry),
            norm_residual=float(np.abs(_power(M, False, r.order)
                                       - r.required * np.eye(nstate)).max()),
            product_anchored=True)
    return found, (D2, norm2)


def _settle_unit(Dsearch, *, group, tree, kept, u_pos, prod, anchor_of, reach,
                 product_tol=1e-9):
    r"""A representation's matrices settled in their canonical phases.

    Each generator is put in its normalization and anchor and the rest are
    propagated (:func:`_settled`).  Where the matrices then miss the
    multiplication table by more than ``product_tol``, the table chooses the
    generators' phases among the roots each normalization allows
    (:func:`_table_phases`).  Every representation that
    :func:`.._build._build_rep` builds, and every one a declaration builds,
    is settled here.

    Parameters
    ----------
    Dsearch : dict
        The matrix of each operation of ``kept``, by position, as
        :func:`_settled` takes it.
    group : .._spacegroup.GroupData
        The group.
    tree : dict
        The word tree: ``gens``, ``words``, ``parent``, ``spf``, ``ident``,
        ``nstate``, ``T_pos`` and ``au_of``, as :func:`_propagated` takes
        them.
    kept : sequence of int
        The positions of the operations represented.
    u_pos : sequence of int
        The positions of the unitary operations among them.
    prod : numpy.ndarray of int
        The multiplication table of the group, by position.
    anchor_of : callable
        The anchor of each generator, as :func:`_settled` takes it.
    reach : float
        The largest composition defect repaired, as :func:`_settled` takes it.
    product_tol : float, default 1e-9
        How far the matrices may miss the multiplication table, as
        :func:`_product_table_residual` measures it, and count as multiplying.

    Returns
    -------
    tuple
        ``(refusal, D, normalization, anchor_T, DT_n, defect, roots)``, the
        first six as :func:`_settled` returns them, ``D`` and
        ``normalization`` rephased where the table chose.  ``roots`` is
        ``(found, D)`` where no root makes the matrices multiply, ``found``
        the record :func:`_multiplying_roots` returns, and ``None``
        otherwise.
    """
    syms, spf = group._symmetries, tree["spf"]
    bad, D, norm, anchor_T, DT_n, defect = _settled(
        Dsearch, group=group, kept=kept, prod=prod, anchor_of=anchor_of, reach=reach, **tree)
    if bad is not None:
        return bad, None, None, None, None, defect, None
    roots = None
    if _product_table_residual(D, kept, syms, prod, spf) > product_tol:
        found, rephased = _table_phases(D, norm, DT_n, group=group, positions=u_pos,
                                        prod=prod, **tree)
        if rephased is not None:
            D, norm = rephased
        if _product_table_residual(D, kept, syms, prod, spf) > product_tol:
            roots = (found, D)
    return None, D, norm, anchor_T, DT_n, defect, roots


# --------------------------------------------------------------------------
# the Bloch symmetry operator
# --------------------------------------------------------------------------

def _orb_positions_3d(model, group):
    """The positions of the orbitals of ``model`` in the 3D cell of ``group``.

    The cell is the one ``group`` was identified in: for a layer, its cell
    with vacuum along the directions it does not repeat in
    (:func:`.._spacegroup._pad_cell`).  Returns an array of shape
    ``(norb, 3)``, in reduced coordinates.
    """
    _lat, pos3, _padded = _spacegroup._pad_cell(model, group._vacuum_length,
                                            lattice=group._lattice)
    return np.asarray(pos3, dtype=float)


def _translation_phase(group, op, k):
    r"""The translation's phase :math:`e^{-2\pi i\,k'\cdot t}` of ``g = {A|t}`` at each ``k``.

    ``k'`` is the image of ``k`` under the operation, :math:`(A^{-1})^{T}k`
    and its negative for an operation combined with time reversal, not folded
    back into the zone.  The Bloch operator there (:func:`bloch_operator` at
    ``k_out = k'``) is this phase times the operation's matrix ``D(g)``, every
    orbital's phase being 1.

    Parameters
    ----------
    group : GroupData
        The model's group.
    op : irrep SymmetryOperation
        The operation.
    k : array, shape (dim_k,) or (Nk, dim_k)
        Reduced ``k`` in the model's periodic directions.

    Returns
    -------
    numpy.ndarray
        The phase, shape ``(Nk,)``, and ``(1,)`` for a single ``k``.
    """
    k_arr = np.atleast_2d(np.asarray(k, dtype=float))
    k3 = group._k_to_3d(k_arr)
    kp3 = np.asarray(op.transform_k(k3), dtype=float)
    trans = np.asarray(op.translation, dtype=float)
    return np.exp(-2j * np.pi * (kp3 @ trans))


def bloch_operator(model, group, assignment, op, D, k, *, k_out=None, tol=1e-9,
                   check=True):
    r"""The Bloch operator ``U(g;k)`` of one operation, at one k-point or several.

    The operator satisfies ``U(g;k) H(k) U(g;k)† = H(k_out)``, with ``H(k)*``
    in place of ``H(k)`` for an operation combined with time reversal.  In
    pythtb's convention the Bloch basis carries
    :math:`e^{2\pi i k\cdot(\mathbf R + \tau)}`, so for :math:`g = \{A|t\}`

    .. math::
        U(g;k) = e^{-2\pi i\,k'\cdot t}\;
                 \operatorname{diag}\!\left(e^{2\pi i\,(k'-k_{\rm out})\cdot
                 \tau_{j}}\right)\, D(g),
        \qquad k' = (A^{-1})^{T}k,

    with ``τ_j`` the position of orbital ``j``, and ``k'`` the negative of
    that for an operation combined with time reversal.  At ``k_out = k'``,
    the image not folded back into the zone, the diagonal is the identity,
    and ``U(g;k)`` is ``D(g)`` times the phase ``e^{−2πi k'·t}``.  The
    orbital positions enter where ``k_out`` differs from ``k'`` by a nonzero
    reciprocal lattice vector.  So they enter where a sewing matrix is read,
    with ``k_out = k``, at a ``k`` the operation carries to ``k`` plus a
    nonzero reciprocal lattice vector.

    The phase ``e^{−2πi k'·t}`` cancels in ``U H U†``.  It carries the
    translations into the products of the operators.  For ``g1 = {A1|t1}``,
    ``g2 = {A2|t2}`` and their product ``g1g2 = {A3|t3}``, with ``t3`` the
    translation of the operation listed for it and ``g2k`` the image of ``k``
    under ``g2``,
    ``U(g1;g2k) U(g2;k) = s·e^{−2πi (g1g2k)·Δ}·U(g1g2;k)``.  Here
    ``Δ = t1 + A1t2 − t3``, and ``s`` is the sign, ``+1`` or ``-1``, with
    which the spin rotations of ``g1`` and ``g2`` multiply to that of
    ``g1g2``.  Where ``g1`` is antiunitary, ``U(g2;k)`` is complex
    conjugated.  The little-group characters of a glide or a screw rest on
    these phases.

    Parameters
    ----------
    model : pythtb.TBModel
        The model the representation was built on.
    group : .groups.GroupData
        Its group, ``r.group``.
    assignment : OrbitalAssignment
        The Wyckoff orbits of the orbitals, the permutation of the sites by
        each operation, and, for each operation and site, the lattice vector
        by which the site's image differs from the site of the cell it lands
        on.
    op : irrep.symmetry_operation.SymmetryOperation
        The operation, one of ``group``'s.
    D : array, shape (nstate, nstate)
        The matrix of the operation, ``r.rep.matrices[op.ind]``.
    k : array, shape (dim_k,) or (Nk, dim_k)
        One k-point or several, in reduced coordinates of the model's periodic
        directions.
    k_out : array or None, default None
        Where the transformed Hamiltonian is read, a k-point that differs from
        ``k'`` by a reciprocal lattice vector.  Pass ``k`` itself at a ``k``
        the operation carries to itself up to a reciprocal lattice vector.
        Leave it ``None`` to read it at ``k'``.
    tol : float, default 1e-9
        How far, in reduced coordinates, ``k' − k_out`` may sit from a
        reciprocal lattice vector, and the image of an orbital from the
        orbital it is carried to.  ``tol``, at least ``1e-12``, times the
        larger of one and the largest modulus of an entry of ``D`` is also the
        modulus below which an entry of ``D`` is zero.
    check : bool, default True
        Whether ``k' − k_out`` and the images of the orbitals are checked
        against ``tol``.  With ``check=False``, ``k' − k_out`` is rounded to a
        reciprocal lattice vector unchecked.

    Returns
    -------
    numpy.ndarray
        ``U(g;k)``, of shape ``(nstate, nstate)`` for one k-point and
        ``(Nk, nstate, nstate)`` for several.

    Raises
    ------
    ValueError
        ``k`` does not have one component for each periodic direction, the
        operation carries ``k`` out of the model's periodic plane, or a block
        of rows of ``D`` that belongs to one orbital is zero.  Unless
        ``check=False``, also where ``k' − k_out`` is not a reciprocal lattice
        vector, or ``D`` has a nonzero entry between two orbitals of which the
        operation does not carry one onto the other.

    See Also
    --------
    SpaceGroupRep.bloch_operator : the same operator, for the operation of a
        given index in ``r.rep``.
    """
    D = np.asarray(D, dtype=complex)
    k_arr = np.asarray(k, dtype=float)
    squeeze = (k_arr.ndim == 1)
    k_arr = np.atleast_2d(k_arr)
    if k_arr.shape[1] != group._dim_k:
        n = int(k_arr.shape[1])
        raise ValueError(f"bloch_operator(): k has {n} component{'' if n == 1 else 's'}; "
                         f"pass {group._dim_k}, one for each periodic direction.")

    k3 = group._k_to_3d(k_arr)
    kp3 = np.asarray(op.transform_k(k3), dtype=float)
    if not group._in_plane(kp3):
        raise ValueError(f"bloch_operator(): operation #{op.ind} carries k out of the "
                         "model's periodic plane.")
    if k_out is None:
        kout3 = kp3
    else:
        kout3 = group._k_to_3d(np.atleast_2d(np.asarray(k_out, dtype=float)))
        if kout3.shape[0] == 1 and kp3.shape[0] > 1:
            kout3 = np.repeat(kout3, kp3.shape[0], axis=0)
    n3 = kp3 - kout3
    if check and np.abs(n3 - np.rint(n3)).max() > tol:
        raise ValueError(f"bloch_operator(): the image of k minus k_out is {n3}, which is "
                         "not a reciprocal lattice vector; it misses one by "
                         f"{np.abs(n3 - np.rint(n3)).max():.3g}. Pass a k_out equal to the "
                         "image of k up to a reciprocal lattice vector.")
    n3 = np.rint(n3)

    nspin = int(model.nspin)
    norb = int(model.norb)
    pos3 = _orb_positions_3d(model, group)
    rot = np.asarray(op.rotation, dtype=float)
    trans = np.asarray(op.translation, dtype=float)
    p = next(i for i, s in enumerate(group._symmetries) if int(s.ind) == int(op.ind))
    site_of = np.asarray(assignment.site_of_orbital, dtype=int)

    # each row's phase is read off the image of its preimage orbital
    preimage = []
    thresh = max(tol, 1e-12) * max(1.0, float(np.abs(D).max()))
    for b in range(norb):
        block = D[b * nspin:(b + 1) * nspin, :]
        cols = np.flatnonzero(np.abs(block).max(axis=0) > thresh)
        if cols.size == 0:
            raise ValueError(f"bloch_operator(): the rows of D for orbital {b} are all "
                             f"zero. Pass the matrix of operation #{op.ind}, "
                             f"r.rep.matrices[{op.ind}].")
        images = [rot @ pos3[int(c) // nspin] + trans for c in cols]
        base = images[0]
        for img in images[1:]:
            d = img - base
            if check and np.abs(d - np.rint(d)).max() > tol:
                raise ValueError(f"bloch_operator(): the rows of D for orbital {b} draw on "
                                 f"orbitals whose images under operation #{op.ind} differ "
                                 f"by {d}, which is not a lattice vector. Call "
                                 f"r.rep.bloch_operator({op.ind}, k), which passes the "
                                 "matrix and the assignment r.rep was built on.")
        preimage.append(int(cols[0]) // nspin)
        if check:
            a = int(cols[0]) // nspin
            shift = np.asarray(assignment.lattice_shift[p, site_of[a]], dtype=float)
            resid = base - shift - pos3[b]
            if np.abs(resid - np.rint(resid)).max() > tol:
                raise ValueError(
                    f"bloch_operator(): the rows of D for orbital {b} draw on orbital {a}, "
                    f"which operation #{op.ind} does not carry onto orbital {b}: its "
                    f"image, less the lattice vector the assignment gives, differs from "
                    f"orbital {b} by {resid}. Call r.rep.bloch_operator({op.ind}, k), "
                    "which passes the matrix and the assignment r.rep was built on.")

    glob = np.exp(-2j * np.pi * (kp3 @ trans))
    if n3.any():
        lam = np.exp(2j * np.pi * _exact_dot(n3, _image_split(rot, trans, pos3, preimage)))
    else:
        lam = np.ones((n3.shape[0], norb), dtype=complex)    # k_out = k'
    lam = np.repeat(lam, nspin, axis=1)                      # (Nk, nstate)
    U = glob[:, None, None] * lam[:, :, None] * D[None, :, :]
    return U[0] if squeeze else U


def _image_positions(rot, trans, pos3, preimage) -> list:
    r"""The image ``A·τ_a + t`` of each preimage ``a``, in exact rationals.

    ``rot`` and ``trans`` are the operation's ``A`` and ``t``, and ``pos3``
    the orbitals' positions ``τ`` in the group's 3D cell, in reduced
    coordinates.  ``preimage`` holds, for each orbital, the orbital ``a`` its
    rows of the operation's matrix draw on.  Each stored float is read as the
    binary number it is, so the image is formed with no rounding.  Returns
    one list of three ``Fraction`` for each entry of ``preimage``.
    """
    A = [[Fraction(float(x)) for x in row] for row in np.asarray(rot, dtype=float)]
    t = [Fraction(float(x)) for x in np.asarray(trans, dtype=float)]
    seen = {}
    for a in set(preimage):
        tau = [Fraction(float(x)) for x in pos3[a]]
        seen[a] = [t[i] + sum(A[i][j] * tau[j] for j in range(len(tau))) for i in range(len(t))]
    return [seen[a] for a in preimage]


#: The power of two :func:`_image_split` splits an exact position at: integers
#: dotted with multiples of ``2**-26`` give a binary number of at most 53 bits
#: while the sum of the products' magnitudes stays under ``2**27``.
_DOT_SPLIT = 2 ** 26


def _image_split(rot, trans, pos3, preimage) -> tuple:
    r"""The images :func:`_image_positions` gives, split for :func:`_exact_dot`.

    ``rot``, ``trans``, ``pos3`` and ``preimage`` are as in
    :func:`_image_positions`, and the split is kept for a repeated call
    (:func:`_image_split_of`).  Returns two read-only arrays, each of shape
    ``(len(preimage), 3)``: the multiple of ``2**-26`` at or below each exact
    coordinate, and the rest.
    """
    arrays = tuple((x.tobytes(), x.shape) for x in
                   (np.ascontiguousarray(y, dtype=float) for y in (rot, trans, pos3)))
    return _image_split_of(arrays, tuple(int(a) for a in preimage))


@functools.lru_cache(maxsize=512)
def _image_split_of(arrays, preimage) -> tuple:
    """:func:`_image_split` on arguments that can be cached.

    ``arrays`` holds the bytes and the shape of ``rot``, ``trans`` and
    ``pos3``, in that order, and ``preimage`` holds the preimage orbitals.
    """
    rot, trans, pos3 = (np.frombuffer(b, dtype=float).reshape(s) for b, s in arrays)
    positions = _image_positions(rot, trans, pos3, preimage)
    split = [[Fraction(math.floor(x * _DOT_SPLIT), _DOT_SPLIT) for x in p] for p in positions]
    hi = np.array([[float(h) for h in p] for p in split])
    lo = np.array([[float(x - h) for x, h in zip(p, q)] for p, q in zip(positions, split)])
    hi.flags.writeable = False
    lo.flags.writeable = False
    return hi, lo


def _exact_dot(n, parts) -> np.ndarray:
    r"""``n · x`` for integer vectors ``n`` and exact rationals ``x``, formed exactly and rounded once.

    Each ``x`` is split into the multiple of ``2**-26`` at or below it and the
    rest.  The first part's dot product with ``n`` is exact in floating point;
    the second part is under ``2**-26``, and its product rounds far below the
    one rounding of the sum.

    Parameters
    ----------
    n : array, shape (Nk, d)
        Integer vectors, stored as floats.
    parts : tuple of two numpy.ndarray
        The two parts of the positions, each of shape ``(norb, d)``, as
        :func:`_image_split` returns them, ``sum_i |n_i x_i|`` under ``2**27``.

    Returns
    -------
    numpy.ndarray
        Shape ``(Nk, norb)``.
    """
    hi, lo = parts
    n = np.asarray(n, dtype=float)
    return n @ hi.T + n @ lo.T


# --------------------------------------------------------------------------
# generators and words, and what an operation does to the orbitals
# --------------------------------------------------------------------------

def _closure(gens, prod, ident):
    """The subgroup generated by ``gens``, as a set of positions.

    ``gens`` are positions, indices into the group's operations, ``prod`` is
    the multiplication table by position, and ``ident`` is the position of
    the identity, which the set always holds.
    """
    cur = {int(ident)} | {int(g) for g in gens}
    changed = True
    while changed:
        changed = False
        for a in list(cur):
            for b in list(cur):
                c = int(prod[a, b])
                if c not in cur:
                    cur.add(c)
                    changed = True
    return cur


def _minimal_generators(positions, prod, ident, max_size=3, max_combos=200_000):
    """The fewest operations of ``positions`` that generate all of them.

    ``positions`` must be closed under ``prod``, the multiplication table by
    position, and hold ``ident``, the position of the identity.  Subsets are
    tried by size up to ``max_size``, in order of position, and at most
    ``max_combos`` of them in all.  Returns the positions of the first that
    generates, ``()`` where ``positions`` holds the identity alone.  Returns
    ``None`` where no subset within those bounds generates them.
    """
    target = set(int(p) for p in positions)
    ordered = sorted(target - {int(ident)})
    if not ordered:
        return ()
    tried = 0
    for size in range(1, max_size + 1):
        for combo in itertools.combinations(ordered, size):
            tried += 1
            if tried > max_combos:  # pragma: no cover - a guard, not a path
                return None
            if _closure(combo, prod, ident) == target:
                return combo
    return None


def _bfs_words(gens, prod, ident, targets):
    """The shortest word in ``gens`` of each of ``targets``, and their tree.

    ``gens`` are positions, indices into the group's operations, and ``prod``
    is the multiplication table by position.  The search starts at
    ``ident``, the position of the identity, and steps only onto
    ``targets``.  Returns ``(words, parent)``: ``words[q] = (i, j, ...)``
    with ``q = g_i·g_j···``, the identity's word empty, and
    ``parent[q] = (i, r)`` with ``q = g_i·r``, the edge a representation is
    propagated along.
    """
    words = {int(ident): ()}
    parent = {}
    frontier = [int(ident)]
    targets = set(int(t) for t in targets)
    while frontier:
        nxt = []
        for p in frontier:
            for gi, gp in enumerate(gens):
                q = int(prod[gp, p])
                if q not in words and q in targets:
                    words[q] = (gi,) + words[p]
                    parent[q] = (gi, p)
                    nxt.append(q)
        frontier = nxt
    return words, parent


def _moves_padded(sym, vacuum_dirs):
    """Whether the operation ``sym`` moves a vacuum direction of the cell.

    ``vacuum_dirs`` are the directions of a layer's cell that hold vacuum.
    The operation moves one where its rotation does not fix the unit vector
    along it, or mixes that component with the others.  Returns ``False``
    where ``vacuum_dirs`` is empty.
    """
    if not vacuum_dirs:
        return False
    R = np.asarray(sym.rotation, dtype=float)
    for d in vacuum_dirs:
        e = np.zeros(3)
        e[d] = 1.0
        if not np.allclose(R @ e, e, atol=1e-9):
            return True
        if not np.allclose(R[d, :], e, atol=1e-9):
            return True
    return False


def _orbital_permutation(assignment, p, model, order=None):
    """The image of each orbital of ``model`` under the operation at ``p``.

    ``assignment`` gives each operation's permutation of the sites and each
    orbital's site.  Within a site the orbitals are matched by slot, their
    place in ``order``, which lists each site's orbitals in the order of the
    ones they are images of at the reference site of their orbit
    (:func:`.._multiplets._slot_order`).  With no ``order``, the slot is the
    place in ``assignment.site_orbitals``.  Returns an integer array of
    length ``model.norb``.
    """
    perm = np.asarray(assignment.permutations, dtype=int)
    site_of = np.asarray(assignment.site_of_orbital, dtype=int)
    site_orbs = assignment.site_orbitals if not order else order
    slot = np.zeros(model.norb, dtype=int)
    for s, orbs in enumerate(site_orbs):
        for j, o in enumerate(orbs):
            slot[o] = j
    out = np.empty(model.norb, dtype=int)
    for a in range(model.norb):
        out[a] = site_orbs[perm[p, site_of[a]]][slot[a]]
    return out


def _channel_permutation(assignment, p, model, layout):
    """The image of each multiplet under the operation at position ``p``.

    A multiplet is a set of orbitals at one site, and ``layout`` is the
    :class:`SiteMultiplets` record that holds them.  Its image is the
    multiplet :func:`_orbital_permutation` carries its orbitals onto, with
    ``assignment`` and ``model`` as there and the slots in ``layout``'s order.
    The multiplets partition the slots alike at every site of an orbit
    (:func:`.._multiplets._site_multiplets`), so the image is a whole
    multiplet.  Returns an integer array, one image for each multiplet.
    """
    orb = _orbital_permutation(assignment, p, model,
                               order=getattr(layout, "_site_order", None))
    chan_of = np.asarray(layout._chan_of, dtype=int)
    out = np.empty(int(layout._nchan), dtype=int)
    for c, orbs in enumerate(layout._chan_orbs):
        out[c] = int(chan_of[int(orb[int(orbs[0])])])
    return out


def _resolve_spin(spin, model, *, call="realized_group"):
    """The spinor basis ``spin=`` declares, or ``None``.

    ``spin`` is a :class:`..spin.SpinBasis`; a string, passed to
    :func:`..spin.spin_basis` as ``order`` (``"interleaved"`` or
    ``"blocked"``); a mapping of :func:`..spin.spin_basis` keyword arguments;
    or ``None``, which is returned as it is.  ``model`` is the model it is
    declared for, and ``call`` the name of the call its user made, which the
    errors open with.  Raises ``ValueError`` where ``model`` is natively
    spinful, where a :class:`..spin.SpinBasis` was written for a model with
    another number of orbitals, or where ``spin`` is of none of those kinds.
    :func:`..spin.spin_basis` raises its own for a string or a mapping it
    cannot read.
    """

    if spin is None:
        return None
    if int(model.nspin) == 2:
        raise ValueError(
            f"{call}(): spin= is passed with a spinful model (nspin=2). Drop spin=.")
    if isinstance(spin, SpinBasis):
        if int(spin._norb) != int(model.norb):
            raise ValueError(
                f"{call}(): spin= is the SpinBasis of a model with "
                f"{int(spin._norb)} orbitals, and the model has {int(model.norb)}. Pass "
                f"the SpinBasis spin_basis returns for this model.")
        return spin
    if isinstance(spin, str):
        return spin_basis(model, order=spin)
    if isinstance(spin, dict):
        return spin_basis(model, **spin)
    raise ValueError(
        f"{call}(): spin= is of type {type(spin).__name__}; pass a SpinBasis, "
        f"\"interleaved\" or \"blocked\", or the arguments of spin_basis written as "
        f"{{\"pairs\": ...}}.")


def _spin_pair_split(spin, site_orbitals, site_positions, lattice):
    r"""The refusal of the first declared spinor pair split across two sites.

    ``spin`` is the declared :class:`..spin.SpinBasis`, ``site_orbitals``
    holds each site's orbitals, ``site_positions`` each site's position in
    reduced coordinates, and ``lattice`` the lattice vectors they are reduced
    in.  The two spin states of one Wannier function share its center, so
    only the sites are read and no group is needed.  Returns the refusal
    ``spin_pair_split`` for the first pair split, its ``value`` the Cartesian
    distance between the pair's two sites modulo the lattice and its
    threshold ``0.0``.  Returns ``None`` where each pair sits at one site.
    """
    site_of = {int(o): s for s, orbs in enumerate(site_orbitals) for o in orbs}
    pos = np.asarray(site_positions, dtype=float)
    lat = np.asarray(lattice, dtype=float)
    for c, (up, dn) in enumerate(spin.pairs):
        su, sd = site_of[int(up)], site_of[int(dn)]
        if su == sd:
            continue
        delta = pos[su] - pos[sd]
        dist = float(np.linalg.norm((delta - np.round(delta)) @ lat))
        return Refusal(
            "spin_pair_split",
            f"declared spin pair {c} = (up {int(up)}, down {int(dn)}) has its two "
            f"orbitals on different sites: orbital {int(up)} on site {su} at "
            f"{np.round(pos[su], 6).tolist()} and orbital {int(dn)} on site {sd} at "
            f"{np.round(pos[sd], 6).tolist()} in reduced coordinates; the two sites are "
            f"{dist:.6g} apart, in the units of the lattice vectors. Check the order of "
            f"the spin states declared with spin=. Where the two centers differ by the "
            f"noise of a fitted model, declare its precision with precision=.",
            value=dist, threshold=0.0,
            context={"pair": int(c), "orbitals": (int(up), int(dn)),
                     "sites": (su, sd),
                     "positions": (tuple(float(x) for x in pos[su]),
                                   tuple(float(x) for x in pos[sd])),
                     "spin_kind": str(spin._kind)})
    return None


def _spin_channel_permutation(assignment, p, model, spin, order=None):
    """The image of each declared spinor pair under the operation at ``p``.

    ``assignment`` gives each operation's permutation of the sites, ``p`` is a
    position in the group's operations, ``model`` is the model and ``spin``
    its declared :class:`..spin.SpinBasis`.  Each orbital's image is read by
    :func:`_orbital_permutation`, in the slot order ``order``, or, where none
    is given, in each site's orbitals read pair by pair (:func:`_pair_order`).
    Returns ``(image, None)`` where every pair is carried onto a pair,
    ``image[c]`` the pair that pair ``c`` is carried onto.  Otherwise returns
    ``(None, (c, a, b))`` for the first pair ``c`` whose up and down orbitals
    are carried into the different pairs ``a`` and ``b``.
    """
    orb = _orbital_permutation(assignment, p, model,
                               order=order or _pair_order(assignment.site_orbitals, spin))
    chan_of = spin._pair_of_orbital
    image = np.empty(len(spin.pairs), dtype=int)
    for c, (up, dn) in enumerate(spin.pairs):
        cu, cd = int(chan_of[int(orb[int(up)])]), int(chan_of[int(orb[int(dn)])])
        if cu != cd:
            return None, (int(c), cu, cd)
        image[c] = cu
    return image, None


def _internal_from_table(sym):
    """The spin factor of the operation ``sym``, a 2 × 2 matrix.

    It is the SU(2) rotation the irrep package gives the operation,
    ``sym.spinor_rotation``, and for an operation combined with time reversal
    ``iσ_y`` times its complex conjugate.
    """
    S = np.asarray(sym.spinor_rotation, dtype=complex)
    if sym.time_reversal:
        S = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=complex) @ S.conj()
    return S


def _lift_of(sym, su2):
    r"""The spin factor ``S(g)`` of the matrix of the operation ``sym``.

    ``su2`` says whether the states carry a spin, natively or by a declared
    spinor basis.  Returns :func:`_internal_from_table`'s matrix where ``su2``
    is true, and the 1 × 1 identity otherwise.  It is the only spin factor an
    operation is represented with.
    """
    return _internal_from_table(sym) if su2 else np.eye(1, dtype=complex)


@dataclass(frozen=True)
class SiteMultiplets:
    """The multiplets the Hamiltonian binds each site's orbitals into.

    A site's multiplets are the finest partition of its orbitals such that
    every operation fixing the site has a matrix carrying each multiplet into
    itself.  Orbitals that a matrix acting within sites and commuting with
    ``H(k)`` at every ``k`` mixes also lie in one multiplet.  A ``(p_x, p_y)``
    pair under a threefold axis is one multiplet.  So is a ``(p_x, p_y)`` pair
    under a fourfold axis that no hopping couples to each other, in whatever
    basis it is written: the fourfold axis or a matrix commuting with
    ``H(k)``, such as the projector onto ``p_x``, mixes the two orbitals it is
    written in.  Identical copies of a set of orbitals, coupled to one another
    by no hopping, are one multiplet, since a matrix commuting with ``H(k)``
    exchanges them.  So are copies of one site irrep whose basis is turned
    from site to site of an orbit, since an operation carrying one site onto
    another then mixes the copies.  An orbital that nothing mixes with another
    is a multiplet of its own.

    Each multiplet's matrices are the model's own blocks, in whatever basis
    its orbitals are written.  A declared site irrep is assigned to the
    multiplets as they are, without changing them.  The record, or its
    ``by_site``, is taken by :func:`.groups.realized_group` as
    ``site_multiplets=``, which states the multiplets instead of measuring
    them.  ``summary()`` returns each site's multiplets as text.

    Attributes
    ----------
    by_site : tuple
        For each site, its multiplets, each the orbitals it holds.
    dims : tuple
        For each site, the dimension of each multiplet: the number of orbitals
        it holds, or of spinor pairs where a spin basis is declared.
    accidental : tuple of dict
        If nonempty, the orbitals whose multiplet could be chosen in more than
        one way, an accidental degeneracy: an operation mixes an orbital with
        several partners, each at least half as strongly as the strongest, and
        the multiplet returned holds the lowest-indexed one.  Each entry holds
        the ``operation``, the ``orbitals``, the ``chosen`` partners and a
        ``note``.
    determined_by : str
        How the multiplets are known: ``"hamiltonian"``, read off the
        Hamiltonian, or ``"declaration"``, stated by ``site_multiplets=``.  A
        record handed back as ``site_multiplets=`` keeps its own value.
    """

    by_site: tuple
    dims: tuple
    accidental: tuple = ()
    determined_by: str = "hamiltonian"
    # the same partition in the forms the arithmetic reads it in
    _spinful: bool = field(default=False, repr=False)
    _site_orbitals: tuple = field(default=(), repr=False)
    _nchan: int = field(default=0, repr=False)
    _chan_states: tuple = field(default=(), repr=False)
    _chan_of: Any = field(default=None, repr=False)
    _chan_orbs: tuple = field(default=(), repr=False)
    _chan_kind: tuple = field(default=(), repr=False)
    _chan_site: tuple = field(default=(), repr=False)
    _measured: bool = field(default=False, repr=False)
    # each site's orbitals in the order of the ones they are images of at the
    # reference site of their orbit, which says which orbital answers to which
    # across an orbit; empty where the sites' own order is it
    _site_order: tuple = field(default=(), repr=False)
    # the declared symmetry tolerance, relative to max|H|, the multiplets were
    # measured at (`noise_floor`), and every later test of their blocks reads
    # it too; None where nothing was declared
    _tol: Any = field(default=None, repr=False, compare=False)
    # how far the members of the operations' solution spaces these multiplets
    # were read off reach outside their block form, relative to their scale
    # (`_block_members`): rounding where the Hamiltonian holds the multiplets
    # exactly, and above `_IN_BLOCKS` where they hold only within the declared
    # tolerance
    _mixing: float = field(default=0.0, repr=False, compare=False)

    def _dim(self, chan) -> int:
        """The dimension of multiplet ``chan``, in orbitals or spinor pairs.

        ``chan`` is the multiplet's index among every site's multiplets.  The
        count is read off its states, two to each orbital of a natively
        spinful model and to each declared spinor pair, one to each orbital
        otherwise.
        """
        return len(self._chan_states[int(chan)]) // (2 if self._spinful else 1)

    @property
    def _near(self) -> bool:
        """Whether these multiplets hold only within the declared tolerance.

        True where the members they were read off reach outside their block
        form by more than :data:`.._tolerance._IN_BLOCKS` (``_mixing``), as
        they do for an orbital frame turned off the one the operations carry
        by less than the declared precision.  Where it is true,
        :func:`..groups.realized_group` projects the model's ``H`` onto its
        symmetric part.
        """
        return float(self._mixing) > _IN_BLOCKS

    def summary(self) -> str:
        """Each site's multiplets, any accidental ties and how they are known.

        A line each, with the attribute that holds it, returned as text.
        """
        lines = ["each site's orbitals, multiplet by multiplet   (.by_site)"]
        for s, (multiplets, dims) in enumerate(zip(self.by_site, self.dims)):
            lines.append(f"  site {s}: " + "; ".join(
                _orbitals_named(orbs) + ("" if int(d) == len(orbs) else
                                         ", a spinor pair" if int(d) == 1 else
                                         f", {int(d)} spinor pairs")
                for orbs, d in zip(multiplets, dims)))
        for entry in self.accidental:
            lines.append(f"  {_orbitals_named(entry['orbitals'])}: {entry['note']}, "
                         f"{_orbitals_named(entry['chosen'])}   (.accidental)")
        lines.append(("  read off the Hamiltonian" if self.determined_by == "hamiltonian"
                      else "  stated by site_multiplets=") + "   (.determined_by)")
        return "\n".join(lines)


def _multiplets_of(model, assignment, spin, parts, *, ties=(), source="measured",
                   order=None, tol=None, mixing=0.0):
    """The :class:`SiteMultiplets` one partition of every site's orbitals is.

    The states each multiplet holds, the multiplet of each orbital and the
    site of each multiplet follow from the partition and the model's spin.  A
    multiplet of dimension one is represented with the spin factor
    :func:`_lift_of` gives; for a larger one the matrix on all its states is
    measured.  The multiplets of every site are indexed in one list, whose
    entries the build calls channels (``_nchan``, ``_chan_states``,
    ``_chan_of``).

    Parameters
    ----------
    model : pythtb.TBModel
        The model; its ``nspin`` and ``norb`` are read.
    assignment : ..groups.OrbitalAssignment
        The model's assignment to the group; its ``site_orbitals`` are read.
    spin : ..spin.SpinBasis or None
        The declared spinor basis, ``None`` where none is declared.
    parts : sequence
        ``parts[s]`` the multiplets of site ``s``, each the orbitals it holds.
    ties : sequence of dict, default ()
        The entries of the record's ``accidental``.
    source : str, default "measured"
        ``"declared"`` for a partition stated by ``site_multiplets=``, any
        other value for one read off the Hamiltonian.
    order : sequence or None, default None
        Each site's orbitals in slot order (:func:`.._multiplets._slot_order`),
        which matches a multiplet to its image across an orbit.  ``None``
        keeps each site's own order, its declared spinor pairs in turn where a
        basis is declared (:func:`_pair_order`).
    tol : float or None, default None
        The declared tolerance the multiplets' blocks are tested at, relative
        to ``max|H|`` (:func:`.._tolerance.noise_floor`), which every later
        test of them reads.  ``None`` where nothing is declared.
    mixing : float, default 0.0
        How far outside their block form the members the multiplets were read
        off reach (:func:`_block_members`).

    Returns
    -------
    SiteMultiplets
        Its ``determined_by`` is ``"declaration"`` where ``source`` is
        ``"declared"`` and ``"hamiltonian"`` otherwise.
    """
    nspin = int(model.nspin)
    # the states carry a spin natively, or by a declared spinor basis, whose
    # (up, down) pairs are each one Wannier function of two orbitals
    su2 = nspin == 2 or spin is not None
    site_orbs = tuple(tuple(int(o) for o in orbs) for orbs in assignment.site_orbitals)
    if order is None:
        # a site's own slots, where nothing measured says which orbital answers
        # to which across an orbit: its orbitals, or its declared pairs in turn
        order = _pair_order(site_orbs, spin)
    chan_orbs, chan_states, chan_site, chan_kind = [], [], [], []
    dims = []
    chan_of = np.zeros(int(model.norb), dtype=int)
    for s, chans in enumerate(parts):
        site_dims = []
        for orbs in chans:
            c = len(chan_orbs)
            orbs = tuple(int(o) for o in orbs)
            d = len(orbs) // (1 if spin is None else 2)
            chan_orbs.append(orbs)
            chan_states.append(tuple(o * nspin + t for o in orbs for t in range(nspin)))
            chan_site.append(int(s))
            chan_kind.append("tables" if d == 1 else "measured")
            site_dims.append(int(d))
            for o in orbs:
                chan_of[o] = c
        dims.append(tuple(site_dims))
    return SiteMultiplets(
        by_site=tuple(tuple(tuple(int(o) for o in ch) for ch in chans) for chans in parts),
        dims=tuple(dims), accidental=tuple(ties),
        determined_by="declaration" if str(source) == "declared" else "hamiltonian",
        _spinful=su2, _site_orbitals=site_orbs, _nchan=len(chan_orbs),
        _chan_states=tuple(chan_states), _chan_of=chan_of, _chan_orbs=tuple(chan_orbs),
        _chan_kind=tuple(chan_kind), _chan_site=tuple(chan_site),
        _measured=any(k == "measured" for k in chan_kind),
        _site_order=(() if order is None or tuple(tuple(int(o) for o in orbs)
                                                   for orbs in order) == site_orbs
                     else tuple(tuple(int(o) for o in orbs) for orbs in order)),
        _tol=tol, _mixing=float(mixing))


def _atoms_of_site(site_orbs, spin):
    """The indivisible pieces of each site: its orbitals, or its spinor pairs.

    ``site_orbs`` holds each site's orbitals and ``spin`` the declared
    :class:`..spin.SpinBasis`, or ``None``.  A declared ``(up, down)`` pair is
    one Wannier function, which no multiplet splits.  Returns, for each site,
    a list of its pieces in the order the site lists them: one orbital each
    where ``spin`` is ``None``, and otherwise a pair's two orbitals as
    ``(up, down)``.
    """
    if spin is None:
        return [[(int(o),) for o in orbs] for orbs in site_orbs]
    pair_of = np.asarray(spin._pair_of_orbital, dtype=int)
    out = []
    for orbs in site_orbs:
        seen, atoms = [], []
        for o in orbs:
            c = int(pair_of[int(o)])
            if c not in seen:
                seen.append(c)
                atoms.append(tuple(int(x) for x in spin.states(c)))
        out.append(atoms)
    return out


def _pair_order(site_orbs, spin):
    """Each site's orbitals read pair by pair, up first in each declared pair.

    ``site_orbs`` holds each site's orbitals and ``spin`` the declared
    :class:`..spin.SpinBasis`, or ``None``, which leaves each site's orbitals
    as listed.  The pairs come in the order the site lists them
    (:func:`_atoms_of_site`).  This is the slot order a site keeps where no
    operation says which of its pairs answers to which at another site of its
    orbit (:func:`.._multiplets._slot_order`).  Returns each site's orbitals
    in that order.
    """
    return tuple(tuple(int(o) for a in atoms for o in a)
                 for atoms in _atoms_of_site(site_orbs, spin))


def _singleton_parts(site_orbs, atoms):
    """The finest partition of every site, each indivisible piece a multiplet.

    ``atoms`` holds each site's pieces (:func:`_atoms_of_site`) and
    ``site_orbs`` each site's orbitals, of which only the number of sites is
    read.  Returns each site's multiplets, in the form :func:`_multiplets_of`
    takes as ``parts``.
    """
    return tuple(tuple(tuple(a) for a in atoms[s]) for s in range(len(site_orbs)))


def _parts_from_merge(site_orbs, atoms, orbit_of, merged):
    """The multiplets that merged slots describe, site by site.

    ``site_orbs`` holds each site's orbitals, ``atoms[s][j]`` the indivisible
    piece of site ``s`` in slot ``j`` (:func:`_atoms_of_site`), ``orbit_of[s]``
    the orbit of site ``s``, and ``merged`` a :class:`_Union` over
    ``(orbit, slot)`` pairs.  The pieces of a site whose slots ``merged``
    joins form one multiplet.  Inside a multiplet the pieces stand in the
    site's own order, and each keeps its own: a declared pair stays
    ``(up, down)``, the order the spin factor is written in.  Returns each
    site's multiplets, ordered by the lowest orbital each holds.
    """
    parts = []
    for s, site_atoms in enumerate(atoms):
        rank = {int(o): i for i, o in enumerate(site_orbs[s])}
        groups = {}
        for j, a in enumerate(site_atoms):
            groups.setdefault(merged.find((int(orbit_of[s]), j)), []).append(a)
        chans = [tuple(int(o) for a in sorted(members, key=lambda a: rank[int(a[0])])
                       for o in a)
                 for members in groups.values()]
        parts.append(tuple(sorted(chans, key=lambda ch: min(ch))))
    return tuple(parts)


class _Union:
    """Classes of elements merged two at a time, each led by its least element.

    ``find(x)`` returns the representative of ``x``'s class, its least
    element, and ``union(x, y)`` merges the classes of ``x`` and ``y`` and
    returns whether they were apart.  The elements are hashable and ordered.
    """

    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        a, b = self.find(x), self.find(y)
        if a == b:
            return False
        lo, hi = (a, b) if a <= b else (b, a)
        self.parent[hi] = lo
        return True


def _in_blocks(basis, blocks, *, tol=_IN_BLOCKS):
    """A basis of the members of a solution space supported on ``blocks``.

    ``basis`` is a basis of the space, an array of shape ``(dim, N, N)``, and
    ``blocks`` the ``(rows, cols)`` index arrays of the blocks a member may be
    nonzero in (:func:`_ansatz_blocks`).  A member, a combination of
    ``basis`` whose coefficients form a unit vector, is in the blocks where its
    norm outside them is at most ``tol``, or ``tol`` times the largest such
    norm where that exceeds one.  ``tol`` is :data:`.._tolerance._IN_BLOCKS`,
    or the declared tolerance where that is larger
    (:func:`.._tolerance.noise_floor`).  Returns an array of shape
    ``(m, N, N)``, empty where no member is in the blocks.
    """
    return _block_members(basis, blocks, tol=tol)[0]


def _block_members(basis, blocks, *, tol=_IN_BLOCKS, lifts=None):
    """The members of a space in block form, and how far outside it they reach.

    ``basis``, ``blocks`` and ``tol`` are as in :func:`_in_blocks`.
    ``lifts`` holds, for each block, the matrix the members are to carry a
    multiple of there, or ``None`` for a block whose whole is allowed
    (:func:`_ansatz_residual`); every block is whole where ``lifts`` is not
    given.  Returns ``(members, mixing)``: ``members`` as :func:`_in_blocks`
    returns them, and ``mixing`` the largest norm outside block form among
    them, on the same scale.  ``mixing`` is rounding where the Hamiltonian
    holds the blocks exactly, above :data:`.._tolerance._IN_BLOCKS` only where
    a declared tolerance let a member in, and ``0.0`` where no member is in
    block form.
    """
    B = np.asarray(basis, dtype=complex)
    lifts = [None] * len(blocks) if lifts is None else list(lifts)
    A = np.stack([_ansatz_residual(B[j], blocks, lifts).reshape(-1)
                  for j in range(B.shape[0])], axis=1)
    _, s, vh = np.linalg.svd(A, full_matrices=False)
    scale = float(s[0]) if s.size else 0.0
    keep = [i for i in range(len(s)) if float(s[i]) <= tol * max(scale, 1.0)]
    if len(keep) == len(s) and scale <= tol:
        # nothing sits outside the blocks at all
        return B, scale / max(scale, 1.0)
    if not keep:
        return np.zeros((0,) + B.shape[1:], dtype=complex), 0.0
    mixing = max(float(s[i]) for i in keep) / max(scale, 1.0)
    return (np.stack([np.einsum("j,jab->ab", vh[i].conj(), B) for i in keep], axis=0),
            mixing)


def _unfilled_blocks(space, blocks, *, tol=None, seed=17, draws=3):
    """Whether a space has a member nonsingular on every block, and where not.

    ``space`` is a basis of one operation's solution space, already restricted
    to its own site blocks (:func:`_in_blocks`), and ``blocks`` the
    ``(rows, cols)`` a candidate partition into multiplets allows the
    operation a nonzero entry in.  ``tol`` is the declared tolerance every
    threshold reads (:func:`.._tolerance.noise_floor`), ``None`` for the exact
    defaults.  The members in block form are found by :func:`_block_members`
    and read by :func:`_filled_check`, to which ``seed`` and ``draws`` are
    handed.  Returns ``(ok, unfilled, mixing)``: ``ok`` and ``unfilled`` as
    :func:`_filled_check` returns them, and ``mixing`` how far outside the
    blocks the members in block form reach.
    """
    V, mixing = _block_members(space, blocks, tol=noise_floor(_IN_BLOCKS, tol))
    ok, unfilled = _filled_check(V, blocks, tol=tol, seed=seed, draws=draws)
    return ok, unfilled, mixing


def _filled_check(V, blocks, *, tol=None, seed=17, draws=3):
    """Whether members in block form are nonsingular on every block.

    ``V`` holds the members, an array of shape ``(m, N, N)`` supported on
    ``blocks`` (:func:`_block_members`), and ``blocks`` their ``(rows, cols)``.
    A block is empty where no member reaches :data:`.._tolerance.ZERO` times
    the members' largest entry.  It is singular in a combination of the
    members where its smallest singular value is at most
    :data:`.._tolerance.SINGULAR` times that combination's largest entry.  A
    scale below one is read as one, and the declared tolerance ``tol``
    replaces either threshold where it is larger
    (:func:`.._tolerance.noise_floor`).  The combinations are the ``draws``
    of :func:`_generic_members` with ``seed``, so the verdict is about the
    space and not about the basis ``V`` is in.  Returns ``(True, [])`` where
    one combination is nonsingular on every block.  Otherwise returns
    ``False`` with the indices of the empty blocks, or, where none is empty,
    of the blocks singular in the last combination tried.
    """
    idx = [(np.asarray(r, dtype=int), np.asarray(c, dtype=int)) for r, c in blocks]
    if not len(V):
        return False, list(range(len(blocks)))
    empty, singular = noise_floor(ZERO, tol), noise_floor(SINGULAR, tol)
    scale = float(np.abs(V).max())
    unfilled = [i for i, (rows, cols) in enumerate(idx)
                if float(np.abs(V[:, rows][:, :, cols]).max()) <= empty * max(scale, 1.0)]
    if unfilled:
        return False, unfilled
    worst = []
    for X in _generic_members(V, seed=seed, draws=draws):
        norm = float(np.abs(X).max())
        worst = []
        for i, (rows, cols) in enumerate(idx):
            sv = np.linalg.svd(X[np.ix_(rows, cols)], compute_uv=False)
            if sv.size == 0 or float(sv[-1]) <= singular * max(norm, 1.0):
                worst.append(i)
        if not worst:
            return True, []
    return False, worst


def _generic_members(V, *, seed=17, draws=3):
    """Fixed pseudo-random combinations of the members ``V``, one at a time.

    ``V`` is an array of shape ``(m, N, N)``.  Yields ``draws`` matrices of
    shape ``(N, N)``, each the members summed with complex Gaussian
    coefficients from a generator seeded with ``seed``, so that the same
    ``V`` gives the same matrices on every call.
    """
    rng = np.random.default_rng(int(seed))
    for _ in range(int(draws)):
        y = rng.normal(size=len(V)) + 1j * rng.normal(size=len(V))
        yield np.einsum("j,jab->ab", y, V)


def _site_spaces(model, group, assignment, spin, validated, *, tol=None):
    """Each operation's solution space, kept to the site blocks it allows.

    ``model`` is the model, ``group`` the group, ``assignment`` the model's
    assignment to it, ``spin`` the declared spinor basis or ``None``, and
    ``validated`` the measurement of the group, whose solution space for each
    operation is read, or ``None``.  The members kept are those supported on
    the blocks the operation's permutation of the sites allows
    (:func:`_in_blocks`), at :data:`.._tolerance._IN_BLOCKS` or the declared
    tolerance ``tol`` where that is larger (:func:`.._tolerance.noise_floor`).
    No partition of a site's orbitals changes those blocks, so every question
    about which orbitals answer to which is asked of these members.

    Returns ``(spaces, site_layout, mixing)``: ``spaces`` the members of each
    operation, by operation index; ``site_layout`` the :class:`SiteMultiplets`
    in which each site is one multiplet; and ``mixing`` how far outside its
    site blocks each operation's members reach (:func:`_block_members`).  An
    operation with no solution space, one too large to read
    (``_READOFF_MAX_BYTES``), or one with no member in its site blocks is in
    neither mapping.
    """
    site_orbs = tuple(tuple(int(o) for o in orbs) for orbs in assignment.site_orbitals)
    site_layout = _multiplets_of(model, assignment, spin,
                             tuple((tuple(orbs),) for orbs in site_orbs), tol=tol)
    bases = (getattr(validated, "_bases", None) or {}) if validated is not None else {}
    pos_of_ind = {int(s.ind): i for i, s in enumerate(group._symmetries)}
    spaces, mixing = {}, {}
    for ind in (validated._indices if validated is not None else ()):
        ind = int(ind)
        if ind not in bases or ind not in pos_of_ind:
            continue
        basis = np.asarray(bases[ind])
        if basis.ndim != 3 or basis.shape[0] == 0:
            continue
        nstate = int(basis.shape[1])
        if basis.shape[0] * nstate * nstate * 16 > _READOFF_MAX_BYTES:
            continue
        image = _channel_permutation(assignment, pos_of_ind[ind], model, site_layout)
        got, reach = _block_members(basis, _ansatz_blocks(site_layout._chan_states, image, 1),
                                    tol=noise_floor(_IN_BLOCKS, tol))
        if len(got):
            spaces[ind], mixing[ind] = got, reach
    return spaces, site_layout, mixing


# --------------------------------------------------------------------------
# reading the tensor structure's discrete choices off a measured intertwiner
# --------------------------------------------------------------------------
#
# The build's matrices have the form `D(g) = O(g) ⊗ S(g)`, a matrix on the
# orbitals times the spin factor, and the detector has measured the whole
# space of matrices that carry `H(k)` onto `H` at the operation's image of
# `k`.  Where a matrix of that form exists it lies in that space, so its
# discrete choices are read off the space rather than enumerated.  What is
# read off is a candidate, which the build still tests.

def _ansatz_blocks(chan_states, image, orient):
    """Where an operation's matrix ``O(g) ⊗ S(g)`` may be nonzero.

    ``O(g)`` acts on the orbitals and ``S(g)`` is the spin factor.
    ``chan_states[c]`` holds the states of multiplet ``c`` and ``image[c]``
    the multiplet the operation carries it onto.  With ``orient = 1``, block
    ``c`` has the states of ``image[c]`` as rows and those of ``c`` as
    columns.  With ``orient = -1``, for the inverse operation, whose matrix
    is the adjoint, rows and columns are exchanged and the block carries
    ``S(g)†``.  Returns a list of ``(rows, cols)`` index arrays, one for each
    multiplet.
    """
    out = []
    for c in range(len(chan_states)):
        src = np.asarray(chan_states[c], dtype=int)
        dst = np.asarray(chan_states[int(image[c])], dtype=int)
        out.append((src, dst) if orient < 0 else (dst, src))
    return out


def _ansatz_residual(X, blocks, lifts, classes=None, weights=None):
    """The part of ``X`` that no matrix of the form ``O ⊗ S`` carries.

    ``X`` is an ``N × N`` matrix and ``blocks`` the ``(rows, cols)`` of the
    allowed blocks (:func:`_ansatz_blocks`).  ``lifts[i]`` is the matrix
    block ``i`` is to carry a multiple of, the spin factor ``S``.  It is
    ``None`` for a free block, whose whole is allowed, as for a multiplet of
    dimension above one.  The coefficients are fitted by least squares, one
    for each block, or, with ``classes``, lists of block indices, one shared
    over each class.  The build's freedom is a scalar per multiplet orbit per
    generator.  With ``weights``, one complex number per block,
    block ``i`` is fitted to ``coef · weights[i] · lifts[i]``.  Each block is
    fitted with its own lift.  Returns ``X`` with the fitted part taken off
    each block and every free block set to zero.
    """
    R = np.array(X, dtype=complex, copy=True)
    groups = classes if classes is not None else [[i] for i in range(len(blocks))]
    for members in groups:
        if weights is None:
            live = [i for i in members if lifts[i] is not None]
            for i in members:
                if lifts[i] is None:
                    R[np.ix_(*blocks[i])] = 0.0
            denom = sum(float(np.vdot(lifts[i], lifts[i]).real) for i in live)
            if denom <= 0:               # pragma: no cover - S is unitary here
                continue
            coef = sum(complex(np.vdot(lifts[i], R[np.ix_(*blocks[i])])) for i in live)
            coef /= denom
            for i in live:
                idx = np.ix_(*blocks[i])
                R[idx] = R[idx] - coef * lifts[i]
            continue
        live = []
        for i in members:
            if lifts[i] is None:
                R[np.ix_(*blocks[i])] = 0.0
            else:
                live.append(i)
        num, den = 0.0 + 0.0j, 0.0
        for i in live:
            w, S = complex(weights[i]), lifts[i]
            num += np.conj(w) * complex(np.vdot(S, R[np.ix_(*blocks[i])]))
            den += float(abs(w) ** 2) * float(np.vdot(S, S).real)
        if den <= 0:                     # pragma: no cover - a guard, not a path
            continue
        coef = num / den
        for i in live:
            idx = np.ix_(*blocks[i])
            R[idx] = R[idx] - coef * complex(weights[i]) * lifts[i]
    return R


def _ansatz_member(basis, blocks, lifts, classes=None, *, prefer=None, weights=None,
                   tol=ZERO):
    """The member of a solution space nearest the form ``O ⊗ S``, with its miss.

    The residuals of the basis elements off that form
    (:func:`_ansatz_residual`) are the columns of an ``N² × dim`` matrix, and
    the member returned is the combination along its last right singular
    vector.  Several directions can reach the form, as where a conserved
    operator acts inside a multiplet or a multiplet is coupled to nothing.
    Which of them is returned then fixes the relative character of the
    multiplets involved, which nothing in the Hamiltonian settles, and
    ``prefer`` makes that choice.

    Parameters
    ----------
    basis : numpy.ndarray, shape (dim, N, N)
        A basis of the solution space.
    blocks : list of (rows, cols)
        The allowed blocks (:func:`_ansatz_blocks`).
    lifts : list
        The spin factor each block carries a multiple of, ``None`` for a free
        block.
    classes : list of list of int or None, default None
        Blocks that share one coefficient, as in :func:`_ansatz_residual`.
    prefer : list of list of int or None, default None
        Groups of block indices, in the shape of ``classes``, which may join
        blocks at different sites.  Among the directions within
        ``tol`` of the best, the member returned is the one whose blocks carry
        the multiples of their lifts ``weights`` asks for, one coefficient over
        each group.  Read only where the space has more than one dimension
        and no block is free.
    weights : sequence of complex or None, default None
        One number per block.  ``None`` is every block the same multiple of
        its lift, the trivial relative character, taken where nothing is
        declared; a declared character asks for its own weights.
    tol : float, default ZERO
        How far above the smallest singular value, in units of the larger of
        one and the largest, a direction still reaches the form.

    Returns
    -------
    tuple
        ``(X, defect, scale)``.  ``X`` is the member, an ``N × N`` matrix.
        ``defect`` is the smallest singular value, which is the norm of what
        ``X`` leaves off the form unless ``prefer`` chose ``X``.  That norm
        then lies between ``defect`` and ``defect`` plus ``tol`` times the
        larger of one and ``scale``.  ``scale`` is the largest singular value,
        how far off the form the worst direction of the space is.  For a space
        of more than one dimension that reaches the form, ``defect`` is
        rounding and ``scale`` of the order of the basis' norm.  For a space
        of dimension one ``X`` is its one element and ``defect = scale``.
    """
    B = np.asarray(basis, dtype=complex)
    A = np.stack([_ansatz_residual(B[j], blocks, lifts, classes).reshape(-1)
                  for j in range(B.shape[0])], axis=1)
    _, s, vh = np.linalg.svd(A, full_matrices=False)
    y = vh[-1].conj()
    if prefer and len(s) > 1 and all(S is not None for S in lifts):
        w_of = ([1.0 + 0.0j] * len(blocks) if weights is None
                else [complex(z) for z in weights])
        floor = float(s[-1]) + tol * max(float(s[0]), 1.0)
        open_ = [i for i in range(len(s)) if float(s[i]) <= floor]
        if len(open_) > 1:
            members = [np.einsum("j,jab->ab", vh[i].conj(), B) for i in open_]
            A2 = np.stack([_ansatz_residual(X, blocks, lifts, prefer,
                                            weights=w_of).reshape(-1)
                           for X in members], axis=1)
            try:
                _u2, _s2, vh2 = np.linalg.svd(A2, full_matrices=False)
            except np.linalg.LinAlgError:  # pragma: no cover - a guard, not a path
                vh2 = None
            if vh2 is not None:
                w = vh2[-1].conj()
                pick = sum(complex(w[k]) * vh[i].conj() for k, i in enumerate(open_))
                nrm = float(np.linalg.norm(pick))
                if nrm > tol:
                    y = pick / nrm
    return np.einsum("j,jab->ab", y, B), float(s[-1]), float(s[0])


def _channel_classes(cperm, u_pos, nchan):
    """The class of each multiplet: its orbit under the operations ``u_pos``.

    ``cperm[p]`` is the image of each multiplet under the operation at
    position ``p`` (:func:`_channel_permutation`), ``u_pos`` the positions of
    the operations, and ``nchan`` the number of multiplets.  A class is a
    multiplet and its images under every product of the operations: on a
    group, its orbit.  The build's freedom is one scalar per class per
    generator.  Returns ``(cls, ncls)``: ``cls[c]`` the class of multiplet
    ``c``, the classes numbered in order of their lowest multiplet, and
    ``ncls`` the number of classes.
    """
    cls = -np.ones(int(nchan), dtype=int)
    ncls = 0
    for c in range(int(nchan)):
        if cls[c] < 0:
            cls[c] = ncls
            orbit = [c]
            for a in orbit:
                for p in u_pos:
                    d = int(cperm[p][a])
                    if cls[d] < 0:
                        cls[d] = ncls
                        orbit.append(d)
            ncls += 1
    return cls, ncls


def _block_space(basis, rows, cols, tol=ZERO):
    """The blocks a solution space carries at ``(rows, cols)``, largest first.

    ``basis`` is a basis of the space, an array of shape ``(dim, N, N)``, and
    ``rows`` and ``cols`` the block's states.  Returns ``(space, scale)``:
    ``space`` the blocks along the principal directions of the space, each
    scaled by its singular value, so that they are orthogonal and
    ``space[0]`` is the largest block a combination of ``basis`` with a unit
    vector of coefficients carries there; and ``scale`` the norm of
    ``space[0]``.  A block of norm at most ``tol`` times the larger of one
    and ``scale`` is left out.
    """
    blk = np.asarray(basis)[:, np.asarray(rows, dtype=int)][:, :,
                                                            np.asarray(cols, dtype=int)]
    d0, d1 = blk.shape[1], blk.shape[2]
    _, s, vh = np.linalg.svd(blk.reshape(blk.shape[0], -1), full_matrices=False)
    scale = float(s[0]) if s.size else 0.0
    keep = [i for i in range(len(s)) if float(s[i]) > tol * max(scale, 1.0)]
    return [complex(s[i]) * vh[i].reshape(d0, d1) for i in keep], scale


def _unitarized(M, *, tol=ZERO):
    """The unitary nearest the square matrix ``M``, or ``None``.

    The unitary is ``M``'s polar part.  ``None`` is returned where ``M`` is
    not square, or where none of its entries reaches ``tol``:
    :data:`.._tolerance.ZERO`, or the declared tolerance where ``M`` was read
    off a model symmetric only within it.
    """
    M = np.asarray(M, dtype=complex)
    if M.ndim != 2 or M.shape[0] != M.shape[1] or float(np.abs(M).max()) < tol:
        return None
    w, _, vh = np.linalg.svd(M)
    return w @ vh


def _projective_irreps(positions, prod, factor, *, seed=101, tries=4, tol=None):
    r"""The irreducible representations of a group with one factor system, each once.

    ``positions`` are the positions of the group's unitary operations, closed
    under ``prod``.  ``prod`` is the multiplication table by position.
    ``factor[(a, b)]`` is the factor system.  Each representation ``ρ``
    returned has ``ρ(a) ρ(b) = factor(a, b) ρ(ab)``.  They are read off the
    regular representation with that factor system,
    ``L(g) e_h = factor(g, h) e_{gh}``, in which each of them occurs.  A
    random Hermitian combination of the right action,
    ``R(g) e_h = factor(h, g) e_{hg}``, which commutes with it, has one copy
    of one of them on each eigenspace, and a copy is kept where its character
    is not one already kept.  The draws are seeded with ``seed``,
    ``seed + 1`` and on, ``tries`` at most.  ``tol`` is the declared tolerance
    the factor system was measured at; the eigenvalue clusters and the
    characters are read at it where it is larger than their exact defaults
    (:func:`.._tolerance.noise_floor`), and ``None`` keeps the defaults.

    Returns the representations, each ``{position: matrix}``, ordered by
    dimension and then by character, from the first draw that separates
    them, the squares of their dimensions summing to the order of the group.
    Returns ``None`` where no draw separates them.
    """
    pos = [int(p) for p in positions]
    n = len(pos)
    at = {p: i for i, p in enumerate(pos)}
    left, right = {}, {}
    for g in pos:
        L = np.zeros((n, n), dtype=complex)
        R = np.zeros((n, n), dtype=complex)
        for h in pos:
            L[at[int(prod[g, h])], at[h]] = complex(factor[(g, h)])
            R[at[int(prod[h, g])], at[h]] = complex(factor[(h, g)])
        left[g], right[g] = L, R
    cluster, agree = noise_floor(CLUSTER, tol), noise_floor(CHARACTER, tol)
    for attempt in range(int(tries)):
        rng = np.random.default_rng(int(seed) + attempt)
        A = sum(complex(rng.normal(), rng.normal()) * right[h] for h in pos)
        w, V = np.linalg.eigh(A + A.conj().T)
        spread = max(float(w[-1] - w[0]), 1.0)
        found, characters, separated, start = [], [], True, 0
        for i in range(1, n + 1):
            if i < n and float(w[i] - w[i - 1]) <= cluster * spread:
                continue
            Vc = V[:, start:i]
            start = i
            rho = {g: Vc.conj().T @ left[g] @ Vc for g in pos}
            chi = np.asarray([complex(np.trace(rho[g])) for g in pos])
            if abs(float(np.real(np.vdot(chi, chi))) / n - 1.0) > agree:
                separated = False
                break
            if any(float(np.abs(chi - c).max()) <= agree for c in characters):
                continue
            found.append(rho)
            characters.append(chi)
        if separated and sum(int(r[pos[0]].shape[0]) ** 2 for r in found) == n:
            order = sorted(range(len(found)), key=lambda j: (
                int(found[j][pos[0]].shape[0]),
                tuple((-round(float(v.real), 9), -round(float(v.imag), 9))
                      for v in characters[j])))
            return [found[j] for j in order]
    return None


def _dimension_sums(dims, k):
    """Every way of writing ``k`` as a sum of entries of ``dims``, listed once.

    ``dims`` is a sequence of positive integers, of which an entry may be used
    several times, and ``k`` a nonnegative integer.  Returns a list of the
    ways, each the indices into ``dims`` of its terms, in nondecreasing order.
    """
    out = []

    def _grow(start, left, taken):
        if left == 0:
            out.append(tuple(taken))
            return
        for i in range(start, len(dims)):
            if int(dims[i]) <= left:
                _grow(i, left - int(dims[i]), taken + [i])

    _grow(0, int(k), [])
    return out


def _count_dimension_sums(dims, k):
    """The number of ways :func:`_dimension_sums` lists, without listing them.

    ``dims`` and ``k`` are as there.  Returns an integer.
    """
    ways = [1] + [0] * int(k)
    for dim in dims:
        for total in range(int(dim), int(k) + 1):
            ways[total] += ways[total - int(dim)]
    return int(ways[int(k)])


def _real_characters(kept, gens, words, parent, au_of, prod, ident, tol=1e-9):
    r"""Every real one-dimensional character of the represented group.

    Multiplying each matrix ``D(g)`` by a character ``χ(g)`` changes nothing
    the Hamiltonian shows, so these relate the labelings of one crystal that
    the hoppings leave open.  Only real characters are enumerated.  In a
    grey group, whose time reversal commutes with every operation, they are
    all the characters time reversal admits; a group with no antiunitary
    operation has complex ones too (:func:`_characters_of_subgroup` with
    ``roots``).  ``χ`` is ``+1`` on the antiunitary operation ``T`` the others
    are written from, whose phase no corep label sees.

    Parameters
    ----------
    kept : iterable of int
        The positions of the operations represented, on which ``χ`` is to
        multiply.
    gens : sequence of int
        The positions of the generators, on each of which ``χ`` is ``+1`` or
        ``-1``.
    words : dict
        Each unitary operation's word in the generators, by position
        (:func:`_bfs_words`).
    parent : dict
        The edge ``(i, r)`` of the word tree by which each unitary operation
        ``q`` is reached, ``q = g_i·r``.
    au_of : dict or None
        The position of the unitary ``u`` with ``p = T·u``, for each
        antiunitary operation's position ``p``; ``χ(p)`` is ``χ(u)``.
    prod : numpy.ndarray of int
        The multiplication table, by position.
    ident : int
        The position of the identity.
    tol : float, default 1e-9
        How far ``χ(a) χ(b)`` may sit from ``χ(ab)`` and still multiply.

    Returns
    -------
    list of dict
        Each character as ``{position of g: χ(g)}``, one for each choice of
        signs on the generators that reaches every operation of ``kept`` and
        multiplies there.
    """
    out = []
    for vals in itertools.product((1.0, -1.0), repeat=len(gens)):
        chi = {int(ident): 1.0 + 0.0j}
        for q in sorted(words, key=lambda x: len(words[x])):
            if q == ident:
                continue
            gi, r = parent[q]
            chi[q] = complex(vals[gi]) * chi[r]
        for p, u in (au_of or {}).items():
            chi[p] = chi[u]
        good = True
        for a in kept:
            for b in kept:
                c = int(prod[a, b])
                if c in chi and a in chi and b in chi:
                    if abs(chi[a] * chi[b] - chi[c]) > tol:
                        good = False
                        break
            if not good:
                break
        if good and all(p in chi for p in kept):
            out.append(chi)
    return out


def _root_of_unity(k, n) -> complex:
    """``exp(2 pi i k / n)``, exactly ``±1`` or ``±i`` where it is one of them."""
    quarter, rest = divmod(4 * int(k), int(n))
    if rest == 0:
        return (1.0 + 0.0j, 1.0j, -1.0 + 0.0j, -1.0j)[quarter % 4]
    return complex(np.exp(2j * np.pi * int(k) / int(n)))


def _characters_of_subgroup(group, subset, *, roots=False, tol=1e-9):
    r"""Every one-dimensional character of a subgroup ``subset`` of ``group``.

    ``subset`` holds the subgroup's operations by index (``.ind``).  Not every
    character of a subgroup is the restriction of one of the group's, so they
    are enumerated on the subgroup itself: generators found on the group's
    multiplication table (:func:`_minimal_generators`), a value for each,
    propagated along the word tree, and the product checked on every pair of
    the subgroup to within ``tol``.  With ``roots`` false each generator takes
    ``+1`` or ``-1``, which is what time reversal in a grey group admits
    (:func:`_real_characters`).  With ``roots`` true it takes every root of
    unity of its order, which gives every one-dimensional character of the
    subgroup, as a group with no antiunitary operation needs.  Returns a list
    of ``{ind: χ(g)}``, empty where ``subset`` is not a subgroup or no three
    of its operations generate it.
    """
    prod = group._get_product_table(get_diff=False)
    pos_of = {int(s.ind): i for i, s in enumerate(group._symmetries)}
    ident = _identity_position(group)
    positions = sorted({pos_of[int(i)] for i in subset if int(i) in pos_of})
    if ident not in positions:
        return []
    if len(positions) == 1:
        return [{int(group._symmetries[ident].ind): 1.0 + 0.0j}]
    gens = _minimal_generators(positions, prod, ident)
    if not gens:
        return []
    words, parent = _bfs_words(gens, prod, ident, positions)
    if set(words) != set(positions):      # pragma: no cover - a closed set is reached
        return []
    ind_of = {p: int(group._symmetries[p].ind) for p in positions}
    if roots:
        values = [[_root_of_unity(k, _order_of(prod, ident, g))
                   for k in range(_order_of(prod, ident, g))] for g in gens]
    else:
        values = [[1.0 + 0.0j, -1.0 + 0.0j]] * len(gens)
    out = []
    for vals in itertools.product(*values):
        chi = {ident: 1.0 + 0.0j}
        for q in sorted(words, key=lambda x: len(words[x])):
            if q == ident:
                continue
            gi, r = parent[q]
            chi[q] = complex(vals[gi]) * chi[r]
        good = True
        for a in positions:
            for b in positions:
                if abs(chi[a] * chi[b] - chi[int(prod[a, b])]) > tol:
                    good = False
                    break
            if not good:
                break
        if good:
            out.append({ind_of[p]: complex(chi[p]) for p in positions})
    return out


def _order_of(prod, ident, p) -> int:
    """The order of the operation at position ``p``.

    ``prod`` is the multiplication table by position and ``ident`` the
    position of the identity.  Returns the least ``n`` with ``p`` to the
    ``n``-th power the identity, or ``49`` where no power up to the 48th is.
    """
    n, q = 1, int(p)
    while q != int(ident) and n <= 48:
        q = int(prod[int(p), q])
        n += 1
    return n


def _primed_partners(group, names):
    """Each unitary operation of ``names`` with its product with time reversal.

    ``group`` is the group and ``names`` a collection of operation indices
    (``.ind``).  The partner of a unitary ``g`` is ``T·g``, ``T`` the pure
    time reversal; a grey group's antiunitary operations are these partners.
    Returns ``{index of g: index of T·g}`` for each unitary ``g`` of ``names``
    whose partner is in ``names`` too, empty where ``group`` holds no pure
    time reversal.
    """
    syms = group._symmetries
    prod = group._get_product_table(get_diff=False)
    at = {int(s.ind): p for p, s in enumerate(syms)}
    t_pos = next((p for p, s in enumerate(syms) if s.time_reversal
                  and np.array_equal(np.rint(np.asarray(s.rotation)).astype(int),
                                     np.eye(3, dtype=int))
                  and np.allclose(np.asarray(s.translation) % 1.0, 0.0, atol=1e-8)), None)
    if t_pos is None:
        return {}
    out = {}
    for ind in names:
        if int(ind) not in at or syms[at[int(ind)]].time_reversal:
            continue
        mate = int(syms[int(prod[t_pos, at[int(ind)]])].ind)
        if mate in names:
            out[int(ind)] = mate
    return out


def _atom_blocks(states, su2):
    """The indivisible pieces of one multiplet's states, each an index array.

    ``states`` is the multiplet's state list
    (``SiteMultiplets._chan_states``) and ``su2`` whether the states carry a
    spin, natively or by a declared spinor basis.  Where ``su2`` is true a
    piece is two consecutive states, which carry the spin factor together:
    one orbital's two spin states, or one declared pair where the states
    stand pair by pair, as the parts :func:`_multiplets_of` is handed keep
    them (:func:`_parts_from_merge`).  Otherwise a piece is one state.
    Summing a scalar of each piece over a multiplet reads its orbital
    character, the trace of its block with the spin factor divided out.
    """
    step = 2 if su2 else 1
    return [np.asarray(states[i:i + step], dtype=int)
            for i in range(0, len(states), step)]


# --------------------------------------------------------------------------
# the covariance a representation is held to
# --------------------------------------------------------------------------



def covariance_gate(n_operations, nstate, scale, phases=0.0) -> float:
    r"""The tolerance on a representation's covariance residual.

    The covariance residual is ``max_k ‖D H(k) D† − H(Rk)‖``, and the
    tolerance is the most an exact representation misses by rounding.  It is
    ``max(5e-14·max|H|, |G|·N·ε·max|H|, phases)``, with ``|G|`` the number
    of operations, ``N`` the number of states, ``max|H|`` the largest modulus
    of an entry of ``H`` and ``ε`` the machine epsilon.  The matrices of an
    exact representation are products of up to ``|G|`` matrices of size
    ``N × N``, each product rounding at ``N·ε``.  ``D H D†`` multiplies that
    relative rounding by entries of ``H`` as large as ``max|H|``.  ``H(Rk)``
    also carries the rounding of its Bloch phases.  It grows with the integer
    entries of the matrix by which the operation acts on reduced k, and with
    the integer hopping vectors, both large on a strongly skewed lattice
    basis.  A wrong matrix misses by a fraction of ``max|H|`` itself, many
    decades above the tolerance.

    Parameters
    ----------
    n_operations : int
        ``|G|``, the number of operations represented, those combined with
        time reversal included.
    nstate : int
        ``N``, the number of states.
    scale : float
        ``max|H|``, the largest modulus of an entry of ``H(k)`` over the
        k-points the covariance is measured at, in the units of ``H``.
    phases : float, default 0.0
        The rounding that ``H(k)`` and ``H(Rk)`` carry from their Bloch phases
        over those k-points and operations, in the units of ``H``.

    Returns
    -------
    float
        The tolerance, in the units of ``H``: ``5e-14·max|H|`` for a few
        operations on a few orbitals, and ``1.2e-12`` for the 96 operations of
        ``Im-3m1'`` on 32 states with ``max|H| = 1.76``.  It is proportional
        to ``max|H|``, so it does not depend on the units ``H`` is written in.
    """
    scale = float(scale)
    return max(EXACT_TOL * (scale if scale > 0.0 else 1.0),
               float(n_operations) * float(nstate) * float(np.finfo(float).eps) * scale,
               float(phases))


def _phase_rounding(bloch, group, operations, k_pts) -> float:
    r"""The rounding of the Bloch phases a covariance ``‖D H(k) D† − H(Rk)‖`` carries.

    The most :meth:`.._bloch._BlochSum.phase_rounding` gives at ``k``, added
    to the most it gives at an image ``Rk`` over the operations, each image
    taken through the operation's integer action on reduced k.

    Parameters
    ----------
    bloch : .._bloch._BlochSum
        The model's Bloch sum, :func:`.._bloch.evaluator`.
    group : .._spacegroup.GroupData
        The group the operations are of.
    operations : iterable of irrep.symmetry_operation.SymmetryOperation
        The operations whose covariance is measured.
    k_pts : array of shape (n, dim_k)
        The k-points it is measured at.

    Returns
    -------
    float
        In the units of ``H``.
    """
    images = 0.0
    for op in operations:
        act = group._k_action_of(op)
        if act is not None:
            images = max(images, bloch.phase_rounding(k_pts, act(np.eye(group._dim_k))))
    return bloch.phase_rounding(k_pts) + images


def _covariance_battery(group, k_pts, n_battery, seed):
    """The k-points a representation's covariance is measured at.

    ``k_pts`` is the points given, in reduced coordinates, or ``None``, which
    draws ``n_battery`` points uniformly in ``[-0.5, 0.5)`` along each of the
    k-directions of ``group``, from a generator seeded with ``seed``.  Returns
    an array of shape ``(n, dim_k)``.
    """
    if k_pts is None:
        rng = np.random.default_rng(int(seed))
        k_pts = rng.uniform(-0.5, 0.5, size=(int(n_battery), group._dim_k))
    return np.atleast_2d(np.asarray(k_pts, dtype=float))


def _default_of(func, name):
    """The default of the argument ``name`` of ``func``, read off its signature.
    """
    return inspect.signature(func).parameters[name].default


#: The tolerance :func:`..symmetry.bloch_operator` checks the integer-shift
#: identity ``A·τ_a + t = τ_{g(a)} + s`` to, read off its own signature: the
#: statement that the operations carry the centers onto one another, which the
#: projection's operators are built on.
_BLOCH_TOL = float(_default_of(bloch_operator, "tol"))
