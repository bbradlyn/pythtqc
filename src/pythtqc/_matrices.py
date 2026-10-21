"""The arithmetic of the operations' matrices: their composition by the group's
table, the phase each one is put in and what anchors it, the Bloch operator of
an operation, the members of a solution space inside given blocks, the
irreducible representations of a group with a factor system, and the one-
dimensional characters of a group and its subgroups.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import inspect
import itertools
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
# the constructor half
# ==========================================================================
#
# Everything below turns "which operations hold" into "what matrix each one
# is".  Three rules are load-bearing and are enforced rather than left to
# convention:
#
#   * nothing group-specific is typed here — orders, required powers, spinor
#     factors and translation differences all come from
#     ``GroupData.get_product_table(get_diff=True)``, site permutations and
#     their integer lattice shifts from ``_tables._assign_orbitals``;
#   * discrete rep choices are *searched*, never assumed, and a non-unique or
#     empty search is returned as a ``Refusal``, never swallowed;
#   * every returned object carries the residual of the identity it claims.


def _identity_position(group) -> int:
    """Index (into ``group.symmetries``) of the identity operation."""
    eye = np.eye(3, dtype=int)
    for p, sym in enumerate(group.symmetries):
        if (not sym.time_reversal and np.all(np.asarray(sym.rotation, dtype=int) == eye)
                and np.allclose(np.asarray(sym.translation) % 1.0, 0.0, atol=1e-8)):
            return p
    raise ValueError("the group carries no identity operation — not a group")


def _order_and_zeta(group, p, prod=None, spf=None):
    r"""Order of operation ``p`` and the value its ``n``-th power must take.

    ``n`` is the smallest integer with :math:`g^n` equal to the identity
    *modulo lattice translations*, read off the group's own product table; and

    .. math:: \zeta = \prod_{m=1}^{n-1} \mathrm{spf}(g, g^{m})

    is the accumulated spinor factor, so that a canonically normalized
    representation satisfies :math:`D^{n} = \zeta\,\mathbb 1` — for a spinful
    two-fold operation :math:`\zeta = -1`, for a spinless one :math:`+1`, and
    nothing about which is typed here.

    Returns
    -------
    n : int
    zeta : int
    """
    if prod is None or spf is None:
        prod, _, spf = group.get_product_table(get_diff=True)
    ident = _identity_position(group)
    q, n, zeta = p, 1, 1
    limit = len(group.symmetries) + 1
    while q != ident:
        zeta *= int(spf[p, q])
        q = int(prod[p, q])
        n += 1
        if n > limit:  # pragma: no cover - impossible for a genuine group
            raise ValueError(f"operation {p} has no finite order in this product table")
    return int(n), int(zeta)


def _compose(D1, antiunitary1, D2):
    """``D1 ∘ D2`` — the antiunitarity of the *left* factor conjugates the right."""
    return D1 @ (np.conj(D2) if antiunitary1 else D2)


def _power(D, antiunitary, n):
    """``n``-fold antiunitary-aware power ``D ∘ D ∘ … ∘ D``."""
    out = np.eye(D.shape[0], dtype=complex)
    for _ in range(n):
        out = _compose(D, antiunitary, out)
    return out


def _product_table_residual(D, positions, syms, prod, spf):
    r"""``max ‖D(g₁)∘D(g₂) − spf(g₁,g₂)·D(g₁g₂)‖`` over the represented pairs.

    The group's own multiplication table — spinor factors included —
    measured on a set of candidate matrices.  One implementation, used twice:
    :func:`.._build._build_rep` applies it **inside** the uniqueness gate (a candidate that
    is Hamiltonian-covariant but does not multiply is not a representation of
    the group, whatever its covariance says) and again at the end to fill
    :attr:`SpaceGroupRep.product_residual`.  The two agree to the last bit:
    the only thing that happens in between is the multiplication by the
    one-dimensional character ``χ``, a homomorphism with ``χ = ±1``, so it
    cancels out of the identity.

    ``positions`` are indices into ``syms``/``prod``/``spf``; ``D`` is keyed the
    same way.  Pairs whose product falls outside the represented set are
    skipped — for a closed set (which :func:`.._build._build_rep` checks separately)
    there are none.
    """
    worst = 0.0
    for a in positions:
        Da, anti_a = D[a], bool(syms[a].time_reversal)
        for b in positions:
            c = int(prod[a, b])
            if c not in D:
                continue
            lhs = _compose(Da, anti_a, D[b])
            worst = max(worst, float(np.abs(lhs - int(spf[a, b]) * D[c]).max()))
    return worst


def _composition_defect(D, anchors, positions, syms, prod):
    r"""How far matrices are from composing by the group's table, a phase per product allowed.

    ``max |D(a)∘D(h) − ρ·D(ah)|`` over the operations ``a`` the matrices were
    propagated from — the generators and the time reversal — against every
    represented ``h``, ``ρ`` the best phase for each product: zero to rounding
    for matrices that form a projective representation of the group at all,
    whatever their phases, and the noise a near-symmetric model's own blocks
    carry otherwise.  The matrices are defined along the generators' word
    tree, so these relations are the whole multiplication table.
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
    r"""The exact representation nearest matrices that nearly form one.

    ``D`` holds one unitary matrix per represented operation, unitary and
    antiunitary.  For an exact projective representation with factor system
    ``ω``, ``D(h)∘D(g) = ω(h, g)·D(hg)`` for every pair, so ``D(g)`` is the
    average over ``h`` of ``ω(h, g)·D(h)⁻¹D(hg)`` — conjugated where ``h`` is
    antiunitary.  For matrices that only nearly compose, that average is a
    nearer representation: the defect is squared in a round, up to a factor of
    order one (``1.2e-6`` to ``1.1e-13`` on a spinless ``R-3m`` model jittered
    at ``1e-7``), and a few rounds reach rounding.  ``ω`` is read off the matrices each round, as the phase
    with which each product is a multiple of its partner's matrix, so what
    comes back carries the factor system nearest the matrices' own, which the
    canonical norm then puts in the tables' (:func:`normalize_rep`).  Each
    matrix moves by the defect it had and no more, so a phase the build read
    off the model stays where it was.

    Returns ``{position: matrix}``, or ``None`` where the rounds do not
    converge: matrices that are no near-representation of the group at all,
    which the canonical norm then refuses.
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
    r"""The generators' phases, among those their canonical norms admit, that multiply.

    ``D`` holds matrices propagated from the unitary generators ``gens`` along
    the word tree ``(words, parent)`` of :func:`_bfs_words`, each generator's
    phase the one :func:`normalize_rep` chose.  The canonical norm
    ``D(g)ⁿ = ζ`` fixes that phase only up to an ``n``-th root of unity, and
    ``offers[i]`` lists the multipliers generator ``i`` may still take: the
    roots of unity its norm admits, ``1`` first.  Multiplying generator ``i`` by
    ``wᵢ`` multiplies every propagated ``D(h)`` by ``W(h)``, the product of the
    ``w``'s along its word.  So the group's relation
    ``D(g)D(h) = spf(g, h)·D(gh)``, measured once as
    ``D(g)D(h) = ω(g, h)·spf(g, h)·D(gh)``, holds for the rephased matrices
    exactly where ``wᵢ·W(h)·ω(gᵢ, h) = W(gᵢh)``.  Asking it of every generator
    against every operation asks it of the whole multiplication table: the
    matrices are defined along the word tree, and every other product follows
    from these by the table's own factor system.  The combinations are tried in
    the order offered, so phases the norms chose that already multiply are the
    ones returned.

    Returns
    -------
    dict
        ``roots``: one multiplier per generator, from the first combination
        that closes every relation to ``tol``, or ``None`` where none does.
        ``projective``: whether every ``D(gᵢ)D(h)`` is a phase times
        ``spf(gᵢ, h)·D(gᵢh)`` at all -- whether the matrices form a projective
        representation of the group, which no rephasing can make of a set
        that is not one.  ``defect``: the worst departure from that
        proportionality.  ``residual``: the smallest worst-relation residual a
        combination reached.  ``tried``: how many combinations were tried.
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
    r"""Every pair of commuting operations whose matrices commute only up to a phase the table lacks.

    Where ``ab = ba`` the group's own relation gives
    ``D(a)D(b) = spf(a, b)·spf(b, a)·D(b)D(a)``, and a phase on either matrix
    cancels from it.  A measured ratio ``c`` other than ``1`` between the two
    sides is therefore a statement no choice of phases changes: the matrices
    form a projective representation whose class is not the table's.  Spin-½
    states read in single-valued tables are the case: two perpendicular
    twofolds commute, and their spinor matrices anticommute.

    ``tol`` bounds both how far ``D(a)D(b)`` may sit from a multiple of the
    other side and how close to ``1`` the ratio may come and still read as the
    table's.

    Returns
    -------
    tuple
        ``(a, b, c)`` per such pair, positions and the measured ratio, in
        operation-index order; empty where every commuting pair commutes as
        the table says.
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
    r"""The first pair of :func:`_commuting_pair_phases`, by operation index, or ``None``."""
    found = _commuting_pair_phases(D, positions, syms, prod, spf, tol=tol)
    return found[0] if found else None


def _rephasable(D, positions, group, prod, spf, *, tol=1e-6):
    r"""Whether matrices on a subgroup are a representation of it once rephased.

    ``D`` holds one matrix per position of the subgroup ``positions`` --
    members of the operations' solution spaces, each with its own free phase.
    They are a representation of the subgroup, with the group's own factor
    system, exactly where some phase per operation makes every product
    ``D(g)D(h) = spf(g, h)·D(gh)`` hold: the projective class they carry is
    trivial there.  The question is asked the way the representation is
    assembled: each generator is brought to its canonical norm
    ``D(g)ⁿ = ζ`` (:func:`normalize_rep`), which a representation of the
    subgroup satisfies and which leaves the phase free up to an ``n``-th root of
    unity, the rest is propagated along the generators' word tree, and the
    relations' scalars are asked of every combination of those roots
    (:func:`_multiplying_roots`).

    ``tol`` is how far a relation may miss and still hold, and how far a
    generator's ``n``-th power may sit from a multiple of the identity.
    """
    return _rephased(D, positions, group, prod, spf, tol=tol) is not None


def _rephased(D, positions, group, prod, spf, *, tol=1e-6):
    r"""Matrices on a subgroup, rephased into a representation of it; or ``None``.

    The measurement :func:`_rephasable` reads, with its matrices: each
    generator in its canonical norm ``D(g)ⁿ = ζ`` and the positivity convention
    (:func:`normalize_rep`), times the roots of unity that make every relation
    of the product table hold (:func:`_multiplying_roots`), and every other
    operation propagated along the generators' word tree.  ``spf`` is the
    factor system asked for -- the group's own, or all ones for the trivial
    one.  Returns ``{position: matrix}`` over the subgroup, the identity
    included, or ``None`` where no phases make them a representation with that
    factor system.
    """
    ident = _identity_position(group)
    positions = sorted({int(p) for p in positions} | {int(ident)})
    gens = _minimal_generators(positions, prod, ident)
    if gens is None:
        return None
    if not gens:
        return {int(ident): np.eye(np.asarray(D[int(ident)]).shape[0], dtype=complex)}
    words, parent = _bfs_words(gens, prod, ident, positions)
    syms = group.symmetries
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
    """Power of ``λ`` picked up by ``(λD)`` composed ``n`` times.

    A unitary operator contributes ``λ`` every time; an antiunitary one
    alternates ``λ`` and ``λ̄``, so an even power is **phase-blind** — which is
    why the canonical norm alone can never fix the phase of a time-reversal
    operator, and the anchor/positivity step is not optional there.
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
    """Best ``rho`` with ``A ≈ rho·B`` (``B`` nonzero), and the residual."""
    denom = float(np.vdot(B, B).real)
    if denom <= 0:
        return 0.0 + 0.0j, float(np.abs(A).max())
    rho = complex(np.vdot(B, A) / denom)
    return rho, float(np.abs(A - rho * B).max())


def _principal_entry(block, atol=1e-9):
    """The entry a positivity condition is applied to: largest modulus, first index."""
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
    """Sort key making "positive" mean ``Re > 0``, ties broken by ``Im > 0``."""
    return (round(float(np.real(z)), 12), round(float(np.imag(z)), 12))


def _table_anchored(candidates, block, internal, *, op_ind, order, zeta, tol=1e-9):
    r"""The candidates whose reference block is a **real** multiple of the tables' ``S``.

    The table anchor of :class:`RepAnchor`, as one measurement.  The tensor
    structure ``D(g) = O(g) ⊗ S(g)`` puts the operation's own ``sym.spinor_rotation`` in
    the reference block (the reference class carries ``+1`` by construction), so
    the normalized block is ``lam·S`` — the extracted internal factor *is* the
    table's, times one phase.  The tables' lift is pseudo-real, so that phase
    can only be real, and this returns the candidates that respect it.

    Raises
    ------
    ..refusal.InternalError
        When the block is not proportional to ``S`` at all (the built matrices
        are not the matrices the tables ship), or when no candidate survives.
        Both are statements about *this package's* use of the tables — the
        convention fixes the gauge completely, so there is nothing to hand back
        and nothing to choose.
    """
    S = np.asarray(internal, dtype=complex)
    B = np.asarray(block, dtype=complex)
    where = (f"operation #{op_ind}: the tables fix this gauge completely (the group ships "
             f"no antiunitary operation, so the reality of the phase comes from the "
             f"pseudo-real SU(2) lift the internal factors are written in). "
             f"Failing to pin it is a bug in this package's application of that "
             f"convention, not a property of the model")
    if B.shape != S.shape:
        raise InternalError(
            f"{where}: the reference block has shape {B.shape} against the operation's own "
            f"internal factor {S.shape}.")
    rho, res = _proportionality(B, S)
    if res > max(tol, 1e-10) or abs(rho) < 0.5:
        raise InternalError(
            f"{where}: the reference block is not a multiple of the operation's own "
            f"sym.spinor_rotation (best ratio {rho:.6g}, defect {res:.3g} against "
            f"{max(tol, 1e-10):.3g}), so the matrix that was built is not written in the "
            f"tables' lift and no phase can put it there.\n"
            f"    block = {np.array2string(B, precision=4, max_line_width=200)}\n"
            f"    S(g)  = {np.array2string(S, precision=4, max_line_width=200)}")
    cands = (1.0 + 0.0j,) if candidates is None else tuple(candidates)
    kept = tuple(lam for lam in cands
                 if abs(complex(lam * rho).imag) <= max(tol, 1e-10) * max(abs(lam * rho), 1.0))
    if not kept:
        raise InternalError(
            f"{where}: the canonical norm D**{order} = {zeta:+d} leaves the phases "
            f"{tuple(f'{c:+.4f}' for c in cands)}, and none of them makes the reference "
            f"block a real multiple of the operation's own sym.spinor_rotation (the block "
            f"is {rho:+.6g} x S, so the products are "
            f"{tuple(f'{complex(c * rho):+.4f}' for c in cands)}). A complex ratio is the "
            f"value of a one-dimensional character of the group: multiplying every matrix "
            f"by one preserves covariance, the canonical norm and the whole product table, "
            f"and relabels every corep at once.")
    return kept


@dataclass(frozen=True)
class RepAnchor:
    """What fixes the overall phase of an operation's matrix, which no covariance can see.

    A matrix carrying ``H(k)`` onto ``H(Rk)`` does so whatever its overall
    phase, and a wrong phase relabels every corep at once.  Time reversal, or
    for a group with none the tables' own spin lift, and the sign of a
    reference block's largest entry fix it (:func:`normalize_rep`).  Each
    coupled part of the crystal is anchored at its own reference block, so
    anchoring a representation at another block can multiply its matrices by
    a one-dimensional character of the group: the labelling, which is chosen
    after the phase is fixed.

    Attributes
    ----------
    antiunitary : numpy.ndarray or None
        ``T``, the matrix of an antiunitary operation commuting with every
        unitary one, the group's pure time reversal: the condition is
        ``T conj(D) T† = D``, blind to ``T``'s own phase.  ``None`` where the
        group has none.
    rows, cols : tuple of int
        The states of the reference block ``D[rows, cols]``, the lead
        multiplet of the part of the crystal being anchored and its image; it
        must not be zero.  Its largest entry, the first on ties, is given a
        positive real part, or a positive imaginary part where its real part is
        zero.
    internal : numpy.ndarray or None
        The operation's own spin factor from the tables, ``[[1]]`` without
        spin, of which the reference block must be a real multiple: what time
        reversal would fix, where the group has none.
    atol : float
        How far a condition may miss, and below which a block entry is zero.
    """

    antiunitary: np.ndarray | None = None
    rows: tuple = ()
    cols: tuple = ()
    internal: np.ndarray | None = None
    atol: float = 1e-9


@dataclass(frozen=True)
class NormalizedRep:
    """An operation's matrix in its canonical phase, with the phase applied and each condition's residual after it.

    :func:`normalize_rep` returns it.

    Attributes
    ----------
    matrix : numpy.ndarray
        The matrix in its canonical phase, ``λ·D``.
    factor : complex
        ``λ``: ``abs(factor - 1)`` is how far the input was from canonical.
    order, required : int
        The operation's order ``n``, modulo lattice translations, and ``ζ``, the
        sign the group's table gives ``D^n = ζ·1``.
    norm_residual : float
        ``max|D^n − ζ·1|`` after the phase is applied.
    norm_informative : bool
        ``False`` for an antiunitary operation of even order, whose ``n``-th
        power cannot see the phase.
    anchor_residual : float or None
        ``max|T conj(D) T† − D|`` after the phase is applied; ``None`` without
        time reversal.
    anchor_used, table_anchored, product_anchored : bool
        Whether time reversal, the tables' own spin lift, and the group's
        multiplication table narrowed the phases allowed; the last where
        nothing else fixes the phase, a group with no time reversal and a
        reference multiplet of several orbitals, or one whose character no
        one-dimensional character of the group restricts to.
    candidates, accepted : tuple of complex
        Every ``λ`` the canonical norm allowed, and those time reversal and
        the tables' lift allowed of them, which the reference block chose from.
    principal_entry : complex
        The reference block's entry the choice was read on; ``0`` where the
        block was empty, and ``D``'s own largest entry was read instead.
    unitarity, polar_shift : float
        How far the input was from unitary, ``max|D†D − 1|``, and how far its
        nearest unitary moved it, ``max|D − polar(D)|``.
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
    phase, and a wrong one, ``(−i)·D`` passing every covariance and norm test,
    relabels every corep at once.  The canonical norm ``D^n = ζ·1``, ``n`` the
    operation's order and ``ζ`` from the group's own multiplication table,
    fixes the phase up to an ``n``-th root of unity; time reversal,
    ``T conj(D) T† = D``, up to a sign, or the tables' own spin lift where the
    group has no time reversal; and the reference block's largest entry, given
    a positive real part, settles the sign.  Normalized one by one, the
    matrices of a group need not multiply as it does: a representation
    normalizes its generators and carries the rest along the group's table.

    Parameters
    ----------
    D : array, shape (nstate, nstate)
        The matrix; its nearest unitary is taken first.
    op : irrep SymmetryOperation
        The operation it represents, one of ``group``'s.
    group : ..groups.GroupData
        The group, read for its multiplication table alone.
    spinful : bool, optional
        Checked against ``group.spinor``, which the table already carries.
    anchor : RepAnchor, optional
        Time reversal's matrix, the reference block, and the tables' spin lift.
        Without it, ``D``'s own largest entry settles the sign.
    tol : float, optional
        How far ``D^n`` may miss a multiple of the identity.  Default ``1e-9``.
    prod, spf : array, optional
        The group's product and spinor-factor tables, computed once for a loop
        over its operations.

    Returns
    -------
    NormalizedRep or Refusal
        ``rec.matrix``, the matrix in its canonical phase; ``rec.factor``, the
        phase applied; and each condition's residual after it.  A ``Refusal`` in
        its place is falsy.

    Raises
    ------
    ValueError
        ``D`` not square, ``op`` not one of ``group``'s, or ``spinful`` against
        ``group.spinor``.
    ..refusal.InternalError
        ``anchor.internal`` given and the reference block no real multiple of
        it, which the tables fix: a defect to report.

    Refusals
    --------
    ``canonical_norm``
        ``D^n`` is not a multiple of the identity: the matrix does not represent
        the operation.
    ``antiunitary_anchor``
        no phase the canonical norm allows satisfies ``T conj(D) T† = D``: the
        matrix and time reversal belong to different representations.

    Examples
    --------
    A wrong overall phase, undone::

        rec = normalize_rep(-1j * D_true, op, group, anchor=anchor)
        abs(rec.factor - 1j) < 1e-12      # the (−i) undone
        np.allclose(rec.matrix, D_true)   # True
    """
    D = np.asarray(D, dtype=complex)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"D must be a square matrix, got shape {D.shape}")
    if spinful is not None and bool(spinful) != bool(group.spinor):
        raise ValueError(f"spinful={spinful} contradicts group.spinor={group.spinor}")

    pos = {int(s.ind): i for i, s in enumerate(group.symmetries)}
    if int(op.ind) not in pos:
        raise ValueError(f"operation .ind={op.ind} is not one of this group's")
    p = pos[int(op.ind)]
    anti = bool(op.time_reversal)

    eye = np.eye(D.shape[0])
    unitarity = float(np.abs(D.conj().T @ D - eye).max())
    w, _, vh = np.linalg.svd(D)
    U = w @ vh
    polar_shift = float(np.abs(D - U).max())

    if prod is None or spf is None:
        prod, _, spf = group.get_product_table(get_diff=True)
    n, zeta = _order_and_zeta(group, p, prod=prod, spf=spf)
    p_exp = _phase_exponent(n, anti)

    c, defect = _scalar_of(_power(U, anti, n))
    if defect > max(tol, 1e-10) or abs(c) < 0.5:
        return Refusal(
            "canonical_norm",
            f"operation #{op.ind}: D**{n} is not a multiple of the identity "
            f"(defect {defect:.3g}, scalar |c| = {abs(c):.3g}), so this matrix does not "
            f"represent the operation and no phase can rescue it.",
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
                    f"operation #{op.ind}: no phase satisfies both the canonical norm "
                    f"D**{n} = {zeta:+d} and the antiunitary anchor T conj(D) T† = D "
                    f"(the anchor demands lam**2 = {rho:.6g}).  The matrix and the anchor "
                    f"belong to different representations; corep labels built on it "
                    f"would be meaningless.",
                    value=float(min(abs(lam - r) for lam in candidates for r in roots)),
                    threshold=1e-6,
                    context={"operation_ind": int(op.ind), "order": n, "required": zeta,
                             "rho": rho, "candidates": candidates, "roots": roots})
        else:
            anchor_residual = prop      # anchor present but not proportional: report it

    rows = np.asarray(anchor.rows, dtype=int)
    cols = np.asarray(anchor.cols, dtype=int)
    block = U[np.ix_(rows, cols)] if (rows.size and cols.size) else U

    # --- the table anchor: what the group's own internal factor still fixes ---
    # Applied where the reference block is the tensor structure's, i.e. nonzero: a
    # candidate whose reference block vanishes is the g^-1 branch of an
    # operation that moves the reference multiplet, which the internal-factor
    # gate and the product table are about to discard on their own.
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
                f"operation #{op.ind}: the table anchor leaves {len(tied)} phases "
                f"{tuple(f'{c:+.4f}' for c in tied)} that the reference orbital's "
                f"positivity condition cannot separate (they agree on its principal entry "
                f"{entry:+.4g}). The tables plus the reference orbital fix this gauge "
                f"completely, so a tie here is a bug in this package's application of that "
                f"convention.")

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
    r"""Every operation's matrix, propagated from the generators' along their word tree.

    The table says ``D(gᵢ)·D(r) = spf(gᵢ,r)·D(gᵢr)``, so the tree edge
    defines ``D(gᵢr) := spf(gᵢ,r)⁻¹ D(gᵢ)D(r)`` — and ``spf = ±1``, so the
    inverse is itself.  Doing it this way (rather than normalizing every
    element on its own) is what makes the full product table close: an
    independently phase-fixed ``D(C₂)`` and ``D(C₄)²`` can legitimately
    differ by a sign.  An antiunitary operation ``p = T·u`` of ``au_of``
    (``{p: u}``) is ``spf(T, u)·D(T) conj(D(u))``, with ``DT`` the matrix of
    the time reversal ``T_pos``.
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
    r"""Matrices put in their canonical phase: ``(refusal, D, normalization, anchor_T, DT_n, defect)``.

    ``Dsearch`` holds one matrix per operation of ``kept``, propagated along the
    generators' word tree, each generator with the free phase of the solution
    space it was read from.  The generators are brought to their canonical norm
    and the operation's anchor (:func:`normalize_rep`; ``anchor_of(p, anchor_T)``
    is the :class:`RepAnchor` of generator ``p``), and every other operation is
    propagated from them (:func:`_propagated`).  What is left is a
    representation of the group fixed up to a one-dimensional character: a real
    one where time reversal anchors the phase, and a root of unity per generator
    where nothing does, which the multiplication table then narrows
    (:func:`_table_phases`).

    **The exact representation nearest the model's own blocks.**  Matrices
    built from a near-symmetric model's own blocks carry its noise, and compose
    by the table only to that noise, so no canonical norm can be read off them.
    Where they miss by more than the norm's own gate and by no more than
    ``reach``, they are replaced by the exact representation nearest them
    (:func:`_nearest_representation`): a projection onto what the operations
    can be, never a loosened gate.  An exact model's matrices compose to
    rounding and are read as they stand.

    ``refusal`` is :func:`normalize_rep`'s own verdict where a generator has no
    canonical norm, and ``None`` otherwise; ``defect`` is the composition defect
    the matrices were read with.
    """
    syms = group.symmetries
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
    r"""``(found, rephased)`` — the generators' phases the multiplication table chooses.

    The canonical norm fixes each generator's phase only up to an ``n``-th root
    of unity.  Time reversal narrows that to a sign, and so does the tables'
    own lift on a one-dimensional reference multiplet; positivity on the
    reference block picks inside what is left.  A group with no antiunitary
    operation and a multi-dimensional reference multiplet has neither anchor --
    the multiplet's block is measured, and no table lift stands behind it -- so
    positivity alone picks among all ``n`` roots, and it is a convention about
    one matrix entry, not about the group: a sixfold's phase a cube root of
    unity away from one that multiplies misses the table by
    ``|1 - exp(2 pi i / 3)| = sqrt(3)``.  There the table itself chooses, among
    the roots each norm admits, in positivity's order, so a choice that already
    multiplies is the one kept (:func:`_multiplying_roots`).  So it does
    wherever positivity's sign does not multiply: a one-dimensional reference
    multiplet whose character on its site group no one-dimensional character
    of the group restricts to -- ``p_y`` at a ``3c`` site of ``P23``, odd under
    two of the three twofolds the threefolds carry onto one another -- carries
    that character, not the positive sign, on the generators that fix it.

    ``Dn`` and ``normalization`` are :func:`_settled`'s.  ``found`` is
    :func:`_multiplying_roots`' record, and ``rephased`` is ``(D, normalization)``
    with the roots it found multiplied in, or ``None`` where it found none or
    none but the ones already taken.
    """
    syms = group.symmetries
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
    r"""``(refusal, D, normalization, anchor_T, DT_n, defect, roots)`` -- matrices settled.

    The one settling every representation built here goes through: each
    generator in its canonical norm and anchor, the rest propagated
    (:func:`_settled`); and wherever the phases positivity picks do not
    multiply, the multiplication table choosing among the roots each
    canonical norm admits, in positivity's order (:func:`_table_phases`).
    Positivity is a convention about one entry of the lead multiplet's block,
    and it fails to multiply where nothing anchors the generators' phases --
    no antiunitary operation, and a multi-dimensional lead with no table lift
    behind its block -- and where the lead is a one-dimensional multiplet
    whose character on its site group no one-dimensional character of the
    group restricts to, ``p_y`` at a ``3c`` site of ``P23``.  ``tree`` is the
    generators' word tree, ``kept`` and ``u_pos`` the operations and the
    unitary ones.  ``roots`` is ``(found, matrices)`` where no root makes the
    matrices multiply, and ``None`` otherwise.
    """
    syms, spf = group.symmetries, tree["spf"]
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
    """Model orbital positions in the group's 3D cell, reduced coordinates.

    Delegated to ``_spacegroup._pad_cell`` so that the 3D presentation is
    literally the one the group was identified in — a second implementation of
    it would be a second convention.
    """
    _lat, pos3, _padded = _spacegroup._pad_cell(model, group.vacuum_length,
                                            lattice=group.lattice)
    return np.asarray(pos3, dtype=float)


def bloch_operator(model, group, assignment, op, D, k, *, k_out=None, tol=1e-9,
                   check=True):
    r"""The Bloch operator ``U(g;k)`` of one operation, with ``U(g;k) H(k) U(g;k)† = H(k_out)``.

    In pythtb's convention the Bloch basis carries
    :math:`e^{2\pi i k\cdot(\mathbf R + \tau)}`, so for :math:`g = \{A|t\}`

    .. math::
        U(g;k) = e^{-2\pi i\,k'\cdot t}\;
                 \operatorname{diag}\!\left(e^{2\pi i\,(k'-k_{\rm out})\cdot
                 \tau_{j}}\right)\, D(g),
        \qquad k' = (A^{-1})^{T}k,

    with ``H(k)*`` in place of ``H(k)`` for an operation combined with time
    reversal.  At ``k_out = k'``, the image not folded back into the zone, the
    diagonal is the identity, which is why ``D(g)`` is independent of ``k``;
    the orbital positions return when the image is folded back, as at a ``k``
    the operation leaves invariant, which is where a sewing matrix is read.
    The factor ``e^{−2πi k'·t}`` cancels in ``U H U†`` and carries the
    translations' phases into the products,
    ``U(g₁;g₂k) U(g₂;k) = spf·e^{−2πi (g₁g₂k)·Δ}·U(g₁g₂;k)`` with
    ``Δ = t₁ + A₁t₂ − t₃``, ``spf`` the group's spinor factor: what a glide's
    or a screw's little-group characters rest on.  Each row's phase is read
    off the position its preimage orbital is carried to, and the lattice
    vector the image lands in is checked against the orbital assignment.

    Parameters
    ----------
    model : pythtb.TBModel
        The model the representation was built on.
    group, assignment : GroupData, OrbitalAssignment
        Its group, and the orbitals' Wyckoff orbits with each operation's
        permutation of the sites.
    op : irrep SymmetryOperation
        The operation.
    D : array, shape (nstate, nstate)
        Its ``k``-independent matrix, ``rep.matrices[op.ind]``.
    k : array, shape (dim_k,) or (Nk, dim_k)
        Reduced ``k`` in the model's periodic directions.
    k_out : array, optional
        Where the transformed Hamiltonian is read, differing from ``k'`` by a
        reciprocal lattice vector.  Default ``k'`` itself; ``k_out = k`` at a
        ``k`` the operation leaves invariant.
    tol : float, optional
        How far the checks of that vector and of the preimages' positions may
        miss.  Default ``1e-9``.
    check : bool, optional
        Run those checks.  Default ``True``.

    Returns
    -------
    numpy.ndarray
        ``U(g;k)``, ``(nstate, nstate)`` for one ``k`` and ``(Nk, nstate,
        nstate)`` for several.

    Raises
    ------
    ValueError
        ``k' − k_out`` is not a reciprocal lattice vector, a row of ``D`` has
        preimages at positions the operation does not carry to one, or the
        operation carries ``k`` out of the model's periodic plane.
    """
    D = np.asarray(D, dtype=complex)
    k_arr = np.asarray(k, dtype=float)
    squeeze = (k_arr.ndim == 1)
    k_arr = np.atleast_2d(k_arr)
    if k_arr.shape[1] != group.dim_k:
        raise ValueError(f"k must have {group.dim_k} components, got {k_arr.shape[1]}")

    k3 = group.k_to_3d(k_arr)
    kp3 = np.asarray(op.transform_k(k3), dtype=float)
    if not group.in_plane(kp3):
        raise ValueError(f"operation #{op.ind} moves in-plane k out of the periodic plane; "
                         "it has no Bloch operator on this model's k-space")
    if k_out is None:
        kout3 = kp3
    else:
        kout3 = group.k_to_3d(np.atleast_2d(np.asarray(k_out, dtype=float)))
        if kout3.shape[0] == 1 and kp3.shape[0] > 1:
            kout3 = np.repeat(kout3, kp3.shape[0], axis=0)
    n3 = kp3 - kout3
    if check and np.abs(n3 - np.rint(n3)).max() > tol:
        raise ValueError("k_out must differ from the rotated k by an integer reciprocal-lattice "
                         f"vector; got k' - k_out = {n3} (max non-integrality "
                         f"{np.abs(n3 - np.rint(n3)).max():.3g})")
    n3 = np.rint(n3)

    nspin = int(model.nspin)
    norb = int(model.norb)
    pos3 = _orb_positions_3d(model, group)
    rot = np.asarray(op.rotation, dtype=float)
    trans = np.asarray(op.translation, dtype=float)
    p = next(i for i, s in enumerate(group.symmetries) if int(s.ind) == int(op.ind))
    site_of = np.asarray(assignment.site_of_orbital, dtype=int)

    # row positions from the operation's action on each row's PREIMAGE orbital
    row_pos = np.zeros((norb, 3))
    thresh = max(tol, 1e-12) * max(1.0, float(np.abs(D).max()))
    for b in range(norb):
        block = D[b * nspin:(b + 1) * nspin, :]
        cols = np.flatnonzero(np.abs(block).max(axis=0) > thresh)
        if cols.size == 0:
            raise ValueError(f"row block {b} of D is identically zero: the matrix does not "
                             f"represent operation #{op.ind} on every orbital")
        images = [rot @ pos3[int(c) // nspin] + trans for c in cols]
        base = images[0]
        for img in images[1:]:
            d = img - base
            if check and np.abs(d - np.rint(d)).max() > tol:
                raise ValueError(f"row block {b} of D has preimages whose images differ by a "
                                 f"non-lattice vector {d}: the site permutation and the matrix "
                                 "disagree")
        row_pos[b] = base
        if check:
            a = int(cols[0]) // nspin
            shift = np.asarray(assignment.lattice_shift[p, site_of[a]], dtype=float)
            resid = base - shift - pos3[b]
            if np.abs(resid - np.rint(resid)).max() > tol:
                raise ValueError(
                    f"the integer-shift identity A·tau_a + t = tau_g(a) + s fails for orbital "
                    f"{a} -> {b} under operation #{op.ind} (residual {resid}); the assignment "
                    "and the representation describe different site permutations")

    glob = np.exp(-2j * np.pi * (kp3 @ trans))
    lam = np.exp(2j * np.pi * (n3 @ row_pos.T))              # (Nk, norb)
    lam = np.repeat(lam, nspin, axis=1)                      # (Nk, nstate)
    U = glob[:, None, None] * lam[:, :, None] * D[None, :, :]
    return U[0] if squeeze else U


# --------------------------------------------------------------------------
# group plumbing for the constructor
# --------------------------------------------------------------------------

def _closure(gens, prod, ident):
    """Subgroup generated by ``gens`` (positions), via the product table."""
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
    """Smallest subset of ``positions`` generating all of it, in index order."""
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
    """Shortest words in ``gens`` and the tree they were found on.

    Returns ``(words, parent)`` with ``words[q] = (i, j, …)`` meaning
    ``gᵢ·gⱼ·…`` and ``parent[q] = (i, r)`` meaning ``q = gᵢ·r`` — the edge the
    representation is propagated along, so that no word ever has to be
    re-parsed.
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
    """True when the operation does not act within the periodic block."""
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
    """Image orbital of every orbital under symmetry position ``p``.

    Orbitals are matched within a site by their slot, their position in
    ``order`` -- each site's orbitals in the order of the ones they are images
    of at the reference site of their orbit (:func:`.._multiplets._slot_order`), or in
    ``assignment.site_orbitals`` where no order is given -- the site
    permutation itself coming from ``assignment.permutations``, never typed.
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
    """Image multiplet of every multiplet under symmetry position ``p``.

    A multiplet is a set of orbitals at one site, and its image is the multiplet
    the orbital permutation carries those orbitals onto: the site permutation
    of ``assignment.permutations`` composed with the slots the multiplets were
    measured in (their slot order), and nothing typed.  The multiplets are the
    same partition of the slots at every site of an orbit
    (:func:`.._multiplets._site_multiplets` transports it), so the image is a multiplet and not
    a set of pieces.
    """
    orb = _orbital_permutation(assignment, p, model,
                               order=getattr(layout, "_site_order", None))
    chan_of = np.asarray(layout._chan_of, dtype=int)
    out = np.empty(int(layout._nchan), dtype=int)
    for c, orbs in enumerate(layout._chan_orbs):
        out[c] = int(chan_of[int(orb[int(orbs[0])])])
    return out


def _resolve_spin(spin, model):
    """Normalize :func:`.._build._build_rep`'s ``spin=`` to a :class:`..spin.SpinBasis` or ``None``.

    Accepts the object itself, the shorthand ``"interleaved"`` / ``"blocked"``,
    or a mapping of :func:`..spin.spin_basis` keyword arguments.  Everything that
    can go wrong here is **grammar** — a declaration for the wrong model, a
    spin axis declared on a model that already has one — so it raises
    ``ValueError`` rather than refusing: none of it is a measurement.
    """

    if spin is None:
        return None
    if int(model.nspin) == 2:
        raise ValueError(
            "`spin=` declares spin encoded in the orbital index of a spinless model "
            "(nspin=1, e.g. a wannier90 import); this model is natively spinful "
            "(nspin=2) and carries its own spin axis, which every measurement here "
            "uses. Drop `spin=`.")
    if isinstance(spin, SpinBasis):
        if int(spin.norb) != int(model.norb):
            raise ValueError(
                f"the spinor basis describes a model with {int(spin.norb)} orbitals and "
                f"this one has {int(model.norb)}; a declaration is only valid for the "
                f"model it was written for.")
        return spin
    if isinstance(spin, str):
        return spin_basis(model, order=spin)
    if isinstance(spin, dict):
        return spin_basis(model, **spin)
    raise ValueError(
        f"`spin=` takes a SpinBasis, the shorthand 'interleaved'/'blocked', or a dict of "
        f"`spin_basis` keyword arguments; got {type(spin).__name__}.")


def _spin_pair_split(spin, site_orbitals, site_positions, lattice):
    r"""The first declared spinor pair whose two members sit at different centres, refused.

    The two spin states of one Wannier function share its centre, so this reads
    the centres and nothing else: the orbitals at one centre are one site,
    ``site_orbitals`` holds each site's orbitals, ``site_positions`` where the
    sites are, reduced, and ``lattice`` the cell they are reduced in.
    :func:`..groups.realized_group` asks it where the declaration enters, on
    the centres of the model as handed in, before any group is identified or
    anything measured; :func:`.._build._build_rep` asks it on the sites of the
    assignment it is handed.

    Returns the :class:`..refusal.Refusal` of the first pair split, its
    ``value`` the distance between the pair's two sites against ``0.0``, or
    ``None`` where every pair sits at one centre.
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
            f"declared spinor pair {c} = (up {int(up)}, dn {int(dn)}) has its two members "
            f"on different sites: orbital {int(up)} on site {su} at "
            f"{np.round(pos[su], 6).tolist()} and orbital {int(dn)} on site {sd} at "
            f"{np.round(pos[sd], 6).tolist()} (reduced), {dist:.6g} apart. The two spin "
            f"states of one Wannier function share its centre, so this pair is not one "
            f"spinor orbital, and it means one of two things. Either the pairing is "
            f"mislabelled: re-read the layout (`order=\"blocked\"` and "
            f"`order=\"interleaved\"` differ by exactly this) or pass the `pairs=` the "
            f"wannier90 file has. Or these orbitals each carry one spin state with no "
            f"partner at their own centre, a magnetic configuration outside this "
            f"package's present scope. Two centres apart only by the numerical noise of a "
            f"fitted model are neither: a declared precision (`precision=`) reads two "
            f"centres closer than it as one.",
            value=dist, threshold=0.0,
            context={"pair": int(c), "orbitals": (int(up), int(dn)),
                     "sites": (su, sd),
                     "positions": (tuple(float(x) for x in pos[su]),
                                   tuple(float(x) for x in pos[sd])),
                     "spin_kind": str(spin.kind)})
    return None


def _spin_channel_permutation(assignment, p, model, spin, order=None):
    """Lift the orbital-level multiplet permutation to declared ``(up, dn)`` pairs.

    Returns ``(image, split)``: ``image[c]`` is the pair the operation carries
    pair ``c`` onto, and ``split`` is ``None`` when every pair went to a pair.
    Otherwise ``image`` is ``None`` and ``split`` is
    ``(c, channel_of(image of up), channel_of(image of dn))`` for the first
    pair whose two members landed in different pairs — which :func:`.._build._build_rep`
    returns as a :class:`..refusal.Refusal` rather than raising here.

    Nothing is typed: the orbital permutation comes from
    :func:`_orbital_permutation`, i.e. from ``assignment.permutations`` and a
    slot order.  Which orbitals form a pair is read from the declaration and
    never from the order a site lists its orbitals in: ``order`` is the slot
    order the multiplets are read in, and where none is given each site's
    orbitals are read pair by pair (:func:`_pair_order`).  What is *checked*
    here is that the declaration and the site content agree about it.
    """
    orb = _orbital_permutation(assignment, p, model,
                               order=order or _pair_order(assignment.site_orbitals, spin))
    chan_of = spin.pair_of_orbital
    image = np.empty(len(spin.pairs), dtype=int)
    for c, (up, dn) in enumerate(spin.pairs):
        cu, cd = int(chan_of[int(orb[int(up)])]), int(chan_of[int(orb[int(dn)])])
        if cu != cd:
            return None, (int(c), cu, cd)
        image[c] = cu
    return image, None


def _internal_from_table(sym):
    """The internal (spin) factor of one operation: the tables' own SU(2) lift,
    and ``iσ_y`` times its conjugate for an antiunitary one."""
    S = np.asarray(sym.spinor_rotation, dtype=complex)
    if sym.time_reversal:
        S = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=complex) @ S.conj()
    return S


def _lift_of(sym, su2):
    r"""``S(g)`` — the internal factor this build represents ``sym`` with.

    Always the **tables' own** lift (:func:`_internal_from_table`), in pythtb's
    spin convention.  There is no per-operation override and there is not meant
    to be one: an operation is a symmetry of this Hamiltonian only when the
    tables' lift for *that* operation intertwines it, and representing it by some
    other SU(2) element — its ``σ_h`` partner's, say — is reading the model
    through a spin frame it was not written in.  A model written in another
    convention reads the group it genuinely has there.

    ``su2`` false is a model with no internal factor at all (a spinless model
    with no declared spinor basis), where ``S`` is ``[[1]]``.
    """
    return _internal_from_table(sym) if su2 else np.eye(1, dtype=complex)


@dataclass(frozen=True)
class SiteMultiplets:
    """The multiplets the Hamiltonian binds each site's orbitals into, each carrying one site content.

    A site's multiplets are the finest partition of its orbitals that every
    operation fixing the site carries block by block, and that splits nothing
    the Hamiltonian's conserved matrices join: a ``(p_x, p_y)`` pair under a
    threefold axis is one multiplet, and so is a square's ``p_x, p_y`` pair
    that no hopping couples, in whatever basis it is written, and identical
    copies of a set of orbitals that nothing joins; an orbital nothing mixes is
    a multiplet of its own.  Copies of one site irrep written in a basis turned
    from site to site of an orbit are one multiplet too, each of its rows
    naming that irrep.  Each multiplet's matrices are the model's own blocks,
    in whatever basis its orbitals are written, and a declaration is read on
    the multiplets without changing them.  ``rep.multiplets.summary()`` prints
    them.

    Attributes
    ----------
    by_site : tuple
        Per site, its multiplets, each the orbitals it holds.
    dims : tuple
        Per site, each multiplet's dimension: the orbitals it holds, or its
        declared spinor pairs.
    accidental : tuple of dict
        Orbitals degenerate without the symmetry forcing it, where two
        partitions are equally fine: the ``operation`` that binds them, the
        ``orbitals``, the ``chosen`` partners and a ``note``.  Empty where the
        partition is unique.
    determined_by : str
        ``"hamiltonian"``, read off the Hamiltonian, or ``"declaration"``,
        stated by ``multiplets=`` to :func:`..groups.realized_group`.
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
    # reference site of their orbit -- which orbital answers to which across an
    # orbit, as the operations carry them; empty where the sites' own order is it
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
        """How many orbitals (or declared spinor pairs) multiplet ``chan`` holds.

        Read off its states: two to each orbital of a spinful model and to each
        declared pair, one to each orbital of a spinless model.
        """
        return len(self._chan_states[int(chan)]) // (2 if self._spinful else 1)

    @property
    def _near(self) -> bool:
        """Whether these multiplets hold only within the declared tolerance, not exactly.

        A frame off the one the operations carry by less than the declared
        precision reads the multiplets of that frame, and the Hamiltonian's own
        symmetry matrices then mix them at the size of the turn: a measurement
        that holds only within what was declared, which the symmetrizing pass
        is licensed to make exact (:func:`..groups.realized_group`).
        """
        return float(self._mixing) > _IN_BLOCKS

    def summary(self) -> str:
        """Each site's multiplets, and how they are known, with the attribute that holds each."""
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
                      else "  stated by multiplets=") + "   (.determined_by)")
        return "\n".join(lines)


def _multiplets_of(model, assignment, spin, parts, *, ties=(), source="measured",
                   order=None, tol=None, mixing=0.0):
    """The :class:`SiteMultiplets` one partition of every site's orbitals is.

    ``parts[s]`` are the multiplets of site ``s``, each the orbitals it holds.
    Everything else — the states each one holds, the multiplet of every
    orbital, whether its internal factor is the tables' lift — follows from it
    and from the model's own spin structure.  ``source`` is ``"declared"`` for
    a stated partition and anything else for one read off the Hamiltonian.
    ``order`` is each site's orbitals in slot order (:func:`.._multiplets._slot_order`),
    which is what matches a multiplet to its image across an orbit; where none
    is given it is each site's own, its declared spinor pairs in turn where a
    basis is declared (:func:`_pair_order`).  ``tol``
    is the declared tolerance the multiplets' blocks are tested at
    (:func:`.._tolerance.noise_floor`), carried so that every later test of them reads the
    same one, and ``mixing`` how far outside the multiplets' block form the
    members they were read off reach (:func:`_block_members`).  The build
    indexes the multiplets of every site in one list and calls an entry of it a
    "channel" — ``_nchan``, ``_chan_states``, ``_chan_of`` — so wherever the
    build names a channel, it means one multiplet.
    """
    nspin = int(model.nspin)
    # the states carry a spin natively, or by a declared spinor basis, whose
    # (up, dn) pairs are each one Wannier function of two orbitals
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
    """The indivisible pieces of every site: its orbitals, or its declared spinor pairs.

    A declared ``(up, dn)`` pair is one Wannier function and no multiplet
    splits it, so the refinement below partitions *these* rather than orbitals.
    """
    if spin is None:
        return [[(int(o),) for o in orbs] for orbs in site_orbs]
    pair_of = np.asarray(spin.pair_of_orbital, dtype=int)
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
    """Each site's orbitals read pair by pair: its declared spinor pairs in turn, up first.

    The pairs come in the order the site lists them, and each pair's two
    orbitals in the order the declaration gives them; without a declared basis
    a site's orbitals are its slots as listed.  This is the order a site keeps
    where no operation says which of its pairs answers to which at another
    site of its orbit (:func:`.._multiplets._slot_order`), so that the slots of a declared
    basis are its pairs wherever the orbitals happen to be listed.
    """
    return tuple(tuple(int(o) for a in atoms for o in a)
                 for atoms in _atoms_of_site(site_orbs, spin))


def _singleton_parts(site_orbs, atoms):
    """One multiplet per atom, at every site — the finest partition there is."""
    return tuple(tuple(tuple(a) for a in atoms[s]) for s in range(len(site_orbs)))


def _parts_from_merge(site_orbs, atoms, orbit_of, merged):
    """The multiplets one union-find over ``(orbit, slot)`` describes, site by site.

    ``atoms[s][j]`` is the atom of site ``s`` in slot ``j``.  Each atom keeps
    its **own** order inside the multiplet it lands in: a declared spinor pair is
    ``(up, dn)``, which is the order the tables' SU(2) lift is written in, and
    reordering it by orbital index would place the lift on the two states
    reversed.  The atoms themselves follow the site's own order, and the
    multiplets the lowest orbital each holds.
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
    """Which slots of an orbit are in one multiplet, as the refinement merges them."""

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
    """The members of a solution space supported on ``blocks``, as a basis of them.

    The question which orbitals form a multiplet lives **inside** the site
    blocks an operation's own permutation of the sites allows: a member of the
    space that carries one site's orbitals onto a site the operation does not
    send it to is no ``O ⊗ S`` for any partition of the orbitals, so it is
    evidence about the operation and not about the multiplets, and merging
    multiplets could never absorb it.  Such a model is not artificial — two
    sublattices with identical Hamiltonians and no hopping between them commute
    with the exchange of the two, which is in the commutant and not in the
    group — so the restriction is made first and the finer question asked on
    what is left.  Empty where the space holds no such member at all.

    ``tol`` is how far outside the blocks, relative to the space's own scale, a
    member may reach and still count as in them: ``1e-8`` for a model built
    from exact numbers, and the declared tolerance where a numerical model's
    is larger (:func:`.._tolerance.noise_floor`) -- a Wannier basis turned off the ideal
    frame by less than its own precision has its blocks within that.
    """
    return _block_members(basis, blocks, tol=tol)[0]


def _block_members(basis, blocks, *, tol=_IN_BLOCKS, lifts=None):
    """``(members, mixing)`` -- :func:`_in_blocks`, and how far outside the blocks its members reach.

    ``mixing`` is the largest weight a member counted in block form still
    carries outside the blocks, relative to the space's own scale: rounding
    where the Hamiltonian holds the blocks exactly, and above
    :data:`_IN_BLOCKS` only where a declared tolerance let the member in --
    an orbital frame off the one the operations carry by less than the declared
    precision mixes the multiplets at the size of that turn.  ``0.0`` where no
    member is in block form.  ``lifts`` holds, per block, the matrix the
    members are to carry a multiple of there, ``None`` for a block left whole
    (:func:`_ansatz_residual`); every block is whole where it is not given.
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
    """``(ok, unfilled, mixing)`` — is there a nonsingular member in block form, and where not.

    ``blocks`` are the ``(rows, cols)`` a candidate partition into multiplets
    allows one operation a nonzero entry in.  ``ok`` says whether ``space`` — already
    restricted to the operation's own site blocks (:func:`_in_blocks`) —
    contains a member supported on them and **nonsingular** on every one, which
    is what a representation's matrix has to be.  Where it does not, the
    ``unfilled`` block indices are the ones the space cannot fill: a block no
    member reaches at all, or one every member is singular on.  Nonsingularity
    is measured on fixed pseudo-random combinations of the block-form members,
    so that the verdict is about the *space* and not about whichever basis
    direction a decomposition returned — the difference between measuring two
    decoupled orbitals and measuring a projector onto one of them.  ``tol`` is
    the declared tolerance every threshold here reads (:func:`.._tolerance.noise_floor`),
    ``None`` for the exact defaults.  ``mixing`` is how far outside the blocks
    the members in block form reach (:func:`_block_members`).
    """
    V, mixing = _block_members(space, blocks, tol=noise_floor(_IN_BLOCKS, tol))
    ok, unfilled = _filled_check(V, blocks, tol=tol, seed=seed, draws=draws)
    return ok, unfilled, mixing


def _filled_check(V, blocks, *, tol=None, seed=17, draws=3):
    """:func:`_unfilled_blocks` on members already restricted to the blocks.

    The restriction is the expensive half and a use that needs the members
    themselves has already paid for it, so the verdict is asked of ``V``
    directly rather than of the space it came from.  A block is empty where no
    member reaches ``1e-9`` of the members' scale, and singular where its
    smallest singular value is within ``1e-8`` of it: both measure noise, so
    both read the declared tolerance ``tol`` where it is larger
    (:func:`.._tolerance.noise_floor`).
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
    """The fixed pseudo-random combinations of ``V``'s members :func:`_filled_check` reads, in turn."""
    rng = np.random.default_rng(int(seed))
    for _ in range(int(draws)):
        y = rng.normal(size=len(V)) + 1j * rng.normal(size=len(V))
        yield np.einsum("j,jab->ab", y, V)


def _site_spaces(model, group, assignment, spin, validated, *, tol=None):
    """``({ind: members}, site_layout, {ind: mixing})`` — each operation's solution space inside its site blocks.

    The site blocks are the operation's own permutation of the sites, and no
    partition of a site's orbitals changes them, so every question about which
    orbitals answer to which is asked of the members supported on them
    (:func:`_in_blocks`, at the declared tolerance ``tol`` where one is larger
    than its default, :func:`.._tolerance.noise_floor`).  An operation the detector kept no
    space for, or one whose space is past the read-off's own ceiling, is absent.
    ``mixing`` is how far outside its site blocks each operation's members
    reach (:func:`_block_members`).
    """
    site_orbs = tuple(tuple(int(o) for o in orbs) for orbs in assignment.site_orbitals)
    site_layout = _multiplets_of(model, assignment, spin,
                             tuple((tuple(orbs),) for orbs in site_orbs), tol=tol)
    bases = (getattr(validated, "_bases", None) or {}) if validated is not None else {}
    pos_of_ind = {int(s.ind): i for i, s in enumerate(group.symmetries)}
    spaces, mixing = {}, {}
    for ind in (validated.indices if validated is not None else ()):
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
# `_build_rep`'s tensor structure is `D(g) = O(g) ⊗ S(g)`, and the detector has
# already measured the *whole* space of operators that intertwine `H` with the
# operation's k-action.  When the tensor structure is realizable the
# representation is an element of that space, so its discrete choices can be
# **read** rather than enumerated.  Four functions do it, and none of them
# accepts anything: the candidate they find still goes through all three gates.

def _ansatz_blocks(chan_states, image, orient):
    """``[(rows, cols)]`` per multiplet — where ``O(g) ⊗ S(g)`` may be nonzero.

    ``orient = +1`` is the tensor structure itself: the block carrying multiplet ``c`` to
    its image ``image[c]``.  ``orient = -1`` is the ``g⁻¹`` branch, whose
    candidate is the adjoint, so its nonzero blocks are the transposed ones and
    the matrix they carry is ``S(g)†``.
    """
    out = []
    for c in range(len(chan_states)):
        src = np.asarray(chan_states[c], dtype=int)
        dst = np.asarray(chan_states[int(image[c])], dtype=int)
        out.append((src, dst) if orient < 0 else (dst, src))
    return out


def _ansatz_residual(X, blocks, lifts, classes=None, weights=None):
    """``X`` minus its tensor-structure component — the part no ``O(g) ⊗ S`` can carry.

    The tensor-structure subspace is spanned by matrices holding ``lifts[i]`` in
    allowed block ``i`` and zero elsewhere, so what is left is what no such
    matrix can reach.  A block whose lift is ``None`` is **free** — the internal
    factor of a multi-dimensional multiplet, measured rather than tabulated, and
    the question the multiplets' own measurement asks, where the whole of the
    block is allowed and only its position is at stake.  Without ``classes`` there is one
    least-squares coefficient per block.  With ``classes`` -- lists of block
    indices, one per multiplet orbit -- there is one coefficient per class, which
    is the freedom :func:`.._build._build_rep` has (a scalar per multiplet orbit per
    generator, propagated): a member whose blocks carry different scalars
    inside one class is a gauge dressing, not a representation the build can
    assemble.

    ``weights`` states the **pattern** one coefficient is shared over: block
    ``i`` is fitted to ``coef · weights[i] · lifts[i]`` rather than to
    ``coef · lifts[i]``, so the residual measures how far ``X`` is from
    carrying those *relative* scalars.  That is how a chosen member is picked
    out of a space that leaves a block free (:func:`_ansatz_member`), and each
    block then carries its own lift rather than the group's first — which it
    has to, since the blocks of one group need not be the same size.

    **Each block carries its own lift** whether or not ``weights`` is given: the
    multiplets of one class are the sites of one Wyckoff orbit, and an orbit that
    permutes the orbital frame carries a *different* matrix at each of them —
    ``p_x`` at one site and ``p_y`` at the next — with the one coefficient the
    class shares on top.  Where every block of a class carries the same lift the
    fit is the same number it always was.
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
    """The member of ``span(basis)`` closest to tensor-structure form, with its defect.

    Returns ``(X, defect, scale)``.  ``defect`` is the smallest singular value
    of the ``nstate² × dim`` matrix whose columns are the basis elements'
    residuals off the tensor structure — the norm of what the winning
    combination could not put in that form — and ``scale`` is the largest, i.e.
    how far from it the *worst* direction of the space is.  Read them as a pair:
    a `dim ≥ 2` space of a conserved operator has ``scale = O(‖B‖)`` and, when
    the tensor structure is realizable, ``defect`` at machine level.

    ``dim = 1`` is the same statement with nothing to choose: ``X`` is the one
    solution and ``defect = scale`` is how far *it* is from tensor-structure
    form.

    ``prefer`` asks for **the chosen member, where the space leaves the choice
    open**.  Several directions can reach tensor-structure form at machine level -- a
    conserved operator acting inside a multiplet, or a multiplet the hoppings
    couple to nothing at all -- and then *which* of them is returned decides
    the relative character between those multiplets.  Nothing in the Hamiltonian
    settles it, so the choice is made rather than stumbled on: among the
    directions that reach tensor-structure form, the one whose blocks carry the
    multiples of the lift ``weights`` asks for, one shared coefficient over each
    group of ``prefer``.  ``weights`` all one is **the trivial relative
    character** -- every block the same multiple of the lift -- which is what an
    undeclared free block takes;
    a declared free block asks for the character that was declared.

    The preference runs over **every** block of the operation, not within one
    site: a multiplet two sites apart that the Hamiltonian couples to nothing is
    as free as one beside its neighbour, and the same rule settles both.  What
    is decoupled is still said, on the record rather than by a refusal --
    :attr:`..groups.RealizedGroup.conserved` counts the matrices that commute
    with ``H`` beyond the identity, and the row's ``determined_by`` says the
    lowest-``l`` rule named its character.

    ``prefer`` is the blocks grouped, in the shape ``classes`` has; ``weights``
    is one complex number per block, defaulting to one.
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
    """Which sign class each multiplet belongs to — the orbits of the multiplets.

    The tensor structure's freedom is one scalar per class per generator, so a
    class is what the operations under test carry into one another.
    """
    cls = -np.ones(int(nchan), dtype=int)
    ncls = 0
    for c in range(int(nchan)):
        if cls[c] < 0:
            for d in {int(cperm[p][c]) for p in u_pos} | {c}:
                cls[d] = ncls
            ncls += 1
    return cls, ncls


def _block_space(basis, rows, cols, tol=ZERO):
    """``(space, scale)`` — the blocks a solution space carries at ``(rows, cols)``.

    ``space`` is an orthonormal basis of them, leading direction first, so
    ``space[0]`` is the block of the member of the space whose block there is
    largest — which is the block to read when the polar-extracted member's own
    vanishes.  ``scale`` is that largest block's norm.
    """
    blk = np.asarray(basis)[:, np.asarray(rows, dtype=int)][:, :,
                                                            np.asarray(cols, dtype=int)]
    d0, d1 = blk.shape[1], blk.shape[2]
    _, s, vh = np.linalg.svd(blk.reshape(blk.shape[0], -1), full_matrices=False)
    scale = float(s[0]) if s.size else 0.0
    keep = [i for i in range(len(s)) if float(s[i]) > tol * max(scale, 1.0)]
    return [complex(s[i]) * vh[i].reshape(d0, d1) for i in keep], scale


def _unitarized(M, *, tol=ZERO):
    """The unitary nearest one block, or ``None`` where there is no block at all.

    A block none of whose entries reaches ``tol`` is none: :data:`.._tolerance.ZERO`,
    or the declared tolerance where the block was read off a model symmetric
    only within it.
    """
    M = np.asarray(M, dtype=complex)
    if M.ndim != 2 or M.shape[0] != M.shape[1] or float(np.abs(M).max()) < tol:
        return None
    w, _, vh = np.linalg.svd(M)
    return w @ vh


def _projective_irreps(positions, prod, factor, *, seed=101, tries=4, tol=None):
    r"""The irreducible representations of a group with one factor system, each once.

    ``positions`` are the group's unitary operations, closed under ``prod``, and
    ``factor[(a, b)]`` the factor system: such a representation ``ρ`` has
    ``ρ(a) ρ(b) = factor(a, b) ρ(ab)``.  They are read off the group's own
    multiplication table, from its regular representation with that factor
    system, ``L(g) e_h = factor(g, h) e_{gh}``: every irreducible representation
    with the factor system occurs in it, and an operator drawn at random from
    the ones commuting with it -- the same table acting from the right,
    ``R(g) e_h = factor(h, g) e_{hg}`` -- has one copy of one of them on each
    of its eigenspaces.  A copy is kept where its character is not one already
    kept, and the set is checked complete, the squares of the dimensions
    summing to the order of the group.

    ``tol`` is the declared tolerance the factor system was measured at, which
    the clusters and the characters are read at where it is larger than their
    exact defaults (:func:`.._tolerance.noise_floor`); ``None`` keeps the
    defaults.

    Returns the representations, each ``{position: matrix}``, ordered by
    dimension and then by character, or ``None`` where no draw of ``tries``
    separates them.
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
    """Every way of writing ``k`` as a sum of the ``dims``, each once: index lists, non-decreasing."""
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
    """How many :func:`_dimension_sums` there are, counted without listing them."""
    ways = [1] + [0] * int(k)
    for dim in dims:
        for total in range(int(dim), int(k) + 1):
            ways[total] += ways[total - int(dim)]
    return int(ways[int(k)])


def _real_characters(kept, gens, words, parent, au_of, prod, ident, tol=1e-9):
    r"""Every real one-dimensional character of the represented group.

    These relate the labellings of one crystal that the hoppings cannot tell
    apart.  The hoppings fix how the orbitals transform relative to one
    another, not what they are: ``D -> chi(g) D(g)`` preserves the Hamiltonian
    covariance (which is quadratic in ``D``), the canonical norm
    (``chi(g)**n = chi(g**n) = 1``), the antiunitary anchor (``chi`` real) and
    the whole product table (``chi`` is a homomorphism), so no measurement on
    the model sees it, and it carries every orbital assigned ``s`` onto every
    orbital assigned the ``s ⊗ chi`` content.  A declaration fixes the
    labelling by site symmetry; without one, :func:`.._build._build_rep` takes the one
    the lowest-``l`` rule picks, and :attr:`AtomicLimitRow.determined_by` says
    which rows that choice decides.  The characters themselves stay inside
    the build.

    Complex characters are *not* enumerated here: the antiunitary anchor kills
    them.  A group with no antiunitary operation at all has more labellings
    than these: there every one-dimensional character is a candidate, the
    complex ones included (:func:`_characters_of_subgroup` with ``roots``),
    declared or not, and the labelling is chosen among them.

    ``chi`` is set to ``+1`` on the anchoring antiunitary operation: its own
    phase is a convention the corep labelling is blind to, so varying it would
    add a freedom nothing can ever pin.
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
    r"""Every one-dimensional character of **one subgroup**, given by ``.ind``.

    :func:`_real_characters` answers for the represented group, whose word tree
    the build already carries.  A covariant subgroup search asks the same
    question of a *subgroup*, whose characters are **not** the restrictions of
    the parent's — ``P3``'s trivial character is the restriction of both of
    ``P3m1``'s, and ``C4``'s two real characters restrict to one of ``C2``'s —
    so they are enumerated for the subgroup itself: generators by closure on
    the parent's own product table, one value per generator, propagated along
    the word tree and the whole multiplication checked on ``S × S``.

    ``roots`` false enumerates ``±1`` per generator, which is what an
    antiunitary anchor admits (:func:`_real_characters`).  ``roots`` true
    enumerates every root of unity the generator's order allows, which is the
    **unitary-only** question: a group with no antiunitary operation has a
    larger residual freedom — every one-dimensional character of it gives a
    labelling nothing measured can tell apart — so a declaration
    that needs a complex one there is returned with it rather than lowering the
    group.

    Returns a list of ``{ind: chi(g)}``, empty when the set is not a subgroup
    or has no generating set small enough to enumerate over.
    """
    prod = group.get_product_table(get_diff=False)
    pos_of = {int(s.ind): i for i, s in enumerate(group.symmetries)}
    ident = _identity_position(group)
    positions = sorted({pos_of[int(i)] for i in subset if int(i) in pos_of})
    if ident not in positions:
        return []
    if len(positions) == 1:
        return [{int(group.symmetries[ident].ind): 1.0 + 0.0j}]
    gens = _minimal_generators(positions, prod, ident)
    if not gens:
        return []
    words, parent = _bfs_words(gens, prod, ident, positions)
    if set(words) != set(positions):      # pragma: no cover - a closed set is reached
        return []
    ind_of = {p: int(group.symmetries[p].ind) for p in positions}
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
    """How many times an operation has to be applied to come back to the identity."""
    n, q = 1, int(p)
    while q != int(ident) and n <= 48:
        q = int(prod[int(p), q])
        n += 1
    return n


def _primed_partners(group, names):
    """``{unitary ind: its antiunitary partner}`` among the operations of ``names``.

    A grey group's antiunitary coset is the pure time reversal times its unitary
    half, so an operation and its primed copy stand or fall together on anything
    the unitary site group decides.  Empty where no pure time reversal is
    represented.
    """
    syms = group.symmetries
    prod = group.get_product_table(get_diff=False)
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
    """The indivisible pieces of one multiplet's state list.

    A multiplet is a set of *atoms* — one orbital, or the two states of one
    Wannier function, which is what carries the operations' SU(2) lift and is
    never split (:func:`_atoms_of_site`, :func:`_multiplets_of`) — and the states of
    each sit together, in the order the lift is written in.  Summing a
    per-atom scalar over these is how a multi-dimensional multiplet's **orbital**
    character is read: it is the block's trace with the spinor divided out, and
    for a spinless model it is the trace itself.
    """
    step = 2 if su2 else 1
    return [np.asarray(states[i:i + step], dtype=int)
            for i in range(0, len(states), step)]


# --------------------------------------------------------------------------
# the covariance a representation is held to
# --------------------------------------------------------------------------



def covariance_gate(n_operations, nstate, scale) -> float:
    r"""The rounding a representation's covariance carries, ``max(5e-14, |G|·N·ε·max|H|)``: the most an exact one misses by.

    An exact representation's covariance ``max_k ‖D H(k) D† − H(Rk)‖`` is the
    rounding of its own arithmetic: its matrices are products of up to ``|G|``
    matrices of size ``N × N``, each rounding at ``N·ε``, and ``D H D†``
    carries that onto entries of ``H`` as large as ``max|H|``.  A wrong matrix
    misses by a fraction of ``max|H|`` itself, many decades above.

    Parameters
    ----------
    n_operations : int
        ``|G|``, the operations represented, those with time reversal among
        them.
    nstate : int
        ``N``, the number of states.
    scale : float
        ``max|H|``, the largest entry of ``H(k)`` over the ``k`` the covariance
        is measured at.

    Returns
    -------
    float
        The covariance check's tolerance, in the units of ``H``: ``5e-14`` for a
        few operations on a few orbitals of order-one size, ``1.2e-12`` for the
        96 operations of ``Im-3m1'`` on 32 states with ``max|H| = 1.76``.
    """
    return max(EXACT_TOL, float(n_operations) * float(nstate)
               * float(np.finfo(float).eps) * float(scale))


def _covariance_battery(group, k_pts, n_battery, seed):
    """The k-points the covariance gate is measured at: random generic ones unless given."""
    if k_pts is None:
        rng = np.random.default_rng(int(seed))
        k_pts = rng.uniform(-0.5, 0.5, size=(int(n_battery), group.dim_k))
    return np.atleast_2d(np.asarray(k_pts, dtype=float))


def _default_of(func, name):
    """A default read off a function's own signature, never retyped here."""
    return inspect.signature(func).parameters[name].default


#: The tolerance :func:`..symmetry.bloch_operator` checks the integer-shift
#: identity ``A·τ_a + t = τ_{g(a)} + s`` to, read off its own signature: the
#: statement that the operations carry the centres onto one another, which the
#: projection's operators are built on.
_BLOCH_TOL = float(_default_of(bloch_operator, "tol"))
