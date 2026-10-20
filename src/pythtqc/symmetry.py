r"""The model's representation of its space group, and the atomic-limit table of its orbitals.

:func:`..groups.realized_group` measures, for each operation ``g = {R|t}`` of
the cell, every matrix ``X`` with ``X H(k) = H(Rk) X`` at once on a grid of
``k``, ``X H(k)* = H(−Rk) X`` for one combined with time reversal, assuming
nothing about what ``g`` does to the orbitals: no solution is a proof that
``g`` is broken, and at the identity the solutions are the operators the
model conserves.  From the operations that hold it builds
:class:`SpaceGroupRep`, one matrix per operation on the model's states in the
tables' convention, spin along ``+z`` in a right-handed frame, through which
every band label is read; and :class:`AtomicLimitTable`, what each orbital
carries at its site.

The hoppings fix how the orbitals transform relative to one another, not what
they are: one one-dimensional character of the group is left open on each part
of the crystal the hoppings couple, real where the group has time reversal and
any, complex ones included, where it has none: the spin–orbit ``I4_1`` model in
its unitary group, its ``4a`` orbitals declared ``B``, is returned with
``χ(4⁺) = −i``.  ``characters=`` declares it, and what the
declarations leave open is assigned by the lowest-angular-momentum rule; each
row of the table says which.
"""

from __future__ import annotations

import inspect
import itertools
import numbers
from dataclasses import dataclass, field, replace
from typing import Any, Callable

import numpy as np

from . import _bloch, _sitechars
from ._sitechars import _L_MAX, _content_shells
from ._tolerance import (ANSATZ_GATE, ASSEMBLED, CHARACTER, CLUSTER, DEFAULT_ENERGY,
                         EXACT_TOL, FLAT_ZERO, IN_SPACE, SINGULAR, SNAP, SPREAD, ZERO,
                         _COMPOSITION_GATE, _EIGEN_WINDOW_CEIL,
                         _IN_BLOCKS, _LAYOUT_MERGE, _VACUOUS_CUT, eigen_window,
                         noise_floor)
from .refusal import InternalError, Refusal

__all__ = ["RepAnchor",
           "NormalizedRep", "AtomicLimitRow", "AtomicLimitTable", "BandRepresentation",
           "MultiplicitySpace", "SiteMultiplets", "SpaceGroupRep",
           "normalize_rep",
           "bloch_operator", "ANSATZ_GATE", "EXACT_TOL", "covariance_gate"]


#: How many seeded pseudo-random complex combinations of a degenerate solution
#: basis are searched for a well-conditioned (hence polar-extractable) element.
#: Sixteen is far more than the generic case needs — a random combination of a
#: ``*``-algebra basis is nonsingular with probability one — and is kept fixed
#: so the search is reproducible without a parameter to thread.
_UNITARY_PROBES = 16

#: Seed of that search.  Fixed, so two calls on the same model return the same
#: matrices bit for bit.
_UNITARY_PROBE_SEED = 1729


#: How many k-points at the head of the grid the eigen route looks at when it
#: picks its **parametrization point** ``k1`` (the one whose source spectrum has
#: the largest spread, so that the fewest eigenvalue pairs match by accident).
#: Four is enough because the choice only has to avoid a *degenerate* point:
#: nothing about the verdict depends on which point is picked, only the size of
#: the stage-0 basis and hence the cost.
_EIGEN_PROBES = 4


#: Consecutive sift stages that must leave the basis size unchanged before the
#: sift stops.  Three is a stopping rule, not a soundness one: an over-counted
#: dimension is refused by its full-grid residual, exactly as a subset solve's
#: is (:func:`_find_intertwiners`, "subset solve, full verify").
_SIFT_STABLE = 3

#: Factor by which an emptied sift widens its window before trying again.
_EIGEN_ESCALATION = 10.0

#: Seeded generic combinations of the surviving basis that the sift streams over
#: the **whole** grid before it stops (see :func:`_grid_probe`).  Two, because
#: one already fails at a point where the basis does not intertwine with
#: probability one and the second costs one pass to make that "with probability
#: one" not matter.
_GRID_PROBES = 2

#: Byte ceiling on the tensor-structure projection :func:`_build_rep`'s read-off runs on
#: one generator's solution space (``16·dim·nstate²``).  It bites only on a
#: **vacuous** space, whose ``dim`` is ``nstate²`` — a scalar Hamiltonian, where
#: the enumeration is the honest instrument anyway — and it makes the read-off
#: defer rather than allocate gigabytes.
_READOFF_MAX_BYTES = 2.5e8


@dataclass(frozen=True)
class _IntertwinerSpace:
    """The matrices an operation could be on this Hamiltonian.

    Its dimension says whether the operation holds at all: ``dim == 0`` is a
    *proof* that no such matrix exists, so the operation is broken rather than
    unfitted, while ``dim >= 2`` says the model conserves something beyond the
    group and at ``g = identity`` the basis is the commutant of ``{H(k)}``
    itself.  The space is falsy at dimension zero, the proof that the
    operation is broken, as a :class:`..refusal.Refusal` is, so
    ``isinstance(out, Refusal)`` is the test for a refusal.
    :attr:`residuals` carries the covariance each candidate achieves,
    in the units of ``H``: the number a symmetry gate reads.  The detector is
    walked through in
    ``docs/background/measurement.md#the-cell-and-a-verdict-on-each-operation``.

    Attributes
    ----------
    dim : int
        Dimension of the solution space.  ``0`` is the detector's *proof* that
        the Hamiltonian does not obey the operation; ``dim == nstate**2`` with
        :attr:`vacuous` set is the opposite verdict, the constraint being the
        zero operator so that *every* ``X`` solves it.
    basis : ndarray, shape (dim, nstate, nstate)
        Orthonormal (in the Frobenius inner product) basis of the solution
        space, in no particular phase.  At ``g = identity`` this basis *is* the
        commutant of ``{H(k)}`` — the model's conserved operators, enumerated.
        When :attr:`vacuous` the basis spans all of ``M_nstate`` and is anchored
        so that ``basis[0]`` is the identity matrix.
    unitaries : ndarray, shape (n_candidates, nstate, nstate)
        Unitary candidates, one per row of ``residuals``/``unitarity``.  The
        first ``dim`` rows are the polar-extracted unitary nearest each basis
        solution (``X = W S V†`` → ``W V†``).  When ``dim >= 2`` one further row
        is appended: the **searched** unitary, an element of the solution space
        itself rather than merely near it (see ``search``).  ``n_candidates`` is
        therefore ``dim`` for ``dim <= 1`` and ``dim + 1`` for ``dim >= 2``.
    residuals : ndarray, shape (n_candidates,)
        ``max_k ‖ U H(k) U† − H(Rk) ‖_max`` for each candidate, **over every
        supplied k-point** (not only the ones the nullspace was solved on) —
        the covariance residual achieved, in the units of ``H``.  This is the
        number to gate on.
    unitarity : ndarray, shape (n_candidates,)
        ``‖X†X − 1‖_max`` of each candidate before polar extraction, with ``X``
        scaled to ``‖X‖_F = sqrt(nstate)``: how far the raw solution already
        was from unitary.  A basis of a degenerate space routinely lands on
        singular, sector-projector-like elements and reads ``O(1)`` here; the
        appended searched row reads machine-zero by construction.
    singular_values : ndarray
        Singular values of the constraint operator, ascending, as solved (on
        ``n_kpts_solve`` k-points).  The gap between ``singular_values[dim-1]``
        and ``singular_values[dim]`` is what the nullspace verdict rests on.
    scale : float
        The scale the cut is measured against: ``dim`` is exactly
        ``count(singular_values <= cut * scale)``, and ``singular_values[0] /
        scale`` is the "how far from a solution" number to read.  When
        :attr:`vacuous` there is no such scale, every singular value being
        rounding noise, and this reads ``max |H|`` over the grid instead.
    tol : float
        The declared symmetry tolerance, absolute and in the units of ``H``:
        the number a caller's covariance gate is set by.  Nothing here is gated
        on it; :attr:`residuals` is what a gate reads.
    cut : float
        The relative, dimensionless nullspace cut applied:
        ``dim == count(singular_values <= cut * scale)``.  Equal to
        :attr:`tol` unless the caller declared the two apart.
    method : str
        Which route solved it — ``"eigen"`` (the default), ``"stacked"``, or
        ``"stacked_subset"``, the same full-precision stacked SVD over a
        deterministic subset of the grid.  None of the three trades precision
        for size.
    n_kpts : int
        Number of k-points supplied — where the residuals were measured.
    n_kpts_solve : int
        Number of k-points the nullspace was solved on.
    window : float
        The eigen route's matching window ``δ``, relative to ``max|H − μ·1|``
        at the parametrization point, ``μ`` the mean on-site energy: the
        stage-0 basis is the outer products of eigenvector pairs with
        ``|E'_a − E_b| <= δ·max|H − μ·1|``.  Never narrower, in the units of
        ``H``, than the covariance a kept candidate may carry, so it reads above
        one where ``H`` sits at its mean at that point.  ``0.0`` for the
        stacked routes, which have no window.
    k_solve : int
        Index (into the supplied ``k_pts``) of the eigen route's
        parametrization point ``k1`` — the probe point of largest spectral
        spread.  ``-1`` for the stacked routes.
    basis_size : tuple of int
        The eigen route's basis size after each sifted point, starting with the
        stage-0 parametrization.  ``()`` for the stacked routes.
    escalations : tuple of float
        Windows the eigen route tried and abandoned, in order, before settling
        on :attr:`window` — what makes a raised tolerance effective on a model
        that is symmetric only to its own numerical defect.
    antiunitary : bool
        Whether the antiunitary equation was solved.
    vacuous : bool
        The constraint operator vanished: ``H(k)`` and ``H(Rk)`` are the *same
        scalar* matrix at every sampled k, so the equation says nothing and
        every ``X`` solves it.  ``dim`` is then ``nstate**2`` and the full
        matrix space is returned rather than a nullspace read out of rounding
        noise — which is what a one-band model does at every operation its
        ``h(k)`` obeys.  It is not an acceptance: what certifies the operation
        is ``residuals``, measured from ``H`` exactly as in every other case.
    search : dict
        Empty unless ``dim >= 2``.  Then: ``n_probes``/``seed`` of the
        combination search, ``picked`` (index into the searched pool, ``< dim``
        if a bare basis element won), ``sigma_min`` of the winner and
        ``sigma_min_basis``, the best any bare basis element managed — the
        pair that shows what the search bought.
    """

    dim: int
    basis: np.ndarray
    unitaries: np.ndarray
    residuals: np.ndarray
    unitarity: np.ndarray
    singular_values: np.ndarray
    scale: float
    tol: float
    cut: float
    method: str
    n_kpts: int
    n_kpts_solve: int = 0
    antiunitary: bool = False
    vacuous: bool = False
    search: dict = field(default_factory=dict)
    window: float = 0.0
    k_solve: int = -1
    basis_size: tuple = ()
    escalations: tuple = ()

    def __bool__(self) -> bool:
        return self.dim > 0

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        res = "-" if self.dim == 0 else f"{float(np.min(self.residuals)):.2e}"
        nk = (f"nk={self.n_kpts}" if self.n_kpts_solve == self.n_kpts
              else f"nk={self.n_kpts_solve}/{self.n_kpts}")
        vac = ", vacuous" if self.vacuous else ""
        win = f", window={self.window:.0e}" if self.window else ""
        return (f"_IntertwinerSpace(dim={self.dim}, best residual={res}, "
                f"sv[0]/scale={self.singular_values[0] / self.scale:.2e}, "
                f"method={self.method!r}, {nk}{win}{vac})")


def _identity_anchored_matrix_basis(n: int) -> np.ndarray:
    r"""Orthogonal basis of *all* ``n x n`` matrices, the identity first.

    The basis a vacuous constraint returns.  Every element carries
    :math:`\lVert\cdot\rVert_F = \sqrt n` — the normalization
    :func:`_find_intertwiners` gives any solution basis — so ``out[0]`` is the
    identity matrix itself, and the candidate loop downstream extracts
    :math:`\mathbb 1` from it before anything else.  The remaining diagonal
    elements are the unitary-DFT combinations of the diagonal matrix units,
    orthogonal to :math:`\mathbb 1` and to one another because the DFT columns
    are; the off-diagonal ones are the matrix units, orthogonal to everything.
    Exact, deterministic, and ``n**2`` of them.
    """
    out = np.zeros((n * n, n, n), dtype=complex)
    j = np.arange(n)
    dft = np.exp(2j * np.pi * np.outer(j, j) / n)     # column m: the m-th DFT column
    for m in range(n):
        out[m][j, j] = dft[:, m]                      # ||.||_F = sqrt(n) already
    a = n
    for i in range(n):
        for k in range(n):
            if i != k:
                out[a][i, k] = np.sqrt(n)
                a += 1
    return out


def _as_k_action(k_action, dim_k: int) -> Callable[[np.ndarray], np.ndarray]:
    """Normalize ``k_action`` to a callable on ``(Nk, dim_k)`` arrays."""
    if callable(k_action):
        return k_action
    M = np.asarray(k_action, dtype=float)
    if M.shape != (dim_k, dim_k):
        raise ValueError(f"k_action matrix must be ({dim_k}, {dim_k}), got {M.shape}")
    return lambda k: np.asarray(k, dtype=float) @ M.T


class _GridHamiltonians:
    r"""``H(k)`` on a fixed k-grid, evaluated once and shared by every operation.

    ``H(k)`` on the detection mesh does not depend on which operation is being
    tested, so the detector builds one of these and hands it to every
    operation's test.  The rotated grid ``H(Rk)`` of a given operation is a
    *different* k-set and gets its own instance from :meth:`rotated`; operations
    that share a k-action share that one too, because they share the whole
    solve.  Every point is summed from the model's lattice-vector blocks, built
    once for the model and handed on to the rotated grids, and the grid is
    evaluated in k-chunks whose phases and matrices stay inside ``max_bytes``.

    Parameters
    ----------
    model : pythtb.TBModel
    k_pts : array, shape (Nk, dim_k)
        The grid.  Held by reference; do not mutate it.
    params : dict, optional
        Model parameter values, as ``model.hamiltonian`` takes them.
    max_bytes : float, optional
        Budget for one evaluation, in bytes.  Default ``256e6``.  A grid whose
        Hamiltonians fit inside it is evaluated **in one call** (bit-identical
        to ``model.hamiltonian`` on the same points) and cached whole; a larger
        one is streamed.
    bloch : optional
        The model's Bloch sum at these parameter values, where another grid of
        the same model already holds it (:meth:`rotated` hands its own on).
        Looked up at the first evaluation otherwise.

    Attributes
    ----------
    n_evaluated : int
        How many k-points have been evaluated — the number the sharing is
        meant to reduce, so it is reported rather than assumed.
    """

    def __init__(self, model, k_pts, params=None, max_bytes=256e6, *, bloch=None):
        self.model = model
        self.k_pts = np.atleast_2d(np.asarray(k_pts, dtype=float))
        self.params = dict(params or {})
        self.max_bytes = float(max_bytes)
        self.n_kpts = int(self.k_pts.shape[0])
        self.nstate = int(model.nstate)
        self.n_evaluated = 0
        self._full = None
        self._points: dict[int, np.ndarray] = {}
        # the model's Bloch sum, looked up at the first evaluation and handed on
        # by `rotated`, so no grid of one model hashes its content twice
        self._bloch_sum = bloch

    def _sum(self):
        """The model's Bloch sum at these parameter values (:func:`.._bloch.evaluator`)."""
        if self._bloch_sum is None:
            self._bloch_sum = _bloch.evaluator(self.model, self.params)
        return self._bloch_sum

    @property
    def chunk(self) -> int:
        """k-points per evaluation: what one point holds, against ``max_bytes``."""
        return max(1, int(self.max_bytes // self._sum().point_bytes))

    @property
    def fits(self) -> bool:
        """Whether the whole grid may be evaluated in one call and kept."""
        return (self.n_kpts <= self.chunk
                and 16.0 * self.n_kpts * self.nstate ** 2 <= self.max_bytes)

    def _evaluate(self, k) -> np.ndarray:
        self.n_evaluated += int(np.shape(k)[0])
        return np.asarray(self._sum()(k))

    def full(self) -> np.ndarray:
        """The whole grid as one ``(Nk, nstate, nstate)`` array, cached.

        Materializes ``16·Nk·nstate²`` bytes whatever the budget says — the
        stacked solver needs the whole stack by construction — so where the
        grid can be streamed :meth:`chunks` is the cheaper route.
        """
        if self._full is None:
            if self.fits:
                self._full = self._evaluate(self.k_pts)
            else:
                out = np.empty((self.n_kpts, self.nstate, self.nstate), dtype=complex)
                for lo, hi, blk in self.chunks():
                    out[lo:hi] = blk
                self._full = out
            self._points.clear()
        return self._full

    def at(self, i: int) -> np.ndarray:
        """``H`` at the ``i``-th grid point, memoized."""
        if self._full is None and self.fits:
            self.full()                  # one call beats Nk single-point calls
        if self._full is not None:
            return self._full[i]
        i = int(i)
        if i not in self._points:
            H = self._evaluate(self.k_pts[i][None, :])[0]
            if 16.0 * (len(self._points) + 1) * self.nstate ** 2 <= self.max_bytes:
                self._points[i] = H
            return H
        return self._points[i]

    def chunks(self):
        """Successive ``(lo, hi, H[lo:hi])`` blocks of the grid, within the budget.

        Caches the whole grid on the first pass when it :attr:`fits`, so a
        shared instance evaluates a grid once however many operations stream
        over it; streams without caching when it does not.
        """
        if self._full is None and self.fits:
            self.full()
        if self._full is not None:
            yield 0, self.n_kpts, self._full
            return
        c = self.chunk
        for lo in range(0, self.n_kpts, c):
            hi = min(lo + c, self.n_kpts)
            yield lo, hi, self._evaluate(self.k_pts[lo:hi])

    def rotated(self, k_rot) -> "_GridHamiltonians":
        """The same model, Bloch sum and budget on the rotated k-set ``Rk``."""
        return _GridHamiltonians(self.model, k_rot, self.params, self.max_bytes,
                                 bloch=self._sum())


# --------------------------------------------------------------------------
# the eigen-sift solver: parametrize at one k, sift the rest
# --------------------------------------------------------------------------
#
# ``X H_k = H'_k X`` at a *single* k has an explicit nullspace.  With
# ``H_k = V E V†`` and ``H'_k = V' E' V'†``, the equation on ``Y = V'† X V``
# reads ``Y_ab (E_b − E'_a) = 0``: ``Y`` is supported on matching-energy pairs,
# so the solution space is spanned by the outer products
# ``B_ab = V'[:, a] V[:, b]†`` over those pairs — ``n`` of them for a
# non-degenerate matching spectrum, ``Σ d_i d'_i`` over degeneracy blocks in
# general, and **zero** when the spectra do not match at all, which is a proof
# of breaking at ``O(n³)`` cost.  Every further k-point is then a *sift*: an
# ``n² × m`` constraint whose nullspace narrows the basis.


class _ConstraintPair:
    """``(H_src(k), H(Rk))`` at one point or in chunks, source already conjugated.

    The one place the antiunitary equation differs from the unitary one: the
    source Hamiltonian is ``H(k)*``.  Keeping the conjugation here means the
    *cache* is shared between the two (``H(k)`` does not know which equation is
    being solved), which is the whole point of :class:`_GridHamiltonians`.

    The two grids chunk **identically** — :attr:`_GridHamiltonians.chunk` and
    :attr:`_GridHamiltonians.fits` depend only on the model, the budget and the
    number of points, all of which the rotated grid inherits — so zipping their
    chunk iterators pairs each k with its own image and not another's.
    """

    def __init__(self, src_grid, rot_grid, conj):
        self.src, self.rot, self.conj = src_grid, rot_grid, bool(conj)
        self.n_kpts = int(src_grid.n_kpts)

    def at(self, i):
        H = self.src.at(i)
        return (H.conj() if self.conj else H), self.rot.at(i)

    def chunks(self):
        for (lo, hi, Hs), (_, _, Hr) in zip(self.src.chunks(), self.rot.chunks()):
            yield lo, hi, (Hs.conj() if self.conj else Hs), Hr


def _stage_nullspace(B, Hs, Hr, block_bytes=6.4e7):
    r"""Right singular vectors of one stage's constraint, streamed and blocked.

    Column ``j`` of the stage operator is ``vec(B_j H_k − H'_k B_j)``, an
    ``n² × m`` matrix, and what is wanted from it is the combinations ``y``
    with ``‖C y‖`` small — the **right** singular vectors, and only those.

    Three things happen here at once, and each is load-bearing.  The operator
    is written into one pre-allocated array in blocks of the basis index, which
    holds the *temporaries* to one block instead of two further copies of the
    whole thing.  It is then reduced to an ``m × m`` triangular factor by
    **TSQR** — a Householder QR of each row block, the factors stacked and
    reduced once more — which is backward stable, so the singular values of the
    final ``R`` are those of the operator to ``eps‖C‖``, and which never copies
    more than one block (measured at ``n = 192, m = 2n``: agreement to 5.7e-16
    relative against a single QR of the whole matrix, at 4.1 s against 7.7 s).
    And only ``R`` is decomposed, so the ``n² × m`` left factor that an SVD of
    the tall matrix would compute and discard is never formed at all (23.7 s,
    same size).

    This is emphatically *not* the normal-equation shortcut: nothing is
    squared, and a true nullvector is not floored at ``sqrt(eps)``.

    Returns
    -------
    s : ndarray, shape (m,)
        Singular values, **descending** (LAPACK order).
    vecs : ndarray, shape (m, m)
        Right singular vectors as columns, in the same order.
    """
    m, n = int(B.shape[0]), int(B.shape[1])
    C = np.empty_like(B)
    step = max(1, int(block_bytes // (16 * n * n)))
    for lo in range(0, m, step):
        hi = min(lo + step, m)
        np.matmul(B[lo:hi], Hs, out=C[lo:hi])
        C[lo:hi] -= Hr @ B[lo:hi]
    flat = C.reshape(m, -1).T                     # (n**2, m), one column per basis element
    rows = max(1, int(block_bytes // (16 * m)))
    parts = [np.linalg.qr(flat[lo:lo + rows], mode="r")
             for lo in range(0, n * n, rows)]
    del C, flat
    R = parts[0] if len(parts) == 1 else np.linalg.qr(np.concatenate(parts), mode="r")
    _, s, vh = np.linalg.svd(R)
    return s, vh.conj().T


def _visit_order(nk):
    """The order the sift walks the grid in: bit-reversed, not lexicographic.

    Consecutive points of a lexicographic mesh are **correlated** — the first
    ``nk**(1/d)`` of them share every coordinate but one, so they lie on a line
    of the Brillouin zone, and a line is exactly where a model has *more*
    symmetry than it has generically.  Measured on inversion-symmetric
    Kane–Mele (``rashba = 0``, ``s_z`` conserved) with a lexicographic 6×6
    mesh: the first seven points all sit at ``kx = 0``, the solution space is
    four-dimensional along that line and two-dimensional on the grid, and a
    sift that stops after three unchanged stages stops at four — a *wrong*
    dimension, whose searched unitary then reads residual 2.86 against the
    stacked route's 2.0e-15.

    The van der Corput (bit-reversal) permutation visits ``0, nk/2, nk/4,
    3nk/4, …`` instead, so consecutive stages are as far apart as the grid
    allows.  It is deterministic, depends on nothing but ``nk``, and costs
    ``O(nk log nk)`` integer operations once.  The whole-grid confirmation of
    :func:`_grid_probe` is what makes the stopping rule *sound*; this only
    makes it cheap.
    """
    bits = max(1, int(max(nk - 1, 1)).bit_length())
    out = []
    for i in range(1 << bits):
        r, j = 0, i
        for _ in range(bits):
            r = (r << 1) | (j & 1)
            j >>= 1
        if r < nk:
            out.append(r)
    return out


def _grid_probe(pair, B, n, threshold, block_bytes=3.2e7):
    r"""Where on the **whole** grid does this basis fail to intertwine?

    The sift stops early — that is the point of it — and an early stop can
    leave a basis that solves the sampled points and not the others.  Before
    stopping, a seeded generic combination ``X = Σ y_j B_j`` of the surviving
    basis is streamed over every supplied k-point: ``X`` is a generic element
    of the span, so it solves at ``k`` exactly when the whole basis does (a
    non-solving direction has nonzero weight in ``y`` with probability one, and
    :data:`_GRID_PROBES` independent draws make that a formality).  The point
    of largest ``max|X H_k − H'_k X|`` is handed back so the sift can resume
    *there* — the most informative point in the grid — instead of guessing.

    ``X`` carries ``‖X‖_F = sqrt(n)``, the normalization the whole package
    gives a solution, so the returned number is on the scale of the covariance
    residual and is compared against the same absolute ``threshold``.

    Returns
    -------
    index : int
        The worst point, or ``-1`` when the basis intertwines everywhere to
        within ``threshold``.
    worst : float
    """
    m = int(B.shape[0])
    rng = np.random.default_rng(_UNITARY_PROBE_SEED)
    y = (rng.normal(size=(_GRID_PROBES, m)) + 1j * rng.normal(size=(_GRID_PROBES, m)))
    y /= np.linalg.norm(y, axis=1)[:, None]
    X = np.einsum("pj,jab->pab", y, B)
    X *= (np.sqrt(n) / np.linalg.norm(X, axis=(1, 2)))[:, None, None]
    step = max(1, int(block_bytes // (16 * n * n)))
    worst, worst_i = 0.0, -1
    for lo, hi, Hs, Hr in pair.chunks():
        for a in range(0, hi - lo, step):
            b = min(a + step, hi - lo)
            r = np.zeros(b - a)
            for p in range(X.shape[0]):
                d = X[p] @ Hs[a:b] - Hr[a:b] @ X[p]
                np.maximum(r, np.abs(d).max(axis=(1, 2)), out=r)
            j = int(np.argmax(r))
            if float(r[j]) > worst:
                worst, worst_i = float(r[j]), lo + a + j
    return (worst_i if worst > threshold else -1), worst


def _size(n_bytes) -> str:
    """A number of bytes as a reader reads it: ``1.449 GB``, ``100.4 kB``."""
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if n_bytes >= scale:
            return f"{n_bytes / scale:.4g} {unit}"
    return f"{n_bytes:.0f} bytes"


def _budget_that_runs(n_bytes) -> str:
    """The smallest two-figure budget holding ``n_bytes``, written as a literal: ``1.5e9``."""
    step = 10.0 ** (int(np.floor(np.log10(float(n_bytes)))) - 1)
    text = f"{np.ceil(float(n_bytes) / step) * step:.2g}"
    if "e" in text:
        mantissa, power = text.split("e")
        text = f"{mantissa}e{int(power)}"
    return text


def _detector_too_large(n_pairs, n, budget) -> Refusal:
    r"""The solve whose exactly degenerate pairs the detector's memory does not hold.

    At one k-point the eigen route's trial basis is one ``n × n`` matrix per
    pair of eigenvectors at matching energies, and a spectrum with exactly
    degenerate levels has more of them within the window's floor than the
    budget holds.  Narrowing the window cannot drop those pairs without
    returning a solution space that is wrong rather than coarse, so the solve
    is refused, with the least budget that forms the basis -- those pairs and
    one more -- and the calls that run it.
    """
    need = 16.0 * float(n) ** 2 * (int(n_pairs) + 1)
    runs = _budget_that_runs(need)
    return Refusal(
        "detector_too_large",
        f"measuring the symmetry matrices at one k-point needs {_size(need)} against the "
        f"{_size(budget)} it may take: {int(n_pairs)} pairs of eigenvectors at matching "
        f"energies, one {int(n)} x {int(n)} matrix each, which is what a spectrum with "
        "exactly degenerate levels costs. Narrowing the energy window to fit would drop "
        "exactly degenerate pairs and return a solution space that is wrong rather than "
        "coarse, so the window is not narrowed. This is a limit on memory, not a "
        f"statement about the model: `realized_group(..., max_bytes={runs})` runs it, "
        "and a model whose degeneracies are lifted does not reach it.",
        value=need, threshold=float(budget),
        context={"n_bytes": need, "max_bytes": float(budget), "n_pairs": int(n_pairs),
                 "n_states": int(n)})


def _pair_basis(E, V, Ep, Vp, delta, h, max_pairs, floor, gate=0.0):
    r"""Outer products over matching-energy pairs — the stage-0 solution space.

    ``B_j = V'[:, a] V[:, b]†`` for every pair with ``|E'_a − E_b| <= max(δ·h,
    gate)``: the window is relative to the Hamiltonian's magnitude at this
    point, and never narrower than ``gate``, the covariance a candidate the
    nullspace keeps may carry, in the units of ``H``.  Levels closer than that
    are one level at the precision the solve is read at, whatever point the
    solve is parametrized at -- where every band is one cluster split below the
    gate, the magnitude at a point where ``H`` sits at its mean is the
    splitting itself, and a relative window there would read the splitting as
    a gap.  These are Frobenius-**orthonormal** by construction (``⟨B_ab, B_cd⟩ =
    δ_ac δ_bd``, both eigenbases being unitary), which is the normalization the
    sifted basis then inherits for free.

    ``max_pairs`` is the memory valve.  Two populations reach it and they are
    not the same problem.  A spectrum *nearly degenerate* at this point, its
    levels crowded inside the window, matches pairs it has no business
    matching, and the honest remedy is a
    **narrower window**: the dropped pairs are the distant ones, the accuracy
    claim ``eps/δ`` degrades smoothly, and the effective ``δ`` is returned, so
    what was used is reported.  A Hamiltonian with genuinely **degenerate**
    blocks (a supercell, whose folded bands are exactly degenerate across the
    star of each folded k-point) has ``Σ d_i d'_i`` pairs at *zero* gap, and
    narrowing cannot drop any of them: the window collapses to machine
    precision, the surviving basis cannot represent an exact solution at all,
    and the returned space is then wrong rather than merely coarse (measured on
    a 4×4×3 Fu–Kane–Mele supercell, ``n = 192``: ``δ = 1.3e-15``, ``dim = 3``,
    residual 0.93).  So below ``floor``, ``eps/cut`` of the Hamiltonian's own
    magnitude read on the window's scale ``h`` — the window at which the
    dropped cross terms reach the nullspace cut — or below ``gate``, where
    the pairs dropped are levels the precision does not tell apart, no basis
    is formed: the first entry comes back ``None`` and the second is how many
    pairs the honest solve needs, the pairs within ``floor`` and ``gate``, for
    :func:`_find_intertwiners` to raise the budget to its ceiling or refuse.
    A resource limit is not a measurement about the physics and must not be
    returned as one.  The window returned is the one used, relative to ``h``.
    """
    gaps = np.abs(Ep[:, None] - E[None, :])
    delta = max(float(delta), float(gate) / h)
    sel = np.argwhere(gaps <= delta * h)
    if len(sel) > max_pairs:
        order = np.argsort(gaps[sel[:, 0], sel[:, 1]], kind="stable")[:max_pairs]
        sel = sel[np.sort(order)]
        delta = float(gaps[sel[:, 0], sel[:, 1]].max() / h)
        if delta < floor or int((gaps <= float(gate)).sum()) > max_pairs:
            return None, int((gaps <= max(floor * h, float(gate))).sum()), None
    a, b = sel[:, 0], sel[:, 1]
    B = Vp.T[a][:, :, None] * V.conj().T[b][:, None, :]
    return B, float(delta), np.sort(gaps.ravel()) / h


def _sift(pair, nk, n, *, cut, floor, delta, stable, max_pairs, probes=_EIGEN_PROBES,
          k1=None):
    """One eigen-sift pass at window ``delta``, over a :class:`_ConstraintPair`.

    The only tolerance the sift needs is the **cut** — relative, applied to each
    stage's singular values against that stage's operator norm, and never
    stricter than ``sqrt(n)·floor`` in the units of ``H``, which a unitary
    candidate whose covariance is within ``floor`` at that point stays under: at
    a point where the operator is itself that small, one band's whole operator
    being the violation, a relative cut alone would read a violation inside it
    as total.  ``floor`` is the cut itself, read in the units of ``H``, unless
    a declared precision sets it at the gate.  The covariance gate is not
    otherwise read here: what the sift hands back is a solution space, and
    whether its members are symmetries is settled afterwards, on residuals
    measured from ``H``.  The energy window is never narrower than ``floor``
    either: levels closer than the covariance a kept candidate may carry are
    paired at any point the sift is parametrized at (:func:`_pair_basis`).

    Hands back one record of the sift; see :func:`_find_intertwiners` for what
    each entry becomes.  ``flat`` is set when the parametrization point is
    degenerate to within the window at *every* grid point — a scalar or flat
    Hamiltonian, where :func:`_find_intertwiners` falls back to the absolute
    vacuity test.  Passing ``k1`` names the parametrization point outright and
    suppresses that exit: it is how the detector comes back after the vacuity
    test said no, and how a widened window re-sifts the same point.
    """
    spectra: dict[int, tuple] = {}
    # The energy window's scale is the Hamiltonian's magnitude about its mean
    # on-site energy, `w`, which a constant shift of H does not move.  The tests
    # that bound rounding -- the window's floor, the vacuity cut, the probe of the
    # whole grid -- read its magnitude with the energy zero, `h`, since rounding
    # is relative to the numbers the arithmetic holds.
    mu = _bloch.onsite_mean(pair.src.model, pair.src.params)

    def spec(i):
        if i not in spectra:
            A, Bm = pair.at(i)
            h = max(float(np.abs(A).max()), float(np.abs(Bm).max()), 1e-300)
            w = max(_bloch.magnitude(A, mu), _bloch.magnitude(Bm, mu), 1e-300)
            E, V = np.linalg.eigh(A)
            Ep, Vp = np.linalg.eigh(Bm)
            spectra[i] = (E, V, Ep, Vp, h, w)
        return spectra[i]

    if k1 is not None:
        i1 = int(k1)
    else:
        probe = list(range(min(nk, probes)))
        i1 = max(probe, key=lambda i: float(np.ptp(spec(i)[0])))
    E, V, Ep, Vp, h, w = spec(i1)
    if k1 is None and bool(np.ptp(E) <= max(delta * w, floor)
                           and np.ptp(Ep) <= max(delta * w, floor)):
        # every probe point is degenerate to within the window: look over the
        # whole grid for a point that is not, and report the per-point norm of
        # the constraint operator on the way -- it is exactly
        # max_ab |E_b - E'_a|, so the absolute vacuity test costs no more than
        # the eigenvalues already needed.
        spread, s_max, h_grid = np.empty(nk), np.empty(nk), 0.0
        for lo, hi, Hs, Hr in pair.chunks():
            Ei, Epi = np.linalg.eigvalsh(Hs), np.linalg.eigvalsh(Hr)
            spread[lo:hi] = Ei[:, -1] - Ei[:, 0]
            s_max[lo:hi] = np.maximum(np.abs(Epi[:, -1] - Ei[:, 0]),
                                      np.abs(Ei[:, -1] - Epi[:, 0]))
            h_grid = max(h_grid, float(np.abs(Hs).max()), float(np.abs(Hr).max()))
        i1 = int(np.argmax(spread))
        E, V, Ep, Vp, h, w = spec(i1)
        flat = bool(np.ptp(E) <= max(delta * w, floor) and np.ptp(Ep) <= max(delta * w, floor))
        if flat:
            return {"flat": True, "k_solve": i1, "n_solve": nk, "h": h_grid,
                    "grid_s_max": np.sort(s_max), "grid_h": h_grid or 1.0,
                    "basis": None, "trace": (), "window": float(delta)}
        # The grid holds a point that is NOT degenerate within the window: the
        # probes were merely unlucky (a stacked layer's first grid points all
        # sit on the k_z line above Gamma, where a Kramers pair is degenerate at
        # every k_z).  Parametrize at the point the scan found -- exactly what a
        # forced re-sift (`k1=i1`) would do, without the round trip.  A
        # record without a basis is only ever a *flat* record: whatever reads
        # one would otherwise have no basis to read, as a Rashba layer's
        # decoupled stack, measured as a crystal, shows.

    # the window reads `w`; its floor, the rounding of an eigenvector, is
    # eps/cut of `h`, carried onto the window's scale; and it never narrows below
    # the covariance a kept candidate may carry, in the units of H
    B, delta_used, gaps = _pair_basis(E, V, Ep, Vp, delta, w, max_pairs,
                                      np.finfo(float).eps / max(float(cut), 1e-300) * (h / w),
                                      gate=floor)
    if B is None:
        # the exactly degenerate pairs alone are past the budget: no basis, and
        # how many pairs the solve needs
        return {"flat": False, "k_solve": i1, "too_large": int(delta_used)}
    m = int(B.shape[0])
    trace, n_solve = [m], 1
    s_last, scale_last = None, h
    if m == 0:
        # the spectra do not match at all: no X, unitary or not, intertwines.
        # The single-point constraint's own singular values are the gaps.
        return {"flat": False, "k_solve": i1, "n_solve": 1, "h": h, "basis": B,
                "trace": tuple(trace), "window": delta_used, "gaps": gaps,
                "singular_values": np.sort(gaps) * w,
                "scale": float(gaps[-1] * w) if gaps.size else h,
                "empty_ratio": None}

    todo = [j for j in _visit_order(nk) if j != i1]
    done, steady, informative, resumed = {i1}, 0, False, False
    probe_cut = max(_VACUOUS_CUT, float(cut)) * h
    # The floor of the stage cut.  A candidate whose covariance is within
    # `floor` at a point, read in the units of H on a unitary member, has a
    # singular value of at most sqrt(n)*floor there, so no smaller one is
    # evidence that it breaks: where the operator at a point is itself that
    # small -- one band, whose operator IS the violation, or H(k) nearly a
    # multiple of 1, at a Dirac point -- the point says nothing at the precision
    # declared, and the relative cut alone would read a violation inside it as
    # total.
    least = float(np.sqrt(n)) * float(floor)
    # ... and the same floor as a multiple of the cut, the scale the empty
    # verdict's ratio is read against: sqrt(n) where the floor is the cut itself
    least_scale = float(np.sqrt(n)) * (1.0 if floor == cut else float(floor) / float(cut))
    while todo:
        i = todo.pop(0)
        if i in done:
            continue
        done.add(i)
        n_solve += 1
        Hs, Hr = pair.at(i)
        # ||L_k||, the operator norm of X -> X H_k - H'_k X on the *whole*
        # matrix space, is max_ab |E_b - E'_a| -- two eigenvalue problems.  It
        # is the scale the cut is relative to, and not the largest singular
        # value of the constraint *restricted to the surviving basis*: those
        # two agree while the basis still contains non-solutions and part
        # company exactly when it does not, where the restricted number is the
        # violation's own size and a relative cut against it is vacuous (at
        # m = 1 it reads 1.0 by construction).  The stacked route's cut means
        # this one: its scale is sqrt(nk)*||L||, against candidate values
        # sqrt(nk)*||L_k X||, so its per-point criterion is exactly
        # ||L_k X|| <= cut*||L_k||.
        Es, Er = np.linalg.eigvalsh(Hs), np.linalg.eigvalsh(Hr)
        op = float(max(abs(Er[-1] - Es[0]), abs(Es[-1] - Er[0])))
        reduced = False
        if op > _VACUOUS_CUT * h:            # a vanished operator constrains nothing
            s, vecs = _stage_nullspace(B, Hs, Hr)
            informative, s_last, scale_last = True, s, op
            keep = np.nonzero(s <= max(cut * op, least))[0][::-1]   # ascending in s
            if keep.size == 0:
                trace.append(0)
                return {"flat": False, "k_solve": i1, "n_solve": n_solve, "h": h,
                        "basis": B[:0], "trace": tuple(trace), "window": delta_used,
                        "gaps": gaps, "singular_values": s[::-1].copy(),
                        "scale": op,
                        # against the scale the cut was read on, the floor's included
                        "empty_ratio": float(s[-1] / max(op, least_scale)),
                        "near_miss": np.einsum("j,jab->ab", vecs[:, -1].conj(), B)}
            reduced = bool(keep.size < m)
            if reduced:
                B = np.einsum("jc,jab->cab", vecs[:, keep], B)
                m = int(keep.size)
            del vecs
        trace.append(m)
        steady = 0 if reduced else steady + 1
        if resumed and not reduced:
            break            # the worst point in the whole grid does not cut
        resumed = False
        if steady < stable:
            continue
        # about to stop on `stable` unchanged stages: confirm the surviving
        # basis against **every** supplied point first, and resume at the worst
        # one if it does not hold there.  This is what makes an early stop
        # sound rather than lucky (see `_visit_order`).
        i_bad, _ = _grid_probe(pair, B, n, probe_cut)
        if i_bad < 0 or i_bad in done:
            break
        todo.insert(0, i_bad)
        steady, resumed = 0, True

    # No stage was informative: the Sylvester operator vanished against max|H|
    # at every point sifted, so there is no scale of its own and `max|H|` is
    # what the (vanishing) constraint is measured against -- the same
    # convention the vacuity branch uses.
    if s_last is None:
        s_last, scale_last, informative = np.zeros(m), h, False
    scale = float(scale_last) if informative else h
    return {"flat": False, "k_solve": i1, "n_solve": n_solve, "h": h, "basis": B,
            "trace": tuple(trace), "window": delta_used, "gaps": gaps,
            "singular_values": np.asarray(s_last)[::-1].copy(), "scale": scale,
            "empty_ratio": None}


def _orthonormal_basis(B, n):
    """Re-orthonormalize a sifted basis to the package's ``‖·‖_F = sqrt(n)``.

    The sift's basis is orthonormal *by construction* — the stage-0 outer
    products are, and every stage multiplies by a matrix of orthonormal columns
    — so this only removes the accumulated rounding, and the order (ascending
    in the last stage's singular values) survives it.
    """
    d = int(B.shape[0])
    if d == 0:
        return np.zeros((0, n, n), dtype=complex)
    Q, _ = np.linalg.qr(B.reshape(d, -1).T)
    return np.ascontiguousarray(Q.T).reshape(d, n, n) * np.sqrt(n)


def _candidates_and_search(basis, dim, n):
    """The candidate pool and the ``dim >= 2`` seeded unitary search."""
    cands, search = list(basis), {}
    if dim >= 2:
        rng = np.random.default_rng(_UNITARY_PROBE_SEED)
        coef = (rng.normal(size=(_UNITARY_PROBES, dim))
                + 1j * rng.normal(size=(_UNITARY_PROBES, dim)))
        pool = np.concatenate([basis, np.einsum("pa,aij->pij", coef, basis)])
        pool *= (np.sqrt(n) / np.linalg.norm(pool, axis=(1, 2)))[:, None, None]
        sigma_min = np.linalg.svd(pool, compute_uv=False)[:, -1]
        pick = int(np.argmax(sigma_min))
        w_, _, vh_ = np.linalg.svd(pool[pick])
        cands.append(w_ @ vh_)             # the polar factor is itself a solution
        search = {"n_probes": int(_UNITARY_PROBES), "seed": int(_UNITARY_PROBE_SEED),
                  "picked": pick, "sigma_min": float(sigma_min[pick]),
                  "sigma_min_basis": float(sigma_min[:dim].max())}
    return cands, search


def _candidate_measures(cands, n, *, full=None, pair=None):
    """Polar factors, their full-grid covariance residuals and their unitarity.

    ``full`` is ``(H_src, H_rot)`` already in memory — the stacked route has
    them by construction and the arithmetic is one batched product.  ``pair``
    is a :class:`_ConstraintPair` instead, and the residual then **streams**
    over the grid in the chunks :class:`_GridHamiltonians` bounds, so the eigen
    route never holds ``nk·n²`` anything either.  Both measure the same number:
    ``max_k ‖U H(k)^(*) U† − H(Rk)‖_max`` over *every* supplied k-point.
    """
    eye = np.eye(n)
    unitaries = np.empty((len(cands), n, n), dtype=complex)
    unitarity = np.empty(len(cands))
    residuals = np.zeros(len(cands))
    for a, X in enumerate(cands):
        w_, _, vh_ = np.linalg.svd(X)
        unitaries[a] = w_ @ vh_
        unitarity[a] = float(np.abs(X.conj().T @ X - eye).max())
    if full is not None:
        Hsrc, Hr = full
        for a in range(len(cands)):
            U = unitaries[a]
            residuals[a] = float(np.abs(U @ Hsrc @ U.conj().T - Hr).max())
        return unitaries, residuals, unitarity
    if not cands:                        # dim == 0: nothing to measure anywhere
        return unitaries, residuals, unitarity
    for _, _, Hsrc, Hr in pair.chunks():
        for a in range(len(cands)):
            U = unitaries[a]
            residuals[a] = max(residuals[a],
                               float(np.abs(U @ Hsrc @ U.conj().T - Hr).max()))
    return unitaries, residuals, unitarity


def _find_intertwiners(model, k_action, k_pts, *, antiunitary=False, tol=DEFAULT_ENERGY,
                       cut=None, solver="eigen", window=None,
                       max_window=_EIGEN_WINDOW_CEIL, max_bytes=512e6,
                       degenerate_max_bytes=None, stable=_SIFT_STABLE, hamiltonians=None,
                       max_solve_entries=2_000_000,
                       params=None, _floor=None) -> _IntertwinerSpace | Refusal:
    r"""Every matrix that commutes ``H(k)`` with ``H(Rk)``.

    An empty space proves the operation is broken, a two-dimensional one means
    the model conserves something extra.  What is solved is
    ``X H(k) = H(Rk) X`` (``X H(k)* = H(Rk) X`` for an antiunitary operation)
    simultaneously at every k of a supplied grid — linear in ``X``, with no
    tensor structure assumed for what the operation does to orbitals, so an
    empty solution space is a *proof* and not a failed fit.  A space of
    dimension one or more is certified not by the count but by
    :attr:`_IntertwinerSpace.residuals`, the covariance each candidate achieves
    over every supplied k-point in the units of ``H``.  What these verdicts
    mean for a model is walked through in
    ``docs/guide/group.md#when-the-hamiltonian-has-less-symmetry-than-its-cell``.

    Parameters
    ----------
    model : pythtb.TBModel
        The model whose Bloch Hamiltonian is measured.  ``H(k)`` is taken from
        ``model.hamiltonian(..., flatten_spin_axis=True)``: spinful models are
        solved in the flattened ``nstate = norb * nspin`` basis with **spin as
        the fast index** (``state = orbital * nspin + spin``).
    k_action : callable or array, shape (dim_k, dim_k)
        The operation's action on reduced k, ``k -> Rk``.  A callable maps an
        ``(Nk, dim_k)`` array to an ``(Nk, dim_k)`` array; an array ``M`` is
        applied as ``k @ M.T``.  **The rotated k must not be wrapped into the
        first Brillouin zone**: in convention I ``H(k + G) ≠ H(k)``, and the
        covariance identity holds for the unwrapped image.  For an ``irrep``
        operation use ``op.transform_k`` on the 3D reduced k and restrict to
        the periodic components (``groups.GroupData.k_action_of`` does this).
    k_pts : array, shape (Nk, dim_k)
        Where the constraint is imposed.  Generic points carry the most
        information; a Γ-centred ``n**dim_k`` mesh with ``n = 8`` is the
        package default.  One k-point is never enough — every function of
        ``H(k)`` commutes with ``H(k)``.
    antiunitary : bool, optional
        Solve ``X H(k)* = H(Rk) X`` instead (time-reversal-type operations;
        the k-action of an antiunitary operation already contains the sign
        flip, which ``irrep``'s ``transform_k`` supplies).
    tol : float, optional
        **The declared symmetry tolerance**, absolute and in the units of
        ``H``: :func:`..groups._validate_group` gates its covariance residuals
        on it, and here it is the number the other two follow — ``cut``
        defaults to it and ``window`` is derived from it.  Default ``1e-8``.  A
        wannier90 Hamiltonian is symmetric only to its own numerical defect and
        has to declare one.
    cut : float, optional
        The **nullspace cut**, relative and dimensionless: a singular value
        counts as zero when ``s <= cut·‖L‖``, with ``‖L‖`` the constraint
        operator's own norm (:attr:`_IntertwinerSpace.scale`), or when it is at
        most ``sqrt(n)·cut`` read in the units of ``H`` (``sqrt(n·N_k)·cut`` on
        the stacked operator of ``N_k`` points) -- the most a unitary candidate
        within the cut can have, so that where the operator is itself that
        small, as one band's is wherever its ``h(k)`` and ``h(Rk)`` nearly
        agree, the cut never discards what the gate would keep.  ``None``
        (default) is ``tol``.  Declare it apart from ``tol`` to move where the
        nullspace is cut without moving the gate the candidates are judged by —
        a model whose *arithmetic* is noisier than its physics.
    solver : {"eigen", "stacked"}, optional
        Which route solves the constraint.  ``"eigen"`` (default) is the one to
        use; ``"stacked"`` is the audit oracle and the faithful instrument for
        a *near*-symmetry, where the eigen route's window inflates the
        breaking, and it is affordable only at small ``nstate``.
    window : float, optional
        The eigen route's matching window ``δ``, relative to ``max|H − μ·1|``
        at the parametrization point, ``μ`` the mean on-site energy: the
        magnitude a constant shift of ``H`` does not move, while the route's
        tests of rounding read ``max|H|`` itself, which it does.  ``None``
        (default) derives it from ``tol``: a
        near-solution of relative residual ``ρ`` has cross terms ``ρ/gap``
        between bands, so a raised tolerance needs a wider window before a
        solution good to it is representable at all.  An explicit value is the
        **starting** window; pass ``max_window`` equal to it to forbid the
        escalation.  In the units of ``H`` the window is never narrower than
        the covariance a kept candidate may carry -- the cut read in those
        units, or the gate a declared precision sets -- so that levels the
        precision does not tell apart are paired at whichever point the solve
        is parametrized at.
    max_window, stable : float, int, optional
        The escalation's ceiling (default ``0.3``) and the number of
        consecutive stages that must leave the basis size unchanged before the
        sift stops (default 3).
    max_bytes : float, optional
        The memory the eigen route's trial basis may take at one k-point, one
        ``16 · nstate²``-byte matrix per pair of eigenvectors at matching
        energies (default ``512e6`` bytes), and with it the **window valve**: a
        nearly degenerate spectrum that matches more pairs than this holds has
        its window narrowed to the nearest pairs, and the window used is
        reported (:attr:`_IntertwinerSpace.window`).
    degenerate_max_bytes : float, optional
        The **ceiling** for a spectrum with exactly degenerate levels, whose
        pairs the valve cannot drop without returning a space that is wrong
        rather than coarse.  Where the window narrowed to ``max_bytes`` would
        drop them, the solve is run again from its start with this budget, as
        a call with ``max_bytes`` equal to it runs it; the valve, and every
        solve it fits, are unchanged.  ``None`` (default) is ``max_bytes``
        itself, and a value below ``max_bytes`` reads as ``max_bytes``.
        :func:`..groups._validate_group` and :func:`..groups.realized_group`
        pass their ``max_bytes`` here.
    hamiltonians : _GridHamiltonians, optional
        A shared evaluator for ``H`` on **this** ``k_pts``.  ``H(k)`` does not
        depend on which operation is being tested, so passing one instance to
        every operation of a group evaluates the grid once;
        :func:`..groups._validate_group` does exactly that.
    max_solve_entries : int, optional
        ``solver="stacked"`` only.  Size cap on the stacked constraint matrix,
        in entries: a grid that would exceed it is solved on a deterministic
        subset of its own points, and every candidate is then verified against
        the full grid.  Default ``2_000_000``.
    params : dict, optional
        Model parameter values, as ``model.hamiltonian`` takes them.

    Returns
    -------
    _IntertwinerSpace or Refusal
        ``.dim == 0`` is a **proof** that no ``X`` — unitary or not —
        intertwines the Hamiltonian at the sampled k-points, and the space is
        then falsy, as a refusal is: ``isinstance(out, Refusal)`` tells the
        two apart.  ``.dim >= 1``
        returns the solution basis, unitary candidates and the covariance
        residual each achieves on the full supplied grid.  A **vacuous**
        constraint — ``H(k)`` and ``H(Rk)`` the same scalar matrix at every
        sampled k, which is what a one-band model does at every operation its
        ``h(k)`` obeys — reports the whole matrix space, ``.dim == nstate**2``
        with ``.vacuous`` set, rather than a nullspace read out of rounding
        noise; ``1`` is returned first and its residual is the verdict.

    Refusals
    --------
    ``"detector_too_large"``
        the eigen route's trial basis at a k-point needs more memory than
        the ceiling (``degenerate_max_bytes``, or ``max_bytes`` without it)
        allows, its exactly degenerate pairs alone being past it.  A
        narrower window would return a wrong space, so none is returned: a
        limit on memory, not a statement about the model.  The refusal says
        what the solve needs and the ``max_bytes=`` that runs it.  This is
        the one place the limit is decided: wherever the detector runs, the
        refusal is handed back as it is, :func:`..groups._validate_group` and
        :func:`..groups.realized_group` included, and nothing converts it to
        an exception.

    Notes
    -----
    **Validity conditions.**

    * The verdict is only as strong as the grid in one direction: a nonzero
      nullspace found on too few or too special k-points can be an accident of
      the sample, so ``.dim >= 1`` is *checked* by the reported ``residuals``
      (recomputed directly from ``H``, not from the SVD) rather than trusted.
      ``.dim == 0`` is sound with any grid — a symmetry that held everywhere
      would have held on the sample too.
    * **What is solved on part of the grid is verified on all of it**, whichever
      route ran, so an early stop and a subset solve are measurements rather
      than guesses: ``dim == 0`` on a subset of the constraints is ``dim == 0``
      on all of them, and an over-reported ``dim`` dies by its own residual.
    * **The dimension never comes from the normal equations**: they square the
      constraint and square-root its singular values, flooring a true
      nullvector at ``sqrt(eps) ≈ 1e-8``, the default tolerance itself, so an
      exact symmetry and its own inverse could land on opposite sides of the
      cut.
    * **A vacuous constraint is not an acceptance.**  ``X H(k) - H(Rk) X`` is
      the zero operator exactly when ``H(k)`` and ``H(Rk)`` are the same scalar
      matrix at every sampled k — automatic for a one-band model whose ``h(k)``
      obeys the operation, otherwise a flat, fully degenerate Hamiltonian.  The
      commutant of ``{H(k)}`` is then everything, so the whole matrix space is
      the true solution space and is reported as such (``.vacuous``,
      ``dim = nstate**2``, ``basis[0] = 1``) rather than read out of rounding
      noise.  The identity candidate's ``residual`` is exactly
      ``max_k |H(k) - H(Rk)|``, so the operation is still certified on a number
      measured from ``H``.
    * ``residuals`` are absolute, in the units of the Hamiltonian; compare them
      against the scale of ``H`` itself, not against 1.
    * ``dim >= 2`` means the model carries a conserved operator, and the
      solutions are then ``D·A`` with ``A`` the commutant of ``{H(k)}`` — a
      ``*``-algebra, so the space contains a *unitary* member.  That member is
      searched for rather than manufactured by polar-projecting an arbitrary
      and often singular basis element, whose polar factor acts arbitrarily on
      its kernel and would refuse a genuine symmetry.  The residual over the
      full grid remains the arbiter; the algebra is never trusted in its place.
    * The solution is defined only up to an overall complex factor and, when
      ``dim > 1``, up to a change of basis in the solution space.  Fixing that
      phase is **not** done here — unnormalized intertwiners silently break
      unitary eigen-projections; :func:`normalize_rep` owns it.

    Examples
    --------
    Detect that a square-lattice model with anisotropic hoppings breaks the
    four-fold rotation but keeps the two-fold::

        C4 = np.array([[0, -1], [1, 0]])
        _find_intertwiners(model, C4, k_pts).dim        # 0  -> proof of breaking
        _find_intertwiners(model, -np.eye(2), k_pts).dim  # 1  -> C2 survives
    """
    k_pts = np.atleast_2d(np.asarray(k_pts, dtype=float))
    dim_k = int(model.lattice.dim_k)
    if k_pts.shape[1] != dim_k:
        raise ValueError(f"k_pts must have {dim_k} columns, got {k_pts.shape[1]}")
    if k_pts.shape[0] < 2:
        raise ValueError("_find_intertwiners needs at least two k-points: at a single k "
                         "every function of H(k) is an intertwiner")
    if solver not in ("eigen", "stacked"):
        raise ValueError(f"solver must be 'eigen' or 'stacked', got {solver!r}")
    params = dict(params or {})
    cut = float(tol) if cut is None else float(cut)
    # the covariance, in the units of H, a candidate the nullspace keeps may carry
    # at a point: the cut read in those units, unless a declared precision, whose
    # cut is relative to the Hamiltonian's own magnitude, sets it at the gate
    floor = cut if _floor is None else float(_floor)

    act = _as_k_action(k_action, dim_k)
    k_rot = np.asarray(act(k_pts), dtype=float)
    if k_rot.shape != k_pts.shape:
        raise ValueError(f"k_action returned shape {k_rot.shape}, expected {k_pts.shape}")

    n = int(model.nstate)
    n2 = n * n
    nk = int(k_pts.shape[0])
    eye = np.eye(n)

    if hamiltonians is None:
        src_grid = _GridHamiltonians(model, k_pts, params)
    else:
        src_grid = hamiltonians
        if src_grid.n_kpts != nk or not np.array_equal(src_grid.k_pts, k_pts):
            raise ValueError("hamiltonians= was built on a different k-grid than the "
                             "k_pts supplied here; a shared cache is keyed by its grid")
        if params and params != src_grid.params:
            raise ValueError("hamiltonians= was built with different model parameters "
                             f"({src_grid.params}) than the ones passed here ({params})")
    rot_grid = src_grid.rotated(k_rot)

    window_used, k_solve, trace, escalated = 0.0, -1, (), ()

    if solver == "stacked":
        H, Hr = src_grid.full(), rot_grid.full()
        Hsrc = H.conj() if antiunitary else H

        def _constraint(a, b):
            """Stacked ``X H(k) - H(Rk) X`` operator on row-major ``vec(X)``."""
            # vec(X H) = (I (x) H^T) vec(X);  vec(H' X) = (H' (x) I) vec(X)
            return (np.einsum("ij,kab->kiajb", eye, a.transpose(0, 2, 1)).reshape(-1, n2, n2)
                    - np.einsum("kij,ab->kiajb", b, eye).reshape(-1, n2, n2))

        # ---- solve.  The audit oracle: one full-precision stacked SVD.  A grid
        # too large to stack is cut down to a deterministic even-strided subset
        # of its own points -- never reordered, and never degraded to the normal
        # equations, whose sqrt(eps) floor would make dim == 0 an unsound verdict.
        nk_solve = int(min(nk, max(2, max_solve_entries // (n2 * n2))))
        if nk_solve >= nk:
            method, nk_solve, take = "stacked", nk, slice(None)
        else:
            method = "stacked_subset"
            take = np.rint(np.linspace(0.0, nk - 1.0, nk_solve)).astype(int)
        blocks = _constraint(Hsrc[take], Hr[take])
        _, s, Vh = np.linalg.svd(blocks.reshape(-1, n2), full_matrices=False)
        order = np.argsort(s)
        s = s[order]
        vecs = Vh.conj().T[:, order]

        # ---- cut.  Two regimes, and the second is not the first with a smaller
        # number: when the constraint operator *vanishes* it has no scale of its
        # own, so a relative cut has nothing to be relative to and would read its
        # own rounding noise as one.  Vacuity is therefore judged absolutely,
        # against the scale of H over the grid -- the scale that noise is
        # proportional to -- and a vacuous constraint means every X solves.
        h_scale = float(max(np.abs(H).max(), np.abs(Hr).max()))
        if not h_scale > 0.0:
            h_scale = 1.0                   # H is identically zero: any scale will do
        vacuous = bool(float(s[-1]) <= _VACUOUS_CUT * h_scale)

        if vacuous:
            scale, dim = h_scale, n2
            basis = _identity_anchored_matrix_basis(n)
        else:
            scale = float(s[-1])
            # the eigen route's floor, on the stacked operator: a unitary
            # candidate within `floor` at each of the points solved has a stacked
            # singular value of at most sqrt(n * nk_solve) * floor
            dim = int(np.count_nonzero(
                s <= max(cut * scale, float(np.sqrt(n * nk_solve)) * floor)))
            basis = np.empty((dim, n, n), dtype=complex)
            for a in range(dim):
                X = vecs[:, a].reshape(n, n)
                basis[a] = X / np.linalg.norm(X) * np.sqrt(n)

        cands, search = _candidates_and_search(basis, dim, n)
        unitaries, residuals, unitarity = _candidate_measures(
            cands, n, full=(Hsrc, Hr))
    else:
        method, vacuous = "eigen", False
        pair = _ConstraintPair(src_grid, rot_grid, antiunitary)

        start = eigen_window(tol) if window is None else float(window)
        delta = start
        # the window valve, and the ceiling a spectrum with exactly degenerate
        # blocks may raise the budget to: never below the valve
        max_pairs = max(1, int(float(max_bytes) // (16 * n2)))
        ceiling = max(float(max_bytes), float(max_bytes if degenerate_max_bytes is None
                                              else degenerate_max_bytes))
        ceiling_pairs = max(max_pairs, int(ceiling // (16 * n2)))
        tried, forced = [], None
        while True:
            rec = _sift(pair, nk, n, cut=cut, floor=floor, delta=delta,
                        stable=int(stable), max_pairs=max_pairs, k1=forced)
            if "too_large" in rec:
                if max_pairs < ceiling_pairs:
                    # the valve would drop exactly degenerate pairs: the solve
                    # again from its start, as a call with the ceiling for its
                    # budget runs it
                    max_pairs, delta, tried, forced = ceiling_pairs, start, [], None
                    continue
                return _detector_too_large(rec["too_large"], n, ceiling)
            if rec["flat"]:
                # every point degenerate to within the window: the absolute
                # vacuity test decides, exactly as the stacked route's does
                h_scale = float(rec["grid_h"])
                s = np.asarray(rec["grid_s_max"], dtype=float)
                if float(s[-1]) <= _VACUOUS_CUT * h_scale:
                    vacuous, scale, dim = True, h_scale, n2
                    basis = _identity_anchored_matrix_basis(n)
                    k_solve, window_used = int(rec["k_solve"]), float(delta)
                    nk_solve = int(rec["n_solve"])
                    break
                forced = int(rec["k_solve"])       # parametrize there anyway
                continue
            dim = int(rec["basis"].shape[0])
            nxt = delta * _EIGEN_ESCALATION
            if dim == 0 and nxt <= max_window * (1.0 + 1e-9):
                # would a wider window have represented what the sift threw
                # away?  Only then is a re-sift anything but wasted work.
                ratio = rec["empty_ratio"]
                worth = (float(rec["gaps"][0]) <= max_window if ratio is None
                         else ratio <= cut / delta)
                wider = int(np.searchsorted(rec["gaps"], nxt, side="right"))
                if worth and wider <= max_pairs:
                    tried.append(delta)
                    delta, forced = nxt, int(rec["k_solve"])
                    continue
            basis = _orthonormal_basis(rec["basis"], n)
            s, scale = np.asarray(rec["singular_values"], float), float(rec["scale"])
            k_solve, window_used = int(rec["k_solve"]), float(rec["window"])
            nk_solve, trace, escalated = int(rec["n_solve"]), rec["trace"], tuple(tried)
            break

        cands, search = _candidates_and_search(basis, dim, n)
        unitaries, residuals, unitarity = _candidate_measures(cands, n, pair=pair)

    return _IntertwinerSpace(dim=dim, basis=basis, unitaries=unitaries,
                             residuals=residuals, unitarity=unitarity,
                             singular_values=s, scale=scale, tol=tol, cut=cut,
                             method=method, n_kpts=nk, n_kpts_solve=nk_solve,
                             antiunitary=bool(antiunitary), vacuous=vacuous,
                             search=search, window=window_used, k_solve=k_solve,
                             basis_size=tuple(int(t) for t in trace),
                             escalations=escalated)


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
#     their integer lattice shifts from ``groups._assign_orbitals``;
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
    :func:`_build_rep` applies it **inside** the uniqueness gate (a candidate that
    is Hamiltonian-covariant but does not multiply is not a representation of
    the group, whatever its covariance says) and again at the end to fill
    :attr:`SpaceGroupRep.product_residual`.  The two agree to the last bit:
    the only thing that happens in between is the multiplication by the
    one-dimensional character ``χ``, a homomorphism with ``χ = ±1``, so it
    cancels out of the identity.

    ``positions`` are indices into ``syms``/``prod``/``spf``; ``D`` is keyed the
    same way.  Pairs whose product falls outside the represented set are
    skipped — for a closed set (which :func:`_build_rep` checks separately)
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

    Delegated to ``groups._pad_cell`` so that the 3D presentation is
    literally the one the group was identified in — a second implementation of
    it would be a second convention.
    """
    from . import groups as _groups
    _lat, pos3, _padded = _groups._pad_cell(model, group.vacuum_length,
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
    of at the reference site of their orbit (:func:`_slot_order`), or in
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
    (:func:`_site_multiplets` transports it), so the image is a multiplet and not
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
    """Normalize :func:`_build_rep`'s ``spin=`` to a :class:`..spin.SpinBasis` or ``None``.

    Accepts the object itself, the shorthand ``"interleaved"`` / ``"blocked"``,
    or a mapping of :func:`..spin.spin_basis` keyword arguments.  Everything that
    can go wrong here is **grammar** — a declaration for the wrong model, a
    spin axis declared on a model that already has one — so it raises
    ``ValueError`` rather than refusing: none of it is a measurement.
    """
    from .spin import SpinBasis, spin_basis

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
    anything measured; :func:`_build_rep` asks it on the sites of the
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
    pair whose two members landed in different pairs — which :func:`_build_rep`
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
    ``order`` is each site's orbitals in slot order (:func:`_slot_order`),
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
    site of its orbit (:func:`_slot_order`), so that the slots of a declared
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


def _declared_parts(declared, site_orbs, site_of, atoms):
    """Stated multiplets, re-keyed onto this assignment's own sites.

    The multiplets are a partition of the model's **orbitals**, and which site
    index holds which multiplet is the assignment's business rather than the
    declaration's — so the multiplets are read as a flat set, each assigned to
    the site its orbitals sit on, and the sites' own order is applied here.
    That is what carries the multiplets from a model to another presentation of
    it, where the same orbitals are grouped in a different order.

    Everything that can go wrong is grammar, and raises ``ValueError``
    (:func:`_declaration_grammar`), which every call that takes a statement
    has asked before anything was measured.
    """
    pairs = tuple(tuple(int(o) for o in atom) for site_atoms in atoms
                  for atom in site_atoms if len(atom) == 2)
    chans = _multiplets_grammar(declared, len(site_of), pairs, site_of)
    return tuple(tuple(sorted((ch for ch in chans if int(site_of[ch[0]]) == s),
                              key=lambda c: c[0]))
                 for s in range(len(site_orbs)))


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


def _slot_order(model, group, assignment, atoms, spaces, *, tol=ZERO):
    r"""Each site's atoms in the order of the atoms they are images of at its orbit's reference site.

    Which orbital at one site of an orbit answers to which at another is what
    the operations carry, never where the orbitals happen to be listed: an
    operation carrying the orbit's reference site onto another carries each of
    the reference's atoms into the atoms its solution space puts weight on
    there, and those share its slot.  Atoms an operation carries into one
    another's span -- the members of a multiplet, or copies of one irrep
    written in bases turned from site to site -- are paired in the order each
    site lists them, which is all a slot says about them: the multiplet they
    land in is decided by the site's own operations.  Of the operations
    reaching a site, the one that pairs the atoms most finely is read.  Where
    none reaches it, or none pairs them one to one, the site keeps its own
    order.  A weight below ``tol`` of the largest links nothing: ``1e-9``, or
    the declared tolerance where it is larger (:func:`.._tolerance.noise_floor`).

    ``atoms[s]`` is site ``s``'s atoms in its own order; returns them per site
    in slot order.
    """
    nspin = int(model.nspin)
    orbit_of = np.asarray(assignment.orbit_of_site, dtype=int)
    perms = np.asarray(assignment.permutations, dtype=int)
    pos_of_ind = {int(s.ind): i for i, s in enumerate(group.symmetries)}
    ref = {}
    for s in range(len(atoms)):
        ref.setdefault(int(orbit_of[s]), s)

    def _states(atom):
        return np.asarray([o * nspin + t for o in atom for t in range(nspin)], dtype=int)

    out = [list(site) for site in atoms]
    for t in range(len(atoms)):
        s0 = ref[int(orbit_of[t])]
        n = len(atoms[t])
        if t == s0 or n <= 1 or n != len(atoms[s0]):
            continue
        cols = [_states(a) for a in atoms[s0]]
        rows = [_states(a) for a in atoms[t]]
        best = None
        for ind in sorted(spaces):
            p = pos_of_ind.get(int(ind))
            if p is None or int(perms[p][s0]) != t:
                continue
            V = np.asarray(spaces[ind])
            W = np.array([[float(np.abs(V[:, r][:, :, c]).max()) for c in cols]
                          for r in rows])
            if float(W.max()) <= 0.0:
                continue
            linked = _Union()
            for i in range(n):
                for j in range(n):
                    if float(W[i, j]) > tol * float(W.max()):
                        linked.union(("s", j), ("t", i))
            comps = {}
            for j in range(n):
                comps.setdefault(linked.find(("s", j)), ([], []))[0].append(j)
            for i in range(n):
                comps.setdefault(linked.find(("t", i)), ([], []))[1].append(i)
            if any(len(a) != len(b) for a, b in comps.values()):
                continue                  # not one to one: this operation says nothing
            if best is None or len(comps) > best[0]:
                best = (len(comps), comps)
        if best is None:
            continue
        slot = [None] * n
        for js, its in best[1].values():
            for j, i in zip(sorted(js), sorted(its)):
                slot[j] = atoms[t][i]
        out[t] = slot
    return out


def _site_multiplets(model, group, assignment, spin, validated, declared=None, tol=None,
                     sectors=None):
    r"""The multiplets this Hamiltonian glues at every site, measured.

    The rule, one sentence: the **finest** partition of each site's orbitals
    into multiplets for which every validated operation's solution space —
    ``validated._bases[ind]``, the detector's whole nullspace for that operation
    — still contains a nonsingular member in **block form**, block-diagonal
    within every site the operation fixes, the geometric permutation between
    the sites of an orbit, and a declared spinor pair never split.  *Finest*
    assumes nothing the hoppings do not force: two decoupled orbitals at one
    site, each with hoppings of its own, are two one-dimensional multiplets and
    never a doublet, because a diagonal member exists.  What those two carry
    *relatively* is a different question, and one a Hamiltonian that neither
    permutes nor couples them cannot answer.  Two identical ones are copies of
    one another, and those are one multiplet (below).

    The search is a refinement from singletons.  For a candidate partition and
    an operation, the members of the solution space that are in block form are
    measured (:func:`_in_blocks`, :func:`_unfilled_blocks`); where the space
    cannot fill a block, what the space *does* couple that multiplet to says
    which multiplets merge.  Only operations the **coarsest** partition — the
    whole site as one multiplet, its blocks carried round the orbit as the
    tensor test carries them — makes realizable drive it: an operation not even
    that can carry is refuted by its own tensor structure, and gluing orbitals
    would substitute a partition for that verdict.  Sites in one orbit share one
    partition, so a merge measured at one site is transported to the whole
    orbit rather than measured again there, and the refinement terminates at
    the whole site at worst.

    **Orbitals on a site are glued by the operations that fix it.**  Those
    decide each site's finest multiplets, and every merge they ask for is made
    before any other.  An operation that carries a site onto another asks for
    a merge where the two sites write their orbitals in bases it does not carry
    into one another, and that merge is made only inside one isotypic
    component: the multiplets it would join carry copies of one site irrep, at
    every site of the orbit.  Their characters under the site group say so —
    the traces of their blocks in one member of each operation's solution
    space, so that the member's free phase is common to both, and on every
    member where the space holds more than one, since a relation the hoppings
    leave open is no evidence of one irrep.  Copies of one irrep written in
    bases rotated from site to site (``A'`` and ``A'`` at a mirror site) become
    one multiplet.  A merge across different irreps is not made, and the
    operation that asked for it stays refuted by its own tensor structure: a
    half translation the hoppings obey only dressed by a rotation of one
    site's frame would glue a ``p_x`` even under the site's mirror to a
    ``p_y`` odd under it, and it is no symmetry of the crystal.

    **No basis splits what the conserved matrices join.**  The matrices that
    commute with ``H(k)`` at every ``k`` and act within sites split the states
    into sectors (:mod:`._sectors`) whatever basis the orbitals are written in,
    and two orbitals of one site that each hold part of one sector are one
    multiplet.  A square's ``p_x, p_y`` pair that nothing couples is two
    sectors; written as ``(p_x ± p_y)/√2`` each orbital holds half of each, and
    the pair is one multiplet there as in every other basis, though in that
    one basis the fourfold's solution space holds a diagonal member and no
    operation would bind the two.  These merges are made first and carried
    round the orbit as every merge is.  **Identical copies are one multiplet**:
    where a sector holds several copies of one set of states, each copy written
    as orbitals of its own, the matrices exchanging the copies join every state
    of one copy to the same state of every other, so that a content mixing
    them -- two identical ``s`` orbitals declared the doublet ``E``, with a spin
    or without -- can be read on them.  ``sectors`` is the reading
    (:func:`.._sectors.read_sectors`), made here where none is handed in.

    Two incomparable finest partitions need an accidental degeneracy (a
    decoupled orbital whose bands coincide with a partner's).  The tie is
    resolved deterministically — the lowest-indexed partner — and recorded on
    :attr:`SiteMultiplets.accidental`; nothing is refused.

    **Which orbital answers to which across an orbit** is measured too, never
    read off the order the sites list their orbitals in: the operation carrying
    the orbit's reference site onto another says which of the reference's
    orbitals each one there is the image of (:func:`_slot_order`), and a slot is
    that correspondence.  So a site that lists ``(p_y, p_x)`` where the
    reference lists ``(p_x, p_y)`` has each orbital in its own multiplet, each
    reading its own irrep, and a merge is asked only where the bases genuinely
    turn.

    ``declared`` holds multiplets stated rather than measured, by
    ``multiplets=``: a :class:`SiteMultiplets`, or the multiplets of every
    site, each given as the orbitals it holds.  They are used verbatim, and a
    statement carries the multiplets from one presentation of a model to
    another.

    ``tol`` is the declared symmetry tolerance relative to ``max|H|``
    (:func:`.._tolerance.noise_floor`): every threshold here that measures noise -- how far
    a member sits outside the blocks, when a block is empty or singular, which
    weights link or couple, when two characters agree, which operations the
    whole-site partition carries -- is the larger of its exact default and it,
    and the record carries it for every later test of its blocks.  ``None``,
    nothing declared, is every default exactly.  A Wannier basis whose orbital
    frame is off the ideal one by less than the declared precision then reads
    the multiplets of the ideal one, measured as holding only within the
    declared precision: the Hamiltonian's own symmetry matrices mix those
    multiplets at the size of the turn, which is how far outside their block
    form the members they were read off reach (:attr:`SiteMultiplets._mixing`,
    :func:`_block_members`), and so do its conserved matrices, by how far a
    sector reaches between two of them (:func:`.._sectors.joined`).
    """
    site_orbs = tuple(tuple(int(o) for o in orbs) for orbs in assignment.site_orbitals)
    atoms = _atoms_of_site(site_orbs, spin)
    pos_of_ind = {int(s.ind): i for i, s in enumerate(group.symmetries)}
    if declared is not None:
        if (isinstance(declared, SiteMultiplets)
                and tuple(declared._site_orbitals) == site_orbs):
            # the same record, about the same sites: carried, not re-made, so
            # that multiplets measured once keep saying so wherever they are used
            return declared
        parts = _declared_parts(
            declared.by_site if isinstance(declared, SiteMultiplets) else declared,
            site_orbs, np.asarray(assignment.site_of_orbital, dtype=int), atoms)
        spaces = site_mixing = order = None
        if validated is not None and max((len(a) for a in atoms), default=1) > 1:
            spaces, _site, site_mixing = _site_spaces(model, group, assignment, spin,
                                                      validated, tol=tol)
        if spaces is not None:
            order = tuple(tuple(int(o) for a in site for o in a)
                          for site in _slot_order(model, group, assignment, atoms, spaces,
                                                  tol=noise_floor(ZERO, tol)))
        layout = _multiplets_of(model, assignment, spin, parts, source="declared",
                                order=order, tol=tol)
        if not tol or not spaces:
            return layout
        # WHETHER THE DECLARED MULTIPLETS HOLD EXACTLY or only within the
        # declared tolerance: how far outside their block form the members of
        # each operation's space that sit in it reach
        mixing = 0.0
        for ind, space in spaces.items():
            image = _channel_permutation(assignment, pos_of_ind[int(ind)], model, layout)
            got, reach = _block_members(space, _ansatz_blocks(layout._chan_states, image, 1),
                                        tol=noise_floor(_IN_BLOCKS, tol))
            if len(got):
                mixing = max(mixing, float(site_mixing.get(int(ind), 0.0)), float(reach))
        return replace(layout, _mixing=mixing)
    singletons = _singleton_parts(site_orbs, atoms)
    if max((len(a) for a in atoms), default=1) <= 1 or validated is None:
        # nothing at any site to glue, so nothing to measure: the finest
        # partition is the only one, and it is the whole verdict
        return _multiplets_of(model, assignment, spin, singletons, source="singletons",
                              tol=tol)
    bases = getattr(validated, "_bases", None) or {}
    if not bases:
        return _multiplets_of(model, assignment, spin, singletons, source="singletons",
                              tol=tol)

    orbit_of = np.asarray(assignment.orbit_of_site, dtype=int)
    perms = np.asarray(assignment.permutations, dtype=int)
    # THE SITE BLOCKS, once: they are the operation's own permutation of the
    # sites and no partition of a site's orbitals changes them, so the space
    # each operation is asked about is restricted to them before the
    # refinement starts.
    spaces, site_layout, site_mixing = _site_spaces(model, group, assignment, spin,
                                                    validated, tol=tol)
    # THE SLOTS: which atom of every site answers to which of its orbit's
    # reference site, as the operations carry them.  From here on `atoms[s][j]`
    # is the atom of site `s` in slot `j`.
    atoms = _slot_order(model, group, assignment, atoms, spaces,
                        tol=noise_floor(ZERO, tol))
    order = tuple(tuple(int(o) for a in site for o in a) for site in atoms)
    slot_of, orbs_of_slot = {}, {}
    for s, site_atoms in enumerate(atoms):
        for j, a in enumerate(site_atoms):
            for o in a:
                slot_of[int(o)] = (int(orbit_of[s]), j)
            orbs_of_slot.setdefault((int(orbit_of[s]), j), tuple(int(o) for o in a))
    merged, pending = _Union(), []
    # NO BASIS SPLITS WHAT THE CONSERVED MATRICES JOIN.  The sectors of the
    # matrices that commute with H at every k and act within sites are sets of
    # states, whatever basis the orbitals are written in, and two orbitals of a
    # site that each hold part of one sector are one multiplet: a square's
    # p_x, p_y pair nothing couples, written as (p_x +- p_y)/sqrt(2), where the
    # fourfold's solution space holds a diagonal member and no operation would
    # bind the two.  Made first, and carried round the orbit as every merge is.
    from . import _sectors
    if sectors is None:
        sectors = _sectors.read_sectors(model, group, assignment, spin, validated,
                                        tol=tol or None)
    # ... and IDENTICAL COPIES ARE ONE MULTIPLET: where the conserved matrices
    # exchange copies of one set of states, which representation the copies
    # carry together is the freedom the hoppings leave, and a content that mixes
    # them -- two identical s orbitals declared the doublet E -- has to be
    # readable on them, so every state of a copy is one multiplet with the same
    # state of every other.  On states carrying a spin the copies are read in
    # the orbital factor, every copy carrying the operation's own spin lift
    joins, joining = {}, noise_floor(_IN_BLOCKS, tol)
    if sectors is not None:
        for s, site_atoms in enumerate(atoms):
            joins[s] = _sectors.joined(sectors, site_atoms)
            for i in range(len(site_atoms)):
                for j in range(i + 1, len(site_atoms)):
                    if float(joins[s][i, j]) > joining:
                        merged.union((int(orbit_of[s]), i), (int(orbit_of[s]), j))
    # WHICH OPERATIONS THE MULTIPLETS ARE ABOUT.  Gluing a site's orbitals can
    # only ever help an operation that the COARSEST partition -- the whole site
    # as one multiplet -- makes realizable; an operation not even that can carry
    # is refuted by its own tensor structure (a lattice supergroup's coset
    # partner, a sigma_h partner), and answering it by gluing orbitals would
    # substitute a partition for a verdict.  How far a member may sit from that
    # form and still be realizable is noise, so it reads the declared tolerance
    # too: a model symmetric only within its precision has every member that far off.
    positions = [pos_of_ind[int(i)] for i in validated.indices
                 if int(i) in bases and int(i) in pos_of_ind]
    coarse = _ansatz_defects(model, group, assignment, validated, positions,
                             layout=site_layout, spin=spin,
                             tol=noise_floor(ANSATZ_GATE, tol))
    if isinstance(coarse, Refusal):
        return _multiplets_of(model, assignment, spin, singletons, source="singletons",
                              order=order, tol=tol)
    glueable = {int(ind) for ind, rec in coarse.items() if rec["realizable"]}
    # the noise floors of the weights and characters read below
    link, agree = noise_floor(ZERO, tol), noise_floor(CHARACTER, tol)
    syms = group.symmetries
    # the operations a site's own characters are read under: the unitary ones
    # that fix it, among those the multiplets are about
    fixing_of = {int(s): [int(ind) for ind in spaces if int(ind) in glueable
                          and not syms[pos_of_ind[ind]].time_reversal
                          and int(perms[pos_of_ind[ind]][s]) == int(s)]
                 for s in range(len(site_orbs))}
    sites_of_orbit = {}
    for s in range(len(site_orbs)):
        sites_of_orbit.setdefault(int(orbit_of[s]), []).append(int(s))

    def _verdict(layout, asked, ind):
        """``(p, image, unfilled)`` of one operation on one partition.

        ``asked`` holds the partition's verdicts measured so far, each with how
        far outside its blocks the members in block form reach, so that the
        partition the refinement ends on is read once.
        """
        if ind not in asked:
            p = pos_of_ind[ind]
            image = _channel_permutation(assignment, p, model, layout)
            ok, unfilled, reach = _unfilled_blocks(
                spaces[ind], _ansatz_blocks(layout._chan_states, image, 1), tol=tol)
            asked[ind] = (p, image, () if ok else tuple(unfilled), float(reach))
        return asked[ind][:3]

    # the partition last swept, and its verdicts
    last = {}

    def _sweep(parts):
        """One merge, the first a verdict asks for; ``True`` when one ran.

        The merges an operation asks for at a site it fixes are made first;
        only where none is left is a merge asked across the sites of an orbit
        read, and made only between copies of one site irrep.
        """
        layout = _multiplets_of(model, assignment, spin, parts, order=order)
        asked = {}
        last.clear()
        last.update(parts=parts, layout=layout, asked=asked)

        def _one_irrep(orbit, ja, jb):
            """Do the multiplets of slots ``ja`` and ``jb`` carry copies of one site irrep?

            Asked at every site of the orbit, of the characters the operations
            fixing it carry on the two multiplets: the traces of their blocks in
            the same member of the operation's solution space, so that its free
            phase is common to both, and on every member of it.
            """
            for u in sites_of_orbit[int(orbit)]:
                a = int(layout._chan_of[atoms[u][ja][0]])
                b = int(layout._chan_of[atoms[u][jb][0]])
                here = np.asarray(site_layout._chan_states[u], dtype=int)
                for ind in fixing_of[u]:
                    members = spaces[ind]
                    lift = _lift_of(syms[pos_of_ind[ind]], layout._spinful)
                    size = [float(np.linalg.norm(X[np.ix_(here, here)]))
                            / np.sqrt(len(here)) for X in members]
                    for X, scale in zip(members, size):
                        if scale <= agree * max(size):
                            continue          # a member that does not live here
                        chi = [sum(_proportionality(X[np.ix_(at, at)], lift)[0]
                                   for at in _atom_blocks(layout._chan_states[c],
                                                          layout._spinful))
                               / float(layout._dim(c)) for c in (a, b)]
                        if abs(complex(chi[0]) - complex(chi[1])) > agree * scale:
                            return False
            return True

        for across in (False, True):
            for ind in spaces:
                if int(ind) not in glueable:
                    continue
                p, image, unfilled = _verdict(layout, asked, ind)
                space = spaces[ind]
                # A block the space cannot fill says this partition is too
                # fine: the multiplet the operation must carry there is glued to
                # another one of the same site, and WHICH one the space itself
                # says -- the multiplets of the image site it does put weight on.
                floor = link * max(float(np.abs(space).max()), 1.0)
                for c in unfilled:
                    if (int(perms[p][int(layout._chan_site[c])])
                            != int(layout._chan_site[c])) != across:
                        continue
                    cols = np.asarray(layout._chan_states[c], dtype=int)
                    weight = {}
                    for d in range(int(layout._nchan)):
                        if d == int(image[c]) or merged.find(
                                slot_of[layout._chan_orbs[d][0]]) == merged.find(
                                    slot_of[layout._chan_orbs[c][0]]):
                            continue
                        if int(layout._chan_site[d]) != int(perms[p][
                                int(layout._chan_site[c])]):
                            continue
                        rows = np.asarray(layout._chan_states[d], dtype=int)
                        w = float(np.abs(space[:, rows][:, :, cols]).max())
                        if w > floor:
                            weight[d] = w
                    if not weight:
                        continue
                    top = sorted(d for d, w in weight.items()
                                 if w >= _LAYOUT_MERGE * max(weight.values()))
                    mine = slot_of[layout._chan_orbs[c][0]]
                    if across:
                        # an operation carrying the site onto another glues
                        # copies of one site irrep and nothing else
                        top = [d for d in top if _one_irrep(
                            mine[0], mine[1], slot_of[layout._chan_orbs[d][0]][1])]
                        if not top:
                            continue
                    keys = [slot_of[layout._chan_orbs[d][0]] for d in top]
                    if not merged.union(mine, keys[0]):
                        continue
                    if len(top) > 1:
                        # several partners couple equally: the merge taken is
                        # the lowest-indexed one and the rest are recorded, to
                        # be kept as ties only where the multiplets end with
                        # them still apart
                        pending.append({"operation": int(ind), "chosen": (mine, keys[0]),
                                        "alternatives": tuple((mine, k) for k in keys[1:])})
                    return True
        return False

    parts = singletons
    for _ in range(sum(len(a) for a in atoms) + 1):
        parts = _parts_from_merge(site_orbs, atoms, orbit_of, merged)
        if not _sweep(parts):
            break
    parts = _parts_from_merge(site_orbs, atoms, orbit_of, merged)
    # WHETHER THE MULTIPLETS HOLD EXACTLY, or only within the declared
    # tolerance: how far outside the block form of the partition returned the
    # members it was read off reach, at the site blocks and at the multiplets,
    # over the operations it carries.  The last sweep measured that partition,
    # and is read rather than measured again.
    if last.get("parts") == parts:
        final, asked = last["layout"], last["asked"]
    else:
        final, asked = _multiplets_of(model, assignment, spin, parts, order=order), {}
    mixing = 0.0
    for ind in spaces:
        if int(ind) not in glueable or _verdict(final, asked, ind)[2]:
            continue
        mixing = max(mixing, float(site_mixing.get(int(ind), 0.0)), float(asked[ind][3]))
    # ... and how far a sector reaches between two multiplets of one site,
    # below what joins them
    for s, W in joins.items():
        for i in range(W.shape[0]):
            for j in range(i + 1, W.shape[0]):
                if merged.find((int(orbit_of[s]), i)) != merged.find((int(orbit_of[s]), j)):
                    mixing = max(mixing, float(W[i, j]))
    ties = []
    for rec in pending:
        apart = tuple(alt for alt in rec["alternatives"]
                      if merged.find(alt[0]) != merged.find(alt[1]))
        if not apart:
            continue
        ties.append({
            "operation": int(rec["operation"]),
            "orbitals": tuple(sorted({o for pair in (rec["chosen"],) + apart
                                      for key in pair for o in orbs_of_slot[key]})),
            "chosen": tuple(sorted(o for key in rec["chosen"]
                                   for o in orbs_of_slot[key])),
            "note": ("two partitions are equally fine here, an accidental degeneracy, "
                     "and the multiplet returned binds the lowest-indexed partners")})
    return _multiplets_of(model, assignment, spin, parts, ties=ties, order=order, tol=tol,
                          mixing=mixing)


def _channel_perms(model, assignment, syms, positions, layout, spin):
    """``{p: image}`` — where each operation carries each multiplet.

    Nothing is assumed: the image is lifted from ``assignment.permutations`` and
    the slot order of the site's orbitals (:func:`_channel_permutation`), and a
    **declared** spinor pair additionally has to land on one pair, which
    :func:`_spin_channel_permutation` measures, in the same slot order.  Each
    site's orbitals are read pair by pair as the pairs are declared, so every
    operation carries a pair onto a pair, and one that does not is this package
    disagreeing with itself.  That check is about the pair and the image is
    about the **multiplets**, one of which a pair may share with others, so the
    two are asked separately.  ``positions`` are indices into ``syms``.
    """
    if spin is None:
        return {p: _channel_permutation(assignment, p, model, layout)
                for p in positions}
    order = getattr(layout, "_site_order", None)
    cperm = {}
    for p in positions:
        image, split = _spin_channel_permutation(assignment, p, model, spin, order=order)
        if image is None:
            c, cu, cd = split
            up, dn = spin.states(c)
            raise InternalError(
                f"operation #{syms[p].ind} does not carry declared spinor pair {c} = "
                f"(up {up}, dn {dn}) onto one pair: it sends orbital {up} into pair "
                f"{cu} and orbital {dn} into pair {cd}. Each site's orbitals are read "
                f"pair by pair as the pairs are declared, and an operation carries a "
                f"site onto a site, so it carries each pair onto a pair: this is this "
                f"package's own reading failing, not a statement about the model or "
                f"its pairing.")
        cperm[p] = _channel_permutation(assignment, p, model, layout)
    return cperm


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
    is the freedom :func:`_build_rep` has (a scalar per multiplet orbit per
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


#: The roots of unity of order dividing twelve, ``e^{2πik/12}`` for ``k = 0, …, 11``,
#: each to the last bit: the characters of the one-dimensional irreps of an axis
#: up to sixfold, single- and double-valued.
_HALF_SQRT3 = float(np.sqrt(3.0)) / 2.0
_TWELFTH_ROOTS = tuple(complex(re_, im_) for re_, im_ in (
    (1.0, 0.0), (_HALF_SQRT3, 0.5), (0.5, _HALF_SQRT3), (0.0, 1.0),
    (-0.5, _HALF_SQRT3), (-_HALF_SQRT3, 0.5), (-1.0, 0.0), (-_HALF_SQRT3, -0.5),
    (-0.5, -_HALF_SQRT3), (0.0, -1.0), (0.5, -_HALF_SQRT3), (_HALF_SQRT3, -0.5)))


def _orbit_factor(z, noise):
    r"""The factor an orbit of multiplets carries relative to the reference orbit, as read.

    ``z`` is the ratio measured between the two orbits' blocks of one
    operation.  Within ``noise`` of a root of unity of order dividing twelve --
    a sign, a quarter turn, a character of a three- or sixfold axis, or one of
    those times a spin rotation's sign -- it is that root, exactly: the
    orbitals are written as one another's images, and what the measurement
    adds to the root is rounding, or a numerical model's noise.  Any other
    factor of unit modulus is a phase the basis functions carry, an orbital
    written as ``e^{iφ}`` times its partner's image, and it is kept as
    measured.  ``None`` where ``z`` is no phase: its modulus misses one by
    more than ``noise``.  The build's read-off reads every factor this way, and
    the three gates then say whether what it read is a representation.
    """
    z = complex(z)
    if not abs(abs(z) - 1.0) <= float(noise):
        return None
    root = _TWELFTH_ROOTS[int(round(float(np.angle(z)) * 6.0 / np.pi)) % 12]
    return root if abs(z - root) <= float(noise) else z / abs(z)


def _factor_said(z) -> str:
    """A factor between orbits of multiplets in words: a root of unity by name, a phase by value."""
    z = complex(z)
    if z in _TWELFTH_ROOTS:
        k = _TWELFTH_ROOTS.index(z)
        n = int(np.gcd(k, 12))
        return {0: "1", 3: "i", 6: "-1", 9: "-i"}.get(k, f"e^(2πi·{k // n}/{12 // n})")
    return f"{z:.6g}"


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


def _reversal_scalars(basis, blocks, lifts, classes, ref_chan, *, tol=_IN_BLOCKS):
    r"""The scalar time reversal carries on each orbit of multiplets, as its solution space pins it.

    Time reversal is assembled as its tensor structure: its own internal factor
    on every multiplet, times one scalar per orbit of multiplets.  Where the
    orbitals are real functions every scalar is one.  A basis function that is
    ``i`` times a real one turns its scalar to ``-1`` instead, and the
    Hamiltonian pins that sign wherever such an orbital couples to the
    reference: spinor states read spinless, whose time reversal of square
    ``+1`` the model's commutant supplies, are written that way, the states of
    one ``(p_x, p_y)`` spin sector coupled by an imaginary spin-orbit term.  So
    the scalars are read off the solution space and not assumed.  Its members
    in tensor form (:func:`_ansatz_residual`, one scalar per block) carry a
    vector of scalars on each orbit; two orbits whose vectors are parallel are
    pinned to one another at their ratio, an orbit pinned to the reference's
    takes its ratio to it, and a set of orbits the hoppings leave free of the
    reference is anchored at one on its lowest orbit -- rephasing orbitals the
    Hamiltonian does not couple to the rest changes nothing it measures.

    ``tol`` is how far from tensor form a member may be and still count as in
    it, relative to the space's own scale.  Returns one unit scalar per orbit,
    or ``None`` where the space holds no member in tensor form, or the scalars
    it pins are not of unit modulus -- where time reversal keeps the scalar one
    on every orbit.
    """
    B = np.asarray(basis, dtype=complex)
    if B.ndim != 3 or not len(B):
        return None
    A = np.stack([_ansatz_residual(B[j], blocks, lifts).reshape(-1)
                  for j in range(B.shape[0])], axis=1)
    _, s, vh = np.linalg.svd(A, full_matrices=False)
    top = float(s[0]) if s.size else 0.0
    inside = [k for k in range(len(s)) if float(s[k]) <= tol * max(top, 1.0)]
    if not inside:
        return None
    members = [np.einsum("j,jab->ab", vh[k].conj(), B) for k in inside]
    classes = np.asarray(classes, dtype=int)
    ncls = int(classes.max()) + 1 if classes.size else 0
    ref_cls = int(classes[int(ref_chan)])
    vec = {}
    for cls in range(ncls):
        c0 = int(ref_chan) if cls == ref_cls else int(np.flatnonzero(classes == cls)[0])
        rows, cols = blocks[c0]
        vec[cls] = np.asarray([_proportionality(M[np.ix_(rows, cols)], lifts[c0])[0]
                               for M in members])
        if float(np.linalg.norm(vec[cls])) <= tol:
            return None                  # no member carries this orbit at all
    # which orbits the space pins to one another: parallel vectors of scalars,
    # read within the space's own noise; and a sign read within that noise is
    # that sign exactly, so that the matrices assembled from it carry none of it
    snap = max(SNAP, 10.0 * float(tol))
    anchor = {}
    for cls in [ref_cls] + [c for c in range(ncls) if c != ref_cls]:
        if cls in anchor:
            continue
        anchor[cls] = cls
        for other in range(ncls):
            if other in anchor:
                continue
            a, b = vec[cls], vec[other]
            if abs(complex(np.vdot(a, b))) >= (1.0 - snap) * float(
                    np.linalg.norm(a) * np.linalg.norm(b)):
                anchor[other] = cls
    out = np.ones(ncls, dtype=complex)
    for cls in range(ncls):
        a = vec[anchor[cls]]
        t = complex(np.vdot(a, vec[cls]) / np.vdot(a, a))
        if abs(abs(t) - 1.0) > snap:
            return None
        sign = 1.0 if t.real >= 0.0 else -1.0
        out[cls] = sign if abs(t - sign) <= snap else t / abs(t)
    return out


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


def _induced_blocks(model, group, assignment, spin, validated, layout, kept, cperm, bases, *,
                    ref_chan=None, sectors=None, max_candidates=2 ** 20):
    r"""Every orbit of sectors built by induction that the sectors read, each as a record.

    The sectors are read where the model entered (``sectors``, read here where
    none is handed in); where the multiplets hold each sector's copies
    together, every orbit of sectors under the operations ``kept`` that holds
    identical copies, or that uncoupled orbitals a symmetry exchanges make one
    the group's one-dimensional characters do not carry
    (:func:`.._sectors.induced_orbits`), is read (:class:`.._sectors.InducedOrbit`):
    its candidates counted before anything is built, each a representation of
    the orbit the hoppings leave open, the first built and checked against the
    group's own table and every operation's solution space.  On states that
    carry a spin the orbits are read on the orbital factor, each matrix
    carrying the operation's own lift on the spin, so identical copies there
    carry an ordinary representation, the double-valued factor system the
    spin's alone.  An orbit the sectors do not read, an orbit of one copy with
    no representation, or one whose first candidate misses those checks, is
    left to the multiplets' own blocks.  Returns the records (:meth:`.._sectors.InducedOrbit.record`), the
    orbit itself under ``"induced"``, or the ``rep_search_too_large`` refusal
    where the candidates exceed ``max_candidates``.
    """
    from . import _sectors

    eps = getattr(layout, "_tol", None) or None

    def _induced(reading):
        return (reading is not None
                and (any(int(s.copies) > 1 for s in reading.sectors)
                     or any(len(h) > 1 for h in _sectors.held_sectors(reading, layout, tol=eps))
                     or bool(_sectors.crossed(reading, layout, group, kept, cperm, tol=eps))))

    reading = (sectors if sectors is not None else
               _sectors.read_sectors(model, group, assignment, spin, validated, tol=eps))
    if not _induced(reading):
        return []
    if sectors is not None and reading.tol is not None:
        # the frames the matrices are built in are the conserved matrices' of
        # THIS model: read where a precision was declared, the model handed in
        # may have been projected since, and its sectors are then exact
        reading = _sectors.read_sectors(model, group, assignment, spin, validated, tol=eps)
        if not _induced(reading):
            return []
    if _sectors.carried_reading(reading, layout, tol=eps) is not reading:
        return []                          # the multiplets keep the copies apart
    action = _sectors.sector_action(reading, model, group, assignment, spin, validated,
                                    tol=eps)
    try:
        orbits = _sectors.induced_orbits(reading, action, group, layout, kept, cperm,
                                         reference=ref_chan, tol=eps,
                                         max_candidates=max_candidates)
    except _sectors._TooMany as many:
        if many.sets is not None:
            return Refusal(
                "rep_search_too_large",
                f"the multiplets of orbitals {many.orbitals} hold {many.sets} sets of states "
                "the Hamiltonian conserves apart and no operation exchanges, and which "
                f"representation each carries is not fixed by it: the {many.count:,} ways of "
                f"choosing one for each are more than max_candidates = "
                f"{int(max_candidates):,}. Declare what the orbitals are (`characters=`), or "
                "raise `max_candidates=` if the enumeration is worth waiting for.",
                value=float(many.count), threshold=float(max_candidates),
                context={"multiplet": tuple(many.orbitals), "sets": int(many.sets),
                         "n_candidates": int(many.count),
                         "max_candidates": int(max_candidates)})
        return Refusal(
            "rep_search_too_large",
            f"the Hamiltonian conserves a {_fold(many.copies)} pseudospin on orbitals "
            f"{many.orbitals}, and which representation its copies carry is not fixed by "
            f"it: the {many.count:,} sums of irreducible representations of dimension "
            f"{many.copies} that could carry it are more than max_candidates = "
            f"{int(max_candidates):,}. Declare what the orbitals are (`characters=`), or "
            "raise `max_candidates=` if the enumeration is worth waiting for.",
            value=float(many.count), threshold=float(max_candidates),
            context={"multiplet": tuple(many.orbitals), "multiplicity": int(many.copies),
                     "n_candidates": int(many.count), "max_candidates": int(max_candidates)})
    gate = noise_floor(ASSEMBLED, eps)
    out = []
    for orbit in orbits:
        if orbit is None:
            continue
        record = orbit.record()
        if not record["n_admitted"] and int(record["dim"]) < 2:
            continue
        if record["n_admitted"] and orbit.check(record["candidates"][0]["W"], bases) > gate:
            continue
        out.append(record)
    return out


def _orbit_blocks(basis, image, chan_states, c0, members, Y, *, identity=False, tol=ZERO):
    r"""``{c: block}`` — one operation's block at every multiplet of a Wyckoff orbit.

    The block at the reference multiplet ``c0`` is ``Y``, a block of the
    operation's solution space at ``(image[c0], c0)``; the block at each other
    multiplet ``c`` is read at ``(image[c], c)`` off the **same member** of that
    space — the one whose reference block is ``Y``.  An intertwiner is one
    matrix, so its blocks at the other sites of the orbit are the operation
    acting in *their* frames: where the orbit permutes the orbital frame they
    are different matrices, the reference's frame carried round the orbit, and
    where every site carries the same functions they are the reference's block
    again.  Where the space is one-dimensional this is the model's own
    intertwiner read at every multiplet, up to one scalar for the operation.

    A multiplet whose block the reference's does not determine — one the
    Hamiltonian decouples from it, so that the member read there vanishes —
    takes the leading block the space carries there.  ``identity`` says the
    operation is the identity, whose reference block ``Y`` is the identity
    matrix in the phase the rest of the build is written in: it is that block
    at every multiplet, since ``D(E) = 1`` is what a representation means,
    whatever conserved operator the identity's solution space holds.  A block
    below ``tol`` of the reference's scale vanishes: :data:`.._tolerance.ZERO`,
    or the declared tolerance the space was measured at where that is larger.
    """
    c0 = int(c0)
    if identity:
        return {int(c): Y for c in members}
    out = {c0: Y}
    others = [int(c) for c in members if int(c) != c0]
    if not others:
        return out
    B = np.asarray(basis)
    r0 = np.asarray(chan_states[int(image[c0])], dtype=int)
    k0 = np.asarray(chan_states[c0], dtype=int)
    A = B[:, r0][:, :, k0].reshape(B.shape[0], -1).T
    coef = np.linalg.lstsq(A, np.asarray(Y, dtype=complex).reshape(-1), rcond=None)[0]
    floor = tol * max(float(np.abs(Y).max()), 1.0)
    for c in others:
        rows = np.asarray(chan_states[int(image[c])], dtype=int)
        cols = np.asarray(chan_states[c], dtype=int)
        blk = B[:, rows][:, :, cols]
        M = np.einsum("j,jab->ab", coef, blk)
        if float(np.abs(M).max()) <= floor:
            lead, _scale = _block_space(B, rows, cols, tol=tol)
            M = lead[0] if lead else M
        U = _unitarized(M, tol=tol)
        out[c] = U if U is not None else np.asarray(M, dtype=complex)
    return out


#: The signs a one-dimensional multiplet's transport between two sites of its
#: orbit is taken exactly at, where the ratio read off a solution space is
#: within :data:`.._tolerance.SNAP` of one, or within the declared tolerance
#: where that is larger: the orbitals written as one another's images, up to a
#: sign or a quarter turn of phase.
_TRANSPORT_SIGNS = (1.0 + 0.0j, -1.0 + 0.0j, 1.0j, -1.0j)


def _transported_lifts(syms, kept, cperm, layout, members, c0, bases, intertwiners,
                       prod, ident, *, tol=SPREAD):
    r"""``{p: {c: S}}`` — a class of one-dimensional multiplets, carried round its orbit.

    A ``p`` along each site's own fourfold axis, the axes pointing along
    different cube edges, is a one-dimensional multiplet at every site, and an
    operation carrying one site onto the next carries it onto plus or minus the
    one there: the relative signs inside the orbit are what the model's
    transport says they are.  They are read here off
    **one member** of each operation's solution space -- its block at every
    multiplet of the class relative to its block at the reference multiplet,
    which keeps the tables' own lift -- and only where the space pins them:
    every member of a space with more than one gives the same ratio.

    What is read is taken only where it is a representation, since a sign per
    site that is not one is a gauge the hoppings carry and not a symmetry of
    the crystal (a flux threading the cell): the operations read must be closed
    under the group's product, and any two that commute must carry every
    multiplet of the class the same way in either order.  Otherwise, or where
    nothing is read at all, ``None``, and every multiplet keeps the tables' lift.

    Every threshold here measures the noise of the solution spaces, so each
    reads the declared tolerance the multiplets were measured at where that is
    larger than its exact default (:func:`.._tolerance.noise_floor`): a
    numerical model's transported sign misses the exact one by its own noise.
    """
    su2 = layout._spinful
    eps = getattr(layout, "_tol", None)
    tol = noise_floor(tol, eps)
    zero, snap = noise_floor(ZERO, eps), noise_floor(SNAP, eps)
    rel = {}
    for p in kept:
        ind = int(syms[p].ind)
        if p == ident:
            rel[p] = {int(c): 1.0 + 0.0j for c in members}
            continue
        basis = bases.get(ind)
        if basis is None or len(np.asarray(basis)) == 0:
            U = intertwiners.get(ind)
            basis = None if U is None else np.asarray(U)[None, :, :]
        if basis is None:
            continue
        B = np.asarray(basis)
        lift = _lift_of(syms[p], su2)
        floor = zero * max(float(np.abs(B).max()), 1.0)
        scalars = {}
        for c in members:
            rows = np.asarray(layout._chan_states[int(cperm[p][int(c)])], dtype=int)
            cols = np.asarray(layout._chan_states[int(c)], dtype=int)
            v = []
            for j in range(B.shape[0]):
                blk = B[j][np.ix_(rows, cols)]
                rho, res = _proportionality(blk, lift)
                if res > tol * max(float(np.abs(blk).max()), floor):
                    return None           # not the lift times a number: no such class
                v.append(complex(rho))
            scalars[int(c)] = np.asarray(v)
        s0 = scalars[int(c0)]
        n0 = float(np.vdot(s0, s0).real)
        if n0 <= floor ** 2:
            continue                      # the space does not carry the reference here
        got = {}
        for c in members:
            sc = scalars[int(c)]
            sig = complex(np.vdot(s0, sc) / n0)
            spread = float(np.abs(sc - sig * s0).max())
            if abs(sig) < snap or spread > tol * max(float(np.abs(sc).max()),
                                                     float(np.abs(s0).max())):
                got = None                # not pinned: the hoppings leave it open
                break
            # the sign an orbit's sites carry is discrete wherever the orbitals
            # are written as one another's images, and is taken exactly there:
            # the ratio read off a solution space carries its rounding, and a
            # numerical model's its noise.  A phase no such sign is near is a
            # gauge of the orbitals' own, and is kept as measured.
            phase = sig / abs(sig)
            near = min(_TRANSPORT_SIGNS, key=lambda w: abs(phase - w))
            got[int(c)] = near if abs(phase - near) <= snap else phase
        if got is not None:
            rel[p] = got
    unitary = [p for p in kept if not syms[p].time_reversal and p in rel]
    if len(unitary) < 2 or any(int(prod[a, b]) not in rel for a in unitary for b in unitary):
        return None
    for i, a in enumerate(unitary):
        for b in unitary[i + 1:]:
            if int(prod[a, b]) != int(prod[b, a]):
                continue
            for c in members:
                ab = rel[a][int(cperm[b][int(c)])] * rel[b][int(c)]
                ba = rel[b][int(cperm[a][int(c)])] * rel[a][int(c)]
                if abs(ab - ba) > snap:
                    return None           # a projective sign: a gauge, not a symmetry
    out = {}
    for p, got in rel.items():
        lift = _lift_of(syms[p], su2)
        out[p] = {int(c): (got[int(c)] * lift).astype(complex) for c in members}
    return out


def _phase_multiplet(ones, c0, chan_states, commutant, *, tol=SPREAD):
    """The one-dimensional multiplet a measured multiplet's blocks are phased against, or ``None``.

    The first of the one-dimensional multiplets ``ones`` the hoppings couple to
    multiplet ``c0``, directly or through others.  It is read off
    ``commutant``, the identity's solution space -- every matrix commuting with
    ``H`` -- whose members carry one scalar per coupled part of the crystal:
    two multiplets of one part carry proportional scalars on every member, and
    a multiplet the hoppings never reach carries a scalar of its own.  Where
    that space is not at hand the first of ``ones`` is taken.
    """
    if not ones:
        return None
    if commutant is None or len(np.asarray(commutant)) == 0:
        return int(ones[0])
    B = np.asarray(commutant, dtype=complex)

    def _scalar(c):
        st = np.asarray(chan_states[int(c)], dtype=int)
        return np.asarray([complex(np.trace(B[j][np.ix_(st, st)])) / len(st)
                           for j in range(B.shape[0])])

    t = _scalar(c0)
    for o in ones:
        a = _scalar(o)
        base = float(np.abs(a).max()) if a.size else 0.0
        if base <= tol:
            continue
        sig = complex(np.vdot(a, t) / np.vdot(a, a))
        if abs(sig) > tol and float(np.abs(t - sig * a).max()) <= tol * max(
                float(np.abs(t).max()), base, 1.0):
            return int(o)
    return None


def _multiplet_blocks(model, group, assignment, spin, validated, layout, positions, *,
                      ref_chan=None, tol=ZERO, max_candidates=2 ** 20, sectors=None):
    r"""``{ind: {channel: S}}`` — the internal block of every multiplet, the model's own.

    A one-dimensional multiplet's internal factor is the tables' own lift, times
    the sign the model's own transport gives it relative to its class's
    reference multiplet (:func:`_transported_lifts`): what an operation does to
    the ``p`` along a site's own fourfold axis is as much a measurement as what
    it does to a doublet, and it is the relative signs inside the orbit that
    carry it.

    A wider multiplet's block is the **model's own**, whatever basis its
    orbitals are written in, carried round the Wyckoff orbit.  The reference
    multiplet's block is the block of an intertwiner at ``(image channel,
    channel)``, read off the whole solution space rather than off the
    polar-extracted member alone, whose block at the operation's own image can
    vanish while the space still carries one; every other multiplet of the
    class takes its block from the same member of that space
    (:func:`_orbit_blocks`), which is the operation acting in that site's own
    frame.  So the representation of the orbit is the one induced from the
    reference site's in the model's own frames, and where every solution space
    is one-dimensional it is the model's own intertwiner read at every
    multiplet.  Nothing about which orbital functions a multiplet is enters:
    the site irreps are read off these blocks' characters afterwards
    (:func:`_multiplet_characters`), and a basis turned inside a site changes
    the blocks and not their characters.

    Where a space is more than one-dimensional the extra members are the
    model's commutant acting inside the multiplet, and where the sectors of the
    conserved matrices read it the block is built by induction (below).  A
    multiplet they leave alone keeps the block its own intertwiner gives,
    carried round the orbit, and :func:`_build_rep` asks of the matrices it
    propagates to, as of every other, whether they represent the group.

    Where the sectors of the conserved matrices (``sectors``,
    :func:`.._sectors.read_sectors`, read here where none is handed in) hold
    identical copies -- a pseudospin ``H`` conserves -- the hoppings do not
    settle what the copies carry; and where uncoupled orbitals a symmetry
    exchanges -- the ``p_x`` of one site and the ``p_y`` a screw carries it
    onto -- lie on sectors whose stabilizer has characters the group's do not
    restrict to, no one-dimensional character of the group carries what they
    can be.  Every representation of each such orbit of sectors the hoppings
    leave open is built from the reading, induced from the stabilizer of one of
    its sectors (:class:`.._sectors.InducedOrbit`), and the orbit's multiplets
    are written in the first; which one they carry is the labelling's choice.
    The sectors are what says whether a multiplet's conserved operators are
    copies or projectors.
    States carrying a spin, native or declared, carry that spin's own lift, as
    a one-dimensional multiplet carries the tables': every solution space read
    here was read in its operation's own form (:func:`_own_form_spaces`), its
    members ``O ⊗ S(g)`` with ``S(g)`` the operation's own lift and ``O`` the
    orbital factor ``H`` fixes, so the spin is never chosen, and an orbit of
    sectors there is built on the orbital factor, identical copies carrying an
    ordinary representation.

    Returns ``{ind: {channel: S}}`` over the multiplets a block was built for,
    with the orbits built by induction on its ``multiplicity``, or a
    :class:`..refusal.Refusal` from the multiplet permutation or from the cap
    on the representations the induction tries.  Every threshold here
    measures the noise of the solution spaces the blocks are read off, so each
    reads the declared tolerance the multiplets were measured at where that is
    larger than its exact default (:attr:`SiteMultiplets._tol`,
    :func:`.._tolerance.noise_floor`).
    """
    syms = group.symmetries
    prod = group.get_product_table(get_diff=False)
    ident = _identity_position(group)
    kept = list(positions)
    eps = getattr(layout, "_tol", None)
    tol, zero = noise_floor(tol, eps), noise_floor(ZERO, eps)
    u_pos = [p for p in kept if not syms[p].time_reversal]
    cperm = _channel_perms(model, assignment, syms, kept, layout, spin)
    if isinstance(cperm, Refusal):
        return cperm
    classes, ncls = _channel_classes(cperm, u_pos, layout._nchan)
    ones = [c for c in range(int(layout._nchan)) if layout._chan_kind[c] == "tables"]
    if ref_chan is None:
        ref_chan = int(np.asarray(layout._chan_of, dtype=int)[0])
    bases = getattr(validated, "_bases", None) or {}

    out, multiplicity = {}, []
    commutant = bases.get(int(syms[ident].ind))
    # IDENTICAL COPIES, and uncoupled orbitals a symmetry exchanges, read once
    # where the model entered: every representation of each orbit of sectors
    # the one-dimensional characters of the group do not carry that the
    # hoppings leave open, built from the reading and counted before anything
    # is built; the orbit's multiplets are written in the first, and their
    # classes are read no further here.  On states that carry a spin the
    # orbits are read on the orbital factor, each matrix carrying the
    # operation's own lift on the spin
    done = set()
    got = _induced_blocks(model, group, assignment, spin, validated, layout, kept, cperm,
                          bases, ref_chan=ref_chan, sectors=sectors,
                          max_candidates=max_candidates)
    if isinstance(got, Refusal):
        return got
    for record in got:
        multiplicity.append(record)
        if not record["n_admitted"]:
            continue
        W = record["candidates"][0]["W"]
        local = {int(a): i for i, a in enumerate(record["states"])}
        for p in kept:
            for c in record["multiplets"]:
                rows = [local[int(a)] for a in layout._chan_states[int(cperm[p][c])]]
                cols = [local[int(a)] for a in layout._chan_states[int(c)]]
                out.setdefault(int(syms[p].ind), {})[int(c)] = W[p][np.ix_(rows, cols)]
        record["candidates"][0].release()
        done |= {int(classes[int(c)]) for c in record["multiplets"]}
    for cls in range(ncls):
        if cls in done:
            continue
        members = [c for c in range(int(layout._nchan)) if int(classes[c]) == cls]
        c0 = ref_chan if int(classes[ref_chan]) == cls else members[0]
        d = len(layout._chan_states[c0])
        if layout._chan_kind[members[0]] != "measured":
            # A ONE-DIMENSIONAL CLASS keeps the tables' lift at its reference
            # multiplet, and at every other multiplet the sign the model's own
            # transport gives it, read off one member of each solution space --
            # where that is a representation, and the lift everywhere where it
            # is not.
            if len(members) > 1:
                got = _transported_lifts(syms, kept, cperm, layout, members, c0, bases,
                                         validated._intertwiners, prod, ident)
                for p, per in (got or {}).items():
                    for c, S in per.items():
                        out.setdefault(int(syms[p].ind), {})[int(c)] = S
            continue
        one = _phase_multiplet(ones, c0, layout._chan_states, commutant,
                               tol=noise_floor(SPREAD, eps))
        space, polar, basis_of = {}, {}, {}
        for p in kept:
            ind = int(syms[p].ind)
            rows = layout._chan_states[int(cperm[p][c0])]
            cols = layout._chan_states[c0]
            got = validated._intertwiners.get(ind)
            basis = bases.get(ind)
            if basis is None or len(np.asarray(basis)) == 0:
                U = validated._intertwiners.get(ind)
                basis = None if U is None else np.asarray(U)[None, :, :]
            if basis is not None:
                basis_of[p] = np.asarray(basis)
                space[p], _scale = _block_space(basis, rows, cols, tol=tol)
            blk = (None if got is None else
                   np.asarray(got)[np.ix_(np.asarray(rows, dtype=int),
                                          np.asarray(cols, dtype=int))])
            if blk is None or float(np.abs(blk).max()) < zero:
                blk = space.get(p, [None])[0] if space.get(p) else None
            unit = None if blk is None else _unitarized(blk, tol=zero)
            if p == ident:
                # THE IDENTITY'S BLOCK IS THE IDENTITY.  Its solution space is
                # the model's whole commutant, so a polar extraction there
                # returns whichever conserved operator the decomposition
                # happened to hand back -- `-1` on a decoupled doublet -- and
                # that is not a measurement about the operation at all:
                # `D(E) = 1` is what a representation means.
                unit = np.eye(d, dtype=complex)
            if unit is not None and unit.shape == (d, d):
                # THE PHASE A MEASURED BLOCK IS READ IN.  The member this block
                # came from carries one free phase, and a block extracted with
                # it makes the *other* multiplets' scalars carry its inverse --
                # which the read-off then has to find among the two signs the
                # tensor structure offers and cannot.  Where the hoppings couple
                # the multiplet to a one-dimensional one there is a phase
                # convention to hand: the tables' own lift.  So the block is put
                # in it, and the scalars the rest of the build reads are the
                # model's own.  A one-dimensional multiplet the hoppings never
                # reach fixes nothing about this one's phase.
                if one is not None:
                    a_rows = layout._chan_states[int(cperm[p][one])]
                    a_cols = layout._chan_states[one]
                    got_u = got
                    if got_u is not None:
                        alpha, _res = _proportionality(
                            np.asarray(got_u)[np.ix_(np.asarray(a_rows, dtype=int),
                                                     np.asarray(a_cols, dtype=int))],
                            _lift_of(syms[p], layout._spinful))
                        if abs(alpha) > zero:
                            unit = unit * np.conj(alpha) / abs(alpha)
                polar[p] = unit

        def _carried(p, Y):
            """``{c: block}`` -- the member of ``p``'s space with block ``Y`` at ``c0``."""
            return _orbit_blocks(basis_of.get(p), cperm[p], layout._chan_states, c0,
                                 members, Y, identity=(p == ident), tol=tol)

        # CARRIED ROUND THE ORBIT, NEVER COPIED.  The sites of one Wyckoff
        # orbit are one another's images, so the block an operation has at the
        # next site is the operation acting in THAT site's frame, which a copy
        # of the reference's block is only where every site writes its orbitals
        # in the same frame.  The member of the solution space the reference
        # block came from says it at every multiplet at once.
        for p, S in polar.items():
            got = _carried(p, S)
            for c in members:
                out.setdefault(int(syms[p].ind), {})[int(c)] = got[int(c)]
    return _MultipletBlocks(out, multiplicity=multiplicity)


class _MultipletBlocks(dict):
    """``{ind: {multiplet: block}}``, and the orbits of sectors built by induction on the way.

    ``multiplicity`` holds one record per orbit of sectors built by induction
    (:func:`_induced_blocks`): the orbit's orbitals and states, the number of
    identical copies it holds -- one where it holds none -- how many
    representations of it there were and were admitted, and every admitted one
    with the site content it gives each multiplet and its own character.
    """

    def __init__(self, blocks=(), *, multiplicity=()):
        super().__init__(blocks)
        self.multiplicity = tuple(multiplicity)


def _ansatz_defects(model, group, assignment, validated, positions, *,
                    layout=None, spin=None, tol=None, orients=(1, -1),
                    blocks=None, max_candidates=2 ** 20):
    r"""Per-operation: does the detector's solution space contain an ``O ⊗ S``?

    The detector answers about a **k-action class**, not about an operation:
    two operations acting identically on reduced ``k`` pose the identical
    intertwiner problem and share one verdict, and on a slab every class is the
    pair ``{g, σ_h·g}`` (:attr:`..groups.ValidatedGroup.verdicts`).  A surviving
    multi-member class therefore proves only that *some* operator with that
    k-action is a symmetry, and which member it is, is a statement about the
    **tensor structure** of the representation rather than about ``H(k)``'s
    k-dependence: ``D(g) = O(g) ⊗ S(g)`` with ``O`` fixed by the site
    permutation and, on a one-dimensional multiplet, ``S`` the tables' own lift,
    and ``S(σ_h·g)`` is a different SU(2) element from ``S(g)``.

    This measures exactly that, per operation, with the arithmetic
    :func:`_build_rep`'s read-off already uses on a generator: project the
    detector's whole solution space onto the tensor-structure subspace of that
    operation's own blocks and take the closest member (:func:`_ansatz_member`).
    The reported ``scaled`` defect is the norm of what that member could **not**
    put in tensor-structure form, divided by the member's own norm —
    dimensionless, ``0`` when the tensor structure is realizable and ``1`` when
    its component vanishes outright.
    The branches ``orients`` names are measured -- the ``g⁻¹`` branch's
    candidate is the adjoint and carries ``S(g)†`` on the transposed blocks --
    and an operation is realizable when *some* branch measured is, and
    ``orientation`` says which.

    ``positions`` are indices into ``group.symmetries``.  ``S(g)`` on a
    one-dimensional multiplet is that operation's own lift in pythtb's
    convention (:func:`_lift_of`) — there is no override of the lift, because
    an operation represented by some other SU(2) element is a different
    operation — carried round the orbit with the signs the model's own
    transport gives it (:func:`_transported_lifts`); on a wider multiplet it is
    the model's own block for **that** multiplet (:func:`_multiplet_blocks`),
    so that the tensor test asks exactly whether ``P(g) ⊗ S(g)`` is in the
    space the detector solved, and asks it of the frame a Wyckoff orbit
    actually has rather than of a copy of one site's.  A wider multiplet's
    block is read off a solution space both members of a ``σ_h`` fiber share,
    so there it tells them apart by nothing: only a one-dimensional multiplet's
    lift does.  So each operation's own form -- each site's states carried onto
    its image's, with its own lift on the spin -- is read on its whole solution
    space first (:func:`_own_form_spaces`), which tells such a pair apart on a
    multiplet of any dimension, and the spaces this test reads are the members
    of that form.  ``layout`` holds the multiplets to test against, measured
    here when omitted.

    ``tol`` is the gate on the scaled defect: :data:`ANSATZ_GATE` for a model
    built from exact numbers, or the declared tolerance the multiplets were
    measured at where that is larger (:func:`.._tolerance.noise_floor`) -- a model symmetric
    only within its precision has every member of a solution space that far
    from tensor form, while an operation the tensor structure refutes misses by
    order one.  ``None`` reads it off ``layout``.

    ``orients`` says which branches are measured, both by default.  The
    verdict on an operation the detector left undetermined asks the ``g``
    branch alone, ``(1,)``, in every environment
    (:func:`..groups._ansatz_refuted`): on the ``g⁻¹`` branch the candidate
    carries ``S(g)†``, which for a fourfold **is** the ``σ_h`` partner's lift
    (``S(C4)† = −S(σ_h)·S(C4)``), so asking it would keep an operation on the
    strength of a lift the tables do not give it.  Which operations the
    multiplets are about is asked on both.  Returns
    ``{ind: record}`` with ``dim``, ``defect``, ``scale``, ``norm``, ``scaled``,
    ``orientation``, ``branches`` and ``realizable``.  An operation the
    detector kept no solution space for is reported with ``scaled = nan`` and
    ``realizable = True``: nothing was measured, so nothing is refuted.
    ``spin`` is the declared :class:`..spin.SpinBasis` itself, or ``None``.
    """
    if layout is None:
        layout = _site_multiplets(model, group, assignment, spin, validated)
    eps = getattr(layout, "_tol", None)
    if tol is None:
        tol = noise_floor(ANSATZ_GATE, eps)
    syms = group.symmetries
    positions = list(positions)
    cperm = _channel_perms(model, assignment, syms, positions, layout, spin)
    if isinstance(cperm, Refusal):
        return cperm
    chan_states, su2 = layout._chan_states, layout._spinful
    ref_chan = int(np.asarray(layout._chan_of, dtype=int)[0])
    # the multiplet orbits of the operations under test, as _build_rep forms them:
    # the tensor structure's freedom is one scalar per orbit, not one per multiplet
    nchan = int(layout._nchan)
    u_pos = [p for p in positions if not bool(syms[p].time_reversal)]
    cls, ncls = _channel_classes(cperm, u_pos, nchan)
    class_blocks = [[c for c in range(nchan) if cls[c] == k] for k in range(ncls)]
    # the multiplets' blocks: the same ones the representation is assembled
    # from, so that the tensor test and the build rest on one tensor structure
    # rather than two.  Measured once per set of multiplets and per round, and
    # handed in where the call already has it: the block IS the same
    # measurement, and re-solving it per test is the one cost this package
    # cannot afford on a wannier90 Hamiltonian.
    got = blocks if blocks is not None else _multiplet_blocks(
        model, group, assignment, spin, validated, layout,
        [p for p, s in enumerate(syms) if int(s.ind) in set(validated.indices)],
        ref_chan=ref_chan, max_candidates=max_candidates)
    if isinstance(got, Refusal):
        return got
    blocks_of = got

    out = {}
    for p in positions:
        sym = syms[p]
        ind = int(sym.ind)
        rec = {"ind": ind, "dim": 0, "defect": float("nan"), "scale": float("nan"),
               "norm": float("nan"), "scaled": float("nan"), "orientation": 0,
               "branches": {}, "realizable": True, "note": None}
        best = None
        basis = getattr(validated, "_bases", {}).get(ind)
        if basis is None or len(np.asarray(basis)) == 0:
            member = validated._intertwiners.get(ind)
            if member is None:
                rec["note"] = ("no solution of this operation's condition is carried, "
                               "so its tensor structure was not tested")
                out[ind] = rec
                continue
            # the polar-extracted member IS the space when dim == 1, and where it
            # is not, the projection has nothing to choose from and the defect
            # says so -- a fallback, not a wrong verdict (as in the read-off).
            basis = np.asarray(member)[None, :, :]
            rec["note"] = ("only the unitary member nearest the solution space was "
                           "tested, not the whole space")
        basis = np.asarray(basis)
        rec["dim"] = int(basis.shape[0])
        nstate = int(basis.shape[1])
        if basis.shape[0] * nstate * nstate * 16 > _READOFF_MAX_BYTES:
            # the same ceiling the read-off applies: a projection that would
            # allocate more than this is not run, and an untested operation is
            # reported as such rather than refuted on a measurement not made.
            rec["note"] = (f"the solution space is {basis.shape[0]}-dimensional on "
                           f"{nstate} states, too large to test its tensor structure "
                           f"within {_READOFF_MAX_BYTES / 1e6:.0f} MB")
            out[ind] = rec
            continue
        lift = _lift_of(sym, su2)
        zero = None
        for c in range(nchan):
            if layout._chan_kind[c] != "measured":
                continue
            rows = np.asarray(chan_states[int(cperm[p][c])], dtype=int)
            cols = np.asarray(chan_states[c], dtype=int)
            if float(np.abs(basis[:, rows][:, :, cols]).max()) < noise_floor(ZERO, eps):
                zero = int(c)
                break
        if zero is not None:
            # No member of the space sends this multiplet where the operation
            # must -- the case a LATTICE SUPERGROUP puts every coset partner
            # in, where the measured operator is the partner's and never this
            # operation's -- so the tensor structure is refuted rather than left
            # untested.
            rec.update(defect=1.0, scale=1.0, norm=1.0, scaled=1.0,
                       orientation=0, realizable=False,
                       note=(f"no member of the measured solution space carries multiplet "
                             f"{zero} to the multiplet the operation carries it to: every "
                             f"member's block there is zero"))
            out[ind] = rec
            continue
        blocks_here = blocks_of.get(ind, {})
        if layout._measured and not blocks_here:
            # Where no block could be read off the model for a multi-dimensional
            # multiplet there is nothing to test the tensor structure against:
            # reported, never refuted on it.
            rec["note"] = ("no internal block could be read for this operation on a "
                           "multi-dimensional multiplet")
            out[ind] = rec
            continue
        S_of_chan = [blocks_here.get(c, None if layout._chan_kind[c] == "measured" else lift)
                     for c in range(nchan)]
        if any(S is None for S in S_of_chan):  # pragma: no cover - a guard, not a path
            rec["note"] = "no internal block of the right size could be read"
            out[ind] = rec
            continue
        for orient in tuple(orients):
            lifts = [S.conj().T if orient < 0 else S for S in S_of_chan]
            blocks = _ansatz_blocks(chan_states, cperm[p], orient)
            try:
                X, defect, scale = _ansatz_member(basis, blocks, lifts, class_blocks)
            except np.linalg.LinAlgError:      # pragma: no cover - not seen on a fixture
                rec["note"] = "the projection onto the tensor-structure form did not converge"
                continue
            norm = float(np.linalg.norm(X))
            scaled = float(defect / norm) if norm > 0 else float("inf")
            rec["branches"][int(orient)] = scaled
            if best is None or scaled < best:
                best = scaled
                rec.update(defect=float(defect), scale=float(scale), norm=norm,
                           scaled=scaled, orientation=int(orient))
        if best is not None:
            rec["realizable"] = bool(best <= float(tol))
        out[ind] = rec
    return out


def _own_form_spaces(model, group, assignment, validated, positions, *, spin=None, tol=None,
                     covariance=None):
    r"""Each operation's solution space restricted to the members of its own form, and the verdict on it.

    The detector finds every matrix carrying ``H(k)`` onto ``H(gk)``, whichever
    operation of the crystal it belongs to: one action on ``k`` poses one
    problem, and where several operations pose it -- a translation and the
    identity, time reversal and time reversal composed with a translation, a
    layer's ``g`` and ``σ_h·g`` -- or where the Hamiltonian conserves more than
    the crystal does, a spin rotation among it, the space holds members of
    other operations and other symmetries.  A matrix represents ``g`` only in
    ``g``'s own form, asked of one member: it carries each site's states onto
    its image's under ``g``, with any change of basis within the site, and on
    states that carry a spin, native or declared, it carries ``g``'s own spin
    lift ``S(g)`` (:func:`_lift_of`) -- ``O ⊗ S(g)``, the orbital factor ``O``
    block by block on the site permutation.  Both conditions are linear, so the
    members of that form are read off the space once: a member is of it where
    what no matrix of the form carries is within :data:`ANSATZ_GATE` of the
    member's own norm, the defect and the gate every test of tensor structure
    reads, so that matrices missing the form by a model's own numerical noise
    are its matrices still.  The operation holds where one of them is
    nonsingular on every site block (:func:`_filled_check`).  Neither condition
    alone is enough: in a
    Néel antiferromagnet time reversal's space holds members keeping every site
    -- time reversal with a spin rotation the exchange field allows -- and
    members carrying ``iσ_y`` -- time reversal with the half translation,
    which exchanges the two sites -- and no member that does both.

    ``tol`` is the declared tolerance, relative to ``max|H|``, every threshold
    here reads (:func:`.._tolerance.noise_floor`); ``None`` keeps the exact
    defaults.  Where ``covariance`` is given, a form missed by more than
    rounding is also held to the detector's own measure: the member nearest
    the form, its part in the form, is held to the covariance gate
    ``validated.tol`` the operation was kept at (``covariance`` is called with
    the operation's position and that matrix, and returns its covariance
    residual over the detector's grid).  A gate on the form alone cannot tell
    a model's noise from a small angle: spin blocks written in a frame turned
    against the lattice by ``δ`` carry every operation whose lift the turn
    changes through a spin factor ``δ`` from its own, inside any such gate,
    and with its own lift they carry ``H(k)`` onto ``H(gk)`` only to what the
    turn does to ``H``.  So where members are of the form within the gate, the
    operation holds where that part is within the covariance gate; where it is
    not, the condition that failed is the spin lift where the member's part on
    the site blocks, with any spin factor, carries ``H(k)`` within it, and the
    site permutation otherwise.  Where no member is of the form within the
    gate, as a model symmetric only within its precision needs, the operation
    holds where that part is within the covariance gate and nonsingular on
    every site block -- a part the form leaves empty, as it leaves a
    ferrimagnet's spin-diagonal members under time reversal's ``iσ_y``, is no
    matrix of the operation.

    Returns ``(spaces, records)``.  ``spaces[ind]`` holds the members of the
    form -- the space itself where every member has it, the one member nearest
    it where the covariance kept an operation no member of which is of the
    form within the gate, none where the operation is refuted -- in the
    model's own state order; it is what every later reading of the operation
    reads.  ``records[ind]`` carries :func:`_ansatz_defects`'s fields
    (``dim``, ``defect``, ``scale``, ``norm``, ``scaled``, ``orientation``,
    ``branches``, ``realizable``, ``note``), the ``covariance`` residual where
    it was measured, and ``own_form``: ``None`` where the operation holds,
    else the condition that failed -- ``"spin lift"`` where members keeping
    every site exist and none carries the lift within the gates, ``"site
    permutation"`` otherwise, the members carrying the lift, if any, carrying
    the sites elsewhere -- with ``spin_lift`` marking the first.  An operation
    with no measured space, or a space past the read-off's own ceiling, is
    absent from both: nothing was measured, and it is read as it stands.
    """
    syms = group.symmetries
    nstate = int(model.nstate)
    spinful = ((int(model.nspin) == 2 or spin is not None)
               and bool(getattr(group, "spinor", True)))
    site_orbs = tuple(tuple(int(o) for o in orbs) for orbs in assignment.site_orbitals)
    sites = _multiplets_of(model, assignment, spin, tuple((orbs,) for orbs in site_orbs),
                           tol=tol)
    spinors = None
    if spinful:
        spinors = ([[(2 * o, 2 * o + 1) for o in orbs] for orbs in site_orbs]
                   if int(model.nspin) == 2 else
                   [[tuple(int(x) for x in a) for a in site]
                    for site in _atoms_of_site(site_orbs, spin)])
    in_blocks = noise_floor(_IN_BLOCKS, tol)
    form_gate = noise_floor(ANSATZ_GATE, tol)
    spaces, records = {}, {}
    for p in positions:
        sym = syms[p]
        ind = int(sym.ind)
        basis = (getattr(validated, "_bases", None) or {}).get(ind)
        if basis is None or not len(np.asarray(basis)):
            continue
        basis = np.asarray(basis, dtype=complex)
        if basis.shape[0] * nstate * nstate * 16 > _READOFF_MAX_BYTES:
            continue
        image = _channel_permutation(assignment, p, model, sites)
        blocks = _ansatz_blocks(sites._chan_states, image, 1)
        if spinful:
            S = _lift_of(sym, True)
            pieces = [(np.asarray(to, dtype=int), np.asarray(of, dtype=int))
                      for s in range(len(site_orbs)) for to in spinors[int(image[s])]
                      for of in spinors[s]]
            lifts = [S] * len(pieces)
        else:
            pieces, lifts = blocks, [None] * len(blocks)
        # the members of the form: what no member of it carries is the
        # residual, read once over the whole space, and a member is the form's
        # where its residual is within the gate of its own norm -- the defect
        # every test of tensor structure reads, at their gate, so that a model
        # whose matrices miss the form by its own numerical noise keeps them
        A = np.stack([_ansatz_residual(basis[j], pieces, lifts).reshape(-1)
                      for j in range(len(basis))], axis=1)
        _u, s, vh = np.linalg.svd(A, full_matrices=False)
        top = float(s[0]) if s.size else 0.0
        members = np.einsum("ij,jab->iab", vh.conj(), basis)
        norms = np.linalg.norm(members.reshape(len(members), -1), axis=1)
        keep = [i for i in range(len(s)) if float(s[i]) <= form_gate * float(norms[i])]
        if len(keep) == len(s):
            V = basis                         # every member is of the form
        else:
            V = members[keep] if keep else basis[:0]
        held = bool(len(V)) and _filled_check(V, blocks, tol=tol)[0]
        X = members[-1]
        norm = float(norms[-1])
        scaled = float(s[-1]) / norm if norm > 0 else float("inf")
        rec = {"ind": ind, "dim": int(len(V)) if held else 0, "defect": float(s[-1]),
               "scale": top, "norm": norm, "scaled": scaled, "orientation": 1,
               "branches": {1: scaled}, "realizable": held, "note": None,
               "own_form": None, "spin_lift": False}
        records[ind] = rec
        if held and covariance is not None and scaled > FLAT_ZERO:
            # of the form only within the gate: the member nearest it, its part
            # in the form, held to the covariance gate the operation was kept
            # at, as below where no member is within the gate -- spin blocks
            # written in a frame turned against the lattice by a small angle
            # miss the form within any gate on it, and carry H(k) onto H(gk)
            # with the operation's own lift only as far as the turn allows
            part = X - _ansatz_residual(X, pieces, lifts)
            residual = float(covariance(p, part))
            rec["covariance"] = residual
            if residual > float(validated.tol):
                # the condition that fails: the spin lift where the member's
                # part carrying each site's states onto its image's, with any
                # spin factor, carries H(k) within the gate, else the sites
                in_sites = X - _ansatz_residual(X, blocks, [None] * len(blocks))
                lift = spinful and float(covariance(p, in_sites)) <= float(validated.tol)
                rec.update(realizable=False, dim=0,
                           own_form="spin lift" if lift else "site permutation",
                           spin_lift=lift,
                           note=("its own form carries H(k) onto H(gk) only to "
                                 f"{residual:.3g}, beyond the {float(validated.tol):.3g} it was "
                                 "kept at"
                                 + ("; with a spin factor other than its own lift S(g), its "
                                    "matrices carrying each site's states onto its image's "
                                    "carry it within that" if lift else "")))
                spaces[ind] = basis[:0]
                continue
        if held:
            spaces[ind] = V
            continue
        if not len(V) and covariance is not None:
            # no member of the form within the gate: the member nearest it, its
            # part in the form, held to the covariance gate the operation was
            # kept at -- where that part is a matrix of the operation at all,
            # nonsingular on every site block; a part the form leaves empty or
            # singular is no member of it, whatever its nearest unitary carries
            part = X - _ansatz_residual(X, pieces, lifts)
            if _filled_check(part[None], blocks, tol=tol)[0]:
                residual = float(covariance(p, part))
                rec["covariance"] = residual
                if residual <= float(validated.tol):
                    rec.update(realizable=True, dim=1)
                    spaces[ind] = (part * (np.sqrt(nstate)
                                           / float(np.linalg.norm(part))))[None]
                    continue
        # refuted: which condition fails -- the site permutation, unless members
        # keeping every site exist and none carries the operation's own lift
        site_space = _block_members(basis, blocks, tol=in_blocks)[0]
        keeps_sites = bool(len(site_space)) and _filled_check(site_space, blocks, tol=tol)[0]
        carries_lift = False
        if spinful and keeps_sites:
            lifted = _spin_lift_defects(model, group, validated, [p], spin=spin, tol=tol,
                                        orients=(1,))
            carries_lift = bool(lifted.get(ind, {}).get("realizable", False))
        condition = "spin lift" if (spinful and keeps_sites and not carries_lift) \
            else "site permutation"
        if condition == "spin lift":
            note = ("no member of its solution space carries this operation's own spin "
                    "lift S(g)")
        elif spinful:
            note = ("no member of its solution space carries each site's states onto its "
                    "image's under this operation with its own spin lift: the members that "
                    "carry the lift carry the sites elsewhere, as another operation does")
        else:
            note = ("no member of its solution space carries each site's states onto its "
                    "image's under this operation: its matrices are another operation's")
        if len(V):
            note = "every member of its solution space of its own form is singular"
        rec.update(realizable=False, dim=0, own_form=condition,
                   spin_lift=condition == "spin lift", note=note,
                   **({"defect": 1.0, "scaled": 1.0} if len(V) else {}))
        spaces[ind] = basis[:0]
    return spaces, records


def _spin_lift_defects(model, group, validated, positions, *, spin=None, tol=None,
                       orients=(1, -1)):
    r"""Per operation: does its solution space hold a member carrying its own spin lift?

    Every symmetry matrix of a spinful crystal has the form
    ``D(g) = O(g) ⊗ S(g)``: an orbital factor, and on the spin the operation's
    own SU(2) lift in pythtb's convention -- the tables' lift carried onto the
    model's cell by the computed frame rotation (:func:`_lift_of`), the one
    :func:`_build_rep` builds every representation with.  The detector assumes
    no such structure: it finds every matrix that carries ``H(k)`` onto
    ``H(gk)``.  So a Hamiltonian whose spin blocks are written in a frame
    turned against its lattice passes it for operations it obeys only through
    a spin factor other than their lift, and this is where that is measured.

    With the spin pairing explicit the question is linear.  A member ``X`` of
    the solution space has the form ``O ⊗ S(g)``, for some orbital factor
    ``O``, exactly where ``X (1 ⊗ S(g))†`` commutes with ``1 ⊗ σ_a`` for
    ``a = x, y, z``, that is where it equals its own spin trace times the
    identity on the spin.  The space is projected onto that form once, and
    ``scaled`` is the norm of what the closest member could not put in it,
    relative to that member's own norm: ``0`` where the lift is carried and
    of order one where it is not.  No orbital structure is asked for -- no
    multiplets, no site permutation, any orbital factor at all -- so the test
    reads the spin alone, for a multiplet of any dimension, and a model with
    no spin-orbit coupling passes it for every operation it obeys.  That is
    also what tells a layer's ``σ_h`` pair apart where a site's orbitals are
    bound into a multiplet wider than one: the two members share one solution
    space and differ only in their lift.  A member of
    that form is a symmetry matrix only where it is nonsingular, which is
    measured on fixed pseudo-random combinations of the members in the form.

    Both orientation branches are measured where ``orients`` asks for them,
    the ``g⁻¹`` branch carrying ``S(g)†`` as the build reads it, and an
    operation passes where either does, as in :func:`_ansatz_defects`; an
    antiunitary operation's lift is its unitary part's times time reversal's
    ``iσ_y``, as :func:`_lift_of` gives it.  ``positions`` index
    ``group.symmetries``.  The states are paired as pythtb stores a spinful
    model's, ``2·orbital + spin``, or as the declared
    :class:`..spin.SpinBasis` ``spin`` pairs them; a spinless model with no
    declared basis has no spin factor to test, and nothing is returned for it.
    ``tol`` is the declared tolerance, relative to ``max|H|``, that every
    threshold here reads (:func:`.._tolerance.noise_floor`): the gate on ``scaled`` is
    :data:`ANSATZ_GATE` or it, whichever is larger, and ``None`` keeps the
    exact defaults.

    Returns ``{ind: record}`` in the fields :func:`_ansatz_defects` returns --
    ``dim``, ``defect``, ``scale``, ``norm``, ``scaled``, ``orientation``,
    ``branches``, ``realizable`` and ``note`` -- each record marked
    ``spin_lift``.  An
    operation with no measured space, a single member of a larger space, or a
    space too large to project is reported untested and realizable: nothing
    was measured, so nothing is refuted.
    """
    if int(model.nspin) == 2:
        order = np.arange(int(model.nstate))
    elif spin is not None:
        order = np.asarray(spin.interleaved_index, dtype=int)
    else:
        return {}
    if not bool(getattr(group, "spinor", True)):
        return {}
    gate = noise_floor(ANSATZ_GATE, tol)
    singular = noise_floor(SINGULAR, tol)
    npair = int(order.size) // 2
    syms = group.symmetries
    out = {}
    for p in positions:
        sym = syms[p]
        ind = int(sym.ind)
        rec = {"ind": ind, "dim": 0, "defect": float("nan"), "scale": float("nan"),
               "norm": float("nan"), "scaled": float("nan"), "orientation": 0,
               "branches": {}, "realizable": True, "note": None, "spin_lift": True}
        out[ind] = rec
        basis = getattr(validated, "_bases", {}).get(ind)
        if basis is None or len(np.asarray(basis)) == 0:
            member = validated._intertwiners.get(ind)
            if member is None:
                rec["note"] = ("no solution of this operation's condition is carried, "
                               "so its spin lift was not tested")
                continue
            if int((getattr(validated, "_nullspace_dims", None) or {}).get(ind, 1)) != 1:
                rec["note"] = ("only one member of a larger solution space is carried, "
                               "so its spin lift was not tested")
                continue
            basis = np.asarray(member)[None, :, :]
        basis = np.asarray(basis, dtype=complex)
        nstate = int(basis.shape[1])
        if basis.shape[0] * nstate * nstate * 16 > _READOFF_MAX_BYTES:
            rec["note"] = (f"the solution space is {basis.shape[0]}-dimensional at "
                           f"nstate = {nstate}, whose projection would allocate more "
                           f"than {_READOFF_MAX_BYTES / 1e6:.0f} MB, so its spin lift "
                           "was not tested")
            continue
        # the space in the pairing's own order, (pair, spin) fast, and an
        # orthonormal basis of it: a unit coefficient vector is a unit member
        flat = basis[:, order][:, :, order].reshape(len(basis), -1).T
        u, sv, _vh = np.linalg.svd(flat, full_matrices=False)
        cut = noise_floor(ZERO, tol) * max(float(sv[0]), 1e-300) if sv.size else 0.0
        u = u[:, sv > cut] if sv.size else u[:, :0]
        dim = int(u.shape[1])
        rec["dim"] = dim
        if not dim:                               # pragma: no cover - a guard, not a path
            rec["note"] = "the measured solution space is empty"
            continue
        members = u.T.reshape(dim, npair, 2, npair, 2)
        lift = _lift_of(sym, True)
        best, singular_only = None, False
        for orient in tuple(orients):
            S = lift.conj().T if orient < 0 else lift
            # X (1 (x) S^dagger), its spin trace, and what no orbital factor carries
            W = np.einsum("jasbu,tu->jasbt", members, S.conj())
            O = 0.5 * np.einsum("jasbs->jab", W)
            R = W - np.einsum("jab,st->jasbt", O, np.eye(2))
            _u2, s, vh = np.linalg.svd(R.reshape(dim, -1).T, full_matrices=False)
            scaled = float(s[-1])
            # the members in that form, nonsingular on a pseudo-random combination
            inside = [k for k in range(len(s)) if float(s[k]) <= gate]
            regular = False
            rng = np.random.default_rng(17)
            for _ in range(3 if inside else 0):
                y = (rng.normal(size=len(inside))
                     + 1j * rng.normal(size=len(inside)))
                coef = np.einsum("k,kj->j", y, vh[inside].conj())
                o_sv = np.linalg.svd(np.einsum("j,jab->ab", coef, O), compute_uv=False)
                if float(o_sv[-1]) > singular * float(o_sv[0]):
                    regular = True
                    break
            rec["branches"][int(orient)] = scaled
            key = (not (bool(inside) and regular), scaled)
            if best is None or key < best:
                best = key
                singular_only = bool(inside) and not regular
                rec.update(defect=scaled, scale=float(s[0]), norm=1.0, scaled=scaled,
                           orientation=int(orient))
        if best is None or not best[0]:
            continue
        if singular_only:
            # in the form, and singular in every member that is: no symmetry
            # matrix of this operation carries its lift
            rec.update(realizable=False, defect=1.0, scaled=1.0,
                       note=("every member of the solution space that carries this "
                             "operation's own spin lift is singular"))
        else:
            rec["realizable"] = False
    return out


# --------------------------------------------------------------------------
# declared orbital characters
# --------------------------------------------------------------------------

#: The one sentence :meth:`AtomicLimitTable.summary` prints under the table
#: wherever a row's site irrep rests on the lowest-``l`` rule.
_LOWEST_L_LEGEND = (
    "the hoppings fix how the orbitals transform relative to one another, not what "
    "they are; where several site irreps fit, the one carried by the "
    "lowest-angular-momentum functions is shown, and `characters=` states another.")

#: What :attr:`AtomicLimitRow.determined_by` reads on a declared row whose
#: declared irrep the Hamiltonian's own blocks do not carry.
_NOT_CARRIED = "declaration not carried"


@dataclass(frozen=True)
class AtomicLimitRow:
    """One orbital's row of the atomic-limit table: its Wyckoff position, the site irrep it carries, and how that is known.

    ``rep.characters[i]`` is orbital ``i``'s row, and ``row.summary()`` prints
    it.

    Attributes
    ----------
    orbital, site : int
        The orbital, and the site it sits on.
    wyckoff : str
        The site's Wyckoff position in the tables' setting, multiplicity and
        letter (``"3c"``); ``""`` where the tables do not reach the cell.
    site_symmetry : str
        The Hermann–Mauguin symbol of the site-symmetry group (``"-42m"``).
    function : str
        The orbital function declared, as ``characters=`` was given it
        (``"px"``, ``"dx2-y2"``, the shell ``"p"``); ``""`` where none was.
    multiplet : tuple of int
        The orbitals the Hamiltonian binds this one with, which carry its site
        irrep together.
    irrep : str
        The site irrep the states carry, in the site group of the group
        returned, and with spin the double-valued site corep: ``-E1`` for an
        ``s`` orbital at diamond's ``8a``.  A multiplet holding several irreps
        reads their sum, ``"2x-E1 + -1E2E"`` for a spin–orbit ``p`` shell at a
        ``3m`` site, and one holding copies of one irrep reads it on each row;
        ``""`` where nothing names it.
    named_by : str
        Where the name comes from: ``"tables"``, the tables' band
        representations at this position; ``"point group"``, the site's point
        group at the origin of a symmorphic group; ``"Mulliken"``, the row's
        Mulliken symbol; or ``"unnamed"``.
    determined_by : str
        How the site irrep is known: ``"declaration"``; ``"declaration not
        carried"``, declared and not carried by the Hamiltonian's own blocks;
        ``"hamiltonian"``, the same under every labelling the declarations
        leave open; or ``"lowest-l"``, chosen by the lowest-angular-momentum
        rule, which ``characters=`` overrides.
    dim : int
        The dimension of what ``irrep`` names: twice the orbital factor's with
        spin.
    site_group : tuple of int
        The indices of the unitary operations that fix the site.
    character, generator_character : dict
        The orbital factor's character, single-valued with spin or without: on
        every operation of the site group by index, and on one operation of each
        class by name.
    declared_as : object or None
        Exactly what was written for this orbital.
    declared_irrep, declared_in, reduced_in : str
        The irrep, or sum, that declaration resolved to; the site group and
        group it was read in (``"3m in P3m11' #156.50"``); and, where the
        declared content lowered the cell first, the site group and group it
        was resolved in (``"m in Cm1' #8.33"``).  ``""`` where nothing applies.
    declaration_broke : tuple of str
        The operations the declared content breaks.
    note : str
        How a declaration was read; on a declared spinor basis, which member of
        its pair the row is; for a name Mulliken's convention gives another
        row, which row; what a declaration not carried stated; and where the
        cell could not be matched to the tables' setting, why no name comes
        from them.
    residual : float
        How far the orbital's block misses its character times its spin factor,
        or, on a multiplet of several orbitals, misses multiplying as the site
        group does: a large one says the irrep is not to be trusted.
    """

    orbital: int
    site: int
    wyckoff: str
    site_symmetry: str
    function: str
    multiplet: tuple
    irrep: str
    named_by: str
    determined_by: str
    dim: int
    site_group: tuple
    character: dict
    generator_character: dict
    declared_as: Any = None
    declared_irrep: str = ""
    declared_in: str = ""
    reduced_in: str = ""
    declaration_broke: tuple = ()
    note: str = ""
    residual: float = 0.0

    def summary(self) -> str:
        """The row, a line for each part that applies, ending with the attribute that holds it."""
        named = {"tables": "the tables", "point group": "the site's point group",
                 "Mulliken": "Mulliken's convention"}.get(self.named_by)
        lines = [f"orbital {self.orbital}, site {self.site}, at {self.wyckoff or '?'} "
                 f"({self.site_symmetry}): {self.irrep or 'unnamed'}"
                 + (f", named by {named}" if named else "") + "   (.irrep, .named_by)"]
        echo = ""
        if self.determined_by in ("declaration", _NOT_CARRIED):
            said = repr(self.declared_as) + (
                f" ({self.declared_irrep})" if self.declared_irrep
                and self.declared_irrep != str(self.declared_as) else "")
            lines.append(f"  declared {said}, read at the {self.declared_in}"
                         + ("" if self.determined_by == "declaration" else
                            ", and the Hamiltonian's own blocks do not carry it")
                         + "   (.declared_as, .declared_irrep, .declared_in)")
            echo = f"{self.declared_as!r} resolved to {self.declared_irrep} on this site group"
        else:
            lines.append("  " + ("fixed by the Hamiltonian" if self.determined_by == "hamiltonian"
                                 else "chosen by the lowest-angular-momentum rule among the "
                                      "site irreps the hoppings allow")
                         + "   (.determined_by)")
        if self.reduced_in:
            lines.append("  the declared content lowered the cell before any hopping was read, "
                         f"and resolves in the {self.reduced_in}   (.reduced_in)")
        if self.declaration_broke:
            lines.append("  the declaration breaks " + ", ".join(self.declaration_broke)
                         + "   (.declaration_broke)")
        others = [int(o) for o in self.multiplet if int(o) != int(self.orbital)]
        if others:
            lines.append(f"  one multiplet with {_orbitals_named(others)}   (.multiplet)")
        lines.append("  orbital characters: "
                     + ", ".join(f"{k}={_fmt_char(v)}" for k, v in self.generator_character.items())
                     + "   (.generator_character)")
        said = "; ".join(part for part in str(self.note).split("; ") if part and part != echo)
        if said:
            lines.append(f"  {said}   (.note)")
        return "\n".join(lines)


@dataclass(frozen=True)
class BandRepresentation:
    """The band representation an orbit of orbitals induces, as a sum of elementary band representations.

    The sum is written at a position the orbitals can be moved to along their
    variety, their own first.  Where none carries the content, as for ``E`` at
    ``3c`` of ``P-43m``, it is the sum at other positions with the orbit's own
    stable real-space invariants, or, where the group has none tabulated, the
    sum at the first position on the closure of a variety through the site
    that carries it; on a free-standing layer only positions within the layer
    are read.  It is the band representation and nothing finer: which site
    irrep goes to which term is a subduction the tables do not carry.
    ``br.summary()`` prints it; it is one entry of
    ``rep.characters.band_representations``.

    Attributes
    ----------
    orbitals : tuple of int
        The orbitals on the orbit's sites.
    wyckoff, site_symmetry : str
        The orbit's Wyckoff position, multiplicity and letter (``"6c"``), and
        its site-symmetry group (``"3m"``).
    terms : tuple
        ``(multiplicity, elementary band representation, position)`` for each
        term, as :func:`..groups.ebr_data` writes them
        (``(1, "-E1g↑G(2)", "3a(-3m1',-3m)")``); empty where no sum is returned.
    alternatives : tuple
        The other sums with the same content and the same stable real-space
        invariants, which nothing here distinguishes from it, each written as
        ``terms`` is; empty where the group has no tabulated invariants.
    note : str
        Where no sum is returned, why: none carries the content, several do and
        nothing here says which is induced, or the content could not be
        identified.
    """

    orbitals: tuple
    wyckoff: str
    site_symmetry: str
    terms: tuple
    alternatives: tuple = ()
    note: str = ""

    def spelled(self, terms=None) -> str:
        """A sum on one line in the labels :func:`..groups.ebr_data` lists, ``"1 x -E1g↑G(2)@3a(-3m1',-3m) + 1 x -E1u↑G(2)@3a(-3m1',-3m)"``, or ``""``.

        :attr:`terms` by default, or one of :attr:`alternatives`.
        """
        terms = self.terms if terms is None else terms
        return " + ".join(f"{m} x {name}@{position}" for m, name, position in terms)

    def sentence(self) -> str:
        """The sentence the atomic-limit table's summary prints for this orbit, where it prints one.

        Orbits at one Wyckoff position that induce the same sums share one
        line there instead, each orbit's orbitals named.
        """
        one = len(self.orbitals) == 1
        who = _orbitals_named(self.orbitals)
        if not self.terms:
            return f"{who} at {self.wyckoff}: {_no_sum(self)}"
        return (f"{who} at {self.wyckoff} induce{'s' if one else ''} "
                f"{_ebrs_of(self.terms)}{_not_distinguished(self)}")

    def summary(self) -> str:
        """The sum this orbit induces, then those it is not distinguished from, a line each."""
        who = (f"{_orbitals_named(self.orbitals)} at {self.wyckoff or '?'} "
               f"({self.site_symmetry})")
        if not self.terms:
            return f"{who}: {_no_sum(self)}   (.note)"
        lines = [f"{who} induce{'s' if len(self.orbitals) == 1 else ''} "
                 f"{_ebrs_of(self.terms)}   (.terms)"]
        if self.alternatives:
            several = len(self.alternatives) > 1
            lines.append(f"  not distinguished from th{'ese sums' if several else 'is sum'}, "
                         f"which ha{'ve' if several else 's'} the same content and the same "
                         "stable real-space invariants   (.alternatives)")
            lines += _sum_lines(self.alternatives)
        return "\n".join(lines)


def _fold(k) -> str:
    """``"twofold"``, ``"threefold"``, ...: how many identical copies a pseudospin makes."""
    return {2: "twofold", 3: "threefold", 4: "fourfold"}.get(int(k), f"{int(k)}-fold")


def _in_words(n) -> str:
    """A small count in words, ``"two"``; a larger one in figures."""
    return {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}.get(int(n), str(int(n)))


@dataclass(frozen=True)
class MultiplicitySpace:
    """A pseudospin the Hamiltonian conserves on an orbit of orbitals, and the site content assigned its copies.

    Where the Hamiltonian holds identical copies of a part of the crystal that
    nothing joins, every band there is as many times degenerate as there are
    copies, and nothing the hoppings measure says what the copies carry, which
    moves the site content itself.  A declaration on their orbitals states it,
    and the lowest-angular-momentum rule assigns it otherwise, the copies read
    copy by copy first.  ``space.summary()`` prints it; it is one entry of
    ``rep.characters.multiplicity_spaces``.

    Attributes
    ----------
    orbitals : tuple of int
        The orbitals the copies are made of.
    wyckoff : str
        The Wyckoff position of the site whose content is read; ``""`` where the
        tables do not reach the cell.
    dim : int
        The number of copies, the degeneracy of every band on these orbitals.
    content : str
        The site content returned.
    contents : tuple of str
        The site contents the representations of the copies the group's own
        table admits give, in the order the lowest-angular-momentum rule ranks
        them; with spin, two can name one content.
    n_representations : int
        How many representations of the copies the group's own table admits.
    determined_by : str
        ``"declaration"``; ``"lowest-l"``, the rule's choice; or
        ``"hamiltonian"``, where the table admits one representation.
    """

    orbitals: tuple
    wyckoff: str
    dim: int
    content: str
    contents: tuple
    n_representations: int = 1
    determined_by: str = "lowest-l"
    # how many Wannier functions of the copies one site holds -- orbitals, or
    # declared spinor pairs where ``_pairs`` -- which says what one copy is
    _per_site: int = field(default=0, repr=False, compare=False)
    _pairs: bool = field(default=False, repr=False, compare=False)

    def _distinct(self) -> tuple:
        """The site contents admitted, each named once, in the rule's order.

        With spin, different orbital contents can lift to one site corep
        content, ``2xA1`` and ``2xA2`` of ``3m`` both to ``2x-E1``, and a sum
        can be spelled in either order of its terms, ``-E1 + -E2`` and
        ``-E2 + -E1``: a content is the irreps it holds and how many of each.
        """
        seen, out = set(), []
        for content in self.contents:
            key = tuple(sorted((name, n) for n, name in _content_terms(content)))
            if key not in seen:
                seen.add(key)
                out.append(content)
        return tuple(out)

    def sentence(self) -> str:
        """The sentence the atomic-limit table's summary prints for this pseudospin."""
        head = (f"orbitals {tuple(self.orbitals)}"
                + (f" at {self.wyckoff}" if self.wyckoff else "")
                + f" carry a {_fold(self.dim)} pseudospin the Hamiltonian conserves")
        content = self.content or "(unnamed)"
        if self.determined_by == "declaration":
            return (f"{head}, and what its copies carry it does not fix: the declaration "
                    f"states the site content {content}, of the {len(self._distinct())} the "
                    f"group's own table admits")
        if len(self._distinct()) > 1:
            return (f"{head}, and what its copies carry it does not fix: the "
                    f"lowest-angular-momentum rule assigns the site content {content}, of "
                    f"the {len(self._distinct())} the group's own table admits")
        if int(self.n_representations) > 1:
            return (f"{head}; the {int(self.n_representations)} representations of its "
                    f"copies the group's own table admits all give the site content "
                    f"{content}, and the lowest-angular-momentum rule chose the one "
                    f"returned among them")
        return (f"{head}; the group's own table admits one representation of its copies, "
                f"giving the site content {content}")

    def summary(self) -> str:
        """The pseudospin, what its copies carry and how that was chosen, a line each."""
        how = {"declaration": "stated by the declaration",
               "lowest-l": "assigned by the lowest-angular-momentum rule",
               "hamiltonian": "the one content the group's own table admits"}.get(
                   self.determined_by, self.determined_by)
        # what one copy is: the Wannier functions it holds at one site, orbitals
        # or declared spinor pairs, and whether the orbit has several sites
        unit = "spinor pair" if self._pairs else "orbital"
        held = len(self.orbitals) // (2 if self._pairs else 1)
        per_site = int(self._per_site) or held
        each = max(1, per_site // max(1, int(self.dim)))
        copy = (f"one {unit}'s states" if each == 1 else f"{_in_words(each)} {unit}s' states")
        lines = [f"{_orbitals_named(self.orbitals)}"
                 + (f" at {self.wyckoff}" if self.wyckoff else "")
                 + f": {_in_words(self.dim)} identical copies of {copy}"
                 + (" at each site" if held > per_site else "")
                 + ", a pseudospin the Hamiltonian conserves   (.dim)",
                 f"  the copies carry {self.content or 'a content no irrep names'}, {how}"
                 "   (.content, .determined_by)"]
        distinct = self._distinct()
        if len(distinct) > 1:
            lines.append("  every content the group's own table admits, in the rule's order: "
                         + ", ".join(distinct) + "   (.contents)")
        if int(self.n_representations) > len(distinct):
            lines.append(f"  {int(self.n_representations)} representations of the copies give "
                         + ("these contents" if len(distinct) > 1 else "that content")
                         + "   (.n_representations)")
        return "\n".join(lines)


@dataclass(frozen=True)
class AtomicLimitTable:
    """What each orbital carries at its site: the table an atomic limit is written up in, a row per orbital.

    ``rep.characters`` is the table; ``table.summary()`` prints it, and
    ``table[i]`` is orbital ``i``'s :class:`AtomicLimitRow`.  The table is where
    a mis-oriented declaration shows, in each row's orbital characters.

    Attributes
    ----------
    orbitals : tuple of AtomicLimitRow
        One row per orbital, in the model's order.
    irreps : tuple of str
        Each orbital's site irrep, in the model's order.
    band_representations : tuple of BandRepresentation
        The band representation each orbit of orbitals induces, a sum of
        elementary band representations, in the order of the orbits; empty
        where the tables do not reach the cell.
    multiplicity_spaces : tuple of MultiplicitySpace
        One per orbit on which the Hamiltonian conserves a pseudospin, with the
        site content assigned its copies; empty where none does.
    declarations : tuple of dict
        Each declaration as it was read: ``orbitals``, ``declared_as`` and the
        ``declared_irrep`` it resolved to, ``dim``, ``note``, ``declared_in``,
        ``reduced_in``, the operations it ``broke``; ``mulliken``, each name
        written that Mulliken's convention gives another row, with that row's
        name; and ``returned``, what its multiplets carry together.
    checks : tuple of dict
        Each declaration against each operation of its site group the hoppings
        say something about, ``orbitals``, ``ind``, ``name``, ``measured`` and
        ``declared``: the table a ``declaration_breaks_group`` refusal prints.
        ``measured`` is read against a multiplet of the declaration's own part
        of the crystal, so the two differ by a one-dimensional character of
        that part.
    frame : dict
        ``{ind: name}``, every operation represented, named in the model's
        frame.
    spin : ..spin.SpinBasis or None
        The declared spinor basis, where one was; the table then has a ``spin``
        column, each row's member of its pair.
    residual : float
        The largest ``residual`` of any row.
    site_tables : dict
        ``{site: table}``, each site group's character table.
    """

    orbitals: tuple
    declarations: tuple
    checks: tuple
    frame: dict
    residual: float
    site_tables: dict = field(default_factory=dict, repr=False)
    spin: Any = None
    band_representations: tuple = ()
    multiplicity_spaces: tuple = ()
    # the chooser's own record of the labellings that give every orbital the
    # site irrep the one returned gives it and label some band differently:
    # each coupled part's states, and per part the band labels of every energy
    # multiplet under the one returned and under each of those, with the key
    # that set it aside -- read by `sewing.hsp_table` to say where a set of
    # bands' labels rest on that choice
    _labellings: dict = field(default_factory=dict, repr=False, compare=False)

    def __len__(self) -> int:
        return len(self.orbitals)

    def __getitem__(self, orbital) -> AtomicLimitRow:
        return self.orbitals[int(orbital)]

    @property
    def irreps(self) -> tuple:
        """Each orbital's site irrep, in the model's order."""
        return tuple(o.irrep for o in self.orbitals)

    def summary(self) -> str:
        """The table, a row per orbital, and a line under it for each thing the rows do not say.

        The columns are the row's fields; ``orbital characters`` is the orbital
        factor's character on one operation of each class, without spin.  Under
        the rows, each line ending with the attribute that holds it: a
        declaration that lowered the group, that the Hamiltonian's own blocks
        do not carry, or that is written in a name Mulliken's convention gives
        another row, with what was declared, where it was read and what is
        returned in its place; an orbit whose band representation is not its
        rows' own elementary band representations at its own position; and a
        conserved pseudospin, with the site content assigned its copies.  Where
        a row reads ``lowest-l``, a line says what that means.
        """
        head = ["orbital", "Wyckoff", "site symmetry", "function"]
        spins = None
        if self.spin is not None:
            sgn = np.asarray(self.spin.spin_of_orbital, dtype=int)
            spins = ["up" if sgn[o.orbital] > 0 else "down" for o in self.orbitals]
            head.append("spin")
        head += ["irrep", "named by", "determined by"]
        body = []
        for i, o in enumerate(self.orbitals):
            cells = [str(o.orbital), o.wyckoff, o.site_symmetry, o.function]
            if spins is not None:
                cells.append(spins[i])
            cells += [o.irrep, o.named_by, o.determined_by]
            body.append(cells)
        width = [max([len(h)] + [len(r[j]) for r in body]) for j, h in enumerate(head)]
        rows = ["  ".join(f"{h:>{w}s}" for h, w in zip(head, width)) + "  orbital characters"]
        for o, cells in zip(self.orbitals, body):
            chars = ", ".join(f"{k}={_fmt_char(v)}" for k, v in o.generator_character.items())
            rows.append("  ".join(f"{c:>{w}s}" for c, w in zip(cells, width)) + f"  {chars}")
        for d in self.declarations:
            orbs = tuple(int(a) for a in d["orbitals"])
            row = self.orbitals[orbs[0]]
            dropped = row.determined_by == _NOT_CARRIED
            if not (d.get("reduced_in") or d.get("broke") or dropped or d.get("mulliken")):
                continue
            who = f"orbital {orbs[0]}" if len(orbs) == 1 else f"orbitals {orbs}"
            said = f"{d['declared_as']!r}" + (
                f" ({d['declared_irrep']})" if d["declared_irrep"]
                and d["declared_irrep"] != d["declared_as"] else "")
            parts = [f"  {who}: declared {said}, read at the {d['declared_in']}"]
            if d.get("mulliken"):
                parts.append(_sitechars.mulliken_clause(d["mulliken"].items()))
            if d.get("reduced_in"):
                parts.append("the declared content lowered the cell before any hopping "
                             f"was read and resolves as "
                             f"{row.declared_irrep or '(unnamed)'} of the {d['reduced_in']}")
            if d.get("broke"):
                parts.append("the declaration breaks " + ", ".join(d["broke"]))
            if dropped:
                parts.append("the Hamiltonian's own blocks do not carry it under the "
                             "one-dimensional character returned")
            parts.append(f"returned as {d.get('returned', row.irrep) or '(unnamed)'} of "
                         + (row.site_symmetry
                            or f"the site group of order {len(row.site_group)}"))
            rows.append("; ".join(parts) + "   (.declarations)")
        alike: dict = {}
        for orbit in self.band_representations:
            if not _its_rows_own(self, orbit):
                key = (orbit.wyckoff, orbit.terms, orbit.alternatives, orbit.note)
                alike.setdefault(key, []).append(orbit)
        for orbits in alike.values():
            rows.append("  " + _orbits_sentence(orbits) + "   (.band_representations)")
        for space in self.multiplicity_spaces:
            rows.append("  " + space.sentence() + "   (.multiplicity_spaces)")
        if any(o.determined_by == "lowest-l" for o in self.orbitals):
            rows.append("  lowest-l: " + _LOWEST_L_LEGEND)
        return "\n".join(rows)

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return (f"AtomicLimitTable({'/'.join(self.irreps)}"
                + (", lowest-l" if any(o.determined_by == "lowest-l"
                                       for o in self.orbitals) else "") + ")")


def _its_rows_own(table, orbit) -> bool:
    """Whether an orbit's band representation is its rows' own elementary ones.

    That is, every row of the orbit named by the tables' entries at its own
    position, every term of the sum at that position, and no other sum at that
    position beside it: the terms are then the elementary band representations
    of the rows' own site irreps, and a line under the table would repeat them.
    A sum at other positions with the same content and the same stable
    real-space invariants is on the record
    (:attr:`BandRepresentation.alternatives`) and prints no line of its own.
    """
    rows = [table.orbitals[int(a)] for a in orbit.orbitals]

    def _here(terms):
        return all(str(position).split("(")[0] == orbit.wyckoff
                   for _m, _name, position in terms)

    return (bool(orbit.terms) and _here(orbit.terms)
            and not any(_here(other) for other in orbit.alternatives)
            and all(o.named_by == "tables" for o in rows))


def _orbits_sentence(orbits) -> str:
    """The line under the atomic-limit table for orbits that induce the same sums.

    One orbit reads as :meth:`BandRepresentation.sentence` writes it; several at
    one Wyckoff position, inducing the same sums, read once, each orbit's
    orbitals named.
    """
    first = orbits[0]
    if len(orbits) == 1:
        return first.sentence()
    names = [_index_ranges(o.orbitals) for o in orbits]
    who = (f"{len(orbits)} orbits at {first.wyckoff}, orbitals "
           f"{', '.join(names[:-1])} and {names[-1]}")
    if not first.terms:
        return f"{who}: {_no_sum(first)}"
    return f"{who}, each induce {_ebrs_of(first.terms)}{_not_distinguished(first)}"


def _no_sum(orbit) -> str:
    """What an orbit's line says in place of a sum: its note, which says it whole."""
    return orbit.note or "no sum of elementary band representations"


def _not_distinguished(orbit) -> str:
    """The orbit's alternatives on its line, ``", not distinguished by ... from the ... of A or of B"``, or ``""``."""
    others = orbit.alternatives
    if not others:
        return ""
    if len(others) == 1:
        joined = _ebrs_of(others[0])
    else:
        words = [_sum_words(terms) for terms in others]
        joined = ("the elementary band representations of " + ", of ".join(words[:-1])
                  + " or of " + words[-1])
    return f", not distinguished by the stable real-space invariants from {joined}"


def _fmt_char(z) -> str:
    # a character that vanishes prints as the 0 a table would carry -- unsigned,
    # and not the rounding it was measured to; the value itself stays complex
    z = complex(z)
    re_, im_ = (0.0 if abs(z.real) < 1e-9 else z.real), (0.0 if abs(z.imag) < 1e-9 else z.imag)
    if im_ == 0.0:
        return "0" if re_ == 0.0 else f"{re_:+.3g}"
    return f"{re_:+.3g}{im_:+.3g}j"


#: What a printout that counts operations counts: one per coset of the lattice
#: translations, and on states carrying a spin the single-valued group's, the
#: double group holding each one's negative besides.  Said once in a printout,
#: wherever the count is the information; where the group's symbol fixes the
#: count, no count is printed.
_COSETS = "one per coset of the lattice translations"
_SINGLE_VALUED = ("the single-valued group's, each standing for itself and its negative "
                  "in the double group")


def _counted_as(spinful) -> str:
    """The unit a count of operations is in, as a printout says it once."""
    return _COSETS + (f", {_SINGLE_VALUED}" if spinful else "")


def _index_ranges(indices) -> str:
    """Sorted indices as runs, ``0-3, 5, 7-8``."""
    runs, start, prev = [], None, None
    for i in sorted(int(i) for i in indices):
        if start is not None and i == prev + 1:
            prev = i
            continue
        if start is not None:
            runs.append(f"{start}-{prev}" if prev > start else f"{start}")
        start = prev = i
    if start is not None:
        runs.append(f"{start}-{prev}" if prev > start else f"{start}")
    return ", ".join(runs)


def _orbitals_named(orbitals) -> str:
    """A set of orbitals in words: ``"orbital 3"``, ``"orbitals 0-2"``."""
    orbitals = tuple(int(o) for o in orbitals)
    return f"orbital{'s' if len(orbitals) > 1 else ''} {_index_ranges(orbitals)}"


#: How many sums of elementary band representations a summary prints before it
#: says how many more there are.
_SHOWN = 6


def _term_words(multiplicity, name, position) -> str:
    """One term of a sum as the site irrep at its Wyckoff position: ``2×-E2 at 1b (4mm)``."""
    irrep = str(name).split("↑")[0]
    position = str(position)
    letter = position.split("(")[0]
    if "(" in position:
        site = position[position.index("(") + 1:position.rindex(")")].split(",")[-1]
        letter = f"{letter} ({site})"
    count = f"{int(multiplicity)}×" if int(multiplicity) != 1 else ""
    return f"{count}{irrep} at {letter}"


def _sum_words(terms) -> str:
    """A sum of elementary band representations in words: ``-E1g at 3a (-3m) + -E1u at 3a (-3m)``."""
    return " + ".join(_term_words(*term) for term in terms)


def _sum_lines(sums) -> list:
    """Each sum of elementary band representations on its own line, the first few."""
    lines = ["    " + _sum_words(terms) for terms in sums[:_SHOWN]]
    if len(sums) > _SHOWN:
        lines.append(f"    and {len(sums) - _SHOWN} more")
    return lines


def _ebrs_of(terms) -> str:
    """``"the elementary band representations of -E1g at 3a (-3m) + -E1u at 3a (-3m)"``."""
    one = sum(int(m) for m, _name, _position in terms) == 1
    return f"the elementary band representation{'' if one else 's'} of {_sum_words(terms)}"


def _content_terms(irrep) -> list:
    """A site content as a row spells it, ``"2x-E1 + -1E2E"``, as ``[(2, "-E1"), (1, "-1E2E")]``."""
    out = []
    for term in str(irrep).split(" + "):
        count, times, name = term.partition("x")
        out.append((int(count), name) if times and count.isdigit() else (1, term))
    return out


def _content_by_position(table, states_per_orbital) -> list:
    """The orbitals at each Wyckoff position and what one site there holds, a line each.

    ``"2b (3m): orbitals 0-1, -E1"``.  The content is read at the position's
    first site, each multiplet there once, and a multiplet of several
    identical copies of one irrep counts that irrep as many times, its rows
    each naming it (:attr:`AtomicLimitRow.irrep`).  ``states_per_orbital`` is
    two on a model whose orbitals carry their own spin, where a row's
    dimension counts states and its multiplet orbitals, and one otherwise.
    """
    by_position = {}
    for row in table.orbitals:
        by_position.setdefault((row.wyckoff, row.site_symmetry), []).append(row)
    lines = []
    for (wyckoff, site_symmetry), rows in by_position.items():
        first = min(int(row.site) for row in rows)
        counts, seen = {}, set()
        for row in rows:
            if int(row.site) != first or tuple(row.multiplet) in seen:
                continue
            seen.add(tuple(row.multiplet))
            copies = max(1, len(row.multiplet) * int(states_per_orbital) // max(1, int(row.dim)))
            for n, name in _content_terms(row.irrep or "unnamed"):
                counts[name] = counts.get(name, 0) + n * copies
        content = " + ".join(f"{n}x{name}" if n > 1 else name for name, n in counts.items())
        lines.append(f"{wyckoff or '?'} ({site_symmetry}): "
                     f"{_orbitals_named(row.orbital for row in rows)}, {content}")
    return lines


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
    labelling by site symmetry; without one, :func:`_build_rep` takes the one
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


def _covariant_subgroup(group, kept_u, requirements, *, unitary_only=False, tol=CHARACTER,
                        joint=(), search=None):
    r"""The largest subgroup the declared orbital content is covariant on.

    Read off the sectors of the Hamiltonian's conserved matrices wherever they
    read the round (``search``, :class:`.._sectors.SectorSearch`): a subgroup
    carries the declaration where one representation per orbit of sectors under
    **that subgroup**, of those the hoppings leave open, gives every declared
    site its content (:func:`_subgroup_read`).  A smaller group splits the
    orbits by itself -- three uncoupled lines a threefold axis exchanges are one
    orbit of ``6mm`` and two of ``mm2``, each with a character of its own -- so
    nothing is ruled out on the larger group's orbits before the smaller one is
    asked.  Where the sectors read nothing the requirements below decide.

    The hoppings fix every multiplet's block up to one one-dimensional
    character per **coupled part** of the crystal — the multiplets they
    connect, directly or through others — so each part's declarations are
    read against a scale of its own and ask for a character of their own.
    ``requirements`` holds what the measurement left, one mapping per part,
    with one entry per unitary operation it could speak about: ``None`` where
    the operation is *individually* disallowed — two declared multiplets of
    the part it fixes need different scalars, or the scalar one of them needs
    is not ``±1`` — and otherwise the scalar ``λ(g)`` the part's character has
    to take there.  An operation absent from a mapping constrains nothing
    there.  A single mapping is one part.

    Every declaration is such a constraint, wherever it sits: an undeclared
    orbital, the reference orbital included, imposes none.  The declared content
    is then a representation covariant with ``H`` on a subgroup ``S`` exactly
    when every part has a one-dimensional **real** character of ``S`` meeting
    all of its requirements inside it, and the returned representation on each
    part is ``λ·D|_S``; which of the characters left the crystal is assigned is
    the lowest-angular-momentum rule's to choose, and the build chooses it once.
    Those subgroups are closed under intersection but not under join, so the
    largest is not always unique; where several of the largest order tie, each
    a consistent reading on its own, the one returned is the first by the tie
    rule every such choice is made by (:func:`..groups._tie_ranked`) — never
    their intersection, which would keep less than any of them carries — and
    all of them are recorded in the rule's order, because which readings were
    possible is the physics of what was declared.  The floor is the identity,
    which is always consistent: a descent by declaration never refuses.

    ``unitary_only`` is a group with no antiunitary operation represented,
    where the residual freedom is larger than the real characters: **every**
    one-dimensional character of a subgroup is a candidate labelling there —
    roots of unity per generator — so those are what is searched, and a declared
    content met only by a complex character is **returned** with it rather than
    descended on.  Nothing in such a group's measurement forbids one: it
    preserves the covariance, the canonical norm and the whole product table.

    ``joint`` holds the contents read together (:func:`_verdicts_and_choices`):
    one spread over several coupled parts, or one stating part of what its
    site holds.  Each asks something of several parts at once, so a subgroup
    keeps it only where one character of the subgroup for each part it spans
    carries what was declared: the same content where the whole site was
    declared, and a content containing it where part of the site was
    (:func:`_joint_holds`).

    ``tol`` is how far a character may sit from the value a requirement asks
    and still meet it (:func:`_meets`): the requirements are read off the
    solution spaces, so it is the declared tolerance where that is larger than
    its exact default.

    Returns ``{"subgroup", "candidates", "broken"}`` — the returned set of
    ``.ind``, the maximal candidates in the rule's order where several tie, the
    one returned first, and the operations of ``kept_u`` outside ``S`` — or the
    :class:`..refusal.Refusal` of a search past its ceiling.
    """
    from .groups import _subgroups_inside, _tie_ranked

    if search is not None and search.complete:
        got = _subgroup_read(group, kept_u, search)
        if got is not None:
            return got
    parts = ([requirements] if isinstance(requirements, dict)
             else [dict(part) for part in requirements])
    joint = tuple(joint or ())
    keep = {int(i) for i in kept_u}

    def _barred(i):
        return any(part.get(int(i), "free") is None for part in parts)

    # WHAT THE GROUP WOULD BE WITHOUT THE DECLARATION.  The survivors of a
    # measurement need not compose -- a sigma_h fiber partner whose lift
    # coincides with a kept operation's survives beside it, and the closure step
    # is what removes those -- so the question is asked inside the group they
    # DO form, and such a fiber partner is neither kept by the declaration nor
    # blamed on it: it leaves by the closure resolution, which is its own verdict.
    plain = _subgroups_inside(group, keep)
    widest = sorted((s for s in plain if len(s) == max(len(t) for t in plain)),
                    key=sorted) if plain else []
    if len(widest) != 1:
        # THE SURVIVORS ARE NOT A GROUP AND NO LARGEST SUBGROUP OF THEM IS
        # UNIQUE.  A declaration is read on a *group*, and there is not one here
        # yet -- the closure resolution speaks first, and where the candidates
        # tie it keeps the one the tie rule ranks first.  So only the operations
        # the declaration disallows *individually* are its own, and the
        # character question is asked again on the next round, on whatever
        # group that resolution lands on.
        bad = {i for i in keep if _barred(i)}
        return {"subgroup": frozenset(keep - bad), "candidates": (),
                "broken": tuple(sorted(bad))}
    baseline = set(widest[0])
    allowed = {i for i in baseline if not _barred(i)}

    def _met(sub):
        """Whether every coupled part has a character of ``sub`` meeting its requirements.

        And, where contents read together speak inside ``sub``, whether one
        character of each part they span meets those and its own part's at once.
        """
        inside = {int(i) for i in sub}
        asked = [i for i, wants in enumerate(parts) if any(int(g) in inside for g in wants)]
        together = [j for j in joint
                    if any(int(g) in inside for row in j["theta"].values() for g in row)]
        if not asked and not together:
            return True
        chars = (_characters_of_subgroup(group, sub, roots=bool(unitary_only))
                 or [{int(i): 1.0 + 0.0j for i in sub}])
        own = {i: [chi for chi in chars if _meets(chi, parts[i], over=inside, tol=tol)]
               for i in asked}
        if not all(own[i] for i in asked):
            return False
        if not together:
            return True
        spanned = sorted({int(i) for j in together for i in j["part"].values()})
        options = [own.get(i, chars) for i in spanned]
        for combo in itertools.product(*options):
            pick = dict(zip(spanned, combo))
            if all(_joint_holds(j, pick, inside, tol=tol) for j in together):
                return True
        return False

    consistent = {frozenset(int(i) for i in sub)
                  for sub in _subgroups_inside(group, allowed) if _met(sub)}
    if not consistent:                   # pragma: no cover - {E} always is
        return {"subgroup": frozenset(), "candidates": (),
                "broken": tuple(sorted(keep))}
    # The LARGEST consistent subgroup, and the tie rule only where the largest
    # is not unique.  Maximality by inclusion alone would let a small subgroup
    # nothing constrains -- the sigma_h fiber partners that survive beside a kept
    # operation, which the closure step removes for exactly this reason -- stand
    # beside the group the declaration reaches as a candidate.  A tie needs two consistent subgroups
    # of the SAME size, which is the case the record names.
    top = max(len(s) for s in consistent)
    ranked = _tie_ranked(group, [s for s in consistent if len(s) == top])
    best = frozenset(ranked[0])
    candidates = ranked if len(ranked) > 1 else ()
    return {"subgroup": best, "candidates": candidates,
            "broken": tuple(sorted(i for i in baseline if i not in best))}


def _cyclic(group, ind):
    """The operations one unitary operation generates, by ``.ind``."""
    prod = group.get_product_table(get_diff=False)
    at = {int(s.ind): p for p, s in enumerate(group.symmetries)}
    ind_of = [int(s.ind) for s in group.symmetries]
    out, x = {int(ind)}, int(ind)
    while True:
        x = ind_of[int(prod[at[int(ind)], at[x]])]
        if x in out:
            return frozenset(out)
        out.add(x)


def _subgroup_read(group, kept_u, search):
    r"""The largest subgroup the declared content is carried on, read off the sectors.

    The subgroups inside the group the survivors form are asked from the
    largest down (:meth:`.._sectors.SectorSearch.met`), and the first order at
    which any carries the declaration is the one kept, the tie rule choosing
    among several (:func:`..groups._tie_ranked`); the identity alone always
    carries it.  Where the survivors form no one largest group, the closure
    resolution speaks first and the declaration is asked again on the group it
    lands on: only an operation the declaration rules out on its own -- whose
    own cyclic group carries no representation that holds it -- is refuted
    here.  Returns what :func:`_covariant_subgroup` does; ``None`` where the
    sectors do not read the group, and the ``rep_search_too_large``
    :class:`..refusal.Refusal` where the representations to weigh at once are
    more than the search's ``max_candidates``.
    """
    from . import _sectors
    from .groups import _subgroups_inside, _tie_ranked

    keep = frozenset(int(i) for i in kept_u)
    try:
        plain = _subgroups_inside(group, keep)
        top = max(len(s) for s in plain) if plain else 0
        widest = sorted((s for s in plain if len(s) == top), key=sorted)
        if len(widest) != 1:
            bad = {i for i in keep if not search.met(_cyclic(group, i))}
            return {"subgroup": frozenset(keep - bad), "candidates": (),
                    "broken": tuple(sorted(bad))}
        baseline = frozenset(widest[0])
        carried, size = [], None
        for sub in sorted(_subgroups_inside(group, baseline), key=lambda s: (-len(s), sorted(s))):
            if size is not None and len(sub) < size:
                break
            if search.met(sub):
                size = len(sub)
                carried.append(frozenset(sub))
    except _sectors._Unread:
        return None
    except _sectors._TooMany as many:
        return Refusal(
            "rep_search_too_large",
            f"the declared content of orbitals {many.orbitals} is weighed against every "
            f"representation the hoppings leave open on the sectors those orbitals lie on -- "
            f"{many.copies} identical cop{'y' if many.copies == 1 else 'ies'} of each -- and the "
            f"{many.count:,} to weigh at once are more than max_candidates = "
            f"{search.max_candidates:,}. Declare less of what the copies hold, or raise "
            "`max_candidates=` if the search is worth waiting for.",
            value=float(many.count), threshold=float(search.max_candidates),
            context={"orbitals": many.orbitals, "copies": many.copies,
                     "n_candidates": many.count, "max_candidates": search.max_candidates})
    if not carried:
        raise InternalError(
            f"no subgroup of {group.name} #{group.number_str}, the identity alone included, "
            "carries the declared content on the sectors of the Hamiltonian's conserved "
            "matrices, where the identity always carries it: the copies as they stand hold "
            "every declared dimension.")
    ranked = _tie_ranked(group, carried)
    best = frozenset(ranked[0])
    return {"subgroup": best, "candidates": ranked if len(ranked) > 1 else (),
            "broken": tuple(sorted(i for i in baseline if i not in best))}


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


def _declaration_breaks(group, assignment, decls, verdict, *, standing, subgroup, broken,
                        readings, not_lowered, site_tables=None, admissible=None):
    r"""The ``declaration_breaks_group`` refusal, made in one place.

    The verdict is made once, per operation, on the detector's own solution
    spaces, the declaration read on the measured multiplets
    (:func:`_verdicts_and_choices` and
    :func:`_covariant_subgroup`, which is also what
    :func:`..groups.realized_group` descends on).  This writes it down for the
    two calls that do not lower the group — :func:`_build_rep`, handed a fixed
    group, and :func:`..groups.realized_group` told ``descend=False``, which
    ``not_lowered`` says in one sentence: which operations the declared content
    is not covariant under, which subgroup it **is** covariant on, the maximal
    readings where several tie, that subgroup the first of them by the tie rule,
    and in irrep names what the failing declaration measured.

    ``standing`` are the operations the verdict was read on, unitary and
    antiunitary.  ``subgroup`` and ``broken`` are the verdict, as unitary
    operations or with their primed partners already added; the partners are
    added here wherever they are missing.  ``site_tables`` and ``admissible``
    — the site character tables and the real one-dimensional characters of
    the group read — are computed here when they are not handed in.
    """
    from .groups import _record_name

    lattice = np.asarray(group.lattice, dtype=float)
    by_ind = {int(s.ind): s for s in group.symmetries}
    standing = sorted({int(i) for i in standing})
    # each operation as the refusal lists it, and as the frame a declaration is
    # written in names it: a site's own operations by their rotations alone
    names = {i: _record_name(by_ind[i], lattice) for i in standing}
    frame = {i: (_sitechars.operation_name(by_ind[i], lattice)
                 + ("'" if by_ind[i].time_reversal else "")) for i in standing}
    unitary = [i for i in standing if not by_ind[i].time_reversal]
    # A site character is a character of the UNITARY site group, so the verdict
    # is about unitary operations -- but the group returned is the grey one they
    # generate, and the primed copy of an operation stands or falls with it.  A
    # primed partner is counted as what it is, and never as a unitary operation.
    partners = _primed_partners(group, names)
    broken_u = sorted({int(i) for i in broken if not by_ind[int(i)].time_reversal})
    kept_u = sorted({int(i) for i in subgroup if not by_ind[int(i)].time_reversal})
    primed = [partners[i] for i in broken_u if i in partners]
    every = sorted(set(broken_u) | set(primed))
    kept = sorted(set(kept_u) | {partners[i] for i in kept_u if i in partners})
    rows = tuple(verdict["rows"])
    reasons = verdict.get("reasons") or {}
    read_together = tuple(verdict.get("joint", ())) + tuple(verdict.get("by_sectors", ()))
    why_of = {i: str(reasons.get(i) or _joint_reason(read_together, i) or (
        "the declared content is carried by this operation on its own, but no group "
        "of operations containing it and those kept carries every declared multiplet "
        "in its declared irrep at once")) for i in broken_u}
    for i in broken_u:
        if i in partners:
            why_of[partners[i]] = (
                f"the primed copy of #{i} {names[i]}: a site character is a character "
                "of the unitary site group, so the two stand or fall together")

    # WHAT THE FAILING DECLARATION MEASURED, in irrep names: the same naming on
    # both calls, from the site tables and the readings of the group read
    if site_tables is None:
        try:
            site_tables = _site_tables(group, assignment, set(unitary),
                                       set(standing) - set(unitary),
                                       range(len(assignment.site_orbitals)))
        except Exception:                        # noqa: BLE001 - the names are a courtesy
            site_tables = {}
    if admissible is None:
        admissible = _characters_of_subgroup(group, unitary)
    named = _mismatch_named(decls, rows, admissible, site_tables,
                            ref_orb=int(verdict.get("reference_orbital", 0)))

    def _listed(inds):
        return ", ".join(f"#{i} {names[i]}" for i in inds)

    read = ""
    if readings:
        from .groups import _tie_reason, _tie_rule_words

        read = ("\n  several readings are maximal and none contains the others -- "
                + "; ".join("{" + ", ".join(f"#{i}" for i in r) + "}" for r in readings)
                + " -- and the subgroup named above is the first by the tie rule "
                f"`realized_group` states: {_tie_rule_words(group)}. It is first since "
                f"{_tie_reason(group, readings)}.")
    why = "\n".join(f"    #{i} {names[i]}: {why_of[i]}" for i in every)
    table = "\n".join(
        f"    #{r['ind']:<4d} {r['name']:<14s} orbitals {tuple(r['orbitals'])}"
        f"   measured {_fmt_char(r['measured'])}"
        f"   declared {_fmt_char(r['declared'])}"
        for r in rows if int(r["ind"]) in set(broken_u))
    return Refusal(
        "declaration_breaks_group",
        f"the declared orbital content is not covariant under {len(broken_u)} of the "
        f"{len(unitary)} unitary operations of {group.name} #{group.number_str}, "
        f"{_counted_as(group.spinor)}: "
        + (_listed(broken_u) or "(none)")
        + (f", nor under their primed partners {_listed(primed)}" if primed else "")
        + ". A declaration says what the orbitals are, so a Hamiltonian whose hoppings "
        "do not respect it is not a wrong declaration -- it is a crystal with less "
        f"symmetry. The declared content is covariant on the {len(kept)} operations "
        + _listed(kept)
        + ", and `realized_group` descends to them and returns that group, with its own "
        f"labels, EBR rows and indicators. {not_lowered}"
        + read
        + (f"\n{named['verdict']}" if named else "")
        + (f"\n  why the declared content is not carried by each:\n{why}" if why else "")
        + (f"\n  measured vs declared:\n{table}" if table else "")
        + "\n  frame: " + "; ".join(f"#{i} = {frame[i]}" for i in standing),
        value=float(len(every)), threshold=0.0,
        context=dict({
            "group": group, "number_str": str(group.number_str),
            "subgroup": tuple(kept),
            "broken": tuple({"ind": int(i), "name": names[i], "note": why_of[i],
                             "rows": _returned_rows(r for r in rows
                                                    if int(r["ind"]) == int(i))}
                            for i in every),
            "tied": tuple(tuple(int(i) for i in r) for r in readings),
            "checks": _returned_rows(rows),
            "declarations": tuple({"orbitals": tuple(d["orbitals"]),
                                   "declared_as": d["spec"],
                                   "declared_irrep": d["label"]} for d in decls),
            "frame": dict(frame)}, **(named.get("named", {}) if named else {})))


def _joint_reason(joint, ind):
    """Why a content read together breaks one operation, in words; ``None`` where none speaks there.

    ``joint`` holds the contents the verdict stage reads over several
    multiplets at once (:func:`_verdicts_and_choices`); the first measured at
    ``ind`` is named.
    """
    for j in joint:
        if not any(int(ind) in row for row in j["theta"].values()):
            continue
        d = j["decl"]
        said = d.get("label") or repr(d.get("spec"))
        orbs = tuple(int(o) for o in d["orbitals"])
        if d.get("partial"):
            return (f"orbitals {orbs} are declared {said}, part of what their site holds, "
                    "and no labelling the hoppings leave open puts it inside the site's "
                    "content on a group holding this operation and those kept")
        return (f"orbitals {orbs} are declared {said} together, and no labelling the "
                "hoppings leave open carries that content on a group holding this "
                "operation and those kept")
    return None


def _declared_key(decls):
    """A comparable identity for one set of declared contents.

    What a reading of the solution spaces was made *about*, so that a reading
    handed down from :func:`..groups.realized_group` is used only where it
    answers the same question.
    """
    return tuple(sorted(
        (tuple(int(o) for o in d["orbitals"]),
         tuple(sorted((int(k), round(complex(v).real, 12), round(complex(v).imag, 12))
                      for k, v in d["constraints"].items())),
         bool(d.get("corep")))
        for d in decls))


def _multiplet_scale(group, layout, positions, cperm, classes, ref, blocks, *,
                     reversal=None, reversal_image=None, reach):
    r"""``{position: block}`` — one multiplet's own transport block, its phase settled.

    The scale a part of the crystal holding no one-dimensional multiplet reads
    its declared characters against, taken at the multiplet ``ref`` of that
    part, and the scale of a one-dimensional multiplet whose tables' lift is
    no representation of the group (:func:`_lifted_scale`).  The Hamiltonian
    fixes an operation's block of the multiplet, carried round its Wyckoff
    orbit (:func:`_multiplet_blocks`), only up to the free phase of the
    solution space it was read from, and a declared character is a statement
    about the representation, not about one member of a space.  So the orbit's
    own blocks are settled exactly as the build settles its matrices
    (:func:`_settle_unit`): each generator in its canonical norm and the
    time-reversal anchor, the rest propagated along the generators' word tree,
    and the multiplication table choosing the roots where positivity's choice
    does not multiply.  What is left is a representation of the group on the
    orbit, fixed up to one one-dimensional character, and that character is the
    part's to choose: its declarations are read against this scale with every
    character admitted (:func:`_covariant_subgroup`), so the frame the numbers
    are written in, the settling's own, constrains nothing.

    ``positions`` are the unitary operations read, ``cperm`` where each
    carries each multiplet, ``blocks`` the multiplets' blocks by ``.ind``;
    ``reversal`` is the position of the time reversal kept, with
    ``reversal_image`` its multiplet permutation, and ``reach`` the composition
    defect up to which noisy blocks are projected onto the nearest exact
    representation.  Returns ``{position: block}`` -- the block at the
    multiplet's image and the multiplet -- or ``None`` where the operations are
    no group, a block is missing, or the blocks settle into no representation
    of the operations: nothing is measured on this scale then.
    """
    syms = group.symmetries
    prod, _tdiff, spf = group.get_product_table(get_diff=True)
    ident = _identity_position(group)
    u_pos = sorted({int(p) for p in positions})
    if ident not in u_pos or set(_closure(u_pos, prod, ident)) != set(u_pos):
        return None
    gens = _minimal_generators(u_pos, prod, ident)
    if gens is None:
        return None
    words, parent = _bfs_words(gens, prod, ident, u_pos)
    if set(words) != set(u_pos):
        return None
    ref = int(ref)
    members = [c for c in range(int(layout._nchan)) if int(classes[c]) == int(classes[ref])]
    states = [int(a) for c in members for a in layout._chan_states[c]]
    local = {a: i for i, a in enumerate(states)}
    n = len(states)

    def _local(c):
        return np.asarray([local[int(a)] for a in layout._chan_states[int(c)]], dtype=int)

    def _raw(p, image):
        """One operation's matrix on the reference orbit, off the model's own blocks."""
        got = blocks.get(int(syms[p].ind), {})
        M = np.zeros((n, n), dtype=complex)
        for c in members:
            S = got.get(int(c))
            if S is None or int(image[c]) not in members:
                return None
            rows, cols = _local(image[c]), _local(c)
            if np.asarray(S).shape != (len(rows), len(cols)):
                return None
            M[np.ix_(rows, cols)] = S
        return M

    Dgen = [_raw(gp, cperm[gp]) for gp in gens]
    if any(M is None for M in Dgen):
        return None
    T_pos, au_of, DT = None, {}, None
    if reversal is not None:
        T_pos = int(reversal)
        DT = _raw(T_pos, reversal_image)
        if DT is None:
            return None
        au_of = {T_pos: ident}
    tree = {"gens": gens, "words": words, "parent": parent, "spf": spf, "ident": ident,
            "nstate": n, "T_pos": T_pos, "au_of": au_of}
    kept = list(u_pos) + ([T_pos] if T_pos is not None else [])
    cols = tuple(int(i) for i in _local(ref))

    def _anchor_of(p, anchor_T):
        image = reversal_image if p == T_pos else cperm[p]
        return RepAnchor(antiunitary=anchor_T,
                         rows=tuple(int(i) for i in _local(image[ref])), cols=cols)

    bad, D, _norm, _anchor_T, _DT_n, _defect, _roots = _settle_unit(
        _propagated(Dgen, DT, **tree), group=group, tree=tree, kept=kept, u_pos=u_pos,
        prod=prod, anchor_of=_anchor_of, reach=reach)
    # the build's own product-table gate: what the table settles is a
    # representation, and a set of phases that is not one is no scale
    if bad is not None or _product_table_residual(D, u_pos, syms, prod, spf) > 1e-9:
        return None
    return {p: D[p][np.ix_(_local(cperm[p][ref]), _local(ref))] for p in u_pos}


def _lifted_scale(group, layout, positions, cperm, classes, ref, blocks, *, bases,
                  intertwiners, reversal=None, reversal_image=None, reach):
    r"""``{position: block}`` -- a one-dimensional multiplet's scale, where its lift is no representation.

    A part holding a one-dimensional multiplet reads its declared characters
    against that multiplet's lift, the tables' own, carried round its Wyckoff
    orbit with the signs the hoppings give it (:func:`_transported_lifts`): the
    multiplet's own character is then the part's one-dimensional character,
    which the labelling chooses.  That holds wherever its character on its site
    group is the restriction of a one-dimensional character of the group.
    Where it is not, the lift is no representation, and the multiplet's
    character is measured: ``p_y`` at a ``3c`` site of ``P23``, whose three
    twofolds are carried onto one another by the threefolds while ``p_y`` is
    odd under two of them, is the ``B2`` of ``222`` that no character of ``T``
    restricts to.  The orbit's blocks are then settled into a representation
    as the build settles its matrices (:func:`_multiplet_scale`), the
    multiplication table choosing the signs, and that is the scale the
    declared characters are read against.

    ``ref`` is the multiplet, ``blocks`` the multiplets' blocks by ``.ind`` as
    the build assembles them, ``bases`` and ``intertwiners`` the measured
    solution spaces, the rest as for :func:`_multiplet_scale`.  Returns
    ``None`` where the lift is a representation of the operations or the
    orbit's blocks settle into none, and the settled blocks otherwise.
    """
    syms = group.symmetries
    prod, _tdiff, spf = group.get_product_table(get_diff=True)
    ident = _identity_position(group)
    u_pos = sorted({int(p) for p in positions})
    if reversal is not None and reversal_image is None:
        return None
    images = {int(p): cperm[p] for p in u_pos}
    if reversal is not None:
        images[int(reversal)] = reversal_image
    members = [int(c) for c in range(int(layout._nchan))
               if int(classes[c]) == int(classes[int(ref)])]
    # the orbit's blocks as the build assembles them: the ones handed down, the
    # signs the hoppings give the orbit relative to ``ref``, or the lift alone
    carried = None
    if len(members) > 1 and not all(int(c) in blocks.get(int(syms[p].ind), {})
                                    for p in images for c in members):
        carried = _transported_lifts(syms, list(images), images, layout, members, int(ref),
                                     bases, intertwiners, prod, ident)
    lifted = {}
    for p in images:
        ind = int(syms[p].ind)
        got = blocks.get(ind, {})
        lift = _lift_of(syms[p], layout._spinful)
        lifted[ind] = {c: got[c] if c in got else (carried or {}).get(p, {}).get(c, lift)
                       for c in members}
    states = [int(a) for c in members for a in layout._chan_states[c]]
    local = {a: i for i, a in enumerate(states)}
    raw = {}
    for p in u_pos:
        M = np.zeros((len(states), len(states)), dtype=complex)
        for c in members:
            image = int(images[p][c])
            if image not in members:      # pragma: no cover - a class is an orbit
                return None
            rows = [local[int(a)] for a in layout._chan_states[image]]
            cols = [local[int(a)] for a in layout._chan_states[c]]
            M[np.ix_(rows, cols)] = lifted[int(syms[p].ind)][c]
        raw[p] = M
    if _product_table_residual(raw, u_pos, syms, prod, spf) <= noise_floor(
            ZERO, getattr(layout, "_tol", None)):
        return None                       # the lift is a representation
    return _multiplet_scale(group, layout, u_pos, cperm, classes, int(ref), lifted,
                            reversal=reversal, reversal_image=reversal_image, reach=reach)


def _verdicts_and_choices(model, group, assignment, spin, validated, layout, decls,
                          positions, *, classes=None, ref_orb=0,
                          blocks=None, space_tol=IN_SPACE, tol=CHARACTER,
                          max_candidates=2 ** 20, reversal=None, sectors=None):
    r"""One verdict per operation, and the scalars it pins, off the solution spaces.

    The second step of the pipeline, and the only place a solution space is read
    for what it says about the orbitals.  Everything downstream — the descent by
    declaration, the representation's own discrete choices, the characters
    record — is written from what comes back here, so that a measurement is made
    once and read once.

    Per operation, inside the site blocks its own permutation allows (a member
    carrying one site's orbitals onto a site the operation does not send them to
    is evidence about the *operation*, which is :func:`_ansatz_defects`'s
    business, and no partition into multiplets absorbs it):

    * **the multiplet verdict.**  No member block-diagonal on the multiplets
      and nonsingular there means the Hamiltonian *mixes* multiplets that are
      said to be separate, and where a ``multiplets=`` states them beside a
      declaration, the operation is broken by the declaration.  Measured
      multiplets a space cannot carry are the operation's own tensor
      structure, which :func:`_ansatz_defects` reads.
    * **the relative scalar, per multiplet orbit.**  The scalar relating the
      member's block at the class's representative multiplet to that
      multiplet's own internal factor, against the anchor's — orbital 0's
      multiplet — which is exactly the freedom :func:`_build_rep` reads the
      representation off with, one scalar per multiplet orbit per
      generator.  Where every member of the space carries the same ratio the
      hoppings **pin** it and the number is measured; where the ratio varies
      over the space they leave it **free**, which is a multiplet decoupled from
      the rest, and nothing was measured there at all.
    * **the character verdict**, for every declared multiplet the operation fixes:
      the scalar relating the member's block to the multiplet's internal factor,
      normalized by the transport block of a multiplet whose factor is fixed
      beyond the member's free phase, so that what is read is in the canonical
      induced gauge the record is written in.  The hoppings fix every block up
      to one one-dimensional character per **coupled part** of the crystal —
      the multiplets they connect, directly or through others: the anchor's
      part, and the orbits of multiplets they leave free of it,
      one part where they pin one another — so a declared multiplet is read
      against a scale inside its own part.  Where the part has a
      one-dimensional multiplet that scale is its lift, the tables' own, or,
      where that lift carried round its orbit is no representation of the
      group, the multiplet's own character, its blocks settled into one
      (:func:`_lifted_scale`).  In a part of wider multiplets it is one of
      them, its blocks settled into a representation of the group as the build
      settles its matrices (:func:`_multiplet_scale`); ``reversal`` is the position of the
      time reversal that anchors those phases, ``None`` where the group keeps
      none.  The scale's own one-dimensional character is the part's to choose:
      each part's declarations ask for one character of their own, and every
      character is admitted against them (:func:`_covariant_subgroup`).  A free
      block imposes nothing, because nothing was measured.  Two multiplets of
      one part needing different scalars break the operation, and so does a
      scalar no one-dimensional character can take: not ``±1`` where the group
      keeps an antiunitary operation, which leaves only a sign, and not a unit
      phase where it keeps none, every one-dimensional character, the complex
      ones included, being a labelling there.  A declaration written in the
      site coreps states the corep content and nothing finer, so it is not read
      at an operation whose spin-1/2 lift has zero trace, where every orbital
      content lifting to it reads zero.
    * **the contents read together**: one declared over multiplets of several
      coupled parts, and one stating part of what its site holds.  Each is
      read multiplet by multiplet, against its own part's scale, at every
      operation fixing all of them, and is one requirement on those parts'
      labellings together: the declared content carried by them, or contained
      in what they carry where part of a site was declared
      (:func:`_covariant_subgroup`).  A multiplet free at an operation leaves
      that operation unread.

    An antiunitary operation is not asked: a site character is a character of
    the site group's unitary operations, and the primed copy of a broken
    operation is broken with it, which the records downstream carry.

    ``max_candidates`` bounds the one enumeration this reading can run, the
    representations tried for an orbit of sectors where the blocks are built
    here, none having been handed down (:func:`_multiplet_blocks`), exactly as
    it bounds the build's.

    ``space_tol`` is how far a member may sit outside the blocks and still
    count as being in them (:func:`_in_blocks`).  It is a *statement about the
    model*, not a knob: an exactly symmetric Hamiltonian's solution space is in
    block form to machine precision, and a numerical one's is in block form to
    its own defect.  :func:`_build_rep` therefore hands down its
    ``internal_tol`` — the same dimensionless "how far from the ideal block"
    the near-symmetric recipe loosens — so that a declaration read on a
    wannier90-scale model is measured rather than silently found unmeasurable.
    The multiplet verdict reads the declared tolerance the multiplets were
    measured at as well, where it is larger (:func:`.._tolerance.noise_floor`): the blocks
    a declared multiplet is tested in are the blocks of that measurement, so a
    frame off the ideal one by less than the declared precision holds a
    declared multiplet as it holds a measured one.  ``tol`` is how far a
    character read off the spaces may sit from a value and still be read as
    it: :data:`.._tolerance.CHARACTER` for a model built from exact numbers,
    and the declared tolerance where that is larger.  Every other threshold
    here -- a scalar that vanishes, a ratio the hoppings pin -- reads the
    declared tolerance the multiplets were measured at in the same way.

    Returns a mapping — ``"rows"`` the measured-versus-declared table in
    :attr:`AtomicLimitTable.checks`'s own shape, ``"parts"`` the coupled parts
    as sets of multiplet orbits, the anchor's first,
    ``"requirements"`` and ``"scalars"`` one mapping per part, the verdict per
    operation (``None`` where the declaration breaks it) and the raw scalar,
    ``"joint"`` the contents read together, each with what its multiplets carry
    per operation and the part each is in, ``"by_sectors"`` the contents
    declared on orbits built by induction, which the sectors alone weigh, each
    with what its multiplets carry under the representation they are assembled
    with, ``"reasons"`` one sentence per broken operation, ``"classes"``/``"nclass"``
    the multiplet orbits, ``"chosen"`` the pinned class scalar at ``(ind,
    class)``, ``"within"`` the ratio an orbit of a part free of the anchor
    holds to that part's first at ``(ind, class)``, ``"free"`` the ``(ind,
    class)`` pairs the hoppings leave open, ``"free_classes"`` the classes free
    at every operation that was measured, ``"seen"`` what was measured where,
    ``"blocks"`` the multiplets' own blocks with every orbit of sectors built
    by induction and its candidates (:func:`_multiplet_blocks`), ``"reversal"`` what time reversal
    carries on each orbit (:func:`_reversal_scalars`), and ``"anchor"``,
    ``"reference_orbital"``, ``"declared"``, ``"space_tol"`` and ``"indices"``
    — what the reading was made about, so that one handed down can be checked
    against the question being asked — or the :class:`..refusal.Refusal` the
    multiplet permutation or the block enumeration made.  ``spin`` is the
    declared :class:`..spin.SpinBasis` itself, or ``None``.
    """
    syms = group.symmetries
    positions = [p for p in positions if not bool(syms[p].time_reversal)]
    cperm = _channel_perms(model, assignment, syms, positions, layout, spin)
    if isinstance(cperm, Refusal):
        return cperm
    site_orbs = tuple(tuple(int(o) for o in orbs) for orbs in assignment.site_orbitals)
    site_layout = _multiplets_of(model, assignment, spin,
                             tuple((tuple(orbs),) for orbs in site_orbs))
    nchan = int(layout._nchan)
    chan_of = np.asarray(layout._chan_of, dtype=int)
    # THE ANCHOR every scalar is read against: orbital 0's multiplet, whichever
    # orbital the lowest-angular-momentum rule later reads the crystal from, so
    # that every build of this group reads the one reading
    ref = int(chan_of[0])
    if classes is None:
        classes, nclass = _channel_classes(cperm, positions, nchan)
    else:
        classes = np.asarray(classes, dtype=int)
        nclass = int(classes.max()) + 1 if classes.size else 0
    ref_cls = int(classes[ref]) if nclass else 0
    reps = {}
    for cls in range(nclass):
        members = np.flatnonzero(classes == cls)
        reps[cls] = ref if cls == ref_cls else int(members[0])

    measured_of = dict(blocks or {})
    multiplicity = tuple(getattr(blocks, "multiplicity", ()) or ())
    if layout._measured and not measured_of:
        got = _multiplet_blocks(model, group, assignment, spin, validated, layout,
                                [p for p, s in enumerate(syms)
                                 if int(s.ind) in set(validated.indices)],
                                ref_chan=ref, max_candidates=int(max_candidates))
        if isinstance(got, Refusal):
            return got
        measured_of = got
        multiplicity = tuple(got.multiplicity)

    lattice = np.asarray(group.lattice, dtype=float)
    chan_states, su2 = layout._chan_states, layout._spinful
    want_of = {int(c): d for d in decls for c in d["channels"]}
    bases = getattr(validated, "_bases", None) or {}
    # the multiplets' blocks are tested at the tolerance they were measured at
    block_tol = getattr(layout, "_tol", None)
    in_tol = noise_floor(space_tol, block_tol)
    zero, spread_tol = noise_floor(ZERO, block_tol), noise_floor(SPREAD, block_tol)
    rows, broken, why = [], {}, {}
    chosen, free, seen = {}, set(), {}
    # the in-block solution spaces the character verdict reads, once the
    # coupled parts they are read in are known, and what each orbit's block
    # is across each space
    spaces, carried = {}, {}
    for p in positions:
        ind = int(syms[p].ind)
        basis = bases.get(ind)
        if basis is None or len(np.asarray(basis)) == 0:
            member = validated._intertwiners.get(ind)
            if member is None:
                continue
            basis = np.asarray(member)[None, :, :]
        basis = np.asarray(basis)
        nstate = int(basis.shape[1])
        if basis.shape[0] * nstate * nstate * 16 > _READOFF_MAX_BYTES:
            continue                      # the read-off's own ceiling: not measured
        site_image = _channel_permutation(assignment, p, model, site_layout)
        space = _in_blocks(basis, _ansatz_blocks(site_layout._chan_states, site_image, 1),
                           tol=in_tol)
        if not len(space):
            continue                      # the operation's own tensor structure
        image = cperm[p]
        blocks_d = _ansatz_blocks(chan_states, image, 1)
        V = _in_blocks(space, blocks_d, tol=in_tol)
        ok, unfilled = (False, list(range(len(blocks_d)))) if not len(V) else \
            _filled_check(V, blocks_d, tol=block_tol)
        if not ok:
            if not decls or layout.determined_by != "declaration":
                # nothing stated these multiplets: the Hamiltonian measured them,
                # so multiplets the space cannot carry are the operation's own
                # tensor structure and `_ansatz_refuted` owns that verdict, not
                # this one
                continue
            touched = sorted({int(c) for c in unfilled}
                             | {int(image[int(c)]) for c in unfilled})
            named = [c for c in touched if c in want_of] or touched
            broken[ind] = None
            why[ind] = (
                "the measured solution space holds no nonsingular member that is "
                "block-diagonal on the declared multiplets: the hoppings mix "
                + ", ".join(f"the multiplet of orbitals {tuple(layout._chan_orbs[c])}"
                            for c in named)
                + " with the rest of the site")
            continue
        lift = _lift_of(syms[p], su2)
        blocks_here = measured_of.get(ind, {})
        lift_of = [blocks_here.get(int(c),
                                   None if layout._chan_kind[int(c)] == "measured" else lift)
                   for c in range(nchan)]

        # ---- the choice, one scalar per multiplet orbit -----------------------
        S_ref = lift_of[ref]
        if S_ref is not None:
            rrows = np.asarray(chan_states[int(image[ref])], dtype=int)
            rcols = np.asarray(chan_states[ref], dtype=int)
            nu_c = np.asarray([_proportionality(V[j][np.ix_(rrows, rcols)], S_ref)[0]
                               for j in range(len(V))])
            base = float(np.abs(nu_c).max()) if nu_c.size else 0.0
            if base > zero:
                for cls in range(nclass):
                    c0 = reps[cls]
                    S0 = lift_of[c0]
                    if S0 is None:
                        continue
                    crow = np.asarray(chan_states[int(image[c0])], dtype=int)
                    ccol = np.asarray(chan_states[c0], dtype=int)
                    t = np.asarray([_proportionality(V[j][np.ix_(crow, ccol)], S0)[0]
                                    for j in range(len(V))])
                    sig = complex(np.vdot(nu_c, t) / np.vdot(nu_c, nu_c))
                    spread = float(np.abs(t - sig * nu_c).max())
                    seen.setdefault(ind, set()).add(int(cls))
                    carried.setdefault(ind, {})[int(cls)] = t
                    if spread > spread_tol * max(float(np.abs(t).max()), base, 1.0):
                        free.add((ind, int(cls)))
                    else:
                        chosen[(ind, int(cls))] = sig
        if want_of:
            spaces[p] = (V, image, lift, lift_of)

    # A class is free where NOTHING measured it: free at every operation whose
    # solution space spoke about it.  That is a decoupled multiplet -- what it
    # carries relative to the anchor is a question this Hamiltonian cannot
    # answer, on this site or any other -- and it is the labelling's to settle
    # (:func:`_labelling`), as a declaration states it.
    free_classes = frozenset(
        cls for cls in range(nclass)
        if any(cls in got for got in seen.values())
        and all((ind, cls) in free for ind, got in seen.items() if cls in got))
    # IDENTICAL COPIES, and uncoupled orbitals a symmetry exchanges that the
    # group's characters do not carry, are one part of the crystal each, an
    # orbit of sectors the hoppings couple to nothing else: its multiplets are
    # assembled with the blocks of one representation of it
    # (:func:`_induced_blocks`), and which one it carries is the labelling's
    # choice among all of them.  So no scalar is read between its multiplets,
    # which carry those blocks as they stand -- a scalar read at one multiplet
    # of an orbit of multiplets lying on several sectors says nothing of the
    # others -- and it is pinned to the anchor where it holds the anchor's
    # multiplet and free of it otherwise.
    induced_classes = []
    for rec in multiplicity:
        if rec.get("induced") is None or not rec["n_admitted"]:
            continue
        own = frozenset(int(classes[int(c)]) for c in rec["multiplets"])
        induced_classes.append(own)
        if ref_cls in own:
            free_classes = free_classes - own
            for ind in seen:
                for cls in own:
                    free.discard((int(ind), int(cls)))
                    chosen[(int(ind), int(cls))] = 1.0 + 0.0j
        else:
            free_classes = free_classes | own

    # THE COUPLED PARTS.  The hoppings fix every multiplet's block up to one
    # one-dimensional character per part of the crystal they connect: the
    # anchor's part, every orbit they pin to it, and the orbits
    # they leave free of it, which are one part where they pin one another --
    # two orbits whose blocks keep one ratio across a solution space.  A
    # declaration is a constraint on its own part's character, read against a
    # scale inside that part, so a part the hoppings never reach measures
    # nothing about it.
    joined = {int(cls): int(cls) for cls in free_classes}

    def _joined(cls):
        while joined[cls] != cls:
            cls = joined[cls]
        return cls

    for got in carried.values():
        both = [(int(cls), t) for cls, t in sorted(got.items())
                if int(cls) in joined and float(np.abs(t).max()) > zero]
        for (a, ta), (b, tb) in itertools.combinations(both, 2):
            sig = complex(np.vdot(ta, tb) / np.vdot(ta, ta))
            if float(np.abs(tb - sig * ta).max()) <= spread_tol * max(
                    float(np.abs(tb).max()), float(np.abs(ta).max()), 1.0):
                joined[max(_joined(a), _joined(b))] = min(_joined(a), _joined(b))
    for own in induced_classes:
        if ref_cls in own:
            continue
        first = min(own)
        for cls in own:
            a, b = _joined(int(first)), _joined(int(cls))
            if a != b:
                joined[max(a, b)] = min(a, b)
    apart = {}
    for cls in sorted(joined):
        apart.setdefault(_joined(cls), set()).add(int(cls))
    parts = ([frozenset(cls for cls in range(nclass) if cls not in free_classes)]
             + [frozenset(apart[k]) for k in sorted(apart)])
    part_of = {int(cls): i for i, part in enumerate(parts) for cls in part}
    # ... and the ratio every member of a space holds between an orbit of such
    # a part and the part's first, where the operation's space speaks of both;
    # one between the multiplets of an orbit built by induction, which carry
    # their blocks as they stand
    within = {}
    for ind, got in carried.items():
        for part in parts[1:]:
            lead_cls = min(part)
            if any(part == own for own in induced_classes):
                for cls in part:
                    if int(cls) != int(lead_cls):
                        within[(int(ind), int(cls))] = 1.0 + 0.0j
                continue
            t0 = got.get(int(lead_cls))
            if t0 is None or float(np.abs(t0).max()) <= zero:
                continue
            for cls in part:
                if int(cls) != int(lead_cls) and int(cls) in got:
                    within[(int(ind), int(cls))] = complex(
                        np.vdot(t0, got[int(cls)]) / np.vdot(t0, t0))
    # EVERY DECLARATION IS READ ONCE, in one of two ways.  One content on the
    # multiplets of one coupled part is a requirement on that part's
    # one-dimensional character, operation by operation.  One spread over
    # several parts, or one stating part of what its site holds, is one
    # requirement on what those parts carry together: read here per multiplet,
    # at every operation fixing its site, and met where one labelling of each
    # part it spans carries it (:func:`_covariant_subgroup`).
    part_of_decl, joint_decls, on_induced = {}, [], []
    induced = frozenset(cls for own in induced_classes for cls in own)
    for d in decls:
        held = {part_of.get(int(classes[int(c)])) for c in d["channels"]}
        if any(int(classes[int(c)]) in induced for c in d["channels"]):
            # a content declared on an orbit built by induction: what it
            # carries is no measurement, so nothing is read here, and the
            # sectors' own reading of every representation of it is the verdict
            part_of_decl[id(d)] = None
            on_induced.append(d)
            continue
        if len(held) == 1 and not d.get("partial"):
            part_of_decl[id(d)] = held.pop()
        else:
            part_of_decl[id(d)] = None
            if d["channels"]:
                joint_decls.append(d)
    spoken = ({i for i in part_of_decl.values() if i is not None}
              | {part_of.get(int(classes[int(c)])) for d in joint_decls for c in d["channels"]})
    spoken = {int(i) for i in spoken if i is not None}

    # THE SCALE each part's characters are read against.  A member of a
    # solution space is a representation matrix times one free complex number,
    # and the scalar a declared multiplet needs is only a statement about the
    # physics once that number is fixed -- in the gauge the record is written
    # in, which is the canonical induced form: one multiplet's transport block
    # carries that multiplet's own internal factor and nothing more.  A
    # one-dimensional multiplet's factor is PINNED, the tables' own lift, and
    # where the part has one that is its scale -- unless that lift, carried
    # round its orbit, is no representation of the group, where the
    # multiplet's own character is measured and its settled blocks are the
    # scale (:func:`_lifted_scale`).  A wider multiplet's block is
    # the model's own and carries the member's free phase, so in a part of
    # multiplets the scale is one of its own multiplets' blocks, settled into
    # a representation of the group; with no antiunitary operation to anchor
    # them they are fixed only up to a unit phase per generator, which the
    # part's one-dimensional character can then carry.  Either way, what the
    # part's character may take is a unit phase where the group keeps no
    # antiunitary operation -- every one-dimensional character, the complex
    # ones included, is then a labelling nothing measured tells apart -- and a
    # sign where it keeps one.
    reversal_image = None
    if reversal is not None:
        got = _channel_perms(model, assignment, syms, [int(reversal)], layout, spin)
        reversal_image = None if isinstance(got, Refusal) else got[int(reversal)]
    scales, roots, wider = {}, {}, []
    for i in sorted(spoken):
        lead = _lead_of([c for c in range(nchan) if part_of.get(int(classes[c])) == i], ref)
        tables = [c for c in range(nchan) if part_of.get(int(classes[c])) == i
                  and layout._chan_kind[c] == "tables"]
        if tables:
            scale_chan = lead if lead in tables else int(tables[0])
            # the lift, where it is a representation; the one-dimensional
            # multiplet's own character, settled, where it is not
            scales[i] = (scale_chan, _lifted_scale(
                group, layout, positions, cperm, classes, scale_chan, measured_of,
                bases=bases, intertwiners=validated._intertwiners, reversal=reversal,
                reversal_image=reversal_image, reach=min(1e-2, 100.0 * in_tol)))
            roots[i] = reversal is None
        elif layout._measured and measured_of:
            wider.append((i, lead))
    if wider:
        for i, lead in wider:
            if reversal is not None and reversal_image is None:
                break
            scale_of = _multiplet_scale(group, layout, positions, cperm,
                                        classes, lead, measured_of, reversal=reversal,
                                        reversal_image=reversal_image,
                                        reach=min(1e-2, 100.0 * in_tol))
            if scale_of is not None:
                scales[i] = (lead, scale_of)
                roots[i] = reversal is None

    # ---- the character verdict, for what was declared -----------------------
    def _theta(x, V, lift, nu, floor):
        """Multiplet ``x``'s character against its part's scale ``nu``, or ``None`` where it is free."""
        cols = np.asarray(chan_states[int(x)], dtype=int)
        if layout._chan_kind[int(x)] == "measured":
            # the multiplet's ORBITAL character: the block's trace with the
            # spinor divided out, atom by atom, which is what the declared
            # single-valued irrep is a statement about
            atoms = _atom_blocks(chan_states[int(x)], su2)
            f = np.asarray([sum(_proportionality(V[j][np.ix_(a, a)], lift)[0]
                                for a in atoms) for j in range(len(V))])
        else:
            f = np.asarray([_proportionality(V[j][np.ix_(cols, cols)], lift)[0]
                            for j in range(len(V))])
        theta = complex(np.vdot(nu, f) / np.vdot(nu, nu))
        gap = float(np.abs(f - theta * nu).max())
        if gap > spread_tol * max(float(np.abs(f).max()), floor, 1.0):
            return None                   # free: the hoppings do not measure it
        return theta

    pre, joint_theta = [], {}
    for p in positions:
        if p not in spaces:
            continue
        V, image, lift, lift_of = spaces[p]
        ind = int(syms[p].ind)
        name = _sitechars.operation_name(syms[p], lattice)
        nus = {}
        for i, (scale_chan, scale_of) in scales.items():
            if scale_of is None:
                S_norm = lift_of[scale_chan]
                if S_norm is None:        # pragma: no cover - a guard, not a path
                    continue
            elif p in scale_of:
                S_norm = scale_of[p]
            else:
                continue                  # nothing to fix the scale against
            nrows = np.asarray(chan_states[int(image[scale_chan])], dtype=int)
            ncols = np.asarray(chan_states[scale_chan], dtype=int)
            nu = np.asarray([_proportionality(V[j][np.ix_(nrows, ncols)], S_norm)[0]
                             for j in range(len(V))])
            floor = float(np.abs(nu).max()) if nu.size else 0.0
            if floor > zero:
                nus[i] = (nu, floor)
        silent = abs(complex(np.trace(lift))) <= zero
        # ONE CONTENT OVER SEVERAL PARTS, OR PART OF A SITE'S: per multiplet,
        # what it carries against its own part's scale, at every operation that
        # fixes all of its multiplets
        for d in joint_decls:
            if ind not in d["constraints"] or (d.get("corep") and silent):
                continue                  # a corep content says nothing here
            if any(int(image[int(x)]) != int(x) for x in d["channels"]):
                continue
            got = {}
            for x in d["channels"]:
                i = part_of.get(int(classes[int(x)]))
                if i not in nus:
                    break                 # no scale in its part at this operation
                theta = _theta(x, V, lift, *nus[i])
                if theta is None:
                    break
                got[int(x)] = theta
            if len(got) != len(d["channels"]):
                continue
            for x, theta in got.items():
                joint_theta.setdefault(id(d), {}).setdefault(int(x), {})[ind] = theta
        if not nus:
            continue
        read = set()
        for c in sorted(want_of):
            if int(image[int(c)]) != int(c):
                continue
            d = want_of[int(c)]
            # a content declared over several multiplets is read once, as the
            # sum of theirs
            if id(d) in read or part_of_decl[id(d)] is None:
                continue
            read.add(id(d))
            if part_of_decl[id(d)] not in nus:
                continue                  # no scale in its part at this operation
            if ind not in d["constraints"] or (d.get("corep") and silent):
                continue                  # a corep content says nothing here
            nu, floor = nus[part_of_decl[id(d)]]
            thetas = []
            for x in d["channels"]:
                if int(image[int(x)]) != int(x):
                    break
                theta = _theta(x, V, lift, nu, floor)
                if theta is None:
                    break
                thetas.append(theta)
            if len(thetas) != len(d["channels"]):
                continue
            pre.append({"orbitals": tuple(d["orbitals"]), "ind": ind, "name": name,
                        "measured": thetas[0] if len(thetas) == 1 else complex(sum(thetas)),
                        "declared": complex(d["constraints"][ind]),
                        "part": int(part_of_decl[id(d)])})

    for r in pre:
        theta, want = complex(r["measured"]), complex(r["declared"])
        rows.append(dict(r, required=None if abs(theta) <= tol else want / theta))
    # EVERY DECLARATION CONSTRAINS ITS OWN PART'S CHARACTER, wherever it sits:
    # what each operation asks of that character, part by part.  An operation
    # the multiplets break is broken in every part, and one two declarations
    # of a part ask different values of is broken there; which characters meet
    # the rest is the subgroup search's question, never an anchor's.
    requirements, scalars = [], []
    for i in range(len(parts)):
        req, raw, said = dict(broken), {}, {}
        _needs_per_operation([r for r in rows if r["part"] == i], req, raw, said,
                             tol=tol, roots=roots.get(i, False))
        requirements.append(req)
        scalars.append(raw)
        for ind, sentence in said.items():
            why.setdefault(int(ind), sentence)
    # ... and every content read together over its multiplets, per multiplet,
    # with what reads it on a smaller group: the site group the operations
    # kept leave it, and the trace of each operation's own spin-1/2 lift
    kept_inds = {int(syms[p].ind) for p in positions}
    lift_trace = {int(syms[p].ind): complex(np.trace(_lift_of(syms[p], su2)))
                  for p in positions}
    joint = []
    for d in joint_decls:
        theta = joint_theta.get(id(d))
        if not theta:
            continue
        site = int(d["site"])
        joint.append({"decl": d, "theta": theta,
                      "part": {int(c): int(part_of[int(classes[int(c)])])
                               for c in d["channels"]},
                      "site_group": tuple(int(i) for i in assignment.site_group[site]
                                          if int(i) in kept_inds),
                      "table_of": _SiteTableOf(group, assignment, site,
                                               corep=bool(d.get("corep"))),
                      "lift_trace": lift_trace})
    # ... and every content declared on an orbit built by induction, which the
    # sectors alone weigh: what its multiplets carry under the representation
    # they are assembled with, at every operation fixing all of them, so that
    # an operation it breaks is said of its own orbitals
    by_sectors = []
    held_by = {int(c): rec for rec in multiplicity
               if rec.get("induced") is not None and rec["n_admitted"]
               for c in rec["multiplets"]}
    for d in on_induced:
        if not all(int(x) in held_by for x in d["channels"]):
            continue
        theta = {}
        for p in positions:
            ind = int(syms[p].ind)
            if ind not in d["constraints"] or any(int(cperm[p][int(x)]) != int(x)
                                                   for x in d["channels"]):
                continue
            if d.get("corep") and abs(complex(np.trace(_lift_of(syms[p], su2)))) <= zero:
                continue                  # a corep content says nothing here
            for x in d["channels"]:
                got = held_by[int(x)]["candidates"][0]["contents"].get(int(x), {})
                if ind in got:
                    theta.setdefault(int(x), {})[ind] = complex(got[ind])
        if theta:
            by_sectors.append({"decl": d, "theta": theta})

    # WHAT TIME REVERSAL CARRIES ON EACH ORBIT OF MULTIPLETS, read here once: its
    # own internal factor on every multiplet, times the scalar its solution space
    # pins on each orbit -- one on orbitals that are real functions
    # (:func:`_reversal_scalars`)
    reversal_scalars = None
    if reversal is not None and reversal_image is not None:
        t_ind = int(syms[int(reversal)].ind)
        basis_t = bases.get(t_ind)
        if basis_t is not None and len(np.asarray(basis_t)):
            basis_t = np.asarray(basis_t)
            n_t = int(basis_t.shape[1])
            lift_t = _lift_of(syms[int(reversal)], su2)
            got_t = measured_of.get(t_ind, {})
            lifts_t = [got_t.get(int(c), None if layout._chan_kind[int(c)] == "measured"
                                 else lift_t) for c in range(nchan)]
            if (basis_t.shape[0] * n_t * n_t * 16 <= _READOFF_MAX_BYTES
                    and all(S is not None for S in lifts_t)):
                reversal_scalars = _reversal_scalars(
                    basis_t, _ansatz_blocks(chan_states, reversal_image, 1), lifts_t,
                    classes, ref, tol=noise_floor(_IN_BLOCKS, block_tol))
                # an orbit built by induction carries time reversal's blocks of
                # its own representation as they stand
                if reversal_scalars is not None and induced_classes:
                    reversal_scalars = np.array(reversal_scalars, dtype=complex)
                    for own in induced_classes:
                        for cls in own:
                            reversal_scalars[int(cls)] = 1.0 + 0.0j
    # WHAT A DECLARATION ASKS OF THE SECTORS, on any subgroup: read off the
    # conserved matrices the multiplets carry and what this round's operations
    # do to them (:class:`.._sectors.SectorSearch`)
    search = None
    if decls:
        search = _declaration_search(model, group, assignment, spin, validated, layout,
                                     decls, positions, reversal=reversal, sectors=sectors,
                                     tol=tol, max_candidates=max_candidates)
    return {"rows": tuple(rows), "parts": tuple(parts),
            "requirements": tuple(requirements), "scalars": tuple(scalars),
            "joint": tuple(joint), "by_sectors": tuple(by_sectors),
            "reasons": why,
            "classes": classes, "nclass": int(nclass), "chosen": chosen, "within": within,
            "free": frozenset(free), "free_classes": free_classes,
            "seen": {int(k): frozenset(v) for k, v in seen.items()},
            "anchor": int(ref), "reference_orbital": int(ref_orb),
            "declared": _declared_key(decls), "space_tol": float(space_tol),
            "indices": frozenset(int(syms[p].ind) for p in positions),
            "blocks": _MultipletBlocks(measured_of, multiplicity=multiplicity),
            "reversal": reversal_scalars, "search": search}


def _declaration_search(model, group, assignment, spin, validated, layout, decls, positions,
                        *, reversal=None, sectors=None, tol=CHARACTER, max_candidates=2 ** 20):
    r"""What a declaration asks of the sectors on any subgroup, or ``None`` where they read nothing.

    The sectors read where the model entered (``sectors``, read here where none
    is handed in), as the round's multiplets carry them
    (:func:`.._sectors.carried_reading`), and what the round's operations do to
    them (:func:`.._sectors.sector_action`); ``reversal`` is time reversal's
    position, which the search reads only where it is the pure one.
    """
    from . import _sectors

    eps = getattr(layout, "_tol", None) or None
    reading = (sectors if sectors is not None else
               _sectors.read_sectors(model, group, assignment, spin, validated, tol=eps))
    if reading is None:
        return None
    carried = _sectors.carried_reading(reading, layout, tol=eps)
    action = _sectors.sector_action(carried, model, group, assignment, spin, validated, tol=eps)
    pure = None
    if reversal is not None:
        sym = group.symmetries[int(reversal)]
        if (np.array_equal(np.rint(np.asarray(sym.rotation)).astype(int), np.eye(3, dtype=int))
                and np.allclose(np.asarray(sym.translation) % 1.0, 0.0, atol=ZERO)):
            pure = int(reversal)
    search = _sectors.SectorSearch(carried, action, group, assignment, decls, positions,
                                   reversal=pure, tol=tol, max_candidates=max_candidates)
    return search if search.complete else None


def _reference_declaration(decls, ref_orb):
    """The declaration made **about the reference orbital**, or ``None``.

    The reference orbital (:func:`_build_rep`'s ``reference_orbital``) is where
    the lowest-angular-momentum rule reads the crystal first, and its own
    declaration is read first among the declarations for the same reason; it
    constrains the one-dimensional character as every declaration does, and no
    more.  A declaration that names the reference orbital inside a multiplet is
    one of them — what the multiplet is *is* what the reference orbital is.
    """
    return next((d for d in decls if int(ref_orb) in
                 {int(o) for o in d["orbitals"]}), None)


def _needs_per_operation(rows, req, raw, why, *, tol=CHARACTER, roots=False):
    """Which operations the declared content is carried by, read off the measured table.

    One reading of the measured-versus-declared table ``rows`` of one coupled
    part of the crystal, written into ``req`` (the scalar ``λ(g)`` the part's
    one-dimensional character has to take for its declared content to be what
    the hoppings carry, ``None`` where the declaration breaks the operation
    outright), ``raw`` (that scalar before it is rounded onto ``±1``) and
    ``why`` (one sentence per broken operation, in the words of the physics:
    what was declared, and what the hoppings carry).  An operation already in
    ``req`` was broken by the multiplets and is left alone.

    ``roots`` is a table read in a group with no antiunitary operation, where
    every one-dimensional character, the complex ones included, is a labelling
    nothing measured tells apart: a scalar of unit modulus is then the value of
    one and kept as it is, where an antiunitary operation leaves only a sign.

    Nothing is decided here and nothing is measured: the comparison was made on
    the solution spaces (:func:`_verdicts_and_choices`), and this turns those
    numbers into one number per operation.  Which characters meet them is
    :func:`_covariant_subgroup`'s question.
    """
    by_ind: dict = {}
    for r in rows:
        by_ind.setdefault(int(r["ind"]), []).append(r)
    for ind, group_rows in by_ind.items():
        if ind in req:                    # the multiplets already broke this operation
            continue
        need, first = None, None
        for r in group_rows:
            if need == "broken":
                break
            orbs = tuple(r["orbitals"])
            if r["required"] is None:
                if abs(complex(r["declared"])) > tol:
                    need = "broken"
                    why[ind] = (f"orbitals {orbs} are declared "
                                f"{_fmt_char(r['declared'])} under this operation, and "
                                "the character the hoppings carry them with vanishes")
                continue
            lam = complex(r["required"])
            if need is None:
                need, first = lam, r
            elif abs(need - lam) > tol:
                # both measured characters are nonzero here: a row whose
                # measured character vanishes carries no requirement
                ratio = complex(r["measured"]) / complex(first["measured"])
                base = complex(first["declared"])
                said = (f"state {_fmt_char(complex(r['declared']) / base)}"
                        if abs(base) > tol else "are not in that ratio")
                why[ind] = (f"relative to orbitals {tuple(first['orbitals'])}, the "
                            f"hoppings carry orbitals {orbs} with {_fmt_char(ratio)} "
                            f"under this operation, and the declarations "
                            f"({tuple(first['orbitals'])} "
                            f"{_fmt_char(first['declared'])}, {orbs} "
                            f"{_fmt_char(r['declared'])}) {said}")
                need = "broken"
        if need is None:
            continue
        if need == "broken":
            req[ind] = None
            continue
        raw[ind] = complex(need)
        if roots and min(abs(need - 1.0), abs(need + 1.0)) > tol:
            if abs(abs(need) - 1.0) > tol:
                req[ind] = None
                why[ind] = (f"orbitals {tuple(first['orbitals'])} are declared "
                            f"{_fmt_char(first['declared'])} under this operation and "
                            f"the hoppings carry them with {_fmt_char(first['measured'])}, "
                            f"a ratio of {_fmt_char(need)} that is not a phase, and so "
                            "the value of no one-dimensional character")
            else:
                req[ind] = complex(need) / abs(need)
        elif min(abs(need - 1.0), abs(need + 1.0)) > tol:
            req[ind] = None
            why[ind] = (f"orbitals {tuple(first['orbitals'])} are declared "
                        f"{_fmt_char(first['declared'])} under this operation and the "
                        f"hoppings carry them with {_fmt_char(first['measured'])}, a "
                        f"ratio of {_fmt_char(need)} that is not a sign: an antiunitary "
                        "operation of the group leaves only a sign between two "
                        "labellings of one crystal")
        else:
            req[ind] = complex(round(float(np.real(need))))
    return req, raw, why


def _site_tables(group, assignment, kept_u, kept_au, sites):
    """``{site: SiteGroupTable}`` for the measured site groups of ``sites``.

    Sites that share a site group (every site of one orbit does, up to
    conjugation, and symmetry-equivalent sites often literally do) share one
    computed table.  Each row is then named by the band representation it
    induces (:func:`..ebr._site_irrep_labels`), which is a statement about the
    site's Wyckoff position as well as about its group, so two sites share a
    table only when they share both.

    A row is named in the valuedness the model's own states carry: a spinless
    model's site irreps are the single-valued ones the EBR entries at that
    position are keyed by, and a spinful model's are the double-valued site
    coreps the group's own double-valued entries are keyed by
    (:func:`..ebr._site_corep_labels`).
    """
    letters = tuple(getattr(assignment, "wyckoff_tables", ()) or ())
    out, cache = {}, {}
    for s in sites:
        ind_u = tuple(int(i) for i in assignment.site_group[s] if int(i) in kept_u)
        ind_au = tuple(int(i) for i in assignment.site_group[s] if int(i) in kept_au)
        letter = letters[int(s)] if int(s) < len(letters) else ""
        key = (ind_u, ind_au, letter)
        if key not in cache:
            table = _site_group_table(group, ind_u, antiunitary=ind_au)
            cache[key] = _named_from_tables(group, assignment, int(s), table)
        out[int(s)] = cache[key]
    return out


def _site_group_table(group, indices, antiunitary=()):
    """A site group's character table, each row under the tables' name for that irrep.

    :func:`.._sitechars.site_group_table` computes the table, and each row is
    named by one rule.  The tables' name for that irrep of the site's point
    group, read at the origin of a symmorphic group
    (:func:`..ebr._point_group_irreps`), comes first, so that one irrep has one
    name at every site of a model (``named_by = "point group"``).  Where those
    names leave the row open, because they follow an orientation no table fixes
    at this site, the row keeps the name its Mulliken symbol gives it
    (``"Mulliken"``), where that is a name rather than a placeholder
    (:func:`.._sitechars.is_placeholder`).  A row neither names is unnamed.
    """
    from . import ebr as _ebr

    table = _sitechars.site_group_table(group, indices, antiunitary=antiunitary)
    names = _ebr._point_group_irreps(group, table) or ("",) * len(table.labels)
    taken = {name for name in names if name}
    labels, named_by = [], []
    for name, own in zip(names, table.labels):
        if name:
            labels.append(name)
            named_by.append("point group")
        elif own and not _sitechars.is_placeholder(own) and own not in taken:
            labels.append(own)
            named_by.append("Mulliken")
        else:
            labels.append("")
            named_by.append("unnamed")
    return replace(table, labels=tuple(labels), named_by=tuple(named_by))


class _SiteTableOf:
    """The character table of one site's group inside a subgroup, built once per subgroup.

    What part of a site's content is read against on a smaller group
    (:func:`_covariant_subgroup`): a containment is a decomposition on the site
    group the subgroup leaves, which needs only its characters.  A content
    written in the site coreps is compared as coreps, and that needs the rows
    named as the coreps the states carry (:func:`_site_tables`), the
    antiunitary operations the subgroup keeps at the site gluing their
    complex pairs.  Called with the ``.ind`` of the site group's unitary
    operations inside the subgroup; ``None`` where they carry no table.
    """

    def __init__(self, group, assignment, site, *, corep=False):
        self.group, self.assignment = group, assignment
        self.site, self.corep = int(site), bool(corep)
        self._built = {}

    def __call__(self, indices):
        key = frozenset(int(i) for i in indices)
        if key not in self._built:
            own = [int(i) for i in self.assignment.site_group[self.site]]
            try:
                if self.corep:
                    partners = _primed_partners(self.group, own)
                    primed = {partners[i] for i in key if i in partners}
                    self._built[key] = _site_tables(self.group, self.assignment, set(key),
                                                    primed, [self.site])[self.site]
                else:
                    self._built[key] = _site_group_table(
                        self.group, tuple(i for i in own if i in key))
            except _sitechars.DeclarationError:   # pragma: no cover - a site group closes
                self._built[key] = None
        return self._built[key]


def _named_from_tables(group, assignment, site, table):
    """``table`` with the rows the EBR entries at the site's position name.

    A row the tables reach is named by the band representation it induces: the
    entry at the site's own Wyckoff position whose content it reproduces
    (``named_by = "tables"``).  A row those entries do not name keeps the site
    point group's own name, ``named_by = "point group"``: the tables' own name
    for that irrep of the point group, read at the origin of a symmorphic group,
    single-valued for a spinless row (:func:`..ebr._point_group_irreps`) and
    double-valued for a spinful one (:func:`..ebr._point_group_names`).  A
    spinless row those names leave open keeps its Mulliken symbol, ``named_by =
    "Mulliken"``, where that is a name (:func:`_site_group_table`).  A row
    nothing names is left **empty**, ``named_by = "unnamed"``, and its generator
    characters are the whole statement: naming is a reading of the tables and
    never a condition for a build.

    Where the states carry spin the returned name is the double-valued site
    corep, and the row's **orbital** factor — single-valued, and what a
    declaration states — is named at the same position by the same group's
    single-valued entries (:attr:`.._sitechars.SiteGroupTable.declarable`).
    Both come off the tables; the second is never returned.
    """
    from . import ebr as _ebr

    if group.spinor:
        # The states at this site carry the row times the operations' spin-1/2
        # lift, and it is that -- the double-valued site corep -- the tables
        # name.  A declaration states the ORBITAL factor instead, which is the
        # row itself: single-valued, and named at the same position by the same
        # group's single-valued entries.  Both names are read from the tables,
        # from the group's two table files; only the corep is ever returned.
        labels, named_by = _ebr._site_corep_labels(group, assignment, int(site), table)
        orbital = _ebr._site_orbital_labels(group, assignment, int(site), table)
        return replace(table, labels=labels, named_by=named_by,
                       declarable=tuple(name or own for name, own
                                        in zip(orbital, table.labels)))
    # the single-valued names: one tables entry per row, and the tables' own
    # name for the point group's irrep where the entries do not name the row,
    # so a row needs no second spelling
    labels, named_by = _ebr._site_irrep_labels(group, assignment, int(site), table)
    return replace(table, labels=labels, named_by=named_by,
                   declarable=("",) * len(labels))


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


def _measure_channel_characters(D, syms, kept_u, cperm, layout, chan):
    """``({ind: chi(g)}, residual)`` of one multiplet over the operations that fix it.

    This reads **the representation**, not the Hamiltonian: ``D`` is what
    :func:`_build_rep` settled on the scalars the verdict stage pinned, before
    any labelling is chosen, so the number that comes back is the character
    a labelling then multiplies, with the product table anchoring it.  No
    verdict is made here — the declaration's verdict was made once, on the
    solution spaces (:func:`_verdicts_and_choices`), and this writes the
    record.

    On a **one-dimensional** multiplet the block of a fixed multiplet is ``chi(g)``
    times the operation's own internal (spin) factor, and ``chi`` is extracted
    by least-squares proportionality — the spinor part is *divided out*, which
    is what makes the declaration a statement about the **orbital** factor and
    what makes ``chi`` a single-valued character of the site point group.  On a
    **multi-dimensional** multiplet the block *is* the site representation and
    the character is its trace — read the same way, atom by atom
    (:func:`_atom_blocks`), so that the spinor is divided out there too and the
    character stays a single-valued character of the site point group.  On a
    spinless model the two are the same number.

    ``layout._spinful`` says whether a one-dimensional multiplet carries the
    operations' SU(2) lift: true for a native spinful model, and equally true
    for a spinless model whose orbital index was **declared** to encode spin
    (:class:`..spin.SpinBasis`), where the multiplet is a declared ``(up, dn)``
    pair and the block is the same ``2x2`` object.  It is that flag, not
    ``nspin``, that decides.
    """
    chi, worst = {}, 0.0
    states = np.asarray(layout._chan_states[chan], dtype=int)
    measured = layout._chan_kind[int(chan)] == "measured"
    atoms = _atom_blocks(layout._chan_states[chan], layout._spinful) if measured else ()
    for p in kept_u:
        if int(cperm[p][chan]) != int(chan):
            continue
        S = _lift_of(syms[p], layout._spinful)
        if measured:
            chi[int(syms[p].ind)] = complex(sum(
                _proportionality(D[p][np.ix_(a, a)], S)[0] for a in atoms))
            continue
        B = D[p][np.ix_(states, states)]
        rho, res = _proportionality(B, S)
        chi[int(syms[p].ind)] = complex(rho)
        worst = max(worst, float(res))
    return chi, worst


def _spec_key(spec):
    """A comparable identity for one declaration value.

    Labels compare case-insensitively; a **generator character mapping**
    compares by its content rather than by object identity, because the two
    members of a declared spinor pair can be handed the same mapping written
    twice.  Anything else falls
    back to identity, which is what the ordered-list grouping does.
    """
    if isinstance(spec, str):
        return ("label", spec.lower())
    if isinstance(spec, dict):
        # sorted on the key alone: a complex value is not orderable, and two
        # entries can stringify to the same key (operation #1 and its name "1")
        return ("chi", tuple((str(k), complex(v)) for k, v in
                             sorted(spec.items(), key=lambda kv: str(kv[0]))))
    return ("id", id(spec))


# ---- the grammar of a declaration ---------------------------------------------
# `characters=` and `multiplets=` are read as grammar once, where a call starts
# and before anything is measured, and a malformed one raises `ValueError`
# stating the argument's grammar.  Which words name something at a site is
# read later, in the site group the measurement reaches.

#: ``characters=`` in one sentence, which every error about its grammar opens with.
_CHARACTERS_GRAMMAR = (
    "`characters=` is a list with one entry per orbital, each a word or None, or a "
    "mapping from an orbital, or from several orbitals of one site written together, "
    "to a word or to a mapping from operations to characters")

#: ``multiplets=`` in one sentence, which every error about its grammar opens with.
_MULTIPLETS_GRAMMAR = (
    "`multiplets=` is one list per site of the multiplets it holds, each written as "
    "its orbitals, such as [[(0, 1)], [(2,)]], or the SiteMultiplets a representation "
    "carries")


def _shown(value, width=60) -> str:
    """``value`` as a grammar error quotes it: its type, and its repr cut to ``width``."""
    if value is None:
        return "None"
    text = repr(value)
    if len(text) > width:
        text = text[:width - 3] + "..."
    return f"the {type(value).__name__} {text}"


def _orbital_index(value) -> bool:
    """Whether ``value`` can name an orbital: an integer, and not a truth value."""
    return (isinstance(value, (int, np.integer))
            and not isinstance(value, (bool, np.bool_)))


def _character_value(value) -> bool:
    """Whether ``value`` can be a character: a finite number, and not a truth value."""
    return (isinstance(value, numbers.Number) and not isinstance(value, (bool, np.bool_))
            and bool(np.isfinite(complex(value))))


def _declaration_grammar(norb, *, characters=None, multiplets=None, spin=None,
                         site_of=None):
    r"""Raise ``ValueError`` where ``characters=`` or ``multiplets=`` is malformed.

    The grammar of the two declarations, read where :func:`_build_rep`,
    :func:`..groups.realized_group` and :func:`..symmetrization.symmetrize`
    start, before anything is measured.  ``characters=`` is a list with one
    entry per orbital, each a word, a mapping from operations to characters or
    ``None``, or a mapping from an orbital, or from several orbitals of one
    site together, to a word or to such a mapping: each orbital is named once,
    a mapping of characters names at least one operation, by its index or its
    name, and gives each a finite number, and the two orbitals of a declared
    spin pair, the up and down states of one Wannier function, are given one
    word.  ``multiplets=`` is one list per site of the multiplets it holds,
    each named by its orbitals, or a :class:`SiteMultiplets`: the multiplets
    hold every orbital once, each the orbitals of one site, and each holds
    both states of a declared spin pair.  Whether a word names something at
    its site depends on the site group the measurement reaches, and is read
    there.

    ``norb`` is the model's orbital count, ``spin`` the declared
    :class:`..spin.SpinBasis` or ``None``, and ``site_of`` each orbital's site,
    or ``None`` where the sites are not read yet.  An entry whose value is a
    :class:`.._sitechars.DeclaredContent` is a content already read, carried
    from one group to the next, and passes.

    Returns the multiplets ``multiplets=`` states, every site's in one
    sequence, or ``None`` where it states none.
    """
    pairs = () if spin is None else tuple((int(u), int(d)) for u, d in spin.pairs)
    if characters is not None:
        _characters_grammar(characters, int(norb), pairs, site_of)
    if multiplets is None:
        return None
    return _multiplets_grammar(multiplets, int(norb), pairs, site_of)


def _names_joint_keys(characters) -> bool:
    """Whether ``characters=`` has a key of several orbitals, whose site the grammar reads."""
    return isinstance(characters, dict) and any(
        isinstance(key, (tuple, list)) and len(key) > 1 for key in characters)


def _characters_grammar(characters, norb, pairs, site_of=None) -> None:
    """The grammar of ``characters=``, the first half of :func:`_declaration_grammar`.

    ``site_of`` is each orbital's site, where the sites are read: a key of
    several orbitals names what one site holds together.
    """

    def malformed(what):
        raise ValueError(f"{_CHARACTERS_GRAMMAR}; {what}.")

    named = []                         # (orbitals, value, where it was written)
    if isinstance(characters, dict):
        for key, value in characters.items():
            if isinstance(value, _sitechars.DeclaredContent):
                continue
            if _orbital_index(key):
                orbs = (int(key),)
            elif isinstance(key, (tuple, list)) and not key:
                malformed("the key () names no orbital")
            elif isinstance(key, (tuple, list)) and all(_orbital_index(o) for o in key):
                orbs = tuple(int(o) for o in key)
            else:
                malformed(f"it is given the key {key!r}, a {type(key).__name__}")
            named.append((orbs, value, f"the key {key!r}"))
    elif isinstance(characters, (list, tuple)):
        if len(characters) != norb:
            malformed(f"this model has {norb} orbitals, and the list has "
                      f"{len(characters)} entr{'y' if len(characters) == 1 else 'ies'}")
        named = [((i,), value, f"entry {i}") for i, value in enumerate(characters)
                 if value is not None
                 and not isinstance(value, _sitechars.DeclaredContent)]
    else:
        malformed(f"it is given {_shown(characters)}")
    holder = {}
    for orbs, value, where in named:
        for o in orbs:
            if not 0 <= o < norb:
                malformed(f"{where} names orbital {o}, and this model's orbitals are 0 "
                          f"to {norb - 1}")
            if o in holder:
                malformed(f"it names each orbital once, and orbital {o} is named twice, in "
                          + (where if holder[o][1] == where
                             else f"{holder[o][1]} and in {where}"))
            holder[o] = (value, where)
        if site_of is not None and len(orbs) > 1:
            sites = sorted({int(site_of[o]) for o in orbs})
            if len(sites) > 1:
                malformed(f"{where} names orbitals of sites {sites}")
        if isinstance(value, str):
            continue
        if isinstance(value, dict):
            if not value:
                malformed(f"{where} is given an empty mapping, and a mapping from "
                          "operations to characters names at least one operation")
            for op, chi in value.items():
                if not (_orbital_index(op) or isinstance(op, str)):
                    malformed(f"{where} names an operation by {_shown(op)}, and an "
                              "operation is named by its index or its name, such as 3 "
                              "or 'C4(z)'")
                if not _character_value(chi):
                    malformed(f"{where} gives the operation {op!r} {_shown(chi)}, and a "
                              "character is a finite number")
            continue
        if value is None:
            malformed(f"{where} is given None, and a mapping leaves out an orbital it "
                      "declares nothing about")
        malformed(f"{where} is given {_shown(value)}")
    # the two states of one Wannier function hold one content
    for c, (up, dn) in enumerate(pairs):
        if (up in holder and dn in holder
                and _spec_key(holder[up][0]) != _spec_key(holder[dn][0])):
            raise ValueError(
                f"{_CHARACTERS_GRAMMAR}, and gives the two orbitals of a declared spin pair "
                f"one word: they are the up and down states of one Wannier function, "
                f"which has one site character. Orbitals {up} and {dn}, spin pair {c}, "
                f"are given {holder[up][0]!r} and {holder[dn][0]!r}; give both the same "
                "word, or declare only one of them.")


def _multiplets_grammar(multiplets, norb, pairs, site_of) -> tuple:
    """The grammar of ``multiplets=``, the second half of :func:`_declaration_grammar`.

    Returns the multiplets it states, every site's in one sequence.
    """

    def malformed(what):
        raise ValueError(f"{_MULTIPLETS_GRAMMAR}; {what}.")

    if isinstance(multiplets, SiteMultiplets):
        stated = multiplets.by_site
    elif isinstance(multiplets, (list, tuple)):
        stated = multiplets
    else:
        malformed(f"it is given {_shown(multiplets)}")
    flat = []
    for i, site in enumerate(stated):
        if not isinstance(site, (list, tuple)):
            malformed(f"entry {i} is {_shown(site)}, where the list of one site's "
                      "multiplets goes")
        for orbs in site:
            if not isinstance(orbs, (list, tuple)):
                malformed(f"entry {i}, {site!r}, holds {_shown(orbs)} where a multiplet "
                          "goes" + _per_site_spelling(stated, norb, site_of))
            if not orbs:
                malformed(f"it holds no empty multiplet, and entry {i} holds one")
            for o in orbs:
                if not _orbital_index(o):
                    malformed(f"the multiplet {tuple(orbs)!r} names an orbital by "
                              f"{_shown(o)}, and an orbital is named by its index")
                if not 0 <= int(o) < norb:
                    malformed(f"the multiplet {tuple(orbs)!r} names orbital {int(o)}, out "
                              f"of range for a model with {norb} orbitals")
            flat.append(tuple(int(o) for o in orbs))
    holder = {}
    for j, orbs in enumerate(flat):
        for o in orbs:
            if o in holder:
                malformed(f"it partitions every orbital of the model into multiplets, and "
                          f"orbital {o} is named twice, in {orbs!r}"
                          + ("" if holder[o] == j else f" and {flat[holder[o]]!r}"))
            holder[o] = j
    missing = tuple(o for o in range(norb) if o not in holder)
    if missing:
        malformed("it partitions every orbital of the model into multiplets, and no "
                  f"multiplet holds orbital{'s' if len(missing) > 1 else ''} "
                  + (str(missing) if len(missing) > 1 else str(missing[0])))
    for c, (up, dn) in enumerate(pairs):
        if holder[up] != holder[dn]:
            malformed(f"each multiplet holds both states of a declared spin pair, the up "
                      f"and down states of one Wannier function, and spin pair {c}, "
                      f"orbitals ({up}, {dn}), is split between {flat[holder[up]]!r} and "
                      f"{flat[holder[dn]]!r}")
    if site_of is not None:
        for orbs in flat:
            sites = sorted({int(site_of[o]) for o in orbs})
            if len(sites) > 1:
                malformed(f"each multiplet holds the orbitals of one site, and the "
                          f"multiplet {orbs!r} spreads over sites {sites}")
    return tuple(flat)


def _per_site_spelling(stated, norb, site_of) -> str:
    """The one-list-per-site spelling of multiplets written as one flat list, or ``""``.

    ``[(0,), (1,)]`` lists multiplets where ``multiplets=`` lists sites; where
    every entry is a multiplet of orbitals of this model, the same multiplets
    are offered one list per site, by ``site_of`` where the sites are read.
    """
    if not stated or not all(
            isinstance(m, (list, tuple)) and m
            and all(_orbital_index(o) and 0 <= int(o) < norb for o in m)
            for m in stated):
        return ""
    by_site = {}
    for m in stated:
        site = 0 if site_of is None else int(site_of[int(m[0])])
        by_site.setdefault(site, []).append(tuple(int(o) for o in m))
    spelled = [by_site[s] for s in sorted(by_site)]
    return f"; one list per site, {spelled!r} states these multiplets"


def _declaration_entries(characters, model, assignment, *, spin=None):
    r"""``characters=`` parsed into ``(entries, atom_of, atom_orbs)`` — grammar only.

    The first half of :func:`_normalize_declaration`, kept on its own because
    **step 1** reads the same declaration before any site group has been
    reduced and before any multiplet has been measured
    (:func:`_declared_site_contents`): one parse, so the two can never
    disagree about which orbitals were named.

    ``entries`` is one ``(orbitals, spec, joint)`` per declaration in the order
    it was written; ``atom_of`` and ``atom_orbs`` are the **atoms** — one
    orbital, or the two states of one declared spinor pair, which is one
    Wannier function and never split.

    Raises
    ------
    ValueError
        Where the declaration is malformed (:func:`_declaration_grammar`), which
        every call that takes one has asked before anything was measured.
    """
    _declaration_grammar(int(model.norb), characters=characters, spin=spin)
    site_of = np.asarray(assignment.site_of_orbital, dtype=int)
    site_orbs = tuple(tuple(int(o) for o in orbs) for orbs in assignment.site_orbitals)
    atom_of, atom_orbs = {}, {}
    for s, site_atoms in enumerate(_atoms_of_site(site_orbs, spin)):
        for a in site_atoms:
            key = (int(s), int(min(a)))
            atom_orbs[key] = tuple(int(o) for o in a)
            for o in a:
                atom_of[int(o)] = key
    raw = []                    # (the orbitals of one declaration, its spec, joint?)
    if isinstance(characters, dict):
        for key, spec in characters.items():
            joint = isinstance(key, (tuple, list)) and len(key) > 1
            orbs = tuple(int(k) for k in (key if isinstance(key, (tuple, list)) else (key,)))
            raw.append((orbs, spec, bool(joint)))
    else:
        seq = list(characters)
        by_site = {}
        for a, spec in enumerate(seq):
            if spec is None:
                continue
            key = (int(site_of[a]), spec.lower() if isinstance(spec, str) else id(spec))
            by_site.setdefault(key, []).append((a, spec))
        for (site, _), entries in by_site.items():
            orbs = tuple(a for a, _ in entries)
            spec = entries[0][1]
            if not isinstance(spec, str):
                for a in orbs:                   # a per-operation declaration never auto-groups
                    raw.append(((a,), spec, False))
                continue
            raw.append((orbs, spec, False))
    return tuple(raw), atom_of, atom_orbs


def _declared_site_contents(model, group, assignment, characters, *, spin=None):
    r"""What each site was declared to carry — read in the **cell group**'s site tables.

    Step 1's reading of ``characters=``, made before any Hamiltonian is: per
    site, the Cartesian polynomials the declared shorthands are
    (:data:`.._sitechars.SHORTHANDS`) and the characters the declared labels
    are, each on the site group the cell allows.  A shorthand is a *function*
    and a label an abstract *content*, and the difference is the whole of
    step 1: a function's span need not be invariant under the site group, a
    label's is invariant by construction.

    Returns ``(contents, read_in)`` — ``contents`` keyed by site with
    ``"polys"``, ``"chars"`` (one ``{ind: chi}`` per declaration, the span's
    own character standing in for a function), ``"entries"``, and ``"whole"``
    and ``"corep"``: the whole content the site was declared to hold, and
    whether any of it was written in the site coreps; ``read_in``
    keyed by the declaration's orbitals with the site group and group its word
    was read in, which is what :attr:`.._sitechars.DeclaredContent.declared_in`
    then says however far the cell is afterwards reduced.

    **The whole content** is the sum of what the site's declarations state,
    however they partition its orbitals: ``"p"`` on a ``p`` shell and ``pz``
    beside ``E`` on its orbitals are one content, ``A1 + E`` at a ``3m`` site.
    A word naming a content wider than the orbitals it is written on names part
    of it, and the parts written with one content at one site add up to whole
    copies of it -- ``px`` and ``py`` are one ``E`` -- counted as one copy where
    they make no whole number of them, since a word says which content its
    orbital belongs to.  A declared spinor pair counts once.

    Raises
    ------
    _sitechars.DeclarationError
        ``gate="unknown_label"`` where a declared word names nothing in this
        cell's site groups.  On a free-standing layer it says that the group to
        read the word in is two operations away, and the declaration is carried
        (``groups.py``).
    ValueError
        Where the declaration is malformed (:func:`_declaration_grammar`).
    """
    entries, atom_of, _atom_orbs = _declaration_entries(
        characters, model, assignment, spin=spin)
    site_of = np.asarray(assignment.site_of_orbital, dtype=int)
    kept_u = {int(s.ind) for s in group.symmetries if not s.time_reversal}
    kept_au = {int(s.ind) for s in group.symmetries if s.time_reversal}
    tables = _site_tables(group, assignment, kept_u, kept_au,
                          sorted({int(site_of[a]) for orbs, _s, _j in entries
                                  for a in orbs}))
    contents: dict = {}
    read_in: dict = {}
    for orbs, spec, joint in entries:
        site = int(site_of[orbs[0]])
        table = tables[site]
        where = f"orbital{'s' if len(orbs) > 1 else ''} {orbs if len(orbs) > 1 else orbs[0]}"
        rec = contents.setdefault(int(site), {"polys": [], "chars": [], "entries": [],
                                              "atoms": set(), "pool": {}, "corep": False})
        read_in[tuple(sorted(int(a) for a in orbs))] = (
            f"{_sitechars.point_group_symbol(group, table.indices)} in "
            f"{group.name} #{group.number_str}")
        named = {atom_of[int(a)] for a in orbs}
        if isinstance(spec, _sitechars.DeclaredContent):
            # already read, on this same cell: carried, never read twice
            chi = {int(k): complex(v) for k, v in spec.constraints.items()
                   if int(k) in set(table.indices)}
            dim, corep = int(spec.dim), bool(spec.corep)
            if spec.functions:
                rec["polys"].extend(spec.functions)
                rec["atoms"].update(named)
        elif _sitechars.shorthand_key(spec) is not None:
            poly = _sitechars.SHORTHANDS[_sitechars.shorthand_key(spec)]
            span, dim = _sitechars.span_character(group, table, [poly])
            rec["polys"].append(poly)
            rec["atoms"].update(named)
            chi = {int(ind): complex(span[table.class_of[int(ind)]]) for ind in table.indices}
            corep = False
        else:
            cons, _label, dim, _note = _sitechars.resolve_spec(group, table, spec, where)
            chi = {int(k): complex(v) for k, v in cons.items()}
            corep = _sitechars.named_in_coreps(table, spec)
        rec["chars"].append(chi)
        rec["entries"].append((orbs, spec, joint))
        rec["corep"] = rec["corep"] or bool(corep)
        # the orbitals written with one content at this site, pooled: parts of
        # a wider content add up to copies of it, and a declared spinor pair
        # written twice, once per member, is one atom
        part = rec["pool"].setdefault(_declared_character_key(chi),
                                      {"chi": chi, "dim": max(1, int(dim)), "atoms": set()})
        part["atoms"].update(named)
    for rec in contents.values():
        pieces = list(rec.pop("pool").values())
        asked = set.intersection(*(set(p["chi"]) for p in pieces)) if pieces else set()
        whole = {}
        for p in pieces:
            copies = len(p["atoms"]) if p["dim"] == 1 else max(
                1, -(-len(p["atoms"]) // p["dim"]))
            for ind in asked:
                whole[int(ind)] = whole.get(int(ind), 0.0) + copies * complex(p["chi"][ind])
        rec["whole"] = whole
    # WHICH OF THE TWO the declared functions name, read exactly as
    # :func:`_normalize_declaration` reads them.  The functions declared at a
    # site generate an orbit under its site group; where that orbit is as wide
    # as the orbitals they were declared for, the content declared IS the
    # orbit -- the doublet `px` and `py` span at a `3m` site, or the `1E2E`
    # pair one `dz2` names at a `23` site -- and an orbit is invariant, so it
    # reduces nothing.  Where the orbit is wider than the orbitals named, what
    # was declared is the functions themselves, and a span of those is exactly
    # what a site group need not preserve.
    for site, rec in contents.items():
        if not rec["polys"]:
            continue
        _chi, rank = _sitechars.span_character(group, tables[site], rec["polys"])
        if int(rank) <= len(rec["atoms"]):
            rec["polys"] = []
    return contents, read_in


def _content_compatible(group, assignment, contents):
    r"""Step 1: which operations of the cell the **declared content** leaves.

    A declaration is a statement about the crystal, made before any
    Hamiltonian is read, so an operation the declared content is not carried
    by is not an operation of the cell at all.  Two halves, measured over the
    unitary operations and inherited by their primed partners (a site
    character is a character of the *unitary* site group, so ``m(x)`` and
    ``m(x)'`` stand or fall together):

    **The functions.**  The declared shorthands at site ``a`` span ``V_a``
    (:func:`.._sitechars.span_basis`).  An operation carrying ``a`` onto ``b``
    is compatible when it carries ``V_a`` **onto** ``V_b``
    (:func:`.._sitechars.carries_span`) — with ``b = a`` that is the
    stabilizer of a declared function, which is why a single ``px`` at a
    ``3m`` site is a crystal with one mirror and nothing more.

    **The characters.**  Sites ``a`` and ``b = g·a`` carry the same declared
    content when ``chi_b(g h g⁻¹) = chi_a(h)`` for every ``h`` fixing ``a``,
    the conjugation taken through the group's own product table, modulo
    lattice translations, which is the sense in which a site group is
    measured.  ``chi`` is each site's **whole** declared content, whatever
    partition of its orbitals the words were written on (``"whole"`` of
    :func:`_declared_site_contents`): a ``p`` shell declared ``"p"`` at one
    site and ``p_z`` beside its doublet at the next is one content.  What is
    compared is characters, never words: an irrep label, the shorthand whose
    orbit it is and the shell or sum that holds the same content are one
    declaration wherever their characters are one.  A function's own span
    character stands in where one end names a function and the other a label.
    Where either site's content was written in the site coreps it states the
    corep content alone, and the two are compared as corep content: on the
    operations whose spin-1/2 lift has a nonzero trace, where every orbital
    content lifting to one corep content has its characters, and not on the
    twofolds and mirrors, where every one of them lifts to zero.

    A site nothing was declared about imposes nothing: its content is the
    transported one, which is the inference a partial declaration rests on.

    **The largest group.**  Where the declaration states every site of each
    orbit it touches, and in the same terms at each -- a function at all of
    them or at none -- compatibility is a relation between the sites' declared
    contents and the compatible operations are a group.  Otherwise they need
    not be one: an operation carrying a declared site onto one with nothing to
    compare there -- left undeclared, or declared a content where the first
    holds a function -- is compatible whatever it carries the declared content
    onto, and two such can compose into one the declared content is not
    carried by: ``p_y`` declared at one ``3c`` site of ``Pm-3m`` and nothing at
    the other two, where a fourfold axis and a threefold one together carry
    ``p_y`` onto ``p_z`` at its own site.  The declared crystal's operations
    form a group, so the operations kept are the largest subgroup of the
    cell's group inside the compatible ones, the first by the tie rule every
    such choice is made by where several of that order exist
    (:func:`_largest_compatible_group`), and an undeclared site carries what it
    transports onto it: there ``m-3``, the twelve rotations of the
    tetrahedron and their products with inversion, which carries ``p_y`` onto
    ``p_z`` and ``p_x`` at the other two sites.  The compatible operations it
    leaves out are excluded beside the others, each with its reason.

    Returns ``(kept, excluded)`` — the ``.ind`` kept as a frozenset over
    **all** operations, and ``{ind: {"reason", "note", "orbitals", "sites"}}``
    for the rest.
    """
    syms = group.symmetries
    at = {int(s.ind): p for p, s in enumerate(syms)}
    prod = group.get_product_table(get_diff=False)
    perm = np.asarray(assignment.permutations, dtype=int)
    unitary = [int(s.ind) for s in syms if not s.time_reversal]
    spans = {s: _sitechars.span_basis(rec["polys"]) if rec["polys"] else None
             for s, rec in contents.items()}
    fixed = {s: tuple(int(i) for i in assignment.site_group[s] if int(i) in at
                      and not syms[at[int(i)]].time_reversal)
             for s in contents}
    # the inverse of each UNITARY operation, read off the group's own product
    # table modulo lattice translations -- never `get_inverse_table`, which is
    # asked about the antiunitary coset too and warns that it has no inverse
    ident = _identity_position(group)
    inverse = {a: b for a in unitary for b in unitary
               if int(prod[at[a], at[b]]) == ident}

    def _conjugate(g, h):
        """``g h g⁻¹`` as an ``.ind``, modulo lattice translations."""
        gi = inverse[int(g)]
        return int(syms[int(prod[at[int(g)], int(prod[at[int(h)], at[gi]])])].ind)

    # the operations a content written in the site coreps states nothing under:
    # those whose spin-1/2 lift has zero trace
    silent = ({ind for ind, v in _sitechars.lift_squares(
                  {ind: syms[at[ind]].rotation_cart for ind in unitary}).items() if v <= 1e-9}
              if any(rec.get("corep") for rec in contents.values()) else set())

    def _matched(g, a, b):
        """Does ``a``'s whole declared content transport onto ``b``'s?

        ``chi_b(g h g⁻¹) = chi_a(h)`` on every ``h`` fixing ``a`` both state --
        a relation between the two sites' whole contents, so it is symmetric,
        and among declared sites what it keeps is closed under products.  Two
        sites declared contents of different dimensions differ on the identity.
        """
        chi_a, chi_b = contents[a]["whole"], contents[b]["whole"]
        coreps = bool(contents[a].get("corep") or contents[b].get("corep"))
        for h in fixed[a]:
            if int(h) not in chi_a or (coreps and int(h) in silent):
                continue
            c = _conjugate(g, h)
            if c not in chi_b:
                # the smaller group no longer asks this question, and that
                # dropping IS the restriction of the declared content
                continue
            if abs(complex(chi_b[c]) - complex(chi_a[int(h)])) > 1e-6:
                return False
        return True

    def _orbitals(*sites):
        return tuple(sorted({int(o) for s in sites
                             for orbs, _spec, _joint in contents[int(s)]["entries"]
                             for o in orbs}))

    excluded: dict = {}
    for g in unitary:
        for a in sorted(contents):
            b = int(perm[at[g], int(a)])
            if b not in contents:
                continue                      # an undeclared site imposes nothing
            if (spans[a] is not None and spans[b] is not None
                    and not _sitechars.carries_span(group, g, spans[a], spans[b])):
                excluded[g] = {
                    "reason": ("the declared function at this site is not carried to "
                               "itself" if b == a else
                               "the declared function is not carried onto the image "
                               "site's own declared content"),
                    "sites": (int(a), int(b)), "orbitals": _orbitals(a, b)}
                break
            if not _matched(g, a, b):
                excluded[g] = {
                    "reason": ("the declared content is not carried to itself" if b == a
                               else "declared content differs across the orbit"),
                    "sites": (int(a), int(b)), "orbitals": _orbitals(a, b)}
                break
    partners = _primed_partners(group, [int(s.ind) for s in syms])
    for g, rec in list(excluded.items()):
        mate = partners.get(int(g))
        if mate is not None:
            excluded[int(mate)] = dict(rec)
    for g, rec in excluded.items():
        _excluded_named(group, int(g), rec)
    return _largest_compatible_group(group, assignment, _orbitals(*contents), excluded)


#: Why step 1 leaves out an operation the declared content is carried by on its
#: own: the operations kept are the largest group inside the compatible ones
#: (:func:`_largest_compatible_group`).  ``_OUTSIDE_THE_GROUP`` is said where
#: no subgroup of that order holds the operation, ``_OUTSIDE_THE_TIE`` where
#: one does and the tie rule keeps another.
_OUTSIDE_THE_GROUP = ("the declared content is carried by this operation on its own, but "
                      "not by every product of it with the operations kept")
_OUTSIDE_THE_TIE = ("the declared content is carried by a group holding this operation, as "
                    "large as the one kept, and the tie rule keeps the other")


def _outside_the_group(rec) -> bool:
    """Whether a step 1 exclusion is the largest group's, rather than the declared content's own."""
    return rec.get("reason") in (_OUTSIDE_THE_GROUP, _OUTSIDE_THE_TIE)


def _excluded_named(group, ind, rec, *, said=""):
    """One step 1 exclusion given its operation's ``ind``, ``name`` and the ``note`` saying it in words."""
    from .groups import _record_name

    sym = next(s for s in group.symmetries if int(s.ind) == int(ind))
    rec["ind"] = int(ind)
    rec["name"] = _record_name(sym, group.lattice)
    rec["note"] = (f"#{int(ind)} {rec['name']}: {rec['reason']} (site "
                   + (f"{rec['sites'][0]}" if rec["sites"][0] == rec["sites"][1]
                      else f"{rec['sites'][0]} -> site {rec['sites'][1]}")
                   + f", orbitals {rec['orbitals']})" + said)
    return rec


def _largest_compatible_group(group, assignment, declared, excluded):
    r"""Step 1's kept operations: the largest group inside the ones the declared content is carried by.

    ``excluded`` holds the operations the declared content is not carried by,
    unitary and primed alike (:func:`_content_compatible`), and ``declared``
    the orbitals the declaration names.  The compatible operations are
    whatever is left, and they need not compose: a declared site carried onto
    an undeclared one constrains nothing there, nor does a declared function
    carried onto a site declared a content alone, so two compatible operations
    can make one the declared content is not carried by.  The crystal declared
    has a group of operations, and it is the largest subgroup of ``group``
    whose unitary operations all lie among the compatible ones -- the question
    the descent asks of operations measured to hold that do not compose, asked
    here of the operations the declared content is carried by, and answered by
    the same enumeration and the same tie rule
    (:func:`..groups._resolve_closure`).  A primed operation stands or falls
    with its unitary partner, as a site character is a character of the unitary
    site group.

    Where the compatible operations already compose, which they do wherever
    the declaration states every site of an orbit, as a function at all of
    them or at none, nothing is added.  Otherwise each operation left out is
    excluded with ``_OUTSIDE_THE_GROUP`` as its reason, or
    ``_OUTSIDE_THE_TIE`` where a subgroup of the largest order holds it and
    the tie rule keeps another, whose note then names the subgroups that tie
    and why the first is kept.  Its ``sites`` are a declared site and its
    image, the first declared site carried onto an undeclared one where there
    is one, and its ``orbitals`` the declared ones there.

    Returns ``(kept, excluded)`` -- the ``.ind`` kept as a frozenset over all
    operations, and ``excluded`` with the operations left out added.
    """
    from .groups import (_descent_closure, _record_name, _resolve_closure, _tie_reason,
                         _tie_rule_words)

    syms = group.symmetries
    compatible = [s for s in syms if not s.time_reversal and int(s.ind) not in excluded]
    if _descent_closure(group, compatible):
        _group, dropped, tied = _resolve_closure(group, [int(s.ind) for s in compatible])
        at = {int(s.ind): p for p, s in enumerate(syms)}
        perm = np.asarray(assignment.permutations, dtype=int)
        site_of = np.asarray(assignment.site_of_orbital, dtype=int)
        named = sorted({int(o) for o in declared})
        sites = sorted({int(site_of[o]) for o in named})
        lattice = np.asarray(group.lattice, dtype=float)

        def _set(inds):
            return "{" + ", ".join(_record_name(syms[at[int(i)]], lattice) for i in inds) + "}"

        said = ""
        if tied:
            said = ("; the largest groups the declared content is carried by tie, "
                    + " and ".join(_set(c) for c in tied)
                    + f", and the tie rule -- {_tie_rule_words(group)} -- keeps the first "
                    f"since {_tie_reason(group, tied)}")
        partners = _primed_partners(group, [int(s.ind) for s in syms])
        for g, why in dropped:
            images = [(a, int(perm[at[int(g)], a])) for a in sites]
            a, b = next(((a, b) for a, b in images if b not in sites), images[0])
            rec = {"reason": _OUTSIDE_THE_TIE if why == "tie" else _OUTSIDE_THE_GROUP,
                   "sites": (int(a), int(b)),
                   "orbitals": tuple(o for o in named if int(site_of[o]) in (a, b))}
            for ind in (int(g), partners.get(int(g))):
                if ind is not None and ind not in excluded:
                    excluded[ind] = _excluded_named(group, ind, dict(rec),
                                                    said=said if why == "tie" else "")
    kept = frozenset(int(s.ind) for s in syms if int(s.ind) not in excluded)
    return kept, excluded


def _normalize_declaration(characters, model, assignment, group, tables, layout,
                           *, spin=None, read_in=None):
    """Parse ``characters=`` into the declared contents, each with the orbitals it is declared on (grammar only).

    A declaration names **what a set of orbitals holds**.  The piece the
    grammar counts is the **atom** — one orbital, or the two states of one
    declared spinor pair, which is one Wannier function and never split — and
    a content's dimension is a number of atoms: ``{(0, 2): "E"}`` on an
    interleaved spinor basis with pairs ``(0, 1)`` and ``(2, 3)`` declares the
    two *pairs* jointly as a doublet, while ``{(0, 1): "s"}`` is one atom.
    The joint form names the content its orbitals hold together -- an irrep,
    a sum of irreps or a shell -- and that content's dimension has to be the
    number of atoms it names.

    A word whose content is wider than the one orbital it is written on --
    ``px`` at a ``3m`` site, whose orbit is the doublet ``E`` -- names that
    orbital's part of the content.  Where the multiplet measured there (in
    ``layout``) has the content's own dimension, the part is that whole
    multiplet (``{0: "E'"}`` on a ``p_x, p_y`` pair).  Elsewhere the parts
    written with one content at one site add up, and have to make whole
    copies of it: ``["pz", "px", "py"]`` on a ``p`` shell whose orbitals are
    turned inside the site is ``A1`` on ``p_z`` and one ``E`` on the other
    two, however the Hamiltonian binds them.  How the declared contents are
    then read against the measured multiplets is :func:`_declared_blocks`'.

    Each declaration comes back with the :class:`.._sitechars.DeclaredContent`
    it resolved to — the constraint ``{ind: chi(g)}``, the label, the dimension,
    the Cartesian polynomials a shorthand names and the group it was read in —
    which is what every later re-identification re-keys rather than resolving
    the label again.

    ``read_in`` is step 1's own reading, keyed by the declaration's orbitals:
    where the word was read before the declared content reduced the cell, and
    what site group it resolves in now (:func:`_declared_site_contents`).
    Without it the two are one place, written once.

    Raises
    ------
    _sitechars.DeclarationError
        ``gate="ambiguous_multiplet"`` for a joint declaration whose content
        is not the dimension of the orbitals it names, or parts of a content
        that make no whole number of copies of it at their site;
        ``gate="unknown_label"`` from the label resolution itself.
    ValueError
        Where the declaration is malformed (:func:`_declaration_grammar`).
    """
    site_of = np.asarray(assignment.site_of_orbital, dtype=int)
    paired = spin is not None              # each atom a declared (up, dn) pair
    chan_of, chan_orbs = layout._chan_of, layout._chan_orbs
    # the indivisible pieces, which is what the dimension check counts
    raw, atom_of, atom_orbs = _declaration_entries(characters, model, assignment,
                                                   spin=spin)

    def _atoms_named(orbs):
        """The distinct indivisible pieces a declaration names, order-preserving."""
        out = []
        for a in orbs:
            k = atom_of[int(a)]
            if k not in out:
                out.append(k)
        return tuple(out)

    def _members(keys):
        """Every orbital of those pieces, each piece in its own order."""
        return tuple(int(o) for k in keys for o in atom_orbs[k])

    # THE WORD WRITTEN FOR EACH ORBITAL, whichever declaration it came in: both
    # states of a declared spinor pair, one Wannier function, and every orbital
    # of a content already resolved.  A multiplet the Hamiltonian glued and
    # declared orbital by orbital (``{0: "px", 1: "py"}`` on a ``p_x, p_y``
    # pair) resolves once per orbital to the whole multiplet, and each of those
    # contents carries every word its orbitals were written with, so that the
    # one a later round keeps still says all of them.  A shell is a function
    # word of every orbital it was written on (``"p"``); an irrep or a sum of
    # irreps names a content and no function.
    said = {}
    for orbs, spec, _joint in raw:
        if isinstance(spec, _sitechars.DeclaredContent):
            for a, w in spec.words:
                said.setdefault(int(a), str(w))
            continue
        w = _sitechars.canonical_shorthand(spec) or _sitechars.shell_key(spec) or ""
        if w:
            for a in orbs:
                for o in atom_orbs[atom_of[int(a)]]:
                    said.setdefault(int(o), w)
    # ... and WHAT WAS WRITTEN for each orbital, word or not: one declaration can
    # cover orbitals written as different words, `px` on one and `py` on its
    # partner, and each row says its own
    written = {}
    for orbs, spec, _joint in raw:
        if isinstance(spec, _sitechars.DeclaredContent):
            for a, w in (spec.written or tuple((int(o), spec.spec) for o in spec.orbitals)):
                written.setdefault(int(a), w)
            continue
        for a in orbs:
            for o in atom_orbs[atom_of[int(a)]]:
                written.setdefault(int(o), spec)

    decls, placed, parts = [], [], {}

    def _one(rec, members, ks):
        base = rec["base"]
        content = _sitechars.DeclaredContent(
            orbitals=tuple(int(o) for o in members),
            spec=(base.spec if base is not None else rec["spec"]),
            label=str(rec["label"]), dim=int(rec["dim"]), declared_in=rec["read_where"],
            constraints={int(k): complex(v) for k, v in rec["cons"].items()},
            note=str(rec["note"]), broke=(base.broke if base is not None else ()),
            functions=tuple(rec["functions"]), reduced_in=str(rec["reduced_where"]),
            words=tuple((int(o), said[int(o)]) for o in members if int(o) in said),
            written=tuple((int(o), written[int(o)]) for o in members if int(o) in written),
            corep=bool(rec["corep"]), mulliken=tuple(rec["mulliken"]))
        decls.append({"orbitals": content.orbitals,
                      "channels": tuple(sorted({int(chan_of[int(o)])
                                                for o in content.orbitals})),
                      "site": rec["site"], "spec": content.spec, "label": content.label,
                      "dim": content.dim, "note": content.note,
                      "constraints": content.constraints, "content": content,
                      "joint": bool(rec["joint"]), "corep": content.corep})
        placed.append(tuple(ks))

    for orbs, spec, joint in raw:
        site = int(site_of[orbs[0]])
        keys = _atoms_named(orbs)
        # the grammar gives both members of a pair one word, so a repeat of the
        # same pair is the same declaration written twice (once per member)
        # rather than a second, independent one
        if paired and keys in placed:
            continue
        table = tables[site]
        where = f"orbital{'s' if len(orbs) > 1 else ''} {orbs if len(orbs) > 1 else orbs[0]}"
        cons, label, dim, note = _sitechars.resolve_spec(group, table, spec, where)
        base = spec if isinstance(spec, _sitechars.DeclaredContent) else None
        # a name the tables give a row Mulliken's convention calls by another:
        # the declaration's note names that row, once, where the name is read
        mulliken = (base.mulliken if base is not None
                    else _sitechars.mulliken_names(group, table, spec))
        if mulliken and base is None:
            note = (note + "; " if note else "") + _sitechars.mulliken_clause(mulliken)
        site_symbol = _sitechars.point_group_symbol(group, table.indices)
        here = f"{site_symbol} in {group.name} #{group.number_str}"
        # WHERE THE DECLARED WORD WAS READ, which is not always where it
        # resolves: step 1 reads it in the cell's own site groups and may lower
        # the cell before anything is resolved, and the record has to say both.
        first = (read_in or {}).get(tuple(sorted(int(a) for a in orbs)))
        read_where = (base.declared_in if base is not None else
                      (first if first is not None else here))
        reduced_where = (base.reduced_in if base is not None else
                         ("" if first is None or first == here else here))
        key = _sitechars.shorthand_key(spec)
        functions = ((base.functions if base is not None else ())
                     if key is None else (_sitechars.SHORTHANDS[key],))
        # a sum written in the site coreps states the corep content and nothing
        # finer, which every comparison with the Hamiltonian's blocks reads
        corep = (base.corep if base is not None
                 else _sitechars.named_in_coreps(table, spec))
        rec = {"site": site, "spec": spec, "label": label, "dim": dim, "cons": cons,
               "note": note, "base": base, "read_where": read_where,
               "reduced_where": reduced_where, "functions": functions, "joint": joint,
               "corep": corep, "mulliken": mulliken}

        if joint and dim != len(keys):
            n = len(keys)
            noun = "Wannier function" if paired else "orbital"
            # a content in the site coreps is counted in what holds it, each
            # orbital with both its spin states
            wide = (f"holds {dim} {noun}{'' if dim == 1 else 's'}" if paired or corep
                    else f"is {dim}-dimensional")
            # the spelling to offer: separate words for a content of one, and
            # otherwise what the site holds says what fits it
            fix = (("; {" + ", ".join(f"{int(min(atom_orbs[k]))}: {label!r}" for k in keys)
                    + f"}} is the spelling for {n} separate {label} {noun}s.")
                   if dim == 1 else
                   ". " + _multiplet_spelling(site, _site_atoms(atom_orbs, site), dim, spec,
                                              table, paired=paired))
            raise _sitechars.DeclarationError(
                "ambiguous_multiplet",
                f"{where}: the joint form names the content its {noun}s hold together, and "
                f"{label!r} {wide} against the {n} {noun}{'' if n == 1 else 's'} named, "
                "so it is not what they hold jointly" + fix,
                {"orbitals": orbs, "site": site, "declared_irrep": label, "dim": dim,
                 "multiplet": tuple(atom_orbs[k] for k in keys)})
        if dim == 1 and len(keys) > 1:
            for k in keys:                       # k copies of a 1D label = k declarations
                _one(rec, atom_orbs[k], (k,))
            continue
        if dim == len(keys):
            _one(rec, _members(keys), keys)
            continue
        if len(keys) == 1 and dim > 1:
            # a content wider than the one orbital it is written on, where the
            # HAMILTONIAN glued a multiplet of its dimension: the declaration is
            # about the whole multiplet measured there
            c = int(chan_of[int(orbs[0])])
            if int(layout._dim(c)) == int(dim):
                _one(rec, tuple(int(o) for o in chan_orbs[c]), keys)
                continue
        # A PART OF A CONTENT: the orbitals written with it at one site add up,
        # below, to the copies of it they hold together
        pool = parts.setdefault((site, _declared_character_key(cons)),
                                {"rec": rec, "keys": [], "orbs": []})
        for k in keys:
            if k not in pool["keys"]:
                pool["keys"].append(k)
        pool["orbs"].extend(int(a) for a in orbs)

    for (site, _chi), pool in parts.items():
        rec, keys = pool["rec"], tuple(pool["keys"])
        dim, table = int(rec["dim"]), tables[site]
        if len(keys) % dim:
            orbs = tuple(pool["orbs"])
            noun = "Wannier function" if paired else "orbital"
            wide = (f"holds {dim} {noun}{'' if dim == 1 else 's'}" if paired or rec["corep"]
                    else f"is {dim}-dimensional")
            raise _sitechars.DeclarationError(
                "ambiguous_multiplet",
                f"orbital{'s' if len(orbs) > 1 else ''} "
                f"{orbs if len(orbs) > 1 else orbs[0]}: {rec['label']!r} {wide}, "
                f"and the {len(keys)} {noun}{'s' if len(keys) > 1 else ''} of site {site} "
                f"written with it make{'s' if len(keys) == 1 else ''} no whole number of "
                f"copies of it, so which {noun}s hold it is not said. "
                + _multiplet_spelling(site, _site_atoms(atom_orbs, site), dim, rec["spec"],
                                      table, paired=paired),
                {"orbitals": orbs, "site": site, "declared_irrep": rec["label"],
                 "dim": dim,
                 "multiplet": tuple(atom_orbs[k] for k in keys)})
        copies = len(keys) // dim
        cons = {int(k): copies * complex(v) for k, v in rec["cons"].items()}
        label = rec["label"]
        if copies > 1:
            chi = [cons.get(int(cls[0]), 0.0) for cls in table.classes]
            named = _sitechars.corep_label if rec["corep"] else _sitechars.content_label
            label = named(table, chi) or f"{copies}x{label}"
        _one(dict(rec, cons=cons, label=label, dim=len(keys)), _members(keys), keys)
    return decls


def _declared_character_key(cons):
    """A comparable identity for one declared character, ``{ind: chi(g)}``."""
    return tuple(sorted((int(k), round(complex(v).real, 9), round(complex(v).imag, 9))
                        for k, v in cons.items()))


def _site_atoms(atom_orbs, site) -> tuple:
    """The atoms of one site, each its orbitals: its orbitals, or its declared spin pairs."""
    return tuple(atom_orbs[k] for k in sorted(k for k in atom_orbs if int(k[0]) == int(site)))


def _multiplet_spelling(site, atoms, dim, spec, table, *, paired) -> str:
    """The spelling to suggest for a ``dim``-dimensional content at one site.

    ``atoms`` are the site's own: its orbitals, or, where ``paired``, its
    declared spin pairs, each one Wannier function, and ``dim`` counts them.  A
    suggestion has to be a declaration this package accepts, so the joint form
    is offered only where the site holds enough of them to make one, each named
    by one of its orbitals; where it does not, the message says what the site
    holds and lists the contents that fit it (:func:`.._sitechars.fitting_words`).
    """
    n, dim = len(atoms), int(dim)
    if n >= dim > 1:
        joint = tuple(int(min(a)) for a in atoms[:dim])
        return ("Use the mapping form and name the partners jointly, e.g. "
                f"{{{joint}: {spec!r}}}.")
    if paired:
        held = (f"{n} Wannier function{'' if n == 1 else 's'}, the spin "
                f"pair{'' if n == 1 else 's'} " + " and ".join(str(tuple(a)) for a in atoms))
        wide = f"no multiplet of {dim} Wannier functions"
        room = "one Wannier function" if n == 1 else f"at most {n} Wannier functions"
    else:
        held = f"{n} orbital{'' if n == 1 else 's'} {tuple(int(a[0]) for a in atoms)}"
        wide = f"no {dim}-dimensional multiplet"
        room = "one orbital" if n == 1 else f"at most {n} orbitals"
    words = _sitechars.fitting_words(table, n)
    return (f"Site {site} carries {held}, so {wide} sits on it at all: name a content of "
            f"{room} instead" + (f" ({', '.join(words)})" if words else "") + ".")


def _declared_blocks(decls, layout, assignment, tables, *, spin=None):
    r"""``decls`` — a declaration read on the multiplets the Hamiltonian measures.

    A declaration states what each site holds; how the model's orbitals are
    partitioned inside a site is its basis, which a Wannier basis turned inside
    its site need not honour.  So the multiplets are always the measured ones,
    and each site's declarations are read on them one of two ways:

    * where the declared contents *are* the measured multiplets they touch, each
      is read on its own multiplet: those multiplets are the Hamiltonian's own
      subspaces, not a choice of basis;
    * otherwise the site's declarations are one content over all of its
      measured multiplets, the sum of what was declared there
      (:func:`_site_reading`) -- a content declared orbital by orbital on a
      multiplet the Hamiltonian binds whole, one declared whole over several
      multiplets, a word on one orbital of a multiplet.  Where some of the
      site's orbitals were declared nothing, what was declared is part of what
      the site holds.

    **A content written in the site coreps** states the corep content alone,
    and which of a site's multiplets holds which of its coreps is the basis's,
    so it is read over the site's multiplets whole wherever the site holds
    several: ``2x-E1`` on a ``(p_x, p_y)`` pair measured apart from its
    ``p_z`` is part of the site's ``2x-E1 + -1E2E``.

    Whether the content is what the site's multiplets carry is then the
    verdict's comparison of characters (:func:`_verdicts_and_choices`): one
    content spread over several coupled parts of the crystal is one
    requirement on their characters together, and a part of what a site holds
    has to be contained in it.  ``layout`` holds the measured multiplets,
    ``tables`` the site tables the declarations were read in, and ``spin`` --
    the declared :class:`..spin.SpinBasis`, whose ``(up, dn)`` pairs are each
    one atom of two orbitals, or ``None`` -- is what the contents were read
    with.

    Returns the declarations to read, in the order written, a site's one
    content in the place of its first.
    """
    site_orbs = tuple(tuple(int(o) for o in orbs) for orbs in assignment.site_orbitals)
    pair = 1 if spin is None else 2
    measured = tuple(tuple(tuple(int(o) for o in ch) for ch in site) for site in layout.by_site)
    # each declared content once, and the SAME orbitals declared twice (a
    # check on itself, which is how a carried record spells a measured pair)
    # are one content, not two
    groups_of = {}
    for d in decls:
        groups_of.setdefault(int(d["site"]), {})[tuple(int(o) for o in d["orbitals"])] = d
    merged = {}
    for s in range(len(site_orbs)):
        merged.update(_site_reading(site_orbs[s], measured[s], groups_of.get(s, {}),
                                    tables.get(s), pair))
    out, placed = [], set()
    for d in decls:
        one = merged.get((int(d["site"]), tuple(int(o) for o in d["orbitals"])))
        if one is None:
            out.append(d)
        elif id(one) not in placed:
            placed.add(id(one))
            out.append(one)
    return out


def _site_reading(orbitals, measured, groups, table, pair):
    """``{(site, orbitals): content}`` -- one site's declarations, read on its measured multiplets.

    ``orbitals`` are the site's, ``measured`` its multiplets and ``groups`` its
    declared contents keyed by their orbitals (:func:`_declared_blocks`), and
    ``pair`` how many orbitals make one atom.  Where the declared contents are
    exactly the measured multiplets they touch, each is read on its own
    multiplet and nothing comes back -- unless one was written in the site
    coreps over a site of several multiplets, which states the site's corep
    content.  Otherwise every declared content of the site comes back as the
    one content they add up to over all of its multiplets
    (:func:`_block_content`).
    """
    if not groups:
        return {}
    named = {int(o) for g in groups for o in g}
    touched = [m for m in measured if set(m) & named]
    corep = any(groups[g].get("corep") for g in groups)
    if ({frozenset(g) for g in groups} == {frozenset(m) for m in touched}
            and not (corep and len(measured) > 1)):
        return {}
    one = _block_content([groups[g] for g in groups], list(measured), table, pair)
    if one is None:                       # pragma: no cover - a guard, not a path
        raise InternalError(
            f"the declarations at the site of orbitals {tuple(sorted(orbitals))} were each "
            "resolved in its site group, and their sum is no content of that group as wide "
            "as the orbitals they name. A content resolved in a site table is a sum of its "
            "irreps by construction, so the two disagreeing is a defect in this package, "
            "not a tolerance to loosen.")
    site = int(next(iter(groups.values()))["site"])
    return {(site, tuple(int(o) for o in g)): one for g in groups}


def _block_content(groups, multiplets, table, pair):
    r"""One declared content over a site's measured multiplets, or ``None`` where it is none.

    ``groups`` are the site's declarations, ``multiplets`` its measured
    multiplets and ``pair`` how many orbitals make one atom.  What was declared
    is the sum of the groups' contents, each a sum of the site's irreps as wide
    as the orbitals it names; narrower than the multiplets, it is part of what
    they hold.  Whether they carry it is the verdict's comparison of
    characters (:func:`_verdicts_and_choices`).  ``None`` where a declaration is
    no such sum, or the content is wider than the multiplets.
    """
    if table is None:
        return None
    counts = {}
    for d in groups:
        chi = np.asarray([complex(d["constraints"].get(int(cls[0]), np.nan))
                          for cls in table.classes])
        if not np.all(np.isfinite(chi)):
            return None
        width = 0
        for p, n in enumerate(_sitechars.decompose(table, chi)):
            n = complex(n)
            if abs(n.imag) > 1e-6 or abs(n.real - round(n.real)) > 1e-6 or n.real < -0.5:
                return None
            if round(n.real):
                counts[p] = counts.get(p, 0) + int(round(n.real))
                width += int(round(n.real)) * int(round(float(np.real(table.dims[p]))))
        if width != len(d["orbitals"]) // pair:
            return None
    declared = sum(int(round(float(np.real(table.dims[p])))) * n for p, n in counts.items())
    total = sum(len(m) // pair for m in multiplets)
    if declared > total:
        return None
    partial = declared < total
    corep = any(bool(d.get("corep")) for d in groups)

    orbitals = tuple(o for m in multiplets for o in m)
    named = tuple(int(o) for d in groups for o in d["orbitals"])
    chi = sum(n * np.asarray(table.characters[p], dtype=complex) for p, n in counts.items())
    label = _sitechars.content_label(table, chi)
    specs = tuple(dict.fromkeys(d["spec"] for d in groups
                                if not isinstance(d["spec"], dict)))
    cons = {int(ind): complex(sum(complex(d["constraints"][int(ind)]) for d in groups))
            for ind in table.indices}
    first = groups[0]["content"]
    content = _sitechars.DeclaredContent(
        orbitals=(named if partial else orbitals),
        spec=(specs[0] if len(specs) == 1 else (specs or groups[0]["spec"])),
        label=label, dim=int(declared), declared_in=first.declared_in, constraints=cons,
        note="; ".join(dict.fromkeys(str(d["note"]) for d in groups if d["note"])),
        broke=tuple(dict.fromkeys(b for d in groups for b in d["content"].broke)),
        functions=tuple(f for d in groups for f in d["content"].functions),
        reduced_in=first.reduced_in,
        words=tuple(w for d in groups for w in d["content"].words),
        written=tuple(w for d in groups for w in d["content"].written),
        corep=corep,
        mulliken=tuple(dict.fromkeys(t for d in groups for t in d["content"].mulliken)))
    return {"orbitals": content.orbitals, "block": orbitals, "channels": (),
            "site": int(groups[0]["site"]), "spec": content.spec, "label": label,
            "dim": int(declared), "note": content.note, "constraints": cons,
            "content": content, "joint": True, "partial": bool(partial),
            "groups": tuple(groups), "corep": content.corep}


def _mismatch_named(decls, rows, admissible, site_tables, *, ref_orb=0):
    """The ``verdict=`` and ``named=`` halves of a ``declaration_breaks_group`` message.

    Which declaration is the one a reader has to act on: reading the
    declarations in the order the lowest-angular-momentum rule reads the
    crystal — the reference orbital's own first where it has one, then the
    others as written — the first that no one-dimensional character meets
    together with those before it in its coupled part of the crystal.  So
    which declaration is named does not depend on where the reference's own
    was written.  Nothing is measured here and nothing is decided;
    :func:`_name_the_mismatch` turns those numbers into irrep names.
    """
    def _needs(d):
        """What the rows measured for one declaration ask of its part's character."""
        out = {}
        for r in rows:
            if tuple(r["orbitals"]) != tuple(d["orbitals"]):
                continue
            if r["required"] is not None:
                out[int(r["ind"])] = complex(r["required"])
            elif abs(complex(r["declared"])) > 1e-6:
                out[int(r["ind"])] = None
        return out

    part_of = {tuple(r["orbitals"]): int(r.get("part", 0)) for r in rows}
    first = _reference_declaration(decls, ref_orb)
    order = ([first] if first is not None else []) + [d for d in decls if d is not first]
    failed, still_free, left = None, None, {}
    for d in order:
        if tuple(d["orbitals"]) not in part_of:
            continue                      # nothing was measured about it
        part = part_of[tuple(d["orbitals"])]
        survivors = left.get(part, list(admissible))
        needs = _needs(d)
        keep = [chi for chi in survivors if _meets(chi, needs)]
        if not keep:
            failed, still_free = d, survivors
            break
        left[part] = keep
    if failed is None:
        return {}
    measured = {}
    for d in decls:
        measured[tuple(d["orbitals"])] = {
            int(r["ind"]): complex(r["measured"]) for r in rows
            if tuple(r["orbitals"]) == tuple(d["orbitals"])}
    verdict, named = _name_the_mismatch(failed, measured, still_free, decls, site_tables)
    return {"verdict": verdict,
            "named": dict(named, orbitals=tuple(failed["orbitals"]),
                          declared_as=failed["spec"], declared_irrep=failed["label"])}


def _name_the_mismatch(failed, measured, still_free, decls, site_tables, tol=1e-6):
    r"""What a ``declaration_breaks_group`` measured, said *in irrep names*.

    The numeric measured-vs-declared table below the message is complete but
    unreadable; what a reader needs is "the declaration said A1, the hoppings
    say B1".  Both halves of that are computable here:

    * the **measured label** of the failing multiplet — its character times
      each one-dimensional character still admissible when the declaration was
      tested,
      named against its own site table and so in the vocabulary the record
      **returns** its labels in: the double-valued site corep where the states
      carry spin, the single-valued site irrep where they do not.  When the
      earlier declarations already pinned the one-dimensional character there is
      exactly one such name; otherwise the alternatives are all listed, because the
      ambiguity is real.
    * the **relative character** ``θ_measured / θ_declared``, which is a
      character of the site group in its own right — it is exactly the
      one-dimensional character the declaration would need and no labelling
      can supply.  It is a one-dimensional character and not a
      site representation, so it is named in the orbital-factor vocabulary
      rather than the returned one.  For the standing example (lin3g's two
      ``1a`` orbitals, both declared ``s``) it is ``B1``, and it stays ``B1``
      when the two declarations are written the other way round, because both
      are read in the gauge the *earlier* declarations already pinned rather
      than in the reference orbital's.

    Nothing about the spin frame is said here.  A declaration is what the
    declaration states the orbitals are, and a Hamiltonian that does not respect it
    is a crystal with less symmetry rather than a transcription to check: the
    verdict is the smaller group, and the message is about which operations it
    keeps.

    This names; it never decides.  Every number it reads was measured once, by
    :func:`_verdicts_and_choices`, and the verdict it is spelling out was made
    by :func:`_covariant_subgroup`.

    Returns ``(text, context)``: the block to splice into the message, and the
    machine-readable half (``measured_irrep``, ``relative_character``,
    ``declared_irreps``) for the ``Refusal``'s context.
    """
    tab = site_tables.get(int(failed["site"]))
    theta = measured[tuple(failed["orbitals"])]

    meas, rels = [], []
    for chi in (still_free or [None]):
        gauged = (dict(theta) if chi is None
                  else {ind: complex(chi.get(int(ind), 1.0)) * v for ind, v in theta.items()})
        lab = _label_character(tab, gauged)
        if lab and lab not in meas:
            meas.append(lab)
        # a content written in the coreps states nothing where the spin-1/2
        # lift has zero trace, and nothing was measured there
        rel = {ind: complex(gauged.get(int(ind), 0.0)) / complex(want)
               for ind, want in failed["constraints"].items() if abs(complex(want)) > tol
               and (not failed.get("corep") or int(ind) in gauged)}
        lab = _label_character(tab, rel, orbital=True)
        if lab and lab not in rels:
            rels.append(lab)
    measured_irrep = " or ".join(meas)
    rel_label = " or ".join(rels)

    def _declared(d):
        """A declaration named as the record returns its irrep, against its own site."""
        return (_label_character(site_tables.get(int(d["site"])), d["constraints"],
                                 sums=True) or d["label"] or d["spec"])

    at = next((i for i, d in enumerate(decls) if d is failed), len(decls))
    said = ", ".join(f"{tuple(d['orbitals'])}={_declared(d)}" for d in decls[:at])
    lines = []
    if said:
        lines.append(f"  declared {said}, {tuple(failed['orbitals'])}="
                     f"{_declared(failed)}")
    wanted = _declared(failed)
    lines.append(
        f"  measured: orbital(s) {tuple(failed['orbitals'])} carry "
        f"{measured_irrep or 'a character matching no single irrep of this site group'}"
        f", not the declared {wanted or 'character'}"
        + (f"; relative character {rel_label}." if rel_label else
           "; the ratio measured/declared is not a character of this site group either, so "
           "no other labelling accommodates it."))
    return "\n".join(lines), {"measured_irrep": measured_irrep,
                              "relative_character": rel_label,
                              "declared_irreps": tuple(
                                  (tuple(d["orbitals"]), d["label"]) for d in decls)}


def _label_character(table, chi, tol=1e-6, *, sums=False, orbital=False):
    """Name a character against a table's rows — ``""`` when no row matches it.

    The label :func:`_named_content` gives, without where it comes from.
    ``sums`` names a **reducible** character as the sum of the irreps it
    decomposes into; off by default, because a relative character is
    one-dimensional by construction and a sum there would name an object that
    cannot exist.
    """
    return _named_content(table, chi, tol, sums=sums, orbital=orbital)[0]


#: How directly a row's name comes off the tables, most direct first.  A sum of
#: rows is named as directly as its least direct part.
_NAMED_BY_RANK = {"tables": 0, "point group": 1, "Mulliken": 2}


def _label_terms(label):
    """The irreps one label spells, each with its multiplicity, in the order written.

    ``"2x-E1 + -1E2E"`` is ``[("-E1", 2), ("-1E2E", 1)]``, and a single irrep is
    one term of multiplicity one.
    """
    out = []
    for part in str(label).split(" + "):
        part = part.strip()
        if not part:
            continue
        count, mark, name = part.partition("x")
        if mark and count.isdigit() and name:
            out.append((name, int(count)))
        else:
            out.append((part, 1))
    return out


def _named_content(table, chi, tol=1e-6, *, sums=True, orbital=False):
    """``(label, named_by)`` — the site irreps a character holds, and where their name comes from.

    A character one row of the table matches is that row, with the row's own
    name and provenance.  With ``sums``, a **reducible** character is named by
    the irreps it decomposes into, which are fixed by the character alone
    whatever basis the orbitals are written in: a ``p`` shell whose orbitals
    are turned inside a ``3m`` site so that no subset of them carries the site
    group is ``A1 + E`` all the same, one row of the table per term.  Where the
    states carry spin a row's own name can already be a sum -- ``E ⊗ ½`` at a
    ``3m`` site is ``-1E2E + -E1`` -- and the terms are counted across the
    rows, so the same shell with its spin is ``2x-E1 + -1E2E``: the site
    content an aligned basis spreads over two multiplets, ``-E1`` and
    ``-1E2E + -E1``, spelled once.  An irrep subduced to a smaller site group
    is named the same way (``E`` of ``4mm`` restricted to ``mm2`` is
    ``B1 + B2``).

    A sum is named from the tables only as far as its least direct part is --
    ``tables``, then ``point group`` -- and a sum one of whose rows has no name
    is not named at all, ``("", "unnamed")``, since naming it would claim a
    name for that row too.  The generator characters stay the whole statement
    there.

    ``orbital`` asks for the row's **orbital-factor** spelling where it has one
    (:attr:`.._sitechars.SiteGroupTable.declarable`).  That is what a character
    which is not a site representation at all wants — the ratio of a measured
    character to a declared one is a one-dimensional character of the site
    group, the character the declaration would need, and naming it after the
    double-valued corep of the row with those values would name the wrong kind
    of object.
    """
    if table is None:
        return "", "unnamed"

    def name(p):
        if orbital and p < len(table.declarable) and table.declarable[p]:
            return table.declarable[p]
        return table.labels[p]

    def where(p):
        return str(table.named_by[p]) if p < len(table.named_by) else "unnamed"

    for p in range(len(table.labels)):
        if all(abs(table.character_on(p, ind) - complex(chi.get(int(ind), 0.0))) <= tol
               for ind in table.indices if int(ind) in chi):
            return (name(p), where(p)) if name(p) else ("", "unnamed")
    if not sums or any(int(ind) not in chi for ind in table.indices):
        return "", "unnamed"
    mult = _sitechars.decompose(
        table, np.asarray([complex(chi[int(cls[0])]) for cls in table.classes]))
    counts, how = {}, set()
    for p in range(len(mult)):
        n = complex(mult[p])
        if abs(n.imag) > 1e-6 or abs(n.real - round(n.real)) > 1e-6 or round(n.real) < 0:
            return "", "unnamed"
        if round(n.real) > 0:
            if not name(p):
                # one of the parts has no name; naming the sum would claim one
                return "", "unnamed"
            for term, k in _label_terms(name(p)):
                counts[term] = counts.get(term, 0) + int(round(n.real)) * k
            how.add(where(p))
    if not counts:
        return "", "unnamed"
    label = " + ".join(term if k == 1 else f"{k}x{term}" for term, k in counts.items())
    return label, max(how, key=lambda w: _NAMED_BY_RANK.get(w, len(_NAMED_BY_RANK)))


def _copies_of_one_irrep(table, chi, tol=1e-6):
    """``(chi_one, m)`` — a character that is ``m`` copies of one irrep, split.

    A multiplet the Hamiltonian binds can hold several copies of one site
    irrep, written in a basis no orbital subset splits (``A'`` and ``A'`` at a
    mirror site, rotated into one another), and every vector of such an
    isotypic component carries that irrep.  So its rows are the irrep's:
    ``chi_one`` is the character of one copy and ``m`` how many the multiplet
    holds.  Anything else — an irrep, or a sum of different ones that the
    site's own operations glue — comes back unchanged with ``m = 1``.
    """
    if table is None or any(int(cls[0]) not in chi for cls in table.classes):
        return chi, 1
    mult = _sitechars.decompose(
        table, np.asarray([complex(chi[int(cls[0])]) for cls in table.classes]))
    held = [complex(n) for n in mult if abs(complex(n)) > tol]
    if len(held) != 1:
        return chi, 1
    n = held[0]
    if abs(n.imag) > tol or abs(n.real - round(n.real)) > tol or round(n.real) < 2:
        return chi, 1
    m = int(round(n.real))
    return {int(ind): complex(v) / m for ind, v in chi.items()}, m


def _one_copy(table, chi, k, tol=CHARACTER):
    """``chi / k`` where the content ``chi`` is ``k`` copies of one content, and ``None`` otherwise.

    Read on the site table: every irrep's multiplicity a multiple of ``k``.  A
    content the table cannot read is taken as it stands, divided.
    """
    if table is None or any(int(cls[0]) not in chi for cls in table.classes):
        return {int(ind): complex(v) / k for ind, v in chi.items()}
    mult = _sitechars.decompose(
        table, np.asarray([complex(chi[int(cls[0])]) for cls in table.classes]))
    for n in mult:
        n = complex(n) / k
        if abs(n.imag) > tol or abs(n.real - round(n.real)) > tol:
            return None
    return {int(ind): complex(v) / k for ind, v in chi.items()}


def _distance_from_trivial(chi):
    """Sort key ordering one-dimensional characters by their distance from the trivial one.

    First how many operations the character is not ``+1`` on; then, operation by
    operation in ``.ind`` order, the largest real part and — where a group with
    no antiunitary operation admits complex characters and two of them are
    conjugate — the largest imaginary part.  That last clause puts
    ``λ(4⁺) = +i`` before ``−i`` on a unitary-only fourfold axis, the two being
    equally far from trivial and equally real: the one turning with the axis
    first.  As the labelling's last key it decides only where the content,
    the letter and the band labels all tie; on the spin-orbit ``I4_1`` model
    the band labels decide, for ``−i``.
    """
    vals = [complex(v) for _, v in sorted(chi.items())]
    return (sum(1 for v in vals if abs(v - 1.0) > 1e-9),
            tuple((-round(float(v.real), 9), -round(float(v.imag), 9)) for v in vals))


def _meets(chi, needs, *, over=None, tol=CHARACTER):
    r"""Whether a one-dimensional character meets what declarations ask of it.

    Every reading of a declaration comes to this one test.  ``needs`` maps an
    operation's ``.ind`` to the value the character of the declared
    multiplets' coupled part has to take there for the declared content to be
    what the Hamiltonian carries, ``None`` where no value does; an operation it
    does not name asks nothing.  ``over`` asks the question on those operations
    alone -- a subgroup, or the operations a declaration was measured at.
    """
    for ind, want in needs.items():
        if over is not None and int(ind) not in over:
            continue
        if want is None or abs(complex(chi.get(int(ind), 1.0)) - complex(want)) > tol:
            return False
    return True


def _declared_needs(d, raw, lift_trace, *, tol=1e-6):
    r"""``{ind: value}`` -- what one declaration asks of its part's character, on the matrices.

    ``raw`` holds each multiplet's orbital character on the representation
    built before any labelling is chosen (:func:`_multiplet_characters`); the
    declared multiplets carry the declared content under a character ``chi``
    exactly where ``chi(g)`` times the sum of their characters is the declared
    one, at every operation the declaration speaks about.  A content written in
    the site coreps states nothing at an operation whose spin-1/2 lift has zero
    trace, and an operation that moves one of the multiplets asks nothing.
    """
    out = {}
    for ind, want in d["constraints"].items():
        ind = int(ind)
        if d.get("corep") and abs(complex(lift_trace.get(ind, 1.0))) <= 1e-9:
            continue
        got = [raw[int(c)].get(ind) for c in d["channels"]]
        if any(v is None for v in got):
            continue
        theta = complex(sum(got))
        if abs(theta) > tol:
            out[ind] = complex(want) / theta
        elif abs(complex(want)) > tol:
            out[ind] = None
    return out


def _holds_part(d, assigned, raw, site_tables, lift_trace, *, unread=False):
    """Does a block, its multiplets under ``assigned(c)``, hold what was declared on part of it?

    What is left of the block's content once the declared content is taken
    away has to be a content of the site group, the part nothing was declared
    about.  A content written in the site coreps states the corep content
    alone, so what the block holds has to contain it as coreps
    (:func:`.._sitechars.corep_counts`): ``2x-E1`` is part of the spin-orbit
    ``p`` shell's ``2x-E1 + -1E2E`` at a ``3m`` site although no orbital
    content lifting to it is part of the shell's ``A1 + E``.  Where a row's
    lift has no name, what the block holds has to contain an orbital content
    lifting to it.  ``assigned(c)`` is the character multiplet ``c``'s coupled
    part carries, so the blocks of several parts are read together; ``unread``
    is what comes back where a class of the site group was not measured on
    every multiplet of the block, and so says nothing either way.
    """
    tab = site_tables.get(int(d["site"]))
    if tab is None:
        return False
    held, want, asked = [], [], []
    for cls in tab.classes:
        ind = next((int(g) for g in cls if int(g) in d["constraints"]
                    and all(int(g) in raw[int(c)] for c in d["channels"])), None)
        if ind is None:
            return unread
        held.append(sum(complex(assigned(int(c)).get(ind, 1.0)) * complex(raw[int(c)][ind])
                        for c in d["channels"]))
        want.append(complex(d["constraints"][ind]))
        asked.append(abs(complex(lift_trace.get(ind, 1.0))) > ZERO)
    return _contained(tab, held, want, corep=bool(d.get("corep")), asked=asked)


def _contained(tab, held, want, *, corep=False, asked=None):
    """Whether the content ``held`` contains the content ``want``, both given class by class of ``tab``.

    What is left of ``held`` once ``want`` is taken away has to be a content of
    the site group.  A content written in the site coreps (``corep``) is
    compared as the coreps it lifts to, and where a row's lift has no name,
    ``held`` has to contain an orbital content whose characters are
    ``want``'s on the classes ``asked``, those whose spin-1/2 lift has a
    nonzero trace.
    """
    if not corep:
        return _rows_of(tab, np.asarray(held, dtype=complex)
                        - np.asarray(want, dtype=complex)) is not None
    holds, stated = _rows_of(tab, held), _rows_of(tab, want)
    if holds is None or stated is None:
        return False
    have = _sitechars.corep_counts(tab, holds)
    need = _sitechars.corep_counts(tab, stated)
    if have is not None and need is not None:
        return all(have.get(c, 0) >= n for c, n in need.items())
    counts = [holds.get(p, 0) for p in range(len(tab.characters))]
    asked = [True] * len(want) if asked is None else list(asked)
    for part in itertools.product(*(range(k + 1) for k in counts)):
        chi = sum((m * np.asarray(tab.characters[p], dtype=complex)
                   for p, m in enumerate(part) if m), np.zeros(len(want), dtype=complex))
        if all(abs(complex(chi[i]) - want[i]) <= CHARACTER
               for i in range(len(want)) if asked[i]):
            return True
    return False


def _rows_of(tab, chi):
    """``{row: copies}`` of a content of ``tab``, given class by class, or ``None`` where it is none."""
    out = {}
    for p, n in enumerate(_sitechars.decompose(tab, np.asarray(chi, dtype=complex))):
        n = complex(n)
        if (abs(n.imag) > CHARACTER or abs(n.real - round(n.real)) > CHARACTER
                or round(n.real) < 0):
            return None
        if round(n.real):
            out[p] = int(round(n.real))
    return out


def _joint_holds(j, pick, inside, *, tol=CHARACTER):
    """Whether one content read together is carried under one labelling of the parts it spans.

    ``j`` is one of the contents the verdict stage reads together
    (:func:`_verdicts_and_choices`): what each of its multiplets carries, per
    operation, against its own coupled part's scale.  ``pick`` gives each part
    it spans a character, and ``inside`` is the subgroup asked about.  A
    content declared on the whole of its site is carried where its
    multiplets together carry exactly it; one declared on part of the site
    where what they carry contains it, on the site group the subgroup leaves
    (:func:`_holds_part`), an operation of that site group nothing measured
    leaving the question open.
    """
    d = j["decl"]
    raw = {int(c): {int(g): v for g, v in row.items() if int(g) in inside}
           for c, row in j["theta"].items()}

    def assigned(c):
        return pick[j["part"][int(c)]]

    if not d.get("partial"):
        carried = {c: {g: complex(assigned(c).get(g, 1.0)) * complex(v) for g, v in row.items()}
                   for c, row in raw.items()}
        return _meets({}, _declared_needs(d, carried, j["lift_trace"]), tol=tol)
    table = j["table_of"](tuple(int(g) for g in j["site_group"] if int(g) in inside))
    if table is None:                     # pragma: no cover - a site group closes
        return True
    return _holds_part(d, assigned, raw, {int(d["site"]): table}, j["lift_trace"],
                       unread=True)


def _multiplet_characters(D, syms, u_pos, cperm, layout, prod):
    r"""``(raw, residual)`` -- every multiplet's orbital character on one representation.

    Read once, on the representation built before any labelling is chosen:
    ``raw[c]`` is multiplet ``c``'s character over the operations that fix it
    (:func:`_measure_channel_characters`), and ``residual`` the worst defect of
    that reading -- on a measured multiplet also how far its blocks are from
    closing as a single-valued representation of the site group, which a
    spinful model's are not (the block carries the spinor too).
    """
    raw, worst = {}, 0.0
    chan_states = layout._chan_states
    for c in range(len(chan_states)):
        chi, res = _measure_channel_characters(D, syms, u_pos, cperm, layout, c)
        raw[c] = chi
        worst = max(worst, res)
    if layout._measured:
        pos_of_ind = {int(s.ind): p for p, s in enumerate(syms)}
        for c in range(len(chan_states)):
            if layout._chan_kind[c] != "measured":
                continue
            st = np.asarray(chan_states[c], dtype=int)
            block = {ind: D[pos_of_ind[int(ind)]][np.ix_(st, st)] for ind in raw[c]}
            for a_ind, A in block.items():
                for b_ind, B in block.items():
                    ab = int(syms[int(prod[pos_of_ind[a_ind], pos_of_ind[b_ind]])].ind)
                    if ab in block:
                        worst = max(worst, float(np.abs(A @ B - block[ab]).max()))
    return raw, worst


#: What each key of the labelling reads, in the order :func:`_labelling` asks
#: them: the site content at the lowest angular momentum (orbits read together
#: read one by one, and identical copies copy by copy, first; then at the lead's
#: site, over the crystal, on the lead's row), the letter the orbitals read as,
#: the band labels, and last the group's order of operations, on which no label
#: rests.
_KEY_NAMES = ("content", "content", "content", "content", "content", "orbitals", "bands",
              "order")


class _LazyW:
    """A candidate's matrices on its states, built the first time one is read."""

    def __init__(self, cand):
        self._cand = cand

    def _all(self):
        return self._cand["W"]

    def __getitem__(self, p):
        return self._all()[p]

    def __contains__(self, p):
        return p in self._all()

    def __iter__(self):
        return iter(self._all())

    def __len__(self):
        return len(self._all())

    def items(self):
        return self._all().items()

    def keys(self):
        return self._all().keys()


class _LazyOption(dict):
    """A labelling of an orbit built by induction, its matrices built only where a key asks for them.

    ``"W"`` is the candidate's own matrices on the orbit's states, built the
    first time one is read, and ``"D"`` the representation with them put in --
    built each time it is asked for, so that what a key reads off it is kept
    and the matrices are not.
    """

    def __init__(self, base, cand, states, **fields):
        super().__init__(fields, W=_LazyW(cand))
        self._base, self._states = base, np.asarray(states, dtype=int)

    def __missing__(self, key):
        if key == "D":
            W, st = self["W"], self._states
            M = {}
            for p, Dp in self._base.items():
                M[p] = np.array(Dp, dtype=complex, copy=True)
                if p in W:
                    M[p][np.ix_(st, st)] = W[p]
            return M
        raise KeyError(key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


def _choose(options, keys):
    r"""``(kept, fate)`` -- the option the keys keep, first key first.

    Each key scores the options still tied, and only those with its lowest
    score go on to the next; the first of the last ones left is kept.  A key
    with a ``batch`` attribute scores the tied options together, one score
    each.  ``fate[i]`` is the position in ``keys`` of the key that set option
    ``i`` aside, ``None`` for the one kept, and ``len(keys)`` for one no key set
    apart from it.
    """
    left = list(range(len(options)))
    fate = [None] * len(options)
    for k, key in enumerate(keys):
        if len(left) <= 1:
            break
        scores = (list(key.batch([options[i] for i in left])) if hasattr(key, "batch")
                  else [key(options[i]) for i in left])
        best = min(scores)
        keep = [i for i, s in zip(left, scores) if s == best]
        for i in left:
            if i not in keep:
                fate[i] = k
        left = keep
    for i in left[1:]:
        fate[i] = len(keys)
    return left[0], fate


def _band_fibers(model, group, D, syms, *, params=None):
    r"""The energy multiplets at the high-symmetry points, solved once.

    ``H`` is solved once at every high-symmetry point the tables list, in their
    order, and each point's eigenvectors are grouped into energy multiplets,
    lowest first.  A point whose little group is the identity alone, or holds
    an operation ``D`` does not carry, is left out.

    Returns one ``(point, k, little group, multiplets, V)`` per point; empty
    where the tables do not reach the cell.
    """
    from . import groups as _groups
    from .sewing import DEGENERACY_TOL, _as_columns, multiplets

    if _groups._lattice_not_descended(group) is not None or not group._tables_ready:
        return ()
    points = _groups.high_symmetry_points(group)
    if isinstance(points, Refusal) or not points:
        return ()
    pos_of = {int(s.ind): p for p, s in enumerate(syms)}
    wanted = []
    for name, k in points.items():
        lg = _groups.little_group(group, k)
        if isinstance(lg, Refusal) or len(lg.unitary) <= 1:
            continue
        if any(int(i) not in pos_of or pos_of[int(i)] not in D for i in lg.indices):
            continue
        wanted.append((str(name), np.asarray(k, dtype=float).reshape(-1), lg))
    if not wanted:
        return ()
    evals, evecs = _bloch.solve(model, np.array([k for _n, k, _lg in wanted]), params)
    evals, evecs = np.asarray(evals), np.asarray(evecs)
    if evals.ndim == 1:                     # solve_ham drops a length-1 k axis
        evals, evecs = evals[None], evecs[None]
    return tuple((name, k, lg, multiplets(np.asarray(evals[i], dtype=float).reshape(-1),
                                          DEGENERACY_TOL), _as_columns(evecs[i]))
                 for i, (name, k, lg) in enumerate(wanted))


def _band_share(model, group, assignment, D, syms, fibers, states):
    r"""One coupled part's share of every energy multiplet's character.

    Per point of ``fibers`` (:func:`_band_fibers`), ``chars[row, column]``: the
    trace, over the multiplet's eigenvectors restricted to the part's
    ``states``, of the Bloch operator of ``D`` there (:func:`bloch_operator`),
    one column per unitary operation of the little group.  ``D`` carries each
    part onto itself, so a multiplet's character is the sum of its parts'
    shares, and a character ``chi`` on the part multiplies its share by
    ``chi(g)``.  ``None`` at a point where the operator's own checks fail.
    """
    pos_of = {int(s.ind): p for p, s in enumerate(syms)}
    st = np.asarray(states, dtype=int)
    out = []
    for _name, k, lg, rows, V in fibers:
        chars = np.zeros((len(rows), len(lg.indices)), dtype=complex)
        try:
            for col, ind in enumerate(lg.indices):
                p = pos_of[int(ind)]
                U = bloch_operator(model, group, assignment, syms[p], D[p], k,
                                   k_out=k)[np.ix_(st, st)]
                for r, m in enumerate(rows):
                    Vs = V[np.ix_(st, list(m))]
                    chars[r, col] = np.trace(Vs.conj().T @ U @ Vs)
        except ValueError:
            out.append(None)                # the operator's own checks: nothing read here
            continue
        out.append(chars)
    return out


class _BandKey:
    r"""The band labels a candidate gives, comparable in the tables' order.

    For every high-symmetry point in the tables' order and every energy
    multiplet from the lowest up, the corep content the representation gives
    it once the part carries the candidate and every other part the one
    ``held`` holds it at, as the multiplicity of each of the point's coreps in
    the tables' own order.  Candidates compare at the first place their labels
    differ, the one whose content comes first in the tables' order of that
    point's coreps first.  A point where the labels cannot be read is left out
    for every candidate alike.  ``share(q, option)`` is part ``q``'s share of
    the characters under one of its options (:func:`_band_share`), measured
    the first time any part asks.
    """

    def __init__(self, group, fibers, share, held, part, n_parts):
        self.group, self.fibers, self.share = group, fibers, share
        self.held, self.part, self.n_parts = held, int(part), int(n_parts)
        self.orders = {}

    def _labels(self, i, option):
        from . import groups as _groups

        _name, k, lg, rows, _V = self.fibers()[i]
        total = np.zeros((len(rows), len(lg.indices)), dtype=complex)
        for q in range(self.n_parts):
            got = self.share(q, option if q == self.part else self.held(q))[i]
            if got is None:
                return None
            total = total + got
        found = _groups.identify_from_characters(self.group, k, total,
                                                 multiplet_dims=[len(m) for m in rows])
        if isinstance(found, Refusal):
            return None
        if i not in self.orders:
            tab = _groups.corep_table(self.group, k, kpname=found.kpname)
            self.orders[i] = () if isinstance(tab, Refusal) else tuple(tab)
        return tuple(tuple(-int(row.get(label, 0)) for label in self.orders[i])
                     for row in found.labels)

    def batch(self, options):
        """One comparable key per candidate: its labels, point by point."""
        keys = [[] for _o in options]
        for i in range(len(self.fibers())):
            got = [self._labels(i, o) for o in options]
            if any(g is None for g in got):
                continue
            for j, g in enumerate(got):
                keys[j].append(g)
        return [tuple(k) for k in keys]

    def reading(self, option) -> dict:
        """``{point: ((bands, {corep: m}), ...)}`` -- the labels one candidate gives every energy multiplet.

        Each multiplet is the band indices it holds, from the lowest up, with
        its corep content; a point whose labels cannot be read is left out.
        """
        from . import groups as _groups

        out = {}
        for i, (name, k, _lg, rows, _V) in enumerate(self.fibers()):
            total = None
            for q in range(self.n_parts):
                got = self.share(q, option if q == self.part else self.held(q))[i]
                if got is None:
                    total = None
                    break
                total = got if total is None else total + got
            if total is None:
                continue
            found = _groups.identify_from_characters(self.group, k, total,
                                                     multiplet_dims=[len(m) for m in rows])
            if isinstance(found, Refusal):
                continue
            out[str(name)] = tuple((tuple(int(b) for b in m), dict(labels))
                                   for m, labels in zip(rows, found.labels))
        return out


def _coupled_parts(classes, unpinned, parts):
    """The coupled parts of the crystal as sets of multiplet orbits, the anchor's first.

    ``parts`` is the measurement's, the anchor's own part first; an orbit the
    hoppings leave free of the anchor and that no part names is a part of its
    own.
    """
    nclass = int(np.max(classes)) + 1 if len(classes) else 0
    unpinned = frozenset(int(c) for c in (unpinned or ()))
    out = [frozenset(c for c in range(nclass) if c not in unpinned)]
    for part in tuple(parts or ())[1:]:
        got = frozenset(int(c) for c in part) & unpinned
        if got:
            out.append(got)
    for cls in sorted(unpinned):
        if not any(cls in got for got in out):
            out.append(frozenset({int(cls)}))
    return [p for p in out if p]


def _parts_apart(verdict, layout, sectors):
    """The orbitals of each part of the crystal a verdict read apart and the sectors keep apart.

    ``verdict`` is :func:`_verdicts_and_choices`'s, on the multiplets ``layout``
    holds, and ``sectors`` the reading of the conserved matrices at the
    declared precision (:func:`.._sectors.read_sectors`).  A part the verdict
    leaves free of the others can still be coupled to them, its ratio carried
    by the blocks a wider multiplet reads off the Hamiltonian, so parts are
    held apart only where no sector joins them: the finest partition both
    coarsen into.  ``None`` where that is one part, or nothing was read.
    """
    if not verdict or verdict.get("classes") is None or sectors is None:
        return None
    classes = np.asarray(verdict["classes"], dtype=int)
    parts = _coupled_parts(classes, verdict.get("free_classes"), verdict.get("parts"))
    if len(parts) < 2:
        return None
    joined = [tuple(sorted({int(o) for c in range(len(classes)) if int(classes[c]) in part
                            for o in layout._chan_orbs[c]})) for part in parts]
    joined += [tuple(sorted({int(o) for a in sector.atoms for o in sectors.atoms[int(a)]}))
               for sector in sectors.sectors]
    root = list(range(int(layout._chan_of.shape[0])))

    def _find(a):
        while root[a] != a:
            root[a] = root[root[a]]
            a = root[a]
        return a

    for orbs in joined:
        for o in orbs[1:]:
            ra, rb = _find(int(orbs[0])), _find(int(o))
            if ra != rb:
                root[max(ra, rb)] = min(ra, rb)
    blocks = {}
    for o in range(len(root)):
        blocks.setdefault(_find(o), []).append(int(o))
    if len(blocks) < 2:
        return None
    return tuple(tuple(b) for _r, b in sorted(blocks.items()))


def _held_sets(sectors, layout, model, group, assignment, spin, validated):
    """What the sectors read at the declared precision keep apart that a representation is read on.

    ``sectors`` is the reading of the conserved matrices at the declared
    precision (:func:`.._sectors.read_sectors`), and ``layout`` the multiplets
    held together by them (:func:`.._sectors.carried_reading`).  Returns
    ``{"copies", "shares"}``, or ``None`` where no precision was declared or
    there is neither:

    * ``copies``: every sector of several identical copies, each copy lying
      along orbitals of its own within that precision, as ``(basis, copies,
      dim, orbitals)`` -- its basis in the states, copy-major and held exactly
      along those orbitals (:func:`.._sectors.aligned`), its number of copies,
      the states one copy holds and the orbitals it lies on; on states that
      carry a spin each copy's states are its orbital factor's times the spin;
    * ``shares``, on states read without spin: every set of sectors of one
      copy that one multiplet holds together -- of orbits no operation of the
      group exchanges, the bonding and the antibonding state two orbitals
      make or orbitals nothing couples stated as one multiplet, or of one
      orbit, the square's ``p_x, p_y`` pair nothing couples -- as ``(bases,
      orbitals)``, each sector's basis held to its own share of every site,
      within the multiplets that hold it (:func:`.._sectors.held_shares`).

    What the Hamiltonian holds between such sectors, between copies or
    between them and the rest, and what tells one copy from another, lies
    below that precision, where it counts as absent; and the representations
    are read on them as exact (:func:`.._sectors.induced_orbits`).
    """
    from . import _sectors

    if sectors is None or sectors.tol is None:
        return None
    if _sectors.carried_reading(sectors, layout, tol=sectors.tol) is not sectors:
        return None
    sites = _sectors.atom_sites(sectors, layout)
    nstate = int(model.nstate)

    def _orbitals(ts):
        return tuple(sorted({int(o) for t in ts for a in sectors.sectors[int(t)].atoms
                             for o in sectors.atoms[int(a)]}))

    copies = []
    for t, s in enumerate(sectors.sectors):
        m, d = int(s.copies), int(s.dim)
        if m < 2:
            continue
        Q = _sectors.aligned(s, sectors.tol)
        exact = _sectors._snapped(Q.reshape(int(Q.shape[0]), m, d), sites, sectors.tol)
        if exact is None:
            continue                       # copies that do not lie along orbitals of their own
        basis = exact.reshape(int(Q.shape[0]), m * d)
        if sectors.su2:
            basis, d = _sectors.in_states(sectors, basis, nstate), 2 * d
        copies.append((basis, m, d, _orbitals((t,))))
    shares = []
    held = _sectors.held_sectors(sectors, layout, tol=sectors.tol)
    if not sectors.su2 and any(len(h) > 1 for h in held):
        syms = group.symmetries
        kept = [p for p, s in enumerate(syms) if int(s.ind) in set(validated.indices)]
        action = _sectors.sector_action(sectors, model, group, assignment, spin, validated,
                                        tol=sectors.tol)
        found = _sectors.sector_orbits(sectors, action, group, layout, kept, tol=sectors.tol)
        for roots, _multiplets in (found[1] if found is not None else ()):
            members = [t for r in roots for t in found[0][r]]
            if len(roots) < 2 and not any(len(set(h) & set(members)) > 1 for h in held):
                continue                   # one orbit, no multiplet holding two of its sectors
            if any(int(sectors.sectors[t].copies) != 1 for t in members):
                continue
            bases = _sectors.held_shares(sectors, members, sites, sectors.tol,
                                         on=_sectors.held_on(sectors, layout, held, members))
            if bases is not None:
                shares.append((bases, _orbitals(members)))
    return {"copies": tuple(copies), "shares": tuple(shares)} if copies or shares else None


def _lead_of(chans, first):
    """A coupled part's lead multiplet: ``first`` where the part holds it, its first otherwise."""
    return int(first) if int(first) in {int(c) for c in chans} else int(min(chans))


def _tables_target(group, table, rotations, content, orbitals, words):
    r"""``{ind: matrix}`` -- the tables' matrices of a multiplet's content, along its orbitals; or ``None``.

    ``content`` is ``{ind: chi(g)}`` over the site group ``table`` is the
    table of, and ``rotations`` each operation's Cartesian matrix.  Where every
    orbital was declared a function (``words``, ``{orbital: word}``) and those
    functions carry the content, each orbital is its own function; otherwise
    the content's irreps are laid along the orbitals in order, the lowest
    angular momentum first and then the table's own order, each on the
    functions the tables write it on (:func:`.._sitechars.irrep_functions`):
    ``(p_x, p_y)`` for ``E`` at a ``4mm`` site.  A matrix is the operation's
    action on those functions, ``(g·f)(r) = f(R⁻¹ r)``.  ``None`` where the
    content is no character of the site group, or an irrep in it has no such
    functions.
    """
    if table is None or any(int(cls[0]) not in content for cls in table.classes):
        return None
    indices = [int(i) for i in table.indices]
    orbitals = [int(o) for o in orbitals]
    shells = {}

    def _on(funcs):
        """``{ind: matrix}`` -- every operation on the functions ``[(degree, column)]``."""
        out = {}
        for ind in indices:
            M = np.zeros((len(funcs), len(funcs)), dtype=complex)
            for j, (dj, fj) in enumerate(funcs):
                if (ind, dj) not in shells:
                    shells[(ind, dj)] = _sitechars.shell_operator(group, ind, dj)
                image = shells[(ind, dj)] @ fj
                for i, (di, fi) in enumerate(funcs):
                    if di == dj:
                        M[i, j] = np.vdot(fi, image)
            out[ind] = M
        return out

    def _carries(target):
        return all(float(np.abs(T @ T.conj().T - np.eye(len(T))).max()) <= ASSEMBLED
                   and abs(complex(np.trace(T)) - complex(content[ind])) <= CHARACTER
                   for ind, T in target.items())

    if orbitals and all(o in words for o in orbitals):
        funcs = [_sitechars.function_of(words[o]) for o in orbitals]
        if all(f is not None for f in funcs):
            target = _on(funcs)
            if _carries(target):
                return target
    mult = _sitechars.decompose(table, np.asarray([complex(content[int(cls[0])])
                                                   for cls in table.classes]))
    funcs = []
    for row, n in enumerate(mult):
        k = int(round(float(np.real(n))))
        if abs(complex(n) - k) > CHARACTER or k < 0:
            return None
        got = _sitechars.irrep_functions(group, table, row, rotations) if k else None
        if k and got is None:
            return None
        if k:
            degree, basis = got
            funcs.extend([((degree, row), [(degree, basis[:, j])
                                           for j in range(basis.shape[1])])] * k)
    if sum(len(f) for _key, f in funcs) != len(orbitals):
        return None
    # the lowest angular momentum first, then the table's own order; each
    # irrep's own functions orthogonal to every other's, so the matrices are
    # block-diagonal along the orbitals
    blocks = [_on(f) for _key, f in sorted(funcs, key=lambda item: item[0])]
    target = {}
    for ind in indices:
        T = np.zeros((len(orbitals), len(orbitals)), dtype=complex)
        at = 0
        for on in blocks:
            size = int(on[ind].shape[0])
            T[at:at + size, at:at + size] = on[ind]
            at += size
        target[ind] = T
    return target if _carries(target) else None


def _in_tables_gauge(rec, winner, *, group, layout, syms, u_pos, cperm, site_tables,
                     rotations, words):
    r"""The matrices of the representation chosen for an orbit built by induction, in the tables' own.

    Which of the representations conjugate under what ``H`` conserves on the
    orbit -- a unitary on identical copies, a phase on each of several sectors
    one multiplet holds -- is no measurement, so it is fixed by the convention
    every matrix returned is written in: on the orbit's lead multiplet, the
    tables' matrices of the content it carries, laid along its orbitals in
    order (:func:`_tables_target`), wherever what ``H`` conserves reaches them
    (:func:`.._sectors.tables_gauge`) -- the copies and the sectors ordered as
    their orbitals.  Where it does not, the orbitals of the multiplet each
    holding a share of several irreps, the representation stays as it was
    induced, each sector the image of the first.  On states that carry a spin
    the tables' matrices are those of the orbital content times each
    operation's own lift on the spin.  ``rec`` is the orbit's record
    and ``winner`` the candidate chosen; the matrices are returned unchanged
    where the orbit holds one sector of one copy, which leaves nothing open.
    """
    from . import _sectors

    W = winner["W"]
    induced = rec.get("induced")
    if induced is None:
        return W
    frames = induced.sector_frames()
    if len(frames) < 2 and all(int(f[1]) < 2 for f in frames):
        return W
    lead = int(rec["lead"])
    orbitals, places = layout._chan_orbs[lead], None
    if layout._spinful:
        # the lead's orbitals (or declared pairs) once each, as its states reach them
        reading = getattr(induced, "reading", None) or induced.orbits[0].reading
        places = _sectors.spin_places(reading, layout._chan_states[lead])
        atoms = list(dict.fromkeys(a for a, _s in places))
        orbitals = [int(reading.atoms[a][0]) for a in atoms]
        places = [(atoms.index(a), s) for a, s in places]
    target = _tables_target(group, site_tables.get(int(layout._chan_site[lead])), rotations,
                            winner["contents"][lead], orbitals, words)
    if target is None:
        return W
    pos_of = {int(syms[p].ind): int(p) for p in u_pos}
    target = {pos_of[ind]: T for ind, T in target.items()
              if ind in pos_of and int(cperm[pos_of[ind]][lead]) == lead}
    if places is not None:
        at = [a for a, _s in places]
        spin = [s for _a, s in places]
        target = {p: T[np.ix_(at, at)] * _lift_of(syms[p], True)[np.ix_(spin, spin)]
                  for p, T in target.items()}
    local = {int(a): i for i, a in enumerate(rec["states"])}
    rows = [local[int(a)] for a in layout._chan_states[lead]]
    return _sectors.tables_gauge(frames, W, target, rows, syms,
                                 tol=getattr(layout, "_tol", None) or None)


def _labelling(D, raw, decls, *, pool, model, group, assignment, layout, syms, u_pos,
               cperm, classes, parts, ref_chan, site_tables, measured_at,
               multiplicity=(), params=None):
    r"""The labelling returned: what each coupled part of the crystal carries, chosen once.

    The hoppings fix the representation ``D`` only up to one one-dimensional
    character per coupled part of the crystal -- the multiplets they connect,
    directly or through others (``parts``) -- and, on a part that is an orbit
    of sectors built by induction -- identical copies, a conserved pseudospin,
    or uncoupled orbitals a symmetry exchanges that those characters do not
    carry -- up to the representation induced on it (``multiplicity``,
    :func:`_induced_blocks`).  A part's **candidates** are its characters in
    ``pool``, or the representations induced on it, under which every
    declaration made on it holds
    (:func:`_meets`, :func:`_holds_part`); where none meets them all, those met
    at the operations the measurement read each declaration at
    (``measured_at``).  A content declared over several parts is one
    requirement on all of them, so a part is chosen only where the parts
    still to be chosen can complete it, each among its own candidates.

    **One key** then chooses among each part's candidates, the same for every
    part, its lead multiplet the reference orbital's where the part holds it and
    its first otherwise (:func:`_lead_of`).  Every step scores the orbital
    factor, the spin-1/2 lift divided out:

    1. the part's content at its lead's site at the lowest angular momentum
       (:func:`_content_shells`), then over the crystal, then on the lead's
       own row;
    2. the letter the orbitals read as: each orbit's frame character -- over
       its sites, the trace of the block carrying each multiplet to its image,
       orbital by orbital -- averaged over the point group's classes, divided
       by the orbit's site count and read on the point group's own table at
       the lowest angular momentum, the lead's orbit first and then the part's
       orbits together;
    3. the band labels: at the high-symmetry points in the tables' order, and
       from the lowest energy multiplet up, the candidate whose content comes
       first in the tables' order of the point's coreps (:class:`_BandKey`),
       measured only where the keys before leave more than one;
    4. the group's own order of operations relative to ``D``
       (:func:`_distance_from_trivial`), or for the copies of a pseudospin the
       representation whose character is nearest the trivial one: where it
       decides, no band label rests on it.

    Returns ``{"of_chan", "swaps", "raw", "chosen", "candidates", "labelled",
    "parts", "part_states", "spaces", "lift_trace"}``: the character every
    multiplet is assembled with; the representation chosen on each orbit built
    by induction, as matrices on the orbit's states; every multiplet's
    character under the labelling's representation before its characters are
    multiplied in; per part the character chosen, and for each candidate its
    character with what the multiplets carry before it multiplies them
    (``None`` for a pseudospin's copies, which carry the chosen one's); per part
    whose candidates include one giving every multiplet the site corep the
    chosen one gives it and labelling some band differently, the band labels
    of every energy multiplet under the one chosen and under each such, with
    the key that set it aside; the parts' multiplets and states; and each
    pseudospin's content returned and every content its candidates give, in
    the key's order.
    """
    chan_states, su2 = layout._chan_states, layout._spinful
    nchan = len(chan_states)
    classes = np.asarray(classes, dtype=int)
    lift_trace = {int(syms[p].ind): complex(np.trace(_lift_of(syms[p], su2))) for p in u_pos}
    rotations = {int(syms[p].ind): np.asarray(syms[p].rotation_cart, dtype=float)
                 for p in u_pos}
    unitary = sorted(int(syms[p].ind) for p in u_pos)
    shells, lowest = {}, {}
    chans_of = [[int(c) for c in range(nchan) if int(classes[c]) in part] for part in parts]
    part_states = [sorted(int(a) for c in chans for a in chan_states[c]) for chans in chans_of]
    spaces = {tuple(sorted(int(c) for c in rec["multiplets"])): rec
              for rec in (multiplicity or ()) if rec.get("candidates")}
    names = _KEY_NAMES
    # the declared function of each orbital, where one was written: the tables'
    # matrices a representation the hoppings leave open is written in follow it
    words = {int(o): w for d in (decls or ()) if d.get("content") is not None
             for o, w in d["content"].words}

    def _options(chans):
        """A part's options: its characters, or the representations induced on it."""
        rec = spaces.get(tuple(sorted(chans)))
        if rec is None:
            # A COUPLED PART'S CHARACTERS.  The representations the hoppings
            # leave on a part whose sectors lie on whole multiplets are the
            # group's one-dimensional characters times ``D`` -- the same
            # mathematics as the representation induced on that orbit of one
            # copy, Ind(chi|K (x) w) = chi (x) Ind(w) -- so they are offered as
            # characters on the matrices the search settled: built by induction
            # instead they come back in another gauge, and a content declared on
            # such a part is not carried.
            return [{"chi": chi, "W": None} for chi in pool], None
        st = rec["states"]
        # each a representation induced on the orbit, its matrices built where
        # a key reads them
        return [_LazyOption(D, cand, st, chi={}, contents=cand["contents"],
                            character=cand["character"], k=int(rec["dim"]),
                            parts=cand.get("parts"), key=id(cand))
                for cand in rec["candidates"]], rec

    def _source(opt):
        return opt["contents"] if opt["W"] is not None else raw

    def _content(chans, opt):
        """``{ind: chi(g)}`` -- what a set of multiplets carries under one candidate."""
        out, src, chi = {}, _source(opt), opt["chi"]
        for c in chans:
            for ind, v in src[int(c)].items():
                out[int(ind)] = (out.get(int(ind), 0.0)
                                 + complex(chi.get(int(ind), 1.0)) * complex(v))
        return out

    def _shells(tab, theta):
        return _content_shells(tab, theta, rotations, shells=shells, lowest=lowest)

    def _order(opt):
        if opt["W"] is not None:
            return _distance_from_trivial({int(i): complex(v) / opt["k"]
                                           for i, v in opt["character"].items()})
        return _distance_from_trivial({ind: complex(opt["chi"].get(ind, 1.0))
                                       for ind in unitary})

    # THE LETTER THE ORBITALS READ AS: an orbit's frame character, read on the
    # representation as built, on the point group's own single-valued table
    tables = {}

    def _point():
        if "point" not in tables:
            tables["point"] = _sitechars.site_group_table(group, unitary)
        return tables["point"]

    orbit_of_site = np.asarray(assignment.orbit_of_site, dtype=int)
    atoms_of = {int(c): _atom_blocks(chan_states[c], su2) for c in range(nchan)}
    lift_of = {int(p): _lift_of(syms[p], su2) for p in u_pos}

    def _frames(chans, M):
        """``{orbit: (phi, n_sites, width)}`` -- the frame character of each orbit of a part.

        ``phi(g)`` sums, over the orbit's sites, the trace of the orbital factor
        of ``M(g)``'s block carrying the site's multiplets to their images:
        orbital ``k`` of a multiplet read against orbital ``k`` of its image,
        the spin-1/2 lift divided out atom by atom.
        """
        by_orbit = {}
        for c in chans:
            by_orbit.setdefault(int(orbit_of_site[int(layout._chan_site[int(c)])]),
                                []).append(int(c))
        out = {}
        for orbit, cs in sorted(by_orbit.items()):
            phi = {}
            for p in u_pos:
                total = 0j
                for c in cs:
                    c2 = int(cperm[p][c])
                    total += sum(complex(_proportionality(
                        np.asarray(M[p])[np.ix_(a2, a1)], lift_of[int(p)])[0])
                        for a2, a1 in zip(atoms_of[c2], atoms_of[c]))
                phi[int(syms[p].ind)] = total
            sites = {int(layout._chan_site[c]) for c in cs}
            width = sum(len(atoms_of[c]) for c in cs) // max(len(sites), 1)
            out[orbit] = (phi, len(sites), int(width))
        return out

    def _letter(frame, chi):
        """``(single shell, per orbital)`` of the letter one orbit reads as under ``chi``.

        The frame character times ``chi``, averaged over each class of the point
        group and divided by the orbit's site count, decomposed on the point
        group's own table with its multiplicities rounded: the point-group
        representation the orbitals are written as, at the angular momentum
        :func:`_content_shells` reads it at.  A rounding that is no
        representation -- a negative multiplicity, or not the orbit's width --
        scores as no character of the table does, and so ties.
        """
        phi, m, width = frame
        point = _point()
        theta = [sum(complex(chi.get(int(i), 1.0)) * phi[int(i)] for i in cls)
                 / (len(cls) * m) for cls in point.classes]
        n = [int(round(float(np.real(x)))) for x in _sitechars.decompose(point, theta)]
        if (any(k < 0 for k in n)
                or sum(k * int(point.dims[q]) for q, k in enumerate(n)) != int(width)):
            return (_L_MAX + 1, 10 ** 6)
        content = {int(i): sum(k * complex(point.character_on(q, i))
                               for q, k in enumerate(n) if k) for i in point.indices}
        return _content_shells(point, content, rotations, shells=shells, lowest=lowest)

    def _frame_key(chans, lead_orbit):
        """The letter of the lead's orbit, then the letters summed over the part's orbits."""
        frames = {}

        def key(opt):
            src = opt["key"] if opt["W"] is not None else 0
            if src not in frames:
                frames[src] = _frames(chans, opt["D"] if opt["W"] is not None else D)
            scores = {o: _letter(f, opt["chi"]) for o, f in frames[src].items()}
            return (scores[lead_orbit], tuple(sum(s) for s in zip(*scores.values())))
        return key

    # THE BAND LABELS, measured once and only where the keys before them tie:
    # every part's share of every energy multiplet's character at the
    # high-symmetry points (:func:`_band_fibers`, :func:`_band_share`)
    measured = {}

    def _fibers():
        if "fibers" not in measured:
            measured["fibers"] = _band_fibers(model, group, D, syms, params=params)
        return measured["fibers"]

    def _share(q, opt):
        if opt is not None and opt["W"] is not None:
            key = ("W", q, opt["key"])
            if key not in measured:
                measured[key] = _band_share(model, group, assignment, opt["D"], syms,
                                            _fibers(), part_states[q])
        else:
            key = ("D", q)
            if key not in measured:
                measured[key] = _band_share(model, group, assignment, D, syms, _fibers(),
                                            part_states[q])
        base = measured[key]
        if opt is None or not opt["chi"]:
            return base
        out = []
        for (_name, _k, lg, _rows, _V), chars in zip(_fibers(), base):
            f = np.asarray([complex(opt["chi"].get(int(i), 1.0)) for i in lg.indices],
                           dtype=complex)
            out.append(None if chars is None else chars * f[None, :])
        return out

    held_at = []

    def _held(q):
        """The candidate part ``q`` is held at while another is chosen: its own, once chosen."""
        return held_at[q] if q < len(held_at) else None

    def _site_irrep(theta, chi):
        return tuple(sorted(
            (int(ind),
             round(float(np.real(v * complex(chi.get(int(ind), 1.0))
                                 * lift_trace.get(int(ind), 1.0))), 6),
             round(float(np.imag(v * complex(chi.get(int(ind), 1.0))
                                 * lift_trace.get(int(ind), 1.0))), 6))
            for ind, v in theta.items()))

    # WHAT EACH PART MAY BE, before any is chosen: its options, and among them
    # those every declaration made on it alone holds under
    part_of_chan = {int(c): q for q, chans in enumerate(chans_of) for c in chans}
    options_of, records = [], []
    for chans in chans_of:
        options, rec = _options(chans)
        options_of.append(options)
        records.append(rec)
    mine_of = [[d for d in decls
                if d["channels"] and all(int(c) in set(chans) for c in d["channels"])]
               for chans in chans_of]
    # ... and the declarations a part shares with others, one requirement on all
    # of them, read on the parts' options together
    spanned = {id(d): sorted({part_of_chan[int(c)] for c in d["channels"]}) for d in decls
               if d["channels"] and len({part_of_chan[int(c)] for c in d["channels"]}) > 1}
    shared = [d for d in decls if id(d) in spanned]
    needs = {}

    def _ok(q, opt, measured_only=False):
        """Whether every declaration made on part ``q`` alone holds under one of its options."""
        src = _source(opt)
        for d in mine_of[q]:
            if d.get("partial"):
                if not _holds_part(d, lambda _c, chi=opt["chi"]: chi, src, site_tables,
                                   lift_trace):
                    return False
                continue
            key = (id(d), id(src))
            if key not in needs:
                needs[key] = _declared_needs(d, src, lift_trace)
            over = (measured_at.get(tuple(int(o) for o in d["orbitals"]), set())
                    if measured_only else None)
            if not _meets(opt["chi"], needs[key], over=over):
                return False
        return True

    def _together(d, pick, measured_only=False):
        """Whether a declaration spanning several parts holds, each part at its option in ``pick``."""
        def assigned(c):
            return pick[part_of_chan[int(c)]]["chi"]
        src = {int(c): _source(pick[part_of_chan[int(c)]])[int(c)] for c in d["channels"]}
        if d.get("partial"):
            return _holds_part(d, assigned, src, site_tables, lift_trace)
        carried = {c: {g: complex(assigned(c).get(g, 1.0)) * complex(v) for g, v in row.items()}
                   for c, row in src.items()}
        over = (measured_at.get(tuple(int(o) for o in d["orbitals"]), set())
                if measured_only else None)
        return _meets({}, _declared_needs(d, carried, lift_trace), over=over)

    own_of = {}

    def _own(q):
        """Part ``q``'s options under which its own declarations hold."""
        if q not in own_of:
            own_of[q] = ([o for o in options_of[q] if _ok(q, o)]
                         or [o for o in options_of[q] if _ok(q, o, True)] or options_of[q])
        return own_of[q]

    def _completes(q, opt, measured_only=False):
        """Whether ``opt`` for part ``q`` leaves the parts after it a way to meet what they share."""
        live = [d for d in shared if spanned[id(d)][-1] >= q]
        if not live:
            return True
        later = sorted({p for d in live for p in spanned[id(d)] if p > q})
        for combo in itertools.product(*(_own(p) for p in later)):
            pick = dict(enumerate(held_at))
            pick[q] = opt
            pick.update(zip(later, combo))
            if all(_together(d, pick, measured_only) for d in live
                   if all(p in pick for p in spanned[id(d)])):
                return True
        return False

    chosen, candidates, open_of, swaps, spaces_out = [], [], [], [], []
    raw_after = {int(c): dict(v) for c, v in raw.items()}
    for i, chans in enumerate(chans_of):
        options, rec = options_of[i], records[i]
        cands = [o for o in options if _ok(i, o) and _completes(i, o)]
        if not cands:
            cands = [o for o in options if _ok(i, o, True) and _completes(i, o, True)]
        if not cands:
            # what the parts share is said on the rows where no labelling carries it
            cands = ([o for o in options if _ok(i, o)]
                     or [o for o in options if _ok(i, o, True)])
        if not cands:
            if int(ref_chan) in chans:     # pragma: no cover - the verdict rules it out
                raise InternalError(
                    "the declared orbital content was found covariant on this group, and no "
                    "labelling of the crystal carries every declared multiplet in its declared "
                    "irrep on the built matrices. The verdict and the build are two readings of "
                    "one measurement and have to agree; the two disagreeing is a defect in this "
                    "package, not a tolerance to loosen.")
            cands = list(options)
        lead = _lead_of(chans, ref_chan)
        here = int(layout._chan_site[lead])
        at = {}
        for c in chans:
            at.setdefault(int(layout._chan_site[c]), []).append(int(c))
        def _copy_by_copy(o, at=at, here=here):
            """Identical copies read copy by copy: each holding one content, at the lowest angular momentum.

            A representation that puts the same content on every copy of every
            multiplet -- ``k`` copies of one content -- comes first, ordered by
            what one copy holds at the lead's site; any other after them.
            """
            k = int(o.get("k") or 0)
            if k < 2:
                return (0, (0, 0))
            one = {}
            for s, cs in sorted(at.items()):
                for c in cs:
                    got = _one_copy(site_tables.get(s), _content((c,), o), k)
                    if got is None:
                        return (1, (0, 0))
                    if s == here:
                        for ind, v in got.items():
                            one[int(ind)] = one.get(int(ind), 0.0) + complex(v)
            return (0, _shells(site_tables.get(here), one))

        def _part_by_part(o, at=at, here=here):
            """Orbits read together read one by one, each at the lowest angular momentum.

            Orbits of sectors whose multiplets hold shares of one another's are
            each a choice of their own (:class:`.._sectors.SharedOrbits`), so
            each is read on its own share of the lead's site, in turn; any
            other option reads alike here.
            """
            parts = o.get("parts")
            if not parts:
                return ()
            out = []
            for contents in parts:
                share = {}
                for c in at.get(here, ()):
                    for ind, v in contents.get(int(c), {}).items():
                        share[int(ind)] = share.get(int(ind), 0.0) + complex(v)
                out.append(_shells(site_tables.get(here), share))
            return tuple(out)

        keys = [
            _part_by_part,
            _copy_by_copy,
            lambda o, at=at, here=here: _shells(site_tables.get(here),
                                                _content(at.get(here, ()), o)),
            lambda o, at=at: (tuple(sum(w) for w in zip(*[
                _shells(site_tables.get(s), _content(cs, o)) for s, cs in sorted(at.items())]))
                if at else (0, 0)),
            lambda o, lead=lead: _shells(site_tables.get(int(layout._chan_site[lead])),
                                         _content((lead,), o)),
            _frame_key(tuple(chans), int(orbit_of_site[here])),
            _BandKey(group, _fibers, _share, _held, i, len(chans_of)),
            _order]
        kept, fate = _choose(cands, keys)
        winner = cands[kept]
        held_at.append(winner)
        # THE LABELLINGS THE SITE IRREPS LEAVE OPEN: every candidate the
        # Hamiltonian and the declarations leave that gives each multiplet of
        # the part the site irrep the one kept gives it, with the key that set
        # it aside -- the chooser's own decision, whatever freedom it was
        want = [_site_irrep(_source(winner)[c], winner["chi"]) for c in chans]
        open_of.append((cands, [(j, names[min(int(fate[j]), len(names) - 1)])
                                for j, o in enumerate(cands)
                                if j != kept and [_site_irrep(_source(o)[c], o["chi"])
                                                  for c in chans] == want]))
        if winner["W"] is None:
            # a character on ``D``
            chosen.append(winner["chi"])
            candidates.append([(o["chi"], _source(o)) for o in cands])
        else:
            # A PSEUDOSPIN'S COPIES, or uncoupled orbitals a symmetry
            # exchanges: the representation chosen is assembled whole
            for c in chans:
                raw_after[int(c)] = dict(winner["contents"][c])
            swaps.append((tuple(int(a) for a in rec["states"]),
                          _in_tables_gauge(rec, winner, group=group, layout=layout,
                                           syms=syms, u_pos=u_pos, cperm=cperm,
                                           site_tables=site_tables, rotations=rotations,
                                           words=words)))
            chosen.append({})
            # what each multiplet holds under the representations the
            # declarations leave, the rows' own reading of how they are known
            candidates.append([({}, o["contents"]) for o in cands])
            if int(rec["dim"]) > 1:
                # a pseudospin's own record; an orbit of one copy holds none
                ranked, lead_c = [], int(rec["lead"])
                for o in sorted(options, key=lambda o: (keys[-7](o), keys[-6](o),
                                                        keys[-5](o), keys[-4](o), _order(o))):
                    content = o["contents"][lead_c]
                    if not any(set(content) == set(r) and all(
                            abs(complex(content[x]) - complex(r[x])) <= 1e-6 for x in content)
                            for r in ranked):
                        ranked.append(content)
                declared = bool(mine_of[i]) or any(i in spanned[id(d)] for d in shared)
                spaces_out.append({"record": rec, "content": dict(winner["contents"][lead_c]),
                                   "contents": tuple(ranked), "declared": declared,
                                   "open": len({_site_irrep(o["contents"][lead_c], {})
                                                for o in cands})})
    # ... AND THE BAND LABELS EACH OF THEM GIVES, read by the chooser's own band
    # key with every other part at the labelling kept: those that label a band
    # differently are the choice the band labels rest on
    labelled = []
    for i, (cands, ties) in enumerate(open_of):
        if not ties:
            continue
        reader = _BandKey(group, _fibers, _share, _held, i, len(chans_of))
        mine = reader.reading(held_at[i])
        others = tuple((name, got) for name, got in
                       ((name, reader.reading(cands[j])) for j, name in ties) if got != mine)
        if others:
            labelled.append({"part": i, "chosen": mine, "others": others})
    # the matrices built for representations no labelling returns are not kept
    winners = {id(getattr(w, "_cand", None)) for w in (o["W"] for o in held_at
                                                       if o["W"] is not None)}
    for rec in records:
        if rec is not None and rec.get("induced") is not None:
            for cand in rec["candidates"]:
                if id(cand) not in winners:
                    cand.release()
    of_chan = {}
    for i, chans in enumerate(chans_of):
        for c in chans:
            of_chan[int(c)] = chosen[i]
    return {"of_chan": of_chan, "swaps": tuple(swaps), "raw": raw_after, "chosen": chosen,
            "candidates": candidates, "labelled": tuple(labelled), "parts": chans_of,
            "part_states": part_states, "spaces": tuple(spaces_out), "lift_trace": lift_trace}


def _assembled(D, choice, syms, layout):
    r"""``{position: matrix}`` -- the representation with the labelling in it.

    The representation each pseudospin's copies carry, as chosen, on its
    orbit's states; then every multiplet's columns -- its block, carrying it to
    its image -- times the value its coupled part's character takes on the
    operation: a character per part, since the operations carry each part onto
    itself, and one crystal-wide character multiplies the whole matrix.
    """
    out = {}
    for p, M in D.items():
        ind = int(syms[p].ind)
        N = np.array(M, dtype=complex, copy=True)
        for states, W in choice["swaps"]:
            if p in W:
                st = np.asarray(states, dtype=int)
                N[np.ix_(st, st)] = W[p]
        for c in range(int(layout._nchan)):
            f = complex(choice["of_chan"][int(c)].get(ind, 1.0))
            if abs(f - 1.0) <= 1e-12:
                continue
            N[:, np.asarray(layout._chan_states[int(c)], dtype=int)] *= f
        out[p] = N
    return out


def _atomic_limit_table(raw, worst, decls, choice, *, group, assignment, model, layout,
                        syms, kept, checks, pool, site_tables, ident, note="", spin=None,
                        ref_chan):
    r"""The atomic-limit table of the representation returned, read off it and nothing else.

    Every character here was measured before the labelling was chosen
    (:func:`_multiplet_characters`), and the labelling itself is
    :func:`_labelling`'s; this writes them down.  A row's site irrep is what
    its multiplet carries under the labelling returned, named from the
    tables (:func:`_named_content`).  How it is known
    (:attr:`AtomicLimitRow.determined_by`) is ``"declaration"`` for a declared
    row the returned matrices carry, ``"declaration not carried"`` for one they
    do not -- a declaration the measurement could not read, such as a content
    declared on part of what a multiplet holds, which no labelling of these
    blocks carries together with the rest -- and otherwise read off the
    candidates of the row's coupled part: one site irrep across all of them is
    ``"hamiltonian"``, several ``"lowest-l"``.

    Characters live on **multiplets** (``layout._chan_of[a]`` is orbital
    ``a``'s, and ``layout._chan_orbs[c]`` its orbitals); the record is still one
    row per model orbital, so a declared spinor pair's two orbitals report the
    one character of the pair, each flagged ``up``/``down`` in the note and in
    :meth:`AtomicLimitTable.summary`.
    """
    chan_of, chan_orbs = layout._chan_of, layout._chan_orbs
    lattice = np.asarray(group.lattice, dtype=float)
    ind_of = {p: int(syms[p].ind) for p in kept}
    names = {ind_of[p]: _sitechars.operation_name(syms[p], lattice)
                        + ("'" if syms[p].time_reversal else "") for p in kept}
    site_of = np.asarray(assignment.site_of_orbital, dtype=int)
    norb = int(model.norb)
    of_chan, lift_trace = choice["of_chan"], choice["lift_trace"]
    part_of_chan = {int(c): i for i, chans in enumerate(choice["parts"]) for c in chans}
    final = {c: {ind: complex(of_chan[c].get(int(ind), 1.0)) * v for ind, v in chi.items()}
             for c, chi in raw.items()}

    def _site_irrep(theta, chi):
        return tuple(sorted(
            (int(ind),
             round(float(np.real(v * complex(chi.get(int(ind), 1.0))
                                 * lift_trace.get(int(ind), 1.0))), 6),
             round(float(np.imag(v * complex(chi.get(int(ind), 1.0))
                                 * lift_trace.get(int(ind), 1.0))), 6))
            for ind, v in theta.items()))

    # WHETHER EACH DECLARATION IS CARRIED by the matrices returned.  Where the
    # verdict stage measured a declaration this is so by construction; where it
    # could not -- a content declared on part of what a multiplet holds, a
    # block the hoppings leave free of the scale it was read against, blocks
    # too noisy to settle -- a declaration no labelling of these blocks
    # carries is said on its rows, never returned as though it held.
    carried = {}
    for d in decls:
        if d.get("partial"):
            carried[id(d)] = _holds_part(d, lambda c: of_chan[int(c)], raw, site_tables,
                                         lift_trace)
        else:
            # read on the characters returned, so it asks the trivial character
            carried[id(d)] = _meets({}, _declared_needs(d, final, lift_trace))

    decl_of = {}
    for d in decls:
        for a in d["orbitals"]:
            decl_of[int(a)] = d
    # WHAT EACH DECLARATION IS RETURNED AS: what the multiplets it was read on
    # carry together under the labelling returned -- its own multiplet's site
    # irrep, or for a content read over several of a site's multiplets their
    # sum (`_declared_blocks`), which no one row names
    returned = {}
    for d in decls:
        a0 = int(d["orbitals"][0])
        ch0, tab0 = int(chan_of[a0]), site_tables[int(site_of[a0])]
        if d.get("block") is not None or layout._chan_kind[ch0] != "measured":
            theta0 = {ind: sum(final[int(c)].get(int(ind), 0.0) for c in d["channels"])
                      for ind in tab0.indices if int(ind) in final[ch0]}
        else:
            theta0 = dict(final[ch0])
        returned[id(d)] = _named_content(tab0, _copies_of_one_irrep(tab0, theta0)[0])[0]
    spin_note = {}
    if spin is not None:
        for c, (up, dn) in enumerate(spin.pairs):
            spin_note[int(up)] = f"spin up of pair {c}, partner orbital {int(dn)}"
            spin_note[int(dn)] = f"spin down of pair {c}, partner orbital {int(up)}"
    site_symbols = {s: _sitechars.point_group_symbol(group, tab.indices)
                    for s, tab in site_tables.items()}
    wyckoff = _wyckoff_sites(group, assignment)
    functions = _declared_functions(decls)
    rows = []
    for a in range(norb):
        s = int(site_of[a])
        tab = site_tables[s]
        ch = int(chan_of[int(a)])
        d = decl_of.get(a)
        # the declaration written for THIS orbital, which a content declared
        # over several multiplets holds among its parts
        part = d if d is None or not d.get("groups") else next(
            g for g in d["groups"] if int(a) in {int(o) for o in g["orbitals"]})
        # EVERY ROW IS ITS OWN MEASURED MULTIPLET'S, whatever the spelling: a
        # content declared over several of a site's multiplets is read against
        # what they carry together, and stays the row's `declared_irrep`, while
        # each row names what its own multiplet carries
        if d is not None and d.get("block") is None:
            members, chans = tuple(d["orbitals"]), tuple(d["channels"])
        else:
            members, chans = tuple(int(o) for o in chan_orbs[ch]), (ch,)
        theta = (dict(final[int(ch)]) if layout._chan_kind[ch] == "measured"
                 else {ind: sum(final[int(c)].get(int(ind), 0.0) for c in chans)
                       for ind in tab.indices if int(ind) in final[ch]})
        # A multiplet holding several copies of one site irrep -- copies written
        # in a basis rotated from site to site, which no orbital subset splits --
        # is an isotypic component, and every vector of it carries that irrep:
        # each of its rows is the irrep, with the irrep's own characters and
        # dimension, and the site holds as many copies as the rows say.  A
        # multiplet the site's own operations glue across different irreps
        # keeps its sum.
        theta = _copies_of_one_irrep(tab, theta)[0]
        # The returned label is what the orbitals carry in the site group of the
        # group that is RETURNED, read off the multiplet's character and nothing
        # else, and named for the site irrep the STATES carry -- the orbital
        # character times the operations' own lift, the double-valued site corep
        # where the states carry spin.
        label, named_by = _named_content(tab, theta)
        note_a = "" if part is None else part["note"]
        if d is not None:
            determined_by = "declaration" if carried[id(d)] else _NOT_CARRIED
            if not carried[id(d)]:
                needs = _declared_needs(d, raw, lift_trace) if not d.get("partial") else None
                alone = any((_holds_part(d, lambda _c, chi=chi: chi, raw, site_tables,
                                         lift_trace) if needs is None
                             else _meets(chi, needs)) for chi in pool)
                said = d["label"] or repr(d["spec"])
                note_a = ((note_a + "; " if note_a else "")
                          + f"declared {said}, not carried: no one-dimensional character "
                          "the hoppings leave open carries it"
                          + (" together with the other declarations" if alone else "")
                          + f", and the Hamiltonian's own blocks carry "
                          f"{returned[id(d)] or 'a character no irrep of this site group names'}"
                          " under the one returned")
        else:
            options = choice["candidates"][part_of_chan[ch]]
            seen = {_site_irrep({int(ind): v for ind, v in (raw if src is None else src)[ch].items()
                                 if int(ind) in theta}, chi) for chi, src in options}
            determined_by = "hamiltonian" if len(seen) <= 1 else "lowest-l"
        if a in spin_note:
            note_a = (note_a + "; " if note_a else "") + spin_note[a]
        content = None if part is None else part.get("content")
        written = dict(getattr(content, "written", ()) or ())
        if (named_by == "unnamed") == bool(label):    # pragma: no cover - a defect guard
            raise InternalError(
                f"the site table of site {s} names orbital {int(a)} {label!r}, "
                f"named by {named_by!r}. A label the tables gave is a label and says "
                "where it comes from; only a row nothing named may be blank, and it "
                "reads `named_by = \"unnamed\"`. The two disagreeing is a defect in "
                "this package, not a tolerance to loosen.")
        here = wyckoff.get(s)
        rows.append(AtomicLimitRow(
            orbital=int(a), site=s,
            wyckoff="" if here is None else f"{int(here.multiplicity)}{here.letter}",
            site_symmetry=site_symbols.get(s, ""), function=functions.get(a, ""),
            multiplet=members, irrep=label, named_by=named_by,
            determined_by=determined_by,
            dim=int(round(float(np.real(theta.get(int(syms[ident].ind), 1.0)))
                          * (2 if group.spinor else 1))),
            site_group=tuple(int(i) for i in tab.indices),
            character={int(k): complex(v) for k, v in theta.items()},
            generator_character={tab.names[int(r)]: complex(theta.get(int(r), 0.0))
                                 for r in tab.class_reps},
            declared_as=(None if part is None else written.get(int(a), part["spec"])),
            declared_irrep="" if content is None else str(content.label),
            declared_in="" if content is None else str(content.declared_in),
            reduced_in="" if content is None else str(content.reduced_in),
            declaration_broke=() if content is None else tuple(content.broke),
            note=note_a, residual=float(worst)))

    return AtomicLimitTable(
        orbitals=tuple(rows),
        declarations=tuple({"orbitals": tuple(d["orbitals"]), "declared_as": d["spec"],
                            "declared_irrep": d["label"], "dim": d["dim"],
                            "note": d["note"],
                            "declared_in": (d["content"].declared_in
                                            if d.get("content") is not None else ""),
                            "reduced_in": (d["content"].reduced_in
                                           if d.get("content") is not None else ""),
                            "broke": (tuple(d["content"].broke)
                                      if d.get("content") is not None else ()),
                            "mulliken": (dict(d["content"].mulliken)
                                         if d.get("content") is not None else {}),
                            "returned": returned[id(x)]}
                           for x in decls for d in x.get("groups", (x,))),
        checks=_returned_rows(checks),
        frame=dict(names), residual=float(worst), site_tables=dict(site_tables),
        spin=spin,
        _labellings={"parts": tuple(tuple(int(a) for a in states)
                                    for states in choice["part_states"]),
                     "choices": tuple(choice["labelled"])})


def _returned_rows(rows):
    """The measured-versus-declared rows as a record returns them.

    What each operation would have needed of the one-dimensional character, and
    which coupled part of the crystal it was read in, are the package's own
    reading of the row and stay inside it; a returned row says what was
    measured and what was declared.
    """
    return tuple({k: v for k, v in r.items() if k not in ("required", "part")}
                 for r in rows)


def _wyckoff_sites(group, assignment):
    """``{site: SiteReference}`` — each site's Wyckoff position in the tables' setting.

    Multiplicity and letter, as :func:`..groups.wyckoff_sites_tables` reads
    them; a site is absent where the tables do not reach the cell, which is the
    same condition under which the assignment carries no letter for it.
    """
    from . import groups as _groups

    if not tuple(getattr(assignment, "wyckoff_tables", ()) or ()):
        return {}
    sites = _groups.wyckoff_sites_tables(group, assignment.site_positions)
    if isinstance(sites, Refusal):
        return {}
    return dict(enumerate(sites))


def _declared_functions(decls) -> dict:
    """``{orbital: word}`` — the orbital function each orbital was declared to be.

    The word ``characters=`` was given, spelled as ``characters=`` spells it,
    on every orbital a function word was written for, whatever the dimension of
    the multiplet it names: a shorthand (``"px"``), or a shell (``"p"``) on
    every orbital it was written on.  An orbital declared by its irrep, a sum
    of irreps or generator characters, or not declared at all, has none: which
    function a Wannier function is inside its site is a choice of basis the
    Hamiltonian does not measure, and its site irrep is what says what it
    carries.
    """
    out = {}
    for d in decls:
        content = d.get("content")
        for a, word in (getattr(content, "words", ()) or ()):
            out[int(a)] = str(word)
    return out


# --------------------------------------------------------------------------
# the container
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SpaceGroupRep:
    """The model's representation of its space group: one matrix per operation, on the model's states.

    ``r.rep`` of :func:`..groups.realized_group`, through which every band
    label is read: :func:`..sewing.hsp_table` and the calls that label bands
    take it.  ``rep.summary()`` prints what it holds.

    Attributes
    ----------
    group : ..groups.GroupData
        The group represented.
    operations : ..groups.OperationTable
        Its operations by index, each with its Seitz symbol and its name in the
        model's frame.
    indices : tuple of int
        The indices of the operations represented, in the group's order.
    matrices : dict
        ``{ind: D(g)}``, each operation's ``k``-independent matrix on the
        model's states, state ``2 × orbital + spin`` on a spinful model, up
        first: the representation induced from :attr:`characters`, in the
        tables' convention, spin along ``+z`` in a right-handed frame.
    antiunitary : tuple of int
        The operations combined with time reversal, each acting as its matrix
        times complex conjugation.
    characters : AtomicLimitTable
        What each orbital carries at its site, from which the matrices are
        induced.
    multiplets : SiteMultiplets
        The multiplets each site's orbitals form.
    spin : ..spin.SpinBasis or None
        The declared spinor basis of a spinless model carrying spin in its
        orbital index, each pair one spinor Wannier function; ``None`` for
        every other model.
    environment : str or None
        A layer's environment, ``"substrate"`` or ``"free-standing"``; ``None``
        for a crystal periodic in three directions.
    free_standing_group : dict or None
        For a layer on a substrate, the free-standing layer's group as
        :func:`..groups.realized_group` measured it beside this one and found
        it to agree, ``{"number_str", "name", "size"}``, which
        :func:`..ebr.symmetry_indicator` names beside a substrate's class;
        ``None`` otherwise.
    residuals : dict
        ``{ind: max_k ‖D H(k) D† − H(Rk)‖}``, in the units of ``H``: how far each
        matrix misses carrying the Hamiltonian onto itself.
    product_residual, factor_residual : float
        How far the matrices miss multiplying as the group does, ``D(g₁)`` times
        ``D(g₂)``, conjugated where ``g₁`` is antiunitary, against the group's
        spinor factor times ``D(g₁g₂)``; and the same for the Bloch operators,
        whose products carry the translations' phases (:func:`bloch_operator`),
        ``nan`` where not measured.
    model, assignment : pythtb.TBModel, ..groups.OrbitalAssignment
        The model, and the orbitals' Wyckoff orbits, it was built on.
    """

    model: Any
    group: Any
    assignment: Any
    indices: tuple
    # the `irrep` operation objects themselves, in index order
    _operations: tuple
    matrices: dict
    antiunitary: tuple
    residuals: dict
    # `{ind: ‖D^n − ζ·1‖}`: how far each matrix misses its canonical norm, `n` its
    # operation's order and `ζ` the sign the group's own table gives that power
    _norm_residuals: dict
    # `{ind: ‖T conj(D) T† − D‖}`: how far each unitary operation's matrix misses
    # commuting with time reversal, the relation the phases are anchored by;
    # empty where the group has no time reversal
    _anchor_residuals: dict
    product_residual: float
    factor_residual: float
    # the generators of the represented unitary operations, by index, whose
    # matrices the rest are propagated from
    _generators: tuple
    # each unitary operation, by index, as a shortest word in `_generators`, a
    # letter being a generator's place among them: the tree the matrices are
    # propagated along
    _words: dict
    # the discrete choice the build settled on: the orientation branch and the
    # relative signs, {"orientation": ±1, "signs": {(generator ind, orbit): phase}}
    _choice: dict
    # the build's record of its search for that choice: how the candidates were
    # found, read off the detector's solution space or enumerated, what each of
    # the three gates measured on them and which one each failed, the block
    # phases read off, and how far the blocks accepted were from composing by
    # the group's table ("composition", which `symmetrize` reads)
    _search: dict
    # which orbit of multiplets each multiplet belongs to, as the build indexes them
    _classes: np.ndarray
    # `{"orbital", "multiplet", "orbit"}`: the orbital the lowest-l rule reads
    # first, orbital 0 by default, the multiplet it sits in and that multiplet's
    # orbit; on a declared spinor basis also `"spin_pair"`, its `(up, dn)` pair
    _reference: dict
    # the operations the Hamiltonian breaks, one entry each as the measurement
    # refuted it: `{"ind", "refuted_by", "residual"}`
    _failed: tuple
    # the k-points the covariance was measured at, which `covariance()` reads again
    _battery: np.ndarray
    multiplets: Any
    # how each matrix's phase was fixed, operation by operation
    _normalization: dict = field(default_factory=dict, repr=False)
    # the index of the pure time reversal, `None` without one: what
    # `antiunitary_operator` returns with its matrix
    _time_reversal: int | None = None
    characters: Any = None
    spin: Any = None
    environment: str | None = None
    free_standing_group: dict | None = None
    # whether the orientation branch was a convention rather than a measurement:
    # true where g and its inverse closed every gate with the same labels, which
    # only a group of involutions allows (mm2, P-1, a spinful 2/m), and then
    # `_choice["orientation"]` is +1 by convention
    _branch_tie: bool = False

    # -- lookups -----------------------------------------------------------
    def __len__(self) -> int:
        return len(self.indices)

    def __contains__(self, ind) -> bool:
        return int(ind) in self.matrices

    @property
    def operations(self):
        """The operations represented, by index, each with its Seitz symbol and its name in the model's frame.

        ``rep.operations[5].seitz`` is operation 5's Seitz symbol in the tables'
        setting and ``.name`` its name in the model's frame, one row per key of
        :attr:`matrices`.
        """
        from .groups import operations

        return operations(self)

    @property
    def orbits(self) -> tuple:
        """The sites, grouped into the orbits the operations carry them through: ``assignment.orbits``."""
        return self.assignment.orbits

    @property
    def site_orbitals(self) -> tuple:
        """The orbitals on each site: ``assignment.site_orbitals``."""
        return self.assignment.site_orbitals

    @property
    def antiunitary_operator(self):
        """``(ind, T)``, the index and matrix of pure time reversal; ``(None, None)`` where the group has none.

        Every corep label's phase is fixed against it, and its own phase is a
        convention.
        """
        if self._time_reversal is None:
            return None, None
        return int(self._time_reversal), self.matrices[int(self._time_reversal)]

    def _operation(self, ind):
        """The ``irrep`` operation record for ``ind``."""
        for op in self._operations:
            if int(op.ind) == int(ind):
                return op
        raise KeyError(f"operation #{ind} is not part of this representation")

    def k_action(self, ind):
        """One operation's action on the model's reduced ``k``, ``k → Rk``, as a function: the image not folded back into the zone."""
        return self.group.k_action_of(self._operation(ind))

    def transform_k(self, ind, k):
        """The image of ``k`` under one operation, not folded back into the zone."""
        k = np.atleast_2d(np.asarray(k, dtype=float))
        return self.k_action(ind)(k)

    def in_little_group(self, ind, k, *, atol=1e-8):
        """``(member, G)``: whether the operation carries ``k`` to itself up to a reciprocal lattice vector, and that vector, ``G = Rk − k``.

        ``G`` is the ``k' − k_out`` :meth:`bloch_operator` folds back at ``k``.
        """
        k = np.asarray(k, dtype=float).reshape(-1)
        kp = self.transform_k(ind, k)[0]
        G = kp - k
        return bool(np.abs(G - np.rint(G)).max() <= atol), np.rint(G).astype(int)

    def little_group(self, k, *, atol=1e-8):
        """The operations represented that carry ``k`` to itself up to a reciprocal lattice vector, by index."""
        return tuple(ind for ind in self.indices if self.in_little_group(ind, k, atol=atol)[0])

    def bloch_operator(self, ind, k, *, k_out=None, **kwargs):
        """One operation's Bloch operator at ``k``, by index: :func:`bloch_operator` on this representation."""
        return bloch_operator(self.model, self.group, self.assignment,
                              self._operation(ind), self.matrices[int(ind)], k,
                              k_out=k_out, **kwargs)

    def sewing_operator(self, ind, k, *, atol=1e-8, **kwargs):
        """The Bloch operator of an operation that leaves ``k`` invariant, folded back to ``k``.

        It commutes with ``H(k)``, and between the eigenvectors of a set of
        bands it is the sewing matrix of that set.

        Refusals
        --------
        ``little_group``
            the operation does not carry ``k`` to itself up to a reciprocal
            lattice vector.
        """
        ok, G = self.in_little_group(ind, k, atol=atol)
        if not ok:
            kp = self.transform_k(ind, np.asarray(k, dtype=float).reshape(-1))[0]
            return Refusal(
                "little_group",
                f"operation #{ind} maps k = {np.round(np.asarray(k, float), 6).tolist()} to "
                f"{np.round(kp, 6).tolist()}, which is not k modulo a reciprocal-lattice "
                "vector: it has no sewing operator there.",
                value=float(np.abs(kp - np.asarray(k, float).reshape(-1)
                                   - np.rint(kp - np.asarray(k, float).reshape(-1))).max()),
                threshold=atol, context={"operation_ind": int(ind), "k": np.asarray(k, float)})
        return self.bloch_operator(ind, k, k_out=np.asarray(k, dtype=float).reshape(-1),
                                   **kwargs)

    def covariance(self, k_pts=None, *, params=None):
        """``{ind: max_k ‖D H(k) D† − H(Rk)‖}``, each operation's covariance measured again, in the units of ``H``.

        On ``k_pts``, or by default at the ``k`` the representation was built
        at; ``params`` are the values of a model whose terms are given by
        parameter names.
        """
        k_pts = self._battery if k_pts is None else np.atleast_2d(np.asarray(k_pts, float))
        params = dict(params or {})
        bloch = _bloch.evaluator(self.model, params)
        H = np.asarray(bloch(k_pts))
        out = {}
        for ind in self.indices:
            op = self._operation(ind)
            Hr = np.asarray(bloch(self.transform_k(ind, k_pts)))
            D = self.matrices[int(ind)]
            src = H.conj() if op.time_reversal else H
            out[int(ind)] = float(np.abs(D @ src @ D.conj().T - Hr).max())
        return out

    def summary(self) -> str:
        """What the representation holds, a line for each part, with the attribute that holds it."""
        g = self.group
        where = {"substrate": ", on a substrate", "free-standing": ", free-standing"}.get(
            self.environment, "")
        norb, nspin = int(self.model.norb), int(self.model.nspin)
        on = (f"the {norb * nspin} states, each orbital's spin up then down along z"
              if nspin == 2 else f"the {norb} orbital{'s' if norb != 1 else ''}")
        lines = [f"{g.name} #{g.number_str}{where}   (.group)",
                 "  its operations, by index and Seitz symbol   (.operations)",
                 f"  each one's matrix on {on}   (.matrices)"]
        if self.antiunitary:
            lines.append("  an operation with time reversal is its matrix times complex "
                         "conjugation   (.antiunitary)")
        if self.spin is not None:
            kind = {"pairs": "listed pair by pair"}.get(self.spin.kind, self.spin.kind)
            lines.append(f"  the orbitals in {len(self.spin.pairs)} declared spinor pairs, "
                         f"{kind}   (.spin)")
        if self.characters is not None:
            lines.append("  the orbitals, by Wyckoff position   (.characters)")
            lines += ["    " + line for line in _content_by_position(self.characters, nspin)]
        lines.append("  the multiplets each site's orbitals form   (.multiplets)")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        worst = max(self.residuals.values()) if self.residuals else float("nan")
        n_multi = sum(1 for k in self.multiplets._chan_kind if k == "measured")
        wide = f" ({n_multi} multi-dimensional)" if n_multi else ""
        return (f"SpaceGroupRep({self.group.name} #{self.group.number_str}, "
                f"{int(self.multiplets._nchan)} multiplets{wide}, covariance<={worst:.1e}, "
                f"product={self.product_residual:.1e}"
                + (f", spinor basis ({self.spin.kind})" if self.spin is not None else "")
                + ")")


# --------------------------------------------------------------------------
# the constructor
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


def _no_multiplicity_representation(group, record, layout, kept, syms) -> Refusal:
    r"""``rep_not_multiplicative`` — a conserved pseudospin whose copies no representation fits.

    The operators ``H`` conserves on the orbit are a whole matrix algebra on
    ``record["dim"]`` copies, so every representation of the group on its
    states is ``B ⊗ B'`` with ``B`` the factor ``H`` fixes and ``B'`` a
    representation on the copies making up the factor system ``B`` leaves
    them (:class:`.._sectors.InducedOrbit`); no sum of the group's irreducible
    representations with that factor system, of that dimension and extended to
    time reversal with the square the tables read, is admitted.  So no
    representation of the group with its own factor system exists on these
    states: the projective class they carry is one the multiplicity space
    cannot make up.
    """
    k, orbitals = int(record["dim"]), tuple(record["orbitals"])
    named = f"{group.name} #{group.number_str}"
    return Refusal(
        "rep_not_multiplicative",
        f"the Hamiltonian conserves a {_fold(k)} pseudospin on orbitals {orbitals}: the "
        f"operators it conserves there are every {k} x {k} matrix on {k} identical "
        f"copies of the states it acts on, so every operation's matrix on them is the "
        f"one factor the hoppings fix times a matrix on the copies. A representation "
        f"of {named} with its own factor system needs the copies to carry a "
        f"{k}-dimensional representation making up the factor system the fixed "
        f"factor leaves them, time reversal's square included, and of the "
        f"{int(record['n_candidates'])} sums of the group's irreducible representations "
        f"with that factor system that have dimension {k}, none does: no "
        f"representation of {named} with its own factor system exists on these "
        f"states.",
        value=0.0, threshold=1.0,
        context={"multiplets": layout, "multiplet": orbitals, "multiplicity": k,
                 "n_candidates": int(record["n_candidates"]),
                 "indices": tuple(int(syms[p].ind) for p in kept)})


def _raise_unformed(group, unformed, n_cand, gate, *, spinful):
    r"""Raise for operations handed to the build none of whose matrices has its own form.

    The operations a representation is built for are the ones the measurement
    read in their own form, so one that has none is the package disagreeing
    with itself: :class:`..refusal.InternalError`, naming each such operation
    and the condition that failed.

    ``unformed`` is ``{ind: {"own_form": condition, "covariance": residual,
    ...}}``, read off the record each operation's own form was read on
    (:func:`..groups._own_form_read`): the condition that failed, ``"site
    permutation"`` -- no matrix carries each site's states onto its image's --
    or ``"spin lift"`` -- none carries the operation's own spin rotation --
    and the covariance of the matrix of that form nearest its space, where
    one was measured.  ``n_cand`` is how many operations were to be
    represented, and ``gate`` the covariance gate they were kept at.
    ``spinful`` says whether the states carry a spin, the model's own or
    declared: only then is a spin rotation asked of them, and named.
    """
    from .groups import _record_name

    syms = {int(s.ind): s for s in group.symmetries}

    def _named(ind):
        return (f"{_record_name(syms[ind], group.lattice)} "
                f"(#{ind}, its {unformed[ind]['own_form']})")

    named = ", ".join(_named(ind) for ind in sorted(unformed))
    measured = [float(r["covariance"]) for r in unformed.values()
                if r.get("covariance") is not None]
    closest = min(measured) if measured else None
    reach = ("" if closest is None else
             f" The matrix of its own form nearest any of their spaces carries H(k) onto "
             f"H(Rk) to {closest:.3g}, against the {float(gate):.3g} they were kept at.")
    carried = ("every site's states onto its image's with the operation's own spin "
               "rotation, the condition named failing" if spinful else
               "every site's states onto its image's")
    raise InternalError(
        f"{len(unformed)} of the {int(n_cand)} operations handed to the build, "
        f"{_counted_as(spinful)}, are not "
        f"realized on this orbital content: {named}. Matrices with each one's action on k "
        f"carry H(k) onto H(Rk), but none carries {carried}: they are other operations' "
        f"matrices, not these operations'.{reach} The operations a representation is "
        f"built for are the ones measured to hold in their own form, so this is the "
        f"package disagreeing with itself.")


def _build_rep(model, group, assignment, *, validated=None, multiplets=None,
               reference_orbital=0, characters=None, spin=None,
               generators=None, operations=None,
               phases=(1, -1), orientation_branch=None, include_vacuum_ops=False,
               strategy="readoff", max_candidates=2 ** 20,
               k_pts=None, n_battery=6, tol=None, internal_tol=1e-9,
               product_tol=1e-9, seed=101, _choices=None,
               _returned=True, _labelled=True, _search_record=None,
               params=None) -> SpaceGroupRep | Refusal:
    r"""The matrix each symmetry operation is, on this model's orbitals.

    Returns one matrix ``D(g)`` per operation the Hamiltonian obeys: the
    representation induced from the site content
    :attr:`SpaceGroupRep.characters` returns — each orbital's Wyckoff position,
    site symmetry and site irrep — in the tables' convention (spin along
    ``+z``, a right-handed frame).  It is the object every corep label is read
    through.  The construction is the tensor

    .. math:: D(g) = O(g) \otimes S(g),

    with ``O(g)`` a signed permutation of the **multiplets** of
    :class:`SiteMultiplets`, whose permutation comes from
    :attr:`..groups.OrbitalAssignment.permutations` (never typed), and
    ``S(g)`` the operation's own matrix on each multiplet.  Which orbitals a
    multiplet holds is **measured** — one per orbital where nothing mixes, the
    multiplet where the Hamiltonian binds one — and its matrix is the model's
    own, read off its solution spaces and carried round the Wyckoff orbit, in
    whatever basis the orbitals are written.  What is left free is a finite
    set of signs — one phase per generator and orbit of multiplets, plus the
    orientation branch, ``D`` versus ``D†``, i.e. ``g`` versus ``g⁻¹`` — and
    those are pinned by measurement, never by hand.  Time reversal carries its
    own internal factor on every multiplet times the scalar its solution space
    pins on each orbit of multiplets: one where the orbitals are real
    functions, and ``-1`` where an orbital is written as ``i`` times one
    relative to the first orbital's.  Every candidate goes
    through **three independent gates**, and uniqueness is asserted on their
    joint survivor set:

    1. **Hamiltonian covariance** — ``max_k ‖D H(k) D† − H(Rk)‖ ≤ tol``.  A
       candidate that fails this is not a symmetry of *this Hamiltonian*: a
       wrong relative sign inside a permuted Wyckoff orbit dies here, by orders
       of magnitude.
    2. **the group's own internal factor** — on every multiplet an operation
       fixes, the block must be ``χ(g)·S(g)`` with ``S(g)`` the operation's own
       ``sym.spinor_rotation``: ``max ‖B(g) − χ(g)S(g)‖ ≤ internal_tol``, the
       same defect the site characters are read off
       (:attr:`AtomicLimitTable.residual`).  A candidate that fails this
       commutes with ``H`` exactly as the group demands but is **not the matrix
       set of these operations**, and no orbital of it has a site character to
       declare or to label.
    3. **the group's own multiplication table**, spinor factors included —
       ``max ‖D(g₁)∘D(g₂) − spf(g₁,g₂)·D(g₁g₂)‖ ≤ product_tol``, measured after
       :func:`normalize_rep` has phase-canonicalized the generators.  The
       canonical norm fixes a generator's phase up to a root of unity; where
       nothing else narrows the roots — no antiunitary operation, and a
       multi-dimensional lead multiplet with no table lift behind it — the
       table itself chooses among them wherever positivity's own pick does not
       multiply.  A candidate that fails this is
       covariant but is *not a representation of the group* at all: the
       operation set, the generators or the internal blocks are wrong, or its
       matrices are a projective representation whose class the table does
       not carry — spin-½ states read in single-valued tables, where two
       commuting operations' matrices anticommute — and no choice of phases
       will fix it.

    **No two of the three suffice.**  A ``g⁻¹`` branch is a perfectly good
    representation of the abstract group, only of the elements relabelled by an
    automorphism, so it is exactly covariant *and* exactly multiplicative; what
    separates it is gate 2, since it carries ``S(g)†`` where the group says
    ``S(g)``.  Where every operation is an involution that measurement does not
    exist: ``S(g)† = −S(g)``, the internal-factor gate's per-block coefficient
    absorbs the sign, and every gate is blind.  The branch is then settled by
    the convention ``+1``, the intertwiner that carries ``k → gk``, with
    ``SpaceGroupRep._branch_tie`` set and ``_search["branch_note"]`` saying
    so.  ``rep_not_unique`` still fires when the two survivors are not one
    representation up to a one-dimensional character on each coupled part of
    the crystal.  Which gate killed a
    candidate is recorded per candidate in
    ``_search["candidates"][i]["killed_by"]``, and an empty survivor set is a
    refusal carrying all three residual spectra.

    All three gates are blind to multiplying the matrices of a coupled part of
    the crystal -- the orbitals the hoppings connect, directly or through
    others -- by a one-dimensional character of the group: the hoppings fix how
    the orbitals transform relative to one another, not what they are.  The
    phases a canonical norm leaves are narrowed by time reversal in a grey
    group, and where the group ships none by the tables' own lift on a
    one-dimensional lead multiplet (:attr:`RepAnchor.internal`): a
    normalization of the matrices before the labelling is chosen, not a
    fourth gate.  The **labelling** is then chosen once, the same way for
    every part: its candidates are the characters its declarations leave --
    with time reversal the real ones, without it every one, the complex ones
    included -- and one key orders them, read on the orbital factor.  First
    the part's site content at the lowest angular momentum, at the reference
    orbital's site, then over the crystal, then on the reference orbital's own
    row: a character, the same in any basis the orbitals are written in.  Then
    the letter the orbitals read as: the point-group representation their frame
    character -- the trace of each block carrying a site's orbitals onto its
    image -- averages to, at the lowest angular momentum.  Then the band labels
    themselves: the candidate whose labels come first in the tables' order at
    the high-symmetry points.  Last the group's own order of operations, on
    which no band label rests (:attr:`AtomicLimitRow.determined_by`, and
    :func:`..sewing.hsp_table`'s note).  A part the hoppings leave free of
    the rest is brought to its own canonical norm, and the key reads it from
    its first orbital unless it holds the reference one.  Where the
    Hamiltonian conserves a pseudospin on states read spinless -- ``k``
    identical copies of a part of the crystal that nothing joins, every
    ``k × k`` matrix on the copies commuting with it -- the representation the
    copies carry is left open as well, and it moves the site content: the same
    key chooses it, the copies read copy by copy first, among the
    representations the group's own table admits
    (:attr:`AtomicLimitTable.multiplicity_spaces`); so it chooses among every
    representation induced on uncoupled orbitals a symmetry exchanges, where
    those carry what no one-dimensional character gives.  Where the
    model's own blocks carry noise within a hundred times ``internal_tol`` the
    matrices are the exact representation nearest them, which a near-symmetric
    model's symmetrizing pass averages with.  The conventions the tables fix are
    stated in ``docs/background/conventions.md``.

    This is one of the pieces :func:`..groups.realized_group` already ran, and
    it is re-runnable with different arguments on a group already in hand —
    ``_build_rep(r.model, r.group, r.assignment, validated=r.validated,
    multiplets=...)``, or with a different ``characters=`` or ``generators=``.

    Parameters
    ----------
    model : pythtb.TBModel
    group : ..groups.GroupData
        A group comes from :func:`..groups.realized_group`:
        ``realized_group(model, ...).group``.
    assignment : ..groups.OrbitalAssignment
        From :func:`..groups._assign_orbitals` on **this** model and group.
    validated : ..groups.ValidatedGroup, optional
        Which operations the Hamiltonian obeys.  Computed here when omitted,
        by :func:`..groups._validate_group` at its defaults, whose
        ``detector_too_large`` is then handed back as it comes.  Only its
        surviving operations are candidates for representation, and its
        discrepancy list is carried into :attr:`SpaceGroupRep._failed`.  A
        record as the detector measured it, :func:`..groups._validate_group`'s
        own, is read here once, first, as :func:`..groups.realized_group` reads
        its own: each operation's solution space held to the matrices of its
        own form, each site's states carried onto its image's with the
        operation's own spin rotation on states that carry a spin.  A record
        already read is read as it stands.
    multiplets : SiteMultiplets, optional
        The multiplets each site's orbitals are bound into, **stated** instead
        of measured: the multiplets of every site, each given as the orbitals it
        holds, or a :class:`SiteMultiplets` record.  It is a partition of the
        model's orbitals, so which site index holds which multiplet is this
        assignment's business and a statement is re-keyed onto it — which is
        what carries one partition to another presentation of the same model.
        A statement that is not such a partition raises ``ValueError`` before
        anything is measured: an orbital named twice or not at all, a multiplet
        spread over two sites, a declared spinor pair split.  Omitted — the
        default — the multiplets are
        measured, by the rule :class:`SiteMultiplets` states: a site's own
        operations bind its orbitals, and copies of one site irrep turned from
        site to site of an orbit are one multiplet.  A one-dimensional multiplet
        takes ``S(g)`` to be the spin factor from the group
        (``sym.spinor_rotation``, raw, and ``iσ_y conj(·)`` for antiunitary
        operations), with the extraction *checked* against the detector's own
        intertwiner; a multi-dimensional one — a site whose orbitals form an
        irreducible multiplet of the site-symmetry group, a
        ``(d_z², d_x²₋y²)`` doublet at a ``32`` site, which no signed
        permutation of orbitals can represent — takes the model's own
        intertwiner's block, in whatever basis the orbitals are written.
    reference_orbital : int, optional
        The orbital the lowest-angular-momentum rule reads the crystal from
        first: among the labellings the declarations leave, its site's
        content is put at the lowest angular momentum it allows, and every
        orbital the hoppings couple to it follows; of two rows one site content
        leaves open, the reference orbital's is the one put at the lower.
        Naming a different orbital starts the rule there instead.  It
        constrains nothing a declaration does not.
    characters : sequence or mapping, optional
        **A statement of what the orbitals are** — the physics input the
        Hamiltonian cannot supply, since the hoppings do not measure what an
        orbital is: they measure which operations are symmetries once the
        orbitals are stated.  A list with one entry per orbital, each a word or
        ``None``, or a mapping from an orbital, or from several orbitals of
        one site together, to a word; a mapping from operations to characters
        stands for a word in either (``docs/guide/orbitals.md``).  A declaration in
        none of these forms raises ``ValueError`` before anything is measured,
        and a word that names nothing at its site is refused
        ``unknown_label``.  A declaration states
        each site's content and is read on the measured multiplets, however it
        partitions them: a content the site does not hold breaks the operations
        it is not carried by — declaring ``s`` and ``d`` where the hoppings
        realize a ``p`` doublet states a crystal whose fourfold axis is not a
        symmetry, and declaring them where nothing couples the pair's two
        orbitals says which of the two the pair is.  ``None`` (the default)
        assigns every orbital by the lowest-angular-momentum rule.  A declaration the
        hoppings are not covariant under is not a wrong declaration: it is a
        smaller group, refused
        ``declaration_breaks_group`` here — this call is handed a **fixed**
        group and cannot lower it — and descended to by
        :func:`..groups.realized_group`.

        **What a declaration speaks for.**  The hoppings fix how the orbitals
        they connect transform relative to one another, so a declaration on
        any orbital is a statement about every orbital coupled to it, directly
        or through others: they are read relative to it, which relabels them
        all at once, wherever the declaration sits.  Declaring one sublattice
        of an orbit is the same statement as declaring the other.  Where no
        labelling carries every declaration of such a coupled part at once,
        the operations they disagree on are broken and the group is smaller.
    spin : ..spin.SpinBasis, str or dict, optional
        **The spinor basis of a spinless model**, declared: which orbital is the
        ``↑`` partner of which ``↓``.  A :class:`..spin.SpinBasis` (from
        :func:`..spin.spin_basis`), the shorthand ``"interleaved"`` /
        ``"blocked"``, or a mapping of :func:`..spin.spin_basis` keyword
        arguments.  This is the one piece of physics a ``wannier90`` import
        cannot supply — the importer returns ``nspin = 1`` whatever the
        calculation did, and ``H(k)`` is blind to the names of its ``2N``
        indices.  A one-dimensional multiplet is then a declared pair carrying
        the tables' own SU(2) lift, and everything downstream runs as for a
        native spinful model; see
        ``docs/guide/spin.md#a-wannier90-spinor-import``.  Only for a spinless
        model asked in double-valued tables; a native spinful model carries its
        own spin axis and raises.
    generators : sequence of int, optional
        ``.ind`` of the operations to enumerate signs on.  A minimal generating
        set of the represented unitary operations is found from the product
        table when omitted.
    operations : sequence of int, optional
        Restrict the represented set (default: everything ``validated`` kept).
    phases : sequence of complex, optional
        The factors the enumeration tries on each orbit of multiplets, per
        generator, relative to the reference orbit.  The default ``(1, -1)``
        covers real relative signs; a richer set such as ``(1, 1j, -1, -1j)``
        covers orbits related by roots of unity, at
        ``len(phases)**(n_gen·(n_class−1))`` candidates.  The read-off does not
        consult it: it takes each factor as the solution spaces measure it, a
        root of unity of order dividing twelve where the measured factor is one
        to within the noise and the measured phase otherwise, so ``phases``
        matters only where the enumeration runs.
    orientation_branch : None or {+1, -1}, optional
        Which orientation branches enter the candidate set: ``None`` (default)
        both, ``+1`` only ``g``, ``-1`` only ``g⁻¹``.  The branch is degenerate
        whenever every generator is an involution with a Hermitian
        representative, and the deduplication step then merges it.  This is a
        **restriction of the search space**, not a tolerance: a ``+1`` build
        never learns whether ``g⁻¹`` would also have passed, so it never reports
        ``SpaceGroupRep._branch_tie``.  It is not needed to settle the
        branch; keep it for speed on a large group, to state a convention
        deliberately, or (with ``-1``) to measure what the other branch does.
    include_vacuum_ops : bool, optional
        Represent operations that act non-trivially on a slab's vacuum
        directions.  Off by default: a 2D pythtb model's orbitals carry no
        out-of-plane structure for such an operation to move, so the *geometry*
        does not fix their orbital action, and — because their in-plane k-action
        is degenerate with an in-plane partner's — the *detector* cannot refute
        them either.  Whether a flat layer's own orbital content nevertheless
        determines the action is a question only a built representation answers,
        by covariance.  Left out, they are not in the representation's
        :attr:`SpaceGroupRep.indices`.
    strategy : {"readoff", "enumerate"}, optional
        How the candidates are **found**.  ``"readoff"`` (the default) measures
        the free factors off the detector's own solution space — when the tensor
        structure is realizable ``D(g)`` is an element of it, so the space is
        projected onto tensor-structure form, polar-extracted, and the phases of
        the multiplet orbits read off the block ratios against ``S(g)`` — and
        builds one candidate per orientation branch.  ``"enumerate"`` builds
        every combination of the factors ``phases`` offers instead, which is the
        instrument for a full kill record and is what the read-off falls back to
        when what it read is not clean or is not a representation; a refusal
        after such a fallback says what the read-off read.  **Neither is a
        tolerance**: both answer to the same three gates, so where the measured
        factors are among those ``phases`` offers the two return the same
        representation and differ only in what they cost.
        ``_search["strategy"]`` says which ran.
    max_candidates : int, optional
        Refuse (``gate="rep_search_too_large"``) rather than run an enumeration
        of more than this many candidates.  Default ``2**20``.  Reached only
        when an enumeration is needed, so under ``strategy="readoff"`` the
        refusal also carries the reason the read-off deferred.
    k_pts : array, optional
        Covariance battery points.  Random generic k by default: high-symmetry
        points make far too many wrong candidates look right.
    n_battery : int, optional
        How many random points to use when ``k_pts`` is omitted.
    tol : float, optional
        The covariance gate, absolute and in the units of ``H``.  Omitted, it
        is :func:`covariance_gate` of this build: the larger of the exact floor
        :data:`EXACT_TOL`, ``5e-14``, and ``|G|·N·ε·max|H|``, the rounding a
        representation of the ``|G|`` operations represented, on ``N`` states,
        carries, with ``max|H|`` the largest entry of ``H(k)`` on the battery.
        Rounding never refuses a valid model; above the gate the verdict is the
        verdict, and nothing is weakened to make a candidate pass.  A number
        given here is the gate.
    internal_tol : float, optional
        The internal-factor gate: ``max ‖B(g) − χ·S(g)‖`` over **every** block
        of every represented unitary operation — the block carrying a
        multiplet to its image as much as the block of a multiplet the
        operation fixes, with the per-block coefficient free.  Dimensionless
        (the blocks are unitary).  Default ``1e-9``.  Vacuous for a spinless
        model with no declared spinor basis (``S`` is ``1×1``).  On a
        multi-dimensional multiplet ``S(g)`` is not a table's but is
        **measured**, off the whole detector solution space, and the gate is
        the statement that the built blocks are multiples of what was measured.
    product_tol : float, optional
        The product-table gate: ``max ‖D(g₁)∘D(g₂) − spf·D(g₁g₂)‖`` over the
        represented pairs.  The matrices are unitary, so unlike ``tol`` this
        number is **dimensionless** and carries no scale from ``H`` — which is
        why it is a separate knob rather than ``tol`` re-used.  Default
        ``1e-9``; tighten it only after reading
        ``_search["product_residuals"]``.
    seed : int, optional
        Seed of the battery's RNG, so the search is reproducible.
    params : dict, optional
        Model parameter values, as ``model.hamiltonian`` takes them.

    Returns
    -------
    SpaceGroupRep or Refusal
        The single surviving candidate: one matrix per represented operation,
        the record of what pinned every discrete choice, and the resolved site
        character of every orbital.  It records no environment
        (:attr:`SpaceGroupRep.environment` is ``None``); the representation
        :func:`..groups.realized_group` returns keeps the environment it was
        read in.

    Refusals
    --------
    ``"rep_not_covariant"``
        no candidate closes the covariance gate — read it as *this operation
        set is not realized on this orbital content*, e.g. a slab operation
        the model cannot represent on the multiplets the Hamiltonian binds.
    ``"rep_internal_factor"``
        covariant candidates exist but none carries the group's own internal
        factor on the multiplets its operations fix — they are somebody else's
        matrices, not these operations'.
    ``"rep_not_multiplicative"``
        candidates pass both of those but none reproduces the multiplication
        table, and no choice of phases will fix it: the operation set, the
        generators or the internal blocks are wrong, or the matrices form a
        projective representation of a class the table does not carry.  The
        second is measured and named — two commuting operations whose matrices
        commute only up to a phase no rephasing removes
        (``context["commuting_operations"]``, ``context["commutation_phase"]``)
        — and where that phase is the one the operations' spin-½ lifts carry
        (``context["spin_half"]``) the states carry half-integer spin: no
        single-valued representation exists on them, and a spinor basis in the
        orbital index is read in the double group once ``spin=`` declares it.
        This build is handed a fixed group; :func:`..groups.realized_group`
        measures the same class before it builds and descends to the largest
        subgroup that carries an ordinary representation.  On a conserved
        pseudospin the class is not the states' alone -- the copies' own
        representation can make it up -- and the refusal says where no
        representation of the copies' space does (``context["multiplicity"]``).
    ``"rep_not_unique"``
        more than one distinct representation survives **all three** gates,
        which means the battery is too small or the model is more symmetric
        than the group says.  All four of these carry the number of candidates
        tried (``context["n_candidates"]``).  The one survivor pair it does
        **not** fire on is the two orientation branches returning the same site
        labels under the same labelling: that is a convention
        rather than a second representation, and it is pinned to ``+1`` with
        ``SpaceGroupRep._branch_tie`` set.
    ``"rep_search_too_large"``
        not about the model at all: an enumeration was needed (asked for, or
        fallen back to) and would have run more candidates than
        ``max_candidates``.  Under ``strategy="readoff"`` the message also says
        why the read-off deferred, which is the thing to act on — a read that is
        not clean says the tensor structure ``O(g) ⊗ S(g)`` does not describe
        this operation set on this orbital content.  The same ceiling bounds
        the enumeration that grows with the *model*: where the Hamiltonian
        conserves sets of states apart that no operation exchanges, or a
        pseudospin, the hoppings do not fix which representation those states
        carry, and the finite set to try is every choice of one for each set
        -- or, on the pseudospin, every sum of the group's irreducible
        representations the copies' space could carry.
    ``"detector_too_large"``
        only where ``validated`` is omitted and measured here: the detector's
        own memory limit on a spectrum with exactly degenerate levels
        (:func:`_find_intertwiners`), handed back as it comes.
    ``"rep_double_valued_spinless"``
        a structural gate on the valuedness the model and the tables disagree
        about: the model is spinless, the tables asked for are double-valued,
        and nothing said where the spin is.  For a ``wannier90`` import declare
        the spinor basis (``spin="interleaved"``, ``"blocked"``, or explicit
        ``pairs=``) and the build runs.  What stays unsupported is an
        **unpaired** spinor orbital — a single Wannier function carrying
        half-integer spin with no partner, the locally time-reversal-broken
        magnetic case — which needs double-group *site* declarations that
        single-valued site tables cannot express.
    ``"spin_pair_split"``
        a declared spin pair whose two members sit on different sites, where
        the two spin states of one Wannier function share a centre: a
        mislabelled pairing, or orbitals each carrying one spin state with no
        partner at their own centre, a magnetic configuration outside this
        package's present scope.  :func:`..groups.realized_group` refuses it
        from the centres alone before anything is identified, a declared
        ``precision`` reading two centres closer than it as one.  A pairing
        that is well formed and wrong is a different tensor structure, and it
        fails the covariance or the internal-factor gate like any other, where
        :func:`..groups.realized_group` descends instead.
    ``"unknown_label"``, ``"ambiguous_multiplet"``
        what a well-formed ``characters=`` can say that its site cannot read:
        a label that is not an irrep of *that orbital's measured site group*
        (the message names the site group, lists its valid labels and prints
        the generator frame); a flat list that cannot say which orbitals form
        a multiplet; a joint declaration whose irrep is not the dimension of
        the multiplet it names (``{(0, 1): "A"}``, where ``{0: "A", 1: "A"}``
        is the spelling for two ``A`` orbitals).
    ``"declaration_breaks_group"``
        the declared content is not covariant under every operation of the
        group handed in.  Not a wrong declaration and not a tolerance: the
        *positions* allow both of lin3g's ``1a`` orbitals to be ``s`` and the
        Hamiltonian does not, so what was declared is a crystal with less
        symmetry.  This call is handed a fixed group and cannot lower it, so it
        names the subgroup the content **is** covariant on
        (``context["subgroup"]``), the operations it breaks with their rows
        (``context["broken"]``), the maximal readings where several tie, in the
        order the tie rule ranks them with that subgroup first
        (``context["tied"]``), what the declaration that fails measured, in
        irrep names, and the complete ``checks`` table; ``context["group"]`` is
        the group handed in.
        :func:`..groups.realized_group` descends to that subgroup instead of
        refusing.  In a group with **no antiunitary operation** more
        declarations stand: one the hoppings carry up to a one-dimensional
        character of the group breaks nothing, and the representation returned
        is induced from the content as declared.

    A unitary-only build that cannot be anchored to the tables does **not**
    come back as a refusal: see *Raises*.

    Raises
    ------
    ..refusal.InternalError
        A unitary-only build with a one-dimensional reference multiplet (no
        antiunitary operation in the group) whose generator blocks are not real
        multiples of the operations' own ``sym.spinor_rotation``.  The tables fix that gauge
        outright, so this is a bug in the application of the convention and not
        a measurement about the model, and it is deliberately **not** a
        ``Refusal``: no gauge is left to supply and there is nothing to report.
        A model and a table that disagree about *valuedness* falsify the
        anchor's premise rather than its application, and are refused above
        or raise ``ValueError`` below instead.  Also, the package disagreeing
        with itself in what it hands the build: an operation to be represented
        none of whose matrices has its own form, each site's states carried
        onto its image's with its own spin rotation where the states carry a
        spin; operations measured to hold that do not close into a group; no
        generating set of three operations or fewer for the represented
        unitary ones; an internal block the measurement does not reach.
    ValueError
        Grammar in the arguments: a ``characters=`` that is not a list with
        one entry per orbital or a mapping from orbitals to words, that names
        an orbital twice or out of range, that writes orbitals of two sites as
        one key, or that gives the two orbitals of a declared spin pair
        different words; a ``multiplets=`` that is not a
        partition of the model's orbitals into multiplets of one site each; a
        ``spin=`` declaration written for another model; a ``strategy=`` or an
        ``orientation_branch=`` that names nothing; an ``operations=`` that is
        not closed under the group's product, or holds an antiunitary
        operation that is not time reversal times a represented one;
        ``generators=`` that do not generate the represented unitary
        operations; a group of single-valued tables for states that carry
        spin-1/2, natively or by declaration.  The declarations are read
        before anything is measured.

    Notes
    -----
    **Validity conditions.**

    * The search is only as sharp as the battery.  Generic k-points are used
      deliberately; a battery of high-symmetry points would let wrong signs
      through.
    * "Unique" is asserted on *distinct representations*, not on parameter
      choices: candidates that build the identical matrices (an involutive
      generator under the orientation branch) are merged first.
    * The representation's phase is anchored at orbital 0's multiplet, and on
      a coupled part without it at the part's first, whatever
      ``reference_orbital`` names (see :class:`RepAnchor`); the labelling is
      put on top of it, and ``reference_orbital`` picks only the lead the
      labelling's keys read first.
    * The product-table gate runs on the **normalized** matrices — the ones the
      returned object carries — so ``_search["product"]`` and
      ``product_residual`` are the same measurement.  The internal-factor gate
      needs no normalization pass: its defect is invariant under ``D → λD``.
    * ``factor_residual`` — the Bloch operators' factor system, translation
      differences included — stays a **post-hoc** cross-check and is not a
      gate: for matrices that already reproduce the product table it follows
      from the candidate-independent position identity
      ``A τ_a + t = τ_{g(a)} + s``.
    * **The environment is not set here.**  Nothing in this construction knows
      whether a 2D model's layer is free-standing or substrate-bound — that is
      :func:`..groups.realized_group`'s resolved verdict, stamped on the record
      it returns — so :attr:`SpaceGroupRep.environment` and
      :attr:`SpaceGroupRep.free_standing_group` are ``None`` on a
      representation built here.  :func:`..ebr.symmetry_indicator` reads a
      layer's class in the group handed in either way.
    * ``SpaceGroupRep.residuals`` (covariance), ``._norm_residuals``
      (``D^n = ζ``), ``._anchor_residuals`` (``T conj(D) T† = D``),
      ``.product_residual`` and ``.factor_residual`` are all re-measured on the
      returned matrices.

    Examples
    --------
    Redo the build :func:`..groups.realized_group` already made, with each
    site's orbitals stated as one multiplet instead of measured::

        r = tqc.realized_group(model, characters={0: "s"})
        rep = _build_rep(r.model, r.group, r.assignment,
                         validated=r.validated,
                         multiplets=[[orbs] for orbs in r.assignment.site_orbitals])
    """
    from . import groups as _groups

    if strategy not in ("readoff", "enumerate"):
        raise ValueError("strategy must be 'readoff' (read the choices off the measured "
                         "solution spaces) or 'enumerate' (the exhaustive search), got "
                         f"{strategy!r}")
    params = dict(params or {})
    nspin, norb, nstate = int(model.nspin), int(model.norb), int(model.nstate)

    # ---- 0. the spin the tables assume and the model does not carry ---------
    # Everything below reads `group.spinor` as "these orbitals carry the SU(2)
    # lift": the internal factor is the operations' own `sym.spinor_rotation`,
    # the canonical norms are the double cover's, and the table anchor of
    # `RepAnchor.internal` rests on that lift being pseudo-real.  A spinless
    # model's orbitals carry no such factor *unless a declaration says where
    # it is*, so the premise is either false before a single matrix is built --
    # and the anchor's own diagnosis would then be wrong, `_table_anchored`
    # raising `InternalError` to say *this package misapplied the tables*
    # where the truth is that the physics asked for is unsupported -- or it is
    # restored verbatim by a `spin=` declaration, whose multiplet is a declared
    # (up, dn) pair carrying exactly that lift.  Both settled here,
    # structurally, ahead of all of it.
    spin = _resolve_spin(spin, model)
    # ... and what the declarations say, read as grammar before anything is
    # measured: a malformed `characters=` or `multiplets=` raises here
    _declaration_grammar(norb, characters=characters, multiplets=multiplets, spin=spin,
                         site_of=assignment.site_of_orbital)

    if spin is not None and not bool(group.spinor):
        raise ValueError(
            f"the orbitals were declared to carry spin (`spin=`, a "
            f"{spin.kind!r} spinor basis on {spin.norb} orbitals) and #{group.number_str} "
            f"is a group of single-valued tables: states that carry spin-1/2 have a "
            f"double-valued content, which those tables do not describe.")

    if spin is None and nspin == 1 and bool(group.spinor):
        return Refusal(
            "rep_double_valued_spinless",
            f"the model is spinless (nspin=1, spinful=False) and #{group.number_str} "
            f"was asked for its double-valued tables, with nothing said about where the "
            f"spin is. A double-valued representation of a spin-free basis says the "
            f"orbitals themselves carry half-integer local spin -- a spinor Wannier "
            f"basis, the spin inside each Wannier function -- so the internal factor is "
            f"not the SU(2) lift the tables are written in, and the tables cannot fix "
            f"its gauge. "
            f"If the orbitals are the two spin states of N Wannier functions -- the "
            f"wannier90 case, where the importer returns nspin=1 whatever the "
            f"calculation did -- then declare the spinor basis with `spin=` "
            f"(`order=\"interleaved\"`, `order=\"blocked\"`, or explicit `pairs=`) and "
            f"this build runs: a multiplet becomes a declared (up, dn) pair and the "
            f"internal factor is the tables' own SU(2) lift on it. "
            f"What remains unsupported is an unpaired spinor orbital -- a single "
            f"Wannier function carrying half-integer spin with no partner, which is the "
            f"locally time-reversal-broken magnetic case -- because that needs "
            f"double-group site declarations and the site tables of `characters=` are "
            f"single-valued. "
            f"Otherwise ask this model in its single-valued group "
            f"(`realized_group(model, spinor=False)`), or give the orbitals a spin axis "
            f"(`spinful=True`) if the half-integer content is the electron's own.",
            context={"nspin": nspin, "spinful": bool(getattr(model, "spinful", nspin == 2)),
                     "number_str": str(group.number_str), "spinor": bool(group.spinor),
                     "include_TR": bool(getattr(group, "include_TR", False))})

    # The mirror of it, and this one is not a deferred feature but an invalid
    # ask: ``nspin == 2`` always means double-valued tables, at every level.  A
    # spinful model's states carry the electron's own spin-1/2, so the tensor
    # structure below assembles SU(2) blocks (`_internal_from_table`, taken whenever
    # `nspin == 2`) whatever the group says.  Against a single-valued table
    # those blocks miss the canonical norm, so the anchor would blame this
    # package for it (`InternalError`) or -- in the grey presentation --
    # answer with a true `antiunitary_anchor` measurement of a false setup.
    # Neither is a question worth asking: the pair is a malformed argument, and
    # it raises before anything is diagnosed.
    if nspin == 2 and not bool(group.spinor):
        raise ValueError(
            f"the model is spinful (nspin=2) and #{group.number_str} is a group of "
            f"single-valued tables: the electron's own spin-1/2 gives its states a "
            f"double-valued content, which those tables do not describe.")

    # ---- 0b. what a declared spinor basis has to be, to be one --------------
    # A declared (up, dn) pair is one Wannier function, which no multiplet
    # splits, so its two orbitals share a centre: the check `realized_group`
    # asks before anything is identified, here on the sites handed in.
    if spin is not None:
        split = _spin_pair_split(spin, assignment.site_orbitals,
                                 assignment.site_positions, group.lattice)
        if split is not None:
            return split

    syms = group.symmetries
    pos_of_ind = {int(s.ind): i for i, s in enumerate(syms)}
    prod, tdiff, spf = group.get_product_table(get_diff=True)
    ident = _identity_position(group)

    if validated is None:
        validated = _groups._validate_group(model, group, params=params)
        if isinstance(validated, Refusal):
            return validated                # the detector's memory limit, as it came

    # ---- 0c. EACH OPERATION'S OWN FORM, read once ---------------------------
    # The detector's record holds, for each operation, every matrix with its
    # action on k: another operation's among them, or one composed with a spin
    # rotation the Hamiltonian conserves.  Each operation is held to its own
    # form once, as the verdict stage holds it, before anything reads a
    # solution space; a record already read -- realized_group's -- is read as
    # it stands, and the build assembles what was read.
    validated = _groups._own_form_once(
        model, group, assignment, spin, validated,
        tol=(multiplets._tol if isinstance(multiplets, SiteMultiplets) else None),
        params=params)

    # ---- 1. which operations are candidates for representation -------------
    if operations is None:
        cand = [pos_of_ind[int(i)] for i in validated.indices]
    else:
        cand = [pos_of_ind[int(i)] for i in operations]
    if ident not in cand:
        raise ValueError("the identity operation is not among the operations to "
                         "represent: `operations=` leaves it out, and every "
                         "Hamiltonian obeys it")

    # an operation acting on a vacuum direction of a slab cell is left out
    # unless asked for: the model carries no out-of-plane orbital structure for
    # it to move, and the detector cannot refute it, its in-plane k-action
    # duplicating an in-plane partner's
    kept = [p for p in cand
            if include_vacuum_ops or not _moves_padded(syms[p], group.vacuum_dirs)]
    # an operation to be represented none of whose matrices has its own form
    # is represented by none: its matrices are another operation's, and the
    # operations handed to the build are the ones measured to hold in their
    # own form, so one without it is the package disagreeing with itself
    unformed = {int(syms[p].ind): got for p in kept
                for got in [(validated._own_form or {}).get(int(syms[p].ind))]
                if got is not None}
    if unformed:
        _raise_unformed(group, unformed, len(kept), validated.tol,
                        spinful=nspin == 2 or spin is not None)

    u_pos = [p for p in kept if not syms[p].time_reversal]
    au_pos = [p for p in kept if syms[p].time_reversal]
    if set(_closure(u_pos, prod, ident)) != set(u_pos):
        said = (f"the unitary operations {tuple(int(syms[p].ind) for p in u_pos)} to be "
                f"represented are not closed under the group's own product table, so no "
                f"representation of them exists")
        if operations is not None:
            raise ValueError(f"{said}; `operations=` names a subgroup.")
        raise InternalError(f"{said}, and they are the operations measured to hold.")

    # ---- 2. generators and the word tree ----------------------------------
    if generators is None:
        gens = _minimal_generators(u_pos, prod, ident)
        if gens is None:
            raise InternalError(
                f"no generating set of size <= 3 was found for the represented unitary "
                f"operations {tuple(int(syms[p].ind) for p in u_pos)}.")
    else:
        gens = tuple(pos_of_ind[int(i)] for i in generators)
        if set(_closure(gens, prod, ident)) != set(u_pos):
            raise ValueError(
                f"`generators=` {tuple(int(syms[p].ind) for p in gens)} do not generate the "
                f"represented unitary operations {tuple(int(syms[p].ind) for p in u_pos)}.")
    words, parent = _bfs_words(gens, prod, ident, u_pos)
    if set(words) != set(u_pos):  # pragma: no cover - guarded by the closure test
        raise InternalError(
            f"the generator tree does not reach the operations "
            f"{tuple(int(syms[p].ind) for p in set(u_pos) - set(words))}.")

    T_pos, au_of = None, {}
    if au_pos:
        pure = [p for p in au_pos
                if np.all(np.asarray(syms[p].rotation, dtype=int) == np.eye(3, dtype=int))
                and np.allclose(np.asarray(syms[p].translation) % 1.0, 0.0, atol=1e-8)]
        T_pos = pure[0] if pure else au_pos[0]
        for p in au_pos:
            match = [q for q in u_pos if int(prod[T_pos, q]) == p]
            if not match:
                said = (f"antiunitary operation #{syms[p].ind} is not the product of "
                        f"#{syms[T_pos].ind} with a represented unitary operation")
                if operations is not None:
                    raise ValueError(f"{said}; `operations=` names a subgroup.")
                raise InternalError(f"{said}, and they are the operations measured to hold.")
            au_of[p] = match[0]

    # ---- 3. multiplets, their permutations and their orbits ----------------
    # The multiplets are measured, or stated by `multiplets=`; `characters=`
    # never changes them: it states what each site holds, and step 3b reads it
    # on these multiplets.
    layout = _site_multiplets(model, group, assignment, spin, validated,
                             declared=multiplets)
    su2, site_orbs = layout._spinful, layout._site_orbitals
    nchan = int(layout._nchan)
    chan_states, chan_of = layout._chan_states, layout._chan_of

    # ---- 3b. the declared characters, parsed before anything expensive runs --
    kept_u_ind = {int(syms[p].ind) for p in u_pos}
    kept_au_ind = {int(syms[p].ind) for p in au_pos}
    site_tables, decls, char_note = {}, [], ""
    try:
        # the site tables read a declaration and name the labelling; a build
        # whose matrices are wanted only up to that labelling, with nothing
        # declared, needs neither (`_labelled=False`)
        if _labelled or characters is not None:
            site_tables = _site_tables(group, assignment, kept_u_ind, kept_au_ind,
                                       range(len(site_orbs)))
    except Exception as exc:                     # noqa: BLE001 - see below
        if characters is not None:
            gate = getattr(exc, "gate", "site_group")
            return Refusal(gate, str(exc), context={"multiplets": layout})
        # nothing was declared: the echo record is a courtesy and must never
        # break a build that would otherwise succeed.
        site_tables, char_note = {}, f"site character tables unavailable: {exc}"
    if characters is not None:
        try:
            decls = _normalize_declaration(characters, model, assignment, group,
                                           site_tables, layout, spin=spin)
        except _sitechars.DeclarationError as exc:
            return Refusal(exc.gate, exc.message,
                           context=dict(exc.context, multiplets=layout))
        # A DECLARATION STATES WHAT EACH SITE HOLDS, read on the multiplets
        # above, which it never changes
        decls = _declared_blocks(decls, layout, assignment, site_tables, spin=spin)
        for d in decls:
            d["channels"] = tuple(sorted({int(chan_of[int(o)])
                                          for o in d.get("block", d["orbitals"])}))

    cperm = _channel_perms(model, assignment, syms, kept, layout, spin)
    if isinstance(cperm, Refusal):
        return cperm
    classes, ncls = _channel_classes(cperm, u_pos, nchan)
    # THE ANCHOR the representation is written against before any labelling is
    # chosen: orbital 0's multiplet, whichever orbital the lowest-angular-momentum
    # rule reads the crystal from first (``reference_orbital``, the lead of its
    # own coupled part when the labelling is chosen)
    ref_orb = int(reference_orbital)
    ref_chan = int(chan_of[ref_orb])
    anchor = int(chan_of[0])
    anchor_cls = int(classes[anchor])
    free_cls = [c for c in range(ncls) if c != anchor_cls]

    # ---- 4. THE MEASUREMENT, made once ----------------------------------------
    # What each operation's solution space says about the orbitals, read once:
    # the internal matrix S(g) of every multiplet -- the tables' lift on a
    # one-dimensional multiplet, with the sign the model's transport gives it
    # from site to site, and the block of the model's own intertwiner on a wider
    # one, carried round the orbit -- what the declared content does to each
    # operation, what each multiplet orbit carries relative to the anchor where
    # the hoppings pin it, and the scalar time reversal carries on each orbit.
    # Where `..groups.realized_group` made this reading on this same group it is
    # handed in; otherwise it is made here, once, and nothing below reads a
    # solution space for it again.
    # HOW FAR OUTSIDE A SOLUTION SPACE still counts as inside it: the model's
    # own defect, which for a numerical Hamiltonian is what `internal_tol`
    # already states, and the declared tolerance the multiplets were measured
    # at where that is larger.  An exact model leaves it at the read-off's own
    # 1e-8.  The same declared tolerance reaches every threshold below that
    # judges a solution space: a vanishing block or ratio, a sign read off one.
    eps = layout._tol
    zero, snap = noise_floor(ZERO, eps), noise_floor(SNAP, eps)
    space_tol = max(noise_floor(IN_SPACE, eps), float(internal_tol))
    u_inds = [int(syms[p].ind) for p in u_pos]
    verdict = None
    if _choices is not None and (set(_choices.get("indices", ())) == set(u_inds)
                                 and _choices.get("declared") == _declared_key(decls)
                                 and float(_choices.get("space_tol", -1.0)) == space_tol
                                 and np.array_equal(np.asarray(_choices.get("classes")),
                                                    np.asarray(classes))
                                 and int(_choices.get("anchor", -1)) == anchor
                                 and "blocks" in _choices):
        verdict = _choices
    if verdict is None:
        got = _multiplet_blocks(model, group, assignment, spin, validated, layout,
                                kept, ref_chan=anchor, max_candidates=int(max_candidates))
        if isinstance(got, Refusal):
            return got
        verdict = _verdicts_and_choices(model, group, assignment, spin, validated,
                                        layout, decls, list(u_pos), classes=classes,
                                        ref_orb=ref_orb, space_tol=space_tol,
                                        tol=noise_floor(CHARACTER, eps),
                                        blocks=got, max_candidates=int(max_candidates),
                                        reversal=T_pos)
        if isinstance(verdict, Refusal):
            return verdict
    measured_of = verdict["blocks"]
    for record in measured_of.multiplicity:
        if not record["n_admitted"]:
            return _no_multiplicity_representation(group, record, layout, kept, syms)

    def _internal(p):
        """``[S per channel]`` for one operation, or ``None`` where a block is missing."""
        lift = _lift_of(syms[p], su2)
        blocks = measured_of.get(int(syms[p].ind), {})
        out = []
        for c in range(nchan):
            S = blocks.get(int(c))
            if S is None and layout._chan_kind[c] != "measured":
                out.append(lift)
                continue
            if S is None or S.shape != (len(chan_states[c]),) * 2:
                return None
            out.append(S)
        return out

    S_gen = {}
    for p in list(gens) + ([T_pos] if T_pos is not None else []):
        S = _internal(p)
        if S is None or any(float(np.abs(M).max()) < zero for M in S):
            asked = sorted({len(chan_states[c]) for c in range(nchan)})
            raise InternalError(
                f"no internal block of size {asked} could be obtained for operation "
                f"#{syms[p].ind}: a measured multiplet's block comes from the model's own "
                f"solution space, and the record of the operations measured holds none "
                f"that reaches that multiplet's image.")
        S_gen[p] = S

    # WHAT TIME REVERSAL CARRIES ON EACH ORBIT OF MULTIPLETS, as the measurement
    # read it: its internal factor on every multiplet, times one scalar per
    # orbit -- one on orbitals that are real functions
    reversal = np.ones(ncls, dtype=complex)
    if T_pos is not None and verdict.get("reversal") is not None:
        reversal = np.asarray(verdict["reversal"], dtype=complex)

    # THE CANDIDATE CHARACTERS: the one-dimensional characters of the group the
    # representation is blind to, single-valued on the orbital factor.  With
    # time reversal they are the real ones, each primed operation carrying its
    # unitary part's value; without it, every one-dimensional character of the
    # unitary group, the complex ones included.
    ind_of_pos = {p: int(syms[p].ind) for p in kept}
    admissible = [{ind_of_pos[p]: v for p, v in chi.items() if p in ind_of_pos}
                  for chi in _real_characters(kept, gens, words, parent,
                                              (au_of if T_pos is not None else {}),
                                              prod, ident)]
    if not admissible:            # pragma: no cover - the trivial character always is
        admissible = [{ind_of_pos[p]: 1.0 + 0.0j for p in kept}]
    pool = (admissible if au_pos else
            [{int(k): complex(v) for k, v in chi.items()}
             for chi in _characters_of_subgroup(group, u_inds, roots=True)] or admissible)

    unpinned = verdict["free_classes"]
    # WHAT THE DECLARATIONS ASK, as the verdict stage read it: one character per
    # coupled part of the crystal, each part's declarations constraining its
    # own.  A declared content not covariant on all of this group breaks
    # operations of it.
    if decls:
        sub = _covariant_subgroup(group, u_inds, verdict["requirements"],
                                  unitary_only=not au_pos,
                                  tol=noise_floor(CHARACTER, eps),
                                  joint=verdict.get("joint", ()),
                                  search=verdict.get("search"))
        if isinstance(sub, Refusal):
            return sub
        if sub["broken"]:
            # THE DECLARED CONTENT IS NOT COVARIANT ON ALL OF THIS GROUP, and
            # this call was handed a fixed group and cannot lower it.  So the
            # honest answer is the subgroup it IS covariant on -- the group
            # `realized_group` descends to, made by the same function there.
            return _declaration_breaks(
                group, assignment, decls, verdict,
                standing=[int(syms[p].ind) for p in kept], subgroup=sub["subgroup"],
                broken=sub["broken"], readings=sub["candidates"],
                not_lowered=("The representation was asked for on this group, which it "
                             "cannot lower."),
                site_tables=site_tables, admissible=admissible)

    def _relative(p_gen, cls):
        """The class scalar the build assembles with, at one generator.

        What the measurement pinned relative to the anchor; and on an orbit the
        hoppings leave free of it, the trivial relative character -- the
        labelling chooses its part's own afterwards.
        """
        if int(cls) in unpinned:
            return 1.0 + 0.0j
        got = verdict["chosen"].get((int(syms[p_gen].ind), int(cls)))
        return None if got is None else complex(got)

    # ---- 4c. THE COUPLED PARTS OF THE CRYSTAL, as the measurement read them --
    # The hoppings fix every multiplet's block up to one one-dimensional
    # character per coupled part -- the multiplets they connect, directly or
    # through others: the anchor's first, then the orbits the hoppings leave
    # free of it, grouped where they pin one another.  Inside a part the orbits
    # carry the ratios the hoppings pin between them.  Derived once, here.
    coupled_parts = _coupled_parts(classes, unpinned, verdict.get("parts"))
    part_of_class = {int(cls): i for i, part in enumerate(coupled_parts) for cls in part}
    free_lead = {int(cls): int(min(part)) for part in coupled_parts[1:] for cls in part}

    def _assemble(p, S, sign_of_class):
        D = np.zeros((nstate, nstate), dtype=complex)
        for c in range(nchan):
            rows = np.asarray(chan_states[int(cperm[p][c])], dtype=int)
            cols = np.asarray(chan_states[c], dtype=int)
            D[np.ix_(rows, cols)] = sign_of_class[int(classes[c])] * S[c]
        return D

    tree = {"gens": gens, "words": words, "parent": parent, "spf": spf, "ident": ident,
            "nstate": nstate, "T_pos": T_pos, "au_of": au_of}

    def _propagate(Dgen, DT):
        """Propagate the generators along the word tree (:func:`_propagated`)."""
        return _propagated(Dgen, DT, **tree)

    # ---- 5. the covariance battery ----------------------------------------
    k_pts = _covariance_battery(group, k_pts, n_battery, seed)
    bloch = _bloch.evaluator(model, params)
    H = np.asarray(bloch(k_pts))
    if tol is None:
        # the rounding this representation's own arithmetic carries, measured
        # on the battery it is gated on, and never below the exact floor
        tol = covariance_gate(len(kept), nstate, float(np.abs(H).max()))
    Hrot, kact = {}, {}
    for p in kept:
        act = group.k_action_of(syms[p])
        kact[p] = act
        Hrot[p] = np.asarray(bloch(act(k_pts)))

    def _covariance(D):
        worst = 0.0
        for p in kept:
            U = D[p]
            src = H.conj() if syms[p].time_reversal else H
            worst = max(worst, float(np.abs(U @ src @ U.conj().T - Hrot[p]).max()))
        return worst

    def _gate_lift(p):
        r"""``[S per channel]`` the closed gate below measures against.

        It is the block :func:`_multiplet_blocks` settled on — the one the
        matrices were assembled from, so that the gate is the statement that
        every multiplet carries its own block and not a comparison against
        whichever member of the solution space a polar extraction happened to
        return — and the operation's own lift, the tables'
        (``sym.spinor_rotation``), on a one-dimensional multiplet no block was
        read for.  An entry is ``None`` where no block could be measured, and
        that multiplet is skipped: there is nothing to test it against.
        """
        lift = _lift_of(syms[p], su2)
        blocks = measured_of.get(int(syms[p].ind), {})
        return [blocks.get(int(c),
                           None if layout._chan_kind[c] == "measured" else lift)
                for c in range(nchan)]

    _gate_lifts = {p: _gate_lift(p) for p in u_pos}

    def _internal_defect(D):
        r"""Worst factorization defect ``max ‖B(g) − χ·S(g)‖``, over **every** block.

        The tensor structure behind the build is ``D(g) = O(g) ⊗ S(g)``, and
        that is a statement about every multiplet block of ``D(g)`` — the block
        carrying a multiplet to its *image* as much as the block of a multiplet
        the operation fixes.  This measures it on every one of them, with the
        per-block coefficient free (the tensor-structure subspace), so it is
        blind to the one-dimensional character the labelling puts in
        afterwards.  A candidate with a large defect carries no site
        characters at all: its blocks are some *other* SU(2) element, so "the
        ``d`` orbital at ``1a``" has nothing to be measured against.

        **Why every block and not only the fixed ones.**  Tested on the fixed
        multiplets alone the gate has a hole, and it is not a small one: the
        ``σ_h``-partner ``σ_h·g`` of an operation has a *different* lift, so a
        Hamiltonian written in the opposite spin convention can carry the
        partner's lift on every site-swapping block and still close a gate that
        never looks at one.  Measured: the SRSI paper's ``P41'`` model
        transcribed raw admits a faithful representation of ``P-4m2`` under the
        name ``P4mm1' #99.164`` — the two are isomorphic, so the product table
        closes at ``5e-16`` too — with ``C4``, ``C4⁻¹`` and both diagonal
        mirrors carrying ``S(σ_h)·S(g)``.  Closing the gate over every block
        refuses that build instead of returning it.

        This is also what separates the ``g``-versus-``g⁻¹`` orientation branch
        on a spinful model, and neither of the other two gates can see it — the
        covariance battery and the product table are both statements about the
        ``D``'s among themselves, while this one is the statement that they are
        the ``D``'s of *these* operations.  Measured on the search matrices: it
        is invariant under ``D → λD``, so normalization cannot change it.

        Vacuous where there is no internal factor to check: a spinless model
        with no declared spinor basis, whose ``S`` is ``1×1`` and every block a
        multiple of it by arithmetic.
        """
        worst = 0.0
        for p in u_pos:
            lifts = _gate_lifts.get(p) or []
            for c in range(nchan):
                S = lifts[c] if c < len(lifts) else None
                if S is None:
                    continue
                rows = np.asarray(chan_states[int(cperm[p][c])], dtype=int)
                cols = np.asarray(chan_states[c], dtype=int)
                _, res = _proportionality(D[p][np.ix_(rows, cols)], S)
                worst = max(worst, float(res))
        return worst

    # ---- 5b. normalization, as a function of one candidate ------------------
    # Phase-canonicalize the generators (canonical norm + antiunitary anchor +
    # positivity) and propagate along the same BFS tree.  Written as a function
    # of the candidate rather than of the winner so that the product-table gate
    # can run on the matrices the object will carry: `search["product"]`
    # and `SpaceGroupRep.product_residual` are then the same measurement rather
    # than two drafts of it.  Cheap enough because it runs only on candidates
    # that already closed the covariance and internal-factor gates.
    #
    # THE EXACT REPRESENTATION NEAREST THE MODEL'S OWN BLOCKS.  A wider
    # multiplet's block is the model's own, so a near-symmetric model's carries
    # its noise and the candidate composes by the table only to that noise --
    # no canonical norm can be read off it.  Where it misses by more than the
    # norm's own gate it is replaced by the exact representation nearest it
    # (:func:`_settled`).  Only a miss the model's own noise explains is
    # projected -- within a hundred times the internal factor's tolerance,
    # which is the noise the build was told to expect: a candidate that misses
    # the table by a sign misses it by order one, and the product-table gate is
    # what refuses it.
    reach = min(1e-2, 100.0 * float(internal_tol))

    # ---- 5c. THE SETTLING, one coupled part at a time --------------------------
    # A part holding a one-dimensional multiplet is written in the tables' own
    # lift: its blocks are that lift, carried round each orbit with the signs
    # the hoppings give it, and a wider multiplet the hoppings couple to it is
    # put in the same phase (`_multiplet_blocks`), so the parts that hold one
    # are settled together, by one canonical norm.  A part of wider multiplets
    # alone carries its blocks as the model's own, read off one member of a
    # solution space, and so carries that member's free phase, which no other
    # part shares.  So each such part is settled on its own, and a crystal of
    # one part is one of them: its generators in their canonical norm and
    # anchor at its own lead multiplet (the anchor's for the anchor's own part),
    # the rest propagated, and -- wherever the signs positivity picks there do
    # not multiply: no antiunitary operation and a multi-dimensional lead with
    # no table lift behind it, or a one-dimensional lead whose character on its
    # site group no one-dimensional character of the group restricts to -- the
    # multiplication table choosing among the roots each canonical norm admits
    # (:func:`_table_phases`).  Spin-1/2 states read in
    # single-valued tables arrive there: their matrices are a projective
    # representation, and where its class is trivial the phases that make it a
    # representation of the group are among these roots; a group with two
    # perpendicular twofolds, whose spinor matrices anticommute, has none, and
    # the refusal below says so.  The table anchor applies to a one-dimensional
    # lead in a group with no antiunitary operation (:attr:`RepAnchor.internal`):
    # there `sym.spinor_rotation` IS the internal factor, read from the shipped
    # datum rather than from what was assembled.  What is left on each part is
    # its one-dimensional character, which the labelling chooses.
    n_parts = len(coupled_parts)
    lifted = [any(layout._chan_kind[c] == "tables" for c in range(nchan)
                  if part_of_class.get(int(classes[c]), 0) == i) for i in range(n_parts)]
    held_together = ([tuple(range(n_parts))] if n_parts == 1 or all(lifted) else
                     ([tuple(i for i in range(n_parts) if lifted[i])] if any(lifted) else [])
                     + [(i,) for i in range(n_parts) if not lifted[i]])
    units = []
    for held in held_together:
        its = [c for c in range(nchan) if part_of_class.get(int(classes[c]), 0) in held]
        where = sorted(int(a) for c in its for a in chan_states[c])
        units.append({"states": np.asarray(where, dtype=int),
                      "lead": _lead_of(its, anchor),
                      "local": {a: i for i, a in enumerate(where)},
                      "anchor": 0 in held,
                      "orbitals": tuple(sorted(int(o) for c in its
                                               for o in layout._chan_orbs[c]))})

    def _part_anchor(unit):
        """The :class:`RepAnchor` of one part's generators, at the part's own lead."""
        lead, local = int(unit["lead"]), unit["local"]

        def _anchor(p, anchor_T):
            table = None
            if anchor_T is None and layout._chan_kind[lead] == "tables":
                table = (np.asarray(syms[p].spinor_rotation, dtype=complex) if su2
                         else np.eye(1, dtype=complex))
            return RepAnchor(antiunitary=anchor_T,
                             rows=tuple(local[int(a)] for a in chan_states[int(cperm[p][lead])]),
                             cols=tuple(local[int(a)] for a in chan_states[lead]),
                             internal=table)
        return _anchor

    def _part_unsettled(bad, unit):
        """The refusal of a part whose own blocks settle into no representation, said as one."""
        if len(units) == 1:
            return bad
        return Refusal(
            bad.gate,
            f"the coupled part of the crystal holding orbitals {unit['orbitals']} -- the "
            f"multiplets the hoppings connect to one another and to nothing else -- was "
            f"settled on its own in {group.name} #{group.number_str}, and its own blocks do "
            f"not settle into a representation of the group: {bad.reason}",
            value=bad.value, threshold=bad.threshold,
            context=dict(bad.context or {}, part=unit["orbitals"]))

    def _settled_by_part(Dsearch):
        """``(refusal, D, normalization, anchor_T, DT_n, defect, roots)``, part by part.

        Each part is settled on its own by :func:`_settled`, and by
        :func:`_table_phases` where nothing anchors its generators' phases; the
        matrices are then the parts' side by side.  ``normalization`` is the
        anchor's part's, written on the whole matrices, and ``roots`` is
        ``(found, matrices)`` for the first part whose roots do not multiply.
        """
        eye_n = np.eye(nstate, dtype=complex)
        Dn = {p: np.zeros((nstate, nstate), dtype=complex) for p in Dsearch}
        anchor_n = None if T_pos is None else np.zeros((nstate, nstate), dtype=complex)
        DT_all = None if T_pos is None else np.zeros((nstate, nstate), dtype=complex)
        worst, norm_ref, roots = 0.0, {}, None
        for unit in units:
            idx = unit["states"]
            here = dict(tree, nstate=int(len(idx)))
            sub = {p: np.asarray(M, dtype=complex)[np.ix_(idx, idx)] for p, M in Dsearch.items()}
            bad, D_u, norm_u, anchor_u, DT_u, defect_u, roots_u = _settle_unit(
                sub, group=group, tree=here, kept=kept, u_pos=u_pos, prod=prod,
                anchor_of=_part_anchor(unit), reach=reach, product_tol=product_tol)
            worst = max(worst, float(defect_u))
            if bad is not None:
                return _part_unsettled(bad, unit), None, None, None, None, worst, None
            if roots is None:
                roots = roots_u
            for p, M in D_u.items():
                Dn[p][np.ix_(idx, idx)] = M
            if T_pos is not None:
                anchor_n[np.ix_(idx, idx)] = anchor_u
                DT_all[np.ix_(idx, idx)] = DT_u
            if unit["anchor"]:
                norm_ref = norm_u
        at = {int(syms[p].ind): p for p in kept}
        normalization = {}
        for ind, rec in norm_ref.items():
            M = Dn[at[int(ind)]]
            normalization[int(ind)] = replace(
                rec, matrix=M,
                norm_residual=float(np.abs(_power(M, bool(syms[at[int(ind)]].time_reversal),
                                                  rec.order) - rec.required * eye_n).max()))
        return None, Dn, normalization, anchor_n, DT_all, worst, roots

    # ---- 6. the search ----------------------------------------------------
    phases = tuple(complex(z) for z in phases)
    n_free = len(gens) * len(free_cls)
    if orientation_branch is None:
        orients = (1, -1)
    elif isinstance(orientation_branch, bool):
        # a boolean is not a branch: True would read as +1 through int(), silently
        raise ValueError("orientation_branch must be None (both branches), +1 (the g "
                         "branch) or -1 (the g^-1 branch), not a boolean; got "
                         f"{orientation_branch!r}")
    elif int(orientation_branch) == 1:
        orients = (1,)
    elif int(orientation_branch) == -1:
        orients = (-1,)
    else:
        raise ValueError("orientation_branch must be None (both branches), +1 (the g "
                         "branch) or -1 (the g^-1 branch), got "
                         f"{orientation_branch!r}")

    # WHERE THE CHOICE APPLIES: every block of the operation at once.  A
    # relative character the Hamiltonian leaves open is assembled here as the
    # trivial one (`_relative`), the labelling choosing its part's own in
    # step 7b, and that is one statement about the whole matrix, not one per
    # site.  Two sites the Hamiltonian couples to each other not at all are as
    # free as two multiplets of one site, and `RealizedGroup.conserved` is
    # where the decoupling is said.
    all_blocks = [list(range(nchan))]

    # ---- 6a. the read-off ---------------------------------------------------
    # The detector has already solved `X H(k) = H(Rk) X` for every operation
    # this build represents, and `validated._bases` carries the whole solution
    # space.  A realizable tensor structure `D(g) = O(g) (x) S(g)` is an element
    # of it, so the discrete choices are a *measurement* on that space rather than a
    # search through their product set.  Only the generators are read: the
    # antiunitary operator carries the scalars the measurement of step 4 read
    # once, and no sign left to search, in both strategies, and every other
    # operation is propagated along the BFS tree.
    def _solution_space(p):
        """``(basis, note)`` — the detector's solution space for operation ``p``."""
        ind = int(syms[p].ind)
        B = getattr(validated, "_bases", {}).get(ind)
        if B is None or len(np.asarray(B)) == 0:
            U = validated._intertwiners.get(ind)
            if U is None:
                return None, (f"the validated group carries no measured intertwiner for "
                              f"generator #{ind}")
            # A hand-built ValidatedGroup may carry only the polar-extracted
            # member.  That IS the space when dim == 1, and when it is not the
            # projection below has nothing to choose from and the block defects
            # say so -- which is a fallback, not a wrong verdict.
            return np.asarray(U)[None, :, :], f"only `intertwiners[{ind}]`, not the whole space"
        B = np.asarray(B)
        if B.shape[0] * nstate * nstate * 16 > _READOFF_MAX_BYTES:
            return None, (f"the solution space of generator #{ind} is "
                          f"{B.shape[0]}-dimensional at nstate = {nstate}, whose "
                          f"projection onto the tensor structure would allocate more "
                          f"than {_READOFF_MAX_BYTES / 1e6:.0f} MB")
        return B, None

    def _read_off():
        """The discrete choices, measured off the solution spaces of the generators."""
        rec = {"n_enumeration": None, "branches": {}, "reason": None, "notes": []}
        branches, choices, reasons, read_of = {}, [], [], {}
        for orient in orients:
            per_gen, ratios, defects, combo, why = {}, {}, {}, [], None
            read_as = {}
            for gi, gp in enumerate(gens):
                ind = int(syms[gp].ind)
                basis, note = _solution_space(gp)
                if note and note not in rec["notes"]:
                    rec["notes"].append(note)
                if basis is None:
                    why = note
                    break
                lifts = S_gen[gp]
                if orient < 0:
                    # the g^-1 branch's candidate is the adjoint, so its blocks
                    # are the transposed ones and carry S(g)^dagger
                    lifts = [M.conj().T for M in lifts]
                blocks = _ansatz_blocks(chan_states, cperm[gp], orient)
                # THE CHOSEN MEMBER.  Where the space leaves a block open, the
                # member taken is the one carrying the scalar `_relative` gives
                # there -- what the measurement pinned, or the trivial relative
                # character, the labelling being chosen in step 7b -- rather
                # than whichever direction a decomposition happened to return.
                want = {cls: _relative(gp, cls) for cls in range(ncls)}
                w_of = [(1.0 + 0.0j) if want[int(classes[c])] is None else
                        complex(np.conj(want[int(classes[c])]) if orient < 0
                                else want[int(classes[c])])
                        for c in range(nchan)]
                X, defect, scale = _ansatz_member(basis, blocks, lifts,
                                                  prefer=all_blocks, weights=w_of,
                                                  tol=zero)
                w_, sv_, vh_ = np.linalg.svd(X)
                entry = {"ind": ind, "dim": int(np.asarray(basis).shape[0]),
                         "ansatz_defect": float(defect), "ansatz_scale": float(scale),
                         "conditioning": float(sv_[-1] / sv_[0]) if sv_[0] > 0 else 0.0}
                per_gen[ind] = entry
                if float(sv_[0]) <= 1e-12:  # pragma: no cover - a guard, not a path
                    why = (f"the projection of generator #{ind}'s solution space onto "
                           f"the tensor structure is the zero matrix")
                    break
                U = w_ @ vh_
                cls_ratio, cls_defect = {}, {}
                for cls in range(ncls):
                    c0 = (anchor if cls == anchor_cls
                          else int(np.flatnonzero(classes == cls)[0]))
                    rows, cols = blocks[c0]
                    rho, res = _proportionality(U[np.ix_(rows, cols)], lifts[c0])
                    cls_ratio[cls], cls_defect[cls] = complex(rho), float(res)
                entry["block_defect"] = max(cls_defect.values())
                lam = cls_ratio[anchor_cls]
                entry["lambda"] = complex(lam)
                if abs(lam) < snap:
                    why = (f"generator #{ind}'s block on orbital 0's multiplet is not a "
                           f"multiple of S(g): |lambda| = {abs(lam):.3g}")
                    break
                signs = {cls: (np.conj(r / lam) if orient < 0 else r / lam)
                         for cls, r in cls_ratio.items()}
                # WHAT THE MEASUREMENT PINNED is taken from it: on the branch it
                # read, every orbit's ratio to the anchor, rather than read again
                # off the member above, which says only whether the tensor
                # structure holds
                if orient > 0:
                    for cls in range(ncls):
                        got = verdict["chosen"].get((ind, int(cls)))
                        if (int(cls) != anchor_cls and int(cls) not in unpinned
                                and got is not None):
                            signs[int(cls)] = complex(got)
                # An orbit the hoppings leave free of the anchor carries the
                # trivial relative character (`_relative`; the labelling
                # chooses its part's own in step 7b); the other orbits of its
                # part are pinned to its first by the hoppings, at the ratio the
                # measurement read between them.
                led = sorted(unpinned, key=lambda c: (free_lead.get(int(c), int(c)) != int(c),
                                                      int(c)))
                for cls in led:
                    lead_cls = free_lead.get(int(cls), int(cls))
                    rel = (verdict.get("within", {}).get((ind, int(cls))) if orient > 0
                           else None)
                    if rel is None and lead_cls != int(cls) and abs(cls_ratio[lead_cls]) > zero:
                        rel = cls_ratio[int(cls)] / cls_ratio[lead_cls]
                        rel = np.conj(rel) if orient < 0 else rel
                    if lead_cls != int(cls) and rel is not None:
                        signs[int(cls)] = complex(signs[lead_cls]) * complex(rel)
                    elif want.get(int(cls)) is not None:
                        signs[int(cls)] = complex(want[int(cls)])
                entry["signs"] = {int(c): complex(z) for c, z in signs.items()}
                for cls in range(ncls):
                    ratios[(ind, int(cls))] = complex(signs[cls])
                    defects[(ind, int(cls))] = float(cls_defect[cls])
                bad = max(cls_defect, key=lambda c: cls_defect[c])
                if cls_defect[bad] > internal_tol:
                    why = (f"generator #{ind}'s block on orbit {bad} of the multiplets "
                           f"is not a multiple of S(g), the operation's own internal "
                           f"factor: it misses one by "
                           f"{cls_defect[bad]:.3g} > {internal_tol:.3g}")
                    break
                factors = []
                for c in free_cls:
                    if int(c) in unpinned:
                        # settled, not read against the anchor: the hoppings
                        # leave this orbit free of it, and it carries the factor
                        # its part is settled with above
                        factors.append(complex(signs[c]))
                        continue
                    # THE FACTOR AS MEASURED, whatever it is: a root of unity
                    # where it is one within the noise, the measured phase
                    # otherwise; the three gates say whether it builds a
                    # representation
                    z = _orbit_factor(signs[c], snap)
                    if z is None:
                        why = (f"generator #{ind} reads {signs[c]:.6g} on orbit {c} of "
                               f"the multiplets, which is no phase: its modulus misses one "
                               f"by {abs(abs(signs[c]) - 1.0):.3g}")
                        break
                    factors.append(z)
                    read_as[(ind, int(c))] = z
                if why is not None:
                    break
                combo.extend(factors)
            branch = {"orientation": int(orient), "generators": per_gen,
                      "ratios": ratios, "defects": defects,
                      "ok": why is None, "reason": why}
            rec["branches"][int(orient)] = branch
            if why is None:
                branches[int(orient)] = branch
                choices.append((orient, tuple(combo)))
                read_of[int(orient)] = read_as
            else:
                reasons.append(f"branch {orient:+d}: {why}")
        if not choices:
            rec["reason"] = "; ".join(reasons) if reasons else "no branch was read"
        return {"choices": choices, "branches": branches,
                "reason": rec["reason"], "record": rec, "read": read_of}

    def _build(choices):
        """One candidate record per ``(orientation, combo)``, covariance measured."""
        out = []
        for orient, combo in choices:
            sign = {}
            for gi in range(len(gens)):
                sgn = np.ones(ncls, dtype=complex)
                for j, c in enumerate(free_cls):
                    sgn[c] = combo[gi * len(free_cls) + j]
                sign[gi] = sgn
            Dgen = [_assemble(gens[gi], S_gen[gens[gi]], sign[gi]) for gi in range(len(gens))]
            DT = None
            if T_pos is not None:
                DT = _assemble(T_pos, S_gen[T_pos], reversal)
            if orient < 0:
                Dgen = [d.conj().T for d in Dgen]
                DT = None if DT is None else DT.conj().T
            D = _propagate(Dgen, DT)
            out.append({"orientation": int(orient), "combo": tuple(combo),
                        "residual": _covariance(D), "matrices": D})
        return out

    # ---- 6b. the joint gate -------------------------------------------------
    # Three independent measurements, each of which a candidate must pass, run
    # cheapest first so the expensive ones see the fewest candidates:
    #
    #   covariance      |kept| * n_battery triple products of size nstate;
    #                   kills a wrong relative sign inside a Wyckoff orbit.
    #   internal factor |kept_u| * nchan block proportionalities of size nspin
    #                   -- EVERY block, the site-swapping ones too; kills the
    #                   g-versus-g^-1 orientation branch, which is exactly
    #                   covariant AND exactly multiplicative, and a build whose
    #                   swapped blocks carry the sigma_h-partner's lift, which
    #                   is exactly covariant and multiplicative too whenever the
    #                   layer group and the one being named are isomorphic.
    #   product table   one normalize_rep pass + |kept|**2 products of size
    #                   nstate; kills a set that is not a representation.
    #
    # The ordering is a cost choice only -- the gates are independent, and which
    # one fired is recorded per candidate, which is physics rather than
    # accounting.  Written as a function of the candidate *set* so that the
    # read-off's one-or-two candidates and the enumeration's thousands go
    # through the identical arithmetic: the read-off finds the candidate, this
    # accepts it, and nothing here knows which produced the input.
    def _gate(records):
        # merge candidates that build the identical representation
        distinct = []
        for rec in records:
            for grp in distinct:
                ref = grp[0]["matrices"]
                if all(np.abs(rec["matrices"][p] - ref[p]).max() < 1e-12 for p in kept):
                    grp.append(rec)
                    break
            else:
                distinct.append([rec])

        covariant = [grp for grp in distinct if grp[0]["residual"] <= tol]
        for grp in covariant:
            grp[0]["internal"] = _internal_defect(grp[0]["matrices"])
        consistent = [grp for grp in covariant if grp[0]["internal"] <= internal_tol]
        for grp in consistent:
            rec = grp[0]
            # each coupled part settled on its own (step 5c), a crystal of one
            # part being one
            bad, Dn, norm, anchor_T, DT_n, defect, roots = _settled_by_part(rec["matrices"])
            rec["composition"] = float(defect)
            if bad is not None:
                rec["product"], rec["refusal"] = float("inf"), bad
                continue
            rec["normalized"] = (Dn, norm, anchor_T, DT_n)
            rec["product"] = _product_table_residual(Dn, kept, syms, prod, spf)
            if roots is not None:
                rec["roots"], rec["roots_matrices"] = roots
        survivors = [grp for grp in consistent if grp[0]["product"] <= product_tol]
        if len(units) > 1 and len(survivors) > 1:
            # A PART'S OWN PHASE IS ITS SETTLING'S, not a choice: candidates
            # that differ only in the phase a part is settled out of are one
            # representation once each part is settled, and are merged as the
            # identical matrices are above
            merged = []
            for grp in survivors:
                Dn = grp[0]["normalized"][0]
                for keep in merged:
                    same = keep[0]["normalized"][0]
                    if all(np.abs(Dn[p] - same[p]).max() < 1e-12 for p in kept):
                        keep.extend(grp)
                        break
                else:
                    merged.append(grp)
            gone = {id(grp) for grp in survivors} - {id(grp) for grp in merged}
            distinct = [grp for grp in distinct if id(grp) not in gone]
            covariant = [grp for grp in covariant if id(grp) not in gone]
            consistent = [grp for grp in consistent if id(grp) not in gone]
            survivors = merged
        alive = {id(grp) for grp in survivors}

        def _killed_by(grp):
            if id(grp) in alive:
                return None
            if grp[0]["residual"] > tol:
                return "covariance"
            if grp[0]["internal"] > internal_tol:
                return "internal_factor"
            return "product_table"

        spectrum = tuple(sorted(float(grp[0]["residual"]) for grp in distinct))
        int_spectrum = tuple(sorted(float(grp[0]["internal"]) for grp in covariant))
        prod_spectrum = tuple(sorted(float(grp[0]["product"]) for grp in consistent))
        candidates = tuple(
            {"orientation": int(grp[0]["orientation"]), "combo": grp[0]["combo"],
             "n_merged": len(grp), "covariance": float(grp[0]["residual"]),
             "internal": float(grp[0].get("internal", float("nan"))),
             "product": float(grp[0].get("product", float("nan"))),
             "killed_by": _killed_by(grp)}
            for grp in distinct)

        return {
            "records": records, "distinct": distinct, "covariant": covariant,
            "consistent": consistent, "survivors": survivors,
            "search": {"n_candidates": len(records), "n_distinct": len(distinct),
                       "n_covariant": len(covariant), "n_consistent": len(consistent),
                       "n_survivors": len(survivors),
                       "gate": float(tol), "internal_gate": float(internal_tol),
                       "product_gate": float(product_tol),
                       "best": spectrum[0] if spectrum else float("nan"),
                       "runner_up": spectrum[1] if len(spectrum) > 1 else float("nan"),
                       "residuals": spectrum,
                       "internal_best": (int_spectrum[0] if int_spectrum
                                         else float("nan")),
                       "internal_runner_up": (int_spectrum[1] if len(int_spectrum) > 1
                                              else float("nan")),
                       "internal_residuals": int_spectrum,
                       "product_best": (prod_spectrum[0] if prod_spectrum
                                        else float("nan")),
                       "product_runner_up": (prod_spectrum[1] if len(prod_spectrum) > 1
                                             else float("nan")),
                       "product_residuals": prod_spectrum,
                       "killed_by": {"covariance": len(distinct) - len(covariant),
                                     "internal_factor": len(covariant) - len(consistent),
                                     "product_table": len(consistent) - len(survivors)},
                       "candidates": candidates}}

    # ---- 6b'. the orientation branch, pinned by convention when the gates tie
    # The internal-factor gate separates the two branches whenever
    # `S(g)^dag != +-S(g)` -- every rotation of order three or more -- because
    # the `g^-1` candidate carries `S(g)^dag` where the group says `S(g)`.  For a
    # group of involutions only (`mm2`, `P-1`, a `2/m` on a spinful model) the
    # lift satisfies `S(g)^dag = -S(g)`, the free per-block coefficient of that
    # gate absorbs the sign, and all three gates are blind to the branch.  There
    # is then nothing left to measure, and refusing would only demand a
    # convention the build is in a better position to state: when the two
    # branches' candidates both close all three gates AND are one representation
    # up to a one-dimensional character on each coupled part of the crystal --
    # so that the same labellings, and the same site irreps, are open on both --
    # the read-off's own branch `+1` -- the intertwiner that carries `k -> gk` --
    # is returned, and `branch_tie` says the choice was a convention rather than
    # a measurement.  `rep_not_unique` remains the verdict for everything else:
    # survivors that are not one representation up to such a character, more
    # than two of them, or two on the same branch.
    #
    # It runs on the candidate SET, before the strategy dispatch, for the same
    # reason `_gate` does: the read-off's two candidates are then accepted where
    # the tie resolves, instead of being handed to an enumeration that would
    # rediscover the identical pair -- and on a model whose enumeration is
    # refused (`rep_search_too_large`) that is the difference between a returned
    # representation and none at all.
    part_chans = [[int(c) for c in range(nchan) if int(classes[c]) in part]
                  for part in coupled_parts]

    def _one_labelling_apart(A, B, tol=1e-8):
        """Whether two settled candidates differ by one character on each coupled part.

        Asked of the unitary operations, which the site irreps and the band
        labels are characters of.
        """
        for p in u_pos:
            for chans in part_chans:
                ratio = None
                for c in chans:
                    rows = np.asarray(chan_states[int(cperm[p][c])], dtype=int)
                    cols = np.asarray(chan_states[c], dtype=int)
                    a, b = A[p][np.ix_(rows, cols)], B[p][np.ix_(rows, cols)]
                    rho, res = _proportionality(b, a)
                    if res > tol or abs(abs(rho) - 1.0) > tol:
                        return False
                    if ratio is None:
                        ratio = rho
                    elif abs(rho - ratio) > tol:
                        return False
        return True

    def _pin_branch(survivors):
        """``{survivors, tie, note}`` — the tie settled, or left alone."""
        out = {"survivors": survivors, "tie": False, "note": None}
        if len(survivors) != 2:
            return out
        per = {}
        for grp in survivors:
            per.setdefault(int(grp[0]["orientation"]), []).append(grp)
        if sorted(per) != [-1, 1] or any(len(v) != 1 for v in per.values()):
            return out
        if site_tables and not _one_labelling_apart(per[1][0][0]["normalized"][0],
                                                    per[-1][0][0]["normalized"][0]):
            return out
        out.update(survivors=[per[1][0]], tie=True, note=(
            "the two orientation branches both carry H(k) onto itself, factorize on every "
            "site into (character) x (the group's own internal factor) and reproduce the "
            "multiplication table, and are one "
            "representation up to a one-dimensional character on each coupled part, so "
            "the same labellings and site labels are open on both and nothing "
            "measures between them; the branch was pinned by convention to +1, "
            "the intertwiner that carries k -> gk."))
        return out

    # ---- 6c. the read-off, and the enumeration behind it --------------------
    n_enumeration = int(len(phases) ** n_free * len(orients))
    read = {"choices": (), "branches": {}, "reason": None, "read": {},
            "record": {"n_enumeration": n_enumeration, "branches": {}, "reason": None,
                       "notes": []}}
    if strategy == "readoff":
        try:
            read = _read_off()
        except np.linalg.LinAlgError as exc:  # pragma: no cover - not seen on any fixture
            # The read-off is a way to find the candidate faster; it must never
            # be the reason a build refuses.  An SVD that will not converge is a
            # reason to enumerate, not an error to return.
            read["reason"] = (f"the projection onto the tensor structure did not "
                              f"converge ({exc})")
    read["record"]["n_enumeration"] = n_enumeration
    outcome, strategy_used, fallback, pin = None, None, None, None
    declared_tol = float(getattr(validated, "tol", tol))
    if strategy == "readoff" and read["choices"]:
        outcome = _gate(_build(read["choices"]))
        pin = _pin_branch(outcome["survivors"])
        best = float(outcome["search"]["best"])
        if len(pin["survivors"]) == 1:
            strategy_used = "readoff"
        elif not outcome["covariant"] and declared_tol > tol and tol < best <= declared_tol:
            # WHAT WAS READ IS NEAR-SYMMETRIC: the representation assembled from
            # what the solution spaces measured at the declared tolerance closes
            # it and misses only the exact gate.  That is the verdict -- the
            # operations, or parts of the crystal coupled below the precision,
            # obeyed within what was declared and not exactly -- and no other
            # choice of signs, which the measurement never pinned, stands in for
            # it; the projection is the route.
            strategy_used = "readoff"
        else:
            # The read-off is a way to FIND the candidate, never to accept one:
            # a read that the gates do not confirm is handed back to the
            # enumeration of the phases on offer, whose refusal (or survivor) is
            # the one returned, and what was read is said with it.
            said = "; ".join(
                f"branch {orient:+d}: " + (", ".join(
                    f"generator #{ind} reads {_factor_said(z)} on orbit {c}"
                    for (ind, c), z in sorted(got.items())) or "no orbit read")
                for orient, got in sorted(read["read"].items(), reverse=True))
            fallback = (f"the factors measured between orbits of multiplets ({said}) "
                        f"do not give one representation: {len(outcome['survivors'])} of "
                        f"the {outcome['search']['n_distinct']} distinct candidates built "
                        "from them carry H(k) onto itself, factorize on every site into "
                        "(character) x (the group's own internal factor) and reproduce the "
                        "multiplication table (the candidates each test removed: "
                        f"{outcome['search']['killed_by']})")
            outcome, pin = None, None
    elif strategy == "readoff":
        fallback = read["reason"]

    if outcome is None:
        if n_enumeration > max_candidates:
            try:
                measured = float(n_enumeration)
            except OverflowError:  # pragma: no cover - past 2**1024 choices
                measured = float("inf")
            hint = ("" if fallback is None else
                    f" The read-off deferred to it because {fallback}, which is the "
                    f"thing to act on: a read that is not clean says the tensor "
                    f"structure D(g) = O(g) (x) S(g) does not describe this operation "
                    f"set on this orbital content, and enumerating it would not make it "
                    f"one.")
            return Refusal(
                "rep_search_too_large",
                f"the enumeration of this build's discrete choices would run "
                f"{n_enumeration:,} candidates -- {len(phases)} phases on "
                f"{len(gens)} generator(s) x {ncls - 1} free orbit(s) of multiplets of "
                f"{ncls}, times {len(orients)} orientation branch(es) -- against "
                f"max_candidates = {max_candidates:,}. At the few milliseconds a "
                f"candidate costs that is hours, so it is refused rather than run."
                f"{hint} A `multiplets=` stating a multi-orbital site as one multiplet "
                f"cuts the class count sharply, and `realized_group(..., "
                f"max_candidates=...)` with a larger ceiling runs the enumeration if it "
                f"is worth waiting for.",
                value=measured, threshold=float(max_candidates),
                context={"n_generators": len(gens), "n_classes": int(ncls),
                         "n_free": int(n_free), "n_candidates": n_enumeration,
                         "max_candidates": int(max_candidates),
                         "multiplets": layout, "fallback_reason": fallback})
        outcome = _gate(_build((orient, combo) for orient in orients
                               for combo in itertools.product(phases, repeat=n_free)))
        strategy_used = "enumerate"

    records, distinct = outcome["records"], outcome["distinct"]
    covariant, consistent = outcome["covariant"], outcome["consistent"]
    survivors = outcome["survivors"]
    spectrum = outcome["search"]["residuals"]
    int_spectrum = outcome["search"]["internal_residuals"]
    prod_spectrum = outcome["search"]["product_residuals"]

    # the branch tie of step 6b', measured on the candidate set that was accepted
    if pin is None:
        pin = _pin_branch(survivors)
    survivors = pin["survivors"]
    branch_tie, branch_note = pin["tie"], pin["note"]

    search_record = dict(outcome["search"])
    # The headline read-off numbers describe **the read that produced the
    # accepted representation** -- the surviving branch.  The losing branch's
    # numbers are a measurement of a candidate nobody kept (on a generator whose
    # multiplet permutation is not an involution the `g^-1` pattern is disjoint
    # from the `g` one, and its tensor-structure defect is the whole norm of the
    # solution),
    # so they stay in `search["readoff"]["branches"]` rather than standing for
    # the build.
    accepted = {}
    if strategy_used == "readoff" and len(survivors) == 1:
        won = int(survivors[0][0]["orientation"])
        accepted = {won: read["branches"][won]} if won in read["branches"] else {}
    search_record.update({
        "strategy": strategy_used,
        "requested": str(strategy),
        "fallback_reason": fallback,
        "n_enumeration": n_enumeration,
        "max_candidates": int(max_candidates),
        "branch_tie": bool(branch_tie),
        "branch_note": branch_note,
        "ansatz_defect": max((float(g["ansatz_defect"])
                              for b in accepted.values()
                              for g in b["generators"].values()), default=float("nan")),
        "ratios": {k: v for b in accepted.values() for k, v in b["ratios"].items()},
        "defects": {k: v for b in accepted.values() for k, v in b["defects"].items()},
        "readoff": read["record"]})
    search = search_record
    if _search_record is not None:
        # the record of the search, handed back where asked beside a refusal,
        # which carries only the number of candidates tried
        _search_record["search"] = search
    # A declared spinor basis is a tensor structure the gates below test: a pairing that
    # is well-formed but wrong (partners swapped, up and down exchanged, the
    # wrong layout for this w90 file) is a different SU(2) placement, and it
    # dies here rather than in a dedicated gate.  Say so, or the model gets
    # debugged instead of the declaration.
    spin_hint = ("" if spin is None else
                 f" A `spin=` declaration is in force ({spin.kind!r} spinor basis on "
                 f"{spin.norb} orbitals); a wrong pairing -- partners swapped, up and down "
                 f"exchanged, or the wrong layout for this wannier90 file -- fails exactly "
                 f"here, so check the declaration against the file before the model.")
    # WHAT THE READ-OFF COULD NOT USE, said beside the enumeration's verdict:
    # what it read and why that was not a representation, or why it read
    # nothing, which is the measurement to act on
    n_ph = len(phases)
    read_hint = ("" if fallback is None else
                 f" The enumeration ran because {fallback}. The candidates counted above "
                 f"try {n_ph} phase{'' if n_ph == 1 else 's'} on each free orbit.")

    if not covariant:
        # The detector's own declared tolerance is on its record.  When it lies
        # ABOVE this gate and the best candidate lands between the two, the
        # operations are *near*-symmetric -- obeyed within the declared
        # tolerance, not to machine precision -- and that is the symmetrizing
        # signal rather than a modelling error.  Stating it is factual: both
        # numbers were measured, and neither gate moves.
        near_hint = ""
        declared = float(getattr(validated, "tol", tol))
        if declared > tol and spectrum[0] <= declared:
            near_hint = (
                f" The operations were measured at a declared tol = {declared:.3g}, above "
                f"the {tol:.3g} the representation is held to, and the best candidate's "
                f"{spectrum[0]:.3g} lies between the two: these "
                f"operations are near-symmetric -- obeyed within what was declared and not "
                f"to machine precision, or on parts of the crystal the measurement reads "
                f"apart that the hoppings couple only below it -- as a numerical "
                f"(wannier90) Hamiltonian gives. The route is to project H onto its "
                f"group-invariant part and build the representation on the symmetrized "
                f"model: `pythtqc.symmetrize` (in module "
                f"`pythtqc.symmetrization`), or "
                f"`realized_group(..., tol={declared:.3g})`, whose default "
                f"`symmetrize=\"auto\"` does it for you.")
        return Refusal(
            "rep_not_covariant",
            f"none of the {len(records)} candidate representations ({len(distinct)} distinct) "
            f"carries H(k) onto itself within the tolerance: the best reaches "
            f"{spectrum[0]:.3g} against {tol:.3g}. Either the operation set is not realized "
            f"on this orbital content (a slab operation, or one the Hamiltonian breaks "
            f"though a matrix with its action on k carries H(k) onto itself) or the tensor "
            f"structure is too narrow for the multiplets it was given — "
            f"a site whose orbitals mix under its site-symmetry group is one multiplet, "
            f"which is what the measured multiplets read off the solution spaces and "
            f"what a `multiplets=` states.{read_hint}{near_hint}{spin_hint}",
            value=float(spectrum[0]), threshold=float(tol),
            context={"n_candidates": int(search["n_candidates"]), "multiplets": layout,
                     "indices": tuple(int(syms[p].ind) for p in kept),
                     **({} if spin is None else {"spin_kind": str(spin.kind)})})
    if not consistent:
        return Refusal(
            "rep_internal_factor",
            f"{len(covariant)} of the {len(distinct)} distinct candidates carry H(k) onto "
            f"itself within {tol:.3g}, but on none of them does a site block factorize into "
            f"(character) x (the group's own internal factor): the best defect is "
            f"{int_spectrum[0]:.3g} against {internal_tol:.3g}. These matrices commute with H "
            f"the way the group says, but they are not the matrices of *these* operations — "
            f"nothing here has a site character to be read off it, so no orbital could be "
            f"declared or labelled.{read_hint}{spin_hint}",
            value=float(int_spectrum[0]), threshold=float(internal_tol),
            context={"n_candidates": int(search["n_candidates"]), "multiplets": layout,
                     "indices": tuple(int(syms[p].ind) for p in kept),
                     **({} if spin is None else {"spin_kind": str(spin.kind)})})
    if not survivors:
        stuck = [grp[0]["refusal"] for grp in consistent if "refusal" in grp[0]]
        if len(stuck) == len(consistent):
            # every surviving candidate failed to phase-canonicalize at all;
            # normalize_rep's own diagnosis is sharper than anything said here.
            return replace(stuck[0], reason=stuck[0].reason + read_hint) if read_hint \
                else stuck[0]
        # WHY NO PHASE HELPS, where the matrices say so.  The candidate that came
        # closest carries, where nothing anchored its phases, whether its matrices
        # form a projective representation at all and whether any roots the
        # canonical norms admit made it multiply.  Projective, and none did: two
        # commuting operations whose matrices commute only up to a phase the
        # table does not carry are the gauge-free statement of it.
        closest = min((grp[0] for grp in consistent if "normalized" in grp[0]),
                      key=lambda r: float(r["product"]))
        found = closest.get("roots")
        pair = None
        if found is not None and found["projective"] and found["roots"] is None:
            # on the part whose roots did not multiply, where the parts were
            # settled one by one
            on = closest.get("roots_matrices", closest["normalized"][0])
            pair = _commuting_pair_phase(on, u_pos, syms, prod, spf)
        if pair is not None:
            a, b, c = pair
            from .groups import _record_name

            name_a = _record_name(syms[a], group.lattice)
            name_b = _record_name(syms[b], group.lattice)
            Sa = np.asarray(syms[a].spinor_rotation, dtype=complex)
            Sb = np.asarray(syms[b].spinor_rotation, dtype=complex)
            half, _res = _proportionality(Sa @ Sb,
                                          int(spf[a, b]) * int(spf[b, a]) * (Sb @ Sa))
            half = complex(half)
            spin_half = (not bool(group.spinor)) and abs(half - c) <= 1e-6
            asked = int(spf[a, b]) * int(spf[b, a])
            named = f"{group.name} #{group.number_str}"
            return Refusal(
                "rep_not_multiplicative",
                f"{len(consistent)} of the {len(distinct)} distinct candidates carry H(k) onto "
                f"itself within the tolerance and factorize on every site into (character) x "
                f"(the group's own internal factor), and their matrices form a projective "
                f"representation of {named}: every product D(g1)D(g2) is a phase times "
                f"D(g1 g2). No choice among the phases the canonical norms admit turns those "
                f"phases into the group's multiplication table -- the best product-table "
                f"defect is {prod_spectrum[0]:.3g} against {product_tol:.3g} -- because the "
                f"class is not the table's. Operations #{syms[a].ind} ({name_a}) and "
                f"#{syms[b].ind} ({name_b}) commute, the table asks "
                f"D(a)D(b) = {asked:+d} D(b)D(a)"
                + (" (spinor factors included)" if bool(group.spinor) else "")
                + f", and on these states D(a)D(b) = {_fmt_char(asked * c)} D(b)D(a). A phase "
                f"on either matrix cancels from that, so no rephasing removes it, and no "
                f"representation of {named} with its own factor system exists on these states."
                + (f" It is the sign the spin-1/2 lifts of these two operations carry: the "
                   f"states carry half-integer spin, and no single-valued representation of "
                   f"this group exists on them. A spinor basis stored in the orbital index -- a "
                   f"wannier90 import -- is read in the double group once its spin partners are "
                   f"declared (`spin=`)." if spin_half else "") + read_hint,
                value=float(prod_spectrum[0]), threshold=float(product_tol),
                context={"n_candidates": int(search["n_candidates"]), "multiplets": layout,
                         "indices": tuple(int(syms[p].ind) for p in kept),
                         "commuting_operations": (int(syms[a].ind), int(syms[b].ind)),
                         "commutation_phase": c,
                         "spin_half": bool(spin_half)})
        return Refusal(
            "rep_not_multiplicative",
            f"{len(consistent)} of the {len(distinct)} distinct candidates carry H(k) onto "
            f"itself within the tolerance and factorize on every site into (character) x "
            f"(the group's own internal factor), but none of them reproduces the group's "
            f"multiplication table (spinor factors included): the best product-table defect is "
            f"{prod_spectrum[0]:.3g} against {product_tol:.3g}. A Hamiltonian-covariant set of "
            f"matrices that does not multiply is not a representation of this group — the "
            f"operation set, the generators or the internal blocks are wrong, not the phases."
            f"{read_hint}",
            value=float(prod_spectrum[0]), threshold=float(product_tol),
            context={"n_candidates": int(search["n_candidates"]), "multiplets": layout,
                     "indices": tuple(int(syms[p].ind) for p in kept)})
    if len(survivors) > 1:
        branches = sorted({int(g[0]["orientation"]) for g in survivors})
        orient_hint = (
            "" if branches != [-1, 1] or len(survivors) != 2 else
            " The two are the two orientation branches, which is the one tie this "
            "function settles by convention when it can — but only when both return "
            "the same site labels under the same labelling, and these "
            "do not, so the choice is a genuine second representation rather than a "
            "gauge.")
        return Refusal(
            "rep_not_unique",
            f"{len(survivors)} distinct representations of {len(records)} candidates carry "
            f"H(k) onto itself within {tol:.3g}, factorize on every site into (character) x "
            f"(the group's own internal factor) within {internal_tol:.3g} and reproduce the "
            f"multiplication table within {product_tol:.3g}; the discrete choice is not "
            f"pinned. Either the "
            f"k-point grid the covariance is measured on is too small (it has "
            f"{k_pts.shape[0]} k-points) or the model carries more symmetry than the group does."
            f"{orient_hint}{read_hint} Open an issue on pythtqc's GitHub with the model "
            "and the call: the measurement should have fixed this choice.",
            value=float(len(survivors)), threshold=1.0,
            context={"n_candidates": int(search["n_candidates"])})
    winner = survivors[0][0]
    search["internal"] = float(winner["internal"])
    search["product"] = float(winner["product"])
    search["composition"] = float(winner.get("composition", 0.0))

    # ---- 7. the winner's normalization, already measured by the gate -------
    D, normalization, anchor_T, DT_n = winner["normalized"]

    # ---- 7b. the labelling, chosen once, and put in --------------------------
    # Everything above is blind to D(g) -> chi(g) D(g) on a coupled part, with
    # chi a one-dimensional character, and to the representation a conserved
    # pseudospin's copies carry: the hoppings fix how the orbitals transform
    # relative to one another, not what they are.  The multiplets' characters
    # are read once on the representation as settled
    # (:func:`_multiplet_characters`); one chooser takes each part's candidates
    # -- what its declarations leave -- to one labelling (:func:`_labelling`);
    # and it is put in here (:func:`_assembled`), so that the matrices returned
    # are the representation induced from the returned site content in the
    # tables' convention, spin up along +z in a right-handed frame.  The record
    # is then written off what was read and chosen (:func:`_atomic_limit_table`),
    # nothing measured again.
    limits = None
    if site_tables:
        raw, worst = _multiplet_characters(D, syms, u_pos, cperm, layout, prod)
        measured_at = {}
        for r in verdict["rows"]:
            measured_at.setdefault(tuple(int(o) for o in r["orbitals"]), set()).add(
                int(r["ind"]))
        for j in tuple(verdict.get("joint", ())) + tuple(verdict.get("by_sectors", ())):
            measured_at.setdefault(tuple(int(o) for o in j["decl"]["orbitals"]), set()).update(
                int(g) for row in j["theta"].values() for g in row)
        choice = _labelling(D, raw, decls, pool=pool, model=model, group=group,
                            assignment=assignment, layout=layout, params=params,
                            syms=syms, u_pos=u_pos, cperm=cperm, classes=classes,
                            parts=coupled_parts, ref_chan=ref_chan,
                            site_tables=site_tables, measured_at=measured_at,
                            multiplicity=measured_of.multiplicity)
        D = _assembled(D, choice, syms, layout)
        limits = _atomic_limit_table(choice["raw"], worst, decls, choice, group=group,
                                     assignment=assignment, model=model, layout=layout,
                                     syms=syms, kept=kept, checks=verdict["rows"],
                                     pool=pool, site_tables=site_tables, ident=ident,
                                     note=char_note, spin=spin, ref_chan=ref_chan)
        # A CONSERVED PSEUDOSPIN'S CHOICE, on the record: the site content
        # chosen for its copies, and every content the group's table admits
        # beside it, in the order the key ranks them.
        if choice["spaces"]:
            limits = _with_multiplicity_spaces(limits, choice["spaces"], group, assignment)

    # ---- 8. everything the object claims, re-measured ----------------------
    residuals, norm_res, anchor_res = {}, {}, {}
    for p in kept:
        ind = int(syms[p].ind)
        U = D[p]
        src = H.conj() if syms[p].time_reversal else H
        residuals[ind] = float(np.abs(U @ src @ U.conj().T - Hrot[p]).max())
        n, zeta = _order_and_zeta(group, p, prod=prod, spf=spf)
        norm_res[ind] = float(np.abs(_power(U, bool(syms[p].time_reversal), n)
                                     - zeta * np.eye(nstate)).max())
        if anchor_T is not None and not syms[p].time_reversal:
            # the time reversal returned, the labelling's own included
            Tn = D[T_pos] if T_pos in D else (DT_n if DT_n is not None else anchor_T)
            anchor_res[ind] = float(np.abs(Tn @ np.conj(U) @ Tn.conj().T - U).max())
    # THE REPRESENTATION RETURNED closes the gate its candidates closed: every
    # labelling it is chosen among is one the hoppings leave open, a symmetry of
    # H, so one that misses the gate is the package disagreeing with itself
    worst_ind = max(residuals, key=residuals.get) if residuals else None
    if worst_ind is not None and residuals[worst_ind] > tol:
        raise InternalError(
            f"the representation of {group.name} #{group.number_str} with its labelling in "
            f"it does not carry H(k) onto itself within the tolerance its candidates met: "
            f"{residuals[worst_ind]:.3g} against {tol:.3g}, on operation {worst_ind}. Every "
            "labelling chosen among is a "
            "representation the hoppings leave open, a symmetry of the Hamiltonian; one "
            "that is not is a defect in this package, not a tolerance to loosen.")

    product_residual = _product_table_residual(D, kept, syms, prod, spf)

    # The Bloch operators' factor system and each orbit's band representation
    # are what the representation RETURNED carries, measured once its matrices
    # are final.  A representation built as the operators of a projection is
    # never returned, and carries neither (`_returned=False`).
    factor_residual = float("nan")
    try:
        if _returned:
            factor_residual = 0.0
            per = list(group.periodic_dirs)
            for a in kept:
                for b in kept:
                    c = int(prod[a, b])
                    if c not in D:
                        continue
                    kb = kact[b](k_pts)
                    Ua = bloch_operator(model, group, assignment, syms[a], D[a], kb)
                    Ub = bloch_operator(model, group, assignment, syms[b], D[b], k_pts)
                    Uc = bloch_operator(model, group, assignment, syms[c], D[c], k_pts)
                    lhs = Ua @ (np.conj(Ub) if syms[a].time_reversal else Ub)
                    delta = (np.asarray(syms[a].translation)
                             + np.asarray(syms[a].rotation, dtype=float)
                             @ np.asarray(syms[b].translation)
                             - np.asarray(syms[c].translation))
                    omega = int(spf[a, b]) * np.exp(-2j * np.pi
                                                    * (kact[c](k_pts) @ delta[per]))
                    factor_residual = max(factor_residual, float(
                        np.abs(lhs - omega[:, None, None] * Uc).max()))
    except Exception:  # pragma: no cover - the check must never mask what was built
        factor_residual = float("nan")

    indices = tuple(int(syms[p].ind) for p in sorted(kept))
    choice = {"orientation": int(winner["orientation"]),
              "signs": {(int(syms[gens[gi]].ind), int(c)):
                        complex(winner["combo"][gi * len(free_cls) + j])
                        for gi in range(len(gens)) for j, c in enumerate(free_cls)}}
    built = SpaceGroupRep(
        model=model, group=group, assignment=assignment,
        indices=indices,
        _operations=tuple(syms[p] for p in sorted(kept)),
        matrices={int(syms[p].ind): D[p] for p in kept},
        antiunitary=tuple(int(syms[p].ind) for p in sorted(au_pos)),
        residuals=residuals, _norm_residuals=norm_res, _anchor_residuals=anchor_res,
        product_residual=float(product_residual), factor_residual=float(factor_residual),
        _generators=tuple(int(syms[p].ind) for p in gens),
        _words={int(syms[p].ind): w for p, w in words.items()},
        _choice=choice, _search=search, _classes=classes,
        _reference=dict({"orbital": ref_orb, "multiplet": ref_chan,
                         "orbit": int(classes[ref_chan])},
                        **({} if spin is None else
                           {"spin_pair": int(spin.pair_of_orbital[ref_orb])})),
        _failed=tuple(validated.discrepancies),
        _battery=k_pts, multiplets=layout, _normalization=normalization,
        _time_reversal=None if T_pos is None else int(syms[T_pos].ind),
        characters=limits, spin=spin, _branch_tie=bool(branch_tie))
    return _with_tables_unavailable(_with_band_representations(built) if _returned
                                    else built)


def _with_multiplicity_spaces(table, chosen, group, assignment):
    """``table`` with each conserved pseudospin's chosen content on it, and its rows marked.

    ``chosen`` is the chooser's record of each pseudospin (:func:`_labelling`):
    the site content returned at the orbit's lead multiplet, and every content
    the representations of its copies give there, in the order the key ranked
    them, each named against the same site table.  A row of such an orbit that
    nothing else left open reads ``lowest-l`` wherever the representations a
    declaration leaves gave more than one content: its site irrep rests on the
    choice.  A pseudospin whose copies a declaration speaks for is determined
    by it.
    """
    wyckoff = _wyckoff_sites(group, assignment)
    spaces, marked = [], set()
    for got in chosen:
        rec = got["record"]
        site = int(rec["site"])
        tab = table.site_tables.get(site)
        here = wyckoff.get(site)
        # how many Wannier functions of the copies one site holds: its orbitals,
        # or its declared spinor pairs
        on_site = set(int(o) for o in assignment.site_orbitals[site])
        held = sum(1 for o in rec["orbitals"] if int(o) in on_site)
        spaces.append(MultiplicitySpace(
            orbitals=tuple(int(o) for o in rec["orbitals"]),
            wyckoff="" if here is None else f"{int(here.multiplicity)}{here.letter}",
            dim=int(rec["dim"]), content=_named_content(tab, got["content"])[0],
            contents=tuple(_named_content(tab, other)[0] or "(unnamed)"
                           for other in got["contents"]),
            n_representations=int(rec["n_admitted"]),
            determined_by=("declaration" if got.get("declared") else
                           "lowest-l" if int(rec["n_admitted"]) > 1 else "hamiltonian"),
            _per_site=held // (2 if table.spin is not None else 1),
            _pairs=table.spin is not None))
        if int(got.get("open", len(got["contents"]))) > 1:
            marked |= {int(o) for o in rec["orbitals"]}
    rows = tuple(replace(o, determined_by="lowest-l")
                 if o.orbital in marked and o.determined_by == "hamiltonian" else o
                 for o in table.orbitals)
    return replace(table, orbitals=rows, multiplicity_spaces=tuple(spaces))


def _with_band_representations(rep):
    """``rep`` with each orbit's band representation on its atomic-limit table.

    That band representation is read off the representation itself — the
    content its states carry — so it is written once the matrices are
    (:func:`..srsi._band_representations`).
    """
    table = rep.characters
    if table is None:
        return rep
    from . import srsi as _srsi

    orbits = _srsi._band_representations(rep)
    if not orbits:
        return rep
    return replace(rep, characters=replace(table, band_representations=orbits))


def _with_tables_unavailable(rep):
    """``rep`` with every row saying the tables were unavailable, where they were.

    A group whose cell ``irrep`` could not match to the tables' standard
    setting is returned without tables (``GroupData._tables_ready``
    false on a primitive cell), so no row of it takes a name from them.  Each
    row's :attr:`AtomicLimitRow.note` then says so, and why, in the words of the
    group's own note, which carries ``irrep``'s message.
    """
    table, group = rep.characters, rep.group
    if (table is None or bool(group._tables_ready)
            or int(getattr(group, "_lattice_index", 1) or 1) > 1):
        return rep
    why = (f"no site irrep here is named from the tables: they are unavailable for "
           f"{group.name} #{group.number_str} -- "
           + (group._setup_note or "irrep could not match this cell to the BCS standard "
                                  "setting"))
    rows = tuple(replace(o, note=(o.note + "; " if o.note else "") + why)
                 for o in table.orbitals)
    return replace(rep, characters=replace(table, orbitals=rows))


#: What :func:`_build_rep` measures its covariance gate with when it is not told
#: otherwise -- its battery, and the operations it represents -- read off its
#: own signature rather than retyped.
_BUILD_DEFAULTS = {name: p.default
                   for name, p in inspect.signature(_build_rep).parameters.items()
                   if p.default is not inspect.Parameter.empty}

#: Every keyword :func:`_build_rep` takes, read off its own signature once, so that
#: the words handed to the build by name are checked against the build itself.
_BUILD_KEYWORDS = frozenset(name for name, p in inspect.signature(_build_rep).parameters.items()
                            if p.kind is p.KEYWORD_ONLY)


def _exact_gate(model, group, validated, build_kw=None, params=None) -> float:
    r"""The covariance gate :func:`_build_rep` applies to this model and operation set.

    The same :func:`covariance_gate`, measured the way the build measures it:
    ``|G|`` the operations it represents, ``N`` the model's states and
    ``max|H|`` on its battery.  ``build_kw`` are the build's own keywords, of
    which ``operations=``, ``include_vacuum_ops=``, ``k_pts=``, ``n_battery=``
    and ``seed=`` bear on the gate, and ``params`` the model's parameter values.
    A detection record carries neither ``N`` nor ``max|H|``, so
    :func:`..groups.realized_group` and :func:`..symmetrization.symmetrize`
    measure the gate here and classify operations against it.
    """
    kw = dict(build_kw or {})

    def _given(name):
        return kw[name] if name in kw else _BUILD_DEFAULTS[name]

    syms = {int(s.ind): s for s in group.symmetries}
    ops = _given("operations")
    indices = [int(i) for i in (validated.indices if ops is None else ops)]
    padded = bool(_given("include_vacuum_ops"))
    n_operations = sum(1 for i in indices
                       if padded or not _moves_padded(syms[i], group.vacuum_dirs))
    k_pts = _covariance_battery(group, _given("k_pts"), _given("n_battery"), _given("seed"))
    H = np.asarray(_bloch.hamiltonian(model, k_pts, params))
    return covariance_gate(n_operations, int(model.nstate), float(np.abs(H).max()))
