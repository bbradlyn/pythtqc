"""The eigen-sift detector: every matrix ``X`` with ``X H(k) = H(Rk) X`` on a grid
of ``k``, found at one point and sifted over the rest, with the covariance of
each candidate over the whole grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import _bloch
from .refusal import Refusal
from ._tolerance import (
    ANSATZ_GATE, DEFAULT_ENERGY, _EIGEN_WINDOW_CEIL, _IN_BLOCKS, _VACUOUS_CUT, eigen_window)


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
#: is (:attr:`_IntertwinerSpace.residuals`).
_SIFT_STABLE = 3

#: Factor by which an emptied sift widens its window before trying again.
_EIGEN_ESCALATION = 10.0

#: Seeded generic combinations of the surviving basis that the sift streams over
#: the **whole** grid before it stops (see :func:`_grid_probe`).  Two, because
#: one already fails at a point where the basis does not intertwine with
#: probability one and the second costs one pass to make that "with probability
#: one" not matter.
_GRID_PROBES = 2

#: Byte ceiling on the tensor-structure projection :func:`.._build._build_rep`'s read-off runs on
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
        scale`` is the "how far from a solution" number to read.  Where the
        space is the fit over the whole grid (:attr:`fit` ``"fitted"``),
        ``singular_values`` holds each member's largest covariance over the
        grid, ``‖X H(k) − H(Rk) X‖_F`` at unit norm, ascending, and ``scale``
        the largest ``‖L_k‖``: the members were counted on those, point by
        point.  When
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
    fit : str
        What the fit over the whole grid among the matrices of the operation's
        forms (``forms`` of :func:`_find_intertwiners`) settled, where the sift
        kept no candidate within ``tol`` or its space lies off a form
        (:func:`_off_a_form`): ``"fitted"``, the space is the fit's;
        ``"fit not closer"``, the fit's best candidate was not closer and the
        sift's space is returned; ``"spectra"``, ``"none at the points
        sifted"`` and ``"none over the grid"``, a proof that no matrix of the
        forms meets ``tol`` -- two spectra apart by more than ``n·tol`` at a
        point sifted, the stacked constraint at the points sifted, the count
        over the grid; ``"not fitted: memory"``, a form's fit would not fit
        the memory the basis may take and no proof was made.  ``""`` where
        nothing was asked: no forms given, a candidate within ``tol`` and of
        every form it is near to rounding, a vacuous space or the stacked
        route.
    fitted_forms : tuple
        The forms fitted where the sift's candidate met ``tol`` and its space lay
        off them (:func:`_off_a_form`): the space depends on the forms given
        only through these, so it answers for any set of forms holding them.
        Empty otherwise.
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
    fit: str = ""
    fitted_forms: tuple = ()

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
    max_bytes : float, optional
        Budget for one evaluation, in bytes.  Default ``256e6``.  A grid whose
        Hamiltonians fit inside it is evaluated **in one call** (bit-identical
        to ``model.hamiltonian`` on the same points) and cached whole; a larger
        one is streamed.
    bloch : optional
        The model's Bloch sum, where another grid of the same model already
        holds it (:meth:`rotated` hands its own on).
        Looked up at the first evaluation otherwise.

    Attributes
    ----------
    n_evaluated : int
        How many k-points have been evaluated — the number the sharing is
        meant to reduce, so it is reported rather than assumed.
    """

    def __init__(self, model, k_pts, max_bytes=256e6, *, bloch=None):
        self.model = model
        self.k_pts = np.atleast_2d(np.asarray(k_pts, dtype=float))
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
        """The model's Bloch sum (:func:`.._bloch.evaluator`)."""
        if self._bloch_sum is None:
            self._bloch_sum = _bloch.evaluator(self.model)
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
        return _GridHamiltonians(self.model, k_rot, self.max_bytes,
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
    mu = _bloch.onsite_mean(pair.src.model)

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
    # the most two spectra differ at one level, over the points sifted: no
    # unitary carries one onto the other closer, in the spectral norm
    spectral = float(np.abs(Ep - E).max())
    points = [i1]
    s_last, scale_last = None, h
    if m == 0:
        # the spectra do not match at all: no X, unitary or not, intertwines.
        # The single-point constraint's own singular values are the gaps.
        return {"flat": False, "k_solve": i1, "n_solve": 1, "h": h, "basis": B,
                "trace": tuple(trace), "window": delta_used, "gaps": gaps,
                "spectral": spectral, "points": tuple(points),
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
        points.append(i)
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
        spectral = max(spectral, float(np.abs(Er - Es).max()))
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
                        "scale": op, "spectral": spectral, "points": tuple(points),
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
            "spectral": spectral, "points": tuple(points), "empty_ratio": None}


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


def _refit_on_form(pair, form, n, *, cut, floor):
    r"""The matrices of a form that carry ``H(k)`` onto ``H(Rk)``, fitted over the whole grid.

    ``form`` is the boolean ``(n, n)`` pattern of the entries a matrix of the
    operation may hold.  The sift fixes its basis at a few points, and a
    direction it discards at one point -- a singular value small there but not
    zero, or a pair of levels outside the window -- takes with it the part
    of a nearly exact solution that lies along it, the solution's own residual
    at that point divided by that singular value or gap; at the other points
    that part costs its full size.  Here nothing is discarded: the members of
    the form minimizing ``Σ_k ‖X H(k) − H(Rk) X‖²_F`` are the eigenvectors of
    lowest eigenvalue of the Hermitian matrix that sum is, of size ``p``, the
    number of entries the form allows.  For entries ``(a, b)`` and ``(c, d)``

    .. math::

        G_{ab,cd} = \sum_k \delta_{ac} (H_k^2)_{db} + \delta_{bd} (H'^2_k)_{ac}
                    - 2 H'_{k,ac} H_{k,db},

    ``H_k`` the source (conjugated for an antiunitary operation) and ``H'_k`` its
    image, accumulated over the grid's chunks block by block of the form
    (:func:`_cross_term`), at a cost of ``N_k p²`` and the eigenvectors at
    ``p³``.

    The eigenvalues of ``G`` are squares, and the smallest are rounding where
    the cut is read, so they count nothing.  The members are counted as the
    sift counts them, on each point's own constraint: in ascending order, a
    member is one where at every point its covariance ``‖X H(k) − H(Rk) X‖_F``,
    ``X`` of unit norm, is within the sift's cut, ``max(cut·‖L_k‖, sqrt(n)·floor)``,
    and the count stops at the first that is not.  Returns the members with
    unit norm, shape ``(count, n, n)``, each member's largest covariance over
    the grid, and the largest ``‖L_k‖``.
    """
    rows, cols, blocks = _form_blocks(form)
    p = int(rows.size)
    G = np.zeros((p, p), dtype=complex)
    sq, sq_rot = np.zeros((n, n), dtype=complex), np.zeros((n, n), dtype=complex)
    for _, _, Hs, Hr in pair.chunks():
        sq += np.einsum("kij,kjl->il", Hs, Hs)
        sq_rot += np.einsum("kij,kjl->il", Hr, Hr)
        G -= 2.0 * _cross_term(Hs, Hr, blocks, p)
    G += (rows[:, None] == rows[None, :]) * sq[cols[None, :], cols[:, None]]
    G += (cols[:, None] == cols[None, :]) * sq_rot[rows[:, None], rows[None, :]]
    _w, v = np.linalg.eigh((G + G.conj().T) / 2)
    least = float(np.sqrt(n)) * float(floor)
    gates, scale = [], 0.0           # the sift's cut at each point, chunk by chunk
    for _, _, Hs, Hr in pair.chunks():
        Es, Er = np.linalg.eigvalsh(Hs), np.linalg.eigvalsh(Hr)
        op = np.maximum(np.abs(Er[:, -1] - Es[:, 0]), np.abs(Es[:, -1] - Er[:, 0]))
        gates.append(np.maximum(float(cut) * op, least))
        scale = max(scale, float(op.max()))
    members, worst = [], []
    for j in range(p):
        X = np.zeros((n, n), dtype=complex)
        X[rows, cols] = v[:, j]
        held, most = True, 0.0
        for (_, _, Hs, Hr), gate in zip(pair.chunks(), gates):
            got = np.linalg.norm(X @ Hs - Hr @ X, axis=(1, 2))
            most = max(most, float(got.max()))
            held = held and bool(np.all(got <= gate))
        if not held:
            break
        members.append(X)
        worst.append(most)
    return (np.asarray(members, dtype=complex).reshape(len(members), n, n),
            np.asarray(worst, dtype=float), scale)


def _form_blocks(form):
    """A form's entries, block by block: ``rows``, ``cols`` and each block's ``(rows, cols, start)``.

    A form carries each site's states onto its image's, so its entries fall in
    rectangular blocks, one per site: the columns of one site share the rows
    they may reach.  The entries are listed block by block, each block's in
    row-major order.
    """
    form = np.asarray(form, dtype=bool)
    reach = {}
    for b in range(form.shape[1]):
        to = tuple(int(a) for a in np.nonzero(form[:, b])[0])
        if to:
            reach.setdefault(to, []).append(b)
    rows, cols, blocks, start = [], [], [], 0
    for to, of in reach.items():
        r, c = np.asarray(to, dtype=int), np.asarray(of, dtype=int)
        blocks.append((r, c, start))
        rows.append(np.repeat(r, len(c)))
        cols.append(np.tile(c, len(r)))
        start += len(r) * len(c)
    return np.concatenate(rows), np.concatenate(cols), blocks


def _cross_term(Hs, Hr, blocks, p):
    r"""``Σ_k H'_{k,ac} H_{k,db}`` for every pair of entries ``(a, b)``, ``(c, d)`` of a form.

    Block by block one matrix product over the points: for the blocks
    ``(R_I, C_I)`` and ``(R_J, C_J)``, ``H'[R_I, R_J]`` and ``H[C_J, C_I]``,
    each flattened over the points, multiplied.
    """
    nk = int(Hs.shape[0])
    out = np.zeros((p, p), dtype=complex)
    for rI, cI, sI in blocks:
        for rJ, cJ, sJ in blocks:
            A = Hr[:, rI[:, None], rJ[None, :]].reshape(nk, -1)
            B = Hs[:, cJ[:, None], cI[None, :]].reshape(nk, -1)
            T = (A.T @ B).reshape(len(rI), len(rJ), len(cJ), len(cI))
            out[sI:sI + len(rI) * len(cI), sJ:sJ + len(rJ) * len(cJ)] = (
                T.transpose(0, 3, 1, 2).reshape(len(rI) * len(cI), len(rJ) * len(cJ)))
    return out


def _sifted_constraints(pair, points, n, *, cut, floor):
    """``(H_src(k), H(Rk), g_k)`` at each point given, ``g_k`` the sift's cut there."""
    least = float(np.sqrt(n)) * float(floor)
    out = []
    for i in points:
        Hs, Hr = pair.at(int(i))
        Es, Er = np.linalg.eigvalsh(Hs), np.linalg.eigvalsh(Hr)
        op = float(max(abs(Er[-1] - Es[0]), abs(Es[-1] - Er[0])))
        out.append((Hs, Hr, max(float(cut) * op, least)))
    return out


def _form_admits(constraints, form):
    r"""Whether a matrix of a form can meet the sift's cut at every one of the points given.

    A unitary ``U`` within the covariance gate at a point has
    ``‖U H(k) − H(Rk) U‖_F = ‖U H(k) U† − H(Rk)‖_F ≤ n·tol``, so ``U/sqrt(n)``,
    of unit norm, is within ``sqrt(n)·tol`` there, which the sift's cut
    ``g_k = max(cut·‖L_k‖, sqrt(n)·floor)`` never undercuts.  Such a matrix
    of the form has ``x† G x ≤ Σ_k g_k²``, ``G`` the Hermitian matrix of
    :func:`_refit_on_form` summed over the points alone, so a lowest
    eigenvalue of ``G`` above that bound, by more than the rounding of the
    eigenvalues, ``p·eps·‖G‖``, is a proof that no matrix of the form meets the
    gate at those points, nor on any grid that holds them.  It costs ``p²`` a
    point for the ``p`` entries the form allows, and ``p³`` once.
    ``constraints`` is :func:`_sifted_constraints`' list.
    """
    rows, cols, blocks = _form_blocks(form)
    p = int(rows.size)
    bound = float(sum(gate ** 2 for _, _, gate in constraints))
    Hs = np.stack([c[0] for c in constraints])
    Hr = np.stack([c[1] for c in constraints])
    G = -2.0 * _cross_term(Hs, Hr, blocks, p)
    G += (rows[:, None] == rows[None, :]) * np.einsum(
        "kij,kjl->il", Hs, Hs)[cols[None, :], cols[:, None]]
    G += (cols[:, None] == cols[None, :]) * np.einsum(
        "kij,kjl->il", Hr, Hr)[rows[:, None], rows[None, :]]
    w = np.linalg.eigvalsh((G + G.conj().T) / 2)
    rounding = p * np.finfo(float).eps * float(np.abs(w).max())
    return float(w[0]) - rounding <= bound


def _off_a_form(basis, forms, n, *, cut):
    r"""The forms the space lies off by more than rounding while near them.

    Parameters
    ----------
    basis : ndarray, shape (dim, n, n)
        The sift's orthonormal basis, each member of norm ``sqrt(n)``.
    forms : sequence of ndarray of bool, shape (n, n)
        The entries a matrix of each operation sharing this action on ``k`` may
        hold.
    n : int
        The number of states.
    cut : float or None
        The relative cut the solve was asked at.

    Returns
    -------
    list of ndarray
        The forms for which the member of the space nearest it holds outside
        it more than :data:`.._tolerance._IN_BLOCKS` of the space's scale (or
        ``cut``, where larger), the floor every reading of an operation's site
        blocks takes as rounding, and no more than
        :data:`.._tolerance.ANSATZ_GATE` of its own norm (or ``cut``), within
        which the member is read as of the form.  A member that far off its
        form is a matrix of the operation the multiplets cannot be read off,
        though a matrix of the form may meet the gate; a space with nothing near
        a form holds another operation's matrices.
    """
    rounding = max(_IN_BLOCKS, float(cut or 0.0))
    near = max(ANSATZ_GATE, float(cut or 0.0))
    flat = np.asarray(basis, dtype=complex).reshape(len(basis), n, n)
    off = []
    for form in forms:
        outside = flat[:, ~np.asarray(form, dtype=bool)]
        if not outside.size:
            continue
        sv = np.linalg.svd(outside, compute_uv=False)
        least = float(sv[-1]) if len(sv) == len(flat) else 0.0
        if rounding * max(float(sv[0]), 1.0) < least <= near * float(np.sqrt(n)):
            off.append(form)
    return off


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
                       max_solve_entries=2_000_000, forms=(),
                       _floor=None) -> _IntertwinerSpace | Refusal:
    r"""Every matrix ``X`` with ``X H(k) = H(Rk) X`` at every point of a grid.

    For an antiunitary operation the equation is ``X H(k)* = H(Rk) X``.
    Nothing is assumed about how the operation acts on the orbitals.  The
    solutions are fixed up to a factor and, at ``dim > 1``, a change of basis.

    Parameters
    ----------
    model : pythtb.TBModel
        The model; the matrices act on its ``norb * nspin`` states, spin the
        fast index.
    k_action : callable or ndarray, shape (dim_k, dim_k)
        ``k -> Rk`` on reduced ``k``, as
        :meth:`.._spacegroup.GroupData._k_action_of` gives it; a matrix ``M``
        is applied as ``k @ M.T``.  The image must not be wrapped into the
        first zone.  For an antiunitary operation the action includes the
        sign of time reversal.
    k_pts : ndarray, shape (Nk, dim_k)
        The grid, at least two points.
    antiunitary : bool, default False
        Whether the antiunitary equation is solved.
    tol : float, default 1e-8
        The covariance a candidate may carry, in the units of ``H``.
    cut : float, default None
        The relative nullspace cut.  A singular value counts as zero at or
        below ``cut`` times the constraint's norm at a point, or
        ``sqrt(nstate)·_floor``; on ``solver="stacked"`` the norm is the
        stacked operator's and the floor ``sqrt(nstate·n_solve)·_floor``.
        ``None`` is ``tol``.
    solver : {"eigen", "stacked"}, default "eigen"
        The route: the eigen route parametrizes the space at one point and
        sifts it over the others; ``"stacked"`` solves one SVD over the whole
        grid, affordable only at small ``nstate``.
    window : float, default None
        The eigen route's starting energy window, relative to ``max|H − μ·1|``
        at its parametrization point, ``μ`` the mean on-site energy.  It is
        never narrower than ``_floor`` in the units of ``H``.  ``None``
        derives it from ``tol``.
    max_window : float, default 0.3
        The widest window an emptied sift widens to, tenfold at a time.
    max_bytes : float, default 512e6
        The memory the eigen route's trial basis may take at one point,
        ``16·nstate²`` bytes a pair of eigenvectors.  Past it the window is
        narrowed to the nearest pairs.
    degenerate_max_bytes : float, default None
        The budget the solve runs again with where the pairs no window drops
        are past ``max_bytes``.  ``None``, or a smaller value, is
        ``max_bytes``.
    stable : int, default 3
        How many sifted points in a row leave the basis unchanged before the
        sift stops.
    hamiltonians : _GridHamiltonians, default None
        ``H`` evaluated on these ``k_pts``, shared by every operation of a
        group.  ``None`` evaluates it here.
    max_solve_entries : int, default 2_000_000
        ``solver="stacked"`` only.  A constraint past this many entries is
        solved on a subset of the grid and measured on all of it.
    forms : sequence of ndarray of bool, shape (nstate, nstate), default ()
        The entries a matrix of each operation with this action on ``k`` may
        hold.  Read on ``solver="eigen"`` only.
    _floor : float, default None
        The covariance at one point, in the units of ``H``, that a candidate
        the nullspace keeps may carry.  ``None`` is ``cut``.

    Returns
    -------
    _IntertwinerSpace or Refusal
        The space, falsy at ``dim == 0``, which means that no matrix solves the
        equation within the cut at the points sifted, among the eigenvector
        pairs the window admits.  Each candidate's covariance over the whole
        grid is in ``residuals``.  A vacuous constraint returns the whole
        matrix space, the identity first.  With ``forms``, each form is fitted
        over the grid where the sift keeps candidates none of which meets
        ``tol``, or keeps none while the spectra at its points do not exclude
        a matrix within ``tol``.  The members of every form whose fit meets
        ``tol`` are returned together; where none does, the closest fit is
        returned if the sift kept nothing or the fit is closer.  Where the
        sift's candidate meets ``tol`` but its space lies off a form and near
        it, the forms it lies off are fitted, and their members are returned
        where they meet ``tol``, the sift's space otherwise.  ``fit`` names
        the outcome; ``"not fitted: memory"`` leaves the sift's space, and a
        ``dim == 0`` returned with it is no proof.

    Refusals
    --------
    ``detector_too_large``
        the pairs at a point closer than rounding or ``_floor``, which no
        window drops, need more memory than the larger of ``max_bytes`` and
        ``degenerate_max_bytes`` allows.

    See Also
    --------
    _measure._validate_group : the same question for every operation of a
        group, each verdict read on these residuals.
    _matrices.normalize_rep : the phase of an operation's matrix.
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
        src_grid = _GridHamiltonians(model, k_pts)
    else:
        src_grid = hamiltonians
        if src_grid.n_kpts != nk or not np.array_equal(src_grid.k_pts, k_pts):
            raise ValueError("hamiltonians= was built on a different k-grid than the "
                             "k_pts supplied here; a shared cache is keyed by its grid")
    rot_grid = src_grid.rotated(k_rot)

    window_used, k_solve, trace, escalated, spectral, sifted = 0.0, -1, (), (), 0.0, ()
    fit, astray = "", []

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
            spectral, sifted = float(rec["spectral"]), tuple(rec["points"])
            break

        cands, search = _candidates_and_search(basis, dim, n)
        unitaries, residuals, unitarity = _candidate_measures(cands, n, pair=pair)
        # Where the sift kept no matrix within the gate, the space fitted over
        # the whole grid among the matrices of one of the operation's forms:
        # the sift's basis is fixed at a few points, and what it discarded there
        # carries a nearly exact solution's residual at those points onto the
        # others, magnified by the inverse of a small gap.  Each step decides
        # only where the one before cannot: two spectra at a point sifted apart
        # by more than `n·tol`, the gate being on the largest entry, prove that
        # no unitary meets it; the stacked constraint at the points sifted
        # proves the same for a form, at the cost of those few points; and only
        # a form that can meet the gate there is fitted over the grid.
        missed = (not dim and spectral <= n * float(tol)) or (
            dim and float(residuals.min()) > float(tol))
        # ... and where the sift's candidate meets the gate while its space lies
        # off one of the forms by more than rounding, near it: the matrices of
        # that form are fitted the same way, and an operation is read by a
        # matrix of its own form wherever one meets the gate
        astray = (_off_a_form(basis, forms, n, cut=cut)
                  if forms and not vacuous and dim and not missed else [])
        if forms and not vacuous and not dim and not missed:
            fit = "spectra"
        elif forms and not vacuous and (missed or astray):
            outcomes, best = set(), None
            # The proofs below read the sift's cut never below sqrt(n)·tol, the
            # most a unitary within the gate can miss a point by at unit norm
            # (‖U H − H'U‖_F ≤ n·tol): with the floor raised to the gate wherever
            # a smaller one is given, a "none" is a proof whatever cut and floor
            # the solve was asked at.
            proof_floor = max(float(floor), float(tol))
            constraints = _sifted_constraints(pair, sifted, n, cut=cut, floor=proof_floor)
            # every operation's own form is fitted: two operations sharing
            # this action on k can both hold, each by a matrix of its own form,
            # and each is broken only where no matrix of its form meets the gate
            held = []
            for form in (astray or forms):
                if 16 * int(np.count_nonzero(form)) ** 2 > ceiling:
                    outcomes.add("not fitted: memory")
                    continue
                if not _form_admits(constraints, form):
                    outcomes.add("none at the points sifted")
                    continue
                members, worst, op_max = _refit_on_form(pair, form, n, cut=cut,
                                                        floor=proof_floor)
                if not len(members):
                    outcomes.add("none over the grid")
                    continue
                members = _orthonormal_basis(members, n)
                got_cands, got_search = _candidates_and_search(members, len(members), n)
                got = _candidate_measures(got_cands, n, pair=pair)
                if float(got[1].min()) <= float(tol):
                    held.append((members, worst, op_max))
                if best is None or float(got[1].min()) < float(best[4][1].min()):
                    best = (members, worst, op_max, (got_cands, got_search), got)
            if held:
                # the members of every form that meets the gate, one space
                members = _orthonormal_basis(np.concatenate([h[0] for h in held]), n)
                dim, basis = len(members), members
                cands, search = _candidates_and_search(basis, dim, n)
                unitaries, residuals, unitarity = _candidate_measures(cands, n, pair=pair)
                s = np.sort(np.concatenate([h[1] for h in held]))
                scale = max(float(h[2]) for h in held)
                fit = "fitted"
            elif best is not None and not astray and (not dim or float(best[4][1].min())
                                                      < float(residuals.min())):
                members, worst, op_max, (cands, search), got = best
                dim, basis = len(members), members
                unitaries, residuals, unitarity = got
                s, scale = np.sort(worst), float(op_max)
                fit = "fitted"
            elif best is not None:
                fit = "fit not closer"
            elif astray:
                # no form fitted meets the gate: the sift's candidate, which
                # does, stands
                fit = "fit not closer"
            else:
                fit = next(o for o in ("not fitted: memory", "none over the grid",
                                       "none at the points sifted") if o in outcomes)

    return _IntertwinerSpace(dim=dim, basis=basis, unitaries=unitaries,
                             residuals=residuals, unitarity=unitarity,
                             singular_values=s, scale=scale, tol=tol, cut=cut,
                             method=method, n_kpts=nk, n_kpts_solve=nk_solve,
                             antiunitary=bool(antiunitary), vacuous=vacuous,
                             search=search, window=window_used, k_solve=k_solve,
                             basis_size=tuple(int(t) for t in trace),
                             escalations=escalated, fit=fit,
                             fitted_forms=tuple(astray) if astray else ())
