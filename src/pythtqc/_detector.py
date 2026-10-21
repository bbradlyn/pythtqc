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
from ._tolerance import DEFAULT_ENERGY, _EIGEN_WINDOW_CEIL, _VACUOUS_CUT, eigen_window


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
        ``H``: :func:`.._measure._validate_group` gates its covariance residuals
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
        :func:`.._measure._validate_group` and :func:`..groups.realized_group`
        pass their ``max_bytes`` here.
    hamiltonians : _GridHamiltonians, optional
        A shared evaluator for ``H`` on **this** ``k_pts``.  ``H(k)`` does not
        depend on which operation is being tested, so passing one instance to
        every operation of a group evaluates the grid once;
        :func:`.._measure._validate_group` does exactly that.
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
        refusal is handed back as it is, :func:`.._measure._validate_group` and
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
      unitary eigen-projections; :func:`.._matrices.normalize_rep` owns it.

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
