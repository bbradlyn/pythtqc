"""The matrices an operation could be on a Hamiltonian.

Every matrix ``X`` with ``X H(k) = H(Rk) X`` on a grid of ``k`` is found, by
default, at one point and sifted over the others, and each candidate's
covariance is measured over the whole grid.
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
#: Sixteen is far more than the generic case needs, since a random combination
#: of a ``*``-algebra basis is nonsingular with probability one.  It is fixed so
#: that the search is reproducible without a parameter to thread.
_UNITARY_PROBES = 16

#: Seed of that search.  Fixed, so two calls on the same model return the same
#: matrices bit for bit.
_UNITARY_PROBE_SEED = 1729


#: How many k-points at the head of the grid the eigen route looks at when it
#: picks its **parametrization point** ``k1`` (the one whose source spectrum has
#: the largest spread, so that the fewest eigenvalue pairs match by accident).
#: Four is enough because the choice only has to avoid a *degenerate* point.
#: The point picked sets the size of the stage-0 basis, and so the cost.
_EIGEN_PROBES = 4


#: Consecutive sift stages that must leave the basis size unchanged before the
#: sift stops.  Three is a stopping rule: before it stops, the sift checks the
#: basis at every point (:func:`_grid_probe`).
_SIFT_STABLE = 3

#: Factor by which an emptied sift widens its window before trying again.
_EIGEN_ESCALATION = 10.0

#: Seeded generic combinations of the surviving basis that the sift streams over
#: the **whole** grid before it stops (see :func:`_grid_probe`).  Two, because
#: one already fails at a point where the basis does not intertwine with
#: probability one and the second costs one pass to make that "with probability
#: one" not matter.
_GRID_PROBES = 2

#: Byte ceiling on projecting one operation's solution space onto its site
#: blocks, ``16·dim·nstate²`` bytes for a space of dimension ``dim``: the
#: representation's build, the declaration, the descent, the multiplets, the
#: sectors and the site spaces each skip the projection past it rather than
#: allocate gigabytes.  It is meant for a **vacuous** space, whose ``dim`` is
#: ``nstate²``: a scalar Hamiltonian.
_READOFF_MAX_BYTES = 2.5e8


@dataclass(frozen=True)
class _IntertwinerSpace:
    """The matrices an operation could be on a Hamiltonian, with their covariance.

    The record :func:`_find_intertwiners` returns.  It is falsy at dimension
    zero, as a :class:`..refusal.Refusal` is, so ``isinstance(out, Refusal)``
    tells a refusal apart.  ``dim == 0`` means that no matrix solves the
    equation within the cut at the points sifted, among the eigenvector pairs
    the window admits, or at the points solved on the stacked routes.  At
    ``dim == 0`` the operation is read as broken.  Otherwise it is judged on
    :attr:`residuals`, and a nonzero dimension alone does not mean it holds.

    Attributes
    ----------
    dim : int
        The dimension of the space.  Where the operation holds and nothing was
        fitted, it is the dimension of the space of matrices that commute with
        every ``H(k)``.  Where :attr:`fit` is ``"fitted"``, it is the
        dimension of the solutions among the matrices of the forms fitted,
        which can be smaller.
    basis : ndarray, shape (dim, nstate, nstate)
        A basis of the space, orthogonal in the Frobenius inner product, each
        member of norm ``sqrt(nstate)``, in no particular phase.  At the
        identity it spans the model's conserved operators.  Where
        :attr:`vacuous`, it spans every matrix and ``basis[0]`` is the
        identity.
    unitaries : ndarray, shape (n_candidates, nstate, nstate)
        The candidates: the polar factor of each member of :attr:`basis`, and,
        at ``dim >= 2``, one more, a unitary member of the space found by the
        search :attr:`search` describes.  ``n_candidates`` is ``dim`` for
        ``dim <= 1`` and ``dim + 1`` otherwise.
    residuals : ndarray, shape (n_candidates,)
        ``max_k ‖U H(k) U† − H(Rk)‖_max`` of each candidate over every point
        supplied, in the units of ``H``; ``H(k)*`` in place of ``H(k)`` for an
        antiunitary operation.
    unitarity : ndarray, shape (n_candidates,)
        ``‖X†X − 1‖_max`` of each candidate before its polar factor is taken,
        ``X`` of Frobenius norm ``sqrt(nstate)``.
    singular_values : ndarray
        Ascending.  On the eigen route, the singular values of the constraint
        on the basis entering the last point sifted that constrains anything;
        zeros where no point does; every distance between an energy of
        ``H(Rk)`` and one of ``H(k)`` at the parametrization point where no
        pair falls within the window; and the constraint's norm at each point
        of the grid where :attr:`vacuous`.  On the stacked routes, the
        singular values of the stacked constraint on :attr:`n_kpts_solve`
        points.  Where :attr:`fit` is ``"fitted"``, each member's largest
        covariance ``‖X H(k) − H(Rk) X‖_F`` over the grid, at unit norm.
    scale : float
        What the cut is relative to: the constraint's norm at the last point
        sifted that constrains anything, or the stacked constraint's largest
        singular value.  Where :attr:`fit` is ``"fitted"``, the largest such
        norm over the grid.  ``max|H|`` at the parametrization point where no
        point sifted constrains anything, and over the grid where
        :attr:`vacuous`.
    tol : float
        The covariance a candidate may carry, in the units of ``H``.
    cut : float
        The relative cut applied to the singular values.  Equal to :attr:`tol`
        unless the two were declared apart.
    method : str
        The route that solved it: ``"eigen"``, ``"stacked"``, or
        ``"stacked_subset"``, the stacked solve on an evenly strided subset of
        the grid.
    n_kpts : int
        The number of points supplied, where the residuals are measured.
    n_kpts_solve : int
        The number of points the space was solved on.
    window : float
        The eigen route's energy window, relative to ``max|H − μ·1|`` at the
        parametrization point, ``μ`` the mean on-site energy.  The first basis
        is spanned by the outer products of the pairs of eigenvectors, one of
        ``H(k)`` and one of ``H(Rk)``, whose energies differ by at most the
        window times ``max|H − μ·1|``.  It is never narrower, in the units
        of ``H``, than the covariance a kept candidate may carry.  ``0.0``
        for the stacked routes.
    k_solve : int
        The index in the grid of the eigen route's parametrization point;
        ``-1`` for the stacked routes.
    basis_size : tuple of int
        The eigen route's basis size after each point sifted, the
        parametrization point first; ``()`` for the stacked routes.
    escalations : tuple of float
        The windows the eigen route tried and widened, in order, before
        :attr:`window`.
    antiunitary : bool
        Whether the antiunitary equation ``X H(k)* = H(Rk) X`` was solved.
    vacuous : bool
        Whether the constraint vanished: ``H(k)`` and ``H(Rk)`` are the same
        multiple of the identity at every point, to within ``1e-9`` of
        ``max|H|`` over the grid, so every ``X``
        solves it and ``dim`` is ``nstate**2``.  The operation is still judged
        on :attr:`residuals`.
    search : dict
        Empty unless ``dim >= 2``.  Then ``n_probes`` and ``seed`` of the
        seeded search over combinations of the basis; ``picked``, the winner's
        index in the pool searched, below ``dim`` where a basis member won;
        ``sigma_min``, the winner's least singular value; and
        ``sigma_min_basis``, the largest least singular value of a basis
        member.
    fit : str
        What the fit over the whole grid among the matrices of the operation's
        forms settled (``forms`` of :func:`_find_intertwiners`).
        ``"fitted"``: the space is the fit's.  ``"fit not closer"``: the
        sift's space is returned.  ``"spectra"``, ``"none at the points
        sifted"`` and ``"none over the grid"``: no matrix of the forms meets
        ``tol``, proved by two spectra at a point sifted apart by more than
        ``n·tol``, by the forms' constraint at the points sifted, or by the
        count over the grid.  ``"not fitted: memory"``: a form's fit would
        take more memory than the trial basis may, and nothing is proved.
        ``""`` where no fit was made: no forms given, a candidate within
        ``tol`` whose space lies off no form, a vacuous space, or the stacked
        route.
    fitted_forms : tuple
        If nonempty, the forms the space lies near and off by the thresholds
        of :func:`_off_a_form` while its candidate meets ``tol``, for which the
        fit was asked.  The space then depends on the forms given only through
        these.
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
    r"""A basis of every ``n x n`` matrix, the identity first.

    The basis a vacuous constraint returns, of shape ``(n**2, n, n)``.  Its
    members are orthogonal in the Frobenius inner product, each of norm
    ``sqrt(n)`` as every solution basis is, so ``out[0]`` is the identity
    itself and the first candidate.  The other diagonal members carry the
    columns of the discrete Fourier transform on their diagonal, and the
    off-diagonal members are the matrix units times ``sqrt(n)``.
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
    """``k_action`` as a callable on arrays of shape ``(Nk, dim_k)``.

    A callable is returned as it is.  A ``(dim_k, dim_k)`` matrix ``M``
    becomes ``k -> k @ M.T``; a matrix of another shape raises ``ValueError``.
    """
    if callable(k_action):
        return k_action
    M = np.asarray(k_action, dtype=float)
    if M.shape != (dim_k, dim_k):
        raise ValueError(f"_find_intertwiners(): k_action= is a matrix of shape {M.shape}. "
                         f"Pass a ({dim_k}, {dim_k}) matrix or a callable.")
    return lambda k: np.asarray(k, dtype=float) @ M.T


class _GridHamiltonians:
    r"""``H(k)`` on one grid of ``k``, evaluated once and shared by every operation.

    The grid's ``H(k)`` does not depend on the operation tested.  The image
    grid ``H(Rk)`` of an operation is another instance, made by
    :meth:`rotated`.  The grid is evaluated in chunks of points that stay
    within ``max_bytes``.

    Parameters
    ----------
    model : pythtb.TBModel
        The model.
    k_pts : array, shape (Nk, dim_k)
        The grid, in reduced coordinates.  A float array is held by reference
        and must not be changed.
    max_bytes : float, default 256e6
        The memory one evaluation may take, in bytes.  A grid whose
        Hamiltonians fit in it is evaluated in one call, ``model.hamiltonian``
        on the same points bit for bit, and kept whole; a larger one is
        streamed.
    bloch : _bloch._BlochSum, optional
        The model's Bloch sum (:func:`.._bloch.evaluator`), where another grid
        of the same model holds it.  ``None`` looks it up at the first
        evaluation.

    Attributes
    ----------
    n_evaluated : int
        How many k-points have been evaluated.
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
        """The number of k-points one evaluation takes within ``max_bytes``."""
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
        """The whole grid as one array of shape ``(Nk, nstate, nstate)``, kept.

        It takes ``16·Nk·nstate²`` bytes whatever ``max_bytes`` says, since the
        stacked route needs the whole stack.  :meth:`chunks` streams a grid
        instead.
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
        """Successive ``(lo, hi, H[lo:hi])`` blocks of the grid, within ``max_bytes``.

        A grid that :attr:`fits` is kept whole on the first pass, so a shared
        instance evaluates it once however many operations read it.  A larger
        grid is streamed and not kept.
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
        """The same model, Bloch sum and budget on ``k_rot``, the images of the grid."""
        return _GridHamiltonians(self.model, k_rot, self.max_bytes,
                                 bloch=self._sum())


# --------------------------------------------------------------------------
# The eigen-sift solver: the space is parametrized at one k and sifted over the rest.
# --------------------------------------------------------------------------
#
# ``X H_k = H'_k X`` at a *single* k has an explicit nullspace.  With
# ``H_k = V E V†`` and ``H'_k = V' E' V'†``, the equation on ``Y = V'† X V``
# reads ``Y_ab (E_b − E'_a) = 0``: ``Y`` is supported on matching-energy pairs,
# so the solution space is spanned by the outer products
# ``B_ab = V'[:, a] V[:, b]†`` over those pairs.  There are ``n`` of them for a
# non-degenerate matching spectrum, ``Σ d_i d'_i`` over degeneracy blocks in
# general, and **none** when the spectra do not match at all, which proves at
# ``O(n³)`` cost that no ``X`` solves the equation.  Every further k-point is
# then a *sift*: an ``n² × m`` constraint whose nullspace narrows the basis.


class _ConstraintPair:
    """The pairs ``(H_src(k), H(Rk))`` on a grid, at one point or in chunks.

    ``src_grid`` and ``rot_grid`` are the :class:`_GridHamiltonians` of the
    points ``k`` and of their images ``Rk``.  With ``conj``, for an
    antiunitary operation, the source is ``H(k)*``.  The conjugation is taken
    here, so that one grid of ``H(k)`` is read by both equations.  The two grids
    chunk alike, since a rotated grid has the model, the budget and the number
    of points of its source, so each ``k`` is paired with its own image.
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
    r"""The singular values and right singular vectors of one stage's constraint.

    ``B`` holds the ``m`` basis matrices, shape ``(m, n, n)``, and ``Hs`` and
    ``Hr`` the source and image Hamiltonians at one point.  Column ``j`` of
    the constraint ``C`` is ``vec(B_j Hs − Hr B_j)``, an ``n² × m`` matrix.  It
    is reduced to an ``m × m`` triangular factor by a QR of its row blocks,
    whose factors are stacked and reduced once more, and only that factor is
    decomposed.  The QR is backward stable and nothing is squared, so the
    singular values are those of ``C`` to ``eps‖C‖``.  ``block_bytes`` bounds
    the memory of one block, in bytes.

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
    """The order the sift visits ``nk`` grid points in, bit-reversed.

    The indices are bit-reversed over the least power of two not below
    ``nk``, and those past ``nk - 1`` skipped: ``nk = 6`` gives
    ``0, 4, 2, 1, 5, 3``, consecutive points far apart in the grid's order.
    Consecutive points of a lexicographic mesh lie on one line of the
    Brillouin zone, where a model can have more symmetry than on the whole
    grid.  Whether the sift may stop is settled by :func:`_grid_probe`; this
    order only lets it stop sooner.  Returns a list of the indices ``0`` to
    ``nk - 1``.
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
    r"""The point of the whole grid where a sifted basis fails most, if it fails.

    ``pair`` is the :class:`_ConstraintPair` of the grid and ``B`` the basis
    the sift kept, shape ``(m, n, n)``.  :data:`_GRID_PROBES` seeded
    combinations ``X = Σ y_j B_j``, each of norm ``sqrt(n)``, are streamed
    over every point, ``block_bytes`` bounding the memory of one block.  With
    probability one such a combination fails at a point wherever some member
    of the space does.  ``threshold`` is in the units of ``H``.

    Returns
    -------
    index : int
        The point of largest ``max|X H_k − H'_k X|``, or ``-1`` where that is
        at most ``threshold`` everywhere.
    worst : float
        That largest value, in the units of ``H``.
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
    """``n_bytes`` written as a reader reads it: ``1.449 GB``, ``100.4 kB``."""
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
    r"""The refusal ``detector_too_large``: pairs no window drops, past the budget.

    ``n_pairs`` is the number of eigenvector pairs at one point closer than
    rounding or the floor, ``n`` the number of states, and ``budget`` the
    bytes the eigen route's trial basis may take, one ``n × n`` matrix a
    pair.  A narrower window would drop some of those pairs and return a
    wrong space rather than a coarse one.  The refusal names the least budget
    that holds the pairs and one more, as the ``max_bytes`` of
    :func:`.._realized.realized_group` that runs the call.
    """
    need = 16.0 * float(n) ** 2 * (int(n_pairs) + 1)
    runs = _budget_that_runs(need)
    return Refusal(
        "detector_too_large",
        f"measuring the symmetry matrices at one k-point needs {_size(need)} of memory "
        f"for {int(n_pairs)} pairs of eigenvectors of H(k) and H(Rk) at equal energies, "
        f"H having {int(n)} states, and {_size(budget)} is allowed. Pass `max_bytes={runs}` to "
        "`realized_group`.",
        value=need, threshold=float(budget),
        context={"n_bytes": need, "max_bytes": float(budget), "n_pairs": int(n_pairs),
                 "n_states": int(n)})


def _pair_basis(E, V, Ep, Vp, delta, h, max_pairs, floor, gate=0.0):
    r"""The first basis of the sift: outer products over pairs at matching energies.

    ``E`` and ``V`` are the eigenvalues and eigenvectors, as columns, of the
    source Hamiltonian at one point, and ``Ep`` and ``Vp`` those of its image.
    Each pair with ``|Ep_a − E_b|`` at most ``max(delta·h, gate)`` gives the
    member ``Vp[:, a] V[:, b]†``.  ``delta`` is the window relative to ``h``,
    the Hamiltonian's magnitude at this point, and ``gate`` the covariance a
    kept candidate may carry, in the units of ``H``.  The members are
    orthonormal in the Frobenius inner product.

    ``max_pairs`` bounds the number of pairs.  Past it the ``max_pairs``
    nearest pairs are kept and the window narrows to them.  ``floor`` is the
    narrowest window, relative to ``h``, at which the pairs dropped stay
    below the cut.  No basis is formed where the window falls below
    ``floor``, or where the pairs within ``gate`` alone pass ``max_pairs``:
    dropping those pairs would make the space wrong rather than coarse.

    Returns
    -------
    B : ndarray, shape (m, n, n), or None
        The members; ``None`` where no basis is formed.
    delta : float or int
        The window used, relative to ``h``.  Where no basis is formed, the
        number of pairs within ``floor`` and ``gate``, which the solve needs.
    gaps : ndarray or None
        Every ``|Ep_a − E_b|`` over ``h``, ascending; ``None`` where no basis
        is formed.
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
    """One pass of the eigen sift at the window ``delta``, as a record.

    ``pair`` is the :class:`_ConstraintPair` of a grid of ``nk`` points and
    ``n`` states.  The space is parametrized at one point
    (:func:`_pair_basis`, at most ``max_pairs`` pairs) and cut down at the
    others in the order :func:`_visit_order` gives.  At each point a
    combination is kept where its singular value is at most ``cut`` times the
    constraint's norm there, or ``sqrt(n)·floor``, ``floor`` being the
    covariance in the units of ``H`` a kept candidate may carry.  Once
    ``stable`` points in a row leave the basis unchanged, the sift stops,
    unless :func:`_grid_probe` finds a point where the basis fails; the sift
    resumes there.  No candidate is judged here.  The parametrization point
    is ``k1`` where given; otherwise the point of widest source spectrum among
    the first ``probes`` points, or over the whole grid where all of those are
    degenerate within the window.

    Returns a mapping that :func:`_find_intertwiners` reads.  Without ``k1``,
    ``"flat"`` is true where every point is degenerate within the window, and
    the record then holds the norms over the grid the vacuity test reads.
    ``"too_large"`` holds the number of pairs the solve needs where no basis
    is formed.  Otherwise ``"basis"`` is the basis kept, empty where nothing
    survives, with its singular values, its scale, the window used, the
    points sifted and ``"spectral"``, the most the two spectra differ at one
    level over those points.
    """
    spectra: dict[int, tuple] = {}
    # The energy window's scale is the Hamiltonian's magnitude about its mean
    # on-site energy, `w`, which a constant shift of H does not move.  The tests
    # that bound rounding (the window's floor, the vacuity cut, the probe of the
    # whole grid) read its magnitude with the energy zero, `h`, since rounding
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
        # the constraint operator on the way.  That norm is max_ab |E_b - E'_a|,
        # so the absolute vacuity test costs no more than the eigenvalues
        # already needed.
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
        # The grid holds a point that is not degenerate within the window, though
        # every probe point is (a stacked layer's first grid points all sit on the
        # k_z line above GM, where a Kramers pair is degenerate at every k_z).
        # The sift is parametrized at the point the scan found, as a forced
        # re-sift (`k1=i1`) would be.

    # the window reads `w`; its floor, the rounding of an eigenvector, is
    # eps/cut of `h`, carried onto the window's scale; and it never narrows below
    # the covariance a kept candidate may carry, in the units of H
    B, delta_used, gaps = _pair_basis(E, V, Ep, Vp, delta, w, max_pairs,
                                      np.finfo(float).eps / max(float(cut), 1e-300) * (h / w),
                                      gate=floor)
    if B is None:
        # the pairs no window drops are past the budget: no basis, and how
        # many pairs the solve needs
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
    # evidence that it breaks.  Where the operator at a point is itself that
    # small (one band, whose operator is the violation itself, or H(k) nearly a
    # multiple of 1, at a Dirac point), the point says nothing at the precision
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
        # matrix space, is max_ab |E_b - E'_a|, from two eigenvalue problems.  It
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
    # what the (vanishing) constraint is measured against, as in the vacuity
    # branch.
    if s_last is None:
        s_last, scale_last, informative = np.zeros(m), h, False
    scale = float(scale_last) if informative else h
    return {"flat": False, "k_solve": i1, "n_solve": n_solve, "h": h, "basis": B,
            "trace": tuple(trace), "window": delta_used, "gaps": gaps,
            "singular_values": np.asarray(s_last)[::-1].copy(), "scale": scale,
            "spectral": spectral, "points": tuple(points), "empty_ratio": None}


def _orthonormal_basis(B, n):
    """The matrices ``B`` made orthogonal, each of Frobenius norm ``sqrt(n)``.

    ``B`` holds ``d`` matrices of shape ``(n, n)``, orthonormal up to rounding
    as the sift's are by construction.  Their order survives.  Returns an
    array of shape ``(d, n, n)``.
    """
    d = int(B.shape[0])
    if d == 0:
        return np.zeros((0, n, n), dtype=complex)
    Q, _ = np.linalg.qr(B.reshape(d, -1).T)
    return np.ascontiguousarray(Q.T).reshape(d, n, n) * np.sqrt(n)


def _candidates_and_search(basis, dim, n):
    """A space's candidates: its basis and, at ``dim >= 2``, a searched unitary.

    ``basis`` holds the ``dim`` members, each of norm ``sqrt(n)``.  At
    ``dim >= 2`` the members and :data:`_UNITARY_PROBES` seeded combinations
    of them are brought to that norm, the one of largest least singular value
    is picked, and its polar factor, itself a member of the space, is
    appended.  Returns the list of candidates and the record
    :attr:`_IntertwinerSpace.search`.
    """
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
    r"""A form's matrices that carry ``H(k)`` onto ``H(Rk)``, fitted over the grid.

    ``pair`` is the :class:`_ConstraintPair` of the grid, ``form`` the boolean
    ``(n, n)`` pattern of the entries a matrix of the operation may hold, and
    ``n`` the number of states.  ``cut`` is the relative cut, and ``floor``
    the covariance at one point, in the units of ``H``, that a member may
    carry.  The ``p`` entries the form allows are fitted
    over every point: the directions minimizing
    ``Σ_k ‖X H(k) − H(Rk) X‖²_F`` are the eigenvectors of lowest eigenvalue of
    the Hermitian ``p × p`` matrix

    .. math::

        G_{ab,cd} = \sum_k \delta_{ac} (H_k^2)_{db} + \delta_{bd} (H'^2_k)_{ac}
                    - 2 H'_{k,ac} H_{k,db},

    ``H_k`` the source, conjugated for an antiunitary operation, and ``H'_k``
    its image.  The members are counted as the sift counts them.  In
    ascending order, a direction is a member where at every point its
    covariance ``‖X H(k) − H(Rk) X‖_F``, ``X`` of unit norm, is within
    ``max(cut·‖L_k‖, sqrt(n)·floor)``, ``‖L_k‖`` the norm of
    ``X -> X H(k) − H(Rk) X`` there.  The count stops at the first that is not.

    Returns
    -------
    members : ndarray, shape (count, n, n)
        The members, each of unit norm.
    worst : ndarray, shape (count,)
        Each member's largest covariance over the grid.
    scale : float
        The largest ``‖L_k‖`` over the grid.
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
    """A form's entries, listed block by block.

    ``form`` is the boolean ``(n, n)`` pattern of the entries a matrix of an
    operation may hold.  The columns that may reach the same rows make one
    rectangular block, one block a site for a form that carries each site's
    states onto its image's.  Returns ``rows`` and ``cols``, the entries block
    by block, each block's in row-major order, and ``blocks``, each block's
    ``(rows, cols, start)``, ``start`` the position of its first entry.
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
    r"""``Σ_k H'_{k,ac} H_{k,db}`` for every two entries ``(a, b)``, ``(c, d)`` of a form.

    ``Hs`` and ``Hr`` hold the source ``H`` and the image ``H'`` at the
    points, shape ``(Nk, n, n)``, and ``blocks`` and ``p`` are the form's
    blocks and number of entries (:func:`_form_blocks`).  For the blocks
    ``(R_I, C_I)`` and ``(R_J, C_J)`` it is one matrix product of
    ``H'[R_I, R_J]`` and ``H[C_J, C_I]``, each flattened over the points.
    Returns the ``p × p`` matrix.
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
    """The source, the image and the sift's cut at each point given.

    ``pair`` is the :class:`_ConstraintPair` of the grid, ``points`` the
    indices of the points, and ``n`` the number of states.  The cut at a point
    is ``g_k = max(cut·‖L_k‖, sqrt(n)·floor)``, ``‖L_k‖`` the constraint's norm
    there and ``floor`` in the units of ``H``.  Returns a list of
    ``(H_src(k), H(Rk), g_k)``, which :func:`_form_admits` reads.
    """
    least = float(np.sqrt(n)) * float(floor)
    out = []
    for i in points:
        Hs, Hr = pair.at(int(i))
        Es, Er = np.linalg.eigvalsh(Hs), np.linalg.eigvalsh(Hr)
        op = float(max(abs(Er[-1] - Es[0]), abs(Es[-1] - Er[0])))
        out.append((Hs, Hr, max(float(cut) * op, least)))
    return out


def _form_admits(constraints, form):
    r"""Whether a matrix of a form can meet the sift's cut at every point given.

    ``constraints`` is the list :func:`_sifted_constraints` returns, with the
    cut ``g_k`` at each point, and ``form`` the boolean pattern of the entries
    a matrix of the operation may hold.  A matrix ``x`` of the form, of unit
    norm, within ``g_k`` at each point has ``x† G x ≤ Σ_k g_k²``, ``G`` the
    matrix of :func:`_refit_on_form` summed over these points.  So a lowest
    eigenvalue of ``G`` above that bound, by more than its rounding
    ``p·eps·‖G‖`` for the ``p`` entries the form allows, proves that no
    matrix of the form meets the cut at these points, nor on any grid that
    holds them.  A unitary within the covariance gate ``tol`` at a point is
    within ``sqrt(n)·tol`` there at unit norm, so where the constraints were
    made with a ``floor`` of at least ``tol`` the proof holds for the gate as
    well.  Returns ``False`` only on that proof.
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
        The sift's orthogonal basis, each member of norm ``sqrt(n)``.
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
        Each form whose nearest member of the space holds outside it more than
        rounding and no more than a near miss.  Rounding is
        :data:`.._tolerance._IN_BLOCKS`, or ``cut`` where larger, times the
        most any member holds outside the form, or times one where that is
        less.  A near miss is :data:`.._tolerance.ANSATZ_GATE`, or ``cut``
        where larger, of the member's norm.
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
    """The candidates' polar factors, covariance over the grid and unitarity.

    ``cands`` holds the candidate matrices, each ``n x n``.  ``full`` is the
    pair ``(H_src, H_rot)`` of whole grids in memory, as the stacked route has
    them.  Otherwise ``pair``, a :class:`_ConstraintPair`, is streamed over in
    chunks.  Both give ``max_k ‖U H_src(k) U† − H(Rk)‖_max`` over every point
    supplied, ``H_src(k)`` being ``H(k)*`` for an antiunitary operation.
    Returns the unitaries, the residuals and each candidate's
    ``‖X†X − 1‖_max``, as :class:`_IntertwinerSpace` holds them.
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
        raise ValueError(f"_find_intertwiners(): k_pts= has {k_pts.shape[1]} columns. "
                         f"Pass {dim_k} reduced coordinates for each point.")
    if k_pts.shape[0] < 2:
        raise ValueError(f"_find_intertwiners(): k_pts= has length {k_pts.shape[0]}. "
                         "Pass at least two points.")
    if solver not in ("eigen", "stacked"):
        raise ValueError(f"_find_intertwiners(): solver= is {solver!r}. "
                         "Pass 'eigen' or 'stacked'.")
    cut = float(tol) if cut is None else float(cut)
    # the covariance, in the units of H, a candidate the nullspace keeps may carry
    # at a point: the cut read in those units, unless a declared precision, whose
    # cut is relative to the Hamiltonian's own magnitude, sets it at the gate
    floor = cut if _floor is None else float(_floor)

    act = _as_k_action(k_action, dim_k)
    k_rot = np.asarray(act(k_pts), dtype=float)
    if k_rot.shape != k_pts.shape:
        raise ValueError(f"_find_intertwiners(): k_action= maps k_pts= of shape "
                         f"{k_pts.shape} to an array of shape {k_rot.shape}. Pass a "
                         "k_action= that keeps the shape of the points it is handed.")

    n = int(model.nstate)
    n2 = n * n
    nk = int(k_pts.shape[0])
    eye = np.eye(n)

    if hamiltonians is None:
        src_grid = _GridHamiltonians(model, k_pts)
    else:
        src_grid = hamiltonians
        if src_grid.n_kpts != nk or not np.array_equal(src_grid.k_pts, k_pts):
            raise ValueError("_find_intertwiners(): hamiltonians= holds H on points other "
                             "than k_pts=. Pass the hamiltonians= built on these k_pts=, "
                             "or None.")
    rot_grid = src_grid.rotated(k_rot)

    window_used, k_solve, trace, escalated, spectral, sifted = 0.0, -1, (), (), 0.0, ()
    fit, astray = "", []

    if solver == "stacked":
        H, Hr = src_grid.full(), rot_grid.full()
        Hsrc = H.conj() if antiunitary else H

        def _constraint(a, b):
            """The operator ``X -> X a_k − b_k X`` on row-major ``vec(X)``, by point."""
            # vec(X H) = (I (x) H^T) vec(X);  vec(H' X) = (H' (x) I) vec(X)
            return (np.einsum("ij,kab->kiajb", eye, a.transpose(0, 2, 1)).reshape(-1, n2, n2)
                    - np.einsum("kij,ab->kiajb", b, eye).reshape(-1, n2, n2))

        # Solve: one full-precision stacked SVD.  A grid too large to stack is
        # cut down to an evenly strided subset of its own points, never
        # reordered, and never solved by the normal equations, whose sqrt(eps)
        # floor would make dim == 0 an unsound verdict.
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

        # Cut: two regimes, and the second is not the first with a smaller
        # number.  When the constraint operator *vanishes* it has no scale of its
        # own, so a relative cut has nothing to be relative to and would read its
        # own rounding noise as one.  Vacuity is therefore judged absolutely,
        # against the scale of H over the grid, to which that noise is
        # proportional, and a vacuous constraint means every X solves.
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
        # the pairs max_bytes holds, past which the window narrows, and the
        # budget's ceiling, which the pairs no window drops may raise the
        # budget to: never below max_bytes
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
                    # the pairs no window drops are past max_bytes: the solve
                    # again from its start, as a call with the ceiling for its
                    # max_bytes runs it
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
