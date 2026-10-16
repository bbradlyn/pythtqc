r"""``H(k)`` of a tight-binding model, summed from lattice-vector blocks built once.

In pythtb's convention I every hopping ``t_ij(R)`` enters the Bloch Hamiltonian
through the block ``T(R)`` of its lattice vector, and the k-dependence is one
phase per lattice vector and one per orbital:

.. math:: H(k) = \Lambda(k)^\dagger \Big[\sum_R e^{2\pi i\,k\cdot R}\, T(R)
          + \text{h.c.}\Big] \Lambda(k) + \text{on-site terms},
          \qquad \Lambda(k) = \operatorname{diag}(e^{2\pi i\,k\cdot\tau_j}).

pythtb's ``TBModel.hamiltonian`` gathers the hoppings into those blocks on every
evaluation.  On a wannier90 import -- several hundred thousand hoppings over a
few dozen lattice vectors -- the gathering costs about a second each time,
however few k-points are asked for.  The blocks depend on nothing but the model
and its parameter values, so they are gathered once and kept, and every later
``H(k)`` is a product of phases with blocks.  After the gathering the arithmetic
is pythtb's own, step for step, so for the same k-points ``H(k)`` is bit for bit
the array ``model.hamiltonian(k_pts=k, flatten_spin_axis=True, **params)``
returns.  A model with no periodic direction, and parameter values given as a
sweep, are evaluated by pythtb itself.

The blocks are kept by the model's **content** -- the lattice, the orbital
centres, the on-site terms and the hopping arrays, hashed together.  Two models
with the same content share one set of blocks, and a model changed in any way
between two evaluations has a different content, so a kept set of blocks is
never out of date.  What is kept is bounded in bytes.

The hopping list is read here as pythtb stores it, four arrays, rather than as
``model.hoppings``, which builds one record per hopping and takes seconds on a
wannier90 import.
"""

from __future__ import annotations

import hashlib
import math
from collections import OrderedDict

import numpy as np
from pythtb import TBModel

#: How many bytes of lattice-vector blocks the process keeps, over every model,
#: the least recently used going first.  A wannier90 import of 96 orbitals whose
#: hoppings reach 79 lattice vectors holds 79 x 96 x 96 complex numbers, 11.7 MB,
#: so the model, its presentations and its symmetrized copies fit with room to
#: spare.  A set of blocks larger than this is built, used and not kept.
_KEEP_BYTES = 256e6

#: Bytes charged per k-point of one evaluation, as a multiple of the 16 bytes of
#: each complex number the Bloch sum holds for that point: one phase per lattice
#: vector, one per orbital, and the ``nstate x nstate`` matrix.  The multiple is
#: measured with ``tracemalloc``, the returned array included, over 64 and 256
#: k-points: a peak of 2.97 to 2.99 times the numbers counted on 96 spinless and
#: on 48 spinful orbitals with 79 lattice vectors, where the sum holds three
#: matrices at once, and 2.34 on 4 orbitals with 900 lattice vectors, where the
#: phase rows and their real temporaries dominate.
_POINT_FACTOR = 3.0

#: The k-points every presentation this package builds is measured on against
#: the model it presents -- a layer's slab and decoupled stack, the spinless
#: presentation of a spinful model, the model with its centres moved: this
#: many random ``k``, drawn from this seed.  A presentation changes nothing in
#: exact arithmetic, so a difference is the package disagreeing with itself.
CHECK_K = 8
CHECK_SEED = 1729


def point_bytes(n_R: int, norb: int, nstate: int) -> float:
    """What one k-point of an evaluation holds, in bytes, charged at :data:`_POINT_FACTOR`."""
    return _POINT_FACTOR * 16.0 * (int(n_R) + int(norb) + int(nstate) ** 2)


# --------------------------------------------------------------------------
# the hopping list, read as arrays
# --------------------------------------------------------------------------

def hopping_arrays(model):
    """The model's hopping list as the four arrays pythtb stores it in.

    Returns
    -------
    (amplitudes, i, j, R) : ndarray
        ``amplitudes`` of shape ``(n,)``, or ``(n, 2, 2)`` for a spinful model;
        ``i`` and ``j`` the two orbitals of each term ``t_ij(R)``; ``R`` of
        shape ``(n, dim_r)``, integer.  Read-only views of the model's own
        storage, in the order ``model.hoppings`` lists them and with the same
        numbers; a parametrized term is in neither.
    """
    out = []
    for arr in model._hoptable.components():
        view = arr.view()
        view.flags.writeable = False
        out.append(view)
    return tuple(out)


def _digest(model, amplitudes, i, j, R, site) -> str:
    """The content hash of one Hamiltonian: its geometry, on-site terms and hoppings.

    Every array is hashed with its type and shape, so two contents can agree
    only number for number.
    """
    lattice = model.lattice
    h = hashlib.blake2b(digest_size=16)
    h.update(repr((int(model.nspin), tuple(int(d) for d in lattice.periodic_dirs),
                   int(np.shape(i)[0]))).encode())
    for arr, dtype in ((lattice.lat_vecs, np.float64), (lattice.orb_vecs, np.float64),
                       (site, np.complex128), (amplitudes, np.complex128),
                       (i, np.int64), (j, np.int64), (R, np.int64)):
        a = np.ascontiguousarray(arr, dtype=dtype)
        h.update(repr((a.dtype.str, a.shape)).encode())
        h.update(a.data)
    return h.hexdigest()


def model_key(model) -> str:
    """The content hash of the model as it stands: lattice, centres, on-site terms, hoppings.

    Two models with the same key have the same ``H(k)`` at every ``k``.  It is
    also the key the model's blocks are kept under, whenever its terms carry
    no parameter.
    """
    return _digest(model, *hopping_arrays(model), model.onsite)


# --------------------------------------------------------------------------
# the Bloch sum
# --------------------------------------------------------------------------

#: ``exp(2 pi i x)`` of reduced dot products, pythtb's own.
_phases = TBModel._phases_from_reduced_dot_products


class _BlochSum:
    """``H(k)`` of one model at one set of parameter values, from blocks built once.

    Called with k-points in any form ``TBModel.hamiltonian`` takes, it returns
    the ``(Nk, nstate, nstate)`` array that call returns with
    ``flatten_spin_axis=True``, bit for bit.
    """

    def __init__(self, model, arrays):
        hop_amps, i_idx, j_idx, R_vecs, site = arrays
        lattice = model.lattice
        norb = int(lattice.norb)
        nspin = int(model.nspin)
        per = np.asarray(lattice.periodic_dirs, dtype=int)
        hop_amps = np.asarray(hop_amps, dtype=complex)
        i_idx = np.asarray(i_idx, dtype=int)
        j_idx = np.asarray(j_idx, dtype=int)
        R_per = np.asarray(R_vecs, dtype=float)[:, per]
        self.dim_k = int(lattice.dim_k)
        self.norb, self.nspin, self.nstate = norb, nspin, norb * nspin
        self.n_hops = int(hop_amps.shape[0])
        self.orb_per = lattice.orb_vecs[:, per]
        # a copy: the model's own on-site array is changed in place by set_onsite
        self.site = np.array(site, copy=True)
        self.unique_R, self.blocks, self.n_R = None, None, 0
        if self.n_hops:
            M = self.nstate
            unique_R, R_inv = np.unique(R_per, axis=0, return_inverse=True)
            n_R = unique_R.shape[0]
            T_R = np.zeros((n_R, M, M), dtype=complex)
            if nspin == 1:
                np.add.at(T_R, (R_inv, i_idx, j_idx), hop_amps)
            else:
                spin_out = np.repeat(np.arange(nspin), nspin)
                spin_in = np.tile(np.arange(nspin), nspin)
                rows = (i_idx[:, None] * nspin + spin_out[None, :]).reshape(-1)
                cols = (j_idx[:, None] * nspin + spin_in[None, :]).reshape(-1)
                np.add.at(T_R, (np.repeat(R_inv, nspin * nspin), rows, cols),
                          hop_amps.reshape(-1))
            self.unique_R, self.blocks, self.n_R = unique_R, T_R.reshape(n_R, -1), int(n_R)
        self.point_bytes = point_bytes(self.n_R, norb, self.nstate)
        # the arrays, and about a kilobyte for the objects that hold them
        self.nbytes = 1024 + int(sum(a.nbytes for a in (self.unique_R, self.blocks,
                                                        self.orb_per, self.site)
                                     if a is not None))

    def __call__(self, k_pts) -> np.ndarray:
        if k_pts is None:
            raise ValueError("k_pts must be specified for periodic systems (dim_k > 0).")
        # pythtb's own reading of the k-points, which consults nothing but dim_k
        k_vecs = TBModel._normalize_kpoints(self, k_pts)
        n_kpts = k_vecs.shape[0]
        norb, nspin, M = self.norb, self.nspin, self.nstate
        if self.n_hops:
            R_phases = _phases(k_vecs @ self.unique_R.T)
            orb_phases = _phases(k_vecs @ self.orb_per.T)
            if nspin != 1:
                orb_phases = np.repeat(orb_phases, nspin, axis=1)
            T_inner = (R_phases @ self.blocks).reshape(n_kpts, M, M)
            T_inner += np.conj(T_inner.swapaxes(-2, -1))
            ham = np.conj(orb_phases[:, :, None]) * T_inner * orb_phases[:, None, :]
        else:
            ham = np.zeros((n_kpts, M, M), dtype=complex)
        if nspin == 1:
            diag = np.arange(norb)
            ham[:, diag, diag] += self.site
        else:
            rows = np.arange(M).reshape(norb, nspin)
            ham[:, rows[:, :, None], rows[:, None, :]] += self.site[None, :, :, :]
        return ham


class _PythtbSum:
    """``H(k)`` by pythtb's ``hamiltonian`` itself: no periodic direction, or a sweep."""

    def __init__(self, model, params):
        self.model, self.params = model, dict(params)
        self.nstate = int(model.nstate)
        # every hopping its own lattice vector: the most one evaluation can hold
        self.point_bytes = point_bytes(int(model.nhops), int(model.norb), self.nstate)

    def __call__(self, k_pts) -> np.ndarray:
        return self.model.hamiltonian(k_pts=k_pts, flatten_spin_axis=True, **self.params)


_kept: OrderedDict = OrderedDict()
_kept_bytes = 0


def _keep(key, bloch) -> None:
    """Keep one set of blocks, dropping the least recently used past :data:`_KEEP_BYTES`."""
    global _kept_bytes
    if bloch.nbytes > _KEEP_BYTES:
        return
    _kept[key] = bloch
    _kept_bytes += bloch.nbytes
    while _kept_bytes > _KEEP_BYTES:
        _old_key, old = _kept.popitem(last=False)
        _kept_bytes -= old.nbytes


def evaluator(model, params=None):
    """The Bloch sum of ``model`` at these parameter values, built once per content.

    The parameter names are checked as ``TBModel.hamiltonian`` checks them, with
    the same error.  The blocks are gathered from the arrays pythtb itself would
    sum at these values, and kept under their content; a later request for the
    same content, from this model or any other, finds them.
    """
    params = dict(params or {})
    model._check_missing_parameters(params)
    if int(model.lattice.dim_k) == 0:
        return _PythtbSum(model, params)
    scalars, _names, sweeps = model._params_to_sweep(params)
    if sweeps:
        return _PythtbSum(model, params)
    arrays = model._evaluate_params(scalars)
    key = _digest(model, *arrays)
    bloch = _kept.get(key)
    if bloch is not None:
        _kept.move_to_end(key)
        return bloch
    bloch = _BlochSum(model, arrays)
    _keep(key, bloch)
    return bloch


def hamiltonian(model, k_pts, params=None) -> np.ndarray:
    """``model.hamiltonian(k_pts=k_pts, flatten_spin_axis=True, **params)``, bit for bit.

    The checks run in pythtb's order -- the parameter names, then the
    k-points -- so a request pythtb stops stops here with the same error.
    """
    params = dict(params or {})
    model._check_missing_parameters(params)
    if int(model.lattice.dim_k) == 0:
        return model.hamiltonian(k_pts=k_pts, flatten_spin_axis=True, **params)
    if k_pts is None:
        raise ValueError("k_pts must be specified for periodic systems (dim_k > 0).")
    k_vecs = model._normalize_kpoints(k_pts)
    return evaluator(model, params)(k_vecs)


def onsite(model, params=None) -> np.ndarray:
    """The on-site terms at these parameter values: ``(norb,)``, or ``(norb, 2, 2)`` spinful."""
    params = dict(params or {})
    model._check_missing_parameters(params)
    scalars, _names, _sweeps = model._params_to_sweep(params)
    return np.asarray(model._evaluate_params(scalars)[4], dtype=complex)


def onsite_mean(model, params=None) -> float:
    r"""The mean on-site energy per state: the energy zero of the model's own numbers.

    ``tr H(k)`` averages over the Brillouin zone to the trace of the on-site
    terms, so this is what a constant shift ``H → H + c·1`` moves by exactly
    ``c`` -- a wannier90 import's on-site energies carry the energy zero of the
    calculation it came from.  The diagonal on-site entries are summed exactly,
    so on-site energies that cancel read ``0.0``, not a rounding of it.
    """
    site = onsite(model, params)
    diagonal = site if site.ndim == 1 else np.diagonal(site, axis1=-2, axis2=-1)
    return math.fsum(np.real(diagonal).ravel().tolist()) / float(model.nstate)


def magnitude(H, shift=0.0) -> float:
    """``max|H − shift·1|`` over one matrix or a stack of them; ``max|H|`` exactly at ``shift = 0``."""
    H = np.asarray(H)
    if H.size == 0:
        return 0.0
    if shift == 0.0:
        return float(np.abs(H).max())
    out = np.abs(H)
    diagonal = np.arange(H.shape[-1])
    out[..., diagonal, diagonal] = np.abs(H[..., diagonal, diagonal] - shift)
    return float(out.max())


def solve(model, k_pts, params=None):
    """``model.solve_ham(k_pts, return_eigvecs=True, flatten_spin_axis=True, **params)``, bit for bit.

    pythtb's diagonalization of the same ``H(k)``: the same Hermiticity check,
    ``numpy.linalg.eigh`` on the same matrices, eigenvectors as rows, and a
    single k-point's axis dropped.
    """
    ham = hamiltonian(model, k_pts, params)
    if not np.allclose(ham, ham.swapaxes(-1, -2).conj()):
        raise ValueError("Hamiltonian matrix is not Hermitian.")
    evals, evecs = np.linalg.eigh(ham.astype(np.complex128))
    evecs = evecs.swapaxes(-1, -2)
    if int(model.lattice.dim_k) != 0 and evals.shape[0] == 1:
        evals, evecs = evals[0], evecs[0]
    return evals, evecs
