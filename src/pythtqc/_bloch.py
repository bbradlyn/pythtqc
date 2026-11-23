r"""``H(k)`` of a tight-binding model, summed from lattice-vector blocks built once.

In pythtb's convention I every hopping ``t_ij(R)`` enters the Bloch Hamiltonian
through the block ``T(R)`` of its lattice vector, and the k-dependence is one
phase per lattice vector and one per orbital:

.. math:: H(k) = \Lambda(k)^\dagger \Big[\sum_R e^{2\pi i\,k\cdot R}\, T(R)
          + \text{h.c.}\Big] \Lambda(k) + \text{on-site terms},
          \qquad \Lambda(k) = \operatorname{diag}(e^{2\pi i\,k\cdot\tau_j}).

The blocks depend on nothing but the model, so they are gathered once, from
the four arrays pythtb stores the hoppings in, and kept.  Every later ``H(k)``
is a product of phases with blocks, in pythtb's own arithmetic step for step,
so for the same k-points ``H(k)`` is bit for bit the array
``model.hamiltonian(k_pts=k, flatten_spin_axis=True)`` returns.  A model with no
periodic direction is evaluated by pythtb itself.

A parameterized model is evaluated at the values of its parameters once, where
it enters (:func:`bound`), and every step after sees that numeric model.

The blocks are kept under a hash of the model as it stands: the lattice, the
orbital centers, the on-site terms and the hopping arrays.  Two models with the
same hash share one set of blocks, and a model changed in any way between two
evaluations has another hash, so a kept set of blocks is never out of date.
What is kept is bounded in bytes (:data:`_KEEP_BYTES`).
"""

from __future__ import annotations

import hashlib
import math
import numbers
from collections import OrderedDict

import numpy as np
from pythtb import TBModel

#: How many bytes of lattice-vector blocks the process keeps, over every model,
#: the least recently used going first.  A Wannier90 import of 96 orbitals whose
#: hoppings reach 79 lattice vectors holds 79 x 96 x 96 complex numbers, 11.7 MB,
#: so the model, its presentations and its symmetrized copies fit with room to
#: spare.  A set of blocks larger than this is built, used and not kept.
_KEEP_BYTES = 256e6

#: Bytes charged per k-point of one evaluation, as a multiple of the 16 bytes of
#: each complex number the Bloch sum holds for that point: one phase per lattice
#: vector, one per orbital, and the ``nstate x nstate`` matrix.  The multiple is
#: measured with ``tracemalloc``, the returned array included.
_POINT_FACTOR = 3.0

#: The k-points every presentation this package builds is measured on against
#: the model it presents (a layer's slab and decoupled stack, the spinless
#: presentation of a spinful model, the model with its centers moved): this
#: many random ``k``, drawn from this seed.  A presentation changes nothing in
#: exact arithmetic, so a difference is the package disagreeing with itself.
CHECK_K = 8
CHECK_SEED = 1729


def point_bytes(n_R: int, norb: int, nstate: int) -> float:
    """The bytes one k-point of an evaluation holds, charged at :data:`_POINT_FACTOR`.

    ``n_R`` is the number of lattice vectors, ``norb`` of orbitals and
    ``nstate`` of states.  Returns ``_POINT_FACTOR`` times 16 bytes for each of
    the ``n_R + norb + nstate**2`` complex numbers the Bloch sum holds there.
    """
    return _POINT_FACTOR * 16.0 * (int(n_R) + int(norb) + int(nstate) ** 2)


# --------------------------------------------------------------------------
# the hopping list, read as arrays
# --------------------------------------------------------------------------

def hopping_arrays(model):
    """The hopping list of ``model`` as the four arrays pythtb stores it in.

    Returns ``(amplitudes, i, j, R)``: ``amplitudes`` of shape ``(n,)``, or
    ``(n, 2, 2)`` for a spinful model; ``i`` and ``j`` the two orbitals of each
    term ``t_ij(R)``; ``R`` the integer lattice vectors, of shape
    ``(n, dim_r)``.  Each is a read-only view of the model's own storage, in
    the order ``model.hoppings`` lists the terms and with the same numbers.
    """
    out = []
    for arr in model._hoptable.components():
        view = arr.view()
        view.flags.writeable = False
        out.append(view)
    return tuple(out)


def _digest(model, amplitudes, i, j, R, site) -> str:
    """The hash of one Hamiltonian: its geometry, on-site terms and hoppings.

    ``model`` gives the spin, the periodic directions, the lattice vectors and
    the orbital centers; ``amplitudes``, ``i``, ``j`` and ``R`` are the hopping
    arrays (:func:`hopping_arrays`), and ``site`` the on-site terms.  Each
    array is cast to a fixed type and hashed with its type and shape, so two
    Hamiltonians share a hash only number for number.  Returns the hash as a
    hexadecimal string.
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
    """The hash of ``model`` as it stands: lattice, centers, on-site terms, hoppings.

    Two models with the same hash have the same ``H(k)`` at every ``k``.  It is
    also the hash the model's blocks are kept under.
    """
    return _digest(model, *hopping_arrays(model), model.onsite)


def bound(model, params=None, *, call=None):
    """The model with the values of its parameters in place; the model itself where it has none.

    ``params`` maps each parameter's name to its value.  The names are checked
    as ``TBModel.hamiltonian`` checks them, with the same error, and each value
    is checked to be a number, before anything is evaluated.  ``call`` is the
    public call's name, which the error for a value that is not a number
    opens with; where it is ``None`` the error opens with ``params=``.

    Raises
    ------
    ValueError
        A name in ``params`` that is not a parameter of the model, a parameter
        given no value, or a value that is not a number.
    """
    params = dict(params or {})
    model._check_missing_parameters(params)
    for name, value in params.items():
        # a Python or numpy number, or a zero-dimensional numeric array
        one = (value.ndim == 0 and np.issubdtype(value.dtype, np.number)
               if isinstance(value, np.ndarray) else isinstance(value, numbers.Number))
        if not one:
            try:
                dims = np.ndim(value)
            except ValueError:                  # values of different lengths
                dims = None
            arg = name if name.isidentifier() else "m"
            hint = {1: " Pass one number, and make a call for each value.",
                    2: (" Where the parameter enters a term that acts on spin, write that "
                        "term as a function of the parameter that returns the 2x2 block, "
                        f"such as lambda {arg}: {arg} * sz.")
                    }.get(dims, "")
            opening = f"{call}(): " if call is not None else ""
            raise ValueError(f"{opening}params= gives '{name}' a value that is not a "
                             f"number.{hint}")
    if not model._has_parameterized_terms():
        return model
    return model.with_parameters(params)


# --------------------------------------------------------------------------
# the Bloch sum
# --------------------------------------------------------------------------

#: ``exp(2 pi i x)`` of reduced dot products, pythtb's own.
_phases = TBModel._phases_from_reduced_dot_products


class _BlochSum:
    """``H(k)`` of one model, from blocks built once.

    ``model`` gives the lattice and the spin.  ``arrays`` is
    ``(amplitudes, i, j, R, site)``, the hoppings and on-site terms as
    ``model._evaluate_params({})`` returns them.  ``key`` is the hash they were
    built from, kept as ``.key``.  Called with k-points in any form
    ``TBModel.hamiltonian`` takes, the sum returns the ``(Nk, nstate, nstate)``
    array that call returns with ``flatten_spin_axis=True``, bit for bit.
    """

    def __init__(self, model, arrays, key=None):
        self.key = key
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

    def phase_rounding(self, k_pts, rotation=None) -> float:
        r"""The rounding ``H`` carries from its Bloch phases, at ``k`` or at its image ``k W``.

        ``k_pts``, of shape ``(n, dim_k)``, are the reduced k-points ``k``.
        ``rotation``, of shape ``(dim_k, dim_k)``, is ``W``, an operation's
        action on reduced k as rows, ``k \mapsto k W``.  Where it is omitted,
        ``H(k)`` itself is estimated.  Each phase ``e^{2\pi i x}`` is taken of a
        reduced dot product ``x = q\cdot R`` or ``q\cdot\tau``, which rounds at
        ``ε`` times the sum of its terms' sizes.  At ``q = k W`` that sum is at
        most ``(|k||W|)\cdot|R|``, large where ``W`` has large entries, as on a
        strongly skewed lattice basis, while the terms cancel to an ``x`` of
        order one.  An entry of ``H`` rounds at ``2\pi ε`` times each
        hopping's size weighted by those sums for its lattice vector and its
        two orbitals' centers.  The bound is summed along the entry's row,
        which bounds ``|D\,δH\,D^\dagger|`` for any unitary ``D``.  Returns
        the largest such rounding over ``k_pts``, in the units of ``H``;
        ``0.0`` for a model with no hoppings.
        """
        if not self.n_hops:
            return 0.0
        a = np.abs(np.atleast_2d(np.asarray(k_pts, dtype=float)))
        if rotation is not None:
            a = a @ np.abs(np.asarray(rotation, dtype=float))
        M = self.nstate
        size = np.abs(self.blocks).reshape(self.n_R, M, M)
        size = size + size.swapaxes(-2, -1)           # each hopping and its conjugate
        rows = size.sum(axis=-1)                      # (n_R, M): an entry's row sum, per R
        R_abs = np.abs(self.unique_R)
        tau = np.abs(self.orb_per)
        if self.nspin != 1:
            tau = np.repeat(tau, self.nspin, axis=0)
        whole = size.sum(axis=0)                      # (M, M)
        s_R = a @ R_abs.T                             # (n, n_R): |q|.|R|
        s_tau = a @ tau.T                             # (n, M): |q|.|tau|
        bound = (s_R @ rows                           # the lattice vectors' phases
                 + s_tau * whole.sum(axis=-1)[None, :]  # the row's own center
                 + s_tau @ whole.T)                   # each column's center
        return float(2.0 * np.pi * np.finfo(float).eps * bound.max())


_kept: OrderedDict = OrderedDict()
_kept_bytes = 0


def _keep(key, bloch) -> None:
    """Keep one set of blocks, dropping the least recently used past :data:`_KEEP_BYTES`.

    ``bloch`` is the :class:`_BlochSum` and ``key`` its hash.  A set larger
    than :data:`_KEEP_BYTES` alone is not kept.
    """
    global _kept_bytes
    if bloch.nbytes > _KEEP_BYTES:
        return
    _kept[key] = bloch
    _kept_bytes += bloch.nbytes
    while _kept_bytes > _KEEP_BYTES:
        _old_key, old = _kept.popitem(last=False)
        _kept_bytes -= old.nbytes


def evaluator(model):
    """The Bloch sum of ``model``, a model with a periodic direction, built once per hash.

    The blocks are gathered from the arrays pythtb itself sums, and kept under
    their hash; a later request for the same hash, from this model or any
    other, finds them.  The sum returned carries that hash as ``key``, which is
    :func:`model_key`.
    """
    arrays = model._evaluate_params({})
    key = _digest(model, *arrays)
    bloch = _kept.get(key)
    if bloch is not None:
        _kept.move_to_end(key)
        return bloch
    bloch = _BlochSum(model, arrays, key)
    _keep(key, bloch)
    return bloch


def hamiltonian(model, k_pts, bloch=None) -> np.ndarray:
    """``model.hamiltonian(k_pts=k_pts, flatten_spin_axis=True)``, bit for bit.

    ``k_pts`` are checked as pythtb checks them, so a request pythtb rejects
    raises the same error here.  ``bloch``, where it is given, is what
    :func:`evaluator` returns for ``model``, and ``H(k)`` is summed by it.  A
    model with no periodic direction is evaluated by pythtb itself.
    """
    if int(model.lattice.dim_k) == 0:
        return model.hamiltonian(k_pts=k_pts, flatten_spin_axis=True)
    if k_pts is None:
        raise ValueError("k_pts must be specified for periodic systems (dim_k > 0).")
    k_vecs = model._normalize_kpoints(k_pts)
    if bloch is None:
        bloch = evaluator(model)
    return bloch(k_vecs)


def onsite(model) -> np.ndarray:
    """The on-site terms of ``model``: shape ``(norb,)``, or ``(norb, 2, 2)`` spinful."""
    return np.asarray(model._evaluate_params({})[4], dtype=complex)


def onsite_mean(model) -> float:
    r"""The mean on-site energy per state of ``model``, in the units of ``H``.

    ``tr H(k)`` averages over the Brillouin zone to the trace of the on-site
    terms, so a constant shift ``H → H + c·1`` moves this by exactly ``c``.  It
    is the mean energy of the bands over the Brillouin zone, the shift about
    which the magnitude ``max|H − shift·1|`` is read (:func:`magnitude`).  The
    diagonal on-site entries are summed exactly, so on-site energies that
    cancel read ``0.0``, not a rounding of it.
    """
    site = onsite(model)
    diagonal = site if site.ndim == 1 else np.diagonal(site, axis1=-2, axis2=-1)
    return math.fsum(np.real(diagonal).ravel().tolist()) / float(model.nstate)


def magnitude(H, shift=0.0) -> float:
    """The largest modulus of an entry of ``H − shift·1``, ``max|H − shift·1|``.

    ``H`` is one matrix or a stack of them, and ``shift`` an energy in the
    units of ``H``.  At ``shift = 0`` it is ``max|H|`` exactly; an empty ``H``
    gives ``0.0``.
    """
    H = np.asarray(H)
    if H.size == 0:
        return 0.0
    if shift == 0.0:
        return float(np.abs(H).max())
    out = np.abs(H)
    diagonal = np.arange(H.shape[-1])
    out[..., diagonal, diagonal] = np.abs(H[..., diagonal, diagonal] - shift)
    return float(out.max())


def solve(model, k_pts, bloch=None):
    """``model.solve_ham(k_pts, return_eigvecs=True, flatten_spin_axis=True)``, bit for bit.

    pythtb's diagonalization of the same ``H(k)``: the same Hermiticity check,
    ``numpy.linalg.eigh`` on the same matrices, eigenvectors as rows, and a
    single k-point's axis dropped.  ``model``, ``k_pts`` and ``bloch`` are as
    in :func:`hamiltonian`.  Returns ``(evals, evecs)``.
    """
    ham = hamiltonian(model, k_pts, bloch)
    if not np.allclose(ham, ham.swapaxes(-1, -2).conj()):
        raise ValueError("Hamiltonian matrix is not Hermitian.")
    evals, evecs = np.linalg.eigh(ham.astype(np.complex128))
    evecs = evecs.swapaxes(-1, -2)
    if int(model.lattice.dim_k) != 0 and evals.shape[0] == 1:
        evals, evecs = evals[0], evecs[0]
    return evals, evecs
