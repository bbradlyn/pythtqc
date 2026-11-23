r"""The spin of a spinless model that carries it in its orbital index.

A pythtb model made with ``spinful=False`` (``nspin=1``) is spinless. One made
with ``spinful=True`` (``nspin=2``) is spinful, each orbital holding two spin
states. pythtb's ``W90.model`` returns a spinless model, also for a spinor
calculation. Its Wannier functions arrive one orbital each, the spin partners
among them. Nothing in ``H(k)`` says which orbital is the spin partner of
which. :func:`spin_basis` declares the pairs. ``spin=`` of
:func:`..groups.realized_group` then reads each pair as one spinful orbital.
:func:`spin_operator` is the spin along any axis on those pairs.
:func:`spinless_presentation` writes a spinful model in the form ``W90.model``
returns, in the order chosen. The model it returns has known pairs, to compare
an import against.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import _bloch
from ._tolerance import _PRESENTATION_RELATIVE
from .refusal import InternalError, Refusal

__all__ = ["SpinBasis", "spin_basis", "spin_operator", "spinless_presentation"]

#: The Pauli matrices x, y and z on the two states of a pair, up first.
_SIGMA = (
    np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),      # sigma_x
    np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),   # sigma_y
    np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),     # sigma_z
)


def _n_dot_sigma(n):
    r"""The ``2x2`` matrix ``n·σ/|n|``, Hermitian with eigenvalues +1 and -1.

    ``n`` is the axis, a three-vector that need not be normalized.  An ``n``
    that is not a nonzero three-vector raises ``ValueError``, worded for
    :func:`spin_operator`, which passes its ``n`` here.
    """
    n = np.asarray(n, dtype=float)
    if n.shape != (3,):
        raise ValueError(f"spin_operator(): n= has shape {tuple(n.shape)}; pass a "
                         "three-vector.")
    nrm = np.linalg.norm(n)
    if nrm == 0.0:
        raise ValueError("spin_operator(): n= is zero; pass a nonzero axis.")
    return (n[0] * _SIGMA[0] + n[1] * _SIGMA[1] + n[2] * _SIGMA[2]) / nrm


@dataclass(frozen=True)
class SpinBasis:
    r"""The spin pairs of a spinless model whose orbital index carries spin.

    :func:`spin_basis` returns it, and :func:`spinless_presentation` returns
    it beside the model. ``spin=`` of :func:`..groups.realized_group` and of
    :func:`spin_operator` takes it. The record :func:`..groups.realized_group`
    returns holds the pairs it read, as its field ``spin``. ``summary()``
    returns the pairs as text, a line for each.

    Attributes
    ----------
    pairs : tuple of (int, int)
        Each pair as ``(up, down)``: the orbitals that hold its spin states
        along ``+z`` and ``-z`` of the Cartesian axes in which the pythtb
        Lattice is specified, as pythtb's spinful orbitals hold them. Every
        orbital is in exactly one pair.
    """

    # The number of orbitals of the model it describes.
    _norb: int
    # How it was written: "interleaved", the up and then the down state of each
    # pair in turn; "blocked", every up state and then every down state; or
    # "pairs", pair by pair.
    _kind: str
    pairs: tuple

    # What the declaration knows.
    @property
    def _pair_of_orbital(self) -> np.ndarray:
        """The index in ``pairs`` of each orbital's pair, by orbital."""
        out = -np.ones(int(self._norb), dtype=int)
        for c, (up, dn) in enumerate(self.pairs):
            out[int(up)] = c
            out[int(dn)] = c
        return out

    def states(self, c) -> tuple:
        """The orbitals of the pair at index ``c`` of ``pairs``, up first."""
        return tuple(int(x) for x in self.pairs[int(c)])

    @property
    def _interleaved_index(self) -> np.ndarray:
        """The interleaved index ``P`` of the pairs.

        ``P[2c + s]`` is the orbital that holds pair ``c``'s spin ``s``, ``0``
        for up and ``1`` for down. A spinful model stores that state at
        ``2c + s``. Let ``H_spinful(k)`` be the Hamiltonian of the spinful
        model the pairs present, each pair's spin innermost, and ``Π`` the
        permutation matrix with ``Π[P[j], j] = 1``. The spinless model's
        Hamiltonian is ``Π H_spinful(k) Π^T``.
        """
        P = np.empty(2 * len(self.pairs), dtype=int)
        for c, (up, dn) in enumerate(self.pairs):
            P[2 * c] = int(up)
            P[2 * c + 1] = int(dn)
        return P

    @property
    def _permutation_matrix(self) -> np.ndarray:
        """The permutation matrix ``Π`` of the interleaved index ``P``.

        ``Π[P[j], j] = 1``. The spinless model's ``H(k)`` is
        ``Π H_spinful(k) Π^T``, with ``H_spinful(k)`` the Hamiltonian of the
        spinful model the pairs present.
        """
        P = self._interleaved_index
        M = np.zeros((P.size, P.size))
        M[P, np.arange(P.size)] = 1.0
        return M

    # Reading it back.
    def summary(self) -> str:
        """The pairs as text, a line for each.

        The heading names the attribute ``pairs`` that holds them.
        """
        lines = ["each pair one Wannier function, its spin up and down along z   (.pairs)"]
        lines += [f"  pair {c}: orbital {int(up)} up, orbital {int(dn)} down"
                  for c, (up, dn) in enumerate(self.pairs)]
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return f"SpinBasis({len(self.pairs)} pairs)"


def spin_basis(model, *, order="interleaved", pairs=None) -> SpinBasis:
    r"""The spin pairs of a spinless model whose orbital index carries spin.

    A pair is two orbitals, the up and the down state of one spinful orbital.
    The pairs are read off ``order``, or given one by one in ``pairs``.

    Parameters
    ----------
    model : pythtb.TBModel
        A model made with ``spinful=False`` (``nspin=1``) whose orbital index
        carries spin, such as the model pythtb's ``W90.model`` returns for a
        spinor calculation.
    order : {"interleaved", "blocked"}, default "interleaved"
        The order of the spin states in the orbital index. ``"interleaved"``
        is the up and then the down state of each pair in turn. ``"blocked"``
        is every up state, then every down state, each in the pairs' order.
    pairs : sequence of (int, int), optional
        The pairs one by one, each ``(up, down)``, for an order that is
        neither. Every orbital is in exactly one pair. With ``pairs`` given,
        ``order`` is not read.

    Returns
    -------
    SpinBasis
        The pairs.

    Examples
    --------
    >>> from pythtb.models import kane_mele                         # doctest: +SKIP
    >>> flat, _ = spinless_presentation(kane_mele(1.0, -1.0, 0.3, 0.25))   # doctest: +SKIP
    >>> print(spin_basis(flat).summary())                            # doctest: +SKIP
    each pair one Wannier function, its spin up and down along z   (.pairs)
      pair 0: orbital 0 up, orbital 1 down
      pair 1: orbital 2 up, orbital 3 down

    Raises
    ------
    ValueError
        The model is spinful. The model has an odd number of orbitals and
        ``pairs`` is not given. ``order`` is neither ``"interleaved"`` nor
        ``"blocked"``. ``pairs`` is not a sequence of pairs, or does not hold
        every orbital exactly once.
    """
    if int(model.nspin) != 1:
        raise ValueError(
            "spin_basis(): the model is spinful (nspin=2), and realized_group reads its "
            "spin without a declaration.")
    norb = int(model.norb)

    if pairs is not None:
        arr = np.asarray(pairs, dtype=int)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(
                f"spin_basis(): pairs= has shape {tuple(arr.shape)}; pass a sequence of "
                f"(up, down) pairs of orbitals.")
        if not np.array_equal(np.sort(arr.reshape(-1)), np.arange(norb)):
            raise ValueError(
                f"spin_basis(): pairs= holds the indices {sorted(arr.reshape(-1).tolist())}; "
                f"pass pairs that hold each orbital, 0..{norb - 1}, exactly once.")
        return SpinBasis(_norb=norb, _kind="pairs",
                         pairs=tuple((int(u), int(d)) for u, d in arr))

    if norb % 2 != 0:
        raise ValueError(
            f"spin_basis(): the model has {norb} orbitals, and order={order!r} needs "
            f"an even number of them.")
    N = norb // 2
    if order == "interleaved":
        made = tuple((2 * i, 2 * i + 1) for i in range(N))
    elif order == "blocked":
        made = tuple((i, N + i) for i in range(N))
    else:
        raise ValueError(f"spin_basis(): order= is {order!r}; pass 'interleaved' or 'blocked'.")
    return SpinBasis(_norb=norb, _kind=str(order), pairs=made)


def spin_operator(spin, n=(0.0, 0.0, 1.0)) -> np.ndarray:
    r"""The spin along an axis on the orbitals of a spinless model's spin pairs.

    The matrix ``S`` returned is ``n·σ/|n|`` on the two orbitals of each
    pair, with the up orbital as the first state of σ, and zero between
    pairs. Where each pair's two orbitals are its spin-up and spin-down
    states, ``S`` is the spin along ``n`` in units of hbar/2. A spin-resolved
    calculation projects it onto a set of bands as ``P S P``, with ``P`` the
    projector onto those bands. Eigenvalues of ``P S P`` kept away from zero
    split the bands by the sign of their spin along ``n``, the split on which
    the spin Chern number is read.

    Parameters
    ----------
    spin : SpinBasis
        The pairs, from :func:`spin_basis` or :func:`spinless_presentation`.
    n : array_like, default (0, 0, 1)
        The axis, a three-vector in the Cartesian axes in which the pythtb
        Lattice is specified. It need not be normalized.

    Returns
    -------
    numpy.ndarray
        The Hermitian matrix ``S``, of shape ``(norb, norb)`` for a model of
        ``norb`` orbitals.

    Raises
    ------
    ValueError
        ``spin`` is not a :class:`SpinBasis`, or ``n`` is not a nonzero
        three-vector.
    """
    if not isinstance(spin, SpinBasis):
        raise ValueError(
            f"spin_operator(): spin= is of type {type(spin).__name__}; pass the "
            f"SpinBasis that spin_basis returns, or the second of the pair "
            f"spinless_presentation returns.")
    nsig = _n_dot_sigma(n)
    N = len(spin.pairs)
    # The two regular orders as Kronecker products, and pairs given one by one
    # scattered onto their orbitals.
    if spin._kind == "interleaved":
        return np.kron(np.eye(N, dtype=complex), nsig)
    if spin._kind == "blocked":
        return np.kron(nsig, np.eye(N, dtype=complex))
    O = np.zeros((int(spin._norb), int(spin._norb)), dtype=complex)
    for up, dn in spin.pairs:
        O[np.ix_((int(up), int(dn)), (int(up), int(dn)))] = nsig
    return O


def spinless_presentation(model, *, order="interleaved", pairs=None, params=None):
    r"""A spinful model written as a spinless one, and its spin pairs.

    Each orbital of the spinful model becomes two orbitals at its position,
    the up and the down state of one pair. The Hamiltonian returned is the
    spinful one with its states reordered. pythtb's ``W90.model`` returns a
    spinor calculation in this form. The model returned is one whose pairs
    are known, to compare such an import against.

    Parameters
    ----------
    model : pythtb.TBModel
        A model made with ``spinful=True`` (``nspin=2``).
    order : {"interleaved", "blocked"}, default "interleaved"
        The order of the spin states in the orbital index returned.
        ``"interleaved"`` is the up and then the down state of each orbital of
        ``model`` in turn, pythtb's own order of the states. ``"blocked"`` is
        every up state, then every down state, each in orbital order.
    pairs : sequence of (int, int), optional
        The pairs one by one, each ``(up, down)``, pair ``c`` holding the two
        spin states of orbital ``c`` of ``model``. Every orbital of the model
        returned is in exactly one pair. With ``pairs`` given, ``order`` is
        not read.
    params : dict, optional
        The scalar value of each parameter of the model, keyed by name:
        ``{"lam": 0.3}``.

    Returns
    -------
    (pythtb.TBModel, SpinBasis) or Refusal
        The spinless model and its spin pairs. The spinless model holds the
        Hamiltonian of ``model`` at the values in ``params``. A ``Refusal`` in
        its place is falsy.

    Examples
    --------
    >>> from pythtb.models import kane_mele                         # doctest: +SKIP
    >>> flat, sb = spinless_presentation(kane_mele(1.0, -1.0, 0.3, 0.25))  # doctest: +SKIP
    >>> flat.norb, sb.pairs                                          # doctest: +SKIP
    (4, ((0, 1), (2, 3)))

    Refusals
    --------
    ``not_a_spinful_model``
        the model is spinless. Where its orbital index carries spin,
        :func:`spin_basis` declares its pairs.

    Raises
    ------
    ValueError
        ``pairs`` does not hold one pair for each orbital of ``model``, or does
        not hold each orbital returned exactly once. ``order`` is neither
        ``"interleaved"`` nor ``"blocked"``. A name in ``params`` is not a
        parameter of the model, a parameter has no value, or a value is not a
        number.
    """
    from pythtb import Lattice, TBModel

    model = _bloch.bound(model, params)
    nspin = int(model.nspin)
    if nspin != 2:
        return Refusal(
            "not_a_spinful_model",
            f"the model is spinless (nspin={nspin}). Where its orbital index carries "
            f"spin, declare the pairs with `spin_basis(model, order=...)`.",
            value=float(nspin), threshold=2.0,
            context={"nspin": nspin,
                     "spinful": bool(getattr(model, "spinful", nspin == 2))})

    N = int(model.norb)
    dim_r = int(model.lattice.dim_r)
    if pairs is None:
        if order == "interleaved":
            layout = tuple((2 * i, 2 * i + 1) for i in range(N))
        elif order == "blocked":
            layout = tuple((i, N + i) for i in range(N))
        else:
            raise ValueError(f"spinless_presentation(): order= is {order!r}; pass "
                             "'interleaved' or 'blocked'.")
    else:
        arr = np.asarray(pairs, dtype=int)
        if arr.ndim != 2 or arr.shape != (N, 2):
            raise ValueError(
                f"spinless_presentation(): pairs= has shape {tuple(arr.shape)}; pass "
                f"shape ({N}, 2), one (up, down) pair for each of the model's {N} orbitals.")
        if not np.array_equal(np.sort(arr.reshape(-1)), np.arange(2 * N)):
            raise ValueError(
                f"spinless_presentation(): pairs= holds the indices "
                f"{sorted(arr.reshape(-1).tolist())}; pass pairs that hold each orbital "
                f"of the spinless model, 0..{2 * N - 1}, exactly once.")
        layout = tuple((int(u), int(d)) for u, d in arr)

    # Both members of a pair keep the spinor's own center.
    orb = np.asarray(model.orb_vecs, dtype=float)
    orb2 = np.zeros((2 * N, dim_r))
    for c, (up, dn) in enumerate(layout):
        orb2[up] = orb[c]
        orb2[dn] = orb[c]
    out = TBModel(Lattice(np.asarray(model.lat_vecs, dtype=float), orb2.tolist(),
                          periodic_dirs=list(model.periodic_dirs)), spinful=False)

    # The on-site blocks: the diagonal as on-site energies, the spin-flip term
    # as the hopping at R = 0 between the pair's two members.
    ons = np.asarray(model.onsite, dtype=complex).reshape(N, 2, 2)
    diag = np.zeros(2 * N)
    for c, (up, dn) in enumerate(layout):
        diag[up] = float(np.real(ons[c][0, 0]))
        diag[dn] = float(np.real(ons[c][1, 1]))
    out.set_onsite(diag.tolist(), mode="set")
    where = np.asarray(layout, dtype=int)             # (spinor, spin) -> scalar index
    flip = np.flatnonzero(ons[:, 0, 1] != 0.0)

    # Every 2x2 hopping block, scattered onto its nonzero scalar entries in one
    # pass, after the on-site spin-flip terms and in the hopping list's own
    # order. Entry (s, t) of a block goes to (pair i member s, pair j member t)
    # at the block's own R.  The layout is a bijection, so distinct entries land
    # on distinct terms, and a block at R = 0 joins two different pairs, never
    # the two members of one as a spin-flip term does: no term is written twice.
    amps, i_idx, j_idx, R = _bloch.hopping_arrays(model)
    amps = np.asarray(amps, dtype=complex).reshape(-1, 2, 2)
    h, s, t = np.nonzero(amps != 0.0)
    out._append_hops(
        np.concatenate([ons[flip, 0, 1], amps[h, s, t]]),
        np.concatenate([where[flip, 0], where[i_idx[h], s]]),
        np.concatenate([where[flip, 1], where[j_idx[h], t]]),
        np.concatenate([np.zeros((flip.size, dim_r), dtype=int), R[h]]))

    sb = (spin_basis(out, pairs=layout) if pairs is not None
          else spin_basis(out, order=order))
    # The two Hamiltonians are compared on every call, at the package's fixed
    # random k-points (_bloch.CHECK_K of them, drawn with _bloch.CHECK_SEED).
    # The amplitudes are copied and the phases are evaluated once per orbital,
    # 2N of them here against the model's N repeated for the spin, so the two
    # agree to the rounding of the phases.
    Pi = sb._permutation_matrix
    k = np.random.default_rng(_bloch.CHECK_SEED).uniform(
        -0.5, 0.5, size=(_bloch.CHECK_K, int(model.lattice.dim_k)))
    a = np.asarray(_bloch.hamiltonian(model, k))
    b = np.asarray(_bloch.hamiltonian(out, k))
    worst = float(np.abs(b - Pi @ a @ Pi.T).max())
    gate = _PRESENTATION_RELATIVE * float(np.abs(a).max())
    if worst > gate:
        raise InternalError(
            f"the spinless presentation's Hamiltonian differs from the spinful "
            f"model's, its states reordered, by {worst:.3g} over "
            f"{_bloch.CHECK_K} random k, more than the {gate:.3g} rounding allows. "
            "Report it, with the model and the call.")
    return out, sb
