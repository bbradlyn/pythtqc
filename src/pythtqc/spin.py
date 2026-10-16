r"""The spin a spinless model carries in its orbital index, as a ``wannier90`` spinor import does.

pythtb's ``wannier90`` importer returns a spinless model whatever the
calculation did, so a spin–orbit calculation with ``N`` Wannier functions
arrives as ``2N`` orbitals, the two spin states of each in the orbital index,
and nothing in ``H(k)`` says which orbital is the up partner of which.
:func:`spin_basis` declares the pairing, which ``spin=`` on
:func:`..groups.realized_group` reads as a native spinful model's spin, each
pair one Wannier function at its own centre; :func:`spin_operator` is the spin
along any axis on those pairs; and :func:`spinless_presentation` writes a
native spinful model as the importer would, the same crystal to hold a real
import against.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import _bloch
from .refusal import InternalError, Refusal

__all__ = ["SpinBasis", "spin_basis", "spin_operator", "spinless_presentation"]

#: Pauli matrices on the (innermost) spin index — the same three matrices, in
#: the same order, that ``spin_resolved_analysis`` builds its operators from.
_SIGMA = (
    np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),      # sigma_x
    np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),   # sigma_y
    np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),     # sigma_z
)


def _n_dot_sigma(n):
    r"""``n̂·σ`` as a ``2x2`` Hermitian matrix with eigenvalues ``±1``.

    ``n`` need not be normalized.  Copied from
    ``spin_resolved_analysis._n_dot_sigma`` rather than re-derived, so that
    :func:`spin_operator` and ``spin_operator_spinless`` are the same number
    and not two spellings of it.
    """
    n = np.asarray(n, dtype=float)
    if n.shape != (3,):
        raise ValueError(f"spin direction `n` must be a length-3 vector; got shape "
                         f"{tuple(n.shape)}.")
    nrm = np.linalg.norm(n)
    if nrm == 0.0:
        raise ValueError("spin direction `n` must be nonzero.")
    return (n[0] * _SIGMA[0] + n[1] * _SIGMA[1] + n[2] * _SIGMA[2]) / nrm


@dataclass(frozen=True)
class SpinBasis:
    r"""Which orbitals of a spinless model are the spin up and down states of one Wannier function.

    :func:`spin_basis` and :func:`spinless_presentation` return it;
    ``spin=`` on :func:`..groups.realized_group` and
    :func:`..symmetrization.symmetrize` takes it, as does
    :func:`spin_operator`; and the representation carries it as
    :attr:`..symmetry.SpaceGroupRep.spin`.  ``sb.summary()`` prints the pairs.

    Attributes
    ----------
    norb : int
        The number of orbitals of the model it describes.
    kind : {"interleaved", "blocked", "pairs"}
        How it was written: ``(0↑, 0↓, 1↑, 1↓, …)``, ``(0↑, 1↑, …, 0↓, 1↓, …)``,
        or pair by pair.
    pairs : tuple of (int, int)
        ``(up, down)`` for each spinor Wannier function, every orbital in one
        pair: the spin states along ``+z`` and ``−z`` of the model's Cartesian
        frame, as a native pythtb spinor's are.
    """

    norb: int
    kind: str
    pairs: tuple

    # -- what the declaration knows ----------------------------------------
    @property
    def pair_of_orbital(self) -> np.ndarray:
        """The pair each orbital belongs to, by orbital."""
        out = -np.ones(int(self.norb), dtype=int)
        for c, (up, dn) in enumerate(self.pairs):
            out[int(up)] = c
            out[int(dn)] = c
        return out

    @property
    def spin_of_orbital(self) -> np.ndarray:
        """Each orbital's spin along ``z``, by orbital: ``+1`` up, ``−1`` down."""
        out = np.zeros(int(self.norb), dtype=int)
        for up, dn in self.pairs:
            out[int(up)] = 1
            out[int(dn)] = -1
        return out

    def states(self, c) -> tuple:
        """The up and the down orbital of pair ``c``."""
        return tuple(int(x) for x in self.pairs[int(c)])

    @property
    def interleaved_index(self) -> np.ndarray:
        """``P``, with ``P[2c + s]`` the orbital that holds pair ``c``'s spin ``s``, ``0`` up and ``1`` down.

        A native spinful model stores that state at ``2c + s``, so with ``Π``
        the permutation matrix ``Π[P[j], j] = 1`` the spinless model's
        Hamiltonian is ``Π H_native(k) Πᵀ``.
        """
        P = np.empty(2 * len(self.pairs), dtype=int)
        for c, (up, dn) in enumerate(self.pairs):
            P[2 * c] = int(up)
            P[2 * c + 1] = int(dn)
        return P

    @property
    def permutation_matrix(self) -> np.ndarray:
        """``Π``, with ``Π[P[j], j] = 1`` for ``P`` the :attr:`interleaved_index`: the spinless model's ``H(k)`` is ``Π H_native(k) Πᵀ``."""
        P = self.interleaved_index
        M = np.zeros((P.size, P.size))
        M[P, np.arange(P.size)] = 1.0
        return M

    # -- reading it back ----------------------------------------------------
    def summary(self) -> str:
        """The pairing, pair by pair, to read back before trusting a label."""
        kind = {"pairs": "listed pair by pair"}.get(self.kind, self.kind)
        lines = [f"{kind}, on {self.norb} orbitals   (.kind, .norb)",
                 "  each pair one Wannier function, its spin up and down along z   (.pairs)"]
        lines += [f"    pair {c}: orbital {int(up)} up, orbital {int(dn)} down"
                  for c, (up, dn) in enumerate(self.pairs)]
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return f"SpinBasis({self.kind}, norb={self.norb}, {len(self.pairs)} pairs)"


def spin_basis(model, *, order="interleaved", pairs=None) -> SpinBasis:
    r"""Which orbitals of a spinless model are the spin up and down states of one Wannier function.

    Nothing in ``H(k)`` says which orbital is whose spin partner, so the pairing
    is declared, and ``realized_group(model, spin=...)`` then reads the model as
    a native spinful one, each pair a spinor Wannier function at its own centre.

    Parameters
    ----------
    model : pythtb.TBModel
        A spinless model whose orbital index carries spin, as a ``wannier90``
        spinor import's does; a native spinful model raises.
    order : {"interleaved", "blocked"}, optional
        ``"interleaved"`` (default), ``(0↑, 0↓, 1↑, 1↓, …)``, or ``"blocked"``,
        ``(0↑, 1↑, …, 0↓, 1↓, …)``.
    pairs : sequence of (int, int), optional
        The pairs one by one, up first, every orbital in exactly one: an
        irregular ordering, in place of ``order``.

    Returns
    -------
    SpinBasis
        ``sb.summary()`` prints the pairs, which ``sb.pairs`` holds.  It goes to
        ``realized_group(model, spin=sb)`` and to :func:`spin_operator`.

    Raises
    ------
    ValueError
        A native spinful model, an odd number of orbitals under ``order``, an
        ``order`` other than ``"interleaved"`` or ``"blocked"``, or ``pairs``
        that do not hold every orbital once.

    Examples
    --------
    >>> from pythtb.models import kane_mele                         # doctest: +SKIP
    >>> flat, _ = spinless_presentation(kane_mele(1.0, -1.0, 0.3, 0.25))   # doctest: +SKIP
    >>> print(spin_basis(flat).summary())                            # doctest: +SKIP
    interleaved, on 4 orbitals   (.kind, .norb)
      each pair one Wannier function, its spin up and down along z   (.pairs)
        pair 0: orbital 0 up, orbital 1 down
        pair 1: orbital 2 up, orbital 3 down
    """
    if int(model.nspin) != 1:
        raise ValueError(
            "spin_basis declares spin on a spinless model (nspin=1, spin encoded in the "
            "orbital index, e.g. a Wannier90 import); a native spinful model carries its "
            "own spin axis, which every representation of it uses without any "
            "declaration.")
    norb = int(model.norb)

    if pairs is not None:
        arr = np.asarray(pairs, dtype=int)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(
                f"pairs must be a sequence of (up, dn) index pairs, shape (M, 2); "
                f"got shape {tuple(arr.shape)}.")
        if not np.array_equal(np.sort(arr.reshape(-1)), np.arange(norb)):
            raise ValueError(
                f"pairs must partition all orbital indices 0..{norb - 1} exactly once "
                f"(norb={norb}); got {sorted(arr.reshape(-1).tolist())}.")
        return SpinBasis(norb=norb, kind="pairs",
                         pairs=tuple((int(u), int(d)) for u, d in arr))

    if norb % 2 != 0:
        raise ValueError(
            f"order={order!r} needs an even orbital count (each spinor is one (up, dn) "
            f"pair); got norb={norb}. For an irregular layout pass explicit `pairs`.")
    N = norb // 2
    if order == "interleaved":
        made = tuple((2 * i, 2 * i + 1) for i in range(N))
    elif order == "blocked":
        made = tuple((i, N + i) for i in range(N))
    else:
        raise ValueError(f"order must be 'interleaved' or 'blocked'; got {order!r}.")
    return SpinBasis(norb=norb, kind=str(order), pairs=made)


def spin_operator(spin, n=(0.0, 0.0, 1.0)) -> np.ndarray:
    r"""The spin along ``n̂``, in units of ``ħ/2``, on the orbitals of a declared spinor basis: ``n̂·σ`` on each pair.

    It is the operator a spin-resolved calculation projects onto a set of
    bands, ``P (n̂·σ) P``: eigenvalues of that kept away from zero split the
    bands by the sign of their spin along ``n̂``, the split the spin Chern
    number is read on.  It is the matrix
    ``spin_resolved_analysis.spin_operator_spinless`` builds from the same
    pairs, entry for entry.

    Parameters
    ----------
    spin : SpinBasis
        The pairs, from :func:`spin_basis` or :func:`spinless_presentation`.
    n : array_like, optional
        The axis ``n̂``, in the model's Cartesian frame, not necessarily
        normalized.  Default ``(0, 0, 1)``, ``σ_z``.

    Returns
    -------
    numpy.ndarray
        The Hermitian ``norb × norb`` matrix: ``n̂·σ`` on each pair's two
        orbitals, the up one first, and zero between pairs.

    Raises
    ------
    ValueError
        ``spin`` is not a :class:`SpinBasis`, or ``n`` is not a nonzero
        three-vector.
    """
    if not isinstance(spin, SpinBasis):
        raise ValueError(
            f"spin_operator takes a SpinBasis, from `spin_basis(model, ...)` or "
            f"`spinless_presentation`; got {type(spin).__name__}.")
    nsig = _n_dot_sigma(n)
    N = len(spin.pairs)
    # the two regular layouts as Kronecker products and explicit pairs
    # scattered, each the expression `spin_operator_spinless` evaluates, so
    # that the two agree bit for bit
    if spin.kind == "interleaved":
        return np.kron(np.eye(N, dtype=complex), nsig)
    if spin.kind == "blocked":
        return np.kron(nsig, np.eye(N, dtype=complex))
    O = np.zeros((int(spin.norb), int(spin.norb)), dtype=complex)
    for up, dn in spin.pairs:
        O[np.ix_((int(up), int(dn)), (int(up), int(dn)))] = nsig
    return O


def spinless_presentation(model, *, order="interleaved", pairs=None, params=None):
    r"""The same crystal as a ``wannier90`` spinor import delivers it: ``2N`` orbitals, the spin in the orbital index.

    Each orbital becomes two at its own position, the up and down states of
    one pair, and the two models' Hamiltonians are measured to be the same.
    It is a trusted model to hold a real import against.

    Parameters
    ----------
    model : pythtb.TBModel
        A spinful model.
    order : {"interleaved", "blocked"}, optional
        ``"interleaved"`` (default), ``(0↑, 0↓, 1↑, 1↓, …)``, pythtb's own order
        of the states, or ``"blocked"``, ``(0↑, 1↑, …, 0↓, 1↓, …)``.
    pairs : sequence of (int, int), optional
        The pairs one by one, up first, every orbital in exactly one, in place
        of ``order``.
    params : dict, optional
        Values for a model whose terms are given by parameter names.

    Returns
    -------
    (pythtb.TBModel, SpinBasis) or Refusal
        The spinless model, and its pairing, which
        ``realized_group(flat, spin=sb)`` reads as the native model's spin.  A
        ``Refusal`` in its place is falsy.

    Raises
    ------
    ValueError
        ``pairs`` that do not hold every orbital once, or an ``order`` other
        than ``"interleaved"`` or ``"blocked"``.
    ..refusal.InternalError
        The two Hamiltonians differ, which a relabelling of the states cannot
        make them: a defect to report.

    Refusals
    --------
    ``not_a_spinful_model``
        a spinless model, which is in this presentation already;
        :func:`spin_basis` declares its pairs.

    Examples
    --------
    >>> from pythtb.models import kane_mele                         # doctest: +SKIP
    >>> flat, sb = spinless_presentation(kane_mele(1.0, -1.0, 0.3, 0.25))  # doctest: +SKIP
    >>> flat.norb, sb.kind                                           # doctest: +SKIP
    (4, 'interleaved')
    """
    from pythtb import Lattice, TBModel

    nspin = int(model.nspin)
    if nspin != 2:
        return Refusal(
            "not_a_spinful_model",
            f"the spinless presentation is built from a native spinful model (nspin=2, "
            f"spin stored innermost); this one has nspin={nspin}, so it is already in "
            f"the presentation this would produce. To declare how its orbital index "
            f"encodes spin, use `spin_basis(model, order=...)`.",
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
            raise ValueError(f"order must be 'interleaved' or 'blocked'; got {order!r}.")
    else:
        arr = np.asarray(pairs, dtype=int)
        if arr.ndim != 2 or arr.shape != (N, 2):
            raise ValueError(
                f"pairs must be one (up, dn) index pair per spinful orbital, shape "
                f"({N}, 2); got shape {tuple(arr.shape)}.")
        if not np.array_equal(np.sort(arr.reshape(-1)), np.arange(2 * N)):
            raise ValueError(
                f"pairs must partition all orbital indices 0..{2 * N - 1} exactly once; "
                f"got {sorted(arr.reshape(-1).tolist())}.")
        layout = tuple((int(u), int(d)) for u, d in arr)

    # -- the geometry: both members of a pair keep the spinor's own center --
    orb = np.asarray(model.orb_vecs, dtype=float)
    orb2 = np.zeros((2 * N, dim_r))
    for c, (up, dn) in enumerate(layout):
        orb2[up] = orb[c]
        orb2[dn] = orb[c]
    out = TBModel(Lattice(np.asarray(model.lat_vecs, dtype=float), orb2.tolist(),
                          periodic_dirs=list(model.periodic_dirs)), spinful=False)

    # -- the onsite blocks: the diagonal as onsite energies, the spin-flip
    #    term as the R = 0 hopping between the pair's two members ------------
    ons = np.asarray(model.onsite, dtype=complex).reshape(N, 2, 2)
    diag = np.zeros(2 * N)
    for c, (up, dn) in enumerate(layout):
        diag[up] = float(np.real(ons[c][0, 0]))
        diag[dn] = float(np.real(ons[c][1, 1]))
    out.set_onsite(diag.tolist(), mode="set")
    where = np.asarray(layout, dtype=int)             # (spinor, spin) -> scalar index
    flip = np.flatnonzero(ons[:, 0, 1] != 0.0)

    # -- every 2x2 hopping block, scattered onto its nonzero scalar entries --
    # One pass, after the on-site spin-flip terms and in the hopping list's own
    # order: entry (s, t) of a block goes to (pair i member s, pair j member t)
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
    # the identity, measured on every call, on the k-set every presentation
    # here is measured on
    Pi = sb.permutation_matrix
    k = np.random.default_rng(_bloch.CHECK_SEED).uniform(
        -0.5, 0.5, size=(_bloch.CHECK_K, int(model.lattice.dim_k)))
    a = np.asarray(_bloch.hamiltonian(model, k, params))
    b = np.asarray(_bloch.hamiltonian(out, k, params))
    worst = float(np.abs(b - Pi @ a @ Pi.T).max())
    if worst != 0.0:
        raise InternalError(
            f"the spinless presentation is not bit-identical to the spinful model: "
            f"max|H_flat(k) - Pi H_native(k) Pi^T| = {worst:.3g} over {_bloch.CHECK_K} "
            f"random k. The spinless presentation only relabels the state index and "
            f"changes nothing, so this is the package disagreeing with itself, not a "
            f"property of the model.")
    return out, sb
