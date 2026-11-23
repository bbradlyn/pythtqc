r"""The little-group representations of a set of bands, read off its eigenvectors.

At a k-point that an operation ``g`` leaves invariant up to a reciprocal
lattice vector, the sewing matrix ``B(g,k)_mn = <u_m(k)|U(g;k)|u_n(k)>`` is
the matrix of ``g`` on the bands.  Here ``U(g;k)`` is the operator of ``g``
on the model's states at ``k``.  For an operation combined with time
reversal, the ket is complex conjugated before ``U(g;k)`` acts.  For a
unitary operation, the trace of ``B`` over a degenerate multiplet is that
multiplet's character.  The characters of the unitary operations of the
little group identify the little-group representation the multiplet
carries: a corep in a group with time reversal, an irrep in one without.

* :func:`hsp_table`: the little-group representations a set of bands carries
  at every high-symmetry point, whose multiplicities form its symmetry data
  vector; :func:`.ebr.decompose`, :func:`.ebr.symmetry_indicator` and
  :func:`.srsi.srsi_of_bands` take the table it returns;
* :func:`coreps_at`: the little-group representations at one point, multiplet
  by multiplet;
* :func:`characters`: the characters they are read from;
* :func:`sewing_matrix`: the sewing matrix of one operation;
* :func:`solve_at`: the energies and eigenvectors at one k-point, from one
  diagonalization, to read several sets of bands there;
* :func:`multiplets`: the degenerate multiplets of a spectrum, with
  :func:`multiplet_separation`, the smallest gap between multiplets, and
  :func:`multiplet_spread`, the largest spread inside one;
* :func:`format_coreps`: the multiplicities of little-group representations
  on one line, ``2x-GM6 + -GM7``;
* :class:`Thresholds`: the two zero thresholds at a point, one fixed and one
  for a number read off eigenvectors, which grows as the smallest gap
  between distinct multiplets shrinks.

:func:`sewing_matrix`, :func:`characters`, :func:`coreps_at` and
:func:`hsp_table` take the eigenvectors as ``states=``, in pythtb's own forms:
``None``, to solve the model; a solved :class:`pythtb.WFArray`; or, at one
point, the ``(evals, evecs)`` pair of one diagonalization.  A call at one
k-point takes the point by its name in the tables of the irreptables
package, ``"GM"``, or by its reduced coordinates.  The little groups, their
character tables and the names of the little-group representations are
those of the irreptables package.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import Any, Mapping

import numpy as np

from . import _bloch
from . import _spacegroup
from . import _tables
from .refusal import InternalError, Refusal
from ._seitz import _operation_rows
from ._tolerance import DEGENERACY_TOL, FLAT_ZERO, LABEL_TOL, MESH_TOL, STATES_AT_K_GATE
from ._text import _character_text, _index_ranges, _k_text, _listed

__all__ = [
    "FLAT_ZERO", "DEGENERACY_TOL",
    "Thresholds", "two_tier_thresholds", "disagreements",
    "multiplets", "multiplet_separation", "multiplet_spread",
    "solve_at", "format_coreps",
    "SewingMatrix", "SewingMatrixNumerical", "sewing_matrix",
    "BandCharacters", "MultipletCharacters", "BandCharactersNumerical", "characters",
    "Multiplet", "CorepsAt", "CorepsAtNumerical", "coreps_at",
    "HSPTable", "hsp_table",
]

# ==========================================================================
# the two zero thresholds at one k-point
# ==========================================================================

@dataclass(frozen=True)
class Thresholds:
    """The two zero thresholds at one k-point.

    The record :func:`two_tier_thresholds` returns.  A number is treated as
    zero where its magnitude is at most a threshold.  One threshold, ``flat``,
    is fixed and makes no allowance for the gap between multiplets.  The
    other, ``effective``, is for a number read off eigenvectors, and grows as
    the smallest gap between distinct multiplets shrinks.  Both are
    dimensionless.  :func:`characters` compares the departure of each of its
    sewing matrices from unitarity with both.  ``summary()`` prints both,
    with the numbers they are computed from.

    Attributes
    ----------
    flat : float
        The fixed threshold, with no allowance for the gap between multiplets.
    noise_floor : float
        The first-order bound on the amplitude a computed eigenvector carries
        on the states of other multiplets, ``8 ε |H(k)| / Δ``.  Here ``ε`` is
        the machine precision, ``|H(k)|`` is ``norm`` and ``Δ`` is
        ``separation``.  Infinite where ``Δ`` is zero or negative, and zero
        where ``Δ`` is infinite, for a spectrum of one multiplet.
    effective : float
        The threshold for a number read off eigenvectors, the larger of
        ``flat`` and ``noise_floor``.
    norm : float
        The spectral norm of ``H(k)``, in the units of ``H``.
    separation : float
        The smallest gap between distinct multiplets, in the units of ``H``.
    degeneracy_tol : float
        The energy tolerance the multiplets were grouped at, in the units of
        ``H``.
    """

    flat: float
    noise_floor: float
    effective: float
    norm: float
    separation: float
    degeneracy_tol: float

    def zero(self, value) -> bool:
        """Whether ``|value|`` is at most ``effective``.

        ``effective`` is the threshold for a number read off eigenvectors.
        """
        return bool(float(np.abs(value)) <= self.effective)

    def zero_flat(self, value) -> bool:
        """Whether ``|value|`` is at most ``flat``, the fixed threshold."""
        return bool(float(np.abs(value)) <= self.flat)

    def disagree(self, value) -> bool:
        """Whether ``zero`` and ``zero_flat`` classify ``value`` differently."""
        return self.zero(value) != self.zero_flat(value)

    def summary(self) -> str:
        """The two thresholds and the numbers they come from, with their attributes."""
        if not np.isfinite(self.noise_floor):
            return (f"zero below {self.flat:.2g} for a number built from H(k) alone; no number "
                    "read off the eigenvectors here can be told from zero, no gap separating "
                    "distinct multiplets   (.flat, .effective, .separation)")
        if np.isinf(self.separation):
            return "\n".join([
                f"zero below {self.effective:.2g} for a number read off the eigenvectors here, "
                "as for one built from H(k) alone   (.effective, .flat)",
                "  the spectrum is one multiplet, with no other for the eigenvectors to mix "
                "with, so their noise floor is 0   (.separation, .noise_floor)"])
        return "\n".join([
            f"zero below {self.effective:.2g} for a number read off the eigenvectors here, "
            f"below {self.flat:.2g} for one built from H(k) alone   (.effective, .flat)",
            f"  the eigenvectors' noise floor {self.noise_floor:.2g} is 8 ε |H(k)| / Δ, ε the "
            f"machine precision, |H(k)| = {self.norm:.4g} and Δ = {self.separation:.4g}, the "
            "smallest gap between distinct multiplets   (.noise_floor, .norm, .separation)"])

    def __repr__(self) -> str:  # pragma: no cover: convenience only
        return (f"Thresholds(effective={self.effective:.2g}, flat={self.flat:.2g}, "
                f"nf={self.noise_floor:.2g}, |H|={self.norm:.4g}, "
                f"sep={self.separation:.4g})")

def two_tier_thresholds(hamiltonian, separation, *, flat=FLAT_ZERO,
                        degeneracy_tol=DEGENERACY_TOL) -> Thresholds:
    r"""The two zero thresholds at a k-point, from ``H(k)`` and the smallest gap.

    The noise floor is ``8 ε |H(k)| / Δ``, with ``ε`` the machine precision,
    ``|H(k)|`` the spectral norm and ``Δ`` the smallest gap between distinct
    multiplets.  The threshold for a number read off eigenvectors is the larger
    of ``flat`` and the noise floor.

    Parameters
    ----------
    hamiltonian : ndarray or float
        ``H(k)``, or its spectral norm.
    separation : float
        The smallest gap between distinct multiplets at this k, in the units
        of ``H``, as :func:`multiplet_separation` returns it.
    flat : float, default FLAT_ZERO (1e-14)
        The fixed threshold, with no allowance for the gap between
        multiplets, dimensionless.
    degeneracy_tol : float, default DEGENERACY_TOL (1e-8)
        The energy tolerance the multiplets were grouped at, in the units of
        ``H``.

    Returns
    -------
    Thresholds
        The two thresholds, with the numbers they are computed from.
        ``degeneracy_tol`` is kept in the record and enters neither
        threshold.  Where ``separation`` is zero or negative, the noise floor
        is infinite, and no number read off eigenvectors is told from zero.
        Where it is infinite, for a spectrum of one multiplet, the noise floor
        is zero and ``effective`` is ``flat``.

    Examples
    --------
    The thresholds at ``M`` of lin3g (:func:`..models.lin3g`), from ``H(k)``
    and its spectrum alone:

    >>> H = models.lin3g().hamiltonian(k_pts=[[0.5, 0.5]], flatten_spin_axis=True)[0]
    >>> two_tier_thresholds(H, multiplet_separation(np.linalg.eigvalsh(H)))
    Thresholds(effective=8.3e-14, flat=1e-14, nf=8.3e-14, |H|=12.65, sep=0.2706)
    """
    H = np.asarray(hamiltonian)
    norm = float(H) if H.ndim == 0 else float(np.linalg.norm(H, 2))
    sep = float(separation)
    if np.isnan(sep) or sep <= 0.0:
        nf = float("inf")
    else:
        # a spectrum of one multiplet, sep = inf, has no other multiplet to mix with: 0
        nf = 8.0 * float(np.finfo(float).eps) * norm / sep
    return Thresholds(flat=float(flat), noise_floor=nf, effective=max(float(flat), nf),
                      norm=norm, separation=sep, degeneracy_tol=float(degeneracy_tol))

def disagreements(thresholds: Thresholds, readouts: Mapping[str, Any]) -> tuple:
    """The numbers that the two zero thresholds at one k-point classify differently.

    Such a number is zero against ``effective``, the threshold for a number
    read off eigenvectors, and not against ``flat``.  It lies within the noise
    floor the smallest gap between multiplets sets, and cannot be told from
    zero at the point.

    Parameters
    ----------
    thresholds : Thresholds
        The thresholds at the point.
    readouts : mapping
        Numbers read off eigenvectors, ``{name: value}``.

    Returns
    -------
    tuple of dict
        A mapping for each number the two classify differently, in the order
        of ``readouts``, with four keys:
        ``"name"``, the number's name in ``readouts``;
        ``"value"``, its magnitude;
        ``"zero_conditioned"``, whether it is zero against ``effective``;
        ``"zero_flat"``, whether it is zero against ``flat``.
    """
    out = []
    for name, value in readouts.items():
        v = float(np.abs(value))
        if thresholds.disagree(v):
            out.append({"name": name, "value": v,
                        "zero_conditioned": thresholds.zero(v),
                        "zero_flat": thresholds.zero_flat(v)})
    return tuple(out)

# ==========================================================================
# the degenerate multiplets of a spectrum
# ==========================================================================


def _degeneracy(rep, degeneracy_tol) -> float:
    """The energy tolerance within which bands are one degenerate multiplet.

    ``degeneracy_tol`` is the tolerance passed, in the units of ``H``, or
    ``None``.  Returns it where it is given, and otherwise ``1e-8`` times
    ``max|H|`` of ``rep``'s model, or ``1e-8`` where ``H`` vanishes.
    """
    if degeneracy_tol is not None:
        return float(degeneracy_tol)
    scale = rep._scale
    return DEGENERACY_TOL * scale if scale > 0.0 else DEGENERACY_TOL

def _resolved(energies, degeneracy_tol):
    """Energies at the resolution the multiplets were grouped at.

    Each of ``energies`` is rounded to the nearest multiple of
    ``degeneracy_tol``, and returned as it is where that is zero.
    """
    tol = float(degeneracy_tol)
    e = np.asarray(energies, dtype=float)
    return (np.round(e / tol) * tol if tol > 0.0 else e) + 0.0

def _clusters(values, degeneracy_tol):
    """The clusters of a sorted spectrum, each at most ``degeneracy_tol`` wide.

    ``values`` are energies in ascending order.  A value joins a cluster where
    it lies within ``degeneracy_tol`` of the cluster's first value, so every
    two values of a cluster lie within ``degeneracy_tol`` of each other.
    Returns the indices of each cluster, the lowest first.
    """
    out = [[0]]
    for i in range(1, len(values)):
        if values[i] - values[out[-1][0]] <= degeneracy_tol:
            out[-1].append(i)
        else:
            out.append([i])
    return tuple(tuple(c) for c in out)

def _ascending(energies, call):
    """``energies`` as a flat array, checked to be ascending.

    ``call`` is the public call's name, which the ``ValueError`` raised for
    an empty spectrum or one not in ascending order opens with.
    """
    w = np.asarray(energies, dtype=float).reshape(-1)
    if w.size == 0:
        raise ValueError(f"{call}(): no energy is passed. Pass at least one energy.")
    if np.any(np.diff(w) < 0.0):
        raise ValueError(f"{call}(): the energies are not in ascending order. Sort them in "
                         "ascending order, and any eigenvectors with them.")
    return w

def multiplets(energies, degeneracy_tol=DEGENERACY_TOL) -> tuple:
    """The degenerate multiplets of a spectrum, at an energy tolerance.

    The bands are grouped from the lowest up.  A band joins a multiplet where
    it lies within ``degeneracy_tol`` of the multiplet's lowest band, so every
    two bands of a multiplet lie within ``degeneracy_tol`` of each other.
    :func:`characters`, :func:`coreps_at` and :func:`hsp_table` group the
    bands they read the same way.  The grouping is the same at any
    tolerance at least the largest spread inside a multiplet,
    :func:`multiplet_spread`, and below the smallest gap between multiplets,
    :func:`multiplet_separation`.

    Parameters
    ----------
    energies : array_like, shape (nstate,)
        Energies in ascending order, as :func:`solve_at`, ``solve_ham`` and
        ``WFArray.energies`` return them.
    degeneracy_tol : float, default DEGENERACY_TOL (1e-8)
        The energy tolerance, in the units of the spectrum.

    Returns
    -------
    tuple of tuple of int
        The indices of the bands of each multiplet, the lowest first.

    Raises
    ------
    ValueError
        An empty spectrum, or one not in ascending order.
    """
    return _clusters(_ascending(energies, "multiplets"), float(degeneracy_tol))

def multiplet_separation(energies, degeneracy_tol=DEGENERACY_TOL) -> float:
    """The smallest energy gap between distinct multiplets of a spectrum.

    The multiplets are those :func:`multiplets` returns.  The gap between two
    consecutive multiplets is the lowest energy of the upper one less the
    highest energy of the lower one.  The smallest gap enters the noise floor
    of :func:`two_tier_thresholds`, below which a number read off the
    eigenvectors is treated as zero.

    Parameters
    ----------
    energies : array_like, shape (nstate,)
        Energies in ascending order, as :func:`solve_at`, ``solve_ham`` and
        ``WFArray.energies`` return them.
    degeneracy_tol : float, default DEGENERACY_TOL (1e-8)
        The energy tolerance the multiplets are grouped at, in the units of
        the spectrum, as in :func:`multiplets`.

    Returns
    -------
    float
        The smallest gap between consecutive multiplets, in the units of the
        spectrum.  ``inf`` where the spectrum is one multiplet.

    Raises
    ------
    ValueError
        An empty spectrum, or one not in ascending order.
    """
    w = _ascending(energies, "multiplet_separation")
    mult = _clusters(w, float(degeneracy_tol))
    if len(mult) <= 1:
        return float("inf")
    return float(min(w[c[0]] - w[p[-1]] for p, c in zip(mult[:-1], mult[1:])))

def multiplet_spread(energies, degeneracy_tol=DEGENERACY_TOL) -> float:
    """The largest energy spread inside any one multiplet of a spectrum.

    The multiplets are those :func:`multiplets` returns, and the spread of a
    multiplet is its highest energy less its lowest.  The grouping is the same
    at any tolerance at least this spread and below
    :func:`multiplet_separation`.

    Parameters
    ----------
    energies : array_like, shape (nstate,)
        Energies in ascending order, as :func:`solve_at`, ``solve_ham`` and
        ``WFArray.energies`` return them.
    degeneracy_tol : float, default DEGENERACY_TOL (1e-8)
        The energy tolerance the multiplets are grouped at, in the units of
        the spectrum, as in :func:`multiplets`.

    Returns
    -------
    float
        The largest spread inside a multiplet, in the units of the spectrum.

    Raises
    ------
    ValueError
        An empty spectrum, or one not in ascending order.
    """
    w = _ascending(energies, "multiplet_spread")
    return float(max(w[c[-1]] - w[c[0]] for c in _clusters(w, float(degeneracy_tol))))

# ==========================================================================
# resolving the single eigenvector source
# ==========================================================================

def solve_at(model, k, *, params=None):
    r"""The energies and eigenvectors at one k-point, from one diagonalization.

    The pair is handed to :func:`sewing_matrix`, :func:`characters` or
    :func:`coreps_at` as ``states=``, to read several sets of bands at that
    k-point off the same states.

    Parameters
    ----------
    model : pythtb.TBModel
        The model, ``r.model``.
    k : array_like, shape (dim_k,)
        One k-point, in reduced coordinates.
    params : dict, optional
        The value of each parameter of the model, keyed by the parameter's
        name, ``{"lam": 0.3}``.  Pass it for a model whose terms depend on
        parameters.

    Returns
    -------
    (ndarray, ndarray)
        ``(evals, evecs)``: the energies in ascending order, and the
        eigenvectors as rows.  They are what ``model.solve_ham`` returns with
        ``return_eigvecs=True`` and ``flatten_spin_axis=True``, for the model
        with its parameters set to ``params``.

    Raises
    ------
    ValueError
        A name in ``params`` that is not a parameter of the model, a parameter
        given no value, or a value that is not a number.
    """
    model = _bloch.bound(model, params, call="solve_at")
    return _bloch.solve(model, np.atleast_2d(np.asarray(k, dtype=float).reshape(-1)))

def _as_columns(call, states) -> np.ndarray:
    """The eigenvectors at one k-point as the columns of a matrix ``V``.

    ``call`` is the public call's name, and ``states`` holds the eigenvectors
    as rows, in either of pythtb's shapes: ``(nband, norb*nspin)``, as
    ``solve_ham`` returns them with ``flatten_spin_axis=True``, or
    ``(nband, norb, nspin)``, as :attr:`pythtb.WFArray.wfs` holds them.
    Returns ``V``, with ``H V = V diag(w)`` for eigenvectors of ``H``.  Raises
    ``ValueError``, opening with ``call``, for any other shape.
    """
    arr = np.asarray(states)
    if arr.ndim == 3:
        arr = arr.reshape(arr.shape[0], -1)
    if arr.ndim != 2:
        raise ValueError(
            f"{call}(): states= holds eigenvectors of shape {np.asarray(states).shape}. "
            f"Pass them as rows, of shape (n, norb*nspin) or (n, norb, nspin).")
    return np.ascontiguousarray(arr.T)

def _flat_k_points(call, wf) -> np.ndarray:
    """The k-points of the mesh of the ``WFArray`` ``wf``, a row each.

    The rows stand in the order the mesh's states are stored.  Raises
    ``ValueError``, opening with ``call``, the public call's name, where the
    mesh has parameter axes.
    """
    mesh = wf.mesh
    if mesh.dim_lambda:
        raise ValueError(
            f"{call}(): states= is a WFArray whose mesh has parameter axes, dim_lambda = "
            f"{mesh.dim_lambda}. Pass a WFArray on a mesh of k-points alone.")
    return np.asarray(wf.k_points, dtype=float).reshape(-1, int(wf.dim_k))

def _wfarray_fiber(call, wf, k, *, mesh_tol=MESH_TOL):
    """The energies and eigenvectors at ``k`` that a solved ``WFArray`` holds.

    ``call`` is the public call's name, ``wf`` the ``WFArray`` and ``k`` the
    point, in reduced coordinates.  The point of the mesh nearest ``k`` is
    read where each of its coordinates lies within ``mesh_tol`` of ``k``'s.
    Returns ``(w, V)``: ``w`` the energies there, or ``None`` where ``wf``
    holds states without energies, and ``V`` the eigenvectors as columns
    (:func:`_as_columns`).  Returns the refusal ``k_not_on_mesh`` where no
    point of the mesh is that close.  The model is never solved.  Raises
    ``ValueError``, opening with ``call``, where the mesh and ``k`` have
    different numbers of coordinates.
    """
    k = np.asarray(k, dtype=float).reshape(-1)
    pts = _flat_k_points(call, wf)
    if pts.shape[1] != k.size:
        raise ValueError(f"{call}(): states= is a WFArray whose mesh is "
                         f"{pts.shape[1]}-dimensional in k, and the point asked has "
                         f"{k.size} coordinates.")
    dist = np.abs(pts - k).max(axis=1)
    j = int(np.argmin(dist))
    if float(dist[j]) > float(mesh_tol):
        return Refusal(
            "k_not_on_mesh",
            f"the mesh of the WFArray does not hold k = {np.round(k, 8).tolist()}; the "
            f"nearest of its {pts.shape[0]} points is {np.round(pts[j], 8).tolist()}, at "
            f"distance {float(dist[j]):.3g}. Build a mesh that holds this point (k_grid "
            f"says which high-symmetry points a mesh holds), or leave states= out to solve "
            f"the model there.",
            value=float(dist[j]), threshold=float(mesh_tol),
            context={"k": k, "nearest": pts[j], "npoints": int(pts.shape[0])})
    idx = np.unravel_index(j, tuple(wf.shape_mesh))
    V = _as_columns(call, np.asarray(wf.wfs)[idx])
    try:
        w = np.asarray(wf.energies, dtype=float)[idx].reshape(-1)
    except ValueError:
        w = None  # a WFArray not filled by solve_model holds no energies
    return w, V

def _resolve_states(call, source, rep, k, *, mesh_tol=MESH_TOL, bloch=None):
    """The energies and eigenvectors at ``k``, from a call's ``states=``.

    ``call`` is the public call's name, which every ``ValueError`` raised
    opens with.  ``source`` is what the call was handed: ``None`` or
    ``rep``'s model, to solve the model at ``k``; a solved
    :class:`pythtb.WFArray`; the ``(evals, evecs)`` pair
    :meth:`pythtb.TBModel.solve_ham` returns; or eigenvector rows alone.
    ``rep`` is the representation, ``k`` the point in reduced coordinates,
    ``mesh_tol`` as in :func:`_wfarray_fiber`, and ``bloch`` as in
    :func:`_hamiltonian_at`.  Returns ``(w, V)``: ``w`` the energies, or
    ``None`` for states without them, and ``V`` the eigenvectors as columns;
    or the refusal :func:`_wfarray_fiber` returns.  Raises ``ValueError`` for
    a ``WFArray`` solved on another model, for a model other than ``rep``'s,
    and for a pair whose energies and eigenvectors differ in number.
    """
    k = np.asarray(k, dtype=float).reshape(-1)

    if source is None or source is rep._model:
        evals, evecs = _bloch.solve(rep._model, np.atleast_2d(k), bloch)
        return np.asarray(evals, dtype=float).reshape(-1), _as_columns(call, evecs)

    if hasattr(source, "wfs") and hasattr(source, "mesh"):
        model = getattr(source, "_model", None)
        if model is not None and model is not rep._model:
            raise ValueError(
                f"{call}(): states= is a WFArray of a model other than r.model. Pass a "
                "WFArray of r.model, the model realized_group measured, or leave states= "
                "out.")
        return _wfarray_fiber(call, source, k, mesh_tol=mesh_tol)

    if isinstance(source, (tuple, list)) and len(source) == 2:
        evals, evecs = source
        w, V = np.asarray(evals, dtype=float).reshape(-1), _as_columns(call, evecs)
        if w.size != V.shape[1]:
            raise ValueError(
                f"{call}(): the energies of states= number {w.size}, and its eigenvectors "
                f"{V.shape[1]}. Pass one energy for each eigenvector, as solve_at returns "
                "them.")
        return w, V

    if hasattr(source, "solve_ham"):
        raise ValueError(
            f"{call}(): states= is a model other than r.model. Pass r.model, the model "
            "realized_group measured, or leave states= out.")

    return None, _as_columns(call, source)

def _hamiltonian_at(rep, k, bloch=None) -> np.ndarray:
    """``H(k)`` of ``rep``'s model, made exactly Hermitian.

    ``k`` is the point, in reduced coordinates.  ``bloch``, where it is given,
    is the model's Bloch sum, as :func:`.._bloch.evaluator` returns it.
    Returns ``(H + H†) / 2``.
    """
    k = np.asarray(k, dtype=float).reshape(-1)
    H = np.asarray(_bloch.hamiltonian(rep._model, np.atleast_2d(k), bloch))[0]
    return 0.5 * (H + H.conj().T)

def _measured_sum(rep, bloch=None, *, call):
    """The Bloch sum of ``rep``'s model, checked against ``rep``.

    A ``bloch`` handed in has been checked already, and is returned as it is.
    Otherwise the sum is built, and ``ValueError`` is raised where the model
    has been edited since it was measured.  ``call`` is the public call's
    name.
    """
    if bloch is None:
        bloch = _bloch.evaluator(rep._model)
        rep._check_unedited(bloch, call=call)
    return bloch

def _states_at_k(H, w, V):
    """How far states are from solving ``H``, ``max |H V - V diag(w)|``.

    ``H`` is ``H(k)``, ``V`` the states as columns and ``w`` their energies.
    """
    return float(np.abs(H @ V - V * np.asarray(w, dtype=float)[None, :]).max())

def _foreign_states(k, residual, scale):
    """The refusal ``states_not_at_k``, or ``None`` where states solve ``H(k)``.

    ``residual`` is :func:`_states_at_k` at the point ``k``, and ``scale`` the
    model's ``max|H|``.  The states solve ``H(k)`` where the residual is at
    most ``1e-8`` times ``scale``, or ``1e-8`` where ``scale`` is zero.
    """
    gate = STATES_AT_K_GATE * (float(scale) if scale > 0.0 else 1.0)
    if residual <= gate:
        return None
    return Refusal(
        "states_not_at_k",
        f"the states passed do not solve H(k) at "
        f"k = {np.round(np.asarray(k, float).reshape(-1), 8).tolist()}: "
        f"max |H V - V diag(w)| = {residual:.3g}, with V the states and w their energies. "
        f"Pass the states of one diagonalization at this k, as solve_at returns them, or "
        f"leave states= out to solve the model there.",
        value=float(residual), threshold=float(gate), context={"k": np.asarray(k, float)})

def _rep_of(call, realized, example):
    """The representation of what :func:`..groups.realized_group` returned.

    ``realized`` is what the public call ``call`` was handed.  Returns
    ``realized.rep``, or ``realized`` itself where it is a refusal.  Raises
    ``TypeError`` for anything else, its message opening with ``call`` and
    showing ``example``, the call written with ``r``.
    """
    from ._realized import RealizedGroup
    from .symmetry import SpaceGroupRep

    if isinstance(realized, Refusal):
        return realized
    if isinstance(realized, RealizedGroup):
        return realized.rep
    if isinstance(realized, SpaceGroupRep):
        raise TypeError(f"{call}(): the first argument is a SpaceGroupRep, such as r.rep. "
                        f"Pass what realized_group returns, as in {example}.")
    raise TypeError(f"{call}(): the first argument is of type {type(realized).__name__}. "
                    f"Pass what realized_group returns, as in {example}.")


def _band_block(nstate, bands, call=None):
    """The indices of a set of bands, ascending.

    ``nstate`` is the number of states.  ``bands`` is an integer ``n``, for
    the lowest ``n`` bands; a sequence of band indices; or ``None``, for every
    state.  Raises ``ValueError`` for a count outside ``1..nstate``, a band
    named twice, or a band outside the states.  ``call``, where it is given,
    is the public call's name, which each ``ValueError`` opens with, and an
    empty sequence raises too; without it an empty sequence returns ``()``.
    """
    nstate = int(nstate)
    opening = f"{call}(): " if call is not None else ""
    if bands is None:
        return tuple(range(nstate))
    if np.isscalar(bands):
        n = int(bands)
        if not 0 < n <= nstate:
            raise ValueError(f"{opening}bands={n} is not a count from 1 to {nstate}, the "
                             "number of states. Pass a count in that range, or the band "
                             "indices.")
        return tuple(range(n))
    idx = [int(b) for b in bands]
    if not idx and call is not None:
        raise ValueError(f"{call}(): bands= is empty. Pass the number of bands, "
                         "bands=2, or their indices, bands=[0, 1].")
    if len(set(idx)) != len(idx):
        raise ValueError(f"{opening}bands={idx} names a band more than once. Pass each "
                         "index once.")
    if idx and not (0 <= min(idx) and max(idx) < nstate):
        raise ValueError(f"{opening}bands={idx} holds an index outside 0 to {nstate - 1}, "
                         f"the {nstate} states. Pass indices in that range.")
    return tuple(sorted(idx))

# ==========================================================================
# the k-point a call on bands reads
# ==========================================================================

#: How far a point may sit from a tabulated one and be named after it, in
#: reduced reciprocal coordinates: the tables' own reading of a point.
_KTOL = _tables._corep_table.__kwdefaults__["ktol"]

def _in_plane(group, k, call):
    """The point ``k`` in reduced coordinates of the periodic directions.

    ``group`` is the group and ``k`` the point's coordinates.  Three
    coordinates of a layer's three-dimensional cell are read in its plane,
    as :func:`.._tables._little_group` reads them.  ``call``, the public
    call's name, is handed to :func:`.._tables._little_group`.  The refusal
    it returns, ``out_of_plane`` for a point off the plane, is returned as it
    is.  Any other ``k`` is returned as a flat array.
    """
    k = np.asarray(k, dtype=float).reshape(-1)
    if k.size == 3 and int(group._dim_k) != 3:
        lg = _tables._little_group(group, k, call=call)
        if isinstance(lg, Refusal):
            return lg
        return np.asarray(lg.k, dtype=float).reshape(-1)
    return k

def _point_of(call, group, k):
    """The coordinates a call on bands reads ``k`` at, or its refusal there.

    ``call`` is the public call's name, ``group`` the group, and ``k`` a
    tabulated name or coordinates.  A name is read as the coordinates
    :func:`.._tables.high_symmetry_points` gives it
    (:func:`.._tables._resolve_kpoint`, which raises ``ValueError``, opening
    with ``call``, on a name the group does not tabulate).  Coordinates of a
    layer's three-dimensional cell are read in its plane (:func:`_in_plane`).
    A refusal of :func:`.._tables._resolve_kpoint` or of :func:`_in_plane` is
    returned as it is.
    """
    k = _tables._resolve_kpoint(call, group, k)
    if isinstance(k, Refusal):
        return k
    return _in_plane(group, k, call)

def _tabulated_name(group, k):
    """The name of the point ``k`` in the tables, or ``None``.

    ``group`` is the group, and ``k`` holds ``dim_k`` reduced coordinates.
    The point is matched as the identification of the little-group
    representations matches it (:func:`.._tables._resolve_kpname`): the
    first tabulated point it differs from by a reciprocal lattice vector of
    the cell as written, each coordinate within :data:`_KTOL`.  Returns
    ``None`` where it is none of the tabulated points, for a lattice
    supergroup, and where the model's cell cannot be transformed to the
    conventional cell of the irreptables package.
    """
    if _spacegroup._lattice_not_descended(group) is not None or not group._tables_ready:
        return None
    k3 = group._k_to_3d(np.asarray(k, dtype=float).reshape(1, -1))[0]
    for name, tabulated in _tables._table_kpoints(group).items():
        if _tables._same_point(k3, tabulated, _KTOL):
            return name
    return None

def _kpoint_text(kpoint, k) -> str:
    """A k-point as the summaries print it, ``GM (0, 0)``.

    ``kpoint`` is the point's name in the tables, or ``None``, and ``k`` its
    reduced coordinates.  A point without a name, ``k = (0.1, 0.5)``, is
    printed ``k = (1/10, 1/2)``.
    """
    return f"{kpoint} {_k_text(k)}" if kpoint else f"k = {_k_text(k)}"

def _seitz_of(group, ind) -> str:
    """The Seitz symbol of one operation, ``{4+_001|0,0,0}``, or ``""``.

    ``group`` is the group and ``ind`` the operation's index in it.  Returns
    ``""`` where ``ind`` is not an operation of the group, for a lattice
    supergroup, and where the model's cell cannot be transformed to the
    conventional cell of the irreptables package.
    """
    rows = _operation_rows(group, (int(ind),))
    if isinstance(rows, Refusal) or int(ind) not in rows:
        return ""
    return str(rows[int(ind)].seitz)

# ==========================================================================
# the sewing matrix
# ==========================================================================

@dataclass(frozen=True)
class SewingMatrixNumerical:
    """How far a sewing matrix's inputs are from exact.

    Held in :attr:`SewingMatrix.numerical`.  ``summary()`` prints the numbers
    measured, together on one line.

    Attributes
    ----------
    unitarity : float
        The largest entry of ``|B†B − 1|``, with ``B`` the sewing matrix.  It
        is zero where the set of bands is closed under the operation.
    commutator : float
        The largest entry of ``|U H(k) − H(k) U|``, in the units of ``H``,
        with ``U`` the operator of the operation on the model's states at
        ``k``.  For an operation combined with time reversal it is the
        largest entry of ``|U H(k)* − H(k) U|``.  It is zero where ``U`` is a
        symmetry of ``H(k)``.
    residual : float
        If not ``nan``, the largest entry of ``|H(k) V − V diag(w)|``, in the
        units of ``H``, with ``V`` the states as columns and ``w`` their
        energies: how far the states are from eigenstates at ``k``.  ``nan``
        where the states came without energies.
    """

    unitarity: float
    commutator: float
    residual: float = float("nan")

    def summary(self) -> str:
        """The numbers measured, on one line, with the attributes that hold them."""
        measured = [f"unitary to {self.unitarity:.2g}"]
        held = [".unitarity"]
        if np.isfinite(self.commutator):
            measured.append(f"the operator commuting with H(k) to {self.commutator:.2g}")
            held.append(".commutator")
        if np.isfinite(self.residual):
            measured.append(f"the states solving H(k) to {self.residual:.2g}")
            held.append(".residual")
        return ", ".join(measured) + f"   ({', '.join(held)})"

    def __repr__(self) -> str:  # pragma: no cover: convenience only
        return (f"SewingMatrixNumerical(unitarity={self.unitarity:.2e}, "
                f"commutator={self.commutator:.2e}, residual={self.residual:.2e})")


@dataclass(frozen=True)
class SewingMatrix:
    """The sewing matrix of one operation on a set of bands at one k-point.

    The record :func:`sewing_matrix` returns.  ``summary()`` prints the
    operation, the bands, the k-point, the reciprocal lattice vector ``G`` and
    the eigenvalues.  The matrix is ``B(g,k) = V† U(g;k) V``, with ``V`` the
    states of the bands as columns and ``U(g;k)`` the operator of the
    operation ``g`` on the model's states at ``k``.  For an operation combined
    with time reversal it is ``V† U(g;k) V*``.  For a unitary operation, its
    trace over a degenerate multiplet is that multiplet's character.

    Attributes
    ----------
    operation : int
        The operation, by its index in ``r.operations``.
    seitz : str
        If nonempty, the operation's Seitz symbol, ``{4+_001|0,0,0}``, primed
        where the operation is combined with time reversal.  Empty where the
        model's cell cannot be transformed to the conventional cell of the
        irreptables package.
    bands : tuple of int
        The set of bands, by their indices among the states handed in.
    k : ndarray
        The k-point, in the model's reduced coordinates.
    kpoint : str or None
        If not ``None``, the name of the k-point in the tables of the
        irreptables package, ``"M"``.  ``None`` where ``k`` is not a tabulated
        point, or where the model's cell cannot be transformed to the
        conventional cell of the irreptables package.
    G : ndarray of int
        The reciprocal lattice vector by which the operation's image of ``k``
        differs from ``k``, in reduced coordinates.  It is ``Rk − k``, with
        ``R`` the operation's rotation acting on ``k``, or ``−Rk − k`` for an
        operation combined with time reversal.  ``U(g;k)`` carries the phase
        ``e^{2πi G·τ}`` on an orbital at ``τ``.
    matrix : ndarray, shape (nb, nb)
        The sewing matrix ``B``, ``nb`` the number of bands.
    eigenvalues : ndarray
        The eigenvalues of ``B``.  They lie on the unit circle where ``B`` is
        unitary.
    numerical : SewingMatrixNumerical
        How far ``B`` is from unitary, ``U(g;k)`` from commuting with
        ``H(k)``, and the states from eigenstates.
    """

    operation: int
    seitz: str
    bands: tuple
    k: np.ndarray
    kpoint: str | None
    G: np.ndarray
    matrix: np.ndarray
    eigenvalues: np.ndarray
    numerical: SewingMatrixNumerical

    def _named(self) -> str:
        return (f"operation {self.operation} {self.seitz}" if self.seitz
                else f"operation {self.operation}")

    def summary(self) -> str:
        """The operation, bands, k-point, ``G`` and eigenvalues, with their attributes."""
        held = ".operation, .seitz, .bands" if self.seitz else ".operation, .bands"
        held += ", .kpoint, .k" if self.kpoint else ", .k"
        lines = [f"{self._named()}, on band{'s' if len(self.bands) > 1 else ''} "
                 f"{_index_ranges(self.bands)} at {_kpoint_text(self.kpoint, self.k)}   ({held})"]
        if np.any(self.G):
            lines.append(f"  carries k to k + {_k_text(self.G)}, a reciprocal lattice vector "
                         "away   (.G)")
        if len(self.eigenvalues):
            lines.append("  its eigenvalues exp(iθ), at θ/π = "
                         + ", ".join(f"{a / np.pi:.4g}"
                                     for a in np.sort(np.angle(self.eigenvalues)) + 0.0)
                         + "   (.eigenvalues)")
        if not np.isfinite(self.numerical.residual):
            lines.append("  the states came without energies, so whether they solve H(k) and "
                         "close their multiplets was not measured   (.numerical.residual)")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover: convenience only
        return (f"SewingMatrix({self._named()}, {len(self.bands)} bands, "
                f"unitarity={self.numerical.unitarity:.2e})")

def _sewing_core(rep, operation, k, V, idx, H, *, call, atol=1e-8,
                 residual=float("nan")) -> SewingMatrix | Refusal:
    """The sewing matrix of one operation on the bands ``idx``.

    The matrix is ``B = V† U V``, or ``V† U V*`` for an operation combined
    with time reversal, with ``U`` the operation's operator on the model's
    states at ``k``.

    Parameters
    ----------
    rep : ..symmetry.SpaceGroupRep
        The representation.
    operation : int
        The operation's index.
    k : ndarray
        The point, in reduced coordinates.
    V : ndarray
        The states at ``k``, as columns.
    idx : sequence of int
        The indices of the bands among the states.
    H : ndarray or None
        ``H(k)``, or ``None``, which leaves the commutator ``nan``.
    call : str
        The public call's name.
    atol : float, default 1e-8
        How far, in reduced coordinates, the operation's image of ``k`` may
        sit from ``k`` plus a reciprocal lattice vector.
    residual : float, default nan
        How far the states are from eigenstates, kept in the record.

    Returns
    -------
    SewingMatrix or Refusal
        The record of ``B``, holding its departure from unitarity and the
        commutator of ``U`` with ``H``.  Its ``seitz`` is ``""`` and its
        ``kpoint`` ``None``; :func:`_sewing_matrix_rep` fills them in.  The
        refusal ``little_group`` where the operation does not leave ``k``
        invariant.
    """
    U = rep.sewing_operator(int(operation), k, _atol=atol, _call=call)
    if isinstance(U, Refusal):
        return U

    Vb = V[:, list(idx)]
    op = rep._operation(int(operation), call=call)
    anti = bool(op.time_reversal)
    src = np.conj(Vb) if anti else Vb
    B = Vb.conj().T @ (U @ src)
    eye = np.eye(len(idx))
    unit = float(np.abs(B.conj().T @ B - eye).max())
    if H is None:
        comm = float("nan")
    else:
        Hsrc = H.conj() if anti else H
        comm = float(np.abs(U @ Hsrc - H @ U).max())
    ev = np.linalg.eigvals(B) if len(idx) else np.zeros(0, dtype=complex)
    return SewingMatrix(operation=int(operation), seitz="", bands=tuple(idx),
                        k=np.asarray(k, dtype=float).reshape(-1), kpoint=None,
                        G=rep.in_little_group(int(operation), k, _atol=atol)[1],
                        matrix=B, eigenvalues=ev,
                        numerical=SewingMatrixNumerical(unitarity=unit, commutator=comm,
                                                        residual=float(residual)))

def _sewing_matrix_rep(rep, operation=None, k=None, *, states=None, bands=None,
                       degeneracy_tol=None, _atol=1e-8,
                       _mesh_tol=MESH_TOL) -> SewingMatrix | Refusal:
    """What :func:`sewing_matrix` returns, read off a representation.

    ``rep`` is the representation, ``r.rep`` or one built by hand.
    ``operation``, ``k``, ``states``, ``bands`` and ``degeneracy_tol`` are as
    :func:`sewing_matrix` takes them.  ``_atol`` is as ``atol`` in
    :func:`_sewing_core`, and ``_mesh_tol`` as ``mesh_tol`` in
    :func:`_wfarray_fiber`.  Returns and refuses as :func:`sewing_matrix`
    does.
    """
    call = "sewing_matrix"
    if operation is None:
        raise TypeError(f"{call}() missing required argument 'operation'")
    if k is None:
        raise TypeError(f"{call}() missing required argument 'k'")
    k = _point_of(call, rep._group, k)
    if isinstance(k, Refusal):
        return k
    degeneracy_tol = _degeneracy(rep, degeneracy_tol)
    bloch = (_measured_sum(rep, call=call) if states is None or states is rep._model
             else None)
    got = _resolve_states(call, states, rep, k, mesh_tol=_mesh_tol, bloch=bloch)
    if isinstance(got, Refusal):
        return got
    w, V = got
    idx = _band_block(V.shape[1], bands, call)

    bloch = _measured_sum(rep, bloch, call=call)
    H = _hamiltonian_at(rep, k, bloch)
    resid = float("nan") if w is None else _states_at_k(H, w, V)

    if w is not None:
        foreign = _foreign_states(k, resid, rep._scale)
        if foreign is not None:
            return foreign
        chosen = set(idx)
        for m in _clusters(_ascending(w, call), float(degeneracy_tol)):
            inside = chosen & set(m)
            if inside and len(inside) != len(m):
                return Refusal(
                    "multiplet_split",
                    f"the bands {sorted(idx)} cut the degenerate multiplet "
                    f"{list(m)} at k = {np.round(k, 6).tolist()} (energies "
                    f"{np.round(w[list(m)], 12).tolist()}). Ask for every band of the "
                    f"multiplet, or end the bands at a gap.",
                    value=float(w[m[-1]] - w[m[0]]),
                    threshold=float(degeneracy_tol),
                    context={"bands": idx, "multiplet": m, "k": k})

    out = _sewing_core(rep, operation, k, V, idx, H, call=call, atol=_atol, residual=resid)
    if isinstance(out, Refusal):
        return out
    return replace(out, seitz=_seitz_of(rep._group, operation),
                   kpoint=_tabulated_name(rep._group, k))

def sewing_matrix(realized, operation=None, k=None, *, states=None, bands=None,
                  degeneracy_tol=None,
                  _atol=_sewing_matrix_rep.__kwdefaults__["_atol"],
                  _mesh_tol=MESH_TOL) -> SewingMatrix | Refusal:
    """The sewing matrix of one operation on a set of bands at one k-point.

    The sewing matrix ``B(g,k)_mn = <u_m(k)|U(g;k)|u_n(k)>`` is the matrix of
    the operation ``g`` on the bands, at a k-point that ``g`` leaves invariant
    up to a reciprocal lattice vector.  Here ``U(g;k)`` is the operator of
    ``g`` on the model's states at ``k``.  For an operation combined with time
    reversal, the ket is complex conjugated before ``U(g;k)`` acts.  For a
    unitary operation, the trace of ``B`` over a degenerate multiplet is that
    multiplet's character.

    Parameters
    ----------
    realized : ..groups.RealizedGroup
        What :func:`..groups.realized_group` returned, ``r``.
    operation : int
        The operation, by its index in ``r.operations``.
        :func:`..groups.operation_index` finds the index from a Seitz symbol.
        Required.
    k : str or array_like, shape (dim_k,)
        The k-point.  Pass its name in the tables of the irreptables package,
        ``"M"``, or its reduced coordinates.  Required.
    states : None, pythtb.TBModel, pythtb.WFArray, (evals, evecs) or ndarray, optional
        The states at ``k``.  Pass ``None`` or ``r.model`` to solve the model
        there.  Pass a solved ``WFArray``, or the pair :func:`solve_at`
        returns, to reuse one diagonalization.  Pass the eigenvectors alone,
        as the rows of an ndarray, where no energies are at hand.  A
        ``WFArray`` carries energies only where ``WFArray.solve_model``
        filled it.
    bands : int or sequence of int, optional
        The set of bands.  Pass ``n`` for the lowest ``n`` bands, or the band
        indices.  Left out, every state handed in.
    degeneracy_tol : float, optional
        The energy tolerance, in the units of ``H``, within which bands are
        one degenerate multiplet.  Left out, ``1e-8`` times ``max|H|``, the
        largest entry of ``H(k)`` once the mean on-site energy is subtracted.

    Returns
    -------
    SewingMatrix or Refusal
        The record's ``matrix`` is the sewing matrix ``B``, and its
        ``eigenvalues`` are those of ``B``.  For a unitary operation they are
        the eigenvalues of the operation on the bands.  The record's
        ``numerical`` holds how far ``B`` is from unitary, ``U(g;k)`` from
        commuting with ``H(k)``, and the states from eigenstates.  The sewing
        matrix can only be returned at a k-point the operation leaves
        invariant up to a reciprocal lattice vector.  It can only be returned
        if the set of bands cuts no degenerate multiplet at ``k``.  For states
        without energies, the call checks neither whether the set of bands
        cuts a multiplet nor whether the states solve ``H(k)``.  A
        ``Refusal`` in its place is falsy and names what stopped the call.

    Examples
    --------
    The fourfold rotation on the lowest pair of bands of lin3g at ``M``
    (:func:`..models.lin3g`):

    >>> r = realized_group(models.lin3g())
    >>> c4 = operation_index(r.group, "{4+_001|0,0,0}")
    >>> sm = sewing_matrix(r, c4, "M", bands=2)
    >>> np.sort(np.angle(sm.eigenvalues)) / np.pi
    array([-0.75,  0.75])

    Refusals
    --------
    A refusal handed in as ``realized`` is returned as it is.

    ``little_group``
        the operation does not leave ``k`` invariant up to a reciprocal
        lattice vector.
    ``multiplet_split``
        the set of bands cuts a degenerate multiplet at ``k``.
    ``states_not_at_k``
        the states handed in do not solve ``H(k)``.
    ``k_not_on_mesh``
        the mesh of the ``WFArray`` handed in does not hold ``k``.
    ``out_of_plane``
        ``k`` has a nonzero component normal to the plane of a layer, or
        names a tabulated point off that plane.
    ``tables_not_ready``
        ``k`` is a name, and the model's cell cannot be transformed to the
        conventional cell of the irreptables package.

    Raises
    ------
    TypeError
        ``realized`` is not what :func:`..groups.realized_group` returned, or
        ``operation`` or ``k`` is left out.
    ValueError
        States of a model other than ``r.model``.
        Eigenvectors in an array that is neither two- nor three-dimensional.
        An ``(evals, evecs)`` pair with a number of energies other than that
        of eigenvectors.
        ``bands`` naming no band, a band twice, or a band outside the states.
        ``k`` a name the group does not tabulate.
        ``k`` with a number of coordinates other than ``dim_k``, or three for
        a layer.
        A ``WFArray`` with parameter axes in its mesh, or whose mesh has
        another number of coordinates than ``k``.
        ``r.model`` edited since :func:`..groups.realized_group` measured it.
    """
    rep = _rep_of("sewing_matrix", realized, "sewing_matrix(r, operation, k, bands=2)")
    if isinstance(rep, Refusal):
        return rep
    return _sewing_matrix_rep(rep, operation, k, states=states, bands=bands,
                              degeneracy_tol=degeneracy_tol, _atol=_atol,
                              _mesh_tol=_mesh_tol)

# ==========================================================================
# the characters of a set of bands
# ==========================================================================

@dataclass(frozen=True)
class MultipletCharacters:
    """The characters of one degenerate multiplet.

    A row of :attr:`BandCharacters.multiplets`.

    Attributes
    ----------
    bands : tuple of int
        The bands of the multiplet.
    energy : float
        The mean energy of the multiplet, in the units of ``H``.
    characters : dict
        The trace over the multiplet of the sewing matrix of each unitary
        operation of the little group, ``{index: character}``, keyed by the
        operation's index in ``r.operations``.
    """

    bands: tuple
    energy: float
    characters: dict
    # the energy tolerance the multiplet was grouped at, which its energy is
    # printed at
    _degeneracy_tol: float = field(default=DEGENERACY_TOL, repr=False, compare=False)

    def __repr__(self) -> str:  # pragma: no cover: convenience only
        energy = float(_resolved([self.energy], self._degeneracy_tol)[0])
        return (f"MultipletCharacters(band{'s' if len(self.bands) > 1 else ''} "
                f"{_index_ranges(self.bands)} at {energy:.4g})")


@dataclass(frozen=True)
class BandCharactersNumerical:
    """How far the inputs of a set of bands' characters are from exact.

    Held in :attr:`BandCharacters.numerical`.  ``summary()`` prints the
    largest departures from unitarity, from commuting with ``H(k)`` and from
    eigenstates.  It then names the operations whose departure from unitarity
    the two zero thresholds classify differently.

    Attributes
    ----------
    unitarity : dict
        For each unitary operation of the little group, by its index, the
        largest entry of ``|B†B − 1|``, with ``B`` its sewing matrix on the
        whole set of bands.
    commutators : dict
        For each unitary operation of the little group, by its index, the
        largest entry of ``|U H(k) − H(k) U|``, with ``U`` its operator on the
        model's states, in the units of ``H``.
    residual : float
        The largest entry of ``|H(k) V − V diag(w)|``, in the units of ``H``,
        with ``V`` the states as columns and ``w`` their energies: how far the
        states are from eigenstates at ``k``.
    spread : float
        The largest energy spread inside a degenerate multiplet of the
        spectrum, in the units of ``H``.
    thresholds : Thresholds
        The two zero thresholds at ``k``.
    disagreements : tuple of dict
        If nonempty, the departures from unitarity that the two thresholds
        classify differently, by operation, as :func:`disagreements` returns
        them.
    degeneracy_tol : float
        The energy tolerance the multiplets were grouped at, in the units of
        ``H``.
    spectrum : ndarray
        Every energy of the diagonalization the characters were read from.
    """

    unitarity: dict
    commutators: dict
    residual: float
    spread: float
    thresholds: Thresholds
    disagreements: tuple
    degeneracy_tol: float
    spectrum: np.ndarray

    def summary(self) -> str:
        """The largest departures from exact, with the attributes that hold them."""
        unitary = max(self.unitarity.values(), default=0.0)
        commuting = max(self.commutators.values(), default=0.0)
        lines = [f"the sewing matrices unitary to {unitary:.2g}, the operators commuting "
                 f"with H(k) to {commuting:.2g}, the states solving H(k) to "
                 f"{self.residual:.2g}   (.unitarity, .commutators, .residual)"]
        if self.disagreements:
            named = _listed(str(d["name"]).split("#")[-1].rstrip("]")
                            for d in self.disagreements)
            lines.append(f"  operation{'s' if len(self.disagreements) > 1 else ''} {named}: the "
                         "sewing matrix's departure from unitarity is zero against the "
                         "eigenvectors' threshold here and not against the flat one   "
                         "(.disagreements, .thresholds)")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover: convenience only
        return (f"BandCharactersNumerical(residual={self.residual:.2e}, "
                f"spread={self.spread:.2e}, {len(self.disagreements)} disagreements)")


@dataclass(frozen=True)
class BandCharacters:
    """The characters of a set of bands at one k-point.

    The record :func:`characters` returns.  A character is the trace of the
    sewing matrix of a unitary operation of the little group over one
    degenerate multiplet.  ``summary()`` prints the characters as a table, a
    column for each multiplet and a line for each operation.
    :func:`coreps_at` identifies the little-group representations from them.

    Attributes
    ----------
    bands : tuple of int
        The set of bands, by their indices among the states handed in,
        ascending.
    k : ndarray
        The k-point, in the model's reduced coordinates.
    kpoint : str or None
        If not ``None``, the name of the k-point in the tables of the
        irreptables package, ``"GM"``.  ``None`` where ``k`` is not a
        tabulated point, or where the model's cell cannot be transformed to
        the conventional cell of the irreptables package.
    multiplets : tuple of MultipletCharacters
        A row for each degenerate multiplet of the set of bands, lowest
        first, with its bands, its mean energy and its characters.
    operations : ..groups.OperationTable or Refusal
        The operations of the little group, unitary and antiunitary, by their
        index in ``r.operations``, each with its Seitz symbol.  The
        characters in ``multiplets`` are those of the unitary operations
        alone.  A ``Refusal`` where the model's cell cannot be transformed to
        the conventional cell of the irreptables package.
    numerical : BandCharactersNumerical
        How far the sewing matrices are from unitary, the operators of the
        operations on the model's states from commuting with ``H(k)``, and
        the states from eigenstates.  It also holds the zero thresholds at
        ``k``.
    """

    bands: tuple
    k: np.ndarray
    kpoint: str | None
    multiplets: tuple
    numerical: BandCharactersNumerical
    # the little group at k, whose operations `operations` writes out when it
    # is first read
    _little_group: Any = field(default=None, repr=False, compare=False)
    # "realized" for a group measured on H, "descended" for one measured after
    # the cell was lowered
    _group_provenance: str = field(default="cell", repr=False, compare=False)

    @cached_property
    def operations(self):
        """The operations of the little group, unitary and antiunitary, by index."""
        return None if self._little_group is None else self._little_group.operations

    def _seitz(self, ind) -> str:
        """The Seitz symbol of operation ``ind``, or ``""`` where none is written."""
        table = self.operations
        if table is None or isinstance(table, Refusal) or int(ind) not in table:
            return ""
        return str(table[int(ind)].seitz)

    def summary(self) -> str:
        """The characters as a table, with the attributes that hold its parts."""
        held = ".bands, .kpoint, .k" if self.kpoint else ".bands, .k"
        lines = [f"band{'s' if len(self.bands) > 1 else ''} {_index_ranges(self.bands)} at "
                 f"{_kpoint_text(self.kpoint, self.k)}   ({held})"]
        columns = tuple(self.multiplets[0].characters) if self.multiplets else ()
        if columns:
            lines.append("  the trace of each unitary operation of the little group, by index, "
                         "over each degenerate multiplet   (.multiplets: characters of each; "
                         ".operations)")
            energies = _resolved([m.energy for m in self.multiplets],
                                 self.numerical.degeneracy_tol)
            rows = [("bands", [_index_ranges(m.bands) for m in self.multiplets],
                     "   (.multiplets: bands of each)"),
                    ("energy", [f"{e:.4g}" for e in energies],
                     "   (.multiplets: energy of each)")]
            rows += [(f"{int(ind):>3}  {self._seitz(ind)}",
                      [_character_text(m.characters[ind]) for m in self.multiplets], "")
                     for ind in columns]
            head = max(len(label) for label, _cells, _end in rows)
            width = max(len(cell) for _label, cells, _end in rows for cell in cells) + 2
            lines += ["    " + label.ljust(head) + "".join(cell.rjust(width) for cell in cells)
                      + end for label, cells, end in rows]
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover: convenience only
        n = len(self.multiplets)
        return (f"BandCharacters({_kpoint_text(self.kpoint, self.k)}, bands "
                f"{_index_ranges(self.bands)} in {n} multiplet{'' if n == 1 else 's'})")

def _characters_rep(rep, k, *, states=None, bands=None, degeneracy_tol=None,
                    _atol=1e-8, _mesh_tol=MESH_TOL) -> BandCharacters | Refusal:
    """What :func:`characters` returns, read off a representation.

    ``rep`` is the representation, ``r.rep`` or one built by hand.  ``k``,
    ``states``, ``bands`` and ``degeneracy_tol`` are as :func:`characters`
    takes them.  ``_atol`` is as ``atol`` in :func:`_sewing_core`, and
    ``_mesh_tol`` as ``mesh_tol`` in :func:`_wfarray_fiber`.  Returns and
    refuses as :func:`characters` does.
    """
    return _characters(rep, k, states=states, bands=bands, degeneracy_tol=degeneracy_tol,
                       atol=_atol, mesh_tol=_mesh_tol, bloch=None, call="characters")

def characters(realized, k, *, states=None, bands=None, degeneracy_tol=None,
               _atol=_characters_rep.__kwdefaults__["_atol"],
               _mesh_tol=MESH_TOL) -> BandCharacters | Refusal:
    """The characters of a set of bands at one k-point.

    The states at ``k`` are grouped into degenerate multiplets by energy, as
    :func:`multiplets` groups them.  The sewing matrix of each unitary
    operation of the little group at ``k`` is built on the set of bands, and
    its trace over each multiplet is that multiplet's character.  ``k`` may
    be any point, whether or not the tables of the irreptables package list
    it.

    Parameters
    ----------
    realized : ..groups.RealizedGroup
        What :func:`..groups.realized_group` returned, ``r``.
    k : str or array_like, shape (dim_k,)
        The k-point.  Pass its name in the tables, ``"GM"``, or its reduced
        coordinates.  For a layer, the three reduced coordinates of a point of
        its plane, in its three-dimensional cell, are also accepted.
    states : None, pythtb.TBModel, pythtb.WFArray or (evals, evecs), optional
        The states at ``k``, with their energies.  Pass ``None`` or
        ``r.model`` to solve the model there.  Pass a ``WFArray`` filled by
        ``solve_model``, or the pair :func:`solve_at` returns, to reuse one
        diagonalization.
    bands : int or sequence of int
        The set of bands.  Pass ``n`` for the lowest ``n`` bands, or the band
        indices.  Required.
    degeneracy_tol : float, optional
        The energy tolerance, in the units of ``H``, within which bands are
        one degenerate multiplet.  Left out, ``1e-8`` times ``max|H|``, the
        largest entry of ``H(k)`` once the mean on-site energy is subtracted.

    Returns
    -------
    BandCharacters or Refusal
        The characters, a row for each degenerate multiplet of the set of
        bands, lowest first, and an entry in each row for each unitary
        operation of the little group.  The record's ``numerical`` holds how
        far the inputs of the characters are from exact.  Characters can only
        be returned if the set of bands contains all the states of each
        degenerate multiplet at ``k``.  A ``Refusal`` in its place is falsy
        and names what stopped the call.

    Refusals
    --------
    A refusal handed in as ``realized`` is returned as it is.

    ``multiplet_split``
        the set of bands cuts a degenerate multiplet at ``k``.
    ``states_not_at_k``
        the states handed in do not solve ``H(k)``.
    ``k_not_on_mesh``
        the mesh of the ``WFArray`` handed in does not hold ``k``.
    ``out_of_plane``
        ``k`` has a nonzero component normal to the plane of a layer, or
        names a tabulated point off that plane.
    ``tables_not_ready``
        ``k`` is a name, and the model's cell cannot be transformed to the
        conventional cell of the irreptables package.

    Raises
    ------
    TypeError
        ``realized`` is not what :func:`..groups.realized_group` returned, or
        ``bands`` is left out.
    ValueError
        States without energies: eigenvectors alone, or a ``WFArray`` not
        filled by ``solve_model``.
        States of a model other than ``r.model``.
        Eigenvectors in an array that is neither two- nor three-dimensional.
        An ``(evals, evecs)`` pair with a number of energies other than that
        of eigenvectors.
        ``bands`` naming no band, a band twice, or a band outside the states.
        ``k`` a name the group does not tabulate.
        ``k`` with a number of coordinates other than ``dim_k``, or three for
        a layer.
        A ``WFArray`` with parameter axes in its mesh, or whose mesh has
        another number of coordinates than ``k``.
        ``r.model`` edited since :func:`..groups.realized_group` measured it.

    See Also
    --------
    coreps_at : the little-group representations of a set of bands at a
        high-symmetry point, identified from these characters.
    """
    rep = _rep_of("characters", realized, "characters(r, k, bands=6)")
    if isinstance(rep, Refusal):
        return rep
    return _characters_rep(rep, k, states=states, bands=bands, degeneracy_tol=degeneracy_tol,
                           _atol=_atol, _mesh_tol=_mesh_tol)

def _characters(rep, k, *, states, bands, degeneracy_tol, atol, mesh_tol,
                bloch, call, named=True) -> BandCharacters | Refusal:
    """The characters :func:`characters` returns.

    ``rep`` is the representation, or a refusal, which is returned as it is.
    ``k``, ``states``, ``bands`` and ``degeneracy_tol`` are as
    :func:`characters` takes them, ``atol`` as in :func:`_sewing_core`, and
    ``mesh_tol`` as in :func:`_wfarray_fiber`.  ``bloch`` is the model's Bloch
    sum or ``None``, as in :func:`_hamiltonian_at`.  ``call`` names the public
    call in what is raised.  With ``named=False`` the point's name is left
    ``None``, to be read off the identification of the little-group
    representations.  Raises :class:`~.refusal.InternalError` where ``rep``
    leaves out a unitary operation of the little group the tables list at
    ``k``.
    """
    if bands is None:
        raise TypeError(f"{call}() missing required argument 'bands'")
    if isinstance(rep, Refusal):
        return rep
    gate = _spacegroup._lattice_not_descended(rep._group)
    if gate is not None:
        return gate
    k = _point_of(call, rep._group, k)
    if isinstance(k, Refusal):
        return k
    if states is None or states is rep._model:
        bloch = _measured_sum(rep, bloch, call=call)
    got = _resolve_states(call, states, rep, k, mesh_tol=mesh_tol, bloch=bloch)
    if isinstance(got, Refusal):
        return got
    w, V = got
    degeneracy_tol = _degeneracy(rep, degeneracy_tol)
    if w is None:
        # hsp_table takes the model or a WFArray; the other calls take a pair too
        takes = ("r.model, or a WFArray filled by solve_model" if call == "hsp_table" else
                 "the model, or the (evals, evecs) pair that solve_at returns")
        raise ValueError(
            f"{call}(): states= holds eigenvectors without their energies. Pass {takes}; "
            "sewing_matrix() takes eigenvectors alone.")
    idx = _band_block(V.shape[1], bands, call)

    bloch = _measured_sum(rep, bloch, call=call)
    H = _hamiltonian_at(rep, k, bloch)
    resid = _states_at_k(H, w, V)
    foreign = _foreign_states(k, resid, rep._scale)
    if foreign is not None:
        return foreign

    w = _ascending(w, call)
    mult = multiplets(w, degeneracy_tol)
    chosen = set(idx)
    rows = []
    for m in mult:
        inside = chosen & set(m)
        if not inside:
            continue
        if len(inside) != len(m):
            return Refusal(
                "multiplet_split",
                f"the bands {sorted(idx)} cut the degenerate multiplet {list(m)} at "
                f"k = {np.round(k, 6).tolist()} (energies "
                f"{np.round(w[list(m)], 12).tolist()}, spread "
                f"{w[m[-1]] - w[m[0]]:.3g}, degeneracy_tol "
                f"{float(degeneracy_tol):g}). Ask for every band of the multiplet, or end "
                f"the bands at a gap.",
                value=float(w[m[-1]] - w[m[0]]),
                threshold=float(degeneracy_tol),
                context={"bands": idx, "multiplet": m, "k": k,
                         "energies": w[list(m)]})
        rows.append(m)

    lg = _tables._little_group(rep._group, k)
    if isinstance(lg, Refusal):
        return lg
    ours = [int(i) for i in rep.little_group(k, _atol=atol)]
    ours_u = tuple(sorted(i for i in ours
                          if not rep._operation(i, call=call).time_reversal))
    not_carried = tuple(sorted(set(lg.unitary) - set(ours_u)))
    if not_carried:
        # every representation a call returns carries the whole group it was
        # built for, so an operation of that group left out is the package
        # disagreeing with itself
        raise InternalError(
            f"the representation of {rep._group.name} #{rep._group.number_str} leaves out "
            f"part of its own little group at k = {np.round(k, 6).tolist()}; not carried: "
            f"{list(not_carried)}. Report it, with the model and the call.")

    chars = np.zeros((len(rows), len(ours_u)), dtype=complex)
    unit, comm = {}, {}
    for c, ind in enumerate(ours_u):
        whole = _sewing_core(rep, ind, k, V, idx, H, call=call, atol=atol, residual=resid)
        if isinstance(whole, Refusal):  # pragma: no cover: membership tested above
            return whole
        unit[ind] = whole.numerical.unitarity
        comm[ind] = whole.numerical.commutator
        for r, m in enumerate(rows):
            local = [idx.index(b) for b in m]
            chars[r, c] = np.trace(whole.matrix[np.ix_(local, local)])

    thr = two_tier_thresholds(H, multiplet_separation(w, degeneracy_tol),
                              degeneracy_tol=float(degeneracy_tol))
    dis = disagreements(thr, {f"unitarity[#{i}]": v for i, v in unit.items()})
    rows_out = tuple(
        MultipletCharacters(bands=tuple(m), energy=float(w[list(m)].mean()),
                            characters={int(ind): complex(chars[r, c])
                                        for c, ind in enumerate(ours_u)},
                            _degeneracy_tol=float(degeneracy_tol))
        for r, m in enumerate(rows))
    return BandCharacters(
        bands=idx, k=k, kpoint=_tabulated_name(rep._group, k) if named else None,
        multiplets=rows_out, _little_group=lg,
        numerical=BandCharactersNumerical(
            unitarity=unit, commutators=comm, residual=resid,
            spread=multiplet_spread(w, degeneracy_tol), thresholds=thr, disagreements=dis,
            degeneracy_tol=float(degeneracy_tol), spectrum=w),
        _group_provenance=str(rep._group._provenance))

def _character_matrix(table) -> np.ndarray:
    """The characters of the :class:`BandCharacters` ``table`` as a matrix.

    Returns a row for each degenerate multiplet and a column for each unitary
    operation of the little group, in the order of the rows' characters,
    which is ascending index.
    """
    columns = tuple(table.multiplets[0].characters) if table.multiplets else ()
    return np.array([[m.characters[i] for i in columns] for m in table.multiplets],
                    dtype=complex).reshape(len(table.multiplets), len(columns))

# ==========================================================================
# the little-group representations at one point
# ==========================================================================

def format_coreps(coreps) -> str:
    r"""Multiplicities of little-group representations, written on one line.

    ``{'-GM6': 2, '-GM7': 1}`` is written ``2x-GM6 + -GM7``.  The labels are
    ordered point by point, ``GM`` first and the other points alphabetically,
    and at each point by the numbers in the label, ``GM2`` before ``GM10``.
    So one set of multiplicities is always written as the same line.
    :attr:`Multiplet.label` holds multiplicities written so, and the
    summaries of :class:`CorepsAt`, :class:`HSPTable` and of the records
    :func:`.ebr.decompose`, :func:`.ebr.regular_completion` and
    :func:`.ebr.symmetry_indicator` return print them the same way.

    Parameters
    ----------
    coreps : mapping
        ``{label: multiplicity}``, the multiplicities nonnegative integers,
        as :attr:`CorepsAt.coreps` and :attr:`HSPTable.symmetry_data` hold
        them.

    Returns
    -------
    str
        The labels joined by ``" + "``, each after its multiplicity and ``x``
        where the multiplicity is above one.  A label of multiplicity zero
        is not written, and ``""`` is returned where every multiplicity is
        zero.
    """
    return " + ".join(f"{n}x{lab}" if n > 1 else lab
                      for lab, n in sorted(coreps.items(),
                                           key=lambda item: _tables.corep_order(item[0]))
                      if n)


@dataclass(frozen=True, eq=False)
class Multiplet:
    """One degenerate multiplet at a high-symmetry point, with its labels.

    A row of :attr:`CorepsAt.multiplets` and of :attr:`HSPTable.multiplets`.
    Its degeneracy is the number of its ``bands``.

    Attributes
    ----------
    kpoint : str
        The name of the point in the tables of the irreptables package,
        ``"GM"``.
    k : ndarray
        The point, in the model's reduced coordinates.
    bands : tuple of int
        The bands of the multiplet.
    energy : float
        The mean energy of the multiplet, in the units of ``H``.
    corep_names : tuple of str
        The labels of the little-group representations the multiplet
        carries, a representation carried twice named twice.
    label : str
        The same labels on one line, as :func:`format_coreps` writes them:
        ``"-GM6"``, ``"-GM6 + -GM7"``, ``"2x-GM6"``.
    """

    kpoint: str
    k: np.ndarray
    bands: tuple
    energy: float
    corep_names: tuple
    label: str
    # the energy tolerance the multiplet was grouped at, which its energy is
    # printed at
    _degeneracy_tol: float = field(default=DEGENERACY_TOL, repr=False)

    def __eq__(self, other):
        if not isinstance(other, Multiplet):
            return NotImplemented
        return (self.kpoint == other.kpoint and tuple(self.bands) == tuple(other.bands)
                and self.energy == other.energy
                and tuple(self.corep_names) == tuple(other.corep_names)
                and self.label == other.label
                and np.array_equal(np.asarray(self.k), np.asarray(other.k)))

    def _line(self) -> str:
        """The row as the summaries print it: ``bands 0-1 at -16.35: -GM6``."""
        energy = float(_resolved([self.energy], self._degeneracy_tol)[0])
        return (f"band{'s' if len(self.bands) > 1 else ''} {_index_ranges(self.bands)} at "
                f"{energy:.4g}: {self.label}")

    def __repr__(self) -> str:  # pragma: no cover: convenience only
        return f"Multiplet({self.kpoint}, {self._line()})"


@dataclass(frozen=True)
class CorepsAtNumerical:
    """How the little-group representations at one point were identified.

    Held in :attr:`CorepsAt.numerical`.  ``summary()`` prints the thresholds
    and the identification, each under a line naming its attribute.

    Attributes
    ----------
    thresholds : Thresholds
        The two zero thresholds at the point.
    identification : ..groups.CorepIdentification
        The identification of the characters: the little-group
        representations of each multiplet, their multiplicities before
        rounding, and how far those are from integers.
    """

    thresholds: Thresholds
    identification: Any

    def summary(self) -> str:
        """The thresholds and the identification, each under its attribute."""
        lines = [f"the zero thresholds at {self.identification.kpoint}   (.thresholds)"]
        lines += ["  " + line for line in self.thresholds.summary().splitlines()]
        lines.append(f"the {self.identification._kind} read off the characters   "
                     "(.identification)")
        lines += ["  " + line for line in self.identification.summary().splitlines()]
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover: convenience only
        return (f"CorepsAtNumerical({self.identification.kpoint}, integrality="
                f"{float(self.identification.integrality):.2g})")


@dataclass(frozen=True)
class CorepsAt:
    """The little-group representations of a set of bands at one high-symmetry point.

    The record :func:`coreps_at` returns.  ``summary()`` prints the
    multiplicities of the little-group representations over the set of
    bands, then each degenerate multiplet's labels where there are several
    multiplets, and the gap to the nearest other band.

    Attributes
    ----------
    kpoint : str
        The name of the point in the tables of the irreptables package,
        ``"GM"``.
    k : ndarray
        The point, in the model's reduced coordinates.
    bands : tuple of int
        The set of bands, by their indices among the states handed in,
        ascending.
    coreps : dict
        The multiplicities of the little-group representations of the set of
        bands at the point, ``{label: multiplicity}``.
    multiplets : tuple of Multiplet
        A row for each degenerate multiplet of the set of bands, lowest
        first, with its bands, its mean energy and its labels.
    isolation : float
        The energy gap from the set of bands to the nearest band outside it,
        in the units of ``H``.  ``inf`` where the set holds every band.
    numerical : CorepsAtNumerical
        The zero thresholds at the point, and the multiplicities before
        rounding.
    """

    kpoint: str
    k: np.ndarray
    bands: tuple
    coreps: dict
    multiplets: tuple
    isolation: float
    numerical: CorepsAtNumerical
    # the characters the little-group representations were identified from,
    # `characters(r, k, bands=...)`
    _characters: Any = field(default=None, repr=False, compare=False)

    def summary(self) -> str:
        """The little-group representations, a line for each part with its attribute."""
        bands = self.bands
        lines = [f"band{'s' if len(bands) > 1 else ''} {_index_ranges(bands)} at "
                 f"{_kpoint_text(self.kpoint, self.k)}: {format_coreps(self.coreps)}   "
                 "(.bands, .kpoint, .k, .coreps)"]
        if len(self.multiplets) > 1:
            lines.append("  by degenerate multiplet, from the lowest   (.multiplets)")
            lines += [f"    {m._line()}" for m in self.multiplets]
        if np.isfinite(self.isolation):
            these = "these bands" if len(bands) > 1 else "this band"
            lines.append(f"  {self.isolation:.4g} from {these} to the nearest other band, in the "
                         "units of H   (.isolation)")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover: convenience only
        return f"CorepsAt({self.kpoint}: {format_coreps(self.coreps)})"

def _isolation(spectrum, bands):
    """The energy gap from a set of bands to the nearest band outside it.

    ``spectrum`` holds every energy at the point and ``bands`` the indices of
    the set.  Returns ``inf`` where the set is empty or holds every band.
    """
    w = np.asarray(spectrum, dtype=float).reshape(-1)
    inside = sorted(bands)
    outside = [b for b in range(len(w)) if b not in set(inside)]
    if not outside or not inside:
        return float("inf")
    return float(min(abs(w[a] - w[b]) for a in inside for b in outside))

def _coreps_at_rep(rep, k, *, states=None, bands=None, degeneracy_tol=None,
                   label_tol=LABEL_TOL, _ktol=1e-5, _atol=1e-8,
                   _mesh_tol=MESH_TOL) -> CorepsAt | Refusal:
    """What :func:`coreps_at` returns, read off a representation.

    ``rep`` is the representation, ``r.rep`` or one built by hand.  ``k``,
    ``states``, ``bands``, ``degeneracy_tol`` and ``label_tol`` are as
    :func:`coreps_at` takes them.  ``_ktol`` is as ``ktol`` in
    :func:`_coreps_at`, ``_atol`` as ``atol`` in :func:`_sewing_core`, and
    ``_mesh_tol`` as ``mesh_tol`` in :func:`_wfarray_fiber`.  Returns and
    refuses as :func:`coreps_at` does.
    """
    return _coreps_at(rep, k, states=states, bands=bands, degeneracy_tol=degeneracy_tol,
                      label_tol=label_tol, ktol=_ktol, atol=_atol, mesh_tol=_mesh_tol,
                      bloch=None)

def coreps_at(realized, k, *, states=None, bands=None, degeneracy_tol=None,
              label_tol=LABEL_TOL, _ktol=_coreps_at_rep.__kwdefaults__["_ktol"],
              _atol=_coreps_at_rep.__kwdefaults__["_atol"],
              _mesh_tol=MESH_TOL) -> CorepsAt | Refusal:
    """The little-group representations of a set of bands at one high-symmetry point.

    The characters of each degenerate multiplet are measured as
    :func:`characters` measures them.  Each row of characters is expanded
    over the characters of the little-group representations that the tables
    of the irreptables package list at ``k``.  A multiplicity within
    ``label_tol`` of a nonnegative integer is rounded to that integer.

    Parameters
    ----------
    realized : ..groups.RealizedGroup
        What :func:`..groups.realized_group` returned, ``r``.
    k : str or array_like, shape (dim_k,)
        The high-symmetry point.  Pass its name in the tables, ``"GM"``, or
        its reduced coordinates, as :func:`..groups.high_symmetry_points`
        returns them.  For a layer, the three reduced coordinates of a point
        of its plane, in its three-dimensional cell, are also accepted.
    states : None, pythtb.TBModel, pythtb.WFArray or (evals, evecs), optional
        The states at ``k``, with their energies.  Pass ``None`` or
        ``r.model`` to solve the model there.  Pass a ``WFArray`` filled by
        ``solve_model``, or the pair :func:`solve_at` returns, to reuse one
        diagonalization.
    bands : int or sequence of int
        The set of bands.  Pass ``n`` for the lowest ``n`` bands, or the band
        indices.  Required.
    degeneracy_tol : float, optional
        The energy tolerance, in the units of ``H``, within which bands are
        one degenerate multiplet.  Left out, ``1e-8`` times ``max|H|``, the
        largest entry of ``H(k)`` once the mean on-site energy is subtracted.
    label_tol : float, default 1e-3
        The largest distance from an integer at which a multiplicity is
        rounded to that integer.

    Returns
    -------
    CorepsAt or Refusal
        The multiplicities of the little-group representations the set of
        bands carries at ``k``, and the labels of each of its degenerate
        multiplets, lowest first.  Little-group representations can only be
        returned at a point the tables list, at the coordinates
        :func:`..groups.high_symmetry_points` returns for it, up to a
        reciprocal lattice vector.  They can only be returned if the set of
        bands contains all the states of each degenerate multiplet there.  A
        ``Refusal`` in its place is falsy and names what stopped the call.

    Examples
    --------
    The six lowest bands of lin3g (:func:`..models.lin3g`) at ``GM``:

    >>> r = realized_group(models.lin3g())
    >>> c = coreps_at(r, "GM", bands=6)
    >>> print(c.summary())
    bands 0-5 at GM (0, 0): 2x-GM6 + -GM7   (.bands, .kpoint, .k, .coreps)
      by degenerate multiplet, from the lowest   (.multiplets)
        bands 0-1 at -16.35: -GM6
        bands 2-3 at -12.38: -GM7
        bands 4-5 at -4.325: -GM6
      13.02 from these bands to the nearest other band, in the units of H   (.isolation)

    Refusals
    --------
    A refusal handed in as ``realized`` is returned as it is.

    ``multiplet_split``
        the set of bands cuts a degenerate multiplet at ``k``.
    ``kpoint_not_tabulated``
        ``k``, given by its coordinates, is no tabulated point in the model's
        cell, or is a member of a tabulated point's star other than the one
        tabulated.
    ``states_not_at_k``
        the states handed in do not solve ``H(k)``.
    ``k_not_on_mesh``
        the mesh of the ``WFArray`` handed in does not hold ``k``.
    ``out_of_plane``
        ``k`` has a nonzero component normal to the plane of a layer, or
        names a tabulated point off that plane.
    ``integer_multiplicity``
        a multiplicity is farther than ``label_tol`` from a nonnegative
        integer.
    ``multiplet_dimension``
        the dimensions of the little-group representations found in a
        multiplet do not add up to its degeneracy.
    ``little_group_mismatch``
        a unitary operation of the little group in the model's cell is not
        among those the tables list at the point.
    ``tables_not_ready``
        the model's cell cannot be transformed to the conventional cell of
        the irreptables package.

    Raises
    ------
    TypeError
        ``realized`` is not what :func:`..groups.realized_group` returned, or
        ``bands`` is left out.
    ValueError
        States without energies: eigenvectors alone, or a ``WFArray`` not
        filled by ``solve_model``.
        States of a model other than ``r.model``.
        Eigenvectors in an array that is neither two- nor three-dimensional.
        An ``(evals, evecs)`` pair with a number of energies other than that
        of eigenvectors.
        ``bands`` naming no band, a band twice, or a band outside the states.
        ``k`` a name the group does not tabulate.
        ``k`` with a number of coordinates other than ``dim_k``, or three for
        a layer.
        A ``WFArray`` with parameter axes in its mesh, or whose mesh has
        another number of coordinates than ``k``.
        ``r.model`` edited since :func:`..groups.realized_group` measured it.

    See Also
    --------
    hsp_table : the little-group representations of a set of bands at every
        high-symmetry point, and the symmetry data vector their
        multiplicities form.
    """
    rep = _rep_of("coreps_at", realized, "coreps_at(r, k, bands=6)")
    if isinstance(rep, Refusal):
        return rep
    return _coreps_at_rep(rep, k, states=states, bands=bands,
                          degeneracy_tol=degeneracy_tol, label_tol=label_tol,
                          _ktol=_ktol, _atol=_atol, _mesh_tol=_mesh_tol)

def _coreps_at(rep, k, *, states, bands, degeneracy_tol, label_tol, ktol, atol,
               mesh_tol, bloch, call="coreps_at") -> CorepsAt | Refusal:
    """The little-group representations :func:`coreps_at` returns.

    ``rep`` is the representation, or a refusal, which is returned as it is.
    ``k``, ``states``, ``bands``, ``degeneracy_tol`` and ``label_tol`` are as
    :func:`coreps_at` takes them.  ``ktol`` is how far ``k`` may sit from a
    tabulated point and be named after it, in reduced coordinates.
    ``atol``, ``mesh_tol``, ``bloch`` and ``call``, the public call's name,
    are as :func:`_characters` takes them.  The characters are read by
    :func:`_characters` and identified by
    :func:`.._tables._identify_from_characters`, whose refusals are returned
    as they are.
    """
    if bands is None:
        raise TypeError(f"{call}() missing required argument 'bands'")
    if isinstance(rep, Refusal):
        return rep
    gate = _spacegroup._lattice_not_descended(rep._group)
    if gate is not None:
        return gate
    tab = _characters(rep, k, states=states, bands=bands, degeneracy_tol=degeneracy_tol,
                      atol=atol, mesh_tol=mesh_tol, bloch=bloch, call=call,
                      named=False)
    if isinstance(tab, Refusal):
        return tab

    out = _tables._identify_from_characters(
        rep._group, tab.k, _character_matrix(tab),
        multiplet_dims=[len(m.bands) for m in tab.multiplets],
        label_tol=label_tol, ktol=ktol)
    if isinstance(out, Refusal):
        return out
    tab = replace(tab, kpoint=out.kpoint)

    counts: dict = {}
    for row in out.labels:
        for lab, v in row.items():
            counts[lab] = counts.get(lab, 0) + int(v)
    counts = _tables.ordered_content(counts)
    tol = float(tab.numerical.degeneracy_tol)
    rows = tuple(
        Multiplet(kpoint=out.kpoint, k=tab.k, bands=tuple(m.bands), energy=m.energy,
                  corep_names=tuple(lab for lab, n in labels.items() for _ in range(int(n))),
                  label=format_coreps(labels), _degeneracy_tol=tol)
        for m, labels in zip(tab.multiplets, out.labels))
    return CorepsAt(kpoint=out.kpoint, k=tab.k, bands=tab.bands, coreps=counts,
                    multiplets=rows,
                    isolation=_isolation(tab.numerical.spectrum, tab.bands),
                    numerical=CorepsAtNumerical(thresholds=tab.numerical.thresholds,
                                                identification=out),
                    _characters=tab)

# ==========================================================================
# the little-group representations at every high-symmetry point
# ==========================================================================

@dataclass(frozen=True)
class HSPTable:
    """The symmetry data vector of a set of bands, point by point.

    The record :func:`hsp_table` returns.  :func:`..ebr.decompose`,
    :func:`..ebr.symmetry_indicator`, :func:`..ebr.regular_completion` and
    :func:`..srsi.srsi_of_bands` take it without ``r``, and use the group it
    holds.  ``summary()`` prints the multiplicities at each point, the points
    refused, and the note.

    Attributes
    ----------
    bands : tuple of int
        The set of bands, by their indices, ascending.
    group : ..groups.GroupData
        The group whose little-group representations the labels name,
        ``r.group``.
    kpoints : dict
        Every point :func:`..groups.high_symmetry_points` returns, by name,
        with its reduced coordinates, the points refused among them.
    symmetry_data : dict
        If nonempty, the symmetry data vector, ``{label: multiplicity}``, over
        every point not refused.
    multiplets : dict
        If nonempty, ``{kpoint: tuple of Multiplet}``: at each point not
        refused, a row for each degenerate multiplet of the set of bands,
        lowest first, with its labels.
    complete : bool
        Whether the little-group representations were identified at every
        point of ``kpoints``.
    refusals : dict
        If nonempty, ``{kpoint: Refusal}``, the refusal returned at each point
        refused.
    note : str
        If nonempty, a sentence describing any convention choices not fixed by
        the Hamiltonian that are needed to determine the corep labels.
    """

    bands: tuple
    group: Any = field(repr=False, compare=False)
    kpoints: dict
    symmetry_data: dict
    multiplets: dict
    complete: bool
    refusals: dict
    note: str = ""
    # `{kpoint: CorepsAt}` at each point measured, which the calls that take
    # the table read their characters and spectra from
    _records: dict = field(default_factory=dict, repr=False, compare=False)
    # "realized" for a group measured on H, "descended" for one measured after
    # the cell was lowered
    _group_provenance: str = field(default="cell", repr=False, compare=False)
    # for a layer, the representation the table was read with, holding the model
    # and the matrices its decoupled stack is built from; None for a crystal
    _rep: Any = field(default=None, repr=False, compare=False)
    # the site irreps the labels rest on, an ebr.SiteContent, or None without a
    # representation
    _site_content: Any = field(default=None, repr=False, compare=False)
    # for a layer, each orbital's height from its middle plane along the normal,
    # in the model's units; None for a crystal periodic in three directions
    _heights: Any = field(default=None, repr=False, compare=False)

    def _at(self, kpoint) -> dict:
        """The symmetry data vector's entries at the point named ``kpoint``."""
        return {lab: n for lab, n in self.symmetry_data.items()
                if _tables.kpoint_of_label(lab) == kpoint}

    def summary(self) -> str:
        """The labels at each point, a line for each part with its attribute."""
        lines = [f"band{'s' if len(self.bands) > 1 else ''} {_index_ranges(self.bands)}   (.bands)"]
        measured = [name for name in self.kpoints if name in self.multiplets]
        if measured:
            kind = ("irreps" if getattr(self.group, "magnetic_type", None) == "I"
                    else "coreps")
            lines.append(f"  their {kind} at each high-symmetry point   (.symmetry_data)")
            lines += [f"    {name}: {format_coreps(self._at(name))}" for name in measured]
        gates = {}
        for name in self.kpoints:
            if name in self.refusals:
                gates.setdefault(str(self.refusals[name].gate), []).append(name)
        if gates:
            lines.append("  refused at " + "; ".join(f"{_listed(names)} ({gate})"
                                                     for gate, names in gates.items())
                         + ", so the symmetry data vector is incomplete   "
                         "(.refusals, .complete)")
        if self.note:
            lines.append(f"  {self.note}   (.note)")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover: convenience only
        n = len(self.multiplets)
        return (f"HSPTable(bands {_index_ranges(self.bands)}, {n} k-point{'' if n == 1 else 's'} "
                f"measured, {len(self.refusals)} refused)")

# ==========================================================================
# what a table carries for the calls that take it
# ==========================================================================

def _site_content_record(rep):
    """The site irreps a table's labels rest on, an :class:`..ebr.SiteContent`.

    ``rep`` is the representation the labels were read with.  Returns
    ``None`` where ``rep`` is ``None`` or holds no table of site irreps.
    """
    if rep is None:
        return None
    ch = getattr(rep, "_characters", None)
    if ch is None:
        return None
    from .ebr import SiteContent
    return SiteContent(number_str=str(rep._group.number_str),
                       irreps=tuple(str(x) for x in ch.irreps),
                       determined_by=tuple(str(o.determined_by) for o in ch.orbitals),
                       residual=float(ch.residual))

def _layer_height(group):
    """How a coordinate along a layer's vacuum direction gives a height.

    ``group`` is the group, read in its three-dimensional cell.  Returns
    ``(normal_dir, height)`` for a layer with one direction of vacuum:
    ``normal_dir`` that direction, and ``height(z)`` the height of the
    reduced coordinate ``z`` along it from the layer's middle plane, along
    the normal to the two periodic lattice vectors, in the model's units.
    Returns ``None`` otherwise, as for a crystal periodic in three
    directions.
    """
    padded = tuple(int(d) for d in (getattr(group, "_vacuum_dirs", ()) or ()))
    if int(getattr(group, "_dim_k", 3)) >= 3 or len(padded) != 1:
        return None
    normal_dir = padded[0]
    lattice = np.asarray(group._lattice, dtype=float)
    plane = [d for d in range(3) if d != normal_dir]
    normal = np.cross(lattice[plane[0]], lattice[plane[1]])
    scale = float(lattice[normal_dir] @ normal) / float(np.linalg.norm(normal))

    def height(z):
        """The height of ``z``, a coordinate along the vacuum direction."""
        return ((float(z) + 0.5) % 1.0 - 0.5) * scale

    return normal_dir, height

def _orbital_heights(group, where):
    """Each position's height from a layer's middle plane.

    ``group`` is the group, and ``where`` holds the positions in reduced
    coordinates of its three-dimensional cell, a row each.  The heights are
    those of :func:`_layer_height`, in the model's units.  Returns ``None``
    for a crystal periodic in three directions.
    """
    reading = _layer_height(group)
    if reading is None:
        return None
    normal_dir, height = reading
    return tuple(height(z) for z in np.atleast_2d(where)[:, normal_dir])

def _table_heights(rep):
    """The heights of ``rep``'s orbitals, or ``None`` for a crystal.

    They are those :func:`_orbital_heights` gives at the orbitals' positions
    in the group's three-dimensional cell.
    """
    if _layer_height(rep._group) is None:
        return None
    from ._matrices import _orb_positions_3d
    return _orbital_heights(rep._group, _orb_positions_3d(rep._model, rep._group))

#: The keys under which a labeling set aside can label a band otherwise: the
#: site irreps at the lowest angular momentum, the functions the orbitals read
#: as, and the band labels.
_RELABELING_KEYS = ("content", "orbitals", "bands")

#: The note, where some energy multiplet of a set of bands reads otherwise under
#: another labeling the hoppings leave open.
_ANOTHER_LABELING_RELABELS = (
    "another labeling the hoppings leave open keeps every site irrep and labels some energy "
    "multiplets of these bands differently; it keeps which bands are degenerate and the labels "
    "of all the model's bands together")

def _labeling_note(rep, block) -> str:
    """The sentence of :attr:`HSPTable.note` on a labeling, or ``""``.

    ``rep`` is the representation the table is read with, and ``block`` the
    indices of the set of bands.  Where the lowest-angular-momentum rule kept
    one of several labelings of a coupled part that give every orbital the
    same site irrep, the atomic-limit table holds the labels each energy
    multiplet carries under the labeling kept and under each other one that
    labels some band otherwise.  The sentence is returned where an
    energy multiplet the set holds whole, at a point where it cuts none,
    carries other labels under a labeling set aside by one of the keys of
    :data:`_RELABELING_KEYS` (:func:`_labeled_apart`).  A labeling set aside
    by the group's order of operations alone labels every band alike, and
    nothing is said.
    """
    table = getattr(rep, "_characters", None)
    record = getattr(table, "_labelings", {}) or {}
    held = {int(b) for b in block}
    if any(key in _RELABELING_KEYS and _labeled_apart(rec["chosen"], other, held)
           for rec in record.get("choices", ()) for key, other in rec["others"]):
        return _ANOTHER_LABELING_RELABELS
    return ""

def _labeled_apart(chosen, other, held) -> bool:
    """Whether ``other`` gives other labels to a multiplet ``held`` holds whole.

    ``chosen`` and ``other`` are the labels under two labelings,
    ``{point: ((bands, {label: multiplicity}), ...)}``, and ``held`` is a set
    of band indices.  One multiplet whose labels differ is enough.  A point is
    passed over where ``other`` does not hold it or groups the bands into
    other multiplets, and where the bands held cut a multiplet, which the
    table does not label.
    """
    for point, multiplets in chosen.items():
        theirs = other.get(point)
        if theirs is None or len(theirs) != len(multiplets):
            continue
        inside = [(bands, labels, bands_other, labels_other)
                  for (bands, labels), (bands_other, labels_other) in zip(multiplets, theirs)
                  if held.intersection(bands)]
        if any(not held.issuperset(bands) or tuple(bands) != tuple(bands_other)
               for bands, _labels, bands_other, _other in inside):
            continue
        if any(dict(labels) != dict(labels_other)
               for _bands, labels, _bands_other, labels_other in inside):
            return True
    return False

def _multiplicity_note(rep, block, fibers) -> str:
    """The sentence of :attr:`HSPTable.note` on a pseudospin, or ``""``.

    ``rep`` is the representation the table is read with, ``block`` the
    indices of the set of bands, and ``fibers`` the states at each point read,
    ``(evals, evecs)`` pairs where the model was solved.  Where the
    Hamiltonian conserves a pseudospin, the representation its copies carry
    is not fixed by it, and the lowest-angular-momentum rule assigned it
    (:attr:`..symmetry._OrbitalTable.multiplicity_spaces`).  A clause is said
    for each such set of copies where the representations the group's own
    table admits give more than one set of site-irrep multiplicities, no
    declaration states it, and the set of bands has a weight above ``1e-8``
    on its orbitals.  The weight is the largest over the pairs in
    ``fibers``; without a pair the set of bands is taken to have weight
    there.  Copies whose admitted representations give one set of site-irrep
    multiplicities are labelings like any other, which :func:`_labeling_note`
    reads.
    """
    table = getattr(rep, "_characters", None)
    spaces = tuple(getattr(table, "multiplicity_spaces", ()) or ())
    if not spaces:
        return ""
    nspin = int(rep._model.nspin)
    nstate = int(rep._model.nstate)
    said = []
    for space in spaces:
        if len(space._distinct()) <= 1 or space.determined_by == "declaration":
            continue
        states = [int(o) * nspin + s for o in space.orbitals for s in range(nspin)]
        weight = None
        for fiber in fibers:
            if not (isinstance(fiber, (tuple, list)) and len(fiber) == 2):
                continue
            V = _as_columns("hsp_table", fiber[1])
            got = float(np.sum(np.abs(V[np.ix_(states, list(block))]) ** 2))
            weight = got if weight is None else max(weight, got)
        if weight is not None and weight <= 1e-8:
            continue
        these = ("these bands" if len(block) >= nstate
                 else f"these {len(block)} of the {nstate} bands")
        said.append(f"the labels of {these} rest on what the pseudospin conserved on orbitals "
                    f"{tuple(space.orbitals)} carries, which the Hamiltonian does not fix: "
                    "the lowest-angular-momentum rule assigned the site content "
                    f"{space.content or '(unnamed)'}, of the {len(space._distinct())} the "
                    "group's own table admits, and another can label them differently")
    return "; ".join(said)

def _hsp_table_rep(rep, *, states=None, bands=None, degeneracy_tol=None,
                   label_tol=LABEL_TOL, _ktol=1e-5, _atol=1e-8,
                   _mesh_tol=MESH_TOL) -> HSPTable | Refusal:
    """What :func:`hsp_table` returns, read off a representation.

    ``rep`` is the representation, ``r.rep`` or one built by hand, or a
    refusal, which is returned as it is.  ``states``, ``bands``,
    ``degeneracy_tol`` and ``label_tol`` are as :func:`hsp_table` takes them.
    ``_ktol`` is as ``ktol`` in :func:`_coreps_at`, ``_atol`` as ``atol`` in
    :func:`_sewing_core`, and ``_mesh_tol`` as ``mesh_tol`` in
    :func:`_wfarray_fiber`.  Returns and refuses as :func:`hsp_table` does.
    """
    if bands is None:
        raise TypeError("hsp_table() missing required argument 'bands'")
    if isinstance(rep, Refusal):
        return rep
    block = _band_block(int(rep._model.nstate), bands, "hsp_table")
    gate = _spacegroup._lattice_not_descended(rep._group)
    if gate is not None:
        return gate
    if states is not None and hasattr(states, "solve_ham") and states is not rep._model:
        raise ValueError("hsp_table(): states= is a model other than r.model. Pass r.model, "
                         "the model realized_group measured, or a WFArray of it filled by "
                         "solve_model; or leave states= out.")
    if not (states is None or states is rep._model
            or (hasattr(states, "wfs") and hasattr(states, "mesh"))):
        raise ValueError(f"hsp_table(): states= is of type {type(states).__name__}. Pass "
                         "r.model, or a WFArray filled by solve_model, or leave states= "
                         "out; the (evals, evecs) pair of one point is read by coreps_at.")
    points = _tables._high_symmetry_points(rep._group)
    if isinstance(points, Refusal):
        return points

    rows, records, refusals = {}, {}, {}
    counts: dict = {}

    # Collect every point whose little group is read; where the model is solved here,
    # it is solved once at all of them.
    wanted = []
    for name, k in points.items():
        lg = _tables._little_group(rep._group, k)
        if isinstance(lg, Refusal):
            refusals[name] = lg
            continue
        wanted.append((name, np.asarray(k, dtype=float).reshape(-1)))

    per_point = {}
    bloch = None                    # the model's Bloch sum, where the model is solved here
    if wanted and (states is None or states is rep._model):
        karr = np.array([k for _, k in wanted], dtype=float)
        bloch = _measured_sum(rep, call="hsp_table")
        evals, evecs = _bloch.solve(rep._model, karr, bloch)
        evals, evecs = np.asarray(evals), np.asarray(evecs)
        if evals.ndim == 1:                     # solve_ham drops a length-1 k axis
            evals, evecs = evals[None], evecs[None]
        per_point = {name: (evals[i], evecs[i]) for i, (name, _) in enumerate(wanted)}
    else:
        per_point = {name: states for name, _ in wanted}

    if wanted:
        degeneracy_tol = _degeneracy(rep, degeneracy_tol)
    for name, k in wanted:
        got = _coreps_at(rep, k, states=per_point[name], bands=block,
                         degeneracy_tol=degeneracy_tol, label_tol=label_tol, ktol=_ktol,
                         atol=_atol, mesh_tol=_mesh_tol, bloch=bloch, call="hsp_table")
        if isinstance(got, Refusal):
            refusals[name] = got
            continue
        records[name] = got
        rows[name] = got.multiplets
        for lab, v in got.coreps.items():
            counts[lab] = counts.get(lab, 0) + int(v)

    notes = (_multiplicity_note(rep, block, per_point.values()),
             _labeling_note(rep, block))
    return HSPTable(bands=block, group=rep._group, kpoints=dict(points),
                    symmetry_data=_tables.ordered_content(counts), multiplets=rows,
                    complete=not refusals, refusals=refusals,
                    note="; ".join(n for n in notes if n), _records=records,
                    _group_provenance=str(rep._group._provenance),
                    _rep=rep if int(rep._group._dim_k) < 3 else None,
                    _site_content=_site_content_record(rep), _heights=_table_heights(rep))

def hsp_table(realized, *, states=None, bands=None, degeneracy_tol=None,
              label_tol=LABEL_TOL, _ktol=_hsp_table_rep.__kwdefaults__["_ktol"],
              _atol=_hsp_table_rep.__kwdefaults__["_atol"],
              _mesh_tol=MESH_TOL) -> HSPTable | Refusal:
    """The symmetry data vector of a set of bands, point by point.

    The points are the high-symmetry points the tables of the irreptables
    package list for the model's group, those off a layer's plane left out,
    as :func:`..groups.high_symmetry_points` returns them.  At each point the
    little-group representations of the set of bands are identified as
    :func:`coreps_at` identifies them.  Their multiplicities at every point,
    taken together, are the symmetry data vector.

    Parameters
    ----------
    realized : ..groups.RealizedGroup
        What :func:`..groups.realized_group` returned, ``r``.
    states : None, pythtb.TBModel or pythtb.WFArray, optional
        The states at the points.  Pass ``None`` or ``r.model`` to solve the
        model there.  Pass a ``WFArray`` filled by ``solve_model`` to read
        its states at the points its mesh holds.
    bands : int or sequence of int
        The set of bands.  Pass ``n`` for the lowest ``n`` bands, or the band
        indices.  Required.
    degeneracy_tol : float, optional
        The energy tolerance, in the units of ``H``, within which bands are
        one degenerate multiplet.  Left out, ``1e-8`` times ``max|H|``, the
        largest entry of ``H(k)`` once the mean on-site energy is subtracted.
    label_tol : float, default 1e-3
        The largest distance from an integer at which a multiplicity is
        rounded to that integer.

    Returns
    -------
    HSPTable or Refusal
        The labels of each degenerate multiplet of the set of bands at every
        point, and the symmetry data vector.  Little-group representations
        can only be returned at a point if the set of bands contains all the
        states of each degenerate multiplet there.  The table holds, in
        ``refusals``, the refusal :func:`coreps_at` returns at each point
        where they are not returned, and is then not complete.  With a
        ``WFArray``, the table holds the refusal ``k_not_on_mesh`` at each
        point its mesh does not hold.  A ``Refusal`` in place of the table is
        falsy and names what stopped the call.

    Examples
    --------
    The six lowest bands of lin3g (:func:`..models.lin3g`):

    >>> r = realized_group(models.lin3g())
    >>> print(hsp_table(r, bands=6).summary())
    bands 0-5   (.bands)
      their coreps at each high-symmetry point   (.symmetry_data)
        GM: 2x-GM6 + -GM7
        M: -M6 + 2x-M7
        X: 3x-X5

    Refusals
    --------
    A refusal handed in as ``realized`` is returned as it is.

    ``tables_not_ready``
        the model's cell cannot be transformed to the conventional cell of
        the irreptables package.

    Raises
    ------
    TypeError
        ``realized`` is not what :func:`..groups.realized_group` returned, or
        ``bands`` is left out.
    ValueError
        A model other than ``r.model`` as ``states``, or a ``WFArray`` solved
        on one.  Anything else as ``states`` that is neither ``None``,
        ``r.model`` nor a ``WFArray``, such as the ``(evals, evecs)`` pair of
        one point.
        A ``WFArray`` not filled by ``solve_model``, or with parameter axes in
        its mesh.
        ``bands`` naming no band, a band twice, or a band outside the states.
        ``r.model`` edited since :func:`..groups.realized_group` measured it.

    See Also
    --------
    .ebr.decompose : the sums of elementary band representations that carry
        a symmetry data vector.
    .ebr.symmetry_indicator : the symmetry-indicator class of a set of bands.
    .srsi.srsi_of_bands : the integer-valued stable real-space invariants
        (ZSRSIs) of a set of bands.
    """
    rep = _rep_of("hsp_table", realized, "hsp_table(r, bands=6)")
    return _hsp_table_rep(rep, states=states, bands=bands, degeneracy_tol=degeneracy_tol,
                          label_tol=label_tol, _ktol=_ktol, _atol=_atol,
                          _mesh_tol=_mesh_tol)
