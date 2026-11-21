r"""The coreps a band set carries at the high-symmetry points, read off its eigenvectors.

At a k-point an operation ``g`` leaves invariant, the sewing matrix
``B(g,k)_mn = <u_m(k)|U(g;k)|u_n(k)>`` is the matrix of ``g`` on the bands
selected; its trace over a degenerate multiplet is that multiplet's
character, and the characters name the corep.

* :func:`hsp_table`: the coreps of a set of bands at every high-symmetry point,
  its symmetry data vector, which :func:`.ebr.decompose`,
  :func:`.ebr.symmetry_indicator` and :func:`.srsi.srsi_of_bands` take;
* :func:`coreps_at`: the coreps at one point, multiplet by multiplet;
* :func:`characters`: the characters they are read from;
* :func:`sewing_matrix`: one operation's sewing matrix;
* :func:`solve_at`: one diagonalization at a k-point, to read several sets of
  bands there; :func:`multiplets` groups its energies into degenerate levels,
  and :func:`multiplet_separation` and :func:`multiplet_spread` say how well
  they are resolved;
* :func:`format_coreps`: corep multiplicities on one line, ``2x-GM6 + -GM7``;
* :class:`Thresholds`: the zero thresholds at a point, one for numbers built
  from ``H(k)`` and one, widening where multiplets crowd, for numbers read off
  eigenvectors.

Each call takes the eigenvectors as ``states=``, in pythtb's own forms:
``None``, to solve the model at the point; a solved :class:`pythtb.WFArray`;
or the ``(evals, evecs)`` pair of one diagonalization.  A k-point is given by
its tabulated name, ``"GM"``, or by its reduced coordinates.  The little
groups, the corep tables and the labels are ``irrep``'s.
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
from ._text import _character_text, _k_text, _listed, _runs

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
# the two-tier threshold contract
# ==========================================================================

@dataclass(frozen=True)
class Thresholds:
    """The zero thresholds at one k-point: one for numbers built from ``H(k)`` alone, and one, widening where the multiplets crowd, for numbers read off eigenvectors.

    ``summary()`` prints them.  A threshold grown large says nothing read off a
    set of bands at that point is to be trusted.

    Attributes
    ----------
    flat : float
        The threshold for a number built from ``H(k)`` alone, a commutator or a
        residual: :data:`FLAT_ZERO`.
    noise_floor : float
        ``8 ε ‖H(k)‖₂ / Δ``, ``ε`` the machine precision and ``Δ`` the smallest
        gap between distinct multiplets: how far a computed eigenvector can turn
        among its near neighbours.
    effective : float
        ``max(flat, noise_floor)``, the threshold for a number read off
        eigenvectors.
    norm : float
        ``‖H(k)‖₂``, the spectral norm.
    separation : float
        ``Δ``.
    degeneracy_tol : float
        The energy tolerance the multiplets were grouped at.
    """

    flat: float
    noise_floor: float
    effective: float
    norm: float
    separation: float
    degeneracy_tol: float

    def zero(self, value) -> bool:
        """Is ``value`` zero under the **conditioned** (eigenvector) rule?"""
        return bool(float(np.abs(value)) <= self.effective)

    def zero_flat(self, value) -> bool:
        """Is ``value`` zero under the **flat** (operator) rule?"""
        return bool(float(np.abs(value)) <= self.flat)

    def disagree(self, value) -> bool:
        """Do the two rules classify ``value`` differently?"""
        return self.zero(value) != self.zero_flat(value)

    def summary(self) -> str:
        """The two thresholds, a line for each part, with the attribute that holds it."""
        if not np.isfinite(self.noise_floor):
            return (f"zero below {self.flat:.2g} for a number built from H(k) alone; no number "
                    "read off the eigenvectors here can be told from zero, no gap separating "
                    "distinct multiplets   (.flat, .effective, .separation)")
        return "\n".join([
            f"zero below {self.effective:.2g} for a number read off the eigenvectors here, "
            f"below {self.flat:.2g} for one built from H(k) alone   (.effective, .flat)",
            f"  the eigenvectors' noise floor {self.noise_floor:.2g} is 8 ε |H(k)| / Δ, ε the "
            f"machine precision, |H(k)| = {self.norm:.4g} and Δ = {self.separation:.4g}, the "
            "smallest gap between distinct multiplets   (.noise_floor, .norm, .separation)"])

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return (f"Thresholds(effective={self.effective:.2g}, flat={self.flat:.2g}, "
                f"nf={self.noise_floor:.2g}, |H|={self.norm:.4g}, "
                f"sep={self.separation:.4g})")

def two_tier_thresholds(hamiltonian, separation, *, flat=FLAT_ZERO,
                        degeneracy_tol=DEGENERACY_TOL) -> Thresholds:
    r"""The zero thresholds at one k-point, from ``H(k)`` and the smallest gap between distinct multiplets there.

    Parameters
    ----------
    hamiltonian : ndarray or float
        ``H(k)``, or its spectral norm.
    separation : float
        The smallest gap between distinct energy multiplets at this k, as
        :func:`multiplet_separation` gives it.  Zero, negative or infinite, it
        makes the noise floor infinite: no number read off eigenvectors is then
        told from zero.
    flat : float, optional
        The threshold for numbers built from ``H(k)`` alone, in the units of
        ``H``.  Default :data:`FLAT_ZERO`.
    degeneracy_tol : float, optional
        The energy tolerance, in the units of ``H``, the multiplets were grouped
        at, recorded.  Default :data:`DEGENERACY_TOL`.

    Returns
    -------
    Thresholds
        ``summary()`` prints them.

    Examples
    --------
    The thresholds at one k, from ``H(k)`` and the spectrum alone::

        H   = model.hamiltonian(k_pts=np.atleast_2d(k), flatten_spin_axis=True)[0]
        thr = two_tier_thresholds(H, multiplet_separation(np.linalg.eigvalsh(H)))
        # Thresholds(effective=8.3e-14, flat=1e-14, nf=8.3e-14, |H|=12.65, sep=0.2706)
    """
    H = np.asarray(hamiltonian)
    norm = float(H) if H.ndim == 0 else float(np.linalg.norm(H, 2))
    sep = float(separation)
    if not np.isfinite(sep) or sep <= 0.0:
        nf = float("inf")
    else:
        nf = 8.0 * float(np.finfo(float).eps) * norm / sep
    return Thresholds(flat=float(flat), noise_floor=nf, effective=max(float(flat), nf),
                      norm=norm, separation=sep, degeneracy_tol=float(degeneracy_tol))

def disagreements(thresholds: Thresholds, readouts: Mapping[str, Any]) -> tuple:
    """Every number the two zero thresholds at one k-point classify differently.

    Such a number is zero against the eigenvectors' threshold and not against
    the flat one, and which is right depends on how well the multiplets there
    are resolved: the list is the verdict.

    Parameters
    ----------
    thresholds : Thresholds
        The thresholds at the point.
    readouts : mapping
        ``{name: value}``, numbers read off eigenvectors.

    Returns
    -------
    tuple of dict
        ``{"name", "value", "zero_conditioned", "zero_flat"}`` for each number
        the two classify differently.
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
# the multiplet analysis — small functions over pythtb's energy arrays
# ==========================================================================

def _resolved(energies, degeneracy_tol):
    """Energies at the resolution the multiplets were grouped at: each rounded to the nearest multiple of ``degeneracy_tol``."""
    tol = float(degeneracy_tol)
    e = np.asarray(energies, dtype=float)
    return (np.round(e / tol) * tol if tol > 0.0 else e) + 0.0

def _clusters(values, degeneracy_tol):
    """Group a sorted 1-D array into clusters of width at most ``degeneracy_tol``.

    Membership is tested against each cluster's **first** element, never its
    running last, so a cluster can never chain its way past ``degeneracy_tol`` —
    the grouping is then a genuine statement that every member of a multiplet
    lies within ``degeneracy_tol`` of every other.
    """
    out = [[0]]
    for i in range(1, len(values)):
        if values[i] - values[out[-1][0]] <= degeneracy_tol:
            out[-1].append(i)
        else:
            out.append([i])
    return tuple(tuple(c) for c in out)

def _ascending(energies):
    w = np.asarray(energies, dtype=float).reshape(-1)
    if w.size == 0:
        raise ValueError("grouping multiplets needs a non-empty spectrum")
    if np.any(np.diff(w) < 0.0):
        raise ValueError("energies must be in ascending order (pythtb's solve_ham "
                         "and WFArray.energies already are); sort them, and sort the "
                         "eigenvectors with them, before grouping multiplets")
    return w

def multiplets(energies, degeneracy_tol=DEGENERACY_TOL) -> tuple:
    """Which bands of a spectrum are one degenerate level, at a stated energy tolerance.

    The levels are the rows of every character table read at that k.  A spread
    inside the levels (:func:`multiplet_spread`) far below the gap between them
    (:func:`multiplet_separation`) says the tolerance was a safe choice.

    Parameters
    ----------
    energies : array_like, shape (nstate,)
        Ascending energies, as ``solve_ham`` and ``WFArray.energies`` give them.
    degeneracy_tol : float, optional
        The energy, in the units of the spectrum, within which bands are one
        level; every two bands of a level lie within it of each other.  Default
        ``1e-8``.

    Returns
    -------
    tuple of tuple of int
        The bands of each level, lowest first.

    Raises
    ------
    ValueError
        An empty spectrum, or one not in ascending order.
    """
    return _clusters(_ascending(energies), float(degeneracy_tol))

def multiplet_separation(energies, degeneracy_tol=DEGENERACY_TOL) -> float:
    """The smallest energy gap between distinct degenerate levels at one k.

    The gap sets how sharply anything read off the eigenvectors there is known:
    it is the ``Δ`` of :class:`Thresholds`.

    Parameters
    ----------
    energies : array_like, shape (nstate,)
        Ascending energies, as ``solve_ham`` and ``WFArray.energies`` give them.
    degeneracy_tol : float, optional
        The energy within which bands are one level, as in :func:`multiplets`.
        Default ``1e-8``.

    Returns
    -------
    float
        The smallest gap between consecutive levels, in the units of the
        spectrum; ``inf`` where the spectrum is one level.
    """
    w = _ascending(energies)
    mult = _clusters(w, float(degeneracy_tol))
    if len(mult) <= 1:
        return float("inf")
    return float(min(w[c[0]] - w[p[-1]] for p, c in zip(mult[:-1], mult[1:])))

def multiplet_spread(energies, degeneracy_tol=DEGENERACY_TOL) -> float:
    """The largest energy spread inside any one degenerate level.

    A grouping is safe where this is far below :func:`multiplet_separation`:
    the bands called degenerate lie much closer to each other than to anything
    else.

    Parameters
    ----------
    energies : array_like, shape (nstate,)
        Ascending energies, as ``solve_ham`` and ``WFArray.energies`` give them.
    degeneracy_tol : float, optional
        The energy within which bands are one level, as in :func:`multiplets`.
        Default ``1e-8``.

    Returns
    -------
    float
        The largest spread inside a level, in the units of the spectrum.
    """
    w = _ascending(energies)
    return float(max(w[c[-1]] - w[c[0]] for c in _clusters(w, float(degeneracy_tol))))

# ==========================================================================
# resolving the single eigenvector source
# ==========================================================================

def solve_at(model, k, *, params=None):
    r"""The energies and eigenvectors at one k, from one diagonalization.

    Hand the pair to :func:`sewing_matrix`, :func:`characters` or
    :func:`coreps_at` as ``states=`` to read several sets of bands at that
    k off the same states.  A second diagonalization chooses another basis
    inside each degenerate level, and a character read across the two is wrong.

    Parameters
    ----------
    model : pythtb.TBModel
        The model, ``r.model``.
    k : array_like, shape (dim_k,)
        One k-point, in reduced coordinates.
    params : dict, optional
        The scalar value of each parameter of the model, keyed by name:
        ``{"lam": 0.3}``.

    Returns
    -------
    (ndarray, ndarray)
        ``(evals, evecs)``: the energies, ascending, and the eigenvectors as
        rows, as ``model.solve_ham`` returns them with ``return_eigvecs=True``
        and ``flatten_spin_axis=True``, for the model with its parameters
        evaluated at ``params``.

    Raises
    ------
    ValueError
        A name in ``params`` that is not a parameter of the model, a parameter
        given no value, or a value that is not a number.
    """
    model = _bloch.bound(model, params)
    return _bloch.solve(model, np.atleast_2d(np.asarray(k, dtype=float).reshape(-1)))

def _as_columns(states) -> np.ndarray:
    """A fiber of eigenvector **rows** -> the matrix whose **columns** they are.

    Accepts pythtb's two fiber layouts — ``(nband, norb*nspin)`` (``solve_ham``
    with ``flatten_spin_axis=True``, or ``WFArray.states(flatten_spin_axis=True)``)
    and ``(nband, norb, nspin)`` (:attr:`pythtb.WFArray.wfs`) — and returns
    ``V`` with ``H V = V diag(w)``, which is what a sewing sandwich wants.
    """
    arr = np.asarray(states)
    if arr.ndim == 3:
        arr = arr.reshape(arr.shape[0], -1)
    if arr.ndim != 2:
        raise ValueError(
            f"eigenvector fibers are (nband, norb[, nspin]) arrays of rows; got shape "
            f"{np.asarray(states).shape}")
    return np.ascontiguousarray(arr.T)

def _flat_k_points(wf) -> np.ndarray:
    """A ``WFArray``'s mesh k-points, flattened in the order its fibers are stored."""
    mesh = wf.mesh
    if mesh.dim_lambda:
        raise ValueError(
            "sewing_matrix consumes a pure-k WFArray; this one carries lambda axes "
            f"({mesh.dim_lambda}), whose fibers are not indexed by k alone")
    return np.asarray(wf.k_points, dtype=float).reshape(-1, int(wf.dim_k))

def _wfarray_fiber(wf, k, *, mesh_tol=MESH_TOL):
    """``(energies_or_None, V)`` at ``k`` from a solved ``WFArray``, or a Refusal.

    Never re-solves: a ``k`` the mesh does not carry is a refusal
    (``gate="k_not_on_mesh"``) with the distance to the nearest mesh point,
    because silently diagonalizing somewhere else is exactly the second
    diagonalization the one-solve discipline exists to prevent.
    """
    k = np.asarray(k, dtype=float).reshape(-1)
    pts = _flat_k_points(wf)
    if pts.shape[1] != k.size:
        raise ValueError(f"the WFArray's mesh is {pts.shape[1]}-dimensional in k but the "
                         f"requested point has {k.size} components")
    dist = np.abs(pts - k).max(axis=1)
    j = int(np.argmin(dist))
    if float(dist[j]) > float(mesh_tol):
        return Refusal(
            "k_not_on_mesh",
            f"the WFArray's mesh does not carry k = {np.round(k, 8).tolist()}; the nearest "
            f"of its {pts.shape[0]} points is {np.round(pts[j], 8).tolist()} at distance "
            f"{float(dist[j]):.3g}. Sewing there would need a second diagonalization at a "
            f"different point, which would corrupt the character: either build the mesh so "
            f"it contains this k "
            f"(groups.k_grid reports which high-symmetry points a mesh hits) or hand in the "
            f"model and let this call solve at k itself.",
            value=float(dist[j]), threshold=float(mesh_tol),
            context={"k": k, "nearest": pts[j], "npoints": int(pts.shape[0])})
    idx = np.unravel_index(j, tuple(wf.shape_mesh))
    V = _as_columns(np.asarray(wf.wfs)[idx])
    try:
        w = np.asarray(wf.energies, dtype=float)[idx].reshape(-1)
    except ValueError:
        w = None                      # a sector WFArray carries states, not energies
    return w, V

def _resolve_states(source, rep, k, *, mesh_tol=MESH_TOL, bloch=None):
    """The **single** eigenvector source at ``k`` -> ``(energies_or_None, V)``.

    ``source`` is a solved :class:`pythtb.WFArray`, the ``(evals, evecs)`` pair
    :meth:`pythtb.TBModel.solve_ham` returns, a bare fiber of eigenvector rows,
    or ``None``/``rep._model`` (solve here, storing nothing).  There is exactly
    one slot, which is what makes a second diagonalization unreachable by
    accident.  ``bloch`` is as in :func:`_hamiltonian_at`.
    """
    k = np.asarray(k, dtype=float).reshape(-1)

    if source is None or source is rep._model:
        evals, evecs = _bloch.solve(rep._model, np.atleast_2d(k), bloch)
        return np.asarray(evals, dtype=float).reshape(-1), _as_columns(evecs)

    if hasattr(source, "wfs") and hasattr(source, "mesh"):
        model = getattr(source, "_model", None)
        if model is not None and model is not rep._model:
            raise ValueError(
                "the WFArray was solved on a different TBModel than r.model: the "
                "representation's covariance was measured on r.model and its matrices "
                "mean nothing on another one")
        return _wfarray_fiber(source, k, mesh_tol=mesh_tol)

    if isinstance(source, (tuple, list)) and len(source) == 2:
        evals, evecs = source
        return np.asarray(evals, dtype=float).reshape(-1), _as_columns(evecs)

    if hasattr(source, "solve_ham"):
        raise ValueError(
            "the states source must be r.model (or None, meaning it): the "
            "representation's covariance was measured on r.model and on nothing else")

    return None, _as_columns(source)

def _hamiltonian_at(rep, k, bloch=None) -> np.ndarray:
    """``H(k)`` from ``rep._model``, Hermitized — recomputed on demand, never stored.

    ``bloch``, where it is given, is the model's Bloch sum, as
    :func:`.._bloch.evaluator` returns it.
    """
    k = np.asarray(k, dtype=float).reshape(-1)
    H = np.asarray(_bloch.hamiltonian(rep._model, np.atleast_2d(k), bloch))[0]
    return 0.5 * (H + H.conj().T)

def _measured_sum(rep, bloch=None):
    """The Bloch sum of ``rep._model``, checked against ``rep``.

    A ``bloch`` handed in has been checked already, and is returned as it is.
    """
    if bloch is None:
        bloch = _bloch.evaluator(rep._model)
        rep._check_unedited(bloch)
    return bloch

def _states_at_k(H, w, V):
    """``max |H V - V diag(w)|`` — the measured "these states are at this k" number."""
    return float(np.abs(H @ V - V * np.asarray(w, dtype=float)[None, :]).max())

def _foreign_states(H, k, residual):
    """The ``states_not_at_k`` refusal, or ``None`` when the residual closes."""
    gate = STATES_AT_K_GATE * max(1.0, float(np.linalg.norm(H, 2)))
    if residual <= gate:
        return None
    return Refusal(
        "states_not_at_k",
        f"the supplied (evals, evecs) do not diagonalize H(k) at "
        f"k = {np.round(np.asarray(k, float).reshape(-1), 8).tolist()}: "
        f"max |H V - V diag(w)| = {residual:.3g}. Sewing sandwiches bras and kets from one "
        f"diagonalization at one k; states from a second solve, at the image point or "
        f"anywhere else, produce a matrix that is still unitary and whose trace is "
        f"silently wrong.",
        value=float(residual), threshold=float(gate), context={"k": np.asarray(k, float)})

def _rep_of(call, realized, example):
    """The representation of what :func:`..groups.realized_group` returned, or a refusal handed in as it is.

    ``TypeError`` for anything else, saying what to pass: ``example`` is the
    call written with ``r``.
    """
    from ._realized import RealizedGroup
    from .symmetry import SpaceGroupRep

    if isinstance(realized, Refusal):
        return realized
    if isinstance(realized, RealizedGroup):
        return realized.rep
    if isinstance(realized, SpaceGroupRep):
        raise TypeError(f"{call}(): pass what realized_group returns, as in {example}, "
                        "not r.rep")
    raise TypeError(f"{call}(): pass what realized_group returns, as in {example}; got a "
                    f"{type(realized).__name__}")


def _band_block(nstate, bands, call=None):
    """The sorted band indices a band count or a band sequence names.

    An integer ``n`` means the lowest ``n`` bands; a sequence means exactly
    those band indices.  ``None`` means every band of the fiber.  ``call``,
    where it is given, names the public call in the ``ValueError`` an empty
    sequence raises.
    """
    nstate = int(nstate)
    if bands is None:
        return tuple(range(nstate))
    if np.isscalar(bands):
        n = int(bands)
        if not 0 < n <= nstate:
            raise ValueError(f"the band count {n} is outside 1..{nstate}")
        return tuple(range(n))
    idx = [int(b) for b in bands]
    if not idx and call is not None:
        raise ValueError(f"{call}(): bands= names no band; pass the number of bands, "
                         "bands=2, or their indices, bands=[0, 1].")
    if len(set(idx)) != len(idx):
        raise ValueError(f"the bands {idx} name a band more than once")
    if idx and not (0 <= min(idx) and max(idx) < nstate):
        raise ValueError(f"the bands {idx} are not all among the {nstate} bands "
                         f"0..{nstate - 1}")
    return tuple(sorted(idx))

# ==========================================================================
# the k-point a call on bands reads
# ==========================================================================

#: How far a point may sit from a tabulated one and be named after it, in
#: reduced reciprocal coordinates: the tables' own reading of a point.
_KTOL = _tables._corep_table.__kwdefaults__["ktol"]

def _in_plane(group, k, call):
    """``k`` as ``dim_k`` reduced coordinates, or the refusal ``out_of_plane``.

    Three coordinates of a layer's three-dimensional cell are read in its
    plane, as :func:`.._tables.little_group` reads them, and a point with a
    component out of the plane returns its refusal ``out_of_plane``.  Any
    other ``k`` is returned as a flat array.
    """
    k = np.asarray(k, dtype=float).reshape(-1)
    if k.size == 3 and int(group._dim_k) != 3:
        lg = _tables._little_group(group, k, call=call)
        if isinstance(lg, Refusal):
            return lg
        return np.asarray(lg.k, dtype=float).reshape(-1)
    return k

def _point_of(call, group, k):
    """The coordinates a call on bands reads ``k`` at, or the refusal it returns there.

    ``k`` is a tabulated name or coordinates.  A name is read as the
    coordinates :func:`.._tables.high_symmetry_points` gives it
    (:func:`.._tables._resolve_kpoint`, which raises ``ValueError``, starting
    with ``call``, on a name the group does not tabulate); coordinates of a
    layer's three-dimensional cell are read in its plane (:func:`_in_plane`).
    """
    k = _tables._resolve_kpoint(call, group, k)
    if isinstance(k, Refusal):
        return k
    return _in_plane(group, k, call)

def _tabulated_name(group, k):
    """The tables' name of the point ``k``, ``dim_k`` reduced coordinates, or ``None`` where it is none of their points.

    Matched as the identification of coreps matches a point
    (:func:`.._tables._resolve_kpname`): the first tabulated point it differs
    from by a reciprocal lattice vector of the cell as written, each
    coordinate to within :data:`_KTOL`.  ``None`` also where the group's tables
    cannot be read in this cell.
    """
    if _spacegroup._lattice_not_descended(group) is not None or not group._tables_ready:
        return None
    k3 = group._k_to_3d(np.asarray(k, dtype=float).reshape(1, -1))[0]
    for name, tabulated in _tables._table_kpoints(group).items():
        if _tables._same_point(k3, tabulated, _KTOL):
            return name
    return None

def _kpoint_text(kpoint, k) -> str:
    """A k-point as the summaries print it: ``GM (0, 0)``, or ``k = (0.1, 0.5)`` where it has no name."""
    return f"{kpoint} {_k_text(k)}" if kpoint else f"k = {_k_text(k)}"

def _seitz_of(group, ind) -> str:
    """One operation's Seitz symbol, ``{4+_001|0,0,0}``, or ``""`` where the group has no conventional setting to write it in."""
    rows = _operation_rows(group, (int(ind),))
    if isinstance(rows, Refusal) or int(ind) not in rows:
        return ""
    return str(rows[int(ind)].seitz)

# ==========================================================================
# the sewing matrix
# ==========================================================================

@dataclass(frozen=True)
class SewingMatrixNumerical:
    """How closely a sewing matrix is measured: what :attr:`SewingMatrix.numerical` holds.

    ``summary()`` prints it.

    Attributes
    ----------
    unitarity : float
        ``max |B†B − 1|``: zero where the set of bands is closed under the
        operation.
    commutator : float
        ``max |U H(k) − H(k) U|``, ``U`` the operator on the model's states:
        zero where ``U`` is a symmetry of ``H(k)``.
    residual : float
        ``max |H(k) V − V diag(w)|``, how far the states are from eigenstates
        at ``k``.  ``nan`` where they came without energies, and then neither
        this nor whether the bands close their multiplets was measured.
    """

    unitarity: float
    commutator: float
    residual: float = float("nan")

    def summary(self) -> str:
        """The three numbers, on one line, with the attributes that hold them."""
        measured = [f"unitary to {self.unitarity:.2g}"]
        held = [".unitarity"]
        if np.isfinite(self.commutator):
            measured.append(f"the operator commuting with H(k) to {self.commutator:.2g}")
            held.append(".commutator")
        if np.isfinite(self.residual):
            measured.append(f"the states solving H(k) to {self.residual:.2g}")
            held.append(".residual")
        return ", ".join(measured) + f"   ({', '.join(held)})"

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return (f"SewingMatrixNumerical(unitarity={self.unitarity:.2e}, "
                f"commutator={self.commutator:.2e}, residual={self.residual:.2e})")


@dataclass(frozen=True)
class SewingMatrix:
    """The matrix of one operation on a set of bands, at a k-point the operation leaves invariant: ``B(g,k) = V† U(g;k) V``.

    Its trace over a degenerate multiplet is that multiplet's character.  The
    set of bands holds whole degenerate multiplets, so that ``B`` is the
    operation's own and not the basis the diagonalization chose.
    ``summary()`` prints it.

    Attributes
    ----------
    operation : int
        The operation, by its index.
    seitz : str
        Its Seitz symbol, ``{4+_001|0,0,0}``, primed where the operation is
        combined with time reversal.
    bands : tuple of int
        The set of bands, by position among the states handed in: for a
        ``WFArray`` of a sector, that sector's states.
    k : ndarray
        The k-point, in the model's reduced coordinates.
    kpoint : str or None
        The tables' name of the k-point, ``"M"``, or ``None`` where it is no
        tabulated point.
    G : ndarray of int
        The reciprocal lattice vector ``G = Rk − k``.  On the zone boundary it
        brings the factor ``e^{2πi G·τ}`` of an orbital at ``τ`` off the
        origin, which reads that orbital's characters right there.
    matrix : ndarray, shape (nb, nb)
        ``B``.
    eigenvalues : ndarray
        ``B``'s eigenvalues, on the unit circle where ``B`` is unitary.
    numerical : SewingMatrixNumerical
        How far ``B`` is from unitary, the operator from commuting with
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
        """The sewing matrix, a line for each part, with the attribute that holds it."""
        held = ".operation, .seitz, .bands" if self.seitz else ".operation, .bands"
        held += ", .kpoint, .k" if self.kpoint else ", .k"
        lines = [f"{self._named()}, on band{'s' if len(self.bands) > 1 else ''} "
                 f"{_runs(self.bands)} at {_kpoint_text(self.kpoint, self.k)}   ({held})"]
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

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return (f"SewingMatrix({self._named()}, {len(self.bands)} bands, "
                f"unitarity={self.numerical.unitarity:.2e})")

def _sewing_core(rep, operation, k, V, idx, H, *, atol=1e-8,
                 residual=float("nan")) -> SewingMatrix | Refusal:
    """``B = V†UV`` on ``idx`` with its two self-checks — the shared inner loop.

    Private on purpose: it is the piece :func:`characters` reuses across every
    column of one table, so that a table is built from **one** fiber and one
    ``H(k)``.  It stores nothing; ``V``, ``H`` and ``idx`` are local arrays
    handed in.  The record returned has no Seitz symbol and no name of the
    point, which :func:`_sewing_matrix_rep` puts in.
    """
    U = rep.sewing_operator(int(operation), k, _atol=atol)
    if isinstance(U, Refusal):
        return U

    Vb = V[:, list(idx)]
    op = rep._operation(int(operation))
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
                       degeneracy_tol=DEGENERACY_TOL, _atol=1e-8,
                       _mesh_tol=MESH_TOL) -> SewingMatrix | Refusal:
    """What :func:`sewing_matrix` returns, read off a representation: ``realized.rep``, or one built by hand.

    Every argument but the first is :func:`sewing_matrix`'s own.
    """
    if operation is None:
        raise TypeError("sewing_matrix() missing required argument 'operation'")
    if k is None:
        raise TypeError("sewing_matrix() missing required argument 'k'")
    k = _point_of("sewing_matrix", rep._group, k)
    if isinstance(k, Refusal):
        return k
    bloch = (_measured_sum(rep) if states is None or states is rep._model
             else None)
    got = _resolve_states(states, rep, k, mesh_tol=_mesh_tol, bloch=bloch)
    if isinstance(got, Refusal):
        return got
    w, V = got
    idx = _band_block(V.shape[1], bands, "sewing_matrix")

    bloch = _measured_sum(rep, bloch)
    H = _hamiltonian_at(rep, k, bloch)
    resid = float("nan") if w is None else _states_at_k(H, w, V)

    if w is not None:
        foreign = _foreign_states(H, k, resid)
        if foreign is not None:
            return foreign
        chosen = set(idx)
        for m in multiplets(w, degeneracy_tol):
            inside = chosen & set(m)
            if inside and len(inside) != len(m):
                return Refusal(
                    "multiplet_split",
                    f"the bands {sorted(idx)} cut through the degenerate multiplet "
                    f"{list(m)} at k = {np.round(k, 6).tolist()} (energies "
                    f"{np.round(w[list(m)], 12).tolist()}): on part of a degenerate "
                    f"multiplet the sewing matrix depends on the basis chosen inside the "
                    f"multiplet. Ask for every band of the multiplet, or end the bands at "
                    f"a gap.",
                    value=float(w[m[-1]] - w[m[0]]),
                    threshold=float(degeneracy_tol),
                    context={"bands": idx, "multiplet": m, "k": k})

    out = _sewing_core(rep, operation, k, V, idx, H, atol=_atol, residual=resid)
    if isinstance(out, Refusal):
        return out
    return replace(out, seitz=_seitz_of(rep._group, operation),
                   kpoint=_tabulated_name(rep._group, k))

def sewing_matrix(realized, operation=None, k=None, *, states=None, bands=None,
                  degeneracy_tol=DEGENERACY_TOL,
                  _atol=_sewing_matrix_rep.__kwdefaults__["_atol"],
                  _mesh_tol=MESH_TOL) -> SewingMatrix | Refusal:
    """The matrix of one operation on a set of bands, at a k-point the operation leaves invariant: the sewing matrix ``B(g,k)_mn = <u_m(k)|U(g;k)|u_n(k)>``.

    Its trace over a degenerate multiplet is that multiplet's character, and its
    eigenvalues are the operation's on those bands.  The states are checked to
    solve ``H(k)``, and the bands to hold whole degenerate multiplets, wherever
    the states carry their energies.

    Parameters
    ----------
    realized : ..groups.RealizedGroup
        What :func:`..groups.realized_group` returned, ``r``.
    operation : int
        The operation, by its index: ``r.operations`` lists them, and
        :func:`..groups.operation_index` finds one by its Seitz symbol.
        Required.
    k : str or array_like, shape (dim_k,)
        A k-point the operation leaves invariant up to a reciprocal lattice
        vector: its tabulated name, ``"M"``, or its reduced coordinates.
        Required.
    bands : int or sequence of int, optional
        The set of bands: ``n`` the lowest ``n`` bands, or the band indices.
        Default: every state.
    states : None, pythtb.WFArray, (evals, evecs) or ndarray, optional
        The states at ``k``: ``None`` (default) solves the model there; a
        solved ``WFArray``, or the pair :func:`solve_at` returns, reuses one
        diagonalization.  Eigenvector rows alone, or a ``WFArray`` of a sector
        with fewer states than the model, carry no energies.
    degeneracy_tol : float, default 1e-8
        The energy, in the units of ``H``, within which bands are one
        multiplet.

    Returns
    -------
    SewingMatrix or Refusal
        ``B.summary()`` prints it.  ``B.matrix`` is the matrix and
        ``B.eigenvalues`` its eigenvalues; ``B.numerical`` holds how far it is
        from unitary, the operator from commuting with ``H(k)``, and the states
        from eigenstates.  For states without energies, whether they solve
        ``H(k)`` and whether the bands close their multiplets is not measured.
        A ``Refusal`` in its place is falsy and names what stopped the call.

    Refusals
    --------
    ``little_group``
        the operation does not leave ``k`` invariant.
    ``multiplet_split``
        ``bands`` cuts a degenerate multiplet at ``k``.
    ``states_not_at_k``
        the states handed in do not solve ``H(k)``.
    ``k_not_on_mesh``
        the mesh handed in does not hold ``k``.
    ``out_of_plane``
        ``k`` has a component along a layer's vacuum, or names a tabulated
        point out of the layer's plane.
    ``tables_not_ready``
        ``k`` is a name, and the group has no transformation to the tables'
        conventional cell.

    Raises
    ------
    TypeError
        ``realized`` is not what :func:`..groups.realized_group` returned.
    TypeError
        ``operation`` or ``k`` left out.
    ValueError
        States of another model than ``r.model``, ``bands`` outside the
        states or naming none, ``k`` a name the group does not tabulate, or
        ``r.model`` edited since :func:`..groups.realized_group` measured it.

    Examples
    --------
    The fourfold rotation on lin3g's lowest pair at ``M``
    (:func:`..models.lin3g`):

    >>> c4 = operation_index(r.group, "{4+_001|0,0,0}")             # doctest: +SKIP
    >>> B = sewing_matrix(r, c4, "M", bands=2)                         # doctest: +SKIP
    >>> np.sort(np.angle(B.eigenvalues)) / np.pi                       # doctest: +SKIP
    array([-0.75,  0.75])
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
    """The characters of one degenerate multiplet: a row of :attr:`BandCharacters.multiplets`.

    Attributes
    ----------
    bands : tuple of int
        The multiplet's bands.
    energy : float
        Its mean energy, in the units of ``H``.
    characters : dict
        ``{index: character}``: the trace over the multiplet of each unitary
        operation of the little group, by the operation's index, the indices
        :func:`..groups.corep_table` keys its characters by.
    """

    bands: tuple
    energy: float
    characters: dict
    # the energy tolerance the multiplet was grouped at, which its energy is
    # printed at
    _degeneracy_tol: float = field(default=DEGENERACY_TOL, repr=False, compare=False)

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        energy = float(_resolved([self.energy], self._degeneracy_tol)[0])
        return (f"MultipletCharacters(band{'s' if len(self.bands) > 1 else ''} "
                f"{_runs(self.bands)} at {energy:.4g})")


@dataclass(frozen=True)
class BandCharactersNumerical:
    """How closely the characters of a set of bands are measured: what :attr:`BandCharacters.numerical` holds.

    ``summary()`` prints it.

    Attributes
    ----------
    unitarity, commutators : dict
        Per unitary operation, by index: how far its sewing matrix on the
        whole set of bands is from unitary, and how far its operator is from
        commuting with ``H(k)``.
    residual : float
        How far the states are from eigenstates of ``H(k)``.
    spread : float
        The largest energy spread inside a multiplet; a safe grouping has it
        far below the gap between multiplets.
    thresholds : Thresholds
        The zero thresholds at this k.
    disagreements : tuple
        The numbers the two thresholds classify differently.
    degeneracy_tol : float
        The energy tolerance the multiplets were grouped at.
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
        """How closely the characters hold, a line for each part, with the attribute that holds it."""
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

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return (f"BandCharactersNumerical(residual={self.residual:.2e}, "
                f"spread={self.spread:.2e}, {len(self.disagreements)} disagreements)")


@dataclass(frozen=True)
class BandCharacters:
    """The characters of a set of bands at one k: the trace of each unitary operation of the little group over each degenerate multiplet.

    Every corep label is read from such characters.  ``summary()`` prints
    them.

    Attributes
    ----------
    bands : tuple of int
        The set of bands.
    k : ndarray
        The k-point, in the model's reduced coordinates.
    kpoint : str or None
        The tables' name of the k-point, ``"GM"``, or ``None`` where it is no
        tabulated point.
    multiplets : tuple of MultipletCharacters
        A row for each degenerate multiplet of the set, lowest first: its
        bands, its mean energy and its characters by operation index.
    operations : ..groups.OperationTable
        The operations of the little group, unitary and antiunitary, by
        index, each with its Seitz symbol.  The rows' characters are keyed by
        the unitary ones: a corep table holds no characters for an operation
        combined with time reversal.
    numerical : BandCharactersNumerical
        How far the sewing matrices are from unitary, the operators from
        commuting with ``H(k)`` and the states from eigenstates, and the zero
        thresholds at this k.
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
        """The characters, a line for each part, with the attribute that holds it."""
        held = ".bands, .kpoint, .k" if self.kpoint else ".bands, .k"
        lines = [f"band{'s' if len(self.bands) > 1 else ''} {_runs(self.bands)} at "
                 f"{_kpoint_text(self.kpoint, self.k)}   ({held})"]
        columns = tuple(self.multiplets[0].characters) if self.multiplets else ()
        if columns:
            lines.append("  the trace of each unitary operation of the little group, by index, "
                         "over each degenerate multiplet   (.multiplets: characters of each; "
                         ".operations)")
            energies = _resolved([m.energy for m in self.multiplets],
                                 self.numerical.degeneracy_tol)
            rows = [("bands", [_runs(m.bands) for m in self.multiplets],
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

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        n = len(self.multiplets)
        return (f"BandCharacters({_kpoint_text(self.kpoint, self.k)}, bands "
                f"{_runs(self.bands)} in {n} multiplet{'' if n == 1 else 's'})")

def _characters_rep(rep, k, *, states=None, bands=None, degeneracy_tol=DEGENERACY_TOL,
                    _atol=1e-8, _mesh_tol=MESH_TOL) -> BandCharacters | Refusal:
    """What :func:`characters` returns, read off a representation: ``realized.rep``, or one built by hand.

    Every argument but the first is :func:`characters`'s own.
    """
    return _characters(rep, k, states=states, bands=bands, degeneracy_tol=degeneracy_tol,
                       atol=_atol, mesh_tol=_mesh_tol, bloch=None, call="characters")

def characters(realized, k, *, states=None, bands=None, degeneracy_tol=DEGENERACY_TOL,
               _atol=_characters_rep.__kwdefaults__["_atol"],
               _mesh_tol=MESH_TOL) -> BandCharacters | Refusal:
    """The characters of a set of bands at one k: the trace of each unitary operation of the little group over each degenerate multiplet of the set.

    Parameters
    ----------
    realized : ..groups.RealizedGroup
        What :func:`..groups.realized_group` returned, ``r``.
    k : str or array_like, shape (dim_k,)
        A k-point: its tabulated name, ``"GM"``, or its reduced coordinates.
        Only a tabulated point is labeled, by :func:`coreps_at`.
    bands : int or sequence of int
        The set of bands: ``n`` the lowest ``n`` bands, or the band indices.
        Required.
    states : None, pythtb.WFArray or (evals, evecs), optional
        The states at ``k``: ``None`` (default) solves the model there; a
        solved ``WFArray``, or the pair :func:`solve_at` returns, reuses one
        diagonalization.  They carry energies, the rows being the multiplets.
    degeneracy_tol : float, default 1e-8
        The energy, in the units of ``H``, within which bands are one
        multiplet.

    Returns
    -------
    BandCharacters or Refusal
        ``ch.summary()`` prints them.  ``ch.multiplets`` holds a row for each
        degenerate multiplet: its bands, its mean energy and its characters by
        operation index.  ``ch.operations`` names the operations of the little
        group; ``ch.numerical`` holds how closely the characters are measured.
        Characters can only be returned if the set of bands contains all the
        states of each degenerate multiplet at ``k``.  A ``Refusal`` in its
        place is falsy and names what stopped the call.

    Refusals
    --------
    A refusal handed in as ``realized`` is returned unchanged, and one from
    :func:`..groups.little_group` as it is.

    ``multiplet_split``
        the set of bands cuts a degenerate multiplet at ``k``.
    ``states_not_at_k``
        the states handed in do not solve ``H(k)``.
    ``k_not_on_mesh``
        the mesh handed in does not hold ``k``.
    ``out_of_plane``
        ``k`` has a component along a layer's vacuum, or names a tabulated
        point out of the layer's plane.
    ``tables_not_ready``
        ``k`` is a name, and the group has no transformation to the tables'
        conventional cell.

    Raises
    ------
    TypeError
        ``realized`` is not what :func:`..groups.realized_group` returned, or
        ``bands`` left out.
    ValueError
        States without energies or of another model than ``r.model``,
        ``bands`` naming no band, ``k`` a name the group does not tabulate, or
        ``r.model`` edited since :func:`..groups.realized_group` measured it.
    ..refusal.InternalError
        The representation leaves out an operation of the little group at
        ``k``: the package disagreeing with itself.
    """
    rep = _rep_of("characters", realized, "characters(r, k, bands=6)")
    if isinstance(rep, Refusal):
        return rep
    return _characters_rep(rep, k, states=states, bands=bands, degeneracy_tol=degeneracy_tol,
                           _atol=_atol, _mesh_tol=_mesh_tol)

def _characters(rep, k, *, states, bands, degeneracy_tol, atol, mesh_tol,
                bloch, call, named=True) -> BandCharacters | Refusal:
    """The characters :func:`characters` returns.

    ``bloch`` is the model's Bloch sum or ``None``, as in :func:`_hamiltonian_at`;
    ``call`` names the public call in what is raised.  With ``named=False`` the
    point's name is left ``None``, to be read off the identification of the
    coreps.
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
        bloch = _measured_sum(rep, bloch)
    got = _resolve_states(states, rep, k, mesh_tol=mesh_tol, bloch=bloch)
    if isinstance(got, Refusal):
        return got
    w, V = got
    if w is None:
        raise ValueError(
            "characters needs energies: its rows are the degenerate multiplets at k, so a "
            "source that carries states but no spectrum (a feature-sector WFArray, a bare "
            "fiber) cannot define the table. Use sewing_matrix() on such a source, or hand "
            "in the model / the (evals, evecs) pair.")
    idx = _band_block(V.shape[1], bands, call)

    bloch = _measured_sum(rep, bloch)
    H = _hamiltonian_at(rep, k, bloch)
    resid = _states_at_k(H, w, V)
    foreign = _foreign_states(H, k, resid)
    if foreign is not None:
        return foreign

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
                f"the bands {sorted(idx)} cut through the degenerate multiplet {list(m)} at "
                f"k = {np.round(k, 6).tolist()} (energies "
                f"{np.round(w[list(m)], 12).tolist()}, spread "
                f"{w[m[-1]] - w[m[0]]:.3g} at tolerance "
                f"{float(degeneracy_tol):g}): on part of a degenerate multiplet the "
                f"characters depend on the basis chosen inside the multiplet. Ask for every "
                f"band of the multiplet, or end the bands at a gap.",
                value=float(w[m[-1]] - w[m[0]]),
                threshold=float(degeneracy_tol),
                context={"bands": idx, "multiplet": m, "k": k,
                         "energies": w[list(m)]})
        rows.append(m)

    lg = _tables._little_group(rep._group, k)
    if isinstance(lg, Refusal):
        return lg
    ours = [int(i) for i in rep.little_group(k, _atol=atol)]
    ours_u = tuple(sorted(i for i in ours if not rep._operation(i).time_reversal))
    not_carried = tuple(sorted(set(lg.unitary) - set(ours_u)))
    if not_carried:
        # every representation a call returns carries the whole group it was
        # built for, so an operation of that group left out is the package
        # disagreeing with itself
        raise InternalError(
            f"the representation of #{rep._group.number_str} leaves out part of its own "
            f"little group at k = {np.round(k, 6).tolist()}; not carried: "
            f"{list(not_carried)}. Every representation a call returns carries the "
            f"whole group it was built for.")

    chars = np.zeros((len(rows), len(ours_u)), dtype=complex)
    unit, comm = {}, {}
    for c, ind in enumerate(ours_u):
        whole = _sewing_core(rep, ind, k, V, idx, H, atol=atol, residual=resid)
        if isinstance(whole, Refusal):  # pragma: no cover - membership tested above
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
    """The characters of a :class:`BandCharacters` as a matrix: a row per multiplet, a column per unitary operation in index order."""
    columns = tuple(table.multiplets[0].characters) if table.multiplets else ()
    return np.array([[m.characters[i] for i in columns] for m in table.multiplets],
                    dtype=complex).reshape(len(table.multiplets), len(columns))

# ==========================================================================
# the coreps at one point
# ==========================================================================

def format_coreps(coreps) -> str:
    r"""Corep multiplicities on one line: ``{'-GM6': 2, '-GM7': 1}`` as ``2x-GM6 + -GM7``.

    Every multiplicity the package prints reads this way: the coreps of a set
    of bands at one point (:attr:`CorepsAt.coreps`) or of one multiplet
    (:attr:`Multiplet.label`), a symmetry data vector
    (:attr:`HSPTable.symmetry_data`), an elementary band representation's
    coreps (:func:`..ebr.ebr_rows`).  The coreps are written point by point,
    Γ first and the other points alphabetically, and at each point in
    ascending order of the irreps' numbers, ``GM2`` before ``GM10``, so one
    set of multiplicities always prints one line.

    Parameters
    ----------
    coreps : mapping
        ``{corep: multiplicity}``, the multiplicities nonnegative integers.

    Returns
    -------
    str
        The labels joined by ``" + "``, each after its multiplicity and ``x``
        where that is above one.
    """
    return " + ".join(f"{n}x{lab}" if n > 1 else lab
                      for lab, n in sorted(coreps.items(),
                                           key=lambda item: _tables.corep_order(item[0])))


@dataclass(frozen=True, eq=False)
class Multiplet:
    """One degenerate multiplet at a high-symmetry point, with the coreps it carries.

    A row of :attr:`HSPTable.multiplets` and of :attr:`CorepsAt.multiplets`;
    the two hold the same rows.  Its degeneracy is ``len(m.bands)``.

    Attributes
    ----------
    kpoint : str
        The tables' name of the point, ``"GM"``.
    k : ndarray
        Its coordinates, in the model's reduced cell.
    bands : tuple of int
        The multiplet's bands.
    energy : float
        Its mean energy, in the units of ``H``.
    corep_names : tuple of str
        The coreps it carries, a corep carried twice named twice.
    label : str
        The same on one line, ``"-GM6"``, ``"-GM6 + -GM7"``, ``"2x-GM6"``.
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
        return (f"band{'s' if len(self.bands) > 1 else ''} {_runs(self.bands)} at "
                f"{energy:.4g}: {self.label}")

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return f"Multiplet({self.kpoint}, {self._line()})"


@dataclass(frozen=True)
class CorepsAtNumerical:
    """How the coreps at one point were read: what :attr:`CorepsAt.numerical` holds.

    ``summary()`` prints it.

    Attributes
    ----------
    thresholds : Thresholds
        The zero thresholds at the point.
    identification : ..groups.CorepIdentification
        The identification of the characters: each multiplet's coreps, the
        multiplicities before rounding, and how far they sat from integers.
    """

    thresholds: Thresholds
    identification: Any

    def summary(self) -> str:
        """The thresholds and the identification, each under a line naming its field."""
        lines = [f"the zero thresholds at {self.identification.kpoint}   (.thresholds)"]
        lines += ["  " + line for line in self.thresholds.summary().splitlines()]
        lines.append("the coreps read off the characters   (.identification)")
        lines += ["  " + line for line in self.identification.summary().splitlines()]
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return (f"CorepsAtNumerical({self.identification.kpoint}, integrality="
                f"{float(self.identification.integrality):.2g})")


@dataclass(frozen=True)
class CorepsAt:
    """The coreps a set of bands carries at one high-symmetry point, multiplet by multiplet.

    ``summary()`` prints them.

    Attributes
    ----------
    kpoint : str
        The tables' name of the point, ``"GM"``.
    k : ndarray
        Its coordinates, in the model's reduced cell.
    bands : tuple of int
        The set of bands.
    coreps : dict
        ``{corep: multiplicity}`` over the set of bands: its symmetry data
        vector at this point.
    multiplets : tuple of Multiplet
        A row for each degenerate multiplet, lowest first, with the coreps it
        carries.
    isolation : float
        The energy gap from the set of bands to the nearest band outside it, in
        the units of ``H``; ``inf`` where the set is every band.
    numerical : CorepsAtNumerical
        The zero thresholds at the point and the identification of the
        characters.
    """

    kpoint: str
    k: np.ndarray
    bands: tuple
    coreps: dict
    multiplets: tuple
    isolation: float
    numerical: CorepsAtNumerical
    # the characters the coreps were read from, `characters(r, k, bands=...)`
    _characters: Any = field(default=None, repr=False, compare=False)

    def summary(self) -> str:
        """The coreps at this point, a line for each part, with the attribute that holds it."""
        bands = self.bands
        lines = [f"band{'s' if len(bands) > 1 else ''} {_runs(bands)} at "
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

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return f"CorepsAt({self.kpoint}: {format_coreps(self.coreps)})"

def _isolation(spectrum, bands):
    """Energy gap from a band block to the rest of the spectrum."""
    w = np.asarray(spectrum, dtype=float).reshape(-1)
    inside = sorted(bands)
    outside = [b for b in range(len(w)) if b not in set(inside)]
    if not outside or not inside:
        return float("inf")
    return float(min(abs(w[a] - w[b]) for a in inside for b in outside))

def _coreps_at_rep(rep, k, *, states=None, bands=None, degeneracy_tol=DEGENERACY_TOL,
                   label_tol=LABEL_TOL, _ktol=1e-5, _atol=1e-8,
                   _mesh_tol=MESH_TOL) -> CorepsAt | Refusal:
    """What :func:`coreps_at` returns, read off a representation: ``realized.rep``, or one built by hand.

    Every argument but the first is :func:`coreps_at`'s own.
    """
    return _coreps_at(rep, k, states=states, bands=bands, degeneracy_tol=degeneracy_tol,
                      label_tol=label_tol, ktol=_ktol, atol=_atol, mesh_tol=_mesh_tol,
                      bloch=None)

def coreps_at(realized, k, *, states=None, bands=None, degeneracy_tol=DEGENERACY_TOL,
              label_tol=LABEL_TOL, _ktol=_coreps_at_rep.__kwdefaults__["_ktol"],
              _atol=_coreps_at_rep.__kwdefaults__["_atol"],
              _mesh_tol=MESH_TOL) -> CorepsAt | Refusal:
    """The coreps a set of bands carries at one high-symmetry point, multiplet by multiplet.

    The characters of each degenerate multiplet are measured on the bands'
    eigenvectors and expanded over the coreps of the little group.

    Parameters
    ----------
    realized : ..groups.RealizedGroup
        What :func:`..groups.realized_group` returned, ``r``.
    k : str or array_like
        A tabulated high-symmetry point: its name, ``"GM"``, one of those
        :func:`..groups.high_symmetry_points` returns, or its coordinates in
        the model's reduced cell.
    bands : int or sequence of int
        The set of bands: ``n`` the lowest ``n`` bands, or the band indices.
        Required.
    states : None, pythtb.WFArray or (evals, evecs), optional
        The states at ``k``: ``None`` (default) solves the model there; a
        solved ``WFArray``, or the pair :func:`solve_at` returns, reuses one
        diagonalization.
    degeneracy_tol : float, default 1e-8
        The energy, in the units of ``H``, within which bands are one
        multiplet.
    label_tol : float, default 1e-3
        How far a fitted multiplicity may sit from an integer, dimensionless.

    Returns
    -------
    CorepsAt or Refusal
        ``c.summary()`` prints the coreps multiplet by multiplet.  ``c.coreps``
        holds the corep multiplicities of the set, ``{corep: multiplicity}``;
        ``c.multiplets`` a row for each degenerate multiplet, lowest first,
        with its bands, its mean energy and the coreps it carries;
        ``c.isolation`` the gap from these bands to the nearest other band;
        ``c.numerical`` the thresholds and the identification the coreps were
        read with.  Corep labels can only be returned if the set of bands
        contains all the states of each degenerate multiplet at ``k``.  A
        ``Refusal`` in its place is falsy and names what stopped the call.

    Refusals
    --------
    A refusal handed in as ``realized`` is returned unchanged.

    ``multiplet_split``
        the set of bands cuts a degenerate multiplet at ``k``: label the
        multiplets whole.
    ``kpoint_not_tabulated``
        ``k`` is not a tabulated point in this cell, or not the tabulated member
        of its star.
    ``out_of_plane``
        ``k`` has a component along a layer's vacuum, or names a tabulated
        point out of the layer's plane.
    ``integer_multiplicity`` / ``multiplet_dimension``
        the characters fit no whole number of coreps, or coreps that do not fill
        their multiplet: most often a multiplet grouped wrongly.
    ``states_not_at_k`` / ``k_not_on_mesh``
        states handed in from another point, or a mesh that does not hold ``k``.
    ``little_group_mismatch`` / ``tables_not_ready``
        the tables do not apply in this cell.

    Raises
    ------
    TypeError
        ``realized`` is not what :func:`..groups.realized_group` returned, or
        ``bands`` left out.
    ValueError
        States without energies or of another model than ``r.model``,
        ``bands`` naming no band, ``k`` a name the group does not tabulate, or
        ``r.model`` edited since :func:`..groups.realized_group` measured it.
    ..refusal.InternalError
        The representation leaves out an operation of the little group at
        ``k``: the package disagreeing with itself.

    Examples
    --------
    lin3g's six lowest bands at Γ (:func:`..models.lin3g`):

    >>> c = coreps_at(r, "GM", bands=6)                                # doctest: +SKIP
    >>> print(c.summary())                                            # doctest: +SKIP
    bands 0-5 at GM (0, 0): 2x-GM6 + -GM7   (.bands, .kpoint, .k, .coreps)
      by degenerate multiplet, from the lowest   (.multiplets)
        bands 0-1 at -16.35: -GM6
        bands 2-3 at -12.38: -GM7
        bands 4-5 at -4.325: -GM6
      13.02 from these bands to the nearest other band, in the units of H   (.isolation)
    """
    rep = _rep_of("coreps_at", realized, "coreps_at(r, k, bands=6)")
    if isinstance(rep, Refusal):
        return rep
    return _coreps_at_rep(rep, k, states=states, bands=bands,
                          degeneracy_tol=degeneracy_tol, label_tol=label_tol,
                          _ktol=_ktol, _atol=_atol, _mesh_tol=_mesh_tol)

def _coreps_at(rep, k, *, states, bands, degeneracy_tol, label_tol, ktol, atol,
               mesh_tol, bloch) -> CorepsAt | Refusal:
    """The coreps :func:`coreps_at` returns.

    ``bloch`` is the model's Bloch sum or ``None``, as in :func:`_hamiltonian_at`.
    """
    if bands is None:
        raise TypeError("coreps_at() missing required argument 'bands'")
    if isinstance(rep, Refusal):
        return rep
    gate = _spacegroup._lattice_not_descended(rep._group)
    if gate is not None:
        return gate
    tab = _characters(rep, k, states=states, bands=bands, degeneracy_tol=degeneracy_tol,
                      atol=atol, mesh_tol=mesh_tol, bloch=bloch, call="coreps_at",
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
# the coreps at every high-symmetry point
# ==========================================================================

@dataclass(frozen=True)
class HSPTable:
    """The coreps a set of bands carries at every high-symmetry point: its symmetry data vector, point by point.

    The table is what :func:`..ebr.decompose`, :func:`..ebr.symmetry_indicator`
    and :func:`..srsi.srsi_of_bands` take, read in the group it carries.
    ``summary()`` prints it.

    Attributes
    ----------
    bands : tuple of int
        The band indices of the set, ascending.
    group : ..groups.GroupData
        The group whose coreps the labels name, ``r.group``.
    kpoints : dict
        Every tabulated point in the model's periodic plane, by name, with its
        reduced coordinates; a point that refused is here too.
    symmetry_data : dict
        ``{corep: multiplicity}`` over every point measured: the symmetry data
        vector.
    multiplets : dict
        ``{kpoint: tuple of Multiplet}``: a row for each degenerate multiplet
        at each point measured, lowest first, with the coreps it carries.
    complete : bool
        Whether every point was measured; a decomposition and an indicator
        refuse a table that was not.
    refusals : dict
        ``{kpoint: Refusal}`` at each point that refused.
    note : str
        If nonempty, a sentence saying where another labeling the hoppings
        leave open gives some energy multiplet of these bands, at a point the
        table labels, other labels, or a pseudospin's undeclared site irreps
        set them.
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
        """The symmetry data vector's entries at one point."""
        return {lab: n for lab, n in self.symmetry_data.items()
                if _tables.kpoint_of_label(lab) == kpoint}

    def summary(self) -> str:
        """The coreps of these bands, a line for each part, with the attribute that holds it."""
        lines = [f"band{'s' if len(self.bands) > 1 else ''} {_runs(self.bands)}   (.bands)"]
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

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        n = len(self.multiplets)
        return (f"HSPTable(bands {_runs(self.bands)}, {n} k-point{'' if n == 1 else 's'} "
                f"measured, {len(self.refusals)} refused)")

# ==========================================================================
# what a table carries for the calls that take it
# ==========================================================================

def _site_content_record(rep):
    """The site irreps a record's labels rest on, an :class:`..ebr.SiteContent`, or ``None`` without a ``rep``."""
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
    """``(normal_dir, height)`` for a layer read in a slab cell, or ``None`` for a crystal periodic in three directions.

    ``normal_dir`` is the direction of the vacuum; ``height(z)`` takes a
    reduced coordinate along it to the height from the layer's middle plane
    along the normal ``a1 × a2``, in the model's units.
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
        """A reduced coordinate along the vacuum direction as a height from the middle plane."""
        return ((float(z) + 0.5) % 1.0 - 0.5) * scale

    return normal_dir, height

def _orbital_heights(group, where):
    """Each position's height from a layer's middle plane along its normal, in the model's units, or ``None`` for a crystal periodic in three directions.

    ``where`` holds the positions in the group's three-dimensional reduced
    coordinates, a row each.
    """
    reading = _layer_height(group)
    if reading is None:
        return None
    normal_dir, height = reading
    return tuple(height(z) for z in np.atleast_2d(where)[:, normal_dir])

def _table_heights(rep):
    """The heights of ``rep``'s orbitals, :func:`_orbital_heights`, or ``None`` for a crystal."""
    if _layer_height(rep._group) is None:
        return None
    from ._matrices import _orb_positions_3d
    return _orbital_heights(rep._group, _orb_positions_3d(rep._model, rep._group))

#: The keys under which a labeling set aside can label a band otherwise: the
#: site content at the lowest angular momentum, the functions the orbitals read
#: as, and the band labels.
_RELABELLING_KEYS = ("content", "orbitals", "bands")

#: The note, where some energy multiplet of a set of bands reads otherwise under
#: another labeling the hoppings leave open.
_ANOTHER_LABELLING_RELABELS = (
    "another labeling the hoppings leave open keeps every site irrep and labels some energy "
    "multiplets of these bands differently; it keeps which bands are degenerate and the labels "
    "of all the model's bands together")

def _labelling_note(rep, block) -> str:
    """The sentence :attr:`HSPTable.note` carries, or ``""``.

    Where the lowest-angular-momentum rule kept one of several labelings of a
    coupled part that give every orbital the same site irrep, and another of
    them labels some band differently, the atomic-limit table holds the labels
    every energy multiplet carries under each.  A set of bands rests on that
    choice where an energy multiplet it holds whole, at a point where it cuts
    none, carries other labels under another of them.  Where only the group's order of operations set them
    apart, no label rests on the choice and nothing is said.
    """
    table = getattr(rep, "_characters", None)
    record = getattr(table, "_labellings", {}) or {}
    held = {int(b) for b in block}
    if any(key in _RELABELLING_KEYS and _labelled_apart(rec["chosen"], other, held)
           for rec in record.get("choices", ()) for key, other in rec["others"]):
        return _ANOTHER_LABELLING_RELABELS
    return ""

def _labelled_apart(chosen, other, held) -> bool:
    """Whether a multiplet the bands ``held`` hold whole is labeled otherwise under ``other``.

    ``chosen`` and ``other`` are ``{point: ((bands, {corep: m}), ...)}``, and one
    multiplet whose labels differ is enough.  A point where the bands held cut
    a multiplet is one the table does not label, and is passed over.
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
    """The sentence :attr:`HSPTable.note` carries where a pseudospin sets the content, or ``""``.

    Where the Hamiltonian conserves a pseudospin, the representation its copies
    carry is not fixed by it and was assigned by the lowest-angular-momentum
    rule (:attr:`..symmetry._OrbitalTable.multiplicity_spaces`), and every
    band on those orbitals carries it.  Where the representations the table
    admits give different site contents, the site content itself is that
    choice, and a set of bands with weight there is labeled by it and can read
    differently under another; where a declaration states the content there is
    no such choice.  Those giving one site content are labelings like any
    other, and :func:`_labelling_note` says where a set of bands rests on them.
    The weight is read off the eigenvectors the content was read from, where
    they are to hand, and a set of bands is taken to have some where they are
    not.
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
            V = _as_columns(fiber[1])
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

def _hsp_table_rep(rep, *, states=None, bands=None, degeneracy_tol=DEGENERACY_TOL,
                   label_tol=LABEL_TOL, _ktol=1e-5, _atol=1e-8,
                   _mesh_tol=MESH_TOL) -> HSPTable | Refusal:
    """What :func:`hsp_table` returns, read off a representation: ``realized.rep``, or one built by hand.

    Every argument but the first is :func:`hsp_table`'s own.
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
        raise ValueError("hsp_table(): states= takes r.model, the model realized_group "
                         "measured, or nothing")
    points = _tables._high_symmetry_points(rep._group)
    if isinstance(points, Refusal):
        return points

    rows, records, refusals = {}, {}, {}
    counts: dict = {}

    # Collect every point whose little group is read, then take *one* pass over them: on the
    # model path that is a single solve_ham over the whole k-list -- the convenient
    # route for "a handful of high-symmetry points", with no container built.
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
        bloch = _measured_sum(rep)
        evals, evecs = _bloch.solve(rep._model, karr, bloch)
        evals, evecs = np.asarray(evals), np.asarray(evecs)
        if evals.ndim == 1:                     # solve_ham drops a length-1 k axis
            evals, evecs = evals[None], evecs[None]
        per_point = {name: (evals[i], evecs[i]) for i, (name, _) in enumerate(wanted)}
    else:
        per_point = {name: states for name, _ in wanted}

    for name, k in wanted:
        got = _coreps_at(rep, k, states=per_point[name], bands=block,
                         degeneracy_tol=degeneracy_tol, label_tol=label_tol, ktol=_ktol,
                         atol=_atol, mesh_tol=_mesh_tol, bloch=bloch)
        if isinstance(got, Refusal):
            refusals[name] = got
            continue
        records[name] = got
        rows[name] = got.multiplets
        for lab, v in got.coreps.items():
            counts[lab] = counts.get(lab, 0) + int(v)

    notes = (_multiplicity_note(rep, block, per_point.values()),
             _labelling_note(rep, block))
    return HSPTable(bands=block, group=rep._group, kpoints=dict(points),
                    symmetry_data=_tables.ordered_content(counts), multiplets=rows,
                    complete=not refusals, refusals=refusals,
                    note="; ".join(n for n in notes if n), _records=records,
                    _group_provenance=str(rep._group._provenance),
                    _rep=rep if int(rep._group._dim_k) < 3 else None,
                    _site_content=_site_content_record(rep), _heights=_table_heights(rep))

def hsp_table(realized, *, states=None, bands=None, degeneracy_tol=DEGENERACY_TOL,
              label_tol=LABEL_TOL, _ktol=_hsp_table_rep.__kwdefaults__["_ktol"],
              _atol=_hsp_table_rep.__kwdefaults__["_atol"],
              _mesh_tol=MESH_TOL) -> HSPTable | Refusal:
    """The coreps a set of bands carries at every high-symmetry point of the group: its symmetry data vector.

    Parameters
    ----------
    realized : ..groups.RealizedGroup
        What :func:`..groups.realized_group` returned, ``r``.
    bands : int or sequence of int
        The set of bands: ``n`` the lowest ``n`` bands, or the band indices.
        Required.
    states : None or pythtb.WFArray, optional
        ``None`` (default) solves the model at the points, once; a solved
        ``WFArray`` is read as it stands, and a point its mesh does not hold is
        refused there.
    degeneracy_tol : float, default 1e-8
        The energy, in the units of ``H``, within which bands are one
        multiplet.
    label_tol : float, default 1e-3
        How far a fitted multiplicity may sit from an integer at each point,
        dimensionless.

    Returns
    -------
    HSPTable or Refusal
        ``t.summary()`` prints the coreps at each point.  ``t.symmetry_data``
        is the symmetry data vector, ``t.multiplets`` a row for each
        degenerate multiplet at each point, with the coreps it carries;
        ``t.complete`` says whether every point was measured, and
        ``t.refusals`` holds each point that was not, with its own
        ``Refusal``; ``t.note`` says where the labels rest on a choice the
        hoppings leave open.  Corep labels can only be returned at a point if
        the set of bands contains all the states of each degenerate multiplet
        there.  The table is what :func:`..ebr.decompose`,
        :func:`..ebr.symmetry_indicator` and :func:`..srsi.srsi_of_bands`
        take, ``decompose(t)``, in the group it was read in, ``t.group``.  A
        ``Refusal`` in its place is falsy and names what stopped the call.

    Refusals
    --------
    A refusal handed in as ``realized`` is returned unchanged.  A point that refuses
    is collected in ``refusals``.

    ``tables_not_ready``
        the group's tables cannot be read in this cell.

    Raises
    ------
    TypeError
        ``realized`` is not what :func:`..groups.realized_group` returned, or
        ``bands`` left out.
    ValueError
        States of another model than ``r.model``, ``bands`` outside the
        states or naming none, or ``r.model`` edited since
        :func:`..groups.realized_group` measured it.

    Examples
    --------
    lin3g's six lowest bands (:func:`..models.lin3g`):

    >>> print(hsp_table(r, bands=6).summary())                 # doctest: +SKIP
    bands 0-5   (.bands)
      their coreps at each high-symmetry point   (.symmetry_data)
        GM: 2x-GM6 + -GM7
        M: -M6 + 2x-M7
        X: 3x-X5
    """
    rep = _rep_of("hsp_table", realized, "hsp_table(r, bands=6)")
    return _hsp_table_rep(rep, states=states, bands=bands, degeneracy_tol=degeneracy_tol,
                          label_tol=label_tol, _ktol=_ktol, _atol=_atol,
                          _mesh_tol=_mesh_tol)
