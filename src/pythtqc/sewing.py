r"""The coreps a band set carries at the high-symmetry points, read off its eigenvectors.

At a k-point an operation ``g`` leaves invariant, the sewing matrix
``B(g,k)_mn = <u_m(k)|U(g;k)|u_n(k)>`` is the matrix of ``g`` on the bands
selected; its trace over a degenerate multiplet is that multiplet's
character, and the characters name the corep.

* :func:`hsp_table`: the coreps of a set of bands at every high-symmetry point,
  its symmetry vector, which :func:`.ebr.decompose`,
  :func:`.ebr.symmetry_indicator` and :func:`.srsi.srsi_of_bands` take;
* :func:`corep_content`: the coreps at one point, multiplet by multiplet;
* :func:`characters`: the table of characters they are read from;
* :func:`sewing_matrix`: one operation's sewing matrix;
* :func:`solve_at`: one diagonalization at a k-point, to read several sets of
  bands there; :func:`multiplets` groups its energies into degenerate levels,
  and :func:`multiplet_separation` and :func:`multiplet_spread` say how well
  they are resolved;
* :func:`format_counts`: a content on one line, ``2×-GM6 + -GM7``;
* :func:`frontier_subspace`: the eigenvalues of smallest modulus of a
  spectrum, as one set of states;
* :class:`Thresholds`: the zero thresholds at a point, one for numbers built
  from ``H(k)`` and one, widening where multiplets crowd, for numbers read off
  eigenvectors.

Each call takes the eigenvectors as ``states=``, in pythtb's own forms:
``None``, to solve the model at the point; a solved :class:`pythtb.WFArray`;
or the ``(evals, evecs)`` pair of one diagonalization.  The little groups, the
corep tables and the labels are ``irrep``'s.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Mapping

import numpy as np

from . import _bloch, _sitechars
from . import groups as _groups
from ._tolerance import (_LATTICE_TOL, CHARACTER, DEGENERACY_TOL, FLAT_ZERO, LABEL_TOL,
                         MESH_TOL, STATES_AT_K_GATE)
from .refusal import InternalError, Refusal

__all__ = [
    "FLAT_ZERO", "DEGENERACY_TOL",
    "Thresholds", "two_tier_thresholds", "disagreements",
    "Frontier", "frontier_subspace",
    "multiplets", "multiplet_separation", "multiplet_spread",
    "solve_at", "format_counts",
    "SewingMatrix", "sewing_matrix",
    "CharacterTable", "characters",
    "CorepContent", "corep_content",
    "HSPTable", "hsp_table",
]


# ==========================================================================
# the words a summary prints
# ==========================================================================

def _runs(indices) -> str:
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


def _listed(items) -> str:
    """Names as a sentence lists them: ``A``, ``A and B``, ``A, B and C``."""
    items = [str(item) for item in items]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


#: The largest denominator a reduced coordinate is written as a fraction with:
#: a tabulated point's coordinates are halves, thirds, quarters, sixths and
#: eighths.
_DENOMINATOR = 24


def _k_text(k) -> str:
    """A k-point in reduced coordinates, ``(1/3, 2/3)``, a decimal where it is no fraction."""
    out = []
    for x in np.asarray(k, dtype=float).reshape(-1) + 0.0:
        value = Fraction(float(x)).limit_denominator(_DENOMINATOR)
        out.append(f"{float(x):.4g}" if abs(float(value) - x) > _LATTICE_TOL
                   else str(value.numerator) if value.denominator == 1
                   else f"{value.numerator}/{value.denominator}")
    return "(" + ", ".join(out) + ")"


def _character_text(value) -> str:
    """A character as a table prints it: ``2``, ``-1.414``, ``0.5+0.866i``, ``-i``."""
    z = complex(value)

    def part(x):
        return str(int(round(x))) if abs(x - round(x)) <= CHARACTER else f"{x:.4g}"

    if abs(z.imag) <= CHARACTER:
        return part(z.real + 0.0)
    imaginary = {"1": "i", "-1": "-i"}.get(part(z.imag), part(z.imag) + "i")
    if abs(z.real) <= CHARACTER:
        return imaginary
    return part(z.real) + ("" if imaginary.startswith("-") else "+") + imaginary


def _operation_name(rep, ind) -> str:
    """One operation's name in the model's frame, as the operation table writes it."""
    op = rep._operation(int(ind))
    return (_sitechars.operation_name(op, np.asarray(rep.group.lattice, dtype=float))
            + ("'" if op.time_reversal else ""))


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
# the gauge-invariant frontier eigenspace
# ==========================================================================

@dataclass(frozen=True)
class Frontier:
    """The eigenvalues of smallest modulus of a spectrum, collected as one set of states.

    Every eigenvalue within ``modulus_tol`` of the smallest modulus is in it, so
    a quantity read near a degeneracy is read on the whole set, whatever basis
    the diagonalization chose inside it.  ``summary()`` prints it.

    Attributes
    ----------
    indices : tuple of int
        The set's positions in the spectrum handed in.
    value : float
        The smallest modulus, ``min |λ|``.
    paired : bool
        Whether the set holds eigenvalues of both signs, a pair ``±λ``.  Then
        the matrix elements within the set carry the first-order physics, and
        at a high-symmetry point a Kramers-type selection rule makes them zero;
        otherwise those between the set and the rest carry it.
    positive, negative : tuple of int
        The set's positions above zero and below it.
    at_zero : bool
        Whether the set sits on zero, ``value <= modulus_tol``: its signs are
        then not resolved, and a quantity read on the states of one sign is not
        to be trusted there.
    modulus_tol : float
        The window on ``|λ|`` the set was collected at, in the spectrum's own
        units.
    spectrum : ndarray
        The spectrum handed in.
    """

    indices: tuple
    value: float
    paired: bool
    positive: tuple
    negative: tuple
    at_zero: bool
    modulus_tol: float
    spectrum: np.ndarray

    def __len__(self) -> int:
        return len(self.indices)

    def summary(self) -> str:
        """The set of states, a line for each part, with the attribute that holds it."""
        lines = [f"the eigenvalues λ of smallest modulus, |λ| = {self.value:.6g} to within "
                 f"{self.modulus_tol:.2g}: positions {_runs(self.indices)} of the spectrum   "
                 "(.value, .indices)"]
        if self.at_zero:
            lines.append("  on zero, so their signs are not resolved, and a quantity read on the "
                         "states of one sign is not to be trusted here   (.at_zero)")
        elif self.paired:
            lines.append(f"  of both signs, a pair ±λ: positions {_runs(self.negative)} below "
                         f"zero and {_runs(self.positive)} above   "
                         "(.paired, .negative, .positive)")
        else:
            lines.append(f"  all {'negative' if self.negative else 'positive'}, so no pair "
                         "±λ   (.paired)")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return (f"Frontier(|lambda|={self.value:.6g}, dim={len(self.indices)}, "
                f"paired={self.paired}{', at zero' if self.at_zero else ''})")


def frontier_subspace(spectrum, modulus_tol=1e-9) -> Frontier:
    r"""The eigenvalues of smallest modulus of a spectrum, as one set of states: every state within ``modulus_tol`` of the smallest modulus.

    Parameters
    ----------
    spectrum : array_like, shape (n,)
        Real eigenvalues, measured from the level they are read against:
        energies less the chemical potential, the eigenvalues of a projected
        operator ``P O P``, or a sewing matrix's ``-iS`` eigenvalues.
    modulus_tol : float, optional
        The window on ``|λ|``, in the spectrum's own units: an eigenvalue
        within it of the smallest modulus joins the set, and a smallest
        modulus within it of zero puts the set on zero.  Default ``1e-9``.

    Returns
    -------
    Frontier
        ``f.summary()`` prints it.  ``f.indices`` are the set's positions,
        ``f.value`` its modulus, and ``f.paired`` whether it holds both signs, a
        necessary condition for an antiunitary symmetry pairing them.
        ``f.at_zero`` says the set sits on zero: its signs are not resolved,
        and a quantity read there on the states of one sign is a source of
        spurious quantized values.

    Raises
    ------
    ValueError
        An empty spectrum.

    Examples
    --------
    A spectrum of pairs ``±λ`` reads its eigenvalues of smallest modulus as
    one pair::

        frontier_subspace([-1.0, -0.9825, 0.9825, 1.0]).paired      # True
        frontier_subspace([-1.0, -0.9825, 0.9825, 1.0]).value       # 0.9825
    """
    spec = np.asarray(spectrum, dtype=float).ravel()
    if spec.size == 0:
        raise ValueError("frontier_subspace needs a non-empty spectrum")
    mag = np.abs(spec)
    amin = float(mag.min())
    idx = tuple(int(i) for i in np.flatnonzero(mag - amin <= float(modulus_tol)))
    pos = tuple(i for i in idx if spec[i] > 0.0)
    neg = tuple(i for i in idx if spec[i] < 0.0)
    at_zero = amin <= float(modulus_tol)
    return Frontier(indices=idx, value=amin,
                    paired=bool(pos and neg and not at_zero),
                    positive=pos, negative=neg, at_zero=bool(at_zero),
                    modulus_tol=float(modulus_tol), spectrum=spec)


# ==========================================================================
# the multiplet analysis — small functions over pythtb's energy arrays
# ==========================================================================

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
    :func:`corep_content` as ``states=`` to read several sets of bands at that
    k off the same states.  A second diagonalization chooses another basis
    inside each degenerate level, and a character read across the two is wrong.

    Parameters
    ----------
    model : pythtb.TBModel
        The model, ``r.model``.
    k : array_like, shape (dim_k,)
        One k-point, in reduced coordinates.
    params : dict, optional
        Values for a model whose terms are given by parameter names.

    Returns
    -------
    (ndarray, ndarray)
        ``(evals, evecs)``: the energies, ascending, and the eigenvectors as
        rows, as ``model.solve_ham`` returns them with ``return_eigvecs=True``
        and ``flatten_spin_axis=True``.
    """
    return _bloch.solve(model, np.atleast_2d(np.asarray(k, dtype=float).reshape(-1)), params)


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


def _resolve_states(source, rep, k, *, mesh_tol=MESH_TOL, params=None):
    """The **single** eigenvector source at ``k`` -> ``(energies_or_None, V)``.

    ``source`` is a solved :class:`pythtb.WFArray`, the ``(evals, evecs)`` pair
    :meth:`pythtb.TBModel.solve_ham` returns, a bare fiber of eigenvector rows,
    or ``None``/``rep.model`` (solve here, storing nothing).  There is exactly
    one slot, which is what makes a second diagonalization unreachable by
    accident.
    """
    k = np.asarray(k, dtype=float).reshape(-1)
    params = dict(params or {})

    if source is None or source is rep.model:
        evals, evecs = solve_at(rep.model, k, params=params)
        return np.asarray(evals, dtype=float).reshape(-1), _as_columns(evecs)

    if hasattr(source, "wfs") and hasattr(source, "mesh"):
        model = getattr(source, "_model", None)
        if model is not None and model is not rep.model:
            raise ValueError(
                "the WFArray was solved on a different TBModel than rep.model: the "
                "representation's covariance was measured on rep.model and its matrices "
                "mean nothing on another one")
        return _wfarray_fiber(source, k, mesh_tol=mesh_tol)

    if isinstance(source, (tuple, list)) and len(source) == 2:
        evals, evecs = source
        return np.asarray(evals, dtype=float).reshape(-1), _as_columns(evecs)

    if hasattr(source, "solve_ham"):
        raise ValueError(
            "the states source must be rep.model (or None, meaning it): the "
            "representation's covariance was measured on rep.model and on nothing else")

    return None, _as_columns(source)


def _hamiltonian_at(rep, k, params=None) -> np.ndarray:
    """``H(k)`` from ``rep.model``, Hermitized — recomputed on demand, never stored."""
    k = np.asarray(k, dtype=float).reshape(-1)
    H = np.asarray(_bloch.hamiltonian(rep.model, np.atleast_2d(k), params))[0]
    return 0.5 * (H + H.conj().T)


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


def _band_block(nstate, bands):
    """The sorted band indices a band count or a band sequence names.

    An integer ``n`` means the lowest ``n`` bands; a sequence means exactly
    those band indices.  ``None`` means every band of the fiber.
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
    if len(set(idx)) != len(idx):
        raise ValueError(f"the bands {idx} name a band more than once")
    if idx and not (0 <= min(idx) and max(idx) < nstate):
        raise ValueError(f"the bands {idx} are not all among the {nstate} bands "
                         f"0..{nstate - 1}")
    return tuple(sorted(idx))


# ==========================================================================
# the sewing matrix
# ==========================================================================

@dataclass(frozen=True)
class SewingMatrix:
    """The matrix of one operation on a set of bands, at a k-point the operation leaves invariant: ``B(g,k) = V† U(g;k) V``.

    Its trace over a degenerate multiplet is that multiplet's character.  The
    set of bands has to be whole degenerate multiplets for ``B`` to be the
    operation's own: on part of one, its trace follows the basis the
    diagonalization chose.  ``summary()`` prints it.

    Attributes
    ----------
    operation : int
        The operation, by its index.
    k : ndarray
        The k-point, in reduced coordinates.
    wrap : ndarray of int
        The reciprocal lattice vector ``G = Rk − k``.  On the zone boundary it
        brings the factor ``e^{2πi G·τ}`` of an orbital at ``τ`` off the
        origin, which reads that orbital's characters right there.
    bands : tuple of int
        The set of bands, by position among the states handed in: for a
        ``WFArray`` of a sector, that sector's states.
    matrix : ndarray, shape (nb, nb)
        ``B``.
    operator : ndarray, shape (nstate, nstate)
        ``U(g;k)``, the operator on the model's states.
    trace : complex
        ``tr B``: the character, where the set of bands is one multiplet.
    eigenvalues : ndarray
        ``B``'s eigenvalues, on the unit circle where ``B`` is unitary;
        :attr:`angles` holds their phases.
    unitarity : float
        ``max |B†B − 1|``: zero where the set of bands is closed under the
        operation.
    commutator : float
        ``max |U H(k) − H(k) U|``: zero where ``U`` is a symmetry of ``H(k)``.
    antiunitary : bool
        Whether the operation is combined with time reversal; a corep table
        holds no character for one.
    residual : float
        ``max |H(k) V − V diag(w)|``, how far the states are from eigenstates
        at ``k``.  ``nan`` where they came without energies, and then neither
        this nor whether ``bands`` closes its multiplets was measured.
    """

    operation: int
    k: np.ndarray
    wrap: np.ndarray
    bands: tuple
    matrix: np.ndarray
    operator: np.ndarray
    unitarity: float
    commutator: float
    antiunitary: bool
    eigenvalues: np.ndarray
    trace: complex
    residual: float = float("nan")
    # the operation's name in the model's frame, for the summary
    _name: str = field(default="", repr=False, compare=False)

    @property
    def angles(self) -> np.ndarray:
        """The phases of ``B``'s eigenvalues, in radians, ascending."""
        return np.sort(np.angle(self.eigenvalues))

    def summary(self) -> str:
        """The sewing matrix, a line for each part, with the attribute that holds it."""
        named = (f"{self._name}, operation {self.operation}" if self._name
                 else f"operation {self.operation}")
        lines = [f"{named}, on band{'s' if len(self.bands) > 1 else ''} {_runs(self.bands)} "
                 f"at k = {_k_text(self.k)}   (.operation, .bands, .k)"]
        if np.any(self.wrap):
            lines.append(f"  carries k to k + {_k_text(self.wrap)}, a reciprocal lattice vector "
                         "away   (.wrap)")
        if self.antiunitary:
            lines.append("  combined with time reversal: its trace is no character, a corep table "
                         "holding none for it   (.antiunitary)")
        lines.append(f"  its trace, the character of these bands: "
                     f"{_character_text(self.trace)}   (.trace)")
        if len(self.eigenvalues):
            lines.append("  its eigenvalues e^{iθ}, at θ/π = "
                         + ", ".join(f"{a / np.pi:.4g}" for a in self.angles + 0.0)
                         + "   (.angles)")
        measured = [f"unitary to {self.unitarity:.2g}"]
        if np.isfinite(self.commutator):
            measured.append(f"the operator commuting with H(k) to {self.commutator:.2g}")
        if np.isfinite(self.residual):
            measured.append(f"the states solving H(k) to {self.residual:.2g}")
            lines.append("  " + ", ".join(measured) + "   (.unitarity, .commutator, .residual)")
        else:
            lines.append("  " + ", ".join(measured) + "; the states came without energies, so "
                         "whether they solve H(k) and close their multiplets was not "
                         "measured   (.unitarity, .commutator, .residual)")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return (f"SewingMatrix(#{self.operation}, {len(self.bands)} bands, "
                f"tr={self.trace:.6g}, unitarity={self.unitarity:.2e})")


def _sewing_core(rep, operation, k, V, idx, H, *, atol=1e-8, residual=float("nan"),
                 **kwargs) -> SewingMatrix | Refusal:
    """``B = V†UV`` on ``idx`` with its two self-checks — the shared inner loop.

    Private on purpose: it is the piece :func:`characters` reuses across every
    column of one table, so that a table is built from **one** fiber and one
    ``H(k)``.  It stores nothing; ``V``, ``H`` and ``idx`` are local arrays
    handed in.
    """
    U = rep.sewing_operator(int(operation), k, atol=atol, **kwargs)
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
    return SewingMatrix(operation=int(operation), k=np.asarray(k, dtype=float).reshape(-1),
                        wrap=rep.in_little_group(int(operation), k, atol=atol)[1],
                        bands=tuple(idx), matrix=B, operator=U, unitarity=unit,
                        commutator=comm, antiunitary=anti, eigenvalues=ev,
                        trace=complex(np.trace(B)), residual=float(residual),
                        _name=_operation_name(rep, operation))


def sewing_matrix(rep, operation=None, k=None, *, states=None, bands=None, atol=1e-8,
                  degeneracy_tol=DEGENERACY_TOL, mesh_tol=MESH_TOL, check=True,
                  params=None, **kwargs) -> SewingMatrix | Refusal:
    r"""The matrix of one operation on a set of bands, at a k-point the operation leaves invariant: the sewing matrix ``B(g,k)_mn = <u_m(k)|U(g;k)|u_n(k)>``.

    Its trace over a degenerate multiplet is that multiplet's character, and its
    eigenvalues are the operation's on those bands.

    Parameters
    ----------
    rep : ..symmetry.SpaceGroupRep
        The model's representation, ``r.rep`` from :func:`..groups.realized_group`.
    operation : int
        The operation, by its index: ``r.operations`` lists them, and
        :func:`..groups.operation_index` finds one by its Seitz symbol.
        Required.
    k : array_like, shape (dim_k,)
        A k-point the operation leaves invariant up to a reciprocal lattice
        vector, in reduced coordinates.  Required.
    bands : int or sequence of int, optional
        The set of bands: ``n`` the lowest ``n`` bands, or the band indices,
        whole degenerate multiplets.  Default: every state.
    states : None, pythtb.WFArray, (evals, evecs) or ndarray, optional
        The states at ``k``: ``None`` (default) solves the model there; a
        solved ``WFArray``, or the pair :func:`solve_at` returns, reuses one
        diagonalization.  Eigenvector rows alone, or a ``WFArray`` of a sector
        with fewer states than the model, carry no energies: whether they solve
        ``H(k)`` and close their multiplets is then not measured.
    degeneracy_tol : float, optional
        The energy, in the units of ``H``, within which bands are one
        multiplet.  Default ``1e-8``.
    atol, mesh_tol : float, optional
        In reduced reciprocal coordinates: whether the operation leaves ``k``
        invariant (default ``1e-8``), and finding ``k`` on a mesh (``1e-9``).
    check : bool, optional
        Check that the states solve ``H(k)`` and that ``bands`` closes its
        multiplets.  Default ``True``.
    params : dict, optional
        Values for a model whose terms are given by parameter names.
    **kwargs
        Passed to :meth:`..symmetry.SpaceGroupRep.sewing_operator`.

    Returns
    -------
    SewingMatrix or Refusal
        ``B.summary()`` prints it.  ``B.matrix`` is the matrix, ``B.trace`` its
        trace and ``B.angles`` the phases of its eigenvalues; ``B.unitarity``,
        ``B.commutator`` and ``B.residual`` how far it is from unitary, the
        operator from commuting with ``H(k)``, and the states from eigenstates.
        A ``Refusal`` in its place is falsy and names what stopped the call.

    Raises
    ------
    TypeError
        ``operation`` or ``k`` left out.
    ValueError
        States of another model than ``rep.model``, or ``bands`` outside the
        states.

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

    Examples
    --------
    The fourfold rotation on lin3g's lowest pair at ``M``
    (:func:`..models.lin3g`):

    >>> c4 = operation_index(r, "{4+_001|0,0,0}")                   # doctest: +SKIP
    >>> B = sewing_matrix(r.rep, c4, hsp["M"], bands=2)              # doctest: +SKIP
    >>> np.round(B.trace, 6), B.angles / np.pi                      # doctest: +SKIP
    ((-1.414214+0j), array([-0.75,  0.75]))
    """
    if operation is None:
        raise TypeError("sewing_matrix() missing required argument 'operation'")
    if k is None:
        raise TypeError("sewing_matrix() missing required argument 'k'")
    k = np.asarray(k, dtype=float).reshape(-1)
    got = _resolve_states(states, rep, k, mesh_tol=mesh_tol, params=params)
    if isinstance(got, Refusal):
        return got
    w, V = got
    idx = _band_block(V.shape[1], bands)

    H = _hamiltonian_at(rep, k, params)
    resid = float("nan") if w is None else _states_at_k(H, w, V)

    if check and w is not None:
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

    return _sewing_core(rep, operation, k, V, idx, H, atol=atol, residual=resid, **kwargs)


# ==========================================================================
# the character table
# ==========================================================================

@dataclass(frozen=True)
class CharacterTable:
    """The characters of a set of bands at one k: the trace of each unitary operation of the little group over each degenerate multiplet.

    Every corep label is read from such a table.  ``summary()`` prints it.

    Attributes
    ----------
    k : ndarray
        The k-point, in the model's reduced coordinates.
    bands : tuple of int
        The set of bands.
    multiplets : tuple of tuple of int
        The rows: the degenerate multiplets of the set, lowest first, by their
        bands.
    dims : tuple of int
        Each row's degeneracy.
    energies : ndarray
        Each row's mean energy.
    columns : tuple of int
        The columns: the unitary operations of the little group, by index,
        ascending.
    characters : ndarray, shape (rows, columns), complex
        The trace of each operation over each multiplet.
    antiunitary : tuple of int
        The operations of the little group combined with time reversal; a corep
        table holds no characters for them, so none is a column.
    unitarity, commutators : dict
        Per operation, by index: how far its sewing matrix on the whole set of
        bands is from unitary, and how far its operator is from commuting with
        ``H(k)``.
    residual : float
        How far the states are from eigenstates of ``H(k)``.
    spread : float
        The largest energy spread inside a multiplet; a safe grouping has it
        far below the gap between multiplets.
    thresholds : Thresholds
        The zero thresholds at this k.
    disagreements : tuple
        The numbers the two thresholds classify differently.
    spectrum : ndarray
        Every energy of the diagonalization the table was read from.
    degeneracy_tol : float
        The energy tolerance the rows were grouped at.
    group_provenance : str
        ``"realized"`` for a group measured on ``H``, ``"descended"`` for one
        measured after the cell was lowered.
    """

    k: np.ndarray
    bands: tuple
    multiplets: tuple
    dims: tuple
    energies: np.ndarray
    columns: tuple
    characters: np.ndarray
    antiunitary: tuple
    unitarity: dict
    commutators: dict
    thresholds: Thresholds
    spectrum: np.ndarray
    degeneracy_tol: float
    spread: float
    residual: float
    disagreements: tuple = ()
    group_provenance: str = "cell"
    # each column's operation by its name in the model's frame, for the summary
    _names: dict = field(default_factory=dict, repr=False, compare=False)

    def summary(self) -> str:
        """The characters, a line for each part, with the attribute that holds it."""
        lines = [f"band{'s' if len(self.bands) > 1 else ''} {_runs(self.bands)} at "
                 f"k = {_k_text(self.k)}   (.bands, .k)"]
        if self.columns:
            lines.append("  the trace of each unitary operation of the little group, by index, "
                         "over each degenerate multiplet   (.characters, .columns)")
            rows = [("bands", [_runs(m) for m in self.multiplets], "   (.multiplets)"),
                    ("energy", [f"{e:.4g}" for e in self.energies + 0.0], "   (.energies)")]
            rows += [(f"{int(ind):>3}  {self._names.get(int(ind), '')}",
                      [_character_text(z) for z in self.characters[:, c]], "")
                     for c, ind in enumerate(self.columns)]
            head = max(len(label) for label, _cells, _end in rows)
            width = max(len(cell) for _label, cells, _end in rows for cell in cells) + 2
            lines += ["    " + label.ljust(head) + "".join(cell.rjust(width) for cell in cells)
                      + end for label, cells, end in rows]
        if self.antiunitary:
            lines.append("  the operations combined with time reversal are left out: a corep "
                         "table holds no characters for them   (.antiunitary)")
        unitary = max(self.unitarity.values(), default=0.0)
        commuting = max(self.commutators.values(), default=0.0)
        lines.append(f"  the sewing matrices unitary to {unitary:.2g}, the operators commuting "
                     f"with H(k) to {commuting:.2g}, the states solving H(k) to "
                     f"{self.residual:.2g}   (.unitarity, .commutators, .residual)")
        if self.disagreements:
            named = _listed(str(d["name"]).split("#")[-1].rstrip("]")
                            for d in self.disagreements)
            lines.append(f"  operation{'s' if len(self.disagreements) > 1 else ''} {named}: the "
                         "sewing matrix's departure from unitarity is zero against the "
                         "eigenvectors' threshold here and not against the flat one   "
                         "(.disagreements, .thresholds)")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        n = len(self.multiplets)
        return (f"CharacterTable(k={np.round(self.k, 6).tolist()}, bands {_runs(self.bands)} "
                f"in {n} multiplet{'' if n == 1 else 's'}, group={self.group_provenance})")


def characters(rep, k, *, states=None, bands=None, degeneracy_tol=DEGENERACY_TOL,
               atol=1e-8, mesh_tol=MESH_TOL, params=None) -> CharacterTable | Refusal:
    r"""The characters of a set of bands at one k: the trace of each unitary operation of the little group over each degenerate multiplet of the set.

    Parameters
    ----------
    rep : ..symmetry.SpaceGroupRep
        The model's representation, ``r.rep`` from :func:`..groups.realized_group`.
    k : array_like, shape (dim_k,)
        A k-point, in reduced coordinates; only a tabulated high-symmetry point
        is labelled (:func:`corep_content`).
    bands : int or sequence of int
        The set of bands: ``n`` the lowest ``n`` bands, or the band indices,
        whole degenerate multiplets.  Required.
    states : None, pythtb.WFArray or (evals, evecs), optional
        The states at ``k``: ``None`` (default) solves the model there; a
        solved ``WFArray``, or the pair :func:`solve_at` returns, reuses one
        diagonalization.  They carry energies, the rows being the multiplets.
    degeneracy_tol : float, optional
        The energy, in the units of ``H``, within which bands are one
        multiplet.  Default ``1e-8``.
    atol, mesh_tol : float, optional
        In reduced reciprocal coordinates: whether an operation leaves ``k``
        invariant (default ``1e-8``), and finding ``k`` on a mesh (``1e-9``).
    params : dict, optional
        Values for a model whose terms are given by parameter names.

    Returns
    -------
    CharacterTable or Refusal
        ``table.summary()`` prints it.  ``table.characters`` holds a row per
        multiplet and a column per operation; ``table.multiplets`` and
        ``table.energies`` are the rows, ``table.columns`` the operations by
        index.  A ``Refusal`` in its place is falsy and names what stopped the
        call.

    Raises
    ------
    ValueError
        States without energies, or of another model than ``rep.model``.
    ..refusal.InternalError
        The representation leaves out an operation of the little group at
        ``k``: the package disagreeing with itself.

    Refusals
    --------
    A refusal handed in as ``rep`` is returned unchanged, and one from
    :func:`..groups.little_group` as it is.

    ``multiplet_split``
        the set of bands cuts a degenerate multiplet at ``k``.
    ``states_not_at_k``
        the states handed in do not solve ``H(k)``.
    ``k_not_on_mesh``
        the mesh handed in does not hold ``k``.
    """
    if bands is None:
        raise TypeError("characters() missing required argument 'bands'")
    if isinstance(rep, Refusal):
        return rep
    gate = _groups._lattice_not_descended(rep.group)
    if gate is not None:
        return gate
    k = np.asarray(k, dtype=float).reshape(-1)
    got = _resolve_states(states, rep, k, mesh_tol=mesh_tol, params=params)
    if isinstance(got, Refusal):
        return got
    w, V = got
    if w is None:
        raise ValueError(
            "characters needs energies: its rows are the degenerate multiplets at k, so a "
            "source that carries states but no spectrum (a feature-sector WFArray, a bare "
            "fiber) cannot define the table. Use sewing_matrix() on such a source, or hand "
            "in the model / the (evals, evecs) pair.")
    idx = _band_block(V.shape[1], bands)

    H = _hamiltonian_at(rep, k, params)
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

    lg = _groups.little_group(rep.group, k)
    if isinstance(lg, Refusal):
        return lg
    ours = [int(i) for i in rep.little_group(k, atol=atol)]
    ours_u = tuple(sorted(i for i in ours if not rep._operation(i).time_reversal))
    ours_a = tuple(sorted(i for i in ours if rep._operation(i).time_reversal))
    not_carried = tuple(sorted(set(lg.indices) - set(ours_u)))
    if not_carried:
        # every representation a call returns carries the whole group it was
        # built for, so an operation of that group left out is the package
        # disagreeing with itself
        raise InternalError(
            f"the representation of #{rep.group.number_str} leaves out part of its own "
            f"little group at k = {np.round(k, 6).tolist()}; not carried: "
            f"{list(not_carried)}. Every representation a call returns carries the "
            f"whole group it was built for.")

    chars = np.zeros((len(rows), len(ours_u)), dtype=complex)
    unit, comm, names = {}, {}, {}
    for c, ind in enumerate(ours_u):
        whole = _sewing_core(rep, ind, k, V, idx, H, atol=atol, residual=resid)
        if isinstance(whole, Refusal):  # pragma: no cover - membership tested above
            return whole
        unit[ind] = whole.unitarity
        comm[ind] = whole.commutator
        names[ind] = whole._name
        for r, m in enumerate(rows):
            local = [idx.index(b) for b in m]
            chars[r, c] = np.trace(whole.matrix[np.ix_(local, local)])

    thr = two_tier_thresholds(H, multiplet_separation(w, degeneracy_tol),
                              degeneracy_tol=float(degeneracy_tol))
    dis = disagreements(thr, {f"unitarity[#{i}]": v for i, v in unit.items()})
    return CharacterTable(
        k=k, bands=idx, multiplets=tuple(rows), dims=tuple(len(m) for m in rows),
        energies=np.array([float(w[list(m)].mean()) for m in rows]),
        columns=ours_u, characters=chars, antiunitary=ours_a,
        unitarity=unit, commutators=comm, thresholds=thr, spectrum=w,
        degeneracy_tol=float(degeneracy_tol),
        spread=multiplet_spread(w, degeneracy_tol), residual=resid,
        disagreements=dis, group_provenance=str(rep.group.provenance), _names=names)


# ==========================================================================
# corep content
# ==========================================================================

def format_counts(counts) -> str:
    r"""A content on one line: ``{'-GM6': 2, '-GM7': 1}`` as ``2×-GM6 + -GM7``.

    Every content the package returns reads this way: the coreps of a set of
    bands at one point (:attr:`CorepContent.counts`) or per multiplet
    (:attr:`CorepContent.labels`), a symmetry vector (:attr:`HSPTable.counts`),
    an elementary band representation's coreps (:func:`..ebr.ebr_rows`).  The
    labels are sorted, so one content always prints one line.

    Parameters
    ----------
    counts : mapping
        ``{corep: multiplicity}``, the multiplicities nonnegative integers.

    Returns
    -------
    str
        The labels joined by ``" + "``, each after its multiplicity and ``×``
        where that is above one.
    """
    return " + ".join(f"{n}×{lab}" if n > 1 else lab
                      for lab, n in sorted(counts.items()))


@dataclass(frozen=True)
class CorepContent:
    """The coreps a set of bands carries at one high-symmetry point, multiplet by multiplet.

    ``summary()`` prints them.

    Attributes
    ----------
    kpname : str
        The tables' name of the point; ``GM`` is Γ.
    k : ndarray
        Its coordinates, in the model's reduced cell.
    counts : dict
        ``{corep: multiplicity}`` over the set of bands: its symmetry vector at
        this point.
    labels : tuple of dict
        The same for each degenerate multiplet, lowest first.
    partition : tuple of int
        Which multiplets carry the same characters: two share a number exactly
        when their characters agree within ``label_tol``.  It holds under every
        labelling of the orbitals and every phase convention, which the labels
        need not; quote it with them.
    isolation : float
        The energy gap from the set of bands to the nearest band outside it, in
        the units of ``H``; ``inf`` where the set is every band.
    table : CharacterTable
        The characters the labels were read from.
    content : ..groups.IrrepContent
        The identification itself: the multiplicities before rounding, and how
        far they sat from integers.
    thresholds : Thresholds
        The zero thresholds at this point.
    """

    kpname: str
    k: np.ndarray
    labels: tuple
    counts: dict
    partition: tuple
    content: Any
    table: CharacterTable
    thresholds: Thresholds
    isolation: float

    def summary(self) -> str:
        """The coreps at this point, a line for each part, with the attribute that holds it."""
        bands = self.table.bands
        point = f"{self.kpname} (Γ)" if self.kpname == "GM" else self.kpname
        lines = [f"band{'s' if len(bands) > 1 else ''} {_runs(bands)} at {point}: "
                 f"{format_counts(self.counts)}   (.table.bands, .kpname, .counts)"]
        if len(self.labels) > 1:
            lines.append("  by degenerate multiplet, from the lowest   (.labels)")
            lines += [f"    band{'s' if len(bands) > 1 else ''} {_runs(bands)} at {energy:.4g}: "
                      f"{format_counts(labels)}, class {cls}"
                      for bands, energy, labels, cls in zip(
                          self.table.multiplets, self.table.energies + 0.0, self.labels,
                          self.partition)]
            lines.append("  multiplets of one class carry the same characters, under any "
                         "labelling of the orbitals   (.partition)")
        if np.isfinite(self.isolation):
            these = "these bands" if len(bands) > 1 else "this band"
            lines.append(f"  {self.isolation:.4g} from {these} to the nearest other band, in the "
                         "units of H   (.isolation)")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return (f"CorepContent({self.kpname}: {format_counts(self.counts)}, "
                f"partition={self.partition})")


def _isolation(spectrum, bands):
    """Energy gap from a band block to the rest of the spectrum."""
    w = np.asarray(spectrum, dtype=float).reshape(-1)
    inside = sorted(bands)
    outside = [b for b in range(len(w)) if b not in set(inside)]
    if not outside or not inside:
        return float("inf")
    return float(min(abs(w[a] - w[b]) for a in inside for b in outside))


def corep_content(rep, k, *, states=None, bands=None, degeneracy_tol=DEGENERACY_TOL,
                  label_tol=LABEL_TOL, ktol=1e-5, kpname=None, atol=1e-8,
                  mesh_tol=MESH_TOL, params=None) -> CorepContent | Refusal:
    r"""The coreps a set of bands carries at one high-symmetry point, and which of its degenerate multiplets carry the same characters.

    Parameters
    ----------
    rep : ..symmetry.SpaceGroupRep
        The model's representation, ``r.rep`` from :func:`..groups.realized_group`.
    k : array_like
        A tabulated high-symmetry point, in the model's reduced coordinates, as
        :func:`..groups.high_symmetry_points` gives it: ``hsp["GM"]``.
    bands : int or sequence of int
        The set of bands: ``n`` the lowest ``n`` bands, or the band indices,
        whole degenerate multiplets.  Required.
    states : None, pythtb.WFArray or (evals, evecs), optional
        The states at ``k``: ``None`` (default) solves the model there; a
        solved ``WFArray``, or the pair :func:`solve_at` returns, reuses one
        diagonalization.
    degeneracy_tol : float, optional
        The energy, in the units of ``H``, within which bands are one
        multiplet.  Default ``1e-8``.
    label_tol : float, optional
        How far a fitted multiplicity may sit from an integer, and two
        multiplets' characters from each other to share a partition class.
        Default ``1e-3``.
    kpname : str, optional
        The tables' name of the point, in place of finding it from ``k``.
    ktol, atol, mesh_tol : float, optional
        In reduced reciprocal coordinates: matching ``k`` to a tabulated point
        (default ``1e-5``), whether an operation leaves ``k`` invariant
        (``1e-8``), and finding ``k`` on a mesh (``1e-9``).
    params : dict, optional
        Values for a model whose terms are given by parameter names.

    Returns
    -------
    CorepContent or Refusal
        ``c.summary()`` prints the coreps multiplet by multiplet.  ``c.counts``
        is the content, ``{corep: multiplicity}``; ``c.labels`` the same for
        each multiplet, lowest first; ``c.partition`` which multiplets carry the
        same characters, numbered alike, which holds under any labelling of the
        orbitals; ``c.isolation`` the gap from these bands to the nearest other
        band; ``c.table`` the characters.  A ``Refusal`` in its place is falsy
        and names what stopped the call.

    Raises
    ------
    ValueError
        States without energies, or of another model than ``rep.model``.
    ..refusal.InternalError
        The representation leaves out an operation of the little group at
        ``k``: the package disagreeing with itself.

    Refusals
    --------
    A refusal handed in as ``rep`` is returned unchanged.

    ``multiplet_split``
        the set of bands cuts a degenerate multiplet at ``k``: label the
        multiplets whole.
    ``kpoint_not_tabulated``
        ``k`` is not a tabulated point in this cell, or not the tabulated member
        of its star.
    ``integer_multiplicity`` / ``multiplet_dimension``
        the characters fit no whole number of coreps, or coreps that do not fill
        their multiplet: most often a multiplet grouped wrongly.
    ``states_not_at_k`` / ``k_not_on_mesh``
        states handed in from another point, or a mesh that does not hold ``k``.
    ``little_group_mismatch`` / ``tables_not_ready``
        the tables do not apply in this cell.

    Examples
    --------
    lin3g's six lowest bands at Γ (:func:`..models.lin3g`):

    >>> c = corep_content(r.rep, hsp["GM"], bands=6)                # doctest: +SKIP
    >>> print(c.summary())                                          # doctest: +SKIP
    bands 0-5 at GM (Γ): 2×-GM6 + -GM7   (.table.bands, .kpname, .counts)
      by degenerate multiplet, from the lowest   (.labels)
        bands 0-1 at -16.35: -GM6, class 0
        bands 2-3 at -12.38: -GM7, class 1
        bands 4-5 at -4.325: -GM6, class 0
      multiplets of one class carry the same characters, under any labelling of the orbitals   (.partition)
      13.02 from these bands to the nearest other band, in the units of H   (.isolation)
    """
    if bands is None:
        raise TypeError("corep_content() missing required argument 'bands'")
    if isinstance(rep, Refusal):
        return rep
    gate = _groups._lattice_not_descended(rep.group)
    if gate is not None:
        return gate
    tab = characters(rep, k, states=states, bands=bands, degeneracy_tol=degeneracy_tol,
                     atol=atol, mesh_tol=mesh_tol, params=params)
    if isinstance(tab, Refusal):
        return tab

    out = _groups.identify_from_characters(
        rep.group, tab.k, tab.characters, multiplet_dims=list(tab.dims),
        kpname=kpname, label_tol=label_tol, ktol=ktol)
    if isinstance(out, Refusal):
        return out

    counts: dict = {}
    for row in out.labels:
        for lab, v in row.items():
            counts[lab] = counts.get(lab, 0) + int(v)
    return CorepContent(kpname=out.kpname, k=tab.k, labels=out.labels, counts=counts,
                        partition=out.partition, content=out, table=tab,
                        thresholds=tab.thresholds,
                        isolation=_isolation(tab.spectrum, tab.bands))


# ==========================================================================
# the high-symmetry-point table an EBR analysis consumes
# ==========================================================================

@dataclass(frozen=True)
class HSPTable:
    """The coreps a set of bands carries at every high-symmetry point: its symmetry vector, point by point.

    ``summary()`` prints it.

    Attributes
    ----------
    bands : tuple of int
        The band indices of the set, ascending.
    points : dict
        Every tabulated point in the model's periodic plane, by name, with its
        reduced coordinates.
    content : dict
        ``{point: {corep: multiplicity}}`` at each point resolved.
    partitions : dict
        ``{point: partition}``: which multiplets at each point carry the same
        characters (:attr:`CorepContent.partition`).
    counts : dict
        ``{corep: multiplicity}`` over every point resolved: the symmetry vector
        :func:`..ebr.decompose` and :func:`..ebr.symmetry_indicator` take.
    complete : bool
        Whether every point resolved; a decomposition and an indicator refuse a
        table that did not.
    records : dict
        ``{point: CorepContent}`` at each point resolved.
    refusals : dict
        ``{point: Refusal}`` at each point that refused.
    skipped : dict
        ``{point: reason}`` at each point whose little group is the identity
        alone: a band set carries its one irrep once per band, and a
        decomposition and an indicator read the band count there.
    thresholds : dict
        ``{point: Thresholds}``, the zero thresholds at each point.
    group_provenance : str
        ``"realized"`` for a group measured on ``H``, ``"descended"`` for one
        measured after the cell was lowered.
    note : str
        Where these bands' labels rest on a choice the hoppings leave open, a
        sentence naming the orbitals it is made on and what made it, and ``""``
        elsewhere.  The choice is a one-dimensional character on a coupled part
        of the crystal, which keeps every orbital's site irrep and the labels
        of all the bands and can label some of them differently; or, where the
        Hamiltonian conserves a pseudospin and nothing declares what its copies
        carry, that content itself.
    """

    bands: tuple
    points: dict
    content: dict
    partitions: dict
    records: dict
    refusals: dict
    skipped: dict
    counts: dict
    thresholds: dict
    complete: bool
    group_provenance: str = "cell"
    note: str = ""

    def summary(self) -> str:
        """The coreps of these bands, a line for each part, with the attribute that holds it."""
        lines = [f"band{'s' if len(self.bands) > 1 else ''} {_runs(self.bands)}   (.bands)"]
        resolved = [name for name in self.points if name in self.content]
        if resolved:
            lines.append("  their coreps at each high-symmetry point"
                         + (", GM being Γ" if "GM" in resolved else "") + "   (.content)")
            lines += [f"    {name}: {format_counts(self.content[name])}" for name in resolved]
        gates = {}
        for name in self.points:
            if name in self.refusals:
                gates.setdefault(str(self.refusals[name].gate), []).append(name)
        if gates:
            lines.append("  refused at " + "; ".join(f"{_listed(names)} ({gate})"
                                                     for gate, names in gates.items())
                         + ", so the content is incomplete   (.refusals)")
        if self.skipped:
            lines.append(f"  {_listed(self.skipped)}: no operation but the identity, whose one "
                         "irrep a band set carries once per band   (.skipped)")
        if self.note:
            lines.append(f"  {self.note}   (.note)")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return (f"HSPTable(bands={self.bands}, {len(self.content)} points resolved, "
                f"{len(self.refusals)} refused, {len(self.skipped)} skipped, "
                f"group={self.group_provenance})")


#: What the labelling returned rests on, by the key that set the others aside:
#: the site content at the lowest angular momentum, the functions the orbitals
#: read as, and the band labels themselves.
_DECIDED_BY = {
    "content": "the one that reads each site's content at the lowest angular momentum",
    "orbitals": ("the one whose orbitals read as the lowest-angular-momentum functions "
                 "carrying their site content"),
    "bands": "the one whose band labels come first in the tables' order",
}


def _labelling_note(rep, block) -> str:
    """The sentence :attr:`HSPTable.note` carries, or ``""``.

    Where the lowest-angular-momentum rule kept one of several labellings of a
    coupled part that give every orbital the same site irrep, and another of
    them labels some band differently, the atomic-limit table holds the labels
    every energy multiplet carries under each.  A set of bands rests on that
    choice where the multiplets it holds add up, at some point, to other labels
    under another of them; the sentence names the orbitals of the coupled part
    the choice is made on, and what made it.  Where only the group's order of
    operations set them apart, no label rests on the choice and nothing is said.
    """
    table = getattr(rep, "characters", None)
    record = getattr(table, "_labellings", {}) or {}
    choices = record.get("choices", ())
    nstate = int(rep.model.nstate)
    if not choices or len(block) >= nstate:
        return ""
    held = {int(b) for b in block}
    moved = [(int(rec["part"]), key) for rec in choices for key, other in rec["others"]
             if key in _DECIDED_BY and _labelled_apart(rec["chosen"], other, held)]
    if not moved:
        return ""
    decided = next(key for key in _DECIDED_BY if any(k == key for _p, k in moved))
    where = _parts_named(rep, sorted({p for p, _k in moved}), record.get("parts", ()))
    return (f"the labels of these {len(block)} of the {nstate} bands rest on a "
            f"one-dimensional character the hoppings leave open on {where}: another choice "
            f"of it keeps every orbital's site irrep and the labels of all {nstate} bands, "
            f"and labels these {len(block)} differently; the lowest-angular-momentum rule "
            f"chose {_DECIDED_BY[decided]}")


def _labelled_apart(chosen, other, held) -> bool:
    """Whether the multiplets the bands ``held`` fill carry other labels under ``other``.

    ``chosen`` and ``other`` are ``{point: ((bands, {corep: m}), ...)}``, and one
    point whose labels differ is enough.  A point where the bands held cut a
    multiplet says nothing and is passed over.
    """
    for point, multiplets in chosen.items():
        theirs = other.get(point)
        if theirs is None or len(theirs) != len(multiplets):
            continue
        mine_sum, their_sum, whole = {}, {}, True
        for (bands, labels), (bands_other, labels_other) in zip(multiplets, theirs):
            inside = held.intersection(bands)
            if not inside:
                continue
            if len(inside) != len(bands) or tuple(bands) != tuple(bands_other):
                whole = False
                break
            for label, m in labels.items():
                mine_sum[label] = mine_sum.get(label, 0) + int(m)
            for label, m in labels_other.items():
                their_sum[label] = their_sum.get(label, 0) + int(m)
        if whole and mine_sum != their_sum:
            return True
    return False


def _parts_named(rep, parts, states_of) -> str:
    """Coupled parts by their orbitals: ``"the coupled part holding orbitals 1 and 5"``.

    A pseudospin the Hamiltonian conserves on them is named with them.
    """
    nspin = max(int(getattr(rep.model, "nspin", 1)), 1)
    norb = int(rep.model.norb)
    orbitals = sorted({int(s) // nspin for p in parts if int(p) < len(states_of)
                       for s in states_of[int(p)]})
    which = "the coupled parts holding" if len(parts) > 1 else "the coupled part holding"
    if not orbitals or len(orbitals) == norb:
        named = f"{which} every orbital"
    elif len(orbitals) == 1:
        named = f"{which} orbital {orbitals[0]}"
    else:
        named = (f"{which} orbitals " + ", ".join(str(o) for o in orbitals[:-1])
                 + f" and {orbitals[-1]}")
    held = set(orbitals) if orbitals else set(range(norb))
    table = getattr(rep, "characters", None)
    copies = [tuple(int(o) for o in space.orbitals)
              for space in (getattr(table, "multiplicity_spaces", ()) or ())
              if {int(o) for o in space.orbitals} <= held]
    if len(copies) == 1 and set(copies[0]) == held:
        return f"{named}, on which the Hamiltonian conserves a pseudospin"
    if copies:
        return (f"{named}, the Hamiltonian conserving a pseudospin on orbitals "
                + " and ".join(str(c) for c in copies))
    return named


def _multiplicity_note(rep, block, fibers) -> str:
    """The sentence :attr:`HSPTable.note` carries where a pseudospin sets the content, or ``""``.

    Where the Hamiltonian conserves a pseudospin, the representation its copies
    carry is not fixed by it and was assigned by the lowest-angular-momentum
    rule (:attr:`..symmetry.AtomicLimitTable.multiplicity_spaces`), and every
    band on those orbitals carries it.  Where the representations the table
    admits give different site contents, the site content itself is that
    choice, and a set of bands with weight there is labelled by it and can read
    differently under another; where a declaration states the content there is
    no such choice.  Those giving one site content are labellings like any
    other, and :func:`_labelling_note` says where a set of bands rests on them.
    The weight is read off the eigenvectors the content was read from, where
    they are to hand, and a set of bands is taken to have some where they are
    not.
    """
    table = getattr(rep, "characters", None)
    spaces = tuple(getattr(table, "multiplicity_spaces", ()) or ())
    if not spaces:
        return ""
    nspin = int(rep.model.nspin)
    nstate = int(rep.model.nstate)
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


def hsp_table(rep, *, states=None, bands=None, degeneracy_tol=DEGENERACY_TOL,
              label_tol=LABEL_TOL, ktol=1e-5, atol=1e-8, mesh_tol=MESH_TOL,
              params=None) -> HSPTable | Refusal:
    """The coreps a set of bands carries at every high-symmetry point of the group: its symmetry vector.

    Parameters
    ----------
    rep : ..symmetry.SpaceGroupRep
        The model's representation, ``r.rep`` from :func:`..groups.realized_group`.
    bands : int or sequence of int
        The set of bands: ``n`` the lowest ``n`` bands, or the band indices,
        whole degenerate multiplets at every point.  Required.
    states : None or pythtb.WFArray, optional
        ``None`` (default) solves the model at the points, once; a solved
        ``WFArray`` is read as it stands, and a point its mesh does not hold is
        refused there.
    degeneracy_tol : float, optional
        The energy, in the units of ``H``, within which bands are one
        multiplet.  Default ``1e-8``.
    label_tol, ktol, atol, mesh_tol : float, optional
        The tolerances of :func:`corep_content`, read at each point.
    params : dict, optional
        Values for a model whose terms are given by parameter names.

    Returns
    -------
    HSPTable or Refusal
        ``t.summary()`` prints the coreps at each point.  ``t.counts`` is the
        symmetry vector, which :func:`..ebr.decompose`,
        :func:`..ebr.symmetry_indicator` and :func:`..srsi.srsi_of_bands` take;
        ``t.content`` the coreps point by point; ``t.complete`` whether every
        point resolved, and ``t.refusals`` each point that did not, with its own
        ``Refusal``; ``t.note`` where the labels rest on a choice the hoppings
        leave open.  A ``Refusal`` in its place is falsy and names what stopped
        the call.

    Raises
    ------
    ValueError
        States of another model than ``rep.model``, or ``bands`` outside the
        states.

    Refusals
    --------
    A refusal handed in as ``rep`` is returned unchanged.  A point that refuses
    is collected in ``refusals``.

    ``tables_not_ready``
        the group's tables cannot be read in this cell.

    Examples
    --------
    lin3g's six lowest bands (:func:`..models.lin3g`):

    >>> print(hsp_table(r.rep, bands=6).summary())                 # doctest: +SKIP
    bands 0-5   (.bands)
      their coreps at each high-symmetry point, GM being Γ   (.content)
        GM: 2×-GM6 + -GM7
        M: -M6 + 2×-M7
        X: 3×-X5
    """
    if bands is None:
        raise TypeError("hsp_table() missing required argument 'bands'")
    if isinstance(rep, Refusal):
        return rep
    block = _band_block(int(rep.model.nstate), bands)
    gate = _groups._lattice_not_descended(rep.group)
    if gate is not None:
        return gate
    if states is not None and hasattr(states, "solve_ham") and states is not rep.model:
        raise ValueError("hsp_table(rep, states=...) must be handed rep.model")
    points = _groups.high_symmetry_points(rep.group)
    if isinstance(points, Refusal):
        return points

    content, parts, records, refusals, skipped, thr = {}, {}, {}, {}, {}, {}
    counts: dict = {}

    # Collect the informative points first, then take *one* pass over them: on the
    # model path that is a single solve_ham over the whole k-list -- the convenient
    # route for "a handful of high-symmetry points", with no container built.
    wanted = []
    for name, k in points.items():
        lg = _groups.little_group(rep.group, k)
        if isinstance(lg, Refusal):
            refusals[name] = lg
            continue
        if len(lg.unitary) <= 1 and not lg.antiunitary:
            skipped[name] = "trivial little group (identity only): the content is the band count"
            continue
        wanted.append((name, np.asarray(k, dtype=float).reshape(-1)))

    per_point = {}
    if wanted and (states is None or states is rep.model):
        karr = np.array([k for _, k in wanted], dtype=float)
        evals, evecs = _bloch.solve(rep.model, karr, params)
        evals, evecs = np.asarray(evals), np.asarray(evecs)
        if evals.ndim == 1:                     # solve_ham drops a length-1 k axis
            evals, evecs = evals[None], evecs[None]
        per_point = {name: (evals[i], evecs[i]) for i, (name, _) in enumerate(wanted)}
    else:
        per_point = {name: states for name, _ in wanted}

    for name, k in wanted:
        got = corep_content(rep, k, states=per_point[name], bands=block,
                            degeneracy_tol=degeneracy_tol, label_tol=label_tol, ktol=ktol,
                            atol=atol, mesh_tol=mesh_tol, params=params)
        if isinstance(got, Refusal):
            refusals[name] = got
            continue
        records[name] = got
        content[name] = dict(got.counts)
        parts[name] = got.partition
        thr[name] = got.thresholds
        for lab, v in got.counts.items():
            counts[lab] = counts.get(lab, 0) + int(v)

    notes = (_multiplicity_note(rep, block, per_point.values()),
             _labelling_note(rep, block))
    return HSPTable(bands=block, points=dict(points), content=content, partitions=parts,
                    records=records, refusals=refusals, skipped=skipped,
                    counts=counts, thresholds=thr, complete=not refusals,
                    group_provenance=str(rep.group.provenance),
                    note="; ".join(n for n in notes if n))
