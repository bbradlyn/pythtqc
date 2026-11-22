"""Whether a Hamiltonian obeys each operation of a group, as measured.

``H(k)`` on the detection grid and each action's solution space are found once
and shared by every step of one call.  Each kept operation's solution space is
then read once in the operation's own form.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from . import _bloch
from .refusal import Refusal
from ._tolerance import DEFAULT_ENERGY, EXACT, IN_SPACE
from ._text import _DOUBLE_GROUP, _PER_COSET, _names_text
from ._spacegroup import GroupData
from ._seitz import OperationTable, _operation_rows
from ._detector import (
    _ConstraintPair, _GridHamiltonians, _as_k_action, _candidate_measures,
    _find_intertwiners)
from ._matrices import _identity_position, _unitarized
from ._multiplets import _own_form_spaces

__all__ = ["ValidatedGroup", "Discrepancy"]


@dataclass(frozen=True)
class Discrepancy:
    """One operation the Hamiltonian breaks, and how it was found broken.

    An entry of :attr:`ValidatedGroup.discrepancies`.

    Attributes
    ----------
    index : int
        The operation's index, as the group's operations list it
        (``r.operations`` for the group returned).
    refuted_by : {"no symmetry matrix", "covariance", "out of plane"}
        How the operation was found broken.  ``"no symmetry matrix"``: no
        nonzero matrix ``X`` has ``X H(k) = H(Rk) X`` at the k-points
        measured, within ``cut``, ``Rk`` being the operation's image of ``k``
        (``H(k)*`` in place of ``H(k)`` for an operation combined with time
        reversal).  ``"covariance"``: such matrices exist, and the operation's
        covariance residual is above ``tol``.  ``"out of plane"``: the
        operation does not map the layer's plane onto itself, so it has no
        action on the layer's ``k``.
    residual : float
        The operation's covariance residual, in the units of H, as
        :attr:`ValidatedGroup.residuals` holds it.  Infinite where no matrix
        was found, and for an operation out of the plane, for which
        ``residuals`` holds no entry.
    """

    index: int
    refuted_by: str
    residual: float


@dataclass(frozen=True)
class ValidatedGroup:
    """Whether the Hamiltonian obeys each operation of a group, as measured.

    :func:`..groups.realized_group` returns it as ``r.numerical.validated``,
    for the group returned.  A ``cell_group_not_realized`` refusal carries it
    in ``context["validated"]``, for the group in ``context["group"]``.  That
    refusal is returned where an operation combined with time reversal, kept
    before ``H`` is projected onto its symmetric part, is broken after it.

    An operation is kept where one unitary ``U`` has ``U H(k) U† = H(Rk)``
    within ``tol`` at every k-point measured, ``Rk`` being the operation's
    image of ``k``.  For an operation combined with time reversal, ``H(k)*``
    stands in place of ``H(k)``.  The operations kept are those in
    ``obeyed`` and in ``undetermined``.  Operations that act alike on ``k``,
    and alike in whether they are combined with time reversal, are measured
    once: a class found broken breaks every member, and a class kept with
    more than one member leaves each member undetermined.  ``summary()``
    returns the verdicts as text, with the largest covariance residual of
    the operations kept.

    Attributes
    ----------
    obeyed : tuple of int
        The operations kept that share their measurement with no other
        operation, by index.
    undetermined : tuple of int
        The operations kept that share their measurement with another
        operation, by index.  On a free-standing layer, an operation and its
        product with the horizontal mirror share one.  ``H(k)`` does not tell
        such operations apart, so each is then held to its own form: its
        solution space is read for a matrix ``O(g) ⊗ S(g)`` that carries each
        site's orbitals onto its image's, with the operation's own spin factor
        ``S(g)``.  In the record ``realized_group`` returns, each one holds.
    broken : tuple of int
        The operations broken, by index, one for each entry of
        :attr:`discrepancies`.
    discrepancies : tuple of Discrepancy
        One entry per broken operation, with how it was found broken.
    residuals : dict
        ``{index: residual}``, the covariance residual of each operation
        measured, in the units of H.  It is the largest absolute entry of
        ``U H(k) U† - H(Rk)`` over the k-points measured, for the unitary
        ``U`` drawn from the operation's solution space that makes it
        smallest.  It is ``0.0`` for the identity, and infinite where no
        matrix was found.  An operation out of the layer's plane has no
        entry.
    closed : bool
        Whether the operations kept form a group.
    tol : float
        The tolerance each covariance residual is held to, in the units of H.
    n_kpoints : int
        The number of k-points the residuals are measured on.
    """

    # the operations kept, those obeyed and those undetermined
    _indices: tuple
    discrepancies: tuple
    residuals: dict
    # the dimension of each operation's space of solutions, the identity's
    # counting the matrices that commute with H(k) at every k
    _nullspace_dims: dict
    # one member of each kept operation's space of solutions, the unitary
    # nearest it: unnormalized in phase, and generally not of the operation's
    # own form where the space holds more than one matrix
    _intertwiners: dict
    closed: bool
    tol: float
    n_kpoints: int
    # the operations grouped by their action on k and whether they carry time
    # reversal: one measurement settles a class
    _k_action_classes: tuple = ()
    # `{ind: verdict}`: "obeyed", "broken" or "undetermined"
    _verdicts: dict = field(default_factory=dict)
    # the group measured, "P6/mmm1' #191.234"
    _group_label: str = ""
    # what each verdict rests on, per operation solved: the space's dimension,
    # the residual, the solver's own report, the tolerance, cut and window it
    # was read at, and `solved_on`, the operation whose solve this one is
    _records: dict = field(default_factory=dict)
    # each kept operation's whole space of solutions, (dim, nstate, nstate),
    # shared by reference across a k-action class; in the record
    # `realized_group` returns, the members of the operation's own form
    _bases: dict = field(default_factory=dict)
    _group: Any = field(default=None, repr=False, compare=False)
    # ``None`` until each operation's space is read in its own form, once
    # (`_own_form_read`); then, for each operation none of whose members has
    # it, the condition that failed, the covariance of the matrix of that form
    # nearest its space where one was measured, and the detector's whole
    # space, which holds no matrix of the operation's own form and is read only
    # for whether any matrix with its action on k has the square the tables ask
    # for (`_antiunitary_square`)
    _own_form: Any = field(default=None, repr=False, compare=False)

    @property
    def broken(self) -> tuple:
        """The operations broken, by index, one for each entry of ``discrepancies``."""
        return tuple(d.index for d in self.discrepancies)

    @property
    def obeyed(self) -> tuple:
        """The operations kept that share their measurement with no other, by index."""
        return tuple(i for i in self._indices if self._verdicts.get(i) == "obeyed")

    @property
    def undetermined(self) -> tuple:
        """The operations kept that share their measurement with another, by index.

        ``H(k)`` does not tell them apart, so each is then held to its own form
        ``O(g) ⊗ S(g)``.  In the record ``realized_group`` returns, each one
        holds.
        """
        return tuple(i for i in self._indices
                     if self._verdicts.get(i) == "undetermined")

    def _counted(self) -> str:
        """The count of operations kept among those measured, as the summary prints it.

        ``"16 of 16 coset representatives of the lattice translations obeyed"``.
        """
        n_kept = len(self._indices)
        return f"{n_kept} of {n_kept + len(self.discrepancies)} {_PER_COSET} obeyed"

    def summary(self) -> str:
        """The verdicts, a line for each part, with the attribute that holds it.

        The parts are the counts of operations kept, broken and undetermined;
        the double group, where the group is one; the broken operations by
        name; where operations are undetermined, that they act on ``k`` as
        another does and ``H`` alone does not say which of each pair holds;
        whether the operations kept form a group, where they do not; and the
        largest covariance residual of those kept, against ``tol``.
        Returns the lines as one string.
        """
        n_broken, shared = len(self.discrepancies), len(self.undetermined)
        lines = [self._counted() + ", "
                 + ("none broken" if not n_broken else f"{n_broken} broken") + ", "
                 + ("none undetermined" if not shared else f"{shared} undetermined")
                 + "   (.obeyed, .broken, .undetermined)"]
        if bool(getattr(self._group, "spinor", False)):
            lines.append(f"  {_DOUBLE_GROUP}")
        if self.discrepancies:
            table = _operation_rows(self._group) if self._group is not None else {}
            names = [(table[int(i)].name if not isinstance(table, Refusal)
                      and int(i) in table else f"#{int(i)}") for i in self.broken]
            lines.append(f"  broken: {_names_text(names)}   (.broken, .discrepancies)")
        if shared:
            n_kept = len(self._indices)
            lines.append(("  every one acts on k as another does"
                          if shared == n_kept else
                          f"  {shared} act on k as another does")
                         + ", so H alone does not say which of each pair holds, and the "
                         "representation decides   (.undetermined)")
        if not self.closed:
            lines.append("  those obeyed do not form a group   (.closed)")
        held = [float(self.residuals[i]) for i in self._indices
                if i in self.residuals and np.isfinite(self.residuals[i])]
        if held:
            lines.append(f"  each one's covariance residual, at most {max(held):.2g} against "
                         f"the tolerance {float(self.tol):.2g}, on {int(self.n_kpoints)} "
                         "k-points   (.residuals, .tol, .n_kpoints)")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover, a convenience
        shared = len(self.undetermined)
        return (f"ValidatedGroup({self._counted()}, which "
                + ("form a group" if self.closed else "do not form a group")
                + (f"; they are {_DOUBLE_GROUP}"
                   if bool(getattr(self._group, "spinor", False)) else "")
                + (f"; {shared} act on every k as another does, so the Hamiltonian alone "
                   "does not say which of each pair holds, and the representation "
                   "decides" if shared else "")
                + ")")


# --------------------------------------------------------------------------
# the grid of k-points the operations are measured on
# --------------------------------------------------------------------------

def _detection_grid(dim_k: int, nk: int) -> np.ndarray:
    """The detection grid: ``nk`` points along each of ``dim_k`` directions.

    The points are ``j/nk``, ``j = 0, ..., nk - 1``, in reduced coordinates,
    ``k = 0`` included, built by pythtb's ``Mesh``.  Returns an array of shape
    ``(nk**dim_k, dim_k)``.  :func:`_validate_group` and the measurement of a
    translation, :func:`.._descent._translation_residual`, both read it, so a
    translation is measured on the same points as the rest of the group.
    """
    from pythtb import Mesh

    mesh = Mesh(["k"] * int(dim_k))
    mesh.build_grid((int(nk),) * int(dim_k), gamma_centered=False, k_endpoints=False)
    return np.asarray(mesh.flat)


# --------------------------------------------------------------------------
# the per-call measurement context
# --------------------------------------------------------------------------

def _model_key(model) -> str:
    """The key that identifies a Hamiltonian to :class:`_MeasurementContext`.

    It is a hash of ``model``'s lattice, orbital centers, on-site terms and
    hoppings (:func:`.._bloch.model_key`), and two models with the same key
    have the same ``H(k)`` at every ``k``.  So the other environment's slab
    presentation, a new model built from the same terms, shares the grid and the
    solution spaces already found.  Returns the key, a string.
    """
    return _bloch.model_key(model)


class _MeasurementContext:
    r"""``H(k)`` on the detection grid, and one solution space per action on ``k``.

    One is created per call of :func:`.._realized.realized_group` and handed to
    every :func:`_validate_group` the call makes: the first measurement, each
    step of the descent, the measurement after the hoppings are projected, and
    the other environment's.  ``H(k)`` on the grid is evaluated once
    (:class:`.._detector._GridHamiltonians`).  Each class of operations is
    solved once, keyed by its action on the periodic directions of ``k`` and
    whether it is antiunitary, and never by an operation's index, which every
    descent renumbers.  Where the sites are given, the key also holds the forms
    the class's space may be fitted in, unless the space's own candidate meets
    ``tol`` and nothing is fitted.  The translation is not in the key, since
    :func:`.._detector._find_intertwiners` does not read it.

    The grid and the spaces are rebuilt together where the model's key
    (:func:`_model_key`), the k-points, ``tol``, ``cut``, ``window``,
    ``solver``, ``max_bytes`` or the nullspace floor differs from the binding
    before (:meth:`bind`).  A move of the centers or a projection of the
    hoppings makes a model with a new key, and so a rebuild.

    Attributes
    ----------
    grid : _detector._GridHamiltonians or None
        ``H(k)`` on the bound grid, ``None`` before the first binding.
    spaces : dict
        The solution space found under each key.
    hits, misses, rebuilds : int
        The classes reused, the classes solved, and the bindings that rebuilt,
        the first included.
    sectors : dict
        The sectors of each model the call read
        (:func:`.._sectors.read_sectors`), keyed by :func:`_model_key`, the
        sites and spinor pairs they are read on and the tolerance.  A reading
        made where a model enters is kept through the later rounds and the
        other environment.
    """

    __slots__ = ("_key", "grid", "spaces", "hits", "misses", "rebuilds", "sectors")

    def __init__(self):
        self._key = None
        self.grid = None
        self.spaces = {}
        self.hits = 0
        self.misses = 0
        self.rebuilds = 0
        self.sectors = {}

    def bind(self, model, k_pts, *, tol, cut, window, solver, max_bytes=None,
             floor=None):
        """The grid of ``H(k)`` for ``model`` on ``k_pts``, rebuilt where its key differs.

        ``tol``, ``cut``, ``window``, ``solver``, ``max_bytes`` and ``floor``
        are the settings :func:`_validate_group` hands the detector.  With the
        model's key and the k-points they make the binding's key.  Where it
        differs from the last binding's, the grid is evaluated anew and every
        solution space kept is dropped.  Returns the
        :class:`.._detector._GridHamiltonians`.
        """
        key = (_model_key(model),
               np.ascontiguousarray(np.asarray(k_pts, dtype=float)).tobytes(),
               float(tol), None if cut is None else float(cut),
               None if window is None else float(window), str(solver),
               None if max_bytes is None else float(max_bytes),
               None if floor is None else float(floor))
        if key != self._key:
            self._key = key
            self.grid = _GridHamiltonians(model, k_pts)
            self.spaces = {}
            self.rebuilds += 1
        return self.grid

    def solved(self, key, *also):
        """The solution space already found under ``key``, or ``None``.

        ``also`` are keys tried first, under which a space that answers for
        ``key`` as well is kept.  Each call counts a hit where a space is
        found and a miss where none is.
        """
        got = next((self.spaces[k] for k in also if k in self.spaces), None)
        if got is None:
            got = self.spaces.get(key)
        if got is None:
            self.misses += 1
        else:
            self.hits += 1
        return got

    def remember(self, key, space) -> None:
        """Keep the solution space ``space`` under ``key`` until the next rebuild."""
        self.spaces[key] = space

    def grid_for(self, model, k_pts):
        """The bound grid where it is ``model``'s on the same ``k_pts``, else ``None``.

        The grid is returned only where ``model`` is the object it was bound
        to, not only a model of the same key, and ``k_pts`` are the same points
        bit for bit.  The measurement of a translation
        (:func:`.._descent._translation_residual`) and
        :func:`_grid_covariance` measure on the detection grid, and read
        ``H(k)`` off this one where they can.
        """
        if self.grid is None or self.grid.model is not model:
            return None
        k = np.atleast_2d(np.asarray(k_pts, dtype=float))
        if (k.shape == self.grid.k_pts.shape
                and np.ascontiguousarray(k).tobytes()
                == np.ascontiguousarray(self.grid.k_pts).tobytes()):
            return self.grid
        return None

    def record(self) -> dict:
        """The counters ``hits``, ``misses``, ``rebuilds`` and ``spaces``, by name.

        ``spaces`` is how many solution spaces are kept.
        """
        return {"hits": int(self.hits), "misses": int(self.misses),
                "rebuilds": int(self.rebuilds), "spaces": len(self.spaces)}


def _validate_group(model, group: GroupData, *, nk=8, tol=DEFAULT_ENERGY, cut=None,
                    k_pts=None, include_antiunitary=True, keep_intertwiners=True,
                    window=None, solver="eigen", max_bytes=None, context=None,
                    assignment=None, _floor=None) -> ValidatedGroup | Refusal:
    """Which operations of a group ``H(k)`` obeys, each with its covariance residual.

    Each operation of ``group`` is put to
    :func:`.._detector._find_intertwiners`, which finds every matrix ``X``
    with ``X H(k) = H(Rk) X`` on a grid of ``k``, and ``X H(k)* = H(Rk) X``
    for an antiunitary operation.  Operations that act alike on ``k``, and
    alike in whether they are antiunitary, pose one problem, solved once, and
    share its verdict.  Nothing checks that ``group`` is ``model``'s: its
    operations are measured against whatever Hamiltonian is handed in.

    Parameters
    ----------
    model : pythtb.TBModel
        The Hamiltonian measured.
    group : GroupData
        The operations measured.
    nk : int, default 8
        The number of points along each periodic direction of the grid
        (:func:`_detection_grid`).
    tol : float, default 1e-8
        The covariance check's tolerance, an energy in the units of ``H``.  An
        operation is kept where a unitary of its solution space carries
        ``H(k)`` onto ``H(Rk)`` within ``tol`` at every point of the grid.
        The detector's ``cut`` and ``window`` follow it where they are
        ``None``.
    cut : float, default None
        The detector's relative nullspace cut
        (:func:`.._detector._find_intertwiners`).  ``None`` is ``tol``.
    k_pts : ndarray, shape (Nk, dim_k), default None
        The points measured on, in reduced coordinates.  ``None`` is the grid
        of ``nk``.
    include_antiunitary : bool, default True
        Whether the group's antiunitary operations are measured.  With
        ``include_antiunitary=False`` they are absent from what is returned.
    keep_intertwiners : bool, default True
        Whether each kept operation's intertwiner and whole solution space
        are kept in what is returned, for :func:`.._build._build_rep`.
    window : float, default None
        The detector's starting energy window, relative to
        ``max|H - μ·1|``.  ``None`` derives it from ``tol``.
    solver : {"eigen", "stacked"}, default "eigen"
        The detector's route.
    max_bytes : float, default None
        The memory, in bytes, the detector may take at a point whose
        degenerate pairs need more than its own budget, its
        ``degenerate_max_bytes``.  ``None`` keeps its own budget.
    context : _MeasurementContext, default None
        The measurement one call shares among its steps, which holds ``H(k)``
        on the grid and the spaces already solved.  What is returned is the
        same with or without it.
    assignment : .._spacegroup.OrbitalAssignment, default None
        The cell's sites and each operation's permutation of them.  Given,
        the detector may fit a class's space over the whole grid among the
        matrices carrying each site's states onto its image's under one of
        the class's operations, its ``forms``.  ``None`` fits nothing.
    _floor : float, default None
        The detector's nullspace floor: the covariance at one point, in the
        units of ``H``, that a candidate it keeps may carry.  ``None`` is
        ``cut``.

    Returns
    -------
    ValidatedGroup or Refusal
        The verdict on each operation measured.  An operation is broken by
        ``"no symmetry matrix"`` where its space is empty, by
        ``"covariance"`` where its residual, the smallest over the unitaries
        drawn from its space, is above ``tol``, and by ``"out of plane"``
        where it does not map a layer's plane onto itself.  Each kept
        operation's intertwiner is a unitary of its space whose phase is not
        fixed; the identity's is the unit matrix, its residual ``0.0``.
        ``closed`` says whether the operations kept form a group under
        ``group``'s product table.

    Refusals
    --------
    ``detector_too_large``
        the degenerate pairs at one point need more memory than the larger of
        the detector's budget and ``max_bytes``, returned as the detector
        returns it; no operation is recorded broken.

    See Also
    --------
    _own_form_read : each kept operation's space read in its own form, the
        next step of :func:`.._realized.realized_group`.
    """
    if k_pts is None:
        k_pts = _detection_grid(int(group._dim_k), int(nk))
    k_pts = np.atleast_2d(np.asarray(k_pts, dtype=float))

    # One evaluation of H on the detection grid for the whole group: H(k) does
    # not depend on which operation is being tested, and it is summed from the
    # model's lattice-vector blocks inside a memory budget whatever the model's
    # size.  With a shared context, realized_group evaluates H once for the
    # whole call: the descent's steps, the measurement after a projection and
    # the other environment's measurement read the same H on the same grid.
    if context is None:
        grid_H = _GridHamiltonians(model, k_pts)
    else:
        grid_H = context.bind(model, k_pts, tol=tol, cut=cut,
                              window=window, solver=solver, max_bytes=max_bytes,
                              floor=_floor)

    keep, disc, residuals, dims, intertwiners = [], [], {}, {}, {}
    actions, solved_class, records, bases = {}, {}, {}, {}
    probe = np.eye(group._dim_k)
    # the identity, whose member is the unit matrix: it carries every H(k) onto
    # itself exactly, so its residual is that of the unit matrix, 0.0, and never
    # the rounding of a member found numerically on a nearly degenerate spectrum
    identity = int(group._symmetries[_identity_position(group)].ind)
    # the forms of each class's matrices, where the sites are given: for each
    # of the class's operations, which pose one problem and share its space,
    # the entries carrying a site's states onto its image's under it
    forms = {}
    if assignment is not None:
        site_state = np.repeat(np.asarray(assignment.site_of_orbital, dtype=int),
                               int(model.nspin))
        image_of = np.asarray(assignment.permutations, dtype=int)
        for position, sym in enumerate(group._symmetries):
            action = group._k_action_of(sym)
            if action is None:
                continue
            key = (np.round(np.asarray(action(probe), dtype=float), 10).tobytes(),
                   bool(sym.time_reversal))
            carried = site_state[:, None] == image_of[position][site_state][None, :]
            known = forms.setdefault(key, [])
            if not any(np.array_equal(carried, f) for f in known):
                known.append(carried)
    for sym in group._symmetries:
        ind = int(sym.ind)
        if sym.time_reversal and not include_antiunitary:
            continue
        action = group._k_action_of(sym)
        if action is None:
            disc.append({"ind": ind, "refuted_by": "out of plane", "residual": np.inf})
            continue
        actions[ind] = (action, bool(sym.time_reversal))
        # operations sharing (k-action, antiunitarity) pose the identical
        # problem, so it is solved once and `solved_on` names which operation
        key = (np.round(np.asarray(action(probe), dtype=float), 10).tobytes(),
               bool(sym.time_reversal))
        if key not in solved_class:
            # The class may already have been solved earlier in this call, by
            # the measurement before a descent step or by the other environment,
            # which poses the same in-plane actions.  `solved_on` stays this
            # group's own first member either way, so the record names an
            # operation of the group that was asked about.
            # the space depends on the forms it may be fitted in as well, except
            # where the sift's own candidate meets the gate and none is fitted
            class_forms = tuple(forms.get(key, ()))
            cached = key + (b"".join(f.tobytes() for f in class_forms),)
            space = None if context is None else context.solved(cached, key + (None,))
            if space is None and context is not None:
                # a space fitted in the forms its sift lay off answers for every
                # set of the forms it was asked with that holds them: the same
                # sift, the same forms off it, the same fit
                near = context.spaces.get(key + ("forms fitted",))
                if near is not None:
                    asked, near = near

                    def among(f, fs):
                        return any(np.array_equal(f, g) for g in fs)
                    if (all(among(f, class_forms) for f in near.fitted_forms)
                            and all(among(f, asked) for f in class_forms)):
                        space = near
            if space is None:
                space = _find_intertwiners(
                    model, action, k_pts, antiunitary=bool(sym.time_reversal), tol=tol,
                    cut=cut, window=window, solver=solver, degenerate_max_bytes=max_bytes,
                    hamiltonians=grid_H, forms=class_forms, _floor=_floor)
                if isinstance(space, Refusal):
                    return space            # the detector's memory limit, as it came
                if context is not None:
                    # a space depends on the forms unless nothing was fitted and
                    # its own candidate meets the gate
                    alone = (space.fit == "" and space.dim
                             and float(np.min(space.residuals)) <= float(tol))
                    context.remember(key + (None,) if alone else cached, space)
                    if space.fitted_forms:
                        context.remember(key + ("forms fitted",), (class_forms, space))
            solved_class[key] = (ind, space)
        solved_on, space = solved_class[key]
        dims[ind] = space.dim
        best = float(np.min(space.residuals)) if space.dim else np.inf
        if ind == identity and space.dim:
            best = 0.0
        residuals[ind] = best
        solved = {"smallest_singular_value": float(space.singular_values[0] / space.scale),
                  "method": space.method, "n_kpts_solve": int(space.n_kpts_solve)}
        records[ind] = {"dim": int(space.dim), "residual": best, **solved,
                        "window": float(space.window), "solver": str(solver),
                        "tol": float(tol), "cut": float(space.cut),
                        "basis_size": tuple(space.basis_size),
                        "escalations": tuple(space.escalations),
                        "k_solve": int(space.k_solve), "solved_on": int(solved_on),
                        "fit": str(space.fit)}
        if space.dim == 0:
            disc.append({"ind": ind, "refuted_by": "no symmetry matrix",
                         "residual": np.inf})
        elif best > tol:
            disc.append({"ind": ind, "refuted_by": "covariance", "residual": best})
        else:
            keep.append(ind)
            if keep_intertwiners:
                intertwiners[ind] = (np.eye(int(model.nstate), dtype=complex)
                                     if ind == identity else
                                     space.unitaries[int(np.argmin(space.residuals))])
                # The whole space, not only the member with the best residual:
                # where `dim >= 2` the solution of tensor-structure form is
                # generally a different member (see ValidatedGroup._bases), and
                # `_build._build_rep`'s read-off needs the span to find it.
                # Stored by reference, so a k-action class costs one array.
                bases[ind] = space.basis

    closed = True
    if keep:
        prod = group._get_product_table(get_diff=False)
        pos = {int(s.ind): i for i, s in enumerate(group._symmetries)}
        kept = set(keep)
        for a in keep:
            for b in keep:
                c = int(prod[pos[a], pos[b]])
                if int(group._symmetries[c].ind) not in kept:
                    closed = False
                    break
            if not closed:
                break

    # Group the tested operations by (k-action, antiunitarity): members of one
    # class posed the identical intertwiner problem, so a class survival is
    # shared while a class failure is individually conclusive (see the
    # ValidatedGroup docstring).  Unitary and antiunitary operations with the
    # same k-action solve different problems and are never classmates.
    by_key = {}
    for ind, (action, tr) in actions.items():
        # the k-action is a linear callable on reduced k; probing the basis
        # rows recovers its matrix, which keys the class
        key = (np.round(np.asarray(action(probe), dtype=float), 10).tobytes(), tr)
        by_key.setdefault(key, []).append(ind)
    classes = tuple(tuple(sorted(v)) for v in
                    sorted(by_key.values(), key=lambda v: min(v)))
    kept_set = set(keep)
    verdicts = {}
    for members in classes:
        for ind in members:
            if ind not in kept_set:
                verdicts[ind] = "broken"            # class failure breaks every member
            else:
                verdicts[ind] = "obeyed" if len(members) == 1 else "undetermined"

    return ValidatedGroup(_indices=tuple(keep),
                          discrepancies=tuple(Discrepancy(index=int(d["ind"]),
                                                          refuted_by=str(d["refuted_by"]),
                                                          residual=float(d["residual"]))
                                              for d in disc),
                          residuals=residuals, _nullspace_dims=dims, _intertwiners=intertwiners,
                          closed=closed, tol=float(tol), n_kpoints=int(k_pts.shape[0]),
                          _k_action_classes=classes, _verdicts=verdicts,
                          _group_label=f"{group.name} #{group.number_str}",
                          _records=records, _bases=bases, _group=group)


def _grid_covariance(model, group, *, nk, context=None):
    r"""A measure of a candidate matrix's covariance residual, as the detector's own.

    ``model`` and ``group`` are the Hamiltonian and the group it was measured
    against, and ``nk`` the number of points along each periodic direction of
    the grid :func:`_validate_group` measured on (:func:`_detection_grid`).
    ``H(k)`` on that grid is read off ``context`` where it holds it
    (:meth:`_MeasurementContext.grid_for`), and evaluated otherwise.  Returns a
    function of an operation's position in ``group._symmetries`` and a matrix,
    which returns ``max_k ‖U H(k) U† − H(Rk)‖_max`` over the grid, ``U`` the
    unitary nearest the matrix and ``H(k)*`` in place of ``H(k)`` for an
    antiunitary operation.  :func:`.._multiplets._own_form_spaces` holds a
    matrix of an operation's own form to it.
    """

    k_pts = _detection_grid(int(group._dim_k), int(nk))
    grid = None if context is None else context.grid_for(model, k_pts)
    if grid is None:
        grid = _GridHamiltonians(model, k_pts)
    syms = group._symmetries

    def residual(position, candidate):
        sym = syms[int(position)]
        act = _as_k_action(group._k_action_of(sym), int(group._dim_k))
        pair = _ConstraintPair(grid, grid.rotated(np.asarray(act(grid.k_pts), dtype=float)),
                               bool(sym.time_reversal))
        _unitaries, got, _unitarity = _candidate_measures(
            [np.asarray(candidate, dtype=complex)], int(grid.nstate), pair=pair)
        return float(got[0])

    return residual


def _own_form_read(model, group, assignment, spin, validated, *, tolerances, covariance=None):
    r"""Every kept operation's solution space, read once in its own form.

    The detector's space for an operation holds every matrix with its action
    on ``k``, those of other operations with that action among them.  Each kept
    operation's space is cut to the members of its own form, each site's states
    carried onto its image's with the operation's own spin lift on states that
    carry a spin (:func:`.._multiplets._own_form_spaces`).  Every later reading
    of the operation reads those members.

    ``model``, ``group``, ``assignment`` and ``spin`` are the Hamiltonian, the
    group it was measured against, the cell's sites with each operation's
    permutation of them, and the spin partners declared, ``None`` where none
    were.  ``validated`` is :func:`_validate_group`'s record for them.
    ``tolerances`` is the call's :class:`.._tolerance.Tolerances`, which every
    threshold of the reading reads.  ``covariance`` is
    :func:`_grid_covariance`'s measure: where a form is missed by more than
    rounding, the operation is kept only where the part in the form of the
    member nearest the form carries ``H(k)`` onto ``H(Rk)`` within
    ``validated.tol``.  ``None`` holds the form alone.

    Returns ``(validated, refuted)``.  In ``validated`` each kept operation's
    space is the members of its own form, and its intertwiner a member of it:
    the detector's own where it lies in the form, else the unitary nearest its
    projection onto the form, or nearest the form's first member where that
    projection vanishes.  An operation with no member of its form keeps an
    empty space and the detector's intertwiner, and ``_own_form`` holds the
    detector's whole space for it, read only for whether a matrix of an
    antiunitary operation squares to the sign the tables give it
    (:func:`.._descent._antiunitary_square`).  ``_own_form`` is set, so the
    record is read once.  ``refuted`` maps each such operation's index to its
    record from :func:`.._multiplets._own_form_spaces`, whose ``own_form``
    names the condition that failed.
    """

    kept = set(validated._indices)
    positions = [p for p, s in enumerate(group._symmetries) if int(s.ind) in kept]
    spaces, records = _own_form_spaces(model, group, assignment, validated, positions,
                                       spin=spin, tol=tolerances.matrix or None,
                                       covariance=covariance)
    bases = dict(validated._bases)
    intertwiners = dict(validated._intertwiners)
    own_form = {}
    inside = tolerances.noise(IN_SPACE)
    for ind, space in spaces.items():
        if bases.get(ind) is space:
            continue                       # every member already of the form
        if not len(space):
            own_form[int(ind)] = {"own_form": records[ind]["own_form"],
                                  "covariance": records[ind].get("covariance"),
                                  "space": bases.get(ind)}
        bases[ind] = space
        own = intertwiners.get(ind)
        if not len(space) or own is None:
            continue
        own = np.asarray(own, dtype=complex)
        flat = np.asarray(space).reshape(len(space), -1)
        y = np.linalg.lstsq(flat.T, own.reshape(-1), rcond=None)[0]
        near = np.einsum("j,jab->ab", y, space)
        if float(np.linalg.norm(own - near)) <= inside * max(float(np.linalg.norm(own)), 1.0):
            continue                       # the detector's own member is of the form
        member = _unitarized(near)
        intertwiners[ind] = member if member is not None else _unitarized(space[0])
    refuted = {int(i): dict(r) for i, r in records.items() if not r["realizable"]}
    return (replace(validated, _bases=bases, _intertwiners=intertwiners, _own_form=own_form),
            refuted)


def _own_form_once(model, group, assignment, spin, validated, *, tol=None):
    r"""``validated`` with each kept operation's space read in its own form, once.

    ``model``, ``group``, ``assignment``, ``spin`` and ``validated`` are as
    :func:`_own_form_read` takes them.  A record already read, as
    :func:`.._realized.realized_group` reads every record it measures, is
    returned as it stands.  One as :func:`_validate_group` returns it is read
    here by :func:`_own_form_read`, the covariance measured on a grid of
    ``n_kpoints ** (1/dim_k)`` points along each periodic direction, rounded
    and at least 2 (:func:`_grid_covariance`).  ``tol`` is the declared
    tolerance relative to ``max|H|``, the one the multiplets handed to
    :func:`.._build._build_rep` were measured at; ``None`` keeps the exact
    defaults.  Returns the record read; the refuted operations' records are
    not returned.
    """
    if getattr(validated, "_own_form", None) is not None:
        return validated
    dim = max(int(group._dim_k), 1)
    side = max(2, int(round(float(max(int(validated.n_kpoints), 1)) ** (1.0 / dim))))
    covariance = _grid_covariance(model, group, nk=side)
    tolerances = replace(EXACT, matrix=float(tol or 0.0))
    got, _refuted = _own_form_read(model, group, assignment, spin, validated,
                                   tolerances=tolerances, covariance=covariance)
    return got


#: How ``max|H|`` is measured: the largest entry of ``H(k)`` over this many
#: seeded random k, floored by the model's own largest hopping or on-site
#: magnitude.  The largest entry over every k bounds them,
#: ``|t_ij(R)| ≤ max_k |H_ij(k)|``, so the floor only catches a sample of k
#: that undershot.  The mean on-site energy is taken off the diagonal
#: throughout.
_MAX_H_NK, _MAX_H_SEED = 64, 3


def _max_hamiltonian(model) -> float:
    r"""``max|H|``, the largest entry of the Hamiltonian about its mean on-site energy.

    ``model`` is the Hamiltonian.  ``μ``, the mean on-site energy per state
    (:func:`.._bloch.onsite_mean`), is taken off the diagonal, so a constant
    shift of ``H`` moves nothing.  Returns, in the units of ``H``, the largest
    of three magnitudes: of ``H(k) − μ·1`` over 64 seeded random ``k``, of the
    on-site terms less ``μ``, and of the hoppings.  The last two are bounded
    by the largest entry of ``H(k) − μ·1`` over every ``k``,
    ``|t_ij(R)| ≤ max_k |H_ij(k)|``, so they exceed the sampled one only
    where the sample undershot.  A relative precision is read against this
    number.
    """
    mu = _bloch.onsite_mean(model)
    site = _bloch.onsite(model)
    got = (float(np.abs(site - mu).max()) if site.ndim == 1
           else _bloch.magnitude(site, mu)) if site.size else 0.0
    amps = _bloch.hopping_arrays(model)[0]
    if amps.size:
        got = max(got, float(np.abs(amps).max()))
    dim_k = int(model.lattice.dim_k)
    if dim_k:
        k = np.random.default_rng(_MAX_H_SEED).uniform(-0.5, 0.5,
                                                       size=(_MAX_H_NK, dim_k))
        got = max(got, _bloch.magnitude(_bloch.hamiltonian(model, k), mu))
    return float(got)
