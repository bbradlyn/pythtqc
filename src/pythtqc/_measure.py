"""``H(k)`` on the detection grid and one solution space per k-action, shared by
every step of one call: the verdict on each operation of a group, and each kept
operation's solution space read once in its own form.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
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

__all__ = ["ValidatedGroup"]


@dataclass(frozen=True)
class ValidatedGroup:
    """The verdict of the Hamiltonian on each operation of a group, with how well each one holds.

    The record ``realized_group(model).validated`` holds, and the one a
    ``cell_group_not_realized`` refusal carries in ``context["validated"]``;
    ``summary()`` prints it.

    Attributes
    ----------
    verdicts : dict
        ``{ind: verdict}``: ``"obeyed"``; ``"broken"``; or ``"undetermined"``,
        a verdict shared with another operation that acts on ``k`` alike, as an
        operation and its product with the horizontal mirror do on a
        free-standing layer, which the representation then settles.
    indices : tuple of int
        The operations kept, those obeyed and those undetermined.
    obeyed, undetermined, broken : tuple of int
        The indices with each verdict.
    operations : OperationTable
        The operations kept, by index, with their Seitz symbols and names.
    discrepancies : tuple of dict
        One entry per broken operation: its ``ind``; ``refuted_by``, ``"no
        symmetry matrix"`` where no matrix with its action on ``k`` carries
        ``H(k)`` onto itself, a proof, ``"covariance"`` where the nearest one
        misses by more than ``tol``, or ``"out of plane"`` for an operation of
        a layer's slab that the model's ``k`` cannot represent; and its
        ``residual``.
    residuals : dict
        ``{ind: residual}``: each operation's covariance residual,
        ``max_k ‖U H(k) U† − H(Rk)‖`` in the units of ``H``, infinite where no
        matrix exists.
    tol, n_kpts : float and int
        The tolerance the residuals are read against, in the units of ``H``,
        and the number of ``k``-points they are measured on.
    k_action_classes : tuple of tuple of int
        The operations grouped by their action on ``k`` and whether they carry
        time reversal: one measurement settles a class.
    closed : bool
        Whether the operations kept form a group.
    group_label : str
        The group measured, ``"P6/mmm1' #191.234"``.
    """

    indices: tuple
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
    n_kpts: int
    k_action_classes: tuple = ()
    verdicts: dict = field(default_factory=dict)
    group_label: str = ""
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
    # space -- no matrix of the operation's own, read only for whether any
    # matrix with its action on k has the square the tables ask for
    # (`_antiunitary_square`)
    _own_form: Any = field(default=None, repr=False, compare=False)

    @property
    def operations(self) -> OperationTable:
        """The operations kept, by index, as :func:`.._seitz.operations` writes them.

        Returns
        -------
        OperationTable or Refusal
            A row for each entry of :attr:`indices`.

        Refusals
        --------
        ``tables_not_ready``
            The group has no conventional setting to write a Seitz symbol in.
        """
        table = _operation_rows(self._group)
        if isinstance(table, Refusal):
            return table
        return OperationTable((table[int(i)] for i in sorted(self.indices)),
                              name=table.name, number_str=table.number_str)

    @property
    def broken(self) -> tuple:
        """The indices of the operations the Hamiltonian breaks."""
        return tuple(d["ind"] for d in self.discrepancies)

    @property
    def obeyed(self) -> tuple:
        """The indices of the operations obeyed, each the only one of its class to act on ``k`` as it does."""
        return tuple(i for i in self.indices if self.verdicts.get(i) == "obeyed")

    @property
    def undetermined(self) -> tuple:
        """The indices of the operations kept whose verdict another operation, acting on ``k`` alike, shares.

        The measurement says some operation of the class holds; the
        representation settles which.
        """
        return tuple(i for i in self.indices
                     if self.verdicts.get(i) == "undetermined")

    def summary(self) -> str:
        """The verdicts, a line for each part, with the attribute that holds it."""
        n_kept = len(self.indices)
        lines = [(f"{self.group_label}: " if self.group_label else "")
                 + f"{n_kept} of {n_kept + len(self.discrepancies)} {_PER_COSET} obeyed   "
                 "(.indices, .verdicts)"]
        if bool(getattr(self._group, "spinor", False)):
            lines.append(f"  {_DOUBLE_GROUP}")
        if self.discrepancies:
            table = _operation_rows(self._group) if self._group is not None else {}
            names = [(table[int(i)].name if not isinstance(table, Refusal)
                      and int(i) in table else f"#{int(i)}") for i in self.broken]
            lines.append(f"  broken: {_names_text(names)}   (.broken, .discrepancies)")
        shared = len(self.undetermined)
        if shared:
            lines.append(("  every one acts on k as another does"
                          if shared == n_kept else
                          f"  {shared} act on k as another does")
                         + ", so H alone does not say which of each pair holds, and the "
                         "representation decides   (.undetermined, .k_action_classes)")
        if not self.closed:
            lines.append("  those obeyed do not form a group   (.closed)")
        held = [float(self.residuals[i]) for i in self.indices
                if i in self.residuals and np.isfinite(self.residuals[i])]
        if held:
            lines.append(f"  each one's covariance residual, at most {max(held):.2g} against "
                         f"the tolerance {float(self.tol):.2g}, on {int(self.n_kpts)} k-points   "
                         "(.residuals, .tol, .n_kpts)")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        shared = len(self.undetermined)
        return (f"ValidatedGroup({self.group_label + ', ' if self.group_label else ''}"
                f"{len(self.indices)} of {len(self.indices) + len(self.discrepancies)} "
                f"{_PER_COSET} obeyed, which "
                + ("form a group" if self.closed else "do not form a group")
                + (f"; they are {_DOUBLE_GROUP}"
                   if bool(getattr(self._group, "spinor", False)) else "")
                + (f"; {shared} act on every k as another does, so the Hamiltonian alone "
                   "does not say which of each pair holds, and the representation "
                   "decides" if shared else "")
                + ")")


# --------------------------------------------------------------------------
# the "irrep says / Hamiltonian does" split
# --------------------------------------------------------------------------

def _detection_grid(dim_k: int, nk: int) -> np.ndarray:
    """The detection mesh, from pythtb's own Monkhorst-Pack hypercube.

    Not a hand-rolled meshgrid: same points, one owner.  Shared by
    :func:`_validate_group` and by the lattice measurement of
    :func:`.._descent._translation_residual`, so that a translation is measured on exactly
    the k-set the rest of the group was.
    """
    from pythtb import Mesh

    mesh = Mesh(["k"] * int(dim_k))
    mesh.build_grid((int(nk),) * int(dim_k), gamma_centered=False, k_endpoints=False)
    return np.asarray(mesh.flat)


# --------------------------------------------------------------------------
# the per-call measurement context
# --------------------------------------------------------------------------

def _param_fingerprint(value) -> str:
    """A content-complete identity for one ``params`` value.

    ``repr`` is not one: numpy truncates the repr of an array past 1000 elements
    with ``...``, so two arrays that differ only beyond that point repr alike
    and would share one cached ``H(k)``.  Arrays and scalars are hashed by
    dtype, shape and bytes; anything numpy cannot hold (an object array) falls
    back to ``repr``.
    """
    arr = np.asarray(value)
    if arr.dtype != object:
        h = hashlib.blake2b(digest_size=16)
        h.update(repr((arr.dtype.str, arr.shape)).encode())
        h.update(np.ascontiguousarray(arr).tobytes())
        return h.hexdigest()
    return repr(value)


def _model_key(model) -> str:
    """What identifies a Hamiltonian to :class:`_MeasurementContext`: its content.

    The lattice, the orbital centres, the on-site block and the whole hopping
    list — the Hamiltonian and nothing else — hashed together from the arrays
    pythtb stores them in, at every size: a wannier90 import's 723,408
    hoppings hash in under a tenth of a second.  Two models with the same key
    have the same ``H(k)`` at every ``k``, which is the whole claim the context
    rests on: it is what lets the cross-check's realized group in the other
    environment, whose slab presentation is a *different object* with
    bit-identical content (measured, and refused if it were not), share the
    grid and the solution spaces the first one paid for.  Where no term
    carries a parameter, the model's lattice-vector blocks are kept under the
    same key (:func:`.._bloch.model_key`).
    """
    return _bloch.model_key(model)


class _MeasurementContext:
    r"""``H(k)`` on the detection grid, and one solution space per k-action.

    For the bundled pipeline that goes straight from model to realized group
    there is no reason to recompute ``H(k)`` and the solution spaces at every
    step of the descent.  One of these is created per :func:`.._realized.realized_group`
    call and handed to every :func:`_validate_group` that call makes — the first
    measurement, one per descent step, the re-measurement after the hopping
    projection, and the cross-check's other environment — so that

    * ``H(k)`` on the detection mesh is evaluated **once**, by
      :class:`.._detector._GridHamiltonians` from the model's lattice-vector
      blocks — in one call where the grid fits its memory budget, in bounded
      k-chunks where it does not — and
    * each **k-action class** is solved once, keyed by the in-plane action the
      detector poses its problem in — the rotation restricted to the periodic
      directions together with the antiunitary flag — and never by an
      operation's index, which is renumbered by every descent and differs
      between the two environments.  The translation is deliberately **not** in
      the key: :func:`.._detector._find_intertwiners` never sees it (a
      translation's phase is absorbed into the intertwiner ``X``), which is why
      :func:`_validate_group` already shares one solve among the members of a
      class within a single call.  This lifts that same sharing to the whole
      pipeline.

    That the two environments share almost everything is not a coincidence: the
    substrate group is the ``R[2,2] = +1`` subgroup of the free-standing one and
    ``σ_h`` acts trivially on in-plane ``k``, so the two pose the *same* set of
    in-plane k-actions.

    **Invalidation is structural, not remembered.**  The key carries the
    model's content (:func:`_model_key`, whatever the model's size), the k-set,
    the model parameters and the six numbers the solution space depends on —
    ``tol``, ``cut``, ``window``, ``solver``, the detector's ceiling
    ``max_bytes``, and the floor below which its nullspace keeps a candidate,
    which a declared precision sets at the gate — and any difference rebuilds
    grid and spaces together.  The two steps of the chain that change the
    Hamiltonian, the position pass and the hopping projection, both hand back a
    new model with new content and therefore invalidate it by construction.

    The counters are measurements: ``hits`` and ``misses`` count classes reused
    and classes solved, ``rebuilds`` counts invalidations (the first bind
    included).

    ``sectors`` keeps the sectors of each model the call measured
    (:func:`.._sectors.read_sectors`), keyed by the model's content with the
    sites and spinor pairs they are read on and the tolerance they are read at:
    read where a model enters, on the first round, and kept through every later
    one and the other environment, which measure the same Hamiltonian.  A model
    the hopping projection made is a new one, and the reading made before it is
    the one its rounds carry.
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

    def bind(self, model, k_pts, params, *, tol, cut, window, solver, max_bytes=None,
             floor=None):
        """The shared grid for this request, rebuilt when anything it rests on moved."""
        key = (_model_key(model),
               np.ascontiguousarray(np.asarray(k_pts, dtype=float)).tobytes(),
               tuple(sorted((str(k), _param_fingerprint(v))
                            for k, v in (params or {}).items())),
               float(tol), None if cut is None else float(cut),
               None if window is None else float(window), str(solver),
               None if max_bytes is None else float(max_bytes),
               None if floor is None else float(floor))
        if key != self._key:
            self._key = key
            self.grid = _GridHamiltonians(model, k_pts, params)
            self.spaces = {}
            self.rebuilds += 1
        return self.grid

    def solved(self, key):
        """The solution space already found for this k-action class, or ``None``."""
        got = self.spaces.get(key)
        if got is None:
            self.misses += 1
        else:
            self.hits += 1
        return got

    def remember(self, key, space) -> None:
        """Keep a solution space for the rest of this call."""
        self.spaces[key] = space

    def grid_for(self, model, k_pts):
        """The bound grid when it is this model on exactly this k-set, else ``None``.

        The lattice measurement (:func:`.._descent._translation_residual`) runs on the
        detection mesh by construction — see :func:`_detection_grid` — so it can
        read ``H`` off the grid the detector already evaluated instead of
        evaluating it again.  Both halves of that are **checked** rather than
        assumed: the model has to be the one the grid was built on (identity, so
        a context rebound to some other Hamiltonian can never be read here), and
        the k-set has to be the same points byte for byte.
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
        """The counters, so what the sharing did can be reported."""
        return {"hits": int(self.hits), "misses": int(self.misses),
                "rebuilds": int(self.rebuilds), "spaces": len(self.spaces)}


def _validate_group(model, group: GroupData, *, nk=8, tol=DEFAULT_ENERGY, cut=None,
                    k_pts=None, include_antiunitary=True, keep_intertwiners=True,
                    window=None, solver="eigen", max_bytes=None, context=None,
                    params=None, _floor=None) -> ValidatedGroup | Refusal:
    """Which operations of a group ``H(k)`` obeys, one residual each.

    Returned as :attr:`.._realized.RealizedGroup.validated`; the descent and the
    symmetric projection put the same question to a group at a tighter ``tol``,
    on a different mesh, or to a subgroup assembled by hand.  Each candidate operation goes to
    :func:`.._detector._find_intertwiners`, which solves ``X H(k) = H(Rk) X`` (or
    ``X H(k)* = H(Rk) X`` for an antiunitary one) over a k-grid with no tensor
    structure assumed for what the operation does to orbitals, so an empty
    solution space is a *proof* that the operation is broken rather than a
    threshold verdict
    (``docs/background/measurement.md#the-cell-and-a-verdict-on-each-operation``).

    Parameters
    ----------
    model : pythtb.TBModel
        The Hamiltonian under test; its ``hamiltonian`` is evaluated on the
        detection mesh and nothing about its geometry is re-derived here.
    group : GroupData
        The candidate operations.  Which group goes with which model is not
        checked — see the validity conditions below.  A group comes from
        :func:`.._realized.realized_group`: ``realized_group(model, ...).group``.
    nk : int, optional
        Divisions of the Γ-centred detection mesh per periodic direction.
        Default 8.
    tol : float, optional
        **The declared symmetry tolerance**, absolute and in the units of H:
        the covariance gate on ``max_k ‖U H U† − H(Rk)‖_max``, and the number
        the detector's other two follow — ``cut`` defaults to it, and the
        matching ``window`` is derived from it as
        ``clip(10·sqrt(tol), 1e-3, 0.3)``.  So one declaration still sets gate,
        cut and window, and they cannot silently disagree.  A **wannier90**
        Hamiltonian is symmetric only to its own numerical defect, several
        decades above machine precision, so such a model must declare it:
        ``_validate_group(model, group, tol=1e-6)``.  Every per-operation entry
        then carries that number, which is what the verdict rests on
        (``docs/guide/numerical.md``).
    cut : float, optional
        The detector's **nullspace cut**, relative and dimensionless: a
        singular value counts as zero when ``s <= cut·‖L‖``, or when it is at
        most ``sqrt(n)·cut`` in the units of ``H``, the most a unitary candidate
        within the cut can have
        (:func:`.._detector._find_intertwiners`).  ``None`` (the
        default) is ``tol``.  Declaring it apart says that this model's
        *arithmetic* is noisier than the physics the gate judges — the gate
        stays where the residuals are read, the cut moves to where the
        solution space is found.
    k_pts : array, optional
        Use this k-set instead of the default mesh.
    include_antiunitary : bool, optional
        Also test the group's antiunitary operations.
    keep_intertwiners : bool, optional
        Keep, for each surviving operation, the polar-extracted unitary
        (``ValidatedGroup._intertwiners``) **and the detector's whole
        solution space** (``ValidatedGroup._bases``).  That space is what
        :func:`.._build._build_rep`'s ``strategy="readoff"`` reads the
        representation's discrete choices off.
    window, solver : optional
        Forwarded to :func:`.._detector._find_intertwiners`: the eigen route's
        matching window (``None`` derives it from ``tol``) and which solver
        runs (``"eigen"`` by default, ``"stacked"`` for the audit oracle).
    max_bytes : float, optional
        The memory the detector's trial basis may take at a k-point where a
        spectrum's exactly degenerate levels need more than its own
        ``512e6`` bytes: the ceiling
        :func:`.._detector._find_intertwiners` raises its budget to there
        (``degenerate_max_bytes``).  Never below the detector's own budget,
        which is also how wide an energy window it reads on a nearly
        degenerate spectrum and which this does not move.  ``None`` (default)
        keeps that budget as the ceiling.
    context : optional
        A shared measurement context, which :func:`.._realized.realized_group` creates once
        per call and hands to every measurement it makes, so that a k-action
        class posed twice is solved once.  ``None`` (the default) is this
        function's behaviour bit for bit, and what comes back is identical
        either way: the solve is a deterministic function of the model, the
        k-set and ``tol``/``cut``/``window``/``solver``/``max_bytes``.
    params : dict, optional
        Model parameter values, as ``model.hamiltonian`` takes them.

    Returns
    -------
    ValidatedGroup or Refusal
        Carries the surviving indices, the discrepancy list (with reasons and
        residuals), and :attr:`ValidatedGroup.closed` — whether the survivors
        form a subgroup under the group's own product table.

    Refusals
    --------
    ``"detector_too_large"``
        one operation's solve needs more memory than ``max_bytes`` allows, its
        spectrum's exactly degenerate pairs alone being past it: handed back
        as :func:`.._detector._find_intertwiners` returns it, with what the
        solve needs and the ``max_bytes=`` that runs it.  A limit on memory,
        not a statement about the model, so no operation is recorded broken.

    Notes
    -----
    **Validity conditions.**

    * ``reason="no symmetry matrix"`` is a *proof* the operation is broken: no
      matrix with that k-action commutes with ``H(k)`` at all.  At a declared
      ``tol`` above machine precision it is a proof at that tolerance, not a
      claim that the operation fails to machine precision — for the defect
      itself, measure it with ``solver="stacked"``.
      ``reason="covariance"`` is weaker: a solution exists but does
      not close the gate, which usually means an almost-symmetry or a badly
      conditioned grid.
    * The extracted ``intertwiners`` carry an **arbitrary overall phase**.
      They are candidates for :func:`..symmetry.normalize_rep`, not finished
      representations: a stored operator equal to ``(−i)×`` the true one passes
      every covariance test and relabels every corep at once.
    * ``closed`` is a self-check: a surviving set that is not closed under
      composition means the gate, the grid or the model is wrong, and it is
      reported rather than patched.
    * **Verdicts are per k-action class, not per operation.**  Operations
      sharing a k-action and an antiunitarity — for a layer, every pair
      ``{g, σ_h·g}`` — pose one intertwiner problem and share one verdict, so a
      survivor of a multi-member class reads ``"undetermined"`` rather than
      ``"obeyed"``.  **"Everything survives" on a layer is therefore not a
      statement that the horizontal mirror holds.**
    * **The model/group pairing belongs to whoever makes it** and is
      deliberately unchecked: the operations are measured against whatever
      Hamiltonian is handed in.  That is what lets a cell's candidates be
      measured against a model whose own point set would give a larger group —
      but a wrong pairing is then silent.

    Examples
    --------
    A square-lattice model whose cell is ``P4/mmm`` (16 operations) but whose
    hoppings are anisotropic keeps only the 8 operations of ``Pmmm``; the
    discrepancy list names the eight that fail and their residuals.
    """
    if k_pts is None:
        k_pts = _detection_grid(int(group.dim_k), int(nk))
    k_pts = np.atleast_2d(np.asarray(k_pts, dtype=float))

    # One evaluation of H on the detection grid for the whole group: H(k) does
    # not depend on which operation is being tested, and it is summed from the
    # model's lattice-vector blocks inside a memory budget whatever the model's
    # size.  With a shared context that evaluation is for the whole *pipeline*
    # rather than for this call -- the descent's steps, the re-measurement after
    # a projection and the other environment's realized group measure the same H
    # on the same mesh.
    if context is None:
        grid_H = _GridHamiltonians(model, k_pts, params)
    else:
        grid_H = context.bind(model, k_pts, params, tol=tol, cut=cut,
                              window=window, solver=solver, max_bytes=max_bytes,
                              floor=_floor)

    keep, disc, residuals, dims, intertwiners = [], [], {}, {}, {}
    actions, solved_class, records, bases = {}, {}, {}, {}
    probe = np.eye(group.dim_k)
    # the identity, whose member is the unit matrix: it carries every H(k) onto
    # itself exactly, so its residual is that of the unit matrix, 0.0, and never
    # the rounding of a member found numerically on a nearly degenerate spectrum
    identity = int(group.symmetries[_identity_position(group)].ind)
    for sym in group.symmetries:
        ind = int(sym.ind)
        if sym.time_reversal and not include_antiunitary:
            continue
        action = group.k_action_of(sym)
        if action is None:
            disc.append({"ind": ind, "refuted_by": "out of plane", "residual": np.inf})
            continue
        actions[ind] = (action, bool(sym.time_reversal))
        # operations sharing (k-action, antiunitarity) pose the *identical*
        # problem, so it is solved once and `solved_on` names which operation
        key = (np.round(np.asarray(action(probe), dtype=float), 10).tobytes(),
               bool(sym.time_reversal))
        if key not in solved_class:
            # The class may already have been solved earlier in this pipeline --
            # by the measurement before a descent step, or by the other
            # environment, which poses the identical in-plane actions.
            # `solved_on` stays THIS group's own first member either way, so the
            # record says which operation of the group that was asked about.
            space = None if context is None else context.solved(key)
            if space is None:
                space = _find_intertwiners(
                    model, action, k_pts, antiunitary=bool(sym.time_reversal), tol=tol,
                    cut=cut, window=window, solver=solver, degenerate_max_bytes=max_bytes,
                    hamiltonians=grid_H, params=params, _floor=_floor)
                if isinstance(space, Refusal):
                    return space            # the detector's memory limit, as it came
                if context is not None:
                    context.remember(key, space)
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
                        "k_solve": int(space.k_solve), "solved_on": int(solved_on)}
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
                # generally a *different* member (see ValidatedGroup._bases), and
                # `_build._build_rep`'s read-off needs the span to find it.
                # Stored by reference, so a k-action class costs one array.
                bases[ind] = space.basis

    closed = True
    if keep:
        prod = group.get_product_table(get_diff=False)
        pos = {int(s.ind): i for i, s in enumerate(group.symmetries)}
        kept = set(keep)
        for a in keep:
            for b in keep:
                c = int(prod[pos[a], pos[b]])
                if int(group.symmetries[c].ind) not in kept:
                    closed = False
                    break
            if not closed:
                break

    # Group the tested operations by (k-action, antiunitarity): members of one
    # class posed the identical intertwiner problem, so a class survival is
    # shared while a class failure is individually conclusive (see the
    # ValidatedGroup docstring).  Unitary and antiunitary operations with the
    # same k-action solve *different* problems and are never classmates.
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

    return ValidatedGroup(indices=tuple(keep), discrepancies=tuple(disc),
                          residuals=residuals, _nullspace_dims=dims, _intertwiners=intertwiners,
                          closed=closed, tol=float(tol), n_kpts=int(k_pts.shape[0]),
                          k_action_classes=classes, verdicts=verdicts,
                          group_label=f"{group.name} #{group.number_str}",
                          _records=records, _bases=bases, _group=group)


def _grid_covariance(model, group, *, nk, params=None, context=None):
    r"""A candidate matrix's covariance residual, measured as the detector measures its own.

    Returns the measure :func:`.._multiplets._own_form_spaces` hands a
    candidate to: called with an operation's position in
    ``group.symmetries`` and a matrix, it returns
    ``max_k ‖U H(k)^(*) U† − H(Rk)‖`` for ``U`` the unitary nearest that matrix,
    over the detection grid the detector solved on -- the same grid, read off
    the call's measurement context where it holds it, and the same number the
    detector certifies each of its own candidates by.
    """

    k_pts = _detection_grid(int(group.dim_k), int(nk))
    grid = None if context is None else context.grid_for(model, k_pts)
    if grid is None:
        grid = _GridHamiltonians(model, k_pts, params)
    syms = group.symmetries

    def residual(position, candidate):
        sym = syms[int(position)]
        act = _as_k_action(group.k_action_of(sym), int(group.dim_k))
        pair = _ConstraintPair(grid, grid.rotated(np.asarray(act(grid.k_pts), dtype=float)),
                               bool(sym.time_reversal))
        _unitaries, got, _unitarity = _candidate_measures(
            [np.asarray(candidate, dtype=complex)], int(grid.nstate), pair=pair)
        return float(got[0])

    return residual


def _own_form_read(model, group, assignment, spin, validated, *, tolerances, covariance=None):
    r"""Every kept operation's solution space, read once in its own form.

    The detector's spaces are per action on ``k``, and hold every matrix with
    that action: another operation's, or one composed with a symmetry the
    Hamiltonian has beyond the crystal's.  Each operation is held to its own
    form -- each site's states carried onto its image's, with its own spin lift
    on states that carry a spin -- once, here, and the members of that form are
    what every later reading of the operation reads: the multiplets, the
    transport round an orbit, the tensor structure, the sectors and the
    representation (:func:`.._multiplets._own_form_spaces`).  ``tolerances`` is the
    record of what the call declared, every threshold judging noise against
    it, and ``covariance`` is :func:`_grid_covariance`'s measure, which a form
    missed by more than rounding is held to as well: the member nearest it,
    its part in the form, carries ``H(k)`` onto ``H(gk)`` within the covariance
    gate the operation was kept at, or the operation is not one of this
    Hamiltonian's -- as spin blocks written in a frame turned against the
    lattice by more than the gate allows show.

    Returns ``(validated, refuted)``: the detector's record with each kept
    operation's space in its own form and its intertwiner a member of it -- the
    detector's own, made unitary on the form where it lies outside -- and the
    records of the operations none of whose members has their form, refuted by
    it (``"tensor structure"``, with ``own_form`` naming the condition that
    failed and ``covariance`` how far the part in the form carries ``H``,
    where that was measured).  A refuted operation's space is empty, and its
    intertwiner and the detector's whole space are left as the detector
    measured them, for the one question asked of them: whether any matrix
    with its action on ``k`` has the square the tables ask for, which time
    reversal's verdict says (:func:`.._descent._antiunitary_square`).  The record
    returned says it was read, so that it is read once:
    :func:`.._build._build_rep`, handed it, reads it as it stands.
    """

    kept = set(validated.indices)
    positions = [p for p, s in enumerate(group.symmetries) if int(s.ind) in kept]
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


def _own_form_once(model, group, assignment, spin, validated, *, tol=None, params=None):
    r"""The record each operation's own form was read on: as it stands where it was read, else read here, once.

    :func:`.._realized.realized_group` reads every record in the verdict stage
    (:func:`_own_form_read`); a record handed in as the detector measured it,
    :func:`_validate_group`'s own, is read here instead, by the same function.
    ``tol`` is the declared tolerance, relative to ``max|H|``, the reading
    judges noise against -- the one the multiplets handed in were measured
    at, ``None`` for the exact defaults -- and an operation whose members miss
    their form by more than rounding holds where the member nearest it, its
    part in the form, carries ``H(k)`` within the covariance gate it was kept
    at, measured over a mesh as many points as the record's own.
    """
    if getattr(validated, "_own_form", None) is not None:
        return validated
    dim = max(int(group.dim_k), 1)
    side = max(2, int(round(float(max(int(validated.n_kpts), 1)) ** (1.0 / dim))))
    covariance = _grid_covariance(model, group, nk=side, params=params)
    tolerances = replace(EXACT, matrix=float(tol or 0.0))
    got, _refuted = _own_form_read(model, group, assignment, spin, validated,
                                   tolerances=tolerances, covariance=covariance)
    return got


#: How ``max|H|`` is measured: the largest entry of ``H(k)`` over this many
#: seeded random k, floored by the model's own largest hopping or on-site
#: magnitude (which it bounds, ``|t_ij(R)| ≤ max_k |H_ij(k)|``, so the floor only
#: ever catches a k-sample that undershot) -- all of it with the mean on-site
#: energy taken off the diagonal.
_MAX_H_NK, _MAX_H_SEED = 64, 3


def _max_hamiltonian(model, *, params=None) -> float:
    r"""``max|H|`` — the Hamiltonian's own magnitude, measured on the model.

    The declared symmetry tolerance is **absolute**, in the units of ``H``, so a
    *relative* precision only becomes one after being read against this number
    (``docs/guide/numerical.md#how-to-read-it``).  It is measured two ways and
    the larger is taken:
    the largest entry of ``H(k)`` on a fixed set of random k — the thing the
    covariance residual is a residual *of* — and the largest hopping or
    on-site magnitude in the model's own numbers.
    The second is a floor rather than a competitor: each hopping is a Fourier
    coefficient of ``H_ij(k)``, so ``|t_ij(R)| ≤ max_k |H_ij(k)|`` and the
    read-off can only win where the finite k-sample undershot.

    **Both with the energy zero taken off.**  A constant shift ``H → H + c·1``
    moves no covariance residual, ``U(H + c)U† − (H(gk) + c) = U H U† − H(gk)``,
    and no centre defect's either, since a centre carries no phase onto its own
    on-site term; it moves every entry on the diagonal.  So the magnitude is
    that of ``H − μ·1``, ``μ`` the mean on-site energy per state
    (:func:`.._bloch.onsite_mean`): subtracting a Fermi energy from a wannier90
    import moves no tolerance, and a model whose on-site energies average to
    zero reads the largest entry of ``H`` itself.
    """
    mu = _bloch.onsite_mean(model, params)
    site = _bloch.onsite(model, params)
    got = (float(np.abs(site - mu).max()) if site.ndim == 1
           else _bloch.magnitude(site, mu)) if site.size else 0.0
    amps = _bloch.hopping_arrays(model)[0]
    if amps.size:
        got = max(got, float(np.abs(amps).max()))
    dim_k = int(model.lattice.dim_k)
    if dim_k:
        k = np.random.default_rng(_MAX_H_SEED).uniform(-0.5, 0.5,
                                                       size=(_MAX_H_NK, dim_k))
        got = max(got, _bloch.magnitude(_bloch.hamiltonian(model, k, params), mu))
    return float(got)
