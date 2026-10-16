r"""A model symmetric only to its declared precision, made exactly symmetric.

A Hamiltonian fitted to a calculation, such as a ``wannier90`` import, obeys its
group only to the precision of its own numbers.  :func:`symmetrize_positions`
moves its orbital centres onto the positions the group carries, a gauge
transformation that moves no band; :func:`symmetrize` projects its hoppings
onto their group-invariant part, which moves ``H`` by no more than the declared
tolerance; and :func:`classify_operations` says which operations are obeyed to
rounding and which only within that tolerance.
``realized_group(model, precision=...)`` runs both, the centres first, and
returns their records on ``.positions`` and ``.symmetrization``.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import _bloch
from ._tolerance import (_DROP_RELATIVE, _EXACT_CLOSURE, _IN_BLOCKS, _RANGE_RELATIVE,
                         EXACT_TOL, within)
from .refusal import InternalError, Refusal
from .symmetry import (SiteMultiplets, _GridHamiltonians, _budget_that_runs, _counted_as,
                       _declaration_grammar, _exact_gate, _index_ranges, _orbitals_named,
                       _resolve_spin, _size, bloch_operator, _build_rep, covariance_gate)

__all__ = ["Symmetrization", "symmetrize", "symmetrize_positions",
           "classify_operations", "EXACT_GATE"]


def _default_of(func, name):
    """A default read off a function's own signature, never retyped here."""
    return inspect.signature(func).parameters[name].default


#: The least covariance an operation obeyed exactly is held to, ``5e-14`` in the
#: units of ``H``.  A model's own boundary between exact and near is
#: :func:`..symmetry.covariance_gate` of its representation, this or the
#: rounding its arithmetic carries where that is larger, and no declared ``tol``
#: moves it.
EXACT_GATE = float(EXACT_TOL)

#: ``_build_rep``'s own internal-factor gate, the floor of the provisional pass's
#: loosened one.
_INTERNAL_GATE = float(_default_of(_build_rep, "internal_tol"))

#: The tolerance :func:`..symmetry.bloch_operator` checks the integer-shift
#: identity ``A·τ_a + t = τ_{g(a)} + s`` to, read off its own signature: the
#: statement that the operations carry the centres onto one another, which the
#: projection's operators are built on.
_BLOCH_TOL = float(_default_of(bloch_operator, "tol"))


# --------------------------------------------------------------------------
# the classification: exact / near / broken, per operation
# --------------------------------------------------------------------------

def classify_operations(validated, tol, *, exact_gate=EXACT_GATE, anchored=None) -> dict:
    """Each operation's class against a declared tolerance: ``"exact"``, ``"near"`` or ``"broken"``.

    Exact is obeyed to rounding, ``exact_gate``; near is obeyed within ``tol``
    and not to rounding, which licenses :func:`symmetrize`; broken is beyond
    ``tol``, and nothing is symmetrized into it.

    Parameters
    ----------
    validated : ..groups.ValidatedGroup
        The operations a call measured to hold, ``r.validated``, measured at
        ``tol``: ``realized_group(model, tol=..., symmetrize=False)``.
    tol : float
        The declared tolerance, in the units of ``H``: the one ``validated``
        was measured at.
    exact_gate : float, optional
        Rounding, in the units of ``H``: the covariance of the model's own
        representation, which :func:`..symmetry.covariance_gate` gives and
        :func:`symmetrize` passes.  Default :data:`EXACT_GATE`, ``5e-14``, the
        least it is.
    anchored : dict, optional
        ``{ind: residual}``, each operation's covariance in the representation
        the projection is built with, ``rep.residuals``.  An operation it holds
        is classified at the larger of that and its measured residual, which a
        basis turned inside a site, by less than ``tol``, makes near.

    Returns
    -------
    dict
        ``{ind: class}`` for each operation ``validated`` kept.

    Examples
    --------
    >>> classify_operations(r.validated, 1e-6)             # doctest: +SKIP
    {1: 'exact', 2: 'near', 5: 'near', ...}
    """
    tol = float(tol)
    out = {}
    for ind in validated.indices:
        r = _residual_read(validated, anchored, ind, missing=np.inf)
        out[int(ind)] = ("exact" if r <= exact_gate else
                         "near" if r <= tol else "broken")
    return out


def _named(sets) -> str:
    """Sets of orbitals in words: ``"orbitals 0-3"``, ``"orbitals 0-1 and 4-5"``."""
    sets = [tuple(int(o) for o in s) for s in sets]
    if len(sets) == 1:
        return _orbitals_named(sets[0])
    return "orbitals " + " and ".join(_index_ranges(s) for s in sets)


def _residual_read(validated, anchored, ind, *, missing):
    """The residual one operation is classified at: the detector's, or the larger of it and the anchored one."""
    r = float(validated.residuals.get(int(ind), missing))
    if anchored and int(ind) in anchored:
        r = max(r, float(anchored[int(ind)]))
    return r


# --------------------------------------------------------------------------
# the record
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Symmetrization:
    """How far a near-symmetric model was moved onto its symmetric part, and why.

    :func:`symmetrize` returns it beside the model it made, and
    :func:`..groups.realized_group` on ``r.symmetrization``, where ``r.model``
    is that model.  ``rec.summary()`` prints it.

    Attributes
    ----------
    defect, defect_relative : float
        How far the model moved, ``max_k |H_sym(k) − H(k)|`` in the units of
        ``H``, never more than ``tol``; and the same against ``scale``.
    scale : float
        ``max_k |H(k) − μ|``, ``μ`` the mean on-site energy per state: the size
        of ``H``, which a constant shift does not move.
    tol, exact_gate : float
        The tolerance on each operation's covariance residual, as the call was
        handed it or a declared precision sets it, ``10 × precision × max|H|``;
        and rounding.  Both are in the units of ``H``: an operation obeyed to
        ``exact_gate`` is exact, and one obeyed only within ``tol`` is near.
    classes, residuals : dict
        ``{ind: "exact" or "near"}`` for each operation, and the residual each
        was classified at (:func:`classify_operations`).
    multiplets : dict
        Where the multiplets each site's orbitals form hold only within the
        declared precision, as an orbital frame turned by less than it leaves
        them: ``mixing``, how far the Hamiltonian's own symmetry matrices mix
        them, relative to their size; ``exact``, the ``1e-8`` a model of exact
        numbers holds them to; ``declared``, the declared tolerance relative to
        ``max|H|``.  Empty otherwise.
    parts : dict
        Where parts of the crystal are coupled only below the declared
        precision, which the projection leaves out: ``orbitals``, each part's;
        ``coupling``, the largest entry of ``H(k)`` between two of them;
        ``exact``, the rounding it is above.  Empty otherwise.
    copies : dict
        Where identical copies of a part of the crystal are coupled, or told
        apart, only below the declared precision, which the projection makes
        exact copies: ``orbitals`` and the number of ``copies`` of each set;
        ``off``, the largest entry of ``H(k)`` outside exact copies; ``exact``.
        Empty otherwise.
    shares : dict
        Where states that split multiplets between them, the bonding and
        antibonding states of two lattices joined on site or a square's
        ``(p_x, p_y)`` pair, are coupled only below the declared precision, which
        the projection leaves out: ``orbitals``, ``sectors``, ``coupling``,
        ``exact``.  Empty otherwise.
    operations : tuple of int
        The operations averaged over, by index.
    not_represented : tuple of int
        Operations measured to hold that the representation does not carry, a
        slab cell's operations on its vacuum, left out of the average.
    verification : dict
        The operations measured again on the model returned, at the tolerance a
        model of exact numbers is measured at: ``tol``, ``nk``, ``n_kept``,
        ``n_operations``, ``worst_residual``, ``broken``, ``near_kept`` and a
        ``note``.  Empty where ``verify=False``.
    checks : dict
        The hopping list returned, measured: ``k_identity``, how far it misses
        the average at random ``k``; ``outside_box``, its largest coefficient
        beyond ``grid["R_box"]``, relative to the largest hopping, against
        ``range_gate``; ``covariance_residue``, its covariance against the
        matrices the average was taken with; and ``hermiticity``,
        ``drop_threshold``, ``drop_gap``, ``n_dropped``, ``n_hoppings`` and
        ``per_pass``.
    grid : dict
        The ``k`` grid the average was taken on: ``n``, its divisions along each
        periodic direction; ``n_points``; ``R_max``, the hoppings' own range;
        ``R_box``, the range the group carries them to; ``dim_k``.
    operators : dict
        The representation the average was taken with: ``tol``, the tolerance
        it was built at; ``covariance``, the largest residual its matrices
        reach; ``composition``, how far the model's own blocks were from
        multiplying as the group does; ``branch_ambiguity``, ``None`` unless the
        matrices built from each operation and from its inverse both held and
        were the same representation, then their largest difference;
        ``position_defect``, how far the orbital centres miss the group's own
        action, a floor no projection of the hoppings goes below; and
        ``bloch_tol``, the most that defect may be.
    passes, machine_floor : int, float
        How many averaging passes ran, and the rounding the passes stop at: one
        is exact, and another runs only while the covariance left is above
        ``machine_floor`` and ``H`` still moves.
    """

    tol: float
    exact_gate: float
    classes: dict
    residuals: dict
    defect: float
    defect_relative: float
    scale: float
    operations: tuple
    not_represented: tuple = ()
    grid: dict = field(default_factory=dict)
    checks: dict = field(default_factory=dict)
    operators: dict = field(default_factory=dict)
    verification: dict = field(default_factory=dict)
    passes: int = 1
    machine_floor: float = 0.0
    multiplets: dict = field(default_factory=dict)
    parts: dict = field(default_factory=dict)
    copies: dict = field(default_factory=dict)
    shares: dict = field(default_factory=dict)
    # whether the states carry a spin, so that a count of operations says it
    # counts the single-valued group's
    _spinful: bool = field(default=False, repr=False, compare=False)

    @property
    def near(self) -> tuple:
        """The operations obeyed within ``tol`` and not to rounding, by index."""
        return tuple(sorted(i for i, c in self.classes.items() if c == "near"))

    @property
    def exact(self) -> tuple:
        """The operations obeyed to rounding, by index."""
        return tuple(sorted(i for i, c in self.classes.items() if c == "exact"))

    def summary(self) -> str:
        """How far H moved and why, and what was measured on the model returned, a line each."""
        lines = [f"H moved by {self.defect:.3g}, {self.defect_relative:.3g} of its size"
                 + (f", in {self.passes} passes" if self.passes > 1 else "") + "   (.defect)",
                 "  the tolerance on each operation's residual, how far its matrix misses "
                 f"carrying H(k) onto H at the image of k: {self.tol:.3g}, in the units of H"
                 "   (.tol)"]
        if self.near:
            lines.append(f"  {len(self.near)} of the {len(self.classes)} operations were "
                         "near-symmetric, obeyed within that tolerance and not to rounding, "
                         f"{self.exact_gate:.2g}: the worst at "
                         f"{max(self.residuals[i] for i in self.near):.3g}   (.near, .residuals)")
        m, p = self.multiplets or {}, self.parts or {}
        c, sh = self.copies or {}, self.shares or {}
        if m:
            lines.append("  the multiplets hold only within the declared precision: the "
                         f"Hamiltonian's own symmetry matrices mix them at {m['mixing']:.3g} "
                         f"of their size, and the projection made them exact   (.multiplets)")
        if p:
            lines.append(f"  {len(p['orbitals'])} parts of the crystal are coupled only below "
                         f"the declared precision, at most {p['coupling']:.3g}, and that "
                         "coupling was left out   (.parts)")
        if c:
            lines.append(f"  the identical copies on {_named(c['orbitals'])} are copies only "
                         f"within the declared precision, {c['off']:.3g} from exact, and were "
                         "made exact copies   (.copies)")
        if sh:
            lines.append(f"  the states splitting the multiplets on {_named(sh['orbitals'])} "
                         "are coupled only below the declared precision, at most "
                         f"{sh['coupling']:.3g}, and that coupling was left out   (.shares)")
        if not (self.near or m or p or c or sh):
            lines.append(f"  every operation exact to rounding, {self.exact_gate:.2g}, and "
                         "nothing to correct   (.exact)")
        moved = float((self.operators or {}).get("position_defect", 0.0))
        if moved > EXACT_GATE:
            lines.append(f"  the orbital centres miss the group's own positions by {moved:.3g}, "
                         "which no projection of the hoppings removes   (.operators)")
        lines.append("  the hopping list returned is the group average to "
                     f"{self.checks.get('k_identity', float('nan')):.3g} at random k   (.checks)")
        v = self.verification or {}
        lines.append(f"  measured again, at a tolerance of {v['tol']:.3g}: {v['n_kept']} of "
                     f"{v['n_operations']} operations obeyed, the worst at "
                     f"{v['worst_residual']:.3g}   (.verification)" if v else
                     "  not measured again   (.verification)")
        if self.near or v:
            lines.append(f"  the operations counted are {_counted_as(self._spinful)}")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        mixed = (f", multiplets mixed at {self.multiplets['mixing']:.3g}"
                 if self.multiplets else "")
        mixed += (f", parts coupled at {self.parts['coupling']:.3g}" if self.parts else "")
        mixed += (f", copies {self.copies['off']:.3g} from exact" if self.copies else "")
        mixed += (f", sectors coupled at {self.shares['coupling']:.3g}" if self.shares else "")
        return (f"Symmetrization(defect {self.defect:.3g} "
                f"({self.defect_relative:.3g} relative) against tol {self.tol:.3g}{mixed}, "
                f"{self.passes} pass{'' if self.passes == 1 else 'es'}, "
                f"verified at {self.verification.get('worst_residual', float('nan')):.3g}; "
                f"{len(self.near)} of {len(self.classes)} operations near, "
                f"{_counted_as(self._spinful)})")


# --------------------------------------------------------------------------
# the transform grid: the hopping range, and the box the group action reaches
# --------------------------------------------------------------------------

def _range_box(model, syms, per):
    r"""``(R_max, R_box)`` — the model's own hopping range, and the group's reach.

    ``R_max`` is read off the hopping list.  ``R_box`` bounds where the
    *symmetrized* model can have hoppings: a term at displacement
    ``d = R + τ_j − τ_i`` is carried by ``g = {A|t}`` to ``A d``, and its
    lattice index is that minus the image orbitals' own positions, so

    .. math:: |R'_a| \le \max_{d,g} |(A d)_a| + \max_{ij}|τ_{i,a} − τ_{j,a}|.

    The bound is taken over the **actual** displacements rather than by summing
    ``|A|`` over the range, which for a rotation mixing directions is far
    looser; and it is a bound, not a derivation of the action — what the
    symmetrized model does outside it is *measured* (``checks["outside_box"]``).
    """
    dim_r = int(model.lattice.dim_r)
    orb = np.asarray(model.orb_vecs, dtype=float)
    _amps, i_idx, j_idx, R = _bloch.hopping_arrays(model)
    R = np.asarray(R, dtype=float)
    if R.shape[0]:
        R_max = np.abs(R).max(axis=0)
        d = R + orb[j_idx] - orb[i_idx]
    else:
        R_max, d = np.zeros(dim_r), np.zeros((1, dim_r))
    spread = (orb.max(axis=0) - orb.min(axis=0)) if orb.size else np.zeros(dim_r)
    worst = np.abs(d).max(axis=0)
    for s in syms:
        A = np.asarray(s.rotation, dtype=float)[:dim_r, :dim_r]
        worst = np.maximum(worst, np.abs(d @ A.T).max(axis=0))
    R_box = np.ceil(worst + spread - 1e-9).astype(int)
    return R_max[list(per)].astype(int), np.maximum(R_box[list(per)], 0)


def _divisions(group, syms, R_box):
    r"""``(box, divisions)`` — the reach each periodic direction is read to, and the grid's divisions along it.

    A Γ-centred grid is carried onto itself by an operation whose k-action is an
    integer matrix wherever any two directions that action mixes are divided
    alike, and only then.  So the directions fall into the sets the operations'
    k-actions connect, and each is read to the widest reach in its set: ``box``
    is that reach along each direction, and ``divisions = 2·(box + 1) + 1``,
    twice it and a shell more, the shell where an aliased transform would show.
    A group whose operations mix every direction, a cubic one, has one count,
    the widest reach's; a tetragonal one divides its fourfold axis apart from
    the plane.
    """
    R_box = np.asarray(R_box, dtype=int)
    dim_k = int(R_box.size)
    lead = list(range(dim_k))

    def root(a):
        while lead[a] != a:
            a = lead[a]
        return a

    probe = np.eye(dim_k)
    for s in syms:
        image = np.asarray(group.k_action_of(s)(probe), dtype=float)   # rows: images of e_b
        for b in range(dim_k):
            for a in range(dim_k):
                if a != b and abs(image[b, a]) > 0.5:
                    lead[root(a)] = root(b)
    box = np.array([max(int(R_box[b]) for b in range(dim_k) if root(b) == root(a))
                    for a in range(dim_k)], dtype=int)
    return box, 2 * (box + 1) + 1


def _grid(divisions):
    """The Γ-centred grid with ``divisions`` along the periodic directions, as integer multiples and reduced k."""
    counts = [int(c) for c in divisions]
    m = np.stack(np.meshgrid(*[np.arange(c) for c in counts], indexing="ij"),
                 axis=-1).reshape(-1, len(counts))
    return m, m / np.asarray(counts, dtype=float)


# --------------------------------------------------------------------------
# the memory the projection may take
# --------------------------------------------------------------------------

#: What the hopping projection holds at once, as a multiple of one grid of
#: Hamiltonians, ``16 · n_points · n_states²`` bytes.  The Hamiltonian and its
#: group average are held together while the average is taken; the
#: Hamiltonian is let go before the transform, which runs in place on the
#: average; the grid is let go before the hopping list is written; and
#: everything else -- each operation's Bloch operators and products, the
#: Fourier phases, the block norms -- is streamed over slices of the grid in
#: the working space the budget leaves (:data:`_POINT_MATRICES`).  Measured with
#: ``tracemalloc`` over a whole call at the least budget that runs it, on
#: models with dense hoppings in the shape of a wannier90 import, where the
#: grid dominates: 2.02 grids on 96 spinless states and on 48 spinful orbitals
#: (a ``9³`` grid, one grid 107.5 MB), 2.03 on 48 states (``9³``) and on 32
#: states reaching two cells (``11³``), against 8.0 grids before the streaming.
#: The symmetrized model the pass writes is about the size of the model handed
#: in, which is not in the count: writing it reached 0.2 to 1.55 grids on these.
_PROJECTION_GRIDS = 2.1

#: The working space one k-point of that streamed arithmetic takes, as a count
#: of ``n_states × n_states`` complex matrices: an operation's Bloch operator
#: and its adjoint, the two products and the slice of the average they are
#: added to.  Measured with ``tracemalloc`` on the same models as
#: :data:`_PROJECTION_GRIDS` at one and a half times the least budget, slices
#: of 129 to 235 points: 5.95 matrices a point on every one.
_POINT_MATRICES = 6.0


def _memory_budget(max_bytes) -> float:
    """``max_bytes`` read as a number of bytes; anything else is a malformed call.

    Read before anything is measured -- by :func:`symmetrize` beside ``tol``,
    and by :func:`..groups.realized_group` before the cell is identified -- so
    that a budget that is not a number of bytes never reaches a verdict.
    """
    try:
        budget = float(max_bytes)
    except (TypeError, ValueError):
        budget = float("nan")
    if not (budget > 0.0 and np.isfinite(budget)):
        raise ValueError(
            f"max_bytes must be a positive, finite number of bytes, got {max_bytes!r}. It "
            "is the memory the hopping projection may take at its peak -- the Hamiltonian "
            "and its group average on the transform grid, held together, and the "
            "arithmetic streamed beside them -- and the most that measuring the symmetry "
            "matrices may take at one k-point on a spectrum with exactly degenerate "
            "levels.")
    return budget


def _search_ceiling(max_candidates) -> int:
    """``max_candidates`` read as a whole number of candidates; anything else is a malformed call.

    Read before anything is measured -- by :func:`symmetrize` and by
    :func:`..groups.realized_group` -- so that a ceiling that is not a count
    never reaches a search.
    """
    if (isinstance(max_candidates, (bool, np.bool_))
            or not isinstance(max_candidates, (int, np.integer)) or int(max_candidates) < 1):
        raise ValueError(
            f"max_candidates must be a positive whole number, got {max_candidates!r}. It "
            "is the largest search for a representation's discrete choices, where the "
            "measurement does not fix them, that is run rather than refused "
            "`rep_search_too_large`.")
    return int(max_candidates)


def _slice_points(work, point_bytes, n_points) -> int:
    """How many grid points one slice of the streamed arithmetic takes within ``work`` bytes."""
    return int(min(int(n_points), max(1, int(float(work) // float(point_bytes)))))


def _projection_too_large(n_bytes, budget, *, divisions, least, n_states, R_max, R_box):
    r"""The projection whose transform grid the memory budget does not hold.

    A limit on memory, not a statement about the model: the grid is set by the
    model's own hopping range and a coarser one would alias the transform back
    to a hopping list, so the projection is refused rather than run on one, with
    the least peak it runs at and the budget that runs it.
    """
    runs = _budget_that_runs(n_bytes)
    counts = [int(c) for c in divisions]
    n_points = int(np.prod(counts))
    return Refusal(
        "projection_too_large",
        f"the hopping projection needs {_size(n_bytes)} at its peak against max_bytes = "
        f"{_size(budget)}, to hold the Hamiltonian and its group average together on a "
        f"transform grid of {' x '.join(str(c) for c in counts)} = {n_points} points at "
        f"{int(n_states)} states, with the rest of its arithmetic streamed beside them. "
        "The hoppings, carried by the group, reach "
        f"R_box = {R_box.tolist()} cells along the periodic directions, and a grid coarser "
        f"than {[int(c) for c in least]} divisions along them -- twice the reach and a "
        "shell more, alike along directions the group's operations mix -- would alias the "
        "transform back to a hopping list, so the projection is not run on one. This is a "
        "limit on memory, not a statement about the model: "
        f"`realized_group(..., max_bytes={runs})` runs it, as does "
        f"`symmetrize(..., max_bytes={runs})` by hand.",
        value=float(n_bytes), threshold=float(budget),
        context={"n_bytes": float(n_bytes), "max_bytes": float(budget),
                 "k_grid": tuple(counts), "n_points": n_points,
                 "n_states": int(n_states), "R_max": R_max.tolist(),
                 "R_box": R_box.tolist()})


# --------------------------------------------------------------------------
# the operators
# --------------------------------------------------------------------------

def _provisional_rep(model, group, assignment, validated, *, tol, scale, params,
                     build_kw):
    r"""The exact ``D(g)``, from a provisional :func:`..symmetry._build_rep`.

    Two tolerances move for this pass and nothing else does.

    * **The covariance gate** moves to the declared ``tol``.  It must:
      on a near-symmetric model *no* candidate closes the exact gate, and the
      candidate this pass wants is precisely the one whose covariance is the
      model's own defect.
    * **The read-off's internal gate** moves to the declared tolerance,
      expressed relative to ``max|H|`` -- taken about the mean on-site energy,
      which no residual sees -- or to ten times the worst residual so
      expressed where that is larger.  The read-off measures the discrete
      signs of the tensor structure on the detector's solution space, and on a
      near-symmetric operation that space is only an ``O(r)`` intertwiner, so
      its blocks miss ``χ(g)·S(g)`` by ``O(r/|H|)`` — measured 3.0e-7 on the
      1e-7-jittered lin3g — and the read would defer to an enumeration that is
      exponential in the number of multiplet orbits.  A multiplet wider than
      one orbital carries the model's own block for every operation, and the
      gate then compares those noisy blocks with the products of the
      generators' along the group's words: they agree to the model's noise
      carried through a word, which the declared precision bounds and the
      residual alone does not -- measured, on a spinor pair read spinless at
      ``6c`` of ``R-3m`` jittered at ``1e-5``, ``1.6e-4`` to ``1.7e-4``
      against ``10·r/max|H|`` of ``1.4e-4`` to ``1.5e-4``.  So the *read* is
      allowed to be as accurate as the model was declared to be, and no
      further.

    Neither loosening touches the candidate the pass returns.  ``D`` is
    assembled from **discrete** signs, snapped to ``phases``, through the same
    ``_assemble``/``_propagate`` path as any build, so where every multiplet is
    one-dimensional it carries none of the model's noise: measured identical to
    ``0.0`` against the *clean* model's representation on the jittered lin3g.
    A wider multiplet's block is the model's own and carries its noise, and
    the build then returns the exact representation nearest those blocks
    (``composition`` below says how far they were).  The internal-factor and
    product-table gates read machine zero on what comes back, which is why only
    the covariance gate had to move.

    **The orientation branches, when both survive.**  A group whose generators
    are all involutions with Hermitian representatives — inversion and a
    two-fold axis, which is most of what a real crystal offers — builds the
    *same* unitary matrices on both orientation branches, differing only by a
    phase per generator.  :func:`..symmetry._build_rep` settles that tie itself,
    by convention and with ``_branch_tie`` on the record, so the ordinary route
    through here never has a branch to decide and ``branch_ambiguity`` is
    ``None``.  What can still arrive is the tie the search never *asked* —
    ``rep_search_too_large``, the read-off having deferred for some other reason
    and the enumeration being too big to run, which is the wannier90-scale case
    — and ``rep_not_unique`` where the two survivors are genuinely different
    physics.  For *this* pass that ambiguity is decidable by measurement rather
    than by convention, because ``U H U†`` is blind to ``D → λ D``: the two
    branches are built separately and the phase-fitted difference
    ``min_λ ‖D₂(g) − λ D₁(g)‖`` is measured on
    every operation.  At machine level they are the same representation as far
    as any average can tell, the ``g`` branch is taken, and the number is on the
    record (``branch_ambiguity``).  Above it the original refusal is returned
    unchanged — a genuine second representation *would* give a different
    projection, and nothing here is in a position to choose between them.
    """
    kept = [int(i) for i in validated.indices]
    worst = max((float(validated.residuals.get(i, 0.0)) for i in kept), default=0.0)
    internal = max(_INTERNAL_GATE, 10.0 * worst / max(scale, 1e-30),
                   float(tol) / max(scale, 1e-30))
    kw = dict(build_kw)
    kw.setdefault("internal_tol", internal)
    kw.setdefault("_returned", False)
    rep = _build_rep(model, group, assignment, validated=validated, tol=float(tol),
                     params=params, **kw)
    ambiguity = None
    if (not rep and getattr(rep, "gate", None) in ("rep_not_unique",
                                                   "rep_search_too_large")
            and "orientation_branch" not in kw):
        first = rep
        branches = {}
        for branch in (1, -1):
            got = _build_rep(model, group, assignment, validated=validated,
                             tol=float(tol), orientation_branch=branch, params=params, **kw)
            if not got:
                return first, {}
            branches[branch] = got
        shared = [i for i in branches[1].indices if i in branches[-1].matrices]
        ambiguity = 0.0
        for ind in shared:
            a = np.asarray(branches[1].matrices[ind])
            b = np.asarray(branches[-1].matrices[ind])
            ov = complex(np.vdot(a, b))
            lam = ov / abs(ov) if abs(ov) > 1e-12 else 1.0
            ambiguity = max(ambiguity, float(np.abs(b - lam * a).max()))
        if ambiguity > _INTERNAL_GATE:
            return first, {}
        rep = branches[1]
    if not rep:
        return rep, {}
    return rep, {"tol": float(tol),
                 "composition": float(rep._search.get("composition", 0.0)),
                 "branch_ambiguity": ambiguity,
                 "covariance": float(max(rep.residuals.values(), default=0.0))}


def _projection_operators(model, group, assignment, validated, *, tol, params, build_kw):
    r"""``(rep, record)`` -- the representation a projection of this model is built with.

    :func:`_provisional_rep` at the scale the projection reads, ``max|H|`` about
    the mean on-site energy on a fixed probe of eight k-points.  Whether an
    operation is near is read on its matrices' covariance as well as on the
    detector's residual (:func:`classify_operations`), and the average is
    taken with them (:func:`symmetrize`).
    """
    mu = _bloch.onsite_mean(model, params)
    probe = np.random.default_rng(1729).uniform(-0.5, 0.5, size=(8, int(group.dim_k)))
    scale = _bloch.magnitude(_bloch.hamiltonian(model, probe, params), mu)
    return _provisional_rep(model, group, assignment, validated, tol=tol, scale=scale,
                            params=params, build_kw=build_kw)


def _position_defect(model, group, assignment, ops):
    r"""How far this model's own positions are from the group's action on them.

    :func:`..symmetry.bloch_operator` *checks* the integer-shift identity
    ``A·τ_a + t = τ_{g(a)} + s`` rather than assuming it, at a tolerance whose
    default (``1e-9``) is a statement about a model built from exact numbers. A
    **wannier90** model's centres are numerically determined and satisfy it only
    to the Wannierization's own precision — measured **7.7e-9** on a real
    import.  This measures how far they miss it: the largest residual of the
    identity over the operations ``ops`` and every orbital, each component
    modulo the lattice.
    """
    from .symmetry import _orb_positions_3d

    pos = _orb_positions_3d(model, group)
    site_of = np.asarray(assignment.site_of_orbital, dtype=int)
    perm = np.asarray(assignment.permutations, dtype=int)
    shift = np.asarray(assignment.lattice_shift, dtype=float)
    first = {}
    for a in range(int(model.norb)):
        first.setdefault(int(site_of[a]), a)
    pos_of_ind = {int(s.ind): i for i, s in enumerate(group.symmetries)}
    worst = 0.0
    for ind, sym in ops.items():
        p = pos_of_ind[int(ind)]
        A = np.asarray(sym.rotation, dtype=float)
        t = np.asarray(sym.translation, dtype=float)
        for a in range(int(model.norb)):
            s = int(site_of[a])
            b = first[int(perm[p, s])]
            resid = A @ pos[a] + t - shift[p, s] - pos[b]
            worst = max(worst, float(np.abs(resid - np.rint(resid)).max()))
    return worst


def _with_centres(model, orb, *, check=True, n_check=_bloch.CHECK_K, seed=_bloch.CHECK_SEED,
                  params=None):
    r"""The model with its orbital centres at ``orb``, its spectrum measured unchanged.

    The same lattice, on-site terms and hopping table, row for row, with the
    centres replaced: in convention I that is the k-dependent gauge
    transformation ``Λ(k) = diag(e^{−2πi k·Δ})``, which cannot move an
    eigenvalue, and ``check`` measures that it did not, over ``n_check`` random
    ``k``.  Returns ``(model, residual)``, the residual ``0.0`` where nothing
    was measured; a spectrum that moved raises :class:`..refusal.InternalError`.
    """
    from pythtb import Lattice, TBModel

    orb = np.asarray(orb, dtype=float)
    out = TBModel(Lattice(np.asarray(model.lat_vecs, dtype=float), orb.tolist(),
                          periodic_dirs=list(model.periodic_dirs)),
                  spinful=(int(model.nspin) == 2))
    out.set_onsite(np.asarray(model.onsite), mode="set")
    # the model's own hopping table, row for row and in one pass, so the moved
    # model sums exactly the terms the model sums
    out._append_hops(*_bloch.hopping_arrays(model))

    resid = 0.0
    if check:
        k = np.random.default_rng(int(seed)).uniform(
            -0.5, 0.5, size=(int(n_check), int(model.lattice.dim_k)))
        Ea = np.linalg.eigvalsh(np.asarray(_bloch.hamiltonian(model, k, params)))
        Eb = np.linalg.eigvalsh(np.asarray(_bloch.hamiltonian(out, k, params)))
        scale = max(float(np.abs(Ea).max()), 1.0)
        resid = float(np.abs(Ea - Eb).max())
        if resid > 1e-9 * scale:
            worst = float(np.abs(orb - np.asarray(model.orb_vecs, dtype=float)).max(
                initial=0.0))
            raise InternalError(
                f"moving the centres, by at most {worst:.3g} (reduced), changed the "
                f"spectrum: max_k |E_new - E_old| = {resid:.3g} on a scale of "
                f"{scale:.3g}, over {n_check} random k. In convention I that move is the "
                "k-dependent gauge transformation Lambda(k) = diag(e^{-2 pi i k.Delta}), "
                "which cannot change an eigenvalue, so this is the package disagreeing "
                "with itself, not a property of the model.")
    return out, resid


# --------------------------------------------------------------------------
# the average, and the way back to a hopping list
# --------------------------------------------------------------------------

def _same_part(parts, model):
    r"""``[a, b]``, true where states ``a`` and ``b`` lie in one of ``parts``.

    ``parts`` holds each coupled part's orbitals, together every orbital of
    ``model``; ``None`` where there are fewer than two.
    """
    if not parts or len(parts) < 2:
        return None
    nspin = int(model.nspin)
    part_of = np.full(int(model.nstate), -1, dtype=int)
    for i, orbs in enumerate(parts):
        for o in orbs:
            part_of[int(o) * nspin:(int(o) + 1) * nspin] = i
    if (part_of < 0).any():
        raise InternalError(
            f"symmetrize: the coupled parts handed in leave states "
            f"{np.flatnonzero(part_of < 0).tolist()} in none of them; the parts of a crystal "
            "cover every orbital.")
    return part_of[:, None] == part_of[None, :]


def _parts_coupling(model, parts, params=None) -> float:
    r"""The largest entry of ``H(k)`` between two of the coupled parts ``parts`` holds.

    Read on the fixed probe of eight k the projection's scale is read on
    (:func:`_projection_operators`); ``0.0`` where there are fewer than two parts.
    """
    same = _same_part(parts, model)
    if same is None:
        return 0.0
    probe = np.random.default_rng(1729).uniform(-0.5, 0.5, size=(8, int(model.lattice.dim_k)))
    H = np.asarray(_bloch.hamiltonian(model, probe, params))
    return float(np.abs(H[:, ~same]).max(initial=0.0))


def _as_held(X, held):
    r"""``X`` with what the sectors read at the declared precision keep apart held apart.

    ``held`` is :func:`..symmetry._held_sets`'s: ``"copies"``, each set of
    identical copies' basis in the states, copy-major, with its number of
    copies, the states one copy holds and its orbitals; and ``"shares"``,
    each set of sectors splitting multiplets between orbits no operation
    exchanges, with each sector's basis.  What ``X`` holds between a set and
    anything else, between two copies or between two such sectors is left
    out, and each copy carries the copies' mean: the part of ``X`` that
    commutes with every matrix the sectors conserve.  Every operation carries
    a set's copies onto its copies and each such sector onto one of its set's,
    so this and the group average commute.
    """
    X = np.asarray(X, dtype=complex)
    copies = tuple(held.get("copies") or ())
    pieces = [[Q @ Q.conj().T for Q in bases.values()] for bases, _orbs in held.get("shares") or ()]
    P = (sum(Q @ Q.conj().T for Q, _m, _d, _orbs in copies)
         + sum(Pt for ps in pieces for Pt in ps))
    out = X - P @ X - X @ P + P @ X @ P
    for Q, m, d, _orbs in copies:
        Y = (Q.conj().T @ X @ Q).reshape(X.shape[:-2] + (m, d, m, d))
        mean = np.einsum("...aiaj->...ij", Y) / float(m)
        block = np.einsum("ab,...ij->...aibj", np.eye(m), mean).reshape(
            X.shape[:-2] + (m * d, m * d))
        out = out + Q @ block @ Q.conj().T
    for ps in pieces:
        for Pt in ps:
            out = out + Pt @ X @ Pt
    return out


def _held_off(model, held, params=None) -> dict:
    r"""``{"copies", "shares"}`` -- the largest entry of ``H(k)`` each kind ``held`` holds leaves out.

    Read on the probe :func:`_parts_coupling` reads, each kind on its own;
    ``0.0`` where there is none of it.
    """
    out = {"copies": 0.0, "shares": 0.0}
    if not held:
        return out
    probe = np.random.default_rng(1729).uniform(-0.5, 0.5, size=(8, int(model.lattice.dim_k)))
    H = np.asarray(_bloch.hamiltonian(model, probe, params))
    for kind in out:
        if held.get(kind):
            alone = {kind: held[kind]}
            out[kind] = float(np.abs(H - _as_held(H, alone)).max(initial=0.0))
    return out


def _average_on_grid(model, group, assignment, ops, D, kpts, divisions, params, *,
                     bloch_tol, work, apart=None, held=None):
    r"""``(H_sym, moved, scale)`` on the transform grid — the group average.

    For every operation and every grid point, ``U(g;k)`` is
    :func:`..symmetry.bloch_operator` at the **folded** image ``k_out ≡ gk``
    (mod a reciprocal-lattice vector), which is again a grid point because the
    grid is Γ-centred, divided alike along any two directions an operation's
    k-action mixes (:func:`_divisions`), and the k-action is an integer matrix
    there.  The contribution ``U H(k) U†`` —
    with ``H(k)*`` for an antiunitary operation, exactly as the covariance
    identity takes it — is accumulated at that image, so every accumulator slot
    receives one term per operation and the sum over the group is the average.

    Pushing forward rather than pulling back is what makes this cheap: ``H`` is
    evaluated **once** on the grid and every operation is a permutation of it.
    ``H`` and the accumulator are the two grids held here; ``H`` is evaluated,
    and each operation's operators and products are formed, over slices of the
    grid within ``work`` bytes, and a slice is the whole grid wherever that
    fits.  ``moved`` is ``max_k |H_sym(k) − H(k)|`` and ``scale`` is
    ``max_k |H(k)|`` over the grid, both read before ``H`` is let go.

    ``apart`` (:func:`_same_part`) is the coupled parts of the crystal the
    verdict read, where it read several: what ``H`` holds between two of them
    lies below the declared precision, where a coupling counts as absent, and
    the average leaves it out.  Every operation carries each part onto itself,
    so leaving it out and averaging commute exactly.  ``held`` is what the
    sectors read at the declared precision keep apart, held apart the same
    way (:func:`_as_held`).
    """
    n_points = int(kpts.shape[0])
    counts = tuple(int(c) for c in divisions)
    n = np.asarray(counts, dtype=float)
    H = _GridHamiltonians(model, kpts, params, max_bytes=work).full()
    acc = np.zeros_like(H)
    step = _slice_points(work, _POINT_MATRICES * 16.0 * float(H.shape[-1]) ** 2, n_points)
    for ind, sym in ops.items():
        kp = group.k_action_of(sym)(kpts)
        mm = np.rint(kp * n)
        if np.abs(kp * n - mm).max() > 1e-7:      # pragma: no cover - guarded by design
            raise ValueError(
                f"symmetrize: operation #{ind} does not carry the "
                f"{' x '.join(str(c) for c in counts)} grid onto itself, so the average "
                "would not be over one grid. This cannot happen with the divisions "
                "symmetrize derives, alike along every pair of directions an operation "
                "mixes, and says the grid was overridden with one the group does not "
                "preserve.")
        mm = mm.astype(int) % np.asarray(counts, dtype=int)
        idx = np.ravel_multi_index(tuple(mm.T), counts)
        k_out = mm / n
        for lo in range(0, n_points, step):
            hi = min(lo + step, n_points)
            U = bloch_operator(model, group, assignment, sym, D[ind], kpts[lo:hi],
                               k_out=k_out[lo:hi], tol=bloch_tol)
            src = np.conj(H[lo:hi]) if sym.time_reversal else H[lo:hi]
            # the k-action is a permutation of the grid, so no slot is written twice
            acc[idx[lo:hi]] += U @ src @ np.conj(np.swapaxes(U, -1, -2))
            del U, src
    acc /= float(len(ops))
    if apart is not None:
        acc[:, ~apart] = 0.0
    moved = scale = 0.0
    for lo in range(0, n_points, step):
        hi = min(lo + step, n_points)
        if held:
            acc[lo:hi] = _as_held(acc[lo:hi], held)
        moved = max(moved, float(np.abs(acc[lo:hi] - H[lo:hi]).max()))
        scale = max(scale, float(np.abs(H[lo:hi]).max()))
    return acc, moved, scale


def _average_at(model, group, assignment, ops, D, k, params, *, bloch_tol, apart=None,
                held=None):
    r"""``(H_sym, residue)`` at arbitrary k — the average **pulled back**.

    ``(g P H)(k) = U(g;k)† H(gk) U(g;k)`` (conjugated for an antiunitary
    operation) is the covariance identity solved for ``H(k)``, so it equals
    ``H(k)`` exactly when the operation holds and differs by the covariance
    residual when it does not.  Averaged over the group it is the same
    ``H_sym`` the grid push-forward gives — the two differ by relabelling
    ``g -> g^-1``, and the projective phase cancels in the conjugation — with no
    grid needed, which is what makes it the instrument for the off-grid identity
    check.

    ``residue`` is ``max_g max_k |U H(k) U† − H(gk)|``: the covariance of *this*
    model against *these* operators, which is what says whether another pass is
    needed.  ``apart`` leaves out what lies between the coupled parts, and
    ``held`` holds what the sectors keep apart apart, as
    :func:`_average_on_grid` does.
    """
    k = np.atleast_2d(np.asarray(k, dtype=float))
    evaluate = _bloch.evaluator(model, params)
    H = np.asarray(evaluate(k))
    acc = np.zeros_like(H)
    residue = 0.0
    for ind, sym in ops.items():
        kout = group.k_action_of(sym)(k)
        Hrot = np.asarray(evaluate(kout))
        U = bloch_operator(model, group, assignment, sym, D[ind], k, k_out=kout,
                           tol=bloch_tol)
        Ud = np.conj(np.swapaxes(U, -1, -2))
        src = np.conj(H) if sym.time_reversal else H
        residue = max(residue, float(np.abs(U @ src @ Ud - Hrot).max()))
        pull = Ud @ Hrot @ U
        acc += np.conj(pull) if sym.time_reversal else pull
    acc /= float(len(ops))
    if apart is not None:
        acc[:, ~apart] = 0.0
    if held:
        acc = _as_held(acc, held)
    return acc, residue


def _inverse_transform(Hsym, model, kpts, divisions, per, *, work):
    r"""``(t, R)`` — the hopping coefficients, exactly, in place of ``H_sym``.

    In pythtb's convention I ``H_ij(k) = Σ_R t_ij(R) e^{2πi k·(R + τ_j − τ_i)}``
    is a *finite* Fourier series, so on a Γ-centred grid of ``N = Π_a n_a``
    points, ``n_a > 2·R_max,a`` divisions along each periodic direction, the
    inverse

    .. math:: t_{ij}(R) = \frac{1}{N}\sum_k H_{ij}(k)\,
              e^{-2\pi i\,k\cdot(R + \tau_j - \tau_i)}

    is exact — no aliasing, no window, no fit.  ``τ`` is the model's **own**
    ``orb_vecs``, not the slab cell's wrapped copy, because it is the model's
    own that pythtb built ``H(k)`` with.

    The transform overwrites ``H_sym``: the phases multiply it in place, over
    slices of the grid within ``work`` bytes, and the Fourier transform writes
    its output over its input, so the coefficients take no grid beside it.

    Returns ``t`` of shape ``(N, nstate, nstate)``, sharing ``H_sym``'s
    memory, and the folded integer vectors ``R`` of shape ``(N, d)``, each
    component in ``[-(n_a-1)/2, (n_a-1)/2]``.
    """
    counts = tuple(int(c) for c in divisions)
    dim_k = len(counts)
    n_points = int(kpts.shape[0])
    norb, nspin = int(model.norb), int(model.nspin)
    Hsym = np.ascontiguousarray(Hsym)        # the grid is transformed where it lies
    tau = np.asarray(model.orb_vecs, dtype=float)[:, list(per)]
    delta = tau[None, :, :] - tau[:, None, :]                    # [i, j] = tau_j - tau_i
    delta = delta.reshape(-1, dim_k).T
    step = _slice_points(work, _POINT_MATRICES * 16.0 * float(Hsym.shape[-1]) ** 2,
                         n_points)
    for lo in range(0, n_points, step):
        hi = min(lo + step, n_points)
        phase = np.exp(-2j * np.pi * (kpts[lo:hi] @ delta)).reshape(-1, norb, norb)
        if nspin == 2:
            phase = np.repeat(np.repeat(phase, 2, axis=1), 2, axis=2)
        Hsym[lo:hi] *= phase
        del phase
    f = Hsym.reshape(counts + Hsym.shape[-2:])
    np.fft.fftn(f, axes=tuple(range(dim_k)), out=f)
    f /= float(np.prod(counts))
    labs = [np.where(np.arange(c) > (c - 1) // 2, np.arange(c) - c, np.arange(c))
            for c in counts]
    R = np.stack(np.meshgrid(*labs, indexing="ij"), axis=-1).reshape(-1, dim_k)
    return Hsym, R


def _blocks(t, norb, nspin):
    """``t`` reshaped to ``(R, i, j, spin, spin)`` — the spin block is the unit."""
    return np.transpose(t.reshape(t.shape[0], norb, nspin, norb, nspin), (0, 1, 3, 2, 4))


def _block_norms(t, norb, nspin, step):
    """``‖t_ij(R)‖_F`` per ``(R, i, j)`` — the spin block is the unit, not the entry.

    Taken ``step`` lattice vectors at a time, so the arithmetic holds one slice
    of the coefficients beside them rather than two copies of all of them.
    """
    out = np.empty((t.shape[0], int(norb), int(norb)))
    for lo in range(0, t.shape[0], int(step)):
        hi = min(lo + int(step), t.shape[0])
        out[lo:hi] = np.linalg.norm(_blocks(t[lo:hi], norb, nspin), axis=(3, 4))
    return out


#: The half-decades :func:`_drop_threshold` sorts the norms into, and the index
#: of the one starting at 1: ``floor(2·log10(v))`` of a positive double lies in
#: ``[-647, 617]``.
_HALF_DECADE_ZERO = 650
_HALF_DECADES = 1300


def _drop_threshold(norms, relative=_DROP_RELATIVE, *, step=None):
    r"""A drop threshold that never splits a symmetry orbit.

    Dropping rebuilt coefficients by amplitude alone can break a symmetry: a
    truncation **breaks** it whenever two partners of one orbit straddle the
    threshold.  Partners of an orbit carry the same block norm — that is what
    makes them partners — so an orbit-safe threshold is one with a *gap* under
    it.  This walks the block norms upward from zero, never above
    ``relative·max‖t‖``, and cuts at the last one followed by a factor-of-ten
    jump; where no such gap exists it drops **nothing**, because keeping a
    rounding-level coefficient costs one hopping term and dropping half an
    orbit costs the symmetry.

    Returns ``(threshold, gap)`` — the gap being the measured ratio across the
    cut, and ``inf`` when nothing was dropped.

    The walk is read off the least and the greatest norm in each half-decade of
    magnitude, streamed ``step`` lattice vectors at a time, so no copy of the
    norms is made: two norms in one half-decade are less than a factor of ten
    apart, so a jump can only open between the greatest norm of one half-decade
    and the least of the next one occupied, and the cut and the gap are the
    ones the walk over every norm finds.
    """
    top = float(norms.max(initial=0.0))
    if not top > 0.0:
        return 0.0, float("inf")
    bound = float(relative) * top
    step = int(norms.shape[0]) if step is None else int(step)
    least = np.full(_HALF_DECADES, np.inf)
    greatest = np.zeros(_HALF_DECADES)
    edge = float("inf")
    for lo in range(0, int(norms.shape[0]), step):
        part = norms[lo:lo + step]
        under = part[(part > 0.0) & (part <= bound)]
        if under.size:
            where = np.floor(2.0 * np.log10(under)).astype(int) + _HALF_DECADE_ZERO
            np.minimum.at(least, where, under)
            np.maximum.at(greatest, where, under)
        edge = min(edge, float(part.min(where=part > bound, initial=np.inf)))
    occupied = np.flatnonzero(greatest > 0.0)
    if occupied.size == 0:
        return 0.0, float("inf")
    nxt = edge
    for b in occupied[::-1]:
        if nxt > 10.0 * float(greatest[b]):
            return float(greatest[b]) * (1.0 + 1e-12), float(nxt / float(greatest[b]))
        nxt = float(least[b])
    return 0.0, float(edge / float(least[occupied[0]]))


def _kept_terms(t, R, model, per, keep, step):
    """The terms of ``t(R)`` a pythtb model is written from, read off the grid.

    Only the canonical half of each conjugate pair is kept — pythtb adds the
    other — with the ``R = 0`` diagonal as the on-site terms.  The
    anti-Hermitian part dropped in doing so is *measured*, not assumed away,
    ``step`` lattice vectors at a time.  ``keep`` says which ``(R, i, j)``
    blocks survive :func:`_drop_threshold`'s cut.  Every array returned is a
    copy, so the grid can be let go before the model is written.

    Returns ``(onsite, amplitudes, i, j, R, hermiticity)``.
    """
    dim_r = int(model.lattice.dim_r)
    norb, nspin = int(model.norb), int(model.nspin)
    blocks = _blocks(t, norb, nspin)
    order = {tuple(int(x) for x in r): i for i, r in enumerate(R)}
    mirror = np.array([order[tuple(-int(x) for x in r)] for r in R])
    herm = 0.0
    for lo in range(0, blocks.shape[0], int(step)):
        hi = min(lo + int(step), blocks.shape[0])
        herm = max(herm, float(np.abs(blocks[lo:hi] - np.conj(np.swapaxes(
            np.swapaxes(blocks[mirror[lo:hi]], 1, 2), 3, 4))).max()))

    zero = int(np.flatnonzero(np.abs(R).sum(axis=1) == 0)[0])
    if nspin == 2:
        ons = np.asarray([0.5 * (blocks[zero, i, i] + np.conj(blocks[zero, i, i].T))
                          for i in range(norb)])
    else:
        ons = [float(np.real(blocks[zero, i, i][0, 0])) for i in range(norb)]

    # The kept coefficients, in one pass and in (R, i, j) order.  Of each
    # conjugate pair only the half whose R has its first nonzero component
    # positive is written, the upper triangle at R = 0 -- pythtb adds the other
    # half -- and every (R, i, j) is one coefficient of the transform, so no term
    # is written twice.
    Rint = np.asarray(R, dtype=int)
    nonzero = Rint != 0
    at_zero = ~nonzero.any(axis=1)
    lead = Rint[np.arange(Rint.shape[0]), np.argmax(nonzero, axis=1)]
    written = keep & (at_zero | (lead > 0))[:, None, None]
    written[at_zero] &= np.triu(np.ones((norb, norb), dtype=bool), k=1)
    rr, ii, jj = np.nonzero(written)
    del written
    Rfull = np.zeros((rr.size, dim_r), dtype=int)
    Rfull[:, list(per)] = Rint[rr]
    amps = blocks[rr, ii, jj] if nspin == 2 else blocks[rr, ii, jj, 0, 0]
    return ons, amps, ii, jj, Rfull, herm


def _to_model(model, ons, amps, ii, jj, Rfull):
    """A pythtb model on ``model``'s own lattice, from the terms :func:`_kept_terms` read."""
    from pythtb import Lattice, TBModel

    out = TBModel(Lattice(np.asarray(model.lat_vecs, dtype=float),
                          np.asarray(model.orb_vecs, dtype=float).tolist(),
                          periodic_dirs=list(model.periodic_dirs)),
                  spinful=(int(model.nspin) == 2))
    out.set_onsite(ons, mode="set")
    out._append_hops(amps, ii, jj, Rfull)
    return out


# --------------------------------------------------------------------------
# the entry point
# --------------------------------------------------------------------------

def symmetrize(model, group, assignment, validated, *, tol, k_grid=None, params=None,
               max_passes=3, verify=True, verify_nk=None, verify_tol=None,
               max_bytes=1e9, characters=None, multiplets=None, spin=None,
               max_candidates=2 ** 20):
    r"""The model nearest a near-symmetric one that obeys its group exactly, and how far it moved.

    ``H`` is averaged over the group's operations, those combined with time
    reversal among them where the group has any, as ``U(g;k) H(k) U(g;k)†`` at
    every point of a grid, and comes back to a hopping list exactly.  The
    average moves ``H`` by about the residuals of the operations obeyed only
    within ``tol``, which the declaration already calls noise.
    :func:`..groups.realized_group` runs it itself where a declared precision
    calls for it.

    Parameters
    ----------
    model : pythtb.TBModel
        The model the operations were measured on, ``r.model``: for a layer,
        its slab presentation.
    group, assignment, validated : GroupData, OrbitalAssignment, ValidatedGroup
        ``r.group``, ``r.assignment`` and ``r.validated`` of a call measured at
        ``tol``, ``realized_group(model, tol=..., symmetrize=False)``.
    tol : float
        The declared tolerance, in the units of ``H``, the one ``validated`` was
        measured at; for numbers good to a relative ``ε``, about
        ``10·ε·max|H|``.  Required.
    characters, multiplets, spin : optional
        The declarations, as :func:`..groups.realized_group` takes them.  The
        average does not depend on ``characters=``, which the representation it
        is taken with carries.
    k_grid : int, optional
        Divisions of the grid in every periodic direction.  Default: the
        smallest grid on which the hopping list comes back exactly, set by the
        hoppings' range; a coarser one raises.
    max_passes : int, optional
        The most averaging passes.  Default 3: one is exact, and another runs
        only while ``H`` still moves.
    verify, verify_nk, verify_tol : optional
        Measure the operations again on the model returned (default ``True``),
        on a mesh of ``verify_nk`` divisions (default: the one ``validated``
        used), at ``verify_tol`` (default ``1e-8``).
    max_bytes : float, optional
        The memory the average may take at its peak, in bytes.  Default ``1e9``.
    max_candidates : int, optional
        The largest search for the representation's discrete choices that is
        run.  Default ``2**20``.
    params : dict, optional
        Values for a model whose terms are given by parameter names; the model
        returned holds the numbers.

    Returns
    -------
    (pythtb.TBModel, Symmetrization) or Refusal
        The symmetrized model, and the record of the average: ``rec.summary()``;
        ``rec.defect``, how far ``H`` moved, in its units; ``rec.near``, the
        operations obeyed only within ``tol``; ``rec.verification``, the
        operations measured again on the model returned.  A ``Refusal`` in its
        place is falsy and names what stopped the call.

    Raises
    ------
    ValueError
        A malformed argument, before anything is measured, or a ``k_grid`` too
        coarse for the hoppings' range.
    ..refusal.InternalError
        The hopping list returned misses the average, which it reproduces
        exactly: a defect to report.

    Refusals
    --------
    ``symmetrize_broken_operation``
        an operation beyond ``tol``, which the model does not obey.
    ``symmetrize_defect_exceeds_tol``
        the average moved ``H`` further than ``tol``.
    ``symmetrize_range_not_closed``
        the average reaches beyond the hoppings' range on the grid; a larger
        ``k_grid`` runs it.
    ``positions_not_exactly_symmetric``
        the orbital centres miss the operations; :func:`symmetrize_positions`,
        or a declared precision, moves them first.
    ``projection_too_large`` / ``detector_too_large``
        past ``max_bytes``; the reason gives the value that runs it.

    A refusal of the representation's own build, ``rep_not_unique`` among them,
    is returned as it is.

    Examples
    --------
    The average by hand, on a crystal whose operations were measured at the
    tolerance declared:

    >>> r = realized_group(model, tol=1e-5, symmetrize=False)       # doctest: +SKIP
    >>> sym, rec = symmetrize(r.model, r.group, r.assignment, r.validated,
    ...                       tol=1e-5)                              # doctest: +SKIP
    >>> rec.verification["broken"]                                  # doctest: +SKIP
    ()
    """
    return _symmetrized(model, group, assignment, validated, tol=tol, k_grid=k_grid,
                        params=params, max_passes=max_passes, verify=verify,
                        verify_nk=verify_nk, verify_tol=verify_tol, max_bytes=max_bytes,
                        characters=characters, multiplets=multiplets, spin=spin,
                        max_candidates=max_candidates)


def _symmetrized(model, group, assignment, validated, *, tol, k_grid=None, params=None,
                 max_passes=3, verify=True, verify_nk=None, verify_tol=None,
                 max_bytes=1e9, _parts=None, _held=None, **build_kw):
    """The pass :func:`symmetrize` runs, every keyword of its provisional build passed by name.

    :func:`..groups.realized_group` calls it on its own records, handing in the
    parts of the crystal and the identical copies it read apart (``_parts``,
    ``_held``) and whatever the build it will make is asked for.
    """
    from . import groups as _groups

    tol = float(tol)
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError(f"symmetrize: tol must be a positive number, got {tol!r}. It is "
                         "the declared tolerance the operations were measured at, in the "
                         "units of H.")
    budget = _memory_budget(max_bytes)
    if "max_candidates" in build_kw:
        build_kw["max_candidates"] = _search_ceiling(build_kw["max_candidates"])
    # what the provisional build is handed, read as grammar before anything is
    # measured: a malformed `characters=` or `multiplets=` raises here
    _declaration_grammar(int(model.norb), characters=build_kw.get("characters"),
                         multiplets=build_kw.get("multiplets"),
                         spin=_resolve_spin(build_kw.get("spin"), model),
                         site_of=assignment.site_of_orbital)
    params = dict(params or {})
    per = tuple(int(d) for d in group.periodic_dirs)
    dim_k = int(group.dim_k)
    syms_by_ind = {int(s.ind): s for s in group.symmetries}

    # ---- 1. the classification, and the one thing that is refused ----------
    # against the exact gate of THIS model and operation set, measured once: the
    # detection record carries neither the number of states nor max|H|
    exact_gate = _exact_gate(model, group, validated, build_kw, params)
    classes = classify_operations(validated, tol, exact_gate=exact_gate)
    broken = sorted(i for i, c in classes.items() if c == "broken")
    if broken:
        worst = max(float(validated.residuals.get(i, np.inf)) for i in broken)
        return Refusal(
            "symmetrize_broken_operation",
            f"{len(broken)} of the {len(classes)} operations handed in, "
            f"{_counted_as(group.spinor)}, are broken at the "
            f"declared tolerance: {broken} reach covariance residual {worst:.3g} against "
            f"tol = {tol:.3g}. Symmetrizing over them would impose a symmetry this "
            "Hamiltonian does not have -- the projection moves H by O(residual), which is "
            "declared noise only while the residual is inside the declared tolerance. "
            "Either the tolerance is too tight for a defect that is merely numerical "
            "(raise it, reading it against max|H|), or the operations are genuinely broken "
            "and the cell is the wrong one (`typat=`, `decorations=`).",
            value=float(worst), threshold=tol,
            context={"broken": tuple(broken), "classes": classes, "tol": tol,
                     "exact_gate": exact_gate})
    # ... and the centres, which the operators are built on.  The average is
    # taken with Bloch operators that carry each centre onto its image, checked
    # to `bloch_operator`'s own tolerance; on centres the operations miss, the
    # operators do not compose as the group does, and the "average" is no
    # projection: it writes each hopping onto a lattice vector that is not one
    # and moves what was exact.  A projection of the hoppings cannot reach a
    # defect in the centres at all, so such centres are refused here, with the
    # route that moves them.
    pos_defect = float(_position_defect(model, group, assignment,
                                        {int(i): syms_by_ind[int(i)]
                                         for i in validated.indices}))
    if pos_defect > _BLOCH_TOL:
        return Refusal(
            "positions_not_exactly_symmetric",
            f"the orbital centres miss the operations handed in by {pos_defect:.3g} "
            f"(reduced), against the {_BLOCH_TOL:.0e} to which the operators of the "
            "projection carry each centre onto its image. Averaged with operators that "
            "miss the centres, the Hamiltonian is not projected onto its symmetric part: "
            "each hopping is carried to a lattice vector that is not one, and no "
            "projection of the hoppings reaches a defect in the centres. Declare this "
            "model's precision, `realized_group(..., precision=)`, which moves centres "
            "within it onto the group's own positions before anything is projected, or "
            "move them first with `symmetrize_positions`.",
            value=pos_defect, threshold=_BLOCH_TOL,
            context={"position_defect": pos_defect, "bloch_tol": _BLOCH_TOL})

    # ---- 2. the operators, and the classes read on them ---------------------
    rep, prov = _projection_operators(model, group, assignment, validated, tol=tol,
                                      params=params, build_kw=build_kw)
    if not rep:
        return rep
    # each operation read again at the larger of the detector's residual and its
    # own matrix's covariance in this representation: a basis turned inside a
    # site is exact to the detector and near to the matrix the tables' spin lift
    # and the multiplets anchor, which is the matrix the average is taken with
    anchored = {int(i): float(v) for i, v in rep.residuals.items()}
    classes = classify_operations(validated, tol, exact_gate=exact_gate, anchored=anchored)
    indices = tuple(int(i) for i in rep.indices)
    D = {int(i): np.asarray(rep.matrices[int(i)]) for i in indices}
    bloch_tol = _BLOCH_TOL
    # THE PARTS OF THE CRYSTAL READ APART, where several were: what H holds
    # between them is below the declared precision, and the average leaves it out;
    # and what the sectors read at that precision keep apart, held apart
    apart = _same_part((_parts or {}).get("orbitals"), model)
    held_apart = (_held or {}).get("sets") or None
    not_represented = tuple(sorted({int(i) for i in validated.indices} - set(indices)))
    ops = {i: syms_by_ind[i] for i in indices}
    prov["n_operations"] = len(ops)
    prov["position_defect"] = pos_defect
    prov["bloch_tol"] = float(bloch_tol)
    # What every refusal below says when the CENTRES carry a defect of their own
    # above rounding, though within the operators' own tolerance: a projection
    # of the hoppings reaches nothing in them.  `EXACT_GATE` is the exact
    # pipeline's own rounding scale, used only to decide whether there is a
    # number worth quoting -- it gates nothing (the defect is reduced-coordinate,
    # and this number is in the units of H).
    pos_note = ("" if pos_defect <= EXACT_GATE else (
        f" This model's own centres miss the group's action by {pos_defect:.3g} "
        "(reduced), and in convention I that alone costs H(k) a covariance residual "
        "of order 2*pi*delta*max|H| whatever the hoppings do. A projection of the "
        "hoppings cannot remove that. Declare this model's precision, "
        "`realized_group(..., precision=)`, which moves the centres onto the group's "
        "own positions before anything is projected -- a k-dependent gauge, so the "
        "spectrum is untouched -- or move them first with `symmetrize_positions`."))

    # ---- 3. the transform grid --------------------------------------------
    # one division count per periodic direction, twice the reach the group
    # carries a hopping to along it and a shell more, alike along directions an
    # operation's k-action mixes, so that the grid is carried onto itself
    R_max, R_box = _range_box(model, list(ops.values()), per)
    box, need = _divisions(group, list(ops.values()), R_box)
    if k_grid is None:
        divisions = tuple(int(c) for c in need)
    else:
        divisions = (int(k_grid),) * dim_k
        if int(k_grid) < int(need.max()):
            raise ValueError(
                f"symmetrize: k_grid = {int(k_grid)} is too coarse for this model. Its "
                f"hoppings reach R_max = {R_max.tolist()} and the group action carries "
                f"them out to R_box = {R_box.tolist()}, so the inverse transform aliases "
                f"below {[int(c) for c in need]} divisions along the periodic directions "
                "-- twice the reach and the shell the aliasing check is measured on, alike "
                "along directions the group's operations mix -- and k_grid sets one count "
                f"in every direction, at least {int(need.max())}.")
    # What the projection holds at its peak, against the memory it may take: the
    # Hamiltonian and its group average on the grid, and one k-point of the
    # arithmetic streamed beside them -- every model it evaluates carries at
    # most the grid's own lattice vectors.  What the budget leaves beyond the
    # grids is the working space the streamed steps are sliced to.  A coarser
    # grid would alias, so a grid past the budget is a limit on memory, and it is
    # returned as one.
    nstate = int(model.nstate)
    n_points = int(np.prod(divisions))
    grid_bytes = 16.0 * float(n_points) * float(nstate) ** 2
    grids = _PROJECTION_GRIDS * grid_bytes
    matrix = 16.0 * float(nstate) ** 2
    point_bytes = max(_POINT_MATRICES * matrix,
                      2.0 * matrix + _bloch.point_bytes(n_points, int(model.norb), nstate))
    n_bytes = grids + point_bytes
    if n_bytes > budget:
        return _projection_too_large(n_bytes, budget, divisions=divisions, least=need,
                                     n_states=nstate, R_max=R_max, R_box=R_box)
    work = budget - grids
    _, kpts = _grid(divisions)
    k_random = np.random.default_rng(20260902).uniform(-0.5, 0.5, size=(16, dim_k))

    # ---- 4. the average, the transform, and the model ----------------------
    current, passes, per_pass, floor = model, 0, [], 0.0
    checks: dict[str, Any] = {}
    step = _slice_points(work, _POINT_MATRICES * matrix, n_points)
    for _ in range(max(1, int(max_passes))):
        Hsym, moved, scale = _average_on_grid(current, group, assignment, ops, D, kpts,
                                              divisions, params if current is model else {},
                                              bloch_tol=bloch_tol, work=work, apart=apart,
                                              held=held_apart)
        passes += 1
        per_pass.append(moved)
        t, Rlab = _inverse_transform(Hsym, current, kpts, divisions, per, work=work)
        del Hsym
        norms = _block_norms(t, int(model.norb), int(model.nspin), step)
        outside = np.any(np.abs(Rlab) > box[None, :], axis=1)
        biggest = float(norms.max(initial=0.0))
        out_box = (float(norms.max(axis=(1, 2), initial=0.0)[outside].max(initial=0.0))
                   / max(biggest, 1e-300))
        # For the representation's operators the range box is a *bound* on where
        # the symmetrized model can have hoppings, so anything outside it is
        # rounding and the gate is at rounding (measured 1e-17..1e-16 on every
        # model built from exact numbers).  The centres are carried by the
        # operations to their own tolerance, checked above, so nothing carries a
        # hopping to a lattice vector that is not one.
        gate = _RANGE_RELATIVE
        if out_box > gate:
            return Refusal(
                "symmetrize_range_not_closed",
                f"the symmetrized Hamiltonian has weight outside its own range box: the "
                f"largest rebuilt coefficient beyond |R_a| = {box.tolist()} along the "
                f"periodic directions is {out_box:.3g} of max|t|, against {gate:.3g}. The "
                f"inverse transform on a {' x '.join(str(c) for c in divisions)} grid is "
                "exact only while the hoppings fit inside that box, so this says the box "
                "is wrong and the transform aliased -- the hopping list would not be the "
                "symmetrized Hamiltonian. Raise `k_grid`." + pos_note,
                value=out_box, threshold=float(gate),
                context={"R_max": R_max.tolist(), "R_box": R_box.tolist(),
                         "k_grid": tuple(divisions), "pass": int(passes),
                         "position_defect": float(pos_defect)})
        # the orbit-safe drop, decided on the norms; the terms kept are read off
        # the grid, and the grid is let go before the hopping list is written
        cut, gap = _drop_threshold(norms, step=step)
        keep = norms > cut
        n_dropped = int(np.count_nonzero(norms > 0.0) - np.count_nonzero(keep))
        del norms, outside
        ons, amps, ii, jj, Rfull, herm = _kept_terms(t, Rlab, current, per, keep, step)
        del t, keep
        out = _to_model(current, ons, amps, ii, jj, Rfull)
        checks = {"hermiticity": herm, "drop_threshold": float(cut), "drop_gap": float(gap),
                  "n_dropped": n_dropped, "n_hoppings": int(ii.size)}
        del ons, amps, ii, jj, Rfull
        checks["outside_box"] = out_box
        checks["range_gate"] = float(gate)
        # the identity that says the hopping list IS H_sym, measured OFF the
        # grid the transform was taken on -- on the grid it holds by construction
        target, _ = _average_at(current, group, assignment, ops, D, k_random,
                                params if current is model else {}, bloch_tol=bloch_tol,
                                apart=apart, held=held_apart)
        Hnew = np.asarray(_bloch.hamiltonian(out, k_random))
        checks["k_identity"] = float(np.abs(Hnew - target).max())
        if checks["k_identity"] > gate * max(scale, 1.0):
            # The transform is exact by theorem once the range box is closed, and
            # the box has been measured closed at the same relative size -- so
            # this is the rebuild's own arithmetic going wrong, not a statement
            # about the model, and it raises.
            raise InternalError(
                f"symmetrize: the rebuilt hopping list does not reproduce the symmetrized "
                f"Hamiltonian off the transform grid: max_k |H_new(k) - H_sym(k)| = "
                f"{checks['k_identity']:.3g} on {k_random.shape[0]} random k, against "
                f"{gate * max(scale, 1.0):.3g}, while the largest coefficient outside the "
                f"range box is {out_box:.3g} relative (so the transform did not alias). "
                "The inverse transform is exact under those conditions; this is a defect "
                "in the rebuild, not a measurement.")
        _, residue = _average_at(out, group, assignment, ops, D, k_random, {},
                                 bloch_tol=bloch_tol, apart=apart, held=held_apart)
        checks["covariance_residue"] = float(residue)
        floor = covariance_gate(len(ops), int(model.nstate), scale)
        current = out
        # Two ways to be done, and both are measurements: the model is symmetric
        # under the operators it was built with, or the model stopped moving.
        # The second matters where the first cannot be reached at all -- a
        # position defect leaves an O(delta) residue no number of passes
        # touches, and iterating a fixed point only costs arithmetic.
        if residue <= floor or per_pass[-1] <= floor:
            break

    # ---- 5. how far the model moved, in total ------------------------------
    # both Hamiltonians over slices of the grid, no grid of either held, and
    # the scale about the mean on-site energy, which a constant shift moves
    mu = _bloch.onsite_mean(model, params)
    before = _bloch.evaluator(model, params)
    after = _bloch.evaluator(current, {})
    step = _slice_points(work, max(_POINT_MATRICES * matrix, 2.0 * matrix + max(
        before.point_bytes, after.point_bytes)), n_points)
    defect = scale = 0.0
    for lo in range(0, n_points, step):
        hi = min(lo + step, n_points)
        H0 = np.asarray(before(kpts[lo:hi]))
        Hend = np.asarray(after(kpts[lo:hi]))
        defect = max(defect, float(np.abs(Hend - H0).max()))
        scale = max(scale, _bloch.magnitude(H0, mu))
        del H0, Hend
    if defect > tol:
        return Refusal(
            "symmetrize_defect_exceeds_tol",
            f"the group average moved H by {defect:.3g} ({defect / max(scale, 1e-300):.3g} "
            f"relative), which is more than the declared tol = {tol:.3g}. Over operations "
            f"measured near-symmetric the defect is O(their residual), so a defect this "
            "large is a statement about the operators, not about the model: the "
            "representation the average was taken with is not the symmetry these residuals "
            "were measured for. Check the declared tolerance against the covariance "
            "the representation the average was taken with reaches, "
            f"{prov.get('covariance', float('nan')):.3g}."
            + pos_note,
            value=defect, threshold=tol,
            context={"classes": classes, "operators": prov, "passes": int(passes),
                     "per_pass": tuple(per_pass), "scale": scale,
                     "position_defect": float(pos_defect)})

    # ---- 6. the verification: the exact pipeline may now run ---------------
    verification: dict[str, Any] = {}
    if verify:
        vtol = (float(_default_of(_groups._validate_group, "tol")) if verify_tol is None
                else float(verify_tol))
        nk = (int(verify_nk) if verify_nk is not None
              else max(2, int(round(float(validated.n_kpts) ** (1.0 / dim_k)))))
        again = _groups._validate_group(current, group, nk=nk, tol=vtol, keep_intertwiners=False,
                                        max_bytes=budget)
        if isinstance(again, Refusal):
            return again                    # the detector's memory limit, as it came
        kept_now = {int(i) for i in again.indices}
        verification = {
            "tol": vtol, "nk": nk, "n_kept": len(again.indices),
            "n_operations": len(again.indices) + len(again.discrepancies),
            "worst_residual": float(max((again.residuals[i] for i in again.indices),
                                        default=float("nan"))),
            "near_kept": all(i in kept_now for i in
                             (i for i, c in classes.items() if c == "near")),
            "broken": tuple(int(i) for i in again.broken), "note": ""}
        if not verification["near_kept"]:
            # The re-detection, which assumes no tensor structure, is a *second
            # opinion*, and it has its own accuracy floor: the eigen route
            # parametrizes on eigenvectors, which a level splitting `g` resolves
            # only to `eps·‖H‖/g`, so a model with near-degenerate levels cannot
            # be certified to machine precision by that route however symmetric
            # it is.  The first opinion is `checks["covariance_residue"]`,
            # measured directly with the operators the average was taken with,
            # and it is the one that says whether the projection worked.
            verification["note"] = (
                f"measured again, the symmetrized model did not keep every near operation "
                f"at {vtol:.3g}, while "
                f"the covariance of the symmetrized model against the operators it was "
                f"built with measures {checks.get('covariance_residue', float('nan')):.3g}. "
                "Read off eigenvectors, the symmetry matrices measured again are good only "
                "to eps*||H||/(smallest level splitting), a floor symmetrization does not "
                "change, so the covariance against the operators is the measurement that "
                "says whether the projection worked.")

    # WHY, where no operation says it: multiplets measured within the declared
    # tolerance only, as the multiplet measurement handed in records it
    layout = build_kw.get("multiplets")
    held = ({"mixing": float(layout._mixing), "exact": float(_IN_BLOCKS),
             "declared": float(layout._tol or 0.0)}
            if isinstance(layout, SiteMultiplets) and layout._near else {})
    # ... or parts of the crystal coupled only below the declared precision
    coupling = float((_parts or {}).get("coupling", 0.0))
    uncoupled = ({"orbitals": tuple(tuple(int(o) for o in orbs) for orbs in _parts["orbitals"]),
                  "coupling": coupling, "exact": float(exact_gate)}
                 if apart is not None and coupling > exact_gate else {})
    # ... or identical copies held as copies, and sectors splitting multiplets
    # held apart, only within it
    off = dict((_held or {}).get("off") or {})
    sets = held_apart or {}
    as_copies = ({"orbitals": tuple(tuple(orbs) for _Q, _m, _d, orbs in sets["copies"]),
                  "copies": tuple(int(m) for _Q, m, _d, _orbs in sets["copies"]),
                  "off": float(off.get("copies", 0.0)), "exact": float(exact_gate)}
                 if sets.get("copies") and float(off.get("copies", 0.0)) > exact_gate else {})
    as_shares = ({"orbitals": tuple(tuple(orbs) for _bases, orbs in sets["shares"]),
                  "sectors": tuple(len(bases) for bases, _orbs in sets["shares"]),
                  "coupling": float(off.get("shares", 0.0)), "exact": float(exact_gate)}
                 if sets.get("shares") and float(off.get("shares", 0.0)) > exact_gate else {})
    record = Symmetrization(
        tol=tol, exact_gate=exact_gate, classes=classes,
        residuals={int(i): _residual_read(validated, anchored, i, missing=np.nan)
                   for i in classes},
        defect=defect, defect_relative=defect / max(scale, 1e-300), scale=scale,
        operations=tuple(sorted(ops)), not_represented=not_represented,
        grid={"n": tuple(divisions), "n_points": int(kpts.shape[0]), "dim_k": dim_k,
              "R_max": tuple(int(x) for x in R_max),
              "R_box": tuple(int(x) for x in R_box)},
        checks=dict(checks, per_pass=tuple(per_pass)), operators=prov,
        verification=verification, passes=int(passes), machine_floor=float(floor),
        multiplets=held, parts=uncoupled, copies=as_copies, shares=as_shares,
        _spinful=bool(group.spinor))
    return current, record


# --------------------------------------------------------------------------
# the other half of the model: the orbital centres
# --------------------------------------------------------------------------

def symmetrize_positions(model, group, assignment, *, max_move=1e-5, params=None):
    r"""The model with its orbital centres moved onto the nearest positions its group carries, every band where it was.

    Centres that miss the group's own action by ``δ``, in reduced coordinates,
    cost ``H(k)`` a covariance residual of order ``2π·δ·max|H|`` whatever the
    hoppings do, and a projection of the hoppings cannot remove it.  Moving a
    centre is a ``k``-dependent gauge transformation, so the spectrum is
    untouched, and the call measures that it is.  The centres are fitted about
    their own origin, a shift of all of them together being no displacement,
    and a common shift within ``max_move`` is then removed, so that they come
    back in the model's own frame.  ``realized_group(model, precision=...)``
    makes the same move before the cell's setting is read, and returns its
    record on ``.positions``.

    Parameters
    ----------
    model : pythtb.TBModel
        The model whose centres are moved.
    group, assignment : GroupData, OrbitalAssignment
        ``r.group`` and ``r.assignment`` of a call on this model.  A group read
        off the noisy centres themselves is written about an origin their noise
        placed: identify the moved model again before reading it in a group.
    max_move : float, optional
        How far a centre may move and still be the same orbital, in reduced
        coordinates, a shift of all of them together not counted; and how far
        that shift may reach and still be removed.  Default ``1e-5``.
    params : dict, optional
        Values for a model whose terms are given by parameter names.

    Returns
    -------
    (pythtb.TBModel, dict) or Refusal
        The model with its centres moved, and the record of the move:
        ``defect_before`` and ``defect_after``, how far the centres miss the
        group's action before and after, in reduced coordinates;
        ``max_move``, the largest move of one centre; ``offset``, the shift of
        them all together, each centre returned being the one handed in plus
        its own move plus ``offset``; ``frame``, ``"model"`` where they are
        written in the model's own frame and ``"fitted"`` where they stay about
        the origin fitted to them; ``spectrum_residual``, how far the spectrum
        moved over ``n_check`` random ``k``, ``0`` where no centre moved; and
        ``n_moved``.  A ``Refusal`` in its place is falsy.

    Raises
    ------
    ..refusal.InternalError
        The spectrum moved, or the group's operations miss the positions moved
        onto them, neither of which the move can do: a defect to report.

    Refusals
    --------
    ``positions_not_nearly_symmetric``
        a centre would have to move further than ``max_move``: the positions
        are a different set of centres, and the group they carry is the one to
        identify.

    Examples
    --------
    Kane–Mele with one centre moved by ``1e-8`` (``moved``), read in the group
    of the exact model:

    >>> r = realized_group(kane_mele(0.0, -1.0, 0.3, 0.25), characters={0: "A1"})  # doctest: +SKIP
    >>> fixed, rec = symmetrize_positions(moved, r.group, r.assignment)  # doctest: +SKIP
    >>> f"{rec['defect_before']:.1e} -> {rec['defect_after']:.1e}", rec["frame"]  # doctest: +SKIP
    ('2.0e-08 -> 1.1e-16', 'model')
    """
    from . import groups as _groups

    frame = _groups._assigned_sites(model, group, assignment)
    syms = group.symmetries
    unitary = [k for k, s in enumerate(syms) if not s.time_reversal]
    rotations = np.array([np.asarray(syms[k].rotation, dtype=float) for k in unitary])
    perm = np.asarray(assignment.permutations, dtype=int)[unitary]
    shift = np.asarray(assignment.lattice_shift, dtype=float)[unitary]
    rows = list(range(len(unitary)))
    x, trans, closure = _groups._symmetric_sites(frame, perm, shift, rotations, rows)
    if not closure <= _EXACT_CLOSURE:          # pragma: no cover - an assignment is an action
        raise InternalError(
            f"the unitary operations of {group.name} #{group.number_str} do not carry "
            f"the positions projected onto them, missing by {closure:.3g}: the assignment "
            "is the action of a group on the sites by construction, so this is arithmetic "
            "in this package failing, not a measurement about the model.")
    raw = np.asarray(model.orb_vecs, dtype=float)
    dim_r = int(model.lattice.dim_r)
    moves = _groups._moves_of(frame, x)
    worst = float(np.abs(moves).max(initial=0.0))
    before = _groups._frame_defect(frame, perm, rotations, trans, rows)
    if not within(worst, max_move):
        return Refusal(
            "positions_not_nearly_symmetric",
            f"an orbital centre would have to move {worst:.3g} (reduced) onto the nearest "
            f"positions the group carries, against max_move = {float(max_move):.3g}. That "
            "is a different set of centres, beyond a numerical defect: moving them would "
            "substitute a crystal the model does not have. Identify the group the "
            "positions carry (`typat=`, `decorations=`), or raise `max_move` where that "
            "displacement is meant.",
            value=worst, threshold=float(max_move),
            context={"defect_before": float(before),
                     "n_moved": int(np.count_nonzero(np.abs(moves).max(axis=1) > 0.0))})
    delta = _groups._rational_offset(trans, rotations, rows, model.periodic_dirs)
    in_frame = delta is not None and within(float(np.abs(delta).max(initial=0.0)), max_move)
    final = x - delta if in_frame else x
    shifted = _groups._moves_of(frame, final)
    orb = raw + shifted[:, :dim_r]
    if np.array_equal(orb, raw):
        out, resid, measured = model, 0.0, 0
    else:
        (out, resid), measured = _with_centres(model, orb, params=params), _bloch.CHECK_K
    final_trans = np.array([final[perm[r, 0]] + shift[r, 0] - rotations[r] @ final[0]
                            for r in rows])
    offset = (0.0 - np.asarray(delta) * frame["scale"])[:dim_r] if in_frame else np.zeros(dim_r)
    return out, {
        "defect_before": _groups._frame_defect(frame, perm, rotations, final_trans, rows),
        "defect_after": _groups._frame_defect(frame, perm, rotations, final_trans, rows,
                                              sites=final),
        "max_move": worst, "offset": tuple(float(c) for c in offset),
        "frame": "model" if in_frame else "fitted",
        "spectrum_residual": float(resid), "n_check": measured,
        "n_moved": int(np.count_nonzero(np.abs(orb - raw).max(axis=1) > 0.0))}
