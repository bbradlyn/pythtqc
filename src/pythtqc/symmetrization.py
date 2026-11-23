r"""A model symmetric only to its declared precision, made exactly symmetric.

A Hamiltonian fitted to a calculation, such as a Wannier90 import, obeys its
group only to the precision of its own numbers, which ``precision=``
declares.  With ``precision=`` or ``position_tol=`` declared,
:func:`.groups.realized_group` moves the orbital centers onto symmetric
positions.  Wherever an operation is near, and in the other cases
:class:`Symmetrization` names, it projects ``H`` onto its symmetric part, at
the tolerance a precision derives or at the default one where none is
declared.  The projection averages ``H`` over the operations of the group on
a grid of k-points.  The average is written back as a hopping list by an
inverse Fourier transform.  It moves each entry of ``H(k)`` by at most
``Symmetrization.tol``, the tolerance on each operation's covariance
residual.  An operation obeyed to the rounding of the model's own arithmetic
is exact, and one obeyed within ``tol`` and not to rounding is near.

- :class:`Symmetrization`: how far ``H`` was moved and why, the record
  returned on ``r.numerical.symmetrization``.  Its parts are
  :class:`SymmetrizationChecks`, :class:`SymmetrizationVerification`,
  :class:`SymmetrizationSiteMultiplets`, :class:`SymmetrizationParts`,
  :class:`SymmetrizationCopies` and :class:`SymmetrizationShares`.
- :data:`EXACT_GATE`, ``5e-14``: the least the boundary between exact and
  near can be, relative to the largest entry of ``H(k)``.  The boundary is
  larger where the rounding of the model's own arithmetic is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import _bloch
from . import _centers
from . import _measure
from .refusal import InternalError, Refusal
from ._tolerance import (
    EXACT_TOL, _DROP_RELATIVE, _EXACT_CLOSURE, _IN_BLOCKS, _RANGE_RELATIVE, eigen_window,
    within)
from ._text import _counted_as, _index_ranges, _orbitals_named
from ._detector import _GridHamiltonians, _budget_that_runs, _size
from ._matrices import (
    SiteMultiplets, _BLOCH_TOL, _default_of, _nearest_representation, _resolve_spin,
    bloch_operator, covariance_gate)
from ._grammar import _declaration_grammar
from ._centers import _centers_account_for, _position_defect
from ._build import _build_rep, _exact_gate

__all__ = ["Symmetrization", "SymmetrizationChecks", "SymmetrizationVerification",
           "SymmetrizationSiteMultiplets", "SymmetrizationCopies", "SymmetrizationParts",
           "SymmetrizationShares", "EXACT_GATE"]

#: The least covariance an operation obeyed exactly is held to, ``5e-14`` for a
#: Hamiltonian whose largest entry is one.  A model's own boundary between exact
#: and near is :func:`..symmetry.covariance_gate` of its representation, this
#: times ``max|H|`` or the rounding its arithmetic carries where that is larger,
#: and no declared ``tol`` moves it.
EXACT_GATE = float(EXACT_TOL)

#: ``_build_rep``'s own internal-factor gate, the floor of the provisional pass's
#: loosened one.
_INTERNAL_GATE = float(_default_of(_build_rep, "internal_tol"))

# --------------------------------------------------------------------------
# the classification: exact / near / broken, per operation
# --------------------------------------------------------------------------

def _classify_operations(validated, tol, *, exact_gate=EXACT_GATE, anchored=None) -> dict:
    """Each operation's class against a declared tolerance: exact, near or broken.

    An operation is exact where it is obeyed to rounding, near where it is
    obeyed within ``tol`` and not to rounding, and broken beyond ``tol``.
    ``validated`` is the :class:`.._measure.ValidatedGroup` of the operations a
    call measured to hold at ``tol``, an energy in the units of ``H``.
    ``exact_gate`` is the rounding at or below which an operation is exact, in
    the units of ``H``: the covariance gate of the model's own representation
    (:func:`.._build._exact_gate`), :data:`EXACT_GATE` by default.
    ``anchored`` maps an operation's index to its covariance residual in the
    representation the projection is built with; where it holds an operation,
    that residual decides exact or near in place of the measured one.  An
    operation is broken where either residual exceeds ``tol``.  Returns a
    mapping from each operation's index in ``validated`` to ``"exact"``,
    ``"near"`` or ``"broken"``.
    """
    tol = float(tol)
    out = {}
    for ind in validated._indices:
        measured = float(validated.residuals.get(int(ind), np.inf))
        r = (float(anchored[int(ind)]) if anchored and int(ind) in anchored
             else measured)
        out[int(ind)] = ("broken" if measured > tol or r > tol else
                         "exact" if r <= exact_gate else "near")
    return out

def _named(sets) -> str:
    """Sets of orbitals in words, as ``"orbitals 0-3"`` or ``"orbitals 0-1 and 4-5"``.

    ``sets`` is a sequence of sequences of orbital indices.  Returns the words.
    """
    sets = [tuple(int(o) for o in s) for s in sets]
    if len(sets) == 1:
        return _orbitals_named(sets[0])
    return "orbitals " + " and ".join(_index_ranges(s) for s in sets)

def _residual_read(validated, anchored, ind, *, missing):
    """The residual one operation is classified exact or near by, in the units of ``H``.

    ``validated`` is the measurement of the operations, and ``anchored`` maps
    an operation's index to its residual in the representation the projection
    is built with, or is ``None``.  Returns ``anchored``'s residual for the
    operation ``ind`` where it holds one, the measured one otherwise, and
    ``missing`` where neither holds it.
    """
    if anchored and int(ind) in anchored:
        return float(anchored[int(ind)])
    return float(validated.residuals.get(int(ind), missing))

# --------------------------------------------------------------------------
# the record
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SymmetrizationVerification:
    """The operations of the group measured again on the symmetrized model.

    They are measured at ``1e-8`` times ``Symmetrization.scale``; ``1e-8`` is
    the relative tolerance a model built from exact numbers is measured at.

    Attributes
    ----------
    tol : float
        The tolerance on each operation's covariance residual, in the units of
        ``H``: ``1e-8`` times ``Symmetrization.scale``.
    nk : int
        The number of k-points along each periodic direction of the grid the
        operations are measured on, the same as in the measurement before the
        projection.
    n_kept : int
        How many operations held.
    n_operations : int
        How many operations were measured.
    worst_residual : float
        The largest covariance residual of an operation that held, in the
        units of ``H``.
    near_kept : bool
        Whether every operation that was near before the projection held.
    broken : tuple of int
        The operations measured broken, whose covariance residual exceeds
        ``tol``, by their index in ``r.operations``.
    note : str
        If nonempty, a sentence saying that an operation near before the
        projection did not hold.  It quotes
        ``Symmetrization.checks.covariance_residue``, the measurement that says
        whether the projection worked.
    """

    tol: float
    nk: int
    n_kept: int
    n_operations: int
    worst_residual: float
    near_kept: bool
    broken: tuple
    note: str = ""


@dataclass(frozen=True)
class SymmetrizationSiteMultiplets:
    """Site multiplets that hold only within the declared precision.

    A site multiplet is a set of orbitals at one site that the Hamiltonian
    binds into one carrier of a site irrep.  The orbitals' frame can be turned
    inside the site by a small angle.  The symmetry matrices measured on the
    Hamiltonian then mix the multiplets by an amount of the size of that
    angle.  The projection makes the multiplets exact.

    Attributes
    ----------
    mixing : float
        How far the symmetry matrices measured on ``H`` mix the multiplets,
        relative to the size of the matrices.
    exact : float
        The largest mixing at which the multiplets count as exact, ``1e-8``.
    declared : float
        ``Symmetrization.tol`` relative to ``max|H|``, the tolerance the
        multiplets were measured at: ten times the precision where one is
        declared.
    """

    mixing: float
    exact: float
    declared: float


@dataclass(frozen=True)
class SymmetrizationChecks:
    """The measurements on the hopping list the projection returns.

    They measure how closely the hopping list reproduces the group average of
    ``H``.

    Attributes
    ----------
    hermiticity : float
        How far the coefficients ``t(R)`` of the transform miss
        ``t(−R) = t(R)†``, in the units of ``H``.
    drop_threshold : float
        The norm below which a coefficient of the transform was dropped as
        rounding, in the units of ``H``; ``0.0`` where none was.
    drop_gap : float
        Where a coefficient was dropped, the smallest norm kept over the
        largest norm dropped.  Coefficients are dropped only where this ratio
        exceeds ten.  Where none was dropped, it holds another ratio or
        ``inf``, and is no gap.
    n_dropped : int
        How many nonzero coefficients of the transform were dropped.
    n_hoppings : int
        How many hopping terms the model returned holds, each conjugate pair
        counted once and the on-site terms not counted.
    outside_box : float
        The largest coefficient of the transform at a lattice vector outside
        the box that bounds the group's images of the model's hoppings,
        relative to the largest coefficient.
    range_gate : float
        The largest ``outside_box`` the projection accepts, ``1e-10``.
    k_identity : float
        How far the model returned misses the average of ``H``, at 16 random
        k-points off the grid, in the units of ``H``.
    covariance_residue : float
        The covariance residual of the model returned, with the matrices of the
        representation the average is taken with, at the 16 random k-points
        ``k_identity`` is measured at, in the units of ``H``.
    per_pass : tuple of float
        How far each averaging pass moved ``H`` on the grid, in the units of
        ``H``.
    """

    hermiticity: Any = None
    drop_threshold: Any = None
    drop_gap: Any = None
    n_dropped: Any = None
    n_hoppings: Any = None
    outside_box: Any = None
    range_gate: Any = None
    k_identity: Any = None
    covariance_residue: Any = None
    per_pass: tuple = ()


@dataclass(frozen=True)
class SymmetrizationCopies:
    """Identical copies coupled or told apart only below the declared precision.

    Identical copies of a set of orbitals form a pseudospin the Hamiltonian
    conserves within the declared precision.  The projection makes them exact
    copies.

    Attributes
    ----------
    orbitals : tuple of tuple of int
        The orbitals of each set of copies.
    copies : tuple of int
        The number of copies in each set.
    off : float
        The largest entry of ``H(k)`` by which the copies miss being exact
        copies, in the units of ``H``.
    exact : float
        The largest covariance residual at which an operation counts as exact,
        in the units of ``H``.  ``off`` is above it.
    """

    orbitals: tuple
    copies: tuple
    off: float
    exact: float


@dataclass(frozen=True)
class SymmetrizationParts:
    """Parts of the crystal coupled to one another only below the declared precision.

    A part of the crystal is a set of orbitals the hoppings connect, directly
    or through others.  A coupling between parts that lies below the declared
    precision counts as absent.  ``H`` then fixes the matrices of the
    operations on each part only up to a one-dimensional character, chosen for
    each part separately.  The projection leaves the coupling out, so the
    model returned holds the parts exactly apart.

    Attributes
    ----------
    orbitals : tuple of tuple of int
        The orbitals of each part.
    coupling : float
        The largest entry of ``H(k)`` between two parts, in the units of ``H``.
    exact : float
        The largest covariance residual at which an operation counts as exact,
        in the units of ``H``.  ``coupling`` is above it.
    """

    orbitals: tuple
    coupling: float
    exact: float


@dataclass(frozen=True)
class SymmetrizationShares:
    """Sectors that split site multiplets, coupled only below the declared precision.

    The matrices that commute with ``H(k)`` at every k and act within sites
    split the states into sectors, read at the declared precision.  Where the
    states of one site multiplet lie in several sectors, each orbit of sectors
    under the group carries a representation of its own, induced from the
    stabilizer of one of its sectors.  Two such cases are the bonding and the
    antibonding state of two identical lattices coupled on site, and a
    square's ``p_x, p_y`` pair that nothing couples.  Where ``H`` couples such
    sectors only below the declared precision, the projection leaves that
    coupling out.

    Attributes
    ----------
    orbitals : tuple of tuple of int
        The orbitals of each set of sectors.
    sectors : tuple of int
        The number of sectors in each set.
    coupling : float
        The largest entry of ``H(k)`` that couples a sector to anything outside
        it, in the units of ``H``.
    exact : float
        The largest covariance residual at which an operation counts as exact,
        in the units of ``H``.  ``coupling`` is above it.
    """

    orbitals: tuple
    sectors: tuple
    coupling: float
    exact: float


@dataclass(frozen=True)
class Symmetrization:
    """How far a Hamiltonian was moved onto its symmetric part, and why.

    The record :func:`..groups.realized_group` returns on
    ``r.numerical.symmetrization`` where it projects ``H`` onto its symmetric
    part; that field is ``None`` where it does not.  An operation's covariance
    residual is how far its matrix misses carrying ``H(k)`` onto ``H`` at the
    image of k: the largest entry of ``U H(k) U† − H(Rk)`` over k for the
    operation ``{R|t}``, and of ``U H(k)* U† − H(−Rk)`` for ``{R|t}`` combined
    with time reversal, in the units of ``H``.  ``H`` is averaged over the
    operations of the group, those combined with time reversal included, on
    a grid of k-points.  The average
    is taken with a representation of the group on the orbitals that is built
    from ``H`` for the projection and made an exact representation of the
    group.  The average is written back as a hopping list by an inverse
    Fourier transform.

    The projection runs where an operation is near, obeyed within ``tol`` and
    not to rounding.  It runs as well in four cases where something holds
    only within the declared precision, each with an attribute of its own:
    the site multiplets (``site_multiplets``), the separation of parts of the
    crystal (``parts``), identical copies (``copies``), and the separation of
    sectors that split site multiplets (``shares``).  Below the declared
    precision means too small for the measurement of the operations, made at
    ``tol``, to see: it reads the orbitals such a coupling joins as uncoupled,
    and copies that differ by so little as identical.  ``r.model`` is then the
    symmetrized model, and ``r.numerical.source`` the model passed to
    ``realized_group``.  ``summary()`` returns the record as text, a line for
    each finding, ending with the attribute that holds it where there is one.

    Attributes
    ----------
    defect : float
        How far ``H`` moved: the largest change of an entry of ``H(k)`` over
        the grid, in the units of ``H``.  It is at most ``tol``.
    scale : float
        The size of ``H``: the largest entry of ``H(k) − μ`` over the grid,
        ``μ`` the mean on-site energy per state, in the units of ``H``.
    tol : float
        The tolerance on each operation's covariance residual, in the units of
        ``H``: ``10`` times the precision times ``max|H|`` where a precision is
        declared, and ``1e-8`` times ``max|H|`` otherwise.  ``max|H|`` is the
        largest entry of ``H(k) − μ`` on fixed random k-points, or the largest
        hopping or on-site energy less ``μ`` where that is larger: the number
        ``r.numerical.precision.max_h`` holds where a precision is declared.
        It is not ``scale``, which is read on the grid.
    near : tuple of int
        The operations obeyed within ``tol`` and not to rounding, by their
        index in ``r.operations``.
    exact : tuple of int
        The operations obeyed to rounding, by their index in ``r.operations``.
    residuals : dict
        The covariance residual of each operation, by its index in
        ``r.operations``, in the units of ``H``.  For an operation the
        representation the average is taken with holds, it is the residual of
        the operation's own matrix there.  For any other, it is the residual
        of the best matrix the measurement of the operations found on ``H``.
    passes : int
        The number of averaging passes, at most three.  One pass is the
        projection.  A further pass runs only while the covariance residual of
        the averaged model is above rounding and the last pass moved ``H`` by
        more than rounding.
    site_multiplets : SymmetrizationSiteMultiplets or None
        If not ``None``, the site multiplets that hold only within the
        declared precision, which the projection made exact.
    parts : SymmetrizationParts or None
        If not ``None``, the parts of the crystal coupled to one another only
        below the declared precision, a coupling the projection left out.
    copies : SymmetrizationCopies or None
        If not ``None``, the identical copies that ``H`` coupled or told apart
        only below the declared precision, which the projection made exact
        copies.
    shares : SymmetrizationShares or None
        If not ``None``, the sectors splitting site multiplets that ``H``
        coupled only below the declared precision, a coupling the projection
        left out.
    checks : SymmetrizationChecks
        The measurements of how closely the hopping list returned reproduces
        the group average of ``H``.
    verification : SymmetrizationVerification
        The operations of the group measured again on the model returned, at
        ``1e-8`` times ``scale``; ``1e-8`` is the relative tolerance a model
        built from exact numbers is measured at.
    """

    tol: float
    residuals: dict
    defect: float
    scale: float
    checks: SymmetrizationChecks
    passes: int = 1
    verification: SymmetrizationVerification | None = None
    site_multiplets: SymmetrizationSiteMultiplets | None = None
    parts: SymmetrizationParts | None = None
    copies: SymmetrizationCopies | None = None
    shares: SymmetrizationShares | None = None
    # `{ind: "exact" or "near"}` for each operation averaged over
    _classes: dict = field(default_factory=dict, repr=False)
    # rounding, in the units of H: an operation obeyed to it is exact
    _exact_gate: float = 0.0
    # operations measured to hold that the representation does not carry, a
    # slab cell's operations on its vacuum, left out of the average
    _not_represented: tuple = ()
    # the k grid the average was taken on: `n`, `n_points`, `R_max`, `R_box`,
    # `dim_k`
    _grid: dict = field(default_factory=dict, repr=False)
    # the representation the average was taken with: `tol`, `covariance`,
    # `composition`, `branch_ambiguity`, `position_defect` (how far the orbital
    # centers miss the group's own action, a floor no projection of the hoppings
    # goes below), `bloch_tol`, `n_operations`
    _operators: dict = field(default_factory=dict, repr=False)
    # the rounding the passes stop at
    _machine_floor: float = 0.0
    # whether the states carry a spin, so that a count of operations says it
    # counts the single-valued group's
    _spinful: bool = field(default=False, repr=False, compare=False)

    @property
    def _relative(self) -> float:
        """How far the model moved against its size, ``defect / scale``."""
        return self.defect / max(self.scale, 1e-300)

    @property
    def near(self) -> tuple:
        """The operations obeyed within ``tol`` and not to rounding.

        Each is given by its index in ``r.operations``.
        """
        return tuple(sorted(i for i, c in self._classes.items() if c == "near"))

    @property
    def exact(self) -> tuple:
        """The operations obeyed to rounding, by their index in ``r.operations``."""
        return tuple(sorted(i for i, c in self._classes.items() if c == "exact"))

    def summary(self) -> str:
        """The record as text: how far ``H`` moved and why, a line for each finding.

        Each line ends with the attribute that holds what it reads, where
        there is one.
        """
        relative = self._relative
        lines = [f"H moved by {self.defect:.3g}, {relative:.3g} of its size"
                 + (f", in {self.passes} passes" if self.passes > 1 else "")
                 + "   (.defect, .scale)",
                 "  the tolerance on each operation's residual, how far its matrix misses "
                 f"carrying H(k) onto H at the image of k: {self.tol:.3g}, in the units of H"
                 "   (.tol)"]
        if self.near:
            lines.append(f"  {len(self.near)} of the {len(self._classes)} operations were "
                         "near-symmetric, obeyed within that tolerance and not to rounding, "
                         f"{self._exact_gate:.2g}: the worst at "
                         f"{max(self.residuals[i] for i in self.near):.3g}   (.near, .residuals)")
        m, p, c, sh = self.site_multiplets, self.parts, self.copies, self.shares
        if m is not None:
            lines.append("  the site multiplets hold only within the declared precision: the "
                         f"Hamiltonian's own symmetry matrices mix them at {m.mixing:.3g} "
                         "of their size, and the projection made them exact   (.site_multiplets)")
        if p is not None:
            lines.append(f"  {len(p.orbitals)} parts of the crystal are coupled only below "
                         f"the declared precision, at most {p.coupling:.3g}, and that "
                         "coupling was left out   (.parts)")
        if c is not None:
            lines.append(f"  the identical copies on {_named(c.orbitals)} are copies only "
                         f"within the declared precision, {c.off:.3g} from exact, and were "
                         "made exact copies   (.copies)")
        if sh is not None:
            lines.append(f"  the states splitting the multiplets on {_named(sh.orbitals)} "
                         "are coupled only below the declared precision, at most "
                         f"{sh.coupling:.3g}, and that coupling was left out   (.shares)")
        if not (self.near or m or p or c or sh):
            lines.append(f"  every operation exact to rounding, {self._exact_gate:.2g}, and "
                         "nothing to correct   (.exact)")
        moved = float((self._operators or {}).get("position_defect", 0.0))
        if moved > EXACT_GATE:
            lines.append(f"  the orbital centers miss the group's own positions by {moved:.3g}, "
                         "which no projection of the hoppings removes")
        k_identity = self.checks.k_identity
        lines.append("  the hopping list returned is the group average to "
                     f"{float('nan') if k_identity is None else k_identity:.3g} at random k"
                     "   (.checks)")
        v = self.verification
        lines.append(f"  measured again, at a tolerance of {v.tol:.3g}: {v.n_kept} of "
                     f"{v.n_operations} operations obeyed, the worst at "
                     f"{v.worst_residual:.3g}   (.verification)" if v is not None else
                     "  not measured again   (.verification)")
        if self.near or v is not None:
            lines.append(f"  the operations counted are {_counted_as(self._spinful)}"
                         "   (.near, .verification)")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover: convenience only
        relative = self._relative
        mixed = (f", site multiplets mixed at {self.site_multiplets.mixing:.3g}"
                 if self.site_multiplets is not None else "")
        mixed += (f", parts coupled at {self.parts.coupling:.3g}"
                  if self.parts is not None else "")
        mixed += (f", copies {self.copies.off:.3g} from exact"
                  if self.copies is not None else "")
        mixed += (f", sectors coupled at {self.shares.coupling:.3g}"
                  if self.shares is not None else "")
        worst = (float("nan") if self.verification is None
                 else self.verification.worst_residual)
        return (f"Symmetrization(defect {self.defect:.3g} "
                f"({relative:.3g} relative) against tol {self.tol:.3g}{mixed}, "
                f"{self.passes} pass{'' if self.passes == 1 else 'es'}, "
                f"verified at {worst:.3g}; "
                f"{len(self.near)} of {len(self._classes)} operations near, "
                f"{_counted_as(self._spinful)})")


# --------------------------------------------------------------------------
# the transform grid: the hopping range, and the box the group action reaches
# --------------------------------------------------------------------------

def _range_box(model, syms, per):
    r"""The hoppings' range, and the box the group carries them to, in lattice vectors.

    ``model`` is the pythtb model, ``syms`` the operations averaged over and
    ``per`` the periodic directions.  Returns ``(R_max, R_box)``, integer
    arrays over ``per``: ``R_max`` the largest ``|R_a|`` of the hopping list
    along each periodic direction, and ``R_box`` a bound on those of the
    symmetrized model.  A term at displacement ``d = R + τ_j − τ_i`` is
    carried by ``{A|t}`` to ``A d``, so

    .. math:: |R'_a| \le \max_{d,g} |(A d)_a| + \max_{ij}|τ_{i,a} − τ_{j,a}|,

    the maximum taken over the displacements the model holds.  What the
    symmetrized model holds outside the box is measured
    (``checks.outside_box``).
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
    r"""The reach each periodic direction is read to, and the grid's divisions along it.

    ``group`` is the :class:`.._spacegroup.GroupData` whose action on k is
    read, ``syms`` the operations averaged over, and ``R_box`` the box of
    :func:`_range_box`.  The periodic directions fall into the sets the
    operations' actions on k connect, and each is read to the widest reach in
    its set.  Returns ``(box, divisions)``, integer arrays over the periodic
    directions, with ``divisions = 2·(box + 1) + 1``: twice the reach and a
    shell more, the shell where an aliased transform would show.  A grid
    starting at ``k = 0`` and so divided is carried onto itself by every
    operation.
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
        image = np.asarray(group._k_action_of(s)(probe), dtype=float)   # rows: images of e_b
        for b in range(dim_k):
            for a in range(dim_k):
                if a != b and abs(image[b, a]) > 0.5:
                    lead[root(a)] = root(b)
    box = np.array([max(int(R_box[b]) for b in range(dim_k) if root(b) == root(a))
                    for a in range(dim_k)], dtype=int)
    return box, 2 * (box + 1) + 1

def _grid(divisions):
    """The grid starting at ``k = 0`` with ``divisions`` along each periodic direction.

    Returns ``(m, k)``: the points as integer multiples, of shape ``(N, d)``,
    and as reduced k, ``m / divisions``.
    """
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
#: average; the grid is let go before the hopping list is written.  Everything
#: else (each operation's Bloch operators and products, the Fourier phases,
#: the block norms) is streamed over slices of the grid in the working space
#: the budget leaves (:data:`_POINT_MATRICES`).  Measured with ``tracemalloc``
#: over a whole call at the least budget that runs it, on models with dense
#: hoppings in the shape of a Wannier90 import, where the grid dominates: 2.02
#: grids on 96 spinless states and on 48 spinful orbitals (a ``9³`` grid, one
#: grid 107.5 MB), 2.03 on 48 states (``9³``) and on 32 states reaching two
#: cells (``11³``).  The symmetrized model the pass writes is about the size of
#: the model handed in, which is not in the count: writing it reached 0.2 to
#: 1.55 grids on these.
_PROJECTION_GRIDS = 2.1

#: The working space one k-point of that streamed arithmetic takes, as a count
#: of ``n_states × n_states`` complex matrices: an operation's Bloch operator
#: and its adjoint, the two products and the slice of the average they are
#: added to.  Measured with ``tracemalloc`` on the same models as
#: :data:`_PROJECTION_GRIDS` at one and a half times the least budget, slices
#: of 129 to 235 points: 5.95 matrices a point on every one.
_POINT_MATRICES = 6.0

def _memory_budget(max_bytes, call="realized_group") -> float:
    """``max_bytes`` read as a number of bytes.

    Returns it as a float.  Raises ``ValueError``, opening with ``call``, the
    public call's name, where it is not a positive, finite number.
    :func:`_symmetrized` and :func:`..groups.realized_group` read it before
    anything is measured.
    """
    try:
        budget = float(max_bytes)
    except (TypeError, ValueError):
        budget = float("nan")
    if not (budget > 0.0 and np.isfinite(budget)):
        raise ValueError(
            f"{call}(): max_bytes={max_bytes!r} is not a positive, finite number "
            "of bytes. Pass a number of bytes, such as the default 1e9.")
    return budget

def _search_ceiling(max_candidates, call="realized_group") -> int:
    """``max_candidates`` read as a whole number of candidates.

    Returns it as an int.  Raises ``ValueError``, opening with ``call``, the
    public call's name, where it is not a positive whole number; a bool is
    not one.  :func:`_symmetrized` and :func:`..groups.realized_group` read it
    before anything is measured.
    """
    if (isinstance(max_candidates, (bool, np.bool_))
            or not isinstance(max_candidates, (int, np.integer)) or int(max_candidates) < 1):
        raise ValueError(
            f"{call}(): max_candidates={max_candidates!r} is not a positive whole "
            "number. Pass a whole number of candidates, such as the default 2**20.")
    return int(max_candidates)

def _slice_points(work, point_bytes, n_points) -> int:
    """How many grid points one slice of the streamed arithmetic takes.

    ``work`` is the working space and ``point_bytes`` what one point takes, in
    bytes; ``n_points`` is the number of points of the grid.  Returns
    ``work // point_bytes``, at least one and at most ``n_points``.
    """
    return int(min(int(n_points), max(1, int(float(work) // float(point_bytes)))))

def _projection_too_large(n_bytes, budget, *, divisions, least, n_states, R_max, R_box):
    r"""The refusal ``projection_too_large``, for a grid the memory budget does not hold.

    ``n_bytes`` is the peak the projection needs and ``budget`` the
    ``max_bytes`` asked, in bytes.  ``divisions`` is the transform grid,
    ``least`` the coarsest grid that does not alias, ``n_states`` the number of
    states, and ``R_max`` and ``R_box`` the arrays of :func:`_range_box`.
    Returns the refusal, its reason naming the ``max_bytes`` that runs it.
    """
    runs = _budget_that_runs(n_bytes)
    counts = [int(c) for c in divisions]
    n_points = int(np.prod(counts))
    return Refusal(
        "projection_too_large",
        f"the projection of H onto its symmetric part needs {_size(n_bytes)} at its peak, "
        f"more than the {_size(budget)} that max_bytes= allows, on a grid of "
        f"{' x '.join(str(c) for c in counts)} = {n_points} k-points at "
        f"{int(n_states)} states. The symmetrized hoppings can reach "
        f"{R_box.tolist()} cells along the periodic directions, which, read alike along "
        f"directions the operations mix, takes a grid of at least "
        f"{[int(c) for c in least]} divisions. "
        f"`realized_group(..., max_bytes={runs})` runs it.",
        value=float(n_bytes), threshold=float(budget),
        context={"n_bytes": float(n_bytes), "max_bytes": float(budget),
                 "k_grid": tuple(counts), "n_points": n_points,
                 "n_states": int(n_states), "R_max": R_max.tolist(),
                 "R_box": R_box.tolist()})

# --------------------------------------------------------------------------
# the operators
# --------------------------------------------------------------------------

def _provisional_rep(model, group, assignment, validated, *, tol, scale, build_kw):
    r"""The representation a projection averages with, from a provisional build.

    :func:`.._build._build_rep` is run with two gates loosened: its covariance
    gate to ``tol``, and its internal-factor gate to ``tol / scale``, or ten
    times the worst measured residual over ``scale`` where that is larger,
    never below the build's own default.  Its matrices are assembled from
    discrete signs, so where every multiplet is one orbital they carry none of
    the model's noise.  A wider multiplet's block is the model's own, and the
    build returns the exact representation nearest those blocks.

    ``model``, ``group``, ``assignment`` and ``validated`` are the model, its
    group, the orbitals' assignment to its sites and the measurement of the
    operations at ``tol``, an energy in the units of ``H``.  ``scale`` is
    ``max|H|`` about the mean on-site energy.  ``build_kw`` holds the build's
    other keywords; an ``internal_tol`` there stands as given.  Where the build
    returns ``rep_not_unique`` or ``rep_search_too_large``, both orientation
    branches are built, and ``orientation_branch=1`` is taken where they agree,
    up to a phase per operation, to the build's own internal gate.  Returns
    ``(rep, record)``: the representation, or the build's first refusal, and a
    mapping of ``tol``, ``composition`` (how far the blocks were from
    composing), ``branch_ambiguity`` (``None`` where no branch was built) and
    ``covariance`` (the largest covariance residual of the matrices), empty
    beside a refusal.
    """
    kept = [int(i) for i in validated._indices]
    worst = max((float(validated.residuals.get(i, 0.0)) for i in kept), default=0.0)
    internal = max(_INTERNAL_GATE, 10.0 * worst / max(scale, 1e-30),
                   float(tol) / max(scale, 1e-30))
    kw = dict(build_kw)
    kw.setdefault("internal_tol", internal)
    kw.setdefault("_returned", False)
    rep = _build_rep(model, group, assignment, validated=validated, tol=float(tol), **kw)
    ambiguity = None
    if (not rep and getattr(rep, "gate", None) in ("rep_not_unique",
                                                   "rep_search_too_large")
            and "orientation_branch" not in kw):
        first = rep
        branches = {}
        for branch in (1, -1):
            got = _build_rep(model, group, assignment, validated=validated,
                             tol=float(tol), orientation_branch=branch, **kw)
            if not got:
                return first, {}
            branches[branch] = got
        shared = [i for i in branches[1]._indices if i in branches[-1].matrices]
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
                 "covariance": float(max(rep.numerical.residuals.values(), default=0.0))}

def _projection_operators(model, group, assignment, validated, *, tol, build_kw):
    r"""The representation a projection of this model is built with, and its record.

    ``model``, ``group``, ``assignment``, ``validated``, ``tol`` and
    ``build_kw`` are as :func:`_provisional_rep` takes them.  Its ``scale`` is
    read here: ``max|H|`` about the mean on-site energy, on eight fixed random
    k-points.  Returns what :func:`_provisional_rep` returns.  Whether an
    operation is exact or near is read on these matrices' covariance
    (:func:`_classify_operations`), and the average is taken with them
    (:func:`_symmetrized`).
    """
    mu = _bloch.onsite_mean(model)
    probe = np.random.default_rng(1729).uniform(-0.5, 0.5, size=(8, int(group._dim_k)))
    scale = _bloch.magnitude(_bloch.hamiltonian(model, probe), mu)
    return _provisional_rep(model, group, assignment, validated, tol=tol, scale=scale,
                            build_kw=build_kw)

def _exact_representation(group, D):
    r"""``D`` made an exact representation of ``group``.

    ``D`` maps each operation's index to its matrix on the states.  Returns the
    same mapping, with the exact representation nearest ``D``
    (:func:`.._matrices._nearest_representation`): each matrix moves by the
    composition defect it had and no more.  Returns ``D`` itself where no
    nearest representation is found.  The group average is a projection only
    with matrices that compose by the group's table.
    """
    position = {int(s.ind): p for p, s in enumerate(group._symmetries)}
    prod = group._get_product_table(get_diff=False)
    by_position = {position[i]: m for i, m in D.items()}
    nearest = _nearest_representation(by_position, list(by_position), group._symmetries, prod)
    if nearest is None:
        return D
    return {i: nearest[position[i]] for i in D}

def _with_centers(model, orb, *, check=True, n_check=_bloch.CHECK_K,
                  seed=_bloch.CHECK_SEED):
    r"""The model with its orbital centers at ``orb``, its spectrum measured unchanged.

    ``orb`` holds the new centers in reduced coordinates, a row per orbital.
    The lattice, on-site terms and hopping table are ``model``'s, row for row.
    In pythtb's convention I the move is the k-dependent gauge transformation
    ``diag(e^{−2πi k·Δ})``, ``Δ`` the move of each center, which leaves every
    eigenvalue where it was.  With ``check``, the two spectra are compared at
    ``n_check`` random ``k`` drawn from ``seed``.  Returns
    ``(model, residual)``, the largest change of an eigenvalue, ``0.0`` where
    nothing was measured.  Raises :class:`..refusal.InternalError` where an
    eigenvalue moved by more than ``1e-9`` of the largest one.
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
        Ea = np.linalg.eigvalsh(np.asarray(_bloch.hamiltonian(model, k)))
        Eb = np.linalg.eigvalsh(np.asarray(_bloch.hamiltonian(out, k)))
        scale = float(np.abs(Ea).max()) or 1.0
        resid = float(np.abs(Ea - Eb).max())
        if resid > 1e-9 * scale:
            worst = float(np.abs(orb - np.asarray(model.orb_vecs, dtype=float)).max(
                initial=0.0))
            raise InternalError(
                f"moving the orbital centers by at most {worst:.3g}, in reduced coordinates, "
                f"changed the spectrum:max_k |E_new - E_old| = {resid:.3g} on a scale of "
                f"{scale:.3g}, over {n_check} random k. Report it, with the model and the "
                "call.")
    return out, resid

# --------------------------------------------------------------------------
# the average, and the way back to a hopping list
# --------------------------------------------------------------------------

def _same_part(parts, model):
    r"""Which pairs of states lie in one coupled part.

    ``parts`` holds each coupled part's orbitals, together every orbital of
    ``model``.  Returns a boolean array over pairs of states, true where states
    ``a`` and ``b`` lie in one part; ``None`` where ``parts`` holds fewer than
    two.  Raises :class:`..refusal.InternalError` where a state lies in none.
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
            f"the coupled parts of the crystal leave states "
            f"{np.flatnonzero(part_of < 0).tolist()} in none of them. Report it, with the "
            "model and the call.")
    return part_of[:, None] == part_of[None, :]

def _parts_coupling(model, parts) -> float:
    r"""The largest entry of ``H(k)`` between two coupled parts, in the units of ``H``.

    ``model`` is the model and ``parts`` as :func:`_same_part` takes it.  The
    entries are read on the eight k-points the projection's scale is read on
    (:func:`_projection_operators`).  Returns ``0.0`` where there are fewer
    than two parts.
    """
    same = _same_part(parts, model)
    if same is None:
        return 0.0
    probe = np.random.default_rng(1729).uniform(-0.5, 0.5, size=(8, int(model.lattice.dim_k)))
    H = np.asarray(_bloch.hamiltonian(model, probe))
    return float(np.abs(H[:, ~same]).max(initial=0.0))

def _as_held(X, held):
    r"""``X`` with what the sectors read at the declared precision keep apart held apart.

    ``X`` is a matrix on the states, or a stack of them.  ``held`` is the
    mapping :func:`.._labeling._held_sets` returns: ``"copies"``, each set of
    identical copies as ``(basis, copies, dim, orbitals)``, and ``"shares"``,
    each set of sectors splitting multiplets as ``(bases, orbitals)``.
    Returns ``X`` with every entry between a set and the rest, between two
    copies or between two sectors of a set left out, and each copy carrying
    the copies' mean.  This is the part of ``X`` that commutes with every
    matrix the sectors conserve, and it commutes with the group average.
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

def _held_off(model, held) -> dict:
    r"""The largest entry of ``H(k)`` each kind in ``held`` leaves out.

    ``model`` is the model whose ``H(k)`` is read, and ``held`` as
    :func:`_as_held` takes it, or ``None``.  Each kind is read on its own, on
    the k-points :func:`_parts_coupling` reads.  Returns
    ``{"copies": ..., "shares": ...}`` in the units of ``H``, ``0.0`` for a
    kind ``held`` does not hold.
    """
    out = {"copies": 0.0, "shares": 0.0}
    if not held:
        return out
    probe = np.random.default_rng(1729).uniform(-0.5, 0.5, size=(8, int(model.lattice.dim_k)))
    H = np.asarray(_bloch.hamiltonian(model, probe))
    for kind in out:
        if held.get(kind):
            alone = {kind: held[kind]}
            out[kind] = float(np.abs(H - _as_held(H, alone)).max(initial=0.0))
    return out

def _average_on_grid(model, group, assignment, ops, D, kpts, divisions, *,
                     bloch_tol, work, apart=None, held=None):
    r"""The group average of ``H`` on the transform grid.

    For every operation and grid point, ``U(g;k) H(k) U(g;k)†``, with ``H(k)*``
    for an operation combined with time reversal, is added at the image point
    ``gk``, folded back onto the grid, which :func:`_divisions` makes a grid
    point.  ``H`` is evaluated once, and each operation permutes the grid.

    ``model``, ``group`` and ``assignment`` are the model, its group and the
    orbitals' assignment to its sites.  ``ops`` maps each operation's index to
    the operation, and ``D`` maps it to the operation's matrix on the
    orbitals.  ``kpts`` is the grid in reduced k and ``divisions`` its
    divisions.  ``bloch_tol`` is the tolerance
    :func:`..symmetry.bloch_operator` holds the centers to.  ``work`` is the
    working space in bytes, over whose slices of the grid ``H``, the operators
    and their products are formed.  ``apart`` (:func:`_same_part`), where
    given, leaves out what ``H`` holds between the coupled parts, and ``held``
    (:func:`_as_held`), where given, holds apart what the sectors keep apart;
    both commute with the average.  Returns ``(H_sym, moved, scale)``: the
    average on the grid, ``max_k |H_sym(k) − H(k)|`` and ``max_k |H(k)|``, in
    the units of ``H``.  Raises :class:`..refusal.InternalError` where an
    operation does not carry the grid onto itself, which :func:`_divisions`
    rules out.
    """
    n_points = int(kpts.shape[0])
    counts = tuple(int(c) for c in divisions)
    n = np.asarray(counts, dtype=float)
    H = _GridHamiltonians(model, kpts, max_bytes=work).full()
    acc = np.zeros_like(H)
    step = _slice_points(work, _POINT_MATRICES * 16.0 * float(H.shape[-1]) ** 2, n_points)
    for ind, sym in ops.items():
        kp = group._k_action_of(sym)(kpts)
        mm = np.rint(kp * n)
        if np.abs(kp * n - mm).max() > 1e-7:      # pragma: no cover: guarded by design
            raise InternalError(
                f"operation #{ind} of {group.name} #{group.number_str} does not carry the "
                f"{' x '.join(str(c) for c in counts)} grid of k-points onto itself, though "
                "the grid's divisions are chosen so that every operation does. Report it, "
                "with the model and the call.")
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

def _average_at(model, group, assignment, ops, D, k, *, bloch_tol, apart=None,
                held=None):
    r"""The group average of ``H`` at any k, and the covariance there.

    At each point the average over the operations of
    ``U(g;k)† H(Rk) U(g;k)`` for ``g = {R|t}``, and of its complex conjugate,
    with ``H(−Rk)`` in place of ``H(Rk)``, for ``g`` combined with time
    reversal, is the ``H_sym`` :func:`_average_on_grid` gives, with no grid.
    ``model``, ``group``, ``assignment``, ``ops``, ``D``, ``bloch_tol``,
    ``apart`` and ``held`` are as :func:`_average_on_grid` takes them, and
    ``k`` holds the points in reduced k.  Returns ``(H_sym, residue)``: the
    average at ``k``, and the largest entry over the operations and the points
    of ``U H(k) U† − H(Rk)``, or of ``U H(k)* U† − H(−Rk)``, the covariance of
    the model against these operators, in the units of ``H``.
    """
    k = np.atleast_2d(np.asarray(k, dtype=float))
    evaluate = _bloch.evaluator(model)
    H = np.asarray(evaluate(k))
    acc = np.zeros_like(H)
    residue = 0.0
    for ind, sym in ops.items():
        kout = group._k_action_of(sym)(k)
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
    r"""The hopping coefficients of ``Hsym``, by an exact inverse Fourier transform.

    In pythtb's convention I, ``H_ij(k) = Σ_R t_ij(R) e^{2πi k·(R + τ_j − τ_i)}``
    is a finite Fourier series.  On a grid of ``N = Π_a n_a`` points starting
    at ``k = 0``, with ``n_a > 2·R_max,a`` divisions along each periodic
    direction, the inverse

    .. math:: t_{ij}(R) = \frac{1}{N}\sum_k H_{ij}(k)\,
              e^{-2\pi i\,k\cdot(R + \tau_j - \tau_i)}

    is exact.  ``Hsym`` is the average on the grid ``kpts`` of ``divisions``,
    and is overwritten.  ``τ`` is ``model``'s own ``orb_vecs`` along the
    periodic directions ``per``, which pythtb built ``H(k)`` with.  The phases
    are applied over slices of the grid within ``work`` bytes.  Returns
    ``(t, R)``: ``t`` of shape ``(N, nstate, nstate)``, sharing ``Hsym``'s
    memory, and the integer lattice vectors ``R`` of shape ``(N, d)``, each
    component folded into ``[-n_a/2, n_a/2)``.
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
    """The coefficients ``t`` arranged by spin block.

    ``t`` has shape ``(N, norb·nspin, norb·nspin)``, spin the fast index, for
    ``norb`` orbitals of ``nspin`` spin states each.  Returns it with shape
    ``(N, norb, norb, nspin, nspin)``.
    """
    return np.transpose(t.reshape(t.shape[0], norb, nspin, norb, nspin), (0, 1, 3, 2, 4))

def _block_norms(t, norb, nspin, step):
    """The Frobenius norm of each spin block ``t_ij(R)``.

    ``t``, ``norb`` and ``nspin`` are as :func:`_blocks` takes them, read
    ``step`` lattice vectors at a time.  Returns an array of shape
    ``(N, norb, norb)``.
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
    r"""The norm below which rebuilt coefficients are dropped, never splitting an orbit.

    Partners of one orbit carry the same block norm, so a cut with a gap under
    it splits none.  The block norms are walked upward from zero, never above
    ``relative`` times the largest, and the cut is at the last one followed by
    a jump of more than a factor of ten.  ``norms`` is the array of
    :func:`_block_norms`, read ``step`` lattice vectors at a time, all at once
    where ``step`` is ``None``.  Returns ``(threshold, gap)``: the cut and the
    ratio across it.  Where there is no such jump the threshold is ``0.0`` and
    nothing is dropped; the gap is then the least norm above the bound over
    the least nonzero one below it, and ``inf`` where none lies below it.
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
    """The terms a pythtb model is written from, read off the coefficients.

    ``t`` and ``R`` are as :func:`_inverse_transform` returns them, for
    ``model`` with periodic directions ``per``.  ``keep`` is a boolean array
    over ``(R, i, j)``, true for a block above :func:`_drop_threshold`'s cut.
    Of each conjugate pair only the half pythtb does not add itself is kept,
    with the ``R = 0`` diagonal as the on-site terms.  ``step`` lattice vectors
    are read at a time.  Returns ``(onsite, amplitudes, i, j, R, hermiticity)``,
    every array a copy: ``R`` the full lattice vectors of ``model``, and
    ``hermiticity`` the largest entry of ``t_ij(R) − t_ji(−R)†``, spin blocks
    taken whole, in the units of ``H``.
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
    # positive is written, and the upper triangle at R = 0; pythtb adds the
    # other half.  Every (R, i, j) is one coefficient of the transform, so no
    # term is written twice.
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
    """A pythtb model on ``model``'s lattice and centers, from its terms.

    ``ons`` are the on-site terms, and ``amps``, ``ii``, ``jj`` and ``Rfull``
    each hopping's amplitude, orbitals and lattice vector, as
    :func:`_kept_terms` returns them.  Returns the model.
    """
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

def _symmetrize(model, group, assignment, validated, *, tol, k_grid=None, params=None,
                max_passes=3, verify=True, verify_nk=None, verify_tol=None,
                max_bytes=1e9, characters=None, site_multiplets=None, spin=None,
                max_candidates=2 ** 20):
    r"""A near-symmetric model made exactly symmetric, and how far it moved.

    ``H`` is averaged over the group's operations, those combined with time
    reversal among them, as ``U(g;k) H(k) U(g;k)†`` on a grid of k-points, and
    the average is written back as a hopping list by an exact inverse Fourier
    transform.  The average moves ``H`` by about the residuals of the
    operations obeyed only within ``tol``.  :func:`..groups.realized_group`
    runs the same pass itself, :func:`_symmetrized`, wherever an operation is
    near, or a site multiplet, a coupling of parts of the crystal, or a set of
    identical copies or of sectors holds only within the declared precision.

    Parameters
    ----------
    model : pythtb.TBModel
        The model the operations were measured on, ``r.model``: for a layer,
        its slab presentation.
    group, assignment, validated : GroupData, OrbitalAssignment, ValidatedGroup
        ``r.group``, ``r._assignment`` and ``r.numerical.validated`` of a call
        measured at ``tol``.
    tol : float
        The declared tolerance, in the units of ``H``, the one ``validated``
        was measured at; for numbers good to a relative ``ε``, about
        ``10·ε·max|H|``.
    k_grid : int, default None
        The divisions of the grid along every periodic direction.  ``None``
        takes ``2·(box + 1) + 1`` along each, ``box`` the reach the group
        carries the hoppings to (:func:`_divisions`).
    params : dict, default None
        The scalar value of each parameter of the model, keyed by name:
        ``{"lam": 0.3}``.
    max_passes : int, default 3
        The most averaging passes.  A further pass runs only while the
        covariance residual of the averaged model is above rounding and the
        last pass moved ``H`` by more than rounding.
    verify : bool, default True
        Whether the operations are measured again on the model returned.
    verify_nk : int, default None
        The divisions of the mesh they are measured on; ``None`` is the mesh
        ``validated`` was measured on.
    verify_tol : float, default None
        The tolerance they are measured at, in the units of ``H``; ``None`` is
        ``1e-8·max|H|``.
    max_bytes : float, default 1e9
        The memory the average may take at its peak, in bytes.
    characters, site_multiplets, spin : default None
        The declarations, as :func:`..groups.realized_group` takes them.
    max_candidates : int, default 2**20
        The largest search for the representation's discrete choices that is
        run.

    Returns
    -------
    (pythtb.TBModel, Symmetrization) or Refusal
        The symmetrized model, its parameters evaluated at ``params``, and the
        record of the average.

    Refusals
    --------
    ``symmetrize_broken_operation``
        an operation beyond ``tol``, which the model does not obey.
    ``symmetrize_defect_exceeds_tol``
        the average moved ``H`` further than ``tol``.
    ``symmetrize_range_not_closed``
        the symmetrized hoppings reach beyond the box the group carries the
        hoppings to.
    ``positions_not_exactly_symmetric``
        the orbital centers miss the operations, by more than the operators
        carry them or by enough to keep ``H`` from being made exactly
        symmetric; :func:`_symmetrize_positions`, or a declared precision,
        moves them first.
    ``projection_too_large`` / ``detector_too_large``
        past ``max_bytes``; the reason gives the value that runs it.

    A refusal of the representation's own build, ``rep_not_unique`` among them,
    is returned as it is.

    Raises
    ------
    ValueError
        A malformed argument, before anything is measured: a name in
        ``params`` that is not a parameter of the model, a parameter given no
        value or a value that is not a number, a ``tol``, ``max_bytes`` or
        ``max_candidates`` that is not positive, or a malformed declaration.
        Then, a ``k_grid`` coarser than the default grid along some direction.
    ..refusal.InternalError
        The hopping list returned misses the average, which it reproduces
        exactly.
    """
    model = _bloch.bound(model, params, call="_symmetrize")
    return _symmetrized(model, group, assignment, validated, tol=tol, k_grid=k_grid,
                        max_passes=max_passes, verify=verify,
                        verify_nk=verify_nk, verify_tol=verify_tol, max_bytes=max_bytes,
                        characters=characters, multiplets=site_multiplets, spin=spin,
                        max_candidates=max_candidates, _call="_symmetrize")

def _symmetrized(model, group, assignment, validated, *, tol, k_grid=None,
                 max_passes=3, verify=True, verify_nk=None, verify_tol=None,
                 max_bytes=1e9, _parts=None, _held=None, _centers_declared=False,
                 _operators=None, _call="realized_group", **build_kw):
    """The pass :func:`_symmetrize` runs, on a model whose parameters are bound.

    ``model``, ``group``, ``assignment``, ``validated``, ``tol``, ``k_grid``,
    ``max_passes``, ``verify``, ``verify_nk``, ``verify_tol`` and ``max_bytes``
    are as :func:`_symmetrize` takes them.  ``build_kw`` holds the provisional
    build's keywords by name, such as ``characters``, ``multiplets``,
    ``spin`` and ``max_candidates``.  :func:`..groups.realized_group` calls it
    on its own records and hands in the rest.  ``_parts`` is
    ``{"orbitals", "coupling"}``, the coupled parts of the crystal and the
    largest entry of ``H`` between them.  ``_held`` is ``{"sets", "off"}``,
    what the sectors keep apart (:func:`.._labeling._held_sets`) and what
    :func:`_held_off` measured it leaves out.  ``_centers_declared`` says
    whether ``precision=`` or ``position_tol=`` was declared; where it was, a
    failed check is never refused as the centers' miss,
    ``positions_not_exactly_symmetric``, and its own verdict stands.
    ``_operators`` is ``(exact_gate, (rep, record), classes)``, the exact
    gate, the provisional build of :func:`_projection_operators` and the
    classes read on it, used as they are.  ``_call`` is the public call's
    name, which the ``ValueError`` raised for ``tol``, ``k_grid``,
    ``max_bytes`` or ``max_candidates`` opens with.  Returns and refuses as
    :func:`_symmetrize` does, and raises as it does, ``params`` aside.
    """

    tol = float(tol)
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError(f"{_call}(): tol={tol!r} is not a positive number. Pass the "
                         "tolerance the operations were measured at, in the units of H.")
    budget = _memory_budget(max_bytes, _call)
    if "max_candidates" in build_kw:
        build_kw["max_candidates"] = _search_ceiling(build_kw["max_candidates"], _call)
    # what the provisional build is handed, read as grammar before anything is
    # measured: a malformed `characters=` or `multiplets=` raises here
    _declaration_grammar(int(model.norb), characters=build_kw.get("characters"),
                         multiplets=build_kw.get("multiplets"),
                         spin=_resolve_spin(build_kw.get("spin"), model),
                         site_of=assignment.site_of_orbital)
    per = tuple(int(d) for d in group.periodic_dirs)
    dim_k = int(group._dim_k)
    syms_by_ind = {int(s.ind): s for s in group._symmetries}

    # 1. The classification, and what is refused before anything is built
    # against the exact gate of this model and operation set, measured once: the
    # detection record carries neither the number of states nor max|H|
    if _operators is not None:
        exact_gate, handed, classes = _operators
    else:
        exact_gate = _exact_gate(model, group, validated, build_kw)
        handed, classes = None, _classify_operations(validated, tol, exact_gate=exact_gate)
    broken = sorted(i for i, c in classes.items() if c == "broken")
    if broken:
        # the measurement holds an infinite residual, or none, for an operation it
        # found no matrix for: there is no number to quote for it
        measured = {i: float(validated.residuals[i]) for i in broken
                    if np.isfinite(float(validated.residuals.get(i, np.inf)))}
        unmeasured = [i for i in broken if i not in measured]
        worst = max(measured.values()) if measured else None
        found = []
        if measured:
            found.append(f"Operations {sorted(measured)} reach a covariance residual of "
                         f"{worst:.3g} against tol={tol:.3g}.")
        if unmeasured:
            found.append(f"For operations {unmeasured} the measurement found no matrix "
                         "the Hamiltonian is covariant under, so no covariance residual.")
        return Refusal(
            "symmetrize_broken_operation",
            f"{len(broken)} of the {len(classes)} operations handed in are broken at the "
            "declared tolerance. The operations are counted "
            f"{_counted_as(group.spinor)}. {' '.join(found)} Pass a validated record that "
            "lists only the operations holding at tol, or a larger tol where the defect is "
            "numerical.",
            value=worst, threshold=tol,
            context={"broken": tuple(broken), "unmeasured": tuple(unmeasured),
                     "classes": classes, "tol": tol, "exact_gate": exact_gate})
    # ... and the centers, which the operators are built on.  The average is
    # taken with Bloch operators that carry each center onto its image, checked
    # to `bloch_operator`'s own tolerance; on centers the operations miss, the
    # operators do not compose as the group does, and the "average" is no
    # projection: it writes each hopping onto a lattice vector that is not one
    # and moves what was exact.  A projection of the hoppings cannot reach a
    # defect in the centers at all, so such centers are refused here, with the
    # route that moves them.
    pos_defect = float(_position_defect(model, group, assignment,
                                        {int(i): syms_by_ind[int(i)]
                                         for i in validated._indices}))
    if pos_defect > _BLOCH_TOL:
        return Refusal(
            "positions_not_exactly_symmetric",
            f"the orbital centers miss the operations handed in by {pos_defect:.3g}, in "
            f"reduced coordinates, against the {_BLOCH_TOL:.0e} the projection allows. "
            "Declare the model's precision with `precision=`, or a tolerance for the "
            "centers alone with `position_tol=`.",
            value=pos_defect, threshold=_BLOCH_TOL,
            context={"position_defect": pos_defect, "bloch_tol": _BLOCH_TOL})

    # 2. The operators, and the classes read on them
    rep, prov = (handed if handed is not None else
                 _projection_operators(model, group, assignment, validated, tol=tol,
                                       build_kw=build_kw))
    if not rep:
        if rep.gate == "rep_not_covariant":
            built = rep.context.get("indices", validated._indices)
            own = _centers_account_for(
                model, group, assignment, {int(i): syms_by_ind[int(i)] for i in built},
                failure=rep.value, gate=rep.threshold,
                check="the covariance of the representation the projection is built with",
                declared=_centers_declared)
            if own is not None:
                return own
        return rep
    # exact or near is read on each operation's own matrix's covariance in this
    # representation, the matrix the average is taken with: a basis turned inside
    # a site is exact to the detector and near to the matrix the tables' spin lift
    # and the multiplets anchor
    anchored = {int(i): float(v) for i, v in rep.numerical.residuals.items()}
    if handed is None:
        classes = _classify_operations(validated, tol, exact_gate=exact_gate,
                                       anchored=anchored)
    indices = tuple(int(i) for i in rep._indices)
    D = _exact_representation(group, {int(i): np.asarray(rep.matrices[int(i)])
                                      for i in indices})
    bloch_tol = _BLOCH_TOL
    # The parts of the crystal read apart, where several were: what H holds
    # between them is below the declared precision, and the average leaves it out;
    # and what the sectors read at that precision keep apart, held apart
    apart = _same_part((_parts or {}).get("orbitals"), model)
    held_apart = (_held or {}).get("sets") or None
    not_represented = tuple(sorted({int(i) for i in validated._indices} - set(indices)))
    ops = {i: syms_by_ind[i] for i in indices}
    prov["n_operations"] = len(ops)
    prov["position_defect"] = pos_defect
    prov["bloch_tol"] = float(bloch_tol)
    # What every refusal below says when the centers carry a defect of their own
    # above rounding, though within the operators' own tolerance: a projection
    # of the hoppings reaches nothing in them.  `EXACT_GATE` is the exact
    # pipeline's own rounding scale, used only to decide whether there is a
    # number worth quoting.  It gates nothing: the defect is in reduced
    # coordinates, and this number is in the units of H.
    pos_note = ("" if pos_defect <= EXACT_GATE else (
        f" The orbital centers miss the group's operations by {pos_defect:.3g}, in "
        "reduced coordinates. Declare the model's precision with `precision=`, or a "
        "tolerance for the centers alone with `position_tol=`, before reporting it."))

    # 3. The transform grid
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
                f"{_call}(): k_grid={int(k_grid)} is too coarse for this model. Its "
                f"hoppings reach {R_max.tolist()} cells along the periodic directions and "
                f"its symmetrized hoppings can reach {R_box.tolist()}, which, read alike "
                f"along directions the operations mix, takes a grid of at least "
                f"{[int(c) for c in need]} divisions. Pass k_grid of at least "
                f"{int(need.max())}, or leave it out.")
    # What the projection holds at its peak, against the memory it may take: the
    # Hamiltonian and its group average on the grid, and one k-point of the
    # arithmetic streamed beside them, since every model it evaluates carries at
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

    # 4. The average, the transform, and the model
    current, passes, per_pass, floor = model, 0, [], 0.0
    checks: dict[str, Any] = {}
    step = _slice_points(work, _POINT_MATRICES * matrix, n_points)
    for _ in range(max(1, int(max_passes))):
        Hsym, moved, scale = _average_on_grid(current, group, assignment, ops, D, kpts,
                                              divisions,
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
        # For the representation's operators the range box is a bound on where
        # the symmetrized model can have hoppings, so anything outside it is
        # rounding and the gate is at rounding (measured 1e-17..1e-16 on every
        # model built from exact numbers).  The centers are carried by the
        # operations to their own tolerance, checked above, so nothing carries a
        # hopping to a lattice vector that is not one.
        gate = _RANGE_RELATIVE
        if out_box > gate:
            # the average is a finite sum of hoppings only where its operators
            # carry the centers exactly, and each coefficient beyond the box is
            # a mean over the grid of what the centers' miss puts into H
            own = _centers_account_for(current, group, assignment, ops,
                                       failure=out_box * biggest, gate=gate * biggest,
                                       check="the symmetrized Hamiltonian's hoppings beyond its range",
                                       declared=_centers_declared)
            if own is not None:
                return own
            return Refusal(
                "symmetrize_range_not_closed",
                f"the symmetrized Hamiltonian has terms beyond the {box.tolist()} cells "
                "along the periodic directions that the group carries its hoppings to: the "
                f"largest of them is {out_box:.3g} times the largest term of the "
                "symmetrized Hamiltonian, against "
                f"{gate:.3g}, on a "
                f"{' x '.join(str(c) for c in divisions)} grid of k-points. Report it, with "
                "the model and the call." + pos_note,
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
        # the identity that says the hopping list is H_sym, measured off the
        # grid the transform was taken on, where it holds by construction
        target, _ = _average_at(current, group, assignment, ops, D, k_random,
                                bloch_tol=bloch_tol,
                                apart=apart, held=held_apart)
        Hnew = np.asarray(_bloch.hamiltonian(out, k_random))
        checks["k_identity"] = float(np.abs(Hnew - target).max())
        if checks["k_identity"] > gate * (scale if scale > 0.0 else 1.0):
            # The transform is exact by theorem once the range box is closed, and
            # the box has been measured closed at the same relative size.  So
            # this is the rebuild's own arithmetic going wrong, not a statement
            # about the model, and it raises, unless the centers' miss accounts
            # for it: then the average is no finite sum of hoppings to rebuild.
            own = _centers_account_for(current, group, assignment, ops,
                                       failure=checks["k_identity"],
                                       gate=gate * (scale if scale > 0.0 else 1.0),
                                       check="the symmetrized Hamiltonian rebuilt from its "
                                             "hoppings",
                                       declared=_centers_declared)
            if own is not None:
                return own
            raise InternalError(
                f"the hopping list written for the symmetrized Hamiltonian differs from it "
                f"at k-points off the transform grid: max_k |H_new(k) - H_sym(k)| = "
                f"{checks['k_identity']:.3g} on {k_random.shape[0]} random k, against "
                f"{gate * (scale if scale > 0.0 else 1.0):.3g}. Beyond the cells the group "
                f"carries hoppings to, no term exceeds {out_box:.3g} times the largest. "
                "Report it, with the model and the call.")
        _, residue = _average_at(out, group, assignment, ops, D, k_random,
                                 bloch_tol=bloch_tol, apart=apart, held=held_apart)
        checks["covariance_residue"] = float(residue)
        floor = covariance_gate(len(ops), int(model.nstate), scale)
        current = out
        # Two ways to be done, and both are measurements: the model is symmetric
        # under the operators it was built with, or the model stopped moving.
        # The second matters where the first cannot be reached at all: a
        # position defect leaves an O(delta) residue no number of passes
        # touches, and iterating a fixed point only costs arithmetic.
        if residue <= floor or per_pass[-1] <= floor:
            break

    # 5. How far the model moved, in total
    # both Hamiltonians over slices of the grid, no grid of either held, and
    # the scale about the mean on-site energy, which a constant shift moves
    mu = _bloch.onsite_mean(model)
    before = _bloch.evaluator(model)
    after = _bloch.evaluator(current)
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
            f"relative), more than the declared tol = {tol:.3g}. The representation the "
            f"average was taken with carries H(k) onto itself to within "
            f"{prov.get('covariance', float('nan')):.3g}. Where that is close to tol, the "
            "tolerance is too tight: declare a larger precision. Where it is far below "
            "tol, report it, with the model and the call."
            + pos_note,
            value=defect, threshold=tol,
            context={"classes": classes, "operators": prov, "passes": int(passes),
                     "per_pass": tuple(per_pass), "scale": scale,
                     "position_defect": float(pos_defect)})

    # 6. The verification: the exact pipeline may now run
    verification: dict[str, Any] = {}
    if verify:
        # with nothing asked, the gate a model of exact numbers is held to,
        # relative to the magnitude of H as the measurement's own is
        vtol = (float(_default_of(_measure._validate_group, "tol")) * (scale if scale > 0.0 else 1.0)
                if verify_tol is None else float(verify_tol))
        vcut = (float(_default_of(_measure._validate_group, "tol"))
                if verify_tol is None and scale > 0.0 else None)
        nk = (int(verify_nk) if verify_nk is not None
              else max(2, int(round(float(validated.n_kpoints) ** (1.0 / dim_k)))))
        again = _measure._validate_group(current, group, nk=nk, tol=vtol, cut=vcut,
                                        window=None if vcut is None else eigen_window(vcut),
                                        keep_intertwiners=False, max_bytes=budget,
                                        assignment=assignment,
                                        _floor=None if vcut is None else vtol)
        if isinstance(again, Refusal):
            return again                    # the detector's memory limit, as it came
        kept_now = {int(i) for i in again._indices}
        verification = {
            "tol": vtol, "nk": nk, "n_kept": len(again._indices),
            "n_operations": len(again._indices) + len(again.discrepancies),
            "worst_residual": float(max((again.residuals[i] for i in again._indices),
                                        default=float("nan"))),
            "near_kept": all(i in kept_now for i in
                             (i for i, c in classes.items() if c == "near")),
            "broken": tuple(int(i) for i in again.broken), "note": ""}
        if not verification["near_kept"]:
            # The re-detection, which reads none of the operators the average
            # was taken with, is a *second opinion*.  The first opinion is
            # `checks["covariance_residue"]`, measured directly with those
            # operators, and it is the one that says whether the projection
            # worked.
            verification["note"] = (
                f"measured again, the symmetrized model did not keep every near operation "
                f"at {vtol:.3g}, while "
                f"the covariance of the symmetrized model against the operators it was "
                f"built with measures {checks.get('covariance_residue', float('nan')):.3g}, "
                "the measurement that says whether the projection worked.")

    # Why the projection ran, where no operation says it: multiplets measured
    # within the declared tolerance only, as the multiplet measurement handed in
    # records it
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
        tol=tol, _exact_gate=exact_gate, _classes=classes,
        residuals={int(i): _residual_read(validated, anchored, i, missing=np.nan)
                   for i in classes},
        defect=defect, scale=scale, _not_represented=not_represented,
        _grid={"n": tuple(divisions), "n_points": int(kpts.shape[0]), "dim_k": dim_k,
               "R_max": tuple(int(x) for x in R_max),
               "R_box": tuple(int(x) for x in R_box)},
        checks=SymmetrizationChecks(**dict(checks, per_pass=tuple(per_pass))),
        _operators=prov,
        verification=(SymmetrizationVerification(**verification) if verification else None),
        passes=int(passes), _machine_floor=float(floor),
        site_multiplets=SymmetrizationSiteMultiplets(**held) if held else None,
        parts=SymmetrizationParts(**uncoupled) if uncoupled else None,
        copies=SymmetrizationCopies(**as_copies) if as_copies else None,
        shares=SymmetrizationShares(**as_shares) if as_shares else None,
        _spinful=bool(group.spinor))
    return current, record

# --------------------------------------------------------------------------
# the other half of the model: the orbital centers
# --------------------------------------------------------------------------

def _symmetrize_positions(model, group, assignment, *, max_move=1e-5, params=None):
    r"""The model with its orbital centers moved onto its group's nearest positions.

    The centers are fitted about their own origin, a shift of all of them
    together being no displacement, and a common shift within ``max_move`` is
    then removed, so that they come back in the model's own frame.  The move
    is a k-dependent gauge transformation, and the spectrum is measured
    unchanged.  :func:`..groups.realized_group` makes the same move, with
    ``precision=`` or ``position_tol=`` declared, before the cell's setting is
    read.

    ``model`` is the pythtb model whose centers are moved, and ``group`` and
    ``assignment`` are ``r.group`` and ``r._assignment`` of a call on it.  A
    group read off the noisy centers themselves is written about an origin
    their noise placed, so the moved model is identified again before it is
    read in a group.  ``max_move`` is how far a center may move and still be
    the same orbital, in reduced coordinates, a shift of all of them together
    not counted, and how far that shift may reach and still be removed.
    ``params`` holds the scalar value of each parameter of the model, keyed by
    name: ``{"lam": 0.3}``.

    Returns ``(model, record)``: the model with its centers moved and its
    parameters evaluated at ``params``, and the record of the move.  The
    record holds ``defect_before`` and ``defect_after``, how far the centers
    miss the group's action before and after, in reduced coordinates;
    ``max_move``, the largest move of one center; ``offset``, the shift of
    them all together, each center returned being the one handed in plus its
    own move plus ``offset``; ``frame``, ``"model"`` where they are written in
    the model's own frame and ``"fitted"`` where they stay about the origin
    fitted to them; ``spectrum_residual``, how far the spectrum moved over
    ``n_check`` random ``k``, ``0`` where no center moved; and ``n_moved``.
    Returns the refusal ``positions_not_nearly_symmetric`` where a center
    would have to move farther than ``max_move``: the positions are a
    different set of centers, and the group they carry is the one to
    identify.  Raises ``ValueError`` for a name in ``params`` that is not a
    parameter of the model, a parameter given no value, or a value that is
    not a number.  Raises :class:`..refusal.InternalError` where the spectrum
    moved, or the group's operations miss the positions moved onto them,
    neither of which the move can do.

    Examples
    --------
    Kane-Mele with one center moved by ``1e-8`` (``moved``), read in the group
    of the exact model:

    >>> from pythtb.models import kane_mele
    >>> r = realized_group(kane_mele(0.0, -1.0, 0.3, 0.25), characters={0: "A1"})
    >>> fixed, rec = _symmetrize_positions(moved, r.group, r._assignment)
    >>> f"{rec['defect_before']:.1e}", rec["defect_after"] < 1e-14, rec["frame"]
    ('2.0e-08', True, 'model')
    """

    model = _bloch.bound(model, params, call="_symmetrize_positions")
    frame = _centers._assigned_sites(model, group, assignment)
    syms = group._symmetries
    unitary = [k for k, s in enumerate(syms) if not s.time_reversal]
    rotations = np.array([np.asarray(syms[k].rotation, dtype=float) for k in unitary])
    perm = np.asarray(assignment.permutations, dtype=int)[unitary]
    shift = np.asarray(assignment.lattice_shift, dtype=float)[unitary]
    rows = list(range(len(unitary)))
    x, trans, closure = _centers._symmetric_sites(frame, perm, shift, rotations, rows)
    if not closure <= _EXACT_CLOSURE:          # pragma: no cover: an assignment is an action
        raise InternalError(
            f"the unitary operations of {group.name} #{group.number_str} do not carry "
            f"the positions projected onto them, missing by {closure:.3g}. Report it, with "
            "the model and the call.")
    raw = np.asarray(model.orb_vecs, dtype=float)
    dim_r = int(model.lattice.dim_r)
    moves = _centers._moves_of(frame, x)
    worst = float(np.abs(moves).max(initial=0.0))
    before = _centers._frame_defect(frame, perm, rotations, trans, rows)
    if not within(worst, max_move):
        return Refusal(
            "positions_not_nearly_symmetric",
            f"an orbital center would have to move {worst:.3g}, in reduced coordinates, "
            "onto the nearest "
            f"positions the group carries, against max_move={float(max_move):.3g}. Read "
            "the group these centers carry with `realized_group`, or pass a larger "
            "max_move where that move is meant.",
            value=worst, threshold=float(max_move),
            context={"defect_before": float(before),
                     "n_moved": int(np.count_nonzero(np.abs(moves).max(axis=1) > 0.0))})
    delta = _centers._rational_offset(trans, rotations, rows, model.periodic_dirs)
    in_frame = delta is not None and within(float(np.abs(delta).max(initial=0.0)), max_move)
    final = x - delta if in_frame else x
    shifted = _centers._moves_of(frame, final)
    orb = raw + shifted[:, :dim_r]
    if np.array_equal(orb, raw):
        out, resid, measured = model, 0.0, 0
    else:
        (out, resid), measured = _with_centers(model, orb), _bloch.CHECK_K
    final_trans = np.array([final[perm[r, 0]] + shift[r, 0] - rotations[r] @ final[0]
                            for r in rows])
    offset = (0.0 - np.asarray(delta) * frame.scale)[:dim_r] if in_frame else np.zeros(dim_r)
    return out, {
        "defect_before": _centers._frame_defect(frame, perm, rotations, final_trans, rows),
        "defect_after": _centers._frame_defect(frame, perm, rotations, final_trans, rows,
                                              sites=final),
        "max_move": worst, "offset": tuple(float(c) for c in offset),
        "frame": "model" if in_frame else "fitted",
        "spectrum_residual": float(resid), "n_check": measured,
        "n_moved": int(np.count_nonzero(np.abs(orb - raw).max(axis=1) > 0.0))}
