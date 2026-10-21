r"""What a measurement of crystalline symmetry judges noise against, resolved once.

A model built from exact numbers is symmetric to rounding, and every judgement
made about it -- whether an operation holds, whether two orbitals form one
multiplet, whether a centre sits where the operations put it -- has an exact
default.  A model computed from a calculation is symmetric only to the precision
of its own numbers, and one declaration, ``precision=``, says what that is.  Each
judgement then reads a tolerance derived from it, in the units of what it
compares: a centre's in reduced coordinates, the covariance gate's in the units
of ``H``, a matrix read off ``H`` relative to its own scale, and the symmetry
finder's in the model's length units.  :class:`Tolerances` holds them, resolved
once where a model enters :func:`..groups.realized_group` and handed whole to
every step that judges noise.

Every number a judgement compares against is one of three kinds, and each is
named here.  A judgement of the **noise** the model's own data carries -- a
centre, a hopping, a matrix or a character read off ``H`` -- reads the
record: its exact default where nothing is declared, and what the declaration
derives where that is larger (:meth:`Tolerances.noise`), so that a declaration
only ever loosens one and a model built from exact numbers is measured against
its exact defaults.  A judgement of **exact arithmetic** on exact objects --
integer rotations, rational translations, the tables' own printed numbers, a
discrete value read with a fixed share of its spacing -- and one of
**rounding** read no declaration, and their numbers are fixed below, each with
what it was measured against.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------
# the declaration's own rules
# --------------------------------------------------------------------------

#: The covariance gate where nothing is declared, in the units of ``H``: machine
#: level for a model built from exact numbers, which is every model whose numbers
#: were typed rather than computed.
DEFAULT_ENERGY = 1e-8

#: The symmetry finder's distance where nothing is asked, in the model's length
#: units: spglib's own default.
DEFAULT_FINDER = 1e-5

#: The headroom a declared precision carries into every tolerance derived from it:
#: ``10 · p · max|H|`` for the covariance gate and, where the symmetry finder would
#: otherwise be defeated, ``10 · p · L`` for its distance.  A declared noise level
#: is where the noise *is*; a tolerance has to sit above it, and one decade is the
#: room that buys.
HEADROOM = 10.0

#: How far a computed value may exceed a bound and still meet it: a few ulp of a
#: number of order one.  A centre's move is a difference of reduced coordinates
#: carried through a group average, and the rounding of that arithmetic is not a
#: displacement -- a centre moved by exactly the declared precision measures
#: ``1.0000000272e-8`` against ``1e-8``.  The allowance is absolute, because the
#: rounding is that of the coordinates and not of the bound.
ROUNDING = 8.0 * float(np.finfo(float).eps)


def within(value, bound) -> bool:
    r"""Whether a computed value meets a bound: ``value ≤ bound + 8·eps``.

    The rounding allowance is absolute (:data:`ROUNDING`): a move of a centre is
    computed from coordinates of order one, whatever the bound it is held to.
    """
    return float(value) <= float(bound) + ROUNDING


def noise_floor(exact, matrix) -> float:
    r"""One threshold that measures noise: its exact default, or ``matrix`` where larger.

    ``matrix`` is how far a matrix read off ``H`` may miss its exact form,
    relative to its own scale (:attr:`Tolerances.matrix`).  A declaration only
    ever loosens such a threshold, and with nothing declared (``matrix`` ``0``
    or ``None``) every one of them is its exact default.  A margin that reads a
    discrete value -- a sign, a root of unity -- is no such threshold.
    """
    return float(exact) if not matrix else max(float(exact), float(matrix))


def finder_distance(precision, cell, symprec) -> float:
    r"""The distance the symmetry finder is run at on ``cell``, in the model's length units.

    ``symprec`` as asked, or ``10 · p · L`` where that is larger, ``L`` the
    longest of the cell's periodic lattice vectors.  It is arithmetic on the
    declaration itself -- a reduced noise of ``p`` is a Cartesian ``p · L`` -- and a
    finder asked to work below its own noise reports a group that is not there, or
    fails to report the one that is.  Only ever upward: a ``symprec`` asked
    deliberately is the floor.  ``position_tol`` and ``symprec`` keep their
    meanings, and neither is computed from the other.

    ``L`` is measured over the periodic directions alone.  A layer has no
    coordinate along its vacuum direction -- the slab presentation invents it, and
    every position there is exactly zero by construction -- so a declared
    position noise cannot apply to it, and reading ``L`` off the vacuum would raise
    the finder for every layer at a fitted model's precision for no physical
    reason.  The periodic lattice vectors of every cell built for one model are
    the model's own, so the distance is one number for the whole measurement.
    ``precision`` ``None`` declares nothing, and ``symprec`` stands as asked.
    """
    if precision is None:
        return symprec
    lat = np.asarray(cell.lat_vecs, dtype=float)
    per = [int(d) for d in cell.periodic_dirs]
    longest = float(np.linalg.norm(lat[per], axis=1).max()) if per else 0.0
    if float(precision) * longest > float(symprec):
        return HEADROOM * float(precision) * longest
    return symprec


# --------------------------------------------------------------------------
# the exact defaults of the readings of a matrix measured off H
# --------------------------------------------------------------------------
#
# Each is read through Tolerances.noise: this number for a model built from
# exact numbers, the declared matrix tolerance where that is larger.  Each is
# relative to the scale of what it reads, which is that of a unitary block.

#: How far outside a multiplet's blocks a member of an exactly symmetric
#: Hamiltonian's solution space may reach and still count as in them, relative
#: to the space's own scale: rounding, far below it.  A declared tolerance
#: larger than it is read instead, and a member that needed it sits in block
#: form only within the declared precision.
_IN_BLOCKS = 1e-8

#: How far a matrix may sit from the tensor form a conserved pseudospin gives it
#: -- ``1 ⊗ C`` for a conserved operator, ``B ⊗ C`` for a member of a solution
#: space -- and still be read in it, relative to its scale.
_FACTOR_TOL = 1e-8

#: The gate on the **scaled** tensor-structure defect of one operation: the
#: fraction of the member of the detector's solution space closest to
#: tensor-structure form that no ``O(g) ⊗ S(g)`` can carry.  It is
#: dimensionless and lives in ``[0, 1]`` — 0 when the tensor structure is
#: realizable and 1 when its component vanishes outright — and this is the
#: same ``1e-6`` the read-off discards a generator at (``|lambda| < 1e-6`` on
#: the reference-class block, and the phase snap, :data:`SNAP`).
#: Measured margins: 1.3e-15 for every realizable operation of Kane–Mele's
#: slab, against 0.5 or 1.0 for the 24 the Rashba term breaks.
ANSATZ_GATE = 1e-6

#: How far a matrix may sit outside a space of matrices the model measured and
#: still count as a member of it, relative to its own scale.
IN_SPACE = 1e-8

#: How far a unitary read off a solution space may sit outside the space and
#: still be a member of it, relative to its own scale: the unitary nearest a
#: member moves it by the member's own departure from unitarity, which an
#: exact model's members carry only to rounding.
IN_SPAN = 1e-6

#: A block, a scalar or a singular value of a measured space that counts as
#: zero, relative to the scale of what it is read in: nothing the operation
#: carries there.
ZERO = 1e-9

#: How small a block's smallest singular value may be, relative to the scale of
#: the members it is read off, and the block count as singular: a symmetry
#: matrix is nonsingular on every block it fills.
SINGULAR = 1e-8

#: How far a matrix assembled from measured blocks may miss an exact relation
#: -- unitarity, or a product the multiplication table fixes -- and still
#: satisfy it, relative to its own scale.
ASSEMBLED = 1e-8

#: How far the scalars a solution space pins may spread across its members and
#: still be one scalar the hoppings fix, relative to their scale: above it the
#: hoppings leave the scalar open.
SPREAD = 1e-7

#: How far a measured phase may sit from a discrete one -- a sign, a quarter
#: turn, a root of unity -- and be read as it, and how small a measured ratio
#: has to be to be read as vanishing.
SNAP = 1e-6

#: How close two eigenvalues of an operator built from measured matrices have
#: to be, relative to the spread of its spectrum, to be one cluster.
CLUSTER = 1e-6

#: How far a character read off measured blocks may sit from a value -- another
#: character, or the integer a multiplicity is -- and be read as it.
CHARACTER = 1e-6

# --------------------------------------------------------------------------
# the detector's own numbers
# --------------------------------------------------------------------------

#: Cut that declares the constraint operator ``X H(k) - H(Rk) X`` **vacuous**:
#: identically zero, so that every ``X`` solves it.  Measured *absolutely*,
#: against ``max |H|`` over the grid, because a vanishing operator has no scale
#: of its own — the relative nullspace cut has nothing to be relative to, and
#: reading its own rounding noise as a scale would make ``dim == 0`` a false
#: proof on a one-band model.
#:
#: The value is set by the two populations it must separate, and both were
#: measured.  *Generous enough*: a constraint that is zero up to entrywise
#: rounding reads ``s_max / max|H| <= 2.7e-16`` — one-orbital cubic and
#: tetragonal, every operation — which is more than six decades below this cut.
#: *Strict enough*: a genuinely broken one-band operation reads
#: ``s_max / max|H| = 2.0`` — the anisotropic square's four ``C4``-type
#: operations — nine decades above it.  Both bounds are scale-covariant: the
#: rounding noise in ``H`` and the singular values of the constraint are alike
#: proportional to ``max |H|``, so neither moves when the Hamiltonian is
#: rescaled.  A declared cut larger than it is read instead.
#:
#: Nothing is *accepted* on the strength of this number.  Declaring vacuity
#: hands back the full matrix space with ``1`` returned first, and the identity's
#: own covariance residual — remeasured from ``H``, never from the SVD — is
#: exactly the breaking the cut let through.  A residual gate refuses it on
#: that number, as it refuses any other.
_VACUOUS_CUT = 1e-9

#: Floor and ceiling of the eigen route's **matching window** ``δ`` — the
#: relative eigenvalue distance ``|E'_a − E_b| / max|H − μ·1|`` inside which a
#: pair of eigenvectors contributes an outer product to the stage-0 basis, ``μ``
#: the mean on-site energy, so that a constant shift of ``H`` moves no window.
#: The window follows the declared tolerance read relative to ``H``,
#: ``δ = clip(10·sqrt(ρ), floor, ceiling)`` (:func:`eigen_window`): a solution
#: with relative residual ``ρ`` has cross terms ``ρ/gap`` between bands, so a
#: window ``δ`` represents it only to ``ρ/δ`` and a raised tolerance without a
#: wider window would still read ``dim == 0``.
#:
#: The floor is measured: at ``δ = 1e-3`` the dropped cross terms of an *exact*
#: symmetry are ``eps/δ = 2e-13`` relative — five decades under the default cut
#: — and the dim-1 residual floors read 8.3e-15 (Kane–Mele), 2.3e-14 (lin3g),
#: 9.2e-15 (Fu–Kane–Mele) and 2.7e-14 (``i41_soc``, against 4.8e-15 for the
#: stacked route).  Narrower windows lose that accuracy (``i41_soc`` 3.5e-13 at
#: ``1e-4``, 2.6e-11 at ``1e-6``); wider ones buy nothing on an exact symmetry
#: and cost basis size (6.3e-14 at ``1e-1``, ``m = 6n`` against ``m ≤ 3n``).
_EIGEN_WINDOW_FLOOR = 1e-3
_EIGEN_WINDOW_CEIL = 0.3


def eigen_window(relative) -> float:
    r"""The matching window a tolerance asks for: ``clip(10·sqrt(relative), 1e-3, 0.3)``.

    ``relative`` is the tolerance read relative to the Hamiltonian's magnitude:
    ``tol / max|H|`` under a declared precision, and a declared ``tol`` itself
    otherwise, which is that reading for a model whose magnitude is one.
    """
    return float(min(max(10.0 * np.sqrt(float(relative)), _EIGEN_WINDOW_FLOOR),
                     _EIGEN_WINDOW_CEIL))


# --------------------------------------------------------------------------
# exact arithmetic: the tables' printed numbers, and discrete readings
# --------------------------------------------------------------------------

#: Tolerance for "these two tabulated character vectors are the same".  The
#: tables are printed decimals (``irreptables``' ``sqrt(2)`` is 1.414214, its
#: cube roots of unity carry a phase of 0.66667π), so ~1e-5 of rounding rides
#: on every entry, while *distinct* irreducible characters differ by O(1).
_CHAR_TOL = 2e-3

#: Tolerance for "this tabulated phase is real".  Not a tuning parameter: over
#: every (space group, valuedness) instance, unitary and grey, the largest
#: ``|Im ω₀|`` among the real factor systems is 6.1e-16 and the smallest
#: nonzero one the tables realize is ``sin(2π/6) = 0.866``.
_PHASE_REAL_TOL = 1e-3

#: Tolerance for "these two reduced-coordinate vectors differ by a lattice
#: vector".  ``irreptables`` prints coordinates to six decimals, so a hexagonal
#: ``1/3`` carries 3e-7 and a product of two of them a few times that.
_LATTICE_TOL = 1e-4

#: Tolerance for matching two tabulated SU(2) matrices up to sign.  The same
#: operation is printed in the unitary and the grey file with independent
#: rounding; the worst disagreement over all 460 pairs is 1.0e-5.
_SU2_TOL = 1e-3

#: Tolerance on "this tabulated character-table number is an integer".  The
#: tabulated characters carry ~1e-5 of rounding (they are printed decimals —
#: ``irreptables``' ``sqrt(2)`` is 1.414214), and the quantities tested here are
#: integers a whole unit apart, so this is loose enough to pass a real table and
#: three orders tighter than the gap it is guarding.  Measured worst defect on
#: the fixture groups: 1.0e-5 (SG 99 at GM/M).
_INT_TOL = 1e-3

#: How far two matrices the tables write to five printed decimals may sit
#: apart and still be one operation: an orthogonal matrix, or an SU(2) one.
_TABLE_MATRIX_TOL = 1e-3

#: Gate on the measured agreement between two presentations of one layer — the
#: band-block characters at the operations and k-points they share, the stack
#: route's ``"characters"`` check.  The quantity is a trace of unitaries on a
#: handful of bands, so its scale is ``O(1)`` and machine precision is
#: ``1e-15``: a defect anywhere near this bound is not a rounding question.
CHARACTERS_TOL = 1e-6

#: Default dimensionless tolerance the corep labels are read at: the
#: integrality gate on the irrep multiplicities (matching
#: ``irrep.bandstructure.check_multiplicity``) and the character agreement
#: behind the phase-invariant partition.  A multiplicity further than this from
#: an integer is a refusal, because a non-integer count of an irrep is not a
#: statement about a band block.
LABEL_TOL = 1e-3

#: Default tolerance for locating a requested ``k`` among a ``WFArray``'s mesh
#: points, in reduced reciprocal coordinates.  A miss is a refusal, never a
#: fresh solve.
MESH_TOL = 1e-9

#: Fractional-coordinate gate for "this point is on that variety".  Reference
#: -cell positions are exact rationals for a typed model and the change to the
#: reference cell, ``refUC^-1 (x - shiftUC)``, is exact linear algebra, so the
#: measured residual is ~1e-16 there; the gate is loose enough for a numerical
#: (wannier90) cell, whose centres carry a Wannierization's noise, and far
#: tighter than the gap between two distinct Wyckoff varieties (1/8 in the
#: diamond, 1/6 in ``P3_121``).
WYCKOFF_REFERENCE_TOL = 1e-4

#: How far a phase read off the states' matrices may sit from the root of unity
#: it reads as.  The phases the class verdict compares -- the ratio of two
#: commuting operations' matrices taken in the two orders, and the relations a
#: rephasing has to close -- are roots of unity of order at most twelve on a
#: crystallographic point group, never nearer one another than half a unit, and
#: a model's own noise sits decades below.
_CLASS_MARGIN = 0.25

#: How far the square of a unitary time-reversal intertwiner may sit from a sign
#: and still read as it.  The square of a unitary is unitary, so ``+1`` and
#: ``-1`` are the only multiples of the identity it can be, a distance 2 apart;
#: a numerical model's own defect moves it by that defect, and a member of a
#: commutant that is no multiple of the identity at all lands farther off.  A
#: reading of a sign, so the margin is a quarter of the way to the other one.
_TIME_REVERSAL_SQUARE_MARGIN = 0.5

#: The fraction of the largest coupling a multiplet has to reach to be read as
#: the partner a multiplet is glued to rather than as its rounding.  A merge is a
#: discrete verdict, so only the top band is taken, and a coupling this leaves
#: behind is measured again on the next round.
_LAYOUT_MERGE = 0.5

# --------------------------------------------------------------------------
# rounding
# --------------------------------------------------------------------------

#: The exact floor of :func:`.._build._build_rep`'s covariance gate,
#: ``max_k ‖D H(k) D† − H(Rk)‖`` of a built representation, **absolute and in
#: the units of** ``H``.  The gate itself is :func:`..symmetry.covariance_gate`:
#: this floor, or the rounding the representation's own arithmetic carries
#: where that is larger.  The same gate is the boundary between an *exact*
#: operation and a merely near one
#: (:func:`..symmetrization.classify_operations`), and it is not a knob the
#: pipeline turns: a declared detection tolerance moves the detector's gate and
#: cut, never this one.
EXACT_TOL = 5e-14

#: Closure at or below this is **arithmetic**, not a measurement about the cell.
#: A cell whose operations carry non-dyadic rationals cannot close to exactly
#: ``0.0`` in binary floating point: the lifted ``3x3x3`` Fu-Kane-Mele supercell's
#: 2592 operations, whose pure translations are exact thirds, measure
#: ``4.4e-16``.  A genuinely **numerical** cell is five decades away from that —
#: the displaced Kane-Mele cell measures ``2.0e-8`` — so the two are never
#: confused.  Of the repair, only the branch that has *no tabulated operations to
#: repair from* reads this constant: where ``irrep`` matched the cell the repair
#: rebuilds the translations from the tabulated operations and never consults
#: it.  :func:`.._descent._shared_origin` reads it too, as the scale below which two
#: translations of one operation are the same translation.
_EXACT_CLOSURE = 1e-12

#: How far a candidate's matrices may miss composing by the group's table, a
#: phase per product allowed, and still be read as they stand: the canonical
#: norm's own gate (:func:`..symmetry.normalize_rep`'s ``tol``).  Matrices built
#: from an exact model's blocks compose to rounding, far below it; above it they
#: are replaced by the exact representation nearest them, which is no loosened
#: gate.
_COMPOSITION_GATE = 1e-9

#: Relative size, against the model's own ``max|t|``, below which a rebuilt
#: coefficient is rounding rather than a hopping.  The drop is taken
#: **orbit-wise** (:func:`..symmetrization._drop_threshold`), never on this
#: number alone.
_DROP_RELATIVE = 1e-14

#: A rebuilt coefficient outside the range box must be rounding.  Measured
#: 2e-16 relative on every model built from exact numbers; the gate is five
#: decades above that.
_RANGE_RELATIVE = 1e-10

#: How far a presentation may differ from the model it presents, relative to
#: ``max|H|``, where the two evaluate the phases ``e^{2πi k·τ}`` differently.  A
#: slab that moved its orbitals, carried back through that move, evaluates them
#: at the moved centres; the spinless presentation of a spinful model evaluates
#: one for each of its ``2N`` orbitals where the model evaluates ``N`` and
#: repeats each for the spin, in a matrix product whose rounding can depend on
#: its shape.  Measured below 5e-16 on slabs moved by up to four cells and at
#: 5.2e-17 on a twelve-state spin–orbit model; the gate is more than four
#: decades above both.  A presentation that moves no orbital and evaluates the
#: same phases is held to bit identity instead.
_PRESENTATION_RELATIVE = 1e-11

#: How far a carried operation, site or lattice shift may sit from what it must
#: be in the stack's cell.  The two cells share their in-plane vectors and
#: differ along the normal by a length and a sense alone, so what is carried is
#: exact to rounding.
_CARRY_TOL = 1e-9

#: Zero, for a quantity built from ``H(k)`` alone: a commutator, a residual.
FLAT_ZERO = 1e-14

#: Bands closer in energy than this are one degenerate multiplet, and a
#: multiplet is one row of a character table.  The representation a measurement
#: returns is covariant at the exact gate, so a degeneracy its group forces is
#: one to rounding, decades below this.
DEGENERACY_TOL = 1e-8

#: Relative gate on ``max |H(k) V - V diag(w)|`` when ``(evals, evecs)`` are
#: supplied: the measured statement that those states are
#: eigenstates *at this k* and not a second diagonalization from somewhere
#: else.  Genuine states land ~5 orders of magnitude inside it (residuals
#: ``<= 1e-13``); a wrong k gives ``O(||H||)``.
STATES_AT_K_GATE = 1e-8


# --------------------------------------------------------------------------
# the record
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Tolerances:
    r"""What one realization judges noise against, each in the units of what it compares.

    Resolved once from the declaration (:meth:`declared`) and handed whole to
    every step that judges noise.  With nothing declared every field is its exact
    default, and a model built from exact numbers is measured against exactly the
    numbers it always was.

    Attributes
    ----------
    precision : float or None
        The declared relative precision of the model's numbers, ``p``; ``None``
        where nothing was declared.
    scale : float or None
        ``max|H|``, the largest entry of ``H(k)`` about the mean on-site energy,
        measured where a declaration is read against it, ``None`` otherwise.
    energy : float
        How far ``U H(k) U†`` may miss ``H(Rk)`` and the operation still be
        obeyed: the covariance gate, in the units of ``H``.  ``10 · p · max|H|``
        under a precision, a declared ``tol`` as given, ``1e-8`` otherwise.
    cut : float or None
        The detector's nullspace cut, relative and dimensionless: ``energy /
        max|H| = 10 · p`` under a precision, a declared ``cut`` as given, and
        ``None`` where it follows ``energy`` (:func:`.._detector._find_intertwiners`).
    window : float or None
        The detector's matching window, relative: ``clip(10·sqrt(cut), 1e-3,
        0.3)`` under a precision, a declared ``window`` as given, and ``None``
        where it follows ``energy``.
    floor : float or None
        The covariance, in the units of ``H``, below which the detector's
        nullspace keeps a candidate at a point whatever the relative cut says:
        ``energy`` under a precision, the gate itself, so that the cut never
        discards what the gate would keep; ``None`` where it is the cut read in
        the units of ``H``, as a declared ``cut`` is.
    centre : float or None
        How far a centre may sit from where the operations put it, in reduced
        coordinates: ``p`` under a precision, a declared ``position_tol`` as
        given, ``None`` where nothing declares it.
    matrix : float
        How far a matrix read off ``H`` may miss its exact form, relative to its
        own scale: ``energy / max|H|`` where a precision or a ``tol`` is
        declared, and ``0.0`` otherwise, so that every such reading keeps its
        exact default.
    finder : float
        The symmetry finder's distance, in the model's length units: ``symprec``,
        raised where a declared precision would defeat it
        (:func:`finder_distance`).
    finder_floor : float
        ``symprec`` as asked, below which the finder is never run.
    """

    precision: float | None = None
    scale: float | None = None
    energy: float = DEFAULT_ENERGY
    cut: float | None = None
    window: float | None = None
    floor: float | None = None
    centre: float | None = None
    matrix: float = 0.0
    finder: float = DEFAULT_FINDER
    finder_floor: float = DEFAULT_FINDER

    @classmethod
    def declared(cls, model, *, precision=None, tol=None, cut=None, window=None,
                 position_tol=None, symprec=DEFAULT_FINDER, scale_of=None) -> Tolerances:
        r"""The tolerances one call's declaration resolves into.

        ``precision`` is the one number the others follow from.  The centres'
        tolerance is ``p`` itself, in reduced coordinates, where a relative
        precision is already absolute; the covariance gate is in the units of
        ``H``, where it is not, so it is ``10 · p · max|H|`` with ``max|H|``
        measured, which is the whole reason a declaration should not have to be
        resolved by hand.  The detector's cut and window are relative, so they
        are read against the same ``max|H|``: the cut is ``10 · p``, the window
        ``clip(10·sqrt(cut), 1e-3, 0.3)``, and the floor below which the
        nullspace keeps a candidate whatever the cut says is the gate itself.
        The finder's distance follows the model's lattice
        (:func:`finder_distance`).  Without a precision, a declared ``tol``,
        ``cut``, ``window`` and ``position_tol`` stand as given, and the gate is
        ``1e-8`` where ``tol`` is not declared either.

        Every judgement of a matrix read off ``H`` -- whether orbitals form one
        multiplet, whether a member sits in a measured space, which scalar a
        space pins, which sign a transported phase is -- reads a threshold that
        measures noise, and the declared gate is what noise means there too:
        ``energy / max|H|``, relative, because a block is unitary -- ``10 · p``
        under a precision.  Where nothing is declared it is ``0.0``, and every
        such threshold keeps its exact default.

        ``scale_of`` measures ``max|H|``, and is asked only where a declaration
        is read against it.  The grammar of the declaration -- a positive
        number, and not two that say the same thing -- is checked where the call
        starts, before anything is resolved.
        """
        declared_tol = tol is not None
        scale = None
        if precision is not None:
            p = float(precision)
            scale = scale_of()
            tol = HEADROOM * p * scale
            position_tol = p
        matrix, floor = 0.0, None
        if precision is not None and scale > 0.0:
            matrix = float(tol) / float(scale)
            # the detector's relative numbers, read against the same magnitude,
            # and the gate its nullspace never discards a candidate within
            cut, window, floor = matrix, eigen_window(matrix), float(tol)
        elif precision is None and declared_tol:
            scale = scale_of()
            matrix = float(tol) / scale if scale > 0.0 else 0.0
        if tol is None:
            tol = DEFAULT_ENERGY
        return cls(precision=None if precision is None else float(precision),
                   scale=scale, energy=tol, cut=cut, window=window, floor=floor,
                   centre=position_tol, matrix=matrix,
                   finder=finder_distance(precision, model, symprec),
                   finder_floor=symprec)

    def noise(self, exact) -> float:
        r"""One threshold that measures noise: its exact default, or ``matrix`` where larger.

        A declaration only ever loosens such a threshold, and with nothing
        declared every one of them is its exact default (:func:`noise_floor`).
        """
        return noise_floor(exact, self.matrix)

    def returned(self) -> dict | None:
        r"""What a declared precision derived, as :attr:`..groups.RealizedGroup.precision` holds it.

        Returns
        -------
        dict or None
            ``precision``, ``max_h``, ``tol``, ``position_tol``, ``cut``,
            ``window``, ``symprec`` and ``symprec_raised``; ``None`` where no
            precision was declared.
        """
        if self.precision is None:
            return None
        # the numbers the detector ran at: where the magnitude is zero there is
        # nothing to read a relative number against, and the cut follows the gate
        return {"precision": self.precision, "max_h": self.scale,
                "tol": float(self.energy), "position_tol": self.precision,
                "cut": float(self.energy if self.cut is None else self.cut),
                "window": float(eigen_window(self.energy) if self.window is None
                                else self.window),
                "symprec": float(self.finder),
                "symprec_raised": bool(self.finder != self.finder_floor)}


#: The record of a call that declares nothing: every threshold its exact default.
EXACT = Tolerances()
