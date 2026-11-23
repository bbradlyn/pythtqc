"""The space group a Hamiltonian obeys, and the record realized_group returns.

The group of the cell is found, and each of its operations is measured on
``H``.  Where operations are broken, the group is lowered to the largest
subgroup of the operations that hold, the first by the tie rule where several
of that order qualify.  A cell whose group is exactly that subgroup is built by
adding one generic orbit, and identified again, until nothing changes.  Where
a precision or a position_tol is declared, the orbital centers are first moved
onto symmetric positions.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from dataclasses import fields as dataclass_fields
from typing import Any
import warnings

import numpy as np

from . import _bloch
from .refusal import InternalError, MagneticScopeWarning, Refusal
from ._tolerance import (
    DEFAULT_FINDER, Tolerances, _CLASS_MARGIN, _EXACT_CLOSURE, finder_distance,
    within)
from ._text import _DOUBLE_GROUP, _PER_COSET
from ._spacegroup import (
    GroupData, OrbitalAssignment, _OWN_ORBITALS, _PACKAGE_DIR, _group_of, _pad_cell,
    _sites_and_typat)
from ._seitz import OperationTable, _record_name, operations
from ._tables import _assign_orbitals
from ._subgroups import (
    _descent_closure, _operation_key, _resolve_closure, _subgroups_inside,
    _tie_ranked, _tie_reason, _tie_rule_words)
from ._matrices import _resolve_spin, _spin_pair_split
from ._grammar import _declaration_grammar, _names_joint_keys
from ._centers import (
    _cell_sites, _centers_account_for, _centers_account_for_closure, _frame_defect, _geometry,
    _moved_extras, _moves_of, _rational_offset, _site_action, _symmetric_sites)
from .symmetry import _declaration_checks, _wyckoff_orbit_table
from ._sectors import read_sectors, sector_action
from ._multiplets import _site_multiplets
from ._layers import (
    DECORATION_DZ, SLAB_C, _decorated_cell, _environment_record,
    _is_presented_as_a_slab, _raise_environment_contradiction, _slab_presentation)
from ._labeling import _held_sets, _parts_apart
from ._measure import (
    ValidatedGroup, _MeasurementContext, _grid_covariance, _max_hamiltonian,
    _own_form_read, _validate_group)
from ._build import _BUILD_KEYWORDS, _build_rep, _exact_gate
from .symmetrization import (
    EXACT_GATE, _classify_operations, _held_off, _memory_budget, _parts_coupling,
    _projection_operators, _search_ceiling, _symmetrized, _with_centers)
from ._descent import (
    _DROPPED_BY_WORDS, _antiunitary_settled, _beyond_grey, _carried_onto,
    _carried_reduction, _centers_not_exactly_symmetric, _declared_cell,
    _declared_cell_reduction, _declared_contents, _declared_contents_wider,
    _declared_parts_of, _descend_once, _descent_clauses, _descent_counts,
    _lattice_note, _layer_declared_reading, _match_operations, _operation_verdicts,
    _precision_holding, _rekey_contents, _restricted_contents, _restricts_to,
    _shared_origin, _supercell_refusal, _time_reversal_measured,
    _time_reversal_member, _time_reversal_square)

__all__ = ["RealizedGroup", "RealizedGroupNumerical", "LayerEnvironment", "Descent",
           "DescentStep", "DescentCell", "StepDeclaration", "StepCenters", "StepTie",
           "StepProjective", "ProjectivePair", "ProjectiveMirror", "ProjectiveReading",
           "DescentLattice", "LatticeTranslation", "TranslationDressing",
           "RefutedTranslation",
           "DroppedOperation", "NamedGroup", "TimeReversal", "TimeReversalSquare",
           "BeyondGrey", "BeyondGreyOperation", "Conserved", "ConservedSector",
           "CrossCheck", "CrossCheckRow", "CrossCheckOperation", "CrossCheckRefusal",
           "Decoration", "DeclaredReduction", "ExcludedOperation", "Positions", "Precision",
           "Presentation",
           "PresentationTilt", "realized_group"]


# --------------------------------------------------------------------------
# the records on what realized_group returns
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class NamedGroup:
    """A space group by its symbol, its number and its number of operations.

    ``str()`` writes it as the symbol and the number, ``"P6mm1' #183.186"``.

    Attributes
    ----------
    name : str
        The group's Hermann-Mauguin symbol, ``"Fd-3m1'"``.
    number_str : str
        The group's number in the tables of the irreptables package,
        ``"227.129"``.
    size : int
        The number of its operations, one per coset of the lattice
        translations.
    """

    name: str
    number_str: str
    size: int

    def __str__(self) -> str:
        return f"{self.name} #{self.number_str}"


@dataclass(frozen=True)
class DescentCell:
    """The orbit a descent step adds to the cell so that its group is the group kept.

    A descent lowers the cell's group to the group the Hamiltonian obeys, in
    one step or several (:class:`Descent`).  At each step the orbit of a
    generic point under the operations kept is added to the cell as atoms of a
    new species, and the cell's group is found again.

    Attributes
    ----------
    generic_point : tuple of float
        The generic point, in reduced coordinates.
    orbit : tuple of tuple of float
        The points of its orbit, in reduced coordinates.
    species : str
        The species the points are added as.
    """

    generic_point: tuple
    orbit: tuple
    species: str


@dataclass(frozen=True)
class DroppedOperation:
    """One operation a descent dropped, and what dropped it.

    A field that does not apply to what dropped the operation is ``None``.  A
    matrix ``U`` carries ``H(k)`` onto ``H(gk)``, for ``g`` the operation,
    where ``U H(k) U^-1 = H(gk)``, with ``H(k)*`` in place of ``H(k)`` for an
    operation combined with time reversal.  The operation's solution space is
    the set of such matrices at the k measured.  Its covariance residual, for the
    best unitary ``U`` found, is the largest entry of ``U H(k) U^-1 - H(gk)``
    over the k measured, in the units of H, and it is held to the tolerance
    ``r.numerical.validated.tol``.  The operation's form is a matrix that
    carries each site's orbitals onto the orbitals of the site it moves them
    to, times the operation's own spin rotation where the states carry spin.

    Attributes
    ----------
    index : int
        The operation, by index in the group it was dropped from.
    name : str
        Its name, in the Cartesian axes in which the pythtb Lattice is
        specified.  Where its translation is not a lattice vector of the cell,
        as for a screw rotation, a glide or a pure translation, the name
        carries the translation in Seitz form, ``{E|1/2,0,0}``.
    refuted_by : str
        What dropped it.  ``"no symmetry matrix"``: no matrix carries
        ``H(k)`` onto ``H(gk)``.  ``"covariance"``: such matrices exist, and
        the covariance residual is above the tolerance.  ``"tensor
        structure"``: no matrix of the operation's form carries ``H(k)`` onto
        ``H(gk)`` within the tolerance.  ``"out of plane"``: an operation of
        the slab a layer is read as (:class:`Presentation`) that turns the
        layer's plane out of itself, so that it has no action on the layer's
        k.  ``"translation"``: a pure translation whose permutation of the
        sites, with or without a change of basis within each site, does not
        commute with ``H(k)``.  ``"declaration"``: it is outside the largest
        group of the operations still standing that carries the declared site
        irreps.  ``"centers"``: it is outside the largest group that carries
        the orbital centers within the tolerance declared for them.
        ``"projective class"``: the hoppings obey it, and the states carry no
        single-valued representation of a group containing it and the
        operations kept.  ``"closure"``: the operations measured to hold do
        not close into a group, and no subgroup of them of the largest order
        contains it.  ``"tie"``: several subgroups of the operations measured
        to hold have the largest order, and the one the tie rule
        (:class:`Descent`) keeps does not contain it.  For the declaration
        and the centers, an operation combined with time reversal is dropped
        where its unitary part is.
    residual : float
        The number the measurement rests on.  For ``"covariance"``, the
        covariance residual, in the units of H.  For ``"translation"``, the
        same with ``U`` the permutation of the sites.  For
        ``"tensor structure"``, how far the member of the solution space
        nearest the operation's form lies outside that form, over the member's
        own size, dimensionless.
        ``inf`` for ``"no symmetry matrix"`` and ``"out of plane"``; ``nan``
        for the other causes, which rest on no residual.
    step : int or None
        The descent step that dropped it, counted from 1.
    of : str or None
        The number of the group it was dropped from, in the tables of the
        irreptables package.
    note : str or None
        If not ``None``, why it was dropped, in words.
    orbitals : tuple of int or None
        Where the declared site irreps dropped it, the orbitals declared.
    declared, measured : complex or None
        Where the declared site irreps dropped it, the character declared and
        the one the hoppings measure.
    miss : float or None
        Where the centers dropped it, the farthest a center must move for the
        operations this one generates to carry the centers onto one another,
        in reduced coordinates.
    precision : float or None
        Where the centers dropped it, the tolerance declared for the centers,
        ``position_tol=`` or ``precision=``, in reduced coordinates.
    covariance : float or None
        Where the member of the solution space nearest the operation's form
        misses that form by more than rounding, the covariance residual of its
        part in the form, in the units of H.
    pair, names : tuple or None
        Where the states' projective class dropped it, two commuting
        operations whose matrices on the states do not commute, by index and
        by name.
    commutation_phase : complex or None
        The phase ``c`` in ``D(a) D(b) = c D(b) D(a)``, with ``a`` and ``b``
        that pair and ``D`` their matrices.
    spin_half : bool or None
        Whether ``c`` is also the phase between ``S(a) S(b)`` and
        ``S(b) S(a)``, with ``S`` the operations' spin-1/2 rotations, as for
        spinor states read without their spin.
    rows : tuple of ..symmetry.DeclarationCheck or None
        Where the declared site irreps dropped it, each declaration checked
        against it, or against its unitary part for an operation combined
        with time reversal: the character declared and the one the hoppings
        measure.  ``None`` for the other causes.
    """

    index: int
    refuted_by: str
    name: Any = None
    residual: Any = None
    step: Any = None
    of: Any = None
    note: Any = None
    orbitals: Any = None
    declared: Any = None
    measured: Any = None
    miss: Any = None
    precision: Any = None
    covariance: Any = None
    pair: Any = None
    names: Any = None
    commutation_phase: Any = None
    spin_half: Any = None
    # the checks are many and printed elsewhere, so the record's repr leaves them out
    rows: Any = field(default=None, repr=False)


@dataclass(frozen=True)
class StepDeclaration:
    """What the declared site irreps kept at one descent step.

    Held in :attr:`DescentStep.declaration` where the declaration made with
    ``characters=`` is read at the step.

    Attributes
    ----------
    subgroup : tuple of int
        The operations the declared site irreps keep, by index in the step's
        ``cell_group``.
    tied : tuple
        If nonempty, the subgroups of the largest order that carry the
        declared site irreps, each by its operations' indices.  They stand in
        the order of the tie rule (:class:`Descent`), the one kept first.
    rows : tuple of ..symmetry.DeclarationCheck
        The declared site irreps checked against each operation measured.
    names : tuple or None
        If not ``None``, the tied subgroups, each by its operations' names.
    note : str or None
        If not ``None``, which tied subgroup was kept and why, in words.
    """

    subgroup: tuple
    tied: tuple
    rows: tuple
    names: Any = None
    note: Any = None


@dataclass(frozen=True)
class StepCenters:
    """What the orbital centers kept at one descent step.

    Held in :attr:`DescentStep.centers` where the whole group of the step does
    not carry the orbital centers within the tolerance declared for them.  A
    group carries the centers when every center can be moved onto a position
    symmetric under that group by no more than that tolerance.

    Attributes
    ----------
    precision : float
        The tolerance declared for the centers, ``position_tol=`` or
        ``precision=``, in reduced coordinates.
    move : float
        The farthest a center must move for the centers to be symmetric under
        the whole of the step's ``cell_group``, in reduced coordinates.  It is
        larger than ``precision``.
    kept : tuple of int
        The operations kept, by index in ``cell_group``: the largest subgroup
        that carries the centers.  Each operation combined with time reversal
        whose unitary part is in that subgroup is kept with it.
    tied : tuple
        If nonempty, the subgroups of the largest order that carry the
        centers, each by its operations' indices.  They stand in the order of
        the tie rule (:class:`Descent`), the one kept first.
    note : str
        The group kept and the operations it leaves out, in words.
    """

    precision: float
    move: float
    kept: tuple
    tied: tuple
    note: str


@dataclass(frozen=True)
class StepTie:
    """The subgroups of the largest order inside operations that do not close.

    Held in :attr:`DescentStep.tie`.  The operations measured to hold at the
    step do not close into a group, and several of their subgroups have the
    largest order.  The tie rule (:class:`Descent`) keeps one of them.

    Attributes
    ----------
    tied : tuple
        Those subgroups, each by its operations' indices, in the order of the
        tie rule.
    kept : tuple
        The subgroup kept, the first of them.
    note : str
        Which subgroup was kept and why, in words.
    """

    tied: tuple
    kept: tuple
    note: str


@dataclass(frozen=True)
class ProjectivePair:
    """Two commuting operations whose matrices on the states do not commute.

    Such a pair shows that the states carry no single-valued representation
    of a group that contains both.

    Attributes
    ----------
    pair : tuple of int
        The two operations, by index.
    names : tuple of str
        Their names.
    commutation_phase : complex
        The phase ``c`` in ``D(a) D(b) = c D(b) D(a)``, with ``a`` and ``b``
        the two operations and ``D`` their matrices.  No rescaling of either
        matrix removes it.
    spin_half : bool
        Whether ``c`` is also the phase between ``S(a) S(b)`` and
        ``S(b) S(a)``, with ``S`` the operations' spin-1/2 rotations, as for
        spinor states read without their spin.
    """

    pair: tuple
    names: tuple
    commutation_phase: complex
    spin_half: bool


@dataclass(frozen=True)
class ProjectiveMirror:
    """The sign chosen for a flat layer's horizontal mirror, read without spin.

    On a flat layer an operation ``g`` and its product with the horizontal
    mirror, ``m g``, act alike on k and share one measurement, which fixes the
    mirror's matrix only up to a sign.  The matrix of ``m g`` is taken as that
    sign times the matrix of ``g``.  The sign commutes with every matrix, so
    the projective class does not depend on it.

    Attributes
    ----------
    sign : int
        ``+1`` or ``-1``.
    determined_by : str
        ``"declaration"``, the parity the declared site irreps state;
        ``"lowest-l"``, the even sign the lowest-angular-momentum rule
        assigns.
    """

    sign: int
    determined_by: str


@dataclass(frozen=True)
class ProjectiveReading:
    """The group of a layer free-standing, on which its class on a substrate is read.

    For a layer on a substrate, the states' projective class is read on the
    group of the same layer free-standing, and the part of that reading that
    does not turn the layer over is carried onto the substrate.

    Attributes
    ----------
    group : NamedGroup
        The free-standing layer's group.
    kept : int
        The number of its operations in the largest subgroup on which the
        states carry a single-valued representation.
    """

    group: NamedGroup
    kept: int


@dataclass(frozen=True)
class StepProjective:
    """What the states' projective class kept at one descent step.

    Held in :attr:`DescentStep.projective` where the class lowered the group:
    no single-valued representation of the step's group exists on the states,
    as for spinor states read without their spin.

    Attributes
    ----------
    pairs : tuple of ProjectivePair
        Each pair of commuting operations whose matrices do not commute.
    kept : tuple of int
        The operations kept, by index: the largest subgroup on which the
        states carry a single-valued representation.
    tied : tuple
        If nonempty, the subgroups of that order on which they carry one, each
        by its operations' indices.  They stand in the order of the tie rule
        (:class:`Descent`), the one kept first.
    rule, reason : str or None
        If not ``None``, the tie rule, and why it kept the first subgroup.
    horizontal_mirror : ProjectiveMirror or None
        If not ``None``, the sign chosen for a flat layer's horizontal mirror.
    read_on : ProjectiveReading or None
        If not ``None``, the free-standing layer's group the class was read
        on, for a layer on a substrate.
    """

    pairs: tuple
    kept: tuple
    tied: tuple
    rule: Any = None
    reason: Any = None
    horizontal_mirror: ProjectiveMirror | None = None
    read_on: ProjectiveReading | None = None


@dataclass(frozen=True)
class LatticeTranslation:
    """One pure translation of a cell that is not primitive, and what was measured.

    Attributes
    ----------
    index : int
        The operation, by index in the cell's group.
    translation : tuple of float
        The translation, in reduced coordinates.
    cartesian : tuple of float
        The translation, in the Cartesian axes in which the pythtb Lattice is
        specified.
    refuted_by : str or None
        What refuted it, in the words of :attr:`DroppedOperation.refuted_by`;
        ``None`` where the Hamiltonian obeys it.
    scaled : float
        Where its tensor structure refuted it, the dimensionless number
        :attr:`DroppedOperation.residual` holds for ``"tensor structure"``;
        ``nan`` otherwise.
    residual : float or None
        The largest entry of ``P H(k) P^-1 - H(k)`` over the k-points measured,
        with ``P`` the permutation of the sites the translation makes, in the
        units of H.  ``None`` where two sites the translation pairs hold
        different numbers of orbitals.
    tol : float
        The tolerance the residual is held to, in the units of H.
    """

    index: int
    translation: tuple
    cartesian: tuple
    refuted_by: Any
    scaled: float
    residual: Any
    tol: float


@dataclass(frozen=True)
class TranslationDressing:
    """A pure translation the Hamiltonian obeys, and the matrix it is obeyed with.

    Attributes
    ----------
    translation : tuple of float
        The translation, in reduced coordinates.
    turn : float
        How far the matrix turns each site's orbitals as it carries them onto
        its translate's, in radians: half the smallest arc of the unit circle
        that holds every eigenvalue of its blocks, so that a phase common to
        all of them is no turn.  0 where it carries them as written.
    matrix : ndarray
        The matrix, carrying each site's orbitals onto its translate's.
    """

    translation: tuple
    turn: float
    matrix: Any


@dataclass(frozen=True)
class RefutedTranslation:
    """A pure translation the Hamiltonian breaks, which nothing else refuted.

    Its permutation of the sites, with or without a change of basis within
    each site, does not commute with ``H(k)``, and no other measurement
    refuted it.

    Attributes
    ----------
    refuted_by : str
        ``"translation"``.
    residual : float
        The largest entry of ``P H(k) P^-1 - H(k)`` over the k-points
        measured, with ``P`` the permutation of the sites the translation
        makes, in the units of H.
    scaled, defect : float
        The same residual.
    scale : float
        The tolerance the residual is held to, in the units of H.
    orientation, dim : int
        Always 0.
    """

    scaled: float
    defect: float
    scale: float
    orientation: int
    dim: int
    refuted_by: str
    residual: float


@dataclass(frozen=True)
class DescentLattice:
    """What was measured of the pure translations of a cell that is not primitive.

    A cell is not primitive where its orbital centers, with their species, are
    carried onto themselves by a translation that is not a lattice vector of
    the model.

    Attributes
    ----------
    index : int
        The index of the model's lattice in the lattice of the pure
        translations the cell allows: 2 where it allows one pure translation.
    fractional_translations : tuple
        The pure translations the cell allows, in reduced coordinates, the
        identity left out.
    operations : tuple of LatticeTranslation
        Each pure translation, and what was measured of it.
    realized : tuple
        The translations the Hamiltonian obeys.  Empty on every record
        returned: a Hamiltonian that obeys a pure translation of its cell is
        written on a supercell, and is refused ``supercell``.
    dressing : tuple of TranslationDressing
        For each translation in ``realized``, the matrix it is obeyed with.
        Empty on every record returned, as ``realized`` is.
    refuted : dict
        If nonempty, ``{index: RefutedTranslation}``: the pure translations
        the Hamiltonian breaks that nothing else refuted.
    coset_partners_refuted : int
        The number of the cell's unitary operations refuted at the step that
        are not pure translations.
    note : str
        What was measured, in words.
    """

    index: int
    fractional_translations: tuple
    operations: tuple
    realized: tuple
    dressing: tuple
    refuted: dict
    coset_partners_refuted: int
    note: str


@dataclass(frozen=True)
class DescentStep:
    """One step of a descent: the group lowered, the group it arrived at, and why.

    Attributes
    ----------
    step : int
        The step's number, counted from 1.
    cell_group, subgroup : NamedGroup
        The group the step lowered, and the group it arrived at.
    dropped : tuple of DroppedOperation
        The operations the step dropped.
    n_dropped : int
        The number of operations the step dropped.
    refuted_by : dict
        ``{refuted_by: count}``: the number of operations dropped for each
        value of :attr:`DroppedOperation.refuted_by`.
    kept : tuple of int
        The unitary operations kept, by index in ``cell_group``.
    pairs : tuple
        Each unitary operation kept, as its index in ``cell_group`` and its
        index in ``subgroup``.
    cell : DescentCell
        The orbit added to the cell so that its group is the group kept.
    declaration : StepDeclaration or None
        If not ``None``, what the declared site irreps kept.
    centers : StepCenters or None
        If not ``None``, what the orbital centers kept, where the whole of
        ``cell_group`` does not carry them within the tolerance declared for
        them.
    tie : StepTie or None
        If not ``None``, the subgroups of the largest order where the
        operations measured to hold do not close.
    projective : StepProjective or None
        If not ``None``, what the states' projective class kept, where it
        lowered the group.
    lattice : DescentLattice or None
        If not ``None``, what was measured of the pure translations of a cell
        that is not primitive.
    """

    step: int
    cell_group: NamedGroup
    subgroup: NamedGroup
    dropped: tuple
    n_dropped: int
    refuted_by: dict
    kept: tuple
    pairs: tuple
    cell: DescentCell
    declaration: StepDeclaration | None = None
    centers: StepCenters | None = None
    tie: StepTie | None = None
    projective: StepProjective | None = None
    lattice: DescentLattice | None = None


@dataclass(frozen=True)
class Descent:
    """The operations of the cell's group that were dropped, and why.

    Held in :attr:`RealizedGroup.descent`, which is ``None`` where no
    operation was dropped.  A descent lowers the cell's group to the group the
    Hamiltonian obeys: the group of the operations kept is identified again,
    in one step or several.  An operation is dropped where the hoppings break
    it, or where it does not carry the declared site irreps or the orbital
    centers.  An operation the hoppings obey is dropped where the states carry
    no single-valued representation of a group containing it.  It is also
    dropped where the operations that hold do not close into a group and it
    lies outside the subgroup kept.

    Where several subgroups of the largest order qualify and nothing measured
    tells them apart, the tie rule keeps one: on a layer whose group contains
    operations that turn it over, the one with the most operations that keep
    its side; then the one with the most rotations; then the one with the
    highest-order rotation; then the one whose axis comes first in the
    conventional cell used by the irreptables package.

    Attributes
    ----------
    cell_group : NamedGroup
        The group the cell allows, before the first step.
    subgroup : NamedGroup
        The group the last step arrived at.  Where time reversal is then
        measured broken, the group returned is its unitary half.
    dropped : tuple of DroppedOperation
        Every operation dropped, every step's together.
    dropped_by : dict
        ``{cause: count}``: how many operations each cause dropped, every step
        together, in the order the measurement meets the causes.  The causes
        are ``"centers"``, ``"hoppings"``, ``"declaration"``, ``"closure"``,
        ``"tie"``, ``"time reversal"`` and ``"projective class"``, each that
        dropped any.  ``"hoppings"`` counts the operations refuted by ``"no
        symmetry matrix"``, ``"covariance"``, ``"tensor structure"``, ``"out
        of plane"`` and ``"translation"``.  ``"time reversal"`` counts the
        operations combined with time reversal, where time reversal is
        measured broken on the group one step arrived at, before the next
        step.  The counts add up to the order of ``cell_group`` less that of
        ``subgroup``, unless the declared site irreps lowered the cell between
        two steps.
    tied : tuple
        If nonempty, the subgroups of the largest order that carry the
        declared site irreps, every step's together, each by its operations'
        indices, in the order of the tie rule, the one kept first.
    cell : DescentCell
        The orbit the last step added to the cell.
    iterations : int
        The number of steps.
    steps : tuple of DescentStep
        The steps, in order.
    lattice : DescentLattice or None
        If not ``None``, what was measured of the pure translations, from the
        first step whose cell is not primitive.
    note : str
        The descent in words: what dropped the operations, counted by cause,
        and the group arrived at.
    """

    cell_group: NamedGroup
    subgroup: NamedGroup
    dropped: tuple
    dropped_by: dict
    tied: tuple
    cell: DescentCell
    iterations: int
    steps: tuple
    lattice: DescentLattice | None
    note: str


@dataclass(frozen=True)
class TimeReversalSquare:
    """The square of time reversal on the states, expected and measured.

    With ``X`` the matrix of time reversal, its square is ``X conj(X)``.  Where
    the square is a multiple of the identity, its sign is one no rescaling of
    ``X`` changes.  Where the sign measured differs from the sign expected,
    time reversal is broken, as for spinor states read without their spin.

    Attributes
    ----------
    asked : int
        The sign the tables of the irreptables package expect: ``+1`` for the
        single-valued tables, ``-1`` for the double-valued ones.
    measured : float
        The sign of the square of the Hamiltonian's matrix of time reversal;
        ``nan`` where that square is not a multiple of the identity.
    """

    asked: int
    measured: float


@dataclass(frozen=True)
class TimeReversal:
    """How time reversal was measured.

    Held in :attr:`RealizedGroup.time_reversal`.  The group the unitary
    operations settle on is taken as a grey group, since positions say nothing
    about time reversal.  Each of its operations combined with time reversal
    is then measured on ``H``.

    Attributes
    ----------
    outcome : str
        ``"kept"``: every operation combined with time reversal holds, and the
        group returned is grey.  ``"measured broken"``: time reversal does not
        hold, and the group returned is the unitary group.
    n_antiunitary, n_obeyed, n_broken : int or None
        The number of operations combined with time reversal in the grey group
        measured, and how many of them hold and how many do not.  Where time
        reversal is measured broken, an operation combined with it whose
        unitary part is in the group returned does not hold, so ``n_obeyed``
        is 0 unless a later descent by the states' projective class left that
        unitary part out.
    undetermined : tuple or None
        If nonempty, the indices of those that hold and share their
        measurement on ``H(k)`` with another operation acting alike on every
        k, as an operation and its product with the horizontal mirror do on a
        flat layer.  Each of them is held by its tensor structure, the form of
        its matrix on the orbitals and on the spin.
    grey_name, grey_number_str : str or None
        The grey group time reversal was measured on, its symbol and its
        number.
    square : TimeReversalSquare or None
        If not ``None``, the square of time reversal on the states.  ``None``
        where the measurement found no matrix for time reversal.
    note : str or None
        The measurement in words.
    name, number_str : str or None
        Where time reversal is measured broken, the unitary group returned,
        its symbol and its number.
    size : int or None
        Where time reversal is measured broken, the number of operations of
        the unitary group returned.
    """

    outcome: Any = None
    n_antiunitary: Any = None
    n_obeyed: Any = None
    n_broken: Any = None
    undetermined: Any = None
    grey_name: Any = None
    grey_number_str: Any = None
    square: TimeReversalSquare | None = None
    note: Any = None
    name: Any = None
    number_str: Any = None
    size: Any = None


@dataclass(frozen=True)
class BeyondGreyOperation:
    """One antiunitary operation that holds while time reversal itself is broken.

    Attributes
    ----------
    index : int
        The operation, by index in the group it was measured in.
    name : str
        Its name, in the Cartesian axes in which the pythtb Lattice is
        specified, with a ``'`` for time reversal.  Where its translation is
        not a lattice vector of the cell, as for a screw rotation, a glide or
        a pure translation, the name carries the translation in Seitz form,
        ``{E|1/2,0,0}'``.
    of : str
        The number of the group it was measured in, in the tables of the
        irreptables package.
    rotation, translation : tuple
        Its rotation matrix and its translation, in the cell's reduced
        coordinates.
    residual : float
        Its covariance residual, as :class:`DroppedOperation` defines it, in
        the units of H.
    verdict : str
        ``"obeyed"``; or ``"undetermined"``, where it shares its measurement
        on ``H(k)`` with another operation acting alike on every k, and is
        held by its tensor structure.
    """

    index: int
    name: str
    of: Any = None
    rotation: Any = None
    translation: Any = None
    residual: Any = None
    verdict: Any = None


@dataclass(frozen=True)
class BeyondGrey:
    """Antiunitary operations that hold while time reversal itself is broken.

    Held in :attr:`RealizedGroup.beyond_grey`, which is ``None`` where none
    holds.  Each combines time reversal with a unitary operation of the cell
    that a descent left out, so none is in the group returned, and the
    Hamiltonian's magnetic group is larger than the group returned.
    ``magnetic_type`` names the type of a magnetic group whose antiunitary
    operations they are.

    Attributes
    ----------
    operations : tuple of BeyondGreyOperation
        The operations that hold.
    n_operations, n_measured : int
        The number that hold, and the number of operations of the cell
        combined with time reversal that were measured.
    magnetic_type : str
        ``"III"`` where none of them is a translation combined with time
        reversal, and ``"IV"`` where one is.
    closed : bool
        Whether the operations the Hamiltonian was measured to obey in the
        group these were measured in, unitary and antiunitary together, form
        a group.
    note : str
        What they are, in words.
    """

    operations: tuple
    n_operations: int
    n_measured: int
    magnetic_type: Any
    closed: Any
    note: Any


@dataclass(frozen=True)
class ConservedSector:
    """One sector the conserved matrices acting within sites split the states into.

    Attributes
    ----------
    orbitals : tuple of int
        The orbitals its states lie on.
    sites : tuple of int
        The sites those orbitals sit at.
    copies, dim : int
        The sector holds ``copies`` identical copies of a set of ``dim``
        states, counted as orbitals, or as spin pairs where ``spin=`` declares
        them; the spin of a model built with spin is not counted.  The
        conserved matrices on it are every matrix acting on the copies, times
        the identity on each copy's states.
    """

    orbitals: tuple
    sites: tuple
    copies: int
    dim: int


@dataclass(frozen=True)
class Conserved:
    """Symmetry beyond the crystal's, counted by the matrices commuting with H(k).

    Returned by :attr:`RealizedGroup.conserved`.  The little-group
    representations, elementary band representations and symmetry indicators
    returned are those of the space group, whatever ``dim`` is.  Where ``dim``
    is above 0, the Hamiltonian may not fix the site irreps of orbitals it
    decouples from the rest.  Each row's ``determined_by`` in
    :attr:`RealizedGroup.wyckoff_orbits` says whether a declaration or the
    lowest-angular-momentum rule fixed them.

    Attributes
    ----------
    dim : int
        The number of matrices beyond the identity that commute with H(k) at
        every k: 0 for a generic model, and 1 for Kane-Mele without its Rashba
        term, which conserves ``s_z``.
    note : str
        The same number, in a sentence.
    sectors : tuple of ConservedSector or None
        If not ``None``, the sectors the conserved matrices acting within
        sites split the states into.  They are read once, where the model
        enters, at the declared precision, and a spin rotation is never among
        the matrices read.  ``None`` where they could not be read.
    images : dict or None
        If nonempty, the sector each operation carries each sector onto,
        ``{index: images}``, with ``images[i]`` the image of sector ``i``.
        ``None`` where ``sectors`` is ``None``.
    """

    dim: int
    note: str
    sectors: tuple | None = None
    images: dict | None = None


@dataclass(frozen=True)
class CrossCheckRow:
    """One operation of the substrate group, and its index in the free-standing one.

    Attributes
    ----------
    substrate : int
        The operation, by index in the substrate group.
    free_standing : int or None
        The same operation, by index in the free-standing group; ``None``
        where that group does not contain it.
    name : str
        Its name, in the Cartesian axes in which the pythtb Lattice is
        specified.
    rotation, translation : tuple
        Its integer rotation matrix, written row by row as nine entries, and
        its translation modulo the lattice, in the cell's reduced coordinates.
    antiunitary : bool
        Whether it is combined with time reversal.
    """

    substrate: int
    free_standing: Any
    name: str
    rotation: Any
    translation: Any
    antiunitary: bool


@dataclass(frozen=True)
class CrossCheckOperation:
    """One operation named in comparing a layer on a substrate with it free-standing.

    It stands in one of the comparison's lists: an operation missing from a
    group or extra in it, one that turns the layer over, one that is not a
    layer operation, or one the tie rule left out.

    Attributes
    ----------
    index : int
        The operation, by index in its group.
    name : str
        Its name, in the Cartesian axes in which the pythtb Lattice is
        specified.
    group : str or None
        If not ``None``, the group it is in, ``"substrate"`` or
        ``"free-standing"``, where the list it stands in does not say.
    environment : str or None
        If not ``None``, the environment in which the tie rule left it out.
    """

    index: int
    name: str
    group: Any = None
    environment: Any = None


@dataclass(frozen=True)
class CrossCheckRefusal:
    """The refusal returned where the layer was read in the other environment.

    Held in :attr:`CrossCheck.refusal`.  A refusal there is not a mismatch
    between the two environments.  The group of the environment asked for is
    still returned.

    Attributes
    ----------
    gate : str
        The name of the check that stopped the reading of the other
        environment.
    reason : str
        The refusal's reason.
    environment : str
        The other environment, ``"substrate"`` or ``"free-standing"``.
    """

    gate: str
    reason: str
    environment: str


@dataclass(frozen=True)
class CrossCheck:
    """A layer's other environment, measured and compared with the one asked for.

    Held in :attr:`LayerEnvironment.cross_check`, which is ``None`` with
    ``cross_check=False``.  The layer is read on a substrate and free-standing,
    each on its own.  The two groups agree where the substrate group is the
    subgroup of the free-standing group made of the operations that do not
    turn the layer over.

    Attributes
    ----------
    free_standing : NamedGroup or None
        The group of the layer free-standing.  ``None`` where that is the
        other environment and it refused or was not compared.
    substrate : NamedGroup or None
        The group of the layer on a substrate.  ``None`` where that is the
        other environment and it refused or was not compared.
    agrees : bool or None
        ``True`` where the two groups agree, and ``None`` otherwise; never
        ``False``.  A disagreement that the tie rule does not account for is
        not returned.
    outcome : str or None
        ``"agree"``: the two groups agree.  ``"ambiguous"``: in one
        environment several subgroups of the largest order qualified, and the
        tie rule (:class:`Descent`) kept the first of them.  The groups
        returned do not agree, and another pair of candidates, one from each
        environment, does, which ``note`` names.  ``"refused"``: the other
        environment refused.  ``"not compared"``: ``characters=`` names site
        irreps the other environment's site groups do not have, so the other
        environment is not read.  ``None`` where the free-standing cell and
        the substrate's cell do not have the same single non-periodic
        direction.
    rows : tuple of CrossCheckRow
        If nonempty, each operation of the substrate group, paired with the
        same operation in the free-standing group.  Empty where the two
        groups were not compared.
    missing : tuple of CrossCheckOperation
        If nonempty, the operations of the substrate group that the
        free-standing group does not contain.
    extra : tuple of CrossCheckOperation
        If nonempty, the operations of the free-standing group that do not
        turn the layer over and that the substrate group does not contain.
    flipping : tuple of CrossCheckOperation
        If nonempty, the operations of the substrate group that turn the layer
        over.
    unmapped : tuple of CrossCheckOperation
        The operations of either group whose rotation mixes the layer's plane
        with its non-periodic direction.  Empty on every record returned:
        every operation of a group returned keeps the layer's plane.
    tie : tuple of CrossCheckOperation
        If nonempty, the operations the tie rule left out in an environment
        where it chose the group.
    refusal : CrossCheckRefusal or None
        If not ``None``, the refusal the other environment returned.
    note : str
        The comparison in words.
    """

    free_standing: NamedGroup | None
    substrate: NamedGroup | None
    agrees: Any
    outcome: Any
    rows: tuple
    missing: tuple
    extra: tuple
    flipping: tuple
    unmapped: tuple
    tie: tuple
    refusal: CrossCheckRefusal | None
    note: str


@dataclass(frozen=True)
class Presentation:
    """How a layer was written as a slab, and the check that ``H(k)`` is unchanged.

    Held in :attr:`LayerEnvironment.presentation`.  A layer is read as a slab:
    the layer in a three-dimensional cell, with vacuum along its normal.  The
    slab's ``H(k)`` is measured against the layer's.

    Attributes
    ----------
    c : float
        The least length of the slab's lattice vector along the layer's
        normal, in the model's length units.  A layer written in two
        dimensions is given a vector of this length, and a slab's shorter
        vector is stretched to it.
    residual : float
        The largest difference between the slab's ``H(k)`` and the layer's,
        over ``n_check`` random ``k``, in the units of ``H``.  Where
        ``tilted`` is not ``None``, the layer's ``H(k)`` is first written in
        the basis of the moved orbitals.
    n_check : int
        The number of random ``k`` at which ``residual`` is measured.
    tilted : PresentationTilt or None
        If not ``None``, how a slab's non-periodic lattice vector, tilted off
        the layer's normal, was put on the normal.  ``None`` where the vector
        was on the normal already, and for a layer written in two dimensions.
    """

    c: Any
    residual: float
    n_check: int
    tilted: Any = None


@dataclass(frozen=True)
class PresentationTilt:
    """A slab's tilted non-periodic lattice vector, put on the layer's normal.

    Held in :attr:`Presentation.tilted`.  A layer's group does not depend on
    how that vector is chosen, since there is no translation along it.  Every
    orbital keeps its Cartesian position, so its reduced coordinates in the
    plane move by its reduced height times the tilt.  The move is a
    ``k``-dependent change of basis, which leaves every energy unchanged.

    Attributes
    ----------
    tilt : tuple of float
        The vector's in-plane part, in reduced coordinates of the periodic
        directions.
    max_move : float
        The largest move of an orbital's reduced coordinates in the plane.
    n_moved : int
        The number of orbitals moved, those off the plane through the origin.
    spectrum_residual : float
        The largest difference between the spectra before and after the move,
        over the ``k`` at which :attr:`Presentation.residual` is measured, in
        the units of ``H``.
    note : str
        The move in words.
    """

    tilt: tuple
    max_move: float
    n_moved: int
    spectrum_residual: Any
    note: str


@dataclass(frozen=True)
class LayerEnvironment:
    """A layer's environment, and the other environment measured beside it.

    Held in :attr:`RealizedGroup.environment`, which is ``None`` for a crystal
    periodic in three directions.  On a substrate, no operation turns the layer
    over.  Free-standing, the whole layer group applies.

    Attributes
    ----------
    kind : str
        The environment the layer was read in, ``"substrate"`` or
        ``"free-standing"``.
    declared : bool
        Whether ``environment=`` gave it.  Where it did not, a layer written in
        two dimensions is read on a substrate, and a slab written in three
        dimensions is read free-standing.
    cross_check : CrossCheck or None
        If not ``None``, the other environment, measured and compared.
        ``None`` with ``cross_check=False``.
    presentation : Presentation
        How the layer was written as a slab.
    """

    kind: str
    declared: bool
    cross_check: CrossCheck | None
    presentation: Presentation


@dataclass(frozen=True)
class ExcludedOperation:
    """One operation of the cell that the declaration of the orbitals takes out.

    Held in :attr:`DeclaredReduction.excluded`.  The operation is not one of
    the crystal ``characters=`` declares.

    Attributes
    ----------
    index : int
        The operation, by index in the cell's group.
    name : str
        Its name, in the Cartesian axes in which the pythtb Lattice is
        specified.
    reason : str
        Why the operation is taken out, in words: it carries a site onto one
        declared differently; it does not carry the declaration at a site it
        fixes onto itself; it does not carry a declared function onto itself,
        or onto the declaration at the site it moves it to; a product of it
        with an operation kept does not carry the declaration; or a group
        containing it carries the declaration, and the tie rule
        (:class:`Descent`) keeps another of the same order.
    sites : tuple of int
        The site the declaration was read at, and the site the operation
        carries it onto, by index.
    orbitals : tuple of int
        The orbitals declared at those two sites.
    note : str
        The exclusion in words.
    """

    index: int
    name: str
    reason: str
    sites: tuple
    orbitals: tuple
    note: str


@dataclass(frozen=True)
class DeclaredReduction:
    """The cell's group, lowered by the declaration of the orbitals.

    Held in :attr:`Decoration.declared`, which is ``None`` where the
    declaration takes no operation out.  The operations of the cell that do not
    carry what ``characters=`` declares are not operations of the declared
    crystal.  Before any hopping is read, the orbit of a generic point under
    the operations kept is added to the cell, and the cell's group is found
    again.  No operation taken out here is among those of
    :attr:`RealizedGroup.descent`.

    Attributes
    ----------
    cell_group : NamedGroup
        The group the cell allows.
    subgroup : NamedGroup
        The group the cell is identified as once the operations are taken out.
    excluded : tuple of ExcludedOperation
        Each operation taken out, with the reason.
    n_excluded : int
        The number of operations taken out, one for each coset of the lattice
        translations.
    kept : tuple of int
        The operations kept, by index in ``cell_group``.
    generic_point : tuple of float
        The generic point added to the cell, in reduced coordinates.
    orbit : tuple
        The orbit of that point under the operations kept, each point in
        reduced coordinates.
    species : str
        The species label the orbit is added under.
    pairs : tuple
        Each unitary operation kept, as its index in ``cell_group`` and its
        index in ``subgroup``.
    """

    cell_group: NamedGroup
    subgroup: NamedGroup
    excluded: tuple
    n_excluded: int
    kept: tuple
    generic_point: Any
    orbit: Any
    species: Any
    pairs: Any


@dataclass(frozen=True)
class Decoration:
    """The points added to the cell before its group was found.

    Held in :attr:`RealizedGroup.decoration`, which is ``None`` for a crystal
    periodic in three directions to which nothing was added.  On a substrate,
    one point of a species of its own is added under every site of the layer,
    which removes every operation that turns the layer over.
    ``decorations=`` adds its points, and a declaration that lowers the cell's
    group adds an orbit.

    Attributes
    ----------
    dz : float or None
        If not ``None``, the depth of the substrate's points under the sites,
        in reduced coordinates of the lattice vector normal to the layer.
        ``None`` where no substrate was added.
    typat : tuple
        The species label of each point added.
    n_atoms : int
        The number of points added.
    declared : DeclaredReduction or None
        If not ``None``, the cell's group lowered by the declaration of the
        orbitals.  ``None`` where the declaration takes no operation out.
    recipe : str
        What was added and why, in words.
    """

    dz: Any
    typat: tuple
    n_atoms: int
    declared: DeclaredReduction | None
    recipe: str


@dataclass(frozen=True)
class Positions:
    """The orbital centers, moved onto the positions the group puts them at.

    Held in :attr:`RealizedGroupNumerical.positions`, which is ``None`` where
    neither ``precision=`` nor ``position_tol=`` was declared.  The centers
    are moved onto the positions that the largest group of the cell's
    operations carrying them within ``position_tol`` puts them at.  Those
    positions are the set that group carries exactly which lies nearest the
    centers, its origin fitted to them.  Each center's move onto those
    positions is at most ``position_tol``, and so is the shift ``offset`` of
    every center together, both in reduced coordinates.  Moving a center is a
    ``k``-dependent change of basis, which leaves every energy unchanged.

    Attributes
    ----------
    defect : float
        How far the centers as handed in miss the action of the whole group,
        in reduced coordinates.
    position_tol : float
        The tolerance within which a center is moved, in reduced coordinates.
        It is ``position_tol=`` as declared, or ``precision=``.
    adjusted : bool
        Always ``True``.  ``n_moved`` says how many orbitals were moved.
    defect_before : float
        How far the centers as handed in miss the action of the operations
        that carry them, in reduced coordinates.
    defect_after : float
        How far the moved centers miss the action of the same operations, in
        reduced coordinates.
    max_move : float
        The largest move of a center onto the positions the group puts it at,
        about the fitted origin, in reduced coordinates; ``offset`` is not
        included.
    offset : tuple
        The common shift of all centers, in reduced coordinates, that writes
        them in the model's own frame (see ``frame``).  It is zero where
        ``frame`` is ``"fitted"``, and at most ``position_tol`` otherwise.
        Each center returned is the one handed in, plus its own move, plus
        ``offset``.
    frame : str
        ``"model"``: in the model's own origin, every operation's translation
        is a rational fraction of the lattice vectors.  ``"fitted"``: the
        shift that would write the centers so is larger than
        ``position_tol``, or there is none, and the centers stay about the
        origin fitted to them.
    spectrum_residual : float
        The largest difference between the spectra before and after the move,
        over ``n_check`` random ``k``, in the units of ``H``.
    n_check : int
        The number of random ``k`` at which ``spectrum_residual`` is measured,
        and 0 where no center moved.
    n_moved : int
        The number of orbitals moved.
    move : float or None
        If not ``None``, the farthest a center must move for the centers to be
        symmetric under the whole group, in reduced coordinates, which is
        larger than ``position_tol``.  The operations that do not carry the
        centers are dropped, and :attr:`RealizedGroup.descent` lists them as
        broken by the centers.
        ``None`` where the whole group carries the centers.
    carried : int or None
        If not ``None``, the number of operations of the largest group that
        carries the centers, one for each coset of the lattice translations,
        those combined with time reversal included.  ``None`` where the whole
        group carries them.
    note : str
        The move in words.
    """

    defect: Any = None
    position_tol: Any = None
    adjusted: Any = None
    defect_before: Any = None
    defect_after: Any = None
    max_move: Any = None
    offset: Any = None
    frame: Any = None
    spectrum_residual: Any = None
    n_check: Any = None
    n_moved: Any = None
    move: Any = None
    carried: Any = None
    note: Any = None


@dataclass(frozen=True)
class Precision:
    """The tolerances a declared precision sets.

    Held in :attr:`RealizedGroupNumerical.precision`, which is ``None`` where
    no precision was declared.  Each tolerance is computed from the declared
    precision, and ``tol`` from ``max_h`` as well.

    Attributes
    ----------
    precision : float
        The precision declared with ``precision=``, a fraction: of ``max|H|``
        for the hoppings, and a length in reduced coordinates for the orbital
        centers.
    max_h : float
        ``max|H|``, the largest entry of ``H(k)`` once the mean on-site energy
        is subtracted, in the units of ``H``.
    tol : float
        A tolerance for checking whether an operation is a symmetry, in the
        units of ``H``.  An operation is obeyed where its covariance residual,
        the largest entry of ``U H(k) U^-1 - H(gk)`` over the k measured for
        the best matrix ``U`` found, is at most ``tol``.  It decides whether
        an operation holds; ``cut`` decides which matrices are found for it.
        Computed from ``precision`` and ``max_h``.
    position_tol : float
        A tolerance for moving an orbital center onto the position the group
        puts it at, in reduced coordinates.  It equals ``precision``.
    cut : float
        A dimensionless tolerance for checking whether an operation is a
        symmetry.  For each operation, the singular values of the symmetry
        equations are computed.  A singular value is treated as zero when it
        is at most ``cut`` times the largest singular value.  Computed from
        ``precision``.
    window : float
        A dimensionless tolerance for checking degeneracy of ``H``.  Two
        eigenvectors are treated as degenerate when their energies differ by
        at most ``window`` times ``max|H|``.  Computed from ``precision``.
    symprec : float
        The tolerance of spglib, which finds the group from the positions, as
        a Cartesian length in the model's length units.  It is ``symprec=`` as
        asked, raised to ``10 x precision x L`` where ``precision x L``
        exceeds it, with ``L`` the length of the longest periodic lattice
        vector.
    symprec_raised : bool
        Whether the declared precision raised ``symprec``.
    """

    precision: float
    max_h: float
    tol: float
    position_tol: float
    cut: float
    window: float
    symprec: float
    symprec_raised: bool


@dataclass(frozen=True)
class RealizedGroupNumerical:
    """How the group was measured: tolerances, centers, projection, measurements.

    Held in :attr:`RealizedGroup.numerical`.  ``summary()`` prints a line for
    each part that applies, with the field that holds it.

    Attributes
    ----------
    precision : Precision or None
        If not ``None``, the tolerances the declared precision sets.  ``None``
        where no precision was declared.
    positions : Positions or None
        If not ``None``, the orbital centers moved onto the positions the group
        puts them at.  ``None`` where neither ``precision=`` nor
        ``position_tol=`` was declared.
    symmetrization : ..symmetrization.Symmetrization or None
        If not ``None``, the projection of ``H`` onto its symmetric part.
        ``None`` where ``H`` was not projected.
    validated : ..groups.ValidatedGroup
        What was measured of each operation of the group returned.
    source : pythtb.TBModel
        The model as it was handed in.
    """

    precision: Precision | None
    positions: Positions | None
    symmetrization: Any
    validated: ValidatedGroup
    source: Any

    def _lines(self) -> list:
        """The lines :meth:`summary` prints, none of them indented."""
        lines = []
        pr = self.precision
        if pr is not None:
            lines.append(f"precision {pr.precision:.3g} declared, max|H| {pr.max_h:.4g}: tol "
                         f"{pr.tol:.3g}, cut {pr.cut:.3g}, window {pr.window:.3g}, "
                         f"position_tol {pr.position_tol:.3g}"
                         + (f", symprec raised to {pr.symprec:.3g}" if pr.symprec_raised else "")
                         + "   (.precision)")
        ps = self.positions
        if ps is not None and ps.adjusted and not within(ps.max_move or 0.0, 0.0):
            lines.append(f"centers moved by {ps.max_move:.3g}   (.positions)")
        sz = self.symmetrization
        if sz is not None:
            lines.append(f"H symmetrized by {sz.defect:.3g}, "
                         f"{sz._relative:.3g} of its size"
                         "   (.symmetrization)")
        lines.append(self.validated._counted() + "   (.validated)")
        return lines

    def summary(self) -> str:
        """How the group was measured, a line for each part that applies.

        Each line ends with the field that holds that part.
        """
        lines = self._lines()
        return "\n".join([lines[0]] + ["  " + line for line in lines[1:]])

    def __repr__(self) -> str:  # pragma: no cover
        return "RealizedGroupNumerical(" + "; ".join(
            line.split("   (")[0] for line in self._lines()) + ")"


#: For each key of a measurement's mapping, the record field it is read into,
#: where the two names differ.
_RENAMED = {"from": "cell_group", "to": "subgroup", "ind": "index",
            "time_reversal": "antiunitary"}


def _record(cls, mapping, **built):
    """One mapping the measurement builds, read into the record class ``cls``.

    Each key of ``mapping`` fills the field of its name, or the field
    :data:`_RENAMED` gives it.  ``built`` are fields already made, nested records
    among them, and stand in place of the keys of the same names.  Returns
    ``None`` where ``mapping`` is ``None``.  A key no field of ``cls`` takes
    raises ``InternalError``.
    """
    if mapping is None:
        return None
    names = {f.name for f in dataclass_fields(cls)}
    out = {}
    for key, value in dict(mapping).items():
        name = _RENAMED.get(key, key)
        if name in built:
            continue
        if name not in names:
            raise InternalError(
                f"the record {cls.__name__} has no field for the key {key!r} that the "
                "measurement wrote. Report it, with the model and the call.")
        out[name] = value
    out.update(built)
    return cls(**out)


def _named(mapping) -> NamedGroup | None:
    """The group ``mapping`` holds as ``"name"``, ``"number_str"`` and ``"size"``.

    Returns it as a :class:`NamedGroup`, or ``None`` where ``mapping`` is ``None``.
    """
    if mapping is None:
        return None
    return NamedGroup(name=str(mapping["name"]), number_str=str(mapping["number_str"]),
                      size=int(mapping["size"]))


def _dropped(entry) -> DroppedOperation:
    """One dropped operation's mapping ``entry`` as a :class:`DroppedOperation`.

    Its ``"rows"``, where present, are read by
    :func:`.symmetry._declaration_checks`.
    """
    rows = entry.get("rows")
    return _record(DroppedOperation, entry,
                   **({} if rows is None else {"rows": _declaration_checks(rows)}))


def _causes(counts) -> dict:
    """The number of operations each cause dropped, read from the mapping ``counts``."""
    return {str(cause): int(n) for cause, n in dict(counts).items()}


def _projective(rec) -> StepProjective | None:
    """A descent step's projective class ``rec`` as a :class:`StepProjective`.

    Returns ``None`` where ``rec`` is ``None``.
    """
    if rec is None:
        return None
    mirror, read_on = rec.get("horizontal_mirror"), rec.get("read_on")
    return _record(
        StepProjective, rec,
        pairs=tuple(_record(ProjectivePair, pair) for pair in rec["pairs"]),
        horizontal_mirror=_record(ProjectiveMirror, mirror),
        read_on=(None if read_on is None else _record(
            ProjectiveReading, {"kept": read_on["kept"]}, group=_named(read_on))))


def _lattice(rec) -> DescentLattice | None:
    """What ``rec`` says of a cell's pure translations, as a :class:`DescentLattice`.

    Returns ``None`` where ``rec`` is ``None``.
    """
    if rec is None:
        return None
    return _record(
        DescentLattice, rec,
        operations=tuple(_record(LatticeTranslation, entry) for entry in rec["operations"]),
        dressing=tuple(_record(TranslationDressing, entry) for entry in rec["dressing"]),
        refuted={int(index): _record(RefutedTranslation, entry)
                 for index, entry in dict(rec["refuted"]).items()})


def _step(rec) -> DescentStep:
    """One descent step's mapping ``rec`` as a :class:`DescentStep`."""
    decl = rec.get("declaration")
    return _record(
        DescentStep, rec, cell_group=_named(rec["from"]), subgroup=_named(rec["to"]),
        dropped=tuple(_dropped(d) for d in rec["dropped"]),
        refuted_by=_causes(rec["refuted_by"]),
        cell=_record(DescentCell, rec["cell"]),
        declaration=(None if decl is None else _record(
            StepDeclaration, decl, rows=_declaration_checks(decl.get("rows", ())))),
        centers=_record(StepCenters, rec.get("centers")),
        tie=_record(StepTie, rec.get("tie")),
        projective=_projective(rec.get("projective")),
        lattice=_lattice(rec.get("lattice")))


def _descent_record(descent) -> Descent | None:
    """The mapping ``descent`` as a :class:`Descent`; ``None`` where it is ``None``."""
    if descent is None:
        return None
    return _record(
        Descent, descent, cell_group=_named(descent["from"]),
        subgroup=_named(descent["to"]),
        dropped=tuple(_dropped(d) for d in descent["dropped"]),
        dropped_by=_causes(descent["dropped_by"]),
        tied=tuple(descent.get("tied", ())),
        cell=_record(DescentCell, descent["cell"]),
        steps=tuple(_step(s) for s in descent["steps"]),
        lattice=_lattice(descent.get("lattice")))


def _time_reversal_record(tr) -> TimeReversal:
    """The time-reversal verdict, the mapping ``tr``, as a :class:`TimeReversal`.

    ``None`` is read as an empty mapping.
    """
    tr = dict(tr or {})
    return _record(TimeReversal, tr,
                   square=_record(TimeReversalSquare, tr.get("square")))


def _beyond_grey_record(bg) -> BeyondGrey | None:
    """The operations ``bg`` beyond the grey group, as a :class:`BeyondGrey`.

    Returns ``None`` where ``bg`` is empty or ``None``.
    """
    if not bg:
        return None
    return _record(BeyondGrey, bg, operations=tuple(
        _record(BeyondGreyOperation, o) for o in bg.get("operations", ())))


def _cross_check_record(cc) -> CrossCheck | None:
    """A layer's cross-check, the mapping ``cc``, as a :class:`CrossCheck`.

    Returns ``None`` where ``cc`` is ``None``.
    """
    if cc is None:
        return None

    def ops(rows):
        return tuple(_record(CrossCheckOperation, o) for o in rows or ())

    return _record(
        CrossCheck, cc, free_standing=_named(cc.get("free_standing")),
        substrate=_named(cc.get("substrate")),
        rows=tuple(_record(CrossCheckRow, o) for o in cc.get("rows") or ()),
        missing=ops(cc.get("missing")), extra=ops(cc.get("extra")),
        flipping=ops(cc.get("flipping")), unmapped=ops(cc.get("unmapped")),
        tie=ops(cc.get("tie")), refusal=_record(CrossCheckRefusal, cc.get("refusal")))


def _decoration_record(dec) -> Decoration | None:
    """What was added to the cell, the mapping ``dec``, as a :class:`Decoration`.

    Returns ``None`` where ``dec`` is ``None``.
    """
    if dec is None:
        return None
    red = dec.get("declared")
    return _record(Decoration, dec, declared=(None if red is None else _record(
        DeclaredReduction, red, cell_group=_named(red["from"]), subgroup=_named(red["to"]),
        excluded=tuple(_record(ExcludedOperation, d) for d in red.get("excluded", ())))))


@dataclass(frozen=True)
class RealizedGroup:
    """The space group a Hamiltonian obeys, and what was measured with it.

    The record :func:`realized_group` returns.  ``summary()`` prints what was
    measured, a line for each part, with the field that holds it.  Every call
    that labels the bands takes the record whole, as ``hsp_table(r, bands=n)``
    does.

    Attributes
    ----------
    group : GroupData
        The space group the Hamiltonian obeys, every operation of it measured
        to hold.
    operations : OperationTable
        The operations of ``group`` by index, one for each coset of the
        lattice translations, each with its Seitz symbol and its name.
    wyckoff_orbits : ..symmetry.AtomicLimitTable
        The model's orbitals, a row for each Wyckoff orbit: its Wyckoff
        position, its site-symmetry group, the site irreps its orbitals carry
        and the band representation they induce.
    descent : Descent or None
        If not ``None``, the operations of the cell's group that were dropped,
        and why.  ``None`` where no operation was dropped.
    time_reversal : TimeReversal
        How time reversal was measured.
    beyond_grey : BeyondGrey or None
        If not ``None``, the operations of the cell combined with time reversal
        that hold while time reversal itself is broken.  ``None`` where none
        holds.
    conserved : Conserved
        How many matrices beyond the identity commute with ``H(k)`` at every
        ``k``, and the sectors they split the states into: symmetry beyond the
        crystal's own.
    environment : LayerEnvironment or None
        If not ``None``, a layer's environment, and the other environment
        measured beside it.  ``None`` for a crystal periodic in three
        directions.
    decoration : Decoration or None
        If not ``None``, the points added to the cell before its group was
        found.  ``None`` for a crystal periodic in three directions to which
        nothing was added.
    model : pythtb.TBModel
        The model every later call reads.  For a layer it is the slab, with
        the same ``H(k)``, or the same up to the change of basis that
        ``environment.presentation.tilted`` records.  For a model with free
        parameters, the values of ``params=`` are in place.  Where the orbital
        centers were moved, it holds the moved centers.  Where ``H`` was
        projected onto its symmetric part, it is the projected model.
    rep : ..symmetry.SpaceGroupRep
        The representation of the group on the orbitals: the matrix of each
        operation.
    spin : ..spin.SpinBasis or None
        If not ``None``, the spin pairs ``spin=`` declared.  ``None`` where
        none were declared.
    numerical : RealizedGroupNumerical
        How the group was measured: a declared precision, the centers moved,
        ``H`` projected, what was measured of each operation, and the model
        as it was handed in.
    """

    group: GroupData
    wyckoff_orbits: Any
    time_reversal: TimeReversal
    model: Any
    rep: Any
    numerical: RealizedGroupNumerical
    descent: Descent | None = None
    beyond_grey: BeyondGrey | None = None
    environment: LayerEnvironment | None = None
    decoration: Decoration | None = None
    spin: Any = None
    # the point set the group was found on, the orbitals' species and its own
    _cell: Any = field(default=None, repr=False, compare=False)
    _typat: Any = field(default=None, repr=False, compare=False)
    _cell_typat: Any = field(default=None, repr=False, compare=False)
    # each orbital's site and Wyckoff position, and how the operations move sites
    _assignment: OrbitalAssignment | None = field(default=None, repr=False, compare=False)
    # the sectors the conserved matrices acting within sites split the states
    # into, read once where the model entered, and what the operations of the
    # group returned do to them: read on `conserved`
    _sectors: Any = field(default=None, repr=False, compare=False)

    @property
    def conserved(self) -> Conserved:
        """How many matrices beyond the identity commute with ``H(k)`` at every ``k``.

        They are the matrices that solve the symmetry equations of the
        identity operation.  The record also holds the sectors they split the
        states into.

        Returns
        -------
        Conserved
            The number of such matrices, and the sectors into which the
            matrices that act within sites split the states.
        """
        validated = self.numerical.validated
        ident = next((int(s.ind) for s in self.group._symmetries
                      if not s.time_reversal
                      and np.array_equal(np.rint(np.asarray(s.rotation)).astype(int),
                                         np.eye(3, dtype=int))
                      and np.allclose(np.asarray(s.translation) % 1.0, 0.0, atol=1e-8)),
                     None)
        n = int((validated._nullspace_dims or {}).get(int(ident), 1) or 1) \
            if ident is not None else 1
        dim = max(0, n - 1)
        reading, action = self._sectors if self._sectors is not None else (None, None)
        sectors = images = None
        if reading is not None:
            site_of = np.asarray(self._assignment.site_of_orbital, dtype=int)
            sectors = []
            for sector in reading.sectors:
                orbitals = tuple(sorted(int(o) for a in sector.atoms
                                        for o in reading.atoms[int(a)]))
                sectors.append(ConservedSector(
                    orbitals=orbitals,
                    sites=tuple(sorted({int(site_of[o]) for o in orbitals})),
                    copies=int(sector.copies), dim=int(sector.dim)))
            sectors = tuple(sectors)
            images = ({} if action is None else
                      {int(i): tuple(int(j) for j in got)
                       for i, got in sorted(action.images.items())})
        return Conserved(
            dim=int(dim),
            note=(f"additional commuting matrices: {dim} -- this Hamiltonian "
                  f"commutes with {dim} matrix{'' if dim == 1 else 'es'} beyond "
                  "the identity at every k, so it has that much symmetry beyond "
                  "the crystal's own; the labels, EBR rows and indicators "
                  "returned here are the crystalline group's."
                  if dim else
                  "additional commuting matrices: 0 -- no matrix beyond the "
                  "identity commutes with this Hamiltonian at every k, so its "
                  "symmetry is the crystal's own."),
            sectors=sectors, images=images)

    @property
    def operations(self) -> OperationTable:
        """The operations of the group by index, with their Seitz symbols and names.

        ``r.operations[5].seitz`` is the Seitz symbol of operation 5.
        """
        return operations(self.group)

    def summary(self) -> str:
        """What was measured, a line for each part, with the field that holds it."""
        g = self.group
        kind = None if self.environment is None else self.environment.kind
        where = {"substrate": ", on a substrate", "free-standing": ", free-standing"}.get(
            kind, "")
        lines = [f"{g.name} #{g.number_str}{where}   "
                 + ("(.group)" if kind is None else "(.group, .environment.kind)"),
                 "  its operations, by index and Seitz symbol   (.operations)"]
        if len(self.wyckoff_orbits):
            lines.append("  the orbitals, by Wyckoff orbit   (.wyckoff_orbits)")
            lines += ["    " + o._line() for o in self.wyckoff_orbits]
        if self.spin is not None:
            lines.append(f"  the orbitals in {len(self.spin.pairs)} declared spinor pairs"
                         "   (.spin)")
        tr = self.time_reversal
        if tr.outcome == "measured broken":
            square = tr.square
            if (square is not None and square.asked > 0 and np.isfinite(square.measured)
                    and square.measured < 0):
                lines.append("  time reversal broken: its square is -1, so the states carry "
                             "half-integer spin; spin= declares the spin pairs   "
                             "(.time_reversal)")
        if self.beyond_grey is not None:
            names = [str(o.name) for o in self.beyond_grey.operations]
            shown = ", ".join(names[:6]) + (f" and {len(names) - 6} more" if len(names) > 6
                                            else "")
            lines.append(f"  also obeyed: {shown}, making a magnetic group of type "
                         f"{self.beyond_grey.magnetic_type}, which is not returned   "
                         "(.beyond_grey)")
        dim = int(self.conserved.dim)
        if dim:
            lines.append(("  one more matrix commutes" if dim == 1
                          else f"  {dim} more matrices commute")
                         + " with H(k) at every k   (.conserved.dim)")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover
        tr = self.time_reversal
        pr = self.numerical.precision
        positions = self.numerical.positions
        symmetrization = self.numerical.symmetrization
        red = None if self.decoration is None else self.decoration.declared
        # time reversal asked to square to +1 and measured to square to -1: the
        # states carry half-integer spin, and the line says that spin= declares
        # the spin partners
        square = tr.square
        spinor = (square is not None and square.asked > 0 and np.isfinite(square.measured)
                  and square.measured < 0)
        # the other environment, as the cross-check left it
        env = self.environment
        cc = None if env is None else env.cross_check
        if cc is None:
            compared = ""
        elif cc.agrees:
            other = cc.free_standing if env.kind == "substrate" else cc.substrate
            compared = (", the free-standing group " if env.kind == "substrate"
                        else ", the substrate group ") + f"{other} agrees"
        elif cc.outcome == "not compared":
            compared = (", not compared with the free-standing layer: the declaration is "
                        "written in the substrate's site irreps"
                        if env.kind == "substrate" else
                        ", not compared with the substrate: the declaration is written in "
                        "the free-standing layer's site irreps")
        else:
            compared = ", the other environment " + str(cc.outcome)
        # the operations a clause counts are coset representatives of the
        # lattice translations, and a spinful group's are the single-valued
        # group's: the first count of the line says so, once
        double = bool(getattr(self.group, "spinor", False))
        named = []

        def unit():
            if named:
                return "", ""
            named.append(True)
            return f" {_PER_COSET}", (f"; these are {_DOUBLE_GROUP}" if double else "")

        declared = ""
        if red is not None:
            u, d = unit()
            declared = (f"; cell lowered by the declared content from {red.cell_group} "
                        f"({red.n_excluded} of {red.cell_group.size}{u} excluded{d})")
        descended = ""
        descent = self.descent
        if descent is not None and len(descent.steps) > 1:
            # a descent of several steps is counted whole, by cause, as its
            # note counts it
            counts = descent.dropped_by
            u, d = unit()
            descended = (f"; descended from {descent.cell_group} in "
                         f"{len(descent.steps)} steps ({sum(counts.values())} of "
                         f"{descent.cell_group.size}{u} dropped: "
                         + ", ".join(f"{n} {_DROPPED_BY_WORDS[cause]}"
                                     for cause, n in counts.items())
                         + f"{d})")
        elif descent is not None:
            counts = descent.dropped_by
            n_all = sum(counts.values())
            u, d = unit()
            if set(counts) <= {"hoppings", "declaration", "centers"}:
                # every operation dropped was broken, by the hoppings, the
                # declared content or the centers, and the clause says so
                n_h, n_c, n_d = (int(counts.get(cause, 0))
                                 for cause in ("hoppings", "centers", "declaration"))
                parts = ([f"{n_h} by the hoppings"] if n_h else []) + (
                    [f"{n_c} by the centers"] if n_c else []) + (
                    [f"{n_d} by the declared content"] if n_d else [])
                why = ("" if not n_d and not n_c else
                       " by the declared content" if n_d == n_all else
                       " by the centers" if n_c == n_all else
                       ": " + ", ".join(parts))
                descended = (f"; descended from {descent.cell_group} "
                             f"({n_all} of {descent.cell_group.size}{u} broken{why}{d})")
            else:
                # another cause is among them, a closure, the tie rule, time
                # reversal or the states' projective class, which can drop an
                # operation the Hamiltonian obeys: the clause says dropped and
                # counts each cause
                descended = (f"; descended from {descent.cell_group} "
                             f"({n_all} of {descent.cell_group.size}{u} dropped: "
                             + ", ".join(f"{n} {_DROPPED_BY_WORDS[cause]}"
                                         for cause, n in counts.items())
                             + f"{d})")
        beyond = ""
        if self.beyond_grey is not None:
            n_b = int(self.beyond_grey.n_operations)
            u, d = unit()
            beyond = (f" -- a larger magnetic group, of type "
                      f"{self.beyond_grey.magnetic_type}, is realized ("
                      + (f"{n_b} antiunitary operation{'' if n_b == 1 else 's'}" if not u
                         else f"{n_b} operation{'' if n_b == 1 else 's'} combined with "
                         f"time reversal,{u},")
                      + f" measured to hold{d}) but is beyond the validated scope")
        moved = (positions is not None and positions.adjusted
                 and not within(positions.max_move or 0.0, 0.0))
        return (f"RealizedGroup({self.group.name} #{self.group.number_str}"
                + (f"; precision {pr.precision:.3g} declared -> tol "
                   f"{pr.tol:.3g} (cut {pr.cut:.3g}, window "
                   f"{pr.window:.3g}), "
                   f"position_tol {pr.position_tol:.3g} "
                   f"(measured max|H| {pr.max_h:.4g})"
                   + (f", symprec raised to {pr.symprec:.3g}" if pr.symprec_raised else "")
                   if pr is not None else "")
                + (("; time reversal measured broken"
                    # the grey group of the unitary operations kept is named only where
                    # it was the candidate: where antiunitary operations of the cell held
                    # with time reversal broken, they are what held, on `beyond_grey`
                    + ("" if self.beyond_grey is not None else
                       f", descended from {tr.grey_name} #{tr.grey_number_str}")
                    + (": its square is -1, so the states carry half-integer spin, and "
                       "spin= declares which orbitals are spin partners" if spinor else ""))
                   if tr.outcome == "measured broken" else "")
                + declared
                + descended
                + (f"; additional commuting matrices: {self.conserved.dim}"
                   if self.conserved.dim else "")
                + (f"; centers moved {positions.max_move:.3g}" if moved else "")
                + (f"; symmetrized (defect {symmetrization.defect:.3g})"
                   if symmetrization is not None else "")
                + (("; on a substrate" if env.kind == "substrate" else f"; {env.kind}")
                   + ("" if env.declared else " (the default)")
                   + compared
                   if env is not None else "")
                + beyond + ")")


def realized_group(model, *, typat=None, environment=None, cross_check=True,
                   decorations=None, nk=6, max_bytes=1e9, position_tol=None,
                   precision=None, spinor=None, spin=None,
                   symprec=DEFAULT_FINDER, params=None,
                   characters=None, site_multiplets=None, max_candidates=2 ** 20):
    r"""The space group a Hamiltonian obeys.

    The record returned also holds the Wyckoff position of each orbital and the
    representation of the group, which labels the bands.

    Parameters
    ----------
    model : pythtb.TBModel
        A crystal periodic in three directions, or a layer periodic in two.
    typat : sequence, optional
        A species label for each orbital.  Orbitals on atoms of different
        species are given different labels.  Left out, sites that hold the
        same number of orbitals are one species.
    characters : sequence or mapping, optional
        The site irrep, function or shell each orbital carries.  Pass a
        sequence with one entry per orbital, ``None`` for an orbital left
        undeclared.  Or pass a mapping to an entry from an orbital, or from
        several orbitals of one site.  An entry is a word: a function
        (``"s"``, ``"px"``, ``"dx2-y2"``), a shell (``"p"``), a site irrep
        (``"A1"``), a double-valued site irrep for a model with spin
        (``"-E1"``), or a sum of them (``"A1 + E"``).  In place of a word, an
        entry can be a mapping from operations of the site group, by index or
        by name, to their characters, or to the matrices they act with on the
        orbitals named, rows and columns in the order the orbitals are named.
        Matrices say in which basis of its site irrep a multiplet is written.
    environment : {"substrate", "free-standing"}, optional
        The environment of a layer.  ``"substrate"``: no operation turns the
        layer over.  ``"free-standing"``: the whole layer group applies.  Left
        out, a layer written in two dimensions is read on a substrate, and a
        slab written in three dimensions is read free-standing.  A crystal
        periodic in three directions takes none.
    cross_check : bool, default True
        Whether a layer is also read in the other environment, and the two
        groups compared.  It has an effect only for a layer.
    decorations : sequence of (position, label), optional
        Points that carry no orbital, added to the cell before its group is
        found, such as atoms whose crystal field the orbital positions do not
        show.  Each is a position, in reduced coordinates, and a species
        label.
    precision : float, optional
        The precision of the model's numbers, a positive fraction: of
        ``max|H|`` for the hoppings, and a length in reduced coordinates for
        the orbital centers.  ``max|H|`` is the largest entry of ``H(k)`` once
        the mean on-site energy is subtracted.  Pass it for a numerical model,
        such as one read from Wannier90.  Left out, the model is read as
        exact: an operation is obeyed where its covariance residual, the
        largest entry of ``U H(k) U^-1 - H(gk)`` over the k measured for the
        best matrix ``U`` found, is at most ``1e-8`` times ``max|H|``.
    position_tol : float, optional
        How far an orbital center may miss the position the group puts it at
        and still be moved onto it, in reduced coordinates.  Pass it, in place
        of ``precision``, for a model whose centers alone are imprecise.
    symprec : float, default 1e-05
        The tolerance of spglib, which finds the group from the positions, as
        a Cartesian length in the model's length units.
    spin : str, mapping or SpinBasis, optional
        Which pairs of orbitals are the spin states of one Wannier function,
        for a model built without pythtb's spin whose orbital index carries
        spin, as one imported from Wannier90 does.  Pass ``"interleaved"``,
        ``"blocked"``, a mapping of the keyword arguments of
        :func:`.spin.spin_basis`, or what that call returns.  Left out, such a
        model is read spinless.
    spinor : bool, optional
        Whether the states carry spin one-half, so that the double-valued
        tables are read.  Left out, it is ``True`` for a model built with spin
        or with ``spin=`` given, and ``False`` otherwise.
    nk : int, default 6
        The number of ``k``-points along each periodic direction of the grid
        on which the operations are measured.
    site_multiplets : list or SiteMultiplets, optional
        Each site's multiplets, the sets of orbitals the operations fixing the
        site mix, stated instead of measured.  Pass one list per
        site of the multiplets it holds, each written as its orbitals, such as
        ``[[(0, 1)], [(2,)]]``.
    max_candidates : int, default 2**20
        The largest number of candidate representations searched, where the
        measurement leaves a choice of representation open.
    max_bytes : float, default 1e9
        The memory budget, in bytes, of the measurement of an operation and of
        the projection of ``H`` onto its symmetric part.
    params : dict, optional
        The value of each free parameter of the model, keyed by name:
        ``{"lam": 0.3}``.

    Returns
    -------
    RealizedGroup or Refusal
        The space group the Hamiltonian obeys, ``r.group``, with the Wyckoff
        orbit of each orbital, ``r.wyckoff_orbits``, and the representation of
        the group on the orbitals, ``r.rep``.  :class:`RealizedGroup` describes
        every field.  An operation of the cell that does not carry what
        ``characters=`` declares is not an operation of the declared crystal,
        and the group returned does not contain it.  Where neither a declaration
        nor the hoppings fix the site irrep of an orbital, the
        lowest-angular-momentum rule assigns it.  Where an operation holds
        within the tolerance and not exactly, where a multiplet holds only
        within it, or where the hoppings couple separate parts or tell
        identical copies apart only below it, ``H`` is projected onto its
        symmetric part, and ``r.model`` is the projected model.

        With ``precision=``, the orbital centers are moved onto the positions
        the group puts them at, and an operation is obeyed within a tolerance
        computed from the declared precision.  Each center's move onto those
        positions is at most the declared precision, in reduced coordinates,
        and so is a shift of every center together.  With
        ``position_tol=``, the centers are moved within it, and an operation
        is obeyed within the tolerance of a model read as exact.  With
        ``params=``, ``r.model`` is the model with the values in place, and
        ``r.numerical.source`` the model as handed in.  With
        ``cross_check=False``, ``r.environment.cross_check`` is ``None``.

        A ``Refusal`` in its place is falsy, and names the check that stopped
        the call.

    Examples
    --------
    Kane-Mele from pythtb's models, with its two sublattices declared atoms of
    different species and orbital 0 declared ``A1``:

    >>> from pythtb.models import kane_mele                        # doctest: +SKIP
    >>> r = realized_group(kane_mele(1.0, -1.0, 0.3, 0.25), typat=[0, 1],
    ...                    characters={0: "A1"})                    # doctest: +SKIP
    >>> print(r.summary())                                          # doctest: +SKIP
    P3m11' #156.50, on a substrate   (.group, .environment.kind)
      its operations, by index and Seitz symbol   (.operations)
      the orbitals, by Wyckoff orbit   (.wyckoff_orbits)
        1c (3m1'): orbital 0, -E1
        1b (3m1'): orbital 1, -E1

    Refusals
    --------
    Every Hamiltonian periodic in two or three directions obeys some space
    group, ``P1`` at the least, and which one it obeys is not refused.  A
    refusal concerns a model outside that scope, how the model is written, or
    what is built once the group is found.  A refusal returned once the group
    is found names it, and carries it in ``context["group"]``.

    ``fewer_than_two_periodic_directions``
        the model is periodic in fewer than two directions: a chain or a
        molecule.
    ``supercell``
        ``H`` obeys a pure translation of its cell: the model is written on a
        supercell of its primitive cell.
    ``positions_not_exactly_symmetric``
        with neither ``precision=`` nor ``position_tol=`` declared, the
        orbital centers miss symmetric positions by enough to break an
        operation, or to keep ``H`` from being made exactly symmetric.
    ``positions_not_nearly_symmetric``
        with ``precision=`` or ``position_tol=`` declared, the orbital centers
        cannot be moved within it onto the positions the group puts them at.
    ``operations_not_consistent`` / ``operations_not_nearly_consistent``
        the operations read off the orbital centers do not close into a group,
        and cannot be made to close from the operations the irreptables
        package tabulates.
    ``wyckoff_orbit``
        an operation of the group carries an orbital to a point where no
        orbital of its species sits.
    ``spin_pair_split``
        the two orbitals of a declared spin pair sit at different centers.
    ``single_valued_spinful``
        with ``spinor=False``, a model built with spin.
    ``rep_double_valued_spinless``
        with ``spinor=True``, a spinless model with no ``spin=`` declared.
    ``unknown_label`` / ``ambiguous_multiplet``
        a word of ``characters=`` that the orbital's site does not name, or a
        declaration that does not say which orbitals hold a site irrep.
    ``irrep_tables``
        the character tables of the irreptables package are not loaded, or
        hold no table for this group.
    ``rep_search_too_large`` / ``detector_too_large`` / ``projection_too_large``
        a search past ``max_candidates``, or a measurement or a projection past
        ``max_bytes``.
    ``cell_group_not_realized``
        an operation combined with time reversal that held before ``H`` was
        projected onto its symmetric part is broken after it.
    ``rep_not_covariant``
        no candidate representation of the operations ``H`` was measured to
        obey carries ``H(k)`` onto ``H(gk)`` within the tolerance, for every
        operation ``g``.
    ``rep_not_unique``
        more than one representation passes every check.
    ``rep_internal_factor`` / ``rep_not_multiplicative`` / ``site_group``
        the representation, or a site group read for a declaration, fails a
        check that holds in exact arithmetic.
    ``symmetrize_defect_exceeds_tol`` / ``symmetrize_range_not_closed``
        the projection of ``H`` onto its symmetric part moves ``H`` by more
        than the tolerance, or leaves weight on hoppings beyond the range the
        group carries a hopping to.

    Raises
    ------
    ValueError
        A malformed argument, before anything is measured.  Among them are
        ``environment=`` for a crystal periodic in three directions,
        ``precision=`` given with ``position_tol=``, and ``spin=`` given with
        ``spinor=False`` or for a model built with spin.  Among them too are a
        name in ``params`` that is not a parameter of the model, a parameter
        given no value, and a value that is not a number.

    Warns
    -----
    MagneticScopeWarning
        Time reversal is measured broken.  The group returned is the unitary
        subgroup of the Hamiltonian's magnetic group.

    See Also
    --------
    hsp_table : the little-group representations a set of bands carries at
        every high-symmetry point, read with the record returned.
    """
    bound = _bloch.bound(model, params)
    got = _realize(bound, typat=typat, environment=environment,
                   cross_check=cross_check, decorations=decorations,
                   nk=nk, max_bytes=max_bytes, position_tol=position_tol,
                   precision=precision, spinor=spinor, spin=spin,
                   symprec=symprec, selection_only=False, context=None,
                   characters=characters, multiplets=site_multiplets,
                   max_candidates=max_candidates)
    # `source` is the model exactly as it was handed in, its parameters still free
    if bound is not model and isinstance(got, RealizedGroup):
        got = replace(got, numerical=replace(got.numerical, source=model))
    return got


def _realized_selection(model, *, context=None, **kw):
    """The group a layer's other environment settles on, with no representation built.

    ``model`` is the layer, and ``kw`` holds the keywords of :func:`_realize`
    other than ``selection_only``.  Among them ``tolerances`` is the
    :class:`.._tolerance.Tolerances` the first environment resolved, so that
    both judge the same Hamiltonian against the same numbers.  ``context`` is
    the first environment's :class:`_MeasurementContext`: the two pose the same
    in-plane actions on ``k``, and the second reuses ``H(k)`` on the grid and
    the solution spaces the first found.  No cross-check is run on it.

    Returns
    -------
    _Compared or Refusal
        The selection, or the refusal met on the way to it.
    """
    return _realize(model, selection_only=True, context=context, **kw)


def _at_the_edge(rep, validated):
    r"""The refusal of a Hamiltonian whose asymmetry is close to the tolerance.

    For an operation, ``rho`` is its covariance in the best candidate
    representation and ``r`` its covariance residual measured alone, both in
    the units of ``H``; ``tol`` is the refusal's threshold.

    Parameters
    ----------
    rep : Refusal
        The ``rep_not_covariant`` refusal of the representation built at the
        declared tolerance to project ``H``.  Its context holds each
        operation's ``rho`` under ``covariance_by_operation``.
    validated : ValidatedGroup
        The measurement of the operations, each one's ``r`` in ``residuals``.

    Returns
    -------
    Refusal or None
        ``rep_not_covariant`` where some operation has ``rho > tol`` and every
        such operation has ``rho - r <= tol``: the two measurements of each
        operation agree within the tolerance.  Its reason gives the largest
        ``r`` and the largest ``rho`` against ``tol`` and asks for
        ``precision=``, and its context carries ``obeyed_alone``, the largest
        ``r``, and no ``"group"``.  ``None`` where an operation has
        ``rho - r > tol``, the package disagreeing with itself
        (:func:`_raise_not_covariant`), or where none has ``rho > tol``.
    """
    tol = float(rep.threshold)
    by = rep.context.get("covariance_by_operation") or {}
    over = {int(i): float(rho) for i, rho in by.items() if float(rho) > tol}
    if not over:
        return None
    held = {int(i): float(validated.residuals.get(int(i), np.inf)) for i in by}
    if any(rho - held[i] > tol for i, rho in over.items()):
        return None
    alone, together = max(held.values()), max(over.values())
    return Refusal(
        "rep_not_covariant",
        f"each operation alone is obeyed within the tolerance {tol:.3g}, the worst to "
        f"{alone:.3g}, and the representation built for the group carries H(k) onto "
        f"H(gk) only to {together:.3g}, all in the units of H. Declare `precision=`, "
        "which sets the tolerance to 10 x precision x max|H|: above these numbers if "
        "the asymmetry is numerical noise, or below them if it is a term of the model "
        "to keep.",
        value=together, threshold=tol,
        context={**{k: v for k, v in rep.context.items() if k != "group"},
                 "obeyed_alone": alone})


def _raise_not_covariant(rep, group, steps, *, where=""):
    r"""Raise InternalError for a representation that misses the covariance check.

    Every operation of ``group`` was measured to hold, by the detector and by
    the tensor structure of its own matrix, so a representation of them that
    misses the check beyond the tolerance is the package disagreeing with
    itself.  ``rep`` is the ``rep_not_covariant`` refusal, and ``steps`` the
    mapping of each descent step that ran, empty where none ran.  ``where``
    names the build the refusal came from, where that is not the
    representation returned.  It is not called for a miss close to the
    tolerance at the projection's provisional build, which
    :func:`_at_the_edge` returns as a refusal; nor at the final build for a
    residual within the declared tolerance, or where the centers account for
    the miss, each of which is returned as a refusal.

    Raises
    ------
    InternalError
        Always, naming the group, the descent and the best candidate's
        covariance against the tolerance.
    """
    descent_clause = ""
    if steps:
        chain = " -> ".join([f"#{steps[0]['from']['number_str']}"]
                            + [f"#{st['to']['number_str']}" for st in steps])
        descent_clause = f", by the descent {chain}"
    raise InternalError(
        f"the group identified is {group.name} #{group.number_str}{descent_clause}, "
        f"every operation of it obeyed, and its representation{where} still does not "
        "carry H(k) onto H(gk) within the tolerance: the best of "
        f"{rep.context.get('n_candidates', '?')} candidates "
        f"reaches {float(rep.value):.4g} against {float(rep.threshold):.3g}. "
        "Report it, with the model and the call.")


def _named_refusal(rep, group, validated=None) -> Refusal:
    r"""``rep``, with the group it was reached on named in its reason and context.

    ``group`` is the group the measurement stands on.  With ``validated``
    given, the measurement has settled, and the reason opens on the group
    identified, every operation of it obeyed.  With ``validated`` left
    ``None``, it opens on the group the cell allows, an upper bound on what
    ``H`` obeys.  ``validated`` is read only for whether it is ``None``.
    ``context["group"]`` carries ``group``, ready for
    :func:`.._build._build_rep` to take by hand.  A refusal whose context
    already carries ``"group"`` already names it in its reason, and is
    returned as it is.
    """
    if "group" in rep.context:
        return rep
    head = (f"the group the cell allows is {group.name} #{group.number_str}, an "
            "upper bound on what this Hamiltonian obeys"
            if validated is None else
            f"the group identified is {group.name} #{group.number_str}, every operation "
            "of it obeyed")
    return replace(rep, reason=f"{head}; {rep.reason}",
                   context=dict(rep.context, group=group))


@dataclass(frozen=True)
class _Selection:
    """The group the measurement settles on, and what was read on the way to it.

    Attributes
    ----------
    model : pythtb.TBModel
        The model as measured, projected onto its symmetric part where the
        projection ran.
    cell : object
        The cell the group was identified on.
    cell_typat : list
        The species of each point of the cell.
    base_typat : list
        The species of the cell every descent builds on.
    group : GroupData
        The group settled on.
    assignment : OrbitalAssignment
        The orbitals' assignment to its Wyckoff orbits.
    validated : ValidatedGroup
        The verdict on each operation of the group.
    descent : tuple of dict
        The mapping of each descent step; empty where none ran.
    parents : tuple of GroupData
        The group each step lowered, in the order of ``descent``.
    held : tuple of frozenset of int
        The operations each step's round measured to hold, by index in the
        group that step lowered.
    positions : dict or None
        If not ``None``, the record of the centers' move.
    symmetrization : Symmetrization or None
        If not ``None``, the projection of ``H`` onto its symmetric part.
    layout : SiteMultiplets
        The multiplets measured on the group settled on.
    time_reversal : dict or None
        If not ``None``, the time-reversal verdict.
    antiunitary : tuple of _AntiunitaryRound
        What each round measured of the operations combined with time
        reversal, carried onto the group settled on where time reversal is
        broken.
    declared : dict or None
        If not ``None``, the declarations as they stand on the group settled
        on, keyed by the orbitals each names, each carrying the operations it
        broke.
    choices : _Verdict or None
        If not ``None``, the last round's reading of the solution spaces.
    reduction : dict or None
        If not ``None``, what the declarations took out of the cell.
    unread : Refusal or None
        If not ``None``, on the cross-check's selection, the refusal of a
        declaration the site groups of this environment name nothing for.
    sectors : object
        The sectors the model's conserved matrices acting within sites split
        the states into.
    """

    model: object
    cell: object
    cell_typat: object
    base_typat: object
    group: object
    assignment: object
    validated: object
    descent: tuple
    parents: tuple
    held: tuple
    positions: object
    symmetrization: object
    layout: object
    time_reversal: object
    antiunitary: tuple
    declared: object
    choices: object
    reduction: object
    unread: object
    sectors: object


@dataclass(frozen=True)
class _Compared:
    """The selection the cross-check compares, made in a layer's other environment.

    Attributes
    ----------
    environment : str or None
        The environment it was made in.
    group : GroupData
        The group settled on.
    validated : ValidatedGroup
        The verdict on each operation of the group.
    assignment : OrbitalAssignment
        The orbitals' assignment to its Wyckoff orbits.
    model : pythtb.TBModel
        The model as measured.
    descent : tuple of dict
        The mapping of each descent step; empty where none ran.
    parents : tuple of GroupData
        The group each step lowered, in the order of ``descent``.
    held : tuple of frozenset of int
        The operations each step's round measured to hold, by index in the
        group that step lowered.
    positions : dict or None
        If not ``None``, the record of the centers' move.
    time_reversal : dict or None
        If not ``None``, the time-reversal verdict, with its operations
        combined with time reversal read.
    unread : Refusal or None
        If not ``None``, the refusal of a declaration the site groups of this
        environment name nothing for.
    """

    environment: object
    group: object
    validated: object
    assignment: object
    model: object
    descent: tuple
    parents: tuple
    held: tuple
    positions: object
    time_reversal: object
    unread: object


@dataclass(frozen=True, eq=False)
class _Resolved:
    """The declarations of a call, resolved before anything is measured.

    Attributes
    ----------
    environment : str or None
        The environment of a layer, the default filled in; ``None`` for a
        model periodic in three directions.
    environment_declared : bool
        Whether the call declared the environment.
    extras : tuple
        The decorations, each a position and a species label.
    labels : list
        The species of each orbital.
    measured : dict
        The keywords every measurement of the call is handed.
    spin : ..spin.SpinBasis or None
        If not ``None``, the declared spinor basis.
    three_d : bool
        Whether the model is periodic in three directions.
    tolerances : .._tolerance.Tolerances
        The tolerances every step judges noise against.
    """

    environment: object
    environment_declared: object
    extras: object
    labels: object
    measured: object
    spin: object
    three_d: object
    tolerances: object


@dataclass(frozen=True, eq=False)
class _Selected:
    """What the measurement settled on, for a crystal or a layer.

    Attributes
    ----------
    asked : tuple of _AntiunitaryRound
        What each round measured of the operations combined with time reversal.
    assignment : ..groups.OrbitalAssignment
        The orbitals' assignment to the group's Wyckoff orbits.
    cell : object
        The cell the group was identified on.
    cell_typat : list
        The species of each point of the cell.
    cross : dict or None
        If not ``None``, the cross-check of a layer's two environments.
        ``None`` for a model periodic in three directions, and with
        ``cross_check=False``.
    group : ..groups.GroupData
        The group the measurement settled on.
    m3 : pythtb.TBModel
        The model as measured.
    positions : dict or None
        If not ``None``, the record of the centers' move.
    presented : dict
        The slab presentation's ``"residual"``, the largest change of
        ``H(k)`` it measured, and its ``"tilted"``; zero and ``None`` for a
        model periodic in three directions.
    sel : _Selection
        The measurement's record.
    steps : tuple of dict
        The mapping of each descent step; empty where none ran.
    symmetrization : ..symmetrization.Symmetrization or None
        If not ``None``, the projection of ``H`` onto its symmetric part.
    time_reversal : dict or None
        If not ``None``, the time-reversal verdict.
    validated : ..groups.ValidatedGroup
        The verdict on each operation of the group.
    """

    asked: object
    assignment: object
    cell: object
    cell_typat: object
    cross: object
    group: object
    m3: object
    positions: object
    presented: object
    sel: object
    steps: object
    symmetrization: object
    time_reversal: object
    validated: object


@dataclass(frozen=True, eq=False)
class _Represented:
    """The representation built on the group, with what is read once beside it.

    Attributes
    ----------
    beyond_grey : dict or None
        If not ``None``, the operations combined with time reversal that hold
        where time reversal is broken.
    rep : SpaceGroupRep
        The representation, its environment stamped on.
    sectors : tuple or None
        If not ``None``, the sectors read where the model entered, and what
        the group's operations do to them.
    time_reversal : dict or None
        If not ``None``, the time-reversal verdict, with its operations
        combined with time reversal read.
    """

    beyond_grey: object
    rep: object
    sectors: object
    time_reversal: object


@dataclass(frozen=True, eq=False)
class _Described:
    """The descent's mapping with its note, and the parts of the decoration.

    Attributes
    ----------
    bare_3d : bool
        Whether the model is periodic in three directions, with no decoration
        and no reduction of the cell by the declarations.
    base_typat : list
        The species of the cell every descent builds on.
    descent : dict or None
        If not ``None``, the descent.
    n_extra : int
        The number of points added to the model's own.
    reduction : dict or None
        If not ``None``, what the declarations took out of the cell.
    shadow_recipe : bool
        Whether the model is a layer read on a substrate, whose cell carries
        the substrate's shadow species.
    """

    bare_3d: object
    base_typat: object
    descent: object
    n_extra: object
    reduction: object
    shadow_recipe: object


def _realize(model, *, typat=None, environment=None, cross_check=True,
             decorations=None, dz=DECORATION_DZ,
             c=SLAB_C, nk=6, tol=None, cut=None, window=None,
             max_bytes=1e9, position_tol=None,
             precision=None, spinor=None, spin=None,
             symprec=DEFAULT_FINDER, check=True, n_check=_bloch.CHECK_K,
             seed=_bloch.CHECK_SEED,
             selection_only=False, context=None, tolerances=None, **build_kw):
    """The record realized_group returns, from keywords it leaves at their defaults.

    The declarations are resolved (:func:`_call_resolved`), the group is
    measured (:func:`_measured_selection`), the representation is built once
    on the group settled on (:func:`_representation_built`), the descent is
    described (:func:`_descent_said`), and the record is made
    (:func:`_realized_record`).

    Parameters
    ----------
    model : pythtb.TBModel
        The model, its parameters bound.
    typat, decorations : sequence or None, default None
        As in :func:`realized_group`.
    environment : {"substrate", "free-standing"} or None, default None
        As in :func:`realized_group`.
    cross_check : bool, default True
        As in :func:`realized_group`.
    nk : int, default 6
        As in :func:`realized_group`.
    max_bytes : float, default 1e9
        As in :func:`realized_group`.
    position_tol, precision : float or None, default None
        As in :func:`realized_group`.
    spinor : bool or None, default None
        As in :func:`realized_group`.
    spin : str, mapping, SpinBasis or None, default None
        As in :func:`realized_group`.
    symprec : float, default DEFAULT_FINDER
        As in :func:`realized_group`.
    dz : float, default DECORATION_DZ
        The depth of the substrate's shadow sites below a layer, in reduced
        coordinates of the non-periodic lattice vector.
    c : float, default SLAB_C
        The least length of a layer's non-periodic lattice vector in the slab
        presentation, in the model's length units.
    tol : float or None, default None
        The covariance an operation may carry, in the units of ``H``.  Declared
        only without ``precision``, which sets it.
    cut, window : float or None, default None
        The detector's relative cut and energy window.  Declared only without
        ``precision``, which sets them.
    check : bool, default True
        Whether a layer's slab presentation is checked to leave ``H(k)``
        unchanged.
    n_check : int, default _bloch.CHECK_K
        The number of random ``k`` that check is made at.
    seed : int, default _bloch.CHECK_SEED
        The seed those ``k`` are drawn with.
    selection_only : bool, default False
        Whether to stop once the group has settled, with no representation
        built.
    context : _MeasurementContext or None, default None
        The context to measure in; ``None`` makes a new one.
    tolerances : .._tolerance.Tolerances or None, default None
        The tolerances a parent call resolved; ``None`` resolves them here
        from ``precision``, ``tol``, ``cut``, ``window``, ``position_tol`` and
        ``symprec``, which are otherwise not read.
    **build_kw
        The keywords of the build, :func:`.._build._build_rep`, read by the
        measurement too where it reads them: ``characters``, ``multiplets``
        (``site_multiplets`` of :func:`realized_group`) and
        ``max_candidates`` among them.

    Returns
    -------
    RealizedGroup, _Compared or Refusal
        The record; with ``selection_only=True``, the selection with no
        representation; or a refusal.

    Raises
    ------
    ValueError
        A malformed argument, before anything is measured.
    TypeError
        A keyword the build does not take.
    """
    res = _call_resolved(model, typat=typat, environment=environment, decorations=decorations, c=c,
                         nk=nk, tol=tol, cut=cut, window=window,
                         max_bytes=max_bytes, position_tol=position_tol, precision=precision,
                         spinor=spinor, spin=spin, symprec=symprec,
                         selection_only=selection_only, context=context, tolerances=tolerances,
                         build_kw=build_kw)
    if not isinstance(res, _Resolved):
        return res
    sld = _measured_selection(model, res, typat=typat, cross_check=cross_check,
                              decorations=decorations, dz=dz, c=c, check=check, n_check=n_check,
                              seed=seed)
    if not isinstance(sld, _Selected):
        return sld
    rpd = _representation_built(res, sld, selection_only=selection_only, build_kw=build_kw)
    if not isinstance(rpd, _Represented):
        return rpd
    dsc = _descent_said(model, res, sld, rpd)
    return _realized_record(model, res, sld, rpd, dsc, dz=dz, c=c, n_check=n_check)


def _call_resolved(model, *, typat, environment, decorations, c, nk, tol, cut, window,
                   max_bytes, position_tol, precision, spinor, spin, symprec,
                   selection_only, context, tolerances, build_kw):
    """The declarations of a call, resolved before anything is measured.

    The arguments are those of :func:`_realize` of the same names: ``model``,
    ``typat``, ``environment``, ``decorations``, ``c``, ``nk``, ``tol``,
    ``cut``, ``window``, ``max_bytes``, ``position_tol``, ``precision``,
    ``spinor``, ``spin``, ``symprec``, ``selection_only``, ``context``,
    ``tolerances``, and ``build_kw``, the keywords of the build as a mapping.
    ``build_kw["max_candidates"]``, where present, is replaced in place by the
    whole number it was read as.

    Returns
    -------
    _Resolved or Refusal
        The record, or the refusal ``spin_pair_split`` or
        ``fewer_than_two_periodic_directions``, naming no group.

    Raises
    ------
    ValueError
        A malformed argument, or two that contradict each other.
    TypeError
        A keyword the build does not take.
    """
    # every keyword not named goes to `_build_rep` by name: one it does not take, a
    # misspelling or a model's parameter given as a keyword of its own, raises here,
    # before the cell's group is identified
    unknown = sorted(set(build_kw) - _BUILD_KEYWORDS)
    if unknown:
        raise TypeError("_realize() got an unexpected keyword argument "
                        f"{unknown[0]!r}")
    if environment not in (None, "substrate", "free-standing"):
        raise ValueError(
            f"realized_group(): environment={environment!r} is neither \"substrate\" nor "
            "\"free-standing\". Pass one of the two, or leave environment= out.")
    if environment is not None and int(model.lattice.dim_k) == 3:
        raise ValueError(
            f"realized_group(): environment={environment!r} is given for a model "
            "periodic in three directions, which takes no environment. Leave "
            "environment= out.")
    if precision is not None:
        try:
            _prec = float(precision)
        except (TypeError, ValueError):
            _prec = float("nan")
        if not (_prec > 0.0 and np.isfinite(_prec)):
            raise ValueError(
                f"realized_group(): precision={precision!r} is not a positive number. "
                "Pass the relative precision of the model's numbers, or leave precision= "
                "out to read the model as exact.")
        twice = [n for n, v in (("tol", tol), ("cut", cut), ("window", window),
                                ("position_tol", position_tol))
                 if v is not None]
        if twice:
            raise ValueError(
                f"realized_group(): precision={precision!r} and {' and '.join(twice)} "
                "declare the same thing twice. Pass precision= alone, or position_tol= "
                "alone for a model whose centers alone are imprecise.")
    if position_tol is not None:
        try:
            _ptol = float(position_tol)
        except (TypeError, ValueError):
            _ptol = float("nan")
        if not (_ptol > 0.0 and np.isfinite(_ptol)):
            raise ValueError(
                f"realized_group(): position_tol={position_tol!r} is not a positive "
                "number. Pass how far an orbital center may be moved, in reduced "
                "coordinates, or leave position_tol= out.")
    # the memory budget, checked before anything is measured: the hopping
    # projection's at its peak, and the detector's ceiling on an exactly
    # degenerate spectrum
    max_bytes = _memory_budget(max_bytes)
    # the ceiling on a search the measurement does not settle, checked the same way
    if "max_candidates" in build_kw:
        build_kw["max_candidates"] = _search_ceiling(build_kw["max_candidates"])
    if spin is not None and spinor is False:
        raise ValueError(
            "realized_group(): spin= declares states of spin one-half, and "
            "spinor=False declares none. Drop one of the two.")
    # The spinor basis, resolved once, before anything is measured: a word or a
    # mapping becomes the record it names, and every later step is handed that
    # record.  Every presentation of a layer keeps the orbitals' order and count,
    # so one record describes them all.  A malformed declaration raises here.
    spin = _resolve_spin(spin, model)
    # The sites, read off the centers alone where a declaration needs them (a
    # spinor basis, stated multiplets, or a key of several orbitals): the orbitals
    # at one center are one site at the tolerance the cell's group is found at,
    # as every later step groups them.
    site_of = sites = site_orbs = lattice = None
    if (spin is not None or build_kw.get("multiplets") is not None
            or _names_joint_keys(build_kw.get("characters"))):
        lattice, centers, _ = _pad_cell(model, c)
        sites, site_orbs, _ = _sites_and_typat(
            centers, None,
            (finder_distance(precision, model, symprec) if tolerances is None
             else tolerances.finder), lattice)
        site_of = np.zeros(int(model.norb), dtype=int)
        for s, orbs in enumerate(site_orbs):
            site_of[list(orbs)] = s
    # The declarations, read for their form before anything is measured: a
    # malformed `characters=` or `site_multiplets=` raises here.  Which words name
    # something at a site is read in the site group the measurement reaches.
    _declaration_grammar(int(model.norb), characters=build_kw.get("characters"),
                         multiplets=build_kw.get("multiplets"), spin=spin,
                         site_of=site_of)
    # The one thing the centers alone say about a spinor basis: the two spin
    # states of one Wannier function share its center.  A pair across two
    # centers is refused here, before anything is identified or measured, and
    # the refusal names no group.
    if spin is not None:
        split = _spin_pair_split(spin, site_orbs, sites, lattice)
        if split is not None:
            return split
    # The model's periodicity, read before anything is measured.  A crystal
    # periodic in three directions, or a layer periodic in two, has a space
    # group.  A model periodic in fewer directions, a molecule or a chain, has a
    # point, rod or frieze group, and is refused whatever the dimension of the
    # cell it is written in.
    per = tuple(int(d) for d in model.periodic_dirs)
    if len(per) < 2:
        dims = {1: "one", 2: "two", 3: "three"}.get(int(model.lattice.dim_r),
                                                    str(int(model.lattice.dim_r)))
        what = (f"no periodic direction in its {dims}-dimensional cell: it is a "
                "molecule, whose symmetry is a point group" if not per else
                f"one periodic direction in its {dims}-dimensional cell, along lattice "
                f"vector {per[0]}: it is a chain or a ribbon, whose symmetry is a rod or "
                "frieze group")
        return Refusal(
            "fewer_than_two_periodic_directions",
            f"this model has {what}.",
            value=float(len(per)), threshold=2.0,
            context={"periodic_dirs": per, "dim_r": int(model.lattice.dim_r)})
    # The tolerances every step judges noise against, resolved once: the
    # centers', in reduced coordinates; the Hamiltonian's, in the units of H,
    # read against `max|H|` measured on the model; the matrices read off H,
    # relative to their own scale; and the symmetry finder's distance, in the
    # model's length units.  A parent call hands its own down, so that the other
    # environment of a layer judges the same Hamiltonian against the same numbers.
    if tolerances is None:
        tolerances = Tolerances.declared(
            model, precision=precision, tol=tol, cut=cut, window=window,
            position_tol=position_tol, symprec=symprec,
            scale_of=lambda: _max_hamiltonian(model))
    # A model periodic in three directions is its own presentation and its own
    # cell: the slab presentation and the environment's cell are for a layer.
    three_d = int(model.lattice.dim_k) == 3
    # The environment, resolved once.  A layer with dim_r = 2 has no out-of-plane
    # coordinate and says nothing of what lies under it, so its default is the
    # environment that is always well defined, the substrate.  A slab with
    # dim_r = 3 has out-of-plane coordinates, which say where the layer stands,
    # so its default is free-standing, and a substrate there is a declaration
    # like any other.
    environment_declared = environment is not None
    if three_d:
        environment = None
    elif environment is None:
        environment = "substrate" if int(model.lattice.dim_r) == 2 else "free-standing"

    # The measurement context, one per call: H(k) on the grid and the solution
    # space of each in-plane action on k.  Every measurement of the call shares
    # it: every round of the loop, the measurement after a projection of the
    # hoppings, and a layer's other environment, which poses the same in-plane
    # actions.  Its key holds the model, so a move of the centers or a
    # projection makes a new key, and the grid and the spaces are rebuilt.
    # `context` is the parent's where this call is the other environment.
    context = _MeasurementContext() if context is None else context

    labels = [0] * int(model.norb) if typat is None else list(typat)
    extras = () if decorations is None else tuple(decorations)
    if spinor is None:
        # a declared spinor basis says the states carry spin-1/2 as surely as
        # `nspin == 2` does, so the tables to ask for are the double-valued ones
        spinor = int(model.nspin) == 2 or spin is not None

    measured = dict(labels=labels, spinor=bool(spinor), nk=nk, tolerances=tolerances,
                    context=context, spin=spin, max_bytes=max_bytes,
                    build_kw=build_kw, returned=not selection_only)
    return _Resolved(environment=environment, environment_declared=environment_declared,
                     extras=extras, labels=labels, measured=measured, spin=spin, three_d=three_d,
                     tolerances=tolerances)


def _measured_selection(model, res, *, typat, cross_check, decorations, dz, c, check, n_check,
                        seed):
    """What the measurement settled on, for a crystal or a layer.

    ``res`` is the record :func:`_call_resolved` returns.  The other arguments
    are those of :func:`_realize` of the same names: ``model``, ``typat``,
    ``cross_check``, ``decorations``, ``dz``, ``c``, ``check``, ``n_check`` and
    ``seed``.  A model periodic in three directions is measured on its cell,
    with its decorations where any are declared (:func:`_realized_core`), its
    centers made exact first where a tolerance for them is declared
    (:func:`_exact_centers`); a layer is measured through
    :func:`_layer_realized`.

    Returns
    -------
    _Selected or Refusal
        The record, or the refusal the measurement returned.

    Raises
    ------
    ValueError
        A ``typat=`` without one entry per orbital, for a model periodic in
        three directions with no decorations.
    """
    environment = res.environment
    extras = res.extras
    labels = res.labels
    measured = res.measured
    three_d = res.three_d
    tolerances = res.tolerances

    if three_d:
        m3 = model
        if not extras:
            if len(labels) != int(model.norb):
                raise ValueError(f"realized_group(): the length of typat= is "
                                 f"{len(labels)}, and the model has {int(model.norb)} "
                                 "orbitals. Pass one entry per orbital.")

            def rebuild(m, _extras):
                """The cell of the model ``m``, which is ``m`` itself, and its species.

                A model periodic in three directions with no decorations is its
                own cell; ``_extras`` is not read.
                """
                return m, list(labels)
        else:
            def rebuild(m, ex):
                """The cell of the model ``m`` with the decorations ``ex``, and its species."""
                return _decorated_cell(m, labels, dz=dz, decorations=ex)
        cell, cell_typat = rebuild(m3, extras)
        # The centers made exact, where a tolerance for them is declared, before
        # the cell's setting is read: the cell is then identified on exact
        # centers, in the model's own frame wherever noise alone kept them from it.
        read = None
        if tolerances.center is not None:
            m3, cell, cell_typat, read = _exact_centers(
                m3, cell, cell_typat, extras=extras, rebuild=rebuild, tolerances=tolerances)
        got = _realized_core(m3, cell, cell_typat, centers_read=read, **measured)
        if isinstance(got, Refusal):
            return got
        sel, cross = got, None
        presented = {"residual": 0.0, "tilted": None}
    else:
        got = _layer_realized(model, environment=environment, cross_check=cross_check,
                              typat=typat, decorations=decorations, extras=extras,
                              dz=dz, c=c, check=check, n_check=n_check, seed=seed,
                              **measured)
        if isinstance(got, Refusal):
            return got
        sel, cross, presented = got

    m3, group, assignment = sel.model, sel.group, sel.assignment
    validated, cell, cell_typat = sel.validated, sel.cell, sel.cell_typat
    steps, positions = sel.descent, sel.positions
    symmetrization = sel.symmetrization
    time_reversal, asked = sel.time_reversal, sel.antiunitary
    return _Selected(asked=asked, assignment=assignment, cell=cell, cell_typat=cell_typat,
                     cross=cross, group=group, m3=m3, positions=positions, presented=presented,
                     sel=sel, steps=steps, symmetrization=symmetrization,
                     time_reversal=time_reversal, validated=validated)


def _representation_built(res, sld, *, selection_only, build_kw):
    """The representation built once, on the group the measurement settled on.

    The build is handed the declared contents as they stand on that group and
    the verdict stage's reading of its solution spaces, so nothing is read
    twice.

    Parameters
    ----------
    res : _Resolved
        The call's declarations, as :func:`_call_resolved` returns them.
    sld : _Selected
        What the measurement settled on, as :func:`_measured_selection`
        returns it.
    selection_only : bool
        Whether to stop with no representation built, as in :func:`_realize`.
    build_kw : dict
        The keywords of the build, as in :func:`_realize`.

    Returns
    -------
    _Represented, _Compared or Refusal
        The representation, with the operations combined with time reversal
        and the sectors read once beside it.  With ``selection_only=True``,
        the selection, its operations combined with time reversal read without
        matrices.  Or the refusal of the build, the group named in it:
        ``positions_not_exactly_symmetric`` where the orbital centers' miss
        accounts for the build's failed covariance check.

    Raises
    ------
    ..refusal.InternalError
        The representation misses the covariance check by more than the
        declared tolerance, ``tolerances.energy``.
    """
    environment = res.environment
    spin = res.spin
    tolerances = res.tolerances
    asked = sld.asked
    assignment = sld.assignment
    cross = sld.cross
    group = sld.group
    m3 = sld.m3
    positions = sld.positions
    sel = sld.sel
    steps = sld.steps
    symmetrization = sld.symmetrization
    time_reversal = sld.time_reversal
    validated = sld.validated

    if selection_only:
        # a comparison, whose representation is never built: its operations
        # combined with time reversal are read without matrices
        time_reversal, _beyond = _beyond_grey(group, None, asked, time_reversal,
                                              tolerances=tolerances)
        return _Compared(environment=environment, group=group, validated=validated,
                         assignment=assignment, model=m3,
                         descent=steps, parents=sel.parents,
                         held=sel.held,
                         positions=positions,
                         time_reversal=time_reversal, unread=sel.unread)

    # The representation is built with the declared contents as they stand on
    # the group settled on, not with the declaration's words: a label read in
    # the cell's group need not name anything in the group returned (`A2` of
    # `3m` is `A` of `3`, and `E` of `3m` is the pair `1E2E`).
    build_now = {**build_kw, "multiplets": sel.layout}
    if sel.declared:
        build_now["characters"] = sel.declared
    # The verdict stage's reading of this group, the scalars the hoppings pin,
    # is handed to the build rather than read again, and the build chooses the
    # labeling once.  `_build_rep` checks that the reading is of the operations
    # it represents, and makes its own where it is not.
    rep = _build_rep(m3, group, assignment, validated=validated, spin=spin,
                     _choices=sel.choices, **build_now)
    if isinstance(rep, Refusal):
        if rep.gate == "rep_not_covariant":
            # centers that miss the group's positions by enough to account for
            # the miss of covariance are what to act on, whatever its size
            syms = {int(s.ind): s for s in group._symmetries}
            built = rep.context.get("indices", validated._indices)
            own = _centers_account_for(m3, group, assignment,
                                       {int(i): syms[int(i)] for i in built},
                                       failure=rep.value, gate=rep.threshold,
                                       check="the covariance of the representation of "
                                             "the group",
                                       declared=tolerances.center is not None)
            if own is not None:
                return own
        if (rep.gate == "rep_not_covariant"
                and not (rep.value is not None
                         and float(rep.value) <= float(tolerances.energy))):
            # the consistency guard, whether or not a descent ran: a miss beyond
            # the declared tolerance raises, and a miss within it, an operation
            # near-symmetric that the exact gate refuses, is returned as a refusal
            _raise_not_covariant(rep, group, steps)
        rep = _named_refusal(rep, group, validated)
        square = (time_reversal or {}).get("square")
        if (square is not None and time_reversal.get("outcome") == "measured broken"
                and not (np.isfinite(square["measured"])
                         and square["measured"] * square["asked"] > 0)):
            # where time reversal's own square broke it, the group identified is
            # unitary and the refusal says so; the operations combined with time
            # reversal are read without matrices, none having been built
            time_reversal, _beyond = _beyond_grey(group, None, asked, time_reversal,
                                                  tolerances=tolerances)
            tables = "single-valued" if square["asked"] > 0 else "double-valued"
            rep = replace(rep, reason=rep.reason + (
                f" Time reversal is broken: its square is {square['measured']:+.6g}, "
                f"where the {tables} tables ask for {square['asked']:+d}, so the group "
                f"identified is the unitary one (context['time_reversal'])."),
                context=dict(rep.context, time_reversal=time_reversal))
        return _quote_position_defect(rep, positions, symmetrization)

    # The environment is stamped on the representation, and on a substrate the
    # free-standing group the cross-check measured and found to agree.
    # `ebr.symmetry_indicator` reads both without measuring again.  A
    # representation built by `_build_rep` alone carries None for both, which
    # the indicator reads as unstated.
    free_standing_group = None
    if environment == "substrate" and cross and cross.get("outcome") == "agree":
        free_standing_group = dict(cross["free_standing"])
    rep = replace(rep, _environment=(None if environment is None else str(environment)),
                  _free_standing_group=free_standing_group)

    # the operations combined with time reversal, read once: where time reversal
    # is broken, each one a round measured is read, with that round's verdict
    # and matrix, against the group returned and the matrices built for it
    time_reversal, beyond_grey = _beyond_grey(group, rep.matrices, asked, time_reversal,
                                              tolerances=tolerances)

    # the sectors as the measurement read them where the model entered, and
    # what the operations of the group returned do to them
    sectors = sel.sectors
    if sectors is not None:
        sectors = (sectors, sector_action(sectors, m3, group, assignment, spin, validated,
                                          tol=tolerances.matrix or None))
    return _Represented(beyond_grey=beyond_grey, rep=rep, sectors=sectors,
                        time_reversal=time_reversal)


def _descent_said(model, res, sld, rpd):
    """The descent's mapping with its note, and the parts of the decoration.

    Parameters
    ----------
    model : pythtb.TBModel
        The model, its parameters bound, as in :func:`_realize`.  ``max|H|``
        is read off it where the tolerances carry no scale.
    res : _Resolved
        The call's declarations, as :func:`_call_resolved` returns them.
    sld : _Selected
        What the measurement settled on, as :func:`_measured_selection`
        returns it.
    rpd : _Represented
        The representation, as :func:`_representation_built` returns it; its
        time-reversal verdict and ``beyond_grey`` are read.

    Returns
    -------
    _Described
        The descent, ``None`` where no step ran, its ``note`` saying what the
        cell allowed, what broke it and the group arrived at; and the parts of
        the decoration :func:`_realized_record` writes.
    """
    environment = res.environment
    extras = res.extras
    labels = res.labels
    three_d = res.three_d
    tolerances = res.tolerances
    group = sld.group
    sel = sld.sel
    steps = sld.steps
    beyond_grey = rpd.beyond_grey
    time_reversal = rpd.time_reversal

    # the declared decoration, with the orbit step 1 added where the declared
    # content reduced the cell; the descent's own species are on `descent.cell`
    reduction = sel.reduction
    bare_3d = three_d and not extras and reduction is None
    base_typat = sel.base_typat
    n_extra = len(base_typat) - len(labels)
    # and the environment's own base under it: a substrate builds the shadow
    # species, `dz` below each site, whether or not decorations are put on top,
    # so whether that recipe ran is the environment's question
    shadow_recipe = not three_d and environment == "substrate"
    descent = None
    if steps:
        lattice_step = next((s["lattice"] for s in steps if "lattice" in s), None)
        # what the whole descent dropped, every step's together and by cause,
        # and what the operations kept carry: the note's first sentence, its
        # counts against the group the cell allowed adding up to the group
        # arrived at
        counts = _descent_counts(steps, time_reversal)
        split, lowered = _descent_clauses(counts)
        # the states' class, said where it lowered the group: the pair that
        # names it and, where subgroups of the kept order tied, the rule; on a
        # layer, the free-standing layer it was read on, and the sign the
        # horizontal mirror was given there
        projective = ""
        for st in steps:
            rec = st.get("projective")
            if not rec or not rec.get("pairs"):
                continue
            first = rec["pairs"][0]
            a, b = first["names"]
            phase = complex(first["commutation_phase"])
            tied = (f", the first of {len(rec['tied'])} of that order by the tie rule "
                    f"`realized_group` states -- {rec.get('rule')} -- since "
                    f"{rec.get('reason')}" if rec.get("tied") else "")
            projective += (
                f" On the states of this spinless reading the matrices of {a} and {b}, "
                "two operations that commute, "
                + ("anticommute" if abs(phase + 1.0) <= _CLASS_MARGIN else
                   f"commute only up to the phase {phase.real:+.3g}{phase.imag:+.3g}j")
                + (" -- the sign their spin-1/2 lifts carry, stored in the orbital index"
                   if first["spin_half"] else "")
                + f", and {len(rec['pairs'])} such pairs were measured in "
                f"#{st['from']['number_str']}: no single-valued representation of a "
                "group holding one exists on these states")
            free, mirror = rec.get("read_on"), rec.get("horizontal_mirror")
            if free:
                projective += (
                    ". The class is read on the free-standing layer, "
                    f"{free['name']} #{free['number_str']}, this group with the horizontal "
                    "mirror adjoined, where each operation composed with the mirror has "
                    "its partner's matrix times the mirror's sign: the largest subgroup "
                    "there on which the states carry an ordinary representation is kept"
                    + tied + ", and the operations of it that keep the layer's side are "
                    f"#{st['to']['number_str']}, the group kept here")
            else:
                projective += (
                    ", so the group kept is the largest subgroup on which the states "
                    f"carry an ordinary representation, #{st['to']['number_str']}" + tied)
            if mirror:
                projective += (
                    ". Each operation composed with the horizontal mirror acts on these "
                    "orbitals and on in-plane k as its partner does, and has its "
                    "partner's matrix times the mirror's sign, "
                    + (f"{int(mirror['sign']):+d}, the parity the declared content states"
                       if mirror.get("determined_by") == "declaration" else
                       "+1, the even one the lowest-angular-momentum rule assigns")
                    + ", on which the class does not depend")
            projective += ". `spin=` reads the same states in the double group."
        readings = tuple(c for step in steps
                         for c in (step.get("declaration") or {}).get("tied", ()))
        # the ties, said where the tie rule chose: which subgroups tied, which
        # one was kept and why, among the subgroups carrying the centers, the
        # survivors that did not close, and the readings of a declaration
        ties = "".join(
            " " + note[:1].upper() + note[1:]
            for st in steps
            for note in ((st.get("centers") or {}).get("note"),
                         (st.get("tie") or {}).get("note"),
                         (st.get("declaration") or {}).get("note"))
            if note)
        # what holds, where time reversal was measured broken on the group the
        # last descent arrived at: that group with time reversal adjoined is not
        # what the Hamiltonian obeys, and is not named as if it were.  What
        # holds is its unitary half, returned, and the operations combined with
        # time reversal that `beyond_grey` names.
        tr = time_reversal or {}
        unitary_after = (tr.get("outcome") == "measured broken"
                         and tr.get("grey_number_str") == steps[-1]["to"]["number_str"])
        if unitary_after:
            n_au = int((beyond_grey or {}).get("n_operations", 0))
            mtype = (beyond_grey or {}).get("magnetic_type")
            arrived = (
                "so the cell was lowered to "
                + (lowered or "the unitary operations measured to hold")
                + ", and time reversal, measured with them, is broken: what holds is the "
                f"unitary operations of {group.name} #{group.number_str}"
                + ((f" and the {n_au} combined with time reversal that `beyond_grey` names"
                    + (f", together a magnetic group of type {mtype}"
                       if mtype in ("III", "IV") else ""))
                   if n_au else ", and no operation combined with time reversal")
                + f". Only {group.name} #{group.number_str}, the unitary group, is "
                "returned: the labels, EBR rows, indicators and SRSIs below are its own.")
        else:
            last = steps[-1]["to"]
            arrived = (
                "so the cell was lowered to "
                + ("the ones measured to hold and" if lowered is None else f"{lowered}, and")
                + f" re-identified as {last['name']} #{last['number_str']}. "
                "Only that group is returned: the labels, EBR rows, indicators and SRSIs "
                "below are its own.")
        lattice_said = ("" if not lattice_step else
                        (_lattice_note(lattice_step) + " ") if unitary_after else
                        (lattice_step["note"] + " "))
        # a precision that holds them, said where every operation broken here
        # was broken by its own form beyond the covariance gate: the least
        # precision at which the cell's group would hold
        holding = _precision_holding(
            steps, lambda: (tolerances.scale if tolerances.scale is not None
                            else _max_hamiltonian(model)))
        precise = ("" if holding is None else
                   " Each operation broken here carries H(k) onto H(gk) in its own form to "
                   f"{holding[0]:.3g} at most, beyond the tolerance "
                   f"{float(tolerances.energy):.3g} this call holds each operation to: a "
                   f"precision of {holding[1]:.2g} or coarser, declared with `realized_group(..., "
                   "precision=)`, holds every one of them within it, and "
                   f"{steps[0]['from']['name']} #{steps[0]['from']['number_str']} with them.")
        descent = {
            "from": dict(steps[0]["from"]), "to": dict(steps[-1]["to"]),
            "dropped": tuple(b for step in steps for b in step["dropped"]),
            "dropped_by": {cause: int(n) for cause, n in counts.items() if n},
            "tied": readings,
            "cell": dict(steps[-1]["cell"]),
            "iterations": len(steps), "steps": tuple(steps),
            "lattice": lattice_step,
            "note": lattice_said + (
                f"{'The' if lattice_said else 'the'} cell allowed {steps[0]['from']['name']} "
                f"#{steps[0]['from']['number_str']} ({steps[0]['from']['size']} "
                f"{_PER_COSET}"
                + (f"; these are {_DOUBLE_GROUP}" if group.spinor else "")
                + f") and {split}, "
                + arrived
                + " The model handed back is "
                "the original presentation -- the generic orbit that carried the descent "
                "lives on the cell, which is geometry, and never on it.")
            + ((f" Between the two, the declared orbital content lowered the cell itself "
                f"from #{reduction['from']['number_str']} to "
                f"#{reduction['to']['number_str']}: {reduction['n_excluded']} operations "
                "the crystal that was declared does not have, which is not a descent and "
                "is on `decoration.declared`.") if reduction is not None else "")
            + ties
            + projective
            + precise,
        }
    return _Described(bare_3d=bare_3d, base_typat=base_typat, descent=descent, n_extra=n_extra,
                      reduction=reduction, shadow_recipe=shadow_recipe)


def _realized_record(model, res, sld, rpd, dsc, *, dz, c, n_check):
    """The record :func:`realized_group` returns, made from the four stages' records.

    Parameters
    ----------
    model : pythtb.TBModel
        The model, its parameters bound, as in :func:`_realize`; kept as the
        record's source.
    res : _Resolved
        The call's declarations, as :func:`_call_resolved` returns them.
    sld : _Selected
        What the measurement settled on, as :func:`_measured_selection`
        returns it.
    rpd : _Represented
        The representation, as :func:`_representation_built` returns it.
    dsc : _Described
        The descent and the decoration's parts, as :func:`_descent_said`
        returns them.
    dz : float
        The depth of the substrate's shadow sites, as in :func:`_realize`;
        recorded only where the substrate's recipe ran.
    c : float
        The least length of a slab's non-periodic lattice vector, as in
        :func:`_realize`; recorded for a layer.
    n_check : int
        The number of random ``k`` the slab presentation is checked at, as in
        :func:`_realize`; recorded for a layer.

    Returns
    -------
    RealizedGroup
        The record.
    """
    environment = res.environment
    environment_declared = res.environment_declared
    extras = res.extras
    labels = res.labels
    three_d = res.three_d
    tolerances = res.tolerances
    assignment = sld.assignment
    cell = sld.cell
    cell_typat = sld.cell_typat
    cross = sld.cross
    group = sld.group
    m3 = sld.m3
    positions = sld.positions
    presented = sld.presented
    symmetrization = sld.symmetrization
    validated = sld.validated
    beyond_grey = rpd.beyond_grey
    rep = rpd.rep
    sectors = rpd.sectors
    time_reversal = rpd.time_reversal
    bare_3d = dsc.bare_3d
    base_typat = dsc.base_typat
    descent = dsc.descent
    n_extra = dsc.n_extra
    reduction = dsc.reduction
    shadow_recipe = dsc.shadow_recipe

    # what the slab presentation measured, by its own identity check: the
    # layer's H(k) against the slab's, carried back through the move where a
    # tilted non-periodic vector was put on the layer's normal
    tilted = presented["tilted"]
    presentation = Presentation(
        c=None if three_d else float(c), residual=float(presented["residual"]),
        n_check=0 if three_d else int(n_check),
        tilted=None if tilted is None else _record(PresentationTilt, tilted))
    precision = tolerances.returned()
    # the dictionaries the measurement built, each read once into its record
    return RealizedGroup(
        model=m3, _cell=cell, _typat=labels, _cell_typat=list(cell_typat),
        group=group, _assignment=assignment, rep=rep, spin=rep._spin,
        wyckoff_orbits=_wyckoff_orbit_table(rep),
        time_reversal=_time_reversal_record(time_reversal),
        beyond_grey=_beyond_grey_record(beyond_grey),
        descent=_descent_record(descent),
        environment=(None if environment is None else LayerEnvironment(
            kind=str(environment), declared=bool(environment_declared),
            cross_check=_cross_check_record(cross), presentation=presentation)),
        numerical=RealizedGroupNumerical(
            precision=_record(Precision, precision) if precision else None,
            positions=_record(Positions, positions) if positions else None,
            symmetrization=symmetrization, validated=validated, source=model),
        _sectors=sectors,
        # `dz` is recorded where the substrate's recipe ran and is `None` where
        # it did not.  `typat` and `n_atoms` count the whole addition, the
        # substrate's shadow species and the decorations alike, so the recipe
        # names both where both are there.
        decoration=None if bare_3d else _decoration_record(
        {"dz": (float(dz) if shadow_recipe else None),
         "typat": tuple(base_typat[len(labels):]),
         "n_atoms": int(n_extra),
         "declared": reduction,
         "recipe":
         (("one non-orbital shadow species under every site, at that site's "
           "own height minus dz -- the substrate, as geometry, on the point "
           "set as it stands: removes every operation with R[2,2] = -1 "
           "(sigma_h and everything composed with it, 3D inversion included) "
           "and leaves the subgroup that preserves the layer's normal, its fractional "
           "translations and the layer's own plane alone"
           + (f", with {len(extras)} caller-supplied species on top" if extras else ""))
          if shadow_recipe else
          ("none: the layer is free-standing, so the cell is the point set "
           "as given and every plane-preserving operation is left for the "
           "Hamiltonian to select from rather than removed by construction")
          if not extras and not three_d else
          "caller-supplied decoration" if extras else
          "none: the cell is the model's own point set")
         + ("" if reduction is None else _declared_recipe(reduction, group.spinor))}))


def _declared_recipe(reduction, spinor=False) -> str:
    """The clause the decoration's recipe ends with, on the orbit step 1 added.

    ``reduction`` is the cell's reduction by the declared content
    (``_Selection.reduction``), whose groups ``from`` and ``to``, ``orbit``,
    ``species`` and ``n_excluded`` the clause names.  ``spinor`` is whether
    the group is spinful, and the clause then says that each operation stands
    for itself and its negative in the double group.  Returns the clause,
    opening on a semicolon.
    """
    return (
        f"; and one free orbit of {len(reduction['orbit'])} points under species "
        f"{reduction['species']!r}, which carries what the declared content leaves of "
        f"the cell: {reduction['n_excluded']} of "
        f"{reduction['from']['name']} #{reduction['from']['number_str']}'s "
        f"{reduction['from']['size']} {_PER_COSET}"
        + (f", {_DOUBLE_GROUP}," if spinor else "")
        + " are not symmetries of the crystal that "
        f"was declared and are gone from the cell before any hopping is read, so the "
        f"cell was identified as {reduction['to']['name']} "
        f"#{reduction['to']['number_str']} instead. Each excluded operation says why "
        "under `declared.excluded`. This is a statement about what the orbitals are, "
        "not a descent: `descent` carries what the Hamiltonian broke.")


def _layer_realized(model, *, environment, cross_check, typat, decorations, extras,
                    dz, c, check, n_check, seed, labels, spinor, nk,
                    tolerances, context, spin,
                    max_bytes, build_kw, returned):
    r"""A layer's selection, with its cross-check and what its presentation measured.

    Everything a layer has that a crystal periodic in three directions does
    not is here, so that :func:`_realized_core` knows nothing of layers.  The
    layer is presented as a slab (:func:`.._layers._slab_presentation`) unless
    it is one already, and its environment's cell is built
    (:func:`.._layers._decorated_cell`).  The orbital centers are made exact
    before the cell's group is read, where a tolerance for them is declared
    (:func:`_exact_centers`).  On a substrate the declaration is also read in
    the free-standing layer's site groups
    (:func:`.._descent._layer_declared_reading`), and the free-standing cell
    is handed down for the states' projective class.  The cross-check selects
    the same Hamiltonian in the other environment (:func:`_realized_selection`)
    and holds the substrate's group to be the subgroup of the free-standing
    one that preserves the layer's normal
    (:func:`.._layers._environment_record`).

    Parameters
    ----------
    model : pythtb.TBModel
        The layer, its parameters bound.
    environment : {"substrate", "free-standing"}
        The environment, resolved.
    cross_check : bool
        Whether the other environment is selected and compared.
    typat, decorations
        As :func:`realized_group` takes them; handed to the other
        environment's selection.
    extras : tuple
        The decorations, each a position and a species label.
    dz, c, check, n_check, seed
        As in :func:`_realize`.
    labels : list
        The species of each orbital.
    spinor : bool
        Whether the double-valued tables are read.
    nk : int
        The number of grid points along each periodic direction.
    tolerances : .._tolerance.Tolerances
        The tolerances the call resolved, handed to the other environment
        too, so that both judge one Hamiltonian against the same numbers.
    context : _MeasurementContext
        The call's measurement context, shared with the other environment.
    spin : ..spin.SpinBasis or None
        The declared spinor basis.
    max_bytes : float
        The memory budget, as :func:`_realized_core` reads it.
    build_kw : dict
        The keywords of the build.  For a free-standing layer
        ``include_vacuum_ops`` is set in it, unless given.
    returned : bool
        Whether the group is the one returned, rather than the cross-check's
        comparison.

    Returns
    -------
    tuple or Refusal
        ``(selection, cross, presented)``: the :class:`_Selection`; the
        cross-check's mapping, ``None`` with ``cross_check=False`` or where
        ``returned`` is false; and what the presentation measured of itself as
        built, before any center is moved or hopping projected, its
        ``"residual"`` and ``"tilted"``.  The cross-check's ``outcome`` is
        ``"refused"`` where the other environment refused, ``"not compared"``
        where the declaration is in site irreps the other environment's site
        groups name nothing for, and ``"ambiguous"`` where the subgroup
        statement fails for the pair returned and holds for another pair of
        the two environments' candidate groups (:func:`_environment_candidates`);
        otherwise :func:`.._layers._environment_record` sets it.  Or the
        refusal met on the way.

    Raises
    ------
    ..refusal.InternalError
        The subgroup statement holds for no pair of the two environments'
        candidate groups (:func:`.._layers._raise_environment_contradiction`);
        the move of the centers changes the spectrum (:func:`_exact_centers`);
        or a representation built to project ``H`` misses the covariance
        check beyond the declared tolerance (:func:`_realized_core`).
    """
    # The presentation.  A slab whose non-periodic vector lies along the
    # layer's normal, with a vacuum already reaching `c`, is its own
    # presentation; any other layer is presented by `_slab_presentation`, which
    # records here what it measured of itself as built, before any center is
    # moved and any hopping projected.
    presentation = {"residual": 0.0, "tilted": None}
    if _is_presented_as_a_slab(model, c):
        m3 = model
    else:
        m3 = _slab_presentation(model, c=c, check=check, n_check=n_check, seed=seed,
                                record=presentation)

    # The environment's cell.  A free-standing layer's cell is its point set as
    # given, with `include_vacuum_ops` on.  On a substrate it is the same point
    # set with the substrate's shadow species under every site, which makes the
    # substrate's group a subgroup of the free-standing one.  `_decorated_cell`
    # builds either.
    if environment == "free-standing":
        build_kw.setdefault("include_vacuum_ops", True)

    def rebuild(m, ex):
        """The environment's cell on the model ``m`` with the decorations ``ex``."""
        return _decorated_cell(m, labels, dz=dz, environment=environment, decorations=ex)

    cell, cell_typat = rebuild(m3, extras)

    # The centers made exact, where a tolerance for them is declared, before the
    # cell's setting is read, on the environment's own cell: every reading of
    # the layer below, the free-standing one a substrate carries included, is
    # then made on exact centers.
    read = None
    if tolerances.center is not None:
        m3, cell, cell_typat, read = _exact_centers(
            m3, cell, cell_typat, extras=extras, rebuild=rebuild, tolerances=tolerances)
        extras = read.extras

    # What the declared content says the sites are, read in the layer's own
    # site groups.  Two sites declared different content are different sites in
    # every environment, as two sites of different `typat` are: `E'` and `E''`
    # of a `-6m2` site differ, though on a substrate both restrict to one `E` of
    # `3m`.  So the substrate's cell is lowered by what the declaration says in
    # the free-standing layer's site groups, where its words name something
    # there, as well as by what it says in its own.
    wider = None
    if environment == "substrate" and build_kw.get("characters") is not None:
        wider = _layer_declared_reading(m3, labels, extras=extras, spinor=spinor,
                                        spin=spin, symprec=tolerances.finder,
                                        characters=build_kw["characters"])

    # The free-standing cell, on whose group a substrate's projective class is
    # read: there each operation composed with the horizontal mirror has the
    # mirror's chosen sign times its partner's matrix, and the substrate carries
    # the part of that reading that preserves the layer's normal.  Its group is
    # identified only where a class is to be read.
    free_standing = None
    if environment == "substrate":
        fs_cell, fs_typat = _decorated_cell(m3, labels, environment="free-standing",
                                            decorations=extras)
        free_standing = {"cell": fs_cell, "cell_typat": fs_typat, "labels": labels,
                         "symprec": tolerances.finder}

    got = _realized_core(m3, cell, cell_typat, tolerances=tolerances,
                         labels=labels, spinor=spinor,
                         nk=nk,
                         context=context, spin=spin, max_bytes=max_bytes,
                         build_kw=build_kw, returned=returned,
                         declared_wider=wider, free_standing=free_standing,
                         centers_read=read)
    if isinstance(got, Refusal):
        return got
    if not returned or not cross_check:
        return got, None, presentation

    # The cross-check.  The same model is selected again in the other
    # environment, and the substrate group is held to be the subgroup of the
    # free-standing one that preserves the layer's normal, operation for
    # operation and in both cosets.  A refusal there is recorded with its
    # check, and a declaration in site irreps the other environment's site
    # groups name nothing for is not compared.  Where the two groups disagree,
    # every pair of the two environments' candidate groups, the tie rule's
    # alternatives among them, is held to it, and the package raises where no
    # pair agrees.
    group = got.group
    other_environment = ("free-standing" if environment == "substrate"
                         else "substrate")
    other = _realized_selection(
        model, typat=typat, environment=other_environment,
        cross_check=False, context=context,
        decorations=decorations, dz=dz, c=c, nk=nk,
        max_bytes=max_bytes,
        spinor=spinor, spin=spin, check=check,
        n_check=n_check, seed=seed,
        # the tolerances this side resolved: the same Hamiltonian's noise is
        # judged against the same numbers on the other side
        tolerances=tolerances,
        # `characters=` goes with it: a declaration lowers the group the way a
        # hopping does, so a selection made without it is a different crystal
        # and comparing the two would compare two questions.
        **{k: build_kw[k] for k in ("multiplets", "n_battery", "characters")
           if k in build_kw})
    returned_head = {"number_str": str(group.number_str), "name": str(group.name),
                     "size": int(group._size)}
    if isinstance(other, Refusal):
        refusal = {"gate": str(other.gate), "reason": other.reason,
                   "environment": other_environment}
        cross = {
            "free_standing": None if environment == "substrate" else returned_head,
            "substrate": returned_head if environment == "substrate" else None,
            "rows": (), "missing": (), "extra": (), "flipping": (),
            "unmapped": (),
            "agrees": None, "outcome": "refused", "tie": (),
            "refusal": refusal,
            "note": (
                "the selection in the other environment refused, so the two could "
                "not be compared; a refusal there is not a mismatch here, and this "
                "group is returned. The refusal is in `refusal` and is reproducible "
                f"with realized_group(..., environment={other_environment!r})")}
    elif other.unread is not None:
        # The declaration says which environment is meant.  Its words are this
        # environment's site irreps and name nothing in the other's site groups,
        # so the other environment's group, read without them, is the group of
        # another crystal: it is not recorded, and the two are not compared.
        mine = ("the substrate's" if environment == "substrate"
                else "the free-standing layer's")
        theirs = ("the free-standing layer" if environment == "substrate"
                  else "the substrate")
        cross = {
            "free_standing": None if environment == "substrate" else returned_head,
            "substrate": returned_head if environment == "substrate" else None,
            "rows": (), "missing": (), "extra": (), "flipping": (),
            "unmapped": (),
            "agrees": None, "outcome": "not compared", "tie": (),
            "refusal": None,
            "note": (
                f"not compared with {theirs}: the declaration is written in {mine} "
                f"site irreps, and {theirs}'s site groups have none of its words. A "
                "declaration in one environment's site irreps says which environment "
                f"is meant: {theirs} is not read with it, and this group is "
                f"returned. A layer meant {other_environment} is declared in its own "
                "site irreps, which "
                f"realized_group(..., environment={other_environment!r}) lists where "
                "it refuses this declaration")}
    else:
        free = group if environment == "free-standing" else other.group
        sub = other.group if environment == "free-standing" else group
        dropped, ties = _environment_ties(((str(environment), got),
                                           (other_environment, other)))
        cross = dict(_environment_record(free, sub), tie=dropped, refusal=None)
        left = ", ".join(sorted({str(t["name"]) for t in dropped}))
        one = len(ties) == 1
        where = (("the realized group " if one else "the realized groups ")
                 + ", and ".join(ties))
        if cross["agrees"] is False:
            # A tie is a set of candidates: the statement is held to every pair
            # of the two environments' candidate groups.  Where it holds for
            # another pair, the outcome is `ambiguous`, the pair is named, and
            # the group asked for is returned; where it holds for none, the
            # package raises.
            mine, theirs = _environment_candidates(got), _environment_candidates(other)
            free_c, sub_c = ((mine, theirs) if environment == "free-standing"
                             else (theirs, mine))
            pair = _environment_explained(free_c, sub_c)
            if pair is None:
                _raise_environment_contradiction(cross, held=len(free_c) * len(sub_c))
            cross.update(agrees=None, outcome="ambiguous", note=(
                where
                + (", is not unique, and the one returned there is" if one else
                   ", are not unique, and the ones returned there are")
                + f" the first by the tie rule, without {left} -- and the subgroup "
                "statement does not hold for " + ("that one" if one else "those")
                + f"; it holds for {pair[0]} free-standing and {pair[1]} on a substrate, "
                "which the measurement does not tell from the pair returned. The group "
                "asked for is returned, and it is the one the model can answer."))
        elif ties and cross["agrees"]:
            # where the statement holds for the one the rule kept, the two
            # environments agree, and the tie is named beside the agreement
            cross["note"] += (
                ". " + where[:1].upper() + where[1:]
                + (", is the one the tie rule keeps" if one else
                   ", are the ones the tie rule keeps")
                + f", without {left}, and the statement holds for "
                + ("it" if one else "them"))
    return got, cross, presentation


def _environment_ties(records) -> tuple:
    r"""Where each environment's group was the tie rule's choice, and what it left out.

    ``records`` pairs each environment's name with its selection
    (:class:`_Selection` or :class:`_Compared`).  A tie is read off each
    descent step.  Where the survivors of a round do not close into a group,
    the operations that only the candidates not kept contain were dropped
    ``"tie"``.  Where several subgroups of the largest order carry the
    declared site irreps, carry the orbital centers within the declared
    tolerance, or carry an ordinary representation of the states, the
    operations that only the subgroups not kept contain were dropped by the
    declaration, the centers or the projective class.

    Returns
    -------
    tuple
        ``(left, words)``: one mapping per operation the rule left out, with
        its ``environment``, ``ind`` and ``name``; and one clause per tie,
        naming its environment and what tied.  Both are empty where nothing
        tied.
    """
    left, words = [], []
    for where, rec in records:
        for step in (rec.descent or ()):
            broken = tuple(step.get("dropped", ()))
            if any(str(b.get("refuted_by")) == "tie" for b in broken):
                left += [{"environment": where, "ind": int(b["ind"]),
                          "name": str(b["name"])}
                         for b in broken if str(b.get("refuted_by")) == "tie"]
                words.append(f"in the {where} environment, where the measured "
                             "survivors contain more than one subgroup of the "
                             "largest order")
            names = (step.get("declaration") or {}).get("names", ())
            for cause, ranked, said in (
                    ("declaration", (step.get("declaration") or {}).get("tied", ()),
                     "the declared content has more than one reading of the largest "
                     "order, " + " and ".join("{" + ", ".join(r) + "}" for r in names)),
                    ("centers", (step.get("centers") or {}).get("tied", ()),
                     "more than one subgroup of the largest order carries the orbital "
                     "centers within the declared precision"),
                    ("projective class", (step.get("projective") or {}).get("tied", ()),
                     "the states carry an ordinary representation on more than one "
                     "subgroup of the largest order")):
                if len(ranked) < 2:
                    continue
                others = {int(i) for r in ranked[1:] for i in r} - {
                    int(i) for i in ranked[0]}
                left += [{"environment": where, "ind": int(b["ind"]),
                          "name": str(b["name"])}
                         for b in broken if str(b.get("refuted_by")) == cause
                         and int(b["ind"]) in others]
                words.append(f"in the {where} environment, where {said}")
    return tuple(left), tuple(words)


def _environment_candidates(rec) -> tuple:
    r"""The groups one environment leaves: the one returned, and the tie's others.

    ``rec`` is the environment's selection (:class:`_Selection` or
    :class:`_Compared`).  Only a tie at the last descent step offers
    candidates: a tie at an earlier step was followed by a measurement of the
    subgroup kept alone.  Each other subgroup the tie ranked is read in the
    group that step lowered, on the operations kept and those the round
    measured to hold and dropped for nothing but the choice, the tie itself or
    a closure following from it.  Where those do not close, the largest
    subgroups inside them stand in their place (:func:`_resolve_closure`).
    Where the group returned holds time reversal, each candidate holds each of
    its operations' partners combined with time reversal.

    Returns
    -------
    tuple of (str, object)
        The group returned first, labeled by its name and number; then each
        other candidate, labeled by its unitary operations' names, as an
        object with the ``_symmetries``, ``_lattice`` and ``_vacuum_dirs``
        that :func:`.._layers._environment_record` reads.
    """
    from types import SimpleNamespace

    group = rec.group
    out = [(f"{group.name} #{group.number_str}", group)]
    steps, parents = tuple(rec.descent or ()), tuple(rec.parents or ())
    held = tuple(rec.held or ())
    if not steps or len(parents) != len(steps) or len(held) != len(steps):
        return tuple(out)
    step, parent, measured = steps[-1], parents[-1], held[-1]
    kept = {int(i) for i in step["kept"]}
    causes = {}
    for b in step["dropped"]:
        causes.setdefault(int(b["ind"]), set()).add(str(b["refuted_by"]))
    by_ind = {int(s.ind): s for s in parent._symmetries}
    lattice = np.asarray(parent._lattice, dtype=float)
    partner = ({_operation_key(s)[:2]: s for s in parent._symmetries if s.time_reversal}
               if any(bool(s.time_reversal) for s in group._symmetries) else {})
    seen = {frozenset(kept)}
    for cause, ranked in (("tie", (step.get("tie") or {}).get("tied", ())),
                          ("declaration", (step.get("declaration") or {}).get("tied", ())),
                          ("centers", (step.get("centers") or {}).get("tied", ())),
                          ("projective class", (step.get("projective") or {}).get("tied", ()))):
        # what the choice alone left out: the operations the round measured to
        # hold and broke by this tie, or by a closure that follows from the one
        # it kept, and by nothing else
        open_to = kept | {i for i, why in causes.items()
                          if why <= {cause, "closure", "tie"} and i in measured}
        for alternative in tuple(ranked)[1:]:
            inside = open_to & {int(i) for i in alternative}
            if _descent_closure(parent, [by_ind[i] for i in inside]):
                resolved, _dropped, tied = _resolve_closure(parent, sorted(inside))
                subs = tied or (resolved,)
            else:
                subs = (inside,)
            for sub in subs:
                sub = frozenset(int(i) for i in sub)
                if sub in seen:
                    continue
                seen.add(sub)
                unitary = [by_ind[i] for i in sorted(sub)]
                ops = unitary + [partner[_operation_key(s)[:2]] for s in unitary
                                 if _operation_key(s)[:2] in partner]
                label = "{" + ", ".join(_record_name(s, lattice) for s in unitary) + "}"
                out.append((label, SimpleNamespace(
                    _symmetries=tuple(ops), _lattice=parent._lattice,
                    _vacuum_dirs=parent._vacuum_dirs, number_str="", name=label,
                    _size=len(ops))))
    return tuple(out)


def _environment_explained(free, sub):
    r"""The first other pair of candidates that the subgroup statement holds for.

    ``free`` and ``sub`` are the free-standing and the substrate
    environment's candidates (:func:`_environment_candidates`), each with the
    group returned first.  The pairs that keep one side's returned group are
    held first.  Returns ``(free-standing label, substrate label)``, or
    ``None`` where the statement holds for no pair.
    """
    pairs = sorted(((i, j) for i in range(len(free)) for j in range(len(sub)) if i or j),
                   key=lambda ij: (ij[0] > 0 and ij[1] > 0, ij))
    for i, j in pairs:
        if _environment_record(free[i][1], sub[j][1])["agrees"] is True:
            return free[i][0], sub[j][0]
    return None


def _model_sectors(model, group, assignment, spin, validated, *, tolerances, context=None):
    r"""The sectors the conserved matrices acting within sites split the states into.

    They are read off the identity's solution space in ``validated``
    (:func:`.._sectors.read_sectors`) for ``model``, its orbitals assigned to
    the sites of ``group`` by ``assignment``, with the declared spinor basis
    ``spin``, at the tolerance ``tolerances.matrix``.  Where ``context`` has a
    key, what is read is kept on it, keyed by the model's key there, its sites
    as sets of orbitals, the spinor pairs and the tolerance, and read back by
    every later round and by a layer's other environment.

    Returns
    -------
    .._sectors.SectorReading or None
        The sectors; ``None`` where the identity's space is not held, is past
        what the read-off takes, or has no member that acts within sites.
    """

    key = None
    if context is not None and context._key is not None:
        key = (context._key[0],
               tuple(sorted(tuple(sorted(int(o) for o in orbs))
                            for orbs in assignment.site_orbitals)),
               None if spin is None else tuple(tuple(int(o) for o in pair)
                                                for pair in spin.pairs),
               float(tolerances.matrix))
        if key in context.sectors:
            return context.sectors[key]
    got = read_sectors(model, group, assignment, spin, validated,
                       tol=tolerances.matrix or None)
    if key is not None:
        context.sectors[key] = got
    return got


def _realized_core(model, cell, cell_typat, *, labels, spinor, tolerances, nk,
                   context, spin, max_bytes, build_kw,
                   returned, declared_wider=None, free_standing=None,
                   centers_read=None):
    r"""The group a Hamiltonian obeys, measured on a cell by one loop of three steps.

    The group of the cell is identified once (:func:`_group_of`), and lowered
    by what the declared site irreps exclude from it
    (:func:`_declared_cell_reduction`).  That is done before any hopping is
    read, or, where the declaration cannot be read in the cell's site groups,
    on the next group a descent arrives at.  Where a tolerance for the
    centers is declared, they are moved once onto the positions of the
    largest group that carries them (:func:`_position_pass`); where none is,
    centers whose miss alone breaks an operation are refused before anything
    is built (:func:`.._descent._centers_not_exactly_symmetric`).  Each round
    then reads one verdict per operation of the group it is on
    (:func:`_validate_group`, :func:`_operation_verdicts`), and where a
    unitary operation is broken, descends to the group of the operations
    measured to hold, carried by a cell of its own and identified
    (:func:`_descend_once`).  The loop rounds again until no unitary
    operation is broken.  On the group the unitary operations settle on, time
    reversal is measured once (:func:`_time_reversal_measured`), and where it
    is broken the same cell is identified again without it.  ``H`` is
    projected onto its symmetric part (:func:`..symmetrization._symmetrized`)
    where an operation is obeyed within ``tolerances.energy`` and not
    exactly, where multiplets hold only within it, where parts of the crystal
    are coupled only below it, or where identical copies are coupled or told
    apart only below it; every step then runs again on the projected model.
    A layer's own inputs are handed in, and the loop has no branch of its own
    for a layer.

    Parameters
    ----------
    model : pythtb.TBModel
        The model, presented in three dimensions.
    cell : pythtb.TBModel
        The cell the group is identified on: the orbitals, and any species the
        environment and ``decorations=`` add.
    cell_typat : list
        The species of each point of ``cell``.
    labels : list
        The species of each orbital, as ``typat=`` declares them.
    spinor : bool
        Whether the double-valued tables are read.
    tolerances : .._tolerance.Tolerances
        The tolerances every judgment of noise here reads.
    nk : int
        The number of grid points along each periodic direction.
    context : _MeasurementContext
        The measurement context every round shares.
    spin : ..spin.SpinBasis or None
        The declared spinor basis.
    max_bytes : float
        The memory the hopping projection may take at its peak, and the
        ceiling the detector may raise its own budget to on a spectrum with
        exactly degenerate levels.
    build_kw : dict
        The keywords of the build; ``characters``, ``multiplets``,
        ``reference_orbital`` and ``max_candidates`` are read here.
    returned : bool
        Whether the group is the one returned.  On the cross-check's
        selection, where it is false, no warning that time reversal is broken
        is issued, and a declaration its site groups name nothing for is
        recorded in ``unread`` rather than refused.
    declared_wider : dict or None, default None
        For a layer on a substrate, the declaration read in the free-standing
        layer's site groups (:func:`.._descent._layer_declared_reading`),
        whose exclusions step 1 carries onto this cell
        (:func:`.._descent._carried_reduction`).  On the cross-check's
        selection, the group in which words this cell's site groups name
        nothing for are read and restricted from
        (:func:`.._descent._restricted_contents`).
    free_standing : dict or None, default None
        For a layer on a substrate, the free-standing cell of the same point
        set, on whose group the states' projective class is read.
    centers_read : _CentersRead or None, default None
        Where a tolerance for the centers is declared, what
        :func:`_exact_centers` read of them.

    Returns
    -------
    _Selection or Refusal
        The group settled on and what was read on the way to it.  Or the
        refusal met, naming the group it was reached on
        (:func:`_named_refusal`): before the verdicts have kept every
        operation, the group the cell allows, an upper bound; after, the group
        identified.  Where the identification of the cell's group itself
        refuses, the refusal names a group at most by its name and number,
        as ``positions_not_exactly_symmetric`` does.

    Raises
    ------
    ..refusal.InternalError
        A representation built to project ``H`` misses the covariance check
        beyond the declared tolerance (:func:`_raise_not_covariant`).
    """

    group = _group_of(cell, typat=cell_typat, spinor=bool(spinor),
                      symprec=tolerances.finder)
    if isinstance(group, Refusal):
        # no group identified, so none to name; where no tolerance for the
        # centers is declared and their miss accounts for operations that do
        # not close, the refusal is the centers'
        if tolerances.center is None:
            own = _centers_account_for_closure(group)
            if own is not None:
                return own
        return group
    # the group is stamped `realized` here, so that everything downstream
    # carries it: the assignment, the detector, and `rep._group`, which
    # `characters()` echoes
    group = replace(group, _provenance="realized")
    if int(model.nspin) == 2 and not bool(spinor):
        # A cell that holds points other than the orbitals, a layer's or a
        # decorated crystal's, is built without spin, so its identification
        # above did not refuse spinor=False for a spinful model, as a crystal's
        # own cell does.  The same refusal, naming the group the cell allows.
        asked = _group_of(model, spinor=False)
        return _named_refusal(asked, group)

    # The cell every descent builds on: each rebuilds from this cell plus one
    # generic orbit, never from a previous descent's cell, so the last
    # descent's orbit is the only one in the record.  A reduction by the
    # declared content and a move of the centers set it again.
    base_cell, base_typat = cell, list(cell_typat)
    steps: list = []
    # the group each step lowered, and the operations its round measured to hold
    # before anything else left one out, in which a tie's other candidates are
    # read when the two environments of a layer are compared
    parents: list = []
    held: list = []

    def _stamp(g):
        """The group ``g``, stamped "descended" once a step ran and "realized" before."""
        return replace(g, _provenance="descended" if steps else "realized")

    positions = symmetrization = validated = None

    def _refused(refusal, settled=False):
        """``refusal``, the group it was reached on named in it.

        With ``settled`` false, the group named is the group the cell allows,
        an upper bound on what ``H`` obeys; with ``settled`` true, the
        verdicts have kept every operation and it is the group identified
        (:func:`_named_refusal`).  A position defect measured and not removed
        is quoted (:func:`_quote_position_defect`).
        """
        return _quote_position_defect(
            _named_refusal(refusal, group, validated if settled else None),
            positions, symmetrization)

    assignment = _assign_orbitals(model, group, typat=labels)
    if isinstance(assignment, Refusal):
        return _refused(assignment)
    # centers the whole group carried within the declared tolerance were moved
    # before the cell was identified, and that move is the record's
    if centers_read is not None and centers_read.branch == "carried":
        positions = dict(centers_read.positions)

    layout = choices = None
    time_reversal = None
    # the sectors the conserved matrices acting within sites split the states
    # into, read once where the model enters, on the first round's solution
    # spaces, and carried through every later round
    sectors = None
    sectors_read = False
    # what each round measured of the operations combined with time reversal
    # it asks last (`_antiunitary_settled`), kept for the one reading of them
    # once the group settles
    asked: list = []
    contents: tuple = ()
    # the declared words read in a wider site group where the surviving
    # operations' own cannot name them (`_declared_contents_wider`), carried
    # beside `contents` through every re-identification
    wider: tuple = ()
    pending = None
    # the cross-check's selection only: a declaration this environment's site
    # groups name nothing for, read nowhere, which is not compared
    # (`_layer_realized`)
    unread = None
    # ... and, there, the larger group of the same point set the words were
    # written in, which they are read in and restricted from where this
    # environment's site groups name nothing for them
    written_in = None if returned else declared_wider
    reduction = None
    read_in: dict = {}
    reduce_pending = build_kw.get("characters") is not None
    reversal_measured = reversal_head = weighed = projected = False
    centers_measured = False
    while True:
        # Step 1, before any hopping is read: the reductions the declared
        # content makes to the cell itself.  Two sites of one orbit declared
        # different content are different sites, and a declared function whose
        # span its site group does not preserve leaves only the operations that
        # do.  The operations removed are gone from the group, so no verdict
        # below meets them.  Where the declaration cannot be read in this
        # cell's site groups, it is asked again on the next group.
        if reduce_pending:
            got = _declared_cell_reduction(model, group, assignment, spin=spin,
                                           characters=build_kw["characters"])
            if declared_wider is not None:
                # ... and what the same words say in the larger group of this
                # point set, carried onto the cell as it stands before anything
                # is read: once, since both are read on the cell itself
                got = _carried_reduction(group, assignment, got, declared_wider)
                declared_wider = None
            if not isinstance(got, Refusal):
                kept, excluded, where_read = got
                reduce_pending = False
                # where the declared word was read is carried only where the
                # cell moved under it: with nothing excluded, the resolution
                # below reads it in the same place and says so once
                read_in = dict(where_read) if excluded else {}
                if excluded:
                    made = _declared_cell(base_cell, base_typat, group, kept,
                                          spinor=bool(spinor),
                                          include_TR=bool(group._include_TR),
                                          symprec=tolerances.finder)
                    if isinstance(made, Refusal):
                        return _refused(made)
                    lowered, cell, cell_typat, pairs, point, orbit, species = made
                    reduction = {
                        "from": {"number_str": str(group.number_str),
                                 "name": str(group.name), "size": int(group._size)},
                        "to": {"number_str": str(lowered.number_str),
                               "name": str(lowered.name), "size": int(lowered._size)},
                        "excluded": tuple(excluded[i] for i in sorted(excluded)),
                        "n_excluded": len(excluded),
                        "kept": tuple(sorted(int(i) for i in kept)),
                        "generic_point": tuple(float(x) for x in point),
                        "orbit": tuple(tuple(float(x) for x in row) for row in orbit),
                        "species": species, "pairs": pairs}
                    # a cell reduction, not a descent: the group returned is the
                    # declared crystal's own cell, identified once
                    group = replace(lowered, _provenance="realized")
                    base_cell, base_typat = cell, list(cell_typat)
                    # anything already resolved follows the cell, through this
                    # identification's own pairing, exactly as it follows a
                    # descent: a content is resolved once and carried, never
                    # resolved again in a site group it may not belong to
                    contents = _rekey_contents(contents, pairs, {})
                    wider = _rekey_contents(wider, pairs, {})
                    assignment = _assign_orbitals(model, group, typat=labels)
                    if isinstance(assignment, Refusal):
                        return _refused(assignment)
                    continue
        # The centers next, where a tolerance for them is declared
        # (`tolerances.center`, from `precision=` or `position_tol=`) and they
        # were not moved before the cell was identified.  They are moved onto
        # the positions of the largest group of this round's operations that
        # carries them within it, and the group is written in the origin they
        # then satisfy; the operations outside that group are the centers'
        # verdict, read in the verdict stage below.  This runs once, and the
        # cell every later descent builds on is built again on the moved
        # centers.
        centers = None
        if (tolerances.center is not None
                and not (positions or {}).get("adjusted")):
            got = _position_pass(model, group, tolerances, labels, centers_read)
            if isinstance(got, Refusal):
                return _refused(got)
            before = group
            model, group, assignment, positions, centers, extras_now = got
            env_cell, env_typat = centers_read.rebuild(model, extras_now)
            base_cell, base_typat = _cell_carried(
                env_cell, env_typat, base_cell, base_typat,
                len(centers_read.frame.points),
                (np.asarray(group._shiftUC, dtype=float)
                 - np.asarray(before._shiftUC, dtype=float)) * centers_read.frame.scale)
            cell, cell_typat = base_cell, list(base_typat)
            if free_standing is not None:
                # the free-standing cell of the same point set, on the moved centers
                free_standing["cell"], free_standing["cell_typat"] = _decorated_cell(
                    model, free_standing["labels"], environment="free-standing",
                    decorations=extras_now)
                free_standing.pop("group", None)
        elif tolerances.center is None and not centers_measured:
            # where no tolerance for the centers is declared, they are measured
            # all the same, once: a defect that alone breaks operations at the
            # detector's tolerance is refused here, before anything is built,
            # with the number to act on
            centers_measured = True
            refused = _centers_not_exactly_symmetric(model, group, assignment,
                                                     tol=tolerances.energy)
            if refused is not None:
                return _refused(refused)
        got = _validate_group(model, group, nk=nk, tol=tolerances.energy,
                              cut=tolerances.cut, window=tolerances.window,
                              max_bytes=max_bytes, context=context,
                              assignment=assignment, _floor=tolerances.floor)
        if isinstance(got, Refusal):
            # the detector's memory limit: fired before the measurement settles,
            # so what is named is the group the cell allows, an upper bound
            return _refused(got)
        validated = got
        # Each operation's own form, read once, before anything reads a
        # solution space: of every space the detector kept, the members carrying
        # each site's states onto its image's, with the operation's own spin lift
        # on states that carry a spin.  Everything below reads those, and an
        # operation none of whose members has its own form is refuted by it.
        covariance = _grid_covariance(model, group, nk=nk, context=context)
        validated, own_form = _own_form_read(model, group, assignment, spin, validated,
                                             tolerances=tolerances, covariance=covariance)
        # the sectors, read once, on the first round: the identity's solution
        # space is every matrix that commutes with H, the same on every round,
        # and what it says about the states is said once
        if not sectors_read:
            sectors_read = True
            sectors = _model_sectors(model, group, assignment, spin, validated,
                                     tolerances=tolerances, context=context)
        # Time reversal's member of the sign the tables ask for, where its
        # solution space holds one and the detector's own member is not of that
        # sign: the matrix every step below reads, the verdicts and the
        # multiplets' blocks as much as the representation.  Whether time
        # reversal is obeyed is measured once, where the unitary group settles.
        square = _time_reversal_square(group, validated, tolerances=tolerances)
        validated = _time_reversal_member(validated, square)
        # What a multiplet is, measured on this group's own solution spaces
        # before any verdict reads them: the finest partition of every site's
        # orbitals that still admits a block-form intertwiner, which is what
        # the tensor test, the symmetrizing pass and the representation are all
        # written on.  Re-measured after every descent, because a smaller group
        # glues less; what the conserved matrices join is joined on every round,
        # read off the sectors read once
        layout = _site_multiplets(model, group, assignment, spin, validated,
                                  declared=build_kw.get("multiplets"),
                                  tol=tolerances.matrix, sectors=sectors)
        # What the orbitals are, as `characters=` states it.  Resolved once, in the
        # group the loop is on, and carried from there: a declaration states what
        # each site holds and is read on the multiplets measured above, which it
        # never changes, and every re-identification below re-keys the
        # constraint rather than resolving the label again.
        decls = ()
        if build_kw.get("characters") is not None:
            # once resolved, the contents themselves are what is read on every
            # later round: the constraint is carried and restricted, never the
            # label resolved again in a site group it may not belong to
            stated = ({tuple(int(o) for o in c.orbitals): c for c in contents}
                      if contents else build_kw["characters"])
            got = _declared_contents(model, group, assignment, validated, layout,
                                     spin=spin, characters=stated, read_in=read_in)
            if (isinstance(got, Refusal) and getattr(got, "gate", None) == "unknown_label"
                    and not contents and written_in is not None):
                # The cross-check's selection, of words written in the larger
                # group's site irreps, such as `A2''` of a free-standing layer's
                # `-6m2`, which a substrate's `3m` names nothing for: they are
                # read where they were written, on the multiplets measured here,
                # and restricted to this group's site groups, as after any
                # descent; from here on they are carried as contents
                restricted = _restricted_contents(model, written_in, group, layout,
                                                  spin=spin,
                                                  characters=build_kw["characters"])
                if restricted:
                    got = _declared_contents(
                        model, group, assignment, validated, layout, spin=spin,
                        characters={tuple(int(o) for o in c.orbitals): c
                                    for c in restricted},
                        read_in=read_in)
            if isinstance(got, Refusal):
                # A declaration is read in the site tables of the group the loop
                # is on, and a group the Hamiltonian is about to break can have
                # site groups the declaration's words do not belong to: a layer
                # presented as a slab carries sigma_h, whose `-6m2` site group
                # has no `A1`.  So an unreadable declaration is carried rather
                # than refused, and asked again on the smaller group.
                pending = got
                if not contents and not wider:
                    # and read, the first time, where the words do name a
                    # content: the site groups of this group, not only of the
                    # operations it keeps, or their in-plane part.  Carried
                    # beside the words, it is what the smaller group reads if
                    # the loop settles with the words themselves still naming
                    # nothing there.
                    wider = _declared_contents_wider(
                        model, group, assignment, layout, spin=spin,
                        characters=build_kw["characters"], read_in=read_in)
            else:
                pending = None
                decls, layout = got
                contents = _declared_parts_of(decls)
        got = _operation_verdicts(model, group, assignment, validated,
                                  tolerances=tolerances,
                                  nk=nk, context=context, spin=spin,
                                  layout=layout, decls=decls,
                                  reference_orbital=int(
                                      build_kw.get("reference_orbital", 0) or 0),
                                  max_candidates=int(
                                      build_kw.get("max_candidates", 2 ** 20)
                                      or 2 ** 20),
                                  free_standing=free_standing,
                                  centers=(None if centers is None
                                           else centers.refuted),
                                  sectors=sectors, own_form=own_form)
        if isinstance(got, Refusal):
            # fired before the group is settled: what is named is the group the
            # cell allows, an upper bound, which further rounds may lower
            return _refused(got)
        refuted, lattice, declaration, choices, projective = got
        if lattice and lattice["realized"]:
            return _refused(_supercell_refusal(group, validated, lattice))
        # An antiunitary operation refuted here starts no descent: time reversal
        # is its own measurement, made on the group this settles to.  It is
        # recorded whenever a descent does run.
        antiunitary = {int(s.ind) for s in group._symmetries if s.time_reversal}
        if any(i not in antiunitary for i in set(validated.broken) | set(refuted)):
            # a unitary operation broken: the cell is lowered to the operations
            # measured to hold, and only that group is returned
            got = _descend_once(base_cell, base_typat, group, validated,
                                spinor=bool(spinor),
                                include_TR=bool(group._include_TR),
                                symprec=tolerances.finder, step=len(steps) + 1,
                                refuted=refuted, lattice_record=lattice,
                                declaration=declaration, projective=projective,
                                centers=centers)
            if isinstance(got, Refusal):
                return _refused(got)
            # the operations combined with time reversal whose unitary part the
            # descent does not keep: the group it arrives at never asks them, so
            # this round's verdicts are their last, kept for the one reading of
            # them once the group settles
            left = _antiunitary_settled(group, validated, refuted, square,
                                        got[3]["kept"], tolerances=tolerances)
            if left.n:
                asked.append(left)
            parents.append(group)
            held.append(frozenset(int(i) for i in validated._indices))
            group, cell, cell_typat, record = got
            steps.append(record)
            group = _stamp(group)
            # The declared contents follow the cell: the constraint is re-keyed
            # onto the new group's own `.ind` through the step's own pairing, and
            # the operations the smaller group no longer has drop out, which is
            # the restriction of the declared irrep to the smaller site group.
            contents = _rekey_contents(contents, record["pairs"], refuted)
            wider = _rekey_contents(wider, record["pairs"], refuted)
            assignment = _assign_orbitals(model, group, typat=labels)
            if isinstance(assignment, Refusal):
                return _refused(assignment)
            continue

        # The Hamiltonian obeys every unitary operation of the group it holds.
        # Time reversal is measured once, here: its detector verdict, and its
        # square against the sign the tables read here ask for.  Where it is
        # broken, what this round measured of every operation combined with time
        # reversal is kept beside what each descent measured of the ones it left
        # behind, for the one reading of them once the group settles.
        if not reversal_measured:
            reversal_measured = True
            time_reversal, reversal, validated = _time_reversal_measured(
                group, validated, refuted, tolerances=tolerances)
            if reversal is not None:
                asked.append(reversal)
            if time_reversal["outcome"] != "kept":
                if returned:
                    # Said once per call, about the group that is returned: the
                    # cross-check's other environment runs this same branch on
                    # the same Hamiltonian, and the group it reaches is a
                    # comparison and is never returned.
                    warnings.warn(
                        "pythtqc: time reversal is broken by this Hamiltonian, so only the "
                        "unitary (type-I) subgroup of its magnetic symmetry group is returned. "
                        "Antiunitary operations of the cell that hold are named on "
                        "beyond_grey, and those beyond the cell's own are not searched for; "
                        "the labels, EBR rows and indicator that follow are the unitary "
                        "group's own.",
                        MagneticScopeWarning, skip_file_prefixes=(_PACKAGE_DIR,))
                # Re-identify the same cell as the unitary group it is.  A fresh
                # identification, never the grey record with rows removed: the
                # tables the labels come from have to be the returned group's own.
                unitary = _group_of(cell, typat=cell_typat, spinor=bool(spinor),
                                    include_TR=False, symprec=tolerances.finder)
                if isinstance(unitary, Refusal):
                    return _refused(unitary)
                # a fresh identification stamps a fresh `.ind` order, so the
                # declared contents are re-keyed onto it before anything reads
                # them again; and it reads its translations off the cell again,
                # so it is carried onto the origin of the operations it
                # identifies again, the one the centers satisfy
                unitary_ops = [s for s in group._symmetries if not s.time_reversal]
                pairs, lost, extra = _match_operations(
                    unitary_ops, [s for s in unitary._symmetries if not s.time_reversal],
                    tolerances.finder, group._lattice)
                if not lost and not extra:
                    unitary = _shared_origin(unitary, unitary_ops, pairs)
                contents = _rekey_contents(contents, pairs, {})
                wider = _rekey_contents(wider, pairs, {})
                group = _stamp(unitary)
                assignment = _assign_orbitals(model, group, typat=labels)
                if isinstance(assignment, Refusal):
                    return _refused(assignment)
                reversal_head = True
                continue
        if reversal_head:
            reversal_head = False
            time_reversal["number_str"] = str(group.number_str)
            time_reversal["name"] = str(group.name)
            time_reversal["size"] = int(group._size)

        # The symmetrizing pass, run where a measurement holds within the
        # declared tolerance and not exactly: an operation obeyed within it and
        # not to the exact gate; multiplets that hold only within it
        # (`SiteMultiplets._near`); parts of the crystal the verdict read apart
        # that the hoppings couple below it; or identical copies the sectors
        # read at that tolerance, which the hoppings couple or tell apart below
        # it.  Everything after it runs on the symmetrized model, at the exact
        # gates.
        if not weighed:
            weighed = True
            # the exact gate the representation of this model and group will be
            # built with, measured once: the detector's record carries neither
            # the number of states nor max|H|
            exact_gate = _exact_gate(model, group, validated, build_kw)
            classes = _classify_operations(validated, tolerances.energy,
                                           exact_gate=exact_gate)
            # the parts of the crystal the verdict read, where it read several
            # the sectors read at the declared precision also keep apart, and
            # the largest coupling the Hamiltonian holds between two of them:
            # the build gives each part a character of its own, which leaves
            # every covariance as it was only where the parts are exactly apart
            apart = _parts_apart(choices, layout, sectors)
            coupling = 0.0 if apart is None else _parts_coupling(model, apart)
            coupled = apart is not None and coupling > exact_gate
            # and what the sectors read at the declared precision keep apart,
            # identical copies and sectors splitting multiplets that no operation
            # exchanges, with how far the Hamiltonian is from keeping them apart
            # exactly: a representation that mixes the copies is covariant only
            # on exact copies, and one carried by each such sector on its own
            # only where nothing couples them
            held_apart = _held_sets(sectors, layout, model, group, assignment, spin,
                                    validated)
            off = _held_off(model, held_apart)
            alike = held_apart is not None and max(off.values()) > exact_gate
            # the declared contents as they stand on this group, not the words
            # they were written as: the projection builds a representation, and
            # a label read in the cell's group need not name anything in the one
            # the loop has reached
            sym_kw = {**build_kw, "multiplets": layout}
            if contents:
                sym_kw["characters"] = {
                    tuple(int(o) for o in c.orbitals): c for c in contents}
            # Exact or near, read on each operation's own matrix: the matrices
            # the projection is built with, built once up to the labeling, which
            # no covariance sees, and handed to the projection.  The operation's
            # own matrix decides whether H is made symmetric; the detector's
            # residual keeps deciding whether the operation is kept.  Where the
            # build cannot be made, the detector's residual is read, and the
            # projection, if it runs, meets the same refusal.
            provisional, record = _projection_operators(
                model, group, assignment, validated, tol=tolerances.energy,
                build_kw={**sym_kw, "spin": spin, "_labeled": False})
            if provisional:
                classes = _classify_operations(
                    validated, tolerances.energy, exact_gate=exact_gate,
                    anchored=provisional.numerical.residuals)
            if (any(c == "near" for c in classes.values())
                    or layout._near or coupled or alike):
                got = _symmetrized(model, group, assignment, validated,
                                   tol=tolerances.energy, spin=spin,
                                   max_bytes=max_bytes,
                                   _parts=(None if apart is None else
                                           {"orbitals": apart, "coupling": coupling}),
                                   _held=({"sets": held_apart, "off": off} if alike else None),
                                   _centers_declared=tolerances.center is not None,
                                   _operators=(exact_gate, (provisional, record), classes),
                                   **sym_kw)
                if isinstance(got, Refusal):
                    edge = (_at_the_edge(got, validated) if got.gate == "rep_not_covariant"
                            else None)
                    if edge is not None:
                        # every operation obeyed within the tolerance alone, and
                        # the representation built for the group beyond it by no
                        # more than the two measurements can differ: the
                        # asymmetry is close to the tolerance, and a precision
                        # has to say which side of it the Hamiltonian is on
                        return _refused(edge, settled=True)
                    if got.gate == "rep_not_covariant":
                        # The provisional build's covariance gate is the declared
                        # tolerance every operation was kept at, so a miss there
                        # is the consistency guard's case, as the final build's
                        # is, on a free-standing layer as anywhere.
                        _raise_not_covariant(
                            got, group, steps,
                            where=(", built at the declared tolerance to project the "
                                   "Hamiltonian onto its symmetric part,"))
                    # the projection's own verdict, or its provisional build's:
                    # every operation of this group was kept at the declared
                    # tolerance, so it is the group identified that is named
                    return _refused(got, settled=True)
                model, symmetrization = got
                assignment = _assign_orbitals(model, group, typat=labels)
                if isinstance(assignment, Refusal):
                    return _refused(assignment, settled=True)
                # A projection can still leave an operation refuted, the
                # projector being built from the operations that were
                # near-symmetric and not from the broken ones, so the loop runs
                # again with the descent open and the detector at the declared
                # tolerance.  The detection at the default tolerance is on
                # `symmetrization.verification`.
                projected = True
                continue
        if projected and validated.broken:
            # only antiunitary operations can remain: the loop settles when every
            # unitary one is obeyed, and time reversal was measured before the
            # projection ran.
            return _refused(_cell_group_not_realized(group, validated))
        if pending is not None:
            if (wider and not contents
                    and _restricts_to(wider, group, assignment)):
                # The group settled and its site tables still name nothing
                # the words say: a descent went straight past the site group
                # they were written in.  What they were read as in the larger
                # site group the loop was on, restricted to this one by every
                # re-identification since, is the declaration here, as after
                # any other descent, and one more round reads it.
                contents, wider, pending = wider, (), None
                continue
            if returned:
                # the group settled and the declaration still cannot be read in
                # its site tables: now it is a grammar error about this crystal,
                # and the group that settled is named in it
                return _refused(pending, settled=True)
            # On the cross-check's selection, words read in no round, neither in
            # this environment's site groups nor by restriction from the larger
            # group, are the other environment's site irreps, such as a
            # substrate's `3m` words on a free-standing `-6m2`.  They say which
            # environment is meant, so the two are not compared, and the
            # refusal they would meet is recorded.
            if not contents and getattr(pending, "gate", None) == "unknown_label":
                unread = pending
            pending = None
        break

    # The declared contents as they stand on the group that settled, each
    # carrying the operations it broke on the way: `_build_rep` is handed these
    # rather than the declaration as written, since a label read in the cell's
    # group need not exist in the returned one (`A2` of `3m` returned as `A` of
    # `3`).
    declared = None
    if contents:
        broke = {}
        for st in steps:
            for b in st["dropped"]:
                if b.get("refuted_by") == "declaration":
                    for o in b.get("orbitals", ()):
                        broke.setdefault(int(o), []).append(str(b["name"]))
        contents = tuple(
            replace(c, broke=tuple(dict.fromkeys(
                n for o in c.orbitals for n in broke.get(int(o), []))))
            for c in contents)
        declared = {tuple(int(o) for o in c.orbitals): c for c in contents}
    # the operations combined with time reversal each round asked last, carried
    # onto the group that settled: where time reversal is broken they are read
    # once, against it, where its representation is built (`_beyond_grey`)
    if (time_reversal or {}).get("outcome") == "measured broken":
        asked = list(_carried_onto(group, asked, tolerances.finder))
    return _Selection(model=model, cell=cell, cell_typat=cell_typat,
                      base_typat=base_typat, group=group, assignment=assignment,
                      validated=validated, descent=tuple(steps), parents=tuple(parents),
                      held=tuple(held),
                      positions=positions,
                      symmetrization=symmetrization, layout=layout,
                      time_reversal=time_reversal, antiunitary=tuple(asked),
                      declared=declared, choices=choices, reduction=reduction,
                      unread=unread, sectors=sectors)


@dataclass(frozen=True)
class _CentersRead:
    """What the measurement continues from, once the centers are read.

    Attributes
    ----------
    branch : str
        ``"carried"`` where the whole group carried the centers and they were
        moved; ``"named"`` where it did not, and they were not moved;
        ``"as found"`` where they could not be read as a symmetric set.
    frame : _SiteFrame
        The cell's sites as handed in.
    source : pythtb.TBModel
        The model as handed in.
    extras : tuple
        The decorations, moved with the centers where they were moved.
    rebuild : callable
        ``rebuild(model, extras)`` builds the cell a group is identified on.
    positions : dict or None
        If not ``None``, the record of the move of the centers.
    move : float or None
        On the ``"named"`` branch, the largest move the whole group asks of a
        center, in reduced coordinates; ``None`` otherwise.
    """

    branch: str
    frame: object
    source: object
    extras: tuple
    rebuild: object
    positions: dict | None = None
    move: float | None = None


@dataclass(frozen=True)
class _CentersVerdict:
    """Which operations of a group carry the orbital centers within the tolerance.

    Attributes
    ----------
    precision : float
        The bound declared on each center's move, in reduced coordinates.
    move : float
        The largest move the whole group asks of a center, in reduced
        coordinates.
    kept : tuple of int
        The indices of the operations kept, with their partners combined with
        time reversal; every operation where the whole group carries the
        centers.
    refuted : dict
        Each operation the centers break, keyed by its index: its
        ``refuted_by``, ``"centers"``, its ``miss``, the ``precision`` and a
        ``note``.
    tied : tuple
        If nonempty, the subgroups that tie, in the tie rule's order.
    note : str
        If nonempty, what was found, in words.  It is empty where nothing is
        refuted.
    """

    precision: float
    move: float
    kept: tuple
    refuted: dict
    tied: tuple
    note: str


def _exact_centers(model, cell, cell_typat, *, extras, rebuild, tolerances):
    r"""The centers made exact under a declared tolerance, before the group is found.

    The cell's operations are read off the centers as they are by the
    symmetry finder, without time reversal, which moves no center.  Where
    they act on the sites as a group does (:func:`_site_action`), the centers
    are projected onto the nearest set those operations carry exactly, about
    an origin fitted to them (:func:`_symmetric_sites`).  Where that moves no
    center farther than ``p = tolerances.center``, the model is moved there,
    and a shift of every center together within ``p`` that makes every
    translation a rational of the model's own cell (:func:`_rational_offset`)
    is removed with the move.  Where some center would move farther, nothing
    is moved: the cell is built on the projected centers, to identify the
    group the cell allows, and :func:`_position_pass` moves the centers later.

    Parameters
    ----------
    model : pythtb.TBModel
        The model, presented in three dimensions.
    cell : pythtb.TBModel
        The cell built on ``model`` by ``rebuild``.
    cell_typat : list
        The species of each point of ``cell``.
    extras : tuple
        The decorations beside the model, the environment's species and
        ``decorations=``, each a position and a species label; they move with
        the centers.
    rebuild : callable
        ``rebuild(model, extras)`` builds the cell and its species.
    tolerances : .._tolerance.Tolerances
        The tolerances; ``tolerances.center``, the bound on each center's
        move in reduced coordinates, is not ``None``.

    Returns
    -------
    tuple
        ``(model, cell, cell_typat, read)``: the model, its centers moved on
        the ``"carried"`` branch and as handed in otherwise; the cell to
        identify the group on and its species; and the :class:`_CentersRead`
        the measurement continues from.

    Raises
    ------
    ..refusal.InternalError
        The move of the centers changes the spectrum
        (:func:`..symmetrization._with_centers`).
    """
    import spglib


    p = float(tolerances.center)
    extras = tuple(extras or ())
    frame = _cell_sites(cell, cell_typat, tolerances.finder, n_extra=len(extras))
    read = _CentersRead(branch="as found", frame=frame, source=model, extras=extras,
                        rebuild=rebuild, positions=None)
    if not len(frame.moving):                   # pragma: no cover: a model has orbitals
        return model, cell, cell_typat, read
    found = spglib.get_symmetry_dataset(
        (frame.lattice, frame.first, frame.species),
        symprec=float(tolerances.finder))
    if found is None:
        return model, cell, cell_typat, read
    rotations = np.asarray(found.rotations, dtype=float)
    translations = np.asarray(found.translations, dtype=float)
    action = _site_action(frame, rotations, translations)
    if action is None:
        return model, cell, cell_typat, read
    perm, shift = action
    rows = list(range(len(rotations)))
    x, trans, closure = _symmetric_sites(frame, perm, shift, rotations, rows)
    if not closure <= _EXACT_CLOSURE:
        return model, cell, cell_typat, read
    own = frame.own
    dim_r = int(model.lattice.dim_r)
    moves = _moves_of(frame, x)
    whole = float(np.abs(moves[:own]).max(initial=0.0))
    delta = _rational_offset(trans, rotations, rows, model.periodic_dirs)
    in_frame = delta is not None and within(float(np.abs(delta).max(initial=0.0)), p)
    final = x - delta if in_frame else x
    shifted = _moves_of(frame, final)
    raw = np.asarray(model.orb_vecs, dtype=float)
    orb = raw + shifted[:own, :dim_r]
    moved_extras = _moved_extras(extras, shifted, int(cell.norb) - len(extras), dim_r)
    if not within(whole, p):
        # the whole group does not carry them: the group the cell allows is
        # named on the centers it would put them at, and the centers themselves
        # are moved by the measurement, onto the operations that do
        named, named_typat = rebuild(_geometry(model, orb), moved_extras)
        read = replace(read, branch="named", move=whole)
        return model, named, named_typat, read
    if np.array_equal(orb, raw):
        moved, resid, n_check = model, 0.0, 0
    else:
        (moved, resid), n_check = _with_centers(model, orb), _bloch.CHECK_K
    new_cell, new_typat = rebuild(moved, moved_extras)
    final_trans = np.array([final[perm[g, 0]] + shift[g, 0] - rotations[g] @ final[0]
                            for g in rows])
    defect = _frame_defect(frame, perm, rotations, final_trans, rows)
    offset = (0.0 - np.asarray(delta) * frame.scale)[:dim_r] if in_frame else np.zeros(dim_r)
    record = {
        "defect": defect, "position_tol": p, "adjusted": True,
        "defect_before": defect,
        "defect_after": _frame_defect(frame, perm, rotations, final_trans, rows, sites=final),
        "max_move": whole, "offset": tuple(float(c) for c in offset),
        "frame": "model" if in_frame else "fitted",
        "spectrum_residual": float(resid), "n_check": int(n_check),
        "n_moved": int(np.count_nonzero(np.abs(orb - raw).max(axis=1) > 0.0)),
        "note": _moved_note(whole, offset, in_frame, delta, p, resid)}
    read = replace(read, branch="carried", extras=moved_extras, positions=record)
    return moved, new_cell, new_typat, read


def _moved_note(move, offset, in_frame, delta, p, resid,
                where="the operations put them at") -> str:
    """The note on the move of the centers, in words.

    ``move`` is the largest move of a center onto the symmetric positions, in
    reduced coordinates.  ``delta`` is the shift of every center together that
    writes them in the model's own frame, ``None`` where no shift makes every
    translation rational; ``in_frame`` is whether it was within the declared
    bound ``p`` and made; ``offset`` is the shift made, zero where none was.
    ``resid`` is the spectrum's measured change, in the units of ``H``, and
    ``where`` names the positions moved onto.  Returns the note, opening in
    lowercase and ending in a full stop.
    """
    shift = float(np.abs(np.asarray(offset)).max(initial=0.0))
    moved_any = bool(move) or bool(in_frame and shift)
    if not moved_any and in_frame:
        return ("the centers sit exactly where the operations put them, in the model's own "
                "frame: nothing was moved.")
    said = (f"the centers were moved onto the positions {where}, none by more than "
            f"{move:.3g} (reduced), the origin fitted to them" if move else
            "the centers sit where the operations put them, the origin fitted to them")
    if in_frame and shift:
        said += (f", and every center was shifted together by at most {shift:.3g}, within "
                 "the declared precision, which writes them exactly in the model's own "
                 "frame, every operation's translation a rational of the cell")
    elif in_frame:
        said += ", which is the model's own frame"
    elif delta is None:
        said += ("; no shift of all of them together makes every operation's translation a "
                 "rational of the cell, so the model's origin is not one the operations "
                 "single out, and they stay about the origin fitted to them")
    else:
        said += (f"; the shift of all of them together that would write them in the model's "
                 f"own frame, {float(np.abs(delta).max()):.3g}, is beyond the declared "
                 f"precision {p:.3g}, so it is no noise, and they stay about the origin "
                 "fitted to them")
    return said + ((": a k-dependent gauge transformation, so the spectrum is untouched "
                    f"(measured, {float(resid):.3g}).") if moved_any else
                   ": nothing was moved.")


def _centers_carried(group, frame, perm, shift, rotations, ind_of_row,
                     tolerances) -> _CentersVerdict:
    r"""Which operations of a group carry the orbital centers within the tolerance.

    A group carries the centers where projecting them onto the nearest set it
    carries exactly, about an origin fitted to them (:func:`_symmetric_sites`),
    moves none farther than ``p = tolerances.center``, in reduced coordinates.
    Where the whole group does not, the largest subgroup that does is kept,
    the first by the tie rule (:func:`_tie_ranked`) where several of that
    order tie.  Every operation outside it is broken by the centers, with its
    partner combined with time reversal, which moves no center.

    Parameters
    ----------
    group : GroupData
        The group.
    frame : _SiteFrame
        The cell's sites (:func:`_cell_sites`).
    perm, shift : ndarray
        The unitary operations' permutation of the moving sites and their
        lattice shifts (:func:`_site_action`), one row per operation.
    rotations : ndarray, shape (n, 3, 3)
        The unitary operations' rotations, in the rows' order.
    ind_of_row : sequence of int
        Each row's operation, by index in ``group``.
    tolerances : .._tolerance.Tolerances
        The tolerances; ``tolerances.center`` is not ``None``.

    Returns
    -------
    _CentersVerdict
        The verdict.  Each refuted operation's ``note`` says why in its own
        numbers: how far carrying the centers under it alone would move one,
        or, where that is within ``p``, that no group holding it and the
        operations kept carries them within it.
    """
    p = float(tolerances.center)
    syms = group._symmetries
    every = frozenset(int(s.ind) for s in syms)
    row_of = {int(i): r for r, i in enumerate(ind_of_row)}
    own = frame.own

    def move(ops):
        """The largest move of a center the operations ``ops`` ask, the origin fitted."""
        rows = [row_of[int(i)] for i in ops if int(i) in row_of]
        x, _trans, _closure = _symmetric_sites(frame, perm, shift, rotations, rows)
        return float(np.abs(_moves_of(frame, x)[:own]).max(initial=0.0))

    whole = move(every)
    if within(whole, p):
        return _CentersVerdict(precision=p, move=whole, kept=tuple(sorted(every)),
                               refuted={}, tied=(), note="")
    # the primed partner of each unitary operation, through the group's own
    # product table: T * g, with T the pure time reversal, which moves no center
    prod = group._get_product_table(get_diff=False)
    at = {int(s.ind): i for i, s in enumerate(syms)}
    ind_of = [int(s.ind) for s in syms]
    unitary = [int(s.ind) for s in syms if not s.time_reversal]
    t_pos = next((q for q, s in enumerate(syms) if s.time_reversal
                  and np.array_equal(np.rint(np.asarray(s.rotation)).astype(int),
                                     np.eye(3, dtype=int))
                  and np.allclose(np.asarray(s.translation) % 1.0, 0.0, atol=1e-8)), None)
    partner = ({} if t_pos is None else
               {u: ind_of[int(prod[t_pos, at[u]])] for u in unitary})

    def primed(sub):
        return set(sub) | {partner[u] for u in sub if u in partner}

    subs = _subgroups_inside(group, unitary)
    moves = {s: move(s) for s in subs}
    carrying = [s for s in subs if within(moves[s], p)] or [s for s in subs if len(s) == 1]
    top = max(len(s) for s in carrying)
    ranked = _tie_ranked(group, [s for s in carrying if len(s) == top])
    kept = frozenset(ranked[0])
    tied_in = set().union(*(set(c) for c in ranked[1:])) if len(ranked) > 1 else set()

    def cyclic(g):
        """The operations ``g`` generates, read off the product table."""
        out, x = {g}, g
        while True:
            x = ind_of[int(prod[at[g], at[x]])]
            if x in out:
                return out
            out.add(x)

    lattice = np.asarray(group._lattice, dtype=float)

    def name(i):
        return _record_name(syms[at[int(i)]], lattice)

    kept_names = "{" + ", ".join(name(i) for i in sorted(kept)) + "}"
    refuted = {}
    for u in unitary:
        if u in kept:
            continue
        own_move = move(cyclic(u))
        if not within(own_move, p):
            note = (f"the centers miss it by {own_move:.3g} (reduced): carrying them onto one "
                    f"another under it would move a center that far, beyond the declared "
                    f"precision {p:.3g}")
        elif u in tied_in:
            note = (f"it carries the centers within the declared precision {p:.3g} on its "
                    f"own (a move of {own_move:.3g}), and so do the largest groups that hold "
                    f"it, but they tie with the one the tie rule keeps, {kept_names}, which "
                    "does not hold it")
        else:
            note = (f"it carries the centers within the declared precision {p:.3g} on its "
                    f"own (a move of {own_move:.3g}), but no group holding it and the "
                    f"operations kept, {kept_names}, carries them within it")
        rec = {"ind": int(u), "refuted_by": "centers", "miss": own_move, "precision": p,
               "note": note}
        refuted[int(u)] = rec
        if u in partner:
            refuted[int(partner[u])] = dict(rec, ind=int(partner[u]), note=(
                f"the primed copy of #{int(u)} {name(u)}, which the centers are not "
                "carried by: time reversal moves no center, so the two stand or fall "
                "together"))
    note = (f"the orbital centers are not carried by the operations of "
            f"{group.name} #{group.number_str} within the declared precision {p:.3g}: "
            f"moving them onto its positions would move one by {whole:.3g}. The largest "
            f"group of its operations that carries them within it is {kept_names}, "
            + ("which carries them where they are"
               if within(moves[kept], 0.0) else
               f"which moves none by more than {moves[kept]:.3g}")
            + ", and every operation outside it is broken by the centers"
            + (f"; {len(ranked)} such groups tie, and the one kept is the first by the tie "
               f"rule `realized_group` states -- {_tie_rule_words(group)} -- since "
               f"{_tie_reason(group, ranked)}" if len(ranked) > 1 else "")
            + ".")
    return _CentersVerdict(precision=p, move=whole, kept=tuple(sorted(primed(kept))),
                           refuted=refuted, tied=ranked if len(ranked) > 1 else (), note=note)


def _position_pass(model, group, tolerances, labels, read):
    r"""The centers moved onto the largest group of the operations that carries them.

    The group that carries the centers within ``tolerances.center`` is read
    (:func:`_centers_carried`), the whole of ``group`` wherever it does.  The
    centers are projected onto that group's positions
    (:func:`_symmetric_sites`); a shift of all of them together within the
    bound is removed, so that they are written in the model's own frame
    (:func:`_rational_offset`); and the model is moved there.  ``group`` is
    then carried onto the origin the moved centers satisfy
    (:func:`.._descent._shared_origin`), so that every operation the verdicts
    read carries them exactly.

    Parameters
    ----------
    model : pythtb.TBModel
        The model as the round reads it.
    group : GroupData
        The group the cell allows.
    tolerances : .._tolerance.Tolerances
        The tolerances; ``tolerances.center`` is not ``None``.
    labels : list
        The species of each orbital, for the orbitals' assignment.
    read : _CentersRead
        What :func:`_exact_centers` read: the cell's sites and the model as
        handed in, and the decorations that move with the centers.

    Returns
    -------
    tuple or Refusal
        ``(model, group, assignment, record, verdict, extras)``: the moved
        model; ``group`` on the moved centers' origin; the orbitals'
        assignment to it; the record of the move; the centers'
        :class:`_CentersVerdict` where it breaks an operation, ``None``
        otherwise; and the decorations moved with the centers.  Or
        ``positions_not_nearly_symmetric``, where the operations do not
        permute the sites or a center would move farther than the bound, or
        the refusal of the assignment.

    Raises
    ------
    ..refusal.InternalError
        The operations kept do not carry the positions projected onto them,
        or the move changes the spectrum.
    """
    from types import SimpleNamespace


    p = float(tolerances.center)
    frame = read.frame
    source = read.source
    extras = tuple(read.extras)
    syms = group._symmetries
    unitary = [k for k, s in enumerate(syms) if not s.time_reversal]
    rotations = np.array([np.asarray(syms[k].rotation, dtype=float) for k in unitary])
    translations = np.array([np.asarray(syms[k].translation, dtype=float).reshape(-1)
                             for k in unitary])
    ind_of_row = [int(syms[k].ind) for k in unitary]
    action = _site_action(frame, rotations, translations)
    if action is None:
        return Refusal(
            "positions_not_nearly_symmetric",
            f"the operations of {group.name} #{group.number_str} do not carry the orbital "
            "centers onto one another at the symmetry finder's distance. Identify the "
            "group the centers carry with `typat=` or `decorations=`.",
            value=float("nan"), threshold=p, context={"group": group})
    perm, shift = action
    verdict = _centers_carried(group, frame, perm, shift, rotations, ind_of_row, tolerances)
    beyond = bool(verdict.refuted)
    kept = set(int(i) for i in verdict.kept)
    rows = [r for r, i in enumerate(ind_of_row) if i in kept]
    x, trans, closure = _symmetric_sites(frame, perm, shift, rotations, rows)
    if not closure <= _EXACT_CLOSURE:          # pragma: no cover: a group acts as a group
        raise InternalError(
            f"the {len(rows)} operations of {group.name} #{group.number_str} that carry the "
            "orbital centers do not carry the positions projected onto them, missing by "
            f"{closure:.3g}. Report it, with the model and the call.")
    own = frame.own
    dim_r = int(source.lattice.dim_r)
    moves = _moves_of(frame, x)
    worst = float(np.abs(moves[:own]).max(initial=0.0))
    if not within(worst, p):
        return Refusal(
            "positions_not_nearly_symmetric",
            f"an orbital center would move {worst:.3g} (reduced) onto the position the "
            f"largest group of operations of {group.name} #{group.number_str} that carries "
            f"the centers puts it at, more than the tolerance declared for the centers, "
            f"{p:.3g}. "
            "Orbitals read as one site of that group sit farther apart than that. Identify "
            "the group the centers carry with `typat=` or `decorations=`.",
            value=worst, threshold=p, context={"group": group})
    delta = _rational_offset(trans, rotations, rows, source.periodic_dirs)
    in_frame = delta is not None and within(float(np.abs(delta).max(initial=0.0)), p)
    final = x - delta if in_frame else x
    shifted = _moves_of(frame, final)
    raw = np.asarray(model.orb_vecs, dtype=float)
    orb = np.asarray(source.orb_vecs, dtype=float) + shifted[:own, :dim_r]
    if np.array_equal(orb, raw):
        moved, resid, n_check = model, 0.0, 0
    else:
        (moved, resid), n_check = _with_centers(model, orb), _bloch.CHECK_K
    moved_extras = _moved_extras(extras, shifted, len(frame.points) - len(extras), dim_r)
    # the group, carried onto the origin the moved centers are exact about: once
    # in the frame the centers were read in, which the defects below are
    # measured in, and once as every later step reads the moved model against
    # it, in the group's own lattice and centered on the model's own centers
    # (`_pad_cell`), which along a layer's normal need not be that frame
    pairs = tuple((ind_of_row[r], ind_of_row[r]) for r in rows)
    final_trans = {r: final[perm[r, 0]] + shift[r, 0] - rotations[r] @ final[0] for r in rows}
    read_in = _shared_origin(group, [SimpleNamespace(ind=ind_of_row[r],
                                                     translation=final_trans[r])
                                     for r in rows], pairs)
    _lat, at_group, _pad = _pad_cell(moved, group._vacuum_length, lattice=group._lattice)
    move_sites = [int(s) for s in frame.moving]
    first = int(frame.site_orbs[move_sites[0]][0])

    def as_read(r):
        """The translation of row ``r``'s operation on the moved model, in its lattice."""
        image = int(frame.site_orbs[move_sites[int(perm[r, 0])]][0])
        if image >= own:                          # pragma: no cover: sites lead with orbitals
            return final_trans[r]
        t = at_group[image] - rotations[r] @ at_group[first]
        return t + np.rint(final_trans[r] - t)

    on = _shared_origin(group, [SimpleNamespace(ind=ind_of_row[r], translation=as_read(r))
                                for r in rows], pairs)
    asg = _assign_orbitals(moved, on, typat=labels)
    if isinstance(asg, Refusal):
        return asg
    on_trans = np.array([np.asarray(read_in._symmetries[k].translation, dtype=float).reshape(-1)
                         for k in unitary])
    every = list(range(len(unitary)))
    offset = (0.0 - np.asarray(delta) * frame.scale)[:dim_r] if in_frame else np.zeros(dim_r)
    record = {
        "defect": _frame_defect(frame, perm, rotations, on_trans, every),
        "position_tol": p, "adjusted": True,
        "defect_before": _frame_defect(frame, perm, rotations, on_trans, rows),
        "defect_after": _frame_defect(frame, perm, rotations, on_trans, rows, sites=final),
        "max_move": worst, "offset": tuple(float(c) for c in offset),
        "frame": "model" if in_frame else "fitted",
        "spectrum_residual": float(resid), "n_check": int(n_check),
        "n_moved": int(np.count_nonzero(np.abs(orb - np.asarray(source.orb_vecs, dtype=float))
                                        .max(axis=1) > 0.0))}
    if beyond:
        said = _moved_note(worst, offset, in_frame, delta, p, resid, where="of that group")
        record["move"] = float(verdict.move)
        record["carried"] = len(verdict.kept)
        record["note"] = verdict.note + " " + said[:1].upper() + said[1:]
    else:
        record["note"] = _moved_note(worst, offset, in_frame, delta, p, resid)
    return moved, on, asg, record, (verdict if beyond else None), moved_extras


def _cell_carried(env_cell, env_typat, old_cell, old_typat, n_env, delta):
    r"""The cell built again on the moved centers, with the declaration's orbit.

    ``env_cell`` and ``env_typat`` are the environment's cell on the moved
    model and its species.  ``old_cell`` and ``old_typat`` are the cell the
    call had reached and its species: its first ``n_env`` points are the
    environment's, and the others, where it has any, are the orbit the
    declaration added (:func:`.._descent._declared_cell`).  That orbit is
    shifted by ``delta``, the change of origin the move made, in reduced
    coordinates of the cell, and so stays exact about the new origin.
    Returns ``(cell, species)``; ``env_cell`` as it is where there is no
    orbit.
    """
    from pythtb import Lattice, TBModel

    if int(old_cell.norb) <= int(n_env):
        return env_cell, list(env_typat)
    dim_r = int(env_cell.lattice.dim_r)
    added = (np.asarray(old_cell.orb_vecs, dtype=float)[int(n_env):]
             + np.asarray(delta, dtype=float).reshape(-1)[:dim_r])
    pos = np.vstack([np.asarray(env_cell.orb_vecs, dtype=float), added])
    out = TBModel(Lattice(np.asarray(env_cell.lat_vecs, dtype=float), pos.tolist(),
                          periodic_dirs=list(env_cell.periodic_dirs)), spinful=False)
    setattr(out, _OWN_ORBITALS, int(getattr(env_cell, _OWN_ORBITALS, env_cell.norb)))
    return out, list(env_typat) + list(old_typat)[int(n_env):]


def _quote_position_defect(refusal, positions, symmetrization) -> Refusal:
    """``refusal``, with a position defect measured and not removed quoted in it.

    Where the centers were not moved (``positions``, the record of their
    move, is ``None``) and the hopping projection ``symmetrization`` measured
    their miss of the group's action above :data:`..symmetrization.EXACT_GATE`
    (its ``_operators["position_defect"]``, in reduced coordinates), the
    reason gains a sentence naming the defect and pointing to ``precision=``,
    and the context gains ``position_defect`` and ``positions``.  Otherwise,
    and for a ``positions_not_exactly_symmetric`` refusal or a ``refusal``
    that is not a :class:`..refusal.Refusal`, ``refusal`` is returned
    unchanged.
    """

    if not isinstance(refusal, Refusal):
        return refusal
    if (positions is None and symmetrization is not None
            and refusal.gate != "positions_not_exactly_symmetric"
            and float(symmetrization._operators.get("position_defect", 0.0))
            > EXACT_GATE):
        # `EXACT_GATE` is the exact pipeline's own rounding scale, not a gate in
        # the units of H: it separates centers that are exact from centers off
        # by a measurable defect, and the message is the only consequence
        defect = float(symmetrization._operators["position_defect"])
        note = (
            f"No precision is declared, and the orbital centers miss the positions the "
            f"group puts them at by {defect:.3g} (reduced). Declare `precision=` to "
            "move the centers onto those positions.")
    else:
        return refusal
    return replace(refusal, reason=f"{refusal.reason} {note}",
                   context=dict(refusal.context, position_defect=defect,
                                positions=positions))


#: Each cause the refusal cell_group_not_realized counts, as its reason says it.
_BROKEN_BY_WORDS = {"no symmetry matrix": "with no symmetry matrix",
                    "covariance": "for a covariance residual above the tolerance",
                    "out of plane": "for turning the layer's plane out of itself"}


def _cell_group_not_realized(group, validated) -> Refusal:
    """The refusal cell_group_not_realized: operations the cell allows and H breaks.

    It is returned where operations combined with time reversal are broken
    after ``H`` was projected onto its symmetric part.  The broken operations
    are counted by cause, the detector's verdicts by their ``refuted_by``.

    Parameters
    ----------
    group : GroupData
        The group the cell allows.
    validated : ValidatedGroup
        The detector's verdict on each operation of ``group``.

    Returns
    -------
    Refusal
        ``cell_group_not_realized``, the group carried in
        ``context["group"]``, with the counts and ``validated`` in its
        context.
    """
    reasons = {}
    for d in validated.discrepancies:
        reasons[d.refuted_by] = reasons.get(d.refuted_by, 0) + 1
    n_broken = len(set(validated.broken))
    n_obeyed = len(set(validated._indices))
    proven = reasons.get("no symmetry matrix", 0)
    counts = [f"{n} {_BROKEN_BY_WORDS.get(why, why)}" for why, n in reasons.items()]
    by = (", ".join(counts[:-1]) + " and " + counts[-1]) if len(counts) > 1 else \
        "".join(counts)
    # a cause that broke every operation broken is said without a count, which
    # would only repeat the total
    if len(reasons) == 1 and sum(reasons.values()) == n_broken:
        why = next(iter(reasons))
        by = ("" if n_broken == 1 else "every one ") + _BROKEN_BY_WORDS.get(why, why)

    def the(n, said, said_of_one):
        """The sentence of the ``n`` broken one way: all of them, that many, or the one."""
        if n == 1:
            return f" The one {said_of_one}"
        return f" Those {said}" if n == n_broken else f" The {n} {said}"
    lattice = ""
    if int(getattr(group, "_lattice_index", 1) or 1) > 1:
        taus = tuple(getattr(group, "_fractional_translations", ()) or ())
        tau = np.round(np.asarray(taus[0], dtype=float), 6).tolist() if taus else []
        lattice = (
            f" Among the broken operations is the pure translation {tau} (reduced), "
            f"under which the positions repeat, with its coset partners: the group the "
            f"cell allows is a lattice supergroup of index "
            f"{int(group._lattice_index)}.")
    return Refusal(
        "cell_group_not_realized",
        f"the Hamiltonian obeys {n_obeyed} of the {group._size} operations the cell "
        f"allows ({group.name} #{group.number_str}), {_PER_COSET}"
        + (f" and {_DOUBLE_GROUP}" if group.spinor else "")
        + f"; {n_broken} {'is' if n_broken == 1 else 'are'} broken, {by}."
        + (the(proven, "with no symmetry matrix are broken: for each, no matrix "
                       "carries H(k) onto H(gk).",
               "with no symmetry matrix is broken: no matrix carries H(k) onto "
               "H(gk).")
           if proven else "")
        + lattice
        + " "
        "Read each operation's measurement and residual in `context['validated']`.",
        value=float(n_broken), threshold=0.0,
        context={"group": group, "validated": validated,
                 "number_str": group.number_str, "size": int(group._size),
                 "n_obeyed": int(n_obeyed),
                 "broken": tuple(validated.broken),
                 "refuted_by": reasons})
