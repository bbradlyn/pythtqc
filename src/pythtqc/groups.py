r"""The space group a Hamiltonian obeys, and the tables of that group.

:func:`realized_group` measures the group.  The cell is the orbitals'
positions, each with its species, and any points declared with
``decorations=``.  The call finds the cell's space group and lowers it to the
operations the Hamiltonian obeys.  The record it returns also holds the
Wyckoff position of each orbital and the representation of the group, which
labels the bands.  Every Hamiltonian periodic in two or three directions
obeys some space group, ``P1`` at the least.

A model periodic in two directions is a layer.  It is read as a slab, a
three-dimensional cell with vacuum along the layer's normal, whose group is a
space group.  The group of a layer's cell depends on the environment the
layer is read in, on a substrate or free-standing.

The other calls read the group :func:`realized_group` returns:

* :func:`operations` and :func:`operation_index`: the operations of the group,
  one for each coset of the lattice translations, with their indices, Seitz
  symbols and names, and the index a Seitz symbol names;
* :func:`high_symmetry_points`, :func:`little_group` and :func:`k_grid`: the
  tabulated high-symmetry points, the operations that leave a k-point
  invariant up to a reciprocal lattice vector, and a k-point mesh with the
  high-symmetry points on it;
* :func:`corep_table` and :func:`identify_from_characters`: the character
  table of the little group at a high-symmetry point, and the little-group
  representations that measured characters identify;
* :func:`wyckoff_sites_tables`: each site's Wyckoff position in the
  conventional cell used by the irreptables package;
* :func:`ebr_data`, :func:`indicator_group` and :func:`decompose_ebrs`: the
  group's elementary band representations, its symmetry-indicator group, and
  a symmetry data vector written in its elementary band representations;
* :func:`stack_presentation`: a layer repeated along its normal with no
  hopping between the copies, the crystal on which a layer's little-group
  representations at the high-symmetry points with ``k_z = 1/2`` are
  measured;
* :func:`tables_status`: whether the tables are installed.

Every call but :func:`realized_group`, :func:`stack_presentation` and
:func:`tables_status` takes the group, ``r.group``, where ``r`` is the record
:func:`realized_group` returns.  Each record returned prints what it holds
with ``summary()``.  A call that cannot answer the question it is asked
returns a ``Refusal`` in place of the record.  The refusal names the check
that stopped the call, and is falsy.

The tables are those of the Bilbao Crystallographic Server as the irreptables
package ships them.  They cover the 230 space groups and their grey groups,
the space groups with time reversal adjoined.  For a Hamiltonian that breaks
time reversal, :func:`realized_group` returns the unitary subgroup of its
magnetic group, with a :class:`..refusal.MagneticScopeWarning`.

Every k-point taken or returned is in the reduced coordinates of the model's
own reciprocal lattice, which are generally not the tables'.  A call that
takes a high-symmetry point accepts its tabulated name, or its coordinates as
:func:`high_symmetry_points` returns them.  The characters and labels returned
are in the convention in which a lattice translation ``t`` acts as
``e^{-ik·t}``.  The tables of the Bilbao Crystallographic Server write
``e^{+ik·t}``.  At a point that is not time-reversal invariant, the label the
server prints at ``k`` is the label returned here at ``-k``.
"""

from __future__ import annotations

from ._tolerance import WYCKOFF_REFERENCE_TOL
from ._spacegroup import (
    GroupData, CorepIdentification, KGrid, LittleGroup, OrbitalAssignment, SiteReference, Term,
    tables_status)
from ._seitz import Operation, OperationTable, operation_index, operations
from ._tables import (
    EBRConvention, EBRData, HighSymmetryPoints, corep_table, decompose_ebrs, ebr_data,
    high_symmetry_points, identify_from_characters, indicator_group, k_grid,
    little_group, wyckoff_sites_tables)
from ._layers import DECORATION_DZ, SLAB_C, STACK_C, stack_presentation
from ._measure import Discrepancy, ValidatedGroup
from ._realized import (
    BeyondGrey, BeyondGreyOperation, Conserved, ConservedSector, CrossCheck,
    CrossCheckOperation, CrossCheckRefusal, CrossCheckRow, DeclaredReduction, Decoration,
    ExcludedOperation,
    Descent, DescentCell, DescentStep, DroppedOperation, LayerEnvironment, NamedGroup,
    Positions, Precision, Presentation, PresentationTilt, RealizedGroup,
    RealizedGroupNumerical, StepCenters, StepDeclaration, StepTie, TimeReversal,
    TimeReversalSquare, realized_group,
    DescentLattice, LatticeTranslation, ProjectiveMirror, ProjectivePair,
    ProjectiveReading, RefutedTranslation, StepProjective, TranslationDressing)

__all__ = [
    "GroupData", "LittleGroup",
    "CorepIdentification", "KGrid", "ValidatedGroup", "HighSymmetryPoints",
    "Operation", "OperationTable", "Term", "EBRData", "EBRConvention",
    "tables_status",
    "operations", "operation_index",
    "little_group",
    "high_symmetry_points", "corep_table", "identify_from_characters", "ebr_data",
    "indicator_group", "decompose_ebrs", "k_grid",
    "RealizedGroup", "realized_group", "stack_presentation",
    "RealizedGroupNumerical", "LayerEnvironment", "Descent", "DescentStep", "DescentCell",
    "StepDeclaration", "StepCenters", "StepTie", "StepProjective", "ProjectivePair",
    "ProjectiveMirror", "ProjectiveReading", "DescentLattice", "LatticeTranslation",
    "TranslationDressing", "RefutedTranslation", "DroppedOperation", "NamedGroup",
    "TimeReversal", "TimeReversalSquare", "BeyondGrey", "BeyondGreyOperation", "Conserved",
    "ConservedSector", "CrossCheck", "CrossCheckRow", "CrossCheckOperation",
    "CrossCheckRefusal", "Decoration", "DeclaredReduction", "ExcludedOperation", "Positions",
    "Precision",
    "Presentation", "PresentationTilt", "Discrepancy",
    "SLAB_C", "STACK_C",
    "DECORATION_DZ",
    "SiteReference", "wyckoff_sites_tables",
    "WYCKOFF_REFERENCE_TOL",
]
