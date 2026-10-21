r"""The space group a Hamiltonian obeys, and that group's tables in the model's own cell.

:func:`realized_group` measures the group: the space group of the lattice of
occupied sites, lowered to the operations ``H(k)`` obeys, with each orbital's
Wyckoff position and the representation that labels the bands.  The other
calls read the group it returns:

* :func:`operations`, :func:`seitz` and :func:`operation_index`: each
  operation by its index, the key every per-operation record is written on,
  with its Seitz symbol in the tables' setting and its name in the model's
  Cartesian frame; and the index a Seitz symbol names;
* :func:`high_symmetry_points`, :func:`little_group` and :func:`k_grid`: the
  tabulated k-points, the operations that leave a k invariant, and a mesh with
  the high-symmetry points it holds;
* :func:`corep_table` and :func:`identify_from_characters`: the character table
  of the little group at a high-symmetry point, and the coreps measured
  characters carry;
* :func:`wyckoff_sites_tables` and :func:`wyckoff_positions_spglib`: each
  site's Wyckoff position in the tables' setting, which every table is read
  with, and the group's Wyckoff positions in spglib's;
* :func:`ebr_data`, :func:`indicator_group` and :func:`decompose_ebrs`: the
  group's elementary band representations, its indicator group, and a symmetry
  vector written in its elementary band representations;
* :func:`stack_presentation`: a layer repeated along its normal with no hopping
  between the copies, the three-dimensional crystal its rows at ``k_z = ½`` are
  measured on;
* :func:`tables_status`: whether the tables are installed.

Each record returned prints what it holds with ``summary()``, and a call that
cannot complete returns a falsy ``Refusal`` in its place.

The tables are irrep's, the ``irreptables`` package, for the 230 space groups
and their grey groups, time reversal adjoined; a Hamiltonian that breaks time
reversal is returned the unitary group, with a
:class:`..refusal.MagneticScopeWarning`.  Every k-point taken or returned is in
the model's own reduced coordinates, generally not the tables', so the points
are taken from :func:`high_symmetry_points`.  The characters and labels are in
the physical convention for a lattice translation, ``e^{-ik·t}``, where the
tables' own pages write ``e^{+ik·t}``: away from the time-reversal-invariant
momenta, a label read at ``k`` here is the tables' label at ``-k``.  A model
periodic in two directions is a layer, read as a slab, a three-dimensional cell
with vacuum along the layer's normal, whose group is a space group.
"""

from __future__ import annotations

from ._tolerance import WYCKOFF_REFERENCE_TOL
from ._spacegroup import (
    GroupData, IrrepContent, KGrid, LittleGroup, OrbitalAssignment, SiteReference,
    WyckoffPosition, tables_status, wyckoff_positions_spglib)
from ._seitz import Operation, OperationTable, operation_index, operations, seitz
from ._tables import (
    HighSymmetryPoints, corep_table, decompose_ebrs, ebr_data, high_symmetry_points,
    identify_from_characters, indicator_group, k_grid, little_group,
    wyckoff_sites_tables)
from ._layers import DECORATION_DZ, SLAB_C, STACK_C, stack_presentation
from ._measure import ValidatedGroup
from ._realized import RealizedGroup, realized_group

__all__ = [
    "GroupData", "LittleGroup", "WyckoffPosition", "OrbitalAssignment",
    "IrrepContent", "KGrid", "ValidatedGroup", "HighSymmetryPoints",
    "Operation", "OperationTable",
    "tables_status", "wyckoff_positions_spglib",
    "operations", "seitz", "operation_index",
    "little_group",
    "high_symmetry_points", "corep_table", "identify_from_characters", "ebr_data",
    "indicator_group", "decompose_ebrs", "k_grid",
    "RealizedGroup", "realized_group", "stack_presentation",
    "SLAB_C", "STACK_C",
    "DECORATION_DZ",
    "SiteReference", "wyckoff_sites_tables",
    "WYCKOFF_REFERENCE_TOL",
]
