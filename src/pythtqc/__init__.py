"""Crystalline symmetry measured from a PythTB tight-binding model.

From a model's Hamiltonian, pythTQC measures:

1. the space group the Hamiltonian obeys, the Wyckoff position and site irrep
   of each orbital, and the representation of the group on the orbitals:
   ``realized_group``;
2. the little-group representations a set of bands carries at each
   high-symmetry point, whose multiplicities form its symmetry data vector:
   ``hsp_table``;
3. whether that vector is consistent with an atomic limit: ``decompose``;
4. its symmetry-indicator class: ``symmetry_indicator``;
5. the integer-valued stable real-space invariants of a set of bands, from
   its symmetry data vector: ``srsi_of_bands``;
6. every stable real-space invariant of the model, those valued modulo n
   included: ``srsi_of_model``;
7. whether a gapped splitting of the model's bands is
   non-symmetry-indicated topological: ``splitting``.

A first call measures the group of pythtb's Kane-Mele model and the symmetry
data vector of its two lowest bands::

    import pythtqc as tqc
    from pythtb.models import kane_mele

    r = tqc.realized_group(kane_mele(0.0, -1.0, 0.3, 0.0), characters={0: "s"})
    print(r.group.name)
    #  P6mm1'
    print(tqc.hsp_table(r, bands=2).symmetry_data)
    #  {'-GM9': 1, '-K6': 1, '-M5': 1}

A name beginning with ``-`` names a double-valued little-group
representation, in the naming of the irreptables package's tables.

A call that cannot complete returns a ``Refusal`` in place of its record.
Printed, it shows the check that stopped the call and what the check
measured.  A refusal is falsy, and so is the empty symmetry data vector that
``subtract`` returns for two equal vectors.  ``isinstance(out, tqc.Refusal)``
tells a refusal from a returned value.  Where the package disagrees with
itself, it raises ``InternalError``.

The space groups' operations, irreps, elementary band representations and
symmetry indicators are read from the tables of the irreptables package.
``tables_status()`` says whether they loaded.  The documentation is the
repository's ``docs/`` folder, https://github.com/bbradlyn/pythtqc, starting
at ``docs/README.md``.

The modules:

* ``groups``: the space group and its tables;
* ``symmetry``: the model's representation of the group;
* ``sewing``: sewing matrices and the little-group representations of bands;
* ``ebr``: decompositions and symmetry indicators;
* ``srsi``: stable real-space invariants;
* ``symmetrization``: a Hamiltonian symmetric only within its declared
  precision, made exactly symmetric;
* ``spin``: the spin of a spin-orbit model imported as spinless, its two spin
  states in the orbital index;
* ``models``: crystals to measure on;
* ``refusal``: refusals, the errors the package raises, and the warning it
  issues;
* ``utils``: figures for the examples.
"""

from . import (ebr, groups, models, refusal, sewing, spin, srsi, symmetrization,
               symmetry, utils)
from .ebr import (EBRDecomposition, RegularCompletion, Restriction, SymmetryIndicator,
                  decompose, ebr_rows, in_plane_restriction, regular_completion, subtract,
                  symmetry_indicator)
from .groups import (CorepIdentification, GroupData, HighSymmetryPoints, KGrid, LittleGroup,
                     Operation, OperationTable, RealizedGroup,
                     SiteReference, ValidatedGroup,
                     corep_table, ebr_data, high_symmetry_points, k_grid,
                     little_group, operation_index, operations, realized_group,
                     stack_presentation, tables_status,
                     wyckoff_sites_tables)
from .refusal import InternalError, MagneticScopeWarning, Refusal, RefusalError
from .sewing import (BandCharacters, CorepsAt, HSPTable, SewingMatrix, characters,
                     coreps_at, format_coreps, hsp_table, multiplets, sewing_matrix,
                     solve_at)
from .spin import SpinBasis, spin_basis, spin_operator, spinless_presentation
from .srsi import (BandSRSIs, ModelSRSIs, Realizations, SRSITables, Splitting,
                   StablePair, atomic_realizations, nsit_criterion, splitting,
                   srsi_of_bands, srsi_of_model, srsi_tables)
from .symmetrization import Symmetrization
from .symmetry import (AtomicLimitTable, BandRepresentation, SiteMultiplets, SpaceGroupRep,
                       WyckoffOrbit)

__all__ = [
    "ebr", "groups", "models", "refusal", "sewing", "spin", "srsi", "symmetrization",
    "symmetry", "utils",
    "Refusal", "RefusalError", "MagneticScopeWarning",
    "GroupData", "HighSymmetryPoints", "CorepIdentification", "KGrid", "LittleGroup",
    "Operation", "OperationTable",
    "ValidatedGroup", "RealizedGroup",
    "SiteReference", "wyckoff_sites_tables",
    "tables_status",
    "operations", "operation_index",
    "little_group",
    "high_symmetry_points", "corep_table", "ebr_data",
    "k_grid",
    "stack_presentation",
    "realized_group",
    "SpaceGroupRep", "InternalError",
    "AtomicLimitTable", "WyckoffOrbit", "BandRepresentation",
    "SiteMultiplets",
    "SpinBasis", "spin_basis", "spin_operator", "spinless_presentation",
    "SRSITables", "BandSRSIs", "ModelSRSIs", "Splitting", "Realizations", "StablePair",
    "srsi_tables", "srsi_of_bands", "srsi_of_model",
    "splitting", "nsit_criterion", "atomic_realizations",
    "Symmetrization",
    "multiplets",
    "solve_at", "format_coreps",
    "SewingMatrix", "BandCharacters", "CorepsAt", "HSPTable",
    "sewing_matrix", "characters", "coreps_at", "hsp_table",
    "EBRDecomposition", "RegularCompletion", "Restriction",
    "SymmetryIndicator", "decompose", "subtract",
    "regular_completion",
    "in_plane_restriction", "ebr_rows",
    "symmetry_indicator",
]
