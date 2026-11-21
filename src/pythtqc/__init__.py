"""Crystalline symmetry measured from a PythTB tight-binding model.

From a model's own Hamiltonian, pythTQC measures:

1. the space group it obeys, each orbital's Wyckoff position and site irrep,
   and the model's representation of the group: ``realized_group(model)``;
2. the corep content of its bands at every high-symmetry point:
   ``hsp_table(r, bands=n)``;
3. whether that content could be an atomic limit: ``decompose``;
4. its symmetry-indicator class and its stable real-space invariants:
   ``symmetry_indicator``, ``srsi_of_bands``, ``srsi_of_model``, and whether a
   gapped splitting of its bands can be two atomic insulators: ``splitting``.

A first call::

    import pythtqc as tqc
    from pythtb.models import kane_mele

    r = tqc.realized_group(kane_mele(0.0, -1.0, 0.3, 0.0), characters={0: "s"})
    print(r.summary())            # the group, and the orbitals by Wyckoff orbit
    print(tqc.hsp_table(r, bands=2).symmetry_data)

A call that cannot complete returns a ``Refusal`` in its place, which prints
the check that stopped it and what it measured there.  A refusal is falsy, and
so are some values, an empty set of pairs or an empty content:
``isinstance(out, tqc.Refusal)`` is the test.  The package's own measurements
disagreeing raise ``InternalError``.

The space-group data is irrep's, from the package ``irreptables``;
``tables_status()`` says whether it loaded.  The documentation is the
repository's ``docs/`` folder, https://github.com/bbradlyn/pythtqc, starting
at ``docs/README.md``.

The modules: ``groups``, the space group and its tables; ``symmetry``, the
model's representation of the group; ``sewing``, sewing matrices and band
labels; ``ebr``, decompositions and symmetry indicators; ``srsi``, stable
real-space invariants; ``symmetrization``, models symmetric only to a declared
precision; ``spin``, spinor bases and their spin operator; ``models``,
reference crystals; ``refusal``, ``Refusal`` and ``InternalError``; ``utils``,
figures for the examples.
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
                   StablePair, atomic_realizations, nsit_criterion, splitting, srsi_of,
                   srsi_of_bands, srsi_of_columns, srsi_of_model, srsi_tables,
                   stably_equivalent_pairs)
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
    "srsi_tables", "srsi_of", "srsi_of_columns", "srsi_of_bands", "srsi_of_model",
    "splitting", "nsit_criterion", "atomic_realizations", "stably_equivalent_pairs",
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
