# Changelog

## Unreleased

- `hsp_table` reads every high-symmetry point, one whose little group holds the identity
  alone included: there a set of bands carries the point's one irrep once for each band,
  which the table's `content` and `counts` now hold, `M1` at `M` for Haldane's model with
  flux in `P3`. The table's field `skipped` is gone, and with it the `"skipped"` entry of
  a decomposition's `completeness` and of the context of `decompose`'s
  `incomplete_content`. `decompose` and `symmetry_indicator` return what they returned on
  such a table, and the same on its `counts` handed on as a mapping, which they refused.
  A set of bands that cuts a degenerate level at such a point is refused there,
  `multiplet_split`, as at every other point.
- `regular_completion` returns the padding of a band set in a group with such a point,
  which it refused with `not_a_band_structure`; in `P1` without time-reversal symmetry it
  returns the padding of the bands missing, where it counted the band set's own bands
  among them.
- `subtract` and `regular_completion` refuse a table that refused at some point,
  `incomplete_content`, as `decompose` does, where they read the points refused as holding
  nothing. `allow_incomplete=` does not lift it.
- `regular_content` asked at a point out of a layer's plane returns the refusal
  `out_of_plane`, where it raised `KeyError`.
- `realized_group` reads a crystal where it raised `ValueError`, "No magnetic space group
  could be detected!". Most such crystals are written on a lattice basis that is not
  reduced, the square lattice on `a1 + a2, a2` among them. What it returns for every
  other crystal is unchanged.
- `realized_group` takes each site's stated multiplets as `site_multiplets=`, where it
  took `multiplets=`. The multiplets stated are read as before, and
  `SiteMultiplets.summary()` ends `stated by site_multiplets=`.
- `realized_group` takes `precision=` as the one declaration of a numerical model's
  precision, beside `position_tol=` for a model whose orbital centers alone are imprecise
  and `symprec=`. It no longer takes `tol=`, `cut=` or `window=`, which a declared
  precision sets, nor `dz=` or `c=`, the depth of a substrate's shadow sites and the
  height of a layer's slab cell, which stay at `0.05` and `30.0`.
- `realized_group` returns the group the Hamiltonian obeys, and projects the Hamiltonian
  onto its symmetric part where a declared precision calls for it. It no longer takes
  `descend=`, which asked for the group of the cell and refused where the Hamiltonian
  broke any of it, or `symmetrize=`. What it returns on a call without them is unchanged.
- `symmetrize`, `symmetrize_positions` and `classify_operations` are no longer exported:
  `realized_group(model, precision=...)` moves the orbital centers, projects the
  Hamiltonian and classifies each operation itself, and returns the projection's record
  on `r.symmetrization`.
- At a Wyckoff position where the tables list no band representation, a site irrep takes
  the name of the site irrep that the tables of the stable real-space invariants (Hwang et
  al., Nat. Commun. 17, 8564 (2026)) give the same invariants, alike with time-reversal
  symmetry and without, and its `named_by` reads `"invariants"`, where it read
  `"point group"` or `"Mulliken"`. At 28 such positions, the lines of site symmetry `mm2`
  in 12 groups (`4l` to `4o` of `P4/mmm` among them), `B1` and `B2` change places, and a
  declared `B1` or `B2` there states the other row than it did, with spin as without.
  Every character, matrix and band label returned is unchanged.
- Every site of a Wyckoff orbit at such a position reads the orbit's name, whichever site
  is listed first: at `4i` of `P4/mmm`, where a fourfold rotation carries one site onto
  the other and exchanges `B1` with `B2`, the site that read the other symbol reads the
  orbit's.
- A tabulated k-point's reduced coordinates are returned in `[0, 1)`: a coordinate that is
  zero reads `0`, where on some cells, a triclinic one among them, it read `1`. The point
  is the same one.
- A set of bands is read at every tabulated point of a crystal written on a strongly skewed
  lattice basis: on the hexagonal lattice written on `a1 + 10 a2, a2, a3`, `K` and `H`
  were refused `kpoint_not_tabulated`, their coordinates having been compared after
  rounding. A k-point is the tabulated one where it differs from it by a reciprocal
  lattice vector of the cell as written.
- `realized_group` given `site_multiplets=` that separates orbitals a symmetry of the
  hoppings mixes returns the group that keeps them separate: `p_x` and `p_y` on the
  fourfold site of a square lattice, stated as two multiplets, give `Pmm21'` on a
  substrate and `Pmmm1'` free-standing or stacked into a crystal, where it raised
  `InternalError`.
- `realized_group` returns a group where it raised `IndexError`, on a model whose
  operations measured to hold do not close into a group, as a numerical model's
  operations near its tolerance can: the largest group they contain, as for any such
  model.
- `realized_group` returns the group of a model whose hoppings carry a relative noise near
  `1e-10`, no precision declared, where it raised `InternalError`: `P-43m1'` for
  `cubic_3c_perp_doublet` so perturbed. Its Hamiltonian is averaged over the group with
  the exact representation nearest the matrices read off its multiplets, which carry its
  noise.
- `realized_group` builds the representation of a crystal written on a strongly skewed
  lattice basis, where it refused `rep_not_covariant` though every operation held: the
  square lattice with one `s` orbital on `5 a1 + 8 a2, 3 a1 + 5 a2` among them. The
  covariance of `H(k)` is held to the rounding its Bloch phases carry at the rotated
  k-points besides that of the representation's own arithmetic.
- `realized_group` reads zincblende written on `a1, 6 a1 + a2, 4 a1 + 9 a2 + a3`, where it
  raised `InternalError`, the operations rebuilt from the tabulated ones not closing. Each
  rebuilt translation is formed exactly and rounded once.
- `realized_group` reads a crystal of two species written on a strongly skewed lattice
  basis with its sites not told apart by `typat=`, CsCl on
  `a1, 6 a1 + a2, 4 a1 + 9 a2 + a3` among them, where it refused
  `operations_not_consistent`. Whether a cell's operations close is judged relative to
  the size of their rotation matrices in the cell's reduced coordinates.
- `realized_group` with a precision declared reads a crystal written on a strongly skewed
  lattice basis whose centers are off their fractions by the rounding of their
  conversion, hexagonal boron nitride on `7 a1 + 5 a2, 4 a1 + 3 a2` among them, where it
  refused `rep_not_covariant`. The centers are moved onto the positions the operations
  carry onto one another by every coordinate of the move, where a coordinate whose move
  was the size of rounding was left where it was.
- The group `r.group` holds `name`, `number_str`, `spinor`, `periodic_dirs` and
  `magnetic_type`, `"II"` for a grey group and `"I"` for a group without time-reversal
  symmetry. Its other fields are no longer public: `sg_number`, `size`, `include_TR`,
  `magnetic`, `vacuum_dirs`, `symmetries`, `lattice`, `provenance` and the methods on k
  among them. Its operations are `r.operations`. Its repr writes `magnetic_type=` and
  `periodic_dirs=`, and its summary has a line for its magnetic type in place of the one
  for its operations.
- `operations`, `operation_index`, `high_symmetry_points`, `little_group`, `corep_table`,
  `identify_from_characters`, `k_grid`, `wyckoff_sites_tables`, `ebr_rows` and `ebr_data`
  take the group, `r.group`. Handed what `realized_group` returns, its representation or
  anything else, each raises `TypeError` naming `r.group`, where some read `r` or
  `r.rep` as the group and the others raised `AttributeError`.
- `seitz` is gone: `r.operations[i].seitz` is the symbol it returned.
- A row of `r.operations` names its index `index` and whether it is combined with time
  reversal `antiunitary`, where it named them `ind` and `time_reversal`. The table prints
  its first column under `index`.
- `little_group`, `corep_table` and `identify_from_characters` take a k-point by its
  tabulated name, `little_group(r.group, "X")`, as by its coordinates, and return the same
  for both. A name the group does not tabulate raises `ValueError` listing the names, and
  the name of a point out of a layer's plane returns the refusal `out_of_plane`.
  `corep_table` and `identify_from_characters` no longer take `kpname=` or `ktol=`, nor
  `k_grid` `ktol=`.
- What `little_group` returns holds its operations, `operations`, and the indices of its
  unitary and antiunitary operations, `unitary` and `antiunitary`, where `unitary` and
  `antiunitary` held irrep's records and `indices` the unitary indices. Its `G` is keyed by
  index. Its repr writes `k` as its summary does, `(1/3, 2/3)`.
- What `high_symmetry_points` returns names the tabulated points out of a layer's plane
  `out_of_plane`, where it named them `missing`. What `k_grid` returns holds its points as
  `kpoints`, where it held them as `points`, and no longer holds `hsp` or `missing`:
  `contains` says which tabulated points it holds.
- A row of `wyckoff_sites_tables` holds its Wyckoff position as `wyckoff`, `"1a"`, and the
  positions it slides onto as `slides_onto`, `("1a",)`, where it held `letter`,
  `multiplicity` and the letters `closure`. `wyckoff_positions_spglib` and
  `WyckoffPosition` are no longer exported.
- `ebr_rows` is keyed by the site irrep and the Wyckoff position, `rows["-E1", "1a"]`,
  where it was keyed by the band representation's full label. Each row is unchanged.
- `ebr_data` returns a record, `EBRData`, where it returned a mapping: its columns `ebrs`
  are `Term` rows, each a site irrep at a Wyckoff position with its site-symmetry group;
  `corep_names` are the rows' coreps, and `matrix`, `smith` and `convention` are as
  before, the convention an `EBRConvention` whose `corep_names_bcs` are the Bilbao
  Crystallographic Server's names of the rows. Every value is unchanged.

## 0.1.0a3 — the second public version

- On a layer without time-reversal symmetry, where an environment's largest subgroups tie
  and the one the tie rule keeps is not what the other measured, `realized_group` returns
  the cross-check's outcome `"ambiguous"`, as on a layer with time-reversal symmetry, where
  it raised `InternalError`.
- Every content returned or printed lists its coreps in one order: point by point, Γ first
  and the other points alphabetically, a partner point `−k` right after its point, and at
  each point in ascending order of the irreps' numbers, `GM2` before `GM10`. `subtract`
  returns that order from one run to the next, where its order changed between runs.
  Every list of points a call returns comes in that order, `high_symmetry_points` and an
  `hsp_table` among them, which changes their order for a group without time-reversal
  symmetry. Nothing else returned changes.
- `realized_group` and `hsp_table` return sooner, `realized_group` most on a group of many
  operations, a cubic crystal's among them. What they return is unchanged.
- `hsp_table`, `corep_content`, `characters`, `sewing_matrix` and a representation's
  `covariance` raise `ValueError` where the model the representation was measured on has
  been edited since, and say to call `realized_group` again. A layer's representation holds
  a copy of its own, and that of a parameterized model the model with the values of its
  parameters in place, which an edit of the model handed in does not reach.
- A parameterized model, one whose hoppings or on-site energies are free parameters with
  their values given as `params=`, is read by every call that takes a model:
  `realized_group`, `stack_presentation`, `solve_at`, `spinless_presentation`,
  `symmetrize` and `symmetrize_positions`, a layer and a model read under a declared
  precision included. Each returns what it returns on `model.with_parameters(...)`, and a
  record's `model` is the model with the numbers in place. A parameter takes a number: a
  hopping that acts on spin is written as a function of its parameter that returns the
  `2×2` block, as the guide's page on the group shows for the Kane–Mele model, and each
  set of values is one call.
- `hsp_table`, `corep_content`, `characters`, `sewing_matrix` and a representation's
  `covariance` take no `params=`: a representation is that of the model at the values it
  was measured at, and `r.model` is handed on without them. A name that is not a
  parameter of the model, a parameter given no value and a value that is not a number
  raise `ValueError` before anything is measured.
- An `hsp_table`'s `note` is one sentence, and it appears for any set of bands, all of them
  included, one of whose energy multiplets is labelled otherwise under another labelling
  the hoppings leave open.
- The package's name is written pythTQC; it is installed and imported as `pythtqc`, as
  before.
- `CITATION.cff` asks work that uses the package to cite Hwang et al., arXiv:2601.06293,
  which carries the stable real-space invariants to photonic crystals, as its preferred
  citation, beside the paper they come from and the package itself, by its version and
  its repository.
- The script that converts the stable real-space invariants' tables from the paper's TeX
  supplement, and a benchmark of declared content, leave the repository. The shipped tables
  are unchanged but for their provenance, which names the paper and the versions they were
  converted against and no script or source file. `srsi_data` says to reinstall the package,
  and `srsi_ebr_basis` names the `irreptables` version the tables were converted against.
- An eleventh example, `tqc_srsi_sg212_photonic.py`: the space group 212 tight-binding model
  of Hwang et al., arXiv:2601.06293, whose eight bands stand for a photonic crystal's four
  lowest bands and four auxiliary bands. It prints the coreps of the eight bands, the
  auxiliary bands' symmetry data vector, and the stable real-space invariants of the
  physical bands.
- `characters=` takes generator matrices, a mapping from operations of a site group to the
  matrices they act with on the orbitals of a key,
  `{(0, 1): {"C4(z)": [[0, -1], [1, 0]], "m(x)": [[-1, 0], [0, 1]]}}`. The representation
  returned gives those orbitals the declared matrices, which fixes the labelling where every
  labelling gives the site one irrep, as for `E` at `4a` of `P4_332`.
- `decompose`, `symmetry_indicator`, `completeness_check`, `regular_completion` and
  `srsi_of_bands` read a table in the group it carries: `decompose(t)` takes the table
  `hsp_table` returned and nothing else, the group and `rep=` being optional and held to
  the table's own where they are handed in; a content written out by hand still takes its
  group beside it. A layer's indicator asked of its table measures its rows out of the
  plane on the decoupled stack with no `rep=` passed.
- `SRSITables.formulas()` writes every stable real-space invariant's formula out, a line
  each, `theta_2 (Z2) = 1 (-E)_b`; `SRSIRow.summary()` says multiplicity where it said
  copies.
- The ten examples print as a ledger: each measured quantity on a labelled line, a
  record's summary whole and once, every verdict in words, bands counted from 0 and
  said so once per script, the valence and the conduction bands named in words. The
  summaries of `ModelSRSIs`, `Splitting`, `Realizations`, `BandSRSIs` and
  `SRSITables` define nothing and name the two sides of a gap in words, each
  atomic realization and each consistent pair written as the orbitals it is; the
  criterion's `reason` says the same in the same words. No verdict word and no field
  changes.
- An `HSPTable` carries the representation it was read with, `t.rep`, and
  `decompose`, `symmetry_indicator`, `completeness_check`, `regular_completion`
  and `srsi_of_bands` raise `ValueError` when handed the table with another group
  or another representation, as they do for `bands=` naming other bands; a table
  read in one group was decomposed in another with no refusal before.
- The package is re-sorted into thirty-four modules in one acyclic order, each reading
  only the ones beneath it; `groups` and `symmetry` are assembled from the private
  modules below them, and every public name stays where it was.
- `frontier_subspace` and its record `Frontier` leave `pythtqc.sewing` and the
  top level: nothing in the package called them.

## 0.1.0a2 — the first public version

pythtqc reads the crystalline symmetry of a PythTB tight-binding model:

- **The space group the Hamiltonian obeys.** `realized_group` tests every operation
  the cell allows on `H(k)` and returns the group with each orbital's Wyckoff
  position and site irrep, for three-dimensional crystals and for layers on a
  substrate or free-standing, spinless or with spin, built by hand or imported
  from wannier90 with a declared precision.
- **The bands, labelled.** `hsp_table` and `corep_content` read the corep content
  of a set of bands at every high-symmetry point, and `sewing_matrix` the matrix
  of one operation on them.
- **Atomic limits.** `decompose` asks whether that content is a sum of elementary
  band representations, with `completeness_check` and `regular_completion`
  beside it.
- **Invariants.** `symmetry_indicator` reads the indicator class, and
  `srsi_of_bands`, `srsi_of_model` and `splitting` the stable real-space
  invariants, from the tables of Hwang et al., Nat. Commun. **17**, 8564 (2026).

Every record a call returns prints itself with `summary()`; a call that cannot
answer returns a `Refusal` that says why. The space-group data is read from
`irrep`'s tables, and every page of the documentation runs as written.
