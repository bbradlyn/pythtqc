# Functions

Every function `pythtqc` exports, by task: its signature, what it is for, the
arguments it takes, what it returns, and the checks at which it refuses. The
page is written from the docstrings by `scripts/build_reference.py` and is not
edited by hand. A docstring holds more than its entry here, notes and examples
among it, and `help(tqc.<name>)` prints the whole of one, with
`import pythtqc as tqc`: `help(tqc.realized_group)`, for one.

A call that cannot complete returns a `Refusal` in place of its record: a
falsy record that names the check that stopped it and what was measured there.
Each entry's **Refusals** are the checks at which that call can refuse.

The functions fall into seven groups:

- [The group](#the-group) — the space group a Hamiltonian has and which operations it obeys, the orbitals' site orbits, and the operations by index and Seitz symbol.
- [Declaring what the hoppings cannot supply](#declaring-what-the-hoppings-cannot-supply) — the spin a spinless model carries in its orbital index.
- [Group data in this cell's coordinates](#group-data-in-this-cells-coordinates) — the tables' k-points, little groups, character tables, Wyckoff positions and elementary band representations, in the model's own cell.
- [Labelling the bands](#labelling-the-bands) — the states at one k-point, and the sewing matrices, characters and corep content of a set of bands, at one high-symmetry point or at all of them.
- [Atomic limits](#atomic-limits) — whether a symmetry vector is a sum of elementary band representations, and the counting beside it.
- [Stable invariants](#stable-invariants) — the symmetry-indicator class of a band set, and its stable real-space invariants, from the bands or from the orbitals.
- [Numerical and `wannier90` models](#numerical-and-wannier90-models) — a model symmetric only to its own precision: its orbital centres moved onto the nearest positions the group carries, its hoppings onto the group's own average, and each operation classified.

## The group

- [`realized_group`](#realized_group) — The space group obeyed by H, obtained by descent from the space group of the lattice of occupied sites.
- [`operations`](#operations) — Every operation of a group by its index, with its Seitz symbol and its name in the model's frame.
- [`seitz`](#seitz) — The Seitz symbol of one operation, by its index: the one [`operations`](#operations) writes in its row.
- [`operation_index`](#operation_index) — The index of the operation a Seitz symbol names.

### `realized_group`

```text
realized_group(
    model, *, typat=None, environment=None, cross_check=True, decorations=None, dz=0.05,
    c=30.0, nk=6, tol=None, cut=None, window=None, symmetrize='auto',
    max_bytes=1000000000.0, position_tol=None, precision=None, spinor=None, spin=None,
    descend=True, symprec=1e-05, params=None, characters=None, multiplets=None,
    max_candidates=1048576
)
```

The space group obeyed by H, obtained by descent from the space group of the lattice of occupied sites.

**Parameters**

- **`model`** (`pythtb.TBModel`): A crystal periodic in three directions, or a layer periodic in two; a
  chain or a molecule is refused.

- **`typat`** (`sequence, one entry per orbital, optional`): A species label per orbital: orbitals on atoms of different species
  get different labels.  `None` reads sites with the same orbital
  content as one species.

- **`characters`** (`sequence or mapping, optional`): What the orbitals are: a function (`"s"`, `"px"`, `"dx2-y2"`), a
  shell (`"p"`), a site irrep (`"A1"`), with spin a site corep
  (`"-E1"`), or a sum of them (`"A1 + E"`), one per orbital, or a
  mapping from an orbital, or from several orbitals of one site, to such
  a word.  In place of a word, a mapping from operations of the site
  group to their characters, or to the matrices they act with on the
  orbitals named, in the order written: matrices say which basis of its
  site irrep a multiplet is.  The group loses the operations that do not
  carry the declared content.  An orbital left undeclared takes the
  lowest angular momentum the hoppings allow.

- **`environment`** (`{"substrate", "free-standing"}, optional`): For a layer: on a substrate, no operation turns the layer over;
  free-standing, the whole layer group applies.  Default: the substrate
  for a flat layer, free-standing for a slab whose orbitals have heights.

- **`cross_check`** (`bool, optional`): For a layer, also read the other environment and check that the two
  agree.  Default `True`.

- **`decorations`** (`sequence of (position, label), optional`): Non-orbital species added to the cell before its group is found: a
  crystal field the orbital positions do not show.

- **`precision`** (`float, optional`): For a numerical model such as a wannier90 import, the relative
  precision of its numbers.  It sets the tolerances below; centres and
  hoppings symmetric within it are made exactly symmetric.

- **`spin`** (`str, mapping or SpinBasis, optional`): For a spinless model whose orbital index carries spin, as a wannier90
  import's does: which two orbitals are the spin states of one Wannier
  function, `"interleaved"`, `"blocked"` or pair by pair.  Left out,
  such a model is read spinless.

- **`spinor`** (`bool, optional`): Whether the states carry spin one-half.  Default: from the model, and
  `True` with `spin`.

- **`descend`** (`bool, optional`): `False` returns the group of the lattice of occupied sites itself, and
  refuses `cell_group_not_realized` where H does not obey it.  Default
  `True`.

- **`symmetrize`** (`{"auto", True, False}, optional`): For operations or multiplets that hold only within the tolerance:
  `"auto"` (default) projects H onto its symmetric part where there are
  any, `True` always, `False` never, and such a model is then refused.

- **`tol`, `cut`, `window`, `position_tol`, `symprec`** (`float, optional`): The tolerances `precision` sets, one by one: `tol` absolute, in the
  units of H; `cut` and `window` relative; `position_tol` in reduced
  coordinates; `symprec` Cartesian, default `1e-5`.  Left out, each
  is exact.

- **`nk`** (`int, optional`): k-points per periodic direction of the grid the operations are measured
  on.  Default 6.

- **`dz`, `c`** (`float, optional`): A layer's presentation: the depth of the substrate's shadow sites and
  the height of the slab's cell.  Neither changes the group.

- **`multiplets`** (`list or SiteMultiplets, optional`): Each site's multiplets, stated instead of measured: one list per site,
  each multiplet written as its orbitals.

- **`max_candidates`, `max_bytes`** (`int and float, optional`): Limits on the search for the representation (default `2**20`
  candidates) and on the memory of the symmetric projection (default
  `1e9` bytes).

- **`params`** (`dict, optional`): Values for a model whose terms are given by parameter names.

**Returns**

- `RealizedGroup or Refusal`: `r.summary()` prints what was measured.  The record unpacks as
  `model, group, assignment, validated, rep = r`: the group, its
  operations by Seitz symbol in `r.operations`; each orbital's Wyckoff
  position; the verdict on each operation; and the representation, which
  labels bands (`hsp_table(r.rep, bands=n)`) and holds each orbital's
  site irrep (`r.rep.characters`).  By name: `r.descent`, what was
  broken; `r.time_reversal`; `r.environment` and `r.cross_check`,
  for a layer; `r.beyond_grey`, operations combined with time reversal
  that hold where time reversal does not; `r.precision`,
  `r.positions` and `r.symmetrization`, for a numerical model;
  `r.conserved`, symmetry beyond the crystal's.  A `Refusal` in its
  place is falsy and names what stopped the call.

**Refusals**

Once the group is found, a refusal names it and carries it in
`context["group"]`.

- **`fewer_than_two_periodic_directions`**: a chain or a molecule.

- **`irrep_tables` / `single_valued_spinful`**: `irreptables` is not installed, or `spinor=False` was asked of
  states with spin one-half.

- **`spin_pair_split`**: the two orbitals of a declared spin pair sit at different centres.

- **`wyckoff_orbit`**: an orbital sits on no Wyckoff position of the cell.

- **`positions_not_exactly_symmetric`**: centres off symmetric positions by enough to break operations, with no
  `precision` declared.

- **`supercell`**: H obeys a translation shorter than the cell's: the model is written on
  a supercell.

- **`operations_not_nearly_consistent` / `operations_not_consistent`**: the operations found on a numerical cell do not close; `precision`
  makes the centres exact first.

- **`cell_group_not_realized` / `declaration_breaks_group`**: with `descend=False`, operations broken by the hoppings or by the
  declared content.

- **`unknown_label` / `ambiguous_multiplet`**: a declared word the site does not name, or a flat list that cannot say
  which orbitals form one multiplet.

- **`rep_not_covariant`**: with `symmetrize=False`, operations that hold only within the
  tolerance.

- **`rep_internal_factor` / `rep_not_multiplicative`**: no candidate representation carries the group's phases, or multiplies
  as the group does.

- **`rep_not_unique`**: more than one representation fits; report it, with the model and the
  call.

- **`rep_double_valued_spinless`**: double-valued tables asked of a spinless model; `spin=` declares its
  spin pairs.

- **`rep_search_too_large` / `projection_too_large` / `detector_too_large`**: past `max_candidates` or `max_bytes`; the reason gives the value
  that runs it.

- **`symmetrize_broken_operation` / `symmetrize_defect_exceeds_tol` / `symmetrize_range_not_closed`**: the symmetric projection's own ([`symmetrize`](#symmetrize)).

### `operations`

```text
operations(group_or_rep)
```

Every operation of a group by its index, with its Seitz symbol and its name in the model's frame.

**Parameters**

- **`group_or_rep`** (`RealizedGroup, GroupData or SpaceGroupRep`): A group, every operation of which is a row, or a representation, whose
  rows are the operations it represents.  `r`, `r.group` and
  `r.rep` may each be handed in, and `r.operations` is the same
  table.

**Returns**

- `OperationTable or Refusal`: `table[ind]` is one `Operation`, with its `seitz`, its
  `name`, its matrices and whether it carries time reversal;
  `print(table)` lists them all.  A `Refusal` in its place is falsy.

**Refusals**

A refusal handed in as `group_or_rep` is returned as it is.

- **`tables_not_ready`**: The group has no conventional setting to write a Seitz symbol in.

### `seitz`

```text
seitz(group_or_rep, ind)
```

The Seitz symbol of one operation, by its index: the one [`operations`](#operations) writes in its row.

**Parameters**

- **`group_or_rep`** (`RealizedGroup, GroupData or SpaceGroupRep`): The group, or a representation of it.

- **`ind`** (`int`): The operation's index.

**Returns**

- `str or Refusal`: The symbol, `{3+_001|0,0,0}`.  A `Refusal` in its place is falsy.

**Refusals**

A refusal handed in as `group_or_rep` is returned as it is.

- **`tables_not_ready`**: The group has no conventional setting to write a Seitz symbol in.

### `operation_index`

```text
operation_index(group_or_rep, seitz)
```

The index of the operation a Seitz symbol names.

**Parameters**

- **`group_or_rep`** (`RealizedGroup, GroupData or SpaceGroupRep`): The group, or a representation of it, whose own operations are then
  the ones searched.

- **`seitz`** (`str`): A Seitz symbol `{R|t}`, with a trailing `'` for an operation
  combined with time reversal.

**Returns**

- `int or Refusal`: The operation's index.  A `Refusal` in its place is falsy.

**Refusals**

A refusal handed in as `group_or_rep` is returned as it is.

- **`tables_not_ready`**: The group has no conventional setting to read a Seitz symbol in.

- **`unknown_operation`**: No operation of the group is written so; the refusal names the nearest
  spellings it carries.

## Declaring what the hoppings cannot supply

- [`spin_basis`](#spin_basis) — Which orbitals of a spinless model are the spin up and down states of one Wannier function.
- [`spin_operator`](#spin_operator) — The spin along `n̂`, in units of `ħ/2`, on the orbitals of a declared spinor basis: `n̂·σ` on each pair.
- [`spinless_presentation`](#spinless_presentation) — The same crystal as a `wannier90` spinor import delivers it: `2N` orbitals, the spin in the orbital index.

### `spin_basis`

```text
spin_basis(model, *, order='interleaved', pairs=None)
```

Which orbitals of a spinless model are the spin up and down states of one Wannier function.

**Parameters**

- **`model`** (`pythtb.TBModel`): A spinless model whose orbital index carries spin, as a `wannier90`
  spinor import's does; a native spinful model raises.

- **`order`** (`{"interleaved", "blocked"}, optional`): `"interleaved"` (default), `(0↑, 0↓, 1↑, 1↓, …)`, or `"blocked"`,
  `(0↑, 1↑, …, 0↓, 1↓, …)`.

- **`pairs`** (`sequence of (int, int), optional`): The pairs one by one, up first, every orbital in exactly one: an
  irregular ordering, in place of `order`.

**Returns**

- `SpinBasis`: `sb.summary()` prints the pairs, which `sb.pairs` holds.  It goes to
  `realized_group(model, spin=sb)` and to [`spin_operator`](#spin_operator).

### `spin_operator`

```text
spin_operator(spin, n=(0.0, 0.0, 1.0))
```

The spin along `n̂`, in units of `ħ/2`, on the orbitals of a declared spinor basis: `n̂·σ` on each pair.

**Parameters**

- **`spin`** (`SpinBasis`): The pairs, from [`spin_basis`](#spin_basis) or [`spinless_presentation`](#spinless_presentation).

- **`n`** (`array_like, optional`): The axis `n̂`, in the model's Cartesian frame, not necessarily
  normalized.  Default `(0, 0, 1)`, `σ_z`.

**Returns**

- `numpy.ndarray`: The Hermitian `norb × norb` matrix: `n̂·σ` on each pair's two
  orbitals, the up one first, and zero between pairs.

### `spinless_presentation`

```text
spinless_presentation(model, *, order='interleaved', pairs=None, params=None)
```

The same crystal as a `wannier90` spinor import delivers it: `2N` orbitals, the spin in the orbital index.

**Parameters**

- **`model`** (`pythtb.TBModel`): A spinful model.

- **`order`** (`{"interleaved", "blocked"}, optional`): `"interleaved"` (default), `(0↑, 0↓, 1↑, 1↓, …)`, pythtb's own order
  of the states, or `"blocked"`, `(0↑, 1↑, …, 0↓, 1↓, …)`.

- **`pairs`** (`sequence of (int, int), optional`): The pairs one by one, up first, every orbital in exactly one, in place
  of `order`.

- **`params`** (`dict, optional`): Values for a model whose terms are given by parameter names.

**Returns**

- `(pythtb.TBModel, SpinBasis) or Refusal`: The spinless model, and its pairing, which
  `realized_group(flat, spin=sb)` reads as the native model's spin.  A
  `Refusal` in its place is falsy.

**Refusals**

- **`not_a_spinful_model`**: a spinless model, which is in this presentation already;
  [`spin_basis`](#spin_basis) declares its pairs.

## Group data in this cell's coordinates

- [`high_symmetry_points`](#high_symmetry_points) — The tabulated high-symmetry points, in the model's own reduced coordinates.
- [`little_group`](#little_group) — The operations that leave `k` invariant up to a reciprocal lattice vector.
- [`corep_table`](#corep_table) — The character table of the little group at a high-symmetry point: each corep's character on each unitary operation.
- [`wyckoff_sites_tables`](#wyckoff_sites_tables) — The Wyckoff position of each site, lettered in the tables' setting, the one every table is read with.
- [`k_grid`](#k_grid) — A Γ-centred mesh that respects the group, with the high-symmetry points it holds.
- [`ebr_data`](#ebr_data) — Every elementary band representation of the group: the bands each site irrep at each Wyckoff position induces.
- [`tables_status`](#tables_status) — Whether irrep's character tables are in place, and `ortools` installed.

### `high_symmetry_points`

```text
high_symmetry_points(group)
```

The tabulated high-symmetry points, in the model's own reduced coordinates.

**Parameters**

- **`group`** (`GroupData`): The group, `realized_group(model).group`.

**Returns**

- `HighSymmetryPoints or Refusal`: `{name: k}`, each `k` with `dim_k` components; its `missing`
  names the tabulated points out of a layer's plane.  `summary()`
  prints it.  A `Refusal` in its place is falsy.

**Refusals**

A refusal handed in as `group` is returned as it is.

- **`tables_not_ready`**: The group has no transformation to the tables' conventional cell, to
  carry their points into the model's coordinates.

### `little_group`

```text
little_group(group, k)
```

The operations that leave `k` invariant up to a reciprocal lattice vector.

**Parameters**

- **`group`** (`GroupData`): The group, `realized_group(model).group`.

- **`k`** (`array, shape (dim_k,)`): The point, in the model's reduced coordinates.  A layer takes a point
  of its three-dimensional cell too, where it lies in the plane.

**Returns**

- `LittleGroup or Refusal`: Its unitary operations in index order, the columns a character vector
  at `k` is written in, and those combined with time reversal, which
  carry no characters; `summary()` prints it.  A `Refusal` in its
  place is falsy.

**Refusals**

A refusal handed in as `group` is returned as it is.

- **`out_of_plane`**: A point with a component along a layer's vacuum, a direction its zone
  does not have.

### `corep_table`

```text
corep_table(group, k, *, kpname=None, ktol=1e-05)
```

The character table of the little group at a high-symmetry point: each corep's character on each unitary operation.

**Parameters**

- **`group`** (`GroupData`): The group, `realized_group(model).group`.

- **`k`** (`array, shape (dim_k,)`): A tabulated point, in the model's reduced coordinates, as
  [`high_symmetry_points`](#high_symmetry_points) gives it.

- **`kpname`** (`str, optional`): The point's tabulated name, taken as given instead of found from
  `k`.

- **`ktol`** (`float, optional`): How far `k` may sit from a tabulated point and be it, in reduced
  reciprocal coordinates.  Default `1e-5`.

**Returns**

- `dict or Refusal`: `{corep: {ind: character}}`, one entry per corep of the little group,
  its character on each unitary operation by index.  A `Refusal` in
  its place is falsy.

**Refusals**

A refusal handed in as `group` is returned as it is.

- **`tables_not_ready`**: The group has no transformation to the tables' conventional cell.

- **`out_of_plane`**: `k` has a component along a layer's vacuum.

- **`kpoint_not_tabulated`**: `k` is no tabulated point in the model's coordinates, or not the
  tabulated member of its star; the refusal lists the tabulated points.

### `wyckoff_sites_tables`

```text
wyckoff_sites_tables(group, positions=None, *, position_tol=0.0001)
```

The Wyckoff position of each site, lettered in the tables' setting, the one every table is read with.

**Parameters**

- **`group`** (`GroupData`): The group, `realized_group(model).group`.  A grey group's positions
  are those of its unitary subgroup.

- **`positions`** (`array, shape (n, 3), optional`): Points in reduced coordinates of the group's three-dimensional cell.
  Default: the points the group was found on,
  `GroupData.site_positions`.

- **`position_tol`** (`float, optional`): How far a point may sit from a position and lie on it, in reduced
  coordinates of the tables' conventional cell.  Default
  `WYCKOFF_REFERENCE_TOL`, `1e-4`.

**Returns**

- `tuple of SiteReference or Refusal`: One per point: its letter and multiplicity, its free parameters, and
  the positions an orbital confined there can be moved to.  A
  `Refusal` in its place is falsy.

**Refusals**

A refusal handed in as `group` is returned as it is.

- **`tables_not_ready`**: The group has no transformation to the tables' conventional cell.

- **`wyckoff_setting`**: `pyxtal`'s list of the group's positions is not written in the
  tables' setting, so no letter would be a table key.

- **`wyckoff_unresolved`**: A point lies on no position within `position_tol`; the refusal names
  the nearest and its distance.

- **`ebr_tables`, `irrep_tables`, `ebr_convention`**: From [`ebr_data`](#ebr_data), whose positions the letters are checked
  against.

### `k_grid`

```text
k_grid(group, n, *, allow_asymmetric=False, ktol=1e-08)
```

A Γ-centred mesh that respects the group, with the high-symmetry points it holds.

**Parameters**

- **`group`** (`GroupData`): The group, `realized_group(model).group`.

- **`n`** (`int or sequence of int`): The divisions along each periodic direction, or one number for all.

- **`allow_asymmetric`** (`bool, optional`): Return a mesh the group does not carry onto itself, with no
  irreducible points, instead of refusing it.  Default `False`.

- **`ktol`** (`float, optional`): How far a mesh point may sit from a tabulated point and be it, in
  reduced reciprocal coordinates.  Default `1e-8`.

**Returns**

- `KGrid or Refusal`: The mesh, ready for a `pythtb.WFArray`, with the tabulated points it
  holds and those it misses; `summary()` prints it.  A `Refusal` in
  its place is falsy.

**Refusals**

A refusal handed in as `group` is returned as it is.

- **`asymmetric_grid`**: The group does not carry the mesh onto itself, so a sum over it would
  weigh points unlike their partners; `allow_asymmetric=True` returns
  it all the same.

### `ebr_data`

```text
ebr_data(group)
```

Every elementary band representation of the group: the bands each site irrep at each Wyckoff position induces.

**Parameters**

- **`group`** (`GroupData`): The group, `realized_group(model).group`; only its label, its
  valuedness and whether it is grey are read.

**Returns**

- `dict or Refusal`: `matrix`: the EBR matrix, a row per corep at each high-symmetry
  point of the group, those out of a layer's plane included, and a
  column per elementary band representation;
  `labels_physics_convention`: the rows' coreps, in the labels
  `identify_from_characters` returns;
  `ebrs`: each column as `(name, position)`, the position with its
  site group, an improper one with its bar, `"3a(-3m1',-3m)"`;
  `smith`: the Smith normal form `(U, D, V)`, `U·M·V = D`;
  `convention`: what the convention moved, by point and by label, with
  a `note`: `changed_points` and `changed_labels`,
  `conjugated_points`, `divergent_points`, `reglued_points`;
  `conjugation`, each label's partner, and `tr_partner`, each
  column's, the column of the complex-conjugate site irrep;
  `labels_bcs`, the server's names of the rows;
  and `row_permutation`, `P` with `y_BCS[j] = y[P[j]]`, which
  re-keys a form written in the server's labels;
  `raw`: the tables' own record, carried into this convention.
  A `Refusal` in its place is falsy.

**Refusals**

A refusal handed in as `group` is returned as it is.

- **`irrep_tables`**: `irreptables` is not installed.

- **`ebr_tables`**: `irreptables` ships no elementary band representations for this
  group and valuedness.

- **`ebr_convention`**: An entry of the shipped file with no Wyckoff position, so its row
  cannot be carried into this convention; every known case is a magnetic
  group of type III, outside the groups returned.

### `tables_status`

```text
tables_status()
```

Whether irrep's character tables are in place, and `ortools` installed.

**Returns**

- `dict`: `tables`: whether the real tables are in place, and `reason`, where
  they are not, which of the two causes it is and what to do;
  `ortools`: whether the integer solver a decomposition needs is
  installed; `irrep_version` and `irreptables_version`: the versions
  installed, `None` where a package is absent.

## Labelling the bands

- [`solve_at`](#solve_at) — The energies and eigenvectors at one k, from one diagonalization.
- [`sewing_matrix`](#sewing_matrix) — The matrix of one operation on a set of bands, at a k-point the operation leaves invariant: the sewing matrix `B(g,k)_mn = <u_m(k)|U(g;k)|u_n(k)>`.
- [`characters`](#characters) — The characters of a set of bands at one k: the trace of each unitary operation of the little group over each degenerate multiplet of the set.
- [`corep_content`](#corep_content) — The coreps a set of bands carries at one high-symmetry point, and which of its degenerate multiplets carry the same characters.
- [`hsp_table`](#hsp_table) — The coreps a set of bands carries at every high-symmetry point of the group: its symmetry vector.
- [`format_counts`](#format_counts) — A content on one line: `{'-GM6': 2, '-GM7': 1}` as `2×-GM6 + -GM7`.
- [`multiplets`](#multiplets) — Which bands of a spectrum are one degenerate level, at a stated energy tolerance.

### `solve_at`

```text
solve_at(model, k, *, params=None)
```

The energies and eigenvectors at one k, from one diagonalization.

**Parameters**

- **`model`** (`pythtb.TBModel`): The model, `r.model`.

- **`k`** (`array_like, shape (dim_k,)`): One k-point, in reduced coordinates.

- **`params`** (`dict, optional`): Values for a model whose terms are given by parameter names.

**Returns**

- `(ndarray, ndarray)`: `(evals, evecs)`: the energies, ascending, and the eigenvectors as
  rows, as `model.solve_ham` returns them with `return_eigvecs=True`
  and `flatten_spin_axis=True`.

### `sewing_matrix`

```text
sewing_matrix(
    rep, operation=None, k=None, *, states=None, bands=None, atol=1e-08,
    degeneracy_tol=1e-08, mesh_tol=1e-09, check=True, params=None, **kwargs
)
```

The matrix of one operation on a set of bands, at a k-point the operation leaves invariant: the sewing matrix `B(g,k)_mn = <u_m(k)|U(g;k)|u_n(k)>`.

**Parameters**

- **`rep`** (`SpaceGroupRep`): The model's representation, `r.rep` from [`realized_group`](#realized_group).

- **`operation`** (`int`): The operation, by its index: `r.operations` lists them, and
  [`operation_index`](#operation_index) finds one by its Seitz symbol.
  Required.

- **`k`** (`array_like, shape (dim_k,)`): A k-point the operation leaves invariant up to a reciprocal lattice
  vector, in reduced coordinates.  Required.

- **`bands`** (`int or sequence of int, optional`): The set of bands: `n` the lowest `n` bands, or the band indices,
  whole degenerate multiplets.  Default: every state.

- **`states`** (`None, pythtb.WFArray, (evals, evecs) or ndarray, optional`): The states at `k`: `None` (default) solves the model there; a
  solved `WFArray`, or the pair [`solve_at`](#solve_at) returns, reuses one
  diagonalization.  Eigenvector rows alone, or a `WFArray` of a sector
  with fewer states than the model, carry no energies: whether they solve
  `H(k)` and close their multiplets is then not measured.

- **`degeneracy_tol`** (`float, optional`): The energy, in the units of `H`, within which bands are one
  multiplet.  Default `1e-8`.

- **`atol`, `mesh_tol`** (`float, optional`): In reduced reciprocal coordinates: whether the operation leaves `k`
  invariant (default `1e-8`), and finding `k` on a mesh (`1e-9`).

- **`check`** (`bool, optional`): Check that the states solve `H(k)` and that `bands` closes its
  multiplets.  Default `True`.

- **`params`** (`dict, optional`): Values for a model whose terms are given by parameter names.

- **`**kwargs`**: Passed to `SpaceGroupRep.sewing_operator`.

**Returns**

- `SewingMatrix or Refusal`: `B.summary()` prints it.  `B.matrix` is the matrix, `B.trace` its
  trace and `B.angles` the phases of its eigenvalues; `B.unitarity`,
  `B.commutator` and `B.residual` how far it is from unitary, the
  operator from commuting with `H(k)`, and the states from eigenstates.
  A `Refusal` in its place is falsy and names what stopped the call.

**Refusals**

- **`little_group`**: the operation does not leave `k` invariant.

- **`multiplet_split`**: `bands` cuts a degenerate multiplet at `k`.

- **`states_not_at_k`**: the states handed in do not solve `H(k)`.

- **`k_not_on_mesh`**: the mesh handed in does not hold `k`.

### `characters`

```text
characters(
    rep, k, *, states=None, bands=None, degeneracy_tol=1e-08, atol=1e-08,
    mesh_tol=1e-09, params=None
)
```

The characters of a set of bands at one k: the trace of each unitary operation of the little group over each degenerate multiplet of the set.

**Parameters**

- **`rep`** (`SpaceGroupRep`): The model's representation, `r.rep` from [`realized_group`](#realized_group).

- **`k`** (`array_like, shape (dim_k,)`): A k-point, in reduced coordinates; only a tabulated high-symmetry point
  is labelled ([`corep_content`](#corep_content)).

- **`bands`** (`int or sequence of int`): The set of bands: `n` the lowest `n` bands, or the band indices,
  whole degenerate multiplets.  Required.

- **`states`** (`None, pythtb.WFArray or (evals, evecs), optional`): The states at `k`: `None` (default) solves the model there; a
  solved `WFArray`, or the pair [`solve_at`](#solve_at) returns, reuses one
  diagonalization.  They carry energies, the rows being the multiplets.

- **`degeneracy_tol`** (`float, optional`): The energy, in the units of `H`, within which bands are one
  multiplet.  Default `1e-8`.

- **`atol`, `mesh_tol`** (`float, optional`): In reduced reciprocal coordinates: whether an operation leaves `k`
  invariant (default `1e-8`), and finding `k` on a mesh (`1e-9`).

- **`params`** (`dict, optional`): Values for a model whose terms are given by parameter names.

**Returns**

- `CharacterTable or Refusal`: `table.summary()` prints it.  `table.characters` holds a row per
  multiplet and a column per operation; `table.multiplets` and
  `table.energies` are the rows, `table.columns` the operations by
  index.  A `Refusal` in its place is falsy and names what stopped the
  call.

**Refusals**

A refusal handed in as `rep` is returned unchanged, and one from
[`little_group`](#little_group) as it is.

- **`multiplet_split`**: the set of bands cuts a degenerate multiplet at `k`.

- **`states_not_at_k`**: the states handed in do not solve `H(k)`.

- **`k_not_on_mesh`**: the mesh handed in does not hold `k`.

### `corep_content`

```text
corep_content(
    rep, k, *, states=None, bands=None, degeneracy_tol=1e-08, label_tol=0.001,
    ktol=1e-05, kpname=None, atol=1e-08, mesh_tol=1e-09, params=None
)
```

The coreps a set of bands carries at one high-symmetry point, and which of its degenerate multiplets carry the same characters.

**Parameters**

- **`rep`** (`SpaceGroupRep`): The model's representation, `r.rep` from [`realized_group`](#realized_group).

- **`k`** (`array_like`): A tabulated high-symmetry point, in the model's reduced coordinates, as
  [`high_symmetry_points`](#high_symmetry_points) gives it: `hsp["GM"]`.

- **`bands`** (`int or sequence of int`): The set of bands: `n` the lowest `n` bands, or the band indices,
  whole degenerate multiplets.  Required.

- **`states`** (`None, pythtb.WFArray or (evals, evecs), optional`): The states at `k`: `None` (default) solves the model there; a
  solved `WFArray`, or the pair [`solve_at`](#solve_at) returns, reuses one
  diagonalization.

- **`degeneracy_tol`** (`float, optional`): The energy, in the units of `H`, within which bands are one
  multiplet.  Default `1e-8`.

- **`label_tol`** (`float, optional`): How far a fitted multiplicity may sit from an integer, and two
  multiplets' characters from each other to share a partition class.
  Default `1e-3`.

- **`kpname`** (`str, optional`): The tables' name of the point, in place of finding it from `k`.

- **`ktol`, `atol`, `mesh_tol`** (`float, optional`): In reduced reciprocal coordinates: matching `k` to a tabulated point
  (default `1e-5`), whether an operation leaves `k` invariant
  (`1e-8`), and finding `k` on a mesh (`1e-9`).

- **`params`** (`dict, optional`): Values for a model whose terms are given by parameter names.

**Returns**

- `CorepContent or Refusal`: `c.summary()` prints the coreps multiplet by multiplet.  `c.counts`
  is the content, `{corep: multiplicity}`; `c.labels` the same for
  each multiplet, lowest first; `c.partition` which multiplets carry the
  same characters, numbered alike, which holds under any labelling of the
  orbitals; `c.isolation` the gap from these bands to the nearest other
  band; `c.table` the characters.  A `Refusal` in its place is falsy
  and names what stopped the call.

**Refusals**

A refusal handed in as `rep` is returned unchanged.

- **`multiplet_split`**: the set of bands cuts a degenerate multiplet at `k`: label the
  multiplets whole.

- **`kpoint_not_tabulated`**: `k` is not a tabulated point in this cell, or not the tabulated member
  of its star.

- **`integer_multiplicity` / `multiplet_dimension`**: the characters fit no whole number of coreps, or coreps that do not fill
  their multiplet: most often a multiplet grouped wrongly.

- **`states_not_at_k` / `k_not_on_mesh`**: states handed in from another point, or a mesh that does not hold `k`.

- **`little_group_mismatch` / `tables_not_ready`**: the tables do not apply in this cell.

### `hsp_table`

```text
hsp_table(
    rep, *, states=None, bands=None, degeneracy_tol=1e-08, label_tol=0.001, ktol=1e-05,
    atol=1e-08, mesh_tol=1e-09, params=None
)
```

The coreps a set of bands carries at every high-symmetry point of the group: its symmetry vector.

**Parameters**

- **`rep`** (`SpaceGroupRep`): The model's representation, `r.rep` from [`realized_group`](#realized_group).

- **`bands`** (`int or sequence of int`): The set of bands: `n` the lowest `n` bands, or the band indices,
  whole degenerate multiplets at every point.  Required.

- **`states`** (`None or pythtb.WFArray, optional`): `None` (default) solves the model at the points, once; a solved
  `WFArray` is read as it stands, and a point its mesh does not hold is
  refused there.

- **`degeneracy_tol`** (`float, optional`): The energy, in the units of `H`, within which bands are one
  multiplet.  Default `1e-8`.

- **`label_tol`, `ktol`, `atol`, `mesh_tol`** (`float, optional`): The tolerances of [`corep_content`](#corep_content), read at each point.

- **`params`** (`dict, optional`): Values for a model whose terms are given by parameter names.

**Returns**

- `HSPTable or Refusal`: `t.summary()` prints the coreps at each point.  The table is what
  [`decompose`](#decompose), [`symmetry_indicator`](#symmetry_indicator) and
  [`srsi_of_bands`](#srsi_of_bands) take, `decompose(t)`, in the group it
  was read in, which it carries as `t.rep.group`; `t.counts` is its
  symmetry vector, `t.content` the coreps point by point;
  `t.complete` whether every
  point resolved, and `t.refusals` each point that did not, with its own
  `Refusal`; `t.note` where the labels rest on a choice the hoppings
  leave open.  A `Refusal` in its place is falsy and names what stopped
  the call.

**Refusals**

A refusal handed in as `rep` is returned unchanged.  A point that refuses
is collected in `refusals`.

- **`tables_not_ready`**: the group's tables cannot be read in this cell.

### `format_counts`

```text
format_counts(counts)
```

A content on one line: `{'-GM6': 2, '-GM7': 1}` as `2×-GM6 + -GM7`.

**Parameters**

- **`counts`** (`mapping`): `{corep: multiplicity}`, the multiplicities nonnegative integers.

**Returns**

- `str`: The labels joined by `" + "`, each after its multiplicity and `×`
  where that is above one.

### `multiplets`

```text
multiplets(energies, degeneracy_tol=1e-08)
```

Which bands of a spectrum are one degenerate level, at a stated energy tolerance.

**Parameters**

- **`energies`** (`array_like, shape (nstate,)`): Ascending energies, as `solve_ham` and `WFArray.energies` give them.

- **`degeneracy_tol`** (`float, optional`): The energy, in the units of the spectrum, within which bands are one
  level; every two bands of a level lie within it of each other.  Default
  `1e-8`.

**Returns**

- `tuple of tuple of int`: The bands of each level, lowest first.

## Atomic limits

- [`decompose`](#decompose) — Every sum of elementary band representations, with nonnegative integer multiplicities, that carries a band set's content.
- [`subtract`](#subtract) — The content of the bands in one set and not in another: `a − b` as a symmetry vector.
- [`completeness_check`](#completeness_check) — Whether a band set's content is a sum of elementary band representations, and how many bands each sum holds.
- [`regular_completion`](#regular_completion) — The bands that would carry a band set up to whole copies of the regular band representation, and the orbitals they would be.
- [`ebr_rows`](#ebr_rows) — Each elementary band representation of the group as a symmetry vector: the coreps it carries at the high-symmetry points.
- [`in_plane_restriction`](#in_plane_restriction) — Which coreps of the group's elementary band representations a layer measures, and whether each of the others repeats a measured one.

### `decompose`

```text
decompose(
    content, group=None, *, rep=None, points=None, allow_incomplete=False, cap=200,
    signed_combinations=True, bands=None
)
```

Every sum of elementary band representations, with nonnegative integer multiplicities, that carries a band set's content.

**Parameters**

- **`content`** (`HSPTable or mapping`): The band set's content: a table [`hsp_table`](#hsp_table) returned,
  which must be complete, or `{corep: multiplicity}` written out.

- **`group`** (`GroupData, optional`): The group the content was read in, `r.group`.  A table carries its
  own, `t.rep.group`, and is read in it; a content written out by hand
  takes the group beside it.

- **`rep`** (`SpaceGroupRep, optional`): The representation the content was read with, `r.rep`: it says how
  each site irrep is known, and on a layer which Wyckoff positions lie
  within it.  A table carries its own, `t.rep`; beside a content
  written out by hand, pass it whenever it is at hand.

- **`bands`** (`int or sequence of int, optional`): Which bands the content counts: `n` the lowest `n`, or the band
  indices.  Default: a table's own.

- **`points`** (`sequence of str, optional`): The high-symmetry points the model samples.  Default: a table's own, or
  every tabulated point in the model's plane.

- **`allow_incomplete`** (`bool, optional`): Decompose a table that refused at some points, leaving their coreps
  out: for a point whose tables cannot be read, never one where the
  physics refused.  The verdict is then a necessary condition only.
  Default `False`.

- **`cap`** (`int, optional`): The most sums listed.  Default 200.

- **`signed_combinations`** (`bool, optional`): For a `"none"`, search again with negative multiplicities allowed: a
  combination found is the signature of a fragile band set.  Default
  `True`.

**Returns**

- `EBRDecomposition or Refusal`: `d.summary()` prints the verdict and each sum.  `d.verdict` is
  `"symmetry-indicated decomposable"` or `"none"`;
  `d.solution_strings()` the sums and `d.positions` their Wyckoff
  positions, which no labelling of the orbitals moves; `d.certificate`
  what a `"none"` rests on, and `d.signed_combinations` the search
  with negative multiplicities; `d.caveat` what a sum does not certify.
  The record is truthy, a `"none"` included; a `Refusal` in its place
  is falsy and names what stopped the call.

**Refusals**

- **`incomplete_content`**: the table refused at some point, and `allow_incomplete` is not set.

- **`not_a_band_structure`**: the content fails the group's compatibility relations: no set of bands
  apart from the rest at every k has it.

- **`unknown_label`**: a corep the group's elementary band representations do not hold.

- **`restriction_not_exact`**: a corep the model does not measure repeats none it does.

- **`non_integer_multiplicity` / `negative_multiplicity`**: the content is no count of coreps.

- **`ebr_tables` / `irrep_tables` / `ebr_convention`**: no elementary band representations of this group to decompose against.

### `subtract`

```text
subtract(content_a, content_b)
```

The content of the bands in one set and not in another: `a − b` as a symmetry vector.

**Parameters**

- **`content_a`, `content_b`** (`HSPTable or mapping`): The two contents; `b` is contained in `a`.

**Returns**

- `dict or Refusal`: `{corep: multiplicity}`, the zeros dropped: empty, and so falsy, where
  the two are equal, so `isinstance(out, Refusal)` is what tells a
  refusal apart.

**Refusals**

- **`negative_multiplicity`**: `b` is not contained in `a`: a difference of two band sets is one
  only where one contains the other.  A signed combination is what
  `EBRDecomposition.signed_combinations` holds.

- **`non_integer_multiplicity`**: either content is no count of coreps.

### `completeness_check`

```text
completeness_check(content, group=None, *, bands=None, **kwargs)
```

Whether a band set's content is a sum of elementary band representations, and how many bands each sum holds.

**Parameters**

- **`content`** (`HSPTable or mapping`): The band set's content, as [`decompose`](#decompose) takes it.

- **`group`** (`GroupData, optional`): The group the content was read in, `r.group`, as [`decompose`](#decompose)
  takes it: a table carries its own.

- **`bands`** (`int or sequence of int, optional`): The set of bands, as [`decompose`](#decompose) takes it: the bands each sum is
  held against.

- **`**kwargs`**: Passed to [`decompose`](#decompose): `rep=`, `allow_incomplete=`, ...

**Returns**

- `CompletenessReport or Refusal`: `c.summary()` prints it.  `c.complete` is whether a sum carries the
  content, `c.band_counts` the bands each sum holds, and
  `c.band_count_matches` whether those are the bands asked for;
  `c.decomposition` holds the sums.  A `Refusal` in its place is
  falsy and names what stopped the call.

**Refusals**

- **`incomplete_content`, `not_a_band_structure`, `unknown_label`, `restriction_not_exact`, ...**: those of [`decompose`](#decompose).

### `regular_completion`

```text
regular_completion(content, group=None, *, bands=None, rep=None, points=None, **kwargs)
```

The bands that would carry a band set up to whole copies of the regular band representation, and the orbitals they would be.

**Parameters**

- **`content`** (`HSPTable or mapping`): The band set's content.

- **`group`** (`GroupData, optional`): The group the content was read in, `r.group`, as [`decompose`](#decompose)
  takes it: a table carries its own.

- **`bands`** (`int or sequence of int`): The bands of the completed system: a number of them, or their indices,
  a multiple of `|U|` in all.  Required.

- **`rep`** (`SpaceGroupRep, optional`): The representation, `r.rep`, which [`decompose`](#decompose) reads; a table
  carries its own.

- **`points`** (`sequence of str, optional`): The high-symmetry points the model samples.

- **`**kwargs`**: Passed to [`decompose`](#decompose).

**Returns**

- `RegularCompletion or Refusal`: `rc.summary()` prints it.  `rc.padding` holds the coreps the
  missing bands carry, `rc.decomposition` the orbitals they would be,
  `rc.regular` one copy's content and `rc.bands_per_copy` its bands.
  A `Refusal` in its place is falsy and names what stopped the call.

**Refusals**

- **`not_a_regular_multiple`**: the number of bands is no multiple of `|U|`.

- **`negative_multiplicity`**: the band set carries more of a corep than that many regular copies do:
  it is no part of them.

- **`regular_content_not_integral`, `not_a_band_structure`, `unknown_label`, ...**: those of `regular_content` and [`decompose`](#decompose).

### `ebr_rows`

```text
ebr_rows(group)
```

Each elementary band representation of the group as a symmetry vector: the coreps it carries at the high-symmetry points.

**Parameters**

- **`group`** (`GroupData`): The group, `r.group`.

**Returns**

- `dict or Refusal`: `{key: {corep: multiplicity}}`, the zeros dropped, each key an
  elementary band representation at its Wyckoff position,
  `"-E1↑G(2)@1a(6mm1',6mm)"`.  A `Refusal` in its place is falsy and
  names what stopped the call.

**Refusals**

- **`ebr_tables` / `irrep_tables` / `ebr_convention`**: no elementary band representations of this group in the tables
  installed, or none that can be read in this package's convention.

### `in_plane_restriction`

```text
in_plane_restriction(group, *, points=None, require_exact=True)
```

Which coreps of the group's elementary band representations a layer measures, and whether each of the others repeats a measured one.

**Parameters**

- **`group`** (`GroupData`): The group, `r.group`.

- **`points`** (`sequence of str, optional`): The points the model samples.  Default: every tabulated point in its
  plane.

- **`require_exact`** (`bool, optional`): Refuse where a corep left out repeats none measured; `False` returns
  those as `Restriction.orphans`.  Default `True`.

**Returns**

- `Restriction or Refusal`: `r.summary()` prints it; `r.copies` pairs each corep left out with
  the measured ones it repeats, and `r.exact` says whether each does.
  A `Refusal` in its place is falsy and names what stopped the call.

**Refusals**

- **`restriction_not_exact`**: a corep left out repeats none measured, with `require_exact`.

- **`ebr_tables` / `irrep_tables` / `ebr_convention` / `tables_not_ready`**: no elementary band representations of this group, or no tables in this
  cell.

## Stable invariants

- [`symmetry_indicator`](#symmetry_indicator) — The symmetry-indicator class of a band set's content: the class of its symmetry vector among the band structures modulo the atomic limits.
- [`stack_presentation`](#stack_presentation) — The layer repeated along its normal with no hopping between the copies: a crystal with the layer's bands at every `k_z`.
- [`srsi_tables`](#srsi_tables) — The stable real-space invariants of a grey space group, as the paper tabulates them.
- [`srsi_of`](#srsi_of) — `(theta_Z, theta_Zn)`: the stable real-space invariants of a configuration `p` over the site keys.
- [`srsi_of_columns`](#srsi_of_columns) — `(theta_Z, theta_Zn)` of a configuration `p` over the group's elementary band representations.
- [`srsi_of_bands`](#srsi_of_bands) — The $`\mathbb{Z}`$-valued stable real-space invariants of a band set, from its symmetry vector.
- [`srsi_of_model`](#srsi_of_model) — A model's stable real-space invariants, read off its orbitals.
- [`splitting`](#splitting) — Whether the bands asked for and the rest of a model's bands can both be atomic insulators.
- [`nsit_criterion`](#nsit_criterion) — The paper's no-go criterion on a gapped splitting whose invariants are in hand.
- [`atomic_realizations`](#atomic_realizations) — Every atomic insulator with a band set's symmetry data, each a set of orbitals at maximal Wyckoff positions.
- [`stably_equivalent_pairs`](#stably_equivalent_pairs) — Site irreps at one maximal Wyckoff position that no stable real-space invariant distinguishes.

### `symmetry_indicator`

```text
symmetry_indicator(
    content, group=None, *, rep=None, points=None, bands=None, verify=True,
    stack_period=8.0
)
```

The symmetry-indicator class of a band set's content: the class of its symmetry vector among the band structures modulo the atomic limits.

**Parameters**

- **`content`** (`HSPTable or mapping`): The band set's content, as [`decompose`](#decompose) takes it; a table must be
  complete.

- **`group`** (`GroupData, optional`): The group the content was read in, `r.group`.  A table carries its
  own, `t.rep.group`, and is read in it; a content written out by hand
  takes the group beside it.

- **`rep`** (`SpaceGroupRep, optional`): The representation the content was read with, `r.rep`; a table
  carries its own, `t.rep`.  A layer's coreps at `k_z = ½` that
  repeat none in its plane are measured on its decoupled stack, the
  layer repeated along its normal with no hopping between copies, built
  from the representation and the table.

- **`bands`** (`int or sequence of int, optional`): Which bands the content counts, as [`decompose`](#decompose) takes it.

- **`points`** (`sequence of str, optional`): The high-symmetry points the model samples; given, no stack is built.

- **`verify`** (`bool, optional`): Compute the Smith normal form of the group's matrix of elementary band
  representations again, exactly, and check the tables' against it.
  Default `True`.

- **`stack_period`** (`float, optional`): The vacuum between the stack's copies, in the model's length units;
  nothing returned depends on it.  Default `8.0`.

**Returns**

- `SymmetryIndicator or Refusal`: `si.summary()` prints the class.  `si.verdict` is `"indicated nontrivial"`, `"indicator zero"` or `"trivial indicator group"`;
  `si.structure` the indicator group and `si.values` the class in it,
  one integer modulo each of `si.divisors`, of order `si.order`;
  `si.canonical` and `si.irreptables_named` the class in named
  indicators, as the paper that tabulates the stable real-space
  invariants and as `irreptables` name them; `si.completion` how a
  layer's coreps out of its plane were measured.  The record is truthy
  whatever its verdict; a `Refusal` in its place is falsy and names
  what stopped the call.

**Refusals**

- **`incomplete_content`**: the table refused at some point; there is no override.

- **`indicator_not_evaluable`**: a layer's coreps at `k_z = ½` repeat none in its plane and were not
  measured on its stack: no table, `points=` given, or a check of the
  stack that did not complete (`context["stack"]`).

- **`not_a_band_structure`**: the content fails the group's compatibility relations.

- **`unknown_label`**: a corep the group's elementary band representations do not hold.

- **`non_integer_multiplicity` / `negative_multiplicity`**: the content is no count of coreps.

- **`magnetic_type_out_of_scope`**: a magnetic group of type III or IV, which
  [`realized_group`](#realized_group) never returns.

- **`ebr_tables` / `irrep_tables` / `ebr_convention` / `tables_not_ready`**: no elementary band representations of this group, or no tables in this
  cell.

### `stack_presentation`

```text
stack_presentation(model, *, c=8.0, params=None)
```

The layer repeated along its normal with no hopping between the copies: a crystal with the layer's bands at every `k_z`.

**Parameters**

- **`model`** (`pythtb.TBModel`): A layer: periodic in two directions, in a cell of two or three.

- **`c`** (`float, optional`): The vacuum between the copies, in the model's length units; the period
  is the layer's extent along its normal plus `c`.  Default
  `STACK_C`, `8.0`.

- **`params`** (`dict, optional`): Values for a model whose terms are given by parameter names.

**Returns**

- `pythtb.TBModel or Refusal`: The stack, periodic in three directions.  A `Refusal` in its place is
  falsy.

**Refusals**

- **`not_a_layer`**: The model is not periodic in exactly two directions of a two- or
  three-dimensional cell.

- **`hopping_across_the_stack`**: A hopping runs along the non-periodic direction, so the model is not
  one decoupled layer.

### `srsi_tables`

```text
srsi_tables(group_or_number, *, spinor=None)
```

The stable real-space invariants of a grey space group, as the paper tabulates them.

**Parameters**

- **`group_or_number`** (`GroupData or int`): A grey group a model was measured to have, or a space-group number from
  1 to 230.

- **`spinor`** (`bool, optional`): With a number, and only then: the tables with spin-orbit coupling
  (`True`) or without.

**Returns**

- `SRSITables or Refusal`: `tables.summary()` prints the invariants; `tables.theta(p)` evaluates
  them on a configuration.  A `Refusal` in its place is falsy.

**Refusals**

- **`srsi_scope`**: the group is not grey, or the number is outside 1 to 230.

- **`srsi_table_absent`**: one of the thirteen groups whose one invariant is the band count.

- **`srsi_data`**: the tables are not installed (`srsi_status`).

- **`srsi_ebr_basis`**: the installed `irreptables` lists other elementary band representations
  than the tables were converted against; `context["installed"]` holds its
  list.

- **`ebr_tables`, `irrep_tables`, `ebr_convention`**: from [`ebr_data`](#ebr_data).

### `srsi_of`

```text
srsi_of(group_or_tables, p)
```

`(theta_Z, theta_Zn)`: the stable real-space invariants of a configuration `p` over the site keys.

**Parameters**

- **`group_or_tables`** (`GroupData or SRSITables`): A measured grey group, or its tables.

- **`p`** (`mapping`): `{(site irrep, Wyckoff letter): multiplicity}`: integer for an atomic
  configuration, rational for a band set with a nonzero symmetry indicator.

**Returns**

- `tuple or Refusal`: `theta_Z`, the $`\mathbb{Z}`$-valued invariants as exact
  fractions in table order, and `theta_Zn`, `{n: values}`, the
  $`\mathbb{Z}_n`$-valued ones modulo `n`.  `theta_Zn` belongs to
  this `p`: it is a band set's only for a configuration that is known.  A
  `Refusal` in its place is falsy.

**Refusals**

- **`srsi_data`, `srsi_scope`, `srsi_table_absent`, `srsi_ebr_basis`**: from [`srsi_tables`](#srsi_tables): the group has no tables.

### `srsi_of_columns`

```text
srsi_of_columns(group_or_tables, p)
```

`(theta_Z, theta_Zn)` of a configuration `p` over the group's elementary band representations.

**Parameters**

- **`group_or_tables`** (`GroupData or SRSITables`): As for [`srsi_of`](#srsi_of).

- **`p`** (`sequence`): One multiplicity per entry of `SRSITables.ebrs`.

**Returns**

- `tuple or Refusal`: `(theta_Z, theta_Zn)`, as [`srsi_of`](#srsi_of) returns them.  A `Refusal`
  in its place is falsy.

**Refusals**

- **`srsi_data`, `srsi_scope`, `srsi_table_absent`, `srsi_ebr_basis`**: from [`srsi_tables`](#srsi_tables): the group has no tables.

### `srsi_of_bands`

```text
srsi_of_bands(content, group=None, *, rep=None, points=None, bands=None)
```

The $`\mathbb{Z}`$-valued stable real-space invariants of a band set, from its symmetry vector.

**Parameters**

- **`content`** (`HSPTable or mapping`): The band set's content at the high-symmetry points, as
  [`symmetry_indicator`](#symmetry_indicator) takes it.

- **`group`** (`GroupData, optional`): The grey group the content was read in, `r.group`, with tables.  A
  table carries its own, `t.rep.group`, and is read in it; a content
  written out by hand takes the group beside it.

- **`rep`** (`SpaceGroupRep, optional`): The model's representation, `r.rep`, for [`symmetry_indicator`](#symmetry_indicator),
  which measures a layer's rows out of its plane on its decoupled stack; a
  table carries its own.

- **`points`** (`sequence of str, optional`): The high-symmetry points to read, for [`symmetry_indicator`](#symmetry_indicator).

- **`bands`** (`int or sequence of int, optional`): Which bands the content counts: an `int` is the lowest *n*, a sequence
  those bands.  An `HSPTable` says so itself, and another
  set raises `ValueError`.

**Returns**

- `BandSRSIs or Refusal`: `out.summary()` prints the invariants, whether they are fractional, and
  the symmetry-indicator class, and `out.line()` the same on one line;
  `out.theta_Z`, `out.stable` and `out.indicator` hold them.  A
  `Refusal` in its place is falsy.

**Refusals**

- **`srsi_scope`, `srsi_table_absent`, `srsi_data`, `srsi_ebr_basis`**: from [`srsi_tables`](#srsi_tables): the group has no tables.

- **`incomplete_content`, `not_a_band_structure`, `indicator_not_evaluable`**: from [`symmetry_indicator`](#symmetry_indicator), with the refusals of the content it
  reads.

### `srsi_of_model`

```text
srsi_of_model(realized, *, rep=None)
```

A model's stable real-space invariants, read off its orbitals.

**Parameters**

- **`realized`** (`RealizedGroup`): What [`realized_group`](#realized_group) returned, or its parts
  `(model, group, assignment, rep)` or
  `(model, group, assignment, validated, rep)`.

- **`rep`** (`SpaceGroupRep, optional`): A representation to read in place of the one `realized` holds.

**Returns**

- `ModelSRSIs or Refusal`: `out.summary()` prints the invariants and each orbit's; `out.theta_Z`
  and `out.theta_Zn` hold the model's, `out.orbits` each orbit's.  A
  `Refusal` in its place is falsy.

**Refusals**

- **`srsi_scope`, `srsi_table_absent`, `srsi_data`, `srsi_ebr_basis`**: from [`srsi_tables`](#srsi_tables): the group has no tables.

- **`orbit_content_incomplete`**: an orbit's content could not be read at a high-symmetry point.

- **`orbit_not_localizable`**: an orbit's content is a band representation of no position its orbitals
  can slide onto.

- **`wyckoff_setting`**: the content decomposes at a position the geometry says the orbitals are
  not at: the package disagreeing with itself, a case to report.

- **`tables_not_ready`, `wyckoff_unresolved`, `ebr_tables`**: from [`wyckoff_sites_tables`](#wyckoff_sites_tables) and [`ebr_data`](#ebr_data).

### `splitting`

```text
splitting(realized, *, bands=None, rep=None, points=None)
```

Whether the bands asked for and the rest of a model's bands can both be atomic insulators.

**Parameters**

- **`realized`** (`RealizedGroup`): What [`realized_group`](#realized_group) returned, or the parts
  [`srsi_of_model`](#srsi_of_model) takes.

- **`bands`** (`int or sequence of int`): The bands asked for, `V`: an `int` is the lowest *n*, a sequence
  those bands.  Required.

- **`rep`** (`SpaceGroupRep, optional`): A representation to read in place of the one `realized` holds.

- **`points`** (`sequence of str, optional`): The high-symmetry points to read the content of `V` at.

**Returns**

- `Splitting or Refusal`: The verdict on `V`, the bands asked for, and `C`, the rest of the
  model's bands, `H` being all of them.  `out.summary()` prints it and
  what it rests on: the invariants of `V`, `C` and `H`, and the atomic
  realizations of each side.  `out.verdict` is `"diagnosed topological"`, `"invisible to the SRSIs"`, `"no Z_n SRSIs"` or
  `"stable topological"`, the last where `V` carries a nonzero symmetry
  indicator, `out.indicator`.  The record is truthy.  A `Refusal` in its
  place is falsy.

**Refusals**

- **`srsi_scope`, `srsi_table_absent`, `srsi_data`, `srsi_ebr_basis`**: from [`srsi_tables`](#srsi_tables): the group has no tables.

- **`orbit_content_incomplete`, `orbit_not_localizable`, `wyckoff_setting`**: from [`srsi_of_model`](#srsi_of_model), for the orbitals.

- **`incomplete_content`, `not_a_band_structure`, ...**: from [`hsp_table`](#hsp_table) and [`srsi_of_bands`](#srsi_of_bands), for the bands.

- **`srsi_not_atomic`, `srsi_realizations_truncated`**: from [`nsit_criterion`](#nsit_criterion), which lists each side at the default cap of
  [`atomic_realizations`](#atomic_realizations).

### `nsit_criterion`

```text
nsit_criterion(
    tables, theta_total, theta_Z_v, theta_Z_c, *, n_bands_v=None, n_bands_c=None
)
```

The paper's no-go criterion on a gapped splitting whose invariants are in hand.

**Parameters**

- **`tables`** (`SRSITables`): The group's tables.

- **`theta_total`** (`tuple`): `(theta_Z, theta_Zn)` of `H`, all the bands, as
  `SRSITables.theta` returns them for the model's orbitals.

- **`theta_Z_v`, `theta_Z_c`** (`sequence`): The $`\mathbb{Z}`$-valued invariants of `V` and of `C`, the two
  sides of the gap; a pair that does not add up to those of `H` raises
  `ValueError`.

- **`n_bands_v`, `n_bands_c`** (`int, optional`): The bands on each side, for [`atomic_realizations`](#atomic_realizations).

**Returns**

- `Splitting or Refusal`: `out.summary()` prints the verdict, the realizations of each side and
  the pairs that add up; `out.verdict` holds the verdict.  The record is
  truthy.  A `Refusal` in its place is falsy.

**Refusals**

- **`srsi_not_atomic`, `srsi_realizations_truncated`**: from [`atomic_realizations`](#atomic_realizations), which lists each side at its default
  cap: a side with fractional invariants, or more realizations than the
  cap, which [`atomic_realizations`](#atomic_realizations) with a larger `cap=` lists.

### `atomic_realizations`

```text
atomic_realizations(tables, theta_Z, *, n_bands=None, cap=256)
```

Every atomic insulator with a band set's symmetry data, each a set of orbitals at maximal Wyckoff positions.

**Parameters**

- **`tables`** (`SRSITables`): The group's tables, from [`srsi_tables`](#srsi_tables).

- **`theta_Z`** (`sequence`): The band set's $`\mathbb{Z}`$-valued invariants, one per
  `SRSITables.z_rows`.

- **`n_bands`** (`int, optional`): The band count, which `theta_Z` fixes; another raises
  `ValueError`.

- **`cap`** (`int, optional`): The most realizations to list; more is refused.  Default 256.

**Returns**

- `Realizations or Refusal`: `out.summary()` prints every realization with its own
  $`\mathbb{Z}_n`$-valued invariants; `out.realizations` holds them.
  The record is truthy, and an empty list with integral `theta_Z` is a
  fragile band set.  A `Refusal` in its place is falsy.

**Refusals**

- **`srsi_not_atomic`**: `theta_Z` is fractional: the band set carries a nonzero symmetry
  indicator, and no atomic insulator has its symmetry data.

- **`srsi_realizations_truncated`**: more than `cap` realizations; a larger `cap=` lists them.

### `stably_equivalent_pairs`

```text
stably_equivalent_pairs(group_or_tables)
```

Site irreps at one maximal Wyckoff position that no stable real-space invariant distinguishes.

**Parameters**

- **`group_or_tables`** (`GroupData or SRSITables`): A measured grey group, or its tables.

**Returns**

- `tuple of StablePair or Refusal`: Each pair's `summary()` prints it.  Empty for most groups, and then
  falsy, as a `Refusal` in its place is: `isinstance(out, Refusal)`
  tells the two apart.

**Refusals**

- **`srsi_data`, `srsi_scope`, `srsi_table_absent`, `srsi_ebr_basis`**: from [`srsi_tables`](#srsi_tables): the group has no tables.

## Numerical and `wannier90` models

- [`symmetrize_positions`](#symmetrize_positions) — The model with its orbital centres moved onto the nearest positions its group carries, every band where it was.
- [`symmetrize`](#symmetrize) — The model nearest a near-symmetric one that obeys its group exactly, and how far it moved.
- [`classify_operations`](#classify_operations) — Each operation's class against a declared tolerance: `"exact"`, `"near"` or `"broken"`.

### `symmetrize_positions`

```text
symmetrize_positions(model, group, assignment, *, max_move=1e-05, params=None)
```

The model with its orbital centres moved onto the nearest positions its group carries, every band where it was.

**Parameters**

- **`model`** (`pythtb.TBModel`): The model whose centres are moved.

- **`group`, `assignment`** (`GroupData, OrbitalAssignment`): `r.group` and `r.assignment` of a call on this model.  A group read
  off the noisy centres themselves is written about an origin their noise
  placed: identify the moved model again before reading it in a group.

- **`max_move`** (`float, optional`): How far a centre may move and still be the same orbital, in reduced
  coordinates, a shift of all of them together not counted; and how far
  that shift may reach and still be removed.  Default `1e-5`.

- **`params`** (`dict, optional`): Values for a model whose terms are given by parameter names.

**Returns**

- `(pythtb.TBModel, dict) or Refusal`: The model with its centres moved, and the record of the move:
  `defect_before` and `defect_after`, how far the centres miss the
  group's action before and after, in reduced coordinates;
  `max_move`, the largest move of one centre; `offset`, the shift of
  them all together, each centre returned being the one handed in plus
  its own move plus `offset`; `frame`, `"model"` where they are
  written in the model's own frame and `"fitted"` where they stay about
  the origin fitted to them; `spectrum_residual`, how far the spectrum
  moved over `n_check` random `k`, `0` where no centre moved; and
  `n_moved`.  A `Refusal` in its place is falsy.

**Refusals**

- **`positions_not_nearly_symmetric`**: a centre would have to move further than `max_move`: the positions
  are a different set of centres, and the group they carry is the one to
  identify.

### `symmetrize`

```text
symmetrize(
    model, group, assignment, validated, *, tol, k_grid=None, params=None, max_passes=3,
    verify=True, verify_nk=None, verify_tol=None, max_bytes=1000000000.0,
    characters=None, multiplets=None, spin=None, max_candidates=1048576
)
```

The model nearest a near-symmetric one that obeys its group exactly, and how far it moved.

**Parameters**

- **`model`** (`pythtb.TBModel`): The model the operations were measured on, `r.model`: for a layer,
  its slab presentation.

- **`group`, `assignment`, `validated`** (`GroupData, OrbitalAssignment, ValidatedGroup`): `r.group`, `r.assignment` and `r.validated` of a call measured at
  `tol`, `realized_group(model, tol=..., symmetrize=False)`.

- **`tol`** (`float`): The declared tolerance, in the units of `H`, the one `validated` was
  measured at; for numbers good to a relative `ε`, about
  `10·ε·max|H|`.  Required.

- **`characters`, `multiplets`, `spin`** (`optional`): The declarations, as [`realized_group`](#realized_group) takes them.  The
  average does not depend on `characters=`, which the representation it
  is taken with carries.

- **`k_grid`** (`int, optional`): Divisions of the grid in every periodic direction.  Default: the
  smallest grid on which the hopping list comes back exactly, set by the
  hoppings' range; a coarser one raises.

- **`max_passes`** (`int, optional`): The most averaging passes.  Default 3: one is exact, and another runs
  only while `H` still moves.

- **`verify`, `verify_nk`, `verify_tol`** (`optional`): Measure the operations again on the model returned (default `True`),
  on a mesh of `verify_nk` divisions (default: the one `validated`
  used), at `verify_tol` (default `1e-8`).

- **`max_bytes`** (`float, optional`): The memory the average may take at its peak, in bytes.  Default `1e9`.

- **`max_candidates`** (`int, optional`): The largest search for the representation's discrete choices that is
  run.  Default `2**20`.

- **`params`** (`dict, optional`): Values for a model whose terms are given by parameter names; the model
  returned holds the numbers.

**Returns**

- `(pythtb.TBModel, Symmetrization) or Refusal`: The symmetrized model, and the record of the average: `rec.summary()`;
  `rec.defect`, how far `H` moved, in its units; `rec.near`, the
  operations obeyed only within `tol`; `rec.verification`, the
  operations measured again on the model returned.  A `Refusal` in its
  place is falsy and names what stopped the call.

**Refusals**

- **`symmetrize_broken_operation`**: an operation beyond `tol`, which the model does not obey.

- **`symmetrize_defect_exceeds_tol`**: the average moved `H` further than `tol`.

- **`symmetrize_range_not_closed`**: the average reaches beyond the hoppings' range on the grid; a larger
  `k_grid` runs it.

- **`positions_not_exactly_symmetric`**: the orbital centres miss the operations; [`symmetrize_positions`](#symmetrize_positions),
  or a declared precision, moves them first.

- **`projection_too_large` / `detector_too_large`**: past `max_bytes`; the reason gives the value that runs it.

A refusal of the representation's own build, `rep_not_unique` among them,
is returned as it is.

### `classify_operations`

```text
classify_operations(validated, tol, *, exact_gate=5e-14, anchored=None)
```

Each operation's class against a declared tolerance: `"exact"`, `"near"` or `"broken"`.

**Parameters**

- **`validated`** (`ValidatedGroup`): The operations a call measured to hold, `r.validated`, measured at
  `tol`: `realized_group(model, tol=..., symmetrize=False)`.

- **`tol`** (`float`): The declared tolerance, in the units of `H`: the one `validated`
  was measured at.

- **`exact_gate`** (`float, optional`): Rounding, in the units of `H`: the covariance of the model's own
  representation, which `symmetry.covariance_gate` gives and
  [`symmetrize`](#symmetrize) passes.  Default `EXACT_GATE`, `5e-14`, the
  least it is.

- **`anchored`** (`dict, optional`): `{ind: residual}`, each operation's covariance in the representation
  the projection is built with, `rep.residuals`.  An operation it holds
  is classified at the larger of that and its measured residual, which a
  basis turned inside a site, by less than `tol`, makes near.

**Returns**

- `dict`: `{ind: class}` for each operation `validated` kept.
