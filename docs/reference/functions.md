# Functions

Every function `pythtqc` exports, by task: its signature, what it is for, what
it computes, the arguments it takes, what it returns, and the checks at which it
refuses. The page is written from the docstrings by `scripts/build_reference.py`
and is not edited by hand. A docstring holds more than its entry here, its
examples among it, and `help(tqc.<name>)` prints the whole of one, with
`import pythtqc as tqc`: `help(tqc.realized_group)`, for one.

A call that cannot complete returns a `Refusal` in place of its record: a
falsy record that names the check that stopped it and what was measured there.
Each entry's **Refusals** are the checks at which that call can refuse.

The functions fall into six groups:

- [The group](#the-group) — the space group a Hamiltonian has and which operations it obeys, the orbitals' Wyckoff orbits, and the operations by index and Seitz symbol.
- [Declaring what the hoppings cannot supply](#declaring-what-the-hoppings-cannot-supply) — the spin a spinless model carries in its orbital index.
- [Group data in this cell's coordinates](#group-data-in-this-cells-coordinates) — the tables' k-points, little groups, character tables, Wyckoff positions and elementary band representations, in the model's own cell.
- [Labeling the bands](#labeling-the-bands) — the states at one k-point, and the sewing matrices, characters and coreps of a set of bands, at one high-symmetry point or at all of them.
- [Atomic limits](#atomic-limits) — whether a symmetry data vector is a sum of elementary band representations, and the counting beside it.
- [Stable invariants](#stable-invariants) — the symmetry-indicator class of a band set, and its stable real-space invariants, from the bands or from the orbitals.

## The group

- [`realized_group`](#realized_group) — The space group obeyed by H, obtained by descent from the space group of the lattice of occupied sites.
- [`operations`](#operations) — Every operation of a group by its index, with its Seitz symbol and its name in the model's frame.
- [`operation_index`](#operation_index) — The index of the operation a Seitz symbol names.

### `realized_group`

```text
realized_group(
    model, *, typat=None, environment=None, cross_check=True, decorations=None, nk=6,
    max_bytes=1000000000.0, position_tol=None, precision=None, spinor=None, spin=None,
    symprec=1e-05, params=None, characters=None, site_multiplets=None,
    max_candidates=1048576
)
```

The space group obeyed by H, obtained by descent from the space group of the lattice of occupied sites.

The call also places each orbital on its Wyckoff position and builds the
representation of the group that labels the bands.

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
  precision of its numbers: a positive number.  Centers and hoppings
  symmetric within it are made exactly symmetric; left out, the model is
  read as exact, each operation held to `1e-8` times `max|H|`, the
  largest entry of `H(k)` once the mean on-site energy is subtracted.

- **`spin`** (`str, mapping or SpinBasis, optional`): For a spinless model whose orbital index carries spin, as a wannier90
  import's does: which two orbitals are the spin states of one Wannier
  function, `"interleaved"`, `"blocked"` or pair by pair.  Left out,
  such a model is read spinless.

- **`spinor`** (`bool, optional`): Whether the states carry spin one-half.  Default: from the model, and
  `True` with `spin`.

- **`position_tol`, `symprec`** (`float, optional`): `position_tol`: how far an orbital center may miss the position its
  group puts it at and still be moved onto it, in reduced coordinates,
  for a model whose centers alone are imprecise; left out, it is exact,
  and `precision` sets it.  `symprec`: the symmetry finder's
  tolerance, a Cartesian length, default `1e-5`.

- **`nk`** (`int, optional`): k-points per periodic direction of the grid the operations are measured
  on.  Default 6.

- **`site_multiplets`** (`list or SiteMultiplets, optional`): Each site's multiplets, stated instead of measured: one list per site,
  each multiplet written as its orbitals.

- **`max_candidates`, `max_bytes`** (`int and float, optional`): Limits on the search for the representation (default `2**20`
  candidates) and on the memory of the symmetric projection (default
  `1e9` bytes).

- **`params`** (`dict, optional`): The scalar value of each parameter of the model, keyed by name:
  `{"lam": 0.3}`.

**Returns**

- `RealizedGroup or Refusal`: `r.summary()` prints what was measured, each line with the field
  that holds it.  `r.group` is the group, its operations by Seitz
  symbol in `r.operations`; `r.wyckoff_orbits` the orbitals, a row
  per Wyckoff orbit, with their site irreps and the band representation
  they induce; `r.rep` the matrices that label the bands, read through
  the record itself, `hsp_table(r, bands=n)`.  `r.descent` is what
  was broken; `r.time_reversal`; `r.environment`, for a layer;
  `r.beyond_grey`, operations combined with time reversal that hold
  where time reversal does not; `r.conserved`, symmetry beyond the
  crystal's; `r.spin`, the spin pairs declared; `r.numerical`, how
  the measurement was made: the precision declared, the centers moved,
  `H` projected, the verdict on each operation.  For a parameterized
  model, `r.model` is the model with its parameters evaluated at
  `params`, and `r.numerical.source` the model as handed in.  A
  `Refusal` in its place is falsy and names what stopped the call.

**Refusals**

Once the group is found, a refusal names it and carries it in
`context["group"]`.

- **`fewer_than_two_periodic_directions`**: a chain or a molecule.

- **`irrep_tables` / `single_valued_spinful`**: `irreptables` is not installed, or `spinor=False` was asked of
  states with spin one-half.

- **`spin_pair_split`**: the two orbitals of a declared spin pair sit at different centres.

- **`wyckoff_orbit`**: an orbital sits on no Wyckoff position of the cell.

- **`positions_not_exactly_symmetric`**: centers off symmetric positions, with no `precision` declared, by
  enough to break operations or to keep H from being made exactly
  symmetric.

- **`supercell`**: H obeys a translation shorter than the cell's: the model is written on
  a supercell.

- **`operations_not_nearly_consistent` / `operations_not_consistent`**: the operations found on a numerical cell do not close; `precision`
  makes the centres exact first.

- **`cell_group_not_realized`**: an operation combined with time reversal that held before a numerical
  model's Hamiltonian was projected and is broken after it.

- **`unknown_label` / `ambiguous_multiplet`**: a declared word the site does not name, or a flat list that cannot say
  which orbitals form one multiplet.

- **`rep_not_covariant`**: operations that hold within the declared precision and that no
  representation carries within the exact check.

- **`rep_internal_factor` / `rep_not_multiplicative`**: no candidate representation carries the group's phases, or multiplies
  as the group does.

- **`rep_not_unique`**: more than one representation fits; report it, with the model and the
  call.

- **`rep_double_valued_spinless`**: double-valued tables asked of a spinless model; `spin=` declares its
  spin pairs.

- **`rep_search_too_large` / `projection_too_large` / `detector_too_large`**: past `max_candidates` or `max_bytes`; the reason gives the value
  that runs it.

- **`symmetrize_defect_exceeds_tol` / `symmetrize_range_not_closed`**: the projection of the Hamiltonian onto its symmetric part, under a
  declared precision.

### `operations`

```text
operations(group)
```

Every operation of a group by its index, with its Seitz symbol and its name in the model's frame.

The index is the key every per-operation record is written on:
`rep.matrices`, the verdicts of the measurement, the entries of a
descent.  The Seitz symbol `{R|t}` is the one the tables and the Bilbao
Crystallographic Server print, in their conventional setting, with an
underscore before the axis and a minus sign for an overbar: `-4+_001` is
`4̄⁺₀₀₁`.  The name is the same operation in the model's Cartesian frame,
the frame `characters=` is written in.

**Parameters**

- **`group`** (`GroupData`): The group, `r.group` of what `realized_group` returns.

**Returns**

- `OperationTable or Refusal`: `table[index]` is one `Operation`, with its `seitz`, its
  `name`, its matrices and whether it is antiunitary;
  `print(table)` lists them all.  `r.operations` is the same table.
  A `Refusal` in its place is falsy.

**Refusals**

A refusal handed in as `group` is returned as it is.

- **`tables_not_ready`**: The group has no conventional setting to write a Seitz symbol in.

### `operation_index`

```text
operation_index(group, seitz)
```

The index of the operation a Seitz symbol names.

A rotation written about either direction of its axis, `4+` about
`[001]` or `4-` about `[00-1]`, and any representative of its
translation modulo the lattice name the same operation, so a symbol copied
off a printed table is read whichever choices it makes; spaces and the
underscore before the axis are free.

**Parameters**

- **`group`** (`GroupData`): The group, `r.group` of what `realized_group` returns.

- **`seitz`** (`str`): A Seitz symbol `{R|t}`, with a trailing `'` for an operation
  combined with time reversal.

**Returns**

- `int or Refusal`: The operation's index.  A `Refusal` in its place is falsy.

**Refusals**

A refusal handed in as `group` is returned as it is.

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

Nothing in `H(k)` says which orbital is whose spin partner, so the pairing
is declared, and `realized_group(model, spin=...)` then reads the model as
a native spinful one, each pair a spinor Wannier function at its own centre.

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

It is the operator a spin-resolved calculation projects onto a set of
bands, `P (n̂·σ) P`: eigenvalues of that kept away from zero split the
bands by the sign of their spin along `n̂`, the split the spin Chern
number is read on.  It is the matrix
`spin_resolved_analysis.spin_operator_spinless` builds from the same
pairs, entry for entry.

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

Each orbital becomes two at its own position, the up and down states of
one pair, and the two models' Hamiltonians are measured to be the same.
It is a trusted model to hold a real import against.

**Parameters**

- **`model`** (`pythtb.TBModel`): A spinful model.

- **`order`** (`{"interleaved", "blocked"}, optional`): `"interleaved"` (default), `(0↑, 0↓, 1↑, 1↓, …)`, pythtb's own order
  of the states, or `"blocked"`, `(0↑, 1↑, …, 0↓, 1↓, …)`.

- **`pairs`** (`sequence of (int, int), optional`): The pairs one by one, up first, every orbital in exactly one, in place
  of `order`.

- **`params`** (`dict, optional`): The scalar value of each parameter of the model, keyed by name:
  `{"lam": 0.3}`.

**Returns**

- `(pythtb.TBModel, SpinBasis) or Refusal`: The spinless model, its parameters evaluated at `params`, and its
  pairing, which `realized_group(flat, spin=sb)` reads as the native
  model's spin.  A `Refusal` in its place is falsy.

**Refusals**

- **`not_a_spinful_model`**: a spinless model, which is in this presentation already;
  [`spin_basis`](#spin_basis) declares its pairs.

## Group data in this cell's coordinates

- [`high_symmetry_points`](#high_symmetry_points) — The tabulated high-symmetry points, in the model's own reduced coordinates.
- [`little_group`](#little_group) — The operations that leave `k` invariant up to a reciprocal lattice vector.
- [`corep_table`](#corep_table) — The character table of the little group at a high-symmetry point: each corep's character on each unitary operation.
- [`wyckoff_sites_tables`](#wyckoff_sites_tables) — The Wyckoff position of each site in the tables' setting, the one every table is read with.
- [`k_grid`](#k_grid) — A Γ-centred mesh that respects the group, with the high-symmetry points it holds.
- [`ebr_data`](#ebr_data) — Every elementary band representation of the group: the bands each site irrep at each Wyckoff position induces.
- [`tables_status`](#tables_status) — Whether irrep's character tables are in place, and `ortools` installed.

### `high_symmetry_points`

```text
high_symmetry_points(group)
```

The tabulated high-symmetry points, in the model's own reduced coordinates.

The points a symmetry data vector is read at.  The tables write them in their
conventional cell, generally as other numbers: graphene's `K` is
`(1/3, 1/3, 0)` there and `(1/3, 2/3)` in pythtb's honeycomb cell.
Without time reversal a point and its partner `−k` are two points
wherever no operation carries one onto the other, `KA` beside `K`, and
both are returned, the partner's irreps the complex conjugates of the
point's.  The points are listed Γ first, then alphabetically by name, a
partner right after its point.

**Parameters**

- **`group`** (`GroupData`): The group, `r.group` of what `realized_group` returns.

**Returns**

- `HighSymmetryPoints or Refusal`: `{name: k}`, each `k` with `dim_k` components; its
  `out_of_plane` names the tabulated points out of a layer's plane.
  `summary()` prints it.  A `Refusal` in its place is falsy.

**Refusals**

A refusal handed in as `group` is returned as it is.

- **`tables_not_ready`**: The group has no transformation to the tables' conventional cell, to
  carry their points into the model's coordinates.

### `little_group`

```text
little_group(group, k)
```

The operations that leave `k` invariant up to a reciprocal lattice vector.

The group a band's corep at `k` is a corep of: only these operations act
within the states there.

**Parameters**

- **`group`** (`GroupData`): The group, `r.group` of what `realized_group` returns.

- **`k`** (`str or array, shape (dim_k,)`): The point: a tabulated name, `"X"`, one of those
  [`high_symmetry_points`](#high_symmetry_points) returns, or its coordinates in the
  model's reduced coordinates.  A layer takes a point of its
  three-dimensional cell too, where it lies in the plane.

**Returns**

- `LittleGroup or Refusal`: Its operations, the indices of its unitary operations in index order,
  the columns a character vector at `k` is written in, and those of
  its operations combined with time reversal, which carry no
  characters; `summary()` prints it.  A `Refusal` in its place is
  falsy.

**Refusals**

A refusal handed in as `group` is returned as it is.

- **`out_of_plane`**: A point with a component along a layer's vacuum, a direction its zone
  does not have, or the name of a tabulated point out of the layer's
  plane.

- **`tables_not_ready`**: `k` is a name, and the group has no transformation to the tables'
  conventional cell to carry the tabulated points into the model's.

### `corep_table`

```text
corep_table(group, k)
```

The character table of the little group at a high-symmetry point: each corep's character on each unitary operation.

The table measured characters are read against, in the package's
convention for a lattice translation, so characters measured on the model
compare with it directly; the points where that convention moves a label,
or changes how time reversal glues the irreps, are named in
`ebr_data(group).convention`.  Operations combined with time reversal
carry no characters.

**Parameters**

- **`group`** (`GroupData`): The group, `r.group` of what `realized_group` returns.

- **`k`** (`str or array, shape (dim_k,)`): A tabulated point: its name, `"X"`, or its coordinates in the
  model's reduced coordinates, as [`high_symmetry_points`](#high_symmetry_points) gives
  them.

**Returns**

- `dict or Refusal`: `{corep: {index: character}}`, one entry per corep of the little
  group, its character on each unitary operation by index.  A
  `Refusal` in its place is falsy.

**Refusals**

A refusal handed in as `group` is returned as it is.

- **`tables_not_ready`**: The group has no transformation to the tables' conventional cell.

- **`out_of_plane`**: `k` has a component along a layer's vacuum, or names a tabulated
  point out of the layer's plane.

- **`kpoint_not_tabulated`**: `k` is no tabulated point in the model's coordinates, or not the
  tabulated member of its star; the refusal lists the tabulated points.

### `wyckoff_sites_tables`

```text
wyckoff_sites_tables(group, positions=None, *, position_tol=0.0001)
```

The Wyckoff position of each site in the tables' setting, the one every table is read with.

A letter is a table key, and settings disagree about it: the Fu–Kane–Mele
diamond's orbit is `8a` here and `8b` in spglib's setting, and a site
irrep at two positions can carry the same content at every k-point and
different real-space invariants.  The orbitals' own positions are the rows'
`wyckoff` of `r.wyckoff_orbits`.

**Parameters**

- **`group`** (`GroupData`): The group, `r.group` of what `realized_group` returns.  A grey
  group's positions are those of its unitary subgroup.

- **`positions`** (`array, shape (n, 3), optional`): Points in reduced coordinates of the group's three-dimensional cell.
  Default: the points the group was found on, the model's sites and any
  species added to its cell.

- **`position_tol`** (`float, default 1e-4`): How far a point may sit from a position and lie on it, in reduced
  coordinates of the tables' conventional cell.

**Returns**

- `tuple of SiteReference or Refusal`: One per point: its Wyckoff position, `"8a"`, its free parameters,
  and the positions an orbital confined there can be moved to.  A
  `Refusal` in its place is falsy.

**Refusals**

A refusal handed in as `group` is returned as it is.

- **`tables_not_ready`**: The group has no transformation to the tables' conventional cell.

- **`wyckoff_setting`**: `pyxtal`'s list of the group's positions is not written in the
  tables' setting, so no position would be read in it.

- **`wyckoff_unresolved`**: A point lies on no position within `position_tol`; the refusal names
  the nearest and its distance.

- **`ebr_tables`, `irrep_tables`, `ebr_convention`**: From [`ebr_data`](#ebr_data), whose positions the sites are checked
  against.

### `k_grid`

```text
k_grid(group, n, *, allow_asymmetric=False)
```

A Γ-centred mesh that respects the group, with the high-symmetry points it holds.

A mesh holds only the high-symmetry points it happens to contain: `Γ`
always, and the zone boundary only with an even number of divisions.  A
label read off a mesh at a point it misses is no label, so `contains`
names the points it holds, and [`high_symmetry_points`](#high_symmetry_points) gives every
tabulated point's coordinates.  The mesh is pythtb's own, `k` in
`[0, 1)`, periodic in the gauge a `WFArray` reads.

**Parameters**

- **`group`** (`GroupData`): The group, `r.group` of what `realized_group` returns.

- **`n`** (`int or sequence of int`): The divisions along each periodic direction, or one number for all.

- **`allow_asymmetric`** (`bool, default False`): Whether a mesh the group does not carry onto itself is returned.
  With `allow_asymmetric=True`, it is returned with no irreducible
  points.

**Returns**

- `KGrid or Refusal`: The mesh, ready for a `pythtb.WFArray`, with the tabulated points it
  holds; `summary()` prints it.  A `Refusal` in its place is falsy.

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

The atomic limits the group has, and the columns a symmetry data vector is
decomposed over; [`decompose`](#decompose) and
[`symmetry_indicator`](#symmetry_indicator) read them, a layer's restricted to the
points of its plane.  The rows are in the package's convention for a
lattice translation, `e^{-ik·t}`, where the tables' own pages write
`e^{+ik·t}`: away from the time-reversal-invariant momenta a row pairs
labels with Wyckoff positions otherwise than the Bilbao Crystallographic
Server's page does, and `convention` says where.

**Parameters**

- **`group`** (`GroupData`): The group, `r.group` of what `realized_group` returns.

**Returns**

- `EBRData or Refusal`: The columns as terms, `.ebrs`; the rows' coreps, `.corep_names`;
  the EBR matrix and its Smith normal form, `.matrix` and `.smith`;
  what the convention moves, `.convention`; and the group they were
  read for, `.group`.  A `Refusal` in its place is falsy.

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

Without the `irreptables` package, or with it imported before `irrep`
in the same process, `irrep` binds a stand-in that raises nothing and
names no label, and every call that reads the tables refuses
`irrep_tables`.

**Returns**

- `dict`: `tables`: whether the real tables are in place, and `reason`, where
  they are not, which of the two causes it is and what to do;
  `ortools`: whether the integer solver a decomposition needs is
  installed; `irrep_version` and `irreptables_version`: the versions
  installed, `None` where a package is absent.

## Labeling the bands

- [`solve_at`](#solve_at) — The energies and eigenvectors at one k, from one diagonalization.
- [`sewing_matrix`](#sewing_matrix) — The matrix of one operation on a set of bands, at a k-point the operation leaves invariant: the sewing matrix `B(g,k)_mn = <u_m(k)|U(g;k)|u_n(k)>`.
- [`characters`](#characters) — The characters of a set of bands at one k: the trace of each unitary operation of the little group over each degenerate multiplet of the set.
- [`coreps_at`](#coreps_at) — The coreps a set of bands carries at one high-symmetry point, multiplet by multiplet.
- [`hsp_table`](#hsp_table) — The coreps a set of bands carries at every high-symmetry point of the group: its symmetry data vector.
- [`format_coreps`](#format_coreps) — Corep multiplicities on one line: `{'-GM6': 2, '-GM7': 1}` as `2x-GM6 + -GM7`.
- [`multiplets`](#multiplets) — Which bands of a spectrum are one degenerate level, at a stated energy tolerance.

### `solve_at`

```text
solve_at(model, k, *, params=None)
```

The energies and eigenvectors at one k, from one diagonalization.

Hand the pair to [`sewing_matrix`](#sewing_matrix), [`characters`](#characters) or
[`coreps_at`](#coreps_at) as `states=` to read several sets of bands at that
k off the same states.  A second diagonalization chooses another basis
inside each degenerate level, and a character read across the two is wrong.

**Parameters**

- **`model`** (`pythtb.TBModel`): The model, `r.model`.

- **`k`** (`array_like, shape (dim_k,)`): One k-point, in reduced coordinates.

- **`params`** (`dict, optional`): The scalar value of each parameter of the model, keyed by name:
  `{"lam": 0.3}`.

**Returns**

- `(ndarray, ndarray)`: `(evals, evecs)`: the energies, ascending, and the eigenvectors as
  rows, as `model.solve_ham` returns them with `return_eigvecs=True`
  and `flatten_spin_axis=True`, for the model with its parameters
  evaluated at `params`.

### `sewing_matrix`

```text
sewing_matrix(
    realized, operation=None, k=None, *, states=None, bands=None, degeneracy_tol=None
)
```

The matrix of one operation on a set of bands, at a k-point the operation leaves invariant: the sewing matrix `B(g,k)_mn = <u_m(k)|U(g;k)|u_n(k)>`.

Its trace over a degenerate multiplet is that multiplet's character, and its
eigenvalues are the operation's on those bands.  The states are checked to
solve `H(k)`, and the bands to hold whole degenerate multiplets, wherever
the states carry their energies.

**Parameters**

- **`realized`** (`RealizedGroup`): What [`realized_group`](#realized_group) returned, `r`.

- **`operation`** (`int`): The operation, by its index: `r.operations` lists them, and
  [`operation_index`](#operation_index) finds one by its Seitz symbol.
  Required.

- **`k`** (`str or array_like, shape (dim_k,)`): A k-point the operation leaves invariant up to a reciprocal lattice
  vector: its tabulated name, `"M"`, or its reduced coordinates.
  Required.

- **`bands`** (`int or sequence of int, optional`): The set of bands: `n` the lowest `n` bands, or the band indices.
  Default: every state.

- **`states`** (`None, pythtb.WFArray, (evals, evecs) or ndarray, optional`): The states at `k`: `None` (default) solves the model there; a
  solved `WFArray`, or the pair [`solve_at`](#solve_at) returns, reuses one
  diagonalization.  Eigenvector rows alone, or a `WFArray` of a sector
  with fewer states than the model, carry no energies.

- **`degeneracy_tol`** (`float, optional`): A number in the units of `H`: bands closer in energy than it are one
  multiplet.  Left out, `1e-8` times `max|H|`, the largest entry of
  `H(k)` once the mean on-site energy is subtracted.

**Returns**

- `SewingMatrix or Refusal`: `B.summary()` prints it.  `B.matrix` is the matrix and
  `B.eigenvalues` its eigenvalues; `B.numerical` holds how far it is
  from unitary, the operator from commuting with `H(k)`, and the states
  from eigenstates.  For states without energies, whether they solve
  `H(k)` and whether the bands close their multiplets is not measured.
  A `Refusal` in its place is falsy and names what stopped the call.

**Refusals**

- **`little_group`**: the operation does not leave `k` invariant.

- **`multiplet_split`**: `bands` cuts a degenerate multiplet at `k`.

- **`states_not_at_k`**: the states handed in do not solve `H(k)`.

- **`k_not_on_mesh`**: the mesh handed in does not hold `k`.

- **`out_of_plane`**: `k` has a component along a layer's vacuum, or names a tabulated
  point out of the layer's plane.

- **`tables_not_ready`**: `k` is a name, and the group has no transformation to the tables'
  conventional cell.

### `characters`

```text
characters(realized, k, *, states=None, bands=None, degeneracy_tol=None)
```

The characters of a set of bands at one k: the trace of each unitary operation of the little group over each degenerate multiplet of the set.

**Parameters**

- **`realized`** (`RealizedGroup`): What [`realized_group`](#realized_group) returned, `r`.

- **`k`** (`str or array_like, shape (dim_k,)`): A k-point: its tabulated name, `"GM"`, or its reduced coordinates.
  Only a tabulated point is labeled, by [`coreps_at`](#coreps_at).

- **`bands`** (`int or sequence of int`): The set of bands: `n` the lowest `n` bands, or the band indices.
  Required.

- **`states`** (`None, pythtb.WFArray or (evals, evecs), optional`): The states at `k`: `None` (default) solves the model there; a
  solved `WFArray`, or the pair [`solve_at`](#solve_at) returns, reuses one
  diagonalization.  They carry energies, the rows being the multiplets.

- **`degeneracy_tol`** (`float, optional`): A number in the units of `H`: bands closer in energy than it are one
  multiplet.  Left out, `1e-8` times `max|H|`, the largest entry of
  `H(k)` once the mean on-site energy is subtracted.

**Returns**

- `BandCharacters or Refusal`: `ch.summary()` prints them.  `ch.multiplets` holds a row for each
  degenerate multiplet: its bands, its mean energy and its characters by
  operation index.  `ch.operations` names the operations of the little
  group; `ch.numerical` holds how closely the characters are measured.
  Characters can only be returned if the set of bands contains all the
  states of each degenerate multiplet at `k`.  A `Refusal` in its
  place is falsy and names what stopped the call.

**Refusals**

A refusal handed in as `realized` is returned unchanged, and one from
[`little_group`](#little_group) as it is.

- **`multiplet_split`**: the set of bands cuts a degenerate multiplet at `k`.

- **`states_not_at_k`**: the states handed in do not solve `H(k)`.

- **`k_not_on_mesh`**: the mesh handed in does not hold `k`.

- **`out_of_plane`**: `k` has a component along a layer's vacuum, or names a tabulated
  point out of the layer's plane.

- **`tables_not_ready`**: `k` is a name, and the group has no transformation to the tables'
  conventional cell.

### `coreps_at`

```text
coreps_at(realized, k, *, states=None, bands=None, degeneracy_tol=None, label_tol=0.001)
```

The coreps a set of bands carries at one high-symmetry point, multiplet by multiplet.

The characters of each degenerate multiplet are measured on the bands'
eigenvectors and expanded over the coreps of the little group.

**Parameters**

- **`realized`** (`RealizedGroup`): What [`realized_group`](#realized_group) returned, `r`.

- **`k`** (`str or array_like`): A tabulated high-symmetry point: its name, `"GM"`, one of those
  [`high_symmetry_points`](#high_symmetry_points) returns, or its coordinates in
  the model's reduced cell.

- **`bands`** (`int or sequence of int`): The set of bands: `n` the lowest `n` bands, or the band indices.
  Required.

- **`states`** (`None, pythtb.WFArray or (evals, evecs), optional`): The states at `k`: `None` (default) solves the model there; a
  solved `WFArray`, or the pair [`solve_at`](#solve_at) returns, reuses one
  diagonalization.

- **`degeneracy_tol`** (`float, optional`): A number in the units of `H`: bands closer in energy than it are one
  multiplet.  Left out, `1e-8` times `max|H|`, the largest entry of
  `H(k)` once the mean on-site energy is subtracted.

- **`label_tol`** (`float, default 1e-3`): How far a fitted multiplicity may sit from an integer, dimensionless.

**Returns**

- `CorepsAt or Refusal`: `c.summary()` prints the coreps multiplet by multiplet.  `c.coreps`
  holds the corep multiplicities of the set, `{corep: multiplicity}`;
  `c.multiplets` a row for each degenerate multiplet, lowest first,
  with its bands, its mean energy and the coreps it carries;
  `c.isolation` the gap from these bands to the nearest other band;
  `c.numerical` the thresholds and the identification the coreps were
  read with.  Corep labels can only be returned if the set of bands
  contains all the states of each degenerate multiplet at `k`.  A
  `Refusal` in its place is falsy and names what stopped the call.

**Refusals**

A refusal handed in as `realized` is returned unchanged.

- **`multiplet_split`**: the set of bands cuts a degenerate multiplet at `k`: label the
  multiplets whole.

- **`kpoint_not_tabulated`**: `k` is not a tabulated point in this cell, or not the tabulated member
  of its star.

- **`out_of_plane`**: `k` has a component along a layer's vacuum, or names a tabulated
  point out of the layer's plane.

- **`integer_multiplicity` / `multiplet_dimension`**: the characters fit no whole number of coreps, or coreps that do not fill
  their multiplet: most often a multiplet grouped wrongly.

- **`states_not_at_k` / `k_not_on_mesh`**: states handed in from another point, or a mesh that does not hold `k`.

- **`little_group_mismatch` / `tables_not_ready`**: the tables do not apply in this cell.

### `hsp_table`

```text
hsp_table(realized, *, states=None, bands=None, degeneracy_tol=None, label_tol=0.001)
```

The coreps a set of bands carries at every high-symmetry point of the group: its symmetry data vector.

**Parameters**

- **`realized`** (`RealizedGroup`): What [`realized_group`](#realized_group) returned, `r`.

- **`bands`** (`int or sequence of int`): The set of bands: `n` the lowest `n` bands, or the band indices.
  Required.

- **`states`** (`None or pythtb.WFArray, optional`): `None` (default) solves the model at the points, once; a solved
  `WFArray` is read as it stands, and a point its mesh does not hold is
  refused there.

- **`degeneracy_tol`** (`float, optional`): A number in the units of `H`: bands closer in energy than it are one
  multiplet.  Left out, `1e-8` times `max|H|`, the largest entry of
  `H(k)` once the mean on-site energy is subtracted.

- **`label_tol`** (`float, default 1e-3`): How far a fitted multiplicity may sit from an integer at each point,
  dimensionless.

**Returns**

- `HSPTable or Refusal`: `t.summary()` prints the coreps at each point.  `t.symmetry_data`
  is the symmetry data vector, `t.multiplets` a row for each
  degenerate multiplet at each point, with the coreps it carries;
  `t.complete` says whether every point was measured, and
  `t.refusals` holds each point that was not, with its own
  `Refusal`; `t.note` says where the labels rest on a choice the
  hoppings leave open.  Corep labels can only be returned at a point if
  the set of bands contains all the states of each degenerate multiplet
  there.  The table is what [`decompose`](#decompose),
  [`symmetry_indicator`](#symmetry_indicator) and [`srsi_of_bands`](#srsi_of_bands)
  take, `decompose(t)`, in the group it was read in, `t.group`.  A
  `Refusal` in its place is falsy and names what stopped the call.

**Refusals**

A refusal handed in as `realized` is returned unchanged.  A point that refuses
is collected in `refusals`.

- **`tables_not_ready`**: the group's tables cannot be read in this cell.

### `format_coreps`

```text
format_coreps(coreps)
```

Corep multiplicities on one line: `{'-GM6': 2, '-GM7': 1}` as `2x-GM6 + -GM7`.

Every multiplicity the package prints reads this way: the coreps of a set
of bands at one point (`CorepsAt.coreps`) or of one multiplet
(`Multiplet.label`), a symmetry data vector
(`HSPTable.symmetry_data`), an elementary band representation's
coreps ([`ebr_rows`](#ebr_rows)).  The coreps are written point by point,
Γ first and the other points alphabetically, and at each point in
ascending order of the irreps' numbers, `GM2` before `GM10`, so one
set of multiplicities always prints one line.

**Parameters**

- **`coreps`** (`mapping`): `{corep: multiplicity}`, the multiplicities nonnegative integers.

**Returns**

- `str`: The labels joined by `" + "`, each after its multiplicity and `x`
  where that is above one.

### `multiplets`

```text
multiplets(energies, degeneracy_tol=1e-08)
```

Which bands of a spectrum are one degenerate level, at a stated energy tolerance.

The levels are the rows of every character table read at that k.  A spread
inside the levels (`multiplet_spread`) far below the gap between them
(`multiplet_separation`) says the tolerance was a safe choice.

**Parameters**

- **`energies`** (`array_like, shape (nstate,)`): Ascending energies, as `solve_ham` and `WFArray.energies` give them.

- **`degeneracy_tol`** (`float, optional`): The energy, in the units of the spectrum, within which bands are one
  level; every two bands of a level lie within it of each other.  Default
  `1e-8`.

**Returns**

- `tuple of tuple of int`: The bands of each level, lowest first.

## Atomic limits

- [`decompose`](#decompose) — The sums of elementary band representations that carry a symmetry data vector.
- [`subtract`](#subtract) — The symmetry data vector of the bands in one set and not in another, `a - b`.
- [`regular_completion`](#regular_completion) — The bands completing a band set to copies of the regular band representation.
- [`ebr_rows`](#ebr_rows) — Each elementary band representation of the group as a symmetry data vector: the coreps it carries at the high-symmetry points.
- [`in_plane_restriction`](#in_plane_restriction) — Which coreps of the group's elementary band representations a layer measures, and whether each of the others repeats a measured one.

### `decompose`

```text
decompose(
    symmetry_data, realized=None, *, allow_incomplete=False, cap=200,
    signed_combinations=True
)
```

The sums of elementary band representations that carry a symmetry data vector.

The sums are found by enumerating the solutions $`x`$ in nonnegative
integers of $`M x = y`$.  Here $`y`$ is the symmetry data vector
and the columns of $`M`$ are the elementary band representations, both
taken on the little-group representations at the high-symmetry points
the model samples.  Each multiplicity is searched up to the
largest entry of $`y`$, the bound.  The symmetry data vector of an
atomic limit is such a sum.  Some band sets whose vector is such a sum are
not atomic limits.

**Parameters**

- **`symmetry_data`** (`HSPTable or mapping`): The symmetry data vector.  Pass the table [`hsp_table`](#hsp_table)
  returns, or `{corep: multiplicity}` written out by hand.

- **`realized`** (`RealizedGroup, optional`): What [`realized_group`](#realized_group) returned for the model the
  symmetry data vector was measured on.  Pass it with a vector written
  out by hand, and nothing with a table.

- **`allow_incomplete`** (`bool, default False`): Whether a table that refused at some points is decomposed.  With
  `allow_incomplete=True`, the little-group representations at the
  points refused are left out.  Pass `True` only where the tables of
  the irreptables package cannot be read at those points.

- **`cap`** (`int, default 200`): The largest number of sums listed.

- **`signed_combinations`** (`bool, default True`): Whether a vector no sum carries is searched again with negative
  multiplicities allowed.

**Returns**

- `EBRDecomposition or Refusal`: Whether a sum carries the vector, and every sum that does, up to
  `cap` of them.  Where none does, the record also holds what the
  absence of a sum rests on, and the combinations that carry the vector
  with each multiplicity of either sign and at most the bound in size,
  up to `cap` of them.  The record is truthy even where no sum carries
  the vector.  With `allow_incomplete=True`, the little-group
  representations at the points refused enter no equation, and the
  record's `numerical.completeness` says what the sums listed rest on.
  With `signed_combinations=False`, no combination with negative
  multiplicities is returned.

**Refusals**

A refusal handed in as `realized` is returned as it is.

- **`incomplete_content`**: the table refused at some point.  With `allow_incomplete=True`, the
  call does not refuse.

- **`not_a_band_structure`**: the vector fails the compatibility relations of the group.

- **`unknown_label`**: the vector names a little-group representation that the group's
  elementary band representations do not carry.

- **`restriction_not_exact`**: a little-group representation the model does not measure is matched
  by no measured one that occurs as often in every elementary band
  representation.  With `allow_incomplete=True` and a table that
  refused at some point, the call does not refuse.

- **`non_integer_multiplicity`**: an entry of the vector is not an integer.

- **`negative_multiplicity`**: an entry of the vector is negative.

- **`ebr_tables` / `irrep_tables` / `ebr_convention` / `tables_not_ready`**: the tables installed hold no elementary band representations of this
  group that can be read in the model's cell.

### `subtract`

```text
subtract(symmetry_data_a, symmetry_data_b)
```

The symmetry data vector of the bands in one set and not in another, `a - b`.

**Parameters**

- **`symmetry_data_a`, `symmetry_data_b`** (`HSPTable or mapping`): The two symmetry data vectors, `a` and `b`.  Pass for each the
  table [`hsp_table`](#hsp_table) returns, or `{corep: multiplicity}`
  written out by hand.

**Returns**

- `dict or Refusal`: `{corep: multiplicity}`, the zeros dropped.  A difference can only be
  returned where `a` carries every little-group representation of
  `b` at least as many times as `b` does.  Where the two are equal it is empty, and so
  falsy.  `isinstance(out, Refusal)` tells a refusal apart.

**Refusals**

- **`incomplete_content`**: either is a table that refused at some point.

- **`negative_multiplicity`**: `b` carries some little-group representation more times than `a`
  does, or an entry of either is negative.

- **`non_integer_multiplicity`**: an entry of either is not an integer.

### `regular_completion`

```text
regular_completion(symmetry_data, realized=None, *, bands=None)
```

The bands completing a band set to copies of the regular band representation.

One copy of the regular band representation is the band representation
induced from the general Wyckoff position, where it carries one state, or
the Kramers pair for a spinful model in a group with time reversal.  So it
has `n_1` bands at every k: `|U|`, or `2|U|` for the Kramers pair.
`|U|` is the number of unitary operations up to lattice translations,
counted in the single-valued group for a spinful model.  The symmetry data
vector of the missing bands is `n / n_1` times that of one copy, less that of the band
set, where `n` is the number of bands in `bands`.  It is decomposed by
[`decompose`](#decompose).

**Parameters**

- **`symmetry_data`** (`HSPTable or mapping`): The symmetry data vector of the band set.  Pass the table
  [`hsp_table`](#hsp_table) returns, or `{corep: multiplicity}`
  written out by hand.

- **`realized`** (`RealizedGroup, optional`): What [`realized_group`](#realized_group) returned for the model the coreps
  were measured on.  Pass it with a vector written out by hand, and
  nothing with a table.

- **`bands`** (`int or sequence of int`): The bands of the completed system.  Pass their number, or their
  indices.  Only their number is read.  Required.

**Returns**

- `RegularCompletion or Refusal`: The symmetry data vector of the missing bands, and its decomposition
  into elementary band representations.  The record also holds the
  number of copies and the symmetry data vector of one copy.

**Refusals**

A refusal handed in as `realized` is returned as it is.

- **`incomplete_content`**: the table refused at some point.

- **`not_a_regular_multiple`**: the number of bands is not a multiple of `n_1`.

- **`negative_multiplicity`**: the band set carries some little-group representation more times than
  the copies do, or an entry of the vector is negative.

- **`non_integer_multiplicity`**: an entry of the vector is not an integer.

- **`not_a_band_structure`**: the vector of the missing bands fails the compatibility relations of
  the group.

- **`unknown_label`**: the vector of one copy names a little-group representation that the
  group's elementary band representations do not carry.

- **`restriction_not_exact`**: a little-group representation the model does not measure is matched by
  no measured one that occurs as often in every elementary band
  representation.

- **`regular_content_not_integral`**: the characters of the little-group representations give one copy a
  multiplicity that is not an integer, or other than `n_1` bands at a
  point.

- **`magnetic_type_out_of_scope`**: a magnetic group of type III or IV, which
  [`realized_group`](#realized_group) never returns.

- **`ebr_tables` / `irrep_tables` / `ebr_convention` / `tables_not_ready`**: the installed tables cannot be read for this group in this cell.

### `ebr_rows`

```text
ebr_rows(group)
```

Each elementary band representation of the group as a symmetry data vector: the coreps it carries at the high-symmetry points.

The coreps are in the convention every returned label is, a lattice
translation acting as `e^{-ik·t}`, where the Bilbao Crystallographic
Server's tables have `e^{+ik·t}`: away from the time-reversal-invariant
points a row can carry other labels than the server's page, `K3` for its
`K2` in the row of `A1` at `1b` of space group 143, and the Wyckoff
positions never move.

**Parameters**

- **`group`** (`GroupData`): The group, `r.group`.

**Returns**

- `dict or Refusal`: `{(irrep, wyckoff): {corep: multiplicity}}`, the zeros dropped, in
  the order the tables list the elementary band representations: each
  key the site irrep it is induced from and its Wyckoff position,
  `("-E1", "1a")`, so that `rows["-E1", "1a"]` is one row.  A
  `Refusal` in its place is falsy and names what stopped the call.

**Refusals**

A refusal handed in as `group` is returned as it is.

- **`ebr_tables` / `irrep_tables` / `ebr_convention`**: no elementary band representations of this group in the tables
  installed, or none that can be read in this package's convention.

### `in_plane_restriction`

```text
in_plane_restriction(group, *, kpoints=None, require_exact=True)
```

Which coreps of the group's elementary band representations a layer measures, and whether each of the others repeats a measured one.

A layer's `H(k)` has no `k_z`, so its coreps at `k_z = 1/2` are not
measured.  Each that occurs in every elementary band representation as
often as a measured one adds no condition, and where every one does, what
is read on the measured coreps is exact.

**Parameters**

- **`group`** (`GroupData`): The group, `r.group`.

- **`kpoints`** (`sequence of str, optional`): The tabulated names of the points the model samples.  Default: every
  tabulated point in its plane.

- **`require_exact`** (`bool, default True`): Whether a corep left out that repeats none measured is refused.  With
  `require_exact=False`, those are returned as
  `Restriction.orphans`.

**Returns**

- `Restriction or Refusal`: `r.summary()` prints it; `r.copies` pairs each corep left out with
  the measured ones it repeats, and `r.exact` says whether each does.
  A `Refusal` in its place is falsy and names what stopped the call.

**Refusals**

A refusal handed in as `group` is returned as it is.

- **`restriction_not_exact`**: a corep left out repeats none measured.

- **`ebr_tables` / `irrep_tables` / `ebr_convention` / `tables_not_ready`**: no elementary band representations of this group, or no tables in this
  cell.

## Stable invariants

- [`symmetry_indicator`](#symmetry_indicator) — The symmetry-indicator class of a band set: the class of its symmetry data vector among the band structures modulo the atomic limits.
- [`stack_presentation`](#stack_presentation) — The layer repeated along its normal with no hopping between the copies: a crystal with the layer's bands at every `k_z`.
- [`srsi_tables`](#srsi_tables) — The stable real-space invariants of a grey space group, as the paper tabulates them.
- [`srsi_of_bands`](#srsi_of_bands) — The integer-valued stable real-space invariants of a band set, from its symmetry data vector.
- [`srsi_of_model`](#srsi_of_model) — A model's stable real-space invariants, read off its orbitals, a row for each Wyckoff orbit.
- [`splitting`](#splitting) — Whether the bands asked for and the rest of a model's bands can both be atomic insulators.
- [`nsit_criterion`](#nsit_criterion) — The paper's no-go criterion on a gapped splitting whose invariants are in hand.
- [`atomic_realizations`](#atomic_realizations) — Every atomic insulator with a band set's symmetry data, each a set of orbitals at maximal Wyckoff positions.

### `symmetry_indicator`

```text
symmetry_indicator(symmetry_data, realized=None)
```

The symmetry-indicator class of a band set: the class of its symmetry data vector among the band structures modulo the atomic limits.

A nonzero class says no band set with this symmetry data vector is an
atomic limit, nor any with atomic bands added; a zero class certifies
nothing.  A layer's class is read in its slab's three-dimensional group,
as the class of its decoupled stack, the layer repeated along its normal
with no hopping between copies.  A table's coreps at points out of the
layer's plane that repeat none in it are measured on that stack.  A vector
written out by hand is read at the points of the plane and at each point
out of it where it names a corep; a corep out of the plane it does not
name takes the multiplicity of the corep in the plane it repeats in every
elementary band representation of the layer's own Wyckoff positions.

**Parameters**

- **`symmetry_data`** (`HSPTable or mapping`): The table [`hsp_table`](#hsp_table) returns, which must be complete,
  or `{corep: multiplicity}` written out by hand.

- **`realized`** (`RealizedGroup, optional`): What [`realized_group`](#realized_group) returned, `r`, the measurement
  the coreps came from.  Pass it beside a vector written out by hand,
  and with a table nothing.

**Returns**

- `SymmetryIndicator or Refusal`: `si.summary()` prints the class.  `si.values` is the class, one
  integer modulo each of `si.divisors`, in the indicator group
  `si.indicator_group`, of order `si.order`; `si.indicators` the
  class in the named indicators of the paper that tabulates the stable
  real-space invariants; `si.group` the group it lives in.  The record
  is truthy whatever its class; a `Refusal` in its place is falsy and
  names what stopped the call.

**Refusals**

A refusal handed in as `realized` is returned as it is.

- **`incomplete_content`**: the table refused at some point; there is no override.

- **`indicator_not_evaluable`**: a layer's coreps at `k_z = 1/2` repeat none in its plane and could
  not be read: a vector whose layer's own band representations cannot
  be read in the group's cell, or a table whose decoupled stack stopped
  (`context["stack"]`).

- **`not_a_band_structure`**: the vector fails the group's compatibility relations.

- **`unknown_label`**: a corep the group's elementary band representations do not hold.

- **`non_integer_multiplicity` / `negative_multiplicity`**: the vector is no count of coreps.

- **`magnetic_type_out_of_scope`**: a magnetic group of type III or IV, which
  [`realized_group`](#realized_group) never returns.

- **`ebr_tables` / `irrep_tables` / `ebr_convention` / `tables_not_ready`**: no elementary band representations of this group, or no tables in this
  cell.

### `stack_presentation`

```text
stack_presentation(model, *, c=8.0, params=None)
```

The layer repeated along its normal with no hopping between the copies: a crystal with the layer's bands at every `k_z`.

The three-dimensional crystal a layer's rows at `k_z = ½` are measured
on: the layer's own lattice and a third vector along its normal, every
orbital at its own height, every hopping within one copy.  Its
`H(k, k_z)` is the layer's `H(k)` up to the phases of the orbitals'
heights, `Λ(k_z)† H(k) Λ(k_z)` with `Λ` diagonal, which each call
measures, so its spectrum and symmetry are the layer's at every `k_z`
and `c` changes neither.

**Parameters**

- **`model`** (`pythtb.TBModel`): A layer: periodic in two directions, in a cell of two or three.

- **`c`** (`float, optional`): The vacuum between the copies, in the model's length units; the period
  is the layer's extent along its normal plus `c`.  Default
  `STACK_C`, `8.0`.

- **`params`** (`dict, optional`): The scalar value of each parameter of the model, keyed by name:
  `{"lam": 0.3}`.

**Returns**

- `pythtb.TBModel or Refusal`: The stack, periodic in three directions, its parameters evaluated at
  `params`.  A `Refusal` in its place is falsy.

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

From a measured group the tables come with the group, and every call that
reads a model's site irreps reads them in these tables; from a space-group
number with `spinor=` they need no model.

**Parameters**

- **`group_or_number`** (`GroupData or int`): A grey group a model was measured to have, `r.group`, or a
  space-group number from 1 to 230.

- **`spinor`** (`bool, optional`): With a number, and only then: the tables with spin-orbit coupling
  (`True`) or without.

**Returns**

- `SRSITables or Refusal`: `tables.summary()` prints the invariants; `tables.theta(p)` evaluates
  them on a configuration.  From a number, `tables.group` holds the grey
  group's name, number and valuedness and no cell.  A `Refusal` in its
  place is falsy.

**Refusals**

- **`srsi_scope`**: the group is not grey, or the number is outside 1 to 230.

- **`srsi_table_absent`**: one of the thirteen groups whose one invariant is the band count.

- **`srsi_data`**: the tables are not installed (`srsi_status`).

- **`srsi_ebr_basis`**: the installed `irreptables` lists other elementary band representations
  than the tables were converted against; `context["installed"]` holds its
  list.

- **`ebr_tables`, `irrep_tables`, `ebr_convention`**: from [`ebr_data`](#ebr_data).

### `srsi_of_bands`

```text
srsi_of_bands(symmetry_data, realized=None)
```

The integer-valued stable real-space invariants of a band set, from its symmetry data vector.

They are the invariants of a rational site-symmetry vector `p` with the band
set's symmetry data vector, every such `p` giving the same ones, and they are
fractional exactly when the symmetry indicator is nonzero.  The invariants
valued modulo n are no function of the symmetry data vector, so none are
returned; [`srsi_of_model`](#srsi_of_model) reads them off a model's orbitals.

**Parameters**

- **`symmetry_data`** (`HSPTable or mapping`): The table [`hsp_table`](#hsp_table) returns, which must be complete,
  or `{corep: multiplicity}` written out by hand, as
  [`symmetry_indicator`](#symmetry_indicator) takes it.

- **`realized`** (`RealizedGroup, optional`): What [`realized_group`](#realized_group) returned, `r`, the measurement
  the coreps came from, in a grey group with tables.  Pass it beside a
  vector written out by hand, and with a table nothing.

**Returns**

- `BandSRSIs or Refusal`: `out.summary()` prints the invariants, whether they are fractional, and
  the symmetry-indicator class, and `out.line()` the same on one line;
  `out.theta_Z` holds the invariants by name and `out.indicator` the
  class.  A `Refusal` in its place is falsy.

**Refusals**

A refusal handed in as `realized` is returned as it is.

- **`srsi_scope`, `srsi_table_absent`, `srsi_data`, `srsi_ebr_basis`**: from [`srsi_tables`](#srsi_tables): the group has no tables.

- **`incomplete_content`, `not_a_band_structure`, `indicator_not_evaluable`**: from [`symmetry_indicator`](#symmetry_indicator), with the refusals of the
  symmetry data vector it reads.

### `srsi_of_model`

```text
srsi_of_model(realized)
```

A model's stable real-space invariants, read off its orbitals, a row for each Wyckoff orbit.

Each Wyckoff orbit's symmetry data vector is measured on its own orbitals
and decomposed into elementary band representations at its Wyckoff position
or, where the tables list none there, at a position its orbitals slide onto;
a row's invariants are that decomposition's, and the model's their sum.  A
model's orbitals are a known configuration, so both kinds are returned.

**Parameters**

- **`realized`** (`RealizedGroup`): What [`realized_group`](#realized_group) returned, `r`.

**Returns**

- `ModelSRSIs or Refusal`: `out.summary()` prints the invariants and each row's; `out.theta_Z`
  and `out.theta_Zn` hold the model's, by name, and
  `out.wyckoff_orbits` a row for each Wyckoff orbit, the row of
  `r.wyckoff_orbits` with its invariants.  A `Refusal` in its place is
  falsy.

**Refusals**

- **`srsi_scope`, `srsi_table_absent`, `srsi_data`, `srsi_ebr_basis`**: from [`srsi_tables`](#srsi_tables): the group has no tables.

- **`wyckoff_orbit_content_incomplete`**: a Wyckoff orbit's symmetry data vector could not be read at a
  high-symmetry point.

- **`wyckoff_orbit_not_localizable`**: a Wyckoff orbit's symmetry data vector is a band representation of no
  position its orbitals can slide onto.

- **`wyckoff_setting`**: the symmetry data vector decomposes at a position the geometry says the
  orbitals are not at: the package disagreeing with itself, a case to
  report.

- **`tables_not_ready`, `wyckoff_unresolved`, `ebr_tables`**: from [`wyckoff_sites_tables`](#wyckoff_sites_tables) and [`ebr_data`](#ebr_data).

### `splitting`

```text
splitting(realized, *, bands=None)
```

Whether the bands asked for and the rest of a model's bands can both be atomic insulators.

The paper's no-go criterion on a gapped splitting of a model's bands: the
valence bands are the bands asked for, the conduction bands the rest.  The
whole model's invariants are read off its orbitals ([`srsi_of_model`](#srsi_of_model)),
the integer-valued ones of the valence bands off their symmetry data
vector ([`srsi_of_bands`](#srsi_of_bands)), and those of the conduction bands are the
difference; [`nsit_criterion`](#nsit_criterion) decides.  The gap is taken as given.

**Parameters**

- **`realized`** (`RealizedGroup`): What [`realized_group`](#realized_group) returned, `r`.

- **`bands`** (`int or sequence of int`): The valence bands: an `int` is the lowest *n*, a sequence those bands.
  Required.

**Returns**

- `Splitting or Refusal`: `out.nsit` holds whether the splitting is NSIT, which it is only
  where both sides have atomic realizations and no pair of them adds up
  to the whole model's invariants valued modulo n.  `out.valence` and
  `out.conduction` hold each side's realizations, empty where its
  invariants are fractional.  `out.summary()` prints it and what it
  rests on.  The record is truthy.  A `Refusal` in its place is falsy.

**Refusals**

- **`srsi_scope`, `srsi_table_absent`, `srsi_data`, `srsi_ebr_basis`**: from [`srsi_tables`](#srsi_tables): the group has no tables.

- **`wyckoff_orbit_content_incomplete`, `wyckoff_orbit_not_localizable`, `wyckoff_setting`**: from [`srsi_of_model`](#srsi_of_model), for the orbitals.

- **`incomplete_content`, `not_a_band_structure`, ...**: from [`hsp_table`](#hsp_table) and [`srsi_of_bands`](#srsi_of_bands), for the bands.

- **`srsi_realizations_truncated`**: from [`nsit_criterion`](#nsit_criterion), which lists each side at the default cap of
  [`atomic_realizations`](#atomic_realizations).

### `nsit_criterion`

```text
nsit_criterion(tables, theta_total, theta_Z_v, theta_Z_c)
```

The paper's no-go criterion on a gapped splitting whose invariants are in hand.

All the bands together are an atomic insulator.  If the two sides, the
valence and the conduction bands, are atomic insulators too, then the
invariants valued modulo n of some atomic realization of each add up to the
whole model's, modulo each n.  Every realization of each side is listed,
since one pair that does not add up proves nothing.  The splitting is NSIT
where both sides have realizations and no pair adds up; it is not where the
valence bands' invariants are fractional or a side has no realization at
all, which the symmetry data vector already shows, nor where the group has
no invariants valued modulo n or some pair adds up.

**Parameters**

- **`tables`** (`SRSITables`): The group's tables.

- **`theta_total`** (`Invariants or ModelSRSIs`): The whole model's invariants, with `theta_Z` and `theta_Zn` by
  name: what `tables.theta` returns for the model's orbitals, or what
  [`srsi_of_model`](#srsi_of_model) returns.

- **`theta_Z_v`, `theta_Z_c`** (`mapping`): `{name: value}`: the integer-valued invariants of the valence and of
  the conduction bands, as `b.theta_Z` holds them.

**Returns**

- `Splitting or Refusal`: `out.nsit` holds whether the splitting is NSIT; `out.summary()`
  prints it, why, the realizations of each side and the pairs that add
  up.  The record is truthy.  A `Refusal` in its place is falsy.

**Refusals**

- **`srsi_realizations_truncated`**: from [`atomic_realizations`](#atomic_realizations), which lists each side at its default
  cap: more realizations than the cap, which [`atomic_realizations`](#atomic_realizations)
  with a larger `cap=` lists.

### `atomic_realizations`

```text
atomic_realizations(tables, theta_Z, *, cap=256)
```

Every atomic insulator with a band set's symmetry data, each a set of orbitals at maximal Wyckoff positions.

The configurations with nonnegative integer multiplicities over the maximal
site irreps whose integer-valued invariants are `theta_Z`: the same
invariants are the same symmetry data.  Maximal positions are enough, since
an atomic insulator moves onto them with its invariants unchanged, and the
list is complete.  Fractional invariants have none: the band set carries a
nonzero symmetry indicator, and nothing is enumerated.

**Parameters**

- **`tables`** (`SRSITables`): The group's tables, from [`srsi_tables`](#srsi_tables).

- **`theta_Z`** (`mapping`): `{name: value}`: the band set's integer-valued invariants, each once,
  as `b.theta_Z` holds them.

- **`cap`** (`int, default 256`): The most realizations to list; more is refused.

**Returns**

- `Realizations or Refusal`: `out.summary()` prints every realization with its own invariants
  valued modulo n; `out.realizations` holds them.  The record is truthy,
  and an empty list with whole-number `theta_Z` is a fragile band set.
  A `Refusal` in its place is falsy.

**Refusals**

- **`srsi_realizations_truncated`**: more than `cap` realizations; a larger `cap=` lists them.
