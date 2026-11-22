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
- [`in_plane_restriction`](#in_plane_restriction) — The little-group representations a model measures, and what the others repeat.

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

The little-group representations a model measures, and what the others repeat.

The little-group representations of the group's elementary band
representations are split into those at the points in `kpoints`, the
measured ones, and the others, which are left out.  A layer, whose
`H(k)` has no `k_z`, has no point at `k_z = 1/2` measured.  A
little-group representation left out may occur in every elementary band
representation as often as a measured one.  It is then said to repeat
that one, and adds no condition to a symmetry data vector.

**Parameters**

- **`group`** (`GroupData`): The group, `r.group`.

- **`kpoints`** (`sequence of str, optional`): The tabulated names of the points measured.  Default: every tabulated
  point, for a layer's group those in its plane.

- **`require_exact`** (`bool, default True`): Whether the call refuses where a little-group representation left out
  repeats no measured one.

**Returns**

- `Restriction or Refusal`: The little-group representations measured and those left out, and for
  each one left out the measured ones it repeats.  The record also says
  whether every one left out repeats a measured one.  Where each does,
  the decomposition or class read from the measured ones is that of the
  whole symmetry data vector.  With `require_exact=False`, the record
  lists those that repeat none in its `orphans`.  A `Refusal` in its
  place is falsy and names what stopped the call.

**Refusals**

A refusal handed in as `group` is returned as it is.

- **`restriction_not_exact`**: a little-group representation left out repeats no measured one.  With
  `require_exact=False`, the call does not refuse.

- **`ebr_tables` / `irrep_tables` / `ebr_convention` / `tables_not_ready`**: the tables installed hold no elementary band representations of this
  group that can be read in the model's cell.

## Stable invariants

- [`symmetry_indicator`](#symmetry_indicator) — The symmetry-indicator class of a set of bands.
- [`stack_presentation`](#stack_presentation) — The layer repeated along its normal with no hopping between the copies: a crystal with the layer's bands at every `k_z`.
- [`srsi_tables`](#srsi_tables) — The stable real-space invariants of a grey space group, as tabulated.
- [`srsi_of_bands`](#srsi_of_bands) — The integer-valued stable real-space invariants (ZSRSIs) of a band set.
- [`srsi_of_model`](#srsi_of_model) — The stable real-space invariants of a model, a row for each Wyckoff orbit.
- [`splitting`](#splitting) — The no-go test of a gapped splitting of a model's bands.
- [`nsit_criterion`](#nsit_criterion) — The no-go test of a gapped splitting, from its invariants.
- [`atomic_realizations`](#atomic_realizations) — Every atomic realization of given integer-valued stable real-space invariants.

### `symmetry_indicator`

```text
symmetry_indicator(symmetry_data, realized=None)
```

The symmetry-indicator class of a set of bands.

The columns of the group's matrix $`M`$ of elementary band
representations are their symmetry data vectors.  The irreptables package
ships its Smith normal form, $`U M V = D`$.  The class of the symmetry
data vector $`y`$ of the bands is the entries of $`U y`$ at which
the diagonal of $`D`$ holds an elementary divisor above 1, each taken
modulo that divisor.  A layer's class is read in the three-dimensional
space group of its slab, as the class of its decoupled stack, the layer
repeated along its normal with no hopping between copies.  Each of its
little-group representations at `k_z = 1/2` that occurs in every
elementary band representation as often as one in the plane takes that
one's multiplicity.  For a table, the others are measured on the stack.  A
vector written out by hand is read as given at each point out of the plane
where it names a little-group representation.  At the other points out of
the plane, how often each little-group representation occurs is compared
with those in the plane on the elementary band representations of the
Wyckoff positions within the layer.

**Parameters**

- **`symmetry_data`** (`HSPTable or mapping`): The symmetry data vector.  Pass the table [`hsp_table`](#hsp_table)
  returns, or `{corep: multiplicity}` written out by hand.

- **`realized`** (`RealizedGroup, optional`): What [`realized_group`](#realized_group) returned for the model the
  symmetry data vector was measured on.  Pass it with a vector written
  out by hand, and nothing with a table.

**Returns**

- `SymmetryIndicator or Refusal`: The class, one integer modulo each elementary divisor, with its
  indicator group and its order.  For a group with time reversal and a
  nontrivial indicator group, the record also holds the class in the
  named indicators of Hwang *et al.*, Nat. Commun. **17**, 8564 (2026),
  where the tables of that paper are installed.
  A class can only be returned for a table that refused at no point.
  The record is truthy whatever its class.

**Refusals**

A refusal handed in as `realized` is returned as it is.

- **`incomplete_content`**: the table refused at some point.

- **`not_a_band_structure`**: the vector fails the compatibility relations of the group.

- **`unknown_label`**: the vector names a little-group representation that the group's
  elementary band representations do not carry.

- **`indicator_not_evaluable`**: a layer's little-group representations at `k_z = 1/2`, none of
  which occurs in every elementary band representation as often as one
  in its plane, could not be read.  For a table, the measurement on
  the decoupled stack stopped, and `context["stack"]` says where.  For a
  vector written out by hand, the Wyckoff positions of the tables cannot
  be read in the group's cell.

- **`non_integer_multiplicity`**: an entry of the vector is not an integer.

- **`negative_multiplicity`**: an entry of the vector is negative.

- **`magnetic_type_out_of_scope`**: a magnetic group of type III or IV, which
  [`realized_group`](#realized_group) never returns.

- **`ebr_tables` / `irrep_tables` / `ebr_convention` / `tables_not_ready`**: the tables installed hold no elementary band representations of this
  group that can be read in the model's cell.

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

The stable real-space invariants of a grey space group, as tabulated.

The tables are those of Hwang et al., Stable real-space invariants and
topology beyond symmetry indicators, Nat. Commun. 17, 8564 (2026).  They
are read for a group a model was measured to have, or for a space-group
number, which needs no model.

**Parameters**

- **`group_or_number`** (`GroupData or int`): The group.  Pass `r.group` of what [`realized_group`](#realized_group)
  returns, or a space-group number from 1 to 230.

- **`spinor`** (`bool, optional`): Whether the double-valued tables, those with spin-orbit coupling, are
  read.  Pass it with a space-group number, and only then.

**Returns**

- `SRSITables or Refusal`: The tables of the group.  A `Refusal` in its place is falsy.

**Refusals**

A refusal handed in as `group_or_number` is returned as it is.

- **`srsi_scope`**: the group is not grey, or the number is outside 1 to 230.

- **`srsi_table_absent`**: the group is one of the thirteen space groups the tables leave out.
  In each, the general position is the only Wyckoff position, and it has
  one site irrep.  The one invariant is the band count.

- **`srsi_data`**: the tables are not installed.  `srsi_status` says which is
  missing.

- **`srsi_ebr_basis`**: the installed irreptables package lists elementary band
  representations of the measured group other than those the tables
  were converted against.  `context["installed"]` holds its list.

- **`ebr_tables` / `irrep_tables` / `ebr_convention`**: the tables installed hold no elementary band representations of the
  measured group that can be read in this package's convention.

### `srsi_of_bands`

```text
srsi_of_bands(symmetry_data, realized=None)
```

The integer-valued stable real-space invariants (ZSRSIs) of a band set.

The ZSRSIs are fixed by the symmetry data vector of the bands.  They are
computed by writing the symmetry data vector as a rational sum of
elementary band representations and applying to it the linear map
`SRSITables.theta`.  The SRSIs valued modulo n are not fixed by the
symmetry data vector and are not returned; [`srsi_of_model`](#srsi_of_model) computes
them for a full model.

**Parameters**

- **`symmetry_data`** (`HSPTable or mapping`): The symmetry data vector.  Pass the table [`hsp_table`](#hsp_table)
  returns, or `{corep: multiplicity}` written out by hand.

- **`realized`** (`RealizedGroup, optional`): What [`realized_group`](#realized_group) returned for the model the
  symmetry data vector was measured on.  Pass it with a vector written
  out by hand, and nothing with a table.

**Returns**

- `BandSRSIs or Refusal`: The ZSRSIs of the band set by name, the symmetry-indicator class of
  the same bands, and the rational sum of elementary band
  representations the ZSRSIs were computed from.

**Refusals**

A refusal handed in as `realized` is returned as it is.

- **`srsi_scope` / `srsi_table_absent`**: the tables hold no stable real-space invariants of the group: it is
  not grey, or it is one of the thirteen space groups whose one
  invariant is the band count.

- **`srsi_data` / `srsi_ebr_basis`**: the tables of the stable real-space invariants are not installed, or
  were converted against another list of elementary band
  representations than the one installed.

- **`incomplete_content`**: the table holds no little-group representations at some high-symmetry
  point, for example where the bands split a degenerate multiplet.

- **`not_a_band_structure`**: the vector fails the compatibility relations of the group.

- **`indicator_not_evaluable`**: the bands are a layer's, and some of their little-group
  representations at `k_z = 1/2` could not be read.  For each of these,
  no little-group representation in the layer's plane occurs as often as
  it does in every elementary band representation.  For a table, their
  measurement on the layer's decoupled stack stopped.  For a vector
  written out by hand, the Wyckoff positions of the tables cannot be
  read in the group's cell.

- **`unknown_label`**: the vector names a little-group representation that the group's
  elementary band representations do not carry.

- **`non_integer_multiplicity` / `negative_multiplicity`**: an entry of the vector is not a nonnegative integer.

- **`ebr_tables` / `irrep_tables` / `ebr_convention` / `tables_not_ready`**: the tables installed hold no elementary band representations of this
  group that can be read in the model's cell.

### `srsi_of_model`

```text
srsi_of_model(realized)
```

The stable real-space invariants of a model, a row for each Wyckoff orbit.

The symmetry data vector of each Wyckoff orbit is measured on its own
orbitals, from the trace of each little-group operation over them at each
high-symmetry point.  It is decomposed into elementary band
representations at each Wyckoff position that the orbitals can slide onto
by varying the free coordinates of their own, and at which the tables
list elementary band representations.  Their own position is among these
where the tables list one there.  A row of a spinless model whose
symmetry data vector has no such decomposition is given the stable
real-space invariants of its own site irreps, each identified with one
the tables list at its position, in `SRSITables.site_irreps`.  The
stable real-space invariants (SRSIs) of a row are those of its
decompositions, all of which agree, and the model's are the sum over the
rows.  The orbitals are a known configuration, so the SRSIs valued modulo
n are returned with the integer-valued ones.

**Parameters**

- **`realized`** (`RealizedGroup`): What [`realized_group`](#realized_group) returned for the model.

**Returns**

- `ModelSRSIs or Refusal`: The model's integer-valued SRSIs and its SRSIs valued modulo n, by
  name, and a row for each row of `realized.wyckoff_orbits` with its
  own SRSIs and its decompositions.

**Refusals**

A refusal handed in as `realized` is returned as it is.

- **`srsi_scope` / `srsi_table_absent`**: the tables hold no SRSIs of the group: it is not grey, or it is one of
  the thirteen space groups whose one invariant is the band count.

- **`srsi_data` / `srsi_ebr_basis`**: the tables of the SRSIs are not installed, or were converted against
  another list of elementary band representations than the one
  installed.

- **`ebr_tables` / `irrep_tables` / `ebr_convention` / `tables_not_ready`**: the tables installed hold no elementary band representations of this
  group that can be read in the model's cell.

- **`wyckoff_orbit_content_incomplete`**: the symmetry data vector of a Wyckoff orbit could not be read at some
  high-symmetry point.

- **`wyckoff_orbit_not_localizable`**: no sum of elementary band representations at any one position the
  orbitals of a Wyckoff orbit can slide onto carries its symmetry data
  vector.  The model is spinful, or one of the orbit's site irreps could
  not be identified with a site irrep the tables list at its position.

- **`wyckoff_setting` / `wyckoff_unresolved`**: the Wyckoff position of a Wyckoff orbit read from its centers differs
  from the one at which its symmetry data vector is a sum of elementary
  band representations, or the positions could not be read in the
  setting of the tables.

### `splitting`

```text
splitting(realized, *, bands=None)
```

The no-go test of a gapped splitting of a model's bands.

The record returned says whether the splitting is non-symmetry-indicated
topological (NSIT).  The criterion is that of Hwang et al., Nat. Commun.
17, 8564 (2026), as [`nsit_criterion`](#nsit_criterion) applies it.  The valence bands
are the bands asked for and the conduction bands are the rest.  The
stable real-space invariants (SRSIs) of all the bands are read off the
model's orbitals with [`srsi_of_model`](#srsi_of_model).  The integer-valued SRSIs of
the valence bands are read off their symmetry data vector with
[`srsi_of_bands`](#srsi_of_bands), and those of the conduction bands are the
difference.  The splitting is NSIT exactly where both sides have atomic
realizations, the group has SRSIs valued modulo n, and no pair of
realizations has SRSIs valued modulo n adding up to those of all the
bands.  The bands are read at the high-symmetry points alone, so a gap
between those points is taken as given.

**Parameters**

- **`realized`** (`RealizedGroup`): What [`realized_group`](#realized_group) returned for the model.

- **`bands`** (`int or sequence of int`): The valence bands.  Pass an integer `n` for the lowest `n` bands,
  or a sequence of band indices.  Required.

**Returns**

- `Splitting or Refusal`: Whether the splitting is NSIT, every atomic realization of each side,
  and the pairs whose SRSIs valued modulo n add up to those of all the
  bands.  The record is truthy even where `nsit` is `False`.

**Refusals**

A refusal handed in as `realized` is returned as it is.

- **`srsi_scope` / `srsi_table_absent`**: the tables hold no SRSIs of the group: it is not grey, or it is one of
  the thirteen space groups whose one invariant is the band count.

- **`srsi_data` / `srsi_ebr_basis`**: the tables of the SRSIs are not installed, or were converted against
  another list of elementary band representations than the one
  installed.

- **`ebr_tables` / `irrep_tables` / `ebr_convention` / `tables_not_ready`**: the tables installed hold no elementary band representations of this
  group that can be read in the model's cell.

- **`wyckoff_orbit_content_incomplete` / `wyckoff_orbit_not_localizable` / `wyckoff_setting` / `wyckoff_unresolved`**: the SRSIs of the orbitals could not be read, as [`srsi_of_model`](#srsi_of_model)
  says.

- **`incomplete_content`**: the valence bands could not be read at some high-symmetry point, for
  example where they split a degenerate multiplet.

- **`not_a_band_structure` / `indicator_not_evaluable`**: the integer-valued SRSIs of the valence bands could not be read, as
  [`srsi_of_bands`](#srsi_of_bands) says.

- **`srsi_realizations_truncated`**: the enumeration of one side reached the default `cap` of
  [`atomic_realizations`](#atomic_realizations).

### `nsit_criterion`

```text
nsit_criterion(tables, theta_total, theta_Z_v, theta_Z_c)
```

The no-go test of a gapped splitting, from its invariants.

The record returned says whether the splitting is non-symmetry-indicated
topological (NSIT).  The criterion is that of Hwang et al., Nat. Commun.
17, 8564 (2026).  All the bands together are an atomic insulator, whose
stable real-space invariants (SRSIs) are `theta_total`.  The valence
and the conduction bands split them, and their integer-valued SRSIs
(ZSRSIs) are `theta_Z_v` and `theta_Z_c`.  Where both sides are
atomic insulators too, some atomic realization of each has SRSIs valued
modulo n that add up to those of all the bands, modulo each n.  One pair
that does not add up does not show the splitting topological, so every
pair of realizations of the two sides, enumerated with
[`atomic_realizations`](#atomic_realizations), is tested.  A side whose ZSRSIs are
fractional has no realization.  The splitting is NSIT exactly where both
sides have realizations, the group has SRSIs valued modulo n, and no pair
adds up.

**Parameters**

- **`tables`** (`SRSITables`): The tables of the group, what [`srsi_tables`](#srsi_tables) returns.

- **`theta_total`** (`Invariants or ModelSRSIs`): The SRSIs of all the bands, as a record with the fields `theta_Z`
  and `theta_Zn`, each keyed by the invariants' names.  Pass what
  `SRSITables.theta` returns for the model's orbitals, or what
  [`srsi_of_model`](#srsi_of_model) returns.

- **`theta_Z_v`, `theta_Z_c`** (`mapping`): The ZSRSIs of the valence and of the conduction bands,
  `{name: value}`.  Pass for each the `theta_Z` of what
  [`srsi_of_bands`](#srsi_of_bands) returns.

**Returns**

- `Splitting or Refusal`: Whether the splitting is NSIT, every atomic realization of each side,
  and the pairs whose SRSIs valued modulo n add up to those of all the
  bands.  The record is truthy even where `nsit` is `False`.

**Refusals**

- **`srsi_realizations_truncated`**: the enumeration of one side reached the default `cap` of
  [`atomic_realizations`](#atomic_realizations).

### `atomic_realizations`

```text
atomic_realizations(tables, theta_Z, *, cap=256)
```

Every atomic realization of given integer-valued stable real-space invariants.

An atomic realization is a set of site irreps at maximal Wyckoff
positions, with nonnegative integer multiplicities.  Those returned are
every one whose integer-valued stable real-space invariants (ZSRSIs) are
`theta_Z`.  They are found by enumerating the nonnegative integer
solutions of the linear equations that set the ZSRSIs to `theta_Z` and
the band count to the one `theta_Z` fixes.  Maximal positions suffice,
since an atomic insulator moves onto them with its stable real-space
invariants unchanged.  The list of realizations is complete.  All the
realizations returned have one symmetry data vector, since two with the
same ZSRSIs have the same one.  Each realization carries its own stable
real-space invariants valued modulo n.  Fractional ZSRSIs have no
realization.

**Parameters**

- **`tables`** (`SRSITables`): The tables of the group, what [`srsi_tables`](#srsi_tables) returns.

- **`theta_Z`** (`mapping`): The ZSRSIs, `{name: value}`, with one entry for each of the tables'
  integer-valued invariants.  Pass the `theta_Z` of what
  [`srsi_of_bands`](#srsi_of_bands) returns.

- **`cap`** (`int, default 256`): The number of realizations at which the enumeration stops.

**Returns**

- `Realizations or Refusal`: Every atomic realization of `theta_Z`, each with its stable
  real-space invariants valued modulo n.  The list is empty where
  `theta_Z` is fractional, and where no atomic insulator has these
  ZSRSIs.  The record is truthy even where the list is empty.

**Refusals**

- **`srsi_realizations_truncated`**: the enumeration reached `cap` realizations.  A `cap=` above their
  number lists them all.
