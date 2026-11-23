# Returned records

Every record a call of `pythtqc` returns, by task: what it holds, the call that
returns it or the field that holds it, each of its fields, and its methods. A call that
cannot complete returns a [`Refusal`](#refusal) in place of its record. The page
is written from the records' docstrings by `scripts/build_reference.py` and is
not edited by hand. `help` prints the whole docstring of a record, with
`import pythtqc as tqc`: `help(tqc.RealizedGroup)`, or
`help(tqc.groups.Descent)` for a record exported by a module of the package
alone. The `summary()` of most records prints its parts, the line that opens
each part ending with the field or fields that hold it.

The arguments of each call are on the page [Functions](functions.md), what each
refusal means and what to do about it on [Refusals](refusals.md), and the words
in the [Glossary](glossary.md). A record held in a field of another follows it,
one heading level lower down to the sixth, and its line **Held in** names the
field. The records fall into six groups:

- [The group](#the-group): what `realized_group` returns and the records it holds, the site multiplets it can take, and an operation's matrix in its canonical phase.
- [Group data in this cell's coordinates](#group-data-in-this-cells-coordinates): the tables' k-points, little groups, Wyckoff positions, k-point meshes and elementary band representations, in the model's own cell.
- [Labeling the bands](#labeling-the-bands): the sewing matrices, characters and little-group representations of a set of bands, at one high-symmetry point or at all of them.
- [Atomic limits](#atomic-limits): the sums of elementary band representations that carry a symmetry data vector, and the completion to copies of the regular band representation.
- [Stable invariants](#stable-invariants): the symmetry-indicator class of a band set, and the stable real-space invariants of a group, a band set and a model.
- [Refusals](#refusals): what a call returns where it cannot complete, the exception that carries it, the error raised for a defect, and the one warning.

## The group

`realized_group` returns a `RealizedGroup`. It holds the space group the
Hamiltonian obeys, how each operation of the cell was measured, the atomic-limit
table of the orbitals, and the representation of the group on them. Each record
it holds stands under it. After it stand the site multiplets `realized_group`
can take, and the records of `symmetry.normalize_rep`.

- [`RealizedGroup`](#realizedgroup): The space group a Hamiltonian obeys, and what was measured with it.
- [`SiteMultiplets`](#sitemultiplets): The multiplets the Hamiltonian binds each site's orbitals into.
- [`NormalizedRep`](#normalizedrep): An operation's matrix in its canonical phase, and what fixed the phase.
- [`RepAnchor`](#repanchor): What fixes the phase of an operation's matrix that covariance leaves open.

### `RealizedGroup`

The space group a Hamiltonian obeys, and what was measured with it.

The record [`realized_group`](functions.md#realized_group) returns.  `summary()` prints what was
measured, a line for each part, with the field that holds it.  Every call
that labels the bands takes the record whole, as `hsp_table(r, bands=n)`
does.

**Returned by** [`realized_group`](functions.md#realized_group).

**Exported by** `pythtqc`, `pythtqc.groups`.

**Attributes**

- **`group`** (`GroupData`): The space group the Hamiltonian obeys, every operation of it measured
  to hold.

- **`operations`** (`OperationTable`): The operations of `group` by index, one for each coset of the
  lattice translations, each with its Seitz symbol and its name.

- **`wyckoff_orbits`** (`AtomicLimitTable`): The model's orbitals, a row for each Wyckoff orbit: its Wyckoff
  position, its site-symmetry group, the site irreps its orbitals carry
  and the band representation they induce.

- **`descent`** (`Descent or None`): If not `None`, the operations of the cell's group that were dropped,
  and why.  `None` where the descent dropped no unitary operation.
  The operations a declaration excluded before any hopping was read are
  recorded in `decoration.declared`, and time reversal broken on the
  group in `time_reversal`.

- **`time_reversal`** (`TimeReversal`): How time reversal was measured.

- **`beyond_grey`** (`BeyondGrey or None`): If not `None`, the operations of the cell combined with time reversal
  that hold while time reversal itself is broken.  `None` where none
  holds.

- **`conserved`** (`Conserved`): How many matrices beyond the identity commute with `H(k)` at every
  `k`, and the sectors they split the states into: symmetry beyond the
  crystal's own.

- **`environment`** (`LayerEnvironment or None`): If not `None`, a layer's environment, and the other environment
  measured beside it.  `None` for a crystal periodic in three
  directions.

- **`decoration`** (`Decoration or None`): If not `None`, the points added to the cell before its group was
  found.  `None` for a crystal periodic in three directions to which
  nothing was added.

- **`model`** (`pythtb.TBModel`): The model every later call reads.  For a layer it is the slab, with
  the same `H(k)`, or the same up to the change of basis that
  `environment.presentation.tilted` records.  For a model with free
  parameters, the values of `params=` are in place.  Where the orbital
  centers were moved, it holds the moved centers.  Where `H` was
  projected onto its symmetric part, it is the projected model.

- **`rep`** (`SpaceGroupRep`): The representation of the group on the orbitals: the matrix of each
  operation.

- **`spin`** (`SpinBasis or None`): If not `None`, the spin pairs `spin=` declared.  `None` where
  none were declared.

- **`numerical`** (`RealizedGroupNumerical`): How the group was measured: a declared precision, the centers moved,
  `H` projected, what was measured of each operation, and the model
  as it was handed in.

**Methods**

- **`summary()`**: What was measured, a line for each part, with the field that holds it.

#### `GroupData`

A space group, written in the cell of the model's pythtb Lattice.

The record held by `realized_group(model).group`.  The calls that read
the tables take it, among them [`operations`](functions.md#operations) and
[`corep_table`](functions.md#corep_table).  Its operations, one per coset of the
lattice translations, are listed by `realized_group(model).operations`,
each with an index counted from 1.  The group's three-dimensional cell is
the cell of the pythtb Lattice.  For a layer, the lattice vector out of
its plane lies along its normal and spans vacuum.  The group
[`srsi_tables`](functions.md#srsi_tables) reads from a space-group number has no cell.
`summary()` prints it.

**Held in** [`OperationTable.group`](#operationtable), [`RealizedGroup.group`](#realizedgroup), [`SRSITables.group`](#srsitables), [`BandSRSIs.group`](#bandsrsis), [`ModelSRSIs.group`](#modelsrsis), [`Splitting.group`](#splitting), [`Realizations.group`](#realizations), [`HSPTable.group`](#hsptable), [`SymmetryIndicator.group`](#symmetryindicator), [`EBRData.group`](#ebrdata).

**Exported by** `pythtqc`, `pythtqc.groups`.

**Attributes**

- **`name`** (`str`): The Hermann-Mauguin symbol, `"P6/mmm1'"`.  A trailing `1'` marks
  time reversal as one of its operations.

- **`number_str`** (`str`): The group's number in the tables: the BNS number of the magnetic
  space group, `"191.234"`, for a grey group, and the space-group
  number, `"168"`, for a group without time reversal.

- **`spinor`** (`bool`): Whether it is the double group, of states with spin one-half.

- **`magnetic_type`** (`str`): The type of the magnetic space group: `"II"` for a grey group, which
  contains time reversal as an operation of its own, and `"I"` for a
  group with no operation combined with time reversal.

- **`periodic_dirs`** (`tuple of int or None`): If not `None`, the lattice vectors along which the crystal is
  periodic, by their index in the group's three-dimensional cell:
  `(0, 1, 2)` for a crystal periodic in three directions, and
  `(0, 1)` for a layer whose vacuum is along the third.  `None` for
  the group of a space-group number, which has no cell.

**Methods**

- **`summary()`**: The group as text, a line for each quantity.

#### `OperationTable`

The operations of a group, a mapping from each index to its operation.

The record [`operations`](functions.md#operations) returns, which `r.operations` also holds.
It holds one operation for each coset of the lattice translations.
`table[2]` is the [`Operation`](#operation) of index 2, and `table.values()`
gives the operations in index order.  Printed, it is a table of each
operation's index, Seitz symbol and name.  `summary()` prints it.

**Returned by** [`operations`](functions.md#operations).

**Held in** [`RealizedGroup.operations`](#realizedgroup), [`BandCharacters.operations`](#bandcharacters).

**Exported by** `pythtqc`, `pythtqc.groups`.

**Attributes**

- **`group`** (`GroupData`): The group whose operations these are, `r.group`.

**Methods**

- **`summary()`**: The table as printed: each operation's index, Seitz symbol and name.

##### `Operation`

One operation of a group: its index, its Seitz symbol and its name.

One row of an [`OperationTable`](#operationtable), `r.operations[index]`.  Records
with an entry for each operation of the group, such as the matrices of
`r.rep` and the verdicts of `r.numerical.validated`, name the
operation by this index.  `summary()` prints it.

**Held in** [`OperationTable[index]`](#operationtable).

**Exported by** `pythtqc`, `pythtqc.groups`.

**Attributes**

- **`index`** (`int`): Its index, in the order of the tables of the irreptables package.

- **`seitz`** (`str`): Its Seitz symbol `{R|t}`, written in the conventional cell used by
  the irreptables package: `{3+_001|0,0,0}`.  A trailing `'` marks
  an operation combined with time reversal.

- **`name`** (`str`): Its name in the Cartesian axes in which the pythtb Lattice is
  specified: `C3(z)`, `C3^-1(z)`, `m(x)`.  These are the axes
  `characters=` is written in.  An axis, or a mirror's normal, along
  none of `x`, `y` and `z` is named by its direction in the
  lattice vectors: `C2([1-10])`, `m([1-10])`.  A trailing `'`
  marks time reversal, as in the Seitz symbol.

- **`rotation`** (`ndarray of int, shape (3, 3)`): Its rotation, in the model's reduced coordinates.

- **`translation`** (`ndarray, shape (3,)`): Its translation, in the model's reduced coordinates.

- **`antiunitary`** (`bool`): Whether it is combined with time reversal.

- **`rotation_cartesian`** (`ndarray, shape (3, 3)`): Its rotation in the Cartesian axes in which the pythtb Lattice is
  specified.  A layer whose Lattice is specified in two dimensions has
  its normal along `z`.

- **`spin_lift`** (`ndarray, shape (2, 2), or None`): If not `None`, its matrix on a spin one-half.  For a rotation by
  `θ` about the unit vector `n` it is `e^{-iθ n·σ/2}`, with `n`
  in the same Cartesian axes and the spin along their `z`.  An
  improper operation carries the matrix of its proper part.  An
  operation combined with time reversal carries the matrix of its
  unitary part.  `None` where the group is not a double group.

**Methods**

- **`summary()`**: The operation, a line for each part, with the attribute that holds it.

#### `AtomicLimitTable`

The site irreps the model's orbitals carry, a row per Wyckoff orbit.

`r.wyckoff_orbits` of what [`realized_group`](functions.md#realized_group) returns: the
site irreps at each Wyckoff position that specify an atomic limit.
`table[j]` is row `j`, a [`WyckoffOrbit`](#wyckofforbit), the rows in the order
the orbits' first orbitals come in the model.  Iterating and `len()` go
over the rows.  `table.summary()` prints it.  A declaration the
Hamiltonian's blocks do not carry shows here, as a row whose
`determined_by` reads `"declaration not carried"` and a line under the
table.

**Held in** [`RealizedGroup.wyckoff_orbits`](#realizedgroup).

**Exported by** `pythtqc`, `pythtqc.symmetry`.

**Attributes**

- **`numerical`** (`AtomicLimitTableNumerical`): How the table was read: each declaration against the hoppings, and how
  the multiplets were found.

**Methods**

- **`summary()`**: The table, a row per Wyckoff orbit, and a line under it for what the rows omit.

##### `WyckoffOrbit`

One Wyckoff orbit of the model's orbitals, and the site irreps they carry.

A row `o` of `r.wyckoff_orbits`: the orbitals the operations of the
group carry into one another.  `o.summary()` prints it.

**Held in** [`AtomicLimitTable[j]`](#atomiclimittable).

**Exported by** `pythtqc`, `pythtqc.symmetry`.

**Attributes**

- **`wyckoff`** (`str`): The Wyckoff position, multiplicity and letter, `"2b"`, in the
  conventional cell used by the irreptables package; `""` where the
  model's cell could not be matched to that cell.

- **`site_symmetry`** (`str`): The Hermann-Mauguin symbol of the site-symmetry group, read off the
  operations measured to fix a site, with its `1'` where the group
  returned has time reversal: `"3m1'"`, and `"3"` in `P3`.

- **`orbitals`** (`tuple of int`): The model's orbitals on the orbit's sites.

- **`label`** (`str`): The site irreps one site of the orbit holds, in the site-symmetry
  group of the group returned, as printed: `"-E1"`, `"2xA1"`,
  `"-E1u + -E3u"`.  With spin they are double-valued, `-E1` for an
  `s` orbital at a `3m` site.  They are read off the characters, so
  are the same in any basis and whatever the spelling of a declaration.
  Two sites of an orbit can carry conjugate site irreps, as `B1` at
  `(1/2, 0)` and `B2` at `(0, 1/2)` in `P4mm1'`.  The row names
  them once, from the source `numerical.named_by` gives.

- **`terms`** (`tuple of groups.Term`): The same site irreps as terms, `Term(1, "-E1", "2b", "3m1'", "3m")`.

- **`unitary_site_symmetry`** (`str`): The unitary part of the site-symmetry group, `"3m"`.

- **`determined_by`** (`str`): How the site irreps are known: of the four values, listed here weakest
  first, the weakest any of the row's orbitals reads.  The hoppings fix
  the representation only up to a one-dimensional character of the group
  on each set of orbitals they couple, and `characters=` can fix it.
  `"declaration not carried"`: `characters=` stated the site irreps,
  and the Hamiltonian's own blocks do not carry them; `label` is what
  the blocks carry.  `"lowest-l"`: the choices the hoppings and the
  declarations leave open give some orbital of the row different site
  irreps, and the lowest-angular-momentum rule chose these.
  `"declaration"`: `characters=` stated them, and the matrices
  returned carry them.  `"hamiltonian"`: nothing was declared for the
  row, and every choice left open gives the same site irreps.

- **`declared`** (`str, tuple or None`): `None` where nothing was declared for the row.  Otherwise what
  `characters=` declared: one word where every orbital was declared
  the same, otherwise one entry per orbital, `None` for an orbital
  not declared.

- **`band_representation`** (`BandRepresentation`): The sum of elementary band representations the orbit induces.  It is
  written at the orbit's own Wyckoff position where a sum there carries
  the orbit's symmetry data vector, and otherwise at other positions.
  There its terms need not correspond one to one with the row's site
  irreps.  Its terms are empty where no sum is found for the orbit: the
  model's cell could not be matched to the conventional cell used by the
  irreptables package, no sum carries the symmetry data vector, several
  do and nothing read says which is induced, or the vector could not be
  identified.

- **`site_multiplets`** (`tuple of tuple of int`): For each of the orbit's sites, the orbitals of its multiplets there,
  which carry the row's site irreps together: `((0,), (1,))` for an
  orbit of two sites with one orbital each.

- **`numerical`** (`WyckoffOrbitNumerical`): How the row was read.

**Methods**

- **`summary()`**: The row, a line for each part that applies, with the attribute that holds it.

###### `Term`

An integer multiple of one site irrep at a Wyckoff position.

One term of a band representation written as a sum of band
representations induced from site irreps.  The columns of
[`ebr_data`](functions.md#ebr_data) are terms of multiplicity one, each an
elementary band representation.

**Held in** [`WyckoffOrbit.terms`](#wyckofforbit), [`BandRepresentation.terms`](#bandrepresentation), [`EBRData.ebrs`](#ebrdata), [`WyckoffOrbitSRSIsNumerical.composite`](#wyckofforbitsrsisnumerical), [`Localization.terms`](#localization), [`AtomicRealization.terms`](#atomicrealization).

**Exported by** `pythtqc.groups`.

**Attributes**

- **`multiplicity`** (`int`): The integer that multiplies the site irrep, never zero.  It is
  negative only in the signed combinations `decompose` returns.

- **`irrep`** (`str`): The site irrep, `"-E1"`, by the name the list of elementary band
  representations gives it.  A site irrep the list does not name, one
  whose induced band representation is composite, takes the name the
  tables of the stable real-space invariants give it.

- **`wyckoff`** (`str`): The Wyckoff position, with its multiplicity in the conventional cell,
  `"1a"`.

- **`site_symmetry`** (`str`): The site-symmetry group of the position in the space group whose
  band representation the term belongs to, `"4mm1'"` in a grey group
  and `"3"` in a group without time reversal.

- **`unitary_site_symmetry`** (`str`): Its unitary part, `"4mm"`.

###### `BandRepresentation`

A sum of elementary band representations, each induced from a site irrep.

The `band_representation` of each row of `r.wyckoff_orbits` is one, and
so is each candidate sum [`decompose`](functions.md#decompose) returns.  Iterating,
indexing and `len()` go over `terms`.  Two sums are equal where
their terms are.

**Held in** [`WyckoffOrbit.band_representation`](#wyckofforbit), [`EBRDecomposition.candidate_sums`](#ebrdecomposition), [`SignedCombinations.sums`](#signedcombinations), [`WyckoffOrbitNumerical.alternatives`](#wyckofforbitnumerical), [`WyckoffOrbitNumerical.contents`](#wyckofforbitnumerical).

**Exported by** `pythtqc`, `pythtqc.symmetry`.

**Attributes**

- **`terms`** (`tuple of groups.Term`): If nonempty, the terms of the sum.  Each term holds an integer
  multiplicity, a site irrep, its Wyckoff position, and its
  site-symmetry group with and without time reversal:
  `Term(1, "-E1", "2b", "3m1'", "3m")`.

- **`label`** (`str`): The sum as printed, `"-E1 at 1b (4mm1') + 2x-E2 at 1b (4mm1')"`;
  `""` where there are no terms.

###### `WyckoffOrbitNumerical`

How a row of the atomic-limit table was read.

`o.numerical` of a row `o` of `r.wyckoff_orbits`: its sites, each
orbital's characters, and how each declaration was resolved.

**Held in** [`WyckoffOrbit.numerical`](#wyckofforbit).

**Exported by** `pythtqc.symmetry`.

**Attributes**

- **`positions`** (`tuple`): Each site's position in reduced coordinates, in the order of
  `o.site_multiplets`.

- **`residual`** (`float`): The largest misfit over every multiplet of the table, the same on
  every row.  An operation that fixes a multiplet of one orbital acts on
  its states as a number, the orbital's character, times the operation's
  matrix on spin, the spin factor, which is `1` without spin.  On such
  a multiplet, or on one declared spinor pair, the misfit is how far its
  block misses that product.  On a multiplet of several orbitals it
  is how far its blocks miss multiplying as the site-symmetry group
  does, its double group with spin.  Each misfit is the largest modulus
  of an entry of a difference of unitary blocks, and one that is not
  small against `1` says the site irreps are not to be trusted.

- **`characters`** (`dict`): For each orbital, its orbital factor's character on every operation of
  its site-symmetry group, by index.  It is single-valued, with spin or
  without.  A spinor's character is zero on every mirror and twofold
  rotation, so the parity of an orbital under those operations is read
  here.

- **`named_by`** (`str`): Where the row's site irreps take their names.  `"tables"`: the
  tables of elementary band representations list the band representation
  the row's orbitals induce at this Wyckoff position, and the name is the
  site irrep those entries carry.  `"invariants"`: those tables list
  none at this position, and the tables of the stable real-space
  invariants name the site irrep the row's orbitals carry there, which is
  single-valued.  `"point group"`: neither names it, and the name is
  the one the tables give that irrep at the origin of a symmorphic space
  group with the same point group, so that one irrep has one name at
  every site.  `"Mulliken"`: those point-group names rest on an
  orientation no table fixes at this site, and the name is the irrep's
  Mulliken symbol, read off its
  characters on the site's rotations and mirrors in the Cartesian axes
  in which the pythtb Lattice is specified.  `"unnamed"`: nothing names
  the row, and `o.label` is blank.  A sum of site irreps takes the
  value of its term that stands latest in the order above.

- **`declared_irrep`** (`dict`): If nonempty, for each declared orbital, the site irreps its
  declaration resolved to, `{0: "A1"}`.

- **`declared_in`** (`dict`): If nonempty, for each declared orbital, the site-symmetry group and
  the group the declaration was read in,
  `{0: "3m in P6mm1' #183.186"}`.

- **`reduced_in`** (`dict`): If nonempty, for each declared orbital, where its declared site irreps
  lowered the group of the cell before any hopping was read, the
  site-symmetry group and the group the declaration was resolved in;
  `""` where they did not.

- **`declaration_broke`** (`dict`): If nonempty, for each declared orbital, the names of the operations
  the declared site irreps break.

- **`determined_by`** (`dict`): For each orbital, how its site irrep is known, in the values of
  `o.determined_by`.

- **`alternatives`** (`tuple of BandRepresentation`): If nonempty, the other sums of elementary band representations with
  the same symmetry data vector and the same stable real-space invariants
  as `o.band_representation`, which no quantity measured distinguishes
  from it.  Empty where the group has no tabulated invariants.

- **`contents`** (`tuple of BandRepresentation`): If nonempty, the sums of site irreps at the row's Wyckoff position
  that the copies of a pseudospin the Hamiltonian conserves on the row
  can carry, each sum once, ranked by the lowest-angular-momentum rule.
  The Hamiltonian fixes how each operation acts within one copy, and
  leaves open how it acts on the copies themselves.  Each sum comes from
  one representation of the copies: a sum of irreps of the operations
  that carry one copy's part of the crystal onto itself, whose product
  with the action within a copy multiplies as the group returned does.

- **`n_representations`** (`int`): How many representations of the copies there are, as `contents`
  defines them; several can give one sum.  `1` where no pseudospin
  is conserved.

##### `AtomicLimitTableNumerical`

How the atomic-limit table was read.

`r.wyckoff_orbits.numerical`: each declaration against the hoppings, and
how the multiplets were found.

**Held in** [`AtomicLimitTable.numerical`](#atomiclimittable).

**Exported by** `pythtqc.symmetry`.

**Attributes**

- **`checks`** (`tuple of DeclarationCheck`): If nonempty, for each declaration and each operation of its
  site-symmetry group on which the hoppings fix the declared orbitals'
  character, the character measured beside the character declared.

- **`residual`** (`float`): The largest `o.numerical.residual` over the rows `o`.

- **`site_multiplets_determined_by`** (`str`): How the multiplets each site's orbitals form were found:
  `"hamiltonian"`, measured; `"declaration"`, stated with
  `site_multiplets=`.

- **`accidental`** (`tuple of AccidentalDegeneracy`): If nonempty, an entry for each tie between two ways of joining a
  site's orbitals into multiplets, by an accidental degeneracy of H.
  Each entry names the multiplet returned.

###### `DeclarationCheck`

The character a declaration states for one operation, beside the one measured.

One entry of `r.wyckoff_orbits.numerical.checks`: an operation of the
declared orbitals' site-symmetry group on which the hoppings fix their
character relative to the orbitals they couple them to.

**Held in** [`StepDeclaration.rows`](#stepdeclaration), [`DroppedOperation.rows`](#droppedoperation), [`AtomicLimitTableNumerical.checks`](#atomiclimittablenumerical).

**Exported by** `pythtqc.symmetry`.

**Attributes**

- **`orbitals`** (`tuple of int`): The orbitals declared.

- **`index`** (`int`): The operation, by index.

- **`name`** (`str`): The operation's name, in the Cartesian axes in which the pythtb
  Lattice is specified.

- **`measured`** (`complex`): The character of the operation on the declared orbitals, as the
  hoppings give it relative to the orbitals they couple them to.

- **`declared`** (`complex`): The character the declaration states for the operation.  The hoppings
  fix the representation on the orbitals they couple only up to a
  one-dimensional character of the group.  So where the declaration
  holds, `declared` is `measured` times the value one such character
  takes on the operation.

###### `AccidentalDegeneracy`

A tie between ways of joining a site's orbitals into multiplets.

One entry of `r.wyckoff_orbits.numerical.accidental`.  The multiplets
are found by joining a multiplet with another where the matrices of an
operation that carry H onto itself reach from the one to the other.  Here
they reach from one multiplet to several others, by an accidental
degeneracy of H.  The weight of each such partner, the largest modulus of
an entry of those matrices between it and the multiplet, is at least half
the largest weight.  The multiplet could be joined with any of them.  It
is joined with the lowest-indexed, and the tie is recorded where some of
the others stay apart.

**Held in** [`AtomicLimitTableNumerical.accidental`](#atomiclimittablenumerical).

**Exported by** `pythtqc.symmetry`.

**Attributes**

- **`operation`** (`int`): The operation at which the tie arose, by index.

- **`orbitals`** (`tuple of int`): The orbitals of the multiplet, of the partner it was joined with, and
  of the partners that stay apart from it.

- **`chosen`** (`tuple of int`): The orbitals of the multiplet and the partner it was joined with.

- **`note`** (`str`): The entry, in a sentence.

#### `Descent`

The operations of the cell's group that were dropped, and why.

Held in [`RealizedGroup.descent`](#realizedgroup), which is `None` where the descent
dropped no unitary operation.  The operations a declaration excluded before
any hopping was read are recorded in `RealizedGroup.decoration.declared`,
and time reversal broken on the group in
[`RealizedGroup.time_reversal`](#realizedgroup).  A descent lowers the cell's group to
the group the Hamiltonian obeys: the group of the operations kept is
identified again, in one step or several.  An operation is dropped where
the hoppings break it, or where it does not carry the declared site irreps
or the orbital centers.  An operation the hoppings obey is dropped where
the states carry no single-valued representation of a group containing it.
It is also dropped where the operations that hold do not close into a group
and it lies outside the subgroup kept.

Where several subgroups of the largest order qualify and nothing measured
tells them apart, the tie rule keeps one: on a layer whose group contains
operations that turn it over, the one with the most operations that keep
its side; then the one with the most rotations; then the one with the
highest-order rotation; then the one whose axis comes first in the
conventional cell used by the irreptables package.

**Held in** [`RealizedGroup.descent`](#realizedgroup).

**Exported by** `pythtqc.groups`.

**Attributes**

- **`cell_group`** (`NamedGroup`): The group the cell allows, before the first step.

- **`subgroup`** (`NamedGroup`): The group the last step arrived at.  Where time reversal is then
  measured broken, the group returned is its unitary half.

- **`dropped`** (`tuple of DroppedOperation`): Every operation dropped, every step's together.

- **`dropped_by`** (`dict`): `{cause: count}`: how many operations each cause dropped, every step
  together, in the order the measurement meets the causes.  The causes
  are `"centers"`, `"hoppings"`, `"declaration"`, `"closure"`,
  `"tie"`, `"time reversal"` and `"projective class"`, each that
  dropped any.  `"hoppings"` counts the operations refuted by `"no symmetry matrix"`, `"covariance"`, `"tensor structure"`, `"out of plane"` and `"translation"`.  `"time reversal"` counts the
  operations combined with time reversal, where time reversal is
  measured broken on the group one step arrived at, before the next
  step.  The counts add up to the order of `cell_group` less that of
  `subgroup`, unless the declared site irreps lowered the cell between
  two steps.

- **`tied`** (`tuple`): If nonempty, the subgroups of the largest order that carry the
  declared site irreps, every step's together, each by its operations'
  indices, in the order of the tie rule, the one kept first.

- **`cell`** (`DescentCell`): The orbit the last step added to the cell.

- **`iterations`** (`int`): The number of steps.

- **`steps`** (`tuple of DescentStep`): The steps, in order.

- **`lattice`** (`DescentLattice or None`): If not `None`, what was measured of the pure translations, from the
  first step whose cell is not primitive.

- **`note`** (`str`): The descent in words: what dropped the operations, counted by cause,
  and the group arrived at.  Where every operation broken was refuted by
  its tensor structure, it also names the least `precision=` under
  which all of them, and the group of the cell, would hold, where one
  does.

##### `NamedGroup`

A space group by its symbol, its number and its number of operations.

`str()` writes it as the symbol and the number, `"P6mm1' #183.186"`.

**Held in** [`Descent.cell_group`](#descent), [`Descent.subgroup`](#descent), [`DescentStep.cell_group`](#descentstep), [`DescentStep.subgroup`](#descentstep), [`ProjectiveReading.group`](#projectivereading), [`CrossCheck.free_standing`](#crosscheck), [`CrossCheck.substrate`](#crosscheck), [`DeclaredReduction.cell_group`](#declaredreduction), [`DeclaredReduction.subgroup`](#declaredreduction), [`SymmetryIndicatorNumerical.free_standing_group`](#symmetryindicatornumerical), [`StackInput.group`](#stackinput), [`StackGroup.group`](#stackgroup).

**Exported by** `pythtqc.groups`.

**Attributes**

- **`name`** (`str`): The group's Hermann-Mauguin symbol, `"Fd-3m1'"`.

- **`number_str`** (`str`): The group's number in the tables of the irreptables package,
  `"227.129"`.

- **`size`** (`int`): The number of its operations, one per coset of the lattice
  translations.

##### `DroppedOperation`

One operation a descent dropped, and what dropped it.

A field that does not apply to what dropped the operation is `None`.  A
matrix `U` carries `H(k)` onto `H(gk)`, for `g` the operation,
where `U H(k) U^-1 = H(gk)`, with `H(k)*` in place of `H(k)` for an
operation combined with time reversal.  The operation's solution space is
the set of such matrices at the k measured.  Its covariance residual, for the
best unitary `U` found, is the largest entry of `U H(k) U^-1 - H(gk)`
over the k measured, in the units of H, and it is held to the tolerance
`r.numerical.validated.tol`.  The operation's form is a matrix that
carries each site's orbitals onto the orbitals of the site it moves them
to, times the operation's own spin rotation where the states carry spin.

**Held in** [`Descent.dropped`](#descent), [`DescentStep.dropped`](#descentstep).

**Exported by** `pythtqc.groups`.

**Attributes**

- **`index`** (`int`): The operation, by index in the group it was dropped from.

- **`name`** (`str`): Its name, in the Cartesian axes in which the pythtb Lattice is
  specified.  Where its translation is not a lattice vector of the cell,
  as for a screw rotation, a glide or a pure translation, the name
  carries the translation in Seitz form, `{E|1/2,0,0}`.

- **`refuted_by`** (`str`): What dropped it.  `"no symmetry matrix"`: no matrix carries `H(k)`
  onto `H(gk)`.  `"covariance"`: such matrices exist, and the
  covariance residual is above the tolerance.  `"tensor structure"`: no
  matrix of the operation's form carries `H(k)` onto `H(gk)` within
  the tolerance.  A pure translation acts on every k as the identity
  does, so a matrix always carries `H(k)` onto `H(k)`, and one this
  test refutes reads `"tensor structure"`.  `"out of plane"`: an
  operation of the slab a layer is read as ([`Presentation`](#presentation)) that
  turns the layer's plane out of itself, so that it has no action on the
  layer's k.  `"translation"`: a pure translation whose permutation of
  the sites, with or without a change of basis within each site, does not
  commute with `H(k)`.  `"declaration"`: it is outside the largest
  group of the operations still standing that carries the declared site
  irreps.  `"centers"`: it is outside the largest group that carries
  the orbital centers within the tolerance declared for them.
  `"projective class"`: the hoppings obey it, and the states carry no
  single-valued representation of a group containing it and the
  operations kept.  `"closure"`: the operations measured to hold do not
  close into a group, and no subgroup of them of the largest order
  contains it.  `"tie"`: several subgroups of the operations measured
  to hold have the largest order, and the one the tie rule
  ([`Descent`](#descent)) keeps does not contain it.  For the declaration and
  the centers, an operation combined with time reversal is dropped where
  its unitary part is.

- **`residual`** (`float`): The number the measurement rests on.  For `"covariance"`, the
  covariance residual, in the units of H.  For `"translation"`, the
  same with `U` the permutation of the sites.  For
  `"tensor structure"`, how far the member of the solution space
  nearest the operation's form lies outside that form, over the member's
  own size, dimensionless.
  `inf` for `"no symmetry matrix"` and `"out of plane"`; `nan`
  for the other causes, which rest on no residual.

- **`step`** (`int or None`): The descent step that dropped it, counted from 1.

- **`of`** (`str or None`): The number of the group it was dropped from, in the tables of the
  irreptables package.

- **`note`** (`str or None`): If not `None`, why it was dropped, in words.

- **`orbitals`** (`tuple of int or None`): Where the declared site irreps dropped it, the orbitals declared.

- **`declared`, `measured`** (`complex or None`): Where the declared site irreps dropped it, the character declared and
  the one the hoppings measure.

- **`miss`** (`float or None`): Where the centers dropped it, the farthest a center must move for the
  operations this one generates to carry the centers onto one another,
  in reduced coordinates.

- **`precision`** (`float or None`): Where the centers dropped it, the tolerance declared for the centers,
  `position_tol=` or `precision=`, in reduced coordinates.

- **`covariance`** (`float or None`): Where the member of the solution space nearest the operation's form
  misses that form by more than rounding, the covariance residual of its
  part in the form, in the units of H.

- **`pair`, `names`** (`tuple or None`): Where the states' projective class dropped it, two commuting
  operations whose matrices on the states do not commute, by index and
  by name.

- **`commutation_phase`** (`complex or None`): The phase `c` in `D(a) D(b) = c D(b) D(a)`, with `a` and `b`
  that pair and `D` their matrices.

- **`spin_half`** (`bool or None`): Whether `c` is also the phase between `S(a) S(b)` and
  `S(b) S(a)`, with `S` the operations' spin-1/2 rotations, as for
  spinor states read without their spin.

- **`rows`** (`tuple of symmetry.DeclarationCheck or None`): Where the declared site irreps dropped it, each declaration checked
  against it, or against its unitary part for an operation combined
  with time reversal: the character declared and the one the hoppings
  measure.  `None` for the other causes.

##### `DescentCell`

The orbit a descent step adds to the cell so that its group is the group kept.

A descent lowers the cell's group to the group the Hamiltonian obeys, in
one step or several ([`Descent`](#descent)).  At each step the orbit of a
generic point under the operations kept is added to the cell as atoms of a
new species, and the cell's group is found again.

**Held in** [`Descent.cell`](#descent), [`DescentStep.cell`](#descentstep).

**Exported by** `pythtqc.groups`.

**Attributes**

- **`generic_point`** (`tuple of float`): The generic point, in reduced coordinates.

- **`orbit`** (`tuple of tuple of float`): The points of its orbit, in reduced coordinates.

- **`species`** (`str`): The species the points are added as.

##### `DescentStep`

One step of a descent: the group lowered, the group it arrived at, and why.

**Held in** [`Descent.steps`](#descent).

**Exported by** `pythtqc.groups`.

**Attributes**

- **`step`** (`int`): The step's number, counted from 1.

- **`cell_group`, `subgroup`** (`NamedGroup`): The group the step lowered, and the group it arrived at.

- **`dropped`** (`tuple of DroppedOperation`): The operations the step dropped.

- **`n_dropped`** (`int`): The number of operations the step dropped.

- **`refuted_by`** (`dict`): `{refuted_by: count}`: the number of operations dropped for each
  value of [`DroppedOperation.refuted_by`](#droppedoperation).

- **`kept`** (`tuple of int`): The unitary operations kept, by index in `cell_group`.

- **`pairs`** (`tuple`): Each unitary operation kept, as its index in `cell_group` and its
  index in `subgroup`.

- **`cell`** (`DescentCell`): The orbit added to the cell so that its group is the group kept.

- **`declaration`** (`StepDeclaration or None`): If not `None`, what the declared site irreps kept.

- **`centers`** (`StepCenters or None`): If not `None`, what the orbital centers kept, where the whole of
  `cell_group` does not carry them within the tolerance declared for
  them.

- **`tie`** (`StepTie or None`): If not `None`, the subgroups of the largest order where the
  operations measured to hold do not close.

- **`projective`** (`StepProjective or None`): If not `None`, what the states' projective class kept, where it
  lowered the group.

- **`lattice`** (`DescentLattice or None`): If not `None`, what was measured of the pure translations of a cell
  that is not primitive.

###### `StepDeclaration`

What the declared site irreps kept at one descent step.

Held in [`DescentStep.declaration`](#descentstep) where the declaration made with
`characters=` is read at the step.

**Held in** [`DescentStep.declaration`](#descentstep).

**Exported by** `pythtqc.groups`.

**Attributes**

- **`subgroup`** (`tuple of int`): The operations the declared site irreps keep, by index in the step's
  `cell_group`.

- **`tied`** (`tuple`): If nonempty, the subgroups of the largest order that carry the
  declared site irreps, each by its operations' indices.  They stand in
  the order of the tie rule ([`Descent`](#descent)), the one kept first.

- **`rows`** (`tuple of symmetry.DeclarationCheck`): The declared site irreps checked against each operation measured.

- **`names`** (`tuple or None`): If not `None`, the tied subgroups, each by its operations' names.

- **`note`** (`str or None`): If not `None`, which tied subgroup was kept and why, in words.

###### `StepCenters`

What the orbital centers kept at one descent step.

Held in [`DescentStep.centers`](#descentstep) where the whole group of the step does
not carry the orbital centers within the tolerance declared for them.  A
group carries the centers when every center can be moved onto a position
symmetric under that group by no more than that tolerance.

**Held in** [`DescentStep.centers`](#descentstep).

**Exported by** `pythtqc.groups`.

**Attributes**

- **`precision`** (`float`): The tolerance declared for the centers, `position_tol=` or
  `precision=`, in reduced coordinates.

- **`move`** (`float`): The farthest a center must move for the centers to be symmetric under
  the whole of the step's `cell_group`, in reduced coordinates.  It is
  larger than `precision`.

- **`kept`** (`tuple of int`): The operations kept, by index in `cell_group`: the largest subgroup
  that carries the centers.  Each operation combined with time reversal
  whose unitary part is in that subgroup is kept with it.

- **`tied`** (`tuple`): If nonempty, the subgroups of the largest order that carry the
  centers, each by its operations' indices.  They stand in the order of
  the tie rule ([`Descent`](#descent)), the one kept first.

- **`note`** (`str`): The group kept and the operations it leaves out, in words.

###### `StepTie`

The subgroups of the largest order inside operations that do not close.

Held in [`DescentStep.tie`](#descentstep).  The operations measured to hold at the
step do not close into a group, and several of their subgroups have the
largest order.  The tie rule ([`Descent`](#descent)) keeps one of them.

**Held in** [`DescentStep.tie`](#descentstep).

**Exported by** `pythtqc.groups`.

**Attributes**

- **`tied`** (`tuple`): Those subgroups, each by its operations' indices, in the order of the
  tie rule.

- **`kept`** (`tuple`): The subgroup kept, the first of them.

- **`note`** (`str`): Which subgroup was kept and why, in words.

###### `StepProjective`

What the states' projective class kept at one descent step.

Held in [`DescentStep.projective`](#descentstep) where the class lowered the group:
no single-valued representation of the step's group exists on the states,
as for spinor states read without their spin.

**Held in** [`DescentStep.projective`](#descentstep).

**Exported by** `pythtqc.groups`.

**Attributes**

- **`pairs`** (`tuple of ProjectivePair`): Each pair of commuting operations whose matrices do not commute.

- **`kept`** (`tuple of int`): The operations kept, by index: the largest subgroup on which the
  states carry a single-valued representation.

- **`tied`** (`tuple`): If nonempty, the subgroups of that order on which they carry one, each
  by its operations' indices.  They stand in the order of the tie rule
  ([`Descent`](#descent)), the one kept first.

- **`rule`, `reason`** (`str or None`): If not `None`, the tie rule, and why it kept the first subgroup.

- **`horizontal_mirror`** (`ProjectiveMirror or None`): If not `None`, the sign chosen for a flat layer's horizontal mirror.

- **`read_on`** (`ProjectiveReading or None`): If not `None`, the free-standing layer's group the class was read
  on, for a layer on a substrate.

###### `ProjectivePair`

Two commuting operations whose matrices on the states do not commute.

Such a pair shows that the states carry no single-valued representation
of a group that contains both.

**Held in** [`StepProjective.pairs`](#stepprojective).

**Exported by** `pythtqc.groups`.

**Attributes**

- **`pair`** (`tuple of int`): The two operations, by index.

- **`names`** (`tuple of str`): Their names.

- **`commutation_phase`** (`complex`): The phase `c` in `D(a) D(b) = c D(b) D(a)`, with `a` and `b`
  the two operations and `D` their matrices.  No rescaling of either
  matrix removes it.

- **`spin_half`** (`bool`): Whether `c` is also the phase between `S(a) S(b)` and
  `S(b) S(a)`, with `S` the operations' spin-1/2 rotations, as for
  spinor states read without their spin.

###### `ProjectiveMirror`

The sign chosen for a flat layer's horizontal mirror, read without spin.

On a flat layer an operation `g` and its product with the horizontal
mirror, `m g`, act alike on k and share one measurement, which fixes the
mirror's matrix only up to a sign.  The matrix of `m g` is taken as that
sign times the matrix of `g`.  The sign commutes with every matrix, so
the projective class does not depend on it.

**Held in** [`StepProjective.horizontal_mirror`](#stepprojective).

**Exported by** `pythtqc.groups`.

**Attributes**

- **`sign`** (`int`): `+1` or `-1`.

- **`determined_by`** (`str`): `"declaration"`, the parity the declared site irreps state;
  `"lowest-l"`, the even sign the lowest-angular-momentum rule
  assigns.

###### `ProjectiveReading`

The group of a layer free-standing, on which its class on a substrate is read.

For a layer on a substrate, the states' projective class is read on the
group of the same layer free-standing, and the part of that reading that
does not turn the layer over is carried onto the substrate.

**Held in** [`StepProjective.read_on`](#stepprojective).

**Exported by** `pythtqc.groups`.

**Attributes**

- **`group`** (`NamedGroup`): The free-standing layer's group.

- **`kept`** (`int`): The number of its operations in the largest subgroup on which the
  states carry a single-valued representation.

##### `DescentLattice`

What was measured of the pure translations of a cell that is not primitive.

A cell is not primitive where its orbital centers, with their species, are
carried onto themselves by a translation that is not a lattice vector of
the model.

**Held in** [`Descent.lattice`](#descent), [`DescentStep.lattice`](#descentstep).

**Exported by** `pythtqc.groups`.

**Attributes**

- **`index`** (`int`): The index of the model's lattice in the lattice of the pure
  translations the cell allows: 2 where it allows one pure translation.

- **`fractional_translations`** (`tuple`): The pure translations the cell allows, in reduced coordinates, the
  identity left out.

- **`operations`** (`tuple of LatticeTranslation`): Each pure translation, and what was measured of it.

- **`realized`** (`tuple`): The translations the Hamiltonian obeys.  Empty on every record
  returned: a Hamiltonian that obeys a pure translation of its cell is
  written on a supercell, and is refused `supercell`.

- **`dressing`** (`tuple of TranslationDressing`): For each translation in `realized`, the matrix it is obeyed with.
  Empty on every record returned, as `realized` is.

- **`refuted`** (`dict`): If nonempty, `{index: RefutedTranslation}`: the pure translations
  the Hamiltonian breaks that nothing else refuted.

- **`coset_partners_refuted`** (`int`): The number of the cell's unitary operations refuted at the step that
  are not pure translations.

- **`note`** (`str`): What was measured, in words.

###### `LatticeTranslation`

One pure translation of a cell that is not primitive, and what was measured.

**Held in** [`DescentLattice.operations`](#descentlattice).

**Exported by** `pythtqc.groups`.

**Attributes**

- **`index`** (`int`): The operation, by index in the cell's group.

- **`translation`** (`tuple of float`): The translation, in reduced coordinates.

- **`cartesian`** (`tuple of float`): The translation, in the Cartesian axes in which the pythtb Lattice is
  specified.

- **`refuted_by`** (`str or None`): What refuted it, in the words of [`DroppedOperation.refuted_by`](#droppedoperation);
  `None` where the Hamiltonian obeys it.

- **`scaled`** (`float`): Where its tensor structure refuted it, the dimensionless number
  [`DroppedOperation.residual`](#droppedoperation) holds for `"tensor structure"`;
  `nan` otherwise.

- **`residual`** (`float or None`): The largest entry of `P H(k) P^-1 - H(k)` over the k-points measured,
  with `P` the permutation of the sites the translation makes, in the
  units of H.  `None` where two sites the translation pairs hold
  different numbers of orbitals.

- **`tol`** (`float`): The tolerance the residual is held to, in the units of H.

###### `TranslationDressing`

A pure translation the Hamiltonian obeys, and the matrix it is obeyed with.

**Held in** [`DescentLattice.dressing`](#descentlattice).

**Exported by** `pythtqc.groups`.

**Attributes**

- **`translation`** (`tuple of float`): The translation, in reduced coordinates.

- **`turn`** (`float`): How far the matrix turns each site's orbitals as it carries them onto
  its translate's, in radians: half the smallest arc of the unit circle
  that holds every eigenvalue of its blocks, so that a phase common to
  all of them is no turn.  0 where it carries them as written.

- **`matrix`** (`ndarray`): The matrix, carrying each site's orbitals onto its translate's.

###### `RefutedTranslation`

A pure translation the Hamiltonian breaks, which nothing else refuted.

Its permutation of the sites, with or without a change of basis within
each site, does not commute with `H(k)`, and no other measurement
refuted it.

**Held in** [`DescentLattice.refuted`](#descentlattice).

**Exported by** `pythtqc.groups`.

**Attributes**

- **`refuted_by`** (`str`): `"translation"`.

- **`residual`** (`float`): The largest entry of `P H(k) P^-1 - H(k)` over the k-points
  measured, with `P` the permutation of the sites the translation
  makes, in the units of H.

- **`scaled`, `defect`** (`float`): The same residual.

- **`scale`** (`float`): The tolerance the residual is held to, in the units of H.

- **`orientation`, `dim`** (`int`): Always 0.

#### `TimeReversal`

How time reversal was measured.

Held in [`RealizedGroup.time_reversal`](#realizedgroup).  The group the unitary
operations settle on is taken as a grey group, since positions say nothing
about time reversal.  Each of its operations combined with time reversal
is then measured on `H`.

**Held in** [`RealizedGroup.time_reversal`](#realizedgroup).

**Exported by** `pythtqc.groups`.

**Attributes**

- **`outcome`** (`str`): `"kept"`: every operation combined with time reversal holds, and the
  group returned is grey.  `"measured broken"`: time reversal does not
  hold, and the group returned is the unitary group.

- **`n_antiunitary`, `n_obeyed`, `n_broken`** (`int or None`): The number of operations combined with time reversal in the grey group
  measured, and how many of them hold and how many do not.  Where time
  reversal is measured broken, an operation combined with it whose
  unitary part is in the group returned does not hold, so `n_obeyed`
  is 0 unless a later descent by the states' projective class left that
  unitary part out.

- **`undetermined`** (`tuple or None`): If nonempty, the indices of those that hold and share their
  measurement on `H(k)` with another operation acting alike on every
  k, as an operation and its product with the horizontal mirror do on a
  flat layer.  Each of them is held by its tensor structure, the form of
  its matrix on the orbitals and on the spin.

- **`grey_name`, `grey_number_str`** (`str or None`): The grey group time reversal was measured on, its symbol and its
  number.

- **`square`** (`TimeReversalSquare or None`): If not `None`, the square of time reversal on the states.  `None`
  where the measurement found no matrix for time reversal.

- **`note`** (`str or None`): The measurement in words.

- **`name`, `number_str`** (`str or None`): Where time reversal is measured broken, the unitary group returned,
  its symbol and its number.

- **`size`** (`int or None`): Where time reversal is measured broken, the number of operations of
  the unitary group returned.

##### `TimeReversalSquare`

The square of time reversal on the states, expected and measured.

With `X` the matrix of time reversal, its square is `X conj(X)`.  Where
the square is a multiple of the identity, its sign is one no rescaling of
`X` changes.  Where the sign measured differs from the sign expected,
time reversal is broken, as for spinor states read without their spin.

**Held in** [`TimeReversal.square`](#timereversal).

**Exported by** `pythtqc.groups`.

**Attributes**

- **`asked`** (`int`): The sign the tables of the irreptables package expect: `+1` for the
  single-valued tables, `-1` for the double-valued ones.

- **`measured`** (`float`): The sign of the square of the Hamiltonian's matrix of time reversal;
  `nan` where that square is not a multiple of the identity.

#### `BeyondGrey`

Antiunitary operations that hold while time reversal itself is broken.

Held in [`RealizedGroup.beyond_grey`](#realizedgroup), which is `None` where none
holds.  Each combines time reversal with a unitary operation of the cell
that a descent left out, so none is in the group returned, and the
Hamiltonian's magnetic group is larger than the group returned.
`magnetic_type` names the type of a magnetic group whose antiunitary
operations they are.  An operation combined with time reversal holds only
where a matrix measured for it and the matrices of the group returned, up
to a one-dimensional character of that group, make a corepresentation of
the magnetic group it generates with that group.

**Held in** [`RealizedGroup.beyond_grey`](#realizedgroup).

**Exported by** `pythtqc.groups`.

**Attributes**

- **`operations`** (`tuple of BeyondGreyOperation`): The operations that hold.

- **`n_operations`, `n_measured`** (`int`): The number that hold, and the number of operations of the cell
  combined with time reversal that were measured.

- **`magnetic_type`** (`str`): `"III"` where none of them is a translation combined with time
  reversal, and `"IV"` where one is.

- **`closed`** (`bool`): Whether the operations the Hamiltonian was measured to obey in the
  group these were measured in, unitary and antiunitary together, form
  a group.

- **`note`** (`str`): What they are, in words.

##### `BeyondGreyOperation`

One antiunitary operation that holds while time reversal itself is broken.

**Held in** [`BeyondGrey.operations`](#beyondgrey).

**Exported by** `pythtqc.groups`.

**Attributes**

- **`index`** (`int`): The operation, by index in the group it was measured in.

- **`name`** (`str`): Its name, in the Cartesian axes in which the pythtb Lattice is
  specified, with a `'` for time reversal.  Where its translation is
  not a lattice vector of the cell, as for a screw rotation, a glide or
  a pure translation, the name carries the translation in Seitz form,
  `{E|1/2,0,0}'`.

- **`of`** (`str`): The number of the group it was measured in, in the tables of the
  irreptables package.

- **`rotation`, `translation`** (`tuple`): Its rotation matrix and its translation, in the cell's reduced
  coordinates.

- **`residual`** (`float`): Its covariance residual, as [`DroppedOperation`](#droppedoperation) defines it, in
  the units of H.

- **`verdict`** (`str`): `"obeyed"`; or `"undetermined"`, where it shares its measurement
  on `H(k)` with another operation acting alike on every k, and is
  held by its tensor structure.

#### `Conserved`

Symmetry beyond the crystal's, counted by the matrices commuting with H(k).

Returned by [`RealizedGroup.conserved`](#realizedgroup).  The little-group
representations, elementary band representations and symmetry indicators
returned are those of the space group, whatever `dim` is.  Where `dim`
is above 0, the Hamiltonian may not fix the site irreps of orbitals it
decouples from the rest.  Each row's `determined_by` in
[`RealizedGroup.wyckoff_orbits`](#realizedgroup) says whether a declaration or the
lowest-angular-momentum rule fixed them.

**Held in** [`RealizedGroup.conserved`](#realizedgroup).

**Exported by** `pythtqc.groups`.

**Attributes**

- **`dim`** (`int`): The number of matrices beyond the identity that commute with H(k) at
  every k: 0 for a generic model, and 1 for Kane-Mele without its Rashba
  term, which conserves `s_z`.

- **`note`** (`str`): The same number, in a sentence.

- **`sectors`** (`tuple of ConservedSector or None`): If not `None`, the sectors the conserved matrices acting within
  sites split the states into.  They are read once, where the model
  enters, at the declared precision, and a spin rotation is never among
  the matrices read.  `None` where they could not be read.

- **`images`** (`dict or None`): If nonempty, the sector each operation carries each sector onto,
  `{index: images}`, with `images[i]` the image of sector `i`.
  `None` where `sectors` is `None`.

##### `ConservedSector`

One sector the conserved matrices acting within sites split the states into.

**Held in** [`Conserved.sectors`](#conserved).

**Exported by** `pythtqc.groups`.

**Attributes**

- **`orbitals`** (`tuple of int`): The orbitals its states lie on.

- **`sites`** (`tuple of int`): The sites those orbitals sit at.

- **`copies`, `dim`** (`int`): The sector holds `copies` identical copies of a set of `dim`
  states, counted as orbitals, or as spin pairs where `spin=` declares
  them; the spin of a model built with spin is not counted.  The
  conserved matrices on it are every matrix acting on the copies, times
  the identity on each copy's states.  A sector is a block of the states
  in whatever basis splits them, not necessarily a set of orbitals.
  Graphene's two orbitals are one sector of one copy.  Two orbitals on
  one site with the same hoppings, which no hopping joins, are one sector
  of two copies.  A `(p_x, p_y)` pair whose two hoppings differ, and
  which no hopping couples, is two sectors.

#### `LayerEnvironment`

A layer's environment, and the other environment measured beside it.

Held in [`RealizedGroup.environment`](#realizedgroup), which is `None` for a crystal
periodic in three directions.  On a substrate, no operation turns the layer
over.  Free-standing, the whole layer group applies.

**Held in** [`RealizedGroup.environment`](#realizedgroup).

**Exported by** `pythtqc.groups`.

**Attributes**

- **`kind`** (`str`): The environment the layer was read in, `"substrate"` or
  `"free-standing"`.

- **`declared`** (`bool`): Whether `environment=` gave it.  Where it did not, a layer written in
  two dimensions is read on a substrate, and a slab written in three
  dimensions is read free-standing.

- **`cross_check`** (`CrossCheck or None`): If not `None`, the other environment, measured and compared.
  `None` with `cross_check=False`.

- **`presentation`** (`Presentation`): How the layer was written as a slab.

##### `CrossCheck`

A layer's other environment, measured and compared with the one asked for.

Held in [`LayerEnvironment.cross_check`](#layerenvironment), which is `None` with
`cross_check=False`.  The layer is read on a substrate and free-standing,
each on its own.  The two groups agree where the substrate group is the
subgroup of the free-standing group made of the operations that do not
turn the layer over.

**Held in** [`LayerEnvironment.cross_check`](#layerenvironment).

**Exported by** `pythtqc.groups`.

**Attributes**

- **`free_standing`** (`NamedGroup or None`): The group of the layer free-standing.  `None` where that is the
  other environment and it refused or was not compared.

- **`substrate`** (`NamedGroup or None`): The group of the layer on a substrate.  `None` where that is the
  other environment and it refused or was not compared.

- **`agrees`** (`bool or None`): `True` where the two groups agree, and `None` otherwise; never
  `False`.  A disagreement that the tie rule does not account for is
  not returned.

- **`outcome`** (`str or None`): `"agree"`: the two groups agree.  `"ambiguous"`: in one
  environment several subgroups of the largest order qualified, and the
  tie rule ([`Descent`](#descent)) kept the first of them.  The groups
  returned do not agree, and another pair of candidates, one from each
  environment, does, which `note` names.  `"refused"`: the other
  environment refused.  `"not compared"`: `characters=` names site
  irreps the other environment's site groups do not have, so the other
  environment is not read.  `None` where the free-standing cell and
  the substrate's cell do not have the same single non-periodic
  direction.

- **`rows`** (`tuple of CrossCheckRow`): If nonempty, each operation of the substrate group, paired with the
  same operation in the free-standing group.  Empty where the two
  groups were not compared.

- **`missing`** (`tuple of CrossCheckOperation`): If nonempty, the operations of the substrate group that the
  free-standing group does not contain.

- **`extra`** (`tuple of CrossCheckOperation`): If nonempty, the operations of the free-standing group that do not
  turn the layer over and that the substrate group does not contain.

- **`flipping`** (`tuple of CrossCheckOperation`): If nonempty, the operations of the substrate group that turn the layer
  over.

- **`unmapped`** (`tuple of CrossCheckOperation`): The operations of either group whose rotation mixes the layer's plane
  with its non-periodic direction.  Empty on every record returned:
  every operation of a group returned keeps the layer's plane.

- **`tie`** (`tuple of CrossCheckOperation`): If nonempty, the operations the tie rule left out in an environment
  where it chose the group.

- **`refusal`** (`CrossCheckRefusal or None`): If not `None`, the refusal the other environment returned.

- **`note`** (`str`): The comparison in words.

###### `CrossCheckRow`

One operation of the substrate group, and its index in the free-standing one.

**Held in** [`CrossCheck.rows`](#crosscheck).

**Exported by** `pythtqc.groups`.

**Attributes**

- **`substrate`** (`int`): The operation, by index in the substrate group.

- **`free_standing`** (`int or None`): The same operation, by index in the free-standing group; `None`
  where that group does not contain it.

- **`name`** (`str`): Its name, in the Cartesian axes in which the pythtb Lattice is
  specified.

- **`rotation`, `translation`** (`tuple`): Its integer rotation matrix, written row by row as nine entries, and
  its translation modulo the lattice, in the cell's reduced coordinates.

- **`antiunitary`** (`bool`): Whether it is combined with time reversal.

###### `CrossCheckOperation`

One operation named in comparing a layer on a substrate with it free-standing.

It stands in one of the comparison's lists: an operation missing from a
group or extra in it, one that turns the layer over, one that is not a
layer operation, or one the tie rule left out.

**Held in** [`CrossCheck.missing`](#crosscheck), [`CrossCheck.extra`](#crosscheck), [`CrossCheck.flipping`](#crosscheck), [`CrossCheck.unmapped`](#crosscheck), [`CrossCheck.tie`](#crosscheck).

**Exported by** `pythtqc.groups`.

**Attributes**

- **`index`** (`int`): The operation, by index in its group.

- **`name`** (`str`): Its name, in the Cartesian axes in which the pythtb Lattice is
  specified.

- **`group`** (`str or None`): If not `None`, the group it is in, `"substrate"` or
  `"free-standing"`, where the list it stands in does not say.

- **`environment`** (`str or None`): If not `None`, the environment in which the tie rule left it out.

###### `CrossCheckRefusal`

The refusal returned where the layer was read in the other environment.

Held in [`CrossCheck.refusal`](#crosscheck).  A refusal there is not a mismatch
between the two environments.  The group of the environment asked for is
still returned.

**Held in** [`CrossCheck.refusal`](#crosscheck).

**Exported by** `pythtqc.groups`.

**Attributes**

- **`gate`** (`str`): The name of the check that stopped the reading of the other
  environment.

- **`reason`** (`str`): The refusal's reason.

- **`environment`** (`str`): The other environment, `"substrate"` or `"free-standing"`.

##### `Presentation`

How a layer was written as a slab, and the check that `H(k)` is unchanged.

Held in [`LayerEnvironment.presentation`](#layerenvironment).  A layer is read as a slab:
the layer in a three-dimensional cell, with vacuum along its normal.  The
slab's `H(k)` is measured against the layer's.

**Held in** [`LayerEnvironment.presentation`](#layerenvironment).

**Exported by** `pythtqc.groups`.

**Attributes**

- **`c`** (`float`): The least length of the slab's lattice vector along the layer's
  normal, in the model's length units.  A layer written in two
  dimensions is given a vector of this length, and a slab's shorter
  vector is stretched to it.

- **`residual`** (`float`): The largest difference between the slab's `H(k)` and the layer's,
  over `n_check` random `k`, in the units of `H`.  Where
  `tilted` is not `None`, the layer's `H(k)` is first written in
  the basis of the moved orbitals.

- **`n_check`** (`int`): The number of random `k` at which `residual` is measured.

- **`tilted`** (`PresentationTilt or None`): If not `None`, how a slab's non-periodic lattice vector, tilted off
  the layer's normal, was put on the normal.  `None` where the vector
  was on the normal already, and for a layer written in two dimensions.

###### `PresentationTilt`

A slab's tilted non-periodic lattice vector, put on the layer's normal.

Held in [`Presentation.tilted`](#presentation).  A layer's group does not depend on
how that vector is chosen, since there is no translation along it.  Every
orbital keeps its Cartesian position, so its reduced coordinates in the
plane move by its reduced height times the tilt.  The move is a
`k`-dependent change of basis, which leaves every energy unchanged.

**Held in** [`Presentation.tilted`](#presentation).

**Exported by** `pythtqc.groups`.

**Attributes**

- **`tilt`** (`tuple of float`): The vector's in-plane part, in reduced coordinates of the periodic
  directions.

- **`max_move`** (`float`): The largest move of an orbital's reduced coordinates in the plane.

- **`n_moved`** (`int`): The number of orbitals moved, those off the plane through the origin.

- **`spectrum_residual`** (`float`): The largest difference between the spectra before and after the move,
  over the `k` at which [`Presentation.residual`](#presentation) is measured, in
  the units of `H`.

- **`note`** (`str`): The move in words.

#### `Decoration`

The points added to the cell before its group was found.

Held in [`RealizedGroup.decoration`](#realizedgroup), which is `None` for a crystal
periodic in three directions to which nothing was added.  On a substrate,
one point of a species of its own is added under every site of the layer,
which removes every operation that turns the layer over.
`decorations=` adds its points, and a declaration that lowers the cell's
group adds an orbit.

**Held in** [`RealizedGroup.decoration`](#realizedgroup).

**Exported by** `pythtqc.groups`.

**Attributes**

- **`dz`** (`float or None`): If not `None`, the depth of the substrate's points under the sites,
  in reduced coordinates of the lattice vector normal to the layer.
  `None` where no substrate was added.

- **`typat`** (`tuple`): The species label of each point added.

- **`n_atoms`** (`int`): The number of points added.

- **`declared`** (`DeclaredReduction or None`): If not `None`, the cell's group lowered by the declaration of the
  orbitals.  `None` where the declaration takes no operation out.

- **`recipe`** (`str`): What was added and why, in words.

##### `DeclaredReduction`

The cell's group, lowered by the declaration of the orbitals.

Held in [`Decoration.declared`](#decoration), which is `None` where the
declaration takes no operation out.  The operations of the cell that do not
carry what `characters=` declares are not operations of the declared
crystal.  Before any hopping is read, the orbit of a generic point under
the operations kept is added to the cell, and the cell's group is found
again.  No operation taken out here is among those of
[`RealizedGroup.descent`](#realizedgroup).  Where the declaration states every site of
an orbit, the operations kept are every operation that carries it.
Elsewhere they can be fewer, and a site left undeclared carries what the
operations kept carry onto it.

**Held in** [`Decoration.declared`](#decoration).

**Exported by** `pythtqc.groups`.

**Attributes**

- **`cell_group`** (`NamedGroup`): The group the cell allows.

- **`subgroup`** (`NamedGroup`): The group the cell is identified as once the operations are taken out.

- **`excluded`** (`tuple of ExcludedOperation`): Each operation taken out, with the reason.

- **`n_excluded`** (`int`): The number of operations taken out, one for each coset of the lattice
  translations.

- **`kept`** (`tuple of int`): The operations kept, by index in `cell_group`.  They are the largest
  group whose every operation carries the declared site irreps, the
  first by the tie rule ([`Descent`](#descent)) where several of that order
  tie.

- **`generic_point`** (`tuple of float`): The generic point added to the cell, in reduced coordinates.

- **`orbit`** (`tuple`): The orbit of that point under the operations kept, each point in
  reduced coordinates.

- **`species`** (`str`): The species label the orbit is added under.

- **`pairs`** (`tuple`): Each unitary operation kept, as its index in `cell_group` and its
  index in `subgroup`.

###### `ExcludedOperation`

One operation of the cell that the declaration of the orbitals takes out.

Held in [`DeclaredReduction.excluded`](#declaredreduction).  The operation is not one of
the crystal `characters=` declares.

**Held in** [`DeclaredReduction.excluded`](#declaredreduction).

**Exported by** `pythtqc.groups`.

**Attributes**

- **`index`** (`int`): The operation, by index in the cell's group.

- **`name`** (`str`): Its name, in the Cartesian axes in which the pythtb Lattice is
  specified.

- **`reason`** (`str`): Why the operation is taken out, in words: it carries a site onto one
  declared differently; it does not carry the declaration at a site it
  fixes onto itself; it does not carry a declared function onto itself,
  or onto the declaration at the site it moves it to; a product of it
  with an operation kept does not carry the declaration; or a group
  containing it carries the declaration, and the tie rule
  ([`Descent`](#descent)) keeps another of the same order.

- **`sites`** (`tuple of int`): The site the declaration was read at, and the site the operation
  carries it onto, by index.

- **`orbitals`** (`tuple of int`): The orbitals declared at those two sites.

- **`note`** (`str`): The exclusion in words.

#### `SpaceGroupRep`

The model's representation of its space group: a matrix for each operation.

`r.rep` of what [`realized_group`](functions.md#realized_group) returns.  Each matrix
acts on the model's states and does not depend on `k`; the Bloch
operator of an operation at a `k` is built from it.  The calls that label
bands, as `hsp_table(r, bands=n)`, take `r` and read the
representation from it.  Its methods take an operation by its index, and
a k-point as a tabulated name, `"GM"`, or as reduced coordinates;
`covariance` takes reduced coordinates only.  `rep.summary()`
prints what it holds.

**Held in** [`RealizedGroup.rep`](#realizedgroup).

**Exported by** `pythtqc`, `pythtqc.symmetry`.

**Attributes**

- **`matrices`** (`dict`): `{index: D(g)}`, each operation's matrix, keyed on the operation's
  index.  It is the representation induced from the site irreps of
  `r.wyckoff_orbits`, in the convention of the tables used by the
  irreptables package: spin along `+z` in a right-handed frame.  On a
  spinful model the states are each orbital's spin up then down, state
  `2 * orbital + spin`.  The hoppings can leave a multiplet's block
  free up to a change of basis that leaves them unchanged, as on
  identical copies or on uncoupled orbitals.  Such a block is the
  operation on the lowest-angular-momentum functions of its site irreps,
  one function an orbital in order, wherever such a change of basis
  reaches that form: `p_x, p_y` for `E` at a `4mm` site.  Pure time
  reversal's matrix is
  `matrices[operation_index(r.group, "{1|0,0,0}'")]`.

- **`antiunitary`** (`tuple of int`): The indices of the operations combined with time reversal, each of
  which acts as its matrix times complex conjugation.

- **`numerical`** (`SpaceGroupRepNumerical`): How far each matrix misses carrying the Hamiltonian onto itself, and
  how far the matrices and the Bloch operators miss multiplying as the
  group does.

**Methods**

- **`transform_k(ind, k)`**: The image of a k-point under one operation, not folded back into the zone.
- **`in_little_group(ind, k)`**: Whether an operation fixes `k`, and the reciprocal lattice vector `gk − k`.
- **`little_group(k)`**: The operations that fix a k-point up to a reciprocal lattice vector.
- **`bloch_operator(ind, k, *, k_out=None, **kwargs)`**: The Bloch operator `U(g;k)` of one operation at a k-point.
- **`sewing_operator(ind, k, **kwargs)`**: The Bloch operator of an operation that fixes `k`, folded back to `k`.
- **`covariance(kpoints=None)`**: Each operation's covariance residual, measured again, in the units of H.
- **`summary()`**: What the representation holds, a line each, with the attribute that holds it.

##### `SpaceGroupRepNumerical`

How closely the matrices carry the Hamiltonian and multiply as the group does.

`r.rep.numerical`.  `summary()` prints it.  Each residual is the
largest modulus of an entry of a difference of matrices.

**Held in** [`SpaceGroupRep.numerical`](#spacegrouprep).

**Exported by** `pythtqc.symmetry`.

**Attributes**

- **`residuals`** (`dict`): `{index: residual}`, in the units of H: how far each operation's
  matrix `D` misses carrying the Hamiltonian onto itself.  The residual
  is the largest over the k-points measured of `D H(k) D† − H(gk)`,
  `gk` the image of `k`, with `H(k)*` in place of `H(k)` for an
  operation combined with time reversal.

- **`product_residual`** (`float`): How far the matrices miss multiplying as the group does: `D(g1)`
  times `D(g2)`, the second conjugated where `g1` is combined with
  time reversal, against `D(g1 g2)` times the sign the double group
  gives the product, `1` without spin, the largest over every pair of
  operations.

- **`factor_residual`** (`float`): The same for the Bloch operators of `bloch_operator`, whose
  products carry the phases of the operations' translations; `nan`
  where computing it failed with an error.

**Methods**

- **`summary()`**: The residuals, a line each, with the attribute that holds them.

#### `SpinBasis`

The spin pairs of a spinless model whose orbital index carries spin.

[`spin_basis`](functions.md#spin_basis) returns it, and [`spinless_presentation`](functions.md#spinless_presentation) returns
it beside the model. `spin=` of [`realized_group`](functions.md#realized_group) and of
[`spin_operator`](functions.md#spin_operator) takes it. The record [`realized_group`](functions.md#realized_group)
returns holds the pairs it read, as its field `spin`. `summary()`
returns the pairs as text, a line for each.

**Returned by** [`spin_basis`](functions.md#spin_basis), [`spinless_presentation`](functions.md#spinless_presentation).

**Held in** [`RealizedGroup.spin`](#realizedgroup).

**Exported by** `pythtqc`, `pythtqc.spin`.

**Attributes**

- **`pairs`** (`tuple of (int, int)`): Each pair as `(up, down)`: the orbitals that hold its spin states
  along `+z` and `-z` of the Cartesian axes in which the pythtb
  Lattice is specified, as pythtb's spinful orbitals hold them. Every
  orbital is in exactly one pair.

**Methods**

- **`states(c)`**: The orbitals of the pair at index `c` of `pairs`, up first.
- **`summary()`**: The pairs as text, a line for each.

#### `RealizedGroupNumerical`

How the group was measured: tolerances, centers, projection, measurements.

Held in [`RealizedGroup.numerical`](#realizedgroup).  `summary()` prints a line for
each part that applies, with the field that holds it.

**Held in** [`RealizedGroup.numerical`](#realizedgroup).

**Exported by** `pythtqc.groups`.

**Attributes**

- **`precision`** (`Precision or None`): If not `None`, the tolerances the declared precision sets.  `None`
  where no precision was declared.

- **`positions`** (`Positions or None`): If not `None`, the orbital centers moved onto the positions the group
  puts them at.  `None` where neither `precision=` nor
  `position_tol=` was declared.

- **`symmetrization`** (`Symmetrization or None`): If not `None`, the projection of `H` onto its symmetric part.
  `None` where `H` was not projected.

- **`validated`** (`ValidatedGroup`): What was measured of each operation of the group returned.

- **`source`** (`pythtb.TBModel`): The model as it was handed in.

**Methods**

- **`summary()`**: How the group was measured, a line for each part that applies.

##### `Precision`

The tolerances a declared precision sets.

Held in [`RealizedGroupNumerical.precision`](#realizedgroupnumerical), which is `None` where
no precision was declared.  Each tolerance is computed from the declared
precision, and `tol` from `max_h` as well.

**Held in** [`RealizedGroupNumerical.precision`](#realizedgroupnumerical).

**Exported by** `pythtqc.groups`.

**Attributes**

- **`precision`** (`float`): The precision declared with `precision=`, a fraction: of `max|H|`
  for the hoppings, and a length in reduced coordinates for the orbital
  centers.

- **`max_h`** (`float`): `max|H|`, the largest entry of `H(k)` once the mean on-site energy
  is subtracted, in the units of `H`.

- **`tol`** (`float`): A tolerance for checking whether an operation is a symmetry, in the
  units of `H`.  An operation is obeyed where its covariance residual,
  the largest entry of `U H(k) U^-1 - H(gk)` over the k measured for
  the best matrix `U` found, is at most `tol`.  It decides whether
  an operation holds; `cut` decides which matrices are found for it.
  Computed from `precision` and `max_h`.

- **`position_tol`** (`float`): A tolerance for moving an orbital center onto the position the group
  puts it at, in reduced coordinates.  It equals `precision`.

- **`cut`** (`float`): A dimensionless tolerance for checking whether an operation is a
  symmetry.  For each operation, the singular values of the symmetry
  equations are computed.  A singular value is treated as zero when it
  is at most `cut` times the largest singular value.  Computed from
  `precision`.

- **`window`** (`float`): A dimensionless tolerance for checking degeneracy of `H`.  Two
  eigenvectors are treated as degenerate when their energies differ by
  at most `window` times `max|H|`.  Computed from `precision`.

- **`symprec`** (`float`): The tolerance of spglib, which finds the group from the positions, as
  a Cartesian length in the model's length units.  It is `symprec=` as
  asked, raised to `10 x precision x L` where `precision x L`
  exceeds it, with `L` the length of the longest periodic lattice
  vector.

- **`symprec_raised`** (`bool`): Whether the declared precision raised `symprec`.

##### `Positions`

The orbital centers, moved onto the positions the group puts them at.

Held in [`RealizedGroupNumerical.positions`](#realizedgroupnumerical), which is `None` where
neither `precision=` nor `position_tol=` was declared.  The centers
are moved onto the positions that the largest group of the cell's
operations carrying them within `position_tol` puts them at.  Those
positions are the set that group carries exactly which lies nearest the
centers, its origin fitted to them.  Each center's move onto those
positions is at most `position_tol`, and so is the shift `offset` of
every center together, both in reduced coordinates.  Moving a center is a
`k`-dependent change of basis, which leaves every energy unchanged.

**Held in** [`RealizedGroupNumerical.positions`](#realizedgroupnumerical).

**Exported by** `pythtqc.groups`.

**Attributes**

- **`defect`** (`float`): How far the centers as handed in miss the action of the whole group,
  in reduced coordinates.  It is the largest magnitude of a component of
  `A·τ_a + t - τ_b`, taken modulo the lattice, over the operations
  `{A|t}` of the group and the orbitals `a`, with `τ_b` the center
  of the site the operation carries orbital `a`'s site onto.

- **`position_tol`** (`float`): The tolerance within which a center is moved, in reduced coordinates.
  It is `position_tol=` as declared, or `precision=`.

- **`adjusted`** (`bool`): Always `True`.  `n_moved` says how many orbitals were moved.

- **`defect_before`** (`float`): How far the centers as handed in miss the action of the operations
  that carry them, in reduced coordinates.

- **`defect_after`** (`float`): How far the moved centers miss the action of the same operations, in
  reduced coordinates.

- **`max_move`** (`float`): The largest move of a center onto the positions the group puts it at,
  about the fitted origin, in reduced coordinates; `offset` is not
  included.

- **`offset`** (`tuple`): The common shift of all centers, in reduced coordinates, that writes
  them in the model's own frame (see `frame`).  It is zero where
  `frame` is `"fitted"`, and at most `position_tol` otherwise.
  Each center returned is the one handed in, plus its own move, plus
  `offset`.

- **`frame`** (`str`): `"model"`: in the model's own origin, every operation's translation
  is a rational fraction of the lattice vectors.  `"fitted"`: the
  shift that would write the centers so is larger than
  `position_tol`, or there is none, and the centers stay about the
  origin fitted to them.

- **`spectrum_residual`** (`float`): The largest difference between the spectra before and after the move,
  over `n_check` random `k`, in the units of `H`.

- **`n_check`** (`int`): The number of random `k` at which `spectrum_residual` is measured,
  and 0 where no center moved.

- **`n_moved`** (`int`): The number of orbitals moved.

- **`move`** (`float or None`): If not `None`, the farthest a center must move for the centers to be
  symmetric under the whole group, in reduced coordinates, which is
  larger than `position_tol`.  The operations that do not carry the
  centers are dropped, and [`RealizedGroup.descent`](#realizedgroup) lists them as
  broken by the centers.
  `None` where the whole group carries the centers.

- **`carried`** (`int or None`): If not `None`, the number of operations of the largest group that
  carries the centers, one for each coset of the lattice translations,
  those combined with time reversal included.  `None` where the whole
  group carries them.

- **`note`** (`str`): The move in words.

##### `Symmetrization`

How far a Hamiltonian was moved onto its symmetric part, and why.

The record [`realized_group`](functions.md#realized_group) returns on
`r.numerical.symmetrization` where it projects `H` onto its symmetric
part; that field is `None` where it does not.  An operation's covariance
residual is how far its matrix misses carrying `H(k)` onto `H` at the
image of k: the largest entry of `U H(k) U† − H(gk)` over k, in the
units of `H`.  `H` is averaged over the operations of the group, those
combined with time reversal included, on a grid of k-points.  The average
is taken with a representation of the group on the orbitals that is built
from `H` for the projection and made an exact representation of the
group.  The average is written back as a hopping list by an inverse
Fourier transform.

The projection runs where an operation is near, obeyed within `tol` and
not to rounding.  It runs as well in four cases where something holds
only within the declared precision, each with an attribute of its own:
the site multiplets (`site_multiplets`), the separation of parts of the
crystal (`parts`), identical copies (`copies`), and the separation of
sectors that split site multiplets (`shares`).  Below the declared
precision means too small for the measurement of the operations, made at
`tol`, to see: it reads the orbitals such a coupling joins as uncoupled,
and copies that differ by so little as identical.  `r.model` is then the
symmetrized model, and `r.numerical.source` the model passed to
`realized_group`.  `summary()` returns the record as text, a line for
each finding, ending with the attribute that holds it where there is one.

**Held in** [`RealizedGroupNumerical.symmetrization`](#realizedgroupnumerical).

**Exported by** `pythtqc`, `pythtqc.symmetrization`.

**Attributes**

- **`defect`** (`float`): How far `H` moved: the largest change of an entry of `H(k)` over
  the grid, in the units of `H`.  It is at most `tol`.

- **`scale`** (`float`): The size of `H`: the largest entry of `H(k) − μ` over the grid,
  `μ` the mean on-site energy per state, in the units of `H`.

- **`tol`** (`float`): The tolerance on each operation's covariance residual, in the units of
  `H`: `10` times the precision times `max|H|` where a precision is
  declared, and `1e-8` times `max|H|` otherwise.  `max|H|` is the
  largest entry of `H(k) − μ` on fixed random k-points, or the largest
  hopping or on-site energy less `μ` where that is larger: the number
  `r.numerical.precision.max_h` holds where a precision is declared.
  It is not `scale`, which is read on the grid.

- **`near`** (`tuple of int`): The operations obeyed within `tol` and not to rounding, by their
  index in `r.operations`.

- **`exact`** (`tuple of int`): The operations obeyed to rounding, by their index in `r.operations`.

- **`residuals`** (`dict`): The covariance residual of each operation, by its index in
  `r.operations`, in the units of `H`.  For an operation the
  representation the average is taken with holds, it is the residual of
  the operation's own matrix there.  For any other, it is the residual
  of the best matrix the measurement of the operations found on `H`.

- **`passes`** (`int`): The number of averaging passes, at most three.  One pass is the
  projection.  A further pass runs only while the covariance residual of
  the averaged model is above rounding and the last pass moved `H` by
  more than rounding.

- **`site_multiplets`** (`SymmetrizationSiteMultiplets or None`): If not `None`, the site multiplets that hold only within the
  declared precision, which the projection made exact.

- **`parts`** (`SymmetrizationParts or None`): If not `None`, the parts of the crystal coupled to one another only
  below the declared precision, a coupling the projection left out.

- **`copies`** (`SymmetrizationCopies or None`): If not `None`, the identical copies that `H` coupled or told apart
  only below the declared precision, which the projection made exact
  copies.

- **`shares`** (`SymmetrizationShares or None`): If not `None`, the sectors splitting site multiplets that `H`
  coupled only below the declared precision, a coupling the projection
  left out.

- **`checks`** (`SymmetrizationChecks`): The measurements of how closely the hopping list returned reproduces
  the group average of `H`.

- **`verification`** (`SymmetrizationVerification`): The operations of the group measured again on the model returned, at
  `1e-8` times `scale`; `1e-8` is the relative tolerance a model
  built from exact numbers is measured at.

**Methods**

- **`summary()`**: The record as text: how far `H` moved and why, a line for each finding.

###### `SymmetrizationSiteMultiplets`

Site multiplets that hold only within the declared precision.

A site multiplet is a set of orbitals at one site that the Hamiltonian
binds into one carrier of a site irrep.  The orbitals' frame can be turned
inside the site by a small angle.  The symmetry matrices measured on the
Hamiltonian then mix the multiplets by an amount of the size of that
angle.  The projection makes the multiplets exact.

**Held in** [`Symmetrization.site_multiplets`](#symmetrization).

**Exported by** `pythtqc.symmetrization`.

**Attributes**

- **`mixing`** (`float`): How far the symmetry matrices measured on `H` mix the multiplets,
  relative to the size of the matrices.

- **`exact`** (`float`): The largest mixing at which the multiplets count as exact, `1e-8`.

- **`declared`** (`float`): `Symmetrization.tol` relative to `max|H|`, the tolerance the
  multiplets were measured at: ten times the precision where one is
  declared.

###### `SymmetrizationParts`

Parts of the crystal coupled to one another only below the declared precision.

A part of the crystal is a set of orbitals the hoppings connect, directly
or through others.  A coupling between parts that lies below the declared
precision counts as absent.  `H` then fixes the matrices of the
operations on each part only up to a one-dimensional character, chosen for
each part separately.  The projection leaves the coupling out, so the
model returned holds the parts exactly apart.

**Held in** [`Symmetrization.parts`](#symmetrization).

**Exported by** `pythtqc.symmetrization`.

**Attributes**

- **`orbitals`** (`tuple of tuple of int`): The orbitals of each part.

- **`coupling`** (`float`): The largest entry of `H(k)` between two parts, in the units of `H`.

- **`exact`** (`float`): The largest covariance residual at which an operation counts as exact,
  in the units of `H`.  `coupling` is above it.

###### `SymmetrizationCopies`

Identical copies coupled or told apart only below the declared precision.

Identical copies of a set of orbitals form a pseudospin the Hamiltonian
conserves within the declared precision.  The projection makes them exact
copies.

**Held in** [`Symmetrization.copies`](#symmetrization).

**Exported by** `pythtqc.symmetrization`.

**Attributes**

- **`orbitals`** (`tuple of tuple of int`): The orbitals of each set of copies.

- **`copies`** (`tuple of int`): The number of copies in each set.

- **`off`** (`float`): The largest entry of `H(k)` by which the copies miss being exact
  copies, in the units of `H`.

- **`exact`** (`float`): The largest covariance residual at which an operation counts as exact,
  in the units of `H`.  `off` is above it.

###### `SymmetrizationShares`

Sectors that split site multiplets, coupled only below the declared precision.

The matrices that commute with `H(k)` at every k and act within sites
split the states into sectors, read at the declared precision.  Where the
states of one site multiplet lie in several sectors, each orbit of sectors
under the group carries a representation of its own, induced from the
stabilizer of one of its sectors.  Two such cases are the bonding and the
antibonding state of two identical lattices coupled on site, and a
square's `p_x, p_y` pair that nothing couples.  Where `H` couples such
sectors only below the declared precision, the projection leaves that
coupling out.

**Held in** [`Symmetrization.shares`](#symmetrization).

**Exported by** `pythtqc.symmetrization`.

**Attributes**

- **`orbitals`** (`tuple of tuple of int`): The orbitals of each set of sectors.

- **`sectors`** (`tuple of int`): The number of sectors in each set.

- **`coupling`** (`float`): The largest entry of `H(k)` that couples a sector to anything outside
  it, in the units of `H`.

- **`exact`** (`float`): The largest covariance residual at which an operation counts as exact,
  in the units of `H`.  `coupling` is above it.

###### `SymmetrizationChecks`

The measurements on the hopping list the projection returns.

They measure how closely the hopping list reproduces the group average of
`H`.

**Held in** [`Symmetrization.checks`](#symmetrization).

**Exported by** `pythtqc.symmetrization`.

**Attributes**

- **`hermiticity`** (`float`): How far the coefficients `t(R)` of the transform miss
  `t(−R) = t(R)†`, in the units of `H`.

- **`drop_threshold`** (`float`): The norm below which a coefficient of the transform was dropped as
  rounding, in the units of `H`; `0.0` where none was.

- **`drop_gap`** (`float`): Where a coefficient was dropped, the smallest norm kept over the
  largest norm dropped.  Coefficients are dropped only where this ratio
  exceeds ten.  Where none was dropped, it holds another ratio or
  `inf`, and is no gap.

- **`n_dropped`** (`int`): How many nonzero coefficients of the transform were dropped.

- **`n_hoppings`** (`int`): How many hopping terms the model returned holds, each conjugate pair
  counted once and the on-site terms not counted.

- **`outside_box`** (`float`): The largest coefficient of the transform at a lattice vector outside
  the box that bounds the group's images of the model's hoppings,
  relative to the largest coefficient.

- **`range_gate`** (`float`): The largest `outside_box` the projection accepts, `1e-10`.

- **`k_identity`** (`float`): How far the model returned misses the average of `H`, at 16 random
  k-points off the grid, in the units of `H`.

- **`covariance_residue`** (`float`): The covariance residual of the model returned, with the matrices of the
  representation the average is taken with, at the 16 random k-points
  `k_identity` is measured at, in the units of `H`.

- **`per_pass`** (`tuple of float`): How far each averaging pass moved `H` on the grid, in the units of
  `H`.

###### `SymmetrizationVerification`

The operations of the group measured again on the symmetrized model.

They are measured at `1e-8` times `Symmetrization.scale`; `1e-8` is
the relative tolerance a model built from exact numbers is measured at.

**Held in** [`Symmetrization.verification`](#symmetrization).

**Exported by** `pythtqc.symmetrization`.

**Attributes**

- **`tol`** (`float`): The tolerance on each operation's covariance residual, in the units of
  `H`: `1e-8` times `Symmetrization.scale`.

- **`nk`** (`int`): The number of k-points along each periodic direction of the grid the
  operations are measured on, the same as in the measurement before the
  projection.

- **`n_kept`** (`int`): How many operations held.

- **`n_operations`** (`int`): How many operations were measured.

- **`worst_residual`** (`float`): The largest covariance residual of an operation that held, in the
  units of `H`.

- **`near_kept`** (`bool`): Whether every operation that was near before the projection held.

- **`broken`** (`tuple of int`): The operations measured broken, whose covariance residual exceeds
  `tol`, by their index in `r.operations`.

- **`note`** (`str`): If nonempty, a sentence saying that an operation near before the
  projection did not hold.  It quotes
  `Symmetrization.checks.covariance_residue`, the measurement that says
  whether the projection worked.

##### `ValidatedGroup`

Whether the Hamiltonian obeys each operation of a group, as measured.

[`realized_group`](functions.md#realized_group) returns it as `r.numerical.validated`,
for the group returned.  A `cell_group_not_realized` refusal carries it
in `context["validated"]`, for the group in `context["group"]`.  That
refusal is returned where an operation combined with time reversal, kept
before `H` is projected onto its symmetric part, is broken after it.

An operation is kept where one unitary `U` has `U H(k) U† = H(Rk)`
within `tol` at every k-point measured, `Rk` being the operation's
image of `k`.  For an operation combined with time reversal, `H(k)*`
stands in place of `H(k)`.  The operations kept are those in
`obeyed` and in `undetermined`.  Operations that act alike on `k`,
and alike in whether they are combined with time reversal, are measured
once: a class found broken breaks every member, and a class kept with
more than one member leaves each member undetermined.  `summary()`
returns the verdicts as text, with the largest covariance residual of
the operations kept.

**Held in** [`RealizedGroupNumerical.validated`](#realizedgroupnumerical).

**Exported by** `pythtqc`, `pythtqc.groups`.

**Attributes**

- **`obeyed`** (`tuple of int`): The operations kept that share their measurement with no other
  operation, by index.

- **`undetermined`** (`tuple of int`): The operations kept that share their measurement with another
  operation, by index.  On a free-standing layer, an operation and its
  product with the horizontal mirror share one.  `H(k)` does not tell
  such operations apart, so each is then held to its own form: its
  solution space is read for a matrix `O(g) ⊗ S(g)` that carries each
  site's orbitals onto its image's, with the operation's own spin factor
  `S(g)`.  In the record `realized_group` returns, each one holds.

- **`broken`** (`tuple of int`): The operations broken, by index, one for each entry of
  `discrepancies`.

- **`discrepancies`** (`tuple of Discrepancy`): One entry per broken operation, with how it was found broken.

- **`residuals`** (`dict`): `{index: residual}`, the covariance residual of each operation
  measured, in the units of H.  It is the largest absolute entry of
  `U H(k) U† - H(Rk)` over the k-points measured, for the unitary
  `U` drawn from the operation's solution space that makes it
  smallest.  It is `0.0` for the identity, and infinite where no
  matrix was found.  An operation out of the layer's plane has no
  entry.

- **`closed`** (`bool`): Whether the operations kept form a group.

- **`tol`** (`float`): The tolerance each covariance residual is held to, in the units of H.

- **`n_kpoints`** (`int`): The number of k-points the residuals are measured on.

**Methods**

- **`summary()`**: The verdicts, a line for each part, with the attribute that holds it.

###### `Discrepancy`

One operation the Hamiltonian breaks, and how it was found broken.

An entry of [`ValidatedGroup.discrepancies`](#validatedgroup).

**Held in** [`ValidatedGroup.discrepancies`](#validatedgroup).

**Exported by** `pythtqc.groups`.

**Attributes**

- **`index`** (`int`): The operation's index, as the group's operations list it
  (`r.operations` for the group returned).

- **`refuted_by`** (`{"no symmetry matrix", "covariance", "out of plane"}`): How the operation was found broken.  `"no symmetry matrix"`: no
  nonzero matrix `X` has `X H(k) = H(Rk) X` at the k-points
  measured, within `cut`, `Rk` being the operation's image of `k`
  (`H(k)*` in place of `H(k)` for an operation combined with time
  reversal).  `"covariance"`: such matrices exist, and the operation's
  covariance residual is above `tol`.  `"out of plane"`: the
  operation does not map the layer's plane onto itself, so it has no
  action on the layer's `k`.

- **`residual`** (`float`): The operation's covariance residual, in the units of H, as
  [`ValidatedGroup.residuals`](#validatedgroup) holds it.  Infinite where no matrix
  was found, and for an operation out of the plane, for which
  `residuals` holds no entry.

### `SiteMultiplets`

The multiplets the Hamiltonian binds each site's orbitals into.

A site's multiplets are the finest partition of its orbitals such that
every operation fixing the site has a matrix carrying each multiplet into
itself.  Orbitals that a matrix acting within sites and commuting with
`H(k)` at every `k` mixes also lie in one multiplet.  A `(p_x, p_y)`
pair under a threefold axis is one multiplet.  So is a `(p_x, p_y)` pair
under a fourfold axis that no hopping couples to each other, in whatever
basis it is written: the fourfold axis or a matrix commuting with
`H(k)`, such as the projector onto `p_x`, mixes the two orbitals it is
written in.  Identical copies of a set of orbitals, coupled to one another
by no hopping, are one multiplet, since a matrix commuting with `H(k)`
exchanges them.  So are copies of one site irrep whose basis is turned
from site to site of an orbit, since an operation carrying one site onto
another then mixes the copies.  An orbital that nothing mixes with another
is a multiplet of its own.

Each multiplet's matrices are the model's own blocks, in whatever basis
its orbitals are written.  A declared site irrep is assigned to the
multiplets as they are, without changing them.  The record, or its
`by_site`, is taken by [`realized_group`](functions.md#realized_group) as
`site_multiplets=`, which states the multiplets instead of measuring
them.  `summary()` returns each site's multiplets as text.

**Taken by** [`realized_group`](functions.md#realized_group), as `site_multiplets=`.

**Exported by** `pythtqc`, `pythtqc.symmetry`.

**Attributes**

- **`by_site`** (`tuple`): For each site, its multiplets, each the orbitals it holds.

- **`dims`** (`tuple`): For each site, the dimension of each multiplet: the number of orbitals
  it holds, or of spinor pairs where a spin basis is declared.

- **`accidental`** (`tuple of dict`): If nonempty, the orbitals whose multiplet could be chosen in more than
  one way, an accidental degeneracy: an operation mixes an orbital with
  several partners, each at least half as strongly as the strongest, and
  the multiplet returned holds the lowest-indexed one.  Each entry holds
  the `operation`, the `orbitals`, the `chosen` partners and a
  `note`.

- **`determined_by`** (`str`): How the multiplets are known: `"hamiltonian"`, read off the
  Hamiltonian, or `"declaration"`, stated by `site_multiplets=`.  A
  record handed back as `site_multiplets=` keeps its own value.

**Methods**

- **`summary()`**: Each site's multiplets, any accidental ties and how they are known.

### `NormalizedRep`

An operation's matrix in its canonical phase, and what fixed the phase.

The record `normalize_rep` returns.  `D` is the matrix handed
in, `U` the unitary matrix nearest it, and `λU` the matrix returned.
`n` is the order of the operation modulo lattice translations, and the
normalization is `(λU)^n = ζ·1`.  `T` is the matrix of an
antiunitary operation the anchor holds ([`RepAnchor`](#repanchor)).

**Returned by** `symmetry.normalize_rep`.

**Exported by** `pythtqc.symmetry`.

**Attributes**

- **`matrix`** (`numpy.ndarray`): The matrix in its canonical phase, `λU`.

- **`factor`** (`complex`): The phase `λ` applied.

- **`order`** (`int`): The order `n` of the operation, modulo lattice translations.

- **`required`** (`int`): `ζ`, `+1` or `-1`: the multiple of the identity that the
  multiplication table of the group requires of the `n`-th power.

- **`norm_residual`** (`float`): `max|(λU)^n − ζ·1|`.

- **`norm_informative`** (`bool`): Whether the normalization narrows the phase.  `False` for an
  antiunitary operation, whose order is even, since its `n`-th power
  does not depend on the phase.

- **`anchor_residual`** (`float or None`): If not `None`, `max|T conj(λU) T† − λU|`.  `None` where the
  anchor holds no `T`.

- **`anchor_used`** (`bool`): Whether the condition on `T` was applied: `True` where
  `T conj(U) T†` is a multiple of `U`, even where it removes no root.

- **`table_anchored`** (`bool`): Whether the operation's spin rotation in the tables narrowed the
  phases.

- **`product_anchored`** (`bool`): Whether the multiplication table of the group chose the phase among
  the roots the normalization allows.  `False` in what
  `normalize_rep` returns: the table chooses only where a whole
  representation is built.

- **`candidates`** (`tuple of complex`): If nonempty, every `λ` the normalization allows.  Empty where it
  allows every phase.

- **`accepted`** (`tuple of complex`): The `λ` left by `T` and the spin rotation, among which the
  reference block chose.  `(1,)` where the normalization allows every
  phase and nothing narrows it.

- **`principal_entry`** (`complex`): The entry of the matrix returned on which the choice was read: the
  entry of largest modulus of the reference block, the first in row
  order among equals, or of the whole matrix where that block is zero.

- **`unitarity`** (`float`): How far `D` is from unitary, `max|D†D − 1|`.

- **`polar_shift`** (`float`): How far `U` is from `D`, `max|D − U|`.

### `RepAnchor`

What fixes the phase of an operation's matrix that covariance leaves open.

`normalize_rep` reads it after the normalization `(λD)^n = ζ·1`,
which fixes the phase `λ` only up to an `n`-th root of unity.  Here
`n` is the order of the operation modulo lattice translations, and
`ζ`, `+1` or `-1`, the sign the multiplication table of the group
requires.  The matrix of an antiunitary operation and the operation's
spin rotation, each where it is given, narrow the roots.  The entry of
largest modulus of a reference block of the matrix picks among those
left.  Anchoring a representation at another reference block can
multiply its matrices by a one-dimensional character of the group.  That
character is fixed afterwards, where the labeling of the representation
is chosen.

**Taken by** `symmetry.normalize_rep`, as `anchor=`.

**Exported by** `pythtqc.symmetry`.

**Attributes**

- **`antiunitary`** (`numpy.ndarray or None`): If not `None`, the matrix `T` of an antiunitary operation of the
  group.  A representation built by [`realized_group`](functions.md#realized_group) puts
  time reversal here where the group has it, and otherwise another
  antiunitary operation.

- **`rows`** (`tuple of int`): The rows of the reference block `D[rows, cols]`: the states of the
  image, under the operation, of the multiplet whose states are
  `cols`.  Empty to read the whole matrix.

- **`cols`** (`tuple of int`): The columns of the reference block: the states of one multiplet.
  Empty to read the whole matrix.

- **`internal`** (`numpy.ndarray or None`): If not `None`, the operation's spin rotation in the tables,
  `op.spinor_rotation`, or `[[1]]` without spin.  A representation
  built by [`realized_group`](functions.md#realized_group) gives it only where the group
  has no antiunitary operation, and it then takes the place of
  `antiunitary`.

- **`atol`** (`float`): How far the conditions on `antiunitary` and `internal` may miss,
  and the modulus below which an entry of the matrix is zero.

## Group data in this cell's coordinates

The records of the tables of a space group, written in the cell of the model's
pythtb Lattice: its high-symmetry points, the little group of a k-point, the
Wyckoff position of a site, a mesh of k-points, its elementary band
representations, and the little-group representations read off measured
characters.

- [`HighSymmetryPoints`](#highsymmetrypoints): The tabulated high-symmetry points of the model's zone, by name.
- [`LittleGroup`](#littlegroup): The operations that leave one k invariant up to a reciprocal lattice vector.
- [`SiteReference`](#sitereference): One site's Wyckoff position in the tables' setting.
- [`KGrid`](#kgrid): A mesh of k-points that holds `GM`, with the high-symmetry points on it.
- [`EBRData`](#ebrdata): The elementary band representations of a group, and their matrix.
- [`CorepIdentification`](#corepidentification): The little-group representations each measured multiplet carries at one k.

### `HighSymmetryPoints`

The tabulated high-symmetry points of the model's zone, by name.

What [`high_symmetry_points`](functions.md#high_symmetry_points) returns.  It maps each point's name in
the tables, `"GM"`, to its reduced coordinates in the model's cell,
`dim_k` of them.  `summary()` prints it.

**Returned by** [`high_symmetry_points`](functions.md#high_symmetry_points).

**Exported by** `pythtqc`, `pythtqc.groups`.

**Attributes**

- **`out_of_plane`** (`tuple of str`): If nonempty, the tabulated points out of a layer's plane, which the
  layer's zone does not have: `A`, `H` and `L` of a hexagonal
  layer.

**Methods**

- **`summary()`**: The points, a line for each, and a line for those out of a layer's plane.

### `LittleGroup`

The operations that leave one k invariant up to a reciprocal lattice vector.

The record [`little_group`](functions.md#little_group) returns.  `summary()` prints
it, and `len()` is the number of its unitary operations.

**Returned by** [`little_group`](functions.md#little_group).

**Exported by** `pythtqc`, `pythtqc.groups`.

**Attributes**

- **`k`** (`ndarray, shape (dim_k,)`): The point, in the model's reduced coordinates.

- **`kpoint`** (`str or None`): If not `None`, the tables' name of the point, `"X"`.  `None`
  where `k` is none of the group's tabulated points, up to a
  reciprocal lattice vector of the model's cell, or where the group has
  no transformation to the tables' conventional cell.

- **`unitary`** (`tuple of int`): The indices of its unitary operations, ascending.  The characters of
  a little-group representation at `k` are given on these
  operations, in this order.

- **`antiunitary`** (`tuple of int`): If nonempty, the indices of its operations combined with time
  reversal, ascending: those whose rotation carries `k` to `-k`, up
  to a reciprocal lattice vector.  The tables give no character on
  them.

- **`G`** (`dict`): `{index: G}` for each unitary operation `g`, with `G = gk - k`
  the reciprocal lattice vector by which `g` moves `k`.  Its three
  components are in reduced coordinates of the group's
  three-dimensional cell, the one along a layer's normal included.

**Methods**

- **`summary()`**: The little group as text, a line for each quantity.

### `SiteReference`

One site's Wyckoff position in the tables' setting.

The tables' setting is the conventional cell used by the irreptables
package.  The record is one entry of what
[`wyckoff_sites_tables`](functions.md#wyckoff_sites_tables) returns.  It prints as one line:
its position, its free parameters and the positions it slides onto.

**Returned by** [`wyckoff_sites_tables`](functions.md#wyckoff_sites_tables).

**Exported by** `pythtqc`, `pythtqc.groups`.

**Attributes**

- **`wyckoff`** (`str`): The Wyckoff position the site lies on, written with its multiplicity
  in the conventional cell, `"8a"`.  Where the site lies on several
  positions, within the `position_tol` of
  [`wyckoff_sites_tables`](functions.md#wyckoff_sites_tables), it is the one with the fewest
  free parameters, and then the smallest multiplicity.

- **`slides_onto`** (`tuple of str`): The Wyckoff positions at which the tables list elementary band
  representations and whose points all lie on the site's own position,
  that position included where the tables list it.  An orbital at the
  site can be moved onto each of them without leaving its position.
  For the line `6c`, `(0, 0, z)`, of `R-3m` they are
  `("3a", "3b")`, and for a site at `3a` they are `("3a",)`.

- **`n_free`** (`int`): The number of its free parameters: 0 for a point, 1 a line, 2 a
  plane, 3 the general position.  In a polar group no position is a
  point: `P4mm`'s `1a` is `(0, 0, z)`.

- **`position`** (`ndarray, shape (3,)`): The site, in reduced coordinates of the group's three-dimensional
  cell, not of the tables' conventional cell.

### `KGrid`

A mesh of k-points that holds `GM`, with the high-symmetry points on it.

The record [`k_grid`](functions.md#k_grid) returns.  `summary()` prints it.
Its `mesh` is a `pythtb.Mesh`, which `pythtb.WFArray` takes
directly.

**Returned by** [`k_grid`](functions.md#k_grid).

**Exported by** `pythtqc`, `pythtqc.groups`.

**Attributes**

- **`divisions`** (`tuple of int`): The number of k-points along each reciprocal lattice vector of the
  group's three-dimensional cell, `1` along a layer's normal.

- **`symmetric`** (`bool`): Whether the group carries the mesh onto itself.

- **`mesh`** (`pythtb.Mesh`): The mesh, with `k` in `[0, 1)` along each periodic direction and
  one axis per periodic direction.

- **`irreducible`** (`ndarray, shape (n, dim_k), or None`): If not `None`, one point of the mesh from each orbit of the group,
  in the model's reduced coordinates.  `None` for a mesh that is not
  symmetric.

- **`contains`** (`dict`): If nonempty, `{name: bool}`: whether the mesh holds each tabulated
  high-symmetry point.  A point is on the mesh where each of its
  reduced coordinates in the model's cell is a multiple of `1/n`
  along an axis of `n` divisions: `1/2` needs an even number of
  divisions, and `1/3` a multiple of three.
  [`high_symmetry_points`](functions.md#high_symmetry_points) gives the coordinates of each
  point.  Empty where the high-symmetry points cannot be read.

**Methods**

- **`summary()`**: The mesh as text, a line for each quantity.

### `EBRData`

The elementary band representations of a group, and their matrix.

What [`ebr_data`](functions.md#ebr_data) returns.  The matrix has a column for each
elementary band representation and a row for each little-group
representation at each high-symmetry point of the group, those out of a
layer's plane included.

**Returned by** [`ebr_data`](functions.md#ebr_data).

**Exported by** `pythtqc.groups`.

**Attributes**

- **`ebrs`** (`tuple of Term`): The columns, in the order of the tables.  Each names the site irrep
  it is induced from, its Wyckoff position and the position's
  site-symmetry group.  A site-symmetry group whose symbol begins with
  a bar, `-43m`, is written with it, where the tables leave it out.

- **`corep_names`** (`tuple of str`): The label of each row, as `identify_from_characters` returns
  labels.

- **`matrix`** (`ndarray`): The matrix of elementary band representations.  Entry `(i, j)` is
  the multiplicity of `corep_names[i]` in `ebrs[j]`.

- **`smith`** (`tuple of ndarray`): The Smith normal form `(U, D, V)` of `matrix`, with
  `U @ matrix @ V == D`.  For a group with one elementary band
  representation, `D` is a vector and `V` a number.

- **`convention`** (`EBRConvention`): Where the rows differ from those of the Bilbao Crystallographic
  Server.

- **`group`** (`GroupData`): The group the elementary band representations are read for.

#### `EBRConvention`

Where the elementary band representations returned differ from the tables'.

What `ebr_data(group).convention` holds.  The rows returned are labeled
in the convention where a lattice translation `t` acts on a Bloch state
at `k` as `e^{-ik·t}`.  The tables of the Bilbao Crystallographic
Server, which the irreptables package ships, use `e^{+ik·t}`.  The
label the server prints at `k` is the label returned at `-k`, so the
two agree at the time-reversal-invariant momenta.  The Wyckoff positions
are the server's.

**Held in** [`EBRData.convention`](#ebrdata).

**Exported by** `pythtqc.groups`.

**Attributes**

- **`name`** (`str`): The convention, `"physical (translations as e^{-ik.t})"`.

- **`changed_points`** (`tuple of str`): If nonempty, the points of `changed_labels`: the high-symmetry
  points at which a label is renamed or its row differs from the
  server's.

- **`changed_labels`** (`tuple of str`): If nonempty, the server's labels that are renamed here, or whose row
  of the matrix of elementary band representations differs from the
  server's row of that label.

- **`conjugated_points`** (`tuple of str`): If nonempty, the points at which the characters of
  [`corep_table`](functions.md#corep_table) are the complex conjugates of the server's.  Each
  character vector there carries the server's label of the vector it
  equals, where the server prints one.

- **`divergent_points`** (`tuple of str`): If nonempty, the points at which the characters the server prints
  are, in the package's convention, those of `-k`, another point of
  the star of `k`.  The characters [`corep_table`](functions.md#corep_table) returns there
  are their complex conjugates.  No reordering of the labels makes them
  the server's.

- **`reglued_points`** (`tuple of str`): If nonempty, the points at which time reversal glues the irreps into
  coreps differently from the server's tables.  They occur in grey
  groups only.

- **`conjugation`** (`dict`): `{label: label}`, each of the server's labels with the label
  returned whose characters are the complex conjugates of its own.

- **`tr_partner`** (`tuple of int`): For each column of the matrix, the column induced from the
  complex-conjugate site irrep at the same Wyckoff position, the column
  itself for a real site irrep.

- **`corep_names_bcs`** (`tuple of str`): The server's label of each row, position by position with
  `ebr_data(group).corep_names`.  The two hold the same label at
  every position outside `reglued_points`.  Outside them, what can
  differ is the row a label holds, given by `row_permutation`.

- **`row_permutation`** (`tuple of int`): The permutation `P` of the rows.  Row `i` of a column returned is
  the server's row `P[i]` of the column `tr_partner` names.  So a
  linear form whose coefficients are shipped on the server's labels
  has, on row `i`, its coefficient on `corep_names_bcs[P[i]]`.

- **`note`** (`str`): A few sentences on what the convention changes at each kind of point.

### `CorepIdentification`

The little-group representations each measured multiplet carries at one k.

The record `groups.identify_from_characters` returns, and
`coreps_at(...).numerical.identification` holds.  The multiplicities
are read off each multiplet's characters at a high-symmetry point and
rounded to integers.  `summary()` prints it.

**Returned by** `groups.identify_from_characters`.

**Held in** [`CorepsAtNumerical.identification`](#corepsatnumerical).

**Exported by** `pythtqc`, `pythtqc.groups`.

**Attributes**

- **`kpoint`, `k`** (`str and ndarray`): The tabulated point, by name and in the model's reduced coordinates.

- **`labels`** (`tuple of dict`): Each multiplet's little-group representations,
  `{name: multiplicity}`.

- **`dims`** (`ndarray`): The dimension of each multiplet: each multiplicity times the
  dimension of its representation, summed.

- **`indices`** (`tuple of int`): The indices of the little group's unitary operations, on which the
  characters are given.

- **`multiplicities`** (`ndarray`): The multiplicities before rounding, a complex array with a row per
  multiplet and a column per little-group representation.

- **`label_order`** (`tuple of str`): The little-group representations at the point, by name, in the order
  of the columns of `multiplicities`.

- **`integrality`** (`float`): The largest distance of a multiplicity before rounding from the
  nearest integer.

**Methods**

- **`summary()`**: The multiplets as text, a line for each quantity.

## Labeling the bands

The records of the calls that label a set of bands: the sewing matrix of one
operation, the characters and the little-group representations of the bands at
one k-point, and their symmetry data vector over every high-symmetry point.

- [`SewingMatrix`](#sewingmatrix): The sewing matrix of one operation on a set of bands at one k-point.
- [`BandCharacters`](#bandcharacters): The characters of a set of bands at one k-point.
- [`CorepsAt`](#corepsat): The little-group representations of a set of bands at one high-symmetry point.
- [`HSPTable`](#hsptable): The symmetry data vector of a set of bands, point by point.

### `SewingMatrix`

The sewing matrix of one operation on a set of bands at one k-point.

The record [`sewing_matrix`](functions.md#sewing_matrix) returns.  `summary()` prints the
operation, the bands, the k-point, the reciprocal lattice vector `G` and
the eigenvalues.  The matrix is `B(g,k) = V† U(g;k) V`, with `V` the
states of the bands as columns and `U(g;k)` the operator of the
operation `g` on the model's states at `k`.  For an operation combined
with time reversal it is `V† U(g;k) V*`.  For a unitary operation, its
trace over a degenerate multiplet is that multiplet's character.

**Returned by** [`sewing_matrix`](functions.md#sewing_matrix).

**Exported by** `pythtqc`, `pythtqc.sewing`.

**Attributes**

- **`operation`** (`int`): The operation, by its index in `r.operations`.

- **`seitz`** (`str`): If nonempty, the operation's Seitz symbol, `{4+_001|0,0,0}`, primed
  where the operation is combined with time reversal.  Empty where the
  model's cell cannot be transformed to the conventional cell of the
  irreptables package.

- **`bands`** (`tuple of int`): The set of bands, by their indices among the states handed in.

- **`k`** (`ndarray`): The k-point, in the model's reduced coordinates.

- **`kpoint`** (`str or None`): If not `None`, the name of the k-point in the tables of the
  irreptables package, `"M"`.  `None` where `k` is not a tabulated
  point, or where the model's cell cannot be transformed to the
  conventional cell of the irreptables package.

- **`G`** (`ndarray of int`): The reciprocal lattice vector by which the operation's image of `k`
  differs from `k`, in reduced coordinates.  It is `Rk − k`, with
  `R` the operation's rotation acting on `k`, or `−Rk − k` for an
  operation combined with time reversal.  `U(g;k)` carries the phase
  `e^{2πi G·τ}` on an orbital at `τ`.

- **`matrix`** (`ndarray, shape (nb, nb)`): The sewing matrix `B`, `nb` the number of bands.

- **`eigenvalues`** (`ndarray`): The eigenvalues of `B`.  They lie on the unit circle where `B` is
  unitary.

- **`numerical`** (`SewingMatrixNumerical`): How far `B` is from unitary, `U(g;k)` from commuting with
  `H(k)`, and the states from eigenstates.

**Methods**

- **`summary()`**: The operation, bands, k-point, `G` and eigenvalues, with their attributes.

#### `SewingMatrixNumerical`

How far a sewing matrix's inputs are from exact.

Held in [`SewingMatrix.numerical`](#sewingmatrix).  `summary()` prints the numbers
measured, together on one line.

**Held in** [`SewingMatrix.numerical`](#sewingmatrix).

**Exported by** `pythtqc.sewing`.

**Attributes**

- **`unitarity`** (`float`): The largest entry of `|B†B − 1|`, with `B` the sewing matrix.  It
  is zero where the set of bands is closed under the operation.

- **`commutator`** (`float`): The largest entry of `|U H(k) − H(k) U|`, in the units of `H`,
  with `U` the operator of the operation on the model's states at
  `k`.  For an operation combined with time reversal it is the
  largest entry of `|U H(k)* − H(k) U|`.  It is zero where `U` is a
  symmetry of `H(k)`.

- **`residual`** (`float`): If not `nan`, the largest entry of `|H(k) V − V diag(w)|`, in the
  units of `H`, with `V` the states as columns and `w` their
  energies: how far the states are from eigenstates at `k`.  `nan`
  where the states came without energies.

**Methods**

- **`summary()`**: The numbers measured, on one line, with the attributes that hold them.

### `BandCharacters`

The characters of a set of bands at one k-point.

The record [`characters`](functions.md#characters) returns.  A character is the trace of the
sewing matrix of a unitary operation of the little group over one
degenerate multiplet.  `summary()` prints the characters as a table, a
column for each multiplet and a line for each operation.
[`coreps_at`](functions.md#coreps_at) identifies the little-group representations from them.

**Returned by** [`characters`](functions.md#characters).

**Exported by** `pythtqc`, `pythtqc.sewing`.

**Attributes**

- **`bands`** (`tuple of int`): The set of bands, by their indices among the states handed in,
  ascending.

- **`k`** (`ndarray`): The k-point, in the model's reduced coordinates.

- **`kpoint`** (`str or None`): If not `None`, the name of the k-point in the tables of the
  irreptables package, `"GM"`.  `None` where `k` is not a
  tabulated point, or where the model's cell cannot be transformed to
  the conventional cell of the irreptables package.

- **`multiplets`** (`tuple of MultipletCharacters`): A row for each degenerate multiplet of the set of bands, lowest
  first, with its bands, its mean energy and its characters.

- **`operations`** (`OperationTable or Refusal`): The operations of the little group, unitary and antiunitary, by their
  index in `r.operations`, each with its Seitz symbol.  The
  characters in `multiplets` are those of the unitary operations
  alone.  A `Refusal` where the model's cell cannot be transformed to
  the conventional cell of the irreptables package.

- **`numerical`** (`BandCharactersNumerical`): How far the sewing matrices are from unitary, the operators of the
  operations on the model's states from commuting with `H(k)`, and
  the states from eigenstates.  It also holds the zero thresholds at
  `k`.

**Methods**

- **`summary()`**: The characters as a table, with the attributes that hold its parts.

#### `MultipletCharacters`

The characters of one degenerate multiplet.

A row of [`BandCharacters.multiplets`](#bandcharacters).

**Held in** [`BandCharacters.multiplets`](#bandcharacters).

**Exported by** `pythtqc.sewing`.

**Attributes**

- **`bands`** (`tuple of int`): The bands of the multiplet.

- **`energy`** (`float`): The mean energy of the multiplet, in the units of `H`.

- **`characters`** (`dict`): The trace over the multiplet of the sewing matrix of each unitary
  operation of the little group, `{index: character}`, keyed by the
  operation's index in `r.operations`.

#### `BandCharactersNumerical`

How far the inputs of a set of bands' characters are from exact.

Held in [`BandCharacters.numerical`](#bandcharacters).  `summary()` prints the
largest departures from unitarity, from commuting with `H(k)` and from
eigenstates.  It then names the operations whose departure from unitarity
the two zero thresholds classify differently.

**Held in** [`BandCharacters.numerical`](#bandcharacters).

**Exported by** `pythtqc.sewing`.

**Attributes**

- **`unitarity`** (`dict`): For each unitary operation of the little group, by its index, the
  largest entry of `|B†B − 1|`, with `B` its sewing matrix on the
  whole set of bands.

- **`commutators`** (`dict`): For each unitary operation of the little group, by its index, the
  largest entry of `|U H(k) − H(k) U|`, with `U` its operator on the
  model's states, in the units of `H`.

- **`residual`** (`float`): The largest entry of `|H(k) V − V diag(w)|`, in the units of `H`,
  with `V` the states as columns and `w` their energies: how far the
  states are from eigenstates at `k`.

- **`spread`** (`float`): The largest energy spread inside a degenerate multiplet of the
  spectrum, in the units of `H`.

- **`thresholds`** (`Thresholds`): The two zero thresholds at `k`.

- **`disagreements`** (`tuple of dict`): If nonempty, the departures from unitarity that the two thresholds
  classify differently, by operation, as `disagreements` returns
  them.

- **`degeneracy_tol`** (`float`): The energy tolerance the multiplets were grouped at, in the units of
  `H`.

- **`spectrum`** (`ndarray`): Every energy of the diagonalization the characters were read from.

**Methods**

- **`summary()`**: The largest departures from exact, with the attributes that hold them.

##### `Thresholds`

The two zero thresholds at one k-point.

The record `two_tier_thresholds` returns.  A number is treated as
zero where its magnitude is at most a threshold.  One threshold, `flat`,
is fixed and makes no allowance for the gap between multiplets.  The
other, `effective`, is for a number read off eigenvectors, and grows as
the smallest gap between distinct multiplets shrinks.  Both are
dimensionless.  [`characters`](functions.md#characters) compares the departure of each of its
sewing matrices from unitarity with both.  `summary()` prints both,
with the numbers they are computed from.

**Returned by** `sewing.two_tier_thresholds`.

**Held in** [`BandCharactersNumerical.thresholds`](#bandcharactersnumerical), [`CorepsAtNumerical.thresholds`](#corepsatnumerical).

**Exported by** `pythtqc.sewing`.

**Attributes**

- **`flat`** (`float`): The fixed threshold, with no allowance for the gap between multiplets.

- **`noise_floor`** (`float`): The first-order bound on the amplitude a computed eigenvector carries
  on the states of other multiplets, `8 ε |H(k)| / Δ`.  Here `ε` is
  the machine precision, `|H(k)|` is `norm` and `Δ` is
  `separation`.  Infinite where `Δ` is zero or negative, and zero
  where `Δ` is infinite, for a spectrum of one multiplet.

- **`effective`** (`float`): The threshold for a number read off eigenvectors, the larger of
  `flat` and `noise_floor`.

- **`norm`** (`float`): The spectral norm of `H(k)`, in the units of `H`.

- **`separation`** (`float`): The smallest gap between distinct multiplets, in the units of `H`.

- **`degeneracy_tol`** (`float`): The energy tolerance the multiplets were grouped at, in the units of
  `H`.

**Methods**

- **`zero(value)`**: Whether `|value|` is at most `effective`.
- **`zero_flat(value)`**: Whether `|value|` is at most `flat`, the fixed threshold.
- **`disagree(value)`**: Whether `zero` and `zero_flat` classify `value` differently.
- **`summary()`**: The two thresholds and the numbers they come from, with their attributes.

### `CorepsAt`

The little-group representations of a set of bands at one high-symmetry point.

The record [`coreps_at`](functions.md#coreps_at) returns.  `summary()` prints the
multiplicities of the little-group representations over the set of
bands, then each degenerate multiplet's labels where there are several
multiplets, and the gap to the nearest other band.

**Returned by** [`coreps_at`](functions.md#coreps_at).

**Exported by** `pythtqc`, `pythtqc.sewing`.

**Attributes**

- **`kpoint`** (`str`): The name of the point in the tables of the irreptables package,
  `"GM"`.

- **`k`** (`ndarray`): The point, in the model's reduced coordinates.

- **`bands`** (`tuple of int`): The set of bands, by their indices among the states handed in,
  ascending.

- **`coreps`** (`dict`): The multiplicities of the little-group representations of the set of
  bands at the point, `{label: multiplicity}`.

- **`multiplets`** (`tuple of Multiplet`): A row for each degenerate multiplet of the set of bands, lowest
  first, with its bands, its mean energy and its labels.

- **`isolation`** (`float`): The energy gap from the set of bands to the nearest band outside it,
  in the units of `H`.  `inf` where the set holds every band.

- **`numerical`** (`CorepsAtNumerical`): The zero thresholds at the point, and the multiplicities before
  rounding.

**Methods**

- **`summary()`**: The little-group representations, a line for each part with its attribute.

#### `Multiplet`

One degenerate multiplet at a high-symmetry point, with its labels.

A row of [`CorepsAt.multiplets`](#corepsat) and of [`HSPTable.multiplets`](#hsptable).
Its degeneracy is the number of its `bands`.

**Held in** [`CorepsAt.multiplets`](#corepsat).

**Exported by** `pythtqc.sewing`.

**Attributes**

- **`kpoint`** (`str`): The name of the point in the tables of the irreptables package,
  `"GM"`.

- **`k`** (`ndarray`): The point, in the model's reduced coordinates.

- **`bands`** (`tuple of int`): The bands of the multiplet.

- **`energy`** (`float`): The mean energy of the multiplet, in the units of `H`.

- **`corep_names`** (`tuple of str`): The labels of the little-group representations the multiplet
  carries, a representation carried twice named twice.

- **`label`** (`str`): The same labels on one line, as [`format_coreps`](functions.md#format_coreps) writes them:
  `"-GM6"`, `"-GM6 + -GM7"`, `"2x-GM6"`.

#### `CorepsAtNumerical`

How the little-group representations at one point were identified.

Held in [`CorepsAt.numerical`](#corepsat).  `summary()` prints the thresholds
and the identification, each under a line naming its attribute.

**Held in** [`CorepsAt.numerical`](#corepsat).

**Exported by** `pythtqc.sewing`.

**Attributes**

- **`thresholds`** (`Thresholds`): The two zero thresholds at the point.

- **`identification`** (`CorepIdentification`): The identification of the characters: the little-group
  representations of each multiplet, their multiplicities before
  rounding, and how far those are from integers.

**Methods**

- **`summary()`**: The thresholds and the identification, each under its attribute.

### `HSPTable`

The symmetry data vector of a set of bands, point by point.

The record [`hsp_table`](functions.md#hsp_table) returns.  [`decompose`](functions.md#decompose),
[`symmetry_indicator`](functions.md#symmetry_indicator), [`regular_completion`](functions.md#regular_completion) and
[`srsi_of_bands`](functions.md#srsi_of_bands) take it without `r`, and use the group it
holds.  `summary()` prints the multiplicities at each point, the points
refused, and the note.

**Returned by** [`hsp_table`](functions.md#hsp_table).

**Exported by** `pythtqc`, `pythtqc.sewing`.

**Attributes**

- **`bands`** (`tuple of int`): The set of bands, by their indices, ascending.

- **`group`** (`GroupData`): The group whose little-group representations the labels name,
  `r.group`.

- **`kpoints`** (`dict`): Every point [`high_symmetry_points`](functions.md#high_symmetry_points) returns, by name,
  with its reduced coordinates, the points refused among them.

- **`symmetry_data`** (`dict`): If nonempty, the symmetry data vector, `{label: multiplicity}`, over
  every point not refused.

- **`multiplets`** (`dict`): If nonempty, `{kpoint: tuple of Multiplet}`: at each point not
  refused, a row for each degenerate multiplet of the set of bands,
  lowest first, with its labels.

- **`complete`** (`bool`): Whether the little-group representations were identified at every
  point of `kpoints`.

- **`refusals`** (`dict`): If nonempty, `{kpoint: Refusal}`, the refusal returned at each point
  refused.

- **`note`** (`str`): If nonempty, a sentence describing any convention choices not fixed by
  the Hamiltonian that are needed to determine the corep labels.

**Methods**

- **`summary()`**: The labels at each point, a line for each part with its attribute.

## Atomic limits

The records of the calls that compare a symmetry data vector with the
elementary band representations of the group: the sums of them that carry it,
and the bands that complete a band set to copies of the regular band
representation.

- [`EBRDecomposition`](#ebrdecomposition): The sums of elementary band representations that carry a symmetry data vector.
- [`RegularCompletion`](#regularcompletion): The bands completing a band set to copies of the regular band representation.

### `EBRDecomposition`

The sums of elementary band representations that carry a symmetry data vector.

The record [`decompose`](functions.md#decompose) returns.  `summary()` prints whether a sum
carries the vector, and each sum.  The record is truthy even where no sum
carries the vector.

**Returned by** [`decompose`](functions.md#decompose).

**Held in** [`RegularCompletion.decomposition`](#regularcompletion).

**Exported by** `pythtqc`, `pythtqc.ebr`.

**Attributes**

- **`decomposable`** (`bool`): Whether a sum of elementary band representations with nonnegative
  integer multiplicities carries the vector.

- **`candidate_sums`** (`tuple of BandRepresentation`): If nonempty, every sum that carries the vector, up to the cap on the
  list.  The sums with the fewest elementary band representations come
  first.  Of two with as many, the one with more copies of the first
  elementary band representation the tables list comes first, then of
  the second, and so on.  Where the list stops at the cap, the sums
  listed are the first the search found, not the first in this order.
  On a layer, the sums leave out the elementary band representations
  induced from Wyckoff positions off the layer.

- **`symmetry_data`** (`dict`): The symmetry data vector decomposed, `{corep: multiplicity}`.

- **`signed_combinations`** (`SignedCombinations or None`): If not `None`, the combinations of elementary band representations
  that carry the vector, with each multiplicity of either sign and at
  most the bound in size, up to the cap on the list.  `None` where a
  sum with nonnegative multiplicities carries the vector, and with
  `signed_combinations=False`.

- **`certificate`** (`{"proof", "none within the bound"} or None`): If not `None`, what the absence of a sum rests on.  `"proof"`: no
  multiplicity can exceed the bound searched, and no sum within it
  carries the vector.  `"none within the bound"`: no sum within the
  bound carries the vector, and a sum with a larger multiplicity is not
  ruled out.  `None` where a sum carries the vector.

- **`site_content`** (`SiteContent or None`): If not `None`, the site irrep of each orbital, and how each is
  known, on which the names of the sums rest.  `None` where the
  representation holds no site irreps.

- **`numerical`** (`EBRDecompositionNumerical`): The bound searched, the cap on the list, and the little-group
  representations the vector was read on.

**Methods**

- **`summary()`**: Whether a sum carries the vector, a line for each part with its attribute.

#### `SignedCombinations`

The combinations of elementary band representations with signed multiplicities.

Held in [`EBRDecomposition.signed_combinations`](#ebrdecomposition), where no sum with
nonnegative multiplicities carries the symmetry data vector.  Each
multiplicity is searched between minus and plus the bound,
[`EBRDecompositionNumerical.bound`](#ebrdecompositionnumerical).  No bound holds for a signed
multiplicity, so a combination may have a multiplicity beyond the bound
searched.

**Held in** [`EBRDecomposition.signed_combinations`](#ebrdecomposition).

**Exported by** `pythtqc.ebr`.

**Attributes**

- **`sums`** (`tuple of BandRepresentation`): If nonempty, each combination found that carries the symmetry data
  vector, up to the cap on the list.

- **`truncated`** (`bool`): Whether there are more combinations than the cap, so that the list
  stops short of them.

#### `SiteContent`

The site irrep of each orbital, and how each is known.

Held in [`EBRDecomposition.site_content`](#ebrdecomposition) and
[`SymmetryIndicator.site_content`](#symmetryindicator).  The names of the little-group
representations, and so the sums and the class read from them, depend on
these site irreps.

**Held in** [`EBRDecomposition.site_content`](#ebrdecomposition), [`SymmetryIndicator.site_content`](#symmetryindicator).

**Exported by** `pythtqc.ebr`.

**Attributes**

- **`number_str`** (`str`): The number of the group the site irreps are named in, such as
  `"99.164"`.

- **`irreps`** (`tuple of str`): The site irrep of each orbital, in the order of the orbitals.

- **`determined_by`** (`tuple of str`): How the site irrep of each orbital is known: `"hamiltonian"`, fixed
  by the Hamiltonian; `"declaration"`, declared; `"declaration not carried"`, declared, while the Hamiltonian's own blocks carry another
  site irrep, the one named in `irreps`; `"lowest-l"`, assigned
  by the lowest-angular-momentum rule among those the hoppings allow.

- **`residual`** (`float`): The largest defect in reading the orbitals' characters off the
  matrices of the representation: how far an orbital's block is from
  its character times the operation's spin factor, or, for a multiplet
  of several orbitals, from multiplying as the site group does.  It is
  dimensionless, the entries of the matrices being at most 1 in size.

#### `EBRDecompositionNumerical`

How a decomposition was searched, and what it was read on.

Held in [`EBRDecomposition.numerical`](#ebrdecomposition).  `summary()` prints the
points left out, and whether the sums are exact without them.

**Held in** [`EBRDecomposition.numerical`](#ebrdecomposition).

**Exported by** `pythtqc.ebr`.

**Attributes**

- **`bound`** (`int`): The largest multiplicity searched: the largest entry of the symmetry
  data vector on the little-group representations read, and at least 1.

- **`bound_is_rigorous`** (`bool`): Whether every elementary band representation carries a little-group
  representation read.  Where it does, no sum has a multiplicity above
  the bound, so a search that finds no sum proves there is none.

- **`enumeration`** (`Enumeration`): The cap on the number of sums listed, and whether the list stops at it.

- **`completeness`** (`Completeness`): Whether the sums are exact or rest on the little-group representations
  kept alone, and why.

- **`restriction`** (`Restriction`): The little-group representations the vector was read on, and those
  left out.

**Methods**

- **`summary()`**: What the decomposition was read on, a line for each part with its attribute.

##### `Enumeration`

The cap on the number of sums listed, and whether the list stops at it.

Held in [`EBRDecompositionNumerical.enumeration`](#ebrdecompositionnumerical).

**Held in** [`EBRDecompositionNumerical.enumeration`](#ebrdecompositionnumerical).

**Exported by** `pythtqc.ebr`.

**Attributes**

- **`cap`** (`int`): The largest number of sums listed.

- **`n_solutions`** (`int`): The number of sums listed.

- **`truncated`** (`bool`): Whether there are more sums than the cap, so that the list stops
  short of them.

##### `Completeness`

Whether a decomposition's sums are those of the whole symmetry data vector.

Held in [`EBRDecompositionNumerical.completeness`](#ebrdecompositionnumerical).

**Held in** [`EBRDecompositionNumerical.completeness`](#ebrdecompositionnumerical).

**Exported by** `pythtqc.ebr`.

**Attributes**

- **`source`** (`{"HSPTable", "mapping"}`): `"HSPTable"` for a symmetry data vector measured in a table,
  `"mapping"` for one written out by hand.

- **`complete`** (`bool`): Whether the table refused at no point.  `True` for a vector written
  out by hand, which is taken as complete.

- **`claim`** (`{"exact", "necessary-condition"}`): `"exact"` where the sums are those of the whole symmetry data vector.
  `"necessary-condition"` where a little-group representation left out
  repeats no measured one.  The sums then carry the symmetry data vector
  on the little-group representations measured, and may not carry the
  whole vector.

- **`refused`** (`dict`): If nonempty, `{point: gate}`: each point at which the table refused,
  with the name of its refusal.  The decomposition leaves those points
  out.

- **`note`** (`str`): What the record holds, in a sentence.

##### `Restriction`

The little-group representations a model measures, and what the others repeat.

The record [`in_plane_restriction`](functions.md#in_plane_restriction) returns, and the one
[`EBRDecompositionNumerical.restriction`](#ebrdecompositionnumerical) and
[`SymmetryIndicatorNumerical.restriction`](#symmetryindicatornumerical) hold.  A little-group
representation at a point not measured, such as a layer's at
`k_z = 1/2`, may occur in every elementary band representation as often
as a measured one.  It is then said to repeat that one, and adds no
condition to a symmetry data vector.  Where every one left out repeats a
measured one, the decomposition or class read from the measured ones is
that of the whole symmetry data vector.  `summary()` prints those left
out, and the measured one each repeats.

**Returned by** [`in_plane_restriction`](functions.md#in_plane_restriction).

**Held in** [`EBRDecompositionNumerical.restriction`](#ebrdecompositionnumerical), [`SymmetryIndicatorNumerical.restriction`](#symmetryindicatornumerical).

**Exported by** `pythtqc`, `pythtqc.ebr`.

**Attributes**

- **`kpoints`** (`tuple of str`): The tabulated names of the points measured.

- **`retained`** (`tuple of str`): The little-group representations of the group's elementary band
  representations at those points, the measured ones.

- **`dropped`** (`tuple of str`): If nonempty, those at the other points, which are left out.  Empty
  where every tabulated point is measured.

- **`copies`** (`dict`): If nonempty, `{left out: (measured, ...)}`: for each little-group
  representation left out, the measured ones it repeats.  `()` for
  one that repeats no measured one.

- **`orphans`** (`tuple of str`): If nonempty, the little-group representations left out that repeat no
  measured one.  The multiplicity of each is an entry of a symmetry data
  vector that the measured ones do not fix.

- **`exact`** (`bool`): Whether every little-group representation left out repeats a measured
  one.

- **`outside_layer`** (`dict`): If nonempty, `{Wyckoff position: heights}`: the Wyckoff positions of
  a layer's group that are fixed along its normal, with a height outside
  the range of the orbitals' heights.  The heights are measured from the
  layer's middle plane along its normal, in the model's units of length.
  The band representations induced from those positions are left out.

**Methods**

- **`summary()`**: What is left out, a line for each part, with the attribute that holds it.

### `RegularCompletion`

The bands completing a band set to copies of the regular band representation.

The record [`regular_completion`](functions.md#regular_completion) returns.  `summary()` prints the
symmetry data vector of the missing bands, and the orbitals they would be.

**Returned by** [`regular_completion`](functions.md#regular_completion).

**Exported by** `pythtqc`, `pythtqc.ebr`.

**Attributes**

- **`copies`** (`int`): The number of copies of the regular band representation the completed
  bands make.

- **`bands_per_copy`** (`int`): The number of bands in one copy: the number of unitary operations of
  the group up to lattice translations, or twice that for a spinful
  model in a group with time reversal.  For a spinful model these are
  the operations of the single-valued group.

- **`padding`** (`dict`): If nonempty, the symmetry data vector of the missing bands,
  `{corep: multiplicity}`: `copies` times `regular`, less
  `symmetry_data`.

- **`decomposition`** (`EBRDecomposition`): What [`decompose`](functions.md#decompose) returns for `padding`: the orbitals the
  missing bands would be.

- **`symmetry_data`** (`dict`): The symmetry data vector of the band set handed in.

- **`regular`** (`dict`): The symmetry data vector of one copy, computed from the characters of
  the little-group representations at each point.

**Methods**

- **`summary()`**: The padding, a line for each part, with the attribute that holds it.

## Stable invariants

The records of the symmetry-indicator class of a band set and of the stable
real-space invariants (SRSIs): the tables of one group, the integer-valued
SRSIs (ZSRSIs) of a band set, the SRSIs of a model, the no-go test of a gapped
splitting, and a pair of site irreps that no stable real-space invariant tells
apart.

- [`SymmetryIndicator`](#symmetryindicator): The symmetry-indicator class of a set of bands.
- [`SRSITables`](#srsitables): The stable real-space invariants of one grey space group, as tabulated.
- [`BandSRSIs`](#bandsrsis): The integer-valued stable real-space invariants (ZSRSIs) of a band set.
- [`ModelSRSIs`](#modelsrsis): The stable real-space invariants of a model, a row for each Wyckoff orbit.
- [`Splitting`](#splitting): The no-go test of a gapped splitting of bands.
- [`StablePair`](#stablepair): Two site irreps at one position with the same stable real-space invariants.

### `SymmetryIndicator`

The symmetry-indicator class of a set of bands.

The record [`symmetry_indicator`](functions.md#symmetry_indicator) returns.  `summary()` prints the
class, its indicator group and its order, the class in the named
indicators, and the symmetry data vector.  The record is truthy whatever
its class.

**Returned by** [`symmetry_indicator`](functions.md#symmetry_indicator).

**Held in** [`BandSRSIs.indicator`](#bandsrsis).

**Exported by** `pythtqc`, `pythtqc.ebr`.

**Attributes**

- **`values`** (`tuple of int`): If nonempty, the class, one integer modulo each of `divisors`:
  the entries of `U y` at which `D` holds an elementary divisor
  above 1, each modulo that divisor.  Here `U M V = D` is the Smith
  normal form the irreptables package ships for the group's matrix
  `M` of elementary band representations, and `y` is the symmetry
  data vector.  Another Smith normal form of the same matrix can give
  the same class other values, and the same order.  Empty for a trivial
  indicator group.

- **`divisors`** (`tuple of int`): If nonempty, the elementary divisors above 1 of `M`.  The indicator
  group is the product of the cyclic groups of these orders.  Empty for
  a trivial indicator group.

- **`indicator_group`** (`str`): The indicator group as text, `"Z2 x Z4"`, or `"trivial"`.

- **`indicators`** (`dict or None`): If not `None`, the class in the named indicators of Hwang *et al.*,
  Nat. Commun. **17**, 8564 (2026), `{name: value}`.  They are
  computed from the integer-valued stable real-space invariants (ZSRSIs)
  of the symmetry data vector.  `None` in three cases: a group with
  no time reversal, which the paper does not tabulate; a trivial
  indicator group; and a group for which the paper's tables are not
  installed or cannot be read.
  [`SymmetryIndicatorNumerical.indicators_note`](#symmetryindicatornumerical) says which.

- **`order`** (`int`): The order of the class: the smallest number of copies of the set of
  bands that together have the zero class, `1` for the zero class.

- **`symmetry_data`** (`dict`): The symmetry data vector, `{corep: multiplicity}`.  For a layer, the
  multiplicities at the points of its plane alone.

- **`site_content`** (`SiteContent or None`): If not `None`, the site irrep of each orbital, and how each is
  known, on which the names of the little-group representations rest.
  Where one is `"lowest-l"`, another declaration of the orbitals can
  give the bands another class.  `None` where the representation holds
  no site irreps.

- **`group`** (`GroupData`): The group in which the class is computed, `r.group`.

- **`numerical`** (`SymmetryIndicatorNumerical`): The Smith normal form computed again, the symmetry data vector on
  every point of the group's tables, and how a layer's little-group
  representations out of its plane were read.

**Methods**

- **`summary()`**: The class, a line for each part, with the attribute that holds it.

#### `SymmetryIndicatorNumerical`

How a symmetry-indicator class was read.

Held in [`SymmetryIndicator.numerical`](#symmetryindicator).  `summary()` prints the
symmetry data vector on every point of the group's tables, how the
little-group representations a layer does not measure were obtained, and
the check of the Smith normal form.

**Held in** [`SymmetryIndicator.numerical`](#symmetryindicator).

**Exported by** `pythtqc.ebr`.

**Attributes**

- **`verification`** (`SmithVerification`): The Smith normal form of the group's matrix of elementary band
  representations, computed again in exact integers.

- **`lattice_check`** (`LatticeCheck`): The compatibility relations, checked on the symmetry data vector.

- **`restriction`** (`Restriction`): The little-group representations measured, and those left out.

- **`reconstructed`** (`dict`): If nonempty, `{left out: measured}`: each little-group
  representation left out that the symmetry data vector does not name,
  and the measured one whose multiplicity it takes.  The two occur as
  often as each other in every elementary band representation the
  vector was read on.

- **`completed_symmetry_data`** (`dict`): The symmetry data vector on every point of the group's tables,
  `{corep: multiplicity}`, the zeros dropped.  A layer's little-group
  representations out of its plane are measured on its decoupled stack,
  or take the multiplicity of the one in the plane they repeat.  A
  vector written out by hand is read as given at each point out of the
  plane where it names a little-group representation.

- **`completion`** (`StackCompletion or None`): If not `None`, how a layer's little-group representations out of its
  plane were measured on its decoupled stack.  `None` where none had
  to be.

- **`two_dimensional`** (`TwoDimensionalReading or None`): If not `None`, how a layer's class was read.  `None` for a crystal
  periodic in three directions, for a trivial indicator group, and where
  there is no point out of the layer's plane to read.

- **`free_standing_group`** (`groups.NamedGroup or None`): If not `None`, the group of the same layer free-standing, with
  vacuum on both sides, as [`realized_group`](functions.md#realized_group) measured it
  beside the group on a substrate.  It is given where it is larger than
  the group in which the class is computed.

- **`convention`** (`str`): `INDICATOR_CONVENTION`, a statement that the class is computed
  in the convention in which a lattice translation is represented by
  `e^(-ik.t)`, and is the same in the convention of the tables, in
  which a lattice translation is represented by `e^(+ik.t)`.

- **`indicators_note`** (`str`): Where [`SymmetryIndicator.indicators`](#symmetryindicator) came from, or why it is
  `None`.

- **`group_provenance`** (`str`): `"realized"` where the group is that of the model's cell;
  `"descended"` where it was lowered from the group of the cell by the
  operations the Hamiltonian breaks.

**Methods**

- **`summary()`**: How the class was read, a line for each part, with the attribute that holds it.

##### `SmithVerification`

The Smith normal form of the group's matrix, computed again in exact integers.

Held in [`SymmetryIndicatorNumerical.verification`](#symmetryindicatornumerical).  The matrix is
the group's matrix of elementary band representations, whose columns are
their symmetry data vectors.  Its Smith normal form is computed again and
compared with the one the irreptables package ships with it.  A record
returned holds two forms with the same elementary divisors, in which the
class has the same order.

**Held in** [`SymmetryIndicatorNumerical.verification`](#symmetryindicatornumerical).

**Exported by** `pythtqc.ebr`.

**Attributes**

- **`divisors_denovo`** (`tuple of int`): The elementary divisors above 1 of the form computed again.

- **`agrees`** (`bool`): Whether the two forms have the same elementary divisors and give the
  class the same order.  `True` in a record returned.

- **`shipped_is_smith`** (`bool`): Whether the two unimodular matrices the irreptables package ships
  bring the matrix to the diagonal form it ships.  `True` in a record
  returned.

- **`denovo_is_smith`** (`bool`): Whether the form computed again is a Smith normal form of the matrix.
  `True` in a record returned.

- **`values_denovo`** (`tuple of int`): The class read in the form computed again: the entries of its
  `U y` ([`SymmetryIndicator.values`](#symmetryindicator)) at its elementary divisors
  above 1, each modulo that divisor.

- **`order_denovo`** (`int`): The order of the class, computed again.

- **`change_of_basis`** (`tuple`): The automorphism of the indicator group that carries
  [`SymmetryIndicator.values`](#symmetryindicator) to `values_denovo`: a matrix
  with a row for each entry of `values_denovo` and a column for
  each entry of [`SymmetryIndicator.values`](#symmetryindicator), each element modulo
  the elementary divisor of its row.

- **`change_of_basis_is_identity`** (`bool`): Whether that automorphism is the identity.

- **`note`** (`str`): What the record holds, in a sentence.

##### `LatticeCheck`

The compatibility relations, checked exactly on the symmetry data vector.

Held in [`SymmetryIndicatorNumerical.lattice_check`](#symmetryindicatornumerical).  The
compatibility relations require the entries of `U y` at the rows where
`D` is zero to vanish for every band structure of the group.
Here `U M V = D` is the Smith normal form the irreptables package ships
for the group's matrix `M` of elementary band representations, and
`y` is the symmetry data vector.

**Held in** [`SymmetryIndicatorNumerical.lattice_check`](#symmetryindicatornumerical).

**Exported by** `pythtqc.ebr`.

**Attributes**

- **`in_lattice`** (`bool`): Whether the symmetry data vector satisfies every compatibility
  relation.  `True` in a record returned.

- **`residual`** (`int`): The largest absolute value of an entry of `U y` that a compatibility
  relation requires to vanish.  Zero in a record returned.

- **`relations`** (`int`): The number of compatibility relations of the group.

##### `StackCompletion`

The measurement of a layer's little-group representations at `k_z = 1/2`.

Held in [`SymmetryIndicatorNumerical.completion`](#symmetryindicatornumerical).  The decoupled
stack is the layer repeated along its normal with no hopping between
copies, each orbital at its own height.  It obeys the layer's own group,
and the little-group representations at `k_z = 1/2` are measured on it.

**Held in** [`SymmetryIndicatorNumerical.completion`](#symmetryindicatornumerical).

**Exported by** `pythtqc.ebr`.

**Attributes**

- **`route`** (`str`): `"decoupled stack"`.

- **`outcome`** (`str`): `"completed"`.

- **`stopped_at`** (`None`): Where the measurement stopped.  `None` in a record returned.

- **`stop_reason`** (`None`): Why the measurement stopped.  `None` in a record returned.

- **`stack_period`** (`float`): The stack's period, in the model's units of length: the layer's
  extent along its normal plus the vacuum between the copies.

- **`stack_extent`** (`float`): The layer's extent along its normal, in the model's units of length.

- **`environment`** (`str or None`): The layer's environment, `"free-standing"` or `"substrate"`.

- **`note`** (`str`): What the record holds, in a sentence.

- **`input`** (`StackInput`): The layer as handed in.

- **`stack`** (`StackGroup`): The stack's group.

- **`gates`** (`StackChecks`): Each check that ties the stack to the layer, with the number it
  measured.

###### `StackInput`

The layer whose decoupled stack was built, as handed in.

Held in [`StackCompletion.input`](#stackcompletion).

**Held in** [`StackCompletion.input`](#stackcompletion).

**Exported by** `pythtqc.ebr`.

**Attributes**

- **`group`** (`groups.NamedGroup`): The layer's group.

- **`dim_k`** (`int`): The number of the layer's periodic directions.

- **`periodic_dirs`** (`tuple of int`): The layer's periodic directions, by lattice vector.

- **`vacuum_dirs`** (`tuple of int`): The lattice vector along the layer's normal.

- **`kpoints`** (`tuple of str`): The tabulated names of the points the layer's table measured.

- **`orphans`** (`tuple of str`): The little-group representations out of the layer's plane that repeat
  none in it, as [`Restriction.orphans`](#restriction) holds them.

- **`symmetry_data`** (`dict`): The layer's symmetry data vector, `{corep: multiplicity}`.

- **`n_represented`** (`int`): The number of operations the layer's representation holds a matrix
  for.

###### `StackGroup`

The decoupled stack's group: the layer's own, carried onto the stack's cell.

Held in [`StackCompletion.stack`](#stackcompletion).

**Held in** [`StackCompletion.stack`](#stackcompletion).

**Exported by** `pythtqc.ebr`.

**Attributes**

- **`group`** (`groups.NamedGroup`): The group.

- **`dim_k`** (`int`): The number of the stack's periodic directions, three.

- **`periodic_dirs`** (`tuple of int`): The stack's periodic directions, by lattice vector.

- **`typat`** (`tuple of int`): The species of each orbital's site.

- **`divisors`** (`tuple of int`): The elementary divisors above 1 of the group's matrix of elementary
  band representations.

###### `StackChecks`

Each check that ties the stack to the layer, with the number it measured.

Held in [`StackCompletion.gates`](#stackcompletion).

**Held in** [`StackCompletion.gates`](#stackcompletion).

**Exported by** `pythtqc.ebr`.

**Attributes**

- **`hamiltonian_identity`** (`HamiltonianIdentity`): The stack's Hamiltonian compared with the layer's.

- **`no_hopping_across_the_stack`** (`StackHoppings`): The hoppings between the copies.

- **`representation`** (`RepresentationCheck`): The layer's matrices checked on the stack's Hamiltonian.

- **`characters`** (`CharactersCheck`): The band characters of the layer and of the stack at the k-points
  they share.

- **`fibers`** (`tuple of StackFiber`): The stack's bands at each point at `k_z = 1/2`, compared with those
  at the point of the plane with the same in-plane k.

###### `HamiltonianIdentity`

The decoupled stack's Hamiltonian compared with the layer's.

Held in [`StackChecks.hamiltonian_identity`](#stackchecks).  Each orbital sits at
its height `h` along the normal, and the stack's period is `C`.  The
stack's `H(k, k_z)` is then the layer's `H(k)` in the gauge of the
phases `exp(2 pi i k_z h / C)` of the orbitals.  The two are compared in
that gauge.

**Held in** [`StackChecks.hamiltonian_identity`](#stackchecks).

**Exported by** `pythtqc.ebr`.

**Attributes**

- **`residual`** (`float`): The largest difference between the two over the in-plane k-points and
  the values of `k_z` probed, in the units of H.

- **`threshold`** (`float`): The largest residual allowed, in the units of H: zero for a layer
  whose orbitals lie in one plane, otherwise the rounding the phases
  carry.

- **`at_zero`** (`float`): The difference at `k_z = 0`, where every phase is 1.  Zero in a
  record returned.

- **`n_k`** (`int`): The number of in-plane k-points probed.

- **`k_z`** (`tuple of float`): The values of `k_z` probed, in reduced coordinates.

- **`note`** (`str`): What the record holds, in a sentence.

###### `StackHoppings`

The hoppings between the copies of the decoupled stack: none.

Held in [`StackChecks.no_hopping_across_the_stack`](#stackchecks).

**Held in** [`StackChecks.no_hopping_across_the_stack`](#stackchecks).

**Exported by** `pythtqc.ebr`.

**Attributes**

- **`n_hoppings`** (`int`): The number of the layer's hoppings.

- **`worst`** (`float`): The largest hopping between copies.  Zero in a record returned.

- **`note`** (`str`): What the record holds, in a sentence.

###### `RepresentationCheck`

The matrices of the layer's representation, checked on the stack's Hamiltonian.

Held in [`StackChecks.representation`](#stackchecks).

**Held in** [`StackChecks.representation`](#stackchecks).

**Exported by** `pythtqc.ebr`.

**Attributes**

- **`covariance`** (`float`): The largest covariance residual, `max |D H(k) D^-1 - H(Rk)|` over
  every operation and the k-points probed, in the units of H.  Here
  `D` is the operation's matrix on the orbitals and `R` its
  rotation.

- **`threshold`** (`float`): The largest residual allowed, the rounding the arithmetic of the
  matrices carries, in the units of H.

- **`n_operations`** (`int`): The number of operations checked.

- **`n_k`** (`int`): The number of k-points probed.

- **`note`** (`str`): What the record holds, in a sentence.

###### `CharactersCheck`

The band characters of the layer and of its stack at the k-points they share.

Held in [`StackChecks.characters`](#stackchecks).  Each point of the layer's table
is paired with the point of the stack's table in the layer's plane at the
same reduced k.  The characters of the bands are compared there,
operation by operation.

**Held in** [`StackChecks.characters`](#stackchecks).

**Exported by** `pythtqc.ebr`.

**Attributes**

- **`worst`** (`float`): The largest difference between two characters compared.

- **`threshold`** (`float`): The largest difference allowed.

- **`n_comparisons`** (`int`): The number of characters compared.

- **`shared_kpoints`** (`tuple`): `(layer's point, stack's point)` for each pair compared, by
  tabulated name.

- **`labels_equal`** (`bool`): Whether the two tables give the same little-group representations,
  with the same multiplicities, at each pair.

- **`note`** (`str`): What the record holds, in a sentence.

###### `StackFiber`

The stack's bands at one point at `k_z = 1/2`, against those in the plane.

Held in [`StackChecks.fibers`](#stackchecks).  The bands at a point at
`k_z = 1/2` are compared with those at the point of the plane with the
same in-plane k.  With no hopping between the copies, the two are the
same states, with the same energies, multiplets and characters.

**Held in** [`StackChecks.fibers`](#stackchecks).

**Exported by** `pythtqc.ebr`.

**Attributes**

- **`kpoint`** (`str`): The tabulated name of the point at `k_z = 1/2`.

- **`in_plane_kpoint`** (`str`): The tabulated name of the point in the plane it is compared with.

- **`energy_defect`** (`float`): The largest difference between the energies at the two points, in the
  units of H.  Infinite where the two hold different numbers of
  multiplets.

- **`character_defect`** (`float`): The largest difference between the characters at the two points, over
  the operations both little groups hold.

- **`n_shared_operations`** (`int`): The number of operations both little groups hold.

- **`partition_equal`** (`bool`): Whether the bands at the two points split into multiplets alike.

- **`multiplicities_equal`** (`bool`): Whether the two points carry the same multiplicities, whichever
  little-group representations carry them.

##### `TwoDimensionalReading`

How a layer's symmetry-indicator class is read, as the class of its stack.

Held in [`SymmetryIndicatorNumerical.two_dimensional`](#symmetryindicatornumerical).  A layer's
class is read in a three-dimensional space group, whose tables have points
out of the layer's plane, at `k_z = 1/2`.  It is the class of the
layer's decoupled stack, the layer repeated along its normal with no
hopping between copies.

**Held in** [`SymmetryIndicatorNumerical.two_dimensional`](#symmetryindicatornumerical).

**Exported by** `pythtqc.ebr`.

**Attributes**

- **`applied`** (`bool`): `True` in every record returned.

- **`dim_k`** (`int`): The number of the layer's periodic directions.

- **`periodic_dirs`** (`tuple of int`): The layer's periodic directions, by lattice vector.

- **`vacuum_dirs`** (`tuple of int`): The lattice vector along the layer's normal.

- **`presentation`** (`{"SLAB", "decoupled-STACK"}`): `"decoupled-STACK"` where the little-group representations out of
  the plane were measured on the decoupled stack, `"SLAB"` where none
  was.  The class is read in the layer's own group in both cases.

- **`out_of_plane_kpoints`** (`tuple of str`): The tabulated points out of the layer's plane.

- **`out_of_plane_labels`** (`tuple of str`): The little-group representations at those points.

- **`components`** (`tuple of IndicatorComponent`): Each component of the class, and whether it has a coefficient on a
  little-group representation out of the plane.

- **`support`** (`dict`): If nonempty, `{indicator: where}`: for each named indicator, as
  [`SymmetryIndicator.indicators`](#symmetryindicator) names it, the little-group
  representations on which its coefficients are nonzero.
  `"in-plane only"` for those in the plane alone,
  `"out-of-plane only"` for those out of it alone, `"mixed"` for
  both.

- **`note`** (`str`): What the record holds, in a sentence.

###### `IndicatorComponent`

One component of a layer's symmetry-indicator class.

Held in [`TwoDimensionalReading.components`](#twodimensionalreading).  A component is one
entry of `U y`, taken modulo its elementary divisor
([`SymmetryIndicator.values`](#symmetryindicator)).

**Held in** [`TwoDimensionalReading.components`](#twodimensionalreading).

**Exported by** `pythtqc.ebr`.

**Attributes**

- **`slot`** (`int`): The index of the entry of `U y`.

- **`divisor`** (`int`): The elementary divisor the component is taken modulo.

- **`value`** (`int`): The component, modulo `divisor`.

- **`reads_out_of_plane`** (`bool`): Whether the entry, a linear form in the symmetry data vector, has a
  nonzero coefficient on a little-group representation at a point out of
  the layer's plane.

### `SRSITables`

The stable real-space invariants of one grey space group, as tabulated.

The record [`srsi_tables`](functions.md#srsi_tables) returns.  The tables are those of Hwang et
al., Stable real-space invariants and topology beyond symmetry indicators,
Nat. Commun. 17, 8564 (2026).  Each invariant is a linear function of the
site-symmetry vector `p`, the multiplicity of each site irrep at each
Wyckoff position.  `summary` prints the invariants and
`formulas` each one's formula.  `theta` evaluates them on a
site-symmetry vector, and `indicators_of` evaluates the paper's
symmetry indicators.

**Returned by** [`srsi_tables`](functions.md#srsi_tables).

**Exported by** `pythtqc`, `pythtqc.srsi`.

**Attributes**

- **`group`** (`GroupData`): The group the tables are of.  Where [`srsi_tables`](functions.md#srsi_tables) is passed a
  measured group, it is that group.  Where it is passed a space-group
  number, it is the grey group of that number, with its name, number,
  valuedness and magnetic type, and no cell: `periodic_dirs` is
  `None`.

- **`invariants`** (`dict`): `{name: formula}`: each invariant in the paper's order, as a sum of
  terms `c (irrep)_w`.  A term is `c` times the multiplicity of the
  site irrep `irrep` at the Wyckoff position of letter `w`.  In space
  group 99 with spin-orbit coupling, `invariants["theta_4"]` is
  `"1 (-E2)_b"`.

- **`indicators`** (`dict`): If nonempty, `{name: form}`: each of the paper's symmetry indicators
  as the paper prints it, a linear combination of the invariants called
  its form.  In space group 166 with spin-orbit coupling,
  `indicators["z4"]` is
  `"-4 theta_1 + 4 theta_2 - 4 theta_3 + 4 theta_4"`.  Empty where the
  group's symmetry-indicator group is trivial.

- **`moduli`** (`dict`): `{name: modulus}`: the modulus of each invariant and each
  indicator, `None` for an integer-valued invariant.

- **`site_irreps`** (`tuple of SiteIrrep`): Every site irrep the formulas count, at a maximal Wyckoff position or
  not: the coordinates of `p`.

- **`split_ebrs`** (`tuple of SplitEBR`): If nonempty, each splitting the paper lists of an elementary band
  representation into others, one SplitEBR each.

- **`stably_equivalent`** (`tuple of StablePair`): If nonempty, each pair of site irreps at one maximal Wyckoff position
  that no stable real-space invariant distinguishes.

- **`provenance`** (`Provenance`): The paper the tables come from, and what the conversion read and
  counted.

- **`numerical`** (`SRSITablesNumerical`): How each symmetry indicator of the tables was checked, by name.

**Methods**

- **`theta(p)`**: The stable real-space invariants of a site-symmetry vector.
- **`theta_unreduced(p)`**: Every invariant of a site-symmetry vector, none reduced modulo n.
- **`indicators_of(p)`**: The paper's symmetry indicators of a site-symmetry vector.
- **`summary()`**: The invariants and what else the tables hold, a line each.
- **`formulas()`**: Every invariant's formula, a line each, under a line saying what a term is.

#### `SiteIrrep`

A site irrep at a Wyckoff position: one coordinate of the site-symmetry vector.

A site irrep at a maximal Wyckoff position induces an elementary band
representation, unless the band representation it induces is a sum of
others ($`{}^1E{}^2E`$ at site symmetry $`\bar4`$, `E` at
$`\bar4m2`$ and $`\bar42m`$).  One at a non-maximal position
induces a band representation equivalent to a sum of those induced at a
maximal position whose site-symmetry group contains its own.  An entry of
`tables.site_irreps`.

**Held in** [`SRSITables.site_irreps`](#srsitables), [`SplitEBR.parent`](#splitebr), [`SplitEBR.children`](#splitebr).

**Exported by** `pythtqc.srsi`.

**Attributes**

- **`irrep`** (`str`): The site irrep, spelled as the irreptables package's list of
  elementary band representations spells it where the list has it, and
  as the paper spells it elsewhere.  At a `222`
  site the list writes `A` where the paper writes `A1`.

- **`wyckoff`** (`str`): The Wyckoff position in the setting the tables are written in, with
  its multiplicity in the conventional cell, `"4d"`.

- **`maximal`** (`bool`): Whether the Wyckoff position is maximal.

- **`ebr`** (`bool`): Whether the site irrep induces an elementary band representation.

#### `SplitEBR`

One splitting of an elementary band representation, as the paper lists it.

The symmetry data vector of the whole is the sum of the parts', every
part an elementary band representation.  The paper marks whether the
invariants valued modulo n of the parts, each counted once, add up to the
whole's.  An entry of `tables.split_ebrs`; `summary` prints it.

**Held in** [`SRSITables.split_ebrs`](#srsitables).

**Exported by** `pythtqc.srsi`.

**Attributes**

- **`parent`** (`SiteIrrep`): The site irrep that induces the whole.

- **`children`** (`tuple of SiteIrrep`): The site irreps that induce the parts, two or more.

- **`label`** (`str`): The splitting as the paper writes it: `"(-E)_c = (-E2)_a + (-E1)_a"`
  in space group 99 with spin-orbit coupling.

- **`srsi_mismatch`** (`bool`): The paper's mark: whether the parts' invariants valued modulo n do not
  add up to the whole's.

**Methods**

- **`summary()`**: The splitting, and whether its parts' invariants add up, a line each.

#### `Provenance`

The paper the tables come from, and what the conversion read and counted.

The `provenance` of an [`SRSITables`](#srsitables), the same for every group of
one valuedness.

**Held in** [`SRSITables.provenance`](#srsitables).

**Exported by** `pythtqc.srsi`.

**Attributes**

- **`paper`, `doi`** (`str`): The paper the tables are converted from, and its DOI.

- **`valuedness`** (`str`): `"single"` or `"double"`: the single-valued or the double-valued
  tables.

- **`converted`** (`str`): The date the tables were converted, `"YYYY-MM-DD"`.

- **`irrep_version`, `irreptables_version`** (`str`): The versions of the irrep and irreptables packages whose elementary
  band representations the conversion read.

- **`omitted_groups`** (`tuple of int`): The thirteen space groups the tables leave out.

- **`omitted_reason`** (`str`): Why they are left out, as the caption of the paper's tables states it.

- **`label_map`** (`dict`): `{TeX spelling: spelling}`: each site irrep of the paper, as the
  package spells it, `{"A_{1g}": "A1g"}`.

- **`label_aliases`** (`tuple of dict`): If nonempty, each site irrep the paper and the irreptables package's
  list of elementary band representations spell differently: its site
  symmetry (`site`), the paper's spelling (`supplement`), the list's
  (`ebr_label`), and the number of Wyckoff positions over all the
  groups where it occurs (`positions`).

- **`composite_site_irreps`** (`tuple of dict`): If nonempty, each site irrep at a maximal Wyckoff position whose band
  representation is a sum of others, with its site symmetry, the
  paper's spelling and its number of Wyckoff positions, keyed as in
  `label_aliases`.

- **`counts`** (`ProvenanceCounts`): What the conversion counted.

- **`notes`** (`tuple of str`): What the conversion says about the tables, a paragraph each.

##### `ProvenanceCounts`

What the conversion of the paper's tables counted, for one valuedness.

The `counts` of a [`Provenance`](#provenance).  Each count runs over the 217
space groups the tables of that valuedness hold.

**Held in** [`Provenance.counts`](#provenance).

**Exported by** `pythtqc.srsi`.

**Attributes**

- **`groups`** (`int`): The number of space groups tabulated.

- **`srsi_rows`** (`int`): The number of invariants.

- **`zn_rows`** (`dict`): `{modulus: count}`: the number of invariants valued modulo n, by n,
  the modulus written as a string, `{"2": 62}`.

- **`terms`** (`int`): The number of terms in the invariants' formulas.

- **`terms_on_ebr_columns`** (`int`): The number of terms on a site irrep that induces an elementary band
  representation.

- **`terms_at_maximal_wyckoff_without_an_ebr`** (`int`): The number of terms on a site irrep at a maximal Wyckoff position
  whose band representation is a sum of others.

- **`terms_at_nonmaximal_wyckoff`** (`int`): The number of terms on a site irrep at a non-maximal Wyckoff position.

- **`alias_positions`** (`int`): The number of site irreps the paper and the irreptables package's list
  of elementary band representations spell differently, counted at each
  Wyckoff position where one occurs.

- **`composite_positions`** (`int`): The number of site irreps at a maximal Wyckoff position whose band
  representation is a sum of others, counted the same way.

- **`si_groups`** (`int`): The number of space groups with a symmetry indicator.

- **`si_expressions`** (`int`): The number of symmetry indicators.

- **`si_moduli`** (`dict`): `{modulus: count}`: the number of symmetry indicators, by modulus,
  the modulus written as a string.

- **`split_groups`** (`int`): The number of space groups with a splitting of an elementary band
  representation.

- **`splittings`** (`int`): The number of splittings.

- **`splittings_srsi_mismatch`, `splittings_srsi_match`** (`int`): The number of splittings whose parts' invariants valued modulo n do
  not add up to the whole's, and the number whose do.

- **`ebr_columns`** (`int`): The number of elementary band representations.

- **`si_on_ebr_checks`** (`int`): The number of evaluations of a symmetry indicator on an elementary
  band representation.

- **`si_expressions_not_vanishing_on_ebrs`** (`int`): The number of forms the paper prints for a symmetry indicator that do
  not vanish on every elementary band representation of their group.

- **`si_failing_groups`** (`tuple of int`): The space groups those forms belong to.

- **`kernel_checks`** (`int`): The number of space groups in which the integer-valued invariants
  were checked to vanish on every rational combination of elementary
  band representations with a zero symmetry data vector.

- **`derived_indicator_rows`** (`dict`): `{relation: count}`: the number of symmetry indicators by how the
  printed form compares with the derived one, `"none"` where no form
  is derived.

#### `SRSITablesNumerical`

How each symmetry indicator of the tables was checked.

The `numerical` of an [`SRSITables`](#srsitables).

**Held in** [`SRSITables.numerical`](#srsitables).

**Exported by** `pythtqc.srsi`.

**Attributes**

- **`indicators`** (`dict`): `{name: IndicatorCheck}`: the check of each symmetry indicator, by
  the paper's name.

##### `IndicatorCheck`

How one symmetry indicator of the tables was checked.

The form the paper prints for the indicator, a linear combination of the
invariants, is evaluated on every elementary band representation of the
group.  An indicator vanishes on each of them, since each is an atomic
limit.  Where it can be, a second form in the invariants is derived and
compared with the printed one.  On every band set the derived form takes,
modulo the indicator's modulus, the value of the indicator of the same
name that the irreptables package ships.  Where irreptables ships none
and the symmetry-indicator group is cyclic, it takes instead the band
set's symmetry-indicator class, a single number modulo the order of that
cyclic group, read off the Smith normal form of the matrix of elementary
band representations.  An entry of `tables.numerical.indicators`.

**Held in** [`SRSITablesNumerical.indicators`](#srsitablesnumerical).

**Exported by** `pythtqc.srsi`.

**Attributes**

- **`relation`** (`str or None`): If not `None`, how the printed form compares with the derived one.
  `"equal"`: the same coefficients.  `"equivalent"`: other
  coefficients, with the same value modulo the indicator's modulus on
  every band set.  `"different"`: a value that differs on some band
  set.

- **`source`** (`str or None`): If not `None`, what the form is derived from:
  `"irreptables:<name>"`, the indicator of that name the irreptables
  package ships, or `"smith"`, the symmetry-indicator class read off
  the Smith normal form.

- **`derived`** (`str or None`): If not `None`, the derived form in the invariants, written as
  `tables.indicators` writes the printed one.

- **`derived_note`** (`str`): The derivation in a sentence, or why there is none.

- **`ebr_failures`** (`tuple`): If nonempty, each elementary band representation the printed form
  does not vanish on, as `(site irrep, Wyckoff position, value)`:
  `("1Eg2Eg", "1a", "1/2")` for `z2,3` of space group 147 without
  spin-orbit coupling.

- **`vanishes_on_ebrs`** (`bool`): Whether the printed form is zero modulo its modulus on every
  elementary band representation of the group.

#### `Invariants`

The stable real-space invariants of a site-symmetry vector, by name.

The record [`SRSITables.theta`](#srsitables) returns.

**Returned by** [`SRSITables.theta()`](#srsitables).

**Exported by** `pythtqc.srsi`.

**Attributes**

- **`theta_Z`** (`dict`): `{name: value}`: each integer-valued invariant (ZSRSI), in the
  paper's order, as an exact rational number.  For a rational
  site-symmetry vector whose symmetry data vector has a nonzero
  symmetry indicator, some are not integers.

- **`theta_Zn`** (`dict`): `{name: ModularValue}`: each invariant valued modulo n.  Two
  site-symmetry vectors with the same symmetry data vector can differ in
  them.

##### `ModularValue`

One stable real-space invariant valued modulo n, with its modulus.

Every `theta_Zn` field maps an invariant's name to one.

**Held in** [`ModelSRSIs.theta_Zn`](#modelsrsis), [`Splitting.theta_Zn`](#splitting), [`StablePair.theta_Zn`](#stablepair), [`Invariants.theta_Zn`](#invariants), [`WyckoffOrbitSRSIs.theta_Zn`](#wyckofforbitsrsis), [`Localization.theta_Zn`](#localization), [`AtomicRealization.theta_Zn`](#atomicrealization).

**Exported by** `pythtqc.srsi`.

**Attributes**

- **`value`** (`int or Fraction`): The value, reduced modulo `modulus` where it is an integer.  A
  value that is not an integer, which a rational site-symmetry vector
  can give, is not reduced.

- **`modulus`** (`int`): The modulus n.

### `BandSRSIs`

The integer-valued stable real-space invariants (ZSRSIs) of a band set.

The record [`srsi_of_bands`](functions.md#srsi_of_bands) returns.  `summary()` prints the ZSRSIs,
whether they are fractional, and the symmetry-indicator class.
`line` prints the same on one line.  The record holds no stable
real-space invariants valued modulo n, which the symmetry data vector does
not fix.

**Returned by** [`srsi_of_bands`](functions.md#srsi_of_bands).

**Exported by** `pythtqc`, `pythtqc.srsi`.

**Attributes**

- **`theta_Z`** (`dict`): The ZSRSIs, `{name: value}`, as exact fractions in the order of the
  tables.

- **`indicator`** (`SymmetryIndicator`): The symmetry-indicator class of the same bands.

- **`group`** (`GroupData`): The group the symmetry data vector was read in.

- **`numerical`** (`BandSRSIsNumerical`): The rational sum of elementary band representations the ZSRSIs were
  computed from.

**Methods**

- **`summary()`**: The ZSRSIs, whether they are fractional, and the class, a line each.
- **`line()`**: The record on one line, for a table of band sets.

#### `BandSRSIsNumerical`

A rational sum of elementary band representations, for a band set.

The sum has the symmetry data vector of the band set, and it is one of
many sums that do.  Held in [`BandSRSIs.numerical`](#bandsrsis).  `summary()`
prints the sum.

**Held in** [`BandSRSIs.numerical`](#bandsrsis).

**Exported by** `pythtqc.srsi`.

**Attributes**

- **`p`** (`dict`): The sum, `{(site irrep, Wyckoff position): multiplicity}`, each
  elementary band representation written as the site irrep inducing it.
  Its multiplicities are rational, and the zeros are dropped.  Its keys
  are those [`SRSITables.theta`](#srsitables) takes, in the order of the group's
  list of elementary band representations.

**Methods**

- **`summary()`**: The sum on one line.

### `ModelSRSIs`

The stable real-space invariants of a model, a row for each Wyckoff orbit.

The record [`srsi_of_model`](functions.md#srsi_of_model) returns.  `summary()` prints the
model's stable real-space invariants (SRSIs), then each Wyckoff orbit's.
The model's orbitals are an atomic configuration, so its integer-valued
SRSIs are integers.  Its SRSIs valued modulo n are returned too.

**Returned by** [`srsi_of_model`](functions.md#srsi_of_model).

**Exported by** `pythtqc`, `pythtqc.srsi`.

**Attributes**

- **`theta_Z`** (`dict`): The integer-valued SRSIs, `{name: value}`, summed over the rows.
  They equal those [`srsi_of_bands`](functions.md#srsi_of_bands) returns for all the model's
  bands.

- **`theta_Zn`** (`dict`): If nonempty, the SRSIs valued modulo n, `{name: ModularValue}`,
  summed over the rows modulo each n.  Empty where the group has none.

- **`group`** (`GroupData`): The group the SRSIs were read in, the one
  [`realized_group`](functions.md#realized_group) returned.

- **`wyckoff_orbits`** (`tuple of WyckoffOrbitSRSIs`): A row for each Wyckoff orbit, in the order of the rows of the
  atomic-limit table that [`realized_group`](functions.md#realized_group) returned.

**Methods**

- **`summary()`**: The model's SRSIs, then each Wyckoff orbit's, a line each.

#### `WyckoffOrbitSRSIs`

One Wyckoff orbit of a model's orbitals, with its stable real-space invariants.

A row of the `wyckoff_orbits` that [`srsi_of_model`](functions.md#srsi_of_model) returns.  It
holds every field of the same orbitals' row of the atomic-limit table
`r.wyckoff_orbits`, with `r` what [`realized_group`](functions.md#realized_group)
returned, and with the same values.  Among them are `wyckoff`,
`site_symmetry`, `orbitals` and `terms`.  It also holds the fields
below.

**Held in** [`ModelSRSIs.wyckoff_orbits`](#modelsrsis).

**Exported by** `pythtqc.srsi`.

**Attributes**

- **`theta_Z`** (`dict`): The row's integer-valued stable real-space invariants,
  `{name: value}`.

- **`theta_Zn`** (`dict`): If nonempty, the row's stable real-space invariants valued modulo n,
  `{name: ModularValue}`.  Empty where the group has none.

- **`slides_onto`** (`tuple of str`): The Wyckoff positions the orbitals can slide onto, and at which the
  tables list elementary band representations, `("3a", "3b")`.  The
  orbitals slide over the points the free coordinates of their Wyckoff
  position sweep, and so onto the positions of higher symmetry among
  them.  The row's own position is among those returned where the
  tables list one there.

The fields `wyckoff`, `site_symmetry`, `orbitals`, `label`, `terms`, `unitary_site_symmetry`, `determined_by`, `declared`, `band_representation`, `site_multiplets` are those of [`WyckoffOrbit`](#wyckofforbit), with the entries there.

**Methods**

- **`summary()`**: The row, a line for each part that applies, with the attribute that holds it.

##### `WyckoffOrbitSRSIsNumerical`

The decompositions behind one Wyckoff orbit's stable real-space invariants.

Held in `WyckoffOrbitSRSIs.numerical`.  It holds every field of the
`numerical` of the same orbitals' row of the atomic-limit table
`r.wyckoff_orbits`, with `r` what [`realized_group`](functions.md#realized_group)
returned.  It also holds the fields below.

**Held in** [`WyckoffOrbitSRSIs.numerical`](#wyckofforbitsrsis).

**Exported by** `pythtqc.srsi`.

**Attributes**

- **`localizations`** (`tuple of Localization`): Every decomposition of `symmetry_data` into elementary band
  representations at one Wyckoff position the orbitals can slide onto.
  For a row with `composite`, one entry with no terms.

- **`chosen`** (`Localization`): The first of `localizations`.  The row's stable real-space
  invariants are read from it.

- **`stably_equivalent`** (`tuple of StablePair`): If nonempty, the pairs of site irreps that no stable real-space
  invariant tells apart and whose two members are both used among the
  localizations at a position that has several.

- **`composite`** (`tuple of groups.Term or None`): If not `None`, the row's site irreps at its own position, which
  induce a band representation that is not elementary but a sum of
  elementary ones at other positions.  It is set where no sum of
  elementary band representations at any one position the orbitals can
  slide onto carries the row's symmetry data vector.  The row's stable
  real-space invariants are read from these site irreps.

- **`agree`** (`bool`): Whether every localization gives the same stable real-space
  invariants, which holds in every record returned.

- **`symmetry_data`** (`dict`): The symmetry data vector of the row's own orbitals,
  `{corep: multiplicity}`, measured at `kpoints`.

- **`kpoints`** (`tuple of str`): The high-symmetry points at which `symmetry_data` was measured.
  A point whose little group has no unitary operation besides the
  identity is left out.

- **`n_free`** (`int`): The number of free coordinates of the row's Wyckoff position.

The fields `positions`, `residual`, `characters`, `named_by`, `declared_irrep`, `declared_in`, `reduced_in`, `declaration_broke`, `determined_by`, `alternatives`, `contents`, `n_representations` are those of [`WyckoffOrbitNumerical`](#wyckofforbitnumerical), with the entries there.

###### `Localization`

One decomposition of a Wyckoff orbit's symmetry data vector at one position.

The decomposition is a sum of elementary band representations at one
Wyckoff position, each written as the site irrep inducing it.  Each row
[`srsi_of_model`](functions.md#srsi_of_model) returns holds its decompositions in
`numerical.localizations`, and they all have the same stable real-space
invariants.  A localization is a sum of terms, as
[`BandRepresentation`](#bandrepresentation) is.  Iterating, indexing and
`len()` go over `terms`.  Two sums with the same terms compare
equal.

**Held in** [`WyckoffOrbitSRSIsNumerical.localizations`](#wyckofforbitsrsisnumerical), [`WyckoffOrbitSRSIsNumerical.chosen`](#wyckofforbitsrsisnumerical).

**Exported by** `pythtqc.srsi`.

**Attributes**

- **`terms`** (`tuple of groups.Term`): If nonempty, the elementary band representations, each a site irrep
  at `wyckoff` with its multiplicity.  Empty where the row's
  stable real-space invariants are read from its own site irreps,
  `numerical.composite`, which induce a band representation that is
  not elementary.

- **`label`** (`str`): The sum on one line, `"1 (-E1g)_a + 1 (-E1u)_a"`.  Each term is
  written `m (irrep)_w`, with `m` the multiplicity and `w` the
  Wyckoff letter.

- **`wyckoff`** (`str`): The Wyckoff position of the terms, `"3a"`.

- **`theta_Z`** (`dict`): The integer-valued stable real-space invariants of the sum,
  `{name: value}`.

- **`theta_Zn`** (`dict`): If nonempty, the stable real-space invariants valued modulo n of the
  sum, `{name: ModularValue}`.  Empty where the group has none.

### `Splitting`

The no-go test of a gapped splitting of bands.

The record [`splitting`](functions.md#splitting) and [`nsit_criterion`](functions.md#nsit_criterion) return.  It says
whether the splitting is non-symmetry-indicated topological (NSIT).  The
valence bands are one set of the splitting and the conduction bands the
other.  `summary()` prints whether the splitting is NSIT and why, then
the atomic realizations of each side.  The record is truthy even where
`nsit` is `False`.

**Returned by** [`splitting`](functions.md#splitting), [`nsit_criterion`](functions.md#nsit_criterion).

**Exported by** `pythtqc`, `pythtqc.srsi`.

**Attributes**

- **`nsit`** (`bool`): Whether the splitting is NSIT.  It is `True` exactly where both
  sides have atomic realizations, the group has stable real-space
  invariants (SRSIs) valued modulo n, and no pair of a valence and a
  conduction realization has SRSIs valued modulo n adding up to those
  of all the bands.  The two sides of such a splitting cannot both be
  atomic insulators.

- **`valence`** (`Realizations`): Every atomic realization of the valence bands, as
  [`atomic_realizations`](functions.md#atomic_realizations) returns them.  Its list of realizations
  is empty where the valence bands' integer-valued SRSIs are
  fractional, and where no atomic insulator has them.

- **`conduction`** (`Realizations`): Every atomic realization of the conduction bands, as
  [`atomic_realizations`](functions.md#atomic_realizations) returns them.  Its list of realizations
  is empty where the conduction bands' integer-valued SRSIs are
  fractional, and where no atomic insulator has them.

- **`theta_Z`** (`dict`): The integer-valued SRSIs of all the bands, `{name: value}`: the
  valence bands' plus the conduction bands'.

- **`theta_Zn`** (`dict`): If nonempty, the SRSIs valued modulo n of all the bands,
  `{name: ModularValue}`.  Empty where the group has none.

- **`consistent_pairs`** (`tuple`): If nonempty, each pair `(valence, conduction)` of atomic
  realizations whose SRSIs valued modulo n add up to those of all the
  bands, modulo each n.

- **`stably_equivalent`** (`tuple of StablePair`): If nonempty, the pairs of site irreps that no SRSI tells apart and of
  which some realization of either side holds at least one member.

- **`group`** (`GroupData`): The group of the tables, `tables.group`.

**Methods**

- **`summary()`**: Whether the splitting is NSIT and why, then each side, a line each.

#### `Realizations`

Every atomic realization of given integer-valued stable real-space invariants.

The record [`atomic_realizations`](functions.md#atomic_realizations) returns, and each side of a
[`Splitting`](#splitting).  `summary()` prints the integer-valued stable
real-space invariants (ZSRSIs), then every atomic realization with its
stable real-space invariants valued modulo n.  The list of realizations
is complete.  Where it is empty and the ZSRSIs are integers, no atomic
insulator has these ZSRSIs.  The symmetry data vector of a band set with
them is then an integer combination of elementary band representations
and no nonnegative one, as a fragile set's is.  The record is truthy even
where the list is empty.

**Returned by** [`atomic_realizations`](functions.md#atomic_realizations).

**Held in** [`Splitting.valence`](#splitting), [`Splitting.conduction`](#splitting).

**Exported by** `pythtqc`, `pythtqc.srsi`.

**Attributes**

- **`realizations`** (`tuple of AtomicRealization`): If nonempty, every atomic realization of `theta_Z`.  Empty where
  `theta_Z` is fractional, and where no atomic insulator has it.

- **`theta_Z`** (`dict`): The ZSRSIs the realizations have, `{name: value}`, as exact
  fractions.

- **`group`** (`GroupData`): The group of the tables, `tables.group`.

- **`numerical`** (`RealizationsNumerical`): The bound and the cap of the enumeration.

**Methods**

- **`summary()`**: The ZSRSIs, then every realization, a line each.

##### `AtomicRealization`

One atomic realization: site irreps at maximal Wyckoff positions.

An atomic realization is a set of site irreps at maximal Wyckoff
positions, with nonnegative integer multiplicities, whose integer-valued
stable real-space invariants are those asked for.  Each entry of
[`Realizations.realizations`](#realizations) is one.  A realization is a sum of
terms, as [`BandRepresentation`](#bandrepresentation) is.  Iterating, indexing
and `len()` go over `terms`.  Two sums with the same terms compare
equal, whichever call returned them, [`decompose`](functions.md#decompose) among them.

**Held in** [`Realizations.realizations`](#realizations).

**Exported by** `pythtqc.srsi`.

**Attributes**

- **`terms`** (`tuple of groups.Term`): The site irreps, each at its maximal Wyckoff position with its
  multiplicity.

- **`label`** (`str`): The sum on one line, `"1 (-E2)_b + 1 (-E)_c"`.  Each term is
  written `m (irrep)_w`, with `m` the multiplicity and `w` the
  Wyckoff letter.

- **`theta_Zn`** (`dict`): If nonempty, the stable real-space invariants valued modulo n of the
  realization, `{name: ModularValue}`.  Empty where the group has
  none.  Two realizations of one symmetry data vector can differ in
  them.

- **`stably_equivalent`** (`tuple of StablePair`): If nonempty, the pairs of site irreps that no stable real-space
  invariant tells apart and of which the realization holds at least one
  member.

##### `RealizationsNumerical`

The bound and the cap of an enumeration of atomic realizations.

Held in [`Realizations.numerical`](#realizations).

**Held in** [`Realizations.numerical`](#realizations).

**Exported by** `pythtqc.srsi`.

**Attributes**

- **`bound`** (`int`): The largest multiplicity of one site irrep searched: the most that the
  band count allows, so that no realization is missed.  `0` where
  nothing was enumerated.

- **`cap`** (`int`): The `cap=` of [`atomic_realizations`](functions.md#atomic_realizations), the largest number of
  realizations listed.  The call refuses where there are more.

### `StablePair`

Two site irreps at one position with the same stable real-space invariants.

The position is maximal, and the elementary band representations the two
induce have the same symmetry data vector.  Atomic limits that differ
only by one of the two for the other are stably equivalent.  The paper's
example is `B2` and `B3` at `12d` of `I432`.  An entry of
`tables.stably_equivalent`; `summary` prints it.

**Held in** [`SRSITables.stably_equivalent`](#srsitables), [`Splitting.stably_equivalent`](#splitting), [`WyckoffOrbitSRSIsNumerical.stably_equivalent`](#wyckofforbitsrsisnumerical), [`AtomicRealization.stably_equivalent`](#atomicrealization).

**Exported by** `pythtqc`, `pythtqc.srsi`.

**Attributes**

- **`irreps`** (`tuple of str`): The two site irreps, as the irreptables package's list of elementary
  band representations spells them.

- **`wyckoff`** (`str`): The Wyckoff position, `"12d"`.

- **`symmetry_data`** (`dict`): `{corep: multiplicity}`: the symmetry data vector both induce.

- **`theta_Z`** (`dict`): `{name: value}`: the integer-valued invariants both carry.

- **`theta_Zn`** (`dict`): `{name: ModularValue}`: the invariants valued modulo n both carry.

**Methods**

- **`summary()`**: The pair, then the symmetry data vector and invariants both share.

## Refusals

What a call returns in place of its record where it cannot complete, the
exception that carries it, the error raised where the package disagrees with
itself, and the one warning the package issues.

- [`Refusal`](#refusal): What a call returns in place of its record where it cannot complete.
- [`RefusalError`](#refusalerror): A [`Refusal`](#refusal) raised as an exception, by [`Refusal.raise_`](#refusal).
- [`InternalError`](#internalerror): The error raised where the package disagrees with itself.
- [`MagneticScopeWarning`](#magneticscopewarning): The warning issued where time reversal is measured broken.

### `Refusal`

What a call returns in place of its record where it cannot complete.

A refusal names the check that stopped the call and what the check
measured there.  It is returned, never raised.  Printed, a refusal shows
the check, the reason, the number measured and its threshold, and the
keys of `context`.  A refusal is falsy, and so is the empty symmetry
data vector that [`subtract`](functions.md#subtract) returns for two equal vectors.
`isinstance(out, Refusal)` tells a refusal from a returned value.

**Returned by** a call that cannot complete, in place of its record.

**Held in** [`RefusalError.refusal`](#refusalerror).

**Exported by** `pythtqc`, `pythtqc.refusal`.

**Attributes**

- **`gate`** (`str`): The name of the check that stopped the call, such as
  `"multiplet_split"`.  The name does not change between versions.
  Each call lists the checks it can stop at under *Refusals*, and the
  documentation's page Refusals has an entry under each name.

- **`reason`** (`str`): What the check found, in words, naming the orbital, operation, point
  or group concerned.

- **`value`** (`float or None`): If not `None`, the number the check measured.

- **`threshold`** (`float or None`): If not `None`, the number the check compared `value` against.

- **`context`** (`dict`): If nonempty, the data needed to act on the refusal without running
  the call again, such as indices, labels and multiplicities.  A
  refusal that carries the group holds it as `context["group"]`.  One
  that carries only its number holds it as `context["number_str"]`,
  some with its symbol as `context["name"]`.

**Methods**

- **`raise_()`**: This refusal, raised as a [`RefusalError`](#refusalerror).

### `RefusalError`

A [`Refusal`](#refusal) raised as an exception, by [`Refusal.raise_`](#refusal).

A `RuntimeError` whose message is the printed refusal.

**Raised by** [`Refusal.raise_`](#refusal).

**Exported by** `pythtqc`, `pythtqc.refusal`.

**Attributes**

- **`refusal`** (`Refusal`): The refusal raised.

### `InternalError`

The error raised where the package disagrees with itself.

Two measurements of one thing disagree, the tables the package reads
disagree with each other or with a measurement, or a step finds broken
what an earlier step guarantees.  The message says what disagreed.  It is
a defect in the package, not in the model.  Report it on the package's
repository, with the model and the call that raised it.

**Raised by** any call, where the package disagrees with itself.

**Exported by** `pythtqc`, `pythtqc.refusal`.

### `MagneticScopeWarning`

The warning issued where time reversal is measured broken.

[`realized_group`](functions.md#realized_group) issues it.  The group returned is then the
unitary subgroup of the Hamiltonian's magnetic group.  The labels, the
elementary band representations and the symmetry indicators that later
calls return are that subgroup's.  The operations of the group of the
cell combined with time reversal that the Hamiltonian obeys are listed on
[`RealizedGroup.beyond_grey`](#realizedgroup).  No operation combined with
time reversal is searched for outside the group of the cell.
`warnings.simplefilter("ignore", pythtqc.MagneticScopeWarning)` silences
it.

**Issued by** [`realized_group`](functions.md#realized_group), where time reversal is measured broken.

**Exported by** `pythtqc`, `pythtqc.refusal`.
