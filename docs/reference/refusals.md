# Refusals

Every refusal a call can return, by the name of its check: why the check stops the call,
its usual cause, and what to do. A refusal is a falsy `Refusal` record, returned in place
of the call's record. It carries the name of the check that stopped the call and a short
reason. It also carries the number the check measured with its threshold, and a `context`
holding what was found. The guide page [What does a refusal mean, and what do I do about
it?](../guide/refusals.md) reads one through. Each call's entry in
[Functions](functions.md) lists the refusals it can return. Each entry below is headed by
its check's name, which a printed refusal shows in brackets, and opens with the calls that
return it. The words the entries use are defined in the [Glossary](glossary.md).

The entries are grouped by kind, and the kind says where the fix lies:

- [A verdict about the physics](#a-verdict-about-the-physics): what was asked does not
  exist for these bands, or for this model as written. The fix is in the question or in
  the model.
- [A practical limit](#a-practical-limit): a search, an enumeration, a measurement or a
  projection would need more than its ceiling allows, or the installation lacks the tables
  a question needs. Nothing in the model is at fault. Where the ceiling is an argument of
  the call, the reason names the argument to raise. Where something is missing from the
  installation, the reason names it.
- [A grammar error in a declaration or a question](#a-grammar-error-in-a-declaration-or-a-question):
  a declaration that does not fit the orbitals it is written on, or an argument a call
  cannot read. The reason says what fits.
- [The package's checks on itself](#the-packages-checks-on-itself): an identity that
  holds in exact arithmetic was measured not to, or two measurements of one thing
  disagree. Most are worth reporting, as an issue on
  [pythTQC's GitHub](https://github.com/bbradlyn/pythtqc/issues), with the model and the
  call. The entries say where there is something else to do.

`realized_group` first finds the cell's group: spglib finds the space group of the orbital
centers, each with its species, at its tolerance `symprec`. It then measures which
operations of that group the Hamiltonian obeys, and the group is identified once the
measurement has kept every operation of it. A refusal returned after that opens on the
group identified, "every operation of it obeyed", and carries it in `context["group"]`.
One reached while the operations are still being measured names the group of that round,
the cell's group or a subgroup the measurement has lowered it to, as an upper bound on
what the Hamiltonian obeys, and carries it in `context["group"]`. The entries say where a
refusal names no group, or names one without carrying it, and `cell_group_not_realized`
names the group the projection was made in, with how many of its operations the
Hamiltonian obeys.

Where `H` was projected onto its symmetric part and the orbital centers were not moved,
a refusal `realized_group` returns after the projection, other than
`positions_not_exactly_symmetric`, also quotes the miss `δ` of the centers from the
positions the group puts them at, in reduced coordinates, where it is above rounding.
`context["position_defect"]` holds it. pythtb gives a hopping a Bloch phase that depends
on the orbital centers, so the miss alone costs `H(k)` a covariance residual of order
`2π·δ·max|H|`, whatever the hoppings are. `max|H|` is the largest entry of `H(k)` once
the mean on-site energy is subtracted. No projection of the hoppings removes that
residual, and the refusal may come from the miss alone. A declared `precision=` moves the
centers onto those positions, a gauge transformation that leaves the spectrum unchanged.

A malformed argument raises `ValueError` before anything is measured. A disagreement
inside the package raises `InternalError`. Neither is a refusal.

## A verdict about the physics

### The group, and how the model is written

#### `supercell`

Returned by `realized_group`.

- **What it means.** The Hamiltonian obeys a pure translation of its own cell, either as
  written or after a change of basis within each site that carries its orbitals onto
  those of the translated site. The model is then written on a supercell of its primitive
  cell. The bands at one `k` of the supercell's Brillouin zone are those at several `k` of
  the primitive cell's. No table names such a combination, so every label would name the
  wrong k-points. It is one of the two cases in which the question which group the
  Hamiltonian obeys is refused
  ([When the group question is refused](../background/measurement.md#when-the-group-question-is-refused)).
- **Usual cause.** Copies of one site that the hoppings do not tell apart, written alike
  or in frames turned from site to site, as in pythtb's checkerboard with no on-site
  offset.
- **What to do.** Rebuild the model on its primitive cell and ask again
  ([A supercell is refused, not folded](../background/limits.md#a-supercell-is-refused-not-folded)).
  Where the copies are written in turned frames, write them in one basis before
  rebuilding. `context["primitive_lattice"]` holds the primitive cell's vectors,
  `context["translations"]` the translations obeyed, and `context["index"]` the number of
  primitive cells the cell holds. `context["dressing"]` holds, for each translation, the
  `matrix` that carries each site's orbitals onto those of the translated site and the
  `turn` it makes, `0.0` where the copies are written alike. Where the copies are meant to
  differ, the difference belongs in the Hamiltonian: a supercell whose hoppings tell the
  copies apart is a primitive model of the bigger cell, and the group descends to it
  ([When the cell is a supercell](../guide/group.md#when-the-cell-is-a-supercell)).

#### `positions_not_exactly_symmetric`

Returned by `realized_group`.

- **What it means.** The orbital centers miss the positions the group's operations put
  them at by `δ`, in reduced coordinates, and the miss alone moves `H(k)` by more than a
  check of its symmetry allows. Which operations the Hamiltonian obeys would then be read
  off the geometry as much as off the hoppings. It is returned only where no tolerance for
  the centers is declared, in four cases.
    - Once spglib has found the cell's group, before any operation is measured, where
      `2π·δ·max|H|` is above `tol`, the tolerance every operation's covariance residual
      is held to. It is the other case in which the group question is refused, and the
      reason names the cell's group as an upper bound.
    - While spglib finds the cell's group, where the operations it reads off the centers
      do not close into a group and the miss of the centers accounts for it. This is met
      on a cell that cannot be matched to the conventional cell used by the irreptables
      package, as a cell whose sites of one species (`typat=`) repeat under a fraction
      of a lattice translation cannot. On such a cell the operations cannot be rebuilt
      from the tables of the irreptables package. The reason names the group spglib
      reports as an upper bound, by `context["name"]` and `context["number_str"]`, and
      the context carries no group.
    - Once the group is identified, where the representation built for it misses its
      covariance tolerance and the miss of the centers accounts for it. The miss alone
      moves `H(k)` by up to `context["reach"]`. The reason names the group identified.
    - Before `H` is projected onto its symmetric part, where the centers miss the
      operations by more than `1e-9` in reduced coordinates. The average over the
      operations would then carry a hopping to a vector that is not a lattice vector,
      and no projection of the hoppings removes a defect in the centers. The reason
      names the group identified.
- **Usual cause.** A numerical model, such as a Wannier90 import, asked with no precision
  declared; or centers converted onto a strongly skewed lattice basis in floating point.
- **What to do.** Declare the model's precision, `precision=`, or `position_tol=` for the
  centers alone. Centers within it are moved onto the positions the group puts them at, a
  k-dependent gauge transformation that leaves the spectrum unchanged
  ([The centers first: moving them is a change of gauge](../guide/numerical.md#the-centers-first-moving-them-is-a-change-of-gauge)).
  `value` and `context["position_defect"]` are `δ`. Centers noisier than `symprec` are
  not refused this way: spglib reads them as a less symmetric set, and the group returned
  can be as low as `P1`
  ([Centers noisier than `symprec`](../guide/numerical.md#centers-noisier-than-symprec)).

#### `cell_group_not_realized`

Returned by `realized_group`.

- **What it means.** An operation combined with time reversal held before a numerical
  model's Hamiltonian was projected onto its symmetric part, and is broken after it. The
  projection was made in a group the projected Hamiltonian does not obey. A
  representation of that group would label the bands with an operation the Hamiltonian
  breaks. The reason names that group as an upper bound on what the Hamiltonian obeys,
  with how many of its operations the Hamiltonian obeys. `context["broken"]` lists the
  operations broken, and `context["validated"]` holds each operation's verdict and
  covariance residual.
- **Usual cause.** A model that obeys an operation combined with time reversal only
  within its declared precision.

#### `wyckoff_orbit`

Returned by `realized_group`.

- **What it means.** An operation of the group carries an orbital to a point where no
  orbital of its species sits, so the orbitals lie on no Wyckoff orbit of the cell and no
  representation of the group acts on them.
- **Usual cause.** A misplaced orbital.
- **What to do.** Check the orbital's position. `context` holds the site and its
  orbitals, the operation, the image of the site and the distance from it to each site.

#### `operations_not_consistent`

Returned by `realized_group` where the centers cannot be read as a symmetric set at all,
as when two sites of one species sit closer together than spglib can tell apart. Where the
operations fail to close only because the centers miss their positions, with no precision
declared, the refusal is
[`positions_not_exactly_symmetric`](#positions_not_exactly_symmetric).

- **What it means.** The operations spglib reads off the centers do not close into a
  group, and the cell is not matched to the conventional cell used by the irreptables
  package, from whose tables exact operations would be rebuilt. Every later check would
  be measured against a set that is not a group. The reason gives how far they are from
  closing: the largest miss of `t_g + R_g t_h - t_gh` from a lattice vector, in reduced
  coordinates, where `R_g` and `t_g` are the rotation and translation of operation `g`.
  It sets that against the most a cell built from exact numbers misses by, `1e-12` times
  the largest row sum of any `|R_g|`. Met where the cell's group is first found, the
  reason names no group; met in a descent, it names the group of that round.
- **Usual cause.** Sites of one species closer together than `symprec`, in a cell not
  matched to that conventional cell. spglib reads each operation's translation off the
  pairs of sites the operation maps. Sites closer than `symprec` are paired wrongly, so
  the translations read off them do not compose.
- **What to do.** Call `realized_group` again with a `symprec=` smaller than the distance
  between the sites, so that spglib tells them apart.

#### `operations_not_nearly_consistent`

Returned by `realized_group`, on a cell matched to the conventional cell used by the
irreptables package, where the operations spglib reads off the centers do not close into
a group.

- **What it means.** Rebuilding the operations' translations from the tables' own would
  move one by more than `symprec`, a length in the model's units. The tabulated group is
  then not the one these centers carry at this tolerance, and the operations rebuilt
  would describe a crystal the model does not have. Met where the cell's group is first
  found, the reason names no group; met in a descent, it names the group of that round.
- **Usual cause.** Centers noisier than `symprec` allows for, or a `symprec` too large for
  the cell, with no precision declared.
- **What to do.** Declare the model's precision, `precision=`, which moves the centers onto
  symmetric positions before the cell is matched to the tables; call `realized_group`
  again with a `symprec=` the centers support; or say what the crystal is with `typat=` or
  `decorations=`.

#### `positions_not_nearly_symmetric`

Returned by `realized_group`.

- **What it means.** Under a tolerance declared for the orbital centers, `precision=` or
  `position_tol=`, the centers cannot be moved onto the positions the group puts them at.
  An operation that would move a center farther than the tolerance is broken by the
  centers, and the group descends to the largest group of its operations that carries
  them. The refusal comes where no such descent helps. Either the group's operations do
  not carry the sites onto one another at spglib's tolerance `symprec`; or orbitals read
  as one site, closer together than `symprec`, sit farther apart than the tolerance, so
  that moving them onto one position moves one of them farther than it. Centers that far
  off are a different set of points.
- **Usual cause.** A group the centers do not have, or a displacement beyond the noise.
- **What to do.** Give each orbital its species with `typat=`, or add the atoms the
  orbital centers do not show with `decorations=`, so that the group identified is one
  the centers carry.

### Spin

#### `rep_double_valued_spinless`

Returned by `realized_group`.

- **What it means.** `spinor=True` was asked of a spinless model with no `spin=` saying
  where the spin is. The double-valued tables of the irreptables package are read with the
  orbitals carrying the spin one-half, which a spinless model's orbitals do only where
  `spin=` pairs them into spin partners. Without it, a double-valued representation would
  say each orbital carries half-integer spin itself.
- **Usual cause.** A spinor Wannier90 import, which pythtb returns spinless with the spin
  inside the orbital index, asked with `spinor=True` and no `spin=`.
- **What to do.** Where the orbitals come in pairs, the up and the down state of one
  spinful orbital, declare the pairs with `spin=`
  ([A Wannier90 spinor import](../guide/spin.md#a-wannier90-spinor-import)). Otherwise
  ask the single-valued group, `spinor=False`, or build the model with spin where the
  half-integer spin is the electron's own. An orbital that carries one spin state with no
  partner is outside the package's scope
  ([An unpaired spinor orbital](../background/limits.md#an-unpaired-spinor-orbital)).

#### `single_valued_spinful`

Returned by `realized_group`.

- **What it means.** `spinor=False` was asked of a spinful model. Its states carry the
  electron's spin one-half, so the little-group representations of its bands are
  double-valued, and single-valued tables do not describe them. A crystal is refused
  before its group is found and names none. A layer, or a crystal given `decorations=`,
  is refused once the cell's group is found, and names that group, carried in
  `context["group"]`. `spinor=True` for a spinless model is refused
  [`rep_double_valued_spinless`](#rep_double_valued_spinless) instead, once the group is
  identified.
- **Usual cause.** `spinor=False` on a model built with spin.
- **What to do.** Leave `spinor` at its default, which reads it off the model.

### What the package covers

#### `fewer_than_two_periodic_directions`

Returned by `realized_group`.

- **What it means.** The model is periodic in fewer than two directions, whatever the
  dimension of the cell it is written in. With none it is a molecule, whose symmetry is a
  point group; with one it is a chain or a ribbon, whose symmetry is a rod or frieze
  group. The package reads crystals periodic in three directions and layers periodic in
  two, whose symmetry is a space group, and a model periodic in fewer is refused. The
  refusal comes before anything is measured, so it names no group; `value` is the number
  of periodic directions and `context["periodic_dirs"]` says which they are.
- **Usual cause.** pythtb's `ssh` chain; a ribbon or a rod cut from a crystal with
  `cut_piece`; a cluster with every periodic direction switched off.
- **What to do.** There is no step to take in pythTQC
  ([Fewer than two periodic directions](../background/limits.md#fewer-than-two-periodic-directions)).

#### `srsi_scope`

Returned by `srsi_tables`, `srsi_of_bands`, `srsi_of_model`, `splitting` and
`pythtqc.srsi.canonical_determinism`.

- **What it means.** The group is not grey, or the number handed to `srsi_tables` is
  outside 1 to 230. The stable real-space invariants are tabulated for the grey groups
  alone, each space group with time reversal adjoined: 217 of the 230, and the other
  thirteen are refused [`srsi_table_absent`](#srsi_table_absent).
- **Usual cause.** A Hamiltonian that breaks time reversal, for which `realized_group`
  returns a space group without time reversal; or a number outside 1 to 230.
- **What to do.** No table of stable real-space invariants exists for such a group. The
  decomposition into elementary band representations and the symmetry indicator still
  apply.

#### `srsi_table_absent`

Returned by the same calls as `srsi_scope`.

- **What it means.** The group is one of the thirteen the converted tables of stable
  real-space invariants omit, numbers 1, 4, 7, 9, 19, 29, 33, 76, 78, 144, 145, 169 and
  170, whose only stable real-space invariant is the band count.
- **Usual cause.** A model in one of those groups.
- **What to do.** The band count is the invariant, and there is nothing else to compute.

### Labels at a k-point

#### `multiplet_split`

Returned by `characters`, `coreps_at` and `sewing_matrix`; `hsp_table` keeps it for each
point it refuses at.

- **What it means.** The set of bands asked for cuts a degenerate multiplet at `k`, so its
  sewing matrices and characters would depend on the basis the diagonalization chose
  inside the multiplet.
- **Usual cause.** A band count that ends inside a degeneracy: half a Kramers pair, or bands
  that meet at the point, as a semimetal's do.
- **What to do.** Ask for every band of the multiplet, or for a set of bands separated by
  a gap from the others. `context["multiplet"]` lists the bands that are degenerate there
  ([Label an isolated set of bands](../guide/labels.md#label-an-isolated-set-of-bands)).

#### `integer_multiplicity`

Returned by `coreps_at` and `pythtqc.groups.identify_from_characters`.

- **What it means.** The multiplicities of the little-group representations fitted to the
  characters of a multiplet at a point are not nonnegative integers within `label_tol`,
  `1e-3` by default. Characters that are not a sum of the tables' little-group
  representations carry no label.
- **Usual cause.** A degeneracy split into several multiplets by numerical noise beyond
  `degeneracy_tol`: a two-dimensional little-group representation split in two reads
  exactly 0.5. Levels merged into one multiplet never refuse; they are labeled as a sum of
  little-group representations. From `identify_from_characters`, also characters
  measured in a convention other than the one `characters` measures in.
- **What to do.** On a numerical model, declare its precision to `realized_group`
  (`precision=`), which projects `H` onto its symmetric part, or set `degeneracy_tol`, an
  energy in the units of `H`, to the noise its degeneracies carry. `context["rows"]` names
  the multiplets at fault and `context["multiplicities"]` holds the fitted numbers.

#### `multiplet_dimension`

Returned by `coreps_at`, through its identification of the little-group representations,
and by `pythtqc.groups.identify_from_characters` given `multiplet_dims=`.

- **What it means.** The dimensions of the little-group representations identified from a
  multiplet's characters do not add up to its degeneracy, so the grouping of levels and
  the characters disagree.
- **Usual cause.** Characters that are no representation's, whose fitted multiplicities
  still round to integers. Merged levels never refuse it, and a degeneracy split in two is
  refused `integer_multiplicity` first.
- **What to do.** As for `integer_multiplicity`. On a model built from exact numbers with
  well separated levels, report it.

### Atomic limits and indicators

#### `incomplete_content`

Returned by `decompose`, `symmetry_indicator`, `subtract`, `regular_completion`,
`splitting` and `srsi_of_bands`, each when it is handed a table that refused at some
high-symmetry point; and by `pythtqc.groups.decompose_ebrs`.

- **What it means.** The table holds no little-group representations at that point. A
  decomposition, a class or a difference of the multiplicities it does hold would be a
  verdict on a different symmetry data vector. Leaving a point out removes conditions, so
  a decomposition of the rest is easier to find than one of the whole vector. From
  `decompose_ebrs`, the symmetry data vector handed in names no little-group
  representation at some point where the elementary band representations of the group
  have little-group representations, and would be read as zero there.
- **Usual cause.** A set of bands that cuts a multiplet at that point, or another refusal
  recorded in the table there. From `decompose_ebrs`, also a layer's vector, which holds
  no point out of its plane.
- **What to do.** Read the table's refusals, which `context["refusals"]` repeats, and mend
  the set of bands. Where the tables of the irreptables package cannot be read at a
  point, `decompose` takes `allow_incomplete=True` and decomposes the multiplicities at
  the other points; no other call takes it. The record then says whether its sums are
  exact. They are exact where, for every little-group representation left out, one kept
  occurs as often as it in every elementary band representation. Otherwise a sum found
  matches the points kept and need not match the whole vector. From `decompose_ebrs`,
  pass multiplicities at every point, or for a layer use `decompose`, which reads the
  vector on the points of the plane.

#### `not_a_band_structure`

Returned by `decompose`, `symmetry_indicator`, `regular_completion`, `splitting` and
`srsi_of_bands`.

- **What it means.** The symmetry data vector violates the group's compatibility
  relations: written in the basis of the Smith normal form of the group's matrix of
  elementary band representations, it has a nonzero coordinate where every band structure
  of the group has zero. So it lies outside the lattice of band structures, with no class
  in the indicator group `{BS}/{AI}` and no decomposition
  ([How a symmetry indicator is computed](../background/tqc-primer.md#how-a-symmetry-indicator-is-computed)).
- **Usual cause.** Bands asked for that are not separated by a gap from the others at
  every `k`, as when they cut through a crossing between two high-symmetry points; or a
  symmetry data vector read in another group.
- **What to do.** Check that the set of bands is separated by a gap from the others
  everywhere and that its labels came from this group. `context["violations"]` lists the
  coordinates that do not vanish, each as `(index, value)`, the index that of the
  coordinate in `context["y_prime"]`, the vector in the Smith normal form's basis.

#### `indicator_not_evaluable`

Returned by `symmetry_indicator`, `srsi_of_bands` and `splitting`.

- **What it means.** For a layer, the multiplicities at the points of its plane do not fix
  those of the little-group representations out of it that the refusal names: for each of
  those, no little-group representation in the plane occurs as often as it in every
  elementary band representation. For a table they are measured on the layer's decoupled
  stack, the crystal that has those points, and that measurement stopped. The class is
  read from the whole symmetry data vector, so without them it has no value.
- **Usual cause.** A layer's symmetry data vector written out by hand, whose
  multiplicities out of the plane the layer's elementary band representations do not
  fix; or a check of the stack that did not complete, which `context["stack"]` names.
- **What to do.** For a vector written out by hand, pass the table `hsp_table` returns for
  the same bands, whose little-group representations out of the plane are measured on the
  layer's decoupled stack. For a table, `context["stack"]` holds the measurement on the
  stack up to where it stopped, with the number measured by each check made before then
  ([A layer's class is a three-dimensional class](../guide/indicators.md#a-layers-class-is-a-three-dimensional-class)).

#### `restriction_not_exact`

Returned by `in_plane_restriction`, `decompose` and `regular_completion`. Each returns it
where the measured multiplicities do not fix that of some little-group representation at
a point not measured: no measured little-group representation occurs as often as it in
every elementary band representation of the group.

- **What it means.** The multiplicity of each such little-group representation is a
  condition of its own on a symmetry data vector, which the measured multiplicities do not
  fix, so no verdict read from them is exact for this group.
- **Usual cause.** `in_plane_restriction` handed `kpoints=` naming fewer points than the
  plane holds, or asked of a layer's group with every elementary band representation of
  the group, some of whose little-group representations at `k_z = 1/2` are of this kind.
  From `decompose` and `regular_completion`, the Wyckoff positions of the tables could not
  be read in the group's cell, so a layer was read over every elementary band
  representation of its group rather than those of the Wyckoff positions within it.
- **What to do.** For `in_plane_restriction`, name every point of the plane in `kpoints=`,
  or leave `kpoints=` out, which is enough for a crystal. A layer's little-group
  representations at `k_z = 1/2` are measured on its decoupled stack:
  `symmetry_indicator`, handed the table, reads the class there
  ([Which symmetry-indicator class do these bands occupy?](../guide/indicators.md)). From
  `decompose` and `regular_completion`, report it, with the model and the call.

### Stable real-space invariants

#### `wyckoff_orbit_content_incomplete`

Returned by `srsi_of_model` and `splitting`.

- **What it means.** The symmetry data vector of a Wyckoff orbit's own orbitals could not
  be read at a high-symmetry point.
- **Usual cause.** A refusal at that point, which `context["refusal"]` carries.
- **What to do.** Read that refusal; its entry here says what to do.

#### `wyckoff_orbit_not_localizable`

Returned by `srsi_of_model` and `splitting`.

- **What it means.** No sum of elementary band representations at one Wyckoff position on
  the closure of a Wyckoff orbit's own position carries the orbit's measured symmetry data
  vector. The closure holds the points the position's free coordinates sweep and the
  higher-symmetry positions among them. Nor could the orbit's invariants be read from its
  site irreps, the way a composite site irrep's are read, a composite site irrep being one
  whose band representation is a sum of elementary ones at other positions. For a
  spinless model one of its site irreps is not among those the tables list at its
  position, and for a spinful model the double-valued tables hold no composite site
  irrep. The orbit's symmetry data vector is then consistent with no atomic limit at the
  position it sits on, which is a statement about the model.
- **Usual cause.** Orbitals whose symmetry data vector does not fit where they sit.
- **What to do.** Check the orbitals' positions and their site irreps; `context["tried"]`
  lists the positions tried.

### A layer and its stack

#### `not_a_layer`

Returned by `stack_presentation`.

- **What it means.** The decoupled stack is built from a layer, two periodic directions in a
  two- or three-dimensional cell, and this model is not one.
- **Usual cause.** A three-dimensional crystal or a chain handed to `stack_presentation`.
- **What to do.** Hand it a layer's model.

#### `hopping_across_the_stack`

Returned by `stack_presentation`.

- **What it means.** Hoppings run along the non-periodic direction, so the model is not one
  decoupled layer: stacking it would either drop those hoppings or give `H(k)` a
  dependence on `k_z` that the layer does not have.
- **Usual cause.** A hopping table written directly, past pythtb's `set_hop`, which
  refuses a hopping with a component along a non-periodic direction.
- **What to do.** Check the model's hoppings along its non-periodic direction.

## A practical limit

### Searches and memory

#### `rep_search_too_large`

Returned by `realized_group`.

- **What it means.** The measurement leaves some discrete choices of the representation
  open, and enumerating them would run more than `max_candidates` candidates. Where the
  Hamiltonian conserves a pseudospin of n copies, it does not fix the representation the
  copies carry. The copies fall into sets that the hoppings do not couple and the
  operations carry onto one another. The candidates are then the sums of total dimension
  n of the irreducible representations of the unitary operations that map one set onto
  itself, projective with the factor system the hoppings leave the copies. Where the
  Hamiltonian splits a multiplet into sets of states it does not couple and no operation
  exchanges, it does not fix the representation of each set either. The candidates are
  then every way of choosing one for each set. A declaration on orbitals whose states come
  in identical copies is compared at once with every representation the hoppings leave
  open on the copies, and that comparison has the same limit.
- **Usual cause.** A conserved operator, or an accidental degeneracy, inside a multiplet
  of several orbitals with nothing declared, so the multiplet's block is fixed only up to
  the conserved part; or a declaration on orbitals that carry such a conserved operator.
- **What to do.** Declaring what the orbitals are with `characters=` states what the
  search would otherwise choose among. A larger `max_candidates=` runs a longer search.
  Where the reason names a declaration being compared, declaring fewer of the orbitals it
  names leaves fewer representations to compare. Where the reason counts phases on orbits
  of multiplets, stating a site of several orbitals as one multiplet with
  `site_multiplets=` cuts the count. `context["fallback_reason"]`, where present, says why
  the choices could not be read off the measurement. Where it says a generator's block
  does not have the form of a representation of the operation, a larger search does not
  help.

#### `detector_too_large`

Returned by `realized_group`.

- **What it means.** Measuring an operation at one k-point needs more memory than
  `max_bytes` allows. The operation's matrices are first sought there among pairs of
  eigenvectors, one of `H(k)` and one of `H(Rk)`, `Rk` the operation's image of `k`,
  whose energies are equal within rounding or the measurement's tolerance. The memory
  needed is one matrix of the size of `H` for each such pair, and one more. Narrowing the
  measurement to fit would drop some of those pairs and return a wrong solution rather
  than a coarse one. The refusal comes while the operations are measured, so the group it
  names is the group of that round, the cell's group or one the measurement lowered it
  to, an upper bound on the group `H` obeys.
- **Usual cause.** A large model with exactly degenerate levels, such as a big
  supercell's folded bands. A model whose degeneracies are lifted does not reach it.
- **What to do.** Run the same call with the `max_bytes=` the reason names.

#### `projection_too_large`

Returned by `realized_group`.

- **What it means.** The projection of `H` onto its symmetric part needs more memory at
  its peak than `max_bytes` allows. It holds `H(k)` and its group average together on a
  grid of k-points. Along each periodic direction the grid has `2·(R + 1) + 1`
  divisions, `R` the number of cells the symmetrized hoppings can reach along it, taken as
  the largest over the directions the group's operations carry onto one another. A
  coarser grid would alias the transform back to a hopping list, so the projection is
  not run on one.
- **Usual cause.** A numerical model with many states and long-ranged hoppings.
- **What to do.** Run the same call with the `max_bytes=` the reason names
  ([A large model](../guide/numerical.md#a-large-model)).

#### `srsi_realizations_truncated`

Returned by `atomic_realizations`, and by `nsit_criterion` and `splitting` through it.

- **What it means.** The enumeration of the atomic realizations of the integer-valued
  stable real-space invariants `theta_Z` found more than its cap. That no atomic
  realization exists, or that no pair of them adds up, holds only for a complete list. The
  refusal's `value` is `cap + 1`, the fewest realizations the search showed, and its
  `threshold` is the cap.
- **Usual cause.** Many bands, or a group with many band representations.
- **What to do.** Only `atomic_realizations` takes `cap=`, `256` by default;
  `nsit_criterion` and `splitting` enumerate at that default. Call `atomic_realizations`
  with a larger `cap=` on the group's tables and the `theta_Z` that `context["theta_Z"]`
  carries. From `nsit_criterion` and `splitting`, that is the `theta_Z` of the valence or
  of the conduction bands, the side of the splitting whose enumeration was cut short.

### The tables, as installed

#### `irrep_tables`

Returned by `realized_group`, `ebr_data`, `ebr_rows`, `decompose`, `symmetry_indicator`,
`regular_completion`, `in_plane_restriction`, `srsi_tables`, `srsi_of_bands`,
`srsi_of_model`, `splitting`, `wyckoff_sites_tables`, `pythtqc.groups.decompose_ebrs`,
`pythtqc.groups.indicator_group` and `pythtqc.srsi.canonical_determinism`.

- **What it means.** The irrep package, which reads the character tables the irreptables
  package ships, has no tables loaded, so no little-group representation or site irrep
  can be named; or the tables hold none for this group, so its band representations
  cannot be carried into this package's convention. From `realized_group`, the refusal
  comes before the cell's group is found, and names no group.
- **Usual cause.** `irreptables` not installed, or its module `irreptables.irreps`
  imported before `irrep` in the process, which leaves `irrep` with no character tables
  for the rest of the process.
- **What to do.** `tables_status()` says which. Install `irreptables`, or, in a new
  process, import `pythtqc` before `irreptables`.

#### `tables_not_ready`

Returned by `operations`, `operation_index`, `high_symmetry_points`, `little_group`,
`corep_table`, `characters`, `sewing_matrix`, `coreps_at`, `hsp_table`,
`wyckoff_sites_tables`, `in_plane_restriction`, `decompose`, `symmetry_indicator`,
`regular_content`, `regular_completion`, `srsi_of_bands`, `srsi_of_model`, `splitting`,
`pythtqc.groups.identify_from_characters`, `LittleGroup.operations` and the methods of
`SpaceGroupRep` that take a k-point.

- **What it means.** The group was identified, but irrep could not match its cell to the
  conventional cell used by the irreptables package, so nothing can be read from the
  tables for it. The reason carries `irrep`'s own message.
- **What to do.** Report the model, with the reason.

#### `ebr_tables`

Returned by the same calls as `irrep_tables`, except `realized_group` and
`pythtqc.groups.indicator_group`.

- **What it means.** The installed tables hold no elementary band representations for this
  group with single-valued or double-valued site irreps, whichever the model needs.
- **Usual cause.** A group, or its single- or double-valued site irreps, that the tables
  do not cover.
- **What to do.** None in the model. The group and the little-group representations stand
  without them.

#### `ebr_convention`

Returned by the same calls as `ebr_tables`.

- **What it means.** The little-group representations of this group's band
  representations cannot be carried from the tables' phase convention, `e^(+ik·t)`, into
  this package's, `e^(-ik·t)`
  ([Conventions: operations, labels and band representations](../background/conventions.md)).
  Carrying them sends each label to its complex conjugate. That must keep the
  multiplicities at Γ and each band representation's number of bands at every point. It
  must also send each label to a label at the same point, whose own conjugate is the
  first. Where any of these fails, none of the group's band representations is carried.
- **Usual cause.** A defect in the tabulation itself: an entry without its name or its
  Wyckoff position, or characters at a point that are not closed under complex
  conjugation where neither the factor system nor the time-reversal gluing is complex.
- **What to do.** None in the model. Report it.

#### `ortools`

Returned by `pythtqc.groups.decompose_ebrs`.

- **What it means.** The symmetry data vector is an integer combination of elementary band
  representations. Whether a combination with no negative coefficient exists, which tells
  a vector consistent with an atomic limit from a fragile one, is decided by an integer
  search in `ortools`, which is not importable.
- **Usual cause.** An installation without `ortools`, a dependency of the package.
- **What to do.** Install `ortools` 9.10 or later.

#### `srsi_data`

Returned by the same calls as [`srsi_scope`](#srsi_scope).

- **What it means.** The converted tables of stable real-space invariants, two data files
  the package ships, are not where it reads them.
- **Usual cause.** An installation whose package data is missing a file.
- **What to do.** Reinstall the package. `pythtqc.srsi.srsi_status()` says which file is
  missing and where it belongs.

#### `srsi_ebr_basis`

Returned by the same calls as `srsi_data`.

- **What it means.** The installed `irreptables` lists this group's elementary band
  representations differently from the version the tables of stable real-space invariants
  were converted with, so they would name the wrong elementary band representations.
- **Usual cause.** An `irreptables` installed after the tables were converted.
- **What to do.** Install the version of `irreptables` the reason names.

## A grammar error in a declaration or a question

### In `characters=` and `spin=`

#### `unknown_label`

Returned by `realized_group`, `decompose`, `symmetry_indicator`, `srsi_of_bands` and
`regular_completion`.

- **What it means.** From `realized_group`, a declaration names nothing at the orbital's
  site, so it says nothing the representation can be built on. Such a declaration is a
  word that is none of the forms `characters=` takes
  ([The forms a declaration takes](../guide/orbitals.md#the-forms-a-declaration-takes)),
  or a key naming an operation the site group does not hold. On a spinful site, a sum of
  double-valued site irreps is refused where no sum of single-valued site irreps becomes
  it with spin. A shorthand is refused where its function and its images under the site
  group span more than one site irrep, counted with multiplicity. Generator characters
  are refused where a key matches operations of more than one conjugacy class. Generator
  matrices are refused where a key names more than one operation, where a matrix is not
  unitary, or where the matrices do not multiply as their operations do. A declaration
  outside the grammar of `characters=`, such as an orbital out of range or a list of the
  wrong length, raises `ValueError` before anything is measured. From `decompose`,
  `symmetry_indicator` and `srsi_of_bands`, the symmetry data vector names little-group
  representations that no elementary band representation of the group holds. From
  `regular_completion`, one copy of the regular band representation, computed from the
  group's characters, names a little-group representation that no elementary band
  representation holds.
- **Usual cause.** A word of another site group, such as `B1` at a site of symmetry `3m`;
  for generator characters or matrices, a name such as `m`, which at a `4mm` site
  matches four mirrors in two conjugacy classes; for a symmetry data vector, one read in
  another group, or a mistyped label.
- **What to do.** For a declaration, the reason says what to declare, in the frame it
  prints. A word the cell's group has no name for is read again in each group of the
  descent, and refused only where none of them names it
  ([When it refuses](../guide/orbitals.md#when-it-refuses), on How do I say what my
  orbitals are?). For a symmetry data vector, pass one read in this group. Met from
  `regular_completion`, it is the package disagreeing with itself. Report it, with the
  model and the call.

#### `ambiguous_multiplet`

Returned by `realized_group`.

- **What it means.** A declaration does not say which orbitals hold a site irrep, or
  gives generator matrices that do not stand on whole multiplets the hoppings measure. A
  key of several orbitals declares the site irreps those orbitals hold together, so their
  dimensions add up to the number of orbitals in the key. Orbitals of one site declared
  with a site irrep of dimension above one number a multiple of its dimension; where they
  do not, the declaration does not say which orbitals hold it. Generator matrices fix the
  basis of the site irrep a measured multiplet carries. They are refused on part of a
  multiplet, mixing two multiplets, at a site whose declarations do not each name one
  whole multiplet, or at a site of several multiplets where one declaration is a
  double-valued site irrep. Orbitals of two sites written as one key raise `ValueError`
  before anything is measured.
- **Usual cause.** `{(0, 1): "A"}` written for two `A` orbitals, whose spelling is
  `{0: "A", 1: "A"}`; a two-dimensional site irrep declared on one orbital.
- **What to do.** The reason gives the spelling that says it, or, where the site holds too
  few orbitals for the site irrep, the words that fit it.

#### `spin_pair_split`

Returned by `realized_group`.

- **What it means.** The two orbitals of a declared spin pair sit at different centers,
  where the up and the down state of one spinful orbital share one. Either the pairing is
  mislabeled, or the orbitals each carry one spin state with no partner at their own
  center, a magnetic configuration outside the package's scope
  ([An unpaired spinor orbital](../background/limits.md#an-unpaired-spinor-orbital)).
  Two centers apart only by the numerical noise of a fitted model are neither: a declared
  precision (`precision=`) reads two centers closer than it as one. The pairs are those
  `spin=` declares, and `realized_group` reads whether each pair shares a site off the
  orbitals' centers alone, the orbitals at one center being one site at the tolerance the
  cell's group is found at. It refuses before anything is identified or measured, so it
  names no group. `value` is the distance between the pair's two sites, and
  `context["positions"]` says where they are.
- **Usual cause.** The spin states declared in the wrong order: `spin="blocked"` and
  `spin="interleaved"` differ by exactly this.
- **What to do.** Read the order of the spin states off the Wannier90 files and declare
  it, with explicit pairs where the orbitals are ordered irregularly. Where the two
  centers differ by a fitted model's noise, declare its precision.

### In the arguments of a call

#### `unknown_operation`

Returned by `operation_index`.

- **What it means.** The symbol is read in the conventional cell used by the irreptables
  package. Either it is not a Seitz symbol, or no operation of the group has its rotation
  part, axis, sense, translation modulo the lattice and time reversal.
- **Usual cause.** An operation's Cartesian name, such as `C4(z)`, in place of its Seitz
  symbol; a symbol written in the model's own cell; or an operation the group does not
  hold.
- **What to do.** Write the symbol as `{R|t}`
  ([Operations: a Seitz symbol and a Cartesian name](../background/conventions.md#operations-a-seitz-symbol-and-a-cartesian-name)).
  `R` is the rotation part: `1`, `-1`, `m`, `2`, `3`, `4`, `6`, `-3`, `-4` or `-6`. `3`,
  `4`, `6` and their negatives carry `+` or `-` for the sense of the turn. Every rotation
  part but `1` and `-1` is followed by an underscore and the direction of its axis, or of
  a mirror's normal. `t` is the translation, three fractions. A trailing `'` marks an
  operation combined with time reversal, as in `{-4+_001|0,0,1/2}'`.

#### `kpoint_not_tabulated`

Returned by `coreps_at`, `corep_table` and `pythtqc.groups.identify_from_characters`.

- **What it means.** The k-point is not one of the group's tabulated high-symmetry points
  in this cell, or is a name the tables do not hold. The tables list one member of each
  point's star. Another member has a conjugate little group, whose little-group
  representations the tables do not list.
- **What to do.** Ask a point `high_symmetry_points(group)` returns, by its name or its
  coordinates.

#### `out_of_plane`

Returned by `little_group`, `corep_table`, `coreps_at`, `characters`, `sewing_matrix`, the
methods of `SpaceGroupRep` that take a k-point, `pythtqc.groups.identify_from_characters`
and `regular_content`.

- **What it means.** The k-point has a component out of a layer's periodic plane, where the
  model has no dispersion.
- **What to do.** Ask an in-plane point.

#### `little_group`

Returned by `sewing_matrix` and `SpaceGroupRep.sewing_operator`.

- **What it means.** The operation carries `k` to a point that is not `k` up to a reciprocal
  lattice vector, so it is not in the little group of `k`, and neither `sewing_matrix` nor
  `SpaceGroupRep.sewing_operator` returns a matrix there.
- **What to do.** Pass an operation of the little group of `k`, which `little_group` lists.

#### `states_not_at_k`

Returned by `characters`, `coreps_at` and `sewing_matrix`; `hsp_table` keeps it for each
point where the states handed in do not solve `H(k)`.

- **What it means.** The energies and states handed in do not solve `H(k)` at the k-point
  asked: the largest entry of `|H(k) V − V diag(w)|`, with `V` the states as columns and
  `w` their energies, is above `1e-8` times `max|H|`. Characters read off states from
  another point, or another model, come out wrong without any sign: the sewing matrix
  built on them can be unitary all the same.
- **What to do.** Hand in the states of one diagonalization at that point, or the model,
  and let the call solve there
  ([One diagonalization, reused](../guide/labels.md#one-diagonalization-reused)).

#### `k_not_on_mesh`

Returned by `characters`, `coreps_at` and `sewing_matrix`; `hsp_table` keeps it for each
point its mesh does not contain.

- **What it means.** A solved `WFArray` handed in holds no point of its mesh whose reduced
  coordinates each lie within `1e-9` of those of the point asked. Handed a mesh, the call
  reads its states at that point and never solves the model there. The states at the
  nearest point of the mesh are not those at `k`, whose distance from it is `value`.
- **What to do.** Build the mesh to contain the point (`k_grid` reports which high-symmetry
  points a mesh holds), or hand in the model.

#### `asymmetric_grid`

Returned by `k_grid`.

- **What it means.** The group does not carry the mesh asked for onto itself. The mesh
  returned holds its irreducible points, one of each set of mesh points the group's
  operations carry onto one another. On this mesh some points have images off it, and no
  such sets can be read.
- **What to do.** Ask a mesh the group carries onto itself. With
  `allow_asymmetric=True`, the mesh is returned with no irreducible points.

#### `negative_multiplicity`

Returned by `content_of`, `decompose`, `symmetry_indicator`, `subtract`,
`regular_completion` and `srsi_of_bands`, each when a symmetry data vector handed in has a
negative multiplicity. `subtract` returns it also when the difference has one, and
`regular_completion` when the set of bands has more of some little-group representation
than the regular copies.

- **What it means.** A symmetry data vector counts the little-group representations of a
  set of bands, so none of its multiplicities is negative. A negative multiplicity in the
  difference `subtract` takes means that the second set of bands is not contained in the
  first. From `regular_completion` it means the set of bands has some little-group
  representations more times than the regular copies do, by the amounts the reason
  prints. A combination of elementary band representations with coefficients of either
  sign is meaningful: the symmetry data vector of a fragile set of bands is one. A vector
  with a negative multiplicity is the symmetry data vector of no set of bands.
- **What to do.** Pass nonnegative multiplicities. For `subtract`, pass as second argument
  a set of bands contained in the first. For `regular_completion`, pass a larger multiple
  of the number of bands in one copy to `bands=`; where the reason says the copies hold
  none of some names, check those names in the symmetry data vector.

#### `non_integer_multiplicity`

Returned by `content_of`, `decompose`, `symmetry_indicator`, `subtract`,
`regular_completion` and `srsi_of_bands`, each when a symmetry data vector handed in has a
multiplicity that is not an integer.

- **What it means.** A symmetry data vector counts little-group representations, so each
  of its multiplicities is an integer.
- **What to do.** Pass integer multiplicities.

#### `not_a_regular_multiple`

Returned by `regular_completion`, when the number of bands passed to `bands=` is not a
multiple of the number of bands in one copy of the regular band representation.

- **What it means.** One copy of the regular band representation is the band
  representation induced from the general Wyckoff position. It carries one state there,
  or the Kramers pair for a spinful model in a group with time reversal. So one copy has
  `|U|` bands at every k-point, or `2|U|` for the Kramers pair. `|U|` is the number of
  unitary operations of the group up to lattice translations, counted in the single-valued
  group for a spinful model. Whole copies have a multiple of that number of bands, and no
  completion to another number is defined.
- **What to do.** Pass a multiple of the number of bands in one copy to `bands=`; the
  reason names the next one.

#### `not_a_spinful_model`

Returned by `spinless_presentation`.

- **What it means.** The spinless presentation is built from a model with spin stored by
  pythtb, and this model has none: it is already in the presentation the call would
  produce.
- **What to do.** To say how its orbital index carries spin, use `spin_basis`.

## The package's checks on itself

Each check here measures an identity that holds in exact arithmetic, or compares two
measurements of one thing. Met from `realized_group` and the calls on its record, one that
fails means the package has met a case it does not handle. Nothing in the model is to
change, and the refusal is worth reporting, with the model and the call. A few also say
something about arguments made by hand, and `rep_not_covariant` close to `tol` says
something about the Hamiltonian, as their entries note.

### The representation

#### `rep_not_covariant`

Returned by `realized_group`.

- **What it means.** No candidate representation of the operations the Hamiltonian was
  measured to obey carries `H(k)` onto `H(Rk)` within the tolerance it is held to. Each
  operation was kept because its covariance residual is at most `tol`, so a
  representation of them should pass. The representation is built twice. The
  representation built first, to project `H` onto its symmetric part, is held to `tol`.
  The one returned is built last and held to a tighter tolerance set by rounding, which
  the reason quotes. Two cases are returned as this refusal, each naming the group
  identified. A miss that the centers' miss accounts for, with no tolerance declared for
  the centers, is refused `positions_not_exactly_symmetric`. Any other miss is the
  package disagreeing with itself, and raises `InternalError`.
- **Close to `tol`.** That first representation carries `H(k)` onto `H(Rk)` only to
  `value`, beyond `tol`. Each operation is obeyed within `tol` on its own, the worst to
  `context["obeyed_alone"]`. For each operation the representation misses, the two
  measurements, the operation alone and in the representation, agree within `tol`. The
  Hamiltonian's asymmetry is then close to `tol`, and the precision decides which side
  of it the Hamiltonian is on. A declared `precision=` sets `tol` to
  `10·precision·max|H|`. Declare one whose `tol` lies above these numbers if the
  asymmetry is numerical noise, and the Hamiltonian is made symmetric under the group.
  Declare one below them if it is a term of the model to keep, and the operations it
  breaks are dropped.
- **Within `tol`.** The representation returned carries `H(k)` onto `H(Rk)` to `value`,
  beyond the tighter tolerance and within `tol`. The reason then calls the operations
  near-symmetric, obeyed within `tol` and not exactly. Report it, with the model and the
  call.

#### `rep_internal_factor`

Returned by `realized_group`.

- **What it means.** Candidate matrices carry `H(k)` onto `H(Rk)` as the group requires,
  but on none of them does a site's block take the form a representation of these
  operations has: a character times the operation's own matrix on the multiplet. That
  matrix is the operation's spin rotation, or the identity without spin, on a
  one-dimensional multiplet, and the block the hoppings fix on a wider one. The candidate
  matrices are then some other operations' matrices. With no site character to read off
  them, no orbital can be declared or labeled.
- **What to do.** Where the multiplets were stated with `site_multiplets=`, check the
  statement. Otherwise report it.

#### `rep_not_multiplicative`

Returned by `realized_group`.

- **What it means.** Candidate matrices pass the covariance and site-matrix checks, but none
  reproduces the group's multiplication table. Either they form a projective
  representation of a projective class the table does not carry, or a conserved
  pseudospin admits no representation of the group on its copies. Where the reason names
  neither a projective class nor a pseudospin, the operations, the generators or the
  blocks on the multiplets are wrong.
- **When it has a route.** Spinor states held in a spinless model's orbital index carry
  the sign of spin 1/2 (`context["spin_half"]`). Declare `spin=`, and they are read in
  the double group.
- **Otherwise.** Report it.

#### `rep_not_unique`

Returned by `realized_group`.

- **What it means.** More than one distinct representation passes every check, so the
  choice the measurement should fix is not fixed: the k-points the candidates are tested at
  are too few, or the model carries more symmetry than the group. Two candidates `D(g)`
  and `D(g^-1)` that give the same site irreps under the same labeling are one
  representation up to convention; where they give different site irreps, they are two.
- **What to do.** Report it, with the model and the call: the measurement should have
  fixed this choice.

#### `canonical_norm`

Returned by `realized_group` and `pythtqc.symmetry.normalize_rep`.

- **What it means.** The phase `λ` of an operation's matrix is fixed first by the
  normalization `(λU)^n = ζ·1`. Here `U` is the unitary matrix nearest the matrix `D`
  handed in, `n` the order of the operation modulo lattice translations, and `ζ` the sign
  the group's multiplication table requires. For an antiunitary operation every second
  factor of the power is complex conjugated. Where `U^n` is not a multiple of the
  identity, the normalization has no solution: `D` is the matrix of no operation of the
  group, and no phase can be fixed.
- **What to do.** A matrix handed to `normalize_rep` by hand is not the operation's. Met
  from `realized_group`, report it.

#### `antiunitary_anchor`

Returned by `realized_group` and `pythtqc.symmetry.normalize_rep`.

- **What it means.** No phase the normalization allows makes the matrix agree with time
  reversal, `T conj(λU) T† = λU` for `T` the matrix of time reversal, or of another
  antiunitary operation where the group has none. Where `T conj(U) T† = ρU`, this holds
  only for `λ^2 = ρ`, and no such `λ` is among the phases the normalization leaves. The
  matrix and the time reversal then belong to different representations, and corep
  labels built on them would be meaningless.
- **What to do.** As for `canonical_norm`.

#### `little_group_mismatch`

Returned by `coreps_at` and `pythtqc.groups.identify_from_characters`.

- **What it means.** The unitary operations of the little group measured at a point do
  not match the tables' list for it, so the tables' little-group representations cannot
  be read on them.
- **What to do.** Report it.

#### `site_group`

Returned by `realized_group`, through a declaration.

- **What it means.** A site's operations are not closed under the group's product, or the
  character table computed for the site group departs from orthogonality by more than
  `1e-6` times the site group's order. For each pair of rows, orthogonality requires the
  sum over the site group of one row's characters times the other's conjugate to be the
  site group's order for a row with itself and zero for two different rows. The
  departure the reason prints is the largest amount by which such a sum misses its
  required value. Neither leaves a table on which site irreps can be named.
- **What to do.** Report it, with the model and the call.

### The projection, and the Wyckoff positions

#### `symmetrize_defect_exceeds_tol`

Returned by `realized_group`.

- **What it means.** The group average moved `H` by more than the declared `tol`. Over
  operations obeyed within `tol` the move is of the order of their residuals, so a larger
  one says the matrices averaged with are not the symmetry the residuals were measured
  for.
- **What to do.** Where the covariance the reason quotes is close to `tol`, the tolerance
  is too tight: declare a larger precision. Where the reason quotes a miss of the centers,
  declare the model's precision, and the centers within it are moved onto their positions
  before anything is projected. Otherwise report it, with the model and the call.

#### `symmetrize_range_not_closed`

Returned by `realized_group`.

- **What it means.** The symmetrized hopping list has weight outside the range the group can
  carry a hopping to. The transform back to hoppings on the grid is exact only while the
  hoppings fit inside that range, so it aliased, and the hopping list would not be the
  symmetrized Hamiltonian.
- **What to do.** Where the reason quotes a miss of the centers, declare the model's
  precision, as for `symmetrize_defect_exceeds_tol`. Otherwise report it, with the model
  and the call.

#### `wyckoff_setting`

Returned by `wyckoff_sites_tables`, `srsi_of_model` and `splitting`.

- **What it means.** The list of Wyckoff positions read is not written in the conventional
  cell used by the irreptables package. Or the Wyckoff position of an orbit read from its
  centers differs from the one at which its measured symmetry data vector is a sum of
  elementary band representations, which says the positions are read in another cell or
  about another origin. Every Wyckoff position returned would be misread, so none is, and
  no stable real-space invariant is returned.
- **What to do.** Report it.

#### `wyckoff_unresolved`

Returned by `wyckoff_sites_tables`, `srsi_of_model` and `splitting`.

- **What it means.** A position, carried into the conventional cell used by the
  irreptables package, lies on no Wyckoff position of the group within the tolerance
  `wyckoff_sites_tables` takes as `position_tol=`, `1e-4` in reduced coordinates by
  default. The general position holds every point of the cell, so this is met only where
  the list of Wyckoff positions read from the pyxtal package lacks it. It is the
  package's guard on that list, and nothing in the model is at fault.
- **What to do.** Report it, with the model and the call.

## Checks no returned group or public call reaches

### A group made by hand

#### `lattice_not_descended`

A group whose lattice is finer than the model's, such as the group of the primitive cell
of a model written on a supercell, handed to a call that reads the tables. Its tables are
written on the finer lattice, so every label read from them would name a k-point of that
lattice's Brillouin zone, folded onto the model's. `realized_group` never returns one. It
refuses `supercell`, or descends to the model's own lattice.

#### `magnetic_type_out_of_scope`

A magnetic group of type III or IV handed to `symmetry_indicator`, `regular_content` or
`regular_completion`. The package reads types I and II, and `realized_group` returns no
other: a Hamiltonian that breaks time reversal is returned the unitary subgroup of its
magnetic group
([Magnetic groups of types I and II only](../background/limits.md#magnetic-groups-of-types-i-and-ii-only)).

### No public call

#### `declaration_breaks_group`

`realized_group` lowers the group to the largest subgroup under which the hoppings are
covariant with the declared site irreps, each broken operation under
`refuted_by = "declaration"`
([A declaration that lowers the group](../guide/orbitals.md#a-declaration-that-lowers-the-group)).
A declaration says what the orbitals are, so hoppings that are not covariant with it
describe a crystal with fewer operations, never a wrong declaration. Only a
representation built on a group fixed in advance, which cannot be lowered, refuses
instead, naming the subgroup `realized_group` descends to.

#### `symmetrize_broken_operation`

A measurement handed to the projection of `H` onto its symmetric part lists an operation
broken at `tol`, and the projection is not made. Averaging over an operation the
Hamiltonian does not obey would impose it, moving `H` by its residual. `realized_group`
averages only over the operations it kept
([A broken operation is never imposed](../guide/numerical.md#a-broken-operation-is-never-imposed)).

#### `irreptables_indicator_tables`

The indicator table of the irreptables package could not be read. The names of the
indicator components a call returns are those of the paper that tabulates the stable
real-space invariants, Hwang et al. (2026), and are not paired with the irreptables
package's own.

#### `wyckoff_enumeration`

The Wyckoff positions of the group could not be listed, and the reason quotes `irrep`'s
own message. Where the enumeration fails, the orbital's Wyckoff letter as spglib names it
is left empty.
