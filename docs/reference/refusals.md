# Refusals

Every refusal a call can return, by the name of its check: what it means, its usual cause,
and what to do. A refusal is a returned, falsy `Refusal` record that carries the check that
stopped the call, a short reason, the value the check measured against its
threshold, and a `context` with everything needed to act on it;
[when a call refuses](../guide/refusals.md) reads one through, and each call's entry in
[the functions reference](functions.md) lists the refusals it can return. Each entry below
is headed by its check, the name a refusal prints in brackets, and opens with the calls
that return it; its words are [the glossary's](glossary.md).

The entries are grouped by kind, and the kind says where the fix lies:

- [A verdict about the physics](#a-verdict-about-the-physics): what was asked does not
  exist for these bands, or for this model as written. The fix is in the question or in
  the model.
- [A practical limit](#a-practical-limit): a search, an enumeration or a memory budget
  past its ceiling, or tables the installation lacks. The refusal names the number, or
  the data, that runs it.
- [A grammar error in a declaration or a question](#a-grammar-error-in-a-declaration-or-a-question):
  a declaration that names nothing at its site, or a call asked something it does not
  take. The reason says what would parse.
- [The package's checks on itself](#the-packages-checks-on-itself): an identity that
  holds in exact arithmetic was measured not to. Nothing in the model is to change;
  report it, with the model and the call.

A refusal that `realized_group` returns once the group is identified names that group, its
number and how many of its operations the Hamiltonian obeys, and carries it in
`context["group"]`; one reached while the group is still being measured names the group
the cell has at that point, an upper bound on what the Hamiltonian obeys. Where neither
`precision=` nor `position_tol=` is declared, the orbital centers miss the positions the
group puts them at by more than rounding, and `H` was projected onto its symmetric part,
the reason also quotes that miss `δ`, in reduced coordinates, and `context["position_defect"]`
holds it. In pythtb's convention I the miss alone costs `H(k)` a covariance residual of order
`2π·δ·max|H|`, `max|H|` the largest entry of `H(k)` once the mean on-site energy is
subtracted, which no projection of the hoppings removes, so the refusal may be that miss
alone. A declared `precision=` moves the centers onto those positions, a gauge
transformation that leaves the spectrum untouched. A disagreement
inside the package raises `InternalError`. A malformed argument raises `ValueError`
before anything is measured.

## A verdict about the physics

### The group, and how the model is written

#### `supercell`

Returned by `realized_group`.

- **What it means.** The Hamiltonian obeys a pure translation of its own cell, as written
  or once each site's orbitals are carried onto its translate's by a change of basis
  within the site, so the model is written on a supercell of its primitive cell: the bands
  at one `k` of this cell are the crystal's at several `k` of its own zone, a combination
  no table names. Every label would name the wrong k-points. It is one of the two cases
  in which the group question is refused
  ([when the group question is refused](../background/measurement.md#when-the-group-question-is-refused)),
  and it fires at any index.
- **Usual cause.** Copies of one site that the hoppings do not tell apart, such as
  pythtb's checkerboard with no on-site offset, written alike or in frames turned from
  site to site.
- **What to do.** Rebuild the model on its primitive cell and ask again:
  `context["primitive_lattice"]` holds its vectors, `context["translations"]` the
  translations obeyed and `context["index"]` the index, and `context["dressing"]`, for
  each translation, the `matrix` carrying each site's orbitals onto its translate's and
  the `turn` it makes, `0.0` where the copies are written alike; copies written in turned
  frames are written in one basis first. Where the copies are meant to differ, the
  difference belongs in the Hamiltonian: a supercell whose hoppings tell the copies apart
  is a primitive model of the bigger cell, and the group descends to it
  ([when the cell is a supercell](../guide/group.md#when-the-cell-is-a-supercell)).
  A supercell is refused and is not folded onto its primitive cell
  ([what the package cannot do yet](../background/limits.md#a-supercell-is-refused-not-folded)).

#### `positions_not_exactly_symmetric`

Returned by `realized_group`.

- **What it means.** The orbital centers miss the cell's own operations by a defect `δ`,
  in reduced coordinates, that alone costs `H(k)` a covariance residual of order
  `2π·δ·max|H|` above `tol`. `tol` is the tolerance every operation's covariance residual
  is held to. `max|H|` is the largest entry of `H(k)` once the mean on-site energy is
  subtracted. Which operations the Hamiltonian obeys would then be read off the
  geometry as much as off the hoppings. Nothing is built. It is the other case in
  which the group question is refused, and the reason names the cell's group as
  an upper bound. Centers missed by less are refused the same way once the group is
  identified, and the reason names that group, wherever `H(k)` is to be made or held
  exactly symmetric and the miss accounts for what that check measured. The miss alone
  then moves `H(k)` by more than the check allows, by up to `context["reach"]`, and the
  more at an image point the larger its reduced coordinates. The checks are the
  covariance of the group's representation and the projection of the hoppings onto their
  symmetric part, whose operators carry each center onto its image to `1e-9`. Centers
  that miss the operations by more than `1e-9` are refused before the projection is
  made: averaged with operators that miss the centers, each hopping would be carried to a
  vector that is not a lattice vector, and no projection of the hoppings removes a defect
  in the centers. On a cell whose sites of one species (`typat=`) repeat under a fraction
  of a cell's translation, the tables' setting is not read and the operations cannot be
  rebuilt from it: centers missed by less are refused the same way before anything is
  built wherever their miss leaves the operations read off them open, the cell's group
  named as an upper bound in the reason and by `context["number_str"]` and
  `context["name"]`; no group is identified, and the context carries none. Operations
  that do not close leave no exactly symmetric set of centers to measure `H` against.
- **Usual cause.** A numerical model, such as a Wannier90 import, asked with no precision
  declared; or centers converted onto a strongly skewed lattice basis in floating point.
- **What to do.** Declare the model's precision, `precision=`, or `position_tol=` for the
  centers alone: centers within it are moved onto the positions the group puts them at, a
  k-dependent gauge transformation that leaves the spectrum untouched. `value` and
  `context["position_defect"]` are `δ`
  ([the centres first](../guide/numerical.md#the-centres-first-because-moving-them-is-a-gauge)).
  Centers noisier than the symmetry finder's `symprec` are not refused this way: the
  finder reads them as a less symmetric set, and the group returned can be as low as `P1`
  ([centres noisier than symprec](../guide/numerical.md#centres-noisier-than-symprec)).

#### `cell_group_not_realized`

Returned by `realized_group`.

- **What it means.** An operation combined with time reversal that held before a
  numerical model's Hamiltonian was projected onto its symmetric part is broken after it,
  so the group the projection was made in is not the projected Hamiltonian's group. That
  group is what the positions allow, an upper bound on what the Hamiltonian obeys, and a
  representation of it would label the bands with operations the Hamiltonian breaks.
- **Usual cause.** A model obeying an operation combined with time reversal only within
  its declared precision.
- **What to do.** Read `context["validated"]`, each operation's measurement and residual,
  and `context["broken"]`.

#### `declaration_breaks_group`

Returned by no public call: `realized_group` lowers the group to the subgroup the
declared content is covariant on, each broken operation under
`refuted_by = "declaration"`
([a declaration that lowers the group](../guide/orbitals.md#a-declaration-that-lowers-the-group)).
A declaration says what the orbitals are, so hoppings that are not covariant under it
describe a crystal with fewer operations, never a wrong declaration. Only a
representation built on a group fixed in advance, which cannot be lowered, refuses
instead, naming the subgroup `realized_group` descends to.

#### `wyckoff_orbit`

Returned by `realized_group`.

- **What it means.** An operation of the group carries an orbital to a point where no
  orbital of its kind sits, so the orbital lies on no Wyckoff orbit of the cell.
- **Usual cause.** A misplaced orbital.
- **What to do.** Check the orbital's position: `context` names the orbital, the
  operation, its image and the distance to the nearest orbital of the same kind.

#### `wyckoff_unresolved`

Returned by `wyckoff_sites_tables` and `srsi_of_model`.

- **What it means.** A position, carried into the tables' reference cell, lies on no Wyckoff
  position of the group within the tolerance, `1e-4` in reduced coordinates by default.
- **Usual cause.** A center far off every orbit: a position written in another cell or
  origin, or mistyped.
- **What to do.** Check the position; the reason names the nearest Wyckoff position and
  how far it is.

#### `operations_not_consistent`

Returned by `realized_group` where the centers cannot be read as a symmetric set at all, as
when two sites of one species sit closer together than the symmetry finder can tell apart,
so that an operation does not carry the sites onto one another one for one. Operations
left open by the centers' miss alone, with no precision declared, are refused
[`positions_not_exactly_symmetric`](#positions_not_exactly_symmetric) instead.

- **What it means.** The operations the symmetry finder reads off the centers do not close
  into a group, and the cell could not be matched to the setting of the tables, from which
  exact operations would be rebuilt. Every check after it would be measured against a set
  that is not a group. The reason gives how far they are from closing: the largest miss of
  `t_g + R_g t_h - t_gh` from a lattice vector, in reduced coordinates, where `R_g` and
  `t_g` are the rotation and translation of operation `g`. It sets that against the most
  a cell built from exact numbers misses by, `1e-12` times the largest row sum of any
  `|R_g|`.
- **Usual cause.** Sites of one species closer together than `symprec`, in a cell the
  tables' setting search does not match. The finder reads each operation's translation
  off the pairs of sites the operation maps. Sites closer than `symprec` are paired
  wrongly, so the translations read off them do not compose.
- **What to do.** Identify at a `symprec` smaller than the distance between the sites, so
  that the finder tells them apart.

#### `operations_not_nearly_consistent`

Returned by `realized_group` where `operations_not_consistent` is: with no precision declared
for the centers, or where they cannot be read as a symmetric set at all.

- **What it means.** The operations do not close, and rebuilding their translations from
  the tables' own would move one by more than `symprec`, a length in the model's units: the
  tabulated group is not the one these centers carry at this tolerance. Rebuilt, the
  operations would describe a crystal the model does not have.
- **Usual cause.** Centers noisier than `symprec` allows for, or a `symprec` too large for
  the cell, with no precision declared.
- **What to do.** Declare the model's precision, `precision=`, which makes the centers exact
  before the setting is read; identify again at a `symprec` the centers support; or state
  what the crystal is with `typat=` or `decorations=`.

#### `positions_not_nearly_symmetric`

Returned by `realized_group`.

- **What it means.** Under a tolerance declared for the orbital centers, `precision=` or
  `position_tol=`, the centers cannot be moved onto the positions the group puts them at:
  its operations do not carry them onto one another at the symmetry finder's distance, or
  orbitals read as one site sit farther apart than the tolerance, which no group of
  operations brings together. Centers that far off are a different set of points. Other
  centers off by more than the tolerance are not refused: the operations that would move
  one farther than the tolerance are broken by the centers, and the group descends to the
  largest group of its operations that carries them.
- **Usual cause.** A group the centers do not have, or a displacement beyond the noise.
- **What to do.** Identify the group the centers carry (`typat=`, `decorations=`).

#### `symmetrize_broken_operation`

Returned by no public call: `realized_group` averages only over the operations it kept
at the declared precision
([a broken operation](../guide/numerical.md#a-broken-operation-is-never-symmetrized-into)).

- **What it means.** A measurement handed to the projection of `H` onto its symmetric
  part lists an operation broken at the declared tolerance, and the projection is not
  made. Averaging over an operation the Hamiltonian does not obey would impose it, moving
  `H` by its residual, which is noise only within the declared tolerance.

### Spin

#### `rep_double_valued_spinless`

Returned by `realized_group`.

- **What it means.** A spinless model was asked in double-valued tables, with nothing saying
  where the spin is. A double-valued representation on spin-free orbitals would say each
  orbital carries half-integer spin itself, and no such representation can be anchored to
  the tables.
- **Usual cause.** A spinor wannier90 import, which pythtb returns spinless with the spin
  inside the orbital index, asked with `spinor=True` and no `spin=`.
- **What to do.** Declare the spinor basis, `spin="interleaved"`, `spin="blocked"` or
  explicit pairs ([a wannier90 spinor import](../guide/spin.md#a-wannier90-spinor-import));
  ask the single-valued group, `spinor=False`; or build the model with spin, if the
  half-integer content is the electron's own. [An unpaired spinor
  orbital](../background/limits.md#an-unpaired-spinor-orbital) is outside the package's
  scope.

#### `single_valued_spinful`

Returned by `realized_group`.

- **What it means.** `spinor=False` was asked of a spinful model. Its states carry the
  electron's spin one-half, so the little-group representations of its bands are
  double-valued, and single-valued tables do not describe them. A crystal is refused
  before its group is identified and names none. A layer, or a crystal given
  `decorations=`, is refused once the group its cell allows is identified, and names
  that group, carried in `context["group"]`. The converse, `spinor=True` for a spinless
  model, is not this refusal: the group is identified, and its representation is refused
  [`rep_double_valued_spinless`](#rep_double_valued_spinless).
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
  ([fewer than two periodic directions](../background/limits.md#fewer-than-two-periodic-directions)).

#### `srsi_scope`

Returned by `srsi_tables`, `srsi_of_bands`, `srsi_of_model`, `splitting` and
`pythtqc.srsi.canonical_determinism`.

- **What it means.** The group is not grey, or the number handed to `srsi_tables` is
  outside 1 to 230. The stable real-space invariants are tabulated for grey groups alone,
  time reversal adjoined: 217 of the 230, the other thirteen refusing
  [`srsi_table_absent`](#srsi_table_absent).
- **Usual cause.** A Hamiltonian that breaks time reversal, which is returned an ordinary
  space group; or a number outside 1 to 230.
- **What to do.** No table of stable real-space invariants exists for such a group; the
  decomposition and the indicator still apply.

#### `srsi_table_absent`

Returned by the same calls as `srsi_scope`.

- **What it means.** The group is one of the thirteen the tables omit, numbers 1, 4, 7, 9,
  19, 29, 33, 76, 78, 144, 145, 169 and 170, whose only stable real-space invariant is the
  band count.
- **Usual cause.** A model in one of those groups.
- **What to do.** The band count is the invariant, and there is nothing else to compute.

### Labels at a k-point

#### `multiplet_split`

Returned by `characters`, `coreps_at` and `sewing_matrix`; `hsp_table` keeps it
for each point it refuses at.

- **What it means.** The set of bands asked for cuts a degenerate multiplet at `k`, so its
  sewing matrices and characters would depend on the basis the diagonalization chose
  inside the multiplet.
- **Usual cause.** A band count that ends inside a degeneracy: half a Kramers pair, or bands
  that meet at the point, as a semimetal's do.
- **What to do.** Label the whole multiplet, or the isolated set of bands:
  `context["multiplet"]` lists the bands that are degenerate there
  ([label an isolated set of bands](../guide/labels.md#label-an-isolated-set-of-bands)).

#### `integer_multiplicity`

Returned by `coreps_at` and `pythtqc.groups.identify_from_characters`.

- **What it means.** The multiplicities of the coreps fitted to the characters at a point are
  not non-negative integers within `label_tol`, `1e-3` by default.
- **Usual cause.** A degeneracy split into several multiplets by numerical noise beyond
  `degeneracy_tol`: a two-dimensional corep split in two reads exactly 0.5. Levels merged
  into one multiplet never refuse; they are labeled as a sum of coreps. From
  `identify_from_characters`, also characters measured in a convention other than the
  one `characters` measures in.
- **What to do.** On a numerical model, declare its precision to `realized_group`
  (`precision=`), which moves `H` onto its symmetric part, or set `degeneracy_tol`, an
  energy, to the noise its degeneracies carry; `context["rows"]` names the multiplets at fault and
  `context["multiplicities"]` the fitted numbers.

#### `multiplet_dimension`

Returned by `coreps_at`, through its identification of the coreps, and by
`pythtqc.groups.identify_from_characters` given `multiplet_dims=`.

- **What it means.** The coreps identified from a multiplet's characters do not add up to
  the multiplet's degeneracy, so the grouping of levels and the characters disagree.
- **Usual cause.** Characters that are no representation's, whose fitted multiplicities
  still round to integers. Merged levels are a representation's and fill their multiplet,
  and a degeneracy split in two is refused `integer_multiplicity` first.
- **What to do.** As for `integer_multiplicity`. On a model built from exact numbers with
  well separated levels, report it.

### Atomic limits and indicators

#### `incomplete_content`

Returned by `decompose`, `symmetry_indicator`, `subtract`, `regular_completion`,
`splitting` and `srsi_of_bands`, each when it is handed a table that refused at some
high-symmetry point; and by `pythtqc.groups.decompose_ebrs`.

- **What it means.** The table holds no little-group representations at that point. A
  decomposition, a class or a difference of the multiplicities it does hold would be a
  verdict on a different symmetry data vector. Leaving a point out removes conditions, so a
  decomposition of the rest is easier to find than one of the whole vector. From
  `decompose_ebrs`, the symmetry data vector handed in holds no label at some point of the
  group's elementary band representations, where it would read zero multiplicities.
- **Usual cause.** A set of bands that cuts a multiplet at that point, or another refusal
  recorded in the table there. From `decompose_ebrs`, also a layer's vector, which holds no
  point out of its plane.
- **What to do.** Pass `decompose_ebrs` multiplicities at every point; for a layer, use
  `decompose`, which reads the vector on the points of the plane. Otherwise read the
  table's refusals, which `context["refusals"]` repeats, and fix the set of bands. Where a point's tables cannot be read, `decompose` takes
  `allow_incomplete=True` and decomposes the multiplicities at the other points; it is not
  for a point where the physics refused. The record then says whether its sums are exact.
  They are exact where, for every little-group representation left out, one kept occurs as
  often as it in every elementary band representation, and a necessary condition
  otherwise. No other call takes this override.

#### `not_a_band_structure`

Returned by `decompose`, `symmetry_indicator`,
`regular_completion`, `splitting` and `srsi_of_bands`.

- **What it means.** The symmetry data vector violates the group's compatibility
  relations: written in the basis of the Smith normal form of the group's matrix of
  elementary band representations, it has a nonzero coordinate where every band structure
  of the group has zero. So it lies outside the lattice of band structures, with no class
  in `{BS}/{AI}` and no decomposition
  ([how a symmetry indicator is computed](../background/tqc-primer.md#how-a-symmetry-indicator-is-computed)).
- **Usual cause.** Bands asked for that do not make a full set at every `k`, as when they
  cut through a crossing between two high-symmetry points, or labels read in another group.
- **What to do.** Check that the set of bands is isolated everywhere and that its labels
  came from this group; `context["violations"]` lists the coordinates that do not vanish,
  each as `(index, value)`, the index that of the coordinate in `context["y_prime"]`.

#### `indicator_not_evaluable`

Returned by `symmetry_indicator` and `srsi_of_bands`.

- **What it means.** The multiplicities in the plane do not fix those of the little-group
  representations the refusal names out of it, since none of them occurs in every
  elementary band representation as often as one in the plane. For a table they are
  measured on the layer's decoupled stack, the crystal that has those points, and that
  measurement stopped. The class is read from the whole symmetry data vector, so without
  them it has no value.
- **Usual cause.** A layer's symmetry data vector written out by hand whose little-group
  representations out of the plane cannot be read off the layer's own band representations
  in the group's cell; or a check of the stack that did not complete, which
  `context["stack"]` names.
- **What to do.** For a vector written out by hand, pass the table `hsp_table` returns for
  the same bands, whose little-group representations out of the plane are measured on the
  layer's decoupled stack. For a table, `context["stack"]` holds the measurement on the
  stack up to where it stopped, with the number measured by each check made before then
  ([a layer's class](../guide/indicators.md#a-layers-class-is-a-three-dimensional-class)).

#### `restriction_not_exact`

Returned by `in_plane_restriction`, `decompose` and `regular_completion`. Each returns it
when, for some little-group representation at a point not measured, no measured one occurs
as often as it in every elementary band representation of the group.

- **What it means.** The multiplicity of each such little-group representation is a
  condition of its own on a symmetry data vector, which the measured multiplicities do not
  fix, so no verdict read from them is exact for this group.
- **Usual cause.** `in_plane_restriction` handed `kpoints=` naming fewer points than the
  plane holds, or asked of a free-standing layer's group over every elementary band
  representation, some of whose little-group representations at `k_z = 1/2` are of this
  kind. `decompose` and `regular_completion` read a layer over its own elementary band
  representations, those of the Wyckoff positions within it. A layer's operations have no
  translation along its normal and map the layer onto itself, so on these every
  little-group representation out of the plane occurs as often as one in it; one that does
  not raises `InternalError`. Those two calls return the refusal only where the Wyckoff
  positions of the tables cannot be read in the group's cell, so that every elementary band
  representation is kept.
- **What to do.** For `in_plane_restriction`, name every point of the plane in `kpoints=`,
  or leave `kpoints=` out, which is enough for a crystal. A layer's little-group
  representations at `k_z = 1/2` are measured on its decoupled stack:
  `symmetry_indicator`, handed the table, reads the class there
  ([symmetry indicators](../guide/indicators.md)). From `decompose` and
  `regular_completion`, the tables' Wyckoff positions could not be read in the group's
  cell; report it, with the model and the call.

### Stable real-space invariants

#### `wyckoff_orbit_content_incomplete`

Returned by `srsi_of_model` and `splitting`.

- **What it means.** The symmetry data vector of a Wyckoff orbit's own orbitals could not be
  read at a high-symmetry point, for a reason other than a trivial little group, so it is
  incomplete.
- **Usual cause.** A refusal at that point, which `context["refusal"]` carries.
- **What to do.** Read that refusal; its entry here says what to do.

#### `wyckoff_orbit_not_localizable`

Returned by `srsi_of_model` and `splitting`.

- **What it means.** No sum of elementary band representations at any one Wyckoff position
  a Wyckoff orbit's orbitals can slide onto carries its measured symmetry data vector.
  Those positions are on the closure of its own position: the points its free coordinates
  sweep and the higher-symmetry positions among them. Where the tables list elementary
  band representations at no position on that closure, there is no position to read the
  orbit at. Nor could the orbit's invariants be read from its site irreps. That is how the
  invariants of composite site irreps are read, those whose band representation is a sum
  of elementary ones at other positions. For a spinless model, each site irrep of the
  orbit is identified with one the tables list at its position, and here one could not
  be. The double-valued tables hold no composite site irrep, and a spinful model's orbit
  is not read this way. In either case the orbit's symmetry data vector is consistent
  with no atomic limit at the position it sits on, which is a statement about the model.
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
  decoupled layer: stacking it would either drop those hoppings or give `H(k)` a `k_z` it
  does not have.
- **Usual cause.** A hopping table written directly, past pythtb's `set_hop`, which
  refuses a hopping with a component along a non-periodic direction.
- **What to do.** Check the model's hoppings along its non-periodic direction.

## A practical limit

### Searches and memory

#### `rep_search_too_large`

Returned by `realized_group`.

- **What it means.** The representation's discrete choices, which the measurement does not
  fix, had to be enumerated, and there were more than `max_candidates` of them: a limit on
  the search. Where the Hamiltonian conserves a pseudospin of n copies, it does not fix
  the representation the copies carry. The candidates are then the sums of total
  dimension n of the irreducible projective representations of the unitary operations
  that map one set of the copies onto itself. Where it splits a multiplet into sets of
  states it does not couple and no operation exchanges, it does not fix the representation
  of each set either. The candidates are then every way of choosing one for each set. A
  declaration on orbitals whose states come in identical copies is compared with every
  representation the hoppings leave open on the copies, all at once, and that comparison
  has the same limit.
- **Usual cause.** A conserved operator, or accidental degeneracy, inside a multiplet of
  several orbitals with nothing declared, so the multiplet's block is fixed only up to the
  conserved part; or a declaration on orbitals that carry such a conserved operator.
- **What to do.** Declare what the orbitals are (`characters=`), which states what the
  search would otherwise choose among; or run the same call with a larger
  `max_candidates=`, as in `realized_group(..., max_candidates=...)`, which runs a longer
  search. Where the reason names a declaration being compared, declare fewer of the
  orbitals it names, which leaves fewer representations to compare. Where the reason
  counts phases on orbits of multiplets, stating a site of several orbitals as one
  multiplet with `site_multiplets=` cuts the count.
  `context["fallback_reason"]`, where present, says why the choices could not be read off
  the measurement. Where it says a generator's block is not a multiple of the operation's
  own internal factor, then of the matrices that carry `H(k)` onto `H(Rk)` for that
  generator, the one nearest a permutation of multiplets times a multiple of that factor
  on each block misses that form by more than the tolerance it names, and a larger
  search does not change that.

#### `detector_too_large`

Returned by `realized_group`.

- **What it means.** Measuring an operation at one k-point needs more memory than the
  budget allows: a limit on memory, not a statement about the model. The operation's
  matrices are first sought there among pairs of eigenvectors, one of `H(k)` and one of
  `H(Rk)`, `Rk` the operation's image of `k`, whose energies are equal within rounding
  or the measurement's tolerance. The memory needed is one matrix of the size of `H` for
  each such pair, and one more. Narrowing the measurement to fit would drop some of those
  pairs and return a wrong solution rather than a coarse one.
- **Usual cause.** A large model with exactly degenerate levels, such as a big supercell's
  folded bands. A model whose degeneracies are lifted does not reach it.
- **What to do.** The same call with the `max_bytes=` the reason names. It fires while
  the operations are measured, so the group it names is the group the cell allows, an
  upper bound on the group `H` obeys.

#### `projection_too_large`

Returned by `realized_group`.

- **What it means.** The projection onto the group-invariant part needs more memory at its
  peak than `max_bytes` allows: it holds `H(k)` and its group average together on a grid
  of k-points. Along each periodic direction the grid has `2·(R + 1) + 1` divisions, `R`
  the number of cells the symmetrized hoppings can reach along it, the widest among the
  directions the group's operations mix. A coarser grid would alias the transform back to
  a hopping list, so the projection is not run on one: a limit on memory.
- **Usual cause.** A numerical model with many states and long-ranged hoppings.
- **What to do.** The same call with the `max_bytes=` the reason names
  ([a large model](../guide/numerical.md#a-large-model)).

#### `srsi_realizations_truncated`

Returned by `atomic_realizations`, and by `nsit_criterion` and `splitting` through it.

- **What it means.** The enumeration of atomic realizations of `theta_Z` found more than its
  cap. A no-go is asserted only on a complete enumeration, so with the list truncated neither
  "no pair adds up" nor "none exists" may be claimed. The refusal's `value` is `cap + 1`,
  the fewest realizations the search showed, and its `threshold` the cap.
- **Usual cause.** Many bands, or a group with many band representations.
- **What to do.** Only `atomic_realizations` takes `cap=`, `256` by default: called
  directly, raise it. `nsit_criterion` and `splitting` take no `cap=` and enumerate at
  that default, so their refusal has no argument to raise; `atomic_realizations` with a
  larger `cap=`, handed the group's tables and the side's `theta_Z`, which
  `context["theta_Z"]` carries, enumerates that side's realizations.

### The tables, as installed

#### `irrep_tables`

Returned by `realized_group`, `ebr_data`, `ebr_rows`, `decompose`, `symmetry_indicator`,
`regular_completion`, `in_plane_restriction`, `srsi_tables`, `wyckoff_sites_tables`,
`pythtqc.groups.decompose_ebrs` and `pythtqc.groups.indicator_group`.

- **What it means.** The character tables were not loaded, so no label can be read; or the
  tables carry no irrep table for this group, so its band representations cannot be carried
  into this package's convention.
- **Usual cause.** `irreptables` not installed, or its module `irreptables.irreps`
  imported before `irrep` in the process, which leaves `irrep` with a stand-in that labels
  nothing for the rest of the process.
- **What to do.** `tables_status()` says which: install `irreptables`, or, in a new
  process, import `pythtqc` before `irreptables`.

#### `tables_not_ready`

Returned by `operations`, `operation_index`, `high_symmetry_points`, `little_group`,
`corep_table`, `coreps_at`, `hsp_table`, `wyckoff_sites_tables`,
`in_plane_restriction`, `decompose`, `symmetry_indicator`, `regular_content`,
`regular_completion`, `srsi_of_bands`, `srsi_of_model` and
`pythtqc.groups.identify_from_characters`.

- **What it means.** The group was identified, but its cell could not be matched to the
  setting the tables are written in, so no table can be read for it. The reason carries
  `irrep`'s own message.
- **Usual cause.** A cell the search for the tables' setting does not match.
- **What to do.** The group stands; the labels need the tables. Report the model, with the
  reason.

#### `ebr_tables`

Returned by `ebr_data`, `ebr_rows`, `decompose`, `symmetry_indicator`,
`regular_completion`, `in_plane_restriction`, `srsi_tables`, `srsi_of_model`,
`wyckoff_sites_tables` and `pythtqc.groups.decompose_ebrs`.

- **What it means.** The installed tables carry no elementary band representations for this
  group and valuedness.
- **Usual cause.** A group or valuedness the tables do not cover.
- **What to do.** None in the model; the group and the labels stand without them.

#### `ebr_convention`

Returned by the same calls as `ebr_tables`.

- **What it means.** The shipped rows of this group's band representations cannot be
  carried out of the tables' phase convention into this package's
  ([conventions](../background/conventions.md)). Carried exactly, the rows at `GM` and
  the number of bands of each band representation at every point stay as the tables give
  them, and the complex conjugation of the labels is an involution that keeps each label
  at its point; rows that break any of these are refused, never carried.
- **Usual cause.** A defect in the tabulation itself: an entry without its name or its
  Wyckoff position, or characters at a point that are not closed under complex
  conjugation where neither the factor system nor the time-reversal gluing is complex.
- **What to do.** None in the model; report it.

#### `ortools`

Returned by `pythtqc.groups.decompose_ebrs`.

- **What it means.** The symmetry data vector is an integer combination of elementary band
  representations, and whether a combination with no negative coefficient exists, which
  tells a vector consistent with an atomic limit from a fragile one, needs the integer
  search of `ortools`, which is not importable.
- **Usual cause.** An installation without `ortools`, a dependency of the package.
- **What to do.** Install `ortools` 9.10 or later.

#### `srsi_data`

Returned by the same calls as [`srsi_scope`](#srsi_scope).

- **What it means.** The converted tables of stable real-space invariants, two data files
  the package ships, are not where it reads them.
- **Usual cause.** An installation whose package data is missing a file.
- **What to do.** `pythtqc.srsi.srsi_status()` says which file is missing and where it
  belongs. Reinstall the package, as the reason says.

#### `srsi_ebr_basis`

Returned by the same calls as `srsi_data`.

- **What it means.** The installed `irreptables` lists this group's elementary band
  representations differently from the list the converted tables were resolved against, so
  their stored columns would name the wrong ones.
- **Usual cause.** An `irreptables` installed after the tables were converted.
- **What to do.** Install the `irreptables` the tables were converted against, the
  version the reason names.

#### `irreptables_indicator_tables`

Returned by no call: a call returns the paper's names of the indicator components as they
are, never paired with `irreptables`' own.

- **What it means.** The indicator table of `irreptables` could not be read, so the
  paper's names of the indicator components cannot be paired with `irreptables`' own.
- **Usual cause.** An `irreptables` installation whose indicator data cannot be read.
- **What to do.** Check the `irreptables` installation.

#### `wyckoff_enumeration`

Returned by no call: where the enumeration fails, the orbital's letter in spglib's setting
is left empty.

- **What it means.** The Wyckoff positions of the group could not be enumerated; the reason
  quotes `irrep`'s own message.
- **What to do.** None in the model; report it, with the reason.

## A grammar error in a declaration or a question

### In `characters=` and `spin=`

#### `unknown_label`

Returned by `realized_group`, `decompose`, `symmetry_indicator`, `srsi_of_bands` and
`regular_completion`.

- **What it means.** From `realized_group`, a declaration names nothing at the orbital's
  site: a word that is not an irrep of its site group, a sum of irreps, a shell or a
  shorthand, or an operation the site group does not hold. On a spinful site, a sum of
  double-valued site irreps is refused where no sum of single-valued site irreps becomes
  it with spin. A shorthand is refused where its function and its images under the site
  group span more than one site irrep, counted with multiplicity. Generator characters
  are refused where a key matches operations of more than one conjugacy class. Generator
  matrices are refused where a key names more than one operation, where a matrix is not
  unitary, or where the matrices do not multiply as their operations do. A declaration
  outside the grammar of `characters=`, such as an orbital out of range or a list of the
  wrong length, raises `ValueError` before anything is measured. From `decompose`,
  `symmetry_indicator` and `srsi_of_bands`, the symmetry data vector holds labels that
  no elementary band representation of the group holds. From `regular_completion`, one copy of the regular
  band representation, computed from the group's characters, holds a label that no
  elementary band representation holds.
- **Usual cause.** A word of another site group, such as `B1` at a site of symmetry `3m`;
  for generator characters or matrices, a name such as `m`, which at a `4mm` site
  matches four mirrors in two classes; for a symmetry data vector, one labeled in
  another group, or a mistyped label.
- **What to do.** For a declaration, the reason prints the frame the site group's
  operations are named in, and for an unknown word also the site group and its valid
  words; it says what to declare. A word the cell's group has no name for is asked again
  on the group the Hamiltonian settles to, and refused only where nothing on the way
  names it ([saying what the orbitals are](../guide/orbitals.md#when-it-refuses)). For a
  symmetry data vector, pass one read in this group. Met from `regular_completion`, it is
  the package disagreeing with itself: report it, with the model and the call.

#### `ambiguous_multiplet`

Returned by `realized_group`.

- **What it means.** A declaration does not say which orbitals hold a site irrep, or
  gives generator matrices that do not stand on whole multiplets the hoppings measure. A
  key of several orbitals names the site irreps those orbitals hold together, so the
  orbitals its word holds are as many as the key names. The orbitals of one site
  declared with a word wider than one orbital add up to whole copies of it; where they do
  not, which orbitals hold it is not said. Generator matrices fix the basis of the site
  irrep a measured multiplet carries, so they are refused on part of a multiplet, mixing
  two multiplets, or at a site whose declarations are read as one sum over its multiplets.
  A site's declarations are read as one sum where they do not each name one whole
  multiplet, or where one of them is a double-valued site irrep at a site of several
  multiplets. Orbitals of two sites written as one key raise `ValueError` before anything
  is measured.
- **Usual cause.** `{(0, 1): "A"}` written for two `A` orbitals, whose spelling is
  `{0: "A", 1: "A"}`; a two-dimensional irrep declared on one orbital.
- **What to do.** The reason gives the spelling that says it, or, where the site holds too
  few orbitals for the site irrep, the words that fit it.

#### `spin_pair_split`

Returned by `realized_group`.

- **What it means.** The two members of a declared spin pair sit at different centers,
  where the two spin states of one Wannier function share one. Either the pairing is
  mislabeled, or the orbitals each carry one spin state with no partner at their own
  center, a magnetic configuration outside the package's present scope
  ([an unpaired spinor orbital](../background/limits.md#an-unpaired-spinor-orbital)).
  Two centers apart only by the numerical noise of a fitted model are neither: a declared
  precision (`precision=`) reads two centers closer than it as one. `realized_group` reads
  it off the orbitals' centers alone, the orbitals at one center being one site at the
  tolerance the cell's group is found at, and refuses before anything is identified or
  measured, so it names no group; `value` is the distance between the pair's two sites
  and `context["positions"]` says where they are.
- **Usual cause.** The spin states declared in the wrong order: `spin="blocked"` and
  `spin="interleaved"` differ by exactly this.
- **What to do.** Read the order of the spin states off the Wannier90 files and declare
  it, with explicit pairs where the orbitals are ordered irregularly; where the two
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
  ([conventions](../background/conventions.md#operations-a-seitz-symbol-and-a-cartesian-name)).
  `R` is the rotation part: `1`, `-1`, `m`, `2`, `3`, `4`, `6`, `-3`, `-4` or `-6`. `3`,
  `4`, `6` and their negatives carry `+` or `-` for the sense of the turn. Every rotation
  part but `1` and `-1` is followed by an underscore and the direction of its axis, or of
  a mirror's normal. `t` is the translation, three fractions. A trailing `'` marks an
  operation combined with time reversal, as in `{-4+_001|0,0,1/2}'`.

#### `kpoint_not_tabulated`

Returned by `coreps_at`, `corep_table` and `pythtqc.groups.identify_from_characters`.

- **What it means.** The k-point is not one of the group's tabulated high-symmetry points in
  this cell, or a name the tables do not hold. A point in the star of a tabulated one, but
  not the tabulated member, has a conjugate little group and is not carried here.
- **What to do.** Use a point `high_symmetry_points(group)` returns; the reason lists the
  tabulated ones in this cell's coordinates.

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

Returned by `characters`, `coreps_at` and `sewing_matrix`; `hsp_table` keeps it
for each point where the states handed in do not solve `H(k)`.

- **What it means.** The energies and states handed in do not diagonalize `H(k)` at the
  k-point asked. Characters read off states from another point, or another model, come
  out wrong without any sign: the sewing matrix built on them can be unitary all the same.
- **What to do.** Hand in the states of one diagonalization at that point, or the model, and
  let the call solve there ([one diagonalization, reused](../guide/labels.md#one-diagonalization-reused)).

#### `k_not_on_mesh`

Returned by `characters`, `coreps_at` and `sewing_matrix`; `hsp_table` keeps it
for each point its mesh does not contain.

- **What it means.** A mesh of states handed in does not contain the k-point asked.
  Handed a mesh, the call reads its states at the point asked and does not diagonalize
  there. The states at the nearest point of the mesh are not those at `k`.
- **What to do.** Build the mesh to contain the point (`k_grid` reports which high-symmetry
  points a mesh holds), or hand in the model.

#### `asymmetric_grid`

Returned by `k_grid`.

- **What it means.** The mesh asked for is not symmetric under the group, so quantities read
  off it double-count symmetry-related points.
- **What to do.** Ask a mesh the group leaves invariant, or pass `allow_asymmetric=True`
  deliberately.

#### `negative_multiplicity`

Returned by `content_of`, `decompose`, `symmetry_indicator`, `subtract` and
`regular_completion`, each when a symmetry data vector handed in has a negative
multiplicity. `subtract` returns it also when the difference has one, and
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
- **What to do.** Pass nonnegative multiplicities. For `subtract`, pass second a set of
  bands contained in the first. For `regular_completion`, pass a larger multiple of the
  number of bands in one copy to `bands=`; where the reason says the copies hold none of
  some names, check those names in the symmetry data vector.

#### `non_integer_multiplicity`

Returned by `content_of`, `decompose`, `symmetry_indicator`, `subtract` and
`regular_completion`, each when a symmetry data vector handed in has a multiplicity that is
not an integer.

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

Each check here measures an identity that holds in exact arithmetic, or two measurements of
one thing that must agree. Where a call is handed arguments made by hand, a few say
something about those arguments, as each entry notes; met from `realized_group` and the
calls on its record, they mean the package has met a case it does not handle. Nothing in
the model is to change, and the refusal, the model and the call are worth reporting.

### The representation

#### `rep_not_covariant`

Returned by `realized_group`.

- **What it means.** No candidate representation of the operations the Hamiltonian was
  measured to obey passes the covariance check. Reached once `realized_group` has
  identified the group, it is the package's check on itself: the operations were measured
  to hold, and no representation of them was built.
- **Usual cause.** Multiplets stated with `site_multiplets=` that split a site whose
  orbitals mix under its site-symmetry group.
- **When it has a route.** Close to the tolerance: each operation is obeyed within the
  tolerance on its own, the worst to `context["obeyed_alone"]`, while the representation
  built for the group carries `H(k)` onto `H(Rk)` only to `value`, beyond it, so the
  Hamiltonian's asymmetry is close to the tolerance. A declared `precision=` sets the
  tolerance to `10·precision·max|H|`, `max|H|` as above. Declare one whose tolerance lies
  above these numbers if the asymmetry is numerical noise, and the Hamiltonian is made
  symmetric under the group, or below them if it is a term of the model to keep, and the
  operations it breaks are dropped. Operations the reason calls near-symmetric can also
  hold on parts of the crystal the measurement reads apart while the hoppings couple them
  only below the tolerance, as in a numerical (Wannier90) Hamiltonian.
- **Otherwise.** Report it, with the model and the call.

#### `rep_internal_factor`

Returned by `realized_group`.

- **What it means.** Candidate matrices carry `H(k)` onto `H(Rk)` as the group requires,
  but on none of them does a site's block take the form a representation of these
  operations has, a character times the group's own factor for the operation, its spin
  rotation where there is spin: they are some other operations' matrices. With no site
  character to read off them, no orbital can be declared or labeled.
- **What to do.** Where the multiplets were stated (`site_multiplets=`), the statement;
  otherwise report it.

#### `rep_not_multiplicative`

Returned by `realized_group`.

- **What it means.** Candidate matrices pass the covariance and site-matrix checks, but none
  reproduces the group's multiplication table. Either they form a projective
  representation of a class the table does not carry, or a conserved pseudospin admits no
  representation of the group on its copies. On a pseudospin's copies, every
  operation's matrix is the tensor product of the factor the hoppings fix with a matrix
  on the copies. The copies are held in sets, which the hoppings do not couple to one
  another and the operations carry onto one another. They carry a representation only
  where a sum of irreducible representations, of their dimension, of the unitary
  operations that map one set onto itself makes up the factor system the fixed factor
  leaves them, time reversal's square included.
  Where the reason names neither a projective class nor a pseudospin, the operations,
  the generators or the blocks on the multiplets are wrong.
- **When it has a route.** Spinor states held in a spinless model's orbital index carry
  the sign of spin ½ (`context["spin_half"]`): declare `spin=`, and they are read in the
  double group. `realized_group` measures that class before it builds, and descends to the
  largest subgroup that carries an ordinary representation.
- **Otherwise.** Report it.

#### `rep_not_unique`

Returned by `realized_group`.

- **What it means.** More than one distinct representation passes every check, so the
  choice the measurement should fix is not fixed: the k-points the candidates are tested at
  are too few, or the model carries more symmetry than the group. Two survivors that are
  the two orientation branches, `D(g)` and `D(g^-1)`, are settled by convention where both
  give the same site labels under the same labeling; where they do not, they are two
  different representations.
- **What to do.** Open an issue on pythTQC's GitHub with the model and the call: the
  measurement should have fixed this choice.

#### `canonical_norm`

Returned by `realized_group` and `pythtqc.symmetry.normalize_rep`.

- **What it means.** A matrix `D` raised to its operation's order is not a multiple of
  the identity, so it represents no operation and no phase can fix it. The check reads the
  unitary matrix nearest `D`, raised to the order of the operation modulo lattice
  translations, with every second factor complex conjugated for an antiunitary operation,
  and compares it with the multiple of the identity nearest it.
- **What to do.** A matrix of one's own handed in is not the operation's; met any other way,
  report it.

#### `antiunitary_anchor`

Returned by `realized_group` and `pythtqc.symmetry.normalize_rep`.

- **What it means.** No phase the canonical norm allows makes a matrix `D` agree with time
  reversal, `T conj(D) T† = D` for `T` the matrix of time reversal, or of another
  antiunitary operation where the group has none: the matrix and the time reversal belong
  to different representations, and corep labels built on it would be meaningless. Where `T conj(D) T† = ρD`, the matrix `λD`, for a phase `λ`, agrees
  with time reversal only for `λ**2 = ρ`, and no such `λ` is among the phases the
  canonical norm allows.
- **What to do.** As for `canonical_norm`.

#### `little_group_mismatch`

Returned by `coreps_at` and `pythtqc.groups.identify_from_characters`.

- **What it means.** The operations of the little group measured at a point do not match
  the tables' list for it, so no label would be meaningful.
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

### The projection, and the tables' setting

#### `symmetrize_defect_exceeds_tol`

Returned by `realized_group`.

- **What it means.** The group average moved `H` by more than the declared `tol`. Over
  operations obeyed within `tol` the move is of the order of their residuals, so a larger
  one says the matrices averaged with are not the symmetry the residuals were measured
  for. A miss of the centers by `δ` costs `H(k)` a covariance of order `2π·δ·max|H|`
  whatever the hoppings do.
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
  precision, as for `symmetrize_defect_exceeds_tol`; otherwise report it, with the model
  and the call.

#### `wyckoff_setting`

Returned by `wyckoff_sites_tables`, `srsi_of_model` and `splitting`.

- **What it means.** The list of Wyckoff positions read is not written in the setting the
  tables are, or the Wyckoff position of an orbit read from its centers differs from the
  one at which its measured symmetry data vector is a sum of elementary band
  representations, which says the positions are read in a setting other than the tables'.
  Every Wyckoff position returned would be read in the wrong setting, so none is, and no
  stable real-space invariant is returned.
- **What to do.** Report it.

## Two checks no returned group reaches

Two checks complete the list, and no group a call returns reaches either.

### `lattice_not_descended`

A group whose lattice is a supergroup of the model's, as a supercell's would be, handed to
a call that reads the tables. Its tables are written on the finer lattice, so every label
read from them would name a k-point of that lattice's Brillouin zone, folded onto the
model's. `realized_group` never returns one: it refuses `supercell`, or descends to the
model's own lattice.

### `magnetic_type_out_of_scope`

A magnetic group of type III or IV handed to `symmetry_indicator`, `regular_content` or
`regular_completion`. The package is validated for types I and II, and `realized_group`
returns no other: a Hamiltonian that breaks time reversal is returned the unitary
subgroup of its magnetic group
([the magnetic scope](../background/limits.md#magnetic-groups-of-types-i-and-ii-only)).
