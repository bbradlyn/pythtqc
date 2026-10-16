# Refusals

Every refusal a call can return, by the name of its check: what it means, its usual cause,
and what to do. A refusal is a returned, falsy `Refusal` record that carries the check that
stopped the call, a one-sentence reason, the value the check measured against its
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
the cell has at that point, an upper bound on what the Hamiltonian obeys. A disagreement
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
  no table names, and every label would name the wrong k-points. It is one of the two
  cases in which the group question is refused
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
  Folding a supercell onto its primitive cell is not supported yet
  ([what the package cannot do yet](../background/limits.md#a-supercell-is-refused-not-folded)).

#### `positions_not_exactly_symmetric`

Returned by `realized_group` and `symmetrize`.

- **What it means.** The orbital centres miss the cell's own operations by a defect `δ`,
  in reduced coordinates, that alone costs `H(k)` a covariance residual of order
  `2π·δ·max|H|` above the tolerance, so which operations the Hamiltonian obeys would be
  read off the geometry as much as off the hoppings. Nothing is built. It is the other
  case in which the group question is refused, and the reason names the cell's group as
  an upper bound. Centres missed by less, where an operation reads near-symmetric, are
  refused the same way by the projection of the hoppings, by `symmetrize` itself and
  through `realized_group`: the projection's operators carry each centre onto its image
  to `1e-9`, and no projection of the hoppings reaches a defect in the centres.
- **Usual cause.** A numerical model, such as a wannier90 import, asked with no precision
  declared.
- **What to do.** Declare the model's precision, `precision=`, or `position_tol=` for the
  centres alone: centres within it are moved onto the positions the group puts them at, a
  gauge transformation that leaves the spectrum untouched. `value` and
  `context["position_defect"]` are `δ`
  ([the centres first](../guide/numerical.md#the-centres-first-because-moving-them-is-a-gauge)).
  Centres noisier than the symmetry finder's `symprec` are not refused this way: the
  finder reads them as a less symmetric set, and the group returned can be as low as `P1`
  ([centres noisier than symprec](../guide/numerical.md#centres-noisier-than-symprec)).

#### `cell_group_not_realized`

Returned by `realized_group` with `descend=False`.

- **What it means.** The Hamiltonian breaks operations the cell allows, so the cell's
  group, the upper bound the positions set, is not its group. The reason counts the
  operations obeyed and broken and says what broke each: no symmetry matrix at all, which
  is a proof; the tensor structure an operation's matrix must have; or, for spinor states
  held in a spinless model's orbital index, two commuting operations whose matrices
  anticommute there.
- **Usual cause.** `descend=False`, which asks for the upper bound, on a Hamiltonian with
  less symmetry than its cell.
- **What to do.** Leave `descend` at its default, and the cell is lowered to a group of
  the operations that hold; or read `context["validated"]`, the verdict and residual on
  every operation, and `context["broken"]`.

#### `declaration_breaks_group`

Returned by `realized_group` with `descend=False`.

- **What it means.** The declared orbital content is not covariant under some operations
  of the group. A declaration states what the orbitals are, so this is a crystal with
  less symmetry than the group.
- **Usual cause.** `characters=` stating content the hoppings do not carry under every
  operation, such as both of `lin3g`'s orbitals at `1a` declared `s`, which the positions
  allow and the Hamiltonian does not.
- **What to do.** Let the group descend: `realized_group` at its default returns the
  subgroup the content is covariant on, each broken operation under
  `refuted_by = "declaration"`. `context["subgroup"]` names that subgroup and
  `context["broken"]` the operations broken, with why
  ([a declaration that lowers the group](../guide/orbitals.md#a-declaration-that-lowers-the-group)).

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
- **Usual cause.** A centre far off every orbit: a position written in another cell or
  origin, or mistyped.
- **What to do.** Check the position; the reason names the nearest Wyckoff position and
  how far it is.

#### `operations_not_consistent`

Returned by `realized_group` where no precision is declared for the centres. Under
`precision=` or `position_tol=` they are made exact before the cell is identified, and the
operations read off exact centres close, unless the centres cannot be read as a symmetric
set at all, as when two sites of one species sit closer together than the symmetry finder
can tell apart, so that an operation does not carry the sites onto one another one for one.

- **What it means.** The operations the symmetry finder reads off the centres do not close
  into a group, and the cell could not be matched to the setting of the tables, from which
  exact operations would be rebuilt. Every check after it would be measured against a set
  that is not a group.
- **Usual cause.** Numerically determined centres, whose noise gives each operation its own
  share of error, in a cell the tables' setting search does not match, with no precision
  declared.
- **What to do.** Declare the model's precision, `precision=`, or `position_tol=` for the
  centres alone: the centres are then made exact before the setting is read, and the
  operations close. Or hand in a cell whose centres carry less noise.

#### `operations_not_nearly_consistent`

Returned by `realized_group` where `operations_not_consistent` is: with no precision declared
for the centres, or where they cannot be read as a symmetric set at all.

- **What it means.** The operations do not close, and rebuilding their translations from
  the tables' own would move one by more than `symprec`, a length in the model's units: the
  tabulated group is not the one these centres carry at this tolerance.
- **Usual cause.** Centres noisier than `symprec` allows for, or a `symprec` too large for
  the cell, with no precision declared.
- **What to do.** Declare the model's precision, `precision=`, which makes the centres exact
  before the setting is read; identify again at a `symprec` the centres support; or state
  what the crystal is with `typat=` or `decorations=`.

#### `positions_not_nearly_symmetric`

Returned by `symmetrize_positions`.

- **What it means.** An orbital centre would have to move farther than the bound onto the
  nearest positions the group carries exactly, the origin fitted to the centres:
  `max_move`, `1e-5` in reduced coordinates by default.
  Centres that far off are a different set of points.
  `realized_group` does not refuse such centres under a declared precision: the
  operations that would move one further than the declaration are broken by the centres,
  and the group descends to the largest group of its operations that carries them. It
  refuses only orbitals read as one site that sit further apart than the declaration,
  which no group of operations brings together.
- **Usual cause.** A group the centres do not have, or a displacement beyond the noise.
- **What to do.** Identify the group the centres carry (`typat=`, `decorations=`), or raise
  `max_move` where the displacement is one that can be defended.

#### `symmetrize_broken_operation`

Returned by `symmetrize`.

- **What it means.** An operation handed to the projection is broken at the declared `tol`:
  averaging over it would impose a symmetry the model does not have.
- **Usual cause.** A record of operations measured at a looser tolerance than the one
  `symmetrize` is given; `realized_group` averages only over the operations it kept.
- **What to do.** Hand `symmetrize` a record measured at the same `tol`. A defect that is
  numerical wants a larger `tol`, read against `max|H|`; otherwise the operation is broken
  and the group is smaller
  ([a broken operation](../guide/numerical.md#a-broken-operation-is-never-symmetrized-into)).

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
  half-integer content is the electron's own. An unpaired spinor orbital is not supported
  ([what the package cannot do yet](../background/limits.md#an-unpaired-spinor-orbital)).

#### `single_valued_spinful`

Returned by `realized_group`.

- **What it means.** `spinor=False` was asked of a spinful model. Its states carry the
  electron's spin ½, so their content is double-valued and there is no single-valued
  question to put to them.
- **Usual cause.** `spinor=False` on a model built with spin.
- **What to do.** Leave `spinor` at its default, which reads it off the model.

### What the package covers

#### `fewer_than_two_periodic_directions`

Returned by `realized_group`.

- **What it means.** The model is periodic in fewer than two directions, whatever the
  dimension of the cell it is written in. With none it is a molecule, whose symmetry is a
  point group; with one it is a chain or a ribbon, whose symmetry is a rod or frieze
  group. The package reads crystals periodic in three directions and layers periodic in
  two, whose symmetry is a space group, and neither of these is in its scope yet. The
  refusal comes before anything is measured, so it names no group; `value` is the number
  of periodic directions and `context["periodic_dirs"]` says which they are.
- **Usual cause.** pythtb's `ssh` chain; a ribbon or a rod cut from a crystal with
  `cut_piece`; a cluster with every periodic direction switched off.
- **What to do.** Nothing here yet
  ([fewer than two periodic directions](../background/limits.md#fewer-than-two-periodic-directions)).

#### `srsi_scope`

Returned by `srsi_tables`, `srsi_of`, `srsi_of_columns`, `srsi_of_bands`,
`srsi_of_model`, `splitting`, `stably_equivalent_pairs`,
`pythtqc.srsi.name_correspondence` and `pythtqc.srsi.canonical_determinism`.

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

Returned by `characters`, `corep_content` and `sewing_matrix`; `hsp_table` keeps it
for each point it refuses at.

- **What it means.** The set of bands asked for cuts a degenerate multiplet at `k`, so its
  characters would depend on the basis the diagonalization chose inside the multiplet.
- **Usual cause.** A band count that ends inside a degeneracy: half a Kramers pair, or bands
  that meet at the point, as a semimetal's do.
- **What to do.** Label the whole multiplet, or the isolated set of bands:
  `context["multiplet"]` lists the bands that are degenerate there
  ([label an isolated set of bands](../guide/labels.md#label-an-isolated-set-of-bands)).

#### `integer_multiplicity`

Returned by `corep_content`.

- **What it means.** The multiplicities of the coreps fitted to the characters at a point are
  not non-negative integers within `label_tol`, `1e-3` by default.
- **Usual cause.** A degeneracy split into several multiplets by numerical noise beyond
  `degeneracy_tol`: a two-dimensional corep split in two reads exactly 0.5. Levels merged
  into one multiplet never refuse; they are labelled as a sum of coreps.
- **What to do.** On a numerical model, symmetrize it or set `degeneracy_tol`, an energy, to
  the noise its degeneracies carry; `context["rows"]` names the multiplets at fault and
  `context["multiplicities"]` the fitted numbers.

#### `multiplet_dimension`

Returned by `corep_content`, through its identification of the coreps.

- **What it means.** The coreps identified from a multiplet's characters do not add up to
  the multiplet's degeneracy, so the grouping of levels and the characters disagree.
- **Usual cause.** Characters that are no representation's, whose fitted multiplicities
  still round to integers. Merged levels are a representation's and fill their multiplet,
  and a degeneracy split in two is refused `integer_multiplicity` first.
- **What to do.** As for `integer_multiplicity`. On a model built from exact numbers with
  well separated levels, report it.

### Atomic limits and indicators

#### `incomplete_content`

Returned by `decompose`, `symmetry_indicator`, `completeness_check`, `splitting` and
`srsi_of_bands`.

- **What it means.** The symmetry vector has no label at some high-symmetry point, because
  the table refused there, and a decomposition or a class of the rest would be a verdict on
  a different vector.
- **Usual cause.** A set of bands that cuts a multiplet at some point, or another refusal
  recorded in the table at that point.
- **What to do.** Read the table's refusals, which `context["refusals"]` repeats, and fix
  the set of bands. `decompose` takes `allow_incomplete=True` for a point whose tables
  cannot be read, never for one where the physics refused, and its verdict is then a
  necessary condition only; `symmetry_indicator` has no such option.

#### `not_a_band_structure`

Returned by `decompose`, `symmetry_indicator`, `completeness_check`,
`regular_completion`, `splitting` and `srsi_of_bands`.

- **What it means.** The content violates the group's compatibility relations: coordinates
  of its Smith form that vanish for every band structure of the group do not, so it lies
  outside the lattice of band structures, with no class in `{BS}/{AI}` and no
  decomposition
  ([how a symmetry indicator is computed](../background/tqc-primer.md#how-a-symmetry-indicator-is-computed)).
- **Usual cause.** Bands asked for that do not make a full set at every `k`, as when they
  cut through a crossing between two high-symmetry points, or labels read in another group.
- **What to do.** Check that the set of bands is isolated everywhere and that its labels
  came from this group; `context["violations"]` lists the coordinates that do not vanish.

#### `indicator_not_evaluable`

Returned by `symmetry_indicator` and `srsi_of_bands`.

- **What it means.** A layer's content does not reach some rows of the group's elementary
  band representations, rows at k-points out of the plane that repeat no row in it, and the
  decoupled stack of the layer, on which such rows are measured, did not supply them. The
  indicator has no value.
- **Usual cause.** A layer's indicator asked of counts alone, without the representation
  the stack is built from, or with `points=` given; or a check of the stack that did not
  complete, which `context["stack"]` names.
- **What to do.** Hand `symmetry_indicator` the `hsp_table` record and `rep=`, and leave
  `points=` alone ([a layer's class](../guide/indicators.md#a-layers-class-is-a-three-dimensional-class)).

#### `restriction_not_exact`

Returned by `in_plane_restriction`, `decompose` and `completeness_check`.

- **What it means.** Rows of the group's elementary band representations that the model
  does not measure repeat no row it does, so they impose constraints of their own and no
  verdict from the measured rows is exact for this group.
- **Usual cause.** `points=` naming fewer points than the model samples, or
  `in_plane_restriction` asked of a free-standing layer's group over every band
  representation. `decompose` reads a layer over its own band representations, those of
  the Wyckoff positions within it, on which every row out of its plane repeats one in it.
- **What to do.** Measure the content at every point the model samples; for a layer's
  class, the indicator measures the rows out of its plane on the layer's decoupled stack
  ([symmetry indicators](../guide/indicators.md)).

### Stable real-space invariants

#### `srsi_not_atomic`

Returned by `atomic_realizations`, and by `nsit_criterion` and `splitting` through it.

- **What it means.** `theta_Z`, the ℤ-valued stable real-space invariants of the bands
  ([the stable ones](../guide/srsi.md#the-stable-ones)), has a fractional entry, so the bands
  carry a nonzero symmetry indicator and no atomic insulator has their symmetry data: there
  are no atomic realizations to enumerate.
- **Usual cause.** Bands that are stably topological.
- **What to do.** Nothing: it is the verdict, and a stronger one than an empty list of
  realizations. `splitting` reads a fractional `theta_Z` on the valence side as its own
  verdict, `stable topological`, and returns that record rather than this refusal
  ([the no-go criterion](../guide/srsi.md#the-no-go-criterion)).

#### `orbit_content_incomplete`

Returned by `srsi_of_model` and `splitting`.

- **What it means.** An orbit's content could not be identified at a high-symmetry point,
  for a reason other than a trivial little group, so its symmetry content is incomplete.
- **Usual cause.** A refusal at that point, which `context["refusal"]` carries.
- **What to do.** Read that refusal; its entry here says what to do.

#### `orbit_not_localizable`

Returned by `srsi_of_model` and `splitting`.

- **What it means.** An orbit's measured content is no band representation of any
  Wyckoff position on the closure of its own, the points its free coordinates sweep and the
  higher-symmetry positions among them, nor a composite site irrep the tables carry: the
  orbit's content is not an atomic limit at the position it sits on.
- **Usual cause.** Orbitals whose content does not fit where they sit.
- **What to do.** Check the orbitals' positions and their declared content;
  `context["tried"]` lists the positions tried.

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
- **Usual cause.** Layers coupled across the direction they would be stacked in.
- **What to do.** Check the model's hoppings along its non-periodic direction.

## A practical limit

### Searches and memory

#### `rep_search_too_large`

Returned by `realized_group` and `symmetrize`.

- **What it means.** The representation's discrete choices, which the measurement does not
  fix, had to be enumerated, and there were more than `max_candidates` of them: a limit on
  the search.
- **Usual cause.** A conserved operator, or accidental degeneracy, inside a multiplet of
  several orbitals with nothing declared, so the multiplet's block is fixed only up to the
  conserved part.
- **What to do.** Declare what the orbitals are (`characters=`), which states what the
  search would otherwise choose among; or run the same call with a larger
  `max_candidates=`, as in `realized_group(..., max_candidates=...)`.
  `context["fallback_reason"]`, where present, says why the choices could not be read off
  the measurement, and is the thing to act on first.

#### `detector_too_large`

Returned by `realized_group` and `symmetrize`.

- **What it means.** Measuring an operation at one k-point needs more memory than the
  budget allows, because the spectrum's exactly degenerate pairs alone exceed it: a limit
  on memory. Narrowing the measurement to fit would drop
  exactly degenerate pairs and return a wrong solution rather than a coarse one.
- **Usual cause.** A large model with exactly degenerate levels, such as a big supercell's
  folded bands.
- **What to do.** The same call with the `max_bytes=` the reason names. It fires while
  the operations are measured, so the group it names is the cell's at that point, an upper
  bound.

#### `projection_too_large`

Returned by `realized_group` and `symmetrize`.

- **What it means.** The projection onto the group-invariant part needs more memory at its
  peak than `max_bytes` allows. Its grid is set by the hoppings' range, and a coarser grid
  would alias, so the projection is not run on one: a limit on memory.
- **Usual cause.** A numerical model with many states and long-ranged hoppings.
- **What to do.** The same call with the `max_bytes=` the reason names
  ([a large model](../guide/numerical.md#a-large-model)).

#### `srsi_realizations_truncated`

Returned by `atomic_realizations`, and by `nsit_criterion` and `splitting` through it.

- **What it means.** The enumeration of atomic realizations of `theta_Z` reached its cap. A
  no-go is asserted only on a complete enumeration, so with the list truncated neither
  "no pair adds up" nor "none exists" may be claimed.
- **Usual cause.** Many bands, or a group with many band representations.
- **What to do.** Only `atomic_realizations` takes `cap=`, `256` by default: called
  directly, raise it. `nsit_criterion` and `splitting` take no `cap=` and enumerate at
  that default, so their refusal has no argument to raise; `atomic_realizations` with a
  larger `cap=`, handed the group's tables and the side's `theta_Z`, which
  `context["theta_Z"]` carries, enumerates that side's realizations.

### The tables, as installed

#### `irrep_tables`

Returned by `realized_group`, `ebr_data`, `ebr_rows`, `decompose`, `symmetry_indicator`,
`in_plane_restriction`, `srsi_tables` and `wyckoff_sites_tables`.

- **What it means.** The character tables were not loaded, so no label can be read; or the
  tables carry no irrep table for this group, so its band representations cannot be carried
  into this package's convention.
- **Usual cause.** `irreptables` not installed, or imported before `irrep` in the process,
  which leaves `irrep` with a stand-in that labels nothing.
- **What to do.** `tables_status()` says which: install `irreptables`, or import `pythtqc`
  before `irreptables`.

#### `tables_not_ready`

Returned by `operations`, `seitz`, `operation_index`, `high_symmetry_points`,
`corep_table`, `corep_content`, `hsp_table`, `wyckoff_sites_tables`,
`in_plane_restriction` and `srsi_of_model`.

- **What it means.** The group was identified, but its cell could not be matched to the
  setting the tables are written in, so no table can be read for it. The reason carries
  `irrep`'s own message.
- **Usual cause.** A cell the search for the tables' setting does not match.
- **What to do.** The group stands; the labels need the tables. Report the model, with the
  reason.

#### `ebr_tables`

Returned by `ebr_data`, `ebr_rows`, `decompose`, `symmetry_indicator`,
`in_plane_restriction`, `srsi_tables`, `srsi_of_model` and `wyckoff_sites_tables`.

- **What it means.** The installed tables carry no elementary band representations for this
  group and valuedness.
- **Usual cause.** A group or valuedness the tables do not cover.
- **What to do.** None in the model; the group and the labels stand without them.

#### `ebr_convention`

Returned by the same calls as `ebr_tables`.

- **What it means.** The shipped rows of this group's band representations cannot be
  carried out of the tables' phase convention into this package's
  ([conventions](../background/conventions.md)).
- **Usual cause.** A defect in the tabulation itself.
- **What to do.** None in the model; report it.

#### `ortools`

Returned by `pythtqc.groups.decompose_ebrs`.

- **What it means.** The band set is an integer combination of elementary band
  representations, and whether a combination with no negative coefficient exists, which
  tells a decomposable band set from a fragile one, needs the integer search of `ortools`,
  which is not importable.
- **Usual cause.** An installation without `ortools`, a dependency of the package.
- **What to do.** Install `ortools` 9.10 or later.

#### `srsi_data`

Returned by the same calls as [`srsi_scope`](#srsi_scope).

- **What it means.** The converted tables of stable real-space invariants, two data files
  the package ships, are not where it reads them.
- **Usual cause.** An installation whose package data is missing a file.
- **What to do.** `pythtqc.srsi.srsi_status()` says which file is missing and where it
  belongs. Reinstall the package, or convert the tables again from the paper's
  supplementary material with `scripts/convert_srsi_tables.py`, as the reason says.

#### `srsi_ebr_basis`

Returned by the same calls as `srsi_data`.

- **What it means.** The installed `irreptables` lists this group's elementary band
  representations differently from the list the converted tables were resolved against, so
  their stored columns would name the wrong ones.
- **Usual cause.** An `irreptables` installed after the tables were converted.
- **What to do.** Convert the tables again against the installed `irreptables`.

#### `irreptables_indicator_tables`

Returned by `pythtqc.srsi.name_correspondence`.

- **What it means.** The indicator table of `irreptables` could not be read, so the
  paper's names of the indicator components cannot be paired with `irreptables`' own.
- **Usual cause.** An `irreptables` installation whose indicator data cannot be read.
- **What to do.** Check the `irreptables` installation.

#### `wyckoff_enumeration`

Returned by `pythtqc.groups.wyckoff_positions_spglib`.

- **What it means.** The Wyckoff positions of the group could not be enumerated; the reason
  quotes `irrep`'s own message.
- **What to do.** None in the model; report it, with the reason.

## A grammar error in a declaration or a question

### In `characters=` and `spin=`

#### `unknown_label`

Returned by `realized_group`, `decompose`, `symmetry_indicator`, `completeness_check` and
`regular_completion`.

- **What it means.** A declaration names nothing at the orbital's site: a word that is not
  an irrep of its site group, a sum of irreps, a shell or a shorthand, or an operation the
  site group does not hold. A declaration outside the grammar of `characters=`, such as an
  orbital out of range or a list of the wrong length, raises `ValueError` before anything
  is measured. In `decompose` and `symmetry_indicator`, measured labels outside the
  group's elementary band representations, the content and the basis not written in one
  alphabet.
- **Usual cause.** A word of another site group, such as `B1` at a site of symmetry `3m`;
  in a decomposition, content labelled in another group.
- **What to do.** The reason prints the site group, its valid words and the frame its
  operations are named in; declare one of them. A word the cell's group has no name for is
  asked again on the group the Hamiltonian settles to, and refused only where nothing on
  the way names it ([saying what the orbitals are](../guide/orbitals.md#when-it-refuses)).

#### `ambiguous_multiplet`

Returned by `realized_group`.

- **What it means.** A declaration does not say which orbitals hold a content: a joint
  form whose irrep is not as wide as the orbitals it names, or a word wider than one
  orbital written on orbitals that make no whole number of copies of it. Orbitals of two
  sites written as one key raise `ValueError` before anything is measured.
- **Usual cause.** `{(0, 1): "A"}` written for two `A` orbitals, whose spelling is
  `{0: "A", 1: "A"}`; a two-dimensional irrep declared on one orbital.
- **What to do.** The reason gives the spelling that says it, or, where the site holds too
  few orbitals for the content, the contents that fit it.

#### `spin_pair_split`

Returned by `realized_group`.

- **What it means.** The two members of a declared spin pair sit at different centres,
  where the two spin states of one Wannier function share one. Either the pairing is
  mislabelled, or the orbitals each carry one spin state with no partner at their own
  centre, a magnetic configuration outside the package's present scope
  ([an unpaired spinor orbital](../background/limits.md#an-unpaired-spinor-orbital)).
  Two centres apart only by the numerical noise of a fitted model are neither: a declared
  precision (`precision=`) reads two centres closer than it as one. `realized_group` reads
  it off the orbitals' centres alone, the orbitals at one centre being one site at the
  tolerance the cell's group is found at, and refuses before anything is identified or
  measured, so it names no group; `value` is the distance between the pair's two sites
  and `context["positions"]` says where they are.
- **Usual cause.** The spin states declared in the wrong order: `spin="blocked"` and
  `spin="interleaved"` differ by exactly this.
- **What to do.** Read the order of the spin states off the wannier90 files and declare
  it, with explicit pairs where the orbitals are ordered irregularly; where the two
  centres differ by a fitted model's noise, declare its precision.

### In the arguments of a call

#### `unknown_operation`

Returned by `operation_index`.

- **What it means.** No operation of the group is written with that Seitz symbol.
- **What to do.** The reason lists the nearest spellings the group carries and the grammar
  of a symbol, such as `{-4+_001|0,0,1/2}`; `operations(group)` prints every symbol against
  its index ([conventions](../background/conventions.md#operations-a-seitz-symbol-and-a-cartesian-name)).

#### `kpoint_not_tabulated`

Returned by `corep_content` and `corep_table`.

- **What it means.** The k-point is not one of the group's tabulated high-symmetry points in
  this cell, or a name the tables do not hold. A point in the star of a tabulated one, but
  not the tabulated member, has a conjugate little group and is not carried here.
- **What to do.** Use a point `high_symmetry_points(group)` returns; the reason lists the
  tabulated ones in this cell's coordinates.

#### `out_of_plane`

Returned by `little_group` and `corep_table`.

- **What it means.** The k-point has a component out of a layer's periodic plane, where the
  model has no dispersion.
- **What to do.** Ask an in-plane point.

#### `little_group`

Returned by `sewing_matrix` and `SpaceGroupRep.sewing_operator`.

- **What it means.** The operation carries `k` to a point that is not `k` up to a reciprocal
  lattice vector, so it is not in the little group of `k` and has no sewing matrix there.
- **What to do.** Ask an operation of the little group, which `little_group` lists.

#### `states_not_at_k`

Returned by `characters`, `corep_content` and `sewing_matrix`.

- **What it means.** The energies and states handed in do not diagonalize `H(k)` at the
  k-point asked. Characters read off states from another point, or another model, come
  out wrong without any sign.
- **What to do.** Hand in the states of one diagonalization at that point, or the model, and
  let the call solve there ([one diagonalization, reused](../guide/labels.md#one-diagonalization-reused)).

#### `k_not_on_mesh`

Returned by `characters`, `corep_content` and `sewing_matrix`.

- **What it means.** A mesh of states handed in does not contain the k-point asked, and a
  second diagonalization there would corrupt the character.
- **What to do.** Build the mesh to contain the point (`k_grid` reports which high-symmetry
  points a mesh hits), or hand in the model.

#### `asymmetric_grid`

Returned by `k_grid`.

- **What it means.** The mesh asked for is not symmetric under the group, so quantities read
  off it double-count symmetry-related points.
- **What to do.** Ask a mesh the group leaves invariant, or pass `allow_asymmetric=True`
  deliberately.

#### `negative_multiplicity`

Returned by `decompose`, `subtract` and `regular_completion`.

- **What it means.** A content entry is negative, or the difference `subtract` takes has a
  negative entry: a symmetry vector counts coreps and cannot be negative. A signed
  combination of band representations is meaningful, and is what fragility means, but it
  is not a band set.
- **What to do.** Hand in a count of coreps; for `subtract`, the second content must be
  contained in the first.

#### `non_integer_multiplicity`

Returned by `decompose` and `subtract`.

- **What it means.** A content entry is not an integer.
- **What to do.** Hand in a count of coreps.

#### `not_a_regular_multiple`

Returned by `regular_completion`.

- **What it means.** The band count is not a multiple of `|U|`, the number of bands in one
  regular band representation of the group, the only counts at which the regular padding
  is defined.
- **What to do.** Pick a multiple of `|U|`; the reason names the next one.

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
- **What to do.** Report it, with the model and the call.
- **Known cases.** One is known from `realized_group`, with a way past it. With
  `symmetrize=False`, operations obeyed only within the declared tolerance meet the exact
  check unprojected: leave `symmetrize` at its default
  ([numerical models](../guide/numerical.md#then-symmetrize)).

#### `rep_internal_factor`

Returned by `realized_group`.

- **What it means.** Candidate matrices carry `H(k)` onto `H(Rk)` as the group requires,
  but on none of them does a site's block take the form a representation of these
  operations has, a character times the group's own factor for the operation, its spin
  rotation where there is spin: they are some other operations' matrices.
- **What to do.** Where the multiplets were stated (`multiplets=`), the statement;
  otherwise report it.

#### `rep_not_multiplicative`

Returned by `realized_group`.

- **What it means.** Candidate matrices pass the covariance and site-matrix checks, but none
  reproduces the group's multiplication table. Either they form a projective
  representation of a class the table does not carry, or a conserved pseudospin admits no
  representation of the group on its copies.
- **When it has a route.** Spinor states held in a spinless model's orbital index carry
  the sign of spin ½ (`context["spin_half"]`): declare `spin=`, and they are read in the
  double group. `realized_group` measures that class before it builds, and descends to the
  largest subgroup that carries an ordinary representation.
- **Otherwise.** Report it.

#### `rep_not_unique`

Returned by `realized_group` and `symmetrize`.

- **What it means.** More than one distinct representation passes every check, so the
  choice the measurement should fix is not fixed: the k-points the candidates are tested at
  are too few, or the model carries more symmetry than the group.
- **What to do.** Open an issue on pythtqc's GitHub with the model and the call: the
  measurement should have fixed this choice.

#### `canonical_norm`

Returned by `pythtqc.symmetry.normalize_rep`.

- **What it means.** A matrix raised to its operation's order is not a multiple of the
  identity, so it represents no operation and no phase can fix it.
- **What to do.** A matrix of one's own handed in is not the operation's; met any other way,
  report it.

#### `antiunitary_anchor`

Returned by `pythtqc.symmetry.normalize_rep`.

- **What it means.** No phase the canonical norm allows makes a matrix agree with time
  reversal, `T conj(D) T† = D`: the matrix and the time reversal belong to different
  representations, and corep labels built on it would be meaningless.
- **What to do.** As for `canonical_norm`.

#### `little_group_mismatch`

Returned by `corep_content`.

- **What it means.** The operations of the little group measured at a point do not match
  the tables' list for it, so no label would be meaningful.
- **What to do.** Report it.

#### `regular_content_not_integral`

Returned by `regular_completion`.

- **What it means.** A multiplicity of the regular band representation comes out
  non-integer, or the regular content does not carry `|U|` bands at every point: the
  tables' characters and the little group disagree.
- **What to do.** Report it.

#### `site_group`

Returned by `realized_group`, through a declaration.

- **What it means.** A site's operations are not closed under the group's product, or the
  character table computed for the site group fails its own orthogonality check.
- **What to do.** Report it.

### The projection, and the tables' setting

#### `symmetrize_defect_exceeds_tol`

Returned by `symmetrize` and `realized_group`.

- **What it means.** The group average moved `H` by more than the declared `tol`. Over
  operations obeyed within `tol` the move is of the order of their residuals, so a larger
  one says the matrices averaged with are not the symmetry the residuals were measured
  for.
- **What to do.** Check the declared tolerance, and the centres' defect the reason
  quotes; otherwise report it.

#### `symmetrize_range_not_closed`

Returned by `symmetrize` and `realized_group`.

- **What it means.** The symmetrized hopping list has weight outside the range the group can
  carry a hopping to, so the transform back to hoppings aliased.
- **What to do.** In `symmetrize`, a larger `k_grid=`; from `realized_group`, report it.

#### `wyckoff_setting`

Returned by `wyckoff_sites_tables`, `srsi_of_model` and `splitting`.

- **What it means.** The list of Wyckoff positions read is not written in the setting the
  tables are, or an orbit's geometry and its measured content name different Wyckoff
  positions. Every letter returned would be a key into the wrong setting, so none is.
- **What to do.** Report it.

## Two checks no returned group reaches

Two checks complete the list, and no group a call returns reaches either.

### `lattice_not_descended`

A group whose lattice is a supergroup of the model's, as a supercell's would be, handed to
a call that reads the tables. `realized_group` never returns one: it refuses `supercell`,
or descends to the model's own lattice.

### `magnetic_type_out_of_scope`

A magnetic group of type III or IV handed to `symmetry_indicator`. The package is validated
for types I and II, and `realized_group` returns no other: a Hamiltonian that breaks time
reversal is returned the unitary subgroup of its magnetic group
([the magnetic scope](../background/limits.md#magnetic-groups-of-types-i-and-ii-only)).
