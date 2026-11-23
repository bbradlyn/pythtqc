# Changelog

## Unreleased

## 0.2.0a1 — the third public version

In this version what the calls return is held in records read by name, whose fields name
the physics, and many calls, records, fields, arguments and checks are renamed or gone:
a script written for 0.1.0a3 needs editing wherever it reads them. `realized_group`
takes one declaration of precision, holds each operation to a tolerance relative to the
size of the Hamiltonian, and returns the group of models it refused or raised on, many
of them written on a strongly skewed lattice basis or carrying a little noise. A few
values that were wrong are now right, each named below with the group or the model it
concerns.

### What `realized_group` takes and returns

- `realized_group` takes `precision=` as the one declaration of a numerical model's
  precision, beside `position_tol=` for a model whose orbital centers alone are
  imprecise, and `symprec=`. It no longer takes `tol=`, `cut=` or `window=`, which a
  declared precision sets, nor `dz=` or `c=`, the depth of a substrate's shadow sites and
  the height of a layer's slab cell, which stay at `0.05` and `30.0`.
- `realized_group` no longer takes `descend=`, which asked for the group of the cell and
  refused where the Hamiltonian broke any of it, or `symmetrize=`. It returns the group
  the Hamiltonian obeys, and projects the Hamiltonian onto its symmetric part where a
  declared precision calls for it. What it returns on a call without them is unchanged.
- `symmetrize`, `symmetrize_positions` and `classify_operations` are no longer exported.
  `realized_group(model, precision=...)` moves the orbital centers, projects the
  Hamiltonian and classifies each operation itself, and returns the projection's record
  on `r.numerical.symmetrization`.
- `realized_group` takes each site's stated multiplets as `site_multiplets=`, where it
  took `multiplets=`. The multiplets stated are read as before, and
  `SiteMultiplets.summary()` ends `stated by site_multiplets=`.
- `RealizedGroup`, what `realized_group` returns, is read by name and does not unpack:
  `r.model`, `r.group`, `r.rep`. `r.bundle` is gone. Its top level holds `group`,
  `operations`, `wyckoff_orbits`, `descent`, `time_reversal`, `beyond_grey`,
  `conserved`, `environment`, `decoration`, `model`, `rep` and `spin`, the spin pairs
  `spin=` declared. How the measurement was made is `r.numerical`: `precision`,
  `positions`, `symmetrization`, `validated` and `source`, the model as handed in.
  `r.numerical.summary()` prints the precision declared, the centers moved and the
  projection of `H`, which `r.summary()` no longer prints.
- `r.descent`, its steps and the operations dropped, `r.time_reversal`, `r.beyond_grey`,
  `r.conserved`, `r.decoration`, `r.numerical.precision` and `r.numerical.positions` are
  records where they were mappings with fixed keys, each key a field of the same name.
  So are a descent's `lattice`, a step's `lattice` and `projective`, and the atomic-limit
  table's `numerical.accidental`: `DescentLattice` with its `LatticeTranslation` and
  `TranslationDressing` rows, `StepProjective` with its `ProjectivePair` rows, and
  `AccidentalDegeneracy`. A descent names its two groups `cell_group` and `subgroup`,
  each a `NamedGroup` with `name`, `number_str` and `size`. An operation is named by
  `index` and `antiunitary`. The centers are `"centers"` among the causes of a descent.
  `r.environment` is a record for a layer, with `kind`, `declared`, `cross_check` and
  `presentation`, and `None` for a crystal periodic in three directions. `repr(r)` names
  a second group by its name and number, `descended from Fd-3m1' #227.129`.
- `r.descent.dropped`: an operation the declared site irreps dropped holds in `rows` each
  declaration checked against it, where `rows` was always `None`.
- `r.wyckoff_orbits` is the atomic-limit table, a row per Wyckoff orbit of orbitals, in
  place of `r.rep.characters` and its row per orbital. A row, a `WyckoffOrbit`, holds the
  orbit's `wyckoff`, its `site_symmetry` with its `1'` where the group has time reversal
  and `unitary_site_symmetry` without, its `orbitals`, its site irreps as `label` and as
  `terms`, `determined_by`, what was `declared`, its `band_representation` and its
  `site_multiplets`. How the row was read is `o.numerical`. A row takes the orbit's name,
  the one the tables of elementary band representations write at its position, whichever
  of its sites is listed first: `p_x` and `p_y` at `2c` of `P4mm1'` read one row `B1`,
  where the two sites read `B1` and `B2`.
- `r.wyckoff_orbits` reads each row's band representation on the row's own orbitals: two
  Wyckoff orbits at one site each induce their own sum, the sums adding up to the one
  read for the site's orbitals together. lin3g's two `1a` orbitals induce
  `-E1 at 1a (4mm1')` and `-E2 at 1a (4mm1')`.
- `BandRepresentation` is a sum of term rows with its printed `label`, compared by its
  terms. `AtomicLimitRow`, `MultiplicitySpace` and `OrbitalAssignment` are no longer
  exported, and `WyckoffOrbit` is.
- `r.rep` holds the matrices, `antiunitary`, its methods and `r.rep.numerical`, the
  residuals of its matrices. What repeats the record (`model`, `group`, `assignment`,
  `operations`, `characters`, `multiplets`, `spin`, `environment`) is gone from it, with
  `indices` and `antiunitary_operator`.
- `r.numerical.validated` holds `obeyed`, `broken`, `undetermined`, `residuals`, `tol`,
  `n_kpoints`, `closed` and `discrepancies`, each a row with `index`, `refuted_by` and
  `residual`. `indices`, `verdicts`, `operations` and `group_label` are gone.
  `r.numerical.symmetrization` holds `site_multiplets` where it held `multiplets`, and
  its `checks`, `verification`, `copies`, `parts` and `shares` are records, `None` where
  empty.
- `SpinBasis` holds `pairs`. `kind`, `norb`, `spin_of_orbital` and its index tables are
  gone, and its summary prints the pairs.
- `r.group` holds `name`, `number_str`, `spinor`, `periodic_dirs` and `magnetic_type`,
  `"II"` for a grey group and `"I"` for a group without time-reversal symmetry. Its other
  fields are no longer public: `sg_number`, `size`, `include_TR`, `magnetic`,
  `vacuum_dirs`, `symmetries`, `lattice`, `provenance` and the methods on k among them.
  Its operations are `r.operations`. Its repr writes `magnetic_type=` and
  `periodic_dirs=`, and its summary has a line for its magnetic type in place of the one
  for its operations.
- `r.operations` names a row's index `index` and whether it is combined with time
  reversal `antiunitary`, where it named them `ind` and `time_reversal`. The table prints
  its first column under `index`.
- `seitz` is gone: `r.operations[i].seitz` is the symbol it returned.

### What the other calls take and return

- `hsp_table`, `coreps_at`, `characters` and `sewing_matrix` take what `realized_group`
  returns, `hsp_table(r, bands=6)`, and raise `TypeError` on `r.rep`. `srsi_of_model` and
  `splitting` take it alone, and raise `TypeError` on a tuple of its parts, where they
  took one.
- `operations`, `operation_index`, `high_symmetry_points`, `little_group`, `corep_table`,
  `identify_from_characters`, `k_grid`, `wyckoff_sites_tables`, `ebr_rows` and `ebr_data`
  take the group, `r.group`. Handed what `realized_group` returns, its representation or
  anything else, each raises `TypeError` naming `r.group`, where some read `r` or `r.rep`
  as the group and the others raised `AttributeError`.
- `coreps_at`, `characters`, `sewing_matrix`, the representation's methods,
  `little_group`, `corep_table` and `identify_from_characters` take a k-point by its
  tabulated name or by its coordinates, and return the same for both:
  `coreps_at(r, "GM", bands=6)`, `little_group(r.group, "X")`. A name the group does not
  tabulate raises `ValueError` listing the names. The name of a point out of a layer's
  plane returns the refusal `out_of_plane`, and so do three coordinates of such a point
  in `coreps_at`, `characters`, `sewing_matrix` and the representation's methods, where
  pythtb raised. `kpname=` is gone from all of them but `little_group`. `ktol`, `atol`
  and `mesh_tol` are no longer arguments of the first four, nor `ktol=` of
  `corep_table`, `identify_from_characters` and `k_grid`. `r.rep.k_action` is gone,
  `r.rep.transform_k` computing the same, and `r.rep.covariance` takes `kpoints=`.
- `corep_content` is `coreps_at`, and `format_counts` is `format_coreps`.
- `hsp_table` returns the symmetry data vector as `t.symmetry_data`, a row for each
  degenerate multiplet at each point as `t.multiplets["GM"]`, the points as `t.kpoints`,
  and the group the labels name as `t.group`. `t.counts`, `t.points`, `t.content`,
  `t.partitions`, `t.records`, `t.thresholds`, `t.group_provenance` and `t.rep` are gone.
- `coreps_at` returns a `CorepsAt`, in place of `CorepContent`: `kpoint`, `k`, `bands`,
  the corep multiplicities `coreps`, a row for each multiplet in `multiplets`, and
  `isolation`. The zero thresholds and the identification of the characters are under
  `c.numerical`. A row, a `Multiplet`, holds `kpoint`, `k`, `bands`, `energy`,
  `corep_names` and `label`, and `t.multiplets["GM"]` holds the same rows. The partition
  of the multiplets is gone: rows with the same `corep_names` carry the same coreps.
- `identify_from_characters` returns a `CorepIdentification`, in place of
  `IrrepContent`, naming its point `kpoint`.
- `characters` returns a `BandCharacters`, in place of `CharacterTable`: a row for each
  multiplet with its `bands`, `energy` and `characters` by operation index, the little
  group's `operations`, and the residuals under `ch.numerical`. Its summary names each
  operation by its index and Seitz symbol.
- `SewingMatrix` holds `operation`, `seitz`, `bands`, `k`, `kpoint`, `G`, `matrix`,
  `eigenvalues`, and its residuals under `sm.numerical`. `wrap` is `G`, and `trace`,
  `angles`, `operator` and `antiunitary` are gone. `sewing_matrix` always checks that the
  states solve `H(k)` and that the bands close their multiplets: `check=` is gone.
- `little_group` returns a record that holds `kpoint`, the tabulated name of its point,
  `"X"`, or `None` where the point is none of the tables'. Its summary and repr write
  the point `X (0, 1/2)`.
- `little_group` returns its operations as `operations`, and the indices of its unitary
  and antiunitary operations as `unitary` and `antiunitary`, where `unitary` and
  `antiunitary` held irrep's records and `indices` the unitary indices. Its `G` is keyed
  by index. Its repr writes `k` as its summary does, `(1/3, 2/3)`.
- `high_symmetry_points` names the tabulated points out of a layer's plane
  `out_of_plane`, where it named them `missing`. `k_grid` holds its points as `kpoints`,
  where it held them as `points`, and no longer holds `hsp` or `missing`: `contains`
  says which tabulated points it holds.
- `wyckoff_sites_tables` holds a row's Wyckoff position as `wyckoff`, `"1a"`, and the
  positions it slides onto as `slides_onto`, `("1a",)`, where it held `letter`,
  `multiplicity` and the letters `closure`. `wyckoff_positions_spglib` and
  `WyckoffPosition` are no longer exported.
- `ebr_rows` is keyed by the site irrep and the Wyckoff position, `rows["-E1", "1a"]`,
  where it was keyed by the band representation's full label. Each row is unchanged.
- `ebr_data` returns a record, `EBRData`, where it returned a mapping. Its columns `ebrs`
  are `Term` rows, each a site irrep at a Wyckoff position with its site-symmetry group.
  `corep_names` are the rows' coreps, and `matrix`, `smith` and `convention` are as
  before, the convention an `EBRConvention` whose `corep_names_bcs` are the Bilbao
  Crystallographic Server's names of the rows. Every value is unchanged.
- `operations` returns a table that names its group as `ops.group`. `ops.name` and
  `ops.number_str` are gone. It prints as before.
- `Term` is a term of every sum, in a candidate sum, an atomic realization, a Wyckoff
  orbit's band representation or the columns of `ebr_data`: a row with `multiplicity`,
  `irrep`, `wyckoff`, `site_symmetry` and `unitary_site_symmetry`, the site symmetry
  written with its `1'` in a group with time-reversal symmetry, `1a (4mm1')`.
- `decompose`, `symmetry_indicator`, `srsi_of_bands` and `regular_completion` take the
  table `hsp_table` returns alone, `decompose(t)`, or a symmetry data vector written out
  by hand with what `realized_group` returned beside it, `decompose(vector, r)`. The
  first parameter is `symmetry_data`. `group=`, `rep=`, `bands=`, `points=`, `verify=`
  and `stack_period=` are gone, and so are `regular_completion`'s keywords passed on to
  `decompose`, whose own keywords take its padding: `decompose(rc.padding, r, cap=...)`.
  A table with anything beside it, a vector with nothing beside it, and `r.group` or
  `r.rep` in place of `r` raise `TypeError` saying what to pass. The Smith form is
  computed again on every call.
- `decompose` returns each sum as a `BandRepresentation` in `d.candidate_sums`, printed
  as `-E2 at 1b (4mm1') + -E at 2c (mm21')`, and whether one carries the vector as
  `d.decomposable`. `verdict`, `solutions`, `terms`, `positions`, `solution_strings()`,
  `ebrs`, `caveat`, `unique`, `n_bands`, `labels` and `y` are gone, and `content` is
  `symmetry_data`. The bound, the cap, the completeness and the restriction are under
  `d.numerical`, and `d.numerical.summary()` prints what the vector was read on. The
  search with negative multiplicities is a `SignedCombinations`, with its `sums` and
  `truncated`, and the site irreps a `SiteContent`. The summary's first line reads
  `decomposable: 3 sums of elementary band representations carry this symmetry data
  vector`, or `not decomposable: ...`.
- `symmetry_indicator` returns the class as `values` in `indicator_group`, `"Z6 x Z12"`,
  with `divisors`, `order`, the paper's named indicators as `indicators`, the vector of
  the plane as `symmetry_data` and the group the class lives in as `group`. `verdict`,
  `nontrivial`, `indicated`, `structure`, `canonical`, `canonical_note`,
  `canonical_tex`, `irreptables_named`, `as_irreptables()`, `number_str`, `group_name`,
  `magnetic_type` and `n_bands` are gone. The vector on every point of the group's
  tables is `si.numerical.completed_symmetry_data`, and the Smith form computed again,
  the compatibility relations, the stack's completion and the two-dimensional reading
  are records under `si.numerical`. The summary's first line reads
  `class (1, 0) in Z6 x Z12, in P6/mmm1' #191.234`, or
  `the indicator group of P4mm1' #99.164 is trivial`.
- `regular_completion` returns `symmetry_data` in place of `occupied`, and `target` is
  gone. `Restriction` holds `kpoints` in place of `points`, and `applied`, `rank`,
  `full_rank`, `layer_extent` and `require_exact` are gone. `in_plane_restriction` and
  `regular_content` take `kpoints=`, raise `TypeError` on `r` or `r.rep` in place of
  the group, and raise `ValueError` listing the names on a name the group does not
  tabulate, where `regular_content` raised `KeyError`.
- `completeness_check`, `CompletenessReport` and `WannierCaveat` are gone: the band count
  of a set of bands is `len(t.bands)`. `to_irreptables_name`, `from_irreptables_name`,
  `indicator_vocabularies`, `name_correspondence`, `NamePair` and `NAME_RELATIONS` are
  no longer in `pythtqc.srsi`. `splitting` no longer takes `rep=` or `points=`.
- `subtract` names its two parameters `symmetry_data_a` and `symmetry_data_b`. Every call
  passes them by position, as before.
- `theta_Z` and `theta_Zn` hold every set of stable real-space invariants by name:
  `theta_Z` maps each integer-valued invariant's name to its value, `theta_Zn` each one
  valued modulo n to a `ModularValue` with `value` and `modulus`,
  `mo.theta_Zn["theta_5"].value`. A printed tuple of invariants names them,
  `theta_1 to theta_3 = (2, 2, 0); theta_4, theta_5 = (0, 1) mod 2`. The values are
  unchanged.
- `srsi_tables` returns the paper's tables flat: `invariants`, each invariant's formula
  by name; `indicators`, each indicator's form in the invariants as the paper prints
  it; `moduli`, each one's modulus; `site_irreps`, rows of `SiteIrrep` with `irrep` and
  `wyckoff`, `"4d"`; `split_ebrs`; `stably_equivalent`, the site irreps no invariant
  tells apart; `group`; and `provenance`, a `Provenance`. The tables of a space-group
  number hold the grey group by its name, `"I4321'"` for 211, with no cell, and a lookup
  that needs a cell raises `TypeError` on it. `rows`, `names`, `row()`, `structure`,
  `ebrs`, `sg`, `number_str`, `spinor`, `site_keys`, `maximal_keys`,
  `band_count_form`, `z_rows`, `zn_rows`, `theta_on_columns`,
  `terms_without_ebr_column` and the three `_columns` methods are gone. `srsi_tables`
  raises `TypeError` on what `realized_group` returns.
- `tables.theta(p)`, `theta_unreduced` and `indicators_of` take a mapping keyed by site
  irrep and Wyckoff position, `{("-E", "2c"): 1}`, or a sum of term rows, such as a
  candidate sum of `decompose` or an atomic realization. `theta` returns an
  `Invariants` with `theta_Z` and `theta_Zn`, and `theta_unreduced` a mapping by name. A
  position written without its multiplicity, or with another one than the tables give
  it, raises `ValueError` naming the position to write, and a sequence of numbers raises
  `TypeError`. A site irrep is spelled as the list of elementary band representations
  spells it, `A` at a `222` site, wherever the invariants' records print one.
- `srsi_of`, `srsi_of_columns` and `stably_equivalent_pairs` are gone: `tables.theta(p)`
  and `tables.stably_equivalent` return the same values.
- `srsi_of_bands` returns `theta_Z` by name, `indicator` and `group`. The rational
  site-symmetry vector is `b.numerical.p`, keyed as `theta` takes it. `names`,
  `stable`, `denominator`, `n_bands`, `number_str` and `spinor` are gone.
- `srsi_of_model` returns a row for each Wyckoff orbit in `mo.wyckoff_orbits`, the row of
  `r.wyckoff_orbits` with its `theta_Z`, `theta_Zn` and `slides_onto`, the positions its
  orbitals slide onto, `("3a", "3b")`. Its localizations, its measured symmetry data
  vector and the composite site irreps are under `o.numerical`. A model with two Wyckoff
  orbits on one orbit of sites, lin3g's two orbitals at `1a`, has a row for each where
  it had one, and a composite multiplet beside another on one site is read on its own
  orbitals. The model's totals are unchanged. `orbits`, `names`, `moduli`, `structure`,
  `tables`, `number_str` and `spinor` are gone, and `srsi_of_model` no longer takes
  `rep=`.
- `atomic_realizations` takes `theta_Z` by name, `b.theta_Z`, and no longer takes
  `n_bands=`. It returns the record with no realization on fractional invariants, where
  it refused `srsi_not_atomic`, which is gone. A realization is an
  `AtomicRealization`, a sum of term rows equal to the same sum from `decompose`, with
  `label`, `theta_Zn` and `stably_equivalent`. `n_bands`, `atomic_possible`, `note` and
  `keys` are gone, and `bound` and `cap` are under `rz.numerical`.
- `splitting` and `nsit_criterion` return `nsit`, `valence` and `conduction`, each side's
  realizations as `atomic_realizations` returns them, the whole model's `theta_Z` and
  `theta_Zn`, `consistent_pairs` as pairs of realizations, and `stably_equivalent`.
  `verdict`, `reason`, `realizations_v`, `realizations_c`, `theta_Z_v`, `theta_Z_c`,
  `theta_Z_total`, `theta_Zn_total`, `n_bands_v`, `n_bands_c`, `v_atomic_possible`,
  `c_atomic_possible`, `indicator`, `moduli`, `same_content` and `SPLITTING_VERDICTS`
  are gone. `nsit_criterion` takes the whole model's invariants as what `tables.theta`
  or `srsi_of_model` returns, and each side's by name. `n_bands_v=` and `n_bands_c=` are
  gone. The summary's first line says whether the splitting is NSIT and why not.
- `wyckoff_orbit_content_incomplete` and `wyckoff_orbit_not_localizable` are the
  refusals that were `orbit_content_incomplete` and `orbit_not_localizable`.
- `WyckoffOrbitSRSIs` is the record that was `OrbitSRSIs`, and `SiteIrrep` the one that
  was `SiteKey`. `SRSIRow`, `SRSIIndicator` and `MaximalKey` are no longer in
  `pythtqc.srsi`. `pythtqc.srsi` holds the new records `BandSRSIsNumerical`,
  `WyckoffOrbitSRSIsNumerical`, `Localization`, `RealizationsNumerical`,
  `SRSITablesNumerical`, `IndicatorCheck`, `Provenance`, `ProvenanceCounts`,
  `Invariants` and `ModularValue`.

### Values that change

- `symmetry_indicator` holds each entry of `numerical.verification.change_of_basis`
  modulo the elementary divisor of its row, the cyclic factor the entry maps into, where
  it was reduced modulo that of its column. The matrix changes in 31 of the 170 grey
  tables with a nontrivial indicator group. In 16 of them the old matrix carried some
  class wrongly, as in `Pm-3m1'` with spin-orbit coupling, `Z4 x Z8`, whose entry from
  `Z4` into `Z8` is `6` where it read `2`. In 8 spinful tables
  `change_of_basis_is_identity` read `True` and now reads `False`.
- `groups.decompose_ebrs` reads each named indicator with every coefficient on the row it
  stands for, by the row's label on the Bilbao Crystallographic Server and the change of
  convention's row permutation, where it looked each coefficient up by its label alone.
  In `I4_1/a1'` with spin the two conventions name the rows at `P` differently, so the
  `P` part of `z2` was dropped: `z2` was `1.0` on each of the four elementary band
  representations induced from `4a` and `4b`, and could read `0.0` on a vector of the
  class where it is `1.0`. The four read `0.0` now. No other value of any group changes.
- `regular_completion` and `regular_content` count one copy of the regular band
  representation as the band representation induced from the general Wyckoff position:
  `|U|` bands, and `2|U|`, the Kramers pair there, for a spinful model in a group with
  time reversal, where they counted `|U|` bands in every case. A spinful `s` orbital on
  a triclinic lattice, `P-11'`, was refused `regular_content_not_integral` at every
  number of bands. One copy is 4 bands now, and its 2 bands complete to it with `-AuAu`
  at `1a`, the Kramers pair of the other parity. `lin3g`'s one copy is 16 bands, where
  it was 8. Spinless models and groups without time reversal count as before.
- `regular_completion` returns the padding of a band set in a group with a point whose
  little group holds the identity alone, which it refused with `not_a_band_structure`.
  In `P1` without time-reversal symmetry it returns the padding of the bands missing,
  where it counted the band set's own bands among them.
- `nsit` is true only where both sides have atomic realizations and no pair of them adds
  up to the whole model's invariants valued modulo n. A side with whole-number
  invariants and no atomic realization makes it false, where it made it true: Kane-Mele
  split at half filling reads `nsit` false, its conduction bands having no atomic
  realization.
- `atomic_realizations` and `decompose` list every solution when there are exactly `cap`
  of them, where they marked the list cut at the cap. `atomic_realizations` with `cap=4`
  returns the four realizations of `(B3)_12d`'s invariants in `I4321'`, where it refused
  `srsi_realizations_truncated`, and a decomposition with as many sums as its `cap` has
  `numerical.enumeration.truncated` `False`. The refusal and the flag now mean that
  there are more solutions than the cap, and the refusal's `value` is `cap + 1`. Both
  calls raise a `ValueError` for a `cap` below 1, which asks for no solution.
- `hsp_table` reads every high-symmetry point, one whose little group holds the identity
  alone included: there a set of bands carries the point's one irrep once for each band,
  which the table's symmetry data vector now holds, `M1` at `M` for Haldane's model with
  flux in `P3`. The table's field `skipped` is gone, and with it the `"skipped"` entry of
  a decomposition's `completeness` and of the context of `decompose`'s
  `incomplete_content`. `decompose` and `symmetry_indicator` return what they returned on
  such a table, and the same on its symmetry data vector handed on as a mapping, which
  they refused. A set of bands that cuts a degenerate level at such a point is refused
  there, `multiplet_split`, as at every other point.
- `symmetry_indicator` and `srsi_of_bands` complete a layer's symmetry data vector written
  out by hand in its plane: a corep at a point out of the plane takes the multiplicity of
  the corep in the plane it repeats in every elementary band representation of the
  layer's own Wyckoff positions, and the two calls return its class and invariants, where
  they refused `indicator_not_evaluable`. The free-standing Kane-Mele layer's valence
  bands, `{'-GM9': 1, '-K8': 1, '-M6': 1}` with `r`, read class `(1, 0)` in `Z6 x Z12`,
  as their table does. A vector that names coreps at a point out of the plane is read as
  written there, by `decompose` too.
- `realized_group` with nothing declared holds each operation to `1e-8` times `max|H|`,
  the largest entry of `H(k)` once the mean on-site energy is subtracted, where it held
  it to `1e-8` in whatever units `H` was written in, so that the same model in other
  units read differently: `lin3g` with every hopping and on-site energy off by a
  relative `1e-9` returned `P1` as written and `P4mm1'` with every energy divided by
  100. It returns `P4mm1'` at both scales now. An operation obeyed exactly is held to
  `5e-14` times `max|H|` at the least. `hsp_table`, `coreps_at`, `characters` and
  `sewing_matrix` read bands within `1e-8` times `max|H|` as one multiplet unless
  `degeneracy_tol` is given, a number in the units of `H`. A model multiplied by a
  constant returns the same group, rows and labels, and every number in the units of `H`
  multiplied by it.
- `realized_group` keeps an operation the Hamiltonian obeys within the tolerance in force
  where it read it as broken: an `F23` crystal of `s` orbitals whose hoppings carry a
  relative noise of `1e-11`, read with no precision declared, returns `F231'`, where it
  returned `C21'`, and at a noise of `1e-10` `F231'` where it returned `P11'` with no
  symmetry matrix found for any operation. Where the symmetry matrix found at a few
  k-points misses the tolerance, or none is found there, and that does not prove the
  operation broken, the operation is measured again by the matrix carrying each site's
  orbitals onto its image's that fits best over the whole k-point mesh. What it returns
  for a model without noise is unchanged.
- `realized_group` keeps an operation whose symmetry matrix meets the tolerance while
  lying a little off the operation's own form, each site's orbitals carried onto its
  image's, where it dropped it as broken by the tensor structure. The `p` and `d`
  orbitals at `8c` of `Im-3m` with a relative noise of `1e-9` on every term, every
  operation of which they obey within `3.2e-9` of `max|H|`, return `Im-3m1'` with
  nothing declared, where they returned `I4/mmm1'` with 64 operations dropped. Where the
  matrix found lies off a form it is near by more than rounding, the matrices of that
  form are fitted over the whole k-point mesh, as where the matrix found misses the
  tolerance, and the operation is read by the fitted one wherever it meets the
  tolerance. What it returns for a model without noise is unchanged.
- `realized_group` returns a model built from exact numbers as it was handed in, where it
  could return it made symmetric by a defect of rounding's size: the model of
  `examples/tqc_srsi_sg212_photonic.py` was returned `P4_3321'` "symmetrized (defect
  3e-14)", its measurement run three times over. Whether an operation is exact or only
  near is now read on its own matrix, in the representation the projection is built
  with, and no longer on the measurement's best matrix, whose accuracy at the k-point it
  starts from can be a little worse than rounding. Which operations are kept is
  unchanged, and a model with noise is made symmetric as before.
- `named_by` reads `"invariants"` at a Wyckoff position where the tables list no band
  representation, where it read `"point group"` or `"Mulliken"`: a site irrep there takes
  the name of the site irrep that the tables of the stable real-space invariants (Hwang
  et al., Nat. Commun. 17, 8564 (2026)) give the same invariants, alike with
  time-reversal symmetry and without. At 28 such positions, the lines of site symmetry
  `mm2` in 12 groups (`4l` to `4o` of `P4/mmm` among them), `B1` and `B2` change places,
  and a declared `B1` or `B2` there states the other row than it did, with spin as
  without. Every character, matrix and band label returned is unchanged.
- Every site of a Wyckoff orbit at such a position reads the orbit's name, whichever site
  is listed first: at `4i` of `P4/mmm`, where a fourfold
  rotation carries one site onto the other and exchanges `B1` with `B2`, the site that
  read the other symbol reads the orbit's.
- `r.wyckoff_orbits` sets, in `numerical.residual`, the product of two blocks of a
  multiplet of several orbitals with spin against the block of the product times the
  sign the double group gives it. Such a multiplet read `2.0` with its site irreps
  right, and now reads at rounding: the `p` shell of a `3m` site with spin-orbit
  coupling, which read `2.0`, reads `4e-15`.
- A tabulated k-point's reduced coordinates are returned in `[0, 1)`: a
  coordinate that is zero reads `0`, where on some cells, a triclinic one among them, it
  read `1`. The point is the same one.
- `two_tier_thresholds` gives a spectrum of one multiplet, whose smallest gap is
  infinite, a noise floor of zero, where it gave an infinite one: `8 ε |H(k)| / Δ` goes
  to zero as `Δ` grows. Its `effective` threshold is then the flat one, `1e-14`, and the
  characters of a model of one band are held to it.
- `format_coreps` leaves out a label of multiplicity zero, where it wrote it as a
  multiplicity of one: `format_coreps({'-GM6': 0})` is `''`.
- `tables_status()` reports `'ortools'` `True` wherever `ortools` is installed, where it
  reported `False` while the character tables were not loaded.

### What refuses or raises differently

- `realized_group` reads a crystal where it raised `ValueError`, "No magnetic space group
  could be detected!". Most such crystals are written on a lattice basis that is not
  reduced, the square lattice on `a1 + a2, a2` among them. What it returns for every
  other crystal is unchanged.
- `kpoint_not_tabulated` no longer refuses a tabulated point of a crystal written on a
  strongly skewed lattice basis, and a set of bands is read at every one: on the
  hexagonal lattice written on `a1 + 10 a2, a2, a3`, `K` and `H` were refused
  `kpoint_not_tabulated`, their coordinates having been compared after rounding. A k-point is the tabulated one where
  it differs from it by a reciprocal lattice vector of the cell as written.
- `realized_group` builds the representation of a crystal written on a strongly skewed
  lattice basis, where it refused `rep_not_covariant` though every operation held: the
  square lattice with one `s` orbital on `5 a1 + 8 a2, 3 a1 + 5 a2` among them. The
  covariance of `H(k)` is held to the rounding its Bloch phases carry at the rotated
  k-points besides that of the representation's own arithmetic.
- `realized_group` reads zincblende written on `a1, 6 a1 + a2, 4 a1 + 9 a2 + a3`, where it
  raised `InternalError`, the operations rebuilt from the tabulated ones not closing.
  Each rebuilt translation is formed exactly and rounded once.
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
- `realized_group` with a declared precision returns the group of a layer written on a
  strongly skewed lattice basis whose Hamiltonian it makes exactly symmetric: hexagonal
  boron nitride on `7 a1 + 5 a2, 4 a1 + 3 a2`, one center displaced by `1e-9` of the
  lattice constant and `precision=1e-8` declared, returns `P3m11'`, where it was refused
  `rep_not_covariant`. The phases of the symmetry operators at images tens of Brillouin
  zones away are formed exactly from the stored positions and rounded once, where they
  missed by up to `1e-12` and left the averaged Hamiltonian with hoppings that break its
  symmetry at that size.
- `realized_group` given `site_multiplets=` that separates orbitals a symmetry of the
  hoppings mixes returns the group that keeps them separate: `p_x` and `p_y` on the
  fourfold site of a square lattice, stated as two multiplets, give `Pmm21'` on a
  substrate and `Pmmm1'` free-standing or stacked into a crystal, where it raised
  `InternalError`.
- `realized_group` returns a group where it raised `IndexError`, on a model whose
  operations measured to hold do not close into a group, as a numerical model's
  operations near its tolerance can: the largest group they contain, as for any such
  model.
- `realized_group` returns the group of a model whose hoppings carry a relative noise
  near `1e-10`, no precision declared, where it raised `InternalError`: `P-43m1'` for
  `cubic_3c_perp_doublet` so perturbed. Its Hamiltonian is averaged over the group with
  the exact representation nearest the matrices read off its multiplets, which carry its
  noise.
- `realized_group` refuses a model whose asymmetry is close to its tolerance,
  `rep_not_covariant`, naming the group, where it raised `InternalError`: the spinless
  `p_x, p_y` honeycomb with a relative noise of `3e-9` on every term, each operation of
  `P6mm1'` obeyed within the tolerance `2.67e-8` on its own, the worst to `1.67e-8`,
  while the representation built for the group carries `H(k)` onto itself only to
  `2.77e-8`. The reason asks for a precision, above the asymmetry if it is noise and
  below it if it is a term to keep. `context["obeyed_alone"]` holds the worst an
  operation reaches alone. A representation that misses an operation by more than the
  tolerance beyond what that operation reaches alone still raises.
- `realized_group` refuses a model whose orbital centers miss their symmetric positions
  by too little to be refused before anything is built, read with no precision
  declared, `positions_not_exactly_symmetric` wherever that miss keeps `H` from being
  made or held exactly symmetric, the group identified named in it and the reason
  saying to declare `precision=`. It was refused `rep_not_covariant` or
  `symmetrize_range_not_closed`, or raised an `InternalError`. On a cell whose sites of
  one content repeat under a fraction of a cell's translation, such as CsCl, the same
  miss leaving the operations read off the centers open is refused this way before
  anything is built, the cell's group named, where it was refused
  `operations_not_consistent`. Every model that was read is read as before.
- `realized_group(model, spinor=False)` on a spinful layer, or on a spinful crystal given
  `decorations=`, returns the refusal `single_valued_spinful`, naming the group the cell
  allows, where it raised `ValueError` after the group was measured.
- `stack_presentation` builds the stack of a slab whose non-periodic lattice vector is
  tilted off the layer's normal, with an orbital off the plane through the origin, such
  as pythtb's `fu_kane_mele(1.0, 0.1).cut_piece(1, 2)`, where it raised `InternalError`.
- `regular_content` and `regular_completion` raise `InternalError` for a copy of the
  regular band representation that does not hold every little-group representation a
  whole number of times, and the check `regular_content_not_integral` is gone. One copy
  holds every little-group representation a whole number of times in every group of the
  tables, so a copy that does not is a defect of the package.
- `subtract` and `regular_completion` refuse a table that refused at some point,
  `incomplete_content`, as `decompose` does, where they read the points refused as
  holding nothing. `allow_incomplete=` does not lift it.
- `regular_content` asked at a point out of a layer's plane returns the refusal
  `out_of_plane`, where it raised `KeyError`.
- `operation_index` refuses `unknown_operation` for a Seitz symbol with a translation
  component that is no fraction of denominator 24 or less, `{1|0.37,0,0}` or
  `{1|1/25,0,0}`, where it raised `InternalError`. Its reason calls the symbols it lists
  the nearest in spelling only where some symbol is close.
- `atomic_realizations` and `nsit_criterion` return a refusal handed in as `tables`, such
  as `srsi_tables(1, spinor=True)`, where they raised `AttributeError`. `nsit_criterion`
  returns a refusal handed in as `theta_total`, where it raised `TypeError`.
- `SpaceGroupRep.sewing_operator` returns the operator wherever the operation carries `k`
  to `k` plus a reciprocal lattice vector within the little group's tolerance, `1e-8` by
  default, where for a miss above `1e-9` it raised an error of `bloch_operator()` naming
  `k_out`, which it does not take. The operator carries `H(k)` onto the Hamiltonian at
  the image folded back by that vector, and `tol=` no longer bears on `k`. One point
  given as a row of shape `(1, dim)` returns one matrix of shape `(nstate, nstate)`, as a
  flat `k` does, where it returned `(1, nstate, nstate)`.
- `transform_k`, `in_little_group`, `little_group` and `sewing_operator` of `r.rep`, given
  a `k` with the wrong number of components, raise a `ValueError` that opens with the
  call and says how many to pass, where numpy raised an error naming no call.
- `sewing_matrix`, `characters` and `coreps_at` handed an `(evals, evecs)` pair whose
  energies and eigenvectors differ in number raise a `ValueError` that opens with the
  call, where numpy's error on shapes reached the user. `hsp_table` handed anything for
  `states=` other than `None`, `r.model` or a `WFArray`, such as the pair of one point,
  raises a `ValueError` naming what it takes, where it read that point and refused every
  other one `states_not_at_k`.
- `bands=[]` raises `ValueError` saying what to pass, where numpy raised on a reduction
  of an empty array.
- `models.i41_soc` raises a `ValueError` that opens `i41_soc():` for a name of `soc=`
  other than `"generic"` or `"axial"`, where it raised a bare `KeyError`, and for a
  `soc=` that is not a pair of (2, 2) matrices, where one (2, 2) matrix was read as two
  seeds and returned a model. `models.atomic_model` handed orbits that an operation does
  not carry onto themselves raises an `AssertionError` that opens `atomic_model():`,
  where it opened `closed_bonds():`.
- `splitting` with `bands=` outside the model's bands, or naming none, raises a
  `ValueError` that opens `splitting():`, where the message named no call or
  `hsp_table()`. It raises before the group's tables are read, where a group whose
  tables refuse returned that refusal. `canonical_determinism` handed something that is
  neither a group nor its tables raises a `TypeError` that opens
  `canonical_determinism():`, where it opened `srsi_tables():`.

### Messages and printouts

- The refusals' reasons, the errors' messages and the summaries are spelled the American
  way, "the orbital centers", "labeling", "modeling", and name the symmetry data vector
  where they said "the content". The reason a refusal gives for centers a little off
  their symmetric positions opens
  `the orbital centers miss the cell's own operations by delta = ...`. What every call returns is unchanged.
- `hsp_table`, `coreps_at`, `characters` and `sewing_matrix` raise the errors for their
  `states=` and `bands=` with messages that open with the call's own name, where some
  named no call and `hsp_table`'s opened `coreps_at():` and asked for the
  `(evals, evecs)` pair it does not take. `hsp_table` handed another model names the
  solved `WFArray` it also takes. `multiplets`, `multiplet_separation` and
  `multiplet_spread` open their errors with their own names, an empty spectrum reading
  "no energy is passed", and `solve_at` opens its error for a parameter's value that is
  not a number with `solve_at():`.
- `r.rep` opens the `KeyError` for an operation index it does not hold with the name of
  the method called: `transform_k():`, `in_little_group():`, `bloch_operator():` or
  `sewing_operator():`. The `ValueError` of `r.rep.covariance` for a model edited since
  it was measured opens `covariance():`.
- `k_grid` given a sequence of divisions of the wrong length, and
  `pythtqc.symmetry.bloch_operator` given a k-point with the wrong number of components,
  raise `ValueError` beginning with the call's name and saying what to pass.
- `ambiguous_multiplet`, the refusal, counts every orbital of a site declared with one
  word and names the multiplet each leftover orbital is: `{0: 'E', 1: 'E', 2: 'E'}` on
  an `s` orbital beside a bound `(p_x, p_y)` pair says that 3 orbitals are declared with
  `E`, orbital 0 left over, a multiplet of one orbital, and asks for a word for one
  orbital there, such as `A1`, where it offered a key across the `s` orbital. A key that
  names a bound multiplet with other orbitals, `{(0, 1, 2): 'E'}`, is offered the
  multiplet alone, `{(1, 2): 'E'}`. Generator matrices at a site whose declarations name
  no whole multiplet, and those beside a double-valued site irrep at a site of several
  multiplets, each have their own reason.
- `unknown_label`, the refusal, for a generator character whose key matches operations
  of two classes names the classes, `2m(x), 2m([110])`, where it printed their positions
  in a list.
- `not_a_band_structure`, `indicator_not_evaluable`, `restriction_not_exact` and
  `unknown_label`, the refusals of `decompose` and `symmetry_indicator`, carry the group
  in `context["group"]`, and `not_a_band_structure` names it, `R-3m1' #166.98`, where it
  gave the number alone. `restriction_not_exact` lists the little-group representations
  by point and in ascending order at each, `-A7` before `-A10`, in its reason and in
  `context["missing"]`, where both had the tables' order.
- `not_a_band_structure` holds each entry of `context["violations"]` as
  `(index, value)`, where it held a label between them: the label of the matrix's row of
  that index, which does not name the coordinate.
- `indicator_not_evaluable` lists the little-group representations out of a layer's
  plane in ascending order at each point, `-A7` before `-A10`, and its reason says
  "no matrix for operation 48" for one operation. In `context["stack"]`, a stack that
  was never built has `stack_period` `None`, where it held the vacuum between the copies.
- `kpoint_not_tabulated` gives the tabulated points as lists of floats, in its reason and
  in `context["available"]`, where it gave numpy arrays.
- `srsi_table_absent` no longer ends its reason in a parenthetical on the caption of the
  supplement's tables.
- `format_coreps` and every summary that prints a sum of coreps or of site irreps print a
  multiplicity `2x-GM6`, where they printed `2×-GM6`.
- `coreps_at`, its `numerical` record and the identification `identify_from_characters`
  returns print "irreps" in their summaries where the group has no time-reversal
  symmetry, as every other summary does, and the summary of `hsp_table` prints "their
  irreps at each high-symmetry point" there. The repr of what `ebr_data` returns counts
  the group's irreps there.
- `characters` and `coreps_at` print each multiplet's energy in their summaries at the
  resolution its bands were grouped at, the table's `degeneracy_tol`: a level at the
  rounding of zero prints `at 0`, where it printed `at -1.388e-15`.
- `SpinBasis.summary()` heads its pairs "each pair one spinful orbital", where it said
  "each pair one Wannier function".

### Documentation

- The pages of the documentation are rewritten in one style: each of the ten guide pages
  answers the question its title asks and closes on where to go next.
  `docs/reference/records.md` is generated from the records' docstrings, as the functions
  reference is from the functions', in place of a page written by hand. Every example
  in a docstring is run by the tests and prints what it shows.

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
