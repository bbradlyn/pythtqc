# Glossary

One name for each thing, and every tolerance with its unit and default. The
pages, the records and the docstrings call each thing by one name; this page
lists those names, says in a line what each one is, and points to where it is
explained. The ideas themselves are in
[the TQC primer](../background/tqc-primer.md) and
[how pythTQC reads a Hamiltonian](../background/measurement.md), and every field
of every record is in [the records reference](records.md).

## The names on the records

What each thing is called where a call returns it.

### The group

| what it is | its name | where |
|---|---|---|
| the space group the Hamiltonian obeys | `name`, its Hermann–Mauguin symbol (`"P6/mmm1'"`); `number_str`, the label the tables file it under (`"191.234"`) | `r.group`, and `number_str` on every record read in it |
| an operation | `index`, its index in the tables' order; `seitz`, its Seitz symbol `{R\|t}` in the tables' conventional setting; `name`, its name in the model's Cartesian frame, `C3(z)` or `m(x)`, primed where it carries time reversal | `r.operations`, `operation_index` |
| the verdict on each operation | `obeyed`, `broken` and `undetermined`, the operations by index | `r.numerical.validated` |
| the group the cell allowed, and the group the descent arrived at | `descent.cell_group` and `descent.subgroup` | `r.descent` |
| what dropped an operation | `refuted_by` | `r.descent.dropped` |
| what a declaration excluded before any hopping was read | `decoration.declared` | `r.decoration` |
| what lies under a layer | `kind`: `"substrate"` or `"free-standing"`; `declared`, whether anybody said | `r.environment`; the `environment=` argument |
| the other environment's group | `cross_check` | `r.environment` |
| time reversal, as measured | `time_reversal.outcome`: `"kept"` or `"measured broken"` | `r.time_reversal` |
| symmetry beyond the crystal's own | `conserved.dim`, printed `additional commuting matrices` | `r.conserved` |
| how the conserved matrices acting within sites split the states | `conserved.sectors`, each sector's `orbitals`, `sites`, `copies` and `dim`; `conserved.images`, where each operation carries each sector | `r.conserved` |

### The orbitals

| what it is | its name | where |
|---|---|---|
| the Wyckoff position | `wyckoff`, multiplicity and letter (`"6c"`); `letter`, the letter alone | `wyckoff` on the atomic-limit table and on `wyckoff_sites_tables`' positions; `letter` on the SRSI records |
| the sites the group's operations carry onto one another, all at one Wyckoff position | a Wyckoff orbit: a row, its `orbitals`, and its sites' positions as `numerical.positions` | the atomic-limit table, `r.wyckoff_orbits` |
| the site-symmetry group | `site_symmetry`, its Hermann–Mauguin symbol (`"-42m1'"`); `unitary_site_symmetry`, without time reversal (`"-42m"`) | the atomic-limit table |
| the orbital function | `declared`, the word a declaration wrote, spelled as `characters=` takes it; `None` on a row nothing was declared for | the atomic-limit table |
| the site irrep | `label`, and `numerical.declared_irrep` for what a declaration resolved to | the atomic-limit table |
| how a site irrep is known | `determined_by`: `declaration`, `declaration not carried`, `hamiltonian` or `lowest-l` | the atomic-limit table |
| where its name comes from | `numerical.named_by`: `tables`, `invariants`, `point group`, `Mulliken` or `unnamed` | the atomic-limit table |
| the orbitals the Hamiltonian binds into one carrier of a site irrep | a multiplet: per site, `site_multiplets` on a row | the atomic-limit table |
| a choice of the one-dimensional characters the hoppings leave open, one on each coupled part | a labelling: a declaration fixes it, and the lowest-ℓ rule makes it otherwise; `determined_by` says which rows rest on it, and `note` which band sets' labels | the atomic-limit table; `hsp_table` |
| a conserved pseudospin, and the site content assigned its copies | on a row, `label` the one assigned, `numerical.contents` every one the group's table admits, `numerical.n_representations` how many representations of the copies it admits | the atomic-limit table |
| the band representation a Wyckoff orbit induces | `band_representation`, a `BandRepresentation` on each row, its `terms` the elementary band representations it is the sum of | the atomic-limit table |
| the matrix of each operation on the orbitals | `matrices`, keyed on the operation's index | `r.rep` |
| a declared spinor basis on a spinless model | `spin`, a `SpinBasis` | `r.spin`; the `spin=` argument |

### The bands

| what it is | its name | where |
|---|---|---|
| the bands asked for | `bands=`: an integer is the lowest that many, a sequence exactly those band indices; `bands` on a record, the indices themselves | every call that reads bands |
| a degenerate multiplet of bands at one `k` | a row of `multiplets`: its `bands`, its `energy`, and its `characters` or its coreps | `characters`; `coreps_at`; `hsp_table` |
| the coreps of a set of bands at one point | `corep_names` and `label` on each row of `multiplets`; `coreps`, summed over the set | `coreps_at` |
| the symmetry data vector of a band set | `symmetry_data`; per point, the rows of `multiplets` | `hsp_table` |
| the gap from a set of bands to the nearest band outside it | `isolation`, in the units of `H` | `coreps_at` |
| a high-symmetry point | its name in the tables (`"GM"`, `"K"`) and its coordinates in the model's own reduced cell | `high_symmetry_points`; `kpoints` on an `HSPTable` |
| tabulated points a layer's zone does not have | `out_of_plane` | `high_symmetry_points(group).out_of_plane` |

### What the content says

| what it is | its name | where |
|---|---|---|
| whether a band set could be an atomic limit | `decomposable`: `True` or `False` | `decompose` |
| the sums of elementary band representations with the band set's symmetry data | `candidate_sums`, each a `BandRepresentation`, its terms at their Wyckoff positions | `decompose` |
| what a vector that is not decomposable rests on | `certificate`: `"proof"` or `"none within the bound"` | `decompose` |
| the indicator group `X_BS` | `indicator_group` (`"Z6 x Z12"`) and its `divisors` | `symmetry_indicator` |
| a band set's indicator class | `values`, each modulo its divisor; `order`; `indicators`, in the names of the stable-RSI paper ([citing](../citing.md#stable-real-space-invariants)) | `symmetry_indicator` |
| the stable real-space invariants | `theta_Z`, the `ℤ`-valued ones; `theta_Zn`, the `ℤ_n`-valued ones; each a mapping by name | `srsi_of_bands`, `srsi_of_model`, `SRSITables.theta` |
| the site-symmetry vector | `p`, keyed `(site irrep, Wyckoff position)`, the position written `1a` | `SRSITables.theta` |
| an atomic insulator with a band set's symmetry data | a realization | `atomic_realizations`; `valence` and `conduction` on a splitting |
| a splitting no symmetry indicator sees | `nsit`: `True` or `False` | `splitting`, `nsit_criterion` |
| pairs of realizations whose invariants add up | `consistent_pairs` | `splitting` |
| site irreps the stable invariants do not distinguish | a `StablePair` | `SRSITables.stably_equivalent`; `stably_equivalent` |

### A numerical model

| what it is | its name | where |
|---|---|---|
| the model's own relative precision | `precision`, and the tolerances it derives | the `precision=` argument; `r.numerical.precision` |
| how far the orbital centres miss the group's own action, and their correction | `defect`, `adjusted`, `max_move`, `offset`, `frame` | `r.numerical.positions` |
| an operation measured against a declared tolerance | `"exact"`, `"near"` or `"broken"` | `r.numerical.symmetrization`, whose `exact` and `near` hold the first two |
| how far the symmetrization moved `H` | `defect`, in the units of `H`; `scale`, the size of `H` it is read against | `r.numerical.symmetrization` |

### A refusal

| what it is | its name | where |
|---|---|---|
| the check that stopped a call | `gate` | every `Refusal` |
| what it measured, and what the check demanded | `value` and `threshold` | every `Refusal` |
| the diagnosis, and what is needed to act on it | `reason` and `context`; `context["group"]` or `context["number_str"]` once the group is identified | every `Refusal` |

## Words the pages use

**Assignment.** Where the orbitals sit: `r.wyckoff_orbits`, a row per Wyckoff
orbit, placing each orbital on its site and Wyckoff position. What the orbitals carry there is
a labelling.

**Atomic limit.** A band set carried by exponentially localized, symmetric
Wannier functions: the band representation orbitals at Wyckoff positions
induce. [The primer](../background/tqc-primer.md#band-representations-and-atomic-limits);
[atomic limits](../guide/atomic-limits.md).

**Band representation; elementary band representation (EBR).** The
representation of the space group on the Bloch states of orbitals at a Wyckoff
position, induced from the site irrep they carry. It is elementary when it is no
sum of other band representations. A site irrep at a maximal Wyckoff position
usually induces an elementary one; the exceptions induce a composite band
representation and are named as exceptions. A summary writes an elementary one
as its site irrep at its Wyckoff position, `-E1 at 1a (6mm1')`, and the records of
the stable real-space invariants as `1 (-E1)_a`; a decomposition and the
atomic-limit table hold it as a term, its parts as fields
([the band-representation key](../background/conventions.md#the-band-representation-key)).
[The primer](../background/tqc-primer.md#band-representations-and-atomic-limits).

**BCS.** The Bilbao Crystallographic Server, whose tables `irreptables` ships;
never Bardeen–Cooper–Schrieffer here. How its labels read across:
[conventions](../background/conventions.md).

**Cell.** The point set the group is found on: the orbitals' positions and any
species a declaration adds. Its group is an upper bound on the Hamiltonian's.

**Conserved operator.** A matrix that commutes with `H(k)` at every `k`.
Each one beyond the identity is counted as `additional commuting matrices`.

**Corep.** A co-representation, the irreducible object of a group with
antiunitary operations. Where time reversal glues two unitary irreps, the
level carries their sum and the name spells both (`-M3M4`); a leading `-`
marks a double-valued, spinful label.

**Declaration.** A statement about the crystal:
`typat=` (which sites are the same atom), `characters=` (what each site's
orbitals carry), and `decorations=`, `environment=` and `spin=`. The
Hamiltonian is measured against the crystal they define, and a declaration is
never second-guessed: a `typat=` or `characters=` the hoppings are not
covariant under describes a crystal with less symmetry, and the group descends
to it. `characters=` states each site's content, not how its orbitals are
partitioned inside the site.
[Saying what the orbitals are](../guide/orbitals.md).

**Decoupled stack.** A layer repeated along its normal with no hopping between
copies: a three-dimensional crystal whose `H(k, k_z)` does not depend on
`k_z`, and which obeys exactly what the layer obeys. Where a layer's group has
rows out of its plane that duplicate none in it, `symmetry_indicator` measures
them on this stack, in the layer's own group; a layer's class is the class of
its decoupled stack.
[A layer's out-of-plane rows](../background/measurement.md#a-layers-out-of-plane-rows).

**Descent.** The cell lowered to a group of the operations that hold, and
identified again; `r.descent`. A **reduction** is the cell lowered by declared
content before any hopping is read, which is not a descent and is on
`r.decoration.declared`.

**Elementary divisor; Smith normal form.** The EBR matrix, a row per corep at
each high-symmetry point and a column per EBR, is brought to a diagonal `D` by
integer matrices with integer inverses, `U·M·V = D`; the diagonal entries are
its elementary divisors, and those above 1 make the indicator group.
[How a symmetry indicator is computed](../background/tqc-primer.md#how-a-symmetry-indicator-is-computed).

**Environment.** What lies under a layer, which a two-dimensional model cannot
say: **on a substrate**, every operation that turns the layer over is absent;
**free-standing**, the full layer group applies. A two-dimensional model is
read on a substrate unless `environment=` says otherwise; a layer written with
three spatial coordinates, whose geometry says which it is, is read
free-standing. [A 2D model](../guide/layers.md).

**Fragile.** Not an atomic limit, and an atomic limit once suitable atomic
bands are added. Its indicator class is zero and its `ℤ`-valued invariants are
integral, so its symmetry data shows it only where that data has no
nonnegative realization; where it has one, as for the pair of lin3g bands in
[atomic limits](../guide/atomic-limits.md#what-a-decomposition-cannot-certify),
only a finer measurement, such as a Wilson loop, does.
[The primer](../background/tqc-primer.md#fragile-topology).

**Grey group.** A space group with time reversal adjoined, written with a
trailing `1'` (`P6/mmm1'`) and filed under its magnetic-group number,
`191.234`, where the group without time reversal is `191`. Its irreducible
objects are coreps.

**High-symmetry point.** A tabulated `k` whose little group the tables
carry; `GM` is `Γ`.

**Indicator group, indicator class.** `X_BS = {BS}/{AI}`, the band structures
the compatibility relations allow modulo the atomic limits, `⊕ᵢ ℤ_{dᵢ}` with
the `dᵢ` the elementary divisors of the EBR matrix above 1. A band set's class
in it is its symmetry indicator: nonzero, no band set with that content is an
atomic limit; zero certifies nothing.
[How it is computed](../background/tqc-primer.md#how-a-symmetry-indicator-is-computed);
[symmetry indicators](../guide/indicators.md).

**Inversion.** The three-dimensional `r → −r`, squaring to `E` and never to `Ē`
on spinors, or its restriction to a layer group. The twofold rotation `C2z` is
a rotation, and a wallpaper group holds no inversion.

**`k`-action class.** Operations that act alike on `k`, unitary or
antiunitary alike. One measurement settles the whole class, so on a flat layer
an operation and its product with the horizontal mirror share a verdict, and
an operation kept that shares its class reads `undetermined`.

**Labelling.** The hoppings fix the model's representation only up to a
one-dimensional character of the group on each part of the crystal they couple,
the orbitals they connect directly or through others. A labelling is one
choice of those characters, and with it of what each orbital carries. A
declaration fixes it, and the lowest-ℓ rule makes it where the declarations
leave it open. Labellings that give every orbital the same site irrep give the
whole set of bands the same labels together, while a subset of the bands, or
one energy multiplet, can read differently under each, and an `hsp_table`'s
`note` says where a band set's labels rest on the one returned.
[What the hoppings fix](../background/measurement.md#what-the-hoppings-fix).

**Little group.** The operations that leave `k` invariant up to a reciprocal
lattice vector.

**Lowest-ℓ rule.** What the hoppings and the declarations leave open about the
orbitals is assigned the lowest-angular-momentum representation they allow, read at the reference
orbital's site first; a row it decides reads `lowest-l`.
[The rule and its keys](../background/measurement.md#the-lowest-angular-momentum-rule).

**Maximal Wyckoff position.** A Wyckoff position whose site-symmetry group lies
in no larger site-symmetry group of the crystal.
[The primer](../background/tqc-primer.md#site-symmetry-and-wyckoff-positions).

**Multiplet.** Two things, named apart. A multiplet of **bands** is a set of
bands degenerate at one `k`, one row of a character table. A multiplet of
**orbitals** is the orbitals of one site that the Hamiltonian binds into one
carrier of a site irrep.

**Near-symmetric.** An operation obeyed within the declared tolerance but not to
the tolerance of the exact check the representation is built with; the signal
on which the Hamiltonian is projected onto its symmetric part.
[Numerical models](../guide/numerical.md).

**NSIT.** Non-symmetry-indicated topological: a splitting whose two sides each
have atomic realizations, no pair of which adds up to the whole's `ℤ_n`-valued
invariants, so the valence set, the conduction set, or both is not an atomic
insulator, and no symmetry indicator says so. A side with fractional invariants,
or with integral ones and no realization at all, is seen from its symmetry data
and is not NSIT. The stable-RSI paper, Hwang *et al.*, Nat. Commun. **17**, 8564
(2026), which tabulates the invariants
([citing](../citing.md#stable-real-space-invariants)), names it.

**Orbit.** The sites the group's operations carry one site onto, all at one
Wyckoff position.

**Orbital function.** A word such as `s`, `px` or `dxy`: shorthand for the
site-symmetry representation that function carries. Only characters and irreps
are real: how a set of Wannier functions is turned inside its site is a choice
of basis, so a function is a declaration's word, echoed in the atomic-limit
table's `function` column, and never a measurement; where nothing is declared,
the lowest-ℓ rule assigns the lowest-ℓ representation the hoppings allow.
[What the hoppings fix](../background/measurement.md#what-the-hoppings-fix).

**Precision.** The relative precision of a numerical model's own numbers, the
one declaration a fitted model makes about itself; every tolerance it needs is
derived from it.

**Pseudospin, conserved.** Identical copies of a part of the crystal that
nothing joins, which the operators the Hamiltonian conserves mix freely, so
that every band on those orbitals is as many times degenerate as there are
copies and nothing measured says what the copies carry. The copies are one
multiplet at each site, with a spin or without; a declaration states what they
carry, and the lowest-ℓ rule assigns it otherwise, the copies read copy by copy
first. With a spin they carry an ordinary representation and the spin its own
lift.
A row of `r.wyckoff_orbits` whose orbitals hold a conserved pseudospin lists the
site irreps its copies can carry in `numerical.contents`, and
`numerical.n_representations` counts the representations of the copies.

**Real-space invariant.** A combination of the multiplicities of site irreps at
Wyckoff positions that no symmetric deformation of the Wannier functions can
change: the functions move only in symmetric clusters, and each move changes
the multiplicities by a fixed pattern.
[What a real-space invariant is](../guide/srsi.md#what-a-real-space-invariant-is).

**Realization.** An atomic insulator with a band set's symmetry data: a
nonnegative number of copies of each site irrep at each maximal Wyckoff
position. Every realization is enumerated, so finding none is a proof.

**Reference orbital.** The orbital the lowest-ℓ rule reads first, orbital 0.

**Refusal.** A returned, falsy record, of one of four kinds: a verdict about
the physics, a practical limit, a grammar error in a declaration or a question,
or one of the package's checks on itself, which means the case is to be
reported. A disagreement inside the package raises `InternalError`.
[The four kinds](../guide/refusals.md#four-kinds).

**Returned.** What a call gives back.

**Set of bands.** The bands a call reads, named by `bands=`. A set of bands has
to be closed under degeneracy at every point it is read at; one that cuts a
degenerate multiplet is refused.

**Sewing matrix.** `B(g,k)_mn = ⟨u_m(k)| U(g;k) |u_n(k)⟩`, the matrix of an
operation in the basis of the bands selected at one invariant `k`. Its trace
over a degenerate multiplet is that multiplet's character.

**Site irrep; site-symmetry group.** The site-symmetry group of a site is the
operations that fix it. A site irrep is an irreducible representation of it;
for a spinful model, a double-valued site corep.

**Site-symmetry vector.** The site-symmetry vector `p` counts how many times
each site irrep occurs at each Wyckoff position, maximal or not: one entry per
pair (site irrep, Wyckoff position), the position written with its
multiplicity, `1a`.

**Slab cell.** A layer set in a three-dimensional cell, its non-periodic
direction vacuum. A layer's group is that slab's space group.

**Stable real-space invariant (SRSI).** A real-space invariant written as a
linear form `θ = Θ·p` over the whole site-symmetry vector, every site irrep at
every Wyckoff position at once. Together they classify atomic configurations up
to stable equivalence: two configurations share every SRSI exactly when they
deform into each other once the same atomic orbitals are added to both. Some
are `ℤ`-valued and some `ℤ_n`-valued, and the tables are the stable-RSI
paper's. [Stable real-space invariants](../guide/srsi.md).

**Stable topological.** A band set whose `ℤ`-valued invariants are fractional:
no atomic insulator has its symmetry data, since a nonzero indicator class and
a fractional `ℤ`-valued invariant are the same statement.

**Symmetry-indicated decomposable.** `decompose`'s `decomposable`: the measured
content at the high-symmetry points is a nonnegative-integer sum of elementary
band representations. Necessary for an atomic limit, never sufficient.

**Symmetry data vector.** How many times each corep occurs at each high-symmetry
point in a band set.

**Tie rule.** The one rule that picks among subgroups of the largest order the
measurement allows when nothing measured separates them.
[The descent, and the tie rule](../background/measurement.md#the-descent-and-the-tie-rule).

**Variety; closure.** A Wyckoff position's variety is the set of points it
covers in the cell: a point, or a line or plane where it has free parameters.
An orbital confined to a variety can slide along it, and the variety's closure
adds the more special positions it meets: the line `6c` of `R-3m`,
`(0, 0, z)`, closes on `3a` at `z = 0` and on `3b` at `z = 1/2`. A site's
`slides_onto`, and a Wyckoff orbit's in the SRSI records, name the positions
there at which the tables have elementary band representations, the positions
such an orbital can be moved to.

**Weak index.** In three dimensions, a component of an indicator class carried
by the high-symmetry points of one plane of the zone, such as `k_z = ½`: an
invariant of how layers are stacked. A single layer's class, read in a
three-dimensional group, is the class of its decoupled stack, and every
component of it is an invariant of that stack.
[A layer's class](../guide/indicators.md#a-layers-class-is-a-three-dimensional-class).

## Tolerances

A tolerance carries the unit it is measured in, and a number given in the wrong
unit is the one mistake that passes silently, since it is still a number. Each
is one of four kinds: an energy, a pure ratio, a reduced coordinate or a
Cartesian length.

| name | what it checks, and where | unit | default |
|---|---|---|---|
| `precision` | the model's own relative precision, one fact about its numbers, from which `tol = 10 × precision × max\|H\|`, `position_tol = precision`, `cut = tol / max\|H\|` and `window` from `cut` are derived, and `symprec` raised where the declared noise in the positions would defeat the symmetry finder (`realized_group`) | dimensionless | `None` |
| `tol` | the covariance check `max_k ‖U H(k) U† − H(Rk)‖`, which separates an operation the Hamiltonian obeys from one it breaks, and the declared tolerance a near-symmetric operation is read against (`realized_group`) | energy, in the units of `H` | `1e-8`, unless `precision` derives one |
| `cut` | the nullspace cut `s ≤ cut · s_max`, never stricter than `s ≤ √n · cut` read in the units of `H` for `n` states (`√n · tol` under `precision`), which decides what the measurement finds before any check applies (`realized_group`) | dimensionless, relative to the largest singular value | `tol`; `tol / max\|H\|` under `precision` |
| `window` | the matching window inside which two eigenvectors are paired at the first `k` (`realized_group`) | dimensionless, relative to `max\|H − μ\|`, `μ` the mean on-site energy | `clip(10·sqrt(tol), 1e-3, 0.3)`; `clip(10·sqrt(cut), 1e-3, 0.3)` under `precision` |
| `position_tol` | how far an orbital centre may miss where the group puts it, about the origin that fits the centres, and still be moved onto it, and how far a shift of every centre together may reach and still be removed, so that the centres are written in the model's own frame (`realized_group`); a variety's width (`wyckoff_sites_tables`) | reduced coordinates of the cell | `None`; `1e-4` in `wyckoff_sites_tables` |
| `symprec` | spglib's tolerance for finding the group from the positions (`realized_group`) | Cartesian length, in the model's units | `1e-5` |
| `degeneracy_tol` | which bands are one degenerate multiplet, and so the rows of every character table (`characters`, `coreps_at`, `hsp_table`, `sewing_matrix`, `multiplets`) | energy | `1e-8` |
| `label_tol` | how far a fitted corep multiplicity may sit from an integer (`coreps_at`, `hsp_table`) | dimensionless | `1e-3` |

The covariance check's tolerance a representation is built at is an energy, as
`tol` is: the larger of `5e-14` and `|G|·N·ε·max|H|`, the rounding a
representation of `|G|` operations on `N` states carries, which is the exact
class of [numerical models](../guide/numerical.md). No `precision`
handed to `realized_group` moves it, since those set the tolerance of the
measurement's check. Rounding never refuses a valid model; a miss above the
check's tolerance is a real one.

Every default of a keyword above is the signature's own, and these lines print them:

```python
import inspect
import pythtqc as tqc

required = inspect.Parameter.empty
for call, names in [(tqc.realized_group, "position_tol precision symprec"),
                    (tqc.wyckoff_sites_tables, "position_tol"),
                    (tqc.coreps_at, "degeneracy_tol label_tol"),
                    (tqc.hsp_table, "degeneracy_tol label_tol"),
                    (tqc.characters, "degeneracy_tol"),
                    (tqc.sewing_matrix, "degeneracy_tol"),
                    (tqc.multiplets, "degeneracy_tol")]:
    parameters = inspect.signature(call).parameters
    print(f"{call.__name__:<22}", ", ".join(
        f"{n}={'required' if parameters[n].default is required else parameters[n].default}"
        for n in names.split()))
#  realized_group         position_tol=None, precision=None, symprec=1e-05
#  wyckoff_sites_tables   position_tol=0.0001
#  coreps_at              degeneracy_tol=1e-08, label_tol=0.001
#  hsp_table              degeneracy_tol=1e-08, label_tol=0.001
#  characters             degeneracy_tol=1e-08
#  sewing_matrix          degeneracy_tol=1e-08
#  multiplets             degeneracy_tol=1e-08
```

## Budgets, meshes and lengths

Numbers that are not tolerances: a memory budget, a count, a mesh, or a length
that no verdict depends on.

| name | what it sets, and where | unit | default |
|---|---|---|---|
| `max_bytes` | the memory the symmetrizing projection may take at its peak, and the ceiling the measurement's own budget is raised to on exactly degenerate levels (`realized_group`) | bytes | `1e9` |
| `max_candidates` | the largest search for a representation's discrete choices that is run rather than refused (`realized_group`) | candidates | `2**20` |
| `cap` | how many candidate sums an enumeration lists before it says it was cut (`decompose`); how many realizations it may hold, past which it refuses (`atomic_realizations`) | sums; realizations | `200`; `256` |
| `nk` | divisions of the measurement's mesh per periodic direction (`realized_group`) | divisions | `6` |
| `c` | the vacuum between the stack's copies, its period the layer's extent along its normal plus this (`stack_presentation`) | the model's length units | `8.0` |

Nothing returned depends on `c`: no hopping crosses the
vacuum between a stack's copies. The defaults, as the signatures hold them:

```python
for call, names in [(tqc.realized_group, "max_bytes max_candidates nk"),
                    (tqc.decompose, "cap"),
                    (tqc.atomic_realizations, "cap"),
                    (tqc.stack_presentation, "c")]:
    parameters = inspect.signature(call).parameters
    print(f"{call.__name__:<22}", ", ".join(f"{n}={parameters[n].default}" for n in names.split()))
#  realized_group         max_bytes=1000000000.0, max_candidates=1048576, nk=6
#  decompose              cap=200
#  atomic_realizations    cap=256
#  stack_presentation     c=8.0
```
