# Glossary

This page defines each term the pages and the records use, in a sentence or
two, with a link where a page treats it. [Tolerances](#tolerances) gives
every tolerance with its unit, its default and the formula that derives it.
[The topological quantum chemistry this package uses](../background/tqc-primer.md)
and [How pythTQC reads a Hamiltonian](../background/measurement.md) give the
ideas at length, and [Returned records](records.md) gives every field of every
record.

## The names on the records

Each table pairs a thing with the field that holds it, and says where the field
is found.

### The group

| what it is | its name | where |
|---|---|---|
| the space group the Hamiltonian obeys | `name`, its Hermann-Mauguin symbol (`"P6/mmm1'"`); `number_str`, the number the tables file it under (`"191.234"`) | `r.group` |
| an operation | `index`, its number in `r.operations`; `seitz`, its Seitz symbol `{R\|t}` in the conventional cell used by the irreptables package; `name`, its name in the Cartesian axes in which the pythtb Lattice is specified, `C3(z)` or `m(x)`, with a trailing `'` where it is combined with time reversal | `r.operations`; `operation_index` |
| whether each operation holds | `obeyed`, `broken` and `undetermined`, the operations by index | `r.numerical.validated` |
| the group of the cell, and the group the descent ends at | `cell_group` and `subgroup` | `r.descent` |
| what broke an operation | `refuted_by`, on each entry of `dropped` | `r.descent` |
| what a declaration lowered the cell's group by before any hopping was read | `declared` | `r.decoration` |
| what lies under a layer | `kind`, `"substrate"` or `"free-standing"`; `declared`, whether `environment=` gave it | `r.environment`; the `environment=` argument |
| the group read in the other environment | `cross_check` | `r.environment` |
| time reversal, as measured | `outcome`, `"kept"` or `"measured broken"` | `r.time_reversal` |
| the matrices beyond the identity that commute with `H(k)` at every `k` | `dim`, the number of linearly independent ones, printed `additional commuting matrices` | `r.conserved` |
| the sectors into which the conserved matrices that act within sites split the states | `sectors`, each with its `orbitals` and its `sites`, and holding `copies` identical copies of a set of `dim` states; `images`, the sector each operation carries each sector onto | `r.conserved` |

### The orbitals

| what it is | its name | where |
|---|---|---|
| the Wyckoff position | `wyckoff`, its multiplicity and letter (`"6c"`) | each row of `r.wyckoff_orbits`; each entry `wyckoff_sites_tables` returns; each row of `srsi_of_model` |
| the sites the group's operations carry onto one another, all at one Wyckoff position | a Wyckoff orbit: a row, with its `orbitals` and its sites' positions, `numerical.positions` on the row | `r.wyckoff_orbits` |
| the site-symmetry group | `site_symmetry`, its Hermann-Mauguin symbol (`"-42m1'"`); `unitary_site_symmetry`, the same without time reversal (`"-42m"`) | each row of `r.wyckoff_orbits` |
| the orbital function a declaration names | `declared`, the word as `characters=` takes it; `None` on a row nothing was declared for | each row of `r.wyckoff_orbits` |
| the site irrep | `label`; `numerical.declared_irrep`, the site irrep each declaration resolved to | each row of `r.wyckoff_orbits` |
| how a site irrep is known | `determined_by`: `declaration`, `declaration not carried`, `hamiltonian` or `lowest-l` | each row of `r.wyckoff_orbits` |
| where its name comes from | `numerical.named_by`: `tables`, the tables of elementary band representations (EBRs); `invariants`, the tables of the stable real-space invariants; `point group`, `Mulliken` or `unnamed` | each row of `r.wyckoff_orbits` |
| the orbitals of a site that the site's operations mix | a multiplet: `site_multiplets`, the multiplets of each site of the row | each row of `r.wyckoff_orbits` |
| one choice of the one-dimensional characters the hoppings leave open | a labeling; `determined_by` says which rows rest on it, and `note` which sets of bands have little-group representations that rest on it | each row of `r.wyckoff_orbits`; what `hsp_table` returns |
| the site irreps assigned to the copies of a conserved pseudospin | `label`, the one assigned; `numerical.contents`, every sum of site irreps the copies can carry; `numerical.n_representations`, how many representations of the site-symmetry group on the copies its table admits | each row of `r.wyckoff_orbits` |
| the band representation a Wyckoff orbit induces | `band_representation`, with its `terms`, the EBRs it is the sum of | each row of `r.wyckoff_orbits` |
| the matrix of each operation on the orbitals | `matrices`, keyed by the operation's index | `r.rep` |
| a declared pairing of spin partners in a spinless model | `spin`, a `SpinBasis` | `r.spin`; the `spin=` argument |

### The bands

| what it is | its name | where |
|---|---|---|
| the bands asked for | `bands=`, an integer for the lowest that many or a sequence of band indices; `bands`, the indices themselves | every call that reads bands, and its record |
| a degenerate multiplet of bands at one `k` | an entry of `multiplets`, with its `bands`, its `energy`, and its `characters` or its `corep_names` and `label` | what `characters`, `coreps_at` and `hsp_table` return |
| the little-group representations of a set of bands at one point | `coreps`, `{label: multiplicity}` over the set; `corep_names` and `label` on each entry of `multiplets` | what `coreps_at` returns |
| the symmetry data vector of a set of bands | `symmetry_data`, `{label: multiplicity}` over every high-symmetry point | what `hsp_table` returns |
| the gap at the point from a set of bands to the nearest band outside it | `isolation`, in the units of H | what `coreps_at` returns |
| a high-symmetry point | its name in the tables (`"GM"`, the tables' name for Γ; `"K"`), and its reduced coordinates in the model's reciprocal lattice | `high_symmetry_points`; `kpoints` on what `hsp_table` returns |
| the tabulated points a layer's zone does not have | `out_of_plane` | `high_symmetry_points(group).out_of_plane` |

### What the symmetry data say

| what it is | its name | where |
|---|---|---|
| whether a symmetry data vector has a decomposition | `decomposable`, `True` or `False` | what `decompose` returns |
| every decomposition found | `candidate_sums`, each a `BandRepresentation`, its terms at their Wyckoff positions | what `decompose` returns |
| what the absence of a decomposition rests on | `certificate`, `"proof"` or `"none within the bound"` | what `decompose` returns |
| the indicator group | `indicator_group` (`"Z6 x Z12"`), and the elementary divisors above 1 of the EBR matrix, `divisors` | what `symmetry_indicator` returns |
| the symmetry indicator of a set of bands | `values`, one integer modulo each divisor; `order`, the order of the class in the indicator group; `indicators`, the values of the indicators Hwang *et al.* name ([Stable real-space invariants](../citing.md#stable-real-space-invariants)) | what `symmetry_indicator` returns |
| the stable real-space invariants (SRSIs) | `theta_Z`, the integer-valued SRSIs (ZSRSIs); `theta_Zn`, the SRSIs valued modulo n; each a mapping by name | what `srsi_of_bands` (`theta_Z` alone) and `srsi_of_model` return; `SRSITables.theta` |
| the site-symmetry vector | `p`, `{(site irrep, Wyckoff position): multiplicity}`, the position written with its number of sites in the conventional cell, `"2c"` | the argument of `SRSITables.theta` |
| a configuration of site irreps at maximal Wyckoff positions with the ZSRSIs of a set of bands | an atomic realization | what `atomic_realizations` returns; `valence` and `conduction` on a splitting |
| whether the no-go criterion finds a splitting non-symmetry-indicated topological (NSIT) | `nsit`, `True` or `False` | what `splitting` and `nsit_criterion` return |
| the pairs of realizations whose SRSIs valued modulo n add up to the whole model's | `consistent_pairs` | what `splitting` returns |
| two site irreps no SRSI distinguishes | a `StablePair` | `SRSITables.stably_equivalent`; `stably_equivalent` on a splitting |

### A numerical model

| what it is | its name | where |
|---|---|---|
| the precision declared, and the tolerances derived from it | `precision`, with `max_h`, `tol`, `cut`, `window`, `position_tol`, `symprec` and `symprec_raised` | `r.numerical.precision`; the `precision=` argument |
| how far the orbital centers missed the positions the group's operations put them at, and their move onto those positions | `defect_before`, `defect_after`, `max_move`, `offset`, `frame`, `move`, `carried` | `r.numerical.positions` |
| the class of an operation against the `tol` in force | `"exact"`, `"near"` or `"broken"` | `r.numerical.symmetrization`, whose `exact` and `near` list the first two, where `H` was projected; `None` otherwise |
| how far the projection onto the symmetric part moved `H` | `defect`, in the units of H; `scale`, the size of `H` it is read against | `r.numerical.symmetrization`, where `H` was projected; `None` otherwise |

### A refusal

| what it is | its name | where |
|---|---|---|
| the check that stopped a call | `gate` | every `Refusal` |
| what the check measured, and what it compared it with | `value` and `threshold` | every `Refusal` |
| what was found and what to do, and the objects it names | `reason` and `context` | every `Refusal` |
| the group identified | `context["group"]`; `context["number_str"]` where a refusal carries only its number | a `Refusal` that carries the group: one `realized_group` returns once the group is identified, and `not_a_band_structure`, `indicator_not_evaluable`, `restriction_not_exact` and `unknown_label` from `decompose`, `symmetry_indicator` and `srsi_of_bands` |

## Words the pages use

**Atomic limit.** A set of bands whose states can be written as exponentially
localized, symmetric Wannier functions. Its symmetry data vector is a sum of
the symmetry data vectors of elementary band representations, with
nonnegative integer multiplicities.
[What elementary band representations are](../guide/atomic-limits.md#what-elementary-band-representations-are).

**Atomic realization.** A configuration of site irreps at maximal Wyckoff
positions, with nonnegative integer multiplicities, that has the ZSRSIs of a
set of bands. `atomic_realizations` lists every one, and refuses
`srsi_realizations_truncated` where there are more than `cap=`.
[The no-go criterion](../guide/srsi.md#the-no-go-criterion).

**Band representation; elementary band representation (EBR).** The
representation of the space group on the Bloch states of the orbitals at a
Wyckoff position, induced from the site irrep they carry. It is elementary
when it is not a sum of other band representations. A site irrep at a maximal
Wyckoff position induces an EBR, but for a few exceptions, whose band
representations are sums of EBRs. A summary writes an EBR as its site irrep at
its Wyckoff position, `-E1 at 1a (6mm1')`, and the records of the SRSIs write
it with its multiplicity, `1 (-E1)_a`
([The band-representation key](../background/conventions.md#the-band-representation-key)).
[What elementary band representations are](../guide/atomic-limits.md#what-elementary-band-representations-are).

**BCS.** The Bilbao Crystallographic Server, whose tables of little-group
representations the irreptables package ships.
[Reading across to BCS](../background/conventions.md#reading-across-to-bcs).

**Cell.** The set of points the group is found on: the orbital centers, each
with its species, and the points the package or a declaration adds, the
substrate under a layer and those of `decorations=`. spglib finds its space
group at the tolerance `symprec`, and time reversal is adjoined to it. The
group of `H` is a subgroup of that group.
[The cell, and a verdict on each operation](../background/measurement.md#the-cell-and-a-verdict-on-each-operation).

**Conserved operator.** A matrix that commutes with `H(k)` at every `k`. The
number of linearly independent ones beyond the identity is printed as
`additional commuting matrices`.
[A conserved operator](../guide/group.md#a-conserved-operator).

**Corep.** An irreducible co-representation of a group that holds antiunitary
operations. Where time reversal is kept, a set of bands carries coreps of the
little group of `k`. Where time reversal joins two irreps into one corep, its
label names both, `-M3M4`. A leading `-` marks a double-valued one.
[Little-group representations at a k-point](../background/conventions.md#little-group-representations-at-a-k-point).

**Covariance residual.** How far a matrix `U` for an operation misses carrying
`H(k)` onto `H(Rk)`: the largest modulus of an entry of `U H(k) U† − H(Rk)`
over the `k`-points measured, in the units of H. For an operation combined
with time reversal, `H(k)*` stands in place of `H(k)`. An operation holds
where the residual is at most `tol`.
[Exact, near and broken](../guide/numerical.md#exact-near-and-broken).

**Declaration.** A statement about the crystal that `H(k)` does not contain:
`typat=`, which sites are the same atom; `characters=`, the site irrep each
orbital carries; `decorations=`, `environment=` and `spin=`. The Hamiltonian
is measured against the crystal the declarations define, and a declaration is
never checked against the Hamiltonian. Where the hoppings are not covariant
under what a `typat=` or `characters=` states, the declaration states a
crystal with less symmetry, and the group returned is the one `H` obeys in
that crystal. A word in `characters=` states the site irreps a site's orbitals
carry, and not the basis they are written in. Generator matrices, a mapping in
`characters=` from an operation of the site to the matrix it acts with on the
orbitals, state the basis too
([Which basis a doublet is](../guide/orbitals.md#which-basis-a-doublet-is)).
[How do I say what my orbitals are?](../guide/orbitals.md).

**Decomposition.** A sum of the symmetry data vectors of EBRs, with
nonnegative integer multiplicities, equal to the symmetry data vector of a set
of bands. An atomic limit has one, and some topological sets of bands have one
too. `decompose` returns every one, up to `cap=`.
[Could these bands be an atomic limit?](../guide/atomic-limits.md).

**Decoupled stack.** A layer repeated along its normal with no hopping between
the copies, each orbital at its own height: a three-dimensional crystal whose
bands do not depend on `k_z`, and which obeys exactly what the layer obeys. A
layer's EBR matrix has rows at `k_z = 1/2`, where the layer has no `k`-points.
Where such a row occurs as often as a row in the plane in every EBR of the
layer, the bands' multiplicity on it is that row's. Where no row in the plane
does, `symmetry_indicator` reads the multiplicity at `k_z = 1/2` off the
decoupled stack, in the layer's own group. A layer's symmetry indicator is the
symmetry indicator of its decoupled stack.
[A layer's out-of-plane rows](../background/measurement.md#a-layers-out-of-plane-rows);
[A layer's class is a three-dimensional class](../guide/indicators.md#a-layers-class-is-a-three-dimensional-class).

**Descent; reduction.** The descent lowers the cell's group to a subgroup
whose operations all hold, of the largest order the measurement allows.
`r.descent` records it. A reduction lowers the cell's group by what a
declaration states, before any hopping is read, and `r.decoration.declared`
records it.
[The descent, and the tie rule](../background/measurement.md#the-descent-and-the-tie-rule);
[A declaration that lowers the group](../guide/orbitals.md#a-declaration-that-lowers-the-group).

**Double group.** The group that holds each operation of a space group
together with its product with `Ē`, the rotation by 2π, which acts as −1 on
a spin one-half. The operations a record lists are the single-valued group's,
each standing for itself and its negative in the double group.

**Elementary divisor; Smith normal form.** The EBR matrix has a row for each
little-group representation at each high-symmetry point, and a column for each
EBR, its symmetry data vector. Integer matrices `U` and `V` with integer
inverses bring the EBR matrix `M` to a diagonal matrix, `U M V = D`, its Smith
normal form. The diagonal entries of `D` are its elementary divisors, and
those above 1 give the indicator group.
[What a symmetry indicator is](../guide/indicators.md#what-a-symmetry-indicator-is);
[How a symmetry indicator is computed](../background/tqc-primer.md#how-a-symmetry-indicator-is-computed).

**Environment.** What lies under a layer, which a model periodic in two
directions does not state. On a substrate, every operation that turns the
layer over is absent. Free-standing, the whole layer group applies. Unless
`environment=` says otherwise, a layer with two-dimensional positions is read
on a substrate, and a slab, a layer with three-dimensional positions
(`dim_r = 3`), is read free-standing.
[Is my 2D model on a substrate or free-standing?](../guide/layers.md).

**Exact, near and broken.** The three classes of an operation of the cell, by
its covariance residual. An exact operation holds to rounding, within the
rounding tolerance of [Tolerances](#tolerances). A near operation holds within
`tol` and not to rounding, and `H` is projected onto its symmetric part. A
refusal's reason calls such an operation near-symmetric. A broken operation
misses by more than `tol`, and is dropped.
[Exact, near and broken](../guide/numerical.md#exact-near-and-broken).

**Fragile.** A set of bands that is not an atomic limit, and becomes one once
suitable atomic bands are added. Its symmetry indicator is zero. Its symmetry
data vector can have a decomposition, as that of bands 4 and 5 of lin3g, from
`pythtqc.models`, does
([What a decomposition cannot certify](../guide/atomic-limits.md#what-a-decomposition-cannot-certify)).
[Fragile topology](../background/tqc-primer.md#fragile-topology).

**Grey group.** A space group with time reversal adjoined, written with a
trailing `1'` (`P6/mmm1'`), and filed under its number as a magnetic space
group, `191.234`, where the group without time reversal is `191`. The
little-group representations of its bands are coreps.

**Hamiltonian's own blocks.** The matrices of an operation on the orbitals of
a site that the hoppings fix, under the labeling returned. A row reads
`declaration not carried` where no labeling the hoppings leave open carries a
declaration: declared generator matrices that no labeling gives the declared
orbitals, or a declaration that no labeling carries together with the others.
The matrices returned on such a row are the Hamiltonian's own blocks.
[How do I say what my orbitals are?](../guide/orbitals.md#how-to-read-it).

**High-symmetry point.** A point of the Brillouin zone at which the tables of
the irreptables package list the little-group representations.
`high_symmetry_points` gives each one's name and its reduced coordinates.

**Indicator group; symmetry indicator.** The indicator group is
`X_BS = {BS}/{AI}`: the group `{BS}` of integer vectors that satisfy the
compatibility relations, modulo its subgroup `{AI}` of integer combinations of
the symmetry data vectors of the EBRs. It is `Z_d1 x Z_d2 x ...`, with
`d1, d2, ...` the elementary divisors of the EBR matrix above 1. The symmetry
indicator of a set of bands is the class of its symmetry data vector in it, one
integer modulo each divisor. A set of bands with a nonzero class is not an
atomic limit, and does not become one when atomic bands are added. A set with
a zero class may be an atomic limit, fragile or stably topological.
[Which symmetry-indicator class do these bands occupy?](../guide/indicators.md).

**Inversion.** The operation `r → −r`, `i`, whose spin rotation is the
identity, so that `i² = E`. A flat layer's group holds it only free-standing,
as `i = m(z)·C2(z)`, where `C2(z)` and the horizontal mirror hold. `C2(z)` is
a rotation, whose spin rotation squares to `Ē`, though it acts on in-plane `k`
as inversion would.
[Inversion in a layer](../guide/layers.md#inversion-in-a-layer).

**Irrep.** An irreducible representation. Where time reversal is broken, a set
of bands carries irreps of the little group of `k`.

**`k`-action class.** The operations that act alike on `k`, and alike in
whether they are combined with time reversal. `H(k)` does not tell the members
of a class apart, so a class is measured once. A class found broken breaks
every member. A class kept with more than one member leaves each member
`undetermined` in `r.numerical.validated`. On a free-standing layer an
operation and its product with the horizontal mirror are one class.

**Label.** The name of a corep or an irrep, as the tables of the irreptables
package write it: `-GM6`, `-M3M4`, `A1`.
[What the little-group representations of a set of bands are](../guide/labels.md#what-the-little-group-representations-of-a-set-of-bands-are).

**Labeling.** The hoppings fix the site irreps of the orbitals they couple only
up to a one-dimensional character of the group. One such character is left
open on each coupled part of the crystal, a set of orbitals the hoppings
connect directly or through others. A labeling is one choice of those
characters. A declaration fixes it, and the lowest-angular-momentum rule
chooses it where the declarations leave it open. Two labelings that give each
orbital the same site irrep give the same degeneracies, and the same symmetry
data vector of all the model's bands together.
Under another such labeling, a part of the bands can carry other little-group
representations, and the `note` of what `hsp_table` returns says so.
[When to declare](../guide/orbitals.md#when-to-declare);
[What the hoppings fix](../background/measurement.md#what-the-hoppings-fix).

**Little group.** The operations that leave `k` invariant up to a reciprocal
lattice vector.

**Little-group representation.** A corep or an irrep of the little group of
`k`: a corep where time reversal is kept, and an irrep where it is not. The
states of a set of bands at `k` carry a sum of them.
[What the little-group representations of a set of bands are](../guide/labels.md#what-the-little-group-representations-of-a-set-of-bands-are).

**Lowest-angular-momentum rule.** The rule that assigns what the declarations
and the hoppings leave open about the site irreps of the orbitals. The site
irreps at the reference orbital's site are taken from the lowest angular
momentum whose real harmonics carry them all, `s` before `p` before `d`. The
site irreps of every orbital the hoppings couple to it then follow from the
hoppings. A row the rule decides reads `lowest-l` in the `determined by`
column.
[The lowest-angular-momentum rule](../background/measurement.md#the-lowest-angular-momentum-rule).

**Maximal Wyckoff position.** A Wyckoff position whose site-symmetry group lies
in no larger site-symmetry group of the crystal.
[Site symmetry and Wyckoff positions](../background/tqc-primer.md#site-symmetry-and-wyckoff-positions).

**Multiplet.** Two things, named apart. A multiplet of bands is a set of bands
degenerate at one `k`, within `degeneracy_tol`. `characters` returns its trace
of each unitary operation of the little group. A multiplet of orbitals is a
set of orbitals of one site that the operations fixing the site mix.

**NSIT.** Non-symmetry-indicated topological. A splitting of a model's bands
is NSIT where both sides have atomic realizations, and no pair of
realizations, one of each side, has SRSIs valued modulo n that add up to those
of all the bands. One side or both is then not an atomic limit. No symmetry
indicator says so, since the ZSRSIs of both sides take integer values. The
name is that of Hwang *et al.*
[The no-go criterion](../guide/srsi.md#the-no-go-criterion).

**Orbit; Wyckoff orbit.** The sites the group's operations carry one site onto,
all at one Wyckoff position. `r.wyckoff_orbits` holds a row for each.

**Orbital function.** A word such as `s`, `px` or `dxy`, which `characters=`
takes as shorthand for the site irrep that function carries at the site. The
atomic-limit table prints it in its `declared` column. A function is never
measured: the hoppings fix the site irreps of the orbitals, not the basis
inside a multiplet.
[What the hoppings fix](../background/measurement.md#what-the-hoppings-fix).

**Precision.** How far the numbers of a numerical model may miss the symmetry
of the crystal they approximate, declared with `precision=`: relative to
`max|H|` for the hoppings, and in reduced coordinates for the orbital centers.
`tol`, `cut`, `window` and `position_tol` are derived from it
([Tolerances](#tolerances)).
[How do I measure a numerical or Wannier90 model?](../guide/numerical.md).

**Pseudospin, conserved.** Identical copies of a part of the crystal that no
hopping joins. Every band on those orbitals is as many times degenerate as
there are copies, and the copies are one multiplet at each site. Nothing
measured says which site irreps the copies carry. A declaration states them,
and otherwise the lowest-angular-momentum rule assigns them. The row of such
an orbit lists in `numerical.contents` every sum of site irreps the copies can
carry.
[More than one character left open](../background/measurement.md#more-than-one-character-left-open).

**Real-space invariant.** A combination of the entries of the site-symmetry
vector that no smooth deformation preserving the symmetry changes.
[What the stable real-space invariants are](../guide/srsi.md#what-the-stable-real-space-invariants-are).

**Reference orbital.** Orbital 0, whose site the lowest-angular-momentum rule
reads first.

**Refusal.** A falsy record a call returns in place of its record where it
cannot complete. It names the check that stopped the call and what the check
measured, and its reason says what was found. Every refusal is one of four
kinds:

- a verdict about the physics: what was asked does not exist for these bands,
  or for this model as written;
- a practical limit: a search, an enumeration, a measurement or a projection
  would need more than its ceiling allows, or the installation lacks the
  tables a question needs;
- a grammar error in a declaration or a question: a declaration the orbitals
  it is written on cannot hold, or an argument a call cannot read;
- one of the package's checks on itself: an identity that holds in exact
  arithmetic fails, or two measurements of one thing disagree, in a case the
  package does not handle. Nothing in the model is to change, and the refusal
  is worth reporting.

An `InternalError` is not a refusal. It is raised where the package disagrees
with itself, and is a defect in the package.
[What does a refusal mean, and what do I do about it?](../guide/refusals.md);
[Refusals](refusals.md).

**Set of bands.** The bands a call reads, named by `bands=`: an integer for the
lowest that many, or a sequence of band indices. Little-group representations
can only be returned for a set that contains all the states in degenerate
multiplets at each point it is read at. A set that cuts a multiplet is refused
with `multiplet_split`.

**Sewing matrix.** `B(g, k)_mn = ⟨u_m(k)| U(g; k) |u_n(k)⟩`, the matrix of an
operation `g` of the little group of `k` on the eigenvectors `u_n(k)` of a set
of bands, with `U(g; k)` the operator of `g` on the model's states at `k`. For
an operation combined with time reversal, the ket is complex conjugated before
`U(g; k)` acts. For a unitary operation, the trace of `B` over a degenerate
multiplet is that multiplet's character of `g`.
[One operation's sewing matrix](../guide/labels.md#one-operations-sewing-matrix).

**Site irrep; site-symmetry group.** The site-symmetry group of a site is the
set of operations that leave it fixed. A site irrep is an irreducible
representation of it, which an orbital or a Wannier function at the site
carries. For a spinful model it is double-valued.
[Site irreps](../background/conventions.md#site-irreps).

**Site-symmetry vector.** `p`, the multiplicity of each site irrep among an
atomic limit's Wannier functions at each Wyckoff position, maximal or not.
[What the stable real-space invariants are](../guide/srsi.md#what-the-stable-real-space-invariants-are).

**Slab cell.** A layer written in a three-dimensional cell, its direction out
of the plane filled with vacuum. A layer's group is a space group of its slab
cell.
[A layer is silent about what lies under it](../background/measurement.md#a-layer-is-silent-about-what-lies-under-it).

**Spinful orbital.** An orbital that holds both spin states, as each orbital of
a pythtb model made with `spinful=True` does. A spinless model whose orbital
index carries spin, as a Wannier90 spin-orbit import does, holds the up and
the down state of each spinful orbital as two orbitals, and `spin=` says
which. [How does pythTQC read the spin of my model?](../guide/spin.md).

**Splitting.** A division of a model's bands at a gap into valence bands, the
bands asked for, and conduction bands, the rest. `splitting` reads one.
[The no-go criterion](../guide/srsi.md#the-no-go-criterion).

**Stable real-space invariant (SRSI).** A real-space invariant that is linear in
the whole site-symmetry vector. The SRSIs `θ` are `θ = Θ p`, with `Θ` an
integer matrix the tables give. Two atomic limits with the same SRSIs can be
smoothly deformed into each other once the same atomic orbitals are added to
both. The ZSRSIs are fixed by the symmetry data vector. The SRSIs valued
modulo n are not, and are read off a model's orbitals. The tables are those of
Hwang *et al.*, Nat. Commun. **17**, 8564 (2026)
([Stable real-space invariants](../citing.md#stable-real-space-invariants)).
[What do the stable real-space invariants say about these bands?](../guide/srsi.md).

**Stably topological.** A set of bands that is not an atomic limit, and does
not become one when atomic bands are added. A nonzero symmetry indicator shows
it, and a zero one does not rule it out.
[What a zero does not certify](../guide/indicators.md#what-a-zero-does-not-certify).

**Symmetry data vector.** The multiplicities of the little-group
representations of a set of bands at every high-symmetry point, taken
together.
[What the little-group representations of a set of bands are](../guide/labels.md#what-the-little-group-representations-of-a-set-of-bands-are).

**Tie rule.** The rule that picks one group where several subgroups of the
largest order the measurement allows remain, and nothing measured tells them
apart.
[The descent, and the tie rule](../background/measurement.md#the-descent-and-the-tie-rule).

**Variety; closure.** A Wyckoff position's variety is the set of points it
covers in the cell: a point, or a line or plane where it has free parameters.
An orbital confined to a variety can slide along it, and the variety's closure
adds the more special positions it meets: the line `6c` of `R-3m`,
`(0, 0, z)`, closes on `3a` at `z = 0` and on `3b` at `z = 1/2`.
`slides_onto`, on what `wyckoff_sites_tables` returns and on each row of
`srsi_of_model`, names the positions on the closure at which the tables list
EBRs, the positions such an orbital can be moved to.

**Weak index; weak indicator.** In a crystal with inversion and time reversal,
the weak indices `ν1`, `ν2` and `ν3` are those Fu and Kane define, each 0 or 1.
`(−1)^νi` is the product of the inversion parities of the occupied Kramers
pairs, one parity a pair, over the four time-reversal-invariant momenta with
`k_i = 1/2`. `z2w,1`, `z2w,2` and `z2w,3` are the three weak indicators among
the named indicators. In each phase of the Fu-Kane-Mele model their values are
the weak indices. A layer's weak indicators are those of its decoupled stack.
[Three phases, and the Z4 indicator](../start/tutorial-3d.md#three-phases-and-the-z4-indicator);
[A layer's class is a three-dimensional class](../guide/indicators.md#a-layers-class-is-a-three-dimensional-class).

## Tolerances

Each tolerance has a unit. A number given in the wrong unit is not detected.
Each tolerance is an energy in the units of H, a dimensionless ratio, a length
in reduced coordinates, or a Cartesian length.

`max|H|` is the largest modulus of an entry of `H(k)` over `k`, once the mean
on-site energy is subtracted from its diagonal, in the units of H. A constant
shift of `H` leaves it unchanged. `realized_group` measures it at 64 random
`k`-points and on the hoppings themselves, and prints it as `measured max|H|`
where a precision is declared.

`tol`, `cut` and `window` are not arguments of any call. They are derived from
`precision`, or take the values of the `default` column where none is
declared. The rounding tolerance and the noise floor are computed as their
entries say.

| name | what it is for | unit | default |
|---|---|---|---|
| `precision` | the relative precision of a numerical model's numbers, from which `realized_group` derives its tolerances | dimensionless | `None` |
| `tol` | whether an operation holds | energy, in the units of H | `1e-8 * max\|H\|` |
| `cut` | which matrices the measurement of an operation keeps | dimensionless | `1e-8` |
| `window` | which eigenvectors of `H(k)` and `H(Rk)` are paired | dimensionless | `1e-3` |
| `position_tol` | how far an orbital center is moved onto a symmetric position | reduced coordinates | `None` in `realized_group`; `1e-4` in `wyckoff_sites_tables` |
| `symprec` | the tolerance of spglib, which finds the group of the cell | Cartesian length, in the units of the pythtb Lattice vectors | `1e-5` |
| the rounding tolerance | whether an operation holds to rounding | energy, in the units of H | the formula below |
| `degeneracy_tol` | which bands are one degenerate multiplet | energy, in the units of H | `1e-8 * max\|H\|`; `1e-8` in `multiplets` |
| `label_tol` | whether a multiplicity is read as an integer | dimensionless | `1e-3` |
| the noise floor | whether a number read off the eigenvectors at a point is zero | dimensionless | the formula below |

- `precision`: the relative precision of a numerical model's numbers, a
  positive number. It is how far the numbers may miss the symmetry of the
  crystal they approximate: relative to `max|H|` for the hoppings, and in
  reduced coordinates for the orbital centers. `realized_group` derives `tol`,
  `cut`, `window` and `position_tol` from it, and raises `symprec` where
  `precision * L` exceeds it (below). `r.numerical.precision` holds them.
  Declared with `precision=`. `realized_group` raises `ValueError` where
  `precision=` and `position_tol=` are both passed.
- `tol`: an energy tolerance for checking whether an operation is a symmetry
  of `H`. For each operation, candidate matrices are found from the equations
  `X H(k) = H(Rk) X` (`cut`, below), and the smallest of their covariance
  residuals is measured. The operation holds where it is at most `tol`, and is
  near where it holds and is not exact. With nothing declared,
  `tol = 1e-8 * max|H|`, or `1e-8` where `max|H|` is zero. With `precision=`,
  `tol = 10 * precision * max|H|`.
- `cut`: a dimensionless tolerance for checking whether an operation is a
  symmetry. The operations are measured on a grid of `nk` `k`-points along
  each periodic direction. At each point of the grid, the singular values of
  the equations `X H(k) = H(Rk) X` are computed on the matrices `X` kept so
  far. A combination of those matrices, a singular vector of the equations, is
  kept where its singular value is at most `cut` times the largest singular
  value of that point's equations on all matrices, or at most `sqrt(n) * tol`
  for `n` states, whichever is larger.
  `cut = tol / max|H|`: `1e-8` with nothing declared, and `10 * precision`
  with `precision=`.
- `window`: a dimensionless tolerance for pairing an eigenvector of `H(k)`
  with one of `H(Rk)` when the equations `X H(k) = H(Rk) X` are solved. At one
  `k`-point of the grid, the outer products of the paired eigenvectors span
  the matrices `X` kept at first, and the other points keep those that `cut`
  keeps there. Two eigenvectors are paired where their energies differ by at
  most `window` times the largest modulus of an entry of `H(k)` or `H(Rk)` at
  that point, the mean on-site energy subtracted, or by at most `tol`,
  whichever is larger. `window` is `10 * sqrt(cut)`, kept between `1e-3` and
  `0.3`: `1e-3` with nothing declared, and `0.01` with `precision=1e-7`.
- `position_tol`: a tolerance for moving the orbital centers onto symmetric
  positions, in reduced coordinates. A center that misses the position the
  group puts it at by at most `position_tol` is moved onto it. The move is
  measured once the shift of all the centers together that best fits them is
  taken out, since such a shift is a change of origin. That shift, where it is
  within `position_tol`, is removed too, so that the centers keep the model's
  origin. An operation that would move a center farther is broken by the
  centers. Equal to `precision` with `precision=`, or declared alone with
  `position_tol=`. With neither, the centers are not moved. In
  `wyckoff_sites_tables`, it is the largest distance at which a point is read
  as lying on a Wyckoff position, in reduced coordinates of the conventional
  cell used by the irreptables package.
- `symprec`: the tolerance of spglib, which finds the group of the cell from
  the orbital centers. It is a Cartesian length, in the units the pythtb
  Lattice vectors are given in, and is declared with `symprec=`. With
  `precision=`, it is raised to `10 * precision * L` where `precision * L`
  exceeds it, `L` being the length of the longest periodic lattice vector, and
  it is kept otherwise. `r.numerical.precision.symprec_raised` says whether it
  was raised.
- The rounding tolerance: the largest covariance residual that rounding alone
  gives an operation the Hamiltonian obeys exactly, an energy in the units of
  H. An operation whose covariance residual is at most the rounding tolerance
  is exact. The matrices of the representation `realized_group` returns carry
  `H(k)` onto `H(Rk)` within it. It is measured at random `k`-points, and `M`
  is the largest modulus of an entry of `H(k)` over them, the energy zero kept.
  It is the largest of three terms. The first is `5e-14 * M`. The second is
  `|G| * N * ε * M`, with `|G|` the number of operations represented, `N` the
  number of states and `ε` the machine epsilon. The third is the rounding of
  the Bloch phases of `H(k)` and `H(Rk)`. A phase `e^{2πi k·R}` rounds at `ε`
  times `|k|·|R|`, the sum of the sizes of its terms in reduced coordinates.
  The third term is `2π * ε` times the largest sum, along a row of `H`, of the
  hoppings' moduli, each weighted by those sums for its lattice vector and for
  its two orbitals' centers, at `k` and at `Rk` added together. No declaration
  moves the rounding tolerance.
- `degeneracy_tol`: an energy tolerance for grouping bands into degenerate
  multiplets, in the units of H. Going up in energy, a band joins a multiplet
  where its energy is within `degeneracy_tol` of the multiplet's lowest band.
  Left out, it is `1e-8 * max|H|`, with `max|H|` as `realized_group` measured
  it, or `1e-8` where `max|H|` is zero. `multiplets` is handed energies alone,
  and its default is `1e-8`.
- `label_tol`: a dimensionless tolerance for reading a multiplicity as an
  integer. The multiplicity of each little-group representation in a multiplet
  is computed from the multiplet's characters. It is read as an integer where
  it lies within `label_tol` of a nonnegative integer, and the call is refused
  with `integer_multiplicity` otherwise. `1e-3` unless `label_tol=` is passed.
- The noise floor: the estimate `8 * ε * |H(k)| / Δ` of how far a computed
  eigenvector at one `k`-point mixes with the states of other multiplets,
  dimensionless. Here `ε` is the machine epsilon, `|H(k)|` the spectral norm of
  `H(k)` and `Δ` the smallest gap between distinct multiplets at the point, in
  the units of H. A number read off the eigenvectors at the point, such as an
  entry of a sewing matrix or a character, is zero below the larger of `1e-14`
  and the noise floor, `coreps_at(...).numerical.thresholds.effective`. The
  noise floor is infinite where `Δ` is zero. It is zero for a spectrum of one
  multiplet, where there is no other multiplet to mix with and `Δ` is
  infinite, and the threshold is then `1e-14`.

The defaults of the tolerances that are keywords of a call are the signatures'
own. These lines print them:

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
#  coreps_at              degeneracy_tol=None, label_tol=0.001
#  hsp_table              degeneracy_tol=None, label_tol=0.001
#  characters             degeneracy_tol=None
#  sewing_matrix          degeneracy_tol=None
#  multiplets             degeneracy_tol=1e-08
```

## Budgets, meshes and lengths

The numbers that are not tolerances: a memory budget, a count, a mesh, or a
length that nothing returned depends on.

| name | what it sets, and where | unit | default |
|---|---|---|---|
| `max_bytes` | the peak memory the projection of `H` onto its symmetric part may take. It is also the most memory the measurement of one operation may take where exactly degenerate levels need more than its own `5.12e8` bytes (`realized_group`) | bytes | `1e9` |
| `max_candidates` | the largest number of candidate representations searched where the measurement leaves a choice of representation open. Past it the call refuses `rep_search_too_large` (`realized_group`) | candidates | `2**20` |
| `cap` | the largest number of decompositions listed, past which `numerical.enumeration.truncated` is `True` (`decompose`); the largest number of atomic realizations, past which the call refuses `srsi_realizations_truncated` (`atomic_realizations`) | decompositions; realizations | `200`; `256` |
| `nk` | the number of `k`-points along each periodic direction of the grid the operations are measured on (`realized_group`) | points | `6` |
| `c` | the width of the vacuum between the copies of a decoupled stack (`stack_presentation`) | the length units of the pythtb Lattice vectors | `8.0` |

No hopping crosses the vacuum between the copies. The defaults, as the
signatures hold them:

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
