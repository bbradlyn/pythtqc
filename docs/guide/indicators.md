# Which symmetry-indicator class do these bands occupy?

## What a symmetry indicator is

The integer vectors that satisfy a space group's compatibility relations,
negative entries allowed, form a lattice, `{BS}`. The symmetry data vector of
every set of bands separated from the rest at every `k` lies in it. The
integer combinations of the symmetry data vectors of the group's elementary
band representations (EBRs), negative multiplicities allowed, form a
sublattice of the same rank, `{AI}`
([What elementary band representations are](atomic-limits.md#what-elementary-band-representations-are)).
The quotient `X_BS = {BS}/{AI}` is a finite abelian group fixed by the space
group alone, its indicator group. The symmetry indicator of a set of bands is
the class of its symmetry data vector in `X_BS`. The EBR matrix has the
symmetry data vector of each EBR as a column. The elementary divisors of its
Smith normal form above 1, `d1, d2, ...`, write `X_BS` as
`Z_d1 x Z_d2 x ...`. A class is then one integer modulo each divisor.

This package uses two facts about the indicator. First, the class of an atomic
limit is zero, and classes add. So a set of bands with a nonzero class is not
an atomic limit, and does not become one when atomic bands are added. Second,
the symmetry data vector of a fragile set of bands is the difference of the
symmetry data vectors of two atomic limits, so its class is zero. Where the
class `symmetry_indicator` returns is zero, the bands may be an atomic limit,
fragile, or stably topological.

The definitions and proofs are in Po, Vishwanath and Watanabe,
[*Symmetry-based indicators of band topology in the 230 space groups*, Nat. Commun. **8**, 50 (2017)](https://doi.org/10.1038/s41467-017-00133-2),
and Kruthoff *et al.*,
[*Topological classification of crystalline insulators through band structure combinatorics*, Phys. Rev. X **7**, 041069 (2017)](https://doi.org/10.1103/PhysRevX.7.041069).
[How a symmetry indicator is computed](../background/tqc-primer.md#how-a-symmetry-indicator-is-computed),
in [The topological quantum chemistry this package uses](../background/tqc-primer.md),
gives a longer account.

## The call

`symmetry_indicator` takes the table `hsp_table` returns for a set of bands. It
returns the class of the table's symmetry data vector in the indicator group of
the space group the table was read in. A symmetry data vector written out by
hand, `{corep: multiplicity}`, is passed with what `realized_group` returned
for the model, `r`: `symmetry_indicator(vector, r)`. A class can only be
returned if the table holds the little-group representations at every
high-symmetry point the model samples, which for a layer are the points in its
plane. A table that refused at some point is refused with `incomplete_content`.

The example is lin3g, from `pythtqc.models`, the spinful square layer of Lin
*et al.*, [Nat. Commun. **15**, 550 (2024)](https://doi.org/10.1038/s41467-024-44762-w).
[Could these bands be an atomic limit?](atomic-limits.md) decomposes it. It
has two orbitals at Wyckoff position `1a` and two at `2c`. The example reads
the class of its six lowest bands, in a group whose indicator group is
trivial:

```python
import pythtqc as tqc
from pythtb.models import kane_mele
from pythtqc import models

r = tqc.realized_group(models.lin3g(), typat=[0, 0, 1, 1])
t = tqc.hsp_table(r, bands=6)                      # the six lowest bands
si = tqc.symmetry_indicator(t)
print(si)
#  SymmetryIndicator(P4mm1' #99.164, trivial indicator group)
print(si.group.number_str, si.divisors, si.indicator_group, si.values, si.order)
#  99.164 () trivial () 1
print(si.numerical.reconstructed)
#  {'-A6': '-M6', '-A7': '-M7', '-R5': '-X5', '-Z6': '-GM6', '-Z7': '-GM7'}
```

## How to read it

- `P4mm1' #99.164`: `si.group`, the space group the table was read in.
- `trivial indicator group`: its indicator group. The EBR matrix of `P4mm1'`
  has no elementary divisor above 1, so its indicator group is trivial, and
  every set of bands has class zero.
- `99.164`: `si.group.number_str`, the group's number.
- `()` after `99.164`: `si.divisors`, the elementary divisors above 1, none
  here.
- `trivial`: `si.indicator_group`, the indicator group written as text,
  `trivial` where it has no factor.
- `()` after `trivial`: `si.values`, the class, one integer modulo each
  divisor, empty here.
- `1`: `si.order`, the order of the class: the fewest copies of the bands that
  together have class zero, `1` for the zero class.
- `{'-A6': '-M6', ...}`: `si.numerical.reconstructed`. lin3g is a layer, and
  its class is read in the three-dimensional space group of its decoupled
  stack ([The group a class is read in](#the-group-a-class-is-read-in)).
  A layer's `H(k)` has no `k_z`, so the coreps at `A`, `R` and `Z`, the points
  at `k_z = 1/2`, are not measured. Each of them occurs in every EBR as often
  as a measured corep, and takes that corep's multiplicity: `-A6` takes the
  multiplicity of `-M6`, and `-Z6` that of `-GM6`. `GM` is the tables' name
  for Γ.

## What a zero does not certify

Every set of lin3g's bands has class zero, as in every group with a trivial
indicator group. Its bands 4 and 5 have class zero and are fragile
([What a decomposition cannot certify](atomic-limits.md#what-a-decomposition-cannot-certify)).
Stably topological sets of bands that the indicator does not detect have class
zero too
([What do the stable real-space invariants say about these bands?](srsi.md)).
Nothing `symmetry_indicator` returns states that a set of bands is an atomic
limit.

A nonzero class and the absence of a decomposition say different things. Where
`decompose` returns no decomposition, the bands are not an atomic limit, and
may be fragile. Where the class is nonzero, they are stably topological.

## A layer whose group has an indicator

`realized_group` reads a 2D model on a substrate by default, and free-standing
with `environment="free-standing"`
([Is my 2D model on a substrate or free-standing?](layers.md)). A free-standing
layer has vacuum on both sides, so the horizontal mirror is a candidate
symmetry.

The example is Kane-Mele, from `pythtb.models`, with intrinsic spin-orbit
coupling and neither a staggered potential nor a Rashba term, its orbitals
declared `s`. Free-standing, its group is `P6/mmm1'`. The example reads the
class of its two lowest bands:

```python
km = kane_mele(0.0, -1.0, 0.3, 0.0)                # delta, t, soc, rashba
free = tqc.realized_group(km, environment="free-standing", characters={0: "s"})
t_free = tqc.hsp_table(free, bands=2)              # the two lowest bands
si_free = tqc.symmetry_indicator(t_free)
print(si_free.summary())
#  class (1, 0) in Z6 x Z12, in P6/mmm1' #191.234   (.values, .indicator_group, .group)
#    of order 6: 6 copies of these bands are in the zero class   (.order)
#    the indicator group: Z6 x Z12, a class being an integer mod 6 and one mod 12   (.indicator_group, .divisors)
#    indicators: z2w,3 = 1, z4 = 2, z3m,pi = 1, z3m,0 = 1, z6m,0 = 5, z6m,pi = 5, z12 = 10, the names of the paper that tabulates the stable real-space invariants, Hwang et al. (2026)   (.indicators)
#    the coreps at the high-symmetry points: -GM9 + -K8 + -M6   (.symmetry_data)
print(si_free.numerical.summary())
#  the coreps at every point of the group's tables, A, H and L completed: -GM9 + -A9 + -H8 + -K8 + -L6 + -M6   (.completed_symmetry_data)
#  A, H and L, at k_z = 1/2 out of the layer's plane: their coreps measured on the layer's decoupled stack, the layer repeated along its normal with no hopping between copies, in the layer's own group P6/mmm1' #191.234   (.completion)
#  the class of the layer's decoupled stack: every component is an invariant of that stack, fixed by the layer's own symmetry data vector   (.two_dimensional)
#  the Smith normal form of the group's matrix of elementary band representations, computed again exactly, gives the same elementary divisors and the same order of the class   (.verification)
```

- `class (1, 0) in Z6 x Z12, in P6/mmm1' #191.234`: the class,
  `si_free.values`, `1` modulo 6 and `0` modulo 12, followed by its indicator
  group and the space group it is read in. Each line of a summary ends with
  the fields that hold it. `values` depends on the Smith normal form, and the
  irreptables package ships the one used here. Another Smith normal form of
  the same matrix can write the same class with other values. The order is
  the same in every one.
- `of order 6`: `si_free.order`. The line is printed where the order is above 1.
- `the indicator group: Z6 x Z12`: `si_free.indicator_group`, and its
  elementary divisors above 1, `si_free.divisors`, here `(6, 12)`.
- `indicators: z2w,3 = 1, ...`: `si_free.indicators`, the class written as the
  values of the named indicators of the paper that tabulates the stable
  real-space invariants
  ([Named indicators](#named-indicators)).
- `the coreps at the high-symmetry points: -GM9 + -K8 + -M6`:
  `si_free.symmetry_data`, the symmetry data vector at the points of the
  layer's plane, the same as `t_free.symmetry_data`.
- `the coreps at every point of the group's tables, A, H and L completed`, from
  `si_free.numerical.summary()`: `si_free.numerical.completed_symmetry_data`,
  the symmetry data vector with the coreps at `A`, `H` and `L`, the points at
  `k_z = 1/2`, off the layer's plane.
- `A, H and L, at k_z = 1/2 out of the layer's plane`:
  `si_free.numerical.completion`, the measurement of those coreps on the
  layer's decoupled stack
  ([The group a class is read in](#the-group-a-class-is-read-in)).
- `the class of the layer's decoupled stack`:
  `si_free.numerical.two_dimensional`
  ([A layer's class is a three-dimensional class](#a-layers-class-is-a-three-dimensional-class)).
- `the Smith normal form of the group's matrix of elementary band
  representations, computed again exactly`: `si_free.numerical.verification`.
  The Smith normal form of the EBR matrix, computed again in exact integers,
  has the elementary divisors of the one the irreptables package ships, and
  gives the class the same order.

Where the indicator group is nontrivial and an orbital's site irrep was
assigned by the lowest-angular-momentum rule, the summary prints one more line,
ending `(.site_content)`. A declaration of the orbitals can then change the
class ([When to declare](orbitals.md#when-to-declare)).

## The group a class is read in

Unlike lin3g's, some little-group representations of `P6/mmm1'` at `A`, `H`
and `L` pair with none in the plane. For each, no little-group representation
in the plane occurs as often as it does in every EBR of the group, those of
the positions off the layer included. They are measured on the layer's
decoupled stack. The decoupled stack is the layer repeated along its normal
with no hopping between copies, each orbital at its own height. Its bands do
not depend on `k_z`, so at `k_z = 1/2` it carries the bands it carries at
`k_z = 0`. It obeys exactly the operations the layer obeys, and its bands are
labeled in the layer's own group with the layer's own representation matrices
([A layer's out-of-plane rows](../background/measurement.md#a-layers-out-of-plane-rows)).

`symmetry_indicator` builds the stack itself from the free-standing Kane-Mele
layer's table. `si_free.numerical.completion` holds the measurement:

```python
comp = si_free.numerical.completion
print(comp.outcome, comp.stack_period, comp.stack_extent)
#  completed 8.0 0.0
print([(f.kpoint, f.in_plane_kpoint) for f in comp.gates.fibers])
#  [('A', 'GM'), ('H', 'K'), ('L', 'M')]
```

- `completed`: `comp.outcome`, which says that the coreps out of the plane were
  measured on the stack.
- `8.0`: `comp.stack_period`, the stack's period along the layer's normal, in
  the model's units of length.
- `0.0`: `comp.stack_extent`, the layer's extent along its normal, in the same
  units. The difference of the two is the vacuum between the copies, 8 in the
  model's units of length whatever the layer. The class does not depend on it.
- `[('A', 'GM'), ('H', 'K'), ('L', 'M')]`: `comp.gates.fibers`, which pairs
  each point at `k_z = 1/2` with the point in the plane at the same in-plane
  `k`. At each pair the stack carries the same energies and characters.

A vector written out by hand carries no model to build a stack from. It is
read at the points of the plane, and at each point out of it where it names a
little-group representation. Among the EBRs of the Wyckoff positions within
the layer, each little-group representation out of the plane occurs as often
as one in the plane. One the vector does not name takes that one's
multiplicity, and no stack is needed. The example passes the free-standing
Kane-Mele layer's symmetry data vector written out by hand, which holds the
coreps of the plane alone:

```python
print(tqc.symmetry_indicator(t_free.symmetry_data, free).values)
#  (1, 0)
```

- `(1, 0)`: the class `si_free.values` read from the table, read again without
  the stack.

On a substrate, the default for a 2D model, one side of the layer differs from
the other, and the horizontal mirror is not an operation. The example reads the
class of the same Kane-Mele layer on a substrate. It then decomposes the
free-standing layer's two bands:

```python
sub = tqc.realized_group(km, characters={0: "s"})  # on a substrate, the default
si_sub = tqc.symmetry_indicator(tqc.hsp_table(sub, bands=2))
print(sub.group.name, si_sub)
#  P6mm1' SymmetryIndicator(P6mm1' #183.186, trivial indicator group)
print(si_sub.numerical.completion, sub.environment.cross_check.free_standing)
#  None P6/mmm1' #191.234
print(si_sub.summary())
#  the indicator group of P6mm1' #183.186 is trivial   (.indicator_group, .group)
#    the coreps at the high-symmetry points: -GM9 + -K6 + -M5   (.symmetry_data)
print(si_sub.numerical.summary())
#  the coreps at every point of the group's tables, A, H and L completed: -GM9 + -A9 + -H6 + -K6 + -L5 + -M5   (.completed_symmetry_data)
#  the coreps at A, H and L are not measured: each occurs in every elementary band representation as often as a measured one, and takes its multiplicity   (.reconstructed)
#    -A7 as -GM7, -A8 as -GM8, -A9 as -GM9, -H4 as -K4, -H5 as -K4, -H6 as -K6, -L5 as -M5
#  the free-standing layer's group, as realized_group measured it, is P6/mmm1' #191.234, larger than this one: the same layer with vacuum on both sides, whose class is asked with environment='free-standing'   (.free_standing_group)
#  the Smith normal form of the group's matrix of elementary band representations, computed again exactly, gives the same elementary divisors and the same order of the class   (.verification)
d_free = tqc.decompose(t_free)
print(d_free.decomposable, d_free.certificate, sorted(d_free.numerical.restriction.outside_layer))
#  False proof ['1b', '2d', '3g']
```

- `P6mm1' SymmetryIndicator(P6mm1' #183.186, trivial indicator group)`: on a
  substrate the group is `P6mm1'`, whose indicator group is trivial, so these
  bands have class zero there.
- `None`: `si_sub.numerical.completion`. Every corep of `P6mm1'` out of the
  plane occurs in every EBR as often as one in the plane, so no stack is built.
- `P6/mmm1' #191.234`: `sub.environment.cross_check.free_standing`, the
  free-standing layer's group, which `realized_group` measured beside the
  substrate's.
- `the coreps at A, H and L are not measured`:
  `si_sub.numerical.reconstructed`. The next line pairs each corep at `A`, `H`
  and `L` with the measured corep whose multiplicity it takes.
- `the free-standing layer's group, as realized_group measured it, is P6/mmm1'
  #191.234`: `si_sub.numerical.free_standing_group`. It is named where the
  free-standing group is larger than `si_sub.group`, and where the cross-check
  of `realized_group` found the substrate group to be exactly the operations of
  the free-standing group that preserve the layer's normal. Otherwise it is
  `None`, as with `cross_check=False`. The class in that group is not read
  here; `environment="free-standing"` asks for it.
- `False proof ['1b', '2d', '3g']`: the free-standing layer's two bands have no
  decomposition, `d_free.decomposable`, with a proof, `d_free.certificate`.
  These bands' class is `(1, 0)`, and a symmetry data vector with a nonzero
  class has no decomposition. `d_free.numerical.restriction.outside_layer`
  names the positions of `P6/mmm1'` off the layer, `1b`, `2d` and `3g`, which
  the decomposition of a free-standing layer leaves out
  ([The call](atomic-limits.md#the-call)).

## A layer's class is a three-dimensional class

A layer's `H(k)` has no `k_z`, so its class is the class of its decoupled
stack. The stack carries the same states at each point at `k_z = 1/2` as at
the point of the plane with the same in-plane `k`, so its little-group
representations there are fixed by those in the plane. Each component of the
class, and each named indicator, is fixed by the layer's own symmetry data
vector, even where its linear form reads only little-group representations out
of the plane. Kane-Mele's two bands
read `(1, 0)` on the stack, with `z2w,3 = 1` and `z4 = 2`.
`si_free.numerical.two_dimensional` says which points lie out of the plane,
and which coreps each named indicator reads:

```python
print(si_free.numerical.two_dimensional.out_of_plane_kpoints)
#  ('A', 'H', 'L')
print(si_free.numerical.two_dimensional.support)
#  {'z2w,3': 'out-of-plane only', 'z4': 'mixed', 'z3m,pi': 'out-of-plane only', 'z3m,0': 'in-plane only', 'z6m,0': 'in-plane only', 'z6m,pi': 'out-of-plane only', 'z12': 'mixed'}
```

- `('A', 'H', 'L')`: `out_of_plane_kpoints`, the points of the group's tables
  out of the layer's plane, at `k_z = 1/2`.
- `{'z2w,3': 'out-of-plane only', ...}`: `support`. For each named indicator,
  keyed by the paper's name, it says whether the nonzero coefficients of the
  indicator's linear form in the symmetry data vector lie on coreps in the
  plane, out of it, or both. The linear forms are those of the irreptables
  package ([Named indicators](#named-indicators)).
  `z2w,3` reads coreps out of the plane alone, and its value, 1, is fixed by
  the layer's coreps in the plane all the same. `z6m,0` reads coreps in the
  plane alone, and `z4` both.

## Named indicators

A class is also written as the values of named indicators, each a linear form
modulo an integer. Two sets of names exist. The irreptables package ships
linear forms in the symmetry data vector. The paper that tabulates the stable
real-space invariants (SRSIs), Hwang *et al.*,
[*Stable real-space invariants and topology beyond symmetry indicators*, Nat. Commun. **17**, 8564 (2026)](https://doi.org/10.1038/s41467-026-74844-w),
gives linear forms in the integer-valued SRSIs (ZSRSIs)
([What do the stable real-space invariants say about these bands?](srsi.md)).
The two sets name the same indicators, each in its own spelling. The two names
of one indicator take the same value on every set of bands. Their linear forms
are written in different variables, the symmetry data vector and the ZSRSIs.
`si.indicators` is the class
in the paper's names. It is computed by applying the paper's forms to the
ZSRSIs of the symmetry data vector, the ZSRSIs `srsi_of_bands` returns. The
example prints the free-standing Kane-Mele layer's indicators, and lin3g's,
which are `None`:

```python
print(si_free.indicators)
#  {'z2w,3': 1, 'z4': 2, 'z3m,pi': 1, 'z3m,0': 1, 'z6m,0': 5, 'z6m,pi': 5, 'z12': 10}
print(si.indicators, si.numerical.indicators_note)
#  None the indicator group of #99.164 is trivial, so there is no class to read in the paper's named indicators; the paper's tables carry an indicator row for exactly the 170 grey systems that have one, and this is not one of them
```

- `{'z2w,3': 1, ...}`: `si_free.indicators`, each named indicator and its
  value. The named values determine the class, as `values` does.
- `None the indicator group of #99.164 is trivial, ...`: lin3g's
  `si.indicators`, and `si.numerical.indicators_note`, which says why it is
  `None`. It is `None` for a trivial indicator group, for which the paper names
  no indicator. It is `None` for a group without time reversal, which the
  paper's tables do not hold. It is `None` where the tables of stable
  real-space invariants are not installed, or cannot be read for this group.
  The class in `values` is returned in each case.

## When it refuses

A refusal is a returned record. It is falsy, and names the check that stopped
the call and what it measured
([What does a refusal mean, and what do I do about it?](refusals.md)). The
example hands in a vector written out by hand for lin3g, with two bands at Γ
and at `M` and four at `X`, which is no band structure:

```python
print(tqc.symmetry_indicator({"-GM6": 1, "-M6": 1, "-X5": 2}, r).gate)
#  not_a_band_structure
```

The checks `symmetry_indicator` meets:

- [`not_a_band_structure`](../reference/refusals.md#not_a_band_structure): the
  symmetry data vector fails the group's compatibility relations.
- [`incomplete_content`](../reference/refusals.md#incomplete_content): the table
  refused at some high-symmetry point. The table's `refusals` names each such
  point's refusal.
- [`indicator_not_evaluable`](../reference/refusals.md#indicator_not_evaluable):
  a layer's little-group representations at `k_z = 1/2` that take the
  multiplicity of none in its plane could not be read. For a table, their
  measurement on the decoupled stack stopped, and `context["stack"]` says
  where. For a vector written out by hand, the tables' Wyckoff positions could
  not be read in the group's cell, and the reason names the table `hsp_table`
  returns as what to pass.
- [`unknown_label`](../reference/refusals.md#unknown_label): a little-group
  representation that the group's EBRs do not hold.
- [`negative_multiplicity`](../reference/refusals.md#negative_multiplicity),
  [`non_integer_multiplicity`](../reference/refusals.md#non_integer_multiplicity):
  a vector that is not a count of little-group representations.
- [`tables_not_ready`](../reference/refusals.md#tables_not_ready): the group's
  cell could not be matched to the setting the tables are written in, so no
  table can be read for it.
- [`ebr_tables`](../reference/refusals.md#ebr_tables): the installed tables
  hold no EBRs for this group.
- [`irrep_tables`](../reference/refusals.md#irrep_tables): the character tables
  were not loaded, or hold no irrep table for this group.
- [`ebr_convention`](../reference/refusals.md#ebr_convention): the tables' EBRs
  of this group cannot be carried into the phase convention of the labels
  returned.

## Where to go next

- [Returned records](../reference/records.md) lists every field of
  [`SymmetryIndicator`](../reference/records.md#symmetryindicator), its
  `numerical.completion` and `numerical.two_dimensional` among them.
- [Functions](../reference/functions.md) gives the arguments of
  [`symmetry_indicator`](../reference/functions.md#symmetry_indicator); of
  [`stack_presentation`](../reference/functions.md#stack_presentation), which
  returns a layer's decoupled stack as a pythtb model; and of
  [`in_plane_restriction`](../reference/functions.md#in_plane_restriction),
  which returns the little-group representations of a group's EBRs that a
  layer measures.
- `pythtqc.groups.indicator_group(r.group)` returns the elementary divisors of
  the group's EBR matrix and, for a group with time reversal, the named forms
  of the irreptables package, with no class.
- [What do the stable real-space invariants say about these bands?](srsi.md)
  reads the SRSIs of a set of bands and of a model, among them the SRSIs valued
  modulo n, which the symmetry data vector does not fix.
- [Fu-Kane-Mele: a 3D crystal, a distortion, the Z4 indicator](../start/tutorial-3d.md)
  reads a nonzero class in three dimensions, in three phases of one crystal.
