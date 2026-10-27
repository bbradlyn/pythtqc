# Which symmetry-indicator class do these bands occupy?

A symmetry indicator asks of a band set's
[symmetry vector](labels.md#every-high-symmetry-point-at-once) a stronger
question than [a decomposition](atomic-limits.md) does: which class it
occupies in the group's band structures modulo its atomic limits. This chapter
reads that class, what it rules out and what it does not, how a layer's class
is completed and in which group it lives, and the names a class is spoken in.
The models are
lin3g, from `pythtqc.models`, the model of [labelling the bands](labels.md) and
[atomic limits](atomic-limits.md), whose group has no indicator; and
Kane–Mele, from `pythtb.models`, a layer whose free-standing group has one. How
the class is computed from the elementary band representations is in
[the primer](../background/tqc-primer.md#how-a-symmetry-indicator-is-computed);
the three phases of Fu–Kane–Mele's
distorted diamond, which a three-dimensional indicator tells apart, are in
[the 3D tutorial](../start/tutorial-3d.md).

## The call

```python
import pythtqc as tqc
from pythtb.models import kane_mele
from pythtqc import models

r = tqc.realized_group(models.lin3g(), typat=[0, 0, 1, 1])
model, group, assignment, validated, rep = r
t = tqc.hsp_table(rep, bands=6)                    # the six lowest bands
si = tqc.symmetry_indicator(t)
print(si)
#  SymmetryIndicator(trivial indicator group, #99.164, trivial)
print(si.number_str, si.divisors, si.structure, si.values, si.order)
#  99.164 () trivial () 1
print(si.reconstructed)
#  {'-A6': '-M6', '-A7': '-M7', '-R5': '-X5', '-Z6': '-GM6', '-Z7': '-GM7'}
```

## How to read it

The band structures the compatibility relations allow, modulo the atomic
limits, form a finite abelian group fixed by the space group alone,
`X_BS = {BS}/{AI} ≅ ⊕ᵢ ℤ_{dᵢ}`, with the `dᵢ` the elementary divisors above 1
of the group's EBR matrix, whose columns are its EBRs' symmetry vectors,
brought to its Smith normal form; a band set's indicator is the class of its
symmetry vector there
([how a symmetry indicator is computed](../background/tqc-primer.md#how-a-symmetry-indicator-is-computed)
sets out the construction). `number_str` is the group the class lives in,
`P4mm1'` (`99.164`), the one handed in. It has no divisor above 1: `divisors`
is empty and `structure` reads `trivial`, so there is no class to measure, and
`trivial indicator group` is the verdict. Where there is a class, `values`
holds it in Smith coordinates, one integer modulo each `dᵢ`, and `order` is its order in `X_BS`, `1` when it vanishes; unlike
the components, the order does not depend on the choice of basis.

`reconstructed` concerns the layer. Its group is the three-dimensional space
group of its slab, whose EBR basis has rows at `A`, `R` and `Z`, at
`k_z = ½`, where the layer's zone has no `k`. Each repeats a row in the plane,
`-A6` taking `-M6`'s multiplicity in every band structure of the group, so its
multiplicity is arithmetic: filled in, and named on the record.

## What a zero does not certify

A **nonzero** class says no band set with this content is an atomic limit, nor
any with atomic bands added to it, since an atomic limit's class is zero and
classes add: that is what makes the indicator a stable invariant. A `none`
from `decompose` says the first half only, and a fragile content decomposes
once suitable atomic bands are added. A **zero** class certifies nothing:
fragile topology lives in the kernel, the zero class, by construction, since a
fragile content is the difference of two atomic limits' contents. lin3g is the
case, its group's one class zero and its bands 4 and 5 fragile all the same
([what a decomposition cannot certify](atomic-limits.md#what-a-decomposition-cannot-certify)).
The record carries no triviality certificate, because there is none to carry.

## A layer whose group has an indicator

Kane–Mele with intrinsic spin–orbit coupling and neither a staggered potential
nor a Rashba term, free-standing: vacuum on both sides, so the horizontal
mirror is a candidate symmetry ([a 2D model](layers.md)), and its orbitals
declared `s`:

```python
km = kane_mele(0.0, -1.0, 0.3, 0.0)                # delta, t, soc, rashba
free = tqc.realized_group(km, environment="free-standing", characters={0: "s"})
t_free = tqc.hsp_table(free.rep, bands=2)          # the two lowest bands
si_free = tqc.symmetry_indicator(t_free)
print(si_free.summary())
#  indicated nontrivial, in P6/mmm1' #191.234: no atomic limit has this content, nor any with atomic bands added   (.verdict)
#    the indicator group, the band structures modulo the atomic limits: Z6 x Z12, a class being an integer mod 6 and one mod 12   (.structure)
#    canonical: z2w,3 = 1, z4 = 2, z3m,pi = 1, z3m,0 = 1, z6m,0 = 5, z6m,pi = 5, z12 = 10, the names of the paper that tabulates the stable real-space invariants, Hwang et al. (2026)   (.canonical)
#    indicator: 1 mod 6, 0 mod 12, of order 6: 6 copies of these bands are in the zero class   (.values, .order)
#    irreptables: z2w,3 = 1, z4 = 2, z3m,π = 1, z3m,0 = 1, z6m,0 = 5, z6m,π = 5, z12 = 10, the same indicators as the irreptables package names them   (.irreptables_named)
#    the coreps at the high-symmetry points: -A9 + -GM9 + -H8 + -K8 + -L6 + -M6   (.content)
#    A, H and L, at k_z = ½ out of the layer's plane: their coreps measured on the layer's decoupled stack, the layer repeated along its normal with no hopping between copies, in the layer's own group #191.234   (.completion)
#    the class of the layer's decoupled stack: every component is an invariant of that stack, fixed by the layer's own content   (.two_dimensional)
```

The summary, line by line:

- `indicated nontrivial, in P6/mmm1' #191.234`: the verdict, the group the class
  lives in, and the stable statement a nonzero class makes; the next line is
  its indicator group, `ℤ₆ × ℤ₁₂`.
- `canonical` and `irreptables`: the class named twice, in the paper's names
  and in those of `irreptables`, the package that ships the tables pythtqc
  reads ([below](#the-names-a-class-is-spoken-in)).
- `indicator: 1 mod 6, 0 mod 12, of order 6`: the class in Smith coordinates,
  `values = (1, 0)`, and its order. The Smith components and the named forms
  are two coordinate systems on one class.
- The coreps at the high-symmetry points: the symmetry vector, now with rows
  at `A`, `H` and `L`, where the layer has no `k`: measured
  ([below](#the-group-the-class-belongs-to)).
- `A, H and L, at k_z = ½`: how those rows were obtained, on the layer repeated
  with `c = 8` of vacuum between its copies, in the model's length units, which
  nothing returned depends on, and read in the layer's own group
  ([below](#the-group-the-class-belongs-to)).
- `the class of the layer's decoupled stack`: how a layer's class reads
  ([below](#a-layers-class-is-a-three-dimensional-class)).

The labels, and with them the class a band set is called, rest on what the
orbitals are declared to be, and `site_content` records how each site irrep is
known ([saying what the orbitals are](orbitals.md)).

## The group the class belongs to

Where an out-of-plane row repeats an in-plane one, its multiplicity is
arithmetic, as for lin3g. Where it does not, as for `P6/mmm1'`, the row is
**measured** on a crystal that has it: the same layer repeated along its normal
with no hopping between copies, each orbital at its own height, whose bands do
not depend on `k_z`, so that the stack carries at `k_z = ½` the bands it
carries at `k_z = 0`. A slab, a buckled layer or a bilayer, is stacked at its
orbitals' heights, its period the slab's extent along the normal plus `c`. That
decoupled stack obeys exactly what the layer obeys, since an operation of the
stack, combined with a translation along the stacking, carries one copy onto
itself, so its group is the layer's own: each operation is carried to the
stack by its Cartesian action, with the layer's own matrix, and neither the
group nor the matrices are measured again. `symmetry_indicator` builds the
stack itself and records, in
`completion["gates"]`, the measured number of every check that ties it to the
layer
([a layer's out-of-plane rows](../background/measurement.md#a-layers-out-of-plane-rows)
sets out the construction and its checks); each is an identity, and one that
fails raises `InternalError`, the package disagreeing with itself.
`completion["outcome"]` reads `completed`.

A larger group is a question about the layer with vacuum on both sides, the
free-standing one. On a substrate, the default for a 2D model, one side differs
and there is no horizontal mirror to find, and the free-standing layer's group
is the one `realized_group` measured beside the substrate's:

```python
sub = tqc.realized_group(km, characters={0: "s"})  # on a substrate, the default
si_sub = tqc.symmetry_indicator(tqc.hsp_table(sub.rep, bands=2))
print(sub.group.name, si_sub)
#  P6mm1' SymmetryIndicator(trivial indicator group, #183.186, trivial)
print(si_sub.completion, sub.rep.free_standing_group == sub.cross_check["free_standing"])
#  None True
print(si_sub.summary())
#  trivial indicator group, in P6mm1' #183.186: there is no class to read   (.verdict)
#    the coreps at the high-symmetry points: -GM9 + -K6 + -M5   (.content)
#    the coreps at A, H and L are not measured: each occurs in every elementary band representation as often as a measured one, and takes its multiplicity   (.reconstructed)
#      -A7 as -GM7, -A8 as -GM8, -A9 as -GM9, -H4 as -K4, -H5 as -K4, -H6 as -K6, -L5 as -M5
#    the free-standing layer's group, as realized_group measured it, is P6/mmm1' #191.234, larger than this one: the same layer with vacuum on both sides, whose class is asked with environment='free-standing'   (.free_standing_group)
d_free = tqc.decompose(t_free)
print(d_free.verdict, d_free.certificate, sorted(d_free.restriction.outside_layer))
#  none proof ['1b', '2d', '3g']
```

On a substrate the class lives in `#183.186`, `P6mm1'`, whose indicator group
is trivial: that is the verdict for this layer there. Every
row of its basis out of the plane repeats an in-plane one, so no stack is
built and `completion` is `None`. The last line of the summary names the
free-standing layer's group, `P6/mmm1'` `#191.234`, as `realized_group`
measured it in the other environment and found it to agree
(`r.rep.free_standing_group`), without using it; it is named where it is larger
than the substrate's, and not at all where that environment was not measured
(`cross_check=False`) or not compared, as for a declaration written in the
substrate's site irreps. A decomposition of the free-standing presentation is
read over the layer's own band representations, those of the Wyckoff positions
within the layer: `1b`, `2d` and `3g` lie off the layer and are left out, and
on the rest every row out of the plane repeats an in-plane one, so no stack is
needed. It finds no atomic limit, with a proof, as the class `(1, 0)` says it
must.

## A layer's class is a three-dimensional class

A two-dimensional model's class is read in the three-dimensional space group of
its slab, whose basis has rows at points out of the layer's plane, at
`k_z = ½`:

```python
print(si_free.two_dimensional["out_of_plane_points"])
#  ('A', 'H', 'L')
print(si_free.two_dimensional["irreptables_support"])
#  {'z2w,3': 'out-of-plane only', 'z4': 'mixed', 'z3m,π': 'out-of-plane only', 'z3m,0': 'in-plane only', 'z6m,0': 'in-plane only', 'z6m,π': 'out-of-plane only', 'z12': 'mixed'}
```

The layer's `H(k)` has no `k_z`, so the class is the class of the layer's
decoupled stack, the layer repeated along its normal with no hopping between
copies, whose rows at `k_z = ½` repeat the in-plane ones. Every component is
an invariant of that stack, fixed by the layer's own content, whether its form
reads `A`, `H` and `L` alone, as `z2w,3` does, both planes, as `z4` does, or
the plane alone, as `z6m,0` does: `irreptables_support` records where each
form's coefficients sit, and `two_dimensional["note"]` states the reading on
the record. A topological invariant of the layer is one of the stack its
copies make: Kane–Mele's two bands read `(1, 0)` there, with `z2w,3 = 1` and
`z4 = 2`.

## The names a class is spoken in

Two literatures name the indicators, and differently: `irreptables` ships
linear forms in the momentum-space symmetry vector, and the stable-RSI paper,
Hwang *et al.*,
[*Stable real-space invariants and topology beyond symmetry indicators*, Nat. Commun. **17**, 8564 (2026)](https://doi.org/10.1038/s41467-026-74844-w),
gives named forms in the real-space invariants
([stable real-space invariants](srsi.md)). `canonical` is the class in the
paper's names, the ones to read; `irreptables_named` is the same class in
`irreptables`' own, its formulas carried into the labels pythtqc returns
([reading across to BCS](../background/conventions.md#reading-across-to-bcs));
`as_irreptables()` converts the first into the second:

```python
print(si_free.canonical)
#  {'z2w,3': 1, 'z4': 2, 'z3m,pi': 1, 'z3m,0': 1, 'z6m,0': 5, 'z6m,pi': 5, 'z12': 10}
print(si_free.irreptables_named)
#  {'z2w,3': 1, 'z4': 2, 'z3m,π': 1, 'z3m,0': 1, 'z6m,0': 5, 'z6m,π': 5, 'z12': 10}
print(si_free.as_irreptables() == si_free.irreptables_named, si_free.canonical_tex["z3m,pi"])
#  True z_{3m,\pi}
print(si.canonical, si.canonical_note)
#  None the indicator group of #99.164 is trivial, so there is no class to speak in any vocabulary; the paper's tables carry an indicator row for exactly the 170 grey systems that have one, and this is not one of them
```

Here `z3m,pi` against `z3m,π` is the whole of the difference, and
`canonical_tex` carries the paper's typography for a plot label. Where
`canonical` is `None`, `canonical_note` says why, and two such cases are
verdicts: a trivial indicator group, as lin3g's, has no class to speak in any
vocabulary; and a unitary group, returned for a Hamiltonian that breaks time
reversal, is outside the paper's tables, which are the time-reversal-symmetric
ones. The Smith class is returned in both.

## When it refuses

A refusal is a returned record, falsy, naming the check that stopped the call
and what it measured ([when a call refuses](refusals.md)). A vector written by
hand with two bands at Γ and `M` and four at `X` is no band structure, and a
layer's counts written out by hand carry no representation to build the stack
from, so its rows out of the plane go unmeasured:

```python
print(tqc.symmetry_indicator({"-GM6": 1, "-M6": 1, "-X5": 2}, group).gate)
#  not_a_band_structure
print(tqc.symmetry_indicator(dict(t_free.counts), free.group).gate)
#  indicator_not_evaluable
```

- [`not_a_band_structure`](../reference/refusals.md#not_a_band_structure): the
  content fails the compatibility relations; the refusal names the offending
  Smith coordinates.
- [`incomplete_content`](../reference/refusals.md#incomplete_content): a table
  with a point that did not resolve. The class of a vector with a point missing
  is the class of a different vector, and there is no override.
- [`indicator_not_evaluable`](../reference/refusals.md#indicator_not_evaluable):
  rows out of the plane that neither repeat an in-plane row nor were measured
  on the stack: for want of a measured table, whose representation the stack
  is built from, because `points=` was given, or because a check of the stack
  did not complete, which `context["stack"]` names.
- [`unknown_label`](../reference/refusals.md#unknown_label): a measured label
  the EBR basis does not carry.
- `magnetic_type_out_of_scope`: a magnetic group of type III or IV, which
  `realized_group` never returns
  ([what the package cannot do yet](../background/limits.md)).
- [`ebr_tables`](../reference/refusals.md#ebr_tables),
  [`irrep_tables`](../reference/refusals.md#irrep_tables),
  [`ebr_convention`](../reference/refusals.md#ebr_convention): no EBR data for
  the group.

One check is not a refusal: `verify=True`, the default, recomputes the Smith
form of the EBR matrix in exact integers and checks the tables' factors against
it, and a disagreement raises `InternalError`, the package disagreeing with
itself; report it.

## Every field and argument

The fields of [`SymmetryIndicator`](../reference/records.md#symmetryindicator),
its `completion` and its `two_dimensional` among them, are in the records
reference, and each call's arguments in its entry:
[`symmetry_indicator`](../reference/functions.md#symmetry_indicator),
[`stack_presentation`](../reference/functions.md#stack_presentation), which
builds the decoupled stack on its own, and
[`in_plane_restriction`](../reference/functions.md#in_plane_restriction). A
layer's two environments are [a 2D model](layers.md).
