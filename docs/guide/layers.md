# Is my 2D model on a substrate or free-standing?

A pythtb model with two periodic directions is a layer. Its hoppings say nothing
about what lies under it. `realized_group` reads it on a substrate or
free-standing, as `environment=` says, and a default applies where it is not
given. This page reads Kane-Mele in both environments, without and with its
Rashba term. It then reads a slab cut from a crystal, and shows what the
environment changes in the site irreps and in the coreps of the bands. Why a
layer is silent about its environment is in
[A layer is silent about what lies under it](../background/measurement.md#a-layer-is-silent-about-what-lies-under-it).

## The call

`realized_group` takes `environment=`, either `"substrate"` or
`"free-standing"`. On a substrate one side of the layer differs from the other,
so no operation that turns the layer over is in the group. Free-standing, the
layer has vacuum on both sides, and every operation of its layer group is put to
the Hamiltonian. Left out, `environment=` is `"substrate"` for a model written
with two-dimensional positions, `dim_r = 2`. The call returns the group the
Hamiltonian obeys in the environment asked for. It also finds the group in the
other environment and compares the two. With `cross_check=False`, the other
environment is not read, and `cross_check` is `None`.

The example is Kane-Mele, from `pythtb.models`, with spin-orbit coupling and no
Rashba term. It reads the layer on a substrate and free-standing:

```python
import pythtqc as tqc
from pythtb.models import kane_mele

km0 = kane_mele(0.0, -1.0, 0.3, 0.0)            # kane_mele(delta, t, soc, rashba): no Rashba
sub = tqc.realized_group(km0)
print(repr(sub))
#  RealizedGroup(P6mm1' #183.186; additional commuting matrices: 1; on a substrate (the default), the free-standing group P6/mmm1' #191.234 agrees)
free = tqc.realized_group(km0, environment="free-standing")
print(repr(free))
#  RealizedGroup(P6/mmm1' #191.234; additional commuting matrices: 1; free-standing, the substrate group P6mm1' #183.186 agrees)
print(sub.environment.cross_check.outcome, "--", sub.environment.cross_check.note)
#  agree -- the substrate group is exactly the subgroup of the free-standing layer group that preserves the layer's normal -- every operation with R[2,2] = +1 and no other, in both cosets
```

## How to read it

- `P6mm1' #183.186` and `P6/mmm1' #191.234`: the group in each environment. A
  layer is presented as a three-dimensional cell with vacuum along its normal,
  and its in-plane reduced `k` are the cell's. The group returned is the space
  group of that cell, since the irreptables package tabulates coreps,
  elementary band representations and symmetry indicators for space groups.
  The same layer has one space group on a substrate and another free-standing.
- `on a substrate (the default)`: the environment, `sub.environment.kind`. On a
  substrate every operation that turns the layer over is left out before the
  Hamiltonian is read: those whose rotation part `R` has `R[2,2] = −1`, the
  third axis being the layer's normal. Here that is half of the free-standing
  group's operations. `(the default)` says that `environment=` was not given.
- `free-standing`: every operation of the layer group is put to the
  Hamiltonian, and here every one holds. The intrinsic spin-orbit term is
  proportional to `s_z`, which is even under the horizontal mirror `m(z)`,
  since the mirror turns a spin by π about `z`.
- `the free-standing group P6/mmm1' #191.234 agrees`: the cross-check,
  `sub.environment.cross_check`. The call reads the layer in the other
  environment on its own and compares the two groups. The outcome,
  `cross_check.outcome`, is one of four, each with the clause `repr` prints for
  it:
  - `agree`, printed `the free-standing group ... agrees` or
    `the substrate group ... agrees`: the substrate group is exactly the
    subgroup of the free-standing group whose operations preserve the layer's
    normal. `cross_check.note` says so in words, in the example's last line.
    The note's `in both cosets` names the unitary operations and those
    combined with time reversal.
  - `ambiguous`, printed `the other environment ambiguous`: in an environment,
    several subgroups of the largest order qualified with nothing measured
    between them, and the tie rule kept one
    ([The descent, and the tie rule](../background/measurement.md#the-descent-and-the-tie-rule)).
    The two groups agree for another pair of the candidates, which
    `cross_check.note` names, and not for the pair kept.
  - `refused`, printed `the other environment refused`: the call in the other
    environment refused, and the cross-check records its check and reason.
  - `not compared`, printed `not compared with the free-standing layer: ...`
    or `not compared with the substrate: ...`: `characters=` is written in site
    irreps of the environment asked for that name nothing in the other
    ([What the environment does to the labels](#what-the-environment-does-to-the-labels)).

  The group asked for is returned in every case. The two groups returned
  disagree only where the outcome is `ambiguous`, and another pair of the
  candidates agrees there. The two groups are found from one Hamiltonian, and
  no other disagreement is returned.
- `additional commuting matrices: 1`: one matrix beyond the identity commutes
  with `H(k)` at every `k`. Here it is `s_z`, which is conserved without the
  Rashba term ([A conserved operator](group.md#a-conserved-operator)).

## A slab cut from a crystal

A pythtb model with three-dimensional positions and two periodic directions,
`dim_r = 3`, is a slab. Its orbitals have heights, and the heights decide which
operations that turn the layer over hold. So its default environment is
free-standing, and `environment="substrate"` reads it on a substrate. Its
non-periodic lattice vector need not lie along the layer's normal: pythtb's
`cut_piece` keeps the crystal's lattice vectors, so a slab cut from a crystal
can have that vector tilted off the normal. The layer's group does not depend
on that vector, so pythTQC puts it on the normal and keeps every orbital where
it is. The tilt is the vector's in-plane part, in reduced coordinates of the
periodic directions. Each orbital's in-plane reduced coordinates then move by
its reduced height times the tilt. This is a change of gauge of `H(k)`, which
moves no energy. `buckled.environment.presentation.tilted` records the move,
and is `None` where the vector was on the normal already.

The example is one layer of the Fu-Kane-Mele model of diamond, from
`pythtb.models`, cut along its third lattice vector: a buckled honeycomb with
spin-orbit coupling. The symmetry-indicator class of its two lowest bands is
read on the decoupled stack, as a flat layer's is
([The group a class is read in](indicators.md#the-group-a-class-is-read-in)).
The decoupled stack is the layer repeated along its normal with no hopping
between copies, each orbital at its own height:

```python
from pythtb.models import fu_kane_mele

slab = fu_kane_mele(1.0, 0.1).cut_piece(1, 2)   # fu_kane_mele(t, soc), one cell along a3
buckled = tqc.realized_group(slab)
print(repr(buckled))
#  RealizedGroup(P-3m11' #164.86; free-standing (the default), the substrate group P3m11' #156.50 agrees)
tilt = buckled.environment.presentation.tilted
print(tilt.n_moved, round(tilt.max_move, 4))
#  1 0.0833
si = tqc.symmetry_indicator(tqc.hsp_table(buckled, bands=2))
print(si, si.indicators)
#  SymmetryIndicator(P-3m11' #164.86, class (1, 0) in Z2 x Z4, completed on the stack) {'z2w,3': 1, 'z4': 2}
```

- `P-3m11' #164.86`: the group of the buckled layer, which has no horizontal
  mirror and no sixfold rotation. `free-standing (the default)` says the slab
  was read free-standing without `environment=`. On a substrate the same slab
  has the group the cross-check names, `P3m11' #156.50`.
- `1 0.0833`: `tilt.n_moved`, the number of orbitals off the plane through the
  origin, and `tilt.max_move`, the largest change of an orbital's in-plane
  reduced coordinate.
- `class (1, 0) in Z2 x Z4, completed on the stack`: the symmetry-indicator
  class of the two lowest bands. The group's tables have high-symmetry points
  at `k_z = 1/2`, which the layer has no coordinate for.
  `completed on the stack` says that the corep multiplicities there were filled
  in: those that repeat none at a point of the plane were measured on the
  decoupled stack, here at A and L, and the others were taken from the point of
  the plane they repeat, here H from K.
- `{'z2w,3': 1, 'z4': 2}`: the same class in the indicators named by Hwang et
  al., Nat. Commun. 17, 8564 (2026), the paper that tabulates the stable
  real-space invariants. `(1, 0)` and the paper's indicators are two sets of
  coordinates on one class, so their entries differ
  ([Named indicators](indicators.md#named-indicators)).

## The horizontal mirror, and what breaks it

An operation composed with the horizontal mirror acts on in-plane `k` as the
operation alone does. So on a free-standing layer only the representation on
the orbitals tells the two apart: with spin, by how each turns the spin;
without spin, by the mirror's sign on each orbital. The Rashba term comes from
a field normal to the layer, and is odd under the mirror. The example reads
Kane-Mele with its Rashba term free-standing, and compares the operations it
breaks with those that turn the layer over:

```python
turned = {op.name for op in free.operations.values() if op.rotation[2, 2] == -1}
rashba = tqc.realized_group(kane_mele(0.0, -1.0, 0.3, 0.25), environment="free-standing")
print(repr(rashba))
#  RealizedGroup(P6mm1' #183.186; descended from P6/mmm1' #191.234 (24 of 48 coset representatives of the lattice translations broken; these are the single-valued group's, each standing for itself and its negative in the double group); free-standing, the substrate group P6mm1' #183.186 agrees)
broken = {b.name for b in rashba.descent.dropped}
print(broken == turned, sorted({b.refuted_by for b in rashba.descent.dropped}))
#  True ['tensor structure']
print(sorted(name for name in turned if not name.endswith("'")))
#  ['C2([010])', 'C2([1-10])', 'C2([110])', 'C2([2-10])', 'C2(x)', 'C2(y)', 'S3(z)', 'S3^-1(z)', 'S6(z)', 'S6^-1(z)', 'i', 'm(z)']
```

The descent clause reads as in
[When the Hamiltonian has less symmetry than its cell](group.md#when-the-hamiltonian-has-less-symmetry-than-its-cell):
the Rashba term breaks half the operations of the free-standing group. The
operations broken are exactly the ones that turn the layer over: the six
in-plane twofold axes, the rotoreflections `S3` and `S6` and their inverses,
the inversion `i` and the mirror `m(z)`, each also combined with time reversal.
Each acts on in-plane `k` as its partner does, its product with `m(z)`, which
preserves the layer's normal and holds. So its action on `k` cannot show it
broken. `refuted_by` reads `tensor structure`: matrices that carry `H(k)` onto
`H(Rk)` exist, the partner's, and none has the operation's own form, its
orbital matrix times its own spin rotation. The operation's matrix differs from
its partner's by the mirror's, which rotates the spin by π about `z` and
reverses the Rashba term. With every operation that turns the layer over
broken, the free-standing group is the substrate's, and the cross-check says
so.

A spin structure like Rashba's is one way to break the horizontal mirror.
Out-of-plane geometry breaks it too, as in the buckled slab above. So does a
declaration that gives coupled orbitals opposite signs under it, `s` on one
sublattice and `pz` on the other. On a substrate it is no operation of the
group. On a flat layer, the hoppings fix the relative sign of coupled orbitals
under the mirror, and not their common sign. That common sign is the parity
declared, or the even one where nothing is declared, so a pair that rotates as
(x, y) reads as (p_x, p_y) unless (d_zx, d_yz) is declared.

## What the environment does to the labels

A declaration is read in the site-symmetry groups of the environment asked
for. The honeycomb site's site-symmetry group is `-6m2` free-standing, which
holds the horizontal mirror, and `3m` on a substrate, which does not. The
example reads Kane-Mele without its Rashba term in both environments, with
nothing declared and with orbital 0 declared a `pz` orbital, and prints the
site irrep of the honeycomb's orbitals:

```python
pz_free = tqc.realized_group(km0, environment="free-standing", characters={0: "pz"})
pz_sub = tqc.realized_group(km0, characters={0: "pz"})
for r in (free, pz_free, sub, pz_sub):
    o = r.wyckoff_orbits[0]                     # the honeycomb's one Wyckoff orbit
    print(r.environment.kind, r.group.name, o.unitary_site_symmetry, o.label)
#  free-standing P6/mmm1' -6m2 -E1
#  free-standing P6/mmm1' -6m2 -E2
#  substrate P6mm1' 3m -E1
#  substrate P6mm1' 3m -E1
```

With nothing declared, the orbitals carry the site irrep of an `s` orbital, by
[the lowest-angular-momentum rule](../background/measurement.md#the-lowest-angular-momentum-rule).
With its spin, an `s` orbital carries the site irrep `-E1` of `-6m2`, the
leading `-` marking it double-valued. A p_z orbital is odd under the horizontal
mirror, so declared `pz`, the same layer carries `-E2`, and the coreps of the
bands built on it change with it. The group does not change: the mirror changes
the sign of each p_z orbital, and the layer is symmetric under it. On a
substrate the mirror is not in the site-symmetry group, and `s` and p_z carry
the same `-E1` of `3m`. The hoppings cannot tell `s` from p_z, so which one a
layer holds is a declaration
([How do I say what my orbitals are?](orbitals.md)).

With nothing declared, each environment applies that rule in its own
site-symmetry groups, and the two can assign different site irreps where the
horizontal mirror matters. Kane-Mele with its Rashba term shows it, read
without its spin through `spinless_presentation`
([How does pythTQC read the spin of my model?](spin.md)). Each site then holds
two orbitals, its two spin components. Read so, the group has no time reversal
and no vertical mirror. It is `P6` on a substrate, where the honeycomb site's
site-symmetry group is `3`, and `P6/m` free-standing, where it is `-6`. The
rule assigns `A1` and `2E` of `3` on a substrate, and `2E'` and `1E'` of `-6`
free-standing, which restrict to `2E` and `1E` of `3`. Restricted to `3`, the
free-standing assignment is the substrate's multiplied by the one-dimensional
irrep `2E`, and the irreps the same bands carry differ by it too. The two
environments are different crystals, and the cross-check compares their groups,
not the irreps of their bands. Where the orbitals are declared, nothing is left
to the rule, and the free-standing little-group representations restrict to the
substrate's.

A site irrep is named in the site-symmetry group of one environment, and a call
refuses a name that its own site-symmetry groups do not have. `A1` is an irrep
of the substrate's `3m` and of nothing in the free-standing `-6m2`. Kane-Mele
without its Rashba term, declared `A1`, is read on a substrate, and the
cross-check does not compare it with the free-standing layer, which has no
irrep of that name:

```python
a1 = tqc.realized_group(km0, characters={0: "A1"})
print(repr(a1))
#  RealizedGroup(P6mm1' #183.186; additional commuting matrices: 1; on a substrate (the default), not compared with the free-standing layer: the declaration is written in the substrate's site irreps)
```

A free-standing call refuses `A1`, and a substrate call refuses `A2''` of
`-6m2` ([When it refuses](#when-it-refuses)). The orbital names, `s`, `pz` and
the rest, are read in either environment. A free-standing call declared in its
own site irreps is compared: the cross-check reads `A2''` of `-6m2` on the
substrate as the `A1` of `3m` it restricts to.

## Inversion in a layer

The example prints which of `C2(z)`, `i` and `m(z)` Kane-Mele without its
Rashba term keeps in each environment:

```python
for r in (sub, free):
    print(r.environment.kind, [op.name for op in r.operations.values() if op.name in ("C2(z)", "i", "m(z)")])
#  substrate ['C2(z)']
#  free-standing ['C2(z)', 'i', 'm(z)']
```

On a substrate the group keeps `C2(z)`, a rotation whose action on in-plane `k`
is the one inversion would have. A substrate's group holds no inversion, since
inversion turns the layer over. Where `C2(z)` holds, as here, inversion
`i = m(z)·C2(z)` is in the group exactly when the horizontal mirror is. Its
spin rotation squares to the identity `E`, where that of `C2(z)` squares to
`Ē`, the rotation by 2π. Its eigenvalues `±1` are the parities that the
Fu-Kane parity criterion multiplies.
[`examples/tqc_kane_mele_three_presentations.py`](../../examples/tqc_kane_mele_three_presentations.py)
reads one layer three ways: free-standing, free-standing with p_z declared, and
as its decoupled stack. It reads the symmetry-indicator class each way and
compares the three. How the corep multiplicities at `k_z = 1/2` are found for a
layer is in
[A layer whose group has an indicator](indicators.md#a-layer-whose-group-has-an-indicator).

## When it refuses

Every model periodic in two directions has a space group. The question which
one is not refused, with two exceptions, both about how the model is written:
`supercell` and `positions_not_exactly_symmetric`. A refusal is a returned
record. It is falsy, and names the check that stopped the call and what it
measured ([What does a refusal mean, and what do I do about it?](refusals.md)).
A refusal after the group is identified names the group. The two exceptions
and the other checks any model can meet are on
[What group does my Hamiltonian have?](group.md#when-it-refuses).

One check is met on a layer in particular,
[`unknown_label`](../reference/refusals.md#unknown_label), for a declaration in
the other environment's site irreps. Kane-Mele without its Rashba term,
declared `A1'`, an irrep of the free-standing site-symmetry group `-6m2`, is
refused on a substrate, naming the group identified:

```python
out = tqc.realized_group(km0, characters={0: "A1'"})
print(str(out).splitlines()[0])
#  Refusal[unknown_label]: the group identified is P6mm1' #183.186, every operation of it obeyed; orbital 0: "A1'" is not an irrep of this site group, a sum of its irreps, a shell or a shorthand. Declare one of the words listed below.
```

The lines that follow it list the words the site accepts
([When it refuses](orbitals.md#when-it-refuses)).

A model periodic in fewer than two directions is not a layer: a chain or a
ribbon such as pythtb's `ssh`, periodic in one, or a molecule, periodic in none,
whatever the dimension of its positions. It is refused with
[`fewer_than_two_periodic_directions`](../reference/refusals.md#fewer_than_two_periodic_directions)
before any group is sought. Its symmetry is a rod, frieze or point group, and
the package reads space groups only
([Fewer than two periodic directions](../background/limits.md#fewer-than-two-periodic-directions)).

## Where to go next

- [Functions](../reference/functions.md) gives the arguments of
  [`realized_group`](../reference/functions.md#realized_group),
  `environment=`, `cross_check=` and `decorations=` among them.
- [Returned records](../reference/records.md) lists every field of
  [`RealizedGroup`](../reference/records.md#realizedgroup), among them
  `environment` with its `kind`, `declared`, `cross_check` and `presentation`.
- [When the Hamiltonian has less symmetry than its cell](group.md#when-the-hamiltonian-has-less-symmetry-than-its-cell)
  states a crystal field with `decorations=`, which adds species to the cell in
  either environment.
- [How pythTQC reads a Hamiltonian](../background/measurement.md) says why a
  layer is silent about its environment. Its section
  [A layer's out-of-plane rows](../background/measurement.md#a-layers-out-of-plane-rows)
  says how the corep multiplicities at `k_z = 1/2`, which a layer has no
  coordinate for, are found.
