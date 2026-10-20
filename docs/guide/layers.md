# Is my 2D model on a substrate or free-standing?

A pythtb model with two periodic directions is a layer, and its hoppings say
nothing about what lies under it: its environment is stated.
`pythtqc` reads a layer in one of two environments, on a substrate (the
default) or free-standing, and returns the group of that crystal; it measures
the other as well and checks that the two agree. This
chapter shows both on Kane–Mele, without and with its Rashba term, and what the
environment changes downstream. Why a layer is silent about its environment is
in [how pythtqc reads a Hamiltonian](../background/measurement.md#a-layer-is-silent-about-what-lies-under-it).

## The call

```python
import pythtqc as tqc
from pythtb.models import kane_mele

km0 = kane_mele(0.0, -1.0, 0.3, 0.0)            # kane_mele(delta, t, soc, rashba): no Rashba
sub = tqc.realized_group(km0)
print(repr(sub))
#  RealizedGroup(P6mm1' #183.186; additional commuting matrices: 1; on a substrate (the default), the free-standing group #191.234 agrees)
free = tqc.realized_group(km0, environment="free-standing")
print(repr(free))
#  RealizedGroup(P6/mmm1' #191.234; additional commuting matrices: 1; free-standing, the substrate group #183.186 agrees)
print(sub.cross_check["outcome"], "--", sub.cross_check["note"])
#  agree -- the substrate group is exactly the subgroup of the free-standing layer group that preserves the layer's normal -- every operation with R[2,2] = +1 and no other, in both cosets
```

## How to read it

- **The groups are space groups of a slab.** A layer is presented as a
  three-dimensional cell with vacuum along its normal, reduced `k` unchanged, so
  the group returned is the space group of that slab, which the tables and the
  indicators are written for: `P6mm1' #183.186` on a substrate and
  `P6/mmm1' #191.234` free-standing. Read the number as a statement about the
  layer and what is under it.
- **On a substrate** one side of the layer is different, so every operation that
  turns the layer over, with `R[2,2] = −1`, half of the layer group's, is absent
  before the Hamiltonian is read. `(the default)` says the environment was not
  stated ([why the substrate is the default](../background/measurement.md#a-layer-is-silent-about-what-lies-under-it)).
- **Free-standing**, the full layer group applies and the Hamiltonian selects
  among its operations. Here every one holds: the intrinsic spin–orbit term,
  proportional to `s_z`, is even under the horizontal mirror, which turns a spin
  by π about `z`.
- **`agrees`** is the cross-check. Whichever environment is asked, the other is
  measured independently, and the substrate group must be exactly the
  operations of the free-standing one that preserve the layer's normal. The
  outcome is `agree` here;
  `ambiguous` where an environment's largest subgroups tie, the one the tie
  rule keeps is not what the other measured, and another of them is,
  `refused` where the other environment refused, and `not compared` where the
  declaration is written in a substrate's site irreps, which the free-standing
  layer has no word for
  ([below](#what-the-environment-does-to-the-labels)); the group asked for is
  returned in each. A genuine mismatch is the package disagreeing with itself,
  and raises `InternalError`. `cross_check=False` leaves the other environment
  unmeasured.
- **`additional commuting matrices: 1`** is `s_z`, conserved without the Rashba
  term ([a conserved operator](group.md#a-conserved-operator)).

A slab given real out-of-plane coordinates, `dim_r = 3`, is not silent: its own
`z` coordinates say which environment it is in, and its default is
free-standing. Its non-periodic lattice vector need not lie along the layer's
normal: a slab cut from a crystal along one of the crystal's own lattice
vectors, as pythtb's `cut_piece` cuts it, has one tilted off it. The layer's
group cannot depend on that choice, so the vector is put on the normal with
every orbital where it was. Each orbital's coordinates in the plane then move by
its height times the tilt, a change of gauge that moves no energy, and the
record's `presentation` says so
([every field](../reference/records.md#every-field)). Its class is read as a
flat layer's is, on the decoupled stack, each orbital at its own height
([the group the class belongs to](indicators.md#the-group-the-class-belongs-to)):
a buckled Kane–Mele layer's valence pair reads `(1, 0)` in `ℤ2 × ℤ4`, with
`z2w,3 = 1` and `z4 = 2`, the flat layer's values on the names the two groups
share.

## The horizontal mirror, and what breaks it

An operation composed with the horizontal mirror acts on in-plane `k` exactly
as the operation alone, so on a free-standing layer only the representation
tells the two apart: how each turns the spin, and the mirror's sign on each
orbital ([the horizontal mirror](../background/measurement.md#a-layer-is-silent-about-what-lies-under-it)).
Kane–Mele answers it through its spin. The Rashba term, born of a field normal
to the layer, is odd under the mirror:

```python
turned = {op.name for op in free.operations.values() if op.rotation[2, 2] == -1}
rashba = tqc.realized_group(kane_mele(0.0, -1.0, 0.3, 0.25), environment="free-standing")
print(repr(rashba))
#  RealizedGroup(P6mm1' #183.186; descended from #191.234 (24 of 48 coset representatives of the lattice translations broken; these are the single-valued group's, each standing for itself and its negative in the double group); free-standing, the substrate group #183.186 agrees)
broken = {b["name"] for b in rashba.descent["dropped"]}
print(broken == turned, sorted({b["refuted_by"] for b in rashba.descent["dropped"]}))
#  True ['tensor structure']
print(sorted(name for name in turned if not name.endswith("'")))
#  ['C2([010])', 'C2([1-10])', 'C2([110])', 'C2([2-10])', 'C2(x)', 'C2(y)', 'S3(z)', 'S3^-1(z)', 'S6(z)', 'S6^-1(z)', 'i', 'm(z)']
```

The broken operations are exactly the ones that turn the layer over: the
six in-plane twofold axes, the rotoinversions `S3` and `S6`, the inversion `i`
and the mirror `m(z)`, and each of them combined with time reversal. Each acts
on in-plane `k` as a partner that preserves the layer's normal and holds, so `k`
alone cannot refute
it, and `refuted_by` reads `tensor structure`: the matrix it needs turns the
spin as the operation does, and the Rashba term admits only its partner's. With
every operation that turns the layer over gone, the free-standing group is the substrate's,
and the cross-check says so. Besides a spin structure like this one, real
out-of-plane geometry such as buckling breaks the horizontal mirror, and so does
a declaration that gives coupled orbitals opposite signs under it, `s` on one
sublattice and `pz` on the other; a substrate removes it. Without spin, nothing
in the hoppings measures the sign a flat layer's coupled orbitals carry under
the mirror: it is the parity declared, or the even one, so a pair that rotates
as `(x, y)` reads as `(p_x, p_y)` unless `(d_zx, d_yz)` is declared.

## What the environment does to the labels

A declaration is read in the site group of the environment asked. Free-standing,
the honeycomb site's group is `-6m2`, which holds the horizontal mirror; on a
substrate it is `3m`, which does not:

```python
pz_free = tqc.realized_group(km0, environment="free-standing", characters={0: "pz"})
pz_sub = tqc.realized_group(km0, characters={0: "pz"})
for r in (free, pz_free, sub, pz_sub):
    print(r.environment, r.group.name, r.assignment.site_symmetry[0], r.rep.characters.irreps)
#  free-standing P6/mmm1' -6m2 ('-E1', '-E1')
#  free-standing P6/mmm1' -6m2 ('-E2', '-E2')
#  substrate P6mm1' 3m ('-E1', '-E1')
#  substrate P6mm1' 3m ('-E1', '-E1')
```

With nothing declared, the orbitals read `s`-like, by
[the lowest-angular-momentum rule](../background/measurement.md#the-lowest-angular-momentum-rule),
and carry with their spin the double-valued site irrep `-E1` of `-6m2`, the
leading `-` marking a spinful label. A `p_z` orbital is odd under the
mirror, so declared `pz` the same layer carries `-E2`, and every band label built
on it follows. The group does not change: a layer of odd orbitals is symmetric
under the mirror with character `−1`. On a substrate the mirror is not in the
site group, and `s` and `p_z` carry the same `-E1` of `3m`. The hoppings cannot
tell `s` from `p_z`, so which one a layer holds is a declaration
([saying what the orbitals are](orbitals.md)).

With nothing declared, each environment applies that rule in its own site groups,
and the two can choose differently where the horizontal mirror matters: a pair
of orbitals read as `p_z` and `p_-` on a substrate, where `p_z` keeps the
company of `s`, is read as `p_+` and `p_-` free-standing, where the mirror sets
`p_z` apart, and the same bands then carry labels that differ by a
one-dimensional character. The two environments are different crystals, and the
cross-check compares their groups, not their labels. Declared, nothing is left
to the rule, and the free-standing labels restrict to the substrate's.

A declaration in one environment's site irreps says which environment is meant.
`A1` is an irrep of the substrate's `3m` and of nothing in the free-standing
layer's `-6m2`, so on a substrate it is read, and the free-standing layer, which
has no word for it, is not compared:

```python
a1 = tqc.realized_group(km0, characters={0: "A1"})
print(repr(a1))
#  RealizedGroup(P6mm1' #183.186; additional commuting matrices: 1; on a substrate (the default), not compared with the free-standing layer: the declaration is written in the substrate's site irreps)
```

Free-standing, the same word is refused ([below](#when-it-refuses)): a layer
meant free-standing is declared in its own site irreps, or in function words,
which both environments read. A free-standing layer's own words are read on a
substrate by restriction: `A2''` of `-6m2` is the `A1` of `3m` it restricts to,
so the substrate reads the crystal the layer was declared to be, and the two
are compared.

## Inversion in a layer

```python
for r in (sub, free):
    print(r.environment, [op.name for op in r.operations.values() if op.name in ("C2(z)", "i", "m(z)")])
#  substrate ['C2(z)']
#  free-standing ['C2(z)', 'i', 'm(z)']
```

What a substrate keeps is `C2(z)`, a rotation whose action on in-plane `k` is
the one inversion would have; a wallpaper group holds no inversion
([inversion in a layer](../background/measurement.md#a-layer-is-silent-about-what-lies-under-it)).
Inversion `i = m(z)·C2(z)` is in the group only where the horizontal mirror
holds: its spin rotation squares to `+E` where that of `C2(z)` squares to `Ē`,
and it carries the parity eigenvalues `±1` that the Fu–Kane parity criterion
multiplies. Asking for it is one keyword: `environment="free-standing"`.
[`examples/tqc_kane_mele_three_presentations.py`](../../examples/tqc_kane_mele_three_presentations.py)
asks one layer several ways and shows that its indicator class belongs to the
crystal, not to the presentation; how a layer's indicator is completed out of
the plane is in
[a layer whose group has an indicator](indicators.md#a-layer-whose-group-has-an-indicator).

`environment=` supplies the cell's base, and `decorations=` adds species on top
of it, none by default: a crystal field the positions do not show is stated that
way in either environment, and in three dimensions.

## When it refuses

- [`unknown_label`](../reference/refusals.md#unknown_label): an irrep name from
  the other environment's site group. `A1'` names a free-standing site's irrep,
  and on a substrate the call refuses it, naming the group it identified:

  ```python
  out = tqc.realized_group(km0, characters={0: "A1'"})
  print(str(out).splitlines()[0])
  #  Refusal[unknown_label]: The group identified is P6mm1' #183.186, every operation of it obeyed; its representation was not built: orbital 0: "A1'" is not an irrep of this site group, a sum of its irreps, a shell or a known shorthand.
  ```

  A substrate's `A1` is refused on a free-standing call the same way. Function
  words, `s`, `pz` and the rest, are read in either environment.
- [`fewer_than_two_periodic_directions`](../reference/refusals.md#fewer_than_two_periodic_directions):
  a model periodic in one direction, a chain or a ribbon such as pythtb's `ssh`,
  or in none, a molecule, whatever the dimension of its cell; neither is read yet
  ([fewer than two periodic directions](../background/limits.md#fewer-than-two-periodic-directions)).

## Further

The arguments `environment=`, `cross_check=` and `decorations=`:
[`realized_group`](../reference/functions.md#realized_group). The fields
`environment`, `environment_declared` and `cross_check`:
[the records reference](../reference/records.md). Why a layer is silent, and how
its out-of-plane rows are measured:
[how pythtqc reads a Hamiltonian](../background/measurement.md#a-layer-is-silent-about-what-lies-under-it).
