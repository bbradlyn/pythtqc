# What group does my Hamiltonian have?

`realized_group` takes a pythtb model and returns the space group its
Hamiltonian obeys. The record returned also holds the Wyckoff position of each
orbital and the representation of the group on the orbitals.

## The call

`realized_group` takes a pythtb `TBModel` periodic in two or three directions:
a crystal, or a layer. Nothing else is required. The cell is the lattice with
the orbitals' positions, each marked by its species (`typat=` below). The call
finds the cell's space group, with time reversal adjoined. It is the largest
group the model can have. The call checks each operation of that group against the
Hamiltonian, and lowers the group to a group of the operations the Hamiltonian
obeys ([How pythTQC reads a Hamiltonian](../background/measurement.md)). Every
Hamiltonian periodic in two or three directions obeys some space group, `P1` at
the least.

The example is graphene, `graphene(delta, t)` from `pythtb.models`, with equal
on-site energies on its two sublattices: a layer with one orbital on each site
of the honeycomb.

```python
import pythtqc as tqc
from pythtb import Lattice, TBModel
from pythtb.models import checkerboard, fu_kane_mele, graphene, haldane, kane_mele

r = tqc.realized_group(graphene(0.0, -1.0))     # graphene(delta, t): equal on-site energies
print(repr(r))
#  RealizedGroup(P6mm1' #183.186; on a substrate (the default), the free-standing group P6/mmm1' #191.234 agrees)
print(r.group.name, r.group.number_str)
#  P6mm1' 183.186
for o in r.wyckoff_orbits:
    print(repr(o))
#  WyckoffOrbit(2b (3m1'): orbitals 0-1, A1)
```

## How to read it

- `P6mm1' #183.186`: the group the Hamiltonian obeys, `r.group`, by its
  Hermann-Mauguin symbol, `r.group.name`, and its number, `r.group.number_str`.
  The trailing `1'` marks time reversal adjoined to space group 183. Such a
  group is a grey group, which holds each of its operations also combined with
  time reversal. The bands of a Hamiltonian with a grey group carry coreps of
  its little groups
  ([Little groups, irreps and coreps](../background/tqc-primer.md#little-groups-irreps-and-coreps)).
  A grey group's number is its number among the magnetic space groups in the
  Belov-Neronova-Smirnova (BNS) numbering. A group without time reversal is an
  ordinary space group, and has its number in the International Tables for
  Crystallography, as `P6 #168` does below.
- `on a substrate (the default)`: the environment the layer was read in,
  `r.environment.kind`. A layer's model does not say what lies under it. A
  layer is read on a substrate, where no operation turns the layer over, unless
  `environment=` says otherwise. `(the default)` is printed where
  `environment=` was not given.
- `the free-standing group P6/mmm1' #191.234 agrees`: the same layer measured
  free-standing, `r.environment.cross_check`. The two groups agree: `P6mm1'` is
  the subgroup of `P6/mmm1'` whose operations do not reverse the layer's normal.
- `WyckoffOrbit(2b (3m1'): orbitals 0-1, A1)`: one Wyckoff orbit of the
  orbitals, a row of `r.wyckoff_orbits`. `2b` is its Wyckoff position,
  multiplicity and letter, and `3m1'` its site-symmetry group, `3m` with time
  reversal. `orbitals 0-1` are the model's orbitals on the orbit, and `A1` is
  the site irrep the orbital at each site carries. The hoppings do not fix
  graphene's `A1`. It is assigned by
  [the lowest-angular-momentum rule](../background/measurement.md#the-lowest-angular-momentum-rule),
  and `characters=` can declare another site irrep
  ([How do I say what my orbitals are?](orbitals.md)).

Graphene is a layer, so its group is that of a slab: the layer in a
three-dimensional cell, with vacuum along its normal
([Is my 2D model on a substrate or free-standing?](layers.md)).

Every later call takes `r` whole. It holds the model measured, `r.model`, and
the matrix of each operation on the orbitals, `r.rep`, through which every band
label is read ([Which coreps do these bands carry?](labels.md)). For a layer,
`r.model` is its slab, with the same `H(k)`. `r.descent`, `r.time_reversal`,
`r.beyond_grey` and `r.conserved` are read below. The declared precision and
the centers moved onto symmetric positions, under `r.numerical`, are read in
[How do I measure a numerical or wannier90 model?](numerical.md).
[Returned records](../reference/records.md) lists every field.

## When the Hamiltonian has less symmetry than its cell

A Hamiltonian can break operations its orbitals' positions allow.
`realized_group` then lowers the cell's group to a group of the operations
measured to hold. The group returned is that group, and every later call labels
the bands by little-group representations of it. `r.descent` records what was
dropped, and is `None` where nothing was.

The example is Fu-Kane-Mele's diamond, `fu_kane_mele(t, soc, dt)` from
`pythtb.models`, with one of its four nearest-neighbor bonds detuned. No atom
moves, so the positions still allow the diamond's full group. The Hamiltonian
does not obey it.

```python
diamond = fu_kane_mele(1.0, 0.125, [0.4, 0, 0, 0])  # t, soc, one of four bonds detuned
rd = tqc.realized_group(diamond)
print(repr(rd))
#  RealizedGroup(R-3m1' #166.98; descended from Fd-3m1' #227.129 (72 of 96 coset representatives of the lattice translations broken; these are the single-valued group's, each standing for itself and its negative in the double group))
d = rd.descent
print(d.cell_group.name, "->", d.subgroup.name, d.dropped_by,
      sorted({b.refuted_by for b in d.dropped}))
#  Fd-3m1' -> R-3m1' {'hoppings': 72} ['no symmetry matrix']
print(sorted({b.name for b in d.dropped})[:4])
#  ['C2(x)', "C2(x)'", 'C2(y)', "C2(y)'"]
print(sorted({b.name for b in d.dropped})[-2:])
#  ['{m(z)|1/4,1/4,1/4}', "{m(z)|1/4,1/4,1/4}'"]
for o in rd.wyckoff_orbits:
    print(repr(o))
#  WyckoffOrbit(6c (3m1'): orbitals 0-1, -E1)
```

- `R-3m1' #166.98`: the group returned, `rd.group`. The Hamiltonian obeys a
  quarter of the operations of the cell's group.
- `descended from Fd-3m1' #227.129`: the group the positions allow,
  `d.cell_group`. `d.subgroup`, the group the descent arrives at, is here the
  group returned.
- `72 of 96 coset representatives of the lattice translations broken`: the
  operations the Hamiltonian breaks, one counted for each coset of the lattice
  translations. `d.dropped_by` says what dropped them, here the hoppings.
- `these are the single-valued group's, each standing for itself and its negative in the double group`:
  printed for a model with spin. The operations counted are those of the
  single-valued group. Each stands for two operations of the double group,
  which differ by a rotation through 2π.
- `['no symmetry matrix']`: the `refuted_by` of each operation dropped, which
  says what showed it broken. No matrix on the orbitals carries `H(k)` onto
  `H(Rk)` at every `k`, where `R` is the operation's action on `k`. So the
  Hamiltonian breaks the operation. The other values are listed under
  [The descent](../reference/records.md#the-descent).
- `['C2(x)', "C2(x)'", 'C2(y)', "C2(y)'"]`: the names of operations dropped,
  each written in the Cartesian axes in which the pythtb Lattice is specified.
  A prime marks an operation combined with time reversal.
- `{m(z)|1/4,1/4,1/4}`: a name carries the operation's translation, in the
  model's reduced coordinates, where that is not a lattice vector. The
  operations that exchange the diamond's two atoms carry `1/4,1/4,1/4`, the
  vector from one atom to the other.
- `WyckoffOrbit(6c (3m1'): orbitals 0-1, -E1)`: the orbitals form one orbit at
  `6c` of `R-3m`. Its multiplicity, 6, counts the sites in the hexagonal
  conventional cell, three times the primitive cell's two. The Wyckoff letters
  and site-symmetry groups are those of the group returned. `-E1` is a
  double-valued site irrep, since the model has spin.

Several subgroups of the largest order can fit the measurement: where the
operations that hold do not close into a group, where a declaration leaves
several readings, and where spinor states read without their spin carry an
ordinary representation on several subgroups. One rule picks the group returned
([The descent, and the tie rule](../background/measurement.md#the-descent-and-the-tie-rule)).

A distortion can also be declared before anything is measured. `decorations=`
takes a list of pairs. Each is a point that carries no orbital: a position in
reduced coordinates and a species label. The call adds those points to the cell
before its group is found. A decoration of species `"X"` at the detuned bond's
midpoint, `(1/8, 1/8, 1/8)`, gives the cell the group `R-3m1'`. The same group
is returned, with nothing to descend from:

```python
stated = tqc.realized_group(diamond, decorations=[([0.125, 0.125, 0.125], "X")])
print(repr(stated))
#  RealizedGroup(R-3m1' #166.98)
```

`typat=` is the other declaration of this kind. It takes one species label per
orbital, and orbitals on atoms of different species take different labels.
`typat=[0, 1]` declares the diamond's two sites atoms of different species.
With `typat=` left out, sites holding the same number of orbitals are one
species. A declaration states what the crystal is, and is never checked against
the Hamiltonian
([A declaration defines the crystal](../background/measurement.md#a-declaration-defines-the-crystal)).

## Time reversal is measured

Positions say nothing about time reversal, so the cell's group is always taken
with time reversal adjoined, a grey group. Its operations combined with time
reversal are measured like any other. Where time reversal is broken, the group
returned is an ordinary space group. The call then issues a
`MagneticScopeWarning`, which says that the group returned is the unitary
subgroup of the Hamiltonian's magnetic group.

The example is Haldane's model, `haldane(delta, t1, t2)` from `pythtb.models`,
which breaks time reversal with a complex next-nearest-neighbor hopping.

```python
hal = haldane(0.0, -1.0, 0.15)                  # delta, t1, t2
rh = tqc.realized_group(hal)
print(repr(rh))
#  RealizedGroup(P6 #168; time reversal measured broken; descended from P6mm1' #183.186 (12 of 24 coset representatives of the lattice translations broken); on a substrate (the default), the free-standing group P6/m #175 agrees -- a larger magnetic group, of type III, is realized (6 antiunitary operations measured to hold) but is beyond the validated scope)
print(rh.time_reversal.outcome)
#  measured broken
print(rh.operations)
#  P6 #168: one operation per coset of the lattice translations
#  index  Seitz           name
#      1  {1|0,0,0}       E
#      2  {3+_001|0,0,0}  C3(z)
#      3  {3-_001|0,0,0}  C3^-1(z)
#      4  {2_001|0,0,0}   C2(z)
#      5  {6-_001|0,0,0}  C6^-1(z)
#      6  {6+_001|0,0,0}  C6(z)
print(rh.operations[2].summary())
#  operation 2: {3+_001|0,0,0} in the tables' setting, C3(z) in the model's frame   (.index, .seitz, .name)
#    its rotation and translation in the model's reduced coordinates   (.rotation, .translation)
#    its rotation in the model's Cartesian axes   (.rotation_cartesian)
```

- `P6 #168`: the group returned, an ordinary space group. The bands are then
  labeled by irreps of its little groups.
- `time reversal measured broken`: `rh.time_reversal.outcome`. Time reversal was
  measured on `P61'`, the grey group of the six rotations. That is the group the
  descent arrives at, `rh.descent.subgroup`, and time reversal is broken there.
- `descended from P6mm1' #183.186 (12 of 24 coset representatives of the lattice translations broken)`:
  the hoppings break the six mirrors, and the six rotations combined with time
  reversal, time reversal itself among them.
- `a larger magnetic group, of type III, is realized (6 antiunitary operations measured to hold) but is beyond the validated scope`:
  operations of the cell combined with time reversal hold while time reversal
  itself is broken. `rh.beyond_grey` names them, below.
- `P6 #168: one operation per coset of the lattice translations`: the head of
  the table `rh.operations`, which lists the operations of the group, one for
  each coset of the lattice translations. Each operation has an index, a Seitz
  symbol and a name. The Seitz symbol `{R|t}` is written in the conventional
  cell used by the irreptables package. The name is written in the Cartesian
  axes in which the pythtb Lattice is specified
  ([Operations: a Seitz symbol and a Cartesian name](../background/conventions.md#operations-a-seitz-symbol-and-a-cartesian-name)).
- `operation 2: {3+_001|0,0,0} in the tables' setting, C3(z) in the model's frame`:
  one operation, reached by its index, `rh.operations[2]`. Its `summary()` ends
  each line with the fields that hold that part.

In Haldane's model, `rh.beyond_grey` names the operations of the cell combined
with time reversal that hold:

```python
b = rh.beyond_grey
print(b.n_operations, "of", b.n_measured, b.magnetic_type, [o.name for o in b.operations])
#  6 of 12 III ["m([010])'", "m(x)'", "m([1-10])'", "m([2-10])'", "m(y)'", "m([110])'"]
```

- `6`: the operations combined with time reversal that hold, `b.n_operations`.
- `12`: the twelve operations of the cell's group combined with time reversal,
  each measured, `b.n_measured`.
- `III`: `b.magnetic_type`, the type of the magnetic group they form with the
  group returned.
- `["m([010])'", ...]`: the six vertical mirrors combined with time reversal.

Twelve operations hold: the six rotations, and the six mirrors combined with
time reversal. Haldane's flux pattern is odd under each vertical mirror and odd
under time reversal, so their product is a symmetry. The twelve form a magnetic
group of type III, which has operations combined with time reversal and does
not contain time reversal itself. The group returned is its unitary half, `P6`,
of type I. The labels returned are those of `P6`, and take no account of the six
primed mirrors
([Magnetic groups of types I and II only](../background/limits.md#magnetic-groups-of-types-i-and-ii-only)).

An operation of the cell combined with time reversal holds only where a matrix
measured for it and the matrices of the group returned, up to a one-dimensional
character of that group, make a corepresentation of the magnetic group it
generates with that group. `beyond_grey` names those that hold. An operation
combined with time reversal whose unitary part holds is never named, since with
that unitary part it makes time reversal, which is broken. Here the six
rotations combined with time reversal are such operations.

`beyond_grey` names only operations of the cell's group. An operation combined
with time reversal whose unitary part is not in the cell's group is not
measured, for example one exchanging two sites that `typat=` declares atoms of
different species.

## A conserved operator

For each operation, the measurement finds every matrix that carries `H(k)` onto
`H(Rk)` at every `k`
([The cell, and a verdict on each operation](../background/measurement.md#the-cell-and-a-verdict-on-each-operation)).
For the identity these are the matrices that commute with `H(k)` at every `k`.
They are the model's conserved operators. `r.conserved.dim` is the number of
linearly independent ones besides the identity.

The example is Kane-Mele, `kane_mele(delta, t, soc, rashba)` from
`pythtb.models`. Without its Rashba term it conserves `s_z`, being two copies of
Haldane's model with opposite flux, one for each spin. The Rashba term mixes the
spins.

```python
km = tqc.realized_group(kane_mele(0.0, -1.0, 0.3, 0.0))    # delta, t, soc, rashba
print(km.conserved.dim)
#  1
print(tqc.realized_group(kane_mele(0.0, -1.0, 0.3, 0.25)).conserved.dim)
#  0
```

- `1`: `km.conserved.dim`, the conserved `s_z`.
- `0`: the same model with the Rashba term on.

A count above zero comes from a residual spin rotation, as here, or from
orbitals the hoppings leave decoupled from the rest. The labels,
[band representations](../background/tqc-primer.md#band-representations-and-atomic-limits)
and indicators returned are those of the group returned, and the extra
symmetry lies outside it. Where a conserved operator decouples some orbitals of
a site from the rest, the hoppings do not fix the site irrep those orbitals
carry. A declaration or the lowest-angular-momentum rule then settles it
([When to declare](orbitals.md#when-to-declare)).
[`examples/tqc_kane_mele_realized_group.py`](../../examples/tqc_kane_mele_realized_group.py)
shows a band degeneracy that comes from the conserved operator and not from the
little group.

## When the cell is a supercell

A Hamiltonian that obeys a pure translation of its cell is written on a
supercell of its primitive cell. The bands at one `k` of the supercell are those
at several `k` of the primitive cell's zone, so every label would name the wrong
k-points. The call refuses with `supercell`, however many copies of the
primitive cell the supercell holds, and does not fold the model onto its
primitive cell
([A supercell is refused, not folded](../background/limits.md#a-supercell-is-refused-not-folded)).

The example is pythtb's checkerboard, `checkerboard(delta, t)`: two orbitals at
`(0, 0)` and `(1/2, 1/2)` of a square cell, with on-site energies `-delta` and
`+delta`. At `delta = 0` the two orbitals are copies, and the Hamiltonian obeys
the pure translation by `(1/2, 1/2)`. The model is then written on a supercell.
Its primitive cell is a square of half the area, turned by 45 degrees. At
`delta = 0.5` the on-site energies tell the copies apart, and the cell's group
loses the half of its operations that carry the translation:

```python
out = tqc.realized_group(checkerboard(0.0, -1.0))  # checkerboard(delta, t)
print(out.gate, out.context["index"], out.context["translations"])
#  supercell 2 ((0.5, 0.5, 0.0),)
print(out.context["primitive_lattice"]["cartesian"][:2], out.context["dressing"][0]["turn"])
#  ((0.5, 0.5, 0.0), (0.0, 1.0, 0.0)) 0.0
rc = tqc.realized_group(checkerboard(0.5, -1.0))
print(repr(rc))
#  RealizedGroup(P4mm1' #99.164; descended from P4mm1' #99.164 (16 of 32 coset representatives of the lattice translations broken); on a substrate (the default), the free-standing group P4/mmm1' #123.340 agrees)
for o in rc.wyckoff_orbits:
    print(repr(o))
#  WyckoffOrbit(1a (4mm1'): orbital 0, A1)
#  WyckoffOrbit(1b (4mm1'): orbital 1, A1)
```

- `supercell`: the check that stopped the call, `out.gate`.
- `2`: the index of the supercell, `out.context["index"]`, the number of copies
  of the primitive cell it holds.
- `((0.5, 0.5, 0.0),)`: the pure translations the Hamiltonian obeys, in reduced
  coordinates, `out.context["translations"]`.
- `((0.5, 0.5, 0.0), (0.0, 1.0, 0.0))`: two in-plane vectors of the primitive
  lattice, in Cartesian coordinates, from `out.context["primitive_lattice"]`.
  They generate the square lattice of half the area turned by 45 degrees.
- `0.0`: `out.context["dressing"][0]["turn"]`, the angle in radians of the
  rotation that carries the orbitals of each site onto those of the site the
  translation takes it to. It is `0.0` where the copies are written alike.
  `out.context["dressing"][0]["matrix"]` is that map, as a matrix on the
  orbitals.
- `descended from P4mm1' #99.164 (16 of 32 ...)`: at `delta = 0.5` the
  operations that carry the half translation are broken. The symbol stays
  `P4mm1'`: the cell's group has the lattice of half the area, and the group
  returned has the lattice the model is written on.
- `1a` and `1b`: the two sublattices are two orbits.

Where the Hamiltonian tells the copies apart, the cell it is written on is its
primitive cell, and the group descends to that cell's group with nothing
declared. `typat=` declares the same where the copies are atoms of different
species. To read a refused model, write it on its primitive cell and call
`realized_group` again. The checkerboard at `delta = 0` is a square lattice with
one orbital per cell:

```python
square = TBModel(Lattice([[0.5, 0.5], [-0.5, 0.5]], [[0.0, 0.0]], periodic_dirs=[0, 1]))
for step in ([1, 0], [0, 1]):
    square.set_hop(-1.0, 0, 0, step)
print(repr(tqc.realized_group(square)))
#  RealizedGroup(P4mm1' #99.164; on a substrate (the default), the free-standing group P4/mmm1' #123.340 agrees)
```

The copies need not be written alike. A square lattice with two `(p_x, p_y)`
sites per cell, the second written in a frame turned against the first's, is
the same crystal as one with both written alike, and has the same bands. The
translation holds once the orbitals of each site are rotated by that angle onto
those of the next. Like every operation, a translation is allowed a change of
basis within each site. So the call refuses with `supercell` here too, and
`turn` gives the angle. To read the model, rewrite the second site in the first
site's frame and build it on the primitive cell, as above.

## A model with free parameters

pythtb allows the hoppings and on-site energies of a model to be specified as
free parameters: names whose values are given when the Hamiltonian is
evaluated. `realized_group` takes the value of each parameter by name through
`params=`. The model is read at those values. The group, the representation
and `r.model` are those of the model with the numbers in place.
`r.numerical.source` is the model as handed in. The calls that follow take `r`
as for any other model. Each set of values is one call. The value of a
parameter is a number. A hopping that acts on spin is therefore given as a
function of its parameter that returns the two-by-two spin block.

The example is the Kane-Mele model as a function of its spin-orbit coupling
strength `lam`, with a staggered potential `delta = 1` and no Rashba term, read
as a free-standing layer:

```python
import numpy as np

sz = np.diag([1.0, -1.0])
named = kane_mele(1.0, -1.0, 0.0, 0.0)          # delta, t, soc, rashba: spin-orbit coupling added below
for site, R, nu in ((0, [0, 1], -1), (0, [1, 0], 1), (0, [1, -1], -1),
                    (1, [0, 1], 1), (1, [1, 0], -1), (1, [1, -1], 1)):
    named.set_hop(lambda lam, nu=nu: 1j * lam * nu * sz, site, site, R)   # i lam nu sigma_z
for lam in (0.06, 0.3):
    r = tqc.realized_group(named, params={"lam": lam}, environment="free-standing")
    si = tqc.symmetry_indicator(tqc.hsp_table(r, bands=2))
    print(lam, r.group.name, r.group.number_str, si.values)
#  0.06 P-6m21' 187.210 (0, 0)
#  0.3 P-6m21' 187.210 (2, 0)
```

The group is the same at both strengths, and the symmetry indicator of the two
lowest bands differs
([Which symmetry-indicator class do these bands occupy?](indicators.md)). The
gap closes in between, near `lam = 0.19`, where `3*sqrt(3)*lam` equals `delta`.

## When it refuses

A refusal is a returned record. It is falsy, and names the check that stopped
the call and what it measured
([What does a refusal mean, and what do I do about it?](refusals.md)). The
question which group a Hamiltonian obeys is not refused, with two exceptions,
both about how the model is written:

- [`supercell`](../reference/refusals.md#supercell): the Hamiltonian obeys a pure
  translation of its cell, above.
- [`positions_not_exactly_symmetric`](../reference/refusals.md#positions_not_exactly_symmetric):
  with neither `precision=` nor `position_tol=` declared, the orbital centers
  miss symmetric positions by enough that the miss alone moves `H(k)` by more
  than the tolerance an operation, or the projection of `H` onto its symmetric
  part, is held to. With `precision=`, or `position_tol=` for the centers
  alone, centers within the declared tolerance are moved onto symmetric
  positions
  ([How do I measure a numerical or wannier90 model?](numerical.md)).

A model periodic in fewer than two directions, a chain such as pythtb's `ssh` or
a molecule, is outside the package's scope. It is refused with
[`fewer_than_two_periodic_directions`](../reference/refusals.md#fewer_than_two_periodic_directions)
before any group is found
([Fewer than two periodic directions](../background/limits.md#fewer-than-two-periodic-directions)).

The other checks concern a practical limit or what is built once the group is
found, chiefly the representation of the group on the orbitals. A refusal
reached after the group is identified names it, and carries it in
`context["group"]`. One reached while the operations of the cell are measured,
as `detector_too_large` is, names the cell's group, an upper bound on what the
Hamiltonian obeys. Three kinds have a step for the user:

- [`rep_search_too_large`](../reference/refusals.md#rep_search_too_large): the
  representation had more than `max_candidates=` choices to search. Declaring
  what the orbitals are with `characters=` settles the search, and a larger
  `max_candidates=` runs it.
- [`detector_too_large`](../reference/refusals.md#detector_too_large) and
  [`projection_too_large`](../reference/refusals.md#projection_too_large): the
  measurement of an operation, or the projection of `H` onto its symmetric
  part, needs more memory than `max_bytes=` allows. The reason names the
  `max_bytes=` that runs the call.
- [`unknown_label`](../reference/refusals.md#unknown_label) and
  [`ambiguous_multiplet`](../reference/refusals.md#ambiguous_multiplet): a word
  in `characters=` that is not a function, a shell or a site irrep at its
  orbital's site, or a declaration in `characters=` that does not say which
  orbitals hold one site irrep
  ([How do I say what my orbitals are?](orbitals.md)).

[`realized_group`](../reference/functions.md#realized_group) lists every check
the call can return. A malformed argument raises `ValueError` before anything
is measured.

## Where to go next

- [Functions](../reference/functions.md) gives every argument of
  [`realized_group`](../reference/functions.md#realized_group).
- [Returned records](../reference/records.md) lists every field of
  [`RealizedGroup`](../reference/records.md#realizedgroup) and of
  [the descent](../reference/records.md#the-descent).
- [How pythTQC reads a Hamiltonian](../background/measurement.md) gives the
  measurement, the tie rule and what a declaration does to the group.
- [Fu-Kane-Mele: a 3D crystal, a distortion, the Z4 indicator](../start/tutorial-3d.md)
  and [`examples/tqc_fkm_diamond.py`](../../examples/tqc_fkm_diamond.py) lower
  the diamond's symmetry with a bond distortion and read the
  symmetry-indicator class of the three phases it makes.
