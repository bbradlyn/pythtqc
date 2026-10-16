# What group does my Hamiltonian have?

`realized_group` measures the space group a tight-binding Hamiltonian obeys and
returns it with the orbitals' Wyckoff positions and the matrices every later
call labels the bands with. This chapter reads what it returns: the group, a
descent when the Hamiltonian has less symmetry than its cell, time reversal, a
conserved operator, and a cell that is a supercell. The models are pythtb's own:
graphene and Haldane's model, Kane–Mele, Fu–Kane–Mele's diamond and the
checkerboard. Why the group is measured rather than read off the positions is in
[how pythtqc reads a Hamiltonian](../background/measurement.md).

## The call

```python
import pythtqc as tqc
from pythtb import Lattice, TBModel
from pythtb.models import checkerboard, fu_kane_mele, graphene, haldane, kane_mele

r = tqc.realized_group(graphene(0.0, -1.0))     # graphene(delta, t): no sublattice offset
print(repr(r))
#  RealizedGroup(P6mm1' #183.186; on a substrate (the default), the free-standing group #191.234 agrees)
model, group, assignment, validated, rep = r
print(group.name, group.number_str)
#  P6mm1' 183.186
print(assignment)
#  OrbitalAssignment(2 orbitals on 2 sites: sites (0, 1), letter b, site symmetry 3m)
print(group is r.group, rep is r.rep)
#  True True
```

## How to read it

- **`P6mm1' #183.186`**: the Hermann–Mauguin symbol and the number the tables
  file the group under. The trailing `1'` marks time reversal adjoined to space
  group 183, so that the group holds each of its operations also combined with
  time reversal: a **grey** group, whose irreducible objects are coreps
  ([the primer](../background/tqc-primer.md#little-groups-irreps-and-coreps)).
  Graphene is a layer, so its group is a slab's: the wallpaper group `p6mm`,
  with vacuum along the layer's normal ([a 2D model](layers.md)).
- **`on a substrate (the default), the free-standing group #191.234 agrees`**:
  a layer's model says nothing about what lies under it. Told nothing, the call
  reads the layer on a substrate, measures it free-standing as well, and checks
  that the two agree ([a 2D model](layers.md)).

The record unpacks into five of its own attributes, so the unpacked `rep` is
`r.rep` itself: `r.model`, the model measured and the one every later call takes
(for a layer, its slab presentation, with the same `H(k)`); `r.group`, the
group, written in the model's own reduced coordinates; `r.assignment`, the
Wyckoff orbit of each orbital, here one orbit at letter `b`, site symmetry
`3m`; `r.validated`, the verdict on each operation; and `r.rep`, the matrix of
each operation on the orbitals, which every band label is read through
([labelling the bands](labels.md)). Everything else the call measured is
reached by name, so keep the record: `.descent`, `.time_reversal` and
`.conserved` are read below, `.environment` and `.cross_check` in
[a 2D model](layers.md), `.precision` and `.positions` in
[numerical models](numerical.md), and `.source` is the model as it was handed
in. [The records reference](../reference/records.md) lists every field.

Every operation carries an index in the tables' own order, which everything
kept per operation is keyed on. `r.operations` lists them, one per coset of the
lattice translations, each by its Seitz symbol `{R|t}` in the conventional cell
of the tables and the Bilbao Crystallographic Server and by its name in the
model's own Cartesian axes, the frame `characters=` is written in; the section
on time reversal prints one, and [the conventions](../background/conventions.md)
read one spelling against the other.

## When the Hamiltonian has less symmetry than its cell

Detune one of Fu–Kane–Mele's four nearest-neighbour bonds. No atom moves, so the
positions still allow the diamond's full group; the Hamiltonian does not.

```python
diamond = fu_kane_mele(1.0, 0.125, [0.4, 0, 0, 0])  # t, soc, one of four bonds detuned
rd = tqc.realized_group(diamond)
print(repr(rd))
#  RealizedGroup(R-3m1' #166.98; descended from #227.129 (72 of 96 coset representatives of the lattice translations broken; these are the single-valued group's, each standing for itself and its negative in the double group))
d = rd.descent
print(d["from"]["name"], "->", d["to"]["name"], d["dropped_by"],
      sorted({b["refuted_by"] for b in d["dropped"]}))
#  Fd-3m1' -> R-3m1' {'hoppings': 72} ['no symmetry matrix']
print(sorted({b["name"] for b in d["dropped"]})[:4])
#  ['C2(x)', "C2(x)'", 'C2(y)', "C2(y)'"]
print(sorted({b["name"] for b in d["dropped"]})[-2:])
#  ['{m(z)|1/4,1/4,1/4}', "{m(z)|1/4,1/4,1/4}'"]
print(rd.assignment)
#  OrbitalAssignment(2 orbitals on 2 sites: sites (0, 1), letter c, site symmetry 3m)
```

The positions allow `Fd-3m1'`; the Hamiltonian obeys a quarter of its
operations, so the cell was lowered to them and identified again as
`R-3m1' #166.98`, the only group returned and the one everything downstream is
read in. `rd.descent` is the measurement, `None` where nothing broke: `from`,
`to`, `dropped_by`, and for each operation dropped its `name` in the model's
Cartesian frame and `refuted_by`. A name carries the operation's translation
where that is not a lattice vector of the cell: the operations that exchange
the diamond's two atoms, its glides among them, carry a quarter of the cell's
diagonal, `1/4,1/4,1/4`. Every entry here reads `no symmetry matrix`: no matrix
with that operation's action on `k` carries `H(k)` onto `H(Rk)`, which is a proof.
The other values, `declaration` among them, are in
[the records reference](../reference/records.md). The Wyckoff letters are the
returned group's, which the tables downstream are keyed on: one orbit at letter
`c` of `R-3m`. Where several subgroups of the largest order fit the
measurement, one documented rule picks the group
([the tie rule](../background/measurement.md#the-descent-and-the-tie-rule)).

A distortion can also be stated before anything is measured: `decorations=`
adds a non-orbital species to the cell, here at the detuned bond's midpoint,
`(⅛, ⅛, ⅛)` in reduced coordinates, and the same group comes back with nothing
to descend from.

```python
stated = tqc.realized_group(diamond, decorations=[([0.125, 0.125, 0.125], "X")])
print(repr(stated))
#  RealizedGroup(R-3m1' #166.98)
```

`typat=`, one species label per orbital, is the other such declaration:
`typat=[0, 1]` says the two sites are different atoms, and left out, sites of
the same orbital content are one species. Both state what the crystal is and
are never checked against the Hamiltonian
([a declaration defines the crystal](../background/measurement.md#a-declaration-defines-the-crystal)).
`descend=False` asks for the cell's own group instead, and refuses
`cell_group_not_realized` where the Hamiltonian breaks any of it, carrying the
verdict on every operation.

## Time reversal is measured

Positions say nothing about time reversal, so the cell is always asked as a grey
group and its operations combined with time reversal are measured like any
other. Haldane's model breaks time reversal with a complex next-neighbour
hopping:

```python
hal = haldane(0.0, -1.0, 0.15)                  # delta, t1, t2; the flux phase phi = pi/2
rh = tqc.realized_group(hal)
print(repr(rh))
#  RealizedGroup(P6 #168; time reversal measured broken; descended from #183.186 (12 of 24 coset representatives of the lattice translations broken); on a substrate (the default), the free-standing group #175 agrees -- a larger magnetic group, of type III, is realized (6 antiunitary operations measured to hold) but is beyond the validated scope)
print(rh.time_reversal["outcome"])
#  measured broken
print(rh.operations)
#  P6 #168: one operation per coset of the lattice translations
#   ind  Seitz           name
#     1  {1|0,0,0}       E
#     2  {3+_001|0,0,0}  C3(z)
#     3  {3-_001|0,0,0}  C3^-1(z)
#     4  {2_001|0,0,0}   C2(z)
#     5  {6-_001|0,0,0}  C6^-1(z)
#     6  {6+_001|0,0,0}  C6(z)
```

The hoppings break half the cell's operations: the six mirrors, and the six
rotations combined with time reversal, time reversal itself among them. On
`P61'`, the grey group of the six rotations, time reversal is measured broken,
and the same cell is identified again without it: `P6 #168`, an ordinary space
group in whose own tables every label is then read. The call issues a
`MagneticScopeWarning` as it does so: the group returned is the unitary part of
the Hamiltonian's magnetic symmetry. That part undersells this model, and the
printout's last clause says by how much: `beyond_grey` names the operations
combined with time reversal that hold.

```python
b = rh.beyond_grey
print(b["n_operations"], "of", b["n_measured"], b["magnetic_type"], [o["name"] for o in b["operations"]])
#  6 of 12 III ["m([010])'", "m(x)'", "m([1-10])'", "m([2-10])'", "m(y)'", "m([110])'"]
```

Twelve operations hold: the six rotations, and the six mirrors combined with time
reversal. Haldane's flux pattern is odd under each vertical mirror and odd under
time reversal, so their product is a symmetry. The twelve form a magnetic group
of type III, with antiunitary operations but no time reversal of its own, and the
group returned is of type I or II: `P6` is correct and in scope, and the
constraint the primed mirrors add is not part of what is returned
([magnetic groups](../background/limits.md#magnetic-groups-of-types-i-and-ii-only)).
The mirrors themselves are broken, so `P61'`, the group time reversal is measured
on, does not contain them combined with time reversal either: those six are
measured in the cell's own group, `P6mm1'`, the last group that asks them, and
named by their index there. Once the group has settled they are read against
`P6`: each holds where the matrices measured make a representation of the
magnetic group it generates with `P6`
([every field of the record](../reference/records.md#every-field)). The cell's
own verdicts, which `descend=False` carries, list the same twelve:

```python
cell = tqc.realized_group(hal, descend=False)
held = cell.context["validated"].operations
print(cell.gate, len(held), [op.name for op in held.values() if op.time_reversal])
#  cell_group_not_realized 12 ["m([010])'", "m(x)'", "m([1-10])'", "m([2-10])'", "m(y)'", "m([110])'"]
```

An operation combined with time reversal whose unitary part is not in the
cell's group at all, as where a declaration has made two sites different, is
never put to the Hamiltonian.

## A conserved operator

At the identity the measurement's solutions are every matrix that commutes with
`H(k)` at every `k`: the model's conserved operators, found with nothing
assumed. Kane–Mele without its Rashba term conserves `s_z`, being two copies of
Haldane's model with opposite flux, one per spin; the Rashba term mixes the
spins.

```python
km = tqc.realized_group(kane_mele(0.0, -1.0, 0.3, 0.0))    # delta, t, soc, rashba
print(km.conserved["dim"])
#  1
print(tqc.realized_group(kane_mele(0.0, -1.0, 0.3, 0.25)).conserved["dim"])
#  0
```

`conserved["dim"]` counts the commuting matrices beyond the identity, and the
record's printout carries it as `additional commuting matrices: 1`; every
operation's solutions then form a two-dimensional space, its own matrix times
anything conserved. A residual spin rotation like this one, a sublattice the
hoppings leave decoupled and an accidental degeneracy all raise the count, and
the labels,
[band representations](../background/tqc-primer.md#band-representations-and-atomic-limits)
and indicators returned are the crystalline group's, the extra symmetry lying
outside it. Where a conserved operator decouples a multiplet from the rest,
what that multiplet carries is beyond the hoppings, and a declaration or
[the lowest-angular-momentum rule](../background/measurement.md#the-lowest-angular-momentum-rule)
settles it ([when to declare](orbitals.md#when-to-declare)).
[`examples/tqc_kane_mele_realized_group.py`](../../examples/tqc_kane_mele_realized_group.py)
shows which degeneracies the operator holds.

## When the cell is a supercell

pythtb's checkerboard puts two orbitals at `(0, 0)` and `(½, ½)` of a square
cell, with on-site energies `−delta` and `+delta`. At `delta = 0` they are
copies, and the Hamiltonian obeys the pure translation by `(½, ½)`: the model is
written on a supercell of its own primitive cell, a square lattice of half the
area turned by 45°. The bands at one `k` of the supercell are the crystal's at
two `k` of its own zone, a combination no table names, so every label would name
the wrong k-points and the call refuses `supercell`, at any index. At
`delta ≠ 0` the on-site energies tell the copies apart, and the cell loses the
half of its operations that carry the translation:

```python
out = tqc.realized_group(checkerboard(0.0, -1.0))  # checkerboard(delta, t)
print(out.gate, out.context["index"], out.context["translations"])
#  supercell 2 ((0.5, 0.5, 0.0),)
rc = tqc.realized_group(checkerboard(0.5, -1.0))
print(repr(rc))
#  RealizedGroup(P4mm1' #99.164; descended from #99.164 (16 of 32 coset representatives of the lattice translations broken); on a substrate (the default), the free-standing group #123.340 agrees)
print(rc.assignment)
#  OrbitalAssignment(2 orbitals on 2 sites: sites (0,), letter a, site symmetry 4mm; sites (1,), letter b, site symmetry 4mm)
```

The symbol stays `P4mm1'`, now on the model's own lattice, with the sublattices
two orbits at letters `a` and `b`. A supercell whose Hamiltonian tells the copies
apart is a primitive model of the bigger cell and is descended to with nothing
declared; `typat=` states the same where the copies are chemically distinct. A
refused model is rebuilt on its primitive cell, whose vectors the refusal
carries in `context["primitive_lattice"]`, and asked again: the checkerboard at
`delta = 0` is a square lattice with one orbital per cell.

```python
square = TBModel(Lattice([[0.5, 0.5], [-0.5, 0.5]], [[0.0, 0.0]], periodic_dirs=[0, 1]))
for step in ([1, 0], [0, 1]):
    square.set_hop(-1.0, 0, 0, step)
print(repr(tqc.realized_group(square)))
#  RealizedGroup(P4mm1' #99.164; on a substrate (the default), the free-standing group #123.340 agrees)
```

The copies need not be written alike. A chain of `(p_x, p_y)` sites whose
second site is written in a frame turned by an angle against the first's is the
same crystal, its bands unchanged, and the translation holds once each site's
orbitals are carried onto its translate's by that turn. A translation is read
as every operation is, with any change of basis within each site, so the call
refuses `supercell` all the same, and says how far the orbitals turn.
`context["dressing"]` holds, for each translation obeyed, the `matrix`
carrying each site's orbitals onto its translate's and the `turn` it makes,
`0.0` where the copies are written alike; written in one basis, the copies are
rebuilt on the primitive cell as above.

Folding a supercell onto its primitive cell is not supported yet
([what the package cannot do yet](../background/limits.md#a-supercell-is-refused-not-folded)).

## When it refuses

The question which group a Hamiltonian has is never refused, with two
exceptions, both about how the model is written:
[`supercell`](../reference/refusals.md#supercell), above, and
[`positions_not_exactly_symmetric`](../reference/refusals.md#positions_not_exactly_symmetric),
orbital centres off the cell's own operations by enough to break them with no
precision declared, which `precision=` settles
([numerical models](numerical.md#the-centres-first-because-moving-them-is-a-gauge)).
A model periodic in fewer than two directions, a chain such as pythtb's `ssh`
or a molecule, is outside what the package reads: it is refused
[`fewer_than_two_periodic_directions`](../reference/refusals.md#fewer_than_two_periodic_directions)
before any group is found, whatever the dimension of its cell, since point,
rod and frieze groups are not read yet. Every other refusal
is a verdict on a question asked after the group, a practical limit, a grammar
error in a declaration or one of the package's checks on itself, and it names
the group it was reached in and carries it in `context["group"]`:

- [`cell_group_not_realized`](../reference/refusals.md#cell_group_not_realized)
  and [`declaration_breaks_group`](../reference/refusals.md#declaration_breaks_group),
  asked for with `descend=False`: the Hamiltonian, or the declared content,
  breaks operations the cell allows;
- [`rep_search_too_large`](../reference/refusals.md#rep_search_too_large),
  [`detector_too_large`](../reference/refusals.md#detector_too_large) and
  [`projection_too_large`](../reference/refusals.md#projection_too_large),
  limits on a search or on memory, which the same call runs past with a larger
  `max_candidates=` or with the `max_bytes=` the refusal names; a declaration of
  what the orbitals are (`characters=`) settles the search instead;
- [`unknown_label`](../reference/refusals.md#unknown_label) and
  [`ambiguous_multiplet`](../reference/refusals.md#ambiguous_multiplet), a
  `characters=` its site cannot read ([saying what the orbitals are](orbitals.md)).

A disagreement inside the package raises `InternalError`;
a malformed argument raises `ValueError` before anything is measured.
[When a call refuses](refusals.md) shows how to read one.

## Further

Every argument: [`realized_group`](../reference/functions.md#realized_group).
Every field of the record: [the records reference](../reference/records.md). The
measurement, the tie rule and what a declaration does to the group:
[how pythtqc reads a Hamiltonian](../background/measurement.md). The diamond's
three phases, read off the hoppings alone:
[the three-dimensional tutorial](../start/tutorial-3d.md) and
[`examples/tqc_fkm_diamond.py`](../../examples/tqc_fkm_diamond.py).
