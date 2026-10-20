# What the package cannot do yet

This page lists what `pythtqc` does not do yet, each limit once: what a call does instead,
and where the rest of the documentation treats it. A limit is a statement about the
package: where a call meets one it refuses, or returns what it can measure, and what it
returns holds for the group it names. The blocks run on pythtb's
`ssh` chain and on models built in place; the words are
[the glossary's](../reference/glossary.md).

## Fewer than two periodic directions

The package reads crystals periodic in three directions and layers periodic in two, whose
symmetry is a space group. A model periodic in fewer is refused before anything is
measured: with one periodic direction it is a chain or a ribbon, whose symmetry is a rod
or frieze group, and with none a molecule, whose symmetry is a point group.

```python
import pythtqc as tqc
from pythtb import Lattice, TBModel
from pythtb.models import ssh

chain = tqc.realized_group(ssh(1.0, 0.5))            # ssh(v, w): intra- and intercell hopping
print(chain)
#  Refusal[fewer_than_two_periodic_directions]: this model has one periodic direction in its one-dimensional cell, along lattice vector 0: it is a chain or a ribbon, whose symmetry is a rod or frieze group. Neither a molecule nor a chain is in this package's scope yet: it reads a crystal periodic in three directions and a layer periodic in two, whose symmetry is a space group, and a model periodic in fewer is refused whatever the dimension of its cell.  (measured 1, threshold 2)
#      context: dim_r, periodic_dirs
```

The refusal is the same whatever the dimension of the cell the model is written in: the
same chain in a plane and in space, and one orbital in a plane with no periodic direction
at all, a molecule.

```python
plane = TBModel(Lattice([[1.0, 0.0], [0.0, 1.0]], [[0.0, 0.0]], periodic_dirs=[0]))
plane.set_hop(-1.0, 0, 0, [1, 0])                    # a chain along x, in a plane
print(tqc.realized_group(plane).gate)
#  fewer_than_two_periodic_directions
space = TBModel(Lattice([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [[0.0, 0.0, 0.0]],
                        periodic_dirs=[0]))
space.set_hop(-1.0, 0, 0, [1, 0, 0])                 # the same chain, in space
print(tqc.realized_group(space).gate)
#  fewer_than_two_periodic_directions
dot = TBModel(Lattice([[1.0, 0.0], [0.0, 1.0]], [[0.0, 0.0]], periodic_dirs=[]))
print(tqc.realized_group(dot))                       # one orbital, no periodic direction
#  Refusal[fewer_than_two_periodic_directions]: this model has no periodic direction in its two-dimensional cell: it is a molecule, whose symmetry is a point group. Neither a molecule nor a chain is in this package's scope yet: it reads a crystal periodic in three directions and a layer periodic in two, whose symmetry is a space group, and a model periodic in fewer is refused whatever the dimension of its cell.  (measured 0, threshold 2)
#      context: dim_r, periodic_dirs
```

A chain is silent about its transverse environment, as a layer is about what lies under
it ([a layer is silent](measurement.md#a-layer-is-silent-about-what-lies-under-it)), and
its rod or frieze group is a question of its own, as a molecule's point group is. Each
waits for that work.

## A supercell is refused, not folded

A model whose Hamiltonian obeys a pure translation of its own cell is written on a
supercell of its primitive cell, and is refused `supercell` at any index: on its folded
zone every label would name the wrong k-points
([when the cell is a supercell](../guide/group.md#when-the-cell-is-a-supercell)). The model
is rebuilt on its primitive cell instead. A layer is refused the same in either
environment, since neither a substrate nor free-standing vacuum breaks a translation in
its plane. Folding a supercell's bands onto the primitive cell may come later, since there
is physics where the extra translation is the point: the critical point between two
topological phases can obey one, as the SSH chain does at equal hoppings, and so can
certain charge-density-wave critical points.

A cell whose positions repeat under a fractional translation the Hamiltonian breaks is
not a supercell, and the group descends onto the model's own lattice. A translation the
Hamiltonian obeys once each site's orbitals are carried onto its translate's by a change
of basis within the site, a frame turned or a phase on an orbital, is a translation, and
the cell is a supercell; a spin is carried unturned, as a translation's own spin rotation
leaves it ([the spin frame is the model's](#the-spin-frame-is-the-models)). Translations
the Hamiltonian obeys only with changes of basis that do not commute, magnetic
translations about a flux, are outside the groups returned: no crystal's translations are
obeyed, and the cell descends to the group of the translations it obeys as written.

## A slab with glued edges is read as the slab it is written on

pythtb's `cut_piece(n, direction, glue_edges=True)` keeps `n` cells of a crystal along one
of its periodic directions and joins the hoppings that leave the last cell onto the first:
the slab closes into a ring, the crystal with periodic boundary conditions over `n` cells,
whose Hamiltonian is the crystal's at the `n` wavevectors `2πm/n` along the cut, folded onto
one zone. Nothing on the model records the glue, only its hoppings, so the package reads
the slab its orbitals sit in and returns that slab's group, which the glued Hamiltonian
obeys:

```python
crystal = TBModel(Lattice([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [[0.0, 0.0, 0.0]],
                          periodic_dirs=[0, 1, 2]))
for bond in ([1, 0, 0], [0, 1, 0], [0, 0, 1]):
    crystal.set_hop(-1.0, 0, 0, bond)                # simple cubic, one s orbital
open_slab = crystal.cut_piece(4, 2)                  # four cells along z
ring = crystal.cut_piece(4, 2, glue_edges=True)      # the same four, top joined to bottom
print(tqc.realized_group(open_slab).group.name, tqc.realized_group(ring).group.name)
#  P4/mmm1' P4/mmm1'
```

The ring also obeys the shift of its cells by one along the cut, the top cell onto the
bottom, which no layer group holds; it commutes with every `H(k)`, so it is among the
record's additional commuting matrices. The group returned is therefore one the ring obeys
and not all of its symmetry, and its band labels at each wavevector in the plane gather the
`n` wavevectors along the cut. For those, read the crystal the slab was cut from.

## Layer groups are not named

A two-dimensional model is returned a three-dimensional space group, that of its slab with
an empty third direction, and its layer group is not reported: its number is the slab's.
A layer's indicator is read in the group the layer was returned, the rows its plane does not
contain measured on a stack of the layer
([a layer's class](../guide/indicators.md#a-layers-class-is-a-three-dimensional-class)).

Whether the layer lies on a substrate or stands free is not in the model, and the
substrate is a default: the record's `environment_declared` says whether it was stated.
A free-standing layer handed in with nothing declared is returned its
one-sided group, without its inversion and the indicator that would read it
([a 2D model](../guide/layers.md)).

## The spin frame is the model's

The package assumes pythtb's spin quantization axis, `+z` in a right-handed frame: an
operation is a symmetry only with its own spin rotation in that convention. Spin along
another axis usually comes from a source written for `−z`, or from a wannier90 spinor
projection or DFT calculation whose spin axis (VASP's `SAXIS`) is set along another
direction. A model written for the opposite quantization axis,
or with its spin blocks turned against its lattice, obeys other operations and is returned
a different group: a true verdict on the model as written. Nothing here converts it; the
conversion is the model's to make
([a model written for the other axis](../guide/spin.md#a-model-written-for-the-other-axis)).

## Magnetic groups of types I and II only

The groups returned are the 230 space groups as unitary groups and the 230 grey groups,
time reversal adjoined, and the package is validated over those two magnetic types alone.

- A Hamiltonian that breaks time reversal is returned the unitary subgroup of its magnetic
  group, of type I, with a `MagneticScopeWarning` saying so. The operations of the cell
  that combine a unitary operation with time reversal are read against the group
  returned once it has settled, and those that hold while time reversal itself is
  broken, the states' matrices representing the magnetic group they make with it, are
  named on the record's `beyond_grey`, with the magnetic type they make, and are not
  part of the group returned: Haldane's model breaks its six mirrors and obeys each
  combined with time reversal, a magnetic group of type III with its six rotations, and
  the group returned is `P6`, the unitary half
  ([time reversal is measured](../guide/group.md#time-reversal-is-measured),
  [every field of the record](../reference/records.md#every-field)).
- An antiunitary operation that is not an operation of the cell, a composite `g·T` obeyed
  while neither factor is, is not searched for: the group returned is the largest one
  measured, with the composite invisible. Everything returned in that group is correct;
  what is not returned is the constraint the composite would add, so a zero indicator on a
  magnetic model is a statement about the group found.
- For types III and IV the data are missing as well as the conventions. The tables read
  here carry their
  [elementary band representations](tqc-primer.md#band-representations-and-atomic-limits)
  and the characters of the unitary part's irreps at the maximal k-points, but not how
  the antiunitary operations glue those irreps at a k-point, nor the site-symmetry coreps,
  and nothing read here carries the coordinates of the magnetic Wyckoff positions. No
  named indicator formulas exist for a spinless group of those types, and the
  [stable real-space invariants](tqc-primer.md#real-space-invariants-and-the-stable-ones)
  are tabulated for the grey groups alone.

## An unpaired spinor orbital

A spinless model whose orbitals are the up and down states of Wannier functions, as a
spinor wannier90 import is, is represented once the pairing is declared with `spin=`
([a wannier90 spinor import](../guide/spin.md#a-wannier90-spinor-import)). Out of reach is
the unpaired spinor orbital: a single Wannier function carrying half-integer spin with no
partner in the model at all, the locally time-reversal-broken magnetic basis. It needs
site declarations in the double group, which the declaration's tables, single-valued,
cannot state, and it waits on the magnetic types as well. A pairing whose two members
sit at different centres, each orbital one spin state with no partner at its own centre,
is that configuration or a mislabelled pairing, and is refused
[`spin_pair_split`](../reference/refusals.md#spin_pair_split) before any group is
identified.

## Missing content is refused, not filled

A symmetry indicator reads every row of the matrix of the group's elementary band
representations, one row per corep at each high-symmetry point. Rows out of a layer's
plane that repeat an in-plane row are filled from it and named, and the others are
measured on a stack of the layer. What is left, a k-point whose labels refused
or a layer whose stack did not pass its checks, is refused (`incomplete_content`,
`indicator_not_evaluable`). Completing missing
content by solving for it against the stable real-space invariants is future work, which
is why the indicator takes no `allow_incomplete`
([symmetry indicators](../guide/indicators.md)).

## Centres noisier than the symmetry finder

The cell's operations are found from the orbital centres, to the finder's tolerance
`symprec`. Centres noisier than that are a less symmetric set of points to the finder, and
with nothing declared the group returned is that set's, as low as `P1`, rather than a
refusal that names the noise. `precision=` is the route
([centres noisier than symprec](../guide/numerical.md#centres-noisier-than-symprec)).

## A precision too coarse for the cell's geometry

A declared precision also sets the tolerance the finder reads positions at, `10 × precision
× L` with `L` the longest lattice vector along the periodic directions
([numerical precision](../guide/numerical.md)). Where that is a sizable fraction of the
distances between the cell's points, the finder cannot tell them apart. The points it is
handed include, where the Hamiltonian obeys fewer operations than the cell allows, a
generic orbit placed to carry a group of the operations that hold, and those lie closer
together than the orbital centres. The call then raises `InternalError`, its message naming
the finder's tolerance as the cause. Kane–Mele with its two sites declared different atoms reads
`P3m11'` up to `precision=0.015` and raises from `0.02`, where the finder reads at 0.2
lattice constants and the centres are 0.58 apart. A model's own precision lies decades
below this; a wannier90 model's is about `1e-6`.

## Two equally near origins

The names of Wyckoff positions, and of the irreps at some k-points, depend on the point
the tables are read about, and they are read about the one nearest the model's own origin
([conventions](conventions.md#the-origin-the-names-are-read-about)). A model whose origin
is exactly as near two such points has no nearest one: a centred layer with its orbitals
at `(½, 0)` and `(0, ½)`, whose group `Cmm2` can be read about any of `(±¼, ±¼)`, all
four equally near. Its names are read about the one `irrep` chooses on the cell, the same
choice in every run; another of them would give the same bands other names, and nothing
about the model picks one.

## Parametrized models

pythtb lets a hopping or an on-site energy be named and its value given at the call
through `params=`. Wherever the call rebuilds the model, for every layer and wherever a
declared precision moves a centre, the named term does not survive the rebuild and pythtb
stops; binding the parameters first is the route:

```python
square = TBModel(Lattice([[1.0, 0.0], [0.0, 1.0]], [[0.0, 0.0]], periodic_dirs=[0, 1]))
square.set_hop("t", 0, 0, [1, 0])                    # a hopping named, its value given later
square.set_hop("t", 0, 0, [0, 1])
try:
    tqc.realized_group(square, params={"t": -1.0})
except ValueError as error:
    print("ValueError:", error)
#  ValueError: Unknown parameter name(s): t
print(repr(tqc.realized_group(square.with_parameters(t=-1.0))))
#  RealizedGroup(P4mm1' #99.164; on a substrate (the default), the free-standing group #123.340 agrees)
```

A sweep over a parameter is one call per value.

## Feature-resolved and Wilson-loop verdicts

The symmetry analysis over spectra, sectors and Wilson loops resolved by a feature,
everything built on the eigenspaces of a projected operator `P Ô P`, the spin-resolved
case `Ô = n̂·s` among them, is planned and not present: sewing matrices projected onto a
sector, the corep content of a sector and verdicts read off a Wilson spectrum. A decomposition reads
the content at isolated k-points only, and is blind to what lies between them
([what a decomposition cannot certify](../guide/atomic-limits.md#what-a-decomposition-cannot-certify)).
