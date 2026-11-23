# What the package does not cover

This page lists what pythTQC does not cover, each limit once. For each it says
what the call does instead, and links the page that treats it. Where a call meets
a limit it refuses, raises, or returns what it can measure. What a call returns
is correct for the group it names, which can be smaller than all of the model's
symmetry. The examples are pythtb's SSH chain, Kane-Mele and Haldane models, and
models built in place. The terms are defined in the
[Glossary](../reference/glossary.md).

## Fewer than two periodic directions

pythTQC reads crystals periodic in three directions and layers periodic in two,
whose symmetry is a space group. A model periodic in fewer directions is refused
with `fewer_than_two_periodic_directions` before anything is measured. With one
periodic direction it is a chain or a ribbon, whose symmetry is a rod or frieze
group. With none it is a molecule, whose symmetry is a point group.

The example is the SSH chain, `ssh(v, w)` from `pythtb.models`:

```python
import pythtqc as tqc
from pythtb import Lattice, TBModel
from pythtb.models import ssh

chain = tqc.realized_group(ssh(1.0, 0.5))            # ssh(v, w): intra- and intercell hopping
print(chain)
#  Refusal[fewer_than_two_periodic_directions]: this model has one periodic direction in its one-dimensional cell, along lattice vector 0: it is a chain or a ribbon, whose symmetry is a rod or frieze group.  (measured 1, threshold 2)
#      context: dim_r, periodic_dirs
```

- `Refusal[fewer_than_two_periodic_directions]`: the check that stopped the
  call, `chain.gate`. A refusal is a returned record, and it is falsy.
- `this model has one periodic direction ...`: the reason, `chain.reason`.
- `(measured 1, threshold 2)`: the number of periodic directions,
  `chain.value`, and the number the check asks for, `chain.threshold`.
- `context: dim_r, periodic_dirs`: the keys of `chain.context`, the dimension
  of the cell and its periodic directions.

The refusal is the same whatever the dimension of the cell the model is written
in. The example builds the same chain in a plane and in space, and one orbital
in a plane with no periodic direction, a molecule:

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
#  Refusal[fewer_than_two_periodic_directions]: this model has no periodic direction in its two-dimensional cell: it is a molecule, whose symmetry is a point group.  (measured 0, threshold 2)
#      context: dim_r, periodic_dirs
```

## A supercell is refused, not folded

A model whose Hamiltonian obeys a fractional translation, one that is not a
lattice vector of its cell, is written on a supercell of its primitive cell. A
model at a critical point can obey one: at the transition between two
topological phases, as a plane of coupled SSH chains does at equal hoppings, or
at a charge-density-wave critical point. The call refuses such a model with
`supercell`, whatever the index of the supercell. On the folded zone every label
would name the wrong k-points. Rebuilt on its primitive cell, the model is read
as any other
([When the cell is a supercell](../guide/group.md#when-the-cell-is-a-supercell)).
A layer is refused the same in either environment, since neither a substrate
nor vacuum breaks a translation in its plane.

A cell whose positions repeat under a fractional translation that the
Hamiltonian breaks is not a supercell. The group descends onto the model's own
lattice. A translation that holds only together with a change of basis within
each site, a frame turned or a phase on an orbital, is still a translation of
the crystal, and the cell is a supercell. The change of basis acts on the
orbitals and leaves a spin untouched, since the spin rotation of a translation
is the identity ([The spin frame is the model's](#the-spin-frame-is-the-models)).
Translations that hold only with changes of basis that do not commute with one
another, the magnetic translations of a lattice threaded by a flux, are not the
translations of any crystal. The group returned then has as its lattice the
translations the Hamiltonian obeys as written.

## A slab with glued edges is read as the slab it is written on

pythtb's `cut_piece(n, direction, glue_edges=True)` keeps `n` cells of a crystal
along one of its periodic directions, and joins the hoppings that leave the last
cell onto the first. The slab closes into a ring, the crystal with periodic
boundary conditions over `n` cells. Its Hamiltonian is the crystal's at the `n`
wavevectors `2πm/n` along the cut, folded onto one zone. Its orbitals sit in a
slab, open along the cut, and the glue is held only in its hoppings. The call
reads that slab and returns its group, which the glued Hamiltonian obeys.

The example is a simple cubic crystal with one s orbital, cut four cells thick
along z, open and glued:

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

- `P4/mmm1' P4/mmm1'`: the group of the open slab and the group of the ring,
  the same.

The ring also obeys the shift of its cells by one along the cut, which no layer
group holds. The shift commutes with every `H(k)`, so it is among the conserved
operators counted in `r.conserved`, where `r` is the record `realized_group`
returns. The group returned is one the ring obeys, and not all of its symmetry.
The little-group representations returned for the ring's bands at a wavevector
in the plane are those of the crystal's bands at all `n` wavevectors along the
cut together. To label the crystal's bands at each of them, read the crystal the
slab was cut from.

## Layer groups are not named

A model with two periodic directions is returned a space group, that of its
slab: a three-dimensional cell with vacuum along the normal. `r.group` names
that space group, by its symbol and number, and the layer group is not
returned. A layer's symmetry indicator is that of its decoupled stack, read in
the space group returned
([A layer's class is a three-dimensional class](../guide/indicators.md#a-layers-class-is-a-three-dimensional-class)).

Whether the layer lies on a substrate or stands free is not in the model. A
layer written with two-dimensional positions is read on a substrate by default,
and a slab with three-dimensional positions free-standing.
`r.environment.declared` says whether the environment was given. A free-standing
layer written with two-dimensional positions and handed in with nothing declared
is returned its group on a substrate, without the operations that turn the layer
over. The record also names the group in the other environment, in
`r.environment.cross_check`. The example is Kane-Mele,
`kane_mele(delta, t, soc, rashba)` from `pythtb.models`, whose arguments are the
staggered potential, the hopping, the spin-orbit coupling and the Rashba term.
With no staggered potential and no Rashba term, `kane_mele(0.0, -1.0, 0.3, 0.0)`
is returned `P6mm1'`, whose symmetry-indicator group is trivial. Declared
free-standing, with `environment="free-standing"`, it is returned `P6/mmm1'`,
whose symmetry-indicator group is `Z6 x Z12`
([Is my 2D model on a substrate or free-standing?](../guide/layers.md)).

## The spin frame is the model's

pythTQC takes pythtb's spin quantization axis, `+z` in a right-handed frame. An
operation is a symmetry only together with its own spin rotation in that
convention. A model written for another axis, or with its spin written in axes
rotated from those of its lattice, obeys other operations and is returned a
different group. That group is a true verdict on the model as written. Spin
along another axis usually comes from a source written for `−z`, or from a
Wannier90 spinor projection or a DFT calculation whose spin axis (VASP's
`SAXIS`) is set along another direction. The conversion is made where the source
is transcribed
([A model with its spin up along -z](../guide/spin.md#a-model-with-its-spin-up-along--z)).
Kane-Mele, `kane_mele(1.0, -1.0, 0.3, 0.25)` with its two sites declared
different atoms, `typat=[0, 1]`, is returned `P3m11'`. With every spin block
conjugated by `i σ_x`, the same spectrum written for `−z`, it is returned
`Cm1'`.

## Magnetic groups of types I and II only

The groups returned are the 230 space groups, of magnetic type I, and the 230
grey groups, of type II, which have time reversal adjoined.

- A Hamiltonian that breaks time reversal is returned the unitary subgroup of
  its magnetic group, of type I. The call issues a `MagneticScopeWarning` that
  says so. The operations of the cell that hold only combined with time
  reversal are listed in `r.beyond_grey`, with the magnetic type they make with
  the group returned. They are not part of the group returned. Haldane's model,
  `haldane(0.0, -1.0, 0.15)` from `pythtb.models`, with no staggered potential,
  breaks its six mirrors and obeys each combined with time reversal. With its
  six rotations they make a magnetic group of type III, and the group returned
  is `P6`, the unitary half
  ([Time reversal is measured](../guide/group.md#time-reversal-is-measured)).
- The labels, decompositions and symmetry indicators returned for such a model
  are those of the type-I group. The stable real-space invariants (SRSIs) are
  tabulated for the grey groups alone, and every call on them refuses a type-I
  group with `srsi_scope`.
- An operation combined with time reversal whose unitary part is not an
  operation of the cell is not measured. The group returned is the largest one
  measured, and such an operation is not in it. What is returned holds for that
  group and carries no constraint from an operation outside it. A zero
  indicator of a magnetic model is a zero in the indicator group of the group
  returned. Haldane's model with its two sites declared different atoms,
  `typat=[0, 1]`, is returned `P3`. Its `beyond_grey` names three of the six
  mirrors combined with time reversal, and the three that exchange the two
  sites are not measured.

## An unpaired spinor orbital

From a spin-orbit calculation, pythtb's Wannier90 importer returns a spinless
model whose orbitals are the up and down states of each spinful orbital.
Declared with `spin=`, each pair is read as one spinful orbital
([A Wannier90 spinor import](../guide/spin.md#a-wannier90-spinor-import)). A
model in which an orbital carries one spin state and no orbital at its center
carries the other, as in a locally time-reversal-broken magnet, is not read with
its spin. pythTQC reads half-integer spin only as the two states of a spinor at
one center: a pythtb spinful orbital, or a pair declared with `spin=`. Handed
in without a pairing at one center, a model whose orbitals are spin states is
read in one of three ways:

- With no `spin=`, the model is read as spinless, in the single-valued tables,
  where time reversal squares to +1. Where no matrix of time reversal the
  Hamiltonian obeys squares to +1, time reversal is measured broken and the
  unitary group is returned. A Hamiltonian that mixes the spin states, such as
  Kane-Mele with its Rashba term, is such a case: its time reversal squares to
  −1. One that conserves the spin along `z` obeys a time reversal that squares
  to +1, and is returned a grey group.
- With `spinor=True`, which asks `realized_group` to read the double-valued
  tables, and no `spin=`, the call refuses with `rep_double_valued_spinless`.
- A pairing whose two members sit at different centers is refused with
  [`spin_pair_split`](../reference/refusals.md#spin_pair_split) before any group
  is identified. Such a pairing means either that each orbital carries one spin
  state with no partner at its center, the case above, or that the wrong
  orbitals were paired.

## A symmetry indicator reads every high-symmetry point

A symmetry indicator is read from the multiplicities of the little-group
representations at every high-symmetry point of the group's tables. Where
`hsp_table` refused at some high-symmetry point, `symmetry_indicator` of that
table refuses with `incomplete_content`. `decompose` with
`allow_incomplete=True` leaves the points refused out, for points where the
tables of the irreptables package cannot be read. `symmetry_indicator` takes no
such argument, since a class is read from the whole symmetry data vector
([Which symmetry-indicator class do these bands occupy?](../guide/indicators.md)).

A layer's `H(k)` has no `k_z`, and its points at `k_z = 1/2` are completed
instead. A little-group representation there that occurs in every elementary
band representation as often as one at a point in the plane takes that one's
multiplicity. The others are measured on the layer's decoupled stack. Where that
measurement stops at one of its checks, the call refuses with
`indicator_not_evaluable`, whose `context["stack"]` names the check.

## Centers noisier than the symmetry finder

The cell's operations are found from the orbital centers by spglib, the symmetry
finder, to its tolerance `symprec`. Centers noisier than that are a less
symmetric set of points to the finder. With no `precision=` declared, the call
returns that set's group, as low as `P1`. Where the centers still miss that
group's operations by enough to move `H(k)` beyond the tolerance its covariance
is held to, the call refuses with `positions_not_exactly_symmetric` instead.
Kane-Mele, `kane_mele(0.0, -1.0, 0.3, 0.25)`, with orbital 0 moved by
`(5e-5, 0)` in reduced coordinates, is returned `P21'`. With `precision=1e-3`
declared, it is returned `P6mm1'`
([How do I measure a numerical or Wannier90 model?](../guide/numerical.md)).

## A precision too coarse for the cell's geometry

A declared precision also sets the tolerance the finder reads positions at:
`10 * precision * L` where `precision * L` exceeds `symprec`, and `symprec`
otherwise, with `L` the length of the longest lattice vector along the periodic
directions
([How do I measure a numerical or Wannier90 model?](../guide/numerical.md)).
Where the Hamiltonian obeys fewer operations than the cell allows, the points
handed to the finder include, besides the orbital centers, the orbit of a point
in general position under the operations that hold, whose symmetry is the group
of those operations. Those points lie closer together than the centers. Where
the finder's tolerance is comparable to or larger than their distance, the
finder cannot tell them apart, and the call raises `InternalError`, whose
message names the finder's tolerance. Kane-Mele,
`kane_mele(1.0, -1.0, 0.3, 0.25)` with its two sites declared different atoms,
is returned `P3m11'` up to `precision=0.015` and raises from `0.02`. There the
finder reads at 0.2 lattice constants, and the centers are 0.58 apart. The
orbit the message lists, of the last of the three points the call tries, has
its points 0.09 apart. A fitted model's precision lies far below this: the
silicon model of
[Silicon from Wannier90: the group the model has](../start/tutorial-wannier90.md)
declares `precision=1e-4`.

## Two equally near origins

The names of Wyckoff positions, and of the little-group representations at some
k-points, depend on the point the tables are read about. They are read about the
point, among those the tables' origin can be placed at, nearest the model's own
origin
([The origin the names are read about](conventions.md#the-origin-the-names-are-read-about)).
A model whose origin is equally near two or more such points has no nearest one.
Its names are read about the one the irrep package chooses on the cell, where
that is one of them, and otherwise about the first of them on a fixed grid. The
choice is the same in every run. Another of them would give the same bands other
names, and nothing in the model picks one.

The example is a checkerboard of two species on a square cell, its orbitals at
`(1/2, 0)` and `(0, 1/2)`. Its group `P4mm1'` can be read about either orbital,
and the fourfold axes nearest the origin, `(±1/2, 0)` and `(0, ±1/2)`, are all
equally near. Moved by `(−0.1, 0)`, the same crystal has orbital 0 nearest its
origin:

```python
def board(centers):
    m = TBModel(Lattice([[1.0, 0.0], [0.0, 1.0]], centers, periodic_dirs=[0, 1]))
    m.set_onsite([0.5, -0.5])
    for R in ([0, 0], [1, 0], [0, -1], [1, -1]):
        m.set_hop(-1.0, 0, 1, R)
    return m

for centers in ([[0.5, 0.0], [0.0, 0.5]], [[0.4, 0.0], [-0.1, 0.5]]):
    rb = tqc.realized_group(board(centers), typat=[0, 1])
    print([o.wyckoff for o in rb.wyckoff_orbits], tqc.hsp_table(rb, bands=1).symmetry_data)
#  ['1b', '1a'] {'GM1': 1, 'M1': 1, 'X1': 1}
#  ['1a', '1b'] {'GM1': 1, 'M3': 1, 'X3': 1}
```

- `['1b', '1a']`: the Wyckoff positions of orbitals 0 and 1, as written. The
  rule reads the names about orbital 1.
- `['1a', '1b']`: moved, the names are read about orbital 0, the one nearest
  the origin.
- `{'GM1': 1, 'M1': 1, 'X1': 1}` and `{'GM1': 1, 'M3': 1, 'X3': 1}`: the
  irreps of the lowest band as written and as moved, the same band under other
  names. `GM` is the tables' name for Γ.

## A block as the value of a parameter

pythtb's `hamiltonian` takes a two-by-two block as the value of a parameter of a
spinful model. Every call that takes `params=`, `realized_group`, `solve_at`,
`spinless_presentation` and `stack_presentation`, takes a number for each
parameter and raises `ValueError` for a block. A hopping that acts on spin is
written as a function of its parameter that returns the block
([A model with free parameters](../guide/group.md#a-model-with-free-parameters)).

## Wilson loops and spectra resolved by an operator

pythTQC reads the little-group representations of a set of bands at isolated
high-symmetry points
([What a decomposition cannot certify](../guide/atomic-limits.md#what-a-decomposition-cannot-certify)).
It computes no Wilson loop, and no eigenspaces of an operator `O` projected onto
a set of bands, `P O P` with `P` their projector, such as the spin-resolved case
`O = n·σ`, with `n` a unit vector and `σ` the Pauli matrices. `spin_operator`
returns `n·σ` on each spin pair, the operator a spin-resolved calculation
projects onto the occupied bands.
