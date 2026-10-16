# How pythtqc reads a Hamiltonian

This page states once the ideas every chapter of the guide leans on: what
`pythtqc` measures on a Hamiltonian, what a declaration states, and what is
chosen where neither settles it. The chapters link here.
The blocks use pythtb's graphene, `graphene(delta, t)`, whose on-site offset
`delta` tells the two sublattices apart.

## The group is measured, not read off the positions

A set of positions is an upper bound. The lattice and the orbitals' positions
say what a Hamiltonian on that cell could be symmetric under, and a bond
distortion, a spin–orbit term, a magnetic order or a substrate can leave every
atom where it is and still remove operations: Fu–Kane–Mele's diamond with one
bond detuned keeps the positions of `Fd-3m` and obeys only `R-3m`
([a descent](../guide/group.md#when-the-hamiltonian-has-less-symmetry-than-its-cell)).
So the group is measured from `H(k)`, and every
[corep](tqc-primer.md#little-groups-irreps-and-coreps) label,
[band representation](tqc-primer.md#band-representations-and-atomic-limits),
[symmetry indicator](tqc-primer.md#how-a-symmetry-indicator-is-computed) and
[stable real-space invariant](tqc-primer.md#real-space-invariants-and-the-stable-ones)
read off the model belongs to a group the model was measured to have.

### The cell, and a verdict on each operation

The **cell** is the point set the group is found on: the orbitals' positions,
each labelled by its species so that sites declared different atoms are told
apart, and any species a declaration adds, such as a substrate under a layer or
a species added with `decorations=`. Its space group, found by a symmetry
finder to the tolerance `symprec`, a length in the model's own units, is what
the geometry allows. It is always asked with time reversal adjoined, as a grey
group, since positions say nothing about time reversal.

Each operation `g = {R|t}` of the cell is then put to the Hamiltonian. The
measurement looks for every matrix `X` with `X H(k) = H(Rk) X` at once on a grid
of `k`, and `X H(k)* = H(−Rk) X` for `g` combined with time reversal, which
sends `k` to `−Rk`, assuming nothing about what `g` does to the orbitals. The
equations are linear in `X`, so their solutions form a space. A space of
dimension zero is a proof that `g` is broken: no matrix with that action on `k`
carries `H(k)` onto itself at all. A staggered on-site energy breaks graphene's
twofold axis this way:

```python
import pythtqc as tqc
from pythtb.models import graphene

even, offset = graphene(0.0, -1.0), graphene(0.5, -1.0)   # graphene(delta, t)
print(tqc.realized_group(even).descent)                  # nothing broken
#  None
broken = tqc.realized_group(offset).descent["dropped"]
print({b["name"]: b["refuted_by"] for b in broken}["C2(z)"])
#  no symmetry matrix
```

A space that is not empty must still hold a matrix of the form an operation
has, a permutation of the sites times each site's own matrix and, with spin,
the operation's own spin rotation; and that matrix `U` must pass the
covariance check, `max_k ‖U H(k) U† − H(Rk)‖` at most `tol`, an energy in the
units of `H`, `1e-8` unless `tol=` or `precision=` declares another
([the glossary](../reference/glossary.md#tolerances)). Each operation that fails
is recorded with what refuted it
([the records reference](../reference/records.md)).

### The descent, and the tie rule

Where operations are broken, the cell is lowered to a group of the unitary
operations measured to hold and identified again, still with time reversal
adjoined, which is then measured on that group (below). The descent repeats until
every operation left holds, and only the group it ends at is returned, its
Wyckoff letters and site symmetries its own, in the tables' setting. The cell's
group, an upper bound, is never returned unmeasured: `descend=False` asks for
it, and refuses `cell_group_not_realized` where the Hamiltonian breaks any of it. Every
tight-binding Hamiltonian is symmetric under some space group, `P1` at the
least, so a descent always ends at one.

The group returned is the largest subgroup the measurement allows, and several
of that order can qualify with nothing measured between them: where the
operations that hold do not close into a group, where a declaration leaves
several readings, and where spinor states read without their spin carry an
ordinary representation on several subgroups. The one returned is the first by
one rule:

1. on a layer whose group holds operations that turn it over, the most
   operations that keep its side, since those hold in every environment and are
   what the layer keeps on a substrate;
2. then, and everywhere else first, the most rotations, so that a subgroup
   keeping rotations stands before one keeping mirrors or rotoinversions;
3. then the highest-order rotation;
4. then that rotation's axis in the conventional cell, `[001]` before `[100]`
   before `[010]`, then the other directions `[uvw]`: fewest nonzero indices
   first, then the smallest sum of their sizes, then the larger `w`, `u` and `v`
   in turn, so that `[111]` leads the body diagonals; where only the identity is
   a rotation, the axis of the highest-order improper operation;
5. then every other operation's axis, the same way, and last the operations'
   indices, so that the rule always picks one.

So `mmm` keeps the `2/m` along `c`, `6/mmm` keeps `6/m` before either `-3m`, and
`m-3m` keeps the `-3m` along `[111]`; and a free-standing layer whose declared
content leaves the threefold axis with either its in-plane twofolds or its
vertical mirrors keeps the mirrors, which the same layer keeps on a substrate.
The descent's `note` names the subgroups that tie, the one kept and why.

### Time reversal, and conserved operators

Time reversal is measured like any other operation, once the unitary
operations have settled, on the grey group they form. Obeyed, the group
returned is grey, and its irreducible objects are coreps. Refuted, the same
cell is identified again without time reversal, so every label afterwards is
read in the ordinary space group's own tables. The group returned is of type I
or type II: operations that combine
a unitary operation with time reversal, where each alone is broken, are not
part of it, and Haldane's model shows what that leaves out
([time reversal is measured](../guide/group.md#time-reversal-is-measured)).

At the identity, the measurement's solutions are every matrix that commutes with
`H(k)` at every `k`: the model's conserved operators, enumerated with nothing
assumed. Each one beyond the identity widens every operation's solution space,
and the record counts them as `additional commuting matrices`; the labels
returned are still the crystalline group's, the extra symmetry lying outside it
([a conserved operator](../guide/group.md#a-conserved-operator)).

## A declaration defines the crystal

Some of what defines a crystal is not in `H(k)`: which sites are the same atom,
what each orbital is, what lies under a layer, which orbitals are the two spin
components of one Wannier function. Declarations say it: `typat=`, the type of
each atom; `characters=`, what each orbital is; `decorations=`, species that
state a crystal field the positions do not show; `environment=`, what lies under
a layer; and `spin=`, the spin partners of a spinor import. A declaration is a
statement about the crystal. It is never measured and never checked against the
Hamiltonian, and the Hamiltonian is measured against the crystal the
declarations define.

On a substrate, graphene's hoppings obey all of `P6mm1'`. Declared two
different atoms, the crystal of boron nitride, it is returned that crystal's
group with nothing broken; a staggered on-site energy reaches the same group by
measurement, as a descent:

```python
print(repr(tqc.realized_group(even, typat=[0, 1])))
#  RealizedGroup(P3m11' #156.50; on a substrate (the default), the free-standing group #187.210 agrees)
print(repr(tqc.realized_group(offset)))
#  RealizedGroup(P3m11' #156.50; descended from #183.186 (12 of 24 coset representatives of the lattice translations broken); on a substrate (the default), the free-standing group #187.210 agrees)
```

A declaration the hoppings are not covariant under states a crystal with less
symmetry, and the group descends to it, each operation it broke recorded with
`refuted_by = "declaration"`
([a declaration that lowers the group](../guide/orbitals.md#a-declaration-that-lowers-the-group)).
A declaration
that makes two sites of one orbit different lowers the cell before any hopping
is read; that is a statement about the crystal and not a descent, and
`.decoration["declared"]` records it. A site the declaration leaves out carries
what the crystal's operations carry onto it, so a declaration made at some
sites of an orbit and not at the others keeps the largest group whose every
operation carries the declared content, the first by the tie rule where several
of that order tie. A declaration states each site's content,
the representation its orbitals carry together, and not how the orbitals are
partitioned inside the site, which is a basis
([the forms a declaration takes](../guide/orbitals.md#the-forms-a-declaration-takes)).

## What the hoppings fix

A pythtb orbital carries no symmetry label. The hoppings fix how the orbitals
transform relative to one another, not what they are: multiplying every
operation's matrix on a coupled part of the crystal, the orbitals the hoppings
connect directly or through others, by a one-dimensional character of the group
changes nothing the Hamiltonian shows. One such character is left open on each
coupled part. A declaration anywhere in the part fixes it, and every orbital of
the part is read relative to the declared one, so declaring one sublattice of an
orbit is the same statement as declaring the other. A choice of these
characters, one on each part, is a **labelling** of the representation, and the
labels of the bands rest on it. Wannier functions are not
atoms: how the orbitals are turned inside their site is a basis, and a site's
content is named from characters, which are the same in every basis. A model
whose spin sits in its orbital index, as a wannier90 spinor import's does, is
read without spin unless `spin=` says which orbitals are partners.

### The lowest-angular-momentum rule

What the declarations leave open is assigned once, by the lowest-angular-momentum
rule, the lowest-`ℓ` rule: the content of the reference orbital's site, orbital
0, which the rule reads first, is put at the lowest angular
momentum whose real harmonics hold all of it, `s` before `p` before `d`, and
every orbital the hoppings relate to it carries what they fix relative to it; an
orbital nothing relates is put at its own lowest `ℓ`. A function word such as
`s` or `pz` is shorthand for the representation it carries; only characters and
irreps are real. Graphene with nothing declared reads `s`-like, and declared
`A2` its whole coupled part is relabelled at once:

```python
print(tqc.realized_group(even).rep.characters.summary())
#  orbital  Wyckoff  site symmetry  function  irrep  named by  determined by  orbital characters
#        0       2b             3m               A1    tables       lowest-l  E=+1, C3(z)=+1, m(x)=+1
#        1       2b             3m               A1    tables       lowest-l  E=+1, C3(z)=+1, m(x)=+1
#    lowest-l: the hoppings fix how the orbitals transform relative to one another, not what they are; where several site irreps fit, the one carried by the lowest-angular-momentum functions is shown, and `characters=` states another.
print(tqc.realized_group(even, characters={0: "A2"}).rep.characters.summary())
#  orbital  Wyckoff  site symmetry  function  irrep  named by  determined by  orbital characters
#        0       2b             3m               A2    tables    declaration  E=+1, C3(z)=+1, m(x)=-1
#        1       2b             3m               A2    tables    hamiltonian  E=+1, C3(z)=+1, m(x)=-1
```

The rule is deterministic, and it is a choice: the hoppings cannot make it, so
nothing measured checks it, and the `determined by` column says how each site
irrep is known, `declaration` where it was stated, `hamiltonian` where every
labelling left open gives it one site irrep, and `lowest-l` where the rule
chose it. Where several labellings tie under the rule, its keys are read in
order:

1. the site content at the lowest angular momentum: at the reference orbital's
   site, then over the whole crystal, then on the reference orbital's own row;
2. the functions the orbitals read as: the point-group representation that the
   traces of the blocks carrying a site's orbitals onto their images average to,
   at the lowest angular momentum, so that a turn common to every site never
   moves the choice, while one site's orbitals turned against another's can;
3. the band labels: the labelling whose labels at the high-symmetry points come
   first in the tables' order;
4. last, the group's own order of operations, which decides only where every
   band label is the same.

A site's content is placed whole, at the lowest `ℓ` whose shell of real
harmonics holds all of it. So a site whose orbitals are one higher shell and
nothing below it is assigned that shell: five `d` orbitals at a `-3m` site read
`A1g + 2Eg`, the `d` shell, though a `p` shell and a second `p` pair,
`A2u + 2Eu`, would sit lower orbital by orbital. At a site with inversion, a
content that mixes parities, an `s` beside a `p`, lies in no single shell, since
the real harmonics of one `ℓ` all have the parity `(−1)^ℓ`; there the sum of
each orbital's own lowest `ℓ` decides.

Where a Hamiltonian conserves a pseudospin -- identical copies of a part of the
crystal that nothing joins, two Lieb lattices on the same sites -- every band
there is several-fold degenerate, the copies are one multiplet at each site,
and what they carry together is left open too: it moves the site content
itself. The same rule assigns it, reading the copies copy by copy first: two
identical `s` orbitals that no hopping joins hold `2A1`, each copy `A1`, rather
than the doublet `E` they could carry together, which a declaration states
instead. A content declared on the copies is read on every representation of
them the group's own table admits, and one that none of them carries lowers
the group as any declaration does. With a spin the copies carry an ordinary
representation, the double-valued factor system being the spin's own: two
spinful `s` orbitals declared `E` at a `4mm` site hold `-E1 + -E2`.
`rep.characters.multiplicity_spaces` names each such orbit with every site
content the group's own table admits, and the atomic-limit table's summary
prints a line for it.

Uncoupled orbitals a symmetry exchanges leave more open than one character as
well. Stack two `(p_x, p_y)` sites along `z`, no hopping joining a `p_x` to a
`p_y`, the second site's bonds the first's turned by 90°: the `4_2` screw carries
the first site's `p_x` onto the second's `p_y`, the group is `P4_2/mmc1'`, and
each site's `mmm` sees `p_x` and `p_y` as `B3u` and `B2u`, which no
one-dimensional character of the group puts there. Every representation the
hoppings leave on such orbitals is weighed, a declaration states which one
they carry, and the rule otherwise reads the site whole: `B3u + B2u`, the `p`
shell, rather than two `Ag`, which only a `d` shell holds together.

States that carry a spin are read the same way. Every operation's matrix is its
own spin rotation times a matrix on the orbitals, so what the hoppings leave
open is left on the orbitals, and the rule reads the orbitals' content: the
same chain with a spin on each orbital reads `-Eu` in every row, `B3u` and
`B2u` each with spin ½, whether or not a coupling `λ L_z S_z` joins each site's
spin to its orbitals.

Two lattices joined on site leave as much open. Two identical Lieb lattices
coupled at every site are a bonding and an antibonding lattice that nothing
joins and no operation exchanges; the two orbitals at a site make one multiplet
holding a share of each, and each lattice carries a character of its own. The
rule reads the two one by one, each `A1` undeclared rather than `A1 + B1`
together; the doublet `E` declared at the corner, which the coupling there
breaks, is carried as the `B2 + B1` it restricts to in the `mm2` that remains.

## A layer is silent about what lies under it

A model with two periodic directions is a layer. It is presented as a
three-dimensional slab cell, its missing direction filled with vacuum and
reduced `k` unchanged, and its group is a space group of that slab, which the
tables and the indicators are written for: Kane–Mele without its Rashba term
comes back as `P6mm1'` on a substrate and as `P6/mmm1'` free-standing
([a 2D model](../guide/layers.md#the-call)).

A flat set of orbitals is symmetric under the horizontal mirror whatever the
hoppings do, and in-plane `k` cannot see the mirror: an operation composed with
it acts on in-plane `k` exactly as the operation alone. So a layer has an
environment its model says nothing about, and the two possibilities are
different crystals:

- **free-standing**, where the full layer group applies and the Hamiltonian
  selects among its operations;
- **on a substrate**, where one side is different, every operation with
  `R[2,2] = −1` is absent by construction, and the group is the subgroup of
  the free-standing one that preserves the layer's normal.

The substrate is the default for a layer: its group is always well defined,
where a free-standing lift need not be unique (and where it is not, the tie rule
picks it), and it is the environment of every wallpaper-group model in the
literature. A slab with real out-of-plane coordinates is not silent, and its
default is free-standing. Whichever environment is asked, the other is measured
too, and the substrate group must be exactly the part of the free-standing one
that preserves the layer's normal; a mismatch is the package disagreeing with itself and raises
`InternalError` ([a 2D model](../guide/layers.md)).

The horizontal mirror is a question only here. A three-dimensional crystal has
a `k_z`, so the mirror gets a verdict like any operation; a two-dimensional model
read as a wallpaper group has no horizontal mirror at all; only a layer read in
three dimensions, free-standing, puts two operations in one action on `k`. The
representation tells them apart: with spin, by how each turns the spin; without
spin, by the mirror's sign on the orbitals, which the hoppings do not measure on
a flat layer: it is the parity the declared content states, or the even one the
lowest-angular-momentum rule assigns ([what the hoppings fix](#what-the-hoppings-fix)).
Inversion, `r → −r` squaring to `E` and never to `Ē`, exists
for a flat layer only as the layer-group operation `m(z)·C2(z)`, present where
the mirror holds; `C2(z)` alone is a rotation, whatever it does to in-plane `k`.

### A layer's out-of-plane rows

A layer's group is three-dimensional, and the
[EBR matrix an indicator is read off](tqc-primer.md#how-a-symmetry-indicator-is-computed),
a row per corep at each high-symmetry point, has rows at `k_z = ½`, points the
model has no coordinate for. Where such a row duplicates one the plane
contains, the bands' multiplicity on it follows by arithmetic, and the
indicator's record names each row it filled. Where it does not, the row is
measured on a crystal that has it: the layer repeated along its normal with no
hopping between copies, each orbital at its own height, whose bands do not
depend on `k_z`, so that the stack carries at `k_z = ½` the bands it carries
at `k_z = 0`. That decoupled stack obeys exactly what the layer obeys: an
operation of the stack, combined with a translation along the stacking,
carries one copy onto itself and is an operation of the layer, and every
operation of the layer is one of the stack's. So `symmetry_indicator` builds
the stack and reads it in the layer's own
group, each operation carried to the stack by its Cartesian action and
represented by the layer's own matrix, without measuring either again. It
measures every identity that ties the two: the stack's Hamiltonian is the
layer's at every `k_z`, up to the phases of its orbitals' heights, the layer's
matrices carry the stack's Hamiltonian onto itself, and the two presentations'
band characters agree where they share k-points and operations. The physics
guarantees each, so one that fails raises `InternalError`, the package
disagreeing with itself. On a substrate the
group is the part that preserves the layer's normal, and the free-standing
layer's group is the one
`realized_group` measured beside it, named and not used. A layer's class is the
class of its decoupled stack,
every component an invariant of that stack, fixed by the layer's own content
([a layer's class](../guide/indicators.md#a-layers-class-is-a-three-dimensional-class)).
A decomposition is read in the plane: its sums are the layer's own band
representations, those of the Wyckoff positions within the layer, and on those
every row at `k_z = ½` repeats one in the plane. A free-standing layer's group
also has positions off the layer, where none of its orbitals sits, and their
band representations are left out.

## What a verdict certifies

The verdicts on topology are proofs in one direction only. A decomposition
whose verdict is `none`, with the certificate `proof`, means no atomic limit
carries the bands' symmetry content; a nonzero symmetry indicator means the
same, and adding the bands of an atomic limit cannot change it; and where a set
of bands is split into valence and conduction bands, the verdict
`diagnosed topological` from the stable real-space invariants means the two
cannot both be atomic limits. The converses certify nothing. A
[symmetry vector](tqc-primer.md#how-a-symmetry-indicator-is-computed) that
decomposes into elementary band representations, and a zero indicator, leave
[fragile topology](tqc-primer.md#fragile-topology) and every obstruction the
symmetry content cannot see; a consistent pair of valence and conduction
realizations, atomic limits with each side's symmetry data, means only that any
topology there is invisible to the stable real-space invariants
([what a decomposition cannot certify](../guide/atomic-limits.md#what-a-decomposition-cannot-certify),
[what a zero does not certify](../guide/indicators.md#what-a-zero-does-not-certify),
[the no-go criterion](../guide/srsi.md#the-no-go-criterion)).

## When the group question is refused

Every tight-binding Hamiltonian is symmetric under some space group, `P1` at the
least, so the question which group it has is never refused, with two exceptions,
each about how the model is written.

- **Centres off the cell's own operations**, `positions_not_exactly_symmetric`.
  Orbital centres that miss the operations by `δ`, in reduced coordinates, cost
  `H(k)` a covariance residual of order `2π·δ·max|H|` whatever the hoppings do.
  Where that alone is above the covariance check's tolerance `tol` and no
  precision is declared, the call refuses before anything is built, naming the
  cell's group as an upper bound; `precision=` is the fix
  ([the centres first](../guide/numerical.md#the-centres-first-because-moving-them-is-a-gauge)).
  A declared precision is believed, and it bounds each centre: centres within
  it are moved onto the positions the group puts them at, about the origin that
  fits them, and a shift of all of them together within it is removed with the
  move, so that the group is named on centres exact in the model's own frame.
  Centres further off are a real displacement: an operation that
  would move one beyond the declaration is broken by the centres, and the group
  descends to the largest group of its operations that carries them within it,
  in which the bands are labelled. Centres noisier than the symmetry finder's
  own tolerance `symprec` are a less symmetric point set to it, and the group
  returned is that set's, as low as `P1`
  ([centres noisier than symprec](../guide/numerical.md#centres-noisier-than-symprec)).
- **A supercell of its own primitive cell**, `supercell`. No point set can say
  whether a cell that repeats under a fractional translation is a supercell,
  since only the hoppings tell the copies apart, so the translation is read as
  every operation is: it carries each site's orbitals onto its translate's,
  the permutation of the sites itself where the copies are written alike and
  any change of basis of each site's orbitals where they are written in turned
  frames, a spin carried unturned, and it holds where some such matrix `D`
  gives `D H(k) D† = H(k)`.
  Translations commute, and so must the matrices they are obeyed with: a flux
  that makes them anticommute leaves magnetic translations, which no crystal
  has. A Hamiltonian that obeys a pure translation of its cell is written on a
  supercell: the bands at one `k` of that cell are the crystal's at several `k`
  of its own zone, a combination no table names, so every label would name the
  wrong k-points. The refusal names the translation, the index, the primitive
  lattice and how far each site's orbitals turn onto its translate's, at any
  index, and the model is rebuilt on its primitive cell. A supercell whose
  Hamiltonian tells the copies apart is a primitive model of the bigger cell,
  and is descended to
  ([when the cell is a supercell](../guide/group.md#when-the-cell-is-a-supercell)).
  Folding a supercell onto its primitive cell may come later
  ([what the package cannot do yet](limits.md#a-supercell-is-refused-not-folded)).

The package is built for models periodic in two or three directions. A model
periodic in fewer, a chain such as pythtb's `ssh` or a molecule, is refused
`fewer_than_two_periodic_directions` before any group is found, whatever the
dimension of the cell it is written in: a limit of the package
([fewer than two periodic directions](limits.md#fewer-than-two-periodic-directions)).

Every other refusal is a verdict on a question asked after the group, a
practical limit (a search or memory budget, or tables the installation lacks),
a grammar error in a declaration, or one of the package's checks on itself,
which means a case it does not handle and is worth reporting. Once the group is
identified a refusal opens by naming it and its number and stating that every
operation of it is obeyed, and carries it in `context["group"]`; one reached
while the group is still being measured names the group the cell has at that
point, an upper bound. A disagreement inside the package raises
`InternalError`
([when a call refuses](../guide/refusals.md),
[every refusal](../reference/refusals.md)).
