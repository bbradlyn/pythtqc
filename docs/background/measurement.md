# How pythTQC reads a Hamiltonian

This page says how pythTQC measures the space group of a Hamiltonian, what a
declaration states about the crystal, and what is chosen where neither the
Hamiltonian nor a declaration settles it. It also says why a layer's
Hamiltonian does not say what lies under the layer, and what a decomposition,
an indicator and the no-go criterion each certify. The guide pages link here
for the full account. The examples use pythtb's graphene, `graphene(delta, t)`,
whose on-site offset `delta` tells the two sublattices apart.

## The group is measured, not read off the positions

The lattice and the orbitals' positions say what a Hamiltonian on that cell
could be symmetric under, and no more. A bond distortion, a spin-orbit term, a
magnetic order or a substrate can leave every atom where it is and still
remove operations. Fu-Kane-Mele's diamond with one bond detuned keeps the
positions of `Fd-3m` and obeys only `R-3m`
([When the Hamiltonian has less symmetry than its cell](../guide/group.md#when-the-hamiltonian-has-less-symmetry-than-its-cell)).
So the group is measured from `H(k)`. Every label of a
[little-group representation](tqc-primer.md#little-groups-irreps-and-coreps),
[band representation](tqc-primer.md#band-representations-and-atomic-limits),
[symmetry indicator](tqc-primer.md#how-a-symmetry-indicator-is-computed) and
[stable real-space invariant](tqc-primer.md#real-space-invariants-and-the-stable-ones)
read off the model belongs to the group the model was measured to have.

### The cell, and a verdict on each operation

The **cell** is the point set the group is found on. It holds the orbitals'
positions, each tagged with its species so that sites declared different atoms
are told apart. It also holds any atoms the package or a declaration adds: the
substrate under a layer, and the species `decorations=` adds. Its space group
is found by spglib to the tolerance `symprec`, a Cartesian length in the units
the pythtb Lattice vectors are given in. It is what the geometry allows. The
cell's group is always found with time reversal adjoined, as a grey group,
since positions say nothing about time reversal. It is then written in the
conventional cell used by the irreptables package, the tables' setting, by the
irrep package's search for that setting. That search fits the spin basis the
tables' SU(2) matrices are written in, and it is run from fixed seeds, so that
a cell is identified the same way in every run. Where no search matches the
cell, the group is still returned, and every call that reads the tables for it
refuses `tables_not_ready`.

Each operation `g = {R|t}` of the cell is then put to the Hamiltonian. The
measurement looks for every matrix `X` with `X H(k) = H(Rk) X` at once on a grid
of `k`. For `g` combined with time reversal, which sends `k` to `−Rk`, the
equation is `X H(k)* = H(−Rk) X`. Nothing is assumed about what `g` does to the
orbitals. The equations are linear in `X`, so their solutions form a space. The
equations depend on `g` only through its action on `k` and whether it carries
time reversal. So two operations with the same action on `k`, both with or both
without time reversal, share one space. That happens on a free-standing layer,
for an operation and its product with the horizontal mirror, and in a cell that
repeats under a pure translation, for an operation and its product with that
translation. A space of dimension zero means that no matrix solves the
equations at the points the space is found at. How the operation is then
judged is said below. A staggered on-site energy breaks graphene's twofold axis
this way:

```python
import pythtqc as tqc
from pythtb.models import graphene

even, offset = graphene(0.0, -1.0), graphene(0.5, -1.0)   # graphene(delta, t)
print(tqc.realized_group(even).descent)                  # nothing broken
#  None
broken = tqc.realized_group(offset).descent.dropped
print({b.name: b.refuted_by for b in broken}["C2(z)"])
#  no symmetry matrix
```

- `None`: `realized_group(even).descent`, for the graphene without an offset.
  The descent records the operations of the cell the Hamiltonian breaks, and is
  `None` where it obeys every unitary operation its cell allows.
- `no symmetry matrix`: `refuted_by` of the twofold axis `C2(z)` in the graphene
  with an offset. No matrix solves the axis's equations.
  [Returned records](../reference/records.md) lists every value `refuted_by`
  takes, under the descent.

The grid is the one `realized_group` measures on: `nk` points along each
periodic direction, starting at Γ, 6 by default. The space is found at one `k`
and cut down at the others. At one `k` a solution carries each eigenvector of
`H(k)` onto eigenvectors of `H(Rk)` at the same energy, so the outer products
of such pairs span the solutions there. Two eigenvectors are paired when their
energies differ by at most `window` times `max|H|`, `window` a dimensionless
tolerance ([Tolerances](../reference/glossary.md#tolerances)) and `max|H|`
defined below. At each further point, the space found so far is restricted to
that point's equations, and the directions whose singular value there is at
most `cut` times the largest are kept, `cut` another dimensionless tolerance.
Further points cut the space down until it stops changing.

`max|H|` is the largest entry of `H(k) − μ`, `μ` the mean on-site energy, over
64 fixed random points of the zone. Where the largest hopping, or the largest
on-site energy measured from `μ`, comes out larger, `max|H|` is raised to it.
Every energy tolerance is read against `max|H|`. Where a precision is declared,
the record prints it as `measured max|H|`. A model shifted by a constant energy
has the same `max|H|`, and so the same tolerances.

The points are visited in bit-reversed order, so that consecutive points do not
lie on one line of the zone, where a model can have more symmetry than on the
whole grid. Before the cutting stops, a generic combination of what is left is
checked at every point of the grid, and the cutting resumes at the point where
it fails most. Where the space comes out empty and a wider window could pair
eigenvectors the narrower one left apart, the window is widened tenfold, up to
`0.3`, and the space is found again. Where the pairs a window admits need more
memory than the measurement's budget, it narrows to the nearest pairs. Where
`H(k)` and `H(Rk)` are the same multiple of the identity at every point of the
grid, as for a single band whose energies at `k` and `Rk` agree, every matrix
solves the equations, and the operation is judged by the covariance check
alone.

A space that is not empty must still hold a matrix of the form an operation
has: a permutation of the sites times each site's own matrix and, with spin,
the operation's own spin rotation. That matrix `U` must pass the covariance
check, `max_k ‖U H(k) U† − H(Rk)‖` at most `tol`. `tol` is an energy in the
units of `H`, `1e-8 × max|H|` unless a declared `precision=` derives another
([Tolerances](../reference/glossary.md#tolerances)). With the default
tolerances or a declared precision, two eigenvectors whose energies differ by
at most `tol` are paired whatever the window, so that levels the covariance
check cannot tell apart are always paired. The space is determined at the few
points it was cut down at, where the singular-value cut can exclude a matrix
that passes the covariance check. So where the space is empty, or none of its
matrices passes the check, matrices of the operation's form are fitted over
every point of the grid. The operation holds where a fitted matrix passes. It
is broken where none of them passes, or where the energies of `H(k)` and
`H(Rk)` at one point already differ by more than `tol` allows. Each operation
that fails is recorded with what refuted it
([Returned records](../reference/records.md)).

### The descent, and the tie rule

Where operations are broken, the cell is lowered to a group of the unitary
operations measured to hold, and identified again with time reversal adjoined.
The lower group is identified on the cell with one generic point added, and its
orbit under the operations that hold, as a species of its own. The point is
fixed by no operation but the identity, so the cell so decorated is symmetric
under exactly those operations; `descent.cell` holds the point and its orbit.
Time reversal is then measured on the lower group (below). Each round measures
the smaller group again, and the descent repeats until every operation left
holds. Only the group it ends at is returned, its Wyckoff letters and site
symmetries its own, in the tables' setting. The cell's group is never returned
unmeasured.

The group returned is the largest subgroup the measurement allows. Several of
that order can qualify with nothing measured between them:

- the operations that hold do not close into a group;
- each of several subgroups keeps the declared site irreps, and no larger group
  does;
- each of several groups keeps every orbital center within a declared
  precision;
- spinor states read without their spin carry an ordinary representation on
  each of several subgroups.

The one returned is the first by one rule:

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

So where the three `2/m` subgroups of `mmm` tie, the one along `c` is kept.
Where `6/m` and the two `-3m` of `6/mmm` tie, `6/m` is kept, for its sixfold
rotation. Where the four `-3m` of `m-3m` tie, the one along `[111]` is kept. A
free-standing layer whose declared site irreps leave the threefold axis with
either its in-plane twofolds or its vertical mirrors keeps the mirrors, which
the same layer keeps on a substrate. The descent's `note` names the subgroups
that tie, the one kept and why.

### Time reversal, and conserved operators

Time reversal is measured like any other operation, once the unitary
operations have settled, on the grey group they form. Where it is obeyed, the
group returned is grey, and its irreducible representations are coreps. Where
it is refuted, the same cell is identified again without time reversal, and
every label afterwards is read in the tables of the space group without it.
The group returned is of magnetic type I or II.

Time reversal is broken in three ways. First, no matrix solves its equations,
as under a Zeeman term. Second, its square can have the wrong sign. With `K`
complex conjugation, time reversal acts as `X K`, which squares to `X X*`.
Where that is a multiple of the identity, its sign is the Hamiltonian's own,
since rescaling `X` changes it only by a positive factor. The tables ask for
`+1` where they are single-valued and `-1` where they are double-valued. So a
spinless model that carries spinors in its orbital index, as a Wannier90 spinor
import does, measures `-1` and breaks time reversal in the single-valued
tables. `spin=` declares which orbitals are spin partners, and the
double-valued tables are then read
([How does pythTQC read the spin of my model?](../guide/spin.md)). Last, no
matrix of its solution space has its form: none acts as `iσ_y` on every
declared spin pair, or none carries each site's states onto that site's own.

Where time reversal is broken, an operation of the cell combined with it can
still hold, as the mirrors combined with time reversal do in Haldane's model
([Time reversal is measured](../guide/group.md#time-reversal-is-measured)).
Such an operation holds where its matrix and the matrices of the group
returned, up to a one-dimensional character of that group, make a
corepresentation of the magnetic group the two generate: Wigner's conditions.
One whose unitary part is in the group returned never holds, since with that
part it makes time reversal itself. The operations that hold are listed on
`beyond_grey`, with the magnetic type of the group they form with the group
returned. They are not part of the group returned, which stays of type I.

At the identity, the measurement's solutions are every matrix that commutes with
`H(k)` at every `k`: the model's conserved operators, enumerated with nothing
assumed. Each one beyond the identity widens every operation's solution space,
and the record counts them as `additional commuting matrices`. The labels
returned are still the crystalline group's, the extra symmetry lying outside it
([A conserved operator](../guide/group.md#a-conserved-operator)).

## A declaration defines the crystal

Some of what defines a crystal is not in `H(k)`: which sites are the same atom,
what each orbital is, what lies under a layer, which pairs of orbitals are the
up and the down state of one spinful orbital. Declarations say it. `typat=` is
the type of each atom. `characters=` is what each orbital is. `decorations=`
adds species that state a crystal field the positions do not show.
`environment=` says what lies under a layer. `spin=` names the spin partners of
a spinor import. A declaration is a statement about the crystal. It is never
measured and never checked against the Hamiltonian, and the Hamiltonian is
measured against the crystal the declarations define.

On a substrate, graphene's hoppings obey all of `P6mm1'`. With its two sites
declared different atoms, the crystal of boron nitride, graphene is returned
that crystal's group, `P3m11'`, with nothing broken. A staggered on-site energy
reaches the same group by measurement, as a descent:

```python
print(repr(tqc.realized_group(even, typat=[0, 1])))
#  RealizedGroup(P3m11' #156.50; on a substrate (the default), the free-standing group P-6m21' #187.210 agrees)
print(repr(tqc.realized_group(offset)))
#  RealizedGroup(P3m11' #156.50; descended from P6mm1' #183.186 (12 of 24 coset representatives of the lattice translations broken); on a substrate (the default), the free-standing group P-6m21' #187.210 agrees)
```

- `P3m11' #156.50`: the group the Hamiltonian obeys, its Hermann-Mauguin
  symbol and its number among the magnetic space groups in the
  Belov-Neronova-Smirnova (BNS) numbering. The trailing `1'` marks time
  reversal adjoined, a grey group.
- `on a substrate (the default)`: the layer's environment, which nothing
  declared.
- `the free-standing group P-6m21' #187.210 agrees`: the layer measured
  free-standing as well. Its operations that keep the layer's side are exactly
  the substrate group.
- `descended from P6mm1' #183.186 (12 of 24 coset representatives of the
  lattice translations broken)`: the cell's group, and how many of its
  operations, one for each coset of the lattice translations, the hoppings
  broke.

A declaration is read against the character table of the site group measured
at the site. That table is computed from the multiplication table of the
operations that fix the site, so it exists at every site of every model. Its
rows are named with the tables' names for the site irreps, where the tables
name them.

A declaration the hoppings are not covariant under states a crystal with less
symmetry, and the group descends to it, each operation it broke recorded with
`refuted_by = "declaration"`
([A declaration that lowers the group](../guide/orbitals.md#a-declaration-that-lowers-the-group)).
A declaration that makes two sites of one orbit different lowers the cell
before any hopping is read. That is a statement about the crystal and not a
descent, and `decoration.declared` records it. A site the declaration leaves
out holds whatever the crystal's operations carry onto it. So a declaration
made at some sites of an orbit and not at the others keeps the largest group
whose every operation carries each declared site onto a site declared with the
same site irreps or with nothing, and keeps the span of a declared function,
such as `px`, in itself. Where several groups of that order tie, it keeps the
first by the tie rule. A declaration states the site irreps each site holds,
and not how the orbitals divide them inside the site, which is a basis
([The forms a declaration takes](../guide/orbitals.md#the-forms-a-declaration-takes)).

## What the hoppings fix

A pythtb orbital carries no symmetry label. The hoppings fix how the orbitals
transform relative to one another, and not what they are. A **coupled part**
of the crystal is a set of orbitals the hoppings connect, directly or through
others. Multiplying every operation's matrix on a coupled part by a
one-dimensional character of the group changes nothing the Hamiltonian shows.
One such character is left open on each coupled part. A declaration anywhere in
the part fixes it, and every orbital of the part is read relative to the
declared one, so declaring one sublattice of an orbit is the same statement as
declaring the other. Where every character leaves the declared site irrep
unchanged, as for `E` at a `4mm` site, declaring the irrep does not fix the
character, and declared generator matrices do
([Which basis a doublet is](../guide/orbitals.md#which-basis-a-doublet-is)). A
choice of these characters, one on each part, is a **labeling** of the
representation, and the labels of the bands rest on it.

With time reversal the character left open is real, a sign on each operation.
Without time reversal every one-dimensional character of the group is a
labeling, the complex ones included. So a declaration that only a complex
character meets is carried, and the group is not lowered. Haldane's model,
`haldane(0.0, -1.0, 0.15)` from `pythtb.models`, declared
`characters={0: "1E"}`, keeps `P6 #168`, the group it has undeclared, and its
`2b` row reads `1E`, determined by the declaration.

Where several orbitals sit at one site, the Hamiltonian binds them into
**multiplets**, which carry the site irreps it holds: `site_multiplets` on a row
of the atomic-limit table, which `realized_group` returns as `wyckoff_orbits`.
They are the finest partition of the site's orbitals for which every operation
has a nonsingular solution of its equations in block form: block-diagonal on
the multiplets of each site it fixes, and carrying the multiplets of a site
onto those of its image. Two orbitals at one site that nothing couples, each
with hoppings of its own, are two multiplets. A square's `(p_x, p_y)` pair that
nothing couples is one multiplet, in every basis it is written in, since the
fourfold axis mixes the two orbitals. An operation that carries one site onto
another joins two of a site's multiplets into one only where the two carry
copies of one site irrep. Where two partitions are equally fine, which takes an
accidental degeneracy, the one that binds the lowest-indexed orbitals is
returned, and the table's `numerical.accidental` records it.

Multiplets stated with `site_multiplets=` are a statement about the crystal, as
a declaration is. Each stated multiplet carries a site irrep of its own, so an
operation that mixes two of them is no operation of the crystal stated. A
`(p_x, p_y)` pair at a site with a fourfold axis, stated as two multiplets of
one orbital each, breaks the fourfold axis, with
`refuted_by = "tensor structure"`. Multiplets measured from the Hamiltonian
break nothing.

Wannier functions are not atoms: how the orbitals are turned inside their site
is a basis, and a site's irreps are named from characters, which are the same
in every basis. A model whose spin sits in its orbital index, as a Wannier90
spinor import's does, is read without spin unless `spin=` says which orbitals
are partners.

### How the representation is built

Once the group is settled, the record `r` that `realized_group` returns holds
in `r.rep` one matrix `D(g)` for each operation `g`, on the model's states.
`D(g)` carries each multiplet onto its image, as `g` carries the sites. On a
multiplet of one orbital its block is the operation's spin rotation, or `1`
without spin. On a wider multiplet it is the block of the measured solutions,
in whatever basis the orbitals are written. What the measurement leaves open is
finite: a factor on each orbit of multiplets for each generator. Each candidate
is held to three checks. It carries `H(k)` onto `H(Rk)` within `tol`. On every
block it is a multiple of the operation's own matrix. It multiplies as the
group does, the signs of the spin-1/2 lifts included. A candidate that fails
only the last carries a projective representation of another class, as
spin-1/2 states read in single-valued tables do.

Each generator's matrix is put in a fixed phase, and every other matrix is a
product of the generators'. The phase sets the matrix's `n`-th power, `n` the
operation's order modulo lattice translations, to the sign the group's
multiplication table requires. Of the `n`-th roots of unity this leaves, those
are kept under which the matrix commutes with time reversal and its block on a
reference orbital is a real multiple of the operation's spin rotation. Of
those, the one taken gives the largest entry of that block the largest real
part (`pythtqc.symmetry.normalize_rep`). Where the phases so chosen do not
multiply as the group does, the multiplication table chooses among the roots
of unity the normalization leaves. The matrices returned are the
representation induced from the site irreps the atomic-limit table returns, in
the tables' convention, spin along `+z` in a right-handed frame.

### The lowest-angular-momentum rule

What the declarations leave open is assigned once, by the
lowest-angular-momentum rule. The rule reads first the site of the reference
orbital, orbital 0. Its site irreps are put at the lowest angular momentum
whose real harmonics hold all of them, `s` before `p` before `d`. Every orbital
the hoppings relate to it carries what they fix relative to it. An orbital
nothing relates is put at its own lowest angular momentum. A function name such
as `s` or `pz` stands for the site irrep that function carries. Graphene with
nothing declared reads `A1`, the site irrep of an `s` orbital. With orbital 0
declared `A2`, both orbitals of its coupled part read `A2`:

```python
print(tqc.realized_group(even).wyckoff_orbits.summary())
#  Wyckoff  site symmetry  orbitals  label  determined by  declared
#       2b           3m1'       0-1     A1       lowest-l
#    lowest-l: the hoppings fix how the orbitals transform relative to one another, not what they are; where several site irreps fit, the one carried by the lowest-angular-momentum functions is shown, and `characters=` states another.
print(tqc.realized_group(even, characters={0: "A2"}).wyckoff_orbits.summary())
#  Wyckoff  site symmetry  orbitals  label  determined by  declared
#       2b           3m1'       0-1     A2    declaration      A2/-
```

- `2b`, `3m1'`, `0-1`: the Wyckoff position of the orbit, its site-symmetry
  group with time reversal, and the orbitals on it.
- `A1`: the site irrep one site of the orbit holds.
- `lowest-l`: the site irrep was assigned by the lowest-angular-momentum rule.
- `A2/-`: what `characters=` declared, one entry per orbital, `-` for an
  orbital not declared.
- `lowest-l: the hoppings fix ...`: printed under the table where some row
  reads `lowest-l`.

The rule is a choice: the hoppings cannot make it, so nothing measured checks
it. The `determined by` column says how each site irrep is known. It reads
`declaration` where it was declared, `hamiltonian` where every labeling left
open gives it one site irrep, and `lowest-l` where the rule chose it.

It reads `declaration not carried` where no labeling the hoppings leave agrees
with a declaration and the group was not lowered by it. Declared site irreps
that no labeling gives lower the group, as above, so on the group returned some
labeling gives them. Declared generator matrices state a basis as well as a
site irrep. Where the hoppings couple the declared orbitals to one another,
matrices no labeling gives them lower the group as well. Where no hopping
couples them, a labeling gives them the matrices' site irrep, none gives them
the matrices, the group keeps its operations, and the row reads
`declaration not carried`. So does a declaration the measurement could not
read before the labeling, such as site irreps declared on part of what one
multiplet holds, where no labeling agrees with it together with the other
declarations. Declared matrices that are not carried choose the labeling by
their traces alone. The row's label is then what its orbitals carry under the
labeling returned, and a line under the table names what was declared.

Where several labelings tie under the rule, its keys are read in order. The
labeling is chosen on each coupled part on its own. The part's lead multiplet
is the reference orbital's where the part holds it, and its lowest-numbered
multiplet otherwise.

1. the angular momentum of the site irreps, the lower first: the lowest `l`
   whose real harmonics hold them all, and then the sum over them of
   multiplicity times dimension times each one's own lowest `l`. It is read at
   the lead multiplet's site, then summed over the part's sites, then on the
   lead multiplet alone. Before these, on identical copies, a labeling that
   gives every copy the same site irreps comes first, read on one copy, and
   orbits of sectors that hold shares of the same multiplets are read one by
   one ([More than one character left open](#more-than-one-character-left-open));
2. the functions the orbitals read as: the point-group representation that the
   traces of the blocks carrying a site's orbitals onto their images average
   to, over the sites of an orbit and over each class of the point group. It is
   compared by the same two numbers, the lead multiplet's orbit first and then
   the sum over the part's orbits, the lower first. So a turn common to every
   site never moves the choice, while one site's orbitals turned against
   another's can;
3. the labels of the bands: the labeling whose labels at the high-symmetry
   points come first in the tables' order;
4. last, the one-dimensional character nearest the trivial one: equal to 1 on
   the most operations, then, operation by operation in the group's order, with
   the largest real part and then the largest imaginary part. It decides only
   where every label of the bands is the same.

A site's irreps are placed together, at the lowest angular momentum `l` whose
shell of real harmonics holds all of them. So a site whose orbitals are one
higher shell and nothing below it is assigned that shell. Five `d` orbitals at a
`-3m` site read `A1g + 2Eg`, the `d` shell, though a `p` shell and a second `p`
pair, `A2u + 2Eu`, would sit lower orbital by orbital. At a site with
inversion, site irreps of mixed parity, an `s` beside a `p`, lie in no single
shell, since the real harmonics of one `l` all have the parity `(-1)^l`. There
the sum of each orbital's own lowest `l` decides.

### More than one character left open

What the hoppings leave open is read off the matrices that commute with `H(k)`
at every `k` and act within sites. They split the states into sectors. A sector
is `m` identical copies of a set of `d` states, and on it those matrices are
every `m × m` matrix on the copies, times the identity on each copy's states. A
crystal whose hoppings connect every orbital is one sector of one copy. Two
orbitals at one site that no hopping reaches, at different energies, are two
sectors. Two identical copies of a lattice that no hopping joins are one sector
of two copies, a pseudospin the Hamiltonian conserves. The group carries the
sectors in orbits. On an orbit of one sector of one copy, a coupled part, the
matrices are fixed up to any one-dimensional character of the group. On an
orbit of several sectors, uncoupled orbitals a symmetry exchanges, they are
fixed up to any representation induced from a character of one sector's
stabilizer. On an orbit of identical copies, they are fixed up to any
representation of the copies that the stabilizer admits. On these last two,
the last key of the rule takes the representation whose character per copy is
nearest the trivial one, in that key's order. The sectors are read at the
declared precision. A coupling below it leaves the pieces it joins apart, and
one above it joins them.

A pseudospin the Hamiltonian conserves is a set of identical copies of a part
of the crystal that nothing joins, such as two Lieb lattices on the same sites.
Every band there is several-fold degenerate. The copies are one multiplet at
each site, and what they carry together is left open too, which moves the site
irreps themselves. The same rule assigns them, reading the copies copy by copy
first. Two identical `s` orbitals that no hopping joins hold `2A1`, each copy
`A1`, rather than the doublet `E` they could carry together, which a
declaration states instead. Site irreps declared on the copies are read on
every representation of them the group's own table admits, and site irreps
that none of them carries lower the group as any declaration does. With a spin
the copies carry an ordinary representation, the double-valued factor system
being the spin's own: two spinful `s` orbitals declared `E` at a `4mm` site
hold `-E1 + -E2`. The row of each such orbit lists in `numerical.contents`
every set of site irreps the group's own table admits, and the atomic-limit
table's summary prints a line for it.

Uncoupled orbitals a symmetry exchanges leave more open than one character as
well. Stack two `(p_x, p_y)` sites along `z`, no hopping joining a `p_x` to a
`p_y`, the second site's bonds the first's turned by 90°. The `4_2` screw
carries the first site's `p_x` onto the second's `p_y`, and the group is
`P4_2/mmc1'`. Each site's `mmm` reads `p_x` as `B3u` and `p_y` as `B2u`. That
is not what the hoppings fix times a one-dimensional character of the group,
but a representation induced from a character of one site's `mmm`. Every
representation the hoppings leave on such orbitals is weighed, and a
declaration states which one they carry. The rule otherwise reads the site
whole: `B3u + B2u`, the `p` shell, rather than two `Ag`, which only a `d` shell
holds together.

States that carry a spin are read the same way. Every operation's matrix is its
own spin rotation times a matrix on the orbitals, so what the hoppings leave
open is left on the orbitals, and the rule reads the orbitals' site irreps. The
same chain with a spin on each orbital reads `-Eu` in every row, `B3u` and
`B2u` each with spin 1/2, whether or not a coupling `λ L_z S_z` joins each
site's spin to its orbitals.

Two lattices joined on site leave as much open. Two identical Lieb lattices
coupled at every site are a bonding and an antibonding lattice that nothing
joins and no operation exchanges. The two orbitals at a site make one multiplet
holding a share of each, and each lattice carries a character of its own. The
rule reads the two lattices one by one. Undeclared, each is `A1`, and the
corner holds `2A1` rather than `A1 + B1`. The doublet `E` declared at the
corner is no symmetry of the coupling there. It lowers the group to `Pmm21'`,
the first by the tie rule of the largest groups in which the bonding and the
antibonding state carry its restriction, `B1 + B2`.

Which matrices are returned on such orbitals is not measured either.
Conjugating the representation by a unitary on each sector's copies, a phase
where each sector holds one copy, changes no character, and the hoppings obey
every such conjugate equally. The matrices returned are written in the tables'
convention: on one multiplet of the orbit, they are the tables' own matrices
for its site irreps, laid along its orbitals, wherever such a conjugation
reaches them. Where none does, as for a pair turned inside its site, each
sector of one copy is put in the phase that brings it nearest the tables'
matrices.

## When an operation holds only within the tolerance

Where an operation holds within `tol` and not to rounding, `H` is projected
onto its symmetric part, averaged over the group with the matrices of a
representation of it. `H` is projected the same way where the hoppings join
two parts of the crystal, or tell identical copies apart, only below the
tolerance. The record prints `symmetrized`, with how far this moved `H`
([Then symmetrize](../guide/numerical.md#then-symmetrize)).

Two covariance residuals are read for each operation of the cell. One is that
of the best matrix the measurement finds for it. The other is that of its
matrix in the representation the projection is built with, whose spin part is
the operation's spin-1/2 matrix in the tables and whose blocks follow the site
multiplets, declared or measured. An operation is broken where either is above
`tol`. A kept operation is exact where the second is at rounding, and near
where the second is above rounding and at most `tol`
([Exact, near and broken](../guide/numerical.md#exact-near-and-broken)). The
matrix the measurement finds turns with the model's frame, and the
representation's matrix does not. So a spin frame turned inside a site by less
than the declared precision leaves the operation exact by the first residual
and near by the second. The projection then makes `H` symmetric under the
representation's matrices, with the spin unturned. A basis turned so that the
site multiplets mix is near in the same way. The projection removes the mixing
between the multiplets, and leaves the orientation inside each multiplet as
the model has it. On a model built from exact numbers, the first residual can
lie above rounding where the second does not, and such a model is returned as
it was handed in.

The representation the projection averages with is built as `r.rep` is, its
covariance check held to `tol` in place of rounding. Its matrices are assembled
from discrete choices of sign and phase, so where each multiplet of a site is a
single orbital they carry none of the model's noise. A multiplet of several
orbitals carries the model's own block for every operation, and the projection
averages with the exact representation nearest those blocks. The average,
written back as hoppings, leaves some of them at rounding. A hopping is
dropped only where it lies below `1e-14` of the largest and below a gap of
more than a factor of ten in the sizes of the hoppings. Hoppings the group
relates have the same size, so the gap never separates them. Where there is no
such gap, nothing is dropped, and the model returned can hold hoppings at
rounding.

## What a row of the atomic-limit table names

Each row of the atomic-limit table holds `band_representation`, the band
representation its orbitals induce, written in elementary band representations
(EBRs). Its symmetry data vector is computed from the trace of each
little-group operation over the orbitals' own states at each high-symmetry
point. It is decomposed at a Wyckoff position the orbitals can be moved onto
along the point, line or plane of their own position, their own position
first. Moving the orbitals there is a symmetric deformation, and by induction
in stages the band representation induced there is the one the orbit induces.
Where several sums at one position carry the symmetry data vector, the one
whose site irreps are the row's own is taken.

A site irrep at a maximal position whose band representation is not
elementary, such as `E` at `3c` of `P-43m`, is carried by no EBR at its own
position. It is read through its stable real-space invariants, where the
tables carry them. Otherwise the sum is sought at the positions on each line or
plane through the site, the most special first, and last it is the one sum the
symmetry data vector has, where it has only one. The other sums with the same
symmetry data vector and the same stable real-space invariants stand in the
row's `numerical.alternatives`. Where none of this settles the sum, the row's
terms are empty, and a line under the table says why.

A site irrep is named from the tables by induction. Each row of the site
group's character table induces a band representation. Its multiplicities at
the high-symmetry points are matched against the EBRs the tables list at the
site's Wyckoff position: at Γ first, then at the other points in the tables'
order while more than one EBR matches. With spin, a row is induced times the
spin-1/2 lift. That product can be reducible, and is then named by a sum: at a
`3m` site the `E` row is named `-1E2E + -E1`, and the `A1` and `A2` rows are
both `-E1`. A row the EBRs do not name is named by the tables of the stable
real-space invariants, through a table of names shipped with pythTQC and keyed
on a reference site of the position. An operation of the group carries each
site onto that reference site, so every site of an orbit reads one name. A row
neither source names takes the name the tables give that irrep of the site's
point group at the origin of a symmorphic space group, where every rotation
carrying that origin's operations onto the site's gives the same name.
Otherwise, without spin, it keeps its Mulliken symbol, where that symbol is one
a table prints and no other row takes it. A row still without a name is left
unnamed, as is every such row with spin. `numerical.named_by` says which of
these named a row.

## A layer is silent about what lies under it

A model with two periodic directions is a layer. It is presented as a
three-dimensional slab cell, its missing direction filled with vacuum and
reduced `k` unchanged, and its group is a space group of that slab, which the
tables and the indicators are written for. Kane-Mele without its Rashba term
is returned `P6mm1'` on a substrate and `P6/mmm1'` free-standing
([The call](../guide/layers.md#the-call)).

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

The substrate is put in the cell as one atom of a species of its own, a short
distance `decoration.dz` below every site of the layer. An operation that turns
the layer over carries each such atom above the layer, where there is none, so
it is absent. An atom under every site, rather than one under the origin,
keeps the operations that do not fix the origin, and a buckled layer keeps its
own geometry.

The substrate is the default for a layer written with two-dimensional
positions. On a substrate no two operations of the group returned share an
action on `k`. Free-standing, an operation and its product with the horizontal
mirror share one, so the free-standing group need not be unique, and where it
is not, the tie rule picks it. The substrate is also the environment of every
wallpaper-group model in the literature. A slab, a model with three-dimensional
positions and two periodic directions, has heights, and they decide which
operations that turn it over hold. So its default is free-standing.

Unless `cross_check=False`, the other environment is measured too, by an
independent identification and descent of the same Hamiltonian, with the same
declarations and the same tolerances. The two groups are not compared where
the other environment refuses, or where the declaration is written in site
irreps only the environment asked for has. Where they are compared, the
substrate group is exactly the part of the free-standing one that preserves
the layer's normal, or the two agree for another pair of the groups the tie
rule chose among. Where neither holds, the package disagrees with itself and
raises `InternalError`
([Is my 2D model on a substrate or free-standing?](../guide/layers.md)).

The horizontal mirror is a question only here. A three-dimensional crystal has
a `k_z`, so the mirror gets a verdict like any operation. Of the groups
returned, only a free-standing layer's, read in three dimensions, holds two
operations with one action on `k`. The representation tells them apart: with
spin, by how each turns the spin; without spin, by the mirror's sign on the
orbitals, which the hoppings do not measure on a flat layer. That sign is the
parity of the declared site irreps, or the even one the lowest-angular-momentum
rule assigns ([What the hoppings fix](#what-the-hoppings-fix)). Inversion,
`r → −r` squaring to the identity and never to the rotation by 2π, exists for
a flat layer only as the layer-group operation `m(z)·C2(z)`, present where the
mirror holds. `C2(z)` alone is a rotation, whatever it does to in-plane `k`.

### A layer's out-of-plane rows

A layer's group is three-dimensional. The
[EBR matrix an indicator is read off](tqc-primer.md#how-a-symmetry-indicator-is-computed),
a row per little-group representation at each high-symmetry point, has rows at
`k_z = 1/2`, points the model has no coordinate for. Where such a row
duplicates one the plane contains, the bands' multiplicity on it follows by
arithmetic, and the indicator's record names each row it filled. Where it does
not, the row is measured on a crystal that has it: the layer repeated along its
normal with no hopping between copies, each orbital at its own height. The
bands of that stack do not depend on `k_z`, so the stack carries at
`k_z = 1/2` the bands it carries at `k_z = 0`. The decoupled stack obeys
exactly what the layer obeys. An operation of the stack, combined with a
translation along the stacking, carries one copy onto itself and is an
operation of the layer, and every operation of the layer is one of the stack's.
So `symmetry_indicator` builds the stack and reads it in the layer's own group,
each operation carried to the stack by its Cartesian action and represented by
the layer's own matrix, without measuring either again. Every identity that
ties the stack to the layer is checked, and one that fails raises
`InternalError`. On a substrate the group is the part that preserves the
layer's normal, and the free-standing layer's group is the one
`realized_group` measured beside it, named and not used. A layer's
symmetry-indicator class is the class of its decoupled stack, every component
an invariant of that stack, fixed by the layer's own symmetry data vector
([A layer's class is a three-dimensional class](../guide/indicators.md#a-layers-class-is-a-three-dimensional-class)).

A decomposition is read in the plane. Its sums are the layer's own band
representations, those of the Wyckoff positions within the layer, and on those
every row at `k_z = 1/2` repeats one in the plane. A Wyckoff position is within
the layer where it is free to move along the normal, or where each of its
heights from the layer's middle plane lies between the lowest and the highest
of the layer's orbitals, within `symprec`. A free-standing layer's group also
has positions at a fixed height off the layer, half the cell's length along the
normal from the middle plane, halfway between copies, where none of its
orbitals sits. Their band representations are left out, and a decomposition's
`numerical.restriction.outside_layer` names them. On a substrate no operation
turns the layer over, every position is free to move along the normal, and
every band representation is kept.

## What a verdict certifies

A decomposition, an indicator and the no-go criterion can each prove a set of
bands topological, and none of them can prove it an atomic limit. A set of
bands whose symmetry data vector is no sum of elementary band representations
with nonnegative integer multiplicities is not an atomic limit. `decompose`
returns that with the certificate `proof`. Nor is a set of bands with a nonzero
symmetry indicator, and adding the bands of an atomic limit does not change the
indicator. Where a set of bands is split into valence and conduction bands,
the no-go criterion, `nsit` true from `splitting`, means the two cannot both be
atomic limits. A
[symmetry data vector](tqc-primer.md#how-a-symmetry-indicator-is-computed) with
a decomposition, or a zero indicator, leaves open
[fragile topology](tqc-primer.md#fragile-topology) and every obstruction the
symmetry data vector cannot see. Where an atomic limit with the valence bands'
symmetry data vector and one with the conduction bands' have stable real-space
invariants valued modulo n adding up to those of all the bands, the no-go
criterion does not fire. That means only that any topology there is invisible
to the stable real-space invariants
([What a decomposition cannot certify](../guide/atomic-limits.md#what-a-decomposition-cannot-certify),
[What a zero does not certify](../guide/indicators.md#what-a-zero-does-not-certify),
[The no-go criterion](../guide/srsi.md#the-no-go-criterion)).

## When the group question is refused

Every tight-binding Hamiltonian periodic in two or three directions is
symmetric under some space group, `P1` at the least. So the question which
group it has is refused in two cases only, each about how the model is
written.

- **Centers off the cell's own operations**, `positions_not_exactly_symmetric`.
  Orbital centers that miss the operations by `δ`, in reduced coordinates, cost
  `H(k)` a covariance residual of order `2π·δ·max|H|`, whatever the hoppings
  do. Where that alone is above `tol`, and neither `precision=` nor
  `position_tol=` is declared, the call refuses before anything is built. The
  reason names the cell's group as an upper bound. Centers that miss by less
  are refused the same way once the group is identified, the reason naming it,
  wherever the miss keeps `H` from being made or held exactly symmetric
  ([`positions_not_exactly_symmetric`](../reference/refusals.md#positions_not_exactly_symmetric)).
  `precision=` is the fix
  ([The centers first: moving them is a change of gauge](../guide/numerical.md#the-centers-first-moving-them-is-a-change-of-gauge)).
  A declared precision is believed, and no center is moved farther than it, in
  reduced coordinates. Centers within it are moved onto positions the group
  carries exactly, by the move of least Cartesian length that has no part
  common to every center. Where the operations' translations about those
  positions are not fractions of the model's own cell, a shift of all the
  centers together, within the precision, makes them so. The group is then
  named on centers exact in the model's own frame. Centers further off are a
  real displacement. An operation that would move one beyond the declaration
  is broken by the centers. The group descends to the largest subgroup whose
  operations keep every center within the declared precision, and the bands
  are labeled in it. Centers noisier than the symmetry finder's own tolerance
  `symprec` are a less symmetric point set to it, and the group returned is
  that set's, as low as `P1`
  ([Centers noisier than `symprec`](../guide/numerical.md#centers-noisier-than-symprec)).
- **A supercell of its own primitive cell**, `supercell`. No point set can say
  whether a cell that repeats under a fractional translation is a supercell,
  since only the hoppings tell the copies apart. So the translation is read as
  every operation is. It carries each site's orbitals onto its translate's: by
  the permutation of the sites itself where the copies are written alike, and
  by any change of basis of each site's orbitals where they are written in
  turned frames, a spin carried unturned. It holds where some such matrix `D`
  gives `D H(k) D† = H(k)`. Translations commute, and so must the matrices
  they are obeyed with: a flux that makes them anticommute leaves magnetic
  translations, which no space group's tables describe. A Hamiltonian that
  obeys a pure translation of its cell is written on a supercell. The bands at
  one `k` of that cell are the crystal's at several `k` of its own zone, a
  combination no table names, so every label would name the wrong k-points.
  The refusal names the translation, the index of the supercell, the primitive
  lattice, and how far each site's orbitals turn onto its translate's. It is
  returned whatever the index. A model rebuilt on its primitive cell is
  measured as usual. A supercell whose Hamiltonian tells the copies apart is a
  primitive model of the bigger cell, and is descended to
  ([When the cell is a supercell](../guide/group.md#when-the-cell-is-a-supercell)).
  A supercell is refused and is not folded onto its primitive cell
  ([A supercell is refused, not folded](limits.md#a-supercell-is-refused-not-folded)).

The package is built for models periodic in two or three directions. A model
periodic in fewer, a chain such as pythtb's `ssh` or a molecule, is refused
`fewer_than_two_periodic_directions` before any group is found, whatever the
dimension of the cell it is written in
([Fewer than two periodic directions](limits.md#fewer-than-two-periodic-directions)).

Every other refusal is one of four kinds:

- a verdict on a question asked after the group;
- a practical limit: a search or memory budget, or tables the installation
  lacks;
- a grammar error in a declaration or an argument;
- a check of the package on itself: an identity that holds in exact arithmetic
  measured not to, or two measurements of one thing that disagree. The checks
  that return a refusal are those
  [Refusals](../reference/refusals.md#the-packages-checks-on-itself) lists
  under that heading, and most mean the package has met a case it does not
  handle, worth reporting. Any other such check that fails raises
  `InternalError` and returns no refusal.

Once the group is identified, a refusal opens by naming it and its number and
stating that every operation of it is obeyed, and carries it in
`context["group"]`. One reached while the group is still being measured names
the group the cell has at that point, an upper bound
([What does a refusal mean, and what do I do about it?](../guide/refusals.md),
[Refusals](../reference/refusals.md)).
