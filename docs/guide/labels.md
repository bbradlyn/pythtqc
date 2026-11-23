# Which little-group representations do these bands carry?

## What the little-group representations of a set of bands are

The eigenvectors `u_n(k)` of `H(k)` that belong to a set of bands carry a
representation of the little group of `k`. Its matrix for an operation `g` is
the sewing matrix `B(g, k)_mn = ⟨u_m(k)| U(g; k) |u_n(k)⟩`, with `U(g; k)` the
operator of `g` on the model's states at `k`. For an operation combined with
time reversal, the ket is complex conjugated before `U(g; k)` acts. The traces
of `B` over a degenerate multiplet, its characters, fix which little-group
representations the multiplet carries: irreps without time reversal, and
co-representations (coreps) with it, where time reversal can join two irreps
into one corep. The multiplicities of the little-group representations of a
set of bands at every high-symmetry point form its symmetry data vector.

This package uses two facts about them. First, a decomposition into elementary
band representations (EBRs), a symmetry indicator and the integer-valued
stable real-space invariants (ZSRSIs) are fixed by the symmetry data vector. So
`decompose`, `symmetry_indicator` and `srsi_of_bands` each take the table
`hsp_table` returns. Second, the hoppings fix the matrices of the operations on
the orbitals they couple only up to one one-dimensional character of the group,
common to all of them. A choice of that character on each coupled part of the
crystal is a labeling, and the little-group representations of the bands rest
on it. So where a site irrep is assigned by the lowest-angular-momentum rule,
declaring it otherwise with `characters=` can change the little-group
representations the bands carry.

The theory of these representations is in Bradley and Cracknell, *The
Mathematical Theory of Symmetry in Solids* (Oxford, 1972). A label is the name
of a corep or an irrep in the tables of the Bilbao Crystallographic Server
([Elcoro *et al.*, J. Appl. Cryst. **50**, 1457 (2017)](https://doi.org/10.1107/S1600576717011712)),
which the irreptables package ships. At a point those tables leave out, such as
`KA` below, it is the name in the server's band-representation tables.
[Little groups, irreps and coreps](../background/tqc-primer.md#little-groups-irreps-and-coreps),
in [The topological quantum chemistry this package uses](../background/tqc-primer.md),
gives a longer account.

## The call

`coreps_at` takes what `realized_group` returned, `r`, a high-symmetry point
and `bands=`, the set of bands. A point is given by its name, such as `"GM"`,
or by its coordinates as `high_symmetry_points(r.group)` holds them. That
mapping holds the tables' points in the model's reduced coordinates.
`coreps_at` returns the little-group representations the set of bands carries
at that point, one row for each degenerate multiplet. `characters` takes the
same arguments and returns the characters they are read from.
Little-group representations can only be returned at the tabulated member of a
point's star. `coreps_at` at another member is refused with
`kpoint_not_tabulated`.

`sewing_matrix`, `characters`, `coreps_at` and `hsp_table` take
`degeneracy_tol=`, an energy tolerance for grouping the bands into degenerate
multiplets. The bands are grouped from the lowest up, and a band joins a
multiplet where it lies within `degeneracy_tol` of the multiplet's lowest band,
in the units of H. Left out, it is `1e-8` times `max|H|`, the largest entry of
`H(k)` once the mean on-site energy is subtracted.

`coreps_at` and `hsp_table` also take `label_tol=`, a dimensionless tolerance
for reading the multiplicities of the little-group representations off the
characters. A multiplicity within `label_tol` of a nonnegative integer is read
as that integer. It is `1e-3` by default.

The example is lin3g, from `pythtqc.models`, the spinful square layer of Lin
*et al.*, [Nat. Commun. **15**, 550 (2024)](https://doi.org/10.1038/s41467-024-44762-w),
with two orbitals at Wyckoff `1a`, two at `2c`, and eight bands.
`typat=[0, 0, 1, 1]` says the first two orbitals sit on one species of atom and
the last two on another. The example reads the six lowest bands at Γ: the EBR
of the `2c` orbitals and a fragile pair, two bands that are no atomic limit on
their own and become one once suitable atomic bands are added.
[Could these bands be an atomic limit?](atomic-limits.md) and
[Which symmetry-indicator class do these bands occupy?](indicators.md) read the
same six bands.

```python
import numpy as np
import pythtqc as tqc
from pythtb import WFArray
from pythtqc import models

r = tqc.realized_group(models.lin3g(), typat=[0, 0, 1, 1])
model, group = r.model, r.group
hsp = tqc.high_symmetry_points(group)              # the tables' k-points, in this cell
table = tqc.characters(r, hsp["GM"], bands=6)    # the six lowest bands at GM
print(table, [m.bands for m in table.multiplets])
#  BandCharacters(GM (0, 0), bands 0-5 in 3 multiplets) [(0, 1), (2, 3), (4, 5)]
columns = tuple(table.multiplets[0].characters)     # the unitary operations, by index
chars = np.array([[m.characters[i] for i in columns] for m in table.multiplets])
print(np.round(chars.real, 3) + 0.0)
#  [[ 2.     0.    -1.414 -1.414  0.     0.     0.     0.   ]
#   [ 2.     0.     1.414  1.414  0.     0.     0.     0.   ]
#   [ 2.     0.    -1.414 -1.414  0.     0.     0.     0.   ]]
content = tqc.coreps_at(r, hsp["GM"], bands=6)
print(content, [m.corep_names for m in content.multiplets])
#  CorepsAt(GM: 2x-GM6 + -GM7) [('-GM6',), ('-GM7',), ('-GM6',)]
print(round(content.isolation, 2))
#  13.02
```

## How to read it

- `BandCharacters(GM (0, 0), bands 0-5 in 3 multiplets)`: `table`, the
  characters of the six bands at Γ, at the reduced coordinates `(0, 0)`. `GM`
  is the tables' name for Γ.
- `[(0, 1), (2, 3), (4, 5)]`: the rows of the table, `table.multiplets`, each a
  degenerate multiplet of the set of bands, given by its bands. Here they are
  three doublets.
- `[[ 2. 0. -1.414 ...`: the characters, a row for each multiplet. The columns
  are the unitary operations of the little group at Γ, by index, in the order
  of `columns`. The identity's column is each multiplet's dimension. The first
  and third rows are equal, and differ from the second only at the two
  fourfold rotations, `−√2` against `√2`. The characters at Γ are real here, so
  the real parts are printed.
- `CorepsAt(GM: 2x-GM6 + -GM7)`: `content`, the coreps of the six bands at Γ,
  read off the characters against the corep table of the little group. The
  printout sums them: `-GM6` twice and `-GM7` once. A leading `-` marks a
  double-valued corep, carried by states with spin.
- `[('-GM6',), ('-GM7',), ('-GM6',)]`: the coreps of each multiplet,
  `corep_names` of each row of `content.multiplets`. The first and third
  doublets carry the same corep.
- `13.02`: `content.isolation`, the energy gap at Γ from the set of bands to
  the nearest band outside it, in the units of H.

The operations are those of `r.group`, the group measured from the
Hamiltonian. The example names the operations of lin3g's columns, and lists
the antiunitary operations of the little group at Γ, which `table.operations`
also holds:

```python
print(group.name, [tqc.operations(group)[i].name for i in columns])
#  P4mm1' ['E', 'C2(z)', 'C4(z)', 'C4^-1(z)', 'm(y)', 'm(x)', 'm([110])', 'm([1-10])']
print(tuple(i for i, op in table.operations.items() if op.antiunitary), np.allclose(chars.imag, 0))
#  (9, 10, 11, 12, 13, 14, 15, 16) True
```

- `P4mm1' ['E', 'C2(z)', ...]`: the group and the name of the operation of each
  column. `P4mm1'` is a grey group, `P4mm` with time reversal adjoined, as the
  trailing `1'` marks.
- `(9, 10, 11, 12, 13, 14, 15, 16)`: the antiunitary operations of the little
  group at Γ, by index. They are not columns, since a corep table has
  characters for the unitary operations alone.
- `True`: the characters have no imaginary part.

Each row of `content.multiplets` holds its bands, its energy and its coreps.
The example prints lin3g's lowest multiplet at Γ, and then every multiplet:

```python
print(content.multiplets[0])
#  Multiplet(GM, bands 0-1 at -16.35: -GM6)
for m in content.multiplets:
    print(m.bands, len(m.bands), m.corep_names, round(m.energy, 3))
#  (0, 1) 2 ('-GM6',) -16.35
#  (2, 3) 2 ('-GM7',) -12.375
#  (4, 5) 2 ('-GM6',) -4.325
```

- `Multiplet(GM, bands 0-1 at -16.35: -GM6)`: the lowest multiplet, at Γ: its
  bands, its mean energy in the units of H, and its corep.
- `(0, 1) 2 ('-GM6',) -16.35`: each multiplet's bands, its degeneracy, its
  coreps and its energy, lowest first.

`content.summary()` prints the same record part by part, and
`table.summary()` prints the characters with a line for each unitary operation
of the little group: its index, its Seitz symbol and its trace over each
multiplet.

## Every high-symmetry point at once

`hsp_table` takes `r` and `bands=`. It returns the little-group
representations of the set of bands at every high-symmetry point the model
samples, which for a layer are the points in its plane.

The example prints lin3g's points, asks for another member of `X`'s star,
`(1/2, 0)`, by its coordinates, and reads the six lowest bands at every point:

```python
print({name: k.tolist() for name, k in hsp.items()}, hsp.out_of_plane)
#  {'GM': [0.0, 0.0], 'M': [0.5, 0.5], 'X': [0.0, 0.5]} ('A', 'R', 'Z')
# another member of X's star, (1/2, 0), is not the tabulated one
print(tqc.coreps_at(r, [0.5, 0.0], bands=6).gate)
#  kpoint_not_tabulated
t = tqc.hsp_table(r, bands=6)
print(t.summary())
#  bands 0-5   (.bands)
#    their coreps at each high-symmetry point   (.symmetry_data)
#      GM: 2x-GM6 + -GM7
#      M: -M6 + 2x-M7
#      X: 3x-X5
print({name: [m.label for m in rows] for name, rows in t.multiplets.items()})
#  {'GM': ['-GM6', '-GM7', '-GM6'], 'M': ['-M6', '-M7', '-M7'], 'X': ['-X5', '-X5', '-X5']}
print(content.coreps, tqc.format_coreps(content.coreps))
#  {'-GM6': 2, '-GM7': 1} 2x-GM6 + -GM7
print(t.symmetry_data, t.complete)
#  {'-GM6': 2, '-GM7': 1, '-M6': 1, '-M7': 2, '-X5': 3} True
```

- `{'GM': [0.0, 0.0], 'M': [0.5, 0.5], 'X': [0.0, 0.5]}`: the tabulated points
  in the model's plane, by name, in its reduced coordinates.
- `('A', 'R', 'Z')`: `hsp.out_of_plane`, the tabulated points out of the
  layer's plane. A layer's bands are labeled in the three-dimensional space
  group of its slab
  ([Is my 2D model on a substrate or free-standing?](layers.md)). `A`, `R` and
  `Z` lie at `k_z = 1/2`.
  [How to read it](atomic-limits.md#how-to-read-it), on the page on atomic
  limits, and [How to read it](indicators.md#how-to-read-it), on the page on
  symmetry indicators, say how a decomposition and a symmetry indicator treat
  these points.
- `kpoint_not_tabulated`: `(1/2, 0)` is in the star of `X` and is not the
  tabulated member.
- `bands 0-5   (.bands)`: the set of bands. Each line of `t.summary()` ends with
  the field that holds it.
- `GM: 2x-GM6 + -GM7`: the coreps at each point, summed over the multiplets.
  Each point holds three doublets, and at `X` all three carry `-X5`.
- `{'GM': ['-GM6', '-GM7', '-GM6'], ...}`: `t.multiplets`, the rows point by
  point, lowest first, each with its `label`, the line `format_coreps` writes
  for its coreps.
- `{'-GM6': 2, '-GM7': 1} 2x-GM6 + -GM7`: `content.coreps`, the corep
  multiplicities at Γ, and the line `tqc.format_coreps` writes for them, as
  `summary()` prints it.
- `{'-GM6': 2, ..., '-X5': 3}`: `t.symmetry_data`, the symmetry data vector of
  the six bands.
- `True`: `t.complete`, whether every point was read without a refusal.
  `decompose` and `symmetry_indicator` refuse a table that is not complete.

`t.note`, empty for lin3g, is a sentence that `t.summary()` prints where it is
not empty. It is written where another labeling that keeps every orbital's
site irrep labels some multiplet of the set of bands with other little-group
representations. Every such labeling keeps which bands are degenerate, and the
symmetry data vector of all the model's bands taken together. The note is also
written where the Hamiltonian conserves a pseudospin, identical copies of a
part of the crystal that no hopping joins, and the set of bands has weight on
the copies. The hoppings then leave the site irreps of the copies open, and
where none is declared the lowest-angular-momentum rule chose them among
several the group admits.

Without time reversal, `k` and `−k` are inequivalent, and tabulated under two
names, wherever no operation of the group carries one onto the other. The
example is Haldane's honeycomb model, from `pythtb.models`, with a sublattice
offset. Its group is `P3`, and its `K` and `KA` are such a pair. The example
reads its lowest band:

```python
from pythtb.models import haldane

hal = tqc.realized_group(haldane(0.3, -1.0, 0.15, 0.5))
print(hal.group.name, list(tqc.high_symmetry_points(hal.group)))
#  P3 ['GM', 'K', 'KA', 'M']
print(tqc.hsp_table(hal, bands=1).summary())
#  band 0   (.bands)
#    their irreps at each high-symmetry point   (.symmetry_data)
#      GM: GM1
#      K: K2
#      KA: KA3
#      M: M1
```

- `P3 ['GM', 'K', 'KA', 'M']`: the group, and its points, `KA` at `−k` for `K`
  at `k`.
- `K: K2` and `KA: KA3`: the irreps at the pair. The tables of the irreptables
  package list the irreps at `K` alone. Those at `KA` are their complex
  conjugates, named as the band-representation tables of the Bilbao
  Crystallographic Server name them: `KA3` is the conjugate of `K3`.
- `M: M1`: the little group at `M` holds the identity alone, and a set of bands
  carries its one irrep, `M1`, once for each band.

## Label an isolated set of bands

`bands` is the set of bands to label. It can be specified either as an integer
n for the lowest n bands, or a sequence of band indices. `characters`,
`coreps_at` and `hsp_table` require it. Little-group representations can only
be returned if the set of bands contains all the states in degenerate
multiplets at each point it is read at. A set that cuts a multiplet is refused
with `multiplet_split`, which names the multiplet and its energies.

The example reads lin3g's four lowest bands at Γ, and then five, the fifth
being half of a doublet:

```python
print(tqc.coreps_at(r, hsp["GM"], bands=4))
#  CorepsAt(GM: -GM6 + -GM7)
cut = tqc.coreps_at(r, hsp["GM"], bands=5)
print(bool(cut), cut.gate, cut.context["multiplet"])
#  False multiplet_split (4, 5)
```

- `CorepsAt(GM: -GM6 + -GM7)`: the four lowest bands, two whole doublets.
- `False multiplet_split (4, 5)`: the refusal is falsy, and
  `context["multiplet"]` holds the bands of the multiplet cut.

## One diagonalization, reused

`sewing_matrix`, `characters`, `coreps_at` and `hsp_table` take the states to
read as `states=`. With `states=None`, the default, or `states=r.model`, the
model is solved at `k`. A solved `pythtb.WFArray` is read at `k` as it stands.
`sewing_matrix`, `characters` and `coreps_at` also take the `(evals, evecs)`
pair `tqc.solve_at` returns at one point, so that several sets of bands are
read off one diagonalization. Little-group representations can only be
returned if the states diagonalize `H(k)` at the point asked for. States that
do not are refused with `states_not_at_k`. For lin3g the states of an
equivalent `k + G` are refused too, since its `2c` orbitals sit off the cell's
origin and `H(k + G)` differs from `H(k)`.

The example solves lin3g once at Γ, reads three sets of bands off those
states, and hands the same states in at `M`:

```python
gm = tqc.solve_at(model, hsp["GM"])                # energies and eigenvectors, once
print(tqc.multiplets(gm[0]))
#  ((0, 1), (2, 3), (4, 5), (6, 7))
for bands in (2, [4, 5], [6, 7]):
    print(bands, tqc.coreps_at(r, hsp["GM"], states=gm, bands=bands))
#  2 CorepsAt(GM: -GM6)
#  [4, 5] CorepsAt(GM: -GM6)
#  [6, 7] CorepsAt(GM: -GM7)
print(tqc.coreps_at(r, hsp["M"], states=gm, bands=2).gate)
#  states_not_at_k
```

- `((0, 1), (2, 3), (4, 5), (6, 7))`: `tqc.multiplets`, the degenerate
  multiplets of the spectrum at Γ, each given by its bands, lowest first.
- `[4, 5] CorepsAt(GM: -GM6)`: each set of bands, read off the same states.
- `states_not_at_k`: the states of Γ, handed in at `M`.

A `WFArray` holds states on a mesh. Little-group representations can only be
returned at a point the mesh contains. Any other point is refused with
`k_not_on_mesh`. `k_grid(group, n)` builds a mesh through Γ, with `n`
divisions along each periodic direction, and says which high-symmetry points
it contains. `n` is one number for every direction, or one for each. A mesh
the group does not carry onto itself, such as `k_grid(group, (4, 5))` for
lin3g's `P4mm1'`, is refused with `asymmetric_grid`.

The example builds lin3g's 4x4 and 5x5 meshes, solves the model on the 5x5
one, and reads the six lowest bands on it:

```python
print(tqc.k_grid(group, 4), tqc.k_grid(group, 5), sep="\n")
#  KGrid(4x4x1 mesh, symmetric; samples GM, M, X)
#  KGrid(5x5x1 mesh, symmetric; samples GM; misses M, X)
wf = WFArray(model.lattice, tqc.k_grid(group, 5).mesh, spinful=True)
wf.solve_model(model)
print(tqc.hsp_table(r, states=wf, bands=6).summary())
#  bands 0-5   (.bands)
#    their coreps at each high-symmetry point   (.symmetry_data)
#      GM: 2x-GM6 + -GM7
#    refused at M and X (k_not_on_mesh), so the symmetry data vector is incomplete   (.refusals, .complete)
```

- `KGrid(4x4x1 mesh, symmetric; samples GM, M, X)`: the even mesh holds the
  zone boundary, and so `M` and `X`.
- `samples GM; misses M, X`: the odd mesh holds Γ alone of the tabulated
  points.
- `refused at M and X (k_not_on_mesh)`: the points the mesh misses, each
  refused, as `t.refusals` holds them. The table is not complete.

`multiplets` takes `degeneracy_tol=` too, in the units of H. It is handed
energies alone, so its default is `1e-8`, not scaled by `max|H|`. A model
whose degenerate multiplets carry noise, as a fitted one's can, takes a
tolerance above the noise and below the smallest gap between multiplets. The
example groups lin3g's spectrum at `M`, where two doublets lie `0.271` apart,
at the default and at `0.3`:

```python
em = tqc.solve_at(model, hsp["M"])[0]
print(tqc.multiplets(em), tqc.multiplets(em, degeneracy_tol=0.3))
#  ((0, 1), (2, 3), (4, 5), (6, 7)) ((0, 1, 2, 3), (4, 5), (6, 7))
```

- `((0, 1, 2, 3), (4, 5), (6, 7))`: at `0.3` the two lowest doublets read as
  one multiplet.

## One operation's sewing matrix

`sewing_matrix` takes `r`, the index of an operation `g`, a point `k` and
`bands=`. It returns the sewing matrix of `g` on the set of bands at `k`. A
sewing matrix can only be returned if `g` leaves `k` invariant up to a
reciprocal-lattice vector. Any other `k` is refused with `little_group`.

`r.operations` lists the operations of the group, one for each coset of the
lattice translations. Each operation has an index, a Seitz symbol and a name.
The Seitz symbol `{R|t}` is written in the conventional cell used by the
irreptables package. The name is written in the Cartesian axes in which the
pythtb Lattice is specified. The example prints lin3g's:

```python
print(r.operations)
#  P4mm1' #99.164: one operation per coset of the lattice translations; the double group also holds each one's negative
#  index  Seitz            name
#      1  {1|0,0,0}        E
#      2  {2_001|0,0,0}    C2(z)
#      3  {4+_001|0,0,0}   C4(z)
#      4  {4-_001|0,0,0}   C4^-1(z)
#      5  {m_010|0,0,0}    m(y)
#      6  {m_100|0,0,0}    m(x)
#      7  {m_110|0,0,0}    m([110])
#      8  {m_1-10|0,0,0}   m([1-10])
#      9  {1|0,0,0}'       E'
#     10  {2_001|0,0,0}'   C2(z)'
#     11  {4+_001|0,0,0}'  C4(z)'
#     12  {4-_001|0,0,0}'  C4^-1(z)'
#     13  {m_010|0,0,0}'   m(y)'
#     14  {m_100|0,0,0}'   m(x)'
#     15  {m_110|0,0,0}'   m([110])'
#     16  {m_1-10|0,0,0}'  m([1-10])'
```

- `P4mm1' #99.164`: the group's name and number. The double group also holds
  each operation's negative, the operation combined with a rotation by 2π.
- `3  {4+_001|0,0,0}   C4(z)`: an operation's index, its Seitz symbol and its
  name.
- `11  {4+_001|0,0,0}'  C4(z)'`: a trailing `'` marks an operation combined
  with time reversal.

The table holds only the operations the Hamiltonian obeys. `r.descent` names
each operation of the cell it breaks, and why
([When the Hamiltonian has less symmetry than its cell](group.md#when-the-hamiltonian-has-less-symmetry-than-its-cell)).

`operation_index` returns the index of the operation a Seitz symbol names. It
accepts the axis written in either direction, and the translation shifted by
any lattice vector
([Operations: a Seitz symbol and a Cartesian name](../background/conventions.md#operations-a-seitz-symbol-and-a-cartesian-name)).
An operation can also be found by its matrix: each operation of `r.operations`
holds its rotation in the Cartesian axes of the Lattice, `rotation_cartesian`.
The example finds the quarter turn about `z` both ways, and reads its sewing
matrix on the lowest pair at `M`:

```python
c4 = tqc.operation_index(r.group, "{4+_001|0,0,0}")
quarter = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
print(c4, [(op.index, op.name, op.translation.tolist(), op.antiunitary)
           for op in r.operations.values() if np.allclose(op.rotation_cartesian, quarter)])
#  3 [(3, 'C4(z)', [0.0, 0.0, 0.0], False), (11, "C4(z)'", [0.0, 0.0, 0.0], True)]
B = tqc.sewing_matrix(r, c4, hsp["M"], bands=2)
print(np.round(np.trace(B.matrix), 6))
#  (-1.414214+0j)
print(np.round(r.operations[c4].spin_lift, 6) + 0)
#  [[0.707107-0.707107j 0.      +0.j      ]
#   [0.      +0.j       0.707107+0.707107j]]
```

- `3`: the index `operation_index` returns.
- `(3, 'C4(z)', [0.0, 0.0, 0.0], False), (11, "C4(z)'", ...)`: the two
  operations with that rotation, `C4(z)` and `C4(z)'`. `antiunitary` tells
  apart the second, combined with time reversal. `translation` is the
  translation that comes with the rotation, in the model's reduced
  coordinates: zero in this symmorphic group, the fractional part of a screw or
  a glide in others.
- `(-1.414214+0j)`: the trace of `B` at `M`, `−√2`, the pair's character of
  the fourfold rotation, one of those from which its corep `-M6` is read. The
  fourfold leaves `M` invariant up to a reciprocal-lattice vector.
- `[[0.707107-0.707107j ...`: `spin_lift`, the operation's SU(2) matrix on the
  model's spin, `exp(−iθ n·σ/2)` for a rotation by θ about the axis `n`, with σ
  the Pauli matrices, `n` in the Cartesian axes of the Lattice and spin along
  their `z`. For the quarter turn about `z` it is `diag(exp(−iπ/4), exp(iπ/4))`.

The fourfold carries `X` to another member of its star, `(1/2, 0)`, so it has
no sewing matrix at `X`, and the call is refused with
[`little_group`](../reference/refusals.md#little_group).

## When a label is not trustworthy

**The conditioning at the point.** A number read off eigenvectors, such as a
sewing matrix or a character, is known only as well as the multiplets at that
point are separated. `coreps_at(...).numerical.thresholds.effective` is the
threshold below which such a number is treated as zero at that point. It is
the larger of `1e-14` and the eigenvectors' noise floor, how far a computed
eigenvector can turn toward the states of other multiplets, which grows as
`Δ`, the smallest gap between distinct multiplets, closes.
`coreps_at(...).numerical.summary()` prints how the noise floor is computed.
The example prints the threshold at each of lin3g's points, beside `Δ` in the
units of H:

```python
thresholds = {name: tqc.coreps_at(r, name, bands=6).numerical.thresholds for name in t.kpoints}
print({name: (f"{thr.effective:.1e}", round(thr.separation, 3))
       for name, thr in thresholds.items()})
#  {'GM': ('1.0e-14', 3.975), 'M': ('8.3e-14', 0.271), 'X': ('1.8e-14', 1.33)}
```

- `'M': ('8.3e-14', 0.271)`: the threshold at `M`, `thr.effective`, and `Δ`
  there, `thr.separation`. It is largest at `M`, where the multiplets lie
  closest.

The threshold does not enter the labels: each multiplicity read off the
characters is read as an integer within `label_tol`. Loosening
`degeneracy_tol` lowers the threshold only by reading distinct multiplets as
one.

**What the labels rest on.** The little-group representations of the bands
rest on the labeling
([What the hoppings fix](../background/measurement.md#what-the-hoppings-fix)).
No site irreps were declared for lin3g (no `characters=`). The hoppings fix
the site irrep of its `2c` orbitals, and those of its two orbitals at `1a`
were assigned by the lowest-angular-momentum rule: each the site irrep of the
lowest angular momentum the hoppings allow. Declaring orbital 0 a `dx2-y2`
orbital, the irrep `B1` of its site-symmetry group `4mm`, which with spin is
`-E2`, describes another crystal with the same hoppings. The example prints the
orbitals' site irreps, declares orbital 0, and reads the six lowest bands at Γ
again:

```python
print([(o.wyckoff, o.label, o.determined_by) for o in r.wyckoff_orbits])
#  [('1a', '-E1', 'lowest-l'), ('1a', '-E2', 'lowest-l'), ('2c', '-E', 'hamiltonian')]
other = tqc.realized_group(models.lin3g(), typat=[0, 0, 1, 1], characters={0: "dx2-y2"})
print(tuple(o.label for o in other.wyckoff_orbits))
#  ('-E2', '-E1', '-E')
relabeled = tqc.coreps_at(other, hsp["GM"], bands=6)
print(relabeled, [m.label for m in relabeled.multiplets])
#  CorepsAt(GM: -GM6 + 2x-GM7) ['-GM7', '-GM6', '-GM7']
```

- `('1a', '-E1', 'lowest-l')`: each orbital's Wyckoff position, its site irrep,
  and how the site irrep is known: `lowest-l`, assigned by the
  lowest-angular-momentum rule; `hamiltonian`, fixed by the Hamiltonian.
- `('-E2', '-E1', '-E')`: under the declaration, orbital 0 carries `-E2`, and
  the hoppings give orbital 1 the `-E1` that orbital 0 carried.
- `CorepsAt(GM: -GM6 + 2x-GM7)`: every doublet at Γ changes its corep, `-GM6`
  to `-GM7` and `-GM7` to `-GM6`. The first and third doublets still carry the
  same corep.

Where an orbital reads `lowest-l`, the labels rest on that rule. Declaring an
orbital's site irrep with `characters=` puts the declaration in place of the
rule's choice ([How do I say what my orbitals are?](orbitals.md)). `t.note`
says where the labels rest on one of the two choices described under
[Every high-symmetry point at once](#every-high-symmetry-point-at-once).

## When it refuses

A refusal is a returned record. It is falsy, and names the check that stopped
the call and what it measured
([What does a refusal mean, and what do I do about it?](refusals.md)).
`hsp_table` collects each point's refusal in `t.refusals`, and `t.complete` is
then `False`. The checks these calls meet:

- [`multiplet_split`](../reference/refusals.md#multiplet_split): the set of
  bands cuts a degenerate multiplet.
- [`states_not_at_k`](../reference/refusals.md#states_not_at_k): the states
  handed in do not diagonalize `H(k)` at the point.
- [`k_not_on_mesh`](../reference/refusals.md#k_not_on_mesh): the mesh of a
  `WFArray` handed in does not contain the point.
- [`asymmetric_grid`](../reference/refusals.md#asymmetric_grid): `k_grid` was
  asked for a mesh the group does not carry onto itself.
- [`kpoint_not_tabulated`](../reference/refusals.md#kpoint_not_tabulated): the
  point is not a tabulated point in this cell, or not the tabulated member of
  its star.
- [`out_of_plane`](../reference/refusals.md#out_of_plane): the point has a
  component out of a layer's plane, or names a tabulated point out of it, as
  `A`, `R` and `Z` do for lin3g.
- [`little_group`](../reference/refusals.md#little_group): from
  `sewing_matrix`, an operation that does not leave the point invariant.
- [`integer_multiplicity`](../reference/refusals.md#integer_multiplicity),
  [`multiplet_dimension`](../reference/refusals.md#multiplet_dimension): a
  multiplicity read off the characters lies further than `label_tol` from a
  nonnegative integer, or the little-group representations read have
  dimensions that do not add up to the multiplet's degeneracy. Most often a
  multiplet was grouped wrongly.
- [`little_group_mismatch`](../reference/refusals.md#little_group_mismatch),
  [`tables_not_ready`](../reference/refusals.md#tables_not_ready): the
  operations measured at the point do not match the little group of the
  tables, or the group's cell could not be matched to the setting the tables
  are written in.

## Where to go next

- [Returned records](../reference/records.md) lists every field of
  [`BandCharacters` and `CorepsAt`](../reference/records.md#bandcharacters-and-corepsat),
  of [`HSPTable`](../reference/records.md#hsptable) and of
  [`Operation`](../reference/records.md#operationtable-and-operation).
- [Functions](../reference/functions.md) gives the arguments of
  [`high_symmetry_points`](../reference/functions.md#high_symmetry_points),
  [`characters`](../reference/functions.md#characters),
  [`coreps_at`](../reference/functions.md#coreps_at),
  [`hsp_table`](../reference/functions.md#hsp_table),
  [`solve_at`](../reference/functions.md#solve_at),
  [`multiplets`](../reference/functions.md#multiplets),
  [`k_grid`](../reference/functions.md#k_grid),
  [`format_coreps`](../reference/functions.md#format_coreps),
  [`operations`](../reference/functions.md#operations),
  [`operation_index`](../reference/functions.md#operation_index) and
  [`sewing_matrix`](../reference/functions.md#sewing_matrix).
- [Coreps at a k-point](../background/conventions.md#coreps-at-a-k-point), in
  [Conventions: operations, labels and band representations](../background/conventions.md),
  says how the labels are spelled.
- [Could these bands be an atomic limit?](atomic-limits.md),
  [Which symmetry-indicator class do these bands occupy?](indicators.md) and
  [What do the stable real-space invariants say about these bands?](srsi.md)
  go on from the table.
