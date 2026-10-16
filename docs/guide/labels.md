# Which coreps do these bands carry?

This chapter labels a set of bands: which
[coreps](../background/tqc-primer.md#little-groups-irreps-and-coreps) its
states carry at each high-symmetry point, read off the model's own
eigenvectors, characters first and labels second; which set of bands may be
labelled; how to label several from one diagonalization; how to find one
operation and read its sewing matrix; and when a label is not to be trusted.
The model is lin3g, from `pythtqc.models`: the spinful
square layer of Lin *et al.*, [*Nat. Commun.* **15**, 550 (2024)](https://doi.org/10.1038/s41467-024-44762-w),
with four orbitals, two at Wyckoff `1a` and two at `2c`, and eight bands. Its
six lowest bands are the
[elementary band representation](../background/tqc-primer.md#band-representations-and-atomic-limits) of the `2c`
orbitals and a fragile pair: two bands that are no
[atomic limit](atomic-limits.md) on their own and become one once suitable
atomic bands are added. `typat=[0, 0, 1, 1]` says the first two orbitals sit on
one species of atom and the last two on another. The chapters on
[atomic limits](atomic-limits.md) and [symmetry indicators](indicators.md)
continue on the same model.

## The call

```python
import numpy as np
import pythtqc as tqc
from pythtb import WFArray
from pythtqc import models

r = tqc.realized_group(models.lin3g(), typat=[0, 0, 1, 1])
model, group, assignment, validated, rep = r
hsp = tqc.high_symmetry_points(group)              # the tables' k-points, in this cell
table = tqc.characters(rep, hsp["GM"], bands=6)    # the six lowest bands at Gamma
print(table, table.multiplets)
#  CharacterTable(k=[0.0, 0.0], bands 0-5 in 3 multiplets, group=realized) ((0, 1), (2, 3), (4, 5))
print(np.round(table.characters.real, 3) + 0.0)
#  [[ 2.     0.    -1.414 -1.414  0.     0.     0.     0.   ]
#   [ 2.     0.     1.414  1.414  0.     0.     0.     0.   ]
#   [ 2.     0.    -1.414 -1.414  0.     0.     0.     0.   ]]
content = tqc.corep_content(rep, hsp["GM"], bands=6)
print(content, content.labels)
#  CorepContent(GM: 2×-GM6 + -GM7, partition=(0, 1, 0)) ({'-GM6': 1}, {'-GM7': 1}, {'-GM6': 1})
print(round(content.isolation, 2))
#  13.02
```

## How to read it

`table` is the **character table** of the set of bands at Γ (`GM`), the one
object every label is read from. Its rows are the degenerate multiplets of the
six bands, three doublets, grouped at the energy tolerance `degeneracy_tol`
(`1e-8` in the model's energy units, by default). Its columns are the unitary
operations of the little group at Γ, those that leave Γ invariant up to a
reciprocal-lattice vector, and `group=realized` says they belong to the group
measured from the Hamiltonian. An entry is the trace, over one multiplet, of the
**sewing matrix** `B(g, k)_mn = ⟨u_m(k)| U(g; k) |u_n(k)⟩`, the matrix of the
operation `g` in the basis of the states of the set:

```python
print(group.name, [tqc.operations(group)[i].name for i in table.columns])
#  P4mm1' ['E', 'C2(z)', 'C4(z)', 'C4^-1(z)', 'm(y)', 'm(x)', 'm([110])', 'm([1-10])']
print(table.antiunitary, np.allclose(table.characters.imag, 0))
#  (9, 10, 11, 12, 13, 14, 15, 16) True
```

The identity's column is each multiplet's dimension; the first and third rows
are equal, and differ from the middle one only at the two fourfold rotations,
`∓√2`. `P4mm1'` is a grey group, time reversal adjoined as the trailing `1'`
marks: its antiunitary operations at Γ are carried but are not columns, since a
corep table has no characters for them. The characters at Γ are real here, so
the real parts are printed.

`content` reads the table against the little group's own corep table. A
**corep** (co-representation) is the irreducible object of a group with
antiunitary operations; a leading `-` marks a double-valued label, carried by
states with spin. `labels` holds one corep per row, and the printout sums them,
`2×-GM6 + -GM7`. `partition=(0, 1, 0)` says the first and third rows carry the
same corep and the middle one another: two rows share a number exactly when
their character vectors agree, so the partition stands under any labelling of
the orbitals and any phase convention, where the labels need not
([below](#when-a-label-is-not-trustworthy)). `isolation` is the energy gap from
the set of bands to the nearest band outside it, `13.02` in the model's energy
units.

`content.table` is the character table the labels were read from, the table
`tqc.characters` returned above. Its `multiplets` and `dims` are the rows' bands
and degeneracies, and its `spectrum` every energy of the one diagonalization at
the point, so each level reads with its corep, its degeneracy and its energy:

```python
print(content.table)
#  CharacterTable(k=[0.0, 0.0], bands 0-5 in 3 multiplets, group=realized)
for multiplet, dim, label in zip(content.table.multiplets, content.table.dims, content.labels):
    print(multiplet, dim, label, round(float(content.table.spectrum[multiplet[0]]), 3))
#  (0, 1) 2 {'-GM6': 1} -16.35
#  (2, 3) 2 {'-GM7': 1} -12.375
#  (4, 5) 2 {'-GM6': 1} -4.325
```

## Every high-symmetry point at once

`hsp_table` reads the content at every tabulated point the model's zone
contains, off one solve over that k-list. The points are in the model's reduced
coordinates; ask for one by its name in `hsp`, never by coordinates typed from
elsewhere:

```python
print({name: k.tolist() for name, k in hsp.items()}, hsp.missing)
#  {'GM': [0.0, 0.0], 'M': [0.5, 0.5], 'X': [0.0, 0.5]} ('A', 'R', 'Z')
# X's partner in its star, (1/2, 0), is not the tabulated point
print(tqc.corep_content(rep, [0.5, 0.0], bands=6).gate)
#  kpoint_not_tabulated
t = tqc.hsp_table(rep, bands=6)
print(t.summary())
#  bands 0-5   (.bands)
#    their coreps at each high-symmetry point, GM being Γ   (.content)
#      GM: 2×-GM6 + -GM7
#      M: -M6 + 2×-M7
#      X: 3×-X5
print(t.partitions)
#  {'GM': (0, 1, 0), 'M': (0, 1, 1), 'X': (0, 0, 0)}
print(t.content["GM"], tqc.format_counts(t.content["GM"]))
#  {'-GM6': 2, '-GM7': 1} 2×-GM6 + -GM7
print(t.counts, t.complete)
#  {'-GM6': 2, '-GM7': 1, '-M6': 1, '-M7': 2, '-X5': 3} True
```

`t.content` holds the coreps point by point: at each point a mapping from
corep to multiplicity, the line `summary()` prints for it, which
`tqc.format_counts` writes in the same form. How many times each corep appears
at each high-symmetry point is the band set's **symmetry vector**, `t.counts`:
what a [decomposition](atomic-limits.md),
a [symmetry indicator](indicators.md) and the
[stable real-space invariants](srsi.md) read. Each point holds three doublets;
at `X` all three carry `-X5`, `partition=(0, 0, 0)`. `t.complete` says every
point resolved, and a decomposition and an indicator refuse a table that did
not. `hsp.missing` names the tabulated points the zone has no `k` for: a layer
is labelled in the three-dimensional space group of its slab
([a 2D model](layers.md)), and `A`, `R` and `Z` sit at `k_z = ½`. What becomes
of their rows is read in [atomic limits](atomic-limits.md#how-to-read-it) and
[symmetry indicators](indicators.md#how-to-read-it).

Without time reversal, a point and its partner `−k` are two points of the zone
wherever no operation of the group carries one onto the other. Haldane's
honeycomb with a sublattice offset is `P3`, and its `K` and `KA` are such a
pair:

```python
from pythtb.models import haldane

hal = tqc.realized_group(haldane(0.3, -1.0, 0.15, 0.5))
print(hal.group.name, list(tqc.high_symmetry_points(hal.group)))
#  P3 ['GM', 'K', 'KA', 'M']
print(tqc.hsp_table(hal.rep, bands=1).summary())
#  band 0   (.bands)
#    their coreps at each high-symmetry point, GM being Γ   (.content)
#      GM: GM1
#      K: K2
#      KA: KA3
#    M: no operation but the identity, whose one irrep a band set carries once per band   (.skipped)
```

The tables list the irreps at `K`, and those at `KA` are their complex
conjugates, named as the tables' band representations name them: `KA3` is the
conjugate of `K3`. At `M` no characters are read, and a decomposition and an
indicator read the band count.

## Label an isolated set of bands

`bands` names the set of bands, here and in the next two chapters: an integer
is the lowest that many bands, a filling, and a sequence is exactly those band
indices. A set of bands must be a union of whole multiplets at every point it
is read at, since the characters of part of a degenerate multiplet depend on
the basis the diagonalization happened to choose inside it. A set of bands that
cuts one is refused, `multiplet_split`, with the multiplet and its energies
named; at Γ the fifth band is half of a doublet:

```python
print(tqc.corep_content(rep, hsp["GM"], bands=4))
#  CorepContent(GM: -GM6 + -GM7, partition=(0, 1))
cut = tqc.corep_content(rep, hsp["GM"], bands=5)
print(bool(cut), cut.gate, cut.context["multiplet"])
#  False multiplet_split (4, 5)
```

Near a degeneracy, ordering by energy pairs the wrong partners: the top band of
a filled set of bands and the first band above it can be one multiplet, and
then the filled bands are not a set the symmetry can label. Label the enclosing
set of bands that closes the multiplet, or accept that the point carries no
isolated content; `isolation` is returned either way.

## One diagonalization, reused

Each call above solved the model at its point itself. To read several sets of
bands at one point, solve once and hand the states in. `sewing_matrix`,
`characters`, `corep_content` and `hsp_table` all take them as `states=`, in
pythtb's own forms: `None` (the default) or `rep.model`, to solve at `k` and
keep nothing; a solved `pythtb.WFArray`, whose states at `k` are used as they
stand; or, at one point, the `(evals, evecs)` pair of `tqc.solve_at`.

```python
gm = tqc.solve_at(model, hsp["GM"])                # energies and eigenvectors, once
print(tqc.multiplets(gm[0]))
#  ((0, 1), (2, 3), (4, 5), (6, 7))
for bands in (2, [4, 5], [6, 7]):
    print(bands, tqc.corep_content(rep, hsp["GM"], states=gm, bands=bands))
#  2 CorepContent(GM: -GM6, partition=(0,))
#  [4, 5] CorepContent(GM: -GM6, partition=(0,))
#  [6, 7] CorepContent(GM: -GM7, partition=(0,))
print(tqc.corep_content(rep, hsp["M"], states=gm, bands=2).gate)
#  states_not_at_k
```

A sewing matrix sandwiches bras and kets that must come
from one diagonalization: take them from two, at the same point or at an
equivalent `k + G`, and each picks its own basis inside every degenerate
multiplet, so `B` stays unitary, its trace changes, and the label changes with
no warning. Hence states that do not diagonalize `H(k)` at the point asked for
are refused, `states_not_at_k`, as Γ's states handed in at `M` are; and a
`WFArray` whose mesh does not contain the point is refused, `k_not_on_mesh`.
`k_grid` builds a Γ-centred mesh that
respects the group and says which high-symmetry points it contains:

```python
print(tqc.k_grid(group, 4), tqc.k_grid(group, 5), sep="\n")
#  KGrid(4x4x1 mesh, symmetric; samples GM, M, X)
#  KGrid(5x5x1 mesh, symmetric; samples GM; misses M, X)
wf = WFArray(model.lattice, tqc.k_grid(group, 5).mesh, spinful=True)
wf.solve_model(model)
print(tqc.hsp_table(rep, states=wf, bands=6).summary())
#  bands 0-5   (.bands)
#    their coreps at each high-symmetry point, GM being Γ   (.content)
#      GM: 2×-GM6 + -GM7
#    refused at M and X (k_not_on_mesh), so the content is incomplete   (.refusals)
```

The odd mesh misses the zone boundary, and the table read on it is not
complete.

`multiplets` reads as one level the energies closer than `degeneracy_tol`, in
the units of `H`, `1e-8` by default; `characters`, `corep_content` and
`hsp_table` take the same keyword for their rows. A model whose degenerate
levels carry noise, as a fitted one's can, takes a tolerance above the noise
and below the closest gap between levels: at `M` two doublets lie `0.271`
apart, and a tolerance above that reads them as one level.

```python
em = tqc.solve_at(model, hsp["M"])[0]
print(tqc.multiplets(em), tqc.multiplets(em, degeneracy_tol=0.3))
#  ((0, 1), (2, 3), (4, 5), (6, 7)) ((0, 1, 2, 3), (4, 5), (6, 7))
```

## One operation's sewing matrix

`sewing_matrix` reads the matrix of one operation `g` on a set of bands, at a
`k` that `g` leaves invariant ([how to read it](#how-to-read-it)), and takes
`g` by its index. The table of the operations the Hamiltonian obeys gives each
one's index, its Seitz symbol and its name in the model's axes:

```python
print(tqc.operations(r))
#  P4mm1' #99.164: one operation per coset of the lattice translations; the double group also holds each one's negative
#   ind  Seitz            name
#     1  {1|0,0,0}        E
#     2  {2_001|0,0,0}    C2(z)
#     3  {4+_001|0,0,0}   C4(z)
#     4  {4-_001|0,0,0}   C4^-1(z)
#     5  {m_010|0,0,0}    m(y)
#     6  {m_100|0,0,0}    m(x)
#     7  {m_110|0,0,0}    m([110])
#     8  {m_1-10|0,0,0}   m([1-10])
#     9  {1|0,0,0}'       E'
#    10  {2_001|0,0,0}'   C2(z)'
#    11  {4+_001|0,0,0}'  C4(z)'
#    12  {4-_001|0,0,0}'  C4^-1(z)'
#    13  {m_010|0,0,0}'   m(y)'
#    14  {m_100|0,0,0}'   m(x)'
#    15  {m_110|0,0,0}'   m([110])'
#    16  {m_1-10|0,0,0}'  m([1-10])'
```

`g` is found by its Seitz symbol with `operation_index`, which reads either
spelling of the axis and any representative of the translation
([conventions](../background/conventions.md#operations-a-seitz-symbol-and-a-cartesian-name)),
or by its matrix: each operation of the same table, `r.operations`, holds its
rotation in the Cartesian axes the lattice vectors are written in,
`rotation_cartesian`. Here a quarter turn about `z`, and its sewing matrix on
the lowest pair at `M`:

```python
c4 = tqc.operation_index(r, "{4+_001|0,0,0}")
quarter = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
print(c4, [(op.ind, op.name, op.translation.tolist(), op.time_reversal)
           for op in r.operations.values() if np.allclose(op.rotation_cartesian, quarter)])
#  3 [(3, 'C4(z)', [0.0, 0.0, 0.0], False), (11, "C4(z)'", [0.0, 0.0, 0.0], True)]
B = tqc.sewing_matrix(rep, c4, hsp["M"], bands=2)
print(np.round(B.trace, 6))
#  (-1.414214+0j)
print(np.round(r.operations[c4].spin_lift, 6) + 0)
#  [[0.707107-0.707107j 0.      +0.j      ]
#   [0.      +0.j       0.707107+0.707107j]]
```

Two operations share the rotation, `C4(z)` and `C4(z)'`, and `time_reversal`
tells apart the second, combined with time reversal. `translation`, in the
model's reduced coordinates, is the translation that comes with the rotation:
zero in this symmorphic group, the fractional part of a screw or a glide in
others. The fourfold leaves `M` invariant up to a reciprocal-lattice vector,
and the trace of `B` there, `−√2`, is the character the pair's label `-M6` is
read from. `spin_lift` is the operation's SU(2) matrix on the model's spin,
`e^{−iθ n̂·σ/2}` with `n̂` in the same axes and spin along their `z`: for the
quarter turn about `z`, `diag(e^{−iπ/4}, e^{iπ/4})`. At `X`, which the fourfold
carries to its partner, there is no sewing matrix, and the call refuses
[`little_group`](../reference/refusals.md#little_group). The table holds only
the operations the Hamiltonian obeys: an operation it breaks is not in it, and
`r.descent` names what broke it
([less symmetry than the cell](group.md#when-the-hamiltonian-has-less-symmetry-than-its-cell)).

## When a label is not trustworthy

**The conditioning at the point.** A quantity read off eigenvectors, a sewing
matrix or a character, is only as sharp as the multiplets at that point are
separated. It is judged against a dimensionless threshold measured there, the
larger of `1e-14` and the noise floor `8 ε ‖H(k)‖₂ / Δ`, with `ε` the machine
precision and `Δ` the smallest gap between distinct multiplets; `t.thresholds`
holds it per point, printed here beside `Δ` in the model's energy units:

```python
print({name: (f"{thr.effective:.1e}", round(thr.separation, 3))
       for name, thr in t.thresholds.items()})
#  {'GM': ('1.0e-14', 3.975), 'M': ('8.3e-14', 0.271), 'X': ('1.8e-14', 1.33)}
```

It is widest at `M`, where the multiplets crowd closest. A threshold grown
large says nothing read off a set of bands at that point can be trusted, and is
never a reason to loosen a tolerance.

**What the labels rest on.** The hoppings fix how the orbitals transform
relative to one another, not what they are
([what the hoppings fix](../background/measurement.md#what-the-hoppings-fix)). No orbital
content was declared here, so the two orbitals at `1a` were assigned by the
lowest-ℓ rule, the site irreps of the lowest angular momentum the hoppings
allow, and their rows say so. Declaring orbital 0 a `d_{x²−y²}`, the irrep
`B1` of its site group `4mm`, states another crystal the same hoppings
describe:

```python
print([(row.wyckoff, row.irrep, row.determined_by) for row in rep.characters.orbitals])
#  [('1a', '-E1', 'lowest-l'), ('1a', '-E2', 'lowest-l'), ('2c', '-E', 'hamiltonian'), ('2c', '-E', 'hamiltonian')]
other = tqc.realized_group(models.lin3g(), typat=[0, 0, 1, 1], characters={0: "dx2-y2"}).rep
print(other.characters.irreps)
#  ('-E2', '-E1', '-E', '-E')
relabelled = tqc.corep_content(other, hsp["GM"], bands=6)
print(relabelled, relabelled.partition == content.partition)
#  CorepContent(GM: -GM6 + 2×-GM7, partition=(0, 1, 0)) True
```

The hoppings then give orbital 1 the `-E1` that orbital 0 carried, and every
doublet at Γ changes its label, `-GM6` for `-GM7` and back, while the partition
stays. The labels are canonical relative to the site content the
representation carries: quote the partition with them always, prefer it
wherever a row reads `lowest-l`, and declare what the orbitals are where the
labels matter ([saying what the orbitals are](orbitals.md)).

**Near a reference level.** `frontier_subspace` takes a spectrum measured from
a reference level and returns as one set every state whose distance from the
level is within `modulus_tol` of the nearest state's, so a quantity read near a
degeneracy is read on the whole set. Its
`at_zero` says the set sits on the level itself: the signs of its states are
then not resolved, and a quantity read sector by sector, on the states of each
sign, is a source of spurious quantized values there
([its entry](../reference/functions.md#frontier_subspace)).

## When it refuses

A refusal is a returned record, falsy, naming the check that stopped the call
and what it measured ([when a call refuses](refusals.md)). `hsp_table` collects
each point's refusal in `t.refusals`, and `t.complete` is then `False`. The
checks a labelling meets:

- [`multiplet_split`](../reference/refusals.md#multiplet_split): the bands
  asked for cut a degenerate multiplet; label the enclosing set of bands.
- [`states_not_at_k`](../reference/refusals.md#states_not_at_k): the states
  handed in do not diagonalize `H(k)` at that point.
- [`k_not_on_mesh`](../reference/refusals.md#k_not_on_mesh): the mesh does not
  contain the point; build it with `k_grid`.
- [`asymmetric_grid`](../reference/refusals.md#asymmetric_grid): `k_grid` was
  asked for divisions the group does not leave invariant.
- [`kpoint_not_tabulated`](../reference/refusals.md#kpoint_not_tabulated): not
  a tabulated point in this cell, or not the tabulated member of its star.
- [`integer_multiplicity`](../reference/refusals.md#integer_multiplicity),
  [`multiplet_dimension`](../reference/refusals.md#multiplet_dimension): a
  multiplicity further than `label_tol` (`1e-3`, dimensionless) from an
  integer, or coreps whose dimensions do not add up to the multiplet's
  degeneracy; most often a multiplet grouped wrongly.
- [`little_group_mismatch`](../reference/refusals.md#little_group_mismatch),
  [`tables_not_ready`](../reference/refusals.md#tables_not_ready): the tables
  do not apply in this cell.

## Every field and argument

The fields of
[`CharacterTable` and `CorepContent`](../reference/records.md#charactertable-and-corepcontent),
of [`HSPTable`](../reference/records.md#hsptable) and of
[`Operation`](../reference/records.md#operationtable-and-operation) are in the
records reference, and each call's arguments and tolerances in its entry:
[`high_symmetry_points`](../reference/functions.md#high_symmetry_points),
[`characters`](../reference/functions.md#characters),
[`corep_content`](../reference/functions.md#corep_content),
[`hsp_table`](../reference/functions.md#hsp_table),
[`solve_at`](../reference/functions.md#solve_at),
[`multiplets`](../reference/functions.md#multiplets),
[`k_grid`](../reference/functions.md#k_grid),
[`format_counts`](../reference/functions.md#format_counts),
[`operations`](../reference/functions.md#operations),
[`operation_index`](../reference/functions.md#operation_index),
[`sewing_matrix`](../reference/functions.md#sewing_matrix). How the labels are
spelled is in [the conventions](../background/conventions.md#coreps-at-a-k-point).
