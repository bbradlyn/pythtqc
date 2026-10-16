# Conventions: operations, labels and band representations

The labels pythtqc prints come from one set of tables: the irreducible
representations, site-symmetry groups and
[elementary band representations](tqc-primer.md#band-representations-and-atomic-limits)
of the Bilbao Crystallographic Server (BCS), as the `irreptables` package
ships them. This page reads the names a printout carries, and then says what to
expect when comparing with the server's own pages, which represent a lattice
translation with the opposite phase. No orientation of a model is assumed: each
operation is identified by how it acts on the model's own cell, and each label by
measured characters. The models are pythtb's Kane–Mele,
`kane_mele(delta, t, soc, rashba)`, a spinful honeycomb on a substrate;
pythtb's graphene, `graphene(delta, t)`, free-standing and spinless; and
`pythtqc.models.i41_soc`, a spin–orbit crystal in the screw-axis group `I4_1`
(number 80).

```python
import numpy as np
import pythtqc as tqc
from pythtb.models import graphene, kane_mele
from pythtqc import models

km = kane_mele(1.0, -1.0, 0.3, 0.25)            # delta, t, soc, rashba
r = tqc.realized_group(km, typat=[0, 1], characters={0: "A1"})
print(tqc.operations(r))
#  P3m11' #156.50: one operation per coset of the lattice translations; the double group also holds each one's negative
#   ind  Seitz            name
#     1  {1|0,0,0}        E
#     2  {3+_001|0,0,0}   C3(z)
#     3  {3-_001|0,0,0}   C3^-1(z)
#     4  {m_110|0,0,0}    m([010])
#     5  {m_100|0,0,0}    m(x)
#     6  {m_010|0,0,0}    m([1-10])
#     7  {1|0,0,0}'       E'
#     8  {3+_001|0,0,0}'  C3(z)'
#     9  {3-_001|0,0,0}'  C3^-1(z)'
#    10  {m_110|0,0,0}'   m([010])'
#    11  {m_100|0,0,0}'   m(x)'
#    12  {m_010|0,0,0}'   m([1-10])'
```

## Operations: a Seitz symbol and a Cartesian name

Each operation has an index, the key every per-operation record is written
on, and two names, which `tqc.operations` prints side by side:

- **The Seitz symbol** `{R|t}` is written in the tables' conventional setting,
  as the International Tables and the server print it, with two stand-ins for
  typography: an underscore before the direction, and a minus sign for an
  overbar, so `4̄⁺₀₀₁` reads `-4+_001`. The rotation part is `1`, `-1`, `m`,
  `2`, `3`, `4`, `6`, `-3`, `-4` or `-6`; one of order above two carries its
  sense, `+` or `-`; the direction is that of the axis, or of a mirror's
  normal, as integers in the conventional basis with the first nonzero one
  positive; the translation is three fractions of the conventional cell, the
  representative the tables print. A trailing `'` marks an antiunitary
  operation, one combined with time reversal.
- **The Cartesian name** is read off the operation's matrix in the model's own
  Cartesian axes, and it is the frame `characters=` is written in: `E`;
  `C3(z)`, a counterclockwise turn by `2π/3` about `+z`, and `C3^-1(z)` its
  inverse; `C2(z)`, the half turn, which has no sense; `i`, the inversion;
  `m(x)`, the mirror whose normal is x — the plane `x = 0`, which Mulliken's
  notation calls `σ_v(yz)`. An axis is `x`, `y` or `z` where it lies along
  one, and a direction `[uvw]` in the model's own lattice vectors otherwise. In
  a declaration `"sigma_v(x)"`, `"sigma(x)"` and `"mx"` all read as `m(x)`. A
  record that lists operations by name writes one whose translation is not a
  lattice vector of the cell as `{name|t}`, the translation in the model's
  reduced coordinates: a broken half translation is `{E|1/2,0,0}`, which `E`
  alone would print as the identity.

The two directions are written in different bases, so one axis can carry two
triples. Kane–Mele's lattice vectors are 60° apart and the conventional
hexagonal ones 120°, and the mirror whose normal is the model's second lattice
vector, `m([010])`, is `m_110` in the conventional basis.

An improper rotation other than `i` and a mirror is named `S_n` after its
rotoinversion: a counterclockwise turn by `2π/n` about the axis followed by the
inversion, the Seitz `-n+`, and `S_n^-1` is the Seitz `-n-`. The free-standing
honeycomb holds both kinds a sixfold axis has:

```python
gr = tqc.realized_group(graphene(0.0, -1.0), environment="free-standing")   # spinless
ops = tqc.operations(gr)
for ind in (13, 14, 15, 16, 17, 18):
    print(ind, ops[ind].seitz, ops[ind].name)
#  13 {-1|0,0,0} i
#  14 {-3+_001|0,0,0} S3(z)
#  15 {-3-_001|0,0,0} S3^-1(z)
#  16 {m_001|0,0,0} m(z)
#  17 {-6-_001|0,0,0} S6^-1(z)
#  18 {-6+_001|0,0,0} S6(z)
```

`S3(z)` is `-3+_001`, the inversion after a counterclockwise turn by `2π/3`,
and `S6(z)` is `-6+_001`, as `S4(z)` is `-4+_001`: the `n` of a name is the
Seitz symbol's, and `^-1` goes with its `-`. Schoenflies notation names an
improper rotation after its rotation–reflection instead, the turn that,
combined with the mirror perpendicular to the axis, makes the operation, so
that `-3+_001` is the Schoenflies `S6⁵` and `-6+_001` the Schoenflies `S3⁵`.

The lookups go both ways: `tqc.seitz(group, ind)` returns one symbol, and
`tqc.operation_index(group, symbol)` the index of the operation a symbol
names, whichever representative of the translation and whichever of the two
spellings of the axis is written. In `I4_1` the fourfold axis is a screw:

```python
soc = tqc.realized_group(models.i41_soc("generic", lam=0.4), characters={0: "s"})
print(tqc.operations(soc))
#  I4_11' #80.30: one operation per coset of the lattice translations; the double group also holds each one's negative
#   ind  Seitz                name
#     1  {1|0,0,0}            E
#     2  {2_001|0,0,0}        C2(z)
#     3  {4+_001|0,1/2,1/4}   C4(z)
#     4  {4-_001|0,1/2,1/4}   C4^-1(z)
#     5  {1|0,0,0}'           E'
#     6  {2_001|0,0,0}'       C2(z)'
#     7  {4+_001|0,1/2,1/4}'  C4(z)'
#     8  {4-_001|0,1/2,1/4}'  C4^-1(z)'
for symbol in ("{4+_001|0,1/2,1/4}", "{4+_001|1/2,0,3/4}", "{4-_00-1|0,1/2,1/4}"):
    print(symbol, tqc.operation_index(soc, symbol))
#  {4+_001|0,1/2,1/4} 3
#  {4+_001|1/2,0,3/4} 3
#  {4-_00-1|0,1/2,1/4} 3
missing = tqc.operation_index(soc, "{4+_001|0,0,1/4}")
print(missing.gate, missing.context["nearest"])
#  unknown_operation ('{4+_001|0,1/2,1/4}', "{4+_001|0,1/2,1/4}'", '{4-_001|0,1/2,1/4}')
```

The translation written is the tables' own representative. Adding the body
centring, `1/2,1/2,1/2`, names the same operation, and so does `4-` about
`-z`, the same turn about the opposite direction. A symbol no operation of the
group carries is refused
[`unknown_operation`](../reference/refusals.md#unknown_operation), with the
nearest spellings the group does carry. Finding an operation by its matrix,
and reading its sewing matrix at a `k` it leaves invariant, are in
[one operation's sewing matrix](../guide/labels.md#one-operations-sewing-matrix).

## Site irreps

A site irrep is an irreducible representation of a site-symmetry group. Its
name is the one the tables give it in their listing of site-symmetry groups,
the listing a BANDREP page of the server prints beside each Wyckoff position,
so a label copied off such a page is the label returned here; the elementary
band representations, the
[stable real-space invariants](../guide/srsi.md) and the published tables
they are checked against all key on it. Where the tables' entries at a
position do not name a row, it keeps the site point group's own name, and the
atomic-limit table says which
([`named by`](../reference/records.md#the-atomic-limit-table)). The listing
differs from an abstract point-group table at some sites. The free-standing
honeycomb's site group, with its rows as the tables name them:

```python
print(gr.rep.characters.orbitals[0].wyckoff, gr.rep.characters.orbitals[0].site_symmetry)
#  2c -6m2
table = gr.rep.characters.site_tables[0]        # the site group's own character table
print("      " + "".join(f"{name:>8}" for name in table.class_names))
#               E  2C3(z)  3C2(y)    m(z)  2S6(z)   3m(x)
for label, row in zip(table.labels, table.characters.real.round(6) + 0):
    print(f"{label:>6}" + "".join(f"{x:>8g}" for x in row))
#    A2''       1       1      -1      -1      -1       1
#    A1''       1       1      -1       1       1      -1
#     A2'       1       1       1      -1      -1      -1
#     A1'       1       1       1       1       1       1
#     E''       2      -1       0      -2       1       0
#      E'       2      -1       0       2      -1       0
```

`A1''` here is the row even under the horizontal mirror `m(z)` and odd under
the in-plane twofolds. Mulliken's convention (R. S. Mulliken, J. Chem. Phys.
**23**, 1997 (1955)) names a row by its characters: `A` or `B` by the principal
axis, `1` or `2` by the twofolds perpendicular to it, `'` or `''` by the
horizontal mirror, `g` or `u` by inversion, and at a sixfold site `E1` the pair
that transforms like `(x, y)`. Its table of `-6m2` calls that row `A2'`, and
calls the row printed `A2'` here `A1''`; `A1'`, `A2''`, `E'` and `E''` agree.
The tables name rows against a mark no choice of axes moves at four kinds of
site, the first two told apart by the orientation the
[band-representation key](#the-band-representation-key) writes their group in:

| site group | sites | the tables' names against Mulliken's |
|---|---|---|
| `-6m2`, keyed `-6m2` | `1a`–`1f` of `P-6m2`; `2c`, `2d` of `P6/mmm`, graphene's; `2b`, `2c`, `2d` of `P6_3/mmc` | `A1''` is Mulliken's `A2'`, and `A2'` his `A1''` |
| `-42m`, keyed `-4m2` | `1a`–`1d` of `P-4m2`; `2a`–`2d` of `I-4m2`; `2a`, `2b` of `P4/nmm`, iron's in FeSe; `2e`, `2f` of `P4_2/mmc`; `2a`, `2b` of `P4_2/nmc`; `4d` of `I4/mmm`, iron's in BaFe₂As₂; `4a`, `4b` of `I4_1/amd`, titanium's in anatase; `6c`, `6d` of `Pm-3n`; `24c` of `Fm-3c`; `12d` of `Im-3m` | `B2`, `A2` and `B1` are Mulliken's `A2`, `B1` and `B2`, so `p_z` and `d_{x²−y²}` are the tables' `B1` and `d_xy` their `A2` |
| `6` | `1a` of `P6`; `2a` of `P6cc`; `2e` of `P6/m`; `2e` of `P622`; `4e` of `P6/mcc` | `E1` and `E2` are exchanged |
| `6/m` | `1a`, `1b` of `P6/m`; `2b` of `P6/mcc` | `E1g` and `E2g` are exchanged, and `E1u` and `E2u` |

A `-6m2` site keyed `-62m` and a `-42m` site keyed `-42m` follow Mulliken's
convention. A declaration is resolved against the tables' names, so at the
sites above Mulliken's spelling of an exchanged name states the other irrep,
and a declaration written in one carries a note saying which row Mulliken's
convention calls it, on its row and on its line under the atomic-limit table:

```python
declared = tqc.realized_group(graphene(0.0, -1.0), environment="free-standing",
                              characters={0: "A1''"})
print(declared.rep.characters.summary())
#  orbital  Wyckoff  site symmetry  function  irrep  named by  determined by  orbital characters
#        0       2c           -6m2             A1''    tables    declaration  E=+1, C3(z)=+1, C2(y)=-1, m(z)=+1, S6(z)=+1, m(x)=-1
#        1       2c           -6m2             A1''    tables    hamiltonian  E=+1, C3(z)=+1, C2(y)=-1, m(z)=+1, S6(z)=+1, m(x)=-1
#    orbital 0: declared "A1''", read at the -6m2 in P6/mmm1' #191.234; the tables' A1'' at this site is the row Mulliken's convention calls A2'; returned as A1'' of -6m2   (.declarations)
```

At a site whose group is the threefold axis `3` alone, the tables call the
trivial irrep `A1`, where a point-group table prints `A`, so `{0: "A1"}` is the
word at such a site. Two spellings cannot be misread: `"s"`, which reaches the
trivial irrep of any site group, and generator characters, which name no irrep
at all ([declaring orbitals](../guide/orbitals.md#the-forms-a-declaration-takes)).

The other marks: `g` and `u` give the character of the inversion, and `'` and
`''` that of a horizontal mirror, but at the `-6m2` sites above; a leading `-`
marks a double-valued irrep, the label of states with spin; `1E2E`
is two complex-conjugate one-dimensional irreps glued by time reversal into
one (`-1E2E` with spin). A site content of several irreps is their sum, with
multiplicities: `A1 + E`, or `2x-E1 + -1E2E` for a spin–orbit `p` shell at a
`3m` site.

## Coreps at a k-point

The band labels at a high-symmetry point are the tables' irreducible
representations of its little group, or, in a group with time reversal, its
coreps, the irreducible objects of a group that holds antiunitary operations
([primer](tqc-primer.md)). Kane–Mele's two lowest bands:

```python
hsp = tqc.high_symmetry_points(r.group)
for point in ("GM", "K", "M"):
    print(tqc.corep_content(r.rep, hsp[point], bands=2))
#  CorepContent(GM: -GM6, partition=(0,))
#  CorepContent(K: -K5 + -K6, partition=(0, 1))
#  CorepContent(M: -M3M4, partition=(0,))
```

A label is the point's name, `GM` for Γ and the tables' letters otherwise,
and an index, with a leading `-` where it is double-valued. Where an
antiunitary operation fixes the point, Wigner's three cases are spelled
apart: an irrep that is a corep alone keeps its name (`-GM6`); two different
irreps glued into one level are written together (`-M3M4`, of the unitary
irreps `-M3` and `-M4`); one irrep doubled is written twice (`-P4P4`,
[below](#points-where-the-gluing-differs)). Constituents stand in the
lexicographic order of their bare names, with the minus once in front.
Separate levels are joined by ` + `: at `K` the two bands are two
one-dimensional coreps, and `partition=(0, 1)` numbers them apart because
their characters differ
([`CorepContent`](../reference/records.md#charactertable-and-corepcontent)). A
content is returned as a mapping from each label to its multiplicity.

## The origin the names are read about

The tables write a group about one origin, and a group can have several points
the tables could be read about equally well: every point its operations'
translations allow. In `P4mm` the fourfold axes through `(0, 0)` and `(½, ½)`
both can be, and between the two the Wyckoff positions `1a` and `1b` trade
names, as do irreps at `X` and `M`. The names returned are read about the one
nearest the model's own origin, so an orbital the model writes at `[0, 0]` is
named at the tables' origin wherever that point can be one. On every model
built from exact numbers measured that is the point `irrep` chooses anyway; with
a precision declared, the rule is applied to centres first made exact. The Lieb
lattice, its corner orbital at the origin, and the same lattice with every
centre shifted by a few millionths of a cell, as a fitted model's can be, its
precision declared:

```python
from pythtb import Lattice, TBModel

corners = [[0.0, 0.0], [0.5, 0.0], [0.0, 0.5]]
lieb = TBModel(Lattice([[1.0, 0.0], [0.0, 1.0]], corners, periodic_dirs=[0, 1]))
lieb.set_onsite([0.5, 0.0, 0.0], mode="set")
for a, b, R in ((0, 1, [0, 0]), (0, 1, [-1, 0]), (0, 2, [0, 0]), (0, 2, [0, -1])):
    lieb.set_hop(-1.0, a, b, R, mode="add")
shifted = TBModel(Lattice([[1.0, 0.0], [0.0, 1.0]], np.asarray(corners) + [3e-6, -2e-6],
                          periodic_dirs=[0, 1]))
shifted.set_onsite([0.5, 0.0, 0.0], mode="set")
for a, b, R in ((0, 1, [0, 0]), (0, 1, [-1, 0]), (0, 2, [0, 0]), (0, 2, [0, -1])):
    shifted.set_hop(-1.0, a, b, R, mode="add")
for model, declared in ((lieb, {}), (shifted, {"precision": 1e-4})):
    rl = tqc.realized_group(model, typat=[0, 1, 1], **declared)
    print(rl.rep.characters.orbitals[0].wyckoff, tqc.hsp_table(rl.rep, bands=1).counts)
#  1a {'GM1': 1, 'X1': 1}
#  1a {'GM1': 1, 'X1': 1}
```

Under a declared precision the centres are made exact before anything is
named, and a shift of all of them together no larger than the precision is
noise and is removed
([numerical models](../guide/numerical.md#the-centres-first-because-moving-them-is-a-gauge)),
so the shifted lattice is named as the exact one is. Where two such points are
equally near the model's origin, the names are read about the one `irrep`
chooses on the cell, the same in every run: a convention of the tables
([limits](limits.md#two-equally-near-origins)).

## The band-representation key

An elementary band representation (EBR) is the band set one site irrep
induces from its Wyckoff position, where that set is no sum of others
([primer](tqc-primer.md)). The records' fields hold each under one key, the
label the group's list of EBRs gives it, and so do the methods that write a sum
on one line, a decomposition's `solution_strings()` and a band
representation's `spelled()`:

```python
d = tqc.decompose(tqc.hsp_table(r.rep, bands=2), r.group, rep=r.rep)
print(d.solution_strings())
#  ("1 x -E1↑G(2)@1a(3m1',3m)",)
print(gr.rep.characters.band_representations[0].spelled())
#  1 x A1'↑G(2)@2c(-6m21',-6m2)
print([key for key in tqc.ebr_rows(gr.group) if key.startswith("A1'")])
#  ["A1'↑G(2)@2c(-6m21',-6m2)", "A1''↑G(2)@2c(-6m21',-6m2)", "A1'↑G(2)@2d(-6m21',-6m2)", "A1''↑G(2)@2d(-6m21',-6m2)"]
```

Read `1 x -E1↑G(2)@1a(3m1',3m)` left to right:

- `1 x`: how many times this EBR enters the sum.
- `-E1`: the site irrep, here double-valued.
- `↑G`: induced up to the space group `G`.
- `(2)`: the number of bands it makes, the site irrep's dimension times the
  number of sites of the orbit in a primitive cell: 2 × 1 here, and 1 × 2 for
  graphene's `A1'` at `2c`.
- `@1a`: the Wyckoff position, multiplicity and letter, in the tables'
  setting. The multiplicity counts the sites in the conventional cell, which
  on a centred lattice holds more than one primitive cell: `3a` of `R-3m` is
  one site per primitive cell, so `-E1g↑G(2)@3a(-3m1',-3m)` is two bands.
- `(3m1',3m)`: the site-symmetry group twice, first with time reversal (`1'`
  adjoined), then its unitary part. In a group without time reversal the two
  entries are one group.

A summary writes the same EBR as its site irrep at its Wyckoff position, the
site symmetry in parentheses, `-E1 at 1a (3m)`, with a multiplicity above one
in front, `2×-E1 at 1a (3m)`; the records of the
[stable real-space invariants](../guide/srsi.md) write it as the invariants'
formulas do, `1 (-E1)_a`, one copy of `-E1` at the Wyckoff position of letter `a`.

The EBR files the tables ship write every improper site group without its
bar: `6m2` for the `-6m2` at `2c` of `P6/mmm`, `3m` for the `-3m` at `3a` of
`R-3m`. pythtqc names each site group from the operations that fix the
position and writes the bar back, so graphene's position reads `2c (-6m2)` in
every summary and `2c(-6m21',-6m2)` in every key pythtqc returns: a
decomposition's, an orbit's localization, the atomic-limit table's band
representations', `ebr_rows`' and `ebr_data`'s.

## Reading across to BCS

A Seitz symbol is the server's general position, its axis at most written the
other way round (`3+_111` where a table may print `3-_-1-1-1`, the same
operation, which `operation_index` reads either way), and the site irrep and
corep names are the server's, so most of what pythtqc prints reads straight
across. One convention differs. The server's tables represent a lattice
translation `t` by `e^{+ik·t}`; Bloch functions transformed as
`(g f)(r) = f(g⁻¹r)` carry `e^{−ik·t}`, which is what pythtb's Bloch
Hamiltonian realizes and a
[sewing matrix](tqc-primer.md#little-groups-irreps-and-coreps) measures.
pythtqc carries the tables
into that physical convention once, where they are read, and every label and
EBR row it returns is in it. Since `e^{+ik·t} = e^{−i(−k)·t}`, the difference
is one sentence: **the labels the server ships at `k` are the labels of the
physical irreps at `−k`.** At a time-reversal-invariant momentum (TRIM), where
`k ≡ −k`, nothing moves, which is why the two agree almost everywhere; and the
Wyckoff positions never move.

### Labels away from the time-reversal-invariant points

At a point that is not a TRIM, an EBR row pairs labels with Wyckoff positions
differently here and on the server's BANDREP pages. Kane–Mele with
`delta = 50` puts its two lowest bands on orbital 1, at `1b`, whose on-site
energy is `−50`:

```python
deep = tqc.realized_group(kane_mele(50.0, -1.0, 0.3, 0.25), typat=[0, 1], characters={0: "A1"})
low = tqc.hsp_table(deep.rep, bands=2)
print(low.counts, tqc.decompose(low, deep.group, rep=deep.rep).solution_strings())
#  {'-GM6': 1, '-K5': 1, '-K4': 1, '-M3M4': 1} ("1 x -E1↑G(2)@1b(3m1',3m)",)
convention = tqc.ebr_data(deep.group)["convention"]
print(convention["name"], convention["changed_points"], convention["conjugation"]["-K5"])
#  physical (translations as e^{-ik.t}) ('H', 'K') -K6
```

The decomposition places the pair at `1b`, the site it sits on. On the
server's page the `-E1` row at `1b` carries `-K6` where the row returned here
carries `-K5`: the convention record lists the points whose labels moved,
`changed_points`, and pairs each label with its partner under
`conjugation`. Read literally against the server's page, this pair's labels
match the `1c` row, the other site. So, comparing with the server: at a TRIM
read the labels as they are; at a point of `changed_points` exchange each
label for its partner; and expect the Wyckoff positions, which are what a
decomposition is about, to be the ones returned here.

In a group without time reversal the server's band representations also hold
rows at a point's partner `−k`, `KA` beside `K` and likewise `HA`, `PA` and
`WA`, whose irreps its character tables do not list. They are the complex
conjugates of the listed point's, `KAn` of `Kn`, and they are read so, in the
same physical convention as every other point.

### Points where the gluing differs

At a point that is not a TRIM, time reversal alone does not fix it, and the
antiunitary operation that does decides which irreps form one corep, through a
sum whose phases carry the convention. At a few points the two conventions
glue differently, so that a corep can change its dimension: the re-glued
points. `P` of `I4_1`, `(¼, ¼, ¼)` in the primitive cell's reduced
coordinates and `(½, ½, ½)` in the conventional cell's, is one: the
antiunitary operations that fix it are the fourfold screws combined with time
reversal, `C4(z)'` and `C4^-1(z)'`, and there the spin–orbit model's four
lowest states split `[1, 2, 1]`:

```python
P = tqc.high_symmetry_points(soc.group)["P"]
print(P)
#  [0.25 0.25 0.25]
states = tqc.solve_at(soc.model, P)
print(np.round(states[0][:4], 6))
#  [-2.744031 -2.501964 -2.501964 -2.259907]
for multiplet in ([0], [1, 2], [3]):
    print(multiplet, tqc.corep_content(soc.rep, P, states=states, bands=multiplet).counts)
#  [0] {'-P3': 1}
#  [1, 2] {'-P4P4': 1}
#  [3] {'-P3': 1}
for multiplet in ([0], [1, 2]):
    print(multiplet, np.round(tqc.characters(soc.rep, P, states=states, bands=multiplet).characters, 6) + 0)
#  [0] [[1.+0.j 0.-1.j]]
#  [1, 2] [[2.+0.j 0.+2.j]]
lg = tqc.little_group(soc.group, P)                # the operations that fix P
print([soc.operations[op.ind].name for op in lg.unitary],
      [soc.operations[op.ind].name for op in lg.antiunitary])
#  ['E', 'C2(z)'] ["C4(z)'", "C4^-1(z)'"]
print(soc.group.spinor, soc.group.magnetic)        # which of the tables are read
#  True True
for label, row in tqc.corep_table(soc.group, P).items():
    print(label, np.round(list(row.values()), 6) + 0)
#  -P4P4 [2.+0.j 0.+2.j]
#  -P3 [1.+0.j 0.-1.j]
convention = tqc.ebr_data(soc.group)["convention"]
print(convention["conjugated_points"], convention["divergent_points"], convention["reglued_points"],
      convention["changed_labels"])
#  ('M', 'P') ('P',) ('P',) ('-P3P3', '-P4')
```

`little_group` holds the operations that fix `P`: `E` and `C2(z)` are
`unitary`, and the two screws combined with time reversal `antiunitary`. The
unitary irreps are one-dimensional: `-P3`, whose characters `(1, −i)` the
lowest state carries, and `-P4`, with `(1, +i)`. A pair of one-dimensional
irreps cannot be held together by anything unitary, so the antiunitary
operation pairs `-P4` with itself: the doublet is `-P4P4`, its characters
twice `-P4`'s. `corep_table` is the page returned here, each corep of the
little group with its characters by operation index, read from the
double-valued (`spinor`) magnetic (`magnetic`) tables a grey group takes; its
two rows are the characters measured above. The server's page doubles the
other irrep, and `-P3P3` and `-P4` are the names it carries at `P` that the
page returned here does not. The two
pages are complex conjugates of each other: they describe the two star members
`P` and `−P`, and the measured trace says which one the Hamiltonian is at.
[`examples/tqc_sg80_soc_doublet.py`](../../examples/tqc_sg80_soc_doublet.py)
runs this measurement whole.

`ebr_data(group)["convention"]` records, group by group, what the convention
moved. At `changed_points` a label moved. At `conjugated_points` the
characters returned are the conjugates of the server's, under the server's name
of the row that comes out, so the server's table is still the one printed there
and only which label a band lands on can differ. At `divergent_points` the
server's page tabulates the other star member, `−k`, and its rows are not a
reordering of these; `reglued_points` are those of them where time reversal
glues the irreps differently, and `changed_labels` the names that moved.
`row_permutation` re-keys a linear form written in the server's labels, an
indicator formula copied off its pages, so that it can be evaluated on a
returned
[symmetry vector](tqc-primer.md#how-a-symmetry-indicator-is-computed).

## Spin

The double-valued labels are written with one lift of every operation onto
the spin: spin quantized along `+z`, every `2×2` block in `(↑, ↓)` order, a
rotation by `θ` about `n̂` acting on the spinor as `e^{−iθ n̂·σ/2}`, an
improper operation lifted through its proper part (`S(i·R) = S(R)`, so a
spinful `s` orbital at an inversion centre is the even spinor), and time
reversal acting as `iσ_y` with complex conjugation. The tables' spin matrices
are written in the conventional cell's standard orientation, and the rotation
carrying that frame onto the model's cell is computed, not fitted: the frame is
right-handed, and the model's own Cartesian axes are the ones its spin blocks
are read in. pythtb writes spinful models in this convention, so a pythtb
model needs nothing; a model transcribed from a source written for `−z` does.
[Spin](../guide/spin.md) prints the lift in a model's own matrices, converts
a model written for the other axis, and shows what a wrong frame does to the
group.
