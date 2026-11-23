# Conventions: operations, labels and band representations

pythTQC writes each operation's Seitz symbol, and names Wyckoff positions, site
irreps and little-group representations, as the tables of the Bilbao
Crystallographic Server (BCS) do, in the form the `irreptables` package ships
them. An elementary band representation (EBR) is written as its site irrep at
its Wyckoff position. Each operation also carries a name read in the Cartesian
axes in which the pythtb Lattice is specified. This page says how each name is
spelled. It then says where a name read off the server's own pages differs from
the one returned: the server represents a lattice translation `t` by
`e^{+ik.t}`, and pythTQC by `e^{-ik.t}`. A model need not be written in the
tables' setting. Each operation is named from how it acts on the model's own
cell, and each label from measured characters.

The examples use three models: Kane-Mele, `kane_mele(delta, t, soc, rashba)`
from `pythtb.models`, a spinful honeycomb layer read on a substrate, the
default of `environment=`; graphene, `graphene(delta, t)` from `pythtb.models`,
spinless and read with `environment="free-standing"`
([Is my 2D model on a substrate or free-standing?](../guide/layers.md)); and
`pythtqc.models.i41_soc`, a spin-orbit crystal in the space group `I4_1`,
number 80, whose fourfold axis is a screw.

## Operations: a Seitz symbol and a Cartesian name

`r.operations`, on the record `realized_group` returns, lists the operations of
the group, one for each coset of the lattice translations.
`tqc.operations(r.group)` returns the same table. Each operation has an index,
a Seitz symbol and a name. The example is Kane-Mele with a staggered potential
and a Rashba term, its two sites declared atoms of different species:

```python
import numpy as np
import pythtqc as tqc
from pythtb.models import graphene, kane_mele
from pythtqc import models

km = kane_mele(1.0, -1.0, 0.3, 0.25)            # delta, t, soc, rashba
r = tqc.realized_group(km, typat=[0, 1], characters={0: "A1"})
print(r.operations)
#  P3m11' #156.50: one operation per coset of the lattice translations; the double group also holds each one's negative
#  index  Seitz            name
#      1  {1|0,0,0}        E
#      2  {3+_001|0,0,0}   C3(z)
#      3  {3-_001|0,0,0}   C3^-1(z)
#      4  {m_110|0,0,0}    m([010])
#      5  {m_100|0,0,0}    m(x)
#      6  {m_010|0,0,0}    m([1-10])
#      7  {1|0,0,0}'       E'
#      8  {3+_001|0,0,0}'  C3(z)'
#      9  {3-_001|0,0,0}'  C3^-1(z)'
#     10  {m_110|0,0,0}'   m([010])'
#     11  {m_100|0,0,0}'   m(x)'
#     12  {m_010|0,0,0}'   m([1-10])'
```

- `P3m11' #156.50`: the group returned, `r.group`.
- `the double group also holds each one's negative`: printed for a model with
  spin. Each operation listed stands for two operations of the double group,
  which differ by a rotation through 2π.
- `{3+_001|0,0,0}`: the Seitz symbol `{R|t}`, written in the conventional cell
  used by the irreptables package.
- `C3(z)`: the name.
- `{1|0,0,0}'` and `E'`: a trailing `'` marks an operation combined with time
  reversal, in the Seitz symbol and in the name.

The Seitz symbol is written as the International Tables for Crystallography
and the server write it, with two changes for plain text. An underscore stands
before the direction, and a minus sign stands for an overbar: the overbarred 4
with sense `+` about `[001]` is written `-4+_001`. The rotation part is `1`,
`-1`, `m`, `2`, `3`, `4`, `6`, `-3`, `-4` or `-6`. A part of order above two
carries its sense, `+` or `-`. The direction is that of the axis, or of a
mirror's normal, as integers in the conventional basis, the first nonzero one
positive. The translation is three fractions of the conventional cell, the
representative the tables give.

The name is read off the operation's matrix in the Cartesian axes in which the
pythtb Lattice is specified, the axes `characters=` is written in. An axis
along x, y or z is named by that letter, and any other by its direction
`[uvw]` in the model's lattice vectors. `E` is the identity and `i` the
inversion. `C3(z)` is a counterclockwise turn by 2π/3 about `+z`, and
`C3^-1(z)` its inverse. `C2(z)` is the half turn, which has no sense. `m(x)`
is the mirror whose normal is x, the plane `x = 0`. Mulliken's notation calls
the same mirror `σ_v(yz)`, after the axes its plane contains. In a
declaration, `"sigma_v(x)"`, `"sigma(x)"` and `"mx"` all read as `m(x)`.

A record that lists operations by name, such as the operations of the cell's
group that the Hamiltonian breaks, writes an operation whose translation is
not a lattice vector of the cell as `{name|t}`, the translation in the model's
reduced coordinates. A broken half translation is `{E|1/2,0,0}`; written `E`
it would read as the identity.

The Seitz symbol and the name write directions in different bases, so one axis
can carry two triples. Kane-Mele's lattice vectors are 60 degrees apart, and
the conventional hexagonal ones 120 degrees. The mirror whose normal is the
model's second lattice vector, `m([010])`, is `m_110` in the conventional
basis.

An improper rotation other than `i` and a mirror is named after its
rotoinversion. `Sn(z)` is a counterclockwise turn by 2π/n about `+z` followed
by the inversion, the Seitz `-n+`, and `Sn^-1(z)` is the Seitz `-n-`. So
`S4(z)` is `-4+_001`. The example is graphene read free-standing, whose group
holds the rotoinversions `-3` and `-6` about its sixfold axis:

```python
gr = tqc.realized_group(graphene(0.0, -1.0), environment="free-standing")   # spinless
ops = gr.operations
for ind in (13, 14, 15, 16, 17, 18):
    print(ind, ops[ind].seitz, ops[ind].name)
#  13 {-1|0,0,0} i
#  14 {-3+_001|0,0,0} S3(z)
#  15 {-3-_001|0,0,0} S3^-1(z)
#  16 {m_001|0,0,0} m(z)
#  17 {-6-_001|0,0,0} S6^-1(z)
#  18 {-6+_001|0,0,0} S6(z)
```

- `{-3+_001|0,0,0} S3(z)`: `n = 3`, the turn by 2π/3.
- `{-6+_001|0,0,0} S6(z)`: `n = 6`, the turn by 2π/6.
- `{m_001|0,0,0} m(z)`: the horizontal mirror, the rotoinversion of order two.

Schoenflies notation names an improper rotation after its rotation-reflection
instead: the turn that, followed by the mirror perpendicular to the axis, makes
the operation. In that notation `-3+_001` is `S6^5` and `-6+_001` is `S3^5`.

`tqc.operation_index(r.group, symbol)` returns the index of the operation a
Seitz symbol names. The example is the spin-orbit crystal in `I4_1`, whose
fourfold axis is a screw:

```python
soc = tqc.realized_group(models.i41_soc("generic", lam=0.4), characters={0: "s"})
print(soc.operations)
#  I4_11' #80.30: one operation per coset of the lattice translations; the double group also holds each one's negative
#  index  Seitz                name
#      1  {1|0,0,0}            E
#      2  {2_001|0,0,0}        C2(z)
#      3  {4+_001|0,1/2,1/4}   C4(z)
#      4  {4-_001|0,1/2,1/4}   C4^-1(z)
#      5  {1|0,0,0}'           E'
#      6  {2_001|0,0,0}'       C2(z)'
#      7  {4+_001|0,1/2,1/4}'  C4(z)'
#      8  {4-_001|0,1/2,1/4}'  C4^-1(z)'
for symbol in ("{4+_001|0,1/2,1/4}", "{4+_001|1/2,0,3/4}", "{4-_00-1|0,1/2,1/4}"):
    print(symbol, tqc.operation_index(soc.group, symbol))
#  {4+_001|0,1/2,1/4} 3
#  {4+_001|1/2,0,3/4} 3
#  {4-_00-1|0,1/2,1/4} 3
missing = tqc.operation_index(soc.group, "{4+_001|0,0,1/4}")
print(missing.gate, missing.context["nearest"])
#  unknown_operation ('{4+_001|0,1/2,1/4}', "{4+_001|0,1/2,1/4}'", '{4-_001|0,1/2,1/4}')
```

- `{4+_001|0,1/2,1/4}`: the screw, a quarter turn about `+z` with its
  translation, the representative the tables give.
- `{4+_001|1/2,0,3/4} 3`: a translation that differs by a lattice vector, here
  the body-centering vector `1/2,1/2,1/2`, names the same operation.
- `{4-_00-1|0,1/2,1/4} 3`: a turn written about the opposite direction of its
  axis with the opposite sense, `4-` about `-z`, names the same operation.
- `unknown_operation`: a symbol that names no operation of the group is refused
  with [`unknown_operation`](../reference/refusals.md#unknown_operation).
  `context["nearest"]` holds up to three symbols the group carries: the nearest
  in spelling, or the first three of its operations where none is close.

Finding an operation by its matrix, and reading its sewing matrix at a `k` it
leaves invariant, are in
[Which little-group representations do these bands carry?](../guide/labels.md).

## Site irreps

A site irrep is an irreducible representation of a site-symmetry group,
carried by an orbital or a Wannier function at its site. It takes the name the
tables of EBRs give it at its Wyckoff position, which is the name a BANDREP
page of the server prints beside that position. The EBRs and the tables of the
stable real-space invariants are written in these names
([What do the stable real-space invariants say about these bands?](../guide/srsi.md)).
At a Wyckoff position where the tables list no EBR, a single-valued site irrep
takes the name of the one site irrep that has the same stable real-space
invariants in the tables of those invariants. Otherwise it takes the name the
tables of EBRs give the same irrep of the site's point group at the origin of a
symmorphic group. A site irrep those names leave open keeps its Mulliken
symbol, and one that nothing names is left without a name.
`numerical.named_by`, on each row of `r.wyckoff_orbits`, says which of these
named it: `tables`, `invariants`, `point group`, `Mulliken` or `unnamed`
([`WyckoffOrbitNumerical`](../reference/records.md#wyckofforbitnumerical)).
The example is graphene read free-standing, its one orbit at `2c`:

```python
print(gr.wyckoff_orbits[0].wyckoff, gr.wyckoff_orbits[0].unitary_site_symmetry,
      gr.wyckoff_orbits[0].numerical.named_by)
#  2c -6m2 tables
```

- `2c`: the Wyckoff position, multiplicity and letter.
- `-6m2`: the site-symmetry group, without time reversal.
- `tables`: the site irreps are named by the EBRs the tables list at `2c`.

Mulliken's convention (R. S. Mulliken, J. Chem. Phys. 23, 1997 (1955)) names a
one-dimensional irrep by its characters: `A` or `B` by the principal axis, `1`
or `2` by the twofolds perpendicular to it, `'` or `''` by the horizontal
mirror, and `g` or `u` by the inversion. At a sixfold site, `E1` is the pair
that transforms like `(x, y)`. At four kinds of site the tables' names differ
from Mulliken's, and no choice of axes removes the difference. Graphene's `2c`
is one. The tables' `A1''` there is even under the horizontal mirror `m(z)`
and odd under the in-plane twofolds: it is the row Mulliken's table of `-6m2`
calls `A2'`. The tables' `A2'` is Mulliken's `A1''`, and `A1'`, `A2''`, `E'`
and `E''` agree. The EBR files the tables ship write the symbol of a site group
of `-6m2` in one of two orientations, `6m2` or `62m`, and that of `-42m` as
`4m2` or `42m`, each without its bar. A term returned writes the bar back on
the files' symbol: `-6m2`, `-62m`, `-4m2` or `-42m`. At these two point groups
the names differ only where the files write the orientation the first column
gives:

| site group | sites | the tables' names against Mulliken's |
|---|---|---|
| `-6m2`, written `-6m2` | `1a` to `1f` of `P-6m2`; `2c`, `2d` of `P6/mmm`, graphene's; `2b`, `2c`, `2d` of `P6_3/mmc` | `A1''` is Mulliken's `A2'`, and `A2'` his `A1''` |
| `-42m`, written `-4m2` | `1a` to `1d` of `P-4m2`; `2a` to `2d` of `I-4m2`; `2a`, `2b` of `P4/nmm`, iron's in FeSe; `2e`, `2f` of `P4_2/mmc`; `2a`, `2b` of `P4_2/nmc`; `4d` of `I4/mmm`, iron's in BaFe2As2; `4a`, `4b` of `I4_1/amd`, titanium's in anatase; `6c`, `6d` of `Pm-3n`; `24c` of `Fm-3c`; `12d` of `Im-3m` | `B2`, `A2` and `B1` are Mulliken's `A2`, `B1` and `B2`, so, with `x` and `y` normal to the site's mirrors, `pz` and `dx2-y2` are the tables' `B1` and `dxy` their `A2` |
| `6` | `1a` of `P6`; `2a` of `P6cc`; `2e` of `P6/m`; `2e` of `P622`; `4e` of `P6/mcc` | `E1` and `E2` are exchanged |
| `6/m` | `1a`, `1b` of `P6/m`; `2b` of `P6/mcc` | `E1g` and `E2g` are exchanged, and `E1u` and `E2u` |

A `-6m2` site written `-62m`, and a `-42m` site written `-42m`, follow
Mulliken's convention. A declaration is read in the tables' names. So at the
sites above, a name as Mulliken's convention spells it declares the other
irrep of the exchanged pair. Where a declaration uses such a name, the
atomic-limit table prints a note under it saying which row Mulliken's
convention calls it. The example is graphene read free-standing, its first
orbital declared `A1''`:

```python
declared = tqc.realized_group(graphene(0.0, -1.0), environment="free-standing",
                              characters={0: "A1''"})
print(declared.wyckoff_orbits.summary())
#  Wyckoff  site symmetry  orbitals  label  determined by  declared
#       2c         -6m21'       0-1   A1''    declaration    A1''/-
#    orbital 0: declared "A1''", read at the -6m2 in P6/mmm1' #191.234; the tables' A1'' at this site is the row Mulliken's convention calls A2'; returned as A1'' of -6m2   (.declared)
```

- `A1''/-`: what `characters=` declared, orbital by orbital, `-` for an
  orbital not declared.
- `the tables' A1'' at this site is the row Mulliken's convention calls A2'`:
  the note. The row returned is the tables' `A1''`, the one declared.

At a site whose group is the threefold axis `3` alone, the tables call the
trivial irrep `A1` where a point-group table prints `A`. So `{0: "A1"}`
declares it at such a site, and `"A"` is refused with `unknown_label`. Two
ways of declaring do not depend on the tables' names: `"s"`, which names the
trivial irrep of any site group, and characters given on the generators, which
use no name ([How do I say what my orbitals are?](../guide/orbitals.md)).

A leading `-` marks a double-valued irrep, carried by states with spin. `1E2E`
is two complex-conjugate one-dimensional irreps that time reversal joins into
one site irrep, and `-1E2E` is its double-valued counterpart. Several site
irreps at one site are written as a sum, a multiplicity above one in front:
`A1 + E`, or `2x-E1 + -1E2E` for a spin-orbit `p` shell at a `3m` site.

## Little-group representations at a k-point

The bands at a high-symmetry point carry little-group representations. In a
group without time reversal these are the irreps of the point's little group.
In a group with time reversal they are its coreps, the irreducible
co-representations of a group that holds antiunitary operations
([Little groups, irreps and coreps](tqc-primer.md#little-groups-irreps-and-coreps)).
A label is the name of one. The example is Kane-Mele's two lowest bands:

```python
hsp = tqc.high_symmetry_points(r.group)
for point in ("GM", "K", "M"):
    print(tqc.coreps_at(r, hsp[point], bands=2))
#  CorepsAt(GM: -GM6)
#  CorepsAt(K: -K5 + -K6)
#  CorepsAt(M: -M3M4)
```

- `-GM6`: a label, the point's name and an index. `GM` is the tables' name for
  Γ. A leading `-` marks a double-valued corep.
- `-K5 + -K6`: two one-dimensional coreps, joined by ` + `.

Where an antiunitary operation fixes the point, each of Wigner's three cases is
written its own way. An irrep that is a corep alone keeps its name (`-GM6`).
Two different irreps glued into one corep are written together (`-M3M4`, of
the unitary irreps `-M3` and `-M4`). One irrep doubled is written twice
(`-P4P4`, in
[Points where the gluing differs](#points-where-the-gluing-differs)). The
constituents stand in the lexicographic order of their bare names, the minus
written once in front. The field `coreps` of what `coreps_at` returns maps each
label to its multiplicity
([`CorepsAt`](../reference/records.md#corepsat)).

## The origin the names are read about

The tables write a group about one origin. A group can have several points the
tables could be read about equally well: every point about which each
operation of the group is written with the same Seitz symbol. In `P4mm` the
fourfold axes through `(0, 0)` and `(1/2, 1/2)` are two such points. Between
them the Wyckoff positions `1a` and `1b` exchange names. So do the irreps at
`X` among themselves, and those at `M` among themselves. The names returned
are read about the point nearest the model's own origin. So an orbital at the
model's `[0, 0]` is named at the tables' origin wherever `[0, 0]` is one of
these points. Under a declared precision the centers are first made exact, and
a shift of all of them together within the precision is removed
([How do I measure a numerical or Wannier90 model?](../guide/numerical.md)).

The example is the Lieb lattice, its corner orbital at the origin, beside the
same lattice with every center shifted by a few millionths of a cell, as a
fitted model's can be, read with a precision declared:

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
    print(rl.wyckoff_orbits[0].wyckoff, tqc.hsp_table(rl, bands=1).symmetry_data)
#  1a {'GM1': 1, 'X1': 1}
#  1a {'GM1': 1, 'X1': 1}
```

- `1a`: the corner orbital's Wyckoff position, in both models.
- `{'GM1': 1, 'X1': 1}`: the corep multiplicities of the lowest band, the same
  in both. The shifted lattice is named as the exact one is. `M` is left out:
  there the lowest band is degenerate with the next one.

Where two or more such points are equally near the model's origin, the names
are read about one of them, the same in every run. pythTQC asks the `irrep`
package for the group, and keeps the point that package chooses where it is
among the nearest
([Two equally near origins](limits.md#two-equally-near-origins)).

## The band-representation key

An EBR is a band representation that is not equivalent to a sum of other band
representations. Each is induced from one site irrep at a Wyckoff position
([Band representations and atomic limits](tqc-primer.md#band-representations-and-atomic-limits)).
The records hold each EBR as a `Term`, its parts as fields, and write a sum of
EBRs on one line as its `label`. A decomposition's candidate sums and a
Wyckoff orbit's band representation are written alike. The examples are the
sum of EBRs whose symmetry data vector is that of Kane-Mele's two lowest
bands, graphene's orbit, and the EBRs of graphene's group induced from the
site irreps `A1'` and `A1''`:

```python
d = tqc.decompose(tqc.hsp_table(r, bands=2))
print([s.label for s in d.candidate_sums])
#  ["-E1 at 1a (3m1')"]
print(gr.wyckoff_orbits[0].band_representation.terms)
#  (Term(multiplicity=1, irrep="A1'", wyckoff='2c', site_symmetry="-6m21'", unitary_site_symmetry='-6m2'),)
print([key for key in tqc.ebr_rows(gr.group) if key[0].startswith("A1'")])
#  [("A1'", '2c'), ("A1''", '2c'), ("A1'", '2d'), ("A1''", '2d')]
```

- `-E1 at 1a (3m1')`: the sum's `label`, the site irrep at its Wyckoff
  position, the site-symmetry group in parentheses. A multiplicity above one
  stands in front, `2x-E1 at 1a (3m1')`.
- `multiplicity=1`: how many times this EBR enters the sum.
- `irrep="A1'"`: the site irrep, here single-valued. Kane-Mele's `-E1` is
  double-valued.
- `wyckoff='2c'`: the Wyckoff position, multiplicity and letter, in the
  tables' setting. The multiplicity counts the sites in the conventional cell,
  which on a centered lattice holds more than one primitive cell: `3a` of
  `R-3m` is one site per primitive cell.
- `site_symmetry="-6m21'"` and `unitary_site_symmetry='-6m2'`: the
  site-symmetry group with time reversal, `1'` adjoined, and its unitary part.
  In a group without time reversal the two are one group.
- `("A1'", '2c')`: a key of `ebr_rows`, one EBR as its site irrep and its
  Wyckoff position.

An EBR makes as many bands as the dimension of its site irrep times the number
of sites of its orbit in a primitive cell: 2 times 1 for Kane-Mele's `-E1` at
`1a`, and 1 times 2 for graphene's `A1'` at `2c`. The records of the stable
real-space invariants write an EBR as the invariants' formulas do: `1 (-E1)_a`
is one copy of `-E1` at the Wyckoff position of letter `a`.

The EBR files the tables ship write a site group's symbol without its bar:
`6m2` for the `-6m2` at `2c` of `P6/mmm`, `3m` for the `-3m` at `3a` of
`R-3m`. Every summary and every term returned writes the bar: graphene's
position reads `2c (-6m21')`.

## Reading across to BCS

A Seitz symbol returned is the one the server prints for that operation of the
general position, or the same operation written about the opposite direction
of its axis: `3+_111` where a table may print `3-_-1-1-1`, which
`operation_index` reads either way. The site irreps and the little-group
representations carry the server's names. So most of what pythTQC prints reads
straight across.

One convention differs. The server's tables represent a lattice translation
`t` by `e^{+ik.t}`. Bloch functions transformed as `(g f)(r) = f(g^-1 r)`
carry `e^{-ik.t}`, which is what pythtb's Bloch Hamiltonian realizes and a
sewing matrix measures. Every label and EBR row returned is in this
convention. Since `e^{+ik.t} = e^{-i(-k).t}`, **the labels the server ships at
`k` are the labels of the physical irreps at `-k`.** At a
time-reversal-invariant momentum (TRIM), where `k` and `-k` are one point,
nothing changes. Wyckoff positions and site irreps are named the same here and
on the server.

### Labels away from the time-reversal-invariant points

At a point that is not a TRIM, an EBR row pairs labels with Wyckoff positions
differently here and on the server's BANDREP pages. The example is Kane-Mele
with `delta = 50`, which puts its two lowest bands on orbital 1, at `1b`,
whose on-site energy is `-50`:

```python
deep = tqc.realized_group(kane_mele(50.0, -1.0, 0.3, 0.25), typat=[0, 1], characters={0: "A1"})
low = tqc.hsp_table(deep, bands=2)
print(low.symmetry_data, [s.label for s in tqc.decompose(low).candidate_sums])
#  {'-GM6': 1, '-K4': 1, '-K5': 1, '-M3M4': 1} ["-E1 at 1b (3m1')"]
convention = tqc.ebr_data(deep.group).convention
print(convention.name, convention.changed_points, convention.conjugation["-K5"])
#  physical (translations as e^{-ik.t}) ('H', 'K') -K6
```

- `{'-GM6': 1, '-K4': 1, '-K5': 1, '-M3M4': 1}`: the symmetry data vector of
  the two lowest bands.
- `["-E1 at 1b (3m1')"]`: the one sum of EBRs with that symmetry data vector,
  `-E1` at `1b`, the site of orbital 1.
- `physical (translations as e^{-ik.t})`: `convention.name`, the convention
  every row returned is in.
- `('H', 'K')`: `convention.changed_points`, the points at which a row's labels
  differ from the server's.
- `-K6`: `convention.conjugation["-K5"]`, the partner of `-K5` under complex
  conjugation.

On the server's page the `-E1` row at `1b` carries `-K6` where the row
returned here carries `-K5`. Read against the server's page as printed, the
symmetry data vector of these bands matches the `-E1` row at `1c`, the other
site. So, to compare with the server:

1. At a TRIM, read the labels as they are.
2. At a point of `changed_points`, exchange each label for its partner under
   `conjugation`.

The Wyckoff positions then agree with those returned here.

In a group without time reversal, the server's band representations also hold
rows at the partner `-k` of some points: `KA` beside `K`, and likewise `HA`,
`PA` and `WA`. The server's character tables do not list their irreps. These
are the complex conjugates of the listed point's irreps, `KAn` of `Kn`. The
rule above holds at them as at every other point.

### Points where the gluing differs

At a point that is not a TRIM, time reversal alone does not fix the point. The
antiunitary operations that do fix it decide which irreps form one corep. They
do so through a sum over those operations, and the phases in that sum depend
on the convention for a translation. At a few points the two conventions glue
differently, so that a corep can change its dimension. These are the re-glued
points. `P` of `I4_1` is one: `(1/4, 1/4, 1/4)` in the reduced coordinates of
the primitive cell, and `(1/2, 1/2, 1/2)` in those of the conventional cell.
The antiunitary operations that fix it are the fourfold screws combined with
time reversal, `C4(z)'` and `C4^-1(z)'`. There the four lowest states of the
spin-orbit crystal split `[1, 2, 1]`:

```python
P = tqc.high_symmetry_points(soc.group)["P"]
print(P)
#  [0.25 0.25 0.25]
states = tqc.solve_at(soc.model, P)
print(np.round(states[0][:4], 6))
#  [-2.744031 -2.501964 -2.501964 -2.259907]
for multiplet in ([0], [1, 2], [3]):
    print(multiplet, tqc.coreps_at(soc, P, states=states, bands=multiplet).coreps)
#  [0] {'-P3': 1}
#  [1, 2] {'-P4P4': 1}
#  [3] {'-P3': 1}
for multiplet in ([0], [1, 2]):
    rows = tqc.characters(soc, P, states=states, bands=multiplet).multiplets
    print(multiplet, np.round([list(m.characters.values()) for m in rows], 6) + 0)
#  [0] [[1.+0.j 0.-1.j]]
#  [1, 2] [[2.+0.j 0.+2.j]]
lg = tqc.little_group(soc.group, P)                # the operations that fix P
print([lg.operations[i].name for i in lg.unitary],
      [lg.operations[i].name for i in lg.antiunitary])
#  ['E', 'C2(z)'] ["C4(z)'", "C4^-1(z)'"]
for label, row in tqc.corep_table(soc.group, P).items():
    print(label, np.round(list(row.values()), 6) + 0)
#  -P4P4 [2.+0.j 0.+2.j]
#  -P3 [1.+0.j 0.-1.j]
convention = tqc.ebr_data(soc.group).convention
print(convention.conjugated_points, convention.divergent_points, convention.reglued_points,
      convention.changed_labels)
#  ('M', 'P') ('P',) ('P',) ('-P3P3', '-P4')
```

- `[0.25 0.25 0.25]`: `P`, in the model's reduced coordinates.
- `[-2.744031 -2.501964 -2.501964 -2.259907]`: the four lowest energies, in
  the units of H: one state, a doublet, and one state.
- `{'-P3': 1}` and `{'-P4P4': 1}`: the corep each multiplet carries.
- `[[1.+0.j 0.-1.j]]`: the characters of the lowest state on `E` and `C2(z)`,
  `(1, -i)`, those of the unitary irrep `-P3`.
- `[[2.+0.j 0.+2.j]]`: the characters of the doublet, twice `(1, +i)`, those
  of the unitary irrep `-P4`.
- `['E', 'C2(z)'] ["C4(z)'", "C4^-1(z)'"]`: the operations of the little group
  of `P`, `unitary` and `antiunitary`.
- `-P4P4 [2.+0.j 0.+2.j]` and `-P3 [1.+0.j 0.-1.j]`: `corep_table`, each corep
  of the little group with its characters by operation index. Its two rows are
  the characters measured above.
- `('M', 'P')`: `conjugated_points`, the points at which the characters
  returned are the complex conjugates of the server's. At such a point that is
  not divergent, each conjugated row equals one of the server's rows and takes
  its name. There the table returned is the server's own, and only the label a
  band lands on can differ.
- `('P',)`, the first: `divergent_points`, the points at which the server's
  table is that of `-k`, and is not a reordering of the table returned.
- `('P',)`, the second: `reglued_points`, those of the divergent points at
  which time reversal glues the irreps into coreps differently from the
  server's tables.
- `('-P3P3', '-P4')`: `changed_labels`, the labels that differ: here the two
  the server carries at `P` and the table returned does not.

The antiunitary operations leave `-P3` a corep alone and double `-P4`, so the
doublet is `-P4P4`. The server's table doubles `-P3` instead. The server's
table at `P` is the complex conjugate of the one returned: it is the table of
physical irreps at `-P`.
[`examples/tqc_sg80_soc_doublet.py`](../../examples/tqc_sg80_soc_doublet.py)
runs this measurement whole.

`ebr_data(group).convention` holds the fields read above. Its
`row_permutation` re-keys a linear form written in the server's labels, such
as an indicator formula copied off the server, so that it can be evaluated on
a returned symmetry data vector
([How a symmetry indicator is computed](tqc-primer.md#how-a-symmetry-indicator-is-computed)).

## Spin

The double-valued labels are read with one fixed choice, for each operation, of
the two spin matrices that represent it. The spin is quantized along `+z`, and
every two-by-two spin block is in the order up, down. A rotation by θ about the
unit vector `n` acts on the spinor as `e^{-iθ n·σ/2}`. An improper operation
acts on the spin as its proper part does, so a spinful `s` orbital at an
inversion center is even under the inversion. Time reversal acts as `iσ_y`
times complex conjugation.
The tables write the spin matrices in the standard orientation of the
conventional cell. Every spin block returned is read in the Cartesian axes in
which the pythtb Lattice is specified, a right-handed frame. pythtb writes
spinful models in this convention, so a pythtb model needs nothing. A model
transcribed from a source whose spin is quantized along `-z` needs converting.
[How does pythTQC read the spin of my model?](../guide/spin.md) prints those
spin matrices for a model's own operations, converts a model written for the
other axis, and shows what a wrong frame does to the group.
