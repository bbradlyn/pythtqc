# Fu–Kane–Mele: a 3D crystal, a distortion, the ℤ₄ indicator

This page is the second hour, after [the quick start](quickstart.md): one
three-dimensional crystal from its space group to its symmetry-indicator class.
It reads the group and the atomic-limit table of an undistorted crystal, finds
that crystal a semimetal, lowers its symmetry with a bond distortion and reads
what broke, and measures the ℤ₄ indicator of the three phases the distortion
makes, against Fu and Kane's count of parities.

The model is Fu, Kane and Mele's spin–orbit model on the diamond lattice
([Phys. Rev. Lett. 98, 106803 (2007)](https://doi.org/10.1103/PhysRevLett.98.106803)),
from pythtb's own collection. `fu_kane_mele(t, soc, dt)` puts one orbital with
spin on each of the two sites of the primitive cell, an `s` orbital in the
paper — four bands, the lower two occupied — with a nearest-neighbour hopping
`t`, a spin–orbit coupling `soc` between second neighbours, and `dt`, four
offsets to the four nearest-neighbour bonds; the first is the bond inside the
cell, which points along `[111]`. Energies are in units of `t`. The blocks run
in order, as written, and the lines under a statement are what it prints. The
whole page runs in about a minute, most of it the first call, which measures
every operation of the diamond structure.

## The undistorted crystal

```python
import itertools

import numpy as np
from pythtb.models import fu_kane_mele
import pythtqc as tqc

fkm = fu_kane_mele(1.0, 0.125, [0.0, 0.0, 0.0, 0.0])      # every bond equal
r = tqc.realized_group(fkm, characters={0: "s"})         # orbital 0 is an s orbital
print(repr(r))
#  RealizedGroup(Fd-3m1' #227.129)
print(r.rep.characters.summary())
#  orbital  Wyckoff  site symmetry  function  irrep  named by  determined by  orbital characters
#        0       8a           -43m         s    -E1    tables    declaration  E=+1, C2(x)=+1, C3([111])=+1, m([001])=+1, S4(x)=+1
#        1       8a           -43m              -E1    tables    hamiltonian  E=+1, C2(x)=+1, C3([111])=+1, m([001])=+1, S4(x)=+1
```

`realized_group` measures which operations `H(k)` obeys and returns the group
they form. `Fd-3m1'` is diamond's space group, `Fd-3m` (number 227), with time
reversal adjoined: the trailing `1'` marks a **grey group**, and the irreducible
objects of a group with antiunitary operations are **coreps**
(co-representations), which is what every band label on this page is.
`227.129` is the group's number among the magnetic space groups. Nothing more
is printed: the Hamiltonian obeys every operation the orbital positions allow,
and where it obeys fewer, the record names the group it descended from, as it
does below. A three-dimensional model needs no preparation: its own positions
are the cell the group is identified on.

The table is the model's **atomic-limit table**, one row per orbital. Both
orbitals sit at Wyckoff `8a`, the two atoms of the diamond structure, whose site
symmetry is the tetrahedral group `-43m`. A pythtb orbital carries no symmetry
character of its own: the hoppings fix how the orbitals transform relative to
one another, not what they are. So orbital 0 is `s` because `characters=` said
so (`determined by: declaration`), and orbital 1 is fixed relative to it by the
hoppings (`hamiltonian`). The `irrep` column is the site corep the orbital
carries with its spin, `-E1`, where the leading `-` marks a double-valued
label; the `orbital characters` are the orbital's own, without its spin, on one
operation of each class of the site group, all `+1` for an `s` orbital, and
`named by: tables` says the tables of band representations name that corep at
`8a`. Every column is in
[the records reference](../reference/records.md#the-atomic-limit-table).

## A semimetal at X

Solve the model at the high-symmetry points of its group:

```python
hsp = tqc.high_symmetry_points(r.group)                  # in the model's reduced coordinates
for name in sorted(hsp):
    energies = np.asarray(tqc.solve_at(fkm, hsp[name])[0], dtype=float).reshape(-1)
    print(f"{name:<3}", " ".join(f"{e:+.3f}" for e in np.round(energies, 3) + 0.0))
#  GM  -4.000 -4.000 +4.000 +4.000
#  L   -2.000 -2.000 +2.000 +2.000
#  W   -1.000 -1.000 +1.000 +1.000
#  X   +0.000 +0.000 +0.000 +0.000
```

Every level is a Kramers pair, since the crystal has both inversion and time
reversal, and at `X` all four bands meet at zero energy. The undistorted crystal
is a semimetal: its two lower bands are not separated from the upper two, so at
`X` there is no set of valence bands to label. The package says so:

```python
all4 = tqc.hsp_table(r.rep, bands=4)
print(tqc.format_counts(all4.counts))
#  -GM6 + -GM9 + -L8 + -L9 + -W5W6 + -W7 + -X5
print(tqc.decompose(all4).solution_strings())
#  ("1 x -E1↑G(4)@8a(-43m1',-43m)",)

pair = tqc.hsp_table(r.rep, bands=2)                     # the two lowest bands alone
print(pair)
#  HSPTable(bands=(0, 1), 3 points resolved, 1 refused, 0 skipped, group=realized)
print(pair.refusals["X"].gate, pair.refusals["X"].context["multiplet"])
#  multiplet_split (0, 1, 2, 3)
refused = tqc.symmetry_indicator(pair)
print(bool(refused), refused.gate)
#  False incomplete_content
```

All four bands together are labelled at every high-symmetry point, and their
content is one **elementary band representation** (EBR) — the bands one site
irrep at one Wyckoff position induces, when they are no sum of others.
`-E1↑G(4)@8a(-43m1',-43m)` reads: the site corep `-E1`, induced (`↑`) to the
whole group `G` in four bands, from Wyckoff `8a`, whose site group is `-43m1'`
with time reversal and `-43m` without — the two spinful `s` orbitals
themselves, as it must be
([the conventions](../background/conventions.md#the-band-representation-key)
spell out every part of this key).

The two lowest bands alone resolve three points and refuse one. At `X` they are
half of the fourfold multiplet `(0, 1, 2, 3)`, and a character read on part of a
degenerate multiplet depends on the basis the diagonalization chose inside it,
so the table refuses that point with the check `multiplet_split`. A **refusal**
is a returned, falsy record that names the check that stopped the call and what
it measured; `symmetry_indicator`, handed a table with a point missing, refuses
`incomplete_content`, because the class of a vector with a point missing is the
class of a different vector. Both are statements about the physics: the
valence pair has no label at `X` because the crystal has no valence pair there.
[When a call refuses](../guide/refusals.md) says how to read one, and the
refusals reference has both checks,
[`multiplet_split`](../reference/refusals.md#multiplet_split) and
[`incomplete_content`](../reference/refusals.md#incomplete_content).

## A distortion lowers the group

Strengthen the `[111]` bond by `0.4`. No atom moves, so the orbital positions
still allow all of `Fd-3m`; the Hamiltonian does not.

```python
strong = fu_kane_mele(1.0, 0.125, [0.4, 0.0, 0.0, 0.0])  # the [111] bond strengthened
rs = tqc.realized_group(strong, characters={0: "s"})
print(repr(rs))
#  RealizedGroup(R-3m1' #166.98; descended from #227.129 (72 of 96 coset representatives of the lattice translations broken; these are the single-valued group's, each standing for itself and its negative in the double group))
d = rs.descent
print(d["from"]["name"], "->", d["to"]["name"], d["dropped_by"],
      sorted({b["refuted_by"] for b in d["dropped"]}))
#  Fd-3m1' -> R-3m1' {'hoppings': 72} ['no symmetry matrix']
for op in rs.operations.values():
    if not op.seitz.endswith("'"):                       # those without time reversal
        print(f"{op.ind:>2}  {op.seitz:<16}  {op.name}")
#   1  {1|0,0,0}         E
#   2  {3+_001|0,0,0}    C3([111])
#   3  {3-_001|0,0,0}    C3^-1([111])
#   4  {2_110|0,0,0}     C2([01-1])
#   5  {2_100|0,0,0}     C2([10-1])
#   6  {2_010|0,0,0}     C2([1-10])
#   7  {-1|0,0,0}        i
#   8  {-3+_001|0,0,0}   S3([111])
#   9  {-3-_001|0,0,0}   S3^-1([111])
#  10  {m_110|0,0,0}     m([01-1])
#  11  {m_100|0,0,0}     m([10-1])
#  12  {m_010|0,0,0}     m([1-10])
for row in rs.rep.characters.orbitals:
    print(row.orbital, row.wyckoff, row.site_symmetry, row.irrep, row.determined_by)
#  0 6c 3m -E1 declaration
#  1 6c 3m -E1 hamiltonian
```

Read the record clause by clause. `R-3m1' #166.98` is the group returned: the
rhombohedral grey group whose threefold axis is the strengthened bond.
`descended from #227.129` is how it was reached: the positions offered
diamond's group, the Hamiltonian broke 72 of its 96 coset representatives, and
the 24 that hold were identified again as a group. Nothing downstream is ever
read in a group the model does not have.

`rs.descent` is that measurement in full: the group before and after, the
count by cause, and one entry per operation dropped, each named as below.
Every operation here was refuted by `"no symmetry matrix"`: no matrix with the
operation's action on `k` carries `H(k)` onto `H(Rk)`, which is a proof.
[The records reference](../reference/records.md#the-descent) lists the other
ways an operation is refuted, and
[the group chapter](../guide/group.md#when-the-hamiltonian-has-less-symmetry-than-its-cell)
works through a descent.

The operations kept without time reversal are the point group `-3m` about the
bond: the threefold about `[111]`, three twofolds perpendicular to
it, the inversion `i` about the bond's centre, the two threefold
rotoinversions (`S3` in the name column) and three mirrors containing the
axis. Each row gives the Seitz symbol in the conventional setting the tables
use, whose `c` axis is the threefold, and the name in the model's own frame,
the one `characters=` is written in: `x`, `y`, `z` for a Cartesian axis, and
otherwise the direction in lattice coordinates, so `C3([111])` turns about the
sum of the three lattice vectors, the bond
([the conventions](../background/conventions.md#operations-a-seitz-symbol-and-a-cartesian-name)
read both names in full).

The orbitals move with the group. In `Fd-3m` they were the orbit `8a`, with
site symmetry `-43m`; in `R-3m` they are `6c`, whose site symmetry is the
threefold about the bond and the three mirrors that contain it, `3m`. The site
corep is still `-E1`, now of `3m`, and the one declaration still fixes both rows.

## Three phases, and the ℤ₄ indicator

The same bond weakened by `0.4` is a second phase, and the other three bonds cut
to zero is a third: every atom keeps one nearest neighbour, its partner along
`[111]`, and the dimers they form are coupled to one another only by the
spin–orbit term. All three have the group `R-3m1'`. The **symmetry indicator**
asks which class the valence bands' content occupies in the group's band
structures modulo its atomic limits, `X_BS = {BS}/{AI}`, a finite abelian group
fixed by the space group alone; a nonzero class rules out every atomic limit
for every band set with that content. How the class is computed from the EBRs
is in
[the TQC primer](../background/tqc-primer.md#how-a-symmetry-indicator-is-computed).

```python
phases = {
    "strong": rs,
    "weak": tqc.realized_group(fu_kane_mele(1.0, 0.125, [-0.4, 0.0, 0.0, 0.0]),
                               characters={0: "s"}),
    "dimers": tqc.realized_group(fu_kane_mele(1.0, 0.125, [0.0, -1.0, -1.0, -1.0]),
                                 characters={0: "s"}),
}
tables = {name: tqc.hsp_table(rr.rep, bands=2) for name, rr in phases.items()}
indicators = {name: tqc.symmetry_indicator(tables[name])
              for name, rr in phases.items()}
print(f"X_BS of #{rs.group.number_str} {rs.group.name}: {indicators['strong'].structure}")
#  X_BS of #166.98 R-3m1': Z2 x Z4
print("phase   group   valence content              class   order  z4  z2w")
#  phase   group   valence content              class   order  z4  z2w
for name, rr in phases.items():
    si = indicators[name]
    z2w = "".join(str(si.canonical[f"z2w,{i}"]) for i in (1, 2, 3))
    print(f"{name:<7} {rr.group.number_str:<7} {tqc.format_counts(tables[name].counts):<28} "
          f"{str(si.values):<7} {si.order:<6} {si.canonical['z4']:<3} {z2w}")
#  strong  166.98  -F5F6 + -GM9 + -L5L6 + -T8   (0, 1)  4      3   111
#  weak    166.98  -F3F4 + -GM9 + -L5L6 + -T8   (1, 0)  2      0   111
#  dimers  166.98  -F5F6 + -GM9 + -L5L6 + -T9   (0, 0)  1      0   000
```

Each valence pair is now an isolated set, labelled at all four high-symmetry
points of `R-3m1'`: `GM`, `T` on the threefold axis, and `F` and `L`. For this
group `X_BS = ℤ₂ × ℤ₄`, and `class` is each phase's coordinates in it, in the
Smith basis the computation uses; `order` is the order of that class, which no
choice of basis changes. The last two columns speak the same class in named
invariants ([the names a class is spoken in](../guide/indicators.md#the-names-a-class-is-spoken-in)):
`z4`, the strong index, and `z2w`, the three weak ones,
`z2w,1 z2w,2 z2w,3`. The strong phase has a class of order 4 with `z4 = 3`; the
weak phase a class of order 2, `z4 = 0` and every weak index 1; the dimers
the zero class. The contents differ only at `T` and `F`.

In a crystal with inversion, Fu and Kane read the same invariants off parities
([Phys. Rev. B 76, 045302 (2007)](https://doi.org/10.1103/PhysRevB.76.045302)).
At each of the eight **time-reversal-invariant momenta** (TRIM), the momenta
`k ≡ −k`, the occupied Kramers pair has one inversion parity `ξ = ±1`; the
strong index is `(−1)^ν0 = ∏ ξ` over all eight, and the weak index `ν_i` is the
same product over the four with `k_i = ½`. The parities are measured here from
the model's own inversion, the one about the bond's centre, `{-1|0,0,0}` in the
tables' setting: its **sewing matrix** at a TRIM, its matrix on the valence
pair there, has the trace `2ξ`.

```python
trim = [np.array(n) / 2 for n in itertools.product((0, 1), repeat=3)]   # the eight TRIM
for name, rr in phases.items():
    inversion = tqc.operation_index(rr.rep, "{-1|0,0,0}")
    parity = [round(tqc.sewing_matrix(rr.rep, inversion, k, bands=2).trace.real / 2)
              for k in trim]                             # +1 or -1 for the one Kramers pair
    nu0 = int(np.prod(parity) < 0)
    weak = [int(np.prod([p for p, k in zip(parity, trim) if k[i] == 0.5]) < 0)
            for i in range(3)]
    odd = parity.count(-1)
    print(f"{name:<7} {' '.join('+' if p > 0 else '-' for p in parity)}   "
          f"({nu0};{weak[0]}{weak[1]}{weak[2]})   {odd} odd pairs, {odd} mod 4 = {odd % 4}")
#  strong  - - - - - - - +   (1;111)   7 odd pairs, 7 mod 4 = 3
#  weak    - - - + - + + +   (0;111)   4 odd pairs, 4 mod 4 = 0
#  dimers  - - - - - - - -   (0;000)   8 odd pairs, 8 mod 4 = 0
print(tables["strong"].note)
#  the labels of these 2 of the 4 bands rest on a one-dimensional character the hoppings leave open on the coupled part holding every orbital: another choice of it keeps every orbital's site irrep and the labels of all 4 bands, and labels these 2 differently; the lowest-angular-momentum rule chose the one whose orbitals read as the lowest-angular-momentum functions carrying their site content
```

The eight momenta are listed in reduced coordinates as `itertools.product`
orders them, from `(0, 0, 0)`, which is `GM`, to `(½, ½, ½)`, which is `T`; the
three with two halves are the three `F` points, and the three with one half the
three `L` points. The dimers, an atomic limit, are odd at every TRIM. The
strong phase differs from them at `T` alone, one parity inverted: `(1;111)`, a
strong topological insulator. The weak phase differs at `T` and the three `F`,
four inverted: `ν0 = 0` with every weak index 1, `(0;111)`, a weak one. These
are the indices Fu, Kane and Mele give for the two signs of the distortion,
measured here from the hoppings alone. And `z4` is the number of odd pairs
modulo 4 — 3, 0 and 0, the column above — so `ν0 = z4 mod 2`, and the weak
indices are the `z2w` column.

Two choices stand behind these parities, and behind `z4`. They are taken about
the bond's centre: about an inversion centre half a lattice vector away, the
parities at four of the eight momenta flip, which can move the count by 2. And
they are the parities of `s` orbitals. The last line, the note under the strong
phase's table, says that its labels rest on a one-dimensional character the
hoppings leave open: the site irreps leave two labellings of the representation
open, and the one returned reads both orbitals as `s` functions, which the
inversion carries onto each other with `+1`. The other differs from it by a
sign on the inversion, the twofold rotations and the rotoinversions, a
one-dimensional character the orbitals' site group `3m` cannot see, so it is
open in every phase; in it every
parity flips, `ν0` and the weak indices stay, and `z4` changes sign, the strong
phase's seven odd pairs becoming one. A formula that counts even pairs against
odd ones, `½ Σ (even − odd)`, carries that same sign: it gives the strong phase
`½ (1 − 7) = −3`, that is `z4 = 1`.

## What a decomposition says

`decompose` asks the content a narrower question than the indicator: whether a
sum of EBRs with nonnegative integer multiplicities reproduces this one vector.

```python
for name, rr in phases.items():
    dec = tqc.decompose(tables[name])
    signed = (f"{dec.signed_combinations['n_solutions']} signed" if dec.signed_combinations
              else "-")
    found = f"{len(dec.solutions)} solution{'' if len(dec.solutions) == 1 else 's'}"
    print(f"{name:<7} {dec.verdict:<32} {dec.certificate or '-':<6} {signed:<11} {found}"
          + "".join(f": {s}" for s in dec.solution_strings()))
#  strong  none                             proof  0 signed    0 solutions
#  weak    none                             proof  0 signed    0 solutions
#  dimers  symmetry-indicated decomposable  -      -           1 solution: 1 x -E1u↑G(2)@3a(-3m1',-3m)
centre = tqc.wyckoff_sites_tables(phases["dimers"].group, [[0.125, 0.125, 0.125]])
print("the centre of the [111] bond:", [f"{s.multiplicity}{s.letter}" for s in centre])
#  the centre of the [111] bond: ['3a']
```

The strong phase reads `none` with the certificate `proof`: the multiplicities
are bounded rigorously, the search is exhaustive, and no sum of EBRs has this
content, so the valence pair is not an atomic limit. The signed combinations ask
the same question with negative multiplicities allowed, and none lies within the
bound, as a nonzero class requires: content in a nonzero class is no integer
combination of EBRs at all. A fragile band set, by contrast, has an integer
combination with a negative coefficient
([when there is no decomposition](../guide/atomic-limits.md#when-there-is-no-decomposition)).
The indicator says more than the verdict does. Both say that no band set with
this content is an atomic limit; the nonzero class says it of the content with
any atomic bands added as well, since an atomic limit's class is zero and
classes add, while a `none` alone can give way to a decomposition once suitable
atomic bands are added, as a fragile content's does. The weak phase reads `none`
as well, and only its class, of order 2 where the strong phase has order 4,
tells the two apart.

The dimers decompose, in one way: the valence pair has the content of the site
corep `-E1u` induced from Wyckoff `3a`, and the last line shows that `3a` is
the centre of the bond the dimers keep, `(⅛, ⅛, ⅛)` in the model's reduced
coordinates. The `u` marks a corep odd under the inversion about that centre:
the combination of each dimer's two `s` orbitals odd under the inversion that
exchanges them, with its spin. Their class is zero, as an atomic limit's must
be; a zero class on its own certifies nothing, since fragile topology lives in
the indicator's kernel
([what a zero does not certify](../guide/indicators.md#what-a-zero-does-not-certify)).

## Where to go next

- The same three phases carry stable real-space invariants:
  [the SRSI chapter](../guide/srsi.md) starts from what a real-space invariant
  is, and reads these phases' [from their bands](../guide/srsi.md#from-the-bands).
- [What group does my Hamiltonian have?](../guide/group.md) takes the descent
  further, and [labelling the bands](../guide/labels.md) the tables used here.
- [`examples/tqc_fkm_diamond.py`](../../examples/tqc_fkm_diamond.py) measures
  the same phases and draws their bands with the valence labels at each
  high-symmetry point ([the examples](../examples.md)).
- Every call on this page, with its arguments and its refusals, is in the
  reference: [`realized_group`](../reference/functions.md#realized_group),
  [`high_symmetry_points`](../reference/functions.md#high_symmetry_points),
  [`solve_at`](../reference/functions.md#solve_at),
  [`hsp_table`](../reference/functions.md#hsp_table),
  [`format_counts`](../reference/functions.md#format_counts),
  [`decompose`](../reference/functions.md#decompose),
  [`symmetry_indicator`](../reference/functions.md#symmetry_indicator),
  [`operation_index`](../reference/functions.md#operation_index),
  [`sewing_matrix`](../reference/functions.md#sewing_matrix) and
  [`wyckoff_sites_tables`](../reference/functions.md#wyckoff_sites_tables).
