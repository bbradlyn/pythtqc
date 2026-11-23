# Fu-Kane-Mele: a 3D crystal, a distortion, the Z4 indicator

This page shows how to read one three-dimensional crystal, from its space group
to its symmetry-indicator class. The crystal is Fu, Kane and Mele's spin-orbit
model on the diamond lattice, first undistorted and then with one bond
distorted. `realized_group` measures which operations the Hamiltonian obeys
and returns the group they form. `hsp_table` returns the little-group
representations the bands carry at the high-symmetry points of that group.
`symmetry_indicator` returns the symmetry-indicator class of the two lowest
bands. `decompose` returns the decompositions of their symmetry data vector
into elementary band representations. The page then compares the class with
the inversion parities of Fu and Kane. Read
[Quick start: Kane-Mele, end to end](quickstart.md) first.

The model is from pythtb's own collection: Fu, Kane and Mele,
[*Topological insulators in three dimensions*, Phys. Rev. Lett. **98**, 106803 (2007)](https://doi.org/10.1103/PhysRevLett.98.106803).
`fu_kane_mele(t, soc, dt)` puts one orbital with spin on each of the two sites
of the primitive cell, an `s` orbital in the paper, so the model has four
bands, the lower two filled at half filling. `t` is the nearest-neighbor
hopping, and `soc` the spin-orbit coupling between second neighbors. `dt`
holds four offsets added to `t` on the four nearest-neighbor bonds of each
site. The first is for the bond inside the cell, which points along `[111]`.
Energies are in units of `t`. The blocks run in order, and the lines under a
statement are what it prints.

## The undistorted crystal

`realized_group` takes the model, and with `characters=` the site irrep of any
orbital the user declares. A pythtb orbital carries no symmetry character of
its own. The hoppings fix how the orbitals transform relative to one another,
and not what each one is. The example reads the undistorted crystal, every bond
equal, with orbital 0 declared an `s` orbital:

```python
import itertools

import numpy as np
from pythtb.models import fu_kane_mele
import pythtqc as tqc

fkm = fu_kane_mele(1.0, 0.125, [0.0, 0.0, 0.0, 0.0])      # every bond equal
r = tqc.realized_group(fkm, characters={0: "s"})         # orbital 0 is an s orbital
print(repr(r))
#  RealizedGroup(Fd-3m1' #227.129)
print(r.wyckoff_orbits.summary())
#  Wyckoff  site symmetry  orbitals  label  determined by  declared
#       8a         -43m1'       0-1    -E1    declaration       s/-
```

- `Fd-3m1' #227.129`: `r.group`, the group the Hamiltonian obeys. `Fd-3m` is
  the space group of the diamond structure, number 227. The trailing `1'`
  marks time reversal adjoined, and a space group with time reversal adjoined
  is a grey group. `227.129` is its number in the Belov-Neronova-Smirnova
  numbering of the magnetic space groups. The record prints a clause after the
  group for each thing the call did or found beyond the group, such as
  lowering the group the orbital positions allow. Here there is none.
- `Wyckoff  site symmetry  orbitals ...`: `r.wyckoff_orbits`, the model's
  atomic-limit table. It has one row for each Wyckoff orbit, the orbitals the
  operations carry into one another.
- `8a`: the orbit's Wyckoff position, multiplicity and letter, in the
  conventional cell used by the irreptables package. Both orbitals sit there,
  the two atoms of the diamond structure.
- `-43m1'`: the site-symmetry group, the tetrahedral group `-43m` with time
  reversal.
- `0-1`: the orbitals on the orbit, by index.
- `-E1`: the site irrep one site of the orbit carries. The leading `-` marks a
  double-valued irrep, which a spinful orbital carries.
- `declaration`: what the row's site irrep rests on. Orbital 0's is declared.
  Orbital 1's is fixed relative to orbital 0 by the hoppings, so it rests on
  the declaration too.
- `s/-`: what `characters=` declared, one entry for each orbital. Orbital 1
  has none, printed `-`.

## A semimetal at X

`high_symmetry_points` returns the high-symmetry points of a group, in the
model's reduced coordinates. `solve_at` returns the energies and states at one
k-point. The example prints the four energies of the undistorted crystal at
each point:

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

- `GM`: the irreptables package's name for Γ.
- `-4.000 -4.000 +4.000 +4.000`: the four energies at Γ, lowest first.

Every level is a Kramers pair, since the crystal has both inversion and time
reversal. At `X` all four bands meet at zero energy. So the two lower bands are
not separated from the upper two, and the undistorted crystal is a semimetal.

In a group with antiunitary operations, such as time reversal, the bands at
each k-point carry irreducible co-representations, coreps, of its little group.
Every label of a band on this page is the name of a corep.

`hsp_table` takes what `realized_group` returned and `bands=`, the set of bands
to read. It can be given as an integer n for the lowest n bands, or as a
sequence of band indices. It returns the coreps the set carries at each
high-symmetry point. A point's coreps can only be returned if the set contains
all the states in degenerate multiplets there. A point where the set cuts a
multiplet is refused with `multiplet_split`, and the table holds that refusal
for the point. A refusal is a returned record that names the check that
refused and what it measured. `decompose` returns the decompositions of a
table's symmetry data vector; the entries under the example define both.
`symmetry_indicator` returns the class of the vector
([Three phases, and the Z4 indicator](#three-phases-and-the-z4-indicator)).
The example reads all four bands of the undistorted crystal and decomposes
them. It then reads the two lowest bands alone and asks `symmetry_indicator`
for their class:

```python
all4 = tqc.hsp_table(r, bands=4)
print(tqc.format_coreps(all4.symmetry_data))
#  -GM6 + -GM9 + -L8 + -L9 + -W5W6 + -W7 + -X5
print([s.label for s in tqc.decompose(all4).candidate_sums])
#  ["-E1 at 8a (-43m1')"]

pair = tqc.hsp_table(r, bands=2)                         # the two lowest bands alone
print(pair)
#  HSPTable(bands 0-1, 3 k-points measured, 1 refused)
print(pair.refusals["X"].gate, pair.refusals["X"].context["multiplet"])
#  multiplet_split (0, 1, 2, 3)
refused = tqc.symmetry_indicator(pair)
print(bool(refused), refused.gate)
#  False incomplete_content
```

- `-GM6 + -GM9 + ... + -X5`: `all4.symmetry_data`, the symmetry data vector of
  the four bands. It holds the corep multiplicities of the bands at every
  high-symmetry point, and `format_coreps` writes it as a sum.
- `["-E1 at 8a (-43m1')"]`: the `label` of each decomposition in
  `decompose(all4).candidate_sums`, one here. A decomposition writes the
  symmetry data vector as a sum of the symmetry data vectors of
  elementary band representations (EBRs), with nonnegative integer
  multiplicities. A site irrep at a Wyckoff position induces a band
  representation. An EBR is one that is not a sum of others. A
  decomposition's `label` names the site irrep that induces each EBR, its
  Wyckoff position, and its site-symmetry group in parentheses. This one is
  the site irrep `-E1` at `8a`, the model's own two orbitals.
- `HSPTable(bands 0-1, 3 k-points measured, 1 refused)`: the table of the two
  lowest bands, with the coreps at three points and a refusal at one.
- `multiplet_split (0, 1, 2, 3)`: `pair.refusals["X"]`, the refusal at `X`.
  Its `gate` names the check that refused the point. Its
  `context["multiplet"]` names the bands of the degenerate multiplet, of which
  the two lowest bands are half.
- `False incomplete_content`: `symmetry_indicator(pair)` is a refusal, which is
  falsy, from the check `incomplete_content`. The table holds no coreps at `X`.

Both refusals here say that the crystal has no gap at `X`, so the two lowest
bands carry no coreps of their own there.
[What does a refusal mean, and what do I do about it?](../guide/refusals.md)
says how to read one. [Refusals](../reference/refusals.md) holds both checks,
[`multiplet_split`](../reference/refusals.md#multiplet_split) and
[`incomplete_content`](../reference/refusals.md#incomplete_content).

## A distortion lowers the group

The example strengthens the `[111]` bond of the Fu-Kane-Mele model by `0.4`.
No orbital moves, so the orbital positions still allow all of `Fd-3m1'`.
`rs.descent` holds what was broken, and `rs.operations` the operations kept:

```python
strong = fu_kane_mele(1.0, 0.125, [0.4, 0.0, 0.0, 0.0])  # the [111] bond strengthened
rs = tqc.realized_group(strong, characters={0: "s"})
print(repr(rs))
#  RealizedGroup(R-3m1' #166.98; descended from Fd-3m1' #227.129 (72 of 96 coset representatives of the lattice translations broken; these are the single-valued group's, each standing for itself and its negative in the double group))
d = rs.descent
print(d.cell_group.name, "->", d.subgroup.name, d.dropped_by,
      sorted({b.refuted_by for b in d.dropped}))
#  Fd-3m1' -> R-3m1' {'hoppings': 72} ['no symmetry matrix']
for op in rs.operations.values():
    if not op.seitz.endswith("'"):                       # a trailing ' marks time reversal
        print(f"{op.index:>2}  {op.seitz:<16}  {op.name}")
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
for o in rs.wyckoff_orbits:
    print(o.orbitals, o.wyckoff, o.unitary_site_symmetry, o.label, o.determined_by)
#  (0, 1) 6c 3m -E1 declaration
```

- `R-3m1' #166.98`: `rs.group`, the group returned, the rhombohedral grey
  group whose threefold axis is the strengthened bond.
- `descended from Fd-3m1' #227.129`: the group the orbital positions allow,
  which the Hamiltonian does not obey.
- `72 of 96 coset representatives of the lattice translations broken`: the
  Hamiltonian broke 72 of that group's 96 operations, counted one for each
  coset of the lattice translations. The 24 that hold form `R-3m1'`.
- `these are the single-valued group's, each standing for itself and its
  negative in the double group`: the 96 and the 72 count operations of the
  single-valued group. In this spinful model each stands for two elements of
  the double group, the operation and its product with the rotation by 2π.
- `Fd-3m1' -> R-3m1'`: `d.cell_group` and `d.subgroup`, the group before the
  descent and the group after it.
- `{'hoppings': 72}`: `d.dropped_by`, how many operations each cause dropped.
  The hoppings dropped all 72.
- `['no symmetry matrix']`: the `refuted_by` of the entries of `d.dropped`,
  one entry for each operation dropped. For an operation `g = {R|t}`,
  `"no symmetry matrix"` means that no matrix `X` satisfies
  `X H(k) = H(Rk) X` at every `k` of the grid the measurement uses, and for
  `g` combined with time reversal none satisfies `X H(k)* = H(−Rk) X`. That
  proves the operation broken.
  [The descent](../reference/records.md#the-descent), in
  [Returned records](../reference/records.md), lists the other ways an
  operation is refuted.
- `1  {1|0,0,0}         E`: each operation kept without time reversal, its
  `index`, its `seitz` and its `name`. The Seitz symbol `{R|t}` is written in
  the conventional cell used by the irreptables package, whose `c` axis is
  here the threefold. The name is read in the Cartesian axes in which the
  pythtb Lattice is specified, the frame `characters=` is written in. Its axis
  is written `x`, `y` or `z` where it lies along one of those axes, and
  otherwise as a direction in the model's lattice vectors. So `C3([111])`
  turns about the sum of the three lattice vectors, the strengthened bond. An
  improper rotation is named after its rotoinversion: `S3([111])` is the
  Seitz `-3+`, which Schoenflies notation calls `S6^5`
  ([Operations: a Seitz symbol and a Cartesian name](../background/conventions.md#operations-a-seitz-symbol-and-a-cartesian-name)).
- `(0, 1) 6c 3m -E1 declaration`: the one Wyckoff orbit's `orbitals`,
  `wyckoff`, `unitary_site_symmetry`, `label` and `determined_by`.
  `unitary_site_symmetry` is the site-symmetry group without time reversal,
  and `label` is the site irrep.

The operations kept without time reversal are the twelve of the point group
`-3m` about the bond, printed above. Each also holds combined with time
reversal, 24 operations in all. Every later call on this model reads its bands
in `R-3m1'`.

In `Fd-3m1'` the orbitals were the orbit `8a`, whose site-symmetry group
without time reversal is `-43m`. In `R-3m1'` they are the orbit `6c`, where it
is `3m`, the threefold about the bond and the three mirrors that contain it.
The site irrep is still `-E1`, now of `3m`, and the one declaration still
fixes both orbitals.
[When the Hamiltonian has less symmetry than its cell](../guide/group.md#when-the-hamiltonian-has-less-symmetry-than-its-cell),
in [What group does my Hamiltonian have?](../guide/group.md), works through a
descent.

## Three phases, and the Z4 indicator

The same bond weakened by `0.4` is a second phase. The other three bonds cut to
zero is a third. There each atom keeps one nearest neighbor, its partner along
`[111]`, and the dimers they form are coupled to one another only by the
spin-orbit term. All three phases have the group `R-3m1'`.

The integer vectors that satisfy a space group's compatibility relations,
negative entries allowed, form a lattice, `{BS}`. The integer combinations of
the symmetry data vectors of its EBRs form a sublattice, `{AI}`. The quotient
`X_BS = {BS}/{AI}` is a finite abelian group fixed by the space group alone,
its indicator group. The symmetry indicator of a set of bands is the class of
its symmetry data vector in `X_BS`. The class of an atomic limit is zero, and
classes add. So a set of bands with a nonzero class is not an atomic limit,
and does not become one when atomic bands are added.
[Which symmetry-indicator class do these bands occupy?](../guide/indicators.md)
gives the definitions and the papers.

`symmetry_indicator` takes the table `hsp_table` returns, and returns the class
of its symmetry data vector. The example reads the class of the two lowest
bands of each phase:

```python
phases = {
    "strong": rs,
    "weak": tqc.realized_group(fu_kane_mele(1.0, 0.125, [-0.4, 0.0, 0.0, 0.0]),
                               characters={0: "s"}),
    "dimers": tqc.realized_group(fu_kane_mele(1.0, 0.125, [0.0, -1.0, -1.0, -1.0]),
                                 characters={0: "s"}),
}
tables = {name: tqc.hsp_table(rr, bands=2) for name, rr in phases.items()}
indicators = {name: tqc.symmetry_indicator(tables[name])
              for name, rr in phases.items()}
print(f"X_BS of #{rs.group.number_str} {rs.group.name}: {indicators['strong'].indicator_group}")
#  X_BS of #166.98 R-3m1': Z2 x Z4
print("phase   group   symmetry data vector         class   order  z4  z2w")
#  phase   group   symmetry data vector         class   order  z4  z2w
for name, rr in phases.items():
    si = indicators[name]
    z2w = "".join(str(si.indicators[f"z2w,{i}"]) for i in (1, 2, 3))
    print(f"{name:<7} {rr.group.number_str:<7} {tqc.format_coreps(tables[name].symmetry_data):<28} "
          f"{str(si.values):<7} {si.order:<6} {si.indicators['z4']:<3} {z2w}")
#  strong  166.98  -GM9 + -F5F6 + -L5L6 + -T8   (0, 1)  4      3   111
#  weak    166.98  -GM9 + -F3F4 + -L5L6 + -T8   (1, 0)  2      0   111
#  dimers  166.98  -GM9 + -F5F6 + -L5L6 + -T9   (0, 0)  1      0   000
```

- `Z2 x Z4`: `si.indicator_group`, the indicator group of `R-3m1'`. A class is
  an integer modulo 2 and one modulo 4.
- `-GM9 + -F5F6 + -L5L6 + -T8`: the symmetry data vector of the strong phase's
  two lowest bands, at the four high-symmetry points of `R-3m1'`: Γ, `T` on the
  threefold axis, `F` and `L`. Each phase's two lowest bands are separated
  from the upper two at all four points.
- `(0, 1)`: `si.values`, the class, `0` modulo 2 and `1` modulo 4. The EBR
  matrix of the group has the symmetry data vector of each EBR as a column.
  The values depend on the Smith normal form of that matrix, and the
  irreptables package ships the one used here. Another Smith normal form can
  write the same class with other values.
- `4`: `si.order`, the order of the class, the fewest copies of the bands that
  together have class zero. The order is the same in every Smith normal form.
- `3` and `111`: `si.indicators`, the class written as the values of named
  indicators. `z4` is the strong indicator. `z2w,1`, `z2w,2` and `z2w,3` are
  the three weak indicators, printed together. The named values determine the
  class, as `values` does
  ([The names a class is spoken in](../guide/indicators.md#the-names-a-class-is-spoken-in)).

The three symmetry data vectors differ only at `T` and `F`.

In a crystal with inversion, Fu and Kane read the strong and weak indices off
the inversion parities
([*Topological insulators with inversion symmetry*, Phys. Rev. B **76**, 045302 (2007)](https://doi.org/10.1103/PhysRevB.76.045302)).
The time-reversal-invariant momenta (TRIM) are the eight momenta `k` with
`k ≡ −k` modulo a reciprocal lattice vector. At each, a Kramers pair has one
inversion parity `ξ = ±1`. With one occupied pair, the strong index `ν0` is
given by `(−1)^ν0 = ∏ ξ` over all eight TRIM. The weak index `νi` is given by
the same product over the four TRIM with `k_i = 1/2`.

The parities are those of the model's own inversion, about the bond's center,
the operation `{-1|0,0,0}` in the conventional cell. `operation_index` returns
the index of the operation a Seitz symbol names. `sewing_matrix` returns the
matrix of an operation on a set of bands at a `k` the operation leaves
invariant. On one Kramers pair, the inversion's sewing matrix has the trace
`2ξ`. The example prints each phase's parities at the eight TRIM:

```python
trim = [np.array(n) / 2 for n in itertools.product((0, 1), repeat=3)]   # the eight TRIM
for name, rr in phases.items():
    inversion = tqc.operation_index(rr.group, "{-1|0,0,0}")
    parity = [round(np.trace(tqc.sewing_matrix(rr, inversion, k, bands=2).matrix).real / 2)
              for k in trim]                             # the pair's inversion parity at each TRIM
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
#  another labeling the hoppings leave open keeps every site irrep and labels some energy multiplets of these bands differently; it keeps which bands are degenerate and the labels of all the model's bands together
```

- `- - - - - - - +`: the parity at each TRIM, in the order `itertools.product`
  gives them in reduced coordinates. The first is `(0, 0, 0)`, which is Γ, and
  the last `(1/2, 1/2, 1/2)`, which is `T`. The three TRIM with two halves are
  the three `F` points, and the three with one half the three `L` points.
- `(1;111)`: `ν0` and the three weak indices, written `(ν0;ν1ν2ν3)` as Fu and
  Kane write them.
- `7 odd pairs, 7 mod 4 = 3`: the number of TRIM where the pair is odd, and
  that number modulo 4.
- `another labeling the hoppings leave open ...`: `tables["strong"].note`, a
  sentence saying that the corep labels of these bands rest on a choice the
  hoppings leave open. The last paragraph of this section reads it.

The dimers are odd at every TRIM. The strong phase differs
from them at `T` alone, one parity inverted: `(1;111)`, a strong topological
insulator. The weak phase differs at `T` and the three `F` points, four
parities inverted: `(0;111)`, a weak topological insulator. These are the
indices Fu, Kane and Mele give for the two signs of the distortion. In each
phase `z4` is the number of odd pairs modulo 4: 3, 0 and 0. So
`ν0 = z4 mod 2`, and the weak indices are the `z2w` column.

Two choices fix these parities, and with them `z4`. First, the parities are
taken about the bond's center. About an inversion center half a lattice vector
away, the parities at four of the eight TRIM flip, which can change `z4` by 2.
Second, they are the parities of `s` orbitals. The representation of the group
on the orbitals is the set of matrices by which its operations act on them.
The hoppings leave open two such representations, and both give the two
orbitals the site irrep `-E1`. The note calls each a labeling.
`tables["strong"].note` says that the corep labels of the two lowest bands
differ between them. The labeling returned reads both orbitals as `s`
functions, which the inversion carries onto each other with `+1`. The other
differs from it by a sign on the inversion, the twofold rotations and the
rotoinversions. That sign is a one-dimensional character of the group that is
`+1` on the site-symmetry group `3m`, so the site irrep does not fix it. The
tables of all three phases carry the same note. Under the other labeling every
parity flips. `ν0` and the weak indices stay, and `z4` changes sign: the strong
phase's seven odd pairs become one. Khalaf, Po, Vishwanath and Watanabe write
the Z4 index as `(1/2) Σ (even − odd)` modulo 4, summed over the eight TRIM
with each Kramers pair counted once
([*Symmetry indicators and anomalous surface states of topological crystalline insulators*, Phys. Rev. X **8**, 031070 (2018)](https://doi.org/10.1103/PhysRevX.8.031070)).
It is minus the number of odd pairs modulo 4, which is the `z4` of the other
labeling. For the strong phase it is `(1/2)(1 − 7) = −3`, which is 1 modulo 4.

## What a decomposition says

`decompose` takes the same table and returns every decomposition of its
symmetry data vector, up to `cap=` of them, 200 by default. Where there is
none, `certificate` says whether its absence is proved. `signed_combinations`
holds the same search with negative multiplicities allowed.
`wyckoff_sites_tables` returns the Wyckoff position of
a point, given in the model's reduced coordinates. The example decomposes the
two lowest bands of each phase, and finds the Wyckoff position of the center of
the `[111]` bond:

```python
for name, rr in phases.items():
    dec = tqc.decompose(tables[name])
    signed = (f"{len(dec.signed_combinations.sums)} signed"
              if dec.signed_combinations is not None else "-")
    answer = "decomposable" if dec.decomposable else "not decomposable"
    found = f"{len(dec.candidate_sums)} sum{'' if len(dec.candidate_sums) == 1 else 's'}"
    print(f"{name:<7} {answer:<17} {dec.certificate or '-':<6} {signed:<11} {found}"
          + "".join(f": {s.label}" for s in dec.candidate_sums))
#  strong  not decomposable  proof  0 signed    0 sums
#  weak    not decomposable  proof  0 signed    0 sums
#  dimers  decomposable      -      -           1 sum: -E1u at 3a (-3m1')
center = tqc.wyckoff_sites_tables(phases["dimers"].group, [[0.125, 0.125, 0.125]])
print("the center of the [111] bond:", [s.wyckoff for s in center])
#  the center of the [111] bond: ['3a']
```

- `not decomposable`: `dec.decomposable` is false. No decomposition was found.
- `proof`: `dec.certificate`. No sum within the bound searched carries the
  symmetry data vector, and no multiplicity can exceed that bound.
- `0 signed`: the number of combinations in `dec.signed_combinations.sums`,
  none within the bound. `signed_combinations` is `None` where a decomposition
  exists, printed `-`.
- `-E1u at 3a (-3m1')`: the dimers' one decomposition, the site irrep `-E1u`
  at Wyckoff position `3a`, whose site-symmetry group is `-3m1'`.
- `['3a']`: the Wyckoff position of the point `(1/8, 1/8, 1/8)` in the model's
  reduced coordinates, the center of the bond the dimers keep.

The strong phase has no decomposition, with a proof, so its two lowest bands
are not an atomic limit. No signed combination lies within the bound either. A
symmetry data vector with a nonzero class is no integer combination of EBRs at
all. The symmetry data vector of a fragile set of bands, by contrast, is an
integer combination of EBRs only with some negative multiplicity
([When there is no decomposition](../guide/atomic-limits.md#when-there-is-no-decomposition)).
Where `decompose` returns no decomposition, the bands are not an atomic limit,
and may be fragile. Where the class is nonzero, they are stably topological.
`decompose` returns the same for the weak phase. Its class tells it from the
strong phase: order 2, where the strong phase's has order 4.

The dimers have one decomposition, the site irrep `-E1u` at the bond's center.
The `u` marks a site irrep odd under the inversion about that center. It is
the site irrep of the odd combination of each dimer's two `s` orbitals, with
both spin states. Their class is zero, as an atomic limit's is. Fragile sets
of bands, and stably topological sets the indicator does not detect, have
class zero too
([What a zero does not certify](../guide/indicators.md#what-a-zero-does-not-certify)).

## Where to go next

- [What do the stable real-space invariants say about these bands?](../guide/srsi.md)
  reads the same three phases
  [from their bands](../guide/srsi.md#from-the-bands).
- [What group does my Hamiltonian have?](../guide/group.md) takes the descent
  further.
- [Which little-group representations do these bands carry?](../guide/labels.md)
  reads the tables `hsp_table` returns.
- [`examples/tqc_fkm_diamond.py`](../../examples/tqc_fkm_diamond.py) measures
  the same phases and draws their bands, with the coreps of the two lowest
  bands at each high-symmetry point ([The examples](../examples.md)).
- [Returned records](../reference/records.md) lists every field of the records
  this page prints.
- [Functions](../reference/functions.md) gives the arguments and the refusals
  of every call on this page:
  [`realized_group`](../reference/functions.md#realized_group),
  [`high_symmetry_points`](../reference/functions.md#high_symmetry_points),
  [`solve_at`](../reference/functions.md#solve_at),
  [`hsp_table`](../reference/functions.md#hsp_table),
  [`format_coreps`](../reference/functions.md#format_coreps),
  [`decompose`](../reference/functions.md#decompose),
  [`symmetry_indicator`](../reference/functions.md#symmetry_indicator),
  [`operation_index`](../reference/functions.md#operation_index),
  [`sewing_matrix`](../reference/functions.md#sewing_matrix) and
  [`wyckoff_sites_tables`](../reference/functions.md#wyckoff_sites_tables).
