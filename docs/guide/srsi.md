# What do the stable real-space invariants say about these bands?

A symmetry indicator places a band set's content in a class of band structures
modulo atomic limits
([how the class is computed](../background/tqc-primer.md#how-a-symmetry-indicator-is-computed);
[symmetry indicators](indicators.md) reads one). The **stable
real-space invariants** (SRSIs) speak from the other side, of where Wannier
functions sit and which
[site irreps](../background/tqc-primer.md#site-symmetry-and-wyckoff-positions)
they carry, and they say more: two atomic
limits with the same content at every high-symmetry point can differ in an
SRSI, and then no symmetric deformation turns one into the other. This chapter
says what a real-space invariant is, reads the tables of one space group with
no model, reads a model's invariants from its bands and from its orbitals, and
ends with the one verdict they issue, a no-go criterion for a gapped splitting.

The running model is lin3g, from `pythtqc.models`, as in
[labelling the bands](labels.md): four spinful orbitals of a square layer, two
at Wyckoff `1a` and two at `2c`, whose six lowest bands hold a
[fragile](../background/tqc-primer.md#fragile-topology) pair.
Fu–Kane–Mele's diamond, from `pythtb.models`, in the three phases a bond
distortion makes ([the 3D tutorial](../start/tutorial-3d.md)), supplies the
nonzero indicators lin3g's group cannot have. The tables are those of
Hwang *et al.*,
[*Stable real-space invariants and topology beyond symmetry indicators*, Nat. Commun. **17**, 8564 (2026)](https://doi.org/10.1038/s41467-026-74844-w),
shipped with the package.

## What a real-space invariant is

An atomic limit is a band set spanned by exponentially localized, symmetric
Wannier functions ([the primer](../background/tqc-primer.md#band-representations-and-atomic-limits)),
and in real space it is a list of whole numbers: at each Wyckoff position, how
many times each site irrep occurs among the Wannier functions centred there,
the **site-symmetry vector** `p`.
A deformation that keeps the symmetry moves Wannier functions only in symmetric
clusters: a function leaves a special position with its images, as one orbit
of a less symmetric position on a line or plane through it, and such an orbit
can slide onto any special position on that line or plane, arriving as a fixed
set of site irreps there. A combination of the entries of `p` that no such move
changes is a **real-space invariant**.

The simplest sits on a twofold axis. Spinless orbitals on a site of site
symmetry `C2` are even or odd under the rotation, the site irreps `A` and `B`.
Two orbitals that the rotation exchanges arrive on the axis as one even and one
odd combination, so every move changes `m(A)` and `m(B)` together, and
`m(A) − m(B)` is a ℤ-valued invariant. Where every move changes a multiplicity
in steps of `n`, the multiplicity modulo `n` is a ℤₙ-valued one. `Pmm2`, space
group 25, with spin–orbit coupling and time reversal, has those: Wannier
functions on its line `1a`, of site symmetry `mm2`, come in Kramers pairs in
`Ē`, the one double-valued irrep of `mm2`, and they leave `1a` two at a time,
as an orbit of two sites on a mirror plane through it, or four at a time, at
the general position. The number of `Ē` at `1a` is invariant modulo 2. And one
Kramers pair at `1a` has the same content at every high-symmetry point as one
at `1b`, every little group there being `mm2` with its one double-valued
[corep](../background/tqc-primer.md#little-groups-irreps-and-coreps), while the
two differ in those parities: no symmetric deformation takes one into the
other. The ℤₙ-valued invariants are why real-space invariants see more than a
[symmetry data vector](labels.md#every-high-symmetry-point-at-once).

## The stable ones

The invariants this package ships are written over the whole site-symmetry
vector at once, every site irrep at every Wyckoff position, maximal or not,
since the moves run between all of them; at a maximal position that includes
the site irreps whose band representation is a sum of others, which no list of
elementary band representations (EBRs) carries
([the exceptions](../background/tqc-primer.md#band-representations-and-atomic-limits)).
Each allowed move is a column of an integer matrix `q`, and the SRSIs are the
complete set of linear forms `θ = Θ·p` that vanish on every column, modulo `n`
for a ℤₙ-valued one, the
[elementary divisors](../background/tqc-primer.md#how-a-symmetry-indicator-is-computed)
of `q` fixing which. A move counts with an integer
coefficient, positive or negative, so two configurations with the same SRSIs
deform into each other once the same atomic orbitals are added to both: the
invariants classify atomic configurations up to stable equivalence.

Two facts carry the rest of the chapter. The ℤ-valued SRSIs are a function of
the symmetry data vector alone: two configurations share them exactly when they
share their content at every high-symmetry point. So every band set has
ℤ-valued SRSIs, an atomic limit or not: those of any rational `p` whose content
is the band set's. And, the paper's theorem, a band set has a nonzero symmetry
indicator exactly when its ℤ-valued SRSIs are fractional. The ℤₙ-valued ones
are no function of the symmetry data vector, which makes the SRSIs finer than an
indicator. The tables cover the
[grey groups](../background/tqc-primer.md#little-groups-irreps-and-coreps),
the space groups with time reversal, with spin–orbit coupling and without; a
group whose one invariant is the band count has no row
([when it refuses](#when-it-refuses)).

## The call

`srsi_tables` loads one group's tables, from a group a model was measured to
have or from a space-group number with `spinor=`, which needs no model:

```python
import pythtqc as tqc

t25 = tqc.srsi_tables(25, spinor=True)            # Pmm2 with spin-orbit coupling
print(t25.summary())
#  Pmm21' #25.58, double-valued: 4 stable real-space invariants   (.group, .invariants)
#    each invariant, and its modulus where it is valued modulo n   (.invariants, .moduli)
#      theta_1    Z
#      theta_2   Z2
#      theta_3   Z2
#      theta_4   Z2
#    9 site irreps at Wyckoff positions, 4 of them inducing an elementary band representation   (.site_irreps)
#    no symmetry indicator, the group's indicator group being trivial   (.indicators)
print(t25.formulas())                              # each invariant, term by term
#  each invariant, Z or Z_n by its modulus, as a sum of terms c (irrep)_w, c times the multiplicity of a site irrep at Wyckoff position w   (.invariants, .moduli)
#    theta_1 (Z)  = 2 (-1E2E)_e + 2 (-1E2E)_f + 2 (-1E2E)_g + 2 (-1E2E)_h + 4 (-AA)_i + 1 (-E)_a + 1 (-E)_b + 1 (-E)_c + 1 (-E)_d
#    theta_2 (Z2) = 1 (-E)_b
#    theta_3 (Z2) = 1 (-E)_c
#    theta_4 (Z2) = 1 (-E)_d
print(t25.theta({("-E", "1a"): 1}))                # one Kramers pair at 1a
#  Invariants(theta_1 = (1); theta_2 to theta_4 = (0, 0, 0) mod 2)
print(t25.theta({("-E", "1b"): 1}))                # ... or at 1b
#  Invariants(theta_1 = (1); theta_2 to theta_4 = (1, 0, 0) mod 2)
```

## How to read it

- `Pmm21' #25.58, double-valued`: the grey group of space group 25, with
  spin–orbit coupling. Its SRSIs are one ℤ-valued and three ℤ₂-valued
  invariants, and the nine site irreps at Wyckoff positions are the coordinates
  of `p`. Where a group has an indicator, the summary lists its named forms,
  each a combination of the `θ_i` that `t25.indicators_of(p)` evaluates; four
  printed forms, in space groups 147, 162 and 163 without spin–orbit coupling,
  do not vanish on their own group's EBRs, and the package evaluates forms
  derived from the momentum-space indicator in their place
  ([the tables example](../../examples/tqc_srsi_tables.py) prints them). Where
  the paper lists splittings of an EBR into EBRs that the indicators allow, a
  last line counts them ([below](#the-no-go-criterion)); it lists none here.
- The rows spell the site irreps as the tables do, a leading `-` marking a
  double-valued one ([conventions](../background/conventions.md)): `-E` is `Ē`
  on the lines `1a` to `1d`, `-1E2E` a Kramers pair on the mirror planes `2e`
  to `2h`, `-AA` one at the general position `4i`. `theta_1` counts Kramers
  pairs per cell, the band count being `2·theta_1`; `theta_2` to `theta_4`
  count `Ē` at `1b`, `1c` and `1d` modulo 2, and the parity at `1a` follows.
- `theta(p)` takes `p` as a mapping `{(site irrep, Wyckoff position):
  multiplicity}`, the position written with its multiplicity, `1a`, or as a sum
  of term rows, the shape a decomposition writes it in. It returns the
  invariants by name: `theta_Z`, the ℤ-valued SRSIs as exact fractions, and
  `theta_Zn`, the ℤₙ-valued ones modulo `n`. A Kramers pair at `1a` and one at
  `1b` share `theta_1` and differ in `theta_2`, the paper's first worked
  example.

The same tables, field by field:

```python
print(t25.group.number_str, t25.moduli, t25.provenance.doi)
#  25.58 {'theta_1': None, 'theta_2': 2, 'theta_3': 2, 'theta_4': 2} https://doi.org/10.1038/s41467-026-74844-w
print([(key.irrep, key.wyckoff) for key in t25.site_irreps if key.ebr])
#  [('-E', '1a'), ('-E', '1b'), ('-E', '1c'), ('-E', '1d')]
print([(key.irrep, key.wyckoff, key.maximal) for key in t25.site_irreps if not key.ebr])
#  [('-1E2E', '2e', False), ('-1E2E', '2f', False), ('-1E2E', '2g', False), ('-1E2E', '2h', False), ('-AA', '4i', False)]
p = {("-E", "1b"): 3}                              # three Kramers pairs at 1b
print(t25.theta(p), t25.theta_unreduced(p))
#  Invariants(theta_1 = (3); theta_2 to theta_4 = (1, 0, 0) mod 2) {'theta_1': Fraction(3, 1), 'theta_2': Fraction(3, 1), 'theta_3': Fraction(0, 1), 'theta_4': Fraction(0, 1)}
```

`group.number_str` names the group, `moduli` gives each invariant's modulus,
`None` for a ℤ-valued one, and `provenance.doi` the paper the tables are
converted from. A site irrep at a Wyckoff position is marked `maximal` where
its position is and `ebr` where it induces an EBR: here the four `Ē` on the
lines, and none of the five on the mirror planes and at the general position.
`theta_unreduced(p)` reads every row with the ℤₙ ones unreduced, three pairs at
`1b` giving `3` in `theta_2` where `theta` gives `1` modulo 2.

## From the bands

`srsi_of_bands` reads the ℤ-valued SRSIs off a band set's symmetry data vector `y`
([labelling the bands](labels.md#every-high-symmetry-point-at-once)): it solves
`BR·p = y` over the rationals, `BR` being
[the EBR matrix](../background/tqc-primer.md#how-a-symmetry-indicator-is-computed),
and evaluates `Θ` on that `p`. lin3g's six lowest bands, and the detuned
diamond's three phases:

```python
from pythtqc import models
from pythtb.models import fu_kane_mele

r = tqc.realized_group(models.lin3g(), typat=[0, 0, 1, 1])
group = r.group
six = tqc.srsi_of_bands(tqc.hsp_table(r, bands=6))
print(six.summary())
#  P4mm1' #99.164, double-valued: the Z-valued stable real-space invariants of the bands   (.group, .theta_Z)
#      theta_1 =        1
#      theta_2 =        1
#      theta_3 =        1
#    integral   (.theta_Z)
#    their symmetry-indicator group is trivial   (.indicator)
phases = {"strong": [0.4, 0.0, 0.0, 0.0], "weak": [-0.4, 0.0, 0.0, 0.0],
          "dimers": [0.0, -1.0, -1.0, -1.0]}
diamond = {name: tqc.realized_group(fu_kane_mele(1.0, 0.125, dt), characters={0: "s"})
           for name, dt in phases.items()}
valence = {}                                       # each phase's valence pair
for name, rd in diamond.items():
    b = valence[name] = tqc.srsi_of_bands(tqc.hsp_table(rd, bands=2))
    print(name, b.line())                          # the record on one line
#  strong theta_1 to theta_8 = (-1/4, 0, 1/2, 1, 1/4, -1/2, -1/4, 1/4), fractional; class (0, 1) in Z2 x Z4, of order 4
#  weak theta_1 to theta_8 = (0, 0, 1, 1, 1/2, 0, -1/2, 0), fractional; class (1, 0) in Z2 x Z4, of order 2
#  dimers theta_1 to theta_8 = (0, 0, 0, 1, 0, 0, 0, 0), integral; class zero in Z2 x Z4
```

lin3g's group has a trivial indicator group, and its three ℤ-valued SRSIs are
integers, as they must be. **Read the denominators** in the diamond's group,
`R-3m1'` with indicator group `ℤ₂ × ℤ₄`: the strong phase's valence pair has
quarters and a class of order 4, the weak phase's halves and order 2, the
dimers' integers and the zero class, the common denominator being the order of
the class in each. `theta_Z` holds exact fractions: a band set with a nonzero
class has no integer
site-symmetry vector, not even one with negative entries as a fragile band set
has, and a fraction says so in real space.

The tables carry the paper's indicators as forms in these invariants, keyed
by name: `moduli` holds each one's modulus, and `numerical.indicators` its
`relation`, how the printed form compares with the one derived from the
momentum-space indicator. `indicators_of(p)` evaluates them on a
site-symmetry vector, and on the band route's `p`, `numerical.p`, they give
each phase's class in the paper's names:

```python
t166 = tqc.srsi_tables(diamond["strong"].group)
for name, form in t166.indicators.items():
    print(name, t166.moduli[name], t166.numerical.indicators[name].relation, form)
#  z2w,1 2 equal 2 theta_1 - 2 theta_2 + 2 theta_3 - 2 theta_4 - 2 theta_7
#  z2w,2 2 equal 2 theta_1 - 2 theta_2 + 2 theta_3 - 2 theta_4 - 2 theta_7
#  z2w,3 2 equal 2 theta_1 - 2 theta_2 + 2 theta_3 - 2 theta_4 - 2 theta_7
#  z4 4 equal -4 theta_1 + 4 theta_2 - 4 theta_3 + 4 theta_4
for name, b in valence.items():
    print(name, t166.indicators_of(b.numerical.p))
#  strong {'z2w,1': 1, 'z2w,2': 1, 'z2w,3': 1, 'z4': 3}
#  weak {'z2w,1': 1, 'z2w,2': 1, 'z2w,3': 1, 'z4': 0}
#  dimers {'z2w,1': 0, 'z2w,2': 0, 'z2w,3': 0, 'z4': 0}
```

`z4 = 3` with every weak index 1 in the strong phase, every weak index 1 in the
weak one, and zero in the dimers: the indices
[the 3D tutorial](../start/tutorial-3d.md) reads off the inversion parities.
The three weak forms are one form, the threefold axis carrying each weak index
onto the next.

## From the orbitals

`srsi_of_model` reads both halves off the model's orbitals: a tight-binding
model is an atomic configuration, so its `p` is known rather than solved for.
It measures the content of each Wyckoff orbit's own orbitals, the trace of each
little-group operation over them at every high-symmetry point, and decomposes
it over the EBRs of the orbit's position:

```python
mo = tqc.srsi_of_model(r)
print(mo.summary())
#  P4mm1' #99.164, double-valued: stable real-space invariants   (.group)
#    the whole model: theta_1 to theta_3 = (2, 2, 0); theta_4, theta_5 = (0, 1) mod 2   (.theta_Z, .theta_Zn)
#    each Wyckoff orbit, its site irreps written m (irrep)_w, with theta_1 to theta_3 and theta_4, theta_5 mod 2   (.wyckoff_orbits)
#      1a (4mm1'): orbital 0, 1 (-E1)_a: (0, 1, 0); (0, 0) mod 2
#      1a (4mm1'): orbital 1, 1 (-E2)_a: (1, 0, 0); (0, 0) mod 2
#      2c (mm21'): orbitals 2-3, 1 (-E)_c: (1, 1, 0); (0, 1) mod 2
for orbit in mo.wyckoff_orbits:
    print(f"{orbit.wyckoff}:", tqc.format_coreps(orbit.numerical.symmetry_data))
#  1a: -GM7 + -M7 + -X5
#  1a: -GM6 + -M6 + -X5
#  2c: -GM6 + -GM7 + -M6 + -M7 + 2x-X5
eight = tqc.srsi_of_bands(tqc.hsp_table(r, bands=8))
print(eight.theta_Z == mo.theta_Z)
#  True
```

The model's SRSIs come first, the sum of its orbits', and each Wyckoff orbit
follows with its Wyckoff position, its orbitals, the site irreps its content
decomposes into there, and its own SRSIs, ℤ-valued and then ℤ₂-valued. The
two orbitals at `1a` together and the `2c` pair have the same content at every high-symmetry point
and the same ℤ-valued SRSIs, and the second ℤ₂ invariant tells them apart, as
the tables of `Pmm2` told `1a` from `1b`. The model's ℤ-valued SRSIs are
integral, as a configuration of orbitals must be, and equal what the band route
reads off all eight bands: the identity that ties the two routes together.

The record holds the fields the summary names, and for each Wyckoff orbit its
`orbitals` and, under `numerical`, `chosen`, the localization returned, with its
`terms`, each a multiplicity, a site irrep and its Wyckoff position, and its
own invariants. `localizations` holds every localization in that form, here
one per orbit:

```python
print({name: x.value for name, x in mo.theta_Zn.items()})
#  {'theta_4': 0, 'theta_5': 1}
for orbit in mo.wyckoff_orbits:
    chosen = orbit.numerical.chosen
    print(orbit.orbitals, [(t.multiplicity, t.irrep, t.wyckoff) for t in chosen],
          {name: x.value for name, x in chosen.theta_Zn.items()})
#  (0,) [(1, '-E1', '1a')] {'theta_4': 0, 'theta_5': 0}
#  (1,) [(1, '-E2', '1a')] {'theta_4': 0, 'theta_5': 0}
#  (2, 3) [(1, '-E', '2c')] {'theta_4': 0, 'theta_5': 1}
```

An orbit at a position the tables induce no EBR from slides onto one. The
detuned diamond's orbit sits on the line `6c`, `(0, 0, z)` of `R-3m`; the
positions on that line the tables do induce EBRs from are `3a`, at `z = 0`,
and `3b`, at `z = ½`, and the summary writes the orbit's site irreps at `6c`
and the positions it slides onto:

```python
strong = tqc.srsi_of_model(diamond["strong"])
print(strong.summary())
#  R-3m1' #166.98, double-valued: stable real-space invariants   (.group)
#    the whole model: theta_1 to theta_8 = (0, 0, 1, 1, 0, 0, 0, 0)   (.theta_Z)
#    each Wyckoff orbit, its site irreps written m (irrep)_w, with theta_1 to theta_8   (.wyckoff_orbits, .slides_onto)
#      6c (3m1'): orbitals 0-1, 1 (-E1)_c, slides onto 3a and 3b: (0, 0, 1, 1, 0, 0, 0, 0)
print([(loc.wyckoff, tuple(str(x) for x in loc.theta_Z.values()))
       for loc in strong.wyckoff_orbits[0].numerical.localizations])
#  [('3a', ('0', '0', '1', '1', '0', '0', '0', '0')), ('3b', ('0', '0', '1', '1', '0', '0', '0', '0'))]
```

It localizes at both with the same SRSIs, and that agreement is what an SRSI
is: a number no symmetric move changes. An orbit whose site irreps induce a
band representation that is a sum of others, which no EBR carries, is read
through its site irreps instead, which the tables carry on their own.

## Why the ℤₙ half needs the orbitals

`srsi_of_bands` returns no ℤₙ-valued SRSIs: they are no function
of the symmetry data vector, so a `p` solved back out of a band set's content
carries its own ℤₙ values and not the band set's. lin3g's bands 4 and 5, a
fragile pair with the content of the EBR `-E2↑G(2)@1b`
([what a decomposition cannot certify](atomic-limits.md#what-a-decomposition-cannot-certify)),
show it:

```python
t45 = tqc.hsp_table(r, bands=[4, 5])
b45 = tqc.srsi_of_bands(t45)
tables = tqc.srsi_tables(group)
ebrs = [(e.irrep, e.wyckoff) for e in tqc.ebr_data(group).ebrs]   # the group's list of EBRs
print(ebrs)
#  [('-E1', '1a'), ('-E2', '1a'), ('-E1', '1b'), ('-E2', '1b'), ('-E', '2c')]
print([str(b45.numerical.p.get(key, 0)) for key in ebrs], tables.theta(b45.numerical.p))
#  ['1', '1', '-1', '0', '0'] Invariants(theta_1 to theta_3 = (0, 0, 1); theta_4, theta_5 = (0, 0) mod 2)
d45 = tqc.decompose(t45)           # the EBR the pair's content matches
counts = {(t.irrep, t.wyckoff): t.multiplicity for t in d45.candidate_sums[0]}
print(tuple(counts.get(key, 0) for key in ebrs), tables.theta(d45.candidate_sums[0]))
#  (0, 0, 0, 1, 0) Invariants(theta_1 to theta_3 = (0, 0, 1); theta_4, theta_5 = (1, 0) mod 2)
```

The band route's `p`, `(-E1)_a + (-E2)_a − (-E1)_b`, is one rational solution
of `BR·p = y` among many and reads ℤ₂ values `(0, 0)`; the solution `decompose`
returns, one `-E2` at `1b` over the same list, reads `(1, 0)`; both read the
ℤ-valued `(0, 0, 1)`. Read a ℤₙ value only for a configuration that is known:
the one `srsi_of_model` reads off the orbitals, a single EBR, or a solution
`decompose` returned.

## Which Wyckoff letter is a table key

A Wyckoff letter is a key into the tables, and it belongs to a group and a
setting: one site irrep at `8a` and at `8b` of `Fd-3m`, the undistorted
diamond's group, carries different invariants. `Fd-3m` has two origin choices,
and the tables use the one that puts `8a` at `(⅛, ⅛, ⅛)` of the conventional
cell; the diamond's orbitals are `8a` there
([the 3D tutorial](../start/tutorial-3d.md)). Each row of the atomic-limit
table carries its Wyckoff position in that setting:

```python
print([o.wyckoff for o in r.wyckoff_orbits])
#  ['1a', '1a', '2c']
t227 = tqc.srsi_tables(227, spinor=True)           # Fd-3m with spin-orbit coupling
for position in ("8a", "8b"):
    print(position, [str(x) for x in t227.theta({("-E1", position): 1}).theta_Z.values()])
#  8a ['1', '0', '0', '0', '0', '0', '0', '0']
#  8b ['0', '0', '0', '1', '0', '0', '0', '0']
```

`wyckoff`, one per Wyckoff orbit, is in the setting the tables are written in,
and every table lookup uses it. spglib's letter for the orbital positions alone
belongs to a group that can be larger than the Hamiltonian's: lin3g's positions
have the mirror through the layer's plane, which its spin–orbit hoppings break,
and in that larger group the edge centres are `2f`. Only the tables' letter is a
table key; `wyckoff_sites_tables(group)` returns its full record.

Two site irreps at one maximal position can be indistinguishable to everything
this package measures: their EBRs have identical symmetry data vectors, and their
SRSIs are identical too. The tables' `stably_equivalent` lists such pairs; the
paper's own example is `B2` and `B3` at `12d` of `I432`:

```python
pair, = tqc.srsi_tables(211, spinor=False).stably_equivalent
print(pair.summary())
#  B2 and B3 at 12d, which no stable real-space invariant distinguishes   (.irreps, .wyckoff)
#    their elementary band representations have the same symmetry data vector   (.symmetry_data)
#    the invariants both carry: theta_1 to theta_7 = (1, 0, 1, 1, 1, -1, 0); theta_8 to theta_10 = (0, 0, 1) mod 2   (.theta_Z, .theta_Zn)
print(pair.irreps, pair.wyckoff)
#  ('B2', 'B3') 12d
```

The pair's `irreps` and `wyckoff` are the summary's first line, and
`symmetry_data` holds the symmetry data vector its two EBRs share.
The SRSIs do not distinguish the two, which is not to say they are the same:
telling them apart would take an invariant finer than the stable ones, and
whichever member a decomposition picks, no number this package returns moves.

## The no-go criterion

The one verdict the SRSIs issue is the paper's no-go criterion for a gapped
splitting of a model's bands into a valence set and a conduction set. All the
bands together are an atomic insulator, the model's orbitals being its Wannier
functions, so the symmetry data vectors add, the valence set's plus the conduction
set's, and so do the ℤ-valued SRSIs. If the valence and the conduction set are
both atomic insulators, the ℤₙ-valued SRSIs add up too, modulo `n`, for some
**atomic realizations** of each, configurations of orbitals that carry their
symmetry data vectors; if no pair of realizations adds up, one set or both is not an
atomic insulator, and the gap carries topology. One realization that fails to
add up proves nothing, so `splitting` enumerates them all. It takes the gap as
given, reading content at the high-symmetry points only; lin3g's gaps above its
fourth and its sixth band are measured in
[`examples/tqc_srsi_lin3g_composite_splitting.py`](../../examples/tqc_srsi_lin3g_composite_splitting.py).

```python
gap6 = tqc.splitting(r, bands=6)                   # six bands against two
print(gap6.summary())
#  P4mm1' #99.164, double-valued: nsit, no realization of the valence bands and of the conduction bands adding up to the whole model's invariants valued modulo n   (.group, .nsit)
#    Z-valued stable real-space invariants theta_1 to theta_3: the whole model (2, 2, 0) = valence (1, 1, 1) + conduction (1, 1, -1)   (.theta_Z, .valence.theta_Z, .conduction.theta_Z)
#    invariants valued modulo n of the whole model: theta_4, theta_5 = (0, 1) mod 2   (.theta_Zn)
#    atomic realizations of the valence bands, with (theta_4, theta_5) mod 2   (.valence.realizations)
#      1 (-E2)_b + 1 (-E)_c   (1, 1)
#      1 (-E1)_b + 2 (-E2)_b   (0, 0)
#      1 (-E1)_a + 1 (-E2)_a + 1 (-E2)_b   (1, 0)
#    atomic realizations of the conduction bands, with (theta_4, theta_5) mod 2   (.conduction.realizations)
#      1 (-E1)_b   (0, 0)
#    no pair of a valence and a conduction realization has invariants valued modulo n adding up to the whole model's   (.consistent_pairs)
```

- `nsit`: the criterion finds the splitting topological, and
  no symmetry indicator says so, the valence bands' indicator group being
  trivial: the paper calls it non-symmetry-indicated topological (NSIT). The
  last line is the reason.
- The whole model's invariants, of all 8 bands, are read off the orbitals above;
  the valence bands' ℤ-valued SRSIs are read off the six bands, and the
  conduction bands' are the difference.
- The valence bands' atomic realizations are every arrangement of orbitals on
  maximal Wyckoff positions with their ℤ-valued SRSIs and six bands, with
  nonnegative integer multiplicities, enumerated completely, each with its own
  ℤ₂ values; they differ, which is why every one is needed. The conduction
  bands have one.
- The last line: no realization of the valence bands plus one of the conduction
  bands reaches the `(0, 1)` of all 8, so the two band sets cannot both be
  atomic, and `consistent_pairs` is empty. The criterion has caught, by
  arithmetic in real space and without a Wilson loop, an obstruction no
  decomposition sees.

Each line ends with the fields it is read from; `group` is the group, `nsit`
what the criterion returns, and `valence.realizations` and
`conduction.realizations` hold each side's enumeration, every realization with
its own `theta_Zn`:

```python
print(gap6.group.number_str, gap6.nsit, [str(x) for x in gap6.valence.theta_Z.values()],
      {name: x.value for name, x in gap6.theta_Zn.items()})
#  99.164 True ['1', '1', '1'] {'theta_4': 0, 'theta_5': 1}
print([tuple(x.value for x in v.theta_Zn.values()) for v in gap6.valence.realizations],
      [tuple(x.value for x in c.theta_Zn.values()) for c in gap6.conduction.realizations])
#  [(1, 1), (0, 0), (1, 0)] [(0, 0)]
```

Each of the valence bands' three realizations with the conduction bands' one
adds up to `(1, 1)`, `(0, 0)` or `(1, 0)`, and none to the `(0, 1)` of all 8.

The same Hamiltonian with the gap two bands lower, and the detuned diamond,
read `nsit` false in three other ways, each named on the summary's first line:

```python
four = tqc.splitting(r, bands=4)
print(four.nsit, [(v.label, c.label) for v, c in four.consistent_pairs])
#  False [('1 (-E)_c', '1 (-E1)_a + 1 (-E2)_a'), ('1 (-E1)_a + 1 (-E2)_a', '1 (-E)_c')]
for name, rd in diamond.items():
    print(name, "|", tqc.splitting(rd, bands=2).summary().splitlines()[0])
#  strong | R-3m1' #166.98, double-valued: not nsit, the valence bands' invariants being fractional   (.group, .nsit, .valence.theta_Z)
#  weak | R-3m1' #166.98, double-valued: not nsit, the valence bands' invariants being fractional   (.group, .nsit, .valence.theta_Z)
#  dimers | R-3m1' #166.98, double-valued: not nsit, the group having no invariants valued modulo n   (.group, .nsit, .theta_Zn)
```

**Read `nsit` as written.** `not nsit` with two pairs of realizations adding
up does not say the four-band splitting is trivial: any topology it carries is
invisible to the stable real-space invariants. The criterion is sufficient,
never necessary. In the strong and the weak phase the valence side's ℤ-valued
SRSIs are fractional, so by the theorem no atomic insulator has its content: a
nonzero symmetry indicator says so, and the criterion is not needed. In the
dimers the group, `R-3m1'` here, has no ℤₙ-valued SRSI, so there is no sum to
test. In any group, a side with integral ℤ-valued SRSIs and no atomic
realization at all, a fragile side, also makes `nsit` false: no nonnegative sum
of EBRs has its symmetry data vector, which `decompose` sees on that side.
`valence.realizations` and `conduction.realizations` say which side it is, and
[the records reference](../reference/records.md#splitting-and-realizations)
reads one on Kane–Mele, whose upper pair has no realization.
`atomic_realizations` runs the enumeration on its own, and `nsit_criterion` the
criterion on SRSIs already in hand.

The tables list the paper's own splittings of an EBR into EBRs the indicators
allow, each with `srsi_mismatch`, the paper's mark that the ℤₙ-valued
invariants of the parts' EBRs do not add up to the whole's. In lin3g's group
the `2c` EBR splits into the pair at `1a` or the pair at `1b`, and the
criterion on numbers in hand asks the first over every realization of each
part, the whole EBR as all the bands, one part below the gap and the other
above it:

```python
split = tables.split_ebrs[0]
print(split.srsi_mismatch, (split.parent.irrep, split.parent.wyckoff),
      [(child.irrep, child.wyckoff) for child in split.children])
#  True ('-E', '2c') [('-E2', '1a'), ('-E1', '1a')]
whole, below, above = (tables.theta({(part.irrep, part.wyckoff): 1}) for part in (split.parent, *split.children))
print(tqc.nsit_criterion(tables, whole, below.theta_Z, above.theta_Z).nsit)
#  True
```

The mark and the criterion agree. The two parts together are the `1a` pair of
the orbitals above, which shares the `2c` EBR's content and differs from it in
the second ℤ₂ invariant, `(0, 0)` against `(0, 1)`: no pair of realizations of
the two parts adds up to the whole's.

## When it refuses

A refusal is a returned record, falsy, that names the check which stopped the
call ([when a call refuses](refusals.md)):

```python
absent = tqc.srsi_tables(1, spinor=True)
print(absent.gate, len(absent.context["omitted_groups"]))
#  srsi_table_absent 13
rs = diamond["strong"]
fractional = tqc.srsi_of_bands(tqc.hsp_table(rs, bands=2))
rz = tqc.atomic_realizations(tqc.srsi_tables(rs.group), fractional.theta_Z)
print(rz.realizations)                             # a record, with no realization
#  ()
print(rz.summary())
#  R-3m1' #166.98, double-valued: no atomic realization with Z-valued stable real-space invariants theta_1 to theta_8 = (-1/4, 0, 1/2, 1, 1/4, -1/2, -1/4, 1/4)   (.group, .realizations, .theta_Z)
#    fractional   (.theta_Z)
```

- [`srsi_table_absent`](../reference/refusals.md#srsi_table_absent): one of
  the 13 space groups with a single site irrep in the cell, whose one invariant
  is the band count. [`srsi_scope`](../reference/refusals.md#srsi_scope): the
  group is not grey, or a number is outside 1 to 230; a Hamiltonian that breaks
  time reversal is returned a unitary group
  ([time reversal is measured](group.md#time-reversal-is-measured)).
- On fractional ℤ-valued SRSIs, which no atomic insulator has,
  `atomic_realizations` returns its record with no realization.
  [`srsi_realizations_truncated`](../reference/refusals.md#srsi_realizations_truncated):
  the enumeration reached its cap, and a no-go is asserted only on a complete
  one. `atomic_realizations` takes a larger `cap=`; `splitting` and
  `nsit_criterion` run it at
  [its default](../reference/glossary.md#budgets-meshes-and-lengths).
- [`wyckoff_orbit_content_incomplete`](../reference/refusals.md#wyckoff_orbit_content_incomplete),
  [`wyckoff_orbit_not_localizable`](../reference/refusals.md#wyckoff_orbit_not_localizable):
  a Wyckoff orbit's content was not measured at every point, or is a band
  representation of no position its orbitals can move onto.
  [`wyckoff_setting`](../reference/refusals.md#wyckoff_setting): geometry and
  content name different Wyckoff positions, a check the package makes on
  itself; report a model that meets it. The orbital route also meets the checks
  of the Wyckoff letters it reads,
  [`wyckoff_unresolved`](../reference/refusals.md#wyckoff_unresolved) and
  [`tables_not_ready`](../reference/refusals.md#tables_not_ready).
- [`srsi_data`](../reference/refusals.md#srsi_data),
  [`srsi_ebr_basis`](../reference/refusals.md#srsi_ebr_basis): the converted
  tables are not installed (`pythtqc.srsi.srsi_status()` says what is missing),
  or `irreptables` carries another EBR list than they were converted against.
  The band route also meets the
  checks of the content it reads:
  [`multiplet_split`](../reference/refusals.md#multiplet_split),
  [`incomplete_content`](../reference/refusals.md#incomplete_content),
  [`not_a_band_structure`](../reference/refusals.md#not_a_band_structure),
  [`indicator_not_evaluable`](../reference/refusals.md#indicator_not_evaluable).

## Every field and argument

`SRSITables`, `BandSRSIs`, `ModelSRSIs`, `Splitting`, `Realizations` and
`StablePair`, field by field, are in [the records reference](../reference/records.md),
and the calls among [the functions](../reference/functions.md#stable-invariants):
[`srsi_tables`](../reference/functions.md#srsi_tables),
[`srsi_of_bands`](../reference/functions.md#srsi_of_bands),
[`srsi_of_model`](../reference/functions.md#srsi_of_model),
[`splitting`](../reference/functions.md#splitting),
[`nsit_criterion`](../reference/functions.md#nsit_criterion),
[`atomic_realizations`](../reference/functions.md#atomic_realizations).
[The primer](../background/tqc-primer.md#real-space-invariants-and-the-stable-ones)
places the invariants in the literature. Four [examples](../examples.md) run
them end to end: [the tables of every group](../../examples/tqc_srsi_tables.py),
[the detuned diamond's two routes](../../examples/tqc_srsi_fkm_two_routes.py),
[lin3g's two gaps](../../examples/tqc_srsi_lin3g_composite_splitting.py) and
[the two `P41'` models of the paper's supplement](../../examples/tqc_srsi_p4_flatband.py).
