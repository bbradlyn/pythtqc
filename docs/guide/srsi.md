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
[symmetry vector](labels.md#every-high-symmetry-point-at-once).

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
the symmetry vector alone: two configurations share them exactly when they
share their content at every high-symmetry point. So every band set has
ℤ-valued SRSIs, an atomic limit or not: those of any rational `p` whose content
is the band set's. And, the paper's theorem, a band set has a nonzero symmetry
indicator exactly when its ℤ-valued SRSIs are fractional. The ℤₙ-valued ones
are no function of the symmetry vector, which makes the SRSIs finer than an
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
#  Pmm21' #25.58, double-valued: stable real-space invariants Z x Z2 x Z2 x Z2   (.structure)
#    each invariant, Z- or Z_n-valued, and the number of terms in its formula   (.rows)
#      theta_1    Z  9 terms
#      theta_2   Z2  1 term
#      theta_3   Z2  1 term
#      theta_4   Z2  1 term
#    9 site irreps at Wyckoff positions, 4 of them inducing an elementary band representation   (.site_keys, .ebrs)
#    5 terms on site irreps that induce a composite band representation   (.terms_without_ebr_column)
#    no symmetry indicator, the group's indicator group being trivial   (.indicators)
print(t25.formulas())                              # each invariant, term by term
#  each invariant, Z- or Z_n-valued, as a sum of terms c (irrep)_w, c times the multiplicity of a site irrep at Wyckoff position w   (.rows)
#    theta_1 (Z)  = 2 (-1E2E)_e + 2 (-1E2E)_f + 2 (-1E2E)_g + 2 (-1E2E)_h + 4 (-AA)_i + 1 (-E)_a + 1 (-E)_b + 1 (-E)_c + 1 (-E)_d
#    theta_2 (Z2) = 1 (-E)_b
#    theta_3 (Z2) = 1 (-E)_c
#    theta_4 (Z2) = 1 (-E)_d
print(t25.theta({("-E", "a"): 1}))                 # one Kramers pair at 1a
#  ((Fraction(1, 1),), {2: (0, 0, 0)})
print(t25.theta({("-E", "b"): 1}))                 # ... or at 1b
#  ((Fraction(1, 1),), {2: (1, 0, 0)})
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
- `theta(p)` takes `p` as a mapping `{(site irrep, Wyckoff letter):
  multiplicity}` and returns `(theta_Z, theta_Zn)`: the ℤ-valued SRSIs as
  exact fractions, the ℤₙ-valued ones as `{n: values}` modulo `n`. A Kramers
  pair at `1a` and one at `1b` share `theta_1` and differ in `theta_2`, the
  paper's first worked example. `theta_columns` takes `p` as a sequence over
  the group's EBR columns instead, the shape a decomposition writes it in,
  exact for a configuration on those positions alone. `tqc.srsi_of(group, p)`
  and `tqc.srsi_of_columns(group, p)` are the same two evaluations on a
  measured group.

The same tables, field by field:

```python
print(t25.number_str, t25.structure, t25.provenance["doi"])
#  25.58 Z x Z2 x Z2 x Z2 https://doi.org/10.1038/s41467-026-74844-w
print([(key.irrep, key.letter, key.column) for key in t25.site_keys if key.ebr])
#  [('-E', 'a', 0), ('-E', 'b', 1), ('-E', 'c', 2), ('-E', 'd', 3)]
print([(key.irrep, key.letter, key.maximal) for key in t25.site_keys if not key.ebr])
#  [('-1E2E', 'e', False), ('-1E2E', 'f', False), ('-1E2E', 'g', False), ('-1E2E', 'h', False), ('-AA', 'i', False)]
print(t25.band_count_form, t25.zn_rows)
#  (Fraction(2, 1),) {2: (1, 2, 3)}
p = {("-E", "b"): 3}                               # three Kramers pairs at 1b
print(tqc.srsi_of(t25, p), t25.theta_unreduced(p))
#  ((Fraction(3, 1),), {2: (1, 0, 0)}) (Fraction(3, 1), Fraction(3, 1), Fraction(0, 1), Fraction(0, 1))
print(t25.theta_unreduced_columns([0, 3, 0, 0]) == t25.theta_unreduced(p))   # the same p over the columns
#  True
```

`number_str` and `structure` name the group and its SRSI group, and
`provenance["doi"]` the paper the tables are converted from. A site key is a
site irrep at a Wyckoff letter, marked `maximal` where its position is and
`ebr` where it induces an EBR, `column` being its place among the columns
`theta_columns` reads: here the four `Ē` on the lines, and none of the five
keys on the mirror planes and at the general position. `band_count_form` is the
number of bands as a form in the ℤ-valued invariants, `2·theta_1`, and
`zn_rows` the places of the ℤₙ-valued rows by modulus, counted from 0:
`theta_2` to `theta_4`, modulo 2. `srsi_of(t25, p)` is `t25.theta(p)` as a call;
`theta_unreduced(p)` reads every row with the ℤₙ ones unreduced, three pairs at
`1b` giving `3` in `theta_2` where `theta` gives `1` modulo 2; and
`theta_unreduced_columns` reads `p` over the columns.

## From the bands

`srsi_of_bands` reads the ℤ-valued SRSIs off a band set's symmetry vector `y`
([labelling the bands](labels.md#every-high-symmetry-point-at-once)): it solves
`BR·p = y` over the rationals, `BR` being
[the EBR matrix](../background/tqc-primer.md#how-a-symmetry-indicator-is-computed),
and evaluates `Θ` on that `p`. lin3g's six lowest bands, and the detuned
diamond's three phases:

```python
from pythtqc import models
from pythtb.models import fu_kane_mele

r = tqc.realized_group(models.lin3g(), typat=[0, 0, 1, 1])
model, group, assignment, validated, rep = r
six = tqc.srsi_of_bands(tqc.hsp_table(rep, bands=6))
print(six.summary())
#  P4mm1' #99.164, double-valued: the Z-valued stable real-space invariants of 6 bands   (.theta_Z)
#      theta_1 =        1
#      theta_2 =        1
#      theta_3 =        1
#    integral: the bands are not stable topological   (.stable)
#    their symmetry-indicator group is trivial   (.indicator)
phases = {"strong": [0.4, 0.0, 0.0, 0.0], "weak": [-0.4, 0.0, 0.0, 0.0],
          "dimers": [0.0, -1.0, -1.0, -1.0]}
diamond = {name: tqc.realized_group(fu_kane_mele(1.0, 0.125, dt), characters={0: "s"})
           for name, dt in phases.items()}
valence = {}                                       # each phase's valence pair
for name, rd in diamond.items():
    b = valence[name] = tqc.srsi_of_bands(tqc.hsp_table(rd.rep, bands=2))
    print(name, b.line())                          # the record on one line
#  strong theta_Z = (-1/4, 0, 1/2, 1, 1/4, -1/2, -1/4, 1/4), fractional with common denominator 4; class (0, 1) in Z2 x Z4, of order 4
#  weak theta_Z = (0, 0, 1, 1, 1/2, 0, -1/2, 0), fractional with common denominator 2; class (1, 0) in Z2 x Z4, of order 2
#  dimers theta_Z = (0, 0, 0, 1, 0, 0, 0, 0), integral; class zero in Z2 x Z4
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

The tables carry the paper's indicators as forms in these invariants. Each
form has its `name`, its modulus `mod`, its coefficient on each `theta_i` keyed
by `i` (`theta`), and `relation`, how the printed form compares with the one
derived from the momentum-space indicator. `indicators_of(p)` evaluates them
on a site-symmetry vector, and on the band route's `p`, written over the EBR
columns whose site keys `t166.ebrs` lists, they give each phase's class in the
paper's names:

```python
t166 = tqc.srsi_tables(diamond["strong"].group)
for form in t166.indicators:
    print(form.name, form.mod, form.relation, form.theta)
#  z2w,1 2 equal {1: Fraction(2, 1), 2: Fraction(-2, 1), 3: Fraction(2, 1), 4: Fraction(-2, 1), 7: Fraction(-2, 1)}
#  z2w,2 2 equal {1: Fraction(2, 1), 2: Fraction(-2, 1), 3: Fraction(2, 1), 4: Fraction(-2, 1), 7: Fraction(-2, 1)}
#  z2w,3 2 equal {1: Fraction(2, 1), 2: Fraction(-2, 1), 3: Fraction(2, 1), 4: Fraction(-2, 1), 7: Fraction(-2, 1)}
#  z4 4 equal {1: Fraction(-4, 1), 2: Fraction(4, 1), 3: Fraction(-4, 1), 4: Fraction(4, 1)}
for name, b in valence.items():
    print(name, t166.indicators_of(dict(zip(t166.ebrs, b.p))))
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
It measures the content of each site orbit's own orbitals, the trace of each
little-group operation over them at every high-symmetry point, and decomposes
it over the EBRs of the orbit's position:

```python
mo = tqc.srsi_of_model(r)
print(mo.summary())
#  P4mm1' #99.164, double-valued: stable real-space invariants, Z x Z x Z x Z2 x Z2   (.structure)
#    the whole model: Z-valued (2, 2, 0), mod 2 (0, 1)   (.theta_Z, .theta_Zn)
#    each site orbit, its content written m (irrep)_w, and its SRSIs   (.orbits)
#      orbitals 0-1 at 1a, 1 (-E1)_a + 1 (-E2)_a: Z-valued (1, 1, 0), mod 2 (0, 0)
#      orbitals 2-3 at 2c, 1 (-E)_c: Z-valued (1, 1, 0), mod 2 (0, 1)
for orbit in mo.orbits:
    print(f"{orbit.multiplicity}{orbit.letter}:", tqc.format_counts(orbit.content))
#  1a: -GM6 + -GM7 + -M6 + -M7 + 2×-X5
#  2c: -GM6 + -GM7 + -M6 + -M7 + 2×-X5
eight = tqc.srsi_of_bands(tqc.hsp_table(rep, bands=8))
print(eight.theta_Z == mo.theta_Z)
#  True
```

The model's SRSIs come first, the sum of its orbits', and each site orbit
follows with its orbitals, its Wyckoff position, the site irreps its content
decomposes into there, and its own SRSIs, ℤ-valued and then ℤ₂-valued. The
`1a` pair and the `2c` pair have the same content at every high-symmetry point
and the same ℤ-valued SRSIs, and the second ℤ₂ invariant tells them apart, as
the tables of `Pmm2` told `1a` from `1b`. The model's ℤ-valued SRSIs are
integral, as a configuration of orbitals must be, and equal what the band route
reads off all eight bands: the identity that ties the two routes together.

The record holds the fields the summary names, and for each orbit its
`orbitals` and `chosen`, the localization returned, with its `terms`, each a
multiplicity, an EBR and its position as the group's list labels them
([the band-representation key](../background/conventions.md#the-band-representation-key)),
and its own invariants. `localizations` holds every localization in that form,
here one per orbit:

```python
print(mo.structure, mo.theta_Zn)
#  Z x Z x Z x Z2 x Z2 {2: (0, 1)}
for orbit in mo.orbits:
    print(orbit.orbitals, orbit.chosen["terms"], orbit.chosen["theta_Zn"])
#  (0, 1) ((1, '-E1↑G(2)', "1a(4mm1',4mm)"), (1, '-E2↑G(2)', "1a(4mm1',4mm)")) {2: (0, 0)}
#  (2, 3) ((1, '-E↑G(4)', "2c(mm21',mm2)"),) {2: (0, 1)}
```

An orbit at a position the tables induce no EBR from slides onto one. The
detuned diamond's orbit sits on the line `6c`, `(0, 0, z)` of `R-3m`; the
positions on that line the tables do induce EBRs from are `3a`, at `z = 0`,
and `3b`, at `z = ½`, and the summary writes the orbit's content at the one it
slid onto:

```python
strong = tqc.srsi_of_model(diamond["strong"])
print(strong.summary())
#  R-3m1' #166.98, double-valued: stable real-space invariants, Z x Z x Z x Z x Z x Z x Z x Z   (.structure)
#    the whole model: Z-valued (0, 0, 1, 1, 0, 0, 0, 0)   (.theta_Z)
#    each site orbit, its content written m (irrep)_w, and its SRSIs   (.orbits)
#      orbitals 0-1 at 6c, slid onto 3a, 1 (-E1g)_a + 1 (-E1u)_a: Z-valued (0, 0, 1, 1, 0, 0, 0, 0)
print([(loc["letter"], tuple(str(x) for x in loc["theta_Z"]))
       for loc in strong.orbits[0].localizations])
#  [('a', ('0', '0', '1', '1', '0', '0', '0', '0')), ('b', ('0', '0', '1', '1', '0', '0', '0', '0'))]
```

It localizes at both with the same SRSIs, and that agreement is what an SRSI
is: a number no symmetric move changes. An orbit whose site irreps induce a
band representation that is a sum of others, which no EBR column carries, is
read through its site irreps instead, which the tables carry as keys of their
own.

## Why the ℤₙ half needs the orbitals

`srsi_of_bands` returns no ℤₙ-valued SRSIs: they are no function
of the symmetry vector, so a `p` solved back out of a band set's content
carries its own ℤₙ values and not the band set's. lin3g's bands 4 and 5, a
fragile pair with the content of the EBR `-E2↑G(2)@1b`
([what a decomposition cannot certify](atomic-limits.md#what-a-decomposition-cannot-certify)),
show it:

```python
t45 = tqc.hsp_table(rep, bands=[4, 5])
b45 = tqc.srsi_of_bands(t45)
tables = tqc.srsi_tables(group)
print(tables.ebrs)
#  (('-E1', 'a'), ('-E2', 'a'), ('-E1', 'b'), ('-E2', 'b'), ('-E', 'c'))
print([str(x) for x in b45.p], tables.theta_columns(list(b45.p)))
#  ['1', '1', '-1', '0', '0'] ((Fraction(0, 1), Fraction(0, 1), Fraction(1, 1)), {2: (0, 0)})
d45 = tqc.decompose(t45)           # the EBR the pair's content matches
print(d45.solutions[0], tables.theta_columns(list(d45.solutions[0])))
#  (0, 0, 0, 1, 0) ((Fraction(0, 1), Fraction(0, 1), Fraction(1, 1)), {2: (1, 0)})
```

The band route's `p`, `(-E1)_a + (-E2)_a − (-E1)_b`, is one rational solution
of `BR·p = y` among many and reads ℤ₂ values `(0, 0)`; the solution `decompose`
returns, one `-E2` at `1b` over the same columns, reads `(1, 0)`; both read the
ℤ-valued `(0, 0, 1)`. Read a ℤₙ value only for a configuration that is known:
the one `srsi_of_model` reads off the orbitals, an EBR column, or a solution
`decompose` returned.

## Which Wyckoff letter is a table key

A Wyckoff letter is a key into the tables, and it belongs to a group and a
setting: one site irrep at `8a` and at `8b` of `Fd-3m`, the undistorted
diamond's group, carries different invariants. `Fd-3m` has two origin choices,
and the tables use the one that puts `8a` at `(⅛, ⅛, ⅛)` of the conventional
cell; the diamond's orbitals are `8a` there
([the 3D tutorial](../start/tutorial-3d.md)). The record measures three
letters:

```python
print(assignment.wyckoff_tables, assignment.wyckoff_spglib,
      assignment.wyckoff_spglib_dataset)
#  ('a', 'c', 'c') ('a', 'c') ('a', 'f', 'f')
t227 = tqc.srsi_tables(227, spinor=True)           # Fd-3m with spin-orbit coupling
for letter in "ab":
    print(letter, [str(x) for x in t227.theta({("-E1", letter): 1})[0]])
#  a ['1', '0', '0', '0', '0', '0', '0', '0']
#  b ['0', '0', '0', '1', '0', '0', '0', '0']
```

`wyckoff_tables`, one letter per site, is in the setting the tables are written
in, and every table lookup uses it. `wyckoff_spglib`, one per orbit, is the
same group's letter in spglib's standardization. `wyckoff_spglib_dataset` is
spglib's letter for the orbital positions alone, whose group can be larger than
the Hamiltonian's: lin3g's positions have the mirror through the layer's plane,
which its spin–orbit hoppings break, and in that larger group the edge centres
are `2f`. Only the first is a table key; `wyckoff_sites_tables(group)` returns
its full record.

Two site irreps at one maximal position can be indistinguishable to everything
this package measures: their EBRs have identical symmetry vectors, and their
SRSIs are identical too. `stably_equivalent_pairs` reports such pairs; the
paper's own example is `B2` and `B3` at `12d` of `I432`:

```python
pair, = tqc.stably_equivalent_pairs(tqc.srsi_tables(211, spinor=False))
print(pair.summary())
#  I4321' #211.57, single-valued: B2 and B3 at Wyckoff position d, which no stable real-space invariant distinguishes   (.irreps, .letter)
#    their elementary band representations, columns 15 and 16 of the group's list, have the same content at every high-symmetry point   (.names, .columns, .symmetry_vector)
#    the invariants both carry: Z-valued (1, 0, 1, 1, 1, -1, 0), Z_n-valued mod 2 (0, 0, 1)   (.theta_Z, .theta_Zn)
print(pair.irreps, pair.letter, pair.names)
#  ('B2', 'B3') d ('B2↑G(6)', 'B3↑G(6)')
```

The pair's `irreps` and `letter` are the summary's first line; `names` labels
its two EBRs as the group's list does, at the `columns` the second line gives.
The SRSIs do not distinguish the two, which is not to say they are the same:
telling them apart would take an invariant finer than the stable ones, and
whichever member a decomposition picks, no number this package returns moves.

## The no-go criterion

The one verdict the SRSIs issue is the paper's no-go criterion for a gapped
splitting of a model's bands into a valence set and a conduction set. All the
bands together are an atomic insulator, the model's orbitals being its Wannier
functions, so the symmetry vectors add, the valence set's plus the conduction
set's, and so do the ℤ-valued SRSIs. If the valence and the conduction set are
both atomic insulators, the ℤₙ-valued SRSIs add up too, modulo `n`, for some
**atomic realizations** of each, configurations of orbitals that carry their
symmetry vectors; if no pair of realizations adds up, one set or both is not an
atomic insulator, and the gap carries topology. One realization that fails to
add up proves nothing, so `splitting` enumerates them all. It takes the gap as
given, reading content at the high-symmetry points only; lin3g's gaps above its
fourth and its sixth band are measured in
[`examples/tqc_srsi_lin3g_composite_splitting.py`](../../examples/tqc_srsi_lin3g_composite_splitting.py).

```python
gap6 = tqc.splitting(r, bands=6)                   # six bands against two
print(gap6.summary())
#  P4mm1' #99.164, double-valued, diagnosed topological: the two sides of the gap cannot both be atomic insulators   (.verdict)
#    the bands=6 valence bands, the remaining 2 conduction bands, and all 8 bands   (.n_bands_v, .n_bands_c)
#    Z-valued stable real-space invariants: all 8 (2, 2, 0) = valence (1, 1, 1) + conduction (1, 1, -1)   (.theta_Z_total, .theta_Z_v, .theta_Z_c)
#    Z_n-valued invariants of all 8: mod 2 (0, 1)   (.theta_Zn_total)
#    the valence bands' symmetry-indicator group is trivial   (.indicator)
#    possible atomic realizations of the valence bands, and their Z_n-valued SRSIs   (.realizations_v)
#      1 (-E2)_b + 1 (-E)_c   mod 2 (1, 1)
#      1 (-E1)_b + 2 (-E2)_b   mod 2 (0, 0)
#      1 (-E1)_a + 1 (-E2)_a + 1 (-E2)_b   mod 2 (1, 0)
#    possible atomic realizations of the conduction bands, and their Z_n-valued SRSIs   (.realizations_c)
#      1 (-E1)_b   mod 2 (0, 0)
#    no pair of a valence and a conduction realization has Z_n-valued invariants adding up to those of all 8   (.consistent_pairs)
```

- `diagnosed topological`: the criterion finds the splitting topological, and
  no symmetry indicator says so, the valence bands' indicator group being
  trivial: the paper calls it non-symmetry-indicated topological (NSIT). The
  last line is the reason.
- The invariants of all 8 bands are the model's, read off the orbitals above;
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

Each line ends with the fields it is read from; `number_str` is the group,
`nsit` whether the verdict is `diagnosed topological`, and `realizations_v` and
`realizations_c` hold each side's enumeration, every realization with its own
`theta_Zn`:

```python
print(gap6.number_str, gap6.nsit, [str(x) for x in gap6.theta_Z_v], gap6.theta_Zn_total)
#  99.164 True ['1', '1', '1'] {2: (0, 1)}
print([v.theta_Zn[2] for v in gap6.realizations_v.realizations],
      [c.theta_Zn[2] for c in gap6.realizations_c.realizations])
#  [(1, 1), (0, 0), (1, 0)] [(0, 0)]
```

Each of the valence bands' three realizations with the conduction bands' one
adds up to `(1, 1)`, `(0, 0)` or `(1, 0)`, and none to the `(0, 1)` of all 8.

The same Hamiltonian with the gap two bands lower, and the detuned diamond,
read the other three verdicts:

```python
four = tqc.splitting(r, bands=4)
print(four.verdict, four.consistent_pairs)
#  invisible to the SRSIs ((0, 2), (2, 0))
for name, rd in diamond.items():
    print(name, "|", tqc.splitting(rd, bands=2).verdict)
#  strong | stable topological
#  weak | stable topological
#  dimers | no Z_n SRSIs
```

**Read the verdicts as written.** `invisible to the SRSIs` does not say the
four-band splitting is trivial: two pairs of realizations add up, so any
topology it carries is invisible to the stable real-space invariants. The
criterion is sufficient, never necessary. `stable topological`, in the strong
and the weak phase: the valence side's ℤ-valued SRSIs are fractional, so by the
theorem no atomic insulator has its content, and the criterion is not needed.
`no Z_n SRSIs`: the group, `R-3m1'` here, has no ℤₙ-valued SRSI, so there is no
sum to test. In any group, a side with integral ℤ-valued SRSIs and no atomic
realization at all, a fragile side, also makes a splitting
`diagnosed topological`; `v_atomic_possible` and `c_atomic_possible` say which
of the two a diagnosis is, and
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
print(split.srsi_mismatch, (split.parent["irrep"], split.parent["letter"]),
      [(child["irrep"], child["letter"]) for child in split.children])
#  True ('-E', 'c') [('-E2', 'a'), ('-E1', 'a')]
whole, below, above = (tables.theta({(part["irrep"], part["letter"]): 1}) for part in (split.parent, *split.children))
print(tqc.nsit_criterion(tables, whole, below[0], above[0]).nsit)
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
fractional = tqc.srsi_of_bands(tqc.hsp_table(rs.rep, bands=2))
print(tqc.atomic_realizations(tqc.srsi_tables(rs.group), fractional.theta_Z).gate)
#  srsi_not_atomic
```

- [`srsi_table_absent`](../reference/refusals.md#srsi_table_absent): one of
  the 13 space groups with a single site irrep in the cell, whose one invariant
  is the band count. [`srsi_scope`](../reference/refusals.md#srsi_scope): the
  group is not grey, or a number is outside 1 to 230; a Hamiltonian that breaks
  time reversal is returned a unitary group
  ([time reversal is measured](group.md#time-reversal-is-measured)).
- [`srsi_not_atomic`](../reference/refusals.md#srsi_not_atomic): fractional
  ℤ-valued SRSIs, which no atomic insulator has, so nothing to enumerate.
  [`srsi_realizations_truncated`](../reference/refusals.md#srsi_realizations_truncated):
  the enumeration reached its cap, and a no-go is asserted only on a complete
  one. `atomic_realizations` takes a larger `cap=`; `splitting` and
  `nsit_criterion` run it at
  [its default](../reference/glossary.md#budgets-meshes-and-lengths).
- [`orbit_content_incomplete`](../reference/refusals.md#orbit_content_incomplete),
  [`orbit_not_localizable`](../reference/refusals.md#orbit_not_localizable):
  an orbit's content was not measured at every point, or is a band
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
[`srsi_of`](../reference/functions.md#srsi_of),
[`srsi_of_columns`](../reference/functions.md#srsi_of_columns),
[`srsi_of_bands`](../reference/functions.md#srsi_of_bands),
[`srsi_of_model`](../reference/functions.md#srsi_of_model),
[`splitting`](../reference/functions.md#splitting),
[`nsit_criterion`](../reference/functions.md#nsit_criterion),
[`atomic_realizations`](../reference/functions.md#atomic_realizations),
[`stably_equivalent_pairs`](../reference/functions.md#stably_equivalent_pairs).
[The primer](../background/tqc-primer.md#real-space-invariants-and-the-stable-ones)
places the invariants in the literature. Four [examples](../examples.md) run
them end to end: [the tables of every group](../../examples/tqc_srsi_tables.py),
[the detuned diamond's two routes](../../examples/tqc_srsi_fkm_two_routes.py),
[lin3g's two gaps](../../examples/tqc_srsi_lin3g_composite_splitting.py) and
[the two `P41'` models of the paper's supplement](../../examples/tqc_srsi_p4_flatband.py).
