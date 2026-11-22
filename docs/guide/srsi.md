# What do the stable real-space invariants say about these bands?

## What the stable real-space invariants are

An atomic limit is described in real space by the irreps and multiplicities of
its Wannier functions at each occupied Wyckoff position. These are encoded in
the site-symmetry vector `p`. A smooth deformation of the system that preserves
symmetry can move Wannier functions, and hence can change `p`. A real-space
invariant is a combination of the entries of `p` that is invariant under smooth
deformations. The stable real-space invariants (SRSIs) are a complete set of
these that are linear in `p`. Two atomic limits with the same SRSIs can be
smoothly deformed into each other once the same atomic orbitals are added to
both. SRSIs are either integers or integers modulo n.

This package uses two facts about SRSIs. First, the integer-valued SRSIs
(ZSRSIs) are fixed by the symmetry data vector, and a band set has a nonzero
symmetry indicator exactly when the ZSRSIs take fractional values. Second, the
SRSIs valued modulo n are not fixed by the symmetry data vector, and so can only
be computed for full models.

The definitions and proofs are in Hwang *et al.*,
[*Stable real-space invariants and topology beyond symmetry indicators*, Nat. Commun. **17**, 8564 (2026)](https://doi.org/10.1038/s41467-026-74844-w),
whose tables the package ships.
[What elementary band representations are](atomic-limits.md#what-elementary-band-representations-are)
defines the symmetry data vector, the elementary band representations (EBRs) and
fragile sets of bands this page uses.
[Which symmetry-indicator class do these bands occupy?](indicators.md) reads the
symmetry indicator.
[Real-space invariants, and the stable ones](../background/tqc-primer.md#real-space-invariants-and-the-stable-ones),
in [The topological quantum chemistry this package uses](../background/tqc-primer.md),
gives a longer account.

## The call

`srsi_tables` takes a space group: `r.group` of what `realized_group` returned,
or a space-group number with `spinor=`. With
`spinor=True` it reads the tables with spin-orbit coupling, whose site irreps
are double-valued; with `spinor=False`, those without. A number needs no model.
The record returned holds the group's SRSIs as the paper tabulates them, and its
`theta` evaluates them on a site-symmetry vector.

Tables can only be returned for a grey group, a space group with time reversal.
A group without time reversal, or a number outside 1 to 230, is refused with
`srsi_scope`. The 13 space groups whose only Wyckoff position is the general
position have one SRSI, the band count, and no table. They are refused with
`srsi_table_absent`.

The example is `Pmm2`, space group 25, with spin-orbit coupling, read from its
number. Its Wyckoff positions `1a` to `1d` are lines of site symmetry `mm2`. The
example evaluates the SRSIs on one Kramers pair at `1a` and on one at `1b`:

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

- `Pmm21' #25.58, double-valued: 4 stable real-space invariants`: the group,
  `t25.group`, and how many SRSIs it has, `len(t25.invariants)`. `Pmm21'` is
  `Pmm2` with time reversal, and `25.58` its number among the magnetic space
  groups. `double-valued` marks the tables with spin-orbit coupling. Each line
  of a summary ends with the fields that hold it.
- `theta_2   Z2`: one SRSI and its modulus, `t25.moduli["theta_2"]`. `Z` marks a
  ZSRSI, and `Z2` an SRSI valued modulo 2. `Pmm21'` has one ZSRSI and three
  SRSIs valued modulo 2.
- `9 site irreps at Wyckoff positions, 4 of them inducing an elementary band
  representation`: the site irreps that index the entries of `p`,
  `t25.site_irreps`. They are every site irrep the SRSIs count, at maximal
  Wyckoff positions and at the others. A maximal Wyckoff position is one whose
  site-symmetry group lies in no larger site-symmetry group. The tables list an
  EBR only for a site irrep at a maximal position, and some site irreps at
  maximal positions induce none ([From the orbitals](#from-the-orbitals)).
- `no symmetry indicator, the group's indicator group being trivial`: where the
  group has symmetry indicators, this line lists them, `t25.indicators`
  ([From the bands](#from-the-bands) reads them). Two more lines are printed
  where the tables hold them. One counts the paper's splittings of an EBR,
  `split_ebrs` ([The no-go criterion](#the-no-go-criterion)). The other lists the
  pairs of site irreps no SRSI distinguishes, `stably_equivalent`
  ([Site irreps no invariant distinguishes](#site-irreps-no-invariant-distinguishes)).
- `theta_1 (Z)  = 2 (-1E2E)_e + ... + 1 (-E)_a + ...`: from `t25.formulas()`,
  one SRSI written out. Each term `c (irrep)_w` is `c` times the multiplicity of
  a site irrep at the Wyckoff position with letter `w`. A leading `-` marks a
  double-valued site irrep ([Site irreps](../background/conventions.md#site-irreps)).
  `-E` is the Kramers pair on the lines `1a` to `1d`, `-1E2E` the Kramers pair
  on the mirror planes `2e` to `2h`, and `-AA` the Kramers pair at the general
  position `4i`. So `theta_1` counts Kramers pairs per cell, and the number of
  bands is `2 theta_1`.
- `theta_2 (Z2) = 1 (-E)_b`: the number of `-E` at `1b`, modulo 2. Kramers pairs
  leave a line only two or four at a time, onto a mirror plane through it or
  into the general position, so a symmetric deformation leaves the number on
  each line fixed modulo 2. `theta_3` and `theta_4` count those at `1c` and
  `1d`. The number at `1a`, modulo 2, follows from the four SRSIs.
- `Invariants(theta_1 = (1); theta_2 to theta_4 = (0, 0, 0) mod 2)`: what
  `t25.theta(p)` returns for one Kramers pair at `1a`. Its `theta_Z` holds the
  ZSRSIs by name, as exact fractions, and its `theta_Zn` the SRSIs valued modulo
  n. `theta` takes `p` as a mapping `{(site irrep, Wyckoff position):
  multiplicity}`, each Wyckoff position written with its number and letter, as
  `1a`. It also takes a sum of terms, such as a decomposition `decompose`
  returns.
- `Invariants(theta_1 = (1); theta_2 to theta_4 = (1, 0, 0) mod 2)`: one Kramers
  pair at `1b`. The little group at every high-symmetry point of `Pmm2` is `mm2`,
  with one double-valued corep, so the two pairs have the same symmetry data
  vector. They share `theta_1` and differ in `theta_2`, so no symmetric
  deformation takes one into the other. This is the paper's first worked
  example.

The fields of the same tables, and the SRSIs of three Kramers pairs at `1b`:

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

- `25.58`: `t25.group.number_str`, the group's number.
- `{'theta_1': None, 'theta_2': 2, ...}`: `t25.moduli`, each SRSI's modulus,
  `None` for a ZSRSI. For a group with symmetry indicators it holds each
  indicator's modulus too.
- `https://doi.org/10.1038/s41467-026-74844-w`: `t25.provenance.doi`, the paper
  the tables are converted from.
- `[('-E', '1a'), ...]`: the site irreps that induce an EBR, those whose `ebr` is
  true, here the four on the lines.
- `[('-1E2E', '2e', False), ...]`: the site irreps that induce none, here the
  five on the mirror planes and at the general position. Their positions are
  not `maximal`.
- `{'theta_1': Fraction(3, 1), 'theta_2': Fraction(3, 1), ...}`:
  `t25.theta_unreduced(p)`, every SRSI, those valued modulo n not reduced
  modulo n. Three pairs at `1b` give `3` in `theta_2`, where `theta` gives `1`
  modulo 2.

## From the bands

`srsi_of_bands` takes the table `hsp_table` returns for a set of bands. A
symmetry data vector written out by hand, `{corep: multiplicity}`, is passed
with what `realized_group` returned for the model, `r`:
`srsi_of_bands(vector, r)`. It returns the ZSRSIs of the bands. They are
computed by writing the symmetry data vector as a rational sum of EBRs, a sum
with rational multiplicities. Those multiplicities are a site-symmetry vector
`p`, on which the SRSIs are evaluated. Every such sum gives the same ZSRSIs. No
SRSI valued modulo n is returned
([Why the SRSIs valued modulo n need the orbitals](#why-the-srsis-valued-modulo-n-need-the-orbitals)).

ZSRSIs can only be returned if the table holds the little-group representations
at every high-symmetry point, for a layer every high-symmetry point in its
plane. A table that refused at some point is refused with
`incomplete_content`.

The first example is lin3g, from `pythtqc.models`, the spinful square layer of
Lin *et al.*, [Nat. Commun. **15**, 550 (2024)](https://doi.org/10.1038/s41467-024-44762-w),
with two orbitals at Wyckoff position `1a` and two at `2c`. The second is
Fu-Kane-Mele's diamond, from `pythtb.models`, in three phases reached by
distorting its bonds. `fu_kane_mele(t, soc, dt)` puts one spinful orbital on
each of the two sites of the primitive cell, which the example declares an `s`
orbital with `characters=`. Its `dt` holds offsets to the four nearest-neighbor
bonds. The strong phase strengthens the bond along `[111]` by 0.4. The weak
phase weakens it by 0.4. The dimers cut the other three bonds to zero
([A distortion lowers the group](../start/tutorial-3d.md#a-distortion-lowers-the-group)).
The example reads lin3g's six lowest bands and each phase's two valence bands:

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

- `P4mm1' #99.164, double-valued: the Z-valued stable real-space invariants of
  the bands`: the group the bands were read in, `six.group`. `P4mm1'` is `P4mm`
  with time reversal.
- `theta_1 =        1`: one ZSRSI, `six.theta_Z["theta_1"]`, an exact fraction.
- `integral`: every ZSRSI is a whole number. `fractional` is printed otherwise.
- `their symmetry-indicator group is trivial`: the symmetry indicator of the
  same bands, `six.indicator`.
- `strong theta_1 to theta_8 = (-1/4, ...), fractional; class (0, 1) in Z2 x Z4,
  of order 4`: from `b.line()`, the record on one line. It holds the ZSRSIs,
  whether they are fractional, and the class of the bands in `Z2 x Z4`, their
  symmetry indicator. `Z2 x Z4` is the indicator group of `R-3m1'`, the group
  of the distorted diamond.

lin3g's ZSRSIs are integers, as the trivial indicator group of `P4mm1'`
requires. The strong phase's valence pair has ZSRSIs in quarters and a class of
order 4. The weak phase's has halves and a class of order 2. The dimers' are
integers, with the zero class. In each phase the common denominator of the
ZSRSIs is the order of the class. No integer combination of EBRs, even with
negative multiplicities, has the symmetry data vector of a set of bands with a
nonzero symmetry indicator. The symmetry data vector of a fragile set of bands
has one.

The tables of a group with symmetry indicators hold the paper's indicators,
each a linear combination of the SRSIs, by name in `tables.indicators`, and
each indicator's modulus in `tables.moduli`. `indicators_of(p)` evaluates the
indicators on a site-symmetry vector. The example reads the tables of
`R-3m1'`, `t166`, and prints each indicator. It evaluates the indicators on
each phase's `b.numerical.p`, the rational site-symmetry vector on which
`srsi_of_bands` evaluated the ZSRSIs:

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

- `z2w,1 2 equal 2 theta_1 - ...`: an indicator's name, its modulus, its
  `relation`, and the form the paper prints. The three weak indicators have one
  form, since the threefold axis carries each onto the next.
- `equal`: `t166.numerical.indicators[name].relation`. It compares the form the
  paper prints with a form the package derives from the symmetry indicator
  computed in momentum space. It takes one of four values:
    - `equal`: the two forms have the same coefficients.
    - `equivalent`: they have other coefficients, with the same value on every
      set of bands modulo the indicator's modulus.
    - `different`: their values differ on some set of bands.
    - `None`: no form is derived.
- Four forms the paper prints, in space groups 147, 162 and 163 without
  spin-orbit coupling, are `different`
  ([`examples/tqc_srsi_tables.py`](../../examples/tqc_srsi_tables.py) prints
  those of space group 147). They do not vanish on their own group's EBRs, and
  `indicators_of` evaluates the derived form in their place.
- `strong {'z2w,1': 1, 'z2w,2': 1, 'z2w,3': 1, 'z4': 3}`: the strong phase's
  indicators, `z4 = 3` and every weak index 1. The weak phase has every weak
  index 1 and `z4 = 0`, and the dimers have all four zero. These are the values
  `symmetry_indicator` returns for the same bands.

## From the orbitals

`srsi_of_model` takes `r`, what `realized_group` returned. It returns the
model's SRSIs of both kinds, read off its orbitals, with a row for each Wyckoff
orbit, the orbitals the group's operations carry into one another. It measures
the symmetry data vector of each Wyckoff orbit's orbitals from the trace of each
little-group operation over them at every high-symmetry point. That vector is
decomposed into EBRs at the orbit's Wyckoff position. Where the tables list no
EBR there, it is decomposed at the positions the orbitals can be moved onto
along the line or plane of their own position, the positions they slide onto. A
row's SRSIs are those of its decomposition, and the model's are the sum of its
rows'. A model's orbitals are a known configuration of site irreps, so its SRSIs
valued modulo n are returned too. Those of a set of bands are not fixed by its
symmetry data vector
([Why the SRSIs valued modulo n need the orbitals](#why-the-srsis-valued-modulo-n-need-the-orbitals)).

A Wyckoff orbit whose symmetry data vector could not be read at some
high-symmetry point is refused with `wyckoff_orbit_content_incomplete`. One
whose symmetry data vector is a band representation of no position its orbitals
can slide onto is refused with `wyckoff_orbit_not_localizable`.

The example reads lin3g's orbitals. It compares the model's ZSRSIs with those
`srsi_of_bands` reads off all eight bands:

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

- `the whole model: theta_1 to theta_3 = (2, 2, 0); theta_4, theta_5 = (0, 1)
  mod 2`: the model's SRSIs, `mo.theta_Z` and `mo.theta_Zn`, the sum of its
  rows'. The ZSRSIs are integers, as those of a configuration of orbitals are.
- `1a (4mm1'): orbital 0, 1 (-E1)_a: (0, 1, 0); (0, 0) mod 2`: one row of
  `mo.wyckoff_orbits`. It gives the Wyckoff position and its site-symmetry
  group, the orbitals, the site irreps they carry written `m (irrep)_w`, and the
  row's own ZSRSIs and SRSIs valued modulo 2. A row holds every field of the row
  of `r.wyckoff_orbits` for the same orbitals.
- `2c: -GM6 + -GM7 + -M6 + -M7 + 2x-X5`: `orbit.numerical.symmetry_data`, the
  symmetry data vector measured on the row's orbitals. `GM` is the irreptables
  package's name for Γ.
- `True`: the model's ZSRSIs equal those `srsi_of_bands` reads off all eight
  bands, which have the symmetry data vector of the orbitals.

The two orbitals at `1a` together and the pair at `2c` have the same symmetry
data vector and the same ZSRSIs, `(1, 1, 0)`. The second SRSI valued modulo 2,
`theta_5`, tells them apart, as `theta_2` told a Kramers pair at `1a` of `Pmm2`
from one at `1b`.

Each row's decomposition, the first where there are several, is
`orbit.numerical.chosen`. It is a sum of terms, each a multiplicity, a site
irrep and its Wyckoff position. The sum carries its own SRSIs. The example
prints lin3g's SRSIs valued modulo 2 and each row's decomposition:

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

- `{'theta_4': 0, 'theta_5': 1}`: the model's SRSIs valued modulo 2, by name.
  Each entry of `mo.theta_Zn` holds a `value` and a `modulus`.
- `(2, 3) [(1, '-E', '2c')] {'theta_4': 0, 'theta_5': 1}`: the orbitals of the
  `2c` row, its decomposition, one `-E` at `2c`, and the SRSIs valued modulo 2
  of that decomposition.

Where the tables list no EBR at a row's Wyckoff position, its orbitals are moved
along the line or plane the position covers, onto the positions on it where the
tables list one. The example is the strong phase of the diamond. Its orbit sits
on the line `6c`, `(0, 0, z)` of `R-3m`, which holds `3a` at `z = 0` and `3b` at
`z = 1/2`:

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

- `6c (3m1'): orbitals 0-1, 1 (-E1)_c, slides onto 3a and 3b: (0, 0, 1, 1, 0,
  0, 0, 0)`: the row's site irrep at its own position, the positions its
  orbitals are decomposed at, `orbit.slides_onto`, and its ZSRSIs. `R-3m1'` with
  spin-orbit coupling has no SRSI valued modulo n.
- `[('3a', ...), ('3b', ...)]`: `strong.wyckoff_orbits[0].numerical.localizations`,
  every decomposition at the positions the row's orbitals slide onto: one at
  `3a` and one at `3b`, with the same ZSRSIs. They agree because moving the
  orbitals along the line is a symmetric deformation, which changes no SRSI.

A row at a maximal position may carry a site irrep whose band representation is
not elementary, so that the tables list no EBR for it. The tables list the site
irrep itself, and such a row's SRSIs are evaluated on its site irreps at its own
position. Such site irreps occur only in the tables without spin-orbit coupling.

## Why the SRSIs valued modulo n need the orbitals

`srsi_of_bands` returns no SRSI valued modulo n. The symmetry data vector fixes
the ZSRSIs and not those valued modulo n, so two rational sums of EBRs with one
symmetry data vector can carry different SRSIs valued modulo n. The example is
lin3g's bands 4 and 5, a fragile pair whose symmetry data vector is that of the
EBR `-E2 at 1b`
([What a decomposition cannot certify](atomic-limits.md#what-a-decomposition-cannot-certify)).
It evaluates the SRSIs on the sum `srsi_of_bands` used and on the decomposition
`decompose` returns:

```python
t45 = tqc.hsp_table(r, bands=[4, 5])
b45 = tqc.srsi_of_bands(t45)
tables = tqc.srsi_tables(group)
ebrs = [(e.irrep, e.wyckoff) for e in tqc.ebr_data(group).ebrs]   # the group's list of EBRs
print(ebrs)
#  [('-E1', '1a'), ('-E2', '1a'), ('-E1', '1b'), ('-E2', '1b'), ('-E', '2c')]
print([str(b45.numerical.p.get(key, 0)) for key in ebrs], tables.theta(b45.numerical.p))
#  ['1', '1', '-1', '0', '0'] Invariants(theta_1 to theta_3 = (0, 0, 1); theta_4, theta_5 = (0, 0) mod 2)
d45 = tqc.decompose(t45)           # the decompositions of the pair
counts = {(t.irrep, t.wyckoff): t.multiplicity for t in d45.candidate_sums[0]}
print(tuple(counts.get(key, 0) for key in ebrs), tables.theta(d45.candidate_sums[0]))
#  (0, 0, 0, 1, 0) Invariants(theta_1 to theta_3 = (0, 0, 1); theta_4, theta_5 = (1, 0) mod 2)
```

- `[('-E1', '1a'), ...]`: the EBRs of `P4mm1'`, each the site irrep that induces
  it at its Wyckoff position, in the order of the tables of the irreptables
  package.
- `['1', '1', '-1', '0', '0']`: the sum `srsi_of_bands` used, `b45.numerical.p`,
  over those EBRs: `(-E1)_a + (-E2)_a - (-E1)_b`. Its SRSIs valued modulo 2 are
  `(0, 0)`.
- `(0, 0, 0, 1, 0)`: the decomposition `decompose` returns, one `-E2` at `1b`.
  Its SRSIs valued modulo 2 are `(1, 0)`. Both sums have the ZSRSIs `(0, 0, 1)`.

An SRSI valued modulo n is evaluated on a configuration of site irreps: the
model's orbitals, which `srsi_of_model` reads, or one written out, such as an
EBR or a decomposition `decompose` returns. A set of bands that is an atomic
limit has the SRSIs valued modulo n of the configuration of its Wannier
functions, and its symmetry data vector does not say which configuration that
is.

## Which Wyckoff letter is a table key

A Wyckoff letter names a position in one setting of a group, its conventional
cell and its origin. `theta` reads a letter in the setting of the conventional
cell used by the irreptables package, which the SRSI tables share. In `Fd-3m`,
the group of the undistorted diamond, one site irrep carries different SRSIs at
`8a` and at `8b`. `Fd-3m` has two origin choices, and the irreptables package
uses the one that puts `8a` at `(1/8, 1/8, 1/8)` of the conventional cell. The
orbitals of `fu_kane_mele` are at `8a` there. A letter read in another setting
can name the other position, and `theta` then gives the other SRSIs. Each row
of `r.wyckoff_orbits` names its Wyckoff position in the irreptables package's
setting. The example prints lin3g's, and evaluates the SRSIs of `Fd-3m` with
spin-orbit coupling on one `-E1` at `8a` and on one at `8b`:

```python
print([o.wyckoff for o in r.wyckoff_orbits])
#  ['1a', '1a', '2c']
t227 = tqc.srsi_tables(227, spinor=True)           # Fd-3m with spin-orbit coupling
for position in ("8a", "8b"):
    print(position, [str(x) for x in t227.theta({("-E1", position): 1}).theta_Z.values()])
#  8a ['1', '0', '0', '0', '0', '0', '0', '0']
#  8b ['0', '0', '0', '1', '0', '0', '0', '0']
```

- `['1a', '1a', '2c']`: the `wyckoff` of each row of `r.wyckoff_orbits`, in the
  irreptables package's setting, the letter to pass to `theta`.
- `8a ['1', '0', ...]` and `8b ['0', '0', '0', '1', ...]`: the ZSRSIs of one
  `-E1` at each position.

spglib reads a Wyckoff letter in a setting of its own, off the orbital
positions alone. For the positions of `fu_kane_mele` it reads the orbitals at
`8b`. Its group is the group of the positions, which can be larger than
the Hamiltonian's. lin3g's positions have the mirror through the layer's plane,
which its spin-orbit hoppings break. In that larger group, `P4/mmm`, the edge
centers are `2f`, where `P4mm` has `2c`. The letter `r.wyckoff_orbits` prints is
the one to pass to `theta`. `wyckoff_sites_tables(r.group)` returns each site's
Wyckoff position in the irreptables package's setting, with its free parameters
and the positions it slides onto.

## Site irreps no invariant distinguishes

Two site irreps at one maximal Wyckoff position can induce EBRs with the same
symmetry data vector and carry the same SRSIs. The tables' `stably_equivalent`
lists such pairs. The example is the paper's own, `B2` and `B3` at `12d` of
`I432`, space group 211 without spin-orbit coupling:

```python
pair, = tqc.srsi_tables(211, spinor=False).stably_equivalent
print(pair.summary())
#  B2 and B3 at 12d, which no stable real-space invariant distinguishes   (.irreps, .wyckoff)
#    their elementary band representations have the same symmetry data vector   (.symmetry_data)
#    the invariants both carry: theta_1 to theta_7 = (1, 0, 1, 1, 1, -1, 0); theta_8 to theta_10 = (0, 0, 1) mod 2   (.theta_Z, .theta_Zn)
print(pair.irreps, pair.wyckoff)
#  ('B2', 'B3') 12d
```

- `B2 and B3 at 12d, which no stable real-space invariant distinguishes`: the
  pair, `pair.irreps` and `pair.wyckoff`.
- `their elementary band representations have the same symmetry data vector`:
  the vector both EBRs have, `pair.symmetry_data`.
- `the invariants both carry: theta_1 to theta_7 = (1, 0, 1, 1, 1, -1, 0);
  theta_8 to theta_10 = (0, 0, 1) mod 2`: the SRSIs of each member,
  `pair.theta_Z` and `pair.theta_Zn`.

Telling the two apart takes an invariant finer than the stable ones. A row of
`srsi_of_model` whose decompositions use both members of such a pair holds the
pair in its record.

## The no-go criterion

`splitting` takes `r` and `bands=`, the valence bands. They are given as a
number for the lowest bands or as band indices. The conduction bands are the
rest of the model's bands. It returns `nsit`, true where the paper's no-go
criterion shows that the two sets cannot both be atomic limits. The paper calls
an atomic limit an atomic insulator.

All the bands together are an atomic limit, whose Wannier functions are the
model's orbitals. The symmetry data vectors of the valence and the conduction
bands add up to the whole model's, and so do their ZSRSIs. An atomic
realization of a set of bands is a configuration of site irreps at maximal
Wyckoff positions, with nonnegative integer multiplicities, that has the bands'
ZSRSIs. Maximal positions suffice, since an atomic limit moves onto them with
its SRSIs unchanged. If both sets are atomic limits, the SRSIs valued modulo n
of some atomic realization of each add up to the whole model's, modulo n. One
pair of realizations that does not add up shows nothing, so `splitting` lists
every realization of each side. Where both sides have realizations and no pair
adds up, one set or both is not an atomic limit. No symmetry indicator says so,
since both sides have integer ZSRSIs. The paper calls such a splitting
non-symmetry-indicated topological (NSIT).

The whole model's SRSIs are read off its orbitals, as `srsi_of_model` reads
them. The valence bands' ZSRSIs are read off their symmetry data vector, and the
conduction bands' are the difference. The gap is taken as given: only the
symmetry data vectors at the high-symmetry points are read. An enumeration that
reaches its cap of 256 realizations is refused with
`srsi_realizations_truncated`. A realization that uses a member of a pair of
site irreps no SRSI distinguishes holds the pair in its record
([Site irreps no invariant distinguishes](#site-irreps-no-invariant-distinguishes)),
and so does a splitting with such a realization.

The example is lin3g, its six lowest bands against the two above them.
[`examples/tqc_srsi_lin3g_composite_splitting.py`](../../examples/tqc_srsi_lin3g_composite_splitting.py)
measures lin3g's gaps above its fourth and its sixth band.

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

- `nsit, no realization of the valence bands and of the conduction bands adding
  up to the whole model's invariants valued modulo n`: `gap6.nsit`, true. The
  splitting is NSIT. `P4mm1'` has a trivial indicator group, so no symmetry
  indicator could say so.
- `the whole model (2, 2, 0) = valence (1, 1, 1) + conduction (1, 1, -1)`: the
  ZSRSIs of all eight bands, read off the orbitals, `gap6.theta_Z`; those of the
  six valence bands, `gap6.valence.theta_Z`; and their difference, the
  conduction bands', `gap6.conduction.theta_Z`.
- `theta_4, theta_5 = (0, 1) mod 2`: the whole model's SRSIs valued modulo 2,
  `gap6.theta_Zn`.
- `1 (-E2)_b + 1 (-E)_c   (1, 1)`: one atomic realization of the valence bands,
  with its own SRSIs valued modulo 2. The valence bands have three,
  `gap6.valence.realizations`, and they differ modulo 2. The conduction bands
  have one, `1 (-E1)_b`, in `gap6.conduction.realizations`.
- `no pair of a valence and a conduction realization has invariants valued
  modulo n adding up to the whole model's`: `gap6.consistent_pairs` is empty.
  The three pairs add up to `(1, 1)`, `(0, 0)` and `(1, 0)`, and none to the
  `(0, 1)` of all eight bands. So the two sets cannot both be atomic limits.

`decompose` finds three decompositions of the six valence bands, the same three
configurations as their realizations
([Could these bands be an atomic limit?](atomic-limits.md)), and cannot say that
either side is topological. The criterion shows that one side or both is, by
arithmetic on the SRSIs, with no Wilson loop. The same fields, read from the
record:

```python
print(gap6.group.number_str, gap6.nsit, [str(x) for x in gap6.valence.theta_Z.values()],
      {name: x.value for name, x in gap6.theta_Zn.items()})
#  99.164 True ['1', '1', '1'] {'theta_4': 0, 'theta_5': 1}
print([tuple(x.value for x in v.theta_Zn.values()) for v in gap6.valence.realizations],
      [tuple(x.value for x in c.theta_Zn.values()) for c in gap6.conduction.realizations])
#  [(1, 1), (0, 0), (1, 0)] [(0, 0)]
```

- `99.164 True ['1', '1', '1'] {'theta_4': 0, 'theta_5': 1}`: the group's
  number, `nsit`, the valence bands' ZSRSIs and the whole model's SRSIs valued
  modulo 2.
- `[(1, 1), (0, 0), (1, 0)] [(0, 0)]`: the SRSIs valued modulo 2 of each valence
  and each conduction realization, from the `theta_Zn` of each.

The same Hamiltonian with the gap above its fourth band, and the diamond's three
phases with the gap above their second, have `nsit` false for three other
reasons, which the first line of each summary names:

```python
four = tqc.splitting(r, bands=4)
print(four.nsit, [(v.label, c.label) for v, c in four.consistent_pairs])
#  False [('1 (-E)_c', '1 (-E1)_a + 1 (-E2)_a'), ('1 (-E1)_a + 1 (-E2)_a', '1 (-E)_c')]
print("four |", four.summary().splitlines()[0])
#  four | P4mm1' #99.164, double-valued: not nsit, 2 pairs of a valence and a conduction realization adding up to the whole model's invariants valued modulo n   (.group, .nsit, .consistent_pairs)
for name, rd in diamond.items():
    print(name, "|", tqc.splitting(rd, bands=2).summary().splitlines()[0])
#  strong | R-3m1' #166.98, double-valued: not nsit, the valence bands' invariants being fractional   (.group, .nsit, .valence.theta_Z)
#  weak | R-3m1' #166.98, double-valued: not nsit, the valence bands' invariants being fractional   (.group, .nsit, .valence.theta_Z)
#  dimers | R-3m1' #166.98, double-valued: not nsit, the group having no invariants valued modulo n   (.group, .nsit, .theta_Zn)
```

The first line of the summary names one of five cases, checked in this order:

- `not nsit, the valence bands' invariants being fractional`: the strong and the
  weak phase. The valence bands have fractional ZSRSIs, and so a nonzero
  symmetry indicator. No atomic limit has their symmetry data vector.
- `not nsit, the conduction bands having no atomic realization` (or the valence
  bands, or both): a side with integer ZSRSIs and no atomic realization. No sum
  of EBRs with nonnegative multiplicities has its symmetry data vector, so
  `decompose` finds no decomposition of that side. The side is topological, and
  its symmetry indicator is zero. Kane-Mele's two bands above its gap are such
  a side
  ([When there is no decomposition](atomic-limits.md#when-there-is-no-decomposition)).
- `not nsit, the group having no invariants valued modulo n`: the dimers.
  `R-3m1'` with spin-orbit coupling has no SRSI valued modulo n, so the
  criterion has nothing to compare.
- `not nsit, 2 pairs of a valence and a conduction realization adding up to the
  whole model's invariants valued modulo n`: lin3g's four-band splitting, with
  `four.nsit` printed `False`. Each pair adding up is in `four.consistent_pairs`,
  a valence realization and a conduction realization. The SRSIs show nothing
  against both sets being atomic limits, and either side may still be
  topological.
- `nsit`: both sides have realizations and no pair adds up, as for the six-band
  splitting. One set or both is not an atomic limit, and no symmetry indicator
  says so.

`atomic_realizations(tables, theta_Z)` lists every atomic realization with the
ZSRSIs `theta_Z`. `nsit_criterion(tables, theta_total, theta_Z_v, theta_Z_c)`
applies the criterion to SRSIs already in hand: `theta_total` is what
`tables.theta` or `srsi_of_model` returns for all the bands, and `theta_Z_v` and
`theta_Z_c` are the ZSRSIs of the valence and the conduction bands.

The tables list the paper's splittings of an EBR, `tables.split_ebrs`: each an
EBR whose symmetry data vector is the sum of those of two or more other EBRs,
one of them possibly more than once. Each has `srsi_mismatch`, the paper's mark
that the SRSIs valued modulo n of the parts, each as often as the splitting
lists it, do not add up to the whole's. In lin3g's group the EBR of `-E` at
`2c` splits into the pair at `1a` or the pair at `1b`. The example applies
`nsit_criterion` to the first, with the `2c` EBR as all the bands, the `-E2` at
`1a` as the valence bands and the `-E1` at `1a` as the conduction bands:

```python
split = tables.split_ebrs[0]
print(split.srsi_mismatch, (split.parent.irrep, split.parent.wyckoff),
      [(child.irrep, child.wyckoff) for child in split.children])
#  True ('-E', '2c') [('-E2', '1a'), ('-E1', '1a')]
whole, below, above = (tables.theta({(part.irrep, part.wyckoff): 1}) for part in (split.parent, *split.children))
print(tqc.nsit_criterion(tables, whole, below.theta_Z, above.theta_Z).nsit)
#  True
```

- `True ('-E', '2c') [('-E2', '1a'), ('-E1', '1a')]`: `split.srsi_mismatch`, the
  whole, `split.parent`, and the parts, `split.children`.
- `True`: the criterion's `nsit`, which agrees with the mark.

## When it refuses

A refusal is a returned record. It is falsy, and names the check that stopped
the call and what it measured
([What does a refusal mean, and what do I do about it?](refusals.md)). The
example asks for the tables of space group 1, and for the atomic realizations
of the strong phase's valence pair:

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

- `srsi_table_absent 13`: the check, and how many space groups the tables leave
  out, `absent.context["omitted_groups"]`.
- `()`: `rz.realizations`. `atomic_realizations` handed fractional ZSRSIs,
  which no atomic limit has, returns its record with no realization and is not
  refused. Its summary's second line, `fractional`, says why. The enumeration
  is complete, so an empty list with integer ZSRSIs means that no atomic limit
  has them. A set of bands with them is topological, though its symmetry
  indicator is zero.

The checks these calls meet:

- [`srsi_table_absent`](../reference/refusals.md#srsi_table_absent): one of the
  13 space groups whose only Wyckoff position is the general position, whose
  one SRSI is the band count.
- [`srsi_scope`](../reference/refusals.md#srsi_scope): the group is not grey,
  or a number is outside 1 to 230. `realized_group` returns a group without
  time reversal for a Hamiltonian that breaks it
  ([Time reversal is measured](group.md#time-reversal-is-measured)).
- [`srsi_realizations_truncated`](../reference/refusals.md#srsi_realizations_truncated):
  the enumeration reached its cap, and the criterion is applied only to a
  complete enumeration. `atomic_realizations` takes a larger `cap=`.
  `splitting` and `nsit_criterion` run it at its default, 256
  ([Budgets, meshes and lengths](../reference/glossary.md#budgets-meshes-and-lengths)).
- [`wyckoff_orbit_content_incomplete`](../reference/refusals.md#wyckoff_orbit_content_incomplete):
  a Wyckoff orbit's symmetry data vector could not be read at some
  high-symmetry point.
- [`wyckoff_orbit_not_localizable`](../reference/refusals.md#wyckoff_orbit_not_localizable):
  a Wyckoff orbit's symmetry data vector is a band representation of no
  position its orbitals can slide onto.
- [`wyckoff_setting`](../reference/refusals.md#wyckoff_setting),
  [`wyckoff_unresolved`](../reference/refusals.md#wyckoff_unresolved): the
  Wyckoff position of a Wyckoff orbit read from its geometry and the one read
  from its symmetry data vector differ, or a site could not be matched to a
  Wyckoff position of the group. These are checks the package makes on itself.
  Report a model that meets one.
- [`srsi_data`](../reference/refusals.md#srsi_data): the converted tables are
  not installed. `pythtqc.srsi.srsi_status()` says what is missing.
- [`srsi_ebr_basis`](../reference/refusals.md#srsi_ebr_basis): the installed
  `irreptables` lists other EBRs than the tables were converted against.
- [`tables_not_ready`](../reference/refusals.md#tables_not_ready): the group's
  cell could not be matched to the setting the tables of the irreptables
  package are written in, so no table can be read for it.
- [`ebr_tables`](../reference/refusals.md#ebr_tables): those tables hold no
  EBRs for this group.
- [`irrep_tables`](../reference/refusals.md#irrep_tables): their character
  tables were not loaded, or hold no irrep table for this group.
- [`ebr_convention`](../reference/refusals.md#ebr_convention): their EBRs of
  this group cannot be carried into this package's phase convention. Report
  the group.
- [`incomplete_content`](../reference/refusals.md#incomplete_content): the
  table `srsi_of_bands` or `splitting` reads refused at some point. The refusal
  a point holds is often
  [`multiplet_split`](../reference/refusals.md#multiplet_split), where the set
  of bands cuts a degenerate multiplet there.
- [`not_a_band_structure`](../reference/refusals.md#not_a_band_structure): the
  symmetry data vector fails the group's compatibility relations.
- [`indicator_not_evaluable`](../reference/refusals.md#indicator_not_evaluable):
  a layer's little-group representations out of its plane are not fixed by
  those in it, and could not be read.
- [`unknown_label`](../reference/refusals.md#unknown_label),
  [`non_integer_multiplicity`](../reference/refusals.md#non_integer_multiplicity),
  [`negative_multiplicity`](../reference/refusals.md#negative_multiplicity): a
  symmetry data vector written out by hand names a little-group representation
  the group's EBRs do not carry, or holds an entry that is not a nonnegative
  integer.

## Where to go next

- [Returned records](../reference/records.md) lists every field of
  [`SRSITables`](../reference/records.md#srsitables),
  [`BandSRSIs`](../reference/records.md#bandsrsis),
  [`ModelSRSIs`](../reference/records.md#modelsrsis),
  [`Splitting` and `Realizations`](../reference/records.md#splitting-and-realizations)
  and [`StablePair`](../reference/records.md#stablepair).
- [Functions](../reference/functions.md) gives the arguments of
  [`srsi_tables`](../reference/functions.md#srsi_tables),
  [`srsi_of_bands`](../reference/functions.md#srsi_of_bands),
  [`srsi_of_model`](../reference/functions.md#srsi_of_model),
  [`splitting`](../reference/functions.md#splitting),
  [`nsit_criterion`](../reference/functions.md#nsit_criterion) and
  [`atomic_realizations`](../reference/functions.md#atomic_realizations).
- [Which symmetry-indicator class do these bands occupy?](indicators.md) reads
  the class of a set of bands modulo atomic limits.
- [The examples](../examples.md) runs the SRSIs end to end:
  [`examples/tqc_srsi_tables.py`](../../examples/tqc_srsi_tables.py), the tables
  of every group;
  [`examples/tqc_srsi_fkm_two_routes.py`](../../examples/tqc_srsi_fkm_two_routes.py),
  the diamond's SRSIs from its bands and from its orbitals;
  [`examples/tqc_srsi_lin3g_composite_splitting.py`](../../examples/tqc_srsi_lin3g_composite_splitting.py),
  lin3g's two gaps;
  [`examples/tqc_srsi_p4_flatband.py`](../../examples/tqc_srsi_p4_flatband.py),
  the two `P41'` models of the paper's supplement; and
  [`examples/tqc_srsi_sg212_photonic.py`](../../examples/tqc_srsi_sg212_photonic.py),
  the SRSIs of photonic bands.
- [The topological quantum chemistry this package uses](../background/tqc-primer.md)
  places the SRSIs in the literature.
