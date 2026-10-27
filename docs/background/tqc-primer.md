# The topological quantum chemistry this package uses

This page is the background the guide leans on: the ideas of topological
quantum chemistry the package computes with, a section each, for a reader who
knows band theory, space groups and Wyckoff positions and has met symmetry
indicators as an idea. Each section names the papers it rests on, and
[citing](../citing.md) lists them all. How the package reads a Hamiltonian is
[its own page](measurement.md), and how the labels and band representations
below are spelled is [conventions](conventions.md).

## Site symmetry and Wyckoff positions

The **site-symmetry group** of a point of a crystal is the set of its space
group's operations that leave the point fixed: a finite group, isomorphic to a
crystallographic point group. Points whose site-symmetry groups are conjugate
in the space group make up one **Wyckoff position**, named by its multiplicity,
the number of its points in the conventional cell, and a letter: `1a`, `2c`,
`8a`. A Wyckoff position can be a point, a line or a plane; every operation of
`P4mm` leaves `z` unchanged, so its `1a` is the line `(0, 0, z)`. It is
**maximal** when its site-symmetry group lies in no larger site-symmetry group
of the crystal. A **site irrep** is an irreducible representation of a site-symmetry
group, and it says how the orbitals at that site transform. The letters belong
to a setting, an origin and a choice of axes, and the same site can carry
different letters in two descriptions of one crystal; the package files every
site under the letter of the setting its tables are written in
([which Wyckoff letter is a table key](../guide/srsi.md#which-wyckoff-letter-is-a-table-key)).
The standard tables are those of the *International Tables for
Crystallography*, Volume A, and of the Bilbao Crystallographic Server
([Aroyo *et al.*, Z. Kristallogr. **221**, 15 (2006)](https://doi.org/10.1524/zkri.2006.221.1.15)).

## Little groups, irreps and coreps

An operation `{R|t}` takes a Bloch state at `k` to one at `Rk`. The operations
with `Rk = k` up to a reciprocal lattice vector make up the **little group** of
`k`, and the states at `k` carry its representations, each energy level an
irrep. The matrix of an operation `g` in the basis of the bands selected at `k`
is its **sewing matrix**, `B(g,k)_mn = ⟨u_m(k)|U(g;k)|u_n(k)⟩`, and its trace
over a level is that level's **character**: characters name the irrep. The
tables list the irreps at each group's high-symmetry points, written `GM` for Γ
and so on, and the **compatibility relations** say how an irrep at a point
splits along the lines that leave it, which constrains the combinations a band
structure can have. On a spinor a rotation by `2π` acts as `−1`, so the states
of a spinful model carry the double-valued irreps of the double group, the
tables "with spin–orbit coupling", and a leading `-` marks a double-valued
label: `-GM6`. A **grey group** is a space group with time reversal adjoined,
written with a trailing `1'` (`P6/mmm1'`) and numbered in the BNS notation for
magnetic groups (`191.234`).
Time reversal is antiunitary, and the irreducible objects of a group with
antiunitary operations are co-representations, **coreps**: time reversal can
glue two unitary irreps into one physical level, and the corep's name spells
the constituents, `-M3M4` gluing `-M3` and `-M4`. A Hamiltonian that breaks time
reversal is returned its unitary group, and its labels are irreps
([time reversal is measured](../guide/group.md#time-reversal-is-measured)).
The theory of these representations is in Bradley and Cracknell, *The
Mathematical Theory of Symmetry in Solids* (Oxford, 1972). The tables are the
Bilbao Crystallographic Server's
([Elcoro *et al.*, J. Appl. Cryst. **50**, 1457 (2017)](https://doi.org/10.1107/S1600576717011712)),
read through the `irrep` package
([Iraola *et al.*, Comput. Phys. Commun. **272**, 108226 (2022)](https://doi.org/10.1016/j.cpc.2021.108226));
on these pages BCS always means that server, never Bardeen, Cooper and
Schrieffer.

## Band representations and atomic limits

Orbitals at the points of a Wyckoff position that transform in a site irrep
`ρ` span Bloch states on which the whole space group acts: the **band
representation** induced from `ρ`, written `ρ↑G`, whose irreps at every `k`
are fixed by `ρ` and the position alone
([Zak, Phys. Rev. Lett. **45**, 1025 (1980)](https://doi.org/10.1103/PhysRevLett.45.1025);
[Zak, Phys. Rev. B **26**, 3010 (1982)](https://doi.org/10.1103/PhysRevB.26.3010)).
A band set is an **atomic limit** when its states are spanned by exponentially
localized, symmetric Wannier functions, so that it carries a band
representation. An **elementary band representation** (EBR) is one that is no
sum of others. Topological quantum chemistry tabulates every EBR of every
space group, with spin–orbit coupling and without, by its irreps at the
high-symmetry points, and so turns the question whether a band set can be an
atomic limit into arithmetic on those tables
([Bradlyn *et al.*, Nature **547**, 298 (2017)](https://doi.org/10.1038/nature23268);
[Cano *et al.*, Phys. Rev. B **97**, 035139 (2018)](https://doi.org/10.1103/PhysRevB.97.035139)).
A site irrep at a maximal Wyckoff position usually induces an EBR; the
exceptions are real, site irreps whose band representation is a sum of others,
and in the grey groups they all occur without spin–orbit coupling: `¹E²E` at
site symmetry `-4`, and `E` at `-42m` and `-4m2`. An atomic limit's content is a
sum of EBR columns with nonnegative integer coefficients, so a band set whose
content is no such sum is not an atomic limit; the converse fails, since the
content at isolated
k-points is blind to what happens between them
([could these bands be an atomic limit?](../guide/atomic-limits.md)). An EBR
whose bands split apart across a gap has a topological part: such
disconnected EBRs are one route to topology
([Cano *et al.*, Phys. Rev. Lett. **120**, 266401 (2018)](https://doi.org/10.1103/PhysRevLett.120.266401)).
A summary writes an EBR as its site irrep at its Wyckoff position,
`-E1 at 1a (3m)`, and the records of the stable real-space invariants as
`1 (-E1)_a`; the records' fields, and the methods that write a sum on one line,
hold it under the label the group's list of EBRs gives it,
`-E1↑G(2)@1a(3m1',3m)`, which
[conventions](conventions.md#the-band-representation-key) reads part by part.

## How a symmetry indicator is computed

A band set's content at the high-symmetry points is its **symmetry vector**
`y`, one integer per corep at each point. The integer vectors that satisfy the
compatibility relations form a lattice, `{BS}`, the band structures; the
integer combinations of EBR columns, signs allowed, form a sublattice, `{AI}`,
the atomic insulators. The two have the same rank, so the quotient
`X_BS = {BS}/{AI}` is a finite abelian group, fixed by the space group alone:
the **indicator group**. A band set's **symmetry indicator** is the class of
its `y` in it
([Po, Vishwanath and Watanabe, Nat. Commun. **8**, 50 (2017)](https://doi.org/10.1038/s41467-017-00133-2);
[Kruthoff *et al.*, Phys. Rev. X **7**, 041069 (2017)](https://doi.org/10.1103/PhysRevX.7.041069)).

The class is read off the **EBR matrix** `M`, a row per corep at each point and
a column per EBR, by its **Smith normal form**: unimodular matrices `U` and
`V`, integer with integer inverses, with `U·M·V = D` diagonal, the elementary
divisors on the diagonal.
In the coordinates `y' = U·y` an atomic insulator is `D` times an integer
vector, so its entry `i` is a multiple of the divisor `dᵢ`, and zero beyond the
rank of `M`; every band structure is zero beyond the rank too, those rows being
the compatibility relations. So the entries over the divisors above 1 are
everything that tells a band structure from an atomic insulator:
`X_BS = ⊕ᵢ ℤ_{dᵢ}`, and the class is `y'ᵢ mod dᵢ` there. The detuned diamond of
[the 3D tutorial](../start/tutorial-3d.md), in its strong phase:

```python
import numpy as np
import pythtqc as tqc
from pythtb.models import fu_kane_mele

strong = tqc.realized_group(fu_kane_mele(1.0, 0.125, [0.4, 0.0, 0.0, 0.0]),
                            characters={0: "s"})  # the diamond, one bond detuned
data = tqc.ebr_data(strong.group)                # every EBR of the group
M = np.array(data["matrix"])                     # a row per corep, a column per EBR
U, D, V = (np.array(f) for f in data["smith"])   # U @ M @ V == D, diagonal
print(strong.group.name, M.shape, np.linalg.matrix_rank(M))
#  R-3m1' (12, 12) 8
print(np.diag(D), np.array_equal(U @ M @ V, D))
#  [1 1 1 1 1 1 2 4 0 0 0 0] True
t = tqc.hsp_table(strong.rep, bands=2)           # the valence pair's content
print(tqc.format_counts(t.counts))
#  -F5F6 + -GM9 + -L5L6 + -T8
y = np.array([t.counts.get(label, 0) for label in data["labels_physics_convention"]])
print(y, U @ y)
#  [0 0 0 1 0 1 0 1 0 0 1 0] [0 0 1 1 0 0 0 1 0 0 0 0]
print(tqc.symmetry_indicator(t).values)
#  (0, 1)
```

`R-3m1'` has twelve coreps at its high-symmetry points and twelve EBRs, and
its EBR matrix has rank 8. The elementary divisors above 1 are 2 and 4, so its
indicator group is `ℤ₂ × ℤ₄`. The valence pair's `y` has a 1 at each of the
four coreps it carries; in `y' = U·y` the last four entries vanish, as for any
band structure, and the entries over the divisors 2 and 4 read 0 and 1: the
class `(0, 1)`, which `symmetry_indicator` returns. A **nonzero** class says
`y` is no integer combination of EBR columns at all: no atomic limit carries
the content, not even as a difference of atomic limits, and adding atomic
bands, which adds multiples of the divisors, cannot change it. A **zero**
class is no certificate of an atomic limit: it says only that `y` is an integer
combination of EBR columns, negative coefficients allowed, which is where
fragile topology lives. The named forms a class is also written in, such as
the `z4` of an inversion-symmetric crystal, the number of odd-parity Kramers
pairs at the time-reversal-invariant momenta modulo 4, which refines the parity
criterion of
[Fu and Kane, Phys. Rev. B **76**, 045302 (2007)](https://doi.org/10.1103/PhysRevB.76.045302),
follow the naming of
[Song, Zhang, Fang and Fang, Nat. Commun. **9**, 3530 (2018)](https://doi.org/10.1038/s41467-018-06010-w),
which the tables of stable real-space invariants cite for their indicators with
spin–orbit coupling and without
([the names a class is spoken in](../guide/indicators.md#the-names-a-class-is-spoken-in)).

## Fragile topology

A band set can fail to be an atomic limit while its class is zero. When adding
a suitable atomic band set makes the sum an atomic limit, the obstruction is
**fragile**: removed by adding trivial bands, where a nonzero indicator is
stable against them
([Po, Watanabe and Vishwanath, Phys. Rev. Lett. **121**, 126402 (2018)](https://doi.org/10.1103/PhysRevLett.121.126402)).
The content of a fragile band set is often an integer combination of EBRs only
with some coefficient negative, a difference of atomic limits; it may even
decompose with nonnegative coefficients, the obstruction then lying beyond the
symmetry content altogether, as for the pair of bands of lin3g in
[atomic limits](../guide/atomic-limits.md#what-a-decomposition-cannot-certify).
Fragile bands can come from an EBR that splits across a gap, and they show in
the winding of their Wilson loops
([Bradlyn, Wang, Cano and Bernevig, Phys. Rev. B **99**, 045140 (2019)](https://doi.org/10.1103/PhysRevB.99.045140)).
The contents atomic limits can have, the nonnegative integer combinations of
EBRs, form an affine monoid, and content outside it with a zero class is the
symmetry content's signature of fragility
([Song, Elcoro, Xu, Regnault and Bernevig, Phys. Rev. X **10**, 031001 (2020)](https://doi.org/10.1103/PhysRevX.10.031001)).

## Real-space invariants, and the stable ones

A **real-space invariant** is a combination of the multiplicities of site
irreps at Wyckoff positions that no symmetric deformation of the Wannier
functions can change: Wannier functions move only in symmetric clusters, and
each move changes the multiplicities by a fixed pattern. Real-space invariants
were introduced in an account of fragile topology
([Song, Elcoro and Bernevig, Science **367**, 794 (2020)](https://doi.org/10.1126/science.aaz7650)).
The **stable real-space invariants** (SRSIs) of
[Hwang *et al.*, *Stable real-space invariants and topology beyond symmetry indicators*, Nat. Commun. **17**, 8564 (2026)](https://doi.org/10.1038/s41467-026-74844-w)
are written over every site irrep at every Wyckoff position at once, classify
atomic configurations up to stable equivalence, two configurations sharing them
exactly when they deform into each other once the same atomic orbitals are
added to both, and are tabulated for the grey
space groups with spin–orbit coupling and without, all but the thirteen whose
one invariant is the band count; the package ships those tables. Their
ℤ-valued part is a function of the symmetry vector, fractional exactly when the
indicator is nonzero; their ℤₙ-valued part sees what no symmetry vector does,
and gives a no-go criterion for a splitting of bands that diagnoses topology no
indicator detects
([stable real-space invariants](../guide/srsi.md)).
