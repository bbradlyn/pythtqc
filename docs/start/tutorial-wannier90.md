# Silicon from Wannier90: the group the model has

This page takes a real material through the numerical route. A Hamiltonian
that came out of a Wannier90 calculation is symmetric only to its own numerical
precision, and one declaration, `precision=`, says what that precision is; the
call then finds the crystal's cell, moves the orbital centres onto the
positions its operations require, symmetrizes what is symmetric within the
declaration, and returns the group the Hamiltonian has. Here that group is lower than the crystal's, which is the
lesson this package exists for: the group is the model's, not the lattice's,
and the bands are labelled in the group the model has.

The model is silicon, from the data of pythtb's own Wannier90 tutorial, kept in
this repository at [`docs/data/silicon_w90`](../data/silicon_w90/README.md): a
Wannier90 1.2 calculation of eight Wannier functions projected from `sp³`
hybrids on the two atoms of the primitive cell, four on each, disentangled from
twelve bands on a 4×4×4 k-point mesh. Energies are in eV and lengths in Å. The
path in the first block is relative to the repository root, where the blocks
run; the lines under a statement are what it prints, and the page runs in about
half a minute.

## The import

```python
import itertools

import numpy as np
from pythtb import W90
import pythtqc as tqc

w90 = W90("docs/data/silicon_w90", "si")       # relative to the repository root
si = w90.model(zero_energy=6.2285135)          # eV: the Fermi energy, made the zero
print(si.norb, si.nspin, len(si.hoppings))
#  8 1 2972
```

`W90` reads the files Wannier90 wrote, and `model` builds a pythtb model from
them with the Fermi energy, 6.2285135 eV as pythtb's tutorial gives it, as the
zero of energy, so the four valence bands end at zero. Eight orbitals and no
spin: the calculation has no spin–orbit coupling, and pythtb's importer returns
a spinless model in any case, so a spinor calculation declares its spin with
`spin=` ([a wannier90 spinor import](../guide/spin.md#a-wannier90-spinor-import)).
The 2972 hoppings are every term of the file, each pair of mutually conjugate
terms once: nothing is cut.

## Asked without a precision

```python
r0 = tqc.realized_group(si)
print(r0.group.name, r0.group.number_str, "|", r0.time_reversal["outcome"])
#  P1 1 | measured broken
print(f"largest imaginary hopping: {max(abs(np.imag(h['amplitude'])) for h in si.hoppings):.1e} eV")
#  largest imaginary hopping: 4.2e-04 eV
```

The group returned is `P1`, no operation but the identity, with time reversal
measured broken; the call also warns that only the unitary group is returned.
Both statements are true of the numbers handed in, and neither is a statement
about silicon.

- **Time reversal.** A spinless model whose hoppings are all real obeys it.
  These carry imaginary parts up to 4.2e-4 eV, and the covariance tolerance
  `tol` — how far `U H(k) U†` may miss `H(Rk)`, `U` an operation's matrix on
  the orbitals, for the operation still to count as obeyed — is 1e-8 eV when
  nothing is declared: machine level, the right tolerance for a model built
  from exact numbers and far below a Wannier90 model's own noise.
- **The cell.** The operations a Hamiltonian is measured against are first
  found from the orbital centres, by spglib, to a Cartesian tolerance
  `symprec`, 1e-5 Å by default ([every tolerance, its unit and its
  default](../reference/glossary.md#tolerances)). Wannier90's centres are
  numerical too, and the next section measures how far from symmetric they
  are.

## One declaration: `precision=`

```python
r = tqc.realized_group(si, precision=1e-4)
print(repr(r))
#  RealizedGroup(R-3m1' #166.98; precision 0.0001 declared -> tol 0.00487 (cut 0.001, window 0.3), position_tol 0.0001 (measured max|H| 4.872), symprec raised to 0.00382; descended from #227.129 (72 of 96 coset representatives of the lattice translations broken); centres moved 7.94e-06; symmetrized (defect 0.0014))
```

`precision` is the relative precision of the model's own numbers, a
dimensionless fact about the model: here, one part in 10⁴. Everything the
record prints after it is derived from it or measured with it.

- `tol 0.00487` is the covariance check's tolerance in eV,
  `10 × precision × max|H|`, and `max|H|` is measured, 4.872 eV, about the
  mean on-site energy, so the Fermi energy subtracted at the import moves no
  tolerance. `cut` and `window` are the measurement's two relative numbers,
  read against the same `max|H|`: `cut = tol / max|H| = 10 × precision`
  ([numerical models](../guide/numerical.md#how-to-read-it) reads each).
- `position_tol 0.0001` is the centres' tolerance, in reduced coordinates,
  where a relative precision is already absolute.
- `symprec raised to 0.00382` is the finder's tolerance in Å,
  `10 × precision × L` with `L` = 3.82 Å the longest lattice vector, so that
  centres noisy at the declared level do not hide the cell's operations.
- `descended from #227.129`: with that tolerance the cell is diamond's,
  `Fd-3m1'`. The Hamiltonian broke 72 of its 96 coset representatives, and the
  24 that hold make up the group returned, `R-3m1' #166.98`.
- `centres moved 7.94e-06`: before the cell's group was named and before any
  hopping was touched, the centres were moved onto the positions the cell's
  operations put them at, by up to 7.94e-6 in reduced coordinates, within the
  declared 1e-4, about an origin fitted to them, since a shift of every centre
  together is a choice of origin. The shift of all of them
  together that the fit leaves is within the declaration as well, so it is
  noise and is removed: the centres come back in the cell's own frame. Moving
  a centre is a k-dependent gauge transformation, so no energy moves with it.
- `symmetrized (defect 0.0014)`: the operations that hold only within `tol`
  licensed a projection of `H` onto its group-invariant part, which moved `H`
  by 0.0014 eV.

The centres' move, in Å, is the measurement the undeclared call was missing:

```python
lattice = np.asarray(si.lat_vecs)
together = np.asarray(r.positions["offset"])                    # the shift of all of them
moved = np.asarray(r.model.orb_vecs) - np.asarray(si.orb_vecs)
own = np.linalg.norm((moved - together) @ lattice, axis=1)
print(f"largest move of a centre: {own.max():.1e} Å, then all of them together by "
      f"{np.linalg.norm(together @ lattice):.1e} Å; symprec raised to {r.precision['symprec']:.1e} Å")
#  largest move of a centre: 2.8e-05 Å, then all of them together by 3.9e-05 Å; symprec raised to 3.8e-03 Å
```

The centres sat up to 2.8e-5 Å from the positions diamond's operations require
of them, nearly three times the default `symprec`, so undeclared the finder
found no operation but the identity, and in `P1` every centre is where its one
operation puts it. The shift of all of them together is no displacement and
the finder does not see it; it is the declared noise too, and removing it puts
the centres in the cell's own frame, where the names below are read. Raise the
finder's tolerance alone, and it finds diamond's operations, which the centres
then miss:

```python
seen = tqc.realized_group(si, symprec=1e-4)          # the finder's tolerance alone, in Å
print(seen.gate, f"{seen.value:.1e}", f"{2 * np.pi * seen.value * seen.context['max_h']:.1e} eV",
      seen.context["group"].name, seen.context["number_str"])
#  positions_not_exactly_symmetric 2.9e-05 8.9e-04 eV Fd-3m1' 227.129
```

This is the refusal that names `precision=`. The centres miss the cell's
operations by 2.9e-5 in reduced coordinates, which alone costs `H(k)` a
covariance residual of order `2π × 2.9e-5 × max|H|`, 8.9e-4 eV, far above the
default `tol`; so nothing is built, and the cell's group, `Fd-3m1'`, is named
as an upper bound on what the Hamiltonian obeys
([`positions_not_exactly_symmetric`](../reference/refusals.md#positions_not_exactly_symmetric)).
Centres noisier than the finder's tolerance never reach this refusal, which is
why an undeclared numerical import can return `P1` instead
([centres noisier than `symprec`](../guide/numerical.md#centres-noisier-than-symprec)).
Declared, the finder reads the centres to 3.8e-3 Å, and they are moved onto
the operations it finds.

```python
d = r.descent
print(d["from"]["name"], d["from"]["size"], "->", d["to"]["name"], d["to"]["size"],
      sorted({b["refuted_by"] for b in d["dropped"]}))
#  Fd-3m1' 96 -> R-3m1' 24 ['no symmetry matrix']
s = r.symmetrization
worst = max(s.residuals[i] for i in s.near)
print(f"{len(s.near)} of {len(s.operations)} near-symmetric, worst at {worst:.1e} eV "
      f"against tol {s.tol:.1e} eV; H moved by {s.defect:.1e} eV")
#  23 of 24 near-symmetric, worst at 4.4e-03 eV against tol 4.9e-03 eV; H moved by 1.4e-03 eV
```

Every one of the 72 broken operations is refuted by `no symmetry matrix`: no
matrix with its action on `k` carries `H(k)` onto `H(Rk)`, not even within the
declared precision. Of the 24 kept, all but the identity hold only within the
declaration — **near-symmetric** — and averaging `H` over the group moved it by
about a third of the worst of them.
[Numerical and wannier90 models](../guide/numerical.md#exact-near-and-broken)
has the three classes, exact, near and broken, and the projection in full.

The declaration is a statement about the model, not a dial to turn until the
group looks right:

```python
for p in (1e-5, 1e-3):
    rp = tqc.realized_group(si, precision=p)
    print(f"precision {p:.0e}: tol {rp.precision['tol']:.2g} eV -> {rp.group.name} "
          f"#{rp.group.number_str}, time reversal {rp.time_reversal['outcome']}")
#  precision 1e-05: tol 0.00049 eV -> P1 #1, time reversal measured broken
#  precision 1e-03: tol 0.049 eV -> R-3m1' #166.98, time reversal kept
```

At 1e-5 the operations that hold within one part in 10⁴ read broken, time
reversal among them, and the group is `P1`; at 1e-3 it is the same `R-3m1'`.
The model's numbers are symmetric to about one part in 10⁴ of `max|H|` — the
imaginary parts above, 4.2e-4 eV against 4.872 eV, are one measure of it — and
not to the six decimals `si_hr.dat` is written in.

## The model's own group

The group returned is lower than diamond's, and that is a property of the
model, visible in its spectrum with no symmetry machinery at all: an operation
`R` of the crystal's point group carries the energies at `k` onto those at
`Rk`. Over the 48 operations of the cube:

```python
k = np.random.default_rng(0).uniform(-0.5, 0.5, size=(8, 3))      # eight random k, reduced
energies = si.solve_ham(k)
held, broken = [], []
for perm in itertools.permutations(range(3)):          # the 48 operations of the cube:
    for signs in itertools.product((1, -1), repeat=3):  # every signed permutation of x, y, z
        R = np.zeros((3, 3))
        R[[0, 1, 2], perm] = signs
        Rk = k @ (lattice @ R @ np.linalg.inv(lattice)).T          # R acting on reduced k
        violation = np.abs(si.solve_ham(Rk) - energies).max()
        (held if violation < 0.01 else broken).append((violation, R))
print(f"{len(held)} hold, to {max(v for v, R in held):.4f} eV at most")
#  12 hold, to 0.0027 eV at most
print(f"{len(broken)} are violated by {min(v for v, R in broken):.2f} to {max(v for v, R in broken):.2f} eV")
#  36 are violated by 0.27 to 0.30 eV
print("those that hold keep the (1, 1, 1) axis:", all(abs((R @ np.ones(3)).sum()) == 3 for v, R in held))
#  those that hold keep the (1, 1, 1) axis: True
atoms = np.array([[0.0, 0.0, 0.0], [-0.25, 0.75, -0.25]])        # si.win's two atoms, reduced
print("the bond inside the home cell:", np.round((atoms[1] - atoms[0]) @ lattice, 4), "Å")
#  the bond inside the home cell: [1.3494 1.3494 1.3494] Å
```

Twelve of the 48 hold, to a few meV; the other 36 are violated by 0.27 to
0.30 eV at these eight points, a hundred times more. The twelve that hold map
the Cartesian axis along `(1, 1, 1)` onto itself: they are the point group
`-3m` (`D3d`) about it, the point group of `R-3m`. That axis is the bond
between the two atoms of the home cell, the last line. The model handed in is
not cubic, and `R-3m1'` is its group.

## What Wannier90's own bands show

```python
kpts, w90_energies = w90.bands_w90()           # Wannier90's interpolation along its band path
mismatch = np.abs(si.solve_ham(kpts) - (w90_energies - 6.2285135)).max()
print(len(kpts), f"k-points, largest difference {mismatch:.1e} eV")
#  190 k-points, largest difference 1.5e-05 eV
```

Along the path Wannier90 plotted, `L–Γ–X–K–Γ` in `si.win`, the model reproduces
Wannier90's own interpolated bands to 1.5e-5 eV. That is the comparison an
import is checked with, and it cannot see the lowered symmetry: both sides are
one interpolation, and that interpolation is not cubic. pythtb reads the files
as Wannier90 wrote them, Wigner–Seitz degeneracies included.

The cause is in the calculation. Wannier90 builds `H(k)`
from `H(R)` over the lattice vectors of a Wigner–Seitz supercell of its k-point
mesh. Before version 2.1 it took every `R` about the home cell's origin,
whatever the centres of the two functions a hopping joins; for functions
centred away from that origin, as these are, on a mesh as coarse as 4×4×4, the
interpolation loses operations of the crystal — here 36 of the cube's 48.
Version 2.1 added the correction that takes each hopping by the distance
between the two functions' centres, `use_ws_distance`, the default from
version 3; with it the interpolation keeps the crystal's symmetry, and a finer
mesh makes the loss smaller. This calculation was made with version 1.2.

## The bands, labelled in the model's group

```python
for row in r.rep.characters.orbitals:
    print(row.orbital, row.wyckoff, row.site_symmetry, row.irrep, row.determined_by)
#  0 6c 3m A1 lowest-l
#  1 18h m A' lowest-l
#  2 18h m A' lowest-l
#  3 18h m A' lowest-l
#  4 6c 3m A1 lowest-l
#  5 18h m A' lowest-l
#  6 18h m A' lowest-l
#  7 18h m A' lowest-l
valence = tqc.hsp_table(r.rep, bands=4)                        # the four occupied bands
print(valence.summary())
#  bands 0-3   (.bands)
#    their coreps at each high-symmetry point, GM being Γ   (.content)
#      GM: 2×GM1+ + GM3+
#      F: 2×F1+ + F1- + F2-
#      L: 2×L1+ + L2+ + L2-
#      T: T1+ + T2- + T3-
#    the labels of these 4 of the 8 bands rest on a one-dimensional character the hoppings leave open on the coupled part holding every orbital: another choice of it keeps every orbital's site irrep and the labels of all 8 bands, and labels these 4 differently; the lowest-angular-momentum rule chose the one whose orbitals read as the lowest-angular-momentum functions carrying their site content   (.note)
dec = tqc.decompose(valence, r.group, rep=r.rep)
print(dec.verdict, "|", len(dec.solutions), "solution:", *dec.solution_strings())
#  symmetry-indicated decomposable | 1 solution: 1 x A1g↑G(1)@3a(-3m1',-3m) + 1 x Ag↑G(3)@9d(2/m1',2/m)
indicator = tqc.symmetry_indicator(valence, r.group, rep=r.rep)
print(indicator.structure, indicator.canonical)
#  Z2 {"z2'": 0}
```

```python
# the four nearest neighbours of the atom at the origin: the other atom and three images
neighbours = atoms[1] + np.array([[0, 0, 0], [0, -1, 0], [1, -1, 0], [0, -1, 1]])
print(np.round(np.linalg.norm((neighbours - atoms[0]) @ lattice, axis=1), 4))
#  [2.3372 2.3372 2.3372 2.3372]
centres = (atoms[0] + neighbours) / 2                           # the centres of its four bonds
print([f"{s.multiplicity}{s.letter}" for s in tqc.wyckoff_sites_tables(r.group, centres)])
#  ['3a', '9d', '9d', '9d']
```

Nothing was declared, so the
[atomic-limit table](../reference/records.md#the-atomic-limit-table) is the
[lowest-ℓ rule's](../background/measurement.md#the-lowest-angular-momentum-rule):
the hoppings fix how the functions transform relative to one another, not what
they are, and what they leave open is assigned at the lowest angular momentum,
orbital 0 read as an `s` function. That gives `A1` for the two functions on the
threefold axis, at `6c` with site symmetry `3m`, and `A'` for the six at `18h`,
whose site symmetry is one mirror — each function even under every operation
that fixes its centre, as an `sp³` hybrid is, its axis running from its atom
through that centre. Where the functions are known to carry other content,
`characters=` states it ([saying what the orbitals are](../guide/orbitals.md)).

The model has no spin, so the labels are single-valued coreps of `R-3m1'` at
its four high-symmetry points, `+` and `−` their parity under inversion: at
`GM` the four valence bands are two `GM1+` and one two-dimensional `GM3+`. The
line under them says what they rest on, a one-dimensional character the
hoppings leave open. The site irreps leave two labellings of the
representation open, and the one returned is the lowest-ℓ rule's; the
other differs from it by a sign on the inversion, the twofold rotations and the
rotoinversions, a one-dimensional character the site groups `3m` and `m`
cannot see, and in it every parity of these bands is the opposite one.

The four bands [decompose](../guide/atomic-limits.md) in exactly one way,
`A1g` induced from Wyckoff `3a` plus `Ag` induced from `9d`: one orbital at
`3a` and one at each of the three sites of `9d`, even under every operation
that fixes its site. The positions are the same in either labelling, and the
site irreps are the returned one's
([positions survive every labelling; labels may not](../guide/atomic-limits.md#positions-survive-every-labelling-labels-may-not)).
The last block places them. The atom at the origin has four nearest
neighbours, the home cell's other atom and three of its images, all 2.3372 Å
away; the centre of the bond inside the home cell is `3a`, and the centres of
the other three are `9d`. The atom at the origin is no inversion centre, so the
tables are read about the one nearest it, the centre of that bond
([the origin the names are read about](../background/conventions.md#the-origin-the-names-are-read-about)).
The valence bands carry the content of one orbital
on each of silicon's four bonds, read in the labelling returned as a bonding
orbital, even about its bond's centre — the textbook picture — with the bond
on the threefold axis set apart from the other three, as the model's group has
it. The [indicator group](../guide/indicators.md) of `R-3m1'` without spin is
`ℤ₂`, and the class is zero, as it must be for content that decomposes.

## Where to go next

- [Numerical and wannier90 models](../guide/numerical.md): `precision=`, the
  centres first, the projection, a broken operation never symmetrized into,
  and the memory budget for a large model.
- [Spin](../guide/spin.md#a-wannier90-spinor-import): a spinor calculation,
  which pythtb imports as a spinless model with the spin inside the orbital
  index, declared with `spin=`.
- [The records reference](../reference/records.md#every-field): every field of
  the record's `precision`, `positions` and `symmetrization`.
- [Labelling the bands](../guide/labels.md),
  [atomic limits](../guide/atomic-limits.md) and
  [symmetry indicators](../guide/indicators.md): the calls of the last section.
- The calls on this page: [`realized_group`](../reference/functions.md#realized_group),
  [`hsp_table`](../reference/functions.md#hsp_table),
  [`format_counts`](../reference/functions.md#format_counts),
  [`decompose`](../reference/functions.md#decompose),
  [`symmetry_indicator`](../reference/functions.md#symmetry_indicator) and
  [`wyckoff_sites_tables`](../reference/functions.md#wyckoff_sites_tables).

## Another calculation

The same route on another Wannier90 run: the folder, the prefix and the Fermi
energy change, and so may the precision its numbers hold to.

```python template
from pythtb import W90
import pythtqc as tqc

model = W90("path/to/the/run", "prefix").model(zero_energy=fermi_energy)   # eV
r = tqc.realized_group(model, precision=1e-4)    # the relative precision of the model's numbers
print(repr(r))                                   # what was derived, moved, symmetrized and broken

# a spinor calculation: say which orbitals are the (up, down) pair of one function
r = tqc.realized_group(model, spin="interleaved", precision=1e-4)
```
