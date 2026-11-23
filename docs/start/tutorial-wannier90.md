# Silicon from Wannier90: the group the model has

This page shows how to read a Hamiltonian from a Wannier90 calculation. Such a
Hamiltonian is symmetric only up to its numerical precision, which is declared
with `precision=`. `realized_group` then finds the cell, moves the orbital
centers onto symmetric positions, symmetrizes the Hamiltonian within the
declared precision, and returns its space group. No center is moved farther
than the declared precision, in reduced coordinates.

The model is silicon, from the data of pythtb's own Wannier90 tutorial. The
data are in `docs/data/silicon_w90`, whose README,
[Silicon from Wannier90](../data/silicon_w90/README.md), lists the files. It
is a Wannier90 1.2 calculation of eight Wannier functions, projected from sp3
hybrids on the two atoms of the primitive cell, four on each. They are
disentangled from twelve bands on a 4x4x4 k-point mesh. Energies are in eV and
lengths in Å. The blocks run from the repository root. The lines under a
statement are what it prints. The blocks run in under a minute.

## The import

The example reads silicon's files and prints the number of orbitals, of spin
components and of hoppings:

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

`W90` reads the files Wannier90 wrote. Its `model` builds a pythtb model from
them, with the zero of energy at the Fermi energy, 6.2285135 eV as pythtb's
tutorial gives it, so the top of the four valence bands is at zero.

- `8`: `si.norb`, the eight Wannier functions.
- `1`: `si.nspin`. The calculation has no spin-orbit coupling. pythtb's
  importer returns a spinless model for any calculation. The spin of a spinor
  calculation is declared with `spin=`
  ([A Wannier90 spinor import](../guide/spin.md#a-wannier90-spinor-import)).
- `2972`: the number of hoppings, `len(si.hoppings)`. They are every term of
  `si_hr.dat` but the eight on-site energies, each pair of mutually conjugate
  terms counted once. The importer drops none for being small or distant.

## Asked without a precision

The example asks for silicon's group with nothing declared, and prints the
largest imaginary part among its hoppings:

```python
r0 = tqc.realized_group(si)
print(r0.group.name, r0.group.number_str, "|", r0.time_reversal.outcome)
#  P1 1 | measured broken
print(f"largest imaginary hopping: {max(abs(np.imag(h['amplitude'])) for h in si.hoppings):.1e} eV")
#  largest imaginary hopping: 4.2e-04 eV
```

The group returned is `P1`, no operation but the identity, with time reversal
measured broken. The call also issues a `MagneticScopeWarning`, which says that
only the unitary subgroup of the Hamiltonian's magnetic group is returned.
`P1` and broken time reversal are both true of the numbers handed in, at the
default tolerance. Neither is a property of silicon at the precision of its
numbers.

- **Time reversal.** A spinless model whose hoppings are all real obeys it.
  Silicon's hoppings carry imaginary parts up to 4.2e-4 eV. An operation with
  rotation `R` counts as obeyed when a matrix `U` on the orbitals makes the
  largest magnitude of an entry of `U H(k) U† − H(Rk)` over k, the covariance
  residual, at most the tolerance `tol`. For an operation combined with time
  reversal, `H(k)` is complex-conjugated and `Rk` is replaced by `−Rk`. With
  nothing declared, `tol` is 1e-8 times `max|H|`, 4.9e-8 eV here. `max|H|` is
  the largest magnitude of an entry of `H(k)` over a fixed set of random
  k-points, with the mean on-site energy subtracted. The imaginary parts,
  4.2e-4 eV, are far above that `tol`. That tolerance is right for a model
  built from exact numbers, and far below the noise of a Wannier90 model.
- **The cell.** The operations a Hamiltonian is measured against are first
  found from the orbital centers by the symmetry finder spglib. Its tolerance
  is `symprec`, a Cartesian length, 1e-5 Å by default. For silicon it found no
  operation but the identity. The glossary gives every tolerance with its unit
  and its default ([Tolerances](../reference/glossary.md#tolerances)). The next
  section measures how far from symmetric Wannier90's centers are.

## One declaration: `precision=`

`precision=` is the relative precision of the model's own numbers. It is
dimensionless. `realized_group` derives every tolerance of the measurement
from it. The example declares silicon's numbers precise to 1e-4:

```python
r = tqc.realized_group(si, precision=1e-4)
print(repr(r))
#  RealizedGroup(R-3m1' #166.98; precision 0.0001 declared -> tol 0.00487 (cut 0.001, window 0.3), position_tol 0.0001 (measured max|H| 4.872), symprec raised to 0.00382; descended from Fd-3m1' #227.129 (72 of 96 coset representatives of the lattice translations broken); centers moved 7.94e-06; symmetrized (defect 0.0014))
```

### How to read it

- `R-3m1' #166.98`: the group returned, `r.group`. `R-3m1'` is `R-3m` with
  time reversal adjoined, a grey group. `166.98` is its number in the
  Belov-Neronova-Smirnova (BNS) notation.
- `precision 0.0001 declared`: the relative precision declared with
  `precision=`.
- `tol 0.00487`: a tolerance, in the units of `H`, for accepting an
  operation's matrix. It is the `tol` the covariance residual is held to
  above. Computed from `precision` and `max|H|`.
- `cut 0.001`: a dimensionless tolerance for finding the matrices an operation
  can have. For each operation, the singular values of the equations
  `U H(k) = H(Rk) U`, linear in the matrix `U`, are computed. A singular value
  is treated as zero when it is at most `cut` times the largest singular
  value. Computed from `precision`.
- `window 0.3`: a dimensionless tolerance for pairing eigenvectors when those
  equations are solved. The pairing is done at one k-point. There an
  eigenvector of `H(k)` and one of `H(Rk)` are paired when their energies
  differ by at most `window` times the largest magnitude of an entry of `H(k)`
  or `H(Rk)`, the mean on-site energy subtracted: about 0.9 eV here. Computed
  from `precision`.
- `position_tol 0.0001`: a tolerance, in reduced coordinates, for moving the
  orbital centers. A center within `position_tol` of the position the group's
  operations put it at is moved onto it. Equal to `precision`.
- `measured max|H| 4.872`: `max|H|` as defined above, in the units of `H`.
  Since the mean on-site energy is subtracted, the Fermi energy subtracted at
  the import moves no tolerance.
- `symprec raised to 0.00382`: the symmetry finder's tolerance, in Å, raised so
  that centers noisy at the declared precision do not hide the cell's
  operations. Computed from `precision` and the longest lattice vector,
  3.82 Å.
- `descended from Fd-3m1' #227.129 (72 of 96 ... broken)`: at the raised
  `symprec`, the symmetry finder returns the operations of the diamond
  structure, the group `Fd-3m1'`. The Hamiltonian breaks 72 of its 96
  operations, counted one for each coset of the lattice translations. The 24
  that hold form the group returned.
- `centers moved 7.94e-06`: the largest move of an orbital center onto the
  position the cell's operations put it at, in reduced coordinates. A shift of
  every center together changes no distance between them, so the centers are
  fitted with the origin left free. Every center is then shifted together, by
  at most the declared precision, so that each operation's translation is a
  rational fraction of the model's cell. The centers returned, in `r.model`,
  carry both moves. Moving a center is a k-dependent gauge transformation, so
  no energy moves.
- `symmetrized (defect 0.0014)`: `H` was projected onto its group-invariant
  part, since some operations hold only within `tol`. The projection changed
  no entry of `H(k)` by more than 0.0014 eV.

The example converts silicon's center moves to Å, the unit of `symprec`:

```python
lattice = np.asarray(si.lat_vecs)
together = np.asarray(r.numerical.positions.offset)                    # the shift of every center together
moved = np.asarray(r.model.orb_vecs) - np.asarray(si.orb_vecs)
own = np.linalg.norm((moved - together) @ lattice, axis=1)
print(f"largest move of a center: {own.max():.1e} Å, then all of them together by "
      f"{np.linalg.norm(together @ lattice):.1e} Å; symprec raised to {r.numerical.precision.symprec:.1e} Å")
#  largest move of a center: 2.8e-05 Å, then all of them together by 3.9e-05 Å; symprec raised to 3.8e-03 Å
```

The centers sat up to 2.8e-5 Å from the positions diamond's operations put
them at, nearly three times the default `symprec`. So with nothing declared the
finder found no operation but the identity. The shift of every center together,
3.9e-5 Å, changes no distance between them, and the finder does not see it.

Raising the finder's tolerance alone, with `symprec=`, finds diamond's
operations, which the centers then miss. The example asks for silicon's group
at a `symprec` of 1e-4 Å:

```python
seen = tqc.realized_group(si, symprec=1e-4)          # the finder's tolerance alone, in Å
print(seen.gate, f"{seen.value:.1e}", f"{2 * np.pi * seen.value * seen.context['max_h']:.1e} eV",
      seen.context["group"].name, seen.context["number_str"])
#  positions_not_exactly_symmetric 2.9e-05 8.9e-04 eV Fd-3m1' 227.129
```

The call refuses with `positions_not_exactly_symmetric`, and its reason names
`precision=`. Its value, 2.9e-5 in reduced coordinates, is how far the image
of a center under an operation of the cell misses the center it lands on, the
largest component over every operation and center. It is not a move. The moves
above, up to 7.94e-6 each, bring every image onto its center at once. A miss
that size alone gives a covariance residual of order `2π × 2.9e-5 × max|H|`,
8.9e-4 eV, far above the default `tol`, 4.9e-8 eV. The refusal names the
cell's group, `Fd-3m1'`, as an upper bound on what the Hamiltonian obeys
([`positions_not_exactly_symmetric`](../reference/refusals.md#positions_not_exactly_symmetric)).
Centers that miss by more than the finder's tolerance hide the cell's
operations instead, and the call returns a lower group, `P1` for silicon above
([Centers noisier than `symprec`](../guide/numerical.md#centres-noisier-than-symprec)).
With `precision=1e-4` the finder's tolerance is 3.8e-3 Å, and the centers are
moved onto the positions its operations give.

The descent and the projection are on the record's `descent` and
`numerical.symmetrization`. The example prints silicon's:

```python
d = r.descent
print(d.cell_group.name, d.cell_group.size, "->", d.subgroup.name, d.subgroup.size,
      sorted({b.refuted_by for b in d.dropped}))
#  Fd-3m1' 96 -> R-3m1' 24 ['no symmetry matrix']
s = r.numerical.symmetrization
worst = max(s.residuals[i] for i in s.near)
print(f"{len(s.near)} of {len(r.rep.matrices)} near-symmetric, worst at {worst:.1e} eV "
      f"against tol {s.tol:.1e} eV; H moved by {s.defect:.1e} eV")
#  23 of 24 near-symmetric, worst at 1.5e-03 eV against tol 4.9e-03 eV; H moved by 1.4e-03 eV
```

- `Fd-3m1' 96 -> R-3m1' 24`: `d.cell_group` and `d.subgroup`, the group the
  cell allows and the group the descent arrives at, each with its number of
  operations, one for each coset of the lattice translations.
- `['no symmetry matrix']`: what refuted the 72 operations dropped, each
  entry's `refuted_by`. For each of them, no matrix on the orbitals carries
  `H(k)` onto `H(Rk)`, even within `tol`.
- `23 of 24 near-symmetric`: `s.near`, the operations that hold only within
  `tol`. They are every operation of the group but the identity.
- `worst at 1.5e-03 eV against tol 4.9e-03 eV`: the largest covariance
  residual among them, from `s.residuals`, and the tolerance it is read
  against, `s.tol`.
- `H moved by 1.4e-03 eV`: `s.defect`, how far the projection onto the
  group-invariant part moved `H`. It is about as much as the worst of the
  near-symmetric operations misses.

[Exact, near and broken](../guide/numerical.md#exact-near-and-broken) gives the
three classes and the projection in full.

A declared precision is not adjusted to make the group come out right. The
example declares silicon's precision a decade on either side of 1e-4:

```python
for p in (1e-5, 1e-3):
    rp = tqc.realized_group(si, precision=p)
    print(f"precision {p:.0e}: tol {rp.numerical.precision.tol:.2g} eV -> {rp.group.name} "
          f"#{rp.group.number_str}, time reversal {rp.time_reversal.outcome}")
#  precision 1e-05: tol 0.00049 eV -> P1 #1, time reversal measured broken
#  precision 1e-03: tol 0.049 eV -> R-3m1' #166.98, time reversal kept
```

At 1e-5 the operations that hold within 1e-4 read broken, time reversal among
them, and the group is `P1`. At 1e-3 the group is `R-3m1'`, as at 1e-4. The
model's numbers are symmetric to about 1e-4 of `max|H|`. The imaginary parts
above, 4.2e-4 eV against 4.872 eV, are one measure of it. `si_hr.dat` is
written to six decimals, and its numbers are not symmetric to that many.

## The model's own group

The group returned is lower than diamond's. That is a property of the model,
seen in its spectrum alone. An operation `R` the model obeys carries the
energies at `k` onto those at `Rk`. The example compares silicon's energies at
eight random k-points with those at their images under each of the 48
operations of the cube:

```python
k = np.random.default_rng(0).uniform(-0.5, 0.5, size=(8, 3))      # eight random k, reduced
energies = si.solve_ham(k)
held, broken = [], []
for perm in itertools.permutations(range(3)):          # signed permutations of x, y, z
    for signs in itertools.product((1, -1), repeat=3):
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

Twelve of the 48 hold, to a few meV. The other 36 are violated by 0.27 to
0.30 eV at these eight points, a hundred times more. The twelve that hold map
the Cartesian axis along `(1, 1, 1)` onto itself. They are the point group
`-3m` (`D3d`) about it, the point group of `R-3m`. That axis runs along the
bond between the two atoms of the home cell, printed on the last line. The
model handed in is not cubic, and `R-3m1'` is its group.

## What Wannier90's own bands show

The example compares silicon's bands with Wannier90's own interpolated bands,
along the band path of `si.win`:

```python
kpts, w90_energies = w90.bands_w90()           # Wannier90's interpolation along its band path
mismatch = np.abs(si.solve_ham(kpts) - (w90_energies - 6.2285135)).max()
print(len(kpts), f"k-points, largest difference {mismatch:.1e} eV")
#  190 k-points, largest difference 1.5e-05 eV
```

Along the path L-Γ-X-K-Γ that `si.win` names, the model reproduces Wannier90's
own bands to 1.5e-5 eV. This comparison is how an import is usually checked,
and it cannot see the lowered symmetry. Both sides are one interpolation, and
that interpolation is not cubic. pythtb reads the files as Wannier90 wrote
them, Wigner-Seitz degeneracies included.

The cause is in the calculation. Wannier90 builds `H(k)` from `H(R)` over the
lattice vectors of a Wigner-Seitz supercell of its k-point mesh. Before
version 2.1 it took every `R` about the home cell's origin, whatever the
centers of the two functions a hopping joins. For functions centered away from
that origin, as these are, on a mesh as coarse as 4x4x4, the interpolation
loses operations of the crystal, here 36 of the cube's 48. Version 2.1 added
`use_ws_distance`, the default from version 3. With it, each hopping's `R` is
taken so that the vector between the two functions' centers lies in the
Wigner-Seitz supercell. The interpolation then keeps the crystal's symmetry.
Without it, a finer mesh makes the violation smaller. This calculation was
made with version 1.2.

## The bands, labeled in the model's group

The example prints silicon's atomic-limit table, labels its four valence
bands, decomposes their symmetry data vector and reads its symmetry indicator:

```python
for o in r.wyckoff_orbits:
    print(o.orbitals, o.wyckoff, o.unitary_site_symmetry, o.label, o.determined_by)
#  (0, 4) 6c 3m A1 lowest-l
#  (1, 2, 3, 5, 6, 7) 18h m A' lowest-l
valence = tqc.hsp_table(r, bands=4)                        # the four occupied bands
print(valence.summary())
#  bands 0-3   (.bands)
#    their coreps at each high-symmetry point   (.symmetry_data)
#      GM: 2xGM1+ + GM3+
#      F: 2xF1+ + F1- + F2-
#      L: 2xL1+ + L2+ + L2-
#      T: T1+ + T2- + T3-
#    another labeling the hoppings leave open keeps every site irrep and labels some energy multiplets of these bands differently; it keeps which bands are degenerate and the labels of all the model's bands together   (.note)
dec = tqc.decompose(valence)
print(dec.decomposable, "|", len(dec.candidate_sums), "candidate sum:", *[s.label for s in dec.candidate_sums])
#  True | 1 candidate sum: A1g at 3a (-3m1') + Ag at 9d (2/m1')
indicator = tqc.symmetry_indicator(valence)
print(indicator.indicator_group, indicator.indicators)
#  Z2 {"z2'": 0}
```

- `(0, 4) 6c 3m A1 lowest-l`: one row of the
  [atomic-limit table](../reference/records.md#the-atomic-limit-table), a
  Wyckoff orbit of orbitals. It gives:
  - `o.orbitals`, the orbitals;
  - `o.wyckoff`, their Wyckoff position;
  - `o.unitary_site_symmetry`, the unitary part of its site-symmetry group;
  - `o.label`, the site irrep each orbital carries;
  - `o.determined_by`, how that site irrep is known. `lowest-l` says that
    nothing was declared and the site irrep is assigned by the
    [lowest-angular-momentum rule](../background/measurement.md#the-lowest-angular-momentum-rule).
- `bands 0-3`: the bands labeled, `valence.bands`.
- `GM: 2xGM1+ + GM3+`: the corep multiplicities of the four bands at Γ, part of
  their symmetry data vector, `valence.symmetry_data`. `GM` is the tables' name
  for Γ.
- `another labeling the hoppings leave open ...`: `valence.note`, printed
  where another representation of the group on the orbitals, one the hoppings
  allow, gives some energy multiplet of these bands other coreps.
- `True | 1 candidate sum: A1g at 3a (-3m1') + Ag at 9d (2/m1')`: whether a
  sum of elementary band representations with nonnegative integer
  multiplicities carries this symmetry data vector, `dec.decomposable`, and
  every such sum, `dec.candidate_sums`. Such a sum is a decomposition. Each
  term is an elementary band representation, written as the site irrep that
  induces it, its Wyckoff position, and its site-symmetry group in
  parentheses.
- `Z2 {"z2'": 0}`: the symmetry-indicator group, `indicator.indicator_group`,
  and the class of the four bands in it, `indicator.indicators`.

The hoppings fix how the functions transform relative to one another, not what
they are. The lowest-angular-momentum rule puts orbital 0 at the lowest angular
momentum its site allows, an `s` function here. Every orbital the hoppings
relate to it carries what they fix relative to it. That gives `A1` for the two
functions on the threefold axis, at `6c` with site symmetry `3m`. It gives `A'`
for the six at `18h`, whose site symmetry is one mirror. Each function is even
under every operation that fixes its center. So is an sp3 hybrid, whose axis
runs from its atom through that center. Where the functions are known to carry
other site irreps, `characters=` states them
([How do I say what my orbitals are?](../guide/orbitals.md)).

The model has no spin, so the bands carry single-valued coreps of `R-3m1'` at
its four high-symmetry points. `+` and `-` are their parity under inversion. At
Γ the four valence bands carry two `GM1+` and one two-dimensional `GM3+`.
`valence.note` says that these coreps rest on a choice the hoppings leave open.
The site irreps leave two representations of the group on the orbitals open,
and the one returned is the lowest-angular-momentum rule's. The other differs
from it by a sign on the inversion, the twofold rotations and the threefold
rotoinversions. That sign is a one-dimensional character trivial on the site
groups `3m` and `m`. Under the other representation every parity of these
bands is the opposite one.

The symmetry data vector of the four bands has exactly one decomposition
([Could these bands be an atomic limit?](../guide/atomic-limits.md)): `A1g`
induced from the Wyckoff position `3a`, plus `Ag` induced from `9d`. That is one
orbital at `3a` and one at each of the three sites of `9d`, each even under
every operation that fixes its site. Under the other representation the Wyckoff
positions are the same. Each site irrep is multiplied by the sign above: `A1g`
at `3a` becomes `A2u`, and `Ag` at `9d` becomes `Bu`
([Positions survive every labeling; labels may not](../guide/atomic-limits.md#positions-survive-every-labeling-labels-may-not)).

The example places those Wyckoff positions on silicon's bonds. It finds the
four nearest neighbors of the atom at the origin, and the Wyckoff positions of
the centers of its four bonds:

```python
# the four nearest neighbors of the atom at the origin: the other atom and three images
neighbors = atoms[1] + np.array([[0, 0, 0], [0, -1, 0], [1, -1, 0], [0, -1, 1]])
print(np.round(np.linalg.norm((neighbors - atoms[0]) @ lattice, axis=1), 4))
#  [2.3372 2.3372 2.3372 2.3372]
centers = (atoms[0] + neighbors) / 2                           # the centers of its four bonds
print([s.wyckoff for s in tqc.wyckoff_sites_tables(r.group, centers)])
#  ['3a', '9d', '9d', '9d']
```

The atom at the origin has four nearest neighbors, the home cell's other atom
and three of its images, all 2.3372 Å away. The center of the bond inside the
home cell is at `3a`, and the centers of the other three are at `9d`. The atom
at the origin is not an inversion center, so the Wyckoff positions are named
with the origin at the nearest one, the center of that bond
([The origin the names are read about](../background/conventions.md#the-origin-the-names-are-read-about)).
The symmetry data vector of the valence bands is that of one orbital on each of
silicon's four bonds. Under the representation returned that orbital is even
about its bond's center, a bonding orbital, as in the textbook picture. The
bond on the threefold axis is at a Wyckoff position of its own, `3a`, since
the model's group relates it to none of the other three. The
symmetry-indicator group of `R-3m1'` without spin is `Z2`
([Which symmetry-indicator class do these bands occupy?](../guide/indicators.md)),
and the class is zero, as it is for every symmetry data vector with a
decomposition.

## Another calculation

The same route reads another Wannier90 run. The folder, the prefix and the
Fermi energy change, and so may the precision its numbers hold to.

```python template
from pythtb import W90
import pythtqc as tqc

model = W90("path/to/the/run", "prefix").model(zero_energy=fermi_energy)   # eV
r = tqc.realized_group(model, precision=1e-4)    # the relative precision of the model's numbers
print(repr(r))                                   # what was derived, moved, symmetrized and broken

# a spinor calculation: say which orbitals are the (up, down) pair of one spinful orbital
r = tqc.realized_group(model, spin="interleaved", precision=1e-4)
```

## Where to go next

- [How do I measure a numerical or Wannier90 model?](../guide/numerical.md)
  gives `precision=`, how the centers are corrected first, the projection,
  which never imposes a broken operation, and a large model.
- [How does pythTQC read the spin of my model?](../guide/spin.md) reads a
  spinor calculation, which pythtb imports as a spinless model with the spin
  inside the orbital index, declared with `spin=`.
- [Returned records](../reference/records.md#every-field) lists every field of
  `numerical.precision`, `numerical.positions` and `numerical.symmetrization`.
- [Which little-group representations do these bands carry?](../guide/labels.md),
  [Could these bands be an atomic limit?](../guide/atomic-limits.md) and
  [Which symmetry-indicator class do these bands occupy?](../guide/indicators.md)
  go further with the calls of
  [The bands, labeled in the model's group](#the-bands-labeled-in-the-models-group).
- [Functions](../reference/functions.md) gives every argument of
  [`realized_group`](../reference/functions.md#realized_group),
  [`hsp_table`](../reference/functions.md#hsp_table),
  [`decompose`](../reference/functions.md#decompose),
  [`symmetry_indicator`](../reference/functions.md#symmetry_indicator) and
  [`wyckoff_sites_tables`](../reference/functions.md#wyckoff_sites_tables).
