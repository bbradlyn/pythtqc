# How does pythTQC read the spin of my model?

The little-group representations of a spinful model's bands, and the site
irreps of its orbitals, are double-valued. Each operation acts on an orbital's
two spin states by a fixed 2x2 matrix, the one the double-valued irreps used by
the irreptables package are written with. The group returned and the
representations of the bands are the model's own only where its spin blocks
are written in the convention that matrix assumes. This page shows that
convention, how to convert a model whose spin up is the state along `−z`, and
how to declare where the spin is in a model imported from a Wannier90
spin-orbit calculation, which arrives spinless with the spin in the orbital
index. The model is Kane-Mele, from `pythtb.models`,
`kane_mele(delta, t, soc, rashba)`: one orbital with spin on each sublattice of
the honeycomb, a staggered potential `±delta`, hopping `t`, spin-orbit coupling
`soc` and Rashba coupling `rashba`.

## The call

`spinless_presentation(model)` takes a spinful pythtb model and returns two
things. The first is the same Hamiltonian written as a spinless model, its
spinless presentation, in which each orbital's two spin states are two
orbitals at its center. The second is the pairing of those orbitals, which
`spin=` takes. pythtb's Wannier90 importer delivers a spin-orbit calculation in
this form, so the presentation is a model to compare a real import with.

`spin=` on `realized_group` declares which orbitals of a spinless model are the
up and down states of one spinful orbital, an orbital that holds both spin
states. `"interleaved"` declares the order `(0↑, 0↓, 1↑, 1↓, …)`, and
[A Wannier90 spinor import](#a-wannier90-spinor-import) gives the other forms.
With it, `realized_group` reads each pair as one spinful orbital at the pair's
center. The group returned, and the little-group representations read from it,
are those of the native spinful model. A pair whose two members sit at
different centers is refused with `spin_pair_split`. `typat=` takes one entry
per orbital, the same entry for the two members of a pair.

The example reads Kane-Mele's spinless presentation with `spin="interleaved"`
and the native model, and labels the two lowest bands of each. Both declare
the site irrep of orbital 0 as `A1`:

```python
import numpy as np
import pythtqc as tqc
from pythtb import TBModel
from pythtb.models import kane_mele

km = kane_mele(1.0, -1.0, 0.3, 0.25)            # delta, t, soc, rashba: two orbitals, with spin
flat, basis = tqc.spinless_presentation(km)     # as pythtb's Wannier90 importer delivers it
print(flat.nspin, flat.norb)
#  1 4
r = tqc.realized_group(flat, typat=[0, 0, 1, 1], spin="interleaved", characters={0: "A1"})
print(r.group.name, r.group.number_str, tqc.hsp_table(r, bands=2).symmetry_data)
#  P3m11' 156.50 {'-GM6': 1, '-K5': 1, '-K6': 1, '-M3M4': 1}
native = tqc.realized_group(km, typat=[0, 1], characters={0: "A1"})
print(native.group.name, native.group.number_str, tqc.hsp_table(native, bands=2).symmetry_data)
#  P3m11' 156.50 {'-GM6': 1, '-K5': 1, '-K6': 1, '-M3M4': 1}
print(r.wyckoff_orbits.summary())
#  Wyckoff  site symmetry  orbitals  label  determined by  declared
#       1c           3m1'       0-1    -E1    declaration        A1
#       1b           3m1'       2-3    -E1    hamiltonian
```

## How to read it

- `1 4`: `flat.nspin` and `flat.norb`. The presentation is spinless, and
  Kane-Mele's two orbitals with spin are its four orbitals. Nothing in `H(k)`
  says which orbital is the up partner of which, so the pairing is declared.
- `P3m11' 156.50`: the group of the presentation read with `spin=`, and on the
  next line the same group of the native model. `P3m11'` is `P3m1` with time
  reversal.
- `{'-GM6': 1, '-K5': 1, '-K6': 1, '-M3M4': 1}`: the corep multiplicities of
  the two lowest bands at each high-symmetry point, their symmetry data vector,
  `hsp_table(r, bands=2).symmetry_data`
  ([Which little-group representations do these bands carry?](labels.md)).
  The leading `-` marks a double-valued corep, and `GM` is the tables' name
  for Γ. The presentation and the native model have the same symmetry data
  vector.
- `1c 3m1' 0-1 -E1 declaration A1`: the row of the atomic-limit table for
  orbitals 0 and 1, pair 0, at Wyckoff position `1c`, whose site-symmetry group
  is `3m1'`. Its site irrep is `-E1`, which is `A1` of `3m` times the spinor. It
  was declared, as `A1`. The two members of a pair stand in one row and carry
  one site irrep
  ([How do I say what my orbitals are?](orbitals.md#how-to-read-it) reads every
  column).
- `1b 3m1' 2-3 -E1 hamiltonian`: pair 1, at `1b`. Nothing was declared for it.
  The hoppings that connect it to pair 0, whose site irrep was declared, fix
  its site irrep.

## A pair's site irrep

`characters=` declares a pair's site irrep as for a native spinful model: by
its orbital part, a single-valued site irrep such as `A1`, or by the
double-valued site irrep itself, such as `-E1`. An entry on either member
declares the pair: `{0: "A1"}` and `{1: "A1"}` both declare pair 0. In a list
with one entry per orbital, the two members of a pair take the same site-irrep
name, or a name and `None`. A key naming orbitals of several pairs declares
what those pairs carry together, by its orbital part. On a site holding the
pairs `(0, 1)` and `(2, 3)`, `{(0, 2): "E"}` declares that the four states
carry the doublet `E` times the spinor.

## The convention, in the model's numbers

In pythtb, spin up is the state along `+z` of the Cartesian axes in which the
pythtb Lattice is specified. Every 2x2 spin block, on-site and hopping, is
written in `(↑, ↓)` order. The double-valued irreps used by the irreptables
package are written in the same convention, with a rotation by `θ` about `n̂`
acting on the spinor as `e^{−iθ n̂·σ/2}`, in a right-handed frame
([Spin](../background/conventions.md#spin)). Every matrix pythTQC builds acts
on the spin this way. A spinful pythtb model therefore needs no conversion.

The block prints the matrices of three operations of the native Kane-Mele
model on orbital 0's two states. Each operation fixes the site of orbital 0,
and its block is the orbital character, `+1` for `A1`, times the operation's
spin matrix:

```python
names = tqc.operations(native.group)
for ind in (2, 5, 7):                           # C3(z), m(x), and time reversal
    print(names[ind].name)
    print(np.round(native.rep.matrices[ind][:2, :2], 3) + 0)
#  C3(z)
#  [[0.5-0.866j 0. +0.j   ]
#   [0. +0.j    0.5+0.866j]]
#  m(x)
#  [[0.+0.j 0.-1.j]
#   [0.-1.j 0.+0.j]]
#  E'
#  [[ 0.+0.j  1.+0.j]
#   [-1.+0.j  0.+0.j]]
```

- `C3(z)`: the counterclockwise rotation by `2π/3` about `+z`. Its block is
  `diag(e^{−iπ/3}, e^{+iπ/3}) = e^{−i(2π/3)σ_z/2}`, which fixes at once the
  axis, the order of the two states, the sign in the exponent and the
  handedness of the frame.
- `m(x)`: the mirror whose normal is x, the inversion times the twofold
  rotation `C2(x)`. The inversion does not act on the spin, so its spin matrix
  is that of `C2(x)` alone, `−iσ_x = e^{−iπσ_x/2}`.
- `E'`: time reversal. Its block is `iσ_y`, and it acts as that matrix times
  complex conjugation.

## A model written for the other axis

A source whose spin up is the state along `−z` writes a rotation by `θ` about
`z` as `e^{+iθσ_z/2}`. The supplement of the paper on the stable real-space
invariants,
[Hwang et al., Nat. Commun. **17**, 8564 (2026)](https://doi.org/10.1038/s41467-026-74844-w),
writes its fourfold rotation about `z` this way. In its models that rotation
acts as `τ_1 e^{+iπσ_3/4}`, where `τ_1` acts on the orbitals and `σ_3` is the
paper's `σ_z`. A model's bands do not show whether its spin up is along `+z`
or `−z`. Conjugating every spin block by `W = iσ_x` conjugates `H(k)` by the
unitary `1_N ⊗ W`, `N` the number of orbitals, so no band, gap or Wilson loop
changes. What can change is which operations `H(k)` obeys, each operation
acting on the spin by the tables' matrix. The group returned and the coreps
change with them. The conversion belongs where the source is transcribed,
block by block:

```python template
W = 1j * np.array([[0, 1], [1, 0]])                      # i sigma_x
model = TBModel(lattice, spinful=True)
for i, block in enumerate(onsite_blocks):                # the source's 2x2 blocks, along -z
    model.set_onsite(W @ block @ W.conj().T, i)
for amplitude, i, j, R in hoppings:
    model.set_hop(W @ amplitude @ W.conj().T, i, j, R)
```

[`examples/tqc_srsi_p4_flatband.py`](../../examples/tqc_srsi_p4_flatband.py)
converts that paper's two Hamiltonians this way. Spin up written along `−z`
can change the group returned. The example conjugates every spin block of
Kane-Mele by `W`, which writes the same spectrum with its spin up along `−z`,
and reads its group:

```python
def respun(model, U):                           # every 2x2 spin block conjugated by U
    out = TBModel(model.lattice, spinful=True)
    for i, block in enumerate(model.onsite):
        out.set_onsite(U @ block @ U.conj().T, i)
    for hop in model.hoppings:
        out.set_hop(U @ hop["amplitude"] @ U.conj().T, hop["from_orbital"],
                    hop["to_orbital"], hop.get("lattice_vector", [0] * model.dim_r))
    return out

W = 1j * np.array([[0, 1], [1, 0]])             # i sigma_x
minus_z = respun(km, W)                         # Kane-Mele written along -z
k = np.random.default_rng(0).uniform(-0.5, 0.5, size=(20, 2))
print(np.allclose(km.solve_ham(k), minus_z.solve_ham(k), atol=1e-12))
#  True
wrong = tqc.realized_group(minus_z, typat=[0, 1], characters={0: "A1"})
print(repr(wrong))
#  RealizedGroup(Cm1' #8.33; descended from P3m11' #156.50 (8 of 12 coset representatives of the lattice translations broken; these are the single-valued group's, each standing for itself and its negative in the double group); on a substrate (the default), the free-standing group Cm1' #8.33 agrees)
print({b.refuted_by for b in wrong.descent.dropped})
#  {'tensor structure'}
back = tqc.realized_group(respun(minus_z, W), typat=[0, 1], characters={0: "A1"})
print(back.group.name, back.group.number_str)
#  P3m11' 156.50
```

- `True`: the two spectra agree at twenty random k-points.
- `Cm1' #8.33`: the group the model written along `−z` obeys.
- `descended from P3m11' #156.50 (8 of 12 coset representatives of the lattice
  translations broken; ...)`: the group of the cell is `P3m11'`, and the
  Hamiltonian breaks 8 of its 12 operations, counted one per coset of the
  lattice translations. For a spinful model each is an operation of the
  single-valued group, standing for itself and its negative in the double
  group.
- `on a substrate (the default), the free-standing group Cm1' #8.33 agrees`:
  the layer was read on a substrate, the default for a 2D model, and the
  free-standing layer has the same group
  ([Is my 2D model on a substrate or free-standing?](layers.md)).
- `{'tensor structure'}`: what refuted each dropped operation, from
  `wrong.descent.dropped` ([The descent](../reference/records.md#the-descent)).
  `tensor structure` means that at every `k` some matrix maps `H(k)` to `H` at
  the k-point the operation takes `k` to, but none that is an orbital part
  times the operation's own spin matrix.
- `P3m11' 156.50`: conjugated back by `W`, the model obeys `P3m11'` again.

Each operation is tested with its own spin matrix, the one the tables fix.
Conjugation by `W` turns the spin matrix of each threefold rotation into that
of its inverse, and exchanges the spin matrices of two of the three mirrors.
`H(k)` then obeys those operations only with a spin matrix that is not their
own, and each is refuted by tensor structure. Their products with time
reversal are refuted too, and the group descends to `Cm1'`. The mirror that
stays, `m(x)`, acts on the spin as `−iσ_x`, which commutes with `W`.

A model whose spin blocks are turned by a rotation about a generic axis, the
lattice left as it is, keeps only the operations whose spin matrix commutes
with every rotation: time reversal, and the inversion where the crystal has
one. Kane-Mele with a staggered potential has no inversion, and keeps time
reversal alone:

```python
n = np.array([0.3, 0.5, 0.8]) / np.linalg.norm([0.3, 0.5, 0.8])
sigma = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
U = np.cos(0.35) * np.eye(2) - 1j * np.sin(0.35) * np.tensordot(n, sigma, axes=1)
turned = tqc.realized_group(respun(km, U), typat=[0, 1], characters={0: "A1"})
print(turned.group.name, turned.group.number_str)
#  P11' 1.2
```

A crystal turned whole, its lattice and its spin blocks by the same rotation,
keeps its group and the coreps of its bands.

Whether a small turn of the spin blocks breaks an operation is decided by
`tol`, a tolerance in the units of `H` for checking whether an operation is a
symmetry. An operation is kept when `H` misses obeying it by at most `tol`.
With no precision declared, `tol` is `1e-8` times `max|H|`
([How do I measure a numerical or Wannier90 model?](numerical.md) gives it
under a declared precision). A turn that changes `H` by at most `tol` breaks
nothing, and `H` is projected onto its symmetric part. A larger turn breaks
the operations whose spin matrix it changes. Where the turn is small enough
for a declared precision to keep every operation it broke, the record's
`descent.note` names the least precision, declared with `precision=`, that
does.

## A Wannier90 spinor import

pythtb's Wannier90 importer returns a spinless model whatever the calculation
did. Each Wannier function of a spin-orbit calculation arrives as one orbital,
so the up and the down state of a spinful orbital are two orbitals of the
model. `spin=` takes three forms:

- `"interleaved"`, the order `(0↑, 0↓, 1↑, 1↓, …)`.
- `"blocked"`, the order `(0↑, 1↑, …, 0↓, 1↓, …)`.
- The pairs one by one, up first, for any other order, as
  `spin_basis(model, pairs=…)` or the mapping `{"pairs": …}`.

Every call that takes `r` reads the model as it reads a native spinful one.
`r.spin` holds the pairing as `realized_group` read it, to be checked before
the coreps returned are relied on. The example prints it for Kane-Mele's
interleaved presentation read above, reads the blocked presentation with
`spin="blocked"`, and declares the interleaved pairs one by one:

```python
print(r.spin.summary())
#  each pair one Wannier function, its spin up and down along z   (.pairs)
#    pair 0: orbital 0 up, orbital 1 down
#    pair 1: orbital 2 up, orbital 3 down
blocked, _ = tqc.spinless_presentation(km, order="blocked")
rb = tqc.realized_group(blocked, typat=[0, 1, 0, 1], spin="blocked", characters={0: "A1"})
print(rb.group.number_str, tqc.hsp_table(rb, bands=2).symmetry_data)
#  156.50 {'-GM6': 1, '-K5': 1, '-K6': 1, '-M3M4': 1}
print(tqc.spin_basis(flat, pairs=[(0, 1), (2, 3)]).summary())
#  each pair one Wannier function, its spin up and down along z   (.pairs)
#    pair 0: orbital 0 up, orbital 1 down
#    pair 1: orbital 2 up, orbital 3 down
```

- `(.pairs)`: the attribute that holds the pairs, `r.spin.pairs`.
- `pair 0: orbital 0 up, orbital 1 down`: orbitals 0 and 1 are the states with
  spin along `+z` and `−z` of one spinful orbital.
- `156.50 {'-GM6': 1, ...}`: the blocked presentation, read with
  `spin="blocked"`, has the group and the symmetry data vector of the
  interleaved one.

The pairs are declared, and not each orbital's spin alone: with two up and two
down orbitals at one site, the spins do not say which up goes with which down.

`spin_operator(basis, n)` takes the pairing and an axis `n`, in the Cartesian
axes in which the pythtb Lattice is specified. It returns `n̂·σ` on each pair,
`n̂` the unit vector along `n`, in units of `ħ/2`. Projected onto the occupied
bands, `P (n̂·σ) P`, it is the operator a spin-resolved calculation uses. Where
the eigenvalues of that projection stay away from zero, they split the bands by
the sign of their spin along `n̂`, and the spin Chern number is read on that
split. The block prints it along x for Kane-Mele's interleaved presentation:

```python
print(tqc.spin_operator(basis, n=(1, 0, 0)).real + 0)  # sigma_x on each pair
#  [[0. 1. 0. 0.]
#   [1. 0. 0. 0.]
#   [0. 0. 0. 1.]
#   [0. 0. 1. 0.]]
```

Without `spin=`, Kane-Mele's spinless presentation is read as a spinless
model:

```python
bare = tqc.realized_group(flat, typat=[0, 0, 1, 1])     # no spin=: read as nspin = 1 says
print(bare.group.name, bare.group.number_str, bare.time_reversal.outcome,
      bare.time_reversal.square.measured, tuple(o.label for o in bare.wyckoff_orbits))
#  P3m1 156 measured broken -1.0 ('E', 'E')
```

- `P3m1 156`: the group returned, space group 156 without time reversal.
- `measured broken`: `bare.time_reversal.outcome`, time reversal measured and
  broken.
- `-1.0`: `bare.time_reversal.square.measured`, the square of time reversal on
  these states, `-1` as on a spinor. Time reversal in the single-valued tables
  squares to `+1`, so it is broken.
- `('E', 'E')`: each site's two states read as the doublet `E` of `3m`.

Where two operations that commute have spinor matrices that anticommute, as
two perpendicular twofold rotations do, these states carry no single-valued
representation of a group holding both. The group then descends further, to
the largest subgroup on which they carry one, and each operation left out has
`refuted_by = "projective class"`
([The descent](../reference/records.md#the-descent)). On a layer, the subgroup
is chosen for the free-standing layer
([The horizontal mirror, and what breaks it](layers.md#the-horizontal-mirror-and-what-breaks-it)).
On a substrate, the layer keeps those operations of that subgroup that do not
turn the layer over. Kane-Mele without a staggered potential,
`kane_mele(0.0, -1.0, 0.3, 0.25)`, read this way is `P6/m` free-standing and
`P6` on a substrate. Neither is the group of the spinful crystal, which
`spin=` reads.

## When it refuses

A refusal is a returned record. It is falsy, and names the check that stopped
the call and what it measured
([What does a refusal mean, and what do I do about it?](refusals.md)). The
checks these calls meet:

- [`rep_double_valued_spinless`](../reference/refusals.md#rep_double_valued_spinless):
  `spinor=True`, which asks for the double-valued tables, passed for a
  spinless model without `spin=`.
- [`spin_pair_split`](../reference/refusals.md#spin_pair_split): a declared
  pair whose two members sit at different centers. With `precision=` declared,
  two centers closer than the precision are one site.
- [`not_a_spinful_model`](../reference/refusals.md#not_a_spinful_model):
  `spinless_presentation` of a model that is already spinless.

Two members of one pair declared with different site-irrep names raise
`ValueError`. The block meets each of these on Kane-Mele's spinless
presentation:

```python
print(tqc.realized_group(flat, typat=[0, 0, 1, 1], spinor=True).gate)
#  rep_double_valued_spinless
try:
    tqc.realized_group(flat, typat=[0, 0, 1, 1], spin="interleaved",
                       characters=["A1", "A2", None, None])
except ValueError as error:
    print(str(error).split(". ")[-1])
#  Orbitals 0 and 1, spin pair 0, are given 'A1' and 'A2'; give both the same value, or declare only one of them.
split = tqc.spin_basis(flat, pairs=[(0, 2), (1, 3)])    # each pair across two sites
print(tqc.realized_group(flat, typat=[0, 0, 1, 1], spin=split).gate)
#  spin_pair_split
print(tqc.spinless_presentation(flat).gate)
#  not_a_spinful_model
```

A pairing whose pairs each sit at one center is not refused, even where it
does not match the model's spin blocks. It declares another crystal, with a
smaller group. Each operation that `H(k)` does not obey with its own spin
matrix on the declared pairs is refuted by tensor structure. Time reversal is
refuted too where no antiunitary symmetry of `H(k)` with time reversal's
action on `k` acts as `iσ_y` times complex conjugation on every declared pair.
Pairs that join the states of two spinful orbitals at one site read this way.
Up and down exchanged in every pair is the model written with its spin up
along `−z`. The block declares the exchanged pairs on Kane-Mele's spinless
presentation, and the group descends to `Cm1'`, as it did above:

```python
swapped = tqc.realized_group(flat, typat=[0, 0, 1, 1], spin={"pairs": [(1, 0), (3, 2)]},
                             characters={0: "A1"})
print(swapped.group.name, swapped.group.number_str)
#  Cm1' 8.33
```

## Where to go next

- [How do I measure a numerical or Wannier90 model?](numerical.md) says how
  `precision=` is declared, how the orbital centers are placed and how a real
  import is symmetrized.
- [Silicon from Wannier90: the group the model has](../start/tutorial-wannier90.md)
  reads a real import end to end.
- [Conventions: operations, labels and band representations](../background/conventions.md#spin)
  gives, under "Spin", the spin matrices the tables are written with.
- [How do I say what my orbitals are?](orbitals.md) lists the site-irrep names
  a pair is declared with.
- [Functions](../reference/functions.md) gives every argument of
  [`spin_basis`](../reference/functions.md#spin_basis),
  [`spin_operator`](../reference/functions.md#spin_operator),
  [`spinless_presentation`](../reference/functions.md#spinless_presentation)
  and [`realized_group`](../reference/functions.md#realized_group).
