# How does pythtqc read the spin of my model?

The labels of a spinful model are double-valued: each is an orbital factor
times the operations' own action on the spin, and that action is a convention
the model's numbers have to follow. This chapter says what the numbers must
be, how to convert a model written for the other spin axis, and how to say
where the spin is in a wannier90 import, which arrives with the spin inside
the orbital index. The model is pythtb's Kane–Mele,
`kane_mele(delta, t, soc, rashba)`, natively spinful, and its spinless
presentation: the same Hamiltonian as a wannier90 spin–orbit calculation would
deliver it.

## The call

```python
import numpy as np
import pythtqc as tqc
from pythtb import TBModel
from pythtb.models import kane_mele

km = kane_mele(1.0, -1.0, 0.3, 0.25)            # delta, t, soc, rashba: two orbitals, with spin
flat, basis = tqc.spinless_presentation(km)     # the same crystal as wannier90 delivers it
print(flat.nspin, flat.norb)
#  1 4
r = tqc.realized_group(flat, typat=[0, 0, 1, 1], spin="interleaved", characters={0: "A1"})
print(r.group.name, r.group.number_str, tqc.hsp_table(r.rep, bands=2).counts)
#  P3m11' 156.50 {'-GM6': 1, '-K5': 1, '-K6': 1, '-M3M4': 1}
native = tqc.realized_group(km, typat=[0, 1], characters={0: "A1"})
print(native.group.name, native.group.number_str, tqc.hsp_table(native.rep, bands=2).counts)
#  P3m11' 156.50 {'-GM6': 1, '-K5': 1, '-K6': 1, '-M3M4': 1}
print(r.rep.characters.summary())
#  orbital  Wyckoff  site symmetry  function  spin  irrep  named by  determined by  orbital characters
#        0       1c             3m              up    -E1    tables    declaration  E=+1, C3(z)=+1, m(x)=+1
#        1       1c             3m            down    -E1    tables    declaration  E=+1, C3(z)=+1, m(x)=+1
#        2       1b             3m              up    -E1    tables    hamiltonian  E=+1, C3(z)=+1, m(x)=+1
#        3       1b             3m            down    -E1    tables    hamiltonian  E=+1, C3(z)=+1, m(x)=+1
```

## How to read it

- `1 4`: what pythtb's wannier90 importer returns whatever the calculation
  did, a spinless model with two orbitals per spinor Wannier function, the
  two spin states of each at one centre. `H(k)` is blind to the names of its
  indices, so nothing measured says which orbital is the `↑` partner of which:
  the pairing is a declaration, the one piece of physics an import cannot
  supply.
- `spin="interleaved"` declares it: `(0↑, 0↓, 1↑, 1↓)`. `typat=` takes one
  entry per orbital, the two members of a pair one atom. `characters=` states
  a pair's spatial factor, exactly as for a native spinful model, and an entry
  on either member speaks for the pair: `{0: "A1"}` declares pair 0, an
  ordered list gives a pair's two members one word or `None` for one of them,
  and a key of several orbitals groups the pairs they belong to: on two pairs
  that share a site, `(0, 1)` and `(2, 3)`, `{(0, 2): "E"}` declares one
  doublet.
- The two group lines agree: `P3m11'`, and the same corep content of the two
  lowest bands at every high-symmetry point (`hsp_table`,
  [labelling the bands](labels.md)), for the import and for the native model.
- The atomic-limit table grows a `spin` column, which member of its pair each
  row is; the two members carry the pair's one site corep
  ([reading the table](orbitals.md#how-to-read-it)).

## The convention, in the model's numbers

pythtb writes spin along `+z` of the model's Cartesian frame and every `2×2`
block, on-site and hopping, in `(↑, ↓)` order; a rotation by `θ` about `n̂`
acts on the spinor as `e^{−iθ n̂·σ/2}`. That is the lift the tables'
double-valued labels are written with
([conventions](../background/conventions.md#spin)), and every matrix the
package builds carries it. Each operation below fixes orbital 0's site, and
its block on orbital 0's two states is the orbital character, `+1` for `A1`,
times the lift:

```python
names = tqc.operations(native)
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

- `C3(z)`, a counterclockwise turn by `2π/3` about `+z`, is
  `diag(e^{−iπ/3}, e^{+iπ/3}) = e^{−i(2π/3)σ_z/2}`: the axis, the order of the
  two states, the sign in the exponent and a right-handed frame, in one block.
- `m(x)`, the mirror whose normal is x, is the inversion times the twofold
  rotation `C2(x)`, and its block is `−iσ_x = e^{−iπσ_x/2}`, the lift of
  `C2(x)` alone. The spin does not transform under the inversion,
  `S(i·R) = S(R)`, so a spinful `s` orbital at an inversion centre is the even
  (gerade) spinor.
- `E'`, time reversal, is `iσ_y` times complex conjugation.

## A model written for the other axis

A source that writes a rotation as `e^{+iθ n̂·σ/2}` is quantized along `−z`:
the supplement of the paper on stable real-space invariants,
[Hwang et al., Nat. Commun. **17**, 8564 (2026)](https://doi.org/10.1038/s41467-026-74844-w),
is one, its fourfold operator `τ₁·e^{+iπσ₃/4}`. Its bands cannot say so.
Conjugating every spin block by `W = iσ_x` is the unitary `1_N ⊗ W` on
`H(k)`, so no band, gap or Wilson loop moves; what moves is every comparison
with the tables' lift, and with it the group and the labels. The conversion
belongs where the source is transcribed, block by block:

```python template
W = 1j * np.array([[0, 1], [1, 0]])                      # i sigma_x
model = TBModel(lattice, spinful=True)
for i, block in enumerate(onsite_blocks):                # the source's 2x2 blocks, along -z
    model.set_onsite(W @ block @ W.conj().T, i)
for amplitude, i, j, R in hoppings:
    model.set_hop(W @ amplitude @ W.conj().T, i, j, R)
```

[`examples/tqc_srsi_p4_flatband.py`](../../examples/tqc_srsi_p4_flatband.py)
does it for that paper's two Hamiltonians. A wrong axis shows in the group.
Kane–Mele with every spin block conjugated by `W` is the
same spectrum written along `−z`:

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
#  RealizedGroup(Cm1' #8.33; descended from #156.50 (8 of 12 coset representatives of the lattice translations broken; these are the single-valued group's, each standing for itself and its negative in the double group); on a substrate (the default), the free-standing group #8.33 agrees)
print({b["refuted_by"] for b in wrong.descent["dropped"]})
#  {'tensor structure'}
back = tqc.realized_group(respun(minus_z, W), typat=[0, 1], characters={0: "A1"})
print(back.group.name, back.group.number_str)
#  P3m11' 156.50
```

The spectra agree and the groups do not. Every operation is held to its own
lift, so one whose action on `k` the hoppings carry only through another spin
factor is refuted by tensor structure: the threefold rotations and two of the
three mirrors go, each with its partner combined with time reversal, and the
group descends to `Cm1'`. The mirror that stays, `m(x)`, has the lift `−iσ_x`,
which commutes with `W`. Conjugating back restores `P3m11'`. Spin blocks turned
by a generic rotation, the lattice left where it is, keep only the operations
whose lift no rotation of the frame changes: time reversal, and the inversion
where the crystal has one.

```python
n = np.array([0.3, 0.5, 0.8]) / np.linalg.norm([0.3, 0.5, 0.8])
sigma = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
U = np.cos(0.35) * np.eye(2) - 1j * np.sin(0.35) * np.tensordot(n, sigma, axes=1)
turned = tqc.realized_group(respun(km, U), typat=[0, 1], characters={0: "A1"})
print(turned.group.name, turned.group.number_str)
#  P11' 1.2
```

A crystal turned whole, lattice and spin blocks together, keeps its group and
its labels: the frame is the lattice's, and only spin blocks turned against it
change the group.

A turn by a small angle is read against the tolerance of the call. Where what
it does to `H` is within `tol`, every operation holds, as on any model symmetric
only within its precision, and `H` is symmetrized onto the group. Beyond it,
the operations whose lift the turn changes are broken, and where a precision a
model's numbers could have holds them, the note of the descent names it.

## A wannier90 spinor import

pythtb's `wannier90` importer returns `nspin = 1` whatever the calculation
did, so a spin–orbit calculation with `N` Wannier functions arrives as a
spinless model with `2N` orbitals. `spin=` declares which are partners, in
three spellings: `"interleaved"`, `(0↑, 0↓, 1↑, 1↓, …)`; `"blocked"`,
`(0↑, 1↑, …, 0↓, 1↓, …)`; and the pairs one by one, for an irregular
wannier90 order, as `spin_basis(model, pairs=…)` or the mapping
`{"pairs": …}`. Read the pairing back before trusting a label:

```python
print(basis.summary())
#  interleaved, on 4 orbitals   (.kind, .norb)
#    each pair one Wannier function, its spin up and down along z   (.pairs)
#      pair 0: orbital 0 up, orbital 1 down
#      pair 1: orbital 2 up, orbital 3 down
blocked, _ = tqc.spinless_presentation(km, order="blocked")
rb = tqc.realized_group(blocked, typat=[0, 1, 0, 1], spin="blocked", characters={0: "A1"})
print(rb.group.number_str, tqc.hsp_table(rb.rep, bands=2).counts)
#  156.50 {'-GM6': 1, '-K5': 1, '-K6': 1, '-M3M4': 1}
print(tqc.spin_basis(flat, pairs=[(0, 1), (2, 3)]).summary())
#  listed pair by pair, on 4 orbitals   (.kind, .norb)
#    each pair one Wannier function, its spin up and down along z   (.pairs)
#      pair 0: orbital 0 up, orbital 1 down
#      pair 1: orbital 2 up, orbital 3 down
```

With the pairing declared, each pair is one spinor Wannier function, keeping
its own centre: the operations act on its two states by their own SU(2) lift,
and everything downstream reads as for a native spinful model. The pairing is
what is declared, since each orbital's `s_z` alone fixes no basis: with two up
and two down orbitals at one site it does not say which up goes with which
down. The spin operator is built from the pairing along any axis `n̂`:
`spin_operator(basis, n)` is `n̂·σ` on each pair, the operator a spin-resolved
calculation projects onto the occupied bands.

```python
print(tqc.spin_operator(basis, n=(1, 0, 0)).real + 0)  # sigma_x on each pair
#  [[0. 1. 0. 0.]
#   [1. 0. 0. 0.]
#   [0. 0. 0. 1.]
#   [0. 0. 1. 0.]]
```

Without any declaration the model is what `nspin = 1` says, a spinless one:

```python
bare = tqc.realized_group(flat, typat=[0, 0, 1, 1])     # no spin=: read as nspin = 1 says
print(bare.group.name, bare.group.number_str, bare.time_reversal["outcome"],
      bare.time_reversal["square"]["measured"], bare.rep.characters.irreps)
#  P3m1 156 measured broken -1.0 ('E', 'E', 'E', 'E')
```

Its time reversal, measured, squares to `−1`, a spinor's, which the
single-valued tables do not describe: time reversal is measured broken, the
unitary group `P3m1` is returned, and each site's two states read as the
doublet `E` of `3m`. Where two commuting operations' spinor matrices
anticommute, as two perpendicular twofolds' do, no single-valued
representation exists, and the group descends further, to the largest
subgroup that carries one, each operation left out recorded with
`refuted_by = "projective class"` ([the descent](../reference/records.md#the-descent)).
A layer's largest such subgroup is found once, free-standing, where each
operation composed with the horizontal mirror carries its partner's matrix
times the mirror's sign
([a spinless layer's mirror](../background/limits.md#a-spinless-free-standing-layers-horizontal-mirror)),
and on a substrate the layer keeps the operations of that subgroup that keep
its side: read this way, Kane–Mele without a staggered potential,
`kane_mele(0.0, -1.0, 0.3, 0.25)`, is `P6/m` free-standing and `P6` on a
substrate. Neither is the spinful crystal: declare the pairing.

## When it refuses

A spinless model asked for in double-valued tables (`spinor=True`), with
nothing said about where its spin is, is refused
[`rep_double_valued_spinless`](../reference/refusals.md#rep_double_valued_spinless).
Two members of one pair declared different words raise `ValueError` before
anything is measured, since up and down are one Wannier function with one site
character. A pairing whose
two members sit at different centres is no spinor basis, the two spin states of
one Wannier function sharing its centre, and it means one of two things: a
mislabelled pairing, which an interleaved file read as blocked, or the reverse,
makes wherever each site holds one Wannier function; or orbitals that each
carry one spin state with no partner at their own centre, a magnetic
configuration outside the package's present scope. The centres alone say so, and
`realized_group` refuses it
[`spin_pair_split`](../reference/refusals.md#spin_pair_split) before any group
is identified, naming none. Two centres apart only by the noise of a fitted
model are neither: a declared precision (`precision=`) reads them as one.
`spinless_presentation` of a model that is already spinless is refused
[`not_a_spinful_model`](../reference/refusals.md#not_a_spinful_model):

```python
print(tqc.realized_group(flat, typat=[0, 0, 1, 1], spinor=True).gate)
#  rep_double_valued_spinless
try:
    tqc.realized_group(flat, typat=[0, 0, 1, 1], spin="interleaved",
                       characters=["A1", "A2", None, None])
except ValueError as error:
    print(str(error).split(". ")[-1])
#  Orbitals 0 and 1, spin pair 0, are given 'A1' and 'A2'; give both the same word, or declare only one of them.
split = tqc.spin_basis(flat, pairs=[(0, 2), (1, 3)])    # each pair across two sites
print(tqc.realized_group(flat, typat=[0, 0, 1, 1], spin=split).gate)
#  spin_pair_split
print(tqc.spinless_presentation(flat).gate)
#  not_a_spinful_model
```

A pairing that is well formed and wrong is read: it states
another crystal, whose group is smaller. Which orbitals form a pair is read
from the declaration, in whatever order a site lists its orbitals; each
operation whose action the declared pairs do not carry is refuted by tensor
structure, and time reversal with them where no member of its solution space
acts as `iσ_y` on every pair, as on pairs that join the states of two Wannier
functions of one site. Up and down exchanged in every pair is the model
written along `−z`, and the group descends as it did above:

```python
swapped = tqc.realized_group(flat, typat=[0, 0, 1, 1], spin={"pairs": [(1, 0), (3, 2)]},
                             characters={0: "A1"})
print(swapped.group.name, swapped.group.number_str)
#  Cm1' 8.33
```

## Where next

- [Numerical and wannier90 models](numerical.md): `precision=`, the orbital
  centres, and the symmetrizing of a real import.
- [Silicon from wannier90](../start/tutorial-wannier90.md): a real import,
  end to end.
- [Conventions](../background/conventions.md#spin): the lift the tables are
  written with.
- [How do I say what my orbitals are?](orbitals.md): the words a pair's
  spatial factor is declared in.
- Every argument of [`spin_basis`](../reference/functions.md#spin_basis),
  [`spin_operator`](../reference/functions.md#spin_operator),
  [`spinless_presentation`](../reference/functions.md#spinless_presentation)
  and [`realized_group`](../reference/functions.md#realized_group).
