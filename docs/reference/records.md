# Returned records

A call returns what it measured as a record, a frozen object whose fields hold
the measurement, or a `Refusal` in its place where it cannot complete; the few
calls that return a plain mapping or a number say so in their entries. This
page takes the records a physicist reads in the order a calculation meets them:
which call returns each, its printout and how to read it, and every field,
named as the physics it holds. The blocks build one model at the top and print
every record from it, with four variants of the same model where a printout has
more to show and one table that needs no model. `help(tqc.RealizedGroup)`, or
any other record's name in its place, prints the whole of a record's docstring;
every argument of every call is in [the functions reference](functions.md), and
every check a call can refuse at is in [the refusal reference](refusals.md). The
words are [the glossary's](glossary.md).

- [The model on this page](#the-model-on-this-page)
- [`RealizedGroup`](#realizedgroup), and [the descent](#the-descent)
- [`GroupData`](#groupdata), [`OperationTable`](#operationtable-and-operation),
  [`OrbitalAssignment`](#orbitalassignment), [`ValidatedGroup`](#validatedgroup),
  [`SpaceGroupRep`](#spacegrouprep)
- [The atomic-limit table](#the-atomic-limit-table), and
  [`BandRepresentation`](#bandrepresentation)
- [`CharacterTable` and `CorepContent`](#charactertable-and-corepcontent),
  [`HSPTable`](#hsptable)
- [`EBRDecomposition`](#ebrdecomposition), and
  [`CompletenessReport` and `RegularCompletion`](#completenessreport-and-regularcompletion)
- [`SymmetryIndicator`](#symmetryindicator)
- [The stable real-space invariants](#the-stable-real-space-invariants)
- [`Refusal`](#refusal)
- [The other records](#the-other-records)

## The model on this page

Kane and Mele's honeycomb, from pythtb's collection,
`kane_mele(delta, t, soc, rashba)`, with the intrinsic spin–orbit coupling and
nothing else: four bands, two of them occupied. It is read as a layer on a
substrate, the default for a two-dimensional model, and as the same layer
free-standing; orbital 0 is declared an `s` orbital.

```python
import numpy as np
import pythtqc as tqc
from pythtb.models import kane_mele

km = kane_mele(0.0, -1.0, 0.3, 0.0)           # no staggered potential, no Rashba term
r = tqc.realized_group(km, characters={0: "s"})
model, group, assignment, validated, rep = r
t = tqc.hsp_table(rep, bands=2)                 # the two occupied bands
fs = tqc.realized_group(km, environment="free-standing", characters={0: "s"})
tf = tqc.hsp_table(fs.rep, bands=2)
```

`r` and `fs` are what `realized_group` returns, and `t` and `tf` the content of
the occupied pair at every high-symmetry point; every record below is one of
these or is returned by a call on them, except where a variant built in
[the printout, clause by clause](#the-printout-clause-by-clause) shows what this
model cannot.

## `RealizedGroup`

What `realized_group` returns: the space group the Hamiltonian obeys, the
Wyckoff orbit of each orbital, the verdict on each operation and the model's
representation of the group, measured together and kept with the model they
were measured on. Why the group is measured rather than read off the positions
is in [how pythtqc reads a Hamiltonian](../background/measurement.md).

```python
print(r)
#  RealizedGroup(P6mm1' #183.186; additional commuting matrices: 1; on a substrate (the default), the free-standing group #191.234 agrees)
```

The Hamiltonian obeys `P6mm1'`, space group 183 with time reversal adjoined,
which the tables file as `183.186`, every one of its operations measured to
hold; one matrix beyond the identity commutes with `H(k)` at every `k`, the spin
component `s_z` this model conserves without a Rashba term; and the layer was
read on a substrate, by default, with the free-standing layer measured as well
and found to agree.

`summary()` reads the record part by part, each line ending with the attribute
that holds that part:

```python
print(r.summary())
#  P6mm1' #183.186, on a substrate   (.group)
#    its operations, by index and Seitz symbol   (.operations)
#    the orbitals, by Wyckoff position   (.rep.characters)
#      2b (3m): orbitals 0-1, -E1
#    one more matrix commutes with H(k) at every k   (.conserved)
```

### The printout, clause by clause

The printout names the group and adds a clause for each thing the call did
beyond identifying it. Four variants of the same honeycomb print the clauses
the model above does not: a staggered Zeeman term, `+h σ_z` on one sublattice
and `−h σ_z` on the other, with the two sublattices declared different atoms
(`typat=[0, 1]`), which breaks time reversal; the two sublattices declared
different irreps of their site group, `A1` and `A2`, with the Rashba term on;
the honeycomb with one centre moved by `1e-8` in reduced coordinates and every
hopping changed at random by about `1e-7` of itself, as a model fitted to a
calculation is, with its precision declared; and the honeycomb of the quick
start, `kane_mele(1.0, -1.0, 0.3, 0.25)`, read without its spin.
`spinless_presentation` writes its states as pythtb's wannier90 importer
delivers a spinor calculation, a spinless model with the spin inside the
orbital index, and nothing declares which orbitals are spin partners
([spin](../guide/spin.md)).

```python
from pythtb import Lattice, TBModel

zeeman = kane_mele(0.0, -1.0, 0.3, 0.0)
zeeman.set_onsite([[0, 0, 0, 0.3], [0, 0, 0, -0.3]], mode="add")
broken = tqc.realized_group(zeeman, typat=[0, 1])
print(broken)
#  RealizedGroup(P3 #143; time reversal measured broken; descended from #156.50 (6 of 12 coset representatives of the lattice translations broken; these are the single-valued group's, each standing for itself and its negative in the double group); additional commuting matrices: 1; on a substrate (the default), the free-standing group #174 agrees -- a larger magnetic group, of type III, is realized (3 antiunitary operations measured to hold) but is beyond the validated scope)

declared = tqc.realized_group(kane_mele(0.0, -1.0, 0.3, 0.25), characters={0: "A1", 1: "A2"})
print(declared)
#  RealizedGroup(P31' #143.2; cell lowered by the declared content from #183.186 (12 of 24 coset representatives of the lattice translations excluded; these are the single-valued group's, each standing for itself and its negative in the double group); descended from #156.50 (6 of 12 broken by the declared content); on a substrate (the default), the free-standing group #143.2 agrees)

rng = np.random.default_rng(7)
centres = np.asarray(km.orb_vecs, dtype=float)
centres[0, 0] += 1e-8
fitted = TBModel(Lattice(np.asarray(km.lat_vecs, dtype=float), centres.tolist(),
                         periodic_dirs=[0, 1]), spinful=True)
fitted.set_onsite(np.asarray(km.onsite), mode="set")
for h in km.hoppings:
    fitted.set_hop(np.asarray(h["amplitude"]) * (1 + 1e-7 * rng.normal()), h["from_orbital"],
                   h["to_orbital"], h.get("lattice_vector", [0, 0]), mode="add")
numerical = tqc.realized_group(fitted, characters={0: "s"}, precision=1e-6)
print(numerical)
#  RealizedGroup(P6mm1' #183.186; precision 1e-06 declared -> tol 2.97e-05 (cut 1e-05, window 0.0316), position_tol 1e-06 (measured max|H| 2.969); additional commuting matrices: 1; centres moved 5e-09; symmetrized (defect 1.13e-07); on a substrate (the default), the free-standing group #191.234 agrees)

flat, basis = tqc.spinless_presentation(kane_mele(1.0, -1.0, 0.3, 0.25))
unpaired = tqc.realized_group(flat)
print(unpaired)
#  RealizedGroup(P3m1 #156; time reversal measured broken, descended from P3m11' #156.50: its square is -1, so the states carry half-integer spin, and spin= declares which orbitals are spin partners; descended from #183.186 (12 of 24 coset representatives of the lattice translations broken); on a substrate (the default), the free-standing group #187 agrees)
```

The calls on the Zeeman variant and on the spinless reading also issue a
`MagneticScopeWarning`, on the error stream rather than in the printout: the
group returned is the unitary subgroup of the Hamiltonian's magnetic group, and
operations that combine a unitary operation with time reversal are not searched
for beyond the cell's own. Where a clause counts operations, it counts one per
coset of the lattice translations, and for a spinful model the single-valued
group's, each standing for itself and its negative in the double group: the
first count of a printout names that unit, and every count after it is in the
same unit. The clauses, in the order they print:

| clause | what it says | on the record |
|---|---|---|
| `P6mm1' #183.186` | the group the Hamiltonian obeys: its Hermann–Mauguin symbol and the label the tables file it under. A trailing `1'` marks time reversal adjoined, a grey group; `P3 #143` has none | `group` |
| `precision 1e-06 declared -> tol 2.97e-05 (cut 1e-05, window 0.0316), position_tol 1e-06 (measured max\|H\| 2.969)` | a declared precision, and the tolerances it derived against the Hamiltonian's measured magnitude: `tol = 10 × precision × max\|H\|`, `cut = tol / max\|H\|`. `, symprec raised to …` follows where the declared noise in the positions would defeat the symmetry finder | `precision` |
| `time reversal measured broken, descended from P3m11' #156.50: its square is -1, …` | time reversal is broken, so the cell was identified again without it, and the grey group it was identified from is named; where its square is `-1` on states read without spin, as on the spinless reading, the clause says the states carry half-integer spin and names `spin=`, which reads them as spinors; where operations of the cell combined with time reversal hold, as in the Zeeman variant, the clause ends at `broken` and the last clause names what holds | `time_reversal` |
| `cell lowered by the declared content from #183.186 (12 of 24 coset representatives of the lattice translations excluded; …)` | a reduction: 12 of the cell's 24 coset representatives do not carry the declared content, so they were never operations of the declared crystal, and the cell was identified again before any hopping was read | `decoration["declared"]` |
| `descended from #156.50 (6 of 12 coset representatives of the lattice translations broken; …)` | a descent: the Hamiltonian broke 6 of the 12 coset representatives of the group it was measured in, and the cell was lowered to a group of the operations that hold. Where the hoppings did not break them all, the parenthesis says what did: `by the centres` (operations the orbital centres are not carried by within a declared precision), `by the declared content`, as in the declared variant's `(6 of 12 broken by the declared content)`, or the count split between them. Where operations the Hamiltonian obeys are among them, left out by the states' projective class, to close a group or by the tie rule, they were dropped rather than broken, and the parenthesis counts each cause, as it does for a descent of several steps: `descended from #123.340 in 2 steps (28 of 32 coset representatives of the lattice translations dropped: 16 by the hoppings, 8 with time reversal, 4 by the states' projective class)` | `descent` |
| `additional commuting matrices: 1` | matrices beyond the identity that commute with `H(k)` at every `k`: symmetry beyond the crystal's own, outside the group the labels are read in | `conserved` |
| `centres moved 5e-09` | the orbital centres were moved onto the positions the group puts them at, about the origin that fits them: the largest move, in reduced coordinates | `positions` |
| `symmetrized (defect 1.13e-07)` | the Hamiltonian was moved onto its symmetric part, by at most this much in the units of `H` | `symmetrization` |
| `on a substrate (the default)` | a layer's environment, and that nobody declared it: `free-standing` is the other, and `(the default)` is absent where `environment=` was given | `environment`, `environment_declared` |
| `the free-standing group #191.234 agrees` | the other environment, measured on its own and compared: the substrate group is the subgroup of the free-standing one that preserves the layer's normal. Where they do not agree it reads `the other environment refused` or `the other environment ambiguous`; where the declaration is written in a substrate's site irreps, which the free-standing layer has no word for, `not compared with the free-standing layer: the declaration is written in the substrate's site irreps` | `cross_check` |
| `-- a larger magnetic group, of type III, is realized (3 antiunitary operations measured to hold) but is beyond the validated scope` | operations of the cell combined with time reversal that hold while time reversal itself is broken: with the group returned, a magnetic group beyond the grey groups, its type named | `beyond_grey` |

A three-dimensional model prints no environment, so one that nothing else
happened to prints its group alone.

### Every field

`model, group, assignment, validated, rep = r` binds five of the record's own
fields, so the unpacked `rep` is `r.rep`. The rest are reached by name.

| field | what it holds |
|---|---|
| `model` | the model measured, which every later call takes: a layer's slab presentation, its `H(k)` measured identical to the layer's, or through the move `presentation` records; a three-dimensional model itself; the symmetrized model, where one was made |
| `source` | the model exactly as it was handed in |
| `group` | the space group the Hamiltonian obeys ([`GroupData`](#groupdata)) |
| `assignment` | the Wyckoff orbit of each orbital, and how each operation moves the sites ([`OrbitalAssignment`](#orbitalassignment)) |
| `validated` | the verdict on each operation ([`ValidatedGroup`](#validatedgroup)) |
| `rep` | the matrix of each operation on the orbitals, and the atomic-limit table ([`SpaceGroupRep`](#spacegrouprep)) |
| `operations` | the group's operations by index, one per coset of the lattice translations, each with its Seitz symbol and its name in the model's frame ([`OperationTable`](#operationtable-and-operation)) |
| `descent` | the operations dropped and what dropped each, or `None` where the Hamiltonian obeys every unitary operation its cell allows ([the descent](#the-descent)) |
| `time_reversal` | how time reversal was measured |
| `conserved` | symmetry beyond the crystal's own |
| `environment` | `"substrate"` or `"free-standing"`; `None` for a three-dimensional model |
| `environment_declared` | whether `environment=` was given |
| `cross_check` | the other environment, measured and compared; `None` for a three-dimensional model and with `cross_check=False` |
| `decoration` | what was added to the cell and why: `dz`, the depth of a substrate under the layer; `typat` and `n_atoms`, the added atoms' species and their number; `recipe`, the reason in a sentence; and `declared`, a reduction by declared content, or `None` |
| `precision` | what a declared precision derived; `None` without one |
| `positions` | how far the centres missed the group's own action, and their correction; `None` without a `position_tol`, declared or derived from `precision` |
| `symmetrization` | how far the Hamiltonian was moved onto its symmetric part; `None` where nothing needed it |
| `beyond_grey` | the cell's operations combined with time reversal that hold while time reversal itself is broken; `None` otherwise |
| `presentation` | how a layer was presented as a slab: `c`, the slab's lattice constant; `residual`, the measured difference between the slab's `H(k)` and the layer's at `n_check` random `k`, which is zero; and `tilted`, `None` unless the layer was handed in as a slab whose non-periodic lattice vector was tilted off the layer's normal, as one cut from a crystal along a lattice vector of the crystal is. A layer's group cannot depend on that choice, so the vector is put on the normal with every orbital where it was, and `tilted` records the move: `tilt`, the vector's in-plane part in reduced coordinates of the periodic directions; `max_move` and `n_moved`, each orbital's reduced coordinates in the plane having moved by its reduced height times the tilt; `spectrum_residual`, the measured difference of the two spectra; and a `note`. The move is a change of gauge, which moves no energy, and `residual` is then measured through it, at rounding. A three-dimensional model is its own presentation, `c` and `tilted` reading `None` |
| `cell` | the point set the group was identified on, the orbitals and any species added to them, read for its positions alone |
| `typat`, `cell_typat` | the species of each orbital the cell was read with, and the same with the added atoms' |
| `bundle` | the five unpacked values, in order |

```python
print(broken.time_reversal["outcome"], broken.time_reversal["grey_name"], "->", broken.time_reversal["name"])
#  measured broken P31' -> P3
print(r.time_reversal["outcome"], r.time_reversal["n_obeyed"], "of", r.time_reversal["n_antiunitary"])
#  kept 12 of 12
print(r.conserved["dim"])
#  1
print(r.environment, r.environment_declared, r.cross_check["outcome"], r.cross_check["free_standing"])
#  substrate False agree {'number_str': '191.234', 'name': "P6/mmm1'", 'size': 48}
```

`time_reversal` is time reversal as measured: the cell is
asked as a grey group, since positions say nothing about time reversal, and
the Hamiltonian obeys the operations combined with time reversal or not.

| key | what it holds |
|---|---|
| `outcome` | `"kept"`, every operation combined with time reversal holds and the group is grey; or `"measured broken"`, they are refuted and the group returned is the unitary group, identified again from the cell |
| `n_antiunitary`, `n_obeyed`, `n_broken` | the operations combined with time reversal of the grey group measured, and how many hold, counted by the rule `beyond_grey` names its operations by: where time reversal is broken, one whose unitary part is in the group returned makes time reversal with it and does not hold, so none holds unless a descent after time reversal was measured, by the states' projective class, left that unitary part out |
| `undetermined` | those that hold with a verdict their `k`-action class shares (a flat layer's) |
| `grey_name`, `grey_number_str` | the grey group measured |
| `name`, `number_str`, `size` | where time reversal is broken, the unitary group returned |
| `square` | time reversal's own square `X X̄`, a sign no rescaling of its matrix `X` changes: `asked`, the sign the tables read ask for, `+1` single-valued and `−1` double-valued, and `measured`, the Hamiltonian's. Differing, they break time reversal: spinor states read without their spin are the case. `None` where time reversal has no matrix at all |
| `note` | the measurement in a sentence |

`conserved` counts symmetry beyond the crystal's: `dim`, the matrices beyond the
identity that commute with `H(k)` at every `k`, with the `note` saying so. A number
above zero is ordinary physics, a residual spin rotation, a decoupled sublattice or
an accidental degeneracy. Where a multiplet of orbitals is decoupled from the rest, what it
carries relative to the rest is not a measurement this Hamiltonian can make, and
its row of [the atomic-limit table](#the-atomic-limit-table) says whether it was
declared or assigned. `sectors` says how the conserved matrices that act within
sites split the states, read once where the model entered and at the declared
precision, a spin rotation never among them: on each sector they are every
matrix acting on `copies` identical copies of a set of `dim` orbitals (or
declared spinor pairs), each entry naming the `orbitals` its states lie on and
the `sites` those sit at. A crystal whose hoppings connect every orbital is one
sector of one copy, a `p_x, p_y` pair that no hopping couples is two sectors in
whatever basis the pair is written, and two identical lattices that no hopping
joins are one sector of two copies. `images` gives, per operation by index, the
sector it carries each sector onto.

`cross_check` holds the other environment's measurement: `outcome`, `"agree"`,
`"ambiguous"` (the tie rule kept a subgroup the other environment did not
measure, and the statement holds for another pair of the two environments'
candidates, which the note names: no conflict about the Hamiltonian),
`"refused"` (the other environment refused, with its `refusal`) or
`"not compared"` (the declaration
is written in a substrate's site irreps, which the free-standing layer's site
groups name nothing for, so it says a substrate is meant and the free-standing
layer is not read with it; a free-standing layer's words are read on the
substrate by restriction, and compared); `agrees`; `free_standing` and
`substrate`, each group's label, symbol and size, the other environment's
`None` where it refused or was not compared; `rows`, each substrate
operation and the free-standing operation it is; `missing`, `extra`, `flipping`
and `unmapped`, the four ways the comparison can fail, each naming its
operations; `tie`, the operations a tie left out; and `note`, the comparison in
words. A genuine mismatch raises `InternalError`: two measurements of one
Hamiltonian disagreeing is the package disagreeing with itself.

A model fitted to a calculation carries three more records:

```python
p = numerical.precision
print(p["precision"], round(p["max_h"], 3), f"{p['tol']:.3g}", p["position_tol"], f"{p['window']:.3g}",
      p["symprec"], p["symprec_raised"])
#  1e-06 2.969 2.97e-05 1e-06 0.0316 1e-05 False
q = numerical.positions
print(f"{q['defect']:.1e}", q["position_tol"], q["adjusted"], f"{q['max_move']:.1e}", q["n_moved"],
      q["frame"], f"{max(abs(x) for x in q['offset']):.1e}")
#  2.0e-08 1e-06 True 5.0e-09 1 model 5.0e-09
s = numerical.symmetrization
print(f"{s.defect:.3g}", f"{s.defect_relative:.3g}", len(s.near), len(s.exact), s.passes)
#  1.13e-07 3.76e-08 22 2 1
print(numerical.source is fitted, numerical.model is fitted)
#  True False
```

- `precision`: the declared relative `precision`; `max_h`, the Hamiltonian's
  largest entry measured about its mean on-site energy, in the units of `H`;
  the four tolerances derived from them, `tol`, `cut`, `window` and
  `position_tol`; and `symprec` with `symprec_raised`, the symmetry finder's
  tolerance and whether the declaration raised it. Each is in
  [the glossary](glossary.md#tolerances), with its unit.
- `positions`: `defect`, how far the centres as handed in miss the group's own
  action, `max |A·τ_a + t − τ_g(a) − s|` in reduced coordinates;
  `position_tol`, the bound the declaration puts on each centre's move;
  `adjusted`, whether the centres were read, and moved onto the positions of the
  largest group of the operations that carries them within `position_tol`, the
  origin fitted to them, a shift of every centre together being no
  displacement; `max_move`, the largest such move; `offset`, the shift of every
  centre together that then writes them in the model's own frame, at most
  `position_tol`, so that each returned centre is the one handed in plus its own
  move plus `offset`; `frame`, `"model"` where they are written in the model's
  own frame, every operation's translation a rational of the cell, and
  `"fitted"` where the shift that would do it is larger than `position_tol`, or
  there is none, and they stay about the origin fitted to them; then
  `defect_before` and `defect_after`, against the operations that carry them,
  `n_moved`, and `spectrum_residual` with `n_check`, the measured statement
  that the move left the spectrum where it was. Where the whole group did not
  carry the centres, `move` is what it would have asked of one and `carried`
  how many operations do, and the rest are broken by the centres on `descent`.
  Moving a centre is a `k`-dependent change of basis, so no band moves; `note`
  says what was done.
- `symmetrization`, a `Symmetrization`: `defect`, how far `H(k)` moved, in the
  units of `H`, and `defect_relative`, the same against its magnitude `scale`;
  `near` and `exact`, the operations that were near-symmetric, which licensed
  the projection, and those already exact, with their measured `residuals` and
  `classes`; `tol` and `exact_gate`, the tolerances of the two checks the
  classes were read against; `operations`, the operations averaged over, and
  `not_represented`, those kept that the representation does not carry, a slab
  cell's operations on its vacuum;
  `multiplets`, how far the orbitals' multiplets held only within the declared
  precision; `parts`, the parts of the crystal coupled to one another only
  below it and how large that coupling was, which the projection leaves out;
  `copies`, identical copies coupled or told apart only below it, which the
  projection holds as exact copies; `shares`, states that split multiplets
  between them, the bonding and the antibonding state of two lattices joined
  on site or a square's `p_x, p_y` pair, coupled only below it, which the projection
  holds apart; `passes`; `verification`, the symmetrized model measured again at
  the default tolerance; and `grid`, `checks`, `operators` and `machine_floor`,
  the numbers the projection's exactness rests on. Its `summary()` reads it
  part by part. `model` is then the symmetrized model and `source` the one
  handed in.

`beyond_grey` names the operations of the cell combined with time reversal
that hold while time reversal itself is broken. Each entry of `operations`
carries the operation's `name`, its index `ind` in the group it was measured in
and that group's label, `of`, as an entry of [the descent](#the-descent) does,
and its rotation, translation, residual and verdict. Where the translation is
not a lattice vector of the cell, the name carries it in Seitz form, in the
cell's reduced coordinates: an antiferromagnetic chain's half translation
combined with time reversal is `{E|1/2,0,0}'`.
Beside them: their number, `n_operations`, out of the cell's operations
combined with time reversal measured, `n_measured`; whether the operations kept
close in each group they were measured in, `closed`; the magnetic type they
make with the group returned, `"III"`, `"IV"` or `"magnetic, beyond grey"`; and
a `note`.

Each operation named combines time reversal with a unitary operation that does
not hold, which a descent left out. The group the descent arrives at does not
contain that unitary part and never asks it, so the operation carries the
verdict of the group the descent lowered, and is named by its index there. The
Zeeman variant is the case. Each vertical mirror flips `σ_z` on the sublattice
it fixes, and so does time reversal, so each mirror is broken and each mirror
combined with time reversal holds; the cell's own group, which `descend=False`
asks for and refuses where the Hamiltonian breaks any of it, lists them among
the operations measured to hold:

```python
print([(o["name"], o["of"]) for o in broken.beyond_grey["operations"]], broken.beyond_grey["magnetic_type"])
#  [("m([010])'", '156.50'), ("m(x)'", '156.50'), ("m([1-10])'", '156.50')] III
allowed = tqc.realized_group(zeeman, typat=[0, 1], descend=False)
print(allowed.gate, [op.name for op in allowed.context["validated"].operations.values()])
#  cell_group_not_realized ['E', 'C3(z)', 'C3^-1(z)', "m([010])'", "m(x)'", "m([1-10])'"]
```

The six operations listed make a magnetic group of type III, and the group
returned is its unitary half, `P3`. An operation combined with time reversal
that is not an operation of the cell is not searched for: `C2(z)` exchanges the
two sublattices declared different atoms, so `C2(z)` combined with time
reversal, which this Hamiltonian obeys, is named nowhere
([magnetic groups](../background/limits.md#magnetic-groups-of-types-i-and-ii-only)).

Each operation is read once, after the group has settled, against the group
returned, and is held to one rule: it holds only where a matrix measured for it
and the matrices of the group returned make a corepresentation of the magnetic
group it generates with that group. Every operation of that group combined with
time reversal then has a symmetry matrix, and with `X` the matrix of the
operation `a` and `D(h)` that of each operation `h` of the group returned,
`X X̄ = D(a²)` and `X D(h)* X⁻¹ = D(a h a⁻¹)`, up to a one-dimensional
character of the group returned: the labelling, which the reading does not
depend on. An operation can have a symmetry matrix and fail the rule, and then
it does not hold. A descent after time reversal is measured is read the same
way. A spin-orbit-coupled `(p_x, p_y)` doublet of `P4/mmm`, read without its
spin, descends by its states' projective class to `P4/m`, and the eight
operations of `4/mmm` outside `4/m`, combined with time reversal, make
`P4/mm'm'` with it: they are named, each measured in `P4/mmm1'`, the group time
reversal was measured on, and counted in its `n_obeyed`.

An operation combined with time reversal whose unitary part holds is never
named: with that unitary part it makes time reversal, which is broken. The
spinless reading is the case. Read in single-valued tables, time reversal's
square is the spinor's `−1` where the tables ask for `+1`, so time reversal is
broken. Each mirror of `P3m1` combined with time reversal is a symmetry of
`H(k)` and squares to `+1`, as the tables ask; with the mirror itself, which
holds, it makes time reversal, whose square is `−1`. No group read in
single-valued tables holds the two, so none of the six operations combined
with time reversal of `P3m11'`, the grey group time reversal was measured on,
counts: nothing beyond grey is realized, and the printout names that grey group
instead. It is the larger group these states obey, and `spin=` reads it in the
double-valued tables ([spin](../guide/spin.md)):

```python
square = unpaired.time_reversal["square"]
print(square["asked"], round(square["measured"], 6), unpaired.time_reversal["n_obeyed"], "of",
      unpaired.time_reversal["n_antiunitary"], unpaired.beyond_grey)
#  1 -1.0 0 of 6 None
```

## The descent

`r.descent` is `None` where the Hamiltonian obeys every unitary operation its
cell allows, the common case. Otherwise the cell was lowered to a group of the
operations measured to hold and identified again, and this is that measurement.
Some operations it drops are ones the Hamiltonian obeys: those that do not carry
the orbital centres within a declared precision; those on which spinor states
read without their spin carry no single-valued representation; and those left
out so that the rest close into a group, or by the tie rule where several
largest groups tie. The cell is lowered to the largest group the reading allows,
and the `note` says which reading dropped what. The
Zeeman variant above was lowered by the hoppings, and the declared one by its
declaration:

```python
d = broken.descent
print(d["from"], "->", d["to"])
#  {'number_str': '156.50', 'name': "P3m11'", 'size': 12} -> {'number_str': '143.2', 'name': "P31'", 'size': 6}
print(d["dropped_by"], sorted({b["refuted_by"] for b in d["dropped"]}))
#  {'hoppings': 6} ['no symmetry matrix']
b = d["dropped"][0]
print(b["ind"], b["name"], b["refuted_by"], b["step"], b["of"])
#  4 m([010]) no symmetry matrix 1 156.50
e = declared.descent
b = e["dropped"][0]
print(e["dropped_by"], b["name"], b["refuted_by"], b["orbitals"])
#  {'declaration': 6} m([010]) declaration (0, 1)
print(b["note"])
#  relative to orbitals (0,), the hoppings carry orbitals (1,) with +1 under this operation, and the declarations ((0,) +1, (1,) -1) state -1
reduction = declared.decoration["declared"]
x = reduction["excluded"][0]
print(reduction["from"]["number_str"], "->", reduction["to"]["number_str"],
      reduction["n_excluded"], x["name"], x["reason"])
#  183.186 -> 156.50 12 C2(z) declared content differs across the orbit
```

The Zeeman term is odd under the three vertical mirrors and under time
reversal. The six operations of the cell it breaks, the mirrors and time
reversal alone and with each threefold rotation, are each refuted by `"no
symmetry matrix"`: no matrix with the operation's action on `k` carries `H(k)`
onto itself, a proof. The cell was lowered
to the unitary operations that hold, the threefold rotations, and identified
again as the grey group `P31'`; time reversal, measured there, is broken, which
is the clause printed before the descent. In the declared variant each mirror
fixes both sublattices, and relative to orbital 0 the hoppings carry orbital 1
with `+1` under it, where the declaration, `A1` against `A2`, states `−1`: the
crystal declared has no such mirror, so the group descends to one that carries
the declared content. The twelve operations exchanging the sublattices are on
`decoration["declared"]` instead: the two sublattices were declared different
content, so those operations were never operations of the declared crystal,
and nothing is charged to the Hamiltonian twice.

`refuted_by` charges each operation dropped to what dropped it:

| `refuted_by` | what refuted the operation |
|---|---|
| `"no symmetry matrix"` | no matrix with the operation's action on `k` carries `H(k)` onto itself: a proof that the Hamiltonian breaks it |
| `"covariance"` | such a matrix exists, and its covariance residual is above `tol` |
| `"tensor structure"` | the operation's action on `k` survives, and no matrix of it has the form `O(g) ⊗ S(g)`: it shares its action on `k` with a partner it cannot be told from, or, with spin, its spin factor is not its own lift `S(g)`, which is how spin blocks written in a frame turned against the lattice show, by any angle whose effect on `H` is beyond `tol`. A pure translation shares the identity's action on `k`, so a broken one reads this way wherever this test refutes it, as the checkerboard's does |
| `"out of plane"` | an operation of a layer's slab that the model's `k`-space cannot represent |
| `"translation"` | a pure translation the hoppings break where the tensor structure did not refute it: `D H(k) D† = H(k)` fails for the permutation `D` it makes of the sites, and for every change of basis of each site's orbitals beside it that commutes with the other translations' |
| `"declaration"` | the declared orbital content is not carried by the operation |
| `"projective class"` | the hoppings obey it, and no single-valued representation on these states carries it: spinor states read without their spin, where it commutes with an operation whose matrix anticommutes with its own |
| `"closure"` | the operations that hold do not close into a group, and no subgroup inside them holds this one |
| `"tie"` | several subgroups of the largest order tie, and the one [the tie rule](../background/measurement.md#the-descent-and-the-tie-rule) keeps does not hold this one |

Every entry of `dropped` carries the operation's `ind`, its `name` in the frame
`characters=` is written in, `refuted_by`, the `residual` the verdict rests on
(`nan` where none does: a declaration, a projective class, a closure or a tie),
`step`, the descent that dropped it, and `of`, the group that step was measured
in. A name carries the operation's translation where that is not a lattice
vector of the cell: a broken half translation is `{E|1/2,0,0}`
([the conventions](../background/conventions.md#operations-a-seitz-symbol-and-a-cartesian-name)).
A tensor-structure entry adds, where the operation's form was missed by more
than rounding, `covariance`: how far the matrix of that form nearest its
solution space carries `H(k)` from `H(gk)`, against `tol`; a declaration entry `orbitals`, `declared`, `measured` and a `note`;
a projective-class entry the anticommuting `pair` and their `names`, the
`commutation_phase`, whether it is the sign spin-½ lifts carry (`spin_half`), and a
`note`. The rest of the record:

| key | what it holds |
|---|---|
| `from`, `to` | the group the cell allowed and the group the descent arrived at, each `number_str`, `name` and `size`; where time reversal is then measured broken, as in the Zeeman variant, the group returned is `to`'s unitary half |
| `dropped_by` | how many operations each cause dropped, every step together: `"centres"`, `"hoppings"`, `"declaration"`, `"closure"`, `"tie"`, `"time reversal"` and `"projective class"`, in the order the measurement meets them, each that dropped any; together, the group the cell allowed less the group the descent arrived at |
| `tied` | where a declared content leaves several readings of the largest order, all of them, in the order the tie rule ranks them, the one returned first |
| `cell` | the orbit that carries the group the descent arrived at: the generic point added to the cell, its orbit under the operations kept, and the species it was added as |
| `iterations`, `steps` | how many descents ran, and each one's own record: its groups, the operations it dropped (`dropped`, `n_dropped`), a count by `refuted_by`, the operations kept, and the tie or the declaration's readings (`tied`) where there were some; where the states' projective class lowered the group, under `projective`, the pairs that name it, what was kept and, where subgroups of that order tied, why the first was kept (`reason`), and on a layer the sign the horizontal mirror was given free-standing (`horizontal_mirror`) or, on a substrate, the free-standing group the class was read on (`read_on`) |
| `lattice` | `None` unless the cell was a supercell of its own point set and the hoppings broke the translation; then that first descent, with each translation's measured residual |
| `note` | the descent in words: what dropped the operations, counted by cause over every step, with operations left out to close a group or by the tie rule, and time reversal measured broken between two steps, among the causes, what the operations kept carry, the tie rule's choice and why, and the reduction; where time reversal is then measured broken, what holds, the unitary operations and those combined with time reversal, and that the unitary group alone is returned; where every operation broken was broken by its own form beyond `tol`, and a precision a model's numbers could have holds them, that precision |

A reduction, `decoration["declared"]`, holds `from` and `to`, the group the
cell allowed and the one it was identified as; `excluded`, one entry per
operation the declared crystal does not have, each with its `ind`, its `name`,
the `sites` and `orbitals` concerned, the `reason`, and a `note`, the exclusion
in words; `n_excluded`; `kept`, the operations kept; `pairs`, each unitary
operation kept as its index in the group before and in the group after; and
`generic_point`, `orbit` and `species`, the orbit that carries the reduced
cell. The `reason` is `"declared content differs across the orbit"`, a
declared function not carried onto its image, or, for an operation the
declared content is carried by on its own, that it is not carried by every
product of the operation with the operations kept, or that a group as large as
the one kept holds the operation and the tie rule keeps the other. The
operations kept are the largest group whose every operation carries the
declared content, the first by
[the tie rule](../background/measurement.md#the-descent-and-the-tie-rule)
where several of that order tie, and the note of an operation the rule left
out names the groups that tie. That is every operation the content is carried
by where the declaration states every site of an orbit, as a function at all of
them or at none; elsewhere it can be fewer, and a site left undeclared carries
what that group carries onto it.

## `GroupData`

The space group, in the model's own cell: `r.group`, and the group every table
call takes. Its `summary()` reads it part by part:

```python
print(group)
#  GroupData(P6mm1' #183.186, spinor=True, TR=True, periodic=(0, 1), realized)
print(group.summary())
#  P6mm1' #183.186   (.name, .number_str)
#    its operations, one per coset of the lattice translations   (.symmetries)
#    the double group also holds each one's negative   (.spinor)
#    a layer: periodic along the first two lattice vectors, vacuum along the third   (.periodic_dirs, .vacuum_dirs)
print(group.name, group.number_str, group.sg_number, group.size, group.spinor, group.provenance)
#  P6mm1' 183.186 183 24 True realized
```

The first printout adds two things the summary does not print: time reversal
adjoined (`TR`), and the group measured on a Hamiltonian (`realized`).

| field | what it holds |
|---|---|
| `name`, `number_str`, `sg_number` | the Hermann–Mauguin symbol, the label the tables file the group under (`"183.186"`), and the space-group number (`183`) |
| `size` | how many operations `symmetries` lists, one per coset of the lattice translations, each standing for itself and its negative where the group is the double group |
| `spinor`, `include_TR`, `magnetic` | whether the group is the double group, whether time reversal is adjoined, and whether it is read in the magnetic tables, which a grey group is |
| `provenance` | `"realized"`, measured on a Hamiltonian, or `"descended"`, measured after the cell was lowered |
| `symmetries`, `u_symmetries`, `au_symmetries` | the operations themselves, as `irrep` records them, in the tables' order; the unitary ones; the antiunitary ones |
| `lattice`, `periodic_dirs`, `vacuum_dirs`, `vacuum_length`, `dim_k`, `dim_r` | the three-dimensional cell the group was found on, rows the lattice vectors; which of its axes are periodic and which are a layer's vacuum, and the vacuum's length; the model's own periodic and spatial dimensions |
| `site_positions`, `site_orbitals`, `typat` | the distinct sites in reduced coordinates, the orbitals on each, and each site's species |
| `symprec` | the symmetry finder's tolerance, a Cartesian length |

Its methods are the group's own arithmetic: `get_product_table()` and
`get_inverse_table()`, its multiplication table and inverses; `k_action_of(op)`,
an operation's action on the model's reduced `k`; and `k_to_3d`, `k_from_3d` and
`in_plane`, which carry `k` between a layer and its slab.

## `OperationTable` and `Operation`

The operations of a group, by index: `r.operations`, `tqc.operations(...)`, and
`rep.operations` for the ones a representation carries. A mapping from each
operation's index to its `Operation`, which prints as the whole correspondence
under a line saying what its rows are, as its `summary()` returns it; an
`Operation`'s own `summary()` reads it part by part:

```python
print(broken.operations)
#  P3 #143: one operation per coset of the lattice translations; the double group also holds each one's negative
#   ind  Seitz           name
#     1  {1|0,0,0}       E
#     2  {3+_001|0,0,0}  C3(z)
#     3  {3-_001|0,0,0}  C3^-1(z)
op = r.operations[2]
print(op.ind, op.seitz, op.name, op.time_reversal)
#  2 {3+_001|0,0,0} C3(z) False
print(r.operations[14].summary())
#  operation 14: {3+_001|0,0,0}' in the tables' setting, C3(z)' in the model's frame   (.ind, .seitz, .name)
#    its rotation and translation in the model's reduced coordinates   (.rotation, .translation)
#    its rotation in the model's Cartesian axes   (.rotation_cartesian)
#    its matrix on the spin, spin along the z of those axes   (.spin_lift)
#    combined with time reversal, which the ' marks   (.time_reversal)
```

An `Operation` holds `ind`, its index in the tables' order, which every
per-operation record is keyed on; `seitz`, its Seitz symbol `{R|t}` in the
tables' conventional setting; `name`, the same operation in the model's own
Cartesian frame, the frame `characters=` is written in, an axis along none of
its x, y, z named by its direction in the lattice vectors; `rotation` and
`translation`, in the model's reduced coordinates; `rotation_cartesian`, the
rotation in the Cartesian axes the model's lattice vectors are written in, a
layer's normal as z; `spin_lift`, its SU(2) matrix on the model's spin, with
its axis in those same axes and spin along their z, `None` where the states
carry no spin; and `time_reversal`, whether it is antiunitary, which both
spellings mark with a trailing `'`. How the symbols and names are spelled is in
[the conventions](../background/conventions.md#operations-a-seitz-symbol-and-a-cartesian-name).

## `OrbitalAssignment`

Which site orbit each orbital sits on, and how the operations move the sites:
`r.assignment`, which `summary()` reads part by part.

```python
print(assignment)
#  OrbitalAssignment(2 orbitals on 2 sites: sites (0, 1), letter b, site symmetry 3m)
print(assignment.summary())
#  2 orbitals on 2 sites   (.site_of_orbital, .site_positions)
#    the orbits, by Wyckoff letter in the tables' setting and site symmetry   (.orbits, .wyckoff_tables, .site_symmetry)
#      b (3m): sites 0-1, orbitals 0-1
#    the operations that fix each site, by index   (.site_group)
#    the site each operation carries each site onto, and the lattice vector it lands in   (.permutations, .lattice_shift)
print(assignment.wyckoff_tables, assignment.wyckoff_spglib, assignment.wyckoff_spglib_dataset)
#  ('b', 'b') ('b',) ('c', 'c')
```

A Wyckoff letter depends on the setting, and three are measured:
`wyckoff_tables`, one per site, the letter in the tables' own setting, which
every table is read with; `wyckoff_spglib`, one per orbit, and
`wyckoff_spglib_dataset`, one per site, the letters in spglib's
standardization, which are information about the geometry and never a table
key. Here spglib's two disagree with each other, `b` against `c`.

| field | what it holds |
|---|---|
| `orbits` | the sites, grouped into the orbits the operations carry them through |
| `orbit_of_site`, `orbit_of_orbital`, `site_of_orbital` | which orbit each site is on, which orbit each orbital is on, and which site each orbital sits on |
| `site_positions`, `site_orbitals` | the sites, and the orbitals on each |
| `site_group`, `site_symmetry` | per site, the indices of the operations that fix it, and their point group's Hermann–Mauguin symbol |
| `permutations`, `lattice_shift` | per operation and site, the site it is carried to, and the lattice vector the image lands in, which sets the phases a representation is built with |
| `wyckoff_tables`, `wyckoff_spglib`, `wyckoff_spglib_dataset` | the Wyckoff letters, as above |
| `max_distance` | the largest Cartesian mismatch accepted while matching images to sites |

## `ValidatedGroup`

The verdict on each operation of a group: `r.validated`, which `summary()`
reads part by part.

```python
print(validated)
#  ValidatedGroup(P6mm1' #183.186, 24 of 24 coset representatives of the lattice translations obeyed, which form a group; they are the single-valued group's, each standing for itself and its negative in the double group)
print(validated.summary())
#  P6mm1' #183.186: 24 of 24 coset representatives of the lattice translations obeyed   (.indices, .verdicts)
#    the single-valued group's, each standing for itself and its negative in the double group
#    each one's covariance residual, at most 1.6e-14 against the tolerance 1e-08, on 36 k-points   (.residuals, .tol, .n_kpts)
print(validated.broken, validated.closed)
#  () True
print(fs.validated)
#  ValidatedGroup(P6/mmm1' #191.234, 48 of 48 coset representatives of the lattice translations obeyed, which form a group; they are the single-valued group's, each standing for itself and its negative in the double group; 48 act on every k as another does, so the Hamiltonian alone does not say which of each pair holds, and the representation decides)
print(fs)
#  RealizedGroup(P6/mmm1' #191.234; additional commuting matrices: 1; free-standing, the substrate group #183.186 agrees)
```

On the substrate every operation is obeyed and the operations that hold close
into a group; each one's space of solutions is two-dimensional, because the
conserved `s_z` multiplies every solution. Free-standing, every operation shares
its action on `k` with its product with the horizontal mirror, the two pose one
measurement, and each verdict is `undetermined`: whether each operation itself
holds is settled by its tensor structure, read before the representation is
built, and here every one of them holds.

| field | what it holds |
|---|---|
| `indices` | the operations kept, `obeyed` and `undetermined` together |
| `obeyed`, `undetermined`, `broken` | the operations settled one by one; the ones kept whose verdict their `k`-action class shares; the ones the Hamiltonian breaks |
| `verdicts` | each operation's verdict: `"obeyed"`, `"broken"` or `"undetermined"` |
| `operations` | the operations kept, by index, as an `OperationTable` |
| `residuals` | each operation's covariance residual |
| `discrepancies` | one entry per broken operation: its `ind`, what refuted it, `refuted_by` (`"no symmetry matrix"`, `"covariance"` or `"out of plane"`), and its `residual` |
| `closed` | whether the operations kept close into a group |
| `tol`, `n_kpts` | the tolerance each covariance residual is held to, in the units of `H`, and the number of `k`-points measured |
| `k_action_classes` | the operations grouped by their action on `k` and whether they are antiunitary: one measurement settles each class |
| `group_label` | the group tested, as `"P6/mmm1' #191.234"` |

## `SpaceGroupRep`

The model's representation of its group: `r.rep`. One matrix per
operation on the orbitals, in the tables' convention, spin along `+z` in a
right-handed frame; every band label is read through it. Its `summary()` reads
it part by part:

```python
print(rep.summary())
#  P6mm1' #183.186, on a substrate   (.group)
#    its operations, by index and Seitz symbol   (.operations)
#    each one's matrix on the 4 states, each orbital's spin up then down along z   (.matrices)
#    an operation with time reversal is its matrix times complex conjugation   (.antiunitary)
#    the orbitals, by Wyckoff position   (.characters)
#      2b (3m): orbitals 0-1, -E1
#    the multiplets each site's orbitals form   (.multiplets)
print(len(rep.matrices), len(rep.antiunitary), rep.antiunitary_operator[0], rep.matrices[2].shape)
#  24 12 13 (4, 4)
print(rep.multiplets.summary())
#  each site's orbitals, multiplet by multiplet   (.by_site)
#    site 0: orbital 0
#    site 1: orbital 1
#    read off the Hamiltonian   (.determined_by)
```

Twenty-four matrices, one per operation listed, twelve of them for operations
combined with time reversal, the pure time reversal being operation 13; each
site's orbital is a multiplet of its own.

| field | what it holds |
|---|---|
| `matrices` | each operation's `k`-independent matrix, keyed on its index: the representation induced from the site content the atomic-limit table returns; where the hoppings leave it open up to a change of basis they conserve, as on identical copies or on orbitals nothing couples, a multiplet's block is the operation on the lowest-angular-momentum functions of its content laid along its orbitals in order, `p_x, p_y` for `E` at a `4mm` site, wherever such a change of basis reaches it |
| `indices`, `operations` | the operations represented, by index, and the same as an `OperationTable` |
| `antiunitary` | which of them are antiunitary |
| `characters` | [the atomic-limit table](#the-atomic-limit-table) |
| `multiplets` | the multiplets the Hamiltonian binds each site's orbitals into, a `SiteMultiplets`: `by_site`, `dims`, `determined_by`, and `accidental`, orbitals degenerate without the symmetry forcing it |
| `spin` | the declared spinor basis of a spinless model carrying spin in its orbital index, a `SpinBasis`; `None` for every other model |
| `environment` | the environment a layer's representation was built in; `None` for a three-dimensional model |
| `free_standing_group` | for a layer read on a substrate, the free-standing layer's group as `realized_group` measured it beside this one and found it to agree: its label, symbol and size, which the indicator names beside a substrate's class where it is larger; `None` otherwise |
| `residuals`, `product_residual`, `factor_residual` | the covariance each matrix achieves; how far the matrices miss multiplying as the group does; the same for the Bloch operators, translations included |
| `model`, `group`, `assignment` | the model, the group and the `OrbitalAssignment` it was built on |

Its methods act with one operation: `k_action(ind)`,
`transform_k(ind, k)`, `in_little_group(ind, k)`, `little_group(k)`,
`bloch_operator(ind, k)`, `sewing_operator(ind, k)` and `covariance(k_pts)`;
`antiunitary_operator` is the pure time reversal's index and matrix, which
every corep label is anchored against; `orbits` and `site_orbitals` repeat
those of `assignment`.

## The atomic-limit table

`rep.characters`, an `AtomicLimitTable` with one `AtomicLimitRow` per orbital:
the table an atomic limit is written up in, and the place a mis-oriented
declaration shows. A row's own `summary()` reads it part by part.

```python
print(rep.characters.summary())
#  orbital  Wyckoff  site symmetry  function  irrep  named by  determined by  orbital characters
#        0       2b             3m         s    -E1    tables    declaration  E=+1, C3(z)=+1, m(x)=+1
#        1       2b             3m              -E1    tables    hamiltonian  E=+1, C3(z)=+1, m(x)=+1
print(rep.characters.multiplicity_spaces)
#  ()
row = rep.characters[0]
print(row.summary())
#  orbital 0, site 0, at 2b (3m): -E1, named by the tables   (.irrep, .named_by)
#    declared 's' (A1), read at the 3m in P6mm1' #183.186   (.declared_as, .declared_irrep, .declared_in)
#    orbital characters: E=+1, C3(z)=+1, m(x)=+1   (.generator_character)
print(row.dim, row.site_group)
#  2 (1, 2, 3, 7, 8, 9)
```

Each column is a field of the row:

- **`orbital`**, the model's orbital index.
- **`Wyckoff`** (`wyckoff`), the site's Wyckoff position in the tables'
  setting, multiplicity and letter.
- **`site symmetry`** (`site_symmetry`), the Hermann–Mauguin symbol of the
  site-symmetry group, read off the operations measured to fix the site.
- **`function`**, the orbital function a declaration wrote, spelled as
  `characters=` takes it so that a row declares back verbatim: a shorthand
  (`px`, `dx2-y2`) or a shell (`p`). It is blank where nothing declared a
  function: an undeclared orbital, or one declared by an irrep, a sum of irreps
  or its characters, since which real functions a set of Wannier functions is,
  inside its site, is a choice of basis the hoppings cannot make.
- **`spin`**, on a declared spinor basis only: which member of its pair the row
  is, `up` or `down`.
- **`irrep`**, the site irrep the states carry, in the site group of the group
  returned: for spinful states the double-valued site corep, `-E1` for an `s`
  orbital at a `3m` site. It is what the row's own multiplet carries, whatever
  the spelling of a declaration: a `p` shell declared whole on a site whose
  `p_z` the hoppings keep apart from its `(p_x, p_y)` pair reads `A1`, `E`,
  `E` at a `3m` site, and the shell's `A1 + E` is `declared_irrep`. Where one
  multiplet's orbitals hold different irreps it is their sum, `2x-E1 + -1E2E`,
  read off the characters and so the same in any basis; where it holds several
  copies of one irrep, that irrep on every row. Blank where nothing named it.
- **`named by`** (`named_by`), where the irrep's name comes from.
- **`determined by`** (`determined_by`), how the site irrep is known.
- **`orbital characters`** (`generator_character`), the orbital factor's
  character on one operation per conjugacy class, single-valued whatever the
  states carry.
  A spinor's character is zero on every mirror and twofold rotation, which are
  the operations a mis-oriented declaration is caught on, so the parity is read
  here.

`determined_by` says how the site irrep is known, and with it the band
representation the irrep induces. The hoppings fix the model's representation
only up to a one-dimensional character on each part of the crystal they couple
([what the hoppings fix](../background/measurement.md#what-the-hoppings-fix)),
and a **labelling** of the representation is one choice of those characters;
`r.assignment` is something else, where the orbitals sit.
`determined_by` takes four values:

| value | how the site irrep is known |
|---|---|
| `declaration` | `characters=` stated it, and the matrices returned carry it |
| `declaration not carried` | `characters=` stated it, and the Hamiltonian's own blocks do not carry it under the labelling returned; `irrep` is what they carry, and a line under the table names what was declared |
| `hamiltonian` | nothing was declared for this orbital, and every labelling the declarations leave open gives it this site irrep: an orbital the hoppings fix relative to a declared one, or an irrep such as the `E` of a `(p_x, p_y)` pair at a `3m` site, which no labelling changes |
| `lowest-l` | nothing was declared, and the labellings left open give it different site irreps; the one the lowest-ℓ rule picks is shown, and `characters=` states another ([the lowest-ℓ rule](../background/measurement.md#the-lowest-angular-momentum-rule)) |

`named_by` takes four:

| value | where the name comes from |
|---|---|
| `tables` | the tables list the band representation these states induce at this Wyckoff position, and the name is the site irrep those entries carry |
| `point group` | the tables' entries at this position do not name it, because they list none there or none whose band representation is this row's; the name is the one the tables give that irrep of the site point group, read at the origin of a symmorphic space group, so one irrep has one name at every site. One of the exceptions to a site irrep at a maximal Wyckoff position inducing an elementary band representation, the `E` at `3c` of `P-43m`, reads this way |
| `Mulliken` | those names follow an orientation no table fixes at this site, so the name is the row's Mulliken symbol: `B1` and `B2` of an `mm2` line whose twofold axis lies along `x` |
| `unnamed` | nothing named the row: `irrep` is blank, and the characters are the whole statement |

A sum of irreps is named as directly as the least direct of its terms. Under
the rows the table prints a line for each of four things, each but the last
ending with the attribute that holds it: a declaration that
lowered the group, that the Hamiltonian's blocks do not carry, or that is
written in a name the tables give a row Mulliken's convention calls by another
([site irreps](../background/conventions.md#site-irreps)), with what was
declared and where it was read, which row Mulliken's convention calls it, what
it cost, and what is returned in its place; an orbit whose band representation
is not its rows' own elementary band representations at its own position,
written out in elementary ones ([`BandRepresentation`](#bandrepresentation)); a
pseudospin the Hamiltonian conserves on an orbit, with the site content the
lowest-ℓ rule assigned its copies; and, where some row reads `lowest-l`, one
sentence saying what that means. The declared variant prints the first kind:

```python
print(declared.rep.characters.summary())
#  orbital  Wyckoff  site symmetry  function  irrep  named by  determined by  orbital characters
#        0       1c              3            -1E2E    tables    declaration  E=+1, C3(z)=+1, C3^-1(z)=+1
#        1       1b              3            -1E2E    tables    declaration  E=+1, C3(z)=+1, C3^-1(z)=+1
#    orbital 0: declared 'A1', read at the 3m in P6mm1' #183.186; the declared content lowered the cell before any hopping was read and resolves as A1 of the 3m in P3m11' #156.50; the declaration breaks m([010]), m(x), m([1-10]), m([010])', m(x)', m([1-10])'; returned as -1E2E of 3   (.declarations)
#    orbital 1: declared 'A2', read at the 3m in P6mm1' #183.186; the declared content lowered the cell before any hopping was read and resolves as A2 of the 3m in P3m11' #156.50; the declaration breaks m([010]), m(x), m([1-10]), m([010])', m(x)', m([1-10])'; returned as -1E2E of 3   (.declarations)
```

Each line says the word declared and the site group it was read in, the
reduction it made to the cell, the operations it broke, and the site irrep
returned in the group that holds: the site group `3` no longer tells `A1` from
`A2`, both restricting to its `A`, so both rows read `A` carried by the spinor,
`-1E2E`.

Every field of a row:

| field | what it holds |
|---|---|
| `orbital`, `site` | the orbital, and the site it sits on |
| `wyckoff`, `site_symmetry`, `function`, `irrep`, `named_by`, `determined_by` | the columns above |
| `multiplet` | the orbitals the Hamiltonian binds this one with, which carry its site irrep together |
| `dim` | the dimension of what `irrep` names: twice the orbital factor's where the states carry spin |
| `site_group` | the indices of the unitary operations that fix the site |
| `character`, `generator_character` | the orbital factor's character on every operation of the site group, and on one per conjugacy class by name |
| `declared_as` | exactly what was written for this orbital |
| `declared_irrep` | the irrep, or sum, that declaration resolved to, on every row it was written for |
| `declared_in` | the site group and the group the declaration was read in |
| `reduced_in` | where the declaration was resolved, where the cell was lowered under it |
| `declaration_broke` | the operations the declared content breaks |
| `note` | how a declaration was read, and, where the tables give a declared name to a row Mulliken's convention names otherwise, which row that is; which member of its pair a row on a spinor basis is; and what a declaration not carried stated |
| `residual` | how cleanly the orbital's block factorizes into its character and its spin factor; a large one says the irrep is not to be trusted |

And of the table: `orbitals`, the rows; `irreps`, one site irrep per orbital;
`summary()`; `band_representations`, one per orbit ([below](#bandrepresentation));
`multiplicity_spaces`, one `MultiplicitySpace` per orbit carrying a conserved
pseudospin, its `content` the site content assigned, `contents` every one the
group's table admits and `n_representations` how many representations of the
copies it admits, empty here; `declarations`, each declaration as it was read,
with `mulliken`, each declared name the tables give a row Mulliken's convention
names otherwise, mapped to Mulliken's name of that row, and `returned`, what
the multiplets it was read on carry together;
`checks`, the declared characters against the measured ones, operation by
operation, which a `declaration_breaks_group` refusal prints; `frame`, the name
of every operation by index; `spin`, a declared spinor basis; `residual`, the
worst row's; and `site_tables`, the site groups' character tables. A table can
be indexed by orbital, as `rep.characters[0]`. Labellings that give every
orbital these same site irreps label the whole set of bands alike, and where
they label a subset of it differently, as they do the occupied pair here, the
[`HSPTable`](#hsptable) of that subset says so.

## `BandRepresentation`

The band representation an orbit of orbitals induces, written in elementary band
representations: one per orbit, in `rep.characters.band_representations`, its
`summary()` the sum and the sums it is not distinguished from, a line each.

```python
br = rep.characters.band_representations[0]
print(br.summary())
#  orbitals 0-1 at 2b (3m) induce the elementary band representation of -E1 at 2b (3m)   (.terms)
print(br.orbitals, br.wyckoff, br.site_symmetry, br.terms, br.alternatives)
#  (0, 1) 2b 3m ((1, '-E1↑G(4)', "2b(3m1',3m)"),) ()
print(br.spelled())
#  1 x -E1↑G(4)@2b(3m1',3m)
```

The two orbitals at `2b` induce one elementary band representation, the `-E1`
of `3m` induced to four bands, at their own position, so the table prints no
line for it. `terms` are the multiplicity, the elementary band representation
and its position for each term of the sum, each elementary band representation
under the label the group's list gives it,
[the band-representation key](../background/conventions.md#the-band-representation-key);
`alternatives` the other sums with the same content and the same stable
real-space invariants, which nothing read here distinguishes from it, and none
where the group has no tabulated invariants; `note`, where no sum is returned,
what is said in its place: that none carries the content, that several do and
nothing read here says which is induced, or why the content could not be
identified. `spelled()` writes a sum on one line in the list's labels, and
`sentence()` writes it in words, the line the table prints for the orbit where
it prints one. It is the band representation and nothing finer: which site
irrep goes to which term is a subduction the tables do not carry.

## `CharacterTable` and `CorepContent`

`corep_content` reads the coreps a set of bands carries at one high-symmetry
point, from the characters of the set, which `characters` returns as a table;
the `summary()` of each reads it part by part:

```python
hsp = tqc.high_symmetry_points(group)
c = tqc.corep_content(rep, hsp["GM"], bands=2)
print(c)
#  CorepContent(GM: -GM9, partition=(0,))
print(c.summary())
#  bands 0-1 at GM (Γ): -GM9   (.table.bands, .kpname, .counts)
#    6 from these bands to the nearest other band, in the units of H   (.isolation)
print(c.labels, c.counts, c.partition, round(float(c.isolation), 3))
#  ({'-GM9': 1},) {'-GM9': 1} (0,) 6.0
print(c.table)
#  CharacterTable(k=[0.0, 0.0], bands 0-1 in 1 multiplet, group=realized)
print(c.table.summary())
#  bands 0-1 at k = (0, 0)   (.bands, .k)
#    the trace of each unitary operation of the little group, by index, over each degenerate multiplet   (.characters, .columns)
#      bands             0-1   (.multiplets)
#      energy             -3   (.energies)
#        1  E              2
#        2  C3(z)          1
#        3  C3^-1(z)       1
#        4  C2(z)          0
#        5  C6^-1(z)   1.732
#        6  C6(z)      1.732
#        7  m([010])       0
#        8  m(x)           0
#        9  m([1-10])      0
#       10  m([2-10])      0
#       11  m(y)           0
#       12  m([110])       0
#    the operations combined with time reversal are left out: a corep table holds no characters for them   (.antiunitary)
#    the sewing matrices unitary to 5.6e-16, the operators commuting with H(k) to 9.9e-16, the states solving H(k) to 0   (.unitarity, .commutators, .residual)
print(c.table.columns, c.table.multiplets, c.table.dims)
#  (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12) ((0, 1),) (2,)
```

At `GM` the two occupied bands are one Kramers doublet carrying `-GM9`, one
multiplet, its partition a single class, and `6.0`, in the units of `H`, from
the nearest band outside the set. The character table has one row, that
doublet, and a column for each unitary operation of the little group, by index;
its summary prints each column as a line, the operation's index, name and trace.

A `CorepContent` holds:

| field | what it holds |
|---|---|
| `kpname`, `k` | the tabulated point, and its coordinates in the model's reduced cell |
| `labels` | one reading per degenerate multiplet of the set of bands: each corep and its multiplicity |
| `counts` | the same summed over the set of bands: its symmetry vector at this point |
| `partition` | which multiplets carry the same characters, as numbers: two rows share a number exactly when their characters agree. It survives every labelling of the orbitals and every phase convention, which the labels do not |
| `isolation` | the gap from the set of bands to the nearest band outside it, in the units of `H` |
| `table` | the `CharacterTable` the labels were read from |
| `content` | the identification's own record: the multiplicities before rounding, and how far they sat from integers |
| `thresholds` | the zero thresholds in force at this `k`, which widen where multiplets crowd |

and a `CharacterTable`:

| field | what it holds |
|---|---|
| `k`, `bands` | the point, and the set of bands |
| `multiplets`, `dims`, `energies` | the rows: the degenerate multiplets of the set of bands, their degeneracies and their mean energies |
| `columns` | the indices of the unitary operations of the little group, in the tables' order |
| `characters` | the trace of each operation over each multiplet |
| `antiunitary` | the antiunitary operations of the little group, reported and never used: the tables carry no characters for them |
| `unitarity`, `commutators`, `residual`, `spread` | how far each sewing matrix is from unitary, how far each operator is from commuting with `H(k)`, how far the states are from eigenstates at this `k`, and the largest energy spread inside a multiplet |
| `thresholds`, `disagreements` | the zero thresholds at this `k`, and the numbers the two thresholds classify differently |
| `spectrum`, `degeneracy_tol` | the whole spectrum the table was built from, and the energy tolerance its multiplets were grouped at |
| `group_provenance` | the provenance of the group the columns belong to: `"realized"` for a group measured on `H`, `"descended"` for one measured after the cell was lowered |

## `HSPTable`

The symmetry vector of a set of bands, its coreps at every high-symmetry point,
each point read off one diagonalization: `hsp_table`.

```python
print(t)
#  HSPTable(bands=(0, 1), 3 points resolved, 0 refused, 0 skipped, group=realized)
print(t.summary())
#  bands 0-1   (.bands)
#    their coreps at each high-symmetry point, GM being Γ   (.content)
#      GM: -GM9
#      K: -K6
#      M: -M5
#    the labels of these 2 of the 4 bands rest on a one-dimensional character the hoppings leave open on the coupled part holding every orbital: another choice of it keeps every orbital's site irrep and the labels of all 4 bands, and labels these 2 differently; the lowest-angular-momentum rule chose the one whose orbitals read as the lowest-angular-momentum functions carrying their site content   (.note)
print(t.complete, t.counts, hsp.missing)
#  True {'-GM9': 1, '-K6': 1, '-M5': 1} ('A', 'H', 'L')
```

The two occupied bands carry `-GM9`, `-K6` and `-M5`, every point resolved. The
last line, the table's `note`, says the pair's labels rest on the
one-dimensional character returned; its partition does not. The points `A`,
`H` and `L` of the slab group lie out of the layer's plane, and a layer has no
`k` for them.

| field | what it holds |
|---|---|
| `bands` | the band indices of the set, ascending |
| `points` | every tabulated point in the model's periodic plane, with its coordinates |
| `content`, `partitions` | per point, the coreps and their multiplicities, and the partition |
| `counts` | the coreps summed over every point resolved: the symmetry vector, which `decompose` and `symmetry_indicator` take |
| `complete` | whether every point that constrains anything resolved; a table that is not complete is refused by the calls that need all of it |
| `records`, `refusals`, `skipped` | per point, the `CorepContent` read there, the `Refusal` returned there, and the points skipped for carrying no constraint, a little group of the identity alone |
| `thresholds` | per point, the zero thresholds in force |
| `note` | where the labels of these bands rest on the labelling returned, a one-dimensional character the hoppings leave open on a coupled part of the crystal: the orbitals that part holds, and what chose it; empty otherwise |
| `group_provenance` | the provenance of the group the labels are read in, as [`GroupData`](#groupdata) gives it: `"realized"` for a group measured on `H`, `"descended"` for one measured after the cell was lowered |

## `EBRDecomposition`

Whether a band set's symmetry vector is a sum, with nonnegative integer
coefficients, of elementary band representations, and every such sum:
`decompose`. It is the call to use: `pythtqc.groups.decompose_ebrs` beneath it
returns a plain mapping and restricts nothing to a layer's plane, so it refuses
a layer's content as incomplete. The occupied pair, then the two bands above
it:

```python
dec = tqc.decompose(t, group, rep=rep)
print(dec.summary())
#  symmetry-indicated decomposable: 1 sum of elementary band representations carries this content   (.verdict)
#    the coreps at the high-symmetry points: -GM9 + -K6 + -M5   (.content)
#    every sum, each elementary band representation written as the site irrep inducing it at its Wyckoff position (site symmetry)   (.terms)
#      -E1 at 1a (6mm)
#    A, H and L lie out of the layer's plane: every corep there occurs in every elementary band representation as often as a measured one, so the verdict is exact   (.restriction, .completeness)
#    necessary for an atomic limit, never sufficient: an obstruction beyond symmetry indicators can survive it   (.caveat)
conduction = tqc.decompose(tqc.hsp_table(rep, bands=[2, 3]), group, rep=rep)
print(conduction.summary())
#  no sum of elementary band representations carries this content, so no atomic limit has it   (.verdict)
#    the coreps at the high-symmetry points: -GM8 + -K4 + -K5 + -M5   (.content)
#    proved: no elementary band representation can occur more than once, and no sum with each at most once carries the content   (.certificate, .bound)
#    3 combinations of elementary band representations with some negative multiplicities reproduce the content (evidence of fragility)   (.signed_combinations)
#    A, H and L lie out of the layer's plane: every corep there occurs in every elementary band representation as often as a measured one, so the verdict is exact   (.restriction, .completeness)
print(conduction.signed_combinations["terms"][0])
#  ((-1, '-E1↑G(2)', "1a(6mm1',6mm)"), (1, '-E1↑G(4)', "2b(3m1',3m)"))
```

The occupied pair's one sum is the `-E1` site corep at Wyckoff `1a` induced to
two bands: the pair could be one elementary band representation at the hexagon
centre, where no orbital sits. The rows left out, at `A`, `H` and `L`, are 7 of
the 14. The pair above it is no such sum, and that is proved, the bound on the
coefficients being rigorous; each of the three combinations found with the sign
restriction lifted, the first of them printed last, is a difference of
elementary band representations: the arithmetic signature of a fragile band
set.

`verdict` is `"symmetry-indicated decomposable"` or `"none"`: a decomposition is
a statement about symmetry indicators, and any obstruction to an atomic limit
beyond them can survive it.

| field | what it holds |
|---|---|
| `verdict`, `decomposable` | the verdict, and whether it is the first |
| `solutions`, `unique` | every nonnegative-integer solution, as multiplicities over `ebrs`, the fewest elementary band representations in all first and, among those with as many, the one with more of those the tables list first; and whether there is exactly one |
| `terms`, `solution_strings()` | the same in the labels of the group's list: each solution's terms as (multiplicity, elementary band representation, position), and each solution on one line |
| `positions` | each solution's Wyckoff positions, the part a different labelling of the orbitals never moves |
| `ebrs`, `labels`, `y` | the elementary band representations a solution is written over, and the rows kept with the measured vector on them |
| `content`, `n_bands` | the symmetry vector decomposed, and the number of bands |
| `certificate` | what a `"none"` rests on: `"proof"`, or `"none within the bound"`; `None` for a decomposition, which an exhibited solution needs no certificate for |
| `bound`, `bound_is_rigorous` | the bound on every coefficient, and whether it provably excludes no solution |
| `enumeration` | `cap`, the most solutions listed, `n_solutions`, and `truncated`, whether the list was cut |
| `signed_combinations` | for a `"none"`, the same question with the sign restriction lifted, every coefficient between `−bound` and `bound`: its `solutions` and `terms`, `n_solutions`, zero where there is none within the bound, `truncated` where the list was cut at the cap, and a `note`. A combination found is evidence of fragility; the bound is not rigorous, since signed terms can cancel without limit |
| `restriction` | the rows a model can measure, a `Restriction`, and on a free-standing layer the positions outside it, whose band representations every solution leaves at zero |
| `site_content` | the site irrep of each orbital and how it is known, from `rep`; where one is `lowest-l`, a different declared content renames the site irreps in `terms` while `positions` stay |
| `completeness` | `claim`, `"exact"` or `"necessary-condition"`, and the points skipped or refused that it rests on |
| `caveat` | what a decomposition does not certify, a `WannierCaveat`, whose `text` says it in a sentence |

A `Restriction`, whose `summary()` names the rows left out and the measured row
each repeats, holds `applied`, whether any row was dropped, false for a
three-dimensional model; `points`, the points the model samples; `retained` and
`dropped`, the rows kept and dropped; `copies`, each dropped row with the kept
rows it duplicates; `exact`, whether every dropped row is such a copy, so that
the restricted verdict is the full one; `orphans`, the dropped rows that are
not; `rank` and `full_rank`, the two matrices' ranks, equal whenever `exact`;
and `require_exact`, whether a dropped row that is no copy would have been
refused. From `decompose` on a layer, the rows are compared on the layer's own
band representations, those of the Wyckoff positions within it:
`layer_extent` holds the heights its orbitals span along its normal, from its
middle plane, and `outside_layer` the positions of a free-standing layer's group
off the layer, or outside the slab its orbitals span, each with its height,
which carry none of its orbitals and are left out.

## `CompletenessReport` and `RegularCompletion`

The counting beside a decomposition: `completeness_check` reports how many bands
each solution accounts for, and `regular_completion` the bands that would carry a
band set up to whole copies of the regular band representation, `|U|` bands per
copy for `|U|` unitary operations, counted one per coset of the lattice
translations and, for a spinful model, the single-valued group's. The
`summary()` of each reads it part by part:

```python
cr = tqc.completeness_check(t, group, rep=rep)
print(cr)
#  CompletenessReport(complete=True, n_bands=2, band_counts=(2,))
print(cr.summary())
#  complete: 1 sum of elementary band representations carries this content   (.complete)
#    each sum is 2 bands, as many as were asked for   (.band_counts, .n_bands)
#    the sums, and what a sum does not certify   (.decomposition)
rc = tqc.regular_completion(t, group, bands=12, rep=rep)
print(rc.summary())
#  one copy of the regular band representation, the one the regular representation of a site group induces: 12 bands at every k   (.copies, .bands_per_copy)
#    the coreps the missing bands carry: 2×-GM7 + 2×-GM8 + -GM9 + 2×-K4 + 2×-K5 + 3×-K6 + 5×-M5   (.padding)
#    the orbitals they would be, each the site irrep at its Wyckoff position (site symmetry)   (.decomposition)
#      -E2 at 1a (6mm) + -E3 at 1a (6mm) + -E at 3c (mm2)
#      -E1 at 1a (6mm) + 2×-E2 at 1a (6mm) + 2×-E3 at 1a (6mm)
print(rc.decomposition.solution_strings()[0])
#  1 x -E2↑G(2)@1a(6mm1',6mm) + 1 x -E3↑G(2)@1a(6mm1',6mm) + 1 x -E↑G(6)@3c(mm21',mm2)
```

A `CompletenessReport` holds `complete`, whether the content is a sum of
elementary band representations, with the reach of `decompose`'s verdict;
`decomposition`, that `EBRDecomposition`; `n_bands`; `band_counts`, the bands
each solution accounts for; and `band_count_matches`, whether they equal
`n_bands`. A `RegularCompletion` holds `copies` and `bands_per_copy`; `target`,
the content of that many regular copies; `occupied`, the content handed in;
`padding`, the difference, which the bands still missing must carry;
`decomposition`, the padding's own, whose `solution_strings()` writes its sums
in the labels of the group's list, the first printed above; and `regular`, one
copy's content, computed from the group's characters.

## `SymmetryIndicator`

Which class a band set's content occupies in the group's band structures modulo
atomic limits, `X_BS = {BS}/{AI}`, read off the Smith normal form of the
group's EBR matrix
([how the class is computed](../background/tqc-primer.md#how-a-symmetry-indicator-is-computed)):
`symmetry_indicator`. `pythtqc.groups.indicator_group` gives a group's
elementary divisors and the tables' named forms, and no class. On the
substrate, and free-standing:

```python
si = tqc.symmetry_indicator(t, group, rep=rep)
print(si)
#  SymmetryIndicator(trivial indicator group, #183.186, trivial)
print(si.completion, si.free_standing_group)
#  None {'number_str': '191.234', 'name': "P6/mmm1'", 'size': 48}
sf = tqc.symmetry_indicator(tf, fs.group, rep=fs.rep)
print(sf)
#  SymmetryIndicator(indicated nontrivial, #191.234, Z6 x Z12, [1|6, 0|12], completed on the stack)
print(sf.summary())
#  indicated nontrivial, in P6/mmm1' #191.234: no atomic limit has this content, nor any with atomic bands added   (.verdict)
#    the indicator group, the band structures modulo the atomic limits: Z6 x Z12, a class being an integer mod 6 and one mod 12   (.structure)
#    canonical: z2w,3 = 1, z4 = 2, z3m,pi = 1, z3m,0 = 1, z6m,0 = 5, z6m,pi = 5, z12 = 10, the names of the paper that tabulates the stable real-space invariants, Hwang et al. (2026)   (.canonical)
#    indicator: 1 mod 6, 0 mod 12, of order 6: 6 copies of these bands are in the zero class   (.values, .order)
#    irreptables: z2w,3 = 1, z4 = 2, z3m,π = 1, z3m,0 = 1, z6m,0 = 5, z6m,π = 5, z12 = 10, the same indicators as the irreptables package names them   (.irreptables_named)
#    the coreps at the high-symmetry points: -A9 + -GM9 + -H8 + -K8 + -L6 + -M6   (.content)
#    A, H and L, at k_z = ½ out of the layer's plane: their coreps measured on the layer's decoupled stack, the layer repeated along its normal with no hopping between copies, in the layer's own group #191.234   (.completion)
#    the class of the layer's decoupled stack: every component is an invariant of that stack, fixed by the layer's own content   (.two_dimensional)
```

The printout gives the verdict, the group the class lives in, the indicator
group, and each component as `value|divisor`, with how the out-of-plane rows were
obtained. The indicator group of `P6mm1'` is trivial, with no class to measure,
and every row of its basis out of the plane repeats an in-plane one, so nothing
had to be measured on a stack; the free-standing layer's group, `#191.234`,
which `realized_group` measured beside it, is the free-standing question and is
named, not used. Free-standing, the summary reads, line by line:

- the verdict, in the group the class lives in, with what a nonzero class
  means: no band set with this content is an atomic limit, and adding atomic
  bands cannot change that (a zero class certifies nothing);
- the indicator group, `X_BS = ℤ6 × ℤ12`;
- the class in the canonical names, those of the stable-RSI paper, the paper of
  Hwang *et al.*, Nat. Commun. **17**, 8564 (2026), that tabulates the stable
  real-space invariants ([citing](../citing.md#stable-real-space-invariants));
- the class as its components over the elementary divisors of the Smith normal
  form, `1` modulo 6 and `0` modulo 12, and its order, which no change of basis
  moves;
- the same class in `irreptables`' names;
- the symmetry vector;
- the stack's measurement: the rows out of the plane, `A`, `H` and `L` at
  `k_z = ½`, measured on the layer's decoupled stack in the layer's own group,
  the characters the stack shares with the layer agreeing;
- how a layer's class reads: the class of its decoupled stack, every component
  an invariant of that stack, fixed by the layer's own content.

`verdict` takes three values: `"indicated nontrivial"`, `"indicator zero"`, and
`"trivial indicator group"`, a group with no elementary divisor above 1 and so no
class to measure, which is itself a verdict.

| field | what it holds |
|---|---|
| `verdict`, `nontrivial`, `indicated` | the verdict, and whether the class is nonzero, under both names |
| `structure`, `divisors` | the indicator group, `"Z6 x Z12"`, and its elementary divisors above 1 |
| `values`, `order` | each component of the class modulo its divisor, and the class's order, which is basis-independent |
| `canonical`, `canonical_tex`, `canonical_note` | the class in the stable-RSI paper's names, the same names typeset for a plot, and where they came from or why they are `None`: a unitary group, a trivial indicator group, or tables not installed |
| `irreptables_named`, `irreptables_note`, `irreptables_convention_touched`, `as_irreptables()` | the class in `irreptables`' names, re-keyed into this package's convention; where it came from; the forms the change of convention moved; and `canonical` respelled in those names |
| `number_str`, `group_name`, `magnetic_type` | the group the class lives in, the one handed in, and its magnetic type, `"I"` or `"II"` |
| `content`, `n_bands`, `site_content` | the symmetry vector, the bands, and the site irreps the labels rest on, with how each is known |
| `labels`, `y`, `y_prime`, `slots` | the indicator's arithmetic: every row of the group's EBR basis, the vector on it, the vector in the Smith basis, and where the divisors sit in it |
| `reconstructed` | rows a layer cannot measure, filled from a row they duplicate |
| `restriction` | the rows the model measures, a `Restriction` |
| `completion` | how the rows out of a layer's plane were measured on its decoupled stack, or `None` where nothing had to be: `route`, the decoupled stack; `outcome`, `"completed"` in the group handed in, or `"stopped"`; `stopped_at` and `stop_reason`; `input`, the layer's group; `stack`, the group the stack was read in, once its representation there exists; `environment`; `stack_period` and `stack_extent`, the stack's period and the layer's extent along its normal, whose difference is the vacuum between the copies; `gates`, each check's measured number; and a `note` |
| `two_dimensional` | how to read a layer's class, the class of its decoupled stack: `out_of_plane_points` and `out_of_plane_labels`, the rows at `k_z = ½`; `components`, whether each Smith component reads them; `irreptables_support`, where each named form's coefficients sit; and a `note` |
| `free_standing_group` | for a layer read on a substrate, the free-standing layer's group as `realized_group` measured it beside this one, where it is larger than the group the class lives in; `None` otherwise |
| `lattice_check`, `verification`, `convention` | the compatibility relations checked exactly; an independent computation of the Smith form, which agrees; and the translation convention, in a sentence |
| `group_provenance` | whether the group was measured on `H` |

## The stable real-space invariants

The stable real-space invariants (SRSIs) are linear forms `θ = Θ·p` in the
site-symmetry vector `p`, how many times each site irrep occurs at each Wyckoff
position, that no symmetric deformation of the Wannier functions changes: two
atomic configurations share them exactly when they deform into each other once
the same atomic orbitals are added to both. Some are `ℤ`-valued and some
`ℤ_n`-valued ([what they are](../guide/srsi.md#what-a-real-space-invariant-is)),
and the tables are the stable-RSI paper's.

### `SRSITables`

The tables of one group: `srsi_tables`, for a measured group or a space-group
number with `spinor=`.

```python
tables = tqc.srsi_tables(group)
print(tables.summary())
#  P6mm1' #183.186, double-valued: stable real-space invariants Z x Z x Z x Z x Z2   (.structure)
#    each invariant theta_i, Z- or Z_n-valued, and how many terms its formula has, each term counting the copies of one site irrep at one Wyckoff position   (.rows)
#      theta_1    Z  6 terms
#      theta_2    Z  6 terms
#      theta_3    Z  6 terms
#      theta_4    Z  2 terms
#      theta_5   Z2  1 term
#    9 site irreps at Wyckoff positions, 6 of them inducing an elementary band representation   (.site_keys, .ebrs)
#    9 terms on site irreps that induce none: at a non-maximal Wyckoff position, or at a maximal one whose band representation is a sum of others   (.terms_without_ebr_column)
#    no symmetry indicator, the group's indicator group being trivial   (.indicators)
#    1 splitting of an elementary band representation into others that the symmetry indicators allow   (.split_ebrs)
print(tables.theta({("-E1", "a"): 1}))
#  ((Fraction(0, 1), Fraction(0, 1), Fraction(1, 1), Fraction(0, 1)), {2: (0,)})
```

A term counts the copies at one site key, a site irrep at a Wyckoff letter,
written `(irrep, letter)`. `theta(p)` evaluates the invariants on a
configuration, here one `-E1` at `1a`, an `s` orbital with its spin at the
hexagon centre: the `ℤ`-valued ones as exact fractions, and the `ℤ_n`-valued
ones by modulus.

| field | what it holds |
|---|---|
| `number_str`, `sg`, `spinor` | the group, its space-group number, and whether these are the tables with spin–orbit coupling |
| `rows`, `names`, `moduli`, `structure` | the invariants `θ_i` in the paper's order, their names, each one's modulus (`None` for a `ℤ`-valued one), and the group they form, `"Z x Z x Z x Z x Z2"` |
| `z_rows`, `zn_rows` | which rows are `ℤ`-valued, and the `ℤ_n`-valued ones by modulus |
| `site_keys`, `maximal_keys` | every key the rows touch, maximal or not, each marked maximal or not and with its column where it has one; and the maximal keys with the bands each orbital there contributes, the basis the atomic realizations are enumerated over |
| `ebrs`, `theta_on_columns` | the elementary band representations as columns, and `Θ` restricted to them |
| `band_count_form` | the number of bands as a form in the `ℤ`-valued invariants, itself an invariant |
| `indicators`, `split_ebrs`, `terms_without_ebr_column` | the paper's indicators as forms in the invariants, its splittings of elementary band representations, and the terms on keys no column carries |
| `provenance` | where the converted tables came from, with the paper's DOI |

Its methods evaluate the invariants: `theta(p)` on a mapping over site keys,
`theta_columns(p)` on a sequence over the columns, `theta_unreduced` and
`theta_unreduced_columns` with the `ℤ_n` rows not reduced, `indicators_of` and
`indicators_of_columns` for the tabulated indicators, and `row(name)`.
`srsi_of` and `srsi_of_columns` are the same evaluations as calls.

### `ModelSRSIs`

The invariants read off the orbitals: `srsi_of_model`.

```python
print(tqc.srsi_of_model(r).summary())
#  P6mm1' #183.186, double-valued: stable real-space invariants Z x Z x Z x Z x Z2   (.structure)
#    the model's: Z-valued (1, 1, 1, -1), Z_n-valued mod 2 (0)   (.theta_Z, .theta_Zn)
#    each site orbit: its orbitals, its Wyckoff position, and its content, written m (irrep)_w, m copies of a site irrep at Wyckoff position w; then its invariants   (.orbits)
#      orbitals 0-1 at 2b: 1 (-E1)_b
#        (1, 1, 1, -1), mod 2 (0)
```

An orbit at a position the tables list no elementary band representation at is
moved along its own variety, the point, line or plane its Wyckoff position
covers, to the positions on the variety's closure that list one, and its line
names the one its content is written at, `at 6c, slid onto 3a` in
[the SRSI guide](../guide/srsi.md#from-the-orbitals). A model is an atomic
configuration, so its total is integral, and it carries the `ℤ_n`-valued
invariants as well.

`ModelSRSIs` holds `orbits`, `theta_Z`, `theta_Zn`, `names`, `moduli`,
`structure`, `number_str`, `spinor` and the `tables`. Each orbit, an
`OrbitSRSIs`, holds its `orbitals` and `sites`; its `letter`, `multiplicity` and
`n_free` in the tables' setting; `closure`, the letters an orbital confined to
its variety can be moved to; `content`, the coreps its states carry, measured;
`points`, where; `localizations`, every decomposition of that content over the
columns of a letter of `closure`, each with its terms and invariants; `chosen`,
the one returned, and `agree`, that every localization gave the same invariants;
`same_content`, columns the content cannot tell apart; `composite`, the site
irreps of an orbit whose band representation no column carries; and its own
`theta_Z` and `theta_Zn`.

### `BandSRSIs`

The `ℤ`-valued invariants of a band subset, read off its symmetry vector:
`srsi_of_bands`. The occupied pair of the free-standing layer:

```python
print(tqc.srsi_of_bands(tf, fs.group, rep=fs.rep).summary())
#  P6/mmm1' #191.234, double-valued: theta_Z, the Z-valued stable real-space invariants of 2 bands   (.theta_Z)
#      theta_1 =     -1/6
#      theta_2 =      1/6
#      theta_3 =      1/2
#      theta_4 =      1/3
#      theta_5 =        0
#      theta_6 =      2/3
#      theta_7 =        0
#      theta_8 =        0
#      theta_9 =        0
#     theta_10 =     -1/3
#     theta_11 =     -2/3
#     theta_12 =        0
#     theta_13 =      1/2
#     theta_14 =        0
#    fractional, with common denominator 6: the bands are stable topological   (.stable, .denominator)
#    their symmetry-indicator class: (1, 0) in Z6 x Z12, of order 6   (.indicator)
```

Fractional, so no configuration of orbitals has this pair's symmetry data; the
common denominator, 6, is the order of its indicator class. The `ℤ_n`-valued
invariants are not functions of a symmetry vector, so this record carries none.

`BandSRSIs` holds `theta_Z` and `names`; `stable`, whether any is fractional;
`denominator`; `indicator`, the `SymmetryIndicator` measured with it; `p`, the
rational site-symmetry vector that was evaluated, one of many; `n_bands`,
`number_str` and `spinor`.

### `Splitting` and `Realizations`

Whether the bands below a gap and the bands above it can both be atomic
insulators: `splitting`, and `nsit_criterion` on invariants already in hand. The
occupied pair against the pair above it, on the substrate:

```python
split = tqc.splitting(r, bands=2)
print(split.summary())
#  P6mm1' #183.186, double-valued, diagnosed topological: the two sides of the gap cannot both be atomic insulators   (.verdict)
#    V, the 2 bands asked for; C, the model's other 2; H, all 4   (.n_bands_v, .n_bands_c)
#    theta_Z, the Z-valued stable real-space invariants: H (1, 1, 1, -1) = V (0, 0, 1, 0) + C (1, 1, 0, -1)   (.theta_Z_total, .theta_Z_v, .theta_Z_c)
#    theta_Zn, the Z_n-valued ones of H: mod 2 (0)   (.theta_Zn_total)
#    V's symmetry-indicator group is trivial   (.indicator)
#    V's atomic realizations, every set of orbitals at maximal Wyckoff positions with V's symmetry data, written m (irrep)_w, m copies of a site irrep at Wyckoff position w, with its own theta_Zn   (.realizations_v)
#      [0] 1 (-E1)_a   mod 2 (0)
#    C has no atomic realization: theta_Z is integral, so no symmetry indicator sees these bands, and no set of orbitals at maximal Wyckoff positions has their symmetry data: a fragile band set   (.realizations_c, .c_atomic_possible)
```

The verdict comes first and its reason last: the pair above the gap, `C`, has
integral `ℤ`-valued invariants and no atomic realization at all, a fragile side.
The invariants of the whole Hilbert space `H` are read off the orbitals, those
of the valence side `V` off its bands, and `C`'s are their difference; where
both sides have atomic realizations, a line names the pairs whose `ℤ_n`-valued
invariants add up to the total's, or says that no pair does.

`verdict` takes four values:

| value | what it says |
|---|---|
| `"diagnosed topological"` | no pair of realizations of the two sides adds up to the total's `ℤ_n`-valued invariants, or one side has no realization at all: the valence set, the conduction set or both is not an atomic insulator. The paper's name is NSIT, non-symmetry-indicated topological |
| `"invisible to the SRSIs"` | some pair adds up: any topology of the splitting is invisible to the stable real-space invariants |
| `"no Z_n SRSIs"` | the group has no `ℤ_n`-valued invariant, so there is no additivity to test |
| `"stable topological"` | a side's `ℤ`-valued invariants are fractional, so it is not an atomic insulator and the criterion is not needed |

`Splitting` also holds `nsit`, whether the verdict is the first; `reason`;
`theta_Z_total` and `theta_Zn_total`, the total's invariants; `theta_Z_v` and
`theta_Z_c`; `realizations_v` and `realizations_c`, the enumerations, `None`
where none was run; `consistent_pairs`, as indices into them;
`v_atomic_possible` and `c_atomic_possible`, whether each side has any
realization, which separates a fragile side from two atomic sides that cannot
add up; `moduli`; `indicator`, the valence side's class; `n_bands_v` and
`n_bands_c`; `same_content`, site irreps the invariants do not distinguish that
a realization uses; and `number_str` and `spinor`.

A `Realizations`, also returned by `atomic_realizations`, is read part by part
by its `summary()`, here the valence side's:

```python
print(split.realizations_v.summary())
#  P6mm1' #183.186, double-valued: every atomic realization, 1 of them   (.realizations)
#    of 2 bands with theta_Z = (0, 0, 1, 0), their Z-valued stable real-space invariants   (.n_bands, .theta_Z)
#    each a set of orbitals at maximal Wyckoff positions with that symmetry data, written m (irrep)_w, m copies of a site irrep at Wyckoff position w, with its own Z_n-valued invariants   (.label, .theta_Zn of each)
#      1 (-E1)_a   mod 2 (0)
```

It holds `theta_Z`, the invariants reproduced; `n_bands`; `keys`, the maximal
site keys it is written over; `realizations`, every one, each an
`AtomicRealization` with its multiplicities `p`, its `terms` and `label`, its
own `theta_Zn`, its `n_bands`, and the `pairs` it touches; `atomic_possible`,
whether there is any; and `bound`, `cap` and `note`. The enumeration is
complete, so an empty one is a proof: integral invariants and no realization is
a fragile band set.

### `StablePair`

Two site irreps at one position that no measurement in this package tells
apart: identical columns of elementary band representations, so nothing in
momentum space, and identical invariants, so nothing in real space.
`stably_equivalent_pairs` returns them for a group:

```python
pair, = tqc.stably_equivalent_pairs(tqc.srsi_tables(211, spinor=False))
print(pair.summary())
#  I4321' #211.57, single-valued: B2 and B3 at Wyckoff position d, which no stable real-space invariant distinguishes   (.irreps, .letter)
#    their elementary band representations, columns 15 and 16 of the group's list, have the same content at every high-symmetry point   (.names, .columns, .symmetry_vector)
#    the invariants both carry: Z-valued (1, 0, 1, 1, 1, -1, 0), Z_n-valued mod 2 (0, 0, 1)   (.theta_Z, .theta_Zn)
```

It holds `letter`, `irreps` and `columns`; `names`, the two elementary band
representations under the labels of the group's list; `symmetry_vector`, the
one both members induce; `theta_Z` and `theta_Zn`, which both carry; and
`number_str` and `spinor`. The two are not distinguished, which is not the same
as being one thing.

## `Refusal`

What a call returns in place of its record when it cannot complete: a falsy
record carrying the check that stopped it and what was measured there. The
indicator of the free-standing layer's pair, asked without its representation,
is one:

```python
refused = tqc.symmetry_indicator(tf, fs.group)
print(refused)
#  Refusal[indicator_not_evaluable]: the EBR rows ['-A10', '-A11', '-A12', '-A7', '-A8', '-A9', '-H8', '-H9', '-L5', '-L6'] of #191.234 sit at k-points this content does not cover, out of a 2D model's periodic plane or carrying no information, and they duplicate no row it does cover, so the symmetry vector cannot be completed and the indicator has no value here. The rows can be measured on the decoupled stack of the layer, which needs the representation, with its model and declared characters, and the bands the content counts: pass an HSPTable and rep=, and do not override points=. Completing missing content from the stable real-space invariants is future work.  (measured 10, threshold 0)
#      context: completion, number_str, orphans, points, retained, stack
print(bool(refused), refused.gate, refused.value, refused.threshold, refused.context["number_str"])
#  False indicator_not_evaluable 10.0 0.0 191.234
try:
    refused.raise_()
except tqc.RefusalError as error:
    print(type(error).__name__, error.refusal.gate)
#  RefusalError indicator_not_evaluable
```

The printout is `Refusal[gate]:`, the diagnosis in words, the number measured
against the one the check demanded, and the names of the keys its `context`
carries. Here ten rows of `P6/mmm1'`'s basis lie out of the layer's plane and
duplicate none in it, and without the representation there is no decoupled
stack to measure them on; the indicator above, handed `rep=`, measures them
there.

A refusal is one of four things: an honest verdict about the physics, as this
one is; a practical limit, a search or a memory budget that the same call runs
past with a larger number; a grammar error in a declaration or a question; or
one of the package's checks on itself, which means the case is to be reported
([the refusal reference](refusals.md#the-packages-checks-on-itself) lists them).
Every refusal reached once the group is identified names it.

| field | what it holds |
|---|---|
| `gate` | the name of the check that stopped the call, stable and listed under each call's refusals |
| `reason` | the diagnosis, naming the orbital, operation, point or group concerned |
| `value`, `threshold` | the number measured and the number the check demanded, `None` for a check with no number |
| `context` | everything needed to act on the refusal; once the group is identified, `context["group"]` or `context["number_str"]` names it |
| `raise_()` | the refusal raised as a `RefusalError`, which carries it on `.refusal`, where a script wants an exception |

`isinstance(out, tqc.Refusal)` is the test: it separates a refusal from every
value a call returns. A refusal is falsy too, and so are two verdicts, which a
truth test would misread: `stably_equivalent_pairs`, which returns its pairs
together and on most groups has none, and `subtract` of two equal contents,
which leaves none. Every record is truthy, a `"none"` decomposition, a zero
indicator class and a `"diagnosed topological"` splitting included. Two things
raise instead: a malformed argument raises
`ValueError` before anything is measured, and the package disagreeing with
itself raises `InternalError`, a defect to report and nothing in the model to
change. `MagneticScopeWarning` is the one warning issued, when time reversal is
measured broken; `warnings.simplefilter("ignore", tqc.MagneticScopeWarning)`
silences it. What each check means and what to do about it is in
[the refusal reference](refusals.md), and how to handle one in
[when a call refuses](../guide/refusals.md).

## The other records

Each is returned by the call named, and `help(tqc.<name>)` prints every field;
every one but `SiteReference` has a `summary()` that reads it part by part, each
line ending with the attribute that holds that part, as the records above do.

| record | returned by | what it is |
|---|---|---|
| `SewingMatrix` | `sewing_matrix` | the matrix of one operation in the basis of a set of bands at an invariant `k`, `B(g,k) = V† U(g;k) V`: its `matrix`, `trace`, `eigenvalues` and `angles`, and its checks |
| `Frontier` | `frontier_subspace` | every eigenvalue within `modulus_tol` of the smallest modulus of a spectrum, collected as one set of states, and whether it sits on zero (`at_zero`) |
| `KGrid` | `k_grid` | a mesh, as pythtb's own `Mesh` (`mesh`), with which high-symmetry points it samples (`contains`, `missing`) and the coordinates of every tabulated point (`hsp`) |
| `LittleGroup` | `little_group` | the operations that leave a `k` invariant up to a reciprocal lattice vector, unitary and antiunitary, and for each unitary one that vector, `G` |
| `HighSymmetryPoints` | `high_symmetry_points` | each tabulated point's coordinates in the model's own reduced cell, a mapping by name, with `missing`, the points out of a layer's plane |
| `SiteReference` | `wyckoff_sites_tables` | one site's Wyckoff position in the tables' setting: its `letter`, `multiplicity`, `reference` coordinates, `varieties`, and `closure`, the letters an orbital confined to its variety can be moved to |
| `WyckoffPosition` | `groups.wyckoff_positions_spglib` | one Wyckoff position in spglib's standardization, whose letter is not a table key |
| `IrrepContent` | `groups.identify_from_characters`, and `CorepContent.content` | the identification of measured characters: each row's multiplicities before rounding, and how far they sat from integers |
| `SpinBasis` | `spin_basis`, `spinless_presentation` | which orbitals of a spinless model are the up and down partners of one spinor Wannier function: `kind` and `pairs` |
| `Symmetrization` | `symmetrize`, and `RealizedGroup.symmetrization` | how far a near-symmetric model was moved onto its symmetric part ([above](#every-field)) |
| `MultiplicitySpace`, `SiteMultiplets` | the atomic-limit table, `rep.multiplets` | a conserved pseudospin and the site content assigned its copies; the multiplets each site's orbitals are bound into |
| `Thresholds`, as `pythtqc.sewing.Thresholds` | `CorepContent.thresholds`, `HSPTable.thresholds` | the two zero thresholds at one `k`, flat for quantities built from `H(k)` alone and conditioned for those read off eigenvectors |

The records inside the SRSI ones are reached as `pythtqc.srsi.<name>`: `SRSIRow`,
one invariant with its modulus and terms; `SRSIIndicator`, one tabulated
indicator as a form in the invariants; `SiteKey` and `MaximalKey`, one site key,
with the bands one orbital there contributes for a maximal one; `SplitEBR`, one
splitting of an elementary band representation the paper lists;
`AtomicRealization`; and `OrbitSRSIs`. `SRSIRow`, `SRSIIndicator` and
`SplitEBR` have a `summary()` too.
