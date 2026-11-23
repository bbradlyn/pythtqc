# What does a refusal mean, and what do I do about it?

A call that cannot complete returns a `Refusal` in place of its record. A
refusal names the check that stopped the call and the number that check
measured. Its reason says what was found, and what to do where there is a step
to take. [Refusals](../reference/refusals.md) gives, for every check, what it
means, its usual cause and what to do.

## The call

A call returns a refusal and never raises it. A refusal is falsy.
`isinstance(x, tqc.Refusal)` tells it apart from every other value a call
returns. Printed, it shows the check, the reason, the number measured, the
threshold it is compared with, and the keys of its `context`.

The example is Kane-Mele, `kane_mele(delta, t, soc, rashba)` from
`pythtb.models`, with a staggered potential, spin-orbit coupling and a Rashba
term. `coreps_at` is asked for the corep of its lowest band alone at Γ. A corep
is the little-group representation a set of bands carries where time reversal
is kept. `high_symmetry_points(r.group)["GM"]` is Γ, `GM` being the name the
irreptables package's tables use for it. The call refuses:

```python
import pythtqc as tqc
from pythtb.models import kane_mele

km = kane_mele(1.0, -1.0, 0.3, 0.25)                  # delta, t, soc, rashba
r = tqc.realized_group(km, typat=[0, 1], characters={0: "A1"})
gamma = tqc.high_symmetry_points(r.group)["GM"]
out = tqc.coreps_at(r, gamma, bands=1)                # the lowest band alone
print(isinstance(out, tqc.Refusal), bool(out))
#  True False
print(out)
#  Refusal[multiplet_split]: the bands [0] cut the degenerate multiplet [0, 1] at k = [0.0, 0.0] (energies [-3.162277660168, -3.162277660168], spread 0, degeneracy_tol 2.96873e-08). Ask for every band of the multiplet, or end the bands at a gap.  (measured 0, threshold 2.96873e-08)
#      context: bands, energies, k, multiplet
```

## How to read it

- `True False`: `isinstance(out, tqc.Refusal)` and `bool(out)`. The `isinstance`
  test is the one to branch on, since some returned values are falsy too.
  `subtract` of a table from itself returns an empty mapping of multiplicities,
  and the `stably_equivalent` of an `SRSITables` is empty for most groups.
  Every returned record other than a refusal is truthy, including a
  decomposition that finds no sum.
- `Refusal[multiplet_split]`: `out.gate`, the name of the check that stopped
  the call. The name is stable, and it heads the check's entry in
  [Refusals](../reference/refusals.md#multiplet_split).
- the text after the colon: `out.reason`, what was found, with the bands, the
  k-point and the energies concerned, and what to do.
- `measured 0`: `out.value`, the number the check measured. Here it is the
  energy spread of the degenerate multiplet the lowest band belongs to, in the
  units of the Hamiltonian `H`. A check with no number to measure carries
  `None` in `out.value` and `out.threshold`, and the printout shows neither.
  The check on a word in `characters=` that names nothing is one.
- `threshold 2.96873e-08`: `out.threshold`, the threshold the measured value is
  compared with. Here it is `coreps_at`'s `degeneracy_tol`, an energy in the
  units of `H`. Two energies closer than it belong to one multiplet. Left out,
  it is computed from `max|H|`, the largest entry of `H(k)` once the mean
  on-site energy is subtracted.
- `context: bands, energies, k, multiplet`: the keys of `out.context`, which
  holds what acting on the refusal needs without running the call again.

The fields can be read directly. Asked for the two lowest bands, the whole
multiplet, `coreps_at` returns their corep:

```python
print(out.gate, out.value, out.threshold, out.context["multiplet"])
#  multiplet_split 0.0 2.9687334080044094e-08 (0, 1)
print(tqc.coreps_at(r, gamma, bands=2).coreps)
#  {'-GM6': 1}
```

- `(0, 1)`: `out.context["multiplet"]`, the bands degenerate with the lowest
  band at Γ.
- `{'-GM6': 1}`: the corep multiplicities of the two lowest bands at Γ.

The refusal is a statement about the bands. Kane-Mele is spinful and keeps time
reversal, and Γ is a time-reversal-invariant momentum, so every level there is
a Kramers pair. The lowest band alone is half of one and carries no corep. The
pair carries `-GM6`
([Which little-group representations do these bands carry?](labels.md)).

## The group named in it

A refusal that `realized_group` returns once the group is identified opens by
naming the group, by its symbol and its number, and carries it in
`context["group"]`. The example declares the two-dimensional single-valued site
irrep `E` on orbital 0 of Kane-Mele, with `characters={0: "E"}`. `E` needs two
orbitals at its site, and orbital 0 is the only one there:

```python
bad = tqc.realized_group(km, typat=[0, 1], characters={0: "E"})
print(bad)
#  Refusal[ambiguous_multiplet]: the group identified is P3m11' #156.50, every operation of it obeyed; orbital 0: 'E' is 2-dimensional, and 1 orbital of site 0 declared with it is left over. Site 0 holds 1 orbital (0,), too few for a multiplet of 2 orbitals. Declare a word for one orbital, such as A1, A2, -E1.
#      context: declared_irrep, dim, group, multiplet, multiplets, orbitals, site
print(bad.context["group"].name, bad.context["group"].number_str)
#  P3m11' 156.50
```

- `the group identified is P3m11' #156.50, every operation of it obeyed`: the
  group the Hamiltonian obeys. The rest of the reason says what is wrong with
  the declaration, and lists the site irreps of one orbital that fit.
- `P3m11' 156.50`: `bad.context["group"]`, by its `name` and its `number_str`.

A refusal reached while the group is still being measured names the group the
cell allows, which contains the group the Hamiltonian obeys. For a model
periodic in two or three directions, the question which group its Hamiltonian
obeys is refused in two cases only, both about how the model is written
([When the group question is refused](../background/measurement.md#when-the-group-question-is-refused)).

## Four kinds

Every refusal is one of four kinds, and the kind says where the fix lies.
[Refusals](../reference/refusals.md) groups its entries by them.

1. **A verdict about the physics.** What was asked does not exist for these
   bands, or for this model as written. A set of bands that cuts a degenerate
   multiplet is one, as above. A symmetry data vector is the multiplicities of
   the little-group representations at every high-symmetry point, taken
   together. One that fails the group's compatibility relations is another
   verdict, since no set of bands separated by a gap from the others at every
   `k` has it. A model written on a supercell of its own primitive cell is a
   third. The fix is in the question or in the model.
2. **A practical limit.** A search, an enumeration, a measurement or a
   projection would need more than its ceiling allows, or the installation
   lacks the tables a question needs. Nothing in the model is at fault. Where
   the ceiling is an argument of the call, the reason says which argument to
   raise. Where something is missing from the installation, the reason names
   it.
3. **A grammar error in a declaration or a question.** `characters=` or `spin=`
   declares something that the orbitals it is written on cannot hold: a word
   that is not a site irrep of its site-symmetry group, a site irrep wider than
   one orbital written on orbitals that make no whole number of copies of it,
   as above, or a declared spin pair whose two orbitals sit at different
   centers. For a word in `characters=`, the reason lists the site irreps that
   would fit. A call handed an argument it cannot read refuses the same way: a
   k-point off a layer's plane, a Seitz symbol the group does not hold
   ([Operations: a Seitz symbol and a Cartesian name](../background/conventions.md#operations-a-seitz-symbol-and-a-cartesian-name)),
   or a negative multiplicity.
4. **The package's checks on itself.** A few checks measure an identity that
   holds in exact arithmetic, or compare two measurements of one thing. They
   check the representation of the group on the orbitals, the projection of
   `H` onto its symmetric part, and the setting in which Wyckoff positions are
   read against the tables. One that fails means the package has met a case it
   does not handle. Nothing in the model is to change, and the refusal is worth
   reporting with the model and the call
   ([The package's checks on itself](../reference/refusals.md#the-packages-checks-on-itself)).
   `tol` is the tolerance of the covariance check, on how far a matrix for an
   operation misses carrying `H(k)` onto `H(gk)`, in the units of `H`. The
   exception is `rep_not_covariant` reached close to `tol`. There each
   operation alone is obeyed within `tol`, and only the representation built
   for the group carries `H(k)` onto `H(gk)` beyond it. Its reason asks for
   `precision=` ([Refusals](../reference/refusals.md#rep_not_covariant)).

The example of a practical limit is a copy of Kane-Mele with every hopping off
by a random relative amount of order `1e-7`, read with `precision=1e-7`. The
declared precision calls for the projection of `H` onto its symmetric part,
and `max_bytes=1e4` sets the memory budget small on purpose:

```python
import numpy as np
from pythtb import TBModel
rng = np.random.default_rng(7)
noisy = TBModel(km.lattice, spinful=True)
noisy.set_onsite(km.onsite, mode="set")
for h in km.hoppings:
    noisy.set_hop(h["amplitude"] * (1 + 1e-7 * rng.normal()), h["from_orbital"],
                  h["to_orbital"], h.get("lattice_vector", [0, 0]), mode="add")
limit = tqc.realized_group(noisy, typat=[0, 1], characters={0: "A1"}, precision=1e-7,
                           max_bytes=1e4)
print(limit.gate, limit.context["group"].name, f"{limit.value:.3g} against {limit.threshold:.3g}")
#  projection_too_large P3m11' 3.01e+04 against 1e+04
print(repr(tqc.realized_group(noisy, typat=[0, 1], characters={0: "A1"}, precision=1e-7,
                              max_bytes=1e5)))
#  RealizedGroup(P3m11' #156.50; precision 1e-07 declared -> tol 2.97e-06 (cut 1e-06, window 0.01), position_tol 1e-07 (measured max|H| 2.969); symmetrized (defect 1.09e-07); on a substrate (the default), the free-standing group P3m11' #156.50 agrees)
```

- `projection_too_large`: `limit.gate`, the check that stopped the call.
- `P3m11'`: the group identified, `limit.context["group"]`, by its name.
- `3.01e+04`: `limit.value`, the bytes the projection needs at its peak.
- `1e+04`: `limit.threshold`, the bytes `max_bytes=` allows.
- `RealizedGroup(P3m11' #156.50; ...; symmetrized (defect 1.09e-07); ...)`: the
  same call with `max_bytes=1e5` returns the group, with `H` projected onto its
  symmetric part. [How do I measure a numerical or Wannier90 model?](numerical.md)
  explains the rest of the line.

## A refusal inside a record

A call that reads several high-symmetry points keeps each point's refusal
inside the record it returns. The example asks `hsp_table` for the corep of
Kane-Mele's lowest band alone at every high-symmetry point. At Γ and `M`, both
time-reversal invariant, the band is half of a Kramers pair. The table returned
holds those two refusals, and `decompose` refuses the table:

```python
table = tqc.hsp_table(r, bands=1)
print(bool(table), table.complete, {point: why.gate for point, why in table.refusals.items()})
#  True False {'GM': 'multiplet_split', 'M': 'multiplet_split'}
missing = tqc.decompose(table)
print(missing.gate, missing.context["refusals"])
#  incomplete_content {'GM': 'multiplet_split', 'M': 'multiplet_split'}
```

- `True`: `bool(table)`. The table is a record, and truthy.
- `False`: `table.complete`, whether every high-symmetry point was labeled.
- `{'GM': 'multiplet_split', 'M': 'multiplet_split'}`: `table.refusals`, the
  refusal met at each point, here read by its `gate`.
- `incomplete_content {...}`: by default, `decompose` refuses a table that
  holds a refusal at some point. `missing.context["refusals"]` names the check
  that refused at each.

With a point missing, the multiplicities in hand are not the bands' symmetry
data vector, so `decompose` refuses. `symmetry_indicator` refuses an
incomplete table the same way. With `allow_incomplete=True`, `decompose`
leaves out the points where a refusal was met, and does not refuse.

## Turning one into an exception

`out.raise_()` raises a refusal as a `RefusalError`, a `RuntimeError` that
carries the refusal on `error.refusal`. A script that stops at the first
refusal calls it. The Kane-Mele refusal above, raised:

```python
try:
    out.raise_()
except tqc.RefusalError as error:
    print(type(error).__name__, isinstance(error, RuntimeError), error.refusal is out)
#  RefusalError True True
```

- `RefusalError True True`: the exception's type, that it is a `RuntimeError`,
  and that `error.refusal` is the refusal raised.

## What is not a refusal

- **A malformed argument** raises `ValueError` before anything about the
  physics is measured. A keyword a call does not take raises `TypeError`, as in
  any Python call. Kane-Mele read with `precision=0`, which is not a positive
  number:

  ```python
  try:
      tqc.realized_group(km, precision=0)
  except ValueError as error:
      print(str(error).split(".")[0])
  #  realized_group(): precision=0 is not a positive number
  ```

- **The package disagreeing with itself** raises `InternalError`. One case is
  a layer measured on a substrate and free-standing, where the substrate's
  group is not the subgroup of the free-standing group that preserves the
  layer's normal. Another is a representation of the identified group that
  does not carry `H(k)` onto `H(gk)` within `tol`, though every operation of
  the group was measured to hold within it. Close to `tol` that case is the
  refusal `rep_not_covariant` instead, as kind 4 above says. With neither
  `precision=` nor `position_tol=` declared, it is the refusal
  `positions_not_exactly_symmetric` where the orbital centers sit far enough
  from the positions the group puts them at to account for the miss. The
  package also builds other forms of
  the model that equal it up to a relabeling of orbitals and a gauge: the slab
  and the decoupled stack of a layer, the spinless form of a spinful model, and
  the model with its centers moved. Each is compared with the model at eight
  random `k`, and a difference raises `InternalError`.
  An `InternalError` is a defect in the package and says nothing about the
  model: report it, with the model and the call.
- **`MagneticScopeWarning`** is a warning. `realized_group` issues it where time
  reversal is measured broken, and the group returned is then the unitary
  subgroup of the Hamiltonian's magnetic group
  ([Time reversal is measured](group.md#time-reversal-is-measured)).
  `warnings.simplefilter("ignore", tqc.MagneticScopeWarning)` silences it.

## Where to go next

- [Refusals](../reference/refusals.md) gives, for every check, what it means,
  its usual cause and what to do.
- [Functions](../reference/functions.md) lists, under each call, the refusals it
  can return.
- [Returned records](../reference/records.md) lists the fields of
  [`Refusal`](../reference/records.md#refusal).
- [How pythTQC reads a Hamiltonian](../background/measurement.md#when-the-group-question-is-refused)
  gives the two cases in which the question which group a Hamiltonian obeys is
  refused.
