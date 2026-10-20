# What does a refusal mean, and what do I do about it?

A call that cannot complete returns a `Refusal` in place of its record: a falsy record that
names the check that stopped it, the number that check measured, and what to do. This
chapter reads one, sorts refusals into their four kinds, follows one through the calls
after it, and says what raises instead. The model is pythtb's Kane–Mele with a sublattice
offset and a Rashba term. [The refusals reference](../reference/refusals.md) lists every
refusal a call can return, each with what it means, its usual cause and what to do.

## The call

Asked for the [corep](../reference/glossary.md#words-the-pages-use) content of Kane–Mele's
lowest band alone at Γ, `corep_content` refuses:

```python
import pythtqc as tqc
from pythtb.models import kane_mele

km = kane_mele(1.0, -1.0, 0.3, 0.25)                  # delta, t, soc, rashba
r = tqc.realized_group(km, typat=[0, 1], characters={0: "A1"})
gamma = tqc.high_symmetry_points(r.group)["GM"]
out = tqc.corep_content(r.rep, gamma, bands=1)        # the lowest band alone, at Γ
print(isinstance(out, tqc.Refusal), bool(out))
#  True False
print(out)
#  Refusal[multiplet_split]: the bands [0] cut through the degenerate multiplet [0, 1] at k = [0.0, 0.0] (energies [-3.162277660168, -3.162277660168], spread 0 at tolerance 1e-08): on part of a degenerate multiplet the characters depend on the basis chosen inside the multiplet. Ask for every band of the multiplet, or end the bands at a gap.  (measured 0, threshold 1e-08)
#      context: bands, energies, k, multiplet
```

## How to read it

- **`True False`**: `isinstance(out, tqc.Refusal)` is the test to branch on; it tells a
  refusal apart from every value a call returns. A refusal is falsy as well, but so are
  some ordinary verdicts, which a truth test would read as refusals:
  `stably_equivalent_pairs`, which has no pairs for most groups, and `subtract` of two
  equal contents, which leaves none.
- **`Refusal[multiplet_split]`**: `out.gate`, the name of the check that stopped the call.
  The name is stable, and it is the heading of the refusal's entry in
  [the reference](../reference/refusals.md#multiplet_split).
- **the sentence after it**: `out.reason`, the diagnosis, naming what is at fault (here the
  bands, the k-point and their energies) and what to do.
- **`(measured 0, threshold 1e-08)`**: `out.value`, the number the check measured, and
  `out.threshold`, what the check demanded. Here the value is the energy spread of the
  levels the lowest band belongs to, in the units of `H`, and two levels closer than the
  threshold `1e-8` are one multiplet. A check with no number to measure, such as a word
  that names nothing, carries `None` in both.
- **`context: bands, energies, k, multiplet`**: the keys of `out.context`, everything
  needed to act on the refusal without running the call again.

```python
print(out.gate, out.value, out.threshold, out.context["multiplet"])
#  multiplet_split 0.0 1e-08 (0, 1)
print(tqc.corep_content(r.rep, gamma, bands=2).counts)
#  {'-GM6': 1}
```

The verdict is physics. Kane–Mele keeps time reversal, so every level at Γ, a
time-reversal-invariant momentum, is a Kramers pair, and the lowest band alone is half of
one: its characters would depend on the basis the diagonalization chose inside the pair,
so it has no label. The pair has one, `-GM6`
([label an isolated set of bands](labels.md#label-an-isolated-set-of-bands)).

## The group named in it

A refusal that `realized_group` returns once the group is identified opens by naming it,
with its symbol and its number, and carries it in `context["group"]`, so a question asked
after the group never loses it:

```python
bad = tqc.realized_group(km, typat=[0, 1], characters={0: "E"})
print(bad)
#  Refusal[ambiguous_multiplet]: The group identified is P3m11' #156.50, every operation of it obeyed; its representation was not built: orbital 0: 'E' is 2-dimensional, and the 1 orbital of site 0 written with it makes no whole number of copies of it, so which orbitals hold it is not said. Site 0 carries 1 orbital (0,), so no 2-dimensional multiplet sits on it at all: name a content of one orbital instead (A1, A2, -E1).
#      context: declared_irrep, dim, group, multiplet, multiplets, orbitals, site
print(bad.context["group"].name, bad.context["group"].number_str)
#  P3m11' 156.50
```

The declaration puts the two-dimensional `E` on a site that holds one orbital, and no
content of one orbital is two-dimensional. A refusal reached while the
group is still being measured names the group the cell has at that point, an upper bound
on what the Hamiltonian obeys. The question which group a Hamiltonian has is itself refused
in two cases only, each about how the model is written
([when the group question is refused](../background/measurement.md#when-the-group-question-is-refused)).

## Four kinds

Every refusal is one of four kinds, and the kind says where the fix lies;
[the reference](../reference/refusals.md) groups them so.

1. **A verdict about the physics.** What was asked does not exist for these bands or this
   model as written: a set of bands cutting a multiplet, as above; a
   [symmetry vector](../reference/glossary.md#words-the-pages-use), the count of each
   corep at each high-symmetry point, that fails the group's compatibility relations and
   so has no [indicator class](../background/tqc-primer.md#how-a-symmetry-indicator-is-computed);
   a representation the tables' conventions cannot carry; a model written on a supercell
   of its own primitive cell. The reason says what was measured, and the fix is in the
   question or in the model.
2. **A practical limit.** A search, an enumeration or a memory budget would pass its
   ceiling, or the installation lacks the tables a question needs. The reason names the
   number, or the data, that runs it, and the same call with that number does. Below, the
   projection `symmetrize=True` forces on an exact model, given a budget set small on
   purpose:

   ```python
   limit = tqc.realized_group(km, typat=[0, 1], characters={0: "A1"}, symmetrize=True,
                              max_bytes=1e4)
   print(limit.gate, limit.context["group"].name, f"{limit.value:.3g} against {limit.threshold:.3g}")
   #  projection_too_large P3m11' 3.01e+04 against 1e+04
   print(repr(tqc.realized_group(km, typat=[0, 1], characters={0: "A1"}, symmetrize=True,
                                 max_bytes=1e5)))
   #  RealizedGroup(P3m11' #156.50; symmetrized (defect 1.35e-15); on a substrate (the default), the free-standing group #156.50 agrees)
   ```

   The projection needs `3.01e4` bytes against the `1e4` allowed; its reason names the
   `max_bytes=` that runs it, and with `1e5` it runs.
3. **A grammar error in a declaration or a question.** `characters=` or `spin=` says
   something that names nothing at the site: a word that is not an irrep of its site group,
   a doublet on a site of one orbital as above, the two spin states of one Wannier function
   on two centres. The reason lists what would parse. A call asked a question it does not take,
   such as a k-point off a layer's plane, a
   [Seitz symbol](../background/conventions.md#operations-a-seitz-symbol-and-a-cartesian-name)
   the group does not carry or a negative multiplicity, refuses the same way.
4. **The package's checks on itself.** A few checks measure an identity that holds in exact
   arithmetic, or two measurements of one thing that must agree: a representation of a
   group the Hamiltonian was measured to obey, the projection onto the symmetric part of
   `H`, the tables' setting. One that fails means the package has met a case it does not
   handle; nothing in the model is to change, and the refusal is worth reporting with the
   model and the call ([these checks](../reference/refusals.md#the-packages-checks-on-itself)).

## A refusal inside a record

A call that reads several points keeps each point's refusal inside its record. `hsp_table`
labels the lowest band at every high-symmetry point, and two points, the
time-reversal-invariant `GM` and `M`, refuse; the table is truthy and says it is
incomplete, and the call after it does not fill the gap:

```python
table = tqc.hsp_table(r.rep, bands=1)
print(bool(table), table.complete, {point: why.gate for point, why in table.refusals.items()})
#  True False {'GM': 'multiplet_split', 'M': 'multiplet_split'}
missing = tqc.decompose(table, r.group)
print(missing.gate, missing.context["refusals"])
#  incomplete_content {'GM': 'multiplet_split', 'M': 'multiplet_split'}
```

`decompose` refuses `incomplete_content`, naming the points and what refused at each: a
decomposition of a symmetry vector with a point missing would be a confident verdict on a
different vector, and `symmetry_indicator` refuses it the same way. A refusal is carried
forward.

## Turning one into an exception

No call raises a refusal. A script that wants to stop at the first one asks for it:
`.raise_()` raises `RefusalError`, a `RuntimeError` that carries the refusal on `.refusal`.

```python
try:
    out.raise_()
except tqc.RefusalError as error:
    print(type(error).__name__, isinstance(error, RuntimeError), error.refusal is out)
#  RefusalError True True
```

## What is not a refusal

- **A malformed argument** raises `ValueError` before anything about the physics is
  measured: here a precision given beside a tolerance it derives. A keyword a call does
  not take raises `TypeError`, as in any Python call.

  ```python
  try:
      tqc.realized_group(km, precision=1e-6, tol=1e-5)
  except ValueError as error:
      print(str(error).split(":")[0])
  #  precision=1e-06 and tol declare the same thing twice
  ```

- **The package disagreeing with itself** raises `InternalError`: two measurements of one
  Hamiltonian that contradict each other, such as a layer's two environments contradicting
  the statement that the substrate's group is the part of the free-standing one that
  preserves the layer's normal, or a representation of the identified group that the covariance check refuses beyond
  the declared tolerance. So does a presentation of the model that changes it, though it
  only relabels the Hamiltonian or moves it by a gauge: the slab and the decoupled stack
  of a layer, the spinless presentation of a spinful model and the model with its centres
  moved are each measured against the model on eight random `k`. It is a defect in the
  package and says nothing about the model: report it, with the model and the call.
- **`MagneticScopeWarning`** is a warning: time reversal was measured
  broken, and the group returned is the unitary group the Hamiltonian obeys
  ([time reversal is measured](group.md#time-reversal-is-measured)).

## Further

Every refusal, with its meaning, its usual cause and what to do:
[the refusals reference](../reference/refusals.md). The refusals each call can return: its
entry in [the functions reference](../reference/functions.md). The record's fields:
[`Refusal`](../reference/records.md#refusal). The two cases in which the group question is
refused: [how pythtqc reads a Hamiltonian](../background/measurement.md#when-the-group-question-is-refused).
