# Quick start: Kane-Mele, end to end

This page shows a first analysis with pythTQC, on one model from pythtb's
collection. The model is Kane-Mele
([Phys. Rev. Lett. 95, 226801 (2005)](https://doi.org/10.1103/PhysRevLett.95.226801)).
`kane_mele(delta, t, soc, rashba)` returns a spinful model on the honeycomb
lattice. Its terms are an on-site energy `+delta` on one sublattice and `-delta`
on the other, a nearest-neighbor hopping `t`, an intrinsic spin-orbit coupling
`soc` between second neighbors, and a Rashba coupling `rashba` between nearest
neighbors. `realized_group` finds the space group the model's Hamiltonian obeys.
`coreps_at` reads the little-group representations the model's two occupied
bands carry at each high-symmetry point. `decompose` asks whether those bands
have the symmetry data vector of an atomic limit. Each block uses the names the
blocks before it define. The lines under a statement that start with `#  ` are
what it prints.

## The group

`typat=` gives the species of each orbital's atom. Here the two sublattices are
atoms of different species, so no operation of the group exchanges them.
`characters=` declares orbitals by their index, each as a function such as
`"s"` or as a site irrep such as `"A1"`. Here orbital 0 is an `s` orbital,
declared as `A1`, the irrep of its site-symmetry group that an `s` orbital
carries. `realized_group` takes the pythtb model with these two arguments, and
names the Wyckoff position of each orbital in the group it finds. The example
finds the group of Kane-Mele with a staggered potential, spin-orbit coupling and
a Rashba term:

```python
from pythtb.models import kane_mele
import pythtqc as tqc

km2 = kane_mele(1.0, -1.0, 0.3, 0.25)           # spinful honeycomb, 4 bands

r = tqc.realized_group(km2, typat=[0, 1],       # one species per sublattice
                       characters={0: "A1"})    # orbital 0 is an s orbital
group = r.group
print(f"#{group.number_str} {group.name}, orbitals at Wyckoff "
      + " and ".join(o.wyckoff for o in r.wyckoff_orbits))
#  #156.50 P3m11', orbitals at Wyckoff 1c and 1b
```

- `P3m11'`: the group, `r.group`, by its Hermann-Mauguin symbol, `group.name`.
  The trailing `1'` marks a grey group: the space group `P3m1` with time
  reversal as one of its operations.
- `156.50`: the group's number in the Belov-Neronova-Smirnova (BNS) notation
  for magnetic space groups, `group.number_str`.
- `1c and 1b`: the Wyckoff positions of the two orbitals, `o.wyckoff` for each
  row `o` of `r.wyckoff_orbits`, in the setting of the irreptables package's
  tables. `P3m1` has three Wyckoff positions whose site-symmetry group is `3m`,
  `3m1'` with time reversal. The third, `1a`, is the center of the hexagon and
  holds no orbital.

The record returned, `r`, also holds the representation of the group on the
orbitals, `r.rep`. The calls below are passed `r`.

A call that stops at one of its checks returns a `Refusal` in place of its
record. A refusal is falsy. Its printout names the check that stopped the call
and what it measured. No call on this page refuses.
[What does a refusal mean, and what do I do about it?](../guide/refusals.md)
shows how to read one.

## What the orbitals are

`r.wyckoff_orbits` is the model's atomic-limit table: the Wyckoff positions of
the model's orbitals and the site irreps they carry there. It has one row for
each multiplet, the orbitals the Hamiltonian binds together at a site, with the
orbitals the operations carry it into. The example prints the table for
Kane-Mele's two orbitals:

```python
print(r.wyckoff_orbits.summary())
#  Wyckoff  site symmetry  orbitals  label  determined by  declared
#       1c           3m1'         0    -E1    declaration        A1
#       1b           3m1'         1    -E1    hamiltonian
```

- `1c`, under `Wyckoff`: the Wyckoff position of the row.
- `3m1'`, under `site symmetry`: the site-symmetry group, with the `1'` of time
  reversal.
- `0`, under `orbitals`: the indices of the model's orbitals in the row.
- `-E1`, under `label`: the site irrep the row's orbitals carry at one of its
  sites. The leading `-` marks a double-valued irrep, one carried by states of
  half-integer spin. `-E1` is the site irrep of an `s` orbital with its spin at
  a `3m` site.
- `declaration` and `hamiltonian`, under `determined by`: how the site irrep is
  known. `declaration` means that `characters=` stated it. `hamiltonian` means
  that nothing declared it, and the hoppings and the declarations fix it.
  `lowest-l`, not printed here, means that the hoppings and the declarations
  leave it open, and the lowest-angular-momentum rule assigned it: the site
  irrep carried by the functions of lowest angular momentum the hoppings allow.
  [Returned records](../reference/records.md#the-atomic-limit-table) gives all
  four values.
- `A1`, under `declared`: what `characters=` declared for the row, blank where
  nothing was.

pythtb attaches no site irrep to an orbital. The hoppings fix how the orbitals
transform relative to one another. `A1` is the site irrep of orbital 0 without
its spin. The `label` column holds its site irrep with its spin, `-E1`.
`characters={0: "-E1"}` declares the same row by that site irrep. `A2` with its
spin is `-E1` as well, so declaring orbital 0 as `A1`, `A2` or `-E1` gives the
same site irreps and the same coreps below. With nothing declared, both rows
read `hamiltonian`. So the declaration in the example changes no site irrep and
no corep. It is there to show the form a declaration takes.

A declaration the hoppings do not agree with lowers the group. Each operation
the hoppings are not symmetric under, with the orbitals as declared, is left out
of the group returned
([How do I say what my orbitals are?](../guide/orbitals.md)).

## The bands, by symmetry

The group holds time reversal, so the little-group representations the model's
bands carry are coreps: the irreducible corepresentations of a group with
antiunitary operations. `high_symmetry_points` returns the group's high-symmetry
points, by name, in the model's reduced coordinates. `coreps_at` reads the
coreps a set of bands carries at one of them. `bands=` is the set of bands. It
is given either as an integer n for the lowest n bands, or as a sequence of band
indices. Coreps can only be returned if the set contains all the states in
degenerate multiplets at the point. A set that cuts a multiplet is refused with
`multiplet_split`, which names the multiplet and its energies. The example reads
the coreps of Kane-Mele's two occupied bands at Γ, `K` and `M`:

```python
hsp = tqc.high_symmetry_points(group)           # in the model's reduced coordinates
for name in ("GM", "K", "M"):
    content = tqc.coreps_at(r, hsp[name], bands=2)
    print(content)
#  CorepsAt(GM: -GM6)
#  CorepsAt(K: -K5 + -K6)
#  CorepsAt(M: -M3M4)
```

- `CorepsAt(GM: -GM6)`: the coreps the two bands carry at Γ. `GM` is the
  tables' name for Γ. `-GM6` is one two-dimensional corep, so the two bands are
  one degenerate multiplet there. The leading `-` marks a double-valued corep.
- `-K5 + -K6`: at `K` the two bands are two multiplets, one carrying the
  one-dimensional corep `-K5` and the other `-K6`.
- `-M3M4`: at `M` the two bands are one multiplet. `-M3M4` is one
  two-dimensional corep, made of the one-dimensional irreps `-M3` and `-M4`,
  which time reversal pairs.

## Atomic limits

An atomic limit is a set of bands whose states can be written as exponentially
localized, symmetric Wannier functions. A site irrep at a Wyckoff position
induces a band representation. An elementary band representation (EBR) is one
that is not a sum of others. The corep multiplicities of a set of bands at every
high-symmetry point form its symmetry data vector. A decomposition writes a
symmetry data vector as a sum of the symmetry data vectors of EBRs, with
nonnegative integer multiplicities. The symmetry data vector of an atomic limit
has one. The definitions are in Bradlyn *et al.*,
[*Topological quantum chemistry*, Nature **547**, 298 (2017)](https://doi.org/10.1038/nature23268).

`hsp_table` reads the coreps of a set of bands at every high-symmetry point at
once. `decompose` takes the table and returns every decomposition of its
symmetry data vector, up to `cap=` of them. The example decomposes Kane-Mele's
two occupied bands:

```python
t = tqc.hsp_table(r, bands=2)                   # the two occupied bands
print(tqc.format_coreps(t.symmetry_data))
#  -GM6 + -K5 + -K6 + -M3M4

d = tqc.decompose(t)
print(d.decomposable, "|", d.candidate_sums[0].label)
#  True | -E1 at 1a (3m1')
```

- `-GM6 + -K5 + -K6 + -M3M4`: the symmetry data vector of the two bands,
  `t.symmetry_data`, written on one line by `format_coreps`.
- `True`: whether the symmetry data vector has a decomposition,
  `d.decomposable`.
- `-E1 at 1a (3m1')`: the one decomposition, `d.candidate_sums[0]`, by its
  `label`. Each term is an EBR, written as the site irrep that induces it, its
  Wyckoff position, and its site-symmetry group in parentheses.

The two bands have the symmetry data vector of the EBR induced from `-E1` at
`1a`, the center of the hexagon, where the model has no orbital. At these
parameters Kane-Mele is a quantum spin Hall insulator: its two occupied bands
have the Z2 invariant 1, and are not an atomic limit. A decomposition says that
a set of bands has the symmetry data vector of an atomic limit, and not that it
is one.

## Where to go next

- [Functions](../reference/functions.md) gives every argument of
  [`realized_group`](../reference/functions.md#realized_group),
  [`high_symmetry_points`](../reference/functions.md#high_symmetry_points),
  [`coreps_at`](../reference/functions.md#coreps_at),
  [`hsp_table`](../reference/functions.md#hsp_table),
  [`format_coreps`](../reference/functions.md#format_coreps) and
  [`decompose`](../reference/functions.md#decompose), what each returns, and the
  refusals each can return.
- [Returned records](../reference/records.md) lists every field of the records
  printed here.
- [How do I say what my orbitals are?](../guide/orbitals.md) gives the forms a
  declaration takes.
- [Which coreps do these bands carry?](../guide/labels.md) reads the coreps of a
  set of bands, and says when a label rests on a choice the hoppings leave open.
- [Could these bands be an atomic limit?](../guide/atomic-limits.md) gives what
  a decomposition can and cannot say.
- [Fu-Kane-Mele: a 3D crystal, a distortion, the Z4 indicator](tutorial-3d.md)
  takes a three-dimensional crystal to its symmetry-indicator class.
