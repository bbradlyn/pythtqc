# Quick start: Kane–Mele, end to end

This page takes one model from the first call to the last: the space group its
Hamiltonian has, the symmetry labels of its two occupied bands, and whether
those two bands could be an atomic limit. The model is Kane–Mele on the
honeycomb, from pythtb's own collection: `kane_mele(delta, t, soc, rashba)`
puts an on-site energy `+delta` on one sublattice and `-delta` on the other, a
nearest-neighbour hopping `t`, an intrinsic spin–orbit coupling `soc` and a
Rashba coupling `rashba`, with spin. The blocks run in order,
as written, and the lines under a statement are what it prints.

## The group

```python
from pythtb.models import kane_mele
import pythtqc as tqc

km2 = kane_mele(1.0, -1.0, 0.3, 0.25)           # 2D spinful honeycomb, 4 bands

r = tqc.realized_group(km2, typat=[0, 1],       # typat: the type of each atom -- the A and
                                                # B sublattices are different atoms
                       characters={0: "A1"})    # ... and orbital 0 IS an s: A1 of its site group
model, group, assignment, validated, rep = r
sites = tqc.wyckoff_sites_tables(group, assignment.site_positions)
print(f"#{group.number_str} {group.name}, orbitals at Wyckoff "
      + " and ".join(f"{s.multiplicity}{s.letter}" for s in sites))
#  #156.50 P3m11', orbitals at Wyckoff 1c and 1b
```

One call measures the group of `H(k)`, assigns the orbitals to its sites and
builds the model's representation of it. `156.50` is the *grey* group: space
group 156 with time reversal adjoined, which the trailing `1'` marks and which
makes the labels below coreps. A **corep**
(co-representation) is the irreducible object of a group containing
antiunitary operations, and its name spells out the unitary irreps time
reversal glues together. The two sublattices land on `1c` and `1b`, so the
hexagon centre `1a` carries no orbital; that matters two calls below.

The record the call returns unpacks into five: the model, the group, the
orbitals' assignment to its sites, the verdict on each operation, and the
representation. A call that cannot complete returns a `Refusal` in place of
its record, a falsy record whose printout names what was being measured and
what stopped it; [the reference](../reference/functions.md) lists every call's
refusals, and no call on this page meets one.

## What the orbitals are

A pythtb orbital carries no symmetry character: the hoppings fix how the
orbitals transform relative to one another, not what they are. `characters=`
states what they are, exactly as `typat=` states what the atoms are, and
`rep.characters` is the model's atomic-limit table, one row per orbital, whose
`summary()` reads the declaration back:

```python
print(rep.characters.summary())
#  orbital  Wyckoff  site symmetry  function  irrep  named by  determined by  orbital characters
#        0       1c             3m              -E1    tables    declaration  E=+1, C3(z)=+1, m(x)=+1
#        1       1b             3m              -E1    tables    hamiltonian  E=+1, C3(z)=+1, m(x)=+1
```

A declaration states the **orbital** factor: `A1` is the irrep of the site
group `3m` that an `s` orbital carries. These states carry spin, so the irrep
returned is the double-valued site corep they carry there, `-E1`.
The two are one row read twice, and both spellings declare: `A1` the orbital
factor, and `-E1` the corep alone, which `A2` lifts to as well, leaving which
of the two the orbital holds to the lowest-`l` rule: what no declaration fixes
takes the lowest-angular-momentum representation the hoppings allow
([saying what the orbitals are](../guide/orbitals.md#how-to-read-it)).

Each column is a field of the row, and `determined by` says how its site irrep
is known: `declaration` for orbital 0, which was stated, and `hamiltonian` for
orbital 1, which the hoppings fix relative to it. Every column, and the lines a
table can print under its rows, are read in
[the records reference](../reference/records.md#the-atomic-limit-table).

A declaration fixes every orbital the hoppings couple to it, anything
undeclared being assigned by the lowest-`l` rule, and where the hoppings are
not covariant under what was declared, the crystal declared has less symmetry
and the group **descends**. Read the echo back every time: a mis-oriented
declaration relabels the coreps at once and nothing else complains. With spin
it can relabel none, since a spinor's character vanishes on every mirror and twofold
rotation: `B1` and `B2` of `4mm`, which differ only on the mirrors, become one
site corep, and the table's `orbital characters` column is where the difference
shows ([the forms a declaration takes](../guide/orbitals.md#the-forms-a-declaration-takes)).
A `wannier90` spinor import needs a second declaration of the same kind, since
pythtb's importer returns `nspin = 1` whatever the calculation did: `spin=`
says which orbital is the `↑` partner of which
([a wannier90 spinor import](../guide/spin.md#a-wannier90-spinor-import)).

## The bands, by symmetry

Now the corep content of the two occupied bands, at each high-symmetry point:

```python
hsp = tqc.high_symmetry_points(group)           # in THIS cell's reduced coordinates
for name in ("GM", "K", "M"):
    content = tqc.corep_content(rep, hsp[name], bands=2)
    print(content)
#  CorepContent(GM: -GM6, partition=(0,))
#  CorepContent(K: -K5 + -K6, partition=(0, 1))
#  CorepContent(M: -M3M4, partition=(0,))
```

Read those three lines as physics. At `GM` the two occupied bands are one
two-dimensional corep `-GM6` — a single Kramers level, `partition=(0,)` saying
both bands sit in the same row; the leading `-` marks a double-valued, spinful
label. At `K` they split into two one-dimensional coreps, `partition=(0, 1)`
saying the two rows differ. At `M` they are `-M3M4`: two
one-dimensional unitary irreps, `M3` and `M4`, that time reversal glues into
one physically irreducible level. The partition is the statement that survives
every phase convention; the labels are canonical only relative to the declared
characters. With no `states=` given, the states are left to the call: one
diagonalization per k-point, reused by every operation there, since a second
one returns a different basis inside a degenerate multiplet and changes the
traces.

## Atomic limits

The content at every high-symmetry point at once is the band set's symmetry
vector, and that vector is what a decomposition is asked of. `decompose`
writes it as a sum, with nonnegative integer multiplicities, of **elementary
band representations** (EBRs) — each the band set one site-symmetry irrep
induces from its Wyckoff position, where that band set is no sum of others —
or says that no such sum exists:

```python
t = tqc.hsp_table(rep, bands=2)                 # the same content everywhere, one solve
print(tqc.format_counts(t.counts))
#  -GM6 + -K5 + -K6 + -M3M4

d = tqc.decompose(t)
print(d.verdict, "|", d.solution_strings()[0])
#  symmetry-indicated decomposable | 1 x -E1↑G(2)@1a(3m1',3m)
```

These two bands could be the atomic limit of one spinful orbital in the `-E1`
corep of the site group at Wyckoff `1a` — the hexagon centre, where the model
has no orbital at all — induced to two bands, which is what `↑G(2)` says. And
that is the whole claim: a decomposition verdict is a statement about symmetry
indicators, so any obstruction to an atomic limit must lie beyond them.

## The calls on this page

Every argument of each call above, what it returns and how it refuses are in
the reference: [`realized_group`](../reference/functions.md#realized_group),
[`wyckoff_sites_tables`](../reference/functions.md#wyckoff_sites_tables),
[`high_symmetry_points`](../reference/functions.md#high_symmetry_points),
[`corep_content`](../reference/functions.md#corep_content),
[`hsp_table`](../reference/functions.md#hsp_table),
[`format_counts`](../reference/functions.md#format_counts) and
[`decompose`](../reference/functions.md#decompose).
