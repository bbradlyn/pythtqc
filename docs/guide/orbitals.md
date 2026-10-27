# How do I say what my orbitals are?

A pythtb orbital is a position and a list of hoppings, with no symmetry
character of its own: the hoppings fix how the orbitals transform relative to
one another, not what they are
([why](../background/measurement.md#what-the-hoppings-fix)). `characters=`
says what they are, as `typat=` says which sites hold the same kind of atom.
Both are statements about the crystal, never checked against the Hamiltonian,
which is measured against the crystal they define. This chapter writes a
declaration, reads it back, and shows what one can do to the group. Its model
is pythtb's Kane–Mele, `kane_mele(delta, t, soc, rashba)`: one orbital with
spin on each sublattice of the honeycomb, on-site energies `±delta`, hopping
`t`, intrinsic spin–orbit coupling `soc` and Rashba coupling `rashba`; where a
site holds two orbitals, `pythtqc.models.lin3g`.

## The call

```python
import pythtqc as tqc
from pythtb.models import kane_mele
from pythtqc import models

km = kane_mele(1.0, -1.0, 0.3, 0.25)            # delta, t, soc, rashba: two orbitals, with spin
r = tqc.realized_group(km, typat=[0, 1], characters={0: "s"})
print(r.group.name, r.group.number_str)
#  P3m11' 156.50
print(r.rep.characters.summary())
#  orbital  Wyckoff  site symmetry  function  irrep  named by  determined by  orbital characters
#        0       1c             3m         s    -E1    tables    declaration  E=+1, C3(z)=+1, m(x)=+1
#        1       1b             3m              -E1    tables    hamiltonian  E=+1, C3(z)=+1, m(x)=+1
```

## How to read it

`P3m11'` (`156.50`) is the group `realized_group` measured on the Hamiltonian
([what group does my Hamiltonian have?](group.md)), space group 156 with time
reversal adjoined, the trailing `1'`: the two sublattices declared different
atoms, on a substrate, the default for a layer ([layers](layers.md)). Under
it is the model's
[atomic-limit table](../reference/records.md#the-atomic-limit-table), one row
per orbital:

- **Wyckoff**, **site symmetry**: the Wyckoff position, multiplicity and
  letter, in the setting of the tables the labels come from
  ([conventions](../background/conventions.md)), and its site-symmetry group:
  `1c` and `1b`, each `3m`.
- **function**: the word declared, spelled as `characters=` takes it; blank
  where no function was declared.
- **irrep**: the site irrep the states carry, an irreducible representation of
  the site group. With spin it is the double-valued site
  [corep](../background/tqc-primer.md#little-groups-irreps-and-coreps), marked
  by its leading `-`: `-E1` is `A1` of `3m`, what an `s` orbital carries,
  times the spinor. A declaration names the orbital factor, `s` or `A1`, or
  the corep, `-E1`.
- **named by**: `tables`: the tables list the
  [band representation](../background/tqc-primer.md#band-representations-and-atomic-limits)
  these states induce at that position, and the name is the one they give the
  site irrep there.
- **determined by**: how the irrep is known. Orbital 1 reads `hamiltonian`:
  nothing was declared for it, and the hoppings carry it relative to orbital 0,
  so the one declaration fixed both. A declaration speaks for every orbital the
  hoppings couple to it.
- **orbital characters**: the orbital factor's character on one operation of
  each class, in the model's Cartesian frame (`m(x)` is the mirror whose normal
  is x). It is single-valued with or without spin, and it says which function
  was meant.

Each column is a field of the row, and beside the rows the table holds, for
each orbit, the band representation it induces:

```python
row = r.rep.characters.orbitals[0]
print(repr(row.function), row.named_by, row.generator_character["m(x)"].real)
#  's' tables 1.0
for br in r.rep.characters.band_representations:
    print(br.orbitals, br.wyckoff, br.terms)
#  (0,) 1c ((1, '-E1↑G(2)', "1c(3m1',3m)"),)
#  (1,) 1b ((1, '-E1↑G(2)', "1b(3m1',3m)"),)
```

`function` is the word declared, empty on orbital 1. `generator_character` is
the orbital characters column as a mapping, from the name of one operation per
class to the orbital factor's character on it, a complex number: `+1` on
`m(x)`, under which an `s` orbital is even. A
[band representation](../reference/records.md#bandrepresentation) holds its
orbit's orbitals and Wyckoff position, and as `terms` the sum they induce, each
term a multiplicity, an elementary band representation and its position. Each
sublattice induces the one `-E1↑G(2)` at its own position, an entry of the
tables, and the row takes the name that entry gives its site irrep: `tables`
in `named_by`. An orbit at a position where the tables list no band
representation is named by the `point group`, and its sum is written at other
positions ([below](#a-declaration-that-lowers-the-group)).

Every field of a row, and the lines a table can print under its rows, are in
[the records reference](../reference/records.md#the-atomic-limit-table);
operation names and label spellings are in
[conventions](../background/conventions.md).

## When to declare

Undeclared, an orbital's site irrep is assigned by the lowest-ℓ rule: where
the hoppings leave several possible, the one carried by the lowest
angular-momentum functions is taken
([the rule and its keys](../background/measurement.md#the-lowest-angular-momentum-rule)).
Each row says whether its label rests on that choice, and that is the test for
when to declare. The honeycomb without stagger or Rashba coupling, on a
substrate, free-standing, and free-standing with its orbitals declared `p_z`,
each with its rows and the labels of its two lowest bands at the high-symmetry
points (`hsp_table`, [labelling the bands](labels.md)):

```python
km0 = kane_mele(0.0, -1.0, 0.3, 0.0)            # delta = 0, rashba = 0
sub = tqc.realized_group(km0, typat=[0, 0])                                 # on a substrate
free = tqc.realized_group(km0, typat=[0, 0], environment="free-standing")
pz = tqc.realized_group(km0, typat=[0, 0], environment="free-standing", characters={0: "pz"})
for rec in (sub, free, pz):
    rows = rec.rep.characters
    print(rec.group.number_str, rows.irreps, [row.determined_by for row in rows.orbitals],
          tqc.hsp_table(rec.rep, bands=2).counts)
#  183.186 ('-E1', '-E1') ['hamiltonian', 'hamiltonian'] {'-GM9': 1, '-K6': 1, '-M5': 1}
#  191.234 ('-E1', '-E1') ['lowest-l', 'lowest-l'] {'-GM9': 1, '-K8': 1, '-M6': 1}
#  191.234 ('-E2', '-E2') ['declaration', 'hamiltonian'] {'-GM12': 1, '-K9': 1, '-M5': 1}
```

On a substrate (`P6mm1'`, `183.186`) no mirror lies in the layer's plane, so
an `s` and a `p_z` orbital carry the same characters on every operation of the
site group `3m`; with spin, every site irrep the hoppings leave possible
becomes the one corep `-E1`, both rows read `hamiltonian`, and declaring `pz`
there changes nothing. Free-standing (`P6/mmm1'`, `191.234`) the site group
`-6m2` holds the horizontal mirror, which tells the two apart: the rows read
`lowest-l`, and the rule takes `s`. A honeycomb of carbon is a `p_z` crystal,
and declaring it returns `-E2` and renames the two lowest bands at every
high-symmetry point. Their [symmetry-indicator class](indicators.md) moves
with them:

```python
for rec in (free, pz):
    si = tqc.symmetry_indicator(tqc.hsp_table(rec.rep, bands=2))
    print(si.structure, si.values, si.order, {name: si.canonical[name] for name in ("z2w,3", "z4")})
#  Z6 x Z12 (1, 0) 6 {'z2w,3': 1, 'z4': 2}
#  Z6 x Z12 (5, 0) 6 {'z2w,3': 1, 'z4': 2}
```

In the indicator group `ℤ6 × ℤ12`, the class is `(1, 0)` read as `s` and
`(5, 0)`, its inverse, declared `p_z`. What survives the declaration is the
order of the class, 6, and every component that is its own inverse, here
`z2w,3 = 1` and `z4 = 2`
([the names a class is spoken in](indicators.md#the-names-a-class-is-spoken-in)),
so a statement made in those terms does not move, while the class itself needs
the orbitals stated. Declare whenever a row reads `lowest-l`: its label is a
choice the hoppings cannot make. An orbital the hoppings couple to no other is
the extreme case: nothing measured relates it to the rest, so its declaration
breaks no operation and is the only statement of what it is.

## The forms a declaration takes

lin3g is four orbitals with spin on a square layer, eight bands: orbitals 0
and 1 share the site `1a`, orbitals 2 and 3 sit on the two sites of `2c`.
Declared as an `s` and a `d_{x²−y²}` at `1a` and an `s` on each `2c` site:

```python
lin = tqc.realized_group(models.lin3g(), typat=[0, 0, 1, 1],
                         characters=["s", "dx2-y2", "s", "s"])
print(lin.rep.characters.summary())
#  orbital  Wyckoff  site symmetry  function  irrep  named by  determined by  orbital characters
#        0       1a            4mm         s    -E1    tables    declaration  E=+1, C2(z)=+1, C4(z)=+1, m(x)=+1, m([110])=+1
#        1       1a            4mm    dx2-y2    -E2    tables    declaration  E=+1, C2(z)=+1, C4(z)=-1, m(x)=+1, m([110])=-1
#        2       2c            mm2         s     -E    tables    declaration  E=+1, C2(z)=+1, m(y)=+1, m(x)=+1
#        3       2c            mm2         s     -E    tables    declaration  E=+1, C2(z)=+1, m(y)=+1, m(x)=+1
```

A declaration has several spellings, and the same call reads each:

```python
def irreps(characters):                         # lin3g, its orbitals declared again
    rec = tqc.realized_group(models.lin3g(), typat=[0, 0, 1, 1], characters=characters)
    return rec.rep.characters.irreps

print(irreps(["s", "dx2-y2", "s", "s"]))        # one word per orbital
#  ('-E1', '-E2', '-E', '-E')
print(irreps(["A1", "B1", None, None]))         # site irreps; None claims nothing
#  ('-E1', '-E2', '-E', '-E')
print(irreps(["-E1", "-E2", None, None]))       # site coreps
#  ('-E1', '-E2', '-E', '-E')
print(irreps({1: "dx2-y2"}))                    # a mapping: only the orbitals named
#  ('-E1', '-E2', '-E', '-E')
print(irreps({(0, 1): "A1 + B1"}))              # what two orbitals hold together
#  ('-E1', '-E2', '-E', '-E')
print(irreps({0: {"C4(z)": -1, "m(x)": +1}}))   # characters on one operation per class
#  ('-E2', '-E1', '-E', '-E')
print(irreps({0: {3: 1, 6: 1}}))                # the operations by their index: A1 again
#  ('-E1', '-E2', '-E', '-E')
print(tqc.operations(lin.rep)[3].name, tqc.operations(lin.rep)[6].name)
#  C4(z) m(x)
```

- **A list**, one entry per orbital, a function or a site irrep, `None` where
  nothing is claimed.
- **A site corep**, with spin: `-E1` of `4mm` is `A1` or `A2` times the
  spinor, and the lowest-ℓ rule chooses between them.
- **A mapping** from orbital to word: `{1: "dx2-y2"}` fixes orbital 0 too,
  through the hoppings.
- **A key of several orbitals**, for what the orbitals of one site hold
  together: `"A1 + B1"` is the content of `1a` as one sum. It is read on the
  multiplets the hoppings measure there, and each row names what its own
  multiplet carries, `-E1` and `-E2`, as for the list. Wannier functions
  turned inside their site hold that sum in any basis, so this is the form for
  functions no one orientation describes.
- **Generator characters**, a mapping from operation to character, each
  operation named as `tqc.operations` prints it or given by its index there
  ([conventions](../background/conventions.md#operations-a-seitz-symbol-and-a-cartesian-name)).
  `C4(z) = −1` with `m(x) = +1` is `B1`, a `d_{x²−y²}` on orbital 0, and every
  row moves with it. No irrep is named, so no orientation convention can spoil
  it: `B1` of `4mm` is here the one even under `m(x)`, the mirror whose normal
  is x, where a source that takes the diagonal mirrors as its reference calls
  the same function `B2`.

With spin, a mis-oriented declaration need not change any label. `B1` and
`B2` of `4mm` differ only on the mirrors, where a spinor's character is zero,
so both become `-E2`: a `B2` written for orbital 1's `B1` returns the same site
coreps and the same labels at every high-symmetry point, and only the
orbital characters column shows what was stated:

```python
b2 = tqc.realized_group(models.lin3g(), typat=[0, 0, 1, 1], characters={1: "B2"}).rep
print(b2.characters.irreps, tqc.hsp_table(b2, bands=6).counts == tqc.hsp_table(lin.rep, bands=6).counts)
#  ('-E1', '-E2', '-E', '-E') True
print(b2.characters.summary().splitlines()[2])
#        1       1a            4mm              -E2    tables    declaration  E=+1, C2(z)=+1, C4(z)=-1, m(x)=-1, m([110])=+1
```

Whatever the form, that column of the table returned is where to check what
was stated.

The words are function shorthands (`s`, `px`, `py`, `pz`, `dxy`, `dyz`,
`dzx`, `dx2-y2`, `dz2`), shells by their angular momentum (`s`, `p`, `d`, `f`:
what the real harmonics of that ℓ carry at the site, `A1 + E` for `p` at a
`3m` site), site irreps and sums of them as a returned row spells them, and
the site coreps of a spinful row. A shell or an irrep names a content and no
function. A shorthand given to all the partners its site group turns it into
names the **multiplet** they make, the orbitals of a site that the
operations, as the Hamiltonian carries them, mix among themselves: at a `3m`
site holding a `(p_x, p_y)` pair as orbitals 0 and 1, `{0: "px", 1: "py"}`
and `{(0, 1): "px"}` both state the doublet `E`. Given to one of them, a
shorthand names one direction
([below](#a-declaration-that-lowers-the-group)).

**A site's content, not its partition.** A declaration states what each site
holds; how the orbitals are partitioned inside the site is their basis. It is
read on the multiplets the Hamiltonian measures, which it never changes: words
naming exactly those multiplets are read one multiplet at a time, and any
other spelling as the one content the site's words add up to. A `p` shell
whose Wannier functions are turned inside a `3m` site is one measured
multiplet, and `["pz", "px", "py"]`, `{0: "pz", (1, 2): "E"}` and
`{(0, 1, 2): "p"}` all declare it, its rows reading the shell's content and its
matrices staying the Hamiltonian's own blocks. Where the hoppings keep the
shell's `p_z` apart from its pair, every spelling reads `A1`, `E`, `E`, each
row its own multiplet's, and the shell declared whole keeps its `A1 + E` as
each row's `declared_irrep`. A word on one orbital, at a site with several,
states part of what the site holds: `{1: "A1"}` on that shell says it holds
an `A1`, which its `p_z` is. A content the site does not hold under any
labelling the hoppings leave open lowers the group: `s` and
`d_{x²−y²}` declared on a `(p_x, p_y)` pair the hoppings bind at a `4mm` site
state `A1 + B1` where the pair holds the doublet `E`, and the fourfold axis
goes. Where nothing couples `p_x` to `p_y`, the hoppings carry the pair as
`A1 + B1` exactly as they carry it as `E`, and the same declaration keeps the
fourfold and says which the pair is: its `s` and its `d` are the sum and the
difference of the two orbitals as written, not either orbital alone.

## A declaration that lowers the group

A declaration can state a crystal with fewer operations than the cell allows,
in two ways, each on its own part of the record. Kane–Mele without stagger,
its Rashba term kept, with its two sublattices declared `A1` and `A2` of `3m`
(`A2` is even under `C3` and odd under the three mirrors), does both:

```python
km_r = kane_mele(0.0, -1.0, 0.3, 0.25)          # no stagger; the Rashba term breaks m(z)
lowered = tqc.realized_group(km_r, characters={0: "A1", 1: "A2"})
print(repr(lowered))
#  RealizedGroup(P31' #143.2; cell lowered by the declared content from #183.186 (12 of 24 coset representatives of the lattice translations excluded; these are the single-valued group's, each standing for itself and its negative in the double group); descended from #156.50 (6 of 12 broken by the declared content); on a substrate (the default), the free-standing group #143.2 agrees)
broke = lowered.descent
print(broke["from"]["number_str"], "->", broke["to"]["number_str"],
      sorted({b["name"] for b in broke["dropped"]}), {b["refuted_by"] for b in broke["dropped"]})
#  156.50 -> 143.2 ['m([010])', "m([010])'", 'm([1-10])', "m([1-10])'", 'm(x)', "m(x)'"] {'declaration'}
```

- **A reduction, before any hopping is read.** Two sites of one orbit declared
  different contents, the whole content each holds however its orbitals are
  partitioned, are different sites, as a different `typat` would make them,
  so the cell is identified without the operations that exchange the
  sublattices: *cell lowered by the declared content*. Nothing is charged to
  the Hamiltonian; the reduction is kept on `decoration["declared"]`. A
  layer's contents are compared in its own site groups, the horizontal mirror
  among them, in either environment: `s` and `p_z` are different sites on a
  substrate too, where each becomes `A1` of `3m`.
- **A descent, in the Hamiltonian.** Under each mirror that fixes the
  sublattices the hoppings carry orbital 1 with the sign of orbital 0, and
  `A1` beside `A2` states the opposite sign. Those mirrors are not symmetries
  of the declared crystal, and the group descends to the operations that are,
  each broken one on `descent` with `refuted_by = "declaration"`. A descent by
  declaration never refuses: its floor is `P1`, or `P11'` with time reversal.

A shorthand given to one orbital whose site group turns it into others names
one direction, and that is a reduction alone: of the operations of a `3m`
site, only the identity and the mirror whose normal is x carry `p_x` to
itself, the mirror as `−p_x`.

```python
px = tqc.realized_group(km_r, typat=[0, 1], characters={0: "px"})
print(repr(px), px.descent)
#  RealizedGroup(Cm1' #8.33; cell lowered by the declared content from #156.50 (8 of 12 coset representatives of the lattice translations excluded; these are the single-valued group's, each standing for itself and its negative in the double group); on a substrate (the default), the free-standing group #8.33 agrees) None
print(px.rep.characters.summary())
#  orbital  Wyckoff  site symmetry  function  irrep  named by  determined by  orbital characters
#        0       2a              m        px  -1E2E    tables    declaration  E=+1, m(x)=-1
#        1       2a              m            -1E2E    tables    hamiltonian  E=+1, m(x)=-1
#    orbital 0: declared 'px' (A''), read at the 3m in P3m11' #156.50; the declared content lowered the cell before any hopping was read and resolves as A'' of the m in Cm1' #8.33; returned as -1E2E of m   (.declarations)
```

Nothing was broken (`descent` is `None`): the declared crystal has only the
operations of `Cm1'`. `A''` is the irrep of the mirror group `m` that `p_x`
carries, odd under its mirror, and `-1E2E` the site corep it becomes with
spin. A spinor's character is zero on a mirror, so `A'` and `A''` both become
`-1E2E`, and it is `m(x)=-1` in the orbital characters column that says which.
The line under the table is where the printout sets the declared label beside
the returned one; each row keeps both, as `declared_irrep`, `declared_in` and
`reduced_in` beside `irrep`.

Without `typat=`, the two sublattices are one orbit and the declaration says
nothing about the second: an operation carrying the first site onto it carries
`p_x` onto whatever that site holds. Two such operations compose into a
threefold rotation, which turns `p_x` out of itself, so the operations that each
carry the declared content are not a group. The cell keeps the largest group
inside them, the first by
[the tie rule](../background/measurement.md#the-descent-and-the-tie-rule)
where several of that order tie, and the undeclared site carries what that
group carries onto it:

```python
orbit = tqc.realized_group(km_r, characters={0: "px"})
print(repr(orbit))
#  RealizedGroup(Cmm21' #35.166; cell lowered by the declared content from #183.186 (16 of 24 coset representatives of the lattice translations excluded; these are the single-valued group's, each standing for itself and its negative in the double group); on a substrate (the default), the free-standing group #35.166 agrees)
```

Of the six operations exchanging the sublattices, `Cmm21'` keeps the twofold
axis and one mirror, which carry `p_x` onto `±p_x` at the second site. The two
sixfolds and the other two mirrors are on `decoration["declared"]` beside the
four that carry `p_x` out of itself at its own site: the declared content is
carried by each on its own, and not by every product of it with the operations
kept.

The two sites sit at `4e` of `Cmm21'`, whose site group is the mirror `m`, and
the tables list no band representation there: the rows are named by the
`point group`, and the sum the orbit induces is written at the positions it can
be moved onto.

```python
br, = orbit.rep.characters.band_representations
print([row.named_by for row in orbit.rep.characters.orbitals], br.wyckoff, br.terms, br.alternatives)
#  ['point group', 'point group'] 4e ((2, '-E↑G(2)', "2a(mm21',mm2)"),) (((2, '-E↑G(2)', "2b(mm21',mm2)"),),)
```

Twice `-E` at `2a`, and as `alternatives` the same sum at `2b`, which the
stable real-space invariants do not distinguish from it; the summary of the
table writes both out on a line under the rows.

## When it refuses

A declaration in none of the forms above, such as a key naming an orbital the
model does not have, naming one twice or naming orbitals of two sites, raises
`ValueError` before anything is measured, and the message states the forms. A
well-formed declaration can
still say what its site cannot read, and two refusals say so. A
word the site group does not know is refused
[`unknown_label`](../reference/refusals.md#unknown_label). The group is
identified before the word is read, so the refusal names it, and it prints
every word the site accepts:

```python
refusal = tqc.realized_group(km, typat=[0, 1], characters={0: "B1"})
print("\n".join(str(refusal).splitlines()[:6]))     # its first lines; the table follows
#  Refusal[unknown_label]: The group identified is P3m11' #156.50, every operation of it obeyed; its representation was not built: orbital 0: 'B1' is not an irrep of this site group, a sum of its irreps, a shell or a known shorthand.
#    site group: order 6, classes E, 2C3(z), 3m(x)
#    valid labels: -E1, A1, A2, -1E2E + -E1, E
#    shells:       s, p, d, f
#    shorthands:   s, px, py, pz, dxy, dyz, dzx, dx2-y2, dz2
#    frame: #1 = E; #2 = C3(z); #3 = C3^-1(z); #4 = m([010]); #5 = m(x); #6 = m([1-10])
```

`3m` has no `B` irrep. With spin, the valid words name each row twice, as the
returned site corep and as the orbital factor's own name: `-E1` stands for
both `A1` and `A2`, and the `E` doublet becomes the sum `-1E2E + -E1`. The
printout goes on with the site group's character table, in the frame line's
operations: a row for each irrep by its orbital name, its characters, and the
site corep it becomes with spin.

A joint declaration narrower than the orbitals it names is refused
[`ambiguous_multiplet`](../reference/refusals.md#ambiguous_multiplet), as is a
shell given to fewer orbitals than hold it:

```python
print(tqc.realized_group(models.lin3g(), typat=[0, 0, 1, 1], characters={(0, 1): "A1"}))
#  Refusal[ambiguous_multiplet]: The group identified is P4mm1' #99.164, every operation of it obeyed; its representation was not built: orbitals (0, 1): the joint form names the content its orbitals hold together, and 'A1' is 1-dimensional against the 2 orbitals named, so it is not what they hold jointly; {0: 'A1', 1: 'A1'} is the spelling for 2 separate A1 orbitals.
#      context: declared_irrep, dim, group, multiplet, multiplets, orbitals, site
```

Where the hoppings are not covariant under a declaration, `realized_group`
descends. Asked not to descend
(`descend=False`), it refuses
[`declaration_breaks_group`](../reference/refusals.md#declaration_breaks_group)
and names the operations the declared content is covariant on. A doublet at
`1a` states the diagonal mirror group:

```python
wrong = tqc.realized_group(models.lin3g(), typat=[0, 0, 1, 1], characters={(0, 1): "E"},
                           descend=False)
print(wrong.gate, [tqc.operations(wrong.context["group"])[i].name
                   for i in wrong.context["subgroup"]])
#  declaration_breaks_group ['E', 'C2(z)', 'm([110])', 'm([1-10])', "E'", "C2(z)'", "m([110])'", "m([1-10])'"]
```

## Where next

- [How pythtqc reads a Hamiltonian](../background/measurement.md): why the
  hoppings cannot say what an orbital is, and the lowest-ℓ rule with its keys.
- [Records](../reference/records.md#the-atomic-limit-table): every field of
  the atomic-limit table.
- [Conventions](../background/conventions.md): operation names, Seitz symbols,
  the spellings of site irreps.
- [Spin](spin.md): a wannier90 spinor import, where a declaration names a pair
  of spin partners.
- Every argument of [`realized_group`](../reference/functions.md#realized_group).
