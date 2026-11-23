# How do I say what my orbitals are?

A pythtb orbital is given by its position, and carries no symmetry label. The
hoppings fix how the orbitals transform relative to one another, and not which
site irreps they carry
([What the hoppings fix](../background/measurement.md#what-the-hoppings-fix)).
The argument `typat=` of `realized_group` says which sites hold the same kind
of atom. Its argument `characters=` says which site irreps each site's orbitals
carry. A declaration is never checked against the Hamiltonian.
The Hamiltonian is measured against the crystal the declarations define. What
the declarations and the hoppings leave open is assigned by the
lowest-angular-momentum rule.

## The call

`typat=` takes one entry per orbital, the same entry for orbitals on atoms of
one species. `characters=` takes a word for an orbital, here in a mapping from
orbital 0 to `"s"`.
[The forms a declaration takes](#the-forms-a-declaration-takes) gives the
others.

The example is Kane-Mele, from `pythtb.models`.
`kane_mele(delta, t, soc, rashba)` builds a model with one orbital with spin on
each sublattice of the honeycomb, with on-site energies `±delta`, hopping `t`,
intrinsic spin-orbit coupling `soc` and Rashba coupling `rashba`. The example
declares the two sublattices different atoms, and orbital 0 an `s` orbital:

```python
import pythtqc as tqc
from pythtb.models import kane_mele
from pythtqc import models

km = kane_mele(1.0, -1.0, 0.3, 0.25)            # delta, t, soc, rashba: two orbitals, with spin
r = tqc.realized_group(km, typat=[0, 1], characters={0: "s"})
print(r.group.name, r.group.number_str)
#  P3m11' 156.50
print(r.wyckoff_orbits.summary())
#  Wyckoff  site symmetry  orbitals  label  determined by  declared
#       1c           3m1'         0    -E1    declaration         s
#       1b           3m1'         1    -E1    hamiltonian
```

## How to read it

- `P3m11' 156.50`: the group `realized_group` measured on the Hamiltonian, by
  name and number ([What group does my Hamiltonian have?](group.md)). It is
  space group 156 with time reversal adjoined, the trailing `1'`. The two
  sublattices are declared different atoms, so the cell's group is 156, and no
  operation exchanges them. The stagger `±delta` alone would break the
  exchange too. The layer is read on a substrate, the default for a layer.
  Read free-standing it has the same group, since the Rashba term breaks the
  horizontal mirror
  ([Is my 2D model on a substrate or free-standing?](layers.md)).
- `Wyckoff  site symmetry  orbitals ...`: the model's atomic-limit table,
  `r.wyckoff_orbits`, one row per Wyckoff orbit of orbitals, the orbitals the
  operations map into one another. Orbitals on one site that no operation maps
  into one another are rows of their own, as at lin3g's `1a` below. Each column
  is a field of the row.
- `1c` and `1b` under `Wyckoff`: each orbit's Wyckoff position, multiplicity
  and letter, `wyckoff`. It is written in the conventional cell used by the
  irreptables package.
- `3m1'` under `site symmetry`: the site-symmetry group, `site_symmetry`, its
  `1'` marking time reversal. `unitary_site_symmetry` holds the group without
  time reversal, `3m`.
- `0` and `1` under `orbitals`: the model's orbitals on the orbit's sites,
  `orbitals`.
- `-E1` under `label`: the site irrep the orbitals of one site carry, `label`.
  With spin it is double-valued, marked by its leading `-`
  ([Little groups, irreps and coreps](../background/tqc-primer.md#little-groups-irreps-and-coreps)).
  `-E1` of `3m` is `A1`, the site irrep of an `s` orbital, or `A2`, times the
  spinor. Where a site's orbitals carry several site irreps, the label is their
  sum. A declaration names either the orbital factor, `s` or `A1`, or the
  double-valued site irrep, `-E1`.
- `declaration` and `hamiltonian` under `determined by`: how the site irrep is
  known, `determined_by`. It takes four values.
    - `declaration`: `characters=` stated it, and the representation returned
      gives the orbitals that site irrep.
    - `hamiltonian`: nothing was declared for the orbital, and the hoppings and
      the declarations leave it one site irrep. Orbital 1 reads `hamiltonian`.
      The hoppings fix its site irrep once orbital 0's is fixed, so the one
      declaration fixes both. A declaration bears in this way on every orbital
      the hoppings couple to the one declared.
    - `lowest-l`: nothing was declared, the hoppings leave several site irreps
      possible, and the lowest-angular-momentum rule assigned one
      ([When to declare](#when-to-declare)).
    - `declaration not carried`: `characters=` gave generator matrices on
      orbitals no hopping couples to one another. A labeling the hoppings
      leave open ([When to declare](#when-to-declare)) gives the orbitals the
      site irrep of those matrices, and none gives them the matrices. The
      group keeps its operations, and the matrices returned are the hoppings'
      own. `label` is what the orbitals carry under
      the labeling returned, and a line under the table names what was
      declared.
- `s` under `declared`: what `characters=` declared for the row, as written,
  `declared`. It is blank where nothing was declared. Where a row's orbitals
  were declared differently it holds one entry per orbital, `-` for an orbital
  not declared.

Under its rows the table prints a line for each of these, where it occurs:

- an orbit whose band representation is not its own site irreps at its own
  position;
- a pseudospin the Hamiltonian conserves, identical copies of a part of the
  crystal that no hopping joins, with the site irreps assigned its copies;
- a declaration on a row that reads `declaration not carried`;
- a declaration written in a name that Mulliken's convention gives to another
  irrep of the site group than the irreptables package does
  ([Site irreps](../background/conventions.md#site-irreps));
- where a row reads `lowest-l`, one sentence saying what that means.

A row also holds the characters its orbitals carry and the band representation
its orbit induces:

```python
o = r.wyckoff_orbits[0]
mirror = next(i for i, op in r.operations.items() if op.name == "m(x)")
print(repr(o.declared), o.numerical.named_by, o.numerical.characters[0][mirror].real)
#  's' tables 1.0
for o in r.wyckoff_orbits:
    print(o.orbitals, o.wyckoff, o.band_representation.terms)
#  (0,) 1c (Term(multiplicity=1, irrep='-E1', wyckoff='1c', site_symmetry="3m1'", unitary_site_symmetry='3m'),)
#  (1,) 1b (Term(multiplicity=1, irrep='-E1', wyckoff='1b', site_symmetry="3m1'", unitary_site_symmetry='3m'),)
```

- `'s'`: `o.declared`, the word declared for orbital 0. It is `None` on
  orbital 1.
- `tables`: `o.numerical.named_by`, where the row's site irrep takes its name.
  It takes five values.
    - `tables`: the tables of elementary band representations in the
      irreptables package list the band representation the row induces at its
      Wyckoff position. The row takes the name of the site irrep there.
    - `invariants`: without spin, where those tables list no band
      representation at the position, the tables of the stable real-space
      invariants single out the site irrep. The row takes their name.
    - `point group`: neither names the row. It takes the name the irreptables
      package gives that irrep at the origin of a symmorphic space group with
      the site's point group, so that one irrep has one name at every site.
    - `Mulliken`: without spin, where no single point-group name fits the
      row. That happens where the site's operations are not those of a
      symmorphic origin, and the rotations that take the origin's operations
      onto them name the row differently, as for the two irreps of a
      complex-conjugate pair in a group without time reversal. The row then
      takes its Mulliken symbol, read off the site's operations as Cartesian
      matrices.
    - `unnamed`: nothing names the row. `label` is blank, and
      `numerical.characters` alone states the site irrep.
- `1.0`: the character of orbital 0 on `m(x)`, the mirror whose normal is x,
  in the Cartesian axes in which the pythtb Lattice is specified.
  `o.numerical.characters` holds, for each orbital of the row, its character on
  each operation of the site group, a complex number, keyed by the operation's
  index. It is the character of the orbital factor, single-valued with or
  without spin. An `s` orbital is even under `m(x)`, so its character is `+1`.
- `(Term(multiplicity=1, irrep='-E1', wyckoff='1c', ...),)`:
  `o.band_representation.terms`, the band representation the row's orbitals
  induce ([`BandRepresentation`](../reference/records.md#bandrepresentation)).
  Each term holds a multiplicity, a site irrep, its Wyckoff position, and its
  site-symmetry group with and without time reversal. Each sublattice induces
  the one `-E1` at its own position, so the table prints no line under its
  rows for either.

Every field of a row is in
[Returned records](../reference/records.md#wyckofforbit). Operation
names and the spellings of site irreps are in
[Conventions: operations, labels and band representations](../background/conventions.md).

## When to declare

The hoppings fix the site irreps of the orbitals they couple only up to a
common one-dimensional character of the space group. Each coupled part of the
crystal, the orbitals the hoppings connect directly or through others, has its
own. A choice of those characters is a labeling
([What the hoppings fix](../background/measurement.md#what-the-hoppings-fix)).
Where the labelings left open give an orbital different site irreps and nothing
is declared, the lowest-angular-momentum rule takes the one carried by the
lowest-angular-momentum functions
([The lowest-angular-momentum rule](../background/measurement.md#the-lowest-angular-momentum-rule)).
Such a row reads `lowest-l`. Its label is a choice the hoppings cannot make, and
a declaration states it instead.

The example is Kane-Mele with no stagger and no Rashba term. It is read on a
substrate, free-standing, and free-standing with its orbitals declared `p_z`.
Each line prints the group, the rows, and the symmetry data vector of the two
lowest bands, which `hsp_table` returns
([Which little-group representations do these bands carry?](labels.md)). `GM`
is the tables' name for Γ.

```python
km0 = kane_mele(0.0, -1.0, 0.3, 0.0)            # delta = 0, rashba = 0
sub = tqc.realized_group(km0, typat=[0, 0])                                 # on a substrate
free = tqc.realized_group(km0, typat=[0, 0], environment="free-standing")
pz = tqc.realized_group(km0, typat=[0, 0], environment="free-standing", characters={0: "pz"})
for rec in (sub, free, pz):
    rows = rec.wyckoff_orbits
    print(rec.group.number_str, [o.label for o in rows], [o.determined_by for o in rows],
          tqc.hsp_table(rec, bands=2).symmetry_data)
#  183.186 ['-E1'] ['hamiltonian'] {'-GM9': 1, '-K6': 1, '-M5': 1}
#  191.234 ['-E1'] ['lowest-l'] {'-GM9': 1, '-K8': 1, '-M6': 1}
#  191.234 ['-E2'] ['declaration'] {'-GM12': 1, '-K9': 1, '-M5': 1}
```

- `183.186 ['-E1'] ['hamiltonian']`: on a substrate the group is `P6mm1'`, and
  the site group is `3m`. The hoppings leave the orbital factor open between
  `A1` and `A2`. With spin both become `-E1`, so the hoppings leave one site
  irrep, and the row reads `hamiltonian`. No mirror of `3m` lies in the layer's
  plane, so an `s` and a `p_z` orbital are both `A1`. Declaring `pz` there
  changes no label.
- `191.234 ['-E1'] ['lowest-l']`: free-standing, the group is `P6/mmm1'`, and
  the site group `-6m2` holds the horizontal mirror, which tells an `s` orbital
  from a `p_z` orbital. The row reads `lowest-l`, and the rule takes `s`.
- `191.234 ['-E2'] ['declaration']`: declared `p_z`, the orbital of a carbon
  honeycomb, the row is `-E2`. The two lowest bands carry other coreps at every
  high-symmetry point.

The symmetry-indicator class of the two lowest bands changes with the
declaration
([Which symmetry-indicator class do these bands occupy?](indicators.md)):

```python
for rec in (free, pz):
    si = tqc.symmetry_indicator(tqc.hsp_table(rec, bands=2))
    print(si.indicator_group, si.values, si.order, {name: si.indicators[name] for name in ("z2w,3", "z4")})
#  Z6 x Z12 (1, 0) 6 {'z2w,3': 1, 'z4': 2}
#  Z6 x Z12 (5, 0) 6 {'z2w,3': 1, 'z4': 2}
```

- `Z6 x Z12`: the indicator group.
- `(1, 0)` and `(5, 0)`: the class read as `s`, and the class declared `p_z`,
  which is its inverse.
- `6`: the order of the class, the same in both.
- `{'z2w,3': 1, 'z4': 2}`: the two indicators whose values are their own
  inverses, and so the same in both. The other indicators change
  ([Named indicators](indicators.md#named-indicators)).

A statement made in the order or in those two indicators holds for both
readings. The class itself needs the orbitals stated. An orbital the hoppings
couple to no other orbital needs a declaration most. Nothing measured relates
it to the rest, so its declaration breaks no operation and is the only
statement of what it is.

## The forms a declaration takes

The example is lin3g, from `pythtqc.models`, the spinful square layer of Lin
*et al.*,
[Nat. Commun. **15**, 550 (2024)](https://doi.org/10.1038/s41467-024-44762-w):
four orbitals with spin, eight bands. Orbitals 0 and 1 share the site `1a`,
and orbitals 2 and 3 sit on the two sites of `2c`. The example declares an `s`
and a `dx2-y2` orbital at `1a`, and an `s` orbital on each `2c` site:

```python
lin = tqc.realized_group(models.lin3g(), typat=[0, 0, 1, 1],
                         characters=["s", "dx2-y2", "s", "s"])
print(lin.wyckoff_orbits.summary())
#  Wyckoff  site symmetry  orbitals  label  determined by  declared
#       1a          4mm1'         0    -E1    declaration         s
#       1a          4mm1'         1    -E2    declaration    dx2-y2
#       2c          mm21'       2-3     -E    declaration         s
```

A declaration can be written in several forms, and the same call reads each.
Each line below declares lin3g's orbitals again in another form:

```python
def irreps(characters):                         # lin3g, its orbitals declared again
    rec = tqc.realized_group(models.lin3g(), typat=[0, 0, 1, 1], characters=characters)
    return tuple(o.label for o in rec.wyckoff_orbits)

print(irreps(["s", "dx2-y2", "s", "s"]))        # one word per orbital
#  ('-E1', '-E2', '-E')
print(irreps(["A1", "B1", None, None]))         # site irreps; None claims nothing
#  ('-E1', '-E2', '-E')
print(irreps(["-E1", "-E2", None, None]))       # double-valued site irreps
#  ('-E1', '-E2', '-E')
print(irreps({1: "dx2-y2"}))                    # a mapping: only the orbitals named
#  ('-E1', '-E2', '-E')
print(irreps({(0, 1): "A1 + B1"}))              # what two orbitals hold together
#  ('-E1', '-E2', '-E')
print(irreps({0: {"C4(z)": -1, "m(x)": +1}}))   # characters on one operation per class
#  ('-E2', '-E1', '-E')
print(irreps({0: {3: 1, 6: 1}}))                # the operations by their index: A1 again
#  ('-E1', '-E2', '-E')
print(lin.operations[3].name, lin.operations[6].name)
#  C4(z) m(x)
```

- **A list**, one entry per orbital: a word, or `None` where nothing is
  claimed.
- **A double-valued site irrep**, with spin. `-E1` of `4mm` is `A1` or `A2`
  times the spinor, and the lowest-angular-momentum rule chooses `A1`.
- **A mapping** from orbital to word, for the orbitals named. `{1: "dx2-y2"}`
  fixes orbital 0 too, through the hoppings, and orbital 0 reads
  `hamiltonian`.
- **A key of several orbitals**, for the site irreps the orbitals of one site
  hold together. `{(0, 1): "A1 + B1"}` states what `1a` holds, as one sum. It
  is read on the site's multiplets. A multiplet of a site is a smallest set of
  its orbitals that the operations, as the hoppings represent them, mix among
  themselves and not with the rest. Each row names what its own multiplet
  carries, `-E1` and `-E2`, as for the list. The sum is the same in every
  basis of the site's orbitals, so this form declares Wannier functions that
  mix the site's functions, where no orbital is one function such as `px`.
- **Generator characters**, a mapping from operation to character. Each
  operation is named as `tqc.operations` prints it, or given by its index there
  ([Operations: a Seitz symbol and a Cartesian name](../background/conventions.md#operations-a-seitz-symbol-and-a-cartesian-name)).
  An operation stands for its whole conjugacy class. `C4(z) = -1` with
  `m(x) = +1` is `B1`, a `dx2-y2` orbital, on orbital 0. Orbital 1, which the
  hoppings fix relative to orbital 0, moves with it. No irrep is named, so the
  declaration does not rest on how a source orients `4mm`. Here `B1` is the
  irrep even under `m(x)`, the mirror whose normal is x. A source that takes
  the diagonal mirrors as its reference calls the same function `B2`.
- **Generator matrices**, a mapping from operation to the matrix it acts with
  on the orbitals of the key, in the order the key writes them. They say which
  basis of its site irrep a multiplet is
  ([Which basis a doublet is](#which-basis-a-doublet-is)).

With spin, a declaration that orients a site irrep the other way can leave
every label unchanged. `B1` and `B2` of `4mm` differ only on the mirrors, where
a spinor's character is zero, so both become `-E2`. lin3g with orbital 1
declared `B2`, where the list declares it `dx2-y2`, returns the same site
irreps and the same symmetry data vector of its six lowest bands. Whatever the
form, what was stated shows in the `declared` column and in the rows'
`numerical.characters`:

```python
b2 = tqc.realized_group(models.lin3g(), typat=[0, 0, 1, 1], characters={1: "B2"})
print(tuple(o.label for o in b2.wyckoff_orbits),
      tqc.hsp_table(b2, bands=6).symmetry_data == tqc.hsp_table(lin, bands=6).symmetry_data)
#  ('-E1', '-E2', '-E') True
print(b2.wyckoff_orbits.summary().splitlines()[2])
#       1a          4mm1'         1    -E2    declaration        B2
mirror = next(i for i, op in b2.operations.items() if op.name == "m(x)")
print([round(rec.wyckoff_orbits[1].numerical.characters[1][mirror].real) for rec in (lin, b2)])
#  [1, -1]
```

- `[1, -1]`: orbital 1's character on `m(x)`, declared `dx2-y2` and declared
  `B2`.

The words of a declaration are of four kinds:

- function shorthands, `s`, `px`, `py`, `pz`, `dxy`, `dyz`, `dzx`, `dx2-y2`
  and `dz2`;
- shells by their angular momentum, `s`, `p`, `d` and `f`, each the site irreps
  the real harmonics of that angular momentum carry at the site: `A1 + E` for
  `p` at a `3m` site;
- site irreps, and sums of them, spelled as a returned row spells them;
- with spin, the double-valued site irreps of a row.

A shell or a site irrep names what a site holds, and no function. A shorthand
names one function. Where it and every function the site group turns it into
are each given to an orbital of the site, together they state the site irrep
of the multiplet those orbitals make. A shorthand given to a key of several
orbitals states the whole site irrep its function belongs to. At a `4mm` site
holding a `(p_x, p_y)` pair as orbitals 0 and 1, `{0: "px", 1: "py"}` and
`{(0, 1): "px"}` both state the doublet `E`. Given to one orbital alone, a
shorthand names one direction
([A declaration that lowers the group](#a-declaration-that-lowers-the-group)).

**What a site holds, not how its orbitals divide it.** A declaration states the
site irreps each site holds. How the orbitals divide them inside the site is a
basis. A declaration is read on the site's multiplets, and never changes them.
Words that name exactly those multiplets are read one multiplet at a time. Any
other spelling is read as the sum the site's words add up to.

A `p` shell whose Wannier functions are rotated within a `3m` site is one
multiplet. `["pz", "px", "py"]`, `{0: "pz", (1, 2): "E"}` and
`{(0, 1, 2): "p"}` all declare it, and its row reads `A1 + E`. Where the
hoppings keep the shell's `p_z` apart from its pair, every spelling gives two
rows, `A1` and `E`, each its own multiplet's. The shell declared whole keeps
`A1 + E` as each orbital's `numerical.declared_irrep`. A word on one orbital,
at a site with several, states part of what the site holds. `{1: "A1"}` on that
shell says the site holds an `A1`, whichever orbital the key names; here that
`A1` is the shell's `p_z`.

A declaration of site irreps that the site holds under no labeling the hoppings
leave open lowers the group. Take a `(p_x, p_y)` pair at a `4mm` site, with
hoppings that couple `p_x` to `p_y`, as the diagonal hoppings of the pair in
the next section do. `s` and `dx2-y2` declared on it state `A1 + B1`, where the
pair holds the doublet `E`. The fourfold axis and the mirrors `m(x)` and `m(y)`
are then not symmetries of the declared crystal. Where nothing couples `p_x` to
`p_y`, the labelings the hoppings leave open give the pair `A1 + B1` as well as
`E`. There the same declaration keeps every operation and says which basis the
two orbitals are. Each is even under `m(x)` and `m(y)`, the fourfold axis
exchanges them, and their sum is the `s` and their difference the `dx2-y2`.

## Which basis a doublet is

Where every labeling the hoppings leave open gives a site the same site irrep,
the coreps the bands carry can still rest on a choice. `E` of `4mm` times any
one-dimensional character of the group is `E` again. The example is a
`(p_x, p_y)` pair on a square lattice, built in pythtb, with `t_σ = 1` and
`t_π = 0.3` to the nearest neighbors and `t_σ = 0.2` along the diagonals:

```python
import numpy as np
from pythtb import Lattice, TBModel

pair = TBModel(Lattice([[1.0, 0.0], [0.0, 1.0]], [[0.0, 0.0]] * 2, periodic_dirs=[0, 1]))
for bond, hop in (([1, 0], [[1.0, 0.0], [0.0, 0.3]]), ([0, 1], [[0.3, 0.0], [0.0, 1.0]]),
                  ([1, 1], [[0.1, 0.1], [0.1, 0.1]]), ([1, -1], [[0.1, -0.1], [-0.1, 0.1]])):
    for a in range(2):
        for b in range(2):
            if hop[a][b]:
                pair.set_hop(hop[a][b], a, b, bond, mode="add")
bare = tqc.realized_group(pair)
print(tuple(o.label for o in bare.wyckoff_orbits), [o.determined_by for o in bare.wyckoff_orbits])
#  ('E',) ['hamiltonian']
print(tqc.hsp_table(bare, bands=1).note)
#  another labeling the hoppings leave open keeps every site irrep and labels some energy multiplets of these bands differently; it keeps which bands are degenerate and the labels of all the model's bands together
```

- `('E',) ['hamiltonian']`: the pair's one row, `E`. Every labeling gives it
  `E`, so it reads `hamiltonian`.
- `another labeling the hoppings leave open ...`: the note of the lowest band's
  table. Another labeling keeps `E` and gives the band another corep at some
  point, so the band's coreps rest on a choice
  ([Which little-group representations do these bands carry?](labels.md)).

Words and generator characters are read through characters, which are the same
in every basis, so they leave this choice open. Generator matrices make it. They
are given on operations of the multiplet's site group, which a row's
`numerical.characters` lists by index:

```python
o = bare.wyckoff_orbits[0]
ops = bare.operations
for i in o.numerical.characters[0]:
    print(i, ops[i].name, ops[i].seitz)
#  1 E {1|0,0,0}
#  2 C2(z) {2_001|0,0,0}
#  3 C4(z) {4+_001|0,0,0}
#  4 C4^-1(z) {4-_001|0,0,0}
#  5 m(y) {m_010|0,0,0}
#  6 m(x) {m_100|0,0,0}
#  7 m([110]) {m_110|0,0,0}
#  8 m([1-10]) {m_1-10|0,0,0}
```

At Γ, `rec.rep.sewing_operator(i, [0, 0])` is the matrix the returned
representation gives operation `i` on the pair. The example declares the
pair's own matrices, and then the same with `m(x)` even on `p_x`:

```python
pxpy = {"C4(z)": [[0, -1], [1, 0]], "m(x)": [[-1, 0], [0, 1]]}
for matrices in (pxpy, {**pxpy, "m(x)": [[1, 0], [0, -1]]}):
    rec = tqc.realized_group(pair, characters={(0, 1): matrices})
    mirror = next(i for i, op in rec.operations.items() if op.name == "m(x)")
    print(tuple(o.label for o in rec.wyckoff_orbits), rec.wyckoff_orbits[0].determined_by,
          (np.real(rec.rep.sewing_operator(mirror, [0, 0])) + 0).tolist(),
          tqc.hsp_table(rec, bands=1).symmetry_data)
#  ('E',) declaration [[-1.0, 0.0], [0.0, 1.0]] {'X3': 1}
#  ('E',) declaration [[1.0, 0.0], [0.0, -1.0]] {'X4': 1}
```

- `[[-1.0, 0.0], [0.0, 1.0]] {'X3': 1}`: the `p` pair's own matrices.
  `C4(z)` maps `p_x` onto `p_y` and `p_y` onto `-p_x`, and `m(x)` turns
  `p_x` over. The lowest band carries `X3` at `X`.
- `[[1.0, 0.0], [0.0, -1.0]] {'X4': 1}`: the pair times `A2`. The site irrep is
  the same, the matrix of every mirror is the opposite, and the lowest band
  carries the other corep at `X`, `X4`.

Column `j` of a matrix is the image of orbital `j`, in the order the key writes
the orbitals. With spin, a matrix is the orbital factor's. Matrices are read
only where the key holds whole multiplets; on part of one they are refused with
`ambiguous_multiplet` ([When it refuses](#when-it-refuses)). They need not be
given on the whole site group. On the operations the declared ones generate,
the matrices returned are the declared ones. On the rest they are the
hoppings' own, under a labeling that gives the declared ones. Take matrices
whose site irrep a labeling gives the key's orbitals, in a basis no labeling
gives them. Where the hoppings couple those orbitals to one another, the
matrices lower the group
([A declaration that lowers the group](#a-declaration-that-lowers-the-group)).
Where no hopping couples them to one another, the group is kept and the row
reads `declaration not carried`.

## A declaration that lowers the group

A declaration can state a crystal with fewer operations than the cell allows,
in two ways: a reduction, recorded in `decoration.declared`, and a descent,
recorded in `descent`, both fields of the record `realized_group` returns. The
example is Kane-Mele without stagger, its Rashba term kept, with its two
sublattices declared `A1` and `A2` of `3m`. `A2` is even under `C3` and odd
under the three mirrors. The declaration lowers the group both ways:

```python
km_r = kane_mele(0.0, -1.0, 0.3, 0.25)          # no stagger; the Rashba term breaks m(z)
lowered = tqc.realized_group(km_r, characters={0: "A1", 1: "A2"})
print(repr(lowered))
#  RealizedGroup(P31' #143.2; cell lowered by the declared content from P6mm1' #183.186 (12 of 24 coset representatives of the lattice translations excluded; these are the single-valued group's, each standing for itself and its negative in the double group); descended from P3m11' #156.50 (6 of 12 broken by the declared content); on a substrate (the default), the free-standing group P31' #143.2 agrees)
cut = lowered.decoration.declared
print(cut.cell_group.number_str, "->", cut.subgroup.number_str, sorted({e.name for e in cut.excluded}))
#  183.186 -> 156.50 ['C2(z)', "C2(z)'", 'C6(z)', "C6(z)'", 'C6^-1(z)', "C6^-1(z)'", 'm([110])', "m([110])'", 'm([2-10])', "m([2-10])'", 'm(y)', "m(y)'"]
broke = lowered.descent
print(broke.cell_group.number_str, "->", broke.subgroup.number_str,
      sorted({b.name for b in broke.dropped}), {b.refuted_by for b in broke.dropped})
#  156.50 -> 143.2 ['m([010])', "m([010])'", 'm([1-10])', "m([1-10])'", 'm(x)', "m(x)'"] {'declaration'}
print(lowered.wyckoff_orbits.summary())
#  Wyckoff  site symmetry  orbitals  label  determined by  declared
#       1c            31'         0  -1E2E    declaration        A1
#       1b            31'         1  -1E2E    declaration        A2
```

- **A reduction, printed `cell lowered by the declared content`.** Two sites of
  one orbit declared to hold different site irreps are different sites, as a
  different `typat` would make them. What each site holds is compared whole,
  however its orbitals divide it. The cell is identified without the operations
  that exchange the two sites, before any hopping is read. No operation is
  recorded as broken by the hoppings. `decoration.declared` records the
  reduction: the cell's group, the group it is lowered to, and each operation
  excluded. The sites of a layer are compared in site groups that keep the
  horizontal mirror, in either environment. So an `s` orbital on one sublattice
  and a `p_z` on the other make two different sites on a substrate too, where
  both are `A1` of `3m`.
- **A descent, printed `descended from P3m11' #156.50`.** Under each mirror that
  maps each sublattice to itself, the hoppings give orbital 1 the same sign as
  orbital 0, and `A1` beside `A2` states opposite signs. Those mirrors are not
  symmetries of the declared crystal, and the group descends to the operations
  that are. Each broken one is on `descent` with `refuted_by = "declaration"`.
  A descent by declaration never refuses: it ends at `P1` at the least, or
  `P11'` with time reversal.
- **`-1E2E` under `label`, on both rows.** The site-symmetry group is now
  `31'`, whose unitary part `3` does not tell `A1` from `A2`: both restrict to
  its `A`, and with spin both rows carry `-1E2E`. What each declaration
  resolved to, the groups it was read and resolved in, and the operations it
  broke are on the row's `numerical`: `declared_irrep`, `declared_in`,
  `reduced_in` and `declaration_broke`.

A shorthand given to one orbital, at a site whose group turns it into others,
names one direction, and lowers the cell by a reduction alone. Of the
operations of a `3m` site, only the identity and the mirror whose normal is x
map `p_x` onto `±p_x`, the mirror onto `-p_x`. The example declares orbital 0
of the same model `p_x`, its sublattices declared different atoms:

```python
px = tqc.realized_group(km_r, typat=[0, 1], characters={0: "px"})
print(repr(px), px.descent)
#  RealizedGroup(Cm1' #8.33; cell lowered by the declared content from P3m11' #156.50 (8 of 12 coset representatives of the lattice translations excluded; these are the single-valued group's, each standing for itself and its negative in the double group); on a substrate (the default), the free-standing group Cm1' #8.33 agrees) None
print(px.wyckoff_orbits.summary())
#  Wyckoff  site symmetry  orbitals  label  determined by  declared
#       2a            m1'         0  -1E2E    declaration        px
#       2a            m1'         1  -1E2E    hamiltonian
o = px.wyckoff_orbits[0]
print(o.numerical.declared_irrep, o.numerical.declared_in, o.numerical.reduced_in)
#  {0: "A''"} {0: "3m in P3m11' #156.50"} {0: "m in Cm1' #8.33"}
```

- `None`: `px.descent`. Nothing was broken, and the declared crystal has only
  the operations of `Cm1'`.
- `-1E2E` under `label`: the double-valued site irrep of the mirror group `m`
  that `p_x` becomes with spin. `p_x` carries `A''`, odd under the mirror. A
  spinor's character is zero on a mirror, so `A'` and `A''` both become
  `-1E2E`. The character `-1` on `m(x)` in the row's `numerical.characters`
  says which.
- `{0: "A''"}`: `numerical.declared_irrep`, for each declared orbital of the
  row the site irrep its declaration resolved to. Each row keeps it beside the
  site irrep returned.
- `{0: "3m in P3m11' #156.50"}`: `numerical.declared_in`, the site group and
  the space group the declaration was read in, the cell's group before the
  declaration lowered it.
- `{0: "m in Cm1' #8.33"}`: `numerical.reduced_in`, the site group and the
  space group the declaration lowered the cell to. Its entry is an empty string
  where the declaration lowered nothing.

Without `typat=`, the two sublattices are one orbit, and the declaration says
nothing about the second site. An operation mapping the first site onto the
second maps `p_x` onto whatever that site holds. Two such operations compose
into a threefold rotation, which turns `p_x` out of itself. So the operations
that each keep the declaration do not form a group. The cell keeps the largest
group inside them, the first by the tie rule where several of that order tie
([The descent, and the tie rule](../background/measurement.md#the-descent-and-the-tie-rule)).
The undeclared site holds what the operations of that group map the declared
`p_x` onto:

```python
orbit = tqc.realized_group(km_r, characters={0: "px"})
print(repr(orbit))
#  RealizedGroup(Cmm21' #35.166; cell lowered by the declared content from P6mm1' #183.186 (16 of 24 coset representatives of the lattice translations excluded; these are the single-valued group's, each standing for itself and its negative in the double group); on a substrate (the default), the free-standing group Cmm21' #35.166 agrees)
```

Of the six operations that exchange the sublattices, `Cmm21'` keeps the
twofold axis and one mirror, which map `p_x` onto `±p_x` at the second site.
`decoration.declared` lists the two sixfold rotations and the other two
mirrors. Each of them alone keeps the declaration, since the second site is
not declared. Its product with an operation kept does not: `C6(z)` times the
twofold axis is a threefold rotation, which turns `p_x` out of itself at its
own site. Beside them it lists the four operations that turn `p_x` out of
itself at its own site.

The two sites sit at `4e` of `Cmm21'`, whose site group is the mirror `m`. The
tables list no band representation there, so the row is named by its
`point group`. The band representation the orbit induces is written at the
positions of higher symmetry its sites can be moved onto:

```python
o, = orbit.wyckoff_orbits
print(o.numerical.named_by, o.wyckoff, o.band_representation.terms, o.numerical.alternatives)
#  point group 4e (Term(multiplicity=2, irrep='-E', wyckoff='2a', site_symmetry="mm21'", unitary_site_symmetry='mm2'),) (BandRepresentation(2x-E at 2b (mm21')),)
```

- `point group 4e`: the row's `named_by`, and its Wyckoff position.
- `(Term(multiplicity=2, irrep='-E', wyckoff='2a', ...),)`: the band
  representation the orbit induces, twice `-E` at `2a`.
- `(BandRepresentation(2x-E at 2b (mm21')),)`: `numerical.alternatives`, the
  same sum at `2b`, which no stable real-space invariant tells apart from it.
  The table's summary writes both out on lines under the rows.

## When it refuses

A declaration in none of the forms above raises `ValueError` before anything is
measured, and the message states the forms. A key that names an orbital the
model does not have, names one orbital twice, or names orbitals of two sites is
such a declaration.

A declaration in one of the forms can still name what its site group does not
have, and two refusals say so. A word the site group does not know is refused
with [`unknown_label`](../reference/refusals.md#unknown_label). The group is
identified before the word is read, so the refusal names it. The refusal prints
every word the site accepts. The example declares orbital 0 of Kane-Mele `B1`,
which `3m` does not have:

```python
refusal = tqc.realized_group(km, typat=[0, 1], characters={0: "B1"})
print("\n".join(str(refusal).splitlines()[:6]))     # its first lines; the table follows
#  Refusal[unknown_label]: the group identified is P3m11' #156.50, every operation of it obeyed; orbital 0: 'B1' is not an irrep of this site group, a sum of its irreps, a shell or a shorthand. Declare one of the words listed below.
#    site group: order 6, classes E, 2C3(z), 3m(x)
#    valid labels: -E1, A1, A2, -1E2E + -E1, E
#    shells:       s, p, d, f
#    shorthands:   s, px, py, pz, dxy, dyz, dzx, dx2-y2, dz2
#    frame: #1 = E; #2 = C3(z); #3 = C3^-1(z); #4 = m([010]); #5 = m(x); #6 = m([1-10])
```

- `valid labels: -E1, A1, A2, -1E2E + -E1, E`: with spin, each irrep of the
  site group named twice, as the double-valued site irrep returned and as the
  orbital factor's own name. `-E1` stands for both `A1` and `A2`, and the
  doublet `E` becomes the sum `-1E2E + -E1`.
- `shells:` and `shorthands:`: the other words the site reads.
- `frame:`: the operations of the site group by index, as generator characters
  and generator matrices name them.

The printout goes on with the site group's character table by class: a row for
each irrep under its orbital name, its characters, and the double-valued site
irrep it becomes with spin. Generator matrices that are not unitary, or that
are not a representation of the site group, are refused with `unknown_label`
too.

A key of several orbitals whose word does not state as many dimensions as the
key has orbitals is refused with
[`ambiguous_multiplet`](../reference/refusals.md#ambiguous_multiplet). So are a
shell given to fewer orbitals than hold it, and generator matrices on part of a
multiplet. The example declares lin3g's two orbitals at `1a` together as `A1`:

```python
print(tqc.realized_group(models.lin3g(), typat=[0, 0, 1, 1], characters={(0, 1): "A1"}))
#  Refusal[ambiguous_multiplet]: the group identified is P4mm1' #99.164, every operation of it obeyed; orbitals (0, 1): 'A1' is 1-dimensional, and the key names 2 orbitals. Write {0: 'A1', 1: 'A1'} for 2 separate A1 orbitals.
#      context: declared_irrep, dim, group, multiplet, multiplets, orbitals, site
```

Where the hoppings are not covariant under a declaration, `realized_group`
descends and does not refuse
([A declaration that lowers the group](#a-declaration-that-lowers-the-group)).

## Where to go next

- [How pythTQC reads a Hamiltonian](../background/measurement.md) says why the
  hoppings cannot say what an orbital is, and gives the lowest-angular-momentum
  rule with the keys it reads where labelings tie.
- [Returned records](../reference/records.md#atomiclimittable) lists
  every field of the atomic-limit table.
- [Conventions: operations, labels and band representations](../background/conventions.md)
  gives operation names, Seitz symbols and the spellings of site irreps.
- [How does pythTQC read the spin of my model?](spin.md) reads a Wannier90
  spinor import, where a declaration names a pair of spin partners.
- [Functions](../reference/functions.md) gives every argument of
  [`realized_group`](../reference/functions.md#realized_group).
