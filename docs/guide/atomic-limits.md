# Could these bands be an atomic limit?

## What elementary band representations are

An atomic limit is a set of bands whose states can be written as exponentially
localized, symmetric Wannier functions. The Wannier functions at each Wyckoff
position carry site irreps of its site-symmetry group. A site irrep at a Wyckoff
position induces a band representation. An elementary band representation (EBR)
is one that is not a sum of others. The corep multiplicities of a set of bands
at every high-symmetry point form its symmetry data vector. The symmetry data
vector of an atomic limit is a sum of the symmetry data vectors of EBRs, with
nonnegative integer multiplicities. Such a sum is a decomposition. A fragile set
of bands is not an atomic limit, and becomes one once suitable atomic bands are
added.

This package uses two facts about EBRs. First, a set of bands whose symmetry
data vector has no decomposition is topological. Second, some topological sets
of bands have a decomposition too, fragile ones and stably topological ones
alike. So where `decompose` returns no decomposition the bands are topological,
and where it returns one they may be an atomic limit or topological.

The definitions and proofs are in Bradlyn *et al.*,
[*Topological quantum chemistry*, Nature **547**, 298 (2017)](https://doi.org/10.1038/nature23268),
and Cano *et al.*,
[*Building blocks of topological quantum chemistry: Elementary band representations*, Phys. Rev. B **97**, 035139 (2018)](https://doi.org/10.1103/PhysRevB.97.035139);
fragile topology is defined in Po, Watanabe and Vishwanath,
[*Fragile topology and Wannier obstructions*, Phys. Rev. Lett. **121**, 126402 (2018)](https://doi.org/10.1103/PhysRevLett.121.126402).
[Band representations and atomic limits](../background/tqc-primer.md#band-representations-and-atomic-limits),
in [The topological quantum chemistry this package uses](../background/tqc-primer.md),
gives a longer account.

## The call

`decompose` takes the table `hsp_table` returns for a set of bands. It returns
every decomposition of the table's symmetry data vector, up to `cap=` of them,
200 by default. A symmetry data vector written out by hand,
`{corep: multiplicity}`, is passed with what `realized_group` returned for the
model, `r`: `decompose(vector, r)`. Decompositions can only be returned if the
table holds the little-group representations at every high-symmetry point the
model samples, which for a layer are the points in its plane. A table that
refused at some point is refused with `incomplete_content`.

A layer is decomposed over the EBRs of the Wyckoff positions within it. On a
free-standing layer the group's positions off the layer are left out, and
`d.numerical.restriction.outside_layer` names them.

The example is lin3g, from `pythtqc.models`, the spinful square layer of Lin
*et al.*, [Nat. Commun. **15**, 550 (2024)](https://doi.org/10.1038/s41467-024-44762-w),
with two orbitals at Wyckoff `1a` and two at `2c`. Its six lowest bands are the
EBR of the `2c` orbitals and a fragile pair, and the example decomposes those
six bands:

```python
import pythtqc as tqc
from pythtb.models import kane_mele
from pythtqc import models

r = tqc.realized_group(models.lin3g(), typat=[0, 0, 1, 1])
t = tqc.hsp_table(r, bands=6)                      # the six lowest bands
d = tqc.decompose(t)
print(d.summary())
#  decomposable: 3 sums of elementary band representations carry this symmetry data vector   (.decomposable, .candidate_sums)
#    the coreps at the high-symmetry points: 2x-GM6 + -GM7 + -M6 + 2x-M7 + 3x-X5   (.symmetry_data)
#    every candidate sum, each elementary band representation written as the site irrep inducing it at its Wyckoff position (site symmetry)   (.candidate_sums)
#      -E2 at 1b (4mm1') + -E at 2c (mm21')
#      -E1 at 1a (4mm1') + -E2 at 1a (4mm1') + -E2 at 1b (4mm1')
#      -E1 at 1b (4mm1') + 2x-E2 at 1b (4mm1')
#    some site irreps were assigned by the lowest-angular-momentum rule: a different declared content renames them in these sums, while the Wyckoff positions stay   (.site_content)
print(d.numerical.summary())
#  A, R and Z lie out of the layer's plane: every corep there occurs in every elementary band representation as often as a measured one, so the answer is exact   (.restriction, .completeness)
def positions(d):                                  # the Wyckoff positions of each sum
    return [sorted(term.wyckoff for term in s for _ in range(int(term.multiplicity)))
            for s in d.candidate_sums]
print(positions(d))
#  [['1b', '2c'], ['1a', '1a', '1b'], ['1b', '1b', '1b']]
```

## How to read it

- `decomposable: 3 sums of elementary band representations carry this symmetry
  data vector`: whether the symmetry data vector has a decomposition,
  `d.decomposable`, and how many were found, `len(d.candidate_sums)`. Each line
  of `d.summary()` ends with the field that holds it.
- `the coreps at the high-symmetry points: 2x-GM6 + -GM7 + -M6 + 2x-M7 + 3x-X5`:
  the symmetry data vector decomposed, the same as `t.symmetry_data`. `GM` is the
  tables' name for Γ.
- `-E2 at 1b (4mm1') + -E at 2c (mm21')`: one decomposition, the orbitals an
  atomic limit with this symmetry data vector would be built from. Each term is
  an EBR, written as the site irrep that induces it, its Wyckoff position, and
  its site-symmetry group in parentheses. `4mm1'` is `4mm` with time reversal.
  Decompositions are listed by the number of EBRs they hold, fewest first. Two
  with as many are compared EBR by EBR, in the order of the tables of the
  irreptables package, and the one with more of the first EBR where they differ
  comes first.
- `some site irreps were assigned by the lowest-angular-momentum rule`: printed
  where an orbital's site irrep was assigned by that rule.
  [Positions survive every labeling; labels may not](#positions-survive-every-labeling-labels-may-not)
  shows what this changes.
- `A, R and Z lie out of the layer's plane`, from `d.numerical.summary()`: the
  points the decomposition was read without. A layer's `H(k)` has no `k_z`, so
  the coreps at `A`, `R` and `Z`, the points at `k_z = 1/2`, are not measured.
  In every EBR each of them occurs as often as a measured corep, so leaving them
  out loses no condition and the decomposition is exact.
  `d.numerical.restriction.copies`, printed below, pairs each corep left out
  with the measured one it repeats.

The three decompositions are three sums with one symmetry data vector. The first
two differ by `-E` at `2c` against `-E1` and `-E2` at `1a`. The last two differ
by that pair at `1a` against the same pair at `1b`.

`positions(d)` lists the Wyckoff positions of each decomposition, an entry for
each EBR counted with its multiplicity. `d.candidate_sums` holds each
decomposition as its terms. Each term holds a multiplicity, a site irrep, its
Wyckoff position, and its site-symmetry group with and without time reversal. A
decomposition's `label` writes it on one line, as the summary does.
[The band-representation key](../background/conventions.md#the-band-representation-key)
reads every part.

```python
print(tuple(d.candidate_sums[0]))
#  (Term(multiplicity=1, irrep='-E2', wyckoff='1b', site_symmetry="4mm1'", unitary_site_symmetry='4mm'), Term(multiplicity=1, irrep='-E', wyckoff='2c', site_symmetry="mm21'", unitary_site_symmetry='mm2'))
print(d.numerical.restriction.copies)
#  {'-A6': ('-M6',), '-A7': ('-M7',), '-R5': ('-X5',), '-Z6': ('-GM6',), '-Z7': ('-GM7',)}
```

lin3g is measured on a substrate, `realized_group`'s default for a 2D model,
where every position is kept
([Is my 2D model on a substrate or free-standing?](layers.md)).
[The group the class belongs to](indicators.md#the-group-the-class-belongs-to)
prints `outside_layer` for a free-standing layer.

## Positions survive every labeling; labels may not

The hoppings fix the site irreps of the orbitals relative to one another. A site
irrep they leave open, and that is not declared, is assigned by the
lowest-angular-momentum rule
([When a label is not trustworthy](labels.md#when-a-label-is-not-trustworthy)).
`d.site_content.determined_by` says for each orbital how its site irrep is
known: `hamiltonian`, fixed by the Hamiltonian; `declaration`, declared;
`lowest-l`, assigned by the lowest-angular-momentum rule. lin3g's two orbitals
at `1a` read `lowest-l`. Declaring orbital 0 a `dx2-y2` orbital with
`characters=` states a crystal the same hoppings describe, with the two site
irreps at `1a` exchanged:

```python
print(d.site_content.determined_by)
#  ('lowest-l', 'lowest-l', 'hamiltonian', 'hamiltonian')
other = tqc.realized_group(models.lin3g(), typat=[0, 0, 1, 1], characters={0: "dx2-y2"})
d_other = tqc.decompose(tqc.hsp_table(other, bands=6))
print(d_other.candidate_sums[0].label)
#  -E1 at 1b (4mm1') + -E at 2c (mm21')
print(positions(d_other) == positions(d))
#  True
```

The declaration changes the corep labels of the bands, and so the site irreps
in every decomposition. The first decomposition now reads `-E1` at `1b` where it
read `-E2`, and every decomposition sits at the same Wyckoff positions. Where a
site irrep reads `lowest-l`, the Wyckoff positions of each decomposition do not
depend on that choice. Its site irreps do. Under either set of names a
decomposition agrees with the band-representation tables of the Bilbao
Crystallographic Server. The one difference is at a point that is not
time-reversal invariant, where the label the server prints at `k` is the label
returned here at `−k`
([Reading across to BCS](../background/conventions.md#reading-across-to-bcs)).

## Counting bands

`regular_completion` takes a set of bands and `bands=`, the bands of a larger
system that contains them, as a number or as band indices. The record returned
holds `rc.padding`, the symmetry data vector the missing bands would carry if
the larger system were whole copies of the regular band representation.

One copy is the band representation induced from the general Wyckoff position.
It carries one state there, or the Kramers pair for a spinful model in a group
with time reversal. So one copy has `|U|` bands at every k-point, or `2|U|` for
the Kramers pair. `|U|` is the number of unitary operations of the group up to
lattice translations. For a spinful model they are counted in the single-valued
group.

A set of bands completed to whole regular copies carries the symmetry data
vector of an atomic limit. So where the padding has a decomposition, it names
atomic orbitals whose bands, added to the set, give the symmetry data vector of
an atomic limit.

A completion can only be returned if the number of bands is a multiple of the
number in one copy. Any other number is refused with `not_a_regular_multiple`.
A set of bands that carries more of a little-group representation than that
many regular copies is refused with `negative_multiplicity`.

lin3g's six lowest bands have a decomposition. Its group `P4mm1'` has
`|U| = 8`, and the model is spinful, so one regular copy has sixteen bands. The
example completes the six bands to sixteen:

```python
print(d.decomposable, len(t.bands))
#  True 6
rc = tqc.regular_completion(t, bands=16)
print(rc.bands_per_copy, rc.regular)
#  16 {'-GM6': 4, '-GM7': 4, '-M6': 4, '-M7': 4, '-X5': 8}
print(tqc.format_coreps(rc.padding), len(rc.decomposition.candidate_sums))
#  2x-GM6 + 3x-GM7 + 3x-M6 + 2x-M7 + 5x-X5 6
print(tqc.subtract(rc.padding, tqc.hsp_table(r, bands=[6, 7])) == tqc.hsp_table(r, bands=8).symmetry_data)
#  True
print(tqc.regular_completion(t, bands=8).gate)
#  not_a_regular_multiple
```

- `True 6`: whether the six bands have a decomposition, `d.decomposable`, and
  how many bands they hold, `len(t.bands)`.
- `16`: `rc.bands_per_copy`, `2|U|`, twice the eight operations of `4mm`.
- `{'-GM6': 4, ...}`: `rc.regular`, the symmetry data vector of one regular
  copy, computed from the characters of the little-group representations at
  each point.
- `2x-GM6 + 3x-GM7 + ...`: `rc.padding`, one regular copy less the symmetry
  data vector of the six bands.
- `6`: the number of decompositions of the padding, `rc.decomposition`, each a
  set of orbitals the missing bands would be.
- `True`: the padding is the symmetry data vector of bands 6 and 7, the two
  above the gap, and of the model's eight bands once more, since one regular
  copy is lin3g's eight bands twice.
- `not_a_regular_multiple`: eight bands, all of lin3g's, are not a multiple of
  sixteen.

## What a decomposition cannot certify

Every atomic limit has a decomposition, and some topological sets of bands have
one too. Between the high-symmetry points, a symmetry data vector fixes only the
little-group representations the compatibility relations carry along the lines.
It cannot see whether the bands there can be written as symmetric,
exponentially localized Wannier functions. lin3g's bands 4 and 5, its fragile
pair, have a decomposition and are not an atomic limit.

`subtract(a, b)` takes two symmetry data vectors, each a table or written out by
hand. It returns `a - b`, the symmetry data vector of the bands in `a` and not
in `b`. A difference can only be returned if `b` is contained in `a`. Any other
pair is refused with `negative_multiplicity`.

The example reads lin3g's pair at bands 4 and 5. It checks that the six lowest
bands' symmetry data vector less that of the four below equals the pair's, and
decomposes the pair:

```python
pair = tqc.hsp_table(r, bands=[4, 5])              # the two bands below the gap
print(tqc.subtract(t, tqc.hsp_table(r, bands=[0, 1, 2, 3])) == pair.symmetry_data)
#  True
d_pair = tqc.decompose(pair)
print(d_pair.decomposable, [s.label for s in d_pair.candidate_sums])
#  True ["-E2 at 1b (4mm1')"]
print(tqc.splitting(r, bands=6).nsit)
#  True
```

The pair has a decomposition, the EBR `-E2 at 1b`, and is fragile all the same,
by the construction of Lin *et al.* The stable real-space invariants (SRSIs)
hold information that the symmetry data vector does not. They are combinations
of the multiplicities of the site irreps at each Wyckoff position among the
Wannier functions, which no symmetric deformation changes
([What do the stable real-space invariants say about these bands?](srsi.md)).
`splitting(r, bands=6)` asks whether the six bands below the gap and the two
above it can both be atomic insulators. Its `nsit`, printed `True`, says they
cannot, though each side has atomic realizations
([The no-go criterion](srsi.md#the-no-go-criterion)).
[`examples/tqc_srsi_lin3g_composite_splitting.py`](../../examples/tqc_srsi_lin3g_composite_splitting.py)
runs that analysis on the whole model.

## When there is no decomposition

The example is the model of
[Quick start: Kane-Mele, end to end](../start/quickstart.md): Kane-Mele, from
`pythtb.models`, with a staggered potential, spin-orbit coupling and a Rashba
term. Its two occupied bands decompose, as `-E1` at the hexagon center `1a`,
which holds no orbital. The two bands above them have no decomposition:

```python
km = tqc.realized_group(kane_mele(1.0, -1.0, 0.3, 0.25), typat=[0, 1],
                        characters={0: "A1"})
upper = tqc.hsp_table(km, bands=[2, 3])            # the two bands above the gap
none = tqc.decompose(upper)
print(none, none.certificate, none.numerical.bound_is_rigorous)
#  EBRDecomposition(not decomposable, 0 candidate sums, exact) proof True
signed = none.signed_combinations
print(len(signed.sums))
#  3
print(signed.sums[0].label)
#  -1x-E1 at 1a (3m1') + -E1 at 1b (3m1') + -E1 at 1c (3m1')
print(km.group.name, tqc.symmetry_indicator(upper).indicator_group)
#  P3m11' trivial
```

`certificate` says what the absence of a decomposition rests on. `proof` means
that no sum within the bound searched carries the symmetry data vector, and
that no multiplicity can exceed that bound. `numerical.bound_is_rigorous`,
printed `True`, states the second. Otherwise the certificate is
`none within the bound`, and a sum with a larger multiplicity is not ruled out.

`signed_combinations` holds the same search with negative multiplicities
allowed, each between minus and plus the bound. The symmetry data vector of a
fragile set of bands has such a combination, since with suitable atomic bands
added it is a sum with nonnegative multiplicities. A symmetry data vector with
no integer combination of EBRs at all has a nonzero symmetry indicator
([Which symmetry-indicator class do these bands occupy?](indicators.md)).
Signed multiplicities have no rigorous bound, so a combination can lie beyond
the bound searched. The symmetry data vector of a stably topological set of
bands whose symmetry indicator is zero has a signed combination too.

Here there are three combinations. The first is the model's own orbitals, `-E1`
at `1b` and `1c`, less the `-E1` at `1a` that the occupied pair decomposes into.
[Quick start: Kane-Mele, end to end](../start/quickstart.md) prints the model's
orbitals and the decomposition of the occupied pair. `P3m11'` has a trivial
symmetry-indicator group, printed `trivial`, so every band structure of it has
a symmetry data vector consistent with an integer combination of EBRs. The
signed combinations therefore cannot say whether the pair is fragile or carries
another obstruction.
[`examples/tqc_multifold_connectivity.py`](../../examples/tqc_multifold_connectivity.py)
finds the same in three dimensions: an EBR of space group 208 splits at half
filling into two halves, and neither has a decomposition.

## When it refuses

A refusal is a returned record. It is falsy, and names the check that stopped
the call and what it measured
([What does a refusal mean, and what do I do about it?](refusals.md)). The
checks these calls meet:

- [`incomplete_content`](../reference/refusals.md#incomplete_content): the table
  refused at some high-symmetry point. The table's `refusals` names each such
  point's refusal. With `allow_incomplete=True`, `decompose` leaves those points
  out. The decompositions returned are then those of the little-group
  representations kept, which the whole symmetry data vector may not share,
  unless every one left out occurs in every EBR as often as a kept one. `allow_incomplete=True` is meant for a point whose
  tables cannot be read, and not for a point where the bands themselves were
  refused, such as one where the set of bands splits a degenerate multiplet.
- [`not_a_band_structure`](../reference/refusals.md#not_a_band_structure): the
  symmetry data vector fails the group's compatibility relations, so no set of
  bands separated from the rest at every `k` has it. A set of bands that cuts
  through a crossing between two high-symmetry points is one cause.
- [`restriction_not_exact`](../reference/refusals.md#restriction_not_exact): a
  little-group representation at `k_z = 1/2`, which a layer does not measure,
  occurs in some EBR a different number of times from each measured one.
- [`unknown_label`](../reference/refusals.md#unknown_label): from `decompose`,
  a measured label that the group's EBRs do not hold. `regular_completion`
  handed such a label returns `negative_multiplicity`.
- [`negative_multiplicity`](../reference/refusals.md#negative_multiplicity),
  [`non_integer_multiplicity`](../reference/refusals.md#non_integer_multiplicity):
  a vector that is not a count of little-group representations, such as `a - b`
  where `b` is not contained in `a`.
- [`not_a_regular_multiple`](../reference/refusals.md#not_a_regular_multiple):
  a completion asked for at a number of bands that is not a multiple of the
  number in one copy.
- [`tables_not_ready`](../reference/refusals.md#tables_not_ready): the group's
  cell could not be matched to the setting the tables are written in, so no
  table can be read for it.
- [`ebr_tables`](../reference/refusals.md#ebr_tables): the installed tables
  hold no EBRs for this group.
- [`irrep_tables`](../reference/refusals.md#irrep_tables): the character tables
  were not loaded, or hold no irrep table for this group.
- [`ebr_convention`](../reference/refusals.md#ebr_convention): the tables' EBRs
  of this group cannot be carried into the phase convention of the labels
  returned.

## Where to go next

- [Returned records](../reference/records.md) lists every field of
  [`EBRDecomposition`](../reference/records.md#ebrdecomposition), its
  `Restriction` among them, and of
  [`RegularCompletion`](../reference/records.md#regularcompletion).
- [Functions](../reference/functions.md) gives the arguments of
  [`decompose`](../reference/functions.md#decompose),
  [`subtract`](../reference/functions.md#subtract) and
  [`regular_completion`](../reference/functions.md#regular_completion); of
  [`ebr_rows`](../reference/functions.md#ebr_rows), which returns each EBR of a
  group as a symmetry data vector; and of
  [`in_plane_restriction`](../reference/functions.md#in_plane_restriction),
  which returns the little-group representations of a group's EBRs that a
  layer measures.
- [Which symmetry-indicator class do these bands occupy?](indicators.md) reads
  the class of a set of bands modulo atomic limits.
- [The topological quantum chemistry this package uses](../background/tqc-primer.md)
  gives EBRs, atomic limits and fragile topology at more length.
