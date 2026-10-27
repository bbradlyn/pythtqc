# Could these bands be an atomic limit?

A band set is an atomic limit when its states can be written as symmetric,
exponentially localized Wannier functions. Its symmetry vector — how many times
each corep appears at each high-symmetry point
([labelling the bands](labels.md#every-high-symmetry-point-at-once)) — is then
a sum of **elementary band representations** (EBRs), each the band set one
site-symmetry irrep induces from its Wyckoff position, where that band set is
no sum of others
([the primer](../background/tqc-primer.md#band-representations-and-atomic-limits)).
This chapter asks `decompose` whether a measured symmetry vector is such a sum,
with nonnegative integer multiplicities; reads its verdict and its solutions;
counts bands; and says what a decomposition cannot certify. The model is lin3g, from
`pythtqc.models`, as in [labelling the bands](labels.md): four spinful orbitals
of a square layer, two at Wyckoff `1a` and two at `2c`, eight bands, whose six
lowest are the EBR of the `2c` orbitals and a **fragile** pair: two bands that
are no atomic limit on their own and become one once suitable atomic bands are
added ([fragile topology](../background/tqc-primer.md#fragile-topology)).
Kane–Mele, from `pythtb.models`, supplies a band set with no decomposition.

## The call

```python
import pythtqc as tqc
from pythtb.models import kane_mele
from pythtqc import models

r = tqc.realized_group(models.lin3g(), typat=[0, 0, 1, 1])
model, group, assignment, validated, rep = r
t = tqc.hsp_table(rep, bands=6)                    # the six lowest bands
d = tqc.decompose(t)
print(d.summary())
#  symmetry-indicated decomposable: 3 sums of elementary band representations carry this content   (.verdict)
#    the coreps at the high-symmetry points: 2×-GM6 + -GM7 + -M6 + 2×-M7 + 3×-X5   (.content)
#    every sum, each elementary band representation written as the site irrep inducing it at its Wyckoff position (site symmetry)   (.terms)
#      -E2 at 1b (4mm) + -E at 2c (mm2)
#      -E1 at 1a (4mm) + -E2 at 1a (4mm) + -E2 at 1b (4mm)
#      -E1 at 1b (4mm) + 2×-E2 at 1b (4mm)
#    A, R and Z lie out of the layer's plane: every corep there occurs in every elementary band representation as often as a measured one, so the verdict is exact   (.restriction, .completeness)
#    some site irreps were assigned by the lowest-angular-momentum rule: a different declared content renames them in these sums, while the Wyckoff positions stay   (.site_content, .positions)
#    necessary for an atomic limit, never sufficient: an obstruction beyond symmetry indicators can survive it   (.caveat)
print(d.positions)
#  (('1b', '2c'), ('1a', '1a', '1b'), ('1b', '1b', '1b'))
```

## How to read it

`summary()` prints the record as it is read, a line for each part, ending with
the field that holds it:

- `symmetry-indicated decomposable` is the verdict, and its name is its reach:
  a nonnegative-integer sum of EBRs has this content, and any obstruction to an
  atomic limit must lie beyond what symmetry indicators see. Three sums were
  found.
- The coreps at the high-symmetry points are the symmetry vector decomposed,
  `t.counts` of the six bands.
- Each line under `every sum` is one solution: the orbitals an atomic limit with
  this content would be built from. `-E2 at 1b (4mm)` is the two-band EBR of the
  site corep `-E2` at Wyckoff `1b`, whose site group is `4mm`. The three
  solutions are one content written three ways: the first two differ by `-E` at
  `2c` against `-E1` and `-E2` at `1a`, the last two by that pair at `1a`
  against the same pair at `1b`, sums the symmetry vector cannot tell apart.
  The fewest elementary band representations in all come first, and among sums
  with as many, the one with more of those the tables list first.
- `A, R and Z lie out of the layer's plane`: the **EBR matrix** has one column
  per EBR, its symmetry vector, and one row per corep at each high-symmetry
  point. A layer has no `k_z`, so the rows at the group's points `A`, `R` and
  `Z`, at `k_z = ½`, are left out, and each is verified to repeat a measured
  row in every EBR. Leaving them out then changes nothing: the matrix has rank
  `3` with them and without, and the verdict is the full decomposition's.
  `d.restriction` holds the rows kept and those left out, each with the row it
  repeats.
- The line on the lowest-angular-momentum rule is the next section. The last
  line, the caveat every decomposition carries, is
  [what a decomposition cannot certify](#what-a-decomposition-cannot-certify).

`d.positions` is the Wyckoff positions of each solution, the part of it no
labelling of the orbitals moves. `d.terms` holds each solution as its terms,
each a multiplicity, an EBR and its position, as the group's list of EBRs
labels them: `-E2↑G(2)` at `1b(4mm1',4mm)` is the summary's `-E2 at 1b (4mm)`,
the `(2)` its two bands and `4mm1'` its site group with time reversal
([the band-representation key](../background/conventions.md#the-band-representation-key)
reads every part), and `solution_strings()` writes each sum on a line in the
same labels. `d.solutions` holds each solution as multiplicities over the EBRs
of `d.ebrs`, the vector the
[stable real-space invariants](srsi.md#why-the-ℤₙ-half-needs-the-orbitals)
are evaluated on; its nonzero entries are the terms:

```python
print(d.terms[0])
#  ((1, '-E2↑G(2)', "1b(4mm1',4mm)"), (1, '-E↑G(4)', "2c(mm21',mm2)"))
print(d.solutions[0], [d.ebrs[i] for i, n in enumerate(d.solutions[0]) if n])
#  (0, 0, 0, 1, 1) [('-E2↑G(2)', "1b(4mm1',4mm)"), ('-E↑G(4)', "2c(mm21',mm2)")]
```

## Positions survive every labelling; labels may not

The hoppings fix how the orbitals transform relative to one another, not what
they are, and lin3g's two orbitals at `1a` were assigned by the lowest-ℓ rule
([labelling the bands](labels.md#when-a-label-is-not-trustworthy)):
`site_content` records how each row is known. Declaring orbital 0 a
`d_{x²−y²}` states a crystal the same hoppings describe, with the site irreps
at `1a` exchanged:

```python
print(d.site_content["determined_by"])
#  ('lowest-l', 'lowest-l', 'hamiltonian', 'hamiltonian')
other = tqc.realized_group(models.lin3g(), typat=[0, 0, 1, 1], characters={0: "dx2-y2"})
d_other = tqc.decompose(tqc.hsp_table(other.rep, bands=6))
print(d_other.solution_strings()[0])
#  1 x -E1↑G(2)@1b(4mm1',4mm) + 1 x -E↑G(4)@2c(mm21',mm2)
print(d_other.positions == d.positions)
#  True
```

The first solution now reads `-E1` at `1b` where it read `-E2`, and every
solution sits at the same Wyckoff positions. Quote `d.positions` beside the
labels always, and lean on them wherever a row reads `lowest-l`: they are what
a decomposition verdict is about, and what a comparison with the Bilbao
Crystallographic Server's band-representation tables agrees on where a label
may not ([reading across to BCS](../background/conventions.md#reading-across-to-bcs)).

## Counting bands

Comparing two models at equal band count, or building a reference atomic
limit, needs each side of a gap to hold complete band representations.
`completeness_check` returns whether the content is a sum of EBRs at all,
`complete`, with the same reach as a decomposition's verdict, and how many
bands each solution accounts for, `band_counts`, against the number of bands
asked for; a mismatch there is a mistake in the call, since the EBR
dimensions and the corep dimensions are the same numbers. `regular_completion`
pads a band set up to **regular** content: the band representation induced
from the regular representation of a site-symmetry group, `|U|` bands at every
k-point, with `|U|` the number of unitary operations of the group, one per coset
of the lattice translations and, for this spinful model, the single-valued
group's: the eight of `4mm` here (`rc.bands_per_copy`):

```python
print(tqc.completeness_check(t))
#  CompletenessReport(complete=True, n_bands=6, band_counts=(6, 6, 6))
rc = tqc.regular_completion(t, bands=8)
print(rc.bands_per_copy, rc.regular)
#  8 {'-GM6': 2, '-GM7': 2, '-M6': 2, '-M7': 2, '-X5': 4}
print(tqc.format_counts(rc.padding), rc.decomposition.solution_strings())
#  -GM7 + -M6 + -X5 ("1 x -E1↑G(2)@1b(4mm1',4mm)",)
print(rc.padding == tqc.hsp_table(rep, bands=[6, 7]).counts)
#  True
print(tqc.regular_completion(t, bands=6).gate)
#  not_a_regular_multiple
```

`rc.regular` is one copy's content, computed from the group's own corep
characters: eight bands at every point. The padding is that content less the
six bands', and its own decomposition names the orbitals it would be. lin3g
has eight bands, one regular copy, so the padding is exactly the content of the
two bands above the gap. Padding both sides of a gap to regular content makes
the total an atomic limit by construction, which is what lets two models be
compared at equal band count. A band count that is not a multiple of `|U|` is
refused, `not_a_regular_multiple`.

## What a decomposition cannot certify

Corep multiplicities at isolated k-points are blind to everything between
them, and lin3g's bands 4 and 5 are the standing example. `tqc.subtract(a, b)`
is `a − b` as a symmetry vector, refused (`negative_multiplicity`) where `b` is
not contained in `a`, since the difference of two band sets is a band set only
when one contains the other; the pair is the six bands less the four below
them:

```python
pair = tqc.hsp_table(rep, bands=[4, 5])            # the two bands below the gap
print(tqc.subtract(t, tqc.hsp_table(rep, bands=[0, 1, 2, 3])) == pair.counts)
#  True
d_pair = tqc.decompose(pair)
print(d_pair.verdict, d_pair.solution_strings(), d_pair.caveat)
#  symmetry-indicated decomposable ("1 x -E2↑G(2)@1b(4mm1',4mm)",) WannierCaveat(necessary for an atomic limit, not sufficient)
print(tqc.splitting(r, bands=6).verdict)
#  diagnosed topological
```

The pair reproduces the EBR `-E2↑G(2)@1b` exactly and is fragile all the same,
by the construction of Lin et al. Every decomposition carries
`caveat`, which says so in general: a symmetry-indicated decomposable verdict
is necessary for an atomic limit and never sufficient. What sees past it here
is the **stable real-space invariants**, combinations of how many Wannier
functions carry each site irrep at each Wyckoff position that no symmetric
deformation can change ([stable real-space invariants](srsi.md)):
`splitting(r, bands=6)` asks whether the six bands below the gap and the two
above it can both be atomic insulators, and its verdict,
`diagnosed topological`, says they cannot
([the no-go criterion](srsi.md#the-no-go-criterion)).
[`examples/tqc_srsi_lin3g_composite_splitting.py`](../../examples/tqc_srsi_lin3g_composite_splitting.py)
runs that analysis on this model whole.

## When there is no decomposition

The other direction is conclusive. The model of the
[quick start](../start/quickstart.md), Kane–Mele with a staggered potential,
spin–orbit coupling and a Rashba term, has two occupied bands that decompose,
as `-E1` at the empty hexagon centre `1a`; the two above them do not:

```python
km = tqc.realized_group(kane_mele(1.0, -1.0, 0.3, 0.25), typat=[0, 1],
                        characters={0: "A1"})
upper = tqc.hsp_table(km.rep, bands=[2, 3])        # the two bands above the gap
none = tqc.decompose(upper)
print(none, none.certificate, none.bound_is_rigorous)
#  EBRDecomposition(none, 0 solutions, exact) proof True
signed = none.signed_combinations
print(signed["n_solutions"])
#  3
print(" + ".join(f"{n} x {ebr}@{site}" for n, ebr, site in signed["terms"][0]))
#  -1 x -E1↑G(2)@1a(3m1',3m) + 1 x -E1↑G(2)@1b(3m1',3m) + 1 x -E1↑G(2)@1c(3m1',3m)
print(km.group.name, tqc.symmetry_indicator(upper).verdict)
#  P3m11' trivial indicator group
```

A `none` verdict says no atomic limit has this content, and `certificate` says
what that rests on: `proof` where the bound placed on every multiplicity
provably excludes no solution (`bound_is_rigorous`), so that finding none
within it is a proof, and `none within the bound` otherwise.

The **signed combinations** ask the same question with negative
multiplicities allowed. A fragile band set has such a solution, an integer
combination of EBRs with a negative coefficient, since adding atomic bands
makes it a nonnegative sum; a content with no integer solution at all is
obstructed stably, which a nonzero symmetry indicator states
([the next chapter](indicators.md)). Signed multiplicities have no rigorous
bound, so the signed combinations are evidence. Here there are `3` solutions, the
first the model's own orbitals, `-E1` at `1b` and `1c`, less the `-E1` at `1a`
the occupied pair decomposed into (both are printed in
[the quick start](../start/quickstart.md)). That cannot say which kind of
obstruction the pair carries: `P3m11'` has no symmetry indicator, so every band
structure of it is an integer combination of EBRs, fragile or not.
[`examples/tqc_multifold_connectivity.py`](../../examples/tqc_multifold_connectivity.py)
meets the same verdict in three dimensions: an EBR of space group 208 split at
half filling into two halves, neither with a decomposition.

## When it refuses

A refusal is a returned record, falsy, naming the check that stopped the call
and what it measured ([when a call refuses](refusals.md)). The checks these
calls meet:

- [`incomplete_content`](../reference/refusals.md#incomplete_content): the
  table has points that did not resolve, and a vector missing a point
  decomposes too easily; read the table's refusals. `allow_incomplete=True`
  decomposes anyway, for a point whose tables cannot be read rather than one
  where the physics failed, and the claim is then a necessary condition unless
  the rows it gives up repeat kept ones.
- [`not_a_band_structure`](../reference/refusals.md#not_a_band_structure): the
  content fails the group's compatibility relations, so no set of bands
  separated from the rest at every `k` has it, as when the bands asked for cut
  through a crossing between two high-symmetry points; there is no
  decomposition to look for.
- [`restriction_not_exact`](../reference/refusals.md#restriction_not_exact): a
  row the model does not measure repeats none it does, so no verdict from the
  measured rows is exact; `points=` naming fewer points than the model samples
  can meet it. A layer is decomposed over its own band representations, those of
  the Wyckoff positions within it, on which every row out of its plane repeats
  one in it: a free-standing layer's group has positions off the layer, which
  are left out and named in `restriction.outside_layer`.
- [`unknown_label`](../reference/refusals.md#unknown_label): a measured label
  the group's EBR basis does not carry.
- [`negative_multiplicity`](../reference/refusals.md#negative_multiplicity),
  [`non_integer_multiplicity`](../reference/refusals.md#non_integer_multiplicity):
  a content that is not a count of coreps, such as a difference of two band
  sets neither of which contains the other.
- [`not_a_regular_multiple`](../reference/refusals.md#not_a_regular_multiple),
  [`regular_content_not_integral`](../reference/refusals.md#regular_content_not_integral):
  a padding asked for at a band count that is not a multiple of `|U|`, or
  tabulated characters that give no whole regular content.
- [`ebr_tables`](../reference/refusals.md#ebr_tables),
  [`irrep_tables`](../reference/refusals.md#irrep_tables),
  [`ebr_convention`](../reference/refusals.md#ebr_convention): no EBR data to
  decompose against for this group.

## Every field and argument

The fields of [`EBRDecomposition`](../reference/records.md#ebrdecomposition),
its `Restriction` and `WannierCaveat` among them, and of
[`CompletenessReport` and `RegularCompletion`](../reference/records.md#completenessreport-and-regularcompletion)
are in the records reference, and each call's arguments in its entry: [`decompose`](../reference/functions.md#decompose),
[`subtract`](../reference/functions.md#subtract),
[`completeness_check`](../reference/functions.md#completeness_check),
[`regular_completion`](../reference/functions.md#regular_completion),
[`ebr_rows`](../reference/functions.md#ebr_rows), which writes each EBR of a
group as a symmetry vector, and
[`in_plane_restriction`](../reference/functions.md#in_plane_restriction). What
an EBR, an atomic limit and fragile topology are is in
[the primer](../background/tqc-primer.md).
