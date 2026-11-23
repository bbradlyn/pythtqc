# How do I measure a numerical or Wannier90 model?

A Hamiltonian fitted to a calculation, such as one imported from Wannier90, is
symmetric only to the precision of its own numbers. This page shows how that
precision is declared, and what `realized_group` does with it.

## The call

`precision=` is the relative precision of the model's numbers, a positive
dimensionless number. It is how far the numbers may miss the symmetry of the
crystal they approximate: relative to the size of `H` for the hoppings, and in
reduced coordinates for the orbital centers. It is not how many decimals a
file prints them to. `realized_group` derives the tolerances of the
measurement from it and from `max|H|`, the largest entry of `H(k)` once the
mean on-site energy is subtracted.

The example is lin3g, from `pythtqc.models`, the spinful square layer of Lin
*et al.*, [Nat. Commun. **15**, 550 (2024)](https://doi.org/10.1038/s41467-024-44762-w).
Every hopping is changed at random by about `1e-7` of itself, as the numbers of
a fitted model are, and that precision is declared:

```python
import numpy as np
import pythtqc as tqc
from pythtb import TBModel
from pythtqc.models import lin3g

clean = lin3g()                                  # a spinful square layer, four orbitals
rng = np.random.default_rng(7)
noisy = TBModel(clean.lattice, spinful=True)     # the same model, every hopping off by ~1e-7
noisy.set_onsite(clean.onsite, mode="set")
for h in clean.hoppings:
    noisy.set_hop(h["amplitude"] * (1 + 1e-7 * rng.normal()), h["from_orbital"],
                  h["to_orbital"], h.get("lattice_vector", [0, 0]), mode="add")
rn = tqc.realized_group(noisy, typat=[0, 0, 1, 1], characters=["s", "dx2-y2", "s", "s"],
                        precision=1e-7)
print(repr(rn))
#  RealizedGroup(P4mm1' #99.164; precision 1e-07 declared -> tol 1.74e-05 (cut 1e-06, window 0.01), position_tol 1e-07 (measured max|H| 17.42); symmetrized (defect 5.54e-07); on a substrate (the default), the free-standing group P4mm1' #99.164 agrees)
```

## How to read it

- `P4mm1' #99.164`: the group of lin3g without its noise. Every operation of it
  holds within `tol`.
- `precision 1e-07 declared`: the precision declared.
- `tol 1.74e-05`: an energy tolerance for checking whether an operation is a
  symmetry of `H`. For an operation `g`, with `U(g;k)` its matrix from the
  Bloch states at `k` to those at `gk`, the covariance residual
  `max_k ‖U(g;k) H(k) U(g;k)† − H(gk)‖` is measured, with `H(k)*` in place of
  `H(k)` for an operation combined with time reversal. The operation holds
  when the residual is at most `tol`, in the units of `H`. Computed from
  `precision` and `max|H|`.
- `cut 1e-06`: a dimensionless tolerance for checking whether an operation is a
  symmetry. For each operation, the singular values of the symmetry equations
  are computed. A singular value is treated as zero when it is at most `cut`
  times the largest singular value. Computed from `precision` if provided.
- `window 0.01`: a dimensionless tolerance for checking degeneracy of the
  Hamiltonian `H`. Two eigenvectors are treated as degenerate when their
  energies differ by at most `window` times `max|H|`. Computed from `precision`
  if provided.
- `position_tol 1e-07`: a tolerance for moving the orbital centers onto
  symmetric positions, in reduced coordinates. A center that misses the
  position the group puts it at by at most `position_tol` is moved onto it.
  The miss is measured about the origin fitted to the centers, as the example
  of Kane-Mele below shows. Equal to `precision` if provided.
- `measured max|H| 17.42`: `max|H|` as measured on the model, in the units of
  `H`. A constant shift of `H`, such as a Fermi energy subtracted at the
  import, moves no tolerance.
- `symmetrized (defect 5.54e-07)`: `H` was projected onto its symmetric part,
  which moved it by `5.54e-7` in the units of `H`
  ([Then symmetrize](#then-symmetrize)).
- `on a substrate (the default), the free-standing group P4mm1' #99.164 agrees`:
  reads as for any layer
  ([What group does my Hamiltonian have?](group.md#how-to-read-it)).

The numbers of the measurement are on `rn.numerical.precision`:

```python
p = rn.numerical.precision
print(f"max|H| {p.max_h:.4g}: tol {p.tol:.3g}, cut {p.cut:.3g}, window {p.window:.3g}")
#  max|H| 17.42: tol 1.74e-05, cut 1e-06, window 0.01
print(f"position_tol {p.position_tol:.0e}, symprec {p.symprec:.0e}, raised: {p.symprec_raised}")
#  position_tol 1e-07, symprec 1e-05, raised: False
```

- `symprec 1e-05`: the tolerance of the symmetry finder, spglib, which finds
  the operations of the
  [cell](../background/measurement.md#the-cell-and-a-verdict-on-each-operation)
  from the centers. It is a Cartesian length, in the units the pythtb Lattice
  vectors are given in, `1e-5` by default.
- `raised: False`: whether the declared precision raised `symprec`. It is
  raised where the precision times the longest periodic lattice vector exceeds
  it.

`position_tol=` may be declared alone, for a model whose centers alone are
imprecise. Declared beside `precision=` it raises `ValueError`, since the
precision sets it. `symprec=` may be declared beside `precision=`. The
precision then raises it on the same condition, and never lowers it. The
formulas of `tol`, `cut`, `window` and `position_tol`, with every tolerance's
unit and default, are in [Tolerances](../reference/glossary.md#tolerances).

With nothing declared, the centers are not moved, and each operation is held
to the default `tol`, `1e-8` times `max|H|`, a tolerance for a model built from
exact numbers. The same noisy model is held to it:

```python
print(repr(tqc.realized_group(noisy, typat=[0, 0, 1, 1], characters=["s", "dx2-y2", "s", "s"])))
#  RealizedGroup(P11' #1.2; descended from P4mm1' #99.164 (14 of 16 coset representatives of the lattice translations broken; these are the single-valued group's, each standing for itself and its negative in the double group); on a substrate (the default), the free-standing group P11' #1.2 agrees)
```

The noise breaks every operation but the identity and time reversal, and the
group returned is `P11'`. That is true of the numbers handed in, and says
nothing about the crystal they approximate. Declaring the precision the numbers
hold returns the crystal's group. The silicon of the tutorial, written to six
decimals and symmetric to about `1e-4`, is returned `P1` with `precision=1e-5`
and `R-3m1'` with `precision=1e-4`
([One declaration: `precision=`](../start/tutorial-wannier90.md#one-declaration-precision)).

## Exact, near and broken

Each operation of the cell is put in one of three classes by its covariance
residual, read against the `tol` in force, declared or the default.

| class | covariance residual | what is done |
|---|---|---|
| **exact** | at most rounding | nothing |
| **near** | above rounding, at most `tol` | `H` is projected onto its symmetric part |
| **broken** | above `tol` | the operation is dropped, and `H` is not projected onto it |

The rounding is in the units of `H`. It grows with the number of operations,
the number of states, the largest entry of `H(k)` with its energy zero kept,
and the rounding of the Bloch phases
([Tolerances](../reference/glossary.md#tolerances)).

```python
s = rn.numerical.symmetrization
print(f"{len(s.near)} near, {len(s.exact)} exact, near up to {s.tol:.3g}")
#  14 near, 2 exact, near up to 1.74e-05
```

The two exact operations of the noisy lin3g are the identity and time reversal,
which a real rescaling of each hopping keeps. The other fourteen hold only
within `tol`.

## The centers first: moving them is a change of gauge

A fitted model's orbital centers are numbers too. A hopping from orbital `i` to
orbital `j` in the cell `R` carries the Bloch phase
`e^{2πi k·(R + τ_j − τ_i)}`. So a defect `δ` in the centers `τ` adds about
`2π·δ·max|H|` to the covariance residual, whatever the hoppings do, and no
projection of the hoppings removes it. Moving a center changes the basis of
`H(k)` by a `k`-dependent phase, and leaves the spectrum at every `k`
unchanged. The call measures the change of the spectrum and returns it. So the
centers are moved before any operation is put to the Hamiltonian and before any
hopping is touched.

The example is Kane-Mele, from `pythtb.models`, with one of its two centers
moved by `1e-8` in reduced coordinates. With nothing declared, the call refuses
before it builds anything:

```python
from pythtb import Lattice
from pythtb.models import kane_mele

def with_centers(model, centers):
    """The model's own hoppings, with its orbital centers at `centers` (reduced)."""
    out = TBModel(Lattice(model.lat_vecs, centers, periodic_dirs=model.periodic_dirs),
                  spinful=True)
    out.set_onsite(model.onsite, mode="set")
    for h in model.hoppings:
        out.set_hop(h["amplitude"], h["from_orbital"], h["to_orbital"],
                    h.get("lattice_vector", [0, 0]), mode="add")
    return out

km = kane_mele(0.0, -1.0, 0.3, 0.25)                               # delta, t, soc, rashba
moved = with_centers(km, km.orb_vecs + [[1e-8, 0.0], [0.0, 0.0]])  # one center 1e-8 off
bad = tqc.realized_group(moved, characters={0: "A1"})
print(bad.gate, f"{bad.value:.1e}", bad.context["group"].name, bad.context["number_str"],
      f"{2 * np.pi * bad.value * bad.context['max_h']:.1e}")
#  positions_not_exactly_symmetric 2.0e-08 P6mm1' 183.186 3.7e-07
```

The centers miss the cell's own operations by up to `2e-8` in reduced
coordinates. The threefold rotation about the displaced center's site, for
one, carries that center to a point `2e-8` from it. The miss costs `H(k)` a
residual of about `3.7e-7`, where the default `tol` is `3e-8` here. Measured as
it stands, which operations `H` obeys would be decided by the centers'
geometry, not by the hoppings alone. So the call refuses with
`positions_not_exactly_symmetric`, which quotes the defect and names the cell's
group, `P6mm1' #183.186`, which contains the group `H` obeys. Declaring a
tolerance for the centers, `position_tol=`, lets the call move them:

```python
fixed = tqc.realized_group(moved, characters={0: "A1"}, position_tol=1e-6)
print(repr(fixed))
#  RealizedGroup(P6mm1' #183.186; centers moved 5e-09; on a substrate (the default), the free-standing group P6mm1' #183.186 agrees)
pos = fixed.numerical.positions
print(f"{pos.defect_before:.1e} -> {pos.defect_after:.1e}, spectrum moved "
      f"{pos.spectrum_residual:.1e}; symmetrized: {fixed.numerical.symmetrization}")
#  2.0e-08 -> 2.2e-16, spectrum moved 1.8e-15; symmetrized: None
print(pos.frame, f"each moved {pos.max_move:.0e}, then both {abs(pos.offset[0]):.0e};",
      "back where Kane-Mele has them:", np.allclose(fixed.model.orb_vecs[:, :2], km.orb_vecs,
                                                   rtol=0, atol=1e-15))
#  model each moved 5e-09, then both 5e-09; back where Kane-Mele has them: True
```

- `P6mm1' #183.186; centers moved 5e-09`: the cell's whole group is returned.
  The centers were moved onto the positions it puts them at. The largest move
  of a center about the origin fitted to the centers is `5e-9`, in reduced
  coordinates.
- `2.0e-08 -> 2.2e-16`: `pos.defect_before` and `pos.defect_after`, how far the
  centers missed the group's operations before the move and after it.
- `spectrum moved 1.8e-15`: `pos.spectrum_residual`, the measured change of the
  spectrum, which is rounding.
- `symmetrized: None`: no projection ran, since the hoppings were exact. Only
  the centers were moved.
- `model`: `pos.frame`, the frame the centers are written in. It is `"model"`
  where they are written in the model's own frame, and `"fitted"` where they
  stay about the origin fitted to them.
- `each moved 5e-09`: `pos.max_move`, the largest move of a center about the
  origin fitted to the centers. The centers are fitted about their own origin,
  since a shift of all of them together is a change of origin, which the
  operations' translations absorb. The fit shares the one center's
  displacement between the two, and each moves by `5e-9`.
- `then both 5e-09`: `pos.offset`, the shift of both centers together that the
  fit leaves. It is within the declared `1e-6`, so it is removed too, and the
  centers are written in the model's own frame.
- `True`: the centers returned are where Kane-Mele has them. The displaced one
  moved back by the whole `1e-8`, and the other did not move.

A declared tolerance bounds each center's move about the origin fitted to the
centers, and the shift of all of them together. Centers that the group's
operations do not carry within it are a different set of points, not a noisy
copy of a symmetric one. An operation under which the centers would have to
move farther than the declaration allows is broken by the centers, as one the
hoppings break is. The group descends to the largest group of its operations
that carries the centers within the declaration, and the centers are moved onto
that group's positions before any hopping is read. The example declares
`1e-9`, less than the move the whole group asks:

```python
below = tqc.realized_group(moved, characters={0: "A1"}, position_tol=1e-9)
print(repr(below))
#  RealizedGroup(P21' #3.2; descended from P6mm1' #183.186 (20 of 24 coset representatives of the lattice translations broken by the centers; these are the single-valued group's, each standing for itself and its negative in the double group); on a substrate (the default), the free-standing group P21' #3.2 agrees)
first = below.descent.dropped[0]
print(first.name, first.refuted_by, f"{first.miss:.0e}")
#  C3(z) centers 5e-09
print(f"{below.numerical.positions.move:.1e} asked of a center by the whole group, "
      f"{below.numerical.positions.carried} of its coset representatives carry them")
#  5.0e-09 asked of a center by the whole group, 4 of its coset representatives carry them
print(tqc.hsp_table(below, bands=2).symmetry_data)
#  {'-GM3GM4': 1, '-A3A4': 1, '-B3B4': 1, '-Y3Y4': 1}
```

- `broken by the centers`: the operations of `P6mm1'` that do not carry the
  centers within `1e-9`.
- `C3(z) centers 5e-09`: one of them, from `below.descent.dropped`, with
  `refuted_by` and `miss`. `refuted_by` is `"centers"` for each. `miss` is the
  largest move of a center, about the origin fitted to the centers, that the
  operation and its powers would ask. It is a move, as `positions.move` is, and
  not the `2e-8` from a center to its image.
- `5.0e-09 asked of a center by the whole group`: `positions.move`, five times
  the declared `1e-9`.
- `4 of its coset representatives carry them`: `positions.carried`, the number
  of coset representatives of `P6mm1'` that carry the centers within `1e-9`.
- `P21' #3.2`: the twofold rotation through the midpoint of the two centers
  exchanges them, as such a rotation does for any pair of points. It and time
  reversal are what the centers keep, and the group returned carries them
  exactly.
- `{'-GM3GM4': 1, ...}`: the corep multiplicities of the two lowest bands at
  each high-symmetry point, read in `P21'`. `GM` is the tables' name for Γ.

The shift of both centers together that would write them in the model's own
frame, `5e-9`, is beyond the declared `1e-9` as well, so it is not removed. The
centers stay about the origin fitted to them, and `numerical.positions.frame`
reads `"fitted"`.

### Centers noisier than `symprec`

`positions_not_exactly_symmetric` is returned for centers close enough to the
cell's operations for the symmetry finder to find them, and far enough off to
break them. Centers noisier than `symprec` are a less symmetric set of points
to the finder. The group returned is that set's, as low as `P1`, with no
refusal. The example is Kane-Mele with both centers displaced at random by
about `5e-5` in reduced coordinates, measured with nothing declared and then
with `precision=1e-3`:

```python
rng = np.random.default_rng(3)
blurred = with_centers(km, km.orb_vecs + 5e-5 * rng.normal(size=(2, 2)))  # every center off
print(repr(tqc.realized_group(blurred)))
#  RealizedGroup(P21' #3.2; on a substrate (the default), the free-standing group P21' #3.2 agrees)
print(repr(tqc.realized_group(blurred, precision=1e-3)))
#  RealizedGroup(P6mm1' #183.186; precision 0.001 declared -> tol 0.0297 (cut 0.01, window 0.3), position_tol 0.001 (measured max|H| 2.969), symprec raised to 0.01; centers moved 4.97e-05; on a substrate (the default), the free-standing group P6mm1' #183.186 agrees)
```

- `P21' #3.2`: with nothing declared, the two centers keep one rotation, the
  twofold axis through their midpoint, which any pair of points in a plane
  lattice has.
- `symprec raised to 0.01`: the declared precision raised the finder's
  tolerance from `1e-5`, and the finder finds the cell's operations.
- `centers moved 4.97e-05`: the centers were then moved onto the positions
  the cell's group puts them at.

With more orbitals in the cell, noisy centers may keep no operation but the
identity. Silicon's eight centers sit up to `2.8e-5` Å from the positions
diamond's operations put them at, nearly three times the default `symprec`,
and its import with nothing declared is returned `P1`
([Asked without a precision](../start/tutorial-wannier90.md#asked-without-a-precision)).
Where a numerical model is returned an implausibly low group with nothing
declared, declaring its precision is the first step.

## Then symmetrize

A near operation holds within `tol` and not to rounding. Projecting `H` onto
its symmetric part then moves it by about the residual, within that same
tolerance, and the model returned is symmetric to rounding. `realized_group`
runs the projection itself wherever an operation is near, and
`numerical.symmetrization` says how far `H` moved. With nothing declared, the
projection runs against the default `tol` the same way: a model whose hoppings
are off by about `1e-11` of themselves is projected too.

```python
print(f"{len(s.near)} near, {len(s.exact)} exact; H moved by {s.defect:.2e}, "
      f"{s.defect / s.scale:.1e} relative; passes {s.passes}")
#  14 near, 2 exact; H moved by 5.54e-07, 3.4e-08 relative; passes 1
print(rn.numerical.source is noisy, rn.model is noisy)
#  True False
rc = tqc.realized_group(clean, typat=[0, 0, 1, 1], characters=["s", "dx2-y2", "s", "s"])
print(rc.group.name == rn.group.name,
      tqc.hsp_table(rn, bands=6).symmetry_data == tqc.hsp_table(rc, bands=6).symmetry_data,
      max(np.abs(rn.rep.matrices[i] - rc.rep.matrices[i]).max() for i in rc.rep.matrices))
#  True True 0.0
```

- `H moved by 5.54e-07, 3.4e-08 relative`: `s.defect`, the largest change of
  an entry of `H(k)` over the projection's grid of `k`-points, in the units of
  `H`. Then `s.defect / s.scale`, the same relative to `s.scale`, the largest
  entry of `H(k)` on that grid once the mean on-site energy is subtracted.
- `passes 1`: `s.passes`, the number of averaging passes, at most three. A
  further pass runs only while the model returned is not symmetric to rounding
  and the last pass moved `H` by more than rounding.
- `True False`: `rn.numerical.source` is the model handed in, and `rn.model` is
  the symmetrized model, which every later call takes.
- `True True 0.0`: the noisy lin3g is returned the clean model's group, the
  clean model's symmetry data vector for its six lowest bands, and
  representation matrices identical to the clean model's.

The projection replaces `H(k)` by the average, over every operation `g` of the
group, of `U(g;k)† H(gk) U(g;k)`, or its complex conjugate for an operation
combined with time reversal. Those operations are among the ones averaged
over, so a model with time reversal keeps it. A tight-binding `H(k)` is a
finite Fourier series, so the average is again a hopping list, exactly, with no
fit.

The projection also runs where no operation is near, in four cases. Each has a
field of `numerical.symmetrization`, which
[Returned records](../reference/records.md) describes:

- `site_multiplets`: a site's orbitals split into
  [multiplets](../reference/glossary.md#words-the-pages-use) only within the
  declared precision, as when a Wannier basis is turned inside its site by less
  than the precision. The projection removes the mixing between multiplets and
  nothing else, so the orientation of the orbitals inside each multiplet stays
  the model's own. The field says how far they mixed.
- `parts`: [parts of the crystal](../background/measurement.md#what-the-hoppings-fix)
  are coupled to one another only below the declared precision. Such a coupling
  counts as absent, so each part leaves open its own one-dimensional character
  of the group, which the hoppings do not fix. The projection leaves the
  coupling out, so the parts of the model returned are exactly apart. The field
  names the parts and says how large the coupling was.
- `copies`: a part of the crystal comes in identical copies, coupled or told
  apart only below the declared precision. The projection makes them exact
  copies, so a declaration that mixes them, such as two identical `s` orbitals
  declared the doublet `E`, holds on the model returned. The field names their
  orbitals and says how far they were from exact copies.
- `shares`: two identical lattices coupled on site above the precision are a
  bonding and an antibonding lattice, each carrying its own representation of
  the site-symmetry group. Where every entry of `H(k)` that couples one of the
  two to anything outside it is below the precision, the projection leaves
  those entries out. The field names their orbitals and says how large the
  largest of them was.

## A broken operation is never imposed

The example scales one hopping of lin3g by `1.01`, with the same precision
declared. The operations the change touches are broken by far more than `tol`,
and the others stay exact. So no operation is near, the projection does not
run, and the group
[descends](../background/measurement.md#the-descent-and-the-tie-rule):

```python
one = TBModel(clean.lattice, spinful=True)                 # one hopping 1% out
one.set_onsite(clean.onsite, mode="set")
for n, h in enumerate(clean.hoppings):
    one.set_hop(h["amplitude"] * (1.01 if n == 0 else 1.0), h["from_orbital"],
                h["to_orbital"], h.get("lattice_vector", [0, 0]), mode="add")
rb = tqc.realized_group(one, typat=[0, 0, 1, 1], characters=["s", "dx2-y2", "s", "s"],
                        precision=1e-7)
print(repr(rb))
#  RealizedGroup(Pmm21' #25.58; precision 1e-07 declared -> tol 1.74e-05 (cut 1e-06, window 0.01), position_tol 1e-07 (measured max|H| 17.42); descended from P4mm1' #99.164 (8 of 16 coset representatives of the lattice translations broken; these are the single-valued group's, each standing for itself and its negative in the double group); on a substrate (the default), the free-standing group Pmm21' #25.58 agrees)
print(rb.numerical.symmetrization, sorted({b.name for b in rb.descent.dropped}))
#  None ['C4(z)', "C4(z)'", 'C4^-1(z)', "C4^-1(z)'", 'm([1-10])', "m([1-10])'", 'm([110])', "m([110])'"]
```

The fourfold rotations and the diagonal mirrors, each with its partner
combined with time reversal, are broken and named on `rb.descent`, and
`numerical.symmetrization` is `None`.

## A large model

The projection holds `H(k)` and its group average together on a grid of
`k`-points. The grid is set by the range the group carries the hoppings to
along each periodic direction, and is the same along directions the group's
operations map into one another. `max_bytes=`, `1e9` bytes by default, is the
memory the projection may take at its peak. A coarser grid would alias, so a
projection past the budget is not run, and the call returns
`projection_too_large`, naming the `max_bytes=` that runs it. Only a model with
many states and long-ranged hoppings reaches it.

The measurement of the operations has a memory budget of its own, `5.12e8`
bytes. Where a spectrum's exactly degenerate levels need more, that budget is
raised to `max_bytes=` where that is larger, and past it the call returns
`detector_too_large`. Either way, the same call with the `max_bytes=` the
refusal names runs it.

The measurement of the operations also takes more memory and time the looser
the declared precision. `window` grows with the precision, up to `0.3`, and on
a model with many states the eigenvectors it treats as degenerate grow in
number with it. So the precision to declare is the one the model's numbers
hold, and not a looser one.

## When it refuses

A refusal is a returned record. It is falsy, and names the check that stopped
the call and what it measured
([What does a refusal mean, and what do I do about it?](refusals.md)). The
checks this route meets:

- [`positions_not_exactly_symmetric`](../reference/refusals.md#positions_not_exactly_symmetric):
  with no precision declared, the centers miss the cell's operations by enough
  to break them, or by enough that the operations read off them do not close
  into a group. The reason names the cell's group. Once the group is
  identified, the same refusal is returned for centers that miss by less,
  where their miss alone keeps `H` from being made exactly symmetric. The
  reason then names the group identified.
- [`operations_not_consistent`](../reference/refusals.md#operations_not_consistent):
  the operations the symmetry finder reads off the centers do not close into a
  group, and the cell matches no setting of the tables whose exact operations
  could replace them. It usually happens where two sites of one species sit
  closer together than `symprec`.
- [`operations_not_nearly_consistent`](../reference/refusals.md#operations_not_nearly_consistent):
  with no precision declared, the operations read off the centers do not close
  into a group, and replacing their translations by the tables' exact ones
  would change a translation by more than `symprec`, read as a Cartesian
  length.
- [`rep_not_covariant`](../reference/refusals.md#rep_not_covariant): every
  operation holds within `tol` on its own, and the representation built for
  the whole group has a covariance residual above `tol`. The Hamiltonian's
  asymmetry is then close to the tolerance.
- [`cell_group_not_realized`](../reference/refusals.md#cell_group_not_realized):
  an operation combined with time reversal holds before the projection and is
  broken after it.
- [`symmetrize_defect_exceeds_tol`](../reference/refusals.md#symmetrize_defect_exceeds_tol)
  and
  [`symmetrize_range_not_closed`](../reference/refusals.md#symmetrize_range_not_closed):
  the projection's own checks, on how far it moved `H` and on the hopping list
  it returns.
- [`projection_too_large`](../reference/refusals.md#projection_too_large) and
  [`detector_too_large`](../reference/refusals.md#detector_too_large): a memory
  budget.

## Where to go next

- [How does pythTQC read the spin of my model?](spin.md) says how `spin=`
  beside `precision=` names the orbitals that are the up and down states of one
  spinful orbital
  ([A Wannier90 spinor import](spin.md#a-wannier90-spinor-import)). pythtb's
  importer delivers a spinor calculation as a spinless model, with the spin
  inside the orbital index.
- [Functions](../reference/functions.md) gives every argument of
  [`realized_group`](../reference/functions.md#realized_group).
- [Returned records](../reference/records.md) lists every field of
  `numerical.precision`, `numerical.positions` and `numerical.symmetrization`.
- [Glossary](../reference/glossary.md) gives every tolerance, with its unit and
  default.
- [Silicon from Wannier90: the group the model has](../start/tutorial-wannier90.md)
  takes a real material through these steps.
