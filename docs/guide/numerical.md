# How do I measure a numerical or wannier90 model?

A Hamiltonian fitted to a calculation, such as a wannier90 import, is symmetric only to
the precision of its own numbers, several decades above machine precision. This chapter
is the route for such a model: one declaration, `precision=`, and the tolerances it
derives; the orbital centres corrected first, since moving a centre is a gauge
transformation; then the hoppings projected onto their group-invariant part; and the line
between an operation obeyed within the declaration and one the model breaks, which no
projection crosses. The models are this package's `lin3g`, a spinful square layer, with a
relative error of about `1e-7` put into every hopping, and pythtb's Kane–Mele with its
orbital centres displaced. [Silicon from wannier90](../start/tutorial-wannier90.md) takes a
real material through the same route.

## The call

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
#  RealizedGroup(P4mm1' #99.164; precision 1e-07 declared -> tol 1.74e-05 (cut 1e-06, window 0.01), position_tol 1e-07 (measured max|H| 17.42); symmetrized (defect 5.54e-07); on a substrate (the default), the free-standing group #99.164 agrees)
```

## How to read it

- **`P4mm1' #99.164`**: the group of `lin3g` without its noise, every operation obeyed
  within the declaration; the environment reads as for any layer
  ([what group does my Hamiltonian have](group.md#how-to-read-it)).
- **`precision 1e-07 declared`**: the relative precision of the model's own numbers,
  dimensionless, one fact about the model. It
  is how far those numbers miss their own symmetry, not how many decimals a file prints
  them to: silicon's wannier90 files are written to six decimals, its numbers are
  symmetric to about one part in 10⁴, and declared `1e-5` it reads `P1`, where `1e-4`
  returns its group, `R-3m1'`
  ([silicon, one declaration](../start/tutorial-wannier90.md#one-declaration-precision)).
- **`tol 1.74e-05`**: the covariance check's tolerance, how far
  `max_k ‖U H(k) U† − H(Rk)‖`, with `U` the operation's matrix on the orbitals, may reach
  and the operation still count as obeyed. It is an energy, absolute and in the units of
  `H`, so the relative precision becomes one only against the Hamiltonian's own size:
  `tol = 10 × precision × max|H|`.
- **`cut 1e-06, window 0.01`**: the measurement's two relative numbers, read against the
  same `max|H|`. A singular value at or below `cut` times the largest is a solution -- and
  so is one at or below `tol` times the square root of the number of states, where the
  Hamiltonian at a `k` is itself that small, as one band's or a Dirac point's is, so that
  the cut never discards what the check keeps -- and two eigenvectors are paired when their
  energies lie within `window` of each other, relative to `max|H|`:
  `cut = tol / max|H| = 10 × precision`, `window = clip(10·sqrt(cut), 1e-3, 0.3)`.
- **`position_tol 1e-07`**: how far an orbital centre may miss the position the group puts
  it at and still be moved onto it, in reduced coordinates, where a relative precision is
  already absolute. The centres are fitted about their own origin, a shift of all of them
  together being no displacement, and a shift of all of them together no larger than
  `position_tol` is then removed as noise, so that they come back in the model's own
  frame.
- **`measured max|H| 17.42`**: the Hamiltonian's largest entry, taken about its mean on-site
  energy, so that a constant shift of `H`, such as a Fermi energy subtracted at the import,
  moves no tolerance.
- **`symmetrized (defect 5.54e-07)`**: operations obeyed within the declaration licensed a
  projection of `H` onto its group-invariant part, which moved it by `5.54e-7` in the units
  of `H` ([then symmetrize](#then-symmetrize)).

Every derived number is on the record's `precision` field:

```python
p = rn.precision
print(f"max|H| {p['max_h']:.4g}: tol {p['tol']:.3g}, cut {p['cut']:.3g}, window {p['window']:.3g}")
#  max|H| 17.42: tol 1.74e-05, cut 1e-06, window 0.01
print(f"position_tol {p['position_tol']:.0e}, symprec {p['symprec']:.0e}, raised: {p['symprec_raised']}")
#  position_tol 1e-07, symprec 1e-05, raised: False
```

`symprec` is the tolerance of the symmetry finder, spglib, which finds the
[cell's](../background/measurement.md#the-cell-and-a-verdict-on-each-operation) operations
from the centres at all: a Cartesian length in the model's units, `1e-5` by default. The
precision leaves it alone unless the declared position noise, `precision × L` with `L` the
longest lattice vector along the periodic directions, would exceed it; the finder then runs
at `10 × precision × L`, and `symprec_raised` is `True`. Giving `tol`, `cut`, `window` or
`position_tol` beside `precision` declares the same thing twice and raises `ValueError`;
`symprec` may be given beside it, as the finder's floor.

With nothing declared, the same model is measured at the default `tol = 1e-8`, the
covariance check's tolerance for a model built from exact numbers:

```python
print(repr(tqc.realized_group(noisy, typat=[0, 0, 1, 1], characters=["s", "dx2-y2", "s", "s"])))
#  RealizedGroup(P11' #1.2; descended from #99.164 (14 of 16 coset representatives of the lattice translations broken; these are the single-valued group's, each standing for itself and its negative in the double group); on a substrate (the default), the free-standing group #1.2 agrees)
```

The noise breaks every operation but the identity and time reversal, and the group returned
is `P11'`: a true statement about the numbers handed in, and no use as a statement about
the crystal. The declaration is what separates the two.

## Exact, near and broken

Under a declared `tol`, each operation's measured covariance residual puts it in one of
three classes:

| class | residual | what it means |
|---|---|---|
| **exact** | at most `max(5e-14, \|G\|·N·ε·h)` | symmetric to the rounding a representation of `\|G\|` operations on `N` states carries, `ε` the machine precision and `h` the largest entry of `H(k)` with its energy zero kept, since rounding is relative to the numbers held. Nothing to do. |
| **near** | above that, at most `tol` | symmetric within the declaration, not to machine precision: the signal for the projection. |
| **broken** | above `tol` | the model does not obey the operation, and nothing is symmetrized into it. |

Two residuals are measured, and the larger is read: that of the best matrix found that
carries `H(k)` onto `H(gk)`, and that of the operation's own matrix in the
representation the projection is built with, whose spin factor is the tables' lift and
whose multiplets are the ones measured or declared. A spin frame, or a declared multiplet,
turned inside a site by less than the declaration is exact to the first, which turns with
it, and near to the second, which does not, and the projection turns the Hamiltonian back.

```python
from collections import Counter
s = rn.symmetrization
print(Counter(s.classes.values()), f"exact up to {s.exact_gate:.2g}, near up to {s.tol:.3g}")
#  Counter({'near': 14, 'exact': 2}) exact up to 3.5e-13, near up to 1.74e-05
```

The two exact operations are the identity and time reversal, which a real rescaling of each
hopping keeps; the other fourteen hold only within the declaration.
[`classify_operations`](../reference/functions.md#classify_operations) makes the same
classification by hand, on the operations a call measured to hold at the declared `tol`,
`r.validated`, and, handed the covariances of the representation
the projection is built with (`anchored=`), at the larger of the two.

## The centres first, because moving them is a gauge

A fitted model's orbital centres are numbers too. A hopping from orbital `i` to orbital `j`
in the cell `R` carries the Bloch phase `e^{2πi k·(R + τ_j − τ_i)}`, so a defect `δ` in the
centres `τ` goes straight into the covariance residual, at order `2π·δ·max|H|` whatever the
hoppings do, and no projection of the hoppings reaches below it. Moving a centre back costs
nothing: it is a k-dependent gauge transformation, which leaves the spectrum at every `k`
untouched, and the call measures that it did. So the centres are measured, and moved,
before the operations are put to the Hamiltonian and before any hopping is touched.

Displace one Kane–Mele centre by `1e-8` and declare nothing, and the call refuses before it
builds anything:

```python
from pythtb import Lattice
from pythtb.models import kane_mele

def with_centres(model, centres):
    """The model's own hoppings, with its orbital centres at `centres` (reduced)."""
    out = TBModel(Lattice(model.lat_vecs, centres, periodic_dirs=model.periodic_dirs),
                  spinful=True)
    out.set_onsite(model.onsite, mode="set")
    for h in model.hoppings:
        out.set_hop(h["amplitude"], h["from_orbital"], h["to_orbital"],
                    h.get("lattice_vector", [0, 0]), mode="add")
    return out

km = kane_mele(0.0, -1.0, 0.3, 0.25)                               # delta, t, soc, rashba
moved = with_centres(km, km.orb_vecs + [[1e-8, 0.0], [0.0, 0.0]])  # one centre 1e-8 off
bad = tqc.realized_group(moved, characters={0: "A1"})
print(bad.gate, f"{bad.value:.1e}", bad.context["group"].name, bad.context["number_str"],
      f"{2 * np.pi * bad.value * bad.context['max_h']:.1e}")
#  positions_not_exactly_symmetric 2.0e-08 P6mm1' 183.186 3.7e-07
```

The centres miss the cell's own operations by `2e-8` in reduced coordinates, which costs
`H(k)` a residual of about `3.7e-7`, far above the default `tol` of `1e-8`. Measured as it
stands, the model would have its operations broken by its geometry as much as by its
hoppings, so the call refuses `positions_not_exactly_symmetric`, quoting the defect and
naming the cell's group, `P6mm1' #183.186`, as an upper bound on what the Hamiltonian obeys.
Declaring the centres' tolerance is the fix:

```python
fixed = tqc.realized_group(moved, characters={0: "A1"}, position_tol=1e-6)
print(repr(fixed))
#  RealizedGroup(P6mm1' #183.186; centres moved 5e-09; on a substrate (the default), the free-standing group #183.186 agrees)
pos = fixed.positions
print(f"{pos['defect_before']:.1e} -> {pos['defect_after']:.1e}, spectrum moved "
      f"{pos['spectrum_residual']:.1e}; symmetrized: {fixed.symmetrization}")
#  2.0e-08 -> 2.2e-16, spectrum moved 1.8e-15; symmetrized: None
print(pos["frame"], f"each moved {pos['max_move']:.0e}, then both {abs(pos['offset'][0]):.0e};",
      "back where Kane-Mele has them:", np.allclose(fixed.model.orb_vecs[:, :2], km.orb_vecs,
                                                   rtol=0, atol=1e-15))
#  model each moved 5e-09, then both 5e-09; back where Kane-Mele has them: True
```

The cell's whole group is back. A shift of every centre together is a choice of
origin, so the centres are fitted about their own origin first,
and the one centre's displacement is shared between the two: each moved by `5e-9`. The
shift of both together that the fit leaves, `5e-9`, is within the declared `1e-6`, so it
is noise as well and is removed: the centres come back in the model's own frame, exactly
where Kane–Mele has them, the displaced one moved by the whole `1e-8` and the other not at
all. The defect, `2e-8`, fell to rounding, the spectrum stayed the displaced model's, and
no projection ran: the Hamiltonian needed no symmetrizing, its centres did.
`precision=` declares `position_tol` with the rest, so on a real import the one
declaration runs both steps, in this order.

A declared precision is believed, and it bounds each centre. Centres the group's
operations do not carry within it are not a defective copy of a symmetric set but a
different set: an operation that would move a centre further than the declaration allows
is broken by the centres, as one the hoppings break is, and the group descends to the
largest group of its operations that carries them within it. The centres are moved onto
that group's positions before any hopping is read:

```python
below = tqc.realized_group(moved, characters={0: "A1"}, position_tol=1e-9)
print(repr(below))
#  RealizedGroup(P21' #3.2; descended from #183.186 (20 of 24 coset representatives of the lattice translations broken by the centres; these are the single-valued group's, each standing for itself and its negative in the double group); on a substrate (the default), the free-standing group #3.2 agrees)
print(f"{below.positions['move']:.1e} asked of a centre by the whole group, "
      f"{below.positions['carried']} of its coset representatives carry them")
#  5.0e-09 asked of a centre by the whole group, 4 of its coset representatives carry them
print(tqc.hsp_table(below.rep, bands=2).counts)
#  {'-GM3GM4': 1, '-A3A4': 1, '-B3B4': 1, '-Y3Y4': 1}
```

The whole group would have moved a centre by `5e-9`, five times the declared `1e-9`. The
twofold rotation through the midpoint of the two centres carries any pair of points, so it
and time reversal are what the centres keep, and the group returned carries them exactly:
its bands are labelled in it. Each broken operation is on the record's `descent`,
`refuted_by = "centres"`, with how far it alone would move a centre and why.
The shift of both centres together that would write them in the model's own frame, `5e-9`,
is beyond the declared `1e-9` as well, so it is no noise: the centres stay where they are,
about the origin fitted to them, `positions["frame"]` reads `"fitted"`, and the labels are
read about the twofold axis nearest the model's own origin.

### Centres noisier than `symprec`

`positions_not_exactly_symmetric` is for centres close enough to the cell's operations for
the finder to find them, and far enough off to break them. Centres noisier than `symprec`
are a less symmetric set of points to the finder, and the group returned is that set's, as
low as `P1`, with no refusal at all:

```python
rng = np.random.default_rng(3)
blurred = with_centres(km, km.orb_vecs + 5e-5 * rng.normal(size=(2, 2)))  # every centre off
print(repr(tqc.realized_group(blurred)))
#  RealizedGroup(P21' #3.2; on a substrate (the default), the free-standing group #3.2 agrees)
print(repr(tqc.realized_group(blurred, precision=1e-3)))
#  RealizedGroup(P6mm1' #183.186; precision 0.001 declared -> tol 0.0297 (cut 0.01, window 0.3), position_tol 0.001 (measured max|H| 2.969), symprec raised to 0.01; centres moved 4.97e-05; on a substrate (the default), the free-standing group #183.186 agrees)
```

Kane–Mele's two centres, each displaced at random on the scale of `5e-5` in reduced
coordinates, keep one rotation: the twofold axis through their midpoint, which any pair of
points in a plane lattice has. A cell of more orbitals need keep none: silicon's eight
centres, noisy at about `5e-5` Å, leave the cell nothing but the identity, and its
undeclared import is returned `P1`
([silicon, asked without a precision](../start/tutorial-wannier90.md#asked-without-a-precision)).
Declared, the finder runs at `10 × precision × L`, here `0.01`, finds the cell's operations,
and the centres are moved onto them. A numerical model whose group comes back implausibly
low with nothing declared wants `precision=` before anything else.

## Then symmetrize

A near operation is one the declaration already calls symmetric within noise, so
projecting `H` onto its group-invariant part moves it by about the residual, inside that
same declaration, and the model that comes back passes the exact checks. `realized_group`
runs the projection at its default `symmetrize="auto"`, and the record's `symmetrization`
says how far the Hamiltonian moved:

```python
print(f"{len(s.near)} near, {len(s.exact)} exact; H moved by {s.defect:.2e}, "
      f"{s.defect_relative:.1e} relative")
#  14 near, 2 exact; H moved by 5.54e-07, 3.4e-08 relative
print(rn.source is noisy, rn.model is noisy)
#  True False
rc = tqc.realized_group(clean, typat=[0, 0, 1, 1], characters=["s", "dx2-y2", "s", "s"])
print(rc.group.name == rn.group.name,
      tqc.hsp_table(rn.rep, bands=6).counts == tqc.hsp_table(rc.rep, bands=6).counts,
      max(np.abs(rn.rep.matrices[i] - rc.rep.matrices[i]).max() for i in rc.rep.matrices))
#  True True 0.0
```

The average moved `H` by `5.5e-7`, `3.4e-8` of its size. `rn.source` is the model handed in
and `rn.model` the symmetrized one, which every later call takes. The noisy model is
returned the clean model's group, the clean model's band content at every high-symmetry
point, and representation matrices identical to the clean model's.

The projection averages `U(g;k) H(k) U(g;k)†` over every operation of the group, those
combined with time reversal included, so a time-reversal-symmetric model stays one. A
tight-binding `H(k)` is a finite Fourier series, so the average goes back to a hopping list
exactly, with no fit. The projection also runs with no operation near, where the
[multiplets](../reference/glossary.md#words-the-pages-use) a site's orbitals form hold only
within the declared precision, as when a Wannier basis is turned inside its site by less
than the precision: `symmetrization.multiplets` says how far they mix, and the projection
removes that mixing and nothing else, so the orientation of the orbitals inside each
multiplet stays the model's own. It runs, too, where
[parts of the crystal](../background/measurement.md#what-the-hoppings-fix) are coupled to
one another only below the declared precision: such a coupling counts as absent, each part
carries a one-dimensional character of its own, and the projection leaves the coupling out,
so the model returned holds the parts exactly apart; `symmetrization.parts` names them and
says how large the coupling was. And it runs where identical copies of a part of the
crystal, a pseudospin the Hamiltonian conserves within the declared precision, are coupled
or told apart only below it: the projection holds them as exact copies, so a declaration
mixing them, two identical `s` orbitals declared the doublet `E`, is carried on the model
returned; `symmetrization.copies` names their orbitals and says how far they were from
exact. Two identical lattices coupled on site above the precision are a bonding and an
antibonding lattice, each carrying a representation of its own; where something joins the
two only below the precision, a bond of one lattice off the other's, the projection leaves
it out, and `symmetrization.shares` names their orbitals and says how large it was.

## A broken operation is never symmetrized into

Scale one `lin3g` hopping by `1.01` instead. The operations the change touches are broken by
far more than the declaration and the others stay exact, so none is near, the projection
never runs, and the group
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
#  RealizedGroup(Pmm21' #25.58; precision 1e-07 declared -> tol 1.74e-05 (cut 1e-06, window 0.01), position_tol 1e-07 (measured max|H| 17.42); descended from #99.164 (8 of 16 coset representatives of the lattice translations broken; these are the single-valued group's, each standing for itself and its negative in the double group); on a substrate (the default), the free-standing group #25.58 agrees)
print(rb.symmetrization, sorted({b["name"] for b in rb.descent["dropped"]}))
#  None ['C4(z)', "C4(z)'", 'C4^-1(z)', "C4^-1(z)'", 'm([1-10])', "m([1-10])'", 'm([110])', "m([110])'"]
```

The fourfold rotations and the diagonal mirrors, each with its partner combined with time
reversal, are broken and named on the record's `descent`, and `symmetrization` is `None`. The
declared precision is what separates near from broken, which is why it is declared.
[`symmetrize`](../reference/functions.md#symmetrize), called by hand on a record
whose operations include one broken at the `tol` it is given, refuses
`symmetrize_broken_operation` rather than impose a symmetry the model does not have.

## A large model

The projection holds `H(k)` and its group average together on a grid set by the hoppings'
range along each periodic direction, alike along directions the group's operations mix,
two arrays of `16 · n_points · n_states²` bytes for `n_points` points and `n_states`
states, and `max_bytes=`, `1e9` bytes by default, is its budget at the peak. A coarser grid
would alias, so a projection past the budget is not run: the call returns
`projection_too_large`, naming the group identified and the `max_bytes=` that runs it. Only
a model with many states and long-ranged hoppings reaches it. The measurement of the
operations keeps a budget of its own, which `max_bytes=` raises where a spectrum's exactly
degenerate levels need more, and past that the call returns `detector_too_large`. Either
way the same call with the number the refusal names runs it.

The measurement of the operations also costs more the looser the declared precision. Two
eigenvectors are paired where their energies lie within `window` of each other, and
`window` grows as the square root of the precision, up to `0.3`; on a model with many
states the pairs it makes, and the memory and time they take, grow with it. So declare the
precision the model's numbers hold, not a looser one: on a model of a hundred states,
declaring one a hundred times looser takes about twice the memory and several times as
long.

## When it refuses

- [`positions_not_exactly_symmetric`](../reference/refusals.md#positions_not_exactly_symmetric):
  centres off the cell's operations with no precision declared, by enough to break them,
  or, where an operation reads near-symmetric, by more than the projection's operators
  carry them to.
- [`rep_not_covariant`](../reference/refusals.md#rep_not_covariant): near-symmetric
  operations meeting the exact check unprojected, with `symmetrize=False`.
- [`projection_too_large`](../reference/refusals.md#projection_too_large) and
  [`detector_too_large`](../reference/refusals.md#detector_too_large): a memory budget.
- [`operations_not_consistent`](../reference/refusals.md#operations_not_consistent) and
  [`operations_not_nearly_consistent`](../reference/refusals.md#operations_not_nearly_consistent):
  operations read off numerical centres, with no precision declared, that do not close and
  cannot be rebuilt; declared, the centres are made exact before the setting is read,
  unless they cannot be read as a symmetric set at all, as when two sites of one species
  sit closer together than the symmetry finder can tell apart.
- The projection's own checks,
  [`symmetrize_broken_operation`](../reference/refusals.md#symmetrize_broken_operation),
  [`symmetrize_defect_exceeds_tol`](../reference/refusals.md#symmetrize_defect_exceeds_tol)
  and [`symmetrize_range_not_closed`](../reference/refusals.md#symmetrize_range_not_closed).

[When a call refuses](refusals.md) reads a refusal through.

## Further

A spinor calculation arrives from pythtb's importer as a spinless model with the spin inside
the orbital index, and `spin=` beside `precision=` declares which orbitals are the up and
down states of one Wannier function ([a wannier90 spinor import](spin.md#a-wannier90-spinor-import)).
Every argument: [`realized_group`](../reference/functions.md#realized_group), and the two
passes by hand, [`symmetrize_positions`](../reference/functions.md#symmetrize_positions) and
[`symmetrize`](../reference/functions.md#symmetrize). Every field of the record's
`precision`, `positions` and `symmetrization`: [the records reference](../reference/records.md).
Every tolerance, with its unit and default: [the glossary](../reference/glossary.md). A real
material on this route: [silicon from wannier90](../start/tutorial-wannier90.md).
