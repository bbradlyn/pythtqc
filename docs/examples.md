# The examples

The ten scripts in [`examples/`](../examples) each take one piece of physics
end to end: they measure it, print what they measured, and draw one figure.
This page lists them by the physics each shows, with the chapter it
illustrates, what it prints, and how long it takes. Every quantity a script
reports is measured in the run; only a model's parameters are typed into it.

| script | the physics | the chapter | time |
|---|---|---|---|
| [`tqc_fkm_diamond.py`](../examples/tqc_fkm_diamond.py) | a bond distortion lowers the group; the `ℤ4` indicator of three phases | [group](guide/group.md), [indicators](guide/indicators.md) | 65 s |
| [`tqc_kane_mele_realized_group.py`](../examples/tqc_kane_mele_realized_group.py) | a Wyckoff position that moves with the regime; a degeneracy held by a conserved operator | [group](guide/group.md), [atomic limits](guide/atomic-limits.md) | 30 s |
| [`tqc_kane_mele_three_presentations.py`](../examples/tqc_kane_mele_three_presentations.py) | one layer asked three ways; the inversion a layer group carries | [layers](guide/layers.md), [indicators](guide/indicators.md) | 50 s |
| [`tqc_3c_transported_orbitals.py`](../examples/tqc_3c_transported_orbitals.py) | the orbitals of one Wyckoff orbit as one another's images; a composite band representation | [orbitals](guide/orbitals.md) | 30 s |
| [`tqc_sg80_soc_doublet.py`](../examples/tqc_sg80_soc_doublet.py) | labelling a degenerate level away from the time-reversal-invariant momenta | [conventions](background/conventions.md), [labels](guide/labels.md) | 10 s |
| [`tqc_multifold_connectivity.py`](../examples/tqc_multifold_connectivity.py) | an elementary band representation that is connected, and one that splits | [atomic limits](guide/atomic-limits.md) | 15 s |
| [`tqc_srsi_tables.py`](../examples/tqc_srsi_tables.py) | the stable real-space invariants' tables themselves, with no model | [SRSIs](guide/srsi.md) | 25 s |
| [`tqc_srsi_fkm_two_routes.py`](../examples/tqc_srsi_fkm_two_routes.py) | a band set's invariants from its bands and from its orbitals | [SRSIs](guide/srsi.md), [indicators](guide/indicators.md) | 35 s |
| [`tqc_srsi_p4_flatband.py`](../examples/tqc_srsi_p4_flatband.py) | a splitting momentum space allows and the no-go criterion diagnoses | [SRSIs](guide/srsi.md), [spin](guide/spin.md) | 25 s |
| [`tqc_srsi_lin3g_composite_splitting.py`](../examples/tqc_srsi_lin3g_composite_splitting.py) | the no-go criterion on two gaps of one model | [SRSIs](guide/srsi.md), [atomic limits](guide/atomic-limits.md) | 10 s |

The times are wall-clock times of a default run on an ordinary laptop; the ten
together take about five minutes.

## Running one

From the root of the repository:

```bash
pip install ".[examples]"                 # the package, and matplotlib for the figures
python examples/tqc_fkm_diamond.py
```

A script runs from a checkout whether `pythtqc` itself is installed or not, as
long as its dependencies and matplotlib are. Every script imports
[`examples/_path.py`](../examples/_path.py) before the package, which leaves an
installed `pythtqc` alone and otherwise puts the checkout's `src` folder first
on the path. Two environment variables change the figure, and neither changes
anything measured:

- `TQC_EXAMPLE_FIGDIR` is the folder the figure is written to, by default
  `examples/figures/tqc_<name>/`; the last line a script prints names it.
- `TQC_EXAMPLE_QUICK=1` draws the figure on a coarser band path. In
  `tqc_sg80_soc_doublet.py` it thins the sweep of spin–orbit strengths instead,
  and in `tqc_srsi_tables.py`, whose figure tallies the tables, it changes
  nothing. The measurement dominates the time, so a quick run is not much
  faster.

```bash
TQC_EXAMPLE_QUICK=1 TQC_EXAMPLE_FIGDIR=/tmp/figures python examples/tqc_sg80_soc_doublet.py
```

The scripts save their figures with `pythtqc.utils.save_figure`, which writes a
figure without the timestamp matplotlib puts into PDF and SVG files, so a rerun
leaves the files unchanged.

Each script is a flat sequence of sections, each a few calls and the lines
that print what they measured, so a script reads top to bottom like a worked
problem; its header comment names the model and where it comes from, and lists
the sections. What each script prints is kept, line for line, in
`tests/data/example_goldens/`, and the test suite runs every script and
compares what it prints with that copy.

## The group a Hamiltonian has

### [`tqc_fkm_diamond.py`](../examples/tqc_fkm_diamond.py)

Fu, Kane and Mele's spin–orbit model on the diamond lattice, pythtb's own
`fu_kane_mele(t, soc, dt)`, with one of the four nearest-neighbour bonds
detuned along `[111]`. It prints the group of the undistorted crystal,
`Fd-3m1'` #227.129, and its four bands meeting at `X`, where the valence pair
has no label; the group of each distorted phase, `R-3m1'` #166.98, measured from
the Hamiltonian although no atom moves; each phase's valence content and its
decomposition, none in the two phases with the bond detuned by `+0.4` and
`-0.4` and an atomic limit at `3a` once three bonds are cut into decoupled
dimers; the content at the eight time-reversal-invariant momenta; and each
phase's class in `ℤ2 × ℤ4`, with `z4` and the weak indices, in Fu, Kane and
Mele's notation `(1;111)`, `(0;111)` and `(0;000)`. The figure is the bands of the three phases.
It illustrates [a descent](guide/group.md#when-the-hamiltonian-has-less-symmetry-than-its-cell),
[symmetry indicators](guide/indicators.md)
and [the three-dimensional tutorial](start/tutorial-3d.md), which takes the
same crystal step by step.

### [`tqc_kane_mele_realized_group.py`](../examples/tqc_kane_mele_realized_group.py)

Kane and Mele's honeycomb, pythtb's `kane_mele(delta, t, soc, rashba)`, in four
regimes and a deep atomic limit. It prints the group of each regime,
`P3m11'` #156.50 with a staggered potential and `P6mm1'` #183.186 without; the
content of the valence and conduction pairs at `GM`, `K` and `M` with their
decompositions, where the valence pair's Wyckoff position moves from `1a` to
`1b` as the staggered potential grows and a conduction pair that does not
decompose is written as a difference of elementary band representations; the
operator `s_z` the Hamiltonian conserves without its Rashba term, read off
`H(k)`; the level at `K` that the Rashba term splits into two one-dimensional
coreps, linearly in its strength, because the little group does not hold them
together and `s_z` did; and the deep atomic limit, where the valence pair sits
on the orbital at `1b`. The figure is the bands of the four regimes. It
illustrates [a conserved operator](guide/group.md#a-conserved-operator) and
[atomic limits](guide/atomic-limits.md).

### [`tqc_kane_mele_three_presentations.py`](../examples/tqc_kane_mele_three_presentations.py)

The same honeycomb with no staggered potential and no Rashba term, so its
horizontal mirror holds, asked three ways: as a free-standing layer with `s`
declared, as the same layer with `p_z` declared, and as a periodic stack of
decoupled layers, a three-dimensional crystal. It prints the group,
`P6/mmm1'` #191.234 each time; the valence content and the class in
`ℤ6 × ℤ12`, `(1, 0)` with `s` declared and `(5, 0)` with `p_z`, the stack with
`p_z` agreeing with the layer; the inversion the layer group carries, the product
of the horizontal mirror and the twofold rotation, whose spinor lift squares to
`+E`; the Rashba term breaking the mirror, which leaves `P6mm1'` and a trivial
indicator group; which named forms of the class read the rows a layer has no
`k_z` for, every one of them an invariant of the layer's decoupled stack; and
the four bands together, an atomic limit, at class `(0, 0)`. The figure is the
layer's bands and the stack's at `k_z = 0` and `k_z = 1/2`. It illustrates
[a two-dimensional model's environment](guide/layers.md) and
[a layer's indicator](guide/indicators.md#a-layer-whose-group-has-an-indicator).

## What the orbitals carry

### [`tqc_3c_transported_orbitals.py`](../examples/tqc_3c_transported_orbitals.py)

A `p` pair at each of the three `3c` sites of a simple cubic cell, each pair
perpendicular to its own site's fourfold axis, with random hoppings averaged
over the operations of `T_d`: `pythtqc.models.cubic_3c_perp_doublet()`. It
prints the group the point set allows, `Pm-3m1'` #221.93, and the one the
hoppings obey, `P-43m1'` #215.71, inversion broken; the atomic-limit table with
nothing declared, each row the `E` of its own site's `-42m` with its characters
named in that site's own axes, and the threefold rotation that carries one
site's characters onto the next; the line under the table that writes the
orbit's band representation, composite, as a sum of elementary ones at `1b`;
that the table is the same line for line after every pair is turned inside
itself by `0.37` rad, which moves no character; and the six bands'
decomposition. The figure is the six
bands in both bases. It illustrates [the atomic-limit table](guide/orbitals.md)
and why a function is not what the hoppings fix
([what the hoppings fix](background/measurement.md#what-the-hoppings-fix)).

## Labels, and the tables' conventions

### [`tqc_sg80_soc_doublet.py`](../examples/tqc_sg80_soc_doublet.py)

`I4_1` (space group 80) at `P = (1/4, 1/4, 1/4)` in primitive coordinates,
which is not a time-reversal-invariant momentum, so the operation that pairs
levels there is a fourfold screw combined with time reversal. On
`pythtqc.models`' `i41_atomic()` and `i41_soc("generic", lam=0.4)` it prints
the fourfold level of the spinful atomic model; the little group's table at `P`
and at `-P`, entrywise conjugates of one another; a spin–orbit bond set
splitting the level as `[1, 2, 1]`; each level labelled from one
diagonalization, the doublet `-P4P4`, whose trace `2i` on the twofold rotation
names the irrep that screw combined with time reversal glues to itself; the
four valence bands as one elementary band representation;
and the control, `P6_2` (space group 171) at `H` on `p62_atomic()`, where the
gluing pairs two different irreps. The figure is the levels at `P` against the
spin–orbit strength. It illustrates
[how the labels read across to the Bilbao Crystallographic Server's](background/conventions.md#points-where-the-gluing-differs)
and [one diagonalization reused](guide/labels.md#one-diagonalization-reused).

## Atomic limits, and connectivity

### [`tqc_multifold_connectivity.py`](../examples/tqc_multifold_connectivity.py)

Two four-band elementary band representations and whether their bands can be
separated: `s` orbitals at `8a` of `I2_13` (space group 199), from Flicker et
al. (2018), and `(d_z², d_x²−y²)` at `2a` of `P4_232` (space group 208), from
Cano et al. (2018) — `pythtqc.models`' `i213_flicker2018()` and
`p4232_cano2018()`. It prints the levels at every high-symmetry point; for space
group 199, the threefold levels at `GM` and `H` and every inter-band gap closing
somewhere on a `13 × 13 × 13` mesh, so the band representation is connected;
for space group 208, with the crystal field the paper draws, a gap at half
filling that never closes; and each isolated half asked on its own, neither of
which decomposes. The figure is both band structures. It illustrates
[what a decomposition cannot certify](guide/atomic-limits.md#what-a-decomposition-cannot-certify).

## Stable real-space invariants

### [`tqc_srsi_tables.py`](../examples/tqc_srsi_tables.py)

The tables of the stable real-space invariants of
[Hwang et al., Nat. Commun. **17**, 8564 (2026)](https://doi.org/10.1038/s41467-026-74844-w),
with no model. It prints space
group 211 without spin–orbit coupling end to end: its rows, its site keys with
which are maximal and which have an elementary band representation, and the band
count as a form in the invariants; the site irreps `B2` and `B3` at Wyckoff `d`,
which the invariants do not distinguish; two printed indicator rows of space
group 147 that are not indicators on the group's own elementary band
representations, and the derived rows returned in their place; the 211
symmetry-indicated splittings of an elementary band representation that the
paper's supplement lists, each asked again with the no-go criterion over a complete
enumeration of atomic realizations; and the 36 pairs of site irreps, over 26
space groups, that the invariants do not distinguish. The figure tallies the
verdicts and the pairs. It illustrates
[stable real-space invariants](guide/srsi.md).

### [`tqc_srsi_fkm_two_routes.py`](../examples/tqc_srsi_fkm_two_routes.py)

Fu–Kane–Mele's distorted diamond again, read in real space. It prints the
valence pair's `ℤ`-valued invariants from its bands, fractional with common
denominator 4 in the strong phase, which is the order of its indicator class;
the orbit's invariants from its orbitals, at Wyckoff `6c`, which is not a
maximal position, localized at `3a` and at `3b` with the same values; all four
bands from the bands and from the orbitals, which agree; the denominator against
the order of the class in all three phases; and the no-go criterion's verdicts,
`stable topological` in the strong and weak phases and `no Z_n SRSIs` for the
dimers. The figure is the bands of the three phases. It illustrates
[stable real-space invariants](guide/srsi.md) and
[symmetry indicators](guide/indicators.md).

### [`tqc_srsi_p4_flatband.py`](../examples/tqc_srsi_p4_flatband.py)

The two `P41'` models of Hwang et al., a flat band and a `ℤ2` model,
transcribed as hoppings from their closed forms and checked against them at 500
random `k`, their spin blocks conjugated by `iσ_x` onto pythtb's spin axis along
`+z`. A chiral next-nearest-neighbour pattern breaks the vertical mirrors so that
the hoppings obey `P41'` #75.2, the group the paper writes. It prints the
invariants read off the orbitals; what momentum space says of the two pairs,
both decomposing, with no indicator group to read; the no-go criterion's
verdict, `diagnosed topological`; and the Wilson loop of each pair, which says
which side carries the obstruction. The figure is the spectra and the Wilson
loops. It illustrates [the no-go criterion](guide/srsi.md#the-no-go-criterion)
and [a model written for the other spin axis](guide/spin.md#a-model-written-for-the-other-axis).

### [`tqc_srsi_lin3g_composite_splitting.py`](../examples/tqc_srsi_lin3g_composite_splitting.py)

`pythtqc.models.lin3g()`, the spinful model of
[Lin et al., Nat. Commun. **15**, 550 (2024)](https://doi.org/10.1038/s41467-024-44762-w):
eight bands from four orbitals at `1a` and `2c`, three elementary band
representations at two Wyckoff positions. It prints the two orbits, which share
one content and are told apart only by their `ℤ2`-valued invariants; bands 4 and
5, which decompose, and why their `ℤ_n`-valued invariants come from the orbitals
and not from the bands; and the no-go criterion on two gaps of the same
Hamiltonian: `diagnosed topological` at the gap above the sixth band, and
`invisible to the SRSIs` at the gap above the fourth. The figure is the eight
bands with both gaps marked. It illustrates
[stable real-space invariants](guide/srsi.md) and
[fragility](guide/atomic-limits.md#what-a-decomposition-cannot-certify).

## What the examples do not show

A model fitted to a calculation is the
[silicon tutorial](start/tutorial-wannier90.md) and
[numerical models](guide/numerical.md); a spinor import is
[spin](guide/spin.md); a cell that is a supercell, and time reversal measured
broken, are in [the group chapter](guide/group.md). No script prints a
refusal itself; the diamond's names the point, `X`, at which its valence pair
has no label. How to read a refusal is [when a call refuses](guide/refusals.md).
