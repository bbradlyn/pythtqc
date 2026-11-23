# The examples

The eleven scripts in [`examples/`](../examples) each take one piece of physics
end to end. A script computes it, prints what it computed, and draws one
figure. Every quantity a script prints is computed when it runs. Only a model's
parameters are typed into it. This page lists the scripts by the physics each
shows, with what it prints and how long it takes. Each script's section names
the pages it illustrates.

| script | the physics | time |
|---|---|---|
| [`tqc_fkm_diamond.py`](../examples/tqc_fkm_diamond.py) | a bond distortion lowers the group; the symmetry-indicator class of three phases, in `Z2 x Z4` | 65 s |
| [`tqc_kane_mele_realized_group.py`](../examples/tqc_kane_mele_realized_group.py) | a decomposition that moves between Wyckoff positions with the staggered potential; a degeneracy held by a conserved operator | 30 s |
| [`tqc_kane_mele_three_presentations.py`](../examples/tqc_kane_mele_three_presentations.py) | one layer presented three ways; the inversion of a free-standing layer, which the twofold rotation about `z` does not replace | 50 s |
| [`tqc_3c_transported_orbitals.py`](../examples/tqc_3c_transported_orbitals.py) | the orbitals at the three sites of a Wyckoff position as images of one another; a composite band representation | 30 s |
| [`tqc_sg80_soc_doublet.py`](../examples/tqc_sg80_soc_doublet.py) | labeling a degenerate level away from the time-reversal-invariant momenta | 10 s |
| [`tqc_multifold_connectivity.py`](../examples/tqc_multifold_connectivity.py) | an elementary band representation that is connected, and one that splits | 15 s |
| [`tqc_srsi_tables.py`](../examples/tqc_srsi_tables.py) | the tables of the stable real-space invariants, with no model | 25 s |
| [`tqc_srsi_fkm_two_routes.py`](../examples/tqc_srsi_fkm_two_routes.py) | a band set's invariants from its bands and from its orbitals | 35 s |
| [`tqc_srsi_p4_flatband.py`](../examples/tqc_srsi_p4_flatband.py) | two sets of bands whose symmetry data vectors are those of atomic limits, and which cannot both be atomic limits | 25 s |
| [`tqc_srsi_lin3g_composite_splitting.py`](../examples/tqc_srsi_lin3g_composite_splitting.py) | the no-go criterion on two gaps of one model | 10 s |
| [`tqc_srsi_sg212_photonic.py`](../examples/tqc_srsi_sg212_photonic.py) | the invariants of photonic bands, all the bands less auxiliary ones | 10 s |

The times are wall-clock times of a default run on an ordinary laptop. The
eleven together take about five minutes.

## Running one

From the root of the repository:

```bash
pip install ".[examples]"                 # the package, and matplotlib for the figures
python examples/tqc_fkm_diamond.py
```

A script runs from a checkout whether or not `pythtqc` is installed, as long as
the package's dependencies and matplotlib are. Two environment variables change
the figure, and neither changes anything measured:

- `TQC_EXAMPLE_FIGDIR` is the folder the figure is written to. By default it is
  a folder under `examples/figures/` named after the script. The last line a
  script prints names it.
- `TQC_EXAMPLE_QUICK=1` draws the figure on a coarser band path. In
  `tqc_sg80_soc_doublet.py` it thins the sweep of spin-orbit strengths instead.
  In `tqc_srsi_tables.py`, whose figure tallies the tables, it changes nothing.
  The measurement takes most of the time, so a quick run is not much faster.

```bash
TQC_EXAMPLE_QUICK=1 TQC_EXAMPLE_FIGDIR=/tmp/figures python examples/tqc_sg80_soc_doublet.py
```

The scripts save their figures as PDFs with `pythtqc.utils.save_figure`, which
leaves out the timestamp, so a rerun leaves the file unchanged.

A script is a sequence of sections, numbered or lettered, each a few calls and
the lines that print what they computed, so it reads from top to bottom like a
worked problem. Its header comment names the model and lists the sections. Most
lines a script prints open with the name of the quantity they hold. A script
counts bands from 0, as pythtb does, and says so once where it names a band.
What each script prints is kept, line for line, in
`tests/data/example_goldens/`.

## The group a Hamiltonian has

### [`tqc_fkm_diamond.py`](../examples/tqc_fkm_diamond.py)

Fu, Kane and Mele's spin-orbit model on the diamond lattice, pythtb's own
`fu_kane_mele(t, soc, dt)`, with one of the four nearest-neighbor bonds detuned
along `[111]`. It shows a bond distortion lowering the group with no atom
moved, and the symmetry-indicator class of three phases. The strong phase has
the bond detuned by `+0.4`, and the weak phase by `-0.4`. The dimer phase has
the other three bonds set to zero, which leaves dimers along `[111]` coupled
only by the spin-orbit hopping. It prints:

- the group of the undistorted crystal, `Fd-3m1'` #227.129, and its four bands
  meeting at `X`. The valence pair holds half of that fourfold level, and so has
  no corep multiplicities at `X`.
- the group of each distorted phase, `R-3m1'` #166.98, measured from the
  Hamiltonian with no atom moved;
- each phase's valence symmetry data vector, its corep multiplicities at every
  high-symmetry point, and its decompositions. A decomposition is a sum of
  elementary band representations, with nonnegative integer multiplicities,
  that has the bands' symmetry data vector. The strong and the weak phases have
  none, and the dimer phase has an atomic limit at `3a`.
- the time-reversal-invariant momenta at which the strong and the weak phases'
  coreps differ from the dimers': `T` in the strong phase, `F` and `T` in the
  weak;
- each phase's class in `Z2 x Z4`, with `z4` and the weak indices, and in Fu,
  Kane and Mele's notation `(1;111)`, `(0;111)` and `(0;000)`.

The figure is the bands of the three phases. It illustrates
[when the Hamiltonian has less symmetry than its cell](guide/group.md#when-the-hamiltonian-has-less-symmetry-than-its-cell)
and [Which symmetry-indicator class do these bands occupy?](guide/indicators.md).
[Fu-Kane-Mele: a 3D crystal, a distortion, the Z4 indicator](start/tutorial-3d.md)
takes the same crystal step by step.

### [`tqc_kane_mele_realized_group.py`](../examples/tqc_kane_mele_realized_group.py)

Kane and Mele's honeycomb, pythtb's `kane_mele(delta, t, soc, rashba)`, in four
regimes of the staggered potential `delta` and the Rashba term, and in a deep
atomic limit. The layer is read on a substrate, the default, so it has no
horizontal mirror. It shows the valence pair's decomposition moving from one
Wyckoff position to another as the staggered potential grows, and a degeneracy
held by a conserved operator. It prints:

- the group of each regime: `P3m11'` #156.50 at `delta = 1.0` and `2.5`, each
  with the Rashba term, and `P6mm1'` #183.186 at `delta = 0`, without and with
  the Rashba term;
- the corep multiplicities of the valence and the conduction pairs at Γ, `K` and
  `M`, with their decompositions. The valence pair's decomposition moves from
  `1a` to `1b` as `delta` grows from `1.0` to `2.5`. In every regime but
  `delta = 2.5`, the conduction pair has no decomposition, and is written as a
  sum of elementary band representations with a negative coefficient.
- the operator `s_z`, which the Hamiltonian conserves without its Rashba term,
  read off `H(k)`;
- the level at `K` that the Rashba term splits into two one-dimensional coreps,
  linearly in its strength. The two coreps are complex conjugates, which `s_z`
  holds together while it is conserved.
- the deep atomic limit at `delta = 50`, where the valence pair sits on the
  orbital at `1b`, which carries 99.96 % of its weight.

The figure is the bands of the four regimes. It illustrates
[a conserved operator](guide/group.md#a-conserved-operator) and
[Could these bands be an atomic limit?](guide/atomic-limits.md).

### [`tqc_kane_mele_three_presentations.py`](../examples/tqc_kane_mele_three_presentations.py)

The same honeycomb with no staggered potential and no Rashba term. Without the
Rashba term its Hamiltonian commutes with the horizontal mirror. It is
presented three ways: as a free-standing layer with `s` declared, as the same
layer with `p_z` declared, and as a periodic stack of decoupled layers, a
three-dimensional crystal with `p_z` declared. The mirror is in the layer's
group because the layer is declared free-standing. It shows the declared
orbital moving the indicator class. It also shows that the inversion, with its
parity eigenvalues, belongs to a free-standing layer, and that the twofold
rotation about `z` does not replace it. It prints:

- the group, `P6/mmm1'` #191.234, each time;
- the valence class in `Z6 x Z12`: `(1, 0)` with `s` declared and `(5, 0)` with
  `p_z`. The stack's class is the same as the layer's with `p_z`.
- the inversion, the product of the horizontal mirror and the twofold rotation
  about `z`. Its spinor lift squares to `+E`, so its eigenvalues are `+1` and
  `-1`. The twofold rotation acts on in-plane `k` as inversion does, but its
  lift squares to `Ebar`, with eigenvalues `+i` and `-i`.
- the Rashba term breaking the mirror, which leaves `P6mm1'` #183.186 and a
  trivial indicator group;
- the named indicators of the class. Of them, `z2w,3`, `z3m,pi` and `z6m,pi`
  depend only on the coreps at `k_z = 1/2`, while `z4` and `z12` depend on those
  at both `k_z = 0` and `k_z = 1/2`.
- the four bands together, an atomic limit, at class `(0, 0)`.

The figure is the layer's bands, and the stack's at `k_z = 0` and `k_z = 1/2`.
It illustrates
[Is my 2D model on a substrate or free-standing?](guide/layers.md) and
[a layer whose group has an indicator](guide/indicators.md#a-layer-whose-group-has-an-indicator).

## What the orbitals carry

### [`tqc_3c_transported_orbitals.py`](../examples/tqc_3c_transported_orbitals.py)

A `p` pair at each of the three `3c` sites of a simple cubic cell, each pair
perpendicular to its own site's fourfold axis, with random hoppings averaged
over the operations of `T_d`: `pythtqc.models.cubic_3c_perp_doublet()`. It
shows the orbitals at the three `3c` sites as images of one another, and a
composite band representation. It prints:

- the group the point set allows, `Pm-3m1'` #221.93, and the group the hoppings
  obey, `P-43m1'` #215.71, with inversion broken;
- the atomic-limit table with nothing declared, where the orbitals at `3c`
  carry the `E` of their site symmetry group `-42m`, assigned from the
  Hamiltonian;
- each site's characters, `-2` on the twofold rotation about the site's own
  axis, and the threefold rotation that carries one site's characters onto the
  next;
- the line under the table that writes the band representation of the orbitals
  at `3c`, which is composite, as a sum of elementary ones at `1b`;
- the same table, line for line, after the two orbitals of every pair are
  rotated into each other by `0.37` rad;
- the six bands' decomposition, `T1` and `T2` at `1b`.

The figure is the six bands in both bases. It illustrates
[How do I say what my orbitals are?](guide/orbitals.md) and
[what the hoppings fix](background/measurement.md#what-the-hoppings-fix):
rotating the two orbitals of a pair into each other is a change of basis, and a
character, being a trace, is the same in every basis.

## Labels, and the tables' conventions

### [`tqc_sg80_soc_doublet.py`](../examples/tqc_sg80_soc_doublet.py)

`I4_1` (space group 80) at `P = (1/4, 1/4, 1/4)` in primitive coordinates.
`P` is not a time-reversal-invariant momentum, so the operation that glues
irreps into coreps there is a fourfold screw combined with time reversal. The
models are `pythtqc.models`' `i41_atomic()` and `i41_soc("generic", lam=0.4)`.
The script shows how a degenerate level away from the time-reversal-invariant
momenta is labeled. It prints:

- the fourfold level at `P` of the spinful atomic model;
- the characters of the coreps at `P`, and the rows the irreptables package
  lists for `P`, which are the coreps at `-P`. Each row of one table is the
  complex conjugate of a row of the other.
- generic spin-orbit hoppings, averaged over the group, splitting the level as
  `1 + 2 + 1`;
- each level labeled from one diagonalization. The doublet is `-P4P4`, whose
  trace `2i` on the twofold rotation names the irrep that the screw combined
  with time reversal glues to itself.
- the four valence bands as one elementary band representation, `-1E2E` at
  `4a`;
- a control, `P6_2` (space group 171) at `H` on `p62_atomic()`, where the corep
  `-H5H6` glues two different irreps.

The figure is the levels at `P` against the spin-orbit strength. It illustrates
[points where the gluing differs](background/conventions.md#points-where-the-gluing-differs),
on how these labels compare with those of the Bilbao Crystallographic Server,
where a label the server lists at `P` is the label returned here at `-P`. It
also illustrates
[one diagonalization, reused](guide/labels.md#one-diagonalization-reused).

## Atomic limits, and connectivity

### [`tqc_multifold_connectivity.py`](../examples/tqc_multifold_connectivity.py)

Two four-band elementary band representations, and whether their bands can be
separated. The first is an `s` orbital at each `8a` site of `I2_13` (space
group 199), from Flicker et al. (2018): `pythtqc.models.i213_flicker2018()`.
The second is a `(d_z2, d_x2-y2)` pair at each `2a` site of `P4_232` (space
group 208), from Cano et al. (2018): `pythtqc.models.p4232_cano2018()`. It
prints:

- the levels at every high-symmetry point of both;
- for space group 199, the threefold levels at Γ and `H`, and the smallest
  direct gap between each pair of neighboring bands, `0.000000` on a
  `13 x 13 x 13` mesh, so that the band representation is connected;
- for space group 208, with the crystal field the paper adds, 24 atoms that
  carry no orbital: a gap at half filling that stays open on that mesh;
- the two halves of the space group 208 bands, each on its own, neither of
  which decomposes.

The figure is both band structures. It illustrates
[what a decomposition cannot certify](guide/atomic-limits.md#what-a-decomposition-cannot-certify).

## Stable real-space invariants

A stable real-space invariant (SRSI) is a linear combination of the site-irrep
multiplicities of an atomic limit that no smooth, symmetric deformation
changes. Each is an integer or an integer modulo n. The integer-valued SRSIs
(ZSRSIs) of a set of bands are fixed by its symmetry data vector. The SRSIs
valued modulo n are not, and are computed from the orbitals of a full model. An
atomic realization of a set of bands is a set of site irreps at maximal Wyckoff
positions, with nonnegative integer multiplicities, that has the bands' ZSRSIs.
The no-go criterion, `splitting`, takes the bands below a gap and those above
it. It returns `nsit`, true where both sides have atomic realizations and no
pair of a realization of each has SRSIs valued modulo n adding up to the whole
model's. The two sides then cannot both be atomic limits.
[What do the stable real-space invariants say about these bands?](guide/srsi.md)
defines them.

### [`tqc_srsi_tables.py`](../examples/tqc_srsi_tables.py)

The tables of the SRSIs of
[Hwang et al., Nat. Commun. **17**, 8564 (2026)](https://doi.org/10.1038/s41467-026-74844-w),
read with no model. It shows what the tables hold, and the no-go criterion run
over every splitting they list. It prints:

- space group 211 without spin-orbit coupling: its 10 SRSIs, and its 29 site
  irreps, with how many sit at maximal Wyckoff positions and which induce an
  elementary band representation;
- the site irreps `B2` and `B3` at Wyckoff position `12d`, which the SRSIs do
  not distinguish;
- the two indicator forms the paper prints for space group 147, `z2,3` and
  `z4`, which do not vanish on an elementary band representation of the group,
  and the forms pythTQC derives from the symmetry indicator, which
  `indicators_of` evaluates in their place;
- how many of the 390 indicator forms the paper prints over all the tables
  differ from the derived ones;
- the 211 splittings of an elementary band representation that the symmetry
  indicators allow and the paper's supplement lists, and the no-go criterion on
  each, over a complete enumeration of atomic realizations;
- the 36 pairs of site irreps, over 26 space groups, that the SRSIs do not
  distinguish.

The figure tallies the criterion's verdicts and the pairs. It illustrates
[What do the stable real-space invariants say about these bands?](guide/srsi.md).

### [`tqc_srsi_fkm_two_routes.py`](../examples/tqc_srsi_fkm_two_routes.py)

Fu, Kane and Mele's distorted diamond again, read in real space. It shows a
band set's SRSIs computed from its bands and from its orbitals. It prints:

- the valence pair's ZSRSIs from its bands, fractional with common denominator
  4 in the strong phase, which is the order of its indicator class;
- the ZSRSIs of the model's orbitals, at Wyckoff position `6c`, which is not
  maximal, and the same values with those orbitals moved to `3a` and to `3b`;
- the ZSRSIs of all four bands from the bands and from the orbitals, which
  agree;
- the denominator against the order of the class in all three phases, equal in
  each;
- the no-go criterion, `nsit` false in all three phases. In the strong and the
  weak phases the valence pair's ZSRSIs are fractional, so it has no atomic
  realization. `R-3m1'`, the group of all three phases, has no SRSI valued
  modulo n.

The figure is the bands of the three phases. It illustrates
[What do the stable real-space invariants say about these bands?](guide/srsi.md)
and [Which symmetry-indicator class do these bands occupy?](guide/indicators.md).

### [`tqc_srsi_p4_flatband.py`](../examples/tqc_srsi_p4_flatband.py)

The two `P41'` models of Hwang et al., Nat. Commun. **17**, 8564 (2026),
supplementary section 5A: a flat-band model and a Z2 model. Each is transcribed
as hoppings from its closed form and checked against it at 500 random `k`. The
paper quantizes spin along `-z`. Each model's spin blocks are conjugated by
`iσ_x`, so that spin is along `+z`, as pythtb has it. A chiral pattern of
diagonal hoppings breaks the vertical mirrors, so that the hoppings obey `P41'`
#75.2, the group the paper writes. The script shows a splitting into two sets
of bands whose symmetry data vectors are those of atomic limits, and which the
no-go criterion shows cannot both be atomic limits. It prints:

- the group as transcribed, `P4mm1'` #99.164, and with the chiral pattern,
  `P41'` #75.2;
- the SRSIs computed from the orbitals;
- the symmetry data vectors of the flat-band model's two pairs of bands, each
  that of an atomic limit at `1a`, and the trivial indicator group;
- the no-go criterion, `nsit` in both models;
- the Wilson loop of each pair. In the flat-band model only the lower pair
  winds, and in the Z2 model both do. So `nsit` says that the two pairs cannot
  both be atomic limits, and not which of them is not one.

The figure is the spectra and the Wilson loops. It illustrates
[the no-go criterion](guide/srsi.md#the-no-go-criterion) and
[a model written for the other axis](guide/spin.md#a-model-written-for-the-other-axis).

### [`tqc_srsi_lin3g_composite_splitting.py`](../examples/tqc_srsi_lin3g_composite_splitting.py)

`pythtqc.models.lin3g()`, the spinful model of
[Lin et al., Nat. Commun. **15**, 550 (2024)](https://doi.org/10.1038/s41467-024-44762-w).
Its eight bands come from four spinful orbitals at `1a` and `2c`, and their
band representation is a sum of three elementary band representations at two
Wyckoff positions. It shows the no-go criterion on two gaps of one Hamiltonian.
It prints:

- the orbitals at its two positions, whose band representations have the same
  symmetry data vector and whose SRSIs valued modulo 2 differ;
- bands 4 and 5, whose symmetry data vector is that of the atomic limit `-E2`
  at `1b`, and their ZSRSIs. A signed sum of site irreps,
  `(-E1)_a + (-E2)_a - (-E1)_b`, induces the same symmetry data vector. Its
  SRSIs valued modulo 2 differ from those of `-E2` at `1b`.
- the no-go criterion at the gap between bands 5 and 6, `nsit`;
- the no-go criterion at the gap between bands 3 and 4, `nsit` false, with two
  pairs of a valence and a conduction realization whose SRSIs valued modulo 2
  add up to the whole model's.

The figure is the eight bands with both gaps marked. It illustrates
[What do the stable real-space invariants say about these bands?](guide/srsi.md)
and [what a decomposition cannot certify](guide/atomic-limits.md#what-a-decomposition-cannot-certify).

### [`tqc_srsi_sg212_photonic.py`](../examples/tqc_srsi_sg212_photonic.py)

The space group 212 (`P4_332`) tight-binding model of
[Hwang et al., arXiv:2601.06293](https://arxiv.org/abs/2601.06293). It has two
orbitals carrying `E` at each `4a` site. Its eight bands stand for a photonic
crystal's four lowest bands together with four auxiliary bands, and its
energies for `ω^2`. It shows the SRSIs of the photonic bands as those of all the
bands less those of the auxiliary ones. It prints:

- the group of the Hamiltonian, `P4_3321'` #212.60, with the basis of `E` at one
  site declared by a generator matrix;
- the coreps of all eight bands at the high-symmetry points, whose symmetry data
  vector is that of the elementary band representation of `E` at `4a`;
- two choices of auxiliary bands, `A1` at `4a` and `A1` at `4b`, which have the
  same symmetry data vector;
- the SRSIs of all eight bands, of each choice of auxiliary bands, and of their
  difference: ZSRSIs `(-1, 0, 1)` for both choices, with an SRSI valued modulo 2
  that tells the two apart.

The figure is the band structure, each multiplet at a high-symmetry point
labeled with its corep. It illustrates
[What do the stable real-space invariants say about these bands?](guide/srsi.md)
and [which basis a doublet is](guide/orbitals.md#which-basis-a-doublet-is).

## What the examples do not show

A model from a Wannier90 calculation is in
[Silicon from Wannier90: the group the model has](start/tutorial-wannier90.md)
and in [How do I measure a numerical or Wannier90 model?](guide/numerical.md).
A Wannier90 calculation with spin-orbit coupling, which pythtb imports as a
spinless model with its spin in the orbital index, is in
[How does pythTQC read the spin of my model?](guide/spin.md). A cell that is a
supercell, and time reversal measured broken, are in
[What group does my Hamiltonian have?](guide/group.md). The diamond's script
prints that the corep multiplicities of its valence pair are refused at `X`,
where the pair holds half of the fourfold level. The refusal is
`multiplet_split`, and no script prints a refusal's reason. How to read a
refusal is in
[What does a refusal mean, and what do I do about it?](guide/refusals.md).
