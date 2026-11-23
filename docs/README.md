# The map of the documentation

pythTQC measures crystalline symmetry from a pythtb tight-binding model: the
space group its Hamiltonian obeys, the little-group representations its bands
carry at every high-symmetry point, whether those bands could be an atomic
limit, their symmetry-indicator class and their stable real-space invariants.
This page lists every other page of the documentation, with a line on what
each one holds. Every Python block on a page runs as written from the root of
the pythTQC source repository, except a template to be filled in with the
reader's own data. The lines under a statement that start with `#  ` are what
that statement prints.

## The first hour

- [Quick start: Kane-Mele, end to end](start/quickstart.md) measures the space
  group of Kane-Mele and the little-group representations its two occupied
  bands carry at each high-symmetry point, and tests whether those bands could
  be an atomic limit.
- [Fu-Kane-Mele: a 3D crystal, a distortion, the Z4 indicator](start/tutorial-3d.md)
  reads the space group of Fu, Kane and Mele's model on the diamond lattice,
  lowers it with a bond distortion, and measures the symmetry-indicator class
  of the valence bands in three phases of the distorted crystal.
- [Silicon from Wannier90: the group the model has](start/tutorial-wannier90.md)
  reads silicon from the data of pythtb's Wannier90 tutorial with a declared
  precision, finds the group its Hamiltonian obeys, `R-3m1'`, below diamond's
  `Fd-3m1'`, and labels its bands in that group.

## The guide

An orbital carries a site irrep, an irreducible representation of the
site-symmetry group of its Wyckoff position. A site irrep is double-valued for
a spinful model, and is a corep where the site-symmetry group has time
reversal. A set of bands carries a little-group representation at each
high-symmetry point: a corep in a group with time reversal, an irrep in one
without. The multiplicities of the little-group representations at every
high-symmetry point form the symmetry data vector of the bands. An atomic
limit is a set of bands whose states can be written as exponentially
localized, symmetric Wannier functions.

The guide has a page for each task, in the order a calculation meets them: the
group, with a page on layers; the orbitals, with a page on spin; the bands; and
what their symmetry data vector says. The last two pages, on numerical models
and on refusals, apply to the whole calculation.

- [What group does my Hamiltonian have?](guide/group.md) introduces
  `realized_group`, which returns the space group the Hamiltonian obeys, the
  Wyckoff position of each orbital and the representation of the group on the
  orbitals. Its other sections treat a Hamiltonian with less symmetry than its
  cell, time reversal as measured, a conserved operator, a model written on a
  supercell and a model with free parameters.
- [Is my 2D model on a substrate or free-standing?](guide/layers.md) shows how
  `environment=` declares whether a two-dimensional model is a layer on a
  substrate or a free-standing one, and treats a slab cut from a crystal. It
  says what the environment changes in the site irreps of the orbitals and in
  the little-group representations of the bands.
- [How do I say what my orbitals are?](guide/orbitals.md) shows how
  `characters=` declares the site irrep of an orbital. The atomic-limit table
  `realized_group` returns gives each orbital's site irrep and how it was
  determined: declared, fixed by the Hamiltonian, or assigned by the
  lowest-angular-momentum rule. The page says when a declaration is needed,
  and shows a declaration that lowers the group.
- [How does pythTQC read the spin of my model?](guide/spin.md) gives the spin
  convention, spin quantized along `+z`, that a spinful model's Hamiltonian
  follows. It shows how to convert a model whose spin is quantized along `-z`,
  and how `spin=` declares which orbitals of a Wannier90 spinor import are the
  spin states of one spinful orbital.
- [Which little-group representations do these bands carry?](guide/labels.md)
  shows how `coreps_at` returns the little-group representations a set of
  bands carries at one high-symmetry point, and `hsp_table` those at every
  one. It says which sets of bands can be labeled and gives one operation's
  sewing matrix. It says when the labels can be trusted, from how well the
  multiplets at a point are separated in energy and from how the site irreps
  of the orbitals were determined.
- [Could these bands be an atomic limit?](guide/atomic-limits.md) introduces
  `decompose`, which tests whether the symmetry data vector of a set of bands
  is a sum of those of elementary band representations (EBRs) with nonnegative
  integer multiplicities. It says what is returned where there is no such sum.
- [Which symmetry-indicator class do these bands occupy?](guide/indicators.md)
  introduces `symmetry_indicator`, which returns the class of a set of bands in
  the indicator group of its space group, for a crystal and for a layer. It
  gives the two sets of names in which a class is written.
- [What do the stable real-space invariants say about these bands?](guide/srsi.md)
  introduces `srsi_of_model`, which computes from a model's orbitals all its
  stable real-space invariants (SRSIs), and `srsi_of_bands`, which computes the
  integer-valued SRSIs (ZSRSIs) of a set of bands. It shows how `splitting`
  applies the no-go criterion of Hwang *et al.*, Nat. Commun. **17**, 8564
  (2026), which can show that the bands on the two sides of a gap cannot both
  be atomic limits.
- [How do I measure a numerical or Wannier90 model?](guide/numerical.md) shows
  how `precision=` declares a model's numerical precision. With it,
  `realized_group` moves the orbital centers onto symmetric positions, projects
  the Hamiltonian onto its symmetric part, and tells an operation obeyed within
  the precision from one the model breaks.
- [What does a refusal mean, and what do I do about it?](guide/refusals.md)
  explains a `Refusal`, which a call that cannot complete returns in place of
  its record. It sorts refusals into four kinds, and says which failures raise
  an exception instead.

## The full account

- [How pythTQC reads a Hamiltonian](background/measurement.md) says how the
  group is measured from `H(k)`, what a declaration defines, what the hoppings
  fix, why a layer's Hamiltonian does not say whether the layer lies on a
  substrate, and what a decomposition, an indicator and the no-go criterion
  each certify.
- [The topological quantum chemistry this package uses](background/tqc-primer.md)
  gives a section each, with its papers, to site symmetry, little groups and
  their irreps and coreps, band representations, symmetry indicators, fragile
  topology and the stable real-space invariants.
- [Conventions: operations, labels and band representations](background/conventions.md)
  gives the names of the operations, site irreps and little-group
  representations a printout carries, the origin the names refer to, how a band
  representation is written and the spin convention, and compares them with
  the pages of the Bilbao Crystallographic Server.
- [What the package does not cover](background/limits.md) lists each limit of
  the package, with what a call does where it meets it.
- [How the modules fit together](background/modules.md) draws the path a model
  takes through the package's modules, and says what each module does.

## The reference

- [Functions](reference/functions.md) gives every function the package
  exports, by task, with its arguments, what it returns and the checks at which
  it refuses. It is written from the docstrings.
- [Returned records](reference/records.md) gives every record a call returns,
  by task: what it holds, the call that returns it or the field that holds it,
  and every field. It is written from the docstrings.
- [Refusals](reference/refusals.md) lists every refusal by the name of its
  check, with what it means, its usual cause and what to do.
- [Glossary](reference/glossary.md) defines each term the pages and the records
  use, and gives every tolerance with its unit and default.

## Examples and citation

- [The examples](examples.md) lists the eleven scripts in `examples/`, each
  measuring one piece of physics end to end, with the page it illustrates, what
  it prints and how long it takes.
- [Citing](citing.md) lists the papers behind each method, and what to cite in
  work that uses pythTQC.

## The models the pages share

Most pages run Kane-Mele, `kane_mele` in pythtb's own collection,
`pythtb.models`. Others run `fu_kane_mele`, `graphene`, `haldane`,
`checkerboard` and `ssh` from the same collection. `pythtqc.models.lin3g` is a
spinful square layer whose lowest four bands have the symmetry data vector of
the EBR its `2c` orbitals induce, and whose next two are a fragile pair: two
bands that are not an atomic limit and become one once suitable atomic bands
are added. The guide pages on orbitals, labels, atomic limits, indicators,
SRSIs and numerical models run it. `pythtqc.models.i41_soc`, a spin-orbit
crystal in the space group `I4_1`, runs on the conventions page. The models of
the example scripts are listed with each script on
[The examples](examples.md). The one real material is silicon, from the data
of pythtb's Wannier90 tutorial.
