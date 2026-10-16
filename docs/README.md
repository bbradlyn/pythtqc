# The pythtqc documentation

`pythtqc` measures crystalline symmetry from a PythTB tight-binding model: the
space group its Hamiltonian obeys, its bands' content at every high-symmetry
point, whether that content could be an atomic limit, and the invariants it
carries. This page is the map of the documentation. Every page runs as written:
its code builds its own models, and every line it shows as printed is what that
code prints.

## The path a model takes

1. **The group.** `realized_group` measures the space group the Hamiltonian
   obeys, testing every operation the cell allows on `H(k)`, and returns it
   with the orbitals' Wyckoff positions and the model's representation of the
   group. → [What group does my Hamiltonian have?](guide/group.md), and for a
   two-dimensional model
   [Is my 2D model on a substrate or free-standing?](guide/layers.md)
2. **The orbitals.** The representation's atomic-limit table says, orbital by
   orbital, the site irrep each one carries and how it is known: declared,
   fixed by the Hamiltonian, or assigned by the lowest-angular-momentum rule.
   → [How do I say what my orbitals are?](guide/orbitals.md), and for a
   spinful model [How does pythtqc read the spin of my model?](guide/spin.md)
3. **The bands.** `hsp_table` reads the corep content of a band set at every
   high-symmetry point. → [Which coreps do these bands carry?](guide/labels.md)
4. **What the content says.** `decompose` asks whether it could be an atomic limit,
   `symmetry_indicator` which class it occupies, and `srsi_of_bands` and
   `srsi_of_model` what it carries in real space.
   → [Could these bands be an atomic limit?](guide/atomic-limits.md),
   [Which symmetry-indicator class do these bands occupy?](guide/indicators.md),
   [What do the stable real-space invariants say about these bands?](guide/srsi.md)

A model fitted to a calculation, such as a `wannier90` import, takes the same
path with one declaration more:
[How do I measure a numerical or wannier90 model?](guide/numerical.md) A call
that cannot complete returns a refusal, which says why:
[What does a refusal mean, and what do I do about it?](guide/refusals.md)

## The pages, by depth

**The first hour.**

- [Quick start: Kane–Mele, end to end](start/quickstart.md) — every printed
  record read.
- [Fu–Kane–Mele: a 3D crystal, a distortion, the ℤ₄ indicator](start/tutorial-3d.md)
  — the diamond's group, the distortion that lowers it, and the indicator
  across its phases.
- [Silicon from Wannier90: the group the model has](start/tutorial-wannier90.md)
  — a real material: a declared precision, and centres that are a gauge.

**The guide, by task.**

- [What group does my Hamiltonian have?](guide/group.md)
- [Is my 2D model on a substrate or free-standing?](guide/layers.md)
- [How do I say what my orbitals are?](guide/orbitals.md)
- [How does pythtqc read the spin of my model?](guide/spin.md)
- [Which coreps do these bands carry?](guide/labels.md)
- [Could these bands be an atomic limit?](guide/atomic-limits.md)
- [Which symmetry-indicator class do these bands occupy?](guide/indicators.md)
- [What do the stable real-space invariants say about these bands?](guide/srsi.md)
- [How do I measure a numerical or wannier90 model?](guide/numerical.md)
- [What does a refusal mean, and what do I do about it?](guide/refusals.md)

**The full account.**

- [How pythtqc reads a Hamiltonian](background/measurement.md) — the ideas
  every chapter leans on, stated once.
- [The topological quantum chemistry this package uses](background/tqc-primer.md)
  — site symmetry, band representations, how an indicator is computed, and the
  stable real-space invariants, a paragraph each with the papers.
- [Conventions: operations, labels and band representations](background/conventions.md)
  — operation names, Seitz symbols, label spellings, the band-representation
  key, the spin convention, and how the Bilbao Crystallographic Server's
  spellings read across.
- [What the package cannot do yet](background/limits.md).
- Reference: [functions](reference/functions.md), every one, generated from
  the docstrings; [returned records](reference/records.md), every one and how
  to read its printout; [refusals](reference/refusals.md), every one and what
  to do about it; and the [glossary](reference/glossary.md), one name for each
  thing, with every tolerance's unit and default.

**Beyond the pages.** [The examples](examples.md), runnable scripts each
showing one piece of physics, and [citing](citing.md), the papers behind each
method.

## The models the pages share

Most pages run models from pythtb's own collection, which a PythTB user
already knows: `graphene`, `haldane`, `kane_mele`, `fu_kane_mele` and
`checkerboard`. The package's own crystals in `pythtqc.models` carry the physics
those cannot: `lin3g`, a layer whose bands hold a fragile pair over an
elementary band representation, and the pairs the examples are built on. One
real material, silicon from pythtb's `wannier90` tutorial data, carries the
numerical route.
