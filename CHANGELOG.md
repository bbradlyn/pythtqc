# Changelog

## Unreleased

- The package's name is written pythTQC; it is installed and imported as `pythtqc`, as
  before.
- `CITATION.cff` asks work that uses the package to cite Hwang et al., arXiv:2601.06293,
  which carries the stable real-space invariants to photonic crystals, beside the paper
  they come from and the package itself.
- The script that converts the stable real-space invariants' tables from the paper's TeX
  supplement, and a benchmark of declared content, leave the repository. The shipped tables
  are unchanged but for their provenance, which names the paper and the versions they were
  converted against and no script or source file. `srsi_data` says to reinstall the package,
  and `srsi_ebr_basis` names the `irreptables` version the tables were converted against.
- An eleventh example, `tqc_srsi_sg212_photonic.py`: the space group 212 tight-binding model
  of Hwang et al., arXiv:2601.06293, whose eight bands stand for a photonic crystal's four
  lowest bands and four auxiliary bands. It prints the coreps of the eight bands, the
  auxiliary bands' symmetry data vector, and the stable real-space invariants of the
  physical bands.
- `characters=` takes generator matrices, a mapping from operations of a site group to the
  matrices they act with on the orbitals of a key,
  `{(0, 1): {"C4(z)": [[0, -1], [1, 0]], "m(x)": [[-1, 0], [0, 1]]}}`. The representation
  returned gives those orbitals the declared matrices, which fixes the labelling where every
  labelling gives the site one irrep, as for `E` at `4a` of `P4_332`.
- `decompose`, `symmetry_indicator`, `completeness_check`, `regular_completion` and
  `srsi_of_bands` read a table in the group it carries: `decompose(t)` takes the table
  `hsp_table` returned and nothing else, the group and `rep=` being optional and held to
  the table's own where they are handed in; a content written out by hand still takes its
  group beside it. A layer's indicator asked of its table measures its rows out of the
  plane on the decoupled stack with no `rep=` passed.
- `SRSITables.formulas()` writes every stable real-space invariant's formula out, a line
  each, `theta_2 (Z2) = 1 (-E)_b`; `SRSIRow.summary()` says multiplicity where it said
  copies.
- The ten examples print as a ledger: each measured quantity on a labelled line, a
  record's summary whole and once, every verdict in words, bands counted from 0 and
  said so once per script, the valence and the conduction bands named in words. The
  summaries of `ModelSRSIs`, `Splitting`, `Realizations`, `BandSRSIs` and
  `SRSITables` define nothing and name the two sides of a gap in words, each
  atomic realization and each consistent pair written as the orbitals it is; the
  criterion's `reason` says the same in the same words. No verdict word and no field
  changes.
- An `HSPTable` carries the representation it was read with, `t.rep`, and
  `decompose`, `symmetry_indicator`, `completeness_check`, `regular_completion`
  and `srsi_of_bands` raise `ValueError` when handed the table with another group
  or another representation, as they do for `bands=` naming other bands; a table
  read in one group was decomposed in another with no refusal before.
- The package is re-sorted into thirty-four modules in one acyclic order, each reading
  only the ones beneath it; `groups` and `symmetry` are assembled from the private
  modules below them, and every public name stays where it was.
- `frontier_subspace` and its record `Frontier` leave `pythtqc.sewing` and the
  top level: nothing in the package called them.

## 0.1.0a2 — the first public version

pythtqc reads the crystalline symmetry of a PythTB tight-binding model:

- **The space group the Hamiltonian obeys.** `realized_group` tests every operation
  the cell allows on `H(k)` and returns the group with each orbital's Wyckoff
  position and site irrep, for three-dimensional crystals and for layers on a
  substrate or free-standing, spinless or with spin, built by hand or imported
  from wannier90 with a declared precision.
- **The bands, labelled.** `hsp_table` and `corep_content` read the corep content
  of a set of bands at every high-symmetry point, and `sewing_matrix` the matrix
  of one operation on them.
- **Atomic limits.** `decompose` asks whether that content is a sum of elementary
  band representations, with `completeness_check` and `regular_completion`
  beside it.
- **Invariants.** `symmetry_indicator` reads the indicator class, and
  `srsi_of_bands`, `srsi_of_model` and `splitting` the stable real-space
  invariants, from the tables of Hwang et al., Nat. Commun. **17**, 8564 (2026).

Every record a call returns prints itself with `summary()`; a call that cannot
answer returns a `Refusal` that says why. The space-group data is read from
`irrep`'s tables, and every page of the documentation runs as written.
