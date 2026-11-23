# Citing

pythTQC applies methods from the literature. The tables it reads come from
published work. This page lists the papers behind each method, table, model and
program the package uses, grouped by subject. The page
[The topological quantum chemistry this package uses](background/tqc-primer.md)
introduces these methods.

Work that uses pythTQC cites these three works, which
[`CITATION.cff`](../CITATION.cff) holds:

- the paper the stable real-space invariants come from, Y. Hwang *et al.*,
  Nat. Commun. **17**, 8564 (2026);
- the paper that carries them to photonic crystals, Y. Hwang *et al.*,
  arXiv:2601.06293 (2026);
- the package itself, by its version and its repository.

The two papers are listed in full under
[stable real-space invariants](#stable-real-space-invariants) below.

## Space groups, their representations and their tables

- C. J. Bradley and A. P. Cracknell, *The Mathematical Theory of Symmetry in
  Solids* (Clarendon Press, Oxford, 1972): the irreps of the space groups, and
  the coreps of the space groups with time reversal.
- *International Tables for Crystallography*, Volume A, *Space-group symmetry*
  (International Union of Crystallography): the space groups, their settings
  and their Wyckoff positions.
- M. I. Aroyo *et al.*, *Bilbao Crystallographic Server: I. Databases and
  crystallographic computing programs*, Z. Kristallogr. **221**, 15 (2006),
  [doi:10.1524/zkri.2006.221.1.15](https://doi.org/10.1524/zkri.2006.221.1.15).
- L. Elcoro *et al.*, *Double crystallographic groups and their
  representations on the Bilbao Crystallographic Server*, J. Appl. Cryst.
  **50**, 1457 (2017),
  [doi:10.1107/S1600576717011712](https://doi.org/10.1107/S1600576717011712):
  the double-valued irreps, and the tables of corep and irrep labels and of
  elementary band representations that the package reads.
- M. Iraola *et al.*, *IrRep: symmetry eigenvalues and irreducible
  representations of ab initio band structures*, Comput. Phys. Commun.
  **272**, 108226 (2022),
  [doi:10.1016/j.cpc.2021.108226](https://doi.org/10.1016/j.cpc.2021.108226):
  the `irrep` package, whose companion `irreptables` package holds those
  tables.

## Band representations and atomic limits

- J. Zak, *Symmetry specification of bands in solids*, Phys. Rev. Lett. **45**,
  1025 (1980), [doi:10.1103/PhysRevLett.45.1025](https://doi.org/10.1103/PhysRevLett.45.1025).
- J. Zak, *Band representations of space groups*, Phys. Rev. B **26**, 3010
  (1982), [doi:10.1103/PhysRevB.26.3010](https://doi.org/10.1103/PhysRevB.26.3010).
- B. Bradlyn *et al.*, *Topological quantum chemistry*, Nature **547**, 298
  (2017), [doi:10.1038/nature23268](https://doi.org/10.1038/nature23268): the
  elementary band representations of every space group, and the
  decomposition of a symmetry data vector into them.
- J. Cano *et al.*, *Building blocks of topological quantum chemistry:
  Elementary band representations*, Phys. Rev. B **97**, 035139 (2018),
  [doi:10.1103/PhysRevB.97.035139](https://doi.org/10.1103/PhysRevB.97.035139).
- J. Cano *et al.*, *Topology of disconnected elementary band
  representations*, Phys. Rev. Lett. **120**, 266401 (2018),
  [doi:10.1103/PhysRevLett.120.266401](https://doi.org/10.1103/PhysRevLett.120.266401),
  arXiv:1711.11045.

## Symmetry indicators

- H. C. Po, A. Vishwanath and H. Watanabe, *Symmetry-based indicators of band
  topology in the 230 space groups*, Nat. Commun. **8**, 50 (2017),
  [doi:10.1038/s41467-017-00133-2](https://doi.org/10.1038/s41467-017-00133-2):
  the indicator group `{BS}/{AI}`.
- J. Kruthoff, J. de Boer, J. van Wezel, C. L. Kane and R.-J. Slager,
  *Topological classification of crystalline insulators through band structure
  combinatorics*, Phys. Rev. X **7**, 041069 (2017),
  [doi:10.1103/PhysRevX.7.041069](https://doi.org/10.1103/PhysRevX.7.041069).
- L. Fu and C. L. Kane, *Topological insulators with inversion symmetry*,
  Phys. Rev. B **76**, 045302 (2007),
  [doi:10.1103/PhysRevB.76.045302](https://doi.org/10.1103/PhysRevB.76.045302):
  the parity criterion that the indicator `z4` of an inversion-symmetric
  crystal refines.
- Z. Song, T. Zhang, Z. Fang and C. Fang, *Quantitative mappings between
  symmetry and topology in solids*, Nat. Commun. **9**, 3530 (2018),
  [doi:10.1038/s41467-018-06010-w](https://doi.org/10.1038/s41467-018-06010-w):
  the names of the indicators, with spin-orbit coupling and without, which the
  tables of stable real-space invariants and the `irreptables` package both
  use.

## Fragile topology

- H. C. Po, H. Watanabe and A. Vishwanath, *Fragile topology and Wannier
  obstructions*, Phys. Rev. Lett. **121**, 126402 (2018),
  [doi:10.1103/PhysRevLett.121.126402](https://doi.org/10.1103/PhysRevLett.121.126402).
- B. Bradlyn, Z. Wang, J. Cano and B. A. Bernevig, *Disconnected elementary
  band representations, fragile topology, and Wilson loops as topological
  indices: An example on the triangular lattice*, Phys. Rev. B **99**, 045140
  (2019), [doi:10.1103/PhysRevB.99.045140](https://doi.org/10.1103/PhysRevB.99.045140).
- Z.-D. Song, L. Elcoro, Y.-F. Xu, N. Regnault and B. A. Bernevig, *Fragile
  phases as affine monoids: Classification and material examples*, Phys. Rev.
  X **10**, 031001 (2020),
  [doi:10.1103/PhysRevX.10.031001](https://doi.org/10.1103/PhysRevX.10.031001):
  the classification of the sets of bands whose symmetry data vector shows
  them fragile. Such a vector is an integer combination of elementary band
  representations, and no combination with nonnegative coefficients. Where none
  of the integer-valued stable real-space invariants of a set of bands takes a
  fractional value and `atomic_realizations` returns an empty list for them,
  the set's symmetry data vector is such a vector.

## Stable real-space invariants

- Z.-D. Song, L. Elcoro and B. A. Bernevig, *Twisted bulk-boundary
  correspondence of fragile topology*, Science **367**, 794 (2020),
  [doi:10.1126/science.aaz7650](https://doi.org/10.1126/science.aaz7650):
  the real-space invariants.
- Y. Hwang, V. Gupta, F. Schindler, L. Elcoro, Z. Song, B. A. Bernevig and
  B. Bradlyn, *Stable real-space invariants and topology beyond symmetry
  indicators*, Nat. Commun. **17**, 8564 (2026),
  [doi:10.1038/s41467-026-74844-w](https://doi.org/10.1038/s41467-026-74844-w),
  arXiv:2505.09697: the tables of stable real-space invariants the package
  ships, converted from the paper's supplement; the theorem that a set of bands
  has a nonzero symmetry indicator exactly when its integer-valued stable
  real-space invariants take fractional values; the no-go criterion, which can
  show that the bands on the two sides of a gap in a model cannot both be
  atomic limits; the pairs of site irreps the invariants do not distinguish;
  and the named indicators, each a linear form in the integer-valued
  invariants, in which an indicator class is returned.
- Y. Hwang, V. Gupta, A. Morales-Pérez, C. Devescovi, M. García-Díez,
  J. L. Mañes, M. G. Vergniory, A. García-Etxarri and B. Bradlyn, *Building
  blocks of topological band theory for photonic crystals*,
  [arXiv:2601.06293](https://arxiv.org/abs/2601.06293) (2026): the stable
  real-space invariants of a photonic crystal's bands, as those of the bands
  with auxiliary bands added, less those of the auxiliary bands; and the
  photonic model in space group 212 that
  [`examples/tqc_srsi_sg212_photonic.py`](../examples/tqc_srsi_sg212_photonic.py)
  builds, which [The examples](examples.md) describes. The auxiliary bands are
  added because the two transverse modes at zero frequency at Γ carry no
  little-group representation.

## The models

The pages and the example scripts build these models from `pythtb.models` and
`pythtqc.models`.

- C. L. Kane and E. J. Mele, *Quantum spin Hall effect in graphene*, Phys.
  Rev. Lett. **95**, 226801 (2005),
  [doi:10.1103/PhysRevLett.95.226801](https://doi.org/10.1103/PhysRevLett.95.226801):
  pythtb's `kane_mele`.
- L. Fu, C. L. Kane and E. J. Mele, *Topological insulators in three
  dimensions*, Phys. Rev. Lett. **98**, 106803 (2007),
  [doi:10.1103/PhysRevLett.98.106803](https://doi.org/10.1103/PhysRevLett.98.106803):
  pythtb's `fu_kane_mele`.
- F. D. M. Haldane, *Model for a quantum Hall effect without Landau levels:
  condensed-matter realization of the "parity anomaly"*, Phys. Rev. Lett.
  **61**, 2015 (1988),
  [doi:10.1103/PhysRevLett.61.2015](https://doi.org/10.1103/PhysRevLett.61.2015):
  pythtb's `haldane`.
- K.-S. Lin *et al.*, *Spin-resolved topology and partial axion angles in
  three-dimensional insulators*, Nat. Commun. **15**, 550 (2024),
  [doi:10.1038/s41467-024-44762-w](https://doi.org/10.1038/s41467-024-44762-w),
  arXiv:2207.10099: pythTQC's `lin3g`, from its supplementary section 3G.
- F. Flicker, F. de Juan, B. Bradlyn, T. Morimoto, M. G. Vergniory and
  A. G. Grushin, *Chiral optical response of multifold fermions*, Phys. Rev. B
  **98**, 155145 (2018),
  [doi:10.1103/PhysRevB.98.155145](https://doi.org/10.1103/PhysRevB.98.155145),
  arXiv:1806.09642: pythTQC's `i213_flicker2018`, from its appendix F.1.
- Cano *et al.*, Phys. Rev. Lett. **120**, 266401 (2018), above: pythTQC's
  `p4232_cano2018`.

## The software the package stands on

- T. Cole, S. Coh and D. Vanderbilt, *Python Tight Binding (PythTB)*, version
  2.0 (2025), [doi:10.5281/zenodo.12721315](https://doi.org/10.5281/zenodo.12721315):
  the tight-binding models pythTQC reads.
- `irrep` and its companion `irreptables` package, both above.
- A. Togo, K. Shinohara and I. Tanaka, *Spglib: a software library for crystal
  symmetry search*, Sci. Technol. Adv. Mater. Meth. **4**, 2384822 (2024),
  [doi:10.1080/27660400.2024.2384822](https://doi.org/10.1080/27660400.2024.2384822):
  the space group of a model's cell, found from the positions of its orbitals
  and the species of the atoms they sit on.
- S. Fredericks, K. Parrish, D. Sayre and Q. Zhu, *PyXtal: A Python library
  for crystal structure generation and symmetry analysis*, Comput. Phys.
  Commun. **261**, 107810 (2021),
  [doi:10.1016/j.cpc.2020.107810](https://doi.org/10.1016/j.cpc.2020.107810):
  the Wyckoff positions in the setting of the `irreptables` package.
- Google's [OR-Tools](https://developers.google.com/optimization): the integer
  searches for the decompositions of a symmetry data vector into elementary
  band representations, and for atomic realizations.
- G. Pizzi *et al.*, *Wannier90 as a community code: new features and
  applications*, J. Phys.: Condens. Matter **32**, 165902 (2020),
  [doi:10.1088/1361-648X/ab51ff](https://doi.org/10.1088/1361-648X/ab51ff): the
  calculations behind models read from Wannier90's files, such as the silicon
  model of
  [Silicon from Wannier90: the group the model has](start/tutorial-wannier90.md).
