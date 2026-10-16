# Silicon from Wannier90

The data of [the wannier90 tutorial](../../start/tutorial-wannier90.md): a
Wannier90 1.2 calculation of silicon, eight Wannier functions projected from
`sp³` hybrids on the two atoms of the primitive cell, four on each,
disentangled from twelve bands on a 4×4×4 k-point mesh. Energies are in eV and
lengths in Å.

| file | what it holds |
|---|---|
| `si.win` | the Wannier90 input: the lattice, the two atoms, the `sp³` projections, the disentanglement windows, the 4×4×4 mesh and the band path `L–Γ–X–K–Γ` |
| `si_hr.dat` | the Hamiltonian in real space, `H(R)` on 93 lattice vectors `R`, with each vector's Wigner–Seitz degeneracy |
| `si_centres.xyz` | the eight Wannier centres and the two atoms, in Cartesian coordinates |
| `si_band.dat`, `si_band.kpt` | Wannier90's own interpolated bands along the band path, and the path's k-points |

pythtb reads them, from the repository root, as

```python
from pythtb import W90

silicon = W90("docs/data/silicon_w90", "si").model(zero_energy=6.2285135)
```

where `6.2285135` eV is the Fermi energy pythtb's tutorial uses, so that the
four valence bands end at zero.

**Where the files come from.** They are copied unchanged from pythtb's
documentation, [github.com/pythtb/pythtb](https://github.com/pythtb/pythtb),
folder `docs/source/tutorials/silicon_w90`, where they are the data of pythtb's
own Wannier90 tutorial. pythtb ships the rest of the calculation in a subfolder
there, `unused` (the `.amn`, `.mmn`, `.eig`, `.chk` and `.wout` files), which
this copy leaves out.

**License.** pythtb is distributed under the GNU General Public License,
version 3 or later (GPL-3.0-or-later), as is this package.
