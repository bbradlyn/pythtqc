# pythtqc

Crystalline-symmetry diagnostics for [PythTB](https://pythtb.readthedocs.io/)
tight-binding models. Hand it a model and it measures the space group its
Hamiltonian obeys, labels its bands by symmetry at every high-symmetry point,
says whether they could be an atomic limit, and reads their symmetry indicator
and their stable real-space invariants. The space-group data is read from
[`irrep`](https://github.com/irreducible-representations/irrep)'s tables.

## In a few lines

The Kane–Mele model from pythtb's own collection, with intrinsic spin–orbit
coupling and nothing else, read as a free-standing layer:

```python
from pythtb.models import kane_mele
import pythtqc as tqc

r = tqc.realized_group(kane_mele(0.0, -1.0, 0.3, 0.0),
                       environment="free-standing", characters={0: "s"})
print(r.group.name, r.group.number_str)
#  P6/mmm1' 191.234
t = tqc.hsp_table(r.rep, bands=2)
print(tqc.format_counts(t.counts))
#  -GM9 + -K8 + -M6
print(tqc.symmetry_indicator(t, r.group, rep=r.rep))
#  SymmetryIndicator(indicated nontrivial, #191.234, Z6 x Z12, [1|6, 0|12], completed on the stack)
print(tqc.srsi_of_bands(t, r.group, rep=r.rep))
#  BandSRSIs(#191.234, (-1/6, 1/6, 1/2, 1/3, 0, 2/3, 0, 0, 0, -1/3, -2/3, 0, 1/2, 0), stable)
```

The Hamiltonian obeys `P6/mmm` with time reversal adjoined, the `1'`. Its two
occupied bands carry the coreps printed, one at each high-symmetry point, and
that content sits in the class `[1|6, 0|12]` — `1` in `ℤ6` and `0` in `ℤ12` —
of the group's indicator group: nonzero, so no atomic limit has it — the quantum spin Hall
insulator. Their stable real-space invariants are fractional, with common
denominator 6, the order of that class: no configuration of orbitals has them
([stable real-space invariants](docs/guide/srsi.md)). Read on a substrate, the
default for a two-dimensional model, the same Hamiltonian obeys only `P6mm1'`,
whose indicator group is trivial: what is under a layer is part of its physics
([layers](docs/guide/layers.md)). [The quick start](docs/start/quickstart.md)
reads every printed line of a call like this one.

## What it answers

| question | the call | the chapter |
|---|---|---|
| What space group does this Hamiltonian have? | `realized_group` | [group](docs/guide/group.md), [layers](docs/guide/layers.md) |
| What are its bands, by symmetry? | `hsp_table`, `corep_content` | [labels](docs/guide/labels.md) |
| Could they be an atomic limit? | `decompose` | [atomic limits](docs/guide/atomic-limits.md) |
| Which class do they sit in, and what do they carry in real space? | `symmetry_indicator`, `srsi_of_bands`, `srsi_of_model` | [indicators](docs/guide/indicators.md), [stable real-space invariants](docs/guide/srsi.md) |

## Install

```bash
pip install .                 # the package
pip install ".[examples]"     # ... and matplotlib, which the examples draw with
```

`irrep` reads its character tables from the separate package `irreptables`,
which is a dependency here: without it `irrep` falls back to placeholder tables,
and `tqc.tables_status()` says which tables are installed.

## Where to go next

- **A model from pythtb's collection, or one built by hand:** [the quick start](docs/start/quickstart.md),
  then [a three-dimensional crystal](docs/start/tutorial-3d.md).
- **A wannier90 model of a real material:** [silicon from wannier90](docs/start/tutorial-wannier90.md)
  and [numerical models](docs/guide/numerical.md).
- **Labels to compare with the Bilbao Crystallographic Server's:** [the conventions](docs/background/conventions.md).
- **What each call takes and what it returns:** [every function](docs/reference/functions.md),
  and [every returned record](docs/reference/records.md), field by field.
- **A call that refused:** [when a call refuses](docs/guide/refusals.md).
- **Everything, by depth:** [the map of the documentation](docs/README.md), also
  on [the documentation site](https://pythtqc.readthedocs.io/).

## Tests

```bash
pip install ".[test]"
pytest                        # the fast tier
pytest -m gates               # every page and example against what it prints, the stored baseline, and the site
pytest -m "slow or not slow"  # everything
```

## The documentation site

The pages under `docs/` are the site at
[pythtqc.readthedocs.io](https://pythtqc.readthedocs.io/), built with MkDocs and
its Material theme:

```bash
pip install ".[docs]"
mkdocs serve                  # the site, at http://127.0.0.1:8000
```

## Citing

Work that uses pythtqc cites the paper its stable real-space invariants come
from, Y. Hwang *et al.*, Nat. Commun. **17**, 8564 (2026), and the package itself:
[`CITATION.cff`](CITATION.cff) holds both, and GitHub's "Cite this repository"
reads it. The papers behind each method are listed in [citing](docs/citing.md).

## Acknowledgements

Development of these tools was supported by the National Science Foundation under
grant no. DMR-2510219.

## License

GPL-3.0-or-later; see [`LICENSE`](LICENSE).
