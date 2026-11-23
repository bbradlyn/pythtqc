# pythTQC

Crystalline-symmetry diagnostics for [pythtb](https://pythtb.readthedocs.io/)
tight-binding models. pythTQC takes a pythtb model and returns the space group
its Hamiltonian obeys. For a set of the model's bands, it returns the
little-group representations they carry at each high-symmetry point. The
multiplicities of these representations at every high-symmetry point form the
bands' symmetry data vector. The symmetry data vector of an atomic limit is a
sum of the symmetry data vectors of elementary band representations (EBRs),
with nonnegative integer multiplicities. pythTQC returns the sums that equal
the bands' symmetry data vector, up to 200 of them by default. A set of bands
whose symmetry data vector is no such sum is topological. pythTQC also returns
the bands' symmetry-indicator class and the integer-valued ones among their
stable real-space invariants (SRSIs). Of the whole model, it returns every
SRSI, those valued modulo n among them.

The character tables, the EBRs and the symmetry-indicator groups are those of
the irreptables package. The tables of the SRSIs are those of Hwang *et al.*,
Nat. Commun. **17**, 8564 (2026), and are shipped with pythTQC.

## In a few lines

The example is the Kane-Mele model from `pythtb.models`, with intrinsic
spin-orbit coupling and no staggered potential or Rashba term. It is read as a
free-standing layer, with vacuum on both sides. A pythtb orbital does not say
which site irrep it carries. `characters={0: "s"}` declares orbital 0 an `s`
orbital. Orbital 1, at the honeycomb's other site, is in the same Wyckoff
orbit, so the declaration holds for it too. The calls return the group, the
little-group representations of the two lowest bands, their symmetry-indicator
class and their integer-valued SRSIs (ZSRSIs):

```python
from pythtb.models import kane_mele
import pythtqc as tqc

r = tqc.realized_group(kane_mele(0.0, -1.0, 0.3, 0.0),
                       environment="free-standing", characters={0: "s"})
print(r.group.name, r.group.number_str)
#  P6/mmm1' 191.234
t = tqc.hsp_table(r, bands=2)
print(tqc.format_coreps(t.symmetry_data))
#  -GM9 + -K8 + -M6
print(tqc.symmetry_indicator(t))
#  SymmetryIndicator(P6/mmm1' #191.234, class (1, 0) in Z6 x Z12, completed on the stack)
print(tqc.srsi_of_bands(t))
#  BandSRSIs(P6/mmm1' #191.234, theta_1 to theta_14 = (-1/6, 1/6, 1/2, 1/3, 0, 2/3, 0, 0, 0, -1/3, -2/3, 0, 1/2, 0), fractional)
```

## How to read it

- `P6/mmm1' 191.234`: the group the Hamiltonian obeys, `r.group`.
  `r.group.name` is its Hermann-Mauguin symbol. The `1'` marks time reversal
  adjoined to `P6/mmm`. `r.group.number_str` is its number among the magnetic
  space groups, in the Belov-Neronova-Smirnova (BNS) numbering.
- `-GM9 + -K8 + -M6`: the symmetry data vector of the two lowest bands,
  `t.symmetry_data`, one corep at each high-symmetry point of the layer's
  two-dimensional Brillouin zone, Γ, K and M. A corep is an irreducible
  little-group representation of a group with time reversal. `GM` is the
  tables' name for Γ. A leading `-` marks a double-valued corep, carried by the
  bands of a model with spin.
- `class (1, 0) in Z6 x Z12`: the symmetry-indicator class of the two bands,
  the class of their symmetry data vector modulo the integer combinations of
  the EBRs' symmetry data vectors. It is `1` modulo 6 and `0` modulo 12, in the
  indicator group `Z6 x Z12` of `P6/mmm1'`. The class is nonzero, so no atomic
  limit has this symmetry data vector. At these parameters the model is the
  quantum spin Hall insulator of Kane and Mele.
- `completed on the stack`: the tables of `P6/mmm1'` also list the
  high-symmetry points `A`, `H` and `L`, at `k_z = 1/2`. A layer's Hamiltonian
  has no `k_z` and does not reach them. The layer's class is the class of its
  decoupled stack, the layer repeated along its normal with no hopping between
  copies. The coreps at `A`, `H` and `L` are measured on that stack
  ([The group the class belongs to](docs/guide/indicators.md#the-group-the-class-belongs-to)).
- `theta_1 to theta_14 = (-1/6, ...)`: the ZSRSIs of the two bands, numbered
  as in Hwang *et al.* A real-space invariant is a combination of the
  site-irrep multiplicities of an atomic limit's Wannier functions that no
  smooth symmetric deformation changes. The SRSIs are a complete set of these
  that are linear in the multiplicities
  ([What the stable real-space invariants are](docs/guide/srsi.md#what-the-stable-real-space-invariants-are)).
  The ZSRSIs of an atomic limit are integers. They are fixed by the symmetry
  data vector, so a set of bands has them too. `fractional` says that some of
  them are not integers, so no atomic limit has these ZSRSIs. Their common
  denominator, 6, is the order of the class.

## What it answers

| question | the call | the page |
|---|---|---|
| What space group does this Hamiltonian obey? | `realized_group` | [What group does my Hamiltonian have?](docs/guide/group.md), [Is my 2D model on a substrate or free-standing?](docs/guide/layers.md) |
| Which little-group representations do its bands carry? | `hsp_table`, `coreps_at` | [Which little-group representations do these bands carry?](docs/guide/labels.md) |
| Could they be an atomic limit? | `decompose` | [Could these bands be an atomic limit?](docs/guide/atomic-limits.md) |
| Which symmetry-indicator class do they occupy? | `symmetry_indicator` | [Which symmetry-indicator class do these bands occupy?](docs/guide/indicators.md) |
| What are the stable real-space invariants of a set of its bands, or of the whole model? | `srsi_of_bands`, `srsi_of_model` | [What do the stable real-space invariants say about these bands?](docs/guide/srsi.md) |

## Install

From the repository's root:

```bash
pip install .                 # the package
pip install ".[examples]"     # ... and matplotlib, which the examples draw with
```

pythTQC needs Python 3.12 or later. pip installs its dependencies, pythtb
2.0.2 or later and the irreptables package among them. The irreptables package
holds the character tables that
[`irrep`](https://github.com/irreducible-representations/irrep) reads.
`tqc.tables_status()` returns whether the character tables can be read. Where
they cannot, every call that reads them refuses with `irrep_tables`.

## Where to go next

- **A model from pythtb's collection, or one built by hand:**
  [Quick start: Kane-Mele, end to end](docs/start/quickstart.md) reads
  Kane-Mele with a staggered potential and a Rashba term, from its group to
  whether its bands could be an atomic limit.
  [Fu-Kane-Mele: a 3D crystal, a distortion, the Z4 indicator](docs/start/tutorial-3d.md)
  follows it.
- **A layer:**
  [Is my 2D model on a substrate or free-standing?](docs/guide/layers.md) On a
  substrate, the default for a model written with two-dimensional positions,
  the Kane-Mele model above obeys `P6mm1'`. The indicator group of `P6mm1'` is
  trivial, so the model's two lowest bands have class zero there.
- **A Wannier90 model of a real material:**
  [Silicon from Wannier90: the group the model has](docs/start/tutorial-wannier90.md),
  then [How do I measure a numerical or Wannier90 model?](docs/guide/numerical.md).
- **Labels to compare with those of the Bilbao Crystallographic Server:**
  [Conventions: operations, labels and band representations](docs/background/conventions.md).
- **What each call takes and returns:**
  [Functions](docs/reference/functions.md), and
  [Returned records](docs/reference/records.md), every field of every record.
- **A call that refused:**
  [What does a refusal mean, and what do I do about it?](docs/guide/refusals.md)
- **Every page, by depth:**
  [The map of the documentation](docs/README.md), also at
  [the documentation site](https://pythtqc.readthedocs.io/).

## Tests

```bash
pip install ".[test]"
pytest                        # the fast tier
pytest -m gates               # every page and example against its printed output, and the site's build
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

A publication that uses pythTQC cites three works. The first is the paper that
defines the stable real-space invariants, Y. Hwang *et al.*, Nat. Commun.
**17**, 8564 (2026). The second is the paper that applies them to photonic
crystals, Y. Hwang *et al.*, arXiv:2601.06293 (2026). The third is the package
itself. [`CITATION.cff`](CITATION.cff) holds all three, and GitHub's "Cite this
repository" reads it. [Citing](docs/citing.md) lists the papers behind each
method.

## Acknowledgments

Development of these tools was supported by the National Science Foundation under
grant no. DMR-2510219.

## License

GPL-3.0-or-later; see [`LICENSE`](LICENSE).
