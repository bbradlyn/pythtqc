r"""What the declared-content wave costs, measured on a wannier90-scale model.

The route this package has to keep open is a wannier90 Hamiltonian of a real
material -- 96 Wannier functions, ~7e5 hoppings, a low-symmetry grey group of eight
operations -- whose **intertwiner solves are the budget**.  Every measurement
the declared-content wave adds (the channel layout, the per-operation verdict on
the declared content, the character matching) has to be cheap beside those
solves and may not re-solve them.  This script measures that rather than
assuming it.

The model is synthetic and deliberately awkward for the wave: 98 spinless
orbitals on a monoclinic ``C2/m``-shaped cell, several sites carrying two to
eight orbitals with hoppings that mix them under the site group -- so the layout
has something to refine -- on a cell whose site groups are ``2/m``, ``2``, ``m``
and ``1``, so the declared-content verdict has every shape of site to speak
about.  The hoppings are random and symmetric **by construction**, so the model
genuinely obeys the group it is asked about and the run is the full pipeline
rather than an early refusal.

What this model deliberately does **not** carry is a conserved operator.  A
genuinely decoupled orbital makes the representation not unique -- nothing
measures its phase against the rest -- so the build refuses before any timing
means anything, which is the honest verdict and not a benchmark.  The
enumerations that grow with a commutant (``symmetry._induced_blocks``) are
measured against their cap in ``tests/test_declared_content.py`` instead, on a
four-orbital model whose whole candidate set can be counted.

Run it as::

    python scripts/bench_declared_content.py             # this tree
    python scripts/bench_declared_content.py --json out.json

and compare two trees by running it in each checkout (a ``git worktree`` at the
commit to compare against) and diffing the JSON.  What to look at:

``seconds``
    wall time of one ``realized_group`` call.  The comparison that matters is
    *undeclared here* against *undeclared there*: that is the cost the wave adds
    to every caller, declared or not.
``peak_mb``
    ``tracemalloc``'s peak, which is dominated by ``validated._bases`` -- one
    solution space per k-action class.  Nothing this wave adds may copy them.
``detector_seconds``
    the detector alone, measured separately on the same model and the same
    group, so the wave's own share can be read off rather than guessed at.
"""

from __future__ import annotations

import argparse
import json
import time
import tracemalloc

import numpy as np

import pythtqc
from pythtqc import groups as tqg
from pythtqc import _measure
from pythtqc import _spacegroup

#: The cell: monoclinic, ``beta`` off ninety, so the point group is ``2/m`` at
#: best and the grey group has eight operations -- the shape of such a wannier90
#: model, without its data.
LATTICE = np.array([[1.0, 0.0, 0.0],
                    [0.0, 1.4, 0.0],
                    [0.35, 0.0, 2.1]])

#: The orbits, as ``(seed, n_orbitals)``: one seed per orbit, whose whole orbit
#: under the four operations above is put in the cell.  ``(0, 0, 0)`` and
#: ``(1/2, 0, 0)`` are fixed by all four (site group ``2/m``); ``(0, u, 0)`` by
#: the twofold alone (``2``); ``(u, 0, w)`` by the mirror alone (``m``); a
#: generic seed by nothing, giving a free orbit of four sites.  The last entry
#: is a two-orbital site, the smallest a site group's character table is asked
#: about here.
ORBITS = (
    ((0.0, 0.0, 0.0), 8),
    ((0.5, 0.0, 0.0), 8),
    ((0.0, 0.23, 0.0), 8),
    ((0.17, 0.0, 0.31), 8),
    ((0.21, 0.29, 0.13), 6),
    ((0.41, 0.13, 0.37), 6),
    ((0.0, 0.5, 0.5), 2),
)


#: How far a bond reaches, in the cell's own length units, and the box of
#: neighbour cells searched for one.  The cutoff sets the hopping count -- about
#: 2e4 at ~100 orbitals, which is the *shape* of that model's cost rather than its
#: 7e5: the full size is a run of its own, and this script has to
#: stay quick to rerun.
#: The box has to hold every bond within the cutoff, or the list would be
#: incomplete and the model would not obey its own group.
CUTOFF = 1.5
BOX = (3, 3, 2)


def _orbit(seed, tol=1e-7):
    """Every image of ``seed`` under the four operations, reduced and deduplicated."""
    out = []
    for u1, u2, u3 in ((seed[0], seed[1], seed[2]), (-seed[0], seed[1], -seed[2]),
                       (-seed[0], -seed[1], -seed[2]), (seed[0], -seed[1], seed[2])):
        pt = np.array([u1, u2, u3], dtype=float) % 1.0
        if not any(np.abs((pt - q + 0.5) % 1.0 - 0.5).max() < tol for q in out):
            out.append(pt)
    return out


def build_model(seed: int = 20260917):
    """The synthetic model, symmetric **by construction**.

    Every amplitude is a function of two things a symmetry operation preserves:
    the Cartesian distance between the two centres, and the pair of orbital
    *slots* within their sites -- and the slot order is the same at every site
    of an orbit, which is what the orbital permutation is lifted from.  So the
    hopping list is exactly invariant under the cell's own group, with no
    projection and no tolerance anywhere, and the run below is the pipeline on
    a model that genuinely has the group it is asked about.

    The per-slot matrices are random, so a site's orbitals do mix under its
    site group and the layout has something to measure.
    """
    from pythtb import Lattice, TBModel

    centres, slots = [], []
    for _k, (seed_pt, n) in enumerate(ORBITS):
        for pt in _orbit(seed_pt):
            centres.extend([[float(x) for x in pt]] * n)
            slots.extend([(_k, s) for s in range(n)])
    model = TBModel(Lattice(LATTICE, centres, periodic_dirs=[0, 1, 2]), spinful=False)
    norb = len(centres)
    rng = np.random.default_rng(int(seed))
    # one on-site energy per (orbit, slot), which every site of the orbit shares
    energy = {key: float(rng.normal(scale=0.4)) for key in dict.fromkeys(slots)}
    model.set_onsite([energy[key] for key in slots])
    # one amplitude per (slot pair, distance), rounded so that two bonds the
    # group relates land on the same key however the arithmetic went.  Every
    # cell of the box is searched and every bond within the cutoff is put in,
    # so the list is complete -- which is what makes it invariant.
    cart = np.asarray(centres, dtype=float) @ LATTICE
    amps: dict = {}
    keep = list(range(norb))
    for n1 in range(-BOX[0], BOX[0] + 1):
        for n2 in range(-BOX[1], BOX[1] + 1):
            for n3 in range(-BOX[2], BOX[2] + 1):
                shell = (n1, n2, n3)
                if shell < (0, 0, 0):        # each bond once: R and -R are one
                    continue
                offset = np.asarray(shell, dtype=float) @ LATTICE
                delta = cart[None, :, :] + offset - cart[:, None, :]
                dist = np.linalg.norm(delta, axis=-1)
                for i in keep:
                    for j in keep:
                        if shell == (0, 0, 0) and j <= i:
                            continue
                        d = float(dist[i, j])
                        if d > CUTOFF:
                            continue
                        # the key is symmetric in the two slots and the
                        # amplitude real, because a bond and its image can be
                        # stored in either direction and pythtb reads the
                        # reverse as the conjugate
                        key = (min(slots[i], slots[j]), max(slots[i], slots[j]),
                               round(d, 6))
                        if key not in amps:
                            amps[key] = float(rng.normal(scale=0.12))
                        model.set_hop(amps[key], i, j, list(shell), mode="add")
    group = _spacegroup._group_of(model)
    if not group:
        raise SystemExit(f"the synthetic cell did not identify: {group}")
    return model, group


def _time(fn):
    """``(value, seconds, peak_mb)`` for one call, with ``tracemalloc`` running."""
    tracemalloc.start()
    t0 = time.perf_counter()
    value = fn()
    seconds = time.perf_counter() - t0
    _cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return value, float(seconds), float(peak) / 1e6


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", default=None, help="write the record here as JSON")
    parser.add_argument("--seed", type=int, default=20260917)
    args = parser.parse_args()

    model, cell_group = build_model(args.seed)
    nhop = int(model.nhops)
    record = {
        "version": str(getattr(pythtqc, "__version__", "?")),
        "norb": int(model.norb), "nstate": int(model.nstate),
        "n_hoppings": int(nhop),
        "cell_group": f"{cell_group.name} #{cell_group.number_str}",
        "cell_group_size": int(cell_group.size),
        "runs": {},
    }

    # the detector alone, on the cell's own group: the budget everything else is
    # measured against
    _v, det_s, det_mb = _time(lambda: _measure._validate_group(model, cell_group))
    record["detector_seconds"] = det_s
    record["detector_peak_mb"] = det_mb

    base, sec, mb = _time(lambda: tqg.realized_group(model))
    layout = getattr(getattr(base, "rep", None), "multiplets", None)
    record["runs"]["undeclared"] = {
        "seconds": sec, "peak_mb": mb,
        "group": (f"{base.group.name} #{base.group.number_str}" if base
                  else f"REFUSAL {base.gate}"),
        "kept": (len(base.validated.indices) if base else None),
        "layout": (None if layout is None else layout.summary()),
        "conserved": (base.conserved["dim"]
                      if base and hasattr(base, "conserved") else None),
    }
    if not base:
        print(json.dumps(record, indent=1))
        return 1

    # ... and with a declaration on two sites, in the labels the undeclared run
    # measured, so the declared run is the same crystal and the comparison is
    # the wave's own cost rather than a different measurement
    rows = base.rep.characters.orbitals
    declared = {}
    for site in (0, 2):
        orbs = [o for o in rows if int(o.site) == site]
        if orbs and orbs[0].irrep:
            declared[int(orbs[0].orbital)] = str(orbs[0].irrep)
    record["declaration"] = dict(declared)
    if declared:
        out, sec, mb = _time(lambda: tqg.realized_group(model, characters=declared))
        record["runs"]["declared"] = {
            "seconds": sec, "peak_mb": mb,
            "group": (f"{out.group.name} #{out.group.number_str}" if out
                      else f"REFUSAL {out.gate}"),
            "kept": (len(out.validated.indices) if out else None),
            "descent": (None if not out or out.descent is None
                        else out.descent["steps"][0]["refuted_by"]),
        }

    for name, run in record["runs"].items():
        run["over_detector"] = (run["seconds"] / det_s) if det_s > 0 else float("nan")
    print(json.dumps(record, indent=1))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
