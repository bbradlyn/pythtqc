"""Performance phase, wave P3: symmetrization, and the signal that licenses it.

A Hamiltonian that came out of a wannier90 run is symmetric only to its own
numerical defect (~1e-7 relative, measured), five decades above machine
precision, and every gate downstream of the detector is exact by design.  The
route this wave adds is not a looser gate anywhere: it is to **declare** the
defect as the detector's ``tol``, read the resulting per-operation residual as
*exact* / *near* / *broken*, and — on the ``near`` signal — project ``H`` onto
its group-invariant part and run the exact pipeline on the result.

What is regressed here, and the number that closes each:

* **the theorem of the wave** — lin3g with *every* hopping jittered at relative
  1e-7 comes back, after symmetrization, with the **clean model's** labels,
  representation matrices (identical to ``0.0``), high-symmetry-point content,
  partitions and EBR decomposition, having moved ``H`` by 9.5e-7 against a
  jitter of 1.0e-6 — it recovers the physics the jitter hid and changes ``H`` by
  no more than the jitter did;
* **the same on every presentation the package has** — the layer route, whose
  cell is the slab point set as given (Kane-Mele odd, descended to ``P3m11'``;
  the two-equivalent-site ``km_inversion``, realized outright in the
  48-operation ``P6/mmm1'``), a 3D cell whose group comes from a crystal field
  (the bond-decorated Fu-Kane-Mele strong phase), a declared spinor basis
  (flattened Kane-Mele), and a magnetic layer whose *unitary* group is what is
  averaged over (staggered-Zeeman Kane-Mele: ``P3`` in, ``P3`` out, time
  reversal not invented);
* **the transform is exact, and both halves of that are measured** — the
  largest rebuilt coefficient outside the range box (1e-17 … 3e-16 relative)
  and ``max_k |H_new(k) − H_sym(k)|`` on random k off the transform grid
  (1e-15);
* **broken is not near** — a genuine 1e-2 distortion is refused by
  ``realized_group`` exactly as before this wave, and ``symmetrize`` on a
  hand-built record that *lists* such an operation refuses
  ``symmetrize_broken_operation`` rather than imposing the symmetry;
* **idempotence** — a symmetrized model classifies every operation ``exact``
  and a second pass moves it by 5.3e-15;
* **exact is untouched** — every existing fixture through ``realized_group`` at
  the default tolerance gives ``symmetrization is None`` (the rest of the suite
  is that test);
* **the other defect** — a wannier90 model's *centres* are numerically
  determined too, and centres that miss the group's own action by ``δ`` cost
  ``2π·δ·max|H|`` of covariance whatever the hoppings do (one Kane-Mele orbital
  displaced by 1e-8 costs 2.4e-7).  ``symmetrize_positions`` moves them onto the
  nearest positions the group carries exactly, in the model's own frame, which is
  a **k-dependent gauge transformation** and therefore free: the spectrum moves
  by rounding and the residual returns to rounding.  Positions further off than ``max_move`` are
  refused rather than moved — "broken is not near", for the geometry;
* **and that defect measured first, on a declaration** (§ 8, wave P3b) —
  ``realized_group(position_tol=)`` is the absolute, reduced-coordinate
  tolerance for the centres, measured before the detector runs and before any
  hopping is projected.  The lossless step precedes the lossy one, and where
  both defects are present the ordering is what makes the answer the clean
  model's.

The operators averaged with are a representation's, assembled from discrete
signs, so where every multiplet is one orbital they carry none of the model's
noise.  A site whose orbitals form a wider multiplet carries the model's own
block, and with it the noise; the build takes the exact representation nearest
those blocks, so the operators are a representation all the same.
"""

import dataclasses
import inspect

import numpy as np
import pytest
from pythtb import Lattice, TBModel

from pythtqc import _sitechars
from pythtqc import ebr as tqe
from pythtqc import groups as tqg
from pythtqc import sewing as tqw
from pythtqc import spin as tqspin
from pythtqc import symmetry as tqs
from pythtqc.refusal import InternalError, Refusal
from pythtqc.symmetrization import (EXACT_GATE, classify_operations, symmetrize,
                                    symmetrize_positions)
from pythtqc import _build
from pythtqc import _tolerance
from pythtqc import _detector
from pythtqc import _layers
from pythtqc import _measure
from pythtqc import _realized
from pythtqc import _spacegroup
from pythtqc import _tables

import tqc_models
from conftest import FKM_BOND, fkm_regime, km_z2_regime, patch_everywhere

#: Relative size of the jitter every fixture below is perturbed by — the
#: measured defect scale of a symmetrized wannier90 import.
JITTER = 1e-7

#: The declared tolerance, as the docs state the recipe: ``10 x defect x max|H|``.
#: It is an **absolute** gate in the units of H, so it is read against the
#: Hamiltonian's own magnitude and never typed as a bare number.
TOL_FACTOR = 10.0

_ULP = np.finfo(float).eps          # one unit of double-precision roundoff, 2.2e-16
_FEW_ULP = 8 * _ULP                 # a few ulp of a reduced coordinate: thirds arithmetic


def _max_H(model, n=64, seed=3, **params):
    k = np.random.default_rng(seed).uniform(-0.5, 0.5, size=(n, int(model.lattice.dim_k)))
    return float(np.abs(np.asarray(model.hamiltonian(k_pts=k, flatten_spin_axis=True,
                                                     **params))).max())


def jitter(model, rel=JITTER, seed=7):
    """Every hopping and on-site term multiplied by ``1 + rel*normal``.

    The whole Hamiltonian, not one term: a wannier90 defect is a property of
    the fit, not of a bond, and a single perturbed hopping leaves most of the
    group exactly satisfied.  Spin blocks are perturbed entry by entry and
    re-Hermitized, so the model stays a Hamiltonian.
    """
    rng = np.random.default_rng(seed)
    dim_r = int(model.lattice.dim_r)
    out = TBModel(Lattice(np.asarray(model.lat_vecs, dtype=float),
                          np.asarray(model.orb_vecs, dtype=float).tolist(),
                          periodic_dirs=list(model.periodic_dirs)),
                  spinful=(int(model.nspin) == 2))
    ons = np.asarray(model.onsite, dtype=complex) * (
        1.0 + rel * rng.normal(size=np.asarray(model.onsite).shape))
    if ons.ndim == 3:
        ons = 0.5 * (ons + np.conj(np.swapaxes(ons, -1, -2)))
        out.set_onsite(ons, mode="set")
    else:
        out.set_onsite([float(x.real) for x in ons], mode="set")
    for h in model.hoppings:
        amp = np.asarray(h["amplitude"], dtype=complex)
        amp = amp * (1.0 + rel * rng.normal(size=amp.shape))
        out.set_hop(amp if amp.ndim else complex(amp), h["from_orbital"],
                    h["to_orbital"], list(h.get("lattice_vector", [0] * dim_r)),
                    mode="add")
    return out


def distort(model, rel=1e-2, which=0):
    """One hopping scaled by ``1 + rel`` — a genuine distortion, not noise."""
    dim_r = int(model.lattice.dim_r)
    out = TBModel(Lattice(np.asarray(model.lat_vecs, dtype=float),
                          np.asarray(model.orb_vecs, dtype=float).tolist(),
                          periodic_dirs=list(model.periodic_dirs)),
                  spinful=(int(model.nspin) == 2))
    out.set_onsite(np.asarray(model.onsite), mode="set")
    for n, h in enumerate(model.hoppings):
        amp = np.asarray(h["amplitude"], dtype=complex) * (1.0 + rel if n == which else 1.0)
        out.set_hop(amp if amp.ndim else complex(amp), h["from_orbital"],
                    h["to_orbital"], list(h.get("lattice_vector", [0] * dim_r)),
                    mode="add")
    return out


def _energy_shifted(model, c):
    """``H + c·1``: every on-site energy raised by ``c``, the hoppings as they are."""
    dim_r = int(model.lattice.dim_r)
    out = TBModel(Lattice(np.asarray(model.lat_vecs, dtype=float),
                          np.asarray(model.orb_vecs, dtype=float).tolist(),
                          periodic_dirs=list(model.periodic_dirs)),
                  spinful=(int(model.nspin) == 2))
    ons = np.asarray(model.onsite, dtype=complex)
    if ons.ndim == 3:
        out.set_onsite(ons + float(c) * np.eye(2), mode="set")
    else:
        out.set_onsite([float(x.real) + float(c) for x in ons], mode="set")
    for h in model.hoppings:
        amp = np.asarray(h["amplitude"], dtype=complex)
        out.set_hop(amp if amp.ndim else complex(amp), h["from_orbital"],
                    h["to_orbital"], list(h.get("lattice_vector", [0] * dim_r)),
                    mode="add")
    return out


#: One ``s`` orbital on a tetragonal lattice, ``c = 1.4``: hoppings to the
#: first three shells in the plane, out to three cells along ``a``, and to the
#: planes above and below, one cell along ``c``.
_TETRAGONAL_HOPS = (((1, 0, 0), -1.0), ((0, 1, 0), -1.0),
                    ((1, 1, 0), -0.3), ((1, -1, 0), -0.3),
                    ((2, 0, 0), -0.1), ((0, 2, 0), -0.1),
                    ((2, 1, 0), -0.04), ((1, 2, 0), -0.04),
                    ((2, -1, 0), -0.04), ((1, -2, 0), -0.04),
                    ((3, 0, 0), -0.02), ((0, 3, 0), -0.02),
                    ((0, 0, 1), -0.25), ((1, 0, 1), -0.05), ((0, 1, 1), -0.05),
                    ((-1, 0, 1), -0.05), ((0, -1, 1), -0.05))


def _tetragonal_reach(jitter=0.0, seed=11):
    """``P4/mmm`` whose hoppings reach three cells in the plane and one along the axis, each off by ``jitter``."""
    rng = np.random.default_rng(seed)
    m = TBModel(Lattice(np.diag([1.0, 1.0, 1.4]).tolist(), [[0.0, 0.0, 0.0]],
                        periodic_dirs=[0, 1, 2]))
    m.set_onsite([0.3], mode="set")
    for R, t in _TETRAGONAL_HOPS:
        m.set_hop(t * (1.0 + jitter * rng.normal()), 0, 0, list(R), mode="add")
    return m


def _pipeline(model, typat=None, *, tol, nk=6, decorations=None, spinor=None,
              include_TR=True):
    """slab -> decorate -> _group_of -> assign -> validate, at a declared tol."""
    three_d = int(model.lattice.dim_k) == 3
    m3 = model if three_d else _layers._slab_presentation(model)
    assert not isinstance(m3, Refusal), str(m3)
    labels = [0] * int(model.norb) if typat is None else list(typat)
    if three_d and decorations is None:
        cell, cell_typat = m3, labels
    else:
        cell, cell_typat = _layers._decorated_cell(m3, labels, decorations=decorations)
    if spinor is None:
        spinor = int(model.nspin) == 2
    group = _spacegroup._group_of(cell, typat=cell_typat, spinor=bool(spinor),
                          include_TR=include_TR)
    assert group, str(group)
    asg = _tables._assign_orbitals(m3, group, typat=labels)
    assert asg, str(asg)
    return m3, group, asg, _measure._validate_group(m3, group, nk=nk, tol=tol)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def lin3g_clean(lin3g_declared_cross_checked):
    """lin3g in the group its Hamiltonian realizes, from exact numbers."""
    out = lin3g_declared_cross_checked
    assert out.symmetrization is None, "an exact model has nothing to symmetrize"
    return out


@pytest.fixture(scope="module")
def lin3g_jittered(lin3g):
    """The same model with every hopping jittered at 1e-7, symmetrized.

    ``tol`` is ``10 x defect x max|H|`` — the recipe the docs state — because
    the detector's gate is absolute and this model's ``max|H|`` is 12.375, so a
    1e-7 *relative* defect is a 1.2e-6 *absolute* one.
    """
    model = jitter(lin3g)
    tol = TOL_FACTOR * JITTER * _max_H(model)
    out = tqg.realized_group(model, typat=[0, 0, 1, 1], tol=tol,
                             characters=["s", "dx2-y2", "s", "s"])
    assert out, str(out)
    return out, model, tol


@pytest.fixture(scope="module")
def km_odd_free_standing(realized):
    """Kane-Mele's odd regime as a free-standing layer, from exact numbers.

    ``P-6m21' #187.210`` descended to the ``P3m11' #156.50`` the Hamiltonian
    obeys: the clean record the centre and hopping passes below are measured
    against.  No test here reads the cross-check against the substrate
    reading, so it is not run.
    """
    out = realized(km_z2_regime("odd"), typat=[0, 1], environment="free-standing",
                   characters={0: "A1"}, cross_check=False)
    assert out, str(out)
    assert out.symmetrization is None, "an exact model has nothing to symmetrize"
    return out


# ---------------------------------------------------------------------------
# 1.  the classification: exact / near / broken, per operation
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_the_three_classes_are_read_off_the_measured_residuals(lin3g, lin3g_jittered):
    """The one principle of the wave, as a measurement on ``ValidatedGroup``.

    The exact gate is ``_build_rep``'s own: never below the floor
    ``EXACT_GATE``, which is ``EXACT_TOL`` and not retyped here, and above it
    where the rounding a model's representation carries is larger (the
    classification is handed that gate by ``realized_group`` and
    ``symmetrize``).  The boundary above is whatever the caller declared.
    """
    _out, model, tol = lin3g_jittered
    m3, group, _asg, val = _pipeline(model, [0, 0, 1, 1], tol=tol)
    classes = classify_operations(val, tol)
    assert set(classes) == set(val.indices)
    assert set(classes.values()) == {"exact", "near"}, classes
    # the identity is exact; every other operation carries the jitter
    assert sum(1 for c in classes.values() if c == "near") == len(classes) - 1

    # ... and at a tolerance below the defect the SAME residuals read broken
    assert set(classify_operations(val, 1e-9).values()) == {"exact", "broken"}
    assert EXACT_GATE == 5e-14 == tqs.EXACT_TOL, \
        "the floor of _build_rep's own gate, and that number does not move"


# ---------------------------------------------------------------------------
# 2.  the theorem: symmetrization recovers the clean model's physics
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_the_jittered_lin3g_comes_back_as_the_clean_one(lin3g_clean, lin3g_jittered):
    """The theorem of this wave, on the primary golden.

    Every hopping jittered at 1e-7 breaks the representation outright — at the
    exact covariance gate the best candidate misses by 1.5e-6 — and after the
    projection the model carries the clean one's **labels, matrices, content,
    partitions and EBR decomposition**, having moved ``H`` by less than the
    jitter moved it.
    """
    out, model, _tol = lin3g_jittered
    rec = out.symmetrization
    assert rec is not None

    # it moved by O(the residuals), and less than the jitter itself did
    k = np.random.default_rng(5).uniform(-0.5, 0.5, size=(24, 2))
    clean_H = np.asarray(lin3g_clean.model.hamiltonian(k_pts=k, flatten_spin_axis=True))
    jit_H = np.asarray(_layers._slab_presentation(model).hamiltonian(
        k_pts=k, flatten_spin_axis=True))
    sym_H = np.asarray(out.model.hamiltonian(k_pts=k, flatten_spin_axis=True))
    # the docstring's number, not the gate it passed: `symmetrize` refuses above
    # `tol`, so `defect <= tol` is unfailable on a record that came back
    assert rec.defect == pytest.approx(9.49e-7, rel=1e-2, abs=0)
    assert np.abs(sym_H - clean_H).max() < np.abs(jit_H - clean_H).max()

    # the group, the representation and the declared characters are the clean ones
    assert out.group.number_str == lin3g_clean.group.number_str == "99.164"
    assert out.rep._choice == lin3g_clean.rep._choice
    assert max(float(np.abs(out.rep.matrices[i] - lin3g_clean.rep.matrices[i]).max())
               for i in lin3g_clean.rep.indices) == 0.0
    assert out.rep.characters.irreps == lin3g_clean.rep.characters.irreps

    # ... and so is everything the labels are for
    bands = (0, 1, 2, 3, 4, 5)
    a = tqw.hsp_table(lin3g_clean.rep, bands=bands)
    b = tqw.hsp_table(out.rep, bands=bands)
    assert a.complete and b.complete
    assert a.counts == b.counts == {"-GM6": 2, "-GM7": 1, "-M6": 1, "-M7": 2, "-X5": 3}
    assert a.partitions == b.partitions
    assert {p: dict(v) for p, v in a.content.items()} == {p: dict(v) for p, v in
                                                          b.content.items()}
    da = tqe.decompose(a, lin3g_clean.group, rep=lin3g_clean.rep, bands=6)
    db = tqe.decompose(b, out.group, rep=out.rep, bands=6)
    assert da.verdict == db.verdict == "symmetry-indicated decomposable"
    assert set(da.solution_strings()) == set(db.solution_strings())


@pytest.mark.slow
def test_the_record_carries_the_measurements_that_license_it(lin3g_jittered):
    """The record is the result: what moved, by how much, and what the checks say.

    The transform grid is the smallest one on which the transform is exact,
    plus a shell: ``R_max`` is the model's own hopping range, ``R_box`` bounds
    where the group action can carry it, and the grid has ``2(R_box + 1) + 1``
    divisions along each direction, alike along directions the group's
    operations mix, so its fundamental domain is one cell wider than the box
    and the aliasing check has somewhere to look.  A tetragonal model whose
    hoppings reach three cells in the plane and one along the axis is divided
    ``9 x 9 x 5``, 405 points where the widest reach in every direction would
    take 729, and its projection is the one the ``9 x 9 x 9`` grid makes, to
    rounding.  The rebuilt hopping list is the symmetrized Hamiltonian **off**
    that grid too (``k_identity``), which is what says the finite Fourier
    series was complete.
    """
    out, _model, _tol = lin3g_jittered
    rec = out.symmetrization
    assert len(rec.near) == len(rec.classes) - 1 and len(rec.exact) == 1
    assert rec.passes == 1, "the exact D of the representation route needs one pass"
    # the defect is O(the residuals that licensed it) -- measured 0.031 of the
    # bound below.  Not `<= tol`: `symmetrize` refuses above `tol`, so that one
    # cannot fail on a record it returned
    worst = max(rec.residuals[i] for i in rec.near)
    assert rec.defect <= 10 * worst
    # the grid, derived from the model's own range and the group's reach
    g = rec.grid
    assert g["R_max"] == (1, 1) and g["R_box"] == (2, 2)
    assert g["n"] == (7, 7) and g["dim_k"] == 2
    assert g["n_points"] == 49
    # ... divided along each direction to its own reach, alike where the
    # operations mix them: the fourfold axis apart from the plane
    reach = _tetragonal_reach(jitter=1e-6)
    tol = TOL_FACTOR * 1e-6 * _max_H(reach)
    m3, group, asg, val = _pipeline(reach, tol=tol)
    fine, fine_rec = symmetrize(m3, group, asg, val, tol=tol)
    cubic, cubic_rec = symmetrize(m3, group, asg, val, tol=tol, k_grid=9)
    assert fine_rec.grid["R_box"] == (3, 3, 1)
    assert (fine_rec.grid["n"], fine_rec.grid["n_points"]) == ((9, 9, 5), 405)
    assert (cubic_rec.grid["n"], cubic_rec.grid["n_points"]) == ((9, 9, 9), 729)
    k = np.random.default_rng(5).uniform(-0.5, 0.5, size=(32, 3))
    a = np.asarray(fine.hamiltonian(k_pts=k, flatten_spin_axis=True))
    b = np.asarray(cubic.hamiltonian(k_pts=k, flatten_spin_axis=True))
    assert float(np.abs(a - b).max()) < 10 * _ULP * _max_H(reach), \
        "the same projection on either grid, to rounding"
    # the two exactness checks of the transform
    assert rec.checks["outside_box"] < 1e-13
    assert rec.checks["k_identity"] < 1e-12
    assert rec.checks["hermiticity"] < 1e-13
    assert rec.checks["covariance_residue"] < 1e-12
    # the drop HAPPENED, which is what makes the two bounds after it a
    # statement: `_drop_threshold` returns `(0.0, inf)` when it cuts nothing,
    # and both are satisfied by that branch.  Measured: 552 rebuilt blocks
    # dropped, the cut at 6.7e-16 (0.25 ulp of max|H| = 12.375) under a gap of
    # 6.8e14 to the smallest coefficient kept -- so the drop is orbit-safe, at
    # a gap in the block norms rather than at an amplitude
    assert rec.checks["n_dropped"] > 0
    assert rec.checks["drop_threshold"] < 10 * _ULP * rec.scale
    assert rec.checks["drop_gap"] > 1e10
    assert rec.checks["n_hoppings"] == int(out.model.nhops) == 16
    # the after-the-fact verification, at the DEFAULT tolerance
    assert rec.verification["tol"] == 1e-8
    assert rec.verification["n_kept"] == rec.verification["n_operations"]
    assert rec.verification["worst_residual"] < 1e-12
    assert rec.verification["near_kept"]
    assert "symmetrized (defect" in repr(out)
    assert "operations were near-symmetric" in rec.summary()


@pytest.mark.slow
def test_the_operators_carry_none_of_the_models_noise(lin3g_jittered, lin3g_clean):
    """Why one pass is exact: ``D`` is assembled from **discrete** signs.

    The provisional build reads the ansatz's phases off the detector's solution
    space, each the root of unity it sits at within the noise, and assembles
    ``D = O(g) x S(g)`` through the same path as any build — so the matrices
    are the *clean* model's, bit for bit, and the average with them is a
    projection rather than an ``O(r)`` approximation to one.
    """
    out, model, tol = lin3g_jittered
    m3, group, asg, val = _pipeline(model, [0, 0, 1, 1], tol=tol)
    prov = _build._build_rep(m3, group, asg, validated=val, tol=tol,
                          internal_tol=10 * max(val.residuals[i] for i in val.indices)
                          / _max_H(m3), characters=["s", "dx2-y2", "s", "s"])
    assert prov, str(prov)
    # the same declaration on the clean model
    exact = lin3g_clean
    assert max(float(np.abs(prov.matrices[i] - exact.rep.matrices[i]).max())
               for i in exact.rep.indices) == 0.0
    # the provisional build's own covariance is the model's defect, and its
    # other two gates read machine zero on the assembled candidate
    assert prov._search["internal"] < 1e-14 and prov._search["product"] < 1e-14
    assert 1e-8 < prov._search["best"] <= tol


# ---------------------------------------------------------------------------
# 3.  the presentations: slab/decoration, 3D, a declared spinor basis, magnetic
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("name", ["km_odd", "km_inversion", "fkm_strong"])
def test_symmetrization_recovers_the_clean_answer_on_every_presentation(name, request,
                                                                         realized):
    """The theorem again, on the routes ``symmetrize`` has to respect.

    The layer route (Kane-Mele, whose cell is the slab point set **as given**)
    and a 3D cell whose group comes from a ``decorations=`` crystal field (the
    Fu-Kane-Mele strong phase, ``R-3m1'``).  In each the rebuilt model must keep
    the same group, the same provenance, the same declared labels and the same
    representation matrices as the clean model — the positions,
    ``periodic_dirs`` and padded cell survive the round trip through k-space.

    The two layers are two different routes through that cell, which is why
    both are here.  With the sublattices declared inequivalent
    (``typat=[0, 1]``) the layer cell allows ``P-6m21' #187.210``, 24
    operations, and the Hamiltonian's own ``σ_h`` fibers cut it to the
    ``P3m11' #156.50`` served — a **descent**, so ``provenance`` is
    ``"descended"`` on both the clean and the symmetrized run, and the jitter
    refutes nothing the clean descent did not: the same operations are broken,
    by name and by reason.  With the two sites equivalent (``km_inversion``)
    nothing is cut: the layer cell's ``P6/mmm1' #191.234``, 48 operations, is
    realized outright.  The site group
    there is ``-6m2``, whose labels are ``A1'``/``A2'``/``A1''``/``A2''``/
    ``E'``/``E''``, and ``A1`` names no irrep of it — the declaration for a
    layer of odd (``p_z``) orbitals is ``pz``, which resolves to ``A2''``: even
    under ``C3z`` and the vertical mirrors, odd under ``σ_h``.  That is the
    physics Kane-Mele has and the in-plane reading simply could not state.

    Both layers declare ``environment="free-standing"``, which wave R4c made
    necessary rather than optional: a ``dim_r = 2`` model is read on a
    substrate unless told otherwise, and there the ``σ_h`` family this test's
    descents are made of is not in the cell at all.
    """
    cases = {
        "km_odd": (km_z2_regime("odd"),
                   dict(typat=[0, 1], environment="free-standing",
                        characters={0: "A1"})),
        "km_inversion": (tqc_models.kane_mele_inversion,
                         dict(environment="free-standing", characters={0: "pz"})),
        "fkm_strong": (fkm_regime("strong"),
                       dict(decorations=FKM_BOND, characters={0: "s"})),
    }
    served = {"km_odd": ("156.50", 12, "descended"),
              "km_inversion": ("191.234", 48, "realized"),
              "fkm_strong": ("166.98", 24, "realized")}
    build, kw = cases[name]
    model = build()
    # the clean records are shared: two by fixture name, and the strong phase
    # through the cache every module that reads it shares
    shared = {"km_odd": "km_odd_free_standing",
              "km_inversion": "km_inversion_free_standing"}
    clean = (request.getfixturevalue(shared[name]) if name in shared
             else realized(build, **kw))
    assert clean, str(clean)
    assert clean.symmetrization is None
    assert (clean.group.number_str, clean.group.size,
            clean.group.provenance) == served[name]

    jit = jitter(model)
    tol = TOL_FACTOR * JITTER * _max_H(model)
    # a layer's other environment is not read here; a crystal has none
    out = tqg.realized_group(jit, tol=tol, cross_check=False, **kw)
    assert out, str(out)
    rec = out.symmetrization
    assert rec is not None and rec.passes == 1
    assert rec.defect <= 10 * max(rec.residuals[i] for i in rec.near)
    assert rec.defect_relative < 10 * JITTER
    assert rec.checks["outside_box"] < 1e-13 and rec.checks["k_identity"] < 1e-12
    assert rec.verification["near_kept"] and not rec.verification["broken"]
    assert rec.verification["worst_residual"] < 1e-12

    # the same group by the same route: the jitter moved neither the answer nor
    # how it was reached
    assert (out.group.number_str, out.group.size,
            out.group.provenance) == served[name]
    assert out.group.number_str == clean.group.number_str
    assert (out.descent is None) == (clean.descent is None)
    if clean.descent is not None:
        assert out.descent["from"] == clean.descent["from"]
        assert out.descent["dropped_by"] == clean.descent["dropped_by"]
        # near is not broken: the jitter refutes nothing the clean descent did not
        assert [(b["ind"], b["name"], b["refuted_by"]) for b in out.descent["dropped"]] \
            == [(b["ind"], b["name"], b["refuted_by"]) for b in clean.descent["dropped"]]
    assert out.rep._choice == clean.rep._choice
    assert max(float(np.abs(out.rep.matrices[i] - clean.rep.matrices[i]).max())
               for i in clean.rep.indices) == 0.0
    assert out.rep.characters.irreps == clean.rep.characters.irreps
    # the presentation itself came back unchanged
    assert np.array_equal(np.asarray(out.model.orb_vecs),
                          np.asarray(clean.model.orb_vecs))
    assert list(out.model.periodic_dirs) == list(clean.model.periodic_dirs)
    assert int(out.model.nspin) == int(clean.model.nspin)


@pytest.mark.slow
def test_a_declared_spinor_basis_symmetrizes_like_the_native_model():
    """The wannier90 presentation: a **spinless** model whose orbitals are spinors.

    ``spin=`` is forwarded to the provisional build exactly as it is to the
    final one, so the operators are the declared-pair ansatz's and the rebuilt
    hopping list is written in the flattened basis it came in.
    """
    flat, sb = tqspin.spinless_presentation(tqc_models.kane_mele_z2("odd"))
    clean = tqg.realized_group(flat, typat=[0, 0, 1, 1], spin=sb, characters={0: "A1"},
                               cross_check=False)
    assert clean, str(clean)
    assert clean.symmetrization is None

    jit = jitter(flat)
    tol = TOL_FACTOR * JITTER * _max_H(flat)
    out = tqg.realized_group(jit, typat=[0, 0, 1, 1], spin=sb, tol=tol,
                             characters={0: "A1"}, cross_check=False)
    assert out, str(out)
    rec = out.symmetrization
    assert rec is not None
    assert rec.defect <= 10 * max(rec.residuals[i] for i in rec.near)
    assert int(out.model.nspin) == 1 and int(out.model.norb) == 4
    assert out.group.number_str == clean.group.number_str
    assert out.rep.characters.irreps == clean.rep.characters.irreps
    assert max(float(np.abs(out.rep.matrices[i] - clean.rep.matrices[i]).max())
               for i in clean.rep.indices) == 0.0


#: A chiral decoration: three atoms above and three below the plane in a C3
#: orbit, which keeps C3 and breaks every mirror.  ``test_groups.py`` owns the
#: same recipe for the magnetic layer.
_C3_ORBIT = ((0.2, 0.1), (-0.3, 0.2), (0.1, -0.3))
_CHIRAL_C3 = tuple([([x, y, tqg.DECORATION_DZ], "top") for x, y in _C3_ORBIT]
                   + [([x, y, -tqg.DECORATION_DZ], "bot") for x, y in _C3_ORBIT])


@pytest.mark.slow
def test_a_magnetic_model_is_symmetrized_without_inventing_time_reversal():
    """The antiunitary coset is *measured*, and what is averaged over is what survived.

    Kane-Mele plus a staggered Zeeman term obeys neither time reversal nor
    ``C2z``; its realized group is the unitary ``P3``, and the average runs over
    those three operations alone.  Time reversal has no intertwiner before the
    pass and none after it — symmetrization projects onto the group it was
    handed and invents nothing.
    """
    from pythtb.models import kane_mele

    model = kane_mele(0.0, -1.0, 0.3, 0.0)
    model.set_onsite([[0, 0, 0, 0.3], [0, 0, 0, -0.3]], mode="add")
    jit = jitter(model)
    tol = TOL_FACTOR * JITTER * _max_H(model)
    # `A1` is the trivial irrep of a threefold axis, the way the position's own
    # site-symmetry listing spells it
    out = tqg.realized_group(jit, typat=[0, 1], decorations=_CHIRAL_C3, tol=tol,
                             characters={0: "A1"}, cross_check=False)
    assert out, str(out)
    assert out.group.number_str == "143" and not out.group.include_TR
    assert out.time_reversal["outcome"] == "measured broken"
    rec = out.symmetrization
    assert rec is not None and len(rec.operations) == 3
    assert rec.defect <= 10 * max(rec.residuals[i] for i in rec.near)

    k = np.stack(np.meshgrid(*([np.linspace(0, 1, 6, endpoint=False)] * 2),
                             indexing="ij"), axis=-1).reshape(-1, 2)
    assert _detector._find_intertwiners(out.model, -np.eye(2), k, antiunitary=True).dim == 0
    # The clean model's *beyond-grey* composite (C2z.T, k-action +1 and
    # antiunitary — `test_groups.py` measures its dim = 2) is broken by the
    # jitter like everything else, and it is **not** restored: it is not an
    # operation of the cell that was asked, so it is not among the three the
    # average ran over.  Symmetrization projects onto the group it was handed
    # and no further, which is the same discipline `realized_group` states for
    # magnetic models.
    assert _detector._find_intertwiners(jit, np.eye(2), k, antiunitary=True).dim == 0
    assert _detector._find_intertwiners(out.model, np.eye(2), k, antiunitary=True).dim == 0

    # and the symmetrized model IS exactly symmetric under the three it kept --
    # measured against the operators it was built with, and against the stacked
    # oracle.  The eigen route's own re-detection reads O(1e-8) here and says so
    # on the record: its accuracy floor is eps*||H||/(smallest level splitting),
    # and the jitter split this model's spin degeneracy at 1e-7.
    assert rec.checks["covariance_residue"] < 1e-12
    again = _measure._validate_group(out.model, out.group, nk=6, solver="stacked")
    assert not again.broken and max(again.residuals.values()) < 1e-13
    if not rec.verification["near_kept"]:
        assert "smallest level splitting" in rec.verification["note"]


# ---------------------------------------------------------------------------
# 4.  broken is not near
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_a_genuine_distortion_is_descended_to_and_never_symmetrized_away(lin3g):
    """1e-2 on one hopping is not noise, and nothing symmetrizes it away.

    The layer cell of this model is the slab point set as given, so it allows
    ``P4/mmm1' #123.340`` — 32 operations, ``σ_h`` and everything composed with
    it included — and the Hamiltonian breaks **24** of them.  Sixteen are
    refused ``"no symmetry matrix"``, which is a *proof*: the 4-fold axis, the
    diagonal mirrors and everything ``σ_h`` makes of them, all of which the
    distortion genuinely kills.  The other eight are refused by **tensor
    structure**: each shares its in-plane k-action with the ``σ_h``-free member
    of its fiber, so the detector cannot separate the two and
    ``D(g) = O(g) ⊗ S(g)`` does.  They are exactly the ``σ_h`` partners of the
    four operations that survive — ``m(z) = E·σ_h``, ``i = C2(z)·σ_h``,
    ``C2(x) = m(y)·σ_h``, ``C2(y) = m(x)·σ_h`` — and their time-reversed
    copies.

    So the symmetrizing pass is never reached: there is no ``near`` operation to
    license it.  What the pipeline does instead is what a genuine distortion
    deserves: it **descends**, and the group it serves is the same
    ``Pmm21' #25.58``, 8 operations, that the in-plane reading served before
    this wave; only the upper bound it descended *from* is bigger.  The four
    unitary operations kept are ``E``, ``C2(z)`` and the two vertical mirrors
    ``m(x)``, ``m(y)``, with their time-reversed partners.
    """
    bad = distort(lin3g, rel=1e-2)
    out = tqg.realized_group(bad, environment="free-standing", typat=[0, 0, 1, 1],
                             tol=1e-6,
                             characters=["s", "dx2-y2", "s", "s"], cross_check=False)
    assert out, str(out)
    assert out.symmetrization is None, "broken is not near: nothing was projected"
    assert (out.group.number_str, out.group.name, out.group.size) == ("25.58",
                                                                     "Pmm21'", 8)
    assert out.group.provenance == "descended"
    assert out.descent["from"]["number_str"] == "123.340"
    assert out.descent["from"]["name"] == "P4/mmm1'"
    assert out.descent["from"]["size"] == 32
    assert len(out.descent["dropped"]) == 24 and out.descent["iterations"] == 1
    assert {b["name"] for b in out.descent["dropped"]} == {
        "C4(z)", "C4^-1(z)", "m([110])", "m([1-10])", "C2(x)", "C2(y)",
        "C2([110])", "C2([1-10])", "S4(z)", "S4^-1(z)", "i", "m(z)",
        "C4(z)'", "C4^-1(z)'", "m([110])'", "m([1-10])'", "C2(x)'", "C2(y)'",
        "C2([110])'", "C2([1-10])'", "S4(z)'", "S4^-1(z)'", "i'", "m(z)'"}
    assert {b["refuted_by"] for b in out.descent["dropped"]} == {
        "no symmetry matrix", "tensor structure"}
    assert out.descent["steps"][0]["refuted_by"] == {"no symmetry matrix": 16,
                                                     "tensor structure": 8}


def test_symmetrize_refuses_an_operation_that_is_broken_rather_than_near(lin3g):
    """The refusal that makes the record mean something.

    ``_validate_group``'s own gate cannot hand ``symmetrize`` a broken operation,
    so the case is provoked with a hand-built record that *lists* the eight
    refuted operations as survivors.  Symmetrizing over them would impose a
    symmetry this Hamiltonian does not have, and the refusal names them.
    """
    bad = distort(lin3g, rel=1e-2)
    m3, group, asg, val = _pipeline(bad, [0, 0, 1, 1], tol=1e-6)
    assert len(val.indices) == 8
    forced = dataclasses.replace(
        val, indices=tuple(int(s.ind) for s in group.symmetries))
    got = symmetrize(m3, group, asg, forced, tol=1e-6,
                     characters=["s", "dx2-y2", "s", "s"])
    assert not got and got.gate == "symmetrize_broken_operation"
    # the eight by NAME, not by `_validate_group`'s own list of them: the 4-fold
    # axis, the two diagonal mirrors and their time-reversed partners, which is
    # exactly what a 1e-2 distortion of one hopping kills
    lat = np.asarray(group.lattice, dtype=float)
    assert {_sitechars.operation_name(s, lat) + ("'" if s.time_reversal else "")
            for s in group.symmetries
            if int(s.ind) in got.context["broken"]} == {
        "C4(z)", "C4^-1(z)", "m([110])", "m([1-10])",
        "C4(z)'", "C4^-1(z)'", "m([110])'", "m([1-10])'"}
    assert got.threshold == 1e-6 and not np.isfinite(got.value)
    assert "impose a symmetry this Hamiltonian does not have" in got.reason


@pytest.mark.slow
def test_the_covariance_refusal_points_at_the_route_when_a_tolerance_was_declared(
        lin3g_jittered, monkeypatch):
    """``symmetrize=False`` leaves the near model to the exact gate, which refuses.

    When the detector's *declared* tolerance lies above ``_build_rep``'s own
    gate and the best candidate lands between the two, the operations are
    near-symmetric and the refusal says so — factually, with both numbers, and
    without moving either.  The group is identified either way, so the refusal
    opens by naming it and carries it in ``context["group"]``.

    The representation the projection is made with is gated at the declared
    tolerance itself, every operation having been kept at it, so no candidate
    closing it is three measurements of what the Hamiltonian obeys
    disagreeing: it raises, as the returned representation's does beyond
    ``tol``, rather than coming back as a refusal.  Provoked here.

    ``rep_search_too_large`` is the same story with a different gate: a
    practical limit on the search rather than a defect, so it stays a returned
    refusal too, naming the group and carrying it.  ``lin3g`` reaches it on the
    clean model with the enumeration forced and its ceiling set small, no jitter
    needed; the same ceiling through the public call is no obstacle to the
    read-off, which never enumerates.  So is
    ``projection_too_large``, the other practical limit: the hopping
    projection's memory at its peak.  ``max_bytes=`` is ``realized_group``'s
    own budget for it, at ``symmetrize``'s default, and it reaches the
    projection: on the jittered model's grid, 7 x 7 points at 8 states, the
    projection holds the Hamiltonian and its group average, 2.1 grids of
    ``16 x 49 x 8**2`` bytes as measured, and streams one k-point of the rest
    beside them, 113 kB in all against a budget of ``1e5``; the budget the
    refusal names runs the same projection the default does, in slices of the
    grid.
    """
    projected, model, tol = lin3g_jittered
    got = tqg.realized_group(model, typat=[0, 0, 1, 1], tol=tol, symmetrize=False,
                             characters=["s", "dx2-y2", "s", "s"])
    assert not got and got.gate == "rep_not_covariant"
    # the threshold is this model's own exact gate -- its 16 operations on 8
    # states at max|H| = 12.375 lift it above the floor -- and never the declared tol
    gate = tqs.covariance_gate(len(got.context["indices"]), int(model.nstate), _max_H(model))
    assert got.threshold == pytest.approx(gate, rel=1e-12) and gate > EXACT_GATE
    assert gate < got.value <= tol
    assert "near-symmetric" in got.reason and "symmetrize" in got.reason
    assert "The group identified is P4mm1' #99.164" in got.reason
    group = got.context["group"]
    assert group.name == "P4mm1'" and group.number_str == "99.164"

    # the projection's own representation, provoked to miss its gate
    import pythtqc.symmetrization as _projection

    real_build_rep = _projection._build_rep

    def refusing(*args, **kwargs):
        out = real_build_rep(*args, **kwargs)
        if isinstance(out, Refusal):
            return out
        return Refusal("rep_not_covariant", "provoked: no candidate closes the gate.",
                       value=0.5, threshold=float(kwargs["tol"]),
                       context={"n_candidates": 2})

    patch_everywhere(monkeypatch, "_build_rep", refusing)
    with pytest.raises(InternalError, match="onto its invariant part, still does not") as err:
        tqg.realized_group(model, typat=[0, 0, 1, 1], tol=tol,
                           characters=["s", "dx2-y2", "s", "s"])
    assert "P4mm1' #99.164" in str(err.value) and "reaches 0.5 against" in str(err.value)
    monkeypatch.undo()

    # rep_search_too_large: the same group named and carried
    # the enumeration forced through the private entry, which takes the strategy
    capped = _realized._realize(tqc_models.lin3g(), typat=[0, 0, 1, 1],
                          characters=["s", "dx2-y2", "s", "s"],
                          strategy="enumerate", max_candidates=8)
    assert not capped and capped.gate == "rep_search_too_large"
    assert capped.value == 32 and capped.threshold == 8
    assert "The group identified is P4mm1' #99.164" in capped.reason
    assert capped.context["group"].number_str == "99.164"
    roomy = tqg.realized_group(tqc_models.lin3g(), typat=[0, 0, 1, 1],
                               characters=["s", "dx2-y2", "s", "s"], max_candidates=8,
                               cross_check=False)
    assert roomy, str(roomy)
    assert roomy.rep._search["strategy"] == "readoff"
    assert (roomy.rep._search["max_candidates"], roomy.rep._search["n_enumeration"]) == (8, 32)

    # projection_too_large: the projection's memory, the same group named and
    # carried, and the budget it names runs the projection the default ran
    assert (inspect.signature(tqg.realized_group).parameters["max_bytes"].default
            == inspect.signature(symmetrize).parameters["max_bytes"].default == 1e9)
    tight = tqg.realized_group(model, typat=[0, 0, 1, 1], tol=tol, max_bytes=1e5,
                               characters=["s", "dx2-y2", "s", "s"])
    assert not tight and tight.gate == "projection_too_large"
    grids = 2.1 * 16 * 49 * 8 ** 2
    assert grids < tight.value == pytest.approx(113033.6, rel=1e-12, abs=0)
    assert tight.threshold == 1e5
    assert "113 kB at its peak" in tight.reason
    assert "The group identified is P4mm1' #99.164" in tight.reason
    assert tight.context["group"].number_str == "99.164"
    assert "`realized_group(..., max_bytes=1.2e5)` runs it" in tight.reason
    runs = tqg.realized_group(model, typat=[0, 0, 1, 1], tol=tol, max_bytes=1.2e5,
                              characters=["s", "dx2-y2", "s", "s"], cross_check=False)
    assert runs, str(runs)
    assert runs.symmetrization.grid == projected.symmetrization.grid
    # the grid is evaluated in chunks the budget sizes, so the same projection
    # agrees to rounding rather than bit for bit
    assert runs.symmetrization.defect == pytest.approx(projected.symmetrization.defect,
                                                       rel=1e-6)


# ---------------------------------------------------------------------------
# 5.  idempotence
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_symmetrizing_a_symmetrized_model_changes_nothing(lin3g_jittered):
    """Idempotence: every operation is ``exact`` afterwards, and a second pass moves 5e-15."""
    out, _model, tol = lin3g_jittered
    m3 = out.model
    asg = _tables._assign_orbitals(m3, out.group, typat=[0, 0, 1, 1])
    val = _measure._validate_group(m3, out.group, nk=6, tol=tol)
    assert set(classify_operations(val, tol).values()) == {"exact"}
    sym2, rec2 = symmetrize(m3, out.group, asg, val, tol=tol,
                            characters=["s", "dx2-y2", "s", "s"])
    assert rec2.near == () and rec2.passes == 1
    k = np.random.default_rng(9).uniform(-0.5, 0.5, size=(24, 2))
    a = np.asarray(m3.hamiltonian(k_pts=k, flatten_spin_axis=True))
    b = np.asarray(sym2.hamiltonian(k_pts=k, flatten_spin_axis=True))
    assert float(np.abs(a - b).max()) < 1e-13
    assert int(sym2.nhops) == int(m3.nhops)


# ---------------------------------------------------------------------------
# 6.  the grammar
# ---------------------------------------------------------------------------

def test_the_grammar_gates_are_ValueErrors(lin3g):
    """A missing or impossible declaration is a bug in the caller, not a verdict.

    A transform grid too coarse for the model's own hopping range would alias,
    so that call is malformed too: grammar, not physics.  So is a memory budget
    that is not a number of bytes, or a ceiling on a search that is not a count
    of candidates, raised before anything is measured; a budget too small for
    the grid is a limit instead, and comes back as a refusal.
    """
    tol = TOL_FACTOR * JITTER * _max_H(lin3g)
    m3, group, asg, val = _pipeline(lin3g, [0, 0, 1, 1], tol=tol)
    with pytest.raises(TypeError):
        symmetrize(m3, group, asg, val)                     # tol is required
    with pytest.raises(ValueError, match="positive"):
        symmetrize(m3, group, asg, val, tol=0.0)
    with pytest.raises(ValueError, match="too coarse"):
        symmetrize(m3, group, asg, val, tol=tol, k_grid=5)
    with pytest.raises(ValueError, match="symmetrize must be"):
        tqg.realized_group(m3, typat=[0, 0, 1, 1], symmetrize="yes")
    with pytest.raises(ValueError, match="max_bytes must be a positive"):
        symmetrize(m3, group, asg, val, tol=tol, max_bytes=0)
    with pytest.raises(ValueError, match="max_bytes must be a positive"):
        tqg.realized_group(m3, typat=[0, 0, 1, 1], max_bytes=-1.0)
    with pytest.raises(ValueError, match="max_candidates must be a positive"):
        symmetrize(m3, group, asg, val, tol=tol, max_candidates="many")
    with pytest.raises(ValueError, match="max_candidates must be a positive"):
        tqg.realized_group(m3, typat=[0, 0, 1, 1], max_candidates=0)
    tight = symmetrize(m3, group, asg, val, tol=tol, max_bytes=1e5)
    assert not tight and tight.gate == "projection_too_large"


# ---------------------------------------------------------------------------
# 7.  a multiplet site
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_a_multiplet_site_symmetrizes_through_the_representation_nearest_its_blocks(p4232):
    """A multiplet's own block carries the noise, and the operators do not.

    On a one-dimensional channel the ansatz's internal factor ``S(g)`` is the
    group's own ``sym.spinor_rotation`` — a *table*, carrying no noise.  Each
    site of this model glues its two orbitals into one two-dimensional channel,
    and its factor is the model's own intertwiner's block, which on this
    1e-7-jittered model composes by the group's table only to the model's
    noise.  The operators are the exact representation nearest those blocks —
    an average over the group, with the table's own factors — so ``D``
    satisfies the group's relations to rounding, the projection runs, and one
    pass moves the hoppings by the model's own noise and no more.

    The multiplets here are a measurement and not an ask: nothing is stated,
    and each site's two orbitals are the pair this model's own Hamiltonian
    binds.  The same holds read spinless, through the one declaration a
    numerical model needs: spinors in the orbital index of an ``R-3m`` model,
    each pair one measured doublet, jittered, and read with ``precision=``,
    return their group with the pairs' own blocks made exact rather than
    refused ``canonical_norm``.  That holds at every noise the declaration
    covers.  Which operations bind a site's orbitals is read at the declared
    tolerance too, so at a noise of ``1e-5`` the mirrors, whose members miss
    the whole-site tensor form by about ``1e-6``, still bind each spinor pair
    -- read at the exact ``1e-6`` they did not, the pair came apart into two
    orbitals no mirror can carry, and no candidate came within ``0.03`` of
    covariance; and the blocks the projection's representation is read from,
    off each operation's noisy space, are held to one another at the declared
    precision, where ten times the worst residual alone was a little tighter
    than they agree (a jitter drawn from seed 3).
    """
    cell = tqc_models.p4232_symmetry_cell()
    decorated = TBModel(Lattice(np.eye(3), cell["positions"].tolist(),
                                periodic_dirs=[0, 1, 2]), spinful=False)
    group = _spacegroup._group_of(decorated, typat=cell["typat"].tolist())
    assert group and group.sg_number == 208, str(group)

    jit = jitter(p4232)
    tol = TOL_FACTOR * JITTER * _max_H(p4232)
    asg = _tables._assign_orbitals(jit, group)
    val = _measure._validate_group(jit, group, nk=5, tol=tol)
    assert len(val.indices) == 48, "the whole group survives at the declared tolerance"
    assert set(classify_operations(val, tol).values()) == {"exact", "near"}
    assert asg.site_orbitals == ((0, 1), (2, 3))

    model, rec = symmetrize(jit, group, asg, val, tol=tol)
    assert rec.passes == 1
    assert rec.checks["covariance_residue"] < 1e-13
    assert rec.checks["k_identity"] < 1e-13
    assert 0.0 < rec.checks["per_pass"][0] < 10.0 * JITTER * _max_H(p4232), \
        "the hoppings moved by the model's own noise and no more"
    assert 1e-9 < rec.operators["composition"] < 1e-5, \
        "the pairs' own blocks carried the noise the operators do not"

    flat, _basis = tqspin.spinless_presentation(tqc_models.r3m_s_6c())
    for noise, seed in ((JITTER, 7), (JITTER, 3), (1e-5, 1)):
        spinless = tqg.realized_group(jitter(flat, rel=noise, seed=seed),
                                      precision=10 * noise)
        assert spinless, str(spinless)
        assert (spinless.group.name, spinless.group.number_str) == ("R-3m", "166")
        assert spinless.rep.multiplets.by_site == (((0, 1),), ((2, 3),))
        assert spinless.rep.characters.irreps == ("E",) * 4
        assert spinless.symmetrization is not None
        assert spinless.symmetrization.operators["composition"] > 1e-9
        assert spinless.rep.product_residual < 1e-13


def _thick_slab(shifts=None):
    """Three square layers 6 A apart in a 30 A cell: 12 A thick, 18 A of vacuum.

    The outer layers hold two sites each, at ``(1/2, 0)`` and ``(0, 1/2)``, which
    the fourfold exchanges, and the middle one a site at the origin: the cell is
    ``P4/mmm``.  ``shifts`` moves each centre, in reduced coordinates.
    """
    lat = np.zeros((3, 3))
    lat[:2, :2] = np.eye(2)
    lat[2] = [0.0, 0.0, 30.0]
    pos = np.array([[0.5, 0.0, 0.2], [0.0, 0.5, 0.2], [0.0, 0.0, 0.0],
                    [0.5, 0.0, -0.2], [0.0, 0.5, -0.2]])
    if shifts is not None:
        pos = pos + np.asarray(shifts, dtype=float)
    m = TBModel(Lattice(lat, pos.tolist(), periodic_dirs=[0, 1]))
    m.set_onsite([0.0, 0.0, 0.3, 0.0, 0.0])
    for a, b in ((0, 1), (3, 4)):
        for R in ([0, 0, 0], [1, 0, 0], [0, -1, 0], [1, -1, 0]):
            m.set_hop(-1.0, a, b, R, mode="add")
    for R in ([1, 0, 0], [0, 1, 0]):
        m.set_hop(-0.5, 2, 2, R, mode="add")
    for outer, Rs in ((0, ([0, 0, 0], [-1, 0, 0])), (1, ([0, 0, 0], [0, -1, 0])),
                      (3, ([0, 0, 0], [-1, 0, 0])), (4, ([0, 0, 0], [0, -1, 0]))):
        for R in Rs:
            m.set_hop(-0.4, 2, outer, R, mode="add")
    return m


def _move_one_centre(model, delta, which=1, direction=0):
    """The same model with one orbital centre displaced — the w90 defect, isolated."""
    orb = np.asarray(model.orb_vecs, dtype=float).copy()
    orb[which, direction] += delta
    dim_r = int(model.lattice.dim_r)
    out = TBModel(Lattice(np.asarray(model.lat_vecs, dtype=float), orb.tolist(),
                          periodic_dirs=list(model.periodic_dirs)),
                  spinful=(int(model.nspin) == 2))
    out.set_onsite(np.asarray(model.onsite), mode="set")
    for h in model.hoppings:
        out.set_hop(h["amplitude"], h["from_orbital"], h["to_orbital"],
                    list(h.get("lattice_vector", [0] * dim_r)), mode="add")
    return out


def _lifted_supercell(sc, lift=0.3, seed=0):
    """A Fu-Kane-Mele supercell with its folding degeneracy lifted, and its typing.

    The construction ``test_detector.py`` and ``test_symmetry.py`` § 11b both
    use: a random **inversion-symmetric** on-site potential breaks the small
    cell's translations and the star degeneracy while leaving inversion exact,
    and the sites are typed by inversion orbit because the bare supercell's
    *positions* still carry the small cell's translations — ``_group_of`` reads
    the index-8 lattice supergroup off it (wave R4a; before that it refused
    ``non_primitive_cell``), and the typing is the truth about this Hamiltonian
    as well as much the cheaper route.
    """
    model = tqc_models.fu_kane_mele_z2("strong").make_supercell(sc, to_home=True)
    pos = np.asarray(model.orb_vecs, dtype=float) % 1.0
    perm = None
    for j in range(pos.shape[0]):
        target = (pos[0] + pos[j] - pos) % 1.0
        d = np.abs((pos[None, :, :] - target[:, None, :] + 0.5) % 1.0 - 0.5).max(axis=2)
        if d.min(axis=1).max() < 1e-9:
            p = d.argmin(axis=1)
            if len(set(p.tolist())) == pos.shape[0]:
                perm = np.asarray(p)
                break
    assert perm is not None
    rng = np.random.default_rng(seed)
    e = rng.normal(size=model.norb)
    e = 0.5 * (e + e[perm])
    for i in range(model.norb):
        model.set_onsite(lift * float(e[i]), i, mode="add")
    return model, [int(min(i, int(perm[i]))) for i in range(model.norb)]


def block_jitter(model, rel=JITTER, seed=7):
    """One **real** factor per hopping block: breaks the space group, keeps time reversal.

    The entry-by-entry :func:`jitter` above breaks the SU(2) structure of each
    spinful block and therefore time reversal too, which is right for a
    wannier90 defect and wrong for the fixture below, where the point is a
    **grey** group whose generator is a Hermitian involution.
    """
    rng = np.random.default_rng(seed)
    dim_r = int(model.lattice.dim_r)
    out = TBModel(Lattice(np.asarray(model.lat_vecs, dtype=float),
                          np.asarray(model.orb_vecs, dtype=float).tolist(),
                          periodic_dirs=list(model.periodic_dirs)),
                  spinful=(int(model.nspin) == 2))
    ons = np.asarray(model.onsite, dtype=complex)
    f = 1.0 + rel * rng.normal(size=ons.shape[0])
    out.set_onsite(ons * f.reshape(-1, 1, 1) if ons.ndim == 3 else (ons.real * f),
                   mode="set")
    for h in model.hoppings:
        amp = np.asarray(h["amplitude"], dtype=complex) * (1.0 + rel * rng.normal())
        out.set_hop(amp if amp.ndim else complex(amp), h["from_orbital"],
                    h["to_orbital"], list(h.get("lattice_vector", [0] * dim_r)),
                    mode="add")
    return out


@pytest.mark.slow
def test_a_tied_orientation_branch_is_measured_the_same_representation():
    """The ambiguity a real wannier90 group hits, and why it is not one *here*.

    A group whose generators are all involutions with Hermitian representatives
    — inversion and a two-fold axis, which is most of what a real crystal offers
    — builds the **same** unitary matrices on both orientation branches,
    differing only by a phase per generator.

    **Re-pinned in wave R4e.**  ``_build_rep`` used to assert uniqueness on the
    pre-normalization candidates, see two, and refuse ``rep_not_unique``, and
    this pass then decided the tie for itself by measuring the phase-fitted
    difference ``min_λ ‖D₂(g) − λD₁(g)‖`` between the two separately built
    branches (measured 0.0, and recorded as ``branch_ambiguity``).  The tie is
    now settled where it belongs — in ``_build_rep``, by convention, with
    ``branch_tie`` saying so — so this pass never reaches its own fallback and
    ``branch_ambiguity`` is ``None``.  The claim the test was written for is
    unchanged and is asserted directly here instead: the two branches build the
    identical matrices, measured **0.0** on every operation.  The fallback
    itself stays reachable and is still needed, for the branch tie that arrives
    as ``rep_search_too_large`` (the read-off deferred for some other reason and
    the enumeration is too big to run).

    Measured on the block-jittered cell, where the tie meets a near-symmetric
    model and the pass has something to move.
    """
    model, typat = _lifted_supercell([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    tol = TOL_FACTOR * JITTER * _max_H(model)
    model = block_jitter(model)
    group = _spacegroup._group_of(model, typat=typat)
    assert group and group.number_str == "2.5", str(group)
    asg = _tables._assign_orbitals(model, group, typat=typat)
    val = _measure._validate_group(model, group, nk=3, tol=tol)
    assert len(val.indices) == 4, "the grey group survives the block jitter"

    # the tie this test is about, now served rather than refused
    rep = _build._build_rep(model, group, asg, validated=val, tol=tol, internal_tol=1e-5)
    assert rep, str(rep)
    assert rep._branch_tie is True and rep._choice["orientation"] == 1
    assert rep._search["n_survivors"] == 2, "both branches closed all three gates"
    other = _build._build_rep(model, group, asg, validated=val, tol=tol, internal_tol=1e-5,
                           orientation_branch=-1)
    assert other, str(other)
    assert max(float(np.abs(rep.matrices[i] - other.matrices[i]).max())
               for i in rep.indices) == 0.0

    sym, rec = symmetrize(model, group, asg, val, tol=tol)
    assert rec.verification["near_kept"] and not rec.verification["broken"]
    assert rec.verification["worst_residual"] < 1e-12
    assert len(rec.near) == 2 and rec.defect > 1e-9
    # O(the residuals that licensed it) -- measured 0.016 of this bound
    assert rec.defect <= 10 * max(rec.residuals[i] for i in rec.near)


# ---------------------------------------------------------------------------
# 8.  the position defect, measured FIRST, against a declared tolerance
#     (wave P3b: ``realized_group(position_tol=)``)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_a_declared_position_tol_snaps_the_centres_before_anything_else(
        km_odd, km_odd_free_standing):
    """The public regression that motivated the wiring, through the one call.

    One Kane-Mele centre displaced by 1e-8 gives a position defect of **2.0e-8**
    (the defect is ``|AΔ − Δ|``, up to twice the displacement) and costs the
    Hamiltonian its symmetry outright: ``2π·δ·max|H|`` = 3.7e-7 against the
    default tolerance 1e-8, so with nothing declared about the centres the call
    refuses ``positions_not_exactly_symmetric`` as soon as the layer cell's
    ``P-6m21' #187.210`` is identified, before the detector runs.  Declaring
    ``position_tol = 1e-6`` says those centres are
    noise; they are fitted about their own origin -- a shift of every centre
    together is no displacement -- so the one centre's 1e-8 is shared, each
    centre moved by **5.0e-9** onto the positions the group puts them at, and the
    shift of both together that leaves, 5e-9, is itself within the declaration
    and removed: the centres come back exactly where the clean model has them,
    in its own frame, the defect goes to rounding, and every operation of the
    group the Hamiltonian then realizes reads *exact* (worst residual 1.5e-14,
    below ``_build_rep``'s own 5e-14) — so the **hopping** pass never runs at
    all, and what comes back is the clean model's physics.

    That served group is ``P3m11' #156.50``, twelve operations, reached from the
    24-operation **free-standing** cell by a **descent**: the two sublattices are
    declared inequivalent here, so the Hamiltonian's ``σ_h`` fibers cut the cell
    in half.  That reading is declared, since wave R4c reads a ``dim_r = 2``
    model on a substrate unless told otherwise — and there the ``σ_h`` family is
    not in the cell for a centre defect to be measured against.
    The clean model reaches the same group the same way, which is the point —
    the displacement changes the numbers on the way, not the answer.

    The move is a k-dependent gauge transformation, so it is free in the only
    sense that matters: the spectrum of the model that comes back is the
    **displaced** model's, to 5.3e-15 on a scale of 3.21.

    ``precision=1e-6`` says the same thing about the centres that
    ``position_tol=1e-6`` says, and moves them the same way; it also declares a
    tolerance for the Hamiltonian, which changes nothing here, because with the
    centres right every operation is exact and the hopping pass never runs.
    """
    clean = km_odd_free_standing
    displaced = _move_one_centre(km_odd, 1e-8)

    out = tqg.realized_group(displaced, environment="free-standing", typat=[0, 1],
                             characters={0: "A1"},
                             position_tol=1e-6)
    assert out, str(out)

    # the measurement, and what it licensed
    pos = out.positions
    assert pos is not None and pos["adjusted"] is True
    assert pos["position_tol"] == 1e-6
    assert pos["defect"] == pytest.approx(2e-8, rel=1e-6, abs=0)
    assert pos["max_move"] == pytest.approx(5e-9, rel=1e-6, abs=0), \
        "half the displacement each: the origin is fitted to the centres"
    assert pos["defect_after"] < 1e-14, "the centres now satisfy the group exactly"
    # ... and the shift of both together the fit left is noise, removed: the
    # centres are the clean model's, the displaced one back by the whole 1e-8
    assert pos["frame"] == "model"
    assert pos["offset"] == pytest.approx((-5e-9, 0.0, 0.0), rel=1e-6, abs=1e-15)
    assert float(np.abs(np.asarray(out.model.orb_vecs)
                        - np.asarray(clean.model.orb_vecs)).max()) <= _FEW_ULP
    # the gauge identity, measured on eight random k: 1.3e-15
    assert pos["n_check"] == 8 and 0.0 < pos["spectrum_residual"] < 1e-13, \
        "moving a centre is a gauge, not a change"
    assert "centres moved" in repr(out)
    # the slab presentation survives: the padded third coordinate has no group
    # average to move to (every decorated operation has R[2,2] = +1) and the
    # bit-identity is still measured against the presentation as BUILT
    assert np.all(np.asarray(out.model.orb_vecs)[:, 2] == 0.0)
    assert out.presentation["residual"] == 0.0
    assert list(out.model.periodic_dirs) == [0, 1]

    # ... and the hoppings needed nothing: every operation is exact afterwards
    assert out.symmetrization is None
    assert set(classify_operations(out.validated, 1e-8).values()) == {"exact"}
    assert max(out.validated.residuals[i] for i in out.validated.indices) < EXACT_GATE
    assert len(out.validated.indices) == out.group.size == 12

    # the physics is the clean model's, exactly -- the same group by the same
    # route, descent record and all
    assert out.group.number_str == clean.group.number_str == "156.50"
    assert out.group.provenance == clean.group.provenance == "descended"
    assert out.descent["from"]["number_str"] == "187.210"
    assert out.descent["from"] == clean.descent["from"]
    assert len(out.descent["dropped"]) == len(clean.descent["dropped"]) == 12
    assert out.rep._choice == clean.rep._choice
    assert max(float(np.abs(out.rep.matrices[i] - clean.rep.matrices[i]).max())
               for i in clean.rep.indices) == 0.0
    assert out.rep.characters.irreps == clean.rep.characters.irreps
    a = tqw.hsp_table(clean.rep, bands=(0, 1))
    b = tqw.hsp_table(out.rep, bands=(0, 1))
    assert a.complete and b.complete
    assert a.counts == b.counts == {"-GM6": 1, "-K5": 1, "-K6": 1, "-M3M4": 1}
    assert a.partitions == b.partitions
    assert {p: dict(v) for p, v in a.content.items()} == {p: dict(v) for p, v in
                                                          b.content.items()}
    da = tqe.decompose(a, clean.group, rep=clean.rep, bands=2)
    db = tqe.decompose(b, out.group, rep=out.rep, bands=2)
    assert da.verdict == db.verdict == "symmetry-indicated decomposable"
    assert set(da.solution_strings()) == set(db.solution_strings())

    # the gauge: the spectrum is the DISPLACED model's, not merely the clean one's
    k = np.random.default_rng(11).uniform(-0.5, 0.5, size=(16, 2))
    Ea = np.linalg.eigvalsh(np.asarray(_layers._slab_presentation(displaced).hamiltonian(
        k_pts=k, flatten_spin_axis=True)))
    Eb = np.linalg.eigvalsh(np.asarray(out.model.hamiltonian(
        k_pts=k, flatten_spin_axis=True)))
    assert float(np.abs(Ea - Eb).max()) < 1e-13

    # the one declaration for a numerical model says the same about the centres
    prec = tqg.realized_group(displaced, environment="free-standing", typat=[0, 1],
                              characters={0: "A1"}, precision=1e-6, cross_check=False)
    assert prec, str(prec)
    assert prec.positions == out.positions, "the centres moved exactly as position_tol does"
    assert prec.symmetrization is None, "and the hoppings needed nothing, whatever tol"

    # `position_tol=None` refuses before the detector runs: centres that miss the
    # group's action by 2e-8 cost H(k) a covariance residual of 3.7e-7, above
    # the default tolerance, so every verdict read after it would be the
    # geometry's as much as the hoppings'.  The measured defect opens the
    # refusal -- the number to act on -- and the layer's `P-6m21' #187.210`, 24
    # operations, is named as the upper bound it is.  The threshold quoted is
    # the defect that tolerance allows, tol / (2*pi*max|H|).
    head = tqg.realized_group(displaced, environment="free-standing", typat=[0, 1],
                              characters={0: "A1"})
    assert not head and head.gate == "positions_not_exactly_symmetric"
    assert head.value == pytest.approx(2e-8, rel=1e-6, abs=0)
    assert head.context["position_defect"] == head.value
    assert head.threshold == pytest.approx(
        head.context["tol"] / (2 * np.pi * head.context["max_h"]))
    assert head.context["tol"] == 1e-8
    assert head.context["group"].number_str == "187.210"
    assert head.context["group"].size == 24
    assert head.reason.startswith("the orbital centres miss the cell's own symmetry")
    assert "precision=" in head.reason and "position_tol=" in head.reason
    assert "positions" not in head.context, "none was declared, so none was made"


@pytest.mark.slow
def test_the_position_pass_runs_before_the_hopping_projection(km_odd,
                                                              km_odd_free_standing):
    """The order, on a model carrying **both** defects — and it is the answer.

    A 1e-8 centre displacement *and* every hopping jittered at 1e-7.  With
    ``position_tol`` declared the centres are snapped first (defect 2.0e-8 →
    3.9e-16), so the projection that follows sees a geometry that satisfies the
    group exactly — ``operators["position_defect"]`` on its own record reads
    3.9e-16 — classifies eleven of the twelve operations of the descended
    ``P3m11' #156.50`` *near*, moves ``H`` by 2.6e-7 against the declared
    3.0e-6, and returns the clean model's group, matrices, labels, content and
    EBR verdict.

    Without the declaration the centres' defect costs ``H(k)`` a covariance
    residual of order ``2π·δ·max|H|`` = 3.7e-7, below the declared ``tol`` of
    3.0e-6: at the detector's tolerance it breaks nothing, so it is not refused
    up front.  The **free-standing** cell — declared, since wave R4c reads a flat
    model on a substrate by default — allows ``P-6m21' #187.210``, 24
    operations, and descends by tensor structure to the same ``P3m11'
    #156.50``; the hopping projection is then asked for, and refuses: its
    operators are built on centres the operations carry to ``1e-9``, these
    miss them by 2e-8, and no projection of the hoppings reaches a defect in
    the centres.  It refuses ``positions_not_exactly_symmetric``, naming the
    group, quoting the defect -- the number to act on -- and ``precision=``,
    which is what fixes it.
    """
    clean = km_odd_free_standing
    both = jitter(_move_one_centre(km_odd, 1e-8))
    tol = TOL_FACTOR * JITTER * _max_H(km_odd)

    out = tqg.realized_group(both, environment="free-standing", typat=[0, 1],
                             characters={0: "A1"}, tol=tol,
                             position_tol=1e-6, cross_check=False)
    assert out, str(out)
    pos, rec = out.positions, out.symmetrization
    assert pos["adjusted"] is True and pos["defect"] == pytest.approx(2e-8, rel=1e-6, abs=0)
    assert pos["defect_after"] < 1e-14
    assert rec is not None, "the hoppings still carry their own defect"
    # the ordering, as a number: the projection ran on exact centres, the moved
    # ones, which miss the group by rounding as the pass measured them
    assert rec.operators["position_defect"] < 1e-14 and pos["defect_after"] < 1e-14
    assert len(rec.near) == 11 and len(rec.classes) == 12
    assert rec.defect <= 10 * max(rec.residuals[i] for i in rec.near)
    assert rec.tol == tol and rec.passes == 1
    assert rec.checks["outside_box"] < 1e-13 and rec.checks["k_identity"] < 1e-12
    assert rec.verification["near_kept"] and not rec.verification["broken"]
    assert "centres moved" in repr(out) and "symmetrized (defect" in repr(out)

    assert out.group.number_str == clean.group.number_str == "156.50"
    assert out.group.provenance == clean.group.provenance == "descended"
    assert out.descent["from"] == clean.descent["from"]
    assert out.group.size == len(rec.classes) == 12
    assert out.rep._choice == clean.rep._choice
    assert max(float(np.abs(out.rep.matrices[i] - clean.rep.matrices[i]).max())
               for i in clean.rep.indices) == 0.0
    assert out.rep.characters.irreps == clean.rep.characters.irreps
    a = tqw.hsp_table(clean.rep, bands=(0, 1))
    b = tqw.hsp_table(out.rep, bands=(0, 1))
    assert a.complete and b.complete and a.counts == b.counts
    da = tqe.decompose(a, clean.group, rep=clean.rep, bands=2)
    db = tqe.decompose(b, out.group, rep=out.rep, bands=2)
    assert da.verdict == db.verdict == "symmetry-indicated decomposable"
    assert set(da.solution_strings()) == set(db.solution_strings())

    # the same call with the centres undeclared: their defect is inside the
    # declared tol, so the descent is the clean model's, and the projection
    # refuses centres its operators do not carry, with the defect and the route
    head = tqg.realized_group(both, environment="free-standing", typat=[0, 1],
                              characters={0: "A1"}, tol=tol)
    assert not head and head.gate == "positions_not_exactly_symmetric"
    assert head.context["group"].number_str == "156.50"
    assert head.value == pytest.approx(2e-8, rel=1e-6, abs=0)
    assert head.threshold == 1e-9, "the tolerance the projection's operators carry them to"
    assert head.context["position_defect"] == head.value
    assert "precision=" in head.reason


@pytest.mark.slow
def test_centres_too_far_from_symmetric_break_the_operations_they_miss(km_odd):
    """"Broken is not near", for the geometry — now through the pipeline.

    A centre displaced by **1e-3**, asked (``symprec = 2e-3``) about the group
    it only **nearly** has — the layer cell's ``P-6m21' #187.210``, 24
    operations.  Carrying the centres onto that group's positions, the origin
    fitted to them, would move one by **5.0e-4** (the defect, read against that
    group written about the same fitted origin, which sees each centre's half
    of the displacement from both sides, reads 1.0e-3), far above the declared 1e-6:
    that is not a numerical defect in a symmetric set of centres, it is a
    different set.  So the operations that would move a centre that far are
    broken by the centres, in the verdict stage beside the hoppings' -- twenty of
    them -- and the group descends to what both carry: the centres are carried
    as they are by ``{E, m(z)}``, the hoppings break ``m(z)`` by tensor
    structure, and the group returned is ``P11'``, the group the default
    ``symprec`` finds outright, with the declared ``A1`` restricted to its
    trivial site group and the measurement on the record.

    ``symprec`` and ``position_tol`` are therefore uncoupled by construction:
    the first decides which group is asked about, the second what counts as
    noise once it is.  ``max_move`` is ``position_tol``: the centres' own pass,
    called directly, refuses a centre that far off rather than moving it.  And
    a slab thick enough that the padding it is read in follows its own spread
    along the normal, noisy along it as well, descends the same way: the group
    its cell allows is read against the noisy and the moved centres in that
    group's own lattice.

    Asked as a **free-standing** layer throughout, since wave R4c reads a
    ``dim_r = 2`` model on a substrate unless told otherwise and the
    24-operation cell these counts are measured against is the free-standing
    one.
    """
    bad = _move_one_centre(km_odd, 1e-3)

    # 1. asked about the group they only nearly have: the operations the centres
    #    miss by more than the declaration are broken by them, the group
    #    descends to what the centres and the hoppings both carry, and the
    #    declared `A1` comes back restricted to the site group the descent reached
    got = tqg.realized_group(bad, environment="free-standing", typat=[0, 1],
                             characters={0: "A1"},
                             symprec=2e-3, position_tol=1e-6, cross_check=False)
    assert got, str(got)
    assert got.group.number_str == "1.2"
    assert got.descent["from"]["number_str"] == "187.210"
    assert got.descent["steps"][0]["refuted_by"] == {"centres": 20, "tensor structure": 2}
    assert any("beyond the declared precision" in b["note"] for b in got.descent["dropped"]
               if b["refuted_by"] == "centres")
    assert ("so the cell was lowered to the largest group of its operations that carry the "
            "orbital centres within the declared precision and that the Hamiltonian obeys"
            ) in got.descent["note"], "the note says what dropped the twenty"
    pos = got.positions
    assert pos["adjusted"] is True and pos["position_tol"] == 1e-6
    assert pos["defect"] == pytest.approx(1e-3, rel=1e-6, abs=0)
    assert pos["move"] == pytest.approx(5e-4, rel=1e-6, abs=0), \
        "the move the whole group asks, the origin fitted: half the displacement"
    assert pos["carried"] == 4 and pos["max_move"] == 0.0 and pos["n_moved"] == 0
    assert "broken by the centres" in pos["note"]
    row = got.rep.characters.orbitals[0]
    assert (row.site_symmetry, row.irrep, row.declared_irrep, row.determined_by) == (
        "1", "-AA", "A1", "declaration")

    # 2. `max_move` is `position_tol`, so the direct call refuses above it --
    #    the existing gate, at the tolerance the pipeline hands it, on the move
    #    the pipeline measures: the origin fitted, half the displacement each
    clean3 = _layers._slab_presentation(km_odd)
    cell, cell_typat = _layers._decorated_cell(clean3, [0, 1])
    group = _spacegroup._group_of(cell, typat=cell_typat, spinor=True)
    asg = _tables._assign_orbitals(clean3, group, typat=[0, 1])
    refused = symmetrize_positions(_move_one_centre(clean3, 1e-3), group, asg,
                                   max_move=1e-6)
    assert not refused and refused.gate == "positions_not_nearly_symmetric"
    assert refused.value == pytest.approx(5e-4, rel=1e-6, abs=0)
    assert refused.threshold == 1e-6

    # 3. a slab whose vacuum is thinner than the padding the finder reads it in
    #    -- three layers 12 A thick in a 30 A cell -- is padded by its own spread
    #    along the normal, which noise on an outer site changes: the group the
    #    cell allows, named on the centres projected onto its positions, is read
    #    against the noisy and the moved centres in its own lattice, and the
    #    descent the centres make labels its bands
    noisy = _thick_slab([[0.0, 0.0, 1e-4], [0.0, 0.0, 0.0], [4e-3, 0.0, 0.0],
                         [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    thick = tqg.realized_group(noisy, precision=1e-3, cross_check=False)
    assert thick, str(thick)
    assert (thick.group.number_str, thick.descent["from"]["number_str"]) == ("25.58", "123.340")
    assert thick.descent["steps"][0]["refuted_by"] == {"centres": 24}
    assert thick.positions["frame"] == "model" and thick.positions["defect_after"] < 1e-14
    assert tqw.hsp_table(thick.rep, bands=2).counts, "the moved centres read in the group's frame"

    # 4. and the declaration itself is grammar: a non-positive one is a bug
    with pytest.raises(ValueError, match="position_tol must be"):
        tqg.realized_group(km_odd, environment="free-standing", typat=[0, 1],
                           position_tol=0.0)


# ---------------------------------------------------------------------------
# 9.  the operations themselves: a numerical cell's do not close
#     (wave P3b, second finding)
# ---------------------------------------------------------------------------

def _rigid_shift(model, delta, which=None):
    """The same model with ``delta`` added to one orbital's centre, or to all."""
    orb = np.asarray(model.orb_vecs, dtype=float).copy()
    orb[slice(None) if which is None else which, :len(delta)] += np.asarray(delta, float)
    dim_r = int(model.lattice.dim_r)
    out = TBModel(Lattice(np.asarray(model.lat_vecs, dtype=float), orb.tolist(),
                          periodic_dirs=list(model.periodic_dirs)),
                  spinful=(int(model.nspin) == 2))
    out.set_onsite(np.asarray(model.onsite), mode="set")
    for h in model.hoppings:
        out.set_hop(h["amplitude"], h["from_orbital"], h["to_orbital"],
                    list(h.get("lattice_vector", [0] * dim_r)), mode="add")
    return out


def _cell_group(model2d, typat=(0, 1)):
    """slab -> decorate -> ``_group_of``, the cell identification on its own."""
    m3 = _layers._slab_presentation(model2d)
    cell, cell_typat = _layers._decorated_cell(m3, list(typat))
    group = _spacegroup._group_of(cell, typat=cell_typat, spinor=True)
    assert group, str(group)
    return m3, group


def _tetragonal_pair(phase=0.0):
    """Two ``s`` orbitals on a tetragonal cell's fourfold axis, at ``(0, 0, ±0.2)``.

    The cell allows ``P4/mmm1' #123.340``, with the inversion centre between
    the two.  The hoppings along ``x`` and ``y`` differ, so the Hamiltonian
    obeys ``Pmmm1' #47.250``; a phase ``e^{iφ}`` on the ``x`` hopping breaks
    time reversal and every operation that reverses ``k_x`` as well, leaving
    ``Pmm2 #25``.
    """
    m = TBModel(Lattice(np.diag([1.0, 1.0, 1.6]), [[0.0, 0.0, 0.2], [0.0, 0.0, -0.2]],
                        periodic_dirs=[0, 1, 2]), spinful=False)
    m.set_onsite([0.0, 0.0])
    for o in (0, 1):
        m.set_hop(-1.0 * np.exp(1j * phase), o, o, [1, 0, 0])
        m.set_hop(-0.6, o, o, [0, 1, 0])
    m.set_hop(-0.8, 0, 1, [0, 0, 0])
    m.set_hop(-0.3, 0, 1, [0, 0, 1])
    return m


@pytest.mark.slow
def test_a_numerical_cells_operations_do_not_close_and_are_rebuilt(km_odd, realized):
    """**The operations spglib finds for a noisy cell are not a group**, and that
    is upstream of every defect a position pass could have.

    spglib reads each operation's translation off the atom pairs it maps, so
    positions carrying a Wannierization's noise give every operation its own
    share of it — *inconsistently*.  Displace one Kane-Mele centre by
    ``(+1e-8, −1e-8)`` and the identified ``P3m11'`` carries translations of 0,
    1e-8 and 2e-8 where the exact cell carries 0, closing only to **2.0e-8**
    against exactly ``0.0``.  With operations that do not close there is **no
    exactly symmetric configuration of positions** they describe, and
    ``bloch_operator`` *raises* on its integer-shift identity from the first
    ``hsp_table``; the centres themselves are moved onto the nearest set the
    operations carry exactly without reading those translations at all.

    :func:`..groups._exact_operations` repairs it where it is measurably broken
    and **nowhere else**: each translation is rebuilt from the tabulated
    reference-cell operation through the one origin shift (spglib's rotations
    and integer parts kept), and the closure goes to **0.0**.  That is the cell
    identified with nothing declared.  A representation built by hand on it at
    a declared tolerance is built, and the operations still miss its centres
    by 2e-8: its operators check the centres at their own 1e-9, as ``hsp_table``
    does, so no orbit's band representation is read, and each says why.
    Declared, the centres are made exact
    before the cell's setting is read -- projected onto the nearest set the
    operations found carry exactly, which needs neither tables nor repaired
    translations -- so the operations found on them close to rounding as they
    are, and nothing is rebuilt.

    **Every cell identified after the centres moved reads the moved centres.**
    A descent names its group on the parent cell's own points plus a generic
    orbit, and so does the unitary group's identification once time reversal is
    measured broken.  On a tetragonal pair with one centre off by 1e-5 along the
    axis, under ``precision=1e-4``, those points were once the centres as handed
    in, each identification read translations 1.0e-5 off the operations it was
    built to carry, and the first ``hsp_table`` raised on the integer-shift
    identity, in ``Pmmm1'`` and in ``Pmm2`` alike.  The cell is now built on the
    exact centres, the pair's own midpoint, 5e-6 off the model's origin, moved
    back onto it as noise: a second position pass finds nothing to move, and
    every record is the exact pair's.
    """
    delta, closure = (1e-8, -1e-8), 2e-8
    # the exact cell, on a substrate: the session's record of this call, whose
    # cross-check against the free-standing reading this test does not read
    clean = realized(km_z2_regime("odd"), typat=[0, 1], cross_check=False,
                     characters={0: "A1"})
    assert clean, str(clean)
    assert clean.group._closure_measured == 0.0, "an exact cell closes exactly"

    displaced = _rigid_shift(km_odd, delta, which=0)
    _m3, group = _cell_group(displaced)
    assert group.number_str == clean.group.number_str == "156.50"
    # the measurement, and the repair
    assert group._closure_measured == pytest.approx(closure, rel=1e-6, abs=0)
    assert group._closure_repaired == pytest.approx(closure, rel=1e-6, abs=0)
    # the repaired set closes at roundoff -- four decades below the 1e-12 that
    # `_exact_operations` itself raises above, so this is not that guard read
    # back.  Measured 0.5 ulp of a reduced coordinate (1.1e-16): the rebuilt
    # translations come through spglib's own origin shift, which the noisy
    # positions put off a dyadic value
    assert _spacegroup._translation_closure(group.symmetries) <= _FEW_ULP
    # the rotations were not touched: the same integer matrices as the exact cell
    rots = sorted(np.asarray(s.rotation, dtype=int).tobytes() for s in group.symmetries)
    assert rots == sorted(np.asarray(s.rotation, dtype=int).tobytes()
                          for s in clean.group.symmetries)
    # a representation built by hand on the displaced centres, at a declared
    # tolerance: built, and its orbits read no content, each saying why
    m3, cell_group, asg, val = _pipeline(displaced, [0, 1], tol=1e-6)
    by_hand = _build._build_rep(m3, cell_group, asg, validated=val, tol=1e-6,
                             characters={0: "A1"})
    assert by_hand, str(by_hand)
    orbits = by_hand.characters.band_representations
    assert orbits and all(orbit.terms == () for orbit in orbits)
    assert all("miss the operations by 2e-08 (reduced)" in orbit.note for orbit in orbits)
    # the line says the note whole, which says no sum is written and why
    assert all(orbit.note.startswith("no sum of elementary band representations, ")
               and orbit.sentence().endswith(f" at {orbit.wyckoff}: {orbit.note}")
               for orbit in orbits)

    # ... and now the whole pipeline reaches the answer
    out = tqg.realized_group(displaced, typat=[0, 1], characters={0: "A1"},
                             position_tol=1e-6)
    assert out, str(out)
    assert out.positions["adjusted"] is True
    # declared, the centres are made exact before the setting is read: the
    # operations found on them close to rounding, and nothing is rebuilt
    assert out.group._closure_measured <= _FEW_ULP
    assert out.group._closure_repaired <= _FEW_ULP
    assert out.positions["defect"] == pytest.approx(2e-8, rel=1e-6, abs=0)
    assert out.positions["defect_after"] < 1e-14, "one step, not a stalled fixed point"
    # the gauge identity, measured on eight random k: 1.8e-15
    assert out.positions["n_check"] == 8
    assert 0.0 < out.positions["spectrum_residual"] < 1e-13
    assert out.symmetrization is None
    assert max(out.validated.residuals[i] for i in out.validated.indices) < EXACT_GATE
    assert len(out.validated.indices) == out.group.size == 12
    assert out.rep._choice == clean.rep._choice
    assert max(float(np.abs(out.rep.matrices[i] - clean.rep.matrices[i]).max())
               for i in clean.rep.indices) == 0.0
    assert out.rep.characters.irreps == clean.rep.characters.irreps
    table = tqw.hsp_table(out.rep, bands=2)         # this is what used to RAISE
    assert table.complete and set(table.counts.values()) == {1}
    assert tqe.decompose(table, out.group, rep=out.rep,
                         bands=2).verdict == "symmetry-indicated decomposable"

    # The control that says the answer IS the clean crystal: the same rigid
    # displacement applied to EVERY centre is the clean model in a moved origin,
    # and the corep *names* at K (not a TRIM: an origin shift multiplies the C3
    # character by a cube root of unity) would follow that origin.  It is noise
    # within the declaration, so it is removed, for the control as for the
    # displaced model, and both come back as the clean crystal itself, in its
    # own frame.
    ctrl = tqg.realized_group(_rigid_shift(km_odd, delta), typat=[0, 1],
                              characters={0: "A1"}, position_tol=1e-6, cross_check=False)
    assert ctrl, str(ctrl)
    assert np.allclose(np.asarray(ctrl.group._shiftUC), np.asarray(out.group._shiftUC))
    assert ctrl.positions["max_move"] <= _FEW_ULP, "a rigid shift is no displacement"
    # one rigid shift between the two sets of centres, to a reduced coordinate out
    # of thirds arithmetic through a group average by two routes
    gap = np.asarray(out.model.orb_vecs) - np.asarray(ctrl.model.orb_vecs)
    assert float(np.abs(gap - gap[0]).max()) < _FEW_ULP, "the same crystal about another origin"
    ref = tqw.hsp_table(ctrl.rep, bands=2)
    assert table.counts == ref.counts and table.partitions == ref.partitions
    assert ({p: dict(v) for p, v in table.content.items()}
            == {p: dict(v) for p, v in ref.content.items()})
    assert (set(tqe.decompose(table, out.group, rep=out.rep, bands=2).solution_strings())
            == set(tqe.decompose(ref, ctrl.group, rep=ctrl.rep,
                                 bands=2).solution_strings()))

    # The identifications after the pass: a descent's (Pmmm1'), and with time
    # reversal broken by the phase, the unitary group's after it (Pmm2)
    for phase, number_str, n_broken in ((0.0, "47.250", 16), (0.4, "25", 24)):
        exact = tqg.realized_group(_tetragonal_pair(phase))
        noisy = tqg.realized_group(_rigid_shift(_tetragonal_pair(phase), (0.0, 0.0, 1e-5),
                                                which=0), precision=1e-4)
        assert exact and noisy, str(noisy)
        assert noisy.positions["adjusted"] is True
        # two centres are symmetric about their own midpoint: the origin fitted to
        # them is that midpoint, and neither moves; the midpoint sits 5e-6 off the
        # model's own origin, within the declaration, so both are shifted back
        # together and the pair is the exact one, its height grown by 5e-6
        assert noisy.positions["max_move"] <= _FEW_ULP
        assert noisy.positions["frame"] == "model"
        assert noisy.positions["offset"] == pytest.approx((0.0, 0.0, -5e-6),
                                                          rel=1e-6, abs=1e-15)
        assert noisy.group.number_str == exact.group.number_str == number_str
        assert noisy.descent["from"] == exact.descent["from"]
        assert noisy.descent["from"]["number_str"] == "123.340"
        assert len(noisy.descent["dropped"]) == len(exact.descent["dropped"]) == n_broken
        assert noisy.time_reversal["outcome"] == exact.time_reversal["outcome"]
        # the moved centres satisfy the operations returned: measured 4.4e-16
        _again, rec = symmetrize_positions(noisy.model, noisy.group, noisy.assignment)
        assert rec["defect_before"] <= _FEW_ULP, "nothing left for a second pass to move"
        table = tqw.hsp_table(noisy.rep, bands=1)       # this is what used to RAISE
        ref = tqw.hsp_table(exact.rep, bands=1)
        assert table.complete and table.counts == ref.counts
        assert max(float(np.abs(noisy.rep.matrices[i] - exact.rep.matrices[i]).max())
                   for i in exact.rep.indices) == 0.0
        assert (set(tqe.decompose(table, noisy.group, rep=noisy.rep, bands=1).solution_strings())
                == set(tqe.decompose(ref, exact.group, rep=exact.rep,
                                     bands=1).solution_strings()))


@pytest.mark.parametrize("name", ["km_odd", "km_inversion"])
def test_a_cell_built_from_exact_numbers_closes_exactly_and_is_untouched(name):
    """The bit-identity guard: the repair above must be **unreachable** here.

    Every model in this package's fixtures has positions that were *typed*, and
    where the operations that result carry only dyadic (or zero) translations
    spglib's translations close to ``0.0`` — exactly, not to rounding, on any
    IEEE machine, because dyadic arithmetic is exact whatever the order the
    summands arrive in — so :func:`..groups._exact_operations` returns
    immediately and no translation is rebuilt.  ``closure_repaired == 0.0`` is
    the statement that nothing was touched, bit for bit, which is what makes
    the rest of the suite the regression it was before this wave.

    A cell whose operations carry *thirds* is arithmetic rather than exact.
    ``_decorated_cell``'s honest recipe puts one substrate shadow species at
    ``−dz`` under **every** site (:func:`..groups._substrate_decorations`), so
    ``km_inversion``'s decorated honeycomb has atoms at the exact thirds
    ``(1/3, 1/3)`` and ``(2/3, 2/3)`` rather than only at the origin, and its
    two sites are one orbit — the cell is ``P6mm1' #183.186``, whose 24
    operations include the ones that exchange them.  Sums of thirds are not
    representable in binary floating point, so there the closure is a few ulp
    of a reduced coordinate rather than ``0.0`` — **1.1e-16** on both machines
    measured — read against ``groups._EXACT_CLOSURE = 1e-12``, the package's
    own scale for arithmetic, five decades below the 2.0e-8 a genuinely
    numerical cell reads.  The tables matched, so the repair runs and rebuilds
    each translation from the tabulated reference-cell operation, moving none
    of them past that same roundoff; the repaired set closes at a few ulp,
    which is what ``_translation_closure`` reads below.  Declaring the
    sublattices inequivalent removes the exchanging operations and the closure
    is a bit-exact ``0.0`` before the repair too — which is why ``km_odd``, the
    same honeycomb with ``typat=[0, 1]``, is in the exact column.

    A screw axis's thirds go through the same repair, read off tables that
    print a third to five decimals, which the repair snaps to the third it is:
    alpha-quartz's cell read without time reversal is ``P3_121 #152``.
    """
    cases = {
        "km_odd": (tqc_models.kane_mele_z2("odd"), dict(typat=[0, 1])),
        "km_inversion": (tqc_models.kane_mele_inversion(), {}),
    }
    #: Which cells close at a bit-exact ``0.0`` on every machine: those whose
    #: operations carry only dyadic (or zero) translations, in the cell and in
    #: irrep's reference setting.  The one matched to a hexagonal reference
    #: setting (#183) carries thirds and closes at a few ulp instead; that
    #: number is the machine's, not the cell's.
    exact = {"km_odd": True, "km_inversion": False}
    model, kw = cases[name]
    three_d = int(model.lattice.dim_k) == 3
    m3 = model if three_d else _layers._slab_presentation(model)
    labels = list(kw.get("typat", [0] * int(model.norb)))
    if three_d and "decorations" not in kw:
        cell, cell_typat = m3, labels
    else:
        cell, cell_typat = _layers._decorated_cell(m3, labels,
                                               decorations=kw.get("decorations"))
    group = _spacegroup._group_of(cell, typat=cell_typat,
                          spinor=int(model.nspin) == 2 or not three_d)
    assert group, str(group)
    assert group._closure_measured <= _tolerance._EXACT_CLOSURE, "arithmetic, not noise"
    if exact[name]:
        assert group._closure_measured == 0.0
        assert group._closure_repaired == 0.0, "nothing was touched"
        assert _spacegroup._translation_closure(group.symmetries) == 0.0
    else:
        assert group._closure_measured <= _FEW_ULP
        assert group._closure_repaired <= _FEW_ULP, "the repair moved nothing past roundoff"
        assert _spacegroup._translation_closure(group.symmetries) <= _FEW_ULP
        # alpha-quartz: Si at 3a and O at 6c of P3_121, its screw axis's thirds
        # rebuilt from the tables' own translations
        positions, _centering = _spacegroup._pyxtal_wyckoff(152)
        sites, species = [], []
        for kind, (letter, x) in enumerate((("a", (0.4697, 0.0, 0.0)),
                                            ("c", (0.4135, 0.2669, 0.1191)))):
            for A, b, _P in next(p["arms"] for p in positions if p["letter"] == letter):
                site = (A @ np.asarray(x) + b) % 1.0
                if not any(np.abs((site - q + 0.5) % 1.0 - 0.5).max() < 1e-7 for q in sites):
                    sites.append(site)
                    species.append(kind)
        quartz = TBModel(Lattice([[4.913, 0.0, 0.0], [-2.4565, 4.913 * np.sqrt(3) / 2, 0.0],
                                  [0.0, 0.0, 5.405]], [list(s) for s in sites],
                                 periodic_dirs=[0, 1, 2]), spinful=False)
        assert _spacegroup._group_of(quartz, typat=species, spinor=False,
                             include_TR=False).number_str == "152"


# ---------------------------------------------------------------------------
# 10.  one declaration for a numerical model: ``realized_group(precision=)``
#      (the two tolerances above are one fact about the model, in two units)
# ---------------------------------------------------------------------------

def _spin_frame_turned(model, theta):
    """A spinful ``model`` with every spin block written in a frame turned by ``theta`` about ``x``."""
    U = np.array([[np.cos(theta / 2), -1j * np.sin(theta / 2)],
                  [-1j * np.sin(theta / 2), np.cos(theta / 2)]])
    out = TBModel(Lattice(np.asarray(model.lat_vecs, dtype=float),
                          np.asarray(model.orb_vecs, dtype=float).tolist(),
                          periodic_dirs=list(model.periodic_dirs)), spinful=True)
    out.set_onsite([U @ np.asarray(o, dtype=complex).reshape(2, 2) @ U.conj().T
                    for o in np.asarray(model.onsite, dtype=complex)], mode="set")
    for h in model.hoppings:
        amp = np.asarray(h["amplitude"], dtype=complex)
        out.set_hop(U @ amp @ U.conj().T if amp.shape == (2, 2) else amp,
                    int(h["from_orbital"]), int(h["to_orbital"]),
                    list(h.get("lattice_vector", [0, 0, 0])), mode="add")
    return out


@pytest.mark.slow
def test_one_declaration_gives_the_explicit_tolerances_answer(lin3g_clean,
                                                              lin3g_jittered):
    """``precision=1e-6`` on the P3 golden: the same run, from one number.

    The two declarations this module is about are one fact — the model's own
    numerical precision — wearing two units: the centres are in **reduced**
    coordinates, where a relative precision is already absolute, and the
    covariance gate is **absolute in the units of H**, where it is not.  A
    wannier90 user should not have to know which is which, so ``precision``
    measures ``max|H|`` here (17.42 on this golden, its largest entry about its
    mean on-site energy of -6.19, and the model's own number, not a typed one)
    and derives ``tol = 10 x precision x max|H|`` and ``position_tol =
    precision`` from it.  A constant shift of ``H`` moves no covariance
    residual, so it moves none of these: the golden with every on-site energy
    raised by 8.25 reads the same record to rounding, and the same group and
    labels.  The detector's two **relative**
    numbers are read against the same ``max|H|`` — the nullspace cut is
    ``tol / max|H| = 10 x precision`` and the matching window is
    ``clip(10 x sqrt(cut), 1e-3, 0.3)`` — and the record carries them too, so a
    derived run reads back whole.

    The answer is the one the explicit knobs give, and the record says exactly
    what was declared on the caller's behalf.

    The same declaration is the noise the multiplets are measured to: whether a
    solution-space member is in block form, and whether a block is empty or
    singular, are read at ``tol / max|H| = 10 x precision`` where that is
    larger than the exact defaults.  A spin-orbit ``p`` shell whose orbital
    frame is turned by 1e-8 inside its ``3m`` site -- a Wannier frame off the
    ideal one by less than its own precision -- is one multiplet at the
    defaults, and at ``precision=1e-6`` its ``p_z`` and ``(p_x, p_y)`` pair are
    measured apart.  A declaration states the site's content and is read on
    the multiplets measured, so the shell declared orbital by orbital, as
    ``p_z`` beside the pair, as the shell ``p`` or as ``A1 + E`` is the same
    crystal in both: ``R3m1'``, every row the declaration's, each its own
    multiplet's site irrep, the shell's word in the ``function`` column.
    ``2x-E1`` written on two of its orbitals states
    corep content, and it is part of the shell's ``2x-E1 + -1E2E`` although
    no orbital content lifting to it is part of ``A1 + E``: ``R3m1'`` at either
    precision, those two rows the declaration's.  The detector's
    own members turn with the frame and are exact; the matrices the measured
    multiplets anchor do not, and the Hamiltonian's own symmetry matrices mix
    those multiplets at the frame's own 2e-8, so every operation but the
    identity and time reversal reads near on them, and the multiplets hold
    only within the declared precision, which the multiplet measurement says.
    That licenses the projection: the model's own blocks on each multiplet made
    an exact representation, ``H`` moved by about the turn, and every call returns
    ``R3m1'`` with its covariance and product table at machine precision, the
    rows the characters name.  With ``symmetrize=False`` the exact gate
    refuses as before, naming the route.  The same shell at both sites of
    ``6c`` of ``R-3m`` -- a bismuth-like geometry -- is one multiplet per site
    without the precision, and ``R-3m1'``, 24 of 24, with its ``p_z`` and pair
    apart under it; declared or not, at either precision, every subset of its
    bands reads the same labels.  Written in the coreps its rows are returned
    as, ``2x-E1 + -1E2E``, the shell states the corep content alone, which
    ``A1 + E`` and ``A2 + E`` share, so the four labellings of the undeclared
    crystal stay open where the orbital spellings leave two.  Its spin blocks
    written in a frame turned by 5e-9 -- a spinor import's own numerical
    defect -- and read through ``spin=`` with no precision, the pair is
    ``R-3m1'``, 24 of 24: each operation's own form, its own spin rotation,
    carries ``H(k)`` onto ``H(gk)`` within the covariance gate, and ``H`` is
    moved onto the group by less than the turn.  Turned by 1.5e-8, past what
    the gate allows, the sixteen operations whose lift the turn changes are
    broken by tensor structure, and the pair is ``C2/m1'``, the group it has in
    pythtb's frame, the note naming the precision that holds them, 2.7e-9; at
    ``precision=1e-6`` a turn of 1e-6 is within it and one of 1e-5 is not, the
    note naming 1.8e-6, at which the pair is ``R-3m1'`` again.
    """
    expl, model, _tol = lin3g_jittered
    out = tqg.realized_group(model, typat=[0, 0, 1, 1], precision=1e-6,
                             characters=["s", "dx2-y2", "s", "s"])
    assert out, str(out)

    # 1. what the one number turned into -- measured, and on the record
    rec = out.precision
    assert set(rec) == {"precision", "max_h", "tol", "position_tol", "cut",
                        "window", "symprec", "symprec_raised"}
    assert rec["max_h"] == pytest.approx(17.41932, rel=1e-6, abs=0)
    assert rec["tol"] == pytest.approx(1.741932e-4, rel=1e-6, abs=0)
    assert rec["cut"] == pytest.approx(rec["tol"] / rec["max_h"], rel=1e-12, abs=0), \
        "the cut is the gate read against the Hamiltonian's own magnitude"
    assert rec["cut"] == pytest.approx(1e-5, rel=1e-12, abs=0)
    assert rec["window"] == pytest.approx(0.031623, rel=1e-4, abs=0)
    assert rec["window"] == pytest.approx(10.0 * np.sqrt(rec["cut"]), rel=1e-12,
                                          abs=0), \
        "the window is the width the cut asks for, and 0.0316 is inside 1e-3 .. 0.3"
    # ... about the mean on-site energy, which a constant shift moves and no
    # residual sees
    lifted = tqg.realized_group(_energy_shifted(model, 8.25), typat=[0, 0, 1, 1],
                                precision=1e-6, characters=["s", "dx2-y2", "s", "s"],
                                cross_check=False)
    assert lifted, str(lifted)
    for key in ("max_h", "tol", "cut", "window"):
        assert lifted.precision[key] == pytest.approx(rec[key], rel=1e-12, abs=0), key
    assert lifted.group.number_str == "99.164"
    assert lifted.rep.characters.irreps == out.rep.characters.irreps
    assert out.symmetrization.tol == rec["tol"], "the detector ran at the derived tol"
    assert out.positions["position_tol"] == 1e-6, "and the centres at the other one"
    assert (rec["symprec"], rec["symprec_raised"]) == (1e-5, False), \
        "1e-6 of noise on a lattice constant of 1 does not defeat a finder at 1e-5"
    assert "precision 1e-06 declared" in repr(out)

    # 2. the physics is the explicit call's -- and the clean model's, which the
    #    test above proves that call reproduces
    assert out.symmetrization is not None and len(out.symmetrization.near) == 15
    assert out.symmetrization.defect == pytest.approx(9.49e-7, rel=1e-2, abs=0)
    bands = (0, 1, 2, 3, 4, 5)
    b = tqw.hsp_table(out.rep, bands=bands)
    db = tqe.decompose(b, out.group, rep=out.rep, bands=6)
    for ref in (expl, lin3g_clean):
        assert out.group.number_str == ref.group.number_str == "99.164"
        assert out.rep._choice == ref.rep._choice
        assert max(float(np.abs(out.rep.matrices[i] - ref.rep.matrices[i]).max())
                   for i in ref.rep.indices) == 0.0
        assert out.rep.characters.irreps == ref.rep.characters.irreps
        a = tqw.hsp_table(ref.rep, bands=bands)
        assert a.complete and b.complete
        assert a.counts == b.counts == {"-GM6": 2, "-GM7": 1, "-M6": 1, "-M7": 2,
                                        "-X5": 3}
        assert a.partitions == b.partitions
        assert ({p: dict(v) for p, v in a.content.items()}
                == {p: dict(v) for p, v in b.content.items()})
        da = tqe.decompose(a, ref.group, rep=ref.rep, bands=6)
        assert da.verdict == db.verdict == "symmetry-indicated decomposable"
        assert set(da.solution_strings()) == set(db.solution_strings())

    # 3. the multiplets' block tests read the same declaration, as tol / max|H|:
    #    a spin-orbit p shell at the 3m site of R3m, its orbital frame turned by
    #    1e-8 inside the site and the Hamiltonian exactly symmetric in that frame
    shell = tqc_models.r3m_p_shell(angle=1e-8, spinful=True)
    kw = {"decorations": tqc_models.R3M_SHADOW}
    per_orbital, joint = ["pz", "px", "py"], {0: "pz", (1, 2): "E"}
    whole = ({(0, 1, 2): "p"}, {(0, 1, 2): "A1 + E"})
    exact = tqg.realized_group(shell, **kw)
    assert exact and exact.rep.multiplets.by_site == (((0, 1, 2),),), \
        "at the exact defaults the turned frame binds the shell into one multiplet"
    content = tqw.hsp_table(exact.rep, bands=6).content
    for characters in (per_orbital, joint) + whole:
        # the content declared on the one measured multiplet is what it holds,
        # however the orbitals were partitioned: read on it, nothing descends
        said = tqg.realized_group(shell, characters=characters, **kw)
        assert said, f"{characters}: {said}"
        assert (said.group.number_str, len(said.validated.indices)) == ("160.66", 12)
        assert said.descent is None and said.rep.multiplets.by_site == (((0, 1, 2),),)
        assert said.rep.characters.irreps == ("2x-E1 + -1E2E",) * 3
        assert {o.determined_by for o in said.rep.characters.orbitals} == {"declaration"}
        if characters in whole:
            assert [o.function for o in said.rep.characters.orbitals] == \
                (["p"] if "p" in characters.values() else [""]) * 3, \
                "the shell's word is a function, a sum of irreps is not"
    for precision in (None, 1e-6):
        # a corep word on part of the shell states corep content: 2x-E1 is part of
        # the shell's 2x-E1 + -1E2E, though no orbital content lifting to it is
        # part of A1 + E, and which of the site's multiplets hold it is the basis's
        part = tqg.realized_group(shell, precision=precision, characters={(1, 2): "2x-E1"},
                                  **kw)
        assert part, f"{precision}: {part}"
        assert (part.group.number_str, len(part.validated.indices)) == ("160.66", 12)
        assert part.descent is None
        rows = part.rep.characters.orbitals
        assert [o.determined_by for o in rows[1:]] == ["declaration"] * 2
        assert [o.declared_irrep for o in rows[1:]] == ["2x-E1"] * 2, \
            "the corep content declared, recorded as written"
        assert [o.irrep for o in rows[1:]] == (
            ["2x-E1 + -1E2E"] if precision is None else ["-1E2E + -E1"]) * 2, \
            "each row its own multiplet's: the whole shell at the defaults, the pair apart"
        assert rows[0].determined_by != "declaration" and rows[0].function == ""
    for characters in whole:
        # ... and a shell over the p_z and the pair measured apart is theirs together
        noisy = tqg.realized_group(shell, precision=1e-6, characters=characters, **kw)
        assert noisy, f"{characters}: {noisy}"
        assert (noisy.group.number_str, len(noisy.validated.indices)) == ("160.66", 12)
        assert noisy.rep.multiplets.by_site == (((0,), (1, 2)),)
        assert noisy.rep.characters.irreps == ("-E1", "-1E2E + -E1", "-1E2E + -E1"), \
            "each row its own multiplet's, as the shell declared orbital by orbital reads"
        assert {o.determined_by for o in noisy.rep.characters.orbitals} == {"declaration"}
        assert ({p: dict(v) for p, v in tqw.hsp_table(noisy.rep, bands=6).content.items()}
                == {p: dict(v) for p, v in content.items()})
    for characters in (None, per_orbital, joint):
        noisy = tqg.realized_group(shell, precision=1e-6, characters=characters, **kw)
        # the frame's own 1e-8 is inside the declared precision: the p_z and the
        # (p_x, p_y) pair are measured apart, the declarations are read as
        # written, and nothing descends
        assert noisy, str(noisy)
        assert (noisy.group.number_str, len(noisy.validated.indices)) == ("160.66", 12)
        assert noisy.descent is None and noisy.rep.multiplets.by_site == (((0,), (1, 2)),)
        # ... every operation but the identity and time reversal near, exact to
        # the detector and missing H by the turn on the matrices the measured
        # multiplets anchor, and the multiplets holding only within the declared
        # precision: that is what projected, and by about the turn
        rec = noisy.symmetrization
        assert len(rec.near) == 10 and len(rec.exact) == 2
        assert rec.multiplets["exact"] < rec.multiplets["mixing"] < rec.multiplets["declared"]
        assert 1e-9 < rec.defect < 1e-7
        assert "hold only within the declared precision" in rec.summary()
        assert "no-op" not in rec.summary()
        assert max(noisy.rep.residuals.values()) < 1e-13
        assert noisy.rep.product_residual < 1e-13
        assert noisy.rep.characters.irreps == ("-E1", "-1E2E + -E1", "-1E2E + -E1")
    # without the projection the exact gate refuses, inside the declared tol
    kept = tqg.realized_group(shell, precision=1e-6, symmetrize=False, **kw)
    assert kept.gate == "rep_not_covariant" and 1e-9 < kept.value < 1e-6, str(kept)
    assert kept.context["group"].number_str == "160.66"
    assert 'symmetrize="auto"' in kept.reason
    # ... and the same shell at both sites of 6c of R-3m
    pair = tqc_models.r3m_p_shell(angle=1e-8, spinful=True, pair=True)
    bare = tqg.realized_group(pair)
    assert bare.rep.multiplets.by_site == (((0, 1, 2),), ((3, 4, 5),))
    both = tqg.realized_group(pair, precision=1e-6)
    assert both, str(both)
    assert (both.group.number_str, len(both.validated.indices)) == ("166.98", 24)
    assert both.rep.multiplets.by_site == (((0,), (1, 2)), ((3,), (4, 5)))
    assert len(both.symmetrization.near) == 20 and both.symmetrization.multiplets
    # declared orbital by orbital or as the shell, at either precision: the same
    # crystal, and every subset of the bands reads the undeclared crystal's labels
    labels = {n: {p: dict(v) for p, v in tqw.hsp_table(bare.rep, bands=n).content.items()}
              for n in (2, 4, 6)}
    # the labellings that keep every site corep and label some band otherwise,
    # each by what set it aside
    def set_aside(out):
        return [key for choice in out.rep.characters._labellings["choices"]
                for key, _labels in choice["others"]]
    assert set_aside(bare) == ["content", "orbitals"]
    coreps = {(0, 1, 2): "2x-E1 + -1E2E", (3, 4, 5): "2x-E1 + -1E2E"}
    for precision in (None, 1e-6):
        for characters in (["pz", "px", "py"] * 2, {(0, 1, 2): "p", (3, 4, 5): "p"}, coreps):
            said = tqg.realized_group(pair, precision=precision, characters=characters)
            assert said, f"{precision}, {characters}: {said}"
            assert (said.group.number_str, len(said.validated.indices)) == ("166.98", 24)
            assert {o.determined_by for o in said.rep.characters.orbitals} == {"declaration"}
            assert set_aside(said) == (["content", "orbitals"] if characters is coreps
                                       else ["orbitals"]), "a declared p shell pins its content"
            for n, want in labels.items():
                got = tqw.hsp_table(said.rep, bands=n).content
                assert {p: dict(v) for p, v in got.items()} == want, (precision, characters, n)
    # ... and the pair as a spinor import whose spin frame is turned, read through
    # spin=: each operation's own form carries H within the covariance gate the
    # call reads at, or the operations whose lift the turn changes are broken,
    # the note naming the precision that would hold them
    for theta, precision, want, named in ((5e-9, None, ("166.98", 24), None),
                                          (1.5e-8, None, ("12.59", 8), "2.7e-09"),
                                          (1e-6, 1e-6, ("166.98", 24), None),
                                          (1e-5, 1e-6, ("12.59", 8), "1.8e-06")):
        flat, sb = tqspin.spinless_presentation(
            _spin_frame_turned(tqc_models.r3m_p_shell(spinful=True, pair=True), theta),
            order="interleaved")
        imported = tqg.realized_group(flat, spin=sb, precision=precision)
        assert imported, str(imported)
        assert (imported.group.number_str, len(imported.validated.indices)) == want
        if named is None:
            # the turn within the gate is the model's own noise: H moved onto the
            # group by less than the turn
            assert imported.descent is None and imported.symmetrization.defect < theta
            continue
        broken = imported.descent["dropped"]
        assert len(broken) == 16 and {b["refuted_by"] for b in broken} == {"tensor structure"}
        assert min(b["covariance"] for b in broken) > imported.validated.tol
        assert f"a precision of {named} or coarser" in imported.descent["note"]
    # ... and the precision named holds them
    held = tqg.realized_group(flat, spin=sb, precision=float(named))
    assert (held.group.number_str, len(held.validated.indices)) == ("166.98", 24)
    assert held.symmetrization.defect < theta


def test_the_declaration_is_grammar(km_odd):
    """Declaring the same thing twice, or declaring a precision that is not one.

    ``precision`` **is** the declaration ``tol``, ``cut``, ``window`` and
    ``position_tol`` are derived from, so giving it beside any of them says the
    same thing twice and the
    call cannot pick a winner: that is a mistake in the call, not a verdict about
    a model, and it raises.  The detector's two relative numbers are in the list
    for the same reason the gate is — they follow from the declaration, read
    against the same ``max|H|`` — so declaring one of them is declaring the
    precision a second time, in a second unit.  ``symprec`` is a different
    question (which group is *found*) and may be given alongside.
    """
    for kw in (dict(tol=1e-6), dict(position_tol=1e-6), dict(cut=1e-6),
               dict(window=0.05), dict(cut=1e-6, window=0.05),
               dict(tol=1e-6, position_tol=1e-6)):
        with pytest.raises(ValueError, match="same thing twice") as e:
            tqg.realized_group(km_odd, typat=[0, 1], precision=1e-6, **kw)
        assert f"{' and '.join(kw)} declare the same thing twice" in str(e.value), \
            "the message names exactly what was doubled"
    for bad in (0.0, -1e-6, float("nan"), float("inf"), "six decimals"):
        with pytest.raises(ValueError, match="precision must be"):
            tqg.realized_group(km_odd, typat=[0, 1], precision=bad)


@pytest.mark.slow
def test_the_finders_tolerance_is_raised_only_when_the_declaration_would_defeat_it(
        km_odd, lin3g_clean):
    """The ``symprec`` rule — and why it is not the coupling P3b ruled out.

    ``position_tol`` and ``symprec`` answer different questions and are not
    computed from each other anywhere.  What *can* move both is one declaration
    about the **model**: a reduced position noise of ``p`` is a Cartesian
    ``p·L``, and a finder run below its own noise cannot report the group that
    is there — and a group the finder never found cannot be snapped onto.  So
    where ``p·L > symprec`` the finder is run at ``10·p·L`` and the record says
    so.

    ``L`` is measured over the model's **periodic** directions only — Kane-Mele's
    in-plane lattice constant, **1.0** — and never over the padded one, whose
    vector is 30.0 but along which a 2D model has no coordinate at all (the slab
    presentation invents it and every position there is exactly 0.0), so a
    declared position noise cannot apply to it.  At ``p = 1e-6``, wannier90's own
    precision, the rule therefore does **not** fire; at ``p = 1e-4`` it does, and
    the finder is asked at 1e-3 — a hundred times its default, a thousandth of
    the lattice constant — and the clean Kane-Mele cell is still ``P3m11'``, all
    twelve operations obeyed at 1.7e-14, with representation matrices identical
    to **0.0** against the same model at 1e-6 where the finder was not moved.

    Only ever upward: a caller who set ``symprec`` deliberately keeps it as the
    floor.  And on a model built from exact numbers the declaration is a
    measured no-op: nothing moves, and nothing is projected.
    """
    # L, measured: the periodic vectors are the model's own, the padded one is not
    slab = _layers._slab_presentation(km_odd)
    lat = np.asarray(slab.lat_vecs, dtype=float)
    per = [int(d) for d in slab.periodic_dirs]
    assert per == [0, 1] and float(np.linalg.norm(lat[2])) == 30.0
    assert float(np.linalg.norm(lat[per], axis=1).max()) == 1.0, "L, measured"
    assert np.all(np.asarray(slab.orb_vecs)[:, 2] == 0.0), \
        "and the padded coordinate carries no noise for a declaration to be about"

    # 1e-6 -- wannier90's own -- does not defeat a finder at 1e-5, so it stands
    assert lin3g_clean.precision is None, "no declaration, no record"
    at_w90 = tqg.realized_group(km_odd, typat=[0, 1], characters={0: "A1"},
                                precision=1e-6, cross_check=False)
    assert at_w90, str(at_w90)
    assert at_w90.precision["symprec_raised"] is False
    assert at_w90.precision["symprec"] == 1e-5
    assert "symprec raised" not in repr(at_w90)
    # ... and on this exact model the declaration changes nothing: the centres'
    # own defect is the thirds arithmetic of the honeycomb (measured 1.1e-16)
    # and the move with it (2.2e-16, one ulp of a reduced coordinate)
    assert at_w90.symmetrization is None, "nothing was near, so nothing was projected"
    assert at_w90.positions["defect"] <= _FEW_ULP
    assert at_w90.positions["max_move"] <= _FEW_ULP

    # 1e-4 does, and the group is still the one the positions carry
    out = tqg.realized_group(km_odd, typat=[0, 1], characters={0: "A1"},
                             precision=1e-4, cross_check=False)
    assert out, str(out)
    assert out.precision["symprec_raised"] is True
    assert out.precision["symprec"] == pytest.approx(1e-3, rel=1e-12, abs=0)
    assert out.group.number_str == "156.50" and len(out.validated.indices) == 12
    assert max(out.validated.residuals[i] for i in out.validated.indices) < EXACT_GATE
    assert out.symmetrization is None and out.positions["max_move"] < 1e-14
    assert "symprec raised to 0.001" in repr(out)
    assert out.rep._choice == at_w90.rep._choice
    assert max(float(np.abs(out.rep.matrices[i] - at_w90.rep.matrices[i]).max())
               for i in at_w90.rep.indices) == 0.0, "raising the finder moved nothing"

    # a symprec the caller declared above the noise is the floor, and stands
    floored = tqg.realized_group(km_odd, typat=[0, 1], characters={0: "A1"},
                                 precision=1e-4, symprec=1e-2, cross_check=False)
    assert floored, str(floored)
    assert floored.precision["symprec_raised"] is False
    assert floored.precision["symprec"] == 1e-2
    assert floored.group.number_str == out.group.number_str


@pytest.mark.slow
def test_a_defect_above_the_declaration_descends_as_the_explicit_knobs_do(km_odd,
                                                                         monkeypatch):
    """One centre displaced by **1e-3**, asked about the group it only nearly has.

    The declaration is not a licence: centres beyond it are a crystal of their
    own, and the operations of the substrate cell's ``P3m11' #156.50`` they are
    not carried by within it -- all but the identity, here -- are broken by the
    centres, so the cell descends to ``P11'``, with the measured defect on the
    record and the declared ``A1`` restricted to the trivial site group reached.
    The run is the explicit-knob run, group and rows alike, which is the
    statement that ``precision`` is a convenience and not a second mechanism.

    The descent's ``P11'`` cell is where ``irrep``'s search for the tables'
    setting can miss: its spin-basis fit starts from a random point, and from
    a thousand global seeds it fails on this cell six times.  So the search
    runs from a fixed seed, with the random state the call arrived with left as
    it was, and runs again from the next fixed seed where its fit fails: a
    search that fails at its first seed on every cell still reads the tables,
    and the rows are the ones above.  Where every seed fails the group comes back
    without tables and says so -- its note carries ``irrep``'s own message, and
    every row names the tables as unavailable, and why.
    """
    bad = _move_one_centre(km_odd, 1e-3)
    got = tqg.realized_group(bad, typat=[0, 1], characters={0: "A1"},
                             symprec=2e-3, precision=1e-6, cross_check=False)
    expl = tqg.realized_group(bad, typat=[0, 1], characters={0: "A1"},
                              symprec=2e-3, position_tol=1e-6, cross_check=False)
    assert got and expl, (str(got), str(expl))
    assert got.group.number_str == expl.group.number_str == "1.2"
    assert got.descent["from"]["number_str"] == expl.descent["from"]["number_str"] \
        == "156.50"
    assert got.positions == expl.positions
    assert got.descent["dropped_by"] == {"centres": 10}
    assert got.positions["carried"] == 2 and got.positions["max_move"] == 0.0
    # against the cell's group written about the origin fitted to the centres,
    # each of which then sits half the displacement off
    assert got.positions["defect"] == pytest.approx(1e-3, rel=1e-6, abs=0)
    assert got.rep.characters.irreps == expl.rep.characters.irreps == ("-AA", "-AA")
    row = got.rep.characters.orbitals[0]
    assert (row.irrep, row.declared_irrep, row.determined_by) == (
        "-AA", "A1", "declaration")

    # the tables' setting, searched from fixed seeds: every search whose spin
    # fit fails from the first one is asked again, and the tables are read.
    # The fit draws its start from numpy's global state before it can fail,
    # and so do these stand-ins for it.
    from irrep.spacegroup_irreps import SpaceGroupIrreps

    real = SpaceGroupIrreps.from_cell.__func__
    first = np.random.RandomState(_spacegroup._SETTING_SEEDS[0]).get_state()[1][:4]
    unlucky = []

    def _first_seed_fails(cls, *args, **kwargs):
        if kwargs.get("search_cell"):
            unlucky.append(bool(np.array_equal(np.random.get_state()[1][:4], first)))
            if unlucky[-1]:
                np.random.random(8)
                raise RuntimeError("the accurcy is only 0.5. Is this good?")
        return real(cls, *args, **kwargs)

    def _every_seed_fails(cls, *args, **kwargs):
        if kwargs.get("search_cell"):
            np.random.random(8)
            raise RuntimeError("the accurcy is only 0.5. Is this good?")
        return real(cls, *args, **kwargs)

    try:
        monkeypatch.setattr(SpaceGroupIrreps, "from_cell", classmethod(_first_seed_fails))
        _spacegroup._identification_memo_clear()
        np.random.seed(20260927)
        state = np.random.get_state()
        again = tqg.realized_group(bad, typat=[0, 1], characters={0: "A1"},
                                   symprec=2e-3, position_tol=1e-6)
        after = np.random.get_state()
        assert again and again.group.number_str == "1.2" and again.group._tables_ready
        assert again.rep.characters.irreps == ("-AA", "-AA")
        assert unlucky.count(True) == unlucky.count(False) >= 2, \
            "each search failed at the first seed and read the tables at the next"
        assert np.array_equal(after[1], state[1]) and after[2:] == state[2:], \
            "and the random state the call arrived with is the one it leaves"

        # ... and where every seed fails, the group without tables, said so
        monkeypatch.setattr(SpaceGroupIrreps, "from_cell", classmethod(_every_seed_fails))
        _spacegroup._identification_memo_clear()
        blank = tqg.realized_group(km_odd, typat=[0, 1], cross_check=False)
        assert blank and blank.group.number_str == "156.50", str(blank)
        assert not blank.group._tables_ready
        assert "the accurcy is only 0.5" in blank.group._setup_note
        assert f"in {len(_spacegroup._SETTING_SEEDS)} searches" in blank.group._setup_note
        for o in blank.rep.characters.orbitals:
            assert (o.irrep, o.named_by) == ("", "unnamed")
            assert "unavailable" in o.note and "the accurcy is only 0.5" in o.note
    finally:
        monkeypatch.undo()
        _spacegroup._identification_memo_clear()
