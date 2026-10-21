r"""Regressions for ``tqc.groups`` (phase A1) and the ``tqc.symmetry`` detector.

Self-contained by design: every model here is built inline, or taken from
``pythtqc.models``, so these tests exercise the adapter and nothing else.  The
golden fixtures in ``fixture_models.py`` are wired to this machinery elsewhere.

The anchors are deliberately *literature-checkable* and, where possible,
cross-checked against ``irreptables``' own EBR data rather than against
numbers typed here:

* graphene's :math:`p_z` doublet at K is the two-dimensional ``K6`` of the
  slab group ``P6/mmm`` (``K5`` for the σ_h-even, s-like orbital rep the bare
  model actually realizes);
* the s-orbital band representation of the simple-cubic lattice reads
  ``GM1+ X1+ M1+ R1+`` and decomposes as exactly ``1 × [A1g↑G(1) @ 1a]``;
* diamond's two s orbitals read ``GM1+⊕GM2−``, ``X1``, ``L1+⊕L2−`` — precisely
  the content ``irreptables`` lists for ``A1↑G(2) @ 8a`` of Fd-3m.

No space group is privileged in the *library*; those facts live only here.
"""

import copy
import dataclasses
import sys
import warnings

import numpy as np
import pytest

from conftest import patch_everywhere
from pythtb import Lattice, TBModel
from pythtb.models import graphene

from pythtqc import groups as G
from pythtqc import InternalError
from pythtqc.models import i213_flicker2018
from pythtqc.refusal import MagneticScopeWarning, Refusal, RefusalError
from pythtqc import _build
from pythtqc import _detector
from pythtqc import _layers
from pythtqc import _matrices
from pythtqc import _measure
from pythtqc import _spacegroup
from pythtqc import _tables

# Third-party deprecations (spglib, irrep) are noise here; a deprecated pythtqc
# spelling is not, and the second mark keeps the suite's own guard in force
# (marks apply in order, and the last matching filter wins).
pytestmark = [pytest.mark.filterwarnings("ignore::DeprecationWarning"),
              pytest.mark.filterwarnings("error:pythtqc:DeprecationWarning")]

_ULP = np.finfo(float).eps          # one unit of double-precision roundoff, 2.2e-16
_ROUNDOFF = 16 * _ULP               # a few ulp at |h| ~ 3: the reassociation of a four-term sum


# ---------------------------------------------------------------------------
# inline models
# ---------------------------------------------------------------------------

def square(tx=-1.0, ty=-1.0, spinful=False, td=0.0):
    """Square lattice, one orbital at the origin.  ``tx != ty`` breaks C4.

    ``td`` adds the two diagonal hoppings.  They break nothing -- the model
    keeps every operation the isotropic one has -- but they give ``h(k)`` four
    summands where it had two, so a rotated k reassociates the sum instead of
    merely commuting it and the constraint operator picks up rounding noise
    rather than coming out bit-exactly zero.  See
    :func:`test_the_isotropic_square_passes_by_the_vacuity_test_not_by_luck`.
    """
    lat = Lattice([[1.0, 0.0], [0.0, 1.0]], [[0.0, 0.0]], periodic_dirs=[0, 1])
    m = TBModel(lattice=lat, spinful=spinful)
    one = np.array([1.0, 0, 0, 0]) if spinful else 1.0
    m.set_onsite([0.0])
    m.set_hop(tx * one, 0, 0, [1, 0])
    m.set_hop(ty * one, 0, 0, [0, 1])
    if td:
        m.set_hop(td * one, 0, 0, [1, 1])
        m.set_hop(td * one, 0, 0, [1, -1])
    return m


def square_two_orbitals(shift=0.0):
    """Square lattice with orbitals at 1a = (0,0) and 1c = (1/2,1/2).

    ``shift`` displaces the second orbital off its Wyckoff orbit.
    """
    lat = Lattice([[1.0, 0.0], [0.0, 1.0]],
                  [[0.0, 0.0], [0.5, 0.5 + shift]], periodic_dirs=[0, 1])
    m = TBModel(lattice=lat)
    m.set_onsite([0.3, -0.3])
    m.set_hop(-1.0, 0, 1, [0, 0])
    return m


def cubic(spinful=False, tz=-1.0):
    """Simple cubic lattice, one s orbital -> Pm-3m.

    ``spinful`` doubles the state count with **spin-independent** hoppings, so
    ``H(k)`` stays the scalar matrix ``h(k) * 1_2``; ``tz != -1`` breaks the
    cubic axis down to tetragonal without touching either property.
    """
    lat = Lattice(np.eye(3), [[0.0, 0.0, 0.0]], periodic_dirs=[0, 1, 2])
    m = TBModel(lattice=lat, spinful=spinful)
    one = np.array([1.0, 0, 0, 0]) if spinful else 1.0
    m.set_onsite([0.0])
    for t, R in ((-1.0, [1, 0, 0]), (-1.0, [0, 1, 0]), (tz, [0, 0, 1])):
        m.set_hop(t * one, 0, 0, R)
    return m


def tetragonal():
    """Tetragonal lattice, one s orbital -> P4/mmm.

    The second model: the *cell* is tetragonal, so the 32 operations of
    ``P4/mmm1'`` are all the Hamiltonian is asked about and it obeys every one
    of them.
    """
    lat = Lattice(np.diag([1.0, 1.0, 1.6]), [[0.0, 0.0, 0.0]], periodic_dirs=[0, 1, 2])
    m = TBModel(lattice=lat)
    m.set_onsite([0.0])
    for t, R in ((-1.0, [1, 0, 0]), (-1.0, [0, 1, 0]), (-0.4, [0, 0, 1])):
        m.set_hop(t, 0, 0, R)
    return m


def diamond():
    """Diamond in the primitive fcc cell -> Fd-3m (nonsymmorphic, refUC and shiftUC both nontrivial)."""
    lat = Lattice([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]],
                  [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]], periodic_dirs=[0, 1, 2])
    m = TBModel(lattice=lat)
    m.set_onsite([0.0, 0.0])
    for R in ([0, 0, 0], [-1, 0, 0], [0, -1, 0], [0, 0, -1]):
        m.set_hop(-1.0, 0, 1, R)
    return m


def bilayer_slab(dz=0.3):
    """A genuinely thick slab: 3D cell, two periodic directions, two layers."""
    lat = Lattice(np.eye(3), [[0.0, 0.0, -dz / 2], [0.0, 0.0, dz / 2]], periodic_dirs=[0, 1])
    m = TBModel(lattice=lat)
    m.set_onsite([0.0, 0.0])
    m.set_hop(-1.0, 0, 1, [0, 0, 0])
    m.set_hop(-0.5, 0, 0, [1, 0, 0])
    m.set_hop(-0.5, 0, 0, [0, 1, 0])
    m.set_hop(-0.5, 1, 1, [1, 0, 0])
    m.set_hop(-0.5, 1, 1, [0, 1, 0])
    return m


def rhombohedral(spinful=True, theta_deg=50.0):
    """One orbital in a rhombohedral primitive cell -> R-3m (#166, #166.98 grey).

    Three cell vectors of equal length at equal angles to the 3-fold axis, and
    a single site on that axis: the point set has the full ``-3m`` of the
    lattice and nothing lower.  ``theta_deg`` is the angle to the axis; any
    value that is not the cubic one leaves ``R-3m``.

    Double-valued ``#166`` is the group the bare-key indicator audit bites
    hardest in (:func:`test_the_bare_key_indicator_forms_are_withheld_from_a_unitary_group`),
    and grey ``#166.98`` is the Fu-Kane-Mele example's own group.
    """
    th = np.deg2rad(theta_deg)
    vecs = [[np.sin(th) * np.cos(p), np.sin(th) * np.sin(p), np.cos(th)]
            for p in (0.0, 2 * np.pi / 3, 4 * np.pi / 3)]
    lat = Lattice(vecs, [[0.0, 0.0, 0.0]], periodic_dirs=[0, 1, 2])
    m = TBModel(lattice=lat, spinful=spinful)
    one = np.array([1.0, 0, 0, 0]) if spinful else 1.0
    m.set_onsite([0.0])
    for R in ([1, 0, 0], [0, 1, 0], [0, 0, 1]):
        m.set_hop(-1.0 * one, 0, 0, R)
    return m


# ---------------------------------------------------------------------------
# test-local character builder (phase A3 owns the library version)
# ---------------------------------------------------------------------------

def bloch_characters(model, group, asg, k, blocks, *, orbital_phase=None, cov_tol=1e-6):
    r"""Traces of the convention-I Bloch symmetry operator over multiplets.

    Builds ``U(g;k) = e^{-2πi k'·t} diag(e^{2πi(k'-k)·τ_j}) O(g)`` with ``O(g)``
    the site permutation measured by :func:`groups._assign_orbitals` (times the
    spinor rotation for a spinful model), **asserts** ``U H U† = H`` at the
    invariant k, and returns ``χ[block, op]`` with the operation order of
    ``little_group(...).unitary`` — the column order
    :func:`groups.identify_from_characters` documents.

    ``orbital_phase(op) -> complex`` optionally multiplies the orbital rep by a
    one-dimensional character of the group; that freedom is exactly what
    distinguishes an s-like from a p_z-like orbital on the same lattice, and
    it is still a valid representation (which is why the covariance battery
    alone cannot pin it -- phase A2's ``_build_rep`` does).
    """
    lg = G.little_group(group, k)
    assert not isinstance(lg, Refusal), lg
    kk = np.atleast_2d(np.asarray(k, dtype=float))
    probe = np.vstack([kk, kk + 0.017])          # >=2 k-points: pythtb squeezes a single one
    _, evec = model.solve_ham(k_pts=probe, return_eigvecs=True, flatten_spin_axis=True)
    V = evec[0].T
    H = model.hamiltonian(k_pts=probe, flatten_spin_axis=True)[0]
    tau = asg.site_positions[asg.site_of_orbital]
    where = {int(s.ind): i for i, s in enumerate(group.symmetries)}

    cols = []
    for op in lg.unitary:
        i = where[int(op.ind)]
        O = np.zeros((model.norb, model.norb), dtype=complex)
        for site in range(asg.site_positions.shape[0]):
            j = int(asg.permutations[i, site])
            for a, b in zip(asg.site_orbitals[site], asg.site_orbitals[j]):
                O[b, a] = 1.0
        if orbital_phase is not None:
            O = O * orbital_phase(op)
        kp = op.transform_k(lg.k3)
        Gv = np.round(kp - lg.k3)
        U = np.exp(-2j * np.pi * kp @ op.translation) * (np.diag(np.exp(2j * np.pi * tau @ Gv)) @ O)
        if model.nspin == 2:
            U = np.kron(U, np.asarray(op.spinor_rotation))
        assert np.abs(U @ H @ U.conj().T - H).max() < cov_tol, (op.ind, "covariance")
        M = V.conj().T @ U @ V
        cols.append([np.trace(M[np.ix_(b, b)]) for b in blocks])
    return np.array(cols).T


def z_parity(op):
    """The 1D character ``σ_h = ±1`` of a slab operation (``R[2,2]`` for a slab group)."""
    return float(op.rotation[2, 2])


# ---------------------------------------------------------------------------
# packaging tripwire
# ---------------------------------------------------------------------------

def test_tables_tripwire_refuses_on_the_dummy_fallback(monkeypatch):
    """irrep silently substitutes DummyMethodTables, and every entry point refuses.

    Which of the two causes bound it -- ``irreptables`` missing, or imported
    before ``irrep`` -- is what ``tables_status`` distinguishes, and the two
    child interpreters of ``test_deps.py`` measure both; here the fallback is
    put in place directly, so what is under test is the gate that follows it.
    """
    from irrep import spacegroup_irreps
    from irrep.utility import DummyMethodTables

    monkeypatch.setattr(spacegroup_irreps, "IrrepTable", DummyMethodTables("IrrepTable"))
    st = G.tables_status()
    assert st["tables"] is False and "DummyMethodTables" in st["reason"]
    assert "irreptables" in st["reason"]

    r = _spacegroup._group_of(square())
    assert not r and isinstance(r, Refusal)
    assert r.gate == "irrep_tables"
    assert "irreptables" in str(r)


def test_refusal_semantics():
    r = Refusal("gate", "because", value=0.5, threshold=1e-3, context={"a": 1})
    assert bool(r) is False and not r
    assert "gate" in str(r) and "0.5" in str(r) and "context" in str(r)
    with pytest.raises(RefusalError):
        r.raise_()


# ---------------------------------------------------------------------------
# (a) group identified from lattice + orbital positions alone
# ---------------------------------------------------------------------------

def test_square_lattice_slab_group():
    g = _spacegroup._group_of(square(), include_TR=False)
    assert g, g
    assert (g.sg_number, g.name, g.size) == (123, "P4/mmm", 16)   # slab number, option (c)
    assert g._tables_ready and g.periodic_dirs == (0, 1) and g.vacuum_dirs == (2,)
    asg = _tables._assign_orbitals(square(), g)
    assert asg.wyckoff_spglib == ("a",) and asg.wyckoff_spglib_dataset == ("a",)
    assert asg.site_symmetry == ("4/mmm",)
    assert len(asg.site_group[0]) == 16


def test_graphene_slab_group():
    """The honeycomb's slab group, its grey group, and the orbit labelling splits.

    ``include_TR=True`` doubles ``#191`` into its grey group: the parent stays in
    ``sg_number`` and the BNS key is ``number_str``, ``191.<bns>``.  With the
    default species labelling the two sites are one orbit, ``2c``; declaring the
    sublattices inequivalent -- which is what a staggered potential does
    physically -- splits it into the two one-site orbits ``e`` and ``c`` of
    ``P-6m2``.
    """
    m = graphene(0.0, -1.0)
    g = _spacegroup._group_of(m, include_TR=False)
    assert g, g
    assert (g.sg_number, g.name, g.size) == (191, "P6/mmm", 24)   # slab number, option (c)
    assert g.number_str == "191" and g.magnetic is False
    asg = _tables._assign_orbitals(m, g)
    assert asg.orbits == ((0, 1),)                                # one orbit, both sublattices
    assert asg.wyckoff_spglib == ("c",) and asg.wyckoff_spglib_dataset == ("c", "c")
    assert asg.site_symmetry == ("-6m2", "-6m2")
    assert len(asg.site_group[0]) == 12                        # D_3h
    assert asg.max_distance < 1e-12
    assert repr(asg) == ("OrbitalAssignment(2 orbitals on 2 sites: sites (0, 1), "
                         "letter c, site symmetry -6m2)")

    grey = _spacegroup._group_of(m, include_TR=True)
    assert grey.number_str.startswith("191.") and grey.sg_number == 191
    assert grey.magnetic is True and grey.name.endswith("1'")
    assert grey.size == 48 and len(grey.u_symmetries) == 24 and len(grey.au_symmetries) == 24

    staggered = _spacegroup._group_of(m, include_TR=False, typat=[0, 1])
    split = _tables._assign_orbitals(m, staggered, typat=[0, 1])
    assert not isinstance(split, Refusal), split
    assert split.orbits == ((0,), (1,))
    assert split.wyckoff_spglib == ("e", "c")   # 1e = (1/3,1/3,0), 1c = (2/3,2/3,0)
    assert split.wyckoff_spglib_dataset == ("e", "c")


def test_the_wyckoff_match_gate_is_a_position_tolerance():
    """``position_tol=`` is the reduced-coordinate gate of the variety match.

    The letter is measured by asking which Wyckoff varieties the transported
    site lies on, and the gate of that question is a distance in the reference
    cell's own reduced coordinates -- the same quantity ``position_tol`` names
    wherever a centre is compared with where the group would put it.  Widening
    it past the separation between varieties makes a general position answer to
    a special one, which is what a gate of ``0.5`` shows here.
    """
    m = graphene(0.0, -1.0)
    g = _spacegroup._group_of(m, include_TR=False)

    tight = G.wyckoff_sites_tables(g, [[0.13, 0.29, 0.41]],
                                   position_tol=G.WYCKOFF_REFERENCE_TOL)
    assert tight and (tight[0].letter, tight[0].n_free) == ("r", 3), \
        "a generic point lies on the general position and on nothing else"
    wide = G.wyckoff_sites_tables(g, [[0.13, 0.29, 0.41]], position_tol=0.5)
    assert wide and (wide[0].letter, wide[0].n_free) == ("a", 0), \
        "at half a cell every variety contains it, and the most special one wins"


# ---------------------------------------------------------------------------
# (f) slab padding independence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("build", [lambda: graphene(0.0, -1.0), bilayer_slab])
def test_slab_group_is_independent_of_the_padding_length(build):
    a = _spacegroup._group_of(build(), include_TR=False, vacuum_length=20.0)
    b = _spacegroup._group_of(build(), include_TR=False, vacuum_length=40.0)
    assert a and b
    assert (a.number_str, a.name, a.size) == (b.number_str, b.name, b.size)
    # and the padding really did change the cell
    assert not np.allclose(a.lattice, b.lattice)


@pytest.mark.slow
def test_the_decoupled_stack_is_the_layer_in_three_periodic_directions():
    """``stack_presentation``: the same ``H``, a third periodic direction, no ``k_z``.

    The layer repeated with no hopping across the gap.  It is a genuine 3D
    crystal -- ``periodic_dirs = (0, 1, 2)``, no vacuum, all six k-points of
    ``P6/mmm1'`` reachable -- and its ``H(k, k_z)`` does not depend on ``k_z``
    at all, which is what makes the ``k_z = 1/2`` rows of a 3D EBR basis
    something a monolayer can be *asked* rather than guessed at.

    Every input shape gives the same crystal: the bare ``lat.shape == (2, 2)``
    model, the same model already re-presented as a slab, and a slab whose
    vacuum is its *middle* axis.  That one's vacuum vector points against the
    plane's normal, a right-handed cell having it so, and the stack is
    right-handed all the same, its normal the plane's; the layer's group,
    carried onto it by each operation's Cartesian action, is the group of the
    stack's own cell, index for index.  And ``c`` is arbitrary -- nothing
    crosses it -- so the group does not move with it.  A slab of two planes is
    stacked with each orbital at its height, its period the slab's extent plus
    ``c``: its Hamiltonian is the slab's exactly at ``k_z = 0`` and up to the
    phases of the heights elsewhere, and its group, carried, is again the
    stack cell's own.
    """
    from pythtb.models import kane_mele

    layer = kane_mele(0.0, -1.0, 0.3, 0.0)
    a = np.asarray(layer.lat_vecs, dtype=float)
    orb = np.zeros((int(layer.norb), 3))
    orb[:, [0, 2]] = np.asarray(layer.orb_vecs, dtype=float)
    middle = TBModel(Lattice([[a[0, 0], a[0, 1], 0.0], [0.0, 0.0, -30.0],
                              [a[1, 0], a[1, 1], 0.0]], orb.tolist(), periodic_dirs=[0, 2]),
                     spinful=True)
    middle.set_onsite(np.asarray(layer.onsite), mode="set")
    for h in layer.hoppings:
        R = list(h.get("lattice_vector", [0, 0]))
        middle.set_hop(h["amplitude"], h["from_orbital"], h["to_orbital"], [R[0], 0, R[1]],
                       mode="add")
    two_d = G.stack_presentation(layer)
    from_slab = G.stack_presentation(_layers._slab_presentation(layer))
    from_middle = G.stack_presentation(middle)
    assert two_d and from_slab and from_middle
    for st in (two_d, from_slab, from_middle):
        assert st.lattice.dim_r == 3 and st.lattice.dim_k == 3
        assert tuple(st.periodic_dirs) == (0, 1, 2)
        assert np.linalg.det(np.asarray(st.lat_vecs, dtype=float)) > 0.0
        assert np.abs(np.asarray(st.orb_vecs)[:, 2]).max() == 0.0
        assert all(int(h.get("lattice_vector", [0, 0, 0])[2]) == 0 for h in st.hoppings)
        assert np.allclose(two_d.lat_vecs, st.lat_vecs)
        assert np.allclose(two_d.orb_vecs, st.orb_vecs)
    assert np.asarray(from_middle.lat_vecs)[2] @ np.asarray(middle.lat_vecs)[1] < 0.0, \
        "the stack's normal is against the middle slab's vacuum vector"

    # H is the layer's, at every k_z -- measured, not assumed
    k2 = np.random.default_rng(5).uniform(-0.5, 0.5, size=(6, 2))
    flat = np.asarray(layer.hamiltonian(k_pts=k2, flatten_spin_axis=True))
    for kz in (0.0, 0.25, 0.5, -0.375):
        k3 = np.concatenate([k2, np.full((6, 1), kz)], axis=1)
        for st in (two_d, from_middle):
            assert np.abs(flat - np.asarray(
                st.hamiltonian(k_pts=k3, flatten_spin_axis=True))).max() == 0.0

    # the middle slab's group, carried onto its stack across the reversed normal,
    # is the stack cell's own: the same operations under the same indices, the
    # same setting of the tables, the same k-points
    cell = _spacegroup._group_of(middle)
    assert cell.number_str == "191.234" and cell.vacuum_dirs == (1,)
    carried, _sites = _layers._layer_on_stack(cell, _tables._assign_orbitals(middle, cell), from_middle)
    found = _spacegroup._group_of(from_middle)
    assert [int(s.ind) for s in carried.symmetries] == [int(s.ind) for s in found.symmetries]
    for s, f in zip(carried.symmetries, found.symmetries):
        assert np.array_equal(np.asarray(s.rotation), np.asarray(f.rotation))
        d = np.asarray(s.translation, dtype=float) - np.asarray(f.translation, dtype=float)
        assert np.abs(d - np.rint(d)).max() < 1e-9
    assert np.allclose(carried._refUC, found._refUC)
    assert np.allclose(carried._shiftUC, found._shiftUC)
    assert {n: tuple(np.round(k, 9)) for n, k in G.high_symmetry_points(carried).items()} \
        == {n: tuple(np.round(k, 9)) for n, k in G.high_symmetry_points(found).items()}

    g = _spacegroup._group_of(two_d)
    assert g.number_str == "191.234" and g.dim_k == 3 and g.vacuum_dirs == ()
    assert sorted(G.high_symmetry_points(g)) == ["A", "GM", "H", "K", "L", "M"]
    wide = _spacegroup._group_of(G.stack_presentation(layer, c=13.7))
    assert (wide.number_str, wide.size) == (g.number_str, g.size), "c cannot matter"

    # the gate, a measurement: a fully periodic model is already its own stack
    # and there is nothing to build
    assert G.stack_presentation(square()), "a 2D square layer is a layer"
    assert G.stack_presentation(G.stack_presentation(layer)).gate == "not_a_layer"

    # ... and a slab of two planes is stacked at its heights: the period its 0.3
    # plus c, its H the slab's exactly at k_z = 0 and in the gauge of the heights
    # elsewhere, which they do enter, and its group, carried onto the stack, the
    # stack cell's own, index for index
    slab = bilayer_slab()
    two = G.stack_presentation(slab)
    period = float(np.linalg.norm(np.asarray(two.lat_vecs, dtype=float)[2]))
    assert period == pytest.approx(0.3 + G.STACK_C, abs=1e-12)
    z = np.asarray(two.orb_vecs, dtype=float)[:, 2]
    assert z * period == pytest.approx([-0.15, 0.15], abs=1e-12)
    layer_h = np.asarray(slab.hamiltonian(k_pts=k2, flatten_spin_axis=True))
    for kz in (0.0, 0.25, 0.5, -0.375):
        k3 = np.concatenate([k2, np.full((6, 1), kz)], axis=1)
        stack_h = np.asarray(two.hamiltonian(k_pts=k3, flatten_spin_axis=True))
        lam = np.exp(2j * np.pi * kz * z)
        gauged = np.conj(lam)[None, :, None] * layer_h * lam[None, None, :]
        assert np.abs(stack_h - gauged).max() <= (0.0 if kz == 0.0 else 1e-12)
        assert (np.abs(stack_h - layer_h).max() > 1e-3) == (kz != 0.0)
    cell = _spacegroup._group_of(slab)
    carried, _sites = _layers._layer_on_stack(cell, _tables._assign_orbitals(slab, cell), two)
    found = _spacegroup._group_of(two)
    assert (cell.number_str, found.number_str) == ("123.340", "123.340")
    assert [int(s.ind) for s in carried.symmetries] == [int(s.ind) for s in found.symmetries]
    for s, f in zip(carried.symmetries, found.symmetries):
        assert np.array_equal(np.asarray(s.rotation), np.asarray(f.rotation))
        d = np.asarray(s.translation, dtype=float) - np.asarray(f.translation, dtype=float)
        assert np.abs(d - np.rint(d)).max() < 1e-9


def test_padded_cell_preserves_reduced_k_and_positions():
    """Padding scales non-periodic vectors, so periodic reduced coordinates are untouched.

    The k half pins the **embedding**, not its inverse.  ``k_from_3d(k_to_3d(k))
    == k`` is the identity for any index set at all, right or wrong -- the two
    write and read the same columns -- and ``k_to_3d(k)[:, 2] == 0`` is equally
    blind to a permutation among the periodic axes.  So each direction is
    pinned on its own: which column a reduced component lands in, and which
    columns are read back.
    """
    m = bilayer_slab()
    g = _spacegroup._group_of(m, include_TR=False, vacuum_length=40.0)
    assert np.allclose(g.site_positions[:, :2], m.orb_vecs[:, :2] % 1.0)
    assert tuple(g.periodic_dirs) == (0, 1) and tuple(g.vacuum_dirs) == (2,)
    assert np.array_equal(g.k_to_3d([[0.23, 0.41]]), [[0.23, 0.41, 0.0]])
    assert np.array_equal(g.k_from_3d([[0.23, 0.41, 0.77]]), [[0.23, 0.41]])


# ---------------------------------------------------------------------------
# (c) refusals
# ---------------------------------------------------------------------------

def test_default_typat_treats_same_content_sites_as_one_species():
    """Documented limit of the default: two like sites make the cell look centred.

    Re-pinned in wave R4a, which retired the ``non_primitive_cell`` refusal this
    used to assert: a centred cell is now *served*, as the lattice supergroup it
    is (``lattice_index == 2``, the centring translation among the operations),
    and only ``realized_group`` can say whether the Hamiltonian obeys it.  The
    statement being made is unchanged and is now measured in more detail: the
    default typing makes the cell look centred, and ``typat=[0, 1]`` is the
    declaration that removes the appearance.  ``tests/test_lattice_descent.py``
    carries the rest.
    """
    r = _spacegroup._group_of(square_two_orbitals(0.0), include_TR=False)
    assert r, str(r)
    assert (r.number_str, r.name, r.size) == ("123", "P4/mmm", 32)
    assert r._lattice_index == 2
    assert r._fractional_translations == ((0.5, 0.5, 0.0),)
    assert r._tables_ready is False
    labelled = _spacegroup._group_of(square_two_orbitals(0.0), include_TR=False, typat=[0, 1])
    assert labelled and labelled.sg_number == 123
    assert labelled._lattice_index == 1


def test_orbital_off_every_wyckoff_orbit_refuses():
    ideal = square_two_orbitals(0.0)
    g = _spacegroup._group_of(ideal, include_TR=False, typat=[0, 1])
    assert g and g.sg_number == 123
    assert _tables._assign_orbitals(ideal, g, typat=[0, 1])       # the ideal model is fine

    broken = square_two_orbitals(shift=-0.06)             # 1c orbital pushed off its orbit
    r = _tables._assign_orbitals(broken, g, typat=[0, 1])
    assert not r and isinstance(r, Refusal)
    assert r.gate == "wyckoff_orbit"
    assert r.value == pytest.approx(0.12, abs=1e-9)       # 2 * 0.06, the C4-image mismatch
    assert "modelling bug" in r.reason
    assert set(r.context) >= {"site", "orbitals", "operation_ind", "position"}


def test_a_model_and_a_group_built_on_another_lattice_raise():
    """An assignment is asked of a model and the group of its own cell.

    It maps this model's orbitals onto that group's site orbits, so a group
    whose padded cell is not this model's leaves nothing to map onto.  Every
    call in the package hands it the group of the model's own cell, so a pair
    from two cells is a malformed argument: it raises, saying how far apart the
    two lattices are.
    """
    g = _spacegroup._group_of(graphene(0.0, -1.0), include_TR=False)
    assert g and g.sg_number == 191
    with pytest.raises(ValueError, match="the group was built for another cell"):
        _tables._assign_orbitals(square(), g)


# ---------------------------------------------------------------------------
# (c2) the magnetic scope gate -- types I and II are the validated universe
# ---------------------------------------------------------------------------

#: Opposite moments on the two sites -- the same antiferromagnet on two cells,
#: and the cells decide the type.  On the honeycomb the two sublattices are
#: related by a *rotation*, so every antiunitary operation carries a rotation
#: with it: type III.  On the two-site square cell they are related by the
#: *translation* (1/2,1/2,0), which the moments break and which survives
#: combined with T -- an anti-translation, type IV, and the ``P_C`` prefix of
#: the BNS symbol is that statement in the tables' own notation.
AFM = dict(magmom=[[0, 0, 1.0], [0, 0, -1.0]])


def test_magnetic_types_iii_and_iv_are_refused_at_the_constructor(monkeypatch):
    """The scope gate, on the sole ``GroupData`` constructor.

    This package's validated universe is the 230 space groups x {unitary, grey}
    -- magnetic types I and II.  spglib will happily identify a type-III or
    type-IV group from ``magmom=``, and for all 1651 BNS groups ``irreptables``
    ships the elementary band representations and the unitary part's
    characters at the maximal k-points; but nothing the package reads gives
    the corep type at a k-point, the site-symmetry coreps or the coordinates of
    the magnetic Wyckoff positions, and every convention here is written for
    types I and II.  The refusal is made at :func:`groups._group_of` so that no
    downstream function can be handed a record it has no data or convention
    for.

    The type is read off the operations, never off the number: ``I`` no
    antiunitary operations at all; ``II`` the coset contains ``{1|0}.T``
    (modulo the lattice -- an integer-translation composite is pure ``T``);
    ``IV`` it contains ``{1|t}.T`` with ``t`` fractional; ``III`` it contains
    antiunitary operations but no translation-times-``T`` of either kind.  The
    BNS number is never parsed -- ``191.234`` and ``191.236`` differ by one
    digit and by the whole question.  And the gate reads the result, not the
    input: ``magmom=`` still reaches the group when its symmetry is in scope --
    zero moments give the grey group, and a magnetic arrangement whose symmetry
    admits no antiunitary operation at all is an ordinary type-I magnetic group,
    served with ``magnetic=True`` and a BNS ``number_str``, which is what makes
    the structural detection necessary in the first place.
    """
    three = _spacegroup._group_of(graphene(0.0, -1.0), **AFM)
    assert not three and three.gate == "magnetic_type_out_of_scope"
    assert three.context["magnetic_type"] == "III"
    assert three.context["number_str"] == "191.236"        # P6'/mm'm
    assert three.context["name"] == "P6'/mm'm"
    assert three.context["n_antiunitary"] == 12
    assert three.context["in_scope"] == ("I", "II")
    assert "types I and II" in three.reason
    assert "not the corep type at a k-point" in three.reason

    four = _spacegroup._group_of(square_two_orbitals(0.0), **AFM)
    assert not four and four.gate == "magnetic_type_out_of_scope"
    assert four.context["magnetic_type"] == "IV"
    assert four.context["number_str"] == "127.397"         # P_C4/mbm
    assert "anti-translation" in four.reason

    # The same cell WITHOUT the moments is the lattice supergroup of the test
    # above -- re-pinned in wave R4a, which retired the `non_primitive_cell`
    # refusal this line used to assert: it is the pure translation the moments
    # break, and which the bare cell carries as a UNITARY operation
    # (`lattice_index == 2`), that makes the magnetic group type IV rather than
    # type II.  Grey it stays type II, which is what `_magnetic_type` reads off
    # `{1|0}.T` before any of this.
    bare = _spacegroup._group_of(square_two_orbitals(0.0))
    assert bare and bare._lattice_index == 2
    assert bare._fractional_translations == ((0.5, 0.5, 0.0),)
    assert _spacegroup._magnetic_type(bare) == "II"

    # the type, read off the operations: what the antiunitary coset contains
    m = graphene(0.0, -1.0)
    assert _spacegroup._magnetic_type(_spacegroup._group_of(m, include_TR=False)) == "I"
    grey = _spacegroup._group_of(m, include_TR=True)
    assert _spacegroup._magnetic_type(grey) == "II"
    with monkeypatch.context() as mp:                 # build what the gate stops
        patch_everywhere(mp, "_magnetic_scope_refusal", lambda _sg: None)
        built_three = _spacegroup._group_of(m, **AFM)
        built_four = _spacegroup._group_of(square_two_orbitals(0.0), **AFM)
    assert built_three.number_str == "191.236" and built_four.number_str == "127.397"
    assert _spacegroup._magnetic_type(built_three) == "III" and _spacegroup._magnetic_type(built_four) == "IV"

    def identity_rotation_translations(g):
        out = []
        for op in g.au_symmetries:
            if np.allclose(np.asarray(op.rotation, dtype=float), np.eye(3), atol=1e-8):
                t = np.asarray(op.translation, dtype=float)
                out.append(np.abs(((t + 0.5) % 1.0) - 0.5).max())
        return out

    # II carries pure T; III carries no identity-rotation antiunitary at all;
    # IV carries one, and its translation is fractional -- (1/2,1/2,0) here.
    assert identity_rotation_translations(grey) == [pytest.approx(0.0)]
    assert identity_rotation_translations(built_three) == []
    assert identity_rotation_translations(built_four) == [pytest.approx(0.5)]
    assert _spacegroup._magnetic_scope_refusal(grey) is None
    assert _spacegroup._magnetic_scope_refusal(built_three).context["magnetic_type"] == "III"

    # ... and magmom= in scope still reaches the group
    zero = _spacegroup._group_of(m, magmom=[[0, 0, 0.0], [0, 0, 0.0]])
    assert zero and zero.number_str == "191.234" and _spacegroup._magnetic_type(zero) == "II"
    generic = _spacegroup._group_of(m, magmom=[[0.3, 0.5, 1.0], [0.7, -0.2, 0.4]])
    assert generic and generic.number_str == "1.1"         # P1, type I(BNS)
    assert generic.magnetic is True and not list(generic.au_symmetries)
    assert _spacegroup._magnetic_type(generic) == "I"


@pytest.mark.data
def test_the_bns_census_the_scope_statement_rests_on():
    """The 1651 BNS types by kind.

    The validated universe is types I and II, which is 460 of the 1651 magnetic
    space-group types; the 1191 that are refused are the type-III and type-IV
    ones.  Counted here from ``spglib`` rather than quoted, so the docs' number
    moves only when the tables do.
    """
    import spglib

    census = {}
    for uni_number in range(1, 1652):
        entry = spglib.get_magnetic_spacegroup_type(uni_number)
        kind = entry["type"] if isinstance(entry, dict) else entry.type
        census[kind] = census.get(kind, 0) + 1

    assert census == {1: 230, 2: 230, 3: 674, 4: 517}
    assert census[1] + census[2] == 460                    # the validated scope
    assert census[3] + census[4] == 1191                   # what the gate refuses
    assert sum(census.values()) == 1651


def test_out_of_plane_k_refuses():
    g = _spacegroup._group_of(square(), include_TR=False)
    r = G.little_group(g, [0.0, 0.0, 0.5])
    assert not r and r.gate == "out_of_plane"
    assert G.little_group(g, [0.0, 0.0, 0.0])             # in-plane 3-vector is fine
    with pytest.raises(ValueError):
        G.little_group(g, [0.0])


# ---------------------------------------------------------------------------
# (d) product table, including spinor factors
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_product_table_with_spinor_factors():
    g = _spacegroup._group_of(square(spinful=True), include_TR=False)
    assert g and g.spinor is True and g.size == 16
    table, dt, spinor = g.get_product_table(get_diff=True)

    assert table.shape == (16, 16) and spinor.shape == (16, 16)
    for row in table:                                     # every row/column is a permutation
        assert sorted(row.tolist()) == list(range(16))
    for col in table.T:
        assert sorted(col.tolist()) == list(range(16))

    assert set(np.unique(spinor)) == {-1, 1}, "the double group must be genuinely projective"
    assert np.issubdtype(dt.dtype, np.integer)

    # the reported spinor factor is exactly the SU(2) mismatch of the product
    for i, si in enumerate(g.symmetries):
        for j, sj in enumerate(g.symmetries):
            k = int(table[i, j])
            prod = np.asarray(si.spinor_rotation) @ np.asarray(sj.spinor_rotation)
            assert np.allclose(prod, spinor[i, j] * np.asarray(g.symmetries[k].spinor_rotation))

    # the inverse table against its defining property, g.g^-1 = g^-1.g = e.
    # That both products land on the *same* element is not it: the identity
    # permutation ("every operation is its own inverse") satisfies that for
    # every i, and passes the permutation check above too.  The true table is
    # [0, 1, 3, 2, ..., 11, 10, ...].
    e = next(i for i, s in enumerate(g.symmetries)
             if np.allclose(np.asarray(s.rotation, dtype=float), np.eye(3))
             and np.allclose(np.asarray(s.translation, dtype=float) % 1.0, 0.0)
             and not s.time_reversal)
    assert np.array_equal(table[e], np.arange(16)), "e is the table's identity row"
    inv = g.get_inverse_table()
    assert sorted(inv.tolist()) == list(range(16))
    flat = g.get_product_table(get_diff=False)
    for i in range(16):
        assert int(table[i, inv[i]]) == e == int(flat[inv[i], i])


def test_the_product_table_is_memoized_and_still_handed_out_fresh(monkeypatch):
    """``irrep`` rebuilds the table at O(N^3) per call; we build it once.

    Three calls live inside one ``_build_rep`` -- its own product-table gate,
    plus twice through ``_sitechars.site_group_table`` -- and on the 96
    operations of the FKM diamond group each cost 6.4 s, which is why that
    ``_build_rep`` measured 37.8 s cold and 16.8 s once the table is kept.
    The table is immutable group data, so it is computed once per
    ``GroupData``, with its projective data, and the table alone is read off
    that record.  What callers
    get back is still a *fresh* array every time, so the memo cannot be reached
    through anything they do with it, and ``dataclasses.replace`` starts an
    empty memo rather than inheriting one built for a different ``_sg``.

    The saving is counted, not inferred: ``irrep``'s own
    ``SpaceGroup.get_product_table`` is replaced by a counting wrapper, so the
    number of *builds* is visible.  Equal values across two calls, freshness of
    the returned arrays and an empty ``_product_tables`` on a ``replace()``d
    copy are all equally true of a passthrough that rebuilds every time; the
    counter is what separates the two.
    """
    # `_group_of` hands the same GroupData back for the same model, so its memo
    # may already be warm from an earlier test in this session; `replace` is
    # the documented route to a copy that starts empty, and it is the very
    # thing asserted at the end.
    g = dataclasses.replace(_spacegroup._group_of(square(spinful=True), include_TR=False))
    builds = []
    real_build = g._sg.get_product_table

    def counting(get_diff=True, _real=real_build):
        builds.append(bool(get_diff))
        return _real(get_diff=get_diff)

    monkeypatch.setattr(g._sg, "get_product_table", counting)
    assert g._product_tables == {}, "nothing memoized before the first call"

    first = g.get_product_table(get_diff=True)
    again = g.get_product_table(get_diff=True)
    assert builds == [True], "two calls, one build"
    assert set(g._product_tables) == {True}, "the get_diff=True table is kept"
    assert all(np.array_equal(a, b) for a, b in zip(first, again))
    assert all(a is not b for a, b in zip(first, again)), "a fresh copy per call"

    flat, flat2 = g.get_product_table(get_diff=False), g.get_product_table(get_diff=False)
    assert builds == [True], "the table alone is read off the record: no second build"
    assert set(g._product_tables) == {True}, "one record kept"
    assert np.array_equal(flat, flat2) and flat is not flat2
    assert np.array_equal(flat, first[0]), "the table alone is the record's first array"

    # mutating what a caller was handed cannot reach the memo
    first[0][0, 0] = -99
    assert int(g.get_product_table(get_diff=True)[0][0, 0]) == int(again[0][0, 0])
    assert len(g.get_product_table(get_diff=True)) == 3
    assert builds == [True], "and none of that rebuilt anything"

    fresh = dataclasses.replace(g, provenance="realized")
    assert fresh._product_tables == {}, "replace() does not inherit a memo"
    assert np.array_equal(fresh.get_product_table(get_diff=False), flat)
    assert builds == [True, True], "the fresh copy pays for its own table, built whole"


# ---------------------------------------------------------------------------
# (e) identification from characters -- the tables' index matching
# ---------------------------------------------------------------------------

def test_graphene_doublet_at_K_is_the_two_dimensional_irrep():
    """Literature anchor: the K doublet is 2D -- K5 for s-like, K6 for p_z-like orbitals."""
    m = graphene(0.0, -1.0)
    g = _spacegroup._group_of(m, include_TR=False)
    asg = _tables._assign_orbitals(m, g)
    k = G.high_symmetry_points(g)["K"]
    assert np.allclose(k, [1 / 3, 2 / 3]), "the tables' K = (1/3,1/3) is (1/3,2/3) in this cell"

    chars = bloch_characters(m, g, asg, k, [[0, 1]])
    out = G.identify_from_characters(g, k, chars, multiplet_dims=[2])
    assert out, out
    assert out.labels == ({"K5": 1},) and out.dims == pytest.approx([2.0])

    chars_pz = bloch_characters(m, g, asg, k, [[0, 1]], orbital_phase=z_parity)
    out_pz = G.identify_from_characters(g, k, chars_pz, multiplet_dims=[2])
    assert out_pz, out_pz
    assert out_pz.labels == ({"K6": 1},), "p_z (sigma_h-odd) gives the other 2D irrep"


def test_graphene_gamma_content_and_the_phase_invariant_partition():
    m = graphene(0.0, -1.0)
    g = _spacegroup._group_of(m, include_TR=False)
    asg = _tables._assign_orbitals(m, g)
    chars = bloch_characters(m, g, asg, np.zeros(2), [[0], [1]])
    out = G.identify_from_characters(g, np.zeros(2), chars, multiplet_dims=[1, 1])
    assert out, out
    assert out.labels == ({"GM1+": 1}, {"GM4-": 1})
    assert out.partition == (0, 1), "the two bands carry different content"

    same = G.identify_from_characters(g, np.zeros(2), np.vstack([chars[0], chars[0]]),
                                      multiplet_dims=[1, 1])
    assert same.partition == (0, 0), "identical character rows are the same content"


def test_identify_refuses_a_k_point_the_tables_do_not_carry():
    m = graphene(0.0, -1.0)
    g = _spacegroup._group_of(m, include_TR=False)
    lg = G.little_group(g, [0.17, 0.31])
    chars = np.ones((1, len(lg.unitary)))
    r = G.identify_from_characters(g, [0.17, 0.31], chars)
    assert not r and r.gate == "kpoint_not_tabulated"
    assert set(r.context["available"]) == {"GM", "K", "M"}


def test_corep_table_is_keyed_by_our_operation_indices():
    m = graphene(0.0, -1.0)
    g = _spacegroup._group_of(m, include_TR=True)                     # grey group -> coreps
    k = G.high_symmetry_points(g)["K"]
    lg = G.little_group(g, k)
    tab = G.corep_table(g, k)
    assert not isinstance(tab, Refusal), tab
    assert tab and all(lab.startswith("K") for lab in tab), "single-valued coreps at K"
    for chars in tab.values():
        assert set(chars) == set(lg.indices), "table keys are our unitary little-group indices"
    assert lg.antiunitary, "the grey group must contribute antiunitary operations"
    assert not set(s.ind for s in lg.antiunitary) & set(lg.indices)
    assert repr(lg) == ("LittleGroup(k = (0.3333, 0.6667): unitary operations "
                        "1-3, 7-9, 16-18, 22-24; with time reversal 28-30, 34-39, 43-45)")

    r = G.corep_table(g, [0.17, 0.31])
    assert not r and r.gate == "kpoint_not_tabulated"


def test_identify_raises_on_a_wrongly_shaped_character_table():
    m = graphene(0.0, -1.0)
    g = _spacegroup._group_of(m, include_TR=False)
    with pytest.raises(ValueError, match="one column per unitary"):
        G.identify_from_characters(g, np.zeros(2), np.ones((1, 3)))


def test_diamond_content_matches_the_EBR_oracle_at_every_point_including_W():
    """All four k-points of the ``A1↑G(2) @ 8a`` row, ``W`` included.

    ``irreptables``' own EBR data says the s-orbital band representation
    ``A1↑G(2) @ 8a`` of Fd-3m reads ``GM1+ GM2- L1+ L2- W1 X1``, and every one
    of those is reproduced from the Hamiltonian.

    ``W`` is the historical case: it used to come back as multiplicities
    ``0.5 / 0.5`` and an ``integer_multiplicity`` refusal, because the tabulated
    characters were compared without conjugating ``irreptables``' factor system
    (BCS represents a lattice translation by ``e^{+ik·t}``, pythtb by the
    physical ``e^{-ik·t}``; see ``groups._transported_table``).  ``W`` is the
    only k-point of Fd-3m that is *not* time-reversal invariant and whose
    little group carries fractional translations, which is exactly the
    condition under which the two conventions differ.  The residual
    integrality defect is now 1.3e-6 —
    ``irreptables``' own printed precision, not a physical effect.

    The setting this rests on is asserted first: ``Fd-3m`` is nonsymmorphic,
    half its operations carrying a fractional translation, and the fixture
    exercises both ``refUC`` and ``shiftUC``.
    """
    m = diamond()
    g = _spacegroup._group_of(m, include_TR=False)
    assert (g.sg_number, g.name, g.size) == (227, "Fd-3m", 48)
    frac = sum(1 for s in g.symmetries if np.abs(s.translation).max() > 1e-8)
    assert frac == 24, "half of Fd-3m's operations carry a fractional translation"
    assert not np.allclose(g._refUC, np.eye(3)), "refUC must be exercised by this fixture"
    assert np.abs(g._shiftUC).max() > 1e-8, "shiftUC must be exercised by this fixture"
    asg = _tables._assign_orbitals(m, g)
    assert asg.orbits == ((0, 1),)
    assert asg.wyckoff_spglib == asg.wyckoff_spglib_dataset[:1], "our letters must track spglib's"
    assert len(asg.site_group[0]) == 24                        # Td
    hsp = G.high_symmetry_points(g)

    oracle = G.ebr_data(g)
    assert oracle, oracle
    a1 = [x for x, (name, wy) in zip(oracle["raw"]["ebrs"], oracle["ebrs"])
          if wy.startswith("8a") and name.startswith("A1")][0]
    content = {lab: int(v) for lab, v
               in zip(oracle["labels_physics_convention"], a1["vector"]) if v}
    assert content == {"GM1+": 1, "GM2-": 1, "L1+": 1, "L2-": 1, "W1": 1, "X1": 1}

    got_all = {}
    for name, blocks, dims in (("GM", [[0], [1]], [1, 1]),
                               ("L", [[0], [1]], [1, 1]),
                               ("X", [[0, 1]], [2]),
                               ("W", [[0, 1]], [2])):
        chars = bloch_characters(m, g, asg, hsp[name], blocks)
        out = G.identify_from_characters(g, hsp[name], chars, multiplet_dims=dims)
        assert out, (name, out)
        got = {}
        for d in out.labels:
            for lab, v in d.items():
                got[lab] = got.get(lab, 0) + v
        assert got == {lab: v for lab, v in content.items() if lab.startswith(name)}, name
        assert out.integrality < 2e-6, name
        got_all.update(got)
    assert got_all == content, "the whole EBR row, from the Hamiltonian"


# ---------------------------------------------------------------------------
# the tabulated factor system: a direct unit test of the transport
# ---------------------------------------------------------------------------

def _class_function_defect(rot, trans, k, chi, sign, su2=None):
    r"""Worst violation of ``χ(h) = ±e^{sign·2πi k·m} χ(g h g⁻¹)`` over the group.

    ``rot``/``trans`` give ``{R|t}`` for each index, in the basis ``k`` is
    expressed in; ``chi`` gives the character.  A little-group representation in
    the **physical** convention has ``D({1|m}) = e^{-2πi k·m}`` for a lattice
    translation ``m`` (``sign = -1``), forced by
    ``ψ_k(r - m) = e^{-2πi k·m} ψ_k(r)``, so its character is a class function
    with that phase.  ``sign = +1`` is BCS's historical convention, the one
    ``irreptables`` tabulates in.  Nothing here uses the code under test beyond
    the ``{R|t}`` data itself.

    ``su2`` supplies the spinor matrices ``{ind: S}`` for a **double-valued**
    table, where the conjugate of a group element is only defined up to the
    ``2π`` rotation: ``g̃h̃g̃⁻¹`` is the barred partner of ``{Rc|·}`` whenever
    ``S_g S_h S_g⁻¹ = −S_ℓ``, and the character then picks up that sign.  Omit
    it for a single-valued table, where the sign is always ``+1``.
    """
    inds = sorted(chi)
    by_rot = {}
    for i in inds:
        by_rot.setdefault(tuple(np.round(rot[i]).astype(int).ravel()), i)
    worst = 0.0
    for i in inds:
        Ri, ti = rot[i], trans[i]
        Rinv = np.linalg.inv(Ri)
        for j in inds:
            Rc = Ri @ rot[j] @ Rinv
            ell = by_rot.get(tuple(np.round(Rc).astype(int).ravel()))
            if ell is None:
                continue
            m = ti + Ri @ trans[j] - Rc @ ti - trans[ell]
            phase = np.exp(sign * 2j * np.pi * np.asarray(k, float) @ m)
            worst = max(worst, abs(chi[j] - _spinor_sign(su2, i, j, ell) * phase
                                   * chi[ell]))
    return worst


def _spinor_sign(su2, i, j, ell):
    """``±1``: does ``S_i S_j S_i⁻¹`` equal ``S_ℓ`` or its ``2π``-rotated partner?"""
    if su2 is None:
        return 1.0
    Sc = su2[i] @ su2[j] @ np.linalg.inv(su2[i])
    return (1.0 if np.abs(Sc - su2[ell]).max() <= np.abs(Sc + su2[ell]).max()
            else -1.0)


def _our_gauge_defect(group, k, row, sign=-1):
    """``_class_function_defect`` on **our** operations, in our reduced basis."""
    lg = G.little_group(group, k)
    where = {int(s.ind): i for i, s in enumerate(group.symmetries)}
    rot = {i: np.asarray(group.symmetries[where[i]].rotation, dtype=float)
           for i in lg.indices}
    trans = {i: np.asarray(group.symmetries[where[i]].translation, dtype=float)
             for i in lg.indices}
    return _class_function_defect(rot, trans, lg.k3,
                                  {i: complex(row[i]) for i in lg.indices}, sign)


def _tabulated_row(group, kpname, label):
    """The raw ``irreptables`` row plus its own ``{R|t}`` and k, untouched."""
    from irreptables.irreps import IrrepTable
    table = IrrepTable(group.number_str, group.spinor, magnetic=group.magnetic)
    irr = next(ir for ir in table.irreps
               if ir.kpname == kpname and ir.name == label)
    inds = sorted(irr.characters)
    rot = {i: np.asarray(table.symmetries[i - 1].R, dtype=float) for i in inds}
    trans = {i: np.asarray(table.symmetries[i - 1].t, dtype=float) for i in inds}
    chi = {i: complex(irr.characters[i]) for i in inds}
    return rot, trans, np.asarray(irr.k, dtype=float), chi


#: ``irreptables`` prints |χ| and the phase to five decimals, so a character of
#: modulus 2 arrives with an absolute error of order 1e-5.
_TABLE_PRECISION = 5e-5


@pytest.mark.slow
@pytest.mark.parametrize("build,kname,kind", [
    (lambda: graphene(0.0, -1.0), "K", "symmorphic, primitive, k != -k"),
    (cubic, "X", "symmorphic, primitive, TRIM"),
    (diamond, "X", "nonsymmorphic, face-centred, TRIM"),
    (diamond, "W", "nonsymmorphic, face-centred, k != -k"),
])
def test_the_transported_table_conjugates_only_the_factor_system(build, kname, kind):
    """The boundary rule, asserted on the physically same group element.

    ``corep_table`` transports ``irreptables``' row onto our coset
    representatives by *conjugating the factor system and nothing else*.  Two
    statements, both built from ``irreptables``' own ``{R|t}`` data rather than
    from the transport formula:

    1. the **loaded (point-group) half** survives untouched — stripping BCS's
       phase from the tabulated row, ``Γ(R) = χ_table·e^{-2πi k·t_table}``, and
       stripping *our* phase from what ``corep_table`` returns,
       ``Γ'(R) = χ_ours·e^{+2πi k·t_ours}``, give the same numbers (up to the
       spinor matching sign ``sym.sign``, which is ``irrep``'s and is ±1).
       This one is a *formula* check, not a physics check: it says the served
       row is exactly the stated transport of the tabulated row, and nothing
       about whether either is a character.  So it is asserted at a few ulp of
       the character's own scale rather than at table precision — the defect is
       0.0 at every operation of all four cases, measured, because the two
       phases cancel bit for bit;
    2. the raw tabulated row is a class function in BCS's convention
       ``D({1|m}) = e^{+2πi k·m}``, while what ``corep_table`` hands back is one
       in the **physical** convention ``D({1|m}) = e^{-2πi k·m}``, on *our*
       representatives and in *our* reduced coordinates — the convention
       :mod:`tqc.sewing` measures in.

    The parametrization takes one case of each kind -- symmorphic/nonsymmorphic,
    primitive/centred and TRIM/non-TRIM -- so that the statement is not carried
    by the one case (diamond's ``W``) where the two conventions give different
    numbers.
    """
    m = build()
    g = _spacegroup._group_of(m, include_TR=False)
    hsp = G.high_symmetry_points(g)
    assert kname in hsp, sorted(hsp)
    tab = G.corep_table(g, hsp[kname])
    assert tab, tab
    lg = G.little_group(g, hsp[kname])
    where = {int(s.ind): i for i, s in enumerate(g.symmetries)}

    for label, row in tab.items():
        rot, trans, k_ref, chi = _tabulated_row(g, kname, label)
        assert sorted(row) == sorted(chi), f"{label}: little group index sets"

        # (1) the loaded half is the same object on both sides of the boundary
        for i in lg.indices:
            sym = g.symmetries[where[i]]
            t_ours = sym.translation_refUC(g._refUC, g._shiftUC)
            loaded_table = chi[i] * np.exp(-2j * np.pi * trans[i] @ k_ref)
            loaded_ours = row[i] * np.exp(+2j * np.pi * t_ours @ k_ref) / sym.sign
            assert abs(loaded_table - loaded_ours) <= 8 * _ULP * max(1.0, abs(loaded_table)), (
                f"{label}/op {i}: the point-group content moved")

        # (2) each side is a genuine character in its own convention
        assert _class_function_defect(rot, trans, k_ref, chi, +1) < _TABLE_PRECISION, (
            f"{label}: the raw table is not a class function in BCS's convention")
        assert _our_gauge_defect(g, hsp[kname], row) < _TABLE_PRECISION, (
            f"{label}: the transported row is not a character of our little group")


def test_irreps_own_transport_is_not_a_character_of_ours_at_diamonds_W():
    """The correction is load-bearing, and only where the theory says it is.

    ``irrep``'s ``get_irreps_from_table`` transports by ``sym.sign`` and
    ``exp(2πi·δt·k_ref)`` with ``δt = t_table − t_ours``, which keeps BCS's
    ``e^{+ik·t}`` convention for the translations.  That agrees with ours
    whenever ``k·t_table`` is a half-integer — every k-point of Fd-3m except
    ``W``, the only one that is neither time-reversal invariant nor free of
    fractional translations.  At ``W`` ``irrep``'s row is not a character of our
    little group at all: worst class-function violation 2.83 = 2√2, which is
    exactly where the old ``0.5 / 0.5`` multiplicities came from.
    """
    g = _spacegroup._group_of(diamond(), include_TR=False)
    hsp = G.high_symmetry_points(g)
    lg_of = {n: G.little_group(g, hsp[n]) for n in ("GM", "X", "L", "W")}

    for name in ("GM", "X", "L"):
        ours = G.corep_table(g, hsp[name])
        theirs = g._sg.get_irreps_from_table(name, lg_of[name].k3 % 1.0)
        for label, row in ours.items():
            assert max(abs(row[i] - theirs[label][i]) for i in row) < _TABLE_PRECISION, (
                f"{name}/{label}: the correction must be a no-op here")

    ours = G.corep_table(g, hsp["W"])
    theirs = g._sg.get_irreps_from_table("W", lg_of["W"].k3 % 1.0)
    assert set(ours) == set(theirs) == {"W1", "W2"}
    for label, row in ours.items():
        assert _our_gauge_defect(g, hsp["W"], row) < _TABLE_PRECISION
        assert _our_gauge_defect(g, hsp["W"], theirs[label]) == pytest.approx(
            2 * np.sqrt(2), abs=1e-4), (
            f"{label}: irrep's own transport is not a character at all here")
        assert max(abs(row[i] - theirs[label][i]) for i in row) > 1.0, (
            f"{label}: and it differs from ours by more than rounding")


# ---------------------------------------------------------------------------
# EBR / indicator passthroughs
# ---------------------------------------------------------------------------

def test_ebr_roundtrip_for_the_cubic_s_orbital_band():
    """The simple-cubic ``s`` band is exactly one ``A1g@1a`` column: an atomic limit.

    ``Pm-3m``, with no padding and the orbital at ``1a`` (``m-3m``); the band
    reads ``GM1+ X1+ M1+ R1+`` and decomposes as one EBR column.  The rows were
    carried into the physical convention at the adapter boundary, so the
    ``labels_physics_convention`` the convention record publishes are that
    transport's output and ``labels_bcs`` the shipped naming it started from --
    one entry each, in the same row order.
    """
    m = cubic()
    g = _spacegroup._group_of(m, include_TR=False)
    assert (g.sg_number, g.name, g.size) == (221, "Pm-3m", 48)
    assert g.vacuum_dirs == () and g.periodic_dirs == (0, 1, 2)
    asg = _tables._assign_orbitals(m, g)
    assert asg.wyckoff_spglib == ("a",) and asg.site_symmetry == ("m-3m",)

    counts = {}
    for name, k in G.high_symmetry_points(g).items():
        out = G.identify_from_characters(g, k, bloch_characters(m, g, asg, k, [[0]]),
                                         multiplet_dims=[1])
        assert out, (name, out)
        for lab, v in out.labels[0].items():
            counts[lab] = counts.get(lab, 0) + v
    assert counts == {"GM1+": 1, "X1+": 1, "M1+": 1, "R1+": 1}

    data = G.ebr_data(g)
    assert data and data["matrix"].shape == (len(data["labels_physics_convention"]),
                                             len(data["ebrs"]))
    transport = data["convention"]
    assert transport["applied"] is True
    assert transport["name"] == "physical (translations as e^{-ik.t})"
    assert transport["labels_physics_convention"] == \
        tuple(data["labels_physics_convention"])
    assert len(transport["labels_bcs"]) == len(data["labels_physics_convention"])

    dec = G.decompose_ebrs(g, counts)
    assert dec and dec["classification"] == "atomic limit"
    assert dec["stable_nontrivial"] is False and dec["is_positive"] is True
    assert any("A1g" in s and "1a" in s for s in dec["decomposition_strings"])
    # the measured symmetry vector is literally one EBR column
    col = np.asarray(dec["solutions"][0])
    assert col.sum() == 1 and np.array_equal(data["matrix"] @ col, dec["y"])


def test_decompose_ebrs_refuses_a_vector_that_misses_one_of_the_groups_points():
    """A point nobody mentioned is read as *zero content there*, not as unknown.

    ``create_symmetry_vector`` fills a row the caller said nothing about with
    zero, and a zero row is a constraint like any other -- so the same solve
    runs, on a different band set, and answers confidently.  The band above
    decomposes as one ``A1g@1a`` column; drop ``R`` from its counts, which is
    exactly the situation a slab is in for the out-of-plane rows, and the
    decomposition is refused instead of served.  ``Pm-3m``'s EBR basis is
    written over ``GM``, ``M``, ``R``, ``X``, and the refusal names both the
    point that is missing and the four the basis needs.

    ``unused`` is the other direction and stays an answer rather than a
    refusal: a label the basis does not carry says nothing about completeness,
    and is reported back.
    """
    m = cubic()
    g = _spacegroup._group_of(m, include_TR=False)
    counts = {"GM1+": 1, "X1+": 1, "M1+": 1, "R1+": 1}
    assert G.decompose_ebrs(g, counts)["classification"] == "atomic limit"

    partial = {lab: v for lab, v in counts.items() if lab != "R1+"}
    out = G.decompose_ebrs(g, partial)
    assert isinstance(out, Refusal) and not out
    assert out.gate == "incomplete_content"
    assert (out.value, out.threshold) == (1.0, 0.0)
    assert out.context["missing_points"] == ("R",)
    assert out.context["points"] == ("GM", "M", "R", "X")
    assert "confident wrong answer" in out.reason and "decompose" in out.reason

    # a label outside the basis is not incompleteness: every point is still
    # carried, and the stray label comes back in `unused`
    stray = G.decompose_ebrs(g, dict(counts, ZZ1=1))
    assert stray and stray["unused"] == ["ZZ1"]


@pytest.mark.slow
def test_a_grey_group_with_no_named_form_decomposes_to_the_empty_indicator_sum(p41_substrate):
    """``{}`` is "nothing to sum here"; ``None`` is "the forms are not served".

    ``P4mm1' #99.164`` -- a square layer read on a substrate, the SRSI paper's
    flat-band model here -- is grey, so
    :func:`indicator_group` serves its named forms; ``irreptables`` simply
    ships none for it, and its indicator group is trivial besides.  The sum over
    an empty list of forms is the empty mapping, and ``decompose_ebrs`` serves
    it as one: a caller writing ``for name, value in dec["indicators"].items()``
    is right about this group and would be wrong only where the forms were
    withheld, which is the other value.

    The vector is an EBR column of the group's own basis, so it covers every
    high-symmetry point the basis is written over and the completeness gate has
    nothing to refuse -- and being a band representation it is an atomic limit,
    on which every indicator is zero anyway.
    """
    r = p41_substrate                   # the session's shared record of that reading
    assert r, str(r)
    layer = r.group
    assert (layer.number_str, layer.name) == ("99.164", "P4mm1'")
    assert layer.magnetic is True, "grey, so the named forms are served, not withheld"

    ind = G.indicator_group(layer)
    assert ind["withheld"] is False and ind["reason"] is None
    assert ind["indicators"] == {} and tuple(ind["elementary_divisors"]) == ()

    data = G.ebr_data(layer)
    assert data, str(data)
    column = {lab: int(v) for lab, v in zip(data["labels_physics_convention"],
                                            data["matrix"][:, 0]) if v}
    dec = G.decompose_ebrs(layer, column)
    assert dec and dec["classification"] == "atomic limit"
    assert dec["indicators"] == {} and dec["indicators"] is not None
    assert dec["indicators_note"] == ind["note"]
    assert "ships no named indicator formula" in dec["indicators_note"]


def test_indicator_group_serves_the_smith_structure_and_the_grey_forms():
    """Two objects, one derived here and one looked up, and only one is served.

    ``elementary_divisors`` is the Smith form of *this* group's own transported
    EBR matrix, so it is served for every group.  ``indicators`` is
    ``irreptables``' list of named literature forms, and it is served only where
    the table has been audited -- the grey (type-II) entries.

    ``P4/mmm`` double-valued has the indicator group ``Z2 x Z4 x Z8`` whether or
    not time reversal is adjoined -- the divisors do not care -- and single-valued
    ``Pm-3m`` has none at all, which is an answer and not a failure.

    ``orders`` is the order of each named form, so it is populated exactly where
    ``indicators`` is: measured, ``{}`` for both unitary groups and seven
    entries for the grey one.  Asserting "every order is an integer > 1" across
    the loop would be a statement about one group of the three and vacuously
    true of the other two, so the emptiness is asserted where it belongs and
    the orders are checked where there is something to check.

    ``withheld`` is what tells the two readings of an empty ``indicators``
    apart -- "the forms exist and are not served here" against "they were
    served and this group has none" -- and it is a flag rather than a phrase to
    grep the ``note`` for.  ``reason`` carries the withholding text and is
    ``None`` wherever nothing is withheld.
    """
    spinless = _spacegroup._group_of(cubic(), include_TR=False)
    spinful = _spacegroup._group_of(square(spinful=True), include_TR=False)
    grey = _spacegroup._group_of(square(spinful=True), include_TR=True)
    for g in (spinless, spinful, grey):
        ind = G.indicator_group(g)
        assert not isinstance(ind, Refusal), ind
        assert set(ind) == {"indicators", "orders", "elementary_divisors",
                            "withheld", "reason", "note"}
        assert (ind["orders"] == {}) == (ind["indicators"] == {}), (
            "an order is served with its form and withheld with it")
        assert set(ind["orders"]) == set(ind["indicators"])

    # withheld from the two unitary groups, served to the grey one
    for g in (spinless, spinful):
        ind = G.indicator_group(g)
        assert ind["withheld"] is True
        assert isinstance(ind["reason"], str) and ind["reason"]
        assert ind["reason"] == ind["note"], "the reason is the note on this branch"
    served = G.indicator_group(grey)
    assert served["withheld"] is False and served["reason"] is None
    assert served["indicators"], "and there is something to serve"

    grey_orders = G.indicator_group(grey)["orders"]
    assert len(grey_orders) == 7
    assert all(float(v).is_integer() and v > 1 for v in grey_orders.values())
    assert sorted(int(v) for v in grey_orders.values()) == [2, 2, 2, 2, 4, 4, 8]

    assert tuple(int(d) for d in G.indicator_group(spinful)["elementary_divisors"]) \
        == (2, 4, 8)
    assert tuple(int(d) for d in G.indicator_group(grey)["elementary_divisors"]) \
        == (2, 4, 8)
    assert tuple(G.indicator_group(spinless)["elementary_divisors"]) == ()

    # the named forms: served for the grey group, withheld from the unitary one
    assert set(G.indicator_group(grey)["indicators"]) == {
        "z2", "z2w,1", "z2w,2", "z2w,3", "z4", "z4m,π", "z8"}
    assert G.indicator_group(spinful)["indicators"] == {}
    assert "grey table keyed by the bare space-group number" \
        in G.indicator_group(spinful)["note"]
    assert G.indicator_group(spinless)["indicators"] == {}


def test_indicator_group_serves_a_group_whose_shipped_smith_r_is_flat():
    """One EBR column: ``irreptables`` ships ``R`` as a bare vector, not a matrix.

    Where a group has a single EBR column the shipped Smith factor ``R`` is an
    ``(n_irrep,)`` array, and ``diagonal()`` is not defined on it.  ``P1`` -- a
    triclinic cell with nothing but translations -- is the smallest such case:
    eight irrep rows, one column, ``R = [1, 0, ..., 0]``, so the vector is
    already the diagonal padded with the empty rows.
    :func:`ebr._smith_system` normalizes that shape; ``indicator_group`` now
    carries the same guard, where it used to raise ``ValueError`` on the way to
    an answer.  And the answer is that ``P1`` has no indicator at all -- with
    one atomic limit and no nontrivial divisor there is nothing to be
    nontrivial about, which is a verdict and not a failure.
    """
    lat = Lattice([[1.0, 0.0, 0.0], [0.11, 1.31, 0.0], [0.05, 0.19, 1.73]],
                  [[0.13, 0.27, 0.41], [0.61, 0.09, 0.77]], periodic_dirs=[0, 1, 2])
    m = TBModel(lattice=lat, spinful=False)
    m.set_onsite([0.0, 0.3])
    m.set_hop(0.11, 0, 1, [0, 0, 0], mode="add")
    m.set_hop(0.07, 0, 0, [1, 0, 0], mode="add")
    m.set_hop(0.05, 1, 1, [0, 0, 1], mode="add")

    g = _spacegroup._group_of(m, spinor=False, include_TR=False, typat=[0, 1])
    assert g and (g.number_str, g.name) == ("1", "P1")

    data = G.ebr_data(g)
    assert data and data["matrix"].shape == (8, 1)
    r = np.asarray(data["smith"][1])
    assert r.ndim == 1 and r.tolist() == [1, 0, 0, 0, 0, 0, 0, 0]

    ind = G.indicator_group(g)
    assert not isinstance(ind, Refusal), ind
    assert set(ind) == {"indicators", "orders", "elementary_divisors",
                        "withheld", "reason", "note"}
    assert tuple(int(d) for d in ind["elementary_divisors"]) == ()
    assert ind["indicators"] == {} and ind["orders"] == {}
    assert ind["withheld"] is True and ind["reason"]


@pytest.mark.slow
def test_the_bare_key_indicator_forms_are_withheld_from_a_unitary_group():
    """A5.1 audit finding (ii), fixed: the passthrough no longer partial-sums.

    ``irreptables``' ``single_indicators.json`` / ``double_indicators.json`` are
    the **grey** tables keyed by the bare space-group number, byte for byte, so
    their ``factors`` are grey corep labels.  ``load_si_table(magnetic=False)``
    hands them out for a unitary group anyway, and summing one over that group's
    own labels drops every factor whose label got glued -- a silent partial sum.

    Measured here on double-valued ``R-3m`` (``#166``), which is grey ``#166.98``
    -- the Fu-Kane-Mele example's group -- with time reversal dropped.  All four
    shipped forms are written over twelve labels, of which eight
    (``-GM4GM5``, ``-F3F4``, ...) are grey coreps the unitary group does not
    have; none of the four vanishes on the group's own EBRs (``z2w,i`` fail on
    12 of its 20 EBR columns, ``z4`` on 6), and the old sum reported
    ``z2w,i = 1.5`` and ``z4 = 3`` on *band representations* -- non-integers, on
    objects every indicator is zero on by construction.

    The fix keeps what is valid and drops what is not: the Smith-derived
    structure ``Z2`` is served, the named forms are not, and the note says why.
    ``symmetry_indicator`` is where a value comes from.
    """
    from irreptables.symmetry_indicators import load_si_table

    unitary = _spacegroup._group_of(rhombohedral(), include_TR=False)
    assert unitary and (unitary.number_str, unitary.name) == ("166", "R-3m")
    grey = _spacegroup._group_of(rhombohedral(), include_TR=True)
    assert grey and grey.number_str == "166.98"

    data = G.ebr_data(unitary)
    basis = set(data["convention"]["labels_bcs"])
    shipped = load_si_table(spinor=True, magnetic=False)["166"]["indicators"]
    assert set(shipped) == {"z2w,1", "z2w,2", "z2w,3", "z4"}
    for name, spec in shipped.items():
        missing = sorted(set(spec["factors"]) - basis)
        assert len(missing) == 8, (name, missing)      # the glued grey coreps
        assert "-GM4GM5" in missing and "-F3F4" in missing
        # ... and the truncated form is not an indicator of this group at all
        coef = [float(spec["factors"].get(lab, 0.0))
                for lab in data["convention"]["labels_bcs"]]
        mod = int(round(float(spec["mod"])))
        on_ebrs = data["matrix"].T @ np.array(coef)
        failed = [v for v in on_ebrs
                  if abs(v - round(v)) > 1e-9 or int(round(v)) % mod]
        assert len(failed) == (6 if name == "z4" else 12), (name, len(failed))

    served = G.indicator_group(unitary)
    assert served["indicators"] == {} and served["orders"] == {}
    assert tuple(int(d) for d in served["elementary_divisors"]) == (2,)
    assert served["withheld"] is True, "and it says so as a flag, not only in prose"
    assert "silent partial sum" in served["reason"]
    assert "symmetry_indicator" in served["reason"]
    # the grey partner is untouched: same file, audited key, forms served
    assert set(G.indicator_group(grey)["indicators"]) == set(shipped)
    assert G.indicator_group(grey)["withheld"] is False
    assert G.indicator_group(grey)["reason"] is None
    assert tuple(int(d) for d in G.indicator_group(grey)["elementary_divisors"]) \
        == (2, 4)

    # and the same withholding downstream, where the sum was actually taken.
    # Every EBR column IS an atomic limit, so a true indicator is zero on each
    # of them; the old passthrough came back nonzero on 12 of the 20, at
    # half-integer values.
    matrix, labels = data["matrix"], data["labels_physics_convention"]
    columns = [{lab: int(v) for lab, v in zip(labels, matrix[:, col]) if v}
               for col in range(matrix.shape[1])]
    old = [{name: sum(v * counts.get(lab, 0) for lab, v in spec["factors"].items())
            % spec["mod"] for name, spec in shipped.items()} for counts in columns]
    assert sum(1 for values in old if any(values.values())) == 12
    assert {0.5, 1.5} <= {v for values in old for v in values.values()}

    col = next(i for i, values in enumerate(old) if any(values.values()))
    dec = G.decompose_ebrs(unitary, columns[col])
    assert dec and dec["classification"] == "atomic limit"
    # withheld is served as `None` -- no sum was taken, which is not the empty
    # sum a group with no form of its own gets
    assert dec["indicators"] is None
    assert G.indicator_group(unitary)["withheld"] is True
    assert dec["indicators_note"] == G.indicator_group(unitary)["reason"]


# ---------------------------------------------------------------------------
# the tabulated points a presentation has no k for
# ---------------------------------------------------------------------------

def test_high_symmetry_points_names_the_tabulated_points_out_of_the_plane():
    """The mapping is a dict, and it says which tabulated points it dropped.

    A slab's zone has no ``k_z``, so the ``A``, ``R`` and ``Z`` of the square
    layer's ``P4/mmm1'`` table are not points of it at all -- and a caller
    summing a symmetry vector over this mapping is reading a *partial* one
    whether or not anything tells it.  ``missing`` is the telling: the
    tabulated names out of the periodic plane, disjoint from the keys by
    construction, and empty for a fully periodic model, whose zone carries the
    whole table.

    Everything else about the return value is a ``dict``'s -- iteration,
    ``in``, ``len``, indexing, equality with a plain dict -- which is what
    every other call site in this suite and in the examples relies on.
    """
    layer = _spacegroup._group_of(square(spinful=True))
    hsp = G.high_symmetry_points(layer)
    assert sorted(hsp) == ["GM", "M", "X"]
    assert hsp.missing == ("A", "R", "Z")
    assert not set(hsp.missing) & set(hsp), "a point is served or missing, never both"

    # still a mapping, and still the same one
    assert isinstance(hsp, dict) and len(hsp) == 3 and "M" in hsp
    assert hsp == {name: hsp[name] for name in hsp}
    assert all(np.asarray(k).shape == (2,) for k in hsp.values())

    # a fully periodic cell has a k for every tabulated point
    solid = G.high_symmetry_points(_spacegroup._group_of(cubic(), include_TR=False))
    assert sorted(solid) == ["GM", "M", "R", "X"] and solid.missing == ()


# ---------------------------------------------------------------------------
# meshes
# ---------------------------------------------------------------------------

def test_k_grid_reports_which_high_symmetry_points_it_contains():
    g = _spacegroup._group_of(graphene(0.0, -1.0), include_TR=False)
    six = G.k_grid(g, 6)
    seven = G.k_grid(g, 7)
    eight = G.k_grid(g, 8)
    assert six.contains == {"GM": True, "K": True, "M": True} and six.missing == ()
    assert seven.contains["K"] is False and seven.contains["M"] is False
    assert eight.contains["M"] is True and eight.contains["K"] is False
    assert set(eight.missing) == {"K"}
    assert repr(eight) == "KGrid(8x8x1 mesh, symmetric; samples GM, M; misses K)"
    assert six.points.shape == (36, 2) and six.irreducible.shape[1] == 2
    assert len(six.irreducible) < len(six.points)
    assert six.divisions == (6, 6, 1)


def test_k_grid_returns_a_pythtb_mesh_the_membership_record_describes():
    """A3r: the mesh currency is pythtb's, and ``.points`` is derived from it.

    ``k_grid`` builds its Monkhorst-Pack grid through ``Mesh.build_grid`` rather
    than a hand-rolled ``meshgrid``, so what comes back can be handed straight
    to a ``WFArray`` (or to ``sra.spin_resolved_wfarrays``).  Three things are
    locked: it really is a pythtb ``Mesh``; its points are exactly the
    ``linspace(0, 1, N, endpoint=False)`` hypercube the membership
    record was computed against; and solving a model on it reproduces a direct
    ``solve_ham`` at the high-symmetry point the record says the mesh contains.
    """
    from pythtb import Mesh, WFArray

    model = graphene(0.0, -1.0)
    g = _spacegroup._group_of(model, include_TR=False)
    grid = G.k_grid(g, 6)

    assert isinstance(grid.mesh, Mesh)
    assert grid.mesh.dim_k == 2 and tuple(grid.mesh.shape_axes) == (6, 6)
    assert grid.mesh.is_grid and grid.mesh.filled

    # the point set is pythtb's own hypercube, and `.points` is a view on it
    axes = [np.linspace(0.0, 1.0, 6, endpoint=False)] * 2
    expected = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 2)
    assert np.array_equal(grid.points, expected)
    assert np.array_equal(grid.points, np.asarray(grid.mesh.flat))
    assert not hasattr(type(grid), "__dataclass_fields__") or \
        "points" not in type(grid).__dataclass_fields__, "points is derived, not stored"

    # the k-axes are marked BZ-winding, which is what makes a WFArray on this
    # mesh get the periodic gauge right without any help from tqc
    assert len(grid.mesh.bz_winding_axes) == 2

    # and the mesh is usable: the M-point fiber of a WFArray solved on it is the
    # same diagonalization a direct solve_ham gives
    wf = WFArray(model.lattice, grid.mesh, spinful=False)
    wf.solve_model(model)
    kM = grid.hsp["M"]
    assert grid.contains["M"]
    j = int(np.argmin(np.abs(grid.points - kM).max(axis=1)))
    idx = np.unravel_index(j, wf.shape_mesh)
    evals = model.solve_ham(k_pts=np.atleast_2d(kM))
    assert np.abs(np.asarray(wf.energies)[idx] - evals).max() < 1e-12


def test_k_grid_refuses_a_grid_the_group_does_not_respect():
    """A 4x6 mesh cannot respect a four-fold rotation that exchanges the two axes."""
    g = _spacegroup._group_of(bilayer_slab(), include_TR=False)
    assert g.sg_number == 123
    r = G.k_grid(g, [4, 6])
    assert not r and r.gate == "asymmetric_grid"
    assert r.context["grid"] == (4, 6, 1)

    forced = G.k_grid(g, [4, 6], allow_asymmetric=True)
    assert forced.symmetric is False and forced.irreducible is None
    assert forced.points.shape == (24, 2)


# ---------------------------------------------------------------------------
# (b) the symmetry audit: the cell allows more than the Hamiltonian obeys
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_validate_group_handles_antiunitary_operations():
    """The conjugated solve, on a model where conjugating changes the answer.

    ``square(tx=-1.0, ty=-0.6)`` cannot carry this claim, which is what this
    test used to be asked on: one band and real hoppings make ``H(k)`` a real
    1x1, so ``H(k)* == H(k)`` bit for bit and the antiunitary equation
    ``X H(k)* = H(Rk) X`` *is* the unitary one -- the conjugation could be
    deleted from the library without moving a number.  Kane-Mele's spin-orbit
    hoppings are genuinely complex: ``max|H(k)* - H(k)| = 4.62`` on the
    detection grid, measured below.

    The flat cell is the grey ``P6/mmm1'``, 24 unitary + 24 antiunitary
    operations, and the Hamiltonian obeys all 48 (residuals 1.8e-14, 27 ulp of
    ``max|H| = 3``).  A **uniform** Zeeman term ``h s_z`` is then added.  It is
    an axial vector along z: a spatial operation sends an axial vector ``a`` to
    ``det(R) R a``, and time reversal flips it, so the term survives exactly
    the operations with ``det(R)·R_zz = +1`` (unitary) or ``-1`` (antiunitary).
    That rule is read off each operation's own rotation matrix, never off the
    detector, and it names 24 survivors -- 12 unitary and 12 antiunitary, a
    type-III magnetic group.  ``_validate_group`` returns exactly those.

    Those 12 antiunitary survivors are what the conjugation buys.  Every
    antiunitary k-action here is ``-R`` for a unitary ``R`` that is also in the
    group, so solving the *unconjugated* equation for the antiunitary
    operations returns their exact complement (measured): the count survives,
    the names do not.  That is why the verdict is pinned by name and not by
    ``len``.
    """
    from pythtb.models import kane_mele

    def km():
        return kane_mele(0.0, -1.0, 0.3, 0.0)

    m = km()
    g = _spacegroup._group_of(m)
    assert (g.number_str, g.size) == ("191.234", 48)
    assert sum(1 for s in g.symmetries if s.time_reversal) == 24

    H = np.asarray(m.hamiltonian(k_pts=_measure._detection_grid(int(g.dim_k), 6),
                                 flatten_spin_axis=True))
    hmax = float(np.abs(H).max())
    assert np.abs(H.conj() - H).max() > 1.0, "conjugation is not the identity on this H"

    v = _measure._validate_group(m, g, nk=6)
    assert len(v.indices) == 48 and v.closed        # P6/mmm1' = 24 unitary + 24 antiunitary
    # the kept operations as `operations(group)` writes them, keyed on the index
    assert isinstance(v.operations, G.OperationTable) and list(v.operations) == \
        sorted(v.indices)
    assert sum(1 for s in v.operations.values() if s.time_reversal) == 24
    assert max(v.residuals[i] for i in v.indices) <= 256 * _ULP * hmax

    # h s_z on both sublattices: TR-odd, and odd under every operation that
    # reverses the z axis in the axial-vector sense.
    mz = km()
    mz.set_onsite([[0, 0, 0, 0.3], [0, 0, 0, 0.3]], mode="add")
    expect = {int(s.ind) for s in g.symmetries
              if np.linalg.det(np.asarray(s.rotation, dtype=float))
              * float(np.asarray(s.rotation, dtype=float)[2, 2])
              * (-1.0 if s.time_reversal else 1.0) > 0.0}
    assert len(expect) == 24 and len(expect & {int(s.ind) for s in g.symmetries
                                               if s.time_reversal}) == 12

    vz = _measure._validate_group(mz, g, nk=6)
    assert set(vz.indices) == expect, "the axial-vector rule names the survivors"
    assert vz.closed and len(vz.indices) == 24
    assert sum(1 for s in vz.operations.values() if s.time_reversal) == 12
    assert all(d["refuted_by"] == "no symmetry matrix" for d in vz.discrepancies)
    assert max(vz.residuals[i] for i in vz.indices) <= 256 * _ULP * hmax


@pytest.mark.slow
def test_the_validation_record_prints_what_a_survivor_count_alone_would_hide():
    """The one-line record says "obeyed", and on a layer says how much that means.

    "48 of 48 obeyed" is the sentence a slab's validation invites and
    the sentence its own docstring warns against: a class that survives proves
    only that *some* operator with that k-action is a symmetry, and on a flat
    cell every ``sigma_h``-composed operation shares its in-plane k-action with
    an in-plane partner.  So the count of survivors carrying an inherited
    verdict is printed beside the total, in the physics that makes it matter --
    which of each pair holds, the representation decides.

    A fully periodic primitive cell has singleton k-action classes throughout,
    so there is nothing to qualify and the line stays clean.
    """
    from pythtb.models import kane_mele

    layer = _layers._slab_presentation(kane_mele(0.0, -1.0, 0.3, 0.25))
    g2 = _spacegroup._group_of(layer)
    v2 = _measure._validate_group(layer, g2, nk=6)
    assert g2.number_str == "191.234" and len(v2.indices) == 48 == g2.size
    assert len(v2.undetermined) == 48, "every class on a flat cell has a partner"

    line = repr(v2)
    assert line.startswith("ValidatedGroup(") and line.endswith(")")
    assert "P6/mmm1' #191.234" in line
    assert ("48 of 48 coset representatives of the lattice translations obeyed, which "
            "form a group; they are the single-valued group's, each standing for itself "
            "and its negative in the double group") in line
    assert "48 act on every k as another does" in line
    assert "the representation decides" in line
    assert "SymmetryOperation" not in line and len(line) < 360

    # ... and a fully periodic primitive cell has nothing to qualify: the
    # nonsymmorphic diamond obeys all 48 operations of its cell, to roundoff
    m3 = diamond()
    v3 = _measure._validate_group(m3, _spacegroup._group_of(m3, include_TR=False), nk=4)
    assert len(v3.indices) == 48 and v3.discrepancies == () and v3.closed
    assert max(v3.residuals.values()) < 1e-13
    assert v3.undetermined == (), \
        "every k-action class of a 3D primitive cell is a singleton"
    assert repr(v3).endswith(
        "48 of 48 coset representatives of the lattice translations obeyed, which form a "
        "group)")


def test_find_intertwiners_zero_nullspace_is_a_proof():
    """The verdict, on the default solver and on the oracle that audits it.

    The ``method``/``n_kpts_solve`` bookkeeping differs between the two by
    construction -- the stacked route stacks all 64 points, the eigen route
    parametrizes at one and sifts -- so each is asserted of its own solver
    (2026-09-02).  The *verdict* is asserted of both, and they agree.
    """
    m = square(tx=-1.0, ty=-0.6)
    k = np.stack(np.meshgrid(*([np.linspace(0, 1, 8, endpoint=False)] * 2),
                             indexing="ij"), axis=-1).reshape(-1, 2)
    for solver in ("eigen", "stacked"):
        c4 = _detector._find_intertwiners(m, np.array([[0.0, -1.0], [1.0, 0.0]]), k, solver=solver)
        assert c4.dim == 0
        assert c4.singular_values[0] / c4.scale > 1e-3, "the breaking is far above the cut"

        c2 = _detector._find_intertwiners(m, -np.eye(2), k, solver=solver)
        assert c2.dim == 1
        assert c2.residuals[0] < 1e-13 and c2.unitarity[0] < 1e-13
        assert c2.n_kpts == c2.n_kpts_solve == 64

    assert _detector._find_intertwiners(m, -np.eye(2), k).method == "eigen"
    assert _detector._find_intertwiners(m, -np.eye(2), k, solver="stacked").method == "stacked"

    with pytest.raises(ValueError, match="at least two k-points"):
        _detector._find_intertwiners(m, -np.eye(2), np.zeros((1, 2)))
    with pytest.raises(ValueError, match="solver must be"):
        _detector._find_intertwiners(m, -np.eye(2), k, solver="gram")


def test_find_intertwiners_subset_solve_moves_the_solve_not_the_verdict():
    """The size cap cuts the *solve*, never the precision and never the gate.

    A grid whose stacked constraint matrix would be too large is solved on a
    deterministic subset of its own points -- still one full-precision SVD --
    and every candidate is then re-measured against **all** of them, so the
    capped and uncapped routes agree on ``dim`` and both report a full-grid
    residual.  ``singular_values[0]/scale`` reads machine zero on a true
    nullvector in either, which is the property the normal-equation route this
    replaces did not have: squaring the matrix and square-rooting the
    eigenvalues floors a true nullvector at ``sqrt(eps) ~ 1e-8``, exactly the
    default ``tol``, so it could rule ``dim == 0`` on an exact symmetry.

    ``max_solve_entries`` and the subset are the **stacked** solver's own
    bookkeeping -- the eigen route has no stack to cap -- so this asks for that
    solver by name (2026-09-02).  The default route is measured beside it: it
    reaches the same ``dim`` from 5 of the 64 points.

    The two full-grid covariance residuals are bounded in ulp of ``max |H|``,
    the scale they are proportional to: what is asserted is that a nullvector of
    the stacked operator covaries to roundoff on all 64 points, and *how many*
    ulp the SVD and the 64 matrix products cost is the BLAS's and libm's
    business.  Measured here: ``2.6e-14`` and ``2.7e-14``, some 30 ulp of
    ``max |H| = 4.0``.
    """
    m = diamond()
    k = np.stack(np.meshgrid(*([np.linspace(0, 1, 4, endpoint=False)] * 3),
                             indexing="ij"), axis=-1).reshape(-1, 3)
    action = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])   # a C3 on reduced k
    a = _detector._find_intertwiners(m, action, k, solver="stacked")
    b = _detector._find_intertwiners(m, action, k, max_solve_entries=8 * m.nstate ** 4,
                             solver="stacked")
    assert a.method == "stacked" and b.method == "stacked_subset"
    assert (a.n_kpts, a.n_kpts_solve) == (64, 64)
    assert (b.n_kpts, b.n_kpts_solve) == (64, 8), "the cap is per k-point: 8 * nstate**4"
    assert a.dim == b.dim == 1
    h_scale = float(np.abs(np.asarray(
        m.hamiltonian(k_pts=k, flatten_spin_axis=True))).max())
    assert max(a.residuals[0], b.residuals[0]) <= 128 * _ULP * h_scale, \
        "both residuals are measured on all 64 points"
    assert max(a.singular_values[0] / a.scale, b.singular_values[0] / b.scale) < 1e-14

    # the same economy, one layer up: the eigen route stops after 5 points and
    # verifies on all 64 -- and lands on the same dimension and the same gate
    e = _detector._find_intertwiners(m, action, k)
    assert e.method == "eigen" and e.dim == 1 and (e.n_kpts, e.n_kpts_solve) == (64, 5)
    assert e.basis_size == (2, 1, 1, 1, 1) and e.window == 1e-3
    assert e.residuals[0] < 1e-13 and e.singular_values[0] / e.scale < 1e-14


# ---------------------------------------------------------------------------
# (b') the vacuous constraint -- ``dim == 0`` is a proof, and this was not one
# ---------------------------------------------------------------------------
#
# ``X H(k) - H(Rk) X`` is the **zero operator** whenever ``H(k)`` and ``H(Rk)``
# are the same *scalar* matrix.  A one-band model does that at every operation
# its ``h(k)`` obeys; so does any ``H(k) = c(k) * 1`` at any number of states.
# The nullspace cut used to be purely relative -- ``s <= tol * s_max`` -- so on
# a vanished operator it took the operator's own rounding noise for a scale,
# admitted nothing, and returned ``dim == 0``: the record's documented *proof*
# of breaking, handed down on operations obeyed to ``1.1e-16``.  Measured
# before the fix: the one-orbital cubic kept 32 of 96, the one-orbital
# tetragonal 16 of 32.  Fixed 2026-08-28 by testing vacuity *absolutely*,
# against ``max |H|`` over the grid.
#
# Each test asserts the cure on both solvers: the eigen route, and the stacked
# operator the defect lived in.


def _grid(n, dim_k):
    """Gamma-centred ``n**dim_k`` mesh, as the neighbouring tests build it."""
    return np.stack(np.meshgrid(*([np.linspace(0, 1, n, endpoint=False)] * dim_k),
                                indexing="ij"), axis=-1).reshape(-1, dim_k)


def _census(model, group, k, **kw):
    """``(space, symmetry)`` for every operation of ``group`` the model can act."""
    out = []
    for sym in group.symmetries:
        action = group.k_action_of(sym)
        assert action is not None, "a fully periodic cell moves no k out of plane"
        out.append((_detector._find_intertwiners(model, action, k,
                                         antiunitary=bool(sym.time_reversal), **kw), sym))
    return out


@pytest.mark.slow
def test_the_same_false_proof_on_the_one_orbital_tetragonal():
    """32 of 32, where the relative cut was fooled -- the measured case.

    Every operation of ``P4/mmm1'`` is obeyed by the one-band ``h(k)`` exactly
    -- entrywise rounding and nothing else, measured here at
    ``max_k |H(k) - H(Rk)| = 4.4e-16``, under half an ulp of ``max |H| = 4.8``.
    That number is proportional to ``max |H|`` and to nothing else, so it is
    what the bounds below are written against: scale the hoppings and the
    residual scales with them.  The constraint operator is ``(h(k) - h(Rk)) *
    X``, so it vanishes, so *every* ``X`` solves and the honest answer is the
    whole matrix space: ``dim = nstate**2``.  What the relative cut returned
    instead was ``dim = 0`` on half the group, a **proof** of breaking issued
    against a Hamiltonian that obeys the operation to the last bit -- and the
    ones that survived did so only because their constraint matrix came out
    bit-exactly zero and took the ``else`` branch.  *Which* operations those are
    is a fact about floating-point summation -- a rotation reassociates the
    three-term Bloch sum, and whether that costs a bit is the machine's
    arithmetic (16 of 32 on the development machine; the one-orbital cubic
    measured 32 of 96 the same way) -- so what is asserted is the cure: this
    cut is fooled on no operation, on either solver.
    The identity is served first, so the polar extraction starts there, and one
    band is degenerate at every point, so the eigen route scans the whole grid.
    """
    m = tetragonal()
    g = _spacegroup._group_of(m, include_TR=True)
    assert g.sg_number == 123 and g.size == 32 and m.nstate == 1
    k = _grid(4, 3)
    h_scale = float(np.abs(np.asarray(
        m.hamiltonian(k_pts=k, flatten_spin_axis=True))).max())

    v = _measure._validate_group(m, g, nk=4)
    assert len(v.indices) == 32 and v.discrepancies == () and v.closed
    assert max(v.residuals.values()) <= _ROUNDOFF * h_scale

    seen = _census(m, g, k)
    assert all(sp.vacuous and sp.dim == 1 for sp, _ in seen)
    assert all(np.array_equal(sp.basis[0], np.eye(m.nstate)) for sp, _ in seen), \
        "the identity is served first, so the polar extraction starts there"
    assert all(sp.method == "eigen" and sp.n_kpts_solve == 64 for sp, _ in seen), \
        "one band is degenerate at every point, so the eigen route scans the grid"
    old = _census(m, g, k, solver="stacked")               # the old cut's own object
    assert all(sp.vacuous and sp.dim == 1 for sp, _ in old)


@pytest.mark.slow
def test_vacuity_is_about_scalar_H_not_about_one_band():
    """A spinful model with spin-independent hoppings is vacuous at nstate = 2.

    Nothing in the test references the number of states, and nothing should:
    ``H(k) = h(k) * 1_2`` here, so the constraint operator is
    ``(h(k) - h(Rk)) * X`` on the whole four-dimensional ``M_2`` and vanishes
    just as it did at one band.  The solution space is *all* of ``M_2`` --
    ``dim = 4`` -- because full ``SU(2)`` really is conserved: this Hamiltonian
    commutes with every ``2x2`` unitary there is.  Measured on the spinful
    square with its diagonal hoppings, whose ``h(k)`` has four summands that a
    rotated ``k`` may reassociate.  Where that arithmetic hands the stacked
    operator rounding noise, the old relative cut refused it identically, and
    for the identical wrong reason -- it kept 4 of 16 on the development
    machine; the verdict is asserted on both solvers.

    The identity's residual closes it.  Its candidate is the served ``1``, and
    the measured number is ``max_k |h(k) - h(k)|`` through one polar extraction:
    bit-exactly ``0.0`` on this machine, because this LAPACK's SVD of ``1`` --
    whose singular spectrum is completely degenerate, so the basis is arbitrary
    -- returns ``(1, 1)``.  Another driver's equally valid pair would leave a few
    ulp of ``max |H|``, which is what is asserted.
    """
    m = square(spinful=True, td=-0.3)
    g = _spacegroup._group_of(m, include_TR=False)
    assert m.nstate == 2 and m.nspin == 2 and g.size == 16
    k = _grid(8, 2)
    h_scale = float(np.abs(np.asarray(
        m.hamiltonian(k_pts=k, flatten_spin_axis=True))).max())

    v = _measure._validate_group(m, g, nk=8)
    assert len(v.indices) == g.size and v.discrepancies == () and v.closed
    assert max(v.residuals.values()) <= _ROUNDOFF * h_scale
    assert set(v._nullspace_dims.values()) == {4}, "nstate**2, the whole of M_2"

    seen = _census(m, g, k)
    assert all(sp.vacuous and sp.dim == 4 for sp, _ in seen)
    assert all(np.array_equal(sp.basis[0], np.eye(2)) for sp, _ in seen)
    # dim >= 2, so the searched-unitary path runs on top of the served identity
    assert all(len(sp.unitaries) == 5 and sp.search for sp, _ in seen)
    assert all(float(np.min(sp.residuals)) <= _ROUNDOFF * h_scale for sp, _ in seen)
    old = _census(m, g, k, solver="stacked")               # the old cut's own object
    assert all(sp.vacuous and sp.dim == 4 for sp, _ in old)

    # ... and the identity's own residual is the number the caller gates on
    ident = next(sp for sp, sym in seen
                 if np.array_equal(np.asarray(sym.rotation, dtype=int), np.eye(3, dtype=int))
                 and np.allclose(np.asarray(sym.translation) % 1.0, 0.0))
    assert float(ident.residuals[0]) <= _ROUNDOFF * h_scale


def test_a_genuinely_broken_operation_is_not_vacuous_at_any_number_of_states():
    """The other side of the cut: ``dim == 0`` still means what it says.

    Break the same two models along z and the constraint operator stops
    vanishing -- ``s_max / max|H| = 2.0``, nine decades above the ``1e-9``
    vacuity cut and sixteen above the noise the false proofs were read out of.
    So the ``C4``-type operations get their honest ``dim = 0``, and the ones
    the model still obeys are vacuous as before.  Nothing here depends on the
    number of states: the spinful model refuses on the same numbers.
    """
    for m, k in ((square(tx=-1.0, ty=-0.6), _grid(8, 2)),
                 (cubic(spinful=True, tz=-0.4), _grid(4, 3))):
        g = _spacegroup._group_of(m, include_TR=False)
        seen = _census(m, g, k)
        broken = [sp for sp in (s for s, _ in seen) if not sp.vacuous]
        held = [sp for sp in (s for s, _ in seen) if sp.vacuous]
        assert broken and held, f"{m.nstate} states: both kinds must appear"
        assert all(sp.dim == 0 for sp in broken), \
            "a non-vacuous constraint with no nullvector is a proof, unchanged"
        assert all(sp.dim == m.nstate ** 2 for sp in held)

        # the number the absolute test actually looks at, on both sides
        H = np.asarray(m.hamiltonian(k_pts=k, flatten_spin_axis=True))
        h_scale = float(np.abs(H).max())
        assert min(float(sp.singular_values[-1]) for sp in broken) > 1e-2 * h_scale, \
            "measured 2.0 * max|H| -- nine decades above the 1e-9 vacuity cut"
        assert max(float(sp.singular_values[-1]) for sp in held) < 1e-15 * h_scale

        v = _measure._validate_group(m, g, nk=8 if g.dim_k == 2 else 4)
        assert len(v.indices) == len(held) and v.closed
        assert all(d["refuted_by"] == "no symmetry matrix" for d in v.discrepancies)
        assert len(v.discrepancies) == len(broken)


def test_the_isotropic_square_passes_by_the_vacuity_test_not_by_luck():
    """``square()`` survived the old cut by an accident of the arithmetic.

    Its ``h(k) = f(kx) + f(ky)`` has two summands, and every operation of
    ``P4/mmm`` either leaves them alone or swaps them -- ``f`` is even, and
    floating addition commutes exactly -- so the constraint matrices come out
    **bit-exactly** zero here (measured: ``s_max == 0.0`` at all 16, on both
    solvers), ``s_max > 0`` is false, and the old ``else`` branch handed back
    ``scale = 1.0``, under which every singular value passes any cut.  The right
    answer, for a reason that is not a reason.

    Add the two diagonal hoppings and nothing about the symmetry changes --
    still ``P4/mmm``, still obeyed exactly -- but ``h(k)`` now has four
    summands, so a rotated k may *reassociate* the sum rather than merely
    commuting it.  Whether either model's sum actually costs a bit is the
    machine's arithmetic and not the code's -- the development machine measured
    ``s_max = 1.8e-15`` on the four-summand model, an M1 with OpenBLAS 0.3.33
    measures exactly ``0.0`` -- so neither bit-exactness is asserted: what is
    asserted is that both models vanish to roundoff *against the scale the record
    published*.  Where the noise existed, the old relative cut kept **4 of 16**:
    three quarters of an exactly symmetric group refused, on rounding noise, by a
    proof.

    The cure is asserted unconditionally and on both models: the verdict is
    vacuous, 16 of 16, on both solvers, and the ``scale`` the record publishes is
    ``max |H|`` over the grid -- the absolute test's own scale, which is what the
    old route could never report (the operator's rounding noise where the sum
    reassociated, the ``else`` branch's ``1.0`` where it did not).

    The luck and the old cut were both properties of the *stacked* operator, so
    the verdict is measured on ``solver="stacked"`` as well as on the eigen
    route, which reaches it on both models from its own per-point
    ``|h(k) - h(Rk)|``.
    """
    m = square()
    g = _spacegroup._group_of(m, include_TR=False)
    k = _grid(8, 2)
    lucky = _census(m, g, k, solver="stacked")
    assert all(sp.vacuous and sp.dim == 1 for sp, _ in lucky)
    lucky_e = _census(m, g, k)
    assert all(sp.vacuous and sp.dim == 1 for sp, _ in lucky_e)

    m2 = square(td=-0.3)
    g2 = _spacegroup._group_of(m2, include_TR=False)
    assert g2.sg_number == 123 and g2.size == 16, "the diagonals break nothing"
    noisy = _census(m2, g2, k, solver="stacked")
    assert all(sp.vacuous and sp.dim == 1 for sp, _ in noisy)
    noisy_e = _census(m2, g2, k)
    assert all(sp.vacuous and sp.dim == 1 for sp, _ in noisy_e)

    # the cure, with nothing conditional in it: both models reach the vacuity
    # branch through the **absolute** test on both solvers, and the record names
    # the scale they were judged against -- max|H| over the grid, against which
    # the constraint operator and the covariance residual have both vanished
    for model, census in ((m, lucky), (m, lucky_e), (m2, noisy), (m2, noisy_e)):
        h_scale = float(np.abs(np.asarray(
            model.hamiltonian(k_pts=k, flatten_spin_axis=True))).max())
        for sp, sym in census:
            assert sp.scale == pytest.approx(h_scale), (sp.method, sym.rotation)
            assert float(sp.singular_values[-1]) <= _ROUNDOFF * sp.scale
            assert float(np.min(sp.residuals)) <= _ROUNDOFF * sp.scale

    v = _measure._validate_group(m2, g2, nk=8)
    assert len(v.indices) == 16 and v.discrepancies == () and v.closed


def test_a_generic_multi_band_model_never_reaches_the_vacuity_branch():
    """The control: ``H(k)`` not scalar, so nothing about the solve moves.

    Diamond's two-band ``H(k)`` carries an off-diagonal of its own, so the
    constraint operator has singular values of order ``||H||`` whether the
    operation holds or not -- ``s_max / max|H| >= 0.9`` at every one of the 48
    -- and every call takes the untouched relative-cut branch.  This is the
    model the bit-identity check was run against: ``basis``, ``unitaries``,
    ``residuals``, ``singular_values`` and ``scale`` are unchanged to the last
    bit, on the full solve and on the capped one.  A *scalar* multi-band model
    is deliberately not used here -- that one is expected to change.

    What ``scale`` *is* differs between the two solvers by construction, and
    both are measured (2026-09-02).  The stacked route's is the largest
    singular value of the stacked operator, ``8.0 max|H|`` here -- stacking 64
    points multiplies the per-point value by up to ``sqrt(64)`` -- and it is
    also ``singular_values[-1]``, so the old relative cut can be reconstructed
    from the record.  The eigen route's is the *Sylvester operator norm* at the
    deciding point, ``max_ab |E_b - E'_a| = 1.0 max|H|``, which is **not** its
    largest reported singular value (those are the constraint restricted to the
    surviving basis, and at ``dim == 1`` the survivor's own violation is ~0).
    Both are nine decades clear of the ``1e-9`` vacuity cut, which is the
    property this controls.

    The verdict itself is ``dim == 1`` at all 48 -- one intertwiner per
    operation, up to phase -- and it rests on a wide gap, measured here
    (2026-09-09) rather than recomputed from the record's own cut.  The stacked
    route publishes all ``nstate**2 = 4`` singular values, so the first
    *rejected* one can be read off: ``singular_values[dim] / scale ==
    0.6209484`` at every one of the 48 and at the identity, seven decades above
    the ``1e-8`` relative cut, so the bound below is 6x under the measurement.
    The eigen route publishes only the surviving basis (one value at
    ``dim == 1``), so there is no ``singular_values[dim]`` there; what its
    record does say is that the sift *rejected* a direction it was handed --
    ``basis_size`` reads ``(2, 1, 1, 1, 1)``, two matching-energy pairs
    parametrized and one kept.  The independent oracle on the survivor is its
    covariance residual: ``max_k ||U H U+ - H(Rk)||_max``, measured at 31.4 ulp
    of max|H| (stacked) and 0.71 ulp (eigen) against the 64-ulp bound below.

    ``scale`` is *not* checked here against the fallback ``scale = max|H|`` that
    ``_sift`` uses when no stage is informative: on diamond the eigen route's
    Sylvester norm is 4.0 and max|H| is also 4.0, so the measured value and the
    fallback are the same number and no assertion on this fixture can tell them
    apart (verified by mutation, 2026-09-09).  The stacked floor of ``7.9`` is
    a real separation from max|H|; the eigen ``0.99`` is not, and says only that
    the eigen scale is of order ``||H||``.
    """
    m = diamond()
    g = _spacegroup._group_of(m, include_TR=False)
    k = _grid(4, 3)
    H = np.asarray(m.hamiltonian(k_pts=k, flatten_spin_axis=True))
    h_scale = float(np.abs(H).max())
    census = {}
    for solver, floor in (("stacked", 7.9), ("eigen", 0.99)):
        seen = census[solver] = _census(m, g, k, solver=solver)
        assert len(seen) == 48 and m.nstate == 2
        assert not any(sp.vacuous for sp, _ in seen)
        assert all(sp.dim == 1 for sp, _ in seen), "one intertwiner, up to phase"
        assert max(float(sp.residuals[0]) for sp, _ in seen) <= 4 * _ROUNDOFF * h_scale, \
            "and what the cut kept intertwines H(k) with H(Rk) to roundoff"
        assert min(sp.scale for sp, _ in seen) > floor * h_scale, \
            "the constraint has singular values of order ||H||, holding or not"

        # the identity, where the searched path runs, is untouched as well
        ident = _detector._find_intertwiners(m, np.eye(3), k, solver=solver)
        assert not ident.vacuous and ident.dim == 1

    # the gap the verdict rests on, in the form each record supports
    old = census["stacked"]
    assert all(sp.scale == float(sp.singular_values[-1]) for sp, _ in old), \
        "the stacked scale *is* the stacked spectrum's largest value"
    assert min(float(sp.singular_values[sp.dim]) / sp.scale for sp, _ in old) > 0.1, \
        "the first rejected direction is nowhere near the cut (measured 0.62)"
    assert all(sp.basis_size[0] == 2 and sp.basis_size[-1] == 1
               for sp, _ in census["eigen"]), "the eigen sift rejected one of the two"


@pytest.mark.slow
def test_the_anisotropic_square_descends_from_the_cell_the_isotropic_one_allows():
    """The cell is an upper bound: the same positions, a Hamiltonian that breaks C4.

    The anisotropic square has the isotropic square's positions, so its cell
    reads the same ``P4/mmm1' #123.340``.  ``tx ≠ ty`` costs the Hamiltonian the
    fourfold axis: read free-standing, with ``s`` declared on its site, it
    descends to ``Pmmm1' #47.250``, and the group and the representation built
    on it both say they were lowered from the cell (``"descended"``).  Same
    geometry, different Hamiltonian, different answer.
    """
    low = G.realized_group(square(ty=-0.5), environment="free-standing",
                           characters={0: "s"}, cross_check=False)
    assert low, str(low)
    assert _spacegroup._group_of(square(ty=-0.5)).number_str == _spacegroup._group_of(square()).number_str \
        == "123.340", "the positions alone allow the isotropic square's group"
    assert (low.group.number_str, low.group.size) == ("47.250", 16)
    assert low.descent["from"]["number_str"] == "123.340"
    assert low.group.provenance == "descended"
    assert low.rep.group.provenance == "descended"


@pytest.mark.slow
def test_the_realized_record_unpacks_by_value_and_is_read_by_name(p41_substrate):
    """Five values unpack; everything else the record carries is read by name.

    The unpacked names are the attributes of the same names — the unpacked
    ``rep`` **is** ``r.rep``, one object under two spellings — and the record is
    not a sequence beyond that: it has no length and no positional access, so a
    position can never be read for a field it does not name.
    """
    r = p41_substrate                   # the session's shared record of a layer
    assert r, str(r)

    m, g, asg, val, rep = r
    assert m is r.model and g is r.group and asg is r.assignment
    assert val is r.validated and rep is r.rep
    assert r.bundle == (r.model, r.group, r.assignment, r.validated, r.rep)

    with pytest.raises(TypeError):
        len(r)
    with pytest.raises(TypeError):
        r[4]


# ---------------------------------------------------------------------------
# realized_group on a 3D model (2026-08-25)
# ---------------------------------------------------------------------------
#
# A fully periodic model is already its own presentation and its own cell, so
# the slab and the decoration are skipped and the remaining four calls run on
# the model itself.  What earns the single entry point is that the record is the
# *same* record and the answer is the *same* answer as the hand-chained
# pipeline, object for object.  (A Hamiltonian breaking part of the cell's group
# is `tests/test_descent.py`'s.)

@pytest.mark.slow
def test_realized_group_takes_a_3d_model_and_matches_the_hand_chain(realized):
    """The primitive ``I2_13`` crystal: ``#199.13``, 24 of 24, and no preparation at all.

    The one call and the four calls it replaces must agree *object for object*,
    not merely in the printed numbers: same operation set, same survivors, same
    representation matrices.  The preparation fields are then honestly empty —
    ``model is source`` (nothing was re-presented), ``cell is model`` (the point
    set was the cell), ``presentation["c"] is None`` (no third vector was invented)
    and ``n_check == 0`` (nothing was measured, because nothing changed).

    The same crystal with every centre shifted together by a few millionths of a
    cell, its precision declared, is named as this one is: the shift is noise
    within the declaration and is removed before the tables' setting is read,
    so the tables are read about the same origin.
    """
    r = realized(i213_flicker2018)          # the session's shared record
    assert r, str(r)
    model = r.model

    assert (r.group.number_str, r.group.sg_number) == ("199.13", 199)
    assert r.group.size == 24 and len(r.validated.indices) == 24
    assert r.validated.broken == () and r.validated.closed
    assert set(r.validated.verdicts.values()) == {"obeyed"}, \
        "every k-action class of a fully periodic cell is a singleton"
    assert r.group.provenance == "realized" and r.rep.group.provenance == "realized"

    assert r.model is r.source and r.cell is r.model
    assert r.typat == [0, 0, 0, 0] and r.cell_typat == [0, 0, 0, 0]
    assert r.presentation["c"] is None and r.presentation["n_check"] == 0
    assert r.decoration is None, "nothing was added, so there is no recipe to report"

    # ... and the same answer as the four calls, measured against them, at the
    # one call's own defaults
    g = _spacegroup._group_of(model)
    asg = _tables._assign_orbitals(model, g)
    val = _measure._validate_group(model, g, nk=6)
    rep = _build._build_rep(model, g, asg, validated=val)
    assert rep, str(rep)
    assert [s.ind for s in g.symmetries] == [s.ind for s in r.group.symmetries]
    assert tuple(val.indices) == tuple(r.validated.indices)
    assert asg.wyckoff_spglib == r.assignment.wyckoff_spglib
    assert set(rep.matrices) == set(r.rep.matrices)
    for ind, D in rep.matrices.items():
        assert np.allclose(D, r.rep.matrices[ind], atol=1e-12), ind

    # the same crystal written about an origin a noise-sized shift away, as a
    # fitted model's can be, its precision declared: the shift is noise and is
    # removed, the centres come back where they were, and every name is the
    # exact crystal's -- the tables read about the same origin
    orb = np.asarray(model.orb_vecs, dtype=float)
    shifted = TBModel(Lattice(np.asarray(model.lat_vecs, dtype=float),
                              (orb + [3e-6, -2e-6, 1e-6]).tolist(), periodic_dirs=[0, 1, 2]))
    for h in model.hoppings:
        shifted.set_hop(h["amplitude"], h["from_orbital"], h["to_orbital"],
                        list(h.get("lattice_vector", [0, 0, 0])), mode="add")
    s = G.realized_group(shifted, precision=1e-5)
    assert s, str(s)
    assert s.group.number_str == r.group.number_str
    assert np.allclose(s.group._shiftUC, r.group._shiftUC, rtol=0, atol=1e-12)
    assert s.assignment.wyckoff_tables == r.assignment.wyckoff_tables
    assert s.rep.characters.irreps == r.rep.characters.irreps
    assert s.positions["frame"] == "model" and s.positions["max_move"] == 0.0
    assert float(np.abs(np.asarray(s.model.orb_vecs) - orb).max()) <= 8 * _ULP
    for ind, D in r.rep.matrices.items():
        assert np.allclose(s.rep.matrices[ind], D, rtol=0, atol=1e-12), ind


#: The C3 orbit of a generic point on pythtb's honeycomb cell (``a1 = (1, 0)``,
#: ``a2 = (1/2, sqrt3/2)``, so ``C3: (x, y) -> (-x-y, x)``), placed above and
#: below the plane as two species.  It keeps ``C3`` and breaks every mirror and
#: everything with ``R[2,2] = -1`` -- the cell of a *chiral* layer.
_C3_ORBIT = ((0.2, 0.1), (-0.3, 0.2), (0.1, -0.3))
_CHIRAL_C3 = tuple([([x, y, G.DECORATION_DZ], "top") for x, y in _C3_ORBIT]
                   + [([x, y, -G.DECORATION_DZ], "bot") for x, y in _C3_ORBIT])


def _staggered_zeeman(h=0.3):
    """Kane-Mele plus ``h sz tau_z`` -- a magnetic layer beyond grey.

    Time reversal flips the spin and leaves the sublattice, ``C2z`` exchanges
    the sublattices and leaves the spin, and the term is odd under each; their
    *product* leaves it alone.  So the Hamiltonian has an antiunitary symmetry
    and no time reversal, which is what "beyond grey" means.
    """
    from pythtb.models import kane_mele

    model = kane_mele(0.0, -1.0, 0.3, 0.0)
    model.set_onsite([[0, 0, 0, float(h)], [0, 0, 0, -float(h)]], mode="add")
    return model


def _neel_chain(h=0.6):
    """A spinful ``s`` chain whose exchange field alternates, ``+h σ_z`` at ``x = 0`` and ``-h σ_z`` at ``x = ½``.

    The two sites of the cell ``diag(2, 1, 1)``, joined by a spin-independent
    hopping along ``x``, each to its own image along ``y`` and ``z``.  Time
    reversal flips both moments and the half translation exchanges them: each
    is broken, and their product holds.
    """
    model = TBModel(Lattice(np.diag([2.0, 1.0, 1.0]), [[0, 0, 0], [0.5, 0, 0]],
                            periodic_dirs=[0, 1, 2]), spinful=True)
    model.set_onsite([[0, 0, 0, float(h)], [0, 0, 0, -float(h)]])
    model.set_hop(-1.0, 0, 1, [0, 0, 0])
    model.set_hop(-1.0, 1, 0, [1, 0, 0])
    for o in (0, 1):
        model.set_hop(-0.4, o, o, [0, 1, 0])
        model.set_hop(-0.3, o, o, [0, 0, 1])
    return model


def _checkerboard_neel(h=0.6):
    """A spinful ``s`` square lattice on its ``√2 × √2`` cell, the exchange field ``±h σ_z`` on its two sublattices."""
    model = TBModel(Lattice([[1.0, -1.0], [1.0, 1.0]], [[0.0, 0.0], [0.5, 0.5]],
                            periodic_dirs=[0, 1]), spinful=True)
    model.set_onsite([[0, 0, 0, float(h)], [0, 0, 0, -float(h)]])
    for R in ([0, 0], [-1, -1], [0, -1], [-1, 0]):
        model.set_hop(-1.0, 0, 1, R)
    for o in (0, 1):
        model.set_hop(-0.2, o, o, [1, 0])
        model.set_hop(-0.2, o, o, [0, 1])
    return model


@pytest.mark.slow
def test_time_reversal_is_measured_down_and_the_descent_is_on_the_record():
    """The cell is asked as grey; whether it *is* grey is measured, not declared.

    A point set says nothing about a Hamiltonian's time reversal, so allowing it
    is the honest question and ``_group_of`` is always asked for the grey group.
    The detector then measures the antiunitary coset, and there are three
    answers.  Two of them are here; the third, operations combined with time
    reversal that hold while time reversal itself is broken, is below.

    **Every antiunitary operation obeyed** -- every time-reversal-symmetric
    model in this suite (Kane-Mele's free-standing ``P6/mmm1' #191.234``, whose
    24 antiunitary operations it obeys, is counted in ``test_layer.py``): the
    realization is grey and nothing moves.  The grey branch serves the group the
    cell was asked for, and a scope warning there would be noise on every such
    model, so none is issued -- asked here of the plain square layer.

    **Every one refuted** -- a Chern insulator, and a magnetic layer.  Then the
    same cell is *re-identified* without time reversal, so the labels, EBR rows
    and indicator that follow are the unitary group's own tables rather than a
    grey record with rows struck out.  The descent is narrated on the record and
    printed in its short form: which grey group it started from, how many
    antiunitary operations were refuted, and what it descended to.  The chiral
    decoration puts that case in ``P3`` in both readings, untouched by this
    wave: the decoration already breaks every operation with ``R[2,2] = -1``.
    The scope of what is served is a fact about the physics and not about the
    fixture, so the call says it, once, as a ``MagneticScopeWarning`` naming the
    type-I unitary subgroup and saying that composite antiunitary operations are
    not searched for: the cross-check's other environment runs the same branch
    on the same Hamiltonian, and the group it reaches is never served.
    """
    from pythtb.models import kane_mele

    # a time-reversal-symmetric model: kept, and nothing said about scope
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        grey = G.realized_group(square())
    assert grey, str(grey)
    assert grey.time_reversal["outcome"] == "kept"
    scope = [w for w in caught if issubclass(w.category, MagneticScopeWarning)]
    assert scope == [], [str(w.message) for w in scope]

    # ... and a magnetic layer, on the cell that expresses its unitary group
    with pytest.warns(MagneticScopeWarning) as caught:
        magnetic = G.realized_group(_staggered_zeeman(), typat=[0, 1],
                                    decorations=_CHIRAL_C3, characters={0: "A1"})
    assert magnetic, str(magnetic)
    scope = [w for w in caught if issubclass(w.category, MagneticScopeWarning)]
    assert len(scope) == 1, [str(w.message) for w in scope]
    message = str(scope[0].message)
    assert message.startswith("pythtqc: time reversal is broken by this Hamiltonian")
    assert "unitary (type-I) subgroup of its magnetic symmetry group is returned" in message
    assert "not searched for" in message
    assert "the unitary group's own" in message

    assert magnetic.group.number_str == "143" and magnetic.group.name == "P3"
    assert not magnetic.group.include_TR and _spacegroup._magnetic_type(magnetic.group) == "I"
    tr = magnetic.time_reversal
    assert tr["outcome"] == "measured broken"
    assert tr["grey_number_str"] == "143.2" and tr["grey_name"] == "P31'"
    assert (tr["n_antiunitary"], tr["n_broken"], tr["n_obeyed"]) == (3, 3, 0)
    assert tr["number_str"] == "143" and tr["size"] == 3
    assert (tr["number_str"], tr["name"], tr["size"]) == (
        magnetic.group.number_str, magnetic.group.name, magnetic.group.size)
    assert "time reversal measured broken" in tr["note"]
    assert "the unitary group's own" in tr["note"]
    assert magnetic.beyond_grey is None, "nothing antiunitary survived to warn about"
    assert "time reversal measured broken, descended from P31' #143.2" in repr(magnetic)
    assert len(magnetic.validated.indices) == 3 and magnetic.validated.broken == ()
    assert magnetic.descent is None, "the decoration asked the right cell already"

    # the physics behind that descent, proved ansatz-free and separately: time
    # reversal has NO intertwiner, and neither has C2z, but their product does.
    k = np.stack(np.meshgrid(*([np.linspace(0, 1, 6, endpoint=False)] * 2),
                             indexing="ij"), axis=-1).reshape(-1, 2)
    model = _staggered_zeeman()
    assert _detector._find_intertwiners(model, -np.eye(2), k, antiunitary=True).dim == 0
    assert _detector._find_intertwiners(model, -np.eye(2), k, antiunitary=False).dim == 0
    assert _detector._find_intertwiners(model, np.eye(2), k, antiunitary=True).dim == 2
    # ... and plain Kane-Mele, for contrast, does have time reversal
    assert _detector._find_intertwiners(kane_mele(0.0, -1.0, 0.3, 0.0), -np.eye(2), k,
                                antiunitary=True).dim == 2


@pytest.mark.slow
def test_a_surviving_antiunitary_operation_is_served_in_scope_and_warned_about():
    """Beyond grey: serve the largest validated group, and say what is beyond it.

    If some antiunitary operations are measured to hold while time reversal
    itself is refuted, the Hamiltonian's magnetic group is larger than anything
    in this package's validated universe -- the 230 space groups and their grey
    extensions.  Nothing is refused: the answer served is the largest group
    *inside* that universe, its unitary half, so every label, EBR row and
    indicator remains in scope.  What the served group cannot express is
    recorded, typed by the same detector that types a group, and **printed** --
    a measured fact this consequential does not live only in a dictionary.

    **Two Hamiltonians reach it through the descent.**  pythtb's Haldane model
    breaks its six mirrors and obeys each combined with time reversal, and
    Kane-Mele with a staggered Zeeman term, its sublattices declared different
    atoms, does the same with its three: each mirror flips the term, and so does
    time reversal.  The descent keeps the rotations and asks the group they make
    with time reversal adjoined, which contains no mirror and so no primed
    mirror either: each primed mirror is read where the descent left it, in the
    cell's own group, by the verdict that group's round measured.  The group returned
    is unchanged, the unitary ``P6`` and ``P3``; ``beyond_grey`` names the primed
    mirrors by their index in the cell's group, and the magnetic group they make
    with the rotations is of type III.  The descent's note names no group that
    does not hold: the grey group the descent arrived at is not it.

    **An antiferromagnet.**  A spinful chain whose exchange field alternates
    between its two sites breaks time reversal, and the half translation that
    exchanges the sites, and obeys their product.  Written on its magnetic
    cell, the two sites one species, it is returned the unitary ``Pmma``, and
    the eight operations combined with time reversal that hold, the half
    translation's among them, named with its translation ``{E|1/2,0,0}'``
    and never as time reversal, make a magnetic group of type IV; the sites
    typed apart, no translation exchanges them, and the four that hold beside
    ``P2/m`` make one of type III.  Time reversal is refuted by the sites it
    carries: its matrices with its own spin rotation carry each site onto the
    other, as the half translation does.
    The checkerboard antiferromagnet, a square layer, is the same on a
    substrate and free-standing -- ``P4bm`` and ``P4/mbm``, each with its
    operations combined with time reversal of type IV -- and the call that
    compares the two returns.

    **The matrices are read.**  Each primed mirror is read once, after the
    group settles, against the ``P6`` returned: it counts where the matrices
    measured make a representation of the magnetic group it generates with
    that group, up to a one-dimensional character of it.  The representation
    returned relabelled by a complex one-dimensional character of ``P6`` names
    the same six; turned by a unitary, a representation of ``P6`` still but not
    the one these states carry beside the primed mirrors' matrices, it names
    none.  Both are provoked: the build hands back its matrices so changed.

    **One whose unitary part is in the group returned never counts.**  On the
    group time reversal is measured on every unitary operation holds, so a
    ``g.T`` kept beside its own ``g`` makes ``T = g^-1 (g.T)``, and where time
    reversal is refuted no magnetic group holds the two.  A Hamiltonian's own
    measurement keeps such a ``g.T`` while refuting time reversal only where
    time reversal is broken by its square alone -- the spinless reading of
    spinor states in ``test_spin.py`` -- so here it is provoked.  On the
    free-standing layer group ``P6/mmm1' #191.234``, declared since a flat model
    is read on a substrate by default, a fake detector refutes every
    antiunitary operation but ``C2(z)`` with time reversal, whose own square is
    the ``+1`` the tables give it.  It counts nothing: the unitary
    ``P6/mmm #191`` is served, all 24 antiunitary operations are counted
    broken, nothing beyond grey is named, and the record says why.
    """
    from pythtb.models import haldane, kane_mele

    # THROUGH THE DESCENT: each primed mirror is read in the cell's own group,
    # the last that asks it, and the group returned is the unitary one
    three = {"m([010])'", "m(x)'", "m([1-10])'"}
    for model, typat, returned, cell, n_cell, primed in (
            (haldane(0.0, -1.0, 0.15), None, ("P6", "168"), "183.186", 12,
             three | {"m([2-10])'", "m(y)'", "m([110])'"}),
            (_staggered_zeeman(), [0, 1], ("P3", "143"), "156.50", 6, three)):
        with pytest.warns(MagneticScopeWarning):
            r = G.realized_group(model, typat=typat, cross_check=False)
        assert r, str(r)
        assert (r.group.name, r.group.number_str) == returned
        assert not r.group.include_TR and _spacegroup._magnetic_type(r.group) == "I"
        assert r.time_reversal["outcome"] == "measured broken"
        assert r.time_reversal["n_obeyed"] == 0, "the grey group measured holds no mirror"
        beyond = r.beyond_grey
        assert beyond is not None, "the primed mirrors hold"
        assert beyond["magnetic_type"] == "III" and beyond["closed"]
        assert (beyond["n_operations"], beyond["n_measured"]) == (len(primed), n_cell)
        assert {o["name"] for o in beyond["operations"]} == primed
        assert {o["of"] for o in beyond["operations"]} == {cell}
        assert all(set(o) == {"ind", "name", "of", "rotation", "translation", "residual",
                              "verdict"} for o in beyond["operations"])
        assert all(o["verdict"] == "obeyed" and o["residual"] < 1e-12
                   for o in beyond["operations"])
        held = [np.asarray(o["rotation"], dtype=int) for o in beyond["operations"]]
        assert all(np.array_equal(R @ R, np.eye(3, dtype=int)) and np.trace(R) == 1
                   and round(np.linalg.det(R)) == -1 for R in held), "each one a mirror"
        #  each one's mirror is broken, on the record of the same step
        assert primed <= {b["name"] + "'" for b in r.descent["dropped"] if b["of"] == cell}
        #  the note says what holds and what is returned, and names no group
        #  that does not hold: the grey group the descent arrived at is not it
        note = r.descent["note"]
        assert r.descent["to"]["name"] not in note
        assert f"Only {returned[0]} #{returned[1]}, the unitary group, is returned" in note
        assert "time reversal, measured with them, is broken" in note
        for phrase in ("measured to hold", "largest one inside this package's validated",
                       "not validated here", "is therefore in scope",
                       "read once, against the group returned",
                       "make a corepresentation of the magnetic group it generates"):
            assert phrase in beyond["note"], phrase
        assert (f"a larger magnetic group, of type III, is realized ({len(primed)} antiunitary "
                "operations measured to hold) but is beyond the validated scope") in repr(r)
        if typat is None:
            hal = r

    # AN ANTIFERROMAGNET: time reversal and the half translation each broken,
    # their product obeyed -- type IV on the magnetic cell, type III with the
    # sites typed apart, and time reversal refuted by the sites it carries
    for typat, returned, held, kind in ((None, ("Pmma", "51"), 8, "IV"),
                                        ([0, 1], ("P2/m", "10"), 4, "III")):
        with pytest.warns(MagneticScopeWarning):
            r = G.realized_group(_neel_chain(), typat=typat)
        assert r, str(r)
        assert (r.group.name, r.group.number_str) == returned
        assert r.time_reversal["outcome"] == "measured broken"
        assert "refuted by its site permutation" in r.time_reversal["note"]
        assert (r.beyond_grey["n_operations"], r.beyond_grey["magnetic_type"]) == (held, kind)
        names = {o["name"] for o in r.beyond_grey["operations"]}
        assert "E'" not in names and ("{E|1/2,0,0}'" in names) == (typat is None), names
    # ... and a layer: the same on a substrate and free-standing, so returned
    with pytest.warns(MagneticScopeWarning):
        layer = G.realized_group(_checkerboard_neel())
    assert layer, str(layer)
    assert (layer.group.name, layer.group.number_str) == ("P4bm", "100")
    assert layer.cross_check["outcome"] == "agree"
    assert layer.cross_check["free_standing"]["number_str"] == "127"
    assert (layer.beyond_grey["n_operations"], layer.beyond_grey["magnetic_type"]) == (8, "IV")
    assert "{E|1/2,1/2,0}'" in {o["name"] for o in layer.beyond_grey["operations"]}

    # THE MATRICES ARE READ, up to a one-dimensional character of P6: the
    # build hands back its representation relabelled by a complex one, and
    # turned by a unitary
    chi = next(c for c in _matrices._characters_of_subgroup(
        hal.group, [int(s.ind) for s in hal.group.symmetries], roots=True)
        if any(abs(v.imag) > 0.5 for v in c.values()))
    U = np.array([[np.cos(0.7), 1j * np.sin(0.7)], [1j * np.sin(0.7), np.cos(0.7)]]) @ np.diag(
        [1.0, np.exp(0.4j)])
    real_build = _build._build_rep
    for change, named in ((lambda i, D: chi[int(i)] * D, 6),
                          (lambda i, D: U @ D @ U.conj().T, 0)):
        def build(*args, change=change, **kw):
            """The representation built, its matrices changed."""
            rep = real_build(*args, **kw)
            return rep if isinstance(rep, Refusal) else dataclasses.replace(
                rep, matrices={i: change(i, np.asarray(D)) for i, D in rep.matrices.items()})

        with pytest.MonkeyPatch.context() as mp:
            patch_everywhere(mp, "_build_rep", build)
            with pytest.warns(MagneticScopeWarning):
                again = G.realized_group(haldane(0.0, -1.0, 0.15), cross_check=False)
        assert (again.beyond_grey or {"n_operations": 0})["n_operations"] == named
        assert again.time_reversal["n_obeyed"] == 0
        assert ("and no operation combined with time reversal" in again.descent["note"]) == (
            not named)

    # PROVOKED, on the group time reversal is measured on: a g.T the detector
    # keeps beside its own g, which holds, while time reversal is refuted
    real = _measure._validate_group
    kept = {}               # {number_str: (ind, rotation)} the fake chose to keep

    def leaves_one_antiunitary(model, group, **kw):
        """Refute every antiunitary operation but one whose rotation squares to the identity."""
        out = real(model, group, **kw)
        au = [s for s in group.symmetries if s.time_reversal]
        if not au or out.broken:
            return out
        eye = np.eye(3, dtype=int)
        sym = next(s for s in au
                   if not np.array_equal(np.rint(np.asarray(s.rotation)).astype(int), eye)
                   and np.array_equal(np.rint(np.asarray(s.rotation)).astype(int)
                                      @ np.rint(np.asarray(s.rotation)).astype(int), eye))
        keep = int(sym.ind)
        kept[group.number_str] = (keep, np.asarray(sym.rotation))
        killed = [int(s.ind) for s in au if int(s.ind) != keep]
        return dataclasses.replace(
            out, indices=tuple(i for i in out.indices if i not in killed),
            discrepancies=tuple({"ind": i, "refuted_by": "no symmetry matrix", "residual": 1.0}
                                for i in killed))

    with pytest.MonkeyPatch.context() as mp:
        patch_everywhere(mp, "_validate_group", leaves_one_antiunitary)
        out = G.realized_group(kane_mele(0.0, -1.0, 0.3, 0.0),
                               environment="free-standing", characters={0: "pz"},
                               cross_check=False)
    assert out, str(out)

    # served in scope: the largest validated group, and it is a real one
    assert out.group.number_str == "191" and out.group.name == "P6/mmm"
    assert out.group.size == 24 and out.group.provenance == "realized"
    assert not out.group.include_TR and _spacegroup._magnetic_type(out.group) == "I"
    assert out.validated.broken == () and len(out.rep.indices) == 24
    #  the one the fake kept is C2(z) with time reversal, and C2(z) holds
    _ind, rotation = kept["191.234"]
    assert np.array_equal(np.rint(rotation).astype(int), np.diag([-1, -1, 1]))
    assert "C2(z)" in {op.name for op in out.operations.values()}

    # ... and it counts nothing: with C2(z) it makes time reversal, refuted
    tr = out.time_reversal
    assert tr["outcome"] == "measured broken" and tr["grey_number_str"] == "191.234"
    assert (tr["n_broken"], tr["n_obeyed"]) == (24, 0)
    assert tr["note"].startswith("time reversal measured broken: every operation of "
                                 "P6/mmm1' #191.234 combined with time reversal is refuted")
    assert ("Of those, 1, a coset representative of the lattice translations, the "
            "single-valued group's, standing for itself and its negative in the double "
            "group, has a symmetry matrix and fails the rule") in tr["note"]
    assert "it generates time reversal with its own unitary part" in tr["note"]
    assert out.beyond_grey is None

    # ... and printed: the grey group measured is named, nothing beyond it
    line = repr(out)
    assert "time reversal measured broken, descended from P6/mmm1' #191.234" in line
    assert "a larger magnetic group" not in line


# ---------------------------------------------------------------------------
# the EBR-row convention transport (2026-08-21)
# ---------------------------------------------------------------------------
#
# `irreptables` ships EBR rows in BCS's e^{+ik.t} convention; this package
# measures in the physical e^{-ik.t} one, so `ebr_data` transports the rows at
# the adapter boundary (see its docstring and docs/background/conventions.md,
# "Reading across to BCS").  These tests work at the
# *data* level -- no model, no cell, no refUC -- because that is where the claim
# lives; `test_ebr.py`'s deep-atomic-limit control is the physical one.

#: (number_str, spinor, magnetic) instances spanning the fixture set: the
#: unitary tables and the grey ones the fixtures actually load.
_TRANSPORT_CASES = [
    ("143", False, False), ("143", True, False), ("143.1", False, True),
    ("156", False, False), ("156", True, False), ("156.50", True, True),
    ("183", False, False), ("183", True, False), ("183.186", True, True),
    ("99", False, False), ("99", True, False), ("99.164", True, True),
    ("199", False, False), ("199", True, False), ("199.12", False, True),
    ("227", False, False), ("227", True, False), ("227.129", True, True),
    ("208", False, False), ("208", True, False), ("208.45", False, True),
]

#: The instances of :data:`_TRANSPORT_CASES` whose transported EBR rows differ
#: from the shipped ones -- 7 of the 21, measured 2026-09-09; the other 14 come
#: back row for row identical, so a test that only checks invariants there is
#: checking them against the identity map.  Every row that moves does so at a
#: **non-TRIM** point: ``K``/``H`` (and their ``A`` arms) in the hexagonal C3
#: groups 143 and 156, and ``P``/``PA`` in 199, where only the spinful table
#: moves.  The six instances of 199 and 208 with a non-empty
#: ``conjugated_points`` but no moved row are the reason this is a typed census
#: and not "wherever the record says something was conjugated".
_TRANSPORT_MOVES_ROWS = frozenset({
    ("143", False, False), ("143", True, False), ("143.1", False, True),
    ("156", False, False), ("156", True, False), ("156.50", True, True),
    ("199", True, False),
})

#: Centering translations by Hermann-Mauguin lattice letter.  Generic
#: crystallography, not knowledge of any particular group: it is what turns
#: "2k is an integer vector in the conventional basis" into the real question,
#: "2k is a vector of the *primitive* reciprocal lattice".
_CENTERING = {
    "P": [], "A": [[0, .5, .5]], "B": [[.5, 0, .5]], "C": [[.5, .5, 0]],
    "I": [[.5, .5, .5]], "F": [[0, .5, .5], [.5, 0, .5], [.5, .5, 0]],
    "R": [[2 / 3, 1 / 3, 1 / 3], [1 / 3, 2 / 3, 2 / 3]],
}


def _shipped(number_str, spinor):
    """The EBR rows exactly as ``irreptables`` ships them, untransported."""
    from irreptables.ebrs import load_ebr_data

    raw = load_ebr_data(number_str, spinor)
    labels = list(raw["basis"]["irrep_labels"])
    rows = [{l: int(v) for l, v in zip(labels, e["vector"]) if v} for e in raw["ebrs"]]
    keys = [(e["ebr_name"], e["wyckoff_position"]) for e in raw["ebrs"]]
    return labels, keys, rows


def _transported(number_str, spinor, magnetic):
    """The same rows, in this package's convention (the adapter's own transport)."""
    from irreptables.ebrs import load_ebr_data

    raw = _tables._to_physical_convention(load_ebr_data(number_str, spinor),
                                    number_str, spinor, magnetic)
    labels = list(raw["basis"]["irrep_labels"])
    rows = [{l: int(v) for l, v in zip(labels, e["vector"]) if v} for e in raw["ebrs"]]
    keys = [(e["ebr_name"], e["wyckoff_position"]) for e in raw["ebrs"]]
    return labels, keys, rows


def _table_facts(number_str, spinor, magnetic):
    """Per k-point: is it a TRIM, and does conjugating the factor system relabel it?"""
    from irreptables.irreps import IrrepTable

    table = IrrepTable(number_str, spinor, magnetic=magnetic)
    centering = _CENTERING[table.name.strip()[0]]
    t_of = {i + 1: np.asarray(s.t, dtype=float) for i, s in enumerate(table.symmetries)}
    by_kp = {}
    for irr in table.irreps:
        by_kp.setdefault(irr.kpname, []).append(irr)
    out = {}
    for kp, irreps in by_kp.items():
        k = np.asarray(irreps[0].k, dtype=float)
        two_k = 2 * k
        trim = (np.abs(two_k - np.round(two_k)).max() < 1e-4
                and all(abs(np.dot(two_k, c) - round(np.dot(two_k, c))) < 1e-4
                        for c in centering))
        inds = sorted(irreps[0].characters)
        phase = np.array([np.exp(-4j * np.pi * k.dot(t_of[i])) for i in inds])
        chi = {ir.name: np.array([complex(ir.characters[i]) for i in inds])
               for ir in irreps}
        same = all(np.abs(v * phase - v).max() < 2e-3 for v in chi.values())
        out[kp] = {"trim": bool(trim), "table_unchanged": bool(same)}
    return out


def test_the_EBR_transport_fixes_Gamma_the_dimensions_and_the_positions():
    """What the transport may not do, and where it may move a row, on every fixture group's tables.

    A convention flip of the site-displacement phases ``e^{±2πi k·m_j}`` can
    only *relabel*: at ``k = 0`` the phase is 1, so the Γ block is untouched; the
    band representation's dimension at each k-point is a count of bands and
    cannot move; the Wyckoff position is a property of the orbit and the site
    representation is only exchanged with its complex conjugate, so neither the
    column's name nor its position changes.

    Every one of those is also satisfied by a transport that does nothing at
    all, so the census pinned in :data:`_TRANSPORT_MOVES_ROWS` carries the
    positive half: on 7 of these 21 instances the transported rows *must* differ
    from the shipped ones, and on the other 14 they must not.

    Where a row may move is a property of the *tables*, not of the code.  The
    induced character at ``k`` is convention-blind whenever ``e^{4πi k·m} = 1``
    for every integer displacement ``m`` — i.e. at every TRIM.  A TRIM row can
    still be *relabelled*, but only when conjugating the tabulated factor system
    relabels the little-group table there too (a nonsymmorphic point with
    ``2k·t ∉ ℤ``, measured: SG 208's ``R`` and ``X``), and then the same
    relabelling has already happened on the measurement side in
    :func:`groups._transported_table` — which is the whole point of doing both.
    So: **a row may differ from the shipped one only at a k-point that is not a
    TRIM, or at one whose transported table is not the tabulated table.**  The
    points that do move are ``K``/``H`` (143, 156) and ``P``/``PA`` (spinful 199)
    -- all of them non-TRIM, so the guard is passed on the ``trim`` half; the
    TRIMs whose *table* the conjugation relabels are 199's ``N`` and 208's
    ``R``/``X``, and no row moves at any of them.
    """
    from irreptables.ebrs import load_ebr_data

    for number_str, spinor, magnetic in _TRANSPORT_CASES:
        instance = (number_str, spinor, magnetic)
        labels, keys, old = _shipped(number_str, spinor)
        labels2, keys2, new = _transported(number_str, spinor, magnetic)
        assert labels2 == labels and keys2 == keys, \
            f"{instance}: names and positions must not move"
        assert (new != old) == (instance in _TRANSPORT_MOVES_ROWS), \
            f"{instance}: the invariants below are checked against a transport " \
            "that moved something"

        point = {l: _tables._label_kpoint(l) for l in labels}
        degen = dict(zip(labels,
                         load_ebr_data(number_str, spinor)["basis"]["degeneracies"]))
        for (name, wy), a, b in zip(keys, old, new):
            gamma_a = {l: v for l, v in a.items() if point[l] == "GM"}
            gamma_b = {l: v for l, v in b.items() if point[l] == "GM"}
            assert gamma_a == gamma_b, f"{instance}: Γ moved for {name}@{wy}"
            for kp in {point[l] for l in labels}:
                dim_a = sum(v * degen[l] for l, v in a.items() if point[l] == kp)
                dim_b = sum(v * degen[l] for l, v in b.items() if point[l] == kp)
                assert dim_a == dim_b, \
                    f"{instance}: the dimension at {kp} moved for {name}@{wy}"

        # where a row moved, it may: a non-TRIM, or a TRIM whose table moved too
        facts = _table_facts(number_str, spinor, magnetic)
        moved = set()
        for (name, wy), a, b in zip(keys, old, new):
            for l in set(a) | set(b):
                if a.get(l, 0) != b.get(l, 0):
                    moved.add(point[l])
        assert bool(moved) == (instance in _TRANSPORT_MOVES_ROWS), \
            f"{number_str}: the transport moved rows at {sorted(moved)}, against the census"
        for kp in moved:
            fact = facts.get(kp) or facts[kp[:-1]]          # the untabulated 'A' arm
            assert not (fact["trim"] and fact["table_unchanged"]), (
                f"{number_str}: rows moved at {kp}, which is a TRIM whose transported "
                "little-group table is identical to the tabulated one — nothing there "
                "can legitimately move")


def test_the_EBR_transport_reproduces_the_canonical_induction_on_SG143():
    """The decisive check: induce the rows from scratch, in both conventions.

    SG 143 (``P3``) is the smallest group where the artifact is visible and the
    only ingredient the tables do not supply is three Wyckoff coordinates, so
    the induced character

        χ^(s)({R|t}) = Σ_j e^{s·2πi k·m_j} χ_ρ(g_j),   m_j = R τ_j + t − τ_j

    can be built here from ``irreptables``' own ``{R|t}`` and characters and
    decomposed against the tabulated table (``s = +1``) or the
    factor-system-conjugated one (``s = −1``).  Measured:

    * ``s = +1`` reproduces **all nine shipped rows exactly**, arms included —
      which is the model-free proof that BCS's rows are in the ``e^{+ik·m}``
      convention;
    * ``s = −1`` reproduces **all nine transported rows exactly** — which is the
      proof that what :func:`groups.ebr_data` serves is the physical induction
      and not merely a plausible permutation of BCS's.

    The site characters are not typed here either: they are read off the ``1a``
    rows, which sit at the origin and therefore carry no displacement phase at
    all (asserted, by the ``1a`` rows coming out convention-independent).
    """
    from irreptables.irreps import IrrepTable

    tag, spinor, magnetic = "143.1", False, True     # the grey table carries KA/HA
    sites = {"1a": np.zeros(3), "1b": np.array([1 / 3, 2 / 3, 0.0]),
             "1c": np.array([2 / 3, 1 / 3, 0.0])}    # ITA P3, conventional setting
    table = IrrepTable(tag, spinor, magnetic=magnetic)
    R = [np.asarray(s.R, dtype=int) for s in table.symmetries]
    T = [np.asarray(s.t, dtype=float) for s in table.symmetries]
    by_kp = {}
    for irr in table.irreps:
        by_kp.setdefault(irr.kpname, []).append(irr)

    labels, keys, shipped = _shipped(tag, spinor)
    _, _, physical = _transported(tag, spinor, magnetic)
    rows = {(n.split("↑")[0], w.split("(")[0]): r for (n, w), r in zip(keys, shipped)}
    phys = {(n.split("↑")[0], w.split("(")[0]): r for (n, w), r in zip(keys, physical)}

    site_chars = {}                                  # χ_ρ, read off the origin rows
    for (name, wy), row in rows.items():
        if wy == "1a":
            gm = next(i for i in by_kp["GM"] if i.name in row)
            site_chars[name] = {i: complex(c) for i, c in gm.characters.items()}
    assert set(site_chars) == {"A1", "1E", "2E"}

    def decompose(kpname, chars, s):
        irreps = by_kp[kpname]
        k = np.asarray(irreps[0].k, dtype=float)
        inds = sorted(irreps[0].characters)
        phi = np.array([1.0 if s > 0 else np.exp(-4j * np.pi * k.dot(T[i - 1]))
                        for i in inds])
        A = np.array([[complex(ir.characters[i]) for i in inds]
                      for ir in irreps]).T * phi
        n, *_ = np.linalg.lstsq(A, np.array([chars[i] for i in inds]), rcond=None)
        assert np.abs(A @ n - np.array([chars[i] for i in inds])).max() < 1e-4
        assert np.abs(n - np.round(n.real)).max() < 1e-4, (kpname, n)
        return {ir.name: int(round(v.real)) for ir, v in zip(irreps, n) if round(v.real)}

    for s, reference in ((+1, rows), (-1, phys)):
        for wy, tau in sites.items():
            for name, chi in site_chars.items():
                got = {}
                for kpname, irreps in by_kp.items():
                    k = np.asarray(irreps[0].k, dtype=float)
                    chars = {}
                    for ind in sorted(irreps[0].characters):
                        m = R[ind - 1] @ tau + T[ind - 1] - tau
                        assert np.allclose(m, np.round(m)), (wy, ind)
                        chars[ind] = np.exp(s * 2j * np.pi * k.dot(m)) * chi[ind]
                    got.update(decompose(kpname, chars, s))
                assert got == reference[(name, wy)], (s, name, wy, got)

    # and the headline numbers, spelled out
    assert {l for l in rows[("A1", "1b")] if l.startswith("K")} == {"K2", "KA2"}
    assert {l for l in phys[("A1", "1b")] if l.startswith("K")} == {"K3", "KA3"}
    assert {l for l in phys[("A1", "1c")] if l.startswith("K")} == {"K2", "KA2"}
    assert {l for l in phys[("A1", "1a")] if l.startswith("K")} == {"K1", "KA1"}
    assert rows[("A1", "1a")] == phys[("A1", "1a")], "the origin orbit has no phases"


def test_the_EBR_transport_swaps_1b_and_1c_of_spinful_SG156():
    """The spinful instance, which is the one Kane-Mele lands on.

    The displacement phases are spin-independent, so the double-valued rows must
    move exactly as the single-valued ones do: SG 156's ``1b`` and ``1c``
    exchange their ``K``/``H`` labels and nothing else changes.  This is the
    table-level statement of the position swap that
    ``examples/tqc_kane_mele_realized_group.py`` measures from a Hamiltonian
    and its golden pins: the valence pair ``-> 1 x -E1↑G(2)@1a`` in one regime
    and ``-> 1 x -E1↑G(2)@1b`` in the other.
    """
    for tag, spinor, magnetic in (("156", True, False), ("156.50", True, True)):
        labels, keys, old = _shipped(tag, spinor)
        _, _, new = _transported(tag, spinor, magnetic)
        rows = {f"{n}@{w.split('(')[0]}": (a, b) for (n, w), a, b in zip(keys, old, new)}
        for key, (a, b) in rows.items():
            moved = {l for l in set(a) | set(b) if a.get(l, 0) != b.get(l, 0)}
            assert all(_tables._label_kpoint(l) in ("K", "H") for l in moved), key
            assert ("@1a" in key) == (not moved), f"only 1b/1c move, not {key}"
        def block(letter, which):
            return sorted((tuple(sorted(v[which].items()))
                           for k, v in rows.items() if k.endswith("@" + letter)))
        assert block("1b", 1) == block("1c", 0), f"{tag}: 1b now carries 1c's rows"
        assert block("1c", 1) == block("1b", 0), f"{tag}: 1c now carries 1b's rows"
        assert block("1a", 1) == block("1a", 0), f"{tag}: 1a is untouched"


# ---------------------------------------------------------------------------
# the completion of the transport (2026-08-21): whole-character conjugation
# ---------------------------------------------------------------------------
#
# Where the tabulated labelling names no physical representation -- a complex
# tabulated factor system, or a grey time-reversal gluing built on a complex
# phase -- `groups._transported_table` serves conj(chi_BCS) under the same BCS
# names.  These tests are at the *data* level: they check the rule the adapter
# applies, over every table irreptables ships, without a model in sight.  The
# two screw fixtures in `test_multiorbital.py` are the Hamiltonian-level
# counterpart.


def _served_rows(table, kp, irreps, conjugated):
    """What ``_transported_table`` serves at ``kp``, on the table's own reps.

    ``t_ours = t_table`` and ``sym.sign = 1`` in that gauge, so the two branches
    of :func:`groups._transported_table` reduce to ``χ·φ`` (tabulated labelling
    kept) and ``conj(χ)`` (whole-character conjugation), which is what the
    sweep needs -- the setting transformation is a separate concern, tested on
    real cells by ``test_the_transported_table_conjugates_only_the_factor_system``.

    That this reduction really is what the library serves is not assumed:
    ``test_the_sweeps_served_rows_are_what_transported_table_serves`` hands
    :func:`groups._transported_table` a group whose operations are one table's
    own and compares the two, at k-points where the re-attached phase is not 1.
    """
    t_of = {i + 1: np.asarray(s.t, dtype=float) for i, s in enumerate(table.symmetries)}
    k = np.asarray(irreps[0].k, dtype=float)
    inds = sorted(irreps[0].characters)
    rows = {}
    for irr in irreps:
        row = {}
        for i in inds:
            gamma = complex(irr.characters[i]) * np.exp(-2j * np.pi * t_of[i] @ k)
            if conjugated:
                gamma = np.conj(gamma)
            row[i] = gamma * np.exp(-2j * np.pi * t_of[i] @ k)
        rows[irr.name] = row
    return rows


def _served_named(table, kp, irreps, conjugated, c):
    """:func:`_served_rows` re-keyed by the name ``_transported_table`` serves it under.

    ``_served_rows`` keys by the *tabulated* name because the row checks do not
    care; every statement about **labels** has to go through ``c``, which since
    2026-08-24 is no longer the identity on a unitary table's conjugated points
    (the served name follows the printed character -- see
    ``groups._table_conjugation``).
    """
    rows = _served_rows(table, kp, irreps, conjugated)
    if not conjugated:
        return rows
    return {c.get(name, name): row for name, row in rows.items()}


def _rep_geometry(rot, trans, k, su2=None):
    r"""Everything about a little group that a *physical* character must respect.

    Returns ``(inds, conjugation_terms, cyclic_terms)``, precomputed once per
    k-point so the per-row check is arithmetic:

    * ``conjugation_terms`` are ``(j, ℓ, ±e^{-2πi k·m})`` triples realizing
      ``χ(g_j) = ± e^{-2πi k·m} χ(g_ℓ)`` for ``g_ℓ = g_i g_j g_i⁻¹`` (the
      relation :func:`_class_function_defect` scans);
    * ``cyclic_terms`` are, for each element ``g``, the data of the cyclic
      subgroup it generates: ``g^j`` is the little-group representative
      ``g_{ℓ_j}`` times a lattice translation, so ``p_j = Tr D(g)^j`` is fixed
      by the served characters, while ``g^n = {1|m}`` forces **every eigenvalue
      of D(g) to be an n-th root of** ``e^{-2πi k·m}``.  The multiplicity of
      each root is then ``ν_r = (1/n) Σ_j p_j r^{-j}``, and a genuine
      representation makes every one of them a non-negative integer.

    The second relation is what the first cannot see.  At a k-point whose
    little group is abelian with only trivial translation differences -- SG
    152's ``L``, where the tabulated factor system is a cube root of unity --
    every ``m`` above vanishes and the class-function scan is vacuous, while
    the cyclic relation still separates a representation of ``ω₀`` from one of
    ``ω₀*``.
    """
    inds = sorted(rot)
    pos = {i: a for a, i in enumerate(inds)}
    key = lambda R: tuple(np.round(R).astype(int).ravel())        # noqa: E731
    by_rot = {}
    for i in inds:
        by_rot.setdefault(key(rot[i]), i)
    k = np.asarray(k, dtype=float)
    eye = np.eye(3)

    js, ells, phases = [], [], []
    for i in inds:
        Ri, ti = rot[i], trans[i]
        Rinv = np.linalg.inv(Ri)
        for j in inds:
            Rc = Ri @ rot[j] @ Rinv
            ell = by_rot.get(key(Rc))
            if ell is None:
                continue
            m = ti + Ri @ trans[j] - Rc @ ti - trans[ell]
            js.append(pos[j])
            ells.append(pos[ell])
            phases.append(_spinor_sign(su2, i, j, ell)
                          * np.exp(-2j * np.pi * k @ m))

    cyclic = []
    for i in inds:
        R, t = rot[i], trans[i]
        Rp, tp = eye.copy(), np.zeros(3)
        Sp = np.eye(2, dtype=complex) if su2 is not None else None
        ell_j, ph_j = [], []
        for _ in range(24):
            tp = R @ tp + t
            Rp = R @ Rp
            if su2 is not None:
                Sp = su2[i] @ Sp
            ell = by_rot.get(key(Rp))
            if ell is None:                       # a power left the little group
                ell_j = None
                break
            s = 1.0
            if su2 is not None:
                s = (1.0 if np.abs(Sp - su2[ell]).max() <= np.abs(Sp + su2[ell]).max()
                     else -1.0)
            if np.allclose(Rp, eye) and (su2 is None
                                         or np.abs(Sp - np.eye(2)).max() < 1e-3):
                break
            ell_j.append(pos[ell])
            ph_j.append(s * np.exp(-2j * np.pi * k @ (tp - trans[ell])))
        else:                                     # pragma: no cover - order > 24
            ell_j = None
        if ell_j is None:
            continue
        n = len(ell_j) + 1
        target = np.exp(-2j * np.pi * k @ tp)     # g^n = {1|tp}
        roots = target ** (1 / n) * np.exp(2j * np.pi * np.arange(n) / n)
        cyclic.append((n, pos[by_rot[key(eye)]], np.array(ell_j, dtype=int),
                       np.array(ph_j), roots))
    return inds, (np.array(js, dtype=int), np.array(ells, dtype=int),
                  np.array(phases)), tuple(cyclic)


def _physical_rep_defect(geom, chi):
    """``(class-function defect, cyclic-eigenvalue defect)`` of one served row."""
    inds, (js, ells, phases), cyclic = geom
    v = np.array([complex(chi[i]) for i in inds])
    cf = float(np.abs(v[js] - phases * v[ells]).max()) if js.size else 0.0
    cy = 0.0
    for n, e_pos, ell_j, ph_j, roots in cyclic:
        p = np.concatenate([[v[e_pos]], v[ell_j] * ph_j])
        nu = (p[None, :] * np.conj(roots)[:, None] ** np.arange(n)[None, :]).sum(1) / n
        cy = max(cy, float(np.abs(nu - np.round(nu.real)).max()))
    return cf, cy


def _sweep_instances():
    """Every ``(number_str, spinor, magnetic)`` this package can be handed.

    The unitary set is the 230 space groups in both valuednesses.  The grey set
    is the **type-II** magnetic group of each family -- the second serial in
    the family, taken from ``irreptables``' own file names so that no BNS table
    is typed here; the sweep then *asserts* the defining property (the
    antiunitary half is the unitary half times ``{E|0}·T``) on every table it
    loads, so the shortcut is checked rather than trusted.
    """
    import glob
    import os
    import re

    import irreptables

    unitary = [(str(n), sp, False) for n in range(1, 231) for sp in (False, True)]
    data = os.path.join(os.path.dirname(irreptables.__file__), "data", "correptables")
    tags = {re.match(r"irreps-SG=(.+)-(scal|spin)\.dat", os.path.basename(f)).group(1)
            for f in glob.glob(os.path.join(data, "*.dat"))}
    family: dict[int, list] = {}
    for tag in tags:
        head, serial = tag.split(".")
        family.setdefault(int(head), []).append(int(serial))
    grey = [(f"{head}.{sorted(serials)[1]}", sp, True)
            for head, serials in sorted(family.items()) for sp in (False, True)]
    return unitary, tuple(grey)


def _assert_grey(table):
    """The loaded table is a grey group: its antiunitary half is ``{E|0}·T`` times it."""
    tr = [bool(getattr(s, "time_reversal", False)) for s in table.symmetries]
    def key(s):
        return (tuple(np.asarray(s.R, dtype=int).ravel()),
                tuple(np.round(np.asarray(s.t, dtype=float), 4)))
    unitary = {key(s) for s, a in zip(table.symmetries, tr) if not a}
    assert unitary and unitary == {key(s) for s, a in zip(table.symmetries, tr) if a}


#: ``irreptables`` prints |χ| and phases to five decimals; a served character of
#: modulus 4 therefore carries ~1e-4 of rounding, and every *genuine* violation
#: measured below is O(1).
_SWEEP_TOL = 2e-3


@pytest.fixture(scope="module")
def shipped_tables():
    """Every table :func:`_sweep_instances` names, loaded once for the sweeps.

    ``{(number_str, spinor, magnetic): IrrepTable}`` over the 460 unitary and
    460 grey instances.  ``tables_status()`` is asked first, always:
    ``irreptables.irreps`` imports ``irrep.utility``, so reaching irreptables
    before irrep leaves that module half-built when irrep runs its own
    ``from irreptables.irreps import IrrepTable``, and irrep falls back to
    ``DummyMethodTables`` for the rest of the process.
    """
    assert G.tables_status()["tables"]
    from irreptables.irreps import IrrepTable

    unitary, grey = _sweep_instances()
    return {key: IrrepTable(key[0], key[1], magnetic=key[2])
            for key in unitary + list(grey)}


@pytest.fixture(scope="module")
def shipped_naming(shipped_tables):
    """``{instance: (c, conjugated, broken)}``: :func:`groups._table_conjugation` of every table, once."""
    return {key: _tables._table_conjugation(table) for key, table in shipped_tables.items()}


@pytest.mark.data
def test_the_served_table_is_a_physical_representation_for_every_group(shipped_tables,
                                                                       shipped_naming):
    """The sweep: **every** table ``_transported_table`` serves, everywhere.

    Over all 460 unitary and all 460 grey (space group, valuedness) instances
    -- every table ``irreptables`` ships that this package can be handed -- each
    served little-group row is checked against two independent consequences of
    "this is the character of a representation in the **physical**
    ``e^{-2πi k·t}`` convention": the ω-twisted class-function relation and the
    cyclic-eigenvalue relation of :func:`_rep_geometry`.  Measured: 5462
    k-points, 16 532 rows, worst class-function defect 3.1e-05 and worst cyclic
    defect 1.2e-05 -- both ``irreptables``' own printed precision.

    The same scan run against the *pre-completion* rule (keep the tabulated
    ``Γ``, flip the phase, everywhere) fails on 322 of those rows, all of them
    at the k-points where no tabulated pairing exists; that half is asserted in
    :func:`test_whole_character_conjugation_is_load_bearing_where_it_is_applied`.

    The conjugated census went 173 → 176 with the re-gluing (2026-08-21) and
    then to **685** with the "name follows the character" directive
    (2026-08-24), which serves the whole-character conjugate wherever that is
    what puts the BCS name on the BCS row -- i.e. at every k-point where
    multiplying the tabulated rows by ``φ`` permutes them.  Being in this census
    says nothing about the *rows* checked here -- the served character set is
    the same either way -- only about which name is attached to which row.
    The two subsets that *are* physical statements are asserted alongside:

    * ``no_pairing`` (**174** entries, 28 space-group families): the tabulated
      ``Γ`` rule cannot produce the physical character set at all, so the
      conjugation is load-bearing for the rows and not only for the names.
      This is the census the pre-directive ``conjugated`` count was measuring
      (176 = these 174 plus the spinful ``L`` of ``178.156``/``179.162``, where
      the grey table's own character matching disagreed with the parent's
      relabelling and the parent had to win);
    * ``divergent`` (**59** entries, 25 unitary and 34 grey): the shipped
      character set is not closed under conjugation, so no permutation of BCS's
      rows is the physical table.  Every one of them is a ``P``/``W``-type point
      of a body- or face-centred group, which is the geometry that makes the
      shipped page describe the other star member ``−k``.
    """
    unitary, grey = _sweep_instances()
    assert len(unitary) == 460 and len(grey) == 460, (len(unitary), len(grey))

    n_points = n_rows = 0
    worst_cf = worst_cy = 0.0
    conjugated_points, no_pairing, divergent = [], [], []
    all_real, complex_but_closed = [], []
    for number_str, spinor, magnetic in unitary + list(grey):
        table = shipped_tables[(number_str, spinor, magnetic)]
        if magnetic:
            _assert_grey(table)
        _, conjugated, broken = shipped_naming[(number_str, spinor, magnetic)]
        t_of = {i + 1: np.asarray(s.t, dtype=float)
                for i, s in enumerate(table.symmetries)}
        r_of = {i + 1: np.asarray(s.R, dtype=float)
                for i, s in enumerate(table.symmetries)}
        by_kp = {}
        for irr in table.irreps:
            by_kp.setdefault(irr.kpname, []).append(irr)
        for kp, irreps in by_kp.items():
            if kp in broken:
                continue
            n_points += 1
            inds = sorted(irreps[0].characters)
            su2 = ({i: np.asarray(table.symmetries[i - 1].S, dtype=complex)
                    for i in inds} if spinor else None)
            k = np.asarray(irreps[0].k, dtype=float)
            geom = _rep_geometry({i: r_of[i] for i in inds},
                                 {i: t_of[i] for i in inds}, k, su2)
            if kp in conjugated:
                conjugated_points.append((number_str, spinor, magnetic, kp))
            phi = np.array([np.exp(-4j * np.pi * k @ t_of[i]) for i in inds])
            chi = {irr.name: np.array([complex(irr.characters[i]) for i in inds])
                   for irr in irreps}
            if _tables._matching(chi, lambda v, p=phi: np.conj(v) / p) is None:
                no_pairing.append((number_str, spinor, magnetic, kp))
            if _tables._matching(chi, np.conj) is None:
                divergent.append((number_str, spinor, magnetic, kp))
            elif all(np.abs(v.imag).max() < 1e-9 for v in chi.values()):
                all_real.append((number_str, spinor, magnetic, kp))
            else:
                complex_but_closed.append((number_str, spinor, magnetic, kp))
            for name, row in _served_rows(table, kp, irreps, kp in conjugated).items():
                n_rows += 1
                cf, cy = _physical_rep_defect(geom, row)
                worst_cf, worst_cy = max(worst_cf, cf), max(worst_cy, cy)
                assert max(cf, cy) < _SWEEP_TOL, (
                    f"#{number_str} spinor={spinor} magnetic={magnetic} {kp} {name}: "
                    f"class-function {cf:.3g}, cyclic {cy:.3g}")

    assert n_points == 5462 and n_rows == 16532, (n_points, n_rows)
    assert worst_cf < 1e-4 and worst_cy < 1e-4, (worst_cf, worst_cy)
    assert len(conjugated_points) == 685, len(conjugated_points)

    #  the rows really do need the conjugation here, not just the names
    assert len(no_pairing) == 174, len(no_pairing)
    families = {int(n.split(".")[0]) for n, _, _, _ in no_pairing}
    assert families == {76, 78, 80, 91, 92, 95, 96, 98, 109, 110, 122, 144, 145,
                        151, 152, 153, 154, 169, 170, 171, 172, 178, 179, 180,
                        181, 210, 212, 213}
    assert not families & {99, 143, 156, 183, 191, 199, 208, 227, 123, 47}, (
        "no fixture group of the phases before the re-gluing may be touched")

    #  and the (much smaller) set where the printed table is not BCS's at all
    assert len(divergent) == 59, len(divergent)
    assert (sum(1 for e in divergent if not e[2]),
            sum(1 for e in divergent if e[2])) == (25, 34)
    assert {kp for _n, _s, _m, kp in divergent} == {"P", "PA", "W", "WA"}, (
        "the divergence is a P/W-point phenomenon of the centred lattices")
    assert {int(n.split(".")[0]) for n, _s, _m, _k in divergent} == {
        24, 73, 80, 88, 98, 109, 110, 122, 141, 142, 199, 203, 206, 210, 214,
        220, 227, 228, 230}
    #  the memoized public wrapper, against values typed here rather than
    #  against the scan above -- which would be the same loop over the same
    #  predicate on both sides (measured 2026-09-09); the transport-record test
    #  reads #199 spinful, #80.30 and #143 through the published record
    assert _tables._divergent_kpoints("199", False, False) == frozenset({"P"})
    assert _tables._divergent_kpoints("199.12", False, True) == frozenset({"P", "PA"})
    assert _tables._divergent_kpoints("227", True, False) == frozenset({"W"})
    assert _tables._divergent_kpoints("80.29", True, True) == frozenset()

    #  the owner's suspected equivalence, measured (2026-08-24): "exchange ⟺ all
    #  characters real" vs "irreducible ⟺ some character complex".  One
    #  direction is exact -- an all-real character set is closed under
    #  conjugation by inspection, so it is NEVER divergent -- and the converse
    #  is false: 1137 k-points have a complex character and are still a
    #  permutation of BCS's rows (the complex rows pair up with each other).
    #  Restricted to the class the conjecture was formed on -- the 31 k-point
    #  entries the pre-directive re-gluing census listed -- it *is* exact: the
    #  28 with all-real corep characters all became exchanges, and the 3 with a
    #  complex one (80.30/109.240/110.246 ``P``) are all still irreducible.  The
    #  live half of that is the pair of assertions "every re-glued point is
    #  divergent" (in the re-glued census below) and "all-real is never
    #  divergent" (here).
    assert len(all_real) + len(complex_but_closed) + len(divergent) == 5462
    assert (len(all_real), len(complex_but_closed)) == (4266, 1137), (
        len(all_real), len(complex_but_closed))
    assert not set(all_real) & set(divergent), "all-real is never divergent"


def test_whole_character_conjugation_is_load_bearing_where_it_is_applied():
    """Where the tabulated ``Γ`` has no pairing, it names something unphysical.

    Since the "name follows the character" directive (2026-08-24) the
    conjugation branch is taken at 685 k-points, but at most of them it is
    load-bearing only for the *labels*: the two rules give the same rows there
    and differ in which name is on which, which is
    :func:`test_the_two_transport_rules_agree_as_sets_wherever_the_tabulated_one_stands`.
    This test is about the 174 where it is load-bearing for the **rows** --
    ``no_pairing`` in the sweep above, i.e. where multiplying the tabulated rows
    by ``φ`` does not permute them.  Two statements, on six instances spanning
    both mechanisms (a complex tabulated factor system in #91/#152/#212, a grey
    time-reversal gluing built on a complex phase in 76.8/144.5/169.114):

    1. **the decisive one**: at least one row of the pre-completion rule is not
       *any* of the physical characters -- ``χ_D·φ`` differs from every
       ``conj(χ_E)``.  Since the whole-conjugate set is the physical one
       (asserted by the sweep above), that is the precise sense in which the
       tabulated labelling names no physical representation there, and it is
       what makes the conjugation forced rather than a choice.  The converse is
       asserted too: at a conjugated point that *does* have a tabulated pairing
       (``144.5``'s ``H``/``HA``, ``169.114``'s ``L``) nothing is stranded, and
       only the naming moved;
    2. the pre-completion rows are also caught outright by the representation
       checks -- ``n_old_cf`` of them violate the physical class-function
       relation and ``n_old_cy`` the cyclic-eigenvalue one.  Neither check
       subsumes the other, and SG 152's ``L`` is the case that shows it: its
       little group is ``{E, 2}`` with no translation differences at all, so the
       class-function scan is vacuous there and only the cyclic relation
       separates ``ω₀`` from ``ω₀*``.
    """
    assert G.tables_status()["tables"]
    from irreptables.irreps import IrrepTable

    cases = [("91", False, False), ("152", False, False), ("212", True, False),
             ("76.8", False, True), ("144.5", False, True), ("169.114", True, True)]
    n_old_cf = n_old_cy = n_rows = n_points = n_named_only = 0
    for number_str, spinor, magnetic in cases:
        table = IrrepTable(number_str, spinor, magnetic=magnetic)
        _, conjugated, _ = _tables._table_conjugation(table)
        assert conjugated, f"#{number_str} was chosen because it conjugates"
        t_of = {i + 1: np.asarray(s.t, dtype=float)
                for i, s in enumerate(table.symmetries)}
        r_of = {i + 1: np.asarray(s.R, dtype=float)
                for i, s in enumerate(table.symmetries)}
        by_kp = {}
        for irr in table.irreps:
            by_kp.setdefault(irr.kpname, []).append(irr)
        for kp in conjugated:
            irreps = by_kp[kp]
            inds = sorted(irreps[0].characters)
            k = np.asarray(irreps[0].k, dtype=float)
            su2 = ({i: np.asarray(table.symmetries[i - 1].S, dtype=complex)
                    for i in inds} if spinor else None)
            geom = _rep_geometry({i: r_of[i] for i in inds},
                                 {i: t_of[i] for i in inds}, k, su2)
            phi = np.array([np.exp(-4j * np.pi * k @ t_of[i]) for i in inds])
            chi = {irr.name: np.array([complex(irr.characters[i]) for i in inds])
                   for irr in irreps}
            paired = _tables._matching(chi, lambda v, p=phi: np.conj(v) / p) is not None
            new = _served_rows(table, kp, irreps, True)
            old = _served_rows(table, kp, irreps, False)
            stranded = [name for name in old
                        if min(max(abs(old[name][i] - w[i]) for i in inds)
                               for w in new.values()) > _SWEEP_TOL]
            if paired:                            # the conjugation only renames
                n_named_only += 1
                assert not stranded, (number_str, kp, stranded)
                continue
            n_points += 1
            assert stranded, (
                f"#{number_str} {kp}: every pre-completion row is still one of the "
                "physical characters, so nothing forces the conjugation here")
            for name in new:
                n_rows += 1
                cf, cy = _physical_rep_defect(geom, new[name])
                assert max(cf, cy) < _SWEEP_TOL, (number_str, kp, name)
                ocf, ocy = _physical_rep_defect(geom, old[name])
                n_old_cf += ocf > _SWEEP_TOL
                n_old_cy += ocy > _SWEEP_TOL
                if (number_str, kp) == ("152", "L"):
                    assert ocf < 1e-12, "SG 152's L is invisible to the class function"
                    assert ocy > 0.2, "and it is the cyclic relation that catches it"
    assert (n_points, n_rows, n_named_only) == (13, 36, 3), (
        n_points, n_rows, n_named_only)
    assert n_old_cf > 0 and n_old_cy > 0, (n_old_cf, n_old_cy)


def _identity_gauge_group(number_str, spinor, magnetic):
    """A duck-typed ``GroupData`` presenting one shipped table in its own gauge.

    ``t_ours = t_table`` on every operation, ``sym.sign = 1``, ``refUC = 1`` and
    ``shiftUC = 0`` -- the gauge :func:`_served_rows` computes in.  No cell and
    no model are involved: the operations *are* the table's own rows, so
    :func:`groups._transported_table` can be handed this and asked what it
    serves.
    """
    import types

    from irreptables.irreps import IrrepTable

    table = IrrepTable(number_str, spinor, magnetic=magnetic)
    syms = [types.SimpleNamespace(
        sign=1,
        translation_refUC=(lambda refUC, shiftUC, t=np.asarray(s.t, dtype=float): t))
        for s in table.symmetries]
    group = types.SimpleNamespace(
        number_str=number_str, spinor=spinor, magnetic=magnetic,
        _refUC=np.eye(3), _shiftUC=np.zeros(3),
        _sg=types.SimpleNamespace(symmetries=syms))
    return group, table


def test_the_sweeps_served_rows_are_what_transported_table_serves():
    """The one place the sweep helper is pinned to the library it stands in for.

    :func:`_served_rows` re-implements :func:`groups._transported_table`'s two
    branches in the gauge ``t_ours = t_table``, ``sym.sign = 1``, and every
    sweep in this section uses it, so a change to the transport formula itself
    would otherwise be invisible to all of them.  Here the real function is
    handed a group whose operations *are* one table's, at six k-points chosen so
    that the phase it re-attaches is not 1: ``227``'s ``W`` and ``X`` and
    ``194``'s ``A`` keep the tabulated labelling, with 6 of 8, 8 of 16 and 12 of
    24 little-group operations carrying ``φ = e^{-4πi k·t} ≠ 1``, while
    ``194``'s ``H``, ``152``'s ``L`` and ``80.30``'s ``P`` take the conjugation
    branch -- the last of them onto ``-P4P4``, a corep name the tables do not
    ship, which is also the re-keying :func:`_served_named` performs.

    Both routes agree to **1.6e-16**, under an ulp of the served character's own
    modulus 2: the same product, associated two ways, and not two rules that
    happen to agree at these points.  Reverting the unconjugated branch to
    ``irrep``'s own ``e^{-2πi(t_ours − t_table)·k}`` -- the defect this module
    exists to repair -- moves the served rows at ``227``'s ``W`` and fails here.
    """
    assert G.tables_status()["tables"]

    cases = [("227", False, False, "W", 6, 8), ("227", False, False, "X", 8, 16),
             ("194", False, False, "A", 12, 24), ("194", True, False, "H", 6, 12),
             ("152", False, False, "L", 1, 2), ("80.30", True, True, "P", 0, 2)]
    worst = 0.0
    for number_str, spinor, magnetic, kp, n_phase, n_ops in cases:
        group, table = _identity_gauge_group(number_str, spinor, magnetic)
        c, conjugated, broken = _tables._table_conjugation(table)
        assert kp not in broken, (number_str, kp)
        irreps = [irr for irr in table.irreps if irr.kpname == kp]
        inds = sorted(irreps[0].characters)
        k = np.asarray(irreps[0].k, dtype=float)
        phi = [np.exp(-4j * np.pi * k @ np.asarray(table.symmetries[i - 1].t,
                                                   dtype=float)) for i in inds]
        assert (int(sum(abs(p - 1) > 1e-9 for p in phi)), len(inds)) == (
            n_phase, n_ops), (number_str, kp, phi)

        served = _tables._transported_table(group, kp, np.round(k, 5) % 1)
        local = _served_named(table, kp, irreps, kp in conjugated, c)
        assert sorted(served) == sorted(local), (number_str, kp,
                                                 sorted(served), sorted(local))
        worst = max(worst, max(abs(served[name][i] - local[name][i])
                               for name in served for i in inds))
    assert worst < 8 * _ULP, worst          # measured 1.6e-16, at |χ| = 2


def test_the_two_transport_rules_agree_as_sets_wherever_the_tabulated_one_stands():
    """The consistency check the completion rests on, measured on every table.

    At a k-point where the tabulated labelling *is* kept, the whole-character
    conjugate of each served row is another served row -- that is precisely the
    involution ``c``, and it is why the completion can leave those points
    untouched: the two rules give the **same set** of physical characters there
    and differ only in which BCS name is attached to which.  They agree
    *entry by entry* exactly where the loaded content ``Γ(R)`` is real, and
    differ by ``c`` elsewhere; SG 143's ``K`` (``c: K2 ↔ K3``) is the standing
    example, and the tabulated labelling is the right one there because the
    little group at ``K`` carries no translations at all, so the physical and
    tabulated tables are literally the same operators.

    Nothing in the completion may disturb that -- if the conjugation rule were
    applied at a point where the tabulated labelling stands, SG 143 would start
    reporting ``K3`` for a band whose character is ``e^{2πi/3}``.

    **What carries which half.**  That the two rules give the same *set* is
    :func:`groups._matching`'s own acceptance criterion at these k-points --
    the library reaches the "tabulated labelling stands" branch exactly when it
    has found the pairing, so the row comparison below can only measure how far
    inside its 2e-3 tolerance the arithmetic lands (worst deviation over the 340
    rows: **6.3e-16**).  The teeth are in the **naming**: ``c`` is asserted
    against the shipped row this test finds for itself, by searching the shipped
    set at that k-point for the row printing ``conj(χ_D)`` at the tables' own
    five-decimal precision, with no library matcher consulted (nearest
    non-partner shipped row measured **1.73** away, so the search is not close
    to ambiguous).  That is the "name follows the character" directive itself,
    and it is what pins the *order* of the two naming branches: swap them and
    the nine k-points of ``199``/``208`` where the two matchings exist and
    disagree stop conjugating, land in this loop, and fail here.  At **23** of
    the 340 rows -- the ``P``/``W`` points of ``199``/``227``, which are the
    divergent ones -- no shipped row prints ``conj(χ_D)`` at all; they are
    counted and skipped.
    """
    assert G.tables_status()["tables"]
    from irreptables.irreps import IrrepTable

    checked = agree = unpaired = 0
    worst_set = 0.0
    for number_str, spinor, magnetic in _TRANSPORT_CASES + [
            ("91", False, False), ("152", False, False), ("76.8", False, True)]:
        table = IrrepTable(number_str, spinor, magnetic=magnetic)
        conj, conjugated, _ = _tables._table_conjugation(table)
        by_kp = {}
        for irr in table.irreps:
            by_kp.setdefault(irr.kpname, []).append(irr)
        for kp, irreps in by_kp.items():
            if kp in conjugated:
                continue
            inds = sorted(irreps[0].characters)
            chi = {irr.name: np.array([complex(irr.characters[i]) for i in inds])
                   for irr in irreps}
            new = _served_rows(table, kp, irreps, True)
            old = _served_rows(table, kp, irreps, False)
            for name in old:
                checked += 1
                #  the shipped row that prints what the physical convention
                #  prints -- found here, in the shipped set, not asked for
                hits = [other for other, vec in chi.items()
                        if np.abs(vec - np.conj(chi[name])).max() < _TABLE_PRECISION]
                if len(hits) != 1:
                    unpaired += 1                 # divergent: the book has no such row
                else:
                    assert hits[0] == conj[name], (
                        f"#{number_str} {kp}: the served name of {name} is "
                        f"{conj[name]}, but the shipped row printing conj(χ) is "
                        f"{hits[0]}")
                    agree += hits[0] == name
                #  ... and the two rules do give the same set of rows here
                worst_set = max(worst_set, min(
                    max(abs(old[name][i] - new[other][i]) for i in inds)
                    for other in new))
    assert checked > 300 and agree > 200, (checked, agree)
    assert (checked, agree, unpaired) == (340, 245, 23), (checked, agree, unpaired)
    assert worst_set < 1e-12, worst_set


# ---------------------------------------------------------------------------
# the fourth face (2026-08-21): the time-reversal gluing is convention-sensitive
# ---------------------------------------------------------------------------
#
# Plain conjugation of whole character vectors is the wrong closure test for a
# grey table at a k-point that is not a TRIM: the antiunitary that pairs the
# irreps there is not time reversal but some `g.T`, and the right relation is
# the A-TWISTED conjugation `Dbar(h) = conj(D(A^-1 h A))`.  The Wigner-Herring
# indicator built out of it decides which unitary irrep doubles -- and every
# one of its terms carries `e^{s.2pi.i k.m}`, so the answer is
# convention-sensitive.  These tests measure that machinery against
# irreptables' own shipped corep lists, in both conventions, on every table.

@pytest.mark.data
def test_the_A_twisted_gluing_reproduces_every_shipped_grey_corep_table(shipped_tables):
    r"""The Wigner-Herring sweep: 460 grey instances, 2700 k-points, no exceptions.

    Six statements per tabulated k-point, all from the table's own numbers:

    1. the little group predicted from the *centering* -- ``R^{-T}k ≡ k``
       modulo the primitive reciprocal lattice, which for a body-centred group
       is the ``h+k+l`` even rule -- is exactly ``irreptables``' own tabulated
       little-group index list.  This is what makes ``P = (½,½,½)`` of SG 80 a
       non-TRIM and puts ``C4·T`` rather than ``T`` in its extended little
       group;
    2. the antiunitary coset either is empty (no antiunitary fixes ``k`` at
       all -- the ``K``/``KA`` "other arm" situation) or has exactly ``|H|``
       elements;
    3. every shipped corep decomposes into the **parent unitary table's**
       irreps with non-negative integer multiplicities;
    4. the composite-name rule of :func:`groups._corep_name` -- concatenate the
       constituents in lexicographic order, carry the ``-`` once -- reproduces
       the shipped name, all 2700 of them (this is why the tables print
       ``-A11A9`` and not ``-A9A11``);
    5. the A-twisted conjugation ``Dbar(h) = conj(D(A^-1 h A))`` permutes the
       unitary irreps and is an involution -- built in the test from
       ``_herring_data``'s ``twist`` and checked as a permutation (each twisted
       character matches exactly one shipped row: worst deviation **6.3e-16**,
       nearest other row **1.73** away) and as an involution, at all 8689
       (k-point, irrep) pairs of the 2638 glued points, so that the gluing's own
       internal guards are not the only thing saying so;
    6. the **BCS Herring indicators** ``I_{+1}(D) = (1/|H|) Σ_a χ_D(a²)``
       reproduce the shipped gluing exactly -- ``+1`` alone, ``−1`` doubled,
       ``0`` paired with ``Dbar``.

    Statement 6 is the one that retracts ``80.30``-``P`` from the upstream
    data-defect list: its shipped ``-P3P3 ⊕ -P4`` is precisely what BCS's own
    convention requires there, so ``irreptables`` is faithful and it is the
    *transport* that owed the point an answer.
    """
    _unitary, grey = _sweep_instances()
    n_points = n_glued = n_names = n_twisted = 0
    for number_str, spinor, magnetic in grey:
        table = shipped_tables[(number_str, spinor, magnetic)]
        parent = _tables._parent_table(number_str, spinor)
        centering = _spacegroup._centering(table)
        by_u, by_g = {}, {}
        for irr in parent.irreps:
            by_u.setdefault(irr.kpname, []).append(irr)
        for irr in table.irreps:
            by_g.setdefault(irr.kpname, []).append(irr)
        for kp, girreps in sorted(by_g.items()):
            if kp not in by_u:                    # a grey-only 'A' arm
                continue
            n_points += 1
            uirreps = by_u[kp]
            k = np.asarray(uirreps[0].k, dtype=float)
            H, coset, squares, twist = _tables._herring_data(table, k, centering)
            assert sorted(H) == sorted(int(i) for i in uirreps[0].characters), (
                number_str, kp, H)                                    # 1
            assert len(coset) in (0, len(H)), (number_str, kp, coset)  # 2
            bcs = {irr.name: {int(i): complex(v)
                              for i, v in irr.characters.items()}
                   for irr in uirreps}
            shipped = []
            for irr in girreps:
                parts = _tables._decompose_into(
                    bcs, H, {int(i): complex(v) for i, v in irr.characters.items()})
                assert parts is not None, (number_str, kp, irr.name)   # 3
                assert _tables._corep_name(parts, spinor) == irr.name, (     # 4
                    number_str, kp, irr.name, parts)
                n_names += 1
                shipped.append(tuple(sorted(parts)))
            if not coset:
                assert sorted(shipped) == sorted((n,) for n in bcs), (number_str, kp)
                continue
            n_glued += 1
            #  5: the A-twisted conjugation, built here from `twist` -- a
            #  permutation of the unitary irreps and an involution, asserted
            #  rather than left to `_corep_gluing`'s own internal guards
            partner = {}
            for name, row in bcs.items():
                bar = {h: np.conj(sig * np.exp(2j * np.pi * k @ m) * row[ell])
                       for h, (ell, m, sig) in twist.items()}
                hits = [other for other, r in bcs.items()
                        if max(abs(bar[h] - r[h]) for h in H) < _TABLE_PRECISION]
                assert len(hits) == 1, (number_str, kp, name, hits)
                partner[name] = hits[0]
                n_twisted += 1
            assert all(partner[partner[n]] == n for n in partner), (
                number_str, kp, partner)
            want = _tables._corep_gluing(bcs, k, H, squares, twist, +1)
            assert want is not None, (number_str, kp)                  # 5
            assert sorted(v for v, _ in want) == sorted(shipped), (     # 6
                f"#{number_str} spinor={spinor} {kp}: BCS's own Herring "
                f"indicators give {sorted(v for v, _ in want)} but the shipped "
                f"corep list is {sorted(shipped)}")
    assert (n_points, n_glued, n_names) == (2700, 2638, 7407), (
        n_points, n_glued, n_names)
    assert n_twisted == 8689, n_twisted


@pytest.mark.data
def test_every_served_corep_name_spells_its_own_unitary_constituents(shipped_tables,
                                                                     shipped_naming):
    """The invariance the re-gluing exists to restore, over all 460 grey tables.

    A grey corep name is not a tag: ``-P3P3`` *means* ``-P3 ⊕ -P3``.  So the
    served corep row named ``L`` must decompose, against the served **unitary**
    table of the same group at the same k-point, into exactly the irreps ``L``
    spells.  That statement ties the two tables this package hands out to one
    another, and it is the physical content of the whole mechanism: a band whose
    grey label says ``-P3P3`` must restrict to two copies of the band the
    unitary table calls ``-P3``.

    Measured: it holds at all 2700 shared (grey group, valuedness, k-point)
    entries -- and it is what pins the naming, in both directions.  Where the
    served corep row *does* have a BCS name with its characters, that name is
    the one the spelling forces too (BCS's own names spell BCS's own
    constituents, so the two rules agree by construction); where the two
    disagree the lift wins, which is 203's spinful ``W``, the one point of the
    whole sweep where a grey exchange is refused because the parent unitary
    table has no exchange to induce it.
    """
    checked, bad = 0, []
    for number_str, spinor, magnetic in _sweep_instances()[1]:
        table = shipped_tables[(number_str, spinor, magnetic)]
        parent = _tables._parent_table(number_str, spinor)
        cg, conj_g, broken_g = shipped_naming[(number_str, spinor, magnetic)]
        cu, conj_u, _ = shipped_naming[(number_str.split(".")[0], spinor, False)]
        by_u = {}
        for irr in parent.irreps:
            by_u.setdefault(irr.kpname, []).append(irr)
        by_g = {}
        for irr in table.irreps:
            by_g.setdefault(irr.kpname, []).append(irr)
        for kp, girreps in sorted(by_g.items()):
            if kp not in by_u or kp in broken_g:
                continue
            checked += 1
            H = sorted(int(i) for i in by_u[kp][0].characters)
            #  what `_transported_table` serves, keyed by the name it serves it
            #  under -- on *both* tables, since the served unitary name follows
            #  the printed character
            served_u = _served_named(parent, kp, by_u[kp], kp in conj_u, cu)
            served_g = _served_rows(table, kp, girreps, kp in conj_g)
            for irr in girreps:
                name = cg[irr.name] if kp in conj_g else irr.name
                parts = _tables._decompose_into(served_u, H, served_g[irr.name])
                if parts is None or _tables._corep_name(parts, spinor) != name:
                    bad.append((number_str, spinor, kp, name, parts))
    assert checked == 2700, checked
    assert bad == [], bad[:8]


@pytest.mark.data
def test_every_group_transports_and_the_only_refusal_is_a_data_defect():
    """Zero refusals left in the sweep; the one survivor is an ``irreptables`` bug.

    Over the 920 instances ``_to_physical_convention`` now completes for **all**
    460 unitary and **all** 460 grey ones.  ``80.30`` used to be the exception,
    refused because its shipped corep list at ``P`` is not closed under plain
    conjugation -- which is true, and is not a defect: the gluing there is
    convention-sensitive, so the physical corep list doubles the *other* irrep
    (``-P3 ⊕ -P4P4`` against BCS's ``-P3P3 ⊕ -P4``) and no whole-character
    relabelling could have expressed it.  What is left is ``80.31``, a type-III group outside the sweep
    whose EBR entries ship with no ``wyckoff_position`` at all -- a plain
    missing field, and the only remaining upstream data defect.

    And where a unitary group's EBR rows list a point ``-k`` its character
    table does not -- ``KA``, ``HA``, ``PA``, ``WA``, in 40 tables of 20
    space-group types -- every partner label is the conjugate of the listed
    label the rows give it, read off the rows alone, and it is the tables' own
    naming, ``XAn`` the conjugate of ``Xn``, at all 62 such points.  That
    naming is ``irreptables``' own: its type-I magnetic tables list the
    partner points themselves, and every one of the 218 labels there is the
    conjugate of its listed partner at ``-k`` -- including the ten points
    (``#144``, ``#145``, ``#159`` at ``K`` and ``H``) whose rows do not tell
    their labels apart.
    """
    assert G.tables_status()["tables"]
    from irreptables.ebrs import load_ebr_data

    unitary, grey = _sweep_instances()
    refused = {}
    for number_str, spinor, magnetic in unitary + list(grey):
        try:
            _tables._to_physical_convention(load_ebr_data(number_str, spinor),
                                      number_str, spinor, magnetic)
            _tables._transport_record(number_str, spinor, magnetic)
        except _spacegroup._EBRTransportError as exc:
            refused[(number_str, spinor, magnetic)] = str(exc)

    assert refused == {}, sorted(refused)

    partners = {(number_str, spinor): _tables._partner_points(number_str, spinor)
                for number_str, spinor, _magnetic in unitary}
    partners = {key: arms for key, arms in partners.items() if arms}
    assert (len(partners), sum(len(arms) for arms in partners.values())) == (40, 62)
    assert {int(number_str) for number_str, _spinor in partners} == {
        23, 24, 82, 121, 122, 143, 144, 145, 150, 152, 154, 157, 159, 174, 189, 190,
        197, 199, 217, 220}
    assert all(theirs == ours.replace(base, arm, 1)
               for arms in partners.values() for arm, base, naming in arms
               for ours, theirs in naming), "the tables' naming at every partner point"

    #  ... checked against the type-I magnetic table of each group, the first
    #  serial of its family, which lists both points of every pair
    import glob
    import os
    import re

    import irreptables
    from irreptables.irreps import IrrepTable

    folder = os.path.join(os.path.dirname(irreptables.__file__), "data", "correptables")
    serials: dict[str, set] = {}
    for path in glob.glob(os.path.join(folder, "*.dat")):
        found = re.match(r"irreps-SG=(\d+)\.(\d+)-", os.path.basename(path))
        if found:
            serials.setdefault(found.group(1), set()).add(int(found.group(2)))
    checked = 0
    for (number_str, spinor), arms in partners.items():
        table = IrrepTable(f"{number_str}.{min(serials[number_str])}", spinor, magnetic=True)
        assert not any(s.time_reversal for s in table.symmetries), number_str
        rows = {(irr.kpname, irr.name): irr for irr in table.irreps}
        for arm, base, naming in arms:
            for ours, theirs in naming:
                x, xa = rows[(base, ours)], rows[(arm, theirs)]
                at = np.asarray(x.k, dtype=float) + np.asarray(xa.k, dtype=float)
                assert np.abs(at - np.round(at)).max() < 1e-4, (number_str, theirs)
                assert set(xa.characters) == set(x.characters), (number_str, theirs)
                assert all(abs(complex(xa.characters[g]) - np.conj(complex(x.characters[g])))
                           < _SWEEP_TOL for g in x.characters), (number_str, spinor, theirs)
                checked += 1
    assert checked == 218

    with pytest.raises(_spacegroup._EBRTransportError) as exc:
        _tables._to_physical_convention(load_ebr_data("80.31", False), "80.31", False, True)
    assert "wyckoff_position" in str(exc.value), "the one remaining data defect"

    # and 80.30 -- the retracted one -- transports, conserving dimension at P
    raw = _tables._to_physical_convention(load_ebr_data("80.30", True), "80.30", True, True)
    labels = list(raw["basis"]["irrep_labels"])
    degen = {l: int(d) for l, d in zip(labels, raw["basis"]["degeneracies"])}
    assert [l for l in labels if l.startswith("-P")] == ["-P4P4", "-P3"]
    assert (degen["-P4P4"], degen["-P3"]) == (2, 1)
    row = dict(zip(labels, (int(v) for v in raw["ebrs"][0]["vector"])))
    assert row["-P4P4"] == 1 and row["-P3"] == 2
    assert sum(row[l] * degen[l] for l in labels if l.startswith("-P")) == 4


def test_the_transport_record_publishes_where_the_characters_were_conjugated():
    """``ebr_data(...)["convention"]`` is the public census, in two fields.

    A reader cross-checking a character against Bilbao needs two different
    things, and since the "name follows the character" directive (2026-08-24)
    they are no longer the same set:

    * ``conjugated_points`` -- where the served row is the whole-character
      conjugate.  The *name* served with it is the BCS name of the row that
      comes out, so the printed table there is still BCS's own and the reader
      will find the numbers exactly where the book puts them.  ``c`` is
      correspondingly no longer the identity there: it is that name exchange,
      and it is what the EBR relabelling carries;
    * ``divergent_points`` -- where the printed table is not a permutation of
      BCS's rows at all, so the reader *will* see numbers the book does not
      print under that name.  These are the ``P``/``W`` points where the shipped
      page tabulates the other star member ``−k``.  The two fields are
      independent: ``227``'s ``W`` is divergent without being conjugated (the
      tabulated ``Γ`` rule stands there and still lands on a character set that
      is not BCS's), while ``91``'s ``A``/``Z`` are conjugated without being
      divergent (the rows are BCS's, only exchanged).

    ``143`` stays empty in both -- the anchor that says the directive did not
    disturb the k-points where the tabulated table is already the physical one,
    and where the convention shows up only in the EBR pairing (``K2``/``K3``).
    """
    assert G.tables_status()["tables"]

    for number_str, spinor, magnetic, want, diverge in (
            ("91", False, False, ("A", "Z"), ()),
            ("152", False, False, ("A", "H", "L"), ()),
            ("152.34", False, True, ("A", "H", "HA", "L"), ()),
            ("143", False, False, (), ()),
            ("227", False, False, (), ("W",)),
            ("227.129", False, True, (), ("W",)),
            ("199", True, False, ("N",), ("P",)),
            ("171", True, False, ("A", "H"), ()),
            ("80.30", True, True, ("M", "P"), ("P",))):
        record = _tables._transport_record(number_str, spinor, magnetic)
        assert tuple(record["conjugated_points"]) == want, number_str
        assert tuple(record["divergent_points"]) == diverge, number_str
        #  (`_conjugated_kpoints`/`_divergent_kpoints` are where the record's two
        #  fields come from, so re-reading them here would say nothing new)
        #  the basis moves only where the gluing has no shipped name at all
        assert set(record["reglued_points"]) <= set(diverge), number_str


# ---------------------------------------------------------------------------
# the naming policy (2026-08-24): the served name follows the printed character
# ---------------------------------------------------------------------------


@pytest.mark.data
def test_the_served_table_is_BCS_printed_table_wherever_a_bijection_exists(shipped_tables,
                                                                           shipped_naming):
    r"""The directive, swept: same names against the same numbers as the book.

    Owner's directive of 2026-08-24: the served names follow the served
    *characters*, so that wherever the transport exchanged rows the labels are
    exchanged with them.  The consequence asserted here, over all 460 unitary
    and all 460 grey instances: at every tabulated k-point whose shipped
    character set is closed under complex conjugation, the served table equals
    ``irreptables``' shipped table **row for row, name and numbers**.  Measured:
    5403 of the 5462 entries, 16 361 rows; the other 59 are the ``divergent``
    ones, where no such bijection exists at all (:func:`test_at_a_divergent_point_the_served_table_conjugates_onto_the_shipped_one`).

    ``171.122``'s spinful ``H`` is the anchor, and it is the case the directive
    was written for.  Before it the served coreps there were named by the
    re-gluing lift, ``{-H6 (1,−1,+1), -H4H5 (2,+1,−1)}`` -- character-identical
    to BCS's ``{-H4, -H5H6}`` but with the names exchanged, so the served page
    said "``-H6`` stands alone, ``-H4H5`` is glued" where BCS's book says the
    opposite.  With the names following the characters the served page *is*
    BCS's page, and the whole convention divergence at that k-point is which of
    those two rows a physical band matches.  The rename is forced up into the
    parent unitary table by the spelling invariant -- ``-H5H6`` can only mean
    ``-H5 ⊕ -H6`` -- and SG 171's ``H`` moves with it (``-H6→-H4``,
    ``-H5→-H6``, ``-H4→-H5``), which is asserted here too.
    """
    from irreptables.irreps import IrrepTable

    unitary, grey = _sweep_instances()
    n_points = n_rows = 0
    demoted = []
    for number_str, spinor, magnetic in unitary + list(grey):
        table = shipped_tables[(number_str, spinor, magnetic)]
        c, conjugated, broken = shipped_naming[(number_str, spinor, magnetic)]
        by_kp = {}
        for irr in table.irreps:
            by_kp.setdefault(irr.kpname, []).append(irr)
        for kp, irreps in by_kp.items():
            if kp in broken or kp in _tables._divergent_kpoints(
                    number_str, spinor, magnetic):
                continue
            inds = sorted(irreps[0].characters)
            served = _served_named(table, kp, irreps, kp in conjugated, c)
            shipped = _served_rows(table, kp, irreps, False)   # χ·φ, not χ
            phi = {i: np.exp(-4j * np.pi
                             * np.asarray(irreps[0].k, dtype=float)
                             @ np.asarray(table.symmetries[i - 1].t, dtype=float))
                   for i in inds}
            n_points += 1
            assert sorted(served) == sorted(shipped), (number_str, kp)
            bad = [name for name in served
                   if max(abs(served[name][i] - shipped[name][i] / phi[i])
                          for i in inds) >= _SWEEP_TOL]
            if bad:                        # the spelling invariant vetoed the rename
                demoted.append((number_str, spinor, magnetic, kp))
                continue
            n_rows += len(served)
    assert (n_points, n_rows) == (5403, 16330), (n_points, n_rows)

    #  the one veto in the whole sweep: SG 203's spinful W.  The grey rows there
    #  *are* closed under conjugation, so a name exchange exists on the corep
    #  page -- but the parent unitary table's rows are not, so its irreps cannot
    #  move, and `-W2W4` can only be renamed `-W3W5` if `-W2`/`-W4` can be
    #  renamed `-W3`/`-W5`.  The spelling invariant wins and the exchange is
    #  refused (this is the "demote to class 3" case of the census).
    assert demoted == [("203.27", True, True, "W")], demoted
    parent = _tables._parent_table("203.27", True)
    wp = [irr for irr in parent.irreps if irr.kpname == "W"]
    winds = sorted(wp[0].characters)
    chi = {irr.name: np.array([complex(irr.characters[i]) for i in winds])
           for irr in wp}
    assert _tables._matching(chi, np.conj) is None, (
        "SG 203's unitary W is exactly the case with no exchange to induce")

    #  the anchor, in numbers: 171.122's spinful H, grey and unitary alike
    table = shipped_tables[("171.122", True, True)]
    c, conjugated, _ = shipped_naming[("171.122", True, True)]
    irreps = [irr for irr in table.irreps if irr.kpname == "H"]
    served = _served_named(table, "H", irreps, "H" in conjugated, c)
    inds = sorted(irreps[0].characters)
    assert sorted(served) == ["-H4", "-H5H6"], sorted(served)
    assert [round(served["-H4"][i].real, 6) for i in inds] == [1.0, -1.0, 1.0]
    assert [round(served["-H5H6"][i].real, 6) for i in inds] == [2.0, 1.0, -1.0]
    assert max(abs(served[n][i].imag) for n in served for i in inds) < 1e-9
    #  ... and the parent unitary rename the spelling invariant forces
    parent = _tables._parent_table("171.122", True)
    cu, conj_u, _ = _tables._table_conjugation(parent)
    assert "H" in conj_u, "SG 171's H had to start conjugating to carry the rename"
    assert {d: cu[d] for d in cu if _tables._label_kpoint(d) == "H"} == {
        "-H4": "-H4", "-H5": "-H6", "-H6": "-H5"}


@pytest.mark.data
def test_at_a_divergent_point_the_served_table_conjugates_onto_the_shipped_one(
        shipped_tables, shipped_naming):
    r"""The 59 irreducible entries: the shipped page is the **other star member**.

    Where the shipped character set is not closed under conjugation, no
    permutation of BCS's rows is the physical table, so the served page really
    does print numbers the book does not print.  What it *is* is stated exactly
    by ``χ_phys[c(D)] = conj(χ_BCS[D])`` read backwards: the entrywise conjugate
    of the served table is the shipped table, row for row, under the bijection
    ``c``.  Time reversal maps the ``k`` sector antiunitarily onto the ``−k``
    one, so that identity says the shipped page tabulates the content of the
    other star member ``−k`` and the served page tabulates ``k``'s.  Two
    conventions are not disagreeing about a doubling; two *star members* are
    being described.

    ``80.30``'s spinful ``P`` is the anchor and nothing about it moved with the
    2026-08-24 naming directive, which is the point of asserting it here:

    * the unitary rows are BCS's under BCS's names in both conventions,
      ``-P3 = (1, −i)`` and ``-P4 = (1, +i)``;
    * the served coreps are ``{-P3 (dim 1), -P4P4 (2, +2i)}`` where BCS ships
      ``{-P3P3 (2, −2i), -P4 (dim 1)}``, and conjugating the served pair gives
      the shipped pair exactly;
    * so a renaming is inadmissible: the doubling itself is a
      convention-independent observable -- the trace of the zero-translation
      little-group generator on the degenerate level, which the ``i41_soc``
      demo measures as ``+2i`` -- and there is no shipped name carrying it.

    **Where the row identity has content and where it does not.**  The 59 split
    by naming branch, and the split is asserted because the identity
    ``conj(served[c(D)]) = χ_BCS[D]`` is not one statement:

    * **46 points, 160 rows** keep the tabulated labelling (not in
      ``conjugated``): the served row is ``χ_D·φ`` under ``D``'s own name and
      the identity is :func:`groups._matching`'s postcondition, measured
      9.8e-16 inside its 2e-3 acceptance;
    * **7 points, 19 rows** are the re-glued class, where the *parent unitary*
      table renamed the constituents at that k-point and the corep name has to
      follow them (:func:`groups._lifted_conjugation`).  There the served page
      leaves the book outright, and that is asserted per row: every shipped
      name's image differs from it and is a name the book does not print at
      that k-point.  With the constituent map dropped from the re-gluing the
      served names stop moving and these 19 rows fail;
    * **6 points, 21 rows** -- ``122``/``122.334``'s ``P`` and ``PA`` -- reach
      the last-resort identity naming, and there the entrywise identity really
      is ``conj(conj(χ)) = χ`` and says nothing.  What is asserted instead is
      the divergence itself: all 21 rows are served with numbers the book does
      not print under that name (they are the conjugates).

    **The re-glued class, entry by entry.**  Its seven entries lie in six space
    groups, every one a **grey** table -- the unitary tables have no
    time-reversal gluing to get wrong -- and they are exactly the
    ``reglued_points`` the transport record publishes.  The physical corep list
    there is a partition of the same unitary irreps that BCS's book has no name
    for, so the per-k-point dimension is conserved while the individual corep
    dimensions need not be (``80.30`` ``P``: ``2+1 ↔ 1+2``).  The pairing
    antiunitary is never pure time reversal: every member is a non-TRIM ``P``
    or ``W`` point of a body- or face-centred group, with a non-empty
    antiunitary coset.
    """
    unitary, grey = _sweep_instances()
    n_points = n_rows = 0
    worst = 0.0
    kept = renamed = fallback = (0, 0)
    reglued, published = [], {}
    for number_str, spinor, magnetic in unitary + list(grey):
        divergent = _tables._divergent_kpoints(number_str, spinor, magnetic)
        if not divergent:
            continue
        table = shipped_tables[(number_str, spinor, magnetic)]
        c, conjugated, broken = shipped_naming[(number_str, spinor, magnetic)]
        if magnetic:
            record = _tables._transport_record(number_str, spinor, magnetic)
            published[number_str, spinor] = record
            assert set(record["reglued_points"]) <= divergent, number_str
        by_kp = {}
        for irr in table.irreps:
            by_kp.setdefault(irr.kpname, []).append(irr)
        for kp in sorted(divergent):
            if kp in broken:
                continue
            n_points += 1
            irreps = by_kp[kp]
            inds = sorted(irreps[0].characters)
            served = _served_named(table, kp, irreps, kp in conjugated, c)
            shipped = {irr.name: {i: complex(irr.characters[i]) for i in inds}
                       for irr in irreps}
            #  no shipped name matches a served row -- that is what divergent means
            chi = {n: np.array([shipped[n][i] for i in inds]) for n in shipped}
            assert _tables._matching(chi, np.conj) is None, (number_str, kp)
            #  did the *parent unitary* table rename its irreps here?  Then the
            #  corep name spells constituents that moved, and must move too.
            moved_parent = False
            if magnetic:
                cu, _conj_u, _broken_u = _tables._unitary_conjugation(number_str, spinor)
                moved_parent = any(cu.get(irr.name, irr.name) != irr.name
                                   for irr in _tables._parent_table(number_str,
                                                              spinor).irreps
                                   if irr.kpname == kp)
            #  ... and conj(served[c(D)]) is chi_BCS[D], row for row
            for name in shipped:
                n_rows += 1
                image = c.get(name, name)
                assert image in served, (number_str, kp, name, image)
                dev = max(abs(np.conj(served[image][i]) - shipped[name][i])
                          for i in inds)
                worst = max(worst, dev)
                assert dev < _SWEEP_TOL, (number_str, kp, name, image, dev)
                if kp not in conjugated or not moved_parent:
                    continue
                assert image != name and image not in shipped, (
                    f"#{number_str} {kp}: the parent unitary table renamed this "
                    f"corep's constituents, so {name} cannot still be served as "
                    f"{image}")
            if kp not in conjugated:
                kept = (kept[0] + 1, kept[1] + len(shipped))
            elif moved_parent:
                renamed = (renamed[0] + 1, renamed[1] + len(shipped))
                assert sorted(served) != sorted(shipped), (number_str, kp)
                reglued.append((number_str, spinor, kp))
                #  never a TRIM, so pure time reversal is in no member's extended
                #  little group and the pairing antiunitary is some g.T
                centering = _spacegroup._centering(table)
                k = np.asarray(irreps[0].k, dtype=float)
                _H, coset, _sq, _tw = _tables._herring_data(table, k, centering)
                assert coset, (number_str, kp)
                assert not _spacegroup._is_reciprocal(-2 * k, centering), (number_str, kp)
            else:
                fallback = (fallback[0] + 1, fallback[1] + len(shipped))
                #  the identity naming: the content left is that the numbers moved
                for name in shipped:
                    assert max(abs(served[name][i] - shipped[name][i])
                               for i in inds) > _SWEEP_TOL, (number_str, kp, name)
    assert (n_points, n_rows) == (59, 200), (n_points, n_rows)
    assert (kept, renamed, fallback) == ((46, 160), (7, 19), (6, 21)), (
        kept, renamed, fallback)
    #  machine precision, not table precision: the served row *is* the shipped
    #  row conjugated, entry by entry, at every one of the 59
    assert worst < 1e-12, worst

    #  the re-glued class is the transport record's `reglued_points`, all grey
    assert sorted(reglued) == sorted(
        (number_str, spinor, kp) for (number_str, spinor), record in published.items()
        for kp in record["reglued_points"]) == [
        ("109.240", True, "P"), ("110.246", True, "P"),
        ("210.53", True, "W"),
        ("80.30", True, "P"),
        ("88.82", False, "P"), ("88.82", True, "P"),
        ("98.158", True, "P"),
    ], sorted(reglued)
    #  ... and its anchors, spelled out: they did not move with the directive
    for (number_str, kp), want in {
            ("80.30", "P"): (("-P3P3", "-P4"), ("-P4P4", "-P3")),
            ("210.53", "W"): (("-W2W4", "-W3", "-W5"), ("-W3W5", "-W4", "-W2"))}.items():
        conjugation = published[number_str, True]["conjugation"]
        at_kp = [label for label in conjugation if _tables._label_kpoint(label) == kp]
        assert (tuple(at_kp), tuple(conjugation[label] for label in at_kp)) == want

    #  the anchor, in numbers
    parent = _tables._parent_table("80.30", True)
    cu, conj_u, _ = _tables._table_conjugation(parent)
    up = [irr for irr in parent.irreps if irr.kpname == "P"]
    uinds = sorted(up[0].characters)
    served_u = _served_named(parent, "P", up, "P" in conj_u, cu)
    assert [complex(round(served_u["-P3"][i].real, 6),
                    round(served_u["-P3"][i].imag, 6)) for i in uinds] == [1, -1j]
    assert [complex(round(served_u["-P4"][i].real, 6),
                    round(served_u["-P4"][i].imag, 6)) for i in uinds] == [1, +1j]

    table = shipped_tables[("80.30", True, True)]
    c, conjugated, _ = shipped_naming[("80.30", True, True)]
    gp = [irr for irr in table.irreps if irr.kpname == "P"]
    inds = sorted(gp[0].characters)
    served = _served_named(table, "P", gp, "P" in conjugated, c)
    assert sorted(served) == ["-P3", "-P4P4"], sorted(served)
    assert [complex(round(served["-P3"][i].real, 6),
                    round(served["-P3"][i].imag, 6)) for i in inds] == [1, -1j]
    assert [complex(round(served["-P4P4"][i].real, 6),
                    round(served["-P4P4"][i].imag, 6)) for i in inds] == [2, +2j]
    shipped = {irr.name: {i: complex(irr.characters[i]) for i in inds} for irr in gp}
    assert sorted(shipped) == ["-P3P3", "-P4"], sorted(shipped)
    assert [round(shipped["-P3P3"][i].imag, 6) for i in inds] == [0.0, -2.0]
    assert [round(shipped["-P4"][i].imag, 6) for i in inds] == [0.0, 1.0]


# ---------------------------------------------------------------------------
# the spinor branch (2026-08-25): the table's frame, not irrep's random fit
# ---------------------------------------------------------------------------
#
# `irrep.spacegroup_irreps.match_spinor_rotations` obtains the spin-basis
# transformation that matches our operations' SU(2) lifts to the tables' by
# `scipy.optimize.minimize` from an UNSEEDED `np.random.random` start, and the
# residual it minimises takes a `min` over the sign of each operation -- so both
# branches of the fit are exact minima and `sym.sign` came out differently from
# run to run.  The character ROWS are blind to it (twisting the tabulated set by
# a linear character permutes it), but the name-to-row pairing is not, so a
# per-level corep name was not reproducible.  `groups._canonical_spinor_signs`
# computes the sign instead, from the Cartesian frame the tables' own shipped
# `S` matrices are written in; these are its regressions.

_SPIN_SWEEP_G0 = np.array([[1.000, 0.113, 0.071],      # a generic positive-definite
                           [0.113, 1.317, 0.053],      # seed for the Reynolds average
                           [0.071, 0.053, 1.771]])     # below -- no crystal-system table


def _spin_sweep_rotation():
    """A fixed generic rotation, so the frame map is never accidentally the identity."""
    axis = np.array([0.3, -0.7, 0.64])
    axis = axis / np.linalg.norm(axis)
    ang = 0.9137
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


def _synthetic_spinor_group(number_str, magnetic, table=None):
    """A duck-typed ``SpaceGroupIrreps`` for one shipped table, with no model.

    The cell is built from the table's own rotations: the Reynolds average
    ``(1/|G|) Σ Rᵀ G₀ R`` of a generic positive-definite ``G₀`` is invariant
    under every one of them and generically has exactly the symmetry they
    require, so its Cholesky factor is a conventional cell the table fits --
    for all 230 space groups, with no crystal-system lookup.  It is then turned
    by a fixed generic rotation so that the frame map ``_canonical_spinor_signs``
    derives is never trivially the identity.

    The operations are built by ``irrep``'s **own**
    ``symmetry_operation.SymmetryOperation``, so ``spinor_rotation`` is literally
    the lift this package is handed: ``exp(−i θ/2 n·σ)`` with ``(n, θ)`` from an
    eigen-decomposition of the Cartesian rotation.  That matters for the lift-set
    layer -- ``np.linalg.eig`` fixes ``n`` only up to a sign, so the set is a
    *per-operation* arbitrary sign away from the tables', which is exactly the
    inconsistency ``_pin_spinor_branch`` corrects and this sweep measures.
    """
    import types

    from irrep.symmetry_operation import SymmetryOperation
    from irreptables.irreps import IrrepTable

    if table is None:
        table = IrrepTable(number_str, True, magnetic=magnetic)
    tsyms = table.u_symmetries + table.au_symmetries
    rots = [np.asarray(t.R, dtype=float) for t in tsyms]
    metric = sum(R.T @ _SPIN_SWEEP_G0 @ R for R in rots) / len(rots)
    lattice = np.linalg.cholesky(metric) @ _spin_sweep_rotation().T
    ops = [SymmetryOperation(np.asarray(t.R, dtype=float),
                             np.asarray(t.t, dtype=float), lattice, spinor=True,
                             time_reversal=bool(getattr(t, "time_reversal", False)))
           for t in tsyms]
    return types.SimpleNamespace(number_str=number_str, magnetic=magnetic, spinor=True,
                                 real_lattice=lattice, refUC=np.eye(3), symmetries=ops)


@pytest.fixture(scope="module")
def synthetic_spinor_groups(shipped_tables):
    """``{(number_str, magnetic): group}`` for the 460 double-valued tables, built once."""
    unitary, grey = _sweep_instances()
    return {(number_str, magnetic): _synthetic_spinor_group(
                number_str, magnetic, shipped_tables[(number_str, True, magnetic)])
            for number_str, spinor, magnetic in unitary + list(grey) if spinor}


def _canonical_with_frames(sg, frames):
    """``_canonical_spinor_signs`` restricted to ``frames``; ``None`` if none applies."""
    saved = _spacegroup._TABLE_SPIN_FRAMES
    try:
        _spacegroup._TABLE_SPIN_FRAMES = tuple(frames)
        return _spacegroup._canonical_spinor_signs(sg)
    except InternalError:
        return None
    finally:
        _spacegroup._TABLE_SPIN_FRAMES = saved


def _perp_two_fold(sg):
    """Per operation: is the proper part a two-fold about an axis perpendicular to ``z``?

    Read in the conventional cell's **standard orientation** — the frame the
    twist ``diag(-1, -1, 1)`` is a two-fold about ``z`` of — which is the
    synthetic cell's Cholesky factor, not its turned copy.
    """
    lattice = np.asarray(sg.real_lattice, dtype=float)
    standard = np.linalg.cholesky(lattice @ lattice.T)
    inv = np.linalg.inv(standard).T
    out = []
    for op in sg.symmetries:
        cart = standard.T @ np.asarray(op.rotation, dtype=float) @ inv
        proper = cart if np.linalg.det(cart) > 0 else -cart
        if abs(np.trace(proper) + 1.0) > 1e-6:       # trace -1 <=> a two-fold rotation
            out.append(False)
            continue
        axis = (proper + np.eye(3)) / 2              # = n nᵀ for a two-fold
        out.append(bool(axis[2, 2] < 1e-6))
    return np.array(out)


@pytest.mark.data
def test_the_spinor_branch_is_pinned_for_every_shipped_table(synthetic_spinor_groups):
    """The totality sweep: **every** double-valued table ``irreptables`` ships.

    460 instances -- the 230 space groups and the 230 type-II grey groups, in
    the spinful valuedness that is the only one where ``sym.sign`` exists at all
    (``match_symmetries(signs=self.spinor)``; a scalar table ships ``S = 1``).
    Measured 2026-08-25: **zero** ``InternalError``, worst defect **1.0e-05**
    at SG 149 -- ``irreptables``' own printed precision for ``S``, five decimals
    -- against the ``1e-3`` tolerance, and the competing branch is off by order
    1, so there is nothing in between for the check to be fooled by.

    The frame census is 356 ``"standard"`` and 104 ``"hexagonal"``: the shipped
    ``S`` of the 52 hexagonal-and-trigonal space groups (in both magnetic
    valuednesses) are written in a Cartesian frame turned from the conventional
    cell's standard orientation, and every other lattice system's are written in
    the standard orientation itself -- which the 36 cubic groups pin with no
    residual freedom at all (their point groups have trivial centraliser in
    SO(3), so exactly one frame reproduces their data).

    The *order* of ``_TABLE_SPIN_FRAMES`` is checked not to matter: both frames
    verify at 4 of the 460 instances (SG 1 and SG 2, unitary and grey -- point
    groups ``1`` and ``-1``, whose every operation has non-zero spinor trace and
    is therefore sign-pinned by the trace alone), and they hand back the *same*
    signs there.
    """
    unitary, grey = _sweep_instances()
    spinful = [(ns, mag) for ns, sp, mag in unitary + list(grey) if sp]
    assert len(spinful) == 460, len(spinful)

    frames, worst, worst_at, ambiguous = {}, 0.0, None, []
    for number_str, magnetic in spinful:
        sg = synthetic_spinor_groups[(number_str, magnetic)]
        record = _spacegroup._canonical_spinor_signs(sg)          # raises if it cannot decide
        frames[record["frame"]] = frames.get(record["frame"], 0) + 1
        if record["defect"] > worst:
            worst, worst_at = record["defect"], (number_str, magnetic)
        others = _canonical_with_frames(
            sg, [f for f in _spacegroup._TABLE_SPIN_FRAMES if f[0] != record["frame"]])
        if others is not None:                          # a second frame also fits ...
            ambiguous.append((number_str, magnetic))
            assert others["signs"] == record["signs"], (number_str, magnetic)
    assert frames == {"standard": 356, "hexagonal": 104}, frames
    assert worst < 1e-4, (worst, worst_at)
    assert ambiguous == [("1", False), ("2", False), ("1.2", True), ("2.5", True)], \
        ambiguous


@pytest.mark.data
def test_the_canonical_lifts_realize_the_tabulated_factor_system_everywhere(
        shipped_tables, synthetic_spinor_groups):
    r"""The lift-set census, and the post-condition that closes it (2026-08-25).

    The frame sweep above says every stored lift is ``±`` the tables' own.  This
    one says what that ``±`` *does*: a lift set is fixed by its ℤ₂ factor system
    ``S(g)S(h) = ω(g,h)S(gh)``, twisting it by a per-operation sign vector ``b``
    multiplies ``ω`` by the coboundary ``δb``, and ``δb ≡ 1`` only when ``b`` is a
    homomorphism.  ``irrep`` orients each rotation's axis from an eigenvector of
    the Cartesian rotation matrix one operation at a time (``np.linalg.eig``
    fixes it only up to a sign, and flipping the axis flips the lift at *every*
    angle), so ``b`` is generically **not** a homomorphism and the stored set
    realizes a different factor system from the tabulated one.

    Three statements, over all 460 shipped double-valued instances:

    1. **there is an** ``ω_table`` **to solve against** -- the shipped ``S``
       close as a lift set at every instance; worst ``|½ tr(S_g S_h S_gh†)|``
       deviation from ``±1`` measured **1.4e-05** (SG 207), which is
       ``irreptables``' own five-decimal printed precision;
    2. **the correction solves the coboundary equation** -- the lifts
       :func:`groups._pin_spinor_branch` leaves in the group *are* the shipped
       ``S`` carried into this cell's spin frame, ``S_ours(g) = R S_table(g)
       R†`` with no sign left over (worst entrywise deviation measured
       **1.0e-05** at SG 157, the tables' printed precision), so the corrected
       set's own ℤ₂ factor system is the tabulated one **pair by pair**: the
       ``±1`` relating the two sets, measured here from the corrected lifts
       rather than re-derived by the library's rule, is ``+1`` on every
       operation, and its coboundary is ``+1`` at all 261905 composable pairs.
       The correction
       is applied by the library and never by this test -- flipping the lifts
       here and asking :func:`groups._canonical_spinor_signs` to re-derive the
       sign cannot fail, since that rule reads the sign off the same trace it
       was applied to;
    3. **it was needed almost everywhere** -- measured 2026-08-25 on one generic
       cell per group: **356** of the 460 instances (167 unitary, 189 grey) had a
       lift set that did *not* realize ``ω_table``, 94734 of 261905 composable
       pairs twisted; 62 instances needed no correction at all and 42 needed one
       that happens to be a homomorphism (a harmless character twist -- it
       permutes the tabulated rows and moves no page).

    The *count* in (3) is asserted loosely on purpose.  Which instances are
    affected is a property of the **cell as presented**, not of the group:
    LAPACK's eigenvector sign is not a function of the space group, and a second
    generic cell gives 358 affected instances with a substantially different
    list (measured).  That is itself the argument for correcting the lifts at
    this package's boundary rather than cataloguing the groups where it matters.
    """
    unitary, grey = _sweep_instances()
    spinful = [(ns, mag) for ns, sp, mag in unitary + list(grey) if sp]
    assert len(spinful) == 460, len(spinful)

    worst_cocycle, worst_at, affected, twisted, pairs, untouched = 0.0, None, [], 0, 0, 0
    worst_lift, checked_pairs = 0.0, 0
    for number_str, magnetic in spinful:
        table = shipped_tables[(number_str, True, magnetic)]
        tsyms = table.u_symmetries + table.au_symmetries
        theirs = [np.asarray(t.S, dtype=complex) for t in tsyms]
        #  a copy: the correction below rewrites the group's lifts in place
        sg = copy.deepcopy(synthetic_spinor_groups[(number_str, magnetic)])
        rec = _spacegroup._pin_spinor_branch(sg)          # the library applies the correction
        if rec["cocycle"] > worst_cocycle:                                  # 1
            worst_cocycle, worst_at = rec["cocycle"], (number_str, magnetic)
        if rec["mixed"]:
            affected.append((number_str, magnetic))
        twisted += rec["mixed"]
        pairs += rec["pairs"]
        untouched += int(all(s > 0 for s in rec["signs"]))

        #  2: what the corrected lifts ARE, read off them -- the b(g) carrying
        #  the tables' own matrices onto them, and the factor system that b
        #  twists.  `_lift_set_cocycle(rots, theirs, b)` counts the composable
        #  pairs where δb = −1, i.e. where ω_ours differs from ω_table.
        spin = np.asarray(sg.spin_transf, dtype=complex)
        after = []
        for op, S_table in zip(sg.symmetries, theirs):
            carried = spin @ S_table @ spin.conj().T
            lift = np.asarray(op.spinor_rotation, dtype=complex)
            lam = complex(np.trace(lift @ carried.conj().T) / 2)
            after.append(1 if lam.real > 0 else -1)
            worst_lift = max(worst_lift, float(np.abs(lift - carried).max()))
        assert set(after) == {1}, (number_str, magnetic, after)
        _dev, mixed_after, pairs_after, at = _spacegroup._lift_set_cocycle(
            [t.R for t in tsyms], theirs, after)
        assert mixed_after == 0, (number_str, magnetic, mixed_after, at)
        checked_pairs += pairs_after

    assert worst_cocycle < 1e-4, (worst_cocycle, worst_at)                  # 1
    assert pairs == 261905, pairs
    assert (checked_pairs, worst_lift < 1e-4) == (261905, True), (           # 2
        checked_pairs, worst_lift)
    assert len(affected) > 300, len(affected)                               # 3
    assert twisted > 8e4, twisted
    assert untouched < 100, untouched


def test_the_two_spinor_branches_differ_by_a_character_of_the_point_group():
    """The diagnosis, locked in: what the random fit was choosing between.

    Conjugating the spin frame by the two-fold rotation about the ``z`` axis of
    the conventional cell's standard orientation -- the principal axis of every
    axial class -- fixes every shipped ``S`` up to a sign, so it maps one
    solution of the matching problem onto the other; and the sign it produces is
    ``-1`` exactly on the operations whose **proper part** is a two-fold rotation
    about an axis perpendicular to it.  For the crystal classes whose
    improper operations are all mirrors -- ``m``, ``mm2``, ``3m``, ``4mm``,
    ``6mm``, and SG 183's ``6mm`` is where the defect was found -- that is
    precisely the **determinant** (pseudo-scalar) character: ``+1`` on the
    proper operations, ``-1`` on the improper ones.  It is not the determinant
    character in general (``6/mmm`` and ``-6m2`` are checked here as the
    counter-examples: they carry proper in-plane two-folds as well), and for a
    class with an improper operation of non-zero spinor trace -- inversion,
    ``S3``, ``S4``, ``S6`` -- there is no ambiguity at all, because conjugation
    preserves traces.  ``m-3m`` (SG 227's class) is the checked case of that.
    """
    perp = np.diag([-1.0, -1.0, 1.0])                   # C2 about the principal axis
    equals_det = {"6": True, "25": True, "99": True, "156": True, "183": True,
                  "191": False, "187": False}
    for number_str, det_too in equals_det.items():
        sg = _synthetic_spinor_group(number_str, False)
        base = _spacegroup._canonical_spinor_signs(sg)
        twisted = _canonical_with_frames(
            sg, ((base["frame"], base["correction"] @ perp),))
        assert twisted is not None, number_str          # the twist is a solution too
        ratio = np.array(base["signs"]) * np.array(twisted["signs"])
        assert not np.array_equal(ratio, np.ones_like(ratio)), number_str
        dets = np.array([int(round(np.linalg.det(s.rotation))) for s in sg.symmetries])
        assert np.array_equal(ratio == -1, _perp_two_fold(sg)), number_str
        assert np.array_equal(ratio, dets) == det_too, number_str

    cubic = _synthetic_spinor_group("227", False)       # m-3m: no ambiguity to have
    base = _spacegroup._canonical_spinor_signs(cubic)
    assert _canonical_with_frames(cubic, ((base["frame"], base["correction"] @ perp),)) \
        is None, "a cubic group's spin frame is unique; the twist must not solve it"


def test_an_undecidable_spinor_frame_raises_an_internal_error_not_a_refusal():
    """The error path: a frame that cannot be applied is a **bug**, not a verdict.

    Two ways to make the canonicalisation undecidable, both of which are
    statements about this package's use of a convention the tables fix outright:
    a frame list that does not contain the group's own frame, and an operation
    whose SU(2) lift is not a lift of the rotation the table pairs it with.
    Both must raise ``refusal.InternalError`` -- never return a ``Refusal``,
    because there is no gauge left for a caller to supply and nothing about the
    *model* has been measured.
    """
    sg = _synthetic_spinor_group("183", False)          # P6mm: the defect's home
    assert _spacegroup._canonical_spinor_signs(sg)["frame"] == "hexagonal"

    saved = _spacegroup._TABLE_SPIN_FRAMES
    try:
        _spacegroup._TABLE_SPIN_FRAMES = (("standard-only", np.eye(3)),)
        with pytest.raises(InternalError) as exc:
            _spacegroup._canonical_spinor_signs(sg)
    finally:
        _spacegroup._TABLE_SPIN_FRAMES = saved
    text = str(exc.value)
    assert "the tables fix the spin frame completely" in text, text
    assert "#183" in text and "'standard-only'" in text, text
    assert "_TABLE_SPIN_FRAMES" in text, text
    assert not isinstance(exc.value, Refusal) and isinstance(exc.value, RuntimeError)

    sg.symmetries[7].spinor_rotation = np.array([[0.6, 0.8], [-0.8, 0.6]], dtype=complex)
    with pytest.raises(InternalError) as exc:
        _spacegroup._canonical_spinor_signs(sg)
    assert "operation #8" in str(exc.value), str(exc.value)


_SPIN_DETERMINISM_PAYLOAD = r"""
import numpy as np
from pythtb import Lattice, TBModel
from pythtb.models import kane_mele
import pythtqc as tqc
from pythtqc import _spacegroup
from pythtqc import _layers


def served(group, point):
    table = tqc.corep_table(group, tqc.high_symmetry_points(group)[point])
    return " ".join(
        name + "=" + ",".join("%d:%+.6f%+.6fj" % (i, table[name][i].real,
                                                  table[name][i].imag)
                              for i in sorted(table[name]))
        for name in sorted(table))


cell, labels = _layers._decorated_cell(_layers._slab_presentation(kane_mele(0.0, -1.0, 0.3, 0.25)))
km = _spacegroup._group_of(cell, typat=labels, spinor=True)
print(km.number_str, km._spin_frame["frame"], km._spin_frame["signs"])
print("K", served(km, "K"))

lat = Lattice([[1.0, 0.0, 0.0], [-0.5, np.sqrt(3) / 2, 0.0], [0.0, 0.0, 1.6]],
              [[0.0, 0.0, 0.0], [1 / 3, 2 / 3, 0.3]], periodic_dirs=[0, 1, 2])
p3m1 = _spacegroup._group_of(TBModel(lattice=lat, spinful=True), typat=[0, 1], spinor=True)
print(p3m1.number_str, p3m1._spin_frame["frame"], p3m1._spin_frame["signs"])
for point in ("K", "H"):
    print(point, served(p3m1, point))
"""


@pytest.mark.slow
def test_the_spinor_branch_is_the_same_in_every_fresh_process(capsys):
    """The guarantee, measured where it failed: two processes, one answer.

    A fresh subprocess and the test's own process each build the Kane-Mele
    decorated cell (``#183.186``, the group whose ``K`` exchanged ``-K4`` and
    ``-K5`` from run to run) and a genuinely three-dimensional ``P3m11'``
    (``#156.50``, whose ``K`` and ``H`` carry the same conjugate 1-dimensional
    pairs), and print the *whole* served table -- names **and** rows -- plus
    the canonical sign vector.  Two processes are the point: ``np.random``'s
    state is what used to decide this, so re-running in one interpreter would
    prove nothing.

    Before the convention was applied this exact payload printed ``-K4`` and
    ``-K5`` the other way round on roughly a third of runs (measured 7/10 vs
    3/10 over fresh fits of ``#183.186``); the rows were bit-stable throughout,
    which is why names and rows are compared together.  The rows themselves are
    the owner's datum, which the served-page test below pins in process.

    What is **not** pinned is the sign vector's own entries.  A sign
    is the relation between the tables' lift and ``irrep``'s, and ``irrep`` builds
    its lift from an eigenvector of the Cartesian rotation matrix
    (``_get_operation_type``), whose sign ``np.linalg.eig`` fixes arbitrarily --
    and at the two-folds and mirrors, where the lift is ``exp(-i pi n.sigma/2)``,
    that sign goes straight into the lift.  Wrapping that call to return the
    other eigenvector moved ``#183.186``'s vector from six bars to eight (and
    ``mixed`` from 216 to 192) while **every** served row at ``K`` stayed
    byte-identical to all 17 digits: the bar the tables' frame removes is exactly
    the bar ``irrep`` put in.  So the vectors are compared across the two
    processes -- whatever this LAPACK chooses, the pinned branch reproduces it --
    and checked for shape, while the *served page* carries the values (measured
    here: ``#183.186`` bars 6 of its 24, ``#156.50`` 2 of its 12).
    """
    import os
    import re
    import subprocess

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run([sys.executable, "-c", _SPIN_DETERMINISM_PAYLOAD],
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace", env=env, timeout=600)
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    capsys.readouterr()
    exec(compile(_SPIN_DETERMINISM_PAYLOAD, "<payload>", "exec"), {"__name__": "payload"})
    here = capsys.readouterr().out
    assert proc.stdout == here, (
        "the served table is not the same in every fresh process:\n"
        + "\n--- vs ---\n".join((proc.stdout, here)))

    for number, n_ops in (("183.186", 24), ("156.50", 12)):
        line = re.search(rf"^{re.escape(number)} hexagonal \(([^)]*)\)$", here, re.M)
        assert line is not None, here              # the group, and the frame it is in
        signs = [int(t) for t in line.group(1).split(",") if t.strip()]
        assert len(signs) == n_ops and set(signs) <= {1, -1}, signs


def test_the_served_page_at_K_of_183_is_bcs_element_by_element():
    r"""The owner's Bilbao cross-check, as a standing regression (2026-08-25).

    **The datum** (owner, verified against BCS's *Double space group
    representations* for ``P6mm1'`` on 2026-08-25): at ``K`` the double-valued
    ``−K4`` carries ``−i`` on **all three** mirrors and ``−K5`` carries ``+i`` on
    all three; ``−K6`` is two-dimensional with ``0`` on all three.

    Before the lift-set correction this package served ``−K4`` as
    ``(+i, −i, −i)`` over operations ``(10, 11, 12)`` -- a *mixed* pattern, and
    therefore a page BCS never printed.  Nothing was wrong with the naming (the
    character-anchored naming sweep compares BCS characters evaluated on our
    representatives, so the bars sit on both sides and cancel) and nothing was
    wrong with the frame constant either: the owner's datum is what *confirms*
    ``_TABLE_SPIN_FRAMES``' hexagonal offset, since the other frame would have
    exchanged the two names.  What was wrong was that ``irrep``'s per-operation
    lift for the third mirror stored the ``Ē·m`` partner of the ``C3``-consistent
    set -- ``S(C3⁺)S(m₁₁)S(C3⁺)⁻¹ = +S(m₁₂)`` but
    ``S(C3⁺)S(m₁₂)S(C3⁺)⁻¹ = −S(m₁₀)`` -- and the old convention put that bar on
    the *character*.

    So this asserts the numbers twice over: against the owner's datum spelled
    out, and against ``irreptables``' shipped row entry by entry.
    """
    # `tables_status()` first, always: `irreptables.irreps` imports `irrep.utility`,
    # so reaching irreptables before irrep leaves that module half-built when irrep
    # runs its own `from irreptables.irreps import IrrepTable` and irrep falls back
    # to DummyMethodTables for the rest of the process (see test_fixtures_smoke.py).
    # Measured 2026-09-09: without this line the test refuses with `irrep_tables`
    # whenever it is the first in its process to reach irreptables (`pytest -k`).
    assert G.tables_status()["tables"]
    from irreptables.irreps import IrrepTable

    from pythtb.models import kane_mele

    cell, labels = _layers._decorated_cell(
        _layers._slab_presentation(kane_mele(0.0, -1.0, 0.3, 0.25)))
    g = _spacegroup._group_of(cell, typat=labels, spinor=True)
    assert g.number_str == "183.186", g.number_str
    K = G.high_symmetry_points(g)["K"]
    lg = G.little_group(g, K)
    assert lg.indices == (1, 2, 3, 10, 11, 12), lg.indices
    served = G.corep_table(g, K)
    assert sorted(served) == ["-K4", "-K5", "-K6"], sorted(served)

    mirrors = (10, 11, 12)
    rot = (2, 3)
    for label, mirror, three in (("-K4", -1j, -1.0), ("-K5", +1j, -1.0),
                                 ("-K6", 0.0, +1.0)):
        row = served[label]
        assert row[1] == pytest.approx(2.0 if label == "-K6" else 1.0, abs=1e-6)
        for i in rot:
            assert row[i] == pytest.approx(three, abs=1e-6), (label, i, row[i])
        for i in mirrors:                       # the owner's headline: UNIFORM
            assert row[i] == pytest.approx(mirror, abs=1e-6), (label, i, row[i])

    # ... and the same numbers read straight out of the shipped file.  The
    # count is asserted so the comparison cannot go vacuous on a filtered loop:
    # 3 shipped coreps at K by the 6 operations of the little group.
    table = IrrepTable("183.186", True, magnetic=True)
    compared = 0
    for irr in table.irreps:
        if irr.kpname != "K":
            continue
        for i, chi in irr.characters.items():
            compared += 1
            assert served[irr.name][int(i)] == pytest.approx(complex(chi), abs=5e-5), (
                irr.name, i, served[irr.name][int(i)], chi)
    assert compared == 18, compared

    # WHY the row is uniform: with the corrected lifts the three mirrors are one
    # class of the double group -- conjugation by C3 walks round them with no Ē.
    # Deterministic and BLAS-independent: these are the shipped S carried by a
    # closed-form frame map, with no eigen-decomposition anywhere.
    S = {i: np.asarray(g.symmetries[i - 1].spinor_rotation, dtype=complex)
         for i in lg.indices}
    for src, dst in ((10, 11), (11, 12), (12, 10)):
        assert np.abs(S[2] @ S[src] @ S[2].conj().T - S[dst]).max() < 1e-9, (src, dst)

    # and what it took to get there.  *How many* of irrep's per-operation lifts
    # came out backwards, and which, is a property of the cell as presented
    # (LAPACK's eigenvector sign in `_get_operation_type`), so none of it is
    # asserted: on this machine the barred one inside this little group is
    # operation 10, with 216 of the 576 composable pairs then carrying a twisted
    # factor system.  The frame those signs were read in is not arbitrary, and it
    # is the thing the owner's datum confirms.
    assert g._spin_frame["frame"] == "hexagonal"      # confirmed, not merely assumed


def _star_transport(group, k3):
    r"""``(rel, sigma, bars, phases)`` for carrying the little group at ``k`` to ``-k``.

    The relating element ``c`` is the group's own: the first unitary operation
    whose k-action sends ``k`` to ``-k`` modulo a reciprocal-lattice vector.  For
    ``g`` in the little group at ``-k``, ``h = c⁻¹ g c`` is in the little group at
    ``k``, and the two are related **as space-group operations** by a lattice
    translation ``m`` with ``g c = c h {1|m}``; at the SU(2) level they are
    related by a *bar*,

    ``S(c)⁻¹ S(g) S(c) = ω(g) S(h)``,  ``ω = ±1``,

    because the lift of a two-fold anticommutes with the lift of a perpendicular
    two-fold.  Both are read off the group's own stored data, never typed.
    """
    syms = group.symmetries
    lg = G.little_group(group, k3)
    k = np.asarray(k3, dtype=float)
    rel = None
    for i, s in enumerate(syms):
        if s.time_reversal:
            continue
        act = group.k_action_of(s)
        if act is None:
            continue
        image = np.asarray(act(np.atleast_2d(k))).reshape(-1)
        if np.abs(((image + k) + 0.5) % 1.0 - 0.5).max() < 1e-9:
            rel = i
            break
    assert rel is not None, f"no operation of {group.number_str} relates k and -k"
    c = syms[rel]
    Sc = np.asarray(c.spinor_rotation, dtype=complex)
    Rc = np.asarray(c.rotation, dtype=int)
    Rci = np.round(np.linalg.inv(Rc)).astype(int)
    by_rot = {}
    for i, s in enumerate(syms):
        if not s.time_reversal:
            by_rot.setdefault(np.asarray(s.rotation, dtype=int).tobytes(), i)

    sigma, bars, phases = {}, {}, {}
    for ind in lg.indices:
        s = syms[ind - 1]
        j = by_rot[(Rci @ np.asarray(s.rotation, dtype=int) @ Rc).tobytes()]
        h = syms[j]
        lam = complex(np.trace(Sc.conj().T @ np.asarray(s.spinor_rotation, complex)
                               @ Sc
                               @ np.asarray(h.spinor_rotation, complex).conj().T) / 2)
        assert abs(abs(lam) - 1) < 1e-9 and abs(lam.imag) < 1e-9, (ind, lam)
        sigma[ind], bars[ind] = j + 1, int(np.sign(lam.real))
        m = (np.asarray(s.translation, float)
             + np.asarray(s.rotation, float) @ np.asarray(c.translation, float)
             - np.asarray(c.translation, float)
             - np.asarray(c.rotation, float) @ np.asarray(h.translation, float))
        phases[ind] = complex(np.exp(-2j * np.pi * np.asarray(lg.k3, float) @ m))
    return rel + 1, sigma, bars, phases


@pytest.mark.slow
@pytest.mark.parametrize("build,point,kind", [
    (lambda: (Lattice([[1.0, 0.0, 0.0], [-0.5, np.sqrt(3) / 2, 0.0], [0.0, 0.0, 1.6]],
                      [[0.0, 0.0, 0.0], [0.0, 0.0, 0.3]], periodic_dirs=[0, 1, 2]),
              [0, 1]), "K", "3D P6mm1' #183.186, symmorphic"),
    (lambda: (Lattice([[1.0, 0.0, 0.0], [-0.5, np.sqrt(3) / 2, 0.0], [0.0, 0.0, 1.63]],
                      [[1 / 3, 2 / 3, 0.0], [2 / 3, 1 / 3, 0.5]], periodic_dirs=[0, 1, 2]),
              None), "H", "3D P6_3/mmc1' #194.264, nonsymmorphic, k_z = 1/2"),
])
def test_transporting_a_little_group_table_to_the_other_star_member_needs_the_bars(
        build, point, kind):
    r"""The star-member trap, as a standing guard (owner, 2026-08-25).

    ``k`` and ``−k`` are related by a two-fold ``c`` about the principal axis,
    and *as rotations* every element of the little group is fixed by conjugation
    with it.  Their **lifts** are not: a two-fold lift anticommutes with a
    perpendicular two-fold lift, so ``S(c)⁻¹ S(g) S(c) = −S(g)`` on every
    operation whose proper part is an in-plane two-fold -- a vertical mirror, or
    a proper in-plane ``C2``.  Transporting a little-group page between star
    members therefore **mixes bars**, and it does so at the lift level, with
    nothing to do with the translation-phase mechanism of ``_transported_table``
    (which appears here as the separate ``e^{-2πi k·m}`` factor and is ``−1`` on
    six operations at ``H`` of the nonsymmorphic case).

    Measured, in the canonical convention:

    * transporting a served row with **both** factors lands on the entrywise
      **conjugate** of the *same* label -- which is the physical statement that
      ``−k`` carries the conjugate representations;
    * transporting it **without the bars** lands on a *different* label (``−K4``
      ↔ ``−K5``; ``−H4H5`` ↔ ``−H6H7`` and ``−H8`` ↔ ``−H9``), off by 2 and 4
      respectively.  That is the trap: read the star partner's characters against
      the ``k`` page with no bar bookkeeping and the corep name comes out wrong,
      with nothing anywhere reporting an inconsistency.

    ``corep_table`` itself refuses at ``−k`` (``gate="kpoint_not_tabulated"`` --
    "a point in the star of a tabulated one but not the tabulated arm carries a
    conjugate little group and is not transported here"), which is why this
    transport is the caller's job and why it is worth a regression.
    """
    lat, typat = build()
    kw = dict(spinor=True) if typat is None else dict(spinor=True, typat=typat)
    g = _spacegroup._group_of(TBModel(lattice=lat, spinful=True), **kw)
    k3 = G.high_symmetry_points(g)[point]
    served = G.corep_table(g, k3)
    assert served and not isinstance(served, Refusal), served

    rel, sigma, bars, phases = _star_transport(g, k3)
    assert set(bars.values()) == {1, -1}, bars       # the trap is live here ...
    assert all(sigma[i] == i for i in sigma), sigma  # ... and the rotations hide it

    swapped = 0
    for label, row in served.items():
        moved = {i: bars[i] * phases[i] * row[sigma[i]] for i in row}
        assert max(abs(moved[i] - np.conj(row[i])) for i in row) < 5e-5, (
            f"{label}: the transported row is not the conjugate page")
        naive = max(abs(moved[i] - row[i]) for i in row)
        if naive > 1.0:                                   # a bar-free read mislabels it
            others = [l2 for l2, r2 in served.items()
                      if l2 != label and max(abs(moved[i] - r2[i]) for i in moved) < 5e-5]
            assert others, f"{label}: bar-free transport matches nothing at all"
            swapped += 1
    assert swapped >= 2, (swapped, kind)

    # and -k is genuinely not servable: the trap has no library-side shortcut
    refused = G.corep_table(g, -np.asarray(k3, dtype=float))
    assert isinstance(refused, Refusal) and refused.gate == "kpoint_not_tabulated"


def test_a_shipped_lift_set_that_does_not_close_raises_an_internal_error(monkeypatch):
    r"""The second error path of the coboundary solve (2026-08-25).

    Solving ``δc = ω_ours/ω_table`` needs a right-hand side, and that is the
    statement that the **shipped** SU(2) matrices close as a lift set:
    ``S_t(g)S_t(h) = ±S_t(gh)``.  They do, at all 460 shipped double-valued
    instances -- but that is a property of the installed ``irreptables`` and not
    a theorem, so it is measured at every call rather than assumed.

    The failure is staged the only way it can be without also breaking the frame
    match: one shipped ``S`` and the corresponding stored lift are *both*
    multiplied by ``i``.  The per-operation ratio ``½ tr(S_ours · carried†)`` is
    quadratic in that phase and stays a real ``±1``, so the frame is still
    accepted -- and the factor system is not, because ``i`` survives in every
    product where the operation appears once.  ``InternalError``, never a
    ``Refusal``: nothing about a *model* has been measured.
    """
    import irreptables.irreps as IT

    real_table = IT.IrrepTable
    spoiled = 3                                  # the two-fold; index into table order

    def poisoned(*args, **kwargs):
        table = real_table(*args, **kwargs)
        table.symmetries[spoiled].S = 1j * np.asarray(table.symmetries[spoiled].S,
                                                      dtype=complex)
        return table

    sg = _synthetic_spinor_group("183", False)
    assert _spacegroup._canonical_spinor_signs(sg)["cocycle"] < 1e-4      # ... before poisoning
    monkeypatch.setattr(IT, "IrrepTable", poisoned)
    sg.symmetries[spoiled].spinor_rotation = 1j * np.asarray(
        sg.symmetries[spoiled].spinor_rotation, dtype=complex)

    with pytest.raises(InternalError) as exc:
        _spacegroup._canonical_spinor_signs(sg)
    text = str(exc.value)
    assert "do not close as a lift set" in text, text
    assert "S(g)S(h) is not ±S(gh)" in text, text
    assert "coboundary equation" in text, text
    assert not isinstance(exc.value, Refusal) and isinstance(exc.value, RuntimeError)


@pytest.mark.slow
def test_the_front_door_forwards_its_cut_and_window_to_the_detector():
    """``realized_group(cut=, window=)`` are the numbers the detector ran at.

    The two keywords are resolved at the front door and handed to
    ``_validate_group`` at every call the pipeline makes, so the record of each
    operation carries exactly the declared cut and window -- and, with nothing
    declared, the cut is the gate and the window follows from it.
    """
    m = square()
    r = G.realized_group(m, cut=1e-6, window=0.05, cross_check=False)
    assert r and r.validated._records, r
    for ind, rec in r.validated._records.items():
        assert rec["cut"] == 1e-6 and rec["window"] == 0.05, (ind, rec)
        assert rec["tol"] == 1e-8
    plain = G.realized_group(m, cross_check=False)
    for ind, rec in plain.validated._records.items():
        assert rec["cut"] == rec["tol"] == 1e-8, (ind, rec)
