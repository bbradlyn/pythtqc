r"""The Hamiltonian sets the primitive cell — the lattice descent (wave R4a).

``_group_of`` answers what a *cell* allows.  On a cell whose orbital positions
repeat under a fractional translation that answer is the whole **lattice
supergroup**: every operation spglib finds on the model's own lattice, the pure
translation ``{1|τ}`` and its coset partners included
(``pythtqc.groups.GroupData._lattice_index`` counts the cosets).  Nothing is
refused there and no table is served with it — the tables belong to the finer
lattice and would label a folded Brillouin zone, so every table entry point
refuses ``lattice_not_descended``.

Whether the model *is* a supercell is then a property of the **Hamiltonian**,
and only :func:`pythtqc.groups.realized_group` can say.  It reads ``τ`` on its
solution space at the sites — the site permutation times a global phase, with
any change of basis within each site, the permutation itself measured
directly — and either refuses ``supercell`` (the Hamiltonian obeys it: every
k-point label, EBR row and SRSI on that folded zone would be the wrong zone's)
or descends to the model's own lattice as the first step of the ordinary
unitary descent.

Every number below is measured.
"""

import itertools

import numpy as np
import pytest
from pythtb import Lattice, TBModel

import tqc_models
from pythtqc import groups as tqg
from pythtqc import sewing as tqsew
from pythtqc.refusal import Refusal
from pythtqc import _layers
from pythtqc import _spacegroup
from pythtqc import _tables

_ULP = np.finfo(float).eps          # one unit of double-precision roundoff, 2.2e-16
_FEW_ULP = 16 * _ULP                # a few ulp: thirds arithmetic accumulated in a
                                    # closure sum, a group average over 32 operations


# ---------------------------------------------------------------------------
# the fixtures: a simple-cubic 2x1x1 supercell, uniform and dimerized
# ---------------------------------------------------------------------------

def cubic_2x1x1(t1=1.0, t2=1.0):
    """A ``2x1x1`` supercell of the simple-cubic s model, with two x hoppings.

    Sites at ``(0,0,0)`` and ``(½,0,0)`` of the lattice ``diag(2,1,1)``: the
    point set is the primitive cubic one, so the cell repeats under
    ``τ = (½,0,0)``.  ``t2 == t1`` is that cubic crystal written twice; ``t2 !=
    t1`` is a dimerized chain — a legitimate *primitive* model of ``P4/mmm``
    whose origin sits at a bond centre, and whose bonds no symmetry finder can
    see, since no atom moved.
    """
    m = TBModel(Lattice(np.diag([2.0, 1.0, 1.0]), [[0, 0, 0], [0.5, 0, 0]],
                        periodic_dirs=[0, 1, 2]), spinful=False)
    m.set_hop(-t1, 0, 1, [0, 0, 0])          # intra-cell along x
    m.set_hop(-t2, 1, 0, [1, 0, 0])          # inter-cell along x
    for o in (0, 1):
        m.set_hop(-t1, o, o, [0, 1, 0])
        m.set_hop(-t1, o, o, [0, 0, 1])
    return m


def square_two_orbitals():
    """``test_groups.py``'s square cell: orbitals at ``1a`` and ``1c``, ``±0.3`` on site.

    Under the lattice-honest default typing the two sites are one species, so
    the point set is centred and the cell carries ``τ = (½,½,0)``.  The on-site
    energies are what break it.
    """
    lat = Lattice([[1.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [0.5, 0.5]],
                  periodic_dirs=[0, 1])
    m = TBModel(lattice=lat)
    m.set_onsite([0.3, -0.3])
    m.set_hop(-1.0, 0, 1, [0, 0])
    return m


def supercell_square():
    """A ``2x1`` square supercell, deliberately not primitive — every hopping equal."""
    lat = Lattice([[2.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [0.5, 0.0]],
                  periodic_dirs=[0, 1])
    m = TBModel(lattice=lat)
    m.set_onsite([0.0, 0.0])
    m.set_hop(-1.0, 0, 1, [0, 0])
    m.set_hop(-1.0, 1, 0, [1, 0])
    m.set_hop(-1.0, 0, 0, [0, 1])
    m.set_hop(-1.0, 1, 1, [0, 1])
    return m


def twisted_p_chain(delta, t_sigma=1.0, t_pi=0.3, t_perp=0.2, spinful=False):
    """A ``2x1x1`` chain of ``(p_x, p_y)`` sites, the second site's orbitals in a frame rotated by ``delta``.

    Sites ``A = 0`` and ``B = ½`` along ``x`` of the lattice ``diag(2,1,1)``,
    two orbitals each.  Along ``x`` the hopping is ``T = diag(t_sigma, t_pi)``
    between orbitals written in one frame; with ``B``'s frame rotated by
    ``delta`` about ``z`` the two bonds carry ``T R(delta)ᵀ`` (``A → B``) and
    ``R(delta) T`` (``B → A`` in the next cell).  The transverse hoppings are
    orbital-independent.  ``delta = 0`` is the one-site chain written twice; any
    ``delta`` is a **site-local change of orbital basis** of it -- the spectrum
    is the untwisted one's, and the half translation holds once each site's
    orbitals are carried onto its translate's by that rotation, so the model is
    a supercell of the one-site chain.  As hoppings on the lattice it is a
    two-site cell whose two bonds carry different ``2x2`` matrices.
    ``spinful`` puts a spin on every orbital, every hopping the identity on it.
    """
    lat = Lattice(np.diag([2.0, 1.0, 1.0]),
                  [[0, 0, 0], [0, 0, 0], [0.5, 0, 0], [0.5, 0, 0]], periodic_dirs=[0, 1, 2])
    m = TBModel(lat, spinful=spinful)
    c, s = np.cos(delta), np.sin(delta)
    R = np.array([[c, -s], [s, c]])
    T = np.diag([t_sigma, t_pi])
    T_ab, T_ba = T @ R.T, R @ T
    for a in range(2):
        for b in range(2):
            m.set_hop(T_ab[a, b], a, 2 + b, [0, 0, 0])
            m.set_hop(T_ba[a, b], 2 + a, b, [1, 0, 0])
    for o in range(4):
        m.set_hop(t_perp, o, o, [0, 1, 0])
        m.set_hop(t_perp, o, o, [0, 0, 1])
    return m


def screw_chain(tz=0.4):
    """Two ``(p_x, p_y)`` sites stacked along ``z``, the second's in-plane bonds the first's turned by 90°.

    Sites ``A = 0`` and ``B = ½`` along ``z`` of the unit cube, two orbitals
    each, and no hopping joins a ``p_x`` to a ``p_y``.  ``A``'s ``p_x`` hops
    ``1.0`` along ``x`` and ``0.2`` along ``y``, its ``p_y`` ``0.3`` and
    ``0.6``; ``B``'s orbitals hop as ``A``'s turned by 90° about ``z``; ``tz``
    joins each orbital to its like across both ``z`` bonds.  So the half
    translation ``(0,0,½)`` carries ``A``'s bonds onto bonds ``B`` does not
    have, and composed with the fourfold about the stacking axis -- the 4_2
    screw ``{C4z|0,0,½}``, ``p_x → p_y`` -- it carries every bond onto itself.
    """
    lat = Lattice(np.diag([1.0, 1.0, 1.0]), [[0, 0, 0]] * 2 + [[0, 0, 0.5]] * 2,
                  periodic_dirs=[0, 1, 2])
    m = TBModel(lat, spinful=False)
    Ax, Ay = np.diag([1.0, 0.3]), np.diag([0.2, 0.6])
    Bx, By = np.diag([0.6, 0.2]), np.diag([0.3, 1.0])
    for off, Tx, Ty in ((0, Ax, Ay), (2, Bx, By)):
        for a in range(2):
            m.set_hop(Tx[a, a], off + a, off + a, [1, 0, 0])
            m.set_hop(Ty[a, a], off + a, off + a, [0, 1, 0])
    for a in range(2):
        m.set_hop(tz, a, 2 + a, [0, 0, 0])
        m.set_hop(tz, 2 + a, a, [0, 0, 1])
    return m


def uncoupled_screw_chain(t1=0.5, t2=0.2):
    """Two ``(p_x, p_y)`` sites stacked along ``z``, nothing in the plane, the bonds ``diag(t1, t2)`` then ``diag(t2, t1)``.

    Sites ``A = 0`` and ``B = ½`` along ``z`` of the unit cube.  No hopping
    joins a ``p_x`` to a ``p_y``, so the model is two chains: ``p_x`` hopping
    ``t1`` from ``A`` to ``B`` and ``t2`` on to the next ``A``, and ``p_y`` the
    other way round.  The half translation carries the one chain onto the
    other, a quarter turn of each site's orbitals: two copies of one dimerized
    chain, shifted by half a cell.
    """
    lat = Lattice(np.diag([1.0, 1.0, 1.0]), [[0, 0, 0]] * 2 + [[0, 0, 0.5]] * 2,
                  periodic_dirs=[0, 1, 2])
    m = TBModel(lat, spinful=False)
    for a, (up, on) in enumerate(((t1, t2), (t2, t1))):
        m.set_hop(up, a, 2 + a, [0, 0, 0])
        m.set_hop(on, 2 + a, a, [0, 0, 1])
    return m


def uncoupled_sublattices(t=1.0, ty=0.4, tz=0.3):
    """Two identical simple-orthorhombic ``s`` lattices, the second shifted by half a cell along ``x``, nothing joining them."""
    m = TBModel(Lattice(np.diag([2.0, 1.0, 1.0]), [[0, 0, 0], [0.5, 0, 0]],
                        periodic_dirs=[0, 1, 2]), spinful=False)
    for o in (0, 1):
        m.set_hop(-t, o, o, [1, 0, 0])
        m.set_hop(-ty, o, o, [0, 1, 0])
        m.set_hop(-tz, o, o, [0, 0, 1])
    return m


@pytest.fixture(scope="module")
def cubic_supergroup():
    """``_group_of`` on the ``2x1x1`` cell — the lattice supergroup itself."""
    return _spacegroup._group_of(cubic_2x1x1(1.0, 0.5))


@pytest.fixture(scope="module")
def dimerized():
    """``realized_group`` on the dimerized ``2x1x1`` chain (the descent)."""
    return tqg.realized_group(cubic_2x1x1(1.0, 0.5))


# ---------------------------------------------------------------------------
# 1. the simple-cubic 2x1x1 supercell
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_group_of_serves_the_lattice_supergroup(cubic_supergroup):
    """The cell allows ``P4/mmm1' #123.340``, 64 operations, in 2 cosets of 16.

    Measured: 32 unitary operations = 16 distinct rotations (the point-set
    operations compatible with the ``(2,1,1)`` lattice; the cubic threefolds are
    not integer on it) times the two cosets of ``{1|(½,0,0)}``, and the same
    again antiunitary.  ``tables_ready`` is False and the record says why in one
    sentence, which is *not* irrep's "could not match this cell" note: this is
    not a degraded identification, it is the supergroup.

    The closure below reads exactly ``0.0``, and on this cell it can: every
    translation here is a half, so the closure sum is dyadic and bit-exact.
    That makes it a poor witness that anything was *measured* — the test that a
    stub returning ``0.0`` cannot pass is the thirds cell of section 5, whose
    closure reads 1.1e-16.
    """
    g = cubic_supergroup
    assert not isinstance(g, Refusal), str(g)
    assert (g.number_str, g.name, g.size) == ("123.340", "P4/mmm1'", 64)
    assert g._lattice_index == 2
    assert g._fractional_translations == ((0.5, 0.0, 0.0),)
    assert g._tables_ready is False
    assert g.provenance == "cell"
    assert "supercell" in g._setup_note and "lattice supergroup of index 2" in g._setup_note
    assert "could not match" not in g._setup_note
    assert "lattice supergroup, index 2" in repr(g)
    # the operations are a group: the identified set closes exactly
    assert g._closure_measured == 0.0 and g._closure_repaired == 0.0
    table = g.get_product_table(get_diff=False)
    assert np.asarray(table).shape == (64, 64)
    assert sorted(set(np.asarray(table).ravel().tolist())) == list(range(64))


@pytest.mark.slow
def test_the_dimerized_chain_descends_to_the_bond_centred_p4mmm(dimerized):
    """``P4/mmm1' #123.340``, 32 of 32, with the origin at a bond centre.

    The dimerization moves no atom, so the point set still repeats under
    ``τ = (½,0,0)`` and the cell allows 64 operations.  The Hamiltonian breaks
    ``τ`` (measured covariance residual 1.0 against ``tol = 1e-8``) and the cell
    is lowered to the 32 that hold.  Both sites read Wyckoff ``g`` in the tables'
    setting — ``2g = (0,0,z), (0,0,-z)`` with the origin between them, which is
    where the mirror and inversion centres of a dimerized chain actually sit.

    The record names the lattice step: ``τ`` refuted by ``"tensor structure"``,
    with 15 coset partners.  16 of the supergroup's 32 unitary operations go —
    the pure translation and the 15 partners ``τ·g`` that go with it — and the
    same 16 antiunitary ones, so the step's ``refuted_by`` histogram totals 32.
    (Only the histogram is pinned: with a single descent step ``n_dropped`` is
    ``sum(refuted_by.values())`` by construction, so it carries nothing the
    breakdown does not.)  ``coset_partners_refuted`` counts the refuted unitary
    operations that are not themselves pure translations.
    """
    r = dimerized
    assert not isinstance(r, Refusal), str(r)
    assert (r.group.number_str, r.group.name) == ("123.340", "P4/mmm1'")
    assert (len(r.validated.indices), r.group.size) == (32, 32)
    assert r.group.provenance == "descended"
    assert r.group._lattice_index == 1 and r.group._tables_ready is True
    assert r.assignment.wyckoff_tables == ("g", "g")
    assert r.assignment.wyckoff_spglib == ("g",)

    step = dimerized.descent["lattice"]
    assert step["index"] == 2
    assert step["fractional_translations"] == ((0.5, 0.0, 0.0),)
    assert step["realized"] == ()
    assert step["coset_partners_refuted"] == 15
    (tau,) = step["operations"]
    assert tau["translation"] == (0.5, 0.0, 0.0)
    assert tau["cartesian"] == (1.0, 0.0, 0.0)
    assert tau["refuted_by"] == "tensor structure"
    assert tau["scaled"] == pytest.approx(1.0, abs=1e-9)
    assert tau["residual"] == pytest.approx(1.0, abs=1e-9)
    assert dimerized.descent["steps"][0]["refuted_by"] == {"tensor structure": 32}
    assert "lattice supergroup of index 2" in step["note"]
    # the composed note is the lattice step's sentence plus the unitary descent's,
    # and what the concatenation adds is the group that ended up being served
    assert "re-identified as P4/mmm1' #123.340." in dimerized.descent["note"]


@pytest.mark.slow
def test_the_tables_served_are_the_descended_groups(dimerized):
    """``corep_content`` at Γ and X reads the descended group's own page.

    The point of the wave: the supergroup's tables would have labelled a folded
    zone, and these are the tables of the group on the model's own lattice.  The
    two bands are the bonding/antibonding pair, and they carry opposite parity
    at every one of these points.
    """
    hsp = tqg.high_symmetry_points(dimerized.group)
    assert not isinstance(hsp, Refusal), str(hsp)
    for point, counts in (("GM", {"GM1+": 1, "GM3-": 1}), ("X", {"X1+": 1, "X2-": 1})):
        k = np.asarray(hsp[point])
        out = tqsew.corep_content(dimerized.rep, k,
                                  states=tqsew.solve_at(dimerized.model, k), bands=2)
        assert not isinstance(out, Refusal), str(out)
        assert dict(out.counts) == counts, point


@pytest.mark.slow
def test_symmetrizing_the_dimerized_chain_changes_nothing_it_can_reach(dimerized):
    """``symmetrize=True`` still descends, and the projection that ran moved 1.3e-15.

    The projection onto the group-invariant part provably never sees the
    supergroup — the lattice descent is the *first* iteration of ``_settle``, and
    symmetrization runs after it — so what it projects onto is the descended
    group, of which this Hamiltonian is already exactly invariant.  The residual
    is the projection's own arithmetic (a sum-and-divide over 32 operations), not
    a change to the model: measured ``max_k |H_sym(k) − H(k)| = 1.3e-15`` over 6
    random k, against ``max|H| = 3.9``.

    A projection that never ran would give ``0.0`` and satisfy that bound just as
    well, so the half of the claim that says it *did* run is asserted on the
    record the mechanism writes: ``symmetrization`` is not ``None``, the model
    handed back is a new object, and all 32 classes read ``"exact"`` — which is
    what makes ``symmetrize=True`` the only thing that can have forced the pass,
    since nothing here is merely *near*.  The projection's own defect,
    1.78e-15 = 2 ulp of ``max|H|``, is pinned beside the residual.

    A declared ``position_tol`` moves nothing either, and not on account of the
    translation: the position pass runs before the detector, ahead of the
    lattice descent, against the supergroup's operations -- and spglib *found*
    those in the point set, so the geometry obeys every one of them exactly and
    the position defect is zero.  That is why the pass can keep running first
    without being reordered.
    """
    m = cubic_2x1x1(1.0, 0.5)
    r = tqg.realized_group(m, symmetrize=True, position_tol=1e-6)
    assert not isinstance(r, Refusal), str(r)
    assert (r.group.number_str, len(r.validated.indices)) == ("123.340", 32)
    assert r.descent["lattice"]["index"] == 2
    assert r.positions["defect"] == pytest.approx(0.0, abs=1e-15)
    assert r.positions["max_move"] == pytest.approx(0.0, abs=1e-15)

    # the projection ran: `symmetrize=True` forced it, nothing was near
    assert r.symmetrization is not None
    assert r.model is not m
    assert set(r.symmetrization.classes.values()) == {"exact"}
    assert len(r.symmetrization.classes) == 32

    k = np.random.default_rng(7).uniform(-0.5, 0.5, size=(6, 3))
    before = np.asarray(m.hamiltonian(k_pts=k, flatten_spin_axis=True))
    after = np.asarray(r.model.hamiltonian(k_pts=k, flatten_spin_axis=True))
    scale = float(np.abs(before).max())                      # max|H|, measured 3.9
    assert float(r.symmetrization.defect) <= _FEW_ULP * scale        # 1.78e-15
    assert float(np.abs(after - before).max()) <= _FEW_ULP * scale   # 1.33e-15
    assert dimerized.group.number_str == r.group.number_str


@pytest.mark.slow
def test_descend_false_refuses_and_names_the_translation():
    """``cell_group_not_realized`` — the upper bound, with ``τ`` named in it.

    Asked of the two-site square, whose centred point set repeats under
    ``τ = (½,½,0)``, in its default reading on a substrate: 28 of the 32
    operations the cell allows are broken, and the message says which of them is
    the pure translation and that the descent's first step would be the lattice
    descent.
    """
    r = tqg.realized_group(square_two_orbitals(), descend=False, cross_check=False)
    assert isinstance(r, Refusal) and not r
    assert r.gate == "cell_group_not_realized"
    assert r.value == 28.0 and r.context["size"] == 32
    assert "pure translation [0.5, 0.5, 0.0]" in r.reason
    assert "lattice supergroup of index 2" in r.reason


@pytest.mark.slow
def test_a_translation_obeyed_with_each_sites_orbitals_turned_is_a_supercell():
    """A copy written in a turned frame is still a copy: the twisted chain is refused ``supercell``.

    ``twisted_p_chain(δ)`` writes the second site's ``(p_x, p_y)`` in a frame turned by
    ``δ`` about ``z``: a change of basis within one site of the one-site chain written
    twice, whose bands it keeps.  The half translation is obeyed once each site's orbitals
    are carried onto its translate's by that turn, and a translation is read as every
    operation is -- on its own solution space, any change of basis within each site
    allowed -- so the model is a supercell at every ``δ``.  The refusal names the
    translation, the primitive lattice and the turn, which is **exactly ``δ``** (measured
    2.99999999998e-05, 0.1 and 0.7), and ``context["dressing"]`` carries the matrix: unitary,
    carrying each site's orbitals onto its translate's alone, and commuting with ``H(k)``.
    Above the direct residual ``2 (t_σ + t_π) δ = 7.8e-5`` (``tol = 1e-4``) the permutation
    itself is obeyed, and the turn reads ``0.0``.

    The controls:

    * a **genuine** weak dimerization of the ``s`` chain (``t = 10``, ``ε = 8e-5``,
      ``tol = 1e-4``) breaks the translation in every basis -- defect ``1.0``, direct
      residual ``1.6 × tol`` -- and the cell descends to ``P4/mmm1'``;
    * two ``(p_x, p_y)`` sites stacked along ``z``, nothing in the plane, the bonds
      ``diag(t1, t2)`` and then ``diag(t2, t1)``: the half translation carries the chain of
      ``p_x`` onto the chain of ``p_y``, a quarter turn (``π/2``), and two decoupled copies
      shifted by half a cell are a supercell;
    * two identical sublattices nothing joins, shifted by half a cell: a supercell as
      written, turn ``0.0``;
    * the twisted chain with a spin on every orbital: a supercell, the turn ``δ`` acting
      on the orbitals alone and on both states of every spinor alike, since every
      operation is read with its own spin rotation and a translation's is the identity.

    The spectral statement is asserted too: the twisted chain's bands are the untwisted
    chain's at every k, where the dimerized chain's are not.

    And the guard: a half translation composed with a quarter turn of the second site's own
    bonds is no translation.  ``screw_chain`` writes the second site's in-plane bonds as the
    first's turned by 90°, and no change of basis within the sites carries ``diag(1.0, 0.3)``
    and ``diag(0.6, 0.2)`` onto one another, so ``(0,0,½)`` is refuted by tensor structure
    and the cell descends to ``P4_2/mmc1' #131.436``, keeping the 4_2 screws, the twofolds
    about the diagonals, the ``S4`` and the c-glides -- every coset partner of the refuted
    translation whose untranslated partner fails, each a symmetry of its own measurement.
    Its ``p_x`` and ``p_y`` are uncoupled and the screw exchanges them between the sites, so
    what they carry is a representation induced from a site's ``mmm``: ``B3u`` and ``B2u``,
    which no one-dimensional character of the group times what the hoppings fix gives.
    Declared, orbital by orbital or on site ``A`` alone, they are returned so by
    declaration; undeclared, the lowest-angular-momentum rule reads the site whole and puts
    them in the one ``p`` shell holding both, where two ``Ag`` would need a ``d`` shell.
    """
    delta, t_sigma, t_pi = 3e-5, 1.0, 0.3
    ks = np.random.default_rng(7).uniform(-0.5, 0.5, size=(12, 3))
    e_twist = np.sort(np.linalg.eigvalsh(twisted_p_chain(0.3, t_sigma, t_pi).hamiltonian(k_pts=ks)), axis=-1)
    e_plain = np.sort(np.linalg.eigvalsh(twisted_p_chain(0.0, t_sigma, t_pi).hamiltonian(k_pts=ks)), axis=-1)
    assert np.abs(e_twist - e_plain).max() <= 64 * _ULP, "a basis change: the spectrum does not move"
    e_dimer = np.sort(np.linalg.eigvalsh(cubic_2x1x1(1.0, 1.3).hamiltonian(k_pts=ks)), axis=-1)
    e_uniform = np.sort(np.linalg.eigvalsh(cubic_2x1x1(1.0, 1.0).hamiltonian(k_pts=ks)), axis=-1)
    assert np.abs(e_dimer - e_uniform).max() > 0.1, "a dimerization moves the bands"

    for turn in (delta, 0.1, 0.7):
        model = twisted_p_chain(turn, t_sigma, t_pi)
        r = tqg.realized_group(model)
        assert isinstance(r, Refusal) and r.gate == "supercell", str(r)
        assert r.context["translations"] == ((0.5, 0.0, 0.0),) and r.context["index"] == 2
        assert np.allclose(r.context["primitive_lattice"]["cartesian"], np.eye(3))
        (dressing,) = r.context["dressing"]
        assert dressing["translation"] == (0.5, 0.0, 0.0)
        assert dressing["turn"] == pytest.approx(turn, rel=1e-9, abs=0), "the turn IS the frame's"
        assert f"turns them by up to {turn:.3g} rad" in r.reason
        D = np.asarray(dressing["matrix"])
        assert np.abs(D @ D.conj().T - np.eye(4)).max() <= 64 * _ULP
        assert np.abs(D[:2, :2]).max() == 0.0 and np.abs(D[2:, 2:]).max() == 0.0, \
            "each site's orbitals onto its translate's alone"
        H = np.asarray(model.hamiltonian(k_pts=ks))
        assert np.abs(np.einsum("ab,kbc,dc->kad", D, H, D.conj()) - H).max() <= 64 * _ULP
        (tau,) = r.context["lattice"]["operations"]
        assert tau["refuted_by"] is None
    # above the direct residual the permutation itself is obeyed
    obeyed = tqg.realized_group(twisted_p_chain(delta, t_sigma, t_pi), tol=1e-4)
    assert isinstance(obeyed, Refusal) and obeyed.gate == "supercell", str(obeyed)
    assert obeyed.context["lattice"]["operations"][0]["residual"] == pytest.approx(
        2 * (t_sigma + t_pi) * delta, rel=1e-6, abs=0)
    assert [d["turn"] for d in obeyed.context["dressing"]] == [0.0]
    assert "change of basis" not in obeyed.reason

    # control 1: a genuine weak dimerization is broken in every basis
    dimer = tqg.realized_group(cubic_2x1x1(10.0, 10.0 + 8e-5), tol=1e-4)
    assert dimer and (dimer.group.number_str, len(dimer.validated.indices)) == ("123.340", 32)
    (tau_d,) = dimer.descent["lattice"]["operations"]
    assert tau_d["refuted_by"] == "tensor structure" and tau_d["scaled"] == pytest.approx(1.0, abs=1e-9)
    assert tau_d["residual"] == pytest.approx(1.6e-4, rel=1e-6, abs=0), "1.6 x tol"
    assert dimer.descent["lattice"]["dressing"] == ()

    # control 2: two decoupled copies a quarter turn apart, shifted by half a cell
    copies = tqg.realized_group(uncoupled_screw_chain())
    assert isinstance(copies, Refusal) and copies.gate == "supercell", str(copies)
    assert copies.context["translations"] == ((0.0, 0.0, 0.5),)
    assert [d["turn"] for d in copies.context["dressing"]] == [pytest.approx(np.pi / 2, abs=1e-12)]

    # control 3: identical sublattices nothing joins, a supercell as written
    alike = tqg.realized_group(uncoupled_sublattices())
    assert isinstance(alike, Refusal) and alike.gate == "supercell", str(alike)
    assert alike.context["translations"] == ((0.5, 0.0, 0.0),)
    assert [d["turn"] for d in alike.context["dressing"]] == [0.0]

    # control 4: with a spin on every orbital the turn acts on the orbitals alone
    spun = tqg.realized_group(twisted_p_chain(0.1, t_sigma, t_pi, spinful=True))
    assert isinstance(spun, Refusal) and spun.gate == "supercell", str(spun)
    (dressing,) = spun.context["dressing"]
    assert dressing["turn"] == pytest.approx(0.1, rel=1e-9, abs=0)
    D = np.asarray(dressing["matrix"])
    assert max(np.abs(D[0::2, 1::2]).max(), np.abs(D[1::2, 0::2]).max(),
               np.abs(D[0::2, 0::2] - D[1::2, 1::2]).max()) <= 64 * _ULP, "the spin untouched"

    # the guard: the screw a quarter turn makes of the half translation stands, and
    # what its uncoupled p_x, p_y carry is read off every representation induced on them
    for said, by in ((None, ["lowest-l"] * 4),
                     (["px", "py", "px", "py"], ["declaration"] * 4),
                     ({0: "px", 1: "py"}, ["declaration"] * 2 + ["lowest-l"] * 2)):
        screw = tqg.realized_group(screw_chain(), characters=said, cross_check=False)
        assert screw and (screw.group.number_str, screw.group.name) == ("131.436", "P4_2/mmc1'")
        assert len(screw.validated.indices) == 32 and screw.descent["steps"][0]["n_dropped"] == 32
        (tau_s,) = screw.descent["lattice"]["operations"]
        assert tau_s["translation"] == (0.0, 0.0, 0.5) and tau_s["refuted_by"] == "tensor structure"
        partners = [np.rint(np.asarray(s.rotation)).astype(int) for s in screw.group.symmetries
                    if not s.time_reversal and abs(float(s.translation[2]) % 1.0 - 0.5) < 1e-9]
        assert len(partners) == 8 and all(abs(R[0, 1]) == 1 for R in partners), \
            "the screws, diagonal twofolds, S4 and c-glides: every rotation exchanging x and y"
        rows = screw.rep.characters.orbitals
        assert [o.irrep for o in rows] == ["B3u", "B2u", "B3u", "B2u"], said
        assert {o.site_symmetry for o in rows} == {"mmm"}
        assert [o.determined_by for o in rows] == by, said


# ---------------------------------------------------------------------------
# 2. p4232: the bcc point set in a cubic cell
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("route", ["descended", "decorated"])
def test_p4232_both_routes_give_the_same_group_and_letters(route, request):
    """``P4_232 1' #208.45``, 48 of 48, from the Hamiltonian and from the geometry.

    ``realized_group(m208)`` measures the centring translation
    (covariance residual 2.4 against ``tol = 1e-8``), refutes it and its coset,
    and descends; the paper's own decoration — the 24-fold orbit of a generic
    point, ``p4232_symmetry_cell()`` — reads the same group off the geometry
    before any Hamiltonian is consulted.  The ``(d_z2, d_x2-y2)`` doublet is
    the channel layout's own business: the site's two orbitals mix under its
    threefold, so no partition finer than the pair admits a block-form
    intertwiner and the measured layout glues them.
    """
    if route == "descended":
        # `realized_group(m208)`, the session's shared record
        r = request.getfixturevalue("p4232_descended")
        assert not isinstance(r, Refusal), str(r)
        group, assignment, validated = r.group, r.assignment, r.validated
        assert group.provenance == "descended"
        step = r.descent["lattice"]
        assert step["index"] == 2
        assert step["fractional_translations"] == ((0.5, 0.5, 0.5),)
        assert step["realized"] == ()
        assert step["coset_partners_refuted"] == 71
        assert step["operations"][0]["residual"] == pytest.approx(2.4, abs=5e-2)
        # the residual is O(1) against the gate it is measured at, which is
        # what makes the refutation a measurement and not a near miss
        assert step["operations"][0]["tol"] == 1e-8
    else:
        # `_group_of` of `p4232_symmetry_cell()`, then `_assign_orbitals` and
        # `_validate_group` on the golden: the session's shared decorated record
        _model, group, assignment, validated, _rep = request.getfixturevalue(
            "p4232_decorated")
        assert group._lattice_index == 1
    assert (group.number_str, group.name, group.size) == ("208.45", "P4_2321'", 48)
    assert len(validated.indices) == 48
    assert assignment.wyckoff_tables == ("a", "a")


# ---------------------------------------------------------------------------
# 3. SG 199: nothing is wrong with the crystal, only with the cell
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_i213_conventional_cell_is_refused_as_a_supercell():
    """``supercell`` naming ``(½,½,½)`` — the message the fixture always wanted.

    ``_group_of`` still knows the crystal perfectly well (``I2_131' #199.13``);
    what it now says about the conventional cell is that it is an index-2
    supercell of the body-centred lattice, and the Hamiltonian obeying the
    centring translation (residual below ``1e-12``, measured 3.5e-15 here) is
    what makes the model a supercell rather than a crystal with a bigger cell.
    The refusal names ``τ`` reduced and Cartesian, the index, and the finer
    lattice, and says what a folded zone would do to every label; reached
    inside the measurement, it opens with the cell's group at that point, the
    upper bound it is, and carries it.

    ``_translation_residual`` measures ``max_k |P H(k) P† − H(k)|``, and *which
    way round* it conjugates is not a claim this file can make: ``P`` is a
    permutation matrix, so ``P†(H − P H P†)P = P† H P − H``, and a max-abs norm
    is invariant under a simultaneous row/column permutation.  The two readings
    are therefore equal entry for entry on every cell, and no test can separate
    them.
    """
    g = _spacegroup._group_of(tqc_models.i213_conventional())
    assert not isinstance(g, Refusal), str(g)
    assert (g.number_str, g.name, g.size) == ("199.13", "I2_131'", 48)
    assert g._lattice_index == 2 and g._fractional_translations == ((0.5, 0.5, 0.5),)

    r = tqg.realized_group(tqc_models.i213_conventional())
    assert isinstance(r, Refusal) and not r
    assert r.gate == "supercell" and r.value == 2.0 and r.threshold == 1.0
    assert r.reason.startswith("The cell's group at this point is I2_131' #199.13, an "
                               "upper bound on what this Hamiltonian obeys"), r.reason
    assert set(r.context) >= {"translations", "index", "primitive_lattice", "group",
                              "validated"}
    assert r.context["group"].number_str == "199.13"
    assert r.context["translations"] == ((0.5, 0.5, 0.5),)
    assert r.context["index"] == 2
    assert "[0.5, 0.5, 0.5] (reduced) = [0.5, 0.5, 0.5] (Cartesian)" in r.reason
    assert "folded Brillouin zone" in r.reason
    assert "primitive cell" in r.reason
    assert r.context["lattice"]["operations"][0]["residual"] == pytest.approx(0.0, abs=1e-12)
    # the finer lattice is ``Z^3 + Z.tau``, which is what "body centred" means.
    # It takes two facts to say that and neither does it alone: the index, and
    # that ``tau`` itself is a lattice vector of the basis returned.  |det| = 1/2
    # with half-integer entries does not pin it — diag(1/2, 1, 1) has both and
    # does not contain (1/2, 1/2, 1/2).
    basis = np.asarray(r.context["primitive_lattice"]["reduced"])
    assert abs(abs(float(np.linalg.det(basis))) - 0.5) < 1e-12
    n = np.linalg.solve(basis.T, np.asarray(r.context["translations"][0], dtype=float))
    np.testing.assert_allclose(n, np.rint(n), atol=1e-12)
    # ... and the basis as measured, in the Hermite form `_primitive_lattice`
    # returns: the centring vector and two integer vectors, not three members of
    # Z^3 + (1/2,1/2,1/2)
    np.testing.assert_allclose(basis, [[0.5, 0.5, 0.5], [0.0, 1.0, 1.0],
                                       [0.0, 0.0, 1.0]], atol=1e-12)


# ---------------------------------------------------------------------------
# 4. the square cells of test_groups.py
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_the_square_cells_hamiltonian_breaks_the_centring():
    """``±0.3`` on the two sites breaks ``τ``; the model descends to ``Amm21' #38.188``.

    Measured on the **free-standing** reading, where the cell is the point set
    as given; a ``dim_r = 2`` model is read on a substrate unless told
    otherwise, so the reading is declared.  Both readings reach the lattice
    question (the shadow goes under *every* site, so it never breaks
    a fractional in-plane translation geometrically), and they differ only in
    the ``σ_h`` family: free-standing ``Amm21' #38.188``, 8 of 8, from a
    64-operation cell; on a substrate the in-plane image of it, ``Cm1' #8.33``,
    4 of 4, from a 32-operation one.  Both are measured below.
    """
    r = tqg.realized_group(square_two_orbitals(), environment="free-standing",
                           cross_check=False)
    assert not isinstance(r, Refusal), str(r)
    assert (r.group.number_str, r.group.name) == ("38.188", "Amm21'")
    assert (len(r.validated.indices), r.group.size) == (8, 8)
    assert r.group.provenance == "descended"
    step = r.descent["lattice"]
    assert step["index"] == 2 and step["realized"] == ()
    assert step["operations"][0]["translation"] == (0.5, 0.5, 0.0)
    assert step["operations"][0]["refuted_by"] == "tensor structure"
    assert step["operations"][0]["residual"] == pytest.approx(2.0, abs=1e-9)

    assert step["coset_partners_refuted"] == 27

    # the default reading -- the substrate -- reaches the same lattice question
    # and serves the in-plane image of that group: half the operations, and the
    # translation refuted at the same residual from a cell half the size
    plain = tqg.realized_group(square_two_orbitals())
    assert not isinstance(plain, Refusal), str(plain)
    assert plain.environment == "substrate" and plain.environment_declared is False
    assert (plain.group.number_str, plain.group.name) == ("8.33", "Cm1'")
    assert (len(plain.validated.indices), plain.group.size) == (4, 4)
    assert plain.descent["from"]["number_str"] == "99.164"
    assert plain.descent["lattice"]["index"] == 2
    assert plain.descent["lattice"]["coset_partners_refuted"] == 13
    assert plain.descent["lattice"]["operations"][0]["residual"] == pytest.approx(
        2.0, abs=1e-9)
    assert plain.cross_check["outcome"] == "agree"


# ---------------------------------------------------------------------------
# 5. the lifted Fu-Kane-Mele supercells: the cost of the honest route
# ---------------------------------------------------------------------------

def _inversion_permutation(model):
    """Which orbital each one maps to under the cell's own inversion centre."""
    pos = np.asarray(model.orb_vecs, dtype=float) % 1.0
    for j in range(pos.shape[0]):
        target = (pos[0] + pos[j] - pos) % 1.0
        d = np.abs((pos[None, :, :] - target[:, None, :] + 0.5) % 1.0 - 0.5).max(axis=2)
        if d.min(axis=1).max() < 1e-9:
            perm = d.argmin(axis=1)
            if len(set(perm.tolist())) == pos.shape[0]:
                return np.asarray(perm)
    raise AssertionError("no inversion centre in this orbital set")


def _lifted_supercell(sc, lift=0.3, seed=0):
    """``test_symmetry.py``'s construction: a supercell with its folding lifted."""
    model = tqc_models.fu_kane_mele_z2("strong").make_supercell(sc, to_home=True)
    perm = _inversion_permutation(model)
    rng = np.random.default_rng(seed)
    e = rng.normal(size=model.norb)
    e = 0.5 * (e + e[perm])                       # exactly symmetric under inversion
    for i in range(model.norb):
        model.set_onsite(lift * float(e[i]), i, mode="add")
    return model, [int(min(i, int(perm[i]))) for i in range(model.norb)]


@pytest.mark.slow
@pytest.mark.parametrize("sc, served, size, index, thirds, typed_number", [
    ([[2, 0, 0], [0, 2, 0], [0, 0, 2]], ("227.129", "Fd-3m1'"), 768, 8, False, "2.5"),
    ([[3, 0, 0], [0, 1, 0], [0, 0, 1]], ("166.98", "R-3m1'"), 72, 3, True, "164.86"),
], ids=["2x2x2", "3x1x1"])
def test_the_lifted_fkm_supercells_are_served_as_lattice_supergroups(
        sc, served, size, index, thirds, typed_number):
    """The lattice supergroup of the cell as it stands, at index 8 and 3.

    The random on-site potential is exactly inversion-symmetric and moves no
    atom, so the *positions* still carry the small cell's translations and the
    cell is an index-``det(sc)`` supercell of the diamond primitive cell.  The
    ``2x2x2`` cell keeps the whole cubic point group, ``Fd-3m1' #227.129``: 48
    rotations times 8 cosets, 768 operations, measured at 1.3 s for
    ``_group_of`` (the closure test is ``O(N²)`` in the operation count).  The
    ``3x1x1`` cell's own lattice keeps only the rotations of ``R-3m``: 72
    operations, 12 rotations times 3 cosets and time reversal.

    The ``3x1x1`` cell's translations are exact **thirds**, which cannot close
    to a bit-exact zero in binary floating point: its operation closure
    measures 1.1e-16, eight decades below the 2.0e-8 of a genuinely numerical
    cell, and the record carries that number with nothing repaired.

    That pair of numbers is the closure *measurement's* own test, which is why
    it is asserted here as a band and not as ``< 1e-12``: ``1e-12`` is verbatim
    the library's ``_EXACT_CLOSURE`` gate, so once the call is served the bound
    cannot fail, and a ``_translation_closure`` that answered ``0.0`` for every
    cell would satisfy it too.  ``0.0 < closure <= 16 ulp`` on the thirds cell is
    a band nothing but the arithmetic hands you, and ``== 0.0`` on the dyadic
    ``2x2x2`` cell is the other half of it.
    """
    model, typat = _lifted_supercell(sc)
    g = _spacegroup._group_of(model)
    assert not isinstance(g, Refusal), str(g)
    assert (g.number_str, g.name, g.size) == (*served, size)
    assert g._lattice_index == index
    assert g._tables_ready is False
    if thirds:                        # thirds: 1.1e-16, one ulp, and not zero
        assert 0.0 < g._closure_measured <= _FEW_ULP
    else:                             # halves: the closure sum is bit-exact
        assert g._closure_measured == 0.0
    assert g._closure_repaired == 0.0, "and nothing was repaired"
    # the typed route is unchanged and is what the suite uses everywhere else
    typed = _spacegroup._group_of(model, typat=typat)
    assert typed and typed.number_str == typed_number and typed._lattice_index == 1


# ---------------------------------------------------------------------------
# 6. the consumers' gate
# ---------------------------------------------------------------------------

#: Every entry point that reads a table, called on the group (or representation)
#: in its first slot.
_TABLE_CONSUMERS = {
    "wyckoff_positions_spglib": lambda g: tqg.wyckoff_positions_spglib(g),
    "wyckoff_sites_tables": lambda g: tqg.wyckoff_sites_tables(g),
    "high_symmetry_points": lambda g: tqg.high_symmetry_points(g),
    "little_group": lambda g: tqg.little_group(g, [0, 0, 0]),
    "corep_table": lambda g: tqg.corep_table(g, [0, 0, 0]),
    "identify_from_characters": lambda g: tqg.identify_from_characters(g, [0, 0, 0],
                                                                       {1: 1.0}),
    "ebr_data": lambda g: tqg.ebr_data(g),
    "indicator_group": lambda g: tqg.indicator_group(g),
    "decompose_ebrs": lambda g: tqg.decompose_ebrs(g, {}),
    "k_grid": lambda g: tqg.k_grid(g, 4),
}


def test_every_table_consumer_refuses_a_lattice_supergroup():
    """One gate, ``lattice_not_descended``, wherever a table would be read.

    The refusal names ``τ``, says the tables live on the finer lattice and would
    label a folded zone, and points at ``realized_group`` (which measures the
    Hamiltonian), at reducing the cell, and at ``typat=``.  Asked of the
    ``2x1`` square supercell's lattice supergroup, whose ``τ = (½,0,0)`` is the
    ``2x1x1`` chain's.
    """
    supergroup = _spacegroup._group_of(supercell_square())
    assert not isinstance(supergroup, Refusal), str(supergroup)
    assert supergroup._lattice_index == 2
    for name, call in _TABLE_CONSUMERS.items():
        r = call(supergroup)
        assert isinstance(r, Refusal) and not r, name
        assert r.gate == "lattice_not_descended", name
        assert r.value == 2.0 and r.threshold == 1.0, name
        assert "[0.5, 0.0, 0.0]" in r.reason, name
        assert "folded zone" in r.reason, name
        assert "realized_group" in r.reason and "typat" in r.reason, name
        assert r.context["lattice_index"] == 2, name
        assert r.context["translations"] == ((0.5, 0.0, 0.0),), name


#: what an entry point is handed when the call meant to build its group refused
_EARLIER_REFUSAL = Refusal("an_earlier_gate", "the call that was meant to build it refused",
                           value=1.0, threshold=0.0)


def test_every_table_entry_point_propagates_a_refusal_handed_in():
    """A refusal in the group (or representation) slot comes back unchanged.

    ``_group_of`` and ``_build_rep`` serve a refusal rather than raising, so a
    script that does not branch on it hands that refusal straight to the next
    call.  The first verdict is the one to read, and it is the one served: the
    same object, gate, reason and context, and no attribute of a group looked
    for on something that is not one.
    """
    entry_points = dict(
        _TABLE_CONSUMERS,
        characters=lambda x: tqsew.characters(x, [0, 0, 0], bands=1),
        corep_content=lambda x: tqsew.corep_content(x, [0, 0, 0], bands=1),
        hsp_table=lambda x: tqsew.hsp_table(x, bands=1))
    assert len(entry_points) == 13
    for name, call in entry_points.items():
        served = call(_EARLIER_REFUSAL)
        assert served is _EARLIER_REFUSAL, name
        assert served.gate == "an_earlier_gate", name


@pytest.mark.slow
def test_the_assignment_is_geometric_on_a_supergroup(cubic_supergroup):
    """Orbits, permutations and shifts are measured; every letter is empty.

    A Wyckoff letter is a table's word and the tables here belong to the finer
    lattice, so ``_assign_orbitals`` reports none of them — while the geometry the
    detector and the tensor test read (orbits, the permutation, the lattice
    shift, the site-symmetry groups) is measured from the operations exactly as
    always.  The two sites are one orbit under the supergroup, since ``τ`` swaps
    them.

    The permutation and the shift are asserted on the row where the geometry
    hands the answer independently.  Under ``{1|(½,0,0)}`` the site at ``(0,0,0)``
    goes to ``(½,0,0)`` — site 1, inside the home cell — and the site at
    ``(½,0,0)`` goes to ``(1,0,0)``, which is site 0 **one lattice vector along
    x**.  So ``perm == [1, 0]`` and ``lattice_shift == [[0,0,0], [1,0,0]]``, and
    the second entry is the one a shift that was never measured cannot produce.
    """
    a = _tables._assign_orbitals(cubic_2x1x1(1.0, 0.5), cubic_supergroup)
    assert not isinstance(a, Refusal), str(a)
    assert a.orbits == ((0, 1),)
    assert a.wyckoff_spglib == ("",)
    assert a.wyckoff_tables == ()
    assert a.wyckoff_spglib_dataset == ()
    # ... while the site-symmetry groups, and their names, are read off the
    # operations like the rest of the geometry
    assert a.site_symmetry == ("4/mmm", "4/mmm")
    assert a.permutations.shape == (64, 2)
    assert a.max_distance == pytest.approx(0.0, abs=1e-12)

    # the translation's own row, against the geometry above
    ident = np.eye(3, dtype=int)
    rows = [i for i, s in enumerate(cubic_supergroup.symmetries)
            if not s.time_reversal
            and np.array_equal(ident, np.rint(np.asarray(s.rotation,
                                                         dtype=float)).astype(int))]
    tau_row, = [i for i in rows
                if np.allclose(cubic_supergroup.symmetries[i].translation,
                               [0.5, 0.0, 0.0], atol=1e-12)]
    id_row, = [i for i in rows if i != tau_row]
    assert a.permutations[tau_row].tolist() == [1, 0]
    assert np.asarray(a.lattice_shift)[tau_row].tolist() == [[0, 0, 0], [1, 0, 0]]
    assert a.permutations[id_row].tolist() == [0, 1]
    assert np.asarray(a.lattice_shift)[id_row].tolist() == [[0, 0, 0], [0, 0, 0]]


# ---------------------------------------------------------------------------
# 7. 2D: the flat P4 1' cell reaches the lattice descent in the layer reading
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_the_flat_p41_cell_reaches_the_lattice_descent_and_holds_each_operation_to_its_lift(
        p41_free_standing):
    """The flat two-site cell: ``τ`` refuted, then the ``σ_h`` family, then ``#99.164``.

    The flat point set of the SRSI paper's ``P41'`` model has its two sites at
    ``(½,0)`` and ``(0,½)``, so it repeats under ``τ = (½,½,0)`` and the slab
    cell allows the supergroup ``P4/mmm1' #123.340``, 64 operations.  The
    Hamiltonian breaks ``τ`` (measured covariance residual 1.0 against ``tol =
    1e-8``) and 23 coset partners with it, and the tensor test then refutes the
    horizontal-mirror family — 48 refutations in all, every one ``"tensor structure"``.

    Each operation is held to its own lift, so of every ``σ_h`` fiber one
    member survives, and the survivors are the eight that make ``P4mm1'``:
    ``S4(z)`` and ``S4⁻¹(z)``, which would pass on their partners' lifts --
    ``S(C4)† = −S(σ_h)·S(C4)`` at ``n = 4`` -- are refuted with the rest, so
    nothing is left for the closure resolution.
    The reading is declared, since a ``dim_r = 2`` model is read on a substrate
    by default and the ``σ_h`` family this test is about is then not in the
    cell at all.
    """
    # `realized_group(tqc_models.p41_flat_band(), environment="free-standing")`,
    # the session's shared record
    r = p41_free_standing
    assert not isinstance(r, Refusal), str(r)
    assert (r.group.number_str, r.group.name) == ("99.164", "P4mm1'")
    assert (len(r.validated.indices), r.group.size) == (16, 16)
    assert r.group.provenance == "descended"
    # the s orbitals, assigned by the lowest-angular-momentum rule: A1 of the site group,
    # returned on this spinful model as the corep -E
    assert r.rep.characters.irreps == ("-E", "-E")
    assert r.assignment.wyckoff_spglib == ("c",)

    d = r.descent
    assert d["from"] == {"number_str": "123.340", "name": "P4/mmm1'", "size": 64}
    assert d["to"] == {"number_str": "99.164", "name": "P4mm1'", "size": 16}
    assert d["steps"][0]["n_dropped"] == 48
    assert d["steps"][0]["refuted_by"] == {"tensor structure": 48}
    assert {b["refuted_by"] for b in d["dropped"]
            if b["name"] in ("S4(z)", "S4^-1(z)")} == {"tensor structure"}
    assert d["steps"][0]["kept"] == (1, 5, 9, 13, 19, 23, 27, 31)

    step = d["lattice"]
    assert step["index"] == 2 and step["realized"] == ()
    assert step["coset_partners_refuted"] == 23
    (tau,) = step["operations"]
    assert tau["translation"] == (0.5, 0.5, 0.0)
    assert tau["refuted_by"] == "tensor structure"
    assert tau["residual"] == pytest.approx(1.0, abs=1e-9)


@pytest.mark.slow
def test_the_substrate_decoration_does_not_hide_the_translation():
    """The substrate's shadow runs under every site, and ``τ`` survives it.

    :func:`pythtqc.groups._substrate_decorations` places **one** shadow species
    at ``−dz`` under every projected site rather than once at the origin.  On
    ``supercell_square()`` that is the difference between a cell whose point set
    still repeats under ``τ = (½,0,0)`` and one whose repeat a decoration at the
    origin alone had broken: the decorated cell reads ``Pmm21' #25.58`` at
    lattice index 2, and ``realized_group`` reaches the lattice question and
    **refuses** ``supercell`` — the honest answer for a cell that is two copies
    of the primitive square one.  The verdict is the lattice's and neither
    reading changes it: the bare cell reads the supergroup ``P4/mmm #123``, 16
    operations in 2 cosets of 8, and asked as a **free-standing** layer --
    where the cell is the point set as it stands -- the model is refused the
    same way, with the same ``τ`` at the same residual (2.0e-16 measured).
    Before wave R4b the geometry hid the translation and the call served
    ``#25.58``, 8 of 8, on a folded zone.

    Re-pinned in wave R4c for the recipe itself: one shadow below, not a pair at
    ``±dz``.  The depth is written out as ``-0.05`` rather than read back off
    ``DECORATION_DZ``, so that half of the claim is a pin and not ``f(x) == x``.
    Two species removed the same operations and described a layer
    encapsulated between two different materials; one shadow is the physics of a
    substrate.  The decorated cell here therefore carries 4
    orbitals rather than 6, of two species rather than three, and reads the same
    ``Pmm21' #25.58`` at the same lattice index.

    What the recipe is *for* is unchanged, and the honeycomb beside it is the
    measurement: a point set with no fractional repeat keeps lattice index 1,
    and the decoration removes exactly the operations with ``R[2,2] = −1`` —
    ``P-6m21' #187.210``, 24 operations, down to ``P3m11' #156.50``, 12.
    """
    m3 = _layers._slab_presentation(supercell_square(), c=tqg.SLAB_C)
    assert _layers._substrate_decorations(m3, [0, 0]) == (
        ([0.0, 0.0, -0.05], 1), ([0.5, 0.0, -0.05], 1))
    cell, cell_typat = _layers._decorated_cell(m3, [0, 0])
    assert int(cell.norb) == 4 and list(cell_typat) == [0, 0, 1, 1]
    g = _spacegroup._group_of(cell, typat=cell_typat)
    assert not isinstance(g, Refusal), str(g)
    assert (g.number_str, g.name, g.size) == ("25.58", "Pmm21'", 16)
    assert g._lattice_index == 2 and g._fractional_translations == ((0.5, 0.0, 0.0),)

    r = tqg.realized_group(supercell_square())
    assert isinstance(r, Refusal) and not r
    assert r.gate == "supercell" and r.value == 2.0
    assert r.context["translations"] == ((0.5, 0.0, 0.0),)
    assert r.context["lattice"]["operations"][0]["residual"] == pytest.approx(0.0, abs=1e-14)

    # ... and the bare cell, read free-standing: the same supergroup's refusal
    bare = _spacegroup._group_of(supercell_square(), include_TR=False)
    assert not isinstance(bare, Refusal), str(bare)
    assert (bare.number_str, bare.size) == ("123", 16)
    assert bare._lattice_index == 2 and bare._fractional_translations == ((0.5, 0.0, 0.0),)
    free = tqg.realized_group(supercell_square(), environment="free-standing",
                              cross_check=False)
    assert isinstance(free, Refusal) and not free
    assert free.gate == "supercell" and free.value == 2.0
    assert free.context["translations"] == ((0.5, 0.0, 0.0),)
    assert free.context["lattice"]["operations"][0]["residual"] == pytest.approx(
        0.0, abs=1e-14)

    # the honeycomb: no fractional repeat, and the recipe does its own job
    m3 = _layers._slab_presentation(tqc_models.kane_mele_z2("odd"))
    bare = _spacegroup._group_of(m3, typat=[0, 1], spinor=True)
    assert not isinstance(bare, Refusal), str(bare)
    assert (bare.number_str, bare.size) == ("187.210", 24)
    decorated, decorated_typat = _layers._decorated_cell(m3, [0, 1])
    h = _spacegroup._group_of(decorated, typat=decorated_typat, spinor=True)
    assert not isinstance(h, Refusal), str(h)
    assert (h.number_str, h.name, h.size) == ("156.50", "P3m11'", 12)
    assert h._lattice_index == 1 and h._fractional_translations == ()
    assert {int(np.rint(np.asarray(s.rotation, dtype=float)[2, 2]))
            for s in bare.symmetries} == {1, -1}
    assert {int(np.rint(np.asarray(s.rotation, dtype=float)[2, 2]))
            for s in h.symmetries} == {1}


# ---------------------------------------------------------------------------
# the lattice branch: a translation realized only up to a gauge (2026-09-09)

#: pi flux per plaquette on the 2x2 cell, in a sign gauge under which NO bare
#: translation survives: ``PI_FLUX_SX[(x, y)]`` is the sign of the x hop leaving
#: site ``(x, y)`` (plaquette units), ``PI_FLUX_SY`` the same for the y hop.
#: Found by brute force over the 256 period-2 sign patterns with uniform pi
#: flux; every one of them lets tau_x and tau_y be restored by a gauge
#: transformation periodic with the cell (equal holonomies along both cycles,
#: since 2 pi of flux sits between any two rows or columns), and this one also
#: breaks the diagonal tau_x tau_y, which (-1)^(x+y) on the x hops would not.
PI_FLUX_SITES = ((0, 0), (1, 0), (0, 1), (1, 1))
PI_FLUX_SX = {(0, 0): 1, (1, 0): -1, (0, 1): 1, (1, 1): -1}
PI_FLUX_SY = {(0, 0): 1, (1, 0): -1, (0, 1): -1, (1, 1): 1}


def pi_flux_2x2(t=1.0, tz=1.0):
    """Square-lattice pi flux on its 2x2 cell, stacked along z with no flux.

    The point set -- sites at every half-integer of ``diag(2, 2, 1)`` -- is the
    simple-cubic one, so the cell repeats under ``tau_x = (1/2, 0, 0)``,
    ``tau_y`` and ``tau_x tau_y`` (lattice index 4).  The Hamiltonian repeats
    under none of them as written: the x hops carry the signs above, and a
    shift by one plaquette changes them by the gauge transformation
    ``psi -> (-1)^x psi``, a sign per site.
    """
    idx = {s: i for i, s in enumerate(PI_FLUX_SITES)}
    m = TBModel(Lattice(np.diag([2.0, 2.0, 1.0]),
                        [[x / 2, y / 2, 0.0] for x, y in PI_FLUX_SITES],
                        periodic_dirs=[0, 1, 2]), spinful=False)
    for (x, y), i in idx.items():
        m.set_hop(-t * PI_FLUX_SX[(x, y)], i, idx[((x + 1) % 2, y)],
                  [1 if x == 1 else 0, 0, 0])
        m.set_hop(-t * PI_FLUX_SY[(x, y)], i, idx[(x, (y + 1) % 2)],
                  [0, 1 if y == 1 else 0, 0])
        m.set_hop(-tz, i, i, [0, 0, 1])
    return m


def pi_flux_landau_2x1(t=1.0, tz=1.0):
    """The same flux in Landau gauge on the 2x1 magnetic cell: y hops carry ``(-1)^x``."""
    m = TBModel(Lattice(np.diag([2.0, 1.0, 1.0]), [[0, 0, 0], [0.5, 0, 0]],
                        periodic_dirs=[0, 1, 2]), spinful=False)
    m.set_hop(-t, 0, 1, [0, 0, 0])
    m.set_hop(-t, 1, 0, [1, 0, 0])
    m.set_hop(-t, 0, 0, [0, 1, 0])
    m.set_hop(+t, 1, 1, [0, 1, 0])
    for o in (0, 1):
        m.set_hop(-tz, o, o, [0, 0, 1])
    return m


def _bare_symmetries_of_the_sign_pattern(sx, sy):
    """Every operation of D4h, about every centre that maps the 2x2 site set to itself,
    under which the hopping SIGN PATTERN is invariant as written -- no gauge transformation
    allowed.  Pure combinatorics on the hopping list: nothing from ``pythtqc`` enters.
    Returned as ``{(rotation rows, translation mod 1)}`` in the reduced coordinates of the
    ``2x2x1`` cell, the currency of ``group.symmetries``."""
    bonds = {}
    for (x, y) in sx:
        bonds[((x, y), (1, 0, 0))] = sx[(x, y)]
        bonds[((x, y), (0, 1, 0))] = sy[(x, y)]
        bonds[((x, y), (0, 0, 1))] = 1                 # the z hop, uniform

    def canon(site, d):
        x, y = site
        if d[0] < 0:
            return ((x - 1) % 2, y), (1, 0, 0)
        if d[1] < 0:
            return (x, (y - 1) % 2), (0, 1, 0)
        if d[2] < 0:
            return (x % 2, y % 2), (0, 0, 1)
        return (x % 2, y % 2), d

    point = []
    for perm in ((0, 1), (1, 0)):
        for sgn in itertools.product((1, -1), repeat=3):
            R = np.zeros((3, 3), dtype=int)
            R[0, perm[0]], R[1, perm[1]], R[2, 2] = sgn
            point.append(R)
    out = set()
    for R in point:
        for tx, ty in itertools.product((0, 0.5, 1, 1.5), repeat=2):
            t = np.array([tx, ty, 0.0])
            imgs = [R @ np.array([x, y, 0.0]) + t for x, y in sx]
            if not all(abs(q - np.rint(q)).max() < 1e-9 for q in imgs):
                continue
            ok = True
            for (site, d), sgn in bonds.items():
                q = R @ np.array([*site, 0.0]) + t
                d_img = tuple(int(v) for v in R @ np.array(d))
                b = canon((int(round(q[0])) % 2, int(round(q[1])) % 2), d_img)
                if bonds[b] != sgn:
                    ok = False
                    break
            if ok:
                t_red = tuple(float(v) for v in np.round(np.array([tx / 2, ty / 2, 0.0]) % 1.0, 6))
                out.add((tuple(map(tuple, R.tolist())), t_red))
    return out


@pytest.mark.slow
def test_magnetic_translations_leave_the_cell_its_bare_subgroup():
    """pi flux on its 2x2 cell: the descent lands on the subgroup the sign pattern has as written.

    The cell's point set repeats under ``tau_x``, ``tau_y`` and ``tau_x tau_y``.  The
    Hamiltonian obeys none of them as written -- the direct check ``max_k |P H P^dag - H|``
    reads ``4t``, a whole sign on every x hop -- and each of them once the orbitals are
    carried onto their translates' with a sign per site, ``psi -> (-1)^x psi``: a change of
    basis within each site, which a translation may carry.  But the three signs
    anticommute pairwise (measured ``D_x D_y D_x^-1 D_y^-1 = -1`` to 1e-15): the pi flux
    through every plaquette.  Translations commute, so no translation group of a crystal is
    obeyed -- these are magnetic translations -- and only the translations obeyed as written
    count, of which there are none.  The translations are then refuted as the tensor
    structure refutes them, by one scalar per channel class -- the freedom ``_build_rep``
    has -- of which a sign per site inside one class is a defect of exactly 1, and every
    dressed operation goes with them (112 of 128): the cell descends to ``Pmma1' #51.290``,
    the identity, a twofold about ``(1/4, 0)``, a mirror at ``y = 1/4``, an x-mirror
    composed with a half-cell translation along y (a glide), each with and without
    ``z -> -z``.  Read one at a time, without the commutation, the three translations would
    each hold, and the 2x2 cell would be refused ``supercell`` on the 1x1 lattice no pi flux
    can be written on.

    The oracle owes nothing to the code: the same eight operations come out of a
    brute-force enumeration of D4h about every admissible centre, keeping the operations
    under which the hopping SIGN PATTERN is invariant as written.  Which operations are
    bare is a gauge-dependent notion -- the Landau-gauge 2x1 cell of the same flux
    descends to ``Pmmm1' #47.250``, since there the sign a column shift needs is the boost
    ``k_y -> k_y + pi``, no change of basis within a site, and no point-group k-action can
    pose it -- and the magnetic symmetry common to both gauges is the unsupported feature.
    """
    m = pi_flux_2x2()
    g = _spacegroup._group_of(m)
    assert (g.number_str, g.size, g._lattice_index) == ("123.340", 128, 4)
    taus = {(0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.5, 0.5, 0.0)}
    assert set(g._fractional_translations) == taus

    r = tqg.realized_group(m)
    assert r, str(r)
    assert (r.group.number_str, r.group.name, r.group.size) == ("51.290", "Pmma1'", 16)
    assert r.group._lattice_index == 1, "the cell's own translations are the lattice"
    d = r.descent
    assert d["from"]["number_str"] == "123.340" and d["to"]["number_str"] == "51.290"
    assert d["steps"][0]["n_dropped"] == 112 and d["steps"][0]["refuted_by"] == {"tensor structure": 112}

    # the three translations: kept by the detector, refused by the ansatz at exactly 1
    # (a sign per site inside the one channel class has no component along the ansatz),
    # and the direct check -- 4t, a whole sign on every x hop -- was never needed
    ops = {e["translation"]: e for e in d["lattice"]["operations"]}
    assert set(ops) == taus
    for e in ops.values():
        assert e["refuted_by"] == "tensor structure", e
        assert e["scaled"] == pytest.approx(1.0, abs=1e-12), e
        assert e["residual"] == pytest.approx(4.0, rel=1e-12, abs=0), e
    assert d["lattice"]["refuted"] == {} and d["lattice"]["realized"] == ()

    # the oracle: the operations under which the sign pattern is invariant as written
    served = set()
    for s in r.group.symmetries:
        if s.time_reversal:
            continue
        R = np.rint(np.asarray(s.rotation)).astype(int)
        t = tuple(float(v) for v in np.round(np.asarray(s.translation, dtype=float) % 1.0, 6))
        served.add((tuple(map(tuple, R.tolist())), t))
    bare = _bare_symmetries_of_the_sign_pattern(PI_FLUX_SX, PI_FLUX_SY)
    assert len(bare) == 8 and served == bare
    assert r.time_reversal["outcome"] == "kept"

    # the control: the same flux in Landau gauge, where the dressing is a boost
    landau = tqg.realized_group(pi_flux_landau_2x1())
    assert landau, str(landau)
    assert (landau.group.number_str, landau.group.size) == ("47.250", 16)
    dl = landau.descent
    assert dl["from"]["number_str"] == "123.340" and dl["steps"][0]["n_dropped"] == 48
    assert dl["steps"][0]["refuted_by"] == {"no symmetry matrix": 32, "tensor structure": 16}
    [tau] = dl["lattice"]["operations"]
    assert tau["refuted_by"] == "tensor structure" and tau["scaled"] == pytest.approx(1.0, abs=1e-12)
    assert tau["residual"] == pytest.approx(4.0, rel=1e-12, abs=0)
